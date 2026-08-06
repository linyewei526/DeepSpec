# DSpark 代码实现详解：从 DFlash 到公开仓库的训练与推理调用链

> 阅读对象：已经读完 DFlash 与 DSpark 论文，希望进一步掌握 DeepSpec 公开代码，尤其准备做推理优化的读者。
>
> 对照版本：仓库提交 `b4abde1e071c17ec5780e4be976a3d664f0e347f`（2026-08-05）。
>
> 论文：`DFlash_Block Diffusion for Flash Speculative Decoding.pdf`、`DSpark_Confidence-Scheduled Speculative Decoding.pdf`。

## 1. 先给出最重要的实现结论

1. **这里的 DFlash 不是一个执行多轮去噪的通用 diffusion LM。**它是专门为 speculative decoding 训练的、只有几层 Transformer 的 masked-block drafter。每轮只做一次非因果 Transformer forward，同时给出整个 draft block 的基础 logits。所谓 diffusion 主要体现在“mask block + 块内双向注意力 + 一步并行预测”的建模形式，而不是推理时反复迭代去噪。
2. **DSpark 的重计算主干仍然完全并行。**Qwen3-8B 配置使用 5 层 draft Transformer，一次 forward 产生 7 个位置的 hidden states 与 base logits；随后低秩 Markov head 只用很轻的串行循环，逐位置加 transition bias 并采样。这就是 semi-autoregressive：重的部分并行，轻的部分串行。
3. **公开仓库的评测推理是原生 PyTorch + Hugging Face Transformers，不是 SGLang/vLLM 等推理引擎。**target 和 draft 都以 BF16 载入，每个可见 GPU 对应一个独立 PyTorch/NCCL 进程，注意力后端是 PyTorch SDPA。没有 continuous batching、PagedAttention、CUDA Graph、请求级调度或 tensor parallel。
4. **论文的生产级调度器没有开源在本仓库。**公开代码包含 confidence head 的训练、静态阈值 early stop、ECE/AUROC/Brier 诊断；不包含 STS 参数拟合/加载、SPS(batch size) profiling、Algorithm 1 的全局 greedy scheduler、两步滞后的异步 top-K、ZOS、变长 query flatten 或 DeepSeek-V4 内核改造。
5. **默认 `--confidence-threshold=0.0` 会关闭截断并验证固定长度 7 的 block。**这正是论文 Table 1 为隔离 raw draft quality 所采用的设置；此时会额外记录 raw confidence 的校准指标。阈值大于 0 时，只是单请求上遇到第一个低于阈值的条件置信度便截断。
6. **当前 `eval.py` 复现的是 speculative decoding 接受长度，不是各 benchmark 的任务准确率。**它不保存文本答案，不执行数学判分、代码测试或 LLM judge，也不测 wall-clock latency/speedup；其主要输出是 accepted length、verify rate 和逐位置 conditional acceptance。

后文所有分析都以这六点为边界：论文中的完整 DSpark 是算法与生产系统的组合，而 DeepSpec 公开仓库主要是 draft 模型训练与离线接受率评测框架。

## 2. 从 DFlash 到 DSpark：论文概念如何落到代码

### 2.1 标准 speculative decoding 的目标

一轮中，draft 分布为 \(p_d\)，target 分布为 \(p_t\)，draft 提议 \(x_1,\ldots,x_\gamma\)。第 \(k\) 个 token 的接受概率是

\[
\min\left(1,\frac{p_t^k(x_k)}{p_d^k(x_k)}\right).
\]

验证必须从左向右看作连续前缀；第一个拒绝位置之后的 draft 全部作废。若在位置 \(k\) 拒绝，则从归一化残差分布 \([p_t^k-p_d^k]_+\) 采样纠正 token；若所有 draft 都接受，则从 target 的最后一个 logits 采样 bonus token。这样最终输出分布严格等于 target 分布。

代码中的对应关系是：

- 接受率与前缀：[`verify_draft_tokens`](../../deepspec/eval/base_evaluator.py#L186)；
- 残差采样：[`sample_residual`](../../deepspec/utils/sampling.py#L34)；
- 完整循环：[`generate_decoding_sample`](../../deepspec/eval/base_evaluator.py#L307)。

论文的单 token 平均延迟为

\[
L=\frac{T_{draft}+T_{verify}}{\tau},
\]

因此 DFlash 主要降低 \(T_{draft}\)，DSpark 的 semi-AR head 提升 \(\tau\)，confidence scheduling 则试图减少无收益的 \(T_{verify}\)。

### 2.2 DFlash 的“一次并行起草”到底做了什么

DFlash 每一轮的逻辑可压缩为：

1. target 对已提交上下文 forward，抽取若干层 hidden states；
2. 将选中层按 hidden 维拼接，用共享投影 `fc` 压回 draft hidden size，再做 RMSNorm；
3. draft 输入由一个已知 anchor token 和若干 mask token 构成；
4. 5 层 draft Transformer 的所有 block 位置做**块内双向注意力**；
5. target features 在每一个 draft layer 中只进入 K/V，draft block 本身产生 Q/K/V；
6. 一次 forward 同时输出所有位置 logits，不执行第二次去噪。

Qwen 实现的核心在 [`Qwen3DSparkAttention.forward`](../../deepspec/modeling/dspark/qwen3/modeling.py#L87)：

```text
Q = Wq(Hdraft)
K = concat(Wk(Htarget_projected), Wk(Hdraft), sequence_dim)
V = concat(Wv(Htarget_projected), Wv(Hdraft), sequence_dim)
```

这里 target feature 绕过 draft 的 Q、attention output 和 MLP，仅作为 persistent contextual K/V。它比“只在第一层把 target feature 与输入 embedding 融合”更不容易随 draft 深度增加而稀释。

训练时，多个随机 anchor block 被打包进同一次 forward，使用 FlexAttention `BlockMask` 隔离 block；推理时每个请求只有当前一个 block，传 `attention_mask=None, is_causal=False`，所有 block 位置互相可见。mask token 不含未来真实 token，因此双向连接不会泄露 ground truth。

### 2.3 DSpark 对 DFlash 的第一项修改：anchor 位置也负责预测

原始 DFlash 叙述是“anchor + \(\gamma\) 个 mask，预测 \(\gamma\) 个 mask 位置”。DSpark 论文和本仓库改成：

```text
输入 7 个位置： [anchor, mask, mask, mask, mask, mask, mask]
输出 7 个 logits：预测 [x1, x2, x3, x4, x5, x6, x7]
```

也就是说，第 0 个输入位置虽然放的是 anchor embedding，但该位置的 output hidden 用来预测 anchor 后的第一个 token。这样产生 7 个候选只需 7 个 query，而不是 8 个。

训练代码证据：

- [`create_noise_embed`](../../deepspec/modeling/dspark/common.py#L264) 先全部填 mask，只在每个 block 的第一个位置写入 anchor；
- [`label_offsets`](../../deepspec/modeling/dspark/qwen3/modeling.py#L432) 是 `1..block_size`，所以第 0 个 output 对齐 `anchor+1`；
- 推理 [`_propose`](../../deepspec/eval/dspark/evaluator.py#L99) 创建长度恰为 `block_size` 的 tensor，并把第 0 位换成当前 anchor。

一个容易忽略的点是：仓库内所谓 DFlash 也复用同一个 `Qwen3DSparkModel/Qwen3DSparkTrainer`，只是配置中关闭 Markov 与 confidence。因此公开仓库的 DFlash baseline 同样使用上述 anchor-first prediction 变体，而不是另有一套原始 DFlash 模型代码。

### 2.4 DSpark 对 DFlash 的第二项修改：轻量串行 transition head

并行 backbone 产生位置无条件于块内采样结果的 base logits \(U_k\)。Vanilla Markov head 再加入

\[
B(x_{k-1},\cdot)=W_1[x_{k-1}]W_2,
\]

最终 draft 分布为

\[
p_d^k(\cdot\mid x_{<k})=\operatorname{softmax}(U_k+B(x_{k-1},\cdot)).
\]

Qwen3-8B 默认 `vocab_size=151936, rank=256`：

- `markov_w1`: `Embedding[V, 256]`；
- `markov_w2`: `Linear[256, V, bias=False]`；
- 每步只做一次 embedding lookup、一次低秩到词表的线性投影和采样。

实现位于 [`VanillaMarkov`](../../deepspec/modeling/dspark/markov_head.py#L8)。[`sample_block_tokens`](../../deepspec/modeling/dspark/markov_head.py#L55) 的 Python loop 明确体现了串行依赖：第一步的前驱是 anchor，以后每一步的前驱是刚采样的 token。

仓库还实现两种非默认 head：

- `gated`：用 `[backbone_hidden; previous_markov_embedding]` 产生 gate，再对 previous embedding 做门控；
- `rnn`：维护 rank 维状态，输入 `[state; previous_embedding; backbone_hidden]`，联合线性层切成 gate/candidate/output 三块，能利用整个块内前缀。

论文主实验与发布的 Qwen3-8B checkpoint 使用 `vanilla`。RNN 只在更长 block 上带来小幅收益，串行状态和部署复杂度更高。

### 2.5 DSpark 对 DFlash 的第三项修改：confidence head

默认 confidence feature 为

```text
concat(backbone_hidden[k], markov_w1[previous_token])
    -> Linear(hidden_size + rank, 1)
    -> raw logit
```

见 [`AcceptRatePredictor`](../../deepspec/modeling/dspark/common.py#L43) 与 [`predict_confidence_step`](../../deepspec/modeling/dspark/qwen3/modeling.py#L292)。sigmoid 后的 \(c_k\) 试图预测：在前面 draft 已接受的条件下，第 \(k\) 个 draft 被 target 接受的概率。

训练软标签不是“argmax 是否相同”，而是解析接受率：

\[
c_k^*=1-\frac{1}{2}\lVert p_d^k-p_t^k\rVert_1.
\]

代码在 [`_compute_accept_rate_3d`](../../deepspec/modeling/dspark/loss.py#L60) 中直接计算。confidence loss 是以这个软标签为 target 的 BCE-with-logits。

要区分三个层次：

1. **模型输出 raw conditional confidence**：已开源；
2. **固定阈值截断**：已开源，只服务离线诊断；
3. **STS + hardware-aware global scheduling**：论文有描述，当前仓库没有实现。

### 2.6 论文中的 STS 与 Hardware-Aware Prefix Scheduler

虽然这两部分没有进入公开代码，但理解它们对于后续推理优化至关重要。

#### STS 为什么校准累计概率

第 j 个 draft 能实际贡献输出的前提是 1 到 j 全部存活，因此 request r 的 prefix survival 是

\[
a_{r,j}=\prod_{i=1}^{j}c_{r,i}.
\]

调度器计算期望 accepted tokens 时使用的是 \(a_{r,j}\) 的绝对数值，而不只是置信度排序。raw neural confidence 即使 AUROC 很好，只要系统性过置信，调度器就会高估长前缀收益。论文的 Sequential Temperature Scaling 在 held-out set 上从左向右逐位置搜索一个标量 temperature：校准第 k 位时，固定已经校准的前 k-1 位，最小化累计乘积的 ECE。temperature scaling 保持单位置 score 排序，但修正绝对概率。

公开 checkpoint 的 `config.json` 没有 STS temperatures，评测代码也没有拟合或应用它们。因此不能把 `confidence_head.py` 画出的 raw reliability diagram 当成生产调度器的校准后输入。

#### Algorithm 1 的吞吐目标

设同时有 R 个 active requests，第 r 个选择验证长度 \(\ell_r\)。送入 target 的 token batch size 与期望产出分别为

\[
B=\sum_{r=1}^{R}(1+\ell_r),
\qquad
\tau=\sum_{r=1}^{R}\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right).
\]

engine 预先 profiling 得到 steps-per-second 曲线 `SPS(B)`，调度器最大化

\[
\Theta=\tau\cdot SPS(B).
\]

由于同一请求的 \(a_{r,j}\) 随 j 单调不增，把所有 `(request, position)` 按 survival probability 全局降序排列，会自然满足前缀约束。沿这条 greedy admission path 每加入一个 token，就更新 B、\(\tau\) 和 \(\Theta\)，记录目前最优的各请求长度。

论文算法在 throughput 首次不再提升时立即 `break`。这不仅利用了平滑 SPS 下目标近似单峰的假设，也保证 non-anticipating：由于 \(c_{r,j+1}\) 会使用已采样 \(x_{r,j}\) 的 Markov embedding，若看完未来 confidence 再回头决定是否 admission 较早 token，就会让较早 token 的接纳事件依赖其 realization，改变 target 输出分布。

#### 论文生产系统的异步改造

真实硬件的 SPS 曲线有离散 cliff，直接在首次下降处停止可能错过后面的更优点；同时 ZOS/CUDA Graph 要在当前 step 完成前知道下一 step 的容量。论文内部系统因此：

1. 用两步前的 confidence/负载估计下一 target step 的总容量 K；
2. 当前 step 仍按当前实际累计 confidence 排序，选择 top-K；
3. 因为 K 只依赖历史，可在历史曲线上做无 early-stop 的全局搜索而不泄漏当前 token；
4. 物理执行上把不同请求的变长 query flatten 成 token 流，用 marker/sparse-attention metadata 表示逻辑归属，避免按最长请求 padding；
5. 在 DeepSeek-V4 中只改造与这种变长 routing 相关的 index-attention/compress kernels。

这套设计是论文高并发吞吐提升的来源，不能由当前 `--confidence-threshold` 近似代替。

## 3. 代码组织与模块职责

```text
DeepSpec/
├── README.md / requirements.txt  # 入口说明与精确 Python 依赖版本
├── NOTICE                    # SpecForge、DFlash 等上游代码/设计来源
├── assets/dspark.drawio       # 论文架构图的可编辑源文件
├── config/
│   ├── dspark/                 # DSpark × Qwen3/Gemma4 的训练配置
│   ├── dflash/                 # 同一并行 backbone，关闭 Markov/confidence
│   └── eagle3/                 # 自回归 drafter 对照组
├── deepspec/
│   ├── data/
│   │   ├── parser.py           # chat template、训练文本 token/loss mask
│   │   ├── jsonl_dataset.py    # JSONL mmap 与行索引
│   │   ├── target_cache_dataset.py # target hidden cache 协议、读写、collator
│   │   └── cuda_prefetcher.py  # 后台 DataLoader + 独立 CUDA stream H2D
│   ├── modeling/
│   │   ├── dspark/
│   │   │   ├── common.py       # anchor 采样、FlexAttention mask、公共输出结构
│   │   │   ├── markov_head.py  # vanilla/gated/RNN sequential heads
│   │   │   ├── loss.py         # CE + L1 distribution matching + confidence BCE
│   │   │   ├── qwen3/          # Qwen3 draft config 与定制 Transformer
│   │   │   └── gemma4/         # Gemma4 对应实现
│   │   └── eagle3/             # Eagle3 baseline
│   ├── trainer/
│   │   ├── base_trainer.py     # 分布式、FSDP、compile、数据循环、checkpoint
│   │   ├── dspark_trainer.py   # DSpark model/loss 接线
│   │   └── ckpt_manager.py     # HF 格式权重 + rank optimizer/RNG state
│   ├── eval/
│   │   ├── base_evaluator.py   # lossless speculative verify 与通用评测循环
│   │   ├── dspark/evaluator.py # DSpark context/propose/update
│   │   ├── dspark/draft_ops.py # 并行 draft、Markov sample、阈值截断
│   │   └── dspark/confidence_head.py # raw confidence 可靠性统计
│   └── utils/                  # 配置、采样、分布式、optimizer、metrics、日志
├── scripts/
│   ├── data/                   # prompt 下载、target 重生成、target cache
│   ├── train/train.sh
│   └── eval/eval.sh
├── eval_datasets/              # 已转换成 {"turns": [...]} 的评测快照
├── train.py                    # 训练入口，自行 spawn GPU 进程
└── eval.py                     # 评测入口，自行 spawn GPU 进程
```

Qwen3 与 Gemma4 共享 `common.py`、Markov head、loss、trainer 和 evaluator 流程，只在模型内部的 attention/RoPE/norm/MLP 与配置字段上分叉。Eagle3 是框架对照，不参与 DSpark 的推理路径。`NOTICE` 说明了边界：Eagle3 及部分 loss/optimizer/attention/eval 框架改编自 SpecForge，DSpark/DFlash 建模受 DFlash 项目设计与训练配方影响；因此阅读公共框架时要与 DSpark 特有的 `modeling/dspark` 和 `eval/dspark` 逻辑区分开。

## 4. Qwen3-8B 配置如何构造 draft 模型

[`config/dspark/dspark_qwen3_8b.py`](../../config/dspark/dspark_qwen3_8b.py) 的关键字段如下：

| 字段 | 值 | 含义 |
|---|---:|---|
| target model | `Qwen/Qwen3-8B` | target/tokenizer/共享权重来源 |
| `block_size` | 7 | 每轮最多提议 7 个 draft token |
| `num_draft_layers` | 5 | 并行 backbone 深度 |
| `target_layer_ids` | `[1,9,17,25,33]` | 抽取并拼接 5 个 target decoder layer 输出 |
| `mask_token_id` | 151669 | Qwen tokenizer 中用作 masked draft input 的 token |
| `num_anchors` | 512 | 每条训练序列每个 epoch 最多随机采样的 block 数 |
| `markov_rank` | 256 | transition head 低秩维度 |
| `markov_head_type` | `vanilla` | 一阶 Markov 版本 |
| confidence | enabled + with Markov | hidden 与前一 token embedding 联合作输入 |
| loss weights | CE 0.1 / L1 0.9 / conf 1.0 | 三项训练目标 |
| loss decay | 4.0 | 代码实际使用 `exp(-position/4)` |
| precision | BF16 | 训练与发布权重 dtype |
| global batch | 512 | 所有 GPU、梯度累积后的 batch |
| epochs | 10 | 公开 DSpark 训练设置 |

[`build_draft_config`](../../deepspec/modeling/dspark/qwen3/config.py#L9) 深拷贝 target 的 `Qwen3Config`，再做以下变换：

- `architectures=["Qwen3DSparkModel"]`，供 `eval.py` 自动选 evaluator；
- `num_hidden_layers=5`，同时记录原 target 层数；
- 所有 draft layer 设为 `full_attention`；
- 训练 attention implementation 设为 `flex_attention`；
- `tie_word_embeddings=False`；
- 写入 block、anchor、mask、Markov、confidence 等自定义字段。

Qwen3-8B draft 主要部件为：

```text
embed_tokens                         # 从 target 拷贝并冻结
fc: Linear(5 * 4096 -> 4096)         # 跨 target 层融合
hidden_norm: RMSNorm
5 × Qwen3DSparkDecoderLayer          # parallel backbone
norm
lm_head: Linear(4096 -> 151936)      # 从 target 拷贝并冻结
markov_w1 + markov_w2                # 可训练
confidence Linear(4096+256 -> 1)     # 可训练
```

`embed_tokens` 和 `lm_head` 在 checkpoint 里是 draft 自己的参数副本，但其初始值来自 target 且训练时被冻结，不是运行时跨两个 model object 共享同一块内存。

## 5. 训练数据与 target cache 调用链

### 5.1 为什么训练时不运行完整 target

DSpark 训练需要两类 target 信号：

1. 选中层 `[1,9,17,25,33]` 的 hidden states，作为 draft K/V context；
2. target 最后一层 hidden，经冻结 LM head 还原 target logits，用于 L1 distribution matching 与 confidence 软标签。

公开默认采用 offline cache。数据流程是：

```text
open-perfectblend prompts
  -> target 非思考模式重新生成 assistant response
  -> tokenizer/chat template + assistant-only loss_mask
  -> target forward + layer hooks
  -> 二进制 target cache
  -> draft-only 多 epoch 训练
```

`scripts/data/prepare_target_cache.py` 用 `AutoModel` BF16 + SDPA 运行 target；forward hooks 捕获所选 decoder layer 的**原始 layer 输出**，另存 `last_hidden_state`。每个样本的 cache 协议保存：

- `input_ids`: int32；
- `attention_mask`: uint8；当前训练 reader 已由 `seq_len` 重建有效区间，未再次返回这份 payload；
- `loss_mask`: uint8；
- `target_hidden_states`: BF16 `[seq_len, 5*hidden_size]`；
- `target_last_hidden_states`: BF16 `[seq_len, hidden_size]`。

cache 由 64GB 左右的顺序 shard、固定尺寸 `samples.idx` 和 `manifest.json` 组成；[`CacheDataset`](../../deepspec/data/target_cache_dataset.py#L615) 通过 mmap 按 offset 读取，避免大量小 tensor 文件。由于对每个 token 保存 6 份 4096 维 BF16 hidden，完整 130 万样本 cache 极大；README 对 4B 默认设置估算约 38TB。这与只做推理复现无关，使用发布 checkpoint 时不需要准备训练 cache。

### 5.2 训练入口到一个 batch

完整调用链：

```text
scripts/train/train.sh
  -> train.py: parse config/--opts
  -> torch.multiprocessing.spawn(nprocs=visible_gpu_count)
  -> Qwen3DSparkTrainer(local_rank, args)
  -> BaseTrainer.__init__
       init NCCL
       build Qwen3DSparkModel
       CPU 加载 target，仅复制 embedding/lm_head，随后删除 target
       torch.compile(dynamic=True)
       FSDP(NO_SHARD)
       CacheDataset + manifest 校验
       BF16Optimizer + cosine/warmup
  -> BaseTrainer.train
       distributed sampler
       DataLoader/CacheCollator
       CUDAPrefetcher
       gradient accumulation
  -> Qwen3DSparkTrainer.run_batch
       model.forward
       compute_dspark_loss
       backward / clip / optimizer / checkpoint
```

当前 Qwen 配置 `sharding_strategy="no_shard"`，所以每张卡持有完整 draft 参数；FSDP 主要提供统一的同步、mixed precision 与 checkpoint 路径。global batch 512、local batch 1、4 GPU 时梯度累积步数为 `512/(4*1)=128`。

### 5.3 一个训练 forward 的张量逻辑

设：

- batch `B`；
- padding 后源序列长度 `L`；
- anchors `A=512`；
- block size `G=7`；
- hidden `D=4096`；
- vocab `V=151936`。

输入为：

```text
input_ids                 [B, L]
loss_mask                 [B, L]
target_hidden_states      [B, L, 5D]
target_last_hidden_states [B, L, D]
```

#### 第一步：随机 anchor

[`build_anchor_candidate_mask`](../../deepspec/modeling/dspark/common.py#L109) 要求 anchor 自身与第一个预测 token 都位于有效 assistant loss 区域。每条序列为所有合法位置生成随机数，排序后取最多 512 个，再按位置排序。若有效位置不足，以 dummy anchor 补齐并由 `block_keep_mask` 屏蔽。

后续位置可能越过序列末尾或 loss 区域。[`build_eval_mask`](../../deepspec/modeling/dspark/common.py#L172) 先逐位置检查，再做 `cumprod`，所以一个 block 只监督从第一个位置起的**连续有效前缀**。

#### 第二步：构造 mask block 与位置

`create_noise_embed` 生成 `[B,A*G,D]`：每块第 0 位是 anchor token embedding，其余是 mask embedding。draft position IDs 是：

```text
anchor_position + [0,1,2,3,4,5,6]
```

完整 RoPE position IDs 是源 context 的 `0..L-1` 再拼所有 draft positions。

#### 第三步：构造 sparse attention

对 query block `q_block`，允许的 K/V 是：

- target context 中 `kv_position < anchor_position` 的位置；注意严格小于 anchor；
- 当前 draft block 的全部 7 个位置，双向可见；
- 不允许看到别的 anchor block；
- dummy block 全部屏蔽。

严格小于 anchor 很重要：第一个 draft token 的 target feature 只能来自 anchor 之前；anchor 自身只以已知 token embedding 出现在当前 block。这和实际推理一致，因为当前 bonus/anchor 刚从 logits 采样，还没有经过 target layer 产生自己的 hidden state。

#### 第四步：并行 backbone

`fc + hidden_norm` 把 `[B,L,5D]` 压到 `[B,L,D]`。每一个 draft layer 都重新把它投成自己的 context K/V，并与当前 draft hidden 的 K/V 拼接；Q 仅来自 draft hidden。backbone 输出 `[B,A*G,D]`，再 reshape 成 `[B,A,G,D]`。

#### 第五步：teacher-forced Markov 与 target 对齐

labels 是 `input_ids[anchor+1 .. anchor+G]`。训练 Markov 前驱为：

```text
[anchor_token, ground_truth_x1, ..., ground_truth_x6]
```

因此训练使用 teacher forcing，而推理使用自身刚采样的 token。base logits 加 Markov bias 后得到 `draft_logits [B,A,G,V]`。

target logits 不在 cache 中直接保存。代码从 `target_last_hidden_states` 收集 `anchor .. anchor+G-1` 的 hidden，再乘冻结 draft LM head；由于 LM head 从 target 原样复制，这就恢复出与 labels 对齐的 target next-token logits，同时将 cache/通信维度从词表量级降到 hidden 量级。

### 5.4 三项 loss 的代码实义

位置权重实际为

\[
w_k=\exp(-(k-1)/4),\quad k=1,\ldots,7,
\]

其中分母来自配置 `loss_decay_gamma=4.0`，不要把变量名与 block size \(\gamma=7\) 淆混。

1. `ce_loss`：draft 对 ground-truth token 的交叉熵；
2. `l1_loss`：softmax 后 draft 与 target 全词表概率的 L1 距离；
3. `confidence_loss`：raw confidence logit 对解析接受率软标签的 BCE。

Qwen3-8B 总 loss：

\[
0.1L_{CE}+0.9L_{L1}+1.0L_{conf}.
\]

loss 的 numerator 在本 rank 计算，denominator 跨 rank all-reduce；最终乘 `world_size` 抵消 FSDP/DDP 对梯度的平均，从而得到正确的全局 token 加权目标。训练指标还记录逐位置解析 accept rate、概率期望 \(\tau=1+\sum_j\prod_{i\le j}c_i^*\)、confidence bias 与 cumulative-product bias。

### 5.5 checkpoint

每 3000 optimizer step 保存：

```text
step_N/
├── config.json
├── model.safetensors
├── train_config.py
└── training_state.rank<R>.pt   # optimizer + RNG + next_micro_step
```

rank 0 从 FSDP 汇总 full state dict，用 `save_pretrained` 保存成标准 HF 目录；`step_latest` 是指向最近 checkpoint 的软链接。发布到 Hugging Face 的 checkpoint 只需要 `config.json + model.safetensors` 即可评测，不需要 optimizer state。

## 6. 推理/评测的完整调用链

### 6.1 入口、进程与模型加载

```text
scripts/eval/eval.sh
  -> eval.py
       AutoConfig(draft).architectures[0]
       -> Qwen3DSparkEvaluator
       torch.multiprocessing.spawn(visible_gpu_count)
  -> BaseEvaluator.__init__
       init_dist(NCCL)
       target = AutoModelForCausalLM(... BF16, SDPA).cuda().eval()
       draft  = Qwen3DSparkModel.from_pretrained(... BF16, SDPA).cuda().eval()
       tokenizer = AutoTokenizer(target)
  -> Qwen3DSparkEvaluator.evaluate
       for each hard-coded dataset
```

每个进程都完整复制 target 与 draft；GPU 数量只用于**样本级 data parallel**，不会把单条请求分到多卡。`local_rank=r` 的进程处理 `r, r+world_size, ...` 的样本。

推理没有 `torch.compile`。`@torch.inference_mode()` 禁用 autograd，Qwen attention 走 Transformers 的 SDPA dispatch；实际底层可能根据 shape/dtype/device 选择 PyTorch fused SDPA kernel，但这不等价于独立 serving engine。

### 6.2 数据预处理

`eval.py` 固定九项任务和最大样本数。`load_and_process_dataset` 从当前工作目录下 `./eval_datasets/<name>.jsonl` 读取 `{"turns": [...]}`，并无条件截成 `turns[:1]`。所以 MT-Bench 的第二轮问题不会执行。

每个样本被包装为唯一 user message，然后：

```python
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

这保证与论文的 Qwen3 non-thinking 设置一致。评测没有 prompt truncation，也没有 top-p/top-k/min-p；temperature=1.0 意味着完整 softmax 分布采样。

### 6.3 prefill 与首个 anchor

[`generate_decoding_sample`](../../deepspec/eval/base_evaluator.py#L307) 先创建 target `DynamicCache`，执行一次标准 target prefill：

```text
target(input prompt,
       use_cache=True,
       output_hidden_states=True,
       logits_to_keep=1)
```

从最后 logits 采样第一个输出 token。这个 token 被写到 `output_ids[prompt_len]`，成为第一轮 anchor。target KV cache 此时只含 prompt；anchor 尚未经过 target forward。

DSpark context 初始化为：

- 空的 draft `DynamicCache`；
- 从 target prefill `hidden_states` 抽取层 `[1,9,17,25,33]` 并拼接。

Transformers 的 `hidden_states` 约定中，`hidden_states[0]` 是 embedding output，decoder layer `i` 的输出在 `hidden_states[i+1]`；[`extract_context_feature`](../../deepspec/modeling/dspark/common.py#L52) 正是用 `layer_id+1` 索引。

### 6.4 一轮 propose：并行 backbone + 串行 Markov

假设循环开始时 `start` 指向当前已提交 anchor：

1. 创建长度 7 的 `[anchor, mask×6]`；
2. draft position IDs 从 `draft_cache.seq_len` 切到 `start+7`；
3. `_forward_backbone` 接收“自上一轮以来新提交 token 的 target features”和当前 mask block；
4. `is_causal=False`，当前 block 内双向 attention；
5. `use_cache=True` 将 context K/V 与 speculative block K/V 暂时写入 draft cache；
6. 立刻 `past_key_values_draft.crop(start)`，丢弃当前 speculative block，只保留已提交 context 的 K/V；
7. LM head 一次性得到 7 个 base logits；
8. Markov loop 从左到右对每个 logits 加 transition bias并采样；
9. confidence head 用同一 backbone hidden 和实际前驱 token 产生 7 个 confidence logits。

注意 draft cache 的含义。第一轮 draft forward 时，它把 prefill target features 写入每个 draft layer 的 K/V cache；crop 后只留下长度 `prompt_len` 的 context。target 验证后，下一轮只把上一轮新提交路径的 target features 追加进去。这样不必每轮对完整 context 重做 `fc/K/V projection`。

### 6.5 静态 confidence 截断

[`_confident_prefix_length`](../../deepspec/eval/dspark/draft_ops.py#L82) 的规则是：

```text
threshold <= 0: 保留全部 7 个
否则：保留第一个 sigmoid(confidence_logit) < threshold 之前的前缀
```

它比较的是每位置**条件概率** \(c_k\)，不是累计前缀概率 \(\prod_{i\le k}c_i\)。当前 evaluator 强制 batch size 1，因此只读取 `confidence_logits[0]`。若第一个位置即低于阈值，proposal 长度为 0，下一步退化成 target 对 anchor 生成一个 token。

保留前缀后，代码用 Markov 修正后的 logits 计算完整 `draft_probs`。这点非常关键：rejection sampling 的 \(p_d\) 必须是实际用于采样 token 的分布，而不能使用未加 transition bias 的 base logits。

### 6.6 target 并行验证与无损纠正

proposal 组织为：

```text
verify_input_ids = [current_anchor, draft_1, ..., draft_l]
```

target 一次 forward 得到 `l+1` 个 logits：

- anchor 位置 logits 检验 `draft_1`；
- `draft_k` 位置 logits 检验 `draft_{k+1}`；
- 最后一个位置 logits 用于全接受时的 bonus。

对每个 draft，代码收集该 token 在 target/draft 分布中的概率，采样 Bernoulli 接受 mask，再 `cumprod` 得到连续接受前缀。

- 若第 `k` 个 draft 被拒绝：接受前 `k-1` 个，从 `[p_t-p_d]_+` 采样 correction；
- 若全部接受：追加一个 target bonus；
- 若 proposal 长度为 0：target 直接从 anchor logits 采一个 token。

每一正常轮提交 `accepted_draft_tokens + 1` 个 token，论文与代码报告的 acceptance length \(\tau\) 都包含这个 `+1` bonus。

target forward 曾把完整 proposal 写进 target KV cache。实际只提交前缀与 correction/bonus，所以循环在更新 `start` 后执行 `past_key_values_target.crop(start)`，删除未接受的 speculative KV。新采样的 correction/bonus 本身尚无 target hidden/KV，将在下一轮作为 anchor 进入验证。

### 6.7 target feature 如何推进

验证输出包含 `[anchor, draft_1, ..., draft_l]` 每个输入位置的 target hidden。`_update` 只保留：

```text
hidden[:, :accepted_draft_tokens + 1]
```

即旧 anchor 与真正接受的 draft hidden。下一轮 draft forward 把这一小段追加进 draft K/V cache；新 correction/bonus 仍以当前 block 的 anchor embedding 出现。这个“target hidden 只缓存已 target-forward 的 committed token，新 anchor 用 token embedding补上”的不变量，是理解两个 KV cache 对齐关系的关键。

### 6.8 EOS 与长度边界

- prefill 后首 token 若是 stop token，直接结束，不发生 verify；
- 已接受 draft 中出现 EOS，只提交到第一个 EOS，并裁剪 target cache；不会再追加 bonus；
- correction/bonus 中出现 EOS，在本轮提交后退出；
- block 可能让内部 `start` 越过 `max_length`，最终输出会切到 `prompt_len + max_new_tokens`。

## 7. 指标到底在测什么

每轮记录：

- `proposal_length`：实际送去验证的 draft 数；固定长度模式通常为 7，EOS 附近可能缩短；
- `accepted_draft_length`：真正接受的 draft 数，不含 bonus；
- `acceptance_length`：正常轮为 `accepted_draft + 1`，含 bonus。

聚合后：

\[
\text{accept_len}=\frac{\sum \text{acceptance_length}}{\#\text{proposal rounds}},
\]

\[
\text{verify_rate}=\frac{\sum \text{committed tokens}}{\sum \text{draft proposal length}+\#\text{rounds}}.
\]

逐位置 `accept_rate@k` 的分母是“第 k 位实际被提议的轮数”，分子是“接受前缀至少覆盖第 k 位的轮数”，所以它是**累计 prefix survival**，不是论文 Figure 2 中“以前面位置已经接受为条件”的单步 conditional acceptance。在固定 7 长度时各位置分母基本一致；若要由固定长度结果近似论文的 conditional acceptance，第 0 位直接使用该值，第 k 位可用相邻两个 prefix survival 之比。阈值截断后还要额外考虑 admission selection，不能直接这样相除。

当 threshold 恰为 0 且模型有 confidence head 时，`ConfidenceHeadRecorder` 还会：

1. 对条件 confidence 做 cumulative product，得到预测 prefix survival；
2. 用实际 rejection sampling 的 `accept_prefix_mask` 作 0/1 label；
3. 以 20 个粗 bin 算 ECE，以 1000 个细 bin 近似 AUROC，并算 Brier；
4. 可选写 TensorBoard、`metrics.json` 和 reliability diagram。

这里没有实现 STS 校准，只是在诊断 raw confidence 是否过置信。

## 8. 论文完整系统与公开代码的逐项边界

| 能力 | DSpark 论文 | 当前 DeepSpec 仓库 |
|---|---|---|
| DFlash 式 parallel backbone | 有 | 有 |
| anchor 位置直接预测第一 token | 有 | 有 |
| Vanilla Markov / RNN head | 有 | 都有，发布 checkpoint 用 vanilla |
| confidence end-to-end 训练 | 有 | 有 |
| soft TV acceptance label | 有 | 有 |
| 固定 confidence threshold sweep | 离线分析 | 有 |
| raw confidence ECE/AUC/Brier | 有 | 有 |
| Sequential Temperature Scaling | 有 | **无拟合、无参数存储、无推理应用** |
| SPS(batch size) engine profiling | 有 | **无** |
| Algorithm 1 全局 prefix scheduler | 有 | **无** |
| 因果 early-stop greedy search | 有 | **无，仅单请求阈值 stop** |
| 两步历史预测的异步 top-K | 生产系统 | **无** |
| CUDA Graph / ZOS | 生产系统 | **无** |
| variable-query flatten + marker sparse attention | 生产系统 | **无** |
| continuous batching / serving API | 生产系统 | **无** |
| accepted length 离线评测 | 有 | 有 |
| latency、吞吐、TPS/用户 | 有 | **无计时与压测实现** |
| benchmark correctness/judge | 论文任务用于 rollout | **仓库不判分** |

论文 Algorithm 1 的 early stop 不是普通性能技巧，而是无损性的条件。Markov confidence 的后一个分数依赖已经采样出的前一 token；若 scheduler 看完未来分数后再回头决定是否接纳更早 token，就会让接纳事件依赖 token 自身，产生 selection bias。论文生产系统能移除当步 early stop，是因为只用两步前历史信息决定总容量 K，历史形成 causal barrier；这部分逻辑不在公开 evaluator 中。

## 9. 当前推理实现的性能画像与优化切入点

### 9.1 已经启用的东西

- BF16 target/draft 权重；
- `torch.inference_mode()`；
- Hugging Face `DynamicCache`；
- PyTorch SDPA attention dispatch；
- draft target-feature K/V 的增量 cache；
- 一次并行 block backbone；
- 多 GPU 样本级 data parallel。

### 9.2 明确没有启用的东西

- eval 阶段 `torch.compile`；
- FlashAttention 包或显式 flash-attn backend；
- SGLang/vLLM/TensorRT-LLM；
- CUDA Graph；
- target/draft tensor parallel；
- 多请求 batch/continuous batching；
- paged KV cache；
- fused Markov sampling/confidence kernel；
- wall-clock profiler、NVTX 标记或吞吐统计。

`requirements.txt` 中的 Triton 并不说明此推理路径手写了 Triton kernel；仓库没有 DSpark Triton kernel。SGLang 只用于可选的**训练回答重生成**脚本，不用于 `eval.py`。

### 9.3 热路径上的明显优化机会

1. **Python 级串行 Markov loop。**7 次 `Linear(256,V) + softmax/multinomial` 都从 Python 发射 kernel；rank 很小但词表很大。可以研究 CUDA Graph、compile、Triton/fused projection-sampling，或一次算候选子词表，但必须保持实际 draft 分布及 rejection sampling 一致。
2. **draft softmax 实际重复计算。**`sample_block_tokens` 在每一步的 `sample_tokens` 中为采样做一次全词表 softmax；随后 `build_dspark_proposal` 又对保存下来的 corrected logits 做 `logits_to_probs`，为 rejection sampling 重新 softmax 一次。target verify 也会对全部 target logits softmax。拒绝时残差采样确实需要完整分布，而接受判定本身只需 proposed token 概率；可以考虑复用第一次概率、延迟 materialize residual 或 fused softmax/accept/residual，不能简单只保留 top-k，否则改变 target 分布。
3. **单请求、无 batch。**`generate_decoding_sample` 强制 `bsz=1`，九个任务逐样本 Python loop。多卡只复制模型分摊样本，完全没利用请求级 batching；这是吞吐优化最大空间，也是实现论文 scheduler 的前提。
4. **target 每轮输出所有层 hidden state tuple。**实际只需要 5 层；`output_hidden_states=True` 会保留/返回全部 36 层输出。可以用 hooks、定制 model forward 或选择性 capture 减少内存带宽与 Python object 开销。
5. **draft target feature 拼接。**每轮 `extract_context_feature` 创建 5D concat，再经 `fc`。可将投影与 layer capture 融合，或让 target 直接产出压缩 feature；要保持 cache 对齐。
6. **两套 DynamicCache 的 crop。**HF 通用 cache 便于正确性验证但未针对 speculative rollback/variable-length batch 优化；engine 化时应使用支持 per-request commit/rollback 的 paged cache 元数据。
7. **没有 compile/CUDA Graph。**shape 在固定 block 模式较稳定，很适合先对 draft backbone + Markov loop做局部 capture；target query 长度随 confidence/接受情况变化，需要 bucket 或 padding/flatten 设计。
8. **没有时间测量。**开始优化前应分别量 `target prefill`、`draft parallel`、`Markov+confidence`、`target verify`、cache update/crop 与 Python orchestration，避免只优化非瓶颈。

### 9.4 做 engine 化时必须守住的正确性不变量

- draft probability 必须对应**实际采样**的 Markov-corrected logits；
- target 与 draft 必须使用同一 temperature 语义；
- 第一个拒绝之后不再提交 draft；
- rejection token 必须来自 `[p_t-p_d]_+` 的归一化残差；
- scheduler 对 token k 的 admission 不可依赖尚不应可见的 token realization；
- target cache 只能 commit 接受前缀，draft context cache 只能追加已 target-forward 的 committed hidden；
- EOS 可能位于已接受 draft，而不是只出现在 bonus；
- 多请求变长 batch 不能用 padding token 的隐藏状态污染 K/V 或指标。

## 10. 容易踩坑的实现细节

1. **`block_size=7` 是 7 个 proposal，不是“anchor+6 proposal”。**输入 tensor 长 7，但 anchor 位置也输出第一个 proposal。
2. **DFlash 配置不是另一模型类。**它仍显示 `architectures=["Qwen3DSparkModel"]`，通过 `markov_rank=0, confidence_alpha=0, CE-only` 退化为并行 baseline。
3. **不能把 target final decoder layer 放进 `target_layer_ids`。**cache hook 保存 raw decoder layer output，而 Transformers `output_hidden_states` 最终位置可能是 final normalized hidden，两条路径语义不一致；evaluator 会显式 assert。
4. **threshold 0 的语义是“不截断”，不是“置信度至少为 0”。**同时它开启 calibration recorder；threshold 非 0 则 recorder 完全关闭。
5. **默认 temperature=1，不使用 Qwen generation_config 中的 top-p/top-k。**若改为 0，代码退化为 one-hot greedy 分布，接受逻辑仍能运行，但不再复现论文 Table 1。
6. **只读每条数据的第一 turn。**MT-Bench 在仓库中有两轮，实际仅第一轮。
7. **输出答案被保存在函数返回 tensor 中但不会落盘。**dataset loop 最后只聚合 acceptance metrics。
8. **没有 task CLI。**任务列表和样本上限硬编码在 `eval.py::TASKS`。
9. **多 GPU 不减少单样本 latency。**它只是数据并行，提高整个离线评测的完成速度。
10. **工作目录影响数据路径。**`DEFAULT_DATASET_ROOT="./eval_datasets"`，从别的目录启动必须提供对应相对目录/软链接；当前代码没有 `--dataset-root`。
11. **发布 draft checkpoint 不自带 tokenizer。**evaluator 始终从 target path 加载 tokenizer，这是正确用法。
12. **显存估算要按每卡完整副本。**Qwen3-8B BF16 约 16.4GB，DSpark draft 权重约 4.74GB，还要加两套 KV cache、logits/probability 与临时 activation。
13. **`mask_token_id=151669` 不要求 tokenizer 能把普通文本编码成该 token。**它由 evaluator 直接写入 tensor；该 ID 位于模型的 `vocab_size=151936` 内。当前 Qwen3 tokenizer 的常规 token ID 只到 151668 左右，因此不要用 `len(tokenizer)==config.vocab_size` 作为兼容性断言。

## 11. 推荐的代码阅读顺序

若希望最快形成可修改的心智模型，建议依次看：

1. `config/dspark/dspark_qwen3_8b.py`：固定所有维度与 feature/loss 开关；
2. `deepspec/eval/base_evaluator.py::generate_decoding_sample`：先掌握 token/cache 状态机；
3. `verify_draft_tokens` 与 `utils/sampling.py`：确认无损 rejection sampling；
4. `deepspec/eval/dspark/evaluator.py`：理解 DSpark 如何嵌进通用循环；
5. `deepspec/eval/dspark/draft_ops.py`：propose、Markov、confidence 截断；
6. `deepspec/modeling/dspark/qwen3/modeling.py`：KV injection 与 backbone；
7. `deepspec/modeling/dspark/markov_head.py`：semi-AR 串行部分；
8. `deepspec/modeling/dspark/common.py`：训练 anchor packing 与 sparse mask；
9. `deepspec/modeling/dspark/loss.py`：三项 loss 与解析接受率；
10. `trainer/base_trainer.py`、target cache：最后再理解训练工程。

读完后可以把一轮推理记成一句话：

> target 已提交 hidden 增量进入 draft K/V cache；5 层非因果 draft backbone 一次产生 7 个 base logits；低秩 Markov head 逐 token 修正并采样；可选 confidence 截前缀；target 一次并行验证；按接受前缀提交并回滚两套 cache；继续以 correction/bonus 作为新 anchor。
