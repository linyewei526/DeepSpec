# DSpark + Qwen3-8B 推理复现指南

> 目标：从空环境和空模型目录开始，使用 DeepSpec 发布的 `dspark_qwen3_8b_block7` checkpoint，在 GSM8K、MATH-500、AIME25、HumanEval、MBPP、LiveCodeBench、MT-Bench、Alpaca、Arena-Hard-v2 上复现论文 Table 1 的 speculative decoding accepted length。
>
> 仓库版本：`b4abde1e071c17ec5780e4be976a3d664f0e347f`。
>
> 本机核查日期：2026-08-05，Asia/Shanghai。

## 1. 先明确“复现”产物

本指南复现的是论文的**离线 draft quality 评测**：

- Qwen3-8B target；
- DSpark 5-layer、block size 7、Vanilla Markov head；
- Qwen3 non-thinking；
- temperature 1.0；
- confidence 截断/调度关闭，即固定验证全部 7 个 draft；confidence head 本身仍会输出并记录预测；
- 每轮 accepted length \(\tau\) 包含 target bonus token。

它不会自动产出以下结果：

- GSM8K/MATH/AIME 的答题准确率；
- HumanEval/MBPP/LiveCodeBench 的 pass@1；
- MT-Bench/Arena-Hard 的 judge score；
- 自回归 baseline、端到端 speedup、tokens/s、吞吐或延迟。

原因是当前 `eval.py` 只聚合 speculative acceptance 指标，不保存生成文本，也没有接 benchmark scorer。若后续要测真正的任务质量或 speedup，需要扩展 evaluator；这不属于论文 Table 1 的当前复现路径。

### 1.1 不要混淆 Markov 词表概率与 confidence 接收概率

DSpark 在每个 draft 位置保留两类用途不同的输出：

1. parallel backbone 先产生完整词表的 base logits $U_k$；Markov head 根据前一个 token 加上低秩 bias，得到修正后 logits $\tilde{U}_k=U_k+B(x_{k-1},\cdot)$。经 softmax 后的 $p_d^k$ 用于采样 draft token，也会作为 `draft_probs` 参与 lossless rejection sampling。
2. confidence head 输出的是每个位置一个标量 raw logit $z_k$，`sigmoid(z_k)` 才是预测的 conditional acceptance confidence。默认 checkpoint 会把 backbone hidden 与前一个 token 的 `markov_w1` embedding 拼接后送入线性层；它不输出词表分布，也不替换 Markov 修正后 logits。

训练时的 confidence 软标签是 Markov 修正后 draft 分布与 target 分布的重叠质量：

\[
c_k^*=1-\frac12\lVert p_d^k-p_t^k\rVert_1
=\sum_v\min(p_d^k(v),p_t^k(v)).
\]

因此，严格说它预测的是“给定前缀后，该位置从 draft 分布采样的 token 的**期望接收概率**”，而不是对已采样具体 token $x_k$ 直接计算的 $\min(1,p_t(x_k)/p_d(x_k))$。代码也印证了这一点：第 $k$ 位 confidence 的 Markov feature 是前一 token 的 embedding，不包含当前已采样 token 本身。推理中的真实接收/拒绝仍由 target 验证和 Markov 修正后的 `draft_probs` 精确决定。当前 evaluator 会先采样完整 block、再计算全部 confidence 并选取前缀；`--confidence-threshold=0.0` 时仍会计算和记录 confidence，但不用它截断 draft；阈值大于 0 时才在第一个 `sigmoid(z_k)<threshold` 的位置前停止 proposal。

Qwen3-8B 的论文目标值如下：

| 域 | 数据集 | DSpark accepted length（Table 1） |
|---|---|---:|
| Math | GSM8K | 6.17 |
| Math | MATH-500 | 5.78 |
| Math | AIME25 | 5.01 |
| Code | MBPP | 5.16 |
| Code | HumanEval | 5.52 |
| Code | LiveCodeBench | 5.17 |
| Chat | MT-Bench | 3.72 |
| Chat | Alpaca | 3.58 |
| Chat | Arena-Hard-v2 | 3.21 |

这是随机采样评测。固定代码、数据、模型 revision 和 seed 后应非常接近；SDPA kernel、GPU 和库细节仍可能造成最后小数位波动。

## 2. 已核查的机器条件

当前机器：

- Ubuntu 24.04.4；
- 4 × NVIDIA A100 80GB PCIe；
- NVIDIA driver 590.44.01，`nvidia-smi` 显示 CUDA 13.1 runtime compatibility；
- Conda 26.5.3；
- 四张 GPU 在核查时都有不同程度占用，不能直接假定空闲。

官方仓库中 target 约 16.4GB，draft 约 4.74GB。本指南按照用户指定的默认存放方式操作，不再设置额外的 Hugging Face cache 路径，也不再围绕磁盘空间设计迁移方案。运行前只需确认 GPU 状态：

```bash
nvidia-smi
```

## 3. 固定使用的绝对路径

后续所有命令都直接写绝对路径，不定义任何路径别名：

| 用途 | 绝对路径 |
|---|---|
| DeepSpec 仓库 | `/data/home/wly/dLLM/DeepSpec` |
| Conda 环境 | `/data/home/wly/.conda/envs/dspark` |
| Qwen3-8B target | `/data1/linyewei/models/Qwen3-8B` |
| DSpark draft | `/data1/linyewei/models/dspark_qwen3_8b_block7` |
| 仓库已提供的评测快照 | `/data/home/wly/dLLM/DeepSpec/eval_datasets` |
| 需要额外下载时的数据根目录 | `/data1/linyewei/datasets/DSpark` |
| 可提交的复现辅助脚本 | `/data/home/wly/dLLM/DeepSpec/runtime` |
| evaluator 实际工作目录 | `/data/home/wly/dLLM/DeepSpec` |
| 评测结果根目录 | `/data/home/wly/dLLM/DeepSpec-results` |

创建需要的目录：

```bash
mkdir -p /data1/linyewei/models /data1/linyewei/datasets/DSpark /data/home/wly/dLLM/DeepSpec/runtime /data/home/wly/dLLM/DeepSpec-results/qwen3_8b
```

`runtime` 不是 DSpark 模型的运行时依赖，也不是 KV cache、模型 cache 或临时数据目录。它只用来放本指南中的 `verify_models.py`、统一实验入口 `run_experiment.py` 和只读汇总工具 `summarize_results.py`。这些脚本能固化复现步骤，有价值跟仓库一起提交；下载的数据、模型权重、日志和 TensorBoard artifact 不应放进该目录。

evaluator 默认读取当前工作目录下的 `./eval_datasets`。由于仓库本身已提交了所需的精确数据快照，正式评测直接在 `/data/home/wly/dLLM/DeepSpec` 下运行，不再从 `runtime` 目录运行，也不再建立 `runtime/eval_datasets` 软链接。

确认仓库 commit。若输出不是本指南版本，先记录差异，不要盲目期待逐小数位一致：

```bash
git -C /data/home/wly/dLLM/DeepSpec rev-parse HEAD
git -C /data/home/wly/dLLM/DeepSpec status --short
```

本仓库当前已有用户修改/未跟踪论文文件；复现不需要清理 worktree，也不要执行 `git reset --hard`。

## 4. 创建 Conda 环境 `dspark`

### 4.1 创建与激活

推荐 Python 3.12。不要沿用 base 环境的 Python 3.14，以减少二进制 wheel 兼容风险。

```bash
conda create -n dspark python=3.12 pip -y
conda activate dspark
python --version
which python
```

使用 `-n dspark` 而不是 `-p <path>`；按当前 Conda 配置，该命名环境会自动安装到 `/data/home/wly/.conda/envs/dspark`，`which python` 应输出 `/data/home/wly/.conda/envs/dspark/bin/python`。

### 4.2 安装 CUDA PyTorch 与项目依赖

仓库固定 `torch==2.9.1`。本机 driver 能运行 CUDA 12.8 wheel，推荐先从 PyTorch 官方 index 安装该 wheel：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r /data/home/wly/dLLM/DeepSpec/requirements.txt
python -m pip check
```

`requirements.txt` 的关键版本是：

```text
torch==2.9.1
transformers==5.10.2
triton==3.5.1
numpy==2.4.4
safetensors==0.7.0
datasets==4.8.5
```

不需要为了 `eval.py` 安装 SGLang、vLLM、flash-attn、evalplus 或代码沙箱。SGLang 在此仓库只用于可选的训练答案重生成，不参与发布 checkpoint 的离线评测。

### 4.3 环境自检

```bash
python -c 'import torch, transformers, triton; from torch.nn.attention.flex_attention import create_block_mask; print("torch:", torch.__version__); print("torch CUDA wheel:", torch.version.cuda); print("transformers:", transformers.__version__); print("triton:", triton.__version__); print("cuda available:", torch.cuda.is_available()); print("gpu count:", torch.cuda.device_count()); [print(i, torch.cuda.get_device_properties(i).name, round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1), "GiB") for i in range(torch.cuda.device_count())]'
```

预期核心项：PyTorch 2.9.1、Transformers 5.10.2、CUDA available 为 true，看到 A100。`torch.version.cuda` 显示 wheel 自带的 CUDA 版本，不必与 `nvidia-smi` 顶部的 13.1 完全相同；关键是 driver 向后兼容且 CUDA 可用。

再做项目 import 检查：

```bash
PYTHONPATH=/data/home/wly/dLLM/DeepSpec python -c 'from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel; from deepspec.eval.dspark import Qwen3DSparkEvaluator; from deepspec.utils.sampling import sample_residual; print("DeepSpec imports OK")'
```

## 5. 下载并固定两个模型 revision

### 5.1 为什么要固定 revision

截至 2026-08-05，核查到：

```text
Qwen/Qwen3-8B
revision b968826d9c46dd6066d109eabc6255188de91218

deepseek-ai/dspark_qwen3_8b_block7
revision 03326e5043815da1f81b109078b2889737c26017
```

发布 draft 的 `config.json` 明确包含：

```text
architectures = ["Qwen3DSparkModel"]
block_size = 7
num_hidden_layers = 5
target_layer_ids = [1, 9, 17, 25, 33]
markov_head_type = "vanilla"
markov_rank = 256
enable_confidence_head = true
transformers_version = "5.10.2"
```

固定 revision 能防止远端后续更新悄悄改变 tokenizer/config/权重。

### 5.2 先 dry-run

`transformers` 会安装 `huggingface_hub` 及 `hf` CLI。先看计划下载大小：

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /data1/linyewei/models/Qwen3-8B --dry-run
hf download deepseek-ai/dspark_qwen3_8b_block7 --revision 03326e5043815da1f81b109078b2889737c26017 --local-dir /data1/linyewei/models/dspark_qwen3_8b_block7 --dry-run
hf download Qwen/Qwen3-8B --local-dir /data1/linyewei/models/Qwen3-8B --dry-run
hf download deepseek-ai/dspark_qwen3_8b_block7 --local-dir /data1/linyewei/models/dspark_qwen3_8b_block7 --dry-run
```

若 `hf` 不存在，确认激活了 `dspark`，再执行：

```bash
python -m pip install huggingface_hub
python -m pip check
```

不要为了同一个模型再运行 `git lfs clone`，否则可能产生第二套大对象。

### 5.3 正式下载

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /data1/linyewei/models/Qwen3-8B
hf download deepseek-ai/dspark_qwen3_8b_block7 --revision 03326e5043815da1f81b109078b2889737c26017 --local-dir /data1/linyewei/models/dspark_qwen3_8b_block7
```

`--local-dir` 会直接维护最终目录，并在目录内创建很小的 `.cache/huggingface` 元数据以支持断点续传。网络中断后重复同一命令即可，不要删除半成品再从头开始。

### 5.4 下载后结构与大小检查

```bash
du -sh /data1/linyewei/models/Qwen3-8B /data1/linyewei/models/dspark_qwen3_8b_block7
find /data1/linyewei/models/Qwen3-8B -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
find /data1/linyewei/models/dspark_qwen3_8b_block7 -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
```

target 应包含 5 个 model safetensors shard、index、config、generation config 和 tokenizer 文件；draft 应至少包含 `config.json` 与一个约 4.74GB 的 `model.safetensors`。

### 5.5 完全离线的 config/checkpoint 检查

将下面的 Python 代码保存为 `/data/home/wly/dLLM/DeepSpec/runtime/verify_models.py`；这是脚本内容，不是需要拆分执行的多行 Shell 命令。

```python
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

target = Path("/data1/linyewei/models/Qwen3-8B")
draft = Path("/data1/linyewei/models/dspark_qwen3_8b_block7")

tcfg = AutoConfig.from_pretrained(target, local_files_only=True)
dcfg = AutoConfig.from_pretrained(draft, local_files_only=True)
tok = AutoTokenizer.from_pretrained(target, local_files_only=True)

assert tcfg.model_type == "qwen3"
assert tcfg.hidden_size == 4096
assert tcfg.num_hidden_layers == 36
assert dcfg.architectures[0] == "Qwen3DSparkModel"
assert dcfg.block_size == 7
assert dcfg.num_hidden_layers == 5
assert dcfg.target_layer_ids == [1, 9, 17, 25, 33]
assert dcfg.markov_rank == 256
assert dcfg.markov_head_type == "vanilla"
assert dcfg.enable_confidence_head is True
assert dcfg.vocab_size == tcfg.vocab_size == 151936
assert 0 <= dcfg.mask_token_id < dcfg.vocab_size

with safe_open(draft / "model.safetensors", framework="pt", device="cpu") as handle:
    keys = set(handle.keys())

required = {
    "embed_tokens.weight",
    "lm_head.weight",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
}
assert required <= keys, required - keys

print("tokenizer length:", len(tok))
print("target/draft config and safetensors keys OK")
```

执行命令只有一行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /data/home/wly/dLLM/DeepSpec/runtime/verify_models.py
```

这一步只读 safetensors header，不会把 21GB 权重全部加载进 CPU/GPU。不要断言 `len(tokenizer)==config.vocab_size`：Qwen3 模型词表矩阵保留了 tokenizer 普通 token 之外的 ID，DSpark 的 `mask_token_id=151669` 正是 evaluator 直接写入 tensor 的模型内有效 ID。

## 6. 准备九个评测集

### 6.1 不直接复用 NLD JSONL 的结论

已经检查 `/data1/linyewei/datasets/NLD`：

| 数据 | NLD 状态 | 能否直接作为 DeepSpec `eval_datasets/*.jsonl` |
|---|---|---|
| GSM8K | 1319 条，字段 `problem/expected_answer/...` | 不能；DeepSpec 要 `turns` 且加固定 reasoning suffix |
| MATH-500 | 500 条，字段 `problem/...` | 不能直接；schema/prompt suffix 不同 |
| AIME25 | 30 条 | 不能直接；schema、空格与 prompt suffix 不同 |
| HumanEval | 164 条，完整测试元数据 | 不能直接；DeepSpec 有专用 user prompt wrapper |
| MBPP | 378 条 | 不能；DeepSpec 使用 sanitized 257 条并随机取 256 |
| LiveCodeBench | NLD 是 CPP v5/v6，279/454 条 | 不能；DeepSpec 是 1055 条 code_generation_lite/Python prompt 口径 |
| MT-Bench/Alpaca/Arena-Hard-v2 | NLD 未发现同口径文件 | 不能 |

因此，NLD 同名数据可以作为题目来源交叉核对，却不应复制后直接替换。仓库已经提交了论文评测所需、转换完成的九份 JSONL，总计不到 10MB；这九项评测无需再下载或复制数据。

### 6.2 直接使用仓库中的精确评测快照

```bash
ls -lh /data/home/wly/dLLM/DeepSpec/eval_datasets/*.jsonl
```

当前支持的九个 benchmark 都已在该目录，因此 `/data1/linyewei/datasets/DSpark` 下不需要再建一层 `eval_datasets`，也不需要为本次复现重复复制文件。以后若增加仓库未携带的原始数据集，下载文件和它的转换产物直接放在 `/data1/linyewei/datasets/DSpark` 下；不要创建 `/data1/linyewei/datasets/DSpark/eval_datasets`。新数据要接入当前 evaluator，仍需要转成 `{"turns": [...]}` JSONL，并显式改造 dataset-root 参数或将最终的小型评测快照提交到仓库 `eval_datasets`。

### 6.3 行数、评测抽样数与 SHA256

| 文件 | 文件总行数 | `eval.py` 实际最大样本数 | SHA256 |
|---|---:|---:|---|
| gsm8k.jsonl | 1319 | 500 | `63330a20a17f416fdfca978ebcd5124f8da5546affb3a079b65a6e8daf42b41f` |
| math500.jsonl | 500 | 500 | `4f530d9d3126dca41c12a96e78ec0dd460b7202052ba86101c4da159cea33aac` |
| aime25.jsonl | 30 | 30 | `b8835996839caa1d982c5bb9ecaeca636424f0b8ed9f33a40b084feb1c1766d0` |
| humaneval.jsonl | 164 | 164 | `7aaed6a3987007ecee4851ee4572fde214aa2a36ae83fa21be9c5f443aa71675` |
| mbpp.jsonl | 257 | 256 | `019b981f7122e39ce58866cf93f4206511c72532cb537bbaf298753a0f128ba7` |
| livecodebench.jsonl | 1055 | 500 | `40ed536492f331ac27b3b68506072366ca1d3a304b92c31d822574e778c87318` |
| mt-bench.jsonl | 80 | 80 | `df05defbcea350dff39f7d2996a7cba8a029e6e06069dfcdab11ceb998d743b0` |
| alpaca.jsonl | 52002 | 500 | `c1f1623ac08e4c4f024604bb2689a024963eb7eb9dbbc29e57dda5a11d87e07b` |
| arena-hard-v2.jsonl | 750 | 500 | `e5fcce94ffb1a2ef2082b5c63ac36608f2b7faf00da13ecb94ce2adc649b7b91` |

核查命令：

```bash
wc -l /data/home/wly/dLLM/DeepSpec/eval_datasets/*.jsonl
sha256sum /data/home/wly/dLLM/DeepSpec/eval_datasets/*.jsonl
```

`eval.py` 对大于上限的数据先用 `random.Random(980406).shuffle`，再取前 N 条；不要自己预先截前 N 行，否则样本集合不同。

### 6.4 evaluator 为什么要从仓库根目录运行

[`base_evaluator.py`](../../deepspec/eval/base_evaluator.py#L28) 把数据根目录默认写为 `./eval_datasets`，所以“工作目录”只决定这个相对路径从哪里解析，它不承载模型状态。直接从仓库根目录运行后，`./eval_datasets` 自然就是 `/data/home/wly/dLLM/DeepSpec/eval_datasets`，不需要任何软链接：

```bash
cd /data/home/wly/dLLM/DeepSpec && pwd && ls -l ./eval_datasets/gsm8k.jsonl
```

`/data/home/wly/dLLM/DeepSpec/runtime` 中的辅助脚本可以用绝对路径调用，但 evaluator 的当前工作目录仍保持为仓库根目录。

## 7. 正式运行前的单卡 smoke test

### 7.1 先找真正空闲的 GPU

```bash
nvidia-smi
```

选择一个显存和 GPU-Util 都足够空闲的物理卡。下面命令以物理 GPU 1 为例，实际执行时直接把 `CUDA_VISIBLE_DEVICES=1` 中的 `1` 改成空闲卡编号。评测进程会在每张可见卡上完整加载约 21.2GB target+draft 权重，建议至少预留 30GB 显存。

### 7.2 统一实验入口与结果目录规则

本指南的 smoke、全量和单数据集实验都通过 [`runtime/run_experiment.py`](../../runtime/run_experiment.py) 启动。它不修改 DSpark 推理算法，只负责：

- 把 `all` 或单个数据集名转换为 evaluator 的 `tasks`；
- 调用与原始 `eval.py` 相同的 `torch.multiprocessing.spawn` 和 `eval.main`；
- 把 TensorBoard 和 evaluator artifact 指向本次实验自己的目录；
- 在加载模型前就创建完整结果目录，写入 `experiment_manifest.json`，并创建空的 `dataset_results.jsonl`、TensorBoard/artifact 和 progress 目录；
- 运行时由 rank 0 显示跨所有 GPU 聚合的单条 tqdm，每成功完成一个样本就推进该数据集的全局计数；
- 每完成一个数据集立即向 `dataset_results.jsonl` 追加结果，把 manifest 中该项从 `pending`/`running` 更新为 `completed`，然后继续下一项；
- 整次实验完成或报错后更新总状态、结束时间、耗时和异常；
- 记录数据集及上限、数据文件 SHA256、两个模型路径、所有生成/置信度超参、GPU/分布式设置、Python/PyTorch/Transformers/CUDA 版本、Git commit/worktree 状态和实际 Python 命令。

每条实验命令都会在 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b` 下创建一个全新的：

```text
YYYYMMDD_HHMMSS_任务名
```

例如 `20260806_143205_all`、`20260806_155012_gsm8k`、`20260806_160833_smoke_gsm8k`。命令中的 `RUN_DIR` 只是为了在**同一条 shell 命令内部**把刚生成的动态时间戳路径同时交给 Python 和 `tee`；模型、数据、仓库、结果根目录仍全部使用绝对路径，没有引入 `DS_REPO` 一类持久路径别名。`mkdir "$RUN_DIR"` 刻意不加 `-p`：同名目录已存在时立即停止，避免两次实验混写。

### 7.3 两条样本、64 token smoke test

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_smoke_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29609 python /data/home/wly/dLLM/DeepSpec/runtime/run_experiment.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 64 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --max-samples 2 2>&1 | tee "$RUN_DIR/eval.log"
```

这里 `gsm8k` 选择数据集，`--max-samples 2` 把正常的 500 条上限临时覆盖为 2，`--max-new-tokens 64` 把每条样本的最大生成长度临时改为 64；其他参数保持正式实验口径。只暴露一张 GPU，所以 launcher 只 spawn 一个 worker，distributed world size 为 1。结果仍然完整写入独立目录，而不是只打印到终端。

成功标准：命令开始后立即出现时间戳目录、初始 manifest 和空的 `dataset_results.jsonl`；target 和 draft 成功载入 BF16；没有 architecture/config/key mismatch；终端出现总数为 2 的 GSM8K tqdm；完成后打印 spec 行和 confidence reliability 表；`dataset_results.jsonl` 有一行，manifest 最终为 `"status": "completed"`；生成 `eval.log`、TensorBoard event、`metrics.json` 与 `reliability_diagram.png`。64 token、2 条样本仅验证链路，不能与论文 2048 token、500 条样本的值比较。

## 8. 一次运行全部九项 benchmark

### 8.1 正式复现参数

论文 Table 1 对应：

```text
max_new_tokens       = 2048
temperature          = 1.0
confidence_threshold = 0.0
seed                 = 980406（当前公开 eval.py 默认）
step                 = 0（只作为 TensorBoard/产物目录的横坐标和层级）
dist_backend         = gloo（仅用 CPU 归约评测计数；GPU 仍执行模型推理）
dist_timeout_minutes = 1440（允许不同 rank 因样本长度不均等待最多 24 小时）
enable_thinking      = False（evaluator 代码固定）
```

`confidence_threshold=0.0` 会保留全部 7 个 draft，隔离 DSpark draft quality；confidence head 仍执行且被记录，但不据此提前截断 proposal。改成大于 0 得到的是 static threshold 实验，不再是 Table 1 口径。

### 8.2 推荐的多卡全量命令（示例使用物理 GPU 2、3）

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_all && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29610 python /data/home/wly/dLLM/DeepSpec/runtime/run_experiment.py all --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 2>&1 | tee "$RUN_DIR/eval.log"
```

这是一条物理单行命令；各部分含义如下：  

| 命令片段 | 作用 |
|---|---|
| `set -o pipefail` | 让 Python 失败时整个 `python \| tee` pipeline 返回失败，而不是被成功退出的 `tee` 掩盖 |
| `mkdir -p .../qwen3_8b` | 只保证固定结果根目录存在 |
| `RUN_DIR=.../$(date +%Y%m%d_%H%M%S)_all` | 生成本次全量实验唯一目录名；时间使用当前机器本地时区 |
| `mkdir "$RUN_DIR"` | 新建目录；若碰撞则停止，绝不覆盖旧实验 |
| `cd /data/home/wly/dLLM/DeepSpec` | 使 evaluator 的相对路径 `./eval_datasets` 正确解析到仓库数据快照 |
| `env -u RANK -u WORLD_SIZE` | 清除外部调度器残留；本项目把它们解释为节点 rank/节点数 |
| `CUDA_VISIBLE_DEVICES=2,3` | 暴露物理 GPU 2、3；每卡放一份完整 target+draft，样本按 distributed rank 分片 |
| `PYTHONPATH=.../DeepSpec` | 让 `runtime` 脚本能 import 仓库中的 `eval` 与 `deepspec` |
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` | 强制只读取已经下载到绝对路径的本地模型 |
| `MASTER_ADDR`、`MASTER_PORT` | 建立本机 distributed control/metrics process group；并行实验必须使用不同空闲端口 |
| `run_experiment.py all` | `all` 明确选择固定顺序的全部九个数据集；不要再套 `torchrun` |
| `--run-dir "$RUN_DIR"` | 将本次 manifest、TensorBoard 和 artifact 全部绑定到唯一目录 |
| `--target`、`--draft` | 指定 Qwen3-8B target 和 DSpark block-7 checkpoint |
| `--max-new-tokens` | 每条样本最多生成 2048 个新 token；EOS 可使其提前结束 |
| `--temperature` | target/draft 分布都按温度 1.0 采样；不是 greedy decoding |
| `--confidence-threshold` | 0.0 表示不基于 confidence 截断，同时启用 confidence 指标记录器 |
| `--seed` | 控制数据 shuffle 和每条样本的随机采样种子 |
| `--step` | 写入 `step_0` artifact 层级，并作为 TensorBoard scalar step；不是 checkpoint step |
| `--dist-backend gloo` | 用 CPU Gloo 归约每项结束后的少量 spec 计数和 confidence 直方图；模型推理仍在各张 GPU 上独立执行 |
| `--dist-timeout-minutes` | distributed collective 超时；长生成静态分片可能导致快 rank 等慢 rank 超过一小时，因此正式评测显式设为 1440 分钟 |
| `2>&1 \| tee "$RUN_DIR/eval.log"` | 合并 stdout/stderr，既显示到终端又完整保存到本次目录 |

全量 `all` 展开为 `gsm8k(500)、math500(500)、aime25(30)、humaneval(164)、mbpp(256)、livecodebench(500)、mt-bench(80)、alpaca(500)、arena-hard-v2(500)`，合计最多 3030 条。launcher 只在开始时加载一次模型，然后按上述顺序逐数据集运行。原生 Transformers/PyTorch、batch size 1、最多 2048 token 是长任务，建议在 `tmux` 中运行。

### 8.3 GPU 数与并行语义

若只有物理 GPU 2 空闲，只把示例命令中的 `CUDA_VISIBLE_DEVICES=2,3` 改为 `CUDA_VISIBLE_DEVICES=2`。若实际空闲的是两张不连续卡，可写 `CUDA_VISIBLE_DEVICES=1,3`；进程内部看到的是逻辑 `cuda:0,1`。减少 GPU 数主要增加 wall-clock：它不会做 tensor parallel，而是每卡完整复制 target+draft，再以 `samples[rank::world_size]` 分数据。多进程并发加载还会占用更多主机内存。

不要用 `torchrun`。`run_experiment.py` 已调用 `torch.multiprocessing.spawn(..., nprocs=torch.cuda.device_count())`。同机另开实验时必须同时换一个 `MASTER_PORT`；时间戳结果目录解决的是文件隔离，不能解决 distributed 端口冲突。

评测路径不使用 DDP、tensor parallel 或任何推理期跨卡通信：每张卡独立处理自己的样本，只有一项结束后才归约少量整数计数和 confidence histogram。这里默认用 Gloo 是为了避免活跃 GPU 上的小型 NCCL collective 卡死；Gloo 只搬运 CPU 指标，不会把 target/draft 模型或 logits 移到 CPU，也不改变采样结果。

### 8.4 运行中会打印什么、如何监控

命令中的 `mkdir "$RUN_DIR"` 首先建立时间戳目录；随后 launcher 在**模型加载前**写初始 `experiment_manifest.json`，创建空的 `dataset_results.jsonl`、`progress/` 和 `tensorboard/artifacts/step_0/`。因此即使权重加载或第一项评测很慢，也能立刻看到本次目录和完整配置，不需要等九项结束。

模型加载完成后，rank 0 会打印数据集名、有效样本总数和一条聚合所有 rank 的 tqdm，例如：

```text
Starting dataset: gsm8k
Dataset gsm8k: 500 samples across 2 rank(s)
gsm8k samples:  24%|██████▍                    | 121/500 [18:42<58:31, 9.26s/sample, ranks=2/2]
```

每个 worker 只更新 `/progress/gsm8k/rank_<rank>.json` 的已完成计数；rank 0 的后台线程每秒读取这些小文件并刷新唯一一条全局进度条。这里没有每样本 NCCL collective，不会为显示进度而强制各张 GPU 每条样本同步。`121/500` 表示所有可见卡合计已成功完成 121 条；速度和 ETA 是基于已完成样本的近似值，由于不同 prompt/输出长度差异很大，ETA 会明显波动。

进度条 `500/500` 只表示该数据集的**样本生成阶段**完成，还要依次归约 spec 指标、归约 confidence 指标并写结果。rank 0 会明确打印阶段切换，并同步更新 manifest 的 `datasets[].phase`：

```text
Dataset gsm8k: reducing_spec_metrics
Dataset gsm8k: reducing_confidence_metrics
Dataset gsm8k: writing_artifacts
Dataset gsm8k: writing_incremental_result
```

这些后处理正常只应持续很短时间。如果某行之后长时间无输出，便能直接定位阻塞阶段。后处理完成后 rank 0 执行以下动作，然后才进入下一项：

1. 打印该数据集的一行无 header speculative decoding 结果；
2. threshold=0 时打印 compact confidence JSON，写该项 `metrics.json` 和 reliability 图片；
3. 无论 threshold 是否为 0，都向顶层 `dataset_results.jsonl` 追加一行 spec/confidence summary 并执行 `fsync`；
4. 原子更新 manifest 中该数据集的 `status/completed_at/result` 和总 `completed_dataset_count`。

九项全部完成后，再打印带 header 的完整 speculative decoding 表和完整 confidence reliability 表，并写 TensorBoard scalar。由于 `2>&1 | tee`，tqdm、逐项结果、最终表和错误都同时进入该次实验的 `eval.log`；日志中的 tqdm 会包含回车刷新字符，直接在运行终端观看最清晰。

先列出实验目录并找到所需时间戳：

```bash
find /data/home/wly/dLLM/DeepSpec-results/qwen3_8b -mindepth 1 -maxdepth 1 -type d -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

假设本次目录是 `20260806_143205_all`，监控命令为：

```bash
tail -f /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/eval.log
nvidia-smi
```

还可以在另一个终端查看已经完成的数据集结果，文件会逐项增长：

```bash
tail -f /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/dataset_results.jsonl
```

查看当前 GSM8K 的机器可读聚合进度：

```bash
watch -n 2 python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/progress/gsm8k/progress.json
```

`experiment_manifest.json` 的顶层在运行中为 `running`；每个 `datasets[]` 条目的 `status` 独立经历 `pending → running → completed`，`phase` 则进一步区分 `sampling/reducing_spec_metrics/reducing_confidence_metrics/writing_artifacts/writing_incremental_result/completed`。正常结束后顶层变为 `completed`，Python 可捕获的异常会标为 `failed` 并写入类型、消息和 traceback。因此，不要只凭 tqdm 到达 100% 或目录存在就判断整项完成。

## 9. 如何选择单个数据集或全部九个数据集

### 9.1 选择规则

选择由 `run_experiment.py` 后的第一个位置参数完成：

| 位置参数 | 实际运行内容 | 默认最大样本数 |
|---|---|---:|
| `all` | 顺序运行全部九项 | 3030 条合计 |
| `gsm8k` | 仅 GSM8K | 500 |
| `math500` | 仅 MATH-500 | 500 |
| `aime25` | 仅 AIME25 | 30 |
| `humaneval` | 仅 HumanEval | 164 |
| `mbpp` | 仅 MBPP | 256 |
| `livecodebench` | 仅 LiveCodeBench | 500 |
| `mt-bench` | 仅 MT-Bench | 80 |
| `alpaca` | 仅 Alpaca | 500 |
| `arena-hard-v2` | 仅 Arena-Hard-v2 | 500 |

不支持逗号列表，例如不能写 `gsm8k,math500`。要么用 `all`，要么一次只运行表中的一个名字。`--max-samples N` 是可选的样本上限覆盖，主要用于 smoke/debug；正式复现不要添加它。对 `all` 使用该参数会把**每个**数据集的上限都限制为 N，而不是只限制第一个数据集。

### 9.2 单独运行 GSM8K 的四卡命令

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29611 python /data/home/wly/dLLM/DeepSpec/runtime/run_experiment.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 2>&1 | tee "$RUN_DIR/eval.log"
```

要改跑 `math500`，同时把时间戳目录末尾的 `_gsm8k` 改成 `_math500`，并把 `run_experiment.py gsm8k` 改成 `run_experiment.py math500`；其他八项同理。目录后缀只是方便人阅读，真正决定数据集的是 Python 的位置参数，manifest 会记录实际选择，所以二者应保持一致。

单数据集和全量的所有其余参数语义完全相同，输出目录层级也完全相同。区别只是 `tasks` 中有一项，结束时最终表也只有一行。单项实验每次都会重新加载约 21GB 模型，九项分开跑的启动开销明显大于一次 `all`，但某项失败时只需为该项新建时间戳目录重跑。

### 9.3 中断与重跑

当前 evaluator 会记录样本进度，但仍没有 sample-level resume，也不会从旧 `progress.json` 或 `metrics.json` 继续。全量任务在第六项中断时，旧目录会保留前五项 artifact 和 `dataset_results.jsonl` 的前五行，manifest 中前五项为 `completed`、当前项为 `failed`、后续项为 `pending`；重新执行全量命令会创建新时间戳目录并从 GSM8K 开始。如果不想重算已完成项，后续分别运行剩余单项，但它们属于多个独立实验目录，汇报时必须明确这一点。

不要把新的运行重新指向旧目录。launcher 检测到已有 `experiment_manifest.json` 或 `tensorboard` 会拒绝复用，防止日志、event 和指标互相污染。

## 10. 结果目录、说明文件、汇总与论文对齐

### 10.1 每次实验的完整目录结构

以全量目录 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all` 为例，成功后结构为：

```text
20260806_143205_all/
├── experiment_manifest.json
├── dataset_results.jsonl
├── eval.log
├── progress/
│   ├── gsm8k/
│   │   ├── progress.json
│   │   ├── rank_0.json
│   │   └── rank_1.json
│   └── ...其余数据集...
└── tensorboard/
    ├── events.out.tfevents.*
    └── artifacts/
        └── step_0/
            ├── gsm8k/
            │   ├── metrics.json
            │   └── reliability_diagram.png
            ├── math500/
            │   ├── metrics.json
            │   └── reliability_diagram.png
            └── ...其余七项...
```

因此，同一次实验的终端日志、逐项追加结果、进度快照、结构化指标、图片、TensorBoard event 和实验说明都在一个时间戳目录内；`qwen3_8b` 根目录不再直接存放某次实验的 `eval_all.log` 或共享 `tensorboard`。单数据集目录只有对应的一个 progress/artifact 子目录。

### 10.2 `experiment_manifest.json` 记录什么

这是本次实验的机器可读说明文件，至少包括：

- 顶层 `status/start_time/last_update_time/end_time/elapsed_seconds/run_dir/completed_dataset_count`；
- `mode` 以及 `datasets`：实际数据集名、文件绝对路径、配置/有效样本上限、文件总行数、SHA256，以及每项的 `status/phase/started_at/completed_at/result`；
- `models`：target 与 draft 的本地绝对路径；
- `hyperparameters`：`max_new_tokens`、`temperature`、`confidence_threshold`、`seed`、`step`、`max_samples_override`、`dist_timeout_minutes`，以及代码固定的 non-thinking、block size 7、batch size 1、SDPA；
- `distributed`：可见 GPU 编号/数量/型号、master 地址和端口、Gloo/NCCL backend、数据并行语义；
- `environment`：工作目录、主机名、Python 可执行文件和版本、torch/transformers/CUDA 版本、offline 开关；
- `repository`：仓库绝对路径、Git commit 和运行时 worktree 状态；
- `invocation/outputs/error`：Python 命令、manifest、增量 JSONL、progress、日志、TensorBoard/artifact 的所有位置，以及失败时的异常详情。

直接查看某次说明文件：

```bash
python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/experiment_manifest.json
```

shell 前缀（例如 `CUDA_VISIBLE_DEVICES`、offline 和 NCCL 参数）没有全部出现在 `invocation` 字符串里，但已经拆分记录在 `distributed` 和 `environment` 中；完整原始单行命令仍应与本指南或实验记录一起保留。

### 10.3 evaluator 主表和 `metrics.json`

主表字段：

| 输出列 | 含义 |
|---|---|
| `#propose` | 每轮平均实际 draft token 数，表格再显示 `+1` target bonus；固定 block 模式通常接近 `7.00+1` |
| `accept_len` | 每轮平均最终提交 token 数，已经包含 bonus；这是与论文 Table 1 的 \(\tau\) 对比的核心量 |
| `verify_rate` | 最终提交 token 数总和 / target 一次验证覆盖的 token 数总和，即 `acceptance_length_sum / (proposal_length_sum + proposal_count)` |
| `accept_rate@k` | 第 k 个 draft 位置被验证时的接受率；由于 speculative decoding 只验证连续前缀，它体现 prefix survival，k 从 0 开始 |

每个 `metrics.json` 的核心层级为：

```text
config.args/tasks
spec.dataset/num_samples/draft_tokens_per_proposal/acceptance_length/verify_rate
spec.accept_rates_by_position
confidence.per_position
confidence_summary.ece_mean/auc_mean/brier_mean/pred_mean/target_mean
```

检查某次全量实验是否有九个指标文件：

```bash
find /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/tensorboard/artifacts/step_0 -mindepth 2 -maxdepth 2 -name metrics.json -print | sort
```

应输出九条路径。这里只能说明九项都写出了 artifact；最终是否正常完成仍以 manifest 的 `status=completed` 和命令退出码为准。

### 10.4 只看最终汇总，不重放逐样本输出

[`runtime/summarize_results.py`](../../runtime/summarize_results.py) 是只读工具，输入**某一次时间戳实验目录**，优先读取逐项增长的 `dataset_results.jsonl`，旧实验没有该文件时才回退读取各数据集的 `metrics.json`。它只打印当前实验状态、`completed_datasets` 和已经完成数据集的紧凑表，不重新执行模型，也不会逐样本打印；因此实验尚未全部结束时也可重复运行：

```bash
python /data/home/wly/dLLM/DeepSpec/runtime/summarize_results.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all --step 0
```

输出列为 `dataset、samples、propose、accept_len、verify、ECE、AUC、Brier`。其中前三个 speculative 指标来自 `spec`，后三个置信度校准指标来自 `confidence_summary`。`--step 0` 必须与运行时的 `--step 0` 一致；若正式运行改为 `--step 3`，汇总也要写 `--step 3`。

要查看 TensorBoard：

```bash
tensorboard --logdir /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_143205_all/tensorboard --port 6006
```

### 10.5 与论文对齐的 sanity checks

1. manifest 中 target/draft 应分别为本地 `Qwen3-8B` 和 `dspark_qwen3_8b_block7`，九个数据 SHA256 与第 6.3 节一致；
2. threshold 0 下 `#propose` 应接近 `7.00+1`；
3. `accept_len` 必须在 1 到 8 之间，并与第 1 节 Table 1 目标比较，而不是把 `verify_rate` 与论文的 \(\tau\) 比；
4. 数学/代码的 accepted length 一般显著高于 chat；
5. 九个 artifact 目录都存在，manifest 为 `completed`；
6. confidence 的 ECE 可以不为 0，当前公开路径没有应用论文 STS；
7. smoke 或 `--max-samples` 调试结果不能与正式全量口径混在同一张复现表中。

## 11. 若结果对不上，按此顺序排查

### 11.1 数据/采样设置

先检查：

- 九个 SHA256 是否一致；
- seed 是否 980406；
- temperature 是否 1.0；
- threshold 是否 0.0；
- max new tokens 是否 2048；
- 是否由代码固定 `enable_thinking=False`；
- 是否误用 NLD raw JSONL；
- 是否自行截取了前 N 行，而不是让 `eval.py` seeded shuffle。

### 11.2 模型/代码

```bash
git -C /data/home/wly/dLLM/DeepSpec rev-parse HEAD
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

再核查本地 draft `config.json` 的 architecture、block、Markov、target layer IDs。target 与 draft 必须属于同一个 Qwen3-8B target；不能将 Qwen3-4B/14B checkpoint 混用。

### 11.3 输出语义

不要混淆：

- accepted draft tokens：不含 bonus；
- `accept_len`：含 bonus；
- `accept_rate@k`：累计 prefix survival 的逐位置比例，不是单步 conditional accuracy；
- confidence predicted prefix survival：raw conditional scores 的累计乘积；
- benchmark accuracy：当前根本没有算。

### 11.4 数值非确定性

代码会设 Python/NumPy/Torch/CUDA seed，但没有强制 `torch.use_deterministic_algorithms(True)`。不同 SDPA kernel/GPU 或库实现可能在浮点边界改变极少数采样。先比较两位小数及宏观排序，不要把最后几个采样差异误判为模型错误。

## 12. 常见故障

### 12.1 CUDA OOM

症状：模型加载或 target verify 时 OOM。

- 每卡是完整模型，减少 GPU 数不会降低单卡模型内存；
- 用 `nvidia-smi` 清查别人的进程，不要杀不属于你的任务；
- 先单卡 smoke；
- 评测 batch 已是 1，主要余量来自模型、KV cache 和其他占用；
- 若异常长 prompt 导致 OOM，需要先定位具体 dataset/sample，随意降低 `max_new_tokens` 会改变复现口径。

### 12.2 Distributed 初始化、端口或指标归约卡住

- 不要用 `torchrun`；
- `unset RANK WORLD_SIZE`；
- 设一个不同的 `MASTER_PORT`；
- 确认 `CUDA_VISIBLE_DEVICES` 至少一张卡；
- 多个评测 job不能共享同一个端口；
- 正式评测保留 `--dist-backend gloo`，不要无理由改回 NCCL。

当前按 `samples[rank::world_size]` 静态分片，不同样本的实际输出长度差异很大，快 rank 可能先完成，慢 rank 仍在生成。旧实现使用 GPU NCCL 归约末尾的几个计数；在 GPU 上还有其他活跃进程时，曾出现两个 rank 都完成样本、NCCL kernel 却持续自旋的情况。当前评测默认改用 CPU Gloo 归约，tqdm 进度也只通过文件聚合，不依赖 GPU collective。`--dist-timeout-minutes 1440` 仍用于允许快慢 rank 长时间不均衡；它只延长等待上限，不改变采样、accepted length 或数据集结果。

### 12.3 找不到 dataset

错误通常是 `./eval_datasets/<name>.jsonl` 不存在。确认：

```bash
cd /data/home/wly/dLLM/DeepSpec && pwd
readlink -f /data/home/wly/dLLM/DeepSpec/eval_datasets
```

工作目录应为 `/data/home/wly/dLLM/DeepSpec`，数据目录应解析为 `/data/home/wly/dLLM/DeepSpec/eval_datasets`，不需要软链接。

### 12.4 `KeyError: Qwen3DSparkModel` 或权重 key mismatch

通常是：

- `PYTHONPATH` 没指向 `/data/home/wly/dLLM/DeepSpec`；
- Transformers 版本不对；
- 下载了 DFlash/Eagle3 或不同尺寸 checkpoint；
- draft 下载不完整。

重新执行第 5.5 节的离线验证。

### 12.5 在线请求仍然发生

正式评测命令已在同一行中前置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，并且 target/draft 都传入绝对本地路径。如果仍发生网络请求，先确认执行的是本指南的完整命令，而不是把路径换回 `Qwen/Qwen3-8B` repo ID。

### 12.6 结果日志有表但没有 `metrics.json`

使用本指南的 `run_experiment.py` 时，`tensorboard_dir` 会自动绑定为 `<本次时间戳目录>/tensorboard`，因此正式 threshold=0 实验只需确认命令中保留：

```text
--step <integer>
--confidence-threshold 0.0
```

再检查 manifest 的 `status/error` 和 `eval.log` 中的 `Wrote dataset metrics to ...`。threshold 非 0 会关闭当前代码的 `ConfidenceHeadRecorder`，因而不会创建 `metrics.json` 和 reliability plot；但每项的主 spec 仍会立即追加到 `dataset_results.jsonl`、写入 manifest，并在全部结束时进入 TensorBoard scalar。这是当前实现行为，不是结果目录丢失。

### 12.7 运行非常慢

这是当前公开实现的预期特征：原生 HF/PyTorch、batch size 1、每样本 Python generation loop，不是论文生产 serving engine。先确认 GPU 利用率与进程数正常，再决定是否开展 batch/engine 化；不要安装 SGLang 后期待 `eval.py` 自动切换后端。

## 13. 可选：静态 confidence threshold 实验

完成 Table 1 复现后，可以改变 `--confidence-threshold`。例如下面是 threshold=0.4 的独立全量实验；它仍会创建自己的时间戳目录、manifest、日志和 TensorBoard：

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_all_thr0p4 && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29612 python /data/home/wly/dLLM/DeepSpec/runtime/run_experiment.py all --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.4 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 2>&1 | tee "$RUN_DIR/eval.log"
```

做 0.2/0.4/0.6/0.8 sweep 时，每次分别修改 threshold、目录后缀（如 `_all_thr0p2`）和 `MASTER_PORT`，执行四条独立命令；不要把四次运行写进同一目录。注意当前 recorder 只在 threshold 恰为 0 时启用，所以 threshold>0 的目录中预期有 manifest、逐项追加的 `dataset_results.jsonl`、进度快照、`eval.log` 和 TensorBoard spec scalar，但没有 `metrics.json`/confidence reliability plot；比较静态阈值实验时优先读取各自的 `dataset_results.jsonl`。

这会在第一个 `sigmoid(z_k) < threshold` 的位置之前截断 proposal，其中 $z_k$ 是 confidence head 的 raw logit。预期阈值越高：

- 平均 `#propose` 下降；
- verify rate 上升；
- accepted length 可能下降；
- chat 截断通常比 math/code 更强。

这只是论文 Figure 5 风格的 static threshold sweep，不是 Hardware-Aware Prefix Scheduler。当前代码既不读取 engine load，也不优化 SPS(batch size)，且 batch size 固定为 1。

## 14. 做后续推理优化时的建议基线

在改代码前完整保存：

```text
repo commit/diff
Python、torch、transformers、CUDA/driver 版本
target/draft revision
dataset SHA256
命令行与环境变量
九项 accepted length/verify rate/position acceptance
整次实验 wall-clock（manifest 的 `elapsed_seconds`）
GPU 型号与可见卡数
```

建议先增加不改变语义的 profiling：

1. target prefill；
2. parallel draft backbone；
3. 7-step Markov sampling；
4. confidence head/prefix selection；
5. target verify；
6. target/draft cache crop/update；
7. Python orchestration 与 CPU-GPU synchronization。

任何 engine 化结果都要同时检查 accepted length 与输出分布正确性。尤其不能用未加 Markov bias 的 base logits 做 draft probability，也不能把普通 top-k sampling/greedy 结果与 temperature=1 的论文数值直接比较。

## 15. 外部权威参考

- PyTorch 2.9.1 官方历史版本安装命令：<https://pytorch.org/get-started/previous-versions/>
- Hugging Face `hf download --local-dir`：<https://huggingface.co/docs/huggingface_hub/en/guides/download>
- Hugging Face cache 环境变量：<https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables>
- Qwen3-8B 模型：<https://huggingface.co/Qwen/Qwen3-8B>
- DSpark Qwen3-8B block-7 checkpoint：<https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7>

## 16. 实验结果深度解读：bonus token、acceptance 与 confidence calibration

本章以当前代码和已完成的 `20260806_115804_all` 实验为准。最需要先纠正的结论是：

1. `#propose` 中的 `+1` **不是上一轮的 bonus token**；
2. 上一轮产生的 target token 到了本轮已经是确定提交的 current/anchor token，它是本轮的条件，不再算一次 `accept_len`；
3. 在不是 EOS 终止的普通验证轮中，`accept_len = 本轮连续接收的 draft token 数 + 1 个本轮 target correction/bonus token`；
4. 因此普通轮的已接收 draft 数是 `accept_len - 1`，**不是 `accept_len - 2`**；
5. `accept_rate@k` 只统计第 `k` 个从 0 开始编号的 draft 位置是否被连续前缀接收，不把拒绝后的 target correction 当作该 draft 被接收。

### 16.1 一轮 block-7 DSpark 究竟输入和产生了什么

记本轮开始时已经提交的当前 token 为 $a$，draft block size 为 $\gamma=7$。$a$ 的来源可能是：

- 第一轮时，target prefill 对 prompt 采样得到的第一个输出 token；
- 之后某轮时，上一轮在第一个拒绝位置生成的 target residual correction token；
- 如果上一轮的所有 draft 都被接收，则是上一轮额外采样的真正 bonus token。

所以，把每轮的 $a$ 都叫作“上一轮 bonus”并不严谨；它更准确的名字是“已提交的 current/anchor token”。

`Qwen3DSparkEvaluator._propose()` 构造的 draft backbone 输入是：

```text
[a, MASK, MASK, MASK, MASK, MASK, MASK]    # 长度 7
```

这一步确实是“1 个 current token + 6 个 mask”，但它不是只预测 6 个 token。draft backbone 为 7 个位置都产生 hidden state 和 base vocabulary logits，Markov head 再依次用前一个 token 做因果修正和采样：

```text
a  -> 修正第 1 个位置 logits -> d1
d1 -> 修正第 2 个位置 logits -> d2
...
d6 -> 修正第 7 个位置 logits -> d7
```

因此 draft model 实际提议了 7 个新 token $d_1,\ldots,d_7$。随后送入 target model 的 `verify_input_ids` 是：

```text
[a, d1, d2, d3, d4, d5, d6, d7]           # 长度 8
```

这里 target 确实执行了长度 8 的一次 causal forward，但它不是再“验证 $a$是否正确”。Causal LM 的 8 组输出分布依次表示：

| target 输入位置 | 该位置 logits 的用途 |
|---|---|
| $a$ | 验证 $d_1$ |
| $d_1$ | 验证 $d_2$ |
| $d_2$ | 验证 $d_3$ |
| $\ldots$ | $\ldots$ |
| $d_6$ | 验证 $d_7$ |
| $d_7$ | 当 7 个 draft 全部接收时，采样额外的 target bonus token |

如果在 $d_4$ 发生第一次拒绝，$d_4$ 之后的 target logits 虽然已在这次并行 forward 中算出，但不再用于提交这些 draft；代码会在第一个拒绝处采样 correction，并 crop target cache。

### 16.2 `#propose` 为什么是 `6.95+1` 或约等于 `7+1`

代码先计算每轮实际提议的 draft 数 $Q_i$，再在表格中直接格式化为：

$$
\texttt{\#propose}=\overline Q+1.
$$

其中：

- `7.00` 是平均每轮提议的 draft token 数；
- `+1` 是这一轮在非 EOS 情况下总会再提交的 1 个 target-derived token 槽位：第一个拒绝处的 residual correction，或全接收后的 bonus；
- `+1` 在 `verify_rate` 分母中也代表每轮额外的这一个 target 输出机会。

它在数值上与“target 验证输入有 $a+7$ 个 token”都得到 8，但概念不同：**表格的 `+1` 指本轮新的 correction/bonus 机会，不是已经存在的 anchor $a$**。

本次 GSM8K 结果为 `6.95+1`，而不是严格的 `7.00+1`，主要是代码遇到已接收的 EOS 时，把该终止轮的 `effective_proposal_length` 截到 EOS 所在位置。当 `--confidence-threshold 0.0` 时 confidence head 不会主动缩短 proposal，因此此处小于 7 不是 confidence scheduler 在截断。如果 threshold 大于 0，$Q_i$ 还可能因 confidence 前缀截断而更小。

### 16.3 `accept_len` 怎样计入 correction/bonus，为什么不是减 2

记第 $i$ 轮连续接收的 draft 数为 $R_i$。对非 EOS 终止轮：

$$
L_i=R_i+1,
$$

其中 $L_i$ 就是该轮记入 `acceptance_lengths` 的值。这一轮新提交的 token 是：

```text
[d1, ..., dR, correction_or_bonus]
```

轮开始时的 $a$ 没有出现在这个“新提交列表”里。它在上一轮已经被计过，本轮只是利用它开始预测。代码中虽然会把 `verify_input_ids` 中的 anchor 再覆写到 `output_ids[:, start]`，但 `start` 只前进 $R_i+1$，从而明确表明 anchor 没有再次计数。

四个典型例子如下。

**例 1：7 个 draft 全部接收。**

```text
本轮开始已有：a
draft：              d1 d2 d3 d4 d5 d6 d7
验证结果：         7 个全接收
本轮再生成：     b             # target bonus
本轮新提交：     d1 ... d7 b
R=7，accept_len=8
```

$b$ 会成为下一轮的 anchor，下一轮不会再把 $b$ 计入 `accept_len`。

**例 2：前 3 个接收，$d_4$ 首次被拒绝。**

```text
本轮开始已有：a
draft：              d1 d2 d3 d4 d5 d6 d7
验证结果：         A  A  A  R
本轮再生成：              c4    # target residual correction
本轮新提交：     d1 d2 d3 c4
R=3，accept_len=4
```

这里 $c_4$ 是从归一化的正残差分布 $(p-q)_+$ 采样，用来保持 speculative sampling 与 target model 同分布。$c_4$ 会成为下一轮 anchor。$d_4$ 没有被接收，$d_5,d_6,d_7$ 也不可提交。

**例 3：第 1 个 draft 就被拒绝。**

```text
本轮新提交：c1
R=0，accept_len=1
```

即使一个 draft 都没接收，非 EOS 轮也仍会提交 1 个 target correction，因此 `accept_len` 的下界通常为 1。

**例 4：已接收的 $d_3$ 是 EOS。**

```text
本轮新提交：d1 d2 EOS
R=3，accept_len=3，立即终止
```

这是例外：既然已提交 EOS，就不再在其后提交 correction/bonus，所以该轮 $L_i=R_i$。

整个数据集报告的 `accept_len` 是 $\overline L$，因此：

$$
\overline L=\overline R+1-\frac{N_{\mathrm{EOS\ in\ accepted\ draft}}}{N_{\mathrm{proposal}}}.
$$

这意味着：

- 忽略少量 EOS 终止轮时，平均已接收 draft 数约为 `accept_len - 1`；
- 严格值应直接用代码收集的 `accepted_draft_lengths`，或用后文的逐位计数重建；
- `accept_len - 2` 多减掉了一个本来就没有在本轮计数的上一轮 anchor，因此是错的。

以本次 GSM8K 为例，`accept_len=6.1164`。由逐位的实际接收计数可重建得到 $\overline R\approx5.1333$；`accept_len-1=5.1164` 因 EOS 终止轮而略低，但仍远比错误的 `accept_len-2=4.1164` 接近真实口径。

### 16.4 `accept_rate@k` 的精确定义，是否包含 correction

`k` 从 0 开始，所以：

| 字段 | 对应 draft | 事件 |
|---|---|---|
| `accept_rate@0` | $d_1$ | 接收前缀至少到达并接收 $d_1$ |
| `accept_rate@1` | $d_2$ | $d_1,d_2$ 都被接收 |
| `accept_rate@2` | $d_3$ | $d_1,d_2,d_3$ 都被接收 |
| $\ldots$ | $\ldots$ | $\ldots$ |
| `accept_rate@6` | $d_7$ | 7 个 draft 全部被接收 |

对第 $i$ 轮，记实际 proposal 长度为 $Q_i$，连续接收 draft 数为 $R_i$。代码的精确计算是：

$$
\operatorname{accept\_rate@k}
=
\frac{\sum_i \mathbf 1[Q_i>k]\mathbf 1[R_i>k]}
{\sum_i \mathbf 1[Q_i>k]}.
$$

分母只包含“本轮确实提议了第 $k$ 个位置”的轮；分子是其中连续接收前缀确实越过该位置的轮。因此它是 **prefix survival probability**，不是“已知前面都接收时，当前这一个 token 的条件接收率”。在固定 7 长度且不考虑 EOS 时，后者可以用相邻 survival 比值估计：

$$
P(A_k\mid A_0,\ldots,A_{k-1})
\approx
\frac{\operatorname{accept\_rate@k}}
{\operatorname{accept\_rate@(k-1)}}.
$$

`accept_rate@k` **不包含第 $k$ 个 draft 被拒绝后用 target token 替换成功的情况**。例如 $d_1,d_2,d_3$ 接收、$d_4$ 拒绝后提交 $c_4$：

- 该轮为 `accept_rate@0/@1/@2` 的分子各贡献 1；
- 为 `accept_rate@3` 的分母贡献 1，但分子贡献 0；
- $c_4$ 不会让 `accept_rate@3` 变成接收；
- 该轮 `accept_len=4`，因为它确实一次向前提交了 4 个新 token。

还要注意，本次设置为 `temperature=1.0`，“接收”并不等于“draft token 与 target argmax 相同”或“token 的语义是正确的”。对 draft 采样 token $x$ 的验证接收概率是：

$$
\alpha(x)=\min\left(1,\frac{p_{\mathrm{target}}(x)}
{q_{\mathrm{draft}}(x)}\right),
$$

代码再用一个随机数决定本次是否接收。因此更准确的说法是“通过无损 speculative rejection sampling 验证”，而不是“判断 token 对错”。

当每轮都有 7 个 proposal 且忽略 EOS 时，有一个很有用的检查关系：

$$
E[R]=\sum_{k=0}^{6}\operatorname{accept\_rate@k},
\qquad
\operatorname{accept\_len}=1+E[R].
$$

当 $Q_i$ 因 EOS 或 threshold 而变化时，要用各位置分母 $N_k$ 恢复严格值：

$$
E[R]=\frac{1}{N}\sum_{k=0}^{6}
N_k\operatorname{accept\_rate@k}.
$$

### 16.5 `verify_rate` 与上述口径的关系

当前代码的定义是：

$$
\operatorname{verify\_rate}
=
\frac{\sum_i L_i}
{\sum_i Q_i+N_{\mathrm{proposal}}}
=
\frac{\operatorname{accept\_len}}
{\overline Q+1}.
$$

以本次 GSM8K 的未四舍五入数值为例：

$$
\frac{6.1164346}{6.9467605+1}=0.7696765,
$$

与表格的 `verify_rate=0.7697` 一致。这个量表示 target 一次验证所覆盖的 token 机会中，最终实际向前提交的比例；它不是单个 draft token 的接收概率。

### 16.6 confidence 指标实际比较的两个量

是的，这组评估指标比较的就是：

1. confidence head 对 prefix acceptance 的预测概率；
2. target 实际验证过程中得到的二值 prefix acceptance 结果。

但在理解前还要区分“单步 confidence”和“前缀 confidence”。

对某轮第 $k$ 个 draft 位置，confidence head 首先输出一个标量 raw logit $z_k$，并转成：

$$
s_k=\sigma(z_k).
$$

$s_k$ 被解释为“前面位置可用时，当前这一步的预期接收置信度”。但当前 ECE/AUC/Brier 评估不直接用 $s_k$，而是先计算累乘：

$$
\widehat p_k=\prod_{j=0}^{k}s_j.
$$

$\widehat p_k$ 是“从 $d_1$ 到 $d_{k+1}$ 的整个 draft 前缀都能接收”的预测概率。对应的实际标签是：

$$
y_k=\prod_{j=0}^{k}\mathbf 1[\text{draft }j\text{ 的本次验证事件接收}],
$$

因此 $y_k\in\{0,1\}$。一旦前面某个位置被拒绝，该位置以及所有后续位置的 prefix label 都为 0。

例如 raw 单步 confidence 为：

```text
s0=0.90, s1=0.80, s2=0.70
```

则用于指标计算的前缀预测是：

```text
pred:   [0.90, 0.90*0.80, 0.90*0.80*0.70] = [0.90, 0.72, 0.504]
```

如果本次验证前两个接收、第三个拒绝，则：

```text
target: [1, 1, 0]
```

第三个位置随后用 correction 替换也不会把最后一个 label 改成 1。

### 16.7 confidence head 与 Markov vocabulary probabilities 不是同一个输出

当前 `dspark_qwen3_8b_block7` checkpoint 的配置是：

```text
markov_rank=256
markov_head_type="vanilla"
enable_confidence_head=true
confidence_head_with_markov=true
```

这里有两条不同的输出支路。

**Markov draft 支路：**

$$
\widetilde{\mathbf u}_k
=
\mathbf u_k+W_2E_{\mathrm{Markov}}(d_{k-1}),
\qquad
q_k=\operatorname{softmax}(\widetilde{\mathbf u}_k/T).
$$

$\widetilde{\mathbf u}_k$ 是词表大小的修正 logits，$q_k$ 是词表分布。代码用它采样具体 draft token，并作为 rejection sampling 中的 `draft_probs`。

**Confidence 支路：**

$$
z_k=W_c[h_k;E_{\mathrm{Markov}}(d_{k-1})]+b_c,
\qquad
s_k=\sigma(z_k).
$$

它将 draft hidden state $h_k$ 与前一个 token 的 256 维 Markov embedding 拼接，再通过单独训练的 `AcceptRatePredictor` 线性层输出 1 个标量。它：

- 不输出 151936 维词表 logits；
- 不对 Markov 修正后的 logits 直接做 softmax 就得到 acceptance confidence；
- 不参与选择“具体应该生成哪个 token”；
- 是 checkpoint 中一套单独训练的 `confidence_head.proj.weight/bias` 参数。

`confidence_head_with_markov=true` 表示 confidence head 会使用与 Markov head 相关的前 token embedding 作为特征，**不表示 confidence 值等于 Markov vocabulary softmax 中某个 token 的概率**。

训练时，confidence head 的 soft target 也不是某一次随机验证的 0/1 结果，而是 Markov 修正后 draft 分布 $q$ 与 target 分布 $p$ 的重叠质量：

$$
a^*
=1-\frac12\lVert q-p\rVert_1
=\sum_v\min(q(v),p(v)).
$$

因为 $x\sim q$ 时：

$$
E_{x\sim q}\left[\min\left(1,\frac{p(x)}{q(x)}\right)\right]
=\sum_v\min(q(v),p(v)),
$$

所以 $a^*$ 正好是该分布对下 speculative rejection sampling 的**期望单步接收率**。训练代码用 BCE-with-logits 让 $\sigma(z_k)$ 拟合这个 soft target。

因此，对“是不是一套训练后、更能反映预期接收情况的新概率”的准确回答是：

- **是**，它是一个独立训练的 acceptance estimator，目标就是在 target 验证之前廉价地预测接收可能性；
- **但不能把“更真实”理解为它是 oracle 或必然完美校准**。训练使用的是 soft expected overlap，而评估比较的是实际采样中带随机性的 0/1 prefix outcome，还会存在训练到评测分布偏移；
- ECE、Brier、`pred_mean-target_mean` 用来衡量它是否校准，AUC 用来衡量它是否能排序区分易接收和难接收的 proposal。

以本次 GSM8K 为例，整体 `pred_mean=0.8335`，而 `target_mean=0.7390`，表明这个数据集上 confidence head 整体偏乐观。在 `pos=6` 上，它平均预测“7 个 draft 全接收”的概率为 `0.6968`，实际只有 `0.5615`，所以该位置 `ECE=0.1353`。这就是不能只因为它经过训练就把它当成真实接收概率的具体例子。

### 16.8 ECE、AUC、Brier、pred mean 和 target mean 的定义

对固定位置 $k$，收集所有有效 proposal 的二元组：

$$
(\widehat p_{i,k}, y_{i,k}),
$$

其中 $\widehat p_{i,k}\in[0,1]$ 是 confidence 累乘后的前缀预测，$y_{i,k}\in\{0,1\}$ 是实际前缀接收标签。记该位置有 $N_k$ 个有效观测。

#### 16.8.1 `pred_mean@k` 和 `target_mean@k`

$$
\operatorname{pred\_mean@k}
=\frac{1}{N_k}\sum_i\widehat p_{i,k},
$$

$$
\operatorname{target\_mean@k}
=\frac{1}{N_k}\sum_i y_{i,k}.
$$

`pred_mean@k` 是 confidence head 对到达并接收该前缀的平均预测概率；`target_mean@k` 是实际前缀接收率。在本次 threshold=0 评估中，两套统计使用相同的有效位置口径，所以：

$$
\operatorname{target\_mean@k}
=\operatorname{accept\_rate@k}.
$$

例如 GSM8K 的 `pos=2` 图上 `mean_target=0.7949`，与主 spec 表的 `accept_rate@2=0.7949` 完全一致。

主 confidence 表中不带 `@k` 的 `pred_mean` 和 `target_mean` 是对所有位置按 $N_k$ 加权后的总平均，不是 `pos=0` 的值：

$$
\operatorname{pred\_mean}
=\frac{\sum_kN_k\operatorname{pred\_mean@k}}{\sum_kN_k},
$$

`target_mean` 同理。

#### 16.8.2 `ECE@k`

ECE 是 Expected Calibration Error。当前代码把 $[0,1]$ 等宽分成 20 个 coarse bins，每个宽度为 0.05。对某个非空 bin $b$：

$$
\operatorname{conf}(b)=\frac{1}{|b|}\sum_{i\in b}\widehat p_{i,k},
\qquad
\operatorname{acc}(b)=\frac{1}{|b|}\sum_{i\in b}y_{i,k}.
$$

然后：

$$
\operatorname{ECE@k}
=\sum_b\frac{|b|}{N_k}
\left|\operatorname{conf}(b)-\operatorname{acc}(b)\right|.
$$

它衡量“说 80% 时实际是否约有 80% 接收”。越接近 0 越好。例如在 `[0.80,0.85)` bin 中有 100 个 proposal，平均预测 0.825，其中 70 个实际 prefix 接收，则该 bin 对 ECE 的加权贡献是：

$$
\frac{100}{N_k}|0.825-0.70|.
$$

ECE 依赖分 bin 方式，它不是唯一的 calibration 真理；样本很少的 bin 也容易有很大随机波动。

#### 16.8.3 `AUC@k`

`AUC@k` 是将 $y_{i,k}=1$ 当作正例、$y_{i,k}=0$ 当作负例时的 AUROC。它可以解释为：

$$
P(\widehat p_{\mathrm{positive}}>
\widehat p_{\mathrm{negative}})
+\frac12P(\text{tie}).
$$

- AUC=1：所有实际可接收前缀的 confidence 都高于不可接收前缀；
- AUC=0.5：排序区分能力约等于随机；
- AUC<0.5：排序方向倾向相反；
- 如果某位置的 label 全是 1 或全是 0，AUC 无法定义，代码输出 `nan`。

当前实现没有保存每一个样本后直接 sort，而是用 1000 个 fine bins 累计正负例后近似计算 AUC，同一 fine bin 内按 0.5 个 tie 计。

AUC 只看排序，不看绝对概率是否校准。例如把一组能正确排序的 `[0.9, 0.8, 0.7]` 全改成 `[0.6, 0.55, 0.5]`，AUC 可能不变，但 ECE 和 Brier 会改变。

#### 16.8.4 `Brier@k` 和 `brier_mean`

$$
\operatorname{Brier@k}
=\frac{1}{N_k}\sum_i(\widehat p_{i,k}-y_{i,k})^2.
$$

Brier score 越低越好，0 表示所有概率预测与结果完全一致。它既惩罚错误又过度自信的预测，也保留了概率误差的大小，不依赖 coarse ECE bins。

当前终端 confidence 表没有展开打印 `brier@0...brier@6`，但每个位置的 Brier 都保存在 `metrics.json` 的 `confidence.per_position[k].brier` 中，也写入 TensorBoard scalar。`brier_mean` 是按各位置有效样本数加权的平均：

$$
\operatorname{brier\_mean}
=\frac{\sum_kN_k\operatorname{Brier@k}}{\sum_kN_k}.
$$

#### 16.8.5 一个完整数值例子

假设在某个固定 `pos=k` 上有 4 个 proposal：

```text
prefix prediction: [0.90, 0.80, 0.40, 0.20]
actual label:      [1,    0,    1,    0]
```

则：

$$
\operatorname{pred\_mean@k}=0.575,
\qquad
\operatorname{target\_mean@k}=0.5.
$$

$$
\operatorname{Brier@k}
=\frac{(0.9-1)^2+(0.8-0)^2+(0.4-1)^2+(0.2-0)^2}{4}
=0.2625.
$$

正例 confidence 为 0.9、0.4，负例为 0.8、0.2，四组正负样本对中有 3 组排序正确，所以：

$$
\operatorname{AUC@k}=\frac34=0.75.
$$

如果按当前 20 个 bin 计算，四个值恰好落在不同 bin，则此小例中：

$$
\operatorname{ECE@k}
=\frac{|0.9-1|+|0.8-0|+|0.4-1|+|0.2-0|}{4}
=0.425.
$$

真实评估中一个 bin 通常包含大量 proposal，所以会先在 bin 内求平均，不是对每个样本直接计算绝对误差。

### 16.9 `ece_mean`、`auc_mean` 等整体值怎样从逐位置值得到

记第 $k$ 个位置的 `total_weight` 为 $N_k$。当前代码不是把 7 个位置的样本混在一起重新分 bin 或重新计算一次 pooled AUC，而是先逐位置计算，再加权：

$$
\operatorname{ece\_mean}
=\frac{\sum_kN_k\operatorname{ECE@k}}{\sum_kN_k},
$$

$$
\operatorname{auc\_mean}
=\frac{\sum_{k:\mathrm{AUC}_k\ne\mathrm{nan}}
N_k\operatorname{AUC@k}}
{\sum_{k:\mathrm{AUC}_k\ne\mathrm{nan}}N_k},
$$

`brier_mean`、`pred_mean`、`target_mean` 也按同样的 $N_k$ 加权。例如只有两个位置，$N_0=100,\mathrm{ECE}_0=0.02$，$N_1=80,\mathrm{ECE}_1=0.08$，则：

$$
\operatorname{ece\_mean}
=\frac{100\times0.02+80\times0.08}{180}
\approx0.0467.
$$

由于 threshold=0 时前 7 个位置的 $N_k$ 通常非常接近，整体值通常也接近 7 个逐位置值的算术平均；但严格上必须按 `total_weight` 加权。

### 16.10 怎样看 `reliability_diagram.png`

当前实验的文件位于：

```text
/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all/tensorboard/artifacts/step_0/<dataset>/reliability_diagram.png
```

每个数据集一张图，block-7 对应 `pos=0...6` 七个子图。`pos=k` 的含义是“前缀是否连续接收至 $d_{k+1}$”，不是单独忽略前缀地看第 $k+1$ 个 token。

每个子图元素的含义是：

| 图中元素 | 含义 |
|---|---|
| 横轴 `Predicted prefix acceptance` | confidence 累乘预测 $\widehat p_k$；对橙色点是某个 bin 内的实际 `avg_pred` |
| 纵轴 `Observed prefix acceptance` | 同一 confidence bin 内 $y_k=1$ 的实际比例 `avg_target` |
| 灰色虚线 $y=x$ | 完美校准线；预测 0.8 的样本应约有 80% 实际接收 |
| 橙色圆点和折线 | 每个非空 0.05 宽 confidence bin 的 `(avg_pred, avg_target)`，按 bin 顺序连线 |
| 浅蓝色柱 | 该 confidence bin 内的 proposal 数 `weight`，表示预测值分布在哪些区间 |

解读橙色线时：

- 橙色点在灰色对角线上：该 bin 校准良好；
- 橙色点在对角线下方：实际接收率低于预测，confidence head **过度自信/偏乐观**；
- 橙色点在对角线上方：实际接收率高于预测，confidence head **保守/偏悲观**；
- 离对角线很远但蓝色柱几乎不可见的点，可能只有几个样本，不应像主质量区间那样解读。

蓝色柱使用 twin y-axis，右侧刻度被代码隐藏，而且每个子图都把自己的最高柱设为该轴上限。因此：

- **不能用左边 0到1 的 `Observed prefix acceptance` 刻度读蓝色柱高**；
- 蓝色柱不是 observed acceptance，只是该 bin 的样本量；
- 不能仅凭两个子图中柱高的视觉比例比较绝对样本数，精确值要看 `metrics.json` 里每个 reliability bin 的 `weight`。

子图标题中：

- `ECE` 就是终端 confidence 表的 `ece@k`；
- `AUC` 就是终端 confidence 表的 `auc@k`；
- `mean_pred` 是该 `pos=k` 的 `pred_mean@k`；
- `mean_target` 是该 `pos=k` 的 `target_mean@k`，在本次 threshold=0 实验中也就是主 spec 表的 `accept_rate@k`。

不要把子图中的 `mean_pred/mean_target` 与终端总表不带 `@k` 的 `pred_mean/target_mean` 混淆：前者是单个位置，后者是 7 个位置的加权整体值。

以本次 GSM8K 图为例：

- `pos=0`: `mean_pred=0.9635`，`mean_target=0.9366`，`ECE=0.0272`，预测总体略乐观，但偏差较小；
- `pos=6`: `mean_pred=0.6968`，`mean_target=0.5615`，`ECE=0.1353`，主要样本区间的橙色线大部分位于对角线下方，表明“7 个全接收”的前缀概率明显高估；
- `pos=0` 的 `AUC=0.8947`，`pos=6` 的 `AUC=0.8926`，说明尽管绝对概率有过度自信，该 head 仍有较好的高低风险排序能力。

AUC 是由 1000-bin 正负例排序统计得到，不是直接对图上 20-bin 橙色折线求面积；不应从橙色线的形状直接手算标题 AUC。

### 16.11 关于 `confidence_histogram.png` 的文件名核对和折线/柱状图解释

对当前代码和本次成功实验的实际产物进行核对后，每个数据集目录只有：

```text
metrics.json
reliability_diagram.png
```

当前 `deepspec/eval/dspark/confidence_head.py` 中也只定义和写出 `reliability_diagram.png`，没有生成名为 `confidence_histogram.png` 的独立文件。你描述的“横坐标 `Predicted prefix acceptance`、纵坐标 `Observed prefix acceptance`、折线和柱状图”正是当前 `reliability_diagram.png` 的内容：它已经把 reliability curve 和 confidence histogram 叠加在同一个子图中。

所以这两种图层应这样理解：

- **橙色折线**：回答“模型给出某个预测概率区间时，实际 prefix 接收比例是多少”，用来看 calibration；
- **蓝色柱状图**：回答“confidence 预测值主要分布在哪些概率区间、每个区间有多少 proposal”，用来判断橙色折线某段是主质量区间还是小样本噪声；
- **灰色对角线**：理想校准基准。

如果在其他分支、旧版本或手工导出目录中确实看到了独立的 `confidence_histogram.png`，需要再根据生成该文件的对应版本代码判断纵轴是 raw count、normalized frequency 还是 density；不能将其与本次产物混为同一个当前代码文件。

### 16.12 结果解读的快速检查顺序

对一个数据集，建议按以下顺序看：

1. 先看 `#propose`：确认 threshold=0 时约为 `7+1`，小量低于 7 可由终止 EOS 解释；
2. 再看 `accept_len`：它是每轮新提交 token 数，普通轮已接收 draft 数为 `accept_len-1`；
3. 用 `verify_rate = accept_len / (#propose 的数值总和)` 做算术自检；
4. 看 `accept_rate@0...@6` 的 prefix survival 衰减，记住 correction 不计入被拒绝位置；
5. 对 confidence 先比 `pred_mean` 与 `target_mean`，判断总体乐观/保守方向；
6. 用 ECE/Brier 看绝对概率质量，用 AUC 看排序区分能力，不要用 AUC 代替 calibration；
7. 打开 `reliability_diagram.png`，重点看蓝色柱高的主质量 bin 中橙色线是否系统地在对角线上方或下方；
8. 需要精确 bin count、逐位 Brier 或更多小数位时，以同目录 `metrics.json` 为准，不要从 PNG 手工估算。
