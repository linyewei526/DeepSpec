# DSpark Qwen3-8B Markov 修正草稿概率与纠错排名观测指南

## 1. 实验目的

本实验在既有 Qwen3-8B DSpark 推理复现链路上增加只读观测，不修改 `deepspec/`、`eval.py` 或 `runtime/` 中的复现代码，也不改变采样、验证、接受或纠错逻辑。观测钩子在一次验证已经得到结果后运行，不调用随机数生成函数，因此在相同 checkpoint、数据、超参、rank 数和 seed 下不会改变解码结果。

本实验不是任务正确率、`pass@1`、LLM judge、端到端速度或 TPS 实验。逐轮把 GPU 张量同步到 CPU 并计算完整词表排名会增加观测开销，所以本实验产物不能用来代表无观测 baseline 的耗时。

## 2. 原四项指标与新增拒绝预测的精确定义

令第 $k$ 个 draft 位置经 Markov 头修正后的 logits 为 $\ell_k$。本实验定义温度无关的 diagnostic 分布为

\[
q_k^{obs}=\operatorname{softmax}(\ell_k),
\]

实际提交验证的 draft token 记为 $z_k$，本实验用于分布和差值统计的标量是

\[
P_k=q_k^{obs}(z_k).
\]

也就是对 Markov 修正 logits 直接做普通 softmax 后，实际提交 token 自己的概率，不执行 `/temperature`。采样与 speculative verification 仍使用另一份 operational 分布 $q_k^{verify}$：`temperature>=1e-5` 时为 $\operatorname{softmax}(\ell_k/T)$，`0<=temperature<1e-5` 时为 greedy argmax 的 one-hot 分布。$q_k^{obs}$ 仅供观测，绝不替换 verifier 的 `proposal.draft_probs`。因此 greedy 时实际提交 token 的 $P_k$ 通常小于 1，仍能反映 Markov logits 的概率强弱。这里的“温度无关”指从当前 $\ell_k$ 到 $q_k^{obs}$ 的映射不除以温度；不同温度可能改变已采样 prefix，进而间接改变后续位置的 $\ell_k$。

1. `accepted_selected_draft_probability`：所有验证轮中，逐个记录实际通过验证位置的 $P_k$。报告总数、均值、最小值、最大值，以及宽度为 0.05 的 PMF/CDF。区间固定为 `[0.00,0.05)` 到 `[0.95,1.00]`。
2. `rejected_selected_draft_probability`：仅记录每轮第一个验证失败、随后由 `verification.next_token` 替换的位置上错误 draft token 的 $P_k=q_k^{obs}(z_k)$。这里不是 correction token 的概率。后续未独立验证的位置不纳入，空 proposal 也不伪装成拒绝事件。
3. 同轮有至少一个已通过 draft token 时，令 `accepted_mean` 为本轮这些通过位置的 $P_k$ 均值：
   - `signed_absolute_gap = accepted_mean - rejected_probability`；
   - `signed_relative_gap = signed_absolute_gap / accepted_mean`；
   - `signed_relative_gap_mean_percent = 100 * mean(signed_relative_gap)`。

   有 accepted prefix 的 correction event 先计入 `gap_candidate_events`。若 `signed_absolute_gap < 0`，只增加 `negative_gap_excluded_events`；除这个明确保留并展示的排除审计计数外，该事件不增加 `paired_gap_events`，也不进入 absolute/relative gap 的样本计数、均值、PMF/CDF、CSV、图或 TensorBoard 概率统计。接受位置和拒绝位置各自的 $P_k$ 分布以及 `true_draft_rank` 仍照常记录。若 `signed_absolute_gap >= 0`（包含恰好为零），才计入 `paired_gap_events` 和 gap 分布。因此以下关系必须成立：

   ```text
   correction_events = first_position_correction_events + gap_candidate_events
   gap_candidate_events = paired_gap_events + negative_gap_excluded_events
   ```

   过滤后的 absolute gap 和 relative gap 都位于 `[0,1]`，固定按 0.05 分为 `[0.00,0.05)` 至 `[0.95,1.00]`。若第一个位置就失败，本轮计入 `first_position_correction_events`，但不是 gap candidate。若一个已纳入事件的 `accepted_mean` 数值为零，会计入 `undefined_relative_gap_events`：absolute gap 仍有效，relative gap 不记录。这里报告的是以 `signed_absolute_gap >= 0` 为条件的 gap 分布，不再表示全部 correction events 的无条件分布。
4. `true_draft_rank`：在失败位置完整的温度无关 diagnostic 分布 $q_k^{obs}$ 上，计算实际 correction token 的真实 competition rank。代码直接比较对应的 Markov 修正 logits，利用 softmax 的严格单调性避免极小概率下溢造成伪并列：

\[
\operatorname{rank}=1+\#\{v:\ell_k(v)>\ell_k(\text{correction token})\}.
\]

   排名使用完整词表，不使用 top-k 近似；相同 logits 共享同一名次。输出类别为 `1,2,...,10,other`，每类给出 count 和占全部 correction events 的 probability，同时附该类 correction token 的 $q_k^{obs}$ 概率均值、最小值和最大值。greedy 下排名不会因 verifier 的 one-hot operational 分布而退化。

DSpark 当前验证实现中的 correction token 是从 operational 分布构造的归一化正残差 `[p_k^{verify}-q_k^{verify}]_+` 采样并实际提交的 `verification.next_token`；greedy 时该 token 就是 target argmax。这里的“被替换上的 token”严格指 `verification.next_token`。

## 3. 代码隔离与调用链

本实验所有接口代码都位于独立子目录：

- `observations/markov_draft_probability/__init__.py`：子实验包入口；
- `observations/markov_draft_probability/markov_probability_observation.py`：观测累加器、跨 rank 合并、CDF/rank 产物和隔离 evaluator；
- `observations/markov_draft_probability/run_markov_probability_observation.py`：时间戳目录校验、不可变 settings、自动端口租约、manifest 和多 GPU 启动；
- `observations/markov_draft_probability/summarize_markov_probability_observation.py`：终端汇总工具，并可从新 schema 的精确计数幂等重建拒绝预测 Markdown。
- `observations/markov_diagnostic_draft.py`：两组 Markov 观测共用的隔离 proposal mixin；同时保留 verifier 的 operational `draft_probs` 和只供观测的 corrected logits。

调用链为 `run_markov_probability_observation.py -> torch.multiprocessing.spawn -> MarkovDraftProbabilityEvaluator -> generate_decoding_sample -> DiagnosticMarkovProposalMixin._propose -> build_diagnostic_markov_proposal -> verify_draft_tokens -> _post_verify`。新 evaluator 先保留父类原有 confidence 校准记录，再执行本实验的只读观测。

隔离 proposal builder 复用本轮已经计算出的 Markov corrected logits：按解码温度构造 operational `proposal.draft_probs` 交给 verifier，同时把同一份 corrected logits 暂存到 `diagnostic_markov_logits`。观测器对后者执行普通 float32 softmax，再按 `proposal.verify_input_ids[:,1:]` gather 得到 $q_k^{obs}(z_k)$；不重新执行模型前向、不消费额外随机数，也不改变 verify 使用的分布。

每个 rank 只写本轮结果目录下自己的 `rank_stats/rank_<rank>.json`；barrier 后由 rank 0 合并，因此不同实验只要使用不同时间戳目录就不会互相覆盖。启动器省略 `--master-port` 时会探测空闲本地端口，并在所有观测实验共享的 `/tmp/deepspec_conditional_confidence_ports/` 建立运行期端口租约；已有监听端口及并行启动的观测实验租约都会避开。若手工传入 `--master-port`，已占用或已租约的端口会直接报错。

## 4. 环境与 baseline 对齐

环境沿用复现指南：

- Conda Python：`/data/home/wly/.conda/envs/dspark/bin/python`；
- target：`/data1/linyewei/models/Qwen3-8B`；
- draft：`/data1/linyewei/models/dspark_qwen3_8b_block7`；
- 非 thinking chat template；
- `max_new_tokens=2048`、`temperature=1.0`、`confidence_threshold=0.0`、`seed=980406`；
- SDPA、每进程 batch size 1、Gloo；
- 全量九数据集上限依次为 GSM8K 500、MATH-500 500、AIME25 30、HumanEval 164、MBPP 256、LiveCodeBench 500、MT-Bench 80、Alpaca 500、Arena-Hard-v2 500。

这些设置与 `notes/basis/DSpark_Qwen3-8B_推理复现指南.md` 第 8.2 节示例一致。`CUDA_VISIBLE_DEVICES` 只决定该次进程可见的 GPU；启动器不会替用户抢占、清理或限制其他 GPU 任务。每张可见 GPU 都加载一份完整 target+draft，并按 rank 切分样本。

## 5. 推荐命令

以下每条命令都是单个物理行。先确认使用 dspark 环境，或直接像示例一样使用该环境的绝对 Python 路径。

### 5.1 两个样本的轻量 smoke

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_markov_draft_probability_smoke_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/run_markov_probability_observation.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 64 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 --max-samples 2 2>&1 | tee "$RUN_DIR/eval.log"
```

### 5.2 推荐的两卡九数据集全量命令

该命令按原指南第 8.2 节使用物理 GPU 2、3；可按实际显存情况修改可见卡列表。这里故意不设置固定 `MASTER_PORT`，由启动器自动分配。

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_markov_draft_probability_all && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/run_markov_probability_observation.py all --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 0.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 --score-threshold-min 0.02 --score-threshold-max 1.00 --score-threshold-step 0.02 2>&1 | tee "$RUN_DIR/eval.log"
```

### 5.3 单数据集全量命令

将位置参数 `gsm8k` 换成其余八个合法名字即可。

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_markov_draft_probability_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/run_markov_probability_observation.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 2>&1 | tee "$RUN_DIR/eval.log"
```

### 5.4 汇总已完成的数据集

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/summarize_markov_probability_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>
```

只看一个数据集：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/summarize_markov_probability_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --dataset gsm8k
```

在终端额外展开完整 0.05 CDF：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/summarize_markov_probability_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --show-cdf
```

## 6. 参数逐项说明

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 位置参数 `dataset` | 无 | `all` 或九个数据集名称之一：`gsm8k`、`math500`、`aime25`、`humaneval`、`mbpp`、`livecodebench`、`mt-bench`、`alpaca`、`arena-hard-v2`。 |
| `--run-dir` | 必填 | 必须是结果根目录的直接子目录，名字匹配 `YYYYMMDD_HHMMSS_<标签>`。目录只允许预先存在由 `tee` 建立的 `eval.log`；已有实验目录拒绝复用。 |
| `--target` | Qwen3-8B 本地路径 | target checkpoint。启动前验证本地目录和 `config.json`，不会联网下载。 |
| `--draft` | block7 本地路径 | DSpark draft checkpoint；必须是 `Qwen3DSparkModel` 且启用 Markov 修正。`--confidence-threshold > 0` 时还必须有 confidence head。 |
| `--max-new-tokens` | `2048` | 每个样本最多生成 token 数。smoke 可临时设为 `64`。 |
| `--temperature` | `1.0` | target/draft 采样及验证温度，必须有限且不小于 0；`0<=temperature<1e-5` 为精确 greedy。该参数不缩放本实验的 diagnostic $q_k^{obs}$。 |
| `--confidence-threshold` | `0.0` | draft confidence early-stop 阈值，范围 `[0,1]`。为与 baseline 和全 block 观测一致，全量实验保持 `0.0`。 |
| `--seed` | `980406` | 数据子采样与逐样本随机采样 seed。 |
| `--step` | `0` | TensorBoard step 和原有 confidence artifact 的 step 目录。 |
| `--dist-backend` | `gloo` | 多进程归约后端，可选 `gloo`/`nccl`；baseline 对齐使用 `gloo`。 |
| `--dist-timeout-minutes` | `1440` | 进程组超时分钟数。 |
| `--master-addr` | `127.0.0.1` | 单机分布式 rendezvous 地址。 |
| `--master-port` | 自动 | 不传时自动探测并租约；传入时严格检查端口范围、占用和同类实验租约。 |
| `--max-samples` | 无 | 对每个所选数据集覆盖样本上限，但不会超过内置 cap；仅建议 smoke 使用。 |
| `--score-threshold-min` | `0.02` | 直接拒绝预测的最小 diagnostic Markov 概率阈值，包含端点。 |
| `--score-threshold-max` | `1.00` | 直接拒绝预测的最大概率阈值，包含端点。 |
| `--score-threshold-step` | `0.02` | 阈值步长；区间必须可被它整除。默认共 50 个阈值。 |

环境变量说明：`CUDA_VISIBLE_DEVICES` 决定进程数和逻辑卡映射；`env -u RANK -u WORLD_SIZE` 避免继承外部分布式作业的 rank；`env -u MASTER_PORT` 确保没有旧端口值干扰自动分配；`PYTHONPATH` 指向仓库；两个 offline 变量禁止 Hugging Face 联网解析。

## 7. 结果目录与文件

启动器接受目录后首先以独占创建方式写 `settings.json`，而且后续绝不修改它。它记录实验目标、公式、超参、数据文件 SHA-256、checkpoint config SHA-256、实时 Git commit/工作树、GPU 映射、自动端口和完整命令。随后才建立 manifest、artifact 目录和加载模型；因此模型加载失败时仍保留启动设置。

典型目录如下：

```text
YYYYMMDD_HHMMSS_markov_draft_probability_all/
├── settings.json
├── experiment_manifest.json
├── dataset_results.jsonl
├── markov_draft_probability_rejection_thresholds.md
├── eval.log
├── progress/<dataset>/...
├── tensorboard/
│   └── artifacts/step_0/<dataset>/...
└── observations/markov_draft_probability/<dataset>/
    ├── metrics.json
    ├── markov_draft_probability_cdf.csv
    ├── signed_gap_cdf.csv
    ├── true_draft_rank.csv
    ├── rejection_prediction_thresholds.csv
    ├── observation_plots.png
    └── rank_stats/rank_<rank>.json
```

- `experiment_manifest.json`：可变运行状态；每完成一个数据集立即更新，失败时保留错误堆栈。
- `dataset_results.jsonl`：每完成一个数据集 `flush+fsync` 追加一行，包含 baseline speculative metrics、原有累计 confidence 摘要和本实验摘要。
- `metrics.json`：该数据集的完整定义、计数、均值、0.05 分箱、PMF、CDF 和 rank 分布，是权威聚合结果。
- `markov_draft_probability_cdf.csv`：通过位置提交 token 与失败位置错误 draft token 的 $q_k^{obs}(z_k)$ 分布。
- `signed_gap_cdf.csv`：过滤掉 `signed_absolute_gap < 0` 事件后，非负 absolute gap 和 relative gap 在 `[0,1]` 上的分布。
- `true_draft_rank.csv`：`1..10,other` 的 count/probability，以及各类 correction token 的 diagnostic $q_k^{obs}$ 概率统计。
- `rank_stats/`：每个 rank 的充分统计量，便于审计跨卡合并；不是逐 token 原始日志。
- `tensorboard/`：保留原复现的 speculative/confidence 指标，并增加 `markov_draft_probability/<dataset>/...` 标量。

CDF CSV 中 `probability` 是当前 0.05 区间占全部已纳入该 gap 分布事件的比例，`cdf` 是截至该区间上沿的累计比例。对 relative gap，数值 `0.25` 表示失败位置错误 draft token 的 $P_k$ 比同轮通过位置均值低 25%。负 relative gap 不会出现，因为对应事件已由 `signed_absolute_gap < 0` 条件排除；排除规模应查看 `metrics.json`、`dataset_results.jsonl` 或终端汇总中的 `negative_gap_excluded_events`。

当前温度无关 diagnostic 概率加精确直接拒绝预测口径使用 schema version 4。目录 `20260807_173321_markov_draft_probability_all` 是 schema version 1；`20260807_180318_markov_draft_probability_all` 是 schema version 2，虽已采用非负 gap 过滤，但仍观测 temperature-dependent operational `draft_probs`；schema version 3 已使用正确 diagnostic 概率，但尚未保存 0.02 阈值计数。旧目录都不会被原地改写；新口径必须建立新的时间戳目录运行。schema version 4 settings 会同时写明 diagnostic/operational 分布分离、`[0,1]` gap CDF 范围和完整阈值列表。

## 8. 运行监控与完整性检查

查看 manifest：

```bash
/data/home/wly/.conda/envs/dspark/bin/python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/experiment_manifest.json
```

查看某数据集聚合进度：

```bash
watch -n 2 '/data/home/wly/.conda/envs/dspark/bin/python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/progress/gsm8k/progress.json'
```

持续查看日志：

```bash
tail -f /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/eval.log
```

确认九项全部完成：

```bash
/data/home/wly/.conda/envs/dspark/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["status"], p["completed_dataset_count"], [(d["name"],d["status"]) for d in p["datasets"]])' /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/experiment_manifest.json
```

完整全量运行应满足：manifest 的 `status` 为 `completed`、`completed_dataset_count` 为 9、九个 dataset status 全为 `completed`，并且每个数据集都有上述聚合文件和所有可见 rank 的 `rank_stats`。不能仅凭进程退出或一张图判断实验完整。

## 9. 每 0.02 diagnostic Markov 概率阈值的直接拒绝预测表

schema version 4 在原四项观测之外，对每个实际可判定 token 使用温度无关的 $P_i=\operatorname{softmax}(\text{markov\_corrected\_logits}_i)[z_i]$，其中 $z_i$ 是实际提交的 draft token。默认阈值为 `0.02, 0.04, ..., 1.00`，在阈值 $t$ 下仅当 `P_i < t`（严格小于）才标记为“预测拒绝”。位置 0 纳入；实际通过位置记 accepted，首个失败并被 AR token 替换的位置记 rejected，首拒之后和 accepted EOS 后丢弃的位置不计入。该 diagnostic 概率及阈值统计不做 temperature 缩放，也不替换 operational verifier 分布。

根目录 `markov_draft_probability_rejection_thresholds.md` 每完成一个数据集就写一张表，字段与置信度下降拒绝预测表一致。文档末尾为跨数据集宏平均：每项指标先在各数据集独立计算，再对数据集做算术平均，不让 token 多的数据集主导；count 均值允许为小数。比例出现 0/0 时从该比例的平均中排除，并用 `(有定义数据集数/总数据集数)` 审计。

推荐命令沿用第 5 节全部 baseline 超参；默认参数已经生成 0.02 网格。显式控制时，在启动命令末尾 `2>&1` 前增加 `--score-threshold-min 0.02 --score-threshold-max 1.00 --score-threshold-step 0.02`；区间必须位于 `[0,1]` 且可被 step 精确整除。

对 schema version 4 的已完成目录幂等重建 Markdown（单行命令）：

```bash
cd /data/home/wly/dLLM/DeepSpec && PYTHONPATH=/data/home/wly/dLLM/DeepSpec /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/markov_draft_probability/summarize_markov_probability_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<新时间戳目录> --refresh-rejection-markdown
```

schema version 3 及更早目录只有宽度 0.05 的直方图，没有逐 token 原始值，无法精确还原 0.02 严格阈值表；汇总器会明确提示重跑。原有 probability、gap、rank 结果仍可继续使用，但不能冒充本新增统计。
