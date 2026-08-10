# DSpark Qwen3-8B 置信度下降拒绝预测观测指南

## 1. 实验问题

前一项条件置信度实验回答的是“已经被拒绝的位置，其 confidence 是否通常更低”。本实验反向考察：当某个 draft token 的原始条件接收 confidence 相比同轮此前 token 的均值明显下降时，它是否确实是首个拒绝位置，以及有多少实际会被接受的位置被这种规则误判。

本实验只增加验证后的确定性计数，不修改 `deepspec/`、`eval.py`、`runtime/` 或已有 `observations/conditional_confidence/`，不改变 draft、verify、随机采样、接受、纠错和 EOS 行为。观测代码不调用随机数生成函数，因此相同 checkpoint、数据、进程数、超参和 seed 下不会改变生成结果。

这不是任务正确率、`pass@1`、LLM judge、端到端速度或 TPS 实验。虽然阈值比较只涉及每轮最多 7 个 token，但观测和额外产物仍会引入开销，不能用本实验耗时替代无观测 baseline 的速度。

## 2. Confidence 与前缀均值

对于一次 draft forward 产生的序列，令位置 $i$ 的原始条件接收 confidence 为

\[
C_i=\operatorname{sigmoid}(\text{confidence\_logit}_i).
\]

$C_i$ 表示“此前 draft prefix 全部被接受时，位置 $i$ 自身的接收 confidence”，不是累积概率 $\prod_{j\le i}C_j$。

对 $i\ge1$，此前位置均值定义为

\[
C_{i,\mathrm{mean}}=\frac{1}{i}\sum_{j=0}^{i-1}C_j.
\]

例如 block size 为 7 时，位置 `i=4` 使用 `C_0,C_1,C_2,C_3` 的均值。位置 `i=0` 没有此前 token，因此没有 $C_{i,\mathrm{mean}}$，不能参加下降判断。

## 3. 两类下降规则和阈值

绝对下降定义为

\[
D_i^{abs}=C_{i,\mathrm{mean}}-C_i.
\]

当 $D_i^{abs}\ge x$ 时，该位置标记为 `token_x_drop_abs`。默认 $x$ 从 `0.050` 到 `0.250`，步长 `0.005`，包含两个端点，共 41 个阈值。

比例下降定义为

\[
D_i^{pct}=\max\left(0,1-\frac{C_i}{C_{i,\mathrm{mean}}}\right).
\]

当 $D_i^{pct}\ge y$ 时，该位置标记为 `token_y_drop_pct`。默认 $y$ 从 `0.050` 到 `0.300`，步长 `0.005`，包含两个端点，共 51 个阈值。两者相等时比例下降为 0；$C_i=0$ 且前缀均值大于 0 时比例下降为 1；$C_i>C_{i,\mathrm{mean}}$ 时按 0 处理，不会命中正阈值。

所有比较都包含边界，即使用 `drop >= threshold`。启动器使用十进制数构造完整网格，并要求 `(max-min)` 能被 step 精确整除，避免反复浮点加法丢失端点。若极端数值导致 $C_{i,\mathrm{mean}}=0$，percentage 指标不添加会改变定义的 epsilon，而是将该 accepted/rejected outcome 记入 undefined 计数；absolute 指标仍有效。

## 4. Verify 标签和删失规则

设一轮的 `accepted_draft_tokens = k`：

- 位置 `i < k` 是实际 accepted；
- 非 EOS 且发生 correction 时，位置 `i = k` 是实际 first rejected/replaced position；
- 位置 `i > k` 位于首个拒绝之后，没有独立验证结果，不增加任何阈值的 accepted/rejected 数；
- 若接受的 prefix 中出现 EOS，EOS 位置本身是 accepted，EOS 后已经生成但不再提交的位置全部忽略；
- 若整块通过，所有 `i>=1` 的位置都可作为 accepted outcome；
- 若第一个位置就拒绝，该拒绝记入 `unscorable_first_position_rejections`，因为 `i=0` 没有此前 confidence 均值；本轮所有后续位置也位于首拒绝之后，全部忽略；
- 空 proposal 不产生可判定 token。

因此，结果只描述“存在此前同轮 token、且 verify 确实到达的可判定位置”。它不会把首拒绝之后的未知位置当作接受，也不会把它们当作拒绝。

## 5. 每个阈值记录的指标

absolute 和 percentage 的每个阈值都输出：

| 字段 | 定义 |
|---|---|
| `accepted_count` | 命中下降规则但实际 accepted 的数量，即误报数量。 |
| `rejected_count` | 命中下降规则且实际为首个 rejected position 的数量。 |
| `flagged_evaluable_count` | `accepted_count + rejected_count`。 |
| `accepted_share_among_flagged` | `accepted_count / flagged_evaluable_count`，所有预测低 confidence token 中实际通过的比例，也称 false-discovery share。 |
| `rejected_share_among_flagged` | `rejected_count / flagged_evaluable_count`，把下降规则当作拒绝预测器时的 precision。 |
| `accepted_flag_rate` | `accepted_count / 全部可判定 accepted token 数`，即 accepted positions 的误判率/FPR。 |
| `rejected_capture_rate` | `rejected_count / 全部可判定 rejected token 数`，即拒绝召回率。 |
| `flag_rate_among_evaluable` | 命中规则的 token 占全部可判定 accepted+rejected token 的比例。 |

同一 token 可以同时命中多个阈值。例如绝对下降为 0.12 的 token 会进入 `0.050` 到 `0.120` 的所有阈值计数；这是阈值扫描的预期行为，不能跨阈值相加得到 token 总数。

除阈值结果外，还输出 verification rounds、proposal rounds、生成/接受 token 总数、可判定 accepted/rejected 分母、位置 0 排除数、首位置拒绝数、首拒绝后忽略数和 accepted EOS 后忽略数，便于审计分母。

为便于人工查阅，实验根目录同时维护 `confidence_drop_results.md`。文件在启动时写入说明头；每完成一个数据集，rank 0 写入一个 dataset 章节和两张完整表，并刷新文件末尾两张跨数据集宏平均表。`token_x_drop_abs` 表包含全部 absolute 阈值，`token_y_drop_pct` 表包含全部 percentage 阈值。比例字段在 Markdown 中显示为百分数，原始小数仍保留在 JSON/CSV 中。更新采用原子替换，因此不必等待九项全部结束即可查看已经完成的数据集。

## 6. 代码隔离和调用链

本实验位于独立子目录：

- `observations/confidence_drop_rejection/__init__.py`：实验包入口；
- `observations/confidence_drop_rejection/confidence_drop_observation.py`：同轮前缀均值、GPU 阈值计数、标签映射、跨 rank 归约和结果产物；
- `observations/confidence_drop_rejection/run_confidence_drop_observation.py`：本地 checkpoint/data 校验、时间戳目录、不可变 settings、自动端口租约、manifest 和多 GPU 启动；
- `observations/confidence_drop_rejection/summarize_confidence_drop_observation.py`：终端汇总和阈值筛选；显式传参时可幂等回填旧目录的 Markdown 宏平均表。

调用链为 `run_confidence_drop_observation.py -> torch.multiprocessing.spawn -> ConfidenceDropEvaluator -> Qwen3DSparkEvaluator.generate_one_sample -> generate_decoding_sample -> build_dspark_proposal -> verify_draft_tokens -> _post_verify`。新 evaluator 先执行父类已有的 confidence 校准记录，再增加本实验计数。

92 组阈值计数保留在各 rank 的设备上，数据集结束时才归约；Gloo 会在归约阶段转到 CPU，NCCL 保留在 GPU。每个 rank 同时写自己的 `rank_stats/rank_<rank>.json` 作为审计依据。不同实验使用独立时间戳目录；自动端口探测与前一项观测实验共用租约目录，从而避开已有监听端口和并行启动的观测任务。

## 7. 环境和 baseline 参数对齐

默认配置与 `notes/basis/DSpark_Qwen3-8B_推理复现指南.md` 第 8.2 节一致：

- Python：`/data/home/wly/.conda/envs/dspark/bin/python`；
- target：`/data1/linyewei/models/Qwen3-8B`；
- draft：`/data1/linyewei/models/dspark_qwen3_8b_block7`；
- `max_new_tokens=2048`；
- `temperature=1.0`；
- `confidence_threshold=0.0`；
- `seed=980406`；
- 非 thinking、SDPA、每进程 batch size 1；
- Gloo，分布式超时 1440 分钟；
- 全量数据集 cap：GSM8K 500、MATH-500 500、AIME25 30、HumanEval 164、MBPP 256、LiveCodeBench 500、MT-Bench 80、Alpaca 500、Arena-Hard-v2 500。

每张可见 GPU 加载一份完整 target+draft，样本按 rank 切分。启动器不抢占、不结束也不迁移其他 GPU 任务；`CUDA_VISIBLE_DEVICES` 由用户按显存情况指定。

## 8. 推荐命令

下面所有 shell 命令都是单个物理行。

### 8.1 轻量 smoke

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_confidence_drop_rejection_smoke_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/run_confidence_drop_observation.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 64 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 --abs-drop-min 0.05 --abs-drop-max 0.25 --pct-drop-min 0.05 --pct-drop-max 0.30 --drop-step 0.005 --max-samples 2 2>&1 | tee "$RUN_DIR/eval.log"
```

### 8.2 推荐的两卡九数据集全量命令

示例按原复现指南使用物理 GPU 2、3。这里不设置固定 `MASTER_PORT`，由启动器自动选择并租约。

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_confidence_drop_rejection_all && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/run_confidence_drop_observation.py all --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 --abs-drop-min 0.05 --abs-drop-max 0.25 --pct-drop-min 0.05 --pct-drop-max 0.30 --drop-step 0.005 2>&1 | tee "$RUN_DIR/eval.log"
```

### 8.3 单数据集全量命令

把位置参数 `gsm8k` 换成其他合法数据集名即可。

```bash
set -o pipefail && mkdir -p /data/home/wly/dLLM/DeepSpec-results/qwen3_8b && RUN_DIR=/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/$(date +%Y%m%d_%H%M%S)_confidence_drop_rejection_gsm8k && mkdir "$RUN_DIR" && cd /data/home/wly/dLLM/DeepSpec && env -u RANK -u WORLD_SIZE -u MASTER_PORT CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data/home/wly/dLLM/DeepSpec HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/run_confidence_drop_observation.py gsm8k --run-dir "$RUN_DIR" --target /data1/linyewei/models/Qwen3-8B --draft /data1/linyewei/models/dspark_qwen3_8b_block7 --max-new-tokens 2048 --temperature 1.0 --confidence-threshold 0.0 --seed 980406 --step 0 --dist-backend gloo --dist-timeout-minutes 1440 --master-addr 127.0.0.1 --abs-drop-min 0.05 --abs-drop-max 0.25 --pct-drop-min 0.05 --pct-drop-max 0.30 --drop-step 0.005 2>&1 | tee "$RUN_DIR/eval.log"
```

### 8.4 汇总与可选 Markdown 回填

只看各数据集分母和删失计数：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/summarize_confidence_drop_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>
```

展开所有 92 个阈值：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/summarize_confidence_drop_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --show-thresholds
```

只看 GSM8K 的 absolute 阈值：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/summarize_confidence_drop_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --dataset gsm8k --family absolute --show-thresholds
```

只比较两个 family 在阈值 0.100 的结果：

```bash
/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/summarize_confidence_drop_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --threshold 0.100
```

## 9. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 位置参数 `dataset` | 必填 | `all` 或 `gsm8k`、`math500`、`aime25`、`humaneval`、`mbpp`、`livecodebench`、`mt-bench`、`alpaca`、`arena-hard-v2`。 |
| `--run-dir` | 必填 | 结果根目录的直接子目录，名字必须匹配 `YYYYMMDD_HHMMSS_<标签>`。只允许预先存在由 `tee` 打开的 `eval.log`，拒绝复用已有实验目录。 |
| `--target` | Qwen3-8B 路径 | 本地 target checkpoint。 |
| `--draft` | DSpark block7 路径 | 本地 Qwen3 DSpark checkpoint，必须启用 confidence head。 |
| `--max-new-tokens` | `2048` | 每个样本最多生成 token 数；smoke 可改小。 |
| `--temperature` | `1.0` | target/draft 采样与验证温度，必须有限且不小于 0；`0<=temperature<1e-5` 为精确 greedy。confidence-head 条件接收概率本身不做温度 softmax。 |
| `--confidence-threshold` | `0.0` | DSpark proposal early-stop 阈值；与 baseline 及完整七位置观测对齐时保持 0。 |
| `--seed` | `980406` | 数据子采样及逐样本采样 seed。 |
| `--step` | `0` | TensorBoard step 和父类 confidence artifact 的 step。 |
| `--dist-backend` | `gloo` | 分布式归约后端，可选 `gloo` 或 `nccl`。 |
| `--dist-timeout-minutes` | `1440` | 进程组超时分钟数。 |
| `--master-addr` | `127.0.0.1` | 单机 rendezvous 地址。 |
| `--master-port` | 自动 | 不传时自动探测和租约；显式传入时检查范围、监听占用及观测实验租约。 |
| `--abs-drop-min` | `0.05` | absolute 阈值网格最小值。 |
| `--abs-drop-max` | `0.25` | absolute 阈值网格最大值。 |
| `--pct-drop-min` | `0.05` | percentage 阈值网格最小值。 |
| `--pct-drop-max` | `0.30` | percentage 阈值网格最大值。 |
| `--drop-step` | `0.005` | 两个网格共用步长；必须正好整除两段范围。 |
| `--max-samples` | 无 | 覆盖每个所选数据集 cap，但不超过内置 cap；主要用于 smoke。 |

汇总器参数：`--dataset` 筛选一个数据集；`--family` 可选 `both/absolute/percentage`；`--show-thresholds` 展开全部阈值；`--threshold` 按三位小数标签精确筛选一个阈值。

环境变量中，`CUDA_VISIBLE_DEVICES` 决定进程数和物理 GPU 映射；`env -u RANK -u WORLD_SIZE` 防止继承外部 rank；`env -u MASTER_PORT` 防止旧端口影响自动选择；`PYTHONPATH` 指向仓库；`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 禁止联网解析。

## 10. settings 和结果文件

启动器接受目录后首先以独占方式写 `settings.json`，后续不再修改。它记录公式、包含边界的判断规则、完整 41+51 阈值列表、超参、数据 SHA-256、checkpoint config SHA-256、Git commit/工作树、GPU 映射、自动端口和完整命令。模型加载失败时 settings 仍然保留。

典型目录：

```text
YYYYMMDD_HHMMSS_confidence_drop_rejection_all/
├── settings.json
├── experiment_manifest.json
├── dataset_results.jsonl
├── confidence_drop_results.md
├── eval.log
├── progress/<dataset>/...
├── tensorboard/
│   └── artifacts/step_0/<dataset>/...
└── observations/confidence_drop_rejection/<dataset>/
    ├── metrics.json
    ├── absolute_drop_thresholds.csv
    ├── percentage_drop_thresholds.csv
    ├── threshold_curves.png
    └── rank_stats/rank_<rank>.json
```

- `experiment_manifest.json`：运行状态和逐数据集完成记录；失败时保存 traceback。
- `dataset_results.jsonl`：每完成一个数据集立即 `flush+fsync` 追加，包含 baseline speculative metrics、父类累计 confidence 摘要、全部 absolute 和 percentage 阈值结果。
- `confidence_drop_results.md`：面向人工查阅的增量汇总；每个已完成数据集严格追加 `token_x_drop_abs` 和 `token_y_drop_pct` 两张表，包含 accepted/rejected 数量、内部占比、accepted FPR、rejection recall 和 flag rate。
- `metrics.json`：权威聚合结果，包含定义、全部分母/删失计数和 92 个阈值的所有指标。
- 两个 CSV：分别给出 41 个 absolute 和 51 个 percentage 阈值，便于直接绘图或表格分析。
- `threshold_curves.png`：precision、accepted share、recall、accepted FPR 随阈值变化的快速可视化。
- `rank_stats/`：各 rank 在归约前的本地充分统计量，用于审计；不是逐 token 日志。
- `tensorboard/`：保留父类已有指标，并添加各 family/threshold 的四种比例。

## 11. 监控和完整性检查

查看 manifest：

```bash
/data/home/wly/.conda/envs/dspark/bin/python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/experiment_manifest.json
```

查看 GSM8K 聚合进度：

```bash
watch -n 2 '/data/home/wly/.conda/envs/dspark/bin/python -m json.tool /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/progress/gsm8k/progress.json'
```

持续查看日志：

```bash
tail -f /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/eval.log
```

随数据集完成实时查看 Markdown 汇总：

```bash
tail -F /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/confidence_drop_results.md
```

检查九项状态：

```bash
/data/home/wly/.conda/envs/dspark/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["status"],p["completed_dataset_count"],[(d["name"],d["status"]) for d in p["datasets"]])' /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录>/experiment_manifest.json
```

完整全量实验应满足：manifest 为 `completed`、`completed_dataset_count=9`、九个数据集全部 completed；`confidence_drop_results.md` 中有九个 dataset 章节、十八张逐数据集阈值表和末尾两张宏平均表；每个数据集都有 `metrics.json`、两个阈值 CSV、曲线图和所有可见 rank 的审计文件；settings 中 threshold count 必须分别为 41 和 51。不能把首位置拒绝计入可判定 recall 分母，也不能把首拒绝之后的 draft positions 当作标签样本。

## 12. `confidence_drop_results.md` 表格字段解读

`confidence_drop_results.md` 是“置信度下降与拒绝关系”反向观测实验的可读汇总文件。每完成一个数据集，文件中会追加两张表：

- `token_x_drop_abs`：按绝对下降量筛选。对位置 `i`，先计算此前位置的条件接收置信度均值 `C_i_mean`，再计算 `D_abs = C_i_mean - C_i`；当 `D_abs >= x` 时，该 token 被阈值 `x` 标记。
- `token_y_drop_pct`：按相对下降比例筛选。计算 `D_pct = max(0, 1 - C_i / C_i_mean)`；当 `D_pct >= y` 时，该 token 被阈值 `y` 标记。`C_i_mean == 0` 时比例无定义，该位置不进入百分比口径的分母。

两张表只有第一列不同：绝对下降表的第一列是 `x`，相对下降表的第一列是 `y`；其余各列使用相同定义。下面记：

- `A`：该数据集在当前指标口径下，所有可判定且实际被验证接收的 token 数。
- `R`：所有可判定且处在本轮首个拒绝位置的 token 数。
- `a_t`：在阈值 `t` 下被标记、但最终实际接收的 token 数。
- `r_t`：在阈值 `t` 下被标记、且最终实际拒绝的 token 数。
- `f_t = a_t + r_t`：阈值 `t` 下被标记且可判定的 token 总数。

### 12.1 每一列的含义

| 列名 | 计算方式 | 含义 |
|---|---:|---|
| `x` / `y` | 当前绝对下降阈值 / 相对下降阈值 | 阈值越小，通常标记的 token 越多；阈值越大，筛选越严格。 |
| `accepted_count` | `a_t` | 被低置信度下降规则标记、但 verify 实际通过的 token 数。这些是把“会被拒绝”作为正类时的假阳性，即被误判的接收位置。 |
| `rejected_count` | `r_t` | 被规则标记、且 verify 确实在该位置首次拒绝并由 AR token 替换的 token 数，即真阳性。 |
| `flagged_evaluable_count` | `f_t = a_t + r_t` | 被当前阈值标记且 verify 结果可判定的 token 总数。首个拒绝位置之后的 token 不计入。 |
| `accepted_share` | `a_t / f_t` | 所有已标记 token 中，实际仍被接收的比例，也就是标记集合中的误判占比。 |
| `rejected_share / precision` | `r_t / f_t` | 所有已标记 token 中，实际被拒绝的比例；以“预测拒绝”为正类时，这就是 precision（精确率）。 |
| `accepted_FPR` | `a_t / A` | 所有可判定的实际接收 token 中，有多少被规则错误标记；这是针对接收位置的假阳性率。 |
| `rejection_recall` | `r_t / R` | 所有可判定的实际拒绝 token 中，有多少被当前阈值捕获；这是拒绝位置的 recall（召回率）。 |
| `flag_rate` | `f_t / (A + R)` | 所有可判定 token 中，被当前阈值标记的比例，反映规则的触发覆盖率。 |

只要 `f_t > 0`，就有 `accepted_share + rejected_share = 100%`。如果当前阈值没有标记任何可判定 token，即 `f_t = 0`，这两个“标记集合内部占比”没有定义，Markdown 中显示为 `-`；如果各自总体分母存在，`accepted_FPR`、`rejection_recall` 和 `flag_rate` 则为 `0%`。

### 12.2 可复算示例

假设某数据集共有 `A = 10000` 个可判定的实际接收 token、`R = 500` 个可判定的实际拒绝 token。在绝对下降阈值 `x = 0.100` 下，共标记出 200 个 token，其中 120 个最终被接收、80 个最终被拒绝，则表格行为：

| x | accepted_count | rejected_count | flagged_evaluable_count | accepted_share | rejected_share / precision | accepted_FPR | rejection_recall | flag_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.100 | 120 | 80 | 200 | 60.0000% | 40.0000% | 1.2000% | 16.0000% | 1.9048% |

各比例的计算如下：

- `accepted_share = 120 / 200 = 60%`：被标记的 token 中，六成其实能够接收，是标记集合中的误判。
- `rejected_share / precision = 80 / 200 = 40%`：被标记的 token 中，四成确实被拒绝，因此该阈值预测拒绝的精确率为 40%。
- `accepted_FPR = 120 / 10000 = 1.2%`：全部实际接收位置中，1.2% 被误判为低置信度下降位置。
- `rejection_recall = 80 / 500 = 16%`：全部实际拒绝位置中，当前阈值捕获了 16%。
- `flag_rate = 200 / (10000 + 500) = 1.9048%`：全部可判定位置中，大约 1.90% 触发了规则。

相对下降表的计算完全相同，只需把阈值换成 `y`，并使用百分比口径对应的 `A` 和 `R`。由于 `C_i_mean == 0` 的位置无法定义 `1 - C_i / C_i_mean`，百分比表的总体分母可能小于绝对下降表。

### 12.3 阅读这些指标时的注意事项

- `rejected_share / precision` 与 `rejection_recall` 不是同一个量：前者问“被标记的 token 有多少真的被拒绝”，后者问“所有被拒绝的 token 有多少被标记”。
- `accepted_share` 与 `accepted_FPR` 也不是同一个量：前者以“被标记 token 数”为分母，后者以“所有实际接收 token 数”为分母。因此，即使 `accepted_share` 较高，只要规则触发很少，`accepted_FPR` 仍可能很低。
- draft 序列的第 0 个 token 没有更早 token 可用于计算 `C_i_mean`，不进入上述阈值统计；若首个拒绝恰好发生在第 0 个位置，会单独计入 `unscorable_first_position_rejections`。
- 本轮首个拒绝位置之后的 draft token 无法判断其本来会被接收还是拒绝，属于删失位置，不会增加 `accepted_count` 或 `rejected_count`；已接收 EOS 之后的位置同样忽略。
- Markdown 文件把比例渲染为百分数便于阅读；对应 JSON/CSV 中通常保留 `[0, 1]` 范围内的小数值。例如 Markdown 的 `40.0000%` 在结构化结果中对应 `0.4`。

## 13. 跨数据集宏平均与旧结果回填

文件末尾的 `All-dataset macro average` 含两张表，分别对应 `token_x_drop_abs` 和 `token_y_drop_pct`。对固定阈值和固定列，先在每个数据集内部得到该列的值，再把各数据集值相加并除以数据集数量；这是每个数据集等权的算术平均，不是把所有数据集 token 合并后计算的微平均，因此样本或 token 更多的数据集不会占更大权重。count 列也按数据集求平均，所以可以是小数。

比例列如果某数据集出现 0/0，则该值是未定义而不是 0，宏平均只对有定义的数据集求均值。表内的 `(m/n)` 表示该比例有 `m` 个数据集参与平均、当前文档共含 `n` 个数据集；全部未定义时显示 `N/A (0/n)`。新实验每完成一个数据集都会重建末尾宏平均块，因此中途看到的是“截至当前已完成数据集”的宏平均，九项完成后才是最终九数据集平均。

对已有且不在运行中的结果目录，可直接利用 `dataset_results.jsonl` 中已保存的逐数据集逐阈值结果幂等补写或刷新两张宏平均表；不会重算模型、不会修改 JSON/CSV（单行命令）：

```bash
cd /data/home/wly/dLLM/DeepSpec && PYTHONPATH=/data/home/wly/dLLM/DeepSpec /data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/observations/confidence_drop_rejection/summarize_confidence_drop_observation.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/<时间戳目录> --refresh-markdown-macro
```

汇总器会拒绝改写 manifest 仍为 `running` 的目录，以免与 rank 0 同时更新 Markdown。重复执行上面命令只替换文件末尾带标记的宏平均块，不会重复追加 dataset 章节。
