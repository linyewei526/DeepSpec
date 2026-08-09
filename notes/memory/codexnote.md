时间戳（Git 状态复核）：2026-08-06 17:21:10 CST (UTC+08:00)

# 给新 Codex 会话的 DeepSpec / DSpark 对齐指南

## 使用目标

当用户要求“按 `codexnote.md` 对齐当前项目”时，先按本文的顺序完成阅读和只读核验，再继续用户的新任务。不要在对齐阶段重写已有长文、重跑 GPU 实验或修改代码。

信息优先级：当前源码和本地 checkpoint config 决定“代码实际如何运行”；论文 PDF 决定“论文声称了什么”；`notes/basis` 是已经对照两者的项目索引和解读，但在代码继续变化后应重新核对。

## 第 1 步：确认仓库和保护现有变更

1. 工作目录应为 `/data/home/wly/dLLM/DeepSpec`。
2. 首先执行只读检查：`git -C /data/home/wly/dLLM/DeepSpec status --short`、`git -C /data/home/wly/dLLM/DeepSpec rev-parse HEAD` 和 `git -C /data/home/wly/dLLM/DeepSpec rev-parse origin/main`。
3. 本会话已完成工作的基线提交是 `de552a1063140130d162fe102945642282e93d3a` (`de552a1`, subject: `update`)。Git 复核时 `HEAD=origin/main=de552a1` 且 ahead/behind 均为 0，说明之前的代码、文档、论文归档和 `runtime/` 工具均已提交，并与当前本地 `origin/main` 跟踪引用一致。这一只读对齐步骤不需要为了重新确认远端服务器而自动 fetch/pull。
4. 如果新会话的 `HEAD` 已经晚于 `de552a1`，先检查 `git -C /data/home/wly/dLLM/DeepSpec log --oneline de552a1..HEAD` 和对应 diff，再判断本文哪些状态描述已被后续提交更新；不得为了回到文档快照而回退新提交。
5. 检查仓库内是否新增 `AGENTS.md`；当前快照中 DeepSpec 仓库没有项目级 `AGENTS.md`。
6. 即使基线已提交，仍要将实时工作树差异视为用户正在使用的内容。不得执行 reset/checkout/clean，不得删除 untracked 文件，不得假设任何提交后差异可以丢弃。

截至本文 Git 复核时间：

- commit `de552a1` 包含了之前完成的全部项目修改；`notes/basis/`、`notes/memory/` 和 `runtime/` 均已跟踪；
- 根目录原 DSpark PDF 已以 100% rename 的形式提交到 `notes/basis/DSpark_Confidence-Scheduled Speculative Decoding.pdf`，不再存在待处理的 PDF 删除；
- 复核开始时唯一的提交后差异是 `notes/DSpark_prompt.md` 记录了用户的当前请求；它是需求/会话日志，不是技术事实的优先来源；
- 本次应用用户请求后，`notes/memory/quicknote.md` 和 `notes/memory/codexnote.md` 也会成为 `de552a1` 之后的文档差异，直到用户下次提交。

上述只是时间戳快照，新会话必须以实时 Git 状态为准。

## 第 2 步：快速获取项目地图

1. 先完整阅读 `/data/home/wly/dLLM/DeepSpec/notes/memory/quicknote.md`，获取当前进展和路径索引。
2. 阅读 `/data/home/wly/dLLM/DeepSpec/README.md`、`requirements.txt` 以及 `config/dspark/dspark_qwen3_8b.py`，了解仓库原生入口、依赖和 Qwen3-8B draft 配置。
3. 列出 `deepspec/modeling/dspark/`、`deepspec/eval/`、`runtime/` 和 `notes/basis/` 的当前文件，避免依赖记忆中的旧目录结构。

## 第 3 步：对齐 DFlash/DSpark 算法和代码

按以下顺序阅读，不要只看标题或摘要：

1. `/data/home/wly/dLLM/DeepSpec/notes/basis/DFlash_Block Diffusion for Flash Speculative Decoding.pdf`；
2. `/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Confidence-Scheduled Speculative Decoding.pdf`；
3. `/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_代码实现详解.md`。

阅读第 3 份文档时，按其中的“推荐代码阅读顺序”同步打开当前源码。至少应交叉核对：

- `deepspec/modeling/dspark/common.py`、`markov_head.py`、`loss.py`、`qwen3/modeling.py` 和 `qwen3/config.py`；
- `deepspec/eval/base_evaluator.py`、`deepspec/eval/dspark/draft_ops.py`、`deepspec/eval/dspark/evaluator.py` 和 `deepspec/eval/dspark/confidence_head.py`；
- 训练入口、trainer、target cache 和 dataset 代码，具体路径以 `DSpark_代码实现详解.md` 的调用链为索引。

这一步完成后，应能区分：DFlash 的 block diffusion 并行 draft，DSpark 的 Markov 轻量因果修正和独立 confidence head，target 的无损 speculative verification，以及论文生产 scheduler 与公开代码静态 threshold 之间的边界。不需要在对齐回复中重述这些技术内容。

## 第 4 步：对齐 Qwen3-8B 复现方式和当前结果

1. 完整阅读 `/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Qwen3-8B_推理复现指南.md`，尤其是第 1、7–10、12、14 和 16 章。
2. 阅读 `/data/home/wly/dLLM/DeepSpec/runtime/verify_models.py`、`smoke_test.py`、`run_experiment.py` 和 `summarize_results.py`，理解当前复现入口和产物。
3. 核对成功基线 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all/experiment_manifest.json`；它应显示 `status=completed`、`completed_dataset_count=9`。
4. 阅读同目录的 `dataset_results.jsonl`，并抽查至少一个 `tensorboard/artifacts/step_0/<dataset>/metrics.json` 和 `reliability_diagram.png`。除非排查故障，无需通读很长的 `eval.log`。
5. 如需只读汇总，可使用：`/data/home/wly/.conda/envs/dspark/bin/python /data/home/wly/dLLM/DeepSpec/runtime/summarize_results.py /data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all`。

当前成功基线的关键定位是：

- target 是 `Qwen/Qwen3-8B` base，不是 Instruct；本地路径为 `/data1/linyewei/models/Qwen3-8B`；
- draft 为 `/data1/linyewei/models/dspark_qwen3_8b_block7`；
- evaluator 实际读取 `/data/home/wly/dLLM/DeepSpec/eval_datasets/*.jsonl`；`/data1/linyewei/datasets/DSpark` 是额外数据的下载/存放位置，不要混淆两者；
- 成功运行使用 GPU `2,3`、两个 A100 80GB，`max_new_tokens=2048`、`temperature=1.0`、`confidence_threshold=0.0`、`seed=980406`、`enable_thinking=false`、SDPA 和 Gloo；
- 这是 speculative decoding/confidence 指标复现，不是九项 benchmark 最终 accuracy、pass@1 或 judge score 评测；
- 当前推理热路是原生 Hugging Face/PyTorch + SDPA、batch size 1，没有自动使用 vLLM、SGLang 或论文生产 serving engine；
- `reliability_diagram.png` 已同时包含 reliability curve 和 confidence 分布柱，当前代码不单独生成 `confidence_histogram.png`。

## 第 5 步：理解本会话的评测代码修改

这些修改已经位于 commit `de552a1`，因此不能再依赖空的 working-tree `git diff` 寻找它们。先用 `git -C /data/home/wly/dLLM/DeepSpec show --stat de552a1` 确认提交范围，必要时使用 `git -C /data/home/wly/dLLM/DeepSpec show de552a1 -- <path>` 逐文件阅读该提交引入的修改，再使用以下索引：

- `deepspec/eval/base_evaluator.py`：增加基于 rank JSON 文件汇聚的 tqdm，数据集 phase/manifest 更新，每完成一项就追加 `dataset_results.jsonl`，以及 Gloo 时 CPU tensor 归约。
- `deepspec/eval/dspark/evaluator.py`：将 DSpark 每个数据集的 spec、confidence、artifact 和逐项结果写入纳入阶段跟踪。
- `deepspec/eval/dspark/confidence_head.py`：使 confidence 统计 tensor 在 Gloo 下转到 CPU 后归约。
- `deepspec/utils/distributed.py` 和 `eval.py`：增加 distributed backend 和 timeout 参数，评测默认使用 Gloo，用来避免完成采样后 NCCL 归约卡住。
- `runtime/run_experiment.py`：在运行开始就创建时间戳目录和 manifest，并记录数据、模型、参数、环境、进度和结果。

这些修改已通过一次完整九数据集实验验证，并已提交到 `de552a1`。后续改动时必须保留这些功能，或明确说明为什么替换。

## 第 6 步：继续开发时的约束

- 不要改写成功基线目录或重用已有 run directory；每次实验使用 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/YYYYMMDD_HHMMSS_<label>` 新目录。
- 复现指南内的 shell 命令保持单物理行、使用绝对路径，不再引入 `DS_REPO` 等二次路径变量。
- 新下载数据存到 `/data1/linyewei/datasets/DSpark`，模型存到 `/data1/linyewei/models`，结果存到 `/data/home/wly/dLLM/DeepSpec-results`。
- 只有在用户要求真实运行时才启动耗时 GPU 实验；对齐、诊断、解读可以优先使用现有 manifest、JSONL、metrics 和日志。
- 如要做推理优化，先阅读 `DSpark_代码实现详解.md` 第 8–9 章和复现指南第 14 章，先 profile，再修改。保持 Markov 修正后 `draft_probs`、无损 rejection sampling、cache crop/update、EOS 和 target 输出分布等正确性不变。
- 论文的 Hardware-Aware Prefix Scheduler/生产引擎不能直接当作公开仓库已实现功能；声称已实现前必须找到当前代码证据。

## 第 7 步：修改后的最低验证

根据变更风险选择验证，不要为了“对齐”自动运行全量评测：

1. 文档变更：检查文件路径、Markdown 结构、命令是否单行，以及是否与当前代码/产物一致。
2. Python 变更：至少对受影响文件运行 `python -m py_compile`，并做针对性只读/小规模检查。
3. 模型加载或 checkpoint 变更：先运行 `runtime/verify_models.py`。
4. 解码、分布式、进度或产物变更：在用户授权下按复现指南运行独立时间戳 smoke test；需要强证据时再运行单数据集或全量实验。
5. 对比新旧解码时，同时检查输出分布正确性、`#propose`、`accept_len`、`verify_rate`、逐位 acceptance 和 confidence calibration，不能只看 wall-clock。

## 对齐完成标准

新 Codex 对齐完成后，应能确认以下事实，但回复用户时只需简洁汇报，不要复述所有技术细节：

- 已阅读哪些论文/文档和关键源码；
- 当前 target/draft、数据和结果路径；
- DFlash/DSpark 的推理调用链与公开代码/论文系统边界；
- 成功九项基线及其指标性质；
- 当前基线 commit、实时 Git 差异和不可破坏的用户工作树状态；
- 接下来应根据用户的新任务继续，而不是从零重做已完成的调研。

时间戳（增量对齐快照）：2026-08-07 18:08:45 CST (UTC+08:00)

## 第 8 步：对齐 2026-08-07 新增的四组观测实验

前文以 `de552a1` 的 baseline 复现为主体；不要删除或跳过前文，但也不要把它的 Git 快照误当成最新状态。此增量写入前，实时 `HEAD=origin/main=46a7c451fe70d04d576d7b955e33b75bd9133e38`，其间有 `fe1f90e`、`87cb79f`、`46a7c45` 三个后续提交。新会话应先执行第 1 步的实时 Git 检查，再阅读 `git log --oneline de552a1..HEAD` 和相关 diff；如果 HEAD 已更新，则以新提交和实时工作树为准。更新本增量前，工作树只有用户维护的 `notes/DSpark_prompt.md` 发生变化；追加本次交接后，`notes/memory/quicknote.md` 和本文件也会成为未提交差异，不得覆盖。

### 8.1 必读代码和文档

本节表格和结果状态是 `2026-08-07 18:08:45` 的历史快照。其中两组 Markov 实验当时的 operational probability 口径已被文末第 9 步的 diagnostic probability 口径取代；新会话必须以第 9 步、最新指南和实时源码为准，不得将本节的旧口径当作当前定义。

先完整阅读以下四份指南，再按各指南中的“代码隔离与调用链”章节检查同名代码子目录（两份反向预测指南的第 3 节是阈值定义，不是调用链），不要仅凭目录名推断指标：

| 实验 | 观测标量 | 代码目录 | 指南 |
|---|---|---|---|
| 条件置信度与纠错排名 | confidence head 给出的逐位置条件接收概率 | `observations/conditional_confidence/` | `notes/observations/DSpark_Qwen3-8B_条件置信度与纠错排名观测指南.md` |
| 置信度下降拒绝预测 | 同一 confidence-head 条件接收概率 | `observations/confidence_drop_rejection/` | `notes/observations/DSpark_Qwen3-8B_置信度下降拒绝预测观测指南.md` |
| Markov 草稿概率与纠错排名 | 实际提交 token 在完整 Markov 修正 draft 分布中的 `q_k[z_k]` | `observations/markov_draft_probability/` | `notes/observations/DSpark_Qwen3-8B_Markov修正草稿概率与纠错排名观测指南.md` |
| Markov 草稿概率下降拒绝预测 | 同一 `q_k[z_k]` | `observations/markov_probability_drop_rejection/` | `notes/observations/DSpark_Qwen3-8B_Markov修正草稿概率下降拒绝预测观测指南.md` |

两种标量不能混称：confidence head 的值是“prefix 已通过条件下的当前位置接收置信度”；`q_k[z_k]` 是 Markov 修正 logits 经实际 temperature softmax 后，所提交 token 的 draft 概率质量。二者都不是跨位置累积概率。纠错排名实验中的 `true_draft_rank` 则始终在完整 `q_k` 上计算 correction token 的真实 competition rank，类别为 `1..10,other`。

四组实验均通过独立 evaluator/launcher/summarizer 接入，只写自己的新时间戳结果目录；没有改写 baseline 的 `deepspec/` 解码代码。启动器在加载模型前写不可变 `settings.json`，使用端口探测和租约避免并行实验冲突。后续新增观测优先继续建立新的 `observations/<experiment>/` 子目录，不要为了复用而重构或改动已有实验和 baseline 热路。

### 8.2 当前结果与完成度快照

按以下顺序读取各目录的 `settings.json`、`experiment_manifest.json`、`dataset_results.jsonl` 和聚合 artifact；状态会继续变化，因此下述完成度只是本时间戳快照：

1. `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_105942_conditional_confidence_all/`：manifest 已 `completed`，9/9 数据集、9 行结果。权威逐数据集详情位于 `observations/conditional_confidence/<dataset>/metrics.json`。
2. `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_153909_confidence_drop_rejection_all/`：manifest 已 `completed`，9/9 数据集、9 行结果。根目录 `confidence_drop_results.md` 为人工查阅汇总，每个数据集包含 absolute/pct 两张阈值表；字段含义以对应指南第 12 节为准。
3. `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_173321_markov_draft_probability_all/`：旧 schema v1，manifest 为 `failed`（`KeyboardInterrupt`），只有 GSM8K 完成聚合并写入 1 行结果；即使 `progress/math500` 显示采样 500/500，也不能把 MATH-500 视为已完成数据集。该目录保留负 gap，GSM8K 的极端负 relative gap 正是后续修改原因，不能与 schema v2 的 gap 均值/CDF 混合。
4. `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_180318_markov_draft_probability_all/`：新 schema v2；本时间戳 manifest 为 `running`、尚无完成数据集，GSM8K 进度为 437/500。新会话必须重新读实时 manifest 和 JSONL，不要沿用这个数字。
5. `markov_probability_drop_rejection`：代码和指南已完成，但本时间戳 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/` 下尚无对应全量结果目录；不要误报为已运行或已完成。

原九数据集 baseline `20260806_115804_all` 仍是解码复现基线，以上观测结果不能替代任务 accuracy/pass@1/judge 或无观测端到端性能实验。

### 8.3 必须区分的 gap 口径

- `conditional_confidence` 保留原实验定义，包括有符号 gap 和负值区间；不要因 Markov 实验的后续修改而改动或重解释它。
- `markov_draft_probability` 自 schema v2 起，以 `signed_absolute_gap = accepted_mean - rejected_probability` 为过滤条件。`signed_absolute_gap < 0` 只增加 `negative_gap_excluded_events`，不进入 `paired_gap_events`、absolute/relative gap 的均值、PMF/CDF、CSV、图或概率型 TensorBoard 指标；接受/拒绝概率分布和 `true_draft_rank` 仍完整记录。
- schema v2 必须满足 `correction_events = first_position_correction_events + gap_candidate_events` 和 `gap_candidate_events = paired_gap_events + negative_gap_excluded_events`；纳入的 absolute/relative gap 都固定在 `[0,1]` 按 0.05 分箱。
- `confidence_drop_rejection` 和 `markov_probability_drop_rejection` 是反向预测实验：对位置 `i>0` 比较当前标量与同轮前缀位置均值，absolute 阈值为 0.05–0.25（步长 0.005，共 41 个），percentage 阈值为 0.05–0.30（步长 0.005，共 51 个）；首拒绝之后未实际验证的位置不进入 accepted/rejected 标签计数。表中 `accepted_share`、`rejected_share/precision`、`accepted_FPR`、`rejection_recall`、`flag_rate` 的分母严格按各自指南第 12 节解释。

## 增量对齐完成标准

新会话除了完成前文的 baseline/算法对齐，还应做到：

- 能从四份指南和四个代码子目录准确区分 confidence-head 概率与 Markov `q_k[z_k]`，以及条件分布实验与反向拒绝预测实验；
- 能报告两个已完成 confidence-head 全量目录、旧 Markov schema v1 失败目录、新 schema v2 的实时状态，以及 Markov 反向实验是否已有新结果；
- 能说明 schema v1/v2 gap 口径不可混用，并以 `settings.json` 的 schema、manifest 和 `dataset_results.jsonl` 判定真实完成度；
- 保留实时工作树和正在运行的实验，从用户当前任务继续，不重新实现、覆盖或无授权重跑已有工作。

时间戳（增量对齐快照）：2026-08-08 09:58:35 CST (UTC+08:00)

## 第 9 步：对齐 greedy 支持和 Markov diagnostic 概率

1. 当前所有正式入口均允许 `temperature=0.0`；底层 `deepspec/utils/sampling.py` 将 `0<=temperature<1e-5` 统一定义为 argmax/one-hot greedy，负数、NaN 和无穷值报错。`temperature=1.0` 的既有复现语义不变。
2. verifier 的 `proposal.draft_probs` 始终代表实际 operational proposal distribution，greedy 时为 one-hot；不得用普通 softmax 替换它。
3. 两组 Markov 观测通过 `observations/markov_diagnostic_draft.py` 额外保留本轮已计算的 corrected logits，观测量改为 `q_obs=softmax(markov_corrected_logits)`，不除以解码温度。Markov `true_draft_rank` 也按 diagnostic logits 排序。详细口径以两份 Markov 观测指南和实时源码为准。
4. 新口径 schema：`markov_draft_probability` 为 3，`markov_probability_drop_rejection` 为 2。旧 schema 使用 operational probability，不能与新概率、gap、rank 或阈值结果混合。
5. 可只读核验两个成功的单样本 greedy smoke：`/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260808_095658_markov_diagnostic_greedy_smoke_gsm8k/` 和 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260808_095737_markov_diagnostic_drop_greedy_smoke_gsm8k/`。新会话仍须先检查实时 Git 与结果 manifest。
