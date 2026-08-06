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
