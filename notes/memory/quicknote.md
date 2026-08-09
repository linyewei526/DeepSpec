时间戳（Git 状态复核）：2026-08-06 17:21:10 CST (UTC+08:00)

# DeepSpec / DSpark 当前会话进展速记

## 当前里程碑

- 已阅读并对照 DFlash、DSpark 论文与 DeepSpec 仓库，完成算法、训练/推理调用链、代码组织和公开实现边界的梳理。
- 已核实 Qwen3-8B 实验 target 是 `Qwen/Qwen3-8B` base 版，本地 target/draft 分别为 `/data1/linyewei/models/Qwen3-8B` 和 `/data1/linyewei/models/dspark_qwen3_8b_block7`。
- 已整理 Conda 环境、模型/数据、smoke test、九数据集运行、结果目录与故障排查指南。
- 已为评测增加跨 rank 的 tqdm 样本进度、实验初始 manifest、数据集阶段状态、逐数据集追加结果，并将默认指标归约后端改为 CPU Gloo，解决样本全部完成后 NCCL 归约卡住的问题。
- 九个数据集的 Qwen3-8B DSpark 全量解码评测已成功完成；成功基线目录为 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all`。
- 已详细核对 `#propose`、`accept_len`、bonus/correction、`accept_rate@k`、confidence calibration 指标及 reliability diagram 的代码口径。
- 上述代码、文档、论文归档和 `runtime/` 工具已提交到 Git commit `de552a1063140130d162fe102945642282e93d3a` (`de552a1`, subject: `update`)。

## 快速索引

- 论文原文：`/data/home/wly/dLLM/DeepSpec/notes/basis/DFlash_Block Diffusion for Flash Speculative Decoding.pdf` 和 `/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Confidence-Scheduled Speculative Decoding.pdf`
- 算法、代码组织、训练/推理调用链与优化切入点：`/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_代码实现详解.md`
- Qwen3-8B 环境与九项评测复现：`/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Qwen3-8B_推理复现指南.md`
- 结果口径和图表解读：上述复现指南第 16 章。
- 运行辅助工具：`/data/home/wly/dLLM/DeepSpec/runtime/`，包含模型检查、smoke test、时间戳实验入口和结果汇总脚本。
- 成功基线的配置、进度、逐项结果、日志和 confidence artifacts：`/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all/`

## 已提交的代码变更范围

- `deepspec/eval/base_evaluator.py`：进度、数据集阶段、逐项结果和 Gloo 归约适配。
- `deepspec/eval/dspark/evaluator.py` 与 `deepspec/eval/dspark/confidence_head.py`：DSpark confidence 统计阶段、artifact 和 CPU 归约适配。
- `deepspec/utils/distributed.py` 与 `eval.py`：可配置 distributed backend/超时，评测默认 Gloo。
- `runtime/`：新增评测编排和检查脚本。

## 交接注意

- Git 复核时 `HEAD` 和 `origin/main` 均为 `de552a1`，本会话此前完成的项目工作均已纳入该提交。根目录原 DSpark PDF 的变化已作为 100% rename 提交到 `notes/basis/`，不再是待处理删除。
- 复核当时唯一的提交后差异是 `notes/DSpark_prompt.md` 新记录了当前用户请求；本次对 `quicknote.md`/`codexnote.md` 的更新也会在下次提交前显示为新差异。后续始终以实时 `git status` 为准，不要覆盖任何提交后用户变更。
- 已完成的基线报告是 speculative decoding 与 confidence 指标，当前 evaluator 不等价于九项 benchmark 的最终 accuracy/pass@1/judge 评分。
- 当前推理是 Hugging Face/PyTorch + SDPA、batch size 1 的原生实现，没有接入 vLLM/SGLang 等 serving engine；后续推理优化应以上述成功实验为基线。

时间戳（增量交接）：2026-08-07 18:08:45 CST (UTC+08:00)

## 2026-08-07 新增观测实验

- 已在 `observations/` 下新增四个相互隔离的子实验：`conditional_confidence/`、`confidence_drop_rejection/`、`markov_draft_probability/`、`markov_probability_drop_rejection/`；对应说明和单行运行命令见 `notes/observations/` 下四份 Qwen3-8B 观测指南。
- confidence-head 条件置信度/纠错排名全量实验已完成：`/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_105942_conditional_confidence_all/`。
- confidence-head 置信度下降/拒绝预测全量实验已完成：`/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260807_153909_confidence_drop_rejection_all/`；根目录 `confidence_drop_results.md` 是逐数据集追加的两张阈值表。
- Markov 草稿概率实验旧 schema v1 目录 `20260807_173321_markov_draft_probability_all` 已失败终止，仅 GSM8K 是已完成聚合结果；其负 gap 口径已废止。当前 schema v2 会排除 `signed_absolute_gap < 0` 的事件并保留审计计数，代码和说明见 `observations/markov_draft_probability/` 及对应指南；`20260807_180318_markov_draft_probability_all` 在本时间戳仍在运行，后续以实时 manifest 为准。
- Markov 草稿概率下降/拒绝预测实验的代码和指南已完成，但本时间戳尚无全量结果目录。以上观测代码最初提交于 `87cb79f`，Markov 负 gap 过滤提交于当前 `46a7c45`；更新本交接前 `HEAD=origin/main=46a7c45`，工作树仅有用户维护的 `notes/DSpark_prompt.md` 差异，追加本记录后还会包含两份 memory 文档差异。

时间戳（增量交接）：2026-08-08 09:58:35 CST (UTC+08:00)

## Greedy 温度与 Markov diagnostic 概率

- `eval.py`、`runtime/run_experiment.py` 和四个观测启动器现均支持 `temperature=0.0`；`0<=temperature<1e-5` 为精确 greedy，负数及非有限温度会报错，settings 记录 `temperature_mode`。
- 两组 Markov 观测已将概率定义改为温度无关的 `softmax(markov_corrected_logits)`，并与 verifier 使用的 operational `draft_probs` 分离；Markov 概率/排名为 schema 3，Markov 下降/拒绝预测为 schema 2。共用隔离实现见 `observations/markov_diagnostic_draft.py`，完整定义见两份 Markov 观测指南。
- 两个单样本 greedy 集成 smoke 已完成：`20260808_095658_markov_diagnostic_greedy_smoke_gsm8k` 和 `20260808_095737_markov_diagnostic_drop_greedy_smoke_gsm8k`。旧 schema 结果保持不变，不能与新 diagnostic 口径混合。
