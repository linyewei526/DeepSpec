时间戳：2026-08-06 17:03:51 CST (UTC+08:00)

# DeepSpec / DSpark 当前会话进展速记

## 当前里程碑

- 已阅读并对照 DFlash、DSpark 论文与 DeepSpec 仓库，完成算法、训练/推理调用链、代码组织和公开实现边界的梳理。
- 已核实 Qwen3-8B 实验 target 是 `Qwen/Qwen3-8B` base 版，本地 target/draft 分别为 `/data1/linyewei/models/Qwen3-8B` 和 `/data1/linyewei/models/dspark_qwen3_8b_block7`。
- 已整理 Conda 环境、模型/数据、smoke test、九数据集运行、结果目录与故障排查指南。
- 已为评测增加跨 rank 的 tqdm 样本进度、实验初始 manifest、数据集阶段状态、逐数据集追加结果，并将默认指标归约后端改为 CPU Gloo，解决样本全部完成后 NCCL 归约卡住的问题。
- 九个数据集的 Qwen3-8B DSpark 全量解码评测已成功完成；成功基线目录为 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all`。
- 已详细核对 `#propose`、`accept_len`、bonus/correction、`accept_rate@k`、confidence calibration 指标及 reliability diagram 的代码口径。

## 快速索引

- 论文原文：`/data/home/wly/dLLM/DeepSpec/notes/basis/DFlash_Block Diffusion for Flash Speculative Decoding.pdf` 和 `/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Confidence-Scheduled Speculative Decoding.pdf`
- 算法、代码组织、训练/推理调用链与优化切入点：`/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_代码实现详解.md`
- Qwen3-8B 环境与九项评测复现：`/data/home/wly/dLLM/DeepSpec/notes/basis/DSpark_Qwen3-8B_推理复现指南.md`
- 结果口径和图表解读：上述复现指南第 16 章。
- 运行辅助工具：`/data/home/wly/dLLM/DeepSpec/runtime/`，包含模型检查、smoke test、时间戳实验入口和结果汇总脚本。
- 成功基线的配置、进度、逐项结果、日志和 confidence artifacts：`/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all/`

## 代码变更范围

- `deepspec/eval/base_evaluator.py`：进度、数据集阶段、逐项结果和 Gloo 归约适配。
- `deepspec/eval/dspark/evaluator.py` 与 `deepspec/eval/dspark/confidence_head.py`：DSpark confidence 统计阶段、artifact 和 CPU 归约适配。
- `deepspec/utils/distributed.py` 与 `eval.py`：可配置 distributed backend/超时，评测默认 Gloo。
- `runtime/`：新增评测编排和检查脚本。

## 交接注意

- 当前工作树尚未提交，`notes/basis/`、`notes/memory/` 和 `runtime/` 尚处于 untracked 范围；提交前必须先审阅 `git status` 和 diff，不要回滚或覆盖现有用户变更。
- 已完成的基线报告是 speculative decoding 与 confidence 指标，当前 evaluator 不等价于九项 benchmark 的最终 accuracy/pass@1/judge 评分。
- 当前推理是 Hugging Face/PyTorch + SDPA、batch size 1 的原生实现，没有接入 vLLM/SGLang 等 serving engine；后续推理优化应以上述成功实验为基线。

