时间戳（Git 与结果状态复核）：2026-08-09 11:31:13 CST (UTC+08:00)

# DeepSpec / DSpark 当前进展速记

## 当前里程碑

- 已完成 DFlash、DSpark 论文与 DeepSpec 源码的算法、训练/推理调用链、代码组织和公开实现边界对齐；详见 `notes/basis/DSpark_代码实现详解.md`。
- Qwen3-8B DSpark 九数据集 temperature=1.0 基线已完成，结果为 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/20260806_115804_all/`；指标是 speculative decoding/confidence 指标，不是 accuracy/pass@1/judge 评分。
- 评测已支持跨 rank 进度、manifest、逐数据集追加结果、CPU Gloo 归约和独立时间戳目录；现在还支持 `temperature=0.0` greedy，`0<=temperature<1e-5` 均为 argmax/one-hot。
- `runtime/run_experiment.py` 现会自动探测空闲 distributed 端口，并与四个观测启动器共享租约；baseline 和观测实验可并行启动，命令见 `notes/basis/DSpark_Qwen3-8B_推理复现指南.md`。
- 已完成四组隔离观测实验代码：`conditional_confidence/`、`confidence_drop_rejection/`、`markov_draft_probability/`、`markov_probability_drop_rejection/`；说明与命令见 `notes/observations/` 下四份指南。
- 两组 Markov 观测当前统一使用 diagnostic `softmax(markov_corrected_logits)`，不受解码温度缩放；verifier 仍使用 operational `draft_probs`，Markov `true_draft_rank` 按 diagnostic logits 计算。隔离实现见 `observations/markov_diagnostic_draft.py`。

## 结果快速定位

- confidence-head 两组全量结果：`20260807_105942_conditional_confidence_all/` 和 `20260807_153909_confidence_drop_rejection_all/`。
- temperature=1.0 的新 diagnostic Markov 全量结果：`20260808_100625_markov_draft_probability_all/` 和 `20260808_105230_markov_probability_drop_rejection_all/`，均已完成 9/9。
- temperature=0.0 greedy baseline：`20260808_105331_all/`，已完成 9/9；greedy diagnostic Markov 概率/排名：`20260808_115103_markov_draft_probability_all/`，已完成 9/9。
- greedy diagnostic Markov 下降/拒绝预测：`20260809_105530_markov_probability_drop_rejection_all/`；本次复核时 manifest 为 `running`、完成 1/9，后续必须重读实时 manifest。
- 旧 Markov schema 结果使用 operational probability，不得与当前 `markov_draft_probability` schema 3 或 `markov_probability_drop_rejection` schema 2 混合。

## 快速索引与交接注意

- 论文原文：`notes/basis/DFlash_Block Diffusion for Flash Speculative Decoding.pdf` 和 `notes/basis/DSpark_Confidence-Scheduled Speculative Decoding.pdf`。
- 环境、checkpoint、九项复现、greedy 和自动端口命令：`notes/basis/DSpark_Qwen3-8B_推理复现指南.md`。
- 运行辅助工具：`runtime/`；四组观测代码：`observations/`；结果统一位于 `/data/home/wly/dLLM/DeepSpec-results/qwen3_8b/`。
- 当前 target/draft 为 `/data1/linyewei/models/Qwen3-8B` 和 `/data1/linyewei/models/dspark_qwen3_8b_block7`；评测数据读取 `eval_datasets/*.jsonl`。
- 本次文档修改前 `HEAD=origin/main=409df32a441285223457b054a9df14c42baa71cd`且工作树干净；修改后两份 memory 文档会成为未提交差异。新会话必须以实时 `git status`/`HEAD`、settings 和 manifest 为准，不得覆盖用户变更或重用旧结果目录。
