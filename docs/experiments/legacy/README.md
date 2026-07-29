# 前置工作产物归档（Qwen2.5-1.5B 线）

118 个文件 / 1.9 MiB。大文件（权重、optimizer 状态、中间 checkpoint、数据集缓存）
已按 ADR-0003 删除；这里保留的是**无法重建的一次性事实**。

## runs/ — 每个 run 的训练记录

每个子目录含（视该 run 实际产出）：

| 文件 | 用途 |
|---|---|
| `trainer_state.json` | **完整 loss / lr / grad_norm 曲线**（逐 log step），画图和讲训练过程都用它 |
| `trainer_state_checkpoint-*.json` | 最后一个 checkpoint 处的状态快照 |
| `all_results.json` / `train_results.json` / `eval_results.json` | 最终聚合指标 |
| `adapter_config.json` | LoRA rank / alpha / target_modules —— 讲"我当时怎么配 LoRA"的证据 |
| `config.json` | 模型结构快照 |

| 子目录 | 说明 |
|---|---|
| `sft_medical_85_15_lrfix_rerun` | 医疗:通用 85/15 SFT，修正 lr 后重跑到 10000 step（主 SFT run） |
| `sft_medical_85_15_v1` | 第一版 SFT |
| `sft_lr_verify_5000_5020` | 专门验证 resume 后 lr 调度是否正确的 20 step 小跑 |
| `dpo_zh_medical_v1` | 中文医疗 DPO |
| `grpo_v9_2x3090_600step` | 最终 GRPO run（双卡 600 step） |
| `grpo_from_sft_full_300step` | 从 SFT 全量 checkpoint 起的 GRPO 路线对比 |
| `grpo_v7_100step_eval100`, `grpo_v8_300step_lighteval`, `grpo_v2_300step`, `grpo_v4_batch8_retry` | 显存/batch/长度调参与重试过程 |
| `ablation_no_safety`, `ablation_no_triage`, `ablation_no_redflag`, `ablation_no_mention`, `ablation_no_actionability` | 5 组 reward 分项消融 |

## reports/ — 评测与对比报告

- `v9_score_report.md` / `v9_score_summary.json` / `v9_preds_sample20.jsonl`：最终 GRPO run 的规则评测结果与 20 条样例预测；
- `sft_dpo_grpo_training_comparison_report_20260328.txt`、`sft_grpo_vs_dpo_grpo_route_comparison_report_20260329.txt`：训练路线对比；
- `full_vs_grpo_v2_detailed_report.txt`：reward v1/v2 对比；
- `*_rule_score_report.md`：各批次规则打分报告；
- `medical_benchmark_*.md` / `medical_safety_stress_*.md`：评测集说明；
- `medical_training_stack_map_20260330.txt`：当时的训练栈梳理。

## manifests/ — 数据划分依据

- `sft_mixture_manifest.json`：85/15 混合的来源、抽样数、seed、train/val 划分；
- `dpo_mixture_manifest.json`、`rm_mixture_manifest.json`；
- `medical_benchmark_300_manifest.json`、`medical_safety_stress_*_manifest.json`：
  评测集的类别分布与分诊标签分布。

## 使用提醒

- 这些数字来自 **Qwen2.5-1.5B + 2×3090 + 规则奖励** 的实验，与 CA-OPD
  （Qwen3 + veRL/vLLM + teacher logprob）**不构成受控对比**，不能放进同一张结果表；
- 规则评测分数只反映特定表述模式的命中率，不是临床有效性证明。
