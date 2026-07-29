# 前置工作（Qwen2.5-1.5B 线）

这里是 CA-OPD 之前完成的一轮完整后训练复现：**数据处理 → SFT → DPO → GRPO（含 6 组 reward 消融）**，
模型 Qwen2.5-1.5B-Instruct，硬件 2×RTX 3090。

**它不属于 CA-OPD 主线，也不进入任何主结果表。** 保留它的原因有两个：

1. 它是 CA-OPD 的问题来源——正是在这条线上观察到「医疗方向增强」与「通用能力保持」
   之间的张力，以及规则奖励在开放式问诊上的局限，才引出用 teacher logprob
   做稠密监督、并把能力保持写成显式约束的想法；
2. 它提供了一个真实的对照叙述：SFT / DPO / GRPO / OPD 四种训练信号的差别，
   我可以用自己跑过的曲线来讲，而不是背概念。

## 目录

```text
legacy/
├── training/        真实跑过的训练脚本（未改逻辑）
│   ├── medical_grpo_training.py                     自定义规则奖励 GRPO
│   ├── run_medical_grpo_v9_2x3090.sh                最终 GRPO run（600 step，双卡）
│   ├── run_sft_medical_85_15_qwen2_5_1_5b_3090_24g.sh
│   └── run_dpo_zh_medical_qwen2_5_1_5b_3090_24g.sh
├── data/            数据下载与混合脚本（固定 seed，manifest 可重建）
│   ├── download_medical_data.py                     HuatuoGPT-220k / shibing624-medical
│   ├── build_medical_dominant_sft.py                85/15 医疗:通用混合
│   ├── build_dpo_zh_medical.py
│   └── generate_grpo_dataset_v2.py                  GRPO prompt 集生成
└── eval/            规则评测与数据集
    ├── score_v9.py                                  分诊/红旗/可行动性/不安全率打分
    ├── build_medical_benchmark_300.py               300 题基准构建
    ├── build_safety_stress_set_150.py              150 例安全压力集构建
    ├── score_unified_ablation_split.py
    └── datasets/                                    5 个自建评测集（508 KiB）
```

产物（loss 曲线、评测报告、manifest）在 `../docs/experiments/legacy/`，
两个终态 LoRA adapter 在数据盘 `/root/autodl-tmp/ca-opd/legacy-adapters/`。
中间 checkpoint、merge 后的完整权重、原始数据集已按 ADR-0003 的判据删除
（均可用上述脚本 + seed 42 重建）。

## 复跑须知

这条线依赖的是**旧环境**：torch 2.3.0+cu121 / transformers 4.49.0 / trl 0.15.2 / peft 0.18.1。
不要在 CA-OPD 的 `opd` 环境（transformers ≥ 4.51）里跑它——那会破坏这批结果的可复现性。
基座 `Qwen2.5-1.5B-Instruct` 已删除，复跑前需重新下载（约 2.9 GiB）。

## 被提升进主线的资产

`data/behavior/medical_safety_stress_150.jsonl`（150 例医疗安全行为诊断集）
是这条线唯一进入 CA-OPD 主线的数据资产，打分逻辑重写为
`src/eval/behavior/rule_scorer.py`（去掉了硬编码的历史基线，并补上澄清/编造/
自相矛盾三个探针，修掉了原打分器的否定盲区）。
