# ADR-0003: 前置工作迁移边界与磁盘保留策略

- 状态：已接受（Accepted，已执行）
- 日期：2026-07-29
- 影响范围：仓库内容边界、历史实验可复现性、磁盘预算

## 1. 背景

前置工作（Qwen2.5-1.5B 上的 数据处理 → SFT → DPO → GRPO 全流程，含 6 组 reward 消融）
产生了约 56 GiB 的产物，分布在 30 GiB 系统盘与 50 GiB 数据盘上，两块盘分别只剩
6.5 GiB / 7.9 GiB。而 Qwen3 主线预算约 40 GiB（opd 环境 ~15 + 模型 11.3 + 数据 ~3 + 产物 ~11）。

同时需要回答一个内容边界问题：前置工作应该以什么形态留在面试仓库里？

## 2. 决策

### 2.1 内容边界

- 主线仓库 `CA-OPD-MedicalGPT` 的 README 与 `src/` 只讲 CA-OPD（Qwen3 + veRL/vLLM）；
- 前置工作以三种形态保留，且明确标注"不进入 CA-OPD 主结果表"：
  1. `legacy/{training,data,eval}`：真实跑过的脚本（不改逻辑，保证可复跑）；
  2. `docs/experiments/legacy/`：小体积产物——**完整 loss 曲线**（`trainer_state.json`）、
     `all_results.json`、`adapter_config.json`（LoRA rank/target 证据）、评测报告、数据 manifest；
  3. `legacy/eval/datasets/`：前置工作自建的评测集（5 个 jsonl，508 KiB）。
- 被提升进主线的唯一前置资产：`data/behavior/medical_safety_stress_150.jsonl`
  （150 例医疗安全行为诊断集）。它对应数据方案里的"医疗安全边界"诊断集需求，
  是相对参考项目（只有选择题）的差异化资产。

### 2.2 保留 vs 删除的判据

一句话规则：**能从「脚本 + 固定 seed + 公开数据」重建的，删；记录了一次性事实的，留。**

| 类别 | 处置 | 理由 |
|---|---|---|
| `trainer_state.json` / `all_results.json` / 评测报告 / manifest | 留（1.9 MiB，118 个文件） | 一次性事实：真实训练曲线、真实评测数字、划分依据 |
| 前置工作最终 LoRA adapter（SFT-final、GRPO-v9-final） | 留（141 MiB，放数据盘 `ca-opd/legacy-adapters/`） | 可用于演示与溯源；仅保留 2 个终态，不保留中间 checkpoint |
| merge 后的完整权重（9 份） | 删（23.7 GiB） | `base + adapter → merge_peft_adapter.py` 几分钟可重建 |
| `optimizer.pt` 与中间 checkpoint | 删 | 恢复训练才需要；前置工作已结束 |
| HF datasets arrow 缓存 | 删（12 GiB） | 纯缓存，产物是构建出的 jsonl |
| 前置 SFT 源数据与 85/15 混合数据 | 删（4.3 GiB） | manifest 已归档，`legacy/data/*.py` + seed 42 可重建 |
| Qwen2.5-1.5B-Instruct 基座 | 删（2.9 GiB） | 主线已确定为 Qwen3；需要复跑 legacy 时重新下载 |
| RM/DPO 原始偏好数据 | 删（~0.4 GiB） | 该分支不在新方案范围内 |

### 2.3 后续防膨胀规则（写进训练配置）

1. 只存 LoRA adapter，不存 `optimizer.pt`（HF Trainer 用 `save_only_model=true`）；
   本仓库 `src/opd/loop.py` 的 `checkpoint.keep_last` 默认 2，并在保存后自动清理更旧的。
2. **不产出 merge 后的完整权重**；评测与 rollout 走 base + adapter。
3. 构建完 jsonl 立即清 datasets 缓存；`HF_HOME` 指向数据盘。
4. `outputs/`、`checkpoints/`、`*.safetensors`、`*.pt`、`data/**/*.jsonl` 已在 `.gitignore` 中。
5. 新 conda 环境建在数据盘，`pip install --no-cache-dir`。

## 3. 执行结果（实测）

| 位置 | 清理前可用 | 清理后可用 | 释放 |
|---|---|---|---|
| `/`（系统盘 30 GiB） | 6.5 GiB | **20 GiB** | ~13 GiB |
| `/root/autodl-tmp`（数据盘 50 GiB） | 7.9 GiB | **50 GiB**（已用 161 MiB） | ~43 GiB |

主要删除项及体积（删除前实测）：

```text
系统盘  9.2G  /root/ablation_merged_cache（4 份 merged 全量权重）
        3.8G  14 个 outputs-* 运行目录
        0.3G  RM/DPO 原始数据、生成的 GRPO 数据集
        70M   node 安装包 + 失效 git worktree
数据盘  12.0G MedicalGPT_offload/cache（datasets arrow 缓存）
        14.5G 5 份 merged 全量权重
        8.7G  13 个训练输出目录（含 5 组 reward 消融）
        4.3G  finetune + finetune_mix（前置 SFT 数据）
        2.9G  Qwen2.5-1.5B-Instruct
        1.5G  HF hub 数据集快照 + pip 缓存
```

保留：`docs/experiments/legacy/` 1.9 MiB、`legacy/eval/datasets/` 508 KiB、
`/root/autodl-tmp/ca-opd/legacy-adapters/` 141 MiB。

## 4. 一个未解决的遗留项：旧仓库 .git 3.2 GiB

`/root/MedicalGPT/.git` 仍占 3.2 GiB，其中 3.14 GiB 是 574 个 loose object。

排查过程与结论：

1. `git fsck --unreachable` 只报出 11 个 tree、0 个 commit → 没有可丢失的工作；
2. `git gc --prune=now` 被 OOM killer 杀死（`pack-objects died of signal 9`），
   原因是容器 2 GiB 内存上限下无法 repack 3.14 GiB；
3. 改用更轻的 `git prune --expire=now`，object 数从 670 降到 574，体积不变；
4. 追查发现这些 136 MB 级 blob（`.../checkpoint-*/optimizer.pt`）仍被
   `refs/codex/turn-diffs/*`（Codex 工具的 turn 快照引用）钩住，
   而不是被用户的 `refs/heads/backup/*` 分支钩住——后者经 `git rev-list --objects`
   验证不含 optimizer.pt。

决策：**不动**。这是旧仓库（上游 fork）的 git 内部状态，属于用户自己整理过的 git 工作流；
系统盘已有 20 GiB 可用，收益小而风险不为零。若日后需要回收，用户可执行：

```bash
cd /root/MedicalGPT
git for-each-ref --format='delete %(refname)' refs/codex | git update-ref --stdin
git prune --expire=now          # 避免 gc 的 repack，防止 2GiB 容器内 OOM
```

## 5. 回滚

删除动作不可逆，但全部被删内容都可重建：

| 内容 | 重建方式 |
|---|---|
| merged 全量权重 | `legacy/` 中的 base + 保留的 adapter，`merge_peft_adapter.py` |
| 前置 SFT/DPO 数据 | `legacy/data/download_medical_data.py` + `build_medical_dominant_sft.py --seed 42`（manifest 已归档，可核对数量） |
| Qwen2.5-1.5B-Instruct | HF / ModelScope 重新下载（约 2.9 GiB） |
| 中间 checkpoint | 无法重建（已按上表判定不需要） |
