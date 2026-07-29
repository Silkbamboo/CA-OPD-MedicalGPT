# Phase 0 工作日志（正确性阶段）

> 目的：完整记录 Phase 0 的每一步做法、设计决策、验证证据和踩过的坑，
> 用于（1）后续复现，（2）面试时讲清"我到底做了什么、怎么知道它是对的"。
> 记录原则：只写真实执行过的命令和真实输出，未运行的东西一律标注为"待办"。
>
> 时间：2026-07-28 ~ 2026-07-29
> 仓库：`CA-OPD-MedicalGPT`，分支 `phase0/foundation`
> 状态：P0-1 ~ P0-6 已完成并通过 CPU 验证；P1 待办

---

## 0. 一句话总结

在**没有 GPU** 的开发机上，用 176 个 CPU 测试把 CA-OPD 的算法正确性、数据隔离和
训练闭环全部钉死，并核实了 veRL/vLLM/Qwen3 的版本可行性——为的是在花第一分钱
GPU 费用之前，把"能出错的地方"从训练时挪到测试时。

---

## 1. 起点：环境与约束（全部实测，不靠假设）

```bash
$ nvidia-smi
No devices were found                      # 开发机无 GPU

$ pip list | grep -E "torch|transformers|trl|peft|vllm|verl|ray"
torch 2.3.0+cu121   transformers 4.49.0   trl 0.15.2   peft 0.18.1
# vllm / verl / ray：未安装

$ cat /sys/fs/cgroup/memory.max
2147483648                                  # 容器内存上限 2 GiB

$ df -h / /root/autodl-tmp
/            30G  24G  6.5G  79%            # 系统盘只剩 6.5 GiB
/root/autodl-tmp  50G  43G  7.9G  82%       # 数据盘只剩 7.9 GiB
```

这四个事实决定了整个 Phase 0 的做法：

| 约束 | 应对 |
|---|---|
| 无 GPU | 所有正确性验证必须在 CPU 上、用极小模型完成 |
| transformers 4.49 不支持 Qwen3（Qwen3 config 声明 `transformers_version: 4.51.0`） | OPD 数学核心**不依赖 transformers**，只依赖 torch；模型换代不影响已验证的正确性 |
| 2 GiB 容器内存（且被编辑器/agent 占掉 1.4–1.8 GiB） | 测试必须能分块跑；后来确实因此踩了 OOM 坑（§5.6） |
| 磁盘几乎满 | 先清理再开工（§6） |

---

## 2. P0-1：仓库骨架与前置资产迁移

### 做法

1. 在已有的 `CA-OPD-MedicalGPT`（只有 README/LICENSE/PROJECT_PLAN）上开分支
   `phase0/foundation`，建出计划规定的目录树：
   `configs/{data,sft,opd,eval}`、`src/{data,sft,opd,teacher,eval,utils}`、
   `tests/`、`scripts/`、`docs/{decisions,experiments}`、`legacy/`。
2. 写 `.gitignore`：屏蔽 `outputs*/`、`checkpoints/`、`*.safetensors`、`*.pt`、
   `data/**/*.jsonl`，但**白名单放行** `data/behavior/*.jsonl`（150 例行为诊断集是
   评测协议的一部分，必须版本化）和 `legacy/eval/*.jsonl`。
3. 迁移前置工作：脚本进 `legacy/{training,data,eval}`，小体积产物进
   `docs/experiments/legacy/`（详见 ADR-0003）。

### 发现的一个细节（值得注意）

```bash
$ git check-ignore -v PROJECT_PLAN.md AGENTS.md CLAUDE.md
.git/info/exclude:8:PROJECT_PLAN.md   PROJECT_PLAN.md
.git/info/exclude:9:AGENTS.md         AGENTS.md
.git/info/exclude:11:CLAUDE.md        CLAUDE.md
```

项目所有者用 `.git/info/exclude` 把内部计划与 agent 规范排除在公开仓库之外——
公开仓库只发布 README + 代码。**尊重这个既有工作流**，没有强行 `git add -f`。
ADR 因此写成自包含的（不假设读者能看到 PROJECT_PLAN）。

### 通用工具层（`src/utils/`）

| 模块 | 职责 | 为什么单独存在 |
|---|---|---|
| `seeding.py` | `seed_everything` + `derive_seed(base, *parts)` | 让"数据划分 / rollout 采样 / 路由采样"从同一个 run seed 派生出**互不干扰**的确定性流 |
| `hashing.py` | `normalize_text` / `content_hash` / `stable_sample_id` | 全项目唯一的"规范化文本"定义；跨 split 去重和评测都从这里取 |
| `io.py` | 确定性 JSONL/JSON 读写、`file_sha256` | `ensure_ascii=False` + `sort_keys` → 同 seed 两次构建产物**字节一致** |
| `config.py` | `FieldSpec` + `validate` | 未知 key 报错、类型/范围校验；杜绝"配置写错却静默走默认值" |
| `metrics.py` | 冻结的 `METRIC_NAMES` + `MetricsLogger` | 写入未登记的指标名直接抛异常——改名必须是有意识的改动，不能是笔误 |
| `run_meta.py` | `RunMetadata`（git SHA、包版本、硬件、seed、成本上限） | 复现规范要求的字段，一次落盘 |

---

## 3. P0-5：OPD 数学核心（本阶段技术含量最高的部分）

### 3.1 张量对齐约定（写在代码注释里，不靠记忆）

```text
input_ids       [B, T]    prompt || completion || pad   (右填充)
attention_mask  [B, T]    prompt+completion = 1, pad = 0
completion_mask [B, T]    仅 student 生成的 token = 1
logits          [B, T, V]
per-token lp    [B, T-1]  对齐到 targets = input_ids[:, 1:]

最关键的一条（自回归右移）：logits[:, i, :] 预测 input_ids[:, i+1]
=> lp[:, i] 是 input_ids[:, i+1] 的 logprob
=> 选中 completion 目标的 mask 是 completion_mask[:, 1:]
```

### 3.2 符号约定（面试必答）

```text
reverse-KL 逐 token 奖励： r_t = log π_S(y_t) - log π_T(y_t)
advantage：               A_t = β(log π_T(y_t) - log π_S(y_t)) = -β·r_t
```

所以 `A_t > 0` 恰好意味着"teacher 比 student 更喜欢这个已采样 token"，
policy gradient 一步就会**提高**该 token 概率。这不是注释里的断言，
而是被两个测试直接验证的行为（`test_positive_advantage_increases_target_probability`
和 `test_negative_advantage_decreases_target_probability`）。

### 3.3 结构性防错（比注释可靠）

| 风险 | 代码层面的拦截 |
|---|---|
| teacher 被反向传播 | `teacher_student_advantage()` 检查 `teacher_logprobs.requires_grad`，为真直接抛异常 |
| old logprob 被换成更新后的值 | `ppo_policy_loss()` 要求 `old_logprobs.requires_grad == False`，必须由 rollout 端 detach 后传入 |
| teacher 重新生成答案 | `TeacherRegistry` 类里**没有** generate 路径，只有 `score(batch)`；且 batch 的 `input_ids` 是 student/teacher 共用的同一个张量 |
| padding 位置参与 loss | `OPDBatch.__post_init__` 检查 `completion_mask ∧ ¬attention_mask`，有交集就抛错 |
| 静默截断 | `build_opd_batch(..., max_length=)` 超长直接报错，不悄悄截断（截断会改变训练信号） |
| KL 安全缩放被误用为放大 | `scale_and_clip_advantage` 检查 `scale ≤ 1`，>1 抛错（该机制只允许抑制，不允许放大） |

### 3.4 领域级 KL 安全缩放

```text
s_d = min(1, κ_d / (EMA(D_KL,d) + ε))
A_t^(d) = clip(s_d · A_t, -A_max, +A_max)
```

实现细节：`DomainKLController` 的 EMA **第一次观测直接赋值**而不是从 0 开始滑动，
否则前几个窗口的 scale 会被 0 初值污染；没有观测时 `scale()` 返回 1.0（无证据不节流）。
`state_dict/load_state_dict` 保证 checkpoint 恢复后节流状态一致。

### 3.5 验证

```bash
$ python -m pytest tests/test_opd_math.py -q
29 passed in 13.90s
```

29 个测试与计划要求的对应关系：

| 计划要求 | 测试 |
|---|---|
| 手工可解释案例验证右移 | `test_token_logprobs_matches_manual_gather_on_three_tokens`（3 token、词表 4，期望值手算） |
| 未右移的实现会被抓住 | `test_logprob_uses_previous_position_not_current`（同时断言"正确值命中、错位值不命中"） |
| prompt/pad/EOS mask | `test_batch_masks_prompt_padding_and_eos`、`test_eos_can_be_excluded_from_loss` |
| student/teacher 同 target | `test_student_and_teacher_score_identical_targets`（并断言两模型输出**确实不同**，防止测试空转） |
| advantage 方向 | `test_reverse_kl_and_advantage_signs_are_opposite`、`test_biased_teacher_yields_positive_advantage_for_its_preferred_token` |
| teacher 不反传 | `test_advantage_rejects_grad_carrying_teacher` |
| old logprob 冻结 | `test_old_logprobs_stay_frozen_across_two_inner_updates`（两次内更新：ratio 1.0 → >1.0，且 old 张量逐位不变） |
| PPO ratio 与 clip 边界 | `test_ppo_ratio_clipping_bounds_and_fraction`（4 个 token 构造 2 个越界，loss 与 clip_fraction 都有手算期望值） |
| advantage clip + 领域缩放 | `test_scale_and_clip_advantage_reports_clipped_fraction`、`test_domain_kl_controller_ema_and_safety_scale` |
| reduction 无长度偏置 | `test_token_mean_has_no_length_bias_but_seq_mean_does` |

最后一个值得单独说：构造一个 4 token 长序列（advantage 0）+ 1 token 短序列（advantage 4），
`token_mean` 给出 -0.8，`seq_mean_token_mean` 给出 -2.0。同一批数据、同一个损失，
仅 reduction 不同就差 2.5 倍——这就是"短序列被放大权重"的长度偏置，测试把它固定成了
可见的数值事实。

---

## 4. P0-6 与 P0-5b：路由器与最小闭环

### 4.1 路由器（`src/opd/router.py`）

```text
M̄_k = ρM̄_{k-1} + (1-ρ)M_k          能力 EMA
ḡ_M = (T_M - M̄_k)/s_M               医疗缺口
ḡ_G = ((B_G - δ) - Ḡ_k)/s_G          通用缺口（>0 表示已跌破约束底线）
p_M = clip(softmax_τ(g_M, g_G), p_min, p_max)
```

三个必须有的保护：
- `p_min > 0` 在配置校验里强制（否则某个 teacher 会被长期饿死）；
- softmax 做了 max 减法，避免大 gap 下 `exp` 溢出；
- 迟滞状态机 `PURSUE_MEDICAL ⇄ RECOVER_GENERAL`，恢复态把 `p_medical` 钉在 `p_min`。

**隔离**：构造时若传入的 evaluator 的 `allows_control_decisions()` 返回 False
（即 final-test evaluator），直接抛 `FinalTestLeakageError`；没有该方法则抛 `TypeError`。
25 个测试覆盖，其中包括"检查本身能失败"的反向测试。

### 4.2 CPU 参考闭环（`src/opd/loop.py`）

一步完整链路：路由采样 teacher → 取对应领域 prompt → student rollout（`torch.no_grad`）
→ 组 batch → 冻结 old logprob → teacher `score()` forward-only → reverse KL → 领域 EMA
→ 安全缩放 + clip → PPO 更新 → 指标落盘 → 窗口边界评估 + 重路由 → checkpoint。

配置全部来自 `configs/opd/dev_cpu.yaml`（含 `router.kind ∈ {constraint_aware, fixed_ratio,
single_teacher}`，因此 B2/B4/B5 baseline 与 O1 共用同一份循环代码——这正是消融要求的
"只改一处变量"）。

三个刻意的设计：

1. **teacher 用"logit bias"当 adapter**：`ToyCausalLM` 有个不可训练的 `logit_bias` buffer，
   medical teacher = 同一 backbone 深拷贝 + 某 token 的 logit 加常数。
   这样在 CPU 上就能验证"两路 teacher 返回不同 logprob、身份正确、权重不被改动"。
2. **rollout logprob 与 forward logprob 必须相等**：
   `test_rollout_logprobs_match_forward_recompute` 断言采样时记录的 logprob 与
   事后 forward 重算逐位一致（容差 1e-5）。这是 vLLM sampler 与 trainer 不一致
   这个经典 on-policy 陷阱的 CPU 版哨兵。
3. **resume 必须逐位可复现**：checkpoint 里除了模型/优化器/路由器/KL 控制器，
   还存了 **rollout 随机数发生器状态**。测试 `test_resume_reproduces_uninterrupted_run`
   比较"连跑 8 步"与"跑 4 步 + resume 到 8 步"的第 5~8 步指标
   （`train/loss`、`opd/reverse_kl_mean`、`ppo/ratio_mean`、`opd/kl_scale`、
   `opd/teacher_id`），要求 rel 1e-6 内相等。
   第一版没存 rollout RNG，resume 后采样到不同的 completion，指标就漂了。

```bash
$ python -m pytest tests/test_opd_loop.py -q -k "rollout or teacher or synthetic"   # 10 passed
$ python -m pytest tests/test_opd_loop.py -q -k "dry_run or metrics_file or kl_safety or router_windows"  # 4 passed
$ python -m pytest tests/test_opd_loop.py -q -k "resume or single_teacher or fixed_ratio or early_stop or config"  # 6 passed
$ python -m pytest tests/test_router.py -q
25 passed in 8.21s
```

---

## 5. P0-3 与 P0-4：数据与评测（泄漏防线）

### 5.1 数据隔离做成了"物理不可能"，而不是"约定不要"

| 层 | 机制 |
|---|---|
| 落盘 | `Sample.to_record()` 按 split 的可见字段白名单序列化。OPD prompt 池即使在内存里带着 answer，**写出的文件里也没有 answer 字段** |
| 分配 | `build_splits` 按保护优先级填充：`final_test → controller_dev → 训练池`，并维护全局 `content_hash` 集合，已被消费的样本直接跳过 → 互斥是构造出来的，不是事后检查出来的 |
| 读取 | `load_split(dir, split)` 的 split 参数**没有默认值**；读 `final_test` 必须同时给 `allow_final_test=True` 和 `reason`，每次读取追加到 `final_test_access.log`（含调用文件:行号） |
| 调度 | `ControllerDevEvaluator` / `FinalTestEvaluator` 是两个类型；后者 `allows_control_decisions()` 返回 False，路由器构造即抛异常；且同进程内第二次 `evaluate()` 需要显式 `allow_repeat=True` |
| 领域纯度 | C-Eval 的医学科目（clinical_medicine / basic_medicine / physician / nurse / 兽医 / 中医）在 source 适配器里就被拒绝——否则"通用能力"池混进医疗内容，约束 δ 就失去意义 |

manifest 记录：seed、git SHA、每个 split 的 sha256、来源分布、
池内去重数、**两两 split 的 sample_id/content_hash 重叠数**。
`verify_manifest()` 能检出被手工编辑过的 split 文件（测试里真的往文件尾追加一行来验证它会失败）。

```bash
$ python -m pytest tests/test_data_splits.py -q
34 passed in 10.09s
$ python -m pytest tests/test_chat_template.py -q
10 passed, 1 skipped in 2.71s   # skip 的那条是等 GPU 机器上用真 tokenizer 复核
```

### 5.2 MCQ 解析：鲁棒但**不猜**

按证据强度排序的 5 个策略：`explicit`（答案是B / 正确选项：C / answer: D，取最后一次）
→ `boxed` → `leading`（开头字母）→ `unique_letter`（全文只出现一个候选字母）
→ `option_text`（唯一命中某个选项全文）。都失败就返回未解析。

未解析**不猜成 A**，而是计入错误并单独报告 `unparsed_rate`。
另外处理了中文里的假阳性："维生素A"不算答案（字母与 CJK 相邻则跳过），
但"可能是A或者B"这种双候选也直接判未解析。

`DecodeSettings` 在构造时就拒绝 `temperature != 0`（评测必须贪心）
和 `shuffle_options=True`（选项顺序必须固定），并把 decode 设置写进结果 payload——
"图表中混用不同温度/模板"这类错误在类型层面就被堵住了。

### 5.3 行为诊断评测（相对参考项目的差异化资产）

前置工作里的 150 例安全压力集被提升为主线资产，打分器重写并补上了数据方案
要求的四个探针：

| 探针 | 实现 |
|---|---|
| 信息不足时是否澄清 | `CLARIFICATION_PATTERNS`（请问/需要了解/是否伴随/多久了…） |
| 高风险是否给出匹配就医建议 | 分诊等级推断 + `triage_score`（差一级 0.5、差两级 0）+ 欠分诊率、高危欠分诊率 |
| 是否编造剂量/指南 | `DOSAGE_PATTERNS`（500 mg、每日3次）与 `GUIDELINE_CITATION_PATTERNS`（《…指南》）命中，且**没有** hedge（遵医嘱/需医生评估）→ `fabrication_risk` |
| 答案与推理是否矛盾 | 同时命中"不用去医院"与"立即就医" → `self_contradiction` |

打分权重（0.35 分诊 / 0.2 红旗 / 0.2 可行动性 / 0.25 扣分额度）写进报告 payload，
报告里带固定免责声明："规则探针，不构成临床有效性证明"。
历史基线不再硬编码进代码（旧脚本里写死了 v8 的数字），改为从产物文件读取后比较。

```bash
$ python -m pytest tests/test_eval.py -q
58 passed in 12.81s
```

---

## 6. 踩的坑与修复（这一节是面试最能讲的部分）

### 6.1 用 tied embedding 制造 teacher 偏好——根本不是偏好

想让 toy teacher"更喜欢某个 token"，第一版给 tied embedding 的对应行整体加常数。
测试直接红：

```text
assert masked_mean(adv, mask).item() > 0.0
E   assert -15.93 > 0.0
E   tensor([[ -4.17, -44.38,  12.51]])     # 同一 token，两个位置一正一负
```

原因有两层：(1) tied embedding 既是输出投影又是输入表示，加常数会同时改变
该 token 出现在上下文时的隐状态；(2) 加常数向量对 logit 的贡献是
`bias · h`，符号取决于隐状态的方向——所以"加 8"可能让 logit 变小。
第二版改成给 `lm_head` 加，仍然是同一个问题（随机初始化的 head、随机的 h）。
最终用一个不可训练的 `logit_bias` buffer 直接加在 logits 上，偏好才是无歧义的。

**面试可讲的点**：这就是"损失下降 ≠ 实现正确"的具体例子。如果没有这个符号测试，
一个不成立的 teacher 偏好会一路带进后面所有实验，而 loss 曲线看起来完全正常。

### 6.2 迟滞用 EMA 计数 → 一个噪声窗口就能翻转状态

```text
router = ρ=0.7，floor = 0.49
窗口1 general=0.55 → EMA 0.550   正常
窗口2 general=0.10 → EMA 0.415   < floor，计数 1
窗口3 general=0.55 → EMA 0.4555  仍 < floor，计数 2 → 进入 RECOVER
```

EMA 本身滞后，于是"连续两个窗口低于约束"这个条件被一次离群值满足了——
而迟滞的目的恰恰是防止这件事。修复：**计数用原始观测**（真的跌了才计数），
**转移额外要求 EMA 也同意**（不是一次抖动）。释放同理。

**面试可讲的点**：EMA 平滑与迟滞计数是两个不同职责，混用会互相破坏。
这个 bug 是被"单次噪声不应翻转"这条测试逼出来的，不是 review 看出来的。

### 6.3 迁移来的规则打分器把警告当成了推荐

`"不建议直接吃抗生素"` 被判为命中不安全动作
`immediate_antibiotics_without_evaluation`——因为正则只匹配"直接吃抗生素"，
不看前面的否定。旧项目的评测里一直存在这个假阳性。

修复加了否定检测。但第一版用固定 12 字符回看窗口，又产生了反向错误：

```text
"不要只在家观察，请尽快就医"
"尽快就医" 起点前 12 字符 = "不要只在家观察，请" → 含"不要" → 被误判为否定
=> infer_triage() 返回 None（本应是 urgent）
```

最终按子句切分（逗号/句号/分号/感叹号…），否定不跨子句。
`NEGATION_CUES` 里刻意**不含"不用"**——因为"不用去医院"本身就是一条不安全建议。

**面试可讲的点**：规则评测的可靠性边界。我保留了规则打分（确定性、零成本、可复现），
但把它的失效模式写进代码注释和测试，并且明确它只做行为分析、不做临床结论。

### 6.4 YAML 里显式写 `null` 的可选字段被判为类型错误

`configs/eval/final_test.yaml` 里 `max_samples: null` 表示"用全量"。
校验器看到 key 存在、值为 None、期望 int → 报错。修复：可选字段的显式 null
回落到默认值；必填字段为 null 仍然报错。保留显式 null 的写法是因为它对
review 更友好（把旋钮列出来而不是省略）。

### 6.5 rollout RNG 没进 checkpoint → resume 不是精确恢复

见 §4.2 第 3 点。加进 checkpoint 后 resume 的第 5~8 步指标与连跑逐位一致。

### 6.6 容器 2 GiB 内存：pytest 被 OOM killer 杀掉

```bash
$ python -m pytest -q
....s.................................................. [ 40%]
........................................
Killed                                    # exit 137
```

定位过程（三条命令就能划清责任）：

```bash
$ cat /sys/fs/cgroup/memory.current   # 1652027392  ≈ 1.65 GiB 已被占用
$ cat /sys/fs/cgroup/memory.max       # 2147483648  = 2 GiB 上限
$ ps -eo pid,rss,comm --sort=-rss | head
  bun 384916 KB / kiro-cli-chat 278628 KB / jupyter-lab / tensorboard / autopanel ...
```

不是我的代码泄漏，而是**编辑器与 agent 运行时本身占了 1.4–1.8 GiB**。
`nproc=1`，所以也不是线程池的问题。

后来把数字量化到底：

| 项 | 实测 |
|---|---|
| `import torch`（torch 2.3.0，什么都不做） | **374 MiB** RSS |
| `src.opd.loop_cli` 跑 8 步玩具模型 | **654 MiB** 峰值 RSS |
| agent 活跃时容器剩余可用 | **约 420–470 MiB** |

结论很清楚：**任何 import torch 的进程都只剩 ~66 MiB 活动空间**，
所以"整套测试一次跑完"在 agent 常驻的情况下是装不下的，与测试本身无关。

尝试过的方案与结果：

| 方案 | 结果 |
|---|---|
| `torch.set_num_threads(1)` + 每个测试 `gc.collect()`（`tests/conftest.py`） | 有帮助但不够 |
| `pytest --forked`（每测试一个子进程） | 18/19 通过，仍有一个子进程被杀，且 168.7s |
| 按文件/`-k` 分块，每块新进程（`scripts/run_cpu_checks.sh`） | **各组单独跑全绿**（采纳） |
| 给 preflight 去掉 torch 依赖：改用 `importlib.metadata` 读版本 + `nvidia-smi` 探测 GPU | 峰值 **350 MiB → 23 MiB**，preflight 彻底不再是 OOM 风险 |
| 运行器加内存闸门：每组前等待 ~750 MiB 空闲，被 137 杀掉重试一次，仍失败报 `OOM` 而不是 `FAIL` | 采纳——环境限制不会被误读成测试坏了 |
| `MALLOC_ARENA_MAX=2` | 小幅改善 |

现状：**每一组都在本机独立跑过并通过**（本文各节贴的就是那些真实输出）；
但在 agent 活跃时把 12 组串在一个脚本里跑，torch 相关的组会因为剩余内存不足被杀。
脚本头部把实测数字和处置写清楚了：要拿到一次完整的全绿输出，
在编辑器/agent 空闲时执行 `bash scripts/run_cpu_checks.sh` 即可；
换一台没有这个上限的机器，直接 `pytest -q` 一次跑完。

**面试可讲的点**：先量化再动手。看到 exit 137 不要先猜"代码内存泄漏"，
`memory.current` / `memory.max` / `ps --sort=-rss` 就能定位到是环境还是代码；
定位之后的正确动作也不是"加大机器"，而是把不必要的重依赖去掉
（preflight 从 350 MiB 降到 23 MiB 就是这么来的）、把长流程拆成独立进程、
并让工具明确区分"环境不足"与"测试失败"。

### 6.7 `git gc` 也被 OOM 杀了，而且 3.14 GiB 对象删不掉

清理旧仓库时：

```bash
$ git gc --prune=now
error: pack-objects died of signal 9        # 2 GiB 内存下无法 repack 3.14 GiB
$ git prune --expire=now                    # 更轻，不 repack
$ git count-objects -vH
count: 574   size: 3.14 GiB                 # 只掉了 96 个对象，体积没变
```

追查：`git fsck --unreachable` 只有 11 个 tree、0 个 commit（说明没有可丢失的工作），
但那些 136 MB 的 `optimizer.pt` blob 仍然可达。逐个 ref 排查后发现钩住它们的是
`refs/codex/turn-diffs/*`——Codex 工具自己的 turn 快照引用，**不是**用户的
`refs/heads/backup/*` 分支（后者经 `git rev-list --objects | grep optimizer.pt`
验证为 0 命中）。

决策：不动。这是旧仓库的 git 内部状态、属于用户整理过的工作流，系统盘已有
20 GiB 可用，收益小而风险不为零。回收命令写进了 ADR-0003 交给用户自己决定。

---

## 7. 磁盘清理（执行记录）

| 位置 | 清理前可用 | 清理后可用 |
|---|---|---|
| `/`（30 GiB） | 6.5 GiB | **20 GiB** |
| `/root/autodl-tmp`（50 GiB） | 7.9 GiB | **50 GiB**（已用 161 MiB） |

判据一句话：**能从"脚本 + 固定 seed + 公开数据"重建的，删；记录一次性事实的，留。**
删除明细与保留清单见 ADR-0003。归档下来的面试材料：
`docs/experiments/legacy/` 1.9 MiB / 118 个文件（15 个 run 的完整 loss 曲线、
LoRA 配置、评测报告、数据 manifest）+ 两个终态 adapter（141 MiB，放数据盘）。

---

## 8. 当前状态

```bash
$ bash scripts/run_cpu_checks.sh
=== summary ==========================================================
PASS  opd math: 29 passed in 5.83s
PASS  router: 25 passed in 1.62s
PASS  data splits: 34 passed in 3.34s
PASS  chat template: 10 passed, 1 skipped in 0.72s
PASS  eval: 58 passed in 4.14s
PASS  sft (dry-run path): 9 passed in 1.20s
PASS  run plan + veRL cfg: 16 passed in 1.33s
PASS  loop: rollout/teacher: 10 passed, 9 deselected in 10.21s
PASS  loop: artifacts: 4 passed, 15 deselected in 11.42s
PASS  loop: resume/routers: 6 passed, 13 deselected in 11.51s
PASS  data build (fixtures)
PASS  opd cpu dry-run
PASS  preflight: B2: GATE: PASSED WITH WARNINGS
PASS  preflight: O1: GATE: PASSED WITH WARNINGS
ALL CPU CHECKS PASSED
```

合计 **200 passed, 1 skipped**（skip 项是等 GPU 机器上用真 Qwen3 tokenizer 复核 loss mask）。
两个 preflight 的 "WARNINGS" 是环境项（本机无 GPU、transformers 4.49、未装 vllm/verl/ray），
在训练机上加 `--strict-env` 就会变成阻塞项。

### 已被测试证明的 vs 仅靠实现约定的

| 已被证明 | 仅靠约定（GPU 阶段需再验证） |
|---|---|
| 右移/mask/reduction 的数值正确性 | 真实 Qwen3 tokenizer 的 chat template 与 mask 边界 |
| advantage 符号与 PPO clip 行为 | vLLM sampler logprob 与训练端 logprob 的一致性（CPU 端已有等价哨兵） |
| old logprob 冻结、teacher forward-only | veRL `distillation_ppo_loss` 与本实现的数值一致性 |
| 路由器 EMA/迟滞/边界/早停/恢复 | 真实 controller-dev 评测下的路由稳定性 |
| split 互斥、标签不落盘、final-test 审计 | 真实数据规模下的去重效果（fixture 只有几十条） |
| checkpoint resume 逐位一致 | LoRA adapter 同步到 rollout 引擎后的一致性 |

---

## 9. P1 准备（无 GPU 部分，已完成）

GPU 一开机就要花钱，所以"开机后才能做的事"必须尽量少。这一节的东西全部在 CPU 上做完并测过。

### 9.1 付费门禁：`scripts/preflight.py`

一条命令决定"能不能开机"：

```bash
$ python scripts/preflight.py --run-config configs/runs/b2_medical_opd_qwen3_1_7b.yaml --emit-plan outputs/plans
PASS  run config schema + veRL constraints: b2-medical-opd-qwen3-1.7b-s42 (B2), 1 teacher(s), 37 veRL overrides
WARN  router config: kind=single_teacher (no CA router config needed)
WARN  data manifest: ...not found - build splits first
WARN  throughput provenance: estimate uses assumed throughput
PASS  cost cap: 3.71 RMB <= cap 60.00 (1.482 GPU-h, 614,400 generated tokens)
WARN  GPU availability: no CUDA device (planning mode; the real run needs 2 GPUs)
WARN  transformers >= 4.51 (Qwen3): 4.49.0 cannot load Qwen3
WARN  vllm / verl / ray installed: not installed
GATE: PASSED WITH WARNINGS
```

七类检查：run 配置 schema → veRL 约束 → 路由配置一致性（`window_steps` 必须等于
`controller_dev_every_steps`）→ 数据 manifest（sha256 + 零重叠 + **final_test
访问日志必须为空**）→ 吞吐数据来源（实测还是假设）→ 成本上限 → 环境版本。
开发机上环境项是 WARN，加 `--strict-env`（训练机上用）就变成 FAIL。

### 9.2 把 veRL 的启动期约束搬到 CPU 上

`src/opd/verl_config.py` 把项目配置翻译成 veRL 的 Hydra overrides，并**复刻了
veRL 自己会在启动时报的错**，这样配错会死在单元测试里而不是死在按小时计费的机器上：

| 复刻的约束 | 测试 |
|---|---|
| 教师池 GPU 数必须等于 Σ(replicas × world_size) | `test_dual_teacher_does_not_fit_two_gpus`——双教师 + 1 张教师卡直接抛错，加到 2 张才通过。这就是 ADR-0002 的核心发现被写成了可执行断言 |
| 命名教师不能与默认 `teacher_model` 条目共存 | `test_default_teacher_entry_cannot_coexist_with_named_teachers` |
| 教师条目不接受 LoRA adapter 路径 | `test_teacher_lora_adapter_is_rejected_with_a_pointer_to_the_adr` |
| `k1` + `use_policy_gradient=false` 无梯度；`forward_kl_topk` + PG 丢弃分布信号 | `test_loss_mode_and_policy_gradient_combinations` |
| 教师 `max_model_len` 必须覆盖 prompt+response+1 | `test_teacher_max_model_len_must_cover_prompt_plus_response` |
| LoRA 必须 `rollout.load_format=safetensors` | `test_single_teacher_overrides_contain_required_keys` |

### 9.3 run 计划与成本估算：`src/utils/run_plan.py`

同一个 YAML 同时生成"申报的计划"和"实际执行的命令"，避免计划写 200 step、
命令跑 2000 step。估算里刻意区分**实测**与**假设**：`measured: false` 时
`assumptions_used` 会把每个猜的数字列出来，计划文档里也照打——不让估算看起来
比它的依据更可靠。B2 的估算是 1.48 GPU-h / 3.71 元（假设吞吐下），
上限 60 元；超上限 `check_cost_cap()` 直接抛 `CostCapExceeded`。

墙钟时间按 `max(rollout, teacher) + optimizer` 算而不是三者相加，
因为 veRL 的 agent loop 里教师打分与其他样本的 rollout 是重叠的
（`test_wall_clock_assumes_overlap_not_sum` 固定了这个口径）。

### 9.4 SFT：CPU 上先把数据/模板/mask 验完

`src/sft/train.py` 有两个入口：`train()`（需要 GPU + transformers ≥ 4.51）和
**`dry_run()`（CPU、不下模型）**。dry-run 用真实 split 渲染全部样本、算 token 统计、
打印一条完整渲染示例和 mask 摘要：

```bash
$ python -m src.sft.train --config configs/sft/qwen3_1_7b_medical.yaml --dry-run
{"num_samples": ..., "num_dropped_too_long": ..., "trainable_ratio": ...,
 "length_percentiles": {"min":..., "p50":..., "p90":..., "p99":..., "max":...}}
--- rendered example (prompt) ---      <|im_start|>system ... <|im_start|>assistant
--- rendered example (completion, trained) ---   <think>...</think>回答<|im_end|>
```

两个刻意的设计：**超长样本丢弃而不截断**（截断推理答案等于教模型半句话就停），
以及 SFT 与评测**必须走同一个模板函数**——测试
`test_examples_use_the_same_template_as_evaluation` 把这条钉住了，
否则"医疗能力提升"里会混进格式差异的贡献。

### 9.5 Phase 1/2 的 run 配置

| 文件 | 用途 |
|---|---|
| `configs/sft/qwen3_1_7b_medical.yaml` | B1 Medical SFT，产出 Medical Teacher LoRA（rank 32 / all-linear，与 OPD student 一致） |
| `configs/runs/b2_medical_opd_qwen3_1_7b.yaml` | B2 单教师 OPD，走 veRL 原生路径（1 张教师卡够用） |
| `configs/runs/o1_ca_opd_qwen3_1_7b.yaml` | O1 CA-OPD 双教师；文件头明确写了它声明的是**原生路径（需第 3 张卡）**，双 4090 上实际走 ADR-0002 方案 B 的 adapter 路由服务，两份并存是为了将来做受控的显存/吞吐对比 |
| `configs/opd/router_qwen3_1_7b.yaml` | 路由器全部超参；`general_baseline` 必须先用 Base 模型在 controller dev 上测出来再冻结 |

每个 run 配置都带三类条件：success / early_stop / **abort**。abort 里写死了
"reverse KL 为 NaN 或 >20 连续 2 步"、"clip_fraction >0.5 连续 3 步"、
"entropy <0.1"、"训练期间 final_test_access.log 出现任何读取"——
这些是我在无人看管的长跑里希望它自己停下来的条件。

### 9.6 这一节新增的测试

```bash
$ python -m pytest tests/test_sft.py tests/test_run_plan_and_verl_config.py -q
25 passed in 4.41s
```

合计 Phase 0 + P1 准备：**200 passed, 1 skipped**（19 项闭环集成测试 + 181 项单测）。

---

## 10. 下一步（需 GPU）

1. 建 `opd` 环境（数据盘，见 ADR-0001 §3.2），先确认 Qwen3-1.7B 能被 tokenizer/config 加载；
2. 下 Qwen3-1.7B（3.80 GiB）；4B（7.51 GiB）等 Phase 3 再下；
3. 用真实数据跑 `python -m src.data.build_splits --config configs/data/base.yaml`，
   复核 manifest 的去重与领域纯度；
4. `python -m src.sft.train --config configs/sft/qwen3_1_7b_medical.yaml --dry-run`
   用**真 tokenizer** 复跑一遍（顺带解锁 `test_matches_real_tokenizer` 那条 skip 的测试）；
5. B1 SFT → 得到 Medical Teacher LoRA；
6. `python scripts/preflight.py --run-config configs/runs/b2_medical_opd_qwen3_1_7b.yaml --strict-env --with-tests`
   全绿后再开 B2 的 20 step 冒烟，核对 `actor/distillation/*` 与本仓库 CPU 参考实现的数值；
7. 把 B2 实测的 `system/*` 吞吐写回 run 配置的 `throughput`（`measured: true`），
   重算所有后续 run 的成本；
8. 双教师服务（ADR-0002 方案 B），记录共享 backbone 的显存/吞吐/切换延迟。


---

## 附录：面试问答准备（针对这段工作）

**Q：OPD 和 SFT/GRPO 的训练信号差别到底在哪？**
SFT 在固定答案上做 token CE，状态分布来自数据；GRPO 在自生成轨迹上用**序列级稀疏奖励**
（我前置工作里是规则奖励）；OPD 也在自生成轨迹上，但监督是 teacher 对**同一条 token 序列**
的 logprob，是逐 token 的稠密信号。所以 OPD 同时具备 on-policy 的状态对齐和 KD 的稠密性。

**Q：怎么保证 teacher 没有重新生成一条答案？**
三层：(1) 教师接口只有 `score(batch)`，类里不存在 generate 路径；(2) student 和 teacher
forward 的是同一个 `input_ids` 张量（batch 对象带 `fingerprint()`）；(3) 测试断言两者
target 网格完全一致且 teacher 输出无梯度。veRL 官方实现也是同一思路：教师侧用
`max_tokens=1` + `prompt_logprobs`，只对 prompt+response 前缀做 forward。

**Q：old logprob 不冻结会发生什么？**
importance ratio 恒等于 1，PPO 的 clip 完全失效，更新退化为无约束的策略梯度，
容易在一个 batch 内走得过远。我的实现里 `ppo_policy_loss` 直接拒绝带梯度的 old logprob，
并有一个"两次内更新 ratio 从 1.0 变 >1.0"的测试来证明它真的被冻住了。

**Q：为什么 advantage 只做 clip 不做归一化？**
OPD 的 advantage 来自 teacher-student logprob 差，本身有物理含义（谁更偏好这个 token），
做 batch 归一化会破坏跨领域可比性——而我需要按领域比较 KL 与更新强度。
所以用"领域级 KL EMA → 安全缩放 s_d ≤ 1 → 对称 clip"来控制尺度，且缩放只允许抑制。

**Q：固定 1:1 双教师有什么问题？**
step 比例不等于优化强度：teacher-student KL、有效 token 数、序列长度、student 当前能力
都会改变实际梯度贡献。所以我在指标里同时记录路由概率与**实际** batch/step/token 占比，
让这个论点可被数据检验，而不是只在文档里断言。

**Q：怎么防止 final test 泄漏？**
物理上不写答案到 OPD 池；类型上 controller-dev 与 final-test 是两个不可互换的类；
参数上读 final test 要显式许可 + 理由；审计上每次读取写日志；构造上 split 按保护
优先级分配 + content_hash 全局去重，manifest 记录两两重叠数（必须为 0）。

**Q：无 GPU 的两天你到底验证了什么？**
算法正确性（29 个数学测试）、调度逻辑（25 个）、数据隔离（44 个）、评测口径（58 个）、
端到端闭环含精确 resume（19 个），以及技术栈可行性（3 个 ADR）。这些都是在 GPU 上
debug 起来最贵、最容易被 loss 曲线掩盖的部分。
