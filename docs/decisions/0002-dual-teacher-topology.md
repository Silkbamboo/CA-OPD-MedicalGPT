# ADR-0002: 双 4090 上的共享 Backbone 双教师方案

- 状态：已接受（Accepted，方案 B 为主、方案 A 为回滚）
- 日期：2026-07-29
- 依赖：ADR-0001
- 影响范围：GPU 拓扑、教师服务实现、CA-OPD 路由集成点、系统指标口径

## 1. 问题

CA-OPD 需要在同一次训练里按窗口在两个教师之间切换：

```text
Base Teacher    = Qwen3 Base Backbone
Medical Teacher = 同一个 Backbone + 冻结的 Medical LoRA
```

而 ADR-0001 §2.3 核实到的 veRL 约束是：

1. 教师池 GPU 数必须精确等于 `Σ(num_replicas × per_replica_world_size)`，
   每个教师条目至少占 1 个 replica；
2. `distillation.teacher_models.<name>` 只接受 `model_path`，没有教师侧
   `lora_adapter_path`。

双 4090 的角色分配是 GPU0 = Student（训练 + rollout 分时复用）、GPU1 = 教师池，
即教师池只有 1 张卡。两个原生教师条目需要 2 张卡 → **原生 MOPD 配置直接抛错**。
即使卡够，第二个约束也会强迫我们把 Medical LoRA 合并成一份完整权重
（1.7B 多占 3.8 GiB、4B 多占 7.5 GiB 磁盘，且失去"共享 Backbone 只加载一份权重"
这个本项目声称的系统收益）。

## 2. 候选方案

### 方案 A：合并权重 + 原生 MOPD（需要 3 张卡或降配）

把 Medical LoRA merge 进 base，得到两份完整权重，配置两个 `teacher_models` 条目。

- 优点：完全走 veRL 原生路径，无需改框架代码；
- 缺点：
  - 双 4090 装不下（教师池要 2 张卡，student 还要 1 张）；
  - 显存里出现两份 backbone，与"共享 Backbone 降低峰值显存"的研究假设 H6 相矛盾；
  - 每次更换 Medical LoRA 都要重新 merge 并落盘。

### 方案 B：单教师服务 + 按请求 LoRA 路由（选定）

GPU1 只跑**一个** vLLM 实例，`enable_lora=True` 加载冻结的 Medical LoRA；
教师打分时按路由结果决定该请求是否带 `LoRARequest`：

```text
route = base    -> vLLM.generate(prompt_token_ids, SamplingParams(max_tokens=1, prompt_logprobs=0))
route = medical -> vLLM.generate(..., lora_request=LoRARequest("medical", 1, adapter_path))
```

- 优点：
  - 一份 backbone 权重，符合 H6，可以直接测「共享前后峰值显存」；
  - 教师仍然只做 forward/prefill（`max_tokens=1` + `prompt_logprobs`），
    与 veRL 教师侧行为一致；
  - 路由粒度可以做到 per-request，比窗口级更灵活（窗口级只是调度策略，不是实现限制）。
- 缺点：
  - 需要自建 `src/teacher/` 服务，或扩展 veRL 的 `AsyncTeacherLLMServerManager`
    使其支持「同一 server + 每请求 adapter」；
  - adapter 切换延迟需要实测并记录。

### 方案 C：窗口级换权重

一个教师服务，在窗口边界卸载/加载 adapter。实现最简单，但每次切换要重建
vLLM 的 LoRA 状态，切换开销进入训练关键路径，且无法混合 batch。仅作为
方案 B 实现受阻时的临时手段。

## 3. 决策

**采用方案 B**，分两步落地：

1. **Phase 1（单教师阶段）**：先用 veRL 原生单教师 OPD 打通 B2（Medical OPD）。
   此时 `teacher_models.teacher_model.model_path` 指向 merge 后的 Medical Teacher，
   或直接用 base + `enable_lora`（取决于 ADR-0001 §6 未决问题 1 的实测结果）。
   这一步不需要任何框架修改，用于验证 rollout/teacher 对齐、吞吐与显存。
2. **Phase 2（双教师阶段）**：实现 `src/teacher/lora_router_service.py`——
   一个进程内持有单个 vLLM engine、按 `teacher_id` 决定 `lora_request` 的打分服务，
   并把它接到训练循环上。集成方式按优先级：
   - B1：扩展 veRL `TeacherModelManager`，让一个 server 条目支持
     `adapter_path` + 按样本字段选择是否启用 adapter；
   - B2：veRL 侧配成单教师并指向我们的服务端点；
   - B3：完全用本仓库 `src/opd/loop.py` 的结构替换 veRL trainer（最后手段，
     需要新 ADR 说明放弃原生路径的理由）。

CA-OPD 路由与 veRL 的衔接点已确认可用（ADR-0001 §2.2）：
路由器只需在构造 batch 时写入 `distillation.teacher_key` 对应的样本字段
（例如 `teacher_route ∈ {medical, base}`），不需要改动 veRL 的路由逻辑。

## 4. 必须记录的系统指标

为了让 H6（共享 Backbone 能降低峰值显存）成为可检验的结论，
Phase 2 必须同时记录：

| 指标 | 口径 |
|---|---|
| `system/gpu_memory_peak_gb` | 分别在「共享 backbone + adapter 路由」与「两个独立教师进程」下测同一批数据 |
| `system/teacher_prefill_tokens_per_second` | 教师端 prompt_logprobs 吞吐 |
| adapter 切换延迟 | 从请求带 adapter 到返回 logprob 的额外耗时（p50/p95） |
| Student 等待教师比例 | `teacher_wait_seconds / step_seconds` |
| 两教师实际 batch/step/token 占比 | 与路由概率 `p_medical` 对照，验证"step 比例 ≠ 优化强度" |

最后一条是本项目的一个论点（固定 1:1 不等于两教师产生相同优化强度），
必须用实测数据支持，不能只在文档里断言。

## 5. 回滚方案

- 方案 B 的三种集成方式依次降级（B1 → B2 → B3）；
- 若 vLLM 的 `prompt_logprobs` 与 `LoRARequest` 组合在实测中不可用，
  退到方案 C（窗口级换 adapter），并在结果中记录切换开销；
- 若 AutoDL 能临时提供第三张卡，可用方案 A 作为交叉验证：
  同一数据下比较「原生 MOPD 双教师」与「共享 backbone 单服务」的
  显存与吞吐，这本身就是一张有说服力的系统对比表。

## 6. 未决问题

1. veRL 是否支持教师池与 student 共享同一张卡（fractional GPU）——决定 Phase 1
   是否需要把 student 与教师分时复用同一张卡；
2. vLLM 版本对 `enable_lora` + `prompt_logprobs` 组合的支持情况需实测
   （官方 multilora 示例同时使用 `LoRARequest` 与 `prompt_logprobs`，但未标注
   与 `max_tokens=1` 的组合行为）；
3. Medical LoRA 的 rank 与 target_modules 必须与 student LoRA 一致，
   否则 teacher/student 的能力差不可归因（在 SFT 配置中固定 rank=32、
   `target_modules=all-linear`）。
