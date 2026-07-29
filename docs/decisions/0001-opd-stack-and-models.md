# ADR-0001: OPD 技术栈版本与模型选择

- 状态：已接受（Accepted）
- 日期：2026-07-29
- 决策者：项目所有者 + 实现代理
- 影响范围：训练框架、推理框架、模型、Python 环境；影响所有后续 run 的可比性

## 1. 背景

Phase 0 需要在花任何 GPU/付费资源之前确定：Qwen3 + veRL(OPD) + vLLM(multi-LoRA) + Ray
是否存在一个可用的版本组合，以及双教师方案在双 4090 上是否可行。这是全项目最大的单点风险：
如果 OPD 正式栈不可行，方法设计再完整也无法产出主结果。

调研时的开发环境（已实测）：

| 项 | 值 | 结论 |
|---|---|---|
| GPU | 无（`nvidia-smi: No devices were found`） | Phase 0 只能 CPU |
| torch | 2.3.0+cu121 | 太旧，vLLM/veRL 不可用 |
| transformers | 4.49.0 | **无法加载 Qwen3** |
| trl / peft | 0.15.2 / 0.18.1 | 可用于 legacy 线，不用于主线 |
| vllm / verl / ray | 未安装 | 需新建环境 |
| 容器内存上限 | 2 GiB（`/sys/fs/cgroup/memory.max`） | 影响 CPU 测试组织方式 |

## 2. 核实到的事实（带出处）

### 2.1 veRL 原生支持 OPD，且语义与本项目一致

来源：<https://verl.readthedocs.io/en/latest/algo/opd.html>（文档标注 Last updated 05/26/2026）

- 配置命名空间 `distillation.*`；`enabled=true` 时 `main_ppo` 单独分配 teacher 资源池，
  actor loss 从 `ppo_loss` 切换为 `distillation_ppo_loss`。
- 两种损失变体：
  - **GKD OPD**：`loss_mode=forward_kl_topk`，`use_policy_gradient=false`，直接反传 top-k 前向 KL；
  - **PG OPD**：`loss_mode=k1`，`use_policy_gradient=true`，把 reverse-KL 的单样本估计当作 reward
    做 PPO clip 更新（`clip_ratio_low/high`）。
- 教师侧用 `max_tokens=1` + `prompt_logprobs`，即对 `prompt+response` 前缀做一次 forward，
  **不重新生成**；温度被强制为 1.0（文档明确说明忽略配置值并告警）。
- 文档明确限制：当前推理服务只返回「采样 token + teacher top-k」的 logprob，
  **不支持在任意 token id 上聚集 logprob**，因此只实现 teacher-top-k 前向 KL，
  没有 student-top-k 反向 KL。

对本项目的意义：`src/opd/core.py` 实现的 `A_t = beta*(log pi_T - log pi_S)` + PPO clip
就是 PG OPD 路线，与 veRL `loss_mode=k1 + use_policy_gradient=true` 对应。Phase 0 的 CPU
参考实现因此可以作为 veRL 集成的对照物，而不是另一套自研算法。

### 2.2 veRL 原生多教师（MOPD）按样本字段路由

- `distillation.teacher_key`（默认 `data_source`）指定用哪个样本字段选教师；
  多教师时 `sample[teacher_key]` 必须等于某个教师条目的 `key`，否则报错。
- 每个教师一个 `distillation.teacher_models.<name>` 条目，含 `key`、`model_path`、
  `num_replicas`、`inference.*`。
- 文档给出的陷阱：新增命名教师时，名为 `teacher_model` 的默认条目会被静默弹出，
  必须整体重命名（如 `teacher_model1`）。

对本项目的意义：CA-OPD 的动态路由**不需要改 veRL 的路由机制**——路由器只要在构造
rollout batch 时给每个样本写入 `teacher_key` 字段的值即可。这条是好消息，
降低了实现风险。

### 2.3 硬约束：教师资源池按 GPU 精确切分

- `distillation.n_gpus_per_node × nnodes` 必须**精确等于**
  `Σ(num_replicas × per_replica_world_size)`，否则 `DistillationConfig.__post_init__` 抛错；
- `per_replica_world_size = tensor_model_parallel_size × data_parallel_size × pipeline_model_parallel_size`；
- 教师 replica 按连续 GPU bundle 线性切分，且不允许跨节点边界（`_validate_replica_node_alignment`）。

推论（本项目最重要的一条）：双 4090 拓扑下 GPU0 给 Student、教师池只剩 1 张卡，
而两个教师各需至少 1 个 replica × 1 GPU = 2 张卡 → **veRL 原生 MOPD 在双卡上装不下**。
另外 `teacher_models.<name>` 只接受 `model_path`，文档中**没有** teacher 侧的
`lora_adapter_path`，所以「Base + 冻结 Medical LoRA」两路教师无法直接用原生配置表达。
处理方案见 ADR-0002。

### 2.4 veRL 环境要求与 LoRA 配置

来源：<https://verl.readthedocs.io/en/latest/start/install.html>、
<https://verl.readthedocs.io/en/latest/advance/ppo_lora.html>（Last updated 02/03/2026）

- Python ≥ 3.10；CUDA ≥ 12.8；cuDNN ≥ 9.10；vLLM 0.8.3+ 经过稳定性测试，推荐 `VLLM_USE_V1=1`；
  官方镜像形如 `verlai/verl:vllm011.latest`，v0.6 的基础镜像为 cu128 + torch 2.8.0。
- LoRA 支持 `strategy=fsdp|fsdp2` + `rollout.name=vllm|sglang`，通过 HuggingFace PEFT 实现。
- 必需项：`model.lora_rank>0`、`model.lora_alpha`、`model.target_modules`（通常 `all-linear`）、
  **`rollout.load_format=safetensors`**。
- 可选项：`model.lora_adapter_path` 可加载已有 adapter（支持多阶段训练，
  正好用于「Medical SFT LoRA → OPD student 初始化」）；`model.lora.merge` 控制是否
  把 adapter 合并进 base 再同步给 rollout 引擎。
- 显存受限（< 48GB）建议 `rollout.layered_summon=true`；LoRA rank 建议 ≥ 32
  （文档引用的经验：0.5B 模型 rank=32 收敛与全参近似）。

### 2.5 模型事实（HF API 实测）

| 模型 | 参数量 | 权重体积 | 架构 | config 声明的 transformers |
|---|---|---|---|---|
| Qwen/Qwen3-1.7B | 2.03B（含 embedding） | 3.80 GiB | `Qwen3ForCausalLM` | 4.51.0 |
| Qwen/Qwen3-4B | 4.02B | 7.51 GiB | `Qwen3ForCausalLM` | 4.51.0 |
| Qwen/Qwen3.5-4B | 4.66B | 8.68 GiB | `Qwen3_5ForConditionalGeneration` | 4.57.0.dev0 |

`Qwen3-1.7B` 的 `tie_word_embeddings=true`、28 层、hidden 2048、
`max_position_embeddings=40960`。

### 2.6 无法核实的事项

`PROJECT_PLAN.md` 引用的 MOPD `arXiv:2606.30406` **未能核实**：开发机访问
`export.arxiv.org` 返回空响应（连已知有效 id `2306.13649` 的对照查询也为空），
判断为网络限制而非条目不存在。veRL 的 OPD 文档在 "Multi-Teacher OPD" 一节
引用的是 DeepSeek-V4、Mimo-v2-flash、GLM-5、Nemotron-Cascade 2 等工作，
并未引用某篇单独的 "MOPD" 论文。

行动项：在 README / 简历里引用该条之前必须联网核实；若无法确认，
改为引用 veRL OPD 文档的 Multi-Teacher 章节及其参考文献。

## 3. 决策

### 3.1 模型（已由项目所有者确认）

- 开发模型：**Qwen3-1.7B**；主结果模型：**Qwen3-4B**。
- **不使用 Qwen3.5-4B**。理由：架构类为 `Qwen3_5ForConditionalGeneration`、
  需要 transformers 4.57.0.dev0（未发布版本）、权重多 1.17 GiB；引入未发布依赖会
  同时增加 veRL/vLLM 兼容风险与不可复现风险，收益仅是与参考项目"看起来同代"。
  Qwen3-4B 是纯文本 `Qwen3ForCausalLM`，与 1.7B 同族同 tokenizer，
  满足 OPD 对「teacher 与 student 共享 tokenizer/vocab」的硬性要求。

### 3.2 环境（两套并存，互不干扰）

| 环境 | 用途 | 关键版本 | 位置 |
|---|---|---|---|
| `legacy`（现有 base conda） | Qwen2.5-1.5B 时期的 SFT/DPO/GRPO 复现 | torch 2.3.0+cu121, transformers 4.49.0, trl 0.15.2, peft 0.18.1 | `/root/miniconda3`（不动） |
| `opd`（待建） | Qwen3 + veRL + vLLM 主线 | Python 3.12, CUDA ≥ 12.8, torch 2.8.x(cu128), transformers ≥ 4.51, vLLM ≥ 0.8.3（首选官方镜像同版本 0.11.x）, verl 从源码 `pip install --no-deps -e .`, ray, peft, flash-attn | **数据盘** `/root/autodl-tmp/envs/opd` |

决策要点：

1. **不升级 legacy 环境**。transformers 4.49 → ≥4.51 会破坏已完成的 Qwen2.5 实验的
   可复现性，而 legacy 线是简历里的前置工作，必须保持可复跑。
2. 新环境建在**数据盘**（`conda create --prefix /root/autodl-tmp/envs/opd`）并使用
   `pip install --no-cache-dir`。原因：系统盘只有 30 GiB，vLLM 会拖入整套
   nvidia cu12 运行库（预计 12–15 GiB），放系统盘会挤爆。
3. 优先用 veRL 官方镜像（`verlai/verl:vllm011.latest`）而不是手工装栈；
   AutoDL 若不便用自定义镜像，则严格按官方 `scripts/install_vllm_sglang_mcore.sh`
   的顺序装（**先装推理框架再装 torch 相关**，因为 vLLM 会强制覆盖 torch 版本）。
4. 所有版本在首次成功跑通后写入 `env/requirements-opd.lock`，后续 run 只用锁定版本；
   任何升级都要新开 ADR 并说明对历史结果可比性的影响。

### 3.3 OPD 路线

- 主路线：**PG OPD**（`loss_mode=k1`, `use_policy_gradient=true`, `policy_loss_mode=vanilla`,
  `clip_ratio_low=0.2`, `clip_ratio_high=0.28`）。与本项目的 advantage 定义、
  KL 安全缩放、advantage clip 直接对应。
- 关闭与 reference policy 的 KL：`actor.use_kl_loss=false`、`algorithm.use_kl_in_reward=false`
  （veRL 文档明确建议，否则学生同时被 ref policy 正则化和 teacher 蒸馏，两个信号纠缠）。
- `use_task_rewards=false`：本项目不引入任务奖励，纯蒸馏信号，便于归因。
- 备选：GKD OPD（`forward_kl_topk`, topk=128）仅作为消融项，不作为主结果。

## 4. 影响

- Phase 0 的所有正确性工作可以在 CPU 上完成，不受上述版本决策影响
  （`src/opd/core.py` 只依赖 torch）。
- Phase 1 开始前必须先完成 3.2 的环境搭建，并在单 GPU 上跑通
  「Qwen3-1.7B + vLLM rollout + 单教师 OPD 20 step」冒烟。
- 双教师拓扑需要 ADR-0002 决定的扩展方案，不能直接用原生 MOPD 配置。
- 磁盘预算（清理后实测：系统盘 20 GiB 可用、数据盘 50 GiB 可用）：
  opd 环境 ~15 GiB + Qwen3-1.7B 3.8 GiB + Qwen3-4B 7.5 GiB（Phase 3 再下）+
  数据 ~3 GiB + 产物 ~11 GiB ≈ 40 GiB，全部放数据盘可容纳。

## 5. 回滚方案

| 失败情形 | 回滚动作 |
|---|---|
| veRL OPD 在双 4090 上跑不起来（显存/Ray 调度） | 退到本仓库的 `src/opd/loop.py` 参考实现 + vLLM 推理服务，自建单进程 OPD trainer；记录为「放弃 veRL 原生路径」的新 ADR |
| vLLM 与 torch 版本冲突无法解决 | 改用 veRL 官方 docker 镜像；若 AutoDL 不支持则退到 transformers `generate` 做 rollout（吞吐下降，需在结果中说明） |
| Qwen3-4B 在 24GB 上 LoRA+rollout 显存不足 | 降 `max_response_length`→768、`group_size`→2、开 `layered_summon`、`param_offload`；仍不足则主结果退回 Qwen3-1.7B 并明确写入局限 |
| transformers ≥4.51 与 peft/trl 冲突 | 锁定 veRL 镜像内的组合；SFT 也在同一环境内执行，避免跨环境版本漂移 |

## 6. 未决问题（进入 Phase 1 前必须闭环）

1. veRL 是否允许教师池与 student 共享同一张卡（fractional GPU / Ray placement group）——
   文档未说明，需要实机验证；这直接决定 ADR-0002 选方案 A 还是 B。
2. vLLM 单进程 multi-LoRA 下 `prompt_logprobs` 是否能按请求切换 adapter 并保持吞吐——
   官方 multilora 示例同时使用 `LoRARequest` 与 `prompt_logprobs`，但需实测延迟。
3. MOPD 引用核实（见 2.6）。
