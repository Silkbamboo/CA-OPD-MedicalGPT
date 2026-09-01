"""Build veRL OPD overrides from a project config, and pre-validate them.

Two jobs:

1. **Translate.** Our YAML (project vocabulary: student/teachers/router/domains)
   becomes veRL Hydra overrides (`distillation.*`, `actor_rollout_ref.*`).
2. **Fail on CPU instead of on a paid GPU.** veRL enforces several constraints at
   startup - teacher pool size, LoRA rollout load format, max_model_len - and
   discovering them after `docker run` on a rented dual-3090 costs money. The
   checks below mirror the documented constraints so a bad config dies in a
   unit test.

Documented constraints replicated here (see docs/decisions/0001):

* teacher pool GPUs must equal ``sum(num_replicas x per_replica_world_size)``
* teacher ``inference.max_model_len`` must cover ``prompt + response + 1``
* LoRA requires ``actor_rollout_ref.rollout.load_format=safetensors``
* PG OPD requires ``loss_mode=k1`` + ``use_policy_gradient=true``;
  ``forward_kl_topk`` + policy gradient is explicitly discouraged
* named teachers must not coexist with the default ``teacher_model`` entry
* OPD runs should disable the reference-policy KL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

PG_OPD_LOSS_MODES = ("k1", "kl", "k2", "k3", "abs", "mse", "low_var_kl")
TOPK_LOSS_MODES = ("forward_kl_topk",)


class VerlConfigError(ValueError):
    """Raised for a config veRL would reject at startup."""


@dataclass
class TeacherSpec:
    """One veRL teacher entry."""

    name: str  # config key under distillation.teacher_models
    routing_key: str  # value expected in sample[teacher_key]
    model_path: str
    num_replicas: int = 1
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    max_model_len: Optional[int] = None
    lora_adapter_path: Optional[str] = None  # NOT natively supported - see ADR-0002

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.tensor_model_parallel_size
            * self.data_parallel_size
            * self.pipeline_model_parallel_size
        )

    @property
    def gpu_footprint(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def __post_init__(self) -> None:
        if self.name == "teacher_model" and self.routing_key not in ("", "default"):
            # veRL silently pops the default entry when other named teachers exist
            pass
        for f in ("num_replicas", "tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size"):
            if getattr(self, f) < 1:
                raise VerlConfigError(f"teacher {self.name}: {f} must be >= 1")


@dataclass
class VerlOPDConfig:
    """Project-level description of one veRL OPD run."""

    student_model_path: str
    teachers: Sequence[TeacherSpec]
    teacher_key: str = "teacher_route"
    student_gpus: int = 1
    teacher_gpus_per_node: int = 1
    teacher_nnodes: int = 1
    # rollout / lengths
    max_prompt_tokens: int = 512
    max_response_tokens: int = 768
    prompt_batch_size: int = 2
    group_size: int = 2
    rollout_temperature: float = 1.0
    # student LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: str = "all-linear"
    lora_adapter_path: Optional[str] = None  # e.g. the Medical SFT adapter
    layered_summon: bool = True
    # OPD loss
    loss_mode: str = "k1"
    use_policy_gradient: bool = True
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    use_task_rewards: bool = False
    topk: Optional[int] = None
    loss_max_clamp: Optional[float] = None
    # optimisation
    lr: float = 3e-5
    total_steps: int = 200
    save_freq: int = 20
    extra_overrides: Dict[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------------
    def teacher_pool_size(self) -> int:
        return self.teacher_gpus_per_node * self.teacher_nnodes

    def teacher_footprint(self) -> int:
        return sum(t.gpu_footprint for t in self.teachers)

    def validate(self) -> None:
        if not self.teachers:
            raise VerlConfigError("at least one teacher is required")

        names = [t.name for t in self.teachers]
        if len(set(names)) != len(names):
            raise VerlConfigError(f"duplicate teacher entry names: {names}")
        if len(self.teachers) > 1 and "teacher_model" in names:
            raise VerlConfigError(
                "the default entry name 'teacher_model' is silently dropped when other "
                "named teachers exist; rename it (e.g. teacher_model1)"
            )
        routing = [t.routing_key for t in self.teachers]
        if len(self.teachers) > 1 and len(set(routing)) != len(routing):
            raise VerlConfigError(f"teachers must have distinct routing keys, got {routing}")

        pool, footprint = self.teacher_pool_size(), self.teacher_footprint()
        if pool != footprint:
            raise VerlConfigError(
                f"teacher pool size {pool} (n_gpus_per_node={self.teacher_gpus_per_node} x "
                f"nnodes={self.teacher_nnodes}) must equal the sum of teacher footprints "
                f"{footprint} ({', '.join(f'{t.name}:{t.gpu_footprint}' for t in self.teachers)}). "
                "On a 2-GPU box with GPU0 for the student, only ONE teacher replica fits - "
                "see docs/decisions/0002-dual-teacher-topology.md"
            )

        required_len = self.max_prompt_tokens + self.max_response_tokens + 1
        for t in self.teachers:
            if t.max_model_len is not None and t.max_model_len < required_len:
                raise VerlConfigError(
                    f"teacher {t.name}: max_model_len={t.max_model_len} < prompt+response+1={required_len}"
                )
            if t.lora_adapter_path:
                raise VerlConfigError(
                    f"teacher {t.name}: veRL teacher entries accept only model_path; a LoRA "
                    "teacher needs merged weights or the custom adapter-routing service "
                    "(ADR-0002 方案 B)"
                )

        if self.use_policy_gradient and self.loss_mode in TOPK_LOSS_MODES:
            raise VerlConfigError(
                f"loss_mode={self.loss_mode} with use_policy_gradient=true is discouraged: a PG "
                "update only moves the sampled token, discarding the top-k distributional signal"
            )
        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise VerlConfigError(
                "loss_mode=k1 with use_policy_gradient=false has no gradient through the teacher "
                "logprob; veRL raises ValueError for this combination"
            )
        if self.loss_mode in TOPK_LOSS_MODES and not self.topk:
            raise VerlConfigError(f"loss_mode={self.loss_mode} requires topk")
        if self.lora_rank < 1:
            raise VerlConfigError("lora_rank must be >= 1 (LoRA is required by the 24 GB/GPU budget)")
        if self.rollout_temperature <= 0:
            raise VerlConfigError("rollout_temperature must be > 0 for on-policy sampling")

    # -- override generation ----------------------------------------------
    def to_overrides(self) -> List[str]:
        """Hydra-style ``key=value`` overrides for ``verl.trainer.main_ppo``."""
        self.validate()
        ov: List[str] = [
            f"actor_rollout_ref.model.path={self.student_model_path}",
            f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={self.lora_alpha}",
            f"actor_rollout_ref.model.target_modules={self.lora_target_modules}",
            "actor_rollout_ref.model.use_shm=True",
            # required for vLLM to load the base model when LoRA is enabled
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.layered_summon={self.layered_summon}",
            f"actor_rollout_ref.rollout.temperature={self.rollout_temperature}",
            f"actor_rollout_ref.rollout.n={self.group_size}",
            f"actor_rollout_ref.actor.optim.lr={self.lr}",
            # OPD supervision replaces reference-policy regularisation
            "actor_rollout_ref.actor.use_kl_loss=false",
            "algorithm.use_kl_in_reward=false",
            f"data.max_prompt_length={self.max_prompt_tokens}",
            f"data.max_response_length={self.max_response_tokens}",
            f"data.train_batch_size={self.prompt_batch_size}",
            "data.shuffle=true",  # otherwise one teacher starves for whole epochs
            f"trainer.total_training_steps={self.total_steps}",
            f"trainer.save_freq={self.save_freq}",
            "distillation.enabled=true",
            f"distillation.n_gpus_per_node={self.teacher_gpus_per_node}",
            f"distillation.nnodes={self.teacher_nnodes}",
            f"distillation.teacher_key={self.teacher_key}",
            f"distillation.distillation_loss.loss_mode={self.loss_mode}",
            f"distillation.distillation_loss.use_policy_gradient={str(self.use_policy_gradient).lower()}",
            f"distillation.distillation_loss.use_task_rewards={str(self.use_task_rewards).lower()}",
            f"distillation.distillation_loss.clip_ratio_low={self.clip_ratio_low}",
            f"distillation.distillation_loss.clip_ratio_high={self.clip_ratio_high}",
        ]
        if self.use_policy_gradient:
            ov.append("distillation.distillation_loss.policy_loss_mode=vanilla")
        if self.topk:
            ov.append(f"distillation.distillation_loss.topk={self.topk}")
        if self.loss_max_clamp is not None:
            ov.append(f"distillation.distillation_loss.loss_max_clamp={self.loss_max_clamp}")
        if self.lora_adapter_path:
            ov.append(f"actor_rollout_ref.model.lora_adapter_path={self.lora_adapter_path}")

        for t in self.teachers:
            base = f"distillation.teacher_models.{t.name}"
            ov += [
                f"{base}.key={t.routing_key}",
                f"{base}.model_path={t.model_path}",
                f"{base}.num_replicas={t.num_replicas}",
                f"{base}.inference.name=vllm",
                f"{base}.inference.tensor_model_parallel_size={t.tensor_model_parallel_size}",
                f"{base}.inference.gpu_memory_utilization={t.gpu_memory_utilization}",
                f"{base}.inference.max_model_len="
                f"{t.max_model_len or self.max_prompt_tokens + self.max_response_tokens + 1}",
            ]
        for key, value in sorted(self.extra_overrides.items()):
            ov.append(f"{key}={value}")
        return ov

    def to_command(self, extra_env: Optional[Mapping[str, str]] = None) -> str:
        env = dict(extra_env or {"VLLM_USE_V1": "1"})
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
        overrides = " \\\n    ".join(self.to_overrides())
        prefix = f"{env_str} " if env_str else ""
        return f"{prefix}python3 -m verl.trainer.main_ppo \\\n    {overrides}\n"
