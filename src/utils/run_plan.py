"""Run planning: token / GPU-hour / cost estimates before a paid run starts.

The cost discipline in the project plan requires that every formal run declares,
*before* it starts: expected steps, tokens, duration and a cost ceiling. This
module turns a config into that declaration and refuses plans whose estimate
exceeds the declared ceiling.

Honesty rule enforced in code: throughput numbers are either **measured**
(carried over from a previous run's ``system/*`` metrics) or **assumed**. An
assumed plan is still produced, but ``assumptions_used`` lists every guessed
number so the plan can never look better-grounded than it is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SECONDS_PER_HOUR = 3600.0


class CostCapExceeded(RuntimeError):
    """Raised when the estimated cost exceeds the declared ceiling."""


class CostInputUnverified(RuntimeError):
    """Raised when a paid-run cost gate lacks a verified instance price."""


@dataclass
class ThroughputModel:
    """Throughput inputs for the estimate.

    ``measured`` marks whether these came from a real run. Defaults are
    uncalibrated placeholders for a 24 GB GPU and 1.7B/4B LoRA setup; they must
    be replaced by measurements from the exact rented 2 x RTX 3090 instance.
    """

    rollout_tokens_per_second: float = 300.0
    teacher_prefill_tokens_per_second: float = 3000.0
    optimizer_step_seconds: float = 1.5
    measured: bool = False
    source: str = "assumed-default"

    def __post_init__(self) -> None:
        for name in ("rollout_tokens_per_second", "teacher_prefill_tokens_per_second"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.optimizer_step_seconds <= 0:
            raise ValueError("optimizer_step_seconds must be > 0")


@dataclass
class RunPlan:
    """Everything that must be declared before a paid run starts."""

    run_id: str
    purpose: str
    baseline_id: str  # B0..B5 / O1
    model: str
    seed: int
    steps: int
    prompt_batch_size: int
    group_size: int
    max_prompt_tokens: int
    max_response_tokens: int
    checkpoint_every_steps: int
    controller_dev_every_steps: int
    controller_dev_samples: int
    num_gpus: int = 2
    price_per_gpu_hour_rmb: Optional[float] = None
    cost_cap_rmb: float = 60.0
    throughput: ThroughputModel = field(default_factory=ThroughputModel)
    data_manifest_path: Optional[str] = None
    data_manifest_sha256: Optional[str] = None
    success_criteria: List[str] = field(default_factory=list)
    early_stop_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "prompt_batch_size",
            "group_size",
            "max_prompt_tokens",
            "max_response_tokens",
            "checkpoint_every_steps",
            "controller_dev_every_steps",
            "num_gpus",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.cost_cap_rmb <= 0:
            raise ValueError("cost_cap_rmb must be > 0")
        if self.price_per_gpu_hour_rmb is not None and self.price_per_gpu_hour_rmb < 0:
            raise ValueError("price_per_gpu_hour_rmb must be >= 0")

    # -- derived quantities ------------------------------------------------
    @property
    def sequences_per_step(self) -> int:
        return self.prompt_batch_size * self.group_size

    @property
    def generated_tokens(self) -> int:
        """Upper bound: every rollout runs to the response length cap."""
        return self.steps * self.sequences_per_step * self.max_response_tokens

    @property
    def teacher_prefill_tokens(self) -> int:
        """The teacher scores prompt + response for every sequence."""
        return self.steps * self.sequences_per_step * (self.max_prompt_tokens + self.max_response_tokens)

    @property
    def controller_dev_evaluations(self) -> int:
        return self.steps // self.controller_dev_every_steps

    @property
    def controller_dev_generated_tokens(self) -> int:
        """MCQ answers are short; 16 tokens matches configs/eval/*.yaml."""
        return self.controller_dev_evaluations * self.controller_dev_samples * 16

    def estimate_seconds(self) -> Dict[str, float]:
        t = self.throughput
        rollout = (self.generated_tokens + self.controller_dev_generated_tokens) / t.rollout_tokens_per_second
        teacher = self.teacher_prefill_tokens / t.teacher_prefill_tokens_per_second
        optimizer = self.steps * t.optimizer_step_seconds
        # Student rollout and teacher scoring overlap in veRL's agent loop, so the
        # wall clock is bounded below by max(rollout, teacher), not their sum.
        overlapped = max(rollout, teacher)
        return {
            "rollout_seconds": rollout,
            "teacher_seconds": teacher,
            "optimizer_seconds": optimizer,
            "wall_clock_seconds": overlapped + optimizer,
        }

    def estimate_gpu_hours(self) -> float:
        return self.estimate_seconds()["wall_clock_seconds"] / SECONDS_PER_HOUR * self.num_gpus

    def estimate_cost_rmb(self) -> Optional[float]:
        if self.price_per_gpu_hour_rmb is None:
            return None
        return self.estimate_gpu_hours() * self.price_per_gpu_hour_rmb

    @property
    def assumptions_used(self) -> List[str]:
        out: List[str] = []
        if not self.throughput.measured:
            out.append(
                f"throughput is {self.throughput.source}: "
                f"rollout {self.throughput.rollout_tokens_per_second} tok/s, "
                f"teacher {self.throughput.teacher_prefill_tokens_per_second} tok/s, "
                f"step {self.throughput.optimizer_step_seconds} s"
            )
        out.append("generated-token count assumes every rollout hits the response cap (upper bound)")
        out.append("wall clock assumes rollout and teacher scoring overlap (veRL agent loop)")
        return out

    # -- gate --------------------------------------------------------------
    def check_cost_cap(self) -> float:
        cost = self.estimate_cost_rmb()
        if cost is None:
            raise CostInputUnverified(
                "price_per_gpu_hour_rmb is unverified; record the exact 2 x RTX 3090 "
                "instance-page price before a paid-run cost gate"
            )
        if cost > self.cost_cap_rmb:
            raise CostCapExceeded(
                f"estimated cost {cost:.2f} RMB exceeds cap {self.cost_cap_rmb:.2f} RMB; "
                f"reduce steps ({self.steps}), group_size ({self.group_size}) or "
                f"max_response_tokens ({self.max_response_tokens}), or raise the cap deliberately"
            )
        return cost

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        cost = self.estimate_cost_rmb()
        payload["derived"] = {
            "sequences_per_step": self.sequences_per_step,
            "generated_tokens": self.generated_tokens,
            "teacher_prefill_tokens": self.teacher_prefill_tokens,
            "controller_dev_evaluations": self.controller_dev_evaluations,
            **{k: round(v, 2) for k, v in self.estimate_seconds().items()},
            "estimated_gpu_hours": round(self.estimate_gpu_hours(), 3),
            "estimated_cost_rmb": round(cost, 2) if cost is not None else None,
            "cost_cap_rmb": self.cost_cap_rmb,
            "within_cap": cost <= self.cost_cap_rmb if cost is not None else None,
        }
        payload["assumptions_used"] = self.assumptions_used
        return payload

    def to_markdown(self) -> str:
        d = self.as_dict()["derived"]
        lines = [
            f"# Run plan: {self.run_id}",
            "",
            f"- purpose: {self.purpose}",
            f"- baseline: {self.baseline_id}",
            f"- model: {self.model}",
            f"- seed: {self.seed}",
            f"- data manifest: {self.data_manifest_path or 'NOT SET'}"
            + (f" (sha256 {self.data_manifest_sha256[:12]})" if self.data_manifest_sha256 else ""),
            "",
            "## Budget",
            "",
            f"| item | value |",
            f"|---|---|",
            f"| steps | {self.steps} |",
            f"| sequences/step | {self.sequences_per_step} (prompt {self.prompt_batch_size} x group {self.group_size}) |",
            f"| max prompt / response tokens | {self.max_prompt_tokens} / {self.max_response_tokens} |",
            f"| generated tokens (upper bound) | {d['generated_tokens']:,} |",
            f"| teacher prefill tokens | {d['teacher_prefill_tokens']:,} |",
            f"| controller-dev evaluations | {d['controller_dev_evaluations']} x {self.controller_dev_samples} samples |",
            f"| estimated wall clock | {d['wall_clock_seconds'] / 60:.1f} min |",
            f"| estimated GPU-hours | {d['estimated_gpu_hours']} ({self.num_gpus} GPU) |",
            f"| estimated cost | {d['estimated_cost_rmb'] if d['estimated_cost_rmb'] is not None else 'UNVERIFIED'} RMB (cap {self.cost_cap_rmb}) |",
            f"| within cap | {d['within_cap']} |",
            "",
            "## Assumptions",
            "",
        ]
        lines += [f"- {a}" for a in self.assumptions_used]
        for title, items in (
            ("Success criteria", self.success_criteria),
            ("Early stop", self.early_stop_criteria),
            ("Abort", self.abort_criteria),
        ):
            lines += ["", f"## {title}", ""]
            lines += [f"- {i}" for i in items] or ["- (not declared)"]
        if self.notes:
            lines += ["", "## Notes", "", self.notes]
        return "\n".join(lines) + "\n"
