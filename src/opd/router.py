"""Constraint-aware teacher router (PROJECT_PLAN.md §11.1 - §11.3).

The router answers one question per training *window*: with what probability
should the next window's batches be scored by the Medical Teacher vs the Base
Teacher?

Mechanism
---------
1. Ability EMAs on the controller-dev set::

       M_bar_k = rho * M_bar_{k-1} + (1 - rho) * M_k
       G_bar_k = rho * G_bar_{k-1} + (1 - rho) * G_k

2. Normalised ability gaps::

       g_M = (T_M - M_bar_k) / s_M
       g_G = ((B_G - delta) - G_bar_k) / s_G

   ``g_G > 0`` means general ability has fallen *below* the constraint floor
   ``B_G - delta`` and needs recovery.

3. Softmax routing with hard probability bounds::

       p_M = clip(exp(g_M/tau) / (exp(g_M/tau) + exp(g_G/tau)), p_min, p_max)
       p_G = 1 - p_M

4. Hysteresis state machine so a single noisy evaluation cannot flip the route,
   plus an early stop that only ever reads controller dev.

Hard isolation rule
-------------------
The router may only be constructed with an evaluator that self-identifies as a
controller-dev evaluator (``allows_control_decisions() is True``). Passing a
final-test evaluator raises :class:`FinalTestLeakageError`; PROJECT_PLAN.md §8.1
forbids final test from influencing scheduling in any way.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable

MEDICAL = "medical"
BASE = "base"
TEACHER_IDS = (MEDICAL, BASE)


class FinalTestLeakageError(RuntimeError):
    """Raised when a final-test evaluator is wired into a control path."""


@runtime_checkable
class ControlEvaluator(Protocol):
    """Minimal contract the router requires of an evaluator."""

    split: str

    def allows_control_decisions(self) -> bool:  # pragma: no cover - protocol
        ...


class RouterState(str, Enum):
    """``PURSUE_MEDICAL``: constraint satisfied, push medical ability.

    ``RECOVER_GENERAL``: general ability has been below the floor for
    ``windows_below_to_recover`` consecutive windows; medical probability is
    pinned to ``p_min`` until general recovers for ``windows_above_to_release``
    windows.
    """

    PURSUE_MEDICAL = "pursue_medical"
    RECOVER_GENERAL = "recover_general"


@dataclass(frozen=True)
class RouterConfig:
    """All routing hyper-parameters. Nothing here may be hard-coded elsewhere."""

    medical_target: float  # T_M
    general_baseline: float  # B_G (base model's controller-dev general accuracy)
    delta: float  # allowed general degradation
    scale_medical: float = 0.05  # s_M
    scale_general: float = 0.05  # s_G
    rho: float = 0.7  # EMA decay
    tau: float = 1.0  # softmax temperature
    p_min: float = 0.2
    p_max: float = 0.8
    window_steps: int = 20  # K optimizer steps per window
    windows_below_to_recover: int = 2
    windows_above_to_release: int = 1
    early_stop_patience: int = 3
    early_stop_min_improvement: float = 0.002
    initial_p_medical: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_min <= self.p_max <= 1.0:
            raise ValueError(f"require 0 <= p_min <= p_max <= 1, got {self.p_min}, {self.p_max}")
        if self.p_min == 0.0:
            raise ValueError("p_min must be > 0 so neither teacher can starve (PROJECT_PLAN §11.3)")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        for name in ("scale_medical", "scale_general"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.delta < 0:
            raise ValueError("delta is a magnitude of allowed degradation and must be >= 0")
        if self.window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if self.windows_below_to_recover < 1 or self.windows_above_to_release < 1:
            raise ValueError("hysteresis window counts must be >= 1")
        if not self.p_min <= self.initial_p_medical <= self.p_max:
            raise ValueError("initial_p_medical must respect [p_min, p_max]")

    @property
    def general_floor(self) -> float:
        """``B_G - delta``: the constraint the run must satisfy."""
        return self.general_baseline - self.delta

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RouterConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown router config keys: {sorted(unknown)}")
        return cls(**dict(data))


@dataclass
class RouterDecision:
    """What the router decided for the upcoming window (also the log record)."""

    window: int
    p_medical: float
    p_base: float
    state: RouterState
    medical_ema: float
    general_ema: float
    medical_gap: float
    general_gap: float
    constraint_satisfied: bool
    should_stop: bool
    reason: str = ""

    def as_metrics(self) -> Dict[str, float | str]:
        """Map onto the frozen metric vocabulary (agent.md §8)."""
        return {
            "router/p_medical": self.p_medical,
            "router/p_base": self.p_base,
            "router/state": self.state.value,
            "router/medical_gap": self.medical_gap,
            "router/general_gap": self.general_gap,
            "router/medical_ema": self.medical_ema,
            "router/general_ema": self.general_ema,
        }

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["state"] = self.state.value
        return out


class ConstraintAwareRouter:
    """Window-level Medical/Base teacher router."""

    def __init__(
        self,
        config: RouterConfig,
        evaluator: Optional[ControlEvaluator] = None,
        seed: int = 0,
    ):
        self.config = config
        self._check_evaluator(evaluator)
        self.evaluator = evaluator
        self._rng = random.Random(seed)
        self._seed = seed

        self.window: int = 0
        self.state: RouterState = RouterState.PURSUE_MEDICAL
        self.medical_ema: Optional[float] = None
        self.general_ema: Optional[float] = None
        self.p_medical: float = config.initial_p_medical
        self.consecutive_below: int = 0
        self.consecutive_above: int = 0
        self.windows_without_improvement: int = 0
        self.best_medical_ema: Optional[float] = None
        self.history: List[RouterDecision] = []
        self.teacher_counts: Dict[str, int] = {MEDICAL: 0, BASE: 0}

    # -- guards -----------------------------------------------------------
    @staticmethod
    def _check_evaluator(evaluator: Optional[ControlEvaluator]) -> None:
        if evaluator is None:
            return
        allows = getattr(evaluator, "allows_control_decisions", None)
        if allows is None or not callable(allows):
            raise TypeError(
                "router evaluator must implement allows_control_decisions(); "
                "wrap it in src.eval.runner.ControllerDevEvaluator"
            )
        if not allows():
            split = getattr(evaluator, "split", "<unknown>")
            raise FinalTestLeakageError(
                f"evaluator for split={split!r} must not drive teacher routing "
                "(PROJECT_PLAN.md §8.1: final test may not influence scheduling)"
            )

    # -- scheduling -------------------------------------------------------
    def is_window_boundary(self, step: int) -> bool:
        """True when ``step`` closes a window of ``K`` optimizer steps.

        Routing is updated per window, not per step (PROJECT_PLAN.md §11.2:
        "初始建议按窗口更新").
        """
        if step <= 0:
            return False
        return step % self.config.window_steps == 0

    def _ema(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return float(value)  # first observation initialises exactly
        return self.config.rho * previous + (1.0 - self.config.rho) * float(value)

    def _softmax_p_medical(self, g_m: float, g_g: float) -> float:
        tau = self.config.tau
        m = max(g_m / tau, g_g / tau)
        e_m = math.exp(g_m / tau - m)
        e_g = math.exp(g_g / tau - m)
        return e_m / (e_m + e_g)

    def update(
        self,
        medical_accuracy: float,
        general_accuracy: float,
        step: Optional[int] = None,
    ) -> RouterDecision:
        """Fold one controller-dev evaluation into the router and re-route.

        Call this once per window boundary. ``step`` is only recorded.
        """
        for name, value in (("medical_accuracy", medical_accuracy), ("general_accuracy", general_accuracy)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be an accuracy in [0, 1], got {value}")

        cfg = self.config
        self.window += 1
        self.medical_ema = self._ema(self.medical_ema, medical_accuracy)
        self.general_ema = self._ema(self.general_ema, general_accuracy)

        g_m = (cfg.medical_target - self.medical_ema) / cfg.scale_medical
        g_g = (cfg.general_floor - self.general_ema) / cfg.scale_general

        below = general_accuracy < cfg.general_floor
        if below:
            self.consecutive_below += 1
            self.consecutive_above = 0
        else:
            self.consecutive_above += 1
            self.consecutive_below = 0

        # -- hysteresis state machine
        #
        # The *count* is driven by the raw controller-dev measurement ("连续两个
        # 窗口低于约束", PROJECT_PLAN §11.3) while the *transition* additionally
        # requires the EMA to agree. Counting on the EMA alone was tried first
        # and rejected: because the EMA lags, one outlier window kept it under
        # the floor for two consecutive windows and flipped the state machine,
        # which is exactly the behaviour hysteresis is supposed to prevent
        # (see tests/test_router.py::test_single_noisy_window_does_not_flip_state).
        reason = ""
        if self.state is RouterState.PURSUE_MEDICAL:
            if self.consecutive_below >= cfg.windows_below_to_recover and self.general_ema < cfg.general_floor:
                self.state = RouterState.RECOVER_GENERAL
                reason = (
                    f"general accuracy below floor {cfg.general_floor:.4f} for "
                    f"{self.consecutive_below} consecutive windows (EMA {self.general_ema:.4f})"
                )
        else:  # RECOVER_GENERAL
            if self.consecutive_above >= cfg.windows_above_to_release and self.general_ema >= cfg.general_floor:
                self.state = RouterState.PURSUE_MEDICAL
                reason = (
                    f"general accuracy at/above floor {cfg.general_floor:.4f} for "
                    f"{self.consecutive_above} window(s) (EMA {self.general_ema:.4f})"
                )

        # -- routing probability
        if self.state is RouterState.RECOVER_GENERAL:
            # recovery mode: hand the window to the Base Teacher as much as the
            # starvation bound allows
            p_medical = cfg.p_min
        else:
            p_medical = min(max(self._softmax_p_medical(g_m, g_g), cfg.p_min), cfg.p_max)
        self.p_medical = float(p_medical)

        # -- early stop (controller dev only)
        constraint_satisfied = self.general_ema >= cfg.general_floor
        medical_reached = self.medical_ema >= cfg.medical_target
        if self.best_medical_ema is None or self.medical_ema > self.best_medical_ema + cfg.early_stop_min_improvement:
            self.best_medical_ema = self.medical_ema
            self.windows_without_improvement = 0
        else:
            self.windows_without_improvement += 1
        should_stop = (
            constraint_satisfied
            and medical_reached
            and self.windows_without_improvement >= cfg.early_stop_patience
        )
        if should_stop:
            reason = (
                f"both objectives satisfied and no medical improvement > "
                f"{cfg.early_stop_min_improvement} for {self.windows_without_improvement} windows"
            )

        decision = RouterDecision(
            window=self.window,
            p_medical=self.p_medical,
            p_base=1.0 - self.p_medical,
            state=self.state,
            medical_ema=float(self.medical_ema),
            general_ema=float(self.general_ema),
            medical_gap=float(g_m),
            general_gap=float(g_g),
            constraint_satisfied=bool(constraint_satisfied),
            should_stop=bool(should_stop),
            reason=reason,
        )
        self.history.append(decision)
        return decision

    # -- sampling ---------------------------------------------------------
    def sample_teacher(self) -> str:
        """Sample the teacher for one batch from the current window policy."""
        teacher = MEDICAL if self._rng.random() < self.p_medical else BASE
        self.teacher_counts[teacher] += 1
        return teacher

    def sample_window(self, num_batches: int) -> List[str]:
        if num_batches < 0:
            raise ValueError("num_batches must be >= 0")
        return [self.sample_teacher() for _ in range(num_batches)]

    def realised_medical_fraction(self) -> Optional[float]:
        total = sum(self.teacher_counts.values())
        if total == 0:
            return None
        return self.teacher_counts[MEDICAL] / total

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "window": self.window,
            "state": self.state.value,
            "medical_ema": self.medical_ema,
            "general_ema": self.general_ema,
            "p_medical": self.p_medical,
            "consecutive_below": self.consecutive_below,
            "consecutive_above": self.consecutive_above,
            "windows_without_improvement": self.windows_without_improvement,
            "best_medical_ema": self.best_medical_ema,
            "teacher_counts": dict(self.teacher_counts),
            "rng_state": self._rng.getstate(),
            "seed": self._seed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_cfg = RouterConfig(**dict(state["config"]))
        if saved_cfg != self.config:
            raise ValueError(
                "router config in checkpoint differs from the current config; "
                "changing routing hyper-parameters mid-run breaks comparability"
            )
        self.window = int(state["window"])
        self.state = RouterState(state["state"])
        self.medical_ema = state["medical_ema"]
        self.general_ema = state["general_ema"]
        self.p_medical = float(state["p_medical"])
        self.consecutive_below = int(state["consecutive_below"])
        self.consecutive_above = int(state["consecutive_above"])
        self.windows_without_improvement = int(state["windows_without_improvement"])
        self.best_medical_ema = state["best_medical_ema"]
        self.teacher_counts = dict(state["teacher_counts"])
        rng_state = state.get("rng_state")
        if rng_state is not None:
            # json round-trips tuples as lists; random.setstate needs tuples
            internal = rng_state[1]
            self._rng.setstate((rng_state[0], tuple(internal), rng_state[2]))


@dataclass
class FixedRatioRouter:
    """Baseline router for B4/B5 (IDT 1:1, 2:1) and the "no dynamic routing" ablation.

    Same interface as :class:`ConstraintAwareRouter` so the training loop is
    identical across baselines - the only difference is where ``p_medical``
    comes from, which is exactly what PROJECT_PLAN.md §12 wants to ablate.
    """

    p_medical: float
    window_steps: int = 20
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    teacher_counts: Dict[str, int] = field(default_factory=lambda: {MEDICAL: 0, BASE: 0})

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_medical <= 1.0:
            raise ValueError("p_medical must be in [0, 1]")
        self._rng = random.Random(self.seed)
        self.window = 0
        self.state = RouterState.PURSUE_MEDICAL

    def is_window_boundary(self, step: int) -> bool:
        return step > 0 and step % self.window_steps == 0

    def update(self, medical_accuracy: float, general_accuracy: float, step: Optional[int] = None) -> RouterDecision:
        self.window += 1
        return RouterDecision(
            window=self.window,
            p_medical=self.p_medical,
            p_base=1.0 - self.p_medical,
            state=self.state,
            medical_ema=float(medical_accuracy),
            general_ema=float(general_accuracy),
            medical_gap=float("nan"),
            general_gap=float("nan"),
            constraint_satisfied=True,
            should_stop=False,
            reason="fixed ratio baseline: routing does not react to ability gaps",
        )

    def sample_teacher(self) -> str:
        teacher = MEDICAL if self._rng.random() < self.p_medical else BASE
        self.teacher_counts[teacher] += 1
        return teacher

    def sample_window(self, num_batches: int) -> List[str]:
        return [self.sample_teacher() for _ in range(num_batches)]

    def realised_medical_fraction(self) -> Optional[float]:
        total = sum(self.teacher_counts.values())
        return None if total == 0 else self.teacher_counts[MEDICAL] / total

    def state_dict(self) -> Dict[str, Any]:
        return {"p_medical": self.p_medical, "window": self.window, "teacher_counts": dict(self.teacher_counts)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.p_medical = float(state["p_medical"])
        self.window = int(state["window"])
        self.teacher_counts = dict(state["teacher_counts"])
