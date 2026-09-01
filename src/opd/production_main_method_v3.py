"""Pure P6 method-specific routing and domain-KL safety state.

The B2 optimizer, data schedule, and three-policy objective remain common.  The
only method-specific inputs represented here are the teacher route and, for
CA-OPD only, a damp-only domain-level advantage scale.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


class P6FormalMethodError(RuntimeError):
    """A formal IDT/CA method route or safety state is invalid."""


TEACHERS = ("medical", "base")


def _seed(seed: int, *parts: Any) -> int:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def balanced_teacher_routes(
    *,
    method_id: str,
    step_index: int,
    source_roles: Sequence[str],
    p_medical: float,
    seed: int,
) -> tuple[str, ...]:
    """Return deterministic per-step routes without source/teacher coupling."""

    if method_id not in {"IDT", "CA-OPD"}:
        raise P6FormalMethodError("formal method identity differs")
    if step_index < 0 or not 0.0 <= float(p_medical) <= 1.0:
        raise P6FormalMethodError("formal method route inputs differ")
    if len(source_roles) != 4 or sorted(source_roles).count("medical_opd_o1") != 2 or sorted(source_roles).count("medical_opd_cmb") != 2:
        raise P6FormalMethodError("formal method source composition differs")
    if method_id == "IDT":
        result = [""] * 4
        for source in sorted(set(source_roles)):
            indexes = [index for index, role in enumerate(source_roles) if role == source]
            order = list(TEACHERS)
            random.Random(_seed(seed, method_id, step_index, source)).shuffle(order)
            for index, route in zip(indexes, order, strict=True):
                result[index] = route
        return tuple(result)
    # The CA window scheduler owns exact fraction balancing.  This helper is a
    # deterministic single-step fallback used only by bounded unit tests.
    rng = random.Random(_seed(seed, method_id, step_index))
    return tuple("medical" if rng.random() < p_medical else "base" for _ in source_roles)


class MethodRouteStateV1:
    """Deterministic CA window schedule with exact per-source route counts."""

    def __init__(self, *, method_id: str, seed: int, window_steps: int) -> None:
        if method_id not in {"IDT", "CA-OPD"} or window_steps <= 0:
            raise P6FormalMethodError("formal method route state differs")
        self.method_id = method_id
        self.seed = int(seed)
        self.window_steps = int(window_steps)
        self.window_start_step = 0
        self.p_medical = 0.5
        self._routes_by_source: dict[str, list[str]] = {}
        self._consumed: dict[str, int] = {}
        self.teacher_counts = {"medical": 0, "base": 0}
        self.source_teacher_counts: dict[str, dict[str, int]] = {}

    def start_window(self, *, p_medical: float, start_step: int) -> None:
        if not 0.0 <= float(p_medical) <= 1.0 or start_step < 0 or start_step % self.window_steps != 0:
            raise P6FormalMethodError("formal method window boundary differs")
        self.window_start_step = int(start_step)
        self.p_medical = float(p_medical)
        self._routes_by_source = {}
        self._consumed = {}
        for source in ("medical_opd_o1", "medical_opd_cmb"):
            count = self.window_steps * 2
            medical_count = int(round(self.p_medical * count))
            routes = ["medical"] * medical_count + ["base"] * (count - medical_count)
            random.Random(
                _seed(self.seed, self.method_id, self.window_start_step, source)
            ).shuffle(routes)
            self._routes_by_source[source] = routes
            self._consumed[source] = 0

    def routes_for_step(
        self, step_index: int, source_roles: Sequence[str]
    ) -> tuple[str, ...]:
        if self.method_id == "IDT":
            result = balanced_teacher_routes(
                method_id="IDT",
                step_index=step_index,
                source_roles=source_roles,
                p_medical=0.5,
                seed=self.seed,
            )
        else:
            if not self._routes_by_source:
                raise P6FormalMethodError("CA route window is not initialized")
            if not self.window_start_step <= step_index < self.window_start_step + self.window_steps:
                raise P6FormalMethodError("CA route request escaped frozen window")
            values: list[str] = []
            for source in source_roles:
                cursor = self._consumed[source]
                routes = self._routes_by_source[source]
                if cursor >= len(routes):
                    raise P6FormalMethodError("CA route window was over-consumed")
                values.append(routes[cursor])
                self._consumed[source] = cursor + 1
            result = tuple(values)
        for source, route in zip(source_roles, result, strict=True):
            if route not in TEACHERS:
                raise P6FormalMethodError("formal method teacher route differs")
            self.teacher_counts[route] += 1
            counts = self.source_teacher_counts.setdefault(
                str(source), {"medical": 0, "base": 0}
            )
            counts[route] += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "seed": self.seed,
            "window_steps": self.window_steps,
            "window_start_step": self.window_start_step,
            "p_medical": self.p_medical,
            "routes_by_source": {key: list(value) for key, value in self._routes_by_source.items()},
            "consumed": dict(self._consumed),
            "teacher_counts": dict(self.teacher_counts),
            "source_teacher_counts": {
                key: dict(value) for key, value in self.source_teacher_counts.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not (
            value.get("schema_version") == 1
            and value.get("method_id") == self.method_id
            and int(value.get("seed", -1)) == self.seed
            and int(value.get("window_steps", -1)) == self.window_steps
        ):
            raise P6FormalMethodError("formal method resume route identity differs")
        self.window_start_step = int(value["window_start_step"])
        self.p_medical = float(value["p_medical"])
        self._routes_by_source = {
            str(key): [str(item) for item in values]
            for key, values in dict(value["routes_by_source"]).items()
        }
        self._consumed = {str(key): int(item) for key, item in dict(value["consumed"]).items()}
        self.teacher_counts = {
            str(key): int(item) for key, item in dict(value["teacher_counts"]).items()
        }
        self.source_teacher_counts = {
            str(key): {str(route): int(count) for route, count in dict(items).items()}
            for key, items in dict(value["source_teacher_counts"]).items()
        }


class DomainKLSafetyV1:
    """Frozen Phase-0 domain EMA scale: ``min(1, kappa/(abs(EMA)+eps))``."""

    def __init__(
        self,
        *,
        kappa: Mapping[str, float],
        rho: float,
        eps: float = 1.0e-6,
    ) -> None:
        if set(kappa) != set(TEACHERS) or any(float(value) <= 0 for value in kappa.values()):
            raise P6FormalMethodError("domain KL kappa differs")
        if not 0.0 <= float(rho) < 1.0 or eps <= 0:
            raise P6FormalMethodError("domain KL EMA configuration differs")
        self.kappa = {key: float(value) for key, value in kappa.items()}
        self.rho = float(rho)
        self.eps = float(eps)
        self.ema = {key: 0.0 for key in TEACHERS}
        self.seen = {key: False for key in TEACHERS}
        self.trigger_count = {key: 0 for key in TEACHERS}

    def update(self, domain: str, reverse_kl: float) -> float:
        if domain not in TEACHERS:
            raise P6FormalMethodError("domain KL route differs")
        value = float(reverse_kl)
        if not self.seen[domain]:
            self.ema[domain] = value
            self.seen[domain] = True
        else:
            self.ema[domain] = self.rho * self.ema[domain] + (1.0 - self.rho) * value
        scale = self.scale(domain)
        if scale < 1.0:
            self.trigger_count[domain] += 1
        return scale

    def scale(self, domain: str) -> float:
        if domain not in TEACHERS:
            raise P6FormalMethodError("domain KL route differs")
        if not self.seen[domain]:
            return 1.0
        return min(1.0, self.kappa[domain] / (abs(self.ema[domain]) + self.eps))

    @staticmethod
    def apply_scale(advantage: Tensor, scale: Tensor) -> Tensor:
        if advantage.shape != scale.shape or bool((scale < 0).any()) or bool((scale > 1).any()):
            raise P6FormalMethodError("domain KL scale would amplify or misalign advantage")
        return advantage * scale.detach().to(device=advantage.device, dtype=advantage.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kappa": dict(self.kappa),
            "rho": self.rho,
            "eps": self.eps,
            "ema": dict(self.ema),
            "seen": dict(self.seen),
            "trigger_count": dict(self.trigger_count),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not (
            value.get("schema_version") == 1
            and dict(value.get("kappa", {})) == self.kappa
            and float(value.get("rho", -1.0)) == self.rho
            and float(value.get("eps", -1.0)) == self.eps
        ):
            raise P6FormalMethodError("domain KL resume identity differs")
        self.ema = {str(key): float(item) for key, item in dict(value["ema"]).items()}
        self.seen = {str(key): bool(item) for key, item in dict(value["seen"]).items()}
        self.trigger_count = {
            str(key): int(item) for key, item in dict(value["trigger_count"]).items()
        }


__all__ = [
    "DomainKLSafetyV1",
    "MethodRouteStateV1",
    "P6FormalMethodError",
    "TEACHERS",
    "balanced_teacher_routes",
]
