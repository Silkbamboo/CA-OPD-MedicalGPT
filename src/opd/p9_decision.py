"""Pure preregistered P9 step200/240/300 decision state machine."""

from __future__ import annotations

from typing import Any, Mapping

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError


def _health_ok(health: Mapping[str, Any]) -> bool:
    return health.get("common_health_gate_passed") is True


def _result(status: str, *, continued: bool, next_max_step: int) -> dict[str, Any]:
    return {
        "status": status,
        "continue": continued,
        "next_max_step": next_max_step,
        "decision_artifact_required_before_resume": continued,
        "general_used_for_decision": False,
    }


def decide_step200(
    medical_correct: Mapping[int, int],
    *,
    ci_low: float,
    slope: float,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    if set(medical_correct) != {120, 150, 180, 200}:
        raise P9ProtocolError("P9 step200 Medical curve differs")
    values = {step: int(value) for step, value in medical_correct.items()}
    current = values[200]
    health_ok = _health_ok(health)
    if not health_ok:
        return _result("negative_at_200", continued=False, next_max_step=200)
    if max(values[step] for step in (150, 180, 200)) >= 225 or (current > 219 and float(ci_low) > 0):
        return _result("positive_at_200", continued=False, next_max_step=200)
    if (
        current <= 217
        or (float(slope) <= 0 and all(values[step] < 219 for step in (150, 180, 200)))
    ):
        return _result("negative_at_200", continued=False, next_max_step=200)
    highest = max(values.values())
    if health_ok and float(slope) > 0 and current >= highest - 1 and (
        current >= 220 or (current == 219 and current - values[120] >= 2)
    ):
        return _result("promising_at_200_continue_300", continued=True, next_max_step=300)
    if health_ok and current == 218 and current > values[120] and float(slope) > 0:
        return _result("gray_at_200_continue_240", continued=True, next_max_step=240)
    return _result("ambiguous_at_200_stop", continued=False, next_max_step=200)


def decide_step240(
    medical_correct: Mapping[int, int],
    *,
    ci_low: float,
    slope: float,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    if set(medical_correct) != {120, 150, 180, 200, 240}:
        raise P9ProtocolError("P9 step240 Medical curve differs")
    current = int(medical_correct[240])
    if not _health_ok(health):
        return _result("negative_at_240", continued=False, next_max_step=240)
    if current >= 225 or (current > 219 and float(ci_low) > 0):
        return _result("positive_at_240", continued=False, next_max_step=240)
    highest = max(int(value) for value in medical_correct.values())
    if _health_ok(health) and current >= 220 and float(slope) > 0 and current >= highest - 1:
        return _result("promising_at_240_continue_300", continued=True, next_max_step=300)
    return _result("negative_at_240", continued=False, next_max_step=240)


def enforce_optimizer_boundary(absolute_max_step: int, *, attempted_next_step: int) -> int:
    if absolute_max_step != 300 or attempted_next_step > 300:
        raise P9ProtocolError("P9 step300 absolute optimizer boundary rejected the commit")
    return attempted_next_step


def classify_step300(*, best_medical_correct: int, paired_ci_low: float) -> str:
    correct = int(best_medical_correct)
    if correct > 219 and float(paired_ci_low) > 0:
        return "b2_dose_statistically_supported"
    if correct >= 225:
        return "b2_dose_practical_positive"
    if 220 <= correct <= 224:
        return "b2_dose_weak_positive_trend"
    return "b2_dose_negative_complete"


__all__ = ["classify_step300", "decide_step200", "decide_step240", "enforce_optimizer_boundary"]
