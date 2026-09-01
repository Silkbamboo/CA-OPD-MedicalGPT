"""P9 Medical-primary capability curves and preregistered statistics."""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError


FROZEN_CONTROLLER_ANCHORS = {
    "B0": {"medical_correct": 219, "general_correct": 128},
    "B1": {"medical_correct": 240, "general_correct": 139},
    "B2_step120": {"medical_correct": 217, "general_correct": 128},
}


def validate_frozen_controller_anchors(metrics: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        route: {
            "medical_correct": round(float(metrics[route]["medical_accuracy"]) * 300),
            "general_correct": round(
                float(metrics[route]["general_micro_accuracy"]) * 209
            ),
        }
        for route in FROZEN_CONTROLLER_ANCHORS
    }
    if observed != FROZEN_CONTROLLER_ANCHORS:
        raise P9ProtocolError("P9 frozen Controller anchors did not reproduce")
    return {"passed": True, "expected": FROZEN_CONTROLLER_ANCHORS, "observed": observed}


def theil_sen_slope(points: Mapping[int, int]) -> float:
    ordered = sorted((int(step), int(value)) for step, value in points.items())
    if len(ordered) < 2 or len({step for step, _ in ordered}) != len(ordered):
        raise P9ProtocolError("Theil-Sen requires at least two distinct registered steps")
    slopes = [
        (right_value - left_value) / (right_step - left_step)
        for index, (left_step, left_value) in enumerate(ordered)
        for right_step, right_value in ordered[index + 1 :]
    ]
    return float(median(slopes))


def select_best_medical_checkpoint(curves: Mapping[int, Mapping[str, Any]]) -> int:
    if not curves:
        raise P9ProtocolError("P9 checkpoint selection curve is empty")
    return min(
        (int(step) for step in curves),
        key=lambda step: (-int(curves[step]["medical_correct"]), step),
    )


def summarize_p9_controller(
    report: Mapping[str, Any], *, registered_steps: list[int]
) -> dict[str, Any]:
    metrics = report.get("metrics")
    paired = report.get("paired_vs_reference")
    if not isinstance(metrics, Mapping) or not isinstance(paired, Mapping):
        raise P9ProtocolError("P9 Controller report lacks metrics or paired statistics")
    routes = {120: "B2_step120", **{step: f"B2_step{step}" for step in registered_steps if step != 120}}
    expected = {"B0", "B1", *routes.values()}
    if set(metrics) != expected:
        raise P9ProtocolError("P9 Controller route set differs")
    anchor_gate = validate_frozen_controller_anchors(metrics)
    b0_medical = round(float(metrics["B0"]["medical_accuracy"]) * 300)
    b1_medical = round(float(metrics["B1"]["medical_accuracy"]) * 300)
    curves: dict[int, dict[str, Any]] = {}
    for step, route in routes.items():
        value = metrics[route]
        comparison = paired[route]
        medical = comparison["domains"]["medical"]
        general = comparison["domains"]["general"]
        medical_correct = round(float(value["medical_accuracy"]) * 300)
        general_correct = round(float(value["general_micro_accuracy"]) * 209)
        b0_general = round(float(metrics["B0"]["general_micro_accuracy"]) * 209)
        b1_general = round(float(metrics["B1"]["general_micro_accuracy"]) * 209)
        curves[step] = {
            "route": route,
            "medical_correct": medical_correct,
            "medical_total": 300,
            "medical_accuracy": float(value["medical_accuracy"]),
            "general_correct": general_correct,
            "general_total": 209,
            "general_micro_accuracy": float(value["general_micro_accuracy"]),
            "general_macro_accuracy": float(value["general_macro_accuracy"]),
            "general_subjects": dict(value["per_subject_accuracy"]),
            "medical_delta_correct_vs_b0": medical_correct - b0_medical,
            "medical_delta_correct_vs_b1": medical_correct - b1_medical,
            "general_delta_correct_vs_b0": general_correct - b0_general,
            "general_delta_correct_vs_b1": general_correct - b1_general,
            "medical_outcome_vs_b0": "improved" if medical_correct > b0_medical else "regressed" if medical_correct < b0_medical else "unchanged",
            "medical_outcome_vs_b1": "improved" if medical_correct > b1_medical else "regressed" if medical_correct < b1_medical else "unchanged",
            "general_outcome_vs_b0": "improved" if general_correct > b0_general else "regressed" if general_correct < b0_general else "unchanged",
            "general_outcome_vs_b1": "improved" if general_correct > b1_general else "regressed" if general_correct < b1_general else "unchanged",
            "medical_paired_bootstrap_95_ci": list(medical["bootstrap_95_ci"]),
            "medical_mcnemar": dict(medical["mcnemar"]),
            "general_paired_bootstrap_95_ci": list(general["bootstrap_95_ci"]),
            "general_mcnemar": dict(general["mcnemar"]),
            "migration_rate": (medical_correct - b0_medical) / (b1_medical - b0_medical),
        }
    return {
        "schema_version": 1,
        "artifact_kind": "p9_b2_controller_statistics",
        "baselines": {
            "B0": {"medical_correct": b0_medical, "general_correct": round(float(metrics["B0"]["general_micro_accuracy"]) * 209), **dict(metrics["B0"])},
            "B1": {"medical_correct": b1_medical, "general_correct": round(float(metrics["B1"]["general_micro_accuracy"]) * 209), **dict(metrics["B1"])},
        },
        "frozen_anchor_reproduction": anchor_gate,
        "curves": curves,
        "theil_sen_medical_correct_per_step": theil_sen_slope(
            {step: value["medical_correct"] for step, value in curves.items()}
        ),
        "best_checkpoint_step": select_best_medical_checkpoint(curves),
        "selection_metric": "Medical_correct_over_300_only",
        "tie_break": "earlier_checkpoint",
        "general_used_for_selection": False,
        "controller_access_count": 1,
        "confirmation_access_count": 0,
        "final_access_count": 0,
    }


__all__ = [
    "FROZEN_CONTROLLER_ANCHORS",
    "select_best_medical_checkpoint",
    "summarize_p9_controller",
    "theil_sen_slope",
    "validate_frozen_controller_anchors",
]
