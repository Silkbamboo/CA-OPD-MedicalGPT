"""P7 pre-update backend health with an actual-impact tail path."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.opd.production_b2_ratio_contract_v2 import (
    RatioContractV2Error,
    validate_ratio_evidence_v2,
)


class P7BackendHealthError(RuntimeError):
    """The P7 backend identity or actual-impact contract failed closed."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P7BackendHealthError(f"{label} is absent")
    result = float(value)
    if not math.isfinite(result):
        raise P7BackendHealthError(f"{label} is non-finite")
    return result


def _require_thresholds(value: Mapping[str, Any]) -> None:
    required = {
        "ppo_abs_log_p99_max",
        "ppo_abs_log_p999_max",
        "backend_abs_log_p99_max",
        "backend_abs_log_p999_diagnostic_trigger",
        "backend_clip_fraction_max",
        "pooled_ess_floor",
        "per_prompt_ess_floor",
        "per_prompt_ess_min_tokens",
        "objective_relative_l1_change_max",
        "gradient_relative_l2_change_max",
        "gradient_cosine_min",
        "parameter_delta_relative_l2_change_max",
        "parameter_delta_cosine_min",
        "counterfactual_ppo_identity_max",
        "counterfactual_ess_floor",
    }
    if not (
        value.get("schema_version") == 3
        and value.get("protocol_id") == "p7_backend_health_v3"
        and required.issubset(value)
        and int(value["per_prompt_ess_min_tokens"]) == 32
    ):
        raise P7BackendHealthError("P7 backend health thresholds differ")


def _validate_actual_impact(
    impact: Mapping[str, Any], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if not (
        impact.get("schema_version") == 3
        and impact.get("artifact_kind") == "p7_backend_actual_impact_v3"
        and impact.get("fixed_token_identity_verified") is True
        and impact.get("candidate_committed") is False
        and impact.get("unconditional_rollback_verified") is True
        and isinstance(impact.get("production_objective"), Mapping)
        and isinstance(impact.get("accumulated_gradient"), Mapping)
        and isinstance(impact.get("adam_parameter_delta"), Mapping)
    ):
        raise P7BackendHealthError("backend actual-impact evidence is incomplete")
    failures: list[str] = []
    objective = impact["production_objective"]
    gradient = impact["accumulated_gradient"]
    delta = impact["adam_parameter_delta"]
    checks = (
        (
            "objective_relative_l1_change",
            objective.get("relative_l1_change"),
            thresholds["objective_relative_l1_change_max"],
            "max",
        ),
        (
            "gradient_relative_l2_change",
            gradient.get("relative_l2_change"),
            thresholds["gradient_relative_l2_change_max"],
            "max",
        ),
        (
            "parameter_delta_relative_l2_change",
            delta.get("relative_l2_change"),
            thresholds["parameter_delta_relative_l2_change_max"],
            "max",
        ),
        (
            "counterfactual_ppo_identity",
            impact.get("counterfactual_ppo_identity_max_abs"),
            thresholds["counterfactual_ppo_identity_max"],
            "max",
        ),
    )
    for label, observed, limit, _ in checks:
        if _number(observed, label) > _number(limit, f"{label} limit"):
            failures.append(label)
    for label, item, threshold_key in (
        ("gradient_cosine", gradient, "gradient_cosine_min"),
        ("parameter_delta_cosine", delta, "parameter_delta_cosine_min"),
    ):
        both_zero = item.get("both_zero") is True
        cosine = item.get("cosine_similarity")
        if both_zero:
            if _number(item.get("relative_l2_change"), f"{label} zero relative change") != 0.0:
                failures.append(label)
        elif _number(cosine, label) < _number(thresholds[threshold_key], f"{label} floor"):
            failures.append(label)
    production_ess = _number(
        impact.get("production_correction_ess"), "production correction ESS"
    )
    counterfactual_ess = _number(
        impact.get("counterfactual_correction_ess"), "counterfactual correction ESS"
    )
    ess_floor = _number(thresholds["counterfactual_ess_floor"], "impact ESS floor")
    if production_ess < ess_floor:
        failures.append("production_correction_ess")
    if counterfactual_ess < ess_floor:
        failures.append("counterfactual_correction_ess")
    if failures:
        raise P7BackendHealthError(
            "backend actual-impact rejected: " + ",".join(failures)
        )
    return {
        "passed": True,
        "objective_relative_l1_change": _number(
            objective["relative_l1_change"], "objective relative L1 change"
        ),
        "gradient_relative_l2_change": _number(
            gradient["relative_l2_change"], "gradient relative L2 change"
        ),
        "gradient_cosine_similarity": (
            None if gradient.get("both_zero") is True else _number(gradient["cosine_similarity"], "gradient cosine")
        ),
        "parameter_delta_relative_l2_change": _number(
            delta["relative_l2_change"], "parameter delta relative L2 change"
        ),
        "parameter_delta_cosine_similarity": (
            None if delta.get("both_zero") is True else _number(delta["cosine_similarity"], "parameter delta cosine")
        ),
        "counterfactual_ppo_identity_max_abs": _number(
            impact["counterfactual_ppo_identity_max_abs"], "counterfactual PPO identity"
        ),
        "production_correction_ess": production_ess,
        "counterfactual_correction_ess": counterfactual_ess,
        "candidate_committed": False,
        "unconditional_rollback_verified": True,
    }


def evaluate_backend_health_v3(
    evidence: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
    actual_impact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-token identity, aggregate risk and triggered impact."""

    _require_thresholds(thresholds)
    try:
        validate_ratio_evidence_v2(evidence)
    except RatioContractV2Error as error:
        raise P7BackendHealthError(f"backend identity contract rejected: {error}") from error
    ppo = evidence["ppo_ratio"]
    backend = evidence["backend_correction"]
    failures: list[str] = []
    diagnostics: list[str] = []
    for label, observed, limit in (
        ("ppo_abs_log_p99", ppo["log"]["abs_p99"], thresholds["ppo_abs_log_p99_max"]),
        ("ppo_abs_log_p999", ppo["log"]["abs_p999"], thresholds["ppo_abs_log_p999_max"]),
        ("backend_abs_log_p99", backend["raw_log"]["abs_p99"], thresholds["backend_abs_log_p99_max"]),
        ("backend_clip_fraction", backend["clip_fraction"], thresholds["backend_clip_fraction_max"]),
    ):
        if _number(observed, label) > _number(limit, f"{label} limit"):
            failures.append(label)
    pooled_ess = _number(backend["ess"]["pooled_fraction"], "pooled ESS")
    if pooled_ess < _number(thresholds["pooled_ess_floor"], "pooled ESS floor"):
        failures.append("pooled_ess")
    min_tokens = int(thresholds["per_prompt_ess_min_tokens"])
    for prompt_id, item in backend["ess"]["per_prompt"].items():
        count = int(item["token_count"])
        value = _number(item["ess_fraction"], f"per-prompt ESS {prompt_id}")
        if value < _number(thresholds["per_prompt_ess_floor"], "per-prompt ESS floor"):
            if count >= min_tokens:
                failures.append(f"per_prompt_ess:{prompt_id}")
            else:
                diagnostics.append(f"short_prompt_ess_below_floor:{prompt_id}")
    if failures:
        raise P7BackendHealthError("P7 backend health rejected: " + ",".join(failures))

    raw_p999 = _number(backend["raw_log"]["abs_p999"], "backend raw abs-log P99.9")
    trigger = _number(
        thresholds["backend_abs_log_p999_diagnostic_trigger"],
        "backend P99.9 diagnostic trigger",
    )
    impact_required = raw_p999 > trigger
    impact_result = None
    if impact_required:
        if not isinstance(actual_impact, Mapping):
            raise P7BackendHealthError(
                "backend P99.9 actual-impact qualification is required"
            )
        impact_result = _validate_actual_impact(actual_impact, thresholds=thresholds)
    return {
        "schema_version": 3,
        "protocol_id": "p7_backend_health_v3",
        "accepted": True,
        "failures": [],
        "diagnostic_warnings": diagnostics,
        "raw_backend_abs_log_p999": raw_p999,
        "raw_backend_p999_exceeded_diagnostic_trigger": impact_required,
        "raw_backend_p999_threshold_raised": False,
        "actual_impact_required": impact_required,
        "actual_impact": impact_result,
        "pooled_ess": pooled_ess,
    }


__all__ = ["P7BackendHealthError", "evaluate_backend_health_v3"]
