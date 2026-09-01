"""Fail-closed P4.3 correction calibration readiness checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from src.opd.pg_opd_contract import THREE_POLICY_ARTIFACT_PROTOCOL_VERSION
from src.opd.rollout_probability import (
    THREE_POLICY_TRAJECTORY_PROTOCOL_VERSION,
    RolloutProbabilityError,
    validate_rollout_behavior_provenance,
)


MIN_ESS_FRACTION = 0.80
MAX_CAP_FRACTION = 0.05
MAX_CURRENT_OLD_ABS_LOGPROB = 1e-4
FROZEN_ROLLOUT_IS_THRESHOLD = 2.0


@dataclass(frozen=True)
class ThreePolicyCalibrationReadiness:
    calibration_ready: bool
    opd_backend_ready: bool
    status: str
    failure_reasons: tuple[str, ...]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _summary_valid(value: Any, *, positive: bool = False) -> bool:
    names = ("mean", "std", "min", "max", "p50", "p95", "p99")
    return bool(
        isinstance(value, Mapping)
        and all(_finite_number(value.get(name)) for name in names)
        and value["std"] >= 0
        and value["min"] <= value["p50"] <= value["p95"] <= value["p99"] <= value["max"]
        and (not positive or value["min"] > 0)
    )


def _partition_metrics_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value
        and all(
            isinstance(item, Mapping)
            and _summary_valid(item, positive=True)
            and isinstance(item.get("valid_token_count"), int)
            and item["valid_token_count"] > 0
            and _summary_valid(item.get("rollout_actor_log_ratio"))
            and _summary_valid(item.get("raw_is_weight"), positive=True)
            and _summary_valid(item.get("truncated_is_weight"), positive=True)
            and _finite_number(item.get("ess"))
            and item["ess"] > 0
            and _finite_number(item.get("ess_fraction"))
            and 0 < item["ess_fraction"] <= 1
            and _finite_number(item.get("cap_fraction"))
            and 0 <= item["cap_fraction"] <= 1
            for item in value.values()
        )
    )


def _token_pooled_metrics_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and _summary_valid(value.get("rollout_actor_log_ratio"))
        and _summary_valid(value.get("raw_is_weight"), positive=True)
        and _summary_valid(value.get("truncated_is_weight"), positive=True)
        and _finite_number(value.get("ess"))
        and value["ess"] > 0
        and _finite_number(value.get("ess_fraction"))
        and 0 < value["ess_fraction"] <= 1
        and _finite_number(value.get("cap_fraction"))
        and 0 <= value["cap_fraction"] <= 1
    )


def _prompt_equal_metrics_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and _summary_valid(value, positive=True)
        and _summary_valid(value.get("rollout_actor_log_ratio"))
        and _summary_valid(value.get("raw_is_weight"), positive=True)
        and _summary_valid(value.get("truncated_is_weight"), positive=True)
        and _finite_number(value.get("ess_fraction"))
        and 0 < value["ess_fraction"] <= 1
        and _finite_number(value.get("cap_fraction"))
        and 0 <= value["cap_fraction"] <= 1
    )


def _logprob_evidence_valid(
    value: Any, *, semantic_name: str, requires_grad: bool
) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("semantic_name") == semantic_name
        and value.get("requires_grad") is requires_grad
        and isinstance(value.get("valid_token_count"), int)
        and not isinstance(value.get("valid_token_count"), bool)
        and value["valid_token_count"] > 0
        and value.get("finite_count") == value["valid_token_count"]
        and _summary_valid(value.get("summary"))
    )


def evaluate_three_policy_calibration(
    metrics: Mapping[str, Any], provenance: Mapping[str, Any]
) -> ThreePolicyCalibrationReadiness:
    """Assess the frozen pre-one-step GPU calibration gate without upgrading P4.2."""

    failures: list[str] = []
    try:
        validate_rollout_behavior_provenance(provenance)
    except RolloutProbabilityError as exc:
        failures.append(f"behavior provenance invalid: {exc}")

    if (
        metrics.get("artifact_protocol_version")
        != THREE_POLICY_ARTIFACT_PROTOCOL_VERSION
        or metrics.get("trajectory_protocol_version")
        != THREE_POLICY_TRAJECTORY_PROTOCOL_VERSION
        or metrics.get("trajectory_kind") != "fresh_full_support"
    ):
        failures.append("artifact is not the fresh P4.3 protocol")
    if metrics.get("p4_2_historical_status") != "failed_identity_mismatch":
        failures.append("P4.2 historical failure status was changed")
    for field in ("trajectory_sha256", "token_identity_sha256"):
        value = metrics.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            failures.append(f"{field} is missing or invalid")
    if metrics.get("token_identity_sha256") != provenance.get("token_identity_sha256"):
        failures.append("calibration/provenance token identity mismatch")

    expected_policies = {
        "rollout_behavior_logprob": "log_q_detached",
        "old_actor_logprob": "log_p_old_direct_forward_detached",
        "current_actor_logprob": "log_p_theta_with_gradient",
        "teacher_logprob": "same_tokens_raw_teacher_detached",
    }
    if metrics.get("policy_semantics") != expected_policies:
        failures.append("three-policy logprob semantics are incomplete or mixed")
    expected_ratios = {
        "rollout_correction": "old_actor_minus_rollout_behavior",
        "ppo_ratio": "current_actor_minus_old_actor",
    }
    if metrics.get("ratio_semantics") != expected_ratios:
        failures.append("correction and PPO ratio semantics are not separate")

    for field, semantic_name, requires_grad in (
        ("rollout_behavior_logprob", "log_q", False),
        ("old_actor_logprob", "log_p_old", False),
        ("current_actor_logprob", "log_p_theta", True),
        ("teacher_logprob", "log_p_teacher", False),
    ):
        if not _logprob_evidence_valid(
            metrics.get(field),
            semantic_name=semantic_name,
            requires_grad=requires_grad,
        ):
            failures.append(f"{field} evidence is incomplete or semantically invalid")

    isolation = metrics.get("isolation")
    if not (
        isinstance(isolation, Mapping)
        and set(isolation)
        == {"final_access", "controller_access", "confirmation_access", "label_access"}
        and all(value is False for value in isolation.values())
    ):
        failures.append("evaluation/final isolation evidence is invalid")

    identity = metrics.get("current_pre_vs_old_actor")
    if not (
        isinstance(identity, Mapping)
        and _finite_number(identity.get("max_abs"))
        and 0 <= identity["max_abs"] <= MAX_CURRENT_OLD_ABS_LOGPROB
    ):
        failures.append("current/old actor pre-update identity gate failed")

    q_old = metrics.get("q_vs_old_actor")
    if not (
        isinstance(q_old, Mapping)
        and set(q_old) == {"mae", "abs_p95", "max_abs"}
        and all(
            _finite_number(q_old.get(name)) and q_old[name] >= 0
            for name in ("mae", "abs_p95", "max_abs")
        )
        and q_old["mae"] <= q_old["abs_p95"] <= q_old["max_abs"]
    ):
        failures.append("q/old actor MAE/P95/max diagnostics are missing or invalid")

    nonfinite = metrics.get("nonfinite_counts")
    expected_nonfinite = {
        "rollout_behavior_logprob",
        "old_actor_logprob",
        "current_actor_logprob",
        "teacher_logprob",
        "raw_is_weight",
        "truncated_is_weight",
        "ppo_ratio",
    }
    if not (
        isinstance(nonfinite, Mapping)
        and set(nonfinite) == expected_nonfinite
        and all(isinstance(value, int) and not isinstance(value, bool) and value == 0 for value in nonfinite.values())
    ):
        failures.append("three-policy or ratio nonfinite counts are nonzero/incomplete")

    correction = metrics.get("rollout_correction")
    required_correction = {
        "rollout_is_threshold",
        "rollout_actor_log_ratio",
        "raw_is_weight",
        "truncated_is_weight",
        "ess",
        "ess_fraction",
        "cap_fraction",
        "per_prompt",
        "per_source",
        "token_pooled",
        "prompt_equal",
    }
    correction_complete = bool(
        isinstance(correction, Mapping) and required_correction.issubset(correction)
    )
    if not correction_complete:
        failures.append("correction metrics are missing or incomplete")
    else:
        if correction.get("rollout_is_threshold") != FROZEN_ROLLOUT_IS_THRESHOLD:
            failures.append("rollout IS threshold differs from frozen value 2.0")
        if not _summary_valid(correction.get("rollout_actor_log_ratio")):
            failures.append("rollout/actor log-ratio summary is invalid")
        if not _summary_valid(correction.get("raw_is_weight"), positive=True):
            failures.append("raw IS weight summary is invalid")
        if not _summary_valid(correction.get("truncated_is_weight"), positive=True):
            failures.append("truncated IS weight summary is invalid")
        ess_fraction = correction.get("ess_fraction")
        if not (
            _finite_number(correction.get("ess"))
            and correction["ess"] > 0
            and _finite_number(ess_fraction)
            and MIN_ESS_FRACTION <= ess_fraction <= 1
        ):
            failures.append("ESS fraction is below 0.80 or invalid")
        cap_fraction = correction.get("cap_fraction")
        if not (
            _finite_number(cap_fraction) and 0 <= cap_fraction <= MAX_CAP_FRACTION
        ):
            failures.append("cap fraction exceeds 0.05 or is invalid")
        if not _partition_metrics_valid(correction.get("per_prompt")):
            failures.append("per-prompt correction metrics are invalid")
        sources = correction.get("per_source")
        if not (
            _partition_metrics_valid(sources)
            and {"medical_opd_o1", "medical_opd_cmb"}.issubset(sources)
        ):
            failures.append("Medical-O1/CMB correction metrics are missing or invalid")
        if not _token_pooled_metrics_valid(correction.get("token_pooled")):
            failures.append("token-pooled correction diagnostics are invalid")
        if not _prompt_equal_metrics_valid(correction.get("prompt_equal")):
            failures.append("prompt-equal correction diagnostics are invalid")

    ppo = metrics.get("ppo")
    if not (
        isinstance(ppo, Mapping)
        and _summary_valid(ppo.get("pre_ratio"), positive=True)
        and ppo["pre_ratio"]["min"] >= math.exp(-MAX_CURRENT_OLD_ABS_LOGPROB)
        and ppo["pre_ratio"]["max"] <= math.exp(MAX_CURRENT_OLD_ABS_LOGPROB)
        and _finite_number(ppo.get("pre_clip_fraction"))
        and 0 <= ppo["pre_clip_fraction"] <= 1
    ):
        failures.append("PPO pre-ratio or clip metrics are invalid")

    trainable = metrics.get("trainable_parameter_names")
    if not (
        isinstance(trainable, list)
        and trainable
        and all(isinstance(name, str) and "lora" in name.lower() for name in trainable)
        and metrics.get("base_parameter_versions_unchanged") is True
    ):
        failures.append("trainable ownership is not LoRA-only or Base changed")

    return ThreePolicyCalibrationReadiness(
        calibration_ready=not failures,
        opd_backend_ready=False,
        status=(
            "ready_for_corrected_medical_one_step"
            if not failures
            else "blocked_three_policy_calibration"
        ),
        failure_reasons=tuple(failures),
    )
