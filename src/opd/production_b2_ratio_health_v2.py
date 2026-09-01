"""Composite Ratio Health Protocol v2 evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.opd.production_b2_ratio_contract_v2 import validate_ratio_evidence_v2


class RatioHealthV2Error(RuntimeError):
    """A registered composite health gate rejected a candidate update."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RatioHealthV2Error(f"{label} is absent or non-finite")
    return float(value)


def evaluate_preupdate_backend_health_v2(
    evidence: Mapping[str, Any], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate fixed-token identity and backend correction before backward.

    Backend clipping is token-pooled. Per-prompt ESS is a hard gate only once
    the preregistered minimum support is present; shorter partitions remain
    visible diagnostics and cannot create a one-token denominator failure.
    """

    validate_ratio_evidence_v2(evidence)
    if thresholds.get("schema_version") != 2:
        raise RatioHealthV2Error("ratio health thresholds schema differs")
    ppo = evidence["ppo_ratio"]
    backend = evidence["backend_correction"]
    failures: list[str] = []
    diagnostics: list[str] = []
    checks = (
        ("ppo_abs_log_p99", ppo["log"]["abs_p99"], thresholds["ppo_abs_log_p99_max"]),
        ("ppo_abs_log_p999", ppo["log"]["abs_p999"], thresholds["ppo_abs_log_p999_max"]),
        ("backend_abs_log_p99", backend["raw_log"]["abs_p99"], thresholds["backend_abs_log_p99_max"]),
        ("backend_abs_log_p999", backend["raw_log"]["abs_p999"], thresholds["backend_abs_log_p999_max"]),
        ("backend_clip_fraction", backend["clip_fraction"], thresholds["backend_clip_fraction_max"]),
    )
    for label, observed, limit in checks:
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
        raise RatioHealthV2Error(
            "preupdate backend health v2 rejected: " + ",".join(failures)
        )
    return {
        "accepted": True,
        "failures": [],
        "diagnostic_warnings": diagnostics,
        "hard_gate_aggregation": {
            "backend_clip_fraction": "token_pooled",
            "pooled_ess": "token_pooled",
            "per_prompt_ess": f"minimum_{min_tokens}_valid_tokens",
        },
    }


def evaluate_ratio_health_v2(
    evidence: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
    preclip_grad_norm: float,
    relative_update_norm: float,
    ppo_clip_fraction: float,
    consecutive_warning_count: int,
) -> dict[str, Any]:
    """Return acceptance/warnings or reject with all binding failure reasons."""

    validate_ratio_evidence_v2(evidence)
    if thresholds.get("schema_version") != 2:
        raise RatioHealthV2Error("ratio health thresholds schema differs")
    preupdate = evaluate_preupdate_backend_health_v2(
        evidence, thresholds=thresholds
    )
    ppo = evidence["ppo_ratio"]
    post = evidence["post_update_policy_shift"]
    failures: list[str] = []
    warnings: list[str] = list(preupdate["diagnostic_warnings"])

    checks = (
        ("approx_kl", abs(_number(ppo["approx_kl"], "approx KL")), thresholds["approx_kl_abs_max"]),
        ("ppo_clip_fraction", ppo_clip_fraction, thresholds["ppo_clip_fraction_max"]),
        ("relative_update_norm", relative_update_norm, thresholds["relative_update_norm_max"]),
        ("post_shift_abs_log_p99", post["log"]["abs_p99"], thresholds["post_shift_abs_log_p99_max"]),
        ("post_shift_abs_log_p999", post["log"]["abs_p999"], thresholds["post_shift_abs_log_p999_max"]),
        ("tail_loss_share", post["tail"]["absolute_loss_share"], thresholds["tail_loss_share_max"]),
        ("tail_gradient_proxy_share", post["tail"]["gradient_proxy_share"], thresholds["tail_gradient_proxy_share_max"]),
    )
    for label, observed, limit in checks:
        if _number(observed, label) > _number(limit, f"{label} limit"):
            failures.append(label)

    grad = _number(preclip_grad_norm, "preclip grad norm")
    if grad > _number(thresholds["preclip_grad_norm_absolute_max"], "absolute grad cap"):
        failures.append("preclip_grad_absolute")
    median = _number(thresholds["healthy_grad_median"], "healthy grad median")
    mad = max(_number(thresholds["healthy_grad_mad"], "healthy grad MAD"), 1.0e-12)
    robust_z = max(0.0, (grad - median) / mad)
    if robust_z > _number(thresholds["preclip_grad_robust_z_max"], "grad robust-z cap"):
        failures.append("preclip_grad_robust_z")

    raw_max = _number(post["ratio"]["max"], "raw post ratio max")
    if raw_max > _number(thresholds["raw_post_ratio_max_warning_above"], "raw ratio warning"):
        warnings.append("raw_post_ratio_max")
    composite_warning = "raw_post_ratio_max" in warnings
    next_warning_count = consecutive_warning_count + 1 if composite_warning else 0
    if next_warning_count >= int(thresholds["consecutive_warning_abort_count"]):
        failures.append("consecutive_composite_warnings")

    if failures:
        raise RatioHealthV2Error("ratio health v2 rejected: " + ",".join(failures))
    return {
        "accepted": True,
        "failures": [],
        "warnings": warnings,
        "next_consecutive_warning_count": next_warning_count,
        "preclip_grad_robust_z": robust_z,
        "raw_post_ratio_max_is_diagnostic_only": True,
    }
