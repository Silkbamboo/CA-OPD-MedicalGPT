"""Aggregate the preregistered P9 continuation health gate."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError


def _finite(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def evaluate_p9_common_health(
    records: Sequence[Mapping[str, Any]],
    *,
    rejected_transactions: Sequence[Mapping[str, Any]],
    checkpoint_complete_resume_eligible: bool,
    disk_free_bytes: int,
    projected_cost_cny: float,
) -> dict[str, Any]:
    by_step = {int(record["optimizer_step"]): record for record in records}
    if not all(step in by_step for step in (*range(121, 141), *range(181, 201))):
        raise P9ProtocolError("P9 common health requires steps121-140 and181-200")
    gaps = {step: abs(float(record["reverse_kl"]["mean"])) for step, record in by_step.items()}
    postcap = {
        step: median(float(row["bounded_norm"]) for row in record["ratio_v2"]["bounded_influence_v2"]["prompt_gradients"])
        for step, record in by_step.items()
    }
    early_gap = median(gaps[step] for step in range(121, 141))
    late_gap = median(gaps[step] for step in range(181, 201))
    early_grad = median(postcap[step] for step in range(121, 141))
    late_grad = median(postcap[step] for step in range(181, 201))
    prompt_samples = [row for record in records for row in record["prompt_samples"]]
    prompt_gradients = [
        row for record in records
        for row in record["ratio_v2"]["bounded_influence_v2"]["prompt_gradients"]
    ]
    cap_rate = sum(float(row["clip_scale"]) < 1.0 - 1e-12 for row in prompt_gradients) / len(prompt_gradients)
    cap_scale_median = median(float(row["clip_scale"]) for row in prompt_gradients)
    adapter_deltas = [float(record["adapter_delta_norm"]) for record in records]
    rollbacks_ok = all(
        value.get("adapter_rollback_verified") is True
        and value.get("optimizer_rollback_verified") is True
        and value.get("scheduler_rollback_verified") is True
        and value.get("rng_rollback_verified") is True
        and value.get("cursor_advanced") is False
        and value.get("sampler_advanced") is False
        for value in rejected_transactions
    )
    checks = {
        "identity_ratio_ess_finite_gate": all(
            record.get("ratio_health_v2", {}).get("accepted") is True
            and math.isfinite(float(record["loss"]))
            and math.isfinite(float(record["ess_fraction"]))
            for record in records
        ),
        "all_rejected_transactions_rolled_back": rollbacks_ok,
        "checkpoint_complete_resume_eligible": bool(checkpoint_complete_resume_eligible),
        "late_teacher_gap_ge_20pct_early": late_gap >= early_gap * 0.20,
        "late_postcap_grad_ge_20pct_early": late_grad >= early_grad * 0.20,
        "finite_nonzero_adapter_change": _finite(adapter_deltas) and min(adapter_deltas) > 0,
        "no_invalid_or_repetition": not any(row.get("invalid") or row.get("repetition") for row in prompt_samples),
        "cap_trigger_rate_lt_80pct": cap_rate < 0.80,
        "median_cap_scale_ge_0_1": cap_scale_median >= 0.10,
        "disk_safe": int(disk_free_bytes) >= 10_000_000_000,
        "projected_cost_safe": float(projected_cost_cny) <= 33.0,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "p9_common_training_health",
        "common_health_gate_passed": all(checks.values()),
        "checks": checks,
        "early_teacher_gap_median_abs": early_gap,
        "late_teacher_gap_median_abs": late_gap,
        "early_postcap_grad_median": early_grad,
        "late_postcap_grad_median": late_grad,
        "cap_trigger_rate": cap_rate,
        "median_cap_scale": cap_scale_median,
        "adapter_delta_min": min(adapter_deltas),
        "invalid_count": sum(bool(row.get("invalid")) for row in prompt_samples),
        "repetition_count": sum(bool(row.get("repetition")) for row in prompt_samples),
        "rejected_transaction_count": len(rejected_transactions),
        "disk_free_bytes": int(disk_free_bytes),
        "projected_cost_cny": float(projected_cost_cny),
        "final_access_count": 0,
    }


__all__ = ["evaluate_p9_common_health"]
