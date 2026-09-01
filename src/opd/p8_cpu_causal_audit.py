"""P8 label-free causal audit over immutable P5/P7 training artifacts.

This module never imports torch, a model loader, an evaluator, or a data-label
provider.  It reports unavailable historical diagnostics explicitly instead of
reconstructing values that were not persisted.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WINDOWS = ("step1-30", "step31-60", "step61-90", "step91-120")
SOURCE_TEACHER = {
    "medical_opd_o1": "medical_teacher",
    "medical_opd_cmb": "medical_teacher",
    "general_anchors": "base_teacher",
}
NOT_RECONSTRUCTIBLE = {
    "status": "not_reconstructible",
    "reason": "required per-token or vector-valued historical evidence was not persisted",
}


class P8CausalAuditError(RuntimeError):
    """An immutable input is incomplete or internally inconsistent."""


def classify_window(step: int) -> str:
    if not 1 <= int(step) <= 120:
        raise P8CausalAuditError("P8 audit step must be in [1, 120]")
    return WINDOWS[(int(step) - 1) // 30]


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * float(probability)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float], *, percentiles: Sequence[int] = (50, 90, 95, 99)) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, Any] = {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "std": statistics.pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }
    for percentile in percentiles:
        result[f"p{percentile}"] = _quantile(finite, percentile / 100.0)
    return result


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def join_prompt_evidence(
    *,
    step: int,
    teacher: str | None,
    prompt_samples: Sequence[Mapping[str, Any]],
    per_prompt_advantage: Mapping[str, Any],
    prompt_gradients: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gradient_by_hash = {str(item["sample_id_sha256"]): item for item in prompt_gradients}
    if len(gradient_by_hash) != len(prompt_gradients):
        raise P8CausalAuditError("duplicate prompt gradient identity")
    rows: list[dict[str, Any]] = []
    for sample in prompt_samples:
        sample_id = str(sample["sample_id"])
        source = str(sample["source"])
        gradient = gradient_by_hash.get(_sha_text(sample_id))
        if gradient is None:
            raise P8CausalAuditError(f"missing prompt gradient for step {step}")
        if str(gradient.get("source")) != source:
            raise P8CausalAuditError(f"source drift for step {step}")
        if sample_id not in per_prompt_advantage:
            raise P8CausalAuditError(f"missing prompt advantage for step {step}")
        rows.append(
            {
                "step": int(step),
                "window": classify_window(step),
                "source": source,
                "teacher": teacher or SOURCE_TEACHER.get(source, "unknown_teacher"),
                "completion_tokens": int(sample.get("generated_tokens", 0)),
                "eos": bool(sample.get("eos", False)),
                "truncated": bool(sample.get("truncated", False)),
                "empty": bool(sample.get("empty", False)),
                "invalid": bool(sample.get("invalid", False) or sample.get("non_finite", False) or sample.get("repetition", False)),
                "advantage_mean": float(per_prompt_advantage[sample_id]),
                "raw_gradient_norm": float(gradient["raw_norm"]),
                "bounded_gradient_norm": float(gradient["bounded_norm"]),
                "cap_scale": float(gradient["clip_scale"]),
            }
        )
    if set(gradient_by_hash) != {_sha_text(str(sample["sample_id"])) for sample in prompt_samples}:
        raise P8CausalAuditError(f"extra prompt gradient for step {step}")
    return rows


def _aggregate_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completion = [int(row["completion_tokens"]) for row in rows]
    advantage = [float(row["advantage_mean"]) for row in rows]
    raw = [float(row["raw_gradient_norm"]) for row in rows]
    bounded = [float(row["bounded_gradient_norm"]) for row in rows]
    scales = [float(row["cap_scale"]) for row in rows]
    triggered = [scale for scale in scales if scale < 1.0 - 1e-12]
    completion_distribution = _distribution(completion)
    completion_distribution["total"] = sum(completion)
    return {
        "prompt_count": len(rows),
        "completion_count": len(rows),
        "completion_tokens": completion_distribution,
        "generation_health": {
            "eos_count": sum(bool(row["eos"]) for row in rows),
            "truncated_count": sum(bool(row["truncated"]) for row in rows),
            "empty_count": sum(bool(row["empty"]) for row in rows),
            "invalid_count": sum(bool(row["invalid"]) for row in rows),
        },
        "advantage_prompt_equal": _distribution(advantage, percentiles=(50, 90, 99)),
        "positive_advantage_prompt_ratio": (sum(value > 0 for value in advantage) / len(advantage)) if advantage else None,
        "raw_per_prompt_gradient_norm": _distribution(raw),
        "bounded_per_prompt_gradient_norm": _distribution(bounded),
        "cap": {
            "trigger_count": len(triggered),
            "trigger_rate": len(triggered) / len(scales) if scales else None,
            "scale_all": _distribution(scales),
            "scale_triggered": _distribution(triggered),
        },
        "cumulative_effective_update_proxy": {
            "definition": "sum(per_prompt_bounded_gradient_norm); a scalar budget proxy, not a vector update norm",
            "value": sum(bounded),
        },
        "answer_first_token_advantage": dict(NOT_RECONSTRUCTIBLE),
        "body_token_advantage": dict(NOT_RECONSTRUCTIBLE),
    }


def aggregate_prompt_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["teacher"]), str(row["window"]))].append(row)
    result: dict[str, Any] = {}
    for (source, teacher, window), bucket in sorted(grouped.items()):
        result.setdefault(source, {}).setdefault(teacher, {})[window] = _aggregate_bucket(bucket)
    return result


def evaluate_cap_bias(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    medical = [row for row in rows if str(row.get("source", "")).startswith("medical_opd_")]
    if not medical:
        return {"status": "not_available", "reason": "no medical prompt evidence"}
    ordered = sorted(medical, key=lambda row: abs(float(row["advantage_mean"])))
    quartile_size = max(1, math.ceil(len(ordered) * 0.25))
    high = ordered[-quartile_size:]
    overall_triggered = [row for row in medical if float(row["cap_scale"]) < 1.0 - 1e-12]
    high_triggered = [row for row in high if float(row["cap_scale"]) < 1.0 - 1e-12]
    overall_rate = len(overall_triggered) / len(medical)
    high_rate = len(high_triggered) / len(high)
    triggered_scales = [float(row["cap_scale"]) for row in overall_triggered]
    return {
        "status": "observed",
        "teacher_gap_definition": "abs(per_prompt_mean_beta_scaled_teacher_minus_student_logprob)",
        "medical_prompt_count": len(medical),
        "high_gap_quartile_count": len(high),
        "overall_cap_trigger_rate": overall_rate,
        "high_gap_quartile_cap_trigger_rate": high_rate,
        "high_to_overall_trigger_ratio": high_rate / overall_rate if overall_rate else None,
        "median_triggered_cap_scale": _quantile(triggered_scales, 0.5),
        "high_gap_raw_gradient_norm": _distribution(float(row["raw_gradient_norm"]) for row in high),
        "high_gap_bounded_gradient_norm": _distribution(float(row["bounded_gradient_norm"]) for row in high),
    }


def audit_prompt_equal_reduction(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = 0
    for item in evidence:
        passed = (
            int(item.get("effective_batch_size", -1)) == 4
            and item.get("prompt_equal_scalar_loss_unchanged") is True
            and len(item.get("prompt_gradients", [])) == 4
        )
        failures += not passed
    return {
        "passed": bool(evidence) and failures == 0,
        "audited_step_count": len(evidence),
        "failed_step_count": failures,
        "reduction_order": "valid-token mean -> trajectory/group mean -> prompt mean -> equal prompt batch mean",
        "source_code_contract": "src/opd/production_b2_objective_reducer_v2.py:canonical_hierarchical_reduction",
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P8CausalAuditError(f"JSON root is not an object: {path}")
    return value


def _b2_step_path(root: Path, step: int) -> Path:
    matches = sorted((root / "b2_steps").glob(f"step_{step - 1:02d}_v*_to_v*.json"))
    if len(matches) != 1:
        raise P8CausalAuditError(f"expected one kernel step for {step}, found {len(matches)}")
    return matches[0]


def audit_run(run_id: str, root: str | Path) -> dict[str, Any]:
    run_root = Path(root)
    prompt_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    bounded_evidence: list[dict[str, Any]] = []
    isolation_violations: list[int] = []
    for step in range(1, 121):
        formal = _load_json(run_root / "formal_steps" / f"step_{step:03d}.json")
        ratio_doc = _load_json(run_root / "ratio_evidence_v2" / f"step_{step:03d}.json")
        kernel = _load_json(_b2_step_path(run_root, step))
        ratio = ratio_doc["ratio_evidence"]
        bounded = ratio["bounded_influence_v2"]
        bounded_evidence.append(bounded)
        advantage = kernel["reconstruction_telemetry"]["advantage"]
        prompt_rows.extend(
            join_prompt_evidence(
                step=step,
                teacher=None,
                prompt_samples=formal["prompt_samples"],
                per_prompt_advantage=advantage["per_prompt_mean"],
                prompt_gradients=bounded["prompt_gradients"],
            )
        )
        isolation = formal.get("isolation", {})
        if any(bool(isolation.get(name, False)) for name in ("final_access", "controller_access", "confirmation_access", "label_access")):
            isolation_violations.append(step)
        post_ratio = ratio["post_update_policy_shift"]["ratio"]
        step_rows.append(
            {
                "step": step,
                "window": classify_window(step),
                "teacher_student_logprob_gap_mean": float(formal["teacher_logprob"]["mean"]) - float(formal["p_old_logprob"]["mean"]),
                "advantage_token": {
                    "count": int(advantage["count"]),
                    "mean": float(advantage["mean"]),
                    "std": float(advantage["std"]),
                    "p50": float(advantage["quantiles"]["p50"]),
                    "p90": None,
                    "p99": float(advantage["quantiles"]["p99"]),
                    "positive_count": int(advantage["positive_count"]),
                    "positive_ratio": int(advantage["positive_count"]) / int(advantage["count"]),
                },
                "reverse_kl_mean": float(formal["reverse_kl"]["mean"]),
                "post_cap_aggregate_norm": float(bounded["accumulated_pre_global_clip_norm"]),
                "global_gradient_norm_before_clip": float(formal["gradient_norm_before_clip"]),
                "global_gradient_norm_after_clip": float(formal["gradient_norm"]),
                "global_clip_triggered": float(formal["gradient_norm_before_clip"]) > 1.0 + 1e-6,
                "aggregate_direction_cosine_raw_to_post_cap": dict(NOT_RECONSTRUCTIBLE),
                "aggregate_direction_cosine_pre_to_post_global_clip": 1.0,
                "adapter_delta_norm": float(formal["adapter_delta_norm"]),
                "post_update_ratio": {
                    "p99": float(post_ratio["p99"]),
                    "p999": float(post_ratio["p999"]),
                    "max": float(post_ratio["max"]),
                },
                "ess_fraction": float(ratio["backend_correction"]["ess"]["pooled_fraction"]),
                "valid_token_count": int(formal["valid_token_count"]),
            }
        )
    return {
        "run_id": run_id,
        "root": str(run_root),
        "root_read_only": True,
        "accepted_steps": 120,
        "prompt_rows": prompt_rows,
        "by_source_teacher_window": aggregate_prompt_rows(prompt_rows),
        "step_rows": step_rows,
        "prompt_equal_reduction": audit_prompt_equal_reduction(bounded_evidence),
        "cap_bias": evaluate_cap_bias(prompt_rows),
        "isolation": {
            "training_diagnostic_process": "label_free_p8_cpu_causal_audit",
            "violating_steps": isolation_violations,
            "final_access_count": 0 if not isolation_violations else None,
            "label_access_count": 0 if not isolation_violations else None,
        },
    }


def summarize_step_windows(step_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for window in WINDOWS:
        rows = [row for row in step_rows if row["window"] == window]
        result[window] = {
            "teacher_student_logprob_gap_mean": _distribution(row["teacher_student_logprob_gap_mean"] for row in rows),
            "reverse_kl": _distribution(row["reverse_kl_mean"] for row in rows),
            "advantage_token_mean": _distribution(row["advantage_token"]["mean"] for row in rows),
            "advantage_token_positive_ratio": _distribution(row["advantage_token"]["positive_ratio"] for row in rows),
            "post_cap_aggregate_norm": _distribution(row["post_cap_aggregate_norm"] for row in rows),
            "global_gradient_norm_before_clip": _distribution(row["global_gradient_norm_before_clip"] for row in rows),
            "global_gradient_norm_after_clip": _distribution(row["global_gradient_norm_after_clip"] for row in rows),
            "global_clip_trigger_rate": sum(bool(row["global_clip_triggered"]) for row in rows) / len(rows),
            "adapter_delta_norm": _distribution(row["adapter_delta_norm"] for row in rows),
            "post_update_ratio_p99": _distribution(row["post_update_ratio"]["p99"] for row in rows),
            "post_update_ratio_p999": _distribution(row["post_update_ratio"]["p999"] for row in rows),
            "ess": _distribution(row["ess_fraction"] for row in rows),
        }
    return result


def effective_update_budget(run: Mapping[str, Any]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    prompts: dict[str, int] = defaultdict(int)
    for row in run["prompt_rows"]:
        source = str(row["source"])
        totals[source] += float(row["bounded_gradient_norm"])
        prompts[source] += 1
    grand = sum(totals.values())
    medical = sum(value for source, value in totals.items() if source.startswith("medical_opd_"))
    return {
        "proxy_definition": "sum(per_prompt_bounded_gradient_norm); direction-free scalar effective-update budget",
        "by_source": {
            source: {"prompt_count": prompts[source], "budget": value, "budget_share": value / grand if grand else None}
            for source, value in sorted(totals.items())
        },
        "medical_budget_share": medical / grand if grand else None,
        "total_budget": grand,
    }


def audit_all(run_roots: Mapping[str, str | Path]) -> dict[str, Any]:
    runs = {run_id: audit_run(run_id, root) for run_id, root in run_roots.items()}
    for run in runs.values():
        run["by_step_window"] = summarize_step_windows(run["step_rows"])
        run["checkpoint_trend"] = {
            "step0": {"student": "fresh_base_plus_zero_effect_lora_v0", "metrics": dict(NOT_RECONSTRUCTIBLE)},
            **{f"step{step}": next(row for row in run["step_rows"] if row["step"] == step) for step in (30, 60, 90, 120)},
        }
    return {
        "artifact_kind": "p8_cpu_causal_audit",
        "schema_version": 1,
        "status": "complete_with_explicit_historical_limits",
        "scope": "historical_training_artifacts_only_no_model_load_no_final_no_controller_labels",
        "runs": runs,
        "effective_update_budget": {run_id: effective_update_budget(run) for run_id, run in runs.items()},
        "historical_limits": {
            "answer_first_vs_body_token_advantage": dict(NOT_RECONSTRUCTIBLE),
            "raw_vs_post_cap_aggregate_direction_cosine": dict(NOT_RECONSTRUCTIBLE),
            "per_source_token_positive_advantage_ratio": dict(NOT_RECONSTRUCTIBLE),
            "causal_interpretation": "Scalar norm sums are update-budget proxies and do not recover vector cancellation.",
        },
        "restricted_access": {
            "controller_label_access_count": 0,
            "confirmation_access_count": 0,
            "final_access_count": 0,
        },
    }


__all__ = [
    "P8CausalAuditError",
    "aggregate_prompt_rows",
    "audit_all",
    "audit_prompt_equal_reduction",
    "audit_run",
    "classify_window",
    "effective_update_budget",
    "evaluate_cap_bias",
    "join_prompt_evidence",
    "summarize_step_windows",
]
