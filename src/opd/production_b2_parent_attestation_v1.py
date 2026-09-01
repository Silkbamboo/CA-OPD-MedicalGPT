"""Disk-derived P4.8g parent health recomputation for formal B2."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence

from src.opd.production_b2_formal_v1 import FormalB2Error


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise FormalB2Error("parent metric distribution is empty or non-finite")
    return {
        "min": min(float(value) for value in values),
        "mean": mean(float(value) for value in values),
        "p05": _quantile(values, 0.05),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(float(value) for value in values),
    }


def summarize_parent_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != 20:
        raise FormalB2Error("P4.8g parent must contain exactly 20 step records")
    metric_values: dict[str, list[float]] = {
        name: []
        for name in (
            "loss",
            "objective",
            "reverse_kl_mean",
            "reverse_kl_std",
            "advantage_mean",
            "advantage_std",
            "ratio_mean",
            "ratio_max",
            "ratio_p95",
            "ratio_p99",
            "clip_fraction",
            "ess_fraction",
            "gradient_norm",
            "adapter_delta_norm",
            "step_seconds",
            "gpu0_peak",
            "gpu1_peak",
        )
    }
    source_counts = {"medical_opd_o1": 0, "medical_opd_cmb": 0}
    source_advantage: dict[str, list[float]] = {
        "medical_opd_o1": [],
        "medical_opd_cmb": [],
    }
    finish = {
        key: 0
        for key in (
            "eos",
            "truncated",
            "invalid",
            "empty",
            "non_finite",
            "repetition",
            "unexpected_think_tag",
        )
    }
    completion_lengths: list[float] = []
    truncation_by_source = {"medical_opd_o1": 0, "medical_opd_cmb": 0}
    previous_sha: str | None = None
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    nonzero_counts: list[int] = []
    rows_by_step: list[list[Mapping[str, Any]]] = []
    for expected_step, record in enumerate(records, start=1):
        before = record.get("sampler_adapter_sha256")
        after = record.get("trainer_authority_sha256")
        if not (
            record.get("optimizer_step") == expected_step
            and record.get("policy_version") == expected_step - 1
            and record.get("next_policy_version") == expected_step
            and before == record.get("input_trainer_authority_sha256")
            and after == record.get("runtime_adapter_sha256")
            == record.get("fresh_adapter_sha256")
            and before != after
            and (previous_sha is None or before == previous_sha)
        ):
            raise FormalB2Error("P4.8g policy/adapter identity chain differs")
        previous_sha = str(after)
        nonzero = int(record.get("nonzero_update_tensor_count", 0))
        if nonzero <= 0 or float(record.get("adapter_delta_norm", 0.0)) <= 0:
            raise FormalB2Error("P4.8g step lacks a nonzero LoRA update")
        nonzero_counts.append(nonzero)
        if int(record.get("teacher_gradient_tensor_count", -1)) != 0 or int(
            record.get("base_gradient_tensor_count", -1)
        ) != 0:
            raise FormalB2Error("P4.8g Teacher/Base gradient gate differs")
        isolation = record.get("isolation")
        if not isinstance(isolation, Mapping) or any(bool(isolation.get(key)) for key in isolation):
            raise FormalB2Error("P4.8g restricted-access evidence differs")
        rows = record.get("prompt_samples")
        if not isinstance(rows, list) or len(rows) != 4:
            raise FormalB2Error("P4.8g prompt batch is not four")
        rows_by_step.append(rows)
        step_sources = [row.get("source") for row in rows]
        if sorted(step_sources) != [
            "medical_opd_cmb",
            "medical_opd_cmb",
            "medical_opd_o1",
            "medical_opd_o1",
        ]:
            raise FormalB2Error("P4.8g per-step source ratio differs")
        for row in rows:
            source = str(row["source"])
            sample_id = str(row["sample_id"])
            content_hash = str(row["content_hash"])
            if sample_id in seen_ids or content_hash in seen_hashes:
                raise FormalB2Error("P4.8g reused a prompt identity")
            seen_ids.add(sample_id)
            seen_hashes.add(content_hash)
            source_counts[source] += 1
            completion_lengths.append(float(row["generated_tokens"]))
            for key in finish:
                finish[key] += int(bool(row.get(key, False)))
            truncation_by_source[source] += int(bool(row.get("truncated")))
        telemetry = record.get("reconstruction_telemetry", {})
        update = telemetry.get("optimizer_update", {}) if isinstance(telemetry, Mapping) else {}
        ratio = update.get("ppo_ratio_post", {}) if isinstance(update, Mapping) else {}
        advantage_telemetry = telemetry.get("advantage", {}) if isinstance(telemetry, Mapping) else {}
        per_source = (
            advantage_telemetry.get("per_source_mean", {})
            if isinstance(advantage_telemetry, Mapping)
            else {}
        )
        for source in source_advantage:
            if source in per_source:
                source_advantage[source].append(float(per_source[source]))
        metric_values["loss"].append(float(record["loss"]))
        metric_values["objective"].append(float(record["objective"]))
        metric_values["reverse_kl_mean"].append(float(record["reverse_kl"]["mean"]))
        metric_values["reverse_kl_std"].append(float(record["reverse_kl"]["std"]))
        metric_values["advantage_mean"].append(float(record["advantage"]["mean"]))
        metric_values["advantage_std"].append(float(record["advantage"]["std"]))
        metric_values["ratio_mean"].append(float(ratio.get("mean", record["importance_ratio"]["mean"])))
        metric_values["ratio_max"].append(float(ratio.get("max", 1.0)))
        metric_values["ratio_p95"].append(float(ratio.get("p95", 1.0)))
        metric_values["ratio_p99"].append(float(ratio.get("p99", 1.0)))
        metric_values["clip_fraction"].append(float(record["ppo_clip_fraction"]))
        metric_values["ess_fraction"].append(float(record["ess_fraction"]))
        metric_values["gradient_norm"].append(float(record["gradient_norm"]))
        metric_values["adapter_delta_norm"].append(float(record["adapter_delta_norm"]))
        metric_values["step_seconds"].append(float(record["timings_seconds"]["step"]))
        metric_values["gpu0_peak"].append(float(record["gpu_memory_bytes"]["gpu0_peak"]))
        metric_values["gpu1_peak"].append(float(record["gpu_memory_bytes"]["gpu1_peak"]))
    if min(metric_values["advantage_std"]) <= 0:
        raise FormalB2Error("P4.8g advantage degenerated to a constant")
    length_windows: list[dict[str, Any]] = []
    for start in range(17):
        window_rows = [row for step_rows in rows_by_step[start : start + 4] for row in step_rows]
        source_totals = {
            source: sum(row.get("source") == source for row in window_rows)
            for source in source_counts
        }
        source_truncated = {
            source: sum(
                row.get("source") == source and bool(row.get("truncated"))
                for row in window_rows
            )
            for source in source_counts
        }
        overall_rate = sum(bool(row.get("truncated")) for row in window_rows) / 16
        source_rates = {
            source: source_truncated[source] / source_totals[source]
            for source in source_counts
        }
        length_windows.append(
            {
                "start_step": start + 1,
                "end_step": start + 4,
                "overall_rate": overall_rate,
                "source_rates": source_rates,
                "passed": overall_rate <= 0.20
                and all(value <= 0.20 for value in source_rates.values()),
            }
        )
    if not all(window["passed"] for window in length_windows):
        raise FormalB2Error("P4.8g rolling 1024 truncation gate differs")
    distributions = {key: _distribution(values) for key, values in metric_values.items()}
    return {
        "passed": True,
        "step_count": 20,
        "policy_chain": {"initial": 0, "final": 20},
        "source_counts": source_counts,
        "unique_sample_id_count": len(seen_ids),
        "unique_content_hash_count": len(seen_hashes),
        "nonzero_update_tensor_count": _distribution(nonzero_counts),
        **distributions,
        "completion_length": _distribution(completion_lengths),
        "finish_counts": finish,
        "truncation": {
            "overall_rate": finish["truncated"] / 80,
            "medical_opd_o1_rate": truncation_by_source["medical_opd_o1"] / 40,
            "medical_opd_cmb_rate": truncation_by_source["medical_opd_cmb"] / 40,
        },
        "length_gate": {
            "selected_response_length": 1024,
            "window_steps": 4,
            "window_count": len(length_windows),
            "passed": True,
        },
        "length_windows": length_windows,
        "per_source_advantage_mean": {
            source: (_distribution(values) if values else {"status": "not_recorded"})
            for source, values in source_advantage.items()
        },
        "per_source_objective": {
            "status": "not_recorded_not_reconstructible",
            "reason": "privacy-safe P4.8g step aggregates omit per-source corrected objective",
        },
    }


def derive_rolling_safety_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze conservative bounds from all 20 steps plus fixed safety caps."""

    if summary.get("passed") is not True or summary.get("step_count") != 20:
        raise FormalB2Error("rolling gates require a passing 20-step parent")
    ess_min = float(summary["ess_fraction"]["min"])
    ratio_max = float(summary["ratio_max"]["max"])
    ratio_p99 = float(summary["ratio_p99"]["max"])
    clip_max = float(summary["clip_fraction"]["max"])
    grad_max = float(summary["gradient_norm"]["max"])
    return {
        "schema_version": 1,
        "artifact_kind": "p5_formal_b2_rolling_safety_gates_v1",
        "written_before_formal_results": True,
        "window_steps": 4,
        "derivation": "P4.8g all-step min/max/p95/p99 with fixed safety caps",
        "ess_fraction": {
            "calibration_min": ess_min,
            "abort_below": max(0.90, ess_min - 0.02),
        },
        "ratio_max": {
            "calibration_max": ratio_max,
            # Raw post-update PPO ratios are diagnostic and are not the
            # importance-cap value.  P4.8g contains one finite 3.2167 outlier,
            # so the fixed emergency ceiling must sit above that observation.
            "abort_above": max(5.0, ratio_max * 1.25),
        },
        "ratio_p99": {
            "calibration_max": ratio_p99,
            "abort_above": min(1.8, max(1.5, ratio_p99 + 0.15)),
        },
        "clip_fraction": {
            "calibration_max": clip_max,
            "abort_above": min(0.20, max(0.10, clip_max + 0.05)),
        },
        "gradient_norm_before_clip": {
            "calibration_observed_max_after_clip": grad_max,
            "finite_required": True,
            "after_clip_abort_above": 1.0001,
        },
        "truncation": {
            "window_steps": 4,
            "consecutive_failing_windows": 2,
            "overall_abort_above": 0.20,
            "per_source_abort_above": 0.20,
        },
        "registry_model_count": {"growth_windows_allowed": 0},
    }


def validate_parent_supporting_evidence(
    summary: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate P4.8g evidence that does not live in per-step metrics."""

    length = evidence.get("length_gate")
    canary = evidence.get("canary")
    index = evidence.get("checkpoint_index")
    v10 = evidence.get("v10_reload")
    v20 = evidence.get("v20_reload")
    cleanup = evidence.get("cleanup")
    if not (
        summary.get("passed") is True
        and summary.get("step_count") == 20
        and isinstance(length, Mapping)
        and length.get("passed") is True
        and length.get("selected_response_length") == 1024
        and isinstance(canary, Mapping)
        and canary.get("passed") is True
        and canary.get("fixture_valid_completion_tokens_by_prompt") == [1024] * 4
        and len(canary.get("minimum_free_bytes_by_gpu", [])) == 2
        and min(int(value) for value in canary["minimum_free_bytes_by_gpu"])
        > 1024**3
        and canary.get("teacher_gradient_tensor_count") == 0
        and canary.get("base_gradient_tensor_count") == 0
    ):
        raise FormalB2Error("P4.8g 1024 length/canary supporting evidence differs")
    checkpoints = index.get("checkpoints") if isinstance(index, Mapping) else None
    versions = (
        [int(item.get("logical_version", -1)) for item in checkpoints]
        if isinstance(checkpoints, list)
        else []
    )
    if not (
        versions == [5, 10, 15, 20]
        and all(item.get("resume_eligible") is True for item in checkpoints)
        and evidence.get("checkpoint_file_sha_passed") is True
        and isinstance(v10, Mapping)
        and v10.get("logical_version") == 10
        and v10.get("optimizer_state_restored") is True
        and v10.get("scheduler_state_restored") is True
        and v10.get("rng_state_restored") is True
        and isinstance(v20, Mapping)
        and v20.get("logical_version") == 20
        and v20.get("same_path_max_gap") == 0.0
        and v20.get("finite_rate") == 1.0
        and v20.get("tensor_count") == 504
    ):
        raise FormalB2Error("P4.8g checkpoint/reload supporting evidence differs")
    if not (
        isinstance(cleanup, Mapping)
        and cleanup.get("cleanup_complete") is True
        and cleanup.get("gpu_memory_used_mib") == [0, 0]
        and cleanup.get("compute_pids") == []
        and isinstance(cleanup.get("isolation"), Mapping)
        and not any(bool(value) for value in cleanup["isolation"].values())
        and evidence.get("cleanup_bound_by_final_index") is True
        and evidence.get("failure_artifacts") == []
    ):
        raise FormalB2Error(
            "P4.8g cleanup/failure isolation evidence is absent or unbound"
        )
    return {
        "passed": True,
        "selected_response_length": 1024,
        "canary_minimum_free_bytes_by_gpu": list(
            canary["minimum_free_bytes_by_gpu"]
        ),
        "checkpoint_versions": versions,
        "v10_reload_passed": True,
        "v20_fresh_reload_passed": True,
        "cleanup_bound_by_final_index": True,
        "failure_artifact_count": 0,
        "restricted_access_count": 0,
    }


__all__ = [
    "derive_rolling_safety_gates",
    "summarize_parent_records",
    "validate_parent_supporting_evidence",
]
