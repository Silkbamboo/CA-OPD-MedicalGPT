"""CPU-safe contract for package-bound B2 Medical OPD calibration.

This module contains no model, CUDA, Transformers, PEFT or trainer imports.  It
validates the privacy-safe evidence emitted by the existing production kernel
and is the authoritative disk-finalizer contract for exactly twenty updates.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import mean
from typing import Any, Mapping, Sequence


B2_CALIBRATION_STEPS = 20
SELECTED_RESPONSE_LENGTH = 768
SUPPORTED_RESPONSE_LENGTHS = (768, 1024)
FRESH_STUDENT_INITIALIZATION = "fresh_base_plus_fresh_zero_lora_v1"
CALIBRATION_LENGTH_WINDOW_STEPS = 4
SOURCES = ("medical_opd_o1", "medical_opd_cmb")
PRODUCTION_SLOT = "student_active"
MINIMUM_DISK_BYTES = 10 * 1024**3

_STEP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "optimizer_step",
    "policy_version",
    "next_policy_version",
    "generated_by_policy_version",
    "p_old_policy_version",
    "sampler_adapter_sha256",
    "input_trainer_authority_sha256",
    "trainer_authority_sha256",
    "runtime_adapter_sha256",
    "fresh_adapter_sha256",
    "rollout_provenance_sha256",
    "prompt_samples",
    "q_logprob",
    "p_old_logprob",
    "teacher_logprob",
    "valid_token_count",
    "reverse_kl",
    "advantage",
    "importance_ratio",
    "ppo_clip_fraction",
    "ess_fraction",
    "objective",
    "loss",
    "gradient_norm",
    "nonzero_update_tensor_count",
    "zero_update_tensor_count",
    "teacher_gradient_tensor_count",
    "base_gradient_tensor_count",
    "adapter_delta_norm",
    "teacher_same_token_scoring",
    "teacher_generated_completion",
    "p_old_detached",
    "normal_request_accepted",
    "stale_policy_rejected_before_forward",
    "stale_error_code",
    "sampler_refresh_seconds",
    "timings_seconds",
    "throughput",
    "gpu_memory_bytes",
    "disk_remaining_bytes",
    "checkpoint",
    "isolation",
}
_SAMPLE_FIELDS = {
    "sample_id",
    "content_hash",
    "source",
    "prompt_tokens",
    "generated_tokens",
    "eos",
    "truncated",
    "finish_reason",
    "invalid",
    "empty",
    "non_finite",
    "unexpected_think_tag",
    "repetition",
}
_TIMING_FIELDS = {"generation", "scoring", "backward", "checkpoint", "step"}
_THROUGHPUT_FIELDS = {"rollout_tokens_per_second", "scorer_tokens_per_second"}
_GPU_FIELDS = {
    "gpu0_allocated",
    "gpu0_reserved",
    "gpu0_peak",
    "gpu1_allocated",
    "gpu1_reserved",
    "gpu1_peak",
}
_CHECKPOINT_FIELDS = {
    "logical_version",
    "path",
    "adapter_sha256",
    "complete",
    "resume_eligible",
}
_ISOLATION_FIELDS = {
    "final_access",
    "controller_access",
    "confirmation_access",
    "label_access",
}
_RAW_PRIVATE_KEYS = {
    "question",
    "answer",
    "label",
    "response",
    "completion",
    "prompt_text",
    "response_text",
    "token_ids",
    "prompt_ids",
    "response_ids",
    "logits",
}


class B2CalibrationContractV1Error(RuntimeError):
    """A frozen calibration or evidence invariant failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationContractV1Error(message)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} is not an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    if minimum is not None and result < minimum:
        _fail(f"{label} is below its minimum")
    return result


def _mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{label} fields are not exact")
    return value


def _privacy_scan(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _RAW_PRIVATE_KEYS:
                _fail(f"privacy field is forbidden: {key}")
            _privacy_scan(child)
    elif isinstance(value, list):
        for child in value:
            _privacy_scan(child)


def _validate_distribution(value: Any, label: str) -> None:
    mapping = _mapping(value, {"mean", "std"}, label)
    _finite(mapping["mean"], f"{label} mean")
    _finite(mapping["std"], f"{label} std", minimum=0.0)


def _selected_response_length(value: Any) -> int:
    if value not in SUPPORTED_RESPONSE_LENGTHS:
        _fail("response length is not a registered calibration package length")
    return int(value)


def _validate_samples(
    value: Any,
    *,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
    expected_source_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected_response_length = _selected_response_length(selected_response_length)
    expected = (
        {source: 2 for source in SOURCES}
        if expected_source_counts is None
        else dict(expected_source_counts)
    )
    if not (
        expected
        and sum(expected.values()) == 4
        and all(
            isinstance(source, str)
            and source
            and all(
                marker not in source
                for marker in ("final", "controller", "confirmation")
            )
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            for source, count in expected.items()
        )
    ):
        _fail("expected prompt source contract is invalid")
    if not isinstance(value, list) or len(value) != 4:
        _fail("B2 step must contain exactly four prompt sample records")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {source: 0 for source in expected}
    for raw in value:
        row = dict(_mapping(raw, _SAMPLE_FIELDS, "prompt sample"))
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            _fail("prompt sample identity is absent or duplicated")
        seen.add(sample_id)
        _sha(row["content_hash"], "prompt content hash")
        source = row["source"]
        if source not in counts:
            _fail("prompt sample source is outside the frozen package")
        counts[source] += 1
        _integer(row["prompt_tokens"], "prompt token count", minimum=1)
        generated = _integer(row["generated_tokens"], "generated token count")
        if generated > selected_response_length:
            _fail(
                "generated token count exceeds frozen "
                f"{selected_response_length}"
            )
        if not all(
            isinstance(row[field], bool)
            for field in (
                "eos",
                "truncated",
                "invalid",
                "empty",
                "non_finite",
                "unexpected_think_tag",
                "repetition",
            )
        ):
            _fail("prompt health booleans are invalid")
        if row["finish_reason"] not in {"eos", "length"}:
            _fail("prompt finish reason is invalid")
        if any(
            row[field]
            for field in (
                "invalid",
                "empty",
                "non_finite",
                "unexpected_think_tag",
                "repetition",
            )
        ):
            _fail("prompt generation health gate failed")
        if row["truncated"]:
            if not (
                generated == selected_response_length
                and row["eos"] is False
                and row["finish_reason"] == "length"
            ):
                _fail(
                    "truncated prompt is not an exact "
                    f"{selected_response_length} boundary"
                )
        elif not (row["eos"] is True and row["finish_reason"] == "eos"):
            _fail("non-truncated prompt lacks valid EOS provenance")
        rows.append(row)
    if counts != expected:
        _fail("step source balance differs from the frozen contract")
    return rows


def validate_step_record(
    raw: Mapping[str, Any],
    *,
    expected_step: int,
    expected_version: int,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
    expected_source_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate one completed optimizer step and return a detached projection."""

    if not isinstance(raw, Mapping):
        _fail("B2 step record is not an object")
    _privacy_scan(raw)
    record = dict(_mapping(raw, _STEP_FIELDS, "B2 step"))
    if not (
        record["schema_version"] == 1
        and record["artifact_kind"] == "b2_calibration_step_v1"
        and isinstance(record["run_id"], str)
        and bool(record["run_id"])
        and record["optimizer_step"] == expected_step
        and record["policy_version"] == expected_version
        and record["next_policy_version"] == expected_version + 1
        and record["generated_by_policy_version"] == expected_version
    ):
        _fail("B2 step/version contract drift")
    if record["p_old_policy_version"] != expected_version or record["p_old_detached"] is not True:
        _fail("p_old is not the frozen rollout actor")
    before = _sha(record["sampler_adapter_sha256"], "sampler adapter")
    if before != _sha(record["input_trainer_authority_sha256"], "input trainer authority"):
        _fail("sampler/trainer input chain differs")
    after = _sha(record["trainer_authority_sha256"], "trainer authority")
    if before == after:
        _fail("adapter identity did not change after optimizer step")
    if any(
        _sha(record[field], field) != after
        for field in ("runtime_adapter_sha256", "fresh_adapter_sha256")
    ):
        _fail("trainer/runtime/fresh adapter identity differs")
    _sha(record["rollout_provenance_sha256"], "rollout provenance")
    record["prompt_samples"] = _validate_samples(
        record["prompt_samples"],
        selected_response_length=selected_response_length,
        expected_source_counts=expected_source_counts,
    )
    for field in ("q_logprob", "p_old_logprob", "teacher_logprob", "reverse_kl", "importance_ratio"):
        _validate_distribution(record[field], field)
    advantage = _mapping(record["advantage"], {"mean", "std", "clip_fraction"}, "advantage")
    _finite(advantage["mean"], "advantage mean")
    _finite(advantage["std"], "advantage std", minimum=0.0)
    clip = _finite(advantage["clip_fraction"], "advantage clip fraction", minimum=0.0)
    if clip > 1.0:
        _fail("advantage clip fraction exceeds one")
    _integer(record["valid_token_count"], "valid token count", minimum=1)
    for field in (
        "ppo_clip_fraction",
        "ess_fraction",
        "objective",
        "loss",
        "gradient_norm",
        "adapter_delta_norm",
        "sampler_refresh_seconds",
    ):
        _finite(record[field], field, minimum=(0.0 if field in {"ppo_clip_fraction", "ess_fraction", "gradient_norm", "adapter_delta_norm", "sampler_refresh_seconds"} else None))
    if not (0.0 <= float(record["ppo_clip_fraction"]) <= 1.0 and 0.0 < float(record["ess_fraction"]) <= 1.0):
        _fail("PPO/ESS fraction is outside [0,1]")
    nonzero = _integer(record["nonzero_update_tensor_count"], "nonzero update tensor count")
    zero = _integer(record["zero_update_tensor_count"], "zero update tensor count")
    if nonzero <= 0 or nonzero + zero != 504 or float(record["adapter_delta_norm"]) <= 0.0:
        _fail("adapter update tensor evidence is invalid")
    if record["teacher_gradient_tensor_count"] != 0 or record["base_gradient_tensor_count"] != 0:
        _fail("Teacher/Base gradient ownership gate failed")
    if not (
        record["teacher_same_token_scoring"] is True
        and record["teacher_generated_completion"] is False
    ):
        _fail("Teacher must score the exact Student token sequence without generation")
    if not (
        record["normal_request_accepted"] is True
        and record["stale_policy_rejected_before_forward"] is True
        and record["stale_error_code"] == "STALE_SAMPLER_IDENTITY"
    ):
        _fail("stale policy guard failed before forward")
    timings = _mapping(record["timings_seconds"], _TIMING_FIELDS, "timings")
    for key, value in timings.items():
        _finite(value, f"timing {key}", minimum=0.0)
    throughput = _mapping(record["throughput"], _THROUGHPUT_FIELDS, "throughput")
    for key, value in throughput.items():
        _finite(value, f"throughput {key}", minimum=0.0)
    memory = _mapping(record["gpu_memory_bytes"], _GPU_FIELDS, "GPU memory")
    for key, value in memory.items():
        _integer(value, f"GPU memory {key}")
    if _integer(record["disk_remaining_bytes"], "disk remaining") <= MINIMUM_DISK_BYTES:
        _fail("disk remaining is not strictly above 10 GiB")
    checkpoint = _mapping(record["checkpoint"], _CHECKPOINT_FIELDS, "checkpoint")
    if not (
        checkpoint["logical_version"] == expected_version + 1
        and isinstance(checkpoint["path"], str)
        and bool(checkpoint["path"])
        and _sha(checkpoint["adapter_sha256"], "checkpoint adapter") == after
        and checkpoint["complete"] is True
        and isinstance(checkpoint["resume_eligible"], bool)
    ):
        _fail("checkpoint identity or completeness gate failed")
    isolation = _mapping(record["isolation"], _ISOLATION_FIELDS, "isolation")
    if any(isolation[field] is not False for field in _ISOLATION_FIELDS):
        _fail("restricted evaluation access is not closed")
    return record


def validate_calibration_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_adapter_sha256: str,
    final_reload_adapter_sha256: str,
    forbidden_qualification_adapter_sha256: str | None = None,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
) -> dict[str, Any]:
    selected_response_length = _selected_response_length(selected_response_length)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or len(records) != B2_CALIBRATION_STEPS:
        _fail("calibration must contain exactly 20 optimizer steps")
    initial = _sha(initial_adapter_sha256, "initial adapter")
    if forbidden_qualification_adapter_sha256 is not None and initial == _sha(
        forbidden_qualification_adapter_sha256, "qualification v2 adapter"
    ):
        _fail("qualification v2 cannot initialize the B2 Student")
    expected = initial
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = validate_step_record(
            raw,
            expected_step=index + 1,
            expected_version=index,
            selected_response_length=selected_response_length,
        )
        if record["sampler_adapter_sha256"] != expected:
            _fail("B2 refreshed policy chain is discontinuous")
        should_resume = record["next_policy_version"] in {10, 20}
        if record["checkpoint"]["resume_eligible"] is not should_resume:
            _fail("checkpoint resume policy differs from step10/step20")
        expected = record["trainer_authority_sha256"]
        validated.append(record)
    if _sha(final_reload_adapter_sha256, "final reload adapter") != expected:
        _fail("step20 checkpoint fresh reload identity differs")
    return {
        "steps_completed": B2_CALIBRATION_STEPS,
        "initial_policy_version": 0,
        "final_policy_version": B2_CALIBRATION_STEPS,
        "initial_adapter_sha256": initial,
        "final_adapter_sha256": expected,
        "step_chain_sha256": canonical_json_sha256(validated),
    }


def _quantile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _length_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["generated_tokens"]) for row in rows]
    source_counts = {source: sum(row["source"] == source for row in rows) for source in SOURCES}
    truncation = {source: sum(row["source"] == source and row["truncated"] for row in rows) for source in SOURCES}
    total_truncation = sum(truncation.values())
    count = len(rows)
    reasons: list[str] = []
    if total_truncation / count > 0.20:
        reasons.append("overall_truncation_rate")
    for source in SOURCES:
        if truncation[source] / source_counts[source] > 0.20:
            reasons.append(f"{source}_truncation_rate")
    return {
        "count": count,
        "source_counts": source_counts,
        "truncation_count": total_truncation,
        "truncation_rate": total_truncation / count,
        "source_truncation_count": truncation,
        "source_truncation_rate": {
            source: truncation[source] / source_counts[source] for source in SOURCES
        },
        "eos_count": sum(bool(row["eos"]) for row in rows),
        "completion_length": {
            "min": min(lengths),
            "mean": mean(lengths),
            "p50": _quantile(lengths, 0.50),
            "p90": _quantile(lengths, 0.90),
            "p95": _quantile(lengths, 0.95),
            "p99": _quantile(lengths, 0.99),
            "max": max(lengths),
        },
        "failure_reasons": reasons,
        "passed": not reasons,
    }


def evaluate_calibration_length_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
) -> dict[str, Any]:
    """Apply the P4.7 20% gate to every 4-step (8/source) real window."""

    selected_response_length = _selected_response_length(selected_response_length)
    if len(records) != B2_CALIBRATION_STEPS:
        _fail("length gate requires exactly 20 optimizer steps")
    rows_by_step: list[list[dict[str, Any]]] = []
    for index, raw in enumerate(records):
        validated = validate_step_record(
            raw,
            expected_step=index + 1,
            expected_version=index,
            selected_response_length=selected_response_length,
        )
        rows_by_step.append(validated["prompt_samples"])
    windows: list[dict[str, Any]] = []
    for start in range(0, B2_CALIBRATION_STEPS - CALIBRATION_LENGTH_WINDOW_STEPS + 1):
        rows = [
            row
            for step_rows in rows_by_step[start : start + CALIBRATION_LENGTH_WINDOW_STEPS]
            for row in step_rows
        ]
        windows.append(
            {
                "start_optimizer_step": start + 1,
                "end_optimizer_step": start + CALIBRATION_LENGTH_WINDOW_STEPS,
                **_length_summary(rows),
            }
        )
    aggregate = _length_summary([row for step in rows_by_step for row in step])
    passed = aggregate["passed"] and all(window["passed"] for window in windows)
    return {
        "status": (
            "passed_b2_calibration_length_gate"
            if passed
            else "failed_b2_calibration_length_insufficient"
        ),
        "passed": passed,
        "selected_response_length": selected_response_length,
        "window_steps": CALIBRATION_LENGTH_WINDOW_STEPS,
        "windows": windows,
        "aggregate": aggregate,
        "escalation_recommendation": (
            None
            if passed or selected_response_length != 768
            else {
                "recommended_response_length": 1024,
                "requires_new_versioned_package": True,
                "same_run_switch_allowed": False,
            }
        ),
    }


def evaluate_latest_calibration_length_window(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
) -> dict[str, Any]:
    """Evaluate the newest complete 4-step window during a live calibration."""

    selected_response_length = _selected_response_length(selected_response_length)
    if not (
        isinstance(records, Sequence)
        and not isinstance(records, (str, bytes))
        and CALIBRATION_LENGTH_WINDOW_STEPS <= len(records) <= B2_CALIBRATION_STEPS
    ):
        _fail("live length gate requires 4 through 20 contiguous steps")
    validated = [
        validate_step_record(
            raw,
            expected_step=index + 1,
            expected_version=index,
            selected_response_length=selected_response_length,
        )
        for index, raw in enumerate(records)
    ]
    start = len(validated) - CALIBRATION_LENGTH_WINDOW_STEPS
    rows = [
        row
        for record in validated[start:]
        for row in record["prompt_samples"]
    ]
    result = _length_summary(rows)
    return {
        "status": (
            "passed_b2_calibration_length_window"
            if result["passed"]
            else "failed_b2_calibration_length_insufficient"
        ),
        "selected_response_length": selected_response_length,
        "start_optimizer_step": start + 1,
        "end_optimizer_step": len(validated),
        **result,
        "escalation_recommendation": (
            None
            if result["passed"] or selected_response_length != 768
            else {
                "recommended_response_length": 1024,
                "requires_new_versioned_package": True,
                "same_run_switch_allowed": False,
            }
        ),
    }


__all__ = [
    "B2_CALIBRATION_STEPS",
    "B2CalibrationContractV1Error",
    "CALIBRATION_LENGTH_WINDOW_STEPS",
    "FRESH_STUDENT_INITIALIZATION",
    "MINIMUM_DISK_BYTES",
    "SELECTED_RESPONSE_LENGTH",
    "SUPPORTED_RESPONSE_LENGTHS",
    "SOURCES",
    "canonical_json_sha256",
    "evaluate_calibration_length_gate",
    "evaluate_latest_calibration_length_window",
    "validate_calibration_chain",
    "validate_step_record",
]
