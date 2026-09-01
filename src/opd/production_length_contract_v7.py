"""CPU-safe response-length qualification contract for P4.7.

The module contains no model or CUDA imports.  It converts privacy-safe
generation observations into per-sample and aggregate evidence, selects only
the shortest passing candidate, and decides whether the one bounded
conditional continuation is scientifically admissible.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 7
PROTOCOL_VERSION = "p4.7-production-length-qualification-v7"
PRIMARY_ACTUAL_CAP = 2048
PRIMARY_CANDIDATES = (256, 384, 512, 768, 1024, 1536, 2048)
CONDITIONAL_ACTUAL_CAP = 4096
CONDITIONAL_4096_CANDIDATES = (2048, 2560, 3072, 4096)
SOURCES = ("medical_opd_o1", "medical_opd_cmb")
PROMPTS_PER_SOURCE = 8
OVERALL_TRUNCATION_RATE_MAX = 0.20
PER_SOURCE_TRUNCATION_RATE_MAX = 0.20
PREFIX_LOGPROB_MAX_GAP = 0.0001

_SAMPLE_FIELDS = {
    "sample_id",
    "prompt_hash",
    "source",
    "frozen_order",
    "per_sample_seed",
    "prompt_token_count",
    "actual_generation_cap",
    "generated_token_count",
    "eos_seen",
    "first_eos_position",
    "finish_reason",
    "valid_completion",
    "empty_completion",
    "non_finite",
    "unexpected_think_tag",
    "repetition_detected",
    "repetition_rule_version",
    "candidate_health",
}
_CANDIDATE_HEALTH_FIELDS = {
    "finish_reason",
    "valid_completion",
    "empty_completion",
    "non_finite",
    "unexpected_think_tag",
    "repetition_detected",
}
_HEALTH_FAILURES = {
    "invalid_completion",
    "empty_completion",
    "non_finite",
    "unexpected_think_tag",
    "repetition_detected",
}


class LengthContractV7Error(RuntimeError):
    """The frozen response-length evidence contract was violated."""


def _fail(message: str) -> None:
    raise LengthContractV7Error(message)


def _is_hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        _fail(f"{label} must be finite and non-negative")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LengthContractV7Error(f"value is not canonical JSON: {error}") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_candidate_ladder(
    actual_generation_cap: int, candidates: Sequence[int]
) -> tuple[int, ...]:
    if (
        isinstance(actual_generation_cap, bool)
        or not isinstance(actual_generation_cap, int)
        or actual_generation_cap <= 0
    ):
        _fail("actual generation cap must be a positive integer")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        _fail("candidate ladder must be a sequence")
    values = tuple(candidates)
    if (
        not values
        or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
        or any(item <= 0 or item > actual_generation_cap for item in values)
        or tuple(sorted(values)) != values
        or len(set(values)) != len(values)
        or values[-1] != actual_generation_cap
    ):
        _fail("candidate ladder must be ordered, unique, bounded and end at actual cap")
    return values


def derive_per_sample_seed(base_seed: int, sample_id: str, frozen_order: int) -> int:
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or not isinstance(sample_id, str)
        or not sample_id
        or isinstance(frozen_order, bool)
        or not isinstance(frozen_order, int)
        or frozen_order < 0
    ):
        _fail("per-sample seed identity is invalid")
    encoded = (
        b"p4.7-per-sample-seed-v1\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + str(frozen_order).encode("ascii")
        + b"\0"
        + sample_id.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**31)


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        _fail("completion-length percentile has no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _validate_source_records(
    records: Sequence[Mapping[str, Any]],
    actual_generation_cap: int,
    candidate_health_candidates: Sequence[int],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        _fail("length observations must be a sequence")
    normalized: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    orders: set[int] = set()
    source_counts: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _SAMPLE_FIELDS:
            _fail("length observation fields are not exact")
        value = dict(raw)
        sample_id = value["sample_id"]
        source = value["source"]
        order = value["frozen_order"]
        seed = value["per_sample_seed"]
        prompt_tokens = value["prompt_token_count"]
        observed_cap = value["actual_generation_cap"]
        generated = value["generated_token_count"]
        eos_seen = value["eos_seen"]
        eos_position = value["first_eos_position"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not _is_hex(value["prompt_hash"], 64)
            or source not in SOURCES
            or isinstance(order, bool)
            or not isinstance(order, int)
            or order < 0
            or order in orders
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or observed_cap != actual_generation_cap
            or isinstance(generated, bool)
            or not isinstance(generated, int)
            or generated < 0
            or generated > actual_generation_cap
        ):
            _fail("length observation identity/count is invalid")
        if not isinstance(eos_seen, bool):
            _fail("eos_seen must be boolean")
        if eos_seen:
            if (
                isinstance(eos_position, bool)
                or not isinstance(eos_position, int)
                or eos_position < 1
                or eos_position > generated
            ):
                _fail("first EOS position must be one-based within generated tokens")
        elif eos_position is not None:
            _fail("first EOS position must be null when EOS was not seen")
        for field in (
            "valid_completion",
            "empty_completion",
            "non_finite",
            "unexpected_think_tag",
            "repetition_detected",
        ):
            if not isinstance(value[field], bool):
                _fail(f"{field} must be boolean")
        if value["finish_reason"] not in {"eos", "length", "unexpected_stop"}:
            _fail("finish_reason is absent")
        if (
            not isinstance(value["repetition_rule_version"], str)
            or not value["repetition_rule_version"]
        ):
            _fail("repetition rule version is absent")
        candidate_health = value["candidate_health"]
        expected_health_keys = {
            str(candidate) for candidate in candidate_health_candidates
        }
        if not isinstance(candidate_health, Mapping) or not expected_health_keys.issubset(
            set(candidate_health)
        ):
            _fail("candidate-prefix health evidence is incomplete")
        normalized_health: dict[str, dict[str, Any]] = {}
        for candidate_key, raw_health in candidate_health.items():
            try:
                candidate_value = int(candidate_key)
            except (TypeError, ValueError) as error:
                raise LengthContractV7Error(
                    "candidate-prefix health key is invalid"
                ) from error
            if (
                not isinstance(candidate_key, str)
                or str(candidate_value) != candidate_key
                or candidate_value <= 0
                or candidate_value > actual_generation_cap
                or not isinstance(raw_health, Mapping)
                or set(raw_health) != _CANDIDATE_HEALTH_FIELDS
                or raw_health.get("finish_reason")
                not in {"eos", "length", "unexpected_stop"}
            ):
                _fail("candidate-prefix health evidence is invalid")
            for field in _CANDIDATE_HEALTH_FIELDS - {"finish_reason"}:
                if not isinstance(raw_health.get(field), bool):
                    _fail("candidate-prefix health flag is not boolean")
            normalized_health[candidate_key] = dict(raw_health)
        full_health = normalized_health.get(str(actual_generation_cap))
        if full_health is None or any(
            full_health[field] != value[field]
            for field in _CANDIDATE_HEALTH_FIELDS
        ):
            _fail("full-cap and candidate-prefix health evidence differ")
        value["candidate_health"] = normalized_health
        sample_ids.add(sample_id)
        orders.add(order)
        source_counts[source] += 1
        normalized.append(value)
    if len(normalized) != 16 or source_counts != Counter(
        {source: PROMPTS_PER_SOURCE for source in SOURCES}
    ):
        _fail("length qualification requires exactly 8 prompts per source")
    if orders != set(range(16)):
        _fail("frozen prompt order must be exactly 0..15")
    normalized.sort(key=lambda item: item["frozen_order"])
    return normalized


def _sample_health_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if record["valid_completion"] is not True:
        reasons.append("invalid_completion")
    if record["empty_completion"] is True:
        reasons.append("empty_completion")
    if record["non_finite"] is True:
        reasons.append("non_finite")
    if record["unexpected_think_tag"] is True:
        reasons.append("unexpected_think_tag")
    if record["repetition_detected"] is True:
        reasons.append("repetition_detected")
    if record["finish_reason"] not in {"eos", "length"}:
        reasons.append("finish_reason_invalid")
    return reasons


def _candidate_sample(
    record: Mapping[str, Any], candidate: int, *, actual_generation_cap: int | None = None
) -> dict[str, Any]:
    eos_position = record["first_eos_position"]
    eos_within = bool(
        record["eos_seen"] is True
        and isinstance(eos_position, int)
        and eos_position <= candidate
    )
    effective = (
        int(eos_position)
        if eos_within
        else min(int(record["generated_token_count"]), candidate)
    )
    health = record["candidate_health"][str(candidate)]
    reasons = _sample_health_reasons(health)
    if not eos_within:
        reasons.append("truncated")
    return {
        "actual_generation_cap": (
            candidate if actual_generation_cap is None else actual_generation_cap
        ),
        "generated_token_count": int(record["generated_token_count"]),
        "eos_seen": bool(record["eos_seen"]),
        "first_eos_position": record["first_eos_position"],
        "finish_reason": str(health["finish_reason"]),
        "valid_completion": bool(health["valid_completion"]),
        "empty_completion": bool(health["empty_completion"]),
        "non_finite": bool(health["non_finite"]),
        "unexpected_think_tag": bool(health["unexpected_think_tag"]),
        "repetition_detected": bool(health["repetition_detected"]),
        "truncated": not eos_within,
        "effective_length": effective,
        "eos_within_candidate": eos_within,
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def _aggregate_candidate(
    samples: Sequence[Mapping[str, Any]], candidate: int
) -> dict[str, Any]:
    per_source: dict[str, dict[str, Any]] = {}
    lengths: list[int] = []
    finish_reasons: Counter[str] = Counter()
    eos_count = 0
    truncation_count = 0
    for source in SOURCES:
        rows = [item for item in samples if item["source"] == source]
        source_truncated = 0
        for row in rows:
            derived = row["candidates"][str(candidate)]
            source_truncated += int(derived["truncated"])
            truncation_count += int(derived["truncated"])
            eos_count += int(derived["eos_within_candidate"])
            lengths.append(int(derived["effective_length"]))
            finish_reasons[str(derived["finish_reason"])] += 1
        per_source[source] = {
            "count": len(rows),
            "truncation_count": source_truncated,
            "truncation_rate": source_truncated / len(rows),
        }
    candidate_rows = [item["candidates"][str(candidate)] for item in samples]
    invalid_count = sum(item["valid_completion"] is not True for item in candidate_rows)
    empty_count = sum(
        item["empty_completion"] is True or item["generated_token_count"] == 0
        for item in candidate_rows
    )
    non_finite_count = sum(item["non_finite"] is True for item in candidate_rows)
    think_count = sum(item["unexpected_think_tag"] is True for item in candidate_rows)
    repetition_count = sum(item["repetition_detected"] is True for item in candidate_rows)
    failure_reasons: list[str] = []
    if truncation_count / len(samples) > OVERALL_TRUNCATION_RATE_MAX:
        failure_reasons.append("overall_truncation_rate")
    for source in SOURCES:
        if (
            per_source[source]["truncation_count"] > 1
            or per_source[source]["truncation_rate"]
            > PER_SOURCE_TRUNCATION_RATE_MAX
        ):
            failure_reasons.append(f"{source}_truncation_rate")
    for count, reason in (
        (invalid_count, "invalid_completion"),
        (empty_count, "empty_completion"),
        (non_finite_count, "non_finite"),
        (think_count, "unexpected_think_tag"),
        (repetition_count, "repetition_detected"),
    ):
        if count:
            failure_reasons.append(reason)
    if any(
        "finish_reason_invalid" in item["failure_reasons"]
        for item in candidate_rows
    ):
        failure_reasons.append("finish_reason_invalid")
    return {
        "candidate": candidate,
        "overall_n": len(samples),
        "medical_opd_o1_n": per_source["medical_opd_o1"]["count"],
        "medical_opd_cmb_n": per_source["medical_opd_cmb"]["count"],
        "overall_truncation_count": truncation_count,
        "overall_truncation_rate": truncation_count / len(samples),
        "per_source": per_source,
        "eos_count": eos_count,
        "eos_rate": eos_count / len(samples),
        "invalid_count": invalid_count,
        "empty_count": empty_count,
        "non_finite_count": non_finite_count,
        "unexpected_think_tag_count": think_count,
        "repetition_count": repetition_count,
        "completion_length": {
            "min": min(lengths),
            "mean": statistics.fmean(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
        },
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


def build_length_telemetry(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    actual_generation_cap: int,
    candidates: Sequence[int],
    parent_p4_6_binding_sha256: str,
    generation_backend_identity: str,
    model_revision: str,
    base_revision: str,
    adapter_revision: str,
    student_policy_version: str,
    runtime_adapter_sha256: str,
    decoding_config_sha256: str,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int | None,
    estimated_cost_cny: float,
    actual_cost_cny: float | None,
) -> dict[str, Any]:
    ladder = validate_candidate_ladder(actual_generation_cap, candidates)
    if not isinstance(run_id, str) or not run_id:
        _fail("run ID is absent")
    for value, label, length in (
        (parent_p4_6_binding_sha256, "parent P4.6 binding", 64),
        (model_revision, "model revision", 40),
        (base_revision, "base revision", 40),
        (runtime_adapter_sha256, "runtime adapter SHA", 64),
        (decoding_config_sha256, "decoding config SHA", 64),
    ):
        if not _is_hex(value, length):
            _fail(f"{label} is not immutable")
    for value, label in (
        (generation_backend_identity, "generation backend"),
        (adapter_revision, "adapter revision"),
        (student_policy_version, "Student policy version"),
    ):
        if not isinstance(value, str) or not value:
            _fail(f"{label} is absent")
    elapsed = _finite_number(elapsed_seconds, "elapsed seconds", nonnegative=True)
    estimated_cost = _finite_number(
        estimated_cost_cny, "estimated cost", nonnegative=True
    )
    actual_cost = (
        None
        if actual_cost_cny is None
        else _finite_number(actual_cost_cny, "actual cost", nonnegative=True)
    )
    if peak_gpu_memory_bytes is not None and (
        isinstance(peak_gpu_memory_bytes, bool)
        or not isinstance(peak_gpu_memory_bytes, int)
        or peak_gpu_memory_bytes < 0
    ):
        _fail("peak GPU memory must be a non-negative integer or null")
    normalized = _validate_source_records(
        records, actual_generation_cap, ladder
    )
    samples: list[dict[str, Any]] = []
    for record in normalized:
        candidate_values = {
            str(candidate): _candidate_sample(
                record, candidate, actual_generation_cap=actual_generation_cap
            )
            for candidate in ladder
        }
        samples.append(
            {
                **record,
                "actual_generation_cap": actual_generation_cap,
                "candidates": candidate_values,
                "generation_backend_identity": generation_backend_identity,
                "model_revision": model_revision,
                "base_revision": base_revision,
                "adapter_revision": adapter_revision,
                "student_policy_version": student_policy_version,
                "runtime_adapter_sha256": runtime_adapter_sha256,
                "decoding_config_sha256": decoding_config_sha256,
                "parent_p4_6_binding_sha256": parent_p4_6_binding_sha256,
            }
        )
    aggregates = [_aggregate_candidate(samples, candidate) for candidate in ladder]
    generated_tokens = sum(item["generated_token_count"] for item in samples)
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_telemetry_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "evidence_complete": True,
        "selection_performed": False,
        "generation_mode": "single_actual_trajectory_derived_candidates",
        "actual_generation_cap": actual_generation_cap,
        "candidate_ladder": list(ladder),
        "sample_count": len(samples),
        "source_counts": {source: PROMPTS_PER_SOURCE for source in SOURCES},
        "samples": samples,
        "aggregates": aggregates,
        "health": {
            "invalid_count": sum(not item["valid_completion"] for item in samples),
            "empty_count": sum(
                item["empty_completion"] or item["generated_token_count"] == 0
                for item in samples
            ),
            "non_finite_count": sum(item["non_finite"] for item in samples),
            "unexpected_think_tag_count": sum(
                item["unexpected_think_tag"] for item in samples
            ),
            "repetition_count": sum(item["repetition_detected"] for item in samples),
        },
        "resources": {
            "actual_generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "estimated_cost_cny": estimated_cost,
            "actual_cost_cny": actual_cost,
        },
        "bindings": {
            "parent_p4_6_binding_sha256": parent_p4_6_binding_sha256,
            "generation_backend_identity": generation_backend_identity,
            "model_revision": model_revision,
            "base_revision": base_revision,
            "adapter_revision": adapter_revision,
            "student_policy_version": student_policy_version,
            "runtime_adapter_sha256": runtime_adapter_sha256,
            "decoding_config_sha256": decoding_config_sha256,
        },
    }
    canonical_json_bytes(result)
    return result


def build_explicit_length_telemetry(
    records_by_candidate: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    run_id: str,
    actual_generation_cap: int,
    candidates: Sequence[int],
    parent_p4_6_binding_sha256: str,
    generation_backend_identity: str,
    model_revision: str,
    base_revision: str,
    adapter_revision: str,
    student_policy_version: str,
    runtime_adapter_sha256: str,
    decoding_config_sha256: str,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int | None,
    estimated_cost_cny: float,
    actual_cost_cny: float | None,
) -> dict[str, Any]:
    """Build one artifact from explicitly generated candidate trajectories.

    Prefix mismatch makes derived truncation statistics inadmissible.  This
    builder keeps the same selection schema while sourcing each candidate's
    per-sample result from its own actual generation.
    """

    ladder = validate_candidate_ladder(actual_generation_cap, candidates)
    if set(records_by_candidate) != set(ladder):
        _fail("explicit generation must contain every frozen candidate exactly once")
    normalized = {
        candidate: _validate_source_records(
            records_by_candidate[candidate], candidate, (candidate,)
        )
        for candidate in ladder
    }
    identity_fields = (
        "sample_id",
        "prompt_hash",
        "source",
        "frozen_order",
        "per_sample_seed",
    )
    reference = normalized[ladder[-1]]
    for candidate in ladder[:-1]:
        if [
            tuple(row[field] for field in identity_fields)
            for row in normalized[candidate]
        ] != [tuple(row[field] for field in identity_fields) for row in reference]:
            _fail("explicit candidate prompt identity/seed drift")
    by_order = {
        candidate: {row["frozen_order"]: row for row in rows}
        for candidate, rows in normalized.items()
    }
    reference_for_shell: list[dict[str, Any]] = []
    for reference_row in reference:
        order = reference_row["frozen_order"]
        merged = dict(reference_row)
        merged["candidate_health"] = {
            str(candidate): dict(
                by_order[candidate][order]["candidate_health"][str(candidate)]
            )
            for candidate in ladder
        }
        reference_for_shell.append(merged)
    result = build_length_telemetry(
        reference_for_shell,
        run_id=run_id,
        actual_generation_cap=actual_generation_cap,
        candidates=ladder,
        parent_p4_6_binding_sha256=parent_p4_6_binding_sha256,
        generation_backend_identity=generation_backend_identity,
        model_revision=model_revision,
        base_revision=base_revision,
        adapter_revision=adapter_revision,
        student_policy_version=student_policy_version,
        runtime_adapter_sha256=runtime_adapter_sha256,
        decoding_config_sha256=decoding_config_sha256,
        elapsed_seconds=elapsed_seconds,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        estimated_cost_cny=estimated_cost_cny,
        actual_cost_cny=actual_cost_cny,
    )
    for sample in result["samples"]:
        order = sample["frozen_order"]
        sample["candidates"] = {
            str(candidate): _candidate_sample(
                by_order[candidate][order],
                candidate,
                actual_generation_cap=candidate,
            )
            for candidate in ladder
        }
    result["aggregates"] = [
        _aggregate_candidate(result["samples"], candidate) for candidate in ladder
    ]
    result["generation_mode"] = "explicit_independent_generation"
    actual_generated_tokens = sum(
        row["generated_token_count"]
        for rows in normalized.values()
        for row in rows
    )
    result["resources"]["actual_generated_tokens"] = actual_generated_tokens
    result["resources"]["tokens_per_second"] = (
        actual_generated_tokens / float(result["resources"]["elapsed_seconds"])
        if result["resources"]["elapsed_seconds"] > 0
        else 0.0
    )
    canonical_json_bytes(result)
    return result


def select_shortest_passing_length(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(telemetry, Mapping) or telemetry.get("evidence_complete") is not True:
        _fail("complete telemetry is required before selection")
    if telemetry.get("selection_performed") is not False:
        _fail("telemetry was not committed before selection")
    ladder = validate_candidate_ladder(
        telemetry.get("actual_generation_cap"), telemetry.get("candidate_ladder", [])
    )
    aggregates = telemetry.get("aggregates")
    if not isinstance(aggregates, list) or [
        item.get("candidate") if isinstance(item, Mapping) else None
        for item in aggregates
    ] != list(ladder):
        _fail("candidate aggregates do not match the frozen ladder")
    selected = next(
        (int(item["candidate"]) for item in aggregates if item.get("passed") is True),
        None,
    )
    all_reasons = sorted(
        {
            str(reason)
            for item in aggregates
            for reason in item.get("failure_reasons", [])
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_selection_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": telemetry.get("run_id"),
        "status": "length_frozen" if selected is not None else "no_length_candidate_passed",
        "selected_response_length": selected,
        "evaluated_candidates": list(ladder),
        "decision_rule": "shortest_candidate_passing_frozen_overall_and_per_source_gates_v2",
        "candidate_results": [
            {
                "candidate": item["candidate"],
                "passed": item["passed"],
                "failure_reasons": list(item["failure_reasons"]),
            }
            for item in aggregates
        ],
        "all_failure_reasons": all_reasons,
    }


def compare_prefix_equivalence(
    short_records: Sequence[Mapping[str, Any]],
    long_records: Sequence[Mapping[str, Any]],
    *,
    prefix_length: int,
    max_logprob_gap: float = PREFIX_LOGPROB_MAX_GAP,
) -> dict[str, Any]:
    if (
        isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or prefix_length <= 0
    ):
        _fail("prefix length must be positive")
    tolerance = _finite_number(max_logprob_gap, "prefix logprob gap", nonnegative=True)
    if isinstance(short_records, (str, bytes)) or isinstance(long_records, (str, bytes)):
        _fail("prefix records must be sequences")
    if not all(isinstance(item, Mapping) for item in (*short_records, *long_records)):
        _fail("prefix records must contain objects")
    short_by_id = {item.get("sample_id"): item for item in short_records}
    long_by_id = {item.get("sample_id"): item for item in long_records}
    if len(short_by_id) != len(short_records) or len(long_by_id) != len(long_records):
        _fail("prefix records contain duplicate sample identity")
    if not short_by_id or set(short_by_id) != set(long_by_id):
        _fail("prefix probe sample identities differ")
    per_source: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    overall_passed = True
    maximum_gap = 0.0
    for sample_id in sorted(short_by_id):
        short = short_by_id[sample_id]
        long = long_by_id[sample_id]
        required_identity = (
            "prompt_hash",
            "source",
            "frozen_order",
            "per_sample_seed",
            "generation_backend_identity",
            "model_revision",
            "base_revision",
            "adapter_revision",
            "prefix_invariant_decoding_sha256",
            "runtime_adapter_sha256",
        )
        if any(short.get(field) != long.get(field) for field in required_identity):
            _fail("prefix probe provenance differs across caps")
        short_cap = short.get("actual_generation_cap")
        long_cap = long.get("actual_generation_cap")
        short_full_sha = short.get("full_decoding_config_sha256")
        long_full_sha = long.get("full_decoding_config_sha256")
        if not (
            short_cap == prefix_length
            and long_cap == prefix_length * 2
            and isinstance(short_full_sha, str)
            and len(short_full_sha) == 64
            and isinstance(long_full_sha, str)
            and len(long_full_sha) == 64
            and short_full_sha != long_full_sha
        ):
            _fail("prefix decoding cap provenance is invalid")
        source = short.get("source")
        if source not in SOURCES:
            _fail("prefix probe source is invalid")
        per_source[source] += 1
        short_tokens = short.get("token_ids")
        long_tokens = long.get("token_ids")
        short_logprobs = short.get("selected_logprobs")
        long_logprobs = long.get("selected_logprobs")
        if not all(
            isinstance(value, list)
            for value in (short_tokens, long_tokens, short_logprobs, long_logprobs)
        ) or len(short_tokens) != len(short_logprobs) or len(long_tokens) != len(
            long_logprobs
        ):
            _fail("prefix token/logprob arrays are invalid")
        compared = min(prefix_length, len(short_tokens), len(long_tokens))
        length_consistent = bool(
            compared == prefix_length
            or (len(short_tokens) == len(long_tokens) == compared)
        )
        token_equal = bool(
            length_consistent
            and short_tokens[:compared] == long_tokens[:compared]
        )
        gaps: list[float] = []
        logprob_finite = True
        for left, right in zip(
            short_logprobs[:compared], long_logprobs[:compared], strict=True
        ):
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not math.isfinite(float(left))
                or not math.isfinite(float(right))
            ):
                logprob_finite = False
                break
            gaps.append(abs(float(left) - float(right)))
        gap = max(gaps, default=0.0)
        maximum_gap = max(maximum_gap, gap)
        passed = bool(token_equal and logprob_finite and gap <= tolerance)
        overall_passed = overall_passed and passed
        evidence.append(
            {
                "sample_id": str(sample_id),
                "prompt_hash": short["prompt_hash"],
                "source": source,
                "compared_token_count": compared,
                "generation_provenance_sha256": canonical_json_sha256(
                    {field: short[field] for field in required_identity}
                ),
                "cap_bound_decoding_provenance_sha256": canonical_json_sha256(
                    {
                        "short_actual_cap": short_cap,
                        "short_full_decoding_config_sha256": short_full_sha,
                        "long_actual_cap": long_cap,
                        "long_full_decoding_config_sha256": long_full_sha,
                    }
                ),
                "token_prefix_sha256_short": canonical_json_sha256(
                    short_tokens[:compared]
                ),
                "token_prefix_sha256_long": canonical_json_sha256(
                    long_tokens[:compared]
                ),
                "logprob_prefix_sha256_short": canonical_json_sha256(
                    short_logprobs[:compared]
                ),
                "logprob_prefix_sha256_long": canonical_json_sha256(
                    long_logprobs[:compared]
                ),
                "max_abs_logprob_gap": gap,
                "passed": passed,
            }
        )
    if any(per_source[source] < 1 for source in SOURCES):
        _fail("prefix gate requires at least one probe from each source")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_prefix_equivalence_v7",
        "prefix_length": prefix_length,
        "max_logprob_gap_threshold": tolerance,
        "max_abs_logprob_gap": maximum_gap,
        "per_source_probe_count": dict(per_source),
        "samples": evidence,
        "passed": overall_passed,
        "generation_strategy": (
            "derived_candidates"
            if overall_passed
            else "explicit_independent_generation_required"
        ),
    }


def evaluate_conditional_4096(
    primary_telemetry: Mapping[str, Any],
    *,
    eos_stop_config_valid: bool,
    model_context_limit: int,
    max_prompt_tokens: int,
    disk_preflight_passed: bool,
    gpu_preflight_passed: bool,
    isolation: Mapping[str, Any],
) -> dict[str, Any]:
    decision = select_shortest_passing_length(primary_telemetry)
    reasons: list[str] = []
    if decision["selected_response_length"] is not None:
        reasons.append("primary_candidate_already_passed")
    failure_reasons = set(decision["all_failure_reasons"])
    if not failure_reasons or any("truncation" not in item for item in failure_reasons):
        reasons.append("primary_failure_not_truncation_only")
    health = primary_telemetry.get("health")
    if not isinstance(health, Mapping) or any(
        health.get(field) != 0
        for field in (
            "invalid_count",
            "empty_count",
            "non_finite_count",
            "unexpected_think_tag_count",
            "repetition_count",
        )
    ):
        reasons.append("generation_health_gate_failed")
    if eos_stop_config_valid is not True:
        reasons.append("eos_stop_config_invalid")
    if (
        isinstance(model_context_limit, bool)
        or not isinstance(model_context_limit, int)
        or isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 0
        or max_prompt_tokens + CONDITIONAL_ACTUAL_CAP > model_context_limit
    ):
        reasons.append("model_context_limit_exceeded")
    if disk_preflight_passed is not True:
        reasons.append("disk_preflight_failed")
    if gpu_preflight_passed is not True:
        reasons.append("gpu_preflight_failed")
    required_isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if dict(isolation) != required_isolation:
        reasons.append("isolation_gate_failed")
    allowed = not reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "conditional_4096_eligibility_v7",
        "allowed": allowed,
        "actual_generation_cap": CONDITIONAL_ACTUAL_CAP if allowed else None,
        "candidates": list(CONDITIONAL_4096_CANDIDATES) if allowed else [],
        "reasons": reasons,
        "automatic_further_escalation": False,
    }


def assert_privacy_safe_payload(value: Any, *, path: str = "payload") -> None:
    """Reject common raw medical/supervision fields recursively."""

    forbidden = {
        "question",
        "prompt",
        "prompt_text",
        "response",
        "completion",
        "answer",
        "answer_idx",
        "label",
        "solution",
        "reasoning",
        "output",
        "token_ids",
        "selected_logprobs",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                _fail(f"privacy-unsafe field at {path}.{key}")
            assert_privacy_safe_payload(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_privacy_safe_payload(item, path=f"{path}[{index}]")


__all__ = [
    "CONDITIONAL_4096_CANDIDATES",
    "CONDITIONAL_ACTUAL_CAP",
    "LengthContractV7Error",
    "PRIMARY_ACTUAL_CAP",
    "PRIMARY_CANDIDATES",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "assert_privacy_safe_payload",
    "build_explicit_length_telemetry",
    "build_length_telemetry",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "compare_prefix_equivalence",
    "derive_per_sample_seed",
    "evaluate_conditional_4096",
    "select_shortest_passing_length",
    "validate_candidate_ladder",
]
