"""Evidence-first immutable artifact handling for P4.7 length qualification.

This module is deliberately CPU-safe.  It validates and durably commits the
privacy-safe length telemetry, reopens the committed bytes, verifies their
schema/SHA/size/record counts, and seals an evidence index *before* invoking
the length selector.  No caller-provided readiness value is authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

from src.opd.production_length_contract_v7 import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SOURCES,
    LengthContractV7Error,
    build_explicit_length_telemetry,
    build_length_telemetry,
    select_shortest_passing_length,
    validate_candidate_ladder,
)


TELEMETRY_FILE = "length_telemetry.json"
UNVERIFIED_TELEMETRY_FILE = "length_telemetry.unverified.json"
EVIDENCE_INDEX_FILE = "length_evidence_index.json"
SELECTION_FILE = "length_selection.json"
FAILURE_FILE = "failure.json"
FINAL_INDEX_FILE = "artifact_index.json"
READINESS_FILE = "readiness.json"

_FORBIDDEN_PRIVACY_KEYS = frozenset(
    {
        "question",
        "prompt",
        "raw_prompt",
        "raw_medical_text",
        "answer",
        "standard_answer",
        "label",
        "final_label",
        "response",
        "completion",
        "completion_text",
        "token_ids",
        "selected_logprobs",
    }
)
_OBSERVATION_FIELDS = (
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
)
_CANDIDATE_HEALTH_FIELDS = (
    "finish_reason",
    "valid_completion",
    "empty_completion",
    "non_finite",
    "unexpected_think_tag",
    "repetition_detected",
)


class LengthArtifactV7Error(RuntimeError):
    """A durable P4.7 length artifact operation failed closed."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


Selector = Callable[[Mapping[str, Any]], Mapping[str, Any]]
Sha256Function = Callable[[str | Path], str]


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_privacy_safe_json(
    path: str | Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate privacy keys and atomically commit a non-telemetry artifact."""

    _assert_privacy_safe(value)
    return _atomic_write_json(path, value)


def _document_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LengthArtifactV7Error("artifact_schema_failed") from error


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write one new JSON document via temp+fsync+replace+directory fsync."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable artifact already exists: {target.name}")
    payload = _document_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LengthArtifactV7Error("artifact_schema_failed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LengthArtifactV7Error("artifact_schema_failed") from error
    if not isinstance(value, dict):
        raise LengthArtifactV7Error("artifact_schema_failed")
    return value


def _assert_privacy_safe(value: Any) -> None:
    """Reject raw prompt/answer/response fields without echoing their values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in _FORBIDDEN_PRIVACY_KEYS:
                raise LengthArtifactV7Error("artifact_schema_failed")
            _assert_privacy_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_privacy_safe(child)


def _is_hex(value: Any, length: int = 64) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _observation_from_sample(
    sample: Mapping[str, Any],
    projection: Mapping[str, Any] | None = None,
    *,
    candidate: int | None = None,
) -> dict[str, Any]:
    source = sample if projection is None else {**sample, **projection}
    try:
        result = {field: source[field] for field in _OBSERVATION_FIELDS}
        candidate_items = (
            sample["candidates"].items()
            if projection is None
            else ((str(candidate), projection),)
        )
        result["candidate_health"] = {
            str(candidate_key): {
                field: candidate_value[field]
                for field in _CANDIDATE_HEALTH_FIELDS
            }
            for candidate_key, candidate_value in candidate_items
        }
        return result
    except KeyError as error:
        raise LengthArtifactV7Error("artifact_schema_failed") from error


def _rebuild_length_telemetry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every projection/aggregate/health field from sample evidence."""

    bindings = value["bindings"]
    resources = value["resources"]
    common = {
        "run_id": value["run_id"],
        "actual_generation_cap": value["actual_generation_cap"],
        "candidates": value["candidate_ladder"],
        "parent_p4_6_binding_sha256": bindings["parent_p4_6_binding_sha256"],
        "generation_backend_identity": bindings["generation_backend_identity"],
        "model_revision": bindings["model_revision"],
        "base_revision": bindings["base_revision"],
        "adapter_revision": bindings["adapter_revision"],
        "student_policy_version": bindings["student_policy_version"],
        "runtime_adapter_sha256": bindings["runtime_adapter_sha256"],
        "decoding_config_sha256": bindings["decoding_config_sha256"],
        "elapsed_seconds": resources["elapsed_seconds"],
        "peak_gpu_memory_bytes": resources["peak_gpu_memory_bytes"],
        "estimated_cost_cny": resources["estimated_cost_cny"],
        "actual_cost_cny": resources["actual_cost_cny"],
    }
    samples = value["samples"]
    mode = value["generation_mode"]
    if mode == "single_actual_trajectory_derived_candidates":
        return build_length_telemetry(
            [_observation_from_sample(sample) for sample in samples], **common
        )
    if mode == "explicit_independent_generation":
        records_by_candidate = {
            candidate: [
                _observation_from_sample(
                    sample,
                    sample["candidates"][str(candidate)],
                    candidate=candidate,
                )
                for sample in samples
            ]
            for candidate in value["candidate_ladder"]
        }
        return build_explicit_length_telemetry(records_by_candidate, **common)
    raise LengthArtifactV7Error("artifact_schema_failed")


def validate_length_telemetry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate evidence completeness without performing length selection."""

    if not isinstance(value, Mapping):
        raise LengthArtifactV7Error("artifact_schema_failed")
    _assert_privacy_safe(value)
    required = {
        "schema_version",
        "artifact_kind",
        "protocol_version",
        "run_id",
        "evidence_complete",
        "selection_performed",
        "generation_mode",
        "actual_generation_cap",
        "candidate_ladder",
        "sample_count",
        "source_counts",
        "samples",
        "aggregates",
        "health",
        "resources",
        "bindings",
    }
    if set(value) != required:
        raise LengthArtifactV7Error("artifact_schema_failed")
    try:
        ladder = validate_candidate_ladder(
            value["actual_generation_cap"], value["candidate_ladder"]
        )
    except LengthContractV7Error as error:
        raise LengthArtifactV7Error("artifact_schema_failed") from error
    samples = value.get("samples")
    aggregates = value.get("aggregates")
    health = value.get("health")
    resources = value.get("resources")
    bindings = value.get("bindings")
    if not (
        value["schema_version"] == SCHEMA_VERSION
        and value["artifact_kind"] == "production_length_telemetry_v7"
        and value["protocol_version"] == PROTOCOL_VERSION
        and isinstance(value["run_id"], str)
        and bool(value["run_id"])
        and value["evidence_complete"] is True
        and value["selection_performed"] is False
        and isinstance(value["generation_mode"], str)
        and bool(value["generation_mode"])
        and value["sample_count"] == 16
        and value["source_counts"] == {source: 8 for source in SOURCES}
        and isinstance(samples, list)
        and len(samples) == value["sample_count"]
        and isinstance(aggregates, list)
        and len(aggregates) == len(ladder)
        and isinstance(health, Mapping)
        and isinstance(resources, Mapping)
        and isinstance(bindings, Mapping)
    ):
        raise LengthArtifactV7Error("artifact_schema_failed")
    sample_ids: set[str] = set()
    orders: set[int] = set()
    source_counts = {source: 0 for source in SOURCES}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise LengthArtifactV7Error("artifact_schema_failed")
        sample_id = sample.get("sample_id")
        order = sample.get("frozen_order")
        source = sample.get("source")
        candidates = sample.get("candidates")
        if not (
            isinstance(sample_id, str)
            and sample_id
            and sample_id not in sample_ids
            and isinstance(order, int)
            and not isinstance(order, bool)
            and order not in orders
            and source in SOURCES
            and _is_hex(sample.get("prompt_hash"))
            and sample.get("actual_generation_cap") == value["actual_generation_cap"]
            and isinstance(candidates, Mapping)
            # Canonical JSON sorts object keys lexicographically on disk; the
            # frozen candidate order lives in candidate_ladder, not map order.
            and set(candidates) == {str(candidate) for candidate in ladder}
        ):
            raise LengthArtifactV7Error("artifact_schema_failed")
        sample_ids.add(sample_id)
        orders.add(order)
        source_counts[str(source)] += 1
    if orders != set(range(16)) or source_counts != {source: 8 for source in SOURCES}:
        raise LengthArtifactV7Error("artifact_schema_failed")
    for aggregate, candidate in zip(aggregates, ladder, strict=True):
        if not (
            isinstance(aggregate, Mapping)
            and aggregate.get("candidate") == candidate
            and aggregate.get("overall_n") == 16
            and aggregate.get("medical_opd_o1_n") == 8
            and aggregate.get("medical_opd_cmb_n") == 8
            and isinstance(aggregate.get("passed"), bool)
            and isinstance(aggregate.get("failure_reasons"), list)
        ):
            raise LengthArtifactV7Error("artifact_schema_failed")
    required_resource_fields = {
        "actual_generated_tokens",
        "elapsed_seconds",
        "tokens_per_second",
        "peak_gpu_memory_bytes",
        "estimated_cost_cny",
        "actual_cost_cny",
    }
    if set(resources) != required_resource_fields or not all(
        _finite_nonnegative(resources[field])
        for field in (
            "actual_generated_tokens",
            "elapsed_seconds",
            "tokens_per_second",
            "estimated_cost_cny",
        )
    ):
        raise LengthArtifactV7Error("artifact_schema_failed")
    if resources["actual_cost_cny"] is not None and not _finite_nonnegative(
        resources["actual_cost_cny"]
    ):
        raise LengthArtifactV7Error("artifact_schema_failed")
    if resources["peak_gpu_memory_bytes"] is not None and not (
        isinstance(resources["peak_gpu_memory_bytes"], int)
        and not isinstance(resources["peak_gpu_memory_bytes"], bool)
        and resources["peak_gpu_memory_bytes"] >= 0
    ):
        raise LengthArtifactV7Error("artifact_schema_failed")
    if not (
        _is_hex(bindings.get("parent_p4_6_binding_sha256"))
        and _is_hex(bindings.get("runtime_adapter_sha256"))
        and _is_hex(bindings.get("decoding_config_sha256"))
    ):
        raise LengthArtifactV7Error("artifact_schema_failed")
    # This rejects NaN/Inf and any non-JSON object without retaining a duplicate.
    _document_bytes(dict(value))
    try:
        if _rebuild_length_telemetry(value) != dict(value):
            raise LengthArtifactV7Error("artifact_schema_failed")
    except (KeyError, TypeError, LengthContractV7Error) as error:
        raise LengthArtifactV7Error("artifact_schema_failed") from error
    return dict(value)


def _failure_document(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_failure_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "status": "fail",
        "reason": reason,
        "ready": False,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
    }


def _safe_run_id(value: Any) -> str:
    if (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and re.fullmatch(r"[A-Za-z0-9._-]+", value)
    ):
        return value
    return "invalid"


def _record_failure(output: Path, *, run_id: str, reason: str) -> None:
    path = output / FAILURE_FILE
    if path.exists() or path.is_symlink():
        return
    try:
        _atomic_write_json(path, _failure_document(run_id, reason))
    except BaseException:
        # The structured return remains authoritative when the filesystem itself
        # cannot persist even the failure marker.
        return


def _failed_result(reason: str, *, caller_ready: Any) -> dict[str, Any]:
    return {
        "status": reason,
        "ready": False,
        "selected_response_length": None,
        "caller_ready_ignored": caller_ready is not None,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
    }


def _quarantine_telemetry(output: Path) -> None:
    source = output / TELEMETRY_FILE
    target = output / UNVERIFIED_TELEMETRY_FILE
    if source.is_file() and not source.is_symlink() and not target.exists():
        os.replace(source, target)
        _fsync_directory(output)


def _artifact_entry(path: Path, relative: str) -> dict[str, Any]:
    artifact = path / relative
    return {
        "path": relative,
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def _write_final_index(output: Path, *, run_id: str, failed: bool) -> dict[str, Any]:
    names = [TELEMETRY_FILE, EVIDENCE_INDEX_FILE, SELECTION_FILE]
    if failed:
        names.append(FAILURE_FILE)
    artifacts = [_artifact_entry(output, name) for name in names]
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_final_index_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _atomic_write_json(output / FINAL_INDEX_FILE, value)
    return value


def _index_map(index: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            return {}
        result[str(item["path"])] = item
    return result


def derive_length_readiness(
    output: str | Path, *, caller_ready: Any = None
) -> dict[str, Any]:
    """Recompute length readiness exclusively from reopened indexed artifacts."""

    directory = Path(output)
    errors: list[str] = []
    telemetry: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    try:
        telemetry = validate_length_telemetry(_read_json(directory / TELEMETRY_FILE))
    except LengthArtifactV7Error:
        errors.append("telemetry_invalid")
    evidence_path = directory / EVIDENCE_INDEX_FILE
    if telemetry is not None:
        try:
            evidence = _read_json(evidence_path)
            if not (
                evidence.get("schema_version") == SCHEMA_VERSION
                and evidence.get("protocol_version") == PROTOCOL_VERSION
                and evidence.get("run_id") == telemetry["run_id"]
                and evidence.get("telemetry_path") == TELEMETRY_FILE
                and evidence.get("telemetry_sha256")
                == sha256_file(directory / TELEMETRY_FILE)
                and evidence.get("telemetry_size_bytes")
                == (directory / TELEMETRY_FILE).stat().st_size
                and evidence.get("sample_record_count") == len(telemetry["samples"])
                and evidence.get("aggregate_record_count")
                == len(telemetry["aggregates"])
                and evidence.get("selection_performed") is False
            ):
                errors.append("evidence_index_mismatch")
        except (LengthArtifactV7Error, OSError):
            errors.append("evidence_index_mismatch")
    try:
        selection = _read_json(directory / SELECTION_FILE)
        if telemetry is None:
            raise LengthArtifactV7Error("artifact_schema_failed")
        expected = dict(select_shortest_passing_length(telemetry))
        expected.update(
            {
                "telemetry_path": TELEMETRY_FILE,
                "telemetry_sha256": sha256_file(directory / TELEMETRY_FILE),
                "telemetry_size_bytes": (directory / TELEMETRY_FILE).stat().st_size,
                "sample_record_count": len(telemetry["samples"]),
                "aggregate_record_count": len(telemetry["aggregates"]),
            }
        )
        if selection != expected:
            errors.append("selection_mismatch")
    except (LengthArtifactV7Error, LengthContractV7Error, OSError):
        errors.append("selection_mismatch")

    expected_names = {TELEMETRY_FILE, EVIDENCE_INDEX_FILE, SELECTION_FILE}
    failure_path = directory / FAILURE_FILE
    failure_reason: str | None = None
    if failure_path.exists():
        expected_names.add(FAILURE_FILE)
        try:
            failure = _read_json(failure_path)
            failure_reason = failure.get("reason")
            expected_failure_run_id = (
                str(telemetry["run_id"]) if telemetry is not None else "invalid"
            )
            if failure != _failure_document(
                expected_failure_run_id, str(failure_reason)
            ):
                errors.append("failure_artifact_invalid")
        except LengthArtifactV7Error:
            errors.append("failure_artifact_invalid")
    try:
        final_index = _read_json(directory / FINAL_INDEX_FILE)
        indexed = _index_map(final_index)
        if (
            final_index.get("schema_version") != SCHEMA_VERSION
            or final_index.get("protocol_version") != PROTOCOL_VERSION
            or final_index.get("run_id")
            != (telemetry.get("run_id") if telemetry else "invalid")
            or set(indexed) != expected_names
            or final_index.get("artifact_count") != len(expected_names)
            or any(
                not (directory / name).is_file()
                or (directory / name).is_symlink()
                or indexed[name].get("sha256") != sha256_file(directory / name)
                or indexed[name].get("size_bytes") != (directory / name).stat().st_size
                for name in expected_names
            )
        ):
            errors.append("artifact_index_mismatch")
    except (LengthArtifactV7Error, OSError, KeyError):
        errors.append("artifact_index_mismatch")

    status = selection.get("status") if isinstance(selection, Mapping) else "invalid"
    selected = (
        selection.get("selected_response_length")
        if isinstance(selection, Mapping)
        else None
    )
    success = bool(
        not errors
        and status == "length_frozen"
        and isinstance(selected, int)
        and not failure_path.exists()
    )
    if status == "no_length_candidate_passed":
        if failure_reason != "no_length_candidate_passed":
            errors.append("failure_artifact_invalid")
        if "no_length_candidate_passed" not in errors:
            errors.append("no_length_candidate_passed")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_readiness_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": telemetry.get("run_id") if telemetry else "invalid",
        "status": status,
        "ready": success,
        "selected_response_length": selected if success else None,
        "caller_ready_ignored": caller_ready is not None,
        "failure_reasons": sorted(set(errors)),
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
    }


def commit_length_qualification(
    output: str | Path,
    telemetry: Mapping[str, Any],
    *,
    selector: Selector = select_shortest_passing_length,
    caller_ready: Any = None,
    sha256_func: Sha256Function = sha256_file,
) -> dict[str, Any]:
    """Commit complete telemetry first, then select and seal success/failure.

    The four terminal protocol outcomes are intentionally distinct:
    ``artifact_write_failed``, ``artifact_schema_failed``,
    ``artifact_sha_failed``, and ``no_length_candidate_passed``.
    """

    directory = Path(output)
    run_id = _safe_run_id(
        telemetry.get("run_id", "invalid")
        if isinstance(telemetry, Mapping)
        else "invalid"
    )
    try:
        validated = validate_length_telemetry(telemetry)
    except LengthArtifactV7Error:
        _record_failure(directory, run_id=run_id, reason="artifact_schema_failed")
        return _failed_result("artifact_schema_failed", caller_ready=caller_ready)

    encoded = _document_bytes(validated)
    expected_sha = hashlib.sha256(encoded).hexdigest()
    expected_size = len(encoded)
    telemetry_path = directory / TELEMETRY_FILE
    try:
        _atomic_write_json(telemetry_path, validated)
    except BaseException:
        _record_failure(directory, run_id=run_id, reason="artifact_write_failed")
        return _failed_result("artifact_write_failed", caller_ready=caller_ready)

    try:
        observed_sha = sha256_func(telemetry_path)
        observed_size = telemetry_path.stat().st_size
    except BaseException:
        observed_sha = ""
        observed_size = -1
    if observed_sha != expected_sha or observed_size != expected_size:
        try:
            _quarantine_telemetry(directory)
        except OSError:
            pass
        _record_failure(directory, run_id=run_id, reason="artifact_sha_failed")
        return _failed_result("artifact_sha_failed", caller_ready=caller_ready)

    try:
        reopened = validate_length_telemetry(_read_json(telemetry_path))
        if reopened != validated or len(reopened["samples"]) != 16 or len(
            reopened["aggregates"]
        ) != len(reopened["candidate_ladder"]):
            raise LengthArtifactV7Error("artifact_schema_failed")
    except LengthArtifactV7Error:
        try:
            _quarantine_telemetry(directory)
        except OSError:
            pass
        _record_failure(directory, run_id=run_id, reason="artifact_schema_failed")
        return _failed_result("artifact_schema_failed", caller_ready=caller_ready)

    evidence_index = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_evidence_index_v7",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "telemetry_path": TELEMETRY_FILE,
        "telemetry_sha256": observed_sha,
        "telemetry_size_bytes": observed_size,
        "sample_record_count": len(reopened["samples"]),
        "aggregate_record_count": len(reopened["aggregates"]),
        "selection_performed": False,
    }
    try:
        _atomic_write_json(directory / EVIDENCE_INDEX_FILE, evidence_index)
    except BaseException:
        _record_failure(directory, run_id=run_id, reason="artifact_write_failed")
        return _failed_result("artifact_write_failed", caller_ready=caller_ready)

    try:
        decision = dict(selector(reopened))
    except BaseException:
        _record_failure(directory, run_id=run_id, reason="artifact_schema_failed")
        return _failed_result("artifact_schema_failed", caller_ready=caller_ready)
    selection = {
        **decision,
        "telemetry_path": TELEMETRY_FILE,
        "telemetry_sha256": observed_sha,
        "telemetry_size_bytes": observed_size,
        "sample_record_count": len(reopened["samples"]),
        "aggregate_record_count": len(reopened["aggregates"]),
    }
    try:
        _assert_privacy_safe(selection)
        expected_decision = dict(select_shortest_passing_length(reopened))
    except (LengthArtifactV7Error, LengthContractV7Error):
        _record_failure(directory, run_id=run_id, reason="artifact_schema_failed")
        return _failed_result("artifact_schema_failed", caller_ready=caller_ready)
    if selection.get("status") not in {
        "length_frozen",
        "no_length_candidate_passed",
    } or any(selection.get(key) != value for key, value in expected_decision.items()):
        _record_failure(directory, run_id=run_id, reason="artifact_schema_failed")
        return _failed_result("artifact_schema_failed", caller_ready=caller_ready)
    try:
        selection_metadata = _atomic_write_json(
            directory / SELECTION_FILE, selection
        )
    except BaseException:
        _record_failure(directory, run_id=run_id, reason="artifact_write_failed")
        return _failed_result("artifact_write_failed", caller_ready=caller_ready)

    failed = selection["status"] == "no_length_candidate_passed"
    if failed:
        _record_failure(
            directory, run_id=run_id, reason="no_length_candidate_passed"
        )
    try:
        _write_final_index(directory, run_id=run_id, failed=failed)
        readiness = derive_length_readiness(directory, caller_ready=caller_ready)
        _atomic_write_json(directory / READINESS_FILE, readiness)
    except BaseException:
        _record_failure(directory, run_id=run_id, reason="artifact_write_failed")
        return _failed_result("artifact_write_failed", caller_ready=caller_ready)

    return {
        **readiness,
        "status": selection["status"],
        "selected_response_length": selection.get("selected_response_length"),
        "telemetry_sha256": observed_sha,
        "telemetry_size_bytes": observed_size,
        "selection_sha256": selection_metadata["sha256"],
        "selection_size_bytes": selection_metadata["size_bytes"],
        "selection": selection,
    }


__all__ = [
    "LengthArtifactV7Error",
    "atomic_write_privacy_safe_json",
    "commit_length_qualification",
    "derive_length_readiness",
    "sha256_file",
    "validate_length_telemetry",
]
