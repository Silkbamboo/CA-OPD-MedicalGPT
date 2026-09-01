"""Bounded 1024-token B2 calibration package derived from the 768 failure.

The module is CPU-import safe.  It accepts only the preserved P4.8b package,
the disk-sealed pure-length failure, and the already observed P4.7 1024
candidate.  It never imports a model runtime and registers no length beyond
1024.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from src.opd.production_b2_calibration_contract_v1 import (
    B2_CALIBRATION_STEPS,
    B2CalibrationContractV1Error,
    FRESH_STUDENT_INITIALIZATION,
    canonical_json_sha256,
    validate_step_record,
)
from src.opd.production_b2_calibration_package_v2 import (
    B2CalibrationPackageV2Error,
    directory_tree_sha256,
    verify_replacement_package,
)
from src.opd.production_b2_data_v2 import (
    B2DataAuthorityV2Error,
    resolve_b2_data_authority,
    stream_sha256,
    validate_b2_prompt_schedule,
)


PACKAGE_VERSION = "p4_8c_v3"
SELECTED_RESPONSE_LENGTH = 1024
SOURCE_RESPONSE_LENGTH = 768
PACKAGE_FILES = frozenset(
    {
        "manifest_migration_attestation.json",
        "prompt_schedule.json",
        "length_escalation_attestation.json",
        "b2_20_step_calibration_config.json",
        "b2_20_step_calibration_run_card.json",
        "b2_authorization.json",
        "package_index.json",
        "readiness.json",
    }
)
COMPONENT_ORDER = (
    "manifest_migration_attestation.json",
    "prompt_schedule.json",
    "length_escalation_attestation.json",
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
)


class B2CalibrationPackageV3Error(RuntimeError):
    """A bounded 1024 package or its source evidence failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationPackageV3Error(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B2CalibrationPackageV3Error(
            f"package document is not canonical JSON: {type(error).__name__}"
        ) from error


def _payload_entry(name: str, value: Any) -> dict[str, Any]:
    payload = _canonical_bytes(value)
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2CalibrationPackageV3Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if path.read_bytes() != payload:
            _fail(f"atomic package write verification failed: {path.name}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_index(root: Path, name: str, kind: str) -> dict[str, Any]:
    value = _read_json(root / name, name)
    entries = value.get("artifacts")
    if not (
        value.get("schema_version") == 1
        and value.get("artifact_kind") == kind
        and isinstance(entries, list)
        and value.get("artifact_count") == len(entries)
    ):
        _fail(f"{name} envelope differs")
    seen: set[str] = set()
    for raw in entries:
        if not (
            isinstance(raw, Mapping)
            and set(raw) == {"path", "sha256", "size_bytes"}
            and isinstance(raw.get("path"), str)
            and raw["path"] not in seen
        ):
            _fail(f"{name} entry differs")
        seen.add(raw["path"])
        path = root / raw["path"]
        if not (
            path.is_file()
            and not path.is_symlink()
            and stream_sha256(path) == raw["sha256"]
            and path.stat().st_size == raw["size_bytes"]
        ):
            _fail(f"{name} indexed SHA/size differs")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail("source metrics are absent or symlinked")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise B2CalibrationPackageV3Error(
                    f"source metrics line {number} is invalid"
                ) from error
            if not isinstance(value, dict):
                _fail(f"source metrics line {number} is not an object")
            rows.append(value)
    return rows


def build_length_escalation_attestation(
    *,
    source_package_dir: str | Path,
    source_calibration_dir: str | Path,
    p4_7_length_telemetry_path: str | Path,
    canonical_manifest_path: str | Path,
) -> dict[str, Any]:
    """Recompute the exact evidence that permits one fresh 1024 rerun."""

    try:
        source_package = verify_replacement_package(
            source_package_dir,
            canonical_manifest_path=canonical_manifest_path,
        )
    except B2CalibrationPackageV2Error as error:
        raise B2CalibrationPackageV3Error(str(error)) from error
    root = Path(source_calibration_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        _fail("source 768 calibration output is absent or a symlink")
    evidence = _verify_index(
        root,
        "evidence_index.json",
        "b2_calibration_evidence_index_v1",
    )
    final = _verify_index(
        root, "final_index.json", "b2_calibration_final_index_v1"
    )
    failure = _read_json(root / "failure.json", "source failure")
    summary = _read_json(root / "summary.json", "source summary")
    readiness = _read_json(root / "readiness.json", "source readiness")
    cleanup = _read_json(root / "cleanup.json", "source cleanup")
    abort = _read_json(
        root / "length_abort_recommendation.json", "source length abort"
    )
    binding = _read_json(root / "package_binding.json", "source package binding")
    config = _read_json(root / "config.yaml", "source runtime config")
    if not (
        failure.get("status") == "failed_b2_calibration_length_insufficient"
        and failure.get("primary_failure_code")
        == "failed_b2_calibration_length_insufficient"
        and failure.get("failure_phase") == "rolling_length_gate"
        and failure.get("completed_steps") == 5
        and failure.get("requested_steps") == B2_CALIBRATION_STEPS
        and summary.get("status") == failure.get("status")
        and summary.get("steps_completed") == 5
        and summary.get("selected_response_length") == SOURCE_RESPONSE_LENGTH
        and readiness.get("status") == failure.get("status")
        and readiness.get("ready") is False
        and readiness.get("B2_calibration_complete") is False
        and readiness.get("B2_formal_authorized") is False
        and cleanup.get("cleanup_complete") is True
        and cleanup.get("gpu_memory_used_mib") == [0, 0]
        and cleanup.get("compute_pids") == []
        and cleanup.get("residual_worker_pids") == []
        and config.get("selected_response_length") == SOURCE_RESPONSE_LENGTH
        and binding.get("selected_response_length") == SOURCE_RESPONSE_LENGTH
        and binding.get("package_content_sha256")
        == source_package["package_content_sha256"]
        and abort.get("status") == failure.get("status")
        and abort.get("selected_response_length") == SOURCE_RESPONSE_LENGTH
        and abort.get("failure_reasons")
        == ["medical_opd_o1_truncation_rate"]
        and abort.get("source_truncation_count", {}).get("medical_opd_o1") == 3
        and abort.get("source_truncation_count", {}).get("medical_opd_cmb") == 0
        and abort.get("escalation_recommendation", {}).get(
            "recommended_response_length"
        )
        == SELECTED_RESPONSE_LENGTH
        and abort.get("escalation_recommendation", {}).get(
            "requires_new_versioned_package"
        )
        is True
        and abort.get("escalation_recommendation", {}).get(
            "same_run_switch_allowed"
        )
        is False
    ):
        _fail("source 768 failure is not the registered pure-length branch")
    records = _read_metrics(root / "metrics.jsonl")
    if len(records) != 5:
        _fail("source 768 failure step evidence differs")
    try:
        for index, record in enumerate(records):
            validate_step_record(
                record,
                expected_step=index + 1,
                expected_version=index,
                selected_response_length=SOURCE_RESPONSE_LENGTH,
            )
    except B2CalibrationContractV1Error as error:
        raise B2CalibrationPackageV3Error(str(error)) from error
    health_anomaly_count = sum(
        int(bool(sample[field]))
        for record in records
        for sample in record["prompt_samples"]
        for field in (
            "invalid",
            "empty",
            "non_finite",
            "unexpected_think_tag",
            "repetition",
        )
    )
    telemetry_path = Path(p4_7_length_telemetry_path).resolve()
    telemetry = _read_json(telemetry_path, "P4.7 length telemetry")
    candidates = telemetry.get("aggregates")
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, Mapping) and item.get("candidate") == 1024
        ),
        None,
    ) if isinstance(candidates, list) else None
    if not (
        telemetry.get("artifact_kind") == "production_length_telemetry_v7"
        and telemetry.get("evidence_complete") is True
        and isinstance(candidate, Mapping)
        and candidate.get("passed") is True
        and candidate.get("failure_reasons") == []
        and candidate.get("overall_n") == 16
        and candidate.get("medical_opd_o1_n") == 8
        and candidate.get("medical_opd_cmb_n") == 8
        and candidate.get("overall_truncation_count") == 0
        and candidate.get("per_source", {}).get("medical_opd_o1", {}).get(
            "truncation_count"
        )
        == 0
        and candidate.get("per_source", {}).get("medical_opd_cmb", {}).get(
            "truncation_count"
        )
        == 0
        and candidate.get("invalid_count") == 0
        and candidate.get("empty_count") == 0
        and candidate.get("non_finite_count") == 0
        and candidate.get("unexpected_think_tag_count") == 0
        and candidate.get("repetition_count") == 0
        and health_anomaly_count == 0
    ):
        _fail("P4.7 1024 candidate or source health evidence differs")
    value = {
        "schema_version": 1,
        "artifact_kind": "p4_8c_b2_length_escalation_attestation_v1",
        "source_package": {
            "path": str(Path(source_package_dir).resolve()),
            "package_version": source_package["package_version"],
            "tree_sha256": directory_tree_sha256(source_package_dir),
            "package_content_sha256": source_package["package_content_sha256"],
            "package_index_sha256": source_package["package_index_sha256"],
            "authorization_sha256": source_package["authorization_sha256"],
        },
        "source_calibration": {
            "path": str(root),
            "failure_sha256": stream_sha256(root / "failure.json"),
            "summary_sha256": stream_sha256(root / "summary.json"),
            "readiness_sha256": stream_sha256(root / "readiness.json"),
            "metrics_sha256": stream_sha256(root / "metrics.jsonl"),
            "length_abort_sha256": stream_sha256(
                root / "length_abort_recommendation.json"
            ),
            "evidence_index_sha256": stream_sha256(
                root / "evidence_index.json"
            ),
            "final_index_sha256": stream_sha256(root / "final_index.json"),
            "cleanup_sha256": stream_sha256(root / "cleanup.json"),
            "evidence_artifact_count": evidence["artifact_count"],
            "final_artifact_count": final["artifact_count"],
        },
        "source_failure_status": failure["status"],
        "source_failure_phase": failure["failure_phase"],
        "source_completed_steps": len(records),
        "source_requested_steps": B2_CALIBRATION_STEPS,
        "source_failure_reasons": list(abort["failure_reasons"]),
        "health_anomaly_count": health_anomaly_count,
        "p4_7_length_telemetry": {
            "path": str(telemetry_path),
            "sha256": stream_sha256(telemetry_path),
            "candidate_1024_sha256": canonical_json_sha256(candidate),
        },
        "p4_7_1024_candidate_passed": True,
        "source_response_length": SOURCE_RESPONSE_LENGTH,
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "fresh_run_required": True,
        "fresh_base_and_zero_lora_required": True,
        "same_run_switch_allowed": False,
        "resume_source_run_allowed": False,
        "automatic_further_length_escalation_allowed": False,
        "decision": "run_one_versioned_fresh_1024_calibration",
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }
    value["attestation_content_sha256"] = canonical_json_sha256(value)
    return value


def build_length_escalation_package_documents(
    *,
    source_package_dir: str | Path,
    source_calibration_dir: str | Path,
    p4_7_length_telemetry_path: str | Path,
    package_dir: str | Path,
    runtime_output_dir: str | Path,
    runtime_run_id: str,
    canonical_manifest_path: str | Path,
) -> dict[str, dict[str, Any]]:
    if runtime_run_id != (
        "qwen3-4b-b2-medical-opd-calibration-p4-8c-1024-seed42"
    ):
        _fail("1024 runtime run ID differs from the registered fresh identity")
    try:
        source = verify_replacement_package(
            source_package_dir,
            canonical_manifest_path=canonical_manifest_path,
        )
    except B2CalibrationPackageV2Error as error:
        raise B2CalibrationPackageV3Error(str(error)) from error
    escalation = build_length_escalation_attestation(
        source_package_dir=source_package_dir,
        source_calibration_dir=source_calibration_dir,
        p4_7_length_telemetry_path=p4_7_length_telemetry_path,
        canonical_manifest_path=canonical_manifest_path,
    )
    package_root = Path(package_dir).resolve()
    output_root = Path(runtime_output_dir).resolve()
    migration = deepcopy(source["migration_attestation"])
    schedule = deepcopy(source["schedule"])
    config = deepcopy(source["config"])
    config["schema_id"] = "ca-opd/b2-medical-opd-calibration/v3"
    config["schema_version"] = 3
    config["package_version"] = PACKAGE_VERSION
    config["run"].update(
        {
            "run_id": runtime_run_id,
            "purpose": "P4.8c bounded 1024 B2 20-step calibration",
            "output_dir": str(output_root),
            "status": "authorized_not_started",
            "automatically_start": False,
        }
    )
    config["generation"]["max_new_tokens"] = SELECTED_RESPONSE_LENGTH
    config["data"]["schedule_path"] = str(package_root / "prompt_schedule.json")
    config["p4_8c_start_gate"] = {
        "package_version": PACKAGE_VERSION,
        "source_package_path": str(Path(source_package_dir).resolve()),
        "source_package_content_sha256": source["package_content_sha256"],
        "source_768_calibration_path": str(
            Path(source_calibration_dir).resolve()
        ),
        "length_escalation_attestation_sha256": escalation[
            "attestation_content_sha256"
        ],
        "p4_7_1024_candidate_sha256": escalation["p4_7_length_telemetry"][
            "candidate_1024_sha256"
        ],
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "fresh_run_only": True,
        "resume_source_run": False,
        "same_run_length_switch": False,
        "automatic_further_length_escalation": False,
        "requires_pre_model_semantic_gate": True,
    }
    config["authorization"]["source"] = "p4_8c_bounded_1024_escalation"
    config_sha = _payload_entry(
        "b2_20_step_calibration_config.json", config
    )["sha256"]
    run_card = deepcopy(source["run_card"])
    run_card.update(
        {
            "schema_version": 3,
            "schema_id": "ca-opd/b2-medical-opd-calibration-run-card/v3",
            "package_version": PACKAGE_VERSION,
            "run_id": runtime_run_id,
            "config_sha256": config_sha,
            "selected_response_length": SELECTED_RESPONSE_LENGTH,
            "length_escalation_attestation_path": (
                "length_escalation_attestation.json"
            ),
            "length_escalation_attestation_sha256": escalation[
                "attestation_content_sha256"
            ],
            "status": "authorized_not_started",
            "B2_calibration_started": False,
            "B2_formal_authorized": False,
            "automatically_start_formal_b2": False,
        }
    )
    run_card_sha = _payload_entry(
        "b2_20_step_calibration_run_card.json", run_card
    )["sha256"]
    authorization = deepcopy(source["authorization"])
    authorization.pop("authorization_content_sha256", None)
    authorization.update(
        {
            "schema_version": 3,
            "artifact_kind": "p4_8c_b2_20_step_calibration_authorization_v3",
            "package_version": PACKAGE_VERSION,
            "run_id": runtime_run_id,
            "selected_response_length": SELECTED_RESPONSE_LENGTH,
            "status": "authorized_not_started",
            "B2_authorized": True,
            "B2_started": False,
            "B2_calibration_started": False,
            "B2_calibration_complete": False,
            "B2_formal_authorized": False,
            "automatically_start_formal_b2": False,
        }
    )
    authorization["bindings"].update(
        {
            "config_sha256": config_sha,
            "run_card_sha256": run_card_sha,
            "length_escalation_attestation_sha256": escalation[
                "attestation_content_sha256"
            ],
            "source_768_failure_sha256": escalation[
                "source_calibration"
            ]["failure_sha256"],
            "source_768_final_index_sha256": escalation[
                "source_calibration"
            ]["final_index_sha256"],
            "source_768_cleanup_sha256": escalation[
                "source_calibration"
            ]["cleanup_sha256"],
            "p4_7_length_telemetry_sha256": escalation[
                "p4_7_length_telemetry"
            ]["sha256"],
            "p4_7_1024_candidate_sha256": escalation[
                "p4_7_length_telemetry"
            ]["candidate_1024_sha256"],
        }
    )
    authorization["superseded_length_source"] = {
        "package_path": str(Path(source_package_dir).resolve()),
        "package_content_sha256": source["package_content_sha256"],
        "calibration_output_path": str(Path(source_calibration_dir).resolve()),
        "failure_status": "failed_b2_calibration_length_insufficient",
        "source_response_length": SOURCE_RESPONSE_LENGTH,
        "old_output_retained": True,
    }
    authorization["authorization_content_sha256"] = canonical_json_sha256(
        authorization
    )
    documents: dict[str, dict[str, Any]] = {
        "manifest_migration_attestation.json": migration,
        "prompt_schedule.json": schedule,
        "length_escalation_attestation.json": escalation,
        "b2_20_step_calibration_config.json": config,
        "b2_20_step_calibration_run_card.json": run_card,
        "b2_authorization.json": authorization,
    }
    components = [_payload_entry(name, documents[name]) for name in COMPONENT_ORDER]
    package_content_sha = canonical_json_sha256(components)
    index = {
        "schema_version": 3,
        "artifact_kind": "p4_8c_b2_calibration_package_index_v3",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "components": components,
        "component_count": len(components),
        "package_content_sha256": package_content_sha,
        "source_768_package_retained": True,
        "source_768_failure_retained": True,
        "fresh_gpu_output_required": True,
        "automatic_further_length_escalation": False,
        "B2_formal_authorized": False,
    }
    documents["package_index.json"] = index
    readiness = {
        "schema_version": 3,
        "artifact_kind": "p4_8c_b2_calibration_cpu_readiness_v3",
        "status": "ready_waiting_for_gpu_b2_calibration_1024_revalidation",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "package_index_sha256": _payload_entry("package_index.json", index)[
            "sha256"
        ],
        "package_content_sha256": package_content_sha,
        "length_escalation_attestation_sha256": escalation[
            "attestation_content_sha256"
        ],
        "schedule_sha256": schedule["schedule_sha256"],
        "canonical_manifest_sha256": source["data_authority"][
            "manifest_sha256"
        ],
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "gpu_revalidation_pending": True,
        "B2_authorized": True,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "B2_started": False,
        "automatic_further_length_escalation": False,
        "isolation": deepcopy(authorization["isolation"]),
    }
    documents["readiness.json"] = readiness
    return documents


def materialize_length_escalation_package(**kwargs: Any) -> dict[str, Any]:
    package = Path(kwargs["package_dir"]).resolve()
    if package.exists() or package.is_symlink():
        _fail("1024 package output must be fresh")
    documents = build_length_escalation_package_documents(**kwargs)
    package.mkdir(parents=True, exist_ok=False)
    for name in (*COMPONENT_ORDER, "package_index.json", "readiness.json"):
        _atomic_json(package / name, documents[name])
    directory_fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    audit = verify_length_escalation_package(
        package,
        canonical_manifest_path=kwargs["canonical_manifest_path"],
    )
    return {
        "status": "ready_waiting_for_gpu_b2_calibration_1024_revalidation",
        "package_dir": str(package),
        "package_content_sha256": audit["package_content_sha256"],
        "package_index_sha256": audit["package_index_sha256"],
        "authorization_sha256": audit["authorization_sha256"],
        "config_sha256": audit["config_sha256"],
        "run_card_sha256": audit["run_card_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "length_escalation_attestation_sha256": audit[
            "length_escalation_attestation_sha256"
        ],
    }


def verify_length_escalation_package(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    if package.is_symlink() or not package.is_dir():
        _fail("1024 package is absent or a symlink")
    names = {item.name for item in package.iterdir()}
    if names != set(PACKAGE_FILES) or any(
        item.is_symlink() or not item.is_file() for item in package.iterdir()
    ):
        _fail("1024 package file graph differs")
    docs = {name: _read_json(package / name, name) for name in PACKAGE_FILES}
    index = docs["package_index.json"]
    components = [_payload_entry(name, docs[name]) for name in COMPONENT_ORDER]
    if not (
        index.get("schema_version") == 3
        and index.get("artifact_kind")
        == "p4_8c_b2_calibration_package_index_v3"
        and index.get("package_version") == PACKAGE_VERSION
        and index.get("components") == components
        and index.get("component_count") == len(components)
        and index.get("package_content_sha256")
        == canonical_json_sha256(components)
        and index.get("automatic_further_length_escalation") is False
        and index.get("B2_formal_authorized") is False
    ):
        _fail("1024 package component SHA/contract differs")
    index_sha = stream_sha256(package / "package_index.json")
    readiness = docs["readiness.json"]
    if not (
        readiness.get("status")
        == "ready_waiting_for_gpu_b2_calibration_1024_revalidation"
        and readiness.get("package_version") == PACKAGE_VERSION
        and readiness.get("package_index_sha256") == index_sha
        and readiness.get("package_content_sha256")
        == index["package_content_sha256"]
        and readiness.get("selected_response_length")
        == SELECTED_RESPONSE_LENGTH
        and readiness.get("B2_authorized") is True
        and readiness.get("B2_calibration_started") is False
        and readiness.get("B2_formal_authorized") is False
        and readiness.get("automatic_further_length_escalation") is False
    ):
        _fail("1024 package readiness binding differs")
    config = docs["b2_20_step_calibration_config.json"]
    card = docs["b2_20_step_calibration_run_card.json"]
    authorization = docs["b2_authorization.json"]
    escalation = docs["length_escalation_attestation.json"]
    config_sha = stream_sha256(package / "b2_20_step_calibration_config.json")
    card_sha = stream_sha256(package / "b2_20_step_calibration_run_card.json")
    authorization_sha = stream_sha256(package / "b2_authorization.json")
    escalation_content = dict(escalation)
    claimed_escalation = escalation_content.pop("attestation_content_sha256", None)
    auth_content = dict(authorization)
    claimed_auth = auth_content.pop("authorization_content_sha256", None)
    data = config.get("data")
    run = config.get("run")
    generation = config.get("generation")
    execution = config.get("execution")
    student = config.get("student_initialization")
    isolation = config.get("isolation")
    bindings = authorization.get("bindings")
    if not all(
        isinstance(value, Mapping)
        for value in (data, run, generation, execution, student, isolation, bindings)
    ):
        _fail("1024 package sections are incomplete")
    if not (
        claimed_escalation == canonical_json_sha256(escalation_content)
        and claimed_auth == canonical_json_sha256(auth_content)
        and config.get("schema_id") == "ca-opd/b2-medical-opd-calibration/v3"
        and config.get("schema_version") == 3
        and config.get("package_version") == PACKAGE_VERSION
        and run.get("stage") == "b2_calibration"
        and run.get("seed") == 42
        and run.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and generation.get("max_new_tokens") == SELECTED_RESPONSE_LENGTH
        and execution.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and execution.get("calibration_only") is True
        and student.get("mode") == FRESH_STUDENT_INITIALIZATION
        and student.get("source_adapter_path") is None
        and student.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and data.get("schedule_path") == str(package / "prompt_schedule.json")
        and data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
        and all(
            isolation.get(field) is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
        and card.get("schema_version") == 3
        and card.get("package_version") == PACKAGE_VERSION
        and card.get("config_sha256") == config_sha
        and card.get("selected_response_length") == SELECTED_RESPONSE_LENGTH
        and card.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and card.get("B2_formal_authorized") is False
        and authorization.get("artifact_kind")
        == "p4_8c_b2_20_step_calibration_authorization_v3"
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_calibration_started") is False
        and authorization.get("B2_formal_authorized") is False
        and authorization.get("automatically_start_formal_b2") is False
        and bindings.get("config_sha256") == config_sha
        and bindings.get("run_card_sha256") == card_sha
        and bindings.get("length_escalation_attestation_sha256")
        == claimed_escalation
        and config.get("p4_8c_start_gate", {}).get(
            "automatic_further_length_escalation"
        )
        is False
    ):
        _fail("1024 package frozen contract or SHA differs")
    rebuilt = build_length_escalation_attestation(
        source_package_dir=escalation["source_package"]["path"],
        source_calibration_dir=escalation["source_calibration"]["path"],
        p4_7_length_telemetry_path=escalation["p4_7_length_telemetry"]["path"],
        canonical_manifest_path=canonical_manifest_path,
    )
    if rebuilt != escalation:
        _fail("1024 escalation attestation no longer rebuilds from disk")
    try:
        authority = resolve_b2_data_authority(
            data["prompt_manifest_path"],
            expected_manifest_sha256=data["prompt_manifest_sha256"],
            canonical_manifest_path=canonical_manifest_path,
        )
        schedule_audit = validate_b2_prompt_schedule(
            docs["prompt_schedule.json"], authority=authority
        )
    except B2DataAuthorityV2Error as error:
        raise B2CalibrationPackageV3Error(str(error)) from error
    if not (
        data.get("schedule_sha256") == schedule_audit["schedule_sha256"]
        and bindings.get("canonical_manifest_sha256")
        == authority["manifest_sha256"]
        and bindings.get("schedule_sha256") == schedule_audit["schedule_sha256"]
        and readiness.get("canonical_manifest_sha256")
        == authority["manifest_sha256"]
        and readiness.get("schedule_sha256") == schedule_audit["schedule_sha256"]
    ):
        _fail("1024 package provider/manifest/schedule binding differs")
    return {
        "package_dir": str(package),
        "package_version": PACKAGE_VERSION,
        "package_content_sha256": index["package_content_sha256"],
        "package_index_sha256": index_sha,
        "config_sha256": config_sha,
        "run_card_sha256": card_sha,
        "authorization_sha256": authorization_sha,
        "authorization_content_sha256": claimed_auth,
        "length_escalation_attestation_sha256": claimed_escalation,
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "optimizer_steps": B2_CALIBRATION_STEPS,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "qualification_v2_path": config["qualification"]["v2_checkpoint_path"],
        "qualification_v2_tensor_sha256": config["qualification"][
            "v2_tensor_sha256"
        ],
        "formal_run_root": config["qualification"]["output_path"],
        "runtime_run_id": run["run_id"],
        "runtime_output_dir": run["output_dir"],
        "data_authority": authority,
        "schedule": dict(docs["prompt_schedule.json"]),
        "schedule_audit": schedule_audit,
        "teacher": {
            "data_manifest_sha256": authority["manifest_sha256"],
            "adapter_path": config.get("teacher", {}).get("adapter_path"),
            "adapter_sha256": config.get("teacher", {}).get("adapter_sha256"),
            "adapter_weight_sha256": config.get("teacher", {}).get(
                "adapter_weight_sha256"
            ),
            "manifest_path": config.get("teacher", {}).get("manifest_path"),
            "manifest_sha256": config.get("teacher", {}).get("manifest_sha256"),
            "role": config.get("teacher", {}).get("role"),
            "same_token_scoring": config.get("teacher", {}).get(
                "same_token_scoring"
            ),
        },
        "config": config,
        "authorization": authorization,
        "run_card": card,
        "migration_attestation": docs["manifest_migration_attestation.json"],
        "length_escalation": escalation,
    }


def pre_model_semantic_preflight_v3(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
    call_counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    before = None if call_counters is None else dict(call_counters)
    audit = verify_length_escalation_package(
        package_dir, canonical_manifest_path=canonical_manifest_path
    )
    if call_counters is not None and dict(call_counters) != before:
        _fail("1024 semantic gate invoked a runtime loader/session")
    return {
        "status": "pre_model_semantic_gate_passed",
        "package_content_sha256": audit["package_content_sha256"],
        "manifest_sha256": audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "slot_count": audit["schedule"]["slot_count"],
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
        "student_session_calls": 0,
        "cuda_worker_calls": 0,
        "rollout_calls": 0,
        "audit": audit,
    }


__all__ = [
    "B2CalibrationPackageV3Error",
    "COMPONENT_ORDER",
    "PACKAGE_FILES",
    "PACKAGE_VERSION",
    "SELECTED_RESPONSE_LENGTH",
    "build_length_escalation_attestation",
    "build_length_escalation_package_documents",
    "materialize_length_escalation_package",
    "pre_model_semantic_preflight_v3",
    "verify_length_escalation_package",
]
