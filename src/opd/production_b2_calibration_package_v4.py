"""Versioned P4.8d memory-balanced 1024 B2 calibration package.

The builder consumes the sealed P4.8c package and failed three-step run.  It
changes only execution topology and memory evidence; scientific inputs remain
frozen.  The module is CPU-import safe.
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
from src.opd.production_b2_calibration_package_v2 import directory_tree_sha256
from src.opd.production_b2_calibration_package_v3 import (
    B2CalibrationPackageV3Error,
    verify_length_escalation_package,
)
from src.opd.production_b2_data_v2 import (
    B2DataAuthorityV2Error,
    resolve_b2_data_authority,
    stream_sha256,
    validate_b2_prompt_schedule,
)
from src.opd.production_b2_memory_execution_v1 import (
    CHECKPOINT_VERSIONS,
    MEMORY_EXECUTION_CONTRACT,
    MemoryExecutionV1Error,
    validate_memory_execution_contract,
)


PACKAGE_VERSION = "p4_8d_memory_v4"
SELECTED_RESPONSE_LENGTH = 1024
RUN_ID = "qwen3-4b-b2-medical-opd-calibration-p4-8d-1024-memory-seed42"
REPO_ROOT = Path(__file__).resolve().parents[2]
OOM_REPORT_PATH = REPO_ROOT / "reports/p4_8b_gpu_b2_calibration.json"
COMPONENT_ORDER = (
    "manifest_migration_attestation.json",
    "prompt_schedule.json",
    "length_escalation_attestation.json",
    "oom_memory_attestation.json",
    "memory_execution_contract.json",
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
)
PACKAGE_FILES = frozenset((*COMPONENT_ORDER, "package_index.json", "readiness.json"))


class B2CalibrationPackageV4Error(RuntimeError):
    """The P4.8d package or its immutable parent evidence failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationPackageV4Error(message)


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
        raise B2CalibrationPackageV4Error(
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
        raise B2CalibrationPackageV4Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail("source P4.8c metrics are absent or symlinked")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise B2CalibrationPackageV4Error(
                    f"source metrics line {number} is invalid"
                ) from error
            if not isinstance(row, dict):
                _fail(f"source metrics line {number} is not an object")
            rows.append(row)
    return rows


def _verify_index(root: Path, name: str, kind: str) -> dict[str, Any]:
    value = _read_json(root / name, name)
    artifacts = value.get("artifacts")
    if not (
        value.get("schema_version") == 1
        and value.get("artifact_kind") == kind
        and isinstance(artifacts, list)
        and value.get("artifact_count") == len(artifacts)
    ):
        _fail(f"source {name} envelope differs")
    for item in artifacts:
        if not isinstance(item, Mapping):
            _fail(f"source {name} entry differs")
        path = root / str(item.get("path"))
        if not (
            path.is_file()
            and not path.is_symlink()
            and stream_sha256(path) == item.get("sha256")
            and path.stat().st_size == item.get("size_bytes")
        ):
            _fail(f"source {name} SHA/size differs")
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


def build_oom_memory_attestation(
    *, source_package_dir: str | Path, source_calibration_dir: str | Path
) -> dict[str, Any]:
    root = Path(source_calibration_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        _fail("source P4.8c calibration output is absent or a symlink")
    failure = _read_json(root / "failure.json", "source P4.8c failure")
    summary = _read_json(root / "summary.json", "source P4.8c summary")
    readiness = _read_json(root / "readiness.json", "source P4.8c readiness")
    cleanup = _read_json(root / "cleanup.json", "source P4.8c cleanup")
    worker = _read_json(root / "worker_status.json", "source P4.8c worker status")
    evidence = _verify_index(
        root, "evidence_index.json", "b2_calibration_evidence_index_v1"
    )
    final = _verify_index(
        root, "final_index.json", "b2_calibration_final_index_v1"
    )
    records = _read_metrics(root / "metrics.jsonl")
    if not (
        failure.get("status") == "failed_b2_calibration_oom"
        and failure.get("primary_failure_code") == "failed_b2_calibration_oom"
        and failure.get("failure_phase") == "runtime_memory"
        and failure.get("completed_steps") == 3
        and failure.get("requested_steps") == B2_CALIBRATION_STEPS
        and worker.get("completed_steps") == 3
        and summary.get("steps_completed") == 3
        and summary.get("selected_response_length") == SELECTED_RESPONSE_LENGTH
        and readiness.get("ready") is False
        and readiness.get("B2_calibration_complete") is False
        and readiness.get("B2_formal_authorized") is False
        and cleanup.get("cleanup_complete") is True
        and cleanup.get("gpu_memory_used_mib") == [0, 0]
        and cleanup.get("compute_pids") == []
        and len(records) == 3
    ):
        _fail("source P4.8c OOM branch differs")
    try:
        for index, record in enumerate(records):
            validate_step_record(
                record,
                expected_step=index + 1,
                expected_version=index,
                selected_response_length=SELECTED_RESPONSE_LENGTH,
            )
    except B2CalibrationContractV1Error as error:
        raise B2CalibrationPackageV4Error(str(error)) from error
    v3 = root / "checkpoints/v3"
    forbidden_resume_files = {
        "checkpoint_manifest.json",
        "optimizer_state.pt",
        "rng_state.pt",
        "calibration_state.json",
    }
    if not v3.is_dir() or any((v3 / name).exists() for name in forbidden_resume_files):
        _fail("source v3 unexpectedly appears resume eligible")
    report = _read_json(OOM_REPORT_PATH, "P4.8b GPU report")
    attempts = report.get("attempts")
    attempt = next(
        (
            item
            for item in attempts
            if isinstance(item, Mapping)
            and item.get("run_id") == root.name
        ),
        None,
    ) if isinstance(attempts, list) else None
    oom = attempt.get("oom") if isinstance(attempt, Mapping) else None
    if not (
        report.get("status") == "failed_b2_calibration_oom"
        and isinstance(oom, Mapping)
        and oom.get("failed_before_step_4_commit") is True
        and oom.get("requested_allocation_mib") == 44.0
        and oom.get("gpu0_free_mib") == 6.81
        and oom.get("process_memory_in_use_gib") == 23.54
        and oom.get("pytorch_allocated_gib") == 22.84
        and oom.get("pytorch_reserved_unallocated_mib") == 344.78
    ):
        _fail("committed OOM diagnostic report differs")
    value = {
        "schema_version": 1,
        "artifact_kind": "p4_8d_oom_memory_migration_attestation_v1",
        "source_package": {
            "path": str(Path(source_package_dir).resolve()),
            "tree_sha256": directory_tree_sha256(source_package_dir),
        },
        "source_calibration": {
            "path": str(root),
            "tree_sha256": directory_tree_sha256(root),
            "failure_sha256": stream_sha256(root / "failure.json"),
            "summary_sha256": stream_sha256(root / "summary.json"),
            "readiness_sha256": stream_sha256(root / "readiness.json"),
            "metrics_sha256": stream_sha256(root / "metrics.jsonl"),
            "evidence_index_sha256": stream_sha256(root / "evidence_index.json"),
            "final_index_sha256": stream_sha256(root / "final_index.json"),
            "cleanup_sha256": stream_sha256(root / "cleanup.json"),
            "evidence_artifact_count": evidence["artifact_count"],
            "final_artifact_count": final["artifact_count"],
        },
        "oom_report": {
            "path": str(OOM_REPORT_PATH.resolve()),
            "sha256": stream_sha256(OOM_REPORT_PATH),
        },
        "source_status": "failed_b2_calibration_oom",
        "source_failure_phase": "runtime_memory",
        "source_completed_steps": 3,
        "source_policy_version": 3,
        "source_resume_eligible": False,
        "resume_source_run_allowed": False,
        "fresh_v0_required": True,
        "failed_before_step_4_commit": True,
        "requested_allocation_mib": 44.0,
        "gpu0_free_mib": 6.81,
        "process_memory_in_use_gib": 23.54,
        "pytorch_allocated_gib": 22.84,
        "pytorch_reserved_unallocated_mib": 344.78,
        "exact_failure_operation": None,
        "exact_step4_prompt_token_shape": None,
        "exact_step4_completion_token_shape": None,
        "observed_step_end_monotonic_growth": False,
        "memory_leak_proven": False,
        "allocator_fragmentation_proven": False,
        "decision": "fresh_1024_memory_balanced_revalidation",
        "historical_output_retained": True,
    }
    value["attestation_content_sha256"] = canonical_json_sha256(value)
    return value


def build_memory_execution_package_documents(
    *,
    source_package_dir: str | Path,
    source_calibration_dir: str | Path,
    package_dir: str | Path,
    runtime_output_dir: str | Path,
    runtime_run_id: str,
    canonical_manifest_path: str | Path,
    code_git_commit: str,
) -> dict[str, dict[str, Any]]:
    if runtime_run_id != RUN_ID:
        _fail("memory runtime run ID differs from the registered fresh identity")
    if not (
        isinstance(code_git_commit, str)
        and len(code_git_commit) == 40
        and all(character in "0123456789abcdef" for character in code_git_commit)
    ):
        _fail("memory package code Git commit is invalid")
    try:
        source = verify_length_escalation_package(
            source_package_dir, canonical_manifest_path=canonical_manifest_path
        )
    except B2CalibrationPackageV3Error as error:
        raise B2CalibrationPackageV4Error(str(error)) from error
    oom = build_oom_memory_attestation(
        source_package_dir=source_package_dir,
        source_calibration_dir=source_calibration_dir,
    )
    contract = validate_memory_execution_contract(MEMORY_EXECUTION_CONTRACT)
    package_root = Path(package_dir).resolve()
    output_root = Path(runtime_output_dir).resolve()
    migration = deepcopy(source["migration_attestation"])
    schedule = deepcopy(source["schedule"])
    escalation = deepcopy(source["length_escalation"])
    config = deepcopy(source["config"])
    config.update(
        {
            "schema_id": "ca-opd/b2-medical-opd-calibration/v4",
            "schema_version": 4,
            "package_version": PACKAGE_VERSION,
            "memory_execution": deepcopy(contract),
        }
    )
    config["run"].update(
        {
            "run_id": runtime_run_id,
            "purpose": "P4.8d memory-balanced 1024 B2 20-step calibration",
            "output_dir": str(output_root),
            "status": "authorized_not_started",
            "automatically_start": False,
        }
    )
    config["data"]["schedule_path"] = str(package_root / "prompt_schedule.json")
    config["generation"]["max_new_tokens"] = SELECTED_RESPONSE_LENGTH
    config["execution"].update(
        {
            "optimizer_steps": B2_CALIBRATION_STEPS,
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "target_logit_chunk_size": 128,
            "checkpoint_strategy": "step5_step10_step15_step20_and_final",
        }
    )
    config["p4_8d_start_gate"] = {
        "package_version": PACKAGE_VERSION,
        "source_package_path": str(Path(source_package_dir).resolve()),
        "source_package_content_sha256": source["package_content_sha256"],
        "source_calibration_path": str(Path(source_calibration_dir).resolve()),
        "oom_memory_attestation_sha256": oom["attestation_content_sha256"],
        "memory_execution_contract_sha256": canonical_json_sha256(contract),
        "code_git_commit": code_git_commit,
        "fresh_v0_required": True,
        "resume_source_run": False,
        "selected_response_length": 1024,
        "effective_batch_size": 4,
        "requires_memory_canary": True,
        "requires_six_step_drift_gate": True,
        "formal_b2_automatic_start": False,
    }
    config_sha = _payload_entry("b2_20_step_calibration_config.json", config)["sha256"]
    card = deepcopy(source["run_card"])
    card.update(
        {
            "schema_id": "ca-opd/b2-medical-opd-calibration-run-card/v4",
            "schema_version": 4,
            "package_version": PACKAGE_VERSION,
            "run_id": runtime_run_id,
            "config_sha256": config_sha,
            "selected_response_length": 1024,
            "optimizer_steps": 20,
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "checkpoint_versions": list(CHECKPOINT_VERSIONS),
            "oom_memory_attestation_sha256": oom["attestation_content_sha256"],
            "memory_execution_contract_sha256": canonical_json_sha256(contract),
            "code_git_commit": code_git_commit,
            "status": "authorized_not_started",
            "B2_calibration_started": False,
            "B2_formal_authorized": False,
            "automatically_start_formal_b2": False,
        }
    )
    card_sha = _payload_entry("b2_20_step_calibration_run_card.json", card)["sha256"]
    authorization = deepcopy(source["authorization"])
    authorization.pop("authorization_content_sha256", None)
    authorization.update(
        {
            "schema_version": 4,
            "artifact_kind": "p4_8d_b2_memory_calibration_authorization_v4",
            "package_version": PACKAGE_VERSION,
            "run_id": runtime_run_id,
            "selected_response_length": 1024,
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
            "run_card_sha256": card_sha,
            "oom_memory_attestation_sha256": oom["attestation_content_sha256"],
            "memory_execution_contract_sha256": canonical_json_sha256(contract),
            "source_p4_8c_package_content_sha256": source["package_content_sha256"],
            "source_p4_8c_failure_sha256": oom["source_calibration"]["failure_sha256"],
            "source_p4_8c_final_index_sha256": oom["source_calibration"]["final_index_sha256"],
            "code_git_commit": code_git_commit,
        }
    )
    authorization["source_oom_run"] = {
        "path": str(Path(source_calibration_dir).resolve()),
        "status": "failed_b2_calibration_oom",
        "completed_steps": 3,
        "resume_allowed": False,
        "retained": True,
    }
    authorization["authorization_content_sha256"] = canonical_json_sha256(authorization)
    documents: dict[str, dict[str, Any]] = {
        "manifest_migration_attestation.json": migration,
        "prompt_schedule.json": schedule,
        "length_escalation_attestation.json": escalation,
        "oom_memory_attestation.json": oom,
        "memory_execution_contract.json": contract,
        "b2_20_step_calibration_config.json": config,
        "b2_20_step_calibration_run_card.json": card,
        "b2_authorization.json": authorization,
    }
    components = [_payload_entry(name, documents[name]) for name in COMPONENT_ORDER]
    package_content_sha = canonical_json_sha256(components)
    index = {
        "schema_version": 4,
        "artifact_kind": "p4_8d_b2_memory_package_index_v4",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "components": components,
        "component_count": len(components),
        "package_content_sha256": package_content_sha,
        "source_p4_8c_package_retained": True,
        "source_p4_8c_failure_retained": True,
        "fresh_gpu_output_required": True,
        "fresh_v0_required": True,
        "B2_formal_authorized": False,
    }
    documents["package_index.json"] = index
    index_sha = _payload_entry("package_index.json", index)["sha256"]
    documents["readiness.json"] = {
        "schema_version": 4,
        "artifact_kind": "p4_8d_b2_memory_cpu_readiness_v4",
        "status": "ready_waiting_for_gpu_b2_memory_revalidation",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "package_index_sha256": index_sha,
        "package_content_sha256": package_content_sha,
        "authorization_sha256": _payload_entry("b2_authorization.json", authorization)["sha256"],
        "memory_execution_contract_sha256": canonical_json_sha256(contract),
        "oom_memory_attestation_sha256": oom["attestation_content_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "canonical_manifest_sha256": source["data_authority"]["manifest_sha256"],
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "effective_batch_size": 4,
        "gpu_memory_revalidation_pending": True,
        "B2_authorized": True,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "B2_started": False,
        "isolation": deepcopy(authorization["isolation"]),
    }
    return documents


def materialize_memory_execution_package(**kwargs: Any) -> dict[str, Any]:
    package = Path(kwargs["package_dir"]).resolve()
    if package.exists() or package.is_symlink():
        _fail("memory package output must be fresh")
    documents = build_memory_execution_package_documents(**kwargs)
    package.mkdir(parents=True, exist_ok=False)
    for name in (*COMPONENT_ORDER, "package_index.json", "readiness.json"):
        _atomic_json(package / name, documents[name])
    directory_fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    audit = verify_memory_execution_package(
        package, canonical_manifest_path=kwargs["canonical_manifest_path"]
    )
    return {
        "status": "ready_waiting_for_gpu_b2_memory_revalidation",
        "package_dir": str(package),
        "package_content_sha256": audit["package_content_sha256"],
        "package_index_sha256": audit["package_index_sha256"],
        "authorization_sha256": audit["authorization_sha256"],
        "config_sha256": audit["config_sha256"],
        "run_card_sha256": audit["run_card_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "oom_memory_attestation_sha256": audit["oom_memory_attestation_sha256"],
        "memory_execution_contract_sha256": audit["memory_execution_contract_sha256"],
    }


def verify_memory_execution_package(
    package_dir: str | Path, *, canonical_manifest_path: str | Path
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    if package.is_symlink() or not package.is_dir():
        _fail("memory package is absent or a symlink")
    entries = list(package.iterdir())
    if {item.name for item in entries} != set(PACKAGE_FILES) or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        _fail("memory package file graph differs")
    docs = {name: _read_json(package / name, name) for name in PACKAGE_FILES}
    index = docs["package_index.json"]
    components = [_payload_entry(name, docs[name]) for name in COMPONENT_ORDER]
    if not (
        index.get("schema_version") == 4
        and index.get("artifact_kind") == "p4_8d_b2_memory_package_index_v4"
        and index.get("package_version") == PACKAGE_VERSION
        and index.get("components") == components
        and index.get("component_count") == len(components)
        and index.get("package_content_sha256") == canonical_json_sha256(components)
        and index.get("fresh_v0_required") is True
        and index.get("B2_formal_authorized") is False
    ):
        _fail("memory package component SHA/contract differs")
    config = docs["b2_20_step_calibration_config.json"]
    card = docs["b2_20_step_calibration_run_card.json"]
    auth = docs["b2_authorization.json"]
    readiness = docs["readiness.json"]
    contract = docs["memory_execution_contract.json"]
    oom = docs["oom_memory_attestation.json"]
    config_sha = stream_sha256(package / "b2_20_step_calibration_config.json")
    card_sha = stream_sha256(package / "b2_20_step_calibration_run_card.json")
    auth_sha = stream_sha256(package / "b2_authorization.json")
    index_sha = stream_sha256(package / "package_index.json")
    try:
        validate_memory_execution_contract(contract)
    except MemoryExecutionV1Error as error:
        raise B2CalibrationPackageV4Error(str(error)) from error
    oom_content = dict(oom)
    claimed_oom = oom_content.pop("attestation_content_sha256", None)
    auth_content = dict(auth)
    claimed_auth = auth_content.pop("authorization_content_sha256", None)
    if not (
        claimed_oom == canonical_json_sha256(oom_content)
        and claimed_auth == canonical_json_sha256(auth_content)
        and config.get("schema_id") == "ca-opd/b2-medical-opd-calibration/v4"
        and config.get("schema_version") == 4
        and config.get("package_version") == PACKAGE_VERSION
        and config.get("run", {}).get("run_id") == RUN_ID
        and config.get("run", {}).get("seed") == 42
        and config.get("run", {}).get("optimizer_steps") == 20
        and config.get("generation", {}).get("max_new_tokens") == 1024
        and config.get("execution", {}).get("physical_microbatch_size") == 1
        and config.get("execution", {}).get("gradient_accumulation_steps") == 4
        and config.get("execution", {}).get("effective_batch_size") == 4
        and config.get("execution", {}).get("checkpoint_strategy")
        == "step5_step10_step15_step20_and_final"
        and config.get("student_initialization", {}).get("mode")
        == FRESH_STUDENT_INITIALIZATION
        and config.get("student_initialization", {}).get("initial_logical_version") == 0
        and config.get("student_initialization", {}).get("source_adapter_path") is None
        and config.get("p4_8d_start_gate", {}).get("resume_source_run") is False
        and config.get("memory_execution") == contract
        and card.get("config_sha256") == config_sha
        and card.get("physical_microbatch_size") == 1
        and card.get("gradient_accumulation_steps") == 4
        and card.get("effective_batch_size") == 4
        and card.get("checkpoint_versions") == list(CHECKPOINT_VERSIONS)
        and card.get("B2_formal_authorized") is False
        and auth.get("artifact_kind") == "p4_8d_b2_memory_calibration_authorization_v4"
        and auth.get("B2_authorized") is True
        and auth.get("B2_calibration_started") is False
        and auth.get("B2_formal_authorized") is False
        and auth.get("bindings", {}).get("config_sha256") == config_sha
        and auth.get("bindings", {}).get("run_card_sha256") == card_sha
        and auth.get("bindings", {}).get("oom_memory_attestation_sha256") == claimed_oom
        and readiness.get("status") == "ready_waiting_for_gpu_b2_memory_revalidation"
        and readiness.get("package_index_sha256") == index_sha
        and readiness.get("package_content_sha256") == index["package_content_sha256"]
        and readiness.get("B2_calibration_complete") is False
        and readiness.get("B2_formal_authorized") is False
    ):
        _fail("memory package frozen contract or SHA differs")
    rebuilt_oom = build_oom_memory_attestation(
        source_package_dir=oom["source_package"]["path"],
        source_calibration_dir=oom["source_calibration"]["path"],
    )
    if rebuilt_oom != oom:
        _fail("OOM memory attestation no longer rebuilds from disk")
    try:
        parent = verify_length_escalation_package(
            oom["source_package"]["path"],
            canonical_manifest_path=canonical_manifest_path,
        )
        authority = resolve_b2_data_authority(
            config["data"]["prompt_manifest_path"],
            expected_manifest_sha256=config["data"]["prompt_manifest_sha256"],
            canonical_manifest_path=canonical_manifest_path,
        )
        schedule_audit = validate_b2_prompt_schedule(
            docs["prompt_schedule.json"], authority=authority
        )
    except (B2CalibrationPackageV3Error, B2DataAuthorityV2Error) as error:
        raise B2CalibrationPackageV4Error(str(error)) from error
    if not (
        parent["package_content_sha256"]
        == auth["bindings"]["source_p4_8c_package_content_sha256"]
        and authority["manifest_sha256"] == readiness["canonical_manifest_sha256"]
        and schedule_audit["schedule_sha256"] == readiness["schedule_sha256"]
        and config["data"]["schedule_path"] == str(package / "prompt_schedule.json")
    ):
        _fail("memory package parent/provider/schedule binding differs")
    return {
        "package_dir": str(package),
        "package_version": PACKAGE_VERSION,
        "package_content_sha256": index["package_content_sha256"],
        "package_index_sha256": index_sha,
        "authorization_sha256": auth_sha,
        "authorization_content_sha256": claimed_auth,
        "config_sha256": config_sha,
        "run_card_sha256": card_sha,
        "oom_memory_attestation_sha256": claimed_oom,
        "memory_execution_contract_sha256": canonical_json_sha256(contract),
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "qualification_v2_path": config["qualification"]["v2_checkpoint_path"],
        "qualification_v2_tensor_sha256": config["qualification"]["v2_tensor_sha256"],
        "formal_run_root": config["qualification"]["output_path"],
        "runtime_run_id": config["run"]["run_id"],
        "runtime_output_dir": config["run"]["output_dir"],
        "data_authority": authority,
        "schedule": dict(docs["prompt_schedule.json"]),
        "schedule_audit": schedule_audit,
        "teacher": parent["teacher"],
        "config": config,
        "run_card": card,
        "authorization": auth,
        "readiness": readiness,
        "migration_attestation": docs["manifest_migration_attestation.json"],
        "length_escalation": docs["length_escalation_attestation.json"],
        "oom_attestation": oom,
        "memory_execution_contract": contract,
        "code_git_commit": config["p4_8d_start_gate"]["code_git_commit"],
    }


def pre_model_semantic_preflight_v4(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
    call_counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    before = None if call_counters is None else dict(call_counters)
    audit = verify_memory_execution_package(
        package_dir, canonical_manifest_path=canonical_manifest_path
    )
    if call_counters is not None and dict(call_counters) != before:
        _fail("memory semantic gate invoked a runtime loader/session")
    return {
        "status": "pre_model_memory_semantic_gate_passed",
        "package_content_sha256": audit["package_content_sha256"],
        "manifest_sha256": audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "slot_count": audit["schedule"]["slot_count"],
        "selected_response_length": 1024,
        "physical_microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
        "student_session_calls": 0,
        "cuda_worker_calls": 0,
        "rollout_calls": 0,
        "audit": audit,
    }


__all__ = [
    "B2CalibrationPackageV4Error",
    "COMPONENT_ORDER",
    "PACKAGE_FILES",
    "PACKAGE_VERSION",
    "RUN_ID",
    "build_memory_execution_package_documents",
    "build_oom_memory_attestation",
    "materialize_memory_execution_package",
    "pre_model_semantic_preflight_v4",
    "verify_memory_execution_package",
]
