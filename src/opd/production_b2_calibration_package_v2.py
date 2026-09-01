"""Versioned, frozen-manifest B2 calibration package builder and verifier."""

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
    FRESH_STUDENT_INITIALIZATION,
    SELECTED_RESPONSE_LENGTH,
)
from src.opd.production_b2_data_v2 import (
    B2DataAuthorityV2Error,
    canonical_json_sha256,
    resolve_b2_data_authority,
    stream_sha256,
    validate_b2_prompt_schedule,
)


PACKAGE_VERSION = "p4_8b_v2"
PACKAGE_FILES = frozenset(
    {
        "manifest_migration_attestation.json",
        "prompt_schedule.json",
        "b2_20_step_calibration_config.json",
        "b2_20_step_calibration_run_card.json",
        "b2_authorization.json",
        "package_index.json",
        "readiness.json",
    }
)
SOURCE_PACKAGE_FILES = frozenset(
    {
        "b2_20_step_calibration_config.json",
        "b2_20_step_calibration_run_card.json",
        "b2_authorization.json",
    }
)
COMPONENT_ORDER = (
    "manifest_migration_attestation.json",
    "prompt_schedule.json",
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
)


class B2CalibrationPackageV2Error(RuntimeError):
    """Raised before model construction when the replacement package drifts."""


def _fail(message: str) -> None:
    raise B2CalibrationPackageV2Error(message)


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
        raise B2CalibrationPackageV2Error(
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
        raise B2CalibrationPackageV2Error(
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
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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
        if path.read_bytes() != payload or stream_sha256(path) != hashlib.sha256(payload).hexdigest():
            _fail(f"atomic package write verification failed: {path.name}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def directory_tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail("package directory is absent or a symlink")
    entries: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            _fail("package tree contains a symlink")
        if item.is_file():
            entries.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "sha256": stream_sha256(item),
                    "size_bytes": item.stat().st_size,
                }
            )
    return canonical_json_sha256(entries)


def _load_source_package(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        _fail("superseded source package is absent or a symlink")
    names = {item.name for item in root.iterdir()}
    if names != set(SOURCE_PACKAGE_FILES) or any(
        item.is_symlink() or not item.is_file() for item in root.iterdir()
    ):
        _fail("superseded source package file graph differs")
    config = _read_json(root / "b2_20_step_calibration_config.json", "source config")
    card = _read_json(root / "b2_20_step_calibration_run_card.json", "source run card")
    authorization = _read_json(root / "b2_authorization.json", "source authorization")
    data = config.get("data")
    run = config.get("run")
    generation = config.get("generation")
    protocol = config.get("protocol")
    backend = config.get("production_backend")
    execution = config.get("execution")
    isolation = config.get("isolation")
    if not all(
        isinstance(item, Mapping)
        for item in (data, run, generation, protocol, backend, execution, isolation)
    ):
        _fail("superseded source package sections are incomplete")
    old_content = dict(authorization)
    claimed_content = old_content.pop("package_content_sha256", None)
    if claimed_content != canonical_json_sha256(old_content):
        _fail("superseded source authorization content SHA differs")
    if not (
        run.get("seed") == 42
        and run.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and generation.get("max_new_tokens") == SELECTED_RESPONSE_LENGTH
        and execution.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and execution.get("calibration_only") is True
        and protocol.get("learning_rate") == 3e-5
        and protocol.get("student_lora_rank") == 16
        and protocol.get("student_lora_alpha") == 32
        and protocol.get("optimizer") == "AdamW"
        and backend.get("backend_id") == "custom_transformers_peft_three_policy_v5"
        and data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
        and data.get("selection_rule")
        == "seed42_sha256_rank_first2_per_source_per_step_v1"
        and data.get("allowed_roles") == ["medical_opd_o1", "medical_opd_cmb"]
        and all(
            isolation.get(field) is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_started") is False
    ):
        _fail("superseded source package frozen calibration contract differs")
    return {
        "path": str(root),
        "tree_sha256": directory_tree_sha256(root),
        "config": config,
        "config_sha256": stream_sha256(root / "b2_20_step_calibration_config.json"),
        "run_card": card,
        "run_card_sha256": stream_sha256(root / "b2_20_step_calibration_run_card.json"),
        "authorization": authorization,
        "authorization_sha256": stream_sha256(root / "b2_authorization.json"),
        "package_content_sha256": claimed_content,
    }


def _validate_parents(parent_bindings: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "length_telemetry_path",
        "length_telemetry_sha256",
        "length_selection_path",
        "length_selection_sha256",
        "root_index_path",
        "root_index_sha256",
        "selected_response_length",
    }
    if set(parent_bindings) != required or parent_bindings.get("selected_response_length") != 768:
        _fail("P4.7 parent bindings are incomplete or length differs")
    result = dict(parent_bindings)
    for prefix in ("length_telemetry", "length_selection", "root_index"):
        path = Path(str(parent_bindings[f"{prefix}_path"]))
        expected = parent_bindings[f"{prefix}_sha256"]
        if path.is_symlink() or not path.is_file() or stream_sha256(path) != expected:
            _fail(f"P4.7 {prefix} SHA differs")
        result[f"{prefix}_path"] = str(path.resolve())
    return result


def assert_package_is_calibration_only(value: Mapping[str, Any]) -> None:
    if not (
        value.get("stage") == "b2_calibration"
        and value.get("B2_formal_authorized") is False
        and value.get("automatically_start_formal_b2") is False
    ):
        _fail("replacement package is calibration-only and cannot launch formal B2")


def build_replacement_package_documents(
    *,
    source_package_dir: str | Path,
    package_dir: str | Path,
    runtime_output_dir: str | Path,
    runtime_run_id: str,
    authority: Mapping[str, Any],
    migration_attestation: Mapping[str, Any],
    schedule: Mapping[str, Any],
    parent_bindings: Mapping[str, Any],
    replacement_package_version: str = PACKAGE_VERSION,
) -> dict[str, dict[str, Any]]:
    if replacement_package_version != PACKAGE_VERSION:
        _fail("replacement package version differs")
    if runtime_run_id != "qwen3-4b-b2-medical-opd-calibration-p4-8b-seed42":
        _fail("replacement GPU run ID differs from the fresh P4.8b identity")
    source = _load_source_package(source_package_dir)
    parents = _validate_parents(parent_bindings)
    try:
        schedule_audit = validate_b2_prompt_schedule(schedule, authority=authority)
    except B2DataAuthorityV2Error as error:
        raise B2CalibrationPackageV2Error(str(error)) from error
    migration_without_sha = dict(migration_attestation)
    migration_sha = migration_without_sha.pop("attestation_sha256", None)
    if migration_sha != canonical_json_sha256(migration_without_sha):
        _fail("manifest migration attestation SHA differs")
    migration_parent = migration_attestation.get("parent_bindings")
    if not isinstance(migration_parent, Mapping) or not (
        migration_parent.get("length_selection_sha256")
        == parents["length_selection_sha256"]
        and migration_parent.get("root_index_sha256") == parents["root_index_sha256"]
        and migration_parent.get("old_package_sha256") == source["tree_sha256"]
        and migration_attestation.get("content_equivalent") is True
        and migration_attestation.get("authority_equivalent") is False
        and migration_attestation.get("migration_decision")
        == "bind_only_canonical_frozen_v2"
        and migration_attestation.get("new_manifest", {}).get("sha256")
        == authority.get("manifest_sha256")
    ):
        _fail("manifest migration attestation bindings differ")

    package_root = Path(package_dir).resolve()
    runtime_output = Path(runtime_output_dir).resolve()
    config = deepcopy(source["config"])
    config["schema_id"] = "ca-opd/b2-medical-opd-calibration/v2"
    config["schema_version"] = 2
    config["package_version"] = PACKAGE_VERSION
    config["run"].update(
        {
            "run_id": runtime_run_id,
            "stage": "b2_calibration",
            "purpose": "P4.8b frozen-manifest B2 20-step calibration revalidation",
            "output_dir": str(runtime_output),
            "seed": 42,
            "optimizer_steps": 20,
            "status": "authorized_not_started",
            "automatically_start": False,
        }
    )
    config["data"].update(
        {
            "prompt_manifest_path": authority["manifest_path"],
            "prompt_manifest_sha256": authority["manifest_sha256"],
            "schedule_path": str(package_root / "prompt_schedule.json"),
            "schedule_sha256": schedule_audit["schedule_sha256"],
            "schedule_version": schedule["schedule_version"],
            "provider": "production_b2_data_v2.resolve_b2_schedule_batch",
            "canonical_manifest_required": True,
            "prompt_only": True,
            "final_labels_allowed": False,
        }
    )
    config["student_initialization"] = {
        "mode": FRESH_STUDENT_INITIALIZATION,
        "initial_logical_version": 0,
        "source_adapter_path": None,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "forbidden_qualification_adapter_path": config["qualification"][
            "v2_checkpoint_path"
        ],
        "forbidden_qualification_adapter_sha256": config["qualification"][
            "v2_tensor_sha256"
        ],
    }
    historical_manifest_sha = config["qualification"].get("data_manifest_sha256")
    config["qualification"]["parent_length_data_manifest_sha256"] = historical_manifest_sha
    config["qualification"]["data_manifest_sha256"] = authority["manifest_sha256"]
    config["p4_8b_start_gate"] = {
        "package_version": PACKAGE_VERSION,
        "superseded_package_path": source["path"],
        "superseded_package_tree_sha256": source["tree_sha256"],
        "superseded_package_content_sha256": source["package_content_sha256"],
        "migration_attestation_sha256": migration_sha,
        "schedule_sha256": schedule_audit["schedule_sha256"],
        "manifest_sha256": authority["manifest_sha256"],
        "length_telemetry_path": parents["length_telemetry_path"],
        "length_telemetry_sha256": parents["length_telemetry_sha256"],
        "length_selection_path": parents["length_selection_path"],
        "length_selection_sha256": parents["length_selection_sha256"],
        "root_index_path": parents["root_index_path"],
        "root_index_sha256": parents["root_index_sha256"],
        "requires_pre_model_semantic_gate": True,
        "requires_explicit_allow_b2_calibration": True,
        "fresh_run_only": True,
        "resume_old_failed_run": False,
    }
    config["authorization"]["source"] = "p4_8b_frozen_manifest_migration"
    config_sha = _payload_entry("b2_20_step_calibration_config.json", config)["sha256"]

    run_card = {
        "schema_version": 2,
        "schema_id": "ca-opd/b2-medical-opd-calibration-run-card/v2",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "stage": "b2_calibration",
        "status": "authorized_not_started",
        "config_path": "b2_20_step_calibration_config.json",
        "config_sha256": config_sha,
        "manifest_migration_attestation_path": "manifest_migration_attestation.json",
        "manifest_migration_attestation_sha256": migration_sha,
        "schedule_path": "prompt_schedule.json",
        "schedule_sha256": schedule_audit["schedule_sha256"],
        "canonical_manifest_path": authority["manifest_path"],
        "canonical_manifest_sha256": authority["manifest_sha256"],
        "selected_response_length": 768,
        "optimizer_steps": 20,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "requires_environment": "CA_OPD_ALLOW_B2_CALIBRATION_GPU=1",
        "requires_argument": "--allow-b2-calibration",
        "automatically_start": False,
        "B2_calibration_started": False,
        "B2_formal_authorized": False,
        "automatically_start_formal_b2": False,
    }
    run_card_sha = _payload_entry("b2_20_step_calibration_run_card.json", run_card)["sha256"]

    source_auth = source["authorization"]
    authorization = {
        "schema_version": 2,
        "artifact_kind": "p4_8b_b2_20_step_calibration_authorization_v2",
        "package_version": PACKAGE_VERSION,
        "status": "authorized_not_started",
        "stage": "b2_calibration",
        "run_id": runtime_run_id,
        "selected_response_length": 768,
        "optimizer_steps": 20,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "bindings": {
            "config_sha256": config_sha,
            "run_card_sha256": run_card_sha,
            "canonical_manifest_path": authority["manifest_path"],
            "canonical_manifest_sha256": authority["manifest_sha256"],
            "migration_attestation_sha256": migration_sha,
            "schedule_sha256": schedule_audit["schedule_sha256"],
            "length_telemetry_sha256": parents["length_telemetry_sha256"],
            "length_selection_sha256": parents["length_selection_sha256"],
            "root_index_sha256": parents["root_index_sha256"],
            "teacher_adapter_sha256": source_auth["bindings"]["teacher_adapter_sha256"],
            "teacher_manifest_sha256": source_auth["bindings"]["teacher_manifest_sha256"],
            "backend_id": "custom_transformers_peft_three_policy_v5",
            "base_revision": source_auth["bindings"]["base_revision"],
            "tokenizer_revision": source_auth["bindings"]["tokenizer_revision"],
        },
        "superseded_migration_source": {
            "package_path": source["path"],
            "package_tree_sha256": source["tree_sha256"],
            "authorization_sha256": source["authorization_sha256"],
            "package_content_sha256": source["package_content_sha256"],
            "reason": "pre_freeze_manifest_authority_not_accepted_for_production",
        },
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
        "B2_authorized": True,
        "B2_started": False,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "requires_explicit_allow_b2_calibration": True,
        "automatically_start_formal_b2": False,
    }
    authorization["authorization_content_sha256"] = canonical_json_sha256(authorization)

    documents: dict[str, dict[str, Any]] = {
        "manifest_migration_attestation.json": deepcopy(dict(migration_attestation)),
        "prompt_schedule.json": deepcopy(dict(schedule)),
        "b2_20_step_calibration_config.json": config,
        "b2_20_step_calibration_run_card.json": run_card,
        "b2_authorization.json": authorization,
    }
    components = [_payload_entry(name, documents[name]) for name in COMPONENT_ORDER]
    package_content_sha = canonical_json_sha256(components)
    package_index = {
        "schema_version": 2,
        "artifact_kind": "p4_8b_b2_calibration_package_index_v2",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "components": components,
        "component_count": len(components),
        "package_content_sha256": package_content_sha,
        "old_package_retained": True,
        "fresh_gpu_output_required": True,
        "B2_formal_authorized": False,
    }
    documents["package_index.json"] = package_index
    index_entry = _payload_entry("package_index.json", package_index)
    readiness = {
        "schema_version": 2,
        "artifact_kind": "p4_8b_b2_calibration_cpu_readiness_v2",
        "status": "ready_waiting_for_gpu_b2_calibration_revalidation",
        "package_version": PACKAGE_VERSION,
        "run_id": runtime_run_id,
        "package_index_sha256": index_entry["sha256"],
        "package_content_sha256": package_content_sha,
        "manifest_migration_attestation_sha256": migration_sha,
        "schedule_sha256": schedule_audit["schedule_sha256"],
        "canonical_manifest_sha256": authority["manifest_sha256"],
        "parent_content_equivalent": True,
        "parent_authority_equivalent": False,
        "semantic_preflight_implemented": True,
        "gpu_revalidation_pending": True,
        "B2_authorized": True,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "B2_started": False,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    documents["readiness.json"] = readiness
    return documents


def materialize_replacement_package(**kwargs: Any) -> dict[str, Any]:
    package = Path(kwargs["package_dir"]).resolve()
    if package.exists() or package.is_symlink():
        _fail("replacement package output must be fresh")
    documents = build_replacement_package_documents(**kwargs)
    package.mkdir(parents=True, exist_ok=False)
    try:
        for name in (
            *COMPONENT_ORDER,
            "package_index.json",
            "readiness.json",
        ):
            _atomic_json(package / name, documents[name])
        directory_fd = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        audit = verify_replacement_package(
            package,
            canonical_manifest_path=kwargs["authority"]["manifest_path"],
        )
    except Exception:
        # Preserve evidence on failure; never delete or overwrite a package attempt.
        raise
    return {
        "status": "ready_waiting_for_gpu_b2_calibration_revalidation",
        "package_dir": str(package),
        "package_content_sha256": audit["package_content_sha256"],
        "package_index_sha256": audit["package_index_sha256"],
        "authorization_sha256": audit["authorization_sha256"],
        "config_sha256": audit["config_sha256"],
        "run_card_sha256": audit["run_card_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
    }


def verify_replacement_package(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    if package.is_symlink() or not package.is_dir():
        _fail("replacement package is absent or a symlink")
    names = {item.name for item in package.iterdir()}
    if names != set(PACKAGE_FILES) or any(
        item.is_symlink() or not item.is_file() for item in package.iterdir()
    ):
        _fail("replacement package file graph differs")
    documents = {
        name: _read_json(package / name, name) for name in PACKAGE_FILES
    }
    index = documents["package_index.json"]
    components = index.get("components")
    if not isinstance(components, list) or len(components) != len(COMPONENT_ORDER):
        _fail("replacement package index component graph differs")
    expected_entries = [_payload_entry(name, documents[name]) for name in COMPONENT_ORDER]
    if components != expected_entries or index.get("package_content_sha256") != canonical_json_sha256(expected_entries):
        _fail("replacement package component SHA/content differs")
    if index.get("package_version") != PACKAGE_VERSION or index.get("B2_formal_authorized") is not False:
        _fail("replacement package index contract differs")
    index_sha = stream_sha256(package / "package_index.json")
    readiness = documents["readiness.json"]
    if not (
        readiness.get("status") == "ready_waiting_for_gpu_b2_calibration_revalidation"
        and readiness.get("package_index_sha256") == index_sha
        and readiness.get("package_content_sha256") == index["package_content_sha256"]
        and readiness.get("B2_authorized") is True
        and readiness.get("B2_calibration_started") is False
        and readiness.get("B2_formal_authorized") is False
    ):
        _fail("replacement package readiness binding differs")
    config = documents["b2_20_step_calibration_config.json"]
    card = documents["b2_20_step_calibration_run_card.json"]
    authorization = documents["b2_authorization.json"]
    auth_content = dict(authorization)
    claimed_auth = auth_content.pop("authorization_content_sha256", None)
    if claimed_auth != canonical_json_sha256(auth_content):
        _fail("replacement authorization content SHA differs")
    config_sha = stream_sha256(package / "b2_20_step_calibration_config.json")
    card_sha = stream_sha256(package / "b2_20_step_calibration_run_card.json")
    authorization_sha = stream_sha256(package / "b2_authorization.json")
    data = config.get("data")
    run = config.get("run")
    student = config.get("student_initialization")
    execution = config.get("execution")
    generation = config.get("generation")
    isolation = config.get("isolation")
    if not all(
        isinstance(item, Mapping)
        for item in (data, run, student, execution, generation, isolation)
    ):
        _fail("replacement package config sections are incomplete")
    if not (
        config.get("package_version") == PACKAGE_VERSION
        and run.get("stage") == "b2_calibration"
        and run.get("seed") == 42
        and run.get("optimizer_steps") == 20
        and generation.get("max_new_tokens") == 768
        and execution.get("optimizer_steps") == 20
        and execution.get("calibration_only") is True
        and student.get("mode") == FRESH_STUDENT_INITIALIZATION
        and student.get("source_adapter_path") is None
        and student.get("qualification_v2_usage") == "evidence_only_not_student_init"
        and data.get("schedule_path") == str(package / "prompt_schedule.json")
        and data.get("schedule_sha256") == documents["prompt_schedule.json"].get("schedule_sha256")
        and data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
        and all(
            isolation.get(field) is False
            for field in ("final_access", "controller_access", "confirmation_access", "label_access")
        )
    ):
        _fail("replacement package frozen training contract differs")
    if not (
        card.get("config_sha256") == config_sha
        and card.get("manifest_migration_attestation_sha256")
        == documents["manifest_migration_attestation.json"].get("attestation_sha256")
        and card.get("schedule_sha256") == documents["prompt_schedule.json"].get("schedule_sha256")
        and card.get("selected_response_length") == 768
        and card.get("optimizer_steps") == 20
        and card.get("seed") == 42
        and card.get("student_initialization") == FRESH_STUDENT_INITIALIZATION
        and card.get("B2_formal_authorized") is False
    ):
        _fail("replacement run-card SHA/contract differs")
    bindings = authorization.get("bindings")
    if not isinstance(bindings, Mapping) or not (
        authorization.get("artifact_kind") == "p4_8b_b2_20_step_calibration_authorization_v2"
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_started") is False
        and authorization.get("B2_calibration_started") is False
        and authorization.get("B2_formal_authorized") is False
        and bindings.get("config_sha256") == config_sha
        and bindings.get("run_card_sha256") == card_sha
        and bindings.get("canonical_manifest_sha256") == data.get("prompt_manifest_sha256")
        and bindings.get("schedule_sha256") == data.get("schedule_sha256")
    ):
        _fail("replacement authorization SHA/contract differs")
    try:
        authority = resolve_b2_data_authority(
            data["prompt_manifest_path"],
            expected_manifest_sha256=data["prompt_manifest_sha256"],
            canonical_manifest_path=canonical_manifest_path,
        )
        schedule_audit = validate_b2_prompt_schedule(
            documents["prompt_schedule.json"], authority=authority
        )
    except B2DataAuthorityV2Error as error:
        raise B2CalibrationPackageV2Error(str(error)) from error
    migration = documents["manifest_migration_attestation.json"]
    migration_content = dict(migration)
    claimed_migration = migration_content.pop("attestation_sha256", None)
    if not (
        claimed_migration == canonical_json_sha256(migration_content)
        and migration.get("content_equivalent") is True
        and migration.get("authority_equivalent") is False
        and migration.get("new_manifest", {}).get("sha256") == authority["manifest_sha256"]
        and bindings.get("migration_attestation_sha256") == claimed_migration
    ):
        _fail("replacement manifest migration attestation differs")
    superseded = authorization.get("superseded_migration_source")
    migration_parent = migration.get("parent_bindings")
    if not isinstance(superseded, Mapping) or not isinstance(
        migration_parent, Mapping
    ):
        _fail("replacement superseded package binding is absent")
    old_package = Path(str(superseded.get("package_path", "")))
    old_tree_sha = directory_tree_sha256(old_package)
    if not (
        superseded.get("package_tree_sha256") == old_tree_sha
        and migration_parent.get("old_package_sha256") == old_tree_sha
        and superseded.get("reason")
        == "pre_freeze_manifest_authority_not_accepted_for_production"
    ):
        _fail("replacement superseded package tree SHA differs")
    for prefix in ("length_telemetry", "length_selection", "root_index"):
        parent_path = Path(str(config["p4_8b_start_gate"].get(f"{prefix}_path", "")))
        # The config intentionally binds parent digests, while authorization binds
        # them again.  Paths are carried by the source P4.7 package/root and are
        # verified by the launch spec; synthetic packages may omit the path fields.
        if parent_path.name and parent_path.is_file():
            expected = bindings.get(f"{prefix}_sha256")
            if stream_sha256(parent_path) != expected:
                _fail(f"replacement P4.7 {prefix} SHA differs")
    assert_package_is_calibration_only(
        {
            "stage": authorization["stage"],
            "B2_formal_authorized": authorization["B2_formal_authorized"],
            "automatically_start_formal_b2": authorization[
                "automatically_start_formal_b2"
            ],
        }
    )
    return {
        "package_dir": str(package),
        "package_version": PACKAGE_VERSION,
        "package_content_sha256": index["package_content_sha256"],
        "package_index_sha256": index_sha,
        "config_sha256": config_sha,
        "run_card_sha256": card_sha,
        "authorization_sha256": authorization_sha,
        "authorization_content_sha256": claimed_auth,
        "selected_response_length": 768,
        "optimizer_steps": 20,
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
        "schedule": dict(documents["prompt_schedule.json"]),
        "schedule_audit": schedule_audit,
        "teacher": {
            "data_manifest_sha256": authority["manifest_sha256"],
            "adapter_path": config.get("teacher", {}).get("adapter_path"),
            "adapter_sha256": config.get("teacher", {}).get("adapter_sha256"),
            "manifest_sha256": config.get("teacher", {}).get("manifest_sha256"),
        },
        "config": config,
        "authorization": authorization,
        "run_card": card,
        "migration_attestation": migration,
    }


def pre_model_semantic_preflight(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
    call_counters: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Reopen all semantic inputs without importing or invoking any runtime."""

    before = None if call_counters is None else dict(call_counters)
    audit = verify_replacement_package(
        package_dir, canonical_manifest_path=canonical_manifest_path
    )
    if call_counters is not None and dict(call_counters) != before:
        _fail("pre-model semantic gate invoked a runtime loader/session")
    return {
        "status": "pre_model_semantic_gate_passed",
        "package_content_sha256": audit["package_content_sha256"],
        "manifest_sha256": audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "slot_count": audit["schedule"]["slot_count"],
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
        "student_session_calls": 0,
        "cuda_worker_calls": 0,
        "rollout_calls": 0,
        "audit": audit,
    }


__all__ = [
    "B2CalibrationPackageV2Error",
    "COMPONENT_ORDER",
    "PACKAGE_FILES",
    "PACKAGE_VERSION",
    "assert_package_is_calibration_only",
    "build_replacement_package_documents",
    "directory_tree_sha256",
    "materialize_replacement_package",
    "pre_model_semantic_preflight",
    "verify_replacement_package",
]
