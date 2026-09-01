"""Immutable P4.8e overlay over the verified P4.8d memory package."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_calibration_package_v4 import (
    verify_memory_execution_package,
)
from src.opd.production_b2_memory_execution_v1 import (
    _atomic_write,
    canonical_json_sha256,
)


PACKAGE_VERSION = "p4_8e_memory_v7"
RUN_ID = "qwen3-4b-b2-medical-opd-calibration-p4-8e-r3-1024-memory-seed42"
PARENT_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8d-1024-memory-v2-package"
)
EXPECTED_PARENT_CONTENT_SHA256 = (
    "e06cf509b90d00279f993d63d63dc9f579d934cea41ec2b4ab11fcbeafc2cd03"
)
EXPECTED_PARENT_INDEX_SHA256 = (
    "4ff6f5750274318f3eed9b261a359a412f1adc3c41e301ca7b6e63382c8728ab"
)
COMPONENT_ORDER = (
    "parent_binding.json",
    "prompt_schedule.json",
    "memory_revalidation_contract.json",
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
)


class B2MemoryRevalidationPackageV1Error(RuntimeError):
    """The P4.8e overlay or its immutable parent differs."""


def _fail(message: str) -> None:
    raise B2MemoryRevalidationPackageV1Error(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2MemoryRevalidationPackageV1Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _revalidation_contract(
    *,
    code_git_commit: str,
    current_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if not (
        isinstance(code_git_commit, str)
        and len(code_git_commit) == 40
        and all(character in "0123456789abcdef" for character in code_git_commit)
    ):
        _fail("P4.8e code commit is not immutable")
    bindings = deepcopy(dict(current_bindings))
    if not bindings:
        _fail("P4.8e current bindings are absent")
    for name, descriptor in bindings.items():
        if not (
            isinstance(name, str)
            and isinstance(descriptor, Mapping)
            and set(descriptor) == {"path", "sha256"}
            and isinstance(descriptor["path"], str)
            and isinstance(descriptor["sha256"], str)
            and len(descriptor["sha256"]) == 64
        ):
            _fail("P4.8e current binding is invalid")
    return {
        "schema_version": 1,
        "artifact_kind": "p4_8e_b2_memory_revalidation_contract_v1",
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "code_git_commit": code_git_commit,
        "current_bindings": bindings,
        "requires_real_gpu_math_differential": True,
        "fixed_differential_schedule_step": 1,
        "fixed_rollout_tokens_reused": True,
        "runtime_loading": "serial_student_runtime_on_gpu0",
        "scalar_atol": 1e-6,
        "scalar_rtol": 1e-6,
        "gradient_atol": 2e-6,
        "gradient_rtol": 2e-5,
        "requires_max_shape_valid_completion_tokens": 1024,
        "requires_real_rollout_in_canary": True,
        "requires_throwaway_canary": True,
        "requires_formal_fresh_v0_after_canary": True,
        "minimum_headroom_bytes": 1024**3,
        "backbone_backward_calls_per_prompt": 1,
        "retain_graph_calls_per_prompt": 0,
        "scheduler": "constant_factor_1_no_lr_change",
        "checkpoint_versions": [5, 10, 15, 20],
        "requires_v10_midrun_reload": True,
        "requires_v20_fresh_reload": True,
        "B2_formal_authorized": False,
        "final_authorized": False,
    }


def build_revalidation_overlay_documents(
    *,
    parent_package_dir: str | Path,
    package_dir: str | Path,
    runtime_output_dir: str | Path,
    runtime_run_id: str,
    canonical_manifest_path: str | Path,
    code_git_commit: str,
    current_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    parent_path = Path(parent_package_dir).resolve()
    parent = verify_memory_execution_package(
        parent_path, canonical_manifest_path=canonical_manifest_path
    )
    if not (
        parent["package_content_sha256"] == EXPECTED_PARENT_CONTENT_SHA256
        and parent["package_index_sha256"] == EXPECTED_PARENT_INDEX_SHA256
    ):
        _fail("P4.8d parent package identity differs")
    if runtime_run_id != RUN_ID:
        _fail("P4.8e run ID differs")
    output = Path(runtime_output_dir).resolve()
    package = Path(package_dir).resolve()
    if output.exists() or output.is_symlink():
        _fail("P4.8e runtime output must be fresh")

    contract = _revalidation_contract(
        code_git_commit=code_git_commit,
        current_bindings=current_bindings,
    )
    config = deepcopy(parent["config"])
    config["package_version"] = PACKAGE_VERSION
    config["run"].update(
        {
            "run_id": RUN_ID,
            "output_dir": str(output),
            "purpose": "P4.8e memory-balanced GPU revalidation and 20-step calibration",
            "status": "authorized_not_started",
        }
    )
    config["data"]["schedule_path"] = str(package / "prompt_schedule.json")
    config["execution"]["scheduler"] = "constant_factor_1_no_lr_change"
    config["memory_revalidation"] = deepcopy(contract)
    config["p4_8e_start_gate"] = {
        "package_version": PACKAGE_VERSION,
        "parent_package_path": str(parent_path),
        "parent_package_content_sha256": parent["package_content_sha256"],
        "parent_package_index_sha256": parent["package_index_sha256"],
        "code_git_commit": code_git_commit,
        "branch": "codex/p4-8e-b2-memory-revalidation",
        "fresh_v0_required": True,
        "formal_b2_automatic_start": False,
    }
    parent_binding = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_parent_package_binding_v1",
        "path": str(parent_path),
        "package_version": parent["package_version"],
        "package_content_sha256": parent["package_content_sha256"],
        "package_index_sha256": parent["package_index_sha256"],
        "authorization_sha256": parent["authorization_sha256"],
        "config_sha256": parent["config_sha256"],
        "run_card_sha256": parent["run_card_sha256"],
        "manifest_sha256": parent["data_authority"]["manifest_sha256"],
        "schedule_sha256": parent["schedule"]["schedule_sha256"],
    }
    authorization = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_b2_memory_revalidation_authorization_v1",
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "parent_package_content_sha256": parent["package_content_sha256"],
        "B2_calibration_authorized": True,
        "B2_formal_authorized": False,
        "gpu_math_differential_authorized": True,
        "max_shape_canary_authorized": True,
        "optimizer_steps_authorized": 20,
        "formal_b2_automatic_start": False,
        "final_authorized": False,
        "controller_authorized": False,
        "label_access_authorized": False,
    }
    run_card = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_b2_memory_revalidation_run_card_v1",
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "status": "prepared_not_started",
        "parent_package_content_sha256": parent["package_content_sha256"],
        "manifest_sha256": parent["data_authority"]["manifest_sha256"],
        "schedule_sha256": parent["schedule"]["schedule_sha256"],
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "physical_microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "checkpoint_versions": [5, 10, 15, 20],
        "requires_gpu_math_differential": True,
        "requires_max_shape_canary": True,
        "B2_formal_authorized": False,
    }
    return {
        "parent_binding.json": parent_binding,
        "prompt_schedule.json": deepcopy(parent["schedule"]),
        "memory_revalidation_contract.json": contract,
        "b2_20_step_calibration_config.json": config,
        "b2_20_step_calibration_run_card.json": run_card,
        "b2_authorization.json": authorization,
    }


def materialize_revalidation_overlay_package(**kwargs: Any) -> dict[str, Any]:
    package = Path(kwargs["package_dir"]).resolve()
    if package.exists() or package.is_symlink():
        _fail("P4.8e overlay package path must be absent")
    documents = build_revalidation_overlay_documents(**kwargs)
    package.mkdir(parents=True, exist_ok=False)
    for name in COMPONENT_ORDER:
        _atomic_write(package / name, _canonical_bytes(documents[name]))
    components = [
        {
            "path": name,
            "sha256": _file_sha256(package / name),
            "size_bytes": (package / name).stat().st_size,
        }
        for name in COMPONENT_ORDER
    ]
    index = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_b2_memory_revalidation_package_index_v1",
        "package_version": PACKAGE_VERSION,
        "run_id": RUN_ID,
        "component_count": len(components),
        "components": components,
        "package_content_sha256": canonical_json_sha256(components),
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "fresh_gpu_output_required": True,
        "fresh_v0_required": True,
        "B2_formal_authorized": False,
    }
    _atomic_write(package / "package_index.json", _canonical_bytes(index))
    return verify_revalidation_overlay_package(
        package,
        canonical_manifest_path=kwargs["canonical_manifest_path"],
        repo_root=Path(__file__).resolve().parents[2],
    )


def verify_revalidation_overlay_package(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    index = _read_json(package / "package_index.json", "P4.8e package index")
    documents = {
        name: _read_json(package / name, f"P4.8e {name}")
        for name in COMPONENT_ORDER
    }
    components = [
        {
            "path": name,
            "sha256": _file_sha256(package / name),
            "size_bytes": (package / name).stat().st_size,
        }
        for name in COMPONENT_ORDER
    ]
    if not (
        index.get("package_version") == PACKAGE_VERSION
        and index.get("run_id") == RUN_ID
        and index.get("component_count") == len(COMPONENT_ORDER)
        and index.get("components") == components
        and index.get("package_content_sha256")
        == canonical_json_sha256(components)
        and index.get("parent_package_content_sha256")
        == EXPECTED_PARENT_CONTENT_SHA256
        and index.get("fresh_gpu_output_required") is True
        and index.get("fresh_v0_required") is True
        and index.get("B2_formal_authorized") is False
    ):
        _fail("P4.8e package index differs")

    parent_binding = documents["parent_binding.json"]
    parent = verify_memory_execution_package(
        parent_binding.get("path", ""),
        canonical_manifest_path=canonical_manifest_path,
    )
    if not (
        parent["package_content_sha256"] == EXPECTED_PARENT_CONTENT_SHA256
        and parent["package_index_sha256"] == EXPECTED_PARENT_INDEX_SHA256
        and parent_binding.get("package_content_sha256")
        == parent["package_content_sha256"]
        and parent_binding.get("package_index_sha256")
        == parent["package_index_sha256"]
        and parent_binding.get("authorization_sha256")
        == parent["authorization_sha256"]
        and parent_binding.get("config_sha256") == parent["config_sha256"]
        and parent_binding.get("run_card_sha256") == parent["run_card_sha256"]
        and parent_binding.get("manifest_sha256")
        == parent["data_authority"]["manifest_sha256"]
        and parent_binding.get("schedule_sha256")
        == parent["schedule"]["schedule_sha256"]
    ):
        _fail("P4.8e immutable parent binding differs")

    config = documents["b2_20_step_calibration_config.json"]
    contract = documents["memory_revalidation_contract.json"]
    schedule = documents["prompt_schedule.json"]
    if schedule != parent["schedule"]:
        _fail("P4.8e schedule copy differs from the parent")
    for field in (
        "model",
        "teacher",
        "protocol",
        "generation",
        "qualification",
        "student_initialization",
        "memory_execution",
        "isolation",
    ):
        if config.get(field) != parent["config"].get(field):
            _fail(f"P4.8e frozen {field} differs from the parent")
    parent_execution = parent["config"]["execution"]
    execution = dict(config.get("execution", {}))
    scheduler = execution.pop("scheduler", None)
    if not (
        execution == parent_execution
        and scheduler == "constant_factor_1_no_lr_change"
        and config.get("package_version") == PACKAGE_VERSION
        and config.get("run", {}).get("run_id") == RUN_ID
        and config.get("run", {}).get("optimizer_steps") == 20
        and config.get("run", {}).get("seed") == 42
        and Path(config.get("run", {}).get("output_dir", "")).is_absolute()
        and config.get("data", {}).get("prompt_manifest_sha256")
        == parent["config"]["data"]["prompt_manifest_sha256"]
        and config.get("data", {}).get("schedule_sha256")
        == parent["config"]["data"]["schedule_sha256"]
        and config.get("data", {}).get("schedule_path")
        == str(package / "prompt_schedule.json")
        and config.get("memory_revalidation") == contract
    ):
        _fail("P4.8e runtime overlay differs")
    parent_data = dict(parent["config"]["data"])
    overlay_data = dict(config["data"])
    parent_data.pop("schedule_path")
    overlay_data.pop("schedule_path")
    if overlay_data != parent_data:
        _fail("P4.8e frozen data contract differs")

    if not (
        contract.get("requires_real_gpu_math_differential") is True
        and contract.get("requires_max_shape_valid_completion_tokens") == 1024
        and contract.get("requires_throwaway_canary") is True
        and contract.get("requires_formal_fresh_v0_after_canary") is True
        and contract.get("minimum_headroom_bytes") == 1024**3
        and contract.get("backbone_backward_calls_per_prompt") == 1
        and contract.get("retain_graph_calls_per_prompt") == 0
        and contract.get("scheduler") == "constant_factor_1_no_lr_change"
        and contract.get("checkpoint_versions") == [5, 10, 15, 20]
        and contract.get("B2_formal_authorized") is False
        and contract.get("final_authorized") is False
    ):
        _fail("P4.8e revalidation contract differs")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    bindings = contract.get("current_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        _fail("P4.8e current bindings are absent")
    for name, descriptor in bindings.items():
        if not isinstance(descriptor, Mapping):
            _fail("P4.8e current binding descriptor is invalid")
        path = root / str(descriptor.get("path", ""))
        if not (
            path.is_file()
            and not path.is_symlink()
            and _file_sha256(path) == descriptor.get("sha256")
        ):
            _fail(f"P4.8e current binding differs: {name}")

    authorization = documents["b2_authorization.json"]
    run_card = documents["b2_20_step_calibration_run_card.json"]
    if not (
        authorization.get("B2_calibration_authorized") is True
        and authorization.get("B2_formal_authorized") is False
        and authorization.get("final_authorized") is False
        and authorization.get("optimizer_steps_authorized") == 20
        and run_card.get("run_id") == RUN_ID
        and run_card.get("status") == "prepared_not_started"
        and run_card.get("requires_gpu_math_differential") is True
        and run_card.get("requires_max_shape_canary") is True
        and run_card.get("B2_formal_authorized") is False
    ):
        _fail("P4.8e authorization or run card differs")

    return {
        **parent,
        "package_dir": str(package),
        "package_version": PACKAGE_VERSION,
        "package_content_sha256": index["package_content_sha256"],
        "package_index_sha256": _file_sha256(package / "package_index.json"),
        "authorization_sha256": _file_sha256(package / "b2_authorization.json"),
        "config_sha256": _file_sha256(
            package / "b2_20_step_calibration_config.json"
        ),
        "run_card_sha256": _file_sha256(
            package / "b2_20_step_calibration_run_card.json"
        ),
        "parent_package_dir": str(Path(parent_binding["path"]).resolve()),
        "parent_package_content_sha256": parent["package_content_sha256"],
        "runtime_run_id": RUN_ID,
        "runtime_output_dir": config["run"]["output_dir"],
        "code_git_commit": contract["code_git_commit"],
        "current_bindings": deepcopy(dict(bindings)),
        "config": config,
        "run_card": run_card,
        "authorization": authorization,
        "memory_revalidation_contract": contract,
        "schedule": schedule,
    }


__all__ = [
    "B2MemoryRevalidationPackageV1Error",
    "COMPONENT_ORDER",
    "PACKAGE_VERSION",
    "PARENT_PACKAGE",
    "RUN_ID",
    "build_revalidation_overlay_documents",
    "materialize_revalidation_overlay_package",
    "verify_revalidation_overlay_package",
]
