"""Immutable P4.8f objective-evidence overlay over protected P4.8e assets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_memory_execution_v1 import (
    _atomic_write,
    canonical_json_sha256,
)


PACKAGE_VERSION = "p4_8f_objective_evidence_v2"
RUN_ID = (
    "qwen3-4b-b2-medical-opd-calibration-"
    "p4-8g-r9-1024-objective-memory-revalidation-v9-seed42"
)
PARENT_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8e-1024-memory-v3-package"
)
EXPECTED_PARENT_PACKAGE_VERSION = "p4_8e_memory_v7"
EXPECTED_PARENT_CONTENT_SHA256 = (
    "5c0706c73577c4102add6e23e874d47ed5c89356be2b9c81fce61c1477f1bd84"
)
EXPECTED_PARENT_INDEX_SHA256 = (
    "3e974497d39357813a82ab0579fd2472ec635ff6f10b58238cb1269fa4330dd3"
)
EXPECTED_MANIFEST_SHA256 = (
    "9f1d096d06b635737e1b90be3b92d6de32fd64b03fbcd97813e42d0a2ee88a99"
)
EXPECTED_SCHEDULE_SHA256 = (
    "4567ebb38972c1d37936a77590b5b6d28a6b6c234297fd85c82dacdad1926d88"
)
COMPONENT_ORDER = (
    "parent_binding.json",
    "prompt_schedule.json",
    "equivalence_contract.json",
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
)


class B2ObjectiveRevalidationPackageV2Error(RuntimeError):
    """The P4.8f overlay or a frozen authority differs."""


def _fail(message: str) -> None:
    raise B2ObjectiveRevalidationPackageV2Error(message)


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
        raise B2ObjectiveRevalidationPackageV2Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _verify_protected_parent(
    canonical_manifest_path: str | Path,
) -> dict[str, Any]:
    parent = PARENT_PACKAGE.resolve()
    index_path = parent / "package_index.json"
    if not (
        parent.is_dir()
        and not parent.is_symlink()
        and _file_sha256(index_path) == EXPECTED_PARENT_INDEX_SHA256
    ):
        _fail("protected P4.8e package/index identity differs")
    index = _read_json(index_path, "protected P4.8e package index")
    components = index.get("components")
    if not isinstance(components, list):
        _fail("protected P4.8e component index is absent")
    recomputed = []
    for item in components:
        if not isinstance(item, Mapping):
            _fail("protected P4.8e component descriptor is invalid")
        path = parent / str(item.get("path", ""))
        recomputed.append(
            {
                "path": str(item.get("path", "")),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not (
        recomputed == components
        and index.get("package_version") == EXPECTED_PARENT_PACKAGE_VERSION
        and index.get("package_content_sha256")
        == EXPECTED_PARENT_CONTENT_SHA256
        and canonical_json_sha256(recomputed)
        == EXPECTED_PARENT_CONTENT_SHA256
        and index.get("B2_formal_authorized") is False
    ):
        _fail("protected P4.8e package content differs")
    manifest = Path(canonical_manifest_path).resolve()
    if not (
        manifest.is_file()
        and not manifest.is_symlink()
        and _file_sha256(manifest) == EXPECTED_MANIFEST_SHA256
    ):
        _fail("canonical prompt manifest identity differs")
    config = _read_json(
        parent / "b2_20_step_calibration_config.json",
        "protected P4.8e config",
    )
    schedule = _read_json(parent / "prompt_schedule.json", "protected schedule")
    authorization = _read_json(
        parent / "b2_authorization.json", "protected authorization"
    )
    if not (
        schedule.get("schedule_sha256") == EXPECTED_SCHEDULE_SHA256
        and config.get("data", {}).get("prompt_manifest_sha256")
        == EXPECTED_MANIFEST_SHA256
        and config.get("data", {}).get("schedule_sha256")
        == EXPECTED_SCHEDULE_SHA256
        and authorization.get("B2_formal_authorized") is False
        and authorization.get("final_authorized") is False
    ):
        _fail("protected P4.8e science/authorization differs")
    return {
        "index": index,
        "config": config,
        "schedule": schedule,
        "authorization": authorization,
    }


def _equivalence_contract(
    *,
    code_git_commit: str,
    current_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if not (
        isinstance(code_git_commit, str)
        and len(code_git_commit) == 40
        and all(character in "0123456789abcdef" for character in code_git_commit)
    ):
        _fail("P4.8f code commit is not immutable")
    bindings = deepcopy(dict(current_bindings))
    if not bindings:
        _fail("P4.8f current bindings are absent")
    for descriptor in bindings.values():
        if not (
            isinstance(descriptor, Mapping)
            and set(descriptor) == {"path", "sha256"}
            and isinstance(descriptor["path"], str)
            and isinstance(descriptor["sha256"], str)
            and len(descriptor["sha256"]) == 64
        ):
            _fail("P4.8f current binding is invalid")
    return {
        "schema_version": 2,
        "artifact_kind": "p4_8f_two_layer_equivalence_contract_v2",
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "code_git_commit": code_git_commit,
        "current_bindings": bindings,
        "canonical_reducer_version": "v2",
        "evidence_writer_version": "v2",
        "strict_legacy_scalar_tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "strict_legacy_gradient_tolerance": {"atol": 2e-6, "rtol": 2e-5},
        "cpu_fp32_algebraic_tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "gpu_bf16_operational_tolerance": {
            "objective_loss_atol": 1e-5,
            "objective_loss_rtol": 1e-4,
            "gradient_cosine_min": 0.9999,
            "gradient_relative_l2_max": 1e-3,
            "delta_cosine_min": 0.9999,
            "delta_relative_l2_max": 1e-3,
        },
        "canary_entry_requires": [
            "algebraic_semantic_equivalence_pass",
            "gpu_bf16_operational_equivalence_pass",
        ],
        "legacy_strict_result_remains_independent": True,
        "topology_candidate": (
            "microbatch1_accumulation4_exact_lm_head_chunk128_"
            "one_backbone_backward_per_prompt"
        ),
        "fallback_candidate": None,
        "backbone_backward_calls_per_prompt": 1,
        "retain_graph_calls_per_prompt": 0,
        "target_logit_chunk_size": 128,
        "requires_throwaway_differential": True,
        "requires_throwaway_canary": True,
        "requires_formal_fresh_v0_after_canary": True,
        "minimum_headroom_bytes": 1024**3,
        "checkpoint_versions": [5, 10, 15, 20],
        "requires_v10_midrun_reload": True,
        "requires_v20_fresh_reload": True,
        "B2_formal_authorized": False,
        "final_authorized": False,
    }


def build_objective_revalidation_documents(
    *,
    package_dir: str | Path,
    runtime_output_dir: str | Path,
    runtime_run_id: str,
    canonical_manifest_path: str | Path,
    code_git_commit: str,
    current_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    parent = _verify_protected_parent(canonical_manifest_path)
    if runtime_run_id != RUN_ID:
        _fail("P4.8f run ID differs")
    package = Path(package_dir).resolve()
    output = Path(runtime_output_dir).resolve()
    if output.exists() or output.is_symlink():
        _fail("P4.8f runtime output must be fresh")
    contract = _equivalence_contract(
        code_git_commit=code_git_commit,
        current_bindings=current_bindings,
    )
    config = deepcopy(parent["config"])
    config["package_version"] = PACKAGE_VERSION
    config["run"].update(
        {
            "run_id": RUN_ID,
            "output_dir": str(output),
            "purpose": (
                "P4.8g objective and memory GPU revalidation and exact "
                "20-step B2 calibration"
            ),
            "status": "authorized_not_started",
        }
    )
    config["data"]["schedule_path"] = str(package / "prompt_schedule.json")
    config["objective_revalidation"] = deepcopy(contract)
    config["p4_8f_start_gate"] = {
        "package_version": PACKAGE_VERSION,
        "parent_package_path": str(PARENT_PACKAGE.resolve()),
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "parent_package_index_sha256": EXPECTED_PARENT_INDEX_SHA256,
        "code_git_commit": code_git_commit,
        "branch": "codex/p4-8g-b2-objective-gpu-revalidation",
        "fresh_v0_required": True,
        "formal_b2_automatic_start": False,
    }
    parent_binding = {
        "schema_version": 2,
        "artifact_kind": "p4_8f_protected_p4_8e_parent_binding_v2",
        "path": str(PARENT_PACKAGE.resolve()),
        "package_version": EXPECTED_PARENT_PACKAGE_VERSION,
        "package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "package_index_sha256": EXPECTED_PARENT_INDEX_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }
    authorization = {
        "schema_version": 2,
        "artifact_kind": "p4_8f_b2_objective_revalidation_authorization_v2",
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "B2_calibration_authorized": True,
        "B2_formal_authorized": False,
        "gpu_math_differential_authorized": True,
        "max_shape_canary_authorized": True,
        "optimizer_steps_authorized": 20,
        "formal_b2_automatic_start": False,
        "final_authorized": False,
        "controller_authorized": False,
        "confirmation_authorized": False,
        "label_access_authorized": False,
    }
    run_card = {
        "schema_version": 2,
        "artifact_kind": "p4_8f_b2_objective_revalidation_run_card_v2",
        "run_id": RUN_ID,
        "package_version": PACKAGE_VERSION,
        "status": "prepared_not_started",
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "physical_microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "checkpoint_versions": [5, 10, 15, 20],
        "combined_state_machine": [
            "fixed_token_gpu_differential",
            "max_shape_canary",
            "fresh_v0_steps_1_to_6",
            "same_run_continue_to_step_20",
            "finalizer_and_cleanup",
        ],
        "B2_formal_authorized": False,
    }
    return {
        "parent_binding.json": parent_binding,
        "prompt_schedule.json": deepcopy(parent["schedule"]),
        "equivalence_contract.json": contract,
        "b2_20_step_calibration_config.json": config,
        "b2_20_step_calibration_run_card.json": run_card,
        "b2_authorization.json": authorization,
    }


def materialize_objective_revalidation_package(**kwargs: Any) -> dict[str, Any]:
    package = Path(kwargs["package_dir"]).resolve()
    if package.exists() or package.is_symlink():
        _fail("P4.8f package path must be absent")
    documents = build_objective_revalidation_documents(**kwargs)
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
        "schema_version": 2,
        "artifact_kind": "p4_8f_objective_evidence_package_index_v2",
        "package_version": PACKAGE_VERSION,
        "run_id": RUN_ID,
        "component_count": len(components),
        "components": components,
        "package_content_sha256": canonical_json_sha256(components),
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "parent_package_index_sha256": EXPECTED_PARENT_INDEX_SHA256,
        "fresh_gpu_output_required": True,
        "fresh_v0_required": True,
        "B2_formal_authorized": False,
    }
    _atomic_write(package / "package_index.json", _canonical_bytes(index))
    return verify_objective_revalidation_package(
        package,
        canonical_manifest_path=kwargs["canonical_manifest_path"],
        repo_root=Path(__file__).resolve().parents[2],
    )


def verify_objective_revalidation_package(
    package_dir: str | Path,
    *,
    canonical_manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    index = _read_json(package / "package_index.json", "P4.8f package index")
    if index.get("package_version") != PACKAGE_VERSION:
        _fail("stale/non-P4.8f package is rejected")
    documents = {
        name: _read_json(package / name, f"P4.8f {name}")
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
        index.get("run_id") == RUN_ID
        and index.get("components") == components
        and index.get("component_count") == len(COMPONENT_ORDER)
        and index.get("package_content_sha256")
        == canonical_json_sha256(components)
        and index.get("parent_package_content_sha256")
        == EXPECTED_PARENT_CONTENT_SHA256
        and index.get("parent_package_index_sha256")
        == EXPECTED_PARENT_INDEX_SHA256
        and index.get("fresh_gpu_output_required") is True
        and index.get("fresh_v0_required") is True
        and index.get("B2_formal_authorized") is False
    ):
        _fail("P4.8f package index differs")
    parent = _verify_protected_parent(canonical_manifest_path)
    parent_binding = documents["parent_binding.json"]
    if not (
        parent_binding.get("path") == str(PARENT_PACKAGE.resolve())
        and parent_binding.get("package_version")
        == EXPECTED_PARENT_PACKAGE_VERSION
        and parent_binding.get("package_content_sha256")
        == EXPECTED_PARENT_CONTENT_SHA256
        and parent_binding.get("package_index_sha256")
        == EXPECTED_PARENT_INDEX_SHA256
    ):
        _fail("P4.8f protected parent binding differs")
    config = documents["b2_20_step_calibration_config.json"]
    contract = documents["equivalence_contract.json"]
    schedule = documents["prompt_schedule.json"]
    authorization = documents["b2_authorization.json"]
    run_card = documents["b2_20_step_calibration_run_card.json"]
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
            _fail(f"P4.8f frozen {field} differs from protected P4.8e")
    if not (
        schedule == parent["schedule"]
        and config.get("package_version") == PACKAGE_VERSION
        and config.get("run", {}).get("run_id") == RUN_ID
        and config.get("run", {}).get("optimizer_steps") == 20
        and config.get("run", {}).get("seed") == 42
        and config.get("generation", {}).get("max_new_tokens") == 1024
        and config.get("execution", {}).get("physical_microbatch_size") == 1
        and config.get("execution", {}).get("gradient_accumulation_steps") == 4
        and config.get("execution", {}).get("effective_batch_size") == 4
        and config.get("execution", {}).get("target_logit_chunk_size") == 128
        and config.get("data", {}).get("schedule_path")
        == str(package / "prompt_schedule.json")
        and config.get("objective_revalidation") == contract
        and contract.get("canary_entry_requires")
        == [
            "algebraic_semantic_equivalence_pass",
            "gpu_bf16_operational_equivalence_pass",
        ]
        and contract.get("fallback_candidate") is None
        and authorization.get("B2_calibration_authorized") is True
        and authorization.get("B2_formal_authorized") is False
        and authorization.get("final_authorized") is False
        and run_card.get("status") == "prepared_not_started"
        and run_card.get("B2_formal_authorized") is False
    ):
        _fail("P4.8f runtime/contract/authorization differs")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    bindings = contract.get("current_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        _fail("P4.8f current bindings are absent")
    for name, descriptor in bindings.items():
        path = root / str(descriptor.get("path", ""))
        if not (
            path.is_file()
            and not path.is_symlink()
            and _file_sha256(path) == descriptor.get("sha256")
        ):
            _fail(f"P4.8f current binding differs: {name}")
    p4d_gate = config["p4_8d_start_gate"]
    qualification = config["qualification"]
    return {
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
        "memory_execution_contract_sha256": p4d_gate[
            "memory_execution_contract_sha256"
        ],
        "oom_memory_attestation_sha256": p4d_gate[
            "oom_memory_attestation_sha256"
        ],
        "parent_package_content_sha256": EXPECTED_PARENT_CONTENT_SHA256,
        "parent_package_index_sha256": EXPECTED_PARENT_INDEX_SHA256,
        "runtime_run_id": RUN_ID,
        "runtime_output_dir": config["run"]["output_dir"],
        "code_git_commit": contract["code_git_commit"],
        "current_bindings": deepcopy(dict(bindings)),
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "seed": 42,
        "student_initialization": config["student_initialization"]["mode"],
        "qualification_v2_usage": config["student_initialization"][
            "qualification_v2_usage"
        ],
        "qualification_v2_path": qualification["v2_checkpoint_path"],
        "qualification_v2_tensor_sha256": qualification["v2_tensor_sha256"],
        "data_authority": {
            "manifest_path": config["data"]["prompt_manifest_path"],
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "schedule": schedule,
        "config": config,
        "run_card": run_card,
        "authorization": authorization,
        "equivalence_contract": contract,
    }


__all__ = [
    "B2ObjectiveRevalidationPackageV2Error",
    "COMPONENT_ORDER",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_PARENT_CONTENT_SHA256",
    "EXPECTED_PARENT_INDEX_SHA256",
    "EXPECTED_SCHEDULE_SHA256",
    "PACKAGE_VERSION",
    "PARENT_PACKAGE",
    "RUN_ID",
    "build_objective_revalidation_documents",
    "materialize_objective_revalidation_package",
    "verify_objective_revalidation_package",
]
