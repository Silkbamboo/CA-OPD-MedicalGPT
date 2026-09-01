"""CPU-only formal preflight for the P4.7 length continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

import yaml

from src.opd.production_length_contract_v7 import (
    CONDITIONAL_4096_CANDIDATES,
    PRIMARY_CANDIDATES,
    validate_candidate_ladder,
)
from src.opd.production_length_parent_v7 import (
    build_parent_reuse_attestation,
    canonical_json_sha256,
    sha256_file_stream,
    verify_parent_reuse,
)
from src.opd.production_qualification_preflight_v6 import (
    ProductionQualificationPreflightError,
    validate_base_model_transport,
)


RUN_ID = "qwen3-4b-length-qualification-v7-seed42"
SCHEMA_ID = "ca-opd/p4.7-production-length-qualification/v1"
SOURCE_BRANCH = "codex/p4-7-length-evidence-continuation"
PARENT_HANDOFF_GIT_COMMIT = "059d4aae5fef625c0421f6c6a6bb7979ed32754b"
RUN_CARD_PATH = Path(
    "configs/run_cards/qwen3-4b-length-qualification-v7-seed42.json"
)

CURRENT_BINDING_PATHS = {
    "length_contract_sha256": "src/opd/production_length_contract_v7.py",
    "length_artifacts_sha256": "src/opd/production_length_artifacts_v7.py",
    "parent_verifier_sha256": "src/opd/production_length_parent_v7.py",
    "gpu_runtime_sha256": "src/opd/production_length_gpu_runtime_v7.py",
    "gpu_backend_sha256": "src/opd/production_length_gpu_backend_v7.py",
    "preflight_sha256": "src/opd/production_length_preflight_v7.py",
    "entrypoint_sha256": "src/opd/production_length_v7.py",
    "config_sha256": "configs/opd/qwen3_4b_length_qualification_v7.yaml",
    "artifact_schema_sha256": (
        "configs/opd/p4_7_length_qualification_artifact_schema_v7.json"
    ),
    "b2_schema_sha256": "configs/opd/p4_7_b2_calibration_package_schema_v1.json",
    "decision_sha256": (
        "docs/decisions/0023-production-opd-response-length-escalation-and-evidence-first-qualification.md"
    ),
    "launcher_sha256": "scripts/run_qwen3_4b_length_qualification_v7.sh",
    "prompt_renderer_sha256": "src/opd/calibration_data.py",
    "prompt_loader_sha256": "src/opd/production_qualification_prompts_v6.py",
    "adapter_identity_helper_sha256": "src/opd/production_sampler_refresh_v5.py",
    "base_transport_validator_sha256": "src/opd/production_qualification_preflight_v6.py",
    "b2_executor_sha256": "src/opd/production_qualification_aux_gpu_v6.py",
    "b2_gpu_entrypoint_sha256": (
        "src/opd/production_qualification_gpu_runtime_v6.py"
    ),
    "b2_session_factory_sha256": (
        "src/opd/production_qualification_two_step_gpu_v6.py"
    ),
    "b2_template_sha256": "configs/runs/b2_medical_opd_qwen3_4b_custom_v5_p4_6.yaml",
}

RUN_CARD_FIELDS = {
    "schema_id",
    "schema_version",
    "run_id",
    "stage",
    "status",
    "source_branch",
    "config_path",
    "config_sha256",
    "artifact_schema_path",
    "artifact_schema_sha256",
    "b2_package_schema_path",
    "b2_package_schema_sha256",
    "parent_reuse_attestation_path",
    "parent_reuse_attestation_sha256",
    "protocol_path",
    "protocol_sha256",
    "launcher_path",
    "launcher_sha256",
    "production_backend_id",
    "gpu_authorization_environment_variable",
    "estimated_primary_gpu_minutes",
    "estimated_total_if_4096_gpu_minutes",
    "estimated_explicit_fallback_primary_gpu_minutes",
    "estimated_explicit_fallback_total_if_4096_gpu_minutes",
    "estimated_explicit_fallback_primary_cost_cny",
    "estimated_explicit_fallback_total_if_4096_cost_cny",
    "total_runtime_hard_kill_allowed",
    "automatically_start_b2",
    "allow_b2_calibration_required_for_later_start",
    "gpu_execution_now",
    "production_sampler_refresh_ready_now",
    "OPD_scoring_backend_ready_now",
    "B2_authorized_now",
    "B2_started",
    "next_state",
}


class ProductionLengthPreflightV7Error(RuntimeError):
    """A checked identity, isolation or host condition failed closed."""


def _fail(message: str) -> None:
    raise ProductionLengthPreflightV7Error(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _sha(path: str | Path) -> str:
    return sha256_file_stream(Path(path))


def compute_current_bindings(root: str | Path) -> dict[str, str]:
    base = Path(root).resolve()
    result: dict[str, str] = {}
    for key, relative in CURRENT_BINDING_PATHS.items():
        path = base / relative
        if path.is_symlink() or not path.is_file():
            _fail(f"current P4.7 binding is absent: {relative}")
        result[key] = _sha(path)
    # The P4.6 evidence commit differs from the immutable formal-run code SHA;
    # both are required so the attestation names the reviewed parent handoff
    # as well as the commit recorded by the actual GPU run.
    result["p4_6_handoff_git_commit"] = PARENT_HANDOFF_GIT_COMMIT
    return result


def validate_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        _fail("P4.7 config is not a mapping")
    value = dict(config)
    required_top = {
        "schema_id",
        "schema_version",
        "run",
        "parent_reuse",
        "production_binding",
        "protocol",
        "artifacts",
        "model",
        "teacher_binding_for_future_b2_package",
        "prompt_selection",
        "generation",
        "prefix_equivalence",
        "length_qualification",
        "execution",
        "b2_package",
        "isolation",
        "resources",
        "authorization",
        "versions",
        "status",
    }
    if set(value) != required_top:
        _fail("P4.7 config top-level fields are not exact")
    run = value["run"]
    parent = value["parent_reuse"]
    production = value["production_binding"]
    model = value["model"]
    prompt = value["prompt_selection"]
    generation = value["generation"]
    prefix = value["prefix_equivalence"]
    length = value["length_qualification"]
    execution = value["execution"]
    isolation = value["isolation"]
    resources = value["resources"]
    artifacts = value["artifacts"]
    b2_package = value["b2_package"]
    status = value["status"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            run,
            parent,
            production,
            model,
            prompt,
            generation,
            prefix,
            length,
            execution,
            isolation,
            resources,
            artifacts,
            b2_package,
            status,
        )
    ):
        _fail("P4.7 config section is absent")
    if not (
        value["schema_id"] == SCHEMA_ID
        and value["schema_version"] == 1
        and run.get("run_id") == RUN_ID
        and run.get("seed") == 42
        and run.get("formal_opd_training") is False
        and run.get("automatically_start_b2") is False
        and run.get("total_runtime_hard_kill_allowed") is False
        and parent.get("run_id")
        == "qwen3-4b-production-qualification-v6-seed42"
        and parent.get("parent_status") == "failed_length_not_frozen"
        and parent.get("formal_git_commit")
        == "96a95fbdfd992299807110570d10cffa17e294b9"
        and parent.get("handoff_git_commit") == PARENT_HANDOFF_GIT_COMMIT
        and parent.get("reuse_scope", {}).get("p4_6_length_result") is False
        and parent.get("policy_identity", {}).get("v1_differs_from_v2") is True
        and parent.get("v2", {}).get("logical_version") == "v2"
        and parent.get("v2", {}).get("runtime_slot") == "student_active"
        and production.get("backend_id")
        == "custom_transformers_peft_three_policy_v5"
        and production.get("vllm_used") is False
        and prompt.get("prompt_count") == 16
        and prompt.get("prompts_per_source") == 8
        and prompt.get("prompt_only") is True
        and all(
            prompt.get(field) is False
            for field in (
                "contains_labels",
                "contains_final",
                "contains_controller",
                "contains_confirmation",
            )
        )
        and generation.get("enable_thinking") is False
        and generation.get("full_support") is True
        and generation.get("do_sample") is True
        and generation.get("temperature") == 1.0
        and generation.get("top_p") == 1.0
        and generation.get("top_k") == 0
        and generation.get("output_scores") is False
        and generation.get("output_logits") is False
        and generation.get("eos_token_id") == [151645, 151643]
        and generation.get("pad_token_id") == 151643
        and model.get("prompt_template_sha256")
        == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        and prefix.get("independent_short_cap") == 1024
        and prefix.get("independent_long_cap") == 2048
        and prefix.get("minimum_samples_per_source") >= 1
        and prefix.get("same_per_sample_seed") is True
        and prefix.get("on_mismatch")
        == "explicit_independent_generation_for_each_candidate"
        and prefix.get("silent_derived_fallback_allowed") is False
    ):
        _fail("P4.7 frozen identity/generation config drift")
    validate_candidate_ladder(
        length.get("primary", {}).get("actual_generation_cap"),
        length.get("primary", {}).get("candidate_ladder", []),
    )
    validate_candidate_ladder(
        length.get("conditional_4096", {}).get("actual_generation_cap"),
        length.get("conditional_4096", {}).get("candidate_ladder", []),
    )
    phase_order = execution.get("ordered_phases")
    post_exit_phases = (
        "runtime_release",
        "gpu_worker_process_exit",
        "cpu_only_post_exit_finalizer",
        "resource_cleanup",
        "success_only_b2_package_materialization_after_cleanup",
    )
    post_exit_order_ok = bool(
        isinstance(phase_order, list)
        and all(phase in phase_order for phase in post_exit_phases)
        and all(phase_order.count(phase) == 1 for phase in post_exit_phases)
        and [phase_order.index(phase) for phase in post_exit_phases]
        == sorted(phase_order.index(phase) for phase in post_exit_phases)
    )
    if not (
        tuple(length["primary"]["candidate_ladder"]) == PRIMARY_CANDIDATES
        and tuple(length["conditional_4096"]["candidate_ladder"])
        == CONDITIONAL_4096_CANDIDATES
        and length["conditional_4096"].get("maximum_attempts") == 1
        and length["conditional_4096"].get("requires_only_truncation_failure")
        is True
        and all(
            length["conditional_4096"].get(field) is True
            for field in (
                "requires_no_primary_candidate_passed",
                "requires_invalid_count_zero",
                "requires_empty_count_zero",
                "requires_non_finite_count_zero",
                "requires_unexpected_think_tag_count_zero",
                "requires_repetition_count_zero",
                "requires_eos_stop_check",
                "requires_prompt_plus_cap_within_context",
                "requires_disk_and_gpu_preflight",
                "requires_isolation_false",
            )
        )
        and length.get("selection_rule") == "shortest_passing_candidate"
        and length.get("overall_truncation_rate_max") == 0.20
        and length.get("per_source_truncation_rate_max") == 0.20
        and length.get("maximum_truncations_per_source") == 1
        and all(
            length.get(field) == 0
            for field in (
                "invalid_count_max",
                "empty_count_max",
                "non_finite_count_max",
                "unexpected_think_tag_count_max",
                "repetition_count_max",
            )
        )
        and all(
            length.get(field) is True
            for field in (
                "finite_required",
                "valid_finish_reason_required",
                "output_contract_required",
                "no_oom_required",
            )
        )
        and length.get("automatic_further_escalation") is False
        and post_exit_order_ok
        and all(
            execution.get(field) is False
            for field in (
                "automatically_run_b2",
                "automatically_run_idt",
                "automatically_run_sar",
                "automatically_run_ca_opd",
                "automatically_run_controller",
                "automatically_run_confirmation",
                "automatically_run_final",
            )
        )
        and dict(isolation)
        == {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }
        and resources.get("required_gpus") == 2
        and resources.get("expected_gpu_model") == "NVIDIA GeForce RTX 3090"
        and resources.get("minimum_projected_free_at_peak_gib") == 10
        and resources.get("conditional_4096_min_free_gpu_bytes")
        == 2 * 1024**3
        and resources.get("estimated_primary_gpu_minutes") == [20, 60]
        and resources.get("estimated_total_if_4096_gpu_minutes") == [35, 90]
        and resources.get("estimated_explicit_fallback_primary_gpu_minutes")
        == [60, 210]
        and resources.get(
            "estimated_explicit_fallback_total_if_4096_gpu_minutes"
        )
        == [100, 270]
        and resources.get("estimated_explicit_fallback_primary_cost_cny")
        == [2.96, 10.36]
        and resources.get("estimated_explicit_fallback_total_if_4096_cost_cny")
        == [4.93, 13.32]
        and resources.get("estimate_is_stop_condition") is False
        and all(
            artifacts.get(field) is True
            for field in (
                "write_diagnostics_before_selection",
                "schema_validate_before_write",
                "same_directory_temporary_file",
                "file_flush_and_fsync_required",
                "atomic_replace_required",
                "parent_directory_fsync_required",
                "reopen_schema_count_size_sha_required",
                "evidence_index_before_selection_required",
                "readiness_from_disk_only",
            )
        )
        and b2_package.get("generate_only_after_formal_gpu_length_success") is True
        and b2_package.get("requires_explicit_start_argument")
        == "--allow-b2-calibration"
        and b2_package.get("automatically_start") is False
        and b2_package.get("B2_started") is False
        and status.get("parent_core_evidence_verified") is True
        and status.get("length_writer_fixed") is True
        and status.get("length_protocol_frozen") is True
        and status.get("gpu_length_qualification_pending") is True
        and all(
            status.get(field) is False
            for field in (
                "production_sampler_refresh_ready",
                "OPD_scoring_backend_ready",
                "B2_authorized",
                "B2_started",
            )
        )
    ):
        _fail("P4.7 bounded qualification/isolation config drift")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProductionLengthPreflightV7Error(
            f"cannot read P4.7 config: {type(error).__name__}"
        ) from error
    return validate_frozen_config(value)


def _load_card(root: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = root / RUN_CARD_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthPreflightV7Error(
            f"cannot read P4.7 run card: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict) or set(value) != RUN_CARD_FIELDS:
        _fail("P4.7 run-card fields are not exact")
    relative_config = config_path.relative_to(root).as_posix()
    if not (
        value["schema_id"] == SCHEMA_ID
        and value["schema_version"] == 1
        and value["run_id"] == RUN_ID
        and value["stage"] == "production_length_qualification_continuation"
        and value["status"] == "prepared_cpu_only_not_started"
        and value["source_branch"] == SOURCE_BRANCH
        and value["config_path"] == relative_config
        and value["config_sha256"] == _sha(config_path)
        and value["production_backend_id"]
        == config["production_binding"]["backend_id"]
        and value["gpu_execution_now"] == "not_run_cpu_only"
        and value["total_runtime_hard_kill_allowed"] is False
        and value["automatically_start_b2"] is False
        and value["B2_started"] is False
        and value["next_state"]
        == "ready_waiting_for_gpu_length_qualification"
        and value["estimated_primary_gpu_minutes"] == [20, 60]
        and value["estimated_total_if_4096_gpu_minutes"] == [35, 90]
        and value["estimated_explicit_fallback_primary_gpu_minutes"]
        == [60, 210]
        and value["estimated_explicit_fallback_total_if_4096_gpu_minutes"]
        == [100, 270]
        and value["estimated_explicit_fallback_primary_cost_cny"]
        == [2.96, 10.36]
        and value["estimated_explicit_fallback_total_if_4096_cost_cny"]
        == [4.93, 13.32]
    ):
        _fail("P4.7 run-card identity drift")
    for path_field, sha_field in (
        ("artifact_schema_path", "artifact_schema_sha256"),
        ("b2_package_schema_path", "b2_package_schema_sha256"),
        ("parent_reuse_attestation_path", "parent_reuse_attestation_sha256"),
        ("protocol_path", "protocol_sha256"),
        ("launcher_path", "launcher_sha256"),
    ):
        target = _resolve(root, value[path_field])
        if target.is_symlink() or not target.is_file() or _sha(target) != value[sha_field]:
            _fail(f"P4.7 run-card {path_field} SHA mismatch")
    return value


def _validate_json_schemas(root: Path, config: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    for path_key, sha_key in (
        ("schema_path", "schema_sha256"),
        ("b2_package_schema_path", "b2_package_schema_sha256"),
    ):
        path = _resolve(root, config["artifacts"][path_key])
        expected = config["artifacts"].get(sha_key)
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            _fail(f"P4.7 {path_key} SHA mismatch")
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            raise ProductionLengthPreflightV7Error(
                f"P4.7 schema invalid: {path.name}: {type(error).__name__}"
            ) from error


def _parent_expected(
    config: Mapping[str, Any], attestation: Mapping[str, Any]
) -> dict[str, Any]:
    parent = config["parent_reuse"]
    v2 = parent["v2"]
    audit = attestation.get("parent_audit")
    if not isinstance(audit, Mapping) or not isinstance(
        audit.get("protected_artifacts"), list
    ):
        _fail("parent attestation lacks protected artifact inventory")
    configured_protected = list(parent.get("protected_artifacts", []))
    attested_protected = list(audit["protected_artifacts"])
    sort_key = lambda item: (str(item.get("stage")), str(item.get("path")))
    if not configured_protected or sorted(
        configured_protected, key=sort_key
    ) != sorted(attested_protected, key=sort_key):
        _fail("config/attestation protected artifact inventory drift")
    return {
        "source_branch": parent["source_branch"],
        "formal_git_commit": parent["formal_git_commit"],
        "run_id": parent["run_id"],
        "evidence_index_sha256": parent["evidence_index_sha256"],
        "final_index_sha256": parent["final_index_sha256"],
        "failure_sha256": parent["failure_sha256"],
        "metrics_sha256": parent["metrics_sha256"],
        "readiness_sha256": parent["readiness_sha256"],
        "evidence_readiness_sha256": parent["evidence_readiness_sha256"],
        "micro_readiness_sha256": parent["micro_readiness_sha256"],
        "core_artifact_sha256": dict(parent["core_artifact_sha256"]),
        "v2": {
            "checkpoint_directory": v2["checkpoint_directory"],
            "authority_artifact_sha256": parent["core_artifact_sha256"][
                "authority_v2"
            ],
            "transport_manifest_sha256": v2["transport_manifest_sha256"],
            "transport_manifest_size_bytes": v2[
                "transport_manifest_size_bytes"
            ],
            "adapter_config_sha256": v2["adapter_config_sha256"],
            "adapter_config_size_bytes": v2["adapter_config_size_bytes"],
            "adapter_weights_sha256": v2["adapter_weights_sha256"],
            "adapter_weights_size_bytes": v2["adapter_weights_size_bytes"],
            "aggregate_tensor_sha256": v2["aggregate_tensor_sha256"],
        },
        "protected_artifacts": configured_protected,
    }


def _verify_parent_and_attestation(
    root: Path, config: Mapping[str, Any], card: Mapping[str, Any]
) -> dict[str, Any]:
    attestation_path = _resolve(root, card["parent_reuse_attestation_path"])
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthPreflightV7Error(
            f"parent attestation invalid: {type(error).__name__}"
        ) from error
    if not isinstance(attestation, dict):
        _fail("parent attestation is not an object")
    content = dict(attestation)
    claimed = content.pop("attestation_sha256", None)
    if claimed != canonical_json_sha256(content):
        _fail("parent attestation self-SHA mismatch")
    current = compute_current_bindings(root)
    if attestation.get("current_bindings") != current:
        _fail("parent attestation current code/config SHA drift")
    audit = verify_parent_reuse(
        config["parent_reuse"]["output_dir"],
        expected=_parent_expected(config, attestation),
    )
    rebuilt = build_parent_reuse_attestation(audit, current_bindings=current)
    if rebuilt != attestation:
        _fail("parent attestation does not reproduce from disk")
    return audit


def _ordered_adapter_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = path / name
        if item.is_symlink() or not item.is_file():
            _fail(f"Teacher adapter lacks immutable {name}")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_static_assets(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Stream-verify Base, Teacher and frozen prompt assets without model load."""

    from src.opd.production_qualification_prompts_v6 import (
        validate_frozen_prompt_selection,
    )

    shadow = {
        "run": {"run_id": config["parent_reuse"]["run_id"], "seed": 42},
        "prompt_selection": dict(config["prompt_selection"]),
    }
    validate_frozen_prompt_selection(shadow, repo_root=root)
    static = (
        (config["model"]["artifact_manifest_path"], config["model"]["artifact_manifest_sha256"]),
        (Path(config["model"]["id"]) / "config.json", config["model"]["config_sha256"]),
        (Path(config["model"]["id"]) / "generation_config.json", config["model"]["generation_config_sha256"]),
        (config["teacher_binding_for_future_b2_package"]["route_config"], config["teacher_binding_for_future_b2_package"]["route_config_sha256"]),
        (config["teacher_binding_for_future_b2_package"]["manifest_path"], config["teacher_binding_for_future_b2_package"]["manifest_sha256"]),
        (config["prompt_selection"]["selection_manifest_path"], config["prompt_selection"]["selection_manifest_sha256"]),
        (config["prompt_selection"]["opd_manifest_path"], config["prompt_selection"]["opd_manifest_sha256"]),
        (config["prompt_selection"]["medical_opd_o1_path"], config["prompt_selection"]["medical_opd_o1_sha256"]),
        (config["prompt_selection"]["medical_opd_cmb_path"], config["prompt_selection"]["medical_opd_cmb_sha256"]),
    )
    for path_value, expected in static:
        path = _resolve(root, path_value)
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            _fail(f"static asset SHA mismatch: {path}")
    artifact_manifest_path = _resolve(
        root, config["model"]["artifact_manifest_path"]
    )
    try:
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthPreflightV7Error(
            "Base/tokenizer artifact manifest is invalid"
        ) from error
    artifact_files = artifact_manifest.get("files")
    required_artifact_files = {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if not (
        isinstance(artifact_files, Mapping)
        and set(artifact_files) == required_artifact_files
        and artifact_manifest.get("tokenizer_revision")
        == config["model"]["tokenizer_revision"]
    ):
        _fail("Base/tokenizer artifact manifest inventory drift")
    base_directory = _resolve(root, config["model"]["id"]).resolve()
    for name in sorted(required_artifact_files):
        descriptor = artifact_files[name]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "bytes",
            "sha256",
        }:
            _fail("Base/tokenizer artifact descriptor drift")
        item = (base_directory / name).resolve()
        if (
            item.parent != base_directory
            or item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != descriptor["bytes"]
            or _sha(item) != descriptor["sha256"]
        ):
            _fail(f"Base/tokenizer artifact payload mismatch: {name}")
    tokenizer_config_path = base_directory / "tokenizer_config.json"
    try:
        tokenizer_config = json.loads(
            tokenizer_config_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthPreflightV7Error(
            "tokenizer config is invalid"
        ) from error
    template = tokenizer_config.get("chat_template")
    template_sha = (
        hashlib.sha256(template.encode("utf-8")).hexdigest()
        if isinstance(template, str) and template
        else None
    )
    if template_sha != config["model"]["prompt_template_sha256"]:
        _fail("prompt template SHA mismatch")
    try:
        base_transport = validate_base_model_transport(
            base_path=_resolve(root, config["model"]["id"]),
            weights_manifest_path=_resolve(
                root, config["model"]["weights_manifest_path"]
            ),
            weights_manifest_sha256=config["model"]["weights_manifest_sha256"],
            expected_revision=str(config["model"]["revision"]),
            verify_weight_payloads=True,
        )
    except ProductionQualificationPreflightError as error:
        raise ProductionLengthPreflightV7Error(str(error)) from error
    teacher = config["teacher_binding_for_future_b2_package"]
    adapter = _resolve(root, teacher["adapter_path"])
    weight = adapter / "adapter_model.safetensors"
    if (
        weight.is_symlink()
        or not weight.is_file()
        or _sha(weight) != teacher["adapter_weight_sha256"]
        or _ordered_adapter_sha256(adapter) != teacher["adapter_sha256"]
    ):
        _fail("Teacher adapter identity mismatch")
    manifest_path = _resolve(root, teacher["manifest_path"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthPreflightV7Error("Teacher manifest is invalid") from error
    if not (
        isinstance(manifest, Mapping)
        and Path(str(manifest.get("adapter_path", ""))).resolve() == adapter.resolve()
        and manifest.get("adapter_sha256") == teacher["adapter_sha256"]
        and manifest.get("adapter_weight_sha256") == teacher["adapter_weight_sha256"]
        and manifest.get("base_model_revision") == config["model"]["revision"]
        and manifest.get("tokenizer_revision") == config["model"]["tokenizer_revision"]
    ):
        _fail("Teacher manifest binding mismatch")
    return {
        "base_transport": base_transport,
        "teacher_adapter_weight_verified": True,
        "teacher_ordered_adapter_verified": True,
        "teacher_manifest_binding_verified": True,
        "prompt_data_files_verified": 2,
        "tokenizer_artifact_files_verified": len(required_artifact_files),
        "prompt_template_sha256": template_sha,
    }


def _git_identity(root: Path, *, allow_dirty: bool) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if branch != SOURCE_BRANCH:
        _fail("P4.7 source branch mismatch")
    if dirty and not allow_dirty:
        _fail("P4.7 worktree is not clean")
    return {"branch": branch, "git_commit": commit, "worktree_clean": not dirty}


def _disk_projection(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    projected_increment = int(
        config["resources"].get("projected_peak_increment_bytes", 1024**3)
    )
    minimum = int(
        config["resources"]["minimum_projected_free_at_peak_gib"] * 1024**3
    )
    projected = free - projected_increment
    if projected <= minimum:
        _fail("blocked_disk_projection")
    return {
        "free_bytes": free,
        "projected_peak_increment_bytes": projected_increment,
        "projected_minimum_free_bytes": projected,
        "required_strictly_greater_than_bytes": minimum,
    }


def _gpu_host(config: Mapping[str, Any]) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip().splitlines()
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        header = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True
        ).stdout
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProductionLengthPreflightV7Error("two RTX 3090 GPUs are unavailable") from error
    expected = config["resources"]["expected_gpu_model"]
    parsed = [line.split(", ") for line in lines]
    if len(parsed) != 2 or any(
        len(row) != 5 or row[0] != str(index) or row[1] != expected
        for index, row in enumerate(parsed)
    ):
        _fail("GPU identity mismatch")
    if any(int(row[4]) > 16 for row in parsed):
        _fail("GPU is not idle before formal execution")
    compute_rows = [
        line for line in compute.splitlines() if line and "no running" not in line.lower()
    ]
    if compute_rows:
        _fail("GPU compute PID exists before formal execution")
    residual_workers: list[int] = []
    try:
        for line in subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines():
            lowered = line.lower()
            if any(
                token in lowered
                for token in ("ray::", "vllm", "torchrun", "-m verl", "/verl/")
            ):
                residual_workers.append(int(line.strip().split(None, 1)[0]))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise ProductionLengthPreflightV7Error(
            "cannot inspect residual GPU worker processes"
        ) from error
    if residual_workers:
        _fail("residual Ray/vLLM/torchrun process exists before execution")
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    if cuda_match is None or "GPU0" not in topology or "GPU1" not in topology:
        _fail("GPU driver/CUDA/topology evidence is incomplete")
    return {
        "driver_version": parsed[0][2],
        "cuda_version": cuda_match.group(1),
        "topology": topology,
        "compute_processes": [],
        "residual_worker_pids": [],
        "gpus": [
            {
                "index": index,
                "name": row[1],
                "total_mib": int(row[3]),
                "used_mib": int(row[4]),
            }
            for index, row in enumerate(parsed)
        ]
    }


def _read_text_if_present(path: str) -> str | None:
    item = Path(path)
    if not item.is_file():
        return None
    try:
        return item.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _host_resources() -> dict[str, Any]:
    memory_info: dict[str, str] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory_info[key] = value.strip()
    except OSError:
        pass
    shm = shutil.disk_usage("/dev/shm")
    return {
        "cpu_count": os.cpu_count(),
        "memory": memory_info,
        "cgroup_cpu_max": _read_text_if_present("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _read_text_if_present("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_current": _read_text_if_present(
            "/sys/fs/cgroup/memory.current"
        ),
        "cgroup_swap_max": _read_text_if_present("/sys/fs/cgroup/memory.swap.max"),
        "dev_shm": {
            "total_bytes": shm.total,
            "used_bytes": shm.used,
            "free_bytes": shm.free,
        },
    }


def preflight(
    config_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    allow_dirty_for_development: bool = False,
    execute_gpu: bool = False,
) -> dict[str, Any]:
    root = _repo_root()
    path = Path(config_path).resolve()
    if path.parent.parent.parent != root or path.name != "qwen3_4b_length_qualification_v7.yaml":
        _fail("P4.7 config path is not canonical")
    config = _load_config(path)
    card = _load_card(root, path, config)
    _validate_json_schemas(root, config)
    git = _git_identity(root, allow_dirty=allow_dirty_for_development)
    output = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else Path(str(config["run"]["output_dir"]))
    )
    if output.exists() or output.is_symlink():
        _fail("formal P4.7 output already exists")
    package_dir = Path(str(config["run"]["generated_b2_package_dir"]))
    if output_dir_override is None and (package_dir.exists() or package_dir.is_symlink()):
        _fail("formal B2 package output already exists")
    disk = _disk_projection(config, output)
    static_assets = verify_static_assets(root, config)
    parent_audit = _verify_parent_and_attestation(root, config, card)
    current_bindings = compute_current_bindings(root)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "verl", "vllm")
    }
    if versions != {key: str(value) for key, value in config["versions"].items()}:
        _fail("installed package version drift")
    gpu = _gpu_host(config) if execute_gpu else None
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "ready_waiting_for_gpu_length_qualification",
        "git": git,
        "parent_core_evidence_verified": parent_audit[
            "parent_core_evidence_verified"
        ],
        "v2_adapter_reusable": parent_audit["v2_adapter_reusable"],
        "parent_audit_sha256": canonical_json_sha256(parent_audit),
        "current_bindings_sha256": canonical_json_sha256(current_bindings),
        "disk_projection": disk,
        "versions": versions,
        "static_assets": static_assets,
        "static_assets_sha256": canonical_json_sha256(static_assets),
        "gpu_host": gpu,
        "host_resources": _host_resources(),
        "gpu_used": False,
        "loaded_real_model": False,
        "gpu_length_qualification_pending": True,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
        "isolation": dict(config["isolation"]),
    }


def reverify_finalization_authority(
    caller_config: Mapping[str, Any],
) -> dict[str, Any]:
    """CPU-only post-worker replay of every authority-bearing static input.

    Unlike the initial preflight this deliberately permits the formal output
    directory to exist.  It reopens the canonical committed config/run card,
    current code bindings, protected parent graph, Base/Teacher transports and
    installed package versions before any B2 authorization can be sealed.
    """

    root = _repo_root()
    path = root / "configs/opd/qwen3_4b_length_qualification_v7.yaml"
    config = _load_config(path)
    if canonical_json_sha256(config) != canonical_json_sha256(caller_config):
        _fail("finalizer caller config differs from canonical committed config")
    card = _load_card(root, path, config)
    _validate_json_schemas(root, config)
    git = _git_identity(root, allow_dirty=False)
    static_assets = verify_static_assets(root, config)
    parent_audit = _verify_parent_and_attestation(root, config, card)
    current_bindings = compute_current_bindings(root)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "verl", "vllm")
    }
    if versions != {key: str(value) for key, value in config["versions"].items()}:
        _fail("installed package version drift during CPU finalization")
    return {
        "schema_version": 7,
        "artifact_kind": "p4_7_finalizer_authority_revalidation",
        "parent_core_evidence_verified": parent_audit[
            "parent_core_evidence_verified"
        ],
        "v2_adapter_reusable": parent_audit["v2_adapter_reusable"],
        "current_bindings_verified": True,
        "static_assets_verified": True,
        "git_commit": git["git_commit"],
        "worktree_clean": git["worktree_clean"],
        "parent_audit_sha256": canonical_json_sha256(parent_audit),
        "current_bindings_sha256": canonical_json_sha256(current_bindings),
        "static_assets_sha256": canonical_json_sha256(static_assets),
        "versions": versions,
        "B2_started": False,
        "isolation": dict(config["isolation"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.7 CPU-safe length preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir-override")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args(argv)
    result = preflight(
        args.config,
        output_dir_override=args.output_dir_override,
        allow_dirty_for_development=args.allow_dirty_for_development,
        execute_gpu=args.execute_gpu,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CURRENT_BINDING_PATHS",
    "ProductionLengthPreflightV7Error",
    "compute_current_bindings",
    "preflight",
    "reverify_finalization_authority",
    "validate_frozen_config",
    "verify_static_assets",
]
