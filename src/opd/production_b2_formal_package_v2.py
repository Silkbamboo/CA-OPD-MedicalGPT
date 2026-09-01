"""Build and verify the immutable P5.1 Formal B2 v2 package on CPU."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_data_v1 import validate_formal_b2_prompt_schedule
from src.opd.production_b2_formal_package_v1 import production_environment_metadata
from src.opd.production_b2_formal_v1 import FormalB2Error, validate_production_environment
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_main_protocol_v2 import validate_method_packages_v2


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_PYTHON = Path("artifacts/env/bin/python")
MEASURED_FULL_CHECKPOINT_BYTES = 397_700_000
PREDICTED_LOG_GROWTH_BYTES = 1_000_000_000
DISK_SAFETY_FLOOR_BYTES = (
    10_000_000_000
    + 2 * MEASURED_FULL_CHECKPOINT_BYTES
    + PREDICTED_LOG_GROWTH_BYTES
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalB2Error(f"Formal B2 v2 {label} is invalid") from error
    if not isinstance(value, Mapping):
        raise FormalB2Error(f"Formal B2 v2 {label} is not an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_source_package(source: Path) -> dict[str, Any]:
    index = dict(_json(source / "package_index.json", "source package index"))
    files = index.get("files")
    if not isinstance(files, Mapping):
        raise FormalB2Error("source Formal B2 package descriptors are absent")
    for name, descriptor in files.items():
        path = source / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise FormalB2Error("source Formal B2 package SHA differs")
    if index.get("package_content_sha256") != _canonical_sha(dict(files)):
        raise FormalB2Error("source Formal B2 package content SHA differs")
    return index


def validate_fresh_v0_canary_binding_v2(
    canary: Mapping[str, Any],
    *,
    fixed_qualification_sha256: str,
    formula_sha256: str,
    candidate_acceptance_sha256: str,
    preupdate_backend_health_sha256: str,
    correction_gate_qualification_sha256: str,
    selected_learning_rate: float,
) -> dict[str, Any]:
    """Require the non-promotable eight-commit canary before formal packaging."""

    identity = canary.get("fresh_v0_identity")
    sources = canary.get("source_counts")
    try:
        passed = bool(
            canary.get("passed") is True
            and canary.get("accepted_optimizer_commits") == 8
            and canary.get("rejected_attempts") == 0
            and float(canary.get("selected_common_learning_rate"))
            == float(selected_learning_rate)
            and canary.get("bounded_influence_formula_sha256") == formula_sha256
            and canary.get("fixed_token_qualification_sha256")
            == fixed_qualification_sha256
            and canary.get("candidate_acceptance_sha256")
            == candidate_acceptance_sha256
            and canary.get("preupdate_backend_health_sha256")
            == preupdate_backend_health_sha256
            and canary.get("correction_gate_qualification_sha256")
            == correction_gate_qualification_sha256
            and isinstance(identity, Mapping)
            and identity.get("zero_effect_verified") is True
            and identity.get("tensor_count") == 504
            and identity.get("source_adapter_path") is None
            and identity.get("logical_version") == 0
            and canary.get("policy_transition") == "v0_to_v8"
            and isinstance(sources, Mapping)
            and sources.get("medical_opd_o1") == 16
            and sources.get("medical_opd_cmb") == 16
            and canary.get("transaction_commit_passed") is True
            and canary.get("rollback_primitive_previously_fixed_token_verified")
            is True
            and canary.get("fresh_adapter_reload_each_step_passed") is True
            and canary.get("teacher_gradient_tensor_count") == 0
            and canary.get("base_gradient_tensor_count") == 0
            and canary.get("trainable_tensor_count") == 504
            and int(canary.get("minimum_disk_free_bytes"))
            >= DISK_SAFETY_FLOOR_BYTES
            and canary.get("canary_weights_promoted_to_formal") is False
            and canary.get("final_access_count") == 0
            and canary.get("controller_access_count") == 0
            and canary.get("label_access_count") == 0
        )
    except (TypeError, ValueError) as error:
        raise FormalB2Error("Formal B2 v2 fresh-v0 canary is invalid") from error
    if not passed:
        raise FormalB2Error("Formal B2 v2 fresh-v0 canary is incomplete or leaky")
    return {
        "passed": True,
        "accepted_optimizer_commits": 8,
        "rejected_attempts": 0,
        "candidate_acceptance_sha256": candidate_acceptance_sha256,
        "preupdate_backend_health_sha256": preupdate_backend_health_sha256,
        "correction_gate_qualification_sha256": correction_gate_qualification_sha256,
        "fixed_token_qualification_sha256": fixed_qualification_sha256,
        "formula_sha256": formula_sha256,
        "canary_weights_promoted_to_formal": False,
        "final_access_count": 0,
    }


def _common_method_packages(
    config: Mapping[str, Any],
    *,
    schedule_sha256: str,
    ratio_protocol_sha256: str,
    candidate_acceptance_sha256: str,
    preupdate_backend_health_sha256: str,
) -> list[dict[str, Any]]:
    protocol = config["protocol"]
    backend = config["production_backend"]
    common = {
        "base_revision": backend["model_revision"],
        "tokenizer_revision": backend["tokenizer_revision"],
        "initialization": "fresh_base_plus_zero_effect_lora_v0",
        "seed": 42,
        "precision": "bfloat16",
        "response_length": 1024,
        "physical_microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "target_position_chunk_size": 128,
        "gradient_checkpointing_use_reentrant": False,
        "use_cache": False,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": float(protocol["learning_rate"]),
        "optimizer": "AdamW",
        "scheduler": "constant",
        "bounded_influence": deepcopy(dict(config["bounded_influence_v2"])),
        "stage1_optimizer_steps": 120,
        "prompts_per_step": 4,
        "schedule_sha256": schedule_sha256,
        "ratio_protocol_sha256": ratio_protocol_sha256,
        "candidate_acceptance_sha256": candidate_acceptance_sha256,
        "preupdate_backend_health_sha256": preupdate_backend_health_sha256,
        "checkpoint_steps": [30, 60, 90, 120],
        "controller_steps": [0, 30, 60, 90, 120],
        "production_backend": "custom_transformers_peft_three_policy_loop",
        "final_access": False,
    }
    methods = {
        "B2": {"teacher_route": "medical_only"},
        "IDT": {"teacher_route": "fixed_medical_base_1_to_1"},
        "CA-OPD": {
            "teacher_route": "adaptive_medical_base",
            "kl_safety_scaling": True,
        },
    }
    return [
        {
            "schema_version": 2,
            "artifact_kind": "formal_method_package_template_v2",
            "method_id": method,
            "status": "authorized" if method == "B2" else "template_not_authorized",
            "common": deepcopy(common),
            "method": fields,
            "authorization": {
                "formal_training_authorized": method == "B2",
                "launch_in_p5_1": method == "B2",
            },
        }
        for method, fields in methods.items()
    ]


def build_formal_package_v2(
    package: Path,
    *,
    output: Path,
    source_package: Path,
    cpu_gate: Path,
    fixed_token_qualification: Path,
    formula_path: Path,
    fresh_v0_canary: Path,
    candidate_acceptance_path: Path,
    preupdate_backend_health_path: Path,
    correction_gate_qualification: Path,
) -> dict[str, Any]:
    package = Path(package).resolve()
    output = Path(output).resolve()
    source_package = Path(source_package).resolve()
    if package.exists() or package.is_symlink() or output.exists() or output.is_symlink():
        raise FormalB2Error("Formal B2 v2 package/output must be fresh")
    if _git("status", "--porcelain"):
        raise FormalB2Error("Formal B2 v2 package requires clean committed Git")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    source_index = _verify_source_package(source_package)
    cpu = _json(cpu_gate, "CPU quick gate")
    qualification = _json(fixed_token_qualification, "fixed-token qualification")
    canary_path = Path(fresh_v0_canary).resolve()
    canary = _json(canary_path, "fresh-v0 canary")
    candidate_path = Path(candidate_acceptance_path).resolve()
    candidate_acceptance = _json(candidate_path, "candidate acceptance v2.1")
    preupdate_path = Path(preupdate_backend_health_path).resolve()
    preupdate_backend_health = _json(preupdate_path, "preupdate backend health v2.1")
    correction_qualification_path = Path(correction_gate_qualification).resolve()
    correction_qualification = _json(
        correction_qualification_path, "correction gate qualification"
    )
    formula_path = Path(formula_path).resolve()
    if not (
        formula_path.is_file()
        and not formula_path.is_symlink()
        and canary_path.is_file()
        and not canary_path.is_symlink()
        and candidate_path.is_file()
        and not candidate_path.is_symlink()
        and preupdate_path.is_file()
        and not preupdate_path.is_symlink()
        and correction_qualification_path.is_file()
        and not correction_qualification_path.is_symlink()
    ):
        raise FormalB2Error("Formal B2 v2 versioned formula is absent")
    formula_sha256 = _sha_file(formula_path)
    qualification_sha256 = _sha_file(Path(fixed_token_qualification).resolve())
    canary_sha256 = _sha_file(canary_path)
    candidate_acceptance_sha256 = _sha_file(candidate_path)
    preupdate_backend_health_sha256 = _sha_file(preupdate_path)
    correction_gate_qualification_sha256 = _sha_file(
        correction_qualification_path
    )
    thresholds = dict(_json(REPO / "reports/p5_1_ratio_health_thresholds_v2.json", "thresholds"))
    if not (
        cpu.get("failed") == 0
        and int(cpu.get("passed", 0)) >= 550
        and cpu.get("skipped") == 0
        and cpu.get("production_python") == str(PRODUCTION_PYTHON)
        and qualification.get("passed") is True
        and qualification.get("canonical_identity_passed") is True
        and qualification.get("rollback_passed") is True
        and qualification.get("step25_risk_captured") is True
        and qualification.get("bounded_influence_qualified") is True
        and qualification.get("bounded_influence_formula_sha256")
        == formula_sha256
        and qualification.get("selected_common_learning_rate") in (3.0e-5, 1.0e-5)
        and thresholds.get("written_before_new_gpu_results") is True
    ):
        raise FormalB2Error("Formal B2 v2 CPU/GPU qualification is incomplete")
    if not (
        candidate_acceptance.get("schema_version") == 1
        and candidate_acceptance.get("protocol_id")
        == "p5_1_candidate_acceptance_v2_1"
        and candidate_acceptance.get("formula_sha256") == formula_sha256
        and candidate_acceptance.get("fresh_optimizer_direction_hard_gate") is True
        and candidate_acceptance.get("accumulated_adam_same_batch_monotonicity")
        == "diagnostic_only"
        and set(candidate_acceptance.get("common_methods", []))
        == {"B2", "IDT", "CA-OPD"}
    ):
        raise FormalB2Error("Formal B2 v2 candidate acceptance protocol differs")
    if not (
        preupdate_backend_health.get("schema_version") == 1
        and preupdate_backend_health.get("protocol_id")
        == "p5_1_preupdate_backend_health_v2_1"
        and preupdate_backend_health.get("backend_clip_fraction_aggregation")
        == "token_pooled"
        and preupdate_backend_health.get("per_prompt_ess_min_valid_tokens") == 32
        and set(preupdate_backend_health.get("common_methods", []))
        == {"B2", "IDT", "CA-OPD"}
        and preupdate_backend_health.get("requires_fresh_v0") is True
        and correction_qualification.get("status") == "qualified"
        and correction_qualification.get("protocol_id")
        == "p5_1_preupdate_backend_health_v2_1"
        and correction_qualification.get("protocol_config_sha256")
        == preupdate_backend_health_sha256
        and correction_qualification.get("candidate_counts_as_optimizer_commit")
        is False
        and correction_qualification.get("final_access_count") == 0
        and correction_qualification.get("rollback", {}).get("rollback_verified")
        is True
    ):
        raise FormalB2Error("Formal B2 v2 correction gate protocol differs")
    canary_attestation = validate_fresh_v0_canary_binding_v2(
        canary,
        fixed_qualification_sha256=qualification_sha256,
        formula_sha256=formula_sha256,
        candidate_acceptance_sha256=candidate_acceptance_sha256,
        preupdate_backend_health_sha256=preupdate_backend_health_sha256,
        correction_gate_qualification_sha256=correction_gate_qualification_sha256,
        selected_learning_rate=float(qualification["selected_common_learning_rate"]),
    )
    environment = production_environment_metadata()
    validate_production_environment(environment)
    schedule = deepcopy(dict(_json(source_package / "prompt_schedule.json", "schedule")))
    authority = deepcopy(dict(_json(source_package / "data_authority.json", "authority")))
    validate_formal_b2_prompt_schedule(schedule, authority=authority)
    config = deepcopy(dict(_json(source_package / "formal_b2_config.json", "source config")))
    config.update(
        {
            "schema_id": "ca-opd/formal-b2-medical-opd/v2",
            "schema_version": 2,
            "package_version": "p5_1_formal_b2_v2",
            "run": {
                "run_id": output.name,
                "seed": 42,
                "optimizer_steps": 150,
                "stage1_stop_step": 120,
                "output_dir": str(output),
            },
            "ratio_health_v2": {
                "protocol": "docs/decisions/0031-ratio-health-protocol-v2.md",
                "protocol_sha256": _sha_file(REPO / "docs/decisions/0031-ratio-health-protocol-v2.md"),
                "thresholds_file_sha256": _sha_file(REPO / "reports/p5_1_ratio_health_thresholds_v2.json"),
                "selected_common_learning_rate": float(qualification["selected_common_learning_rate"]),
                "thresholds": thresholds,
                "transactional_commit": True,
                "rejected_attempt_counts_as_step": False,
            },
            "bounded_influence_v2": deepcopy(
                dict(qualification["bounded_influence_protocol"])
            ),
            "candidate_acceptance_v2_1": deepcopy(dict(candidate_acceptance)),
            "preupdate_backend_health_v2_1": deepcopy(
                dict(preupdate_backend_health)
            ),
        }
    )
    config["protocol"]["learning_rate"] = float(qualification["selected_common_learning_rate"])
    config["protocol"]["three_policy_formula_path"] = str(formula_path)
    config["protocol"]["three_policy_formula_sha256"] = formula_sha256
    config["data"]["schedule_path"] = str(package / "prompt_schedule.json")
    formal_b2_runtime_config_v2(config)
    disk = validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output.parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    method_packages = _common_method_packages(
        config,
        schedule_sha256=str(schedule["schedule_sha256"]),
        ratio_protocol_sha256=config["ratio_health_v2"]["protocol_sha256"],
        candidate_acceptance_sha256=candidate_acceptance_sha256,
        preupdate_backend_health_sha256=preupdate_backend_health_sha256,
    )
    common_attestation = validate_method_packages_v2(method_packages)
    run_card = {
        "schema_version": 2,
        "artifact_kind": "p5_1_formal_b2_v2_run_card",
        "run_id": output.name,
        "status": "authorized_not_started",
        "accepted_optimizer_commits_target": 120,
        "rejected_attempts_count_as_steps": False,
        "prompts_per_commit": 4,
        "stage1_prompt_budget": 480,
        "controller_steps": [0, 30, 60, 90, 120],
        "selected_common_learning_rate": qualification["selected_common_learning_rate"],
        "bounded_influence": deepcopy(dict(config["bounded_influence_v2"])),
        "candidate_acceptance_sha256": candidate_acceptance_sha256,
        "preupdate_backend_health_sha256": preupdate_backend_health_sha256,
        "correction_gate_qualification_sha256": correction_gate_qualification_sha256,
        "fresh_v0_canary_sha256": canary_sha256,
        "price": {
            "live_instance_price_cny_per_hour": None,
            "live_price_unavailable_reason": "instance-specific price unavailable inside container",
            "historical_reference_cny_per_hour": 2.96,
            "actual_cost_cny": None,
        },
        "runtime_estimate_hours": [7, 10],
        "derived_reference_cost_cny": [20.72, 29.6],
        "disk": disk,
        "checkpoint_retention": "latest_two_full_resume_plus_registered_adapter_snapshots",
        "backend_fact": "custom Transformers/PEFT three-policy loop; not full veRL trainer",
        "final_authorized": False,
    }
    core: dict[str, Any] = {
        "formal_b2_config.json": config,
        "prompt_schedule.json": schedule,
        "data_authority.json": authority,
        "environment.json": environment,
        "ratio_health_thresholds_v2.json": thresholds,
        "fixed_token_qualification.json": qualification,
        "fresh_v0_canary.json": canary,
        "candidate_acceptance_v2_1.json": candidate_acceptance,
        "preupdate_backend_health_v2_1.json": preupdate_backend_health,
        "correction_gate_qualification.json": correction_qualification,
        "canary_attestation.json": canary_attestation,
        "run_card.json": run_card,
        "common_protocol_attestation.json": common_attestation,
    }
    for method in method_packages:
        core[f"method_{method['method_id'].lower().replace('-', '_')}.json"] = method
    package.mkdir(parents=True)
    for name, value in core.items():
        _atomic_json(package / name, value)
    files = {
        name: {"sha256": _sha_file(package / name), "size_bytes": (package / name).stat().st_size}
        for name in sorted(core)
    }
    index = {
        "schema_version": 2,
        "artifact_kind": "p5_1_formal_b2_v2_package_index",
        "package_version": "p5_1_formal_b2_v2",
        "run_id": output.name,
        "files": files,
        "package_content_sha256": _canonical_sha(files),
        "config_sha256": files["formal_b2_config.json"]["sha256"],
        "schedule_file_sha256": files["prompt_schedule.json"]["sha256"],
        "schedule_semantic_sha256": schedule["schedule_sha256"],
        "manifest_sha256": authority["manifest_sha256"],
        "source_p5_package_content_sha256": source_index["package_content_sha256"],
        "source_p5_package_index_sha256": _sha_file(source_package / "package_index.json"),
        "source_fixed_token_qualification_sha256": qualification_sha256,
        "source_fresh_v0_canary_sha256": canary_sha256,
        "source_candidate_acceptance_sha256": candidate_acceptance_sha256,
        "source_preupdate_backend_health_sha256": preupdate_backend_health_sha256,
        "source_correction_gate_qualification_sha256": correction_gate_qualification_sha256,
    }
    _atomic_json(package / "package_index.json", index)
    authorization = {
        "schema_version": 2,
        "artifact_kind": "p5_1_formal_b2_v2_authorization",
        "formal_B2_authorized": True,
        "user_authorized_this_turn": True,
        "ratio_forensic_complete": True,
        "protocol_v2_frozen": True,
        "quick_gate_zero_failed_zero_skipped": True,
        "fixed_token_gpu_qualification_passed": True,
        "fresh_v0_canary_passed": True,
        "canary_weights_not_promoted": True,
        "candidate_acceptance_v2_1_frozen": True,
        "preupdate_backend_health_v2_1_frozen": True,
        "correction_gate_gpu_qualification_passed": True,
        "fresh_v0_required": True,
        "output_fresh": True,
        "disk_safety_v2_passed": True,
        "common_protocol_hash_passed": True,
        "git_clean_committed": True,
        "git_branch": branch,
        "git_head": head,
        "package_content_sha256": index["package_content_sha256"],
        "package_index_sha256": _sha_file(package / "package_index.json"),
        "fresh_v0_canary_sha256": canary_sha256,
        "candidate_acceptance_sha256": candidate_acceptance_sha256,
        "preupdate_backend_health_sha256": preupdate_backend_health_sha256,
        "correction_gate_qualification_sha256": correction_gate_qualification_sha256,
        "final_authorized": False,
    }
    _atomic_json(package / "authorization.json", authorization)
    return verify_formal_package_v2(package, require_clean_git=True)


def verify_formal_package_v2(
    package: Path,
    *,
    require_clean_git: bool = True,
    allow_existing_output_for_resume: bool = False,
) -> dict[str, Any]:
    package = Path(package).resolve()
    index = _json(package / "package_index.json", "package index")
    authorization = _json(package / "authorization.json", "authorization")
    files = index.get("files")
    if not isinstance(files, Mapping):
        raise FormalB2Error("Formal B2 v2 package descriptors are absent")
    for name, descriptor in files.items():
        path = package / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise FormalB2Error(f"Formal B2 v2 package file differs: {name}")
    content_sha = _canonical_sha(dict(files))
    if not (
        index.get("schema_version") == 2
        and index.get("package_version") == "p5_1_formal_b2_v2"
        and index.get("package_content_sha256") == content_sha
        and authorization.get("formal_B2_authorized") is True
        and authorization.get("package_content_sha256") == content_sha
        and authorization.get("package_index_sha256") == _sha_file(package / "package_index.json")
        and authorization.get("final_authorized") is False
        and all(
            authorization.get(field) is True
            for field in (
                "user_authorized_this_turn",
                "ratio_forensic_complete",
                "protocol_v2_frozen",
                "quick_gate_zero_failed_zero_skipped",
                "fixed_token_gpu_qualification_passed",
                "fresh_v0_canary_passed",
                "canary_weights_not_promoted",
                "candidate_acceptance_v2_1_frozen",
                "preupdate_backend_health_v2_1_frozen",
                "correction_gate_gpu_qualification_passed",
                "fresh_v0_required",
                "output_fresh",
                "disk_safety_v2_passed",
                "common_protocol_hash_passed",
                "git_clean_committed",
            )
        )
    ):
        raise FormalB2Error("Formal B2 v2 package authorization differs")
    config = _json(package / "formal_b2_config.json", "config")
    schedule = _json(package / "prompt_schedule.json", "schedule")
    authority = _json(package / "data_authority.json", "authority")
    environment = _json(package / "environment.json", "environment")
    validate_production_environment(environment)
    formal_b2_runtime_config_v2(config)
    qualification = _json(
        package / "fixed_token_qualification.json", "fixed-token qualification"
    )
    canary = _json(package / "fresh_v0_canary.json", "fresh-v0 canary")
    candidate_acceptance = _json(
        package / "candidate_acceptance_v2_1.json", "candidate acceptance v2.1"
    )
    preupdate_backend_health = _json(
        package / "preupdate_backend_health_v2_1.json",
        "preupdate backend health v2.1",
    )
    correction_qualification = _json(
        package / "correction_gate_qualification.json",
        "correction gate qualification",
    )
    if not (
        candidate_acceptance.get("formula_sha256")
        == config["protocol"]["three_policy_formula_sha256"]
        and authorization.get("fresh_v0_canary_sha256")
        == index.get("source_fresh_v0_canary_sha256")
        and authorization.get("candidate_acceptance_sha256")
        == index.get("source_candidate_acceptance_sha256")
        and authorization.get("preupdate_backend_health_sha256")
        == index.get("source_preupdate_backend_health_sha256")
        and authorization.get("correction_gate_qualification_sha256")
        == index.get("source_correction_gate_qualification_sha256")
        and correction_qualification.get("protocol_config_sha256")
        == index.get("source_preupdate_backend_health_sha256")
        and config.get("preupdate_backend_health_v2_1")
        == preupdate_backend_health
    ):
        raise FormalB2Error("Formal B2 v2 canary authorization binding differs")
    validate_fresh_v0_canary_binding_v2(
        canary,
        fixed_qualification_sha256=str(
            index["source_fixed_token_qualification_sha256"]
        ),
        formula_sha256=str(config["protocol"]["three_policy_formula_sha256"]),
        candidate_acceptance_sha256=str(
            index["source_candidate_acceptance_sha256"]
        ),
        preupdate_backend_health_sha256=str(
            index["source_preupdate_backend_health_sha256"]
        ),
        correction_gate_qualification_sha256=str(
            index["source_correction_gate_qualification_sha256"]
        ),
        selected_learning_rate=float(
            qualification["selected_common_learning_rate"]
        ),
    )
    validate_formal_b2_prompt_schedule(schedule, authority=authority)
    if schedule["schedule_sha256"] != index.get("schedule_semantic_sha256"):
        raise FormalB2Error("Formal B2 v2 schedule semantic SHA differs")
    method_packages = [
        _json(package / name, name)
        for name in ("method_b2.json", "method_idt.json", "method_ca_opd.json")
    ]
    validate_method_packages_v2(method_packages)
    output = Path(str(config["run"]["output_dir"]))
    if output.is_symlink() or (output.exists() and not allow_existing_output_for_resume):
        raise FormalB2Error("Formal B2 v2 output is not fresh")
    if allow_existing_output_for_resume and not output.is_dir():
        raise FormalB2Error("Formal B2 v2 resume output is absent")
    disk = validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output.parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    if require_clean_git:
        if _git("status", "--porcelain"):
            raise FormalB2Error("Formal B2 v2 preflight requires clean Git")
        if _git("rev-parse", "HEAD") != authorization.get("git_head"):
            raise FormalB2Error("Formal B2 v2 Git HEAD differs")
        if _git("branch", "--show-current") != authorization.get("git_branch"):
            raise FormalB2Error("Formal B2 v2 Git branch differs")
    return {
        "passed": True,
        "package_content_sha256": content_sha,
        "config_sha256": index["config_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "output_dir": str(output),
        "disk": disk,
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
        "cuda_model_construction_calls": 0,
        "formal_B2_authorized": True,
        "final_authorized": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--package", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-package", type=Path, required=True)
    build.add_argument("--cpu-gate", type=Path, required=True)
    build.add_argument("--fixed-token-qualification", type=Path, required=True)
    build.add_argument("--formula-path", type=Path, required=True)
    build.add_argument("--fresh-v0-canary", type=Path, required=True)
    build.add_argument("--candidate-acceptance-path", type=Path, required=True)
    build.add_argument("--preupdate-backend-health-path", type=Path, required=True)
    build.add_argument("--correction-gate-qualification", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_formal_package_v2(
            args.package,
            output=args.output,
            source_package=args.source_package,
            cpu_gate=args.cpu_gate,
            fixed_token_qualification=args.fixed_token_qualification,
            formula_path=args.formula_path,
            fresh_v0_canary=args.fresh_v0_canary,
            candidate_acceptance_path=args.candidate_acceptance_path,
            preupdate_backend_health_path=args.preupdate_backend_health_path,
            correction_gate_qualification=args.correction_gate_qualification,
        )
    else:
        result = verify_formal_package_v2(args.package)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "build_formal_package_v2",
    "validate_fresh_v0_canary_binding_v2",
    "verify_formal_package_v2",
]
