"""Build and verify the immutable P5 formal B2 package without loading models."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from src.opd.production_b2_data_v2 import (
    canonical_json_sha256,
    resolve_b2_data_authority,
)
from src.opd.production_b2_formal_data_v1 import (
    build_formal_b2_prompt_schedule,
    validate_formal_b2_prompt_schedule,
)
from src.opd.production_b2_formal_v1 import (
    FORMAL_MEMORY_EXECUTION_CONTRACT,
    FormalB2Error,
    formal_b2_runtime_config,
    validate_production_environment,
)
from src.opd.production_b2_parent_attestation_v1 import (
    derive_rolling_safety_gates,
    summarize_parent_records,
    validate_parent_supporting_evidence,
)


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_PYTHON = Path("artifacts/env/bin/python")
MINIMUM_DISK_FREE_BYTES = 10_000_000_000
CHECKPOINT_OBSERVED_BYTES = 397_653_189
FORMAL_CHECKPOINT_COUNT_150 = 7
TRANSIENT_PEAK_BYTES = 700_000_000


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise FormalB2Error(f"formal package {label} is not a SHA-256")
    return value


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalB2Error(
            f"formal package {label} invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, Mapping):
        raise FormalB2Error(f"formal package {label} is not an object")
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
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def production_environment_metadata() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("verl", "transformers", "peft", "vllm")
    }
    import torch

    packages["torch"] = torch.__version__
    return {
        "python_executable": str(Path(os.sys.executable).resolve()),
        "environment_path": str(Path(os.sys.executable).resolve().parent.parent),
        "python_version": os.sys.version.split()[0],
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pytorch_cuda_available_at_cpu_preflight": bool(torch.cuda.is_available()),
        "backend_fact": "custom Transformers/PEFT three-policy production loop",
        "verl_usage": "veRL 0.8.0 pinned token-correction helper only",
    }


def _load_parent_records(parent: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in range(1, 21):
        record = dict(_json(parent / "steps" / f"step_{step:02d}.json", "parent step"))
        kernel = _json(
            parent
            / "b2_steps"
            / f"step_{step - 1:02d}_v{step - 1}_to_v{step}.json",
            "parent kernel step",
        )
        telemetry = kernel.get("reconstruction_telemetry")
        if not isinstance(telemetry, Mapping):
            raise FormalB2Error("parent reconstruction telemetry is absent")
        record["reconstruction_telemetry"] = deepcopy(dict(telemetry))
        records.append(record)
    return records


def _checkpoint_files_pass(parent: Path, index: Mapping[str, Any]) -> bool:
    checkpoints = index.get("checkpoints")
    if not isinstance(checkpoints, list):
        return False
    for item in checkpoints:
        if not isinstance(item, Mapping):
            return False
        checkpoint = parent / str(item.get("path", ""))
        manifest_path = checkpoint / "checkpoint_manifest.json"
        if not (
            checkpoint.is_dir()
            and not checkpoint.is_symlink()
            and manifest_path.is_file()
            and _sha_file(manifest_path) == item.get("manifest_sha256")
        ):
            return False
        manifest = _json(manifest_path, "parent checkpoint manifest")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            return False
        for name, descriptor in files.items():
            path = checkpoint / str(name)
            if not (
                isinstance(descriptor, Mapping)
                and path.is_file()
                and not path.is_symlink()
                and descriptor.get("sha256") == _sha_file(path)
                and descriptor.get("size_bytes") == path.stat().st_size
            ):
                return False
    return True


def recompute_parent_attestation(parent: Path) -> dict[str, Any]:
    parent = Path(parent).resolve()
    records = _load_parent_records(parent)
    summary = summarize_parent_records(records)
    raw_summary = _json(parent / "summary.json", "parent summary")
    final_index = _json(parent / "final_index.json", "parent final index")
    cleanup = _json(parent / "cleanup.json", "parent cleanup")
    checkpoint_index = _json(parent / "checkpoints/index.json", "checkpoint index")
    final_artifacts = final_index.get("artifacts")
    cleanup_bound = isinstance(final_artifacts, list) and any(
        isinstance(item, Mapping)
        and item.get("path") == "cleanup.json"
        and item.get("sha256") == _sha_file(parent / "cleanup.json")
        and item.get("size_bytes") == (parent / "cleanup.json").stat().st_size
        for item in final_artifacts
    )
    failure_artifacts = sorted(
        str(path.relative_to(parent))
        for path in parent.rglob("*failure*.json")
        if path.is_file()
    )
    supporting = validate_parent_supporting_evidence(
        summary,
        {
            "length_gate": raw_summary.get("length_gate"),
            "canary": _json(parent / "memory_canary.json", "parent canary"),
            "checkpoint_index": checkpoint_index,
            "v10_reload": _json(
                parent / "resume_reload_identity_v10.json", "parent v10 reload"
            ),
            "v20_reload": _json(
                parent / "final_reload_identity.json", "parent v20 reload"
            ),
            "cleanup": cleanup,
            "failure_artifacts": failure_artifacts,
            "cleanup_bound_by_final_index": cleanup_bound,
            "checkpoint_file_sha_passed": _checkpoint_files_pass(
                parent, checkpoint_index
            ),
        },
    )
    gates = derive_rolling_safety_gates(summary)
    return {
        "schema_version": 1,
        "artifact_kind": "p5_calibration_parent_attestation_v1",
        "observed": True,
        "parent_run_id": raw_summary.get("run_id", parent.name),
        "parent_output_path": str(parent),
        "parent_evidence_index_sha256": _sha_file(parent / "evidence_index.json"),
        "parent_final_index_sha256": _sha_file(parent / "final_index.json"),
        "parent_checkpoint_index_sha256": _sha_file(
            parent / "checkpoints/index.json"
        ),
        "summary": summary,
        "supporting_evidence": supporting,
        "rolling_safety_gates": gates,
        "known_observation_gap": {
            "field": "per_source_objective",
            "status": "not_recorded_not_reconstructible",
            "effect": "reported explicitly; formal runtime records it prospectively",
        },
        "passed": True,
    }


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": _sha_file(path), "size_bytes": path.stat().st_size}


def build_formal_package(
    package: Path,
    *,
    output: Path,
    parent: Path,
    parent_package: Path,
    cpu_gate: Path,
) -> dict[str, Any]:
    package = Path(package).resolve()
    output = Path(output).resolve()
    if package.exists() or package.is_symlink():
        raise FormalB2Error("formal package output is not fresh")
    if output.exists() or output.is_symlink():
        raise FormalB2Error("formal training output is not fresh")
    if _git("status", "--porcelain"):
        raise FormalB2Error("formal package requires clean committed worktree")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    gate = _json(cpu_gate, "CPU quick gate")
    if not (
        gate.get("failed") == 0
        and gate.get("passed", 0) >= 550
        and gate.get("production_python") == str(PRODUCTION_PYTHON)
    ):
        raise FormalB2Error("CPU quick gate is not 0 failed on production Python")
    environment = production_environment_metadata()
    validate_production_environment(environment)
    parent_attestation = recompute_parent_attestation(parent)
    if parent_attestation.get("passed") is not True:
        raise FormalB2Error("P4.8g parent attestation failed")
    parent_config = _json(
        parent_package / "b2_20_step_calibration_config.json", "parent config"
    )
    authority = resolve_b2_data_authority(
        Path(str(parent_config["data"]["prompt_manifest_path"])),
        expected_manifest_sha256=str(
            parent_config["data"]["prompt_manifest_sha256"]
        ),
    )
    parent_schedule = _json(parent_package / "prompt_schedule.json", "parent schedule")
    excluded = {
        str(row["sample_id"])
        for row in parent_schedule.get("slots", [])
        if isinstance(row, Mapping)
    }
    if len(excluded) != 80:
        raise FormalB2Error("P4.8g exclusion set is not exactly 80 prompts")
    schedule = build_formal_b2_prompt_schedule(
        authority,
        seed=42,
        optimizer_steps=150,
        excluded_sample_ids=excluded,
    )
    schedule_audit = validate_formal_b2_prompt_schedule(schedule, authority=authority)
    disk = shutil.disk_usage(output.parent)
    projected_checkpoint_bytes = (
        CHECKPOINT_OBSERVED_BYTES * FORMAL_CHECKPOINT_COUNT_150
    )
    projected_minimum_free = (
        disk.free - projected_checkpoint_bytes - TRANSIENT_PEAK_BYTES
    )
    if disk.free < MINIMUM_DISK_FREE_BYTES or projected_minimum_free < MINIMUM_DISK_FREE_BYTES:
        raise FormalB2Error("formal B2 disk projection cannot preserve 10 GB")
    protocol_path = REPO / "docs/decisions/0030-main-experiment-protocol-v1.md"
    if not protocol_path.is_file():
        raise FormalB2Error("Main Experiment Protocol v1 is absent")
    run_id = output.name
    config = deepcopy(dict(parent_config))
    config.update(
        {
            "schema_id": "ca-opd/formal-b2-medical-opd/v1",
            "schema_version": 1,
            "package_version": "p5_formal_b2_v1",
            "run": {
                "run_id": run_id,
                "seed": 42,
                "optimizer_steps": 150,
                "stage1_stop_step": 120,
                "output_dir": str(output),
            },
            "execution": {
                "optimizer_steps": 150,
                "stage1_stop_step": 120,
                "calibration_only": False,
                "automatically_start_b2": False,
            },
            "memory_execution": deepcopy(FORMAL_MEMORY_EXECUTION_CONTRACT),
            "qualification_evidence": deepcopy(dict(parent_config["qualification"])),
            "data": {
                **deepcopy(dict(parent_config["data"])),
                "provider": "production_b2_formal_data_v1.resolve_formal_b2_schedule_batch",
                "schedule_path": str(package / "prompt_schedule.json"),
                "schedule_sha256": schedule["schedule_sha256"],
                "schedule_version": schedule["schedule_version"],
                "selection_rule": "seed42 SHA rank; exclude all 80 P4.8g IDs; pre-freeze 600 unique slots",
            },
            "authorization": {
                "production_sampler_refresh_ready": True,
                "OPD_scoring_backend_ready": True,
                "formal_B2_authorized": True,
                "formal_B2_started": False,
            },
            "parent_calibration": {
                "usage": "engineering_evidence_only_not_student_init",
                "attestation_sha256": canonical_json_sha256(parent_attestation),
                "evidence_index_sha256": parent_attestation[
                    "parent_evidence_index_sha256"
                ],
                "final_index_sha256": parent_attestation[
                    "parent_final_index_sha256"
                ],
            },
            "main_experiment_protocol": {
                "path": str(protocol_path.relative_to(REPO)),
                "sha256": _sha_file(protocol_path),
                "version": "v1",
            },
        }
    )
    formal_b2_runtime_config(config)
    run_card = {
        "schema_version": 1,
        "artifact_kind": "p5_formal_b2_run_card_v1",
        "run_id": run_id,
        "status": "authorized_not_started",
        "method": "B2_100_percent_medical_teacher",
        "stage1_optimizer_steps": 120,
        "registered_max_optimizer_steps": 150,
        "prompts_per_step": 4,
        "stage1_prompt_count": 480,
        "registered_prompt_count": 600,
        "checkpoint_steps": [30, 60, 90, 120, 150],
        "rolling_checkpoint_interval": 10,
        "controller_steps": [0, 30, 60, 90, 120, 150],
        "extension_rule": {
            "step120_is_best_feasible": True,
            "medical_step90_to_120_gain_pp_at_least": 1.0,
            "step120_general_constraint": "B0 General - 1.0pp",
            "all_training_health_gates_pass": True,
            "all_required": True,
        },
        "estimated_runtime_hours": {"stage1": [10, 14], "extension": [12, 16]},
        "price": {
            "live_instance_price_cny_per_hour": None,
            "live_price_unavailable_reason": "instance-specific AutoDL host price is not exposed inside the container",
            "historical_reference_cny_per_hour": 2.96,
            "reference_only_not_live": True,
            "stage1_derived_cost_cny": [29.6, 41.44],
            "extension_derived_cost_cny": [35.52, 47.36],
            "run_card_hard_cap_derived_cny": 53.28,
            "actual_cost_cny": None,
        },
        "disk": {
            "free_bytes_before_package": disk.free,
            "minimum_required_bytes": MINIMUM_DISK_FREE_BYTES,
            "observed_complete_checkpoint_bytes": CHECKPOINT_OBSERVED_BYTES,
            "projected_checkpoint_bytes_at_150": projected_checkpoint_bytes,
            "transient_peak_budget_bytes": TRANSIENT_PEAK_BYTES,
            "projected_minimum_free_bytes": projected_minimum_free,
            "retention": "milestones plus latest/previous nonmilestone rollings",
        },
        "environment": environment,
        "backend_fact": "custom Transformers/PEFT three-policy production loop; not a veRL trainer",
        "final_authorized": False,
        "final_will_run": False,
    }
    package.mkdir(parents=True)
    core = {
        "formal_b2_config.json": config,
        "prompt_schedule.json": schedule,
        "data_authority.json": authority,
        "parent_attestation.json": parent_attestation,
        "rolling_safety_gates.json": parent_attestation["rolling_safety_gates"],
        "run_card.json": run_card,
        "environment.json": environment,
    }
    for name, value in core.items():
        _atomic_json(package / name, value)
    descriptors = {name: _descriptor(package / name) for name in sorted(core)}
    package_content_sha = canonical_json_sha256(descriptors)
    index = {
        "schema_version": 1,
        "artifact_kind": "p5_formal_b2_package_index_v1",
        "package_version": "p5_formal_b2_v1",
        "run_id": run_id,
        "files": descriptors,
        "package_content_sha256": package_content_sha,
        "config_sha256": descriptors["formal_b2_config.json"]["sha256"],
        "schedule_file_sha256": descriptors["prompt_schedule.json"]["sha256"],
        "schedule_semantic_sha256": schedule["schedule_sha256"],
        "manifest_sha256": authority["manifest_sha256"],
    }
    _atomic_json(package / "package_index.json", index)
    authorization = {
        "schema_version": 1,
        "artifact_kind": "p5_formal_b2_authorization_v1",
        "formal_B2_authorized": True,
        "user_authorized_this_turn": True,
        "parent_recomputed_from_disk": True,
        "quick_gate_zero_failed": True,
        "production_python_bound": True,
        "schedule_leakage_check_passed": True,
        "package_sha_passed": True,
        "output_fresh": True,
        "disk_at_least_10gb": True,
        "git_clean_committed": True,
        "run_card_price_runtime_cost_present": True,
        "final_authorized": False,
        "git_branch": branch,
        "git_head": head,
        "package_content_sha256": package_content_sha,
        "package_index_sha256": _sha_file(package / "package_index.json"),
        "cpu_gate_sha256": _sha_file(cpu_gate),
    }
    _atomic_json(package / "authorization.json", authorization)
    return verify_formal_package(package, require_clean_git=True)


def verify_formal_package(
    package: Path,
    *,
    require_clean_git: bool = True,
    allow_existing_output_for_resume: bool = False,
) -> dict[str, Any]:
    package = Path(package).resolve()
    index = _json(package / "package_index.json", "index")
    authorization = _json(package / "authorization.json", "authorization")
    files = index.get("files")
    if not isinstance(files, Mapping):
        raise FormalB2Error("formal package index files are absent")
    for name, descriptor in files.items():
        path = package / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise FormalB2Error(f"formal package file SHA/size mismatch: {name}")
    package_content_sha = canonical_json_sha256(dict(files))
    if not (
        index.get("schema_version") == 1
        and index.get("artifact_kind") == "p5_formal_b2_package_index_v1"
        and index.get("package_version") == "p5_formal_b2_v1"
        and index.get("package_content_sha256") == package_content_sha
        and authorization.get("formal_B2_authorized") is True
        and authorization.get("package_content_sha256") == package_content_sha
        and authorization.get("package_index_sha256")
        == _sha_file(package / "package_index.json")
        and authorization.get("final_authorized") is False
        and all(
            authorization.get(field) is True
            for field in (
                "user_authorized_this_turn",
                "parent_recomputed_from_disk",
                "quick_gate_zero_failed",
                "production_python_bound",
                "schedule_leakage_check_passed",
                "package_sha_passed",
                "output_fresh",
                "disk_at_least_10gb",
                "git_clean_committed",
                "run_card_price_runtime_cost_present",
            )
        )
    ):
        raise FormalB2Error("formal package authorization/index differs")
    config = _json(package / "formal_b2_config.json", "config")
    schedule = _json(package / "prompt_schedule.json", "schedule")
    authority = _json(package / "data_authority.json", "data authority")
    environment = _json(package / "environment.json", "environment")
    validate_production_environment(environment)
    formal_b2_runtime_config(config)
    validate_formal_b2_prompt_schedule(schedule, authority=authority)
    if schedule["schedule_sha256"] != index.get("schedule_semantic_sha256"):
        raise FormalB2Error("formal package schedule semantic SHA differs")
    output = Path(str(config["run"]["output_dir"]))
    if output.is_symlink() or (output.exists() and not allow_existing_output_for_resume):
        raise FormalB2Error("formal training output is not fresh")
    if allow_existing_output_for_resume and not output.is_dir():
        raise FormalB2Error("formal resume output directory is absent")
    free = shutil.disk_usage(output.parent).free
    projected = (
        free
        - CHECKPOINT_OBSERVED_BYTES * FORMAL_CHECKPOINT_COUNT_150
        - TRANSIENT_PEAK_BYTES
    )
    if free < MINIMUM_DISK_FREE_BYTES or projected < MINIMUM_DISK_FREE_BYTES:
        raise FormalB2Error("formal package disk preflight cannot preserve 10 GB")
    if require_clean_git:
        if _git("status", "--porcelain"):
            raise FormalB2Error("formal package preflight requires clean Git")
        if _git("rev-parse", "HEAD") != authorization.get("git_head"):
            raise FormalB2Error("formal package Git HEAD drift")
        if _git("branch", "--show-current") != authorization.get("git_branch"):
            raise FormalB2Error("formal package Git branch drift")
    return {
        "passed": True,
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
        "cuda_model_construction_calls": 0,
        "package_content_sha256": package_content_sha,
        "config_sha256": index["config_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "git_head": authorization["git_head"],
        "output_dir": str(output),
        "disk_free_bytes": free,
        "projected_minimum_free_bytes": projected,
        "formal_B2_authorized": True,
        "final_authorized": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    attest = subparsers.add_parser("attest-parent")
    attest.add_argument("--parent", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--package", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--parent", type=Path, required=True)
    build.add_argument("--parent-package", type=Path, required=True)
    build.add_argument("--cpu-gate", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "attest-parent":
        result = recompute_parent_attestation(args.parent)
        _atomic_json(args.output, result)
    elif args.command == "build":
        result = build_formal_package(
            args.package,
            output=args.output,
            parent=args.parent,
            parent_package=args.parent_package,
            cpu_gate=args.cpu_gate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = verify_formal_package(args.package)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "build_formal_package",
    "production_environment_metadata",
    "recompute_parent_attestation",
    "verify_formal_package",
]
