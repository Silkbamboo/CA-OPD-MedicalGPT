"""CPU-safe preflight for the frozen P4.3 three-policy GPU package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.opd.rollout_probability import (
    RolloutProbabilityError,
    backend_disables_top_k,
    validate_full_support_sampling,
)


class ThreePolicyPreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, root: Path) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else root / candidate


def _assert_sha(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected):
        raise ThreePolicyPreflightError(f"{label} SHA mismatch")


def _ordered_adapter_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = path / name
        if not item.is_file():
            raise ThreePolicyPreflightError(f"Teacher adapter lacks {name}")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _dependency_versions(expected: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution in ("verl", "vllm", "transformers", "peft", "torch"):
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ThreePolicyPreflightError(
                f"required pinned dependency is missing: {distribution}"
            ) from exc
    if observed != {name: str(value) for name, value in expected.items()}:
        raise ThreePolicyPreflightError("pinned dependency version drift")
    return observed


def _assert_no_forbidden_access(config: Mapping[str, Any]) -> None:
    isolation = config.get("isolation", {})
    forbidden = ("final_access", "controller_access", "confirmation_access", "label_access")
    if any(isolation.get(name) is not False for name in forbidden):
        raise ThreePolicyPreflightError("forbidden evaluation/data access is not disabled")
    prompt = config.get("prompt_selection", {})
    attestations = ("contains_labels", "contains_final", "contains_controller", "contains_confirmation")
    if any(prompt.get(name) is not False for name in attestations):
        raise ThreePolicyPreflightError("forbidden prompt-source attestation is not false")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 3
        or protocol.get("protocol_id") != "pg_opd_three_policy_correction_v3"
        or protocol.get("algorithm", {}).get("advantage")
        != "beta_times_stop_grad_teacher_minus_old_actor"
        or protocol.get("algorithm", {}).get("ppo_ratio")
        != "exp_current_actor_minus_old_actor"
        or protocol.get("algorithm", {}).get("rollout_correction")
        != "exp_old_actor_minus_behavior"
        or protocol.get("correction", {}).get("level") != "token"
        or protocol.get("correction", {}).get("upper_threshold") != 2.0
        or protocol.get("correction", {}).get("upper_truncation") is not True
        or protocol.get("correction", {}).get("lower_truncation") is not False
        or protocol.get("correction", {}).get("batch_normalize") is not False
        or protocol.get("correction", {}).get("rejection_sampling") is not False
        or protocol.get("correction", {}).get("sequence_product") is not False
        or protocol.get("reduction", {}).get("denominator")
        != "valid_token_count_not_sum_weights"
        or protocol.get("optimizer", {}).get("backward_mode")
        != "per_prompt_streamed_exact_prompt_equal_mean"
        or protocol.get("optimizer", {}).get("backward_scale")
        != "one_over_unique_prompt_count"
        or protocol.get("optimizer", {}).get("retained_model_graphs_max") != 1
    ):
        raise ThreePolicyPreflightError("frozen three-policy protocol drift")


def preflight(
    config: Mapping[str, Any],
    *,
    execute_gpu: bool,
    require_clean_git: bool = True,
    repo_root: str | Path | None = None,
    allow_empty_launcher_stdout_envelope: bool = False,
) -> dict[str, Any]:
    """Validate immutable inputs and config without importing model runtimes."""

    root = Path(repo_root or Path(__file__).resolve().parents[2])
    run = config.get("run", {})
    execution = config.get("execution", {})
    if (
        config.get("schema_version") != 3
        or run.get("stage") != "three_policy_revalidation"
        or run.get("calibration_only") is not True
        or run.get("formal_opd_training") is not False
        or run.get("one_step_only") is not True
        or run.get("evidence_driven_fix_retries_max") != 0
        or execution.get("stop_on_first_failure") is not True
        or any(
            execution.get(name) is not False
            for name in (
                "automatically_start_b2",
                "automatically_run_controller",
                "automatically_run_confirmation",
                "automatically_run_final",
                "automatically_run_sft",
                "automatically_run_teacher_evaluation",
                "automatically_run_full_p4_1_scorer",
                "automatically_run_full_vllm_diagnostic",
            )
        )
        or config.get("optimizer", {}).get("backward_mode")
        != "per_prompt_streamed_exact_prompt_equal_mean"
        or config.get("optimizer", {}).get("backward_scale")
        != "one_over_unique_prompt_count"
        or config.get("optimizer", {}).get("retained_model_graphs_max") != 1
    ):
        raise ThreePolicyPreflightError("P4.3 run boundary drift")
    _assert_no_forbidden_access(config)

    validation = config.get("validation", {})
    protocol_path = _path(validation.get("config_path", ""), root)
    _assert_sha(protocol_path, validation.get("config_sha256"), "protocol config")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    schema = config.get("artifacts", {})
    schema_path = _path(schema.get("schema_path", ""), root)
    _assert_sha(schema_path, schema.get("schema_sha256"), "artifact schema")
    try:
        artifact_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ThreePolicyPreflightError("artifact schema is invalid JSON") from exc
    if artifact_schema.get("type") != "object" or "rollout_correction" not in artifact_schema.get("required", []):
        raise ThreePolicyPreflightError("artifact schema lacks correction contract")

    formal = config.get("formal_rollout", {})
    try:
        support = validate_full_support_sampling(
            "transformers", formal.get("transformers", {})
        )
    except RolloutProbabilityError as exc:
        raise ThreePolicyPreflightError(str(exc)) from exc
    vllm = formal.get("vllm_adapter_contract", {})
    if (
        vllm.get("version") != "0.11.0"
        or not backend_disables_top_k("vllm", vllm.get("disabled_top_k_formal_value"))
        or vllm.get("temperature") != 1.0
        or vllm.get("top_p") != 1.0
        or vllm.get("min_p") != 0.0
    ):
        raise ThreePolicyPreflightError("vLLM adapter full-support contract drift")

    prompt = config.get("prompt_selection", {})
    _assert_sha(_path(prompt.get("opd_manifest_path", ""), root), prompt.get("opd_manifest_sha256"), "OPD manifest")
    _assert_sha(Path(str(prompt.get("medical_opd_o1_path", ""))), prompt.get("medical_opd_o1_sha256"), "Medical-O1 prompt source")
    _assert_sha(Path(str(prompt.get("medical_opd_cmb_path", ""))), prompt.get("medical_opd_cmb_sha256"), "CMB prompt source")
    if (
        prompt.get("stages") != [4, 16, 32]
        or prompt.get("default_stop_after") != 16
        or prompt.get("equal_source_split") is not True
    ):
        raise ThreePolicyPreflightError("prompt calibration ladder drift")

    model = config.get("model", {})
    teacher = config.get("teacher", {})
    for path_value, sha_value, label in (
        (model.get("artifact_manifest_path"), model.get("artifact_manifest_sha256"), "model manifest"),
        (teacher.get("route_config"), teacher.get("route_config_sha256"), "Teacher route config"),
        (teacher.get("manifest_path"), teacher.get("manifest_sha256"), "Teacher manifest"),
    ):
        _assert_sha(_path(path_value, root), sha_value, label)
    model_manifest = json.loads(
        _path(model.get("artifact_manifest_path", ""), root).read_text(encoding="utf-8")
    )
    if model_manifest.get("tokenizer_revision") != model.get("tokenizer_revision"):
        raise ThreePolicyPreflightError("Base/tokenizer manifest identity drift")
    teacher_manifest = json.loads(
        _path(teacher.get("manifest_path", ""), root).read_text(encoding="utf-8")
    )
    adapter_path = Path(str(teacher.get("adapter_path", "")))
    _assert_sha(
        adapter_path / "adapter_model.safetensors",
        teacher.get("adapter_weight_sha256"),
        "Teacher adapter weight",
    )
    if (
        _ordered_adapter_sha(adapter_path) != teacher.get("adapter_sha256")
        or str(adapter_path.resolve())
        != str(Path(str(teacher_manifest.get("adapter_path", ""))).resolve())
        or teacher_manifest.get("adapter_sha256") != teacher.get("adapter_sha256")
        or teacher_manifest.get("adapter_weight_sha256")
        != teacher.get("adapter_weight_sha256")
        or teacher_manifest.get("base_model_revision") != model.get("revision")
        or teacher_manifest.get("tokenizer_revision") != model.get("tokenizer_revision")
        or teacher_manifest.get("status") != "teacher_frozen_confirmed"
        or teacher_manifest.get("teacher_knowledge_ready") is not True
        or teacher_manifest.get("OPD_scoring_backend_ready") is not False
        or teacher_manifest.get("final_authorized") is not False
    ):
        raise ThreePolicyPreflightError("Teacher/model identity drift")

    dependency_versions = _dependency_versions(config.get("versions", {}))

    historical = config.get("historical", {})
    if historical.get("p4_2_status") != "failed_identity_mismatch" or historical.get("p4_1_use") != "forensic_only":
        raise ThreePolicyPreflightError("P4.1/P4.2 historical status drift")
    protected = Path("artifacts/outputs/qwen3-4b-pg-direction-revalidation-seed42")
    for filename, key in (
        ("three_policy_identity.json", "p4_2_three_policy_metrics_sha256"),
        ("failure.json", "p4_2_failure_sha256"),
        ("artifact_index.json", "p4_2_artifact_index_sha256"),
        ("readiness.json", "p4_2_readiness_sha256"),
        ("resource_cleanup.json", "p4_2_cleanup_sha256"),
    ):
        _assert_sha(protected / filename, historical.get(key), f"protected P4.2 {filename}")
    _assert_sha(
        Path(str(historical.get("p4_1_trajectory_path", ""))),
        historical.get("p4_1_trajectory_sha256"),
        "protected P4.1 trajectory",
    )

    output = Path(str(run.get("output_dir", "")))
    if output.exists():
        existing = list(output.iterdir())
        launcher_stdout_only = bool(
            allow_empty_launcher_stdout_envelope
            and len(existing) == 1
            and existing[0].name == "stdout.log"
            and existing[0].is_file()
            and existing[0].stat().st_size == 0
        )
        if existing and not launcher_stdout_only:
            raise ThreePolicyPreflightError("P4.3 output directory must be new/empty")
    run_card = root / "configs/run_cards" / f"{run.get('run_id')}.json"
    if run_card.is_file():
        card = json.loads(run_card.read_text(encoding="utf-8"))
        configured = root / "configs/opd/qwen3_4b_three_policy_revalidation_v3.yaml"
        if card.get("config_sha256") != _sha256(configured):
            raise ThreePolicyPreflightError("run card/config SHA mismatch")
    disk_parent = output.parent
    if not disk_parent.is_dir():
        raise ThreePolicyPreflightError("P4.3 output parent is missing")
    free_disk_gib = shutil.disk_usage(disk_parent).free / float(1024**3)
    if free_disk_gib < float(config.get("resources", {}).get("minimum_free_disk_gib", 0)):
        raise ThreePolicyPreflightError("insufficient free disk for P4.3 revalidation")
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
        ).stdout
        if status:
            raise ThreePolicyPreflightError("GPU revalidation requires a clean committed worktree")
    gpu_inventory: list[dict[str, Any]] = []
    if execute_gpu:
        if os.environ.get("CA_OPD_ALLOW_THREE_POLICY_REVALIDATION_GPU") != "1":
            raise ThreePolicyPreflightError("GPU three-policy revalidation lacks explicit authorization")
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in query.stdout.splitlines():
            if not line.strip():
                continue
            index, name, uuid, total, used = [value.strip() for value in line.split(",", 4)]
            gpu_inventory.append(
                {
                    "index": int(index),
                    "name": name,
                    "uuid": uuid,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                }
            )
        if (
            len(gpu_inventory)
            != int(config.get("resources", {}).get("required_gpus", 0))
            or sorted(item["index"] for item in gpu_inventory) != [0, 1]
            or any(
                "RTX 3090" not in item["name"]
                or item["memory_total_mib"] < 24576
                or item["memory_used_mib"] != 0
                for item in gpu_inventory
            )
        ):
            raise ThreePolicyPreflightError("idle dual-RTX-3090 topology mismatch")
        gpu_processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if gpu_processes:
            raise ThreePolicyPreflightError("unexpected GPU compute process detected")
        process_lines = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        markers = ("vllm", "raylet", "ray::", "verl_worker")
        if any(any(marker in line.lower() for marker in markers) for line in process_lines):
            raise ThreePolicyPreflightError("stale vLLM/Ray/veRL worker detected")
    return {
        "status": "ready_waiting_for_gpu_three_policy_revalidation",
        "gpu_used": False,
        "loaded_real_model": False,
        "p4_2_status_preserved": "failed_identity_mismatch",
        "formal_sampling": support.support_classification,
        "protocol_config_sha256": validation["config_sha256"],
        "artifact_schema_sha256": schema["schema_sha256"],
        "gpu_inventory": gpu_inventory,
        "dependency_versions": dependency_versions,
        "free_disk_gib": free_disk_gib,
        "B2_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.3 three-policy preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    parser.add_argument("--output-dir-override")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir_override:
        config["run"]["output_dir"] = args.output_dir_override
    report = preflight(
        config,
        execute_gpu=args.execute_gpu,
        require_clean_git=not args.allow_dirty_for_development,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
