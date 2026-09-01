"""CPU-safe preflight for the future P4.5 production refresh GPU micro-smoke."""

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

from src.opd.production_backend_binding_v5 import verify_b2_backend_binding


class ProductionSamplerMicroPreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _assert_sha(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ProductionSamplerMicroPreflightError(f"{label} SHA mismatch")


def _adapter_file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = path / name
        if not item.is_file():
            raise ProductionSamplerMicroPreflightError(f"adapter lacks {name}")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_card_and_sources(
    config: Mapping[str, Any], *, config_path: Path, root: Path
) -> dict[str, Any]:
    card_path = root / "configs/run_cards" / f"{config['run']['run_id']}.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if (
        card.get("config_path") != str(config_path.relative_to(root))
        or card.get("config_sha256") != _sha256(config_path)
    ):
        raise ProductionSamplerMicroPreflightError("P4.5 run card/config SHA mismatch")
    schema_path = _resolve(root, config["artifacts"]["schema_path"])
    _assert_sha(schema_path, config["artifacts"]["schema_sha256"], "P4.5 artifact schema")
    protocol_path = _resolve(root, config["validation"]["config_path"])
    _assert_sha(protocol_path, config["validation"]["config_sha256"], "three-policy protocol")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("algorithm", {}).get("advantage")
        != "beta_times_stop_grad_teacher_minus_old_actor"
        or protocol.get("algorithm", {}).get("ppo_ratio")
        != "exp_current_actor_minus_old_actor"
        or protocol.get("algorithm", {}).get("rollout_correction")
        != "exp_old_actor_minus_behavior"
        or protocol.get("correction", {}).get("upper_threshold") != 2.0
        or protocol.get("correction", {}).get("stop_gradient") is not True
    ):
        raise ProductionSamplerMicroPreflightError("three-policy math drift")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema.get("required", []))
    if not {
        "production_backend_binding",
        "authoritative_manifest_sha256",
        "trainer_identity",
        "runtime_identity",
        "fresh_identity",
        "normal_request",
        "stale_request",
        "registry_before",
        "registry_after",
    } <= required:
        raise ProductionSamplerMicroPreflightError("P4.5 artifact schema is incomplete")

    binding = config["production_binding"]
    try:
        verified = verify_b2_backend_binding(
            binding["b2_config_path"], binding["b2_run_card_path"], repo_root=root
        )
    except Exception as error:
        raise ProductionSamplerMicroPreflightError(
            f"production backend binding failed: {error}"
        ) from error
    if (
        verified["production_backend"]["backend_id"] != binding["backend_id"]
        or verified["config_sha256"] != binding["b2_config_sha256"]
        or verified["run_card_sha256"] != binding["b2_run_card_sha256"]
    ):
        raise ProductionSamplerMicroPreflightError("production backend binding SHA mismatch")
    return {
        "config_sha256": _sha256(config_path),
        "run_card_sha256": _sha256(card_path),
        "artifact_schema_sha256": _sha256(schema_path),
        "three_policy_protocol_sha256": _sha256(protocol_path),
        "production_backend_binding_verified": True,
        "b2_config_sha256": verified["config_sha256"],
        "b2_run_card_sha256": verified["run_card_sha256"],
    }


def _validate_models_and_prompts(config: Mapping[str, Any], root: Path) -> None:
    model = config["model"]
    teacher = config["teacher"]
    _assert_sha(
        _resolve(root, model["artifact_manifest_path"]),
        model["artifact_manifest_sha256"],
        "Base manifest",
    )
    _assert_sha(
        _resolve(root, teacher["route_config"]),
        teacher["route_config_sha256"],
        "Teacher route",
    )
    _assert_sha(
        _resolve(root, teacher["manifest_path"]),
        teacher["manifest_sha256"],
        "Teacher manifest",
    )
    adapter = _resolve(root, teacher["adapter_path"])
    _assert_sha(
        adapter / "adapter_model.safetensors",
        teacher["adapter_weight_sha256"],
        "Teacher adapter weight",
    )
    if _adapter_file_sha(adapter) != teacher["adapter_sha256"]:
        raise ProductionSamplerMicroPreflightError("Teacher ordered adapter SHA mismatch")
    prompts = config["prompt_selection"]
    for field, sha_field in (
        ("opd_manifest_path", "opd_manifest_sha256"),
        ("medical_opd_o1_path", "medical_opd_o1_sha256"),
        ("medical_opd_cmb_path", "medical_opd_cmb_sha256"),
    ):
        _assert_sha(_resolve(root, prompts[field]), prompts[sha_field], field)


def _validate_history(config: Mapping[str, Any]) -> None:
    history = config["historical_protection"]
    root = Path(history["p4_4_output_dir"])
    for name, expected in history["p4_4_artifacts"].items():
        _assert_sha(root / name, expected, f"historical P4.4 {name}")
    existing_adapters = list(root.rglob("adapter_model.safetensors"))
    if existing_adapters or config["historical_v1"]["p4_4_temporary_adapter_present"] is not False:
        raise ProductionSamplerMicroPreflightError(
            "historical v1 presence differs from the frozen regeneration decision"
        )


def preflight(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    execute_gpu: bool,
    require_clean_git: bool = True,
    repo_root: str | Path | None = None,
    allow_launcher_stdout_envelope: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    path = Path(config_path)
    path = path if path.is_absolute() else root / path
    if (
        config.get("schema_version") != 5
        or config.get("run", {}).get("stage") != "production_sampler_refresh_micro_v5"
        or config.get("run", {}).get("formal_opd_training") is not False
        or config.get("run", {}).get("automatically_start_b2") is not False
        or config.get("sampler_refresh", {}).get("candidate_mechanism")
        != "peft_0_17_1_hotswap_stable_slot"
        or config.get("sampler_refresh", {}).get("runtime_slot") != "student_active"
        or config.get("sampler_refresh", {}).get("sampler_self_report_may_be_authority")
        is not False
    ):
        raise ProductionSamplerMicroPreflightError("P4.5 execution boundary drift")
    isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if config.get("isolation") != isolation:
        raise ProductionSamplerMicroPreflightError("P4.5 isolation gate failed")
    if any(
        config.get("prompt_selection", {}).get(field) is not False
        for field in ("contains_labels", "contains_final", "contains_controller", "contains_confirmation")
    ):
        raise ProductionSamplerMicroPreflightError("P4.5 prompt isolation gate failed")
    historical = config.get("historical_v1", {})
    if not (
        historical.get("p4_4_temporary_adapter_present") is False
        and historical.get("reuse_existing_v1") is False
        and historical.get("regenerate_with_minimal_four_prompt_one_step") is True
        and historical.get("rerun_16_prompts") is False
        and historical.get("rerun_32_prompts") is False
        and historical.get("run_base_null") is False
        and config.get("prompt_selection", {}).get("prompts") == 4
    ):
        raise ProductionSamplerMicroPreflightError("minimal v1 regeneration contract drift")

    identities = _validate_card_and_sources(config, config_path=path, root=root)
    _validate_history(config)
    _validate_models_and_prompts(config, root)
    output = Path(config["run"]["output_dir"])
    if output.resolve() == Path(config["historical_protection"]["p4_4_output_dir"]).resolve():
        raise ProductionSamplerMicroPreflightError("historical P4.4 output cannot be reused")
    if output.exists():
        existing = {str(item.relative_to(output)) for item in output.rglob("*") if item.is_file()}
        launcher_envelope = bool(
            allow_launcher_stdout_envelope
            and existing == {"stdout.log"}
            and (output / "stdout.log").stat().st_size == 0
        )
        if not launcher_envelope:
            raise ProductionSamplerMicroPreflightError("P4.5 output must be fresh and absent")
    if not output.parent.is_dir():
        raise ProductionSamplerMicroPreflightError("P4.5 output parent is missing")
    free_disk_gib = shutil.disk_usage(output.parent).free / float(1024**3)
    if free_disk_gib < float(config["resources"]["minimum_free_disk_gib"]):
        raise ProductionSamplerMicroPreflightError("insufficient persistent disk")

    dependency_versions: dict[str, str] = {}
    for distribution, expected in config["versions"].items():
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = "not_installed"
        if execute_gpu and dependency_versions[distribution] != str(expected):
            raise ProductionSamplerMicroPreflightError(f"pinned dependency drift: {distribution}")
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
        ).stdout
        if status:
            raise ProductionSamplerMicroPreflightError("GPU run requires a clean worktree")

    gpu_inventory: list[dict[str, Any]] = []
    if execute_gpu:
        authorization = config["authorization"]
        if os.environ.get(authorization["environment_variable"]) != authorization["required_value"]:
            raise ProductionSamplerMicroPreflightError("GPU authorization is absent")
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in query.stdout.splitlines():
            index, name, total, used = [item.strip() for item in line.split(",")]
            gpu_inventory.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                }
            )
        if (
            len(gpu_inventory) != 2
            or any(
                "RTX 3090" not in item["name"]
                or item["memory_total_mib"] < 24576
                or item["memory_used_mib"] != 0
                for item in gpu_inventory
            )
        ):
            raise ProductionSamplerMicroPreflightError("idle dual RTX 3090 gate failed")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if processes:
            raise ProductionSamplerMicroPreflightError("unknown GPU process detected")
    return {
        "status": "ready_waiting_for_gpu_production_sampler_refresh",
        "gpu_used": False,
        "loaded_real_model": False,
        "existing_v1_adapter_present": False,
        "regenerate_v1_with_four_prompts": True,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "free_disk_gib": free_disk_gib,
        "dependency_versions": dependency_versions,
        "gpu_inventory": gpu_inventory,
        **identities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.5 production sampler micro preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    parser.add_argument("--output-dir-override")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.output_dir_override:
        config["run"]["output_dir"] = args.output_dir_override
    result = preflight(
        config,
        config_path=path,
        execute_gpu=args.execute_gpu,
        require_clean_git=not args.allow_dirty_for_development,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
