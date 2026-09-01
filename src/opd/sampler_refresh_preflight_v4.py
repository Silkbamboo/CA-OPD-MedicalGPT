"""CPU-safe preflight for the fresh P4.4 sampler-refresh GPU package."""

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


class SamplerRefreshPreflightError(RuntimeError):
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
        raise SamplerRefreshPreflightError(f"{label} SHA mismatch")


def _adapter_file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = path / name
        if not item.is_file():
            raise SamplerRefreshPreflightError(f"adapter lacks {name}")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_bound_files(
    config: Mapping[str, Any], *, config_path: Path, root: Path
) -> dict[str, str]:
    card_path = root / "configs/run_cards" / f"{config['run']['run_id']}.json"
    if not card_path.is_file():
        raise SamplerRefreshPreflightError("P4.4 run card is missing")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if (
        card.get("config_path") != str(config_path.relative_to(root))
        or card.get("config_sha256") != _sha256(config_path)
    ):
        raise SamplerRefreshPreflightError("P4.4 run card/config SHA mismatch")
    for section, path_key, sha_key, label in (
        ("three_policy_protocol", "path", "sha256", "P4.3 three-policy protocol"),
        ("sampler_refresh_contract", "path", "sha256", "P4.4 sampler contract"),
        ("artifacts", "schema_path", "schema_sha256", "P4.4 artifact schema"),
    ):
        value = config[section]
        path = _resolve(root, value[path_key])
        _assert_sha(path, value[sha_key], label)
    protocol = yaml.safe_load(
        _resolve(root, config["three_policy_protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        protocol.get("protocol_id") != "pg_opd_three_policy_correction_v3"
        or protocol.get("algorithm", {}).get("advantage")
        != "beta_times_stop_grad_teacher_minus_old_actor"
        or protocol.get("algorithm", {}).get("ppo_ratio")
        != "exp_current_actor_minus_old_actor"
        or protocol.get("algorithm", {}).get("rollout_correction")
        != "exp_old_actor_minus_behavior"
        or protocol.get("correction", {}).get("upper_threshold") != 2.0
        or protocol.get("correction", {}).get("stop_gradient") is not True
        or protocol.get("reduction", {}).get("denominator")
        != "valid_token_count_not_sum_weights"
    ):
        raise SamplerRefreshPreflightError("P4.3 three-policy math drift")
    contract = yaml.safe_load(
        _resolve(root, config["sampler_refresh_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        contract.get("contract_id") != "sampler_refresh_same_path_contract_v4"
        or contract.get("same_path_gates", {}).get(
            "trainer_in_memory_vs_fresh_reload_max_abs"
        )
        != 1e-4
        or contract.get("same_path_gates", {}).get(
            "live_refreshed_vs_fresh_sampler_max_abs"
        )
        != 1e-4
        or contract.get("cross_path_diagnostics", {}).get("hard_gate") is not False
    ):
        raise SamplerRefreshPreflightError("P4.4 sampler contract drift")
    schema = json.loads(
        _resolve(root, config["artifacts"]["schema_path"]).read_text(
            encoding="utf-8"
        )
    )
    if schema.get("type") != "object" or "stale_request_test" not in schema.get(
        "required", []
    ):
        raise SamplerRefreshPreflightError("P4.4 artifact schema is incomplete")
    return {
        "config_sha256": _sha256(config_path),
        "run_card_sha256": _sha256(card_path),
        "three_policy_protocol_sha256": config["three_policy_protocol"]["sha256"],
        "sampler_refresh_contract_sha256": config["sampler_refresh_contract"]["sha256"],
        "artifact_schema_sha256": config["artifacts"]["schema_sha256"],
    }


def _validate_historical(config: Mapping[str, Any]) -> None:
    historical = config["historical"]
    if (
        historical.get("p4_2_status") != "failed_identity_mismatch"
        or historical.get("p4_3_status") != "failed_sampler_refresh"
    ):
        raise SamplerRefreshPreflightError("historical failure status drift")
    _assert_sha(
        Path(historical["p4_1_trajectory_path"]),
        historical["p4_1_trajectory_sha256"],
        "protected P4.1 trajectory",
    )
    roots = {
        "p4_2_artifacts": Path(
            "artifacts/outputs/qwen3-4b-pg-direction-revalidation-seed42"
        ),
        "p4_3_artifacts": Path(
            "artifacts/outputs/qwen3-4b-three-policy-revalidation-seed42"
        ),
    }
    for section, base in roots.items():
        for name, expected in historical[section].items():
            _assert_sha(base / name, expected, f"protected {section} {name}")


def _validate_model_manifests(config: Mapping[str, Any], *, root: Path) -> None:
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
    adapter = Path(teacher["adapter_path"])
    _assert_sha(
        adapter / "adapter_model.safetensors",
        teacher["adapter_weight_sha256"],
        "Teacher adapter weight",
    )
    if _adapter_file_sha(adapter) != teacher["adapter_sha256"]:
        raise SamplerRefreshPreflightError("Teacher ordered adapter SHA mismatch")


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
    if not path.is_absolute():
        path = root / path
    if (
        config.get("schema_version") != 4
        or config.get("run", {}).get("stage") != "sampler_refresh_revalidation_v4"
        or config.get("run", {}).get("formal_opd_training") is not False
        or config.get("execution", {}).get("automatically_start_b2") is not False
        or any(
            config.get("execution", {}).get(name) is not False
            for name in (
                "automatically_run_idt",
                "automatically_run_sar",
                "automatically_run_ca_opd",
                "automatically_run_controller",
                "automatically_run_confirmation",
                "automatically_run_final",
                "automatically_run_sft",
                "automatically_run_teacher_training",
            )
        )
    ):
        raise SamplerRefreshPreflightError("P4.4 execution boundary drift")
    expected_isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if config.get("isolation") != expected_isolation:
        raise SamplerRefreshPreflightError("forbidden evaluation or label access")
    if any(
        config.get("prompt_selection", {}).get(field) is not False
        for field in (
            "contains_labels",
            "contains_final",
            "contains_controller",
            "contains_confirmation",
        )
    ):
        raise SamplerRefreshPreflightError("forbidden prompt-source access")
    optional_32 = config.get("prompt_selection", {}).get("optional_32_gate", {})
    if not (
        optional_32.get("scopes") == ["per_prompt", "per_source"]
        and optional_32.get("ess_fraction_min")
        == config.get("correction", {}).get("ess_fraction_min")
        and optional_32.get("cap_fraction_max")
        == config.get("correction", {}).get("cap_fraction_max")
        and optional_32.get("threshold_source") == "unchanged_correction_gates"
    ):
        raise SamplerRefreshPreflightError("optional 32 ESS/cap gate drift")

    identities = _validate_bound_files(config, config_path=path, root=root)
    _validate_historical(config)
    _validate_model_manifests(config, root=root)
    for field, sha_field in (
        ("opd_manifest_path", "opd_manifest_sha256"),
        ("medical_opd_o1_path", "medical_opd_o1_sha256"),
        ("medical_opd_cmb_path", "medical_opd_cmb_sha256"),
    ):
        _assert_sha(
            _resolve(root, config["prompt_selection"][field]),
            config["prompt_selection"][sha_field],
            field,
        )

    output = Path(config["run"]["output_dir"])
    protected = {
        Path(value).resolve() for value in config["historical"]["protected_output_dirs"]
    }
    if output.resolve() in protected:
        raise SamplerRefreshPreflightError("historical protected output cannot be reused")
    if output.exists():
        existing = {
            str(item.relative_to(output))
            for item in output.rglob("*")
            if item.is_file()
        }
        launcher_envelope = bool(
            allow_launcher_stdout_envelope
            and existing == {"stdout.log"}
            and (output / "stdout.log").stat().st_size == 0
        )
        if not launcher_envelope:
            raise SamplerRefreshPreflightError("P4.4 output must be fresh and absent")
    if not output.parent.is_dir():
        raise SamplerRefreshPreflightError("P4.4 output parent is missing")
    free_disk_gib = shutil.disk_usage(output.parent).free / float(1024**3)
    if free_disk_gib < float(config["resources"]["minimum_free_disk_gib"]):
        raise SamplerRefreshPreflightError("insufficient persistent disk")

    dependency_versions: dict[str, str] = {}
    for distribution, expected in config["versions"].items():
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = "not_verified_in_cpu_dry_run"
        if execute_gpu and dependency_versions[distribution] != str(expected):
            raise SamplerRefreshPreflightError(
                f"pinned dependency drift: {distribution}"
            )
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if status:
            raise SamplerRefreshPreflightError("GPU run requires a clean worktree")

    gpu_inventory: list[dict[str, Any]] = []
    if execute_gpu:
        name = config["authorization"]["environment_variable"]
        if os.environ.get(name) != config["authorization"]["required_value"]:
            raise SamplerRefreshPreflightError("GPU authorization is absent")
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
            index, gpu_name, total, used = [item.strip() for item in line.split(",")]
            gpu_inventory.append(
                {
                    "index": int(index),
                    "name": gpu_name,
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
            raise SamplerRefreshPreflightError("idle dual RTX 3090 gate failed")
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if processes:
            raise SamplerRefreshPreflightError("unknown GPU process detected")
    return {
        "status": "ready_waiting_for_gpu_sampler_refresh_revalidation",
        "gpu_used": False,
        "loaded_real_model": False,
        "B2_authorized": False,
        "p4_3_status_preserved": "failed_sampler_refresh",
        "free_disk_gib": free_disk_gib,
        "dependency_versions": dependency_versions,
        "gpu_inventory": gpu_inventory,
        **identities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.4 sampler refresh preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    parser.add_argument("--output-dir-override")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.output_dir_override:
        config["run"]["output_dir"] = args.output_dir_override
    report = preflight(
        config,
        config_path=path,
        execute_gpu=args.execute_gpu,
        require_clean_git=not args.allow_dirty_for_development,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
