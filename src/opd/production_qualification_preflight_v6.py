"""CPU-safe preflight for the P4.6 combined production qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import yaml

from src.opd.production_backend_binding_v5 import verify_b2_backend_binding
from src.opd.production_qualification_prompts_v6 import (
    validate_frozen_prompt_selection,
)


SCHEMA_ID = "ca-opd/p4.6-combined-production-qualification/v1"
RUN_ID = "qwen3-4b-production-qualification-v6-seed42"
CARD_FIELDS = {
    "schema_id",
    "schema_version",
    "run_id",
    "stage",
    "status",
    "config_path",
    "config_sha256",
    "artifact_schema_path",
    "artifact_schema_sha256",
    "production_backend_id",
    "refresh_implementation",
    "runtime_slot",
    "prompt_selection_manifest_path",
    "prompt_selection_manifest_sha256",
    "probe_spec_sha256",
    "phase_groups",
    "retry_lineage",
    "gpu_authorization_environment_variable",
    "gpu_execution_now",
    "automatically_generate_calibration_after_full_readiness",
    "automatically_start_b2",
    "automatically_run_idt_sar_ca_opd",
    "automatically_access_controller_confirmation_final",
    "production_sampler_refresh_ready_now",
    "OPD_scoring_backend_ready_now",
    "B2_authorized_now",
    "next_state",
}


class ProductionQualificationPreflightError(RuntimeError):
    """A checked-in identity or execution boundary does not match P4.6."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_adapter_sha256(path: str | Path) -> str:
    """Bind the PEFT config and weight bytes in the frozen transport order."""

    directory = Path(path)
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = directory / name
        if not item.is_file():
            raise ProductionQualificationPreflightError(
                f"Teacher adapter lacks {name}"
            )
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_base_model_transport(
    *,
    base_path: str | Path,
    weights_manifest_path: str | Path,
    weights_manifest_sha256: str,
    expected_revision: str,
    verify_weight_payloads: bool = True,
) -> dict[str, Any]:
    """Verify the immutable Qwen transport without loading model tensors.

    The lightweight ``artifact_manifest.json`` beside the model only binds
    tokenizer/config metadata.  This gate instead reopens the download
    manifest, the safetensors index, and every referenced weight shard.
    """

    base = Path(base_path).resolve()
    manifest_path = Path(weights_manifest_path)
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or not isinstance(weights_manifest_sha256, str)
        or sha256_file(manifest_path) != weights_manifest_sha256
    ):
        raise ProductionQualificationPreflightError(
            "Base weights manifest SHA mismatch"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductionQualificationPreflightError(
            f"Base weights manifest is invalid: {error}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ProductionQualificationPreflightError(
            "Base weights manifest is not an object"
        )
    if (
        str(manifest.get("resolved_revision", "")) != expected_revision
        or Path(str(manifest.get("local_persistent_path", ""))).resolve() != base
    ):
        raise ProductionQualificationPreflightError(
            "Base weights manifest revision/path mismatch"
        )
    shards = manifest.get("index_referenced_shards")
    files = manifest.get("files")
    if (
        not isinstance(shards, list)
        or len(shards) != 3
        or len(set(shards)) != 3
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in shards
        )
        or not isinstance(files, list)
    ):
        raise ProductionQualificationPreflightError(
            "Base weights manifest shard inventory mismatch"
        )
    inventory: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ProductionQualificationPreflightError(
                "Base weights manifest file inventory is invalid"
            )
        relative = str(item["path"])
        if relative in inventory:
            raise ProductionQualificationPreflightError(
                "Base weights manifest contains duplicate paths"
            )
        inventory[relative] = item

    index_name = "model.safetensors.index.json"
    required = [index_name, *shards]
    for name in required:
        item = inventory.get(name)
        expected_path = base / name
        label = "Base weight index" if name == index_name else "Base weight shard"
        if (
            item is None
            or expected_path.is_symlink()
            or not expected_path.is_file()
            or Path(str(item.get("local_path", ""))).resolve() != expected_path
            or not isinstance(item.get("size"), int)
            or expected_path.stat().st_size != item["size"]
            or not isinstance(item.get("sha256"), str)
            or (
                (verify_weight_payloads or name == index_name)
                and sha256_file(expected_path) != item["sha256"]
            )
        ):
            raise ProductionQualificationPreflightError(f"{label} SHA mismatch")
    try:
        index = json.loads((base / index_name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductionQualificationPreflightError(
            f"Base weight index is invalid: {error}"
        ) from error
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if (
        not isinstance(weight_map, Mapping)
        or not weight_map
        or set(weight_map.values()) != set(shards)
    ):
        raise ProductionQualificationPreflightError(
            "Base weight index/shard inventory mismatch"
        )
    return {
        "model_id": str(manifest.get("model_id", "")),
        "revision": expected_revision,
        "weights_manifest_sha256": weights_manifest_sha256,
        "index_sha256": str(inventory[index_name]["sha256"]),
        "shard_count": len(shards),
        "shard_sha256": [str(inventory[name]["sha256"]) for name in shards],
        "weight_payloads_verified": verify_weight_payloads,
    }


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _assert_file_sha(root: Path, path: Any, expected: Any, label: str) -> Path:
    resolved = _resolve(root, path)
    if (
        not resolved.is_file()
        or not isinstance(expected, str)
        or sha256_file(resolved) != expected
    ):
        raise ProductionQualificationPreflightError(f"{label} SHA mismatch")
    return resolved


def _strict_card(config: Mapping[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    card_path = root / "configs/run_cards" / f"{config['run']['run_id']}.json"
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductionQualificationPreflightError(f"cannot read P4.6 run card: {error}") from error
    if not isinstance(card, Mapping) or set(card) != CARD_FIELDS:
        raise ProductionQualificationPreflightError("P4.6 run-card fields are not exact")
    if (
        card["schema_id"] != SCHEMA_ID
        or card["schema_version"] != 1
        or card["run_id"] != RUN_ID
        or card["stage"] != "combined_production_qualification"
        or card["status"] != "prepared_cpu_only_not_started"
        or card["config_path"] != str(config_path.relative_to(root))
        or card["config_sha256"] != sha256_file(config_path)
        or card["gpu_execution_now"] != "not_run_cpu_only"
        or card["next_state"] != "ready_waiting_for_gpu_combined_qualification"
    ):
        raise ProductionQualificationPreflightError("P4.6 run-card identity drift")
    for field in (
        "automatically_start_b2",
        "automatically_run_idt_sar_ca_opd",
        "automatically_access_controller_confirmation_final",
        "production_sampler_refresh_ready_now",
        "OPD_scoring_backend_ready_now",
        "B2_authorized_now",
    ):
        if card[field] is not False:
            raise ProductionQualificationPreflightError(f"P4.6 run-card is not fail-closed: {field}")
    if card["automatically_generate_calibration_after_full_readiness"] is not True:
        raise ProductionQualificationPreflightError("P4.6 calibration materialization contract drift")
    return dict(card)


def _validate_contract(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_id") != SCHEMA_ID
        or config.get("schema_version") != 1
        or config.get("run", {}).get("run_id") != RUN_ID
        or config.get("run", {}).get("stage") != "combined_production_qualification"
        or config.get("run", {}).get("formal_opd_training") is not False
        or config.get("run", {}).get("automatically_start_b2") is not False
        or config.get("execution", {}).get("fail_stop") is not True
        or config.get("production_binding", {}).get("backend_id")
        != "custom_transformers_peft_three_policy_v5"
        or config.get("sampler_refresh", {}).get("candidate_mechanism")
        != "peft_0_17_1_hotswap_stable_slot"
        or config.get("sampler_refresh", {}).get("runtime_slot") != "student_active"
        or config.get("sampler_refresh", {}).get("sampler_self_report_may_be_authority")
        is not False
        or config.get("fixed_action_probe", {}).get("selection_rule")
        != "first_32_valid_response_tokens_per_prompt_v1"
        or config.get("fixed_action_probe", {}).get("per_prompt_limit") != 32
    ):
        raise ProductionQualificationPreflightError("P4.6 qualification contract drift")
    expected_groups = [
        "micro_evidence_replay",
        "production_two_step",
        "base_teacher_null",
        "length_calibration",
        "b2_authorization_finalizer",
    ]
    if config.get("execution", {}).get("phase_groups") != expected_groups:
        raise ProductionQualificationPreflightError("P4.6 phase order drift")
    for field in (
        "automatically_start_b2",
        "automatically_run_idt",
        "automatically_run_sar",
        "automatically_run_ca_opd",
        "automatically_run_controller",
        "automatically_run_confirmation",
        "automatically_run_final",
    ):
        if config.get("execution", {}).get(field) is not False:
            raise ProductionQualificationPreflightError(f"P4.6 automatic action is enabled: {field}")
    if config.get("execution", {}).get("generate_calibration_package_after_full_readiness") is not True:
        raise ProductionQualificationPreflightError("calibration package generation is not frozen")
    expected_isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if config.get("isolation") != expected_isolation:
        raise ProductionQualificationPreflightError("P4.6 isolation gate failed")


def _validate_checked_in_sources(
    config: Mapping[str, Any], config_path: Path, root: Path
) -> dict[str, Any]:
    card = _strict_card(config, config_path, root)
    artifacts = config["artifacts"]
    schema_path = _assert_file_sha(
        root, artifacts["schema_path"], artifacts["schema_sha256"], "P4.6 artifact schema"
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ProductionQualificationPreflightError(f"P4.6 artifact schema is invalid: {error}") from error
    if schema.get("additionalProperties") is not False:
        raise ProductionQualificationPreflightError("P4.6 artifact schema is not strict")
    protocol = config["validation"]
    protocol_path = _assert_file_sha(
        root, protocol["config_path"], protocol["config_sha256"], "three-policy protocol"
    )
    production = config["production_binding"]
    try:
        binding = verify_b2_backend_binding(
            production["b2_config_path"], production["b2_run_card_path"], repo_root=root
        )
    except Exception as error:
        raise ProductionQualificationPreflightError(f"production binding failed: {error}") from error
    if (
        binding["production_backend"]["backend_id"] != production["backend_id"]
        or binding["config_sha256"] != production["b2_config_sha256"]
        or binding["run_card_sha256"] != production["b2_run_card_sha256"]
    ):
        raise ProductionQualificationPreflightError("production binding SHA mismatch")
    try:
        from src.opd.production_qualification_v6 import (
            build_current_backend_binding_manifest,
        )

        current_backend = build_current_backend_binding_manifest(
            config, repo_root=root
        )
    except Exception as error:
        raise ProductionQualificationPreflightError(
            f"current production executable chain failed: {error}"
        ) from error
    current_backend_sha = hashlib.sha256(
        json.dumps(
            current_backend,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if card["artifact_schema_path"] != artifacts["schema_path"] or card[
        "artifact_schema_sha256"
    ] != artifacts["schema_sha256"]:
        raise ProductionQualificationPreflightError("run-card artifact schema binding mismatch")
    if (
        card["production_backend_id"] != production["backend_id"]
        or card["refresh_implementation"] != config["sampler_refresh"]["candidate_mechanism"]
        or card["runtime_slot"] != config["sampler_refresh"]["runtime_slot"]
        or card["prompt_selection_manifest_path"]
        != config["prompt_selection"]["selection_manifest_path"]
        or card["prompt_selection_manifest_sha256"]
        != config["prompt_selection"]["selection_manifest_sha256"]
        or card["probe_spec_sha256"] != config["fixed_action_probe"]["probe_spec_sha256"]
        or card["phase_groups"] != config["execution"]["phase_groups"]
    ):
        raise ProductionQualificationPreflightError("P4.6 run-card/config semantic binding mismatch")
    return {
        "config_sha256": sha256_file(config_path),
        "run_card_sha256": sha256_file(
            root / "configs/run_cards" / f"{config['run']['run_id']}.json"
        ),
        "artifact_schema_sha256": sha256_file(schema_path),
        "protocol_sha256": sha256_file(protocol_path),
        "b2_config_sha256": binding["config_sha256"],
        "b2_run_card_sha256": binding["run_card_sha256"],
        "backend_binding_sha256": current_backend_sha,
        "production_backend_binding_verified": True,
    }


def _validate_models_prompts_and_history(config: Mapping[str, Any], root: Path) -> None:
    model = config["model"]
    teacher = config["teacher"]
    prompts = config["prompt_selection"]
    _assert_file_sha(root, model["artifact_manifest_path"], model["artifact_manifest_sha256"], "Base manifest")
    validate_base_model_transport(
        base_path=_resolve(root, model["id"]),
        weights_manifest_path=_resolve(root, model["weights_manifest_path"]),
        weights_manifest_sha256=model["weights_manifest_sha256"],
        expected_revision=str(model["revision"]),
    )
    _assert_file_sha(root, teacher["route_config"], teacher["route_config_sha256"], "Teacher route")
    _assert_file_sha(root, teacher["manifest_path"], teacher["manifest_sha256"], "Teacher manifest")
    adapter = _resolve(root, teacher["adapter_path"])
    _assert_file_sha(
        root,
        adapter / "adapter_model.safetensors",
        teacher["adapter_weight_sha256"],
        "Teacher adapter weight",
    )
    if _ordered_adapter_sha256(adapter) != teacher["adapter_sha256"]:
        raise ProductionQualificationPreflightError(
            "Teacher ordered adapter SHA mismatch"
        )
    for path_field, sha_field in (
        ("selection_manifest_path", "selection_manifest_sha256"),
        ("opd_manifest_path", "opd_manifest_sha256"),
        ("medical_opd_o1_path", "medical_opd_o1_sha256"),
        ("medical_opd_cmb_path", "medical_opd_cmb_sha256"),
    ):
        _assert_file_sha(root, prompts[path_field], prompts[sha_field], path_field)
    validate_frozen_prompt_selection(config, repo_root=root)
    if any(
        prompts.get(field) is not False
        for field in ("contains_labels", "contains_final", "contains_controller", "contains_confirmation")
    ):
        raise ProductionQualificationPreflightError("P4.6 prompt isolation drift")
    history = config["historical_protection"]
    expected_counts = {"p4_1": 1, "p4_2": 5, "p4_3": 9, "p4_4": 6, "p4_5": 7}
    for stage, expected_count in expected_counts.items():
        output_key = f"{stage}_output_dir"
        artifacts_key = f"{stage}_artifacts"
        artifacts = history.get(artifacts_key)
        if not isinstance(artifacts, Mapping) or len(artifacts) != expected_count:
            raise ProductionQualificationPreflightError(
                f"historical {stage.upper()} protection inventory drift"
            )
        protected_output = Path(str(history.get(output_key, "")))
        for name, expected in artifacts.items():
            _assert_file_sha(
                root,
                protected_output / str(name),
                expected,
                f"historical {stage.upper()} {name}",
            )


def preflight(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    execute_gpu: bool,
    require_clean_git: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the future package without importing model or CUDA libraries."""

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    path = Path(config_path)
    path = (path if path.is_absolute() else root / path).resolve()
    _validate_contract(config)
    identities = _validate_checked_in_sources(config, path, root)
    _validate_models_prompts_and_history(config, root)
    output = Path(str(config["run"]["output_dir"]))
    protected = Path(str(config["historical_protection"]["p4_5_output_dir"]))
    if output.resolve() == protected.resolve():
        raise ProductionQualificationPreflightError("P4.5 protected output cannot be reused")
    if output.exists():
        raise ProductionQualificationPreflightError("P4.6 formal output must be fresh and absent")
    generated_package = Path(str(config["run"]["generated_b2_package_dir"]))
    if generated_package.exists() or generated_package.is_symlink():
        raise ProductionQualificationPreflightError(
            "generated B2 calibration package must be fresh and absent"
        )
    if not output.parent.is_dir():
        raise ProductionQualificationPreflightError("P4.6 output parent is absent")
    free_disk_gib = shutil.disk_usage(output.parent).free / float(1024**3)
    if free_disk_gib < float(config["resources"]["minimum_free_disk_gib"]):
        raise ProductionQualificationPreflightError("insufficient persistent disk")
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
        ).stdout
        if status:
            raise ProductionQualificationPreflightError("GPU qualification requires a clean worktree")

    dependency_versions: dict[str, str] = {}
    for distribution, expected in config["versions"].items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = "not_installed"
        dependency_versions[distribution] = observed
        if execute_gpu and observed != str(expected):
            raise ProductionQualificationPreflightError(f"pinned dependency drift: {distribution}")

    gpu_inventory: list[dict[str, Any]] = []
    if execute_gpu:
        authorization = config["authorization"]
        if os.environ.get(authorization["environment_variable"]) != authorization["required_value"]:
            raise ProductionQualificationPreflightError("P4.6 GPU authorization is absent")
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
            raise ProductionQualificationPreflightError("idle dual RTX 3090 gate failed")
        compute = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if compute:
            raise ProductionQualificationPreflightError("unknown GPU process detected")

    return {
        "status": "ready_waiting_for_gpu_combined_qualification",
        "gpu_used": False,
        "loaded_real_model": False,
        "B2_started": False,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "controller_access": False,
        "confirmation_access": False,
        "final_access": False,
        "label_access": False,
        "free_disk_gib": free_disk_gib,
        "dependency_versions": dependency_versions,
        "gpu_inventory": gpu_inventory,
        **identities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.6 combined qualification preflight")
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
