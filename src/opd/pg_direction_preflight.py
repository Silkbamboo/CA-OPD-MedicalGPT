"""CPU-safe identity and protocol preflight for the P4.2 GPU rerun."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


class PGDirectionPreflightError(RuntimeError):
    pass


_SUPERVISION_FIELDS = {
    "answer",
    "answer_idx",
    "answer_index",
    "gold",
    "label",
    "labels",
    "solution",
    "reference_answer",
    "ground_truth",
    "reward",
}
_FORBIDDEN_STRING_FRAGMENTS = ("final", "controller", "confirmation", "label")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, root: Path) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else root / candidate


def _assert_file_sha(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or _sha256(path) != str(expected):
        raise PGDirectionPreflightError(f"{label} SHA mismatch")


def _find_forbidden_string(value: Any, path: str = "config") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_forbidden_string(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_forbidden_string(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_STRING_FRAGMENTS):
            return path
    return None


def _find_supervision(value: Any, path: str = "row") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in _SUPERVISION_FIELDS:
                return child
            found = _find_supervision(item, child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_supervision(item, f"{path}[{index}]")
            if found:
                return found
    return None


def _validate_frozen_trajectory(path: Path) -> dict[str, Any]:
    rows = 0
    tokens = 0
    roles: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            supervision = _find_supervision(row)
            if supervision:
                raise PGDirectionPreflightError(
                    f"frozen trajectory contains forbidden supervision: {supervision}"
                )
            role = str(row.get("source_role", ""))
            if any(fragment in role.lower() for fragment in _FORBIDDEN_STRING_FRAGMENTS):
                raise PGDirectionPreflightError(
                    f"frozen trajectory contains forbidden role at line {line_number}"
                )
            prompt_ids = row.get("prompt_ids")
            response_ids = row.get("response_ids")
            old = row.get("old")
            response_mask = row.get("response_mask")
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise PGDirectionPreflightError("frozen trajectory prompt IDs are missing")
            if not isinstance(response_ids, list) or not response_ids:
                raise PGDirectionPreflightError("frozen trajectory response IDs are missing")
            if not isinstance(old, list) or len(old) != len(response_ids):
                raise PGDirectionPreflightError("frozen trajectory old logprobs are misaligned")
            if not isinstance(response_mask, list) or response_mask != [1] * len(response_ids):
                raise PGDirectionPreflightError("frozen trajectory response mask is misaligned")
            roles[role] = roles.get(role, 0) + 1
            rows += 1
            tokens += len(response_ids)
    return {"rows": rows, "response_tokens": tokens, "roles": roles}


def preflight(
    config: Mapping[str, Any],
    *,
    execute_gpu: bool,
    require_clean_git: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    run = config.get("run", {})
    if (
        config.get("schema_version") != 2
        or run.get("stage") != "pg_direction_revalidation"
        or run.get("calibration_only") is not True
        or run.get("formal_opd_training") is not False
        or run.get("one_step_only") is not True
        or run.get("total_runtime_hard_kill_allowed") is not False
        or run.get("evidence_driven_fix_retries_max") != 2
    ):
        raise PGDirectionPreflightError("P4.2 run boundary drift")
    if "frozen_override" in config:
        raise PGDirectionPreflightError("forbidden algorithm or optimizer override")

    validation = config.get("validation", {})
    protocol_path = _path(validation.get("config_path", ""), root)
    _assert_file_sha(protocol_path, validation.get("config_sha256"), "protocol config")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (
        validation.get("protocol_id") != "pg_opd_frozen_update_validation_v2"
        or protocol.get("protocol_id") != validation.get("protocol_id")
        or protocol.get("algorithm", {}).get("beta") != 1.0
        or protocol.get("algorithm", {}).get("clip_low") != 0.2
        or protocol.get("algorithm", {}).get("clip_high") != 0.28
        or protocol.get("optimizer", {}).get("type") != "AdamW"
        or protocol.get("optimizer", {}).get("fresh_state") is not True
        or protocol.get("optimizer", {}).get("learning_rate") != 3e-5
        or protocol.get("optimizer", {}).get("weight_decay") != 0.0
    ):
        raise PGDirectionPreflightError("frozen protocol algorithm/optimizer drift")

    base = config.get("base_calibration", {})
    base_path = _path(base.get("config_path", ""), root)
    _assert_file_sha(base_path, base.get("config_sha256"), "P4.1 config")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if (
        base_config.get("algorithm", {}).get("beta") != 1.0
        or base_config.get("algorithm", {}).get("student_lora", {}).get("calibration_lr") != 3e-5
        or base_config.get("scoring", {}).get("formal_backend")
        != "transformers_direct_trajectory_logits"
        or base_config.get("scoring", {}).get("vllm", {}).get("diagnostic_only") is not True
        or base_config.get("scoring", {}).get("vllm", {}).get("formal_enabled") is not False
    ):
        raise PGDirectionPreflightError("P4.1 algorithm/scorer identity drift")

    frozen = config.get("frozen_input", {})
    forbidden = _find_forbidden_string(
        {
            key: value
            for key, value in frozen.items()
            if key not in {
                "contains_labels",
                "contains_final",
                "contains_controller",
                "contains_confirmation",
            }
        },
        "frozen_input",
    )
    if forbidden:
        raise PGDirectionPreflightError(f"forbidden data reference at {forbidden}")
    if any(
        frozen.get(key) is not False
        for key in (
            "contains_labels",
            "contains_final",
            "contains_controller",
            "contains_confirmation",
        )
    ):
        raise PGDirectionPreflightError("forbidden capability attestation is not false")
    trajectory_path = _path(frozen.get("trajectory_path", ""), root)
    _assert_file_sha(trajectory_path, frozen.get("trajectory_sha256"), "frozen trajectory")
    trajectory_report = _validate_frozen_trajectory(trajectory_path)
    if (
        trajectory_report["rows"] != frozen.get("expected_rows")
        or trajectory_report["response_tokens"] != frozen.get("expected_response_tokens")
        or trajectory_report["roles"] != frozen.get("expected_roles")
    ):
        raise PGDirectionPreflightError("frozen trajectory composition drift")
    for path_key, sha_key, label in (
        ("trajectory_manifest_path", "trajectory_manifest_sha256", "trajectory manifest"),
        ("live_rollout_manifest_path", "live_rollout_manifest_sha256", "live rollout manifest"),
    ):
        _assert_file_sha(_path(frozen.get(path_key, ""), root), frozen.get(sha_key), label)

    scorer = config.get("historical_scorer_evidence", {})
    if (
        scorer.get("immutable_p4_1_status") != "blocked_pg_opd_direction"
        or scorer.get("vllm_status") != "diagnostic_only"
        or scorer.get("rerun_full_scorer_calibration") is not False
        or scorer.get("rerun_full_vllm_diagnostic") is not False
    ):
        raise PGDirectionPreflightError("P4.1 history or scorer scope drift")
    for path_key, sha_key, label in (
        ("repeatability_report", "repeatability_report_sha256", "repeatability report"),
        ("route_isolation_report", "route_isolation_report_sha256", "route isolation report"),
        ("same_model_null_report", "same_model_null_report_sha256", "same-model null report"),
    ):
        _assert_file_sha(_path(scorer.get(path_key, ""), root), scorer.get(sha_key), label)

    execution = config.get("execution", {})
    if (
        execution.get("ordered_phases")
        != [
            "formal_host_identity_preflight",
            "minimal_medical_scorer_identity_probe",
            "frozen_medical_one_step",
            "frozen_base_teacher_null_update",
            "sampler_refresh_identity",
            "artifact_integrity_and_readiness",
            "release_gpu_resources",
        ]
        or execution.get("stop_on_first_failure") is not True
        or any(
            execution.get(key) is not False
            for key in (
                "automatically_start_b2",
                "automatically_run_controller",
                "automatically_run_confirmation",
                "automatically_run_final",
                "automatically_run_training",
            )
        )
    ):
        raise PGDirectionPreflightError("P4.2 execution scope drift")

    output = _path(run.get("output_dir", ""), root)
    if output.exists() and any(output.iterdir()):
        raise PGDirectionPreflightError("P4.2 output directory must be new/empty")
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if status:
            raise PGDirectionPreflightError("GPU revalidation requires a clean committed worktree")
    if execute_gpu:
        if os.environ.get("CA_OPD_ALLOW_PG_DIRECTION_REVALIDATION_GPU") != "1":
            raise PGDirectionPreflightError("GPU direction revalidation lacks explicit authorization")
        # Reuse the already-audited P4.1 host/model/Teacher identity checks, but
        # point them at the new empty output and do not authorize the old runner.
        from src.opd.scorer_preflight import preflight as scorer_preflight

        identity_config = copy.deepcopy(base_config)
        identity_config["run"]["output_dir"] = str(output)
        scorer_preflight(
            identity_config,
            execute_gpu=False,
            require_clean_git=False,
            repo_root=root,
        )
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        inventory = []
        for line in query:
            if line.strip():
                index, name, uuid, total, used = [item.strip() for item in line.split(",", 4)]
                inventory.append(
                    {
                        "index": int(index),
                        "name": name,
                        "uuid": uuid,
                        "memory_total_mib": int(total),
                        "memory_used_mib": int(used),
                    }
                )
        if len(inventory) != 2 or any(
            "RTX 3090" not in item["name"]
            or item["memory_total_mib"] < 24576
            or item["memory_used_mib"] != 0
            for item in inventory
        ):
            raise PGDirectionPreflightError("GPU identity/idle contract mismatch")
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
            raise PGDirectionPreflightError("unexpected GPU compute process detected")
        return {
            "status": "ready_gpu_pg_direction_revalidation_authorized",
            "gpu_used": False,
            "loaded_real_model": False,
            "trajectory_count": trajectory_report["rows"],
            "response_tokens": trajectory_report["response_tokens"],
            "protocol_config_sha256": validation["config_sha256"],
            "p4_1_status_preserved": scorer["immutable_p4_1_status"],
            "b2_authorized": False,
            "gpu_inventory": inventory,
        }
    return {
        "status": "ready_waiting_for_gpu_pg_direction_revalidation",
        "gpu_used": False,
        "loaded_real_model": False,
        "trajectory_count": trajectory_report["rows"],
        "response_tokens": trajectory_report["response_tokens"],
        "protocol_config_sha256": validation["config_sha256"],
        "p4_1_status_preserved": scorer["immutable_p4_1_status"],
        "b2_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.2 PG direction revalidation preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    parser.add_argument("--output-dir-override")
    args = parser.parse_args(argv)
    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.output_dir_override:
        config["run"]["output_dir"] = args.output_dir_override
    report = preflight(
        config,
        execute_gpu=args.execute_gpu,
        require_clean_git=not args.allow_dirty_for_development,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
