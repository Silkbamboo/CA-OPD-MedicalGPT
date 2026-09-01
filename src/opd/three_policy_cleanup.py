"""CPU-only post-exit cleanup/final readiness for the P4.3 GPU package."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from src.opd.pg_opd_validation import atomic_write_json, sha256_file
from src.opd.three_policy_readiness import evaluate_three_policy_calibration


class ThreePolicyCleanupError(RuntimeError):
    pass


_SUCCESS_ARTIFACTS = (
    "launch_record.json",
    "config.yaml",
    "metadata.json",
    "data_manifest.json",
    "metrics.jsonl",
    "stdout.log",
    "checkpoints/index.json",
    "cost.json",
    "summary.json",
    "generation_provenance_probe.json",
    "fresh_full_support_trajectory_4.jsonl",
    "fresh_trajectory_manifest_4.json",
    "fresh_full_support_trajectory.jsonl",
    "fresh_trajectory_manifest.json",
    "correction_calibration_4.json",
    "correction_calibration_16.json",
    "optional_32_prompt_rung.json",
    "medical_step_checkpoint.json",
    "corrected_medical_one_step.json",
    "null_update_checkpoint.json",
    "real_base_teacher_null_update.json",
    "sampler_refresh_observations.json",
    "sampler_refresh.json",
    "readiness.json",
    "runtime_release.json",
    "resource_cleanup.json",
)


def required_success_artifacts() -> tuple[str, ...]:
    return _SUCCESS_ARTIFACTS


def _query_lines(query: Callable[..., Any], command: list[str]) -> list[str]:
    result = query(command, check=True, text=True, capture_output=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resource_state(query: Callable[..., Any]) -> dict[str, Any]:
    memory = [
        int(value)
        for value in _query_lines(
            query,
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        )
    ]
    gpu_processes = _query_lines(
        query,
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ],
    )
    processes = _query_lines(query, ["ps", "-eo", "pid=,comm=,args="])
    markers = ("vllm", "raylet", "ray::", "verl_worker")
    workers = [line for line in processes if any(marker in line.lower() for marker in markers)]
    return {
        "gpu_memory_used_mib": memory,
        "gpu_compute_processes": gpu_processes,
        "vllm_ray_worker_processes": workers,
    }


def _indexed_artifacts(output: Path, filenames: tuple[str, ...]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for filename in filenames:
        path = output / filename
        if not path.is_file():
            raise ThreePolicyCleanupError(f"required P4.3 artifact is missing: {filename}")
        artifacts[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {
        "schema_version": 3,
        "artifact_protocol_version": "p4.3-three-policy-correction-v3",
        "artifacts": artifacts,
    }
    atomic_write_json(output / "artifact_index.json", payload)
    return payload


def _index_available_failure_artifacts(output: Path) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and str(path.relative_to(output)) not in {"artifact_index.json", "summary.json"}
        and not path.name.endswith(".tmp")
    }
    if not {"failure.json", "resource_cleanup.json"}.issubset(artifacts):
        raise ThreePolicyCleanupError("failure evidence is incomplete before indexing")
    payload = {
        "schema_version": 3,
        "artifact_protocol_version": "p4.3-three-policy-correction-v3",
        "status": "failed_run_evidence_index",
        "artifacts": artifacts,
    }
    atomic_write_json(output / "artifact_index.json", payload)
    return payload


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreePolicyCleanupError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ThreePolicyCleanupError(f"artifact is not an object: {path.name}")
    return value


def _derive_success_readiness(output: Path) -> bool:
    calibration = _read_object(output / "correction_calibration_16.json")
    provenance = calibration.get("behavior_provenance")
    if not isinstance(provenance, Mapping):
        raise ThreePolicyCleanupError("calibration behavior provenance is missing")
    calibration_readiness = evaluate_three_policy_calibration(calibration, provenance)
    probe = _read_object(output / "generation_provenance_probe.json")
    medical = _read_object(output / "corrected_medical_one_step.json")
    manifest = _read_object(output / "fresh_trajectory_manifest.json")
    null = _read_object(output / "real_base_teacher_null_update.json")
    sampler = _read_object(output / "sampler_refresh.json")
    runtime_readiness = _read_object(output / "readiness.json")
    release = _read_object(output / "runtime_release.json")
    trajectory_sha = sha256_file(output / "fresh_full_support_trajectory.jsonl")
    run_id = provenance.get("trajectory_run_id")
    token_identity_sha = provenance.get("token_identity_sha256")
    return bool(
        calibration_readiness.calibration_ready
        and manifest.get("status") == "fresh_full_support"
        and manifest.get("run_id") == run_id
        and manifest.get("trajectory_sha256") == trajectory_sha
        and manifest.get("behavior_provenance") == provenance
        and manifest.get("old_p4_1_trajectory_used_as_formal_evidence") is False
        and manifest.get("P4_2_status_preserved") == "failed_identity_mismatch"
        and calibration.get("rung_prompts") == 16
        and calibration.get("trajectory_sha256") == trajectory_sha
        and calibration.get("token_identity_sha256") == token_identity_sha
        and medical.get("run_id") == run_id
        and medical.get("trajectory_sha256") == trajectory_sha
        and medical.get("token_identity_sha256") == token_identity_sha
        and medical.get("P4_2_status_preserved") == "failed_identity_mismatch"
        and probe.get("status") == "pass"
        and probe.get("GPU_observed") is True
        and medical.get("status") == "pass"
        and medical.get("hard_gate_passed") is True
        and null.get("status") == "pass"
        and null.get("hard_gate_passed") is True
        and sampler.get("status") == "pass"
        and sampler.get("hard_gate_passed") is True
        and runtime_readiness.get("status")
        == "three_policy_revalidation_runtime_passed_pending_post_exit_cleanup"
        and runtime_readiness.get("opd_backend_ready") is False
        and release.get("status") == "pass"
        and release.get("models_released") is True
        and release.get("post_process_exit_verification_required") is True
    )


def _record_cleanup_failure(
    output: Path,
    *,
    runtime_exit_code: int,
    reason: str,
    error_type: str,
    resource_cleanup_written: bool,
) -> None:
    if not resource_cleanup_written:
        atomic_write_json(
            output / "resource_cleanup.json",
            {
                "schema_version": 3,
                "status": "fail",
                "runtime_exit_code": int(runtime_exit_code),
                "query_completed": False,
                "error_type": error_type,
                "error": reason,
                "P4_2_status_preserved": "failed_identity_mismatch",
            },
        )
    atomic_write_json(
        output / "failure.json",
        {
            "schema_version": 3,
            "status": "failed_resource_cleanup",
            "phase": "post_exit_resource_cleanup",
            "error_type": error_type,
            "error": reason,
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
        },
    )
    _index_available_failure_artifacts(output)
    atomic_write_json(
        output / "summary.json",
        {
            "schema_version": 3,
            "status": "failed_resource_cleanup",
            "runtime_exit_code": int(runtime_exit_code),
            "resource_cleanup_verified": False,
            "return_to_cpu_decision": True,
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
            "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
        },
    )


def finalize_post_exit_cleanup(
    config: Mapping[str, Any],
    *,
    runtime_exit_code: int,
    query: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    output = Path(str(config["run"]["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    try:
        state = _resource_state(query)
    except Exception as exc:
        _record_cleanup_failure(
            output,
            runtime_exit_code=runtime_exit_code,
            reason=str(exc),
            error_type=type(exc).__name__,
            resource_cleanup_written=False,
        )
        raise ThreePolicyCleanupError("post-exit GPU resource query failed") from exc
    clean = bool(
        len(state["gpu_memory_used_mib"]) == int(config["resources"]["required_gpus"])
        and all(value == 0 for value in state["gpu_memory_used_mib"])
        and not state["gpu_compute_processes"]
        and not state["vllm_ray_worker_processes"]
    )
    cleanup = {
        "schema_version": 3,
        "status": "pass" if clean else "fail",
        "runtime_exit_code": int(runtime_exit_code),
        **state,
        "vllm_ray_workers_found": bool(state["vllm_ray_worker_processes"]),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "P4_2_status_preserved": "failed_identity_mismatch",
    }
    atomic_write_json(output / "resource_cleanup.json", cleanup)
    if not clean:
        _record_cleanup_failure(
            output,
            runtime_exit_code=runtime_exit_code,
            reason="GPU memory/process cleanup gate failed",
            error_type="ThreePolicyCleanupError",
            resource_cleanup_written=True,
        )
        raise ThreePolicyCleanupError("post-exit GPU resource cleanup failed")
    if runtime_exit_code != 0:
        if not (output / "failure.json").is_file():
            atomic_write_json(
                output / "failure.json",
                {
                    "schema_version": 3,
                    "status": "failed_artifact_integrity",
                    "reason": f"GPU runtime exited nonzero: {runtime_exit_code}",
                    "P4_2_status_preserved": "failed_identity_mismatch",
                    "B2_authorized": False,
                },
            )
        _index_available_failure_artifacts(output)
        summary = {
            "schema_version": 3,
            "status": "runtime_failed_cleanup_recorded",
            "runtime_exit_code": int(runtime_exit_code),
            "resource_cleanup_verified": True,
            "return_to_cpu_decision": True,
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
            "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
        }
        atomic_write_json(output / "summary.json", summary)
        return summary

    try:
        passed = _derive_success_readiness(output)
        if not passed:
            raise ThreePolicyCleanupError(
                "post-exit derived P4.3 readiness failed closed"
            )
    except Exception as exc:
        atomic_write_json(
            output / "failure.json",
            {
                "schema_version": 3,
                "status": "failed_artifact_integrity",
                "phase": "post_exit_artifact_readiness",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "P4_2_status_preserved": "failed_identity_mismatch",
                "B2_authorized": False,
            },
        )
        _index_available_failure_artifacts(output)
        atomic_write_json(
            output / "summary.json",
            {
                "schema_version": 3,
                "status": "failed_artifact_integrity",
                "failure_phase": "post_exit_artifact_readiness",
                "return_to_cpu_decision": True,
                "resource_cleanup_verified": True,
                "artifact_index_sha256": sha256_file(output / "artifact_index.json"),
                "P4_2_status_preserved": "failed_identity_mismatch",
                "B2_authorized": False,
            },
        )
        raise ThreePolicyCleanupError(
            "post-exit derived P4.3 readiness failed closed"
        ) from exc
    summary = {
        "schema_version": 3,
        "status": "three_policy_revalidation_passed_waiting_for_separate_b2_authorization",
        "generation_provenance_ready": True,
        "fresh_full_support_rollout_ready": True,
        "correction_calibration_ready": True,
        "corrected_medical_one_step_ready": True,
        "real_null_update_ready": True,
        "sampler_refresh_ready": True,
        "resource_cleanup_verified": True,
        "opd_backend_ready": False,
        "P4_2_status_preserved": "failed_identity_mismatch",
        "B2_authorized": False,
        "formal_opd_authorized": False,
        "next_step": "request_separate_user_decision_before_B2",
    }
    atomic_write_json(output / "summary.json", summary)
    metadata = _read_object(output / "metadata.json")
    metadata.update(
        {
            "status": summary["status"],
            "post_exit_cleanup_verified_at": cleanup["checked_at"],
            "B2_authorized": False,
        }
    )
    atomic_write_json(output / "metadata.json", metadata)
    cost = _read_object(output / "cost.json")
    cost["post_exit_cleanup_wall_clock_recorded_at"] = cleanup["checked_at"]
    cost["actual_cost_cny"] = None
    atomic_write_json(output / "cost.json", cost)
    _indexed_artifacts(output, _SUCCESS_ARTIFACTS)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.3 post-exit cleanup finalizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-exit-code", required=True, type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = finalize_post_exit_cleanup(
        config, runtime_exit_code=args.runtime_exit_code
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
