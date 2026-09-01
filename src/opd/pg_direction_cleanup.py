"""Post-process GPU cleanup and final P4.2 readiness derivation.

This module runs only after the GPU runtime child has exited.  It imports no
model runtime and never starts CUDA work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from src.opd.pg_opd_validation import (
    atomic_write_json,
    build_artifact_index,
    recompute_readiness,
)


class PGDirectionCleanupError(RuntimeError):
    pass


def _gpu_memory(query: Callable[..., Any]) -> list[int]:
    result = query(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _gpu_processes(query: Callable[..., Any]) -> list[str]:
    result = query(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _worker_processes(query: Callable[..., Any]) -> list[str]:
    result = query(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        text=True,
        capture_output=True,
    )
    markers = ("vllm", "raylet", "ray::", "verl_worker")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(marker in line.lower() for marker in markers)
    ]


def finalize_post_exit_cleanup(
    config: Mapping[str, Any], *, runtime_exit_code: int, query: Callable[..., Any] = subprocess.run
) -> dict[str, Any]:
    output = Path(str(config["run"]["output_dir"]))
    if not output.is_dir():
        raise PGDirectionCleanupError("GPU runtime output directory is missing")
    memory_used = _gpu_memory(query)
    gpu_processes = _gpu_processes(query)
    workers = _worker_processes(query)
    clean = bool(
        len(memory_used) == int(config["resources"]["required_gpus"])
        and all(value == 0 for value in memory_used)
        and not gpu_processes
        and not workers
    )
    cleanup = {
        "schema_version": 2,
        "status": "pass" if clean else "fail",
        "runtime_exit_code": int(runtime_exit_code),
        "gpu_memory_used_mib": memory_used,
        "gpu_compute_processes": gpu_processes,
        "vllm_ray_workers_found": bool(workers),
        "vllm_ray_worker_processes": workers,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    cleanup_sha = atomic_write_json(output / "resource_cleanup.json", cleanup)
    if runtime_exit_code != 0:
        failure_path = output / "failure.json"
        failure_status = "failed_artifact_integrity"
        if failure_path.is_file():
            try:
                existing = json.loads(failure_path.read_text(encoding="utf-8"))
                if isinstance(existing.get("status"), str):
                    failure_status = existing["status"]
            except (OSError, json.JSONDecodeError):
                pass
        else:
            runtime_failure = {
                "schema_version": 2,
                "status": failure_status,
                "runtime_exit_code": int(runtime_exit_code),
                "resource_cleanup_sha256": cleanup_sha,
                "B2_authorized": False,
            }
            runtime_failure_sha = atomic_write_json(
                output / "runtime_exit_failure.json", runtime_failure
            )
            atomic_write_json(
                failure_path,
                {
                    "schema_version": 2,
                    "status": failure_status,
                    "reason": f"GPU runtime exited nonzero: {runtime_exit_code}",
                    "metrics_path": "runtime_exit_failure.json",
                    "metrics_sha256": runtime_failure_sha,
                },
            )
        atomic_write_json(
            output / "summary.json",
            {
                "schema_version": 2,
                "status": failure_status,
                "runtime_exit_code": int(runtime_exit_code),
                "resource_cleanup_verified": clean,
                "B2_authorized": False,
                "formal_opd_authorized": False,
            },
        )
        if not clean:
            raise PGDirectionCleanupError(
                "post-exit GPU resource cleanup failed after nonzero runtime exit"
            )
        return {
            **cleanup,
            "status": "runtime_failed_cleanup_recorded",
            "failure_status": failure_status,
        }
    if not clean:
        raise PGDirectionCleanupError("post-exit GPU resource cleanup failed")

    filenames = (
        "scorer_identity.json",
        "scorer_readiness.json",
        "pre_update_evidence.json",
        "medical_step_checkpoint.json",
        "one_step_result.json",
        "one_step_metrics.json",
        "null_update.json",
        "sampler_refresh.json",
        "sampler_refresh_observations.json",
        "runtime_release.json",
        "resource_cleanup.json",
    )
    build_artifact_index(output, filenames)
    root = Path(__file__).resolve().parents[2]
    base_config_path = Path(str(config["base_calibration"]["config_path"]))
    if not base_config_path.is_absolute():
        base_config_path = root / base_config_path
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    readiness = recompute_readiness(
        output,
        expected_protocol_id=str(config["validation"]["protocol_id"]),
        expected_config_sha256=str(config["validation"]["config_sha256"]),
        expected_trajectory_sha256=str(config["frozen_input"]["trajectory_sha256"]),
        expected_medical_adapter_sha256=str(base_config["teacher"]["adapter_sha256"]),
        expected_repeatability_sha256=str(
            config["historical_scorer_evidence"]["repeatability_report_sha256"]
        ),
        expected_route_isolation_sha256=str(
            config["historical_scorer_evidence"]["route_isolation_report_sha256"]
        ),
        expected_same_model_null_sha256=str(
            config["historical_scorer_evidence"]["same_model_null_report_sha256"]
        ),
    )
    if readiness["status"] != "opd_backend_ready_waiting_for_b2_authorization":
        raise PGDirectionCleanupError("post-exit derived readiness failed closed")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = {
        **readiness,
        "schema_version": 2,
        "P4_1_status": "blocked_pg_opd_direction",
        "vllm_backend": "diagnostic_only",
        "formal_checkpoint_saved": False,
        "B2_authorized": False,
        "formal_opd_authorized": False,
        "git_head": metadata["git_head"],
        "run_config_sha256": metadata["run_config_sha256"],
        "run_card_sha256": metadata["run_card_sha256"],
        "next_step": "request_separate_B2_authorization",
    }
    atomic_write_json(output / "summary.json", summary)
    metadata.update(
        {
            "status": readiness["status"],
            "post_exit_cleanup_verified_at": cleanup["checked_at"],
            "B2_authorized": False,
        }
    )
    atomic_write_json(output / "metadata.json", metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.2 post-exit GPU cleanup finalizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-exit-code", required=True, type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        report = finalize_post_exit_cleanup(
            config, runtime_exit_code=args.runtime_exit_code
        )
    except Exception as error:
        output = Path(str(config["run"]["output_dir"]))
        if output.is_dir():
            failure_metrics = {
                "schema_version": 2,
                "status": "failed_artifact_integrity",
                "error_type": type(error).__name__,
                "error": str(error),
                "B2_authorized": False,
            }
            metrics_sha = atomic_write_json(
                output / "post_exit_cleanup_failure.json", failure_metrics
            )
            atomic_write_json(
                output / "failure.json",
                {
                    "schema_version": 2,
                    "status": "failed_artifact_integrity",
                    "reason": f"{type(error).__name__}: {error}",
                    "metrics_path": "post_exit_cleanup_failure.json",
                    "metrics_sha256": metrics_sha,
                },
            )
            atomic_write_json(
                output / "summary.json",
                {
                    "schema_version": 2,
                    "status": "failed_artifact_integrity",
                    "B2_authorized": False,
                    "formal_opd_authorized": False,
                },
            )
        raise
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
