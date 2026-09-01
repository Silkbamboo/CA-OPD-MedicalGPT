"""Authorized entrypoint for the fresh P4.4 sampler-refresh GPU package."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


class SamplerRefreshGPUError(RuntimeError):
    pass


_PHASES = [
    "formal_preflight",
    "gpu_disk_process_check",
    "generation_probability_provenance",
    "sampler_v0_repeated_probe",
    "sampler_v0_noop_unload_reload_control",
    "fresh_full_support_rollout_4",
    "correction_calibration_4",
    "fresh_full_support_rollout_16",
    "correction_calibration_16",
    "optional_32_only_if_preregistered_instability",
    "corrected_medical_one_step",
    "trainer_v1_in_memory_vs_fresh_reload",
    "long_lived_sampler_v0_to_v1_refresh",
    "fresh_sampler_v1_reference",
    "live_refreshed_vs_fresh_same_path",
    "stale_v0_request_rejection",
    "generation_direct_cross_path_diagnostics",
    "real_base_teacher_null_update",
    "artifact_readiness_metrics_cleanup",
    "stop_without_starting_b2",
]


def revalidation_plan(config: Mapping[str, Any]) -> list[str]:
    if (
        config.get("schema_version") != 4
        or config.get("run", {}).get("stage") != "sampler_refresh_revalidation_v4"
        or config.get("execution", {}).get("ordered_phases") != _PHASES
        or config.get("execution", {}).get("automatically_start_b2") is not False
    ):
        raise SamplerRefreshGPUError("P4.4 execution plan drift")
    return list(_PHASES)


def run_gpu_revalidation(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    revalidation_plan(config)
    authorization = config["authorization"]
    if os.environ.get(authorization["environment_variable"]) != authorization[
        "required_value"
    ]:
        raise SamplerRefreshGPUError("P4.4 GPU authorization is absent")
    from src.opd.sampler_refresh_preflight_v4 import preflight

    preflight(
        config,
        config_path=config_path,
        execute_gpu=True,
        require_clean_git=True,
        allow_launcher_stdout_envelope=True,
    )
    return _run_authorized_protocol(config, config_path=config_path)


def _run_authorized_protocol(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Create a failure envelope before importing the model runtime."""

    from src.opd.sampler_refresh_contract import (
        _atomic_json,
        persist_sampler_refresh_failure_binding,
        persist_sampler_refresh_runtime_failure,
    )

    output = Path(config["run"]["output_dir"])
    existing = (
        {
            str(item.relative_to(output))
            for item in output.rglob("*")
            if item.is_file()
        }
        if output.exists()
        else set()
    )
    if existing - {"stdout.log"}:
        raise SamplerRefreshGPUError("P4.4 output is not a fresh launcher envelope")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "stdout.log").touch(exist_ok=True)
    (output / "metrics.jsonl").touch(exist_ok=True)
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    common = {
        "schema_version": 4,
        "run_id": config["run"]["run_id"],
        "stage": config["run"]["stage"],
        "started_at": started_at,
        "P4_3_status_preserved": "failed_sampler_refresh",
        "B2_authorized": False,
        "B2_started": False,
    }
    _atomic_json(
        output / "launch_record.json",
        {**common, "status": "authorized_runtime_bootstrapped"},
    )
    _atomic_json(output / "config.yaml", config)
    _atomic_json(output / "metadata.json", {**common, "status": "running"})
    _atomic_json(
        output / "data_manifest.json",
        {
            **common,
            "status": "fresh_prompt_only_sources_bound",
            "opd_manifest_sha256": config["prompt_selection"]["opd_manifest_sha256"],
            "medical_opd_o1_sha256": config["prompt_selection"][
                "medical_opd_o1_sha256"
            ],
            "medical_opd_cmb_sha256": config["prompt_selection"][
                "medical_opd_cmb_sha256"
            ],
            "labels_accessed": False,
            "final_accessed": False,
        },
    )
    _atomic_json(
        output / "checkpoints/index.json",
        {
            "schema_version": 4,
            "run_id": config["run"]["run_id"],
            "status": "diagnostic_one_step_no_formal_checkpoint",
            "checkpoints": [],
        },
    )
    _atomic_json(
        output / "cost.json",
        {
            "schema_version": 4,
            "run_id": config["run"]["run_id"],
            "currency": "CNY",
            "runtime_wall_seconds_observed": 0.0,
            "platform_actual_cost_cny": None,
            "actual_cost_cny": None,
            "B2_started": False,
        },
    )
    _atomic_json(
        output / "summary.json",
        {**common, "status": "authorized_runtime_bootstrapped"},
    )
    try:
        # Model/CUDA imports remain behind authorization, formal preflight and
        # the atomic failure envelope.
        from src.opd.three_policy_gpu_runtime import (
            execute_sampler_refresh_gpu_protocol_v4,
        )

        result = execute_sampler_refresh_gpu_protocol_v4(
            config, config_path=config_path
        )
    except Exception as error:
        if not (output / "failure.json").is_file():
            if (output / "sampler_refresh.json").is_file():
                persist_sampler_refresh_failure_binding(
                    output,
                    run_id=config["run"]["run_id"],
                    failed_phase="authorized_runtime_import_or_execution",
                    failure_status="failed_artifact_integrity",
                    error_type=type(error).__name__,
                    error=str(error),
                )
            else:
                persist_sampler_refresh_runtime_failure(
                    output,
                    run_id=config["run"]["run_id"],
                    failed_phase="authorized_runtime_import_or_execution",
                    failure_status="failed_artifact_integrity",
                    error_type=type(error).__name__,
                    error=str(error),
                    correction_metrics={"status": "not_run"},
                    one_step_metrics={"status": "not_run"},
                    null_metrics={"status": "not_run"},
                )
        _atomic_json(
            output / "summary.json",
            {
                **common,
                "status": "runtime_failed_pending_post_exit_cleanup",
                "return_to_cpu_decision": True,
            },
        )
        cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
        cost["runtime_wall_seconds_observed"] = time.time() - started
        _atomic_json(output / "cost.json", cost)
        raise
    cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
    cost["runtime_wall_seconds_observed"] = time.time() - started
    _atomic_json(output / "cost.json", cost)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.4 sampler refresh GPU replay")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(json.dumps(run_gpu_revalidation(config, config_path=path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
