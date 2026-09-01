"""Authorized GPU entrypoint for the frozen P4.3 three-policy revalidation.

Importing this module and inspecting its plan are CPU-only. CUDA/model imports
must remain behind the explicit authorization and preflight in
``run_gpu_revalidation``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ThreePolicyGPUError(RuntimeError):
    pass


_PHASES = [
    "formal_host_sha_preflight",
    "generation_score_provenance_micro_probe",
    "fresh_full_support_rollout_4",
    "correction_calibration_4",
    "fresh_full_support_rollout_16",
    "correction_calibration_16",
    "optional_rollout_32_only_if_low_cost_uncertainty",
    "corrected_medical_one_step",
    "real_base_teacher_null_update",
    "sampler_refresh",
    "artifact_readiness",
    "release_gpu_resources",
]


def revalidation_plan(config: Mapping[str, Any]) -> list[str]:
    run = config.get("run", {})
    execution = config.get("execution", {})
    if (
        run.get("stage") != "three_policy_revalidation"
        or run.get("calibration_only") is not True
        or run.get("formal_opd_training") is not False
        or run.get("one_step_only") is not True
        or execution.get("stop_on_first_failure") is not True
        or execution.get("automatically_start_b2") is not False
        or list(execution.get("ordered_phases", ())) != _PHASES
    ):
        raise ThreePolicyGPUError("P4.3 execution plan drift")
    return list(_PHASES)


def run_gpu_revalidation(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    revalidation_plan(config)
    if os.environ.get("CA_OPD_ALLOW_THREE_POLICY_REVALIDATION_GPU") != "1":
        raise ThreePolicyGPUError("GPU three-policy revalidation lacks explicit authorization")
    from src.opd.three_policy_preflight import preflight

    preflight(
        config,
        execute_gpu=True,
        require_clean_git=True,
        allow_empty_launcher_stdout_envelope=True,
    )
    # The model runtime is isolated in a separate function so the CPU package,
    # tests and dry-run never import torch/Transformers/PEFT.
    return _run_authorized_gpu_protocol(config, config_path=config_path)


def _run_authorized_gpu_protocol(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Bootstrap auditable artifacts, then execute the authorized runtime.

    This wrapper deliberately creates the failure envelope before importing
    torch, Transformers, PEFT, or the model runtime. Framework import,
    run-card validation, and CUDA initialization failures therefore remain
    attributable instead of being replaced by a later cleanup error.
    """

    output = Path(str(config["run"]["output_dir"]))
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    if output.exists():
        existing = {
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file()
        }
        # The shell opens stdout.log for tee immediately before this process.
        if existing - {"stdout.log"}:
            raise ThreePolicyGPUError("P4.3 output is not a fresh launcher envelope")
    else:
        output.mkdir(parents=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "stdout.log").touch(exist_ok=True)
    launch_record = {
        "schema_version": 3,
        "run_id": config["run"]["run_id"],
        "status": "authorized_runtime_bootstrapped",
        "started_at": started_at,
        "config_path": str(config_path),
        "git_head": "pending_runtime_card_validation",
        "P4_2_status_preserved": "failed_identity_mismatch",
        "B2_authorized": False,
    }
    atomic_json(output / "launch_record.json", launch_record)
    # JSON is a valid YAML subset and avoids importing a framework-adjacent
    # package before the failure envelope exists.
    atomic_json(output / "config.yaml", config)
    atomic_json(
        output / "metadata.json",
        {
            **launch_record,
            "stage": "three_policy_revalidation",
            "status": "running",
            "protocol_id": config["validation"]["protocol_id"],
            "protocol_config_sha256": config["validation"]["config_sha256"],
            "artifact_schema_sha256": config["artifacts"]["schema_sha256"],
            "actual_cost_cny": None,
        },
    )
    atomic_json(
        output / "data_manifest.json",
        {
            "schema_version": 3,
            "run_id": config["run"]["run_id"],
            "data_protocol": "p4.3-fresh-full-support-trajectory-v1",
            "source_manifest_path": config["prompt_selection"]["opd_manifest_path"],
            "source_manifest_sha256": config["prompt_selection"]["opd_manifest_sha256"],
            "sources": {
                "medical_opd_o1": {
                    "sha256": config["prompt_selection"]["medical_opd_o1_sha256"],
                    "planned_prompts": 8,
                },
                "medical_opd_cmb": {
                    "sha256": config["prompt_selection"]["medical_opd_cmb_sha256"],
                    "planned_prompts": 8,
                },
            },
            "labels_accessed": False,
            "final_accessed": False,
        },
    )
    (output / "metrics.jsonl").touch(exist_ok=True)
    atomic_json(
        output / "checkpoints" / "index.json",
        {
            "schema_version": 1,
            "run_id": config["run"]["run_id"],
            "status": "no_formal_checkpoint_planned_one_step_diagnostic",
            "checkpoints": [],
        },
    )
    atomic_json(
        output / "cost.json",
        {
            "currency": "CNY",
            "runtime_wall_seconds_observed": 0.0,
            "process_cost_cny": None,
            "platform_billed_cost_cny": None,
            "actual_cost_cny": None,
            "cost_semantics": "wall time observed; price and platform bill unknown",
        },
    )
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 3,
            "status": "authorized_runtime_bootstrapped",
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
        },
    )

    try:
        from src.opd.three_policy_gpu_runtime import execute_three_policy_gpu_protocol

        result = execute_three_policy_gpu_protocol(config, config_path=config_path)
    except Exception as error:
        if not (output / "failure.json").is_file():
            atomic_json(
                output / "failure.json",
                {
                    "schema_version": 3,
                    "status": "failed_artifact_integrity",
                    "phase": "authorized_runtime_import_or_execution",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "P4_2_status_preserved": "failed_identity_mismatch",
                    "B2_authorized": False,
                },
            )
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        metadata.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "failure_reason": str(error),
            }
        )
        atomic_json(output / "metadata.json", metadata)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summary.update(
            {
                "status": "runtime_failed_pending_post_exit_cleanup",
                "return_to_cpu_decision": True,
                "B2_authorized": False,
            }
        )
        atomic_json(output / "summary.json", summary)
        cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
        cost["runtime_wall_seconds_observed"] = time.time() - started
        atomic_json(output / "cost.json", cost)
        raise
    cost = json.loads((output / "cost.json").read_text(encoding="utf-8"))
    cost["runtime_wall_seconds_observed"] = time.time() - started
    atomic_json(output / "cost.json", cost)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized P4.3 three-policy GPU revalidation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    import yaml

    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(json.dumps(run_gpu_revalidation(config, config_path=path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
