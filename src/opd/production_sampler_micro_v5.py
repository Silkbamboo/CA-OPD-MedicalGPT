"""Future authorized entrypoint for the P4.5 production refresh micro-smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


class ProductionSamplerMicroError(RuntimeError):
    pass


_PHASES = [
    "formal_preflight",
    "production_backend_binding",
    "regenerate_v1_four_prompt_corrected_one_step",
    "trainer_v1_memory_vs_reload",
    "load_long_lived_v0_student_active",
    "v0_fixed_action_probe",
    "stable_slot_hotswap_v0_to_v1",
    "runtime_per_tensor_identity",
    "trainer_authoritative_identity_gate",
    "fresh_v1_reference",
    "refreshed_vs_fresh_fixed_action_gate",
    "normal_v1_request",
    "stale_v0_request_rejection",
    "refresh_latency",
    "artifact_readiness_cleanup",
    "stop_without_starting_b2",
]


def micro_plan(config: Mapping[str, Any]) -> list[str]:
    if (
        config.get("schema_version") != 5
        or config.get("run", {}).get("stage") != "production_sampler_refresh_micro_v5"
        or config.get("execution", {}).get("ordered_phases") != _PHASES
        or config.get("execution", {}).get("automatically_start_b2") is not False
        or config.get("historical_v1", {}).get("regenerate_with_minimal_four_prompt_one_step")
        is not True
        or config.get("historical_v1", {}).get("rerun_16_prompts") is not False
        or config.get("historical_v1", {}).get("rerun_32_prompts") is not False
        or config.get("historical_v1", {}).get("run_base_null") is not False
    ):
        raise ProductionSamplerMicroError("P4.5 GPU micro plan drift")
    return list(_PHASES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return _sha256(path)


def _append_metric(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _bootstrap_failure(
    output: Path,
    config: Mapping[str, Any],
    error: Exception,
    *,
    phase: str,
) -> None:
    metrics = output / "metrics.jsonl"
    _append_metric(
        metrics,
        {
            "step": 0,
            "phase": phase,
            "status": "fail",
            "error_code": "PRODUCTION_REFRESH_RUNTIME_FAILURE",
        },
    )
    refresh_path = output / "production_sampler_refresh.json"
    if refresh_path.is_file():
        artifact = json.loads(refresh_path.read_text(encoding="utf-8"))
        artifact_sha = _sha256(refresh_path)
        failure_layer = artifact.get("failure_layer") or phase
    else:
        artifact = {
            "artifact_protocol_version": "p4.5-production-sampler-refresh-v5",
            "run_id": config["run"]["run_id"],
            "status": "fail",
            "production_backend_binding": dict(config["production_binding"]),
            "candidate_mechanism": config["sampler_refresh"]["candidate_mechanism"],
            "authoritative_manifest_sha256": None,
            "adapter_config_sha256": None,
            "trainer_identity": None,
            "runtime_identity": None,
            "fresh_identity": None,
            "logical_versions": {"before": 0, "after": None},
            "runtime_slot": "student_active",
            "active_adapter": None,
            "registry_before": None,
            "registry_after": None,
            "same_path_metrics": {
                "mae": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "finite_rate": 0.0,
                "worst_token": None,
                "threshold": 0.0001,
            },
            "normal_request": {
                "accepted": False,
                "scoring_executed": False,
                "generation_executed": False,
            },
            "stale_request": {
                "rejected": False,
                "scoring_executed": False,
                "generation_executed": False,
            },
            "refresh_latency_seconds": None,
            "failure_layer": phase,
            "gate_result": "fail",
            "failure_reason": f"{type(error).__name__}: {error}",
            "isolation": dict(config["isolation"]),
        }
        artifact_sha = _atomic_json(refresh_path, artifact)
        failure_layer = phase
    failure = {
        "schema_version": 5,
        "run_id": config["run"]["run_id"],
        "status": "failed_production_sampler_refresh",
        "failure_layer": failure_layer,
        "error_type": type(error).__name__,
        "error": str(error),
        "production_sampler_refresh_sha256": artifact_sha,
        "metrics_sha256": _sha256(metrics),
        "B2_authorized": False,
    }
    _atomic_json(output / "failure.json", failure)


def run_gpu_micro(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    micro_plan(config)
    authorization = config["authorization"]
    if os.environ.get(authorization["environment_variable"]) != authorization["required_value"]:
        raise ProductionSamplerMicroError("P4.5 GPU authorization is absent")
    from src.opd.production_sampler_micro_preflight_v5 import preflight

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
    output = Path(config["run"]["output_dir"])
    existing = (
        {str(item.relative_to(output)) for item in output.rglob("*") if item.is_file()}
        if output.exists()
        else set()
    )
    if existing - {"stdout.log"}:
        raise ProductionSamplerMicroError("P4.5 output is not a fresh launcher envelope")
    output.mkdir(parents=True, exist_ok=True)
    (output / "stdout.log").touch(exist_ok=True)
    (output / "metrics.jsonl").touch(exist_ok=True)
    started = time.time()
    common = {
        "schema_version": 5,
        "run_id": config["run"]["run_id"],
        "stage": config["run"]["stage"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "production_backend_id": config["production_binding"]["backend_id"],
        "B2_authorized": False,
        "B2_started": False,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    _atomic_json(output / "launch_record.json", {**common, "status": "authorized_runtime_bootstrapped"})
    _atomic_json(output / "config.json", config)
    _atomic_json(output / "metadata.json", {**common, "status": "running"})
    _atomic_json(output / "summary.json", {**common, "status": "authorized_runtime_bootstrapped"})
    _atomic_json(
        output / "cost.json",
        {
            "schema_version": 5,
            "run_id": config["run"]["run_id"],
            "currency": "CNY",
            "runtime_wall_seconds_observed": 0.0,
            "platform_actual_cost_cny": None,
            "B2_started": False,
        },
    )
    try:
        # This import is deliberately after authorization, host preflight, and
        # the atomic failure envelope. It is the only path that may import CUDA.
        from src.opd.three_policy_gpu_runtime import (
            execute_production_sampler_micro_gpu_protocol_v5,
        )

        result = execute_production_sampler_micro_gpu_protocol_v5(
            config, config_path=config_path
        )
    except Exception as error:
        if not (output / "failure.json").is_file():
            _bootstrap_failure(
                output,
                config,
                error,
                phase="authorized_runtime_import_or_execution",
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
    parser = argparse.ArgumentParser(description="P4.5 production sampler GPU micro-smoke")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(json.dumps(run_gpu_micro(config, config_path=path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
