"""CPU final readiness and cleanup rules for P4.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
import jsonschema

from src.opd.sampler_refresh_contract import (
    _atomic_json,
    persist_sampler_refresh_failure_binding,
    persist_sampler_refresh_runtime_failure,
    persisted_sampler_refresh_failures,
)


class SamplerRefreshCleanupError(RuntimeError):
    pass


_BACKEND_BINDING_FIELDS = (
    "sampler_backend_id",
    "generation_backend",
    "refresh_implementation",
    "adapter_load_unload_implementation",
    "fixed_action_scoring_implementation",
    "model_revision",
    "tokenizer_revision",
    "dtype",
    "attention_backend",
    "cache_policy",
    "sampler_identity_guard",
)


def _b2_backend_binding_failures(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["B2 production backend binding evidence is absent"]
    p4 = value.get("p4_4")
    b2 = value.get("b2")
    if not isinstance(p4, Mapping) or not isinstance(b2, Mapping):
        return ["P4.4/B2 backend identity maps are absent"]
    failures = [
        f"P4.4/B2 backend field is missing or differs: {field}"
        for field in _BACKEND_BINDING_FIELDS
        if not isinstance(p4.get(field), str)
        or not p4.get(field)
        or p4.get(field) != b2.get(field)
    ]
    for field in (
        "p4_4_config_sha256",
        "b2_config_sha256",
        "b2_run_card_sha256",
    ):
        digest = value.get(field)
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        ):
            failures.append(f"backend binding digest is invalid: {field}")
    for field in ("b2_config_path", "b2_run_card_path"):
        if not isinstance(value.get(field), str) or not value.get(field):
            failures.append(f"backend binding path is invalid: {field}")
    return failures


def required_success_artifacts() -> tuple[str, ...]:
    return (
        "launch_record.json",
        "config.yaml",
        "metadata.json",
        "data_manifest.json",
        "metrics.jsonl",
        "stdout.log",
        "cost.json",
        "summary.json",
        "checkpoints/index.json",
        "generation_provenance_probe.json",
        "sampler_v0_controls.json",
        "fresh_full_support_trajectory_4.jsonl",
        "fresh_trajectory_manifest_4.json",
        "fresh_full_support_trajectory.jsonl",
        "fresh_trajectory_manifest.json",
        "correction_calibration_4.json",
        "correction_calibration_16.json",
        "optional_32_prompt_rung.json",
        "medical_step_checkpoint.json",
        "corrected_medical_one_step.json",
        "sampler_refresh_observations.json",
        "sampler_refresh.json",
        "null_update_checkpoint.json",
        "real_base_teacher_null_update.json",
        "runtime_release.json",
        "resource_cleanup.json",
    )


def derive_readiness_from_artifacts(
    artifacts: Mapping[str, Mapping[str, Any] | Any]
) -> dict[str, Any]:
    sampler = artifacts.get("sampler_refresh")
    run_id = artifacts.get("run_id")
    sampler_failures = (
        persisted_sampler_refresh_failures(sampler, expected_run_id=run_id)
        if isinstance(sampler, Mapping) and isinstance(run_id, str)
        else ["sampler report or run_id missing"]
    )
    sampler_passed = not sampler_failures
    cleanup = artifacts.get("cleanup")
    cleanup_passed = bool(
        isinstance(cleanup, Mapping) and cleanup.get("status") == "pass"
    )
    required = (
        "provenance",
        "rollout",
        "correction",
        "medical_one_step",
        "null_update",
    )
    phases_passed = all(
        isinstance(artifacts.get(name), Mapping)
        and artifacts[name].get("status") == "pass"
        for name in required
    )
    isolation = artifacts.get("isolation")
    isolated = isolation == {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    metrics_nonempty = artifacts.get("metrics_nonempty") is True
    artifact_integrity = artifacts.get("artifact_integrity") is True
    ready = bool(
        sampler_passed
        and cleanup_passed
        and phases_passed
        and isolated
        and metrics_nonempty
        and artifact_integrity
    )
    binding_failures = _b2_backend_binding_failures(
        artifacts.get("b2_backend_binding")
    )
    backend_bound = not binding_failures
    b2_authorized = bool(ready and backend_bound)
    if not ready:
        status = "failed_sampler_refresh"
    elif backend_bound:
        status = "opd_backend_ready_b2_authorized_not_started"
    else:
        status = "passed_sampler_refresh_backend_unbound"
    return {
        "schema_version": 4,
        "status": status,
        "sampler_refresh_revalidation": "passed" if ready else "failed",
        "sampler_refresh_runtime_ready": ready,
        "OPD_scoring_backend_ready": ready,
        "B2_backend_bound": backend_bound,
        "B2_backend_binding_failures": binding_failures,
        "B2_authorized": b2_authorized,
        "B2_started": False,
        "sampler_refresh_ready": sampler_passed,
        "sampler_refresh_recomputed_failures": sampler_failures,
        "cleanup_ready": cleanup_passed,
        "artifact_integrity_ready": artifact_integrity,
        "final_label_isolation": isolated,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplerRefreshCleanupError(f"invalid artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise SamplerRefreshCleanupError(f"artifact is not an object: {path.name}")
    return value


def _resource_state(query: Any) -> dict[str, Any]:
    memory_result = query(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    process_result = query(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    ps_result = query(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        text=True,
        capture_output=True,
    )
    memory = [
        int(line.strip())
        for line in memory_result.stdout.splitlines()
        if line.strip()
    ]
    gpu_processes = [
        line.strip() for line in process_result.stdout.splitlines() if line.strip()
    ]
    markers = ("vllm", "raylet", "ray::", "verl_worker", "torchrun")
    workers = [
        line.strip()
        for line in ps_result.stdout.splitlines()
        if any(marker in line.lower() for marker in markers)
    ]
    return {
        "gpu_memory_used_mib": memory,
        "gpu_compute_processes": gpu_processes,
        "project_workers": workers,
    }


def _write_index(output: Path, *, status: str) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(output)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_index.json", "readiness.json"}
        and not path.name.endswith(".tmp")
    }
    value = {
        "schema_version": 4,
        "artifact_protocol_version": "p4.4-sampler-refresh-contract-v4",
        "status": status,
        "artifacts": artifacts,
    }
    _atomic_json(output / "artifact_index.json", value)
    return value


def _verify_index(output: Path, index: Mapping[str, Any]) -> None:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SamplerRefreshCleanupError("artifact index has no artifact mapping")
    missing = set(required_success_artifacts()) - set(artifacts)
    if missing:
        raise SamplerRefreshCleanupError(
            "required success artifacts missing from index: " + ", ".join(sorted(missing))
        )
    for name, identity in artifacts.items():
        path = output / str(name)
        if not (
            isinstance(identity, Mapping)
            and path.is_file()
            and identity.get("sha256") == _sha256(path)
            and identity.get("bytes") == path.stat().st_size
        ):
            raise SamplerRefreshCleanupError(f"artifact index SHA/size mismatch: {name}")


def _read_metrics(output: Path, *, run_id: str) -> list[dict[str, Any]]:
    path = output / "metrics.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        raise SamplerRefreshCleanupError("metrics.jsonl is empty")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SamplerRefreshCleanupError("metrics.jsonl is invalid") from exc
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != 4
            or row.get("run_id") != run_id
        ):
            raise SamplerRefreshCleanupError("metrics run identity mismatch")
        rows.append(row)

    if len(rows) != 4:
        raise SamplerRefreshCleanupError("metrics must contain exactly four aggregates")
    by_phase: dict[str, dict[str, Any]] = {}
    for row in rows:
        phase = row.get("phase")
        if not isinstance(phase, str) or phase in by_phase:
            raise SamplerRefreshCleanupError("metrics phases must be unique")
        by_phase[phase] = row
    correction_phases = set(by_phase).intersection(
        {"correction_calibration_16", "correction_calibration_32"}
    )
    if set(by_phase) - correction_phases != {
        "corrected_medical_one_step",
        "real_base_teacher_null_update",
        "sampler_refresh",
    } or len(correction_phases) != 1:
        raise SamplerRefreshCleanupError("metrics aggregate phases are incomplete")

    correction_phase = next(iter(correction_phases))
    expected_steps = {
        correction_phase: 2,
        "corrected_medical_one_step": 3,
        "real_base_teacher_null_update": 4,
        "sampler_refresh": 5,
    }
    if any(
        type(row.get("step")) is not int
        or row.get("step") != expected_steps[phase]
        or not isinstance(row.get("status"), str)
        for phase, row in by_phase.items()
    ):
        raise SamplerRefreshCleanupError("metrics schema/step/status is invalid")

    def finite_number(row: Mapping[str, Any], key: str) -> float:
        value = row.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise SamplerRefreshCleanupError(
                f"metrics field is missing or non-finite: {key}"
            )
        return float(value)

    correction_row = by_phase[correction_phase]
    ess = finite_number(correction_row, "ess_fraction")
    cap = finite_number(correction_row, "cap_fraction")
    if not 0.80 <= ess <= 1.0 or not 0.0 <= cap <= 0.05:
        raise SamplerRefreshCleanupError(
            "metrics ESS/cap do not satisfy frozen 0.80/0.05 gates"
        )

    medical_row = by_phase["corrected_medical_one_step"]
    if not (
        finite_number(medical_row, "objective_delta") > 0.0
        and finite_number(medical_row, "loss_delta") < 0.0
    ):
        raise SamplerRefreshCleanupError(
            "metrics Medical objective/loss direction gate failed"
        )

    null_row = by_phase["real_base_teacher_null_update"]
    if (
        finite_number(null_row, "advantage_max_abs") != 0.0
        or finite_number(null_row, "parameter_delta_norm") != 0.0
    ):
        raise SamplerRefreshCleanupError("metrics real Base null gate is nonzero")

    sampler_row = by_phase["sampler_refresh"]
    trainer_gap = finite_number(sampler_row, "trainer_reload_max_gap")
    live_gap = finite_number(sampler_row, "live_fresh_max_gap")
    for key in ("generation_direct_max_gap", "refresh_latency_seconds"):
        if finite_number(sampler_row, key) < 0.0:
            raise SamplerRefreshCleanupError(f"metrics field cannot be negative: {key}")
    if (
        trainer_gap > 1.0e-4
        or live_gap > 1.0e-4
        or sampler_row.get("gate_result") != "pass"
        or sampler_row.get("stale_request_rejected") is not True
    ):
        raise SamplerRefreshCleanupError(
            "metrics same-path/stale sampler refresh gate failed"
        )
    return rows


def _validate_medical_null_evidence(
    medical: Mapping[str, Any], null: Mapping[str, Any]
) -> None:
    def finite(value: Any) -> bool:
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def sha(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    advantage = medical.get("advantage")
    if not (
        medical.get("status") == "pass"
        and medical.get("hard_gate_passed") is True
        and medical.get("finite") is True
        and isinstance(advantage, Mapping)
        and all(finite(advantage.get(key)) for key in ("mean", "std", "min", "max"))
        and all(
            type(advantage.get(key)) is int and advantage[key] >= 0
            for key in ("positive_count", "negative_count", "near_zero_count")
        )
        and sum(
            int(advantage[key])
            for key in ("positive_count", "negative_count", "near_zero_count")
        )
        > 0
        and finite(advantage.get("near_zero_threshold"))
        and float(advantage["near_zero_threshold"]) >= 0
        and all(
            finite(medical.get(key))
            for key in (
                "objective_before",
                "objective_after",
                "loss_before",
                "loss_after",
                "alignment",
                "gradient_norm_before_clip",
                "gradient_norm_after_clip",
                "parameter_delta_norm",
            )
        )
        and medical["objective_after"] > medical["objective_before"]
        and medical["loss_after"] < medical["loss_before"]
        and medical["alignment"] > 0
        and medical["gradient_norm_before_clip"] > 0
        and medical["gradient_norm_after_clip"] > 0
        and medical["parameter_delta_norm"] > 0
        and all(
            type(medical.get(key)) is int and medical[key] > 0
            for key in (
                "gradient_nonzero_tensor_count",
                "parameter_delta_nonzero_tensor_count",
                "trainable_tensor_count",
            )
        )
        and medical.get("teacher_gradient_parameters") == []
        and medical.get("base_gradient_parameters") == []
        and medical.get("base_parameter_versions_unchanged") is True
        and sha(medical.get("trainer_v1_ordered_tensor_sha"))
        and medical.get("trainer_v1_ordered_tensor_sha")
        == medical.get("saved_reload_ordered_tensor_sha")
        and sha(medical.get("saved_adapter_file_sha"))
    ):
        raise SamplerRefreshCleanupError("Medical/null evidence contract is incomplete")
    if not (
        null.get("status") == "pass"
        and null.get("hard_gate_passed") is True
        and null.get("finite") is True
        and all(
            finite(null.get(key)) and float(null[key]) == 0.0
            for key in (
                "objective_before",
                "objective_after",
                "loss_before",
                "loss_after",
                "advantage_max_abs",
                "gradient_norm",
                "parameter_delta_norm",
            )
        )
        and sha(null.get("adapter_ordered_tensor_sha_before"))
        and null.get("adapter_ordered_tensor_sha_before")
        == null.get("adapter_ordered_tensor_sha_after")
        and null.get("teacher_gradient_parameters") == []
        and null.get("base_gradient_parameters") == []
    ):
        raise SamplerRefreshCleanupError("Medical/null evidence contract is incomplete")


def _persist_cleanup_failure(
    output: Path,
    *,
    run_id: str,
    phase: str,
    error_type: str,
    error: str,
    failure_status: str = "failed_cleanup",
) -> None:
    if (output / "sampler_refresh.json").is_file():
        persist_sampler_refresh_failure_binding(
            output,
            run_id=run_id,
            failed_phase=phase,
            failure_status=failure_status,
            error_type=error_type,
            error=error,
        )
    else:
        persist_sampler_refresh_runtime_failure(
            output,
            run_id=run_id,
            failed_phase=phase,
            failure_status=failure_status,
            error_type=error_type,
            error=error,
            correction_metrics={"status": "not_run_or_not_observed"},
            one_step_metrics={"status": "not_run_or_not_observed"},
            null_metrics={"status": "not_run_or_not_observed"},
        )
    _atomic_json(
        output / "readiness.json",
        {
            "schema_version": 4,
            "status": failure_status,
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "B2_started": False,
        },
    )
    _atomic_json(
        output / "summary.json",
        {
            "schema_version": 4,
            "status": failure_status,
            "return_to_cpu_decision": True,
            "B2_authorized": False,
            "B2_started": False,
        },
    )
    _write_index(output, status="failed_cleanup_index")


def finalize_post_exit_cleanup(
    config: Mapping[str, Any],
    *,
    runtime_exit_code: int,
    query: Any = subprocess.run,
) -> dict[str, Any]:
    output = Path(config["run"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        state = _resource_state(query)
    except Exception as error:
        cleanup = {
            "schema_version": 4,
            "status": "fail",
            "query_completed": False,
            "runtime_exit_code": int(runtime_exit_code),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(output / "resource_cleanup.json", cleanup)
        _persist_cleanup_failure(
            output,
            run_id=config["run"]["run_id"],
            phase="post_exit_cleanup",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise SamplerRefreshCleanupError("P4.4 resource query failed") from error
    clean = bool(
        len(state["gpu_memory_used_mib"]) == int(config["resources"]["required_gpus"])
        and all(value == 0 for value in state["gpu_memory_used_mib"])
        and not state["gpu_compute_processes"]
        and not state["project_workers"]
    )
    cleanup = {
        "schema_version": 4,
        "status": "pass" if clean else "fail",
        "runtime_exit_code": int(runtime_exit_code),
        "query_completed": True,
        **state,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "resource_cleanup.json", cleanup)
    if not clean:
        _persist_cleanup_failure(
            output,
            run_id=config["run"]["run_id"],
            phase="post_exit_cleanup",
            error_type="SamplerRefreshCleanupError",
            error="GPU memory/process cleanup gate failed",
        )
        raise SamplerRefreshCleanupError("P4.4 cleanup failed")

    if runtime_exit_code != 0:
        existing_failure = (
            _read(output / "failure.json")
            if (output / "failure.json").is_file()
            else {}
        )
        if (output / "sampler_refresh.json").is_file() and existing_failure.get(
            "sampler_refresh_sha256"
        ) != _sha256(output / "sampler_refresh.json"):
            persist_sampler_refresh_failure_binding(
                output,
                run_id=config["run"]["run_id"],
                failed_phase=str(existing_failure.get("phase", "runtime_exit")),
                failure_status=str(
                    existing_failure.get("status", "failed_artifact_integrity")
                ),
                error_type=str(existing_failure.get("error_type", "RuntimeError")),
                error=str(existing_failure.get("error", f"runtime exited {runtime_exit_code}")),
            )
        elif not existing_failure:
            persist_sampler_refresh_runtime_failure(
                output,
                run_id=config["run"]["run_id"],
                failed_phase="runtime_exit",
                failure_status="failed_artifact_integrity",
                error_type="RuntimeError",
                error=f"runtime exited {runtime_exit_code}",
                correction_metrics={"status": "not_run_or_not_observed"},
                one_step_metrics={"status": "not_run_or_not_observed"},
                null_metrics={"status": "not_run_or_not_observed"},
            )
        readiness = {
            "schema_version": 4,
            "status": _read(output / "failure.json").get(
                "status", "failed_artifact_integrity"
            ),
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "B2_started": False,
            "cleanup_ready": True,
        }
        _atomic_json(output / "readiness.json", readiness)
        summary = {
            **readiness,
            "return_to_cpu_decision": True,
        }
        _atomic_json(output / "summary.json", summary)
        index = _write_index(output, status="failed_run_evidence_index")
        return {
            **summary,
            "artifact_index_sha256": _sha256(output / "artifact_index.json"),
            "indexed_artifacts": len(index["artifacts"]),
        }

    run_id = str(config["run"]["run_id"])
    try:
        missing_files = [
            name for name in required_success_artifacts() if not (output / name).is_file()
        ]
        if missing_files:
            raise SamplerRefreshCleanupError(
                "required success artifacts are missing: " + ", ".join(missing_files)
            )
        root = Path(__file__).resolve().parents[2]
        schema_path = root / str(config["artifacts"]["schema_path"])
        if (
            not schema_path.is_file()
            or _sha256(schema_path) != config["artifacts"]["schema_sha256"]
        ):
            raise SamplerRefreshCleanupError("sampler artifact schema SHA mismatch")
        provenance = _read(output / "generation_provenance_probe.json")
        v0_controls = _read(output / "sampler_v0_controls.json")
        rollout = _read(output / "fresh_trajectory_manifest.json")
        optional_32 = _read(output / "optional_32_prompt_rung.json")
        correction_name = (
            "correction_calibration_32.json"
            if optional_32.get("status")
            == "run_completed_due_preregistered_instability"
            else "correction_calibration_16.json"
        )
        correction = _read(output / correction_name)
        medical = _read(output / "corrected_medical_one_step.json")
        null = _read(output / "real_base_teacher_null_update.json")
        sampler = _read(output / "sampler_refresh.json")
        release = _read(output / "runtime_release.json")
        launch = _read(output / "launch_record.json")
        metadata = _read(output / "metadata.json")
        data_manifest = _read(output / "data_manifest.json")
        cost = _read(output / "cost.json")
        checkpoints = _read(output / "checkpoints/index.json")
        sampler_observations = _read(output / "sampler_refresh_observations.json")
        saved_config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        jsonschema.validate(
            sampler,
            json.loads(schema_path.read_text(encoding="utf-8")),
        )
        metric_rows = _read_metrics(output, run_id=run_id)
        _validate_medical_null_evidence(medical, null)
        if any(
            row.get("status") != "pass"
            for row in metric_rows
            if row.get("phase")
            in {
                "correction_calibration_16",
                "correction_calibration_32",
                "corrected_medical_one_step",
                "real_base_teacher_null_update",
                "sampler_refresh",
            }
        ):
            raise SamplerRefreshCleanupError("aggregate metrics contain a non-pass phase")
        trajectory_path = output / "fresh_full_support_trajectory.jsonl"
        trajectory_sha = _sha256(trajectory_path)
        if not (
            isinstance(saved_config, Mapping)
            and saved_config.get("run", {}).get("run_id") == run_id
            and launch.get("run_id") == metadata.get("run_id") == data_manifest.get("run_id")
            == checkpoints.get("run_id") == run_id
            and data_manifest.get("labels_accessed") is False
            and data_manifest.get("final_accessed") is False
            and cost.get("platform_actual_cost_cny") is None
            and cost.get("actual_cost_cny") is None
            and cost.get("B2_started") is False
            and checkpoints.get("checkpoints") == []
            and provenance.get("status") == "pass"
            and v0_controls.get("status") == "pass"
            and v0_controls.get("hard_gate_passed") is True
            and rollout.get("status") == "fresh_full_support"
            and rollout.get("run_id") == run_id
            and rollout.get("trajectory_sha256") == trajectory_sha
            and correction.get("trajectory_sha256") == trajectory_sha
            and correction.get("calibration_readiness", {}).get("calibration_ready")
            is True
            and medical.get("status") == "pass"
            and medical.get("hard_gate_passed") is True
            and medical.get("run_id") == run_id
            and medical.get("trajectory_sha256") == trajectory_sha
            and null.get("status") == "pass"
            and null.get("hard_gate_passed") is True
            and release.get("status") == "pass"
            and release.get("models_released") is True
            and sampler_observations.get("sampler_refresh_sha256")
            == _sha256(output / "sampler_refresh.json")
            and config.get("isolation")
            == {
                "final_access": False,
                "controller_access": False,
                "confirmation_access": False,
                "label_access": False,
            }
        ):
            raise SamplerRefreshCleanupError("standard artifact identity/readiness mismatch")
        artifacts = {
            "run_id": run_id,
            "provenance": {"status": "pass"},
            "rollout": {"status": "pass"},
            "correction": {"status": "pass"},
            "medical_one_step": {"status": "pass"},
            "null_update": {"status": "pass"},
            "sampler_refresh": sampler,
            "cleanup": cleanup,
            "metrics_nonempty": True,
            "artifact_integrity": False,
            "isolation": dict(config["isolation"]),
            "b2_backend_binding": config.get("b2_backend_binding"),
        }
        preliminary = derive_readiness_from_artifacts(artifacts)
        if (
            preliminary["sampler_refresh_ready"] is not True
            or preliminary["cleanup_ready"] is not True
            or preliminary["final_label_isolation"] is not True
        ):
            raise SamplerRefreshCleanupError(
                "sampler/cleanup/isolation recomputation failed: "
                + "; ".join(preliminary["sampler_refresh_recomputed_failures"])
            )
        candidate = {
            **preliminary,
            "status": "pending_authoritative_readiness_after_index",
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "B2_started": False,
            "artifact_integrity_ready": False,
            "resource_cleanup_verified": True,
            "automatically_started_b2": False,
            "authoritative_readiness_path": "readiness.json",
            "next_step": "verify_index_then_write_authoritative_readiness",
        }
        _atomic_json(output / "summary.json", candidate)
        index = _write_index(output, status="complete_success_index")
        _verify_index(output, index)
        artifacts["artifact_integrity"] = True
        readiness = derive_readiness_from_artifacts(artifacts)
        if readiness["OPD_scoring_backend_ready"] is not True:
            raise SamplerRefreshCleanupError("final readiness recomputation failed")
        readiness.update(
            {
                "resource_cleanup_verified": True,
                "automatically_started_b2": False,
                "artifact_index_sha256": _sha256(output / "artifact_index.json"),
            }
        )
        # Authoritative success/authorization is the final atomic write. No
        # artifact mutation is permitted after this point.
        _atomic_json(output / "readiness.json", readiness)
        return {
            **readiness,
            "indexed_artifacts": len(index["artifacts"]),
        }
    except Exception as error:
        _persist_cleanup_failure(
            output,
            run_id=run_id,
            phase="post_exit_artifact_readiness",
            failure_status="failed_artifact_integrity",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise SamplerRefreshCleanupError("P4.4 readiness failed closed") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.4 post-exit cleanup")
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
