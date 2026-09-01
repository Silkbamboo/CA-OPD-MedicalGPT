"""Post-exit cleanup and artifact-derived readiness for the P4.5 GPU micro."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.opd.production_backend_binding_v5 import verify_b2_backend_binding
from src.opd.production_sampler_identity_v5 import rebuild_aggregate_tensor_sha


class ProductionSamplerMicroCleanupError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _identity_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        config_sha = value.get("canonical_config_sha256")
        if not (
            isinstance(config_sha, str)
            and len(config_sha) == 64
            and all(character in "0123456789abcdef" for character in config_sha)
        ):
            return False
        rebuilt = rebuild_aggregate_tensor_sha(value.get("tensors", []))
    except Exception:
        return False
    return bool(
        rebuilt == value.get("aggregate_tensor_sha256")
        and value.get("tensor_count") == len(value.get("tensors", []))
        and value.get("total_canonical_bytes")
        == sum(item.get("canonical_byte_length", -1) for item in value.get("tensors", []))
    )


def _same_tensor_identity(*values: Mapping[str, Any]) -> bool:
    if not all(_identity_valid(value) for value in values):
        return False
    comparable = []
    for value in values:
        comparable.append(
            (
                value.get("canonical_config_sha256"),
                value.get("aggregate_tensor_sha256"),
                [
                    (
                        item.get("canonical_key"),
                        item.get("sha256"),
                        item.get("shape"),
                        item.get("canonical_dtype"),
                        item.get("canonical_byte_length"),
                    )
                    for item in value.get("tensors", [])
                ],
            )
        )
    return all(item == comparable[0] for item in comparable[1:])


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _required_runtime_evidence(directory: Path) -> tuple[bool, bool]:
    try:
        correction = json.loads(
            (directory / "four_prompt_correction.json").read_text(encoding="utf-8")
        )
        one_step = json.loads(
            (directory / "four_prompt_corrected_one_step.json").read_text(
                encoding="utf-8"
            )
        )
        v0 = json.loads(
            (directory / "v0_repeat_control.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False, False

    q_old = correction.get("q_p_old_metrics")
    correction_complete = bool(
        correction.get("status") == "pass"
        and correction.get("prompt_count") == 4
        and isinstance(correction.get("rollout_token_count"), int)
        and correction["rollout_token_count"] > 0
        and correction.get("finite") is True
        and isinstance(q_old, Mapping)
        and q_old.get("finite_rate") == 1.0
        and all(_finite_number(q_old.get(name)) for name in ("mae", "p95", "max"))
        and _finite_number(correction.get("ess_fraction"))
        and correction["ess_fraction"] >= 0.80
        and _finite_number(correction.get("cap_fraction"))
        and correction["cap_fraction"] <= 0.05
        and correction.get("correction_weight_requires_grad") is False
    )
    advantage_numbers = (
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
    )
    advantage_counts = (
        "advantage_positive_count",
        "advantage_negative_count",
        "advantage_near_zero_count",
    )
    trainable = one_step.get("trainable_tensor_count")
    nonzero = one_step.get("nonzero_updated_tensor_count")
    one_step_complete = bool(
        one_step.get("status") == "pass"
        and all(_finite_number(one_step.get(name)) for name in advantage_numbers)
        and all(
            isinstance(one_step.get(name), int) and one_step[name] >= 0
            for name in advantage_counts
        )
        and sum(one_step[name] for name in advantage_counts) > 0
        and _finite_number(one_step.get("objective_before"))
        and _finite_number(one_step.get("objective_after"))
        and one_step["objective_after"] > one_step["objective_before"]
        and _finite_number(one_step.get("loss_before"))
        and _finite_number(one_step.get("loss_after"))
        and one_step["loss_after"] < one_step["loss_before"]
        and _finite_number(one_step.get("alignment"))
        and one_step["alignment"] > 0
        and _finite_number(one_step.get("parameter_delta_norm"))
        and one_step["parameter_delta_norm"] > 0
        and isinstance(trainable, int)
        and isinstance(nonzero, int)
        and 0 < nonzero <= trainable
        and one_step.get("teacher_gradient_parameters") == []
        and one_step.get("base_gradient_parameters") == []
    )

    repeat = v0.get("same_instance_repeat")
    normal_v0 = v0.get("normal_v0_request")
    wrong_authority = v0.get("wrong_authority_request")
    v0_complete = bool(
        v0.get("status") == "pass"
        and v0.get("trainer_saved_runtime_identity_gate") is True
        and isinstance(repeat, Mapping)
        and repeat.get("finite_rate") == 1.0
        and _finite_number(repeat.get("max"))
        and repeat["max"] <= 0.0001
        and isinstance(normal_v0, Mapping)
        and normal_v0.get("accepted") is True
        and normal_v0.get("scoring_executed") is True
        and isinstance(wrong_authority, Mapping)
        and wrong_authority.get("rejected") is True
        and wrong_authority.get("error_code")
        in {"STALE_SAMPLER_IDENTITY", "SAMPLER_RUNTIME_TENSOR_MISMATCH"}
        and wrong_authority.get("scoring_executed") is False
        and wrong_authority.get("generation_executed") is False
        and wrong_authority.get("rejection_phase") == "identity_guard_before_forward"
    )
    return correction_complete and one_step_complete, v0_complete


def derive_readiness_from_artifacts(
    output: str | Path,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
    cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root)
    directory = Path(output)
    refresh_path = directory / "production_sampler_refresh.json"
    metrics_path = directory / "metrics.jsonl"
    if not refresh_path.is_file() or not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise ProductionSamplerMicroCleanupError("required refresh/metrics artifacts are absent")
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    binding = verify_b2_backend_binding(
        config["production_binding"]["b2_config_path"],
        config["production_binding"]["b2_run_card_path"],
        repo_root=root,
    )
    trainer = refresh.get("trainer_identity")
    runtime = refresh.get("runtime_identity")
    fresh = refresh.get("fresh_identity")
    identity_match = _same_tensor_identity(trainer, runtime, fresh)
    normal = refresh.get("normal_request")
    stale = refresh.get("stale_request")
    metrics = refresh.get("same_path_metrics")
    trainer_manifest_path = directory / "trainer_v1_authority_manifest.json"
    authority_manifest_valid = bool(
        trainer_manifest_path.is_file()
        and refresh.get("authoritative_manifest_sha256") == _sha256(trainer_manifest_path)
    )
    isolation_closed = refresh.get("isolation") == {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    registry = refresh.get("registry_after")
    registry_ok = bool(
        isinstance(registry, Mapping)
        and registry.get("adapter_count") == 1
        and registry.get("peft_config_names") == ["student_active"]
        and registry.get("active_adapter") == "student_active"
        and registry.get("adapters_enabled") is True
        and registry.get("merged") is False
    )
    reconstruction_complete, v0_guard_complete = _required_runtime_evidence(directory)
    gates = {
        "production_backend_bound": bool(
            binding["binding_verified"]
            and binding["production_backend"]["backend_id"]
            == config["production_binding"]["backend_id"]
        ),
        "authoritative_sha_verified": bool(
            isinstance(trainer, Mapping)
            and isinstance(runtime, Mapping)
            and authority_manifest_valid
            and runtime.get("aggregate_tensor_sha256") == trainer.get("aggregate_tensor_sha256")
        ),
        "runtime_sha_match": identity_match,
        "per_tensor_match": identity_match,
        "same_path_gap_passed": bool(
            isinstance(metrics, Mapping)
            and metrics.get("finite_rate") == 1.0
            and isinstance(metrics.get("max"), (int, float))
            and metrics["max"] <= 0.0001
            and metrics.get("threshold") == 0.0001
        ),
        "normal_request_passed": bool(
            isinstance(normal, Mapping)
            and normal.get("accepted") is True
            and normal.get("scoring_executed") is True
            and normal.get("silent_fallback") is False
        ),
        "stale_request_rejected": bool(
            isinstance(stale, Mapping)
            and stale.get("rejected") is True
            and stale.get("scoring_executed") is False
            and stale.get("generation_executed") is False
            and stale.get("error_code") == "STALE_SAMPLER_IDENTITY"
            and stale.get("rejection_phase") == "identity_guard_before_forward"
        ),
        "v1_reconstruction_evidence_complete": reconstruction_complete,
        "sampler_v0_guard_complete": v0_guard_complete,
        "registry_stable": registry_ok,
        "artifacts_complete": True,
        "cleanup_complete": bool(
            cleanup.get("status") == "pass"
            and cleanup.get("runtime_exit_code") == 0
            and cleanup.get("gpu_memory_used_mib") == [0, 0]
            and cleanup.get("compute_pids") == []
            and cleanup.get("residual_workers") == []
        ),
        "isolation_closed": isolation_closed,
    }
    ready = bool(
        refresh.get("status") == "pass"
        and refresh.get("gate_result") == "pass"
        and refresh.get("failure_reason") is None
        and all(gates.values())
    )
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    return {
        "schema_version": 5,
        "readiness_kind": "production_sampler_refresh_micro_v5",
        "run_id": config["run"]["run_id"],
        "production_backend_id": binding["production_backend"]["backend_id"],
        "b2_config_sha256": binding["config_sha256"],
        "b2_run_card_sha256": binding["run_card_sha256"],
        "ready": ready,
        "production_sampler_refresh_ready": ready,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "gate_result": "pass" if ready else "fail",
        "failure_reason": (
            None
            if ready
            else "artifact_derived_gates_failed:" + ",".join(failed_gates)
        ),
        "gates": gates,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
        "production_sampler_refresh_sha256": _sha256(refresh_path),
        "metrics_sha256": _sha256(metrics_path),
        "cleanup_sha256": _sha256(directory / "resource_cleanup.json"),
    }


def _gpu_state() -> tuple[list[int], list[dict[str, Any]]]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    memory = [int(line.strip()) for line in query.stdout.splitlines() if line.strip()]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    pids: list[dict[str, Any]] = []
    for line in processes.splitlines():
        if not line.strip():
            continue
        pid, name = [item.strip() for item in line.split(",", 1)]
        pids.append({"pid": int(pid), "process_name": name})
    return memory, pids


def finalize(
    config: Mapping[str, Any],
    *,
    runtime_exit_code: int,
    repo_root: str | Path,
    gpu_state: tuple[list[int], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    output = Path(config["run"]["output_dir"])
    memory, processes = gpu_state if gpu_state is not None else _gpu_state()
    residual = subprocess.run(
        ["pgrep", "-af", "ray|vllm|verl|torchrun"],
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    residual = [line for line in residual if line.strip() and "pgrep -af" not in line]
    cleanup = {
        "schema_version": 5,
        "run_id": config["run"]["run_id"],
        "status": "pass" if memory == [0, 0] and not processes and not residual else "fail",
        "runtime_exit_code": int(runtime_exit_code),
        "models_released": True,
        "gpu_memory_used_mib": memory,
        "compute_pids": processes,
        "residual_workers": residual,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "resource_cleanup.json", cleanup)
    readiness = derive_readiness_from_artifacts(
        output, config, repo_root=repo_root, cleanup=cleanup
    )
    _atomic_json(output / "readiness.json", readiness)

    metrics_path = output / "metrics.jsonl"
    refresh_path = output / "production_sampler_refresh.json"
    failure_path = output / "failure.json"
    if not readiness["ready"]:
        artifact_integrity_failure = bool(
            runtime_exit_code == 0
            and cleanup["status"] == "pass"
            and (
                not readiness["gates"]["v1_reconstruction_evidence_complete"]
                or not readiness["gates"]["sampler_v0_guard_complete"]
            )
        )
        expected = {
            "schema_version": 5,
            "run_id": config["run"]["run_id"],
            "status": (
                "failed_artifact_integrity"
                if artifact_integrity_failure
                else "failed_production_sampler_refresh"
            ),
            "failure_layer": (
                "artifact_integrity"
                if artifact_integrity_failure
                else "post_exit_artifact_readiness_cleanup"
            ),
            "error_type": (
                "ProductionSamplerMicroArtifactIntegrityError"
                if artifact_integrity_failure
                else "ProductionSamplerMicroReadinessError"
            ),
            "error": readiness["failure_reason"],
            "production_sampler_refresh_sha256": _sha256(refresh_path),
            "metrics_sha256": _sha256(metrics_path),
            "B2_authorized": False,
        }
        if not failure_path.is_file() or any(
            json.loads(failure_path.read_text(encoding="utf-8")).get(key) != value
            for key, value in expected.items()
        ):
            _atomic_json(failure_path, expected)
    terminal_status = (
        "passed_production_sampler_refresh_micro"
        if readiness["ready"]
        else (
            "failed_artifact_integrity"
            if runtime_exit_code == 0
            and cleanup["status"] == "pass"
            and (
                not readiness["gates"]["v1_reconstruction_evidence_complete"]
                or not readiness["gates"]["sampler_v0_guard_complete"]
            )
            else "failed_production_sampler_refresh"
        )
    )
    _atomic_json(
        output / "summary.json",
        {
            "schema_version": 5,
            "run_id": config["run"]["run_id"],
            "status": terminal_status,
            "production_sampler_refresh_ready": readiness["ready"],
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "B2_started": False,
            "failure_reason": readiness["failure_reason"],
        },
    )
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    index = {
        "schema_version": 5,
        "run_id": config["run"]["run_id"],
        "status": "pass" if readiness["ready"] else "fail",
        "artifacts": artifacts,
        "historical_artifacts_modified": False,
        "B2_authorized": False,
    }
    _atomic_json(output / "artifact_index.json", index)
    return {"cleanup": cleanup, "readiness": readiness, "artifact_index": index}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.5 production sampler micro cleanup")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = finalize(
        config,
        runtime_exit_code=args.runtime_exit_code,
        repo_root=Path(__file__).resolve().parents[2],
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["cleanup"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
