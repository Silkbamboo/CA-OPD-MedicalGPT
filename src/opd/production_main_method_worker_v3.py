"""Authorized 120-commit P6 IDT/CA worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p6_controller_runtime import score_loaded_controller_metrics
from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_worker_v1 import (
    DERIVED_COST_CAP_CNY,
    REFERENCE_PRICE_CNY_PER_HOUR,
    _append_jsonl,
    _atomic_json,
    _existing_records,
)
from src.opd.production_b2_formal_worker_v2 import checkpoint_index_v2
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_main_method_package_v3 import verify_method_package_v3
from src.opd.production_main_method_v3 import P6FormalMethodError


REPO = Path(__file__).resolve().parents[2]
CONTROLLER_CONFIG = REPO / "configs/eval/qwen3_4b/p6_controller_v1.json"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise P6FormalMethodError("formal method worker JSON differs")
    return dict(value)


def _live_adapter_identity(session: Any) -> dict[str, Any]:
    path = Path(session._current_checkpoint_path).resolve()
    manifest = path / "adapter_transport_manifest.json"
    if not path.is_dir() or not manifest.is_file():
        raise P6FormalMethodError("CA live Controller adapter export is absent")
    return {
        "adapter_path": str(path),
        "adapter_ordered_sha256": _ordered_adapter_sha256(path),
        "adapter_weight_sha256": _sha_file(path / "adapter_model.safetensors"),
        "adapter_manifest_sha256": _sha_file(manifest),
    }


def _ca_controller_window(
    session: Any, *, output: Path, completed_step: int
) -> dict[str, Any]:  # pragma: no cover - GPU
    adapter = _live_adapter_identity(session)
    root = output / "controller_windows" / f"step_{completed_step:03d}"
    cache = root / "prediction_cache"
    if root.exists() or root.is_symlink():
        raise P6FormalMethodError("CA Controller window output must be fresh")
    root.mkdir(parents=True)
    session.student_model.to("cuda:0")
    session.student_model.eval()
    result = score_loaded_controller_metrics(
        model=session.student_model,
        tokenizer=session.tokenizer,
        config_path=CONTROLLER_CONFIG,
        adapter_identity=adapter,
        cache_path=cache,
    )
    metrics = result["metrics"]
    decision = session.update_ca_controller(
        medical_accuracy=float(metrics["medical_accuracy"]),
        general_accuracy=float(metrics["general_micro_accuracy"]),
        completed_step=completed_step,
    )
    report = {
        "schema_version": 3,
        "artifact_kind": "p6_ca_controller_window_v3",
        "completed_optimizer_step": completed_step,
        "adapter_identity": adapter,
        "prediction_sha256": result["prediction_sha256"],
        "cache_manifest_sha256": result["cache_manifest_sha256"],
        "repeatability": result["repeatability"],
        "metrics": metrics,
        "router_decision": decision,
        "labels_opened_after_predictions_frozen": True,
        "controller_access_count": 1,
        "final_access_count": 0,
    }
    _atomic_json(root / "controller_window.json", report)
    return report


def run_method_worker_v3(
    package: Path,
    *,
    target_step: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:  # pragma: no cover - GPU
    if target_step != 120:
        raise P6FormalMethodError("P6 formal method worker is fixed to 120 commits")
    preflight = verify_method_package_v3(
        package, require_clean_git=True
    )
    config = _json(Path(package) / "formal_method_config.json")
    schedule = _json(Path(package) / "prompt_schedule.json")
    authority = _json(Path(package) / "data_authority.json")
    index = _json(Path(package) / "package_index.json")
    runtime = formal_b2_runtime_config_v2(config)
    runtime["formal_method_v3"] = config["formal_method_v3"]
    # The legacy compatibility projection intentionally carries only the
    # Teacher path.  Restore the already package-validated identity ledger so
    # v3 can fail closed on manifest and weight SHA before model loading.
    runtime["teacher"] = config["teacher"]
    output = Path(str(config["run"]["output_dir"]))
    method_id = str(config["formal_method_v3"]["method_id"])
    if resume_checkpoint is None:
        output.mkdir(parents=True)
        for name in (
            "b2_steps",
            "checkpoints",
            "formal_steps",
            "method_steps_v3",
            "memory_step_audits",
            "memory_telemetry/markers",
            "ratio_evidence_v2",
            "rejected_updates_v2",
            "controller_windows",
        ):
            (output / name).mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "package_preflight_v3.json", preflight)
        _atomic_json(output / "runtime_config_v3.json", runtime)
        start_step = 0
        records: list[dict[str, Any]] = []
    else:
        manifest = validate_formal_checkpoint(resume_checkpoint)
        start_step = int(manifest["logical_version"])
        if not (start_step < target_step and start_step % 10 == 0):
            raise P6FormalMethodError("P6 formal method resume boundary differs")
        records = _existing_records(output, through_step=start_step)
    disk = validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    from src.opd.production_b2_formal_gpu_v2 import validate_formal_step_health_v2
    from src.opd.production_main_method_gpu_v3 import FormalMethodSessionV3

    session: FormalMethodSessionV3 | None = None
    started = time.time()
    process_id = f"pid-{os.getpid()}-{int(started)}"
    try:
        session = FormalMethodSessionV3(
            runtime,
            config_path=Path(package) / "formal_method_config.json",
            route="b2_calibration",
        )
        if resume_checkpoint is None:
            identity = session.initial_calibration_identity()
            _atomic_json(output / "fresh_v0_identity.json", identity)
            if not (
                identity.get("zero_effect_verified") is True
                and identity.get("tensor_count") == 504
                and identity.get("source_adapter_path") is None
            ):
                raise P6FormalMethodError("formal method model load is not fresh-v0")
        else:
            prompt_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=start_step
            )
            resume = session.restore_formal_checkpoint_v1(
                resume_checkpoint,
                package_content_sha256=str(index["package_content_sha256"]),
                config_sha256=str(index["config_sha256"]),
                manifest_sha256=str(index["manifest_sha256"]),
                schedule_sha256=str(index["schedule_semantic_sha256"]),
                resume_prompt_rows=prompt_rows,
            )
            _atomic_json(
                output / f"resume_identity_step_{start_step:03d}_{process_id}.json",
                {**resume, "resume_from": str(resume_checkpoint)},
            )
        initial_registry = session._registry_count()
        initial_models = session._model_count()
        _atomic_json(
            output / f"process_{process_id}.json",
            {
                "schema_version": 3,
                "artifact_kind": "p6_formal_method_process_boundary",
                "method_id": method_id,
                "start_accepted_optimizer_commits": start_step,
                "target_accepted_optimizer_commits": target_step,
                "fresh_v0": resume_checkpoint is None,
                "gpu0_role": "Student",
                "gpu1_role": "sampler_plus_step_scoped_shared_Base_Medical_Teacher",
                "final_access_count": 0,
            },
        )
        print(json.dumps({"event": "model_load_complete", "method": method_id, "disk": disk}), flush=True)
        for step_index in range(start_step, target_step):
            next_step = step_index + 1
            current_disk = validate_disk_safety_v2(
                free_bytes=shutil.disk_usage(output).free,
                full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
                predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
            )
            elapsed_hours = (time.time() - started) / 3600.0
            if elapsed_hours * REFERENCE_PRICE_CNY_PER_HOUR > DERIVED_COST_CAP_CNY:
                raise P6FormalMethodError("formal method reference cost cap exceeded")
            prompt_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=step_index
            )
            record = session.run_formal_method_step_v3(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=1024,
            )
            records.append(record)
            health = validate_formal_step_health_v2(
                records,
                initial_registry_count=initial_registry,
                initial_model_count=initial_models,
            )
            controller = None
            if method_id == "CA-OPD" and next_step in {30, 60, 90, 120}:
                controller = _ca_controller_window(
                    session, output=output, completed_step=next_step
                )
            checkpoint = None
            if next_step % 10 == 0:
                checkpoint = session.seal_registered_checkpoint_v2(
                    logical_version=next_step,
                    package_content_sha256=str(index["package_content_sha256"]),
                    config_sha256=str(index["config_sha256"]),
                    manifest_sha256=str(index["manifest_sha256"]),
                    schedule_sha256=str(index["schedule_semantic_sha256"]),
                    environment={
                        "production_python": str(Path(os.sys.executable).resolve()),
                        "method_id": method_id,
                        "final_access_count": 0,
                    },
                )
                checkpoint_index_v2(output)
            removed = session.release_transient_step_artifacts_v1(next_step)
            after_disk = validate_disk_safety_v2(
                free_bytes=shutil.disk_usage(output).free,
                full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
                predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
            )
            elapsed = time.time() - started
            average = elapsed / (next_step - start_step)
            method_evidence = record["formal_method_v3"]
            progress = {
                "method_id": method_id,
                "accepted_optimizer_commits": next_step,
                "target_accepted_optimizer_commits": target_step,
                "policy_version": record["next_policy_version"],
                "data_cursor": next_step * 4,
                "average_step_seconds_this_process": average,
                "eta_seconds": (target_step - next_step) * average,
                "loss": record["loss"],
                "reverse_kl": record["reverse_kl"],
                "ratio_health_v2": record["ratio_health_v2"],
                "teacher_route_counts": method_evidence["teacher_route_counts"],
                "source_teacher_counts_cumulative": method_evidence["source_teacher_counts_cumulative"],
                "kl_safety_scale_by_teacher_route": method_evidence["kl_safety_scale_by_teacher_route"],
                "controller_window_complete": controller is not None,
                "checkpoint_complete": checkpoint is not None,
                "disk": after_disk,
                "health": health,
                "transient_paths_removed": removed,
                "final_access_count": 0,
            }
            _append_jsonl(output / "progress_v3.jsonl", progress)
            _atomic_json(
                output / "heartbeat.json",
                {**progress, "timestamp_unix": time.time()},
            )
            if next_step in {1, 10, 30, 60, 90, 120}:
                print(json.dumps({"event": "progress", **progress}), flush=True)
        final = {
            "schema_version": 3,
            "artifact_kind": "p6_formal_method_training_status",
            "status": f"{method_id.lower()}_trained_to_120_controller_pending",
            "method_id": method_id,
            "accepted_optimizer_commits": 120,
            "policy_version": 120,
            "data_cursor": 480,
            "checkpoint_index": checkpoint_index_v2(output),
            "route_state": session.formal_route_state(),
            "controller_access_count_during_training": 4 if method_id == "CA-OPD" else 0,
            "final_access_count": 0,
            "elapsed_seconds_this_process": time.time() - started,
            "derived_cost_cny_reference_only": (time.time() - started) / 3600.0 * REFERENCE_PRICE_CNY_PER_HOUR,
            "platform_actual_cost_cny": None,
        }
        _atomic_json(output / "training_status_step_120_v3.json", final)
        return final
    except BaseException as error:
        failure = {
            "schema_version": 3,
            "artifact_kind": "p6_formal_method_failure",
            "method_id": method_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "first_failure_field": str(error).split(":", 1)[0],
            "latest_complete_checkpoint_index": (
                checkpoint_index_v2(output) if output.is_dir() else None
            ),
            "root_cause_status": "unresolved_at_failure_write",
            "final_access_count": 0,
        }
        if output.is_dir():
            _atomic_json(output / f"failure_{process_id}.json", failure)
        raise
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=120)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    result = run_method_worker_v3(
        args.package,
        target_step=args.target_step,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_method_worker_v3"]
