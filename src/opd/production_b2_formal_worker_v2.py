"""Authorized Formal B2 v2 worker counting only transactional commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_checkpoint_v2 import validate_controller_snapshot_v2
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
    verify_formal_package_v2,
)
from src.opd.production_b2_formal_v1 import FormalB2Error
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_b2_formal_worker_v1 import (
    DERIVED_COST_CAP_CNY,
    REFERENCE_PRICE_CNY_PER_HOUR,
    _append_jsonl,
    _atomic_json,
    _existing_records,
    _json,
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_index_v2(output: Path) -> dict[str, Any]:
    full: list[dict[str, Any]] = []
    full_root = output / "formal_checkpoints"
    if full_root.is_dir():
        for path in sorted(full_root.glob("step_*")):
            manifest = validate_formal_checkpoint(path)
            full.append(
                {
                    "logical_version": manifest["logical_version"],
                    "path": str(path.relative_to(output)),
                    "adapter_sha256": manifest["adapter_sha256"],
                    "checkpoint_manifest_sha256": _sha_file(path / "checkpoint_manifest.json"),
                    "data_cursor": manifest["data_cursor"],
                    "complete": True,
                    "resume_eligible": True,
                }
            )
    snapshots: list[dict[str, Any]] = []
    snapshot_root = output / "controller_snapshots"
    if snapshot_root.is_dir():
        for path in sorted(snapshot_root.glob("step_*")):
            manifest = validate_controller_snapshot_v2(path)
            snapshots.append(
                {
                    "logical_version": manifest["logical_version"],
                    "path": str(path.relative_to(output)),
                    "adapter_sha256": manifest["adapter_sha256"],
                    "snapshot_manifest_sha256": _sha_file(path / "snapshot_manifest.json"),
                    "complete": True,
                    "resume_eligible": False,
                    "controller_eligible": True,
                }
            )
    retired = None
    retired_path = output / "retired_resume_checkpoints_v2.json"
    if retired_path.is_file():
        retired = dict(_json(retired_path))
    result = {
        "schema_version": 2,
        "artifact_kind": "p5_1_formal_b2_checkpoint_index_v2",
        "retention": "latest_two_full_resume_plus_registered_adapter_snapshots",
        "full_resume_checkpoints": full,
        "controller_adapter_snapshots": snapshots,
        "retired_resume_checkpoints": retired,
    }
    _atomic_json(output / "checkpoint_index_v2.json", result)
    return result


def run_worker_v2(
    package: Path,
    *,
    target_step: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if target_step != 120:
        raise FormalB2Error("P5.1 Formal B2 v2 worker is authorized only to step120")
    preflight = verify_formal_package_v2(
        package, allow_existing_output_for_resume=resume_checkpoint is not None
    )
    config = _json(package / "formal_b2_config.json")
    schedule = _json(package / "prompt_schedule.json")
    authority = _json(package / "data_authority.json")
    environment = _json(package / "environment.json")
    index = _json(package / "package_index.json")
    runtime = formal_b2_runtime_config_v2(config)
    output = Path(str(config["run"]["output_dir"]))
    if resume_checkpoint is None:
        output.mkdir(parents=True)
        for name in (
            "b2_steps",
            "checkpoints",
            "formal_steps",
            "memory_step_audits",
            "memory_telemetry/markers",
            "ratio_evidence_v2",
            "rejected_updates_v2",
        ):
            (output / name).mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "package_preflight_v2.json", preflight)
        _atomic_json(output / "runtime_config_v2.json", runtime)
        _atomic_json(output / "environment.json", environment)
        start_step = 0
        records: list[dict[str, Any]] = []
    else:
        manifest = validate_formal_checkpoint(resume_checkpoint)
        start_step = int(manifest["logical_version"])
        if not (start_step < target_step and start_step % 10 == 0):
            raise FormalB2Error("Formal B2 v2 resume boundary differs")
        records = _existing_records(output, through_step=start_step)
    disk = validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )

    from src.opd.production_b2_formal_gpu_v2 import (
        FormalB2SessionV2,
        validate_formal_step_health_v2,
    )

    session: FormalB2SessionV2 | None = None
    started = time.time()
    process_id = f"pid-{os.getpid()}-{int(started)}"
    try:
        session = FormalB2SessionV2(
            runtime,
            config_path=package / "formal_b2_config.json",
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
                raise FormalB2Error("Formal B2 v2 model load is not fresh-v0")
            resume_evidence = None
        else:
            resume_prompt_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=start_step
            )
            resume_evidence = session.restore_formal_checkpoint_v1(
                resume_checkpoint,
                package_content_sha256=str(index["package_content_sha256"]),
                config_sha256=str(index["config_sha256"]),
                manifest_sha256=str(index["manifest_sha256"]),
                schedule_sha256=str(index["schedule_semantic_sha256"]),
                resume_prompt_rows=resume_prompt_rows,
            )
            _atomic_json(
                output / f"resume_identity_step_{start_step:03d}_{process_id}.json",
                {**resume_evidence, "resume_from": str(resume_checkpoint), "process_boundary": process_id},
            )
        initial_registry = session._registry_count()
        initial_models = session._model_count()
        _atomic_json(
            output / f"process_{process_id}.json",
            {
                "schema_version": 2,
                "artifact_kind": "formal_b2_v2_process_boundary",
                "process_id": process_id,
                "resume_from": None if resume_checkpoint is None else str(resume_checkpoint),
                "start_accepted_optimizer_commits": start_step,
                "target_accepted_optimizer_commits": target_step,
                "model_load_complete": True,
                "fresh_v0": resume_checkpoint is None,
                "gpu0_role": "Student",
                "gpu1_role": "long_lived_sampler_and_step_scoped_Medical_Teacher",
                "trainable_tensor_count": 504,
                "initial_registry_count": initial_registry,
                "initial_model_count": initial_models,
                "final_access_count": 0,
            },
        )
        print(
            json.dumps(
                {
                    "event": "model_load_complete",
                    "start_step": start_step,
                    "target_step": target_step,
                    "disk": disk,
                    "registry_count": initial_registry,
                    "model_count": initial_models,
                }
            ),
            flush=True,
        )
        for step_index in range(start_step, target_step):
            next_step = step_index + 1
            current_disk = validate_disk_safety_v2(
                free_bytes=shutil.disk_usage(output).free,
                full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
                predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
            )
            elapsed_hours = (time.time() - started) / 3600.0
            if elapsed_hours * REFERENCE_PRICE_CNY_PER_HOUR > DERIVED_COST_CAP_CNY:
                raise FormalB2Error("Formal B2 v2 derived reference cost cap exceeded")
            prompt_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=step_index
            )
            if [row["target_role"] for row in prompt_rows] != [
                "medical_opd_o1",
                "medical_opd_o1",
                "medical_opd_cmb",
                "medical_opd_cmb",
            ]:
                raise FormalB2Error("Formal B2 v2 prompt source order differs")
            record = session.run_formal_step_v2(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=1024,
            )
            if int(record["optimizer_step"]) != next_step:
                raise FormalB2Error("Formal B2 v2 accepted commit count differs")
            records.append(record)
            health = validate_formal_step_health_v2(
                records,
                initial_registry_count=initial_registry,
                initial_model_count=initial_models,
            )
            checkpoint = None
            if next_step % 10 == 0:
                checkpoint = session.seal_registered_checkpoint_v2(
                    logical_version=next_step,
                    package_content_sha256=str(index["package_content_sha256"]),
                    config_sha256=str(index["config_sha256"]),
                    manifest_sha256=str(index["manifest_sha256"]),
                    schedule_sha256=str(index["schedule_semantic_sha256"]),
                    environment=environment,
                )
                checkpoint_index_v2(output)
            removed = session.release_transient_step_artifacts_v1(next_step)
            after_disk = validate_disk_safety_v2(
                free_bytes=shutil.disk_usage(output).free,
                full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
                predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
            )
            elapsed = time.time() - started
            average_step = elapsed / (next_step - start_step)
            ratio = record["ratio_v2"]
            progress = {
                "accepted_optimizer_commits": next_step,
                "rejected_attempts": len(list((output / "rejected_updates_v2").glob("attempt_*.json"))),
                "target_accepted_optimizer_commits": target_step,
                "policy_version": record["next_policy_version"],
                "data_cursor": next_step * 4,
                "average_step_seconds_this_process": average_step,
                "eta_seconds": (target_step - next_step) * average_step,
                "loss": record["loss"],
                "reverse_kl": record["reverse_kl"],
                "ppo_ratio_pre": ratio["ppo_ratio"],
                "backend_correction": ratio["backend_correction"],
                "post_update_shift": ratio["post_update_policy_shift"],
                "ratio_health_v2": record["ratio_health_v2"],
                "ess_fraction": record["ess_fraction"],
                "gradient_norm": record["gradient_norm"],
                "truncation_count": sum(bool(row["truncated"]) for row in record["prompt_samples"]),
                "gpu_memory_bytes": record["gpu_memory_bytes"],
                "disk": after_disk,
                "derived_cost_cny_reference_only": elapsed / 3600.0 * REFERENCE_PRICE_CNY_PER_HOUR,
                "checkpoint_complete": checkpoint is not None,
                "checkpoint_rotation": None if checkpoint is None else checkpoint["rotation"],
                "transient_paths_removed": removed,
                "health": health,
                "restricted_access_count": 0,
            }
            _append_jsonl(output / "progress_v2.jsonl", progress)
            if next_step in {1, 10, 30, 60, 90, 120}:
                print(json.dumps({"event": "progress", **progress}), flush=True)
        final = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_v2_training_status",
            "status": "formal_b2_v2_trained_to_120_controller_pending",
            "run_id": config["run"]["run_id"],
            "accepted_optimizer_commits": 120,
            "rejected_attempts_count_as_steps": False,
            "policy_version": 120,
            "data_cursor": 480,
            "checkpoint_index": checkpoint_index_v2(output),
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
            "final_executed": False,
            "elapsed_seconds_this_process": time.time() - started,
            "derived_cost_cny_reference_only": (time.time() - started) / 3600.0 * REFERENCE_PRICE_CNY_PER_HOUR,
            "platform_actual_cost_cny": None,
        }
        _atomic_json(output / "training_status_step_120_v2.json", final)
        return final
    except BaseException as error:
        index_now = checkpoint_index_v2(output) if output.is_dir() else None
        full = [] if index_now is None else index_now["full_resume_checkpoints"]
        failure = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_v2_failure",
            "status": "failed_closed",
            "process_id": process_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "last_complete_accepted_optimizer_commit": (
                int(records[-1]["optimizer_step"]) if records else start_step
            ),
            "latest_complete_resume_checkpoint": full[-1] if full else None,
            "rejected_attempt_count": (
                len(list((output / "rejected_updates_v2").glob("attempt_*.json")))
                if output.is_dir()
                else 0
            ),
            "final_executed": False,
            "restricted_access_count": 0,
            "elapsed_seconds_this_process": time.time() - started,
        }
        if output.is_dir():
            _atomic_json(output / f"failure_{process_id}_v2.json", failure)
        raise
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--target-step", type=int, choices=(120,), required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    result = run_worker_v2(
        args.package,
        target_step=args.target_step,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["checkpoint_index_v2", "run_worker_v2"]
