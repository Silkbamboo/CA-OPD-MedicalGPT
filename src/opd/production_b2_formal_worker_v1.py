"""Authorized formal B2 training worker; imports GPU code only after preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v1 import verify_formal_package
from src.opd.production_b2_formal_v1 import (
    FormalB2Error,
    formal_b2_runtime_config,
)


MINIMUM_DISK_FREE_BYTES = 10_000_000_000
REFERENCE_PRICE_CNY_PER_HOUR = 2.96
DERIVED_COST_CAP_CNY = 53.28


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FormalB2Error(f"worker JSON is not an object: {path.name}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_index(output: Path, *, target_step: int) -> dict[str, Any]:
    root = output / "formal_checkpoints"
    checkpoints: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("step_*")):
            manifest = validate_formal_checkpoint(path)
            checkpoints.append(
                {
                    "logical_version": manifest["logical_version"],
                    "path": str(path.relative_to(output)),
                    "adapter_sha256": manifest["adapter_sha256"],
                    "checkpoint_manifest_sha256": __import__("hashlib").sha256(
                        (path / "checkpoint_manifest.json").read_bytes()
                    ).hexdigest(),
                    "data_cursor": manifest["data_cursor"],
                    "complete": True,
                    "resume_eligible": True,
                }
            )
    result = {
        "schema_version": 1,
        "artifact_kind": "p5_formal_b2_checkpoint_index_v1",
        "target_step": target_step,
        "retention": "milestones plus latest/previous two rolling checkpoints",
        "checkpoints": sorted(checkpoints, key=lambda item: item["logical_version"]),
    }
    _atomic_json(output / "checkpoint_index.json", result)
    return result


def _existing_records(output: Path, *, through_step: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step in range(1, through_step + 1):
        path = output / "formal_steps" / f"step_{step:03d}.json"
        record = dict(_json(path))
        if record.get("optimizer_step") != step:
            raise FormalB2Error("existing formal step chain/cursor differs")
        records.append(record)
    extras = sorted((output / "formal_steps").glob("step_*.json"))
    if len(extras) != through_step:
        raise FormalB2Error(
            "resume requires an exact complete-step artifact boundary; preserve and isolate partial attempt first"
        )
    return records


def run_worker(
    package: Path,
    *,
    target_step: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    if target_step not in {120, 150}:
        raise FormalB2Error("formal worker target must be stage1 120 or extension 150")
    preflight = verify_formal_package(
        package,
        allow_existing_output_for_resume=resume_checkpoint is not None,
    )
    config = _json(package / "formal_b2_config.json")
    schedule = _json(package / "prompt_schedule.json")
    authority = _json(package / "data_authority.json")
    environment = _json(package / "environment.json")
    gates = _json(package / "rolling_safety_gates.json")
    index = _json(package / "package_index.json")
    runtime = formal_b2_runtime_config(config)
    output = Path(str(config["run"]["output_dir"]))
    if resume_checkpoint is None:
        output.mkdir(parents=True)
        for name in (
            "b2_steps",
            "checkpoints",
            "formal_steps",
            "memory_step_audits",
            "memory_telemetry/markers",
        ):
            (output / name).mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "package_preflight.json", preflight)
        _atomic_json(output / "runtime_config.json", runtime)
        _atomic_json(output / "environment.json", environment)
        start_step = 0
        records: list[dict[str, Any]] = []
    else:
        checkpoint_manifest = validate_formal_checkpoint(resume_checkpoint)
        start_step = int(checkpoint_manifest["logical_version"])
        if not (start_step < target_step and start_step in range(10, 151, 10)):
            raise FormalB2Error("resume checkpoint is not before requested target")
        records = _existing_records(output, through_step=start_step)
    free = shutil.disk_usage(output).free
    if free < MINIMUM_DISK_FREE_BYTES:
        raise FormalB2Error("formal worker disk is below 10 GB before model load")

    from src.opd.production_b2_formal_gpu_v1 import (
        FormalB2SessionV1,
        validate_formal_step_health,
    )

    session: FormalB2SessionV1 | None = None
    started = time.time()
    process_id = f"pid-{os.getpid()}-{int(started)}"
    try:
        session = FormalB2SessionV1(
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
                raise FormalB2Error("formal model load fresh-v0 identity failed")
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
                {
                    **resume_evidence,
                    "resume_from": str(resume_checkpoint),
                    "process_boundary": process_id,
                },
            )
        initial_registry = session._registry_count()
        initial_models = session._model_count()
        _atomic_json(
            output / f"process_{process_id}.json",
            {
                "schema_version": 1,
                "artifact_kind": "formal_b2_process_boundary_v1",
                "process_id": process_id,
                "resume_from": None if resume_checkpoint is None else str(resume_checkpoint),
                "start_optimizer_step": start_step,
                "target_optimizer_step": target_step,
                "model_load_complete": True,
                "gpu0_role": "Student",
                "gpu1_role": "long_lived_sampler_and_step_scoped_Medical_Teacher",
                "trainable_tensor_count": 504,
                "initial_registry_count": initial_registry,
                "initial_model_count": initial_models,
            },
        )
        print(
            json.dumps(
                {
                    "event": "model_load_complete",
                    "start_step": start_step,
                    "target_step": target_step,
                    "disk_free_bytes": free,
                    "registry_count": initial_registry,
                    "model_count": initial_models,
                }
            ),
            flush=True,
        )
        for step_index in range(start_step, target_step):
            step = step_index + 1
            current_free = shutil.disk_usage(output).free
            elapsed_hours = (time.time() - started) / 3600.0
            derived_cost = elapsed_hours * REFERENCE_PRICE_CNY_PER_HOUR
            if current_free < MINIMUM_DISK_FREE_BYTES:
                raise FormalB2Error("disk_below_10gb before registered step")
            if derived_cost > DERIVED_COST_CAP_CNY:
                raise FormalB2Error("derived run-card cost cap exceeded")
            prompt_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=step_index
            )
            if [row["target_role"] for row in prompt_rows] != [
                "medical_opd_o1",
                "medical_opd_o1",
                "medical_opd_cmb",
                "medical_opd_cmb",
            ]:
                raise FormalB2Error("formal prompt provider source order differs")
            record = session.run_formal_step_v1(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=1024,
            )
            records.append(record)
            health = validate_formal_step_health(
                records,
                gates=gates,
                initial_registry_count=initial_registry,
                initial_model_count=initial_models,
            )
            checkpoint_result = None
            if step % 10 == 0:
                checkpoint_result = session.seal_registered_checkpoint_v1(
                    logical_version=step,
                    package_content_sha256=str(index["package_content_sha256"]),
                    config_sha256=str(index["config_sha256"]),
                    manifest_sha256=str(index["manifest_sha256"]),
                    schedule_sha256=str(index["schedule_semantic_sha256"]),
                    environment=environment,
                    target_step=target_step,
                )
                _checkpoint_index(output, target_step=target_step)
            removed = session.release_transient_step_artifacts_v1(step)
            after_free = shutil.disk_usage(output).free
            if after_free < MINIMUM_DISK_FREE_BYTES:
                raise FormalB2Error("disk_below_10gb after checkpoint/rotation")
            elapsed = time.time() - started
            completed_this_process = step - start_step
            average_step = elapsed / completed_this_process
            remaining = target_step - step
            progress = {
                "optimizer_step": step,
                "target_step": target_step,
                "policy_version": record["next_policy_version"],
                "data_cursor": step * 4,
                "average_step_seconds_this_process": average_step,
                "eta_seconds": remaining * average_step,
                "loss": record["loss"],
                "reverse_kl": record["reverse_kl"],
                "ratio": record["ratio"],
                "ess_fraction": record["ess_fraction"],
                "gradient_norm": record["gradient_norm"],
                "truncation_count": sum(
                    bool(row["truncated"]) for row in record["prompt_samples"]
                ),
                "gpu_memory_bytes": record["gpu_memory_bytes"],
                "disk_free_bytes": after_free,
                "derived_cost_cny_reference_only": elapsed / 3600.0 * REFERENCE_PRICE_CNY_PER_HOUR,
                "checkpoint_complete": checkpoint_result is not None,
                "checkpoint_rotation": None if checkpoint_result is None else checkpoint_result["rotation"],
                "transient_paths_removed": removed,
                "health": health,
                "restricted_access_count": 0,
            }
            _append_jsonl(output / "progress.jsonl", progress)
            if step in {1, 10, 30, 60, 90, 120, 150}:
                print(json.dumps({"event": "progress", **progress}), flush=True)
        final = {
            "schema_version": 1,
            "artifact_kind": "formal_b2_training_stage_status_v1",
            "status": (
                "formal_b2_trained_to_120_controller_pending"
                if target_step == 120
                else "formal_b2_registered_extension_trained_to_150_controller_pending"
            ),
            "run_id": config["run"]["run_id"],
            "optimizer_steps_completed": target_step,
            "policy_version": target_step,
            "data_cursor": target_step * 4,
            "checkpoint_index": _checkpoint_index(output, target_step=target_step),
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
            "final_executed": False,
            "elapsed_seconds_this_process": time.time() - started,
            "derived_cost_cny_reference_only": (time.time() - started)
            / 3600.0
            * REFERENCE_PRICE_CNY_PER_HOUR,
        }
        _atomic_json(output / f"training_status_step_{target_step:03d}.json", final)
        return final
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "artifact_kind": "formal_b2_failure_v1",
            "status": "failed_closed",
            "process_id": process_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "last_complete_optimizer_step": (
                int(records[-1]["optimizer_step"]) if records else start_step
            ),
            "latest_complete_resume_checkpoint": (
                _checkpoint_index(output, target_step=target_step)["checkpoints"][-1]
                if (output / "formal_checkpoints").is_dir()
                and any((output / "formal_checkpoints").iterdir())
                else None
            ),
            "final_executed": False,
            "restricted_access_count": 0,
            "elapsed_seconds_this_process": time.time() - started,
        }
        _atomic_json(output / f"failure_{process_id}.json", failure)
        raise
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--target-step", type=int, choices=(120, 150), required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    result = run_worker(
        args.package,
        target_step=args.target_step,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
