"""P7 bounded Stage-120 worker for reference-aligned IDT-v2 and CA-OPD-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p6_controller_runtime import score_loaded_controller_metrics
from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_checkpoint_v1 import (
    seal_formal_checkpoint,
    validate_formal_checkpoint,
)
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_b2_formal_worker_v1 import _append_jsonl, _atomic_json, _json
from src.opd.production_b2_formal_worker_v2 import checkpoint_index_v2
from src.opd.production_b2_transaction_v2 import state_tree_sha256
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
)
from src.opd.stage120_gpu_v4 import (
    FormalStage120SessionV4,
    P7ActualImpactMeasurementRollback,
)
from src.opd.stage120_package_v4 import verify_stage120_package_v4
from src.opd.stage120_protocol_v4 import P7Stage120Error
from src.opd.stage120_resume_v4 import (
    archive_uncheckpointed_tail_v4,
    validate_resume_replay_record_v4,
)
from src.opd.stage120_schedule_v4 import resolve_stage120_batch_v4
from src.opd.stage120_transaction_v4 import build_stage120_checkpoint_metadata_v4


REPO = Path(__file__).resolve().parents[2]
CONTROLLER_CONFIG = REPO / "configs/eval/qwen3_4b/p6_controller_v1.json"
REFERENCE_RATE_CNY_PER_HOUR = 2.96


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _prepare_output(output: Path, *, runtime: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    output.mkdir(parents=True)
    for name in (
        "b2_steps",
        "checkpoints",
        "formal_steps",
        "formal_checkpoints",
        "stage120_action_steps_v4",
        "actual_impact_v3",
        "memory_step_audits",
        "memory_telemetry/markers",
        "ratio_evidence_v2",
        "rejected_updates_v2",
        "bounded_rejections_v4",
        "checkpoint_metadata_v4",
        "controller_windows",
        "health_summaries_v4",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "package_preflight_v4.json", preflight)
    _atomic_json(output / "runtime_config_v4.json", runtime)


def _live_adapter_identity(session: Any) -> dict[str, Any]:
    path = Path(session._current_checkpoint_path).resolve()
    manifest = path / "adapter_transport_manifest.json"
    if not path.is_dir() or not manifest.is_file():
        raise P7Stage120Error("P7 live Controller adapter export is absent")
    return {
        "adapter_path": str(path),
        "adapter_ordered_sha256": _ordered_adapter_sha256(path),
        "adapter_weight_sha256": _sha_file(path / "adapter_model.safetensors"),
        "adapter_manifest_sha256": _sha_file(manifest),
    }


def _ca_controller_window_v4(
    session: FormalStage120SessionV4, *, output: Path, completed_step: int
) -> dict[str, Any]:  # pragma: no cover - real GPU Controller
    adapter = _live_adapter_identity(session)
    root = output / "controller_windows" / f"step_{completed_step:03d}"
    cache = root / "prediction_cache"
    if root.exists() or root.is_symlink():
        raise P7Stage120Error("P7 CA Controller output must be fresh")
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
    decision = session.update_ca_controller_v4(
        medical_accuracy=float(metrics["medical_accuracy"]),
        general_accuracy=float(metrics["general_micro_accuracy"]),
        completed_step=completed_step,
    )
    report = {
        "schema_version": 4,
        "artifact_kind": "p7_ca_controller_window_v4",
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


def _existing_records(output: Path, *, through_step: int) -> list[dict[str, Any]]:
    records = []
    for step in range(1, through_step + 1):
        path = output / "formal_steps" / f"step_{step:03d}.json"
        if not path.is_file():
            raise P7Stage120Error("P7 resume formal step evidence is absent")
        records.append(_json(path))
    return records


def _reverse_kl_mean(record: Mapping[str, Any]) -> float:
    value = record.get("reverse_kl")
    if isinstance(value, Mapping):
        value = value.get("mean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise P7Stage120Error("P7 reverse-KL summary value differs") from error
    if not math.isfinite(result):
        raise P7Stage120Error("P7 reverse-KL summary is not finite")
    return result


def _health_summary(records: Sequence[Mapping[str, Any]], session: Any) -> dict[str, Any]:
    recent = list(records[-10:])
    samples = [sample for record in recent for sample in record["prompt_samples"]]
    actions = [record["stage120_v4"]["action"] for record in recent]
    return {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_ten_step_health_summary_v4",
        "through_step": int(records[-1]["optimizer_step"]),
        "window_steps": len(recent),
        "action_counts": {
            "medical": actions.count("medical"),
            "general": actions.count("general"),
        },
        "completion_tokens": sum(int(row["generated_tokens"]) for row in samples),
        "eos_count": sum(bool(row["eos"]) for row in samples),
        "truncation_count": sum(bool(row["truncated"]) for row in samples),
        "mean_loss": sum(float(record["loss"]) for record in recent) / len(recent),
        "mean_reverse_kl": sum(_reverse_kl_mean(record) for record in recent)
        / len(recent),
        "accepted_steps": session.route_state_v4.accepted_steps,
        "rejected_attempts": session.route_state_v4.rejected_attempts,
        "consecutive_rejections": session.route_state_v4.consecutive_rejections,
        "final_access_count": 0,
    }


def run_stage120_worker_v4(
    package: Path,
    *,
    target_step: int,
    resume_checkpoint: Path | None,
    output_override: Path | None = None,
    diagnostic: bool = False,
) -> dict[str, Any]:  # pragma: no cover - real two-GPU worker
    if target_step <= 0 or target_step > 120:
        raise P7Stage120Error("P7 worker target must stop at or before 120")
    if target_step != 120 and not diagnostic:
        raise P7Stage120Error("only an explicit diagnostic may stop before 120")
    if _git("status", "--porcelain"):
        raise P7Stage120Error("P7 worker requires clean committed Git")
    package = package.resolve()
    preflight = verify_stage120_package_v4(package)
    config = _json(package / "formal_method_config.json")
    schedule = _json(package / "stage120_schedule.json")
    authority = _json(package / "data_authority.json")
    health = _json(package / "health_protocol.json")
    index = _json(package / "package_index.json")
    authorization = _json(package / "authorization.json")
    package_head = str(authorization.get("git_head", ""))
    runtime_git_head = _git("rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", package_head, runtime_git_head],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not package_head or ancestry.returncode != 0:
        raise P7Stage120Error("P7 package Git head is not an ancestor")
    output = (
        Path(str(config["run"]["output_dir"]))
        if output_override is None
        else output_override.resolve()
    )
    if diagnostic and output_override is None:
        raise P7Stage120Error("P7 diagnostic output override is required")
    if not diagnostic and output_override is not None:
        raise P7Stage120Error("P7 formal output cannot be overridden")
    runtime = formal_b2_runtime_config_v2(config)
    runtime["stage120_v4"] = config["stage120_v4"]
    runtime["teacher"] = config["teacher"]
    runtime["backend_health_v3"] = health
    runtime["run"] = dict(runtime["run"])
    runtime["run"]["output_dir"] = str(output)
    runtime["run"]["run_id"] = output.name
    if resume_checkpoint is None:
        if output.exists() or output.is_symlink():
            raise P7Stage120Error("P7 fresh output already exists")
        _prepare_output(output, runtime=runtime, preflight=preflight)
        start_step = 0
        records: list[dict[str, Any]] = []
        resume_replay_expected: dict[int, dict[str, Any]] = {}
        resume_archive: dict[str, Any] | None = None
    else:
        resume_checkpoint = resume_checkpoint.resolve()
        manifest = validate_formal_checkpoint(resume_checkpoint)
        start_step = int(manifest["logical_version"])
        if not (output.is_dir() and 0 < start_step < target_step):
            raise P7Stage120Error("P7 resume boundary differs")
        records = _existing_records(output, through_step=start_step)
        resume_replay_expected = {}
        resume_archive = None
    validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output if output.exists() else output.parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    session: FormalStage120SessionV4 | None = None
    started = time.time()
    process_id = f"pid-{os.getpid()}-{int(started)}"
    if resume_checkpoint is not None:
        resume_archive = archive_uncheckpointed_tail_v4(
            output,
            checkpoint_step=start_step,
            process_id=process_id,
        )
        if resume_archive is not None:
            archive_root = Path(str(resume_archive["archive_root"]))
            for path in sorted((archive_root / "formal_steps").glob("step_*.json")):
                archived = _json(path)
                archived_step = int(archived.get("optimizer_step", -1))
                if archived_step <= start_step or archived_step in resume_replay_expected:
                    raise P7Stage120Error("P7 archived replay step identity differs")
                resume_replay_expected[archived_step] = archived
    method_id = str(config["stage120_v4"]["method_id"])
    try:
        session = FormalStage120SessionV4(
            runtime,
            config_path=package / "formal_method_config.json",
            route="b2_calibration",
        )
        if resume_checkpoint is None:
            identity = session.initial_calibration_identity()
            _atomic_json(output / "fresh_v0_identity.json", identity)
            if not (
                identity.get("zero_effect_verified") is True
                and identity.get("source_adapter_path") is None
                and identity.get("tensor_count") == 504
            ):
                raise P7Stage120Error("P7 Student is not fresh-v0")
        else:
            route = _json(resume_checkpoint / "route_state.json")
            route_state = route["route_state"]
            slot = int(route_state["accepted_steps"])
            if method_id == "IDT-v2":
                resume_action = "medical" if slot % 2 == 0 else "general"
            else:
                tape = route_state["random_tape"]
                router_state = route["ca_router_state"]
                resume_action = (
                    "medical"
                    if float(tape[slot]) < float(router_state["p_medical"])
                    else "general"
                )
            resume_rows = resolve_stage120_batch_v4(
                authority,
                schedule,
                accepted_slot=slot,
                action=resume_action,
                reserve_variant=0,
            )
            resume = session.restore_formal_checkpoint_v1(
                resume_checkpoint,
                package_content_sha256=str(index["package_content_sha256"]),
                config_sha256=str(index["config_sha256"]),
                manifest_sha256=str(index["manifest_sha256"]),
                schedule_sha256=str(index["schedule_sha256"]),
                resume_prompt_rows=resume_rows,
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
                "schema_version": 4,
                "artifact_kind": "p7_stage120_process_boundary_v4",
                "method_id": method_id,
                "package_git_head": package_head,
                "runtime_git_head": runtime_git_head,
                "start_accepted_steps": start_step,
                "target_accepted_steps": target_step,
                "diagnostic": diagnostic,
                "gpu0_role": "Student",
                "gpu1_role": "sampler_plus_step_scoped_Base_or_Medical_Teacher",
                "user_provided_estimation_rate_cny_per_hour": REFERENCE_RATE_CNY_PER_HOUR,
                "platform_actual_cost_cny": None,
                "final_access_count": 0,
            },
        )
        while session.route_state_v4.accepted_steps < target_step:
            step_index = session.route_state_v4.accepted_steps
            action = session.select_action_for_attempt_v4()
            reserve = session.route_state_v4.consecutive_rejections
            rows = resolve_stage120_batch_v4(
                authority,
                schedule,
                accepted_slot=step_index,
                action=action,
                reserve_variant=reserve,
            )
            rejected_files_before = len(
                list((output / "rejected_updates_v2").glob("attempt_*.json"))
            )
            record = None
            while record is None:
                try:
                    record = session.run_stage120_attempt_v4(
                        step_index=step_index,
                        prompt_rows=rows,
                        max_new_tokens=1024,
                    )
                except P7ActualImpactMeasurementRollback:
                    try:
                        phase = session.advance_actual_impact_phase_v4()
                    except BaseException as error:
                        rejection = session.reject_stage120_attempt_v4(
                            reason="actual_impact_rejected:" + str(error)
                        )
                        _atomic_json(
                            output
                            / "bounded_rejections_v4"
                            / f"attempt_{session.route_state_v4.rejected_attempts:03d}.json",
                            rejection,
                        )
                        record = False
                        break
                    _append_jsonl(output / "actual_impact_phases_v4.jsonl", phase)
                    rejected_files_before = len(
                        list(
                            (output / "rejected_updates_v2").glob(
                                "attempt_*.json"
                            )
                        )
                    )
                except (ProductionTwoStepQualificationV6Error, P7Stage120Error) as error:
                    rejected_files_after = len(
                        list(
                            (output / "rejected_updates_v2").glob(
                                "attempt_*.json"
                            )
                        )
                    )
                    if rejected_files_after <= rejected_files_before:
                        raise
                    rejection = session.reject_stage120_attempt_v4(
                        reason="qualified_health_rejection:" + str(error)
                    )
                    _atomic_json(
                        output
                        / "bounded_rejections_v4"
                        / f"attempt_{session.route_state_v4.rejected_attempts:03d}.json",
                        rejection,
                    )
                    record = False
                    break
            if record is False:
                total_rejections = session.route_state_v4.rejected_attempts
                consecutive_rejections = session.route_state_v4.consecutive_rejections
                if total_rejections > 3 or consecutive_rejections > 2:
                    raise P7Stage120Error("P7 bounded rejection maximum exceeded")
                continue
            if record is None:
                raise P7Stage120Error("P7 attempt returned no record")
            next_step = int(record["optimizer_step"])
            expected_replay = resume_replay_expected.get(next_step)
            if expected_replay is not None:
                verification = validate_resume_replay_record_v4(
                    expected_replay, record
                )
                _atomic_json(
                    output
                    / "resume_replay_verifications_v4"
                    / f"step_{next_step:03d}.json",
                    {
                        **verification,
                        "archive_root": str(resume_archive["archive_root"]),
                        "runtime_git_head": runtime_git_head,
                    },
                )
            records.append(record)
            if not (
                next_step == step_index + 1
                and session.route_state_v4.accepted_steps == next_step
                and session.current_sampler_version == next_step
                and session._registry_count() == initial_registry
                and session._model_count() == initial_models
            ):
                raise P7Stage120Error("P7 accepted commit/version/registry differs")
            controller = None
            if method_id == "CA-OPD-v2" and next_step in {30, 60, 90, 120}:
                controller = _ca_controller_window_v4(
                    session, output=output, completed_step=next_step
                )
            checkpoint = None
            checkpoint_steps = {30, 60, 90, 120}
            if next_step in checkpoint_steps or (diagnostic and next_step % 10 == 0):
                checkpoint = seal_formal_checkpoint(
                    session,
                    logical_version=next_step,
                    data_cursor=next_step * 4,
                    package_content_sha256=str(index["package_content_sha256"]),
                    config_sha256=str(index["config_sha256"]),
                    manifest_sha256=str(index["manifest_sha256"]),
                    schedule_sha256=str(index["schedule_sha256"]),
                    environment={
                        "production_python": str(Path(os.sys.executable).resolve()),
                        "method_id": method_id,
                        "formula_sha256": index["formula_sha256"],
                        "health_protocol_sha256": index["health_protocol_sha256"],
                        "package_content_sha256": index["package_content_sha256"],
                        "final_access_count": 0,
                    },
                )
                if next_step in checkpoint_steps:
                    metadata = build_stage120_checkpoint_metadata_v4(
                        method_id=method_id,
                        logical_version=next_step,
                        policy_version=next_step,
                        sampler_version=next_step,
                        data_cursor=next_step * 4,
                        scheduler_step=next_step,
                        accepted_steps=next_step,
                        rejected_attempts=session.route_state_v4.rejected_attempts,
                        route_state=session.route_state_v4.state_dict(),
                        controller_state=(
                            None
                            if session.ca_router_v4 is None
                            else session.ca_router_v4.state_dict()
                        ),
                        action_occurrences=session.route_state_v4.action_counts,
                        rng_sha256={
                            "cpu": state_tree_sha256(session.torch.get_rng_state()),
                            "cuda": state_tree_sha256(
                                session.torch.cuda.get_rng_state_all()
                            ),
                        },
                        package_identities={
                            "package_sha256": index["package_content_sha256"],
                            "formula_sha256": index["formula_sha256"],
                            "manifest_sha256": index["manifest_sha256"],
                            "schedule_sha256": index["schedule_sha256"],
                            "health_sha256": index["health_protocol_sha256"],
                        },
                    )
                    _atomic_json(
                        output
                        / "checkpoint_metadata_v4"
                        / f"step_{next_step:03d}.json",
                        metadata,
                    )
            removed = session.release_transient_step_artifacts_v1(next_step)
            summary = None
            if next_step % 10 == 0:
                summary = _health_summary(records, session)
                _atomic_json(
                    output
                    / "health_summaries_v4"
                    / f"through_step_{next_step:03d}.json",
                    summary,
                )
            elapsed = time.time() - started
            progress = {
                "method_id": method_id,
                "accepted_optimizer_commits": next_step,
                "target_accepted_optimizer_commits": target_step,
                "policy_version": session.current_sampler_version,
                "data_cursor": next_step * 4,
                "action": record["stage120_v4"]["action"],
                "action_counts": dict(session.route_state_v4.action_counts),
                "rejected_attempts": session.route_state_v4.rejected_attempts,
                "consecutive_rejections": session.route_state_v4.consecutive_rejections,
                "average_step_seconds_this_process": elapsed / (next_step - start_step),
                "eta_seconds": (target_step - next_step)
                * elapsed
                / (next_step - start_step),
                "loss": record["loss"],
                "reverse_kl": record["reverse_kl"],
                "ratio_health": record["ratio_health_v2"],
                "controller_window_complete": controller is not None,
                "checkpoint_complete": checkpoint is not None,
                "health_summary_complete": summary is not None,
                "transient_paths_removed": removed,
                "disk_free_bytes": shutil.disk_usage(output).free,
                "final_access_count": 0,
            }
            _append_jsonl(output / "progress_v4.jsonl", progress)
            _atomic_json(output / "heartbeat.json", {**progress, "timestamp_unix": time.time()})
            if next_step in {1, 8, 10, 30, 60, 90, 120}:
                print(json.dumps({"event": "progress", **progress}), flush=True)
        elapsed = time.time() - started
        final = {
            "schema_version": 4,
            "artifact_kind": "p7_stage120_training_status_v4",
            "status": (
                "diagnostic_target_reached"
                if diagnostic
                else "trained_to_120_unified_controller_pending"
            ),
            "method_id": method_id,
            "package_git_head": package_head,
            "runtime_git_head": runtime_git_head,
            "accepted_optimizer_commits": target_step,
            "rejected_attempts": session.route_state_v4.rejected_attempts,
            "policy_version": session.current_sampler_version,
            "data_cursor": target_step * 4,
            "action_counts": dict(session.route_state_v4.action_counts),
            "checkpoint_index": checkpoint_index_v2(output),
            "route_state": session.formal_route_state(),
            "controller_access_count_during_training": (
                0 if method_id == "IDT-v2" else target_step // 30
            ),
            "elapsed_seconds_this_process": elapsed,
            "user_provided_estimation_rate_cny_per_hour": REFERENCE_RATE_CNY_PER_HOUR,
            "derived_cost_cny_estimate": elapsed
            / 3600.0
            * REFERENCE_RATE_CNY_PER_HOUR,
            "platform_actual_cost_cny": None,
            "final_access_count": 0,
            "confirmation_access_count": 0,
        }
        _atomic_json(output / f"training_status_step_{target_step:03d}_v4.json", final)
        return final
    except BaseException as error:
        if output.is_dir():
            _atomic_json(
                output / f"failure_{process_id}.json",
                {
                    "schema_version": 4,
                    "artifact_kind": "p7_stage120_failure_v4",
                    "method_id": method_id,
                    "package_git_head": package_head,
                    "runtime_git_head": runtime_git_head,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "latest_complete_checkpoint_index": checkpoint_index_v2(output),
                    "final_access_count": 0,
                },
            )
        raise
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=120)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-override", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    result = run_stage120_worker_v4(
        args.package,
        target_step=args.target_step,
        resume_checkpoint=args.resume_checkpoint,
        output_override=args.output_override,
        diagnostic=args.diagnostic,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_stage120_worker_v4"]
