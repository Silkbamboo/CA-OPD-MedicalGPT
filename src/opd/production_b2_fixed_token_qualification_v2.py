"""P5.1 step20→25 diagnostic replay and transactional fixed-token qualification."""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_v2 import (
    formal_b2_runtime_config_v2,
    validate_bounded_formula_v2,
)
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json


class FixedTokenQualificationV2Error(RuntimeError):
    """Fixed-token identity, replay, risk capture, or rollback failed."""


BOUNDED_INFLUENCE_REPAIR_V2 = {
    "enabled": True,
    "mode": "per_prompt_gradient_clipping",
    "per_prompt_gradient_clip_norm": 0.25,
    "global_gradient_clip_norm": 1.0,
    "effective_batch_size": 4,
    "applies_to": ["B2", "IDT", "CA-OPD"],
    "formula": "global_gradient_clip_norm / effective_batch_size",
    "formula_path": "configs/opd/pg_opd_three_policy_correction_v5_bounded_runtime.yaml",
    "decision_path": "docs/decisions/0032-common-per-prompt-gradient-trust-budget.md",
}


def _sha_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_index(package: Path) -> Mapping[str, Any]:
    index = _json(package / "package_index.json")
    files = index.get("files")
    if not isinstance(files, Mapping):
        raise FixedTokenQualificationV2Error("source package index is absent")
    for name, descriptor in files.items():
        path = package / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise FixedTokenQualificationV2Error("source package SHA differs")
    return index


def _diagnostic_runtime(
    source_config: Mapping[str, Any],
    *,
    output: Path,
    thresholds: Mapping[str, Any],
    repair_formula_path: Path | None = None,
    selected_learning_rate: float = 3.0e-5,
) -> dict[str, Any]:
    config = deepcopy(dict(source_config))
    config["schema_id"] = "ca-opd/formal-b2-medical-opd/v2"
    config["schema_version"] = 2
    config["package_version"] = "p5_1_formal_b2_v2"
    config["run"] = {
        "run_id": output.name,
        "seed": 42,
        "optimizer_steps": 150,
        "stage1_stop_step": 120,
        "output_dir": str(output),
    }
    config["ratio_health_v2"] = {
        "protocol": "docs/decisions/0031-ratio-health-protocol-v2.md",
        "selected_common_learning_rate": float(selected_learning_rate),
        "thresholds": deepcopy(dict(thresholds)),
        "diagnostic_only": True,
    }
    config["protocol"]["learning_rate"] = float(selected_learning_rate)
    if repair_formula_path is not None:
        repair_formula_path = Path(repair_formula_path).resolve()
        config["protocol"]["three_policy_formula_path"] = str(repair_formula_path)
        config["protocol"]["three_policy_formula_sha256"] = _sha_file(
            repair_formula_path
        )
    return formal_b2_runtime_config_v2(config)


def _replay_comparison(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    scalar_tolerances = {
        "loss": 1.0e-6,
        "objective": 1.0e-6,
        "gradient_norm_before_clip": 1.0e-4,
        "adapter_delta_norm": 1.0e-6,
        "ess_fraction": 1.0e-9,
    }
    scalar = {
        field: {
            "historical": float(old[field]),
            "replay": float(new[field]),
            "absolute_difference": abs(float(old[field]) - float(new[field])),
            "tolerance": tolerance,
        }
        for field, tolerance in scalar_tolerances.items()
    }
    historical_samples = [
        (row["sample_id"], row["content_hash"], row["source"])
        for row in old["prompt_samples"]
    ]
    replay_samples = [
        (row["sample_id"], row["content_hash"], row["source"])
        for row in new["prompt_samples"]
    ]
    authority_match = all(
        new[field] == old[field]
        for field in (
            "input_trainer_authority_sha256",
            "trainer_authority_sha256",
            "runtime_adapter_sha256",
            "fresh_adapter_sha256",
        )
    )
    passed = (
        authority_match
        and historical_samples == replay_samples
        and all(item["absolute_difference"] <= item["tolerance"] for item in scalar.values())
    )
    return {
        "passed": passed,
        "authority_hash_match": authority_match,
        "prompt_identity_match": historical_samples == replay_samples,
        "scalar_metrics": scalar,
        "historical_response_tokens_persisted": False,
        "exact_historical_response_token_sha_comparison_available": False,
        "replay_token_pool_binding_sha256": new["ratio_v2"]["pool_binding"]["pool_binding_sha256"],
        "new_replay_tokens_designation": "ignored_diagnostic_artifact",
    }


def run_fixed_token_qualification_v2(
    *,
    source_package: Path,
    source_output: Path,
    step20_checkpoint: Path,
    output: Path,
    bounded_influence_repair: bool = False,
    baseline_qualification: Path | None = None,
    formula_path: Path | None = None,
    common_lr_qualification: Path | None = None,
    prior_bounded_qualification: Path | None = None,
) -> dict[str, Any]:
    source_package = Path(source_package).resolve()
    source_output = Path(source_output).resolve()
    step20_checkpoint = Path(step20_checkpoint).resolve()
    output = Path(output).resolve()
    baseline_path = (
        None if baseline_qualification is None else Path(baseline_qualification).resolve()
    )
    common_lr_path = (
        None
        if common_lr_qualification is None
        else Path(common_lr_qualification).resolve()
    )
    prior_bounded_path = (
        None
        if prior_bounded_qualification is None
        else Path(prior_bounded_qualification).resolve()
    )
    if output.exists() or output.is_symlink():
        raise FixedTokenQualificationV2Error("fixed-token output must be fresh")
    if os.environ.get("CA_OPD_ALLOW_P5_1_FIXED_TOKEN_GPU") != "1":
        raise FixedTokenQualificationV2Error("fixed-token GPU authorization env is absent")
    index = _source_index(source_package)
    baseline: Mapping[str, Any] | None = None
    baseline_qualification_sha256 = None
    bounded_formula_sha256 = None
    bounded_decision_sha256 = None
    repair_formula_path: Path | None = None
    common_lr_qualification_sha256 = None
    prior_bounded_qualification_sha256 = None
    selected_learning_rate = 3.0e-5
    if common_lr_path is not None:
        if (
            not common_lr_path.is_file()
            or common_lr_path.is_symlink()
            or prior_bounded_path is None
            or not prior_bounded_path.is_file()
            or prior_bounded_path.is_symlink()
        ):
            raise FixedTokenQualificationV2Error(
                "common LR qualification/prior bounded qualification is absent"
            )
        common_lr = _json(common_lr_path)
        prior_bounded = _json(prior_bounded_path)
        prior_bounded_qualification_sha256 = _sha_file(prior_bounded_path)
        if not (
            common_lr.get("passed") is True
            and common_lr.get("status") == "common_lr_1e5_gpu_qualified"
            and common_lr.get("same_token_fallback_passed") is True
            and common_lr.get("selected_common_learning_rate") == 1.0e-5
            and common_lr.get("third_learning_rate_tested") is False
            and common_lr.get("fixed_token_qualification_sha256")
            == prior_bounded_qualification_sha256
            and prior_bounded.get("passed") is True
            and prior_bounded.get("bounded_influence_qualified") is True
            and prior_bounded.get("selected_common_learning_rate") == 3.0e-5
        ):
            raise FixedTokenQualificationV2Error(
                "common LR 1e-5 GPU qualification chain differs"
            )
        common_lr_qualification_sha256 = _sha_file(common_lr_path)
        selected_learning_rate = 1.0e-5
    elif prior_bounded_path is not None:
        raise FixedTokenQualificationV2Error(
            "prior bounded qualification is only valid for common LR selection"
        )
    if bounded_influence_repair:
        if baseline_path is None or not baseline_path.is_file() or baseline_path.is_symlink():
            raise FixedTokenQualificationV2Error(
                "bounded influence repair requires a baseline qualification"
            )
        baseline = _json(baseline_path)
        if not (
            baseline.get("passed") is False
            and baseline.get("status")
            == "fixed_token_gpu_qualification_requires_semantic_repair"
            and baseline.get("step25_risk_captured") is True
            and baseline.get("selected_common_learning_rate") is None
            and baseline.get("source_package_content_sha256")
            == index["package_content_sha256"]
            and baseline.get("final_access_count") == 0
            and baseline.get("controller_access_count") == 0
            and baseline.get("label_access_count") == 0
        ):
            raise FixedTokenQualificationV2Error(
                "baseline qualification is not the bound semantic failure"
            )
        baseline_qualification_sha256 = _sha_file(baseline_path)
        formula_path = (
            Path(BOUNDED_INFLUENCE_REPAIR_V2["formula_path"]).resolve()
            if formula_path is None
            else Path(formula_path).resolve()
        )
        decision_path = Path(BOUNDED_INFLUENCE_REPAIR_V2["decision_path"]).resolve()
        try:
            formula = validate_bounded_formula_v2(
                formula_path, selected_learning_rate=selected_learning_rate
            )
            formula_ok = bool(
                float(formula["optimizer"]["learning_rate"])
                == selected_learning_rate
                and float(formula["optimizer"]["global_gradient_clip_norm"]) == 1.0
                and float(formula["optimizer"]["per_prompt_gradient_clip_norm"])
                == 0.25
                and formula["bounded_influence"]["mode"]
                == "per_prompt_gradient_clipping"
            )
        except (OSError, UnicodeError, TypeError, KeyError, ValueError, yaml.YAMLError) as error:
            raise FixedTokenQualificationV2Error(
                "bounded influence formula is invalid"
            ) from error
        if not formula_ok or not decision_path.is_file():
            raise FixedTokenQualificationV2Error(
                "bounded influence formula/decision differs"
            )
        bounded_formula_sha256 = _sha_file(formula_path)
        bounded_decision_sha256 = _sha_file(decision_path)
        repair_formula_path = formula_path
    elif baseline_path is not None:
        raise FixedTokenQualificationV2Error(
            "baseline qualification is only valid for bounded influence repair"
        )
    elif formula_path is not None or common_lr_path is not None:
        raise FixedTokenQualificationV2Error(
            "formula/common LR selection requires bounded influence repair"
        )
    checkpoint = validate_formal_checkpoint(step20_checkpoint)
    if checkpoint.get("logical_version") != 20:
        raise FixedTokenQualificationV2Error("diagnostic checkpoint is not step20")
    validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output.parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    source_config = _json(source_package / "formal_b2_config.json")
    schedule = _json(source_package / "prompt_schedule.json")
    authority = _json(source_package / "data_authority.json")
    thresholds = _json(Path("reports/p5_1_ratio_health_thresholds_v2.json").resolve())
    runtime = _diagnostic_runtime(
        source_config,
        output=output,
        thresholds=thresholds,
        repair_formula_path=repair_formula_path,
        selected_learning_rate=selected_learning_rate,
    )
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
    _atomic_json(output / "diagnostic_runtime.json", runtime)

    from src.opd.production_b2_formal_gpu_v2 import (
        DiagnosticCandidateRollbackV2,
        FormalB2SessionV2,
    )

    session: FormalB2SessionV2 | None = None
    started = time.time()
    try:
        session = FormalB2SessionV2(
            runtime,
            config_path=source_package / "formal_b2_config.json",
            route="b2_calibration",
        )
        resume_prompt_rows = resolve_formal_b2_schedule_batch(
            authority, schedule, step_index=20
        )
        resume = session.restore_formal_checkpoint_v1(
            step20_checkpoint,
            package_content_sha256=str(index["package_content_sha256"]),
            config_sha256=str(index["config_sha256"]),
            manifest_sha256=str(index["manifest_sha256"]),
            schedule_sha256=str(index["schedule_semantic_sha256"]),
            resume_prompt_rows=resume_prompt_rows,
        )
        _atomic_json(output / "step20_resume_identity.json", resume)
        historical_reconstruction_learning_rate = 3.0e-5
        for group in session.optimizer.param_groups:
            group["lr"] = historical_reconstruction_learning_rate
        session.optimizer_config["learning_rate"] = historical_reconstruction_learning_rate
        session.scheduler.base_lrs = [
            historical_reconstruction_learning_rate
            for _ in session.scheduler.base_lrs
        ]
        session.scheduler._last_lr = [
            historical_reconstruction_learning_rate
            for _ in session.optimizer.param_groups
        ]
        comparisons: list[dict[str, Any]] = []
        for step_index in range(20, 24):
            rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=step_index)
            record = session.run_formal_step_v2(
                step_index=step_index, prompt_rows=rows, max_new_tokens=1024
            )
            old = _json(source_output / "formal_steps" / f"step_{step_index + 1:03d}.json")
            comparison = _replay_comparison(record, old)
            comparison["optimizer_step"] = step_index + 1
            comparisons.append(comparison)
            session.release_transient_step_artifacts_v1(step_index + 1)
            if not comparison["passed"]:
                raise FixedTokenQualificationV2Error(
                    f"step{step_index + 1} deterministic replay differs"
                )

        # Restoring P5 step20 also restores its 3e-5 optimizer.  Keep that
        # historical state through steps21–24 so the exact v24 diagnostic
        # pre-state is reconstructed, then apply only the GPU-selected common
        # LR to the fixed step25 candidate.
        for group in session.optimizer.param_groups:
            group["lr"] = selected_learning_rate
        session.optimizer_config["learning_rate"] = selected_learning_rate
        session.scheduler.base_lrs = [
            selected_learning_rate for _ in session.scheduler.base_lrs
        ]
        session.scheduler._last_lr = [
            selected_learning_rate for _ in session.optimizer.param_groups
        ]
        rows25 = resolve_formal_b2_schedule_batch(authority, schedule, step_index=24)
        if bounded_influence_repair:
            bounded_runtime = {
                **BOUNDED_INFLUENCE_REPAIR_V2,
                "formula_path": str(repair_formula_path),
            }
            session.enable_bounded_influence_v2(bounded_runtime)
        session.set_diagnostic_unconditional_rollback_v2(True)
        candidate_3e5_exception = None
        try:
            session.run_formal_step_v2(
                step_index=24, prompt_rows=rows25, max_new_tokens=1024
            )
            raise FixedTokenQualificationV2Error("step25 diagnostic candidate committed")
        except DiagnosticCandidateRollbackV2 as error:
            candidate_3e5_exception = {"type": type(error).__name__, "message": str(error)}
        except Exception as error:
            candidate_3e5_exception = {"type": type(error).__name__, "message": str(error)}
        rejected = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
        if not rejected:
            raise FixedTokenQualificationV2Error("step25 rejected-update artifact is absent")
        candidate_3e5 = dict(_json(rejected[-1]))
        reason_3e5 = str(candidate_3e5["reason"])
        current_risk_captured = bool(
            reason_3e5.startswith("ratio_health_v2_rejected:")
            or reason_3e5.startswith("precommit_gradient_health_v2_rejected:")
        )
        risk_captured = bool(
            current_risk_captured
            or (bounded_influence_repair and baseline is not None)
        )
        true_post_update_risk = any(
            token in reason_3e5
            for token in (
                "post_shift",
                "relative_update_norm",
                "tail_loss_share",
                "tail_gradient_proxy_share",
                "approx_kl",
            )
        )
        candidate_1e5 = None
        selected_lr = None
        if (
            selected_learning_rate == 3.0e-5
            and current_risk_captured
            and true_post_update_risk
        ):
            fixed_rollout = deepcopy(session._last_fixed_rollout_v2)
            if not isinstance(fixed_rollout, Mapping):
                raise FixedTokenQualificationV2Error("fixed step25 rollout was not retained")
            for group in session.optimizer.param_groups:
                group["lr"] = 1.0e-5
            session.optimizer_config["learning_rate"] = 1.0e-5
            session.scheduler.base_lrs = [1.0e-5 for _ in session.scheduler.base_lrs]
            session.scheduler._last_lr = [1.0e-5 for _ in session.optimizer.param_groups]
            session.set_diagnostic_unconditional_rollback_v2(True)
            before_count = len(rejected)
            try:
                session.run_corrected_step(24, fixed_rollout)
                raise FixedTokenQualificationV2Error("1e-5 diagnostic candidate committed")
            except Exception:
                session.release_step_teacher(24)
            rejected_after = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
            if len(rejected_after) != before_count + 1:
                raise FixedTokenQualificationV2Error("1e-5 rejected-update artifact is absent")
            candidate_1e5 = dict(_json(rejected_after[-1]))
            if candidate_1e5["reason"] == "fixed_token_candidate_unconditional_rollback":
                selected_lr = 1.0e-5
        elif candidate_3e5["reason"] == "fixed_token_candidate_unconditional_rollback":
            if bounded_influence_repair:
                selected_lr = selected_learning_rate
            else:
                # The unrepaired diagnostic must capture the historical risk,
                # not merely tolerate it.
                selected_lr = None
        bounded_influence_qualified = bool(
            bounded_influence_repair
            and candidate_3e5["reason"]
            == "fixed_token_candidate_unconditional_rollback"
            and isinstance(candidate_3e5.get("ratio_evidence"), Mapping)
            and isinstance(
                candidate_3e5["ratio_evidence"].get("bounded_influence_v2"),
                Mapping,
            )
        )
        passed = bool(
            all(item["passed"] for item in comparisons)
            and resume.get("same_path", {}).get("finite_rate") == 1.0
            and candidate_3e5.get("rollback", {}).get("rollback_verified") is True
            and risk_captured
            and selected_lr in (3.0e-5, 1.0e-5)
            and (not bounded_influence_repair or bounded_influence_qualified)
        )
        result = {
            "schema_version": 2,
            "artifact_kind": "p5_1_fixed_token_gpu_qualification_v2",
            "passed": passed,
            "status": (
                "fixed_token_gpu_qualification_passed"
                if passed
                else "fixed_token_gpu_qualification_requires_semantic_repair"
            ),
            "canonical_identity_passed": resume.get("same_path", {}).get("finite_rate") == 1.0,
            "step21_to_24_replay": comparisons,
            "step21_to_24_reconstruction_learning_rate": 3.0e-5,
            "historical_fixed_response_tokens_available": False,
            "exact_replay_claimed": False,
            "step25_candidate_selected_lr": candidate_3e5,
            "step25_candidate_selected_learning_rate": selected_learning_rate,
            "step25_candidate_3e5": (
                candidate_3e5 if selected_learning_rate == 3.0e-5 else None
            ),
            "step25_candidate_3e5_exception": candidate_3e5_exception,
            "step25_risk_captured": risk_captured,
            "step25_current_candidate_risk_captured": current_risk_captured,
            "step25_failure_classification": (
                "true_post_update_optimization_risk"
                if true_post_update_risk
                else "precommit_gradient_or_non_lr_semantic_risk"
            ),
            "candidate_1e5": (
                candidate_3e5 if selected_learning_rate == 1.0e-5 else candidate_1e5
            ),
            "selected_common_learning_rate": selected_lr,
            "bounded_influence_repair": bounded_influence_repair,
            "bounded_influence_protocol": (
                {
                    **deepcopy(BOUNDED_INFLUENCE_REPAIR_V2),
                    "formula_path": str(repair_formula_path),
                    "formula_sha256": bounded_formula_sha256,
                    "decision_sha256": bounded_decision_sha256,
                }
                if bounded_influence_repair
                else None
            ),
            "bounded_influence_formula_sha256": bounded_formula_sha256,
            "bounded_influence_decision_sha256": bounded_decision_sha256,
            "bounded_influence_qualified": bounded_influence_qualified,
            "common_lr_qualification_path": (
                str(common_lr_path) if common_lr_path is not None else None
            ),
            "common_lr_qualification_sha256": common_lr_qualification_sha256,
            "prior_bounded_qualification_path": (
                str(prior_bounded_path) if prior_bounded_path is not None else None
            ),
            "prior_bounded_qualification_sha256": prior_bounded_qualification_sha256,
            "baseline_qualification_path": (
                str(baseline_path) if baseline_path is not None else None
            ),
            "baseline_qualification_sha256": baseline_qualification_sha256,
            "rollback_passed": candidate_3e5.get("rollback", {}).get("rollback_verified") is True,
            "source_step20_checkpoint_manifest_sha256": _sha_file(
                step20_checkpoint / "checkpoint_manifest.json"
            ),
            "source_package_content_sha256": index["package_content_sha256"],
            "new_replay_tokens_designation": "ignored_diagnostic_artifact",
            "final_access_count": 0,
            "controller_access_count": 0,
            "label_access_count": 0,
            "elapsed_seconds": time.time() - started,
            "platform_actual_cost_cny": None,
        }
        _atomic_json(output / "qualification.json", result)
        return result
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--step20-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounded-influence-repair", action="store_true")
    parser.add_argument("--baseline-qualification", type=Path)
    parser.add_argument("--formula-path", type=Path)
    parser.add_argument("--common-lr-qualification", type=Path)
    parser.add_argument("--prior-bounded-qualification", type=Path)
    args = parser.parse_args(argv)
    result = run_fixed_token_qualification_v2(
        source_package=args.source_package,
        source_output=args.source_output,
        step20_checkpoint=args.step20_checkpoint,
        output=args.output,
        bounded_influence_repair=args.bounded_influence_repair,
        baseline_qualification=args.baseline_qualification,
        formula_path=args.formula_path,
        common_lr_qualification=args.common_lr_qualification,
        prior_bounded_qualification=args.prior_bounded_qualification,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_fixed_token_qualification_v2"]
