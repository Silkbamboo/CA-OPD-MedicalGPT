"""Same-token 3e-5→1e-5 diagnostic for a failed fresh-v0 canary batch."""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_fixed_token_qualification_v2 import (
    _replay_comparison,
    _sha_file,
    _source_index,
)
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json


class FreshV0LRQualificationV2Error(RuntimeError):
    """The fixed canary batch, decision tree, or rollback proof differed."""


def _same_failed_candidate_v2(
    replay: Mapping[str, Any], original: Mapping[str, Any]
) -> dict[str, Any]:
    replay_evidence = replay["ratio_evidence"]
    original_evidence = original["ratio_evidence"]
    exact_fields = {
        "fixed_batch_sha256": (
            replay["rollback"]["fixed_batch_sha256"],
            original["rollback"]["fixed_batch_sha256"],
        ),
        "pool_binding_sha256": (
            replay_evidence["pool_binding"]["pool_binding_sha256"],
            original_evidence["pool_binding"]["pool_binding_sha256"],
        ),
        "response_token_sha256": (
            replay_evidence["pool_binding"]["response_token_sha256"],
            original_evidence["pool_binding"]["response_token_sha256"],
        ),
        "valid_mask_sha256": (
            replay_evidence["pool_binding"]["valid_mask_sha256"],
            original_evidence["pool_binding"]["valid_mask_sha256"],
        ),
        "valid_token_count": (
            replay_evidence["pool_binding"]["valid_token_count"],
            original_evidence["pool_binding"]["valid_token_count"],
        ),
    }
    scalar_fields = {
        "backend_ess": (
            replay_evidence["backend_correction"]["ess"]["pooled_fraction"],
            original_evidence["backend_correction"]["ess"]["pooled_fraction"],
        ),
        "post_abs_log_max": (
            replay_evidence["post_update_policy_shift"]["log"]["abs_max"],
            original_evidence["post_update_policy_shift"]["log"]["abs_max"],
        ),
        "post_abs_log_p99": (
            replay_evidence["post_update_policy_shift"]["log"]["abs_p99"],
            original_evidence["post_update_policy_shift"]["log"]["abs_p99"],
        ),
        "tail_loss_share": (
            replay_evidence["post_update_policy_shift"]["tail"]["absolute_loss_share"],
            original_evidence["post_update_policy_shift"]["tail"]["absolute_loss_share"],
        ),
        "tail_gradient_proxy_share": (
            replay_evidence["post_update_policy_shift"]["tail"]["gradient_proxy_share"],
            original_evidence["post_update_policy_shift"]["tail"]["gradient_proxy_share"],
        ),
    }
    exact = {
        name: {"replay": left, "original": right, "passed": left == right}
        for name, (left, right) in exact_fields.items()
    }
    scalars = {
        name: {
            "replay": float(left),
            "original": float(right),
            "absolute_difference": abs(float(left) - float(right)),
            "tolerance": 1.0e-6,
        }
        for name, (left, right) in scalar_fields.items()
    }
    passed = bool(
        all(item["passed"] for item in exact.values())
        and all(
            item["absolute_difference"] <= item["tolerance"]
            for item in scalars.values()
        )
        and "tail_loss_share" in str(replay["reason"])
        and "tail_gradient_proxy_share" in str(replay["reason"])
        and replay.get("rollback", {}).get("rollback_verified") is True
    )
    return {"passed": passed, "exact": exact, "scalars": scalars}


def run_fresh_v0_lr_qualification_v2(
    *,
    source_package: Path,
    failed_canary_output: Path,
    fixed_token_qualification: Path,
    formula_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Rebuild the failed step3 batch and test only the frozen 1e-5 fallback."""

    source_package = Path(source_package).resolve()
    failed_canary_output = Path(failed_canary_output).resolve()
    qualification_path = Path(fixed_token_qualification).resolve()
    formula_path = Path(formula_path).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FreshV0LRQualificationV2Error("LR diagnostic output must be fresh")
    if os.environ.get("CA_OPD_ALLOW_P5_1_CANARY_LR_DIAGNOSTIC_GPU") != "1":
        raise FreshV0LRQualificationV2Error("LR diagnostic GPU authorization is absent")
    index = _source_index(source_package)
    qualification = _json(qualification_path)
    formula_sha256 = _sha_file(formula_path)
    bounded = qualification.get("bounded_influence_protocol")
    if not (
        qualification.get("passed") is True
        and qualification.get("selected_common_learning_rate") == 3.0e-5
        and qualification.get("bounded_influence_qualified") is True
        and qualification.get("bounded_influence_formula_sha256") == formula_sha256
        and isinstance(bounded, Mapping)
    ):
        raise FreshV0LRQualificationV2Error("source 3e-5 qualification differs")
    original_steps = [
        _json(failed_canary_output / "formal_steps" / f"step_{step:03d}.json")
        for step in (1, 2)
    ]
    original_rejected_paths = sorted(
        (failed_canary_output / "rejected_updates_v2").glob("attempt_*.json")
    )
    if len(original_rejected_paths) != 1:
        raise FreshV0LRQualificationV2Error("failed canary rejection is not unique")
    original_rejected_path = original_rejected_paths[0]
    original_rejected = _json(original_rejected_path)
    if not (
        original_rejected.get("accepted_optimizer_steps") == 2
        and original_rejected.get("attempted_optimizer_step") == 3
        and original_rejected.get("counts_as_optimizer_commit") is False
        and original_rejected.get("cursor_advanced") is False
        and original_rejected.get("sampler_refreshed") is False
        and original_rejected.get("rollback", {}).get("rollback_verified") is True
        and "tail_loss_share" in str(original_rejected.get("reason"))
        and "tail_gradient_proxy_share" in str(original_rejected.get("reason"))
    ):
        raise FreshV0LRQualificationV2Error("source canary failure class differs")
    validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(output.parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )

    config = deepcopy(dict(_json(source_package / "formal_b2_config.json")))
    thresholds = dict(_json(Path("reports/p5_1_ratio_health_thresholds_v2.json").resolve()))
    config.update(
        {
            "schema_id": "ca-opd/formal-b2-medical-opd/v2",
            "schema_version": 2,
            "package_version": "p5_1_formal_b2_v2",
            "run": {
                "run_id": output.name,
                "seed": 42,
                "optimizer_steps": 150,
                "stage1_stop_step": 120,
                "output_dir": str(output),
            },
            "ratio_health_v2": {
                "protocol": "docs/decisions/0031-ratio-health-protocol-v2.md",
                "selected_common_learning_rate": 3.0e-5,
                "thresholds": thresholds,
                "diagnostic_only": True,
            },
            "bounded_influence_v2": deepcopy(dict(bounded)),
        }
    )
    config["protocol"]["learning_rate"] = 3.0e-5
    config["protocol"]["three_policy_formula_path"] = str(formula_path)
    config["protocol"]["three_policy_formula_sha256"] = formula_sha256
    runtime = formal_b2_runtime_config_v2(config)
    schedule = _json(source_package / "prompt_schedule.json")
    authority = _json(source_package / "data_authority.json")
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

    from src.opd.production_b2_formal_gpu_v2 import (
        DiagnosticCandidateRollbackV2,
        FormalB2SessionV2,
    )
    from src.opd.production_qualification_two_step_gpu_v7 import (
        ProductionTwoStepQualificationV6Error,
    )

    session: FormalB2SessionV2 | None = None
    started = time.time()
    try:
        session = FormalB2SessionV2(
            runtime,
            config_path=source_package / "formal_b2_config.json",
            route="b2_calibration",
        )
        identity = session.initial_calibration_identity()
        if not (
            identity.get("zero_effect_verified") is True
            and identity.get("tensor_count") == 504
            and identity.get("source_adapter_path") is None
        ):
            raise FreshV0LRQualificationV2Error("diagnostic is not fresh-v0")
        replay: list[dict[str, Any]] = []
        for step_index in range(2):
            rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=step_index)
            record = session.run_formal_step_v2(
                step_index=step_index,
                prompt_rows=rows,
                max_new_tokens=1024,
            )
            comparison = _replay_comparison(record, original_steps[step_index])
            comparison["optimizer_step"] = step_index + 1
            replay.append(comparison)
            session.release_transient_step_artifacts_v1(step_index + 1)
            if not comparison["passed"]:
                raise FreshV0LRQualificationV2Error(
                    f"failed canary step{step_index + 1} replay differs"
                )

        rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=2)
        try:
            session.run_formal_step_v2(
                step_index=2,
                prompt_rows=rows,
                max_new_tokens=1024,
            )
            raise FreshV0LRQualificationV2Error("3e-5 replay candidate committed")
        except ProductionTwoStepQualificationV6Error:
            pass
        rejected = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
        if len(rejected) != 1:
            raise FreshV0LRQualificationV2Error("3e-5 replay rejection is absent")
        candidate_3e5 = dict(_json(rejected[0]))
        deterministic_failure = _same_failed_candidate_v2(
            candidate_3e5, original_rejected
        )
        if not deterministic_failure["passed"]:
            raise FreshV0LRQualificationV2Error("3e-5 failed batch replay differs")
        fixed_rollout = deepcopy(session._last_fixed_rollout_v2)
        if not isinstance(fixed_rollout, Mapping):
            raise FreshV0LRQualificationV2Error("fixed step3 rollout was not retained")

        for group in session.optimizer.param_groups:
            group["lr"] = 1.0e-5
        session.optimizer_config["learning_rate"] = 1.0e-5
        session.scheduler.base_lrs = [1.0e-5 for _ in session.scheduler.base_lrs]
        session.scheduler._last_lr = [1.0e-5 for _ in session.optimizer.param_groups]
        session.set_diagnostic_unconditional_rollback_v2(True)
        candidate_1e5_exception: dict[str, str] | None = None
        try:
            session.run_corrected_step(2, fixed_rollout)
            raise FreshV0LRQualificationV2Error("1e-5 diagnostic candidate committed")
        except DiagnosticCandidateRollbackV2 as error:
            candidate_1e5_exception = {
                "type": type(error).__name__,
                "message": str(error),
            }
        except ProductionTwoStepQualificationV6Error as error:
            candidate_1e5_exception = {
                "type": type(error).__name__,
                "message": str(error),
            }
        finally:
            session.release_step_teacher(2)
        rejected = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
        if len(rejected) != 2:
            raise FreshV0LRQualificationV2Error("1e-5 candidate artifact is absent")
        candidate_1e5 = dict(_json(rejected[1]))
        same_token_fallback = bool(
            candidate_1e5.get("rollback", {}).get("fixed_batch_sha256")
            == candidate_3e5.get("rollback", {}).get("fixed_batch_sha256")
            and candidate_1e5.get("ratio_evidence", {})
            .get("pool_binding", {})
            .get("pool_binding_sha256")
            == candidate_3e5.get("ratio_evidence", {})
            .get("pool_binding", {})
            .get("pool_binding_sha256")
            and candidate_1e5.get("ratio_evidence", {})
            .get("pool_binding", {})
            .get("response_token_sha256")
            == candidate_3e5.get("ratio_evidence", {})
            .get("pool_binding", {})
            .get("response_token_sha256")
        )
        selected_lr = (
            1.0e-5
            if candidate_1e5.get("reason")
            == "fixed_token_candidate_unconditional_rollback"
            and candidate_1e5.get("rollback", {}).get("rollback_verified") is True
            else None
        )
        passed = bool(
            all(item["passed"] for item in replay)
            and deterministic_failure["passed"]
            and same_token_fallback
            and selected_lr == 1.0e-5
            and candidate_1e5.get("counts_as_optimizer_commit") is False
            and candidate_1e5.get("cursor_advanced") is False
            and candidate_1e5.get("sampler_refreshed") is False
        )
        result = {
            "schema_version": 2,
            "artifact_kind": "p5_1_fresh_v0_same_token_lr_qualification_v2",
            "passed": passed,
            "status": (
                "common_lr_1e5_gpu_qualified"
                if passed
                else "both_preregistered_learning_rates_failed"
            ),
            "diagnostic_replay_designation": "diagnostic_replay_not_formal",
            "fresh_v0_identity": identity,
            "step1_to_2_replay": replay,
            "source_failed_canary_rejection_path": str(original_rejected_path),
            "source_failed_canary_rejection_sha256": _sha_file(original_rejected_path),
            "candidate_3e5": candidate_3e5,
            "candidate_3e5_deterministic_failure": deterministic_failure,
            "candidate_1e5": candidate_1e5,
            "candidate_1e5_exception": candidate_1e5_exception,
            "same_token_fallback_passed": same_token_fallback,
            "fixed_batch_sha256": candidate_3e5["rollback"]["fixed_batch_sha256"],
            "pool_binding_sha256": candidate_3e5["ratio_evidence"]["pool_binding"][
                "pool_binding_sha256"
            ],
            "response_token_sha256": candidate_3e5["ratio_evidence"]["pool_binding"][
                "response_token_sha256"
            ],
            "candidate_committed": False,
            "selected_common_learning_rate": selected_lr,
            "third_learning_rate_tested": False,
            "formula_v5_sha256": formula_sha256,
            "fixed_token_qualification_sha256": _sha_file(qualification_path),
            "source_package_content_sha256": index["package_content_sha256"],
            "formal_b2_started": False,
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
    parser.add_argument("--failed-canary-output", type=Path, required=True)
    parser.add_argument("--fixed-token-qualification", type=Path, required=True)
    parser.add_argument("--formula-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_fresh_v0_lr_qualification_v2(
        source_package=args.source_package,
        failed_canary_output=args.failed_canary_output,
        fixed_token_qualification=args.fixed_token_qualification,
        formula_path=args.formula_path,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["run_fresh_v0_lr_qualification_v2"]
