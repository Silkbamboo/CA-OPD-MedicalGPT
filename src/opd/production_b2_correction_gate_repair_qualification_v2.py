"""GPU qualification of ADR-0035 on the deterministic step18 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from src.opd.production_b2_correction_gate_forensic_v2 import _comparison
from src.opd.production_b2_fixed_token_qualification_v2 import (
    _diagnostic_runtime,
    _source_index,
)
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json


class CorrectionGateRepairQualificationV2Error(RuntimeError):
    """ADR-0035 fixed-batch replay, candidate, or rollback differed."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_correction_gate_repair_qualification_v2(
    *,
    source_package: Path,
    source_output: Path,
    step10_checkpoint: Path,
    legacy_forensic_report: Path,
    output: Path,
) -> dict[str, Any]:
    source_package = source_package.resolve()
    source_output = source_output.resolve()
    step10_checkpoint = step10_checkpoint.resolve()
    legacy_forensic_report = legacy_forensic_report.resolve()
    output = output.resolve()
    if output.exists():
        raise CorrectionGateRepairQualificationV2Error("qualification output is not fresh")
    index = _source_index(source_package)
    if validate_formal_checkpoint(step10_checkpoint).get("logical_version") != 10:
        raise CorrectionGateRepairQualificationV2Error("qualification checkpoint is not step10")
    if not legacy_forensic_report.is_file():
        raise CorrectionGateRepairQualificationV2Error("legacy forensic report is absent")
    protocol = Path("configs/opd/preupdate_backend_health_v2_1.json").resolve()
    if not protocol.is_file():
        raise CorrectionGateRepairQualificationV2Error("ADR-0035 protocol config is absent")
    source_config = _json(source_package / "formal_b2_config.json")
    schedule = _json(source_package / "prompt_schedule.json")
    authority = _json(source_package / "data_authority.json")
    thresholds_path = Path("reports/p5_1_ratio_health_thresholds_v2.json").resolve()
    thresholds = _json(thresholds_path)
    runtime = _diagnostic_runtime(
        source_config,
        output=output,
        thresholds=thresholds,
        selected_learning_rate=1.0e-5,
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
        resume_rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=10)
        resume = session.restore_formal_checkpoint_v1(
            step10_checkpoint,
            package_content_sha256=str(index["package_content_sha256"]),
            config_sha256=str(index["config_sha256"]),
            manifest_sha256=str(index["manifest_sha256"]),
            schedule_sha256=str(index["schedule_semantic_sha256"]),
            resume_prompt_rows=resume_rows,
        )
        _atomic_json(output / "step10_resume_identity.json", resume)
        comparisons = []
        for step_index in range(10, 17):
            rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=step_index)
            record = session.run_formal_step_v2(
                step_index=step_index, prompt_rows=rows, max_new_tokens=1024
            )
            historical = _json(
                source_output / "formal_steps" / f"step_{step_index + 1:03d}.json"
            )
            comparison = _comparison(record, historical)
            comparison["optimizer_step"] = step_index + 1
            comparisons.append(comparison)
            session.release_transient_step_artifacts_v1(step_index + 1)
            if not comparison["passed"]:
                raise CorrectionGateRepairQualificationV2Error(
                    f"step{step_index + 1} deterministic replay differs"
                )

        session.set_diagnostic_unconditional_rollback_v2(True)
        rows18 = resolve_formal_b2_schedule_batch(authority, schedule, step_index=17)
        rollback_exception = None
        try:
            session.run_formal_step_v2(
                step_index=17, prompt_rows=rows18, max_new_tokens=1024
            )
        except DiagnosticCandidateRollbackV2 as error:
            rollback_exception = {"type": type(error).__name__, "message": str(error)}
        if rollback_exception is None:
            raise CorrectionGateRepairQualificationV2Error(
                "step18 qualification candidate did not roll back"
            )
        attempts = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
        if len(attempts) != 1:
            raise CorrectionGateRepairQualificationV2Error("qualification rejection artifact differs")
        rejected = _json(attempts[0])
        evidence = rejected["ratio_evidence"]
        health = session._last_ratio_health_v2
        preupdate = session._preupdate_backend_health_v2
        acceptance = evidence.get("candidate_acceptance_v2_1")
        if not (
            rejected.get("reason") == "fixed_token_candidate_unconditional_rollback"
            and rejected["rollback"]["rollback_verified"] is True
            and rejected.get("counts_as_optimizer_commit") is False
            and rejected.get("cursor_advanced") is False
            and rejected.get("sampler_refreshed") is False
            and isinstance(preupdate, dict)
            and preupdate.get("accepted") is True
            and isinstance(health, dict)
            and health.get("accepted") is True
            and isinstance(acceptance, dict)
            and acceptance.get("passed") is True
        ):
            raise CorrectionGateRepairQualificationV2Error(
                "qualification candidate health or rollback differs"
            )
        backend = evidence["backend_correction"]
        post = evidence["post_update_policy_shift"]
        result = {
            "schema_version": 2,
            "artifact_kind": "p5_1_correction_gate_repair_gpu_qualification_v2",
            "status": "qualified",
            "protocol_id": "p5_1_preupdate_backend_health_v2_1",
            "protocol_config_sha256": _sha_file(protocol),
            "thresholds_sha256": _sha_file(thresholds_path),
            "adr": "docs/decisions/0035-minimum-support-aware-preupdate-backend-health-v2-1.md",
            "legacy_forensic_report_sha256": _sha_file(legacy_forensic_report),
            "step10_checkpoint_manifest_sha256": _sha_file(
                step10_checkpoint / "checkpoint_manifest.json"
            ),
            "replay_steps": [11, 12, 13, 14, 15, 16, 17],
            "replay_all_passed": all(row["passed"] for row in comparisons),
            "replay_comparisons": comparisons,
            "attempted_optimizer_step": 18,
            "preupdate_health": preupdate,
            "candidate_health": health,
            "candidate_acceptance_v2_1": acceptance,
            "backend_correction": {
                "raw_abs_log_p99": backend["raw_log"]["abs_p99"],
                "raw_abs_log_p999": backend["raw_log"]["abs_p999"],
                "clip_fraction": backend["clip_fraction"],
                "pooled_ess_fraction": backend["ess"]["pooled_fraction"],
            },
            "post_update_shift": {
                "raw_ratio_max": post["ratio"]["max"],
                "abs_log_p99": post["log"]["abs_p99"],
                "abs_log_p999": post["log"]["abs_p999"],
                "tail_loss_share": post["tail"]["absolute_loss_share"],
                "tail_gradient_proxy_share": post["tail"]["gradient_proxy_share"],
            },
            "rollback": rejected["rollback"],
            "accepted_optimizer_steps_after_rollback": 17,
            "data_cursor_after_rollback": 68,
            "policy_version_after_rollback": 17,
            "candidate_counts_as_optimizer_commit": False,
            "rejected_update_sha256": _sha_file(attempts[0]),
            "elapsed_seconds": time.time() - started,
            "final_access_count": 0,
            "controller_access_count": 0,
            "label_access_count": 0,
            "raw_prompt_persisted": False,
            "raw_response_tokens_persisted_in_report": False,
        }
        _atomic_json(output / "correction_gate_repair_qualification_v2.json", result)
        return result
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--step10-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-forensic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_correction_gate_repair_qualification_v2(
        source_package=args.source_package,
        source_output=args.source_output,
        step10_checkpoint=args.step10_checkpoint,
        legacy_forensic_report=args.legacy_forensic_report,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
