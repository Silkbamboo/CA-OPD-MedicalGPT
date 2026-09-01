"""Deterministic step10->18 replay for the P5.1 correction-gate incident."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from src.opd.production_b2_fixed_token_qualification_v2 import (
    _diagnostic_runtime,
    _source_index,
)
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
)


class CorrectionGateForensicV2Error(RuntimeError):
    """Replay, identity, or fail-closed evidence differed."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _comparison(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = {
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
        for field, tolerance in tolerances.items()
    }
    authority_fields = (
        "input_trainer_authority_sha256",
        "trainer_authority_sha256",
        "runtime_adapter_sha256",
        "fresh_adapter_sha256",
    )
    prompt_fields = ("sample_id", "content_hash", "source")
    prompt_identity_match = [
        tuple(row[field] for field in prompt_fields) for row in new["prompt_samples"]
    ] == [
        tuple(row[field] for field in prompt_fields) for row in old["prompt_samples"]
    ]
    new_binding = new["ratio_v2"]["pool_binding"]
    old_binding = old["ratio_v2"]["pool_binding"]
    binding_fields = (
        "pool_binding_sha256",
        "input_token_sha256",
        "response_token_sha256",
        "attention_mask_sha256",
        "response_mask_sha256",
        "valid_mask_sha256",
        "valid_token_count",
    )
    binding_match = all(new_binding[field] == old_binding[field] for field in binding_fields)
    passed = bool(
        all(new[field] == old[field] for field in authority_fields)
        and prompt_identity_match
        and binding_match
        and all(value["absolute_difference"] <= value["tolerance"] for value in scalar.values())
    )
    return {
        "passed": passed,
        "authority_hash_match": all(new[field] == old[field] for field in authority_fields),
        "prompt_identity_match": prompt_identity_match,
        "token_mask_pool_hash_match": binding_match,
        "pool_binding_sha256": new_binding["pool_binding_sha256"],
        "scalar_metrics": scalar,
        "raw_prompt_persisted": False,
        "raw_response_tokens_persisted": False,
    }


def run_correction_gate_forensic_v2(
    *,
    source_package: Path,
    source_output: Path,
    step10_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    source_package = source_package.resolve()
    source_output = source_output.resolve()
    step10_checkpoint = step10_checkpoint.resolve()
    output = output.resolve()
    if output.exists():
        raise CorrectionGateForensicV2Error("forensic output is not fresh")
    index = _source_index(source_package)
    checkpoint = validate_formal_checkpoint(step10_checkpoint)
    if checkpoint.get("logical_version") != 10:
        raise CorrectionGateForensicV2Error("forensic checkpoint is not step10")
    source_config = _json(source_package / "formal_b2_config.json")
    schedule = _json(source_package / "prompt_schedule.json")
    authority = _json(source_package / "data_authority.json")
    thresholds = _json(Path("reports/p5_1_ratio_health_thresholds_v2.json").resolve())
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

    from src.opd.production_b2_formal_gpu_v2 import FormalB2SessionV2

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
        comparisons: list[dict[str, Any]] = []
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
                raise CorrectionGateForensicV2Error(
                    f"step{step_index + 1} deterministic replay differs"
                )

        rows18 = resolve_formal_b2_schedule_batch(authority, schedule, step_index=17)
        error = None
        try:
            session.run_formal_step_v2(
                step_index=17, prompt_rows=rows18, max_new_tokens=1024
            )
        except ProductionTwoStepQualificationV6Error as caught:
            error = {"type": type(caught).__name__, "message": str(caught)}
        if error is None or error["message"] != "step17 correction gate failed":
            raise CorrectionGateForensicV2Error(
                "historical correction-gate failure was not reproduced"
            )
        attempts = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
        if len(attempts) != 1:
            raise CorrectionGateForensicV2Error("correction rejection artifact differs")
        rejected = _json(attempts[0])
        evidence = rejected["ratio_evidence"]
        if not (
            rejected.get("counts_as_optimizer_commit") is False
            and rejected.get("cursor_advanced") is False
            and rejected.get("sampler_refreshed") is False
            and rejected["rollback"]["rollback_verified"] is True
            and evidence.get("legacy_gate_passed") is False
        ):
            raise CorrectionGateForensicV2Error("correction rejection state differs")
        result = {
            "schema_version": 2,
            "artifact_kind": "p5_1_formal_b2_correction_gate_forensic_v2",
            "status": "correction_gate_failure_reproduced",
            "source_run_id": source_config["run"]["run_id"],
            "source_failure_sha256": _sha_file(
                next(source_output.glob("failure_*_v2.json"))
            ),
            "step10_checkpoint_manifest_sha256": _sha_file(
                step10_checkpoint / "checkpoint_manifest.json"
            ),
            "replay_steps": [11, 12, 13, 14, 15, 16, 17],
            "replay_comparisons": comparisons,
            "replay_all_passed": all(row["passed"] for row in comparisons),
            "attempted_optimizer_step": 18,
            "failure": error,
            "rejected_update_sha256": _sha_file(attempts[0]),
            "correction_gate_evidence": evidence,
            "accepted_optimizer_steps_after_rejection": 17,
            "data_cursor_after_rejection": 68,
            "policy_version_after_rejection": 17,
            "final_access_count": 0,
            "controller_access_count": 0,
            "label_access_count": 0,
            "elapsed_seconds": time.time() - started,
            "raw_prompt_persisted": False,
            "raw_response_tokens_persisted": False,
        }
        _atomic_json(output / "correction_gate_forensic_v2.json", result)
        return result
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--step10-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_correction_gate_forensic_v2(
        source_package=args.source_package,
        source_output=args.source_output,
        step10_checkpoint=args.step10_checkpoint,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CorrectionGateForensicV2Error",
    "run_correction_gate_forensic_v2",
]
