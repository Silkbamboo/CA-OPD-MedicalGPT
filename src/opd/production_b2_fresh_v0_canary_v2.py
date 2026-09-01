"""Fresh-v0 eight-commit canary for the selected Ratio Protocol v2 package."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json


class FreshV0CanaryV2Error(RuntimeError):
    """The fresh-v0 transactional canary failed closed."""


def run_fresh_v0_canary_v2(
    *,
    source_package: Path,
    fixed_token_qualification: Path,
    formula_path: Path,
    candidate_acceptance_path: Path,
    correction_gate_qualification: Path,
    output: Path,
    accepted_steps: int = 8,
) -> dict[str, Any]:
    source_package = Path(source_package).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FreshV0CanaryV2Error("fresh-v0 canary output must be fresh")
    if accepted_steps not in range(4, 9):
        raise FreshV0CanaryV2Error("fresh-v0 canary must use four to eight commits")
    if os.environ.get("CA_OPD_ALLOW_P5_1_CANARY_GPU") != "1":
        raise FreshV0CanaryV2Error("fresh-v0 canary GPU authorization env is absent")
    qualification_path = Path(fixed_token_qualification).resolve()
    qualification = _json(qualification_path)
    selected_lr = qualification.get("selected_common_learning_rate")
    formula = Path(formula_path).resolve()
    formula_sha256 = hashlib.sha256(formula.read_bytes()).hexdigest()
    candidate_acceptance_file = Path(candidate_acceptance_path).resolve()
    candidate_acceptance = _json(candidate_acceptance_file)
    candidate_acceptance_sha256 = hashlib.sha256(
        candidate_acceptance_file.read_bytes()
    ).hexdigest()
    correction_qualification_path = Path(correction_gate_qualification).resolve()
    correction_qualification = _json(correction_qualification_path)
    preupdate_protocol_path = Path(
        "configs/opd/preupdate_backend_health_v2_1.json"
    ).resolve()
    preupdate_protocol = _json(preupdate_protocol_path)
    preupdate_protocol_sha256 = hashlib.sha256(
        preupdate_protocol_path.read_bytes()
    ).hexdigest()
    if not (
        correction_qualification.get("status") == "qualified"
        and correction_qualification.get("protocol_id")
        == "p5_1_preupdate_backend_health_v2_1"
        and correction_qualification.get("protocol_config_sha256")
        == preupdate_protocol_sha256
        and correction_qualification.get("candidate_counts_as_optimizer_commit")
        is False
        and correction_qualification.get("final_access_count") == 0
        and correction_qualification.get("rollback", {}).get("rollback_verified")
        is True
        and preupdate_protocol.get("common_methods") == ["B2", "IDT", "CA-OPD"]
    ):
        raise FreshV0CanaryV2Error("correction gate repair qualification differs")
    if not (
        candidate_acceptance.get("schema_version") == 1
        and candidate_acceptance.get("protocol_id")
        == "p5_1_candidate_acceptance_v2_1"
        and candidate_acceptance.get("fresh_optimizer_direction_hard_gate") is True
        and candidate_acceptance.get("accumulated_adam_same_batch_monotonicity")
        == "diagnostic_only"
        and set(candidate_acceptance.get("common_methods", []))
        == {"B2", "IDT", "CA-OPD"}
        and candidate_acceptance.get("formula_sha256") == formula_sha256
    ):
        raise FreshV0CanaryV2Error("candidate acceptance v2.1 differs")
    bounded = qualification.get("bounded_influence_protocol")
    if not (
        qualification.get("passed") is True
        and selected_lr in (3.0e-5, 1.0e-5)
        and qualification.get("bounded_influence_qualified") is True
        and qualification.get("bounded_influence_formula_sha256")
        == formula_sha256
        and isinstance(bounded, Mapping)
    ):
        raise FreshV0CanaryV2Error("fixed-token qualification did not select a common LR")
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
                "selected_common_learning_rate": float(selected_lr),
                "thresholds": thresholds,
                "canary_only": True,
            },
            "bounded_influence_v2": deepcopy(dict(bounded)),
            "candidate_acceptance_v2_1": deepcopy(dict(candidate_acceptance)),
            "preupdate_backend_health_v2_1": deepcopy(dict(preupdate_protocol)),
        }
    )
    config["protocol"]["learning_rate"] = float(selected_lr)
    config["protocol"]["three_policy_formula_path"] = str(formula)
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
        FormalB2SessionV2,
        validate_formal_step_health_v2,
    )

    session: FormalB2SessionV2 | None = None
    started = time.time()
    records: list[dict[str, Any]] = []
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
            raise FreshV0CanaryV2Error("canary did not start from fresh zero-effect LoRA")
        initial_registry = session._registry_count()
        initial_models = session._model_count()
        for step_index in range(accepted_steps):
            rows = resolve_formal_b2_schedule_batch(authority, schedule, step_index=step_index)
            record = session.run_formal_step_v2(
                step_index=step_index, prompt_rows=rows, max_new_tokens=1024
            )
            records.append(record)
            validate_formal_step_health_v2(
                records,
                initial_registry_count=initial_registry,
                initial_model_count=initial_models,
            )
            session.release_transient_step_artifacts_v1(step_index + 1)
        rejected = list((output / "rejected_updates_v2").glob("attempt_*.json"))
        result = {
            "schema_version": 2,
            "artifact_kind": "p5_1_fresh_v0_gpu_canary_v2",
            "passed": len(records) == accepted_steps and not rejected,
            "accepted_optimizer_commits": len(records),
            "rejected_attempts": len(rejected),
            "selected_common_learning_rate": selected_lr,
            "bounded_influence_v2": deepcopy(dict(bounded)),
            "bounded_influence_formula_sha256": formula_sha256,
            "candidate_acceptance_v2_1": deepcopy(dict(candidate_acceptance)),
            "candidate_acceptance_sha256": candidate_acceptance_sha256,
            "preupdate_backend_health_v2_1": deepcopy(dict(preupdate_protocol)),
            "preupdate_backend_health_sha256": preupdate_protocol_sha256,
            "correction_gate_qualification_sha256": hashlib.sha256(
                correction_qualification_path.read_bytes()
            ).hexdigest(),
            "fixed_token_qualification_sha256": hashlib.sha256(
                qualification_path.read_bytes()
            ).hexdigest(),
            "fresh_v0_identity": identity,
            "policy_transition": f"v0_to_v{len(records)}",
            "source_counts": {
                "medical_opd_o1": 2 * len(records),
                "medical_opd_cmb": 2 * len(records),
            },
            "transaction_commit_passed": True,
            "rollback_primitive_previously_fixed_token_verified": qualification.get("rollback_passed"),
            "fresh_adapter_reload_each_step_passed": all(
                row["runtime_adapter_sha256"]
                == row["fresh_adapter_sha256"]
                == row["trainer_authority_sha256"]
                for row in records
            ),
            "teacher_gradient_tensor_count": sum(row["teacher_gradient_tensor_count"] for row in records),
            "base_gradient_tensor_count": sum(row["base_gradient_tensor_count"] for row in records),
            "trainable_tensor_count": 504,
            "registry_count": records[-1]["registry_count"],
            "model_count": records[-1]["model_count"],
            "minimum_disk_free_bytes": min(row["disk_remaining_bytes"] for row in records),
            "gpu_peak_bytes": {
                "gpu0": max(row["gpu_memory_bytes"]["gpu0_peak"] for row in records),
                "gpu1": max(row["gpu_memory_bytes"]["gpu1_peak"] for row in records),
            },
            "canary_weights_promoted_to_formal": False,
            "final_access_count": 0,
            "controller_access_count": 0,
            "label_access_count": 0,
            "elapsed_seconds": time.time() - started,
            "platform_actual_cost_cny": None,
        }
        if not result["passed"]:
            raise FreshV0CanaryV2Error("fresh-v0 canary did not complete all accepted commits")
        _atomic_json(output / "canary.json", result)
        return result
    finally:
        if session is not None:
            session.close()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--fixed-token-qualification", type=Path, required=True)
    parser.add_argument("--formula-path", type=Path, required=True)
    parser.add_argument("--candidate-acceptance-path", type=Path, required=True)
    parser.add_argument("--correction-gate-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-steps", type=int, default=8)
    args = parser.parse_args(argv)
    result = run_fresh_v0_canary_v2(
        source_package=args.source_package,
        fixed_token_qualification=args.fixed_token_qualification,
        formula_path=args.formula_path,
        candidate_acceptance_path=args.candidate_acceptance_path,
        correction_gate_qualification=args.correction_gate_qualification,
        output=args.output,
        accepted_steps=args.accepted_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
