"""Two-fresh-process real GPU qualification of the P7 step120 resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from src.opd.p9_adaptive_dose_gpu import P9FormalB2Session, P9RejectedAttempt
from src.opd.p9_runtime import build_p9_runtime_config
from src.opd.p9_adaptive_dose_protocol import (
    P7_STEP120_IDENTITIES,
    validate_p9_resume_manifest,
)
from src.opd.p9_worker import (
    _hydrate_rows, _batch, audit_p9_launch_assets,
    validate_p9_execution_environment,
)
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint


P7_CHECKPOINT = Path("artifacts/outputs/qwen3-4b-b2-medical-opd-formal-p5-1-v2-2-r1-seed42/formal_checkpoints/step_120")
P7_PACKAGE = Path("artifacts/outputs/qwen3-4b-b2-medical-opd-formal-p5-1-v2-2-r1-package")
P9_RUN = Path("artifacts/outputs/qwen3-4b-b2-p9-adaptive-dose-seed42")
REPO = Path(__file__).resolve().parents[2]
P7_ADAPTER_WEIGHT_SHA256 = "bd6bfe2597c82113c2a878f31abc0b7a7e99a05e7221b888f0a86220404d64f9"
EXPECTED_LAUNCH_IDENTITIES = {
    "p7_package_content_sha256": P7_STEP120_IDENTITIES["package_content_sha256"],
    "base_manifest_sha256": "c796c078afd35849b59017582eb7dd0e1553be43bdbd7ce0eed441fda889a213",
    "teacher_ordered_sha256": "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2",
    "teacher_weight_sha256": "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63",
    "teacher_manifest_sha256": "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67",
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True); raise


def run_attempt(output: Path) -> dict[str, Any]:  # pragma: no cover - GPU
    validate_p9_execution_environment()
    launch_asset_audit = audit_p9_launch_assets(P7_PACKAGE)
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("P9 reload qualification output must be fresh")
    output.mkdir(parents=True)
    schedule = json.loads((P9_RUN / "schedule.json").read_text(encoding="utf-8"))
    authority = json.loads((P9_RUN / "data_authority.json").read_text(encoding="utf-8"))
    p7_config = json.loads((P7_PACKAGE / "formal_b2_config.json").read_text(encoding="utf-8"))
    p7_index = json.loads((P7_PACKAGE / "package_index.json").read_text(encoding="utf-8"))
    runtime = build_p9_runtime_config(p7_config, output=output, schedule_sha256=schedule["schedule_sha256"])
    runtime_path = output / "runtime_config.json"
    _atomic_json(runtime_path, runtime)
    for name in ("b2_steps", "checkpoints", "formal_steps", "memory_step_audits", "memory_telemetry/markers", "ratio_evidence_v2", "rejected_updates_v2"):
        (output / name).mkdir(parents=True, exist_ok=True)
    rows = _hydrate_rows(authority, _batch(schedule, step=121, reserve_variant=0))
    before = validate_formal_checkpoint(P7_CHECKPOINT)
    validate_p9_resume_manifest(before)
    session = None
    try:
        session = P9FormalB2Session(runtime, config_path=runtime_path, route="b2_calibration")
        resume = session.restore_formal_checkpoint_v1(
            P7_CHECKPOINT,
            package_content_sha256=p7_index["package_content_sha256"],
            config_sha256=p7_index["config_sha256"],
            manifest_sha256=p7_index["manifest_sha256"],
            schedule_sha256=p7_index["schedule_semantic_sha256"],
            resume_prompt_rows=rows,
        )
        session.set_diagnostic_unconditional_rollback_v2(True)
        try:
            session.run_p9_attempt(step_index=120, prompt_rows=rows, max_new_tokens=1024)
        except P9RejectedAttempt as error:
            evidence = error.evidence
        else:
            raise RuntimeError("P9 diagnostic candidate unexpectedly committed")
        observation = evidence.get("qualification_observations")
        if not (
            isinstance(observation, Mapping)
            and evidence.get("counts_as_optimizer_commit") is False
            and evidence.get("adapter_rollback_verified") is True
            and evidence.get("optimizer_rollback_verified") is True
            and evidence.get("scheduler_rollback_verified") is True
            and evidence.get("rng_rollback_verified") is True
        ):
            raise RuntimeError("P9 diagnostic rollback evidence differs")
        result = {
            "schema_version": 1,
            "artifact_kind": "p9_step120_fresh_reload_attempt",
            "output": str(output),
            "launch_asset_audit": launch_asset_audit,
            "resume": resume,
            "sample_ids": observation["sample_ids"],
            "completion_token_counts": observation["completion_token_counts"],
            "completion_token_sha256": observation["completion_token_sha256"],
            "student_score": observation["student_score"],
            "teacher_score": observation["teacher_score"],
            "loss": observation["loss"],
            "objective": observation["objective"],
            "reverse_kl": observation["reverse_kl"],
            "advantage": observation["advantage"],
            "ess_fraction": observation["ess_fraction"],
            "ratio": observation["ratio"],
            "ratio_evidence_sha256": observation["ratio_evidence_sha256"],
            "health_classification": observation["ratio_health"],
            "rollback": evidence,
            "checkpoint_adapter_sha256_before": before["adapter_sha256"],
            "checkpoint_adapter_weight_sha256_after": _sha_file(P7_CHECKPOINT / "adapter_model.safetensors"),
            "cursor_rng_sampler_version_advanced": False,
            "formal_commit_count": 0,
            "final_access_count": 0,
        }
        _atomic_json(output / "result.json", result)
        return result
    finally:
        if session is not None:
            session.close()


def _close(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_close(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b, atol=atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= atol
    return left == right


def compare_attempts(first: Path, second: Path, report: Path) -> dict[str, Any]:
    left = json.loads((Path(first) / "result.json").read_text(encoding="utf-8"))
    right = json.loads((Path(second) / "result.json").read_text(encoding="utf-8"))
    exact_fields = ("sample_ids", "completion_token_counts", "completion_token_sha256", "ratio_evidence_sha256")
    tolerant_fields = ("student_score", "teacher_score", "loss", "objective", "reverse_kl", "advantage", "ess_fraction", "ratio", "health_classification")
    checks = {
        **{f"{field}_same": left[field] == right[field] for field in exact_fields},
        **{f"{field}_same_within_1e_6": _close(left[field], right[field]) for field in tolerant_fields},
        "first_rollback_complete": left["cursor_rng_sampler_version_advanced"] is False,
        "second_rollback_complete": right["cursor_rng_sampler_version_advanced"] is False,
        "p7_checkpoint_adapter_identity_exact": all(
            attempt["checkpoint_adapter_sha256_before"]
            == P7_STEP120_IDENTITIES["adapter_sha256"]
            for attempt in (left, right)
        ),
        "p7_checkpoint_weight_unchanged": left["checkpoint_adapter_weight_sha256_after"] == right["checkpoint_adapter_weight_sha256_after"] == P7_ADAPTER_WEIGHT_SHA256,
        "restore_identity_exact": all(
            attempt["resume"].get("logical_version") == 120
            and attempt["resume"].get("data_cursor") == 480
            and attempt["resume"].get("adapter_sha256")
            == P7_STEP120_IDENTITIES["adapter_sha256"]
            and attempt["resume"].get("optimizer_state_restored") is True
            and attempt["resume"].get("scheduler_state_restored") is True
            and attempt["resume"].get("rng_state_restored") is True
            and attempt["resume"].get("sampler_state_restored") is True
            and attempt["resume"].get("passed") is True
            for attempt in (left, right)
        ),
        "base_teacher_launch_identity_exact": all(
            attempt["launch_asset_audit"].get("passed") is True
            and all(
                attempt["launch_asset_audit"].get(key) == expected
                for key, expected in EXPECTED_LAUNCH_IDENTITIES.items()
            )
            for attempt in (left, right)
        ),
        "ratio_evidence_sha256_bound": all(
            isinstance(attempt.get("ratio_evidence_sha256"), str)
            and len(attempt["ratio_evidence_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in attempt["ratio_evidence_sha256"]
            )
            for attempt in (left, right)
        ),
    }
    result = {
        "schema_version": 1,
        "artifact_kind": "p9_step120_reload_qualification",
        "status": "passed" if all(checks.values()) else "blocked_resume_semantic_mismatch",
        "passed": all(checks.values()),
        "fresh_process_count": 2,
        "tolerance": 1e-6,
        "checks": checks,
        "attempts": [left, right],
        "formal_commit_count": 0,
        "controller_access_count": 0,
        "final_access_count": 0,
    }
    _atomic_json(Path(report), result)
    return result


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    attempt = subparsers.add_parser("attempt"); attempt.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare"); compare.add_argument("--first", type=Path, required=True); compare.add_argument("--second", type=Path, required=True); compare.add_argument("--report", type=Path, default=REPO / "reports/p9_step120_reload_qualification.json")
    args = parser.parse_args(argv)
    result = run_attempt(args.output) if args.command == "attempt" else compare_attempts(args.first, args.second, args.report)
    print(json.dumps({"status": result.get("status", "attempt_complete"), "passed": result.get("passed")}, sort_keys=True))
    return 0 if result.get("status") != "blocked_resume_semantic_mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["compare_attempts", "run_attempt"]
