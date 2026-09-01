"""Freeze the confirmed P3.7 Medical Teacher without authorizing OPD."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PERSIST_ROOT = Path("artifacts")
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CANDIDATE_SHA = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
CANDIDATE_WEIGHT_SHA = "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63"
RECORDS_SHA = "cf3306e118e7f946db0539cef50e3fe0ba6768a884b9ae4496864eb10cf8f5e7"
DATA_MANIFEST_SHA = "eae8df56fd9985edd27e32984b155ae9dd569eadc6e6336a858b7079050e223e"
SELECTION_SHA = "fb036851e5663560114753c1d4299b6ab969abf34cfb4117136a7afec18d2153"
CONFIRMATION_MANIFEST_SHA = "f8adb4464c6da9ba9bb6632244d41d2b790e2bc08b707c6d5fc31694b2ea8556"


class TeacherArtifactError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def build_teacher_payload(
    training: Mapping[str, Any],
    screen: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    if training.get("status") not in {
        "complete", "completed", "training_complete_pending_controller"
    } or training.get("optimizer_steps") != 600:
        raise TeacherArtifactError("training artifact is incomplete")
    if screen.get("status") not in {"complete", "screen_complete"} or screen.get("selected_checkpoint_step") != 450:
        raise TeacherArtifactError("screen did not select checkpoint 450")
    selected = screen.get("checkpoint_results", {}).get("450", {})
    if selected.get("screen_gate") != "pass" or selected.get("medical_correct") != 240:
        raise TeacherArtifactError("screen gate identity drift")
    if (
        confirmation.get("status") not in {"complete", "confirmation_complete"}
        or confirmation.get("candidate_step") != 450
        or confirmation.get("candidate_adapter_sha256") != candidate_sha
        or confirmation.get("total") != 600
        or confirmation.get("outcome", {}).get("status") != "confirmed"
    ):
        raise TeacherArtifactError("confirmation candidate identity drift")
    if (
        diagnostic.get("status") != "complete"
        or diagnostic.get("candidate_step") != 450
        or diagnostic.get("candidate_adapter_sha256") != candidate_sha
        or diagnostic.get("knowledge_metric") is not False
    ):
        raise TeacherArtifactError("diagnostic candidate identity drift")
    outcome = confirmation["outcome"]
    return {
        "schema_version": 1,
        "status": "teacher_frozen_confirmed",
        "training_artifact_valid": True,
        "teacher_screen_ready": True,
        "teacher_knowledge_ready": bool(outcome["teacher_knowledge_ready"]),
        "teacher_operational_candidate": bool(outcome["teacher_operational_candidate"]),
        "statistical_evidence": str(outcome["statistical_evidence"]),
        "open_prompt_contract_ready": bool(diagnostic["open_prompt_contract_ready"]),
        "standalone_generation_ready": bool(diagnostic["open_prompt_contract_ready"]),
        "standalone_generation_scope": "32-item Medical-O1 behavior diagnostic only",
        "OPD_scoring_backend_ready": False,
        "OPD_scoring_backend_status": "pending_narrow_gpu_calibration",
        "OPD_authorized": False,
        "final_authorized": False,
    }


def freeze() -> dict[str, Any]:
    from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256

    repo = Path(__file__).resolve().parents[2]
    train_root = PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-mcq-dominant-seed42"
    screen_root = PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-checkpoint-screen"
    confirm_root = PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-confirmation-step450"
    diagnostic_root = PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-open-diagnostic-step450-retry1"
    candidate = train_root / "checkpoint-450"
    output = PERSIST_ROOT / "outputs/qwen3-4b-medical-teacher-sft-v3-step450"
    if output.exists():
        raise TeacherArtifactError("teacher output must be new")
    if _ordered_adapter_sha256(candidate) != CANDIDATE_SHA:
        raise TeacherArtifactError("candidate adapter identity drift")
    if _sha256(candidate / "adapter_model.safetensors") != CANDIDATE_WEIGHT_SHA:
        raise TeacherArtifactError("candidate adapter weight identity drift")

    evidence_paths = {
        "training_summary": train_root / "summary.json",
        "training_artifact_manifest": train_root / "artifact_manifest.json",
        "checkpoint_index": train_root / "checkpoints/index.json",
        "screen_summary": screen_root / "summary.json",
        "screen_artifact_manifest": screen_root / "artifact_manifest.json",
        "confirmation_summary": confirm_root / "summary.json",
        "confirmation_artifact_manifest": confirm_root / "artifact_manifest.json",
        "open_diagnostic_summary": diagnostic_root / "summary.json",
        "open_diagnostic_artifact_manifest": diagnostic_root / "artifact_manifest.json",
    }
    values = {key: json.loads(path.read_text()) for key, path in evidence_paths.items()}
    payload = build_teacher_payload(
        values["training_summary"], values["screen_summary"],
        values["confirmation_summary"], values["open_diagnostic_summary"],
        candidate_sha=CANDIDATE_SHA,
    )
    if values["confirmation_summary"].get("confirmation_manifest_sha256") != CONFIRMATION_MANIFEST_SHA:
        raise TeacherArtifactError("confirmation manifest identity drift")
    checkpoint = next(
        row for row in values["checkpoint_index"]["checkpoints"] if row.get("step") == 450
    )
    if checkpoint.get("sha256") != CANDIDATE_SHA:
        raise TeacherArtifactError("checkpoint index identity drift")
    payload.update({
        "model_id": str(PERSIST_ROOT / "models/Qwen3-4B"),
        "base_model_revision": BASE_REVISION,
        "tokenizer_revision": BASE_REVISION,
        "adapter_path": str(candidate),
        "adapter_sha256": CANDIDATE_SHA,
        "adapter_weight_sha256": CANDIDATE_WEIGHT_SHA,
        "source_training_run_id": values["training_summary"]["run_id"],
        "checkpoint_step": 450,
        "checkpoint_task_counts": {"cmb": 338, "medical_o1": 112},
        "sft_v3_records_sha256": RECORDS_SHA,
        "sft_v3_manifest_sha256": DATA_MANIFEST_SHA,
        "sft_v3_selection_sha256": SELECTION_SHA,
        "confirmation_manifest_sha256": CONFIRMATION_MANIFEST_SHA,
        "screen_medical": {"correct": 240, "total": 300, "accuracy": 0.8},
        "confirmation_medical": {
            "b0_correct": 443, "candidate_correct": 467, "total": 600,
            "paired_delta": 0.04,
            "bootstrap_95_ci": values["confirmation_summary"]["paired_stats"]["bootstrap_95_ci"],
            "mcnemar_exact_two_sided_p": values["confirmation_summary"]["paired_stats"]["mcnemar"]["exact_two_sided_p"],
        },
        "seed": 42,
        "training_git_sha": values["training_artifact_manifest"].get("git_sha", "c56e92f7eef1e54702d3684dfd6f59b8183337a7"),
        "freeze_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
            capture_output=True,
        ).stdout.strip(),
        "evidence": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in evidence_paths.items()
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "teacher_manifest.json", payload)
    artifact_manifest = {
        "schema_version": 1,
        "status": "complete",
        "artifact_type": "confirmed_medical_teacher_reference",
        "adapter_copied": False,
        "adapter_sha256": CANDIDATE_SHA,
        "teacher_manifest_sha256": _sha256(output / "teacher_manifest.json"),
        "final_authorized": False,
    }
    _atomic_json(output / "artifact_manifest.json", artifact_manifest)
    return {**payload, "teacher_manifest_sha256": artifact_manifest["teacher_manifest_sha256"]}


def main() -> int:
    print(json.dumps(freeze(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
