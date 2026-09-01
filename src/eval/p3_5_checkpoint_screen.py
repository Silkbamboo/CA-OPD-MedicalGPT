"""P3.5 checkpoint screening and one-use confirmation with frozen direct logits.

CPU imports do not import torch, Transformers, or PEFT.  Model execution reads
prompt-only rows and writes predictions for every preregistered checkpoint
before the independent scoring phase opens labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PERSIST_ROOT = Path("artifacts")
TRAIN_RUN_ID = "qwen3-4b-medical-sft-v2-ddp-epoch1-seed42"
SCREEN_RUN_ID = "qwen3-4b-medical-sft-v2-ddp-checkpoint-screen"
EXPECTED_STEPS = (149, 297, 446, 594)
EXPECTED_OPTIMIZER_STEPS = 594
EXPECTED_RECORDS = 9500
CONTROLLER_MANIFEST_SHA256 = "43558104872594bc58b90afc68c54d1e22c09edd2c1ecee47cdb00048aa514e5"
BASE_ARTIFACT_MANIFEST_SHA256 = "0e783f8a961ae9c94389daf75d2e93836cf1f448b9cc0866323935b945c847f6"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SFT_MANIFEST_SHA256 = "99e19126f94d2620cc8ecef9aa3c91c10cd51b0be00aafd7786774b90649ec62"
SFT_MANIFEST_RELATIVE_PATH = "data/manifests/sft_v2/medical_sft_v2_manifest.json"
SCREEN_STAGE = "p3_5_checkpoint_screen"
SUPERVISION_FIELDS = frozenset(
    {"answer", "answer_idx", "answer_index", "gold", "label", "solution", "response"}
)


class ScreenError(RuntimeError):
    """Fail-closed P3.5 screening or confirmation violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_adapter_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = directory / name
        if not path.is_file():
            raise ScreenError(f"checkpoint lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def screen_gate(medical_correct: int) -> str:
    if not 0 <= int(medical_correct) <= 300:
        raise ScreenError("medical correct count is invalid")
    if medical_correct >= 228:
        return "pass"
    if medical_correct >= 210:
        return "ambiguous"
    return "fail"


def select_candidate(results: Mapping[int, int]) -> int | None:
    eligible = [(int(correct), int(step)) for step, correct in results.items() if screen_gate(correct) == "pass"]
    if not eligible:
        return None
    best_correct = max(correct for correct, _ in eligible)
    return min(step for correct, step in eligible if correct == best_correct)


def epoch_two_allowed(results: Mapping[int, int]) -> bool:
    if set(map(int, results)) != set(EXPECTED_STEPS) or select_candidate(results) is not None:
        return False
    values = {int(step): int(correct) for step, correct in results.items()}
    return (
        values[594] >= 223
        and values[594] >= values[446]
        and values[594] == max(values.values())
    )


def validate_checkpoint_inventory(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if tuple(int(item.get("step", -1)) for item in entries) != EXPECTED_STEPS:
        raise ScreenError("checkpoint steps differ from the preregistered inventory")
    validated = []
    for item in entries:
        step = int(item["step"])
        directory = Path(str(item.get("path") or "")).resolve()
        ordered = _ordered_adapter_sha256(directory)
        if item.get("sha256") is not None and str(item["sha256"]) != ordered:
            raise ScreenError(f"checkpoint-{step} ordered adapter SHA mismatch")
        verification = directory / "verification.json"
        trainer_state_path = directory / "trainer_state.json"
        trainer_state = (
            json.loads(trainer_state_path.read_text(encoding="utf-8"))
            if trainer_state_path.is_file()
            else {}
        )
        if verification.is_file():
            payload = json.loads(verification.read_text(encoding="utf-8"))
            if (
                payload.get("step") != step
                or payload.get("finite") is not True
                or payload.get("lora_b_nonzero") is not True
                or payload.get("adapter_model_sha256")
                != _sha256(directory / "adapter_model.safetensors")
            ):
                raise ScreenError(f"checkpoint-{step} verification drift")
        validated.append(
            {
                "step": step,
                "path": str(directory),
                "adapter_sha256": ordered,
                "adapter_model_sha256": _sha256(directory / "adapter_model.safetensors"),
                "task_step_counts": trainer_state.get("task_step_counts"),
            }
        )
    return validated


def confirmation_execution_row(row: Mapping[str, Any]) -> dict[str, Any]:
    leaked = SUPERVISION_FIELDS & set(row)
    if leaked:
        raise ScreenError(f"confirmation prompt contains supervision: {sorted(leaked)}")
    role = str(row.get("target_role") or "")
    if "final" in role:
        raise ScreenError("confirmation execution cannot read final")
    if role != "medical_teacher_confirmation_dev":
        raise ScreenError("confirmation source role is invalid")
    result = dict(row)
    result["confirmation_source_role"] = role
    result["target_role"] = "medical_controller_dev"
    return result


def cpu_preflight(*, output_dir: Path, require_gpu: bool) -> dict[str, Any]:
    if output_dir.exists():
        raise ScreenError("output directory must be new")
    if require_gpu and os.environ.get("CA_OPD_ALLOW_P3_5_SCREEN_GPU") != "1":
        raise ScreenError("P3.5 checkpoint GPU execution is not authorized")
    return {
        "status": "PASS",
        "cpu_dry_run": not require_gpu,
        "gpu_used": False,
        "model_weights_loaded": False,
        "final_authorized": False,
    }


def write_aggregate_summary(path: Path, payload: Mapping[str, Any]) -> None:
    if {"predictions", "labels"} & set(payload):
        raise ScreenError("aggregate summary cannot embed predictions or labels")
    _atomic_json(path, payload)


def sft_manifest_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_sha256": _sha256(path),
        "schema_version": payload.get("schema_version"),
        "final_authorized": payload.get("final_authorized"),
    }


def sft_manifest_matches_source(copy: Path, source: Path, expected_source_sha256: str) -> bool:
    if _sha256(source) != expected_source_sha256:
        return False
    return json.loads(copy.read_text(encoding="utf-8")) == json.loads(
        source.read_text(encoding="utf-8")
    )


def _load_screen_inputs(training_run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads((training_run / "summary.json").read_text(encoding="utf-8"))
    coverage = summary.get("sample_coverage", {})
    if (
        summary.get("status") != "training_complete_pending_controller"
        or summary.get("optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or summary.get("world_size") != 2
        or summary.get("global_effective_batch") != 16
        or coverage.get("global_unique_samples") != EXPECTED_RECORDS
        or coverage.get("missing_samples") != 0
        or coverage.get("duplicate_samples") != 0
    ):
        raise ScreenError("formal SFT training identity/coverage is invalid")
    data_copy = training_run / "data_manifest.json"
    data_identity = sft_manifest_identity(data_copy)
    data_source = REPO_ROOT / SFT_MANIFEST_RELATIVE_PATH
    if (
        not sft_manifest_matches_source(data_copy, data_source, SFT_MANIFEST_SHA256)
        or data_identity["schema_version"] != 2
        or data_identity["final_authorized"] is not False
    ):
        raise ScreenError("SFT manifest identity drift")
    index = json.loads((training_run / "checkpoints/index.json").read_text(encoding="utf-8"))
    entries = [item for item in index.get("checkpoints", []) if item.get("step") in EXPECTED_STEPS]
    return validate_checkpoint_inventory(entries), summary


def formal_preflight(*, training_run: Path, output_dir: Path, require_gpu: bool) -> dict[str, Any]:
    base = cpu_preflight(output_dir=output_dir, require_gpu=require_gpu)
    inventory, _ = _load_screen_inputs(training_run)
    controller_manifest = REPO_ROOT / "data/manifests/frozen_v2/controller_manifest.json"
    base_manifest = PERSIST_ROOT / "outputs/qwen3-4b-controller-v2-direct-logit-reeval-retry2/artifact_manifest.json"
    if _sha256(controller_manifest) != CONTROLLER_MANIFEST_SHA256:
        raise ScreenError("controller manifest SHA mismatch")
    if _sha256(base_manifest) != BASE_ARTIFACT_MANIFEST_SHA256:
        raise ScreenError("B0 artifact manifest SHA mismatch")
    if require_gpu:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        if status:
            raise ScreenError("formal checkpoint screening requires a clean worktree")
    return {**base, "checkpoint_inventory": inventory, "reuse_b0": True}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _screen_summary_row(step: int, metrics: Mapping[str, Any], paired: Mapping[str, Any], sha: str) -> dict[str, Any]:
    medical_accuracy = float(metrics["medical_accuracy"])
    general_micro_accuracy = float(metrics["general_micro_accuracy"])
    correct = round(medical_accuracy * 300)
    return {
        "checkpoint_step": step,
        "adapter_sha256": sha,
        "medical_correct": correct,
        "medical_total": 300,
        "medical_accuracy": medical_accuracy,
        "general_correct": round(general_micro_accuracy * 209),
        "general_total": 209,
        "general_micro_accuracy": general_micro_accuracy,
        "general_macro_accuracy": float(metrics["general_macro_accuracy"]),
        "per_subject_accuracy": dict(metrics["per_subject_accuracy"]),
        "candidate_margin_distribution": dict(metrics["candidate_margin_distribution"]),
        "score_margin_le_repeat_tolerance_count": int(
            metrics["score_margin_le_repeat_tolerance_count"]
        ),
        "screen_gate": screen_gate(correct),
        "paired_stats": dict(paired),
    }


def score_existing_predictions(
    *, training_run: Path, output_dir: Path, prediction_execution_git_sha: str
) -> dict[str, Any]:
    """CPU-only recovery boundary after all four model routes were released."""

    from src.eval.controller_v2_runtime import _compact_score, _label_map, _track_metrics
    from src.eval.paired_stats import paired_comparison

    if len(prediction_execution_git_sha) != 40:
        raise ScreenError("prediction execution Git SHA is invalid")
    inventory, _ = _load_screen_inputs(training_run)
    if not output_dir.is_dir() or (output_dir / "summary.json").exists():
        raise ScreenError("score-existing requires an unscored checkpoint output directory")
    prediction_paths = {
        int(checkpoint["step"]): output_dir / f"checkpoint-{checkpoint['step']}-choice-predictions.jsonl"
        for checkpoint in inventory
    }
    for step, path in prediction_paths.items():
        smoke = output_dir / f"checkpoint-{step}-micro-smoke.json"
        if not path.is_file() or sum(1 for _ in _iter_jsonl(path)) != 509 or not smoke.is_file():
            raise ScreenError(f"checkpoint-{step} prediction/smoke artifact is incomplete")

    manifest_path = REPO_ROOT / "data/manifests/frozen_v2/controller_manifest.json"
    roles = ("medical_controller_dev", "general_controller_dev")
    labels = _label_map(manifest_path, roles)
    b0_root = PERSIST_ROOT / "outputs/qwen3-4b-controller-v2-direct-logit-reeval-retry2"
    b0_scored = _compact_score(_iter_jsonl(b0_root / "b0_choice_predictions.jsonl"), labels)
    b0_metrics = _track_metrics(b0_scored)
    results: dict[int, dict[str, Any]] = {}
    for checkpoint in inventory:
        step = int(checkpoint["step"])
        scored = _compact_score(_iter_jsonl(prediction_paths[step]), labels)
        metrics = _track_metrics(scored)
        paired = paired_comparison(b0_scored, scored, seed=42)
        results[step] = _screen_summary_row(
            step, metrics, paired, checkpoint["adapter_sha256"]
        )
        if checkpoint.get("task_step_counts") is not None:
            results[step]["checkpoint_task_counts"] = checkpoint["task_step_counts"]
    correct_by_step = {step: value["medical_correct"] for step, value in results.items()}
    selected = select_candidate(correct_by_step)
    summary = {
        "schema_version": 1,
        "run_id": SCREEN_RUN_ID,
        "status": "screen_complete",
        "choice_backend": "transformers_direct_logits",
        "base_revision": BASE_REVISION,
        "controller_manifest_sha256": CONTROLLER_MANIFEST_SHA256,
        "b0_reused": True,
        "b0_metrics": b0_metrics,
        "checkpoint_results": {str(step): value for step, value in results.items()},
        "selected_checkpoint_step": selected,
        "epoch_two_allowed": epoch_two_allowed(correct_by_step),
        "prediction_execution_git_sha": prediction_execution_git_sha,
        "scoring_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "labels_opened_after_all_models_released": True,
        "final_authorized": False,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    write_aggregate_summary(output_dir / "summary.json", summary)
    manifest_files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest_files.append({"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size})
    _atomic_json(output_dir / "artifact_manifest.json", {
        "schema_version": 1,
        "run_id": SCREEN_RUN_ID,
        "stage": SCREEN_STAGE,
        "controller_manifest_sha256": CONTROLLER_MANIFEST_SHA256,
        "base_artifact_manifest_sha256": BASE_ARTIFACT_MANIFEST_SHA256,
        "choice_backend": "transformers_direct_logits",
        "prediction_execution_git_sha": prediction_execution_git_sha,
        "scoring_git_sha": summary["scoring_git_sha"],
        "files": manifest_files,
        "final_authorized": False,
    })
    return summary


def run_screen(*, training_run: Path, output_dir: Path) -> dict[str, Any]:  # pragma: no cover - GPU
    """Execute all prompt-only checkpoint predictions, then open labels once."""

    from src.eval.controller_v2_runtime import (
        _compact_score,
        _label_map,
        _load_choice_smoke_ids,
        _track_metrics,
        iter_prompt_rows,
        release_model_execution,
        write_prediction_artifact,
    )
    from src.eval.direct_logit_scorer import (
        load_direct_logit_route,
        run_direct_choice_rows,
        validate_direct_logit_repetitions,
    )
    from src.eval.paired_stats import paired_comparison

    preflight = formal_preflight(training_run=training_run, output_dir=output_dir, require_gpu=True)
    if os.environ.get("CA_OPD_CONFIRM_RUN") != SCREEN_RUN_ID:
        raise ScreenError("P3.5 checkpoint screen run identity is not confirmed")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    controller_config_path = REPO_ROOT / "configs/eval/qwen3_4b/controller_v2.yaml"
    base_controller = yaml.safe_load(controller_config_path.read_text(encoding="utf-8"))
    manifest_path = REPO_ROOT / "data/manifests/frozen_v2/controller_manifest.json"
    roles = ("medical_controller_dev", "general_controller_dev")
    smoke_ids = _load_choice_smoke_ids(
        REPO_ROOT / "data/manifests/frozen_v2/controller_v2_length_smoke.json"
    )
    b0_root = PERSIST_ROOT / "outputs/qwen3-4b-controller-v2-direct-logit-reeval-retry2"
    historical_b0 = json.loads(
        (b0_root / "b0_direct_logit_micro_smoke_attempt.json").read_text(encoding="utf-8")
    )["runs"]
    prediction_paths: dict[int, Path] = {}
    smoke_evidence: dict[int, Any] = {}
    try:
        for checkpoint in preflight["checkpoint_inventory"]:
            step = int(checkpoint["step"])
            config = yaml.safe_load(controller_config_path.read_text(encoding="utf-8"))
            config["model"]["medical_lora_path"] = checkpoint["path"]
            model = tokenizer = None
            try:
                model, tokenizer, encode, _ = load_direct_logit_route(config, "B1", device="cuda:0")
                smoke_rows = list(iter_prompt_rows(manifest_path, roles, selected_ids=smoke_ids))
                candidate_runs = [
                    list(run_direct_choice_rows(smoke_rows, model=model, tokenize=encode,
                                                require_expected_qwen_ids=True))
                    for _ in range(3)
                ]
                smoke_evidence[step] = validate_direct_logit_repetitions(
                    {"B0": historical_b0, "B1": candidate_runs}
                )
                _atomic_json(output_dir / f"checkpoint-{step}-micro-smoke.json", smoke_evidence[step])
                prediction_paths[step] = output_dir / f"checkpoint-{step}-choice-predictions.jsonl"
                write_prediction_artifact(
                    prediction_paths[step],
                    run_direct_choice_rows(
                        iter_prompt_rows(manifest_path, roles), model=model, tokenize=encode,
                        require_expected_qwen_ids=True,
                    ),
                )
            finally:
                model = tokenizer = None
                release_model_execution(device="cuda:0")
    except Exception:
        _atomic_json(output_dir / "failure.json", {"status": "failed", "final_authorized": False})
        raise

    # Separate scoring boundary: no model remains resident when labels are opened.
    execution_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    return score_existing_predictions(
        training_run=training_run,
        output_dir=output_dir,
        prediction_execution_git_sha=execution_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "screen", "score-existing"))
    parser.add_argument(
        "--training-run",
        default=str(PERSIST_ROOT / "outputs" / TRAIN_RUN_ID),
    )
    parser.add_argument(
        "--output",
        default=str(PERSIST_ROOT / "outputs" / SCREEN_RUN_ID),
    )
    args = parser.parse_args(argv)
    training_run = Path(args.training_run)
    output = Path(args.output)
    if args.command == "preflight":
        print(json.dumps(formal_preflight(training_run=training_run, output_dir=output, require_gpu=False), sort_keys=True))
        return 0
    if args.command == "score-existing":
        execution_sha = os.environ.get("CA_OPD_P3_5_PREDICTION_GIT_SHA", "")
        print(json.dumps(score_existing_predictions(
            training_run=training_run,
            output_dir=output,
            prediction_execution_git_sha=execution_sha,
        ), sort_keys=True))
        return 0
    print(json.dumps(run_screen(training_run=training_run, output_dir=output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
