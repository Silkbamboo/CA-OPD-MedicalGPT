"""One-use P3.7 confirmation for a screen-selected SFT-v3 checkpoint.

Imports are CPU safe.  GPU execution consumes prompt-only rows for both routes,
releases both models, and only then opens the physically separate label file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PERSIST_ROOT = Path("artifacts")
RUN_ID = "qwen3-4b-medical-sft-v3-confirmation-step450"
CONFIRMATION_MANIFEST_SHA256 = "f8adb4464c6da9ba9bb6632244d41d2b790e2bc08b707c6d5fc31694b2ea8556"
EXPECTED_CANDIDATE_STEP = 450
EXPECTED_CANDIDATE_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
SUPERVISION_FIELDS = frozenset(
    {"answer", "answer_idx", "answer_index", "gold", "label", "solution", "response"}
)


class ConfirmationError(RuntimeError):
    """Fail-closed confirmation protocol error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prompt_execution_row(row: Mapping[str, Any]) -> dict[str, Any]:
    leaked = SUPERVISION_FIELDS & set(row)
    if leaked:
        raise ConfirmationError(f"confirmation prompt contains supervision: {sorted(leaked)}")
    role = str(row.get("target_role") or "")
    if "final" in role or role != "medical_teacher_confirmation_dev":
        raise ConfirmationError("confirmation execution role is invalid")
    result = dict(row)
    result["confirmation_source_role"] = role
    result["target_role"] = "medical_controller_dev"
    return result


def confirmation_outcome(
    b0_correct: int, candidate_correct: int, bootstrap_ci: tuple[float, float]
) -> dict[str, Any]:
    if not 0 <= b0_correct <= 600 or not 0 <= candidate_correct <= 600:
        raise ConfirmationError("confirmation correct count is invalid")
    if candidate_correct > b0_correct and float(bootstrap_ci[0]) > 0:
        return {
            "status": "confirmed",
            "teacher_knowledge_ready": True,
            "teacher_operational_candidate": True,
            "statistical_evidence": "confirmed",
        }
    if candidate_correct > b0_correct:
        return {
            "status": "operational_uncertain",
            "teacher_knowledge_ready": "uncertain",
            "teacher_operational_candidate": True,
            "statistical_evidence": "uncertain",
        }
    return {
        "status": "contradicted",
        "teacher_knowledge_ready": False,
        "teacher_operational_candidate": False,
        "statistical_evidence": "contradicted",
    }


def _paths() -> dict[str, Path]:
    root = PERSIST_ROOT / "data/medical_teacher_confirmation_dev_v1"
    return {
        "manifest": REPO_ROOT / "data/manifests/confirmation_v1/medical_teacher_confirmation_dev_manifest.json",
        "prompts": root / "medical_teacher_confirmation_dev.prompts.jsonl",
        "labels": root / "medical_teacher_confirmation_dev.labels.jsonl",
        "screen": PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-checkpoint-screen/summary.json",
        "candidate": PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-mcq-dominant-seed42/checkpoint-450",
        "output": PERSIST_ROOT / "outputs" / RUN_ID,
    }


def preflight(*, require_gpu: bool) -> dict[str, Any]:
    paths = _paths()
    if _sha256(paths["manifest"]) != CONFIRMATION_MANIFEST_SHA256:
        raise ConfirmationError("confirmation manifest SHA mismatch")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen_before_candidate_results"
        or manifest.get("actual_count") != 600
        or manifest.get("final_authorized") is not False
        or manifest.get("final_artifacts_opened") is not False
        or manifest.get("prompt_label_separated") is not True
    ):
        raise ConfirmationError("confirmation manifest contract drift")
    artifacts = {item["kind"]: item for item in manifest["artifacts"]}
    if _sha256(paths["prompts"]) != artifacts["prompts"]["sha256"]:
        raise ConfirmationError("confirmation prompt artifact SHA mismatch")
    screen = json.loads(paths["screen"].read_text(encoding="utf-8"))
    result = screen.get("checkpoint_results", {}).get(str(EXPECTED_CANDIDATE_STEP), {})
    if (
        screen.get("selected_checkpoint_step") != EXPECTED_CANDIDATE_STEP
        or result.get("screen_gate") != "pass"
        or result.get("adapter_sha256") != EXPECTED_CANDIDATE_SHA256
        or result.get("medical_correct", 0) < 228
    ):
        raise ConfirmationError("confirmation candidate is not the preregistered screen winner")
    from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256

    if _ordered_adapter_sha256(paths["candidate"]) != EXPECTED_CANDIDATE_SHA256:
        raise ConfirmationError("confirmation candidate adapter SHA mismatch")
    if paths["output"].exists():
        raise ConfirmationError("confirmation output directory must be new")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if require_gpu and status:
        raise ConfirmationError("confirmation GPU execution requires a clean worktree")
    if require_gpu and (
        os.environ.get("CA_OPD_ALLOW_P3_7_CONFIRMATION_GPU") != "1"
        or os.environ.get("CA_OPD_CONFIRM_RUN") != RUN_ID
    ):
        raise ConfirmationError("confirmation GPU execution is not explicitly authorized")
    return {
        "status": "PASS",
        "candidate_step": EXPECTED_CANDIDATE_STEP,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "confirmation_manifest_sha256": CONFIRMATION_MANIFEST_SHA256,
        "labels_opened": False,
        "final_authorized": False,
    }


def _prompt_rows(path: Path, *, selected_ids: set[str] | None = None) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for source in _iter_jsonl(path):
        row = prompt_execution_row(source)
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ConfirmationError("confirmation prompts contain a missing/duplicate sample_id")
        seen.add(sample_id)
        if selected_ids is None or sample_id in selected_ids:
            yield row
    if selected_ids is None and len(seen) != 600:
        raise ConfirmationError("confirmation prompt count is not 600")


def run_confirmation() -> dict[str, Any]:  # pragma: no cover - authorized GPU only
    from src.eval.controller_v2_runtime import release_model_execution, write_prediction_artifact
    from src.eval.direct_logit_scorer import (
        load_direct_logit_route,
        run_direct_choice_rows,
        validate_direct_logit_repetitions,
    )
    from src.eval.paired_stats import paired_comparison, score_label_free_predictions

    gate = preflight(require_gpu=True)
    paths = _paths()
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=False)
    config_path = REPO_ROOT / "configs/eval/qwen3_4b/controller_v2.yaml"
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ids = sorted(str(row["sample_id"]) for row in _iter_jsonl(paths["prompts"]))
    if len(ids) != 600 or len(set(ids)) != 600:
        raise ConfirmationError("confirmation sample set is invalid")
    smoke_ids = set(ids[:4])
    route_runs: dict[str, list[list[dict[str, Any]]]] = {}
    prediction_paths: dict[str, Path] = {}
    try:
        for route in ("B0", "B1"):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["model"]["medical_lora_path"] = str(paths["candidate"])
            model = tokenizer = None
            try:
                model, tokenizer, encode, _ = load_direct_logit_route(config, route, device="cuda:0")
                route_runs[route] = [
                    list(
                        run_direct_choice_rows(
                            _prompt_rows(paths["prompts"], selected_ids=smoke_ids),
                            model=model,
                            tokenize=encode,
                            require_expected_qwen_ids=True,
                        )
                    )
                    for _ in range(3)
                ]
                prediction_paths[route] = output / f"{route.lower()}_choice_predictions.jsonl"
                write_prediction_artifact(
                    prediction_paths[route],
                    run_direct_choice_rows(
                        _prompt_rows(paths["prompts"]),
                        model=model,
                        tokenize=encode,
                        require_expected_qwen_ids=True,
                    ),
                )
            finally:
                model = tokenizer = None
                release_model_execution(device="cuda:0")
        repeatability = validate_direct_logit_repetitions(route_runs)
        _atomic_json(output / "micro_smoke.json", repeatability)

        # Separate scoring boundary: this is the first label-file access in this run.
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        label_entry = next(item for item in manifest["artifacts"] if item["kind"] == "labels")
        if _sha256(paths["labels"]) != label_entry["sha256"]:
            raise ConfirmationError("confirmation label artifact SHA mismatch")
        labels = list(_iter_jsonl(paths["labels"]))
        if len(labels) != 600:
            raise ConfirmationError("confirmation label count is not 600")
        scored = {
            route: score_label_free_predictions(_iter_jsonl(path), labels)
            for route, path in prediction_paths.items()
        }
        paired = paired_comparison(scored["B0"], scored["B1"], seed=42)
        b0_correct = sum(bool(row["correct"]) for row in scored["B0"])
        candidate_correct = sum(bool(row["correct"]) for row in scored["B1"])
        outcome = confirmation_outcome(
            b0_correct, candidate_correct, tuple(paired["bootstrap_95_ci"])
        )
        summary = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": "confirmation_complete",
            "candidate_step": EXPECTED_CANDIDATE_STEP,
            "candidate_adapter_sha256": EXPECTED_CANDIDATE_SHA256,
            "confirmation_manifest_sha256": CONFIRMATION_MANIFEST_SHA256,
            "b0_correct": b0_correct,
            "candidate_correct": candidate_correct,
            "total": 600,
            "b0_accuracy": b0_correct / 600,
            "candidate_accuracy": candidate_correct / 600,
            "paired_stats": paired,
            "outcome": outcome,
            "repeatability": repeatability,
            "labels_opened_after_all_models_released": True,
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                capture_output=True, check=True,
            ).stdout.strip(),
            "final_authorized": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(output / "summary.json", summary)
        files = [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        _atomic_json(output / "artifact_manifest.json", {
            **gate,
            "schema_version": 1,
            "run_id": RUN_ID,
            "stage": "p3_7_medical_teacher_confirmation",
            "files": files,
            "labels_opened_after_all_models_released": True,
            "final_authorized": False,
        })
        return summary
    except Exception as error:
        _atomic_json(output / "failure.json", {
            "status": "failed", "reason": f"{type(error).__name__}: {error}",
            "final_authorized": False,
        })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args(argv)
    result = preflight(require_gpu=False) if args.command == "preflight" else run_confirmation()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
