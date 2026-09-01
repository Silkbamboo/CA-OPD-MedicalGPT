"""Frozen one-use 600-item development confirmation for P6 selected methods.

The historical B0/B1 predictions are reused only after exact artifact and model
identity validation.  B2/IDT/CA routes are evaluated once after a fail-closed
``no_more_tuning`` declaration.  This module never resolves or opens final data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Mapping, Sequence

import yaml

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p3_7_confirmation import (
    CONFIRMATION_MANIFEST_SHA256,
    EXPECTED_CANDIDATE_SHA256,
    ConfirmationError,
    _iter_jsonl,
    prompt_execution_row,
)
from src.eval.p6_controller_runtime import adapter_identity_from_spec
from src.eval.p6_controller_runtime import _repeatability


REPO = Path(__file__).resolve().parents[2]
PERSIST = Path("artifacts")
MANIFEST = REPO / "data/manifests/confirmation_v1/medical_teacher_confirmation_dev_manifest.json"
PROMPTS = PERSIST / "data/medical_teacher_confirmation_dev_v1/medical_teacher_confirmation_dev.prompts.jsonl"
LABELS = PERSIST / "data/medical_teacher_confirmation_dev_v1/medical_teacher_confirmation_dev.labels.jsonl"
HISTORICAL = PERSIST / "outputs/qwen3-4b-medical-sft-v3-confirmation-step450"
HISTORICAL_SHAS = {
    "b0_choice_predictions.jsonl": "8fd03f364e81512965cc1e1e1d4144349df7fe2fb058ee1ab7bc27bcacd48546",
    "b1_choice_predictions.jsonl": "3bcfde89ffefc800daedffab9fb8db9f4c62a1dae0a633ee225529db47d54307",
    "summary.json": "1ddd6d75a3fe477b41be2f9aba75206ba9145354da7e8b2a04a592089a967e94",
    "artifact_manifest.json": "60a5f7281ea73328aeb0e46c421ad381af4e12d4ea0a310b31942fc9db2df454",
    "micro_smoke.json": "bd96a375bd27b9017d9533b576df7130506297a074782528d73f8d638a004372",
}
CONTROLLER_CONFIG_SHA256 = "88b93f41637364bfb58cc19458ee8b9830f8e58d3b00a9d988969a4cbe9ad56a"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    from src.eval.p6_controller_runtime import _atomic_json as write

    write(path, value)


def validate_confirmation_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file() or _sha(MANIFEST) != CONFIRMATION_MANIFEST_SHA256:
        raise ConfirmationError("P6 confirmation manifest SHA differs")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = {item["kind"]: item for item in value.get("artifacts", [])}
    overlap = value.get("controller_overlap", {})
    if not (
        value.get("status") == "frozen_before_candidate_results"
        and value.get("role") == "medical_teacher_confirmation_dev"
        and value.get("actual_count") == 600
        and value.get("one_use_confirmation") is True
        and value.get("prompt_label_separated") is True
        and value.get("final_authorized") is False
        and value.get("final_artifacts_opened") is False
        and all(int(overlap.get(key, -1)) == 0 for key in ("sample_id", "group_id", "content_hash"))
        and _sha(PROMPTS) == artifacts["prompts"]["sha256"]
        and _sha(LABELS) == artifacts["labels"]["sha256"]
    ):
        raise ConfirmationError("P6 confirmation isolation/freeze differs")
    return value


def validate_historical_b0_b1() -> dict[str, Any]:
    for name, expected in HISTORICAL_SHAS.items():
        path = HISTORICAL / name
        if not path.is_file() or _sha(path) != expected:
            raise ConfirmationError("historical B0/B1 confirmation artifact differs")
    summary = json.loads((HISTORICAL / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((HISTORICAL / "artifact_manifest.json").read_text(encoding="utf-8"))
    config = REPO / "configs/eval/qwen3_4b/controller_v2.yaml"
    teacher = PERSIST / "outputs/qwen3-4b-medical-sft-v3-mcq-dominant-seed42/checkpoint-450"
    if not (
        _sha(config) == CONTROLLER_CONFIG_SHA256
        and _ordered_adapter_sha256(teacher) == EXPECTED_CANDIDATE_SHA256
        and summary.get("candidate_adapter_sha256") == EXPECTED_CANDIDATE_SHA256
        and summary.get("confirmation_manifest_sha256") == CONFIRMATION_MANIFEST_SHA256
        and summary.get("b0_correct") == 443
        and summary.get("candidate_correct") == 467
        and manifest.get("labels_opened_after_all_models_released") is True
        and manifest.get("final_authorized") is False
    ):
        raise ConfirmationError("historical B0/B1 identity/config differs")
    return summary


def _prompt_rows(selected_ids: set[str] | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    for source in _iter_jsonl(PROMPTS):
        row = prompt_execution_row(source)
        count += 1
        if selected_ids is None or str(row["sample_id"]) in selected_ids:
            yield row
    if count != 600:
        raise ConfirmationError("P6 confirmation prompt count differs")


def _load_route(adapter_path: str):  # pragma: no cover - GPU only
    from src.eval.direct_logit_scorer import load_direct_logit_route

    config_path = REPO / "configs/eval/qwen3_4b/controller_v2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["medical_lora_path"] = adapter_path
    return load_direct_logit_route(config, "B1", device="cuda:0")


def run_confirmation(
    *, routes_path: Path, no_more_tuning_path: Path, output: Path
) -> dict[str, Any]:  # pragma: no cover - authorized GPU only
    from src.eval.controller_v2_runtime import release_model_execution, write_prediction_artifact
    from src.eval.direct_logit_scorer import run_direct_choice_rows
    from src.eval.paired_stats import paired_comparison, score_label_free_predictions

    if os.environ.get("CA_OPD_ALLOW_P6_CONFIRMATION_GPU") != "1":
        raise ConfirmationError("P6 confirmation GPU authorization is absent")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip():
        raise ConfirmationError("P6 confirmation requires a clean worktree")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ConfirmationError("P6 confirmation output must be fresh")
    validate_confirmation_manifest()
    historical = validate_historical_b0_b1()
    declaration = json.loads(no_more_tuning_path.read_text(encoding="utf-8"))
    payload = json.loads(routes_path.read_text(encoding="utf-8"))
    routes = payload.get("routes") if isinstance(payload, Mapping) else None
    if not (
        declaration.get("artifact_kind") == "p6_no_more_tuning"
        and declaration.get("frozen") is True
        and declaration.get("final_access_count") == 0
        and isinstance(routes, list)
        and {str(route.get("name")) for route in routes} == {"B2", "IDT", "CA-OPD"}
        and all(route.get("selected") is True for route in routes)
    ):
        raise ConfirmationError("P6 confirmation freeze/routes differ")
    output.mkdir(parents=True)
    ids = sorted(str(row["sample_id"]) for row in _iter_jsonl(PROMPTS))
    smoke_ids = set(ids[:4])
    prediction_paths = {
        "B0": HISTORICAL / "b0_choice_predictions.jsonl",
        "B1": HISTORICAL / "b1_choice_predictions.jsonl",
    }
    repeatability: dict[str, Any] = {"B0": "historical_exact_reuse", "B1": "historical_exact_reuse"}
    identities: dict[str, Any] = {}
    try:
        for route in routes:
            name = str(route["name"])
            identity = adapter_identity_from_spec(route)
            identities[name] = identity
            model = tokenizer = None
            try:
                model, tokenizer, encode, _ = _load_route(str(identity["adapter_path"]))
                smoke = [
                    list(run_direct_choice_rows(_prompt_rows(smoke_ids), model=model, tokenize=encode))
                    for _ in range(3)
                ]
                repeatability[name] = _repeatability(smoke, tolerance=1.0e-4)
                path = output / f"{name.lower().replace('-', '_')}_choice_predictions.jsonl"
                write_prediction_artifact(
                    path,
                    run_direct_choice_rows(_prompt_rows(), model=model, tokenize=encode),
                )
                prediction_paths[name] = path
            finally:
                model = tokenizer = None
                gc.collect()
                release_model_execution(device="cuda:0")

        labels = list(_iter_jsonl(LABELS))
        if len(labels) != 600:
            raise ConfirmationError("P6 confirmation label count differs")
        scored = {
            name: score_label_free_predictions(_iter_jsonl(path), labels)
            for name, path in prediction_paths.items()
        }
        metrics = {
            name: {
                "correct": sum(bool(row["correct"]) for row in rows),
                "total": 600,
                "accuracy": sum(bool(row["correct"]) for row in rows) / 600.0,
            }
            for name, rows in scored.items()
        }
        paired_vs_b0 = {
            name: paired_comparison(scored["B0"], rows, seed=42, bootstrap_samples=10000)
            for name, rows in scored.items() if name != "B0"
        }
        paired_vs_idt = {
            name: paired_comparison(scored["IDT"], rows, seed=42, bootstrap_samples=10000)
            for name, rows in scored.items() if name != "IDT"
        }
        result = {
            "schema_version": 1,
            "artifact_kind": "p6_frozen_development_confirmation_600",
            "status": "complete",
            "manifest_sha256": CONFIRMATION_MANIFEST_SHA256,
            "historical_b0_b1_reuse": {
                "validated": True,
                "summary_sha256": HISTORICAL_SHAS["summary.json"],
                "b0_correct": historical["b0_correct"],
                "b1_correct": historical["candidate_correct"],
            },
            "route_identities": identities,
            "metrics": metrics,
            "paired_vs_b0": paired_vs_b0,
            "paired_vs_idt": paired_vs_idt,
            "prediction_sha256": {name: _sha(path) for name, path in prediction_paths.items()},
            "repeatability": repeatability,
            "labels_opened_after_all_models_released": True,
            "checkpoint_selection_after_confirmation": False,
            "no_more_tuning_sha256": _sha(no_more_tuning_path),
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "final_access_count": 0,
            "final_executed": False,
            "platform_actual_cost_cny": None,
        }
        _atomic_json(output / "development_confirmation_600.json", result)
        return result
    except Exception as error:
        _atomic_json(output / "failure.json", {
            "status": "failed", "reason": f"{type(error).__name__}: {error}",
            "final_access_count": 0,
        })
        raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--no-more-tuning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_confirmation(
        routes_path=args.routes,
        no_more_tuning_path=args.no_more_tuning,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "validate_confirmation_manifest",
    "validate_historical_b0_b1",
    "run_confirmation",
]
