"""Controller-dev selection for Formal B2 v2 adapter snapshots."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

from src.eval.controller_v2_runtime import (
    _compact_score,
    _label_map,
    _load_choice_smoke_ids,
    _track_metrics,
    iter_prompt_rows,
    load_controller_v2_config,
    release_model_execution,
    write_prediction_artifact,
)
from src.eval.direct_logit_scorer import (
    apply_deterministic_runtime,
    run_direct_choice_rows,
)
from src.eval.paired_stats import paired_comparison
from src.opd.production_b2_formal_checkpoint_v2 import validate_controller_snapshot_v2


class FormalB2ControllerV2Error(RuntimeError):
    """Controller package, prediction, label join, or selection differs."""


REGISTERED_STEPS = (30, 60, 90, 120)
BASELINE_RUN = Path(
    "artifacts/outputs/qwen3-4b-controller-v2-direct-logit-reeval-retry2"
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalB2ControllerV2Error(f"controller JSON invalid: {path.name}") from error
    if not isinstance(value, Mapping):
        raise FormalB2ControllerV2Error("controller JSON is not an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_baseline_reuse_v2(
    config: Mapping[str, Any], baseline_run: Path = BASELINE_RUN
) -> dict[str, Any]:
    summary = _json(baseline_run / "summary.json")
    manifest = _json(baseline_run / "artifact_manifest.json")
    smoke = _json(baseline_run / "direct_logit_micro_smoke.json")
    descriptors = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    required = ("b0_choice_predictions.jsonl", "b1_choice_predictions.jsonl")
    for name in required:
        item = descriptors.get(name)
        path = baseline_run / name
        if not (
            isinstance(item, Mapping)
            and path.is_file()
            and item.get("sha256") == _sha_file(path)
            and item.get("bytes") == path.stat().st_size
        ):
            raise FormalB2ControllerV2Error("B0/B1 reusable prediction SHA differs")
    expected = {
        "protocol_version": config["protocol_version"],
        "protocol_sha256": config["protocol_sha256"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "choice_backend": config["choice_score"]["backend"],
        "base_model_revision": config["model"]["revision"],
    }
    if any(summary.get(field) != value for field, value in expected.items()):
        raise FormalB2ControllerV2Error("B0/B1 baseline protocol/model/data differs")
    if not (
        smoke.get("status") == "PASS"
        and smoke.get("repeat_count") == 3
        and smoke.get("labels_opened_during_execution") is False
        and summary.get("final_authorized") is False
        and manifest.get("final_authorized") is False
    ):
        raise FormalB2ControllerV2Error("B0/B1 baseline repeatability/isolation differs")
    return {
        "passed": True,
        "baseline_run": str(baseline_run),
        "summary_sha256": _sha_file(baseline_run / "summary.json"),
        "artifact_manifest_sha256": _sha_file(baseline_run / "artifact_manifest.json"),
        "prediction_artifacts": {
            route: {
                "path": str(baseline_run / f"{route.lower()}_choice_predictions.jsonl"),
                "sha256": _sha_file(baseline_run / f"{route.lower()}_choice_predictions.jsonl"),
            }
            for route in ("B0", "B1")
        },
        "reused_without_model_load": True,
        "final_access_count": 0,
    }


def select_checkpoint_v2(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    b0_general_micro: float,
) -> dict[str, Any]:
    threshold = float(b0_general_micro) - 0.01
    feasible = [
        step
        for step in REGISTERED_STEPS
        if float(metrics[f"B2_step{step}"]["general_micro_accuracy"]) >= threshold
    ]
    if not feasible:
        return {
            "status": "constraint_not_met",
            "selected_checkpoint": None,
            "general_constraint_threshold": threshold,
            "feasible_steps": [],
        }
    selected = min(
        feasible,
        key=lambda step: (-float(metrics[f"B2_step{step}"]["medical_accuracy"]), step),
    )
    return {
        "status": "selected",
        "selected_checkpoint": f"B2_step{selected}",
        "selected_step": selected,
        "general_constraint_threshold": threshold,
        "feasible_steps": feasible,
        "tie_break": "earlier_checkpoint",
    }


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise FormalB2ControllerV2Error("prediction row is not an object")
                yield dict(value)


def _load_adapter_route(
    config: Mapping[str, Any], adapter_path: Path, *, device: str
):  # pragma: no cover - GPU only
    if device not in {"cuda:0", "cuda:1"}:
        raise FormalB2ControllerV2Error("controller device differs")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    apply_deterministic_runtime(torch_module=torch)
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["path"]),
        revision=str(model_config["tokenizer_revision"]),
        local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(model_config["path"]),
        revision=str(model_config["revision"]),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    del base
    model.eval()

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return model, tokenizer, encode


def _repeatability(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in first}
    right = {str(row["sample_id"]): row for row in second}
    if set(left) != set(right) or len(left) != 4:
        raise FormalB2ControllerV2Error("controller repeatability sample set differs")
    maximum = 0.0
    for sample_id in sorted(left):
        if left[sample_id]["predicted_label"] != right[sample_id]["predicted_label"]:
            raise FormalB2ControllerV2Error("controller prediction repeat differs")
        for label, score in left[sample_id]["candidate_scores"].items():
            maximum = max(maximum, abs(float(score) - float(right[sample_id]["candidate_scores"][label])))
    if maximum > 1.0e-4:
        raise FormalB2ControllerV2Error("controller direct-logit score repeat differs")
    return {"passed": True, "sample_count": 4, "max_abs_score_delta": maximum, "tolerance": 1.0e-4}


def run_formal_b2_controller_v2(
    *,
    config_path: Path,
    training_output: Path,
    output: Path,
    baseline_run: Path = BASELINE_RUN,
) -> dict[str, Any]:  # pragma: no cover - GPU only
    if os.environ.get("CA_OPD_ALLOW_FORMAL_B2_CONTROLLER_GPU") != "1":
        raise FormalB2ControllerV2Error("controller GPU authorization env is absent")
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FormalB2ControllerV2Error("controller output must be fresh")
    config = load_controller_v2_config(config_path)
    if config["execution"].get("final_authorized") is not False:
        raise FormalB2ControllerV2Error("controller cannot authorize final")
    baseline = validate_baseline_reuse_v2(config, Path(baseline_run).resolve())
    training_output = Path(training_output).resolve()
    snapshots: dict[int, dict[str, Any]] = {}
    for step in REGISTERED_STEPS:
        snapshot_path = training_output / "controller_snapshots" / f"step_{step:03d}"
        snapshots[step] = validate_controller_snapshot_v2(snapshot_path)
    output.mkdir(parents=True)
    manifest_path = Path(config["data"]["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    roles = list(config["data"]["roles"])
    smoke_path = Path(config["length_smoke"]["manifest_path"])
    if not smoke_path.is_absolute():
        smoke_path = Path(__file__).resolve().parents[2] / smoke_path
    smoke_ids = _load_choice_smoke_ids(smoke_path)
    prediction_paths: dict[str, dict[str, Any]] = {
        route: baseline["prediction_artifacts"][route] for route in ("B0", "B1")
    }
    repeatability: dict[str, Any] = {}
    started = time.time()
    for step in REGISTERED_STEPS:
        route = f"B2_step{step}"
        adapter = training_output / "controller_snapshots" / f"step_{step:03d}"
        model = tokenizer = None
        try:
            model, tokenizer, encode = _load_adapter_route(config, adapter, device="cuda:0")
            smoke_rows = list(iter_prompt_rows(manifest_path, roles, selected_ids=smoke_ids))
            first = list(run_direct_choice_rows(smoke_rows, model=model, tokenize=encode))
            second = list(run_direct_choice_rows(smoke_rows, model=model, tokenize=encode))
            repeatability[route] = _repeatability(first, second)
            prediction_paths[route] = write_prediction_artifact(
                output / f"{route.lower()}_choice_predictions.jsonl",
                run_direct_choice_rows(
                    iter_prompt_rows(manifest_path, roles), model=model, tokenize=encode
                ),
            )
        finally:
            model = None
            tokenizer = None
            gc.collect()
            release_model_execution(device="cuda:0")

    # Labels are opened only after every route prediction is immutable and all
    # model objects have been released.
    labels = _label_map(manifest_path, roles)
    scored: dict[str, list[dict[str, Any]]] = {}
    for route, descriptor in prediction_paths.items():
        scored[route] = _compact_score(_iter_jsonl(Path(descriptor["path"])), labels)
    metrics = {route: _track_metrics(rows) for route, rows in scored.items()}
    paired = {
        route: paired_comparison(scored["B0"], scored[route], seed=42, bootstrap_samples=10_000)
        for route in ("B1", "B2_step30", "B2_step60", "B2_step90", "B2_step120")
    }
    selection = select_checkpoint_v2(
        metrics, b0_general_micro=float(metrics["B0"]["general_micro_accuracy"])
    )
    result = {
        "schema_version": 2,
        "artifact_kind": "p5_1_formal_b2_controller_dev_v2",
        "status": (
            "formal_b2_controller_selected"
            if selection["status"] == "selected"
            else "constraint_not_met"
        ),
        "controller_protocol_version": config["protocol_version"],
        "controller_protocol_sha256": config["protocol_sha256"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "choice_backend": "transformers_direct_logits",
        "baseline_reuse": baseline,
        "checkpoint_snapshots": snapshots,
        "prediction_artifacts": prediction_paths,
        "repeatability": repeatability,
        "metrics": metrics,
        "paired_vs_b0": paired,
        "selection": selection,
        "selected_checkpoint": selection["selected_checkpoint"],
        "general_constraint": "General micro >= B0 General micro - 1.0pp",
        "labels_opened_after_all_predictions_frozen": True,
        "training_process_controller_access_count": 0,
        "controller_access_count": 1,
        "final_access_count": 0,
        "final_executed": False,
        "elapsed_seconds": time.time() - started,
        "platform_actual_cost_cny": None,
    }
    _atomic_json(output / "controller.json", result)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    args = parser.parse_args(argv)
    result = run_formal_b2_controller_v2(
        config_path=args.config,
        training_output=args.training_output,
        output=args.output,
        baseline_run=args.baseline_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "run_formal_b2_controller_v2",
    "select_checkpoint_v2",
    "validate_baseline_reuse_v2",
]
