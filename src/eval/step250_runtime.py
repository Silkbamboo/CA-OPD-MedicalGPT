"""Fail-closed checkpoint-250 Controller v2 direct-logit evaluation package.

CPU imports and preflight never import torch/Transformers.  The optional GPU
entry loads only the preregistered step-250 PEFT adapter, reuses B0 solely after
an exact compatibility attestation, and opens labels only after model release.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "controller_protocol_v2"
DIRECT_BACKEND = "transformers_direct_logits"
BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class Step250RuntimeError(RuntimeError):
    """Checkpoint-250 configuration, identity, or authorization violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _ordered_adapter_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = checkpoint / name
        if not path.is_file():
            raise Step250RuntimeError(f"step250 checkpoint lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_step250_eval_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_version",
        "status",
        "seed",
        "model",
        "checkpoint",
        "data",
        "b0_reuse",
        "choice_score",
        "teacher_gate",
        "execution",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise Step250RuntimeError("step250 evaluator config schema drift")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise Step250RuntimeError("step250 evaluator protocol drift")
    if payload.get("seed") != 42:
        raise Step250RuntimeError("step250 evaluator seed must remain 42")
    model = payload["model"]
    if (
        model.get("id") != "Qwen/Qwen3-4B"
        or model.get("revision") != BASE_REVISION
        or model.get("tokenizer_revision") != BASE_REVISION
    ):
        raise Step250RuntimeError("step250 base/tokenizer identity drift")
    checkpoint = payload["checkpoint"]
    if (
        checkpoint.get("role") != "candidate_medical_teacher"
        or checkpoint.get("step") != 250
        or checkpoint.get("source_run_id") != "qwen3-4b-sft-epoch1-seed42"
        or _HEX64.fullmatch(str(checkpoint.get("adapter_sha256") or "")) is None
        or _HEX64.fullmatch(str(checkpoint.get("manifest_sha256") or "")) is None
    ):
        raise Step250RuntimeError("step250 checkpoint identity drift")
    data = payload["data"]
    if data.get("roles") != ["medical_controller_dev", "general_controller_dev"]:
        raise Step250RuntimeError("step250 evaluation requires both controller roles and no final role")
    if any("final" in str(role) for role in data.get("roles", [])):
        raise Step250RuntimeError("step250 evaluation cannot read final")
    choice = payload["choice_score"]
    if (
        choice.get("backend") != DIRECT_BACKEND
        or choice.get("batch_size") != 1
        or choice.get("float32_log_softmax") is not True
        or choice.get("use_cache") is not False
        or choice.get("attn_implementation") != "eager"
        or choice.get("score_repeat_tolerance") != 1e-4
        or choice.get("micro_smoke_repeat_count") != 3
        or choice.get("diagnostic_parser_authority") is not False
    ):
        raise Step250RuntimeError("step250 direct-logit contract drift")
    gate = payload["teacher_gate"]
    if gate != {
        "b0_medical_correct": 219,
        "medical_total": 300,
        "ready_min_correct": 228,
        "ambiguous_min_correct": 210,
        "step500_state": "ambiguous",
        "diagnostic_metric_allowed": False,
    }:
        raise Step250RuntimeError("step250 Teacher gate drift")
    execution = payload["execution"]
    if execution.get("final_authorized") is not False or execution.get("gpu_authorized") is not False:
        raise Step250RuntimeError("step250 execution must remain GPU/final unauthorized in config")
    if execution.get("required_confirmation") != execution.get("run_id"):
        raise Step250RuntimeError("step250 confirmation/run identity drift")
    return payload


def assess_b0_compatibility(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "base_model_revision",
        "tokenizer_revision",
        "controller_manifest_sha256",
        "protocol_sha256",
        "prompt_sha256",
        "scorer_sha256",
        "sample_id_set_sha256",
        "label_artifact_attestation",
        "choice_backend",
        "batch_size",
        "float32_log_softmax",
        "final_authorized",
    )
    mismatches = [field for field in fields if expected.get(field) != actual.get(field)]
    if expected.get("final_authorized") is not False or actual.get("final_authorized") is not False:
        if "final_authorized" not in mismatches:
            mismatches.append("final_authorized")
    reusable = not mismatches
    return {
        "status": "PASS" if reusable else "INCOMPATIBLE",
        "reuse_b0": reusable,
        "requires_b0_rerun": not reusable,
        "mismatches": sorted(mismatches),
        "comparison_fields": list(fields),
        "final_authorized": False,
    }


def step250_teacher_gate(
    *,
    b0_medical_correct: int,
    step250_medical_correct: int,
    medical_total: int,
    step500_state: str,
) -> dict[str, Any]:
    if b0_medical_correct != 219 or medical_total != 300 or step500_state != "ambiguous":
        raise Step250RuntimeError("step250 gate baseline/inventory drift")
    if not 0 <= step250_medical_correct <= medical_total:
        raise Step250RuntimeError("step250 medical correct count is invalid")
    if step250_medical_correct >= 228:
        state: bool | str = True
        selected = "checkpoint-250"
    elif step250_medical_correct >= 210:
        state = "ambiguous"
        selected = None
    else:
        state = False
        selected = None
    return {
        "teacher_artifact_valid": True,
        "teacher_knowledge_ready": state,
        "selected_medical_teacher": selected,
        "step250_medical_correct": step250_medical_correct,
        "step250_medical_accuracy": step250_medical_correct / medical_total,
        "b0_medical_correct": b0_medical_correct,
        "b0_medical_accuracy": b0_medical_correct / medical_total,
        "medical_delta": (step250_medical_correct - b0_medical_correct) / medical_total,
        "ready_min_correct": 228,
        "ambiguous_min_correct": 210,
        "step500_state": step500_state,
        "step500_selectable": False,
        "diagnostic_metric_used": False,
        "final_authorized": False,
    }


def assert_step250_gpu_authorized(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> None:
    values = os.environ if environ is None else environ
    if (
        values.get("CA_OPD_ALLOW_STEP250_GPU") != "1"
        or values.get("CA_OPD_CONFIRM_RUN") != config["execution"]["required_confirmation"]
    ):
        raise Step250RuntimeError("checkpoint-250 GPU execution is not authorized")
    if config["execution"].get("final_authorized") is not False:
        raise Step250RuntimeError("checkpoint-250 GPU execution cannot authorize final")


def step250_cpu_preflight(config_path: str | Path, *, require_clean: bool = False) -> dict[str, Any]:
    config = load_step250_eval_config(config_path)
    files = {
        "checkpoint_manifest": (
            _resolve(config["checkpoint"]["manifest_path"]),
            config["checkpoint"]["manifest_sha256"],
        ),
        "controller_manifest": (
            _resolve(config["data"]["manifest_path"]),
            config["data"]["manifest_sha256"],
        ),
        "b0_compatibility": (
            _resolve(config["b0_reuse"]["compatibility_path"]),
            config["b0_reuse"]["compatibility_sha256"],
        ),
        "b0_artifact_manifest": (
            _resolve(config["b0_reuse"]["artifact_manifest_path"]),
            config["b0_reuse"]["artifact_manifest_sha256"],
        ),
        "run_card": (
            _resolve(config["execution"]["run_card_path"]),
            config["execution"]["run_card_sha256"],
        ),
    }
    for label, (path, expected) in files.items():
        if not path.is_file() or _sha256(path) != expected:
            raise Step250RuntimeError(f"step250 {label} SHA mismatch")
    checkpoint_manifest = json.loads(files["checkpoint_manifest"][0].read_text(encoding="utf-8"))
    checkpoint_path = _resolve(config["checkpoint"]["path"]).resolve()
    model_path = _resolve(config["model"]["path"]).resolve()
    if (
        checkpoint_manifest.get("checkpoint_step") != 250
        or checkpoint_manifest.get("checkpoint_role") != "candidate_medical_teacher"
        or checkpoint_manifest.get("source_run_id") != config["checkpoint"]["source_run_id"]
        or checkpoint_manifest.get("adapter_sha256") != config["checkpoint"]["adapter_sha256"]
        or checkpoint_manifest.get("base_model_revision") != config["model"]["revision"]
        or checkpoint_manifest.get("tokenizer_revision") != config["model"]["tokenizer_revision"]
        or checkpoint_manifest.get("data_manifest_sha256")
        != config["data"]["sft_manifest_sha256"]
        or checkpoint_manifest.get("final_authorized") is not False
    ):
        raise Step250RuntimeError("step250 manifest/config binding mismatch")
    if Path(str(checkpoint_manifest.get("checkpoint_path") or "")).resolve() != checkpoint_path:
        raise Step250RuntimeError("step250 checkpoint path differs from frozen manifest")
    if Path(str(checkpoint_manifest.get("base_model_path") or "")).resolve() != model_path:
        raise Step250RuntimeError("step250 model path differs from frozen manifest")
    checkpoint_index_path = Path(
        str(checkpoint_manifest.get("checkpoint_index_path") or "")
    ).resolve()
    if (
        checkpoint_manifest.get("checkpoint_index_lists_step250") is not False
        or not checkpoint_index_path.is_file()
        or _sha256(checkpoint_index_path)
        != checkpoint_manifest.get("checkpoint_index_sha256")
    ):
        raise Step250RuntimeError("step250 checkpoint index attestation mismatch")
    checkpoint_index = json.loads(checkpoint_index_path.read_text(encoding="utf-8"))
    indexed_paths = {
        Path(str(item.get("path") or "")).resolve()
        for item in checkpoint_index.get("checkpoints", [])
        if isinstance(item, dict)
    }
    if checkpoint_path in indexed_paths:
        raise Step250RuntimeError("step250 checkpoint index disclosure drift")
    if _ordered_adapter_sha256(checkpoint_path) != config["checkpoint"]["adapter_sha256"]:
        raise Step250RuntimeError("step250 adapter ordered SHA mismatch")
    compatibility = json.loads(files["b0_compatibility"][0].read_text(encoding="utf-8"))
    if compatibility.get("reuse_b0") is not True or compatibility.get("requires_b0_rerun") is not False:
        raise Step250RuntimeError("step250 B0 reuse is not compatible")
    expected_b0 = compatibility.get("expected")
    config_b0_identity = {
        "base_model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "protocol_sha256": config["choice_score"]["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "choice_backend": config["choice_score"]["backend"],
        "batch_size": config["choice_score"]["batch_size"],
        "float32_log_softmax": config["choice_score"]["float32_log_softmax"],
        "final_authorized": config["execution"]["final_authorized"],
    }
    if not isinstance(expected_b0, dict) or any(
        expected_b0.get(key) != value for key, value in config_b0_identity.items()
    ):
        raise Step250RuntimeError("step250 B0 protocol identity differs from compatibility attestation")
    b0_manifest_path = files["b0_artifact_manifest"][0]
    b0_manifest = json.loads(b0_manifest_path.read_text(encoding="utf-8"))
    inventory = {
        str(item.get("path") or ""): str(item.get("sha256") or "")
        for item in b0_manifest.get("files", [])
        if isinstance(item, dict)
    }
    b0_choice_path = _resolve(config["b0_reuse"]["choice_predictions_path"]).resolve()
    if (
        b0_choice_path.parent != b0_manifest_path.parent.resolve()
        or inventory.get(b0_choice_path.name) != compatibility.get("b0_choice_predictions_sha256")
        or not b0_choice_path.is_file()
        or _sha256(b0_choice_path) != compatibility.get("b0_choice_predictions_sha256")
    ):
        raise Step250RuntimeError("step250 B0 choice prediction identity mismatch")
    b0_smoke_path = _resolve(config["b0_reuse"]["micro_smoke_path"]).resolve()
    if (
        b0_smoke_path.parent != b0_manifest_path.parent.resolve()
        or not b0_smoke_path.is_file()
        or inventory.get(b0_smoke_path.name) != _sha256(b0_smoke_path)
    ):
        raise Step250RuntimeError("step250 B0 micro-smoke identity mismatch")
    run_card = json.loads(files["run_card"][0].read_text(encoding="utf-8"))
    run_card_expected = {
        "run_id": config["execution"]["run_id"],
        "model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "checkpoint_step": 250,
        "checkpoint_role": "candidate_medical_teacher",
        "checkpoint_adapter_sha256": config["checkpoint"]["adapter_sha256"],
        "checkpoint_manifest_sha256": config["checkpoint"]["manifest_sha256"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "controller_protocol_sha256": config["choice_score"]["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "choice_backend": DIRECT_BACKEND,
        "final_authorized": False,
    }
    if any(run_card.get(key) != value for key, value in run_card_expected.items()):
        raise Step250RuntimeError("step250 run card/config identity mismatch")
    output = Path(config["execution"]["output_root"]) / config["execution"]["run_id"]
    if output.exists():
        raise Step250RuntimeError("step250 output path must be new")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    if require_clean and status:
        raise Step250RuntimeError("formal step250 preflight requires clean committed worktree")
    return {
        "status": "PASS",
        "state": "ready_waiting_for_gpu_step250_eval",
        "checkpoint_step": 250,
        "reuse_b0": True,
        "choice_backend": DIRECT_BACKEND,
        "cpu_dry_run": True,
        "model_weights_loaded": False,
        "gpu_used": False,
        "final_authorized": False,
        "worktree_clean": not bool(status),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_step250_artifacts(
    output: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish aggregate identity without copying labels or restricted text."""

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if _HEX40.fullmatch(git_sha) is None:
        raise Step250RuntimeError("step250 artifact requires committed Git SHA")
    (output / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    _atomic_json(output / "metadata.json", {
        "run_id": config["execution"]["run_id"],
        "stage": "controller_eval_checkpoint_candidate",
        "status": summary["status"],
        "checkpoint_step": 250,
        "checkpoint_adapter_sha256": config["checkpoint"]["adapter_sha256"],
        "base_model_revision": config["model"]["revision"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "protocol_sha256": config["choice_score"]["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "git_sha": git_sha,
        "seed": 42,
        "final_authorized": False,
        "actual_cost_cny": None,
    })
    _atomic_json(output / "data_manifest.json", {
        "manifest_path": config["data"]["manifest_path"],
        "manifest_sha256": config["data"]["manifest_sha256"],
        "roles": config["data"]["roles"],
        "medical_count": 300,
        "general_count": 209,
        "labels_physically_separate": True,
        "labels_opened_after_model_release": True,
        "labels_copied_into_run": False,
        "final_authorized": False,
    })
    _atomic_json(output / "checkpoint_manifest.json", {
        "manifest_path": config["checkpoint"]["manifest_path"],
        "manifest_sha256": config["checkpoint"]["manifest_sha256"],
        "adapter_sha256": config["checkpoint"]["adapter_sha256"],
        "step": 250,
        "source_run_id": config["checkpoint"]["source_run_id"],
        "base_model_revision": config["model"]["revision"],
        "merged": False,
    })
    _atomic_json(output / "protocol_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["choice_score"]["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "choice_backend": DIRECT_BACKEND,
        "diagnostic_parser_used": False,
        "git_sha": git_sha,
        "final_authorized": False,
    })
    _atomic_json(output / "paired_stats.json", dict(summary["paired_stats"]))
    _atomic_json(output / "aggregate.json", dict(summary["choice_metrics"]))
    metrics_tmp = output / "metrics.jsonl.tmp"
    with metrics_tmp.open("w", encoding="utf-8") as handle:
        for route, metrics in summary["choice_metrics"].items():
            handle.write(json.dumps({
                "metric_track": "choice_score",
                "route": route,
                "metrics": metrics,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(metrics_tmp, output / "metrics.jsonl")
    _atomic_json(output / "cost.json", {
        "price_cny_per_hour": None,
        "process_runtime_seconds": summary["process_runtime_seconds"],
        "process_cost_cny": None,
        "platform_billed_cost_cny": None,
        "actual_cost_cny": None,
        "cost_status": "pending_live_price_reconciliation",
    })
    (output / "stdout.log").write_text(
        "Checkpoint-250 direct-logit aggregate; raw predictions remain ignored.\n",
        encoding="utf-8",
    )
    files = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "run_id": config["execution"]["run_id"],
        "stage": "controller_eval_checkpoint_candidate",
        "checkpoint_step": 250,
        "checkpoint_adapter_sha256": config["checkpoint"]["adapter_sha256"],
        "b0_reused": True,
        "protocol_sha256": config["choice_score"]["protocol_sha256"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "git_sha": git_sha,
        "final_authorized": False,
        "files": files,
    }
    _atomic_json(output / "artifact_manifest.json", manifest)
    return {"artifact_manifest_sha256": _sha256(output / "artifact_manifest.json"), "git_sha": git_sha}


def run_step250_gpu(config_path: str | Path) -> dict[str, Any]:  # pragma: no cover - GPU only
    """Run step250 three-repeat smoke and full choice, reusing exact B0."""

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

    config = load_step250_eval_config(config_path)
    assert_step250_gpu_authorized(config)
    step250_cpu_preflight(config_path, require_clean=True)
    output = Path(config["execution"]["output_root"]) / config["execution"]["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    model = tokenizer = None
    try:
        base_controller = yaml.safe_load(
            _resolve(config["choice_score"]["controller_v2_config_path"]).read_text(encoding="utf-8")
        )
        base_controller["model"]["medical_lora_path"] = config["checkpoint"]["path"]
        manifest_path = _resolve(config["data"]["manifest_path"])
        roles = config["data"]["roles"]
        smoke_ids = _load_choice_smoke_ids(_resolve(config["choice_score"]["smoke_manifest_path"]))
        historical = json.loads(
            _resolve(config["b0_reuse"]["micro_smoke_path"]).read_text(encoding="utf-8")
        )["runs"]
        model, tokenizer, encode, _ = load_direct_logit_route(
            base_controller, "B1", device=config["choice_score"]["device"]
        )
        smoke_rows = list(iter_prompt_rows(manifest_path, roles, selected_ids=smoke_ids))
        candidate_runs = [
            list(run_direct_choice_rows(smoke_rows, model=model, tokenize=encode))
            for _ in range(3)
        ]
        smoke = validate_direct_logit_repetitions({"B0": historical, "B1": candidate_runs})
        _atomic_json(output / "direct_logit_micro_smoke.json", smoke)
        choice_artifact = write_prediction_artifact(
            output / "step250_choice_predictions.jsonl",
            run_direct_choice_rows(
                iter_prompt_rows(manifest_path, roles), model=model, tokenize=encode
            ),
        )
    finally:
        model = tokenizer = None
        release_model_execution(device=config["choice_score"]["device"])

    labels = _label_map(_resolve(config["data"]["manifest_path"]), config["data"]["roles"])
    from src.eval.controller_v2_runtime import _iter_jsonl

    b0 = _compact_score(
        _iter_jsonl(_resolve(config["b0_reuse"]["choice_predictions_path"])), labels
    )
    step250 = _compact_score(_iter_jsonl(Path(choice_artifact["path"])), labels)
    metrics = {"B0": _track_metrics(b0), "step250": _track_metrics(step250)}
    paired = paired_comparison(b0, step250, seed=42)
    medical_correct = sum(
        bool(row["correct"]) for row in step250 if row.get("domain") == "medical"
    )
    gate = step250_teacher_gate(
        b0_medical_correct=219,
        step250_medical_correct=medical_correct,
        medical_total=300,
        step500_state="ambiguous",
    )
    summary = {
        "run_id": config["execution"]["run_id"],
        "status": "completed",
        "checkpoint_step": 250,
        "choice_metrics": metrics,
        "paired_stats": paired,
        "teacher_gate": gate,
        "micro_smoke": smoke,
        "prediction_artifact": choice_artifact,
        "diagnostic_parser_used": False,
        "labels_opened_after_model_release": True,
        "final_authorized": False,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "process_runtime_seconds": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
        "actual_cost_cny": None,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(output / "teacher_gate.json", gate)
    identity = _write_step250_artifacts(output, config=config, summary=summary)
    return {**summary, **identity}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "gpu-preflight", "run"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(step250_cpu_preflight(args.config), sort_keys=True))
        return 0
    if args.command == "gpu-preflight":
        config = load_step250_eval_config(args.config)
        assert_step250_gpu_authorized(config)
        print(json.dumps(step250_cpu_preflight(args.config, require_clean=True), sort_keys=True))
        return 0
    print(json.dumps(run_step250_gpu(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
