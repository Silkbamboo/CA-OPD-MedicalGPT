"""Identity-bound direct-logit Controller runner for P6."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from src.eval.controller_v2_runtime import (
    _compact_score,
    _label_map,
    _track_metrics,
    iter_prompt_rows,
    release_model_execution,
)
from src.eval.direct_logit_scorer import (
    apply_deterministic_runtime,
    run_direct_choice_rows,
)
from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p6_identity_cache import (
    build_prediction_cache_identity,
    read_prediction_cache,
    write_prediction_cache,
)
from src.eval.paired_stats import paired_comparison


class P6ControllerError(RuntimeError):
    """P6 Controller config, identity, prediction, or selection differs."""


REPO = Path(__file__).resolve().parents[2]
NO_ADAPTER_SHA256 = hashlib.sha256(b"p6:no-adapter:base-route").hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _files_sha(files: Iterable[Path]) -> str:
    projection = [
        {"path": str(Path(path).resolve()), "sha256": _sha_file(Path(path).resolve())}
        for path in files
    ]
    if not projection:
        raise P6ControllerError("Controller prompt/label file set is empty")
    return _canonical_sha(projection)


def build_evaluator_identity(
    config: Mapping[str, Any],
    *,
    adapter_identity: Mapping[str, Any],
    prompt_files: Sequence[Path],
    label_files: Sequence[Path],
) -> dict[str, Any]:
    """Build the complete cache key without conflating prompts and labels."""

    value = {
        "schema_version": 1,
        "base_model_revision": config["base_model_revision"],
        "adapter_ordered_sha256": adapter_identity["adapter_ordered_sha256"],
        "adapter_weight_sha256": adapter_identity["adapter_weight_sha256"],
        "adapter_manifest_sha256": adapter_identity["adapter_manifest_sha256"],
        "tokenizer_revision": config["tokenizer_revision"],
        "template_sha256": config["template_sha256"],
        "scorer_backend": config["scorer_backend"],
        "scorer_version_sha256": config["scorer_version_sha256"],
        "evaluator_config_sha256": config["evaluator_config_sha256"],
        "prompt_manifest_sha256": _files_sha(prompt_files),
        "label_manifest_sha256": _files_sha(label_files),
        "decoding_config_sha256": config["decoding_config_sha256"],
        "code_git_sha": config["code_git_sha"],
    }
    return build_prediction_cache_identity(value)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6ControllerError("P6 Controller config is invalid") from error
    if not (
        isinstance(config, Mapping)
        and config.get("schema_version") == 1
        and config.get("artifact_kind") == "p6_identity_bound_controller_config"
        and config.get("isolation", {}).get("final_authorized") is False
        and config.get("isolation", {}).get("final_access_count") == 0
    ):
        raise P6ControllerError("P6 Controller config schema/isolation differs")
    base = config["base"]
    manifest_path = Path(str(config["data"]["manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    if not (
        manifest_path.is_file()
        and _sha_file(manifest_path) == config["data"]["manifest_sha256"]
        and Path(str(base["artifact_manifest_path"])).is_file()
        and _sha_file(Path(str(base["artifact_manifest_path"])))
        == base["artifact_manifest_sha256"]
    ):
        raise P6ControllerError("P6 Controller base/data SHA differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_files: list[Path] = []
    label_files: list[Path] = []
    for role in config["data"]["roles"]:
        files = manifest["roles"][role]["files"]
        prompts = [Path(item["path"]) for item in files if item["supervision_fields"] == 0]
        labels = [Path(item["path"]) for item in files if item["supervision_fields"] > 0]
        if len(prompts) != 1 or len(labels) != 1:
            raise P6ControllerError("P6 Controller prompt/label separation differs")
        for item in files:
            file = Path(item["path"])
            if not file.is_file() or _sha_file(file) != item["sha256"]:
                raise P6ControllerError("P6 Controller role file SHA differs")
        prompt_files.extend(prompts)
        label_files.extend(labels)
    scorer_projection = {
        "direct_logit_scorer_sha256": _sha_file(REPO / "src/eval/direct_logit_scorer.py"),
        "controller_runtime_sha256": _sha_file(REPO / "src/eval/controller_v2_runtime.py"),
        "p6_runtime_sha256": _sha_file(Path(__file__)),
        "backend": config["choice_score"]["backend"],
    }
    evaluator = {
        "schema_version": 1,
        "base_model_revision": base["revision"],
        "tokenizer_revision": base["tokenizer_revision"],
        "template_sha256": config["choice_score"]["template_sha256"],
        "scorer_backend": config["choice_score"]["backend"],
        "scorer_version_sha256": _canonical_sha(scorer_projection),
        "evaluator_config_sha256": _sha_file(path),
        "decoding_config_sha256": _canonical_sha(config["choice_score"]),
        "code_git_sha": _git_head(),
        "prompt_files": prompt_files,
        "label_files": label_files,
        "manifest_path": manifest_path,
        "roles": tuple(config["data"]["roles"]),
    }
    return dict(config), evaluator


def adapter_identity_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    path_value = spec.get("adapter_path")
    if path_value is None:
        return {
            "adapter_path": None,
            "adapter_ordered_sha256": NO_ADAPTER_SHA256,
            "adapter_weight_sha256": NO_ADAPTER_SHA256,
            "adapter_manifest_sha256": NO_ADAPTER_SHA256,
        }
    path = Path(str(path_value)).resolve()
    manifest = Path(str(spec.get("adapter_manifest_path", ""))).resolve()
    if not (
        path.is_dir()
        and (path / "adapter_model.safetensors").is_file()
        and manifest.is_file()
    ):
        raise P6ControllerError("Controller adapter files are absent")
    actual = {
        "adapter_path": str(path),
        "adapter_ordered_sha256": _ordered_adapter_sha256(path),
        "adapter_weight_sha256": _sha_file(path / "adapter_model.safetensors"),
        "adapter_manifest_sha256": _sha_file(manifest),
    }
    for field in (
        "adapter_ordered_sha256",
        "adapter_weight_sha256",
        "adapter_manifest_sha256",
    ):
        if spec.get(field) != actual[field]:
            raise P6ControllerError(f"Controller adapter identity differs: {field}")
    return actual


def _load_route(config: Mapping[str, Any], adapter_path: str | None):  # pragma: no cover - GPU
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    apply_deterministic_runtime(torch_module=torch)
    base = config["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(base["path"]),
        revision=str(base["tokenizer_revision"]),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base["path"]),
        revision=str(base["revision"]),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return model, tokenizer, encode


def _repeatability(
    repeats: Sequence[Sequence[Mapping[str, Any]]], *, tolerance: float
) -> dict[str, Any]:
    if len(repeats) != 3 or any(len(rows) != 4 for rows in repeats):
        raise P6ControllerError("Controller smoke must be 4 rows x 3 repeats")
    reference = {str(row["sample_id"]): row for row in repeats[0]}
    maximum = 0.0
    for rows in repeats[1:]:
        current = {str(row["sample_id"]): row for row in rows}
        if set(current) != set(reference):
            raise P6ControllerError("Controller smoke sample identity differs")
        for sample_id, left in reference.items():
            right = current[sample_id]
            if left["predicted_label"] != right["predicted_label"]:
                raise P6ControllerError("Controller smoke prediction differs")
            if set(left["candidate_scores"]) != set(right["candidate_scores"]):
                raise P6ControllerError("Controller smoke candidate set differs")
            for label, score in left["candidate_scores"].items():
                maximum = max(
                    maximum,
                    abs(float(score) - float(right["candidate_scores"][label])),
                )
    if maximum > tolerance:
        raise P6ControllerError("Controller smoke score drift exceeds tolerance")
    return {
        "passed": True,
        "sample_count": 4,
        "repeat_count": 3,
        "max_abs_candidate_score_delta": maximum,
        "tolerance": tolerance,
    }


def score_loaded_model(
    *,
    model: Any,
    tokenizer: Any,
    encode: Any,
    config: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    cache_path: Path,
) -> dict[str, Any]:  # pragma: no cover - GPU
    identity = build_evaluator_identity(
        evaluator,
        adapter_identity=adapter_identity,
        prompt_files=evaluator["prompt_files"],
        label_files=evaluator["label_files"],
    )
    all_rows = list(
        iter_prompt_rows(
            evaluator["manifest_path"], list(evaluator["roles"])
        )
    )
    smoke_rows = all_rows[:2] + all_rows[300:302]
    repeats = [
        list(run_direct_choice_rows(smoke_rows, model=model, tokenize=encode))
        for _ in range(3)
    ]
    smoke = _repeatability(
        repeats,
        tolerance=float(config["choice_score"]["score_repeat_tolerance"]),
    )
    predictions = list(
        run_direct_choice_rows(all_rows, model=model, tokenize=encode)
    )
    manifest = write_prediction_cache(
        cache_path, identity=identity, rows=predictions
    )
    return {
        "identity": identity,
        "rows": predictions,
        "prediction_sha256": manifest["prediction_sha256"],
        "cache_manifest_sha256": _sha_file(cache_path / "cache_manifest.json"),
        "cache_hit": False,
        "repeatability": smoke,
    }


def score_loaded_controller_metrics(
    *,
    model: Any,
    tokenizer: Any,
    config_path: Path,
    adapter_identity: Mapping[str, Any],
    cache_path: Path,
) -> dict[str, Any]:  # pragma: no cover - GPU
    """Score a live training adapter for a CA routing window."""

    config, evaluator = _load_config(config_path)

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    result = score_loaded_model(
        model=model,
        tokenizer=tokenizer,
        encode=encode,
        config=config,
        evaluator=evaluator,
        adapter_identity=adapter_identity,
        cache_path=cache_path,
    )
    labels = _label_map(evaluator["manifest_path"], list(evaluator["roles"]))
    scored = _compact_score(result["rows"], labels)
    return {
        **result,
        "metrics": _track_metrics(scored),
        "labels_opened_after_predictions_frozen": True,
        "controller_access_count": 1,
        "final_access_count": 0,
    }


def select_method_checkpoint(
    metrics: Mapping[str, Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    *,
    b0_general_micro: float,
    delta: float,
) -> dict[str, Any]:
    candidates = [route for route in routes if route.get("step") is not None]
    if not candidates:
        return {"selected_checkpoint": None, "status": "not_applicable"}
    feasible = [
        route
        for route in candidates
        if float(metrics[str(route["name"])]["general_micro_accuracy"])
        >= float(b0_general_micro) - float(delta)
    ]
    if not feasible:
        return {
            "selected_checkpoint": None,
            "status": "constraint_not_met",
            "feasible_steps": [],
        }
    selected = min(
        feasible,
        key=lambda route: (
            -float(metrics[str(route["name"])]["medical_accuracy"]),
            int(route["step"]),
        ),
    )
    return {
        "selected_checkpoint": selected["name"],
        "selected_step": int(selected["step"]),
        "status": "selected",
        "feasible_steps": sorted(int(route["step"]) for route in feasible),
        "general_constraint_threshold": float(b0_general_micro) - float(delta),
        "tie_break": "earlier_checkpoint",
    }


def run_p6_controller(
    *,
    config_path: Path,
    routes_path: Path,
    output: Path,
    cache_root: Path,
    allow_cache: bool,
) -> dict[str, Any]:  # pragma: no cover - GPU
    if os.environ.get("CA_OPD_ALLOW_P6_CONTROLLER_GPU") != "1":
        raise P6ControllerError("P6 Controller GPU authorization env is absent")
    output = Path(output).resolve()
    cache_root = Path(cache_root).resolve()
    if output.exists() or output.is_symlink():
        raise P6ControllerError("P6 Controller output must be fresh")
    config, evaluator = _load_config(config_path)
    routes_payload = json.loads(Path(routes_path).read_text(encoding="utf-8"))
    routes = routes_payload.get("routes") if isinstance(routes_payload, Mapping) else None
    if not isinstance(routes, list) or not routes:
        raise P6ControllerError("P6 Controller routes are absent")
    output.mkdir(parents=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict[str, dict[str, Any]] = {}
    runtime_ledgers: dict[str, dict[str, Any]] = {}
    for route in routes:
        name = str(route["name"])
        adapter = adapter_identity_from_spec(route)
        identity = build_evaluator_identity(
            evaluator,
            adapter_identity=adapter,
            prompt_files=evaluator["prompt_files"],
            label_files=evaluator["label_files"],
        )
        cache = cache_root / f"{name}--{identity['cache_key_sha256']}"
        if allow_cache and cache.is_dir():
            loaded = read_prediction_cache(cache, expected_identity=identity)
            results[name] = {
                **loaded,
                "repeatability": {"reused_from_identity_bound_cache": True},
            }
            runtime_ledgers[name] = {
                "model_loaded": False,
                "cache_hit": True,
                "cache_key_sha256": identity["cache_key_sha256"],
            }
            continue
        model = tokenizer = None
        try:
            runtime_ledgers[name] = {
                "model_loaded": True,
                "cache_hit": False,
                "base_revision": evaluator["base_model_revision"],
                **adapter,
            }
            model, tokenizer, encode = _load_route(config, adapter["adapter_path"])
            results[name] = score_loaded_model(
                model=model,
                tokenizer=tokenizer,
                encode=encode,
                config=config,
                evaluator=evaluator,
                adapter_identity=adapter,
                cache_path=cache,
            )
        finally:
            model = None
            tokenizer = None
            gc.collect()
            release_model_execution(device="cuda:0")

    labels = _label_map(evaluator["manifest_path"], list(evaluator["roles"]))
    scored = {
        name: _compact_score(value["rows"], labels)
        for name, value in results.items()
    }
    metrics = {name: _track_metrics(rows) for name, rows in scored.items()}
    reference = str(routes_payload.get("reference_route", "B0"))
    paired = {
        name: paired_comparison(
            scored[reference],
            rows,
            seed=int(config["statistics"]["bootstrap_seed"]),
            bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
        )
        for name, rows in scored.items()
        if name != reference
    }
    selections: dict[str, Any] = {}
    for method in sorted(
        {str(route["method_id"]) for route in routes if route.get("step") is not None}
    ):
        selected_routes = [route for route in routes if route.get("method_id") == method]
        selections[method] = select_method_checkpoint(
            metrics,
            selected_routes,
            b0_general_micro=float(metrics[reference]["general_micro_accuracy"]),
            delta=float(config["selection"]["general_constraint_delta"]),
        )
    result = {
        "schema_version": 1,
        "artifact_kind": "p6_identity_bound_controller_report",
        "status": "complete",
        "allow_cache": bool(allow_cache),
        "all_routes_cache_hit": all(value["cache_hit"] for value in results.values()),
        "route_identities": {name: value["identity"] for name, value in results.items()},
        "prediction_artifacts": {
            name: {
                "prediction_sha256": value["prediction_sha256"],
                "cache_manifest_sha256": value.get("cache_manifest_sha256", value.get("manifest_sha256")),
            }
            for name, value in results.items()
        },
        "repeatability": {name: value["repeatability"] for name, value in results.items()},
        "runtime_load_ledgers": runtime_ledgers,
        "metrics": metrics,
        "paired_vs_reference": paired,
        "selection": selections,
        "labels_opened_after_all_predictions_frozen": True,
        "legacy_cache_used": False,
        "controller_access_count": 1,
        "final_access_count": 0,
        "final_executed": False,
        "elapsed_seconds": time.time() - started,
        "platform_actual_cost_cny": None,
    }
    _atomic_json(output / "controller_report.json", result)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--allow-cache", action="store_true")
    args = parser.parse_args(argv)
    result = run_p6_controller(
        config_path=args.config,
        routes_path=args.routes,
        output=args.output,
        cache_root=args.cache_root,
        allow_cache=args.allow_cache,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "NO_ADAPTER_SHA256",
    "P6ControllerError",
    "adapter_identity_from_spec",
    "build_evaluator_identity",
    "run_p6_controller",
    "score_loaded_model",
    "score_loaded_controller_metrics",
    "select_method_checkpoint",
]
