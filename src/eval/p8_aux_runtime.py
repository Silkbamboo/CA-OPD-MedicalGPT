"""GPU runtimes for P8's read-only, post-gate auxiliary diagnostics."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import yaml

from src.eval.p8_secondary_evaluator import (
    match_cmb_labels_by_stable_sample_id,
    legalize_parsed_choice_v1,
    parse_choice_letter_v1,
    select_cmb_isolation_subset,
    select_label_free_subset,
    summarize_correct_answer_margins,
    summarize_secondary_generations,
)


class P8AuxRuntimeError(RuntimeError):
    """An auxiliary diagnostic identity or isolation contract differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
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
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise P8AuxRuntimeError(f"non-object JSONL row at {path}:{number}")
            yield value


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise P8AuxRuntimeError("P8 auxiliary config schema differs")
    if not (
        value.get("frozen_before_auxiliary_gpu_results") is True
        and value.get("cannot_change_candidate_or_reopen_training") is True
        and value.get("access", {}).get("confirmation_allowed") is False
        and value.get("access", {}).get("final_allowed") is False
    ):
        raise P8AuxRuntimeError("P8 auxiliary isolation boundary differs")
    base = value.get("base_model")
    if not isinstance(base, Mapping):
        raise P8AuxRuntimeError("P8 auxiliary Base identity is absent")
    for field in ("manifest_path",):
        file = Path(str(base[field])).resolve()
        if not file.is_file() or _sha256(file) != base[f"{field.removesuffix('_path')}_sha256"]:
            raise P8AuxRuntimeError("P8 auxiliary Base manifest differs")
    return value


def _authorization() -> None:
    if os.environ.get("CA_OPD_ALLOW_P8_AUX_GPU") != "1":
        raise P8AuxRuntimeError("P8 auxiliary GPU authorization is absent")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise P8AuxRuntimeError("P8 auxiliary deterministic CuBLAS binding is absent")


def _prepare_cuda_peak_stats(torch_module: Any, *, device_index: int) -> None:
    """Initialize the CUDA context before using its optional peak telemetry."""

    torch_module.cuda.init()
    torch_module.cuda.set_device(device_index)
    torch_module.cuda.reset_peak_memory_stats(device_index)


def _validate_cuda_release_residual(
    torch_module: Any, *, device_index: int, maximum_bytes: int = 64 * 1024 * 1024
) -> int:
    """Reject a resident model while allowing a small CUDA runtime context."""

    allocated = int(torch_module.cuda.memory_allocated(device_index))
    if allocated > maximum_bytes:
        raise P8AuxRuntimeError(
            f"model-release CUDA residual {allocated} exceeds {maximum_bytes} bytes"
        )
    return allocated


def _route_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    section = config["secondary_generation"]
    path = Path(str(section["route_spec_path"]))
    if not path.is_file() or _sha256(path) != section["route_spec_sha256"]:
        raise P8AuxRuntimeError("P8 auxiliary route spec SHA differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = payload.get("routes") if isinstance(payload, Mapping) else None
    if not isinstance(routes, list):
        raise P8AuxRuntimeError("P8 auxiliary route spec is invalid")
    selected = {str(row.get("name")): dict(row) for row in routes if row.get("name") in section["routes"]}
    if set(selected) != set(section["routes"]):
        raise P8AuxRuntimeError("P8 auxiliary route set differs")
    for name, route in selected.items():
        adapter = route.get("adapter_path")
        if name == "B0":
            if adapter is not None:
                raise P8AuxRuntimeError("B0 auxiliary route unexpectedly has an adapter")
            continue
        root = Path(str(adapter))
        weight = root / "adapter_model.safetensors"
        manifest = Path(str(route["adapter_manifest_path"]))
        if not (
            weight.is_file()
            and manifest.is_file()
            and _sha256(weight) == route["adapter_weight_sha256"]
            and _sha256(manifest) == route["adapter_manifest_sha256"]
        ):
            raise P8AuxRuntimeError(f"{name} auxiliary adapter identity differs")
    return selected


def run_cmb_margin_isolation(*, config_path: Path, output: Path) -> dict[str, Any]:
    """Score B0/B1 label-free, release both models, then open CMB-train labels."""

    _authorization()
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise P8AuxRuntimeError("CMB isolation output must be fresh")
    config = _config(config_path)
    section = config["cmb_margin_isolation"]
    if not (
        section.get("label_join")
        == "stable_sample_id_v2_exact_adapter_identity"
        and section.get("fuzzy_label_matching") is False
        and float(section.get("required_unique_match_fraction", 0.0)) == 1.0
        and section.get("margin_eligibility")
        == "single_letter_answer_idx_available"
        and section.get("unresolved_answer_handling")
        == "structural_exclusion_counted_not_scored"
        and section.get("all_selected_prediction_rows_required") is True
    ):
        raise P8AuxRuntimeError("CMB isolation label-join/margin contract differs")
    prompt_path = Path(str(section["prompt_path"]))
    raw_label_path = Path(str(section["raw_label_path"]))
    if not (
        prompt_path.is_file()
        and _sha256(prompt_path) == section["prompt_sha256"]
        and raw_label_path.is_file()
        and _sha256(raw_label_path) == section["raw_label_sha256"]
    ):
        raise P8AuxRuntimeError("CMB isolation data identity differs")
    routes = _route_map(config)
    selected = select_cmb_isolation_subset(
        list(_iter_jsonl(prompt_path)), count=int(section["count"])
    )
    selection_sha256 = _canonical_sha256(
        [
            {"sample_id": row["sample_id"], "content_hash": row["content_hash"]}
            for row in selected
        ]
    )
    evaluation_rows = [
        {**row, "target_role": "medical_controller_dev"} for row in selected
    ]

    from src.eval.controller_v2_runtime import release_model_execution
    from src.eval.direct_logit_scorer import load_direct_logit_route, run_direct_choice_rows
    import torch

    started = time.time()
    predictions: dict[str, list[dict[str, Any]]] = {}
    peak_allocated: dict[str, int] = {}
    for route_name in ("B0", "B1"):
        model = tokenizer = None
        try:
            _prepare_cuda_peak_stats(torch, device_index=0)
            route_config = {
                "model": {
                    "path": config["base_model"]["path"],
                    "revision": config["base_model"]["revision"],
                    "tokenizer_revision": config["base_model"]["revision"],
                    "medical_lora_path": routes["B1"]["adapter_path"],
                }
            }
            model, tokenizer, encode, _plan = load_direct_logit_route(
                route_config, route_name, device=str(section["device"])
            )
            predictions[route_name] = [
                {
                    "sample_id": row["sample_id"],
                    "candidate_scores": row["candidate_scores"],
                }
                for row in run_direct_choice_rows(
                    evaluation_rows,
                    model=model,
                    tokenize=encode,
                    require_expected_qwen_ids=True,
                )
            ]
            peak_allocated[route_name] = int(torch.cuda.max_memory_allocated(0))
        finally:
            model = None
            tokenizer = None
            gc.collect()
            release_model_execution(device=str(section["device"]))
    release_residual_bytes = _validate_cuda_release_residual(torch, device_index=0)

    raw_rows = json.loads(raw_label_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise P8AuxRuntimeError("CMB raw label artifact is not a list")
    labels, label_join = match_cmb_labels_by_stable_sample_id(selected, raw_rows)
    aggregate = summarize_correct_answer_margins(
        predictions,
        labels,
        bootstrap_samples=int(section["bootstrap_samples"]),
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "p8_cmb_train_isolation_margin_diagnostic_v1",
        "status": "completed",
        "role": "secondary_mechanistic_diagnostic_not_candidate_selection",
        "selection_sha256": selection_sha256,
        "selection_rule": section["selection"],
        "aggregate": aggregate,
        "label_join": label_join,
        "route_identity": {name: routes[name] for name in ("B0", "B1")},
        "models_released_before_label_access": True,
        "cuda_allocated_bytes_before_label_access": release_residual_bytes,
        "cuda_release_residual_maximum_bytes": 64 * 1024 * 1024,
        "training_diagnostic_process_received_labels": False,
        "label_isolation_access_count": 1,
        "controller_access_count": 0,
        "confirmation_access_count": 0,
        "final_access_count": 0,
        "peak_allocated_bytes": peak_allocated,
        "elapsed_seconds": time.time() - started,
        "reference_price_cny_per_hour": 2.96,
        "derived_cost_cny": (time.time() - started) / 3600.0 * 2.96,
        "platform_actual_cost_cny": None,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    _atomic_json(output, result)
    return result


def _release_vllm(engine: Any) -> None:
    from src.eval.controller_v2_runtime import _release_vllm_engine
    import torch

    _release_vllm_engine(engine)
    gc.collect()
    torch.cuda.empty_cache()


def run_secondary_generation(*, config_path: Path, output: Path) -> dict[str, Any]:
    """Run five frozen routes, release vLLM, then join Controller labels once."""

    _authorization()
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise P8AuxRuntimeError("secondary generation output must be fresh")
    config = _config(config_path)
    section = config["secondary_generation"]
    manifest_path = Path(str(section["controller_manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    if not manifest_path.is_file() or _sha256(manifest_path) != section["controller_manifest_sha256"]:
        raise P8AuxRuntimeError("secondary Controller manifest identity differs")
    routes = _route_map(config)
    from src.eval.controller_v2 import build_generative_prompt
    from src.eval.controller_v2_runtime import (
        _apply_vllm_v1_generation_policy,
        _label_map,
        iter_prompt_rows,
    )

    prompt_rows = list(
        iter_prompt_rows(
            manifest_path,
            ("medical_controller_dev", "general_controller_dev"),
        )
    )
    selected = select_label_free_subset(
        prompt_rows,
        medical_count=int(section["medical_count"]),
        general_count=int(section["general_count"]),
    )
    selection_sha256 = _canonical_sha256(
        [
            {"sample_id": row["sample_id"], "content_hash": row["content_hash"]}
            for row in selected
        ]
    )

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.lora.request import LoRARequest

    base = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(base["path"]),
        revision=str(base["revision"]),
        local_files_only=True,
    )
    original_defaults = EngineArgs._set_default_args

    def frozen_defaults(engine_args, usage_context, model_config):
        _apply_vllm_v1_generation_policy(
            engine_args, usage_context, model_config, original=original_defaults
        )

    EngineArgs._set_default_args = frozen_defaults
    engine = None
    started = time.time()
    predictions: dict[str, list[dict[str, Any]]] = {}
    try:
        engine = LLM(
            model=str(base["path"]),
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=16,
            max_loras=1,
            max_model_len=1536,
            seed=42,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=False,
            enforce_eager=True,
            enable_chunked_prefill=False,
        )
    finally:
        EngineArgs._set_default_args = original_defaults
    try:
        prompts = [build_generative_prompt(row) for row in selected]
        sampling = SamplingParams(temperature=0.0, max_tokens=512, seed=42)
        for adapter_id, route_name in enumerate(section["routes"], 1):
            route = routes[route_name]
            request = (
                None
                if route_name == "B0"
                else LoRARequest(route_name, adapter_id, str(route["adapter_path"]))
            )
            outputs = engine.generate(
                prompts=prompts,
                sampling_params=sampling,
                lora_request=request,
                use_tqdm=False,
            )
            if len(outputs) != len(selected):
                raise P8AuxRuntimeError("secondary generation output count differs")
            compact: list[dict[str, Any]] = []
            for row, item in zip(selected, outputs, strict=True):
                candidate = item.outputs[0]
                text = str(candidate.text)
                parsed = legalize_parsed_choice_v1(
                    parse_choice_letter_v1(text), option_count=len(row["options"])
                )
                compact.append(
                    {
                        "sample_id": row["sample_id"],
                        "domain": row["domain"],
                        "parsed": parsed,
                        "finish_reason": str(candidate.finish_reason),
                        "generated_token_count": len(candidate.token_ids),
                        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
            predictions[route_name] = compact
    finally:
        if engine is not None:
            _release_vllm(engine)
        engine = None
        tokenizer = None
        gc.collect()

    labels_all = _label_map(
        manifest_path, ("medical_controller_dev", "general_controller_dev")
    )
    selected_ids = {str(row["sample_id"]) for row in selected}
    labels = {sample_id: labels_all[sample_id] for sample_id in selected_ids}
    aggregate = summarize_secondary_generations(predictions, labels)
    result = {
        "schema_version": 1,
        "artifact_kind": "p8_secondary_generative_evaluator_v1",
        "status": "completed",
        "role": "secondary_diagnostic_direct_logit_remains_primary",
        "selection_sha256": selection_sha256,
        "selection_rule": section["selection"],
        "medical_count": int(section["medical_count"]),
        "general_count": int(section["general_count"]),
        "decoding": section["decoding"],
        "parser": section["parser"],
        "aggregate": aggregate,
        "route_identity": routes,
        "engine_released_before_label_access": True,
        "raw_prompt_response_label_persisted": False,
        "controller_access_count": 1,
        "confirmation_access_count": 0,
        "final_access_count": 0,
        "elapsed_seconds": time.time() - started,
        "reference_price_cny_per_hour": 2.96,
        "derived_cost_cny": (time.time() - started) / 3600.0 * 2.96,
        "platform_actual_cost_cny": None,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
    }
    _atomic_json(output, result)
    return result


__all__ = ["P8AuxRuntimeError", "run_cmb_margin_isolation", "run_secondary_generation"]
