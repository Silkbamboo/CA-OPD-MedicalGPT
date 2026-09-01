"""Lazy Controller v2 runtime boundary.

CPU imports validate configuration and artifacts without importing torch,
transformers or vLLM. GPU objects are constructed only inside explicitly
authorized execution functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import gc
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from src.eval.controller_v2 import (
    BASE_MODEL_REVISION,
    CONTROLLER_ROLES,
    MEDICAL_LORA_SHA256,
    PROTOCOL_VERSION,
    build_choice_request,
    build_generative_prompt,
    parse_generation_v2,
    protocol_component_hashes,
    score_choice_logprobs,
)
from src.eval.direct_logit_scorer import (
    DIRECT_LOGIT_BACKEND,
    VLLM_CHOICE_BACKEND_STATUS,
    DirectLogitScorerError,
    load_direct_logit_route,
    run_direct_choice_rows,
    validate_direct_logit_repetitions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SUPERVISION_FIELDS = frozenset(
    {"answer", "answer_idx", "answer_index", "gold", "label", "reasoning", "response", "solution"}
)


class ControllerV2RuntimeError(RuntimeError):
    """Runtime/configuration violation before a model can execute."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ControllerV2RuntimeError(f"invalid JSONL at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ControllerV2RuntimeError(f"non-object JSONL at {path}:{number}")
            yield row


def load_controller_v2_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ControllerV2RuntimeError("Controller v2 config must be a mapping")
    required = {
        "schema_version", "protocol_version", "protocol_sha256", "status", "seed", "model", "data",
        "length_smoke", "choice_score", "generation", "statistics", "teacher_gate", "execution",
    }
    if set(payload) != required or payload.get("schema_version") != 2 or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ControllerV2RuntimeError("Controller v2 config schema/protocol drift")
    if payload.get("seed") != 42:
        raise ControllerV2RuntimeError("Controller v2 seed must remain 42")
    model = payload["model"]
    if model.get("id") != "Qwen/Qwen3-4B":
        raise ControllerV2RuntimeError("Controller v2 model identity drift")
    if model.get("revision") != BASE_MODEL_REVISION or model.get("tokenizer_revision") != BASE_MODEL_REVISION:
        raise ControllerV2RuntimeError("Controller v2 model/tokenizer revision drift")
    if model.get("medical_lora_sha256") != MEDICAL_LORA_SHA256:
        raise ControllerV2RuntimeError("Controller v2 Medical LoRA SHA drift")
    for field in (
        "path",
        "manifest_path",
        "manifest_sha256",
        "medical_lora_path",
        "medical_lora_manifest_path",
        "medical_lora_manifest_sha256",
        "medical_lora_weight_sha256",
    ):
        if not str(model.get(field) or "").strip():
            raise ControllerV2RuntimeError(f"Controller v2 model.{field} is required")
    model_manifest_path = _resolve(str(model["manifest_path"]))
    if (
        not model_manifest_path.is_file()
        or _sha256(model_manifest_path) != model["manifest_sha256"]
    ):
        raise ControllerV2RuntimeError("Controller v2 model manifest SHA mismatch")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if (
        model_manifest.get("model_id") != model["id"]
        or model_manifest.get("immutable_revision") != BASE_MODEL_REVISION
        or model_manifest.get("tokenizer_revision") != BASE_MODEL_REVISION
    ):
        raise ControllerV2RuntimeError("Controller v2 model manifest identity drift")
    if Path(str(model["path"])).resolve() != Path(
        str(model_manifest.get("local_persistent_path") or "")
    ).resolve():
        raise ControllerV2RuntimeError("Controller v2 model path is not the verified model root")
    data = payload["data"]
    if data.get("roles") != ["medical_controller_dev", "general_controller_dev"]:
        raise ControllerV2RuntimeError("Controller v2 requires both frozen controller roles")
    if data.get("prompt_label_separated") is not True or any("final" in str(role) for role in data.get("roles", [])):
        raise ControllerV2RuntimeError("Controller v2 prompt/label or final-role contract drift")
    if payload["execution"].get("final_authorized") is not False:
        raise ControllerV2RuntimeError("Controller v2 cannot authorize final")
    execution = payload["execution"]
    if (
        execution.get("runtime_policy") != "progress_aware"
        or execution.get("global_runtime_hard_limit") is not None
        or execution.get("cost_monitor_interval_minutes") != 30
        or execution.get("model_load_no_progress_minutes") != 20
        or execution.get("evaluator_no_progress_minutes") != 30
        or execution.get("reference_cost_checkpoints_cny") != [3.0, 6.0, 10.0]
    ):
        raise ControllerV2RuntimeError("Controller v2 progress-aware execution policy drift")
    choice = payload["choice_score"]
    runtime = choice.get("runtime")
    required_runtime = {
        "model_loader": "AutoModelForCausalLM",
        "lora_loader": "PeftModel.from_pretrained",
        "merge_lora": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
        "torch_compile": False,
        "torch_deterministic_algorithms": True,
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "cuda_device_max_connections": 1,
        "micro_smoke_device": "cuda:0",
        "full_evaluation_device": "cuda:1",
    }
    if (
        choice.get("backend") != DIRECT_LOGIT_BACKEND
        or choice.get("legacy_vllm_prompt_logprobs") != VLLM_CHOICE_BACKEND_STATUS
        or choice.get("batch_size") != 1
        or choice.get("single_token_formal_path") is not True
        or choice.get("float32_log_softmax") is not True
        or choice.get("score_repeat_tolerance") != 1e-4
        or choice.get("micro_smoke_repeat_count") != 3
        or runtime != required_runtime
    ):
        raise ControllerV2RuntimeError("Controller v2 direct-logit choice contract drift")
    generation = payload["generation"]
    if (
        generation.get("backend") != "vllm"
        or generation.get("choice_score_authority") is not False
        or generation.get("vllm_enable_v1_multiprocessing") is not False
        or generation.get("enable_prefix_caching") is not False
        or generation.get("enforce_eager") is not True
        or generation.get("enable_chunked_prefill") is not False
        or generation.get("runtime_device") != "cuda:0"
    ):
        raise ControllerV2RuntimeError("Controller v2 vLLM generation-only contract drift")
    component_hashes = protocol_component_hashes()
    if payload.get("protocol_sha256") != component_hashes["protocol_sha256"]:
        raise ControllerV2RuntimeError("Controller v2 protocol SHA drift")
    for section, fields in (
        (payload["choice_score"], ("prompt_sha256", "scorer_sha256")),
        (payload["generation"], ("prompt_sha256", "parser_sha256")),
    ):
        for field in fields:
            if section.get(field) != component_hashes[field]:
                raise ControllerV2RuntimeError(f"Controller v2 {field} drift")
    manifest = _resolve(data.get("manifest_path", ""))
    if not manifest.is_file() or _sha256(manifest) != data.get("manifest_sha256"):
        raise ControllerV2RuntimeError("Controller manifest SHA mismatch")
    smoke = payload["length_smoke"]
    smoke_manifest = _resolve(smoke.get("manifest_path", ""))
    if not smoke_manifest.is_file() or _sha256(smoke_manifest) != smoke.get("manifest_sha256"):
        raise ControllerV2RuntimeError("length smoke manifest SHA mismatch")
    if smoke.get("medical_count") != 16 or smoke.get("general_count") != 16:
        raise ControllerV2RuntimeError("length smoke must freeze 16 medical and 16 general IDs")
    if smoke.get("initial_max_new_tokens") != 512 or smoke.get("expanded_max_new_tokens") != 1024:
        raise ControllerV2RuntimeError("Controller v2 generation length policy drift")
    return payload


def _artifact_path(manifest_path: Path, declared: str) -> Path:
    value = Path(declared)
    if value.is_absolute():
        return value
    for candidate in (REPO_ROOT / value, manifest_path.parent / value, manifest_path.parent / value.name):
        if candidate.is_file():
            return candidate
    return REPO_ROOT / value


def _label_artifact_attestation(manifest_path: Path, roles: Sequence[str]) -> str:
    """Bind declared label metadata without opening the label artifacts.

    This deliberately reads only the controller manifest.  The full GPU gate
    can therefore prove it is using the same declared label artifacts without
    opening labels before or while the model execution boundary is resident.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = []
    for role in roles:
        metadata = manifest.get("roles", {}).get(role)
        files = metadata.get("files") if isinstance(metadata, Mapping) else None
        labels = [
            item for item in files or []
            if isinstance(item, Mapping) and str(item.get("path", "")).endswith(".labels.jsonl")
        ]
        if len(labels) != 1 or not str(labels[0].get("sha256") or ""):
            raise ControllerV2RuntimeError(f"{role} label artifact metadata is invalid")
        artifacts.append({
            "role": role,
            "path": str(labels[0]["path"]),
            "sha256": str(labels[0]["sha256"]),
        })
    payload = {
        "controller_manifest_sha256": _sha256(manifest_path),
        "label_artifacts": artifacts,
        "roles": list(roles),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def iter_prompt_rows(
    manifest_path: str | Path,
    roles: Sequence[str],
    *,
    selected_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("data_protocol_version") != "ca-opd-data-v2" or manifest.get("final_authorized") is not False:
        raise ControllerV2RuntimeError("controller manifest protocol/final state is invalid")
    seen: set[str] = set()
    for role in roles:
        if role not in CONTROLLER_ROLES:
            raise ControllerV2RuntimeError("prompt iterator cannot read final/non-controller roles")
        metadata = manifest.get("roles", {}).get(role)
        if not isinstance(metadata, dict):
            raise ControllerV2RuntimeError(f"controller manifest lacks role {role}")
        prompt_meta = next(
            (item for item in metadata.get("files", []) if str(item.get("path", "")).endswith(".prompts.jsonl")),
            None,
        )
        if not isinstance(prompt_meta, dict):
            raise ControllerV2RuntimeError(f"{role} prompt artifact is missing")
        prompt_path = _artifact_path(path, str(prompt_meta.get("path", "")))
        if not prompt_path.is_file() or _sha256(prompt_path) != str(prompt_meta.get("sha256", "")):
            raise ControllerV2RuntimeError(f"{role} prompt artifact SHA mismatch")
        for row in _iter_jsonl(prompt_path):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen or row.get("target_role") != role:
                raise ControllerV2RuntimeError("controller prompt identity/role is invalid")
            seen.add(sample_id)
            leaked = _SUPERVISION_FIELDS & set(row)
            if leaked:
                raise ControllerV2RuntimeError(f"controller prompt contains supervision: {sorted(leaked)}")
            if selected_ids is None or sample_id in selected_ids:
                yield row
    if selected_ids is not None and not selected_ids.issubset(seen):
        raise ControllerV2RuntimeError("selected smoke IDs are absent from controller prompts")


def extract_candidate_logprobs(
    output: Any, *, prompt_length: int, candidate_token_ids: Sequence[int]
) -> list[float]:
    values = getattr(output, "prompt_logprobs", None)
    if not isinstance(values, list) or len(values) < prompt_length + len(candidate_token_ids):
        raise ControllerV2RuntimeError("vLLM prompt_logprobs length is misaligned")
    result: list[float] = []
    for offset, token_id in enumerate(candidate_token_ids):
        entry = values[prompt_length + offset]
        if not isinstance(entry, Mapping) or int(token_id) not in entry:
            raise ControllerV2RuntimeError("candidate token is missing from vLLM prompt_logprobs")
        value = entry[int(token_id)]
        logprob = float(getattr(value, "logprob", value))
        if not (-float("inf") < logprob < float("inf")):
            raise ControllerV2RuntimeError("candidate logprob is non-finite")
        result.append(logprob)
    return result


def assert_gpu_execution_authorized(
    config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> None:
    values = os.environ if environ is None else environ
    expected = str(config["execution"]["required_confirmation"])
    if values.get("CA_OPD_ALLOW_CONTROLLER_V2_GPU") != "1" or values.get("CA_OPD_CONFIRM_RUN") != expected:
        raise ControllerV2RuntimeError("Controller v2 GPU execution is not authorized")
    if config["execution"].get("final_authorized") is not False:
        raise ControllerV2RuntimeError("Controller v2 cannot run with final authorization")
    _authorized_budget(config, values)


def _authorized_budget(
    config: Mapping[str, Any], values: Mapping[str, str]
) -> dict[str, Any]:
    """Validate the progress-aware paid-session identity and monitoring price.

    Controller v2 no longer has a global wall-clock or cost kill switch.  The
    live price remains mandatory for honest monitoring, while scientific and
    no-progress gates control stopping decisions.
    """

    try:
        price = float(values["CA_OPD_LIVE_PRICE_CNY_PER_HOUR"])
    except (KeyError, TypeError, ValueError) as error:
        raise ControllerV2RuntimeError(
            "Controller v2 GPU live price is incomplete"
        ) from error
    if price <= 0:
        raise ControllerV2RuntimeError("Controller v2 GPU live price must be positive")
    execution = config["execution"]
    global_limit = str(values.get("CA_OPD_GLOBAL_RUNTIME_HARD_LIMIT", "")).strip().lower()
    if (
        execution.get("runtime_policy") != "progress_aware"
        or execution.get("global_runtime_hard_limit") is not None
        or values.get("CA_OPD_RUNTIME_POLICY") != "progress_aware"
        or global_limit not in {"", "null", "none"}
    ):
        raise ControllerV2RuntimeError("Controller v2 progress-aware runtime policy is invalid")
    return {
        "live_price_cny_per_hour": price,
        "runtime_policy": "progress_aware",
        "global_runtime_hard_limit": None,
        "cost_monitor_interval_minutes": int(execution["cost_monitor_interval_minutes"]),
        "runtime_minutes": None,
        "cost_cap_cny": None,
        "estimated_cost_cny": None,
    }


def verify_medical_lora_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the frozen P3 adapter identity without loading model weights."""

    model = config["model"]
    manifest_path = _resolve(model["medical_lora_manifest_path"])
    adapter_dir = _resolve(model["medical_lora_path"])
    if not manifest_path.is_file() or _sha256(manifest_path) != model["medical_lora_manifest_sha256"]:
        raise ControllerV2RuntimeError("Medical LoRA artifact manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_adapter = Path(str(manifest.get("adapter_path") or "")).resolve()
    if declared_adapter != adapter_dir.resolve():
        raise ControllerV2RuntimeError("Medical LoRA adapter path differs from frozen manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("verification_status") != "files_and_combined_sha_verified"
        or manifest.get("model_revision") != BASE_MODEL_REVISION
        or manifest.get("tokenizer_revision") != BASE_MODEL_REVISION
        or manifest.get("adapter_sha256") != MEDICAL_LORA_SHA256
        or manifest.get("adapter_model_sha256") != model["medical_lora_weight_sha256"]
    ):
        raise ControllerV2RuntimeError("Medical LoRA artifact identity/verification drift")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or len(inventory) != 2:
        raise ControllerV2RuntimeError("Medical LoRA file inventory is incomplete")
    by_name = {Path(str(item.get("path") or "")).name: item for item in inventory}
    if set(by_name) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ControllerV2RuntimeError("Medical LoRA file inventory names drift")
    aggregate = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_dir / name
        item = by_name[name]
        if not path.is_file() or _sha256(path) != str(item.get("sha256") or ""):
            raise ControllerV2RuntimeError(f"Medical LoRA file SHA mismatch: {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                aggregate.update(block)
    if aggregate.hexdigest() != MEDICAL_LORA_SHA256:
        raise ControllerV2RuntimeError("Medical LoRA ordered aggregate SHA mismatch")
    return {
        "artifact_valid": True,
        "manifest_sha256": model["medical_lora_manifest_sha256"],
        "adapter_sha256": MEDICAL_LORA_SHA256,
        "adapter_weight_sha256": model["medical_lora_weight_sha256"],
        "base_model_revision": BASE_MODEL_REVISION,
    }


def controller_v2_cpu_preflight(config_path: str | Path) -> dict[str, Any]:
    """Run the shared stage-aware gate with mock checkpoint and no model imports."""

    from src.utils.preflight import PreflightError, run_preflight

    path = Path(config_path).resolve()
    config = load_controller_v2_config(path)
    hashes = protocol_component_hashes()
    request = {
        "mock_checkpoint": True,
        "controller_manifest": str(_resolve(config["data"]["manifest_path"])),
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "evaluator_config": str(path),
        "prompt_label_separated": True,
        "decoding": {"temperature": 0.0, "do_sample": False, "seed": 42},
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "base_model_revision": BASE_MODEL_REVISION,
        "model_id": config["model"]["id"],
        "model_path": str(Path(config["model"]["path"]).resolve()),
        "model_manifest_sha256": config["model"]["manifest_sha256"],
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "choice_backend": config["choice_score"]["backend"],
        "vllm_choice_backend_status": config["choice_score"]["legacy_vllm_prompt_logprobs"],
        "direct_logit_runtime": config["choice_score"]["runtime"],
        "direct_logit_batch_size": config["choice_score"]["batch_size"],
        "float32_log_softmax": config["choice_score"]["float32_log_softmax"],
        **hashes,
        "length_smoke_status": config["length_smoke"]["status"],
        "evaluation_phase": "length_smoke",
        "cpu_dry_run": True,
        "final_authorized": False,
        "result_output_dir": str(
            Path(config["execution"]["output_root"]) / config["execution"]["run_id"]
        ),
    }
    try:
        result = run_preflight("controller_eval", request, mode="dry-run")
    except PreflightError as error:
        raise ControllerV2RuntimeError(f"stage-aware CPU preflight failed: {error}") from error
    return result.to_dict()


def controller_v2_gpu_preflight(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    evaluation_phase: str = "length_smoke",
    length_decision: str | Path | None = None,
    result_output_dir: str | Path | None = None,
    prevalidated_label_attestation: str | None = None,
) -> dict[str, Any]:
    """Formal metadata/cost gate for the authorized length-smoke phase."""

    from src.utils.preflight import PreflightError, run_preflight

    values = os.environ if environ is None else environ
    config = load_controller_v2_config(config_path)
    assert_gpu_execution_authorized(config, environ=values)
    budget = _authorized_budget(config, values)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    hashes = protocol_component_hashes()
    request = {
        "mock_checkpoint": True,
        "controller_manifest": str(_resolve(config["data"]["manifest_path"])),
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "evaluator_config": str(Path(config_path).resolve()),
        "prompt_label_separated": True,
        "decoding": {"temperature": 0.0, "do_sample": False, "seed": 42},
        "protocol_version": PROTOCOL_VERSION,
        "base_model_revision": BASE_MODEL_REVISION,
        "model_id": config["model"]["id"],
        "model_path": str(Path(config["model"]["path"]).resolve()),
        "model_manifest_sha256": config["model"]["manifest_sha256"],
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "choice_backend": config["choice_score"]["backend"],
        "vllm_choice_backend_status": config["choice_score"]["legacy_vllm_prompt_logprobs"],
        "direct_logit_runtime": config["choice_score"]["runtime"],
        "direct_logit_batch_size": config["choice_score"]["batch_size"],
        "float32_log_softmax": config["choice_score"]["float32_log_softmax"],
        **hashes,
        "length_smoke_status": (
            "frozen_before_full_evaluation"
            if evaluation_phase == "full"
            else config["length_smoke"]["status"]
        ),
        "evaluation_phase": evaluation_phase,
        "final_authorized": False,
        "result_output_dir": str(result_output_dir or (
            Path(config["execution"]["output_root"]) / config["execution"]["run_id"]
        )),
        "git_sha": git_sha,
        "dirty_worktree": bool(dirty),
        "committed_worktree": not bool(dirty),
        "runtime_gate": {
            "runtime_policy": budget["runtime_policy"],
            "global_runtime_hard_limit": budget["global_runtime_hard_limit"],
            "live_price_cny_per_hour": budget["live_price_cny_per_hour"],
            "cost_monitor_interval_minutes": budget["cost_monitor_interval_minutes"],
        },
    }
    if evaluation_phase == "full":
        decision_path = Path(str(length_decision or ""))
        if not decision_path.is_file():
            raise ControllerV2RuntimeError("full evaluation requires a frozen length decision")
        request["length_decision"] = str(decision_path)
        request["length_decision_sha256"] = _sha256(decision_path)
        request["prevalidated_label_attestation"] = str(
            prevalidated_label_attestation or ""
        )
    elif evaluation_phase != "length_smoke":
        raise ControllerV2RuntimeError("unsupported Controller v2 evaluation phase")
    try:
        result = run_preflight("controller_eval", request, mode="formal")
    except (PreflightError, subprocess.CalledProcessError) as error:
        raise ControllerV2RuntimeError(f"formal Controller v2 preflight failed: {error}") from error
    return {
        **result.to_dict(),
        **budget,
        "git_sha": git_sha,
        "evaluation_phase": evaluation_phase,
        "gpu_process_started": False,
        "label_artifact_attestation": _label_artifact_attestation(
            _resolve(config["data"]["manifest_path"]), config["data"]["roles"]
        ),
    }


def write_prediction_artifact(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise ControllerV2RuntimeError("prediction artifact requires a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            if _SUPERVISION_FIELDS & set(row):
                raise ControllerV2RuntimeError("prediction artifact contains supervision")
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if not count:
        temporary.unlink(missing_ok=True)
        raise ControllerV2RuntimeError("prediction artifact cannot be empty")
    os.replace(temporary, destination)
    return {"path": str(destination), "count": count, "sha256": _sha256(destination)}


def run_choice_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenize,
    score_sequence=None,
    score_sequences=None,
    batch_size: int = 8,
) -> Iterator[dict[str, Any]]:
    """Label-free choice orchestration; ``score_sequence`` is the GPU boundary."""
    if (score_sequence is None) == (score_sequences is None):
        raise ControllerV2RuntimeError("provide exactly one choice scoring backend")

    def process(chunk: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        prepared = []
        flattened: list[tuple[list[int], list[int]]] = []
        for row in chunk:
            request = build_choice_request(row, tokenize=tokenize)
            prompt_ids = list(request.prompt_token_ids)
            prepared.append((row, request, prompt_ids))
            flattened.extend(
                (prompt_ids, list(candidate.token_ids)) for candidate in request.candidates
            )
        if score_sequences is not None:
            scored = list(score_sequences(flattened))
        else:
            scored = [score_sequence(prompt, candidate) for prompt, candidate in flattened]
        if len(scored) != len(flattened):
            raise ControllerV2RuntimeError("choice scoring backend response count mismatch")
        cursor = 0
        for row, request, _ in prepared:
            token_logprobs = {}
            for candidate in request.candidates:
                token_logprobs[candidate.label] = list(scored[cursor])
                cursor += 1
            prediction = score_choice_logprobs(
                sample_id=request.sample_id, candidate_token_logprobs=token_logprobs
            )
            yield {
                **prediction.as_dict(),
                "target_role": request.target_role,
                "domain": str(row.get("domain") or ""),
                "subject": str(row.get("subject") or ""),
                "protocol_version": PROTOCOL_VERSION,
                "metric_track": "choice_score",
                "candidate_tokenization": [
                    {"label": item.label, "token_ids": list(item.token_ids)}
                    for item in request.candidates
                ],
            }

    buffered: list[Mapping[str, Any]] = []
    for row in rows:
        buffered.append(row)
        if len(buffered) >= batch_size:
            yield from process(buffered)
            buffered = []
    if buffered:
        yield from process(buffered)


def run_generation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    generate,
    max_new_tokens: int,
    batch_size: int = 8,
) -> Iterator[dict[str, Any]]:
    """Answer-first generation orchestration with strict parsing and finish evidence."""

    buffered: list[Mapping[str, Any]] = []

    def flush(chunk: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        prompts = [build_generative_prompt(row) for row in chunk]
        outputs = generate(prompts, max_new_tokens)
        if len(outputs) != len(chunk):
            raise ControllerV2RuntimeError("generation backend response count mismatch")
        for row, output in zip(chunk, outputs, strict=True):
            options = row.get("options")
            if not isinstance(options, list):
                raise ControllerV2RuntimeError("generation row lacks ordered options")
            text = str(output.get("text") or "")
            parsed = parse_generation_v2(text, option_count=len(options))
            token_ids = output.get("token_ids")
            if not isinstance(token_ids, list):
                raise ControllerV2RuntimeError("generation backend must return token_ids")
            finish_reason = str(output.get("finish_reason") or "unknown")
            truncated = finish_reason == "length" and len(token_ids) >= max_new_tokens
            explicit_positions = [
                index for marker in ("答案：", "最终答案：")
                if (index := text.find(marker)) >= 0
            ]
            yield {
                "sample_id": str(row.get("sample_id") or ""),
                "target_role": str(row.get("target_role") or ""),
                "domain": str(row.get("domain") or ""),
                "subject": str(row.get("subject") or ""),
                "protocol_version": PROTOCOL_VERSION,
                "metric_track": "generation",
                "predicted_label": parsed.letter,
                "parse_method": parsed.method,
                "invalid": not parsed.valid,
                "answer_first_compliant": parsed.method in {
                    "answer_first", "answer_first_and_final_consistent"
                },
                "finish_reason": finish_reason,
                "generated_token_count": len(token_ids),
                "truncated": truncated,
                "stop_reason": output.get("stop_reason"),
                "eos_observed": bool(output.get("eos_observed", False)),
                "explicit_answer_char_index": min(explicit_positions) if explicit_positions else None,
                "thinking_tag": "<think>" in text or "</think>" in text,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                # Raw response remains only in ignored run output.
                "response": text,
            }

    for row in rows:
        buffered.append(row)
        if len(buffered) >= batch_size:
            yield from flush(buffered)
            buffered = []
    if buffered:
        yield from flush(buffered)


def validate_choice_runtime_smoke(
    b0_first: Sequence[Mapping[str, Any]],
    b0_repeat: Sequence[Mapping[str, Any]],
    b1_first: Sequence[Mapping[str, Any]],
    b1_repeat: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate real candidate scoring before the full controller run.

    This boundary is deliberately label-free.  It proves that both the Base
    and Medical-LoRA routes return finite, discriminative and repeatable scores
    for the same frozen prompt identities without opening a label artifact.
    """

    def index(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for value in rows:
            row = dict(value)
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in result:
                raise ControllerV2RuntimeError(f"{name} choice smoke has a missing/duplicate sample_id")
            scores = row.get("candidate_scores")
            tokenization = row.get("candidate_tokenization")
            if not isinstance(scores, Mapping) or not isinstance(tokenization, list):
                raise ControllerV2RuntimeError(f"{name} choice smoke artifact is incomplete")
            numeric = {str(label): float(score) for label, score in scores.items()}
            if not numeric or not all(math.isfinite(score) for score in numeric.values()):
                raise ControllerV2RuntimeError(f"{name} choice smoke has non-finite candidate scores")
            if len(set(numeric.values())) == 1:
                raise ControllerV2RuntimeError(f"{name} choice smoke candidate scores are all identical")
            predicted = str(row.get("predicted_label") or "")
            if predicted not in numeric:
                raise ControllerV2RuntimeError(f"{name} choice smoke argmax is not a legal candidate")
            token_labels = {str(item.get("label") or "") for item in tokenization if isinstance(item, Mapping)}
            if token_labels != set(numeric) or any(
                not isinstance(item.get("token_ids"), list) or not item["token_ids"]
                for item in tokenization if isinstance(item, Mapping)
            ):
                raise ControllerV2RuntimeError(f"{name} choice smoke tokenization is incomplete")
            result[sample_id] = row
        if not result:
            raise ControllerV2RuntimeError(f"{name} choice smoke cannot be empty")
        return result

    score_repeat_tolerance = 1e-4
    first = {"B0": index(b0_first, name="B0"), "B1": index(b1_first, name="B1")}
    repeat = {"B0": index(b0_repeat, name="B0 repeat"), "B1": index(b1_repeat, name="B1 repeat")}
    if set(first["B0"]) != set(first["B1"]):
        raise ControllerV2RuntimeError("B0/B1 choice smoke sample sets differ")
    evidence: dict[str, Any] = {}
    max_abs_score_delta = 0.0
    for name in ("B0", "B1"):
        if set(first[name]) != set(repeat[name]):
            raise ControllerV2RuntimeError(f"{name} choice smoke repeat sample sets differ")
        for sample_id in sorted(first[name]):
            original = first[name][sample_id]
            rerun = repeat[name][sample_id]
            original_scores = {
                str(label): float(score)
                for label, score in original["candidate_scores"].items()
            }
            rerun_scores = {
                str(label): float(score)
                for label, score in rerun["candidate_scores"].items()
            }
            original_rank = sorted(original_scores, key=lambda label: (-original_scores[label], label))
            rerun_rank = sorted(rerun_scores, key=lambda label: (-rerun_scores[label], label))
            if (
                original.get("predicted_label") != rerun.get("predicted_label")
                or original_rank != rerun_rank
            ):
                raise ControllerV2RuntimeError(
                    f"{name} choice smoke prediction/ranking is not deterministic"
                )
            if original.get("candidate_tokenization") != rerun.get("candidate_tokenization"):
                raise ControllerV2RuntimeError(
                    f"{name} choice smoke candidate tokenization is not deterministic"
                )
            if set(original_scores) != set(rerun_scores):
                raise ControllerV2RuntimeError(f"{name} choice smoke candidate sets differ")
            delta = max(
                abs(original_scores[label] - rerun_scores[label])
                for label in original_scores
            )
            max_abs_score_delta = max(max_abs_score_delta, delta)
            if delta > score_repeat_tolerance:
                raise ControllerV2RuntimeError(
                    f"{name} choice smoke score repeat delta {delta:.8g} exceeds "
                    f"{score_repeat_tolerance:.8g}"
                )
        evidence[name] = [
            {
                "sample_id": sample_id,
                "predicted_label": first[name][sample_id]["predicted_label"],
                "candidate_scores": first[name][sample_id]["candidate_scores"],
                "candidate_tokenization": first[name][sample_id]["candidate_tokenization"],
            }
            for sample_id in sorted(first[name])
        ]
    return {
        "status": "PASS",
        "sample_count": len(first["B0"]),
        "deterministic": True,
        "deterministic_definition": "identical_prediction_and_ranking_with_bounded_score_drift",
        "score_repeat_tolerance": score_repeat_tolerance,
        "max_abs_score_delta": max_abs_score_delta,
        "labels_opened_during_execution": False,
        "models": evidence,
    }


def _load_smoke_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("total_count") != 32
        or payload.get("final_authorized") is not False
        or payload.get("labels_included") is not False
    ):
        raise ControllerV2RuntimeError("length smoke identity manifest is invalid")
    ids = {
        str(item["sample_id"])
        for rows in payload.get("roles", {}).values()
        for item in rows
    }
    if len(ids) != 32:
        raise ControllerV2RuntimeError("length smoke identities are not 32 unique IDs")
    return ids


def _load_choice_smoke_ids(path: Path) -> set[str]:
    """Choose two frozen length-smoke identities per controller role."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: set[str] = set()
    for role in ("medical_controller_dev", "general_controller_dev"):
        rows = payload.get("roles", {}).get(role)
        if not isinstance(rows, list) or len(rows) < 2:
            raise ControllerV2RuntimeError(f"choice smoke lacks frozen IDs for {role}")
        selected.update(sorted(str(item.get("sample_id") or "") for item in rows)[:2])
    if len(selected) != 4 or "" in selected:
        raise ControllerV2RuntimeError("choice smoke identities are not four unique frozen IDs")
    return selected


def _label_map(manifest_path: Path, roles: Sequence[str]) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for role in roles:
        metadata = manifest["roles"][role]
        label_meta = next(
            item for item in metadata["files"] if str(item["path"]).endswith(".labels.jsonl")
        )
        label_path = _artifact_path(manifest_path, str(label_meta["path"]))
        if _sha256(label_path) != str(label_meta["sha256"]):
            raise ControllerV2RuntimeError(f"{role} label artifact SHA mismatch")
        for row in _iter_jsonl(label_path):
            sample_id = str(row.get("sample_id") or "")
            label = str(row.get("answer_idx") or "").upper()
            if not sample_id or sample_id in labels or label not in "ABCDE" or row.get("target_role") != role:
                raise ControllerV2RuntimeError("controller label identity/value is invalid")
            labels[sample_id] = label
    return labels


def _write_jsonl_allow_response(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if path.exists():
        raise ControllerV2RuntimeError("generation artifact requires a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if not count:
        temporary.unlink(missing_ok=True)
        raise ControllerV2RuntimeError("generation artifact cannot be empty")
    os.replace(temporary, path)
    return {"path": str(path), "count": count, "sha256": _sha256(path)}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _compact_score(rows: Iterable[Mapping[str, Any]], labels: Mapping[str, str]) -> list[dict[str, Any]]:
    scored = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in seen or sample_id not in labels:
            raise ControllerV2RuntimeError("prediction/label ID mismatch")
        seen.add(sample_id)
        predicted = row.get("predicted_label")
        response = str(row.get("response") or "")
        candidate_scores = row.get("candidate_scores")
        margin = None
        if isinstance(candidate_scores, Mapping) and len(candidate_scores) >= 2:
            ordered_scores = sorted(
                (float(score) for score in candidate_scores.values()), reverse=True
            )
            if all(math.isfinite(score) for score in ordered_scores):
                margin = ordered_scores[0] - ordered_scores[1]
        scored.append({
            "sample_id": sample_id,
            "target_role": row.get("target_role"),
            "domain": row.get("domain"),
            "subject": row.get("subject"),
            "correct": predicted == labels[sample_id],
            "invalid": bool(row.get("invalid", predicted is None)),
            "truncated": bool(row.get("truncated", False)),
            "answer_first_compliant": bool(row.get("answer_first_compliant", False)),
            "thinking_tag": "<think>" in response or "</think>" in response,
            "finish_reason": str(row.get("finish_reason") or "not_applicable"),
            "generated_token_count": int(row.get("generated_token_count") or 0),
            "candidate_margin": margin,
        })
    if seen != set(labels):
        raise ControllerV2RuntimeError("prediction/label sample sets differ")
    return scored


def _track_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    domains: dict[str, list[Mapping[str, Any]]] = {}
    subjects: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        domains.setdefault(str(row.get("domain") or "unknown"), []).append(row)
        if row.get("domain") == "general":
            subjects.setdefault(str(row.get("subject") or "unknown"), []).append(row)
    accuracy = lambda values: sum(bool(row["correct"]) for row in values) / len(values)
    subject_accuracy = {name: accuracy(values) for name, values in sorted(subjects.items())}
    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("finish_reason") or "not_applicable")
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    margins = sorted(
        float(row["candidate_margin"])
        for row in rows
        if row.get("candidate_margin") is not None
    )

    def quantile(values: Sequence[float], fraction: float) -> float:
        return values[int(round((len(values) - 1) * fraction))]

    margin_distribution = None if not margins else {
        "count": len(margins),
        "min": margins[0],
        "p50": quantile(margins, 0.50),
        "p90": quantile(margins, 0.90),
        "p95": quantile(margins, 0.95),
        "max": margins[-1],
    }
    return {
        "count": len(rows),
        "medical_accuracy": accuracy(domains["medical"]),
        "general_micro_accuracy": accuracy(domains["general"]),
        "general_macro_accuracy": sum(subject_accuracy.values()) / len(subject_accuracy),
        "per_subject_accuracy": subject_accuracy,
        "invalid_count": sum(bool(row.get("invalid")) for row in rows),
        "invalid_rate": sum(bool(row.get("invalid")) for row in rows) / len(rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in rows),
        "truncation_rate": sum(bool(row.get("truncated")) for row in rows) / len(rows),
        "answer_first_compliance_count": sum(bool(row.get("answer_first_compliant")) for row in rows),
        "answer_first_compliance_rate": sum(bool(row.get("answer_first_compliant")) for row in rows) / len(rows),
        "thinking_tag_count": sum(bool(row.get("thinking_tag")) for row in rows),
        "thinking_tag_rate": sum(bool(row.get("thinking_tag")) for row in rows) / len(rows),
        "finish_reason_distribution": dict(sorted(finish_reasons.items())),
        "candidate_margin_distribution": margin_distribution,
        "score_margin_le_repeat_tolerance_count": sum(
            margin <= 1e-4 for margin in margins
        ),
    }


def summarize_controller_tracks(
    choice_scored: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    generation_scored: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    artifact_valid: bool,
    generation_failure: str | None = None,
) -> dict[str, Any]:
    """Compute the knowledge gate even when post-choice generation fails.

    A generation failure is represented as a failed output contract, never as
    missing or fabricated choice evidence. This helper runs only after model
    release and after the independent label join.
    """

    from src.eval.paired_stats import paired_comparison, teacher_readiness

    choice_metrics = {name: _track_metrics(rows) for name, rows in choice_scored.items()}
    choice_pair = paired_comparison(choice_scored["B0"], choice_scored["B1"], seed=42)
    if generation_scored is None:
        if not generation_failure:
            raise ControllerV2RuntimeError("missing generation evidence requires an explicit failure")
        generation_metrics: dict[str, Any] = {}
        generation_pair: dict[str, Any] = {}
        invalid_rate = truncation_rate = 1.0
        generation_status = "failed_after_choice_completed"
    else:
        generation_metrics = {
            name: _track_metrics(rows) for name, rows in generation_scored.items()
        }
        generation_pair = paired_comparison(
            generation_scored["B0"], generation_scored["B1"], seed=42
        )
        invalid_rate = generation_metrics["B1"]["invalid_rate"]
        truncation_rate = generation_metrics["B1"]["truncation_rate"]
        generation_status = "completed"
    readiness = teacher_readiness(
        artifact_valid=artifact_valid,
        b0_medical_choice_accuracy=choice_metrics["B0"]["medical_accuracy"],
        b1_medical_choice_accuracy=choice_metrics["B1"]["medical_accuracy"],
        b1_generation_invalid_rate=invalid_rate,
        b1_generation_truncation_rate=truncation_rate,
    )
    return {
        "choice_metrics": choice_metrics,
        "choice_paired": choice_pair,
        "generation_metrics": generation_metrics,
        "generation_paired": generation_pair,
        "generation_status": generation_status,
        "generation_failure": generation_failure,
        "teacher_readiness": readiness,
    }


def validate_teacher_readiness_evidence(
    summary: Mapping[str, Any], *, artifact_valid: bool
) -> dict[str, Any]:
    """Recompute the frozen Teacher gate from complete Controller v2 evidence.

    Readiness booleans are never authoritative inputs.  A formal authorization
    requires the registered four-ID repeat smoke, the 300+209 full paired
    inventory, a pre-result length decision and either complete generation
    metrics or the explicit post-choice generation-failure state.
    """

    from src.eval.paired_stats import teacher_readiness

    if summary.get("choice_backend") != DIRECT_LOGIT_BACKEND:
        raise ControllerV2RuntimeError("Teacher readiness evidence has a non-authoritative choice backend")
    smoke = summary.get("runtime_smoke", {}).get("choice", {})
    if (
        smoke.get("status") != "PASS"
        or smoke.get("choice_backend") != DIRECT_LOGIT_BACKEND
        or smoke.get("sample_count") != 4
        or smoke.get("repeat_count") != 3
        or smoke.get("labels_opened_during_execution") is not False
    ):
        raise ControllerV2RuntimeError("Teacher readiness evidence lacks the frozen direct-logit smoke")
    length = summary.get("length_decision", {})
    selected = length.get("max_new_tokens")
    if (
        length.get("status") != "frozen_before_full_evaluation"
        or length.get("frozen") is not True
        or selected not in {512, 1024}
        or length.get("b0_max_new_tokens") != selected
        or length.get("b1_max_new_tokens") != selected
        or length.get("decision_basis") != "truncation_only"
    ):
        raise ControllerV2RuntimeError("Teacher readiness evidence lacks a valid frozen length decision")

    def accuracy(value: Any, *, field: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ControllerV2RuntimeError(f"Teacher readiness evidence {field} is invalid") from error
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ControllerV2RuntimeError(f"Teacher readiness evidence {field} is out of range")
        return numeric

    def validate_track(metrics: Any, *, track: str) -> dict[str, Mapping[str, Any]]:
        if not isinstance(metrics, Mapping) or set(metrics) != {"B0", "B1"}:
            raise ControllerV2RuntimeError(f"Teacher readiness evidence {track} metrics are incomplete")
        result: dict[str, Mapping[str, Any]] = {}
        for route in ("B0", "B1"):
            value = metrics.get(route)
            if not isinstance(value, Mapping) or value.get("count") != 509:
                raise ControllerV2RuntimeError(
                    f"Teacher readiness evidence {track} {route} does not cover 509 samples"
                )
            accuracy(value.get("medical_accuracy"), field=f"{track}.{route}.medical_accuracy")
            accuracy(value.get("general_macro_accuracy"), field=f"{track}.{route}.general_macro_accuracy")
            accuracy(value.get("general_micro_accuracy"), field=f"{track}.{route}.general_micro_accuracy")
            subjects = value.get("per_subject_accuracy")
            if not isinstance(subjects, Mapping) or len(subjects) != 8:
                raise ControllerV2RuntimeError(
                    f"Teacher readiness evidence {track} {route} lacks all eight general subjects"
                )
            for subject, score in subjects.items():
                accuracy(score, field=f"{track}.{route}.per_subject_accuracy.{subject}")
            result[route] = value
        return result

    def validate_pair(pair: Any, *, track: str) -> None:
        if (
            not isinstance(pair, Mapping)
            or pair.get("count") != 509
            or pair.get("same_sample_ids") is not True
            or pair.get("domains", {}).get("medical", {}).get("count") != 300
            or pair.get("domains", {}).get("general", {}).get("count") != 209
        ):
            raise ControllerV2RuntimeError(
                f"Teacher readiness evidence {track} pairing is not the frozen 300+209 inventory"
            )

    choice = validate_track(summary.get("choice_metrics"), track="choice")
    validate_pair(summary.get("choice_paired"), track="choice")
    generation_status = summary.get("generation_status")
    if generation_status == "completed":
        if summary.get("status") != "completed" or summary.get("generation_failure") is not None:
            raise ControllerV2RuntimeError("Teacher readiness evidence generation completion state conflicts")
        generation = validate_track(summary.get("generation_metrics"), track="generation")
        validate_pair(summary.get("generation_paired"), track="generation")
        invalid_rate = accuracy(
            generation["B1"].get("invalid_rate"), field="generation.B1.invalid_rate"
        )
        truncation_rate = accuracy(
            generation["B1"].get("truncation_rate"), field="generation.B1.truncation_rate"
        )
    elif generation_status == "failed_after_choice_completed":
        if (
            summary.get("status") != "choice_completed_generation_failed"
            or not str(summary.get("generation_failure") or "").strip()
            or summary.get("generation_metrics") not in ({}, None)
            or summary.get("generation_paired") not in ({}, None)
        ):
            raise ControllerV2RuntimeError("Teacher readiness evidence generation failure state conflicts")
        invalid_rate = truncation_rate = 1.0
    else:
        raise ControllerV2RuntimeError("Teacher readiness evidence generation status is incomplete")

    recomputed = teacher_readiness(
        artifact_valid=artifact_valid,
        b0_medical_choice_accuracy=accuracy(
            choice["B0"].get("medical_accuracy"), field="choice.B0.medical_accuracy"
        ),
        b1_medical_choice_accuracy=accuracy(
            choice["B1"].get("medical_accuracy"), field="choice.B1.medical_accuracy"
        ),
        b1_generation_invalid_rate=invalid_rate,
        b1_generation_truncation_rate=truncation_rate,
    )
    if dict(summary.get("teacher_readiness") or {}) != recomputed:
        raise ControllerV2RuntimeError(
            "Teacher readiness evidence does not reproduce the frozen readiness decision"
        )
    return recomputed


def write_standard_run_artifacts(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Write the aggregate v2 contract; raw predictions remain ignored locally."""

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if summary.get("protocol_version") != PROTOCOL_VERSION or summary.get("final_authorized") is not False:
        raise ControllerV2RuntimeError("standard artifacts require Controller v2 with final disabled")
    if summary.get("actual_cost_cny") is not None:
        raise ControllerV2RuntimeError("platform actual cost cannot be inferred by the runtime")
    readiness_summary = summary.get("teacher_readiness")
    if not isinstance(readiness_summary, Mapping) or not {
        "teacher_artifact_valid",
        "teacher_knowledge_ready",
        "teacher_generation_contract_ready",
    }.issubset(readiness_summary):
        raise ControllerV2RuntimeError("Controller v2 Teacher readiness schema is incomplete")
    validate_teacher_readiness_evidence(
        summary, artifact_valid=readiness_summary.get("teacher_artifact_valid") is True
    )
    git_sha = str(summary.get("git_sha") or "")
    if _HEX40.fullmatch(git_sha) is None:
        raise ControllerV2RuntimeError("Controller v2 artifacts require the executing Git SHA")
    try:
        started = datetime.fromisoformat(str(summary["started_at"]))
        ended = datetime.fromisoformat(str(summary["ended_at"]))
    except (KeyError, ValueError) as error:
        raise ControllerV2RuntimeError("run artifact timestamps are invalid") from error
    runtime_seconds = max(0.0, (ended - started).total_seconds())
    price = float(summary.get("live_price_cny_per_hour"))
    if price <= 0:
        raise ControllerV2RuntimeError("Controller v2 artifacts require a positive live price")
    cost = {
        "price_cny_per_hour": price,
        "process_runtime_seconds": runtime_seconds,
        "process_cost_cny": runtime_seconds / 3600.0 * price,
        "estimated_instance_cost_cny": None,
        "platform_billed_cost_cny": None,
        "actual_cost_cny": None,
        "cost_semantics": "process cost is derived from observed wall time; platform bill remains unknown",
    }
    metadata = {
        "run_id": summary["run_id"],
        "stage": "controller_eval",
        "status": summary["status"],
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "base_model_revision": BASE_MODEL_REVISION,
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "parser_sha256": config["generation"]["parser_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "choice_backend": config["choice_score"]["backend"],
        "vllm_choice_backend_status": config["choice_score"]["legacy_vllm_prompt_logprobs"],
        "git_sha": git_sha,
        "seed": 42,
        "final_authorized": False,
        "actual_cost_cny": None,
    }
    metric_rows = []
    for track in ("choice", "generation"):
        values = summary.get(f"{track}_metrics", {})
        for baseline in ("B0", "B1"):
            metric_rows.append({
                "protocol_version": PROTOCOL_VERSION,
                "metric_track": track,
                "baseline": baseline,
                "metrics": values.get(baseline, {}),
            })
        metric_rows.append({
            "protocol_version": PROTOCOL_VERSION,
            "metric_track": track,
            "baseline": "paired",
            "metrics": summary.get(f"{track}_paired", {}),
        })

    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    _atomic_json(directory / "data_manifest.json", {
        "manifest_path": config["data"]["manifest_path"],
        "manifest_sha256": config["data"]["manifest_sha256"],
        "roles": config["data"]["roles"],
        "medical_count": config["data"]["medical_count"],
        "general_count": config["data"]["general_count"],
        "final_authorized": False,
    })
    _atomic_json(directory / "model_manifest.json", {
        "model_id": config["model"]["id"],
        "base_model_revision": BASE_MODEL_REVISION,
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "model_manifest_path": config["model"]["manifest_path"],
        "model_manifest_sha256": config["model"]["manifest_sha256"],
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "medical_lora_weight_sha256": config["model"]["medical_lora_weight_sha256"],
        "medical_lora_manifest_sha256": config["model"]["medical_lora_manifest_sha256"],
        "merge_lora": False,
    })
    prediction_artifacts = summary.get("prediction_artifacts", {})
    _atomic_json(directory / "candidate_scores_manifest.json", {
        "choice_backend": DIRECT_LOGIT_BACKEND,
        "artifacts": prediction_artifacts.get("choice", {}),
        "labels_in_artifact": False,
        "final_authorized": False,
    })
    _atomic_json(directory / "generation_metadata_manifest.json", {
        "backend": "vllm",
        "status": summary.get("generation_status"),
        "artifacts": prediction_artifacts.get("generation", {}),
        "fields": [
            "finish_reason", "stop_reason", "generated_token_count", "eos_observed",
            "explicit_answer_char_index", "invalid", "truncated", "answer_first_compliant",
            "thinking_tag", "output_sha256",
        ],
        "final_authorized": False,
    })
    _atomic_json(directory / "checkpoints" / "index.json", {
        "status": "not_applicable",
        "stage": "controller_eval",
        "entries": [],
    })
    (directory / "stdout.log").write_text(
        "Controller v2 aggregate artifact; raw model outputs remain in ignored run files.\n",
        encoding="utf-8",
    )
    _atomic_json(directory / "summary.json", dict(summary))
    _atomic_json(directory / "metadata.json", metadata)
    _atomic_json(directory / "cost.json", cost)
    _atomic_json(directory / "protocol_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "parser_sha256": config["generation"]["parser_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "choice_backend": config["choice_score"]["backend"],
        "vllm_choice_backend_status": config["choice_score"]["legacy_vllm_prompt_logprobs"],
        "length_calibration_sha256": summary.get("length_calibration_sha256"),
        "base_model_revision": BASE_MODEL_REVISION,
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "git_sha": git_sha,
        "seed": 42,
        "final_authorized": False,
    })
    _atomic_json(directory / "aggregate.json", {
        "protocol_version": PROTOCOL_VERSION,
        "choice_metrics": summary.get("choice_metrics", {}),
        "generation_metrics": summary.get("generation_metrics", {}),
        "final_authorized": False,
    })
    _atomic_json(directory / "paired_stats.json", {
        "protocol_version": PROTOCOL_VERSION,
        "choice": summary.get("choice_paired", {}),
        "generation": summary.get("generation_paired", {}),
        "seed": 42,
        "final_authorized": False,
    })
    _atomic_json(directory / "teacher_gate.json", {
        **dict(readiness_summary),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "prompt_sha256": config["choice_score"]["prompt_sha256"],
        "parser_sha256": config["generation"]["parser_sha256"],
        "scorer_sha256": config["choice_score"]["scorer_sha256"],
        "length_calibration_sha256": summary.get("length_calibration_sha256"),
        "base_model_revision": BASE_MODEL_REVISION,
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "choice_backend": config["choice_score"]["backend"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "git_sha": git_sha,
        "final_authorized": False,
    })
    _atomic_json(directory / "labels_manifest.json", {
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "roles": config["data"]["roles"],
        "labels_physically_separate": True,
        "labels_opened_after_model_release": True,
        "labels_copied_into_run_artifact": False,
        "final_authorized": False,
    })
    metrics_tmp = directory / "metrics.jsonl.tmp"
    with metrics_tmp.open("w", encoding="utf-8") as handle:
        for row in metric_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(metrics_tmp, directory / "metrics.jsonl")

    files = []
    for path in sorted(directory.rglob("*"), key=lambda item: str(item.relative_to(directory))):
        relative = str(path.relative_to(directory))
        if path.is_file() and relative not in {
            "artifact_manifest.json", "teacher_readiness.json", "failure.json"
        }:
            files.append({
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "run_id": summary["run_id"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "base_model_revision": BASE_MODEL_REVISION,
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "choice_backend": config["choice_score"]["backend"],
        "git_sha": git_sha,
        "final_authorized": False,
        "files": files,
    }
    _atomic_json(directory / "artifact_manifest.json", manifest)
    readiness_payload = {
        **readiness_summary,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": config["protocol_sha256"],
        "medical_lora_sha256": MEDICAL_LORA_SHA256,
        "choice_backend": config["choice_score"]["backend"],
        "controller_manifest_sha256": config["data"]["manifest_sha256"],
        "controller_artifact_manifest_sha256": _sha256(directory / "artifact_manifest.json"),
        "final_authorized": False,
    }
    _atomic_json(directory / "teacher_readiness.json", readiness_payload)
    names = {
        "summary.json", "metadata.json", "metrics.jsonl", "cost.json",
        "teacher_readiness.json", "artifact_manifest.json", "config.yaml",
        "data_manifest.json", "stdout.log", "checkpoints/index.json",
        "protocol_manifest.json", "aggregate.json", "paired_stats.json",
        "teacher_gate.json", "labels_manifest.json", "model_manifest.json",
        "candidate_scores_manifest.json", "generation_metadata_manifest.json",
    }
    return {
        name: {
            "path": str(directory / name),
            "sha256": _sha256(directory / name),
            "bytes": (directory / name).stat().st_size,
        }
        for name in sorted(names)
    }


def validate_standard_run_artifacts(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ControllerV2RuntimeError("Controller v2 artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("protocol_sha256") != protocol_component_hashes()["protocol_sha256"]
        or manifest.get("choice_backend") != DIRECT_LOGIT_BACKEND
        or _HEX40.fullmatch(str(manifest.get("git_sha") or "")) is None
        or manifest.get("final_authorized") is not False
    ):
        raise ControllerV2RuntimeError("Controller v2 artifact protocol/final state is invalid")
    required = {
        "summary.json", "metadata.json", "metrics.jsonl", "cost.json",
        "config.yaml", "data_manifest.json", "stdout.log", "checkpoints/index.json",
        "protocol_manifest.json", "aggregate.json", "paired_stats.json",
        "teacher_gate.json", "labels_manifest.json", "model_manifest.json",
        "candidate_scores_manifest.json", "generation_metadata_manifest.json",
    }
    declared = {str(item.get("path")): item for item in manifest.get("files", [])}
    if not required.issubset(declared):
        raise ControllerV2RuntimeError("Controller v2 standard artifacts are incomplete")
    for name, item in declared.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ControllerV2RuntimeError(f"Controller v2 artifact SHA mismatch: {name}")
    readiness_path = directory / "teacher_readiness.json"
    if not readiness_path.is_file():
        raise ControllerV2RuntimeError("Controller v2 teacher readiness artifact is missing")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        readiness.get("controller_artifact_manifest_sha256") != _sha256(manifest_path)
        or readiness.get("protocol_version") != PROTOCOL_VERSION
        or readiness.get("protocol_sha256") != manifest.get("protocol_sha256")
        or readiness.get("medical_lora_sha256") != MEDICAL_LORA_SHA256
        or readiness.get("choice_backend") != DIRECT_LOGIT_BACKEND
        or readiness.get("controller_manifest_sha256") != manifest.get("controller_manifest_sha256")
        or readiness.get("final_authorized") is not False
    ):
        raise ControllerV2RuntimeError("Controller v2 readiness/manifest binding is invalid")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    recomputed_readiness = validate_teacher_readiness_evidence(
        summary, artifact_valid=readiness.get("teacher_artifact_valid") is True
    )
    for field in (
        "teacher_artifact_valid",
        "teacher_knowledge_ready",
        "teacher_generation_contract_ready",
    ):
        if readiness.get(field) != summary.get("teacher_readiness", {}).get(field):
            raise ControllerV2RuntimeError("Controller v2 readiness differs from bound summary")
    return {
        "status": "PASS",
        "validated_files": len(declared),
        "final_authorized": False,
        "manifest_payload": manifest,
        "readiness_payload": readiness,
        "summary_payload": summary,
        "recomputed_readiness": recomputed_readiness,
    }


def release_model_execution(device: str | None = None) -> None:  # pragma: no cover - GPU only
    """Release a previously deleted Transformers model before any label access."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            if device is None:
                torch.cuda.empty_cache()
            else:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
    except ImportError:
        pass


def run_direct_logit_micro_smoke(
    config: Mapping[str, Any],
    *,
    manifest_path: Path,
    roles: Sequence[str],
    selected_ids: set[str],
    run_dir: Path,
) -> dict[str, Any]:  # pragma: no cover - GPU only
    """Run the same four label-free identities three times on both formal routes."""

    route_runs: dict[str, list[list[dict[str, Any]]]] = {}
    for route in ("B0", "B1"):
        model = tokenizer = None
        try:
            device = str(config["choice_score"]["runtime"]["micro_smoke_device"])
            model, tokenizer, encode, _ = load_direct_logit_route(
                config, route, device=device
            )
            rows = list(iter_prompt_rows(manifest_path, roles, selected_ids=selected_ids))
            route_runs[route] = [
                list(
                    run_direct_choice_rows(
                        rows,
                        model=model,
                        tokenize=encode,
                        require_expected_qwen_ids=True,
                    )
                )
                for _ in range(int(config["choice_score"]["micro_smoke_repeat_count"]))
            ]
            _atomic_json(
                run_dir / f"{route.lower()}_direct_logit_micro_smoke_attempt.json",
                {
                    "route": route,
                    "choice_backend": DIRECT_LOGIT_BACKEND,
                    "labels_opened_during_execution": False,
                    "runs": route_runs[route],
                },
            )
        finally:
            model = None
            tokenizer = None
            release_model_execution(device=device)
    try:
        evidence = validate_direct_logit_repetitions(
            route_runs,
            repeat_count=int(config["choice_score"]["micro_smoke_repeat_count"]),
            score_repeat_tolerance=float(config["choice_score"]["score_repeat_tolerance"]),
        )
    except DirectLogitScorerError as error:
        raise ControllerV2RuntimeError(f"direct-logit micro-smoke failed: {error}") from error
    _atomic_json(run_dir / "direct_logit_micro_smoke.json", evidence)
    return evidence


def run_direct_logit_full_choice(
    config: Mapping[str, Any],
    *,
    manifest_path: Path,
    roles: Sequence[str],
    run_dir: Path,
    device: str,
) -> dict[str, dict[str, Any]]:  # pragma: no cover - GPU only
    """Write complete B0/B1 direct-logit predictions, releasing each route."""

    artifacts: dict[str, dict[str, Any]] = {}
    for route in ("B0", "B1"):
        model = tokenizer = None
        try:
            model, tokenizer, encode, _ = load_direct_logit_route(
                config, route, device=device
            )
            artifacts[route] = write_prediction_artifact(
                run_dir / f"{route.lower()}_choice_predictions.jsonl",
                run_direct_choice_rows(
                    iter_prompt_rows(manifest_path, roles),
                    model=model,
                    tokenize=encode,
                    require_expected_qwen_ids=True,
                ),
            )
        finally:
            model = None
            tokenizer = None
            release_model_execution(device=device)
    return artifacts


def _apply_vllm_v1_generation_policy(
    engine_args: Any,
    usage_context: Any,
    model_config: Any,
    *,
    original,
) -> None:
    """Restore the frozen no-chunked policy after vLLM 0.11 V1 defaults.

    vLLM 0.11's V1 ``_set_default_args`` unconditionally enables chunked
    prefill for generation, even when the public LLM argument is false. V0 is
    unavailable on the pinned CUDA backend. This narrow compatibility shim
    preserves every other upstream default and fixes the two coupled scheduler
    fields before VllmConfig construction.
    """

    original(engine_args, usage_context, model_config)
    if engine_args.enable_prefix_caching is not False:
        raise ControllerV2RuntimeError("vLLM V1 prefix caching was not disabled")
    engine_args.enable_chunked_prefill = False
    engine_args.max_num_batched_tokens = int(model_config.max_model_len)


def _make_vllm_generation_backend(config: Mapping[str, Any]):  # pragma: no cover - GPU only
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.lora.request import LoRARequest

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model["path"]), revision=str(model["tokenizer_revision"]), local_files_only=True
    )
    original_defaults = EngineArgs._set_default_args

    def frozen_generation_defaults(engine_args, usage_context, model_config):
        _apply_vllm_v1_generation_policy(
            engine_args, usage_context, model_config, original=original_defaults
        )

    EngineArgs._set_default_args = frozen_generation_defaults
    try:
        engine = LLM(
            model=str(model["path"]), dtype="bfloat16", enable_lora=True,
            max_lora_rank=16, max_model_len=1536, seed=42,
            enable_prefix_caching=False, enforce_eager=True,
            enable_chunked_prefill=False,
        )
    finally:
        EngineArgs._set_default_args = original_defaults
    vllm_config = engine.llm_engine.vllm_config
    actual_runtime = {
        "vllm_use_v1": True,
        "v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "1",
        "enable_prefix_caching": bool(vllm_config.cache_config.enable_prefix_caching),
        "enable_chunked_prefill": bool(vllm_config.scheduler_config.enable_chunked_prefill),
        "max_num_batched_tokens": int(vllm_config.scheduler_config.max_num_batched_tokens),
        "max_model_len": int(vllm_config.scheduler_config.max_model_len),
        "enforce_eager": True,
    }
    if actual_runtime != {
        "vllm_use_v1": True,
        "v1_multiprocessing": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_batched_tokens": 1536,
        "max_model_len": 1536,
        "enforce_eager": True,
    }:
        raise ControllerV2RuntimeError(
            f"vLLM generation runtime differs from frozen config: {actual_runtime}"
        )
    engine._ca_opd_runtime_config = actual_runtime
    medical_lora = LoRARequest("medical_teacher", 1, str(model["medical_lora_path"]))

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def generator(lora_request):
        def generate(prompts: list[str], max_tokens: int) -> list[dict[str, Any]]:
            outputs = engine.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    temperature=0.0, max_tokens=max_tokens, seed=42
                ),
                lora_request=lora_request,
                use_tqdm=False,
            )
            return [
                {
                    "text": str(item.outputs[0].text),
                    "finish_reason": str(item.outputs[0].finish_reason),
                    "stop_reason": item.outputs[0].stop_reason,
                    "token_ids": list(item.outputs[0].token_ids),
                    "eos_observed": bool(
                        item.outputs[0].token_ids
                        and int(item.outputs[0].token_ids[-1]) == int(tokenizer.eos_token_id)
                    ),
                }
                for item in outputs
            ]
        return generate

    return engine, encode, generator(None), generator(medical_lora)


def _release_vllm_engine(engine: Any) -> None:
    """Terminate the pinned vLLM 0.11 V1 EngineCoreClient boundary."""

    llm_engine = getattr(engine, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        raise ControllerV2RuntimeError(
            "vLLM engine has no pinned engine_core.shutdown boundary"
        )
    shutdown()


def _runtime_fixture_probe(encode, generate0, generate1) -> dict[str, Any]:  # pragma: no cover - GPU only
    """Record non-thinking rendering/runtime behavior without a real question."""

    row = {
        "sample_id": "controller-v2-runtime-fixture",
        "target_role": "general_controller_dev",
        "domain": "general",
        "subject": "runtime_fixture",
        "question": "请选择唯一正确的占位选项。",
        "options": ["占位甲", "占位乙", "占位丙", "占位丁"],
    }
    prompt = build_generative_prompt(row)
    prompt_ids = list(encode(prompt))

    def summarize(output: Mapping[str, Any]) -> dict[str, Any]:
        text = str(output.get("text") or "")
        return {
            "finish_reason": str(output.get("finish_reason") or "unknown"),
            "generated_token_count": len(output.get("token_ids") or []),
            "generated_thinking_tag": "<think>" in text or "</think>" in text,
            "raw_output_persisted": False,
        }

    return {
        "fixture_contains_real_question": False,
        "tokenizer_revision": BASE_MODEL_REVISION,
        "rendering_api": "frozen_qwen3_chatml_with_no_think_instruction",
        "enable_thinking": False,
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tail_token_ids": prompt_ids[-16:],
        "prompt_contains_thinking_tag": "<think>" in prompt or "</think>" in prompt,
        "B0": summarize(generate0([prompt], 64)[0]),
        "B1": summarize(generate1([prompt], 64)[0]),
    }


def run_all_gpu(config_path: str | Path) -> dict[str, Any]:  # pragma: no cover - GPU only
    """Direct-logit choice plus vLLM generation under one fail-closed run."""

    from src.eval.controller_v2 import freeze_generation_limit
    config = load_controller_v2_config(config_path)
    initial_gate = controller_v2_gpu_preflight(config_path)
    lora_identity = verify_medical_lora_identity(config)
    started = datetime.now(timezone.utc)
    output_root = Path(str(config["execution"]["output_root"]))
    run_dir = output_root / str(config["execution"]["run_id"])
    if run_dir.exists():
        raise ControllerV2RuntimeError(f"Controller v2 output requires a new run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = _resolve(config["data"]["manifest_path"])
    roles = list(config["data"]["roles"])
    smoke_path = _resolve(config["length_smoke"]["manifest_path"])
    smoke_ids = _load_smoke_ids(smoke_path)
    choice_smoke_ids = _load_choice_smoke_ids(smoke_path)
    engine = None
    generate0 = generate1 = None
    try:
        # Phase 2: the formal choice backend must pass three repeats on both
        # routes before vLLM length smoke or any full controller work begins.
        choice_smoke = run_direct_logit_micro_smoke(
            config,
            manifest_path=manifest_path,
            roles=roles,
            selected_ids=choice_smoke_ids,
            run_dir=run_dir,
        )

        # Phase 3/4: vLLM is generation-only. Freeze 512/1024 using truncation
        # evidence. The pinned in-process V1 engine retains its CUDA allocation
        # after shutdown, so keep it alive on GPU0 and run formal direct logits
        # on the independently frozen GPU1 device before continuing generation.
        engine, encode, generate0, generate1 = _make_vllm_generation_backend(config)
        runtime_fixture = _runtime_fixture_probe(encode, generate0, generate1)
        runtime_fixture["vllm_runtime_config"] = engine._ca_opd_runtime_config
        _atomic_json(run_dir / "runtime_fixture_probe.json", runtime_fixture)
        smoke0 = list(run_generation_rows(
            iter_prompt_rows(manifest_path, roles, selected_ids=smoke_ids),
            generate=generate0, max_new_tokens=512,
        ))
        smoke1 = list(run_generation_rows(
            iter_prompt_rows(manifest_path, roles, selected_ids=smoke_ids),
            generate=generate1, max_new_tokens=512,
        ))
        _write_jsonl_allow_response(run_dir / "b0_length_smoke.jsonl", smoke0)
        _write_jsonl_allow_response(run_dir / "b1_length_smoke.jsonl", smoke1)
        length = freeze_generation_limit(smoke0, smoke1)
        length_payload = {
            **length.as_dict(),
            "protocol_version": PROTOCOL_VERSION,
            "status": "frozen_before_full_evaluation",
        }
        length_path = run_dir / "length_calibration.json"
        _atomic_json(length_path, length_payload)
        controller_v2_gpu_preflight(
            config_path,
            evaluation_phase="full",
            length_decision=length_path,
            result_output_dir=run_dir / "full-evaluation-gate",
            prevalidated_label_attestation=initial_gate["label_artifact_attestation"],
        )
        # Phase 5: full formal choice-score uses Transformers direct logits only
        # on GPU1 while the already-validated generation engine remains on GPU0.
        choice_paths = run_direct_logit_full_choice(
            config,
            manifest_path=manifest_path,
            roles=roles,
            run_dir=run_dir,
            device=str(config["choice_score"]["runtime"]["full_evaluation_device"]),
        )

        # Phase 6: reuse the same generation-only vLLM engine for full B0/B1.
        generation_paths: dict[str, dict[str, Any]] = {}
        generation_failure: str | None = None
        try:
            for name, generation_fn in (
                ("B0", generate0), ("B1", generate1)
            ):
                generation_path = run_dir / f"{name.lower()}_generation_predictions.jsonl"
                generation_paths[name] = _write_jsonl_allow_response(
                    generation_path,
                    run_generation_rows(
                        iter_prompt_rows(manifest_path, roles), generate=generation_fn,
                        max_new_tokens=length.max_new_tokens,
                    ),
                )
        except BaseException as error:
            generation_failure = f"{type(error).__name__}: {error}"
            _atomic_json(run_dir / "generation_failure.json", {
                "status": "failed_after_choice_completed",
                "failure_reason": generation_failure,
                "choice_results_preserved": True,
                "teacher_generation_contract_ready": False,
                "final_authorized": False,
            })
        finally:
            # Release the model execution boundary before opening any label artifact.
            generate0 = generate1 = None
            if engine is not None:
                _release_vllm_engine(engine)
                del engine
                engine = None
            release_model_execution()

        labels = _label_map(manifest_path, roles)
        choice_scored = {
            name: _compact_score(_iter_jsonl(Path(artifact["path"])), labels)
            for name, artifact in choice_paths.items()
        }
        generation_scored = None if generation_failure else {
            name: _compact_score(_iter_jsonl(Path(artifact["path"])), labels)
            for name, artifact in generation_paths.items()
        }
        tracks = summarize_controller_tracks(
            choice_scored,
            generation_scored=generation_scored,
            artifact_valid=bool(lora_identity["artifact_valid"]),
            generation_failure=generation_failure,
        )
        summary = {
            "run_id": config["execution"]["run_id"],
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": config["protocol_sha256"],
            "status": (
                "completed" if generation_failure is None
                else "choice_completed_generation_failed"
            ),
            "base_model_revision": BASE_MODEL_REVISION,
            "medical_lora_sha256": MEDICAL_LORA_SHA256,
            "controller_manifest_sha256": config["data"]["manifest_sha256"],
            "length_decision": length_payload,
            "length_calibration_sha256": _sha256(length_path),
            "runtime_smoke": {
                "choice": choice_smoke,
                "nonthinking_fixture": runtime_fixture,
            },
            "choice_backend": DIRECT_LOGIT_BACKEND,
            "vllm_choice_backend_status": VLLM_CHOICE_BACKEND_STATUS,
            "choice_metrics": tracks["choice_metrics"],
            "generation_metrics": tracks["generation_metrics"],
            "choice_paired": tracks["choice_paired"],
            "generation_paired": tracks["generation_paired"],
            "generation_status": tracks["generation_status"],
            "generation_failure": tracks["generation_failure"],
            "teacher_readiness": tracks["teacher_readiness"],
            "prediction_artifacts": {"choice": choice_paths, "generation": generation_paths},
            "final_authorized": False,
            "started_at": started.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "actual_cost_cny": None,
            "git_sha": initial_gate["git_sha"],
            "live_price_cny_per_hour": initial_gate["live_price_cny_per_hour"],
            "formal_preflight": {
                "length_smoke": "PASS",
                "full_evaluation": "PASS",
            },
        }
        write_standard_run_artifacts(run_dir, config=config, summary=summary)
        validate_standard_run_artifacts(run_dir)
        return summary
    except BaseException as error:
        _atomic_json(run_dir / "failure.json", {
            "status": "failed", "failure_reason": f"{type(error).__name__}: {error}",
            "final_authorized": False,
        })
        raise
    finally:
        if engine is not None:
            try:
                _release_vllm_engine(engine)
            except Exception:
                # Preserve the original execution failure; the outer launcher
                # still has an owned process group and PID-scoped cleanup fallback.
                pass
            del engine
        release_model_execution()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Controller Protocol v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "gpu-preflight", "run-all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    validate = subparsers.add_parser("validate-artifacts")
    validate.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-artifacts":
        print(json.dumps(validate_standard_run_artifacts(args.run_dir), sort_keys=True))
        return 0
    config = load_controller_v2_config(args.config)
    if args.command == "preflight":
        stage_gate = controller_v2_cpu_preflight(args.config)
        print(json.dumps({
            "status": "ready_waiting_for_gpu_direct_logit",
            "protocol_version": PROTOCOL_VERSION,
            "choice_backend": DIRECT_LOGIT_BACKEND,
            "run_id": config["execution"]["run_id"],
            "cpu_dry_run": True,
            "model_weights_loaded": False,
            "gpu_runtime_verified": False,
            "final_authorized": False,
            "stage_aware_preflight": stage_gate,
        }, sort_keys=True))
        return 0
    if args.command == "gpu-preflight":
        print(json.dumps(controller_v2_gpu_preflight(args.config), sort_keys=True))
        return 0
    assert_gpu_execution_authorized(config)
    result = run_all_gpu(args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
