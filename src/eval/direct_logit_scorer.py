"""Deterministic Transformers direct-logit backend for Controller v2 choice score.

The module is safe to import on a CPU-only host: torch, Transformers and PEFT are
loaded only inside explicitly invoked runtime functions. Formal execution accepts
label-free controller prompt rows and supports only the frozen single-token A--E
candidate path. Gold labels remain outside this boundary.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.eval.controller_v2 import (
    ChoicePrediction,
    ChoiceRequest,
    ControllerV2Error,
    build_choice_request,
)


DIRECT_LOGIT_BACKEND = "transformers_direct_logits"
VLLM_CHOICE_BACKEND_STATUS = "diagnostic_only"
EXPECTED_QWEN3_LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
SCORE_REPEAT_TOLERANCE = 1e-4
_LETTERS = "ABCDE"


class DirectLogitScorerError(RuntimeError):
    """Fail-closed direct-logit configuration or execution violation."""


def direct_logit_model_plan(route: str) -> dict[str, Any]:
    """Return the frozen loader/forward contract without importing model code."""

    if route not in {"B0", "B1"}:
        raise DirectLogitScorerError("direct-logit route must be B0 or B1")
    return {
        "backend": DIRECT_LOGIT_BACKEND,
        "route": route,
        "adapter_route": "none" if route == "B0" else "peft_medical_lora",
        "loader": "AutoModelForCausalLM",
        "peft_loader": None if route == "B0" else "PeftModel.from_pretrained",
        "batch_size": 1,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
        "torch_compile": False,
        "merge_lora": False,
        "model_mode": "eval",
        "autograd": "torch.inference_mode",
        "log_softmax_dtype": "float32",
        "seed": 42,
    }


def apply_deterministic_runtime(*, torch_module=None, numpy_module=None) -> dict[str, Any]:
    """Apply the pre-registered deterministic controls for an authorized GPU run."""

    if torch_module is None:  # pragma: no cover - real GPU/runtime only
        import torch as torch_module
    if numpy_module is None:  # pragma: no cover - real GPU/runtime only
        import numpy as numpy_module

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(42)
    numpy_module.random.seed(42)
    torch_module.manual_seed(42)
    torch_module.cuda.manual_seed_all(42)
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.benchmark = False
    return {
        "seed": 42,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "cuda_device_max_connections": 1,
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "tokenizers_parallelism": False,
    }


def _ordered_candidate_ids(candidate_token_ids: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    try:
        ordered = tuple(
            (label, int(candidate_token_ids[label]))
            for label in sorted(candidate_token_ids, key=_LETTERS.index)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectLogitScorerError("candidate token IDs are invalid") from error
    labels = tuple(label for label, _ in ordered)
    if labels not in (tuple(_LETTERS[:4]), tuple(_LETTERS[:5])):
        raise DirectLogitScorerError("direct logits require the complete 4/5-label candidate set")
    if len({token for _, token in ordered}) != len(ordered) or any(token < 0 for _, token in ordered):
        raise DirectLogitScorerError("candidate token IDs must be unique non-negative integers")
    return ordered


def score_last_prompt_position(
    *,
    sample_id: str,
    logits: Any,
    candidate_token_ids: Mapping[str, int],
    torch_module=None,
) -> ChoicePrediction:
    """Score legal single-token candidates at ``model(x).logits[0, -1]``.

    The complete vocabulary logit vector is converted to float32 before
    log-softmax. Only then are the legal A--D/A--E candidate values extracted.
    """

    if not sample_id:
        raise DirectLogitScorerError("direct-logit sample_id is required")
    ordered = _ordered_candidate_ids(candidate_token_ids)
    if getattr(logits, "ndim", None) != 3 or int(logits.shape[0]) != 1 or int(logits.shape[1]) < 1:
        raise DirectLogitScorerError("model logits must have shape [1, prompt_length, vocabulary]")
    vocabulary = int(logits.shape[2])
    if any(token >= vocabulary for _, token in ordered):
        raise DirectLogitScorerError("candidate token is outside the model vocabulary")
    if torch_module is None:  # pragma: no cover - exercised on an authorized GPU
        import torch as torch_module
    last_logits_fp32 = logits[0, -1, :].float()
    log_probs = torch_module.log_softmax(last_logits_fp32, dim=-1)
    scores = {label: float(log_probs[token].item()) for label, token in ordered}
    if any(not math.isfinite(value) for value in scores.values()):
        raise DirectLogitScorerError("candidate direct-logit scores must be finite")
    if len(set(scores.values())) == 1:
        raise DirectLogitScorerError("candidate direct-logit scores are all identical")
    predicted = max((label for label, _ in ordered), key=lambda label: scores[label])
    return ChoicePrediction(
        sample_id=sample_id,
        predicted_label=predicted,
        candidate_scores=scores,
    )


def _single_token_candidates(request: ChoiceRequest) -> dict[str, int]:
    candidate_ids: dict[str, int] = {}
    for candidate in request.candidates:
        if len(candidate.token_ids) != 1:
            raise DirectLogitScorerError(
                "formal direct-logit choice requires every candidate to be single-token"
            )
        candidate_ids[candidate.label] = int(candidate.token_ids[0])
    _ordered_candidate_ids(candidate_ids)
    return candidate_ids


def verify_qwen3_candidate_token_ids(request: ChoiceRequest) -> dict[str, int]:
    """Recompute and compare the fixed-revision A--E token IDs at runtime."""

    actual = _single_token_candidates(request)
    expected = {label: EXPECTED_QWEN3_LABEL_TOKEN_IDS[label] for label in request.labels}
    if actual != expected:
        raise DirectLogitScorerError(
            f"Qwen3 candidate token IDs drift: expected {expected}, observed {actual}"
        )
    return actual


def score_direct_request(
    model: Any,
    request: ChoiceRequest,
    *,
    torch_module=None,
) -> dict[str, Any]:
    """Run one label-free, batch-size-one forward for a prepared request."""

    candidate_ids = _single_token_candidates(request)
    if torch_module is None:  # pragma: no cover - real GPU/runtime only
        import torch as torch_module
    model.eval()
    input_ids = torch_module.tensor([list(request.prompt_token_ids)], dtype=torch_module.long)
    attention_mask = torch_module.ones_like(input_ids)
    device = getattr(model, "device", None)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    with torch_module.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    prediction = score_last_prompt_position(
        sample_id=request.sample_id,
        logits=getattr(output, "logits", None),
        candidate_token_ids=candidate_ids,
        torch_module=torch_module,
    )
    return {
        **prediction.as_dict(),
        "target_role": request.target_role,
        "protocol_version": request.protocol_version,
        "metric_track": "choice_score",
        "choice_backend": DIRECT_LOGIT_BACKEND,
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids": list(request.prompt_token_ids),
        "candidate_tokenization": [
            {"label": candidate.label, "token_ids": list(candidate.token_ids)}
            for candidate in request.candidates
        ],
        "labels_opened_during_execution": False,
    }


def run_direct_choice_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    model: Any,
    tokenize,
    torch_module=None,
    require_expected_qwen_ids: bool = True,
) -> Iterator[dict[str, Any]]:
    """Stream formal choice predictions without buffering the controller pool."""

    seen: set[str] = set()
    for row in rows:
        request = build_choice_request(row, tokenize=tokenize)
        if request.sample_id in seen:
            raise DirectLogitScorerError("direct-logit runtime contains a duplicate sample_id")
        seen.add(request.sample_id)
        if require_expected_qwen_ids:
            verify_qwen3_candidate_token_ids(request)
        result = score_direct_request(model, request, torch_module=torch_module)
        yield {
            **result,
            "domain": str(row.get("domain") or ""),
            "subject": str(row.get("subject") or ""),
        }


def load_direct_logit_route(
    config: Mapping[str, Any], route: str, *, device: str = "cuda:0"
):  # pragma: no cover - GPU only
    """Load Base or Base+PEFT on one visible GPU without merging the adapter."""

    if device not in {"cuda:0", "cuda:1"}:
        raise DirectLogitScorerError("direct-logit device must be frozen to cuda:0 or cuda:1")
    plan = {**direct_logit_model_plan(route), "device": device}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    apply_deterministic_runtime(torch_module=torch)
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["path"]),
        revision=str(model_config["tokenizer_revision"]),
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["path"]),
        revision=str(model_config["revision"]),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.config.use_cache = False
    if route == "B1":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(model_config["medical_lora_path"]),
            is_trainable=False,
        )
    model.eval()

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return model, tokenizer, encode, plan


def _index_repetition(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in indexed:
            raise DirectLogitScorerError(f"{name} contains a missing/duplicate sample_id")
        if row.get("labels_opened_during_execution") is not False:
            raise DirectLogitScorerError(f"{name} indicates label access during execution")
        scores = row.get("candidate_scores")
        tokens = row.get("candidate_tokenization")
        prompt_ids = row.get("prompt_token_ids")
        prompt_sha = str(row.get("prompt_sha256") or "")
        if not isinstance(scores, Mapping) or not isinstance(tokens, list):
            raise DirectLogitScorerError(f"{name} direct-logit evidence is incomplete")
        numeric = {str(label): float(score) for label, score in scores.items()}
        token_ids = {
            str(item.get("label") or ""): int(item["token_ids"][0])
            for item in tokens
            if isinstance(item, Mapping)
            and isinstance(item.get("token_ids"), list)
            and len(item["token_ids"]) == 1
        }
        _ordered_candidate_ids(token_ids)
        if set(numeric) != set(token_ids):
            raise DirectLogitScorerError(
                f"{name} score labels differ from candidate tokenization"
            )
        if not numeric or any(not math.isfinite(score) for score in numeric.values()):
            raise DirectLogitScorerError(f"{name} direct-logit scores are non-finite")
        if len(set(numeric.values())) == 1:
            raise DirectLogitScorerError(f"{name} direct-logit scores are all identical")
        if not isinstance(prompt_ids, list) or not prompt_ids or len(prompt_sha) != 64:
            raise DirectLogitScorerError(f"{name} prompt identity is incomplete")
        predicted = str(row.get("predicted_label") or "")
        if predicted not in numeric or predicted != max(numeric, key=lambda label: numeric[label]):
            raise DirectLogitScorerError(f"{name} prediction is not the legal score argmax")
        indexed[sample_id] = row
    if not indexed:
        raise DirectLogitScorerError(f"{name} cannot be empty")
    return indexed


def validate_direct_logit_repetitions(
    route_runs: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    *,
    repeat_count: int = 3,
    score_repeat_tolerance: float = SCORE_REPEAT_TOLERANCE,
) -> dict[str, Any]:
    """Validate the pre-registered three-repeat B0/B1 four-ID GPU micro-smoke."""

    if set(route_runs) != {"B0", "B1"}:
        raise DirectLogitScorerError("direct-logit smoke requires B0 and B1 routes")
    indexed: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for route in ("B0", "B1"):
        runs = route_runs[route]
        if len(runs) != repeat_count:
            raise DirectLogitScorerError(f"{route} requires exactly {repeat_count} repetitions")
        indexed[route] = [
            _index_repetition(rows, name=f"{route} repeat {index + 1}")
            for index, rows in enumerate(runs)
        ]
        reference_ids = set(indexed[route][0])
        if any(set(run) != reference_ids for run in indexed[route][1:]):
            raise DirectLogitScorerError(f"{route} repeat sample sets differ")
    if set(indexed["B0"][0]) != set(indexed["B1"][0]):
        raise DirectLogitScorerError("B0/B1 direct-logit smoke sample sets differ")
    if len(indexed["B0"][0]) != 4:
        raise DirectLogitScorerError("direct-logit smoke requires exactly four frozen samples")
    for sample_id in sorted(indexed["B0"][0]):
        base = indexed["B0"][0][sample_id]
        medical = indexed["B1"][0][sample_id]
        if (
            base["prompt_sha256"] != medical["prompt_sha256"]
            or base["prompt_token_ids"] != medical["prompt_token_ids"]
            or base["candidate_tokenization"] != medical["candidate_tokenization"]
        ):
            raise DirectLogitScorerError(
                "B0/B1 cross-route prompt/candidate identity differs"
            )

    maximum_delta = 0.0
    route_evidence: dict[str, Any] = {}
    for route in ("B0", "B1"):
        reference = indexed[route][0]
        for run in indexed[route][1:]:
            for sample_id in sorted(reference):
                original = reference[sample_id]
                repeat = run[sample_id]
                if (
                    original["prompt_sha256"] != repeat["prompt_sha256"]
                    or original["prompt_token_ids"] != repeat["prompt_token_ids"]
                    or original["candidate_tokenization"] != repeat["candidate_tokenization"]
                ):
                    raise DirectLogitScorerError(
                        f"{route} prompt/candidate tokenization is not deterministic"
                    )
                first_scores = {key: float(value) for key, value in original["candidate_scores"].items()}
                next_scores = {key: float(value) for key, value in repeat["candidate_scores"].items()}
                first_rank = sorted(first_scores, key=lambda label: (-first_scores[label], label))
                next_rank = sorted(next_scores, key=lambda label: (-next_scores[label], label))
                if original["predicted_label"] != repeat["predicted_label"] or first_rank != next_rank:
                    raise DirectLogitScorerError(
                        f"{route} direct-logit prediction/candidate ordering changed"
                    )
                delta = max(abs(first_scores[label] - next_scores[label]) for label in first_scores)
                maximum_delta = max(maximum_delta, delta)
                if delta > score_repeat_tolerance:
                    raise DirectLogitScorerError(
                        f"{route} direct-logit score repeat delta {delta:.10g} exceeds "
                        f"{score_repeat_tolerance:.10g}"
                    )
        route_evidence[route] = {
            "sample_count": len(reference),
            "repeat_count": repeat_count,
            "predictions": {
                sample_id: reference[sample_id]["predicted_label"]
                for sample_id in sorted(reference)
            },
        }
    return {
        "status": "PASS",
        "choice_backend": DIRECT_LOGIT_BACKEND,
        "repeat_count": repeat_count,
        "sample_count": len(indexed["B0"][0]),
        "score_repeat_tolerance": score_repeat_tolerance,
        "max_abs_score_delta": maximum_delta,
        "candidate_ordering_deterministic": True,
        "labels_opened_during_execution": False,
        "routes": route_evidence,
    }


__all__ = [
    "DIRECT_LOGIT_BACKEND",
    "VLLM_CHOICE_BACKEND_STATUS",
    "EXPECTED_QWEN3_LABEL_TOKEN_IDS",
    "SCORE_REPEAT_TOLERANCE",
    "DirectLogitScorerError",
    "apply_deterministic_runtime",
    "direct_logit_model_plan",
    "load_direct_logit_route",
    "run_direct_choice_rows",
    "score_direct_request",
    "score_last_prompt_position",
    "validate_direct_logit_repetitions",
    "verify_qwen3_candidate_token_ids",
]
