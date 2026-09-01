"""Narrow, fail-closed Teacher scorer for a Student's frozen trajectory.

The CPU test path accepts an injected toy model.  The real Qwen/PEFT loader is
deliberately kept outside this module and may only be called by the separately
authorized GPU calibration launcher.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch


MEDICAL_ADAPTER_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
VLLM_TRAJECTORY_BACKEND_POLICY = {
    "backend": "vllm_prompt_logprobs",
    "formal_enabled": False,
    "diagnostic_only": True,
}
_ROUTES = {"base", "medical"}
_SUPERVISION_FIELDS = {
    "answer", "answer_idx", "label", "labels", "solution", "reasoning",
    "reference_answer", "ground_truth", "reward", "score",
}
_ALLOWED_REQUEST_FIELDS = {
    "request_id", "route", "prompt_ids", "response_ids", "attention_mask",
    "eos_token_id", "finish_reason", "truncated", "source_role", "metadata",
}


class TrajectoryContractError(RuntimeError):
    """Raised instead of returning fabricated or misaligned Teacher scores."""


def _find_supervision(value: Any, path: str = "request") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            child = f"{path}.{key}"
            if normalized in _SUPERVISION_FIELDS:
                return child
            found = _find_supervision(item, child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_supervision(item, f"{path}[{index}]")
            if found:
                return found
    return None


@dataclass(frozen=True)
class TrajectoryScoreRequest:
    request_id: str
    route: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    eos_token_id: int | None = None
    finish_reason: str | None = None
    truncated: bool = False
    source_role: str = "opd_scorer_calibration"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", tuple(int(x) for x in self.prompt_ids))
        object.__setattr__(self, "response_ids", tuple(int(x) for x in self.response_ids))
        object.__setattr__(self, "attention_mask", tuple(int(x) for x in self.attention_mask))
        if not self.request_id:
            raise TrajectoryContractError("request_id is required")
        if self.route not in _ROUTES:
            raise TrajectoryContractError(f"unknown Teacher route: {self.route!r}")
        if not self.prompt_ids or not self.response_ids:
            raise TrajectoryContractError("prompt_ids and response_ids must be non-empty")
        expected = len(self.prompt_ids) + len(self.response_ids)
        if len(self.attention_mask) != expected or any(value != 1 for value in self.attention_mask):
            raise TrajectoryContractError(
                "attention_mask must contain one for every unpadded prompt/response token"
            )
        if any(token < 0 for token in (*self.prompt_ids, *self.response_ids)):
            raise TrajectoryContractError("token IDs must be non-negative")
        if self.eos_token_id is not None:
            eos_positions = [
                index for index, token in enumerate(self.response_ids)
                if token == int(self.eos_token_id)
            ]
            if self.truncated and eos_positions:
                raise TrajectoryContractError("truncated trajectory must not contain a synthetic EOS")
            if eos_positions and eos_positions[-1] != len(self.response_ids) - 1:
                raise TrajectoryContractError("EOS must be the last response token when present")
        if "final" in self.source_role.lower():
            raise TrajectoryContractError("final role is forbidden in Teacher scoring")
        found = _find_supervision(self.metadata)
        if found:
            raise TrajectoryContractError(f"supervision field is forbidden: {found}")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrajectoryScoreRequest":
        found = _find_supervision(payload)
        if found:
            raise TrajectoryContractError(f"supervision field is forbidden: {found}")
        unknown = set(payload) - _ALLOWED_REQUEST_FIELDS
        if unknown:
            raise TrajectoryContractError(f"unknown trajectory request fields: {sorted(unknown)}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class TeacherScoreResult:
    request_id: str
    route: str
    model_id: str
    model_revision: str
    adapter_sha: str | None
    tokenizer_revision: str
    prompt_length: int
    response_length: int
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    response_mask: tuple[int, ...]
    action_logit_positions: tuple[int, ...]
    eos_position: int | None
    finite: bool
    backend: str
    precision: str
    elapsed_seconds: float
    error: str | None = None


class SharedBackboneRoutes:
    """Make the active Base/Medical route explicit on one PEFT backbone."""

    def __init__(
        self,
        *,
        model: Any,
        medical_adapter_name: str,
        medical_adapter_sha256: str,
    ) -> None:
        if medical_adapter_sha256 != MEDICAL_ADAPTER_SHA256:
            raise TrajectoryContractError("Medical adapter SHA differs from the frozen Teacher")
        self.model = model
        self.medical_adapter_name = medical_adapter_name
        self.medical_adapter_sha256 = medical_adapter_sha256

    @contextmanager
    def activate(self, route: str) -> Iterator[str | None]:
        if route == "base":
            disabled = getattr(self.model, "disable_adapter", None)
            context = disabled() if callable(disabled) else nullcontext()
            with context:
                yield None
            return
        if route == "medical":
            set_adapter = getattr(self.model, "set_adapter", None)
            if not callable(set_adapter):
                raise TrajectoryContractError("Medical route requires a PEFT adapter-aware model")
            set_adapter(self.medical_adapter_name)
            yield self.medical_adapter_sha256
            return
        raise TrajectoryContractError(f"unknown Teacher route: {route!r}")


class TransformersTrajectoryLogprobScorer:
    """Score actual response tokens with causal, locally FP32 log-softmax.

    The model is injected.  This prevents CPU preflight from resolving a model
    path through ``from_pretrained`` and gives tests an exact toy boundary.
    """

    backend = "transformers_direct_trajectory_logits"
    cpu_contract_verified = True
    gpu_runtime_verified = False

    def __init__(
        self,
        *,
        model: Any,
        routes: SharedBackboneRoutes,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        logprob_chunk_tokens: int = 64,
    ) -> None:
        if logprob_chunk_tokens < 1:
            raise TrajectoryContractError("logprob_chunk_tokens must be positive")
        self.model = model
        self.routes = routes
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.logprob_chunk_tokens = int(logprob_chunk_tokens)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score(self, request: TrajectoryScoreRequest) -> TeacherScoreResult:
        return self.score_batch((request,), maximum_batch_size=1, length_bucket_width=1)[0]

    def score_batch(
        self,
        requests: Sequence[TrajectoryScoreRequest],
        *,
        maximum_batch_size: int = 2,
        length_bucket_width: int = 128,
    ) -> list[TeacherScoreResult]:
        """Score small same-route length buckets while preserving input order."""

        if maximum_batch_size < 1 or length_bucket_width < 1:
            raise TrajectoryContractError("batch size and length bucket width must be positive")
        if not requests:
            raise TrajectoryContractError("score_batch requires at least one trajectory")
        indexed = []
        for index, request in enumerate(requests):
            if not isinstance(request, TrajectoryScoreRequest):
                raise TrajectoryContractError(
                    "score_batch accepts only validated TrajectoryScoreRequest values"
                )
            length = len(request.prompt_ids) + len(request.response_ids)
            indexed.append((request.route, length // length_bucket_width, index, request))
        indexed.sort(key=lambda item: (item[0], item[1], item[2]))
        results: list[TeacherScoreResult | None] = [None] * len(requests)
        cursor = 0
        while cursor < len(indexed):
            route, bucket = indexed[cursor][0], indexed[cursor][1]
            group = []
            while (
                cursor < len(indexed)
                and indexed[cursor][0] == route
                and indexed[cursor][1] == bucket
                and len(group) < maximum_batch_size
            ):
                group.append(indexed[cursor])
                cursor += 1
            scored = self._score_same_route([item[3] for item in group])
            for item, result in zip(group, scored, strict=True):
                results[item[2]] = result
        if any(result is None for result in results):
            raise TrajectoryContractError("length-bucket scorer lost a trajectory")
        return [result for result in results if result is not None]

    def _score_same_route(
        self, requests: Sequence[TrajectoryScoreRequest]
    ) -> list[TeacherScoreResult]:
        started = time.perf_counter()
        route = requests[0].route
        if any(request.route != route for request in requests):
            raise TrajectoryContractError("one forward may not mix Base and Medical routes")
        combined = [request.prompt_ids + request.response_ids for request in requests]
        maximum = max(len(row) for row in combined)
        input_ids = torch.zeros((len(requests), maximum), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(combined):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, : len(row)] = 1
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = input_ids.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        with torch.inference_mode(), self.routes.activate(route) as adapter_sha:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(output, "logits", None)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[0] != len(requests)
            ):
                raise TrajectoryContractError(
                    "Teacher forward did not return [batch, sequence, vocab] logits"
                )
            if logits.shape[1] < maximum:
                raise TrajectoryContractError("Teacher logits are shorter than the frozen trajectory")
            all_scores: list[list[float]] = []
            for row_index, request in enumerate(requests):
                prompt_length = len(request.prompt_ids)
                response_length = len(request.response_ids)
                scores: list[float] = []
                for offset in range(0, response_length, self.logprob_chunk_tokens):
                    size = min(self.logprob_chunk_tokens, response_length - offset)
                    start = prompt_length - 1 + offset
                    chunk = logits[row_index : row_index + 1, start : start + size, :].float()
                    targets = torch.tensor(
                        request.response_ids[offset : offset + size],
                        dtype=torch.long,
                        device=chunk.device,
                    ).view(1, size, 1)
                    if int(targets.max()) >= int(chunk.shape[-1]):
                        raise TrajectoryContractError("response token is outside the Teacher vocabulary")
                    gathered = torch.log_softmax(chunk, dim=-1).gather(-1, targets).squeeze(-1)
                    scores.extend(float(value) for value in gathered[0].cpu())
                    del chunk, targets, gathered
                all_scores.append(scores)
        elapsed = time.perf_counter() - started
        precision = "model_bfloat16_logsoftmax_float32"
        results = []
        for request, scores in zip(requests, all_scores, strict=True):
            response_length = len(request.response_ids)
            if len(scores) != response_length:
                raise TrajectoryContractError("Teacher token/logprob length mismatch")
            if not all(math.isfinite(value) for value in scores):
                raise TrajectoryContractError("Teacher token logprobs must all be finite")
            eos_position = None
            if request.eos_token_id is not None:
                matches = [
                    index for index, token in enumerate(request.response_ids)
                    if token == request.eos_token_id
                ]
                eos_position = matches[-1] if matches else None
            prompt_length = len(request.prompt_ids)
            results.append(TeacherScoreResult(
                request_id=request.request_id,
                route=request.route,
                model_id=self.model_id,
                model_revision=self.model_revision,
                adapter_sha=adapter_sha,
                tokenizer_revision=self.tokenizer_revision,
                prompt_length=prompt_length,
                response_length=response_length,
                token_ids=request.response_ids,
                token_logprobs=tuple(scores),
                response_mask=(1,) * response_length,
                action_logit_positions=tuple(
                    range(prompt_length - 1, prompt_length + response_length - 1)
                ),
                eos_position=eos_position,
                finite=True,
                backend=self.backend,
                precision=precision,
                elapsed_seconds=elapsed,
            ))
        return results


def score_trajectory(
    scorer: TransformersTrajectoryLogprobScorer,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    attention_mask: Sequence[int],
    route: str,
    request_id: str,
    **metadata: Any,
) -> TeacherScoreResult:
    """Function-shaped public boundary matching the frozen protocol."""

    request = TrajectoryScoreRequest.from_mapping(
        {
            "request_id": request_id,
            "route": route,
            "prompt_ids": tuple(prompt_ids),
            "response_ids": tuple(response_ids),
            "attention_mask": tuple(attention_mask),
            "metadata": metadata,
        }
    )
    return scorer.score(request)
