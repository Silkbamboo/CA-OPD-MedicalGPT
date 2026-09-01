"""Legacy vLLM shared-backbone Base/Medical diagnostic scoring service.

The production topology owns exactly one vLLM engine on GPU1. ``base`` requests
use that backbone without an adapter; ``medical`` requests attach one immutable
Medical LoRA. Both routes score the student's exact ``prompt + completion``
tokens via prompt logprobs and never substitute a teacher-generated completion.

P3.2 observed Medical-LoRA prompt-logprob drift above the frozen tolerance.
Consequently this module is diagnostic-only; the P4.0 formal reference is the
Transformers same-trajectory scorer.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from src.opd.core import OPDBatch

BASE_TEACHER = "base"
MEDICAL_TEACHER = "medical"
FORMAL_ENABLED = False
DIAGNOSTIC_ONLY = True


class TeacherServiceError(RuntimeError):
    """Base error for route, engine and response contract violations."""


class TeacherAlignmentError(TeacherServiceError):
    """Teacher response does not score the exact requested token sequence."""


def _fingerprint(sequences: Sequence[Sequence[int]], prompt_lengths: Sequence[int]) -> str:
    payload = {"token_ids": [list(s) for s in sequences], "prompt_lengths": list(prompt_lengths)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


@dataclass(frozen=True)
class TeacherServiceConfig:
    medical_adapter_path: str
    base_teacher_id: str = BASE_TEACHER
    medical_teacher_id: str = MEDICAL_TEACHER
    medical_adapter_name: str = "medical_teacher"
    medical_adapter_id: int = 1
    temperature: float = 1.0
    max_tokens: int = 1
    prompt_logprobs: int = 0

    def __post_init__(self) -> None:
        if not self.medical_adapter_path:
            raise ValueError("medical_adapter_path must be non-empty")
        if self.base_teacher_id == self.medical_teacher_id:
            raise ValueError("base and medical teacher IDs must differ")
        if self.medical_adapter_id <= 0:
            raise ValueError("medical_adapter_id must be > 0")
        # These are semantic invariants of teacher scoring, not tunable decode
        # settings. A different value could silently generate/score another path.
        if self.temperature != 1.0:
            raise ValueError("teacher scoring temperature must be exactly 1.0")
        if self.max_tokens != 1:
            raise ValueError("teacher scoring requires max_tokens=1 (prefill/logprobs only)")
        if self.prompt_logprobs != 0:
            raise ValueError("prompt_logprobs must be 0 to request each actual prompt token")

    @property
    def teacher_ids(self) -> Tuple[str, str]:
        return self.base_teacher_id, self.medical_teacher_id


@dataclass(frozen=True)
class TeacherScoreRequest:
    request_id: str
    teacher_id: str
    token_ids: Tuple[Tuple[int, ...], ...]
    prompt_lengths: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.token_ids:
            raise ValueError("teacher request cannot be empty")
        if len(self.token_ids) != len(self.prompt_lengths):
            raise ValueError("token_ids and prompt_lengths must have the same batch size")
        for i, (tokens, prompt_len) in enumerate(zip(self.token_ids, self.prompt_lengths)):
            if len(tokens) < 2:
                raise ValueError(f"sequence {i} needs at least two tokens")
            if not 1 <= prompt_len < len(tokens):
                raise ValueError(
                    f"sequence {i}: prompt_length={prompt_len} must leave at least one completion token"
                )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.token_ids, self.prompt_lengths)

    @property
    def num_input_tokens(self) -> int:
        return sum(len(row) for row in self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return sum(len(row) - prompt for row, prompt in zip(self.token_ids, self.prompt_lengths))

    @classmethod
    def from_opd_batch(cls, request_id: str, teacher_id: str, batch: OPDBatch) -> "TeacherScoreRequest":
        sequences: List[Tuple[int, ...]] = []
        prompt_lengths: List[int] = []
        for row in range(batch.batch_size):
            mask = batch.attention_mask[row].detach().cpu().tolist()
            length = int(sum(mask))
            if mask != [1] * length + [0] * (batch.seq_len - length):
                raise TeacherAlignmentError(f"row {row}: attention mask must be contiguous right padding")
            tokens = tuple(int(v) for v in batch.input_ids[row, :length].detach().cpu().tolist())
            sequences.append(tokens)
            prompt_lengths.append(int(batch.prompt_lengths[row].item()))
        return cls(request_id, teacher_id, tuple(sequences), tuple(prompt_lengths))


@dataclass(frozen=True)
class TeacherScoreResponse:
    request_id: str
    teacher_id: str
    request_fingerprint: str
    token_ids: Tuple[Tuple[int, ...], ...]
    # One logprob for each autoregressive target token: len(sequence) - 1.
    token_logprobs: Tuple[Tuple[float, ...], ...]
    elapsed_seconds: float
    adapter_applied: bool

    def validate_against(self, request: TeacherScoreRequest) -> None:
        if self.request_id != request.request_id:
            raise TeacherAlignmentError("response request_id differs from request")
        if self.teacher_id != request.teacher_id:
            raise TeacherAlignmentError("response teacher_id differs from request")
        if self.request_fingerprint != request.fingerprint:
            raise TeacherAlignmentError("response fingerprint differs from requested tokens")
        if self.token_ids != request.token_ids:
            raise TeacherAlignmentError("teacher returned token IDs different from student trajectory")
        if len(self.token_logprobs) != len(request.token_ids):
            raise TeacherAlignmentError("teacher response batch size mismatch")
        for i, (tokens, scores) in enumerate(zip(request.token_ids, self.token_logprobs)):
            if len(scores) != len(tokens) - 1:
                raise TeacherAlignmentError(
                    f"row {i}: got {len(scores)} logprobs for {len(tokens) - 1} targets"
                )

    def to_padded_tensor(self, request: TeacherScoreRequest, batch: OPDBatch):
        """Return detached ``[B,T-1]`` scores aligned to an OPDBatch."""
        self.validate_against(request)
        if request != TeacherScoreRequest.from_opd_batch(request.request_id, request.teacher_id, batch):
            raise TeacherAlignmentError("request no longer matches the supplied OPDBatch")
        import torch

        scores = torch.zeros((batch.batch_size, batch.seq_len - 1), dtype=torch.float32, device=batch.input_ids.device)
        for row, values in enumerate(self.token_logprobs):
            scores[row, : len(values)] = torch.tensor(values, dtype=torch.float32, device=scores.device)
        return scores.detach()


class PromptLogprobEngine(Protocol):
    """Subset of the vLLM ``LLM.generate`` API used by this service."""

    def generate(self, *args: Any, **kwargs: Any) -> Sequence[Any]: ...


SamplingParamsFactory = Callable[..., Any]
LoRARequestFactory = Callable[[str, int, str], Any]


def _vllm_sampling_params(**kwargs: Any) -> Any:  # pragma: no cover - target env
    from vllm import SamplingParams

    return SamplingParams(**kwargs)


def _vllm_lora_request(name: str, adapter_id: int, path: str) -> Any:  # pragma: no cover - target env
    from vllm.lora.request import LoRARequest

    return LoRARequest(name, adapter_id, path)


def _extract_logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, (float, int)):
        return float(value)
    raise TeacherServiceError(f"unsupported vLLM logprob value {type(value).__name__}")


def _parse_vllm_output(output: Any, expected_tokens: Tuple[int, ...], row: int) -> Tuple[float, ...]:
    actual = tuple(int(v) for v in getattr(output, "prompt_token_ids", ()))
    if actual != expected_tokens:
        raise TeacherAlignmentError(
            f"row {row}: vLLM prompt_token_ids differ from request: {actual} != {expected_tokens}"
        )
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_tokens):
        raise TeacherAlignmentError(
            f"row {row}: prompt_logprobs length must equal token length {len(expected_tokens)}"
        )
    scores: List[float] = []
    for position, token_id in enumerate(expected_tokens[1:], start=1):
        entry = prompt_logprobs[position]
        if not isinstance(entry, Mapping) or token_id not in entry:
            raise TeacherAlignmentError(
                f"row {row} position {position}: no logprob for actual token_id={token_id}"
            )
        scores.append(_extract_logprob(entry[token_id]))
    return tuple(scores)


class SharedBackboneTeacherService:
    """Route Base/Medical scoring through one shared prompt-logprob engine."""

    def __init__(
        self,
        engine: PromptLogprobEngine,
        config: TeacherServiceConfig,
        sampling_params_factory: SamplingParamsFactory = _vllm_sampling_params,
        lora_request_factory: LoRARequestFactory = _vllm_lora_request,
    ) -> None:
        self.engine = engine
        self.config = config
        self._sampling_params_factory = sampling_params_factory
        self._lora_request_factory = lora_request_factory
        self._lock = threading.Lock()
        self._requests = {teacher_id: 0 for teacher_id in config.teacher_ids}
        self._input_tokens = {teacher_id: 0 for teacher_id in config.teacher_ids}
        self._completion_tokens = {teacher_id: 0 for teacher_id in config.teacher_ids}
        self._seconds = {teacher_id: 0.0 for teacher_id in config.teacher_ids}

    def _adapter_for(self, teacher_id: str) -> Any:
        if teacher_id == self.config.base_teacher_id:
            return None
        if teacher_id == self.config.medical_teacher_id:
            return self._lora_request_factory(
                self.config.medical_adapter_name,
                self.config.medical_adapter_id,
                self.config.medical_adapter_path,
            )
        raise TeacherServiceError(
            f"unknown teacher_id {teacher_id!r}; known={sorted(self.config.teacher_ids)}"
        )

    def score(self, request: TeacherScoreRequest) -> TeacherScoreResponse:
        adapter = self._adapter_for(request.teacher_id)
        sampling = self._sampling_params_factory(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt_logprobs=self.config.prompt_logprobs,
        )
        t0 = time.perf_counter()
        outputs = self.engine.generate(
            prompts=[{"prompt_token_ids": list(row)} for row in request.token_ids],
            sampling_params=sampling,
            lora_request=adapter,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - t0
        if len(outputs) != len(request.token_ids):
            raise TeacherAlignmentError(
                f"engine returned {len(outputs)} outputs for batch size {len(request.token_ids)}"
            )
        rows = tuple(
            _parse_vllm_output(output, tokens, row)
            for row, (output, tokens) in enumerate(zip(outputs, request.token_ids))
        )
        response = TeacherScoreResponse(
            request_id=request.request_id,
            teacher_id=request.teacher_id,
            request_fingerprint=request.fingerprint,
            token_ids=request.token_ids,
            token_logprobs=rows,
            elapsed_seconds=elapsed,
            adapter_applied=adapter is not None,
        )
        response.validate_against(request)
        with self._lock:
            self._requests[request.teacher_id] += 1
            self._input_tokens[request.teacher_id] += request.num_input_tokens
            self._completion_tokens[request.teacher_id] += request.num_completion_tokens
            self._seconds[request.teacher_id] += elapsed
        return response

    def score_batch(self, request_id: str, teacher_id: str, batch: OPDBatch):
        request = TeacherScoreRequest.from_opd_batch(request_id, teacher_id, batch)
        return self.score(request).to_padded_tensor(request, batch)

    def metrics(self) -> Dict[str, Any]:
        """Cumulative route metrics; GPU memory/throughput are added on target."""
        with self._lock:
            return {
                "shared_engine_instances": 1,
                "requests": dict(self._requests),
                "input_tokens": dict(self._input_tokens),
                "completion_tokens": dict(self._completion_tokens),
                "seconds": dict(self._seconds),
            }


def build_vllm_service(
    model_path: str,
    medical_adapter_path: str,
    *,
    gpu_memory_utilization: float = 0.80,
    max_lora_rank: int = 32,
    diagnostic_only_ack: bool = False,
    **engine_kwargs: Any,
) -> SharedBackboneTeacherService:  # pragma: no cover - target GPU
    """Construct the legacy engine only after an explicit diagnostic acknowledgement."""
    if not diagnostic_only_ack:
        raise TeacherServiceError("vLLM prompt-logprob service is diagnostic_only")
    from vllm import LLM

    if not Path(medical_adapter_path).exists():
        raise FileNotFoundError(f"Medical LoRA not found: {medical_adapter_path}")
    engine = LLM(
        model=model_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        **engine_kwargs,
    )
    return SharedBackboneTeacherService(
        engine,
        TeacherServiceConfig(medical_adapter_path=medical_adapter_path),
    )
