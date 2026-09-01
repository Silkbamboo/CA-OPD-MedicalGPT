"""P4.8d memory-balanced B2 calibration execution primitives.

This module is CPU-import safe.  CUDA inspection is injected into the
telemetry writer by the formal GPU worker; importing or validating the frozen
contract never imports a model and never probes CUDA.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from src.opd.production_b2_objective_reducer_v2 import (
    canonical_prompt_chunk_loss,
    canonical_token_objective_from_advantage,
)


class MemoryExecutionV1Error(RuntimeError):
    """The package-bound P4.8d execution contract failed closed."""


CHECKPOINT_VERSIONS = (5, 10, 15, 20)
MINIMUM_CANARY_HEADROOM_BYTES = 1024**3
MAX_STEP_END_ALLOCATED_DRIFT_BYTES = 256 * 1024**2
MAX_STEP_END_RESERVED_DRIFT_BYTES = 512 * 1024**2
MAX_MONOTONIC_AUXILIARY_DRIFT_BYTES = 64 * 1024**2

MEMORY_EXECUTION_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "artifact_kind": "p4_8d_b2_memory_execution_contract_v1",
    "selected_response_length": 1024,
    "seed": 42,
    "optimizer_steps": 20,
    "source_batch": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
    "schedule_slot_count": 80,
    "physical_microbatch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "target_logit_chunk_size": 128,
    "student_q_backbone_forwards_per_prompt": 1,
    "full_vocabulary_logits_scope": (
        "one_target_position_chunk_per_physical_microbatch_v1"
    ),
    "reduction_contract": (
        "masked_token_mean_per_trajectory_then_group_mean_then_"
        "prompt_mean_then_prompt_batch_mean_v1"
    ),
    "use_cache": False,
    "generation_use_cache": True,
    "gradient_checkpointing": {
        "enabled": True,
        "use_reentrant": False,
    },
    "resume_checkpoint_versions": list(CHECKPOINT_VERSIONS),
    "checkpoint_versions": list(CHECKPOINT_VERSIONS),
    "forbidden_resume_versions": [1, 2, 3],
    "minimum_canary_headroom_bytes": MINIMUM_CANARY_HEADROOM_BYTES,
    "six_step_drift_gate": {
        "required_steps": [1, 2, 3, 4, 5, 6],
        "max_allocated_growth_bytes": MAX_STEP_END_ALLOCATED_DRIFT_BYTES,
        "max_reserved_growth_bytes": MAX_STEP_END_RESERVED_DRIFT_BYTES,
        "max_monotonic_inactive_or_non_releasable_growth_bytes": (
            MAX_MONOTONIC_AUXILIARY_DRIFT_BYTES
        ),
    },
    "equivalence_tolerance": {
        "scalar_atol": 1e-6,
        "scalar_rtol": 1e-6,
        "gradient_atol": 2e-6,
        "gradient_rtol": 2e-5,
    },
    "allocator_policy": "inherit_default_no_override",
    "pytorch_cuda_alloc_conf": None,
    "formal_b2_automatic_start": False,
}


def _fail(message: str) -> None:
    raise MemoryExecutionV1Error(message)


def _canonical_bytes(value: Any) -> bytes:
    validate_cpu_artifact_value(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MemoryExecutionV1Error(
            f"memory artifact is not canonical JSON: {type(error).__name__}"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_memory_execution_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every frozen execution/science field without probing CUDA."""

    if not isinstance(value, Mapping):
        _fail("memory execution contract is not an object")
    expected = MEMORY_EXECUTION_CONTRACT
    for field in (
        "selected_response_length",
        "seed",
        "optimizer_steps",
        "source_batch",
        "schedule_slot_count",
        "physical_microbatch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "target_logit_chunk_size",
        "student_q_backbone_forwards_per_prompt",
        "full_vocabulary_logits_scope",
        "reduction_contract",
        "use_cache",
        "generation_use_cache",
        "gradient_checkpointing",
        "resume_checkpoint_versions",
        "checkpoint_versions",
        "forbidden_resume_versions",
        "minimum_canary_headroom_bytes",
        "six_step_drift_gate",
        "equivalence_tolerance",
        "allocator_policy",
        "pytorch_cuda_alloc_conf",
        "formal_b2_automatic_start",
    ):
        if value.get(field) != expected[field]:
            _fail(f"memory execution {field} differs from the frozen contract")
    if not (
        value.get("schema_version") == 1
        and value.get("artifact_kind")
        == "p4_8d_b2_memory_execution_contract_v1"
    ):
        _fail("memory execution schema differs")
    return deepcopy(dict(value))


def build_target_chunks(valid_token_count: int, *, chunk_size: int) -> list[tuple[int, int]]:
    if (
        not isinstance(valid_token_count, int)
        or isinstance(valid_token_count, bool)
        or valid_token_count <= 0
    ):
        _fail("valid_token_count must be positive")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        _fail("chunk_size must be positive")
    return [
        (start, min(valid_token_count, start + chunk_size))
        for start in range(0, valid_token_count, chunk_size)
    ]


def assert_prompt_equal_reduction_batch(
    *, prompt_ids: Sequence[str], group_ids: Sequence[str]
) -> None:
    """Prove the frozen row-wise accumulation equals grouped reduction.

    P4.8d has one trajectory for each of four distinct prompts.  Under that
    registered shape, the production trajectory/group/prompt reduction is
    exactly the arithmetic mean of the four per-prompt token means.
    """

    if not (
        len(prompt_ids) == 4
        and len(group_ids) == 4
        and len({str(value) for value in prompt_ids}) == 4
        and all(isinstance(value, str) and value for value in group_ids)
    ):
        _fail(
            "memory execution requires four distinct prompt identities for "
            "prompt-equal reduction"
        )


def target_logprobs_from_selected_logits(logits: Any, target_ids: Any) -> Any:
    """Extract same-token log-probabilities from already position-selected logits."""

    if tuple(logits.shape[:-1]) != tuple(target_ids.shape):
        _fail("selected logits and target token shape differ")
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        _fail("selected logits shape is invalid")
    return logits.float().log_softmax(dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)


def scaled_prompt_chunk_loss(
    *,
    current_logprob: Any,
    old_logprob: Any,
    advantage: Any,
    correction_weight: Any,
    prompt_valid_token_count: int,
    effective_batch_size: int,
    clip_low: float,
    clip_high: float,
) -> Any:
    """Return the exact contribution of one target chunk to a prompt-equal loss.

    p_old, advantage, and correction weights are immutable.  Summing every
    chunk for a prompt gives its valid-token mean divided by the effective
    prompt batch; summing all four prompts is the frozen production reduction.
    """

    shapes = {
        tuple(value.shape)
        for value in (current_logprob, old_logprob, advantage, correction_weight)
    }
    if len(shapes) != 1:
        _fail("microbatch objective tensors have different shapes")
    if prompt_valid_token_count <= 0:
        _fail("prompt_valid_token_count must be positive")
    if effective_batch_size != 4:
        _fail("effective_batch_size differs from four")
    if old_logprob.requires_grad or advantage.requires_grad or correction_weight.requires_grad:
        _fail("p_old, advantage, and correction weight must be frozen")
    # Shape [1, chunk] keeps this path on the exact canonical token formula.
    # The prompt denominator is supplied separately so a short final chunk is
    # never treated as an equal-weight chunk mean.
    mask = current_logprob.new_ones(
        (1, int(current_logprob.numel()))
    ).bool()
    token = canonical_token_objective_from_advantage(
        q_target_logprob=current_logprob.reshape(1, -1),
        p_old_target_logprob=old_logprob.reshape(1, -1),
        raw_advantage=advantage.reshape(1, -1),
        correction_weight=correction_weight.reshape(1, -1),
        valid_mask=mask,
        clip_low=clip_low,
        clip_high=clip_high,
        accumulator_dtype=current_logprob.new_empty(()).float().dtype,
    )
    return canonical_prompt_chunk_loss(
        corrected_selected_objective=token[
            "corrected_selected_objective"
        ].reshape(-1),
        prompt_valid_token_count=prompt_valid_token_count,
        effective_batch_size=effective_batch_size,
    )


def backward_selected_hidden_chunks(
    *,
    selected_hidden_states: Any,
    lm_head: Any,
    target_ids: Any,
    old_logprob: Any,
    advantage: Any,
    correction_weight: Any,
    prompt_valid_token_count: int,
    effective_batch_size: int,
    clip_low: float,
    clip_high: float,
    chunk_size: int,
    chunk_observer: Callable[[str, int, int, int], None] | None = None,
) -> Any:
    """Backprop one backbone graph through bounded LM-head vocabulary chunks.

    The transformer is evaluated exactly once for the physical microbatch.
    Only causal target-position hidden states enter the LM head, and each
    vocabulary-logit chunk is released immediately after its backward call.
    Retaining the backbone graph until the last chunk preserves exact BPTT.
    """

    if not (
        getattr(selected_hidden_states, "ndim", None) == 3
        and selected_hidden_states.shape[0] == 1
        and getattr(target_ids, "ndim", None) == 2
        and target_ids.shape[0] == 1
        and selected_hidden_states.shape[1] == target_ids.shape[1]
        and int(target_ids.shape[1]) == prompt_valid_token_count
    ):
        _fail("selected hidden states and target IDs differ")
    for name, value in (
        ("old_logprob", old_logprob),
        ("advantage", advantage),
        ("correction_weight", correction_weight),
    ):
        if getattr(value, "ndim", None) != 1 or int(value.shape[0]) != prompt_valid_token_count:
            _fail(f"{name} differs from the prompt valid-token count")
    total = None
    chunks = build_target_chunks(prompt_valid_token_count, chunk_size=chunk_size)
    for chunk_index, (start, end) in enumerate(chunks):
        if chunk_observer is not None:
            chunk_observer("before", chunk_index, start, end)
        logits = lm_head(selected_hidden_states[:, start:end, :])
        current = target_logprobs_from_selected_logits(
            logits, target_ids[:, start:end]
        ).reshape(-1)
        loss = scaled_prompt_chunk_loss(
            current_logprob=current,
            old_logprob=old_logprob[start:end],
            advantage=advantage[start:end],
            correction_weight=correction_weight[start:end],
            prompt_valid_token_count=prompt_valid_token_count,
            effective_batch_size=effective_batch_size,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        detached = loss.detach()
        total = detached if total is None else total + detached
        loss.backward(retain_graph=chunk_index + 1 < len(chunks))
        del loss, current, logits
        if chunk_observer is not None:
            chunk_observer("after", chunk_index, start, end)
    if total is None:
        _fail("selected hidden chunk backward produced no loss")
    return total


def torch_minimum(left: Any, right: Any) -> Any:
    """Dispatch to the tensor implementation without importing torch here."""

    method = getattr(left, "minimum", None)
    if callable(method):
        return method(right)
    # PyTorch Tensor exposes no stable instance minimum on older releases.
    module = __import__(left.__class__.__module__.split(".", 1)[0])
    minimum = getattr(module, "minimum", None)
    if not callable(minimum):
        _fail("tensor backend does not expose minimum")
    return minimum(left, right)


def configure_training_student(model: Any) -> None:
    config = getattr(model, "config", None)
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if config is None or not hasattr(config, "use_cache") or not callable(enable):
        _fail("training Student lacks cache/checkpointing controls")
    config.use_cache = False
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})


_PRIVATE_KEYS = frozenset(
    {
        "raw_prompt",
        "prompt_text",
        "question",
        "answer",
        "reasoning",
        "label",
        "standard_answer",
        "completion_text",
    }
)


def validate_cpu_artifact_value(value: Any, *, _key: str | None = None) -> bool:
    """Reject Tensor/graph objects, non-finite numbers, and private text fields."""

    if _key in _PRIVATE_KEYS:
        _fail(f"private field {_key} is forbidden in memory artifacts")
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("memory artifact contains a non-finite value")
        return True
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("memory artifact keys must be strings")
            validate_cpu_artifact_value(item, _key=key)
        return True
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_cpu_artifact_value(item)
        return True
    if hasattr(value, "detach") or hasattr(value, "grad_fn") or hasattr(value, "device"):
        _fail("memory artifacts must not retain Tensor or device objects")
    _fail(f"unsupported memory artifact value: {type(value).__name__}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if path.read_bytes() != payload:
            _fail("memory telemetry atomic write verification failed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MemoryTelemetryWriterV1:
    """Atomic pre-OOM markers plus an fsynced privacy-safe JSONL stream."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        gpu_snapshot_provider: Callable[[], Sequence[Mapping[str, Any]]],
    ) -> None:
        self.root = Path(root).resolve()
        self.run_id = str(run_id)
        self._gpu_snapshot_provider = gpu_snapshot_provider
        self._sequence = 0
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "markers").mkdir(parents=True, exist_ok=True)

    def _commit(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = _canonical_bytes(value)
        marker_path = self.root / "markers" / f"{value['sequence']:06d}.json"
        _atomic_write(marker_path, payload)
        jsonl = self.root / "telemetry.jsonl"
        with jsonl.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = json.loads(marker_path.read_text(encoding="utf-8"))
        if persisted != value:
            _fail("memory telemetry marker readback differs")
        result = dict(value)
        result["path"] = str(marker_path)
        return result

    def mark_before(
        self,
        *,
        phase: str,
        step: int,
        sequence_shape: Sequence[int] | None,
        token_shape: Sequence[int] | None,
        registry_count: int,
        model_count: int,
    ) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "artifact_kind": "b2_memory_phase_marker_v1",
            "run_id": self.run_id,
            "sequence": self._sequence,
            "marker": "before",
            "phase": str(phase),
            "step": int(step),
            "sequence_shape": None if sequence_shape is None else list(sequence_shape),
            "token_shape": None if token_shape is None else list(token_shape),
            "registry_count": int(registry_count),
            "model_count": int(model_count),
            "gpus": [dict(item) for item in self._gpu_snapshot_provider()],
            "raw_prompt_persisted": False,
            "response_tokens_persisted": False,
        }
        result = self._commit(value)
        self._sequence += 1
        return result

    def mark_after(
        self, before: Mapping[str, Any], *, elapsed_seconds: float
    ) -> dict[str, Any]:
        if before.get("marker") != "before" or not Path(str(before.get("path"))).is_file():
            _fail("memory phase completion lacks its durable before marker")
        value = {
            "schema_version": 1,
            "artifact_kind": "b2_memory_phase_marker_v1",
            "run_id": self.run_id,
            "sequence": self._sequence,
            "marker": "after",
            "phase": before["phase"],
            "step": before["step"],
            "sequence_shape": before["sequence_shape"],
            "token_shape": before["token_shape"],
            "registry_count": before["registry_count"],
            "model_count": before["model_count"],
            "elapsed_seconds": float(elapsed_seconds),
            "gpus": [dict(item) for item in self._gpu_snapshot_provider()],
            "raw_prompt_persisted": False,
            "response_tokens_persisted": False,
        }
        result = self._commit(value)
        self._sequence += 1
        return result


def _gpu_map(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    values = record.get("gpus")
    if not isinstance(values, list) or len(values) != 2:
        _fail("step-end memory record must contain GPU0 and GPU1")
    mapped: dict[int, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or value.get("device") not in (0, 1):
            _fail("step-end GPU record differs")
        mapped[int(value["device"])] = value
    if set(mapped) != {0, 1}:
        _fail("step-end GPU identities differ")
    return mapped


def _monotonic_growth(values: list[int]) -> tuple[bool, int]:
    return all(right >= left for left, right in zip(values, values[1:])), (
        values[-1] - values[0]
    )


def evaluate_six_step_memory_drift(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 6 or [item.get("step") for item in records] != list(range(1, 7)):
        _fail("six-step drift evidence must be contiguous steps 1 through 6")
    registry = {item.get("registry_count") for item in records}
    models = {item.get("model_count") for item in records}
    reasons: list[str] = []
    if len(registry) != 1:
        reasons.append("registry_count_drift")
    if len(models) != 1:
        reasons.append("model_count_drift")
    gpu_results: list[dict[str, Any]] = []
    mapped = [_gpu_map(item) for item in records]
    for device in (0, 1):
        allocated = [int(item[device]["memory_allocated_bytes"]) for item in mapped]
        reserved = [int(item[device]["memory_reserved_bytes"]) for item in mapped]
        inactive = [int(item[device].get("inactive_split_bytes") or 0) for item in mapped]
        nonrel = [int(item[device].get("non_releasable_bytes") or 0) for item in mapped]
        allocated_growth = allocated[-1] - allocated[0]
        reserved_growth = reserved[-1] - reserved[0]
        inactive_monotonic, inactive_growth = _monotonic_growth(inactive)
        nonrel_monotonic, nonrel_growth = _monotonic_growth(nonrel)
        if allocated_growth > MAX_STEP_END_ALLOCATED_DRIFT_BYTES:
            reasons.append(f"gpu{device}_allocated_growth")
        if reserved_growth > MAX_STEP_END_RESERVED_DRIFT_BYTES:
            reasons.append(f"gpu{device}_reserved_growth")
        if inactive_monotonic and inactive_growth > MAX_MONOTONIC_AUXILIARY_DRIFT_BYTES:
            reasons.append(f"gpu{device}_monotonic_inactive_split_growth")
        if nonrel_monotonic and nonrel_growth > MAX_MONOTONIC_AUXILIARY_DRIFT_BYTES:
            reasons.append(f"gpu{device}_monotonic_non_releasable_growth")
        gpu_results.append(
            {
                "device": device,
                "allocated_growth_bytes": allocated_growth,
                "reserved_growth_bytes": reserved_growth,
                "inactive_split_growth_bytes": inactive_growth,
                "inactive_split_monotonic": inactive_monotonic,
                "non_releasable_growth_bytes": nonrel_growth,
                "non_releasable_monotonic": nonrel_monotonic,
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "b2_six_step_memory_drift_gate_v1",
        "passed": not reasons,
        "steps": list(range(1, 7)),
        "failure_reasons": reasons,
        "gpus": gpu_results,
        "thresholds": deepcopy(MEMORY_EXECUTION_CONTRACT["six_step_drift_gate"]),
    }


def assert_canary_isolated(
    evidence: Mapping[str, Any],
    *,
    formal_initial_adapter_sha256: str,
    formal_policy_version: int,
) -> dict[str, Any]:
    headroom = evidence.get("minimum_free_bytes_by_gpu")
    if not (
        evidence.get("status") == "passed"
        and evidence.get("oom") is False
        and evidence.get("non_finite") is False
        and evidence.get("session_closed") is True
        and isinstance(headroom, list)
        and len(headroom) == 2
        and all(int(value) >= MINIMUM_CANARY_HEADROOM_BYTES for value in headroom)
        and formal_policy_version == 0
        and isinstance(formal_initial_adapter_sha256, str)
        and len(formal_initial_adapter_sha256) == 64
        and evidence.get("initial_adapter_sha256")
        == formal_initial_adapter_sha256
    ):
        _fail("canary isolation or one-GiB headroom gate failed")
    return {
        "schema_version": 1,
        "artifact_kind": "b2_memory_canary_isolation_audit_v1",
        "passed": True,
        "minimum_free_bytes_by_gpu": [int(value) for value in headroom],
        "formal_policy_version": formal_policy_version,
        "formal_initial_adapter_sha256": formal_initial_adapter_sha256,
        "canary_session_closed": True,
    }


class MemoryStepCoordinator:
    """Enforce one optimizer/refresh/version transition after four backwards."""

    def __init__(
        self,
        *,
        policy_version: int,
        expected_microbatches: int,
        rollout_authority_sha256: str,
        p_old_authority_sha256: str,
        optimizer_step: Callable[[], None],
        sampler_refresh: Callable[[], None],
    ) -> None:
        if expected_microbatches != 4:
            _fail("expected_microbatches differs from four")
        if rollout_authority_sha256 != p_old_authority_sha256:
            _fail("rollout and p_old authority differ")
        self.policy_version = int(policy_version)
        self.expected_microbatches = expected_microbatches
        self.rollout_authority_sha256 = rollout_authority_sha256
        self.p_old_authority_sha256 = p_old_authority_sha256
        self.optimizer_step = optimizer_step
        self.sampler_refresh = sampler_refresh
        self._recorded: list[int] = []
        self._committed = False

    def record_backward(
        self,
        *,
        microbatch_index: int,
        rollout_authority_sha256: str,
        p_old_authority_sha256: str,
        teacher_tokens_sha256: str,
        student_tokens_sha256: str,
    ) -> None:
        if self._committed:
            _fail("memory step was already committed")
        if microbatch_index != len(self._recorded):
            _fail("microbatch order is not contiguous")
        if (
            rollout_authority_sha256 != self.rollout_authority_sha256
            or p_old_authority_sha256 != self.p_old_authority_sha256
        ):
            _fail("microbatch rollout/p_old authority drift")
        if teacher_tokens_sha256 != student_tokens_sha256:
            _fail("Teacher and Student must score the same token sequence")
        self._recorded.append(microbatch_index)

    def commit(self) -> dict[str, Any]:
        if self._committed:
            _fail("optimizer/refresh may execute only once")
        if len(self._recorded) != self.expected_microbatches:
            _fail("optimizer cannot run before four backwards")
        self.optimizer_step()
        self.sampler_refresh()
        self._committed = True
        return {
            "from_policy_version": self.policy_version,
            "to_policy_version": self.policy_version + 1,
            "optimizer_step_count": 1,
            "sampler_refresh_count": 1,
            "microbatch_count": len(self._recorded),
        }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MemoryExecutionV1Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    validate_cpu_artifact_value(value)
    return value


def validate_memory_run_artifacts(
    output_dir: str | Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Recompute the canary, drift, per-step and telemetry gates from disk."""

    output = Path(output_dir).resolve()
    if len(records) != 20:
        _fail("memory execution requires exactly 20 committed step records")
    canary = _read_json_object(output / "memory_canary.json", "memory canary")
    drift = _read_json_object(
        output / "memory_six_step_drift.json", "six-step memory drift"
    )
    canary_telemetry = canary.get("telemetry")
    canary_telemetry_path = output / "memory_canary_telemetry.jsonl"
    canary_line_count = 0
    if not (
        isinstance(canary_telemetry, Mapping)
        and canary_telemetry.get("path") == "memory_canary_telemetry.jsonl"
        and canary_telemetry_path.is_file()
        and not canary_telemetry_path.is_symlink()
        and canary_telemetry.get("sha256") == _stream_sha256(canary_telemetry_path)
        and canary_telemetry.get("size_bytes") == canary_telemetry_path.stat().st_size
    ):
        _fail("memory canary telemetry SHA/size binding failed")
    with canary_telemetry_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MemoryExecutionV1Error(
                    f"memory canary telemetry line {line_number} is invalid"
                ) from error
            validate_cpu_artifact_value(row)
            canary_line_count += 1
    if canary_telemetry.get("record_count") != canary_line_count or canary_line_count <= 0:
        _fail("memory canary telemetry record count differs")
    if not (
        canary.get("artifact_kind") == "b2_memory_canary_v1"
        and canary.get("passed") is True
        and canary.get("formal_student_rebuilt_fresh_v0") is True
        and canary.get("canary_session_closed") is True
        and canary.get("minimum_free_bytes_by_gpu")
        and all(
            int(value) >= MINIMUM_CANARY_HEADROOM_BYTES
            for value in canary["minimum_free_bytes_by_gpu"]
        )
        and drift.get("artifact_kind") == "b2_six_step_memory_drift_gate_v1"
        and drift.get("passed") is True
        and drift.get("steps") == [1, 2, 3, 4, 5, 6]
    ):
        _fail("memory canary or six-step drift gate failed")
    audits: list[dict[str, Any]] = []
    for step in range(1, 21):
        value = _read_json_object(
            output / "memory_step_audits" / f"step_{step:02d}.json",
            f"memory step {step} audit",
        )
        if not (
            value.get("artifact_kind") == "b2_memory_step_execution_audit_v1"
            and value.get("optimizer_step") == step
            and value.get("from_policy_version") == step - 1
            and value.get("to_policy_version") == step
            and value.get("physical_microbatch_size") == 1
            and value.get("gradient_accumulation_steps") == 4
            and value.get("effective_batch_size") == 4
            and value.get("backward_prompt_count") == 4
            and value.get("optimizer_step_count") == 1
            and value.get("sampler_refresh_count") == 1
            and value.get("rollout_resampled_during_accumulation") is False
            and value.get("p_old_detached") is True
            and value.get("teacher_same_token_scoring") is True
            and value.get("teacher_generated_completion") is False
        ):
            _fail(f"memory step {step} execution audit differs")
        audits.append(value)
    telemetry_path = output / "memory_telemetry/telemetry.jsonl"
    if telemetry_path.is_symlink() or not telemetry_path.is_file():
        _fail("memory telemetry stream is absent or a symlink")
    marker_count = 0
    open_before: dict[tuple[int, str], int] = {}
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MemoryExecutionV1Error(
                    f"memory telemetry line {line_number} is invalid"
                ) from error
            validate_cpu_artifact_value(row)
            if not (
                isinstance(row, Mapping)
                and row.get("artifact_kind") == "b2_memory_phase_marker_v1"
                and row.get("sequence") == marker_count
                and row.get("marker") in {"before", "after"}
            ):
                _fail("memory telemetry sequence/schema differs")
            identity = (int(row.get("step", -1)), str(row.get("phase", "")))
            if row["marker"] == "before":
                if identity in open_before:
                    _fail("memory telemetry phase opened twice")
                open_before[identity] = marker_count
            elif identity not in open_before:
                _fail("memory telemetry completion lacks before marker")
            else:
                del open_before[identity]
            marker_count += 1
    if open_before or marker_count == 0:
        _fail("memory telemetry contains incomplete phases")
    return {
        "schema_version": 1,
        "artifact_kind": "b2_memory_execution_disk_audit_v1",
        "passed": True,
        "canary_passed": True,
        "six_step_drift_passed": True,
        "step_audit_count": len(audits),
        "telemetry_marker_count": marker_count,
        "telemetry_sha256": _stream_sha256(telemetry_path),
    }


__all__ = [
    "CHECKPOINT_VERSIONS",
    "MEMORY_EXECUTION_CONTRACT",
    "MemoryExecutionV1Error",
    "MemoryStepCoordinator",
    "MemoryTelemetryWriterV1",
    "assert_canary_isolated",
    "assert_prompt_equal_reduction_batch",
    "backward_selected_hidden_chunks",
    "build_target_chunks",
    "canonical_json_sha256",
    "configure_training_student",
    "evaluate_six_step_memory_drift",
    "scaled_prompt_chunk_loss",
    "target_logprobs_from_selected_logits",
    "validate_cpu_artifact_value",
    "validate_memory_execution_contract",
    "validate_memory_run_artifacts",
]
