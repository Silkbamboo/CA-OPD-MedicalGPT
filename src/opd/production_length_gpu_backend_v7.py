"""Lazy real-GPU backend for the P4.7 length-only continuation.

Importing this module is CPU safe.  Model, PEFT and CUDA imports occur only in
``ProductionLengthGpuBackendV7.__init__``, which the authorized GPU launcher
calls after formal preflight.  Raw prompts and token arrays never leave the
live backend; prefix arrays are returned only to the in-process comparison and
must be reduced to hashes before artifact persistence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import unicodedata

from src.opd.production_length_contract_v7 import canonical_json_sha256


REPETITION_RULE_VERSION = "consecutive-identical-16-token-block-x3-v1"
OUTPUT_CONTRACT_VERSION = "qwen3-text-and-terminal-stop-v1"
PRODUCTION_SLOT = "student_active"
_TERMINAL_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>")


class ProductionLengthGpuBackendV7Error(RuntimeError):
    """The fresh v2 sampler or real generation route failed closed."""


def _fail(message: str) -> None:
    raise ProductionLengthGpuBackendV7Error(message)


def detect_repetition_v7(token_ids: Sequence[int]) -> bool:
    """Detect three consecutive identical 16-token blocks."""

    values = list(token_ids)
    block = 16
    window = block * 3
    return any(
        values[start : start + block]
        == values[start + block : start + block * 2]
        == values[start + block * 2 : start + window]
        for start in range(max(0, len(values) - window + 1))
    )


def validate_decoded_output_contract_v7(text: Any, *, eos_seen: bool) -> bool:
    """Parse the privacy-safe text/stop envelope without judging correctness.

    This is deliberately a syntactic generation-health contract, not a
    medical-capability metric: it accepts ordinary prose or a bare MCQ letter,
    but rejects undecodable/control-only output, punctuation-only garbage,
    injected ChatML control tokens and EOS/text provenance disagreements.
    Neither prompts, labels nor parsed answer values are persisted.
    """

    if not isinstance(text, str) or not isinstance(eos_seen, bool):
        return False
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    if "\ufffd" in text or any(
        (
            unicodedata.category(character) in {"Cc", "Cs"}
            and character not in {"\t", "\n", "\r"}
        )
        or character in {"\ufffe", "\uffff"}
        for character in text
    ):
        return False
    stripped = text.strip()
    if eos_seen:
        terminal = [
            marker for marker in _TERMINAL_STOP_MARKERS if stripped.endswith(marker)
        ]
        if len(terminal) != 1:
            return False
        body = stripped[: -len(terminal[0])].rstrip()
    else:
        body = stripped
        if any(marker in body for marker in _TERMINAL_STOP_MARKERS):
            return False
    if not body or "<|" in body or "|>" in body:
        return False
    return any(character.isalnum() for character in body)


def validate_runtime_generation_binding(
    *,
    runtime_eos_token_id: Any,
    runtime_pad_token_id: Any,
    frozen_generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if the loaded model would silently change stop semantics."""

    frozen_eos = frozen_generation.get("eos_token_id")
    frozen_pad = frozen_generation.get("pad_token_id")
    if not (
        isinstance(frozen_eos, list)
        and frozen_eos
        and all(isinstance(value, int) and not isinstance(value, bool) for value in frozen_eos)
        and isinstance(frozen_pad, int)
        and not isinstance(frozen_pad, bool)
        and runtime_eos_token_id == frozen_eos
        and runtime_pad_token_id == frozen_pad
    ):
        _fail("loaded model EOS/pad binding differs from frozen P4.7 config")
    return {"eos_token_id": list(frozen_eos), "pad_token_id": frozen_pad}


def _candidate_health_v7(
    *,
    response_token_ids: Sequence[int],
    selected_logprobs: Sequence[float],
    eos_token_ids: set[int],
    candidate_cap: int,
    decoded_for_health: str,
) -> dict[str, Any]:
    """Reduce one candidate prefix without retaining tokens or decoded text."""

    ids = [int(value) for value in response_token_ids[:candidate_cap]]
    logprobs = [float(value) for value in selected_logprobs[:candidate_cap]]
    finite = len(ids) == len(logprobs) and all(
        math.isfinite(item) for item in logprobs
    )
    eos_positions = [
        index + 1 for index, token in enumerate(ids) if token in eos_token_ids
    ]
    first_eos = eos_positions[0] if eos_positions else None
    empty = not ids
    unexpected_stop = first_eos is None and len(response_token_ids) < candidate_cap
    unexpected_think = "<think>" in decoded_for_health.lower() or (
        "</think>" in decoded_for_health.lower()
    )
    output_contract_valid = validate_decoded_output_contract_v7(
        decoded_for_health, eos_seen=first_eos is not None
    )
    return {
        "finish_reason": (
            "eos"
            if first_eos is not None
            else ("unexpected_stop" if unexpected_stop else "length")
        ),
        "valid_completion": bool(
            not empty and finite and not unexpected_stop and output_contract_valid
        ),
        "empty_completion": empty,
        "non_finite": not finite,
        "unexpected_think_tag": unexpected_think,
        "repetition_detected": detect_repetition_v7(ids),
    }


def safe_generation_observation(
    *,
    sample_identity: Mapping[str, Any],
    per_sample_seed: int,
    prompt_token_count: int,
    actual_generation_cap: int,
    response_token_ids: Sequence[int],
    eos_token_ids: set[int],
    selected_logprobs: Sequence[float],
    decoded_for_health: str,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
    generation_backend_identity: str,
    model_revision: str,
    base_revision: str,
    adapter_revision: str,
    runtime_adapter_sha256: str,
    full_decoding_config_sha256: str,
    prefix_invariant_decoding_sha256: str,
    capture_prefix_provenance: bool,
    candidate_health_caps: Sequence[int] | None = None,
    candidate_decoded_for_health: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Reduce one live generation to privacy-safe evidence fields."""

    required = {"sample_id", "prompt_hash", "source", "frozen_order"}
    if set(sample_identity) != required:
        _fail("sample identity fields are not exact")
    ids = [int(value) for value in response_token_ids]
    logprobs = [float(value) for value in selected_logprobs]
    eos_positions = [
        index + 1 for index, token in enumerate(ids) if token in eos_token_ids
    ]
    first_eos = eos_positions[0] if eos_positions else None
    health_caps = tuple(
        [actual_generation_cap]
        if candidate_health_caps is None
        else candidate_health_caps
    )
    decoded_by_cap = (
        {actual_generation_cap: decoded_for_health}
        if candidate_decoded_for_health is None
        else dict(candidate_decoded_for_health)
    )
    if (
        not health_caps
        or any(
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate <= 0
            or candidate > actual_generation_cap
            for candidate in health_caps
        )
        or tuple(sorted(set(health_caps))) != health_caps
        or actual_generation_cap not in health_caps
        or set(decoded_by_cap) != set(health_caps)
        or any(not isinstance(text, str) for text in decoded_by_cap.values())
    ):
        _fail("candidate-prefix health inputs are invalid")
    candidate_health = {
        str(candidate): _candidate_health_v7(
            response_token_ids=ids,
            selected_logprobs=logprobs,
            eos_token_ids=eos_token_ids,
            candidate_cap=candidate,
            decoded_for_health=decoded_by_cap[candidate],
        )
        for candidate in health_caps
    }
    full_health = candidate_health[str(actual_generation_cap)]
    for value, length, label in (
        (sample_identity["prompt_hash"], 64, "prompt hash"),
        (model_revision, 40, "model revision"),
        (base_revision, 40, "base revision"),
        (runtime_adapter_sha256, 64, "runtime adapter SHA"),
        (full_decoding_config_sha256, 64, "full decoding config SHA"),
        (prefix_invariant_decoding_sha256, 64, "prefix decoding config SHA"),
    ):
        if not (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        ):
            _fail(f"{label} is not immutable")
    if not all(
        isinstance(value, str) and value
        for value in (
            sample_identity["sample_id"],
            sample_identity["source"],
            generation_backend_identity,
            adapter_revision,
        )
    ):
        _fail("generation identity is incomplete")
    if (
        not isinstance(per_sample_seed, int)
        or isinstance(per_sample_seed, bool)
        or per_sample_seed < 0
        or not isinstance(prompt_token_count, int)
        or prompt_token_count <= 0
        or not isinstance(actual_generation_cap, int)
        or actual_generation_cap <= 0
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0
        or not isinstance(peak_gpu_memory_bytes, int)
        or peak_gpu_memory_bytes < 0
    ):
        _fail("generation counts/resources are invalid")
    value: dict[str, Any] = {
        "sample_id": sample_identity["sample_id"],
        "prompt_hash": sample_identity["prompt_hash"],
        "source": sample_identity["source"],
        "frozen_order": sample_identity["frozen_order"],
        "per_sample_seed": per_sample_seed,
        "prompt_token_count": prompt_token_count,
        "actual_generation_cap": actual_generation_cap,
        "generated_token_count": len(ids),
        "eos_seen": first_eos is not None,
        "first_eos_position": first_eos,
        "finish_reason": full_health["finish_reason"],
        "valid_completion": full_health["valid_completion"],
        "empty_completion": full_health["empty_completion"],
        "non_finite": full_health["non_finite"],
        "unexpected_think_tag": full_health["unexpected_think_tag"],
        "repetition_detected": full_health["repetition_detected"],
        "repetition_rule_version": REPETITION_RULE_VERSION,
        "candidate_health": candidate_health,
    }
    if capture_prefix_provenance:
        value.update(
            {
                "token_ids": ids,
                "selected_logprobs": logprobs,
                "generation_backend_identity": generation_backend_identity,
                "model_revision": model_revision,
                "base_revision": base_revision,
                "adapter_revision": adapter_revision,
                "full_decoding_config_sha256": full_decoding_config_sha256,
                "prefix_invariant_decoding_sha256": (
                    prefix_invariant_decoding_sha256
                ),
                "runtime_adapter_sha256": runtime_adapter_sha256,
            }
        )
    return value


class ProductionLengthGpuBackendV7:
    """Fresh-load the immutable P4.6 v2 adapter and generate on cuda:1."""

    def __init__(self, config: Mapping[str, Any], *, repo_root: str | Path) -> None:
        # Explicitly lazy: these imports are forbidden in CPU preflight/tests.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.opd.calibration_data import render_prompt_text
        from src.opd.production_sampler_refresh_v5 import (
            adapter_artifact_identity,
            runtime_identity_from_peft,
        )

        self.config = deepcopy(dict(config))
        self.root = Path(repo_root).resolve()
        self.torch = torch
        self.render_prompt_text = render_prompt_text
        self.device = "cuda:1"
        self._closed = False
        self.last_batch_resources: dict[str, Any] = {}
        self.batch_resource_history: list[dict[str, Any]] = []
        model = self.config.get("model", {})
        parent = self.config.get("parent_reuse", {})
        parent_v2 = parent.get("v2", {})
        checkpoint = Path(
            str(parent_v2.get("canonical_absolute_path", ""))
        ).resolve()
        expected = str(parent_v2.get("aggregate_tensor_sha256", ""))
        base_revision = str(model.get("revision", ""))
        tokenizer_revision = str(model.get("tokenizer_revision", ""))
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            _fail("canonical P4.6 v2 checkpoint is absent")
        if torch.cuda.device_count() != 2:
            _fail("formal P4.7 requires exactly two visible GPUs")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model.get("id", "")),
            local_files_only=True,
            revision=tokenizer_revision,
        )
        disk = adapter_artifact_identity(
            checkpoint,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=base_revision,
            tokenizer_revision=tokenizer_revision,
        )
        if disk.get("aggregate_tensor_sha256") != expected:
            _fail("fresh v2 checkpoint differs from parent authority")
        base = AutoModelForCausalLM.from_pretrained(
            str(model.get("id", "")),
            local_files_only=True,
            revision=base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model = PeftModel.from_pretrained(
            base,
            checkpoint,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del base
        self.model.eval()
        self._generation_binding = validate_runtime_generation_binding(
            runtime_eos_token_id=self.model.generation_config.eos_token_id,
            runtime_pad_token_id=self.model.generation_config.pad_token_id,
            frozen_generation=self.config["generation"],
        )
        runtime = runtime_identity_from_peft(
            self.model,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=base_revision,
            tokenizer_revision=tokenizer_revision,
        )
        comparable = (
            "aggregate_tensor_sha256",
            "canonical_config_sha256",
            "tensor_count",
            "total_canonical_bytes",
            "base_revision",
            "tokenizer_revision",
        )
        if any(disk.get(field) != runtime.get(field) for field in comparable) or (
            runtime.get("active_adapter") != PRODUCTION_SLOT
            or runtime.get("registry_snapshot", {}).get("adapter_count") != 1
        ):
            self.close()
            _fail("fresh runtime/checkpoint v2 identity mismatch")
        self._disk_identity = disk
        self._runtime_identity = runtime
        self.runtime_adapter_sha256 = expected
        self.base_revision = base_revision
        self.model_revision = base_revision
        self.adapter_revision = "p4.6-v2"

    def identity_evidence(self) -> dict[str, Any]:
        return {
            "schema_version": 7,
            "artifact_kind": "p4_7_fresh_v2_reload_identity",
            "logical_version": "v2",
            "checkpoint_tensor_sha256": self._disk_identity[
                "aggregate_tensor_sha256"
            ],
            "runtime_tensor_sha256": self._runtime_identity[
                "aggregate_tensor_sha256"
            ],
            "canonical_config_sha256": self._runtime_identity[
                "canonical_config_sha256"
            ],
            "tensor_count": self._runtime_identity["tensor_count"],
            "total_canonical_bytes": self._runtime_identity[
                "total_canonical_bytes"
            ],
            "active_slot": self._runtime_identity["active_adapter"],
            "registry_count": self._runtime_identity["registry_snapshot"][
                "adapter_count"
            ],
            "eos_stop_config_verified": True,
            "eos_token_id": list(self._generation_binding["eos_token_id"]),
            "pad_token_id": self._generation_binding["pad_token_id"],
            "passed": True,
        }

    def conditional_4096_resource_preflight(
        self,
        *,
        max_prompt_tokens: int,
        actual_cap: int,
        minimum_free_bytes: int,
    ) -> dict[str, Any]:
        """Recheck live GPU headroom immediately before the optional 4096 run."""

        if self._closed:
            _fail("conditional GPU resource preflight used a closed backend")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (max_prompt_tokens, actual_cap, minimum_free_bytes)
        ):
            _fail("conditional GPU resource preflight input is invalid")
        context_limit = int(self.config["model"]["context_limit"])
        self.torch.cuda.synchronize(1)
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(1)
        allocated_bytes = self.torch.cuda.memory_allocated(1)
        reserved_bytes = self.torch.cuda.memory_reserved(1)
        context_within_limit = max_prompt_tokens + actual_cap <= context_limit
        headroom_passed = int(free_bytes) >= minimum_free_bytes
        return {
            "schema_version": 7,
            "artifact_kind": "p4_7_conditional_4096_gpu_resource_preflight",
            "device_index": 1,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(allocated_bytes),
            "reserved_bytes": int(reserved_bytes),
            "minimum_free_bytes": minimum_free_bytes,
            "max_prompt_tokens": max_prompt_tokens,
            "actual_generation_cap": actual_cap,
            "model_context_limit": context_limit,
            "context_within_limit": context_within_limit,
            "headroom_passed": headroom_passed,
            "passed": bool(context_within_limit and headroom_passed),
        }

    def _selected_logprobs(
        self, prompt_ids: list[int], response_ids: list[int]
    ) -> list[float]:
        torch = self.torch
        ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[
                :, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(response_ids), :
            ]
            values: list[float] = []
            for start in range(0, len(response_ids), 32):
                stop = min(start + 32, len(response_ids))
                chunk = logits[:, start:stop, :].float()
                targets = torch.tensor(
                    response_ids[start:stop], device=self.device, dtype=torch.long
                ).view(1, -1, 1)
                selected = (
                    chunk.gather(-1, targets).squeeze(-1)
                    - torch.logsumexp(chunk, dim=-1)
                )
                values.extend(float(item) for item in selected[0].cpu().tolist())
                del chunk, targets, selected
            del output, logits, ids
        return values

    def generate(
        self,
        rows: list[dict[str, object]],
        *,
        actual_cap: int,
        per_sample_seeds: list[int],
        capture_prefix_provenance: bool,
        candidate_health_caps: Sequence[int],
    ) -> list[dict[str, object]]:
        if self._closed or len(rows) != len(per_sample_seeds):
            _fail("generation backend is closed or batch identity is incomplete")
        torch = self.torch
        decoding = deepcopy(dict(self.config["generation"]))
        for metadata_key in (
            "backend",
            "backend_version",
            "batch_size",
            "enable_thinking",
            "full_support",
        ):
            decoding.pop(metadata_key, None)
        decoding["max_new_tokens"] = actual_cap
        decoding["return_dict_in_generate"] = True
        decoding["output_scores"] = False
        decoding["output_logits"] = False
        decoding["eos_token_id"] = list(self._generation_binding["eos_token_id"])
        decoding["pad_token_id"] = self._generation_binding["pad_token_id"]
        invariant = dict(decoding)
        invariant.pop("max_new_tokens")
        full_sha = canonical_json_sha256(decoding)
        invariant_sha = canonical_json_sha256(invariant)
        eos = decoding["eos_token_id"]
        eos_ids = (
            {int(eos)}
            if isinstance(eos, int)
            else {int(value) for value in (eos or [])}
        )
        health_caps = tuple(candidate_health_caps)
        if (
            not health_caps
            or tuple(sorted(set(health_caps))) != health_caps
            or health_caps[-1] != actual_cap
        ):
            _fail("candidate-prefix health caps do not match generation cap")
        observations: list[dict[str, object]] = []
        total_elapsed = 0.0
        peak = 0
        generated_total = 0
        for row, seed in zip(rows, per_sample_seeds, strict=True):
            raw = row.get("_runtime_source_row")
            if not isinstance(raw, Mapping):
                _fail("in-memory prompt source row is absent")
            prompt_text = self.render_prompt_text(raw)
            prompt_ids = [
                int(value)
                for value in self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            ]
            input_ids = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.device
            )
            torch.cuda.reset_peak_memory_stats(1)
            started = time.perf_counter()
            with torch.inference_mode(), torch.random.fork_rng(devices=[1]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    **decoding,
                )
            response_ids = [
                int(value)
                for value in generated.sequences[0, len(prompt_ids) :].tolist()
            ]
            del generated, input_ids
            selected = self._selected_logprobs(prompt_ids, response_ids)
            elapsed = time.perf_counter() - started
            item_peak = int(torch.cuda.max_memory_allocated(1))
            decoded = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            candidate_decoded = {
                candidate: self.tokenizer.decode(
                    response_ids[:candidate], skip_special_tokens=False
                )
                for candidate in health_caps
            }
            identity = {
                "sample_id": str(row["sample_id"]),
                "prompt_hash": str(row["prompt_hash"]),
                "source": str(row["source"]),
                "frozen_order": int(row["frozen_order"]),
            }
            observations.append(
                safe_generation_observation(
                    sample_identity=identity,
                    per_sample_seed=seed,
                    prompt_token_count=len(prompt_ids),
                    actual_generation_cap=actual_cap,
                    response_token_ids=response_ids,
                    eos_token_ids=eos_ids,
                    selected_logprobs=selected,
                    decoded_for_health=decoded,
                    elapsed_seconds=elapsed,
                    peak_gpu_memory_bytes=item_peak,
                    generation_backend_identity=(
                        "transformers_generate_full_support"
                    ),
                    model_revision=self.model_revision,
                    base_revision=self.base_revision,
                    adapter_revision=self.adapter_revision,
                    runtime_adapter_sha256=self.runtime_adapter_sha256,
                    full_decoding_config_sha256=full_sha,
                    prefix_invariant_decoding_sha256=invariant_sha,
                    capture_prefix_provenance=capture_prefix_provenance,
                    candidate_health_caps=health_caps,
                    candidate_decoded_for_health=candidate_decoded,
                )
            )
            generated_total += len(response_ids)
            total_elapsed += elapsed
            peak = max(peak, item_peak)
        self.last_batch_resources = {
            "actual_generation_cap": actual_cap,
            "actual_generated_tokens": generated_total,
            "elapsed_seconds": total_elapsed,
            "tokens_per_second": (
                generated_total / total_elapsed if total_elapsed > 0 else 0.0
            ),
            "peak_gpu_memory_bytes": peak,
            "actual_cost_cny": None,
        }
        self.batch_resource_history.append(dict(self.last_batch_resources))
        return observations

    def close(self) -> None:
        if self._closed:
            return
        torch = getattr(self, "torch", None)
        model = getattr(self, "model", None)
        self.model = None
        self.tokenizer = None
        if model is not None:
            del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self._closed = True


__all__ = [
    "OUTPUT_CONTRACT_VERSION",
    "ProductionLengthGpuBackendV7",
    "ProductionLengthGpuBackendV7Error",
    "REPETITION_RULE_VERSION",
    "detect_repetition_v7",
    "safe_generation_observation",
    "validate_decoded_output_contract_v7",
    "validate_runtime_generation_binding",
]
