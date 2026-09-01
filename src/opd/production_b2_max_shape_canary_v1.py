"""P4.8e throwaway canary helpers for a guaranteed 1024-token stress path."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType
from typing import Any, Callable, Mapping, Sequence


class B2MaxShapeCanaryV1Error(RuntimeError):
    """The max-shape fixture or source-real canary batch differs."""


def _fail(message: str) -> None:
    raise B2MaxShapeCanaryV1Error(message)


def build_legal_max_shape_token_ids(
    tokenizer: Any, *, valid_token_count: int = 1024
) -> list[int]:
    if valid_token_count != 1024:
        _fail("P4.8e max-shape fixture must contain exactly 1024 valid tokens")
    # Increasing decimal fragments are prompt/label independent, decode to
    # alphanumeric text, and cannot form three identical consecutive 16-token
    # blocks.  Generate surplus tokens because real BPE tokenization is not
    # one integer per token.
    source = " ".join(str(value) for value in range(valid_token_count * 4))
    values = [
        int(value)
        for value in tokenizer.encode(source, add_special_tokens=False)
    ]
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    pad = getattr(tokenizer, "pad_token_id", None)
    forbidden = eos_ids | ({int(pad)} if isinstance(pad, int) else set())
    values = [value for value in values if value not in forbidden]
    if len(values) < valid_token_count:
        _fail("tokenizer did not produce enough legal synthetic shape tokens")
    result = values[:valid_token_count]
    from src.opd.production_length_gpu_backend_v7 import (
        detect_repetition_v7,
        validate_decoded_output_contract_v7,
    )

    decoded = tokenizer.decode(result, skip_special_tokens=False)
    if not (
        len(result) == valid_token_count
        and not forbidden.intersection(result)
        and detect_repetition_v7(result) is False
        and validate_decoded_output_contract_v7(decoded, eos_seen=False) is True
    ):
        _fail("synthetic max-shape token fixture is not a legal completion")
    return result


def choose_max_prompt_batch(
    schedule_batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    prompt_length: Callable[[Mapping[str, Any]], int],
    risk_step_index: int,
) -> list[dict[str, Any]]:
    if len(schedule_batches) != 20 or not 0 <= risk_step_index < 20:
        _fail("canary schedule envelope differs")
    if any(len(batch) != 4 for batch in schedule_batches):
        _fail("canary schedule batch is not four prompts")
    global_max = max(
        (row for batch in schedule_batches for row in batch),
        key=prompt_length,
    )
    result = [deepcopy(dict(row)) for row in schedule_batches[risk_step_index]]
    max_id = str(global_max.get("sample_id", ""))
    if max_id not in {str(row.get("sample_id", "")) for row in result}:
        role = global_max.get("target_role")
        candidates = [
            index
            for index, row in enumerate(result)
            if row.get("target_role") == role
        ]
        if not candidates:
            _fail("global max prompt role is absent from the risk batch")
        replace = min(candidates, key=lambda index: prompt_length(result[index]))
        result[replace] = deepcopy(dict(global_max))
    counts = {
        role: sum(row.get("target_role") == role for row in result)
        for role in ("medical_opd_o1", "medical_opd_cmb")
    }
    if not (
        counts == {"medical_opd_o1": 2, "medical_opd_cmb": 2}
        and len({str(row.get("sample_id", "")) for row in result}) == 4
        and max(prompt_length(row) for row in result)
        == max(prompt_length(row) for batch in schedule_batches for row in batch)
    ):
        _fail("max-prompt canary batch differs from the frozen 2+2 schedule")
    return result


def production_prompt_token_length(session: Any, row: Mapping[str, Any]) -> int:
    prompt = session.render_prompt_text(row)
    return len(
        session.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def install_max_shape_rollout_fixture(
    session: Any, *, valid_token_count: int = 1024
) -> None:
    """Retain a real rollout, then replace one row with legal max-shape tokens."""

    original = session._generate_rows

    def generate_with_fixture(
        bound_self: Any,
        model: Any,
        rows: list[Mapping[str, Any]],
        *,
        device: str,
        step_index: int,
    ) -> list[dict[str, Any]]:
        generated = original(
            model, rows, device=device, step_index=step_index
        )
        if not isinstance(generated, list) or len(generated) != 4:
            _fail("real canary rollout did not produce four trajectories")
        real_lengths = [len(row["response_ids"]) for row in generated]
        response_ids = build_legal_max_shape_token_ids(
            bound_self.tokenizer, valid_token_count=valid_token_count
        )
        import torch

        prompt_lengths: list[int] = []
        for fixture_index in range(4):
            fixture = deepcopy(dict(generated[fixture_index]))
            fixture["response_ids"] = list(response_ids)
            fixture["eos_observed"] = False
            with torch.inference_mode():
                behavior = bound_self._selected_chunk_logprobs(
                    bound_self.sampler_model,
                    fixture,
                    device="cuda:1",
                    phase=(
                        "canary_max_shape_behavior_row_"
                        f"{fixture_index + 1}"
                    ),
                )
            fixture["rollout_behavior_logprob"] = [
                float(value) for value in behavior.reshape(-1).tolist()
            ]
            generated[fixture_index] = fixture
            prompt_lengths.append(len(rows[fixture_index]["prompt_ids"]))
        bound_self._p4e_max_shape_fixture = {
            "real_rollout_executed": True,
            "real_rollout_trajectory_count": 4,
            "real_rollout_completion_lengths": real_lengths,
            "fixture_indices": [0, 1, 2, 3],
            "fixture_prompt_tokens": max(prompt_lengths),
            "fixture_prompt_tokens_by_prompt": prompt_lengths,
            "fixture_valid_completion_tokens": len(response_ids),
            "fixture_valid_completion_tokens_by_prompt": [len(response_ids)] * 4,
            "fixture_eos": False,
            "fixture_source": "prompt_only_legal_synthetic_token_shape",
            "synthetic_target_scoring_count": 4,
            "label_access_count": 0,
            "controller_access_count": 0,
            "final_access_count": 0,
            "response_tokens_persisted": False,
        }
        return generated

    session._generate_rows = MethodType(generate_with_fixture, session)


__all__ = [
    "B2MaxShapeCanaryV1Error",
    "build_legal_max_shape_token_ids",
    "choose_max_prompt_batch",
    "install_max_shape_rollout_fixture",
    "production_prompt_token_length",
]
