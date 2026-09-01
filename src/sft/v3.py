"""Frozen MCQ-dominant SFT-v3 targets and task-balanced scheduling.

This module is CPU-safe and model-agnostic.  It deliberately reuses the same
non-thinking chat-template boundary and ``WeightedExample`` contract as SFT-v2,
while changing only the exported assistant supervision described by ADR-0014.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Mapping, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, format_mcq_question
from src.sft.ddp import distributed_sample_indices
from src.sft.weighted import WeightedExample, _answer_index, _encode, _render_prompt


class SFTV3Kind(str, Enum):
    CMB = "cmb_mcq_letter"
    MEDICAL_O1 = "medical_o1_response"


def _reject_role(row: Mapping[str, Any]) -> None:
    role = str(row.get("target_role") or "")
    if "final" in role or "confirmation" in role:
        raise PermissionError("SFT-v3 cannot read final or confirmation roles")
    if role != "medical_sft_train":
        raise PermissionError("SFT-v3 accepts only medical_sft_train")


def render_sft_v3_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_seq_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> WeightedExample | None:
    """Render one minimal equal-weight SFT-v3 target without tail truncation."""

    _reject_role(row)
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    kind = str(row.get("sft_v3_kind") or "")
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("SFT-v3 question is required")
    if "<think>" in question.casefold() or "</think>" in question.casefold():
        raise ValueError("SFT-v3 refuses literal thinking tags")

    if kind == SFTV3Kind.CMB.value:
        options = [str(value).strip() for value in row.get("options") or []]
        if not 2 <= len(options) <= 5 or any(not value for value in options):
            raise ValueError("SFT-v3 CMB requires two to five ordered options")
        answer_index = _answer_index(row, len(options))
        target_text = chr(ord("A") + answer_index)
        user_content = format_mcq_question(question, options)
    elif kind == SFTV3Kind.MEDICAL_O1.value:
        if "reasoning" in row or "complex_cot" in {str(key).casefold() for key in row}:
            raise ValueError("SFT-v3 Medical-O1 export must omit reasoning/Complex_CoT")
        target_text = str(row.get("answer") or "").strip()
        if not target_text:
            raise ValueError("SFT-v3 Medical-O1 Response is required")
        if "<think>" in target_text.casefold() or "</think>" in target_text.casefold():
            raise ValueError("SFT-v3 refuses literal thinking tags")
        user_content = question
    else:
        raise ValueError(f"unsupported SFT-v3 kind: {kind}")

    prompt_text = _render_prompt(tokenizer, user_content, system_prompt)
    prompt_ids = _encode(tokenizer, prompt_text)
    target_ids = _encode(tokenizer, target_text)
    eos = getattr(tokenizer, "eos_token_id", None)
    if not prompt_ids or not target_ids or eos is None:
        raise ValueError("SFT-v3 prompt, target, and EOS must tokenize")
    input_ids = prompt_ids + target_ids + [int(eos)]
    if len(input_ids) > max_seq_length:
        return None
    return WeightedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=[-100] * len(prompt_ids) + target_ids + [int(eos)],
        loss_weights=[0.0] * len(prompt_ids) + [1.0] * (len(target_ids) + 1),
        prompt_length=len(prompt_ids),
        prompt_text=prompt_text,
        target_text=target_text,
        segment_token_counts={
            "answer": len(target_ids),
            "reasoning": 0,
            "eos": 1,
        },
        segment_weighted_contribution={
            "answer": float(len(target_ids)),
            "reasoning": 0.0,
            "eos": 1.0,
        },
    )


def task_for_optimizer_step(step: int) -> SFTV3Kind:
    """Return the pre-registered CMB,CMB,CMB,Medical-O1 task at zero-based step."""

    if step < 0:
        raise ValueError("optimizer step must be non-negative")
    return SFTV3Kind.MEDICAL_O1 if step % 4 == 3 else SFTV3Kind.CMB


def sft_v3_task_counts_through_step(completed_steps: int) -> dict[str, int]:
    """Disclose exact prefix task counts at preregistered checkpoint boundaries."""

    if not 0 <= completed_steps <= 600:
        raise ValueError("completed SFT-v3 steps must be in [0, 600]")
    counts = Counter(task_for_optimizer_step(step) for step in range(completed_steps))
    return {
        "cmb": counts[SFTV3Kind.CMB],
        "medical_o1": counts[SFTV3Kind.MEDICAL_O1],
    }


def validate_sft_v3_schedule(
    *, total_steps: int, checkpoints: Sequence[int], global_batch: int = 16
) -> dict[str, Any]:
    if total_steps != 600 or global_batch != 16:
        raise ValueError("SFT-v3 freezes 600 optimizer steps and global batch 16")
    if list(checkpoints) != [150, 300, 450, 600]:
        raise ValueError("SFT-v3 checkpoint contract drift")
    counts = Counter(task_for_optimizer_step(step).value for step in range(total_steps))
    if counts != {SFTV3Kind.CMB.value: 450, SFTV3Kind.MEDICAL_O1.value: 150}:
        raise ValueError("SFT-v3 3:1 schedule drift")
    return {
        "period": [
            SFTV3Kind.CMB.value,
            SFTV3Kind.CMB.value,
            SFTV3Kind.CMB.value,
            SFTV3Kind.MEDICAL_O1.value,
        ],
        "optimizer_steps": total_steps,
        "task_steps": {
            "cmb": counts[SFTV3Kind.CMB.value],
            "medical_o1": counts[SFTV3Kind.MEDICAL_O1.value],
        },
        "global_exposures": {
            "cmb": counts[SFTV3Kind.CMB.value] * global_batch,
            "medical_o1": counts[SFTV3Kind.MEDICAL_O1.value] * global_batch,
        },
        "checkpoints": list(checkpoints),
        "checkpoint_note": (
            "All checkpoints follow complete DDP accumulation windows; the exact 150/450 "
            "quarter points occur inside the four-step task period, while 300/600 close it."
        ),
    }


def build_task_balanced_rank_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    seed: int,
    accumulation_steps: int,
    start_optimizer_step: int = 0,
) -> list[dict[str, Any]]:
    """Produce rank-local rows in the frozen task order, preserving resume phase."""

    if world_size != 2 or accumulation_steps <= 0 or not 0 <= rank < world_size:
        raise ValueError("SFT-v3 requires two ranks and positive accumulation")
    by_kind = {
        SFTV3Kind.CMB: [dict(row) for row in rows if row.get("sft_v3_kind") == SFTV3Kind.CMB.value],
        SFTV3Kind.MEDICAL_O1: [
            dict(row) for row in rows if row.get("sft_v3_kind") == SFTV3Kind.MEDICAL_O1.value
        ],
    }
    if sum(len(values) for values in by_kind.values()) != len(rows):
        raise ValueError("SFT-v3 dataset contains an unsupported task")
    expected_ratio = len(by_kind[SFTV3Kind.CMB]) == 3 * len(by_kind[SFTV3Kind.MEDICAL_O1])
    if not expected_ratio or any(len(values) % world_size for values in by_kind.values()):
        raise ValueError("SFT-v3 source counts must be a divisible 3:1 mixture")
    local: dict[SFTV3Kind, list[dict[str, Any]]] = {}
    for kind, values in by_kind.items():
        indices = distributed_sample_indices(
            len(values), rank=rank, world_size=world_size, seed=seed, epoch=0
        )
        local[kind] = [values[index] for index in indices]
    positions = {SFTV3Kind.CMB: 0, SFTV3Kind.MEDICAL_O1: 0}
    total_steps = len(rows) // (world_size * accumulation_steps)
    ordered: list[dict[str, Any]] = []
    for step in range(total_steps):
        kind = task_for_optimizer_step(step)
        start = positions[kind]
        stop = start + accumulation_steps
        window = local[kind][start:stop]
        if len(window) != accumulation_steps:
            raise ValueError("SFT-v3 task pool exhausted before the frozen schedule")
        ordered.extend(window)
        positions[kind] = stop
    if any(positions[kind] != len(local[kind]) for kind in positions):
        raise ValueError("SFT-v3 task schedule did not consume every rank-local row exactly once")
    if not 0 <= start_optimizer_step <= total_steps:
        raise ValueError("resume optimizer step is outside the SFT-v3 epoch")
    return ordered[start_optimizer_step * accumulation_steps :]


def build_sft_v3_smoke_rank_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    seed: int,
    accumulation_steps: int,
) -> list[dict[str, Any]]:
    """Select the first complete C,C,C,O optimizer period from the formal order.

    The helper deliberately derives the smoke from the same deterministic rank
    streams as formal training.  It changes neither the source rows nor their
    order and keeps the two ranks disjoint.
    """

    formal = build_task_balanced_rank_rows(
        rows,
        rank=rank,
        world_size=world_size,
        seed=seed,
        accumulation_steps=accumulation_steps,
    )
    selected = formal[: 4 * accumulation_steps]
    expected = [
        task_for_optimizer_step(step).value
        for step in range(4)
        for _ in range(accumulation_steps)
    ]
    if len(selected) != len(expected) or [row.get("sft_v3_kind") for row in selected] != expected:
        raise ValueError("SFT-v3 GPU smoke does not cover the frozen C,C,C,O period")
    return selected
