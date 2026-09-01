"""P4.8e exact LM-head chunking with one backbone backward per prompt.

This module contains no model loader and is safe to import during CPU gates.
It separates the bounded full-vocabulary LM-head graphs from the checkpointed
backbone graph, accumulates the exact selected-hidden gradient, and propagates
that gradient through the backbone once.
"""

from __future__ import annotations

from typing import Any, Callable

from src.opd.production_b2_memory_execution_v1 import (
    MemoryExecutionV1Error,
    build_target_chunks,
    scaled_prompt_chunk_loss,
    target_logprobs_from_selected_logits,
)


def _fail(message: str) -> None:
    raise MemoryExecutionV1Error(message)


def backward_selected_hidden_once(
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
    lm_head_layout_rows: int,
    target_position_offset: int,
    chunk_observer: Callable[[str, int, int, int], None] | None = None,
    capture_per_token: bool = False,
) -> dict[str, Any]:
    """Backprop exact chunks in the causal-LM row layout, then the backbone once."""

    if not (
        getattr(selected_hidden_states, "ndim", None) == 3
        and selected_hidden_states.shape[0] == 1
        and getattr(selected_hidden_states, "requires_grad", False)
        and getattr(target_ids, "ndim", None) == 2
        and target_ids.shape[0] == 1
        and selected_hidden_states.shape[1] == target_ids.shape[1]
        and int(target_ids.shape[1]) == prompt_valid_token_count
    ):
        _fail("selected hidden states and target IDs differ")
    if not (
        isinstance(lm_head_layout_rows, int)
        and isinstance(target_position_offset, int)
        and target_position_offset >= 0
        and lm_head_layout_rows >= prompt_valid_token_count + 1
        and target_position_offset + prompt_valid_token_count
        == lm_head_layout_rows - 1
    ):
        _fail("LM-head row layout differs from the causal sequence contract")
    for name, value in (
        ("old_logprob", old_logprob),
        ("advantage", advantage),
        ("correction_weight", correction_weight),
    ):
        if not (
            getattr(value, "ndim", None) == 1
            and int(value.shape[0]) == prompt_valid_token_count
            and not bool(getattr(value, "requires_grad", False))
        ):
            _fail(f"{name} differs from the frozen prompt tensor contract")

    # The leaf intentionally shares only values with the selected hidden state.
    # Head-chunk backward populates leaf.grad without retaining or re-entering
    # the checkpointed transformer graph.
    hidden_leaf = selected_hidden_states.detach().requires_grad_(True)
    chunks = build_target_chunks(prompt_valid_token_count, chunk_size=chunk_size)
    total_loss = 0.0
    captured_q: list[Any] = []
    import torch

    for chunk_index, (start, end) in enumerate(chunks):
        if chunk_observer is not None:
            chunk_observer("before", chunk_index, start, end)
        # cuBLAS BF16 kernels are shape-sensitive.  A target-only M dimension
        # can produce a different hidden gradient than the mathematically
        # equivalent full causal-LM row layout, which Adam amplifies around
        # near-zero elements.  Zero-gradient prefix/suffix rows preserve the
        # Legacy GEMM layout without retaining non-target hidden states or
        # allowing their values to affect q, loss, or the backbone gradient.
        left = torch.zeros(
            (
                1,
                target_position_offset + start,
                int(hidden_leaf.shape[-1]),
            ),
            dtype=hidden_leaf.dtype,
            device=hidden_leaf.device,
        )
        right = torch.zeros(
            (
                1,
                lm_head_layout_rows - (target_position_offset + end),
                int(hidden_leaf.shape[-1]),
            ),
            dtype=hidden_leaf.dtype,
            device=hidden_leaf.device,
        )
        aligned_hidden = torch.cat(
            (left, hidden_leaf[:, start:end, :], right), dim=1
        )
        aligned_logits = lm_head(aligned_hidden)
        logits = aligned_logits[
            :,
            target_position_offset + start : target_position_offset + end,
            :,
        ]
        current = target_logprobs_from_selected_logits(
            logits, target_ids[:, start:end]
        ).reshape(-1)
        if capture_per_token:
            captured_q.append(current.detach().float().cpu().clone())
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
        total_loss += float(loss.detach().cpu())
        loss.backward()
        del loss, current, logits, aligned_logits, aligned_hidden, left, right
        if chunk_observer is not None:
            chunk_observer("after", chunk_index, start, end)

    hidden_gradient = hidden_leaf.grad
    if hidden_gradient is None:
        _fail("LM-head chunks did not produce a selected-hidden gradient")
    selected_hidden_states.backward(hidden_gradient)
    del hidden_gradient, hidden_leaf
    result = {
        "loss": total_loss,
        "lm_head_chunk_count": len(chunks),
        "backbone_backward_calls": 1,
        "retain_graph_calls": 0,
    }
    if capture_per_token:
        # Concatenation happens after the backbone backward so no LM-head graph
        # survives in diagnostic telemetry.
        result["q_target_logprob"] = torch.cat(captured_q, dim=0)
    return result


__all__ = ["backward_selected_hidden_once"]
