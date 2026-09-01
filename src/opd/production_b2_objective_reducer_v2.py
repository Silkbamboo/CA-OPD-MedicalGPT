"""Canonical P4.8f corrected PPO objective and hierarchical reduction.

The production path always accumulates in FP32.  FP64 is exposed only behind
the explicit ``offline_replay`` boundary so diagnostic replay cannot silently
change training precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


class B2ObjectiveReducerV2Error(RuntimeError):
    """The canonical objective inputs or hierarchy are invalid."""


def _fail(message: str) -> None:
    raise B2ObjectiveReducerV2Error(message)


@dataclass(frozen=True)
class CanonicalHierarchicalReduction:
    token_numerators: Tensor
    valid_token_counts: Tensor
    trajectory_sums: Tensor
    trajectory_means: Tensor
    per_group: dict[tuple[str, str], Tensor]
    per_prompt: dict[str, Tensor]
    batch_objective: Tensor


@dataclass(frozen=True)
class CanonicalObjectiveResult:
    q_target_logprob: Tensor
    p_old_target_logprob: Tensor
    teacher_target_logprob: Tensor
    raw_advantage: Tensor
    scaled_advantage: Tensor
    clipped_advantage: Tensor
    raw_ppo_ratio: Tensor
    clipped_ratio: Tensor
    unclipped_objective: Tensor
    clipped_objective: Tensor
    selected_objective: Tensor
    corrected_selected_objective: Tensor
    clip_boundary_mask: Tensor
    valid_mask: Tensor
    reduction: CanonicalHierarchicalReduction
    objective: Tensor
    loss: Tensor
    accumulator_dtype: torch.dtype


def _accumulator_dtype(
    accumulator_dtype: torch.dtype, *, offline_replay: bool
) -> torch.dtype:
    if accumulator_dtype == torch.float32:
        return accumulator_dtype
    if accumulator_dtype == torch.float64 and offline_replay:
        return accumulator_dtype
    _fail("training accumulation must be FP32; FP64 is offline replay only")


def _validate_shapes(*values: Tensor) -> None:
    if not values or len({tuple(value.shape) for value in values}) != 1:
        _fail("canonical objective tensors must have identical shapes")
    if values[0].ndim != 2:
        _fail("canonical objective tensors must be [trajectory, token]")


def canonical_token_objective_from_advantage(
    *,
    q_target_logprob: Tensor,
    p_old_target_logprob: Tensor,
    raw_advantage: Tensor,
    correction_weight: Tensor,
    valid_mask: Tensor,
    clip_low: float,
    clip_high: float,
    accumulator_dtype: torch.dtype,
) -> dict[str, Tensor]:
    _validate_shapes(
        q_target_logprob,
        p_old_target_logprob,
        raw_advantage,
        correction_weight,
        valid_mask,
    )
    if clip_low < 0 or clip_high < 0:
        _fail("PPO clip bounds must be non-negative")
    if any(
        bool(value.requires_grad)
        for value in (p_old_target_logprob, raw_advantage, correction_weight)
    ):
        _fail("p_old, advantage and correction weight must be frozen")
    mask = valid_mask.to(dtype=torch.bool)
    if not bool(mask.any()) or not bool(
        torch.all((valid_mask == 0) | (valid_mask == 1))
    ):
        _fail("valid-token mask must be binary and nonempty")
    q = q_target_logprob.to(dtype=accumulator_dtype)
    old = p_old_target_logprob.detach().to(dtype=accumulator_dtype)
    advantage = raw_advantage.detach().to(dtype=accumulator_dtype)
    correction = correction_weight.detach().to(dtype=accumulator_dtype)
    for name, value in (
        ("q", q),
        ("p_old", old),
        ("advantage", advantage),
        ("correction", correction),
    ):
        if not bool(torch.isfinite(value[mask]).all()):
            _fail(f"{name} is non-finite on valid tokens")
    if bool((correction[mask] < 0).any()):
        _fail("correction weight must be non-negative")

    raw_ratio = torch.exp(q - old)
    clipped_ratio = raw_ratio.clamp(1.0 - float(clip_low), 1.0 + float(clip_high))
    unclipped = raw_ratio * advantage
    clipped = clipped_ratio * advantage
    selected = torch.minimum(unclipped, clipped)
    corrected = correction * selected
    zeros = torch.zeros_like(corrected)
    selected = torch.where(mask, selected, zeros)
    corrected = torch.where(mask, corrected, zeros)
    return {
        "q": q,
        "p_old": old,
        "raw_advantage": advantage,
        # No advantage clipping exists in the frozen algorithm.  These two
        # explicit arrays make that fact auditable without inventing new math.
        "scaled_advantage": correction * advantage,
        "clipped_advantage": correction * advantage,
        "raw_ppo_ratio": raw_ratio,
        "clipped_ratio": clipped_ratio,
        "unclipped_objective": unclipped,
        "clipped_objective": clipped,
        "selected_objective": selected,
        "corrected_selected_objective": corrected,
        "clip_boundary_mask": clipped < unclipped,
        "valid_mask": mask,
    }


def canonical_hierarchical_reduction(
    values: Tensor,
    valid_mask: Tensor,
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
) -> CanonicalHierarchicalReduction:
    """valid-token mean -> group mean -> prompt mean -> prompt batch mean."""

    _validate_shapes(values, valid_mask)
    batch = int(values.shape[0])
    if len(prompt_ids) != batch or len(group_ids) != batch:
        _fail("prompt/group identity length differs from trajectory count")
    mask = valid_mask.to(dtype=torch.bool)
    counts = mask.to(dtype=values.dtype).sum(dim=1)
    if bool((counts <= 0).any()):
        _fail("every trajectory must contain a valid token")
    masked = torch.where(mask, values, torch.zeros_like(values))
    sums = masked.sum(dim=1, dtype=values.dtype)
    means = sums / counts

    grouped: dict[tuple[str, str], list[Tensor]] = {}
    for index, (prompt, group) in enumerate(
        zip(prompt_ids, group_ids, strict=True)
    ):
        identity = (str(prompt), str(group))
        if not all(identity):
            _fail("prompt/group identity must be nonempty")
        grouped.setdefault(identity, []).append(means[index])
    per_group = {
        identity: torch.stack(items).mean(dtype=values.dtype)
        for identity, items in grouped.items()
    }
    prompted: dict[str, list[Tensor]] = {}
    for (prompt, _group), group_mean in per_group.items():
        prompted.setdefault(prompt, []).append(group_mean)
    per_prompt = {
        prompt: torch.stack(items).mean(dtype=values.dtype)
        for prompt, items in prompted.items()
    }
    if not per_prompt:
        _fail("canonical reduction contains no prompt")
    batch_objective = torch.stack(list(per_prompt.values())).mean(
        dtype=values.dtype
    )
    return CanonicalHierarchicalReduction(
        token_numerators=masked,
        valid_token_counts=counts,
        trajectory_sums=sums,
        trajectory_means=means,
        per_group=per_group,
        per_prompt=per_prompt,
        batch_objective=batch_objective,
    )


def canonical_corrected_objective(
    *,
    q_target_logprob: Tensor,
    p_old_target_logprob: Tensor,
    teacher_target_logprob: Tensor,
    correction_weight: Tensor,
    valid_mask: Tensor,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
    accumulator_dtype: torch.dtype = torch.float32,
    offline_replay: bool = False,
    advantage_scale: Tensor | float | None = None,
) -> CanonicalObjectiveResult:
    """Apply the one canonical corrected objective and hierarchy."""

    dtype = _accumulator_dtype(
        accumulator_dtype, offline_replay=offline_replay
    )
    _validate_shapes(
        q_target_logprob,
        p_old_target_logprob,
        teacher_target_logprob,
        correction_weight,
        valid_mask,
    )
    if beta <= 0:
        _fail("beta must be positive")
    if p_old_target_logprob.requires_grad or teacher_target_logprob.requires_grad:
        _fail("p_old and Teacher logprob must be frozen")
    old = p_old_target_logprob.detach().to(dtype=dtype)
    teacher = teacher_target_logprob.detach().to(dtype=dtype)
    raw_advantage = float(beta) * (teacher - old)
    if advantage_scale is None:
        advantage = raw_advantage
    else:
        scale = torch.as_tensor(
            advantage_scale,
            dtype=dtype,
            device=raw_advantage.device,
        )
        if scale.ndim == 0:
            scale = torch.full_like(raw_advantage, float(scale))
        if scale.shape != raw_advantage.shape:
            _fail("advantage safety scale must match the same-token grid")
        mask = valid_mask.to(dtype=torch.bool)
        if not bool(torch.isfinite(scale[mask]).all()) or bool((scale[mask] < 0).any()) or bool((scale[mask] > 1).any()):
            _fail("advantage safety scale must be finite and cannot amplify")
        advantage = raw_advantage * scale.detach()
    token = canonical_token_objective_from_advantage(
        q_target_logprob=q_target_logprob,
        p_old_target_logprob=old,
        raw_advantage=advantage.detach(),
        correction_weight=correction_weight,
        valid_mask=valid_mask,
        clip_low=clip_low,
        clip_high=clip_high,
        accumulator_dtype=dtype,
    )
    reduction = canonical_hierarchical_reduction(
        token["corrected_selected_objective"],
        token["valid_mask"],
        prompt_ids=prompt_ids,
        group_ids=group_ids,
    )
    return CanonicalObjectiveResult(
        q_target_logprob=token["q"],
        p_old_target_logprob=token["p_old"],
        teacher_target_logprob=teacher,
        raw_advantage=raw_advantage.detach(),
        scaled_advantage=token["raw_advantage"],
        clipped_advantage=token["raw_advantage"],
        raw_ppo_ratio=token["raw_ppo_ratio"],
        clipped_ratio=token["clipped_ratio"],
        unclipped_objective=token["unclipped_objective"],
        clipped_objective=token["clipped_objective"],
        selected_objective=token["selected_objective"],
        corrected_selected_objective=token["corrected_selected_objective"],
        clip_boundary_mask=token["clip_boundary_mask"],
        valid_mask=token["valid_mask"],
        reduction=reduction,
        objective=reduction.batch_objective,
        loss=-reduction.batch_objective,
        accumulator_dtype=dtype,
    )


def canonical_prompt_chunk_loss(
    *,
    corrected_selected_objective: Tensor,
    prompt_valid_token_count: int,
    effective_batch_size: int,
) -> Tensor:
    """Return one chunk's contribution to equal-prompt batch loss."""

    if corrected_selected_objective.ndim != 1:
        _fail("prompt chunk objective must be one-dimensional")
    if prompt_valid_token_count <= 0:
        _fail("prompt valid-token count must be positive")
    if effective_batch_size != 4:
        _fail("effective prompt batch must remain four")
    return -corrected_selected_objective.sum(
        dtype=corrected_selected_objective.dtype
    ) / float(prompt_valid_token_count * effective_batch_size)


def _fp64_route(
    route: Mapping[str, Any],
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
) -> CanonicalObjectiveResult:
    required = (
        "q_target_logprob",
        "p_old_target_logprob",
        "teacher_target_logprob",
        "correction_weight",
        "valid_mask",
    )
    if any(not isinstance(route.get(name), Tensor) for name in required):
        _fail("FP64 replay route lacks a tensor input")
    return canonical_corrected_objective(
        q_target_logprob=route["q_target_logprob"].detach(),
        p_old_target_logprob=route["p_old_target_logprob"].detach(),
        teacher_target_logprob=route["teacher_target_logprob"].detach(),
        correction_weight=route["correction_weight"].detach(),
        valid_mask=route["valid_mask"].detach(),
        prompt_ids=prompt_ids,
        group_ids=group_ids,
        beta=beta,
        clip_low=clip_low,
        clip_high=clip_high,
        accumulator_dtype=torch.float64,
        offline_replay=True,
    )


def canonical_fp64_replay_pair(
    *,
    legacy: Mapping[str, Any],
    balanced: Mapping[str, Any],
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
) -> dict[str, float]:
    """Replay both persisted routes and bound cross-path scalar propagation."""

    left = _fp64_route(
        legacy,
        prompt_ids=prompt_ids,
        group_ids=group_ids,
        beta=beta,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    right = _fp64_route(
        balanced,
        prompt_ids=prompt_ids,
        group_ids=group_ids,
        beta=beta,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    if not torch.equal(left.valid_mask, right.valid_mask):
        _fail("FP64 replay masks differ")
    absolute_token_delta = (
        left.corrected_selected_objective
        - right.corrected_selected_objective
    ).abs()
    bound_reduction = canonical_hierarchical_reduction(
        absolute_token_delta,
        left.valid_mask,
        prompt_ids=prompt_ids,
        group_ids=group_ids,
    )
    left_value = float(left.objective.detach())
    right_value = float(right.objective.detach())
    legacy_native = float(legacy["native_objective"])
    new_native = float(balanced["native_objective"])
    return {
        "legacy_native_objective": legacy_native,
        "legacy_canonical_fp64_objective": left_value,
        "new_native_objective": new_native,
        "new_canonical_fp64_objective": right_value,
        "legacy_native_minus_canonical": legacy_native - left_value,
        "new_native_minus_canonical": new_native - right_value,
        "canonical_cross_path_error": abs(left_value - right_value),
        "objective_error_bound": float(
            bound_reduction.batch_objective.detach()
        ),
    }


def locate_first_divergence(signals: Mapping[str, bool]) -> str:
    """Frozen P4.8f diagnostic decision tree."""

    if signals.get("token_identity_pass") is not True:
        return "identity_or_chunk_boundary"
    if signals.get("q_pass") is not True or signals.get(
        "frozen_scorer_pass"
    ) is not True:
        return "forward_or_scorer"
    if signals.get("per_token_objective_pass") is not True:
        return "formula_clip_or_dtype"
    if signals.get("hierarchical_reduction_pass") is not True:
        return "reduction_or_chunk_weighting"
    if signals.get("canonical_within_error_bound") is not True:
        return "semantic_error_exceeds_input_bound"
    if signals.get("native_scalar_pass") is not True:
        return "native_reduction_tree_or_accumulator"
    if signals.get("gradient_pass") is not True:
        return "backward_checkpoint_or_scaling"
    if signals.get("delta_pass") is not True:
        return "optimizer_clip_or_step_lifecycle"
    return "equivalent"


__all__ = [
    "B2ObjectiveReducerV2Error",
    "CanonicalHierarchicalReduction",
    "CanonicalObjectiveResult",
    "canonical_corrected_objective",
    "canonical_fp64_replay_pair",
    "canonical_hierarchical_reduction",
    "canonical_prompt_chunk_loss",
    "canonical_token_objective_from_advantage",
    "locate_first_divergence",
]
