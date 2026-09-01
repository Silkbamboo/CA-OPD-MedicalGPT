"""Frozen PG-OPD sampled-token math for same-trajectory distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.opd.rollout_correction_adapter import (
    LOG_WEIGHT_SAFETY_BOUND,
    RolloutCorrectionAdapterError,
    native_decoupled_token_is,
)
from src.opd.rollout_probability import (
    RolloutProbabilityError,
    validate_rollout_behavior_provenance,
)
from src.opd.production_b2_objective_reducer_v2 import (
    canonical_corrected_objective,
)


class PGOPDContractError(RuntimeError):
    pass


THREE_POLICY_ARTIFACT_PROTOCOL_VERSION = "p4.3-three-policy-correction-v3"


def _same_shape(*values: Tensor) -> None:
    if len({tuple(value.shape) for value in values}) != 1:
        raise PGOPDContractError("same-trajectory tensors must have identical shapes")


def same_trajectory_advantage(
    old_student_logprob: Tensor, teacher_logprob: Tensor, *, beta: float
) -> Tensor:
    if beta <= 0:
        raise PGOPDContractError("beta must be positive")
    _same_shape(old_student_logprob, teacher_logprob)
    # Both distributions are immutable evidence; the policy gradient only flows
    # through the separately supplied new_student_logprob.
    return float(beta) * (teacher_logprob.detach() - old_student_logprob.detach())


def reverse_kl_estimator(old_student_logprob: Tensor, teacher_logprob: Tensor) -> Tensor:
    _same_shape(old_student_logprob, teacher_logprob)
    return old_student_logprob.detach() - teacher_logprob.detach()


def ppo_ratio(new_student_logprob: Tensor, old_student_logprob: Tensor) -> Tensor:
    _same_shape(new_student_logprob, old_student_logprob)
    if old_student_logprob.requires_grad:
        raise PGOPDContractError("old_student_logprob must be frozen")
    return torch.exp(new_student_logprob - old_student_logprob)


@dataclass(frozen=True)
class GroupedReductionResult:
    trajectory_means: Tensor
    per_group: dict[tuple[str, str], Tensor]
    per_prompt: dict[str, Tensor]
    batch_mean: Tensor


def grouped_reduction(
    values: Tensor,
    response_mask: Tensor,
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
) -> GroupedReductionResult:
    _same_shape(values, response_mask)
    if values.ndim != 2:
        raise PGOPDContractError("trajectory values must be [batch, response]")
    batch = values.shape[0]
    if len(prompt_ids) != batch or len(group_ids) != batch:
        raise PGOPDContractError("prompt/group identity length mismatch")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise PGOPDContractError("response mask must be binary")
    if not torch.isfinite(values).all():
        raise PGOPDContractError("trajectory values must be finite")
    mask = response_mask.to(values.dtype)
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise PGOPDContractError("trajectory has no valid response token")
    trajectory_means = (values * mask).sum(dim=1) / counts

    group_values: dict[tuple[str, str], list[Tensor]] = {}
    for index, (prompt, group) in enumerate(zip(prompt_ids, group_ids, strict=True)):
        group_values.setdefault((str(prompt), str(group)), []).append(trajectory_means[index])
    per_group = {
        identity: torch.stack(items).mean() for identity, items in group_values.items()
    }
    prompt_values: dict[str, list[Tensor]] = {}
    for (prompt, _group), group_mean in per_group.items():
        prompt_values.setdefault(prompt, []).append(group_mean)
    per_prompt = {
        prompt: torch.stack(groups).mean() for prompt, groups in prompt_values.items()
    }
    if not per_prompt:
        raise PGOPDContractError("no prompts to reduce")
    return GroupedReductionResult(
        trajectory_means=trajectory_means,
        per_group=per_group,
        per_prompt=per_prompt,
        batch_mean=torch.stack(list(per_prompt.values())).mean(),
    )


def grouped_trajectory_mean(
    values: Tensor,
    response_mask: Tensor,
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
) -> Tensor:
    return grouped_reduction(
        values,
        response_mask,
        prompt_ids=prompt_ids,
        group_ids=group_ids,
    ).batch_mean


def masked_numeric_summary(values: Tensor, response_mask: Tensor) -> dict[str, float]:
    _same_shape(values, response_mask)
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise PGOPDContractError("response mask must be binary")
    valid = response_mask.to(torch.bool)
    if not bool(valid.any()):
        raise PGOPDContractError("numeric summary has no valid response token")
    selected = values.detach()[valid].to(dtype=torch.float64)
    if not bool(torch.isfinite(selected).all()):
        raise PGOPDContractError("numeric summary values must be finite")
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std(unbiased=False)),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "p50": float(torch.quantile(selected, 0.50)),
        "p95": float(torch.quantile(selected, 0.95)),
        "p99": float(torch.quantile(selected, 0.99)),
    }


@dataclass(frozen=True)
class ThreePolicyLogProbBundle:
    """Same-token log probabilities with explicit three-policy semantics."""

    rollout_behavior_logprob: Tensor
    old_actor_logprob: Tensor
    current_actor_logprob: Tensor
    teacher_logprob: Tensor
    response_mask: Tensor
    behavior_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class RolloutCorrectionResult:
    """Detached token importance weights and auditable distribution metrics."""

    log_weight: Tensor
    clamped_log_weight: Tensor
    raw_weight: Tensor
    truncated_weight: Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class DecoupledCorrectedLossResult:
    """P4.3 objective with PPO and rollout ratios kept separate."""

    loss: Tensor
    surrogate: Tensor
    token_surrogate: Tensor
    advantage: Tensor
    ppo_ratio: Tensor
    ppo_log_ratio: Tensor
    ppo_clip_fraction: float
    correction: RolloutCorrectionResult
    reduction: GroupedReductionResult


def validate_three_policy_bundle(
    bundle: ThreePolicyLogProbBundle,
    *,
    require_pre_update_identity: bool = False,
    identity_tolerance: float = 1e-4,
) -> None:
    """Fail closed on mixed semantics, gradients, nonfinite values, or identity drift."""

    _same_shape(
        bundle.rollout_behavior_logprob,
        bundle.old_actor_logprob,
        bundle.current_actor_logprob,
        bundle.teacher_logprob,
        bundle.response_mask,
    )
    if bundle.rollout_behavior_logprob.ndim != 2:
        raise PGOPDContractError("three-policy tensors must be [batch, response]")
    if identity_tolerance < 0:
        raise PGOPDContractError("identity tolerance must be non-negative")
    if not isinstance(bundle.behavior_provenance, Mapping) or not bundle.behavior_provenance:
        raise PGOPDContractError("behavior provenance is required")
    try:
        validate_rollout_behavior_provenance(bundle.behavior_provenance)
    except RolloutProbabilityError as exc:
        raise PGOPDContractError(f"behavior provenance is invalid: {exc}") from exc
    semantics = bundle.behavior_provenance.get("score_semantics")
    if semantics != "normalized_behavior_logprob":
        if (
            bundle.behavior_provenance.get("score_source") == "generate.scores"
            and semantics == "raw_actor_logprob"
        ):
            raise PGOPDContractError(
                "processed generation scores cannot be labeled as raw actor logprob"
            )
        raise PGOPDContractError("behavior score semantics must be normalized behavior logprob")
    if bundle.rollout_behavior_logprob.requires_grad:
        raise PGOPDContractError("rollout behavior logprob must be frozen")
    if bundle.old_actor_logprob.requires_grad:
        raise PGOPDContractError("old actor logprob must be frozen")
    if bundle.teacher_logprob.requires_grad:
        raise PGOPDContractError("Teacher logprob must be frozen")
    if not bundle.current_actor_logprob.requires_grad:
        raise PGOPDContractError("current actor logprob must carry policy gradient")
    if not torch.all((bundle.response_mask == 0) | (bundle.response_mask == 1)):
        raise PGOPDContractError("response mask must be binary")
    valid = bundle.response_mask.to(torch.bool)
    if not bool(valid.any()):
        raise PGOPDContractError("three-policy batch has no valid response token")
    for name, value in (
        ("rollout behavior", bundle.rollout_behavior_logprob),
        ("old actor", bundle.old_actor_logprob),
        ("current actor", bundle.current_actor_logprob),
        ("Teacher", bundle.teacher_logprob),
    ):
        if not bool(torch.isfinite(value[valid]).all()):
            raise PGOPDContractError(f"{name} logprob must be finite")
    if require_pre_update_identity:
        maximum = float(
            (
                bundle.current_actor_logprob.detach()[valid]
                - bundle.old_actor_logprob.detach()[valid]
            )
            .abs()
            .max()
        )
        if maximum > identity_tolerance:
            raise PGOPDContractError(
                "current actor and old actor fail pre-update identity tolerance"
            )


def _effective_sample_size(weights: Tensor) -> float:
    weights64 = weights.detach().to(dtype=torch.float64)
    denominator = weights64.square().sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise PGOPDContractError("importance weights have zero or nonfinite ESS denominator")
    return float(weights64.sum().square() / denominator)


def _weight_group_metrics(weights: Tensor) -> dict[str, Any]:
    if weights.numel() <= 0:
        raise PGOPDContractError("importance metric group is empty")
    ones = torch.ones_like(weights, dtype=torch.bool)
    summary = masked_numeric_summary(weights.reshape(1, -1), ones.reshape(1, -1))
    ess = _effective_sample_size(weights)
    return {**summary, "ess": ess, "ess_fraction": ess / float(weights.numel())}


def _flat_numeric_summary(values: Tensor) -> dict[str, float]:
    if values.numel() <= 0:
        raise PGOPDContractError("numeric metric group is empty")
    ones = torch.ones_like(values, dtype=torch.bool)
    return masked_numeric_summary(values.reshape(1, -1), ones.reshape(1, -1))


def _partition_correction_metrics(
    log_weight: Tensor,
    raw_weight: Tensor,
    truncated_weight: Tensor,
    response_mask: Tensor,
    identities: Sequence[str],
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    if len(identities) != truncated_weight.shape[0]:
        raise PGOPDContractError("importance metric identity length mismatch")
    output: dict[str, dict[str, Any]] = {}
    for identity in dict.fromkeys(str(value) for value in identities):
        rows = [index for index, value in enumerate(identities) if str(value) == identity]
        selected_mask = response_mask[rows].to(torch.bool)
        selected_log = log_weight[rows][selected_mask]
        selected_raw = raw_weight[rows][selected_mask]
        selected_truncated = truncated_weight[rows][selected_mask]
        # Keep the capped-weight summary at the top level for the compact
        # readiness contract, while retaining every probability diagnostic.
        output[identity] = {
            **_weight_group_metrics(selected_truncated),
            "valid_token_count": int(selected_truncated.numel()),
            "rollout_actor_log_ratio": _flat_numeric_summary(selected_log),
            "raw_is_weight": _weight_group_metrics(selected_raw),
            "truncated_is_weight": _weight_group_metrics(selected_truncated),
            "cap_fraction": float(
                (selected_raw > float(threshold)).to(torch.float64).mean()
            ),
        }
    return output


def rollout_importance_correction(
    *,
    old_actor_logprob: Tensor,
    rollout_behavior_logprob: Tensor,
    response_mask: Tensor,
    threshold: float,
    prompt_ids: Sequence[str] | None = None,
    source_roles: Sequence[str] | None = None,
) -> RolloutCorrectionResult:
    """Compute detached token TIS weights ``min(exp(p_old - q), threshold)``."""

    _same_shape(old_actor_logprob, rollout_behavior_logprob, response_mask)
    if old_actor_logprob.ndim != 2:
        raise PGOPDContractError("rollout correction tensors must be [batch, response]")
    if threshold <= 0 or not torch.isfinite(torch.tensor(float(threshold))):
        raise PGOPDContractError("rollout IS threshold must be finite and positive")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise PGOPDContractError("response mask must be binary")
    valid = response_mask.to(torch.bool)
    if not bool(valid.any()):
        raise PGOPDContractError("rollout correction has no valid response token")
    for name, value in (
        ("old actor", old_actor_logprob),
        ("rollout behavior", rollout_behavior_logprob),
    ):
        if not bool(torch.isfinite(value.detach()[valid]).all()):
            raise PGOPDContractError(f"{name} logprob must be finite")

    log_weight = (
        old_actor_logprob.detach() - rollout_behavior_logprob.detach()
    ).detach()
    clamped_log_weight = torch.clamp(
        log_weight,
        min=-LOG_WEIGHT_SAFETY_BOUND,
        max=LOG_WEIGHT_SAFETY_BOUND,
    ).detach()
    raw_weight = torch.exp(clamped_log_weight).detach()
    try:
        truncated_weight = native_decoupled_token_is(
            log_weight,
            response_mask,
            threshold=float(threshold),
        )
    except RolloutCorrectionAdapterError as exc:
        raise PGOPDContractError(str(exc)) from exc
    expected = torch.minimum(raw_weight, torch.full_like(raw_weight, float(threshold)))
    expected = torch.where(valid, expected, torch.zeros_like(expected)).detach()
    if not torch.equal(truncated_weight, expected):
        raise PGOPDContractError("native veRL token-IS result violates frozen P4.3 semantics")

    valid_raw = raw_weight[valid]
    valid_truncated = truncated_weight[valid]
    ess = _effective_sample_size(valid_truncated)
    cap_fraction = float((valid_raw > float(threshold)).to(torch.float64).mean())
    token_pooled = {
        "rollout_actor_log_ratio": masked_numeric_summary(log_weight, response_mask),
        "raw_is_weight": masked_numeric_summary(raw_weight, response_mask),
        "truncated_is_weight": masked_numeric_summary(truncated_weight, response_mask),
        "ess": ess,
        "ess_fraction": ess / float(valid_truncated.numel()),
        "cap_fraction": cap_fraction,
    }
    metrics: dict[str, Any] = {
        "protocol_version": THREE_POLICY_ARTIFACT_PROTOCOL_VERSION,
        "native_backend": "verl-0.8.0",
        "rollout_is_mode": "token",
        "rollout_is_threshold": float(threshold),
        "batch_normalize": False,
        "lower_truncation": False,
        "valid_token_count": int(valid.sum()),
        "ess": ess,
        "ess_fraction": ess / float(valid_truncated.numel()),
        "cap_fraction": cap_fraction,
        "token_pooled": token_pooled,
    }
    if prompt_ids is not None:
        per_prompt = _partition_correction_metrics(
            log_weight,
            raw_weight,
            truncated_weight,
            response_mask,
            prompt_ids,
            threshold=float(threshold),
        )
        metrics["per_prompt"] = per_prompt
        scalar_keys = ("mean", "std", "min", "max", "p50", "p95", "p99", "ess_fraction")
        metrics["prompt_equal"] = {
            key: sum(float(item[key]) for item in per_prompt.values()) / len(per_prompt)
            for key in scalar_keys
        }
        metrics["prompt_equal"]["cap_fraction"] = sum(
            float(item["cap_fraction"]) for item in per_prompt.values()
        ) / len(per_prompt)
        for name in (
            "rollout_actor_log_ratio",
            "raw_is_weight",
            "truncated_is_weight",
        ):
            metrics["prompt_equal"][name] = {
                key: sum(float(item[name][key]) for item in per_prompt.values())
                / len(per_prompt)
                for key in ("mean", "std", "min", "max", "p50", "p95", "p99")
            }
    else:
        metrics["per_prompt"] = {}
        metrics["prompt_equal"] = {}
    metrics["per_source"] = (
        _partition_correction_metrics(
            log_weight,
            raw_weight,
            truncated_weight,
            response_mask,
            source_roles,
            threshold=float(threshold),
        )
        if source_roles is not None
        else {}
    )
    return RolloutCorrectionResult(
        log_weight=log_weight,
        clamped_log_weight=clamped_log_weight,
        raw_weight=raw_weight,
        truncated_weight=truncated_weight,
        metrics=metrics,
    )


@dataclass(frozen=True)
class PPOLossResult:
    loss: Tensor
    ratio_mean: float
    clip_fraction: float
    valid_tokens: int
    surrogate: Tensor
    ratio: Tensor
    log_ratio: Tensor
    active_clip_mask: Tensor
    token_surrogate: Tensor


@dataclass(frozen=True)
class FrozenPGUpdateAudit:
    objective_before: float
    objective_after: float
    objective_improvement: float
    loss_before: float
    loss_after: float
    alignment: float
    active_clip_fraction_after: float
    alignment_required: bool
    objective_improved: bool
    alignment_passed: bool
    subgroup_direction_passed: bool
    positive_advantage_logprob_change_mean: float | None
    negative_advantage_logprob_change_mean: float | None
    hard_gate_passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PreUpdateAudit:
    passed: bool
    valid_tokens: int
    ratio_mean: float
    ratio_std: float
    ratio_min: float
    ratio_max: float
    ratio_max_abs_error: float
    log_ratio_max_abs: float


def validate_pre_update_contract(
    *,
    new_student_logprob: Tensor,
    old_student_logprob: Tensor,
    teacher_logprob: Tensor,
    advantage: Tensor,
    response_mask: Tensor,
    beta: float,
    max_abs_log_ratio: float,
    evidence_tolerance: float = 0.0,
) -> PreUpdateAudit:
    """Fail closed unless frozen evidence and the initial policy are identical."""

    _same_shape(
        new_student_logprob,
        old_student_logprob,
        teacher_logprob,
        advantage,
        response_mask,
    )
    if beta <= 0 or max_abs_log_ratio < 0 or evidence_tolerance < 0:
        raise PGOPDContractError("pre-update tolerances and beta are invalid")
    if old_student_logprob.requires_grad:
        raise PGOPDContractError("old Student logprob must be frozen")
    if teacher_logprob.requires_grad:
        raise PGOPDContractError("Teacher logprob must be frozen")
    if advantage.requires_grad:
        raise PGOPDContractError("advantage must be stop-gradient")
    if not new_student_logprob.requires_grad:
        raise PGOPDContractError("pre-update new Student logprob must carry policy gradient")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise PGOPDContractError("response mask must be binary")
    valid = response_mask.to(torch.bool)
    if not bool(valid.any()):
        raise PGOPDContractError("pre-update batch has no valid response token")
    for name, value in (
        ("new Student", new_student_logprob),
        ("old Student", old_student_logprob),
        ("Teacher", teacher_logprob),
        ("advantage", advantage),
    ):
        if not torch.isfinite(value[valid]).all():
            raise PGOPDContractError(f"{name} values must be finite")
    expected_advantage = float(beta) * (teacher_logprob - old_student_logprob)
    if not torch.allclose(
        advantage[valid],
        expected_advantage[valid],
        atol=evidence_tolerance,
        rtol=0.0,
    ):
        raise PGOPDContractError("advantage does not match frozen Teacher/old evidence")

    log_ratio = new_student_logprob - old_student_logprob
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio[valid]).all():
        raise PGOPDContractError("pre-update ratio must be finite")
    valid_log_ratio = log_ratio.detach()[valid]
    log_ratio_max_abs = float(valid_log_ratio.abs().max())
    if log_ratio_max_abs > max_abs_log_ratio:
        raise PGOPDContractError("pre-update ratio is not one within frozen tolerance")
    valid_ratio = ratio.detach()[valid]
    return PreUpdateAudit(
        passed=True,
        valid_tokens=int(valid.sum()),
        ratio_mean=float(valid_ratio.mean()),
        ratio_std=float(valid_ratio.std(unbiased=False)),
        ratio_min=float(valid_ratio.min()),
        ratio_max=float(valid_ratio.max()),
        ratio_max_abs_error=float((valid_ratio - 1.0).abs().max()),
        log_ratio_max_abs=log_ratio_max_abs,
    )


def ppo_clipped_objective(
    new_student_logprob: Tensor,
    old_student_logprob: Tensor,
    advantage: Tensor,
    response_mask: Tensor,
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    clip_low: float,
    clip_high: float,
) -> PPOLossResult:
    _same_shape(new_student_logprob, old_student_logprob, advantage, response_mask)
    if advantage.requires_grad:
        raise PGOPDContractError("advantage must be stop-gradient")
    if clip_low < 0 or clip_high < 0:
        raise PGOPDContractError("PPO clip bounds must be non-negative")
    valid = response_mask.to(torch.bool)
    for name, value in (
        ("new_student_logprob", new_student_logprob),
        ("old_student_logprob", old_student_logprob),
        ("advantage", advantage),
    ):
        if not torch.isfinite(value[valid]).all():
            raise PGOPDContractError(f"{name} must be finite on valid response tokens")
    ratio = ppo_ratio(new_student_logprob, old_student_logprob)
    if not torch.isfinite(ratio[valid]).all():
        raise PGOPDContractError("PPO ratio must be finite on valid response tokens")
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    unclipped_objective = ratio * advantage
    clipped_objective = clipped * advantage
    objective = torch.minimum(unclipped_objective, clipped_objective)
    active_clip = clipped_objective < unclipped_objective
    objective = torch.where(valid, objective, torch.zeros_like(objective))
    loss = -grouped_trajectory_mean(
        objective, response_mask, prompt_ids=prompt_ids, group_ids=group_ids
    )
    if not torch.isfinite(loss):
        raise PGOPDContractError("PPO loss must be finite")
    return PPOLossResult(
        loss=loss,
        ratio_mean=float(ratio.detach()[valid].mean()),
        clip_fraction=float(active_clip.detach()[valid].float().mean()),
        valid_tokens=int(valid.sum()),
        surrogate=-loss,
        ratio=ratio,
        log_ratio=new_student_logprob - old_student_logprob,
        active_clip_mask=active_clip,
        token_surrogate=objective,
    )


def decoupled_corrected_objective(
    bundle: ThreePolicyLogProbBundle,
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
    rollout_is_threshold: float,
    source_roles: Sequence[str] | None = None,
    advantage_scale: Tensor | float | None = None,
) -> DecoupledCorrectedLossResult:
    """Evaluate the frozen P4.3 three-policy corrected PG-OPD objective."""

    validate_three_policy_bundle(bundle)
    if source_roles is not None and len(source_roles) != bundle.response_mask.shape[0]:
        raise PGOPDContractError("source role length mismatch")
    # The frozen corrected objective has one canonical FP32 tensor/reduction
    # implementation.  BF16 scorer outputs are cast here while the q cast
    # remains in the training graph; all frozen inputs remain detached.
    behavior_fp32 = bundle.rollout_behavior_logprob.detach().to(torch.float32)
    old_fp32 = bundle.old_actor_logprob.detach().to(torch.float32)
    current_fp32 = bundle.current_actor_logprob.to(torch.float32)
    teacher_fp32 = bundle.teacher_logprob.detach().to(torch.float32)
    correction = rollout_importance_correction(
        old_actor_logprob=old_fp32,
        rollout_behavior_logprob=behavior_fp32,
        response_mask=bundle.response_mask,
        threshold=rollout_is_threshold,
        prompt_ids=prompt_ids,
        source_roles=source_roles,
    )
    canonical = canonical_corrected_objective(
        q_target_logprob=current_fp32,
        p_old_target_logprob=old_fp32,
        teacher_target_logprob=teacher_fp32,
        correction_weight=correction.truncated_weight,
        valid_mask=bundle.response_mask,
        prompt_ids=prompt_ids,
        group_ids=group_ids,
        beta=beta,
        clip_low=clip_low,
        clip_high=clip_high,
        advantage_scale=advantage_scale,
    )
    reduction = GroupedReductionResult(
        trajectory_means=canonical.reduction.trajectory_means,
        per_group=canonical.reduction.per_group,
        per_prompt=canonical.reduction.per_prompt,
        batch_mean=canonical.objective,
    )
    loss = canonical.loss
    if not bool(torch.isfinite(loss)):
        raise PGOPDContractError("corrected PPO loss must be finite")
    valid = canonical.valid_mask
    clip_fraction = float(
        canonical.clip_boundary_mask.detach()[valid].to(torch.float32).mean()
    )
    return DecoupledCorrectedLossResult(
        loss=loss,
        surrogate=canonical.objective,
        token_surrogate=canonical.corrected_selected_objective,
        advantage=canonical.scaled_advantage,
        ppo_ratio=canonical.raw_ppo_ratio,
        ppo_log_ratio=current_fp32 - old_fp32,
        ppo_clip_fraction=clip_fraction,
        correction=correction,
        reduction=reduction,
    )


def audit_frozen_pg_update(
    *,
    before: PPOLossResult,
    after: PPOLossResult,
    advantage: Tensor,
    delta_logprob: Tensor,
    response_mask: Tensor,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    objective_tolerance: float,
    alignment_tolerance: float,
    max_clip_fraction_for_alignment: float,
    subgroup_near_zero_tolerance: float = 1e-6,
) -> FrozenPGUpdateAudit:
    """Evaluate one shared-parameter step on the exact frozen training contract."""

    _same_shape(advantage, delta_logprob, response_mask)
    if advantage.requires_grad or delta_logprob.requires_grad:
        raise PGOPDContractError("frozen update audit tensors must be stop-gradient")
    if objective_tolerance < 0 or alignment_tolerance < 0:
        raise PGOPDContractError("update audit tolerances must be non-negative")
    if not 0 <= max_clip_fraction_for_alignment <= 1:
        raise PGOPDContractError("clip fraction threshold must be in [0, 1]")

    objective_before = float(before.surrogate.detach())
    objective_after = float(after.surrogate.detach())
    loss_before = float(before.loss.detach())
    loss_after = float(after.loss.detach())
    scalar_values = (objective_before, objective_after, loss_before, loss_after)
    if not all(torch.isfinite(torch.tensor(value)) for value in scalar_values):
        raise PGOPDContractError("pre/post objective and loss must be finite")

    alignment = float(
        grouped_trajectory_mean(
            advantage * delta_logprob,
            response_mask,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
        ).detach()
    )
    improvement = objective_after - objective_before
    objective_improved = improvement > objective_tolerance
    alignment_required = after.clip_fraction <= max_clip_fraction_for_alignment
    alignment_passed = alignment > alignment_tolerance

    valid = response_mask.to(torch.bool)
    valid_advantage = advantage.detach()[valid]
    valid_delta = delta_logprob.detach()[valid]
    positive = valid_delta[valid_advantage > subgroup_near_zero_tolerance]
    negative = valid_delta[valid_advantage < -subgroup_near_zero_tolerance]
    positive_mean = float(positive.mean()) if positive.numel() else None
    negative_mean = float(negative.mean()) if negative.numel() else None
    subgroup_passed = bool(
        positive.numel()
        and negative.numel()
        and positive_mean is not None
        and positive_mean > 0
        and negative_mean is not None
        and negative_mean < 0
    )

    failures: list[str] = []
    if not objective_improved:
        failures.append("formal_surrogate_not_improved")
    if not alignment_required:
        failures.append("significant_active_clipping")
    elif not alignment_passed:
        failures.append("first_order_alignment_not_positive")
    return FrozenPGUpdateAudit(
        objective_before=objective_before,
        objective_after=objective_after,
        objective_improvement=improvement,
        loss_before=loss_before,
        loss_after=loss_after,
        alignment=alignment,
        active_clip_fraction_after=after.clip_fraction,
        alignment_required=alignment_required,
        objective_improved=objective_improved,
        alignment_passed=alignment_passed,
        subgroup_direction_passed=subgroup_passed,
        positive_advantage_logprob_change_mean=positive_mean,
        negative_advantage_logprob_change_mean=negative_mean,
        hard_gate_passed=not failures,
        failure_reasons=tuple(failures),
    )
