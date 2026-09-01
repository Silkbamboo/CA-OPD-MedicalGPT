"""CPU-safe reconstruction telemetry for the P4.6 production qualification.

The builder accepts detached policy tensors and CPU parameter snapshots.  It
never imports a model runtime, touches CUDA, or attempts to recover evidence
from log strings.  Every returned value is a JSON primitive so the payload can
be validated and durably written while the live optimizer state still exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SCHEMA_VERSION = 6
SIGNED_DIFFERENCE_SEMANTICS = "p_old_minus_q"
RAW_IS_SEMANTICS = "exp(clamp(log_p_old_minus_log_q,-20,20))"


class ReconstructionTelemetryError(RuntimeError):
    """Raised when live reconstruction evidence is absent or inconsistent."""


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cpu64(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ReconstructionTelemetryError(f"{name} is not a tensor")
    numeric = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return numeric


def _finite_values(value: Tensor, mask: Tensor, *, name: str) -> Tensor:
    numeric = _cpu64(value, name=name)
    if numeric.shape != mask.shape:
        raise ReconstructionTelemetryError(f"{name} shape mismatch")
    selected = numeric[mask]
    if selected.numel() == 0:
        raise ReconstructionTelemetryError(f"{name} has no valid values")
    if not bool(torch.isfinite(selected).all()):
        raise ReconstructionTelemetryError(f"{name} valid values must be finite")
    return selected


def _summary(values: Tensor, quantiles: Sequence[float]) -> dict[str, float]:
    result: dict[str, float] = {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for quantile in quantiles:
        label = f"p{int(round(quantile * 100))}"
        result[label] = float(torch.quantile(values, quantile))
    return result


def _average_ranks(values: Tensor) -> Tensor:
    numeric = [float(item) for item in values]
    order = sorted(range(len(numeric)), key=numeric.__getitem__)
    ranks = [0.0] * len(order)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and numeric[order[end]] == numeric[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return torch.tensor(ranks, dtype=torch.float64)


def _correlation(left: Tensor, right: Tensor, *, name: str) -> float:
    if left.numel() < 2:
        raise ReconstructionTelemetryError(f"{name} correlation needs at least two values")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if not math.isfinite(float(denominator)) or float(denominator) == 0.0:
        raise ReconstructionTelemetryError(f"{name} correlation is undefined")
    return float(torch.sum(left_centered * right_centered) / denominator)


def _ess(values: Tensor) -> tuple[float, float]:
    denominator = float(torch.sum(values * values))
    if denominator <= 0.0:
        raise ReconstructionTelemetryError("importance weights have zero ESS denominator")
    ess = float(torch.sum(values)) ** 2 / denominator
    return ess, ess / int(values.numel())


def _group_rows(
    values: Tensor,
    mask: Tensor,
    labels: Sequence[str],
    *,
    reduce: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    result: dict[str, Any] = {}
    for label in sorted(groups):
        selected = torch.cat([values[row][mask[row]] for row in groups[label]])
        if reduce == "mean":
            result[label] = float(selected.mean())
        elif reduce == "ess":
            ess, fraction = _ess(selected)
            result[label] = {
                "token_count": int(selected.numel()),
                "ess": ess,
                "ess_fraction": fraction,
            }
        else:  # pragma: no cover - internal programming error
            raise AssertionError(reduce)
    return result


def _finite_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconstructionTelemetryError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ReconstructionTelemetryError(f"{name} must be finite")
    return numeric


def build_reconstruction_telemetry(
    *,
    run_id: str,
    step_id: str,
    rollout_logprobs: Tensor,
    old_logprobs: Tensor,
    current_pre_logprobs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    prompt_ids: Sequence[str],
    source_roles: Sequence[str],
    objective_before: float,
    objective_after: float,
    loss_before: float,
    loss_after: float,
    alignment: float,
    ppo_ratio_post: Tensor,
    gradient_norm_before_clip: float,
    gradient_norm_after_clip: float,
    before_parameters: Mapping[str, Tensor],
    after_parameters: Mapping[str, Tensor],
    loss_gradients: Mapping[str, Tensor],
    teacher_gradient_tensor_count: int,
    base_gradient_tensor_count: int,
    optimizer_config: Mapping[str, Any],
    near_zero_threshold: float,
    teacher_detached: bool = True,
    old_actor_detached: bool = True,
    correction_weight_detached: bool = True,
) -> dict[str, Any]:
    """Build complete step telemetry directly from live tensor evidence."""

    if not run_id or not step_id:
        raise ReconstructionTelemetryError("run_id and step_id must be non-empty")
    shapes = {
        tuple(rollout_logprobs.shape),
        tuple(old_logprobs.shape),
        tuple(current_pre_logprobs.shape),
        tuple(advantages.shape),
        tuple(response_mask.shape),
        tuple(ppo_ratio_post.shape),
    }
    if len(shapes) != 1 or response_mask.ndim != 2:
        raise ReconstructionTelemetryError("policy telemetry tensor shape mismatch")
    if len(prompt_ids) != response_mask.shape[0] or len(source_roles) != response_mask.shape[0]:
        raise ReconstructionTelemetryError("prompt/source shape mismatch")
    if len(set(str(value) for value in prompt_ids)) != len(prompt_ids):
        raise ReconstructionTelemetryError("prompt ids must be unique")
    mask_numeric = response_mask.detach().to(device="cpu")
    if not bool(torch.all((mask_numeric == 0) | (mask_numeric == 1))):
        raise ReconstructionTelemetryError("response mask must be binary")
    mask = mask_numeric.to(torch.bool)
    valid_count = int(mask.sum())
    if valid_count <= 0:
        raise ReconstructionTelemetryError("response mask has no valid token")

    q = _cpu64(rollout_logprobs, name="rollout_logprobs")
    old = _cpu64(old_logprobs, name="old_logprobs")
    current_pre = _cpu64(current_pre_logprobs, name="current_pre_logprobs")
    advantage_all = _cpu64(advantages, name="advantages")
    ratio_post_all = _cpu64(ppo_ratio_post, name="ppo_ratio_post")
    q_valid = _finite_values(q, mask, name="rollout_logprobs")
    old_valid = _finite_values(old, mask, name="old_logprobs")
    current_valid = _finite_values(current_pre, mask, name="current_pre_logprobs")
    advantage = _finite_values(advantage_all, mask, name="advantages")
    ratio_post = _finite_values(ratio_post_all, mask, name="ppo_ratio_post")

    log_w_all = old - q
    log_w = log_w_all[mask]
    absolute = log_w.abs()
    raw_all = torch.exp(torch.clamp(log_w_all, min=-20.0, max=20.0))
    raw = raw_all[mask]
    importance_cap = _finite_scalar(optimizer_config.get("importance_cap"), name="importance_cap")
    if importance_cap <= 0:
        raise ReconstructionTelemetryError("importance_cap must be positive")
    capped_all = torch.clamp(raw_all, max=importance_cap)
    capped = capped_all[mask]
    token_ess, ess_fraction = _ess(capped)

    clip_low = _finite_scalar(optimizer_config.get("ppo_clip_low"), name="ppo_clip_low")
    clip_high = _finite_scalar(optimizer_config.get("ppo_clip_high"), name="ppo_clip_high")
    if not 0 < clip_low <= 1 <= clip_high:
        raise ReconstructionTelemetryError("PPO clip bounds are invalid")
    ratio_pre_all = torch.exp(current_pre - old)
    ratio_pre = ratio_pre_all[mask]
    if not bool(torch.isfinite(ratio_pre).all()):
        raise ReconstructionTelemetryError("ppo_ratio_pre valid values must be finite")

    per_prompt_ess = _group_rows(capped_all, mask, prompt_ids, reduce="ess")
    per_source_ess = _group_rows(capped_all, mask, source_roles, reduce="ess")
    for evidence in per_prompt_ess.values():
        evidence["cap_fraction"] = 0.0
    for evidence in per_source_ess.values():
        evidence["cap_fraction"] = 0.0
    for labels, evidence_map in ((prompt_ids, per_prompt_ess), (source_roles, per_source_ess)):
        for row, label in enumerate(labels):
            row_raw = raw_all[row][mask[row]]
            if row_raw.numel():
                current = evidence_map[str(label)]
                capped_count = int((row_raw > importance_cap).sum())
                current.setdefault("_cap_count", 0)
                current["_cap_count"] += capped_count
        for evidence in evidence_map.values():
            evidence["cap_fraction"] = evidence.pop("_cap_count", 0) / evidence["token_count"]

    ppo_pre_summary = _summary(ratio_pre, (0.50, 0.95, 0.99))
    ppo_post_summary = _summary(ratio_post, (0.50, 0.95, 0.99))
    ppo_pre_summary["clip_fraction"] = float(
        ((ratio_pre < clip_low) | (ratio_pre > clip_high)).to(torch.float64).mean()
    )
    ppo_post_summary["clip_fraction"] = float(
        ((ratio_post < clip_low) | (ratio_post > clip_high)).to(torch.float64).mean()
    )

    threshold = _finite_scalar(near_zero_threshold, name="near_zero_threshold")
    if threshold < 0:
        raise ReconstructionTelemetryError("near_zero_threshold must be nonnegative")
    near = advantage.abs() <= threshold
    positive = advantage > threshold
    negative = advantage < -threshold

    parameter_names = tuple(sorted(str(name) for name in before_parameters))
    if not parameter_names or set(parameter_names) != set(after_parameters):
        raise ReconstructionTelemetryError("parameter snapshot keys are missing or mismatched")
    if set(parameter_names) != set(loss_gradients):
        raise ReconstructionTelemetryError("gradient keys do not match parameter snapshots")
    before_squared = 0.0
    delta_squared = 0.0
    gradient_dot_delta = 0.0
    update_norms: dict[str, float] = {}
    trainable_parameter_count = 0
    for name in parameter_names:
        before_value = _cpu64(before_parameters[name], name=f"before_parameters[{name}]")
        after_value = _cpu64(after_parameters[name], name=f"after_parameters[{name}]")
        gradient = _cpu64(loss_gradients[name], name=f"loss_gradients[{name}]")
        if before_value.shape != after_value.shape or before_value.shape != gradient.shape:
            raise ReconstructionTelemetryError("parameter/gradient shape mismatch")
        if not bool(
            torch.isfinite(before_value).all()
            and torch.isfinite(after_value).all()
            and torch.isfinite(gradient).all()
        ):
            raise ReconstructionTelemetryError("parameter telemetry must be finite")
        delta = after_value - before_value
        norm = float(torch.linalg.vector_norm(delta))
        update_norms[name] = norm
        before_squared += float(torch.sum(before_value * before_value))
        delta_squared += float(torch.sum(delta * delta))
        gradient_dot_delta += float(torch.sum(gradient * delta))
        trainable_parameter_count += int(before_value.numel())
    parameter_delta_norm = math.sqrt(delta_squared)
    relative_delta = parameter_delta_norm / max(math.sqrt(before_squared), 1e-12)
    nonzero_count = sum(norm > 0.0 for norm in update_norms.values())

    optimizer_payload = dict(optimizer_config)
    required_optimizer = {
        "name",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "ppo_clip_low",
        "ppo_clip_high",
        "importance_cap",
    }
    if set(optimizer_payload) != required_optimizer:
        raise ReconstructionTelemetryError("optimizer_config keys are incomplete or unknown")
    for key in required_optimizer - {"name"}:
        _finite_scalar(optimizer_payload[key], name=f"optimizer_config.{key}")
    if not isinstance(optimizer_payload["name"], str) or not optimizer_payload["name"]:
        raise ReconstructionTelemetryError("optimizer_config.name must be non-empty")

    objective_before_value = _finite_scalar(objective_before, name="objective_before")
    objective_after_value = _finite_scalar(objective_after, name="objective_after")
    loss_before_value = _finite_scalar(loss_before, name="loss_before")
    loss_after_value = _finite_scalar(loss_after, name="loss_after")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "step_id": step_id,
        "q_p_old": {
            "valid_token_count": valid_count,
            "prompt_count": len(prompt_ids),
            "source_count": len(set(str(value) for value in source_roles)),
            "signed_difference_semantics": SIGNED_DIFFERENCE_SEMANTICS,
            "signed_mean": float(log_w.mean()),
            "absolute_difference": {
                "mae": float(absolute.mean()),
                "p50": float(torch.quantile(absolute, 0.50)),
                "p95": float(torch.quantile(absolute, 0.95)),
                "p99": float(torch.quantile(absolute, 0.99)),
                "max": float(absolute.max()),
            },
            "pearson": _correlation(q_valid, old_valid, name="Pearson"),
            "spearman": _correlation(
                _average_ranks(q_valid), _average_ranks(old_valid), name="Spearman"
            ),
            "log_w": _summary(log_w, ()),
            "raw_is_semantics": RAW_IS_SEMANTICS,
            "raw_is": _summary(raw, (0.50, 0.95, 0.99)),
            "capped_is": _summary(capped, ()),
            "token_ess": token_ess,
            "ess_fraction": ess_fraction,
            "per_prompt_ess": per_prompt_ess,
            "per_source_ess": per_source_ess,
            "cap_fraction": float((raw > importance_cap).to(torch.float64).mean()),
            "finite_rate": 1.0,
            "current_pre_old_max_abs": float((current_valid - old_valid).abs().max()),
            "ppo_ratio_pre": ppo_pre_summary,
        },
        "advantage": {
            "count": valid_count,
            "mean": float(advantage.mean()),
            "std": float(advantage.std(unbiased=False)),
            "min": float(advantage.min()),
            "max": float(advantage.max()),
            "quantiles": {
                "p1": float(torch.quantile(advantage, 0.01)),
                "p5": float(torch.quantile(advantage, 0.05)),
                "p50": float(torch.quantile(advantage, 0.50)),
                "p95": float(torch.quantile(advantage, 0.95)),
                "p99": float(torch.quantile(advantage, 0.99)),
            },
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "near_zero_count": int(near.sum()),
            "near_zero_threshold": threshold,
            "finite_rate": 1.0,
            "per_prompt_mean": _group_rows(advantage_all, mask, prompt_ids, reduce="mean"),
            "per_source_mean": _group_rows(advantage_all, mask, source_roles, reduce="mean"),
            "aggregation": "valid_token_pooled",
            "teacher_detached": bool(teacher_detached),
            "old_actor_detached": bool(old_actor_detached),
        },
        "optimizer_update": {
            "objective_before": objective_before_value,
            "objective_after": objective_after_value,
            "objective_delta": objective_after_value - objective_before_value,
            "loss_before": loss_before_value,
            "loss_after": loss_after_value,
            "loss_delta": loss_after_value - loss_before_value,
            "alignment": _finite_scalar(alignment, name="alignment"),
            "ppo_ratio_pre": ppo_pre_summary,
            "ppo_ratio_post": ppo_post_summary,
            "clip_fraction_pre": ppo_pre_summary["clip_fraction"],
            "clip_fraction_post": ppo_post_summary["clip_fraction"],
            "gradient_norm_before_clip": _finite_scalar(
                gradient_norm_before_clip, name="gradient_norm_before_clip"
            ),
            "gradient_norm_after_clip": _finite_scalar(
                gradient_norm_after_clip, name="gradient_norm_after_clip"
            ),
            "parameter_delta_norm": parameter_delta_norm,
            "relative_parameter_delta": relative_delta,
            "gradient_dot_parameter_delta": gradient_dot_delta,
            "trainable_tensor_count": len(parameter_names),
            "trainable_parameter_count": trainable_parameter_count,
            "nonzero_update_tensor_count": nonzero_count,
            "zero_update_tensor_count": len(parameter_names) - nonzero_count,
            "update_norm_min": min(update_norms.values()),
            "update_norm_max": max(update_norms.values()),
            "teacher_gradient_tensor_count": int(teacher_gradient_tensor_count),
            "base_gradient_tensor_count": int(base_gradient_tensor_count),
            "correction_weight_detached": bool(correction_weight_detached),
            "optimizer_config": optimizer_payload,
            "optimizer_config_sha256": _canonical_sha(optimizer_payload),
        },
    }
    validate_reconstruction_telemetry(payload)
    return payload


def _require_fields(mapping: Mapping[str, Any], required: set[str], *, section: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ReconstructionTelemetryError(f"{section} missing fields: {missing}")


def _reject_nonfinite(value: Any, *, path: str = "payload") -> None:
    if value is None:
        raise ReconstructionTelemetryError(f"{path} is unavailable")
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ReconstructionTelemetryError(f"{path} must be finite")
        return
    if isinstance(value, str):
        if not value:
            raise ReconstructionTelemetryError(f"{path} must be non-empty")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _reject_nonfinite(child, path=f"{path}[{index}]")
        return
    raise ReconstructionTelemetryError(f"{path} is not JSON-compatible")


def validate_reconstruction_telemetry(payload: Mapping[str, Any]) -> None:
    """Fail closed unless every P4.6 reconstruction field and invariant is present."""

    _require_fields(
        payload,
        {"schema_version", "run_id", "step_id", "q_p_old", "advantage", "optimizer_update"},
        section="reconstruction",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ReconstructionTelemetryError("reconstruction schema version mismatch")
    q_p_old = payload["q_p_old"]
    advantage = payload["advantage"]
    update = payload["optimizer_update"]
    if not isinstance(q_p_old, Mapping) or not isinstance(advantage, Mapping) or not isinstance(update, Mapping):
        raise ReconstructionTelemetryError("reconstruction sections must be mappings")
    _require_fields(
        q_p_old,
        {
            "valid_token_count", "prompt_count", "source_count",
            "signed_difference_semantics", "signed_mean", "absolute_difference",
            "pearson", "spearman", "log_w", "raw_is_semantics", "raw_is",
            "capped_is", "token_ess", "ess_fraction", "per_prompt_ess",
            "per_source_ess", "cap_fraction", "finite_rate",
            "current_pre_old_max_abs", "ppo_ratio_pre",
        },
        section="q_p_old",
    )
    _require_fields(
        advantage,
        {
            "count", "mean", "std", "min", "max", "quantiles",
            "positive_count", "negative_count", "near_zero_count",
            "near_zero_threshold", "finite_rate", "per_prompt_mean",
            "per_source_mean", "aggregation", "teacher_detached", "old_actor_detached",
        },
        section="advantage",
    )
    _require_fields(
        update,
        {
            "objective_before", "objective_after", "objective_delta",
            "loss_before", "loss_after", "loss_delta", "alignment",
            "ppo_ratio_pre", "ppo_ratio_post", "clip_fraction_pre",
            "clip_fraction_post", "gradient_norm_before_clip",
            "gradient_norm_after_clip", "parameter_delta_norm",
            "relative_parameter_delta", "gradient_dot_parameter_delta",
            "trainable_tensor_count", "trainable_parameter_count",
            "nonzero_update_tensor_count", "zero_update_tensor_count",
            "update_norm_min", "update_norm_max", "teacher_gradient_tensor_count",
            "base_gradient_tensor_count", "correction_weight_detached",
            "optimizer_config", "optimizer_config_sha256",
        },
        section="optimizer_update",
    )
    _require_fields(
        q_p_old["absolute_difference"], {"mae", "p50", "p95", "p99", "max"},
        section="q_p_old.absolute_difference",
    )
    _require_fields(q_p_old["log_w"], {"mean", "std", "min", "max"}, section="q_p_old.log_w")
    _require_fields(
        q_p_old["raw_is"], {"mean", "std", "min", "max", "p50", "p95", "p99"},
        section="q_p_old.raw_is",
    )
    _require_fields(q_p_old["capped_is"], {"mean", "std", "min", "max"}, section="q_p_old.capped_is")
    _require_fields(advantage["quantiles"], {"p1", "p5", "p50", "p95", "p99"}, section="advantage.quantiles")
    _reject_nonfinite(payload)

    valid_count = q_p_old["valid_token_count"]
    if not isinstance(valid_count, int) or isinstance(valid_count, bool) or valid_count <= 0:
        raise ReconstructionTelemetryError("valid token count is invalid")
    if advantage["count"] != valid_count:
        raise ReconstructionTelemetryError("advantage count differs from valid token count")
    if (
        advantage["positive_count"]
        + advantage["negative_count"]
        + advantage["near_zero_count"]
        != valid_count
    ):
        raise ReconstructionTelemetryError("advantage counts do not add to valid token count")
    if (
        update["nonzero_update_tensor_count"] + update["zero_update_tensor_count"]
        != update["trainable_tensor_count"]
    ):
        raise ReconstructionTelemetryError("update tensor counts do not add to trainable tensor count")
    if sum(value["token_count"] for value in q_p_old["per_prompt_ess"].values()) != valid_count:
        raise ReconstructionTelemetryError("per-prompt token counts do not add to valid token count")
    if sum(value["token_count"] for value in q_p_old["per_source_ess"].values()) != valid_count:
        raise ReconstructionTelemetryError("per-source token counts do not add to valid token count")
    if not advantage["teacher_detached"] or not advantage["old_actor_detached"]:
        raise ReconstructionTelemetryError("Teacher and old actor must be detached")
    if not update["correction_weight_detached"]:
        raise ReconstructionTelemetryError("correction weight must be detached")
    if q_p_old["finite_rate"] != 1.0 or advantage["finite_rate"] != 1.0:
        raise ReconstructionTelemetryError("reconstruction finite rate must be one")
    if update["optimizer_config_sha256"] != _canonical_sha(update["optimizer_config"]):
        raise ReconstructionTelemetryError("optimizer config SHA mismatch")


__all__ = [
    "RAW_IS_SEMANTICS",
    "ReconstructionTelemetryError",
    "SIGNED_DIFFERENCE_SEMANTICS",
    "build_reconstruction_telemetry",
    "validate_reconstruction_telemetry",
]
