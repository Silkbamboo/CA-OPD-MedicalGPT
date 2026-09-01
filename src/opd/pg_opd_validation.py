"""P4.2 optimizer, artifact-integrity, readiness and refresh validation.

This module is CPU-importable and contains no model-runtime dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


PROTOCOL_ID = "pg_opd_frozen_update_validation_v2"
OBJECTIVE_TOLERANCE = 1e-6
PRE_LOG_RATIO_TOLERANCE = 1e-4
SIGNIFICANT_ACTIVE_CLIP_FRACTION = 0.05


class PGUpdateValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerUpdateAudit:
    gradient_norm: float
    parameter_delta_norm: float
    relative_parameter_delta: float
    gradient_dot_parameter_delta: float
    trainable_parameter_count: int
    module_update_norms: dict[str, float]
    optimizer_direction_is_hard_gate: bool
    hard_gate_passed: bool
    failure_reasons: tuple[str, ...]


def _tensor_l2(values: Sequence[Tensor]) -> float:
    total = sum(
        float(torch.sum(value.detach().to(dtype=torch.float64) ** 2)) for value in values
    )
    return math.sqrt(total)


def _average_ranks(values: Tensor) -> Tensor:
    numeric = [float(value) for value in values.detach().to(dtype=torch.float64).cpu()]
    order = sorted(range(len(numeric)), key=numeric.__getitem__)
    ranks = [0.0] * len(numeric)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and numeric[order[end]] == numeric[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return torch.tensor(ranks, dtype=torch.float64)


def _correlation(left: Tensor, right: Tensor) -> float | None:
    if left.numel() < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if float(denominator) == 0.0:
        return None
    return float(torch.sum(left_centered * right_centered) / denominator)


def _policy_values_summary(values: Tensor) -> dict[str, Any]:
    numeric = values.detach().to(dtype=torch.float64).cpu()
    finite = torch.isfinite(numeric)
    if not bool(finite.all()):
        raise PGUpdateValidationError("three-policy identity evidence must be finite")
    return {
        "token_count": int(numeric.numel()),
        "finite_count": int(finite.sum()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(unbiased=False)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p50": float(torch.quantile(numeric, 0.50)),
        "p95": float(torch.quantile(numeric, 0.95)),
        "p99": float(torch.quantile(numeric, 0.99)),
    }


def _policy_pair_summary(
    left: Tensor, right: Tensor, *, tolerance: float
) -> dict[str, Any]:
    left_numeric = left.detach().to(dtype=torch.float64).cpu()
    right_numeric = right.detach().to(dtype=torch.float64).cpu()
    if left_numeric.numel() == 0 or left_numeric.shape != right_numeric.shape:
        raise PGUpdateValidationError("three-policy pair is empty or shape-mismatched")
    if not bool(torch.isfinite(left_numeric).all() and torch.isfinite(right_numeric).all()):
        raise PGUpdateValidationError("three-policy identity evidence must be finite")
    difference = left_numeric - right_numeric
    absolute = difference.abs()
    return {
        "token_count": int(difference.numel()),
        "finite_count": int(torch.isfinite(difference).sum()),
        "difference_semantics": "left_logprob_minus_right_logprob",
        "mean_signed_difference": float(difference.mean()),
        "mae": float(absolute.mean()),
        "abs_p50": float(torch.quantile(absolute, 0.50)),
        "abs_p95": float(torch.quantile(absolute, 0.95)),
        "abs_p99": float(torch.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
        "pearson": _correlation(left_numeric, right_numeric),
        "spearman": _correlation(
            _average_ranks(left_numeric), _average_ranks(right_numeric)
        ),
        "log_ratio": {
            "mean": float(difference.mean()),
            "std": float(difference.std(unbiased=False)),
            "min": float(difference.min()),
            "max": float(difference.max()),
        },
        "tolerance": float(tolerance),
        "fraction_abs_gt_tolerance": float((absolute > tolerance).to(torch.float64).mean()),
    }


def summarize_three_policy_identity(
    *,
    pi_rollout: Tensor,
    pi_old_actor: Tensor,
    pi_current_pre: Tensor,
    response_mask: Tensor,
    prompt_ids: Sequence[str],
    source_roles: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    """Describe rollout, old-actor and current-policy identities without aliasing them."""

    shapes = {
        tuple(pi_rollout.shape),
        tuple(pi_old_actor.shape),
        tuple(pi_current_pre.shape),
        tuple(response_mask.shape),
    }
    if len(shapes) != 1 or pi_rollout.ndim != 2:
        raise PGUpdateValidationError("three-policy identity tensors must share [batch, response]")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise PGUpdateValidationError("three-policy identity tolerance is invalid")
    if len(prompt_ids) != pi_rollout.shape[0] or len(source_roles) != pi_rollout.shape[0]:
        raise PGUpdateValidationError("three-policy prompt/source identity length mismatch")
    if not bool(torch.all((response_mask == 0) | (response_mask == 1))):
        raise PGUpdateValidationError("three-policy response mask must be binary")
    valid = response_mask.to(torch.bool)
    if not bool(valid.any()):
        raise PGUpdateValidationError("three-policy identity has no valid tokens")

    policies = {
        "pi_rollout": pi_rollout,
        "pi_old_actor": pi_old_actor,
        "pi_current_pre": pi_current_pre,
    }
    policy_summaries = {
        name: _policy_values_summary(value[valid]) for name, value in policies.items()
    }

    def pair_with_breakdowns(left: Tensor, right: Tensor) -> dict[str, Any]:
        pooled = _policy_pair_summary(left[valid], right[valid], tolerance=tolerance)
        prompt_groups: dict[str, list[int]] = {}
        source_groups: dict[str, list[int]] = {}
        for row_index, (prompt_id, source_role) in enumerate(
            zip(prompt_ids, source_roles, strict=True)
        ):
            prompt_groups.setdefault(str(prompt_id), []).append(row_index)
            source_groups.setdefault(str(source_role), []).append(row_index)

        def breakdown(groups: Mapping[str, Sequence[int]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for identity, row_indices in groups.items():
                selected_left = torch.cat([left[index][valid[index]] for index in row_indices])
                selected_right = torch.cat([right[index][valid[index]] for index in row_indices])
                result[identity] = _policy_pair_summary(
                    selected_left, selected_right, tolerance=tolerance
                )
            return result

        per_prompt = breakdown(prompt_groups)
        per_source = breakdown(source_groups)
        prompt_values = list(per_prompt.values())
        pooled.update(
            {
                "aggregation": {
                    "token_pooled": {
                        "mean_signed_difference": pooled["mean_signed_difference"],
                        "mae": pooled["mae"],
                        "max_abs": pooled["max_abs"],
                        "fraction_abs_gt_tolerance": pooled[
                            "fraction_abs_gt_tolerance"
                        ],
                    },
                    "prompt_equal": {
                        "prompt_count": len(prompt_values),
                        "mean_signed_difference": statistics.fmean(
                            value["mean_signed_difference"] for value in prompt_values
                        ),
                        "mae": statistics.fmean(value["mae"] for value in prompt_values),
                        "max_abs": max(value["max_abs"] for value in prompt_values),
                        "fraction_abs_gt_tolerance": statistics.fmean(
                            value["fraction_abs_gt_tolerance"] for value in prompt_values
                        ),
                    },
                },
                "per_prompt": per_prompt,
                "per_source": per_source,
            }
        )
        return pooled

    pairwise = {
        "pi_rollout_vs_pi_old_actor": pair_with_breakdowns(
            pi_rollout, pi_old_actor
        ),
        "pi_current_pre_vs_pi_rollout": pair_with_breakdowns(
            pi_current_pre, pi_rollout
        ),
        "pi_current_pre_vs_pi_old_actor": pair_with_breakdowns(
            pi_current_pre, pi_old_actor
        ),
    }
    formal_pair = pairwise["pi_current_pre_vs_pi_rollout"]
    actor_pair = pairwise["pi_current_pre_vs_pi_old_actor"]
    backend_pair = pairwise["pi_rollout_vs_pi_old_actor"]
    return {
        "schema_version": 2,
        "policy_semantics": {
            "pi_rollout": "sampling_time_behavior_policy_logprob",
            "pi_old_actor": "transformers_student_actor_pre_optimizer_detached_logprob",
            "pi_current_pre": "formal_loss_path_student_pre_optimizer_logprob",
        },
        "formal_denominator": "pi_rollout_sampling_time",
        "formal_identity_pair": "pi_current_pre_vs_pi_rollout",
        "tolerance": float(tolerance),
        "policies": policy_summaries,
        "pairwise": pairwise,
        "current_actor_identity_passed": actor_pair["max_abs"] <= tolerance,
        "rollout_training_backend_gap_observed": backend_pair["max_abs"] > tolerance,
        "formal_identity_gate_passed": formal_pair["max_abs"] <= tolerance,
    }


def audit_optimizer_update(
    *,
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
    loss_gradients: Mapping[str, Tensor],
    declared_trainable_names: Sequence[str],
    actual_requires_grad_names: Sequence[str],
    fresh_optimizer: bool,
    weight_decay: float,
    require_nonzero: bool,
    descent_dot_max: float,
    null_gradient_norm_max: float = 0.0,
    null_parameter_delta_norm_max: float = 0.0,
) -> OptimizerUpdateAudit:
    """Audit parameter ownership and the actual optimizer delta.

    The loss gradient uses the minimization convention.  For the frozen P4.2
    fresh, zero-decay AdamW optimizer its dot product with the parameter delta
    must therefore be negative for a non-null update.
    """

    if not before or set(before) != set(after):
        raise PGUpdateValidationError("parameter snapshots have inconsistent keys")
    declared = tuple(str(name) for name in declared_trainable_names)
    actual = tuple(str(name) for name in actual_requires_grad_names)
    if len(set(declared)) != len(declared) or set(declared) != set(actual):
        raise PGUpdateValidationError("trainable parameter manifest does not match requires_grad")
    if set(loss_gradients) != set(declared):
        raise PGUpdateValidationError("gradient parameter set does not match trainable manifest")
    if not set(declared).issubset(before):
        raise PGUpdateValidationError("trainable parameter manifest references an unknown parameter")

    deltas: dict[str, Tensor] = {}
    for name in before:
        if tuple(before[name].shape) != tuple(after[name].shape):
            raise PGUpdateValidationError("parameter snapshot shape changed")
        delta = after[name].detach().to(dtype=torch.float64) - before[name].detach().to(
            dtype=torch.float64
        )
        if not bool(torch.isfinite(delta).all()):
            raise PGUpdateValidationError("parameter delta is non-finite")
        deltas[name] = delta
        if name not in declared and bool(torch.any(delta != 0)):
            raise PGUpdateValidationError(f"frozen parameter changed: {name}")

    gradients: dict[str, Tensor] = {}
    for name in declared:
        gradient = loss_gradients[name].detach().to(dtype=torch.float64)
        if tuple(gradient.shape) != tuple(before[name].shape):
            raise PGUpdateValidationError("gradient shape does not match parameter")
        if not bool(torch.isfinite(gradient).all()):
            raise PGUpdateValidationError("trainable gradient is non-finite")
        gradients[name] = gradient

    trainable_deltas = [deltas[name] for name in declared]
    gradient_norm = _tensor_l2(list(gradients.values()))
    delta_norm = _tensor_l2(trainable_deltas)
    parameter_norm = _tensor_l2([before[name] for name in declared])
    relative_delta = delta_norm / max(parameter_norm, 1e-12)
    gradient_dot_delta = sum(
        float(torch.sum(gradients[name] * deltas[name])) for name in declared
    )
    update_norms = {name: _tensor_l2([deltas[name]]) for name in declared}
    optimizer_direction_is_hard = bool(fresh_optimizer and weight_decay == 0.0)

    failures: list[str] = []
    if require_nonzero:
        if gradient_norm <= 0:
            failures.append("zero_reward_gradient")
        if delta_norm <= 0:
            failures.append("zero_parameter_update")
        if optimizer_direction_is_hard and not gradient_dot_delta < descent_dot_max:
            failures.append("optimizer_not_in_loss_descent_direction")
    else:
        if gradient_norm > null_gradient_norm_max:
            failures.append("null_gradient_exceeded_tolerance")
        if delta_norm > null_parameter_delta_norm_max:
            failures.append("null_parameter_update_exceeded_tolerance")

    return OptimizerUpdateAudit(
        gradient_norm=gradient_norm,
        parameter_delta_norm=delta_norm,
        relative_parameter_delta=relative_delta,
        gradient_dot_parameter_delta=gradient_dot_delta,
        trainable_parameter_count=sum(int(before[name].numel()) for name in declared),
        module_update_norms=update_norms,
        optimizer_direction_is_hard_gate=optimizer_direction_is_hard,
        hard_gate_passed=not failures,
        failure_reasons=tuple(failures),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write canonical finite JSON via fsync + atomic rename and return its SHA."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PGUpdateValidationError("artifact is not finite canonical JSON") from error

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(encoded).hexdigest()


def persist_update_outcome(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    hard_gate_passed: bool,
    failure_status: str,
    failure_reason: str,
) -> dict[str, Any]:
    """Persist complete evidence before returning or raising on the hard gate."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "one_step_metrics.json"
    metrics_sha = atomic_write_json(metrics_path, metrics)
    result = {
        "schema_version": 2,
        "protocol_id": metrics.get("protocol_id"),
        "protocol_config_sha256": metrics.get("protocol_config_sha256"),
        "status": "pass" if hard_gate_passed else "fail",
        "hard_gate_passed": bool(hard_gate_passed),
        "metrics_path": metrics_path.name,
        "metrics_sha256": metrics_sha,
    }
    atomic_write_json(output / "one_step_result.json", result)
    if not hard_gate_passed:
        atomic_write_json(
            output / "failure.json",
            {
                "schema_version": 2,
                "status": str(failure_status),
                "reason": str(failure_reason),
                "metrics_path": metrics_path.name,
                "metrics_sha256": metrics_sha,
            },
        )
        raise PGUpdateValidationError(str(failure_reason))
    return result


def build_artifact_index(
    output_dir: str | Path, filenames: Sequence[str]
) -> dict[str, Any]:
    output = Path(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    for name in filenames:
        path = output / str(name)
        if not path.is_file():
            raise PGUpdateValidationError(f"artifact index input is missing: {name}")
        artifacts[str(name)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"schema_version": 2, "artifacts": artifacts}
    atomic_write_json(output / "artifact_index.json", payload)
    return payload


_REQUIRED_UPDATE_METRICS = frozenset(
    {
        "protocol_id",
        "protocol_config_sha256",
        "status",
        "hard_gate_passed",
        "objective_before",
        "objective_after",
        "loss_before",
        "loss_after",
        "alignment",
        "pre_ratio_summary",
        "post_ratio_summary",
        "pre_log_ratio_summary",
        "post_log_ratio_summary",
        "pre_active_clip_fraction",
        "post_active_clip_fraction",
        "gradient_norm",
        "parameter_delta_norm",
        "relative_parameter_delta",
        "gradient_dot_parameter_delta",
        "trainable_parameter_count",
        "module_update_norms",
        "nonfinite_counts",
        "per_prompt_objective_before_after",
        "per_source_diagnostics",
        "per_domain_diagnostics",
        "advantage_sign_fractions",
        "subgroup_logprob_change_diagnostics",
        "pre_update_audit",
        "objective_audit",
        "optimizer_audit",
        "teacher_gradient_parameters",
        "gradient_parameter_names",
        "trainable_parameter_names",
        "base_parameter_versions_unchanged",
        "frozen_trajectory_sha256",
        "student_initial_adapter_sha256",
        "student_updated_adapter_sha256",
    }
)


def _read_indexed_json(
    output: Path, index: Mapping[str, Any], name: str
) -> dict[str, Any] | None:
    record = index.get("artifacts", {}).get(name)
    path = output / name
    if not isinstance(record, Mapping) or not path.is_file():
        return None
    if record.get("sha256") != sha256_file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_summary(value: Any, *, positive: bool = False) -> bool:
    required = ("mean", "std", "min", "max", "p50", "p95", "p99")
    return bool(
        isinstance(value, Mapping)
        and all(_finite_number(value.get(name)) for name in required)
        and value["std"] >= 0
        and value["min"] <= value["p50"] <= value["p95"] <= value["p99"] <= value["max"]
        and (not positive or value["min"] > 0)
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_mapping_values(value: Any, *, nonnegative: bool = False) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value
        and all(
            _finite_number(item) and (not nonnegative or item >= 0)
            for item in value.values()
        )
    )


def _per_prompt_metrics_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value
        and all(
            isinstance(item, Mapping)
            and _finite_number(item.get("before"))
            and _finite_number(item.get("after"))
            for item in value.values()
        )
    )


def _diagnostic_breakdown_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value
        and all(
            isinstance(item, Mapping)
            and _nonnegative_int(item.get("count"))
            and item["count"] > 0
            and any(name != "count" for name in item)
            and all(
                _finite_number(metric)
                for name, metric in item.items()
                if name != "count"
            )
            for item in value.values()
        )
    )


def _advantage_fractions_valid(value: Any) -> bool:
    names = ("positive", "negative", "near_zero")
    return bool(
        isinstance(value, Mapping)
        and set(value) == set(names)
        and all(_finite_number(value[name]) and 0 <= value[name] <= 1 for name in names)
        and abs(sum(value[name] for name in names) - 1.0) <= 1e-6
    )


def _subgroup_diagnostic_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    counts = tuple(value.get(name) for name in ("positive_tokens", "negative_tokens", "near_zero_tokens"))
    return bool(
        value.get("protocol_version") == 2
        and value.get("gate_role") == "diagnostic_only"
        and isinstance(value.get("passed"), bool)
        and _finite_number(value.get("near_zero_tolerance"))
        and value["near_zero_tolerance"] >= 0
        and all(_nonnegative_int(count) for count in counts)
        and sum(counts) > 0
        and _finite_number(value.get("advantage_weighted_direction_mean"))
        and (
            (counts[0] == 0 and value.get("positive_advantage_logprob_change_mean") is None)
            or (counts[0] > 0 and _finite_number(value.get("positive_advantage_logprob_change_mean")))
        )
        and (
            (counts[1] == 0 and value.get("negative_advantage_logprob_change_mean") is None)
            or (counts[1] > 0 and _finite_number(value.get("negative_advantage_logprob_change_mean")))
        )
    )


def recompute_readiness(
    output_dir: str | Path,
    *,
    expected_protocol_id: str,
    expected_config_sha256: str,
    expected_trajectory_sha256: str,
    expected_medical_adapter_sha256: str,
    expected_repeatability_sha256: str,
    expected_route_isolation_sha256: str,
    expected_same_model_null_sha256: str,
) -> dict[str, Any]:
    """Derive readiness from SHA-verified fields; missing evidence fails closed."""

    output = Path(output_dir)
    defaults: dict[str, Any] = {
        "scorer_ready": False,
        "pg_update_ready": False,
        "null_update_ready": False,
        "sampler_refresh_ready": False,
        "opd_training_ready": False,
        "status": "blocked_artifact_integrity",
    }
    try:
        index = json.loads((output / "artifact_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(index, dict) or index.get("schema_version") != 2:
        return defaults

    scorer = _read_indexed_json(output, index, "scorer_readiness.json")
    scorer_identity = _read_indexed_json(output, index, "scorer_identity.json")
    scorer_ready = bool(
        scorer
        and scorer_identity
        and scorer.get("status") == "pass"
        and scorer.get("formal_backend") == "Transformers"
        and scorer.get("repeatability_passed") is True
        and scorer.get("route_isolation_passed") is True
        and scorer.get("same_model_hard_null_passed") is True
        and scorer.get("minimal_medical_identity_passed") is True
        and scorer.get("p4_1_repeatability_sha256") == expected_repeatability_sha256
        and scorer.get("p4_1_route_isolation_sha256") == expected_route_isolation_sha256
        and scorer.get("p4_1_same_model_null_sha256") == expected_same_model_null_sha256
        and scorer_identity.get("status") == "pass"
        and scorer_identity.get("formal_backend") == "Transformers"
        and scorer_identity.get("route") == "medical"
        and scorer_identity.get("adapter_sha256") == expected_medical_adapter_sha256
        and scorer_identity.get("trajectory_sha256") == expected_trajectory_sha256
        and _finite_number(
            scorer_identity.get("max_abs_delta_from_frozen_p4_1_scores")
        )
        and _finite_number(scorer_identity.get("tolerance"))
        and scorer_identity["max_abs_delta_from_frozen_p4_1_scores"] >= 0
        and scorer_identity["tolerance"] >= 0
        and scorer_identity["max_abs_delta_from_frozen_p4_1_scores"]
        <= scorer_identity["tolerance"]
    )

    result = _read_indexed_json(output, index, "one_step_result.json")
    metrics = _read_indexed_json(output, index, "one_step_metrics.json")
    metrics_sha_bound = bool(
        result
        and metrics
        and result.get("metrics_path") == "one_step_metrics.json"
        and result.get("metrics_sha256") == sha256_file(output / "one_step_metrics.json")
    )
    objective_ok = bool(
        metrics
        and _finite_number(metrics.get("objective_before"))
        and _finite_number(metrics.get("objective_after"))
        and metrics["objective_after"] > metrics["objective_before"] + OBJECTIVE_TOLERANCE
        and _finite_number(metrics.get("loss_before"))
        and _finite_number(metrics.get("loss_after"))
        and metrics["loss_after"] < metrics["loss_before"] - OBJECTIVE_TOLERANCE
        and _finite_number(metrics.get("alignment"))
        and metrics["alignment"] > 0
        and _finite_number(metrics.get("post_active_clip_fraction"))
        and metrics["post_active_clip_fraction"] <= SIGNIFICANT_ACTIVE_CLIP_FRACTION
        and _finite_number(metrics.get("gradient_norm"))
        and metrics["gradient_norm"] > 0
        and _finite_number(metrics.get("parameter_delta_norm"))
        and metrics["parameter_delta_norm"] > 0
        and _finite_number(metrics.get("gradient_dot_parameter_delta"))
        and metrics["gradient_dot_parameter_delta"] < 0
    )
    pre_log_ratio = metrics.get("pre_log_ratio_summary") if metrics else None
    pre_ratio = metrics.get("pre_ratio_summary") if metrics else None
    post_log_ratio = metrics.get("post_log_ratio_summary") if metrics else None
    post_ratio = metrics.get("post_ratio_summary") if metrics else None
    nonfinite = metrics.get("nonfinite_counts") if metrics else None
    pre_update_audit = metrics.get("pre_update_audit") if metrics else None
    objective_audit = metrics.get("objective_audit") if metrics else None
    optimizer_audit = metrics.get("optimizer_audit") if metrics else None
    evidence_contract_ok = bool(
        _finite_summary(pre_log_ratio)
        and max(abs(pre_log_ratio["min"]), abs(pre_log_ratio["max"]))
        <= PRE_LOG_RATIO_TOLERANCE
        and _finite_summary(pre_ratio, positive=True)
        and pre_ratio["min"] >= math.exp(-PRE_LOG_RATIO_TOLERANCE)
        and pre_ratio["max"] <= math.exp(PRE_LOG_RATIO_TOLERANCE)
        and _finite_summary(post_log_ratio)
        and _finite_summary(post_ratio, positive=True)
        and _finite_number(metrics.get("pre_active_clip_fraction"))
        and 0 <= metrics["pre_active_clip_fraction"] <= 1
        and 0 <= metrics["post_active_clip_fraction"] <= 1
        and _finite_number(metrics.get("relative_parameter_delta"))
        and metrics["relative_parameter_delta"] >= 0
        and _nonnegative_int(metrics.get("trainable_parameter_count"))
        and metrics["trainable_parameter_count"] > 0
        and isinstance(nonfinite, Mapping)
        and set(nonfinite) == {"advantage", "logprob", "ratio", "loss"}
        and all(_nonnegative_int(value) and value == 0 for value in nonfinite.values())
        and isinstance(pre_update_audit, Mapping)
        and pre_update_audit.get("passed") is True
        and _nonnegative_int(pre_update_audit.get("valid_tokens"))
        and pre_update_audit["valid_tokens"] > 0
        and all(
            _finite_number(pre_update_audit.get(name))
            for name in (
                "ratio_mean", "ratio_std", "ratio_min", "ratio_max",
                "ratio_max_abs_error", "log_ratio_max_abs",
            )
        )
        and _finite_number(pre_update_audit.get("log_ratio_max_abs"))
        and pre_update_audit["log_ratio_max_abs"] <= PRE_LOG_RATIO_TOLERANCE
        and isinstance(objective_audit, Mapping)
        and objective_audit.get("hard_gate_passed") is True
        and objective_audit.get("objective_improved") is True
        and objective_audit.get("alignment_required") is True
        and objective_audit.get("alignment_passed") is True
        and objective_audit.get("objective_before") == metrics.get("objective_before")
        and objective_audit.get("objective_after") == metrics.get("objective_after")
        and objective_audit.get("loss_before") == metrics.get("loss_before")
        and objective_audit.get("loss_after") == metrics.get("loss_after")
        and objective_audit.get("alignment") == metrics.get("alignment")
        and objective_audit.get("active_clip_fraction_after")
        == metrics.get("post_active_clip_fraction")
        and isinstance(optimizer_audit, Mapping)
        and optimizer_audit.get("hard_gate_passed") is True
        and optimizer_audit.get("optimizer_direction_is_hard_gate") is True
        and optimizer_audit.get("gradient_norm") == metrics.get("gradient_norm")
        and optimizer_audit.get("parameter_delta_norm") == metrics.get("parameter_delta_norm")
        and optimizer_audit.get("relative_parameter_delta")
        == metrics.get("relative_parameter_delta")
        and optimizer_audit.get("gradient_dot_parameter_delta")
        == metrics.get("gradient_dot_parameter_delta")
        and optimizer_audit.get("trainable_parameter_count")
        == metrics.get("trainable_parameter_count")
        and metrics.get("teacher_gradient_parameters") == []
        and isinstance(metrics.get("gradient_parameter_names"), list)
        and isinstance(metrics.get("trainable_parameter_names"), list)
        and bool(metrics["trainable_parameter_names"])
        and all("lora" in name.lower() for name in metrics["trainable_parameter_names"])
        and len(set(metrics["trainable_parameter_names"]))
        == len(metrics["trainable_parameter_names"])
        and set(metrics["gradient_parameter_names"])
        == set(metrics["trainable_parameter_names"])
        and len(metrics["gradient_parameter_names"])
        == len(metrics["trainable_parameter_names"])
        and metrics.get("base_parameter_versions_unchanged") is True
        and _finite_mapping_values(metrics.get("module_update_norms"), nonnegative=True)
        and set(metrics["module_update_norms"]) == set(metrics["trainable_parameter_names"])
        and metrics["module_update_norms"] == optimizer_audit.get("module_update_norms")
        and _per_prompt_metrics_valid(metrics.get("per_prompt_objective_before_after"))
        and _diagnostic_breakdown_valid(metrics.get("per_source_diagnostics"))
        and _diagnostic_breakdown_valid(metrics.get("per_domain_diagnostics"))
        and _advantage_fractions_valid(metrics.get("advantage_sign_fractions"))
        and _subgroup_diagnostic_valid(metrics.get("subgroup_logprob_change_diagnostics"))
    )
    pg_ready = bool(
        scorer_ready
        and result
        and metrics
        and metrics_sha_bound
        and _REQUIRED_UPDATE_METRICS.issubset(metrics)
        and result.get("protocol_id") == expected_protocol_id
        and metrics.get("protocol_id") == expected_protocol_id
        and result.get("protocol_config_sha256") == expected_config_sha256
        and metrics.get("protocol_config_sha256") == expected_config_sha256
        and metrics.get("frozen_trajectory_sha256") == expected_trajectory_sha256
        and _valid_sha256(metrics.get("student_initial_adapter_sha256"))
        and _valid_sha256(metrics.get("student_updated_adapter_sha256"))
        and metrics["student_updated_adapter_sha256"]
        != metrics["student_initial_adapter_sha256"]
        and result.get("status") == "pass"
        and result.get("hard_gate_passed") is True
        and metrics.get("status") == "pass"
        and metrics.get("hard_gate_passed") is True
        and objective_ok
        and evidence_contract_ok
    )

    null = _read_indexed_json(output, index, "null_update.json")
    null_optimizer = null.get("optimizer_audit") if null else None
    null_nonfinite = null.get("nonfinite_counts") if null else None
    null_ready = bool(
        pg_ready
        and null
        and null.get("protocol_id") == expected_protocol_id
        and null.get("protocol_config_sha256") == expected_config_sha256
        and null.get("frozen_trajectory_sha256") == expected_trajectory_sha256
        and null.get("status") == "pass"
        and null.get("hard_gate_passed") is True
        and _finite_number(null.get("advantage_max_abs"))
        and null["advantage_max_abs"] >= 0
        and null["advantage_max_abs"] <= 1e-6
        and _finite_number(null.get("gradient_norm"))
        and null["gradient_norm"] >= 0
        and null["gradient_norm"] <= 1e-10
        and _finite_number(null.get("parameter_delta_norm"))
        and null["parameter_delta_norm"] >= 0
        and null["parameter_delta_norm"] <= 1e-12
        and all(
            _finite_number(null.get(name)) and abs(null[name]) <= 1e-10
            for name in ("objective_before", "objective_after", "loss_before", "loss_after")
        )
        and null.get("teacher_logprob_source") == "same_real_base_forward_detached"
        and null.get("same_objective_mask_reduction_writer") is True
        and isinstance(null_optimizer, Mapping)
        and null_optimizer.get("hard_gate_passed") is True
        and null_optimizer.get("optimizer_direction_is_hard_gate") is True
        and _finite_number(null_optimizer.get("gradient_norm"))
        and null_optimizer["gradient_norm"] >= 0
        and null_optimizer["gradient_norm"] == null["gradient_norm"]
        and _finite_number(null_optimizer.get("parameter_delta_norm"))
        and null_optimizer["parameter_delta_norm"] >= 0
        and null_optimizer["parameter_delta_norm"] == null["parameter_delta_norm"]
        and isinstance(null_nonfinite, Mapping)
        and set(null_nonfinite) >= {"advantage", "logprob", "ratio", "loss"}
        and all(value == 0 for value in null_nonfinite.values())
        and null.get("formal_checkpoint_saved") is False
    )

    sampler = _read_indexed_json(output, index, "sampler_refresh.json")
    sampler_ready = bool(
        null_ready
        and sampler
        and sampler.get("protocol_id") == expected_protocol_id
        and sampler.get("protocol_config_sha256") == expected_config_sha256
        and sampler.get("status") == "pass"
        and sampler.get("hard_gate_passed") is True
        and sampler.get("trainer_sampler_identity_match") is True
        and sampler.get("stale_adapter_rejected") is True
        and sampler.get("probe_logprob_match") is True
        and sampler.get("old_version") == 0
        and sampler.get("trainer_version") == sampler["old_version"] + 1
        and sampler.get("sampler_version") == sampler["trainer_version"]
        and _valid_sha256(sampler.get("old_sha256"))
        and sampler.get("old_sha256") == metrics.get("student_initial_adapter_sha256")
        and _valid_sha256(sampler.get("trainer_sha256"))
        and sampler.get("trainer_sha256") == metrics.get("student_updated_adapter_sha256")
        and sampler["trainer_sha256"] != sampler["old_sha256"]
        and sampler.get("sampler_sha256") == sampler["trainer_sha256"]
        and _finite_number(sampler.get("probe_max_abs_delta"))
        and _finite_number(sampler.get("probe_tolerance"))
        and 0 <= sampler["probe_max_abs_delta"] <= sampler["probe_tolerance"]
        and _finite_number(sampler.get("pre_refresh_probe_max_abs_delta"))
        and 0
        <= sampler["pre_refresh_probe_max_abs_delta"]
        <= sampler["probe_tolerance"]
        and sampler.get("pre_refresh_probe_match") is True
        and sampler.get("actual_adapter_refresh_verified") is True
        and sampler.get("refresh_mode") == "in_place_peft_load_and_set_adapter"
        and sampler.get("load_adapter_called") is True
        and sampler.get("set_adapter_called") is True
        and sampler.get("delete_old_adapter_called") is True
        and sampler.get("old_adapter_removed") is True
        and sampler.get("guarded_probe_identity_verified") is True
        and sampler.get("active_adapter_name") == "version1"
        and sampler.get("cache_disabled") is True
        and sampler.get("cache_identity_reused") is False
        and sampler.get("separate_sampler_model_loaded") is True
        and sampler.get("formal_checkpoint_saved") is False
    )
    runtime_release = _read_indexed_json(output, index, "runtime_release.json")
    runtime_release_ready = bool(
        runtime_release
        and runtime_release.get("status") == "pass"
        and runtime_release.get("models_released") is True
        and runtime_release.get("post_process_exit_verification_required") is True
        and isinstance(runtime_release.get("cuda_allocated_bytes_diagnostic"), list)
        and isinstance(runtime_release.get("cuda_reserved_bytes_diagnostic"), list)
    )
    cleanup = _read_indexed_json(output, index, "resource_cleanup.json")
    cleanup_ready = bool(
        runtime_release_ready
        and cleanup
        and cleanup.get("status") == "pass"
        and cleanup.get("runtime_exit_code") == 0
        and cleanup.get("gpu_memory_used_mib") == [0, 0]
        and cleanup.get("gpu_compute_processes") == []
        and cleanup.get("vllm_ray_workers_found") is False
    )
    status = (
        "opd_backend_ready_waiting_for_b2_authorization"
        if sampler_ready and cleanup_ready
        else "blocked_pg_opd_validation"
    )
    return {
        "scorer_ready": scorer_ready,
        "pg_update_ready": pg_ready,
        "null_update_ready": null_ready,
        "sampler_refresh_ready": sampler_ready,
        "opd_training_ready": False,
        "status": status,
    }


def audit_sampler_refresh(
    *,
    old_version: int,
    old_sha256: str,
    trainer_version: int,
    trainer_sha256: str,
    sampler_version: int,
    sampler_sha256: str,
    probe_max_abs_delta: float,
    probe_tolerance: float,
    stale_adapter_rejected: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if trainer_version != old_version + 1 or trainer_sha256 == old_sha256:
        failures.append("trainer adapter identity did not advance")
    if sampler_version != trainer_version or sampler_sha256 != trainer_sha256:
        failures.append("sampler retained a stale adapter identity")
    if stale_adapter_rejected is not True:
        failures.append("stale adapter was not rejected")
    if (
        not math.isfinite(probe_max_abs_delta)
        or probe_max_abs_delta < 0
        or probe_tolerance < 0
        or probe_max_abs_delta > probe_tolerance
    ):
        failures.append("sampler probe logprob does not match the updated Student")
    if failures:
        raise PGUpdateValidationError("; ".join(failures))
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "hard_gate_passed": True,
        "old_version": old_version,
        "old_sha256": old_sha256,
        "trainer_version": trainer_version,
        "trainer_sha256": trainer_sha256,
        "sampler_version": sampler_version,
        "sampler_sha256": sampler_sha256,
        "trainer_sampler_identity_match": True,
        "stale_adapter_rejected": True,
        "probe_logprob_match": True,
        "probe_max_abs_delta": probe_max_abs_delta,
        "probe_tolerance": probe_tolerance,
    }


def require_sampler_identity(
    state: Mapping[str, Any], *, expected_version: int, expected_sha256: str
) -> None:
    """Reject a rollout request whose adapter identity is stale or ambiguous."""

    if (
        state.get("version") != expected_version
        or state.get("adapter_sha256") != expected_sha256
        or not _valid_sha256(expected_sha256)
    ):
        raise PGUpdateValidationError("stale sampler adapter identity rejected")


def refresh_sampler_adapter(
    sampler: Any,
    *,
    adapter_path: str | Path,
    old_version: int,
    old_sha256: str,
    old_adapter_name: str,
    new_version: int,
    new_sha256: str,
    new_adapter_name: str,
) -> dict[str, Any]:
    """Perform and verify an in-place PEFT adapter refresh on a rollout sampler."""

    if (
        new_version != old_version + 1
        or new_sha256 == old_sha256
        or not _valid_sha256(old_sha256)
        or not _valid_sha256(new_sha256)
        or new_adapter_name == old_adapter_name
    ):
        raise PGUpdateValidationError("invalid sampler adapter refresh identity")
    active_before = getattr(sampler, "active_adapter", None)
    if isinstance(active_before, (list, tuple)):
        active_before = active_before[0] if len(active_before) == 1 else None
    if active_before != old_adapter_name:
        raise PGUpdateValidationError("sampler old adapter identity mismatch")

    sampler.load_adapter(
        str(adapter_path), adapter_name=new_adapter_name, is_trainable=False
    )
    sampler.set_adapter(new_adapter_name)
    active_after = getattr(sampler, "active_adapter", None)
    if isinstance(active_after, (list, tuple)):
        active_after = active_after[0] if len(active_after) == 1 else None
    if active_after != new_adapter_name:
        raise PGUpdateValidationError("sampler did not activate refreshed adapter")
    delete_adapter = getattr(sampler, "delete_adapter", None)
    if not callable(delete_adapter):
        raise PGUpdateValidationError("sampler cannot remove the stale adapter")
    delete_adapter(old_adapter_name)
    peft_config = getattr(sampler, "peft_config", None)
    old_adapter_removed = bool(
        isinstance(peft_config, Mapping) and old_adapter_name not in peft_config
    )
    if not old_adapter_removed:
        raise PGUpdateValidationError("sampler stale adapter remained loadable")
    active_after_delete = getattr(sampler, "active_adapter", None)
    if isinstance(active_after_delete, (list, tuple)):
        active_after_delete = (
            active_after_delete[0] if len(active_after_delete) == 1 else None
        )
    if active_after_delete != new_adapter_name:
        raise PGUpdateValidationError("sampler refresh lost the active new adapter")
    return {
        "version": new_version,
        "adapter_sha256": new_sha256,
        "active_adapter_name": active_after_delete,
        "previous_adapter_name": old_adapter_name,
        "actual_adapter_refresh_verified": True,
        "refresh_mode": "in_place_peft_load_and_set_adapter",
        "load_adapter_called": True,
        "set_adapter_called": True,
        "delete_old_adapter_called": True,
        "old_adapter_removed": True,
    }
