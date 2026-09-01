"""CPU-safe, physically separated probability evidence for P5.1.

The module names the three probability comparisons by stage and refuses to
validate evidence when token-pool, policy, adapter, sampler, or refresh identity
drifts.  It never reads prompts or evaluation labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


class RatioContractV2Error(RuntimeError):
    """A ratio identity or schema invariant failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


@dataclass(frozen=True)
class RatioPoolBindingV2:
    input_token_sha256: str
    response_token_sha256: str
    attention_mask_sha256: str
    response_mask_sha256: str
    valid_mask_sha256: str
    valid_token_count: int
    batch_size: int
    pool_binding_sha256: str

    @classmethod
    def from_tensors(
        cls,
        *,
        input_ids: Tensor,
        response_ids: Tensor,
        attention_mask: Tensor,
        response_mask: Tensor,
        valid_mask: Tensor,
    ) -> "RatioPoolBindingV2":
        if not (
            input_ids.ndim == response_ids.ndim == attention_mask.ndim == response_mask.ndim == valid_mask.ndim == 2
            and input_ids.shape == attention_mask.shape == response_mask.shape
            and response_ids.shape == valid_mask.shape
            and input_ids.shape[0] == response_ids.shape[0]
        ):
            raise RatioContractV2Error("ratio pool tensor shapes differ")
        if valid_mask.dtype is not torch.bool:
            valid_mask = valid_mask.bool()
        fields = {
            "input_token_sha256": _tensor_sha256(input_ids),
            "response_token_sha256": _tensor_sha256(response_ids),
            "attention_mask_sha256": _tensor_sha256(attention_mask.bool()),
            "response_mask_sha256": _tensor_sha256(response_mask.bool()),
            "valid_mask_sha256": _tensor_sha256(valid_mask),
            "valid_token_count": int(valid_mask.sum().item()),
            "batch_size": int(valid_mask.shape[0]),
        }
        if fields["valid_token_count"] <= 0:
            raise RatioContractV2Error("ratio pool is empty")
        return cls(**fields, pool_binding_sha256=_canonical_sha256(fields))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked(value: Tensor, mask: Tensor, label: str) -> Tensor:
    if value.shape != mask.shape:
        raise RatioContractV2Error(f"{label} shape differs from valid mask")
    result = value.detach().float().cpu()[mask.detach().cpu().bool()]
    if result.numel() == 0 or not torch.isfinite(result).all():
        raise RatioContractV2Error(f"{label} is empty or non-finite")
    return result


def _quantile(values: Tensor, probability: float) -> float:
    return float(torch.quantile(values, probability).item())


def _summary(values: Tensor, *, include_abs: bool = True) -> dict[str, float | int]:
    values = values.detach().float().flatten().cpu()
    result: dict[str, float | int] = {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "p999": _quantile(values, 0.999),
    }
    if include_abs:
        absolute = values.abs()
        result.update(
            {
                "abs_max": float(absolute.max().item()),
                "abs_p99": _quantile(absolute, 0.99),
                "abs_p999": _quantile(absolute, 0.999),
            }
        )
    return result


def _ratio_summary(log_values: Tensor) -> dict[str, float | int]:
    return _summary(torch.exp(log_values), include_abs=False)


def _ess(weights: Tensor) -> float:
    weights = weights.detach().double().flatten().cpu()
    denominator = float(weights.square().sum().item()) * int(weights.numel())
    if denominator <= 0.0:
        raise RatioContractV2Error("backend correction ESS denominator is zero")
    return float(weights.sum().item() ** 2 / denominator)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def compute_ratio_evidence_v2(
    *,
    log_q_pre: Tensor,
    log_p_old_canonical: Tensor,
    log_mu_sampler: Tensor,
    log_q_post: Tensor,
    valid_mask: Tensor,
    prompt_ids: Sequence[str],
    source_roles: Sequence[str],
    token_ids: Tensor,
    advantage: Tensor,
    loss_contribution: Tensor,
    gradient_proxy: Tensor,
    pool_binding: RatioPoolBindingV2,
    policy_version: int,
    q_pre_adapter_sha256: str,
    p_old_adapter_sha256: str,
    sampler_version: int,
    refresh_version: int,
    backend_log_clip: float,
    post_shift_tail_abs_log_threshold: float,
) -> dict[str, Any]:
    """Build privacy-safe token-level evidence without persisting prompt text."""

    if not (
        len(prompt_ids) == len(source_roles) == valid_mask.shape[0]
        and token_ids.shape == valid_mask.shape
        and advantage.shape == loss_contribution.shape == gradient_proxy.shape == valid_mask.shape
        and isinstance(policy_version, int)
        and not isinstance(policy_version, bool)
        and backend_log_clip > 0.0
        and post_shift_tail_abs_log_threshold > 0.0
        and _is_sha256(q_pre_adapter_sha256)
        and _is_sha256(p_old_adapter_sha256)
    ):
        raise RatioContractV2Error("ratio evidence metadata or tensor shapes differ")
    mask = valid_mask.detach().cpu().bool()
    q_pre = _masked(log_q_pre, mask, "q_pre")
    old = _masked(log_p_old_canonical, mask, "p_old_canonical")
    mu = _masked(log_mu_sampler, mask, "mu_sampler")
    q_post = _masked(log_q_post, mask, "q_post")
    if not (q_pre.numel() == old.numel() == mu.numel() == q_post.numel() == pool_binding.valid_token_count):
        raise RatioContractV2Error("ratio values differ from bound token pool")

    log_r_ppo = q_pre - old
    log_rho_raw = old - mu
    # Preserve the v1 objective's bounded detached correction: a numerical
    # lower clamp and the registered upper cap.  Both raw and clipped values
    # remain visible in evidence.
    log_rho_clipped = torch.clamp(log_rho_raw, min=-20.0, max=float(backend_log_clip))
    backend_weights = torch.exp(log_rho_clipped)
    post_shift = q_post - old

    per_prompt: dict[str, dict[str, float | int]] = {}
    offset = 0
    for index, prompt_id in enumerate(prompt_ids):
        count = int(mask[index].sum().item())
        prompt_weights = backend_weights[offset : offset + count]
        per_prompt[str(prompt_id)] = {
            "token_count": count,
            "ess_fraction": _ess(prompt_weights),
        }
        offset += count

    flat_loss = _masked(loss_contribution, mask, "loss contribution").abs()
    flat_grad = _masked(gradient_proxy, mask, "gradient proxy").abs()
    flat_advantage = _masked(advantage, mask, "advantage")
    flat_tokens = token_ids.detach().cpu()[mask]
    tail_mask = post_shift.abs() > float(post_shift_tail_abs_log_threshold)
    tail_count = int(tail_mask.sum().item())
    loss_total = float(flat_loss.sum().item())
    grad_total = float(flat_grad.sum().item())
    tail_loss_share = 0.0 if loss_total == 0.0 else float(flat_loss[tail_mask].sum().item()) / loss_total
    tail_grad_share = 0.0 if grad_total == 0.0 else float(flat_grad[tail_mask].sum().item()) / grad_total

    row_indices, column_indices = torch.where(mask)
    ranked = torch.argsort(post_shift.abs(), descending=True)[:20]
    top_tokens: list[dict[str, Any]] = []
    for flat_index in ranked.tolist():
        row_index = int(row_indices[flat_index].item())
        top_tokens.append(
            {
                "rank": len(top_tokens) + 1,
                "prompt_id": str(prompt_ids[row_index]),
                "source": str(source_roles[row_index]),
                "token_id": int(flat_tokens[flat_index].item()),
                "token_position": int(column_indices[flat_index].item()),
                "advantage": float(flat_advantage[flat_index].item()),
                "log_q_pre": float(q_pre[flat_index].item()),
                "log_p_old_canonical": float(old[flat_index].item()),
                "log_mu_sampler": float(mu[flat_index].item()),
                "log_q_post": float(q_post[flat_index].item()),
                "log_post_update_shift": float(post_shift[flat_index].item()),
                "post_update_ratio": float(torch.exp(post_shift[flat_index]).item()),
                "absolute_loss_contribution": float(flat_loss[flat_index].item()),
                "absolute_gradient_proxy": float(flat_grad[flat_index].item()),
                "negative_advantage_branch": bool(flat_advantage[flat_index].item() < 0.0),
            }
        )

    binding = pool_binding.as_dict()
    binding_sha = pool_binding.pool_binding_sha256
    evidence = {
        "schema_id": "ca-opd/ratio-evidence/v2",
        "schema_version": 2,
        "pool_binding": binding,
        "identity": {
            "q_pre_policy_version": policy_version,
            "p_old_policy_version": policy_version,
            "sampler_version": sampler_version,
            "refresh_version": refresh_version,
            "q_pre_adapter_sha256": q_pre_adapter_sha256,
            "p_old_adapter_sha256": p_old_adapter_sha256,
        },
        "ppo_ratio": {
            "stage": "pre_update",
            "formula": "log_q_pre-log_p_old_canonical",
            "pool_binding_sha256": binding_sha,
            "log": _summary(log_r_ppo),
            "ratio": _ratio_summary(log_r_ppo),
            "approx_kl": float((-log_r_ppo).mean().item()),
            "q_pre_p_old_max_abs": float(log_r_ppo.abs().max().item()),
        },
        "backend_correction": {
            "stage": "sampler_to_canonical",
            "formula": "log_p_old_canonical-log_mu_sampler",
            "pool_binding_sha256": binding_sha,
            "raw_log": _summary(log_rho_raw),
            "clipped_log": _summary(log_rho_clipped),
            "raw_weight": _ratio_summary(torch.clamp(log_rho_raw, min=-20.0, max=20.0)),
            "clipped_weight": _summary(backend_weights, include_abs=False),
            "clip_bounds_log": [-20.0, float(backend_log_clip)],
            "clip_fraction": float((log_rho_raw != log_rho_clipped).float().mean().item()),
            "detached_from_gradient": True,
            "ess": {
                "formula": "(sum(w_clipped)^2)/(n*sum(w_clipped^2))",
                "aggregation": "token_pooled_and_per_prompt",
                "pool_binding_sha256": binding_sha,
                "pooled_fraction": _ess(backend_weights),
                "per_prompt": per_prompt,
            },
        },
        "post_update_policy_shift": {
            "stage": "post_update",
            "formula": "log_q_post-log_p_old_canonical",
            "pool_binding_sha256": binding_sha,
            "log": _summary(post_shift),
            "ratio": _ratio_summary(post_shift),
            "tail": {
                "abs_log_threshold": float(post_shift_tail_abs_log_threshold),
                "token_count": tail_count,
                "token_fraction": tail_count / int(post_shift.numel()),
                "absolute_loss_share": tail_loss_share,
                "gradient_proxy_share": tail_grad_share,
                "top_tokens": top_tokens,
            },
        },
    }
    validate_ratio_evidence_v2(evidence)
    return evidence


def _assert_finite_tree(value: Any, path: str = "ratio evidence") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RatioContractV2Error(f"{path} is non-finite")


def validate_ratio_evidence_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_id") != "ca-opd/ratio-evidence/v2" or value.get("schema_version") != 2:
        raise RatioContractV2Error("ratio evidence schema differs")
    _assert_finite_tree(value)
    pool = value.get("pool_binding")
    identity = value.get("identity")
    ppo = value.get("ppo_ratio")
    backend = value.get("backend_correction")
    post = value.get("post_update_policy_shift")
    if not all(isinstance(item, Mapping) for item in (pool, identity, ppo, backend, post)):
        raise RatioContractV2Error("ratio evidence sections are absent")
    expected_stages = ((ppo, "pre_update"), (backend, "sampler_to_canonical"), (post, "post_update"))
    if any(section.get("stage") != stage for section, stage in expected_stages):
        raise RatioContractV2Error("ratio stage is aliased or misplaced")
    binding_sha = pool.get("pool_binding_sha256")
    if not _is_sha256(binding_sha):
        raise RatioContractV2Error("ratio token pool binding SHA is absent")
    if any(section.get("pool_binding_sha256") != binding_sha for section, _ in expected_stages):
        raise RatioContractV2Error("ratio sections use a different token pool")
    ess = backend.get("ess")
    if not isinstance(ess, Mapping) or ess.get("pool_binding_sha256") != binding_sha:
        raise RatioContractV2Error("ratio and ESS use a different token pool")
    if identity.get("q_pre_policy_version") != identity.get("p_old_policy_version"):
        raise RatioContractV2Error("q_pre and p_old policy version differs")
    if identity.get("q_pre_adapter_sha256") != identity.get("p_old_adapter_sha256"):
        raise RatioContractV2Error("q_pre and p_old adapter differs")
    policy_version = identity.get("q_pre_policy_version")
    if identity.get("sampler_version") != policy_version or identity.get("refresh_version") != policy_version:
        raise RatioContractV2Error("sampler/refresh version is stale")
    if backend.get("detached_from_gradient") is not True:
        raise RatioContractV2Error("backend correction is not detached")
    return {"passed": True, "pool_binding_sha256": binding_sha, "policy_version": policy_version}
