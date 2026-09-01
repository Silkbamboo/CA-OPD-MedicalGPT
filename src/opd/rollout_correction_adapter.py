"""Thin compatibility adapter for the pinned veRL 0.8.0 rollout correction."""

from __future__ import annotations

import torch
from torch import Tensor


class RolloutCorrectionAdapterError(RuntimeError):
    """Raised when the pinned native correction path cannot be used safely."""


NATIVE_VERL_VERSION = "0.8.0"
NATIVE_ROLLOUT_IS_MODE = "token"
NATIVE_BATCH_NORMALIZE = False
LOG_WEIGHT_SAFETY_BOUND = 20.0


def native_decoupled_token_is(
    log_ratio: Tensor,
    response_mask: Tensor,
    *,
    threshold: float,
) -> Tensor:
    """Return native veRL token-TIS weights for ``p_old / q``.

    This adapter deliberately exposes only the frozen P4.3 subset: token-level
    upper truncation, no lower cutoff, no batch normalization, no rejection
    sampling, and no bypass mode.
    """

    if not isinstance(threshold, (int, float)) or not torch.isfinite(
        torch.tensor(float(threshold))
    ):
        raise RolloutCorrectionAdapterError("rollout IS threshold must be finite")
    if float(threshold) <= 0:
        raise RolloutCorrectionAdapterError("rollout IS threshold must be positive")
    try:
        from verl.trainer.ppo.rollout_corr_helper import (
            compute_rollout_correction_weights,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RolloutCorrectionAdapterError(
            "pinned veRL 0.8.0 rollout correction is unavailable"
        ) from exc

    weights, _native_metrics = compute_rollout_correction_weights(
        log_ratio=log_ratio,
        response_mask=response_mask.to(dtype=log_ratio.dtype),
        rollout_is=NATIVE_ROLLOUT_IS_MODE,
        rollout_is_threshold=float(threshold),
        rollout_is_batch_normalize=NATIVE_BATCH_NORMALIZE,
    )
    return weights.detach()
