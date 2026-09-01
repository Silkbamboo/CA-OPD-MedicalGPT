"""P4.8e scheduler/checkpoint primitives that preserve frozen optimizer math."""

from __future__ import annotations

from typing import Any


class B2MemoryCheckpointV2Error(RuntimeError):
    """The constant-LR or checkpoint lifecycle differs."""


def build_constant_lr_scheduler(torch_module: Any, optimizer: Any) -> Any:
    return torch_module.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _epoch: 1.0
    )


def step_constant_lr_scheduler(
    scheduler: Any,
    optimizer: Any,
    *,
    expected_learning_rate: float,
) -> dict[str, float | int]:
    before = [float(group["lr"]) for group in optimizer.param_groups]
    if not before or any(value != float(expected_learning_rate) for value in before):
        raise B2MemoryCheckpointV2Error(
            "optimizer learning rate differs before scheduler step"
        )
    scheduler.step()
    after = [float(group["lr"]) for group in optimizer.param_groups]
    if after != before:
        raise B2MemoryCheckpointV2Error(
            "constant scheduler changed the frozen learning rate"
        )
    return {
        "scheduler_steps": 1,
        "learning_rate_before": before[0],
        "learning_rate_after": after[0],
    }


__all__ = [
    "B2MemoryCheckpointV2Error",
    "build_constant_lr_scheduler",
    "step_constant_lr_scheduler",
]
