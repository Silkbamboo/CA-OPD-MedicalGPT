"""Disk admission and per-commit safety checks for Formal B2 v2."""

from __future__ import annotations

from typing import Any


class DiskSafetyV2Error(RuntimeError):
    """The persistent volume cannot safely complete the next atomic write."""


BASE_FREE_FLOOR_BYTES = 10_000_000_000


def disk_safety_requirement_v2(
    *,
    full_checkpoint_bytes: int,
    predicted_log_growth_bytes: int,
    base_free_floor_bytes: int = BASE_FREE_FLOOR_BYTES,
) -> int:
    """Return the registered free-space floor including atomic-write peak."""

    values = (full_checkpoint_bytes, predicted_log_growth_bytes, base_free_floor_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise DiskSafetyV2Error("disk safety inputs must be non-negative integers")
    if full_checkpoint_bytes == 0:
        raise DiskSafetyV2Error("disk safety requires a measured full checkpoint")
    return base_free_floor_bytes + (2 * full_checkpoint_bytes) + predicted_log_growth_bytes


def validate_disk_safety_v2(
    *,
    free_bytes: int,
    full_checkpoint_bytes: int,
    predicted_log_growth_bytes: int,
    base_free_floor_bytes: int = BASE_FREE_FLOOR_BYTES,
) -> dict[str, Any]:
    requirement = disk_safety_requirement_v2(
        full_checkpoint_bytes=full_checkpoint_bytes,
        predicted_log_growth_bytes=predicted_log_growth_bytes,
        base_free_floor_bytes=base_free_floor_bytes,
    )
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool) or free_bytes < requirement:
        raise DiskSafetyV2Error(
            f"disk safety requirement not met: free={free_bytes}, required={requirement}"
        )
    return {
        "schema_version": 2,
        "artifact_kind": "formal_b2_disk_safety_v2",
        "passed": True,
        "free_bytes": free_bytes,
        "required_free_bytes": requirement,
        "base_free_floor_bytes": base_free_floor_bytes,
        "full_checkpoint_bytes": full_checkpoint_bytes,
        "atomic_checkpoint_peak_bytes": 2 * full_checkpoint_bytes,
        "predicted_log_growth_bytes": predicted_log_growth_bytes,
        "margin_bytes": free_bytes - requirement,
    }


__all__ = [
    "BASE_FREE_FLOOR_BYTES",
    "DiskSafetyV2Error",
    "disk_safety_requirement_v2",
    "validate_disk_safety_v2",
]
