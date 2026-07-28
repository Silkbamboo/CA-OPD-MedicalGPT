"""Centralised randomness control.

PROJECT_PLAN.md §15 requires every run to record its seed, and CLAUDE.md §5
requires a single place where randomness is configured. Callers must never
touch ``random.seed`` / ``torch.manual_seed`` directly.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SeedState:
    """Snapshot of what was seeded, so it can be written into run metadata."""

    seed: int
    seeded_python: bool = True
    seeded_numpy: bool = False
    seeded_torch: bool = False
    deterministic_torch: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "seeded_python": self.seeded_python,
            "seeded_numpy": self.seeded_numpy,
            "seeded_torch": self.seeded_torch,
            "deterministic_torch": self.deterministic_torch,
            **self.extra,
        }


def seed_everything(seed: int, deterministic_torch: bool = True) -> SeedState:
    """Seed python/numpy/torch if available and return what was actually seeded.

    ``deterministic_torch`` also disables cuDNN autotuning; on CPU it is a
    no-op but is still recorded so that CPU dry-run metadata matches GPU runs.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)!r}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    seeded_numpy = False
    try:  # pragma: no cover - import guard
        import numpy as np

        np.random.seed(seed)
        seeded_numpy = True
    except ImportError:
        pass

    seeded_torch = False
    applied_deterministic = False
    try:  # pragma: no cover - import guard
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded_torch = True
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied_deterministic = True
    except ImportError:
        pass

    return SeedState(
        seed=seed,
        seeded_numpy=seeded_numpy,
        seeded_torch=seeded_torch,
        deterministic_torch=applied_deterministic,
    )


def derive_seed(base_seed: int, *parts: str, modulus: int = 2**31 - 1) -> int:
    """Derive a stable child seed from a base seed and string parts.

    Used so that e.g. split construction, rollout sampling and router sampling
    get independent but reproducible streams from one run-level seed.
    """
    import hashlib

    key = "|".join([str(base_seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def new_rng(base_seed: int, *parts: str) -> random.Random:
    """Return an isolated ``random.Random`` for a named stream."""
    return random.Random(derive_seed(base_seed, *parts))


def optional_int(value: Optional[int], default: int) -> int:
    return default if value is None else int(value)
