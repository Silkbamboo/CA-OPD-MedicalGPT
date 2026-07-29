"""Shared pytest configuration.

Resource note (found the hard way): this dev container has a 2 GiB cgroup memory
limit while the host exposes many cores, so torch's default intra-op thread pool
allocated enough per-thread arena memory to get the whole pytest process
OOM-killed (exit 137) when the full suite ran in one process. Phase 0 work is
tiny-tensor work, so one thread is both sufficient and faster here.

The same environment variables are exported by ``scripts/run_cpu_checks.sh`` so
CPU dry-runs behave identically outside pytest.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402  (import after env vars so they take effect)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # already initialised in this process
    pass

import gc  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _release_memory_between_tests():
    """Collect garbage after every test.

    The integration tests instantiate a fresh model + optimizer per ``run_loop``
    call; without an explicit collect, reference cycles kept several of them
    alive simultaneously and the suite hit the container's 2 GiB ceiling.
    """
    yield
    gc.collect()
