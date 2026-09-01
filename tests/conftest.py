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
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _release_memory_between_tests():
    """Collect garbage after every test.

    The integration tests instantiate a fresh model + optimizer per ``run_loop``
    call; without an explicit collect, reference cycles kept several of them
    alive simultaneously and the suite hit the container's 2 GiB ceiling.
    """
    # Data-only test files must not pay the several-hundred-MiB resident-memory
    # cost of importing torch from the global conftest.  Tests that actually
    # need torch import it themselves during collection; configure that already
    # loaded module before their first test executes.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:  # already initialised in this process
            pass
    yield
    gc.collect()
