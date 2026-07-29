"""CLI for the CPU reference OPD loop.

    python -m src.opd.loop_cli --config configs/opd/dev_cpu.yaml

Prints the run summary as JSON. Separate from ``loop.py`` so importing the loop
never triggers argument parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

# Keep CPU dry-runs inside the container memory ceiling (see tests/conftest.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

from src.opd.loop import run_loop  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU reference OPD loop")
    parser.add_argument("--config", required=True, help="path to an OPD loop YAML config")
    parser.add_argument("--output-dir", default=None, help="run directory (default: config output_root/<run_id>)")
    parser.add_argument("--resume-from", default=None, help="checkpoint directory or state.pt to resume")
    parser.add_argument("--max-steps", type=int, default=None, help="override optim.max_steps")
    args = parser.parse_args(argv)

    summary = run_loop(
        args.config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        max_steps_override=args.max_steps,
    )
    printable = {k: v for k, v in summary.items() if k != "router_windows"}
    printable["router_windows"] = len(summary.get("router_windows", []))
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
