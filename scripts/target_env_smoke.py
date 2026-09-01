#!/usr/bin/env python3
"""Validate the isolated Qwen3 GPU environment before SFT or OPD training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.env_smoke import run_target_smoke, write_smoke_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision", required=True, help="immutable 40-hex HF model revision")
    parser.add_argument("--requirements", default="env/requirements-opd.txt")
    parser.add_argument("--sft-config", default="configs/sft/qwen3_1_7b_medical.yaml")
    parser.add_argument("--output", default="outputs/smoke/target_env_smoke.json")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-two-gpus", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    report = run_target_smoke(
        model=args.model,
        requirements=args.requirements,
        sft_config=args.sft_config,
        revision=args.revision,
        require_gpu=args.require_gpu,
        require_two_gpus=args.require_two_gpus,
        local_files_only=args.local_files_only,
    )
    write_smoke_report(args.output, report)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    print(f"\nTARGET ENV SMOKE: {'PASSED' if report.passed else 'FAILED'}; report={args.output}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
