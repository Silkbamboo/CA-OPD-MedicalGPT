#!/usr/bin/env python3
"""Run one resumable, CPU-only phase of the P2 formal data build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.formal_pipeline_v2 import FormalPipeline


PHASES = (
    "download",
    "medqa-conflicts",
    "normalize",
    "preselect",
    "dedup",
    "allocate",
    "tokenizer-download",
    "tokenizer-audit",
    "export",
    "all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/formal_v2.yaml"))
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args(argv)
    pipeline = FormalPipeline(args.config)
    methods = {
        "download": pipeline.download_sources,
        "medqa-conflicts": pipeline.audit_medqa_conflicts,
        "normalize": pipeline.normalize,
        "preselect": pipeline.preselect_taxonomy_and_source_quotas,
        "dedup": pipeline.build_near_duplicates,
        "allocate": pipeline.apply_taxonomy_and_quotas,
        "tokenizer-download": pipeline.download_tokenizer,
        "tokenizer-audit": pipeline.audit_token_lengths,
        "export": pipeline.export,
    }
    if args.phase == "all":
        order = (
            "download",
            "medqa-conflicts",
            "normalize",
            "preselect",
            "dedup",
            "allocate",
            "tokenizer-download",
            "tokenizer-audit",
            "export",
        )
        result = {phase: methods[phase]() for phase in order}
    else:
        result = methods[args.phase]()
    # The result contains counts, hashes, paths and statuses only. Adapters never
    # place question/answer text in phase summaries.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
