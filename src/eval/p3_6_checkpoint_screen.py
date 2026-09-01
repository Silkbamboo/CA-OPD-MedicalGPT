"""Frozen P3.6 SFT-v3 checkpoint screen using the proven P3.5 evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.eval import p3_5_checkpoint_screen as screen


screen.TRAIN_RUN_ID = "qwen3-4b-medical-sft-v3-mcq-dominant-seed42"
screen.SCREEN_RUN_ID = "qwen3-4b-medical-sft-v3-checkpoint-screen"
screen.EXPECTED_STEPS = (150, 300, 450, 600)
screen.EXPECTED_OPTIMIZER_STEPS = 600
screen.EXPECTED_RECORDS = 9600
screen.SFT_MANIFEST_SHA256 = "eae8df56fd9985edd27e32984b155ae9dd569eadc6e6336a858b7079050e223e"
screen.SFT_MANIFEST_RELATIVE_PATH = (
    "data/manifests/sft_v3_mcq_dominant/medical_sft_v3_manifest.json"
)
screen.SCREEN_STAGE = "p3_6_sft_v3_checkpoint_screen"


def _no_more_epochs(results):  # noqa: ANN001
    if set(map(int, results)) != set(screen.EXPECTED_STEPS):
        raise screen.ScreenError("SFT-v3 checkpoint set drift")
    return False


screen.epoch_two_allowed = _no_more_epochs


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "screen", "score-existing"))
    parser.add_argument(
        "--training-run",
        default=str(screen.PERSIST_ROOT / "outputs" / screen.TRAIN_RUN_ID),
    )
    parser.add_argument(
        "--output",
        default=str(screen.PERSIST_ROOT / "outputs" / screen.SCREEN_RUN_ID),
    )
    args = parser.parse_args(argv)
    training_run = Path(args.training_run)
    output = Path(args.output)
    if args.command == "preflight":
        print(json.dumps(screen.formal_preflight(
            training_run=training_run, output_dir=output, require_gpu=False
        ), sort_keys=True))
        return 0
    if args.command == "score-existing":
        print(json.dumps(screen.score_existing_predictions(
            training_run=training_run,
            output_dir=output,
            prediction_execution_git_sha=os.environ.get("CA_OPD_P3_6_PREDICTION_GIT_SHA", ""),
        ), sort_keys=True))
        return 0
    # The shared evaluator's frozen authorization variable remains intentionally explicit.
    os.environ["CA_OPD_ALLOW_P3_5_SCREEN_GPU"] = os.environ.get(
        "CA_OPD_ALLOW_P3_6_SCREEN_GPU", ""
    )
    print(json.dumps(screen.run_screen(training_run=training_run, output_dir=output), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - future authorized GPU only
    raise SystemExit(main())
