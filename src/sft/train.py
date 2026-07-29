"""Medical SFT: produces the Medical Teacher LoRA (baseline B1).

Two entry points:

* ``dry_run(config)`` — **CPU only, no model download.** Loads the real
  ``medical_sft`` split, renders every sample with the project chat template,
  applies the assistant-only loss mask and reports token statistics plus one fully
  rendered example. This is what catches template/mask/length problems before a
  GPU is rented.
* ``train(config)`` — the real run: Transformers + TRL ``SFTTrainer`` + PEFT LoRA.
  Heavy imports are lazy so the module (and its tests) work in the CPU environment
  where transformers is too old for Qwen3.

Design constraints that come from the rest of the project:

1. The prompt/completion rendering **must** be the same code path as evaluation
   (``src/data/chat.py``), otherwise SFT and eval disagree on formatting and any
   measured gain is partly a formatting artefact.
2. The loss covers assistant content only, using the same segment-wise masking
   that ``tests/test_chat_template.py`` verifies.
3. LoRA rank/target modules must match the OPD student configuration
   (rank 32 / all-linear), otherwise Medical Teacher and student differ in
   capacity and the teacher-student gap is not attributable to knowledge.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.data.access import load_split
from src.data.chat import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    MaskedExample,
    build_masked_example,
    char_tokenizer,
)
from src.data.schema import MEDICAL_SFT, Sample
from src.utils.config import FieldSpec, load_config
from src.utils.io import write_json
from src.utils.run_meta import RunMetadata, make_run_id
from src.utils.seeding import seed_everything

SFT_SCHEMA: Dict[str, object] = {
    "run": {
        "name": FieldSpec((str,)),
        "purpose": FieldSpec((str,)),
        "baseline_id": FieldSpec((str,), choices=["B1"]),
        "seed": FieldSpec((int,), bounds=(0, None)),
        "output_root": FieldSpec((str,)),
    },
    "model": {
        "path": FieldSpec((str,), doc="HF id or local path of the base model"),
        "max_seq_length": FieldSpec((int,), bounds=(64, None)),
        "attn_implementation": FieldSpec((str,), required=False, default="flash_attention_2"),
        "torch_dtype": FieldSpec((str,), required=False, default="bfloat16", choices=["bfloat16", "float16", "float32"]),
    },
    "data": {
        "data_dir": FieldSpec((str,)),
        "max_samples": FieldSpec((int,), required=False, default=None, bounds=(1, None)),
        "include_reasoning": FieldSpec((bool,), doc="wrap Complex_CoT in <think>...</think>"),
        "system_prompt": FieldSpec((str,), required=False, default=DEFAULT_SYSTEM_PROMPT),
        "drop_longer_than_max_seq": FieldSpec((bool,), required=False, default=True),
    },
    "lora": {
        "rank": FieldSpec((int,), bounds=(1, None)),
        "alpha": FieldSpec((int,), bounds=(1, None)),
        "dropout": FieldSpec((float,), bounds=(0.0, 1.0)),
        "target_modules": FieldSpec((str,)),
    },
    "optim": {
        "lr": FieldSpec((float,), bounds=(0.0, None)),
        "epochs": FieldSpec((float,), bounds=(0.0, None)),
        "per_device_batch_size": FieldSpec((int,), bounds=(1, None)),
        "gradient_accumulation_steps": FieldSpec((int,), bounds=(1, None)),
        "warmup_ratio": FieldSpec((float,), bounds=(0.0, 1.0)),
        "weight_decay": FieldSpec((float,), bounds=(0.0, None)),
        "lr_scheduler_type": FieldSpec((str,)),
        "max_grad_norm": FieldSpec((float,), bounds=(0.0, None)),
        "gradient_checkpointing": FieldSpec((bool,)),
        "save_steps": FieldSpec((int,), bounds=(1, None)),
        "logging_steps": FieldSpec((int,), bounds=(1, None)),
        "save_only_model": FieldSpec((bool,), doc="true: no optimizer.pt (disk discipline, ADR-0003)"),
    },
}


@dataclass
class DatasetReport:
    """What the CPU dry-run learned about the data."""

    num_samples: int
    num_dropped_too_long: int
    total_tokens: int
    trainable_tokens: int
    prompt_tokens: int
    length_percentiles: Dict[str, int]
    trainable_ratio: float
    example_prompt: str
    example_completion: str
    example_mask_summary: str
    tokenizer: str

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _render(sample: Sample, include_reasoning: bool, system_prompt: Optional[str], tokenizer) -> MaskedExample:
    if not sample.answer:
        raise ValueError(f"sample {sample.sample_id} has no answer; it cannot be an SFT target")
    return build_masked_example(
        tokenizer,
        user_content=sample.question,
        answer=sample.answer,
        reasoning=sample.reasoning if include_reasoning else None,
        system_prompt=system_prompt,
        template=DEFAULT_TEMPLATE,
    )


def build_examples(
    config: Dict[str, Any],
    tokenizer: Callable[[str], Sequence[int]],
) -> tuple[List[MaskedExample], int]:
    """Render the medical SFT split into masked examples.

    Returns ``(examples, dropped)``. Over-long samples are dropped (and counted)
    rather than truncated: truncating a reasoning answer mid-sentence teaches the
    model to stop mid-sentence.
    """
    data_cfg = config["data"]
    samples = load_split(
        data_cfg["data_dir"],
        MEDICAL_SFT,
        max_samples=data_cfg["max_samples"],
    )
    max_len = int(config["model"]["max_seq_length"])
    examples: List[MaskedExample] = []
    dropped = 0
    for sample in samples:
        example = _render(sample, bool(data_cfg["include_reasoning"]), data_cfg["system_prompt"], tokenizer)
        if len(example.input_ids) > max_len:
            if data_cfg["drop_longer_than_max_seq"]:
                dropped += 1
                continue
            raise ValueError(
                f"sample {sample.sample_id} renders to {len(example.input_ids)} tokens > "
                f"max_seq_length={max_len}; raise the limit or enable drop_longer_than_max_seq"
            )
        examples.append(example)
    if not examples:
        raise ValueError("no SFT examples survived rendering/filtering")
    return examples, dropped


def dry_run(
    config_path: str | Path,
    tokenizer: Optional[Callable[[str], Sequence[int]]] = None,
    tokenizer_name: str = "char",
) -> DatasetReport:
    """CPU-only data/template/mask verification. Downloads nothing."""
    config = load_config(config_path, SFT_SCHEMA)
    seed_everything(int(config["run"]["seed"]))
    tok = tokenizer or char_tokenizer
    examples, dropped = build_examples(config, tok)

    lengths = sorted(len(e.input_ids) for e in examples)

    def pct(p: float) -> int:
        idx = min(len(lengths) - 1, int(round((len(lengths) - 1) * p)))
        return lengths[idx]

    total = sum(len(e.input_ids) for e in examples)
    trainable = sum(e.trainable_tokens() for e in examples)
    first = examples[0]
    mask = first.loss_mask
    return DatasetReport(
        num_samples=len(examples),
        num_dropped_too_long=dropped,
        total_tokens=total,
        trainable_tokens=trainable,
        prompt_tokens=total - trainable,
        length_percentiles={
            "min": lengths[0],
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": lengths[-1],
            "mean": int(statistics.fmean(lengths)),
        },
        trainable_ratio=trainable / total,
        example_prompt=first.prompt_text,
        example_completion=first.completion_text,
        example_mask_summary=(
            f"{first.prompt_length} prompt tokens masked out (mask={mask[0]}...), "
            f"{first.completion_length} completion tokens trained (mask={mask[-1]})"
        ),
        tokenizer=tokenizer_name,
    )


def train(config_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Any]:  # pragma: no cover - needs GPU
    """Real SFT run. Requires transformers >= 4.51 (Qwen3), trl, peft and a GPU."""
    config = load_config(config_path, SFT_SCHEMA)
    seed_state = seed_everything(int(config["run"]["seed"]))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg, lora_cfg, optim_cfg = config["model"], config["lora"], config["optim"]
    run_cfg = config["run"]
    run_id = make_run_id(str(run_cfg["name"]), int(run_cfg["seed"]))
    out_dir = Path(output_dir) if output_dir else Path(str(run_cfg["output_root"])) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(text: str) -> List[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    examples, dropped = build_examples(config, encode)
    dataset = Dataset.from_dict(
        {
            "input_ids": [e.input_ids for e in examples],
            "attention_mask": [[1] * len(e.input_ids) for e in examples],
            # -100 marks positions excluded from the loss: prompt tokens
            "labels": [
                [tok if m else -100 for tok, m in zip(e.input_ids, e.loss_mask)] for e in examples
            ],
        }
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg["path"]),
        torch_dtype=getattr(torch, str(model_cfg["torch_dtype"])),
        attn_implementation=str(model_cfg["attn_implementation"]),
    )
    peft_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=str(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=float(optim_cfg["lr"]),
        num_train_epochs=float(optim_cfg["epochs"]),
        per_device_train_batch_size=int(optim_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(optim_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(optim_cfg["warmup_ratio"]),
        weight_decay=float(optim_cfg["weight_decay"]),
        lr_scheduler_type=str(optim_cfg["lr_scheduler_type"]),
        max_grad_norm=float(optim_cfg["max_grad_norm"]),
        gradient_checkpointing=bool(optim_cfg["gradient_checkpointing"]),
        save_steps=int(optim_cfg["save_steps"]),
        logging_steps=int(optim_cfg["logging_steps"]),
        save_only_model=bool(optim_cfg["save_only_model"]),
        max_seq_length=int(model_cfg["max_seq_length"]),
        bf16=str(model_cfg["torch_dtype"]) == "bfloat16",
        report_to=[],
        seed=int(run_cfg["seed"]),
    )

    RunMetadata(
        run_id=run_id,
        purpose=str(run_cfg["purpose"]),
        baseline_id=str(run_cfg["baseline_id"]),
        config_path=str(config_path),
        seed=int(run_cfg["seed"]),
        model=str(model_cfg["path"]),
        data_manifest_path=str(Path(config["data"]["data_dir"]) / "data_manifest.json"),
        notes="Medical SFT -> Medical Teacher LoRA",
        extra={"seed_state": seed_state.as_dict(), "dropped_too_long": dropped},
    ).save(out_dir)

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, peft_config=peft_config)
    result = trainer.train()
    trainer.save_model(str(out_dir))
    summary = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "num_examples": len(examples),
        "dropped_too_long": dropped,
        "train_runtime_seconds": getattr(result, "metrics", {}).get("train_runtime"),
        "train_loss": getattr(result, "metrics", {}).get("train_loss"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Medical SFT (produces the Medical Teacher LoRA)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only data/template/mask check")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if args.dry_run:
        report = dry_run(args.config)
        payload = report.as_dict()
        example_prompt = payload.pop("example_prompt")
        example_completion = payload.pop("example_completion")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n--- rendered example (prompt) ---\n" + example_prompt)
        print("\n--- rendered example (completion, trained) ---\n" + example_completion)
        return 0
    print(json.dumps(train(args.config, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
