"""Medical SFT: produces the Medical Teacher LoRA (baseline B1).

Two entry points:

* ``dry_run(config)`` — **CPU only, no model download.** Loads either the legacy
  fixture split or a role-scoped v2 ``medical_sft_train`` artifact, renders it,
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
3. LoRA rank/target modules must match the selected experiment family. The
   prepared 4B MVP uses rank 16 / all-linear; legacy 1.7B configs remain rank
   32. Mixing those families is rejected by their run/preflight contracts.
"""

from __future__ import annotations

import hashlib
import re
import signal
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.data.access import load_manifest_for_trainer, load_split, verify_role_records_artifact
from src.data.chat import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    MaskedExample,
    build_masked_example,
    build_masked_example_nonthinking,
    char_tokenizer,
)
from src.data.schema import (
    DATA_PROTOCOL_VERSION,
    DOMAIN_MEDICAL,
    MEDICAL_SFT,
    TASK_REASONING_SFT,
    Sample,
)
from src.utils.config import FieldSpec, load_config
from src.utils.io import write_json
from src.utils.run_meta import RunMetadata, make_run_id
from src.utils.seeding import seed_everything
from src.sft.artifacts import (
    finalize_lora_run,
    initialize_sft_run_inventory,
    record_sft_failure,
)
from src.sft.weighted import (
    attach_weighted_loss_forward,
    SupervisionWeights,
    WeightedDataCollator,
    WeightedExample,
    make_weighted_trainer_class,
    render_sft_v2_row,
)

SFT_SCHEMA: Dict[str, object] = {
    "run": {
        "name": FieldSpec((str,)),
        "purpose": FieldSpec((str,)),
        "baseline_id": FieldSpec((str,), choices=["B1"]),
        "seed": FieldSpec((int,), bounds=(0, None)),
        "output_root": FieldSpec((str,)),
        "hardware_calibration_status": FieldSpec(
            (str,),
            required=False,
            default="candidate_pending_gpu_calibration",
            choices=["candidate_pending_gpu_calibration", "measured_on_target_hardware"],
        ),
        "fixed_run_id": FieldSpec((bool,), required=False, default=False),
        "resume_from_checkpoint": FieldSpec((str,), required=False, default=None),
    },
    "model": {
        "path": FieldSpec((str,), doc="HF id or local path of the base model"),
        "revision": FieldSpec((str,), doc="immutable 40-hex HF model snapshot revision"),
        "tokenizer_revision": FieldSpec((str,), doc="immutable tokenizer snapshot revision"),
        "max_seq_length": FieldSpec((int,), bounds=(64, None)),
        "attn_implementation": FieldSpec((str,), required=False, default="flash_attention_2"),
        "torch_dtype": FieldSpec((str,), required=False, default="bfloat16", choices=["bfloat16", "float16", "float32"]),
    },
    "data": {
        "data_dir": FieldSpec((str,)),
        "protocol_version": FieldSpec(
            (str,),
            required=False,
            default="legacy-v1-fixture",
            choices=["legacy-v1-fixture", DATA_PROTOCOL_VERSION],
        ),
        "manifest_path": FieldSpec((str,), required=False, default=None),
        "records_path": FieldSpec((str,), required=False, default=None),
        "target_role": FieldSpec(
            (str,), required=False, default="medical_sft", choices=["medical_sft", "medical_sft_train"]
        ),
        "enable_thinking": FieldSpec(
            (bool,), required=False, default=False, choices=[False],
            doc="Data Protocol v2 requires Qwen3 non-thinking mode",
        ),
        "max_samples": FieldSpec((int,), required=False, default=None, bounds=(1, None)),
        "include_reasoning": FieldSpec((bool,), doc="include reasoning as ordinary assistant text; never <think>"),
        "system_prompt": FieldSpec((str,), required=False, default=DEFAULT_SYSTEM_PROMPT),
        "drop_longer_than_max_seq": FieldSpec((bool,), required=False, default=True),
        "supervision_version": FieldSpec(
            (str,),
            required=False,
            default="assistant_uniform_v1",
            choices=[
                "assistant_uniform_v1",
                "answer_first_weighted_v2",
                "mcq_dominant_task_balanced_v3",
            ],
        ),
        "answer_weight": FieldSpec((float,), required=False, default=1.0, bounds=(0.0, None)),
        "reasoning_weight": FieldSpec((float,), required=False, default=1.0, bounds=(0.0, None)),
        "eos_weight": FieldSpec((float,), required=False, default=1.0, bounds=(0.0, None)),
        "loss_chunk_tokens": FieldSpec((int,), required=False, default=64, bounds=(1, None)),
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
        "save_steps": FieldSpec(
            (int, float,),
            bounds=(0.0, None),
            doc="integer optimizer interval or a (0,1) fraction of total steps",
        ),
        "logging_steps": FieldSpec((int,), bounds=(1, None)),
        "save_only_model": FieldSpec((bool,), doc="true: no optimizer.pt (disk discipline, ADR-0003)"),
        "save_total_limit": FieldSpec((int,), required=False, default=2, bounds=(1, None)),
        "max_steps": FieldSpec((int,), required=False, default=None, bounds=(1, None)),
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
    weighted_statistics: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def validate_model_revisions(config: Dict[str, Any]) -> None:
    for key in ("revision", "tokenizer_revision"):
        value = str(config["model"][key])
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"model.{key} must be an immutable 40-hex HF commit, got {value!r}")


def resolve_resume_checkpoint(value: str | None) -> str | None:
    """Resolve only a complete LoRA checkpoint; no latest-directory guessing."""

    if value is None:
        return None
    path = Path(value).resolve()
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
        path / "trainer_state.json",
        path / "optimizer.pt",
        path / "scheduler.pt",
    )
    if not path.is_dir() or any(not item.is_file() for item in required):
        raise ValueError(f"resume checkpoint lacks exact-resume adapter/trainer state: {path}")
    return str(path)


def resolve_sft_run_id(
    name: str, seed: int, *, fixed: bool, timestamp: float | None = None
) -> str:
    """Use frozen run-card IDs for production and timestamped IDs for legacy runs."""

    if fixed:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            raise ValueError("fixed SFT run ID is unsafe")
        return name
    return make_run_id(name, seed, timestamp=timestamp)


def _render(sample: Sample, include_reasoning: bool, system_prompt: Optional[str], tokenizer) -> MaskedExample:
    if not sample.answer:
        raise ValueError(f"sample {sample.sample_id} has no answer; it cannot be an SFT target")
    return build_masked_example_nonthinking(
        tokenizer,
        user_content=sample.question,
        answer=sample.answer,
        reasoning=sample.reasoning if include_reasoning else None,
        system_prompt=system_prompt,
        template=DEFAULT_TEMPLATE,
        enable_thinking=False,
    )


def _load_sft_samples(data_cfg: Dict[str, Any]) -> List[Sample]:
    """Load legacy fixtures or a role-scoped Data Protocol v2 SFT artifact."""

    max_samples = data_cfg["max_samples"]
    if data_cfg["protocol_version"] != DATA_PROTOCOL_VERSION:
        return load_split(data_cfg["data_dir"], MEDICAL_SFT, max_samples=max_samples)

    if data_cfg["enable_thinking"] is not False:
        raise ValueError("Data Protocol v2 requires enable_thinking=false")
    if data_cfg["target_role"] != "medical_sft_train":
        raise ValueError("Data Protocol v2 SFT target_role must be medical_sft_train")
    if not data_cfg["manifest_path"] or not data_cfg["records_path"]:
        raise ValueError("Data Protocol v2 SFT requires manifest_path and records_path")
    manifest = load_manifest_for_trainer(data_cfg["manifest_path"], stage="sft")
    if "medical_sft_train" not in manifest["roles"]:
        raise PermissionError("SFT manifest does not contain medical_sft_train")
    verify_role_records_artifact(
        manifest, data_cfg["records_path"], role="medical_sft_train"
    )

    from src.utils.io import iter_jsonl

    samples: List[Sample] = []
    for row in iter_jsonl(data_cfg["records_path"]):
        if row.get("target_role") != "medical_sft_train":
            raise PermissionError(
                f"SFT record {row.get('sample_id')} has disallowed target_role={row.get('target_role')!r}"
            )
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        reasoning = row.get("reasoning")
        text_values = (question, answer, str(reasoning or ""))
        if any("<think>" in value.casefold() or "</think>" in value.casefold() for value in text_values):
            raise ValueError("Data Protocol v2 SFT record contains forbidden <think> tag")
        if not question or not answer:
            raise ValueError(f"SFT record {row.get('sample_id')} lacks question/answer")
        samples.append(
            Sample(
                source=str(row.get("source") or "unknown"),
                split=MEDICAL_SFT,
                domain=DOMAIN_MEDICAL,
                task=TASK_REASONING_SFT,
                question=question,
                answer=answer,
                reasoning=str(reasoning).strip() if reasoning else None,
                sample_id=str(row.get("sample_id") or ""),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _load_sft_v2_rows(data_cfg: Dict[str, Any]) -> list[dict[str, Any]]:
    """Load only the new role-scoped SFT-v2 mixture."""

    if data_cfg["protocol_version"] != DATA_PROTOCOL_VERSION:
        raise ValueError("weighted SFT-v2 requires Data Protocol v2")
    if data_cfg["enable_thinking"] is not False:
        raise ValueError("weighted SFT-v2 requires enable_thinking=false")
    if data_cfg["target_role"] != "medical_sft_train":
        raise ValueError("weighted SFT-v2 target_role must be medical_sft_train")
    if not data_cfg["manifest_path"] or not data_cfg["records_path"]:
        raise ValueError("weighted SFT-v2 requires manifest_path and records_path")
    manifest = load_manifest_for_trainer(data_cfg["manifest_path"], stage="sft")
    verify_role_records_artifact(
        manifest, data_cfg["records_path"], role="medical_sft_train"
    )
    from src.utils.io import iter_jsonl

    rows: list[dict[str, Any]] = []
    max_samples = data_cfg["max_samples"]
    for raw in iter_jsonl(data_cfg["records_path"]):
        row = dict(raw)
        if row.get("target_role") != "medical_sft_train":
            raise PermissionError("SFT-v2 records may only use medical_sft_train")
        if row.get("sft_v2_kind") not in {
            "medical_o1_answer_first",
            "cmb_mcq_bridge",
        }:
            raise ValueError("SFT-v2 record lacks a frozen supervision kind")
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError("SFT-v2 records artifact is empty")
    return rows


def build_weighted_examples(
    config: Dict[str, Any], tokenizer: Any
) -> tuple[list[WeightedExample], int, dict[str, Any]]:
    """Render weighted answer-first examples and aggregate token contributions."""

    data_cfg = config["data"]
    if data_cfg["supervision_version"] != "answer_first_weighted_v2":
        raise ValueError("build_weighted_examples requires answer_first_weighted_v2")
    weights = SupervisionWeights(
        answer=float(data_cfg["answer_weight"]),
        reasoning=float(data_cfg["reasoning_weight"]),
        eos=float(data_cfg["eos_weight"]),
    )
    examples: list[WeightedExample] = []
    dropped = 0
    segments = {"answer": 0, "reasoning": 0, "eos": 0}
    contributions = {"answer": 0.0, "reasoning": 0.0, "eos": 0.0}
    by_kind: dict[str, int] = {}
    for row in _load_sft_v2_rows(data_cfg):
        example = render_sft_v2_row(
            row,
            tokenizer=tokenizer,
            weights=weights,
            max_seq_length=int(config["model"]["max_seq_length"]),
            system_prompt=str(data_cfg["system_prompt"]),
        )
        if example is None:
            if not data_cfg["drop_longer_than_max_seq"]:
                raise ValueError("SFT-v2 row exceeds max_seq_length; truncation is forbidden")
            dropped += 1
            continue
        examples.append(example)
        kind = str(row["sft_v2_kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
        for name in segments:
            segments[name] += example.segment_token_counts[name]
            contributions[name] += example.segment_weighted_contribution[name]
    if not examples:
        raise ValueError("no SFT-v2 examples survived exact length filtering")
    return examples, dropped, {
        "token_counts": segments,
        "weighted_contribution": contributions,
        "weighted_contribution_total": sum(contributions.values()),
        "examples_by_kind": dict(sorted(by_kind.items())),
        "first_assistant_token_supervised": all(
            example.labels[example.prompt_length] != -100
            and example.loss_weights[example.prompt_length] > 0
            for example in examples
        ),
        "prompt_weight_zero": all(
            all(weight == 0.0 for weight in example.loss_weights[: example.prompt_length])
            for example in examples
        ),
        "eos_supervised": all(example.loss_weights[-1] == weights.eos for example in examples),
    }


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
    samples = _load_sft_samples(data_cfg)
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
    validate_model_revisions(config)
    seed_everything(int(config["run"]["seed"]))
    if config["data"]["supervision_version"] == "answer_first_weighted_v2":
        if tokenizer is None:
            raise ValueError("weighted SFT-v2 dry-run requires the fixed real tokenizer")
        examples, dropped, weighted = build_weighted_examples(config, tokenizer)
        lengths = sorted(len(example.input_ids) for example in examples)

        def weighted_pct(p: float) -> int:
            index = min(len(lengths) - 1, int(round((len(lengths) - 1) * p)))
            return lengths[index]

        total = sum(len(example.input_ids) for example in examples)
        trainable = sum(
            sum(label != -100 for label in example.labels) for example in examples
        )
        first = examples[0]
        return DatasetReport(
            num_samples=len(examples),
            num_dropped_too_long=dropped,
            total_tokens=total,
            trainable_tokens=trainable,
            prompt_tokens=total - trainable,
            length_percentiles={
                "min": lengths[0],
                "p50": weighted_pct(0.5),
                "p90": weighted_pct(0.9),
                "p99": weighted_pct(0.99),
                "max": lengths[-1],
                "mean": int(statistics.fmean(lengths)),
            },
            trainable_ratio=trainable / total,
            example_prompt=first.prompt_text,
            example_completion=first.target_text,
            example_mask_summary=(
                f"{first.prompt_length} prompt tokens carry weight 0; "
                f"{trainable} assistant/EOS tokens are supervised across the dataset"
            ),
            tokenizer=tokenizer_name,
            weighted_statistics=weighted,
        )
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



def lora_config_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Exact PEFT constructor contract, shared by training and env smoke."""
    lora_cfg = config["lora"]
    return {
        "r": int(lora_cfg["rank"]),
        "lora_alpha": int(lora_cfg["alpha"]),
        "lora_dropout": float(lora_cfg["dropout"]),
        "target_modules": str(lora_cfg["target_modules"]),
        "task_type": "CAUSAL_LM",
    }


def sft_config_kwargs(config: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    """Exact TRL 0.23 ``SFTConfig`` contract, shared by train and smoke.

    TRL 0.23 renamed the sequence cap to ``max_length``. Keeping this mapping in
    one function prevents the smoke gate and paid training path from drifting.
    """
    model_cfg, optim_cfg, run_cfg = config["model"], config["optim"], config["run"]
    raw_save_steps = optim_cfg["save_steps"]
    if raw_save_steps <= 0 or (
        isinstance(raw_save_steps, float)
        and raw_save_steps >= 1.0
        and not raw_save_steps.is_integer()
    ):
        raise ValueError("optim.save_steps must be a positive integer or a fraction in (0,1)")
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(optim_cfg["lr"]),
        "num_train_epochs": float(optim_cfg["epochs"]),
        "per_device_train_batch_size": int(optim_cfg["per_device_batch_size"]),
        "gradient_accumulation_steps": int(optim_cfg["gradient_accumulation_steps"]),
        "warmup_ratio": float(optim_cfg["warmup_ratio"]),
        "weight_decay": float(optim_cfg["weight_decay"]),
        "lr_scheduler_type": str(optim_cfg["lr_scheduler_type"]),
        "max_grad_norm": float(optim_cfg["max_grad_norm"]),
        "gradient_checkpointing": bool(optim_cfg["gradient_checkpointing"]),
        "save_steps": (
            float(raw_save_steps)
            if isinstance(raw_save_steps, float) and raw_save_steps < 1.0
            else int(raw_save_steps)
        ),
        "logging_steps": int(optim_cfg["logging_steps"]),
        "save_only_model": bool(optim_cfg["save_only_model"]),
        "save_total_limit": int(optim_cfg["save_total_limit"]),
        "max_length": int(model_cfg["max_seq_length"]),
        "bf16": str(model_cfg["torch_dtype"]) == "bfloat16",
        "report_to": [],
        "seed": int(run_cfg["seed"]),
    }
    if optim_cfg["max_steps"] is not None:
        kwargs["max_steps"] = int(optim_cfg["max_steps"])
    if config["data"]["supervision_version"] == "answer_first_weighted_v2":
        kwargs.update(
            {
                "dataset_kwargs": {"skip_prepare_dataset": True},
                "remove_unused_columns": False,
            }
        )
    return kwargs


def sft_trainer_kwargs(
    *, model: Any, args: Any, train_dataset: Any, peft_config: Any, tokenizer: Any
) -> Dict[str, Any]:
    """Exact TRL 0.23 trainer contract shared by target smoke and training."""
    return {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
    }


def train(config_path: str | Path, output_dir: Optional[str | Path] = None) -> Dict[str, Any]:  # pragma: no cover - needs GPU
    """Real SFT run. Requires transformers >= 4.51 (Qwen3), trl, peft and a GPU."""
    config = load_config(config_path, SFT_SCHEMA)
    validate_model_revisions(config)
    seed_state = seed_everything(int(config["run"]["seed"]))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg, optim_cfg = config["model"], config["optim"]
    run_cfg = config["run"]
    run_id = resolve_sft_run_id(
        str(run_cfg["name"]),
        int(run_cfg["seed"]),
        fixed=bool(run_cfg["fixed_run_id"]),
    )
    out_dir = Path(output_dir) if output_dir else Path(str(run_cfg["output_root"])) / run_id
    manifest_path = (
        Path(str(config["data"]["manifest_path"]))
        if config["data"]["protocol_version"] == DATA_PROTOCOL_VERSION
        else Path(config["data"]["data_dir"]) / "data_manifest.json"
    )
    if run_cfg["resume_from_checkpoint"] is None:
        if config["data"]["protocol_version"] == DATA_PROTOCOL_VERSION:
            initialize_sft_run_inventory(
                out_dir,
                config_path=config_path,
                data_manifest_path=manifest_path,
                run_id=run_id,
            )
        else:
            if out_dir.exists() and any(out_dir.iterdir()):
                raise FileExistsError(f"SFT output directory is not new/empty: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.is_dir():
        raise FileNotFoundError("SFT resume requires the existing run directory")

    class SFTTermination(RuntimeError):
        pass

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def on_sigterm(signum, frame):  # noqa: ARG001
        raise SFTTermination("SIGTERM received; trainer state preserved for explicit resume")

    signal.signal(signal.SIGTERM, on_sigterm)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg["path"]), revision=str(model_cfg["tokenizer_revision"]), local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode(text: str) -> List[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]
        weighted_mode = config["data"]["supervision_version"] == "answer_first_weighted_v2"
        weighted_statistics: dict[str, Any] | None = None
        if weighted_mode:
            examples, dropped, weighted_statistics = build_weighted_examples(config, tokenizer)
            dataset = Dataset.from_dict(
                {
                    "input_ids": [example.input_ids for example in examples],
                    "attention_mask": [example.attention_mask for example in examples],
                    "labels": [example.labels for example in examples],
                    "loss_weights": [example.loss_weights for example in examples],
                }
            )
        else:
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
            revision=str(model_cfg["revision"]),
            local_files_only=True,
            torch_dtype=getattr(torch, str(model_cfg["torch_dtype"])),
            attn_implementation=str(model_cfg["attn_implementation"]),
        )
        if weighted_mode:
            model.config.use_cache = False
        peft_config = LoraConfig(**lora_config_kwargs(config))
        sft_config = SFTConfig(**sft_config_kwargs(config, out_dir))

        RunMetadata(
        run_id=run_id,
        purpose=str(run_cfg["purpose"]),
        baseline_id=str(run_cfg["baseline_id"]),
        config_path=str(config_path),
        seed=int(run_cfg["seed"]),
        model=str(model_cfg["path"]),
        data_manifest_path=str(manifest_path),
        notes="Medical SFT -> Medical Teacher LoRA",
        extra={
            "seed_state": seed_state.as_dict(),
            "dropped_too_long": dropped,
            "model_revision": str(model_cfg["revision"]),
            "tokenizer_revision": str(model_cfg["tokenizer_revision"]),
            "supervision_version": str(config["data"]["supervision_version"]),
            "weighted_supervision": weighted_statistics,
        },
        ).save(out_dir)

        trainer_kwargs = sft_trainer_kwargs(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
        )
        trainer_class = SFTTrainer
        if weighted_mode:
            trainer_class = make_weighted_trainer_class(SFTTrainer)
            trainer_kwargs["data_collator"] = WeightedDataCollator(tokenizer)
        trainer = trainer_class(**trainer_kwargs)
        if weighted_mode:
            attach_weighted_loss_forward(
                trainer.model, chunk_tokens=int(config["data"]["loss_chunk_tokens"])
            )
        result = trainer.train(
            resume_from_checkpoint=resolve_resume_checkpoint(run_cfg["resume_from_checkpoint"])
        )
        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        metrics = dict(getattr(result, "metrics", {}))
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        adapter_manifest = finalize_lora_run(
            out_dir,
            adapter_dir=adapter_dir,
            run_id=run_id,
            model_id=str(model_cfg["path"]),
            model_revision=str(model_cfg["revision"]),
            tokenizer_revision=str(model_cfg["tokenizer_revision"]),
            data_manifest_sha256=manifest_sha,
            metrics=metrics,
            log_history=list(trainer.state.log_history),
        )
        summary = {
            "run_id": run_id,
            "run_dir": str(out_dir),
            "num_examples": len(examples),
            "dropped_too_long": dropped,
            "train_runtime_seconds": metrics.get("train_runtime"),
            "train_loss": metrics.get("train_loss"),
            "adapter_sha256": adapter_manifest["adapter_sha256"],
            "actual_cost_cny": None,
            "status": "completed_pending_cost_reconciliation",
        }
        if weighted_statistics is not None:
            summary["weighted_supervision"] = weighted_statistics
        write_json(out_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        record_sft_failure(out_dir, run_id=run_id, reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Medical SFT (produces the Medical Teacher LoRA)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only data/template/mask check")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    if args.dry_run:
        parsed = load_config(args.config, SFT_SCHEMA)
        if parsed["data"]["supervision_version"] == "answer_first_weighted_v2":
            from transformers import AutoTokenizer

            fixed_tokenizer = AutoTokenizer.from_pretrained(
                str(parsed["model"]["path"]),
                revision=str(parsed["model"]["tokenizer_revision"]),
                local_files_only=True,
            )
            if fixed_tokenizer.pad_token is None:
                fixed_tokenizer.pad_token = fixed_tokenizer.eos_token
            report = dry_run(
                args.config,
                tokenizer=fixed_tokenizer,
                tokenizer_name="fixed_qwen3_4b_local",
            )
        else:
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
