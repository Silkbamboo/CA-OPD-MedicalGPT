"""Explicit two-process DDP runner for frozen answer-first weighted SFT-v2.

This module is safe to import on CPU. Heavy libraries, CUDA device binding and process-group
initialization live inside :func:`run`, which is only reached by the separately authorized
``torchrun`` launcher.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.access import load_manifest_for_trainer, verify_role_records_artifact
from src.sft.artifacts import finalize_lora_run, initialize_sft_run_inventory, record_sft_failure
from src.sft.ddp import (
    DDPExecutionContract,
    DistributedEnvironment,
    accumulation_windows,
    assert_global_sample_coverage,
    classify_memory_margin,
    ddp_scaled_loss,
    distributed_sample_indices,
    local_weight_denominator,
    rank_zero_write_json,
)
from src.sft.train import SFT_SCHEMA, lora_config_kwargs, validate_model_revisions
from src.sft.weighted import (
    SupervisionWeights,
    WeightedDataCollator,
    attach_weighted_loss_forward,
    render_sft_v2_row,
)
from src.sft.v3 import (
    SFTV3Kind,
    build_sft_v3_smoke_rank_rows,
    build_task_balanced_rank_rows,
    render_sft_v3_row,
    sft_v3_task_counts_through_step,
    task_for_optimizer_step,
)
from src.utils.config import FieldSpec, load_config
from src.utils.io import iter_jsonl


DDP_SCHEMA = {
    "launch_mode": FieldSpec((str,), choices=["ddp"]),
    "backend": FieldSpec((str,), choices=["nccl"]),
    "world_size": FieldSpec((int,), choices=[2]),
    "broadcast_buffers": FieldSpec((bool,), choices=[False]),
    "find_unused_parameters": FieldSpec((bool,), choices=[False]),
    "gradient_as_bucket_view": FieldSpec((bool,), choices=[True]),
    "bucket_cap_mb": FieldSpec((int,), choices=[16]),
    "global_weighted_denominator": FieldSpec((bool,), choices=[True]),
    "rank_zero_only_writes": FieldSpec((bool,), choices=[True]),
    "fresh_base_per_rank": FieldSpec((bool,), choices=[True]),
    "device_map": FieldSpec((str,), choices=["none"]),
}
DDP_SFT_SCHEMA = {
    **SFT_SCHEMA,
    "optim": {
        **SFT_SCHEMA["optim"],
        "optimizer": FieldSpec((str,), choices=["adamw_torch_fused"]),
    },
    "distributed": DDP_SCHEMA,
}
SFT_V3_SMOKE_RUN_ID = "qwen3-4b-medical-sft-v3-four-step-gpu-smoke-seed42-retry1"


def load_ddp_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path, DDP_SFT_SCHEMA)
    validate_model_revisions(config)
    frozen = DDPExecutionContract.frozen()
    distributed = config["distributed"]
    if int(config["optim"]["per_device_batch_size"]) != frozen.per_device_micro_batch_size:
        raise ValueError("P3.5 per-device microbatch must remain one")
    if int(config["optim"]["gradient_accumulation_steps"]) != frozen.gradient_accumulation_steps:
        raise ValueError("P3.5 gradient accumulation must remain eight")
    if int(distributed["world_size"]) != frozen.world_size:
        raise ValueError("P3.5 world_size must remain two")
    supervision = config["data"]["supervision_version"]
    if supervision not in {"answer_first_weighted_v2", "mcq_dominant_task_balanced_v3"}:
        raise ValueError("DDP supports only the frozen SFT-v2/v3 supervision protocols")
    if supervision == "mcq_dominant_task_balanced_v3":
        if config["data"]["include_reasoning"] is not False:
            raise ValueError("SFT-v3 excludes Complex_CoT supervision")
        if any(float(config["data"][key]) != 1.0 for key in (
            "answer_weight", "reasoning_weight", "eos_weight"
        )):
            raise ValueError("SFT-v3 freezes equal target/EOS weights")
    if config["run"]["resume_from_checkpoint"] is not None:
        raise ValueError("P3.5 must initialize a fresh LoRA from the Base model")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rendered_dataset(config: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = config["data"]
    manifest = load_manifest_for_trainer(data["manifest_path"], stage="sft")
    verify_role_records_artifact(manifest, data["records_path"], role="medical_sft_train")
    weights = SupervisionWeights(
        answer=float(data["answer_weight"]),
        reasoning=float(data["reasoning_weight"]),
        eos=float(data["eos_weight"]),
    )
    rendered: list[dict[str, Any]] = []
    dropped = 0
    by_kind: dict[str, int] = {}
    token_counts = {"answer": 0, "reasoning": 0, "eos": 0}
    weighted_contribution = {"answer": 0.0, "reasoning": 0.0, "eos": 0.0}
    for row in iter_jsonl(data["records_path"]):
        if row.get("target_role") != "medical_sft_train" or "final" in str(row.get("target_role", "")):
            raise PermissionError("DDP SFT may read only medical_sft_train")
        if data["supervision_version"] == "mcq_dominant_task_balanced_v3":
            example = render_sft_v3_row(
                row,
                tokenizer=tokenizer,
                max_seq_length=int(config["model"]["max_seq_length"]),
                system_prompt=str(data["system_prompt"]),
            )
            kind = str(row["sft_v3_kind"])
        else:
            example = render_sft_v2_row(
                row,
                tokenizer=tokenizer,
                weights=weights,
                max_seq_length=int(config["model"]["max_seq_length"]),
                system_prompt=str(data["system_prompt"]),
            )
            kind = str(row["sft_v2_kind"])
        if example is None:
            dropped += 1
            continue
        rendered_row = {
                "sample_id": str(row["sample_id"]),
                "input_ids": example.input_ids,
                "attention_mask": example.attention_mask,
                "labels": example.labels,
                "loss_weights": example.loss_weights,
                "sequence_length": len(example.input_ids),
                "first_supervised_token_id": next(
                    int(value) for value in example.labels if int(value) != -100
                ),
                "supervised_token_count": sum(int(value) != -100 for value in example.labels),
                "prompt_weights_zero": all(
                    float(value) == 0.0 for value in example.loss_weights[: example.prompt_length]
                ),
                "eos_supervised": (
                    int(example.labels[-1]) == int(tokenizer.eos_token_id)
                    and float(example.loss_weights[-1]) > 0.0
                ),
                "reasoning_token_count": int(example.segment_token_counts["reasoning"]),
            }
        if data["supervision_version"] == "mcq_dominant_task_balanced_v3":
            rendered_row["sft_v3_kind"] = kind
        rendered.append(rendered_row)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        for segment in token_counts:
            token_counts[segment] += example.segment_token_counts[segment]
            weighted_contribution[segment] += example.segment_weighted_contribution[segment]
    expected = 9600 if data["supervision_version"] == "mcq_dominant_task_balanced_v3" else 9500
    if len(rendered) != expected or dropped:
        raise ValueError(
            f"frozen DDP rendering expected {expected} rows and zero drops; got {len(rendered)} + {dropped}"
        )
    return rendered, {
        "records": len(rendered),
        "dropped": dropped,
        "examples_by_kind": dict(sorted(by_kind.items())),
        "token_counts": token_counts,
        "weighted_contribution": weighted_contribution,
    }


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), allow_nan=False, sort_keys=True) + "\n")


def _verify_adapter_checkpoint(path: Path, *, expected_step: int) -> dict[str, Any]:
    from safetensors import safe_open

    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    state = path / "trainer_state.json"
    optimizer = path / "optimizer.pt"
    scheduler = path / "scheduler.pt"
    required = (config, weights, state, optimizer, scheduler)
    if any(not item.is_file() for item in required):
        raise RuntimeError(f"DDP checkpoint incomplete: {path}")
    state_payload = json.loads(state.read_text(encoding="utf-8"))
    if int(state_payload.get("global_step", -1)) != expected_step:
        raise RuntimeError("DDP checkpoint trainer_state step mismatch")
    finite = True
    lora_b_nonzero = False
    tensor_count = 0
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensor_count += 1
            finite = finite and bool(tensor.isfinite().all().item())
            if "lora_B" in name and bool(tensor.ne(0).any().item()):
                lora_b_nonzero = True
            del tensor
    if not finite or not lora_b_nonzero or tensor_count <= 0:
        raise RuntimeError("DDP checkpoint tensor integrity failed")
    return {
        "step": expected_step,
        "tensor_count": tensor_count,
        "finite": finite,
        "lora_b_nonzero": lora_b_nonzero,
        "adapter_model_sha256": _sha256(weights),
    }


def _save_checkpoint(
    *,
    dist: Any,
    ddp_model: Any,
    optimizer: Any,
    scheduler: Any,
    run_dir: Path,
    step: int,
    rank: int,
    trainer_state: Mapping[str, Any],
) -> dict[str, Any] | None:
    dist.barrier()
    verification = None
    if rank == 0:
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=False, exist_ok=False)
        ddp_model.module.save_pretrained(checkpoint, safe_serialization=True)
        rank_zero_write_json(
            checkpoint / "trainer_state.json", {**dict(trainer_state), "global_step": step}, rank=0
        )
        import torch

        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        verification = _verify_adapter_checkpoint(checkpoint, expected_step=step)
    dist.barrier()
    return verification


def _calibration_rank_ids(path: Path, env: DistributedEnvironment) -> tuple[list[str], list[list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_gpu" or payload.get("world_size") != 2:
        raise ValueError("invalid P3.5 calibration selection manifest")
    primary = list(payload["primary_window"][f"rank{env.rank}"])
    conditional = [list(window[f"rank{env.rank}"]) for window in payload["conditional_windows"]]
    if len(primary) != 8 or len(conditional) != 3 or any(len(window) != 8 for window in conditional):
        raise ValueError("calibration manifest must bind 8 + 3x8 rows per rank")
    return primary, conditional


def _formal_rank_rows(dataset: list[dict[str, Any]], env: DistributedEnvironment, *, seed: int) -> list[dict[str, Any]]:
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(
        dataset,
        num_replicas=env.world_size,
        rank=env.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(0)
    indices = list(sampler)
    expected = distributed_sample_indices(
        len(dataset), rank=env.rank, world_size=env.world_size, seed=seed, epoch=0
    )
    if indices != expected:
        raise RuntimeError("DistributedSampler order differs from the frozen coverage contract")
    return [dataset[index] for index in indices]


def _assert_trainable_parameters_synchronized(dist: Any, model: Any, *, device: Any) -> float:
    import torch

    maximum = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).abs().max().float())
        del reference
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    value = float(maximum.cpu())
    if value != 0.0:
        raise RuntimeError(f"LoRA parameters diverged across DDP ranks: max_abs_diff={value}")
    return value


def _calibration_rank_rows(
    dataset: list[dict[str, Any]], manifest_path: Path, env: DistributedEnvironment
) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary, conditional = _calibration_rank_ids(manifest_path, env)
    by_id = {str(row["sample_id"]): row for row in dataset}
    all_primary_ids = list(payload["primary_window"]["rank0"]) + list(
        payload["primary_window"]["rank1"]
    )
    actual_lengths = [int(row["sequence_length"]) for row in dataset]
    primary_lengths = [int(by_id[sample_id]["sequence_length"]) for sample_id in all_primary_ids]
    sorted_lengths = sorted(actual_lengths)
    p95 = sorted_lengths[min(len(sorted_lengths) - 1, int(round((len(sorted_lengths) - 1) * 0.95)))]
    if max(primary_lengths) != max(actual_lengths) or min(primary_lengths) < p95:
        raise ValueError("calibration primary window is not the actual rendered P95-P100 worst case")
    ordered_ids = primary + [sample_id for window in conditional for sample_id in window]
    if any(sample_id not in by_id for sample_id in ordered_ids):
        raise ValueError("calibration manifest references a missing formal row")
    return [by_id[sample_id] for sample_id in ordered_ids], len(primary)


def run(
    config_path: str | Path,
    *,
    mode: str,
    calibration_manifest: str | Path | None = None,
) -> dict[str, Any]:  # pragma: no cover - authorized GPU only
    """Run future GPU calibration or formal training inside a torchrun child."""

    if mode not in {"calibration", "smoke", "train"}:
        raise ValueError("mode must be calibration, smoke, or train")
    if mode == "calibration" and calibration_manifest is None:
        raise ValueError("calibration requires a frozen sample manifest")
    config = load_ddp_config(config_path)
    env = DistributedEnvironment.from_environ()

    # Heavy imports happen only after torchrun has created this child. Device binding precedes
    # process-group initialization and every model-loading call.
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    torch.cuda.set_device(env.local_rank)
    device = torch.device("cuda", env.local_rank)
    dist.init_process_group(backend="nccl", rank=env.rank, world_size=env.world_size)
    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    is_v3 = config["data"]["supervision_version"] == "mcq_dominant_task_balanced_v3"
    if mode == "smoke" and not is_v3:
        raise ValueError("four-step smoke is defined only for frozen SFT-v3")
    run_id = SFT_V3_SMOKE_RUN_ID if mode == "smoke" else str(config["run"]["name"])
    run_dir = Path(str(config["run"]["output_root"])) / run_id
    manifest_path = Path(str(config["data"]["manifest_path"]))
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise RuntimeError("SIGTERM received by DDP rank")

    signal.signal(signal.SIGTERM, _sigterm)
    started = time.monotonic()
    try:
        if env.rank == 0:
            initialize_sft_run_inventory(
                run_dir,
                config_path=config_path,
                data_manifest_path=manifest_path,
                run_id=run_id,
            )
            rank_zero_write_json(
                run_dir / "metadata.json",
                {
                    "run_id": run_id,
                    "stage": (
                        "sft_v3_four_step_gpu_smoke"
                        if mode == "smoke"
                        else "sft_ddp_calibration" if mode == "calibration" else "sft_ddp_formal"
                    ),
                    "status": "running",
                    "git_sha": os.environ.get("CA_OPD_GIT_SHA"),
                    "git_dirty": False,
                    "model_revision": str(config["model"]["revision"]),
                    "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
                    "world_size": 2,
                    "final_authorized": False,
                    "actual_cost_cny": None,
                },
                rank=0,
            )
        dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(
            str(config["model"]["path"]),
            revision=str(config["model"]["tokenizer_revision"]),
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dataset, supervision_summary = _load_rendered_dataset(config, tokenizer)
        if mode in {"smoke", "train"}:
            if is_v3:
                all_rank_rows = [
                    build_task_balanced_rank_rows(
                        dataset, rank=rank, world_size=2, seed=seed, accumulation_steps=8
                    )
                    for rank in range(2)
                ]
                local_rows = (
                    build_sft_v3_smoke_rank_rows(
                        dataset,
                        rank=env.rank,
                        world_size=2,
                        seed=seed,
                        accumulation_steps=8,
                    )
                    if mode == "smoke"
                    else all_rank_rows[env.rank]
                )
                all_ids = [row["sample_id"] for values in all_rank_rows for row in values]
                expected_ids = {row["sample_id"] for row in dataset}
                if len(all_ids) != 9600 or set(all_ids) != expected_ids or len(set(all_ids)) != 9600:
                    raise RuntimeError("SFT-v3 task-balanced sample coverage failed")
                coverage = {
                    "world_size": 2,
                    "rank_sample_counts": [len(values) for values in all_rank_rows],
                    "global_unique_samples": len(set(all_ids)),
                    "duplicate_samples": len(all_ids) - len(set(all_ids)),
                    "missing_samples": len(expected_ids - set(all_ids)),
                    "rank_sample_id_sha256": [
                        hashlib.sha256(
                            "".join(f"{row['sample_id']}\n" for row in values).encode("utf-8")
                        ).hexdigest()
                        for values in all_rank_rows
                    ],
                    "task_schedule": "CMB,CMB,CMB,Medical-O1",
                }
                if mode == "smoke":
                    smoke_ids = [
                        row["sample_id"]
                        for rank in range(2)
                        for row in build_sft_v3_smoke_rank_rows(
                            dataset,
                            rank=rank,
                            world_size=2,
                            seed=seed,
                            accumulation_steps=8,
                        )
                    ]
                    if len(smoke_ids) != 64 or len(set(smoke_ids)) != 64:
                        raise RuntimeError("SFT-v3 GPU smoke rank rows overlap")
                    coverage = {
                        "world_size": 2,
                        "rank_sample_counts": [32, 32],
                        "global_unique_samples": 64,
                        "duplicate_samples": 0,
                        "missing_samples": 0,
                        "task_schedule": "CMB,CMB,CMB,Medical-O1",
                        "formal_records_verified": 9600,
                    }
            else:
                local_rows = _formal_rank_rows(dataset, env, seed=seed)
                all_indices = [
                    distributed_sample_indices(len(dataset), rank=rank, world_size=2, seed=seed, epoch=0)
                    for rank in range(2)
                ]
                coverage = assert_global_sample_coverage(
                    all_indices, expected_size=9500, expected_per_rank=4750
                )
                coverage["rank_sample_id_sha256"] = [
                    hashlib.sha256(
                        "".join(f"{dataset[index]['sample_id']}\n" for index in indices).encode("utf-8")
                    ).hexdigest()
                    for indices in all_indices
                ]
        else:
            local_rows, _ = _calibration_rank_rows(
                dataset, Path(str(calibration_manifest)), env
            )
            coverage = {
                "world_size": 2,
                "rank_sample_counts": [32, 32],
                "calibration_manifest_sha256": _sha256(Path(str(calibration_manifest))),
            }
        collator = WeightedDataCollator(tokenizer)
        local_batches = (collator([row]) for row in local_rows)

        model = AutoModelForCausalLM.from_pretrained(
            str(config["model"]["path"]),
            revision=str(config["model"]["revision"]),
            local_files_only=True,
            torch_dtype=getattr(torch, str(config["model"]["torch_dtype"])),
            attn_implementation=str(config["model"]["attn_implementation"]),
        )
        model.config.use_cache = False
        if config["optim"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = get_peft_model(model, LoraConfig(**lora_config_kwargs(config)))
        model.enable_input_require_grads()
        attach_weighted_loss_forward(
            model, chunk_tokens=int(config["data"]["loss_chunk_tokens"])
        )
        model.to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[env.local_rank],
            output_device=env.local_rank,
            **DDPExecutionContract.frozen().ddp_kwargs,
        )
        trainable = [parameter for parameter in ddp_model.parameters() if parameter.requires_grad]
        initial_trainable = (
            [parameter.detach().clone() for parameter in trainable]
            if mode in {"calibration", "smoke"}
            else []
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["optim"]["lr"]),
            weight_decay=float(config["optim"]["weight_decay"]),
            fused=True,
        )
        steps_per_epoch = math.ceil(len(local_rows) / int(config["optim"]["gradient_accumulation_steps"]))
        smoke_expected_steps = 4
        expected_steps = (
            600
            if is_v3
            else 594
        )
        if mode == "train" and steps_per_epoch != expected_steps:
            raise RuntimeError(f"formal DDP expected {expected_steps} steps, got {steps_per_epoch}")
        if mode == "smoke" and steps_per_epoch != smoke_expected_steps:
            raise RuntimeError(f"SFT-v3 smoke expected four steps, got {steps_per_epoch}")
        total_steps = steps_per_epoch
        scheduler = get_scheduler(
            str(config["optim"]["lr_scheduler_type"]),
            optimizer=optimizer,
            num_warmup_steps=round(total_steps * float(config["optim"]["warmup_ratio"])),
            num_training_steps=total_steps,
        )
        if mode == "train":
            save_steps = (
                {150, 300, 450, 600}
                if config["data"]["supervision_version"] == "mcq_dominant_task_balanced_v3"
                else {149, 297, 446, 594}
            )
        elif mode == "smoke":
            save_steps = {4}
        else:
            save_steps = {1, 4}
        global_step = 0
        log_history: list[dict[str, Any]] = []
        calibration_gate: dict[str, Any] | None = None
        calibration_reserved_windows: list[list[int]] = []
        optimizer.zero_grad(set_to_none=True)
        for window_index, window in enumerate(
            accumulation_windows(
                local_batches,
                accumulation_steps=int(config["optim"]["gradient_accumulation_steps"]),
            )
        ):
            window_rows = local_rows[
                window_index * int(config["optim"]["gradient_accumulation_steps"]) :
                (window_index + 1) * int(config["optim"]["gradient_accumulation_steps"])
            ]
            task = None
            if is_v3:
                expected_task = task_for_optimizer_step(window_index).value
                actual_tasks = {str(row.get("sft_v3_kind")) for row in window_rows}
                if actual_tasks != {expected_task}:
                    raise RuntimeError(
                        f"SFT-v3 task-pure window drift at step {window_index + 1}: {actual_tasks}"
                    )
                task = expected_task
            window_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            moved = [_move_batch(batch, device) for batch in window]
            local_tokens = sum(
                int(batch["attention_mask"].sum().detach().cpu()) for batch in moved
            )
            local_denominator = sum(
                (local_weight_denominator(batch["labels"], batch["loss_weights"]) for batch in moved),
                start=torch.zeros((), device=device),
            )
            global_denominator = local_denominator.detach().clone()
            before_denominator_reduce = time.monotonic()
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
            denominator_all_reduce_seconds = time.monotonic() - before_denominator_reduce
            local_numerator_total = torch.zeros((), device=device)
            forward_seconds = backward_seconds = 0.0
            for micro_index, batch in enumerate(moved):
                synchronization = (
                    ddp_model.no_sync()
                    if micro_index < len(moved) - 1
                    else contextlib.nullcontext()
                )
                with synchronization:
                    before = time.monotonic()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = ddp_model(**batch)
                        numerator = output["weighted_loss_numerator"].sum()
                    forward_seconds += time.monotonic() - before
                    local_numerator_total += numerator.detach()
                    before = time.monotonic()
                    ddp_scaled_loss(
                        numerator, global_denominator, world_size=env.world_size
                    ).backward()
                    backward_seconds += time.monotonic() - before
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(config["optim"]["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("DDP gradient norm is non-finite")
            before_optimizer = time.monotonic()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_seconds = time.monotonic() - before_optimizer
            global_step += 1
            parameter_sync_max_abs = None
            if mode in {"calibration", "smoke"} or global_step in save_steps:
                parameter_sync_max_abs = _assert_trainable_parameters_synchronized(
                    dist, ddp_model, device=device
                )
            global_numerator = local_numerator_total.clone()
            before_numerator_reduce = time.monotonic()
            dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
            numerator_all_reduce_seconds = time.monotonic() - before_numerator_reduce
            loss = global_numerator / global_denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("DDP global weighted loss is non-finite")
            window_seconds = time.monotonic() - window_started
            record = {
                "step": global_step,
                "task": task,
                "rank": env.rank,
                "local_rank": env.local_rank,
                "world_size": env.world_size,
                "local_samples": len(window),
                "local_tokens": local_tokens,
                "local_weighted_numerator": float(local_numerator_total.cpu()),
                "local_weighted_denominator": float(local_denominator.cpu()),
                "global_weighted_numerator": float(global_numerator.cpu()),
                "global_weighted_denominator": float(global_denominator.cpu()),
                "loss": float(loss.cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "forward_seconds": forward_seconds,
                "backward_including_gradient_all_reduce_seconds": backward_seconds,
                "denominator_all_reduce_seconds": denominator_all_reduce_seconds,
                "numerator_all_reduce_seconds": numerator_all_reduce_seconds,
                "optimizer_seconds": optimizer_seconds,
                "window_seconds": window_seconds,
                "local_tokens_per_second": local_tokens / max(window_seconds, 1e-9),
                "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
                "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
                "parameter_sync_max_abs": parameter_sync_max_abs,
                "hostname": os.uname().nodename,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown")),
            }
            if mode == "smoke":
                if not all(bool(row["prompt_weights_zero"]) for row in window_rows):
                    raise RuntimeError("SFT-v3 smoke found supervised prompt/padding")
                if not all(bool(row["eos_supervised"]) for row in window_rows):
                    raise RuntimeError("SFT-v3 smoke found an unsupervised EOS")
                if task == SFTV3Kind.CMB.value:
                    if any(
                        int(row["first_supervised_token_id"]) not in {32, 33, 34, 35, 36}
                        or int(row["supervised_token_count"]) != 2
                        for row in window_rows
                    ):
                        raise RuntimeError("SFT-v3 CMB smoke target is not letter+EOS")
                elif any(int(row["reasoning_token_count"]) != 0 for row in window_rows):
                    raise RuntimeError("SFT-v3 Medical-O1 smoke unexpectedly supervises reasoning")
                record["first_supervised_token_ids"] = sorted(
                    {int(row["first_supervised_token_id"]) for row in window_rows}
                )
                record["prompt_weights_zero"] = all(
                    bool(row["prompt_weights_zero"]) for row in window_rows
                )
                record["eos_supervised"] = all(bool(row["eos_supervised"]) for row in window_rows)
                record["supervised_token_counts"] = [
                    int(row["supervised_token_count"]) for row in window_rows
                ]
            gathered_records: list[dict[str, Any] | None] | None = (
                [None for _ in range(env.world_size)] if env.rank == 0 else None
            )
            dist.gather_object(record, gathered_records, dst=0)
            if env.rank == 0:
                aggregate_record = {
                    "step": global_step,
                    "task": task,
                    "world_size": 2,
                    "loss": record["loss"],
                    "grad_norm": record["grad_norm"],
                    "global_weighted_denominator": record["global_weighted_denominator"],
                    "learning_rate": record["learning_rate"],
                    "ranks": gathered_records,
                }
                _append_jsonl(run_dir / "metrics.jsonl", aggregate_record)
                log_history.append(aggregate_record)
            if global_step in save_steps:
                verification = _save_checkpoint(
                    dist=dist,
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run_dir=run_dir,
                    step=global_step,
                    rank=env.rank,
                    trainer_state={
                        "mode": mode,
                        "world_size": 2,
                        "task_step_counts": (
                            sft_v3_task_counts_through_step(global_step) if is_v3 else None
                        ),
                    },
                )
                if env.rank == 0 and verification is not None:
                    rank_zero_write_json(
                        run_dir / f"checkpoint-{global_step}" / "verification.json",
                        verification,
                        rank=0,
                    )
            if mode == "calibration" and global_step == 1:
                reserved = torch.tensor(
                    [torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory],
                    device=device,
                    dtype=torch.float64,
                )
                gathered = [torch.zeros_like(reserved) for _ in range(2)]
                dist.all_gather(gathered, reserved)
                margins = [float((item[1] - item[0]).cpu()) / (1024 * 1024) for item in gathered]
                calibration_reserved_windows.append([int(item[0].cpu()) for item in gathered])
                statuses = [classify_memory_margin(value) for value in margins]
                gate = max(statuses, key=("pass", "conditional", "fail").index)
                if gate == "fail":
                    raise RuntimeError(f"DDP calibration memory gate FAIL: margins_mib={margins}")
                if gate == "pass":
                    calibration_gate = {
                        "status": "pass",
                        "primary_margin_mib": margins,
                        "conditional_windows_run": 0,
                    }
                    break
                # Conditional mode consumes the three preregistered normal windows already queued.
            elif mode == "calibration":
                reserved = torch.tensor(
                    [torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory],
                    device=device,
                    dtype=torch.float64,
                )
                gathered = [torch.zeros_like(reserved) for _ in range(2)]
                dist.all_gather(gathered, reserved)
                calibration_reserved_windows.append([int(item[0].cpu()) for item in gathered])
                if any(
                    classify_memory_margin(float((item[1] - item[0]).cpu()) / (1024 * 1024)) == "fail"
                    for item in gathered
                ):
                    raise RuntimeError("conditional DDP calibration fell below 256 MiB margin")
                if global_step >= 4:
                    sustained_growth = any(
                        all(
                            calibration_reserved_windows[index][rank]
                            < calibration_reserved_windows[index + 1][rank]
                            for index in range(1, 3)
                        )
                        for rank in range(2)
                    )
                    if sustained_growth:
                        raise RuntimeError("conditional DDP calibration reserved memory kept growing")
                    calibration_gate = {
                        "status": "conditional_pass",
                        "primary_margin_mib": [
                            (
                                torch.cuda.get_device_properties(rank).total_memory
                                - calibration_reserved_windows[0][rank]
                            )
                            / (1024 * 1024)
                            for rank in range(2)
                        ],
                        "conditional_windows_run": 3,
                        "reserved_bytes_by_window": calibration_reserved_windows,
                        "sustained_growth": False,
                    }
                    break

        dist.barrier()
        update_norm = None
        if mode in {"calibration", "smoke"}:
            update_sq = torch.zeros((), device=device, dtype=torch.float32)
            for initial, parameter in zip(initial_trainable, trainable, strict=True):
                update_sq += (parameter.detach().float() - initial.float()).square().sum()
            update_norm = float(update_sq.sqrt().cpu())
            if not math.isfinite(update_norm) or update_norm <= 0:
                raise RuntimeError("calibration did not update LoRA parameters")
        if env.rank == 0:
            final_adapter = run_dir / "adapter"
            ddp_model.module.save_pretrained(final_adapter, safe_serialization=True)
            final_verification = _verify_adapter_checkpoint(
                run_dir / f"checkpoint-{global_step}", expected_step=global_step
            )
            rank_zero_write_json(
                run_dir / "sample_coverage.json", coverage, rank=0
            )
            manifest_sha = _sha256(manifest_path)
            adapter_manifest = finalize_lora_run(
                run_dir,
                adapter_dir=final_adapter,
                run_id=run_id,
                model_id=str(config["model"]["path"]),
                model_revision=str(config["model"]["revision"]),
                tokenizer_revision=str(config["model"]["tokenizer_revision"]),
                data_manifest_sha256=manifest_sha,
                metrics={
                    "global_step": global_step,
                    "train_loss": log_history[-1]["loss"],
                    "train_runtime": time.monotonic() - started,
                },
                log_history=log_history,
            )
            summary = {
                "run_id": run_id,
                "status": (
                    "gpu_smoke_complete_pending_reload"
                    if mode == "smoke"
                    else "calibration_complete_pending_reload"
                    if mode == "calibration"
                    else "training_complete_pending_controller"
                ),
                "mode": mode,
                "world_size": 2,
                "global_effective_batch": 16,
                "optimizer_steps": global_step,
                "supervision": supervision_summary,
                "sample_coverage": coverage,
                "checkpoint_verification": final_verification,
                "adapter_sha256": adapter_manifest["adapter_sha256"],
                "lora_update_norm": update_norm,
                "runtime_seconds": time.monotonic() - started,
                "actual_cost_cny": None,
            }
            if calibration_gate is not None:
                summary["calibration_gate"] = calibration_gate
                rank_zero_write_json(
                    run_dir / "calibration_gate.json",
                    {
                        **calibration_gate,
                        "run_id": run_id,
                        "world_size": 2,
                        "optimizer_steps": global_step,
                        "checkpoint_step": global_step,
                        "parameter_sync_max_abs": 0.0,
                        "loss_finite": True,
                        "grad_finite": True,
                        "checkpoint_verified": True,
                        "final_authorized": False,
                    },
                    rank=0,
                )
            if mode == "smoke":
                tasks = [str(item["task"]) for item in log_history]
                if tasks != [
                    SFTV3Kind.CMB.value,
                    SFTV3Kind.CMB.value,
                    SFTV3Kind.CMB.value,
                    SFTV3Kind.MEDICAL_O1.value,
                ]:
                    raise RuntimeError("SFT-v3 GPU smoke did not complete C,C,C,O")
                smoke_contract = {
                    "status": "PASS",
                    "run_id": run_id,
                    "optimizer_steps": 4,
                    "task_sequence": tasks,
                    "candidate_token_ids": [32, 33, 34, 35, 36],
                    "rank_records": [item["ranks"] for item in log_history],
                    "parameter_sync_max_abs": 0.0,
                    "loss_grad_finite": True,
                    "checkpoint_step": 4,
                    "checkpoint_verified": True,
                    "final_authorized": False,
                }
                rank_zero_write_json(run_dir / "gpu_smoke_contract.json", smoke_contract, rank=0)
            rank_zero_write_json(run_dir / "summary.json", summary, rank=0)
        dist.barrier()

        # Calibration explicitly proves that the saved PEFT checkpoint can be loaded only after
        # the training objects are released. Each rank reloads on its own device; no extra rank-0
        # model copy exists.
        if mode in {"calibration", "smoke"}:
            checkpoint_path = run_dir / f"checkpoint-{global_step}"
            del optimizer, scheduler, ddp_model, model, trainable, initial_trainable
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            base_reload = AutoModelForCausalLM.from_pretrained(
                str(config["model"]["path"]),
                revision=str(config["model"]["revision"]),
                local_files_only=True,
                torch_dtype=getattr(torch, str(config["model"]["torch_dtype"])),
                attn_implementation=str(config["model"]["attn_implementation"]),
            )
            base_reload.config.use_cache = False
            base_reload.to(device)
            reloaded = PeftModel.from_pretrained(
                base_reload, str(checkpoint_path), is_trainable=False
            )
            reloaded.eval()
            if "default" not in reloaded.peft_config:
                raise RuntimeError("reloaded calibration checkpoint lacks the default adapter")
            del reloaded, base_reload
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            if env.rank == 0:
                if mode == "calibration":
                    gate_path = run_dir / "calibration_gate.json"
                    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
                    gate_payload["checkpoint_reload_verified"] = True
                    rank_zero_write_json(gate_path, gate_payload, rank=0)
                else:
                    gate_path = run_dir / "gpu_smoke_contract.json"
                    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
                    gate_payload["checkpoint_reload_verified"] = True
                    rank_zero_write_json(gate_path, gate_payload, rank=0)
                summary_path = run_dir / "summary.json"
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_payload["status"] = (
                    "gpu_smoke_complete" if mode == "smoke" else "calibration_complete"
                )
                summary_payload["checkpoint_reload_verified"] = True
                rank_zero_write_json(summary_path, summary_payload, rank=0)
            dist.barrier()
        result = {"run_id": run_id, "rank": env.rank, "optimizer_steps": global_step, "status": "complete"}
        return result
    except BaseException as error:
        if run_dir.exists():
            try:
                with (run_dir / f"rank-{env.rank}.error.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"{type(error).__name__}: {error}\n")
            except OSError:
                pass
        if env.rank == 0 and run_dir.exists():
            record_sft_failure(run_dir, run_id=run_id, reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="P3.5 exact weighted two-rank DDP SFT-v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["calibration", "smoke", "train"])
    parser.add_argument("--calibration-manifest", default=None)
    arguments = parser.parse_args(argv)
    result = run(
        arguments.config,
        mode=arguments.mode,
        calibration_manifest=arguments.calibration_manifest,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
