from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.sft.ddp import (
    DDPExecutionContract,
    assert_global_sample_coverage,
    accumulation_windows,
    ddp_scaled_loss,
    distributed_sample_indices,
    freeze_calibration_selection,
    local_weight_denominator,
    rank_zero_write_json,
    validate_training_source_contract,
)


def test_ddp_execution_contract_is_the_frozen_two_rank_topology() -> None:
    contract = DDPExecutionContract.frozen()
    assert contract.launch_mode == "ddp"
    assert contract.backend == "nccl"
    assert contract.world_size == 2
    assert contract.per_device_micro_batch_size == 1
    assert contract.gradient_accumulation_steps == 8
    assert contract.global_effective_batch == 16
    assert contract.ddp_kwargs == {
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "bucket_cap_mb": 16,
    }


def test_global_weighted_scaling_is_not_rank_mean() -> None:
    torch = pytest.importorskip("torch")
    numerator = torch.tensor(9.0, requires_grad=True)
    global_denominator = torch.tensor(12.0)
    loss = ddp_scaled_loss(numerator, global_denominator, world_size=2)
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert numerator.grad.item() == pytest.approx(2.0 / 12.0)


def test_weight_denominator_uses_shifted_supervision_and_ignores_prompt_padding() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, -100, 7, 8, -100]])
    weights = torch.tensor([[0.0, 0.0, 1.5, 0.5, 0.0]])
    assert local_weight_denominator(labels, weights).item() == pytest.approx(2.0)


def test_accumulation_windows_keep_the_real_partial_window() -> None:
    windows = list(accumulation_windows(list(range(19)), accumulation_steps=8))
    assert [len(window) for window in windows] == [8, 8, 3]
    assert windows[-1] == [16, 17, 18]


def test_distributed_sampler_indices_are_disjoint_and_cover_9500() -> None:
    rank0 = distributed_sample_indices(9500, rank=0, world_size=2, seed=42, epoch=0)
    rank1 = distributed_sample_indices(9500, rank=1, world_size=2, seed=42, epoch=0)
    summary = assert_global_sample_coverage(
        [rank0, rank1], expected_size=9500, expected_per_rank=4750
    )
    assert summary["global_unique_samples"] == 9500
    assert summary["duplicate_samples"] == 0
    assert summary["missing_samples"] == 0
    assert set(rank0).isdisjoint(rank1)
    assert rank0 == distributed_sample_indices(9500, rank=0, world_size=2, seed=42, epoch=0)


def test_rank_zero_is_the_only_shared_writer(tmp_path: Path) -> None:
    output = tmp_path / "rank-zero.json"
    assert rank_zero_write_json(output, {"rank": 1}, rank=1) is False
    assert not output.exists()
    assert rank_zero_write_json(output, {"rank": 0}, rank=0) is True
    assert json.loads(output.read_text()) == {"rank": 0}


def test_calibration_selection_freezes_longest_and_three_conditional_windows(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    rows = [
        {
            "sample_id": f"s-{index:03d}",
            "target_role": "medical_sft_train",
            "token_count_prompt": 100 + index,
            "token_count_response": 200 + index,
        }
        for index in range(100)
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = freeze_calibration_selection(records, seed=42)
    second = freeze_calibration_selection(records, seed=42)
    assert first == second
    primary_ids = first["primary_window"]["rank0"] + first["primary_window"]["rank1"]
    assert "s-099" in primary_ids
    assert len(primary_ids) == 16 == len(set(primary_ids))
    conditional_ids = [
        sample_id
        for window in first["conditional_windows"]
        for rank in ("rank0", "rank1")
        for sample_id in window[rank]
    ]
    assert len(conditional_ids) == 48 == len(set(conditional_ids))
    assert set(primary_ids).isdisjoint(conditional_ids)


def test_formal_ddp_source_rejects_dataparallel_device_map_and_parent_model_load() -> None:
    good = Path("src/sft/train_ddp.py").read_text(encoding="utf-8")
    validate_training_source_contract(good)
    assert "DistributedSampler" in good and ".set_epoch(0)" in good
    assert "destroy_process_group" in good
    assert "calibration_gate.json" in good
    assert "PeftModel.from_pretrained" in good
    assert "finalize_lora_run" in good
    for forbidden in (
        "nn.DataParallel(model)",
        "device_map=\"auto\"",
        "PARENT_LOADS_MODEL_BEFORE_TORCHRUN = True",
    ):
        with pytest.raises(ValueError):
            validate_training_source_contract(good + "\n" + forbidden)


def _gloo_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from src.sft.ddp import ddp_scaled_loss, local_weight_denominator
    from src.sft.weighted import weighted_causal_lm_components

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(123)

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(6, 4)
                self.lora = torch.nn.Linear(4, 6, bias=False)

            def forward(self, input_ids):
                return self.lora(self.embedding(input_ids))

        model = DistributedDataParallel(TinyLM())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        local = {
            0: [
                ([0, 1, 2, 3], [-100, 1, 2, 3], [0.0, 1.5, 0.5, 0.5]),
                ([1, 2, 3, 4], [-100, 2, 3, 4], [0.0, 1.5, 1.5, 1.5]),
            ],
            1: [
                ([2, 3, 4, 5], [-100, 3, 4, 5], [0.0, 0.5, 0.5, 1.5]),
                ([3, 4, 5, 0], [-100, 4, 5, 0], [0.0, 1.5, 0.5, 1.5]),
            ],
        }[rank]
        windows = list(accumulation_windows(local, accumulation_steps=8))
        assert len(windows) == 1 and len(windows[0]) == 2
        denominator = torch.zeros(())
        for _, labels_raw, weights_raw in windows[0]:
            labels = torch.tensor([labels_raw])
            weights = torch.tensor([weights_raw])
            denominator += local_weight_denominator(labels, weights)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        optimizer.zero_grad(set_to_none=True)
        for index, (ids_raw, labels_raw, weights_raw) in enumerate(windows[0]):
            ids = torch.tensor([ids_raw])
            labels = torch.tensor([labels_raw])
            weights = torch.tensor([weights_raw])
            context = model.no_sync() if index < len(local) - 1 else __import__("contextlib").nullcontext()
            with context:
                logits = model(ids)
                numerator, _, _ = weighted_causal_lm_components(logits, labels, weights)
                ddp_scaled_loss(numerator, denominator, world_size=world_size).backward()
        optimizer.step()
        flattened = torch.cat([parameter.detach().reshape(-1) for parameter in model.module.parameters()])
        gathered = [torch.zeros_like(flattened) for _ in range(world_size)]
        dist.all_gather(gathered, flattened)
        torch.save(gathered[rank], Path(result_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_process_gloo_matches_single_process_global_weighted_batch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.multiprocessing as mp

    init_file = tmp_path / "gloo-init"
    # PyTorch forbids autograd in a forked child after any earlier test starts
    # parent-process autograd threads.  ``spawn`` gives both gloo ranks a clean
    # interpreter and mirrors torchrun's no-parent-model-load boundary.
    mp.start_processes(
        _gloo_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    rank0 = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert torch.equal(rank0, rank1), "DDP must leave every rank with identical parameters"

    from src.sft.weighted import weighted_causal_lm_components

    torch.manual_seed(123)

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 4)
            self.lora = torch.nn.Linear(4, 6, bias=False)

        def forward(self, input_ids):
            return self.lora(self.embedding(input_ids))

    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    rows = [
        ([0, 1, 2, 3], [-100, 1, 2, 3], [0.0, 1.5, 0.5, 0.5]),
        ([1, 2, 3, 4], [-100, 2, 3, 4], [0.0, 1.5, 1.5, 1.5]),
        ([2, 3, 4, 5], [-100, 3, 4, 5], [0.0, 0.5, 0.5, 1.5]),
        ([3, 4, 5, 0], [-100, 4, 5, 0], [0.0, 1.5, 0.5, 1.5]),
    ]
    numerators = []
    denominators = []
    for ids_raw, labels_raw, weights_raw in rows:
        numerator, denominator, _ = weighted_causal_lm_components(
            model(torch.tensor([ids_raw])),
            torch.tensor([labels_raw]),
            torch.tensor([weights_raw]),
        )
        numerators.append(numerator)
        denominators.append(denominator)
    loss = torch.stack(numerators).sum() / torch.stack(denominators).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    reference = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    assert torch.allclose(rank0, reference, rtol=1e-6, atol=1e-7)


def test_checkpoint_mock_requires_finite_nonzero_lora_b_and_step_identity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from src.sft.train_ddp import _verify_adapter_checkpoint

    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 7}) + "\n", encoding="utf-8"
    )
    torch.save({"state": {}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 7}, checkpoint / "scheduler.pt")
    safetensors.save_file(
        {
            "base_model.layer.lora_A.weight": torch.ones((2, 3)),
            "base_model.layer.lora_B.weight": torch.full((3, 2), 0.25),
        },
        checkpoint / "adapter_model.safetensors",
    )
    result = _verify_adapter_checkpoint(checkpoint, expected_step=7)
    assert result["step"] == 7
    assert result["tensor_count"] == 2
    assert result["finite"] is True
    assert result["lora_b_nonzero"] is True


def test_runtime_source_records_rank_local_and_collective_metrics() -> None:
    source = Path("src/sft/train_ddp.py").read_text(encoding="utf-8")
    for field in (
        "local_weighted_numerator",
        "local_weighted_denominator",
        "global_weighted_denominator",
        "backward_including_gradient_all_reduce_seconds",
        "denominator_all_reduce_seconds",
        "local_tokens_per_second",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        assert field in source
