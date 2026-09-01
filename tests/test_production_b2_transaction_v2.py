from __future__ import annotations

import copy

import pytest
import torch

from src.opd.production_b2_transaction_v2 import (
    OptimizerTransactionV2,
    TransactionStateV2,
    TransactionV2Error,
    ordered_trainable_sha256,
    state_tree_sha256,
)


def _objects(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    # Materialize Adam state before the transaction snapshot.
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(3, 4)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    state = TransactionStateV2(
        accepted_optimizer_steps=20,
        data_cursor=80,
        policy_version=20,
        sampler_version=20,
        refresh_version=20,
        registry_count=1,
    )
    return model, optimizer, scheduler, state


def _candidate(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.arange(12, dtype=torch.float32).reshape(3, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_candidate_rejection_restores_all_state_and_does_not_count_step(tmp_path):
    model, optimizer, scheduler, state = _objects(tmp_path)
    transaction = OptimizerTransactionV2.capture(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        scratch_root=tmp_path,
        fixed_batch_sha256="a" * 64,
    )
    before = {
        "lora": ordered_trainable_sha256(model),
        "optimizer": state_tree_sha256(optimizer.state_dict()),
        "scheduler": state_tree_sha256(scheduler.state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "state": copy.deepcopy(state),
    }
    _candidate(model, optimizer, scheduler)
    assert ordered_trainable_sha256(model) != before["lora"]

    audit = transaction.reject(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        reason="candidate_post_shift",
        observed_registry_count=1,
    )
    assert audit["rollback_verified"] is True
    assert ordered_trainable_sha256(model) == before["lora"]
    assert state_tree_sha256(optimizer.state_dict()) == before["optimizer"]
    assert state_tree_sha256(scheduler.state_dict()) == before["scheduler"]
    assert torch.equal(torch.get_rng_state(), before["cpu_rng"])
    assert state == before["state"]
    assert state.accepted_optimizer_steps == 20
    assert state.data_cursor == 80
    assert state.policy_version == state.sampler_version == state.refresh_version == 20
    assert all(parameter.grad is None for parameter in model.parameters())


def test_commit_is_only_transition_that_advances_cursor_and_versions(tmp_path):
    model, optimizer, scheduler, state = _objects(tmp_path)
    transaction = OptimizerTransactionV2.capture(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        scratch_root=tmp_path,
        fixed_batch_sha256="b" * 64,
    )
    _candidate(model, optimizer, scheduler)
    transaction.mark_candidate_validated()
    audit = transaction.commit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        prompts_per_step=4,
        observed_registry_count=1,
    )
    assert audit["committed"] is True
    assert state.accepted_optimizer_steps == 21
    assert state.data_cursor == 84
    assert state.policy_version == state.sampler_version == state.refresh_version == 21


def test_pre_gate_or_grad_gate_failure_never_executes_candidate(tmp_path):
    model, optimizer, scheduler, state = _objects(tmp_path)
    transaction = OptimizerTransactionV2.capture(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        scratch_root=tmp_path,
        fixed_batch_sha256="c" * 64,
    )
    transaction.reject_before_candidate("pre_update_identity")
    with pytest.raises(TransactionV2Error, match="rejected before candidate"):
        transaction.mark_candidate_validated()
    assert state.accepted_optimizer_steps == 20


def test_same_batch_retry_is_deterministic_after_rejection(tmp_path):
    model, optimizer, scheduler, state = _objects(tmp_path)
    x = torch.randn(5, 4)
    expected = model(x).detach().clone()
    transaction = OptimizerTransactionV2.capture(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        scratch_root=tmp_path,
        fixed_batch_sha256="d" * 64,
    )
    _candidate(model, optimizer, scheduler)
    transaction.reject(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        reason="fixture",
        observed_registry_count=1,
    )
    assert torch.equal(model(x).detach(), expected)
    assert transaction.fixed_batch_sha256 == "d" * 64


def test_registry_growth_makes_rollback_unverifiable(tmp_path):
    model, optimizer, scheduler, state = _objects(tmp_path)
    transaction = OptimizerTransactionV2.capture(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        scratch_root=tmp_path,
        fixed_batch_sha256="e" * 64,
    )
    _candidate(model, optimizer, scheduler)
    with pytest.raises(TransactionV2Error, match="registry"):
        transaction.reject(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            reason="fixture",
            observed_registry_count=2,
        )
