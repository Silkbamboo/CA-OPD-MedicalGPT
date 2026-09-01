"""Transactional optimizer snapshot/rollback primitives for Formal B2 v2."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch


class TransactionV2Error(RuntimeError):
    """A transaction transition or rollback invariant failed."""


@dataclass
class TransactionStateV2:
    accepted_optimizer_steps: int
    data_cursor: int
    policy_version: int
    sampler_version: int
    refresh_version: int
    registry_count: int


def _update_tree_hash(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_tree_hash(digest, key)
            _update_tree_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_tree_hash(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    else:
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def state_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_tree_hash(digest, value)
    return digest.hexdigest()


def ordered_trainable_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count += 1
        digest.update(name.encode("utf-8"))
        _update_tree_hash(digest, parameter.detach())
    if count == 0:
        raise TransactionV2Error("transaction model has no trainable tensors")
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return copy.deepcopy(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OptimizerTransactionV2:
    """One batch transaction. Export/refresh remain outside until acceptance."""

    def __init__(
        self,
        *,
        snapshot_path: Path,
        snapshot_file_sha256: str,
        fixed_batch_sha256: str,
        initial_state: TransactionStateV2,
        initial_registry_count: int,
        initial_model_sha256: str,
        initial_optimizer_sha256: str,
        initial_scheduler_sha256: str,
    ) -> None:
        self.snapshot_path = snapshot_path
        self.snapshot_file_sha256 = snapshot_file_sha256
        self.fixed_batch_sha256 = fixed_batch_sha256
        self.initial_state = initial_state
        self.initial_registry_count = initial_registry_count
        self.initial_model_sha256 = initial_model_sha256
        self.initial_optimizer_sha256 = initial_optimizer_sha256
        self.initial_scheduler_sha256 = initial_scheduler_sha256
        self.phase = "prepared"

    @classmethod
    def capture(
        cls,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TransactionStateV2,
        scratch_root: str | Path,
        fixed_batch_sha256: str,
    ) -> "OptimizerTransactionV2":
        if not (
            isinstance(fixed_batch_sha256, str)
            and len(fixed_batch_sha256) == 64
            and all(character in "0123456789abcdef" for character in fixed_batch_sha256)
            and state.policy_version == state.sampler_version == state.refresh_version
            and state.data_cursor == state.accepted_optimizer_steps * 4
        ):
            raise TransactionV2Error("transaction input identity differs")
        root = Path(scratch_root)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise TransactionV2Error("transaction scratch root is a symlink")
        trainable = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        payload = {
            "trainable": trainable,
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": _cpu_tree(scheduler.state_dict()),
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
            "state": asdict(state),
        }
        descriptor, name = tempfile.mkstemp(prefix="p5_1_transaction_", suffix=".pt", dir=root)
        os.close(descriptor)
        path = Path(name)
        try:
            torch.save(payload, path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return cls(
            snapshot_path=path,
            snapshot_file_sha256=_sha256_file(path),
            fixed_batch_sha256=fixed_batch_sha256,
            initial_state=copy.deepcopy(state),
            initial_registry_count=state.registry_count,
            initial_model_sha256=ordered_trainable_sha256(model),
            initial_optimizer_sha256=state_tree_sha256(optimizer.state_dict()),
            initial_scheduler_sha256=state_tree_sha256(scheduler.state_dict()),
        )

    def _load(self) -> Mapping[str, Any]:
        if not self.snapshot_path.is_file() or _sha256_file(self.snapshot_path) != self.snapshot_file_sha256:
            raise TransactionV2Error("transaction snapshot SHA differs")
        value = torch.load(self.snapshot_path, map_location="cpu", weights_only=True)
        if not isinstance(value, Mapping):
            raise TransactionV2Error("transaction snapshot is invalid")
        return value

    def mark_candidate_validated(self) -> None:
        if self.phase.startswith("rejected"):
            raise TransactionV2Error("transaction was rejected before candidate")
        if self.phase != "prepared":
            raise TransactionV2Error("candidate validation transition differs")
        self.phase = "candidate_validated"

    def reject_before_candidate(self, reason: str) -> dict[str, Any]:
        if self.phase != "prepared" or not reason:
            raise TransactionV2Error("pre-candidate rejection transition differs")
        self.phase = "rejected_before_candidate"
        self.snapshot_path.unlink(missing_ok=True)
        return {"rejected": True, "candidate_executed": False, "reason": reason}

    def reject(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TransactionStateV2,
        reason: str,
        observed_registry_count: int,
    ) -> dict[str, Any]:
        if self.phase not in {"prepared", "candidate_validated"} or not reason:
            raise TransactionV2Error("candidate rejection transition differs")
        payload = self._load()
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved in payload["trainable"].items():
                if name not in parameters or not parameters[name].requires_grad:
                    raise TransactionV2Error("trainable tensor set differs during rollback")
                parameters[name].copy_(saved.to(parameters[name].device, dtype=parameters[name].dtype))
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restored = TransactionStateV2(**dict(payload["state"]))
        for field, value in asdict(restored).items():
            setattr(state, field, value)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(payload["cpu_rng"])
        if payload["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        model_sha = ordered_trainable_sha256(model)
        optimizer_sha = state_tree_sha256(optimizer.state_dict())
        scheduler_sha = state_tree_sha256(scheduler.state_dict())
        failures = []
        if model_sha != self.initial_model_sha256:
            failures.append("LoRA ordered SHA")
        if optimizer_sha != self.initial_optimizer_sha256:
            failures.append("optimizer state")
        if scheduler_sha != self.initial_scheduler_sha256:
            failures.append("scheduler state")
        if state != self.initial_state:
            failures.append("cursor/policy/sampler state")
        if observed_registry_count != self.initial_registry_count:
            failures.append("sampler registry")
        self.phase = "rejected"
        self.snapshot_path.unlink(missing_ok=True)
        if failures:
            raise TransactionV2Error("transaction rollback failed: " + ",".join(failures))
        return {
            "rejected": True,
            "candidate_executed": True,
            "reason": reason,
            "rollback_verified": True,
            "lora_ordered_sha256": model_sha,
            "optimizer_state_sha256": optimizer_sha,
            "scheduler_state_sha256": scheduler_sha,
            "cpu_rng_restored": True,
            "cuda_rng_restored": True,
            "fixed_batch_sha256": self.fixed_batch_sha256,
        }

    def commit(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TransactionStateV2,
        prompts_per_step: int,
        observed_registry_count: int,
    ) -> dict[str, Any]:
        del optimizer, scheduler
        if self.phase != "candidate_validated":
            raise TransactionV2Error("only a validated candidate may commit")
        if state != self.initial_state:
            raise TransactionV2Error("cursor/version advanced before commit")
        if observed_registry_count != self.initial_registry_count:
            raise TransactionV2Error("sampler registry changed before commit")
        if ordered_trainable_sha256(model) == self.initial_model_sha256:
            raise TransactionV2Error("candidate made no trainable update")
        state.accepted_optimizer_steps += 1
        state.data_cursor += int(prompts_per_step)
        state.policy_version += 1
        state.sampler_version += 1
        state.refresh_version += 1
        self.phase = "committed"
        self.snapshot_path.unlink(missing_ok=True)
        return {
            "committed": True,
            "accepted_optimizer_steps": state.accepted_optimizer_steps,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "fixed_batch_sha256": self.fixed_batch_sha256,
        }
