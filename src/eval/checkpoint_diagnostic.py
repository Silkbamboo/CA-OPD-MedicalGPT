"""Low-memory PEFT checkpoint identity and evolution diagnostics.

The Qwen backbone and optimizer state are never deserialized.  File hashes are
streamed, and safetensors are materialized one tensor pair at a time on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_TARGET = re.compile(r"\.([a-z0-9_]+)\.lora_([AB])(?:\.|$)")


class CheckpointDiagnosticError(RuntimeError):
    """Invalid or unsafe intermediate LoRA checkpoint."""


def stream_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_adapter_sha256(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = root / name
        if not path.is_file():
            raise CheckpointDiagnosticError(f"checkpoint lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    inventory: list[dict[str, Any]] = []
    for path in sorted((item for item in root.iterdir() if item.is_file()), key=lambda item: item.name):
        inventory.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": stream_sha256(path),
        })
    raw = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return inventory, hashlib.sha256(raw).hexdigest()


def _inspect_tensors(path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - dependency gate
        raise CheckpointDiagnosticError("torch and safetensors are required for checkpoint audit") from error

    count = finite = 0
    lora_a = lora_b = lora_b_nonzero = 0
    shapes: CounterLike = defaultdict(int)
    total_elements = nonzero_elements = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for name in keys:
            tensor = handle.get_tensor(name)
            count += 1
            elements = int(tensor.numel())
            total_elements += elements
            is_finite = bool(torch.isfinite(tensor).all().item())
            finite += int(is_finite)
            nonzero = int(torch.count_nonzero(tensor).item())
            nonzero_elements += nonzero
            shapes[str(tuple(int(value) for value in tensor.shape))] += 1
            if ".lora_A" in name:
                lora_a += 1
            if ".lora_B" in name:
                lora_b += 1
                lora_b_nonzero += int(nonzero > 0)
            del tensor
    if not count or finite != count:
        raise CheckpointDiagnosticError("checkpoint contains missing/non-finite tensors")
    if lora_a < 1 or lora_b < 1 or lora_a != lora_b:
        raise CheckpointDiagnosticError("checkpoint LoRA-A/LoRA-B tensor inventory is invalid")
    if lora_b_nonzero != lora_b:
        raise CheckpointDiagnosticError("checkpoint contains zero LoRA-B tensors")
    return {
        "tensor_count": count,
        "all_tensors_finite": True,
        "lora_a_tensor_count": lora_a,
        "lora_b_tensor_count": lora_b,
        "lora_b_nonzero": True,
        "total_elements": total_elements,
        "nonzero_ratio": nonzero_elements / total_elements,
        "shape_distribution": dict(sorted(shapes.items())),
    }


# Only scalar integer increments are stored; the alias keeps type annotations
# dependency-free and does not retain tensors.
CounterLike = dict[str, int]


def audit_checkpoint_adapter(
    checkpoint: str | Path,
    *,
    expected_step: int,
    expected_base_model: str,
    base_revision: str,
    tokenizer_revision: str,
    data_manifest_sha256: str,
    source_run_id: str,
    source_git_sha: str,
) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    required = (
        root / "adapter_config.json",
        root / "adapter_model.safetensors",
        root / "trainer_state.json",
    )
    if not root.is_dir() or any(not path.is_file() for path in required):
        raise CheckpointDiagnosticError("checkpoint lacks a standard PEFT adapter/trainer state")
    if _HEX40.fullmatch(base_revision) is None or _HEX40.fullmatch(tokenizer_revision) is None:
        raise CheckpointDiagnosticError("checkpoint model/tokenizer revision is not immutable")
    if _HEX64.fullmatch(data_manifest_sha256) is None or _HEX40.fullmatch(source_git_sha) is None:
        raise CheckpointDiagnosticError("checkpoint source identity is incomplete")
    config = json.loads(required[0].read_text(encoding="utf-8"))
    state = json.loads(required[2].read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != expected_step or root.name != f"checkpoint-{expected_step}":
        raise CheckpointDiagnosticError("checkpoint step identity mismatch")
    if (
        config.get("base_model_name_or_path") != expected_base_model
        or config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
        or config.get("r") != 16
        or config.get("lora_alpha") != 32
        or float(config.get("lora_dropout", -1)) != 0.05
        or config.get("inference_mode") is not True
        or not isinstance(config.get("target_modules"), list)
        or not config["target_modules"]
    ):
        raise CheckpointDiagnosticError("checkpoint PEFT configuration drift")
    tensor = _inspect_tensors(required[1])
    inventory, source_sha = _source_inventory(root)
    return {
        "schema_version": 1,
        "artifact_type": "peft_lora_checkpoint",
        "checkpoint_role": "candidate_medical_teacher",
        "checkpoint_step": expected_step,
        "checkpoint_path": str(root),
        "source_run_id": source_run_id,
        "source_git_sha": source_git_sha,
        "source_checkpoint_sha256": source_sha,
        "source_file_count": len(inventory),
        "source_files": inventory,
        "base_model_path": expected_base_model,
        "base_model_revision": base_revision,
        "tokenizer_revision": tokenizer_revision,
        "data_manifest_sha256": data_manifest_sha256,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": sorted(str(value) for value in config["target_modules"]),
        "adapter_weight_sha256": stream_sha256(required[1]),
        "adapter_config_sha256": stream_sha256(required[0]),
        "adapter_sha256": ordered_adapter_sha256(root),
        **tensor,
        "peft_structurally_loadable": True,
        "base_model_loaded": False,
        "optimizer_state_loaded": False,
        "gpu_runtime_verified": False,
        "final_authorized": False,
    }


class _NormAggregate:
    def __init__(self) -> None:
        self.elements = 0
        self.first_sq = 0.0
        self.second_sq = 0.0
        self.delta_sq = 0.0
        self.max_abs_first = 0.0
        self.max_abs_second = 0.0
        self.max_abs_delta = 0.0
        self.first_nonzero = 0
        self.second_nonzero = 0

    def add(self, first: Any, second: Any, delta: Any, torch: Any) -> None:
        self.elements += int(first.numel())
        self.first_sq += float(torch.sum(first * first).item())
        self.second_sq += float(torch.sum(second * second).item())
        self.delta_sq += float(torch.sum(delta * delta).item())
        self.max_abs_first = max(self.max_abs_first, float(torch.max(torch.abs(first)).item()))
        self.max_abs_second = max(self.max_abs_second, float(torch.max(torch.abs(second)).item()))
        self.max_abs_delta = max(self.max_abs_delta, float(torch.max(torch.abs(delta)).item()))
        self.first_nonzero += int(torch.count_nonzero(first).item())
        self.second_nonzero += int(torch.count_nonzero(second).item())

    def report(self) -> dict[str, Any]:
        first = math.sqrt(max(0.0, self.first_sq))
        second = math.sqrt(max(0.0, self.second_sq))
        delta = math.sqrt(max(0.0, self.delta_sq))
        return {
            "elements": self.elements,
            "step250_l2_norm": first,
            "step500_l2_norm": second,
            "delta_l2_norm": delta,
            "relative_delta_to_step250": delta / first if first else None,
            "step250_max_abs": self.max_abs_first,
            "step500_max_abs": self.max_abs_second,
            "delta_max_abs": self.max_abs_delta,
            "step250_nonzero_ratio": self.first_nonzero / self.elements if self.elements else 0.0,
            "step500_nonzero_ratio": self.second_nonzero / self.elements if self.elements else 0.0,
        }


def compare_lora_checkpoints(step250: str | Path, step500: str | Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover
        raise CheckpointDiagnosticError("torch and safetensors are required for checkpoint comparison") from error

    first_path = Path(step250) / "adapter_model.safetensors"
    second_path = Path(step500) / "adapter_model.safetensors"
    if not first_path.is_file() or not second_path.is_file():
        raise CheckpointDiagnosticError("checkpoint comparison requires both adapter safetensors")
    total = _NormAggregate()
    groups: dict[str, dict[str, _NormAggregate]] = {
        "layer": defaultdict(_NormAggregate),
        "target": defaultdict(_NormAggregate),
        "matrix": defaultdict(_NormAggregate),
    }
    with safe_open(first_path, framework="pt", device="cpu") as left, safe_open(
        second_path, framework="pt", device="cpu"
    ) as right:
        left_keys, right_keys = list(left.keys()), list(right.keys())
        if left_keys != right_keys or not left_keys:
            raise CheckpointDiagnosticError("step250/step500 tensor inventories differ")
        for name in left_keys:
            first = left.get_tensor(name).float()
            second = right.get_tensor(name).float()
            if first.shape != second.shape or not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
                raise CheckpointDiagnosticError("checkpoint tensor shape/finite state differs")
            delta = second - first
            layer = (_LAYER.search(name).group(1) if _LAYER.search(name) else "unknown")
            target_match = _TARGET.search(name)
            target = target_match.group(1) if target_match else "unknown"
            matrix = target_match.group(2) if target_match else "unknown"
            total.add(first, second, delta, torch)
            groups["layer"][layer].add(first, second, delta, torch)
            groups["target"][target].add(first, second, delta, torch)
            groups["matrix"][matrix].add(first, second, delta, torch)
            del first, second, delta
    layer_reports = {key: value.report() for key, value in sorted(groups["layer"].items())}
    total_report = total.report()
    layer_delta_sq = sorted(
        (value["delta_l2_norm"] ** 2 for value in layer_reports.values()), reverse=True
    )
    total_delta_sq = total_report["delta_l2_norm"] ** 2
    return {
        "checkpoint_steps": [250, 500],
        "tensor_count": len(left_keys),
        "all_tensors_finite": True,
        **total_report,
        "by_layer": layer_reports,
        "by_target_module": {
            key: value.report() for key, value in sorted(groups["target"].items())
        },
        "by_lora_matrix": {
            key: value.report() for key, value in sorted(groups["matrix"].items())
        },
        "top5_layer_delta_energy_share": (
            sum(layer_delta_sq[:5]) / total_delta_sq if total_delta_sq else 0.0
        ),
        "comparison_mode": "one_tensor_pair_at_a_time_cpu",
        "optimizer_state_loaded": False,
        "base_model_loaded": False,
        "knowledge_interpretation_authorized": False,
    }


__all__ = [
    "CheckpointDiagnosticError",
    "audit_checkpoint_adapter",
    "compare_lora_checkpoints",
    "ordered_adapter_sha256",
    "stream_sha256",
]
