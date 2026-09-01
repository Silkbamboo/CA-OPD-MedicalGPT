"""Small, testable primitives for exact weighted two-rank DDP training.

The module contains no model loading and performs no CUDA work at import time.
That keeps CPU preflight and gloo regression tests honest and inexpensive.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DDPExecutionContract:
    launch_mode: str
    backend: str
    world_size: int
    per_device_micro_batch_size: int
    gradient_accumulation_steps: int
    broadcast_buffers: bool
    find_unused_parameters: bool
    gradient_as_bucket_view: bool
    bucket_cap_mb: int

    @classmethod
    def frozen(cls) -> "DDPExecutionContract":
        return cls(
            launch_mode="ddp",
            backend="nccl",
            world_size=2,
            per_device_micro_batch_size=1,
            gradient_accumulation_steps=8,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=16,
        )

    @property
    def global_effective_batch(self) -> int:
        return (
            self.per_device_micro_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )

    @property
    def ddp_kwargs(self) -> dict[str, Any]:
        return {
            "broadcast_buffers": self.broadcast_buffers,
            "find_unused_parameters": self.find_unused_parameters,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "bucket_cap_mb": self.bucket_cap_mb,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "global_effective_batch": self.global_effective_batch}


@dataclass(frozen=True)
class DistributedEnvironment:
    local_rank: int
    rank: int
    world_size: int

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DistributedEnvironment":
        values = os.environ if environ is None else environ
        try:
            result = cls(
                local_rank=int(values["LOCAL_RANK"]),
                rank=int(values["RANK"]),
                world_size=int(values["WORLD_SIZE"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "DDP entrypoint requires torchrun LOCAL_RANK/RANK/WORLD_SIZE"
            ) from error
        if result.world_size != 2:
            raise RuntimeError("P3.5 requires WORLD_SIZE=2")
        if not 0 <= result.rank < result.world_size:
            raise RuntimeError("RANK is outside WORLD_SIZE")
        if not 0 <= result.local_rank < result.world_size:
            raise RuntimeError("LOCAL_RANK is outside WORLD_SIZE")
        return result


def accumulation_windows(
    values: Iterable[T], *, accumulation_steps: int
) -> Iterator[list[T]]:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    window: list[T] = []
    for value in values:
        window.append(value)
        if len(window) == accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def local_weight_denominator(labels: Any, loss_weights: Any) -> Any:
    """Return the shifted weighted-token denominator without model logits."""

    import torch

    if labels.ndim != 2 or loss_weights.ndim != 2 or labels.shape != loss_weights.shape:
        raise ValueError("labels and loss_weights must be equal-shaped [B,T] tensors")
    shifted_labels = labels[:, 1:]
    shifted_weights = loss_weights[:, 1:].float()
    valid = shifted_labels.ne(-100)
    if torch.any(shifted_weights[~valid] != 0):
        raise ValueError("ignored prompt/padding labels must have zero weight")
    denominator = (shifted_weights * valid.float()).sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise ValueError("weighted accumulation window has no finite supervision")
    return denominator


def ddp_scaled_loss(local_numerator: Any, global_denominator: Any, *, world_size: int) -> Any:
    """Scale a local numerator so DDP's rank-average equals the global token mean."""

    import torch

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if local_numerator.ndim != 0 or global_denominator.ndim != 0:
        raise ValueError("weighted numerator and denominator must be scalar tensors")
    if not torch.isfinite(local_numerator):
        raise FloatingPointError("local weighted numerator is non-finite")
    if not torch.isfinite(global_denominator) or global_denominator.item() <= 0:
        raise FloatingPointError("global weighted denominator is non-finite or empty")
    return local_numerator * float(world_size) / global_denominator


def distributed_sample_indices(
    dataset_size: int, *, rank: int, world_size: int, seed: int, epoch: int
) -> list[int]:
    """Mirror a no-padding shuffled DistributedSampler for the frozen divisible set."""

    import torch

    if dataset_size <= 0 or world_size <= 0 or dataset_size % world_size:
        raise ValueError("formal DDP dataset must be non-empty and divisible by world_size")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world_size")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    return shuffled[rank:dataset_size:world_size]


def _ordered_sha(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_global_sample_coverage(
    per_rank_indices: Sequence[Sequence[int]], *, expected_size: int, expected_per_rank: int
) -> dict[str, Any]:
    if len(per_rank_indices) != 2:
        raise ValueError("P3.5 coverage requires exactly two ranks")
    if any(len(values) != expected_per_rank for values in per_rank_indices):
        raise ValueError("rank-local sample count differs from the frozen contract")
    flattened = [int(value) for values in per_rank_indices for value in values]
    unique = set(flattened)
    expected = set(range(expected_size))
    duplicates = len(flattened) - len(unique)
    missing = len(expected - unique)
    unexpected = len(unique - expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"DDP sample coverage failed: duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {
        "world_size": 2,
        "rank_sample_counts": [len(values) for values in per_rank_indices],
        "rank_index_sha256": [
            _ordered_sha([str(value) for value in values]) for values in per_rank_indices
        ],
        "global_unique_samples": len(unique),
        "duplicate_samples": duplicates,
        "missing_samples": missing,
        "unexpected_samples": unexpected,
    }


def rank_zero_write_json(path: str | Path, payload: Mapping[str, Any], *, rank: int) -> bool:
    if rank != 0:
        return False
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return True


def freeze_calibration_selection(
    records_path: str | Path, *, seed: int
) -> dict[str, Any]:
    """Freeze one worst-length window plus three deterministic conditional windows."""

    source = Path(records_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[tuple[int, str]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target_role") != "medical_sft_train":
                raise ValueError("calibration selection may only inspect medical_sft_train")
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"missing/duplicate calibration sample_id at line {line_number}")
            seen.add(sample_id)
            prompt = row.get("token_count_prompt")
            response = row.get("token_count_response")
            if type(prompt) is not int or type(response) is not int or prompt <= 0 or response <= 0:
                raise ValueError("calibration selection requires frozen positive token counts")
            # The frozen SFT-v2 response count already includes assistant EOS.
            rows.append((prompt + response, sample_id))
    if len(rows) < 64:
        raise ValueError("calibration selection requires at least 64 formal rows")
    ordered = sorted(rows, key=lambda item: (-item[0], item[1]))
    primary = ordered[:16]
    remaining = ordered[16:]
    conditional = sorted(
        remaining,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item[1]}".encode("utf-8")).hexdigest(),
            item[1],
        ),
    )[:48]

    def assign(values: Sequence[tuple[int, str]]) -> dict[str, list[str]]:
        return {
            "rank0": [sample_id for index, (_, sample_id) in enumerate(values) if index % 2 == 0],
            "rank1": [sample_id for index, (_, sample_id) in enumerate(values) if index % 2 == 1],
        }

    windows = [assign(conditional[start : start + 16]) for start in range(0, 48, 16)]
    lengths = sorted(length for length, _ in rows)
    p95_index = min(len(lengths) - 1, int(round((len(lengths) - 1) * 0.95)))
    return {
        "schema_version": 1,
        "status": "frozen_before_gpu",
        "target_role": "medical_sft_train",
        "seed": int(seed),
        "world_size": 2,
        "microbatches_per_rank": 8,
        "primary_selection": "top16_by_frozen_prompt_plus_response_including_eos_length",
        "primary_window": assign(primary),
        "primary_lengths": {
            "min": min(length for length, _ in primary),
            "max": max(length for length, _ in primary),
            "includes_global_max": ordered[0][1]
            in {sample_id for _, sample_id in primary},
        },
        "formal_length_summary": {
            "records": len(rows),
            "p95": lengths[p95_index],
            "max": lengths[-1],
        },
        "conditional_selection": "sha256(seed:sample_id)_from_non_primary",
        "conditional_windows": windows,
        "records_sha256": _file_sha256(source),
        "final_authorized": False,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_memory_margin(margin_mib: float) -> str:
    if margin_mib >= 768.0:
        return "pass"
    if margin_mib >= 256.0:
        return "conditional"
    return "fail"


def validate_training_source_contract(source: str) -> None:
    """Static fail-closed guard for the formal launcher target."""

    forbidden = (
        "nn." + "DataParallel(",
        "device_map=" + '"auto"',
        "device_map=" + "'auto'",
        "PARENT_LOADS_MODEL_BEFORE_TORCHRUN = True",
    )
    hits = [value for value in forbidden if value in source]
    if hits:
        raise ValueError(f"formal DDP source contains forbidden topology: {hits}")
