"""Export only a PEFT LoRA from a one-GPU veRL FSDP checkpoint.

The implementation deliberately stops after veRL's LoRA extraction method; it
never calls the upstream full-model save path and therefore never creates a
merged Qwen3-4B copy. The real export is GPU-host work and may temporarily need
enough CPU RAM to read the single actor checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable


class ExportError(RuntimeError):
    pass


class _VerlMerger:
    def __init__(self, actor_dir: Path) -> None:  # pragma: no cover - GPU host only
        self.actor_dir = actor_dir

    def export_adapter_only(self, target: Path) -> Path:  # pragma: no cover - GPU host only
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger

        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(self.actor_dir),
            target_dir=str(target),
            hf_model_config_path=str(self.actor_dir / "huggingface"),
        )
        merger = FSDPModelMerger(config)
        world_size = merger._get_world_size()
        if world_size != 1:
            raise ExportError("2x3090 topology requires a one-GPU Student actor checkpoint")
        rank_zero = merger._load_rank_zero_state_dict(world_size)
        mesh, mesh_names = merger._extract_device_mesh_info(rank_zero, world_size)
        total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_names)
        del rank_zero
        if total_shards != 1:
            raise ExportError("adapter-only exporter supports exactly one Student FSDP shard")
        state = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_names)
        try:
            adapter = merger.save_lora_adapter(state)
        finally:
            del state
        if adapter is None:
            raise ExportError("veRL checkpoint contains no LoRA tensors")
        return Path(adapter)


def export_verl_lora_adapter(
    checkpoint_dir: str | Path,
    target_dir: str | Path,
    *,
    merger_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish a verified adapter-only artifact and its SHA."""

    checkpoint = Path(checkpoint_dir)
    actor = checkpoint / "actor"
    target = Path(target_dir)
    if target.exists():
        raise ExportError(f"adapter export target must be new/empty: {target}")
    required = (
        actor / "fsdp_config.json",
        actor / "model_world_size_1_rank_0.pt",
        actor / "huggingface" / "config.json",
    )
    if any(not path.exists() for path in required):
        raise ExportError("veRL actor checkpoint is incomplete or not one-GPU FSDP")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise ExportError(f"stale adapter export temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        factory = merger_factory or (lambda path: _VerlMerger(path))
        adapter = Path(factory(actor).export_adapter_only(temporary))
        config = adapter / "adapter_config.json"
        weights = adapter / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise ExportError("adapter exporter did not produce complete PEFT files")
        unexpected = [
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"adapter_config.json", "adapter_model.safetensors"}
        ]
        if unexpected:
            raise ExportError("adapter-only export unexpectedly produced non-adapter files")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    adapter = target / "lora_adapter"
    config, weights = adapter / "adapter_config.json", adapter / "adapter_model.safetensors"
    combined = hashlib.sha256(config.read_bytes() + weights.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_type": "peft_lora_only",
        "source_checkpoint": str(checkpoint.resolve()),
        "adapter_path": str(adapter.resolve()),
        "adapter_sha256": combined,
        "files": {
            "adapter_config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
            "adapter_model.safetensors": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
        "merged_model_saved": False,
    }
    (target / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
