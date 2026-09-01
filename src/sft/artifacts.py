"""Atomic, adapter-only lifecycle artifacts for the production SFT runner."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class SFTArtifactError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize_sft_run_inventory(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    data_manifest_path: str | Path,
    run_id: str,
) -> None:
    """Create the standard run inventory before any model weight is loaded."""

    root = Path(run_dir)
    config = Path(config_path)
    manifest_path = Path(data_manifest_path)
    if not config.is_file() or not manifest_path.is_file():
        raise SFTArtifactError("SFT config and data manifest snapshots must exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("data_protocol_version") != "ca-opd-data-v2"
        or "medical_sft_train" not in (manifest.get("roles") or {})
    ):
        raise SFTArtifactError("SFT inventory requires the formal medical_sft_train manifest")
    if root.exists() and any(root.iterdir()):
        raise SFTArtifactError(f"SFT run directory is not new/empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=False)
    (root / "config.yaml").write_bytes(config.read_bytes())
    _atomic_json(root / "data_manifest.json", manifest)
    (root / "metrics.jsonl").write_bytes(b"")
    (root / "stdout.log").write_bytes(b"")
    _atomic_json(root / "summary.json", {
        "run_id": run_id,
        "stage": "sft",
        "status": "prepared_not_started",
        "metrics_summary": None,
        "actual_cost_cny": None,
    })
    _atomic_json(checkpoints / "index.json", {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoints": [],
    })
    _atomic_json(root / "cost.json", {
        "currency": "CNY",
        "price_cny_per_hour": None,
        "estimated_cost_cny": None,
        "actual_cost_cny": None,
        "cost_status": "pending_live_run_reconciliation",
    })


def _adapter_files(adapter: Path) -> list[Path]:
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    if not adapter.is_dir() or any(not path.is_file() for path in required):
        raise SFTArtifactError("complete PEFT adapter_config/model safetensors are required")
    return required


def _combined_sha(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_lora_run(
    run_dir: str | Path,
    *,
    adapter_dir: str | Path,
    run_id: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    data_manifest_sha256: str,
    metrics: Mapping[str, Any],
    log_history: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Publish one physical LoRA plus auditable best/latest symlink pointers."""

    root, adapter = Path(run_dir).resolve(), Path(adapter_dir).resolve()
    if adapter.parent != root:
        raise SFTArtifactError("adapter must be the run's single physical adapter directory")
    files = _adapter_files(adapter)
    adapter_sha = _combined_sha(files)
    adapter_model_sha = _file_sha(adapter / "adapter_model.safetensors")
    inventory = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
         "sha256": _file_sha(path)}
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "artifact_type": "peft_lora_only",
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "data_protocol_version": "ca-opd-data-v2",
        "data_manifest_sha256": data_manifest_sha256,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha,
        "adapter_hash_algorithm": (
            "sha256(concat(adapter_config.json,adapter_model.safetensors))"
        ),
        "adapter_model_sha256": adapter_model_sha,
        "files": inventory,
        "merged_model_saved": False,
        "actual_cost_cny": None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    for pointer in ("best", "latest"):
        target = root / pointer
        if target.exists() or target.is_symlink():
            raise SFTArtifactError(f"refusing to replace existing adapter pointer: {target}")
        target.symlink_to(adapter.name, target_is_directory=True)
    _atomic_json(root / "artifact_manifest.json", manifest)
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "purpose": "Medical SFT -> Medical Teacher LoRA",
                "status": "completed",
                "completed_at": manifest["completed_at"],
                "data_manifest_hash": data_manifest_sha256,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "adapter_sha256": adapter_sha,
                "adapter_model_sha256": adapter_model_sha,
                "actual_cost_cny": None,
            }
        )
        _atomic_json(metadata_path, metadata)
    checkpoints: list[dict[str, Any]] = []
    for checkpoint in sorted(
        root.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    ):
        try:
            step = int(checkpoint.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoint_files = _adapter_files(checkpoint)
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise SFTArtifactError(f"checkpoint lacks trainer_state.json: {checkpoint}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) != step:
            raise SFTArtifactError(f"checkpoint step/trainer state mismatch: {checkpoint}")
        checkpoints.append(
            {
                "name": checkpoint.name,
                "path": str(checkpoint),
                "step": step,
                "sha256": _combined_sha(checkpoint_files),
                "adapter_model_sha256": _file_sha(checkpoint / "adapter_model.safetensors"),
                "status": "complete",
                "best": False,
                "latest": False,
            }
        )
    checkpoints.append({
        "name": "adapter", "path": str(adapter), "sha256": adapter_sha,
        "adapter_model_sha256": adapter_model_sha,
        "status": "complete", "best": True, "latest": True,
    })
    _atomic_json(
        root / "checkpoints/index.json",
        {"schema_version": 1, "run_id": run_id, "checkpoints": checkpoints},
    )
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        history = list(log_history or [])
        for item in history:
            handle.write(json.dumps(dict(item), sort_keys=True) + "\n")
        final_record = {"step": int(metrics.get("global_step", 0)), **dict(metrics)}
        if not history or any(final_record.get(key) != history[-1].get(key) for key in final_record):
            handle.write(json.dumps(final_record, sort_keys=True) + "\n")
    _atomic_json(root / "cost.json", {
        "currency": "CNY", "price_cny_per_hour": None,
        "estimated_cost_cny": None, "actual_cost_cny": None,
        "cost_status": "pending_invoice_reconciliation",
    })
    return manifest


def record_sft_failure(run_dir: str | Path, *, run_id: str, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise SFTArtifactError("failure reason must be explicit")
    payload = {
        "run_id": run_id,
        "stage": "sft",
        "status": "failed",
        "failure_reason": reason.strip(),
        "actual_cost_cny": None,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    root = Path(run_dir)
    _atomic_json(root / "failure.json", payload)
    _atomic_json(root / "summary.json", {
        "run_id": run_id,
        "stage": "sft",
        "status": "failed",
        "completed_optimizer_steps": None,
        "metrics_summary": None,
        "actual_cost_cny": None,
        "failure_reason": reason.strip(),
        "failed_at": payload["failed_at"],
    })
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "status": "failed",
            "failure_reason": reason.strip(),
            "failed_at": payload["failed_at"],
            "actual_cost_cny": None,
        })
        _atomic_json(metadata_path, metadata)
    return payload
