"""Atomic, SHA-bound formal B2 checkpoint sealing and verification.

The production kernel emits a transient, identity-verified adapter at every
optimizer step.  This module promotes only registered ten-step boundaries to
complete resume checkpoints without copying Base weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from src.opd.production_b2_formal_v1 import (
    FormalB2Error,
    checkpoint_retention,
)


ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_transport_manifest.json",
)
STATE_FILES = (
    "optimizer_state.pt",
    "scheduler_state.pt",
    "rng_state.pt",
    "training_state.json",
    "sampler_state.json",
    "route_state.json",
    "environment.json",
    "evidence_index.json",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise FormalB2Error(f"formal checkpoint {label} is not a SHA-256")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_files(root: Path) -> None:
    for path in sorted(root.iterdir()):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


def _formal_checkpoint_upper_bound(config: Mapping[str, Any]) -> int:
    """Widen sealing only for the exact kernel-authorized P9 discriminator."""

    if "p9_adaptive_dose" not in config:
        return 150
    from src.opd.p9_runtime import p9_optimizer_step_limit

    limit = p9_optimizer_step_limit(config)
    if limit != 300:
        raise FormalB2Error("P9 formal checkpoint upper bound differs")
    return limit


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalB2Error(
            f"formal checkpoint {label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, Mapping):
        raise FormalB2Error(f"formal checkpoint {label} is not an object")
    return value


def validate_formal_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Verify the complete marker, manifest, every file SHA, and resume state."""

    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FormalB2Error("formal checkpoint directory is absent")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    marker_path = checkpoint / "complete_marker.json"
    manifest = _json(manifest_path, "manifest")
    marker = _json(marker_path, "complete marker")
    step = manifest.get("logical_version")
    files = manifest.get("files")
    required = set(ADAPTER_FILES + STATE_FILES)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == "formal_b2_checkpoint_manifest_v1"
        and isinstance(step, int)
        and not isinstance(step, bool)
        and step > 0
        and step % 10 == 0
        and manifest.get("optimizer_step") == step
        and manifest.get("scheduler_step") == step
        and manifest.get("policy_version") == step
        and manifest.get("data_cursor") == step * 4
        and manifest.get("complete") is True
        and manifest.get("resume_eligible") is True
        and isinstance(files, Mapping)
        and set(files) == required
        and marker
        == {
            "schema_version": 1,
            "artifact_kind": "formal_b2_checkpoint_complete_marker_v1",
            "run_id": manifest.get("run_id"),
            "logical_version": step,
            "checkpoint_manifest_sha256": _sha_file(manifest_path),
            "complete": True,
            "resume_eligible": True,
        }
    ):
        raise FormalB2Error("formal checkpoint schema/complete marker drift")
    for field in (
        "adapter_sha256",
        "package_content_sha256",
        "config_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "evidence_index_sha256",
    ):
        _digest(manifest.get(field), field)
    for name, descriptor in files.items():
        path = checkpoint / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise FormalB2Error(f"formal checkpoint file SHA/size mismatch: {name}")
    if manifest["evidence_index_sha256"] != files["evidence_index.json"]["sha256"]:
        raise FormalB2Error("formal checkpoint evidence index SHA differs")
    return dict(manifest)


def _evidence_index(output: Path, *, run_id: str, step: int) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in ("formal_steps", "memory_step_audits", "memory_telemetry/markers"):
        root = output / relative
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.glob("*.json")):
                if path.is_file() and not path.is_symlink():
                    files.append(
                        {
                            "path": str(path.relative_to(output)),
                            "sha256": _sha_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
    return {
        "schema_version": 1,
        "artifact_kind": "formal_b2_checkpoint_evidence_index_v1",
        "run_id": run_id,
        "logical_version": step,
        "files": files,
    }


def seal_formal_checkpoint(
    session: Any,
    *,
    logical_version: int,
    data_cursor: int,
    package_content_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    schedule_sha256: str,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal current transient adapter and full recovery state via atomic rename."""

    step = int(logical_version)
    upper_bound = _formal_checkpoint_upper_bound(getattr(session, "config", {}))
    if not (
        step in range(10, upper_bound + 1, 10)
        and step == session.current_sampler_version
        and data_cursor == step * 4
        and session.optimizer is not None
        and session.scheduler is not None
    ):
        raise FormalB2Error("formal checkpoint version/cursor/state is not eligible")
    package_sha = _digest(package_content_sha256, "package content")
    config_digest = _digest(config_sha256, "config")
    manifest_digest = _digest(manifest_sha256, "manifest")
    schedule_digest = _digest(schedule_sha256, "schedule")
    authority = session.authorities.get(step)
    runtime = session.current_sampler_runtime
    if not isinstance(authority, Mapping) or not isinstance(runtime, Mapping):
        raise FormalB2Error("formal checkpoint trainer/sampler authority is absent")
    adapter_sha = _digest(authority.get("aggregate_tensor_sha256"), "adapter")
    if runtime.get("aggregate_tensor_sha256") != adapter_sha:
        raise FormalB2Error("formal checkpoint trainer/sampler authority differs")

    output = Path(session.output).resolve()
    source = output / "checkpoints" / f"v{step}"
    if source.is_symlink() or not source.is_dir():
        raise FormalB2Error("formal checkpoint transient adapter is absent")
    for name in ADAPTER_FILES:
        if not (source / name).is_file() or (source / name).is_symlink():
            raise FormalB2Error(f"formal checkpoint source file absent: {name}")
    root = output / "formal_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:03d}"
    if target.exists() or target.is_symlink():
        raise FormalB2Error("formal checkpoint target already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".step_{step:03d}.", dir=root))
    try:
        for name in ADAPTER_FILES:
            shutil.copy2(source / name, temporary / name)
        session._atomic_torch_save(
            temporary / "optimizer_state.pt", session.optimizer.state_dict()
        )
        session._atomic_torch_save(
            temporary / "scheduler_state.pt", session.scheduler.state_dict()
        )
        session._atomic_torch_save(
            temporary / "rng_state.pt",
            {
                "cpu": session.torch.get_rng_state(),
                "cuda": session.torch.cuda.get_rng_state_all(),
            },
        )
        run_id = str(session.config["run"]["run_id"]) if hasattr(session, "config") else "formal-b2-fixture"
        _atomic_json(
            temporary / "training_state.json",
            {
                "optimizer_step": step,
                "scheduler_step": step,
                "policy_version": step,
                "data_cursor": data_cursor,
                "package_content_sha256": package_sha,
                "config_sha256": config_digest,
                "manifest_sha256": manifest_digest,
                "schedule_sha256": schedule_digest,
            },
        )
        registry = runtime.get("registry_snapshot")
        _atomic_json(
            temporary / "sampler_state.json",
            {
                "policy_version": step,
                "runtime_adapter_sha256": adapter_sha,
                "active_adapter": runtime.get("active_adapter"),
                "registry_count": (
                    registry.get("adapter_count")
                    if isinstance(registry, Mapping)
                    else None
                ),
            },
        )
        route_state_builder = getattr(session, "formal_route_state", None)
        route_state = (
            route_state_builder()
            if callable(route_state_builder)
            else {
                "method": "B2",
                "teacher_route": "medical_teacher",
                "medical_teacher_fraction": 1.0,
                "base_teacher_fraction": 0.0,
                "adaptive_routing": False,
            }
        )
        if not isinstance(route_state, Mapping):
            raise FormalB2Error("formal checkpoint route state differs")
        _atomic_json(temporary / "route_state.json", dict(route_state))
        _atomic_json(temporary / "environment.json", dict(environment))
        _atomic_json(
            temporary / "evidence_index.json",
            _evidence_index(output, run_id=run_id, step=step),
        )
        _fsync_files(temporary)
        file_names = ADAPTER_FILES + STATE_FILES
        files = {
            name: {
                "sha256": _sha_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in file_names
        }
        manifest = {
            "schema_version": 1,
            "artifact_kind": "formal_b2_checkpoint_manifest_v1",
            "run_id": run_id,
            "logical_version": step,
            "adapter_sha256": adapter_sha,
            "optimizer_step": step,
            "scheduler_step": step,
            "policy_version": step,
            "data_cursor": data_cursor,
            "package_content_sha256": package_sha,
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "schedule_sha256": schedule_digest,
            "evidence_index_sha256": files["evidence_index.json"]["sha256"],
            "complete": True,
            "resume_eligible": True,
            "files": files,
        }
        _atomic_json(temporary / "checkpoint_manifest.json", manifest)
        _atomic_json(
            temporary / "complete_marker.json",
            {
                "schema_version": 1,
                "artifact_kind": "formal_b2_checkpoint_complete_marker_v1",
                "run_id": run_id,
                "logical_version": step,
                "checkpoint_manifest_sha256": _sha_file(
                    temporary / "checkpoint_manifest.json"
                ),
                "complete": True,
                "resume_eligible": True,
            },
        )
        _fsync_files(temporary)
        validate_formal_checkpoint(temporary)
        os.replace(temporary, target)
        _fsync_directory(root)
        validated = validate_formal_checkpoint(target)
        return validated
    except BaseException:
        if temporary.exists() and temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def prune_formal_checkpoints(output: Path, *, target_step: int) -> dict[str, Any]:
    """Remove only formal-owned nonmilestones displaced by rolling retention."""

    output = Path(output).resolve()
    root = output / "formal_checkpoints"
    if root.is_symlink() or not root.is_dir():
        raise FormalB2Error("formal checkpoint root is absent for pruning")
    present: dict[int, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.is_symlink() or not path.name.startswith("step_"):
            raise FormalB2Error("unexpected entry in formal checkpoint root")
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError as error:
            raise FormalB2Error("unexpected formal checkpoint directory name") from error
        if path.name != f"step_{step:03d}" or step % 10 != 0:
            raise FormalB2Error("unexpected formal checkpoint directory name")
        if not (
            (path / "checkpoint_manifest.json").is_file()
            or (path / "owned_by_formal_b2").is_file()
        ):
            raise FormalB2Error("checkpoint pruning target lacks formal ownership")
        present[step] = path
    retained = checkpoint_retention(present, target_step=target_step)
    removed = sorted(set(present) - retained)
    for step in removed:
        path = present[step]
        if path.resolve().parent != root.resolve():
            raise FormalB2Error("checkpoint pruning target escaped formal root")
        shutil.rmtree(path)
    _fsync_directory(root)
    return {
        "target_step": target_step,
        "retained_steps": sorted(retained),
        "removed_steps": removed,
    }


__all__ = [
    "prune_formal_checkpoints",
    "seal_formal_checkpoint",
    "validate_formal_checkpoint",
]
