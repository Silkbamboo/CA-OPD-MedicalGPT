"""Formal B2 v2 retention policy: two full resumes plus controller adapters."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from src.opd.production_b2_formal_checkpoint_v1 import (
    ADAPTER_FILES,
    seal_formal_checkpoint,
    validate_formal_checkpoint,
)
from src.opd.production_b2_formal_v1 import FormalB2Error


CONTROLLER_SNAPSHOT_STEPS_V2 = frozenset({30, 60, 90, 120, 150})


def _steps(values: Iterable[int]) -> set[int]:
    result = set()
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("checkpoint step is invalid")
        result.add(value)
    return result


def rolling_resume_steps_v2(present_steps: Iterable[int]) -> set[int]:
    """Only the newest two complete full states remain resume eligible."""

    present = sorted(_steps(present_steps))
    return set(present[-2:])


def controller_snapshot_steps_v2(present_steps: Iterable[int]) -> set[int]:
    """Only preregistered positions may become immutable eval snapshots."""

    return _steps(present_steps) & CONTROLLER_SNAPSHOT_STEPS_V2


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_controller_snapshot_v2(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise FormalB2Error("controller adapter snapshot is absent")
    manifest_path = path / "snapshot_manifest.json"
    marker_path = path / "complete_marker.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalB2Error("controller adapter snapshot JSON is invalid") from error
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    step = manifest.get("logical_version") if isinstance(manifest, Mapping) else None
    if not (
        manifest.get("schema_version") == 2
        and manifest.get("artifact_kind") == "formal_b2_controller_adapter_snapshot_v2"
        and step in CONTROLLER_SNAPSHOT_STEPS_V2
        and manifest.get("resume_eligible") is False
        and isinstance(files, Mapping)
        and set(files) == set(ADAPTER_FILES)
        and marker
        == {
            "schema_version": 2,
            "artifact_kind": "formal_b2_controller_adapter_complete_marker_v2",
            "logical_version": step,
            "snapshot_manifest_sha256": _sha_file(manifest_path),
            "complete": True,
            "resume_eligible": False,
        }
    ):
        raise FormalB2Error("controller adapter snapshot schema differs")
    for name, descriptor in files.items():
        file = path / str(name)
        if not (
            file.is_file()
            and not file.is_symlink()
            and descriptor.get("sha256") == _sha_file(file)
            and descriptor.get("size_bytes") == file.stat().st_size
        ):
            raise FormalB2Error("controller adapter snapshot file differs")
    return dict(manifest)


def seal_controller_snapshot_v2(output: Path, *, logical_version: int) -> dict[str, Any]:
    output = Path(output).resolve()
    step = int(logical_version)
    if step not in CONTROLLER_SNAPSHOT_STEPS_V2:
        raise FormalB2Error("controller snapshot step is not preregistered")
    checkpoint = output / "formal_checkpoints" / f"step_{step:03d}"
    checkpoint_manifest = validate_formal_checkpoint(checkpoint)
    root = output / "controller_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:03d}"
    if target.exists() or target.is_symlink():
        raise FormalB2Error("controller snapshot target already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".step_{step:03d}.", dir=root))
    try:
        for name in ADAPTER_FILES:
            shutil.copy2(checkpoint / name, temporary / name)
        files = {
            name: {
                "sha256": _sha_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in ADAPTER_FILES
        }
        manifest = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_controller_adapter_snapshot_v2",
            "run_id": checkpoint_manifest["run_id"],
            "logical_version": step,
            "adapter_sha256": checkpoint_manifest["adapter_sha256"],
            "source_checkpoint_manifest_sha256": _sha_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "complete": True,
            "resume_eligible": False,
            "purpose": "controller_dev_only",
            "files": files,
        }
        _atomic_json(temporary / "snapshot_manifest.json", manifest)
        _atomic_json(
            temporary / "complete_marker.json",
            {
                "schema_version": 2,
                "artifact_kind": "formal_b2_controller_adapter_complete_marker_v2",
                "logical_version": step,
                "snapshot_manifest_sha256": _sha_file(
                    temporary / "snapshot_manifest.json"
                ),
                "complete": True,
                "resume_eligible": False,
            },
        )
        for file in temporary.iterdir():
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(temporary)
        validate_controller_snapshot_v2(temporary)
        os.replace(temporary, target)
        _fsync_directory(root)
        return validate_controller_snapshot_v2(target)
    except BaseException:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def prune_full_resume_checkpoints_v2(output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    root = output / "formal_checkpoints"
    if root.is_symlink() or not root.is_dir():
        raise FormalB2Error("formal checkpoint root is absent")
    present: dict[int, Path] = {}
    manifests: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("step_*")):
        manifest = validate_formal_checkpoint(path)
        step = int(manifest["logical_version"])
        present[step] = path
        manifests[step] = manifest
    retained = rolling_resume_steps_v2(present)
    removed = sorted(set(present) - retained)
    retired_path = output / "retired_resume_checkpoints_v2.json"
    retired: list[dict[str, Any]] = []
    if retired_path.is_file():
        value = json.loads(retired_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or not isinstance(value.get("checkpoints"), list):
            raise FormalB2Error("retired checkpoint index differs")
        retired = list(value["checkpoints"])
    for step in removed:
        snapshot = None
        if step in CONTROLLER_SNAPSHOT_STEPS_V2:
            snapshot_path = output / "controller_snapshots" / f"step_{step:03d}"
            snapshot = validate_controller_snapshot_v2(snapshot_path)
            if snapshot["adapter_sha256"] != manifests[step]["adapter_sha256"]:
                raise FormalB2Error("controller snapshot differs before state retirement")
        retired.append(
            {
                "logical_version": step,
                "adapter_sha256": manifests[step]["adapter_sha256"],
                "checkpoint_manifest_sha256": _sha_file(
                    present[step] / "checkpoint_manifest.json"
                ),
                "resume_eligible": False,
                "optimizer_state_removed_after_validation": True,
                "controller_snapshot": (
                    None
                    if snapshot is None
                    else f"controller_snapshots/step_{step:03d}"
                ),
            }
        )
    _atomic_json(
        retired_path,
        {
            "schema_version": 2,
            "artifact_kind": "formal_b2_retired_resume_checkpoint_index_v2",
            "checkpoints": retired,
        },
    )
    for step in removed:
        target = present[step]
        if target.resolve().parent != root:
            raise FormalB2Error("checkpoint retirement target escaped owned root")
        shutil.rmtree(target)
    _fsync_directory(root)
    return {"retained_full_resume_steps": sorted(retained), "retired_steps": removed}


def seal_formal_checkpoint_v2(session: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = seal_formal_checkpoint(session, **kwargs)
    step = int(manifest["logical_version"])
    snapshot = (
        seal_controller_snapshot_v2(session.output, logical_version=step)
        if step in CONTROLLER_SNAPSHOT_STEPS_V2
        else None
    )
    rotation = prune_full_resume_checkpoints_v2(session.output)
    return {"manifest": manifest, "controller_snapshot": snapshot, "rotation": rotation}


__all__ = [
    "CONTROLLER_SNAPSHOT_STEPS_V2",
    "controller_snapshot_steps_v2",
    "prune_full_resume_checkpoints_v2",
    "rolling_resume_steps_v2",
    "seal_controller_snapshot_v2",
    "seal_formal_checkpoint_v2",
    "validate_controller_snapshot_v2",
]
