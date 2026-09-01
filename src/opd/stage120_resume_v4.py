"""Evidence-preserving P7 resume-tail archival and deterministic replay checks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


class P7ResumeReplayError(RuntimeError):
    """A P7 resume archive or deterministic replay invariant failed."""


_STEP_FILE_PATTERNS = {
    "formal_steps": re.compile(r"^step_(\d+)\.json$"),
    "stage120_action_steps_v4": re.compile(r"^step_(\d+)\.json$"),
    "ratio_evidence_v2": re.compile(r"^step_(\d+)\.json$"),
    "memory_step_audits": re.compile(r"^step_(\d+)\.json$"),
    "b2_steps": re.compile(r"^step_\d+_v\d+_to_v(\d+)\.json$"),
    "health_summaries_v4": re.compile(r"^through_step_(\d+)\.json$"),
    "steps": re.compile(r"^generation_health_failure_step_(\d+)\.json$"),
}
_JSON_STEP_DIRS = frozenset(
    {
        "bounded_rejections_v4",
        "actual_impact_rejections_v4",
        "rejected_updates_v2",
        "generation_health_rejections_v4",
        "resume_replay_verifications_v4",
    }
)
_NONDETERMINISTIC_TOP_LEVEL_KEYS = frozenset(
    {
        "disk_remaining_bytes",
        "gpu_memory_bytes",
        "gpu_step_end",
        "sampler_refresh_seconds",
        "throughput",
        "timings_seconds",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P7ResumeReplayError(
            f"resume tail JSON is unreadable: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise P7ResumeReplayError(f"resume tail JSON is not an object: {path}")
    return dict(value)


def _json_step(path: Path) -> int | None:
    value = _json(path)
    for key in (
        "optimizer_step",
        "attempted_optimizer_step",
        "accepted_slot",
        "through_step",
    ):
        raw = value.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw + 1 if key == "accepted_slot" else raw
    route = value.get("route_state")
    if isinstance(route, Mapping):
        accepted = route.get("accepted_steps")
        if isinstance(accepted, int) and not isinstance(accepted, bool):
            return accepted + 1
    return None


def _tail_files(output: Path, *, checkpoint_step: int) -> list[Path]:
    result: list[Path] = []
    for directory, pattern in _STEP_FILE_PATTERNS.items():
        root = output / directory
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            match = pattern.fullmatch(path.name)
            if match and int(match.group(1)) > checkpoint_step:
                result.append(path)
    for directory in _JSON_STEP_DIRS:
        root = output / directory
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            if path.is_symlink():
                raise P7ResumeReplayError("resume tail contains a symlink")
            step = _json_step(path)
            if step is not None and step > checkpoint_step:
                result.append(path)
    progress = output / "progress_v4.jsonl"
    if result and progress.is_file() and not progress.is_symlink():
        result.append(progress)
    return sorted(set(result), key=lambda path: str(path.relative_to(output)))


def _existing_archives(
    output: Path, *, checkpoint_step: int
) -> list[tuple[Path, dict[str, Any]]]:
    root = output / "resume_replay_archives_v4"
    result: list[tuple[Path, dict[str, Any]]] = []
    if not root.is_dir():
        return result
    for archive in sorted(root.iterdir(), key=lambda path: (path.stat().st_mtime_ns, path.name)):
        manifest_path = archive / "archive_manifest_v4.json"
        if not archive.is_dir() or archive.is_symlink() or not manifest_path.is_file():
            continue
        manifest = _json(manifest_path)
        files = manifest.get("files")
        if not (
            manifest.get("schema_version") == 4
            and manifest.get("artifact_kind") == "p7_resume_replay_archive_v4"
            and manifest.get("checkpoint_step") == checkpoint_step
            and manifest.get("historical_artifacts_preserved") is True
            and isinstance(files, list)
            and files
        ):
            continue
        for row in files:
            if not isinstance(row, Mapping):
                raise P7ResumeReplayError("resume replay archive row differs")
            path = archive / str(row.get("relative_path", ""))
            if not (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == row.get("size_bytes")
                and _sha256(path) == row.get("sha256")
            ):
                raise P7ResumeReplayError("resume replay archive SHA differs")
        result.append((archive, manifest))
    return result


def archive_uncheckpointed_tail_v4(
    output_dir: str | Path,
    *,
    checkpoint_step: int,
    process_id: str,
) -> dict[str, Any] | None:
    """Move post-checkpoint evidence into a unique SHA-indexed replay archive.

    Nothing is deleted or overwritten.  The active output retains the complete
    checkpoint boundary; deterministic replay may then recreate later steps.
    """

    output = Path(output_dir).resolve()
    if not (
        output.is_dir()
        and isinstance(checkpoint_step, int)
        and not isinstance(checkpoint_step, bool)
        and checkpoint_step > 0
        and isinstance(process_id, str)
        and process_id
        and "/" not in process_id
    ):
        raise P7ResumeReplayError("resume archive boundary differs")
    archive = (
        output
        / "resume_replay_archives_v4"
        / f"{process_id}_from_step_{checkpoint_step:03d}"
    )
    if archive.exists() or archive.is_symlink():
        raise P7ResumeReplayError("resume replay archive already exists")
    existing = _existing_archives(output, checkpoint_step=checkpoint_step)
    tail = _tail_files(output, checkpoint_step=checkpoint_step)
    if not tail:
        if not existing:
            return None
        source, manifest = existing[0]
        return {
            **manifest,
            "archive_root": str(source),
            "reused_existing_archive": True,
        }
    archive.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    archived_formal_steps: list[int] = []
    for source in tail:
        relative = source.relative_to(output)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise P7ResumeReplayError("resume replay destination already exists")
        digest = _sha256(source)
        size = source.stat().st_size
        source.replace(destination)
        rows.append(
            {
                "relative_path": str(relative),
                "size_bytes": size,
                "sha256": digest,
            }
        )
        match = re.fullmatch(r"formal_steps/step_(\d+)\.json", str(relative))
        if match:
            archived_formal_steps.append(int(match.group(1)))
    manifest = {
        "schema_version": 4,
        "artifact_kind": "p7_resume_replay_archive_v4",
        "checkpoint_step": checkpoint_step,
        "max_archived_accepted_step": max(archived_formal_steps, default=checkpoint_step),
        "historical_artifacts_preserved": True,
        "active_tail_removed_by_move_not_delete": True,
        "recovery": "move archived relative paths back only after proving active paths absent",
        "files": rows,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }
    _atomic_json(archive / "archive_manifest_v4.json", manifest)
    if existing:
        source, source_manifest = existing[0]
        return {
            **source_manifest,
            "archive_root": str(source),
            "reused_existing_archive": True,
            "latest_tail_archive_root": str(archive),
        }
    return {
        **manifest,
        "archive_root": str(archive),
        "reused_existing_archive": False,
    }


def _replay_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(record))
    for key in _NONDETERMINISTIC_TOP_LEVEL_KEYS:
        value.pop(key, None)
    return value


def validate_resume_replay_record_v4(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    expected_identity = _replay_identity(expected)
    observed_identity = _replay_identity(observed)
    step = expected_identity.get("optimizer_step")
    if not (
        isinstance(step, int)
        and not isinstance(step, bool)
        and observed_identity.get("optimizer_step") == step
        and expected_identity == observed_identity
    ):
        raise P7ResumeReplayError("resume replay record identity differs")
    payload = json.dumps(
        expected_identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema_version": 4,
        "artifact_kind": "p7_resume_replay_verification_v4",
        "passed": True,
        "optimizer_step": step,
        "identity_sha256": hashlib.sha256(payload).hexdigest(),
        "ignored_runtime_telemetry_keys": sorted(_NONDETERMINISTIC_TOP_LEVEL_KEYS),
        "final_access_count": 0,
    }


__all__ = [
    "P7ResumeReplayError",
    "archive_uncheckpointed_tail_v4",
    "validate_resume_replay_record_v4",
]
