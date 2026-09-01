"""Fail-closed prediction cache identities for P6 Controller evaluation.

Historical Controller artifacts did not bind every model/evaluator component.
This module deliberately has no legacy fallback: an entry is reusable only
when the complete identity and the immutable prediction bytes both match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


class P6IdentityCacheError(RuntimeError):
    """A P6 prediction cache is incomplete, mutable, or identity-mismatched."""


IDENTITY_FIELDS = (
    "base_model_revision",
    "adapter_ordered_sha256",
    "adapter_weight_sha256",
    "adapter_manifest_sha256",
    "tokenizer_revision",
    "template_sha256",
    "scorer_backend",
    "scorer_version_sha256",
    "evaluator_config_sha256",
    "prompt_manifest_sha256",
    "label_manifest_sha256",
    "decoding_config_sha256",
    "code_git_sha",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def build_prediction_cache_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and hash the complete registered P6 cache-key projection."""

    if value.get("schema_version") != 1 or set(value) != {
        "schema_version",
        *IDENTITY_FIELDS,
    }:
        raise P6IdentityCacheError("prediction cache identity fields differ")
    for field in IDENTITY_FIELDS:
        item = value[field]
        if field in {"base_model_revision", "tokenizer_revision", "code_git_sha"}:
            if not _is_hex(item, 40):
                raise P6IdentityCacheError(f"prediction cache identity invalid: {field}")
        elif field == "scorer_backend":
            if item != "transformers_direct_logits":
                raise P6IdentityCacheError("prediction cache identity scorer differs")
        elif not _is_hex(item, 64):
            raise P6IdentityCacheError(f"prediction cache identity invalid: {field}")
    result = dict(value)
    result["cache_key_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return result


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_prediction_cache(
    root: Path,
    *,
    identity: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a new immutable cache entry; existing targets are never replaced."""

    root = Path(root)
    if root.exists() or root.is_symlink():
        raise P6IdentityCacheError("prediction cache output must be fresh")
    checked = build_prediction_cache_identity(
        {field: identity[field] for field in ("schema_version", *IDENTITY_FIELDS)}
    )
    if identity.get("cache_key_sha256") != checked["cache_key_sha256"]:
        raise P6IdentityCacheError("prediction cache identity digest differs")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise P6IdentityCacheError("prediction cache cannot be empty")
    for row in materialized:
        if not (
            isinstance(row.get("sample_id"), str)
            and isinstance(row.get("predicted_label"), str)
            and isinstance(row.get("candidate_scores"), Mapping)
        ):
            raise P6IdentityCacheError("prediction cache row differs")
    root.mkdir(parents=True)
    prediction_path = root / "predictions.jsonl"
    content = b"".join(_canonical_bytes(row) + b"\n" for row in materialized)
    _atomic_bytes(prediction_path, content)
    prediction_sha = _sha_file(prediction_path)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "p6_identity_bound_prediction_cache",
        "identity": checked,
        "cache_key_sha256": checked["cache_key_sha256"],
        "prediction_path": prediction_path.name,
        "prediction_sha256": prediction_sha,
        "prediction_bytes": prediction_path.stat().st_size,
        "row_count": len(materialized),
        "legacy_cache_used": False,
        "final_access_count": 0,
    }
    _atomic_bytes(root / "cache_manifest.json", _canonical_bytes(manifest) + b"\n")
    return manifest


def read_prediction_cache(
    root: Path, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Read only a byte- and identity-exact P6 entry."""

    root = Path(root)
    manifest_path = root / "cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6IdentityCacheError("prediction cache manifest is invalid") from error
    checked = build_prediction_cache_identity(
        {field: expected_identity[field] for field in ("schema_version", *IDENTITY_FIELDS)}
    )
    if not (
        isinstance(manifest, Mapping)
        and manifest.get("artifact_kind") == "p6_identity_bound_prediction_cache"
        and manifest.get("identity") == checked
        and manifest.get("cache_key_sha256") == checked["cache_key_sha256"]
    ):
        raise P6IdentityCacheError("prediction cache identity differs")
    prediction_path = root / str(manifest.get("prediction_path", ""))
    if not (
        prediction_path.is_file()
        and not prediction_path.is_symlink()
        and _sha_file(prediction_path) == manifest.get("prediction_sha256")
        and prediction_path.stat().st_size == manifest.get("prediction_bytes")
    ):
        raise P6IdentityCacheError("prediction SHA or size differs")
    rows: list[dict[str, Any]] = []
    try:
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise P6IdentityCacheError("prediction cache row is not an object")
                    rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6IdentityCacheError("prediction cache rows are invalid") from error
    if len(rows) != manifest.get("row_count"):
        raise P6IdentityCacheError("prediction cache row count differs")
    return {
        "identity": checked,
        "rows": rows,
        "prediction_sha256": manifest["prediction_sha256"],
        "manifest_sha256": _sha_file(manifest_path),
        "cache_hit": True,
        "legacy_cache_used": False,
    }


__all__ = [
    "IDENTITY_FIELDS",
    "P6IdentityCacheError",
    "build_prediction_cache_identity",
    "read_prediction_cache",
    "write_prediction_cache",
]
