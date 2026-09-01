"""Frozen, prompt-only selection contract for P4.6.

The checked-in manifest contains only stable sample identities.  Raw prompt
text is reopened from the already frozen local prompt-only JSONL files at GPU
runtime and is never copied into a formal artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_ID = "ca-opd/p4.6-frozen-prompt-selection/v1"
SELECTION_RULE = "seed42_null_delimited_sha256_rank_v1"
RUN_ID = "qwen3-4b-production-qualification-v6-seed42"
SOURCE_ROLES = ("medical_opd_o1", "medical_opd_cmb")
GROUP_RANKS = {
    "step0": (0, 1),
    "step1": (2, 3),
    "base_null": (4, 5),
    "length": tuple(range(8)),
}
_TOP_FIELDS = {
    "schema_id",
    "schema_version",
    "run_id",
    "seed",
    "selection_rule",
    "data_manifest_sha256",
    "source_files",
    "groups",
    "contains_labels",
    "contains_final",
    "contains_controller",
    "contains_confirmation",
    "manifest_content_sha256",
}
_SOURCE_FIELDS = {"path", "sha256", "row_count"}
_GROUP_FIELDS = {"group_id", "purpose", "per_source_count", "ordered_samples"}
_SAMPLE_FIELDS = {"sample_id", "content_hash", "source_role", "rank_index"}
_SUPERVISION_FIELDS = {
    "answer",
    "answer_idx",
    "answer_index",
    "completion",
    "final",
    "final_answer",
    "gold",
    "label",
    "labels",
    "output",
    "reasoning",
    "response",
    "reward",
    "solution",
    "target",
}


class FrozenPromptSelectionError(RuntimeError):
    """The checked-in selection or its prompt-only source drifted."""


def _fail(message: str) -> None:
    raise FrozenPromptSelectionError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_source(
    *, path: Path, expected_sha: str, expected_role: str, expected_count: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.is_file() or path.is_symlink() or _sha_file(path) != expected_sha:
        _fail(f"{expected_role} prompt-only source identity mismatch")
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                _fail(f"{expected_role} line {line_number} is invalid: {error}")
            if not isinstance(row, dict):
                _fail(f"{expected_role} line {line_number} is not an object")
            forbidden = sorted(_SUPERVISION_FIELDS.intersection(row))
            if forbidden:
                _fail(f"{expected_role} source contains supervision: {forbidden}")
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            role = row.get("target_role")
            if not isinstance(sample_id, str) or not sample_id:
                _fail(f"{expected_role} source has an invalid sample_id")
            _digest(content_hash, f"{expected_role} content_hash")
            if role != expected_role or not isinstance(row.get("question"), str):
                _fail(f"{expected_role} source row identity is invalid")
            if sample_id in by_id:
                _fail(f"{expected_role} source has duplicate sample_id")
            rows.append(row)
            by_id[sample_id] = row
    if len(rows) != expected_count:
        _fail(f"{expected_role} row count mismatch")
    return rows, by_id


def _read_and_validate(
    config: Mapping[str, Any], *, repo_root: str | Path
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    root = Path(repo_root).resolve()
    prompt = config.get("prompt_selection")
    if not isinstance(prompt, Mapping):
        _fail("prompt_selection config is absent")
    manifest_path = _resolve(root, prompt.get("selection_manifest_path"))
    expected_file_sha = _digest(
        prompt.get("selection_manifest_sha256"), "selection manifest file SHA"
    )
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha_file(manifest_path) != expected_file_sha
    ):
        _fail("selection manifest file SHA mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenPromptSelectionError("selection manifest is invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _TOP_FIELDS:
        _fail("selection manifest fields are not exact")
    if (
        manifest["schema_id"] != SCHEMA_ID
        or manifest["schema_version"] != 1
        or manifest["run_id"] != RUN_ID
        or manifest["run_id"] != config.get("run", {}).get("run_id")
        or manifest["seed"] != 42
        or manifest["seed"] != config.get("run", {}).get("seed")
        or manifest["selection_rule"] != SELECTION_RULE
        or manifest["data_manifest_sha256"] != prompt.get("opd_manifest_sha256")
        or any(
            manifest[field] is not False
            for field in (
                "contains_labels",
                "contains_final",
                "contains_controller",
                "contains_confirmation",
            )
        )
    ):
        _fail("selection manifest protocol drift")
    claimed_content_sha = _digest(
        manifest["manifest_content_sha256"], "selection manifest content SHA"
    )
    content = dict(manifest)
    content.pop("manifest_content_sha256")
    if _canonical_sha(content) != claimed_content_sha:
        _fail("selection manifest content SHA mismatch")

    sources = manifest["source_files"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_ROLES):
        _fail("selection source inventory mismatch")
    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    ranked_ids: dict[str, list[str]] = {}
    for role in SOURCE_ROLES:
        descriptor = sources[role]
        if not isinstance(descriptor, dict) or set(descriptor) != _SOURCE_FIELDS:
            _fail(f"{role} source descriptor fields are not exact")
        config_prefix = "medical_opd_o1" if role == "medical_opd_o1" else "medical_opd_cmb"
        if (
            descriptor["path"] != prompt.get(f"{config_prefix}_path")
            or descriptor["sha256"] != prompt.get(f"{config_prefix}_sha256")
            or not isinstance(descriptor["row_count"], int)
            or descriptor["row_count"] < 8
        ):
            _fail(f"{role} source descriptor/config mismatch")
        rows, by_id = _load_source(
            path=_resolve(root, descriptor["path"]),
            expected_sha=_digest(descriptor["sha256"], f"{role} source SHA"),
            expected_role=role,
            expected_count=descriptor["row_count"],
        )
        ranked = sorted(
            rows,
            key=lambda row: (
                int(
                    hashlib.sha256(
                        f"42\0{row['sample_id']}\0{row['content_hash']}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    16,
                ),
                row["sample_id"],
            ),
        )
        source_rows[role] = by_id
        ranked_ids[role] = [row["sample_id"] for row in ranked]

    groups = manifest["groups"]
    if not isinstance(groups, list) or [item.get("group_id") for item in groups] != list(
        GROUP_RANKS
    ):
        _fail("selection group order mismatch")
    step_ids: dict[str, set[str]] = {}
    for group in groups:
        if not isinstance(group, dict) or set(group) != _GROUP_FIELDS:
            _fail("selection group fields are not exact")
        group_id = group["group_id"]
        ranks = GROUP_RANKS[group_id]
        if group["per_source_count"] != len(ranks):
            _fail(f"{group_id} per-source count mismatch")
        samples = group["ordered_samples"]
        if not isinstance(samples, list) or len(samples) != len(ranks) * 2:
            _fail(f"{group_id} ordered sample count mismatch")
        expected_roles = [role for _ in ranks for role in SOURCE_ROLES]
        expected_ranks = [rank for rank in ranks for _ in SOURCE_ROLES]
        ids: set[str] = set()
        for sample, expected_role, expected_rank in zip(
            samples, expected_roles, expected_ranks, strict=True
        ):
            if not isinstance(sample, dict) or set(sample) != _SAMPLE_FIELDS:
                _fail(f"{group_id} sample fields are not exact")
            if (
                sample["source_role"] != expected_role
                or sample["rank_index"] != expected_rank
                or sample["sample_id"] != ranked_ids[expected_role][expected_rank]
            ):
                _fail(f"{group_id} deterministic rank mismatch")
            source_row = source_rows[expected_role].get(sample["sample_id"])
            if source_row is None or source_row["content_hash"] != sample["content_hash"]:
                _fail(f"{group_id} sample content identity mismatch")
            if sample["sample_id"] in ids:
                _fail(f"{group_id} contains a duplicate sample")
            ids.add(sample["sample_id"])
        step_ids[group_id] = ids
    if not step_ids["step0"].isdisjoint(step_ids["step1"]):
        _fail("step0 and step1 selections must be disjoint")
    return manifest, source_rows


def validate_frozen_prompt_selection(
    config: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Reopen and fully validate the checked-in prompt selection manifest."""

    manifest, _ = _read_and_validate(config, repo_root=repo_root)
    return manifest


def load_frozen_prompt_group(
    config: Mapping[str, Any], group_id: str, *, repo_root: str | Path
) -> list[dict[str, Any]]:
    """Load one frozen group in manifest order for in-memory GPU use only."""

    manifest, sources = _read_and_validate(config, repo_root=repo_root)
    groups = {group["group_id"]: group for group in manifest["groups"]}
    if group_id not in GROUP_RANKS or group_id not in groups:
        _fail(f"unknown frozen prompt group: {group_id}")
    return [
        dict(sources[item["source_role"]][item["sample_id"]])
        for item in groups[group_id]["ordered_samples"]
    ]


__all__ = [
    "FrozenPromptSelectionError",
    "GROUP_RANKS",
    "SELECTION_RULE",
    "load_frozen_prompt_group",
    "validate_frozen_prompt_selection",
]
