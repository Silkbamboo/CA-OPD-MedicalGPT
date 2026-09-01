"""Configuration and artifact contracts for the resumable P2 formal builder."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.data.schema import (
    DATA_PROTOCOL_VERSION,
    FINAL_ROLES_V2,
    PROMPT_ONLY_ROLES_V2,
    SCHEMA_VERSION_V2,
    SOURCE_POLICY_VERSION,
    SUPERVISION_KEYS,
)
from src.data.medqa_conflicts_v2 import CONFLICT_POLICY_VERSION


_HEX40 = re.compile(r"[0-9a-f]{40}")


class FormalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ConfiguredSourceFile:
    source_key: str
    repository: str
    revision: str
    path: str
    file_format: str
    upstream_split: str
    adapter: str | None
    target_role: str | None
    denylist_only: bool
    subject: str | None = None


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FormalConfigError("formal config must be a mapping")
    return dict(payload)


def load_formal_config(path: str | Path) -> dict[str, Any]:
    config = _load_mapping(Path(path))
    if config.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise FormalConfigError("formal config has wrong data protocol version")
    if config.get("source_policy_version") != SOURCE_POLICY_VERSION:
        raise FormalConfigError("formal config has wrong source policy version")
    if config.get("schema_version") != SCHEMA_VERSION_V2:
        raise FormalConfigError("formal config has wrong schema version")
    limits = config.get("resource_limits")
    if not isinstance(limits, Mapping):
        raise FormalConfigError("resource_limits must be a mapping")
    if limits.get("num_proc") != 1:
        raise FormalConfigError("formal builder requires num_proc=1")
    batch = limits.get("batch_size")
    if type(batch) is not int or not 1 <= batch <= 128:
        raise FormalConfigError("resource batch_size must be in [1, 128]")
    abort = limits.get("abort_memory_bytes")
    if type(abort) is not int or abort <= 0 or abort > 1_932_735_283:
        raise FormalConfigError("abort_memory_bytes must be positive and <= 1.80 GiB")
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or _HEX40.fullmatch(
        str(tokenizer.get("revision", ""))
    ) is None:
        raise FormalConfigError("tokenizer revision must be an immutable 40-hex commit")
    if tokenizer.get("enable_thinking") is not False:
        raise FormalConfigError("formal tokenizer requires enable_thinking=false")
    conflict_policy = config.get("medqa_conflict_policy")
    if not isinstance(conflict_policy, Mapping):
        raise FormalConfigError("formal config must bind medqa_conflict_policy")
    if conflict_policy.get("version") != CONFLICT_POLICY_VERSION:
        raise FormalConfigError("formal config has wrong MedQA conflict policy version")
    if conflict_policy.get("consistent_conflict_action") != "keep_test_drop_validation":
        raise FormalConfigError("MedQA consistent conflict action must preserve final test")
    if conflict_policy.get("anomalous_conflict_action") != "quarantine_both_sides":
        raise FormalConfigError("MedQA anomaly policy must quarantine both sides")
    if conflict_policy.get("training_denylist_all_shared_hashes") is not True:
        raise FormalConfigError("all MedQA shared hashes must be training-denylisted")
    if conflict_policy.get("primary_final_frozen") is not False:
        raise FormalConfigError("P2 must not freeze the primary MedQA final subset")
    for source_name, source in config.get("sources", {}).items():
        if not isinstance(source, Mapping):
            raise FormalConfigError(f"source {source_name} must be a mapping")
        if _HEX40.fullmatch(str(source.get("revision", ""))) is None:
            raise FormalConfigError(f"source {source_name} revision must be immutable 40-hex")
        allowed = source.get("exact_file_allowlist")
        if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed):
            raise FormalConfigError(f"source {source_name} needs a unique exact file allowlist")
        for file_spec in source.get("files", ()):
            if file_spec.get("path") not in allowed:
                raise FormalConfigError(f"source {source_name} file is outside exact allowlist")
        validate_source_admission(source_name=source_name, source=source)
    return config


def configured_source_files(config: Mapping[str, Any]) -> list[ConfiguredSourceFile]:
    """Expand the configured files, including exactly 8×3 C-Eval shards."""

    entries: list[ConfiguredSourceFile] = []
    for source_key, source in config["sources"].items():
        if source_key == "ceval":
            template = str(source["exact_path_template"])
            for subject in source["subjects"]:
                for split in source["splits"]:
                    path = template.format(subject=subject, split=split)
                    if path not in source["exact_file_allowlist"]:
                        raise FormalConfigError("expanded C-Eval path is outside exact allowlist")
                    role = {
                        "dev": "ceval_smoke",
                        "val": "general_controller_dev",
                        "test": "general_final_test",
                    }[split]
                    entries.append(
                        ConfiguredSourceFile(
                            source_key=source_key,
                            repository=str(source["repository"]),
                            revision=str(source["revision"]),
                            path=path,
                            file_format="parquet",
                            upstream_split=str(split),
                            adapter="ceval",
                            target_role=role,
                            denylist_only=False,
                            subject=str(subject),
                        )
                    )
            continue
        for file_spec in source.get("files", ()):
            entries.append(
                ConfiguredSourceFile(
                    source_key=str(source_key),
                    repository=str(source["repository"]),
                    revision=str(source["revision"]),
                    path=str(file_spec["path"]),
                    file_format=str(file_spec["format"]),
                    upstream_split=str(file_spec["upstream_split"]),
                    adapter=(str(file_spec["adapter"]) if file_spec.get("adapter") else None),
                    target_role=(str(file_spec["target_role"]) if file_spec.get("target_role") else None),
                    denylist_only=bool(file_spec.get("denylist_only", False)),
                )
            )
    return entries


def validate_source_admission(*, source_name: str, source: Mapping[str, Any]) -> None:
    """Fail closed on ambiguous training licences and mutable representations."""

    if _HEX40.fullmatch(str(source.get("revision", ""))) is None:
        raise FormalConfigError(f"source {source_name} revision must be immutable 40-hex")
    license_name = str(source.get("source_license", "")).strip()
    scope = str(source.get("usage_scope", "")).strip()
    if not license_name or not scope:
        raise FormalConfigError(f"source {source_name} must bind license and usage scope")
    if license_name.casefold() == "unknown":
        local_only = scope == "local_evaluation_only"
        no_redistribution = source.get("redistribution_allowed") is False
        if not (local_only and no_redistribution):
            raise FormalConfigError(
                f"source {source_name} has unknown license outside local-only evaluation"
            )
    allowed = source.get("exact_file_allowlist")
    if not isinstance(allowed, list) or not allowed:
        raise FormalConfigError(f"source {source_name} has no exact file allowlist")


def split_prompt_label(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return physically disjoint prompt and gold-label payloads."""

    row = dict(record)
    label_keys = ("answer", "answer_idx")
    label = {
        key: row[key]
        for key in ("sample_id", "target_role", "content_hash", *label_keys)
        if row.get(key) is not None
    }
    prompt = {
        key: value
        for key, value in row.items()
        if key not in SUPERVISION_KEYS
    }
    return prompt, label


def medqa_option_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invalid_labels = 0
    empty_options = 0
    duplicate_options = 0
    total = 0
    for row in rows:
        total += 1
        candidate = row.get("options") or row.get("choices") or ()
        if isinstance(candidate, Mapping):
            labels = [str(value).upper() for value in candidate]
            values = [str(value).strip() for value in candidate.values()]
        else:
            labels = []
            values = []
            for index, item in enumerate(candidate):
                if isinstance(item, Mapping):
                    labels.append(str(item.get("key") or item.get("label") or chr(65 + index)).upper())
                    values.append(str(item.get("value") or item.get("text") or "").strip())
                else:
                    labels.append(chr(65 + index))
                    values.append(str(item).strip())
        counts[str(len(values))] += 1
        if any(not value for value in values):
            empty_options += 1
        if len(set(values)) != len(values):
            duplicate_options += 1
        label = str(row.get("answer_idx") or row.get("label") or "").upper()
        if label not in labels:
            invalid_labels += 1
    return {
        "total_rows": total,
        "option_count_distribution": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "empty_option_rows": empty_options,
        "duplicate_option_rows": duplicate_options,
        "invalid_label_count": invalid_labels,
        "filtered_to_four_options": False,
    }


def deterministic_group_allocation(
    records: Iterable[Mapping[str, Any]],
    *,
    targets: Mapping[str, int],
    seed: int,
) -> dict[str, str]:
    """Assign whole near-duplicate groups in a stable hash order."""

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[str(record["group_id"])].append(str(record["sample_id"]))
    ordered_groups = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(
            f"{seed}\0medical_o1\0{group_id}".encode("utf-8")
        ).hexdigest(),
    )
    roles = list(targets)
    counts: Counter[str] = Counter()
    role_index = 0
    allocation: dict[str, str] = {}
    for group_id in ordered_groups:
        while role_index < len(roles) and counts[roles[role_index]] >= int(
            targets[roles[role_index]]
        ):
            role_index += 1
        if role_index >= len(roles):
            break
        role = roles[role_index]
        members = sorted(groups[group_id])
        for sample_id in members:
            allocation[sample_id] = role
        counts[role] += len(members)
    return dict(sorted(allocation.items()))


def build_protected_denylist(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build a redacted final-candidate denylist (IDs and hashes only)."""

    denied = []
    for record in records:
        if record.get("target_role") not in FINAL_ROLES_V2:
            continue
        denied.append(
            {
                "sample_id": str(record["sample_id"]),
                "target_role": str(record["target_role"]),
                "content_hash": str(record["content_hash"]),
            }
        )
    return sorted(denied, key=lambda row: row["sample_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(destination: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {"path": str(target), "bytes": len(encoded), "sha256": _sha256(target), "complete": True}


def atomic_jsonl_export(
    destination: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    """Write one role artifact through a temporary file and return its binding."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    count = 0
    supervision_fields = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for original in rows:
            row = dict(original)
            if row.get("target_role") != role:
                raise PermissionError(
                    f"record target_role={row.get('target_role')!r} does not match export role={role!r}"
                )
            if role in PROMPT_ONLY_ROLES_V2:
                for key in SUPERVISION_KEYS:
                    row.pop(key, None)
            supervision_fields += len(set(row) & set(SUPERVISION_KEYS))
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "path": str(target),
        "sha256": _sha256(target),
        "bytes": target.stat().st_size,
        "count": count,
        "supervision_fields": supervision_fields,
        "complete": True,
    }


def validate_formal_manifest(
    value: Mapping[str, Any] | str | Path,
    *,
    require_records: bool = True,
    allow_final: bool = False,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        manifest = dict(value)
    else:
        manifest = json.loads(Path(value).read_text(encoding="utf-8"))
    if manifest.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise ValueError("manifest does not bind Data Protocol v2")
    if manifest.get("source_policy_version") != SOURCE_POLICY_VERSION:
        raise ValueError("manifest does not bind current source policy")
    if manifest.get("conflict_policy_version") != CONFLICT_POLICY_VERSION:
        raise ValueError("manifest does not bind current MedQA conflict policy")
    if manifest.get("schema_version") != SCHEMA_VERSION_V2:
        raise ValueError("manifest does not bind schema v2")
    if manifest.get("build_mode") != "formal" or manifest.get("synthetic_fixture") is not False:
        raise ValueError("manifest is not a non-synthetic formal build")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError("manifest roles must be a non-empty mapping")
    final_roles = set(roles) & set(FINAL_ROLES_V2)
    if final_roles:
        if not allow_final:
            raise PermissionError("final candidate manifest requires final capability")
        if manifest.get("primary_final_frozen") is not True:
            raise PermissionError("primary_final_frozen must be true for final capability")
    if require_records:
        for role, metadata in roles.items():
            files = metadata.get("files") if isinstance(metadata, Mapping) else None
            if not isinstance(files, list) or not files:
                raise ValueError(f"role {role} lacks records files")
            for item in files:
                path = Path(str(item.get("path", "")))
                if not item.get("complete") or not path.is_file():
                    raise ValueError(f"role {role} has incomplete records artifact")
                if path.with_suffix(path.suffix + ".tmp").exists():
                    raise ValueError(f"role {role} still has a partial records artifact")
                if _sha256(path) != item.get("sha256"):
                    raise ValueError(f"role {role} records SHA-256 mismatch")
    return manifest
