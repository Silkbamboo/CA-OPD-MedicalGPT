"""Deterministic, CPU-safe smoke pipeline for Data Protocol v2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .adapters import AdapterContext, adapt_source_row
from .schema import (
    CONTROLLER_ROLES_V2,
    DATA_PROTOCOL_VERSION,
    FINAL_ROLES_V2,
    PROMPT_ONLY_ROLES_V2,
    SUPERVISION_KEYS,
    DataRecordV2,
)
from src.utils.run_meta import git_dirty, git_sha


FORBIDDEN_SUPERVISION_KEYS = frozenset(SUPERVISION_KEYS)
SEPARATED_EVAL_ROLES = CONTROLLER_ROLES_V2 | FINAL_ROLES_V2 | {"ceval_smoke"}


class DataProtocolError(ValueError):
    """Raised when a build would violate Data Protocol v2."""


@dataclass(frozen=True)
class SmokeBuildResult:
    output_root: Path
    manifest_path: Path
    stats_path: Path
    leakage_path: Path
    license_path: Path
    report_path: Path
    manifest: Mapping[str, Any]
    leakage_report: Mapping[str, Any]
    source_input_counts: Mapping[str, int]
    source_accepted_counts: Mapping[str, int]
    tokenizer_audit_status: str
    supervision_fields_in_opd: int


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            payload = _json_bytes(row)
            handle.write(payload)
            digest.update(payload)
            count += 1
    return {"path": path.name, "count": count, "sha256": digest.hexdigest()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects without loading a source file into memory."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DataProtocolError(f"{source}:{line_number} is not a JSON object")
            yield value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataProtocolError(f"{path} must contain a YAML mapping")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _stable_rank(seed: int, *values: str) -> str:
    return hashlib.sha256(
        (str(seed) + "\0" + "\0".join(values)).encode("utf-8")
    ).hexdigest()


def _character_ngrams(value: str, size: int = 3) -> frozenset[str]:
    compact = "".join(value.split())
    if len(compact) <= size:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def find_near_duplicate_candidates(
    records: Sequence[DataRecordV2],
    *,
    threshold: float,
    max_pairwise_records: int,
) -> list[tuple[int, int, float]]:
    """Return conservative character-3gram candidates for a bounded smoke set.

    Formal builds above ``max_pairwise_records`` fail closed and must use a
    separately audited blocking/index implementation rather than quadratic RAM.
    """

    if not 0 <= threshold <= 1:
        raise DataProtocolError("near-duplicate threshold must be in [0, 1]")
    if len(records) > max_pairwise_records:
        raise DataProtocolError(
            "pairwise near-duplicate scan exceeds configured CPU-safe bound"
        )
    grams = [_character_ngrams(record.normalized_question) for record in records]
    candidates: list[tuple[int, int, float]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            union = grams[left] | grams[right]
            score = len(grams[left] & grams[right]) / len(union) if union else 1.0
            if score >= threshold:
                candidates.append((left, right, score))
    return candidates


def _assign_near_duplicate_groups(
    records: Sequence[DataRecordV2],
    *,
    threshold: float,
    max_pairwise_records: int,
) -> tuple[list[DataRecordV2], int]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    candidates = find_near_duplicate_candidates(
        records,
        threshold=threshold,
        max_pairwise_records=max_pairwise_records,
    )
    for left, right, _ in candidates:
        union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members[find(index)].append(index)
    group_ids = {
        root: hashlib.sha256(
            "\n".join(
                sorted(records[index].content_hash for index in indices)
            ).encode("ascii")
        ).hexdigest()
        for root, indices in members.items()
    }
    grouped = [
        replace(record, group_id=group_ids[find(index)])
        for index, record in enumerate(records)
    ]
    return grouped, len(candidates)


def _deduplicate_exact(
    records: Sequence[DataRecordV2], drops: list[dict[str, str]]
) -> list[DataRecordV2]:
    kept: list[DataRecordV2] = []
    owners: dict[str, str] = {}
    for record in sorted(records, key=lambda item: item.sample_id):
        if record.content_hash in owners:
            drops.append(
                {
                    "raw_identity": record.sample_id,
                    "drop_reason": "exact_duplicate_within_source",
                }
            )
            continue
        owners[record.content_hash] = record.sample_id
        kept.append(record)
    return kept


def _assign_medical_o1_roles(
    records: Sequence[DataRecordV2],
    *,
    targets: Mapping[str, int],
    seed: int,
    drops: list[dict[str, str]],
) -> list[DataRecordV2]:
    groups: dict[str, list[DataRecordV2]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    ordered_groups = sorted(
        groups.items(), key=lambda item: _stable_rank(seed, "medical_o1", item[0])
    )
    roles = list(targets)
    counts = Counter()
    assigned: list[DataRecordV2] = []
    role_index = 0
    for _, group in ordered_groups:
        while role_index < len(roles) and counts[roles[role_index]] >= int(
            targets[roles[role_index]]
        ):
            role_index += 1
        if role_index >= len(roles):
            drops.extend(
                {
                    "raw_identity": record.sample_id,
                    "drop_reason": "medical_o1_target_capacity_exhausted",
                }
                for record in group
            )
            continue
        role = roles[role_index]
        assigned.extend(replace(record, target_role=role) for record in group)
        counts[role] += len(group)
    return assigned


def _stratified_take(
    records: Sequence[DataRecordV2],
    *,
    target: int,
    seed: int,
    drops: list[dict[str, str]],
) -> list[DataRecordV2]:
    buckets: dict[str, list[DataRecordV2]] = defaultdict(list)
    for record in records:
        buckets[record.category or "unknown"].append(record)
    for category in buckets:
        buckets[category].sort(
            key=lambda record: _stable_rank(seed, category, record.sample_id)
        )
    selected: list[DataRecordV2] = []
    categories = sorted(buckets)
    while len(selected) < target:
        changed = False
        for category in categories:
            if buckets[category] and len(selected) < target:
                selected.append(buckets[category].pop(0))
                changed = True
        if not changed:
            break
    selected_ids = {record.sample_id for record in selected}
    drops.extend(
        {
            "raw_identity": record.sample_id,
            "drop_reason": "cmb_stratified_target_capacity_exhausted",
        }
        for record in records
        if record.sample_id not in selected_ids
    )
    return selected


def _quota_take(
    records: Sequence[DataRecordV2],
    *,
    quotas: Mapping[str, int],
    seed: int,
    drops: list[dict[str, str]],
) -> list[DataRecordV2]:
    buckets: dict[str, list[DataRecordV2]] = defaultdict(list)
    for record in records:
        buckets[record.category or "unknown"].append(record)
    selected: list[DataRecordV2] = []
    selected_ids: set[str] = set()
    for subsource, quota in quotas.items():
        ranked = sorted(
            buckets.get(subsource, ()),
            key=lambda record: _stable_rank(seed, subsource, record.sample_id),
        )
        for record in ranked[: int(quota)]:
            selected.append(record)
            selected_ids.add(record.sample_id)
    for record in records:
        if record.sample_id not in selected_ids:
            drops.append(
                {
                    "raw_identity": record.sample_id,
                    "drop_reason": "coig_subsource_quota_exhausted",
                }
            )
    return selected


def _prompt_payload(record: DataRecordV2) -> dict[str, Any]:
    payload = {
        "schema_version": record.schema_version,
        "data_protocol_version": record.data_protocol_version,
        "sample_id": record.sample_id,
        "source": record.source,
        "source_revision": record.source_revision,
        "source_license": record.source_license,
        "upstream_split": record.upstream_split,
        "target_role": record.target_role,
        "domain": record.domain,
        "subject": record.subject,
        "category": record.category,
        "question": record.question,
        "options": list(record.options),
        "normalized_question": record.normalized_question,
        "normalized_options": list(record.normalized_options),
        "content_hash": record.content_hash,
        "group_id": record.group_id,
        "token_count_prompt": record.token_count_prompt,
        "quality_flags": list(record.quality_flags),
        "raw_file_sha256": record.raw_file_sha256,
    }
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


def _label_payload(record: DataRecordV2) -> dict[str, Any]:
    payload = {
        "sample_id": record.sample_id,
        "content_hash": record.content_hash,
        "target_role": record.target_role,
        "answer": record.answer,
        "answer_idx": record.answer_idx,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _contains_supervision(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(str(key).casefold() in FORBIDDEN_SUPERVISION_KEYS)
            + _contains_supervision(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_contains_supervision(child) for child in value)
    return 0


def validate_role_isolation(records: Sequence[DataRecordV2]) -> dict[str, Any]:
    """Fail if IDs, exact content or near-duplicate groups cross roles."""

    id_owners: dict[str, str] = {}
    hash_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    final_hashes: set[str] = set()
    non_final_hashes: set[str] = set()
    for record in records:
        if record.sample_id in id_owners:
            raise DataProtocolError(
                f"duplicate sample_id in build: {record.sample_id}"
            )
        id_owners[record.sample_id] = record.target_role
        for label, value, owners in (
            ("content_hash", record.content_hash, hash_owners),
            ("group_id", record.group_id, group_owners),
        ):
            previous = owners.get(value)
            if previous is not None and previous != record.target_role:
                raise DataProtocolError(
                    f"{label} crosses target roles: {previous} and {record.target_role}"
                )
            owners[value] = record.target_role
        if record.target_role in FINAL_ROLES_V2:
            final_hashes.add(record.content_hash)
        else:
            non_final_hashes.add(record.content_hash)
    final_overlap = final_hashes & non_final_hashes
    if final_overlap:
        raise DataProtocolError("final normalized content appears outside final roles")
    return {
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "status": "PASS",
        "record_count": len(records),
        "unique_sample_id_count": len(id_owners),
        "unique_content_hash_count": len(hash_owners),
        "unique_group_id_count": len(group_owners),
        "cross_role_sample_id_overlap_count": 0,
        "cross_role_content_hash_overlap_count": 0,
        "cross_role_group_overlap_count": 0,
        "final_hash_outside_final_count": 0,
    }


def _source_context(
    key: str,
    source: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    raw_sha256: str,
    splits: Mapping[str, Any],
) -> AdapterContext:
    upstream_split = str(row.get(source.get("split_field"), "train"))
    subsource = (
        str(row[source["subsource_field"]])
        if source.get("subsource_field") in row
        else None
    )
    subject = (
        str(row[source["subject_field"]])
        if source.get("subject_field") in row
        else None
    )
    source_license = (
        str(row[source["license_field"]])
        if source.get("license_field") in row
        else str(source["license"])
    )
    if key == "medical_o1":
        target_role = "medical_sft_train"
    elif key == "cmb":
        target_role = str(splits["cmb"]["role_by_split"].get(upstream_split, "medical_opd_cmb"))
    elif key == "medqa_zh":
        target_role = str(splits["medqa_zh"]["role_by_split"].get(upstream_split, "audit_holdout"))
    elif key == "coig":
        target_role = str(splits["coig"]["role"])
    elif key == "ceval":
        target_role = str(splits["ceval"]["role_by_split"].get(upstream_split, "ceval_smoke"))
    else:
        raise DataProtocolError(f"no context builder for source {key}")
    return AdapterContext(
        source_type=str(source["adapter"]),
        source=str(source["source"]),
        source_revision=str(source["revision"]),
        source_license=source_license,
        upstream_split=upstream_split,
        target_role=target_role,
        raw_file_sha256=raw_sha256,
        subsource=subsource,
        subject=subject,
    )


def _config_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def build_smoke_pipeline(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    report_root: str | Path | None = None,
) -> SmokeBuildResult:
    """Build a five-source smoke release without downloading external data."""

    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = _load_yaml(config_file)
    if (
        config.get("data_protocol_version") != DATA_PROTOCOL_VERSION
        or config.get("schema_version") != 2
        or config.get("build_mode") != "smoke"
    ):
        raise DataProtocolError("smoke config must declare Data Protocol v2/schema 2")
    sources_path = _resolve(config_dir, str(config["sources_config"]))
    splits_path = _resolve(config_dir, str(config["splits_config"]))
    filters_path = _resolve(config_dir, str(config["filters_config"]))
    sources_config = _load_yaml(sources_path)
    splits_config = _load_yaml(splits_path)
    filters_config = _load_yaml(filters_path)
    for child in (sources_config, splits_config, filters_config):
        if child.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
            raise DataProtocolError("all data configs must declare Data Protocol v2")
    from .schema import SOURCE_POLICY_VERSION

    for label, child in (("sources", sources_config), ("splits", splits_config)):
        if child.get("source_policy_version") != SOURCE_POLICY_VERSION:
            raise DataProtocolError(f"{label} config has stale source policy version")

    resolved_output = (
        Path(output_root).resolve()
        if output_root is not None
        else _resolve(config_dir, str(config["output_root"]))
    )
    resolved_reports = (
        Path(report_root).resolve()
        if report_root is not None
        else _resolve(config_dir, str(config["report_root"]))
    )
    resolved_manifest = (
        resolved_reports / "manifests"
        if output_root is not None or report_root is not None
        else _resolve(config_dir, str(config["manifest_root"]))
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_reports.mkdir(parents=True, exist_ok=True)
    resolved_manifest.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    expected_rows = int(config["rows_per_source"])
    if not 20 <= expected_rows <= 50:
        raise DataProtocolError("smoke rows_per_source must be between 20 and 50")
    sources = sources_config["sources"]
    source_input_counts: dict[str, int] = {}
    source_accepted_counts: dict[str, int] = {}
    source_manifest: dict[str, Any] = {}
    accepted_by_source: dict[str, list[DataRecordV2]] = {}
    drops: list[dict[str, str]] = []

    for key in config["enabled_sources"]:
        source = sources[key]
        fixture_path = _resolve(sources_path.parent, str(source["fixture_path"]))
        raw_sha256 = _sha256_file(fixture_path)
        accepted: list[DataRecordV2] = []
        input_count = 0
        for row in iter_jsonl(fixture_path):
            input_count += 1
            result = adapt_source_row(
                row,
                _source_context(
                    key,
                    source,
                    row,
                    raw_sha256=raw_sha256,
                    splits=splits_config,
                ),
            )
            if result.record is None:
                drops.append(result.drop_audit_dict())
            else:
                accepted.append(result.record)
        if input_count != expected_rows:
            raise DataProtocolError(
                f"{key} fixture has {input_count} rows; expected {expected_rows}"
            )
        source_input_counts[key] = input_count
        source_accepted_counts[key] = len(accepted)
        accepted_by_source[key] = accepted
        source_manifest[key] = {
            "source": source["source"],
            "revision": source["revision"],
            "declared_license": source["license"],
            "fixture_content": "synthetic_not_upstream_data",
            "fixture_sha256": raw_sha256,
            "input_count": input_count,
            "adapter_accepted_count": len(accepted),
            "observed_record_licenses": sorted(
                {record.source_license for record in accepted}
            ),
        }

    near_config = filters_config["near_duplicate"]
    medical_o1 = _deduplicate_exact(accepted_by_source["medical_o1"], drops)
    medical_o1, near_candidate_count = _assign_near_duplicate_groups(
        medical_o1,
        threshold=float(near_config["threshold"]),
        max_pairwise_records=int(near_config["max_pairwise_records"]),
    )
    records = _assign_medical_o1_roles(
        medical_o1,
        targets=splits_config["medical_o1"]["smoke_targets"],
        seed=seed,
        drops=drops,
    )
    records.extend(
        _stratified_take(
            accepted_by_source["cmb"],
            target=int(splits_config["cmb"]["smoke_target"]),
            seed=seed,
            drops=drops,
        )
    )
    records.extend(accepted_by_source["medqa_zh"])
    records.extend(
        _quota_take(
            accepted_by_source["coig"],
            quotas=splits_config["coig"]["smoke_quotas"],
            seed=seed,
            drops=drops,
        )
    )
    records.extend(accepted_by_source["ceval"])
    records = sorted(records, key=lambda record: (record.target_role, record.sample_id))
    leakage_report = validate_role_isolation(records)
    leakage_report = {
        **leakage_report,
        "near_duplicate_method": near_config["method"],
        "near_duplicate_candidate_count": near_candidate_count,
        "near_duplicate_full_embedding_scan": "not_required_for_smoke",
    }

    role_records: dict[str, list[DataRecordV2]] = defaultdict(list)
    for record in records:
        role_records[record.target_role].append(record)
    output_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    supervision_fields_in_opd = 0
    for role in sorted(role_records):
        role_rows = sorted(role_records[role], key=lambda record: record.sample_id)
        if role in PROMPT_ONLY_ROLES_V2:
            payloads = [_prompt_payload(record) for record in role_rows]
            supervision_fields_in_opd += sum(
                _contains_supervision(payload) for payload in payloads
            )
            output_metadata[role].append(
                _write_jsonl(resolved_output / f"{role}.jsonl", payloads)
            )
        elif role in SEPARATED_EVAL_ROLES:
            output_metadata[role].append(
                _write_jsonl(
                    resolved_output / f"{role}.prompts.jsonl",
                    (_prompt_payload(record) for record in role_rows),
                )
            )
            output_metadata[role].append(
                _write_jsonl(
                    resolved_output / f"{role}.labels.jsonl",
                    (_label_payload(record) for record in role_rows),
                )
            )
        else:
            output_metadata[role].append(
                _write_jsonl(
                    resolved_output / f"{role}.jsonl",
                    (record.to_dict() for record in role_rows),
                )
            )
    if supervision_fields_in_opd:
        raise DataProtocolError("prompt-only output contains supervision fields")

    roles_manifest: dict[str, Any] = {}
    for role in sorted(role_records):
        role_rows = sorted(role_records[role], key=lambda record: record.sample_id)
        roles_manifest[role] = {
            "count": len(role_rows),
            "sample_ids": [record.sample_id for record in role_rows],
            "content_hashes": [record.content_hash for record in role_rows],
            "group_ids": sorted({record.group_id for record in role_rows}),
            "source_distribution": dict(
                sorted(Counter(record.source for record in role_rows).items())
            ),
            "subject_distribution": dict(
                sorted(
                    Counter(record.subject or "unspecified" for record in role_rows).items()
                )
            ),
            "files": output_metadata[role],
        }

    tokenizer = config["tokenizer"]
    tokenizer_status = str(tokenizer["audit_status"])
    repo_root = Path(__file__).resolve().parents[2]
    script_git_sha = git_sha(repo_root)
    if len(script_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in script_git_sha
    ):
        raise DataProtocolError("script_git_sha must be a 40-character Git SHA")
    dirty_worktree = git_dirty(repo_root)
    if dirty_worktree is None:
        raise DataProtocolError("cannot determine Git worktree state")
    script_revision_status = "worktree_uncommitted" if dirty_worktree else "committed"
    manifest = {
        "schema_version": 2,
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "source_policy_version": SOURCE_POLICY_VERSION,
        "build_mode": "smoke",
        "seed": seed,
        "generation_time": config["generation_time"],
        "script_git_sha": script_git_sha,
        "dirty_worktree": dirty_worktree,
        "script_revision_status": script_revision_status,
        "source_config_sha256": _sha256_file(sources_path),
        "split_config_sha256": _sha256_file(splits_path),
        "filter_config_sha256": _sha256_file(filters_path),
        "config_sha256": _config_digest(
            (config_file, sources_path, splits_path, filters_path)
        ),
        "synthetic_fixture": True,
        "fixture_notice": "all question text is synthetic; no upstream dataset was downloaded",
        "tokenizer_audit_status": tokenizer_status,
        "qwen3_enable_thinking": bool(tokenizer["enable_thinking"]),
        "sources": source_manifest,
        "roles": roles_manifest,
        "drop_reason_counts": dict(
            sorted(Counter(item["drop_reason"] for item in drops).items())
        ),
        "dropped_records": sorted(
            drops, key=lambda item: (item["drop_reason"], item["raw_identity"])
        ),
        "overlap_report_sha256": hashlib.sha256(
            _json_bytes(leakage_report)
        ).hexdigest(),
        "supervision_fields_in_opd": supervision_fields_in_opd,
    }
    manifest_path = resolved_manifest / "data_manifest.json"
    _write_json(manifest_path, manifest)

    stats = {
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "source_policy_version": SOURCE_POLICY_VERSION,
        "build_mode": "smoke",
        "source_input_counts": source_input_counts,
        "source_accepted_counts": source_accepted_counts,
        "role_counts": {
            role: len(values) for role, values in sorted(role_records.items())
        },
        "drop_reason_counts": manifest["drop_reason_counts"],
        "token_count_prompt": None,
        "token_count_response": None,
        "tokenizer_audit_status": tokenizer_status,
        "supervision_fields_in_opd": supervision_fields_in_opd,
    }
    stats_path = resolved_reports / "data_stats_smoke_v2.json"
    leakage_path = resolved_reports / "leakage_smoke_v2.json"
    _write_json(stats_path, stats)
    _write_json(leakage_path, leakage_report)

    license_lines = [
        "# Data Protocol v2 smoke license report",
        "",
        "> Smoke rows are synthetic and do not prove upstream content licensing.",
        "",
        "| source key | fixed revision | declared upstream license | smoke status |",
        "|---|---|---|---|",
    ]
    for key, metadata in sorted(source_manifest.items()):
        license_lines.append(
            f"| {key} | `{metadata['revision']}` | {metadata['declared_license']} | "
            "synthetic fixture only |"
        )
    license_lines.extend(
        [
            "",
            "COIG formal remains fail-closed for translated/exam rows until per-row "
            "upstream licenses are audited.",
        ]
    )
    license_path = resolved_reports / "license_smoke_v2.md"
    license_path.write_text("\n".join(license_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Data Protocol v2 smoke report",
        "",
        "- Mode: CPU-safe synthetic fixture smoke; 20 rows per source.",
        f"- Leakage gate: {leakage_report['status']}.",
        f"- OPD supervision fields: {supervision_fields_in_opd}.",
        f"- Tokenizer audit: {tokenizer_status}; no Qwen tokenizer/model downloaded.",
        "- Formal data materialization and final evaluation were not run.",
        "",
        "## Role counts",
        "",
    ]
    report_lines.extend(
        f"- `{role}`: {len(values)}"
        for role, values in sorted(role_records.items())
    )
    report_path = resolved_reports / "data_smoke_v2.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return SmokeBuildResult(
        output_root=resolved_output,
        manifest_path=manifest_path,
        stats_path=stats_path,
        leakage_path=leakage_path,
        license_path=license_path,
        report_path=report_path,
        manifest=manifest,
        leakage_report=leakage_report,
        source_input_counts=source_input_counts,
        source_accepted_counts=source_accepted_counts,
        tokenizer_audit_status=tokenizer_status,
        supervision_fields_in_opd=supervision_fields_in_opd,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_smoke_pipeline(arguments.config)
    print(
        json.dumps(
            {
                "status": result.leakage_report["status"],
                "manifest": str(result.manifest_path),
                "source_input_counts": result.source_input_counts,
                "supervision_fields_in_opd": result.supervision_fields_in_opd,
                "tokenizer_audit_status": result.tokenizer_audit_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
