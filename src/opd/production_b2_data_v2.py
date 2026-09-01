"""Canonical B2 prompt authority, migration, and frozen schedule gates.

This module is deliberately CPU-safe: importing and calling it never imports
Torch, Transformers, PEFT, CUDA, or a model/session factory.  Dry-run,
preflight, and the production prompt provider share these exact validators.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.schema import content_hash_v2
from src.opd.calibration_data import contains_forbidden_supervision


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANIFEST_PATH = REPO_ROOT / "data/manifests/frozen_v2/opd_manifest.json"
CANONICAL_MANIFEST_RELATIVE = "data/manifests/frozen_v2/opd_manifest.json"
EXPECTED_BUILD_STATUS = "formal_ready_mvp_waived"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_DATA_PROTOCOL = "ca-opd-data-v2"
ALL_ROLES = ("general_anchors", "medical_opd_cmb", "medical_opd_o1")
CALIBRATION_ROLES = ("medical_opd_o1", "medical_opd_cmb")
SCHEDULE_VERSION = "seed42_sha256_rank_first2_per_source_per_step_v1"
SCHEDULE_KIND = "p4_8b_b2_prompt_schedule_v2"
MIGRATION_KIND = "p4_8b_manifest_migration_attestation_v1"
FORBIDDEN_ROLE_MARKERS = ("final", "controller", "confirmation", "label")


class B2DataAuthorityV2Error(RuntimeError):
    """Raised before any model/runtime construction when prompt authority drifts."""


def _fail(message: str) -> None:
    raise B2DataAuthorityV2Error(message)


def canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B2DataAuthorityV2Error(
            f"value is not canonical JSON: {type(error).__name__}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def stream_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise B2DataAuthorityV2Error(
            f"cannot stream artifact SHA: {type(error).__name__}"
        ) from error
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2DataAuthorityV2Error(
            f"{label} is invalid JSON: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _set_sha(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _schema_signature(row: Mapping[str, Any]) -> str:
    def kind(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            nested = sorted({kind(item) for item in value})
            return f"list[{','.join(nested)}]"
        if isinstance(value, Mapping):
            return "object"
        return type(value).__name__

    return canonical_json_sha256({str(key): kind(value) for key, value in row.items()})


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    _fail(f"payload row {line_number} is not an object")
                yield line_number, value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if isinstance(error, B2DataAuthorityV2Error):
            raise
        raise B2DataAuthorityV2Error(
            f"payload stream is invalid: {type(error).__name__}"
        ) from error


def _validate_prompt_row(row: Mapping[str, Any], *, role: str) -> dict[str, str]:
    if contains_forbidden_supervision(row):
        _fail("production payload violates the prompt-only supervision gate")
    observed_role = row.get("target_role")
    if observed_role != role or any(
        marker in str(observed_role).lower() for marker in FORBIDDEN_ROLE_MARKERS
    ):
        _fail("production payload target role differs from the frozen prompt-only role")
    sample_id = row.get("sample_id")
    observed_hash = row.get("content_hash")
    question = row.get("question")
    options = row.get("options", ())
    if not (
        isinstance(sample_id, str)
        and sample_id
        and isinstance(observed_hash, str)
        and len(observed_hash) == 64
        and isinstance(question, str)
        and question.strip()
    ):
        _fail("production payload lacks stable prompt-only identity")
    if options is None:
        options = ()
    if not isinstance(options, (list, tuple)) or any(
        not isinstance(item, str) for item in options
    ):
        _fail("production payload options are invalid")
    if content_hash_v2(question, options) != observed_hash:
        _fail("production payload normalized content hash mismatch")
    source = row.get("source")
    split = row.get("upstream_split")
    if not isinstance(source, str) or not source or not isinstance(split, str) or not split:
        _fail("production payload source/split identity is absent")
    return {
        "sample_id": sample_id,
        "content_hash": observed_hash,
        "source": source,
        "split": split,
        "target_role": role,
    }


def _scan_payload(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{role} payload is absent or a symlink")
    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    records: list[str] = []
    schema_signatures: set[str] = set()
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    count = 0
    for _line_number, row in _iter_jsonl(path):
        identity = _validate_prompt_row(row, role=role)
        if identity["sample_id"] in sample_ids:
            _fail(f"{role} payload has duplicate sample identity")
        if identity["content_hash"] in content_hashes:
            _fail(f"{role} payload has duplicate normalized content hash")
        sample_ids.add(identity["sample_id"])
        content_hashes.add(identity["content_hash"])
        sources[identity["source"]] += 1
        splits[identity["split"]] += 1
        records.append(canonical_json_sha256(identity))
        schema_signatures.add(_schema_signature(row))
        count += 1
    if count < 1:
        _fail(f"{role} payload is empty")
    return {
        "target_role": role,
        "record_count": count,
        "payload_bytes": path.stat().st_size,
        "payload_sha256": stream_sha256(path),
        "records_sha256": _set_sha(records),
        "sample_id_set_sha256": _set_sha(sample_ids),
        "normalized_hash_set_sha256": _set_sha(content_hashes),
        "schema_sha256": _set_sha(schema_signatures),
        "sources": dict(sorted(sources.items())),
        "splits": dict(sorted(splits.items())),
        "prompt_only_field_audit": {
            "record_count": count,
            "supervision_record_count": 0,
            "invalid_identity_count": 0,
        },
        "restricted_access_audit": {
            "final_record_count": 0,
            "controller_record_count": 0,
            "confirmation_record_count": 0,
            "label_record_count": 0,
        },
        "_sample_ids": sample_ids,
        "_content_hashes": content_hashes,
    }


def _manifest_payloads(
    manifest: Mapping[str, Any],
    *,
    payload_overrides: Mapping[str, str | Path] | None = None,
    validate_rows: bool,
) -> list[dict[str, Any]]:
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ALL_ROLES):
        _fail("manifest roles differ from the frozen production contract")
    payloads: list[dict[str, Any]] = []
    for role in ALL_ROLES:
        descriptor = roles.get(role)
        if not isinstance(descriptor, Mapping):
            _fail(f"{role} manifest descriptor is absent")
        files = descriptor.get("files")
        if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], Mapping):
            _fail(f"{role} must bind exactly one payload")
        item = dict(files[0])
        declared_path = Path(str(item.get("path", "")))
        override = None if payload_overrides is None else payload_overrides.get(role)
        actual_path = Path(override) if override is not None else declared_path
        if not actual_path.is_absolute():
            actual_path = REPO_ROOT / actual_path
        expected_sha = item.get("sha256")
        expected_count = item.get("count")
        expected_bytes = item.get("bytes")
        if not (
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and isinstance(expected_count, int)
            and expected_count > 0
            and isinstance(expected_bytes, int)
            and expected_bytes > 0
            and item.get("complete") is True
            and item.get("supervision_fields") == 0
        ):
            _fail(f"{role} payload descriptor is incomplete")
        if actual_path.is_symlink() or not actual_path.is_file():
            _fail(f"{role} payload is absent or a symlink")
        observed_sha = stream_sha256(actual_path)
        if observed_sha != expected_sha or actual_path.stat().st_size != expected_bytes:
            _fail(f"{role} payload SHA or size mismatch")
        result: dict[str, Any] = {
            "target_role": role,
            "declared_path": str(declared_path),
            "resolved_path": str(actual_path.resolve()),
            "payload_sha256": observed_sha,
            "payload_bytes": expected_bytes,
            "record_count": expected_count,
        }
        if validate_rows:
            scan = _scan_payload(actual_path, role=role)
            if scan["record_count"] != expected_count:
                _fail(f"{role} record count differs from its manifest")
            result.update(scan)
        payloads.append(result)
    return payloads


def resolve_b2_data_authority(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    canonical_manifest_path: str | Path = CANONICAL_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate the sole production manifest and all prompt payload semantics."""

    path = Path(manifest_path)
    canonical = Path(canonical_manifest_path)
    if path.is_symlink() or canonical.is_symlink() or path.resolve() != canonical.resolve():
        _fail("B2 prompt manifest path is not the canonical frozen_v2 authority")
    observed_sha = stream_sha256(path)
    if not (
        isinstance(expected_manifest_sha256, str)
        and len(expected_manifest_sha256) == 64
        and observed_sha == expected_manifest_sha256
    ):
        _fail("B2 canonical manifest SHA mismatch")
    manifest = _read_json(path, "B2 canonical manifest")
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        _fail("B2 canonical manifest schema is not v2")
    if manifest.get("data_protocol_version") != EXPECTED_DATA_PROTOCOL:
        _fail("B2 canonical manifest data protocol differs")
    if manifest.get("build_status") != EXPECTED_BUILD_STATUS:
        _fail("B2 canonical manifest status is not production-ready")
    if manifest.get("primary_final_frozen") is not True:
        _fail("B2 canonical manifest is not primary-final frozen")
    if manifest.get("final_authorized") is not False:
        _fail("B2 canonical manifest must keep final authorization false")
    if manifest.get("prompt_label_separated") is not True:
        _fail("B2 canonical manifest does not assert prompt/label separation")
    payloads = _manifest_payloads(manifest, validate_rows=True)
    for payload in payloads:
        payload.pop("_sample_ids", None)
        payload.pop("_content_hashes", None)
    return {
        "schema_version": 2,
        "artifact_kind": "b2_production_data_authority_v2",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": observed_sha,
        "manifest_schema_version": manifest["schema_version"],
        "data_protocol_version": manifest["data_protocol_version"],
        "build_status": manifest["build_status"],
        "primary_final_frozen": True,
        "final_authorized": False,
        "prompt_only": True,
        "payloads": payloads,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }


def build_manifest_migration_attestation(
    old_manifest_path: str | Path,
    new_manifest_path: str | Path,
    *,
    canonical_manifest_path: str | Path = CANONICAL_MANIFEST_PATH,
    old_payload_overrides: Mapping[str, str | Path] | None = None,
    parent_bindings: Mapping[str, str],
    replacement_package_version: str,
) -> dict[str, Any]:
    """Prove content equivalence while explicitly denying authority equivalence."""

    old_path = Path(old_manifest_path)
    new_path = Path(new_manifest_path)
    old = _read_json(old_path, "pre-freeze OPD manifest")
    new = _read_json(new_path, "canonical frozen OPD manifest")
    new_authority = resolve_b2_data_authority(
        new_path,
        expected_manifest_sha256=stream_sha256(new_path),
        canonical_manifest_path=canonical_manifest_path,
    )
    if not (
        old.get("schema_version") == EXPECTED_SCHEMA_VERSION
        and old.get("data_protocol_version") == EXPECTED_DATA_PROTOCOL
        and old.get("build_status") == "built_pending_manual_audit"
        and old.get("primary_final_frozen") is False
        and old.get("prompt_label_separated") is True
    ):
        _fail("pre-freeze manifest metadata differs from the migration source contract")
    old_payloads = _manifest_payloads(
        old,
        payload_overrides=old_payload_overrides,
        validate_rows=True,
    )
    new_payloads = _manifest_payloads(new, validate_rows=True)
    new_by_role = {item["target_role"]: item for item in new_payloads}
    payload_attestations: list[dict[str, Any]] = []
    combined_sample_ids: set[str] = set()
    combined_content_hashes: set[str] = set()
    all_equivalent = True
    for old_item in old_payloads:
        role = old_item["target_role"]
        new_item = new_by_role[role]
        keys = (
            "record_count",
            "payload_bytes",
            "payload_sha256",
            "records_sha256",
            "sample_id_set_sha256",
            "normalized_hash_set_sha256",
            "schema_sha256",
            "sources",
            "splits",
        )
        equivalent = all(old_item[key] == new_item[key] for key in keys)
        all_equivalent = all_equivalent and equivalent
        combined_sample_ids.update(new_item["_sample_ids"])
        combined_content_hashes.update(new_item["_content_hashes"])
        payload_attestations.append(
            {
                "target_role": role,
                "old_declared_path": old_item["declared_path"],
                "old_resolved_path": old_item["resolved_path"],
                "new_declared_path": new_item["declared_path"],
                "new_resolved_path": new_item["resolved_path"],
                "sources": new_item["sources"],
                "splits": new_item["splits"],
                "record_count": new_item["record_count"],
                "records_sha256": new_item["records_sha256"],
                "payload_sha256": new_item["payload_sha256"],
                "payload_bytes": new_item["payload_bytes"],
                "schema_sha256": new_item["schema_sha256"],
                "sample_id_set_sha256": new_item["sample_id_set_sha256"],
                "normalized_hash_set_sha256": new_item[
                    "normalized_hash_set_sha256"
                ],
                "prompt_only_field_audit": new_item["prompt_only_field_audit"],
                "restricted_access_audit": new_item["restricted_access_audit"],
                "content_equivalent": equivalent,
            }
        )
    if not all_equivalent:
        _fail("pre-freeze and frozen_v2 payload content is not equivalent")
    required_parent = {
        "length_selection_sha256",
        "root_index_sha256",
        "old_package_sha256",
    }
    if set(parent_bindings) != required_parent or any(
        not isinstance(value, str) or len(value) != 64
        for value in parent_bindings.values()
    ):
        _fail("migration parent bindings are incomplete")
    sample_set = _set_sha(combined_sample_ids)
    normalized_set = _set_sha(combined_content_hashes)
    attestation = {
        "schema_version": 1,
        "artifact_kind": MIGRATION_KIND,
        "old_manifest": {
            "path": str(old_path.resolve()),
            "sha256": stream_sha256(old_path),
            "schema_version": old.get("schema_version"),
            "data_protocol_version": old.get("data_protocol_version"),
            "build_version": old.get("build_version"),
            "status": old.get("build_status"),
            "primary_final_frozen": old.get("primary_final_frozen"),
            "final_authorized": old.get("final_authorized"),
        },
        "new_manifest": {
            "path": new_authority["manifest_path"],
            "sha256": new_authority["manifest_sha256"],
            "schema_version": new.get("schema_version"),
            "data_protocol_version": new.get("data_protocol_version"),
            "build_version": new.get("build_version"),
            "status": new.get("build_status"),
            "primary_final_frozen": new.get("primary_final_frozen"),
            "final_authorized": new.get("final_authorized"),
        },
        "payloads": payload_attestations,
        "sample_id_set_sha256": sample_set,
        "normalized_hash_set_sha256": normalized_set,
        "prompt_only_field_audit": {
            "passed": True,
            "supervision_record_count": 0,
        },
        "final_controller_label_access_audit": {
            "passed": True,
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
        "content_equivalent": True,
        "authority_equivalent": False,
        "migration_decision": "bind_only_canonical_frozen_v2",
        "parent_bindings": dict(parent_bindings),
        "replacement_package_version": replacement_package_version,
    }
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    return attestation


def _rank_identity(seed: int, sample_id: str, content_hash: str) -> str:
    return hashlib.sha256(
        f"{seed}\0b2-calibration\0{sample_id}\0{content_hash}".encode("utf-8")
    ).hexdigest()


def build_b2_prompt_schedule(
    authority: Mapping[str, Any], *, seed: int, optimizer_steps: int
) -> dict[str, Any]:
    if seed != 42 or optimizer_steps != 20:
        _fail("B2 schedule must remain seed 42 and exactly 20 optimizer steps")
    if not (
        authority.get("artifact_kind") == "b2_production_data_authority_v2"
        and authority.get("primary_final_frozen") is True
        and authority.get("final_authorized") is False
    ):
        _fail("B2 schedule authority is not canonical frozen_v2")
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    selected: dict[str, list[dict[str, str]]] = {}
    for role in CALIBRATION_ROLES:
        payload = payloads.get(role)
        if not isinstance(payload, Mapping):
            _fail(f"schedule source {role} is absent")
        rows: list[dict[str, str]] = []
        for _line_number, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            identity = _validate_prompt_row(raw, role=role)
            rows.append(identity)
        ranked = sorted(
            rows,
            key=lambda row: (
                _rank_identity(seed, row["sample_id"], row["content_hash"]),
                row["sample_id"],
            ),
        )
        required = optimizer_steps * 2
        if len(ranked) < required:
            _fail(f"schedule source {role} has fewer than {required} prompts")
        selected[role] = ranked[:required]
    slots: list[dict[str, Any]] = []
    global_cursor = 0
    for step_index in range(optimizer_steps):
        slot = 1
        for role in CALIBRATION_ROLES:
            for row in selected[role][step_index * 2 : step_index * 2 + 2]:
                slots.append(
                    {
                        "step": step_index + 1,
                        "slot": slot,
                        "source": role,
                        "sample_id": row["sample_id"],
                        "content_hash": row["content_hash"],
                        "payload_sha256": payloads[role]["payload_sha256"],
                        "manifest_sha256": authority["manifest_sha256"],
                        "data_cursor": global_cursor,
                        "schedule_version": SCHEDULE_VERSION,
                    }
                )
                slot += 1
                global_cursor += 1
    schedule = {
        "schema_version": 2,
        "artifact_kind": SCHEDULE_KIND,
        "schedule_version": SCHEDULE_VERSION,
        "manifest_path": authority["manifest_path"],
        "manifest_sha256": authority["manifest_sha256"],
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "prompts_per_step": 4,
        "source_counts_per_step": {
            "medical_opd_o1": 2,
            "medical_opd_cmb": 2,
        },
        "slot_count": len(slots),
        "prompt_only": True,
        "raw_prompt_text_persisted": False,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
        "slots": slots,
    }
    schedule["schedule_sha256"] = canonical_json_sha256(schedule)
    validate_b2_prompt_schedule(schedule, authority=authority)
    return schedule


def validate_b2_prompt_schedule(
    schedule: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(schedule)
    claimed_sha = value.pop("schedule_sha256", None)
    if claimed_sha != canonical_json_sha256(value):
        _fail("B2 prompt schedule SHA mismatch")
    if not (
        schedule.get("schema_version") == 2
        and schedule.get("artifact_kind") == SCHEDULE_KIND
        and schedule.get("schedule_version") == SCHEDULE_VERSION
        and schedule.get("manifest_path") == authority.get("manifest_path")
        and schedule.get("manifest_sha256") == authority.get("manifest_sha256")
        and schedule.get("seed") == 42
        and schedule.get("optimizer_steps") == 20
        and schedule.get("prompts_per_step") == 4
        and schedule.get("slot_count") == 80
        and schedule.get("prompt_only") is True
        and schedule.get("raw_prompt_text_persisted") is False
        and all(
            schedule.get(field) is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
    ):
        _fail("B2 prompt schedule contract differs")
    slots = schedule.get("slots")
    if not isinstance(slots, list) or len(slots) != 80:
        _fail("B2 prompt schedule does not contain exactly 80 slots")
    allowed_fields = {
        "step",
        "slot",
        "source",
        "sample_id",
        "content_hash",
        "payload_sha256",
        "manifest_sha256",
        "data_cursor",
        "schedule_version",
    }
    payload_shas = {
        item["target_role"]: item["payload_sha256"]
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    identities: set[tuple[str, str]] = set()
    for step in range(1, 21):
        expected_cursor = (step - 1) * 4
        step_slots = [item for item in slots if isinstance(item, Mapping) and item.get("step") == step]
        if len(step_slots) != 4:
            _fail("B2 prompt schedule step width differs")
        if [item.get("slot") for item in step_slots] != [1, 2, 3, 4]:
            _fail("B2 prompt schedule slot order differs")
        if [item.get("source") for item in step_slots] != [
            "medical_opd_o1",
            "medical_opd_o1",
            "medical_opd_cmb",
            "medical_opd_cmb",
        ]:
            _fail("B2 prompt schedule source ratio differs")
        for offset, item in enumerate(step_slots):
            if set(item) != allowed_fields:
                _fail("B2 prompt schedule contains raw or unknown fields")
            role = item["source"]
            identity = (str(item["sample_id"]), str(item["content_hash"]))
            if identity in identities:
                _fail("B2 prompt schedule reuses a prompt identity")
            identities.add(identity)
            if not (
                isinstance(item["sample_id"], str)
                and item["sample_id"]
                and isinstance(item["content_hash"], str)
                and len(item["content_hash"]) == 64
                and item["payload_sha256"] == payload_shas.get(role)
                and item["manifest_sha256"] == authority.get("manifest_sha256")
                and item["data_cursor"] == expected_cursor + offset
                and item["schedule_version"] == SCHEDULE_VERSION
            ):
                _fail("B2 prompt schedule identity/binding differs")
    return {
        "schedule_sha256": claimed_sha,
        "slot_count": 80,
        "optimizer_steps": 20,
        "manifest_sha256": authority["manifest_sha256"],
        "deterministic_rebuild": True,
    }


def resolve_b2_schedule_batch(
    authority: Mapping[str, Any],
    schedule: Mapping[str, Any],
    *,
    step_index: int,
) -> list[dict[str, Any]]:
    """Hydrate one frozen schedule window from the already-authorized payloads."""

    validate_b2_prompt_schedule(schedule, authority=authority)
    if not 0 <= step_index < 20:
        _fail("B2 schedule step index is outside the 20-step envelope")
    slots = [
        dict(item)
        for item in schedule["slots"]
        if item.get("step") == step_index + 1
    ]
    wanted = {(item["sample_id"], item["content_hash"]): item for item in slots}
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    for role in CALIBRATION_ROLES:
        payload = payloads.get(role)
        if not isinstance(payload, Mapping):
            _fail("B2 provider cannot resolve a schedule payload")
        for _line_number, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            sample_id = raw.get("sample_id")
            content_hash = raw.get("content_hash")
            key = (sample_id, content_hash)
            if key not in wanted:
                continue
            _validate_prompt_row(raw, role=role)
            hydrated[key] = dict(raw)
    if set(hydrated) != set(wanted):
        _fail("B2 provider cannot hydrate every frozen schedule identity")
    return [hydrated[(item["sample_id"], item["content_hash"])] for item in slots]


__all__ = [
    "ALL_ROLES",
    "B2DataAuthorityV2Error",
    "CALIBRATION_ROLES",
    "CANONICAL_MANIFEST_PATH",
    "CANONICAL_MANIFEST_RELATIVE",
    "EXPECTED_BUILD_STATUS",
    "SCHEDULE_VERSION",
    "build_b2_prompt_schedule",
    "build_manifest_migration_attestation",
    "canonical_json_sha256",
    "resolve_b2_data_authority",
    "resolve_b2_schedule_batch",
    "stream_sha256",
    "validate_b2_prompt_schedule",
]
