"""P8-only 300-step prompt schedule with the single 3 O1 + 1 CMB change."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_data_v2 import _iter_jsonl, _validate_prompt_row, canonical_json_sha256
from src.opd.production_b2_formal_data_v1 import validate_formal_b2_prompt_schedule


P8_SCHEDULE_KIND = "p8_single_variable_b2_prompt_schedule_v1"
P8_SCHEDULE_VERSION = "seed42_3o1_1cmb_300step_sha256_rank_v1"


class P8FormalProtocolError(RuntimeError):
    """The single-variable schedule or identity differs."""


def _rank(seed: int, row: Mapping[str, str]) -> str:
    return hashlib.sha256(
        f"{seed}\0p8-3o1-1cmb-v1\0{row['target_role']}\0{row['sample_id']}\0{row['content_hash']}".encode()
    ).hexdigest()


def build_p8_prompt_schedule(
    authority: Mapping[str, Any],
    *,
    baseline_schedule: Mapping[str, Any],
    seed: int,
    optimizer_steps: int,
    excluded_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    if seed != 42 or optimizer_steps != 300:
        raise P8FormalProtocolError("P8 schedule must freeze 300 steps at seed 42")
    if not (
        authority.get("artifact_kind") == "b2_production_data_authority_v2"
        and authority.get("primary_final_frozen") is True
        and authority.get("final_authorized") is False
    ):
        raise P8FormalProtocolError("P8 authority is not frozen prompt-only v2")
    excluded = set() if excluded_sample_ids is None else set(excluded_sample_ids)
    validate_formal_b2_prompt_schedule(baseline_schedule, authority=authority)
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    baseline_slots = list(baseline_schedule["slots"])
    protected_ids = {str(row["sample_id"]) for row in baseline_slots}
    protected_hashes = {str(row["content_hash"]) for row in baseline_slots}
    available: dict[str, list[dict[str, str]]] = {}
    required_new = {"medical_opd_o1": 600, "medical_opd_cmb": 150}
    for role, count in required_new.items():
        payload = payloads.get(role)
        if not isinstance(payload, Mapping):
            raise P8FormalProtocolError(f"P8 source is absent: {role}")
        ranked = []
        for _line, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            identity = _validate_prompt_row(raw, role=role)
            if (
                identity["sample_id"] not in excluded
                and identity["sample_id"] not in protected_ids
                and identity["content_hash"] not in protected_hashes
            ):
                ranked.append(identity)
        ranked.sort(key=lambda row: (_rank(seed, row), row["sample_id"]))
        if len(ranked) < count:
            raise P8FormalProtocolError(f"P8 source has insufficient unique rows: {role}")
        available[role] = ranked[:count]
    slots: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_hashes: set[str] = set()
    cursor = 0
    for step_index in range(optimizer_steps):
        if step_index < 150:
            old = [row for row in baseline_slots if row["step"] == step_index + 1]
            if [row["source"] for row in old] != [
                "medical_opd_o1", "medical_opd_o1", "medical_opd_cmb", "medical_opd_cmb"
            ]:
                raise P8FormalProtocolError("baseline B2 source order differs")
            batch = [old[0], old[1], available["medical_opd_o1"][step_index], old[2]]
        else:
            extension = step_index - 150
            o1_start = 150 + extension * 3
            batch = [
                *available["medical_opd_o1"][o1_start : o1_start + 3],
                available["medical_opd_cmb"][extension],
            ]
        for slot, row in enumerate(batch, start=1):
            role = str(row.get("target_role", row.get("source")))
            if row["sample_id"] in used_ids or row["content_hash"] in used_hashes:
                raise P8FormalProtocolError("P8 schedule has duplicate identity")
            used_ids.add(str(row["sample_id"]))
            used_hashes.add(str(row["content_hash"]))
            slots.append(
                {
                    "step": step_index + 1,
                    "slot": slot,
                    "source": role,
                    "sample_id": row["sample_id"],
                    "content_hash": row["content_hash"],
                    "payload_sha256": payloads[role]["payload_sha256"],
                    "manifest_sha256": authority["manifest_sha256"],
                    "data_cursor": cursor,
                    "schedule_version": P8_SCHEDULE_VERSION,
                }
            )
            cursor += 1
    schedule = {
        "schema_version": 1,
        "artifact_kind": P8_SCHEDULE_KIND,
        "schedule_version": P8_SCHEDULE_VERSION,
        "manifest_path": authority["manifest_path"],
        "manifest_sha256": authority["manifest_sha256"],
        "seed": 42,
        "optimizer_steps": 300,
        "stage1_optimizer_steps": 120,
        "prompts_per_step": 4,
        "slot_count": 1200,
        "stage1_slot_count": 480,
        "source_counts_per_step": {"medical_opd_o1": 3, "medical_opd_cmb": 1},
        "single_training_semantic_variable": "medical_source_mix",
        "group_size": 1,
        "learning_rate": 1e-5,
        "per_prompt_gradient_clip_norm": 0.25,
        "response_length": 1024,
        "extension_slots_frozen_before_training": True,
        "prompt_only": True,
        "raw_prompt_text_persisted": False,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
        "slots": slots,
    }
    schedule["schedule_sha256"] = canonical_json_sha256(schedule)
    validate_p8_prompt_schedule(schedule, authority=authority)
    return schedule


def validate_p8_prompt_schedule(schedule: Mapping[str, Any], *, authority: Mapping[str, Any]) -> dict[str, Any]:
    exact_fields = {
        "schema_version", "artifact_kind", "schedule_version", "manifest_path",
        "manifest_sha256", "seed", "optimizer_steps", "stage1_optimizer_steps",
        "prompts_per_step", "slot_count", "stage1_slot_count", "source_counts_per_step",
        "single_training_semantic_variable", "group_size", "learning_rate",
        "per_prompt_gradient_clip_norm", "response_length",
        "extension_slots_frozen_before_training", "prompt_only",
        "raw_prompt_text_persisted", "final_access", "controller_access",
        "confirmation_access", "label_access", "slots", "schedule_sha256",
    }
    if set(schedule) != exact_fields:
        raise P8FormalProtocolError("P8 schedule contains missing or unregistered fields")
    unsigned = dict(schedule)
    claimed = unsigned.pop("schedule_sha256")
    if claimed != canonical_json_sha256(unsigned):
        raise P8FormalProtocolError("P8 schedule SHA mismatch")
    if not (
        schedule["schema_version"] == 1
        and schedule["artifact_kind"] == P8_SCHEDULE_KIND
        and schedule["schedule_version"] == P8_SCHEDULE_VERSION
        and schedule["manifest_path"] == authority.get("manifest_path")
        and schedule["manifest_sha256"] == authority.get("manifest_sha256")
        and schedule["seed"] == 42
        and schedule["optimizer_steps"] == 300
        and schedule["stage1_optimizer_steps"] == 120
        and schedule["prompts_per_step"] == 4
        and schedule["slot_count"] == 1200
        and schedule["stage1_slot_count"] == 480
        and schedule["source_counts_per_step"] == {"medical_opd_o1": 3, "medical_opd_cmb": 1}
        and schedule["single_training_semantic_variable"] == "medical_source_mix"
        and schedule["group_size"] == 1
        and float(schedule["learning_rate"]) == 1e-5
        and float(schedule["per_prompt_gradient_clip_norm"]) == 0.25
        and schedule["response_length"] == 1024
        and schedule["extension_slots_frozen_before_training"] is True
        and schedule["prompt_only"] is True
        and schedule["raw_prompt_text_persisted"] is False
        and all(schedule[name] is False for name in ("final_access", "controller_access", "confirmation_access", "label_access"))
    ):
        raise P8FormalProtocolError("P8 schedule contract differs")
    slots = schedule["slots"]
    if not isinstance(slots, list) or len(slots) != 1200:
        raise P8FormalProtocolError("P8 schedule slot count differs")
    if Counter(row.get("source") for row in slots if isinstance(row, Mapping)) != {
        "medical_opd_o1": 900,
        "medical_opd_cmb": 300,
    }:
        raise P8FormalProtocolError("P8 aggregate source counts differ")
    ids: set[str] = set()
    hashes: set[str] = set()
    for step in range(1, 301):
        rows = [row for row in slots if isinstance(row, Mapping) and row.get("step") == step]
        if [row.get("slot") for row in rows] != [1, 2, 3, 4] or [row.get("source") for row in rows] != [
            "medical_opd_o1", "medical_opd_o1", "medical_opd_o1", "medical_opd_cmb"
        ]:
            raise P8FormalProtocolError("P8 per-step 3+1 contract differs")
        for offset, row in enumerate(rows):
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            if not (
                isinstance(sample_id, str) and sample_id not in ids
                and isinstance(content_hash, str) and len(content_hash) == 64 and content_hash not in hashes
                and row.get("data_cursor") == (step - 1) * 4 + offset
                and row.get("manifest_sha256") == authority.get("manifest_sha256")
                and row.get("schedule_version") == P8_SCHEDULE_VERSION
            ):
                raise P8FormalProtocolError("P8 slot identity/cursor differs")
            ids.add(sample_id)
            hashes.add(content_hash)
    return {"passed": True, "schedule_sha256": claimed, "optimizer_steps": 300, "slot_count": 1200}


def resolve_p8_schedule_batch(authority: Mapping[str, Any], schedule: Mapping[str, Any], *, step_index: int) -> list[dict[str, Any]]:
    validate_p8_prompt_schedule(schedule, authority=authority)
    if not 0 <= step_index < 300:
        raise P8FormalProtocolError("P8 step index is outside 300 steps")
    slots = [dict(row) for row in schedule["slots"] if row["step"] == step_index + 1]
    wanted = {(row["sample_id"], row["content_hash"]): row for row in slots}
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    payloads = {item["target_role"]: item for item in authority.get("payloads", []) if isinstance(item, Mapping)}
    for role in ("medical_opd_o1", "medical_opd_cmb"):
        for _line, raw in _iter_jsonl(Path(str(payloads[role]["resolved_path"]))):
            key = (raw.get("sample_id"), raw.get("content_hash"))
            if key in wanted:
                _validate_prompt_row(raw, role=role)
                hydrated[key] = dict(raw)
    if set(hydrated) != set(wanted):
        raise P8FormalProtocolError("P8 provider cannot hydrate every slot")
    return [hydrated[(row["sample_id"], row["content_hash"])] for row in slots]


__all__ = [
    "P8FormalProtocolError", "P8_SCHEDULE_KIND", "P8_SCHEDULE_VERSION",
    "build_p8_prompt_schedule", "resolve_p8_schedule_batch", "validate_p8_prompt_schedule",
]
