"""Frozen 600-slot formal B2 schedule built before training starts."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_data_v2 import (
    CALIBRATION_ROLES,
    _iter_jsonl,
    _validate_prompt_row,
    canonical_json_sha256,
)
from src.opd.production_b2_formal_v1 import FormalB2Error


FORMAL_SCHEDULE_KIND = "p5_formal_b2_prompt_schedule_v1"
FORMAL_SCHEDULE_VERSION = "seed42_formal_sha256_rank_excluding_p4_8g_v1"


def _rank(seed: int, row: Mapping[str, str]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0formal-b2-v1\0{row['target_role']}\0"
            f"{row['sample_id']}\0{row['content_hash']}"
        ).encode("utf-8")
    ).hexdigest()


def build_formal_b2_prompt_schedule(
    authority: Mapping[str, Any],
    *,
    seed: int,
    optimizer_steps: int,
    excluded_sample_ids: set[str],
) -> dict[str, Any]:
    if seed != 42 or optimizer_steps != 150:
        raise FormalB2Error("formal schedule must freeze all 150 steps at seed 42")
    if not (
        authority.get("artifact_kind") == "b2_production_data_authority_v2"
        and authority.get("primary_final_frozen") is True
        and authority.get("final_authorized") is False
    ):
        raise FormalB2Error("formal schedule authority is not frozen prompt-only v2")
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    selected: dict[str, list[dict[str, str]]] = {}
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    required = optimizer_steps * 2
    for role in CALIBRATION_ROLES:
        payload = payloads.get(role)
        if not isinstance(payload, Mapping):
            raise FormalB2Error(f"formal schedule source is absent: {role}")
        ranked: list[dict[str, str]] = []
        for _line, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            identity = _validate_prompt_row(raw, role=role)
            if identity["sample_id"] not in excluded_sample_ids:
                ranked.append(identity)
        ranked.sort(key=lambda row: (_rank(seed, row), row["sample_id"]))
        if len(ranked) < required:
            raise FormalB2Error(
                f"formal schedule source {role} has fewer than {required} unused prompts"
            )
        selected[role] = ranked[:required]
        for row in selected[role]:
            if row["sample_id"] in all_ids or row["content_hash"] in all_hashes:
                raise FormalB2Error("formal schedule has cross-source duplicate identity")
            all_ids.add(row["sample_id"])
            all_hashes.add(row["content_hash"])
    slots: list[dict[str, Any]] = []
    cursor = 0
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
                        "data_cursor": cursor,
                        "schedule_version": FORMAL_SCHEDULE_VERSION,
                    }
                )
                slot += 1
                cursor += 1
    schedule = {
        "schema_version": 1,
        "artifact_kind": FORMAL_SCHEDULE_KIND,
        "schedule_version": FORMAL_SCHEDULE_VERSION,
        "manifest_path": authority["manifest_path"],
        "manifest_sha256": authority["manifest_sha256"],
        "seed": 42,
        "optimizer_steps": 150,
        "stage1_optimizer_steps": 120,
        "prompts_per_step": 4,
        "slot_count": 600,
        "stage1_slot_count": 480,
        "source_counts_per_step": {
            "medical_opd_o1": 2,
            "medical_opd_cmb": 2,
        },
        "excluded_calibration_sample_id_count": len(excluded_sample_ids),
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
    validate_formal_b2_prompt_schedule(schedule, authority=authority)
    return schedule


def validate_formal_b2_prompt_schedule(
    schedule: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = dict(schedule)
    claimed = unsigned.pop("schedule_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise FormalB2Error("formal schedule SHA mismatch")
    if not (
        schedule.get("schema_version") == 1
        and schedule.get("artifact_kind") == FORMAL_SCHEDULE_KIND
        and schedule.get("schedule_version") == FORMAL_SCHEDULE_VERSION
        and schedule.get("manifest_path") == authority.get("manifest_path")
        and schedule.get("manifest_sha256") == authority.get("manifest_sha256")
        and schedule.get("seed") == 42
        and schedule.get("optimizer_steps") == 150
        and schedule.get("stage1_optimizer_steps") == 120
        and schedule.get("slot_count") == 600
        and schedule.get("stage1_slot_count") == 480
        and schedule.get("extension_slots_frozen_before_training") is True
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
        raise FormalB2Error("formal schedule contract differs")
    slots = schedule.get("slots")
    if not isinstance(slots, list) or len(slots) != 600:
        raise FormalB2Error("formal schedule must contain 600 slots")
    counts = Counter(row.get("source") for row in slots if isinstance(row, Mapping))
    if counts != {"medical_opd_o1": 300, "medical_opd_cmb": 300}:
        raise FormalB2Error("formal schedule aggregate source balance differs")
    ids: set[str] = set()
    hashes: set[str] = set()
    for step in range(1, 151):
        rows = [row for row in slots if isinstance(row, Mapping) and row.get("step") == step]
        if len(rows) != 4 or [row.get("slot") for row in rows] != [1, 2, 3, 4]:
            raise FormalB2Error("formal schedule step width/order differs")
        if [row.get("source") for row in rows] != [
            "medical_opd_o1",
            "medical_opd_o1",
            "medical_opd_cmb",
            "medical_opd_cmb",
        ]:
            raise FormalB2Error("formal schedule per-step source balance differs")
        for offset, row in enumerate(rows):
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            if not (
                isinstance(sample_id, str)
                and sample_id not in ids
                and isinstance(content_hash, str)
                and len(content_hash) == 64
                and content_hash not in hashes
                and row.get("data_cursor") == (step - 1) * 4 + offset
                and row.get("manifest_sha256") == authority.get("manifest_sha256")
                and row.get("schedule_version") == FORMAL_SCHEDULE_VERSION
            ):
                raise FormalB2Error("formal schedule identity/cursor differs")
            ids.add(sample_id)
            hashes.add(content_hash)
    return {
        "passed": True,
        "schedule_sha256": claimed,
        "slot_count": 600,
        "stage1_slot_count": 480,
        "optimizer_steps": 150,
    }


def resolve_formal_b2_schedule_batch(
    authority: Mapping[str, Any], schedule: Mapping[str, Any], *, step_index: int
) -> list[dict[str, Any]]:
    validate_formal_b2_prompt_schedule(schedule, authority=authority)
    if not 0 <= step_index < 150:
        raise FormalB2Error("formal schedule step index is outside 150 steps")
    slots = [dict(row) for row in schedule["slots"] if row["step"] == step_index + 1]
    wanted = {(row["sample_id"], row["content_hash"]): row for row in slots}
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    for role in CALIBRATION_ROLES:
        payload = payloads[role]
        for _line, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            key = (raw.get("sample_id"), raw.get("content_hash"))
            if key in wanted:
                _validate_prompt_row(raw, role=role)
                hydrated[key] = dict(raw)
    if set(hydrated) != set(wanted):
        raise FormalB2Error("formal schedule provider cannot hydrate every slot")
    return [hydrated[(row["sample_id"], row["content_hash"])] for row in slots]


__all__ = [
    "FORMAL_SCHEDULE_KIND",
    "FORMAL_SCHEDULE_VERSION",
    "build_formal_b2_prompt_schedule",
    "resolve_formal_b2_schedule_batch",
    "validate_formal_b2_prompt_schedule",
]
