"""P9-only frozen 2 O1 + 2 CMB dose-extension and resume contracts."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_data_v2 import (
    _iter_jsonl,
    _validate_prompt_row,
    canonical_json_sha256,
)
from src.opd.production_b2_formal_data_v1 import (
    _rank,
    validate_formal_b2_prompt_schedule,
)


P9_SCHEDULE_KIND = "p9_b2_adaptive_dose_prompt_schedule_v1"
P9_SCHEDULE_VERSION = "p7_seed42_2o1_2cmb_append_to300_v1"
ROLES = ("medical_opd_o1", "medical_opd_cmb")
P7_STEP120_IDENTITIES = {
    "package_content_sha256": "c21ca9acec85bb72014ddfc48b5cf9079f680807cbfed348fe5ce1cc619583e1",
    "config_sha256": "130c91b300aab30d6bbbbf7f7893d77fa7c636c361b01a8dd4fc948f92c44835",
    "manifest_sha256": "9f1d096d06b635737e1b90be3b92d6de32fd64b03fbcd97813e42d0a2ee88a99",
    "schedule_sha256": "ddba16637318580a9f31a938da14d7d6d59e49e50046f3f1faebc1ef38e6382c",
    "adapter_sha256": "6e34e1b9b83064016968dd7d1c9f9c4d70ff87058aa3cab2e2be52bee7570408",
}


class P9ProtocolError(RuntimeError):
    """A P9 identity, schedule, decision, or isolation contract differs."""


def _semantic_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": row["step"],
            "slot": row["slot"],
            "source": row["source"],
            "sample_id": row["sample_id"],
            "content_hash": row["content_hash"],
            "data_cursor": row["data_cursor"],
        }
        for row in rows
    ]


def rows_sha256(rows: list[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(list(rows))


def semantic_rows_sha256(rows: list[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(_semantic_rows(rows))


def build_p9_prompt_schedule(
    authority: Mapping[str, Any],
    *,
    baseline_schedule: Mapping[str, Any],
    seed: int,
    excluded_sample_ids: set[str] | None = None,
    reserve_variants_per_step: int = 0,
) -> dict[str, Any]:
    """Preserve the P7 schedule and append the same ranked 2:2 action to step300."""

    if seed != 42 or reserve_variants_per_step not in range(0, 4):
        raise P9ProtocolError("P9 schedule seed differs from 42")
    validate_formal_b2_prompt_schedule(baseline_schedule, authority=authority)
    baseline_slots = [dict(row) for row in baseline_schedule["slots"]]
    excluded = set(excluded_sample_ids or ())
    used_ids = {str(row["sample_id"]) for row in baseline_slots}
    used_hashes = {str(row["content_hash"]) for row in baseline_slots}
    payloads = {
        item["target_role"]: item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    selected: dict[str, list[dict[str, str]]] = {}
    for role in ROLES:
        payload = payloads.get(role)
        if not isinstance(payload, Mapping):
            raise P9ProtocolError(f"P9 source is absent: {role}")
        ranked: list[dict[str, str]] = []
        for _line, raw in _iter_jsonl(Path(str(payload["resolved_path"]))):
            identity = _validate_prompt_row(raw, role=role)
            if (
                identity["sample_id"] not in excluded
                and identity["sample_id"] not in used_ids
                and identity["content_hash"] not in used_hashes
            ):
                ranked.append(identity)
        ranked.sort(key=lambda row: (_rank(seed, row), row["sample_id"]))
        required = 300 + reserve_variants_per_step * 360
        if len(ranked) < required:
            raise P9ProtocolError(f"P9 source has fewer than {required} unused prompts: {role}")
        selected[role] = ranked[:required]

    slots = baseline_slots
    cursor = len(slots)
    for step in range(151, 301):
        offset = (step - 151) * 2
        batch = selected[ROLES[0]][offset : offset + 2] + selected[ROLES[1]][offset : offset + 2]
        for slot, row in enumerate(batch, start=1):
            role = str(row["target_role"])
            slots.append({
                "step": step,
                "slot": slot,
                "source": role,
                "sample_id": row["sample_id"],
                "content_hash": row["content_hash"],
                "payload_sha256": payloads[role]["payload_sha256"],
                "manifest_sha256": authority["manifest_sha256"],
                "data_cursor": cursor,
                "schedule_version": P9_SCHEDULE_VERSION,
            })
            cursor += 1
    reserves: list[dict[str, Any]] = []
    reserve_cursor = 0
    for accepted_step in range(121, 301):
        step_offset = accepted_step - 121
        for variant in range(1, reserve_variants_per_step + 1):
            pool_offset = 300 + (variant - 1) * 360 + step_offset * 2
            batch = selected[ROLES[0]][pool_offset : pool_offset + 2] + selected[ROLES[1]][pool_offset : pool_offset + 2]
            for slot, row in enumerate(batch, start=1):
                role = str(row["target_role"])
                reserves.append({
                    "accepted_step": accepted_step,
                    "reserve_variant": variant,
                    "slot": slot,
                    "source": role,
                    "sample_id": row["sample_id"],
                    "content_hash": row["content_hash"],
                    "payload_sha256": payloads[role]["payload_sha256"],
                    "manifest_sha256": authority["manifest_sha256"],
                    "reserve_cursor": reserve_cursor,
                    "schedule_version": P9_SCHEDULE_VERSION,
                })
                reserve_cursor += 1
    schedule: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": P9_SCHEDULE_KIND,
        "schedule_version": P9_SCHEDULE_VERSION,
        "parent_schedule_sha256": baseline_schedule["schedule_sha256"],
        "manifest_path": authority["manifest_path"],
        "manifest_sha256": authority["manifest_sha256"],
        "seed": 42,
        "optimizer_steps": 300,
        "resume_step": 120,
        "prompts_per_step": 4,
        "slot_count": 1200,
        "reserve_variants_per_step": reserve_variants_per_step,
        "reserve_slot_count": len(reserves),
        "source_counts_per_step": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
        "single_training_semantic_variable": "accepted_optimizer_commit_dose",
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
        "reserves": reserves,
    }
    schedule["schedule_sha256"] = canonical_json_sha256(schedule)
    validate_p9_prompt_schedule(
        schedule, authority=authority, baseline_schedule=baseline_schedule
    )
    return schedule


def validate_p9_prompt_schedule(
    schedule: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    baseline_schedule: Mapping[str, Any],
    restricted_content_hashes: set[str] | None = None,
) -> dict[str, Any]:
    validate_formal_b2_prompt_schedule(baseline_schedule, authority=authority)
    slots = schedule.get("slots")
    if not isinstance(slots, list) or len(slots) != 1200:
        raise P9ProtocolError("P9 schedule slot count differs")
    if slots[:480] != list(baseline_schedule["slots"][:480]):
        raise P9ProtocolError("P9 schedule prefix through step120 differs")
    if slots[:600] != list(baseline_schedule["slots"]):
        raise P9ProtocolError("P9 pre-generated P7 schedule through step150 differs")
    if not (
        schedule.get("schema_version") == 1
        and schedule.get("artifact_kind") == P9_SCHEDULE_KIND
        and schedule.get("schedule_version") == P9_SCHEDULE_VERSION
        and schedule.get("parent_schedule_sha256") == baseline_schedule.get("schedule_sha256")
        and schedule.get("manifest_path") == authority.get("manifest_path")
        and schedule.get("manifest_sha256") == authority.get("manifest_sha256")
        and schedule.get("seed") == 42
        and schedule.get("optimizer_steps") == 300
        and schedule.get("resume_step") == 120
        and schedule.get("prompts_per_step") == 4
        and schedule.get("slot_count") == 1200
        and schedule.get("reserve_variants_per_step") in range(0, 4)
        and schedule.get("reserve_slot_count")
        == int(schedule.get("reserve_variants_per_step")) * 180 * 4
        and schedule.get("source_counts_per_step") == {"medical_opd_o1": 2, "medical_opd_cmb": 2}
        and schedule.get("single_training_semantic_variable") == "accepted_optimizer_commit_dose"
        and schedule.get("group_size") == 1
        and float(schedule.get("learning_rate", -1)) == 1e-5
        and float(schedule.get("per_prompt_gradient_clip_norm", -1)) == 0.25
        and schedule.get("response_length") == 1024
        and schedule.get("extension_slots_frozen_before_training") is True
        and schedule.get("prompt_only") is True
        and schedule.get("raw_prompt_text_persisted") is False
        and all(schedule.get(key) is False for key in ("final_access", "controller_access", "confirmation_access", "label_access"))
    ):
        raise P9ProtocolError("P9 schedule contract differs")
    unsigned = dict(schedule)
    claimed = unsigned.pop("schedule_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise P9ProtocolError("P9 schedule SHA differs")
    if Counter(row.get("source") for row in slots) != {
        "medical_opd_o1": 600,
        "medical_opd_cmb": 600,
    }:
        raise P9ProtocolError("P9 aggregate source balance differs")
    ids: set[str] = set()
    hashes: set[str] = set()
    restricted = set(restricted_content_hashes or ())
    for step in range(1, 301):
        rows = [row for row in slots if row.get("step") == step]
        if [row.get("slot") for row in rows] != [1, 2, 3, 4] or [row.get("source") for row in rows] != [
            "medical_opd_o1", "medical_opd_o1", "medical_opd_cmb", "medical_opd_cmb"
        ]:
            raise P9ProtocolError("P9 per-step 2 O1 plus 2 CMB contract differs")
        for offset, row in enumerate(rows):
            sample_id, content_hash = row.get("sample_id"), row.get("content_hash")
            if not (
                isinstance(sample_id, str) and sample_id not in ids
                and isinstance(content_hash, str) and len(content_hash) == 64 and content_hash not in hashes
                and row.get("data_cursor") == (step - 1) * 4 + offset
                and row.get("manifest_sha256") == authority.get("manifest_sha256")
            ):
                raise P9ProtocolError("P9 schedule identity/cursor differs")
            if step >= 121 and content_hash in restricted:
                raise P9ProtocolError("P9 suffix overlaps restricted Controller/final content")
            ids.add(sample_id); hashes.add(content_hash)
    reserves = schedule.get("reserves")
    variants = int(schedule["reserve_variants_per_step"])
    if not isinstance(reserves, list) or len(reserves) != variants * 180 * 4:
        raise P9ProtocolError("P9 frozen reserve count differs")
    for step in range(121, 301):
        for variant in range(1, variants + 1):
            rows = [
                row for row in reserves
                if row.get("accepted_step") == step
                and row.get("reserve_variant") == variant
            ]
            if [row.get("slot") for row in rows] != [1, 2, 3, 4] or [row.get("source") for row in rows] != [
                "medical_opd_o1", "medical_opd_o1", "medical_opd_cmb", "medical_opd_cmb"
            ]:
                raise P9ProtocolError("P9 reserve action is not frozen 2 O1 plus 2 CMB")
            for row in rows:
                sample_id, content_hash = row.get("sample_id"), row.get("content_hash")
                if not (
                    isinstance(sample_id, str) and sample_id not in ids
                    and isinstance(content_hash, str) and len(content_hash) == 64 and content_hash not in hashes
                    and row.get("manifest_sha256") == authority.get("manifest_sha256")
                ):
                    raise P9ProtocolError("P9 reserve identity differs or duplicates primary")
                if content_hash in restricted:
                    raise P9ProtocolError("P9 reserve overlaps restricted Controller/final content")
                ids.add(sample_id); hashes.add(content_hash)
    return {
        "passed": True,
        "schedule_sha256": claimed,
        "prefix_rows_sha256": rows_sha256(slots[:480]),
        "prefix_semantic_sha256": semantic_rows_sha256(slots[:480]),
        "suffix_rows_sha256": rows_sha256(slots[480:]),
        "suffix_semantic_sha256": semantic_rows_sha256(slots[480:]),
        "slot_count": 1200,
        "reserve_slot_count": len(reserves),
        "optimizer_steps": 300,
        "restricted_overlap_count": 0,
    }


def resolve_p9_schedule_batch(
    authority: Mapping[str, Any],
    schedule: Mapping[str, Any],
    *,
    baseline_schedule: Mapping[str, Any],
    step_index: int,
) -> list[dict[str, Any]]:
    validate_p9_prompt_schedule(
        schedule, authority=authority, baseline_schedule=baseline_schedule
    )
    if not 0 <= step_index < 300:
        raise P9ProtocolError("P9 step is outside the step300 boundary")
    slots = [dict(row) for row in schedule["slots"] if row["step"] == step_index + 1]
    wanted = {(row["sample_id"], row["content_hash"]): row for row in slots}
    payloads = {item["target_role"]: item for item in authority["payloads"] if item["target_role"] in ROLES}
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    for role in ROLES:
        for _line, raw in _iter_jsonl(Path(str(payloads[role]["resolved_path"]))):
            key = (raw.get("sample_id"), raw.get("content_hash"))
            if key in wanted:
                _validate_prompt_row(raw, role=role)
                hydrated[key] = dict(raw)
    if set(hydrated) != set(wanted):
        raise P9ProtocolError("P9 provider cannot hydrate every scheduled prompt")
    return [hydrated[(row["sample_id"], row["content_hash"])] for row in slots]


def validate_p9_resume_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if any(str(key).startswith("p8") for key in manifest):
        raise P9ProtocolError("P8 candidate identity is forbidden in P9 resume")
    if not (
        manifest.get("complete") is True
        and manifest.get("resume_eligible") is True
        and manifest.get("logical_version") == 120
        and manifest.get("optimizer_step") == 120
        and manifest.get("scheduler_step") == 120
        and manifest.get("policy_version") == 120
        and manifest.get("data_cursor") == 480
    ):
        raise P9ProtocolError("P9 resume is not the exact complete P7 step120 state")
    for field, expected in P7_STEP120_IDENTITIES.items():
        if manifest.get(field) != expected:
            raise P9ProtocolError(f"P9 resume {field} differs")
    return {"passed": True, "resume_step": 120, "data_cursor": 480}


__all__ = [
    "P7_STEP120_IDENTITIES", "P9ProtocolError", "P9_SCHEDULE_KIND", "P9_SCHEDULE_VERSION",
    "build_p9_prompt_schedule", "resolve_p9_schedule_batch",
    "rows_sha256", "semantic_rows_sha256", "validate_p9_prompt_schedule",
    "validate_p9_resume_manifest",
]
