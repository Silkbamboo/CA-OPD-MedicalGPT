"""Deterministic prompt-identity tapes for P7 Stage-120 actions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.opd.production_b2_data_v2 import _iter_jsonl
from src.opd.stage120_protocol_v4 import (
    P7Stage120Error,
    _has_supervision,
    audit_general_anchor_records_v4,
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ordered(rows: Sequence[Mapping[str, Any]], *, seed: int, label: str) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            f"p7-schedule:{seed}:{label}:{row.get('sample_id')}".encode("utf-8")
        ).hexdigest(),
    )


def _project(row: Mapping[str, Any], *, teacher_route: str) -> dict[str, Any]:
    required = ("sample_id", "group_id", "content_hash", "target_role", "source")
    if any(not row.get(field) for field in required):
        raise P7Stage120Error("P7 schedule prompt identity differs")
    return {
        "sample_id": str(row["sample_id"]),
        "group_id": str(row["group_id"]),
        "content_hash": str(row["content_hash"]),
        "target_role": str(row["target_role"]),
        "source": str(row["source"]),
        "subject": str(row.get("subject", "")),
        "category": str(row.get("category", "")),
        "source_license": str(row.get("source_license", "")),
        "upstream_split": str(row.get("upstream_split", "")),
        "teacher_route": teacher_route,
    }


def _pair(pool: Sequence[Mapping[str, Any]], *, slot: int, variant: int) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if len(pool) < 8:
        raise P7Stage120Error("P7 action pool is too small for three reserves")
    start = (slot * 8 + variant * 2) % len(pool)
    second = (start + 1) % len(pool)
    if start == second:
        raise P7Stage120Error("P7 reserve prompt identity repeated")
    return pool[start], pool[second]


def build_stage120_schedule_v4(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    accepted_steps: int,
) -> dict[str, Any]:
    """Build 120 primary action slots with three deterministic reserves each."""

    if seed != 42 or accepted_steps != 120:
        raise P7Stage120Error("P7 Stage-120 schedule envelope differs")
    try:
        o1 = _ordered(pools["medical_opd_o1"], seed=seed, label="o1")
        cmb = _ordered(pools["medical_opd_cmb"], seed=seed, label="cmb")
        general = list(pools["general_anchors"])
    except KeyError as error:
        raise P7Stage120Error("P7 action pool is absent") from error
    audit_general_anchor_records_v4(general, expected_count=len(general))
    coig = _ordered(
        [row for row in general if row.get("source") == "BAAI/COIG"],
        seed=seed,
        label="coig_leetcode",
    )
    alpaca = _ordered(
        [
            row
            for row in general
            if row.get("source") == "Instruction-Tuning-with-GPT-4/GPT-4-LLM"
        ],
        seed=seed,
        label="gpt4_llm_alpaca_zh",
    )
    slots: list[dict[str, Any]] = []
    for slot in range(accepted_steps):
        medical_batches: list[list[dict[str, Any]]] = []
        general_batches: list[list[dict[str, Any]]] = []
        for variant in range(4):
            o1_pair = _pair(o1, slot=slot, variant=variant)
            cmb_pair = _pair(cmb, slot=slot, variant=variant)
            coig_pair = _pair(coig, slot=slot, variant=variant)
            alpaca_pair = _pair(alpaca, slot=slot, variant=variant)
            medical_batches.append(
                [
                    _project(o1_pair[0], teacher_route="medical"),
                    _project(cmb_pair[0], teacher_route="medical"),
                    _project(o1_pair[1], teacher_route="medical"),
                    _project(cmb_pair[1], teacher_route="medical"),
                ]
            )
            general_batches.append(
                [
                    _project(coig_pair[0], teacher_route="base"),
                    _project(alpaca_pair[0], teacher_route="base"),
                    _project(coig_pair[1], teacher_route="base"),
                    _project(alpaca_pair[1], teacher_route="base"),
                ]
            )
        slots.append(
            {
                "accepted_slot": slot,
                "medical_batches": medical_batches,
                "general_batches": general_batches,
            }
        )
    value = {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_action_schedule_v4",
        "seed": seed,
        "accepted_steps": accepted_steps,
        "primary_attempts_per_slot": 1,
        "reserve_attempts_per_slot": 3,
        "medical_batch_contract": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
        "general_batch_contract": {
            "BAAI/COIG:leetcode:CC-BY-SA-4.0": 2,
            "Instruction-Tuning-with-GPT-4/GPT-4-LLM:gpt4_llm_alpaca_zh:CC-BY-NC-4.0": 2,
        },
        "slots": slots,
        "prompt_text_persisted": False,
        "supervision_fields": 0,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }
    value["schedule_sha256"] = _canonical_sha(value)
    return deepcopy(value)


def validate_stage120_schedule_v4(schedule: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = deepcopy(dict(schedule))
    claimed = unsigned.pop("schedule_sha256", None)
    if not isinstance(claimed, str) or _canonical_sha(unsigned) != claimed:
        raise P7Stage120Error("P7 Stage-120 schedule SHA differs")
    slots = schedule.get("slots")
    if not (
        schedule.get("schema_version") == 4
        and schedule.get("artifact_kind") == "p7_stage120_action_schedule_v4"
        and schedule.get("seed") == 42
        and schedule.get("accepted_steps") == 120
        and schedule.get("primary_attempts_per_slot") == 1
        and schedule.get("reserve_attempts_per_slot") == 3
        and schedule.get("prompt_text_persisted") is False
        and schedule.get("supervision_fields") == 0
        and schedule.get("final_access_count") == 0
        and schedule.get("confirmation_access_count") == 0
        and isinstance(slots, list)
        and len(slots) == 120
    ):
        raise P7Stage120Error("P7 Stage-120 schedule contract differs")
    for accepted_slot, slot in enumerate(slots):
        if not (
            isinstance(slot, Mapping)
            and slot.get("accepted_slot") == accepted_slot
            and isinstance(slot.get("medical_batches"), list)
            and isinstance(slot.get("general_batches"), list)
            and len(slot["medical_batches"]) == 4
            and len(slot["general_batches"]) == 4
        ):
            raise P7Stage120Error("P7 Stage-120 schedule slot differs")
        for action, batches in (
            ("medical", slot["medical_batches"]),
            ("general", slot["general_batches"]),
        ):
            for batch in batches:
                if not isinstance(batch, list) or len(batch) != 4:
                    raise P7Stage120Error("P7 Stage-120 action batch differs")
                roles = [row.get("target_role") for row in batch]
                routes = {row.get("teacher_route") for row in batch}
                if action == "medical" and not (
                    roles.count("medical_opd_o1") == 2
                    and roles.count("medical_opd_cmb") == 2
                    and routes == {"medical"}
                ):
                    raise P7Stage120Error("P7 medical action schedule differs")
                if action == "general" and not (
                    roles == ["general_anchors"] * 4 and routes == {"base"}
                ):
                    raise P7Stage120Error("P7 general action schedule differs")
    return {
        "passed": True,
        "schedule_sha256": claimed,
        "accepted_steps": 120,
        "reserve_attempts_per_slot": 3,
        "final_access_count": 0,
    }


def resolve_stage120_batch_v4(
    authority: Mapping[str, Any],
    schedule: Mapping[str, Any],
    *,
    accepted_slot: int,
    action: str,
    reserve_variant: int,
) -> list[dict[str, Any]]:
    """Hydrate one frozen prompt-only action without opening any label split."""

    validate_stage120_schedule_v4(schedule)
    if not (
        authority.get("artifact_kind") == "b2_production_data_authority_v2"
        and authority.get("primary_final_frozen") is True
        and authority.get("final_authorized") is False
        and authority.get("prompt_only") is True
        and action in {"medical", "general"}
        and 0 <= int(accepted_slot) < 120
        and 0 <= int(reserve_variant) <= 3
    ):
        raise P7Stage120Error("P7 action hydration authority differs")
    key = "medical_batches" if action == "medical" else "general_batches"
    selected = [
        dict(row)
        for row in schedule["slots"][int(accepted_slot)][key][int(reserve_variant)]
    ]
    wanted = {
        (row["sample_id"], row["content_hash"]): row for row in selected
    }
    if len(wanted) != 4:
        raise P7Stage120Error("P7 action batch identity is duplicated")
    roles = {str(row["target_role"]) for row in selected}
    payloads = {
        str(item.get("target_role")): item
        for item in authority.get("payloads", [])
        if isinstance(item, Mapping)
    }
    if not roles.issubset(payloads):
        raise P7Stage120Error("P7 action payload is absent")
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    for role in sorted(roles):
        path = payloads[role].get("resolved_path")
        if not isinstance(path, str):
            raise P7Stage120Error("P7 action payload path differs")
        for _line, raw in _iter_jsonl(Path(path)):
            identity = (raw.get("sample_id"), raw.get("content_hash"))
            projected = wanted.get(identity)
            if projected is None:
                continue
            if not (
                raw.get("target_role") == projected["target_role"]
                and raw.get("source") == projected["source"]
                and str(raw.get("source_license", ""))
                == projected["source_license"]
                and str(raw.get("upstream_split", ""))
                == projected["upstream_split"]
                and not _has_supervision(raw)
            ):
                raise P7Stage120Error("P7 action hydrated prompt identity differs")
            hydrated[identity] = {
                **dict(raw),
                "teacher_route": projected["teacher_route"],
            }
    if set(hydrated) != set(wanted):
        raise P7Stage120Error("P7 action provider cannot hydrate every slot")
    result = [
        hydrated[(row["sample_id"], row["content_hash"])] for row in selected
    ]
    if action == "general":
        audit_general_anchor_records_v4(result, expected_count=4)
    return result


__all__ = [
    "build_stage120_schedule_v4",
    "resolve_stage120_batch_v4",
    "validate_stage120_schedule_v4",
]
