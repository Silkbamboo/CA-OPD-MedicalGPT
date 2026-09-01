"""Deterministic P2 waiver resolution and capability-set freezing.

The module never serialises prompt, option, or answer text into tracked
manifests. Text-bearing processed artifacts remain under ignored persistent
storage. The user waiver is evidence of a policy decision, not human review.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FREEZE_POLICY_VERSION = "mvp-final-controller-freeze-v1"
NEAR_DUPLICATE_WAIVER_VERSION = "time-constrained-interview-mvp-v1"

_ROLE_PRIORITY = {
    "medical_final_test": 4,
    "general_final_test": 4,
    "medical_controller_dev": 3,
    "general_controller_dev": 3,
    "audit_holdout": 2,
    "medical_sft_train": 1,
    "medical_sft_dev": 1,
    "medical_opd_o1": 1,
    "medical_opd_cmb": 1,
    "general_anchors": 1,
}
_TRAINING_ROLES = frozenset(
    {
        "medical_sft_train",
        "medical_sft_dev",
        "medical_opd_o1",
        "medical_opd_cmb",
        "general_anchors",
    }
)


@dataclass(eq=True)
class FreezeResolution:
    candidate_count: int
    unresolved_count: int
    drop_reasons: dict[str, str]
    denylist_hashes: set[str]
    decisions: list[dict[str, Any]]


def _canonical_label(record: Mapping[str, Any]) -> str:
    return str(record.get("answer_idx") or record.get("label") or "").strip().upper()


def _ordered_options(record: Mapping[str, Any]) -> tuple[str, ...]:
    options = record.get("normalized_options") or record.get("options") or ()
    return tuple(str(option) for option in options)


def _is_valid_mcq(record: Mapping[str, Any]) -> bool:
    options = _ordered_options(record)
    label = _canonical_label(record)
    labels = tuple(chr(65 + index) for index in range(len(options)))
    return (
        2 <= len(options) <= 8
        and all(option.strip() for option in options)
        and len(set(options)) == len(options)
        and label in labels
    )


def _pair_reason(left: Mapping[str, Any], right: Mapping[str, Any]) -> str | None:
    if not _is_valid_mcq(left) or not _is_valid_mcq(right):
        return "cross_role_parse_error_quarantine"
    left_options = _ordered_options(left)
    right_options = _ordered_options(right)
    if len(left_options) != len(right_options):
        return "cross_role_option_mismatch_quarantine"
    left_question = str(left.get("normalized_question", ""))
    right_question = str(right.get("normalized_question", ""))
    if left.get("content_hash") == right.get("content_hash") and (
        left_question != right_question or left_options != right_options
    ):
        return "normalization_collision_quarantine"
    # Equal normalized questions with different ordered options are structurally
    # inconsistent versions of the same item; neither side is selected.
    if left_question == right_question and left_options != right_options:
        return "cross_role_option_mismatch_quarantine"
    # Equal ordered options paired with different canonical labels are also
    # unsafe without human semantic review.
    if left_options == right_options and _canonical_label(left) != _canonical_label(right):
        return "cross_role_label_mismatch_quarantine"
    return None


def resolve_cross_role_candidates(
    records_by_id: Mapping[str, Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    high_confidence_threshold: float = 0.92,
) -> FreezeResolution:
    """Resolve high-confidence cross-role candidates independently of input order.

    Every selected candidate contributes both content hashes to the global
    training denylist. Multiplicity and structural anomalies quarantine both
    sides; otherwise the fixed role priority determines the loser.
    """

    selected: list[tuple[str, str, float]] = []
    for candidate in candidate_rows:
        score = float(candidate["similarity"])
        if score < high_confidence_threshold:
            continue
        declared_left_role = str(candidate.get("left_role", ""))
        declared_right_role = str(candidate.get("right_role", ""))
        if declared_left_role and declared_left_role == declared_right_role:
            continue
        left_id = str(candidate["left_sample_id"])
        right_id = str(candidate["right_sample_id"])
        if left_id not in records_by_id or right_id not in records_by_id:
            raise ValueError("near-duplicate candidate references an unknown sample_id")
        left_role = str(records_by_id[left_id].get("target_role", ""))
        right_role = str(records_by_id[right_id].get("target_role", ""))
        if declared_left_role and declared_left_role != left_role:
            raise ValueError("near-duplicate candidate left role differs from record")
        if declared_right_role and declared_right_role != right_role:
            raise ValueError("near-duplicate candidate right role differs from record")
        if left_role == right_role:
            continue
        selected.append((min(left_id, right_id), max(left_id, right_id), score))
    selected = sorted(set(selected), key=lambda item: (item[0], item[1], -item[2]))

    degree: Counter[str] = Counter()
    for left_id, right_id, _ in selected:
        degree[left_id] += 1
        degree[right_id] += 1

    drop_reasons: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    denylist: set[str] = set()
    for left_id, right_id, score in selected:
        left = records_by_id[left_id]
        right = records_by_id[right_id]
        for record in (left, right):
            content_hash = str(record.get("content_hash", ""))
            if len(content_hash) != 64:
                raise ValueError("candidate record lacks a 64-character content_hash")
            denylist.add(content_hash)

        if degree[left_id] > 1 or degree[right_id] > 1:
            reason = "cross_role_duplicate_multiplicity_quarantine"
            drop_reasons[left_id] = reason
            drop_reasons[right_id] = reason
            decision = "quarantine_both"
        else:
            reason = _pair_reason(left, right)
            if reason is not None:
                drop_reasons[left_id] = reason
                drop_reasons[right_id] = reason
                decision = "quarantine_both"
            else:
                left_role = str(left["target_role"])
                right_role = str(right["target_role"])
                left_priority = _ROLE_PRIORITY.get(left_role, 0)
                right_priority = _ROLE_PRIORITY.get(right_role, 0)
                if left_priority == right_priority:
                    reason = "equal_priority_cross_role_quarantine"
                    drop_reasons[left_id] = reason
                    drop_reasons[right_id] = reason
                    decision = "quarantine_both"
                else:
                    loser_id, winner_role, loser_role = (
                        (left_id, right_role, left_role)
                        if left_priority < right_priority
                        else (right_id, left_role, right_role)
                    )
                    drop_reasons[loser_id] = (
                        "near_duplicate_with_final"
                        if "final" in winner_role
                        else "near_duplicate_with_controller"
                        if "controller" in winner_role
                        else "near_duplicate_with_audit_holdout"
                    )
                    decision = f"keep_{winner_role}_drop_{loser_role}"
                    if {winner_role, loser_role} == {
                        "medical_final_test",
                        "medical_controller_dev",
                    }:
                        decision = "keep_final_drop_controller"
        decisions.append(
            {
                "left_sample_id": left_id,
                "right_sample_id": right_id,
                "similarity": score,
                "decision": decision,
                "reason": reason,
            }
        )

    return FreezeResolution(
        candidate_count=len(selected),
        unresolved_count=0,
        drop_reasons=dict(sorted(drop_reasons.items())),
        denylist_hashes=denylist,
        decisions=decisions,
    )


def apply_drop_policy(
    rows: Iterable[Mapping[str, Any]],
    *,
    drop_reasons: Mapping[str, str],
    global_training_denylist: set[str] | frozenset[str],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        role = str(row["target_role"])
        if sample_id in drop_reasons:
            continue
        if role in _TRAINING_ROLES and str(row["content_hash"]) in global_training_denylist:
            continue
        kept.append(dict(row))
    return kept


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def stable_stratified_sample(
    rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    stratum_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Take a stable proportionate sample using largest-remainder allocation."""

    materialized = [dict(row) for row in rows]
    if count < 0 or count > len(materialized):
        raise ValueError("sample count must be within the available row count")
    if count == len(materialized):
        return sorted(materialized, key=lambda row: str(row["sample_id"]))
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        key = tuple(str(row.get(field, "")) for field in stratum_keys)
        strata[key].append(row)
    total = len(materialized)
    quotas: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, tuple[str, ...]]] = []
    for key, members in strata.items():
        exact = count * len(members) / total
        floor = min(len(members), int(exact))
        quotas[key] = floor
        remainders.append((exact - floor, key))
    remaining = count - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        ordered = sorted(
            strata[key], key=lambda row: (_rank(seed, str(row["sample_id"])), str(row["sample_id"]))
        )
        selected.extend(ordered[: quotas[key]])
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def build_redacted_freeze_manifest(
    *,
    role: str,
    rows: Iterable[Mapping[str, Any]],
    seed: int,
    source_license: str,
    usage_scope: str,
    redistribution_allowed: bool,
) -> dict[str, Any]:
    items = [
        {
            "sample_id": str(row["sample_id"]),
            "content_hash": str(row["content_hash"]),
            "upstream_split": str(row["upstream_split"]),
        }
        for row in rows
    ]
    items.sort(key=lambda item: item["sample_id"])
    is_final = "final" in role
    return {
        "policy_version": FREEZE_POLICY_VERSION,
        "role": role,
        "seed": seed,
        "actual_count": len(items),
        "source_license": source_license,
        "usage_scope": usage_scope,
        "redistribution_allowed": redistribution_allowed,
        "primary_final_frozen": is_final,
        "final_authorized": False,
        "items": items,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def _load_capability_rows(source_root: Path, role: str) -> list[dict[str, Any]]:
    prompts = {
        str(row["sample_id"]): row
        for row in _iter_jsonl(source_root / f"{role}.prompts.jsonl")
    }
    labels = {
        str(row["sample_id"]): row
        for row in _iter_jsonl(source_root / f"{role}.labels.jsonl")
    }
    if prompts.keys() != labels.keys():
        raise ValueError(f"{role} prompt and label sample IDs differ")
    rows = []
    for sample_id in sorted(prompts):
        prompt = prompts[sample_id]
        label = labels[sample_id]
        if prompt.get("content_hash") != label.get("content_hash"):
            raise ValueError(f"{role} prompt/label content hash mismatch")
        rows.append({**prompt, **label})
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
            byte_count += len(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "count": count,
        "complete": True,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _split_prompt_label(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = {
        key: value
        for key, value in row.items()
        if key not in {"answer", "answer_idx", "label", "reasoning", "solution", "output", "response", "completion"}
    }
    label = {
        key: row[key]
        for key in ("sample_id", "target_role", "content_hash", "answer", "answer_idx")
        if row.get(key) is not None
    }
    return prompt, label


def freeze_processed_artifacts(
    *,
    source_root: str | Path,
    destination_root: str | Path,
    tracked_manifest_root: str | Path,
    candidate_csv: str | Path,
    seed: int = 42,
    medical_controller_count: int = 300,
    medical_final_count: int = 600,
    general_final_count: int = 300,
    high_confidence_threshold: float = 0.92,
) -> dict[str, Any]:
    """Freeze four capability sets and write only redacted tracked manifests."""

    source = Path(source_root)
    destination = Path(destination_root)
    manifest_root = Path(tracked_manifest_root)
    roles = {
        role: _load_capability_rows(source, role)
        for role in (
            "medical_controller_dev",
            "medical_final_test",
            "general_controller_dev",
            "general_final_test",
        )
    }
    med_records = {
        str(row["sample_id"]): row
        for role in ("medical_controller_dev", "medical_final_test")
        for row in roles[role]
    }
    with Path(candidate_csv).open("r", encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    resolution = resolve_cross_role_candidates(
        med_records,
        candidate_rows,
        high_confidence_threshold=high_confidence_threshold,
    )
    cleaned_medical = {
        role: apply_drop_policy(
            rows,
            drop_reasons=resolution.drop_reasons,
            global_training_denylist=resolution.denylist_hashes,
        )
        for role, rows in roles.items()
        if role.startswith("medical_")
    }
    frozen = {
        "medical_controller_dev": stable_stratified_sample(
            cleaned_medical["medical_controller_dev"],
            count=medical_controller_count,
            seed=seed,
            stratum_keys=("answer_idx",),
        ),
        "general_controller_dev": sorted(
            roles["general_controller_dev"], key=lambda row: str(row["sample_id"])
        ),
        "medical_final_test": stable_stratified_sample(
            cleaned_medical["medical_final_test"],
            count=medical_final_count,
            seed=seed,
            stratum_keys=("answer_idx",),
        ),
        "general_final_test": stable_stratified_sample(
            roles["general_final_test"],
            count=general_final_count,
            seed=seed,
            stratum_keys=("subject",),
        ),
    }

    artifacts: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for role, rows in frozen.items():
        prompt_rows, label_rows = zip(*(_split_prompt_label(row) for row in rows), strict=True)
        artifacts[role] = [
            _write_jsonl_atomic(destination / f"{role}.prompts.jsonl", prompt_rows),
            _write_jsonl_atomic(destination / f"{role}.labels.jsonl", label_rows),
        ]
        is_medqa = role.startswith("medical_")
        manifests[role] = build_redacted_freeze_manifest(
            role=role,
            rows=rows,
            seed=seed,
            source_license="unknown" if is_medqa else "CC BY-NC-SA 4.0",
            usage_scope=(
                "local_evaluation_only" if is_medqa else "noncommercial_research_evaluation"
            ),
            redistribution_allowed=False,
        )
        manifests[role]["artifacts"] = artifacts[role]
        _write_json_atomic(manifest_root / f"{role}.json", manifests[role])

    decisions = []
    for decision in resolution.decisions:
        left = med_records[str(decision["left_sample_id"])]
        right = med_records[str(decision["right_sample_id"])]
        decisions.append(
            {
                **decision,
                "left_content_hash": left["content_hash"],
                "right_content_hash": right["content_hash"],
            }
        )
    resolution_manifest = {
        "policy_version": NEAR_DUPLICATE_WAIVER_VERSION,
        "high_confidence_threshold": high_confidence_threshold,
        "candidate_count": resolution.candidate_count,
        "unresolved_cross_role_candidates": resolution.unresolved_count,
        "manual_audit_rows": 100,
        "human_reviewed": False,
        "manual_audit_waived_by_user": True,
        "waiver_reason": "time_constrained_interview_mvp",
        "cross_role_candidates_conservatively_resolved": True,
        "global_training_denylist_hashes": sorted(resolution.denylist_hashes),
        "drop_reasons": dict(sorted(Counter(resolution.drop_reasons.values()).items())),
        "decisions": decisions,
    }
    _write_json_atomic(manifest_root / "near_duplicate_waiver.json", resolution_manifest)
    result = {
        "policy_version": FREEZE_POLICY_VERSION,
        "manual_audit_waived_by_user": True,
        "human_reviewed": False,
        "unresolved_cross_role_candidates": resolution.unresolved_count,
        "cross_role_candidates_conservatively_resolved": True,
        "primary_final_frozen": True,
        "final_authorized": False,
        "role_counts": {role: len(rows) for role, rows in sorted(frozen.items())},
        "cleaned_candidate_counts": {
            "medical_controller_dev": len(cleaned_medical["medical_controller_dev"]),
            "medical_final_test": len(cleaned_medical["medical_final_test"]),
        },
        "global_training_denylist_count": len(resolution.denylist_hashes),
        "artifacts": artifacts,
    }
    _write_json_atomic(manifest_root / "freeze_summary.json", result)
    return result
