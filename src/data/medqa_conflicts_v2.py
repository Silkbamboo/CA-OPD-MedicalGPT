"""Deterministic, privacy-safe MedQA validation/test conflict policy.

Raw question/option text is stored only in the ignored SQLite evidence file.
The returned public report contains IDs, hashes, counts and consistency booleans.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from src.data.schema import DataRecordV2


CONFLICT_POLICY_VERSION = "medqa-final-precedence-v1"
_SPLIT_ROLE = {
    "validation": "medical_controller_dev",
    "test": "medical_final_test",
}
_ANOMALY_REASON = {
    "duplicate_multiplicity": "ambiguous_cross_split_quarantine",
    "label_mismatch": "cross_split_label_mismatch_quarantine",
    "option_mismatch": "cross_split_option_mismatch_quarantine",
    "normalization_collision": "normalization_collision_quarantine",
    "parse_error": "cross_split_parse_error_quarantine",
    "ambiguous": "ambiguous_cross_split_quarantine",
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_sequence(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConflictMember:
    sample_id: str
    split: str
    content_hash: str
    normalized_question_sha256: str
    normalized_options_sha256: str
    raw_question_sha256: str
    raw_options_sha256: str
    option_count: int
    canonical_label: str
    parse_ok: bool


@dataclass(frozen=True)
class ConflictDecision:
    sample_id: str
    upstream_split: str
    target_role: str
    content_hash: str
    action: str
    drop_reason: str | None
    conflict_class: str | None
    counterpart_content_hash: str | None


def member_from_record(record: DataRecordV2) -> ConflictMember:
    """Convert an adapted MedQA record to an audit member without losing order."""

    if record.upstream_split not in _SPLIT_ROLE:
        raise ValueError("MedQA conflict audit only accepts validation/test")
    if record.target_role != _SPLIT_ROLE[record.upstream_split]:
        raise ValueError("MedQA split/role provenance mismatch")
    label = str(record.answer_idx or "").strip().upper()
    valid_labels = tuple(chr(65 + index) for index in range(len(record.options)))
    parse_ok = bool(label and label in valid_labels and len(record.options) >= 2)
    return ConflictMember(
        sample_id=record.sample_id,
        split=record.upstream_split,
        content_hash=record.content_hash,
        normalized_question_sha256=_sha_text(record.normalized_question),
        normalized_options_sha256=_sha_sequence(record.normalized_options),
        raw_question_sha256=_sha_text(record.question),
        raw_options_sha256=_sha_sequence(record.options),
        option_count=len(record.options),
        canonical_label=label,
        parse_ok=parse_ok,
    )


def classify_conflict_group(
    validation: Sequence[ConflictMember], test: Sequence[ConflictMember]
) -> str:
    """Classify one shared hash; anomaly classes always fail closed."""

    if not validation or not test:
        raise ValueError("cross-split conflict must contain both splits")
    members = tuple(validation) + tuple(test)
    if any(not member.parse_ok for member in members):
        return "parse_error"

    validation_labels = {member.canonical_label for member in validation}
    test_labels = {member.canonical_label for member in test}
    if len(validation_labels) != 1 or len(test_labels) != 1:
        return "ambiguous"
    if validation_labels != test_labels:
        return "label_mismatch"

    if len({member.normalized_question_sha256 for member in members}) != 1:
        return "normalization_collision"
    if (
        len({member.normalized_options_sha256 for member in members}) != 1
        or len({member.option_count for member in members}) != 1
    ):
        return "option_mismatch"
    if (
        len({member.raw_question_sha256 for member in members}) != 1
        or len({member.raw_options_sha256 for member in members}) != 1
    ):
        return "normalization_collision"
    if len(validation) != 1 or len(test) != 1:
        return "duplicate_multiplicity"
    return "exact_consistent"


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE members (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            normalized_question_sha256 TEXT NOT NULL,
            normalized_options_sha256 TEXT NOT NULL,
            raw_question_sha256 TEXT NOT NULL,
            raw_options_sha256 TEXT NOT NULL,
            option_count INTEGER NOT NULL,
            canonical_label TEXT NOT NULL,
            parse_ok INTEGER NOT NULL
        );
        CREATE INDEX members_hash_split ON members(content_hash, split, sample_id);
        CREATE TABLE decisions (
            sample_id TEXT NOT NULL,
            upstream_split TEXT NOT NULL,
            target_role TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            action TEXT NOT NULL,
            drop_reason TEXT,
            conflict_class TEXT,
            counterpart_content_hash TEXT,
            PRIMARY KEY(sample_id, upstream_split, content_hash)
        );
        CREATE INDEX decisions_hash ON decisions(content_hash, action);
        CREATE TABLE training_denylist (
            content_hash TEXT PRIMARY KEY,
            reason TEXT NOT NULL
        );
        """
    )


def _insert_member(connection: sqlite3.Connection, member: ConflictMember) -> None:
    connection.execute(
        """
        INSERT INTO members(
            split, sample_id, content_hash, normalized_question_sha256,
            normalized_options_sha256, raw_question_sha256,
            raw_options_sha256, option_count, canonical_label, parse_ok
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            member.split,
            member.sample_id,
            member.content_hash,
            member.normalized_question_sha256,
            member.normalized_options_sha256,
            member.raw_question_sha256,
            member.raw_options_sha256,
            member.option_count,
            member.canonical_label,
            int(member.parse_ok),
        ),
    )


def _row_to_member(row: sqlite3.Row) -> ConflictMember:
    return ConflictMember(
        sample_id=str(row["sample_id"]),
        split=str(row["split"]),
        content_hash=str(row["content_hash"]),
        normalized_question_sha256=str(row["normalized_question_sha256"]),
        normalized_options_sha256=str(row["normalized_options_sha256"]),
        raw_question_sha256=str(row["raw_question_sha256"]),
        raw_options_sha256=str(row["raw_options_sha256"]),
        option_count=int(row["option_count"]),
        canonical_label=str(row["canonical_label"]),
        parse_ok=bool(row["parse_ok"]),
    )


def _insert_decision(
    connection: sqlite3.Connection,
    member: ConflictMember,
    *,
    action: str,
    drop_reason: str | None,
    conflict_class: str | None,
    counterpart_content_hash: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            member.sample_id,
            member.split,
            _SPLIT_ROLE[member.split],
            member.content_hash,
            action,
            drop_reason,
            conflict_class,
            counterpart_content_hash,
        ),
    )


def _consistency_flags(
    validation: Sequence[ConflictMember], test: Sequence[ConflictMember]
) -> dict[str, bool]:
    members = tuple(validation) + tuple(test)
    return {
        "normalized_question_equal": len(
            {member.normalized_question_sha256 for member in members}
        )
        == 1,
        "normalized_options_equal": len(
            {member.normalized_options_sha256 for member in members}
        )
        == 1,
        "raw_option_count_equal": len({member.option_count for member in members})
        == 1,
        "raw_option_order_equal": len(
            {member.raw_options_sha256 for member in members}
        )
        == 1,
        "canonical_label_equal": len(
            {member.canonical_label for member in validation}
        )
        == 1
        and len({member.canonical_label for member in test}) == 1
        and {member.canonical_label for member in validation}
        == {member.canonical_label for member in test},
        "parse_ok": all(member.parse_ok for member in members),
        "within_side_duplicate": len(validation) > 1 or len(test) > 1,
    }


def build_medqa_conflict_audit(
    validation_records: Iterable[DataRecordV2],
    test_records: Iterable[DataRecordV2],
    *,
    sqlite_path: str | Path,
    config_sha256: str,
) -> dict[str, object]:
    """Build the B+ decision index and return a deterministic redacted report."""

    if len(config_sha256) != 64:
        raise ValueError("config_sha256 must be a SHA-256")
    destination = Path(sqlite_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_suffix(destination.suffix + ".building")
    connection = sqlite3.connect(building)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        "DROP TABLE IF EXISTS members; DROP TABLE IF EXISTS decisions; "
        "DROP TABLE IF EXISTS training_denylist;"
    )
    _create_schema(connection)

    revisions: set[str] = set()
    licenses: set[str] = set()

    for split, records in (
        ("validation", validation_records),
        ("test", test_records),
    ):
        for record in records:
            if record.source != "bigbio/med_qa":
                raise ValueError("unexpected MedQA source provenance")
            revisions.add(record.source_revision)
            licenses.add(record.source_license)
            member = member_from_record(record)
            if member.split != split:
                raise ValueError("record appeared in the wrong MedQA input stream")
            _insert_member(connection, member)
        connection.commit()
    if len(revisions) != 1 or len(licenses) != 1:
        connection.close()
        raise ValueError("MedQA audit requires one immutable revision/license state")
    sample_id_collisions = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT sample_id FROM members GROUP BY sample_id
                HAVING COUNT(DISTINCT split || ':' || content_hash) > 1
            )
            """
        ).fetchone()[0]
    )
    if sample_id_collisions:
        connection.close()
        raise ValueError("MedQA sample_id maps to multiple split/content identities")

    raw_counts = {
        split: int(
            connection.execute(
                "SELECT COUNT(*) FROM members WHERE split=?", (split,)
            ).fetchone()[0]
        )
        for split in ("validation", "test")
    }
    within_duplicate_rows = {
        split: int(
            connection.execute(
                """
                SELECT COALESCE(SUM(row_count - 1), 0) FROM (
                    SELECT COUNT(*) AS row_count FROM members
                    WHERE split=? GROUP BY content_hash HAVING row_count > 1
                )
                """,
                (split,),
            ).fetchone()[0]
        )
        for split in ("validation", "test")
    }
    shared_hashes = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT content_hash FROM members GROUP BY content_hash
            HAVING COUNT(DISTINCT split)=2 ORDER BY content_hash
            """
        )
    ]
    groups: list[dict[str, object]] = []
    classes: Counter[str] = Counter()

    for content_hash in shared_hashes:
        rows = connection.execute(
            "SELECT * FROM members WHERE content_hash=? ORDER BY split, sample_id",
            (content_hash,),
        ).fetchall()
        validation = tuple(
            _row_to_member(row) for row in rows if row["split"] == "validation"
        )
        test = tuple(_row_to_member(row) for row in rows if row["split"] == "test")
        conflict_class = classify_conflict_group(validation, test)
        classes[conflict_class] += 1
        connection.execute(
            "INSERT INTO training_denylist VALUES (?, ?)",
            (content_hash, "medqa_validation_test_overlap"),
        )
        if conflict_class == "exact_consistent":
            for member in validation:
                _insert_decision(
                    connection,
                    member,
                    action="drop",
                    drop_reason="overlap_with_final_test",
                    conflict_class=conflict_class,
                    counterpart_content_hash=content_hash,
                )
            for member in test:
                _insert_decision(
                    connection,
                    member,
                    action="keep",
                    drop_reason=None,
                    conflict_class=conflict_class,
                    counterpart_content_hash=content_hash,
                )
            action = "keep_test_drop_validation"
        else:
            reason = _ANOMALY_REASON[conflict_class]
            for member in (*validation, *test):
                _insert_decision(
                    connection,
                    member,
                    action="quarantine",
                    drop_reason=reason,
                    conflict_class=conflict_class,
                    counterpart_content_hash=content_hash,
                )
            action = "quarantine_both_sides"
        groups.append(
            {
                "content_hash": content_hash,
                "validation_sample_ids": sorted(
                    member.sample_id for member in validation
                ),
                "test_sample_ids": sorted(member.sample_id for member in test),
                "validation_record_count": len(validation),
                "test_record_count": len(test),
                **_consistency_flags(validation, test),
                "conflict_class": conflict_class,
                "action": action,
                "training_denylisted": True,
            }
        )

    shared_set = set(shared_hashes)
    within_split_drops = {"validation": 0, "test": 0}
    for row in connection.execute(
        """
        SELECT content_hash, split FROM members
        GROUP BY content_hash, split ORDER BY content_hash, split
        """
    ):
        content_hash, split = str(row[0]), str(row[1])
        if content_hash in shared_set:
            continue
        members = [
            _row_to_member(member)
            for member in connection.execute(
                "SELECT * FROM members WHERE content_hash=? AND split=? ORDER BY sample_id",
                (content_hash, split),
            )
        ]
        representative = members[0]
        _insert_decision(
            connection,
            representative,
            action="keep",
            drop_reason=None,
            conflict_class=None,
        )
        for duplicate in members[1:]:
            _insert_decision(
                connection,
                duplicate,
                action="drop",
                drop_reason="within_split_exact_duplicate",
                conflict_class="duplicate_multiplicity",
            )
            within_split_drops[split] += 1

    connection.commit()
    unique_decision_counts = {
        (str(split), str(action)): int(count)
        for split, action, count in connection.execute(
            "SELECT upstream_split, action, COUNT(*) FROM decisions "
            "GROUP BY upstream_split, action"
        )
    }
    raw_action_counts = {
        (str(split), str(action)): int(count)
        for split, action, count in connection.execute(
            """
            SELECT m.split, d.action, COUNT(*)
            FROM members AS m JOIN decisions AS d
              ON m.sample_id=d.sample_id
             AND m.split=d.upstream_split
             AND m.content_hash=d.content_hash
            GROUP BY m.split, d.action
            """
        )
    }
    kept_hashes = {
        split: {
            str(row[0])
            for row in connection.execute(
                "SELECT content_hash FROM decisions WHERE upstream_split=? AND action='keep'",
                (split,),
            )
        }
        for split in ("validation", "test")
    }
    report: dict[str, object] = {
        "conflict_policy_version": CONFLICT_POLICY_VERSION,
        "config_sha256": config_sha256,
        "source": "bigbio/med_qa",
        "source_revision": next(iter(revisions)),
        "representation": "med_qa_zh_source",
        "source_license": "unknown",
        "usage_scope": "local_evaluation_only",
        "redistribution_allowed": False,
        "primary_final_frozen": False,
        "validation_original_count": raw_counts["validation"],
        "test_original_count": raw_counts["test"],
        "shared_hash_count": len(shared_hashes),
        "conflict_class_counts": dict(sorted(classes.items())),
        "within_split_duplicate_rows": within_duplicate_rows,
        "within_split_duplicate_drops": within_split_drops,
        "validation_removed_records": raw_counts["validation"]
        - unique_decision_counts.get(("validation", "keep"), 0),
        "test_retained_records": unique_decision_counts.get(("test", "keep"), 0),
        "consistent_test_records_retained": sum(
            int(group["test_record_count"])
            for group in groups
            if group["conflict_class"] == "exact_consistent"
        ),
        "both_sides_quarantined_records": sum(
            raw_action_counts.get((split, "quarantine"), 0)
            for split in ("validation", "test")
        ),
        "cleaned_controller_candidate_count": unique_decision_counts.get(
            ("validation", "keep"), 0
        ),
        "cleaned_final_candidate_count": unique_decision_counts.get(("test", "keep"), 0),
        "controller_final_exact_overlap": len(
            kept_hashes["validation"] & kept_hashes["test"]
        ),
        "training_denylist_hash_count": int(
            connection.execute("SELECT COUNT(*) FROM training_denylist").fetchone()[0]
        ),
        "groups": groups,
    }
    if connection.execute("SELECT 1 FROM members LIMIT 1").fetchone() is None:
        connection.close()
        raise ValueError("MedQA conflict audit received no records")
    report["report_payload_sha256"] = _payload_sha256(report)
    connection.close()
    os.replace(building, destination)
    return report


def load_conflict_decisions(
    sqlite_path: str | Path,
) -> dict[str, ConflictDecision]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        return {
            str(row["sample_id"]): ConflictDecision(
                sample_id=str(row["sample_id"]),
                upstream_split=str(row["upstream_split"]),
                target_role=str(row["target_role"]),
                content_hash=str(row["content_hash"]),
                action=str(row["action"]),
                drop_reason=(
                    str(row["drop_reason"]) if row["drop_reason"] is not None else None
                ),
                conflict_class=(
                    str(row["conflict_class"])
                    if row["conflict_class"] is not None
                    else None
                ),
                counterpart_content_hash=(
                    str(row["counterpart_content_hash"])
                    if row["counterpart_content_hash"] is not None
                    else None
                ),
            )
            for row in connection.execute("SELECT * FROM decisions ORDER BY sample_id")
        }
    finally:
        connection.close()


def load_training_denylist(sqlite_path: str | Path) -> set[str]:
    connection = sqlite3.connect(sqlite_path)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT content_hash FROM training_denylist ORDER BY content_hash"
            )
        }
    finally:
        connection.close()


class ConflictDecisionIndex:
    """Read decisions one row at a time during the streaming normalize phase."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.path = Path(sqlite_path)
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "ConflictDecisionIndex":
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.connection is not None
        self.connection.close()
        self.connection = None

    def decision(self, sample_id: str) -> ConflictDecision:
        if self.connection is None:
            raise RuntimeError("conflict decision index is not open")
        row = self.connection.execute(
            "SELECT * FROM decisions WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return ConflictDecision(
            sample_id=str(row["sample_id"]),
            upstream_split=str(row["upstream_split"]),
            target_role=str(row["target_role"]),
            content_hash=str(row["content_hash"]),
            action=str(row["action"]),
            drop_reason=(
                str(row["drop_reason"]) if row["drop_reason"] is not None else None
            ),
            conflict_class=(
                str(row["conflict_class"])
                if row["conflict_class"] is not None
                else None
            ),
            counterpart_content_hash=(
                str(row["counterpart_content_hash"])
                if row["counterpart_content_hash"] is not None
                else None
            ),
        )

    def iter_training_denylist(self) -> Iterator[str]:
        if self.connection is None:
            raise RuntimeError("conflict decision index is not open")
        for row in self.connection.execute(
            "SELECT content_hash FROM training_denylist ORDER BY content_hash"
        ):
            yield str(row[0])
