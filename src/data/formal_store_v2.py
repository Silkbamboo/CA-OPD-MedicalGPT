"""Ignored SQLite staging store for the resumable P2 formal build."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.data.schema import DataRecordV2
from dataclasses import replace


def release_file_cache_paths(*paths: str | Path) -> None:
    """Best-effort eviction hint for already-persisted build files."""

    if not hasattr(os, "posix_fadvise"):
        return
    for value in paths:
        candidate = Path(value)
        if not candidate.is_file():
            continue
        descriptor = os.open(candidate, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def _record_json(record: DataRecordV2) -> str:
    return json.dumps(
        record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _from_json(payload: str) -> DataRecordV2:
    value = json.loads(payload)
    allowed = {field.name for field in dataclasses.fields(DataRecordV2)}
    kwargs = {key: item for key, item in value.items() if key in allowed}
    for key in ("options", "normalized_options", "quality_flags"):
        kwargs[key] = tuple(kwargs.get(key, ()))
    return DataRecordV2(**kwargs)


class FormalStore:
    """A durable store; all text-bearing rows stay under ignored `data/interim`."""

    def __init__(self, path: str | Path, *, seed: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-4096")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA wal_autocheckpoint=256")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                sample_id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                group_id TEXT NOT NULL,
                target_role TEXT NOT NULL,
                category TEXT,
                rank_key TEXT NOT NULL,
                record_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                protected INTEGER NOT NULL DEFAULT 0,
                capability TEXT,
                taxonomy_status TEXT,
                taxonomy_rules TEXT,
                drop_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS records_role_active ON records(target_role, active);
            CREATE INDEX IF NOT EXISTS records_source_active ON records(source_key, active);
            CREATE INDEX IF NOT EXISTS records_group ON records(group_id);
            CREATE INDEX IF NOT EXISTS records_category_rank ON records(category, rank_key);
            CREATE TABLE IF NOT EXISTS protected_hashes (
                content_hash TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                target_role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS drops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                drop_reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS phases (
                name TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_progress (
                entry_key TEXT PRIMARY KEY,
                next_row INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def __enter__(self) -> "FormalStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def _drop(self, sample_id: str, source_key: str, reason: str) -> str:
        self.connection.execute(
            "INSERT INTO drops(sample_id, source_key, drop_reason) VALUES (?, ?, ?)",
            (sample_id, source_key, reason),
        )
        return reason

    def commit(self) -> None:
        self.connection.commit()

    def release_file_cache(self, *additional_paths: str | Path) -> None:
        """Commit and advise Linux that completed file pages are no longer needed."""

        self.connection.commit()
        try:
            self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.OperationalError:
            pass
        release_file_cache_paths(
            self.path, Path(f"{self.path}-wal"), *map(Path, additional_paths)
        )

    def checkpoint_source(
        self,
        entry_key: str,
        *,
        next_row: int,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        if status not in {"in_progress", "complete"}:
            raise ValueError("source progress status must be in_progress or complete")
        if next_row < 0:
            raise ValueError("source progress next_row must be non-negative")
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.connection.execute(
            """
            INSERT OR REPLACE INTO source_progress(entry_key, next_row, status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (entry_key, next_row, status, encoded),
        )
        self.connection.commit()

    def source_progress(self, entry_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT next_row, status, payload_json FROM source_progress WHERE entry_key=?",
            (entry_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "next_row": int(row["next_row"]),
            "status": str(row["status"]),
            "payload": json.loads(row["payload_json"]),
        }

    def protect_hash(
        self,
        content_hash: str,
        *,
        sample_id: str,
        target_role: str,
    ) -> None:
        """Protect a capability hash even when every source row is quarantined."""

        self.connection.execute(
            """
            INSERT OR IGNORE INTO protected_hashes(content_hash, sample_id, target_role)
            VALUES (?, ?, ?)
            """,
            (content_hash, sample_id, target_role),
        )

    def stage(
        self, record: DataRecordV2, *, source_key: str, protected: bool
    ) -> str:
        if not protected and self.connection.execute(
            "SELECT 1 FROM protected_hashes WHERE content_hash=?",
            (record.content_hash,),
        ).fetchone():
            return self._drop(
                record.sample_id, source_key, "protected_hash_overlap"
            )
        existing = self.connection.execute(
            "SELECT sample_id, target_role, protected FROM records WHERE content_hash=?",
            (record.content_hash,),
        ).fetchone()
        if existing:
            if protected and bool(existing["protected"]) and existing["target_role"] != record.target_role:
                raise ValueError(
                    "controller/final content hash appears in multiple capability roles"
                )
            return self._drop(record.sample_id, source_key, "exact_duplicate")
        rank_key = hashlib.sha256(
            f"{self.seed}\0{source_key}\0{record.sample_id}".encode("utf-8")
        ).hexdigest()
        self.connection.execute(
            """
            INSERT INTO records(
                sample_id, source_key, content_hash, group_id, target_role,
                category, rank_key, record_json, protected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.sample_id,
                source_key,
                record.content_hash,
                record.group_id,
                record.target_role,
                record.category,
                rank_key,
                _record_json(record),
                int(protected),
            ),
        )
        if protected:
            self.connection.execute(
                "INSERT OR IGNORE INTO protected_hashes(content_hash, sample_id, target_role) VALUES (?, ?, ?)",
                (record.content_hash, record.sample_id, record.target_role),
            )
        return "accepted"

    def add_drop(self, sample_id: str, source_key: str, reason: str) -> None:
        self._drop(sample_id, source_key, reason)

    def count(self, *, active_only: bool = False, role: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active=1")
        if role is not None:
            clauses.append("target_role=?")
            params.append(role)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM records{where}", params
            ).fetchone()[0]
        )

    def iter_records(
        self,
        *,
        role: str | None = None,
        source_key: str | None = None,
        active_only: bool = True,
    ) -> Iterator[DataRecordV2]:
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active=1")
        if role is not None:
            clauses.append("target_role=?")
            params.append(role)
        if source_key is not None:
            clauses.append("source_key=?")
            params.append(source_key)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT record_json FROM records{where} ORDER BY sample_id"
        # A separate read connection lets callers update/deactivate records
        # through the primary connection while streaming this cursor.
        reader = sqlite3.connect(self.path)
        try:
            cursor = reader.execute(query, params)
            for row in cursor:
                yield _from_json(row[0])
        finally:
            reader.close()

    def iter_index_rows(self, *, active_only: bool = True) -> Iterator[sqlite3.Row]:
        where = " WHERE active=1" if active_only else ""
        yield from self.connection.execute(
            "SELECT sample_id, source_key, content_hash, group_id, target_role, "
            f"record_json, protected, category, rank_key FROM records{where} ORDER BY sample_id"
        )

    def update_record(self, record: DataRecordV2) -> None:
        self.connection.execute(
            "UPDATE records SET group_id=?, target_role=?, category=?, record_json=? WHERE sample_id=?",
            (
                record.group_id,
                record.target_role,
                record.category,
                _record_json(record),
                record.sample_id,
            ),
        )

    def get_record(self, sample_id: str) -> DataRecordV2:
        row = self.connection.execute(
            "SELECT record_json FROM records WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if not row:
            raise KeyError(sample_id)
        return _from_json(row[0])

    def is_protected(self, sample_id: str) -> bool:
        row = self.connection.execute(
            "SELECT protected FROM records WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if not row:
            raise KeyError(sample_id)
        return bool(row[0])

    def assign_role(self, sample_id: str, role: str) -> None:
        record = replace(self.get_record(sample_id), target_role=role)
        self.update_record(record)

    def assign_group(self, sample_id: str, group_id: str) -> None:
        record = replace(self.get_record(sample_id), group_id=group_id)
        self.update_record(record)

    def selected_ids_for_source(
        self, source_key: str, *, limit: int, stratified: bool
    ) -> set[str]:
        if limit < 0:
            raise ValueError("selection limit must be non-negative")
        if stratified:
            query = """
                WITH ranked AS (
                    SELECT sample_id, COALESCE(category, 'unknown') AS bucket,
                           rank_key,
                           ROW_NUMBER() OVER (
                               PARTITION BY COALESCE(category, 'unknown')
                               ORDER BY rank_key, sample_id
                           ) AS bucket_rank
                    FROM records
                    WHERE active=1 AND source_key=?
                )
                SELECT sample_id FROM ranked
                ORDER BY bucket_rank, bucket, rank_key, sample_id
                LIMIT ?
            """
        else:
            query = """
                SELECT sample_id FROM records
                WHERE active=1 AND source_key=?
                ORDER BY rank_key, sample_id
                LIMIT ?
            """
        return {
            str(row[0])
            for row in self.connection.execute(query, (source_key, limit))
        }

    def deactivate_unselected_source(
        self, source_key: str, selected: set[str], reason: str
    ) -> None:
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS selected_ids(sample_id TEXT PRIMARY KEY)"
        )
        self.connection.execute("DELETE FROM selected_ids")
        self.connection.executemany(
            "INSERT INTO selected_ids(sample_id) VALUES (?)",
            ((sample_id,) for sample_id in sorted(selected)),
        )
        self.connection.execute(
            """
            INSERT INTO drops(sample_id, source_key, drop_reason)
            SELECT sample_id, source_key, ? FROM records
            WHERE active=1 AND source_key=?
              AND sample_id NOT IN (SELECT sample_id FROM selected_ids)
            """,
            (reason, source_key),
        )
        self.connection.execute(
            """
            UPDATE records SET active=0, drop_reason=?
            WHERE active=1 AND source_key=?
              AND sample_id NOT IN (SELECT sample_id FROM selected_ids)
            """,
            (reason, source_key),
        )

    def set_taxonomy(
        self,
        sample_id: str,
        *,
        status: str,
        capability: str | None,
        rule_ids: tuple[str, ...],
    ) -> None:
        self.connection.execute(
            "UPDATE records SET taxonomy_status=?, capability=?, taxonomy_rules=? WHERE sample_id=?",
            (status, capability, json.dumps(rule_ids), sample_id),
        )

    def deactivate(self, sample_id: str, reason: str) -> None:
        row = self.connection.execute(
            "SELECT source_key, active FROM records WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if not row or not row["active"]:
            return
        self.connection.execute(
            "UPDATE records SET active=0, drop_reason=? WHERE sample_id=?",
            (reason, sample_id),
        )
        self.connection.execute(
            "INSERT INTO drops(sample_id, source_key, drop_reason) VALUES (?, ?, ?)",
            (sample_id, row["source_key"], reason),
        )

    def role_counts(self) -> dict[str, int]:
        return {
            str(role): int(count)
            for role, count in self.connection.execute(
                "SELECT target_role, COUNT(*) FROM records WHERE active=1 GROUP BY target_role ORDER BY target_role"
            )
        }

    def source_counts(self) -> dict[str, int]:
        return {
            str(source): int(count)
            for source, count in self.connection.execute(
                "SELECT source_key, COUNT(*) FROM records WHERE active=1 GROUP BY source_key ORDER BY source_key"
            )
        }

    def capability_counts(self) -> dict[str, int]:
        return {
            str(capability or "unset"): int(count)
            for capability, count in self.connection.execute(
                "SELECT capability, COUNT(*) FROM records WHERE active=1 AND target_role='general_anchors' GROUP BY capability ORDER BY capability"
            )
        }

    def drop_counts(self) -> dict[str, int]:
        counter = Counter(
            str(reason)
            for (reason,) in self.connection.execute("SELECT drop_reason FROM drops")
        )
        return dict(sorted(counter.items()))

    def mark_phase(self, name: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.connection.execute(
            "INSERT OR REPLACE INTO phases(name, payload_json) VALUES (?, ?)",
            (name, encoded),
        )
        self.connection.commit()

    def phase(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM phases WHERE name=?", (name,)
        ).fetchone()
        return json.loads(row[0]) if row else None
