"""SQLite-backed exact/protected-hash index and explainable n-gram similarity."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any


def character_ngrams(text: str, n: int = 3) -> frozenset[str]:
    compact = " ".join(text.split())
    if not compact:
        return frozenset()
    if len(compact) < n:
        return frozenset({compact})
    return frozenset(compact[index : index + n] for index in range(len(compact) - n + 1))


def ngram_jaccard(left: str, right: str, *, n: int = 3) -> float:
    left_set = character_ngrams(left, n)
    right_set = character_ngrams(right, n)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


class DiskRecordIndex:
    """Durable exact-dedup, role-group and phase checkpoint index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS protected_hashes (
                content_hash TEXT PRIMARY KEY,
                role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS group_roles (
                group_id TEXT PRIMARY KEY,
                role TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                sample_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                group_id TEXT NOT NULL,
                role TEXT NOT NULL,
                normalized_text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS phases (
                name TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def __enter__(self) -> "DiskRecordIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def protect_hash(self, content_hash: str, *, role: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO protected_hashes(content_hash, role) VALUES (?, ?)",
            (content_hash, role),
        )
        self.connection.commit()

    def add_record(
        self,
        *,
        sample_id: str,
        content_hash: str,
        group_id: str,
        role: str,
        normalized_text: str,
    ) -> str:
        if self.connection.execute(
            "SELECT 1 FROM protected_hashes WHERE content_hash=?", (content_hash,)
        ).fetchone():
            return "protected_hash_overlap"
        if self.connection.execute(
            "SELECT 1 FROM records WHERE content_hash=?", (content_hash,)
        ).fetchone():
            return "exact_duplicate"
        group_row = self.connection.execute(
            "SELECT role FROM group_roles WHERE group_id=?", (group_id,)
        ).fetchone()
        if group_row and group_row[0] != role:
            return "group_role_conflict"
        self.connection.execute(
            "INSERT OR IGNORE INTO group_roles(group_id, role) VALUES (?, ?)",
            (group_id, role),
        )
        self.connection.execute(
            "INSERT INTO records(sample_id, content_hash, group_id, role, normalized_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (sample_id, content_hash, group_id, role, normalized_text),
        )
        self.connection.commit()
        return "accepted"

    def record_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def mark_phase_complete(self, name: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.connection.execute(
            "INSERT OR REPLACE INTO phases(name, payload_json) VALUES (?, ?)",
            (name, encoded),
        )
        self.connection.commit()

    def phase_payload(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM phases WHERE name=?", (name,)
        ).fetchone()
        return json.loads(row[0]) if row else None


class DiskNearDuplicateIndex:
    """Disk-backed MinHash/LSH candidate index; raw text never leaves its DB."""

    _PRIME = 18_446_744_073_709_551_557

    def __init__(
        self,
        path: str | Path,
        *,
        ngram_size: int,
        signature_size: int,
        bands: int,
    ) -> None:
        if ngram_size <= 0 or signature_size <= 0 or bands <= 0:
            raise ValueError("near-duplicate dimensions must be positive")
        if signature_size % bands:
            raise ValueError("signature_size must be divisible by bands")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ngram_size = ngram_size
        self.signature_size = signature_size
        self.bands = bands
        self.rows_per_band = signature_size // bands
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS items (
                sample_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS buckets (
                bucket_key TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                PRIMARY KEY(bucket_key, sample_id)
            );
            CREATE INDEX IF NOT EXISTS buckets_sample ON buckets(sample_id);
            CREATE TABLE IF NOT EXISTS candidates (
                left_sample_id TEXT NOT NULL,
                right_sample_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                PRIMARY KEY(left_sample_id, right_sample_id)
            );
            """
        )
        self.connection.commit()

    def __enter__(self) -> "DiskNearDuplicateIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def _signature(self, text: str) -> tuple[int, ...]:
        grams = character_ngrams(text, self.ngram_size)
        if not grams:
            grams = frozenset({""})
        values = [
            int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for gram in grams
        ]
        signature: list[int] = []
        for index in range(self.signature_size):
            # Deterministic universal-hash parameters; neither is zero modulo P.
            multiplier = 2 * index + 1
            offset = 0x9E3779B97F4A7C15 * (index + 1)
            signature.append(
                min((multiplier * value + offset) % self._PRIME for value in values)
            )
        return tuple(signature)

    def add(self, sample_id: str, role: str, source: str, text: str) -> None:
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT OR REPLACE INTO items(sample_id, role, source, text, text_sha256) VALUES (?, ?, ?, ?, ?)",
            (sample_id, role, source, text, text_sha),
        )
        self.connection.execute("DELETE FROM buckets WHERE sample_id=?", (sample_id,))
        signature = self._signature(text)
        for band in range(self.bands):
            start = band * self.rows_per_band
            values = signature[start : start + self.rows_per_band]
            payload = f"{band}:" + ",".join(str(value) for value in values)
            bucket = hashlib.sha256(payload.encode("ascii")).hexdigest()
            self.connection.execute(
                "INSERT INTO buckets(bucket_key, sample_id) VALUES (?, ?)",
                (bucket, sample_id),
            )

    def commit(self) -> None:
        self.connection.commit()

    def build_candidates(self, *, threshold: float) -> int:
        if not 0 <= threshold <= 1:
            raise ValueError("near-duplicate threshold must be in [0, 1]")
        self.connection.execute("DELETE FROM candidates")
        query = """
            SELECT DISTINCT
                left_bucket.sample_id AS left_id,
                right_bucket.sample_id AS right_id,
                left_item.text AS left_text,
                right_item.text AS right_text
            FROM buckets AS left_bucket
            JOIN buckets AS right_bucket
              ON left_bucket.bucket_key = right_bucket.bucket_key
             AND left_bucket.sample_id < right_bucket.sample_id
            JOIN items AS left_item ON left_item.sample_id = left_bucket.sample_id
            JOIN items AS right_item ON right_item.sample_id = right_bucket.sample_id
            ORDER BY left_id, right_id
        """
        count = 0
        for row in self.connection.execute(query):
            score = ngram_jaccard(
                str(row["left_text"]), str(row["right_text"]), n=self.ngram_size
            )
            if score < threshold:
                continue
            self.connection.execute(
                "INSERT OR IGNORE INTO candidates(left_sample_id, right_sample_id, similarity) VALUES (?, ?, ?)",
                (row["left_id"], row["right_id"], score),
            )
            count += 1
            if count % 1000 == 0:
                self.connection.commit()
        self.connection.commit()
        return int(
            self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        )

    def iter_candidates(self):
        query = """
            SELECT c.left_sample_id, c.right_sample_id, c.similarity,
                   l.role AS left_role, r.role AS right_role,
                   l.source AS left_source, r.source AS right_source,
                   l.text_sha256 AS left_text_sha256,
                   r.text_sha256 AS right_text_sha256
            FROM candidates AS c
            JOIN items AS l ON l.sample_id=c.left_sample_id
            JOIN items AS r ON r.sample_id=c.right_sample_id
            ORDER BY c.similarity DESC, c.left_sample_id, c.right_sample_id
        """
        yield from self.connection.execute(query)

    def audit_rows(self, *, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.iter_candidates():
            rows.append(
                {
                    "left_sample_id": str(row["left_sample_id"]),
                    "right_sample_id": str(row["right_sample_id"]),
                    "left_role": str(row["left_role"]),
                    "right_role": str(row["right_role"]),
                    "left_source": str(row["left_source"]),
                    "right_source": str(row["right_source"]),
                    "left_text_sha256": str(row["left_text_sha256"]),
                    "right_text_sha256": str(row["right_text_sha256"]),
                    "similarity": round(float(row["similarity"]), 6),
                    "human_reviewed": False,
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def private_audit_rows(self, *, limit: int):
        """Yield text-bearing rows for ignored human review only."""

        query = """
            SELECT c.left_sample_id, c.right_sample_id, c.similarity,
                   l.role AS left_role, r.role AS right_role,
                   l.source AS left_source, r.source AS right_source,
                   l.text AS left_text, r.text AS right_text
            FROM candidates AS c
            JOIN items AS l ON l.sample_id=c.left_sample_id
            JOIN items AS r ON r.sample_id=c.right_sample_id
            ORDER BY c.similarity DESC, c.left_sample_id, c.right_sample_id
            LIMIT ?
        """
        for row in self.connection.execute(query, (limit,)):
            yield dict(row)
