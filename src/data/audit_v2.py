"""Privacy-safe, bounded audit primitives for pinned Data Protocol v2 sources.

This module deliberately reuses :mod:`src.data.adapters` and
:mod:`src.data.schema`.  It audits repository metadata and a small number of
rows; it is not a dataset builder and never emits source text in a report.
"""

from __future__ import annotations

import hashlib
import csv
import codecs
import http.client
import io
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adapters import (
    CEVAL_SUBJECT_ALLOWLIST,
    AdapterContext,
    adapt_source_row,
)
from .schema import DATA_PROTOCOL_VERSION
from src.utils.run_meta import git_dirty, git_sha


AUDIT_VERSION = "p1.5-upstream-audit-v2"
MAX_PER_SOURCE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 125 * 1024 * 1024
AUDIT_INJECTED_FIELDS = frozenset(
    {"_source_license", "_subject", "_subsource", "_upstream_split"}
)
AUDIT_IMPLEMENTATION_INPUTS = (
    "configs/data/audit_v2.yaml",
    "configs/data/sources_v2.yaml",
    "scripts/audit_upstream_v2.py",
    "src/data/adapters.py",
    "src/data/audit_v2.py",
    "src/data/schema.py",
)
BLOCKED_REVISION = "blocked_revision"
BLOCKED_SCHEMA = "blocked_schema"
BLOCKED_LICENSE = "blocked_license"
BLOCKED_DOWNLOAD_BUDGET = "blocked_download_budget"
BLOCKED_NETWORK = "blocked_network"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    """A fail-closed audit outcome with a stable public classification."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        public_details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.public_details = public_details or message

    @classmethod
    def network(cls, url: str, error: BaseException) -> "AuditError":
        del error  # Exception bodies may contain credentials or response text.
        return cls(
            BLOCKED_NETWORK,
            f"network request failed for {url}",
            public_details=f"network request failed for {url}",
        )


class PartialDownloadError(OSError):
    """Transport failure carrying a conservative byte count for the ledger."""

    def __init__(self, *, url: str, bytes_received: int) -> None:
        super().__init__(f"partial response failed for {url}")
        self.url = url
        self.bytes_received = max(0, int(bytes_received))


@dataclass(frozen=True)
class AuditLimits:
    """Hard limits applied before any response or row is accepted."""

    per_source_bytes: int = MAX_PER_SOURCE_BYTES
    total_bytes: int = MAX_TOTAL_BYTES
    max_records_per_source: int = 50
    timeout_seconds: int = 20
    max_retries: int = 2

    def __post_init__(self) -> None:
        for name in (
            "per_source_bytes",
            "total_bytes",
            "max_records_per_source",
            "timeout_seconds",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_records_per_source > 50:
            raise ValueError("max_records_per_source must not exceed 50")
        if (
            self.per_source_bytes > MAX_PER_SOURCE_BYTES
            or self.total_bytes > MAX_TOTAL_BYTES
        ):
            raise ValueError("download budget exceeds the project ceiling")
        if self.max_retries < 0 or self.max_retries > 2:
            raise ValueError("max_retries must be between zero and two")


def validate_session_transfer_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate cumulative real-network evidence against immutable hard caps."""

    try:
        attempts = [int(item) for item in value["attempt_bytes"]]
        log_hashes = [str(item) for item in value.get("attempt_log_sha256", [])]
        source_bytes = {
            str(key): int(item) for key, item in dict(value["source_bytes"]).items()
        }
        total = int(value["total_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(
            BLOCKED_DOWNLOAD_BUDGET, "invalid session transfer evidence"
        ) from error
    valid = (
        bool(attempts)
        and all(item >= 0 for item in attempts)
        and all(item >= 0 for item in source_bytes.values())
        and len(log_hashes) == len(attempts)
        and all(_SHA256.fullmatch(item) for item in log_hashes)
        and sum(attempts) == total
        and sum(source_bytes.values()) == total
        and total <= MAX_TOTAL_BYTES
        and all(item <= MAX_PER_SOURCE_BYTES for item in source_bytes.values())
    )
    if not valid:
        raise AuditError(
            BLOCKED_DOWNLOAD_BUDGET,
            "session transfer evidence is inconsistent or exceeds a hard cap",
        )
    return {
        "attempt_bytes": attempts,
        "attempt_log_sha256": log_hashes,
        "source_bytes": dict(sorted(source_bytes.items())),
        "total_bytes": total,
    }


@dataclass(frozen=True)
class HttpPayload:
    """A downloaded response body; callers charge it before parsing."""

    body: bytes
    url: str
    status_code: int = 200
    headers: Mapping[str, str] | None = None


class BudgetLedger:
    """Track response bytes without loading or estimating dataset size."""

    def __init__(self, limits: AuditLimits) -> None:
        self.limits = limits
        self._source_bytes: Counter[str] = Counter()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def source_bytes(self, source_key: str) -> int:
        return self._source_bytes[source_key]

    def remaining_for(self, source_key: str) -> int:
        return min(
            self.limits.per_source_bytes - self.source_bytes(source_key),
            self.limits.total_bytes - self.total_bytes,
        )

    def charge(self, source_key: str, payload: HttpPayload) -> int:
        return self.charge_bytes(source_key, len(payload.body))

    def charge_bytes(self, source_key: str, size: int) -> int:
        if size < 0:
            raise ValueError("download byte charge cannot be negative")
        proposed_source = self.source_bytes(source_key) + size
        if proposed_source > self.limits.per_source_bytes:
            raise AuditError(
                BLOCKED_DOWNLOAD_BUDGET,
                f"{source_key} response exceeds per-source download budget",
            )
        proposed_total = self.total_bytes + size
        if proposed_total > self.limits.total_bytes:
            raise AuditError(
                BLOCKED_DOWNLOAD_BUDGET,
                "response exceeds total download budget",
            )
        self._source_bytes[source_key] = proposed_source
        self._total_bytes = proposed_total
        return size


class UrlLibTransport:
    """Small streaming HTTP transport with a hard per-response byte ceiling."""

    @staticmethod
    def _read_response(response: Any, *, url: str, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        read_size = 64 * 1024
        try:
            while received < max_bytes:
                chunk = response.read(min(read_size, max_bytes - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if received > max_bytes:
                    raise PartialDownloadError(
                        url=url,
                        bytes_received=received,
                    )
        except PartialDownloadError:
            raise
        except (OSError, TimeoutError, http.client.IncompleteRead) as error:
            partial = getattr(error, "partial", b"")
            if isinstance(partial, bytes):
                received += len(partial)
            # A failed read can have consumed a not-yet-returned chunk. Charge
            # one chunk conservatively so a retry cannot undercount traffic.
            uncertain = min(read_size, max(0, max_bytes - received))
            raise PartialDownloadError(
                url=url,
                bytes_received=received + uncertain,
            ) from error
        return b"".join(chunks)

    def get(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_bytes: int,
    ) -> HttpPayload:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "CA-OPD-MedicalGPT-p1.5-audit/1",
            },
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            headers = {key.casefold(): value for key, value in response.headers.items()}
            declared = headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise AuditError(
                    BLOCKED_DOWNLOAD_BUDGET,
                    "HTTP Content-Length exceeds remaining download budget",
                )
            body = self._read_response(response, url=url, max_bytes=max_bytes)
        return HttpPayload(
                body=body,
                url=url,
                status_code=int(response.status),
                headers=headers,
            )

    def get_range(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_bytes: int,
    ) -> HttpPayload:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/octet-stream, application/json, text/plain, */*",
                "Range": f"bytes=0-{max_bytes - 1}",
                "User-Agent": "CA-OPD-MedicalGPT-p1.5-audit/1",
            },
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            headers = {key.casefold(): value for key, value in response.headers.items()}
            declared = headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise AuditError(
                    BLOCKED_DOWNLOAD_BUDGET,
                    "Range response Content-Length exceeds requested prefix",
                )
            body = self._read_response(response, url=url, max_bytes=max_bytes)
            return HttpPayload(
                body=body,
                url=url,
                status_code=int(response.status),
                headers=headers,
            )


class BudgetedHttpClient:
    """Retry bounded GETs and charge only complete responses."""

    def __init__(
        self,
        *,
        transport: Any,
        ledger: BudgetLedger,
        limits: AuditLimits,
    ) -> None:
        self.transport = transport
        self.ledger = ledger
        self.limits = limits

    def get_bytes(self, source_key: str, url: str) -> HttpPayload:
        last_error: BaseException | None = None
        for _attempt in range(self.limits.max_retries + 1):
            try:
                remaining = self.ledger.remaining_for(source_key)
                if remaining <= 0:
                    raise AuditError(
                        BLOCKED_DOWNLOAD_BUDGET,
                        f"{source_key} has no remaining download budget",
                    )
                payload = self.transport.get(
                    url,
                    timeout_seconds=self.limits.timeout_seconds,
                    max_bytes=remaining,
                )
                self.ledger.charge(source_key, payload)
                if not 200 <= payload.status_code < 300:
                    raise AuditError(
                        BLOCKED_NETWORK,
                        f"HTTP {payload.status_code} for {url}",
                    )
                return payload
            except AuditError:
                raise
            except PartialDownloadError as error:
                self.ledger.charge_bytes(source_key, error.bytes_received)
                last_error = error
            except (OSError, TimeoutError) as error:
                last_error = error
        assert last_error is not None
        raise AuditError.network(url, last_error)

    def get_range_bytes(
        self,
        source_key: str,
        url: str,
        *,
        max_bytes: int,
    ) -> HttpPayload:
        if max_bytes <= 0 or max_bytes > self.ledger.remaining_for(source_key):
            raise AuditError(
                BLOCKED_DOWNLOAD_BUDGET,
                "requested Range exceeds remaining download budget",
            )
        last_error: BaseException | None = None
        for _attempt in range(self.limits.max_retries + 1):
            try:
                remaining = self.ledger.remaining_for(source_key)
                if remaining <= 0:
                    raise AuditError(
                        BLOCKED_DOWNLOAD_BUDGET,
                        f"{source_key} has no remaining download budget",
                    )
                attempt_max_bytes = min(max_bytes, remaining)
                payload = self.transport.get_range(
                    url,
                    timeout_seconds=self.limits.timeout_seconds,
                    max_bytes=attempt_max_bytes,
                )
                self.ledger.charge(source_key, payload)
                if not 200 <= payload.status_code < 300:
                    raise AuditError(
                        BLOCKED_NETWORK,
                        f"HTTP {payload.status_code} for {url}",
                    )
                return payload
            except AuditError:
                raise
            except PartialDownloadError as error:
                self.ledger.charge_bytes(source_key, error.bytes_received)
                last_error = error
            except (OSError, TimeoutError) as error:
                last_error = error
        assert last_error is not None
        raise AuditError.network(url, last_error)

    def get_json(self, source_key: str, url: str) -> Any:
        value, _payload = self.get_json_payload(source_key, url)
        return value

    def get_json_payload(self, source_key: str, url: str) -> tuple[Any, HttpPayload]:
        payload = self.get_bytes(source_key, url)
        try:
            return json.loads(payload.body.decode("utf-8")), payload
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuditError(
                BLOCKED_SCHEMA,
                f"remote response is not valid UTF-8 JSON: {url}",
            ) from error


def require_exact_revision(value: Any) -> str:
    """Reject branch names, tags and malformed hashes."""

    revision = str(value or "")
    if not _SHA40.fullmatch(revision):
        raise AuditError(
            BLOCKED_REVISION,
            "source revision must be an immutable 40-hex commit SHA",
        )
    return revision


def require_matching_revision(configured: Any, resolved: Any) -> str:
    configured_sha = require_exact_revision(configured)
    resolved_sha = require_exact_revision(resolved)
    if configured_sha != resolved_sha:
        raise AuditError(
            BLOCKED_REVISION,
            "resolved revision does not match configured revision",
        )
    return resolved_sha


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "array"
    return type(value).__name__.casefold()


def _nested_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            paths.add(path)
            paths.update(_nested_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        array_path = f"{prefix}[]" if prefix else "[]"
        paths.add(array_path)
        for child in value:
            paths.update(_nested_paths(child, array_path))
    return paths


def classify_label(value: Any) -> str:
    """Classify a gold-label representation without returning its content."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "sequence"
    token = str(value).strip()
    if re.fullmatch(r"[A-Z]", token):
        return "uppercase_letter"
    if re.fullmatch(r"[a-z]", token):
        return "lowercase_letter"
    if re.fullmatch(r"[0-9]+", token):
        return "numeric_string"
    return "text"


def _length_summary(values: Iterable[Any]) -> dict[str, int | None]:
    lengths = [len(value) for value in values if isinstance(value, str)]
    if not lengths:
        return {"count": 0, "min": None, "max": None, "sum": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "sum": sum(lengths),
    }


def _first_field(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _option_count(row: Mapping[str, Any]) -> int | None:
    value = _first_field(row, ("options", "option", "choices"))
    if value is None and all(label in row for label in "ABCD"):
        return sum(row.get(label) not in (None, "") for label in "ABCD")
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schema_fingerprint(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a deterministic structural fingerprint containing no row text."""

    materialized = [
        {
            str(key): value
            for key, value in row.items()
            if str(key) not in AUDIT_INJECTED_FIELDS
        }
        for row in rows
    ]
    field_names = sorted({str(key) for row in materialized for key in row})
    field_summary: dict[str, Any] = {}
    for field in field_names:
        present = [row[field] for row in materialized if field in row]
        field_summary[field] = {
            "types": sorted({_type_name(value) for value in present}),
            "missing_count": len(materialized) - len(present),
            "nullable_count": sum(value is None for value in present),
        }
    nested = sorted(
        {path for row in materialized for path in _nested_paths(row)}
    )
    option_counts = Counter(
        count for row in materialized if (count := _option_count(row)) is not None
    )
    labels = Counter(
        classify_label(
            _first_field(row, ("answer_idx", "answer_index", "label", "answer"))
        )
        for row in materialized
    )
    text_lengths = {
        "question": _length_summary(
            _first_field(row, ("question", "Question", "instruction"))
            for row in materialized
        ),
        "reasoning": _length_summary(
            _first_field(
                row,
                ("reasoning", "Complex_CoT", "complex_cot", "analysis", "explanation"),
            )
            for row in materialized
        ),
        "answer": _length_summary(
            _first_field(row, ("answer", "Answer", "Response", "response", "output"))
            for row in materialized
        ),
    }
    payload: dict[str, Any] = {
        "sampled_count": len(materialized),
        "top_level_fields": field_summary,
        "nested_field_paths": nested,
        "option_count_distribution": {
            str(key): option_counts[key] for key in sorted(option_counts)
        },
        "label_format_distribution": dict(sorted(labels.items())),
        "text_lengths": text_lengths,
    }
    payload["schema_fingerprint_sha256"] = _canonical_sha256(payload)
    return payload


def _target_context_overrides(
    source_key: str,
    row: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    if source_key == "coig":
        return (
            str(row.get("_subsource")) if row.get("_subsource") else None,
            None,
            str(row.get("_source_license"))
            if row.get("_source_license")
            else None,
        )
    if source_key == "ceval":
        return (
            None,
            str(row.get("_subject")) if row.get("_subject") else None,
            None,
        )
    return None, None, None


def audit_adapter_rows(
    *,
    source_key: str,
    source_config: Mapping[str, Any],
    split: str,
    target_role: str,
    rows: Iterable[Mapping[str, Any]],
    raw_file_sha256: str,
    max_records: int,
) -> dict[str, Any]:
    """Run the existing adapter and retain hashes/counters, never raw values."""

    if max_records > 50:
        raise AuditError(BLOCKED_DOWNLOAD_BUDGET, "record cap must not exceed 50")
    if not _SHA256.fullmatch(raw_file_sha256):
        raise AuditError(BLOCKED_SCHEMA, "raw file SHA-256 is invalid")
    materialized: list[Mapping[str, Any]] = []
    accepted_hashes: list[str] = []
    raw_file_hashes: set[str] = set()
    drops: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if index >= max_records:
            raise AuditError(
                BLOCKED_DOWNLOAD_BUDGET,
                f"source attempted to process more than {max_records} records (maximum 50)",
            )
        materialized.append(row)
        subsource, subject, per_row_license = _target_context_overrides(
            source_key, row
        )
        context = AdapterContext(
            source_type=str(source_config["adapter"]),
            source=str(source_config["source"]),
            source_revision=require_exact_revision(source_config["revision"]),
            source_license=per_row_license or str(source_config["license"]),
            upstream_split=split,
            target_role=target_role,
            raw_file_sha256=raw_file_sha256,
            subsource=subsource,
            subject=subject,
        )
        result = adapt_source_row(row, context)
        raw_file_hashes.add(raw_file_sha256)
        if result.record is None:
            drops[str(result.drop_reason)] += 1
        else:
            accepted_hashes.append(result.record.content_hash)
    return {
        "sampled_count": len(materialized),
        "adapter_accepted": len(accepted_hashes),
        "adapter_dropped": sum(drops.values()),
        "exact_duplicate_count": len(accepted_hashes) - len(set(accepted_hashes)),
        "drop_reason_counts": dict(sorted(drops.items())),
        "normalized_content_hash_summary_sha256": hashlib.sha256(
            "\n".join(sorted(accepted_hashes)).encode("ascii")
        ).hexdigest(),
        "raw_file_sha256s": sorted(raw_file_hashes),
        "schema": schema_fingerprint(materialized),
    }


def coig_license_decision(
    *,
    subsource: str,
    declared_license: str,
    evidence_sha256: str | None,
) -> dict[str, Any]:
    """Fail closed unless a subsource-specific license has file evidence."""

    normalized = declared_license.strip().casefold()
    known = normalized not in {
        "",
        "unknown",
        "unverified",
        "none",
        "null",
        "subsource-specific",
    }
    evidenced = bool(evidence_sha256 and _SHA256.fullmatch(evidence_sha256))
    if known and evidenced:
        return {
            "subsource": subsource,
            "declared_license": declared_license,
            "decision": "include",
            "audit_status": "verified",
        }
    return {
        "subsource": subsource,
        "declared_license": declared_license,
        "decision": "exclude" if subsource in {"translated", "exam"} else "blocked",
        "audit_status": BLOCKED_LICENSE,
    }


def validate_ceval_subject(subject: str) -> str:
    if subject not in CEVAL_SUBJECT_ALLOWLIST:
        raise AuditError(BLOCKED_SCHEMA, "C-Eval subject is outside the frozen allowlist")
    return subject


def build_source_result(
    *,
    source_key: str,
    configured_revision: str,
    resolved_revision: str,
    repository: str,
    configs: Sequence[str],
    splits: Sequence[str],
    metadata_sha256: str,
    schema_summary: Mapping[str, Any],
    license_status: str,
    sample_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic, redacted per-source report fragment."""

    require_matching_revision(configured_revision, resolved_revision)
    result = {
        "source": source_key,
        "repository": repository,
        "configured_revision": configured_revision,
        "resolved_revision": resolved_revision,
        "configs": sorted(str(value) for value in configs),
        "splits": sorted(str(value) for value in splits),
        "metadata_sha256": metadata_sha256,
        "schema_fingerprint_sha256": schema_summary.get(
            "schema_fingerprint_sha256"
        ),
        "license_status": license_status,
        "sampled_count": int(sample_stats.get("sampled_count", 0)),
        "adapter_accepted": int(sample_stats.get("adapter_accepted", 0)),
        "adapter_dropped": int(sample_stats.get("adapter_dropped", 0)),
        "audit_only": True,
        "formal_final_manifest": False,
    }
    return result


def is_path_gitignored(repo_root: Path, path: Path) -> bool:
    """Ask Git itself whether a prospective raw-audit path is ignored."""

    relative = path.resolve().relative_to(repo_root.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(BLOCKED_SCHEMA, f"{path} must contain a YAML mapping")
    return value


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"


def _write_raw_payload(
    raw_root: Path,
    source_key: str,
    label: str,
    payload: HttpPayload,
) -> tuple[Path, str]:
    digest = _sha256_bytes(payload.body)
    directory = raw_root / _safe_component(source_key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_component(label)}-{digest[:16]}.raw"
    path.write_bytes(payload.body)
    return path, digest


def _find_revision(value: Any) -> str | None:
    """Find datasets-server's revision field without inspecting row values."""

    if isinstance(value, Mapping):
        for key in (
            "dataset_git_revision",
            "dataset_revision",
            "git_revision",
            "resolved_revision",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and _SHA40.fullmatch(candidate):
                return candidate
        for key, child in value.items():
            if key in {"row", "rows", "features"}:
                continue
            found = _find_revision(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value[:1000]:
            found = _find_revision(child)
            if found:
                return found
    return None


def _hf_api_url(repository: str, revision: str) -> str:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"


def _hf_tree_url(repository: str, revision: str) -> str:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    return (
        f"https://huggingface.co/api/datasets/{repo}/tree/{rev}"
        "?recursive=true&expand=false"
    )


def _hf_raw_url(repository: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(repository, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{encoded_path}"


def _datasets_server_url(
    endpoint: str,
    repository: str,
    revision: str,
    **parameters: Any,
) -> str:
    query = {
        "dataset": repository,
        "revision": revision,
        **{key: value for key, value in parameters.items() if value is not None},
    }
    return (
        f"https://datasets-server.huggingface.co/{endpoint}?"
        + urllib.parse.urlencode(query)
    )


def _tree_entries(repo_info: Mapping[str, Any], tree_value: Any) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    siblings = repo_info.get("siblings")
    if isinstance(siblings, list):
        for item in siblings:
            if not isinstance(item, Mapping):
                continue
            path = item.get("rfilename") or item.get("path")
            if isinstance(path, str):
                entries[path] = {
                    "path": path,
                    "type": str(item.get("type", "file")),
                    "size": item.get("size"),
                    "oid": item.get("blobId") or item.get("oid"),
                }
    if isinstance(tree_value, list):
        for item in tree_value:
            if not isinstance(item, Mapping):
                continue
            path = item.get("path") or item.get("rfilename")
            if isinstance(path, str):
                entries[path] = {
                    "path": path,
                    "type": str(item.get("type", "file")),
                    "size": item.get("size"),
                    "oid": item.get("oid") or item.get("blobId"),
                }
    return [entries[path] for path in sorted(entries)]


def _evidence_candidates(tree: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    preferred: list[Mapping[str, Any]] = []
    for item in tree:
        path = str(item.get("path", ""))
        basename = Path(path).name.casefold()
        if basename in {
            "license",
            "license.md",
            "license.txt",
            "copying",
            "readme.md",
        }:
            preferred.append(item)
    return sorted(
        preferred,
        key=lambda item: (
            0 if Path(str(item.get("path", ""))).name.casefold().startswith("license") else 1,
            str(item.get("path", "")),
        ),
    )


def _license_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _license_body_supports(declared: str, body: bytes) -> bool:
    try:
        text = body.decode("utf-8").casefold()
    except UnicodeDecodeError:
        return False
    token = _license_token(declared)
    compact = _license_token(text[:1_000_000])
    if token in {"apache20", "apachelicense20"}:
        return "apachelicenseversion20" in compact or "licenseapache20" in compact
    if token in {"ccbyncsa40", "creativecommonsattributionnoncommercialsharealike40"}:
        return "ccbyncsa40" in compact or (
            "creativecommons" in compact
            and "noncommercial" in compact
            and "sharealike" in compact
        )
    if token in {"ccbys a40".replace(" ", ""), "ccbysa40"}:
        return "ccbysa40" in compact
    return bool(token and token in compact)


def _extract_rows(value: Any) -> list[Mapping[str, Any]]:
    rows_value = value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows_value, list):
        raise AuditError(BLOCKED_SCHEMA, "datasets-server response has no rows list")
    rows: list[Mapping[str, Any]] = []
    for item in rows_value:
        if isinstance(item, Mapping) and isinstance(item.get("row"), Mapping):
            rows.append(dict(item["row"]))
        elif isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            raise AuditError(BLOCKED_SCHEMA, "datasets-server row is not a mapping")
    return rows


def _extract_splits(value: Any) -> list[dict[str, str]]:
    split_values = value.get("splits") if isinstance(value, Mapping) else None
    if not isinstance(split_values, list):
        return []
    result: list[dict[str, str]] = []
    for item in split_values:
        if not isinstance(item, Mapping):
            continue
        config = item.get("config")
        split = item.get("split")
        if isinstance(config, str) and isinstance(split, str):
            result.append({"config": config, "split": split})
    return sorted(result, key=lambda item: (item["config"], item["split"]))


def _parse_exact_rows(path: str, body: bytes, limit: int) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.casefold()
    if suffix == ".csv":
        text = io.TextIOWrapper(io.BytesIO(body), encoding="utf-8-sig", newline="")
        return [dict(row) for _, row in zip(range(limit), csv.DictReader(text))]
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line in body.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(BLOCKED_SCHEMA, "JSONL row is not an object")
            rows.append(value)
            if len(rows) >= limit:
                break
        return rows
    raise AuditError(BLOCKED_SCHEMA, f"unsupported exact sample file type: {suffix}")


def parse_bounded_json_rows(
    path: str,
    body: bytes,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Decode complete rows from a bounded JSON/JSONL prefix.

    A Range response normally ends mid-record.  The incomplete tail is ignored;
    complete rows are decoded one at a time and never materialized beyond
    ``limit``.
    """

    if limit < 1 or limit > 50:
        raise AuditError(BLOCKED_DOWNLOAD_BUDGET, "range row limit must be 1..50")
    try:
        decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="strict")
        text = decoder.decode(body, final=False)
    except UnicodeDecodeError as error:
        raise AuditError(BLOCKED_SCHEMA, "range prefix is not valid UTF-8") from error
    suffix = Path(path).suffix.casefold()
    rows: list[dict[str, Any]] = []
    if suffix in {".jsonl", ".ndjson"}:
        lines = text.splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                is_incomplete_tail = index == len(lines) - 1 and not raw_line.endswith(
                    ("\n", "\r")
                )
                if is_incomplete_tail:
                    break
                raise AuditError(BLOCKED_SCHEMA, "malformed complete JSONL row") from error
            if not isinstance(value, dict):
                raise AuditError(BLOCKED_SCHEMA, "JSONL row is not an object")
            rows.append(value)
            if len(rows) >= limit:
                break
        return rows
    if suffix != ".json":
        raise AuditError(BLOCKED_SCHEMA, "range sampling supports JSON/JSONL only")

    position = 0
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "[":
        raise AuditError(BLOCKED_SCHEMA, "range JSON root must be an array")
    position += 1
    decoder = json.JSONDecoder()
    while len(rows) < limit:
        while position < len(text) and (
            text[position].isspace() or text[position] == ","
        ):
            position += 1
        if position >= len(text) or text[position] == "]":
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            raise AuditError(BLOCKED_SCHEMA, "JSON array row is not an object")
        rows.append(value)
    return rows


def _select_exact_sample_file(
    tree: Sequence[Mapping[str, Any]],
    *,
    config: str,
    split: str,
    remaining_bytes: int,
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    config_token = config.casefold()
    split_token = split.casefold()
    for item in tree:
        path = str(item.get("path", ""))
        folded = path.casefold()
        size = item.get("size")
        if Path(path).suffix.casefold() not in {".csv", ".jsonl", ".ndjson"}:
            continue
        if config != "default" and config_token not in folded:
            continue
        if split_token not in folded:
            continue
        if not isinstance(size, int) or size <= 0 or size > remaining_bytes:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return min(candidates, key=lambda item: (int(item["size"]), str(item["path"])))


def _coig_category(row: Mapping[str, Any]) -> str | None:
    fields = (
        row.get("task_name_in_eng"),
        row.get("task_name"),
        row.get("domain"),
        row.get("source"),
        row.get("dataset"),
    )
    folded = " ".join(str(value or "") for value in fields).casefold()
    if any(marker in folded for marker in ("leetcode", "code", "program")):
        return "leetcode"
    if any(marker in folded for marker in ("human value", "human_value", "alignment", "harmless")):
        return "human_value"
    if any(marker in folded for marker in ("exam", "ceval", "gaokao", "考试")):
        return "exam"
    if any(marker in folded for marker in ("translat", "alpaca", "general", "instruction")):
        return "translated"
    return None


def _inject_audit_context(
    row: Mapping[str, Any],
    *,
    source_key: str,
    split: str,
    config: str,
    subsource: str | None,
    source_license: str | None,
) -> dict[str, Any]:
    value = dict(row)
    if source_key in {"medqa_zh", "ceval"}:
        value["_upstream_split"] = split
    if source_key == "ceval":
        value["_subject"] = config
    if source_key == "coig":
        value["_subsource"] = subsource or "unclassified"
        value["_source_license"] = source_license or "unknown"
    return value


def _merge_adapter_stats(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    drops: Counter[str] = Counter()
    sampled = accepted = dropped = 0
    exact_duplicates = 0
    raw_hashes: set[str] = set()
    fingerprint_hashes: list[str] = []
    normalized_hash_summaries: list[str] = []
    for result in results:
        sampled += int(result["sampled_count"])
        accepted += int(result["adapter_accepted"])
        dropped += int(result["adapter_dropped"])
        exact_duplicates += int(result.get("exact_duplicate_count", 0))
        drops.update(result.get("drop_reason_counts", {}))
        raw_hashes.update(str(value) for value in result.get("raw_file_sha256s", []))
        schema = result.get("schema", {})
        if isinstance(schema, Mapping) and schema.get("schema_fingerprint_sha256"):
            fingerprint_hashes.append(str(schema["schema_fingerprint_sha256"]))
        if result.get("normalized_content_hash_summary_sha256"):
            normalized_hash_summaries.append(
                str(result["normalized_content_hash_summary_sha256"])
            )
    return {
        "sampled_count": sampled,
        "adapter_accepted": accepted,
        "adapter_dropped": dropped,
        "exact_duplicate_count": exact_duplicates,
        "drop_reason_counts": dict(sorted(drops.items())),
        "raw_file_sha256s": sorted(raw_hashes),
        "combined_schema_fingerprint_sha256": hashlib.sha256(
            "\n".join(sorted(fingerprint_hashes)).encode("ascii")
        ).hexdigest(),
        "combined_normalized_hash_summary_sha256": hashlib.sha256(
            "\n".join(sorted(normalized_hash_summaries)).encode("ascii")
        ).hexdigest(),
    }


def _payload_revision(value: Any, payload: HttpPayload) -> str | None:
    revision = _find_revision(value)
    if revision:
        return revision
    headers = payload.headers or {}
    for key in (
        "x-dataset-git-revision",
        "x-dataset-revision",
        "x-revision",
    ):
        candidate = headers.get(key)
        if isinstance(candidate, str) and _SHA40.fullmatch(candidate):
            return candidate
    return None


def _sample_plan_rows(
    *,
    client: BudgetedHttpClient,
    ledger: BudgetLedger,
    raw_root: Path,
    source_key: str,
    repository: str,
    revision: str,
    tree: Sequence[Mapping[str, Any]],
    config: str,
    split: str,
    limit: int,
    range_file: str | None = None,
    range_prefix_bytes: int | None = None,
    prefer_range_file: bool = False,
) -> tuple[list[Mapping[str, Any]], str, str, str]:
    """Read pinned rows via datasets-server or a small exact text file."""

    if prefer_range_file and range_file and range_prefix_bytes:
        matching = [item for item in tree if str(item.get("path")) == range_file]
        if len(matching) != 1:
            raise AuditError(BLOCKED_SCHEMA, "configured Range file is absent from fixed tree")
        raw_url = _hf_raw_url(repository, revision, range_file)
        payload = client.get_range_bytes(
            source_key,
            raw_url,
            max_bytes=min(int(range_prefix_bytes), ledger.remaining_for(source_key)),
        )
        _write_raw_payload(
            raw_root,
            source_key,
            f"range-{config}-{split}-{Path(range_file).name}",
            payload,
        )
        rows = parse_bounded_json_rows(range_file, payload.body, limit=limit)
        if not rows:
            raise AuditError(BLOCKED_SCHEMA, "Range prefix contains no complete rows")
        return rows, _sha256_bytes(payload.body), "fixed_revision_http_range", raw_url

    rows_url = _datasets_server_url(
        "rows",
        repository,
        revision,
        config=config,
        split=split,
        offset=0,
        length=limit,
    )
    rows_error: AuditError | None = None
    try:
        value, payload = client.get_json_payload(source_key, rows_url)
        _write_raw_payload(
            raw_root,
            source_key,
            f"rows-{config}-{split}",
            payload,
        )
        response_revision = _payload_revision(value, payload)
        if response_revision is None:
            raise AuditError(
                BLOCKED_REVISION,
                "datasets-server rows response is not bound to a commit SHA",
            )
        require_matching_revision(revision, response_revision)
        rows = _extract_rows(value)
        if len(rows) > limit:
            raise AuditError(
                BLOCKED_DOWNLOAD_BUDGET,
                "datasets-server returned more rows than requested",
            )
        return rows, _sha256_bytes(payload.body), "datasets_server_rows", rows_url
    except AuditError as error:
        rows_error = error

    exact = _select_exact_sample_file(
        tree,
        config=config,
        split=split,
        remaining_bytes=ledger.remaining_for(source_key),
    )
    if exact is not None:
        path = str(exact["path"])
        raw_url = _hf_raw_url(repository, revision, path)
        payload = client.get_bytes(source_key, raw_url)
        _write_raw_payload(
            raw_root,
            source_key,
            f"exact-{config}-{split}-{Path(path).name}",
            payload,
        )
        rows = _parse_exact_rows(path, payload.body, limit)
        return rows, _sha256_bytes(payload.body), "fixed_revision_exact_file", raw_url

    if range_file and range_prefix_bytes:
        matching = [item for item in tree if str(item.get("path")) == range_file]
        if len(matching) != 1:
            raise AuditError(BLOCKED_SCHEMA, "configured Range file is absent from fixed tree")
        path = str(matching[0]["path"])
        raw_url = _hf_raw_url(repository, revision, path)
        prefix_bytes = min(
            int(range_prefix_bytes),
            ledger.remaining_for(source_key),
        )
        payload = client.get_range_bytes(
            source_key,
            raw_url,
            max_bytes=prefix_bytes,
        )
        _write_raw_payload(
            raw_root,
            source_key,
            f"range-{config}-{split}-{Path(path).name}",
            payload,
        )
        rows = parse_bounded_json_rows(path, payload.body, limit=limit)
        if not rows:
            raise AuditError(BLOCKED_SCHEMA, "Range prefix contains no complete rows")
        return rows, _sha256_bytes(payload.body), "fixed_revision_http_range", raw_url

    assert rows_error is not None
    raise rows_error


def _ceval_plans(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = int(config["samples_per_subject_split"])
    if count < 1 or count > 2:
        raise AuditError(BLOCKED_SCHEMA, "C-Eval audit count must be one or two")
    subjects = [str(value) for value in config["subjects"]]
    if len(subjects) != len(CEVAL_SUBJECT_ALLOWLIST) or set(subjects) != set(
        CEVAL_SUBJECT_ALLOWLIST
    ):
        raise AuditError(
            BLOCKED_SCHEMA,
            "C-Eval audit must contain exactly the eight unique frozen subjects",
        )
    expected_splits = {
        "dev": "ceval_smoke",
        "val": "general_controller_dev",
        "test": "general_final_test",
    }
    actual_splits = {str(key): str(value) for key, value in config["splits"].items()}
    if actual_splits != expected_splits:
        raise AuditError(BLOCKED_SCHEMA, "C-Eval split-to-role mapping is not frozen")
    plans: list[dict[str, Any]] = []
    for subject in subjects:
        validate_ceval_subject(subject)
        for split, role in expected_splits.items():
            plans.append(
                {
                    "config": str(subject),
                    "split": str(split),
                    "target_role": str(role),
                    "limit": count,
                }
            )
    return plans


def sample_audit_complete(
    *,
    requested_count: int,
    sampled_count: int,
    accepted_count: int,
    dropped_count: int,
    sample_errors: Sequence[Mapping[str, Any]],
) -> bool:
    """Require complete, drop-free evidence before declaring sample verified."""

    return (
        requested_count > 0
        and sampled_count == requested_count
        and accepted_count == requested_count
        and dropped_count == 0
        and not sample_errors
    )


def _source_plans(source_key: str, audit_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if source_key == "ceval":
        return _ceval_plans(audit_config)
    samples = audit_config.get("samples")
    if not isinstance(samples, list) or not samples:
        raise AuditError(BLOCKED_SCHEMA, f"{source_key} has no audit sample plan")
    return [dict(item) for item in samples if isinstance(item, Mapping)]


def _audit_coig_rows(
    *,
    source_config: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    raw_sha256: str,
    source_scoped: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quotas = {str(plan["subsource"]): int(plan["limit"]) for plan in plans}
    selected: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in quotas
    }
    if source_scoped and len(plans) == 1:
        only_subsource = str(plans[0]["subsource"])
        selected[only_subsource].extend(rows[: quotas[only_subsource]])
    else:
        for row in rows:
            category = _coig_category(row)
            if category in selected and len(selected[category]) < quotas[category]:
                selected[category].append(row)

    candidate_licenses = {
        str(key): str(value)
        for key, value in dict(
            source_config.get("candidate_subsource_licenses", {})
        ).items()
    }
    stats: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for plan in plans:
        subsource = str(plan["subsource"])
        upstream_split = str(plan["split"])
        license_name = candidate_licenses.get(subsource, "unknown")
        decision = coig_license_decision(
            subsource=subsource,
            declared_license=license_name,
            evidence_sha256=None,
        )
        matrix.append(
            {
                **decision,
                "umbrella_license": str(source_config.get("license", "unknown")),
                "evidence_type": "fixed_revision_repo_metadata_only",
                "evidence_url": None,
                "evidence_revision": str(source_config["revision"]),
                "evidence_file_sha256": None,
                "research_training_allowed": False,
                "derived_manifest_allowed": True,
                "notes": "no subsource-specific license closure at the pinned revision",
            }
        )
        injected = [
            _inject_audit_context(
                row,
                source_key="coig",
                split=upstream_split,
                config=str(plan["config"]),
                subsource=subsource,
                source_license=license_name if decision["decision"] == "include" else "unknown",
            )
            for row in selected[subsource]
        ]
        schema_only_injected = [
            _inject_audit_context(
                row,
                source_key="coig",
                split=upstream_split,
                config=str(plan["config"]),
                subsource=subsource,
                source_license="schema-audit-only",
            )
            for row in selected[subsource]
        ]
        schema_only = audit_adapter_rows(
            source_key="coig",
            source_config=source_config,
            split=upstream_split,
            target_role=str(plan["target_role"]),
            rows=schema_only_injected,
            raw_file_sha256=raw_sha256,
            max_records=int(plan["limit"]),
        )
        stats.append(
            {
                "config": str(plan["config"]),
                "split": upstream_split,
                "target_role": str(plan["target_role"]),
                "subsource": subsource,
                "requested_count": int(plan["limit"]),
                **audit_adapter_rows(
                    source_key="coig",
                    source_config=source_config,
                    split=upstream_split,
                    target_role=str(plan["target_role"]),
                    rows=injected,
                    raw_file_sha256=raw_sha256,
                    max_records=int(plan["limit"]),
                ),
                "schema_only_if_license_verified": {
                    "adapter_accepted": schema_only["adapter_accepted"],
                    "adapter_dropped": schema_only["adapter_dropped"],
                    "drop_reason_counts": schema_only["drop_reason_counts"],
                },
            }
        )
    return stats, matrix


def _load_raw_evidence_by_sha(
    raw_root: Path,
    source_key: str,
    expected_sha256: str,
) -> bytes:
    if not _SHA256.fullmatch(expected_sha256):
        raise AuditError(BLOCKED_SCHEMA, "raw replay SHA256 is invalid")
    source_dir = raw_root / _safe_component(source_key)
    matches: list[Path] = []
    for path in source_dir.glob(f"*-{expected_sha256[:16]}.raw"):
        if path.is_file() and path.stat().st_size <= MAX_PER_SOURCE_BYTES:
            body = path.read_bytes()
            if _sha256_bytes(body) == expected_sha256:
                matches.append(path)
    if len(matches) != 1:
        raise AuditError(
            BLOCKED_SCHEMA,
            f"raw replay evidence for {source_key} is missing or ambiguous",
        )
    return matches[0].read_bytes()


def replay_source_sample_audits(
    *,
    source_key: str,
    source_config: Mapping[str, Any],
    audit_config: Mapping[str, Any],
    source_result: Mapping[str, Any],
    raw_root: Path,
) -> dict[str, Any]:
    """Replay stored, hash-bound raw payloads through the current adapter."""

    current_revision = require_exact_revision(source_config.get("revision"))
    binding_matches = (
        str(source_config.get("source")) == str(source_result.get("repository"))
        and current_revision == str(source_result.get("configured_revision"))
        and current_revision == str(source_result.get("resolved_revision"))
        and str(source_config.get("license"))
        == str(source_result.get("declared_license"))
    )
    evidence_matches = all(
        str(item.get("evidence_revision")) == current_revision
        and str(item.get("declared_license")) == str(source_config.get("license"))
        for item in source_result.get("license_evidence", [])
        if isinstance(item, Mapping)
    )
    if not binding_matches or not evidence_matches:
        raise AuditError(
            BLOCKED_REVISION,
            "stored raw evidence does not match the current source binding",
        )

    plans = _source_plans(source_key, audit_config)
    expected_contracts = sorted(
        (
            str(plan["config"]),
            str(plan["split"]),
            str(plan.get("subsource", "")),
            str(plan["target_role"]),
            int(plan["limit"]),
        )
        for plan in plans
    )
    stored_contracts = sorted(
        (
            str(sample["config"]),
            str(sample["split"]),
            str(sample.get("subsource", "")),
            str(sample["target_role"]),
            int(sample["requested_count"]),
        )
        for sample in source_result.get("sample_audits", [])
        if isinstance(sample, Mapping)
    )
    if stored_contracts != expected_contracts:
        raise AuditError(
            BLOCKED_SCHEMA,
            "stored raw evidence does not exactly cover the current sample plans",
        )
    plan_index = {
        (
            str(plan["config"]),
            str(plan["split"]),
            str(plan.get("subsource", "")),
        ): plan
        for plan in plans
    }
    replayed: list[dict[str, Any]] = []
    for previous in source_result.get("sample_audits", []):
        if not isinstance(previous, Mapping):
            raise AuditError(BLOCKED_SCHEMA, "stored sample audit is invalid")
        key = (
            str(previous["config"]),
            str(previous["split"]),
            str(previous.get("subsource", "")),
        )
        plan = plan_index.get(key)
        if plan is None:
            raise AuditError(BLOCKED_SCHEMA, "stored sample audit no longer matches config")
        raw_hashes = previous.get("raw_file_sha256s", [])
        if not isinstance(raw_hashes, list) or len(raw_hashes) != 1:
            raise AuditError(BLOCKED_SCHEMA, "sample replay requires one raw payload SHA")
        raw_sha = str(raw_hashes[0])
        body = _load_raw_evidence_by_sha(raw_root, source_key, raw_sha)
        limit = int(plan["limit"])
        method = str(previous.get("access_method", ""))
        if not method and source_key == "coig" and plan.get("range_file"):
            method = "fixed_revision_http_range"
        if method == "datasets_server_rows":
            try:
                value = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AuditError(BLOCKED_SCHEMA, "raw rows payload is invalid JSON") from error
            rows = _extract_rows(value)[:limit]
            raw_revision = _find_revision(value)
            if raw_revision is not None:
                require_matching_revision(current_revision, raw_revision)
        elif method in {"fixed_revision_http_range", "fixed_revision_exact_file"}:
            range_file = str(plan.get("range_file", ""))
            if not range_file:
                raise AuditError(BLOCKED_SCHEMA, "raw replay lacks its pinned file path")
            rows = parse_bounded_json_rows(range_file, body, limit=limit)
        else:
            raise AuditError(BLOCKED_SCHEMA, "raw replay access method is unsupported")

        if source_key == "coig":
            stats, _matrix = _audit_coig_rows(
                source_config={**source_config, **audit_config},
                plans=[plan],
                rows=rows,
                raw_sha256=raw_sha,
                source_scoped=True,
            )
            current = stats[0]
            current["access_method"] = method
        else:
            injected = [
                _inject_audit_context(
                    row,
                    source_key=source_key,
                    split=str(plan["split"]),
                    config=str(plan["config"]),
                    subsource=None,
                    source_license=None,
                )
                for row in rows
            ]
            stats = audit_adapter_rows(
                source_key=source_key,
                source_config=source_config,
                split=str(plan["split"]),
                target_role=str(plan["target_role"]),
                rows=injected,
                raw_file_sha256=raw_sha,
                max_records=limit,
            )
            current = {
                "config": str(plan["config"]),
                "split": str(plan["split"]),
                "target_role": str(plan["target_role"]),
                "requested_count": limit,
                "access_method": method,
                **stats,
            }
        replayed.append(current)

    merged = _merge_adapter_stats(replayed)
    requested_count = sum(int(plan["limit"]) for plan in plans)
    return {
        "sample_audits": replayed,
        "sampled_count": merged["sampled_count"],
        "adapter_accepted": merged["adapter_accepted"],
        "adapter_dropped": merged["adapter_dropped"],
        "exact_duplicate_count": merged["exact_duplicate_count"],
        "drop_reason_counts": merged["drop_reason_counts"],
        "raw_file_sha256s": merged["raw_file_sha256s"],
        "schema_fingerprint_sha256": merged["combined_schema_fingerprint_sha256"],
        "normalized_hash_summary_sha256": merged[
            "combined_normalized_hash_summary_sha256"
        ],
        "schema_only_if_license_verified": {
            "adapter_accepted": sum(
                int(item.get("schema_only_if_license_verified", {}).get("adapter_accepted", 0))
                for item in replayed
            ),
            "adapter_dropped": sum(
                int(item.get("schema_only_if_license_verified", {}).get("adapter_dropped", 0))
                for item in replayed
            ),
            "status": source_result.get("schema_only_if_license_verified", {}).get(
                "status", "not_applicable"
            ),
        },
        "sample_audit_complete": sample_audit_complete(
            requested_count=requested_count,
            sampled_count=merged["sampled_count"],
            accepted_count=merged["adapter_accepted"],
            dropped_count=merged["adapter_dropped"],
            sample_errors=[],
        ),
    }


def bind_replayed_source(
    source_result: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind replay statistics and fail closed when a verified sample regresses."""

    source = dict(source_result)
    source.update(replayed)
    if source_result.get("status") == "verified" and not replayed.get(
        "sample_audit_complete", False
    ):
        source["status"] = BLOCKED_SCHEMA
        source["blocking_reason"] = BLOCKED_SCHEMA
    return source


def audit_one_source(
    *,
    source_key: str,
    source_config: Mapping[str, Any],
    audit_config: Mapping[str, Any],
    client: BudgetedHttpClient,
    ledger: BudgetLedger,
    raw_root: Path,
    max_evidence_bytes: int,
) -> dict[str, Any]:
    """Audit one source serially and return only privacy-safe evidence."""

    repository = str(source_config["source"])
    revision = require_exact_revision(source_config["revision"])
    info_url = _hf_api_url(repository, revision)
    try:
        repo_info_value, repo_payload = client.get_json_payload(source_key, info_url)
    except AuditError as error:
        if "HTTP 404" in str(error):
            raise AuditError(BLOCKED_REVISION, "configured repository revision was not found") from error
        raise
    if not isinstance(repo_info_value, Mapping):
        raise AuditError(BLOCKED_SCHEMA, "HF repository metadata is not a mapping")
    resolved = repo_info_value.get("sha")
    require_matching_revision(revision, resolved)
    _write_raw_payload(raw_root, source_key, "repo-info", repo_payload)

    tree_url = _hf_tree_url(repository, revision)
    tree_value, tree_payload = client.get_json_payload(source_key, tree_url)
    _write_raw_payload(raw_root, source_key, "tree", tree_payload)
    tree = _tree_entries(repo_info_value, tree_value)
    if not tree:
        raise AuditError(BLOCKED_SCHEMA, "fixed-revision repository tree is empty")

    evidence: list[dict[str, Any]] = []
    evidence_bodies: list[bytes] = []
    for item in _evidence_candidates(tree)[:2]:
        size = item.get("size")
        if isinstance(size, int) and size > max_evidence_bytes:
            continue
        if not isinstance(size, int) and Path(str(item["path"])).name.casefold() != "readme.md":
            continue
        url = _hf_raw_url(repository, revision, str(item["path"]))
        try:
            payload = client.get_bytes(source_key, url)
        except AuditError:
            continue
        _write_raw_payload(raw_root, source_key, f"evidence-{Path(str(item['path'])).name}", payload)
        digest = _sha256_bytes(payload.body)
        evidence_bodies.append(payload.body)
        evidence.append(
            {
                "source": source_key,
                "subsource": None,
                "declared_license": str(source_config["license"]),
                "evidence_type": "fixed_revision_file",
                "evidence_url": url,
                "evidence_revision": revision,
                "evidence_file_sha256": digest,
                "audit_status": "evidence_collected",
                "notes": str(item["path"]),
            }
        )

    split_url = _datasets_server_url("splits", repository, revision)
    available_splits: list[dict[str, str]] = []
    split_metadata_status = "unavailable"
    split_metadata_sha256: str | None = None
    split_metadata_error: str | None = None
    try:
        split_value, split_payload = client.get_json_payload(source_key, split_url)
        _write_raw_payload(raw_root, source_key, "datasets-server-splits", split_payload)
        split_metadata_sha256 = _sha256_bytes(split_payload.body)
        split_revision = _payload_revision(split_value, split_payload)
        if split_revision is None:
            split_metadata_status = "revision_unbound"
        else:
            require_matching_revision(revision, split_revision)
            available_splits = _extract_splits(split_value)
            split_metadata_status = "fixed_revision_verified"
    except AuditError as error:
        split_metadata_error = error.status

    plans = _source_plans(source_key, audit_config)
    if sum(int(plan.get("limit", 0)) for plan in plans) > 50:
        raise AuditError(BLOCKED_DOWNLOAD_BUDGET, "source audit plan exceeds 50 records")
    expected_pairs = {
        f"{plan['config']}::{plan['split']}" for plan in plans
    }
    metadata_only_config = str(audit_config.get("metadata_only_config", "default"))
    for split in audit_config.get("metadata_only_splits", []):
        expected_pairs.add(f"{metadata_only_config}::{split}")
    available_pair_set = {
        f"{item['config']}::{item['split']}" for item in available_splits
    }
    missing_fixed_pairs = (
        sorted(expected_pairs - available_pair_set)
        if split_metadata_status == "fixed_revision_verified"
        else []
    )
    protocol_differences: list[str] = []
    expected_protocol_split = audit_config.get("formal_protocol_expected_split")
    if expected_protocol_split is not None:
        actual_splits = sorted({str(plan["split"]) for plan in plans})
        if actual_splits != [str(expected_protocol_split)]:
            protocol_differences.append(
                "formal protocol declares upstream split "
                f"{expected_protocol_split!r}, fixed-revision metadata exposes "
                f"audit split(s) {actual_splits!r}"
            )

    sample_results: list[dict[str, Any]] = []
    sample_errors: list[dict[str, str]] = []
    access_methods: set[str] = set()
    access_urls: set[str] = set()
    coig_matrix: list[dict[str, Any]] = []

    if source_key == "coig":
        for plan in plans:
            subsource = str(plan["subsource"])
            try:
                rows, raw_sha, method, access_url = _sample_plan_rows(
                    client=client,
                    ledger=ledger,
                    raw_root=raw_root,
                    source_key=source_key,
                    repository=repository,
                    revision=revision,
                    tree=tree,
                    config=str(plan["config"]),
                    split=str(plan["split"]),
                    limit=int(plan["limit"]),
                    range_file=(
                        str(plan["range_file"]) if plan.get("range_file") else None
                    ),
                    range_prefix_bytes=(
                        int(plan["range_prefix_mib"]) * 1024 * 1024
                        if plan.get("range_prefix_mib")
                        else None
                    ),
                    prefer_range_file=True,
                )
                one_stats, one_matrix = _audit_coig_rows(
                    source_config={**source_config, **audit_config},
                    plans=[plan],
                    rows=rows,
                    raw_sha256=raw_sha,
                    source_scoped=method in {
                        "fixed_revision_exact_file",
                        "fixed_revision_http_range",
                    },
                )
                sample_results.extend(one_stats)
                coig_matrix.extend(one_matrix)
                access_methods.add(method)
                access_urls.add(access_url)
            except AuditError as error:
                sample_errors.append(
                    {"status": error.status, "context": f"coig/{subsource}"}
                )
                _empty_stats, one_matrix = _audit_coig_rows(
                    source_config={**source_config, **audit_config},
                    plans=[plan],
                    rows=[],
                    raw_sha256="0" * 64,
                )
                coig_matrix.extend(one_matrix)
    else:
        remaining_records = 50
        for plan in plans:
            config_name = str(plan["config"])
            split_name = str(plan["split"])
            target_role = str(plan["target_role"])
            requested = int(plan["limit"])
            if requested > remaining_records:
                raise AuditError(BLOCKED_DOWNLOAD_BUDGET, "source row cap exceeded")
            try:
                rows, raw_sha, method, access_url = _sample_plan_rows(
                    client=client,
                    ledger=ledger,
                    raw_root=raw_root,
                    source_key=source_key,
                    repository=repository,
                    revision=revision,
                    tree=tree,
                    config=config_name,
                    split=split_name,
                    limit=requested,
                    range_file=(
                        str(plan["range_file"]) if plan.get("range_file") else None
                    ),
                    range_prefix_bytes=(
                        int(plan["range_prefix_mib"]) * 1024 * 1024
                        if plan.get("range_prefix_mib")
                        else None
                    ),
                )
                injected = [
                    _inject_audit_context(
                        row,
                        source_key=source_key,
                        split=split_name,
                        config=config_name,
                        subsource=None,
                        source_license=None,
                    )
                    for row in rows
                ]
                stats = audit_adapter_rows(
                    source_key=source_key,
                    source_config=source_config,
                    split=split_name,
                    target_role=target_role,
                    rows=injected,
                    raw_file_sha256=raw_sha,
                    max_records=requested,
                )
                sample_results.append(
                    {
                        "config": config_name,
                        "split": split_name,
                        "target_role": target_role,
                        "requested_count": requested,
                        "access_method": method,
                        **stats,
                    }
                )
                remaining_records -= len(rows)
                access_methods.add(method)
                access_urls.add(access_url)
            except AuditError as error:
                sample_errors.append(
                    {
                        "status": error.status,
                        "context": f"{config_name}/{split_name}",
                    }
                )

    merged = _merge_adapter_stats(sample_results)
    schema_only_accepted = sum(
        int(
            result.get("schema_only_if_license_verified", {}).get(
                "adapter_accepted", 0
            )
        )
        for result in sample_results
    )
    schema_only_dropped = sum(
        int(
            result.get("schema_only_if_license_verified", {}).get(
                "adapter_dropped", 0
            )
        )
        for result in sample_results
    )
    declared_license = str(source_config["license"])
    if source_key == "medqa_zh":
        license_status = "unknown"
        license_verified = False
    elif source_key == "coig":
        license_status = BLOCKED_LICENSE
        license_verified = False
    else:
        license_verified = any(
            _license_body_supports(declared_license, body) for body in evidence_bodies
        )
        license_status = "verified" if license_verified else BLOCKED_LICENSE

    requested_total = sum(int(plan["limit"]) for plan in plans)
    sample_complete = sample_audit_complete(
        requested_count=requested_total,
        sampled_count=merged["sampled_count"],
        accepted_count=merged["adapter_accepted"],
        dropped_count=merged["adapter_dropped"],
        sample_errors=sample_errors,
    )
    schema_ok = (
        sample_complete
        and not missing_fixed_pairs
        and not protocol_differences
    )
    if missing_fixed_pairs or protocol_differences:
        status = BLOCKED_SCHEMA
    elif sample_errors and not schema_ok:
        statuses = {item["status"] for item in sample_errors}
        if BLOCKED_DOWNLOAD_BUDGET in statuses:
            status = BLOCKED_DOWNLOAD_BUDGET
        elif BLOCKED_REVISION in statuses:
            status = BLOCKED_REVISION
        elif BLOCKED_NETWORK in statuses:
            status = BLOCKED_NETWORK
        else:
            status = "metadata_only"
    elif not schema_ok:
        status = BLOCKED_SCHEMA
    elif not license_verified:
        status = BLOCKED_LICENSE
    else:
        status = "verified"

    metadata_sha256 = hashlib.sha256(
        ( _sha256_bytes(repo_payload.body)
          + _sha256_bytes(tree_payload.body)
          + (split_metadata_sha256 or "")
        ).encode("ascii")
    ).hexdigest()
    configured_pairs = sorted(expected_pairs)
    available_pairs = sorted(
        f"{item['config']}::{item['split']}" for item in available_splits
    )
    tree_report = [
        {
            "path": str(item["path"]),
            "type": str(item.get("type", "file")),
            "size": item.get("size"),
            "oid": item.get("oid"),
        }
        for item in tree
    ]
    return {
        "source": source_key,
        "repository": repository,
        "configured_revision": revision,
        "resolved_revision": revision,
        "revision_status": "exact_match",
        "status": status,
        "formal_ready": status == "verified",
        "blocking_reason": None if status == "verified" else status,
        "declared_license": declared_license,
        "license_status": license_status,
        "license_evidence": evidence,
        "configs_and_splits_requested": configured_pairs,
        "configs_and_splits_available": available_pairs,
        "missing_config_or_split": missing_fixed_pairs,
        "protocol_differences": protocol_differences,
        "split_metadata_status": split_metadata_status,
        "split_metadata_error": split_metadata_error,
        "repo_tree": tree_report,
        "repo_tree_file_count": sum(item["type"] == "file" for item in tree_report),
        "metadata_sha256": metadata_sha256,
        "split_metadata_sha256": split_metadata_sha256,
        "sampled_count": merged["sampled_count"],
        "adapter_accepted": merged["adapter_accepted"],
        "adapter_dropped": merged["adapter_dropped"],
        "exact_duplicate_count": merged["exact_duplicate_count"],
        "schema_only_if_license_verified": {
            "adapter_accepted": schema_only_accepted,
            "adapter_dropped": schema_only_dropped,
            "status": (
                "verified_with_adapter_patch"
                if source_key == "coig"
                and schema_only_accepted == merged["sampled_count"]
                and merged["sampled_count"] > 0
                else "not_applicable"
            ),
        },
        "drop_reason_counts": merged["drop_reason_counts"],
        "raw_file_sha256s": merged["raw_file_sha256s"],
        "schema_fingerprint_sha256": merged["combined_schema_fingerprint_sha256"],
        "normalized_hash_summary_sha256": merged[
            "combined_normalized_hash_summary_sha256"
        ],
        "sample_audits": sample_results,
        "sample_errors": sample_errors,
        "access_methods": sorted(access_methods),
        "access_urls": sorted(access_urls),
        "coig_license_matrix": coig_matrix,
        "downloaded_bytes": ledger.source_bytes(source_key),
        "audit_only": True,
        "formal_final_manifest": False,
        "raw_location": "data/raw/audit_v2 (ignored)",
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def apply_evidence_boundary(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Label verified samples without claiming a formal dataset was built."""

    bounded = dict(audit)
    producer_status = str(
        dict(audit.get("producer_binding", {})).get("status", "unbound")
    )
    producer_verified = producer_status in {
        "verified_network_run",
        "verified_offline_raw_replay",
    }
    bounded_sources: dict[str, Any] = {}
    for key, raw_source in dict(audit.get("sources", {})).items():
        source = dict(raw_source)
        source["sample_audit_ready"] = (
            source.get("status") == "verified" and producer_verified
        )
        source["formal_ready"] = False
        source["formal_blocking_reason"] = "p1.5_small_sample_audit_only"
        bounded_sources[str(key)] = source
    bounded["sources"] = bounded_sources
    bounded["evidence_boundary"] = {
        "sample_audit_only": True,
        "formal_dataset_built": False,
        "formal_ready_for_training": False,
    }
    return bounded


def sanitize_audit_fingerprints(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Remove audit-only context fields from stored upstream schema evidence."""

    sanitized = dict(audit)
    sources: dict[str, Any] = {}
    for source_key, raw_source in dict(audit.get("sources", {})).items():
        source = dict(raw_source)
        sample_audits: list[dict[str, Any]] = []
        fingerprint_hashes: list[str] = []
        for raw_sample in source.get("sample_audits", []):
            sample = dict(raw_sample)
            raw_schema = sample.get("schema")
            if isinstance(raw_schema, Mapping):
                schema = dict(raw_schema)
                fields = schema.get("top_level_fields", {})
                if isinstance(fields, Mapping):
                    schema["top_level_fields"] = {
                        str(key): value
                        for key, value in fields.items()
                        if str(key) not in AUDIT_INJECTED_FIELDS
                    }
                paths = schema.get("nested_field_paths", [])
                if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
                    schema["nested_field_paths"] = [
                        str(path)
                        for path in paths
                        if str(path).split(".", 1)[0] not in AUDIT_INJECTED_FIELDS
                    ]
                schema.pop("schema_fingerprint_sha256", None)
                schema["schema_fingerprint_sha256"] = _canonical_sha256(schema)
                fingerprint_hashes.append(schema["schema_fingerprint_sha256"])
                sample["schema"] = schema
            sample_audits.append(sample)
        source["sample_audits"] = sample_audits
        source["schema_fingerprint_sha256"] = hashlib.sha256(
            "\n".join(sorted(fingerprint_hashes)).encode("ascii")
        ).hexdigest()
        sources[str(source_key)] = source
    sanitized["sources"] = sources
    return sanitized


def _implementation_sha256(repo_root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = repo_root / relative
        if not path.is_file():
            raise AuditError(BLOCKED_SCHEMA, f"implementation input is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _source_manifest_entry(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": source["source"],
        "repository": source["repository"],
        "configured_revision": source["configured_revision"],
        "resolved_revision": source.get("resolved_revision"),
        "config_subset": source.get("configs_and_splits_requested", []),
        "split": sorted(
            {
                item.split("::", 1)[1]
                for item in source.get("configs_and_splits_requested", [])
                if "::" in item
            }
        ),
        "metadata_sha256": source.get("metadata_sha256"),
        "schema_fingerprint_sha256": source.get("schema_fingerprint_sha256"),
        "license_status": source.get("license_status"),
        "sampled_count": source.get("sampled_count", 0),
        "adapter_accepted": source.get("adapter_accepted", 0),
        "adapter_dropped": source.get("adapter_dropped", 0),
        "raw_location": "data/raw/audit_v2 (ignored)",
        "formal_ready": bool(source.get("formal_ready", False)),
        "blocking_reason": source.get("blocking_reason"),
        "sample_audit_ready": bool(source.get("sample_audit_ready", False)),
        "formal_blocking_reason": source.get("formal_blocking_reason"),
    }


def _write_markdown_reports(
    *,
    report_root: Path,
    audit: Mapping[str, Any],
) -> None:
    sources = audit["sources"]
    session_evidence = audit.get("session_transfer_evidence", {})
    cumulative_sources = session_evidence.get("source_bytes", {})
    lines = [
        "# P1.5 pinned upstream audit",
        "",
        "> Metadata/schema/license audit only. No model, tokenizer, training, or final evaluation was run.",
        "",
        f"- Audit version: `{audit['audit_version']}`",
        f"- Data protocol: `{audit['data_protocol_version']}`",
        f"- Network bytes: {audit['download_budget']['used_bytes']} / {audit['download_budget']['total_bytes']}",
        f"- Cumulative session network bytes: {session_evidence.get('total_bytes', 'not-recorded')}.",
        "- Cumulative per-source bytes: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(cumulative_sources.items())
        )
        + ".",
        "- `actual_cost_cny`: `null` (this is not a training-cost measurement).",
        "",
        "| source | configured / resolved revision | sampled | accepted / dropped | license | status | formal ready |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for key in ("medical_o1", "cmb", "medqa_zh", "coig", "ceval"):
        source = sources[key]
        lines.append(
            f"| {key} | `{source['configured_revision']}` / "
            f"`{source.get('resolved_revision') or 'unresolved'}` | "
            f"{source.get('sampled_count', 0)} | "
            f"{source.get('adapter_accepted', 0)} / {source.get('adapter_dropped', 0)} | "
            f"{source.get('license_status', 'unknown')} | {source['status']} | "
            f"{str(bool(source.get('formal_ready'))).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Reports contain field names, types, counters, lengths, hashes, repository paths, and license evidence metadata only.",
            "They do not contain question, reasoning, answer, option, patient, entity, or complete raw-record text.",
            "MedQA audit rows are not a final manifest; no primary final 600 IDs were frozen.",
            "",
            "## Blocking semantics",
            "",
            "A blocked source was not switched to `main` or `latest`. Other sources continued serially.",
        ]
    )
    (report_root / "upstream_audit_v2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    matrix = sources["coig"].get("coig_license_matrix", [])
    matrix_lines = [
        "# COIG subsource license matrix (P1.5)",
        "",
        "> The umbrella repository metadata is not treated as proof for a mixed dataset's subsources.",
        "",
        "| quota category | declared/candidate license | evidence | research training | derived manifest | decision | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in matrix:
        matrix_lines.append(
            f"| {item['subsource']} | {item['declared_license']} | "
            f"{item.get('evidence_type') or 'none'} | "
            f"{str(bool(item.get('research_training_allowed'))).lower()} | "
            f"{str(bool(item.get('derived_manifest_allowed'))).lower()} | "
            f"{item['decision']} | {item['notes']} |"
        )
    matrix_lines.extend(
        [
            "",
            "## Formal anchor decision",
            "",
            "COIG formal anchors remain blocked unless every selected subsource obtains source-specific license evidence.",
            "The translated/exam quota is not relaxed to manufacture 4,000 rows.",
            "",
            "At most two follow-up options are retained for user decision:",
            "",
            "1. Obtain and pin authoritative upstream license files for the existing four quota categories.",
            "2. If closure is impossible, propose a separate ADR for a replacement source without changing Data Protocol v2 silently.",
        ]
    )
    (report_root / "coig_license_matrix_v2.md").write_text(
        "\n".join(matrix_lines) + "\n", encoding="utf-8"
    )


def _write_handoff(repo_root: Path, audit: Mapping[str, Any]) -> None:
    sources = audit["sources"]
    session_evidence = audit.get("session_transfer_evidence", {})
    cumulative_sources = session_evidence.get("source_bytes", {})
    lines = [
        "# P1.5 upstream audit handoff",
        "",
        "## Scope",
        "",
        "This milestone queried the five configured immutable revisions under strict byte/row limits and reused the existing Data Protocol v2 adapters.",
        "It did not download a complete dataset, model, or tokenizer; it did not run SFT, OPD, controller evaluation, or final evaluation.",
        "",
        "## Source outcomes",
        "",
    ]
    for key in ("medical_o1", "cmb", "medqa_zh", "coig", "ceval"):
        source = sources[key]
        lines.append(
            f"- `{key}`: `{source['status']}`; sampled={source.get('sampled_count', 0)}, "
            f"accepted={source.get('adapter_accepted', 0)}, dropped={source.get('adapter_dropped', 0)}, "
            f"formal_ready={str(bool(source.get('formal_ready'))).lower()}."
        )
    lines.extend(
        [
            "",
            "## Evidence that is not established",
            "",
            "- No full-corpus near-duplicate scan or streaming build was run.",
            "- No tokenizer length, GPU memory, throughput, veRL/vLLM/Ray, cost, quality, or final-test result was measured.",
            "- MedQA remains license `unknown`; COIG unknown subsource licenses fail closed.",
            "",
            "## Resource evidence",
            "",
            f"- Cumulative bounded network transfer: {session_evidence.get('total_bytes', 'not-recorded')} bytes.",
            "- Per-source transfer: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(cumulative_sources.items())
            )
            + ".",
            f"- Attempt logs are bound by {len(session_evidence.get('attempt_log_sha256', []))} SHA256 values in the versioned audit config/report.",
            "- `actual_cost_cny` is `null`; this was not a paid training run.",
            "",
            "## Next decision",
            "",
            "Enter P2 formal streaming-builder work only for sources whose audit is sufficient; otherwise resolve the recorded COIG/license/revision blocker first.",
        ]
    )
    path = repo_root / "docs" / "experiments" / "p1_5_upstream_audit_handoff.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_upstream_audit(config_path: str | Path) -> dict[str, Any]:
    """Execute the five-source audit serially and write redacted evidence."""

    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    repo_root = Path(__file__).resolve().parents[2]
    config = _load_yaml(config_file)
    if config.get("audit_version") != AUDIT_VERSION:
        raise AuditError(BLOCKED_SCHEMA, "unsupported audit version")
    if config.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise AuditError(BLOCKED_SCHEMA, "audit config must use Data Protocol v2")
    sources_file = (config_dir / str(config["sources_config"])).resolve()
    sources_config = _load_yaml(sources_file)
    if sources_config.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise AuditError(BLOCKED_SCHEMA, "source config must use Data Protocol v2")

    raw_root = (config_dir / str(config["raw_root"])).resolve()
    report_root = (config_dir / str(config["report_root"])).resolve()
    manifest_root = (config_dir / str(config["manifest_root"])).resolve()
    prospective_raw = raw_root / "ignore-contract-check.raw"
    if not is_path_gitignored(repo_root, prospective_raw):
        raise AuditError(BLOCKED_SCHEMA, "raw audit directory is not gitignored")

    limits_config = config["limits"]
    limits = AuditLimits(
        per_source_bytes=int(limits_config["per_source_mib"]) * 1024 * 1024,
        total_bytes=int(limits_config["total_mib"]) * 1024 * 1024,
        max_records_per_source=int(limits_config["max_records_per_source"]),
        timeout_seconds=int(limits_config["timeout_seconds"]),
        max_retries=int(limits_config["max_retries"]),
    )
    max_evidence_bytes = int(limits_config["max_evidence_file_mib"]) * 1024 * 1024
    ledger = BudgetLedger(limits)
    client = BudgetedHttpClient(
        transport=UrlLibTransport(),
        ledger=ledger,
        limits=limits,
    )

    results: dict[str, Any] = {}
    ordered_sources = ("medical_o1", "cmb", "medqa_zh", "coig", "ceval")
    for source_key in ordered_sources:
        source = sources_config["sources"][source_key]
        try:
            results[source_key] = audit_one_source(
                source_key=source_key,
                source_config=source,
                audit_config=config["sources"][source_key],
                client=client,
                ledger=ledger,
                raw_root=raw_root,
                max_evidence_bytes=max_evidence_bytes,
            )
        except AuditError as error:
            results[source_key] = {
                "source": source_key,
                "repository": str(source["source"]),
                "configured_revision": str(source["revision"]),
                "resolved_revision": None,
                "status": error.status,
                "formal_ready": False,
                "blocking_reason": error.status,
                "public_error": error.public_details,
                "declared_license": str(source["license"]),
                "license_status": "unknown" if source_key == "medqa_zh" else "unverified",
                "sampled_count": 0,
                "adapter_accepted": 0,
                "adapter_dropped": 0,
                "downloaded_bytes": ledger.source_bytes(source_key),
                "audit_only": True,
                "formal_final_manifest": False,
                "raw_location": "data/raw/audit_v2 (ignored)",
                "configs_and_splits_requested": [],
            }

    audit_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    audit_sha = git_sha(repo_root)
    dirty = git_dirty(repo_root)
    if not _SHA40.fullmatch(audit_sha) or dirty is None:
        raise AuditError(BLOCKED_SCHEMA, "cannot establish audit Git state")
    audit = {
        "audit_version": AUDIT_VERSION,
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "audit_git_sha": audit_sha,
        "audit_dirty_worktree": dirty,
        "audit_time": audit_time,
        "actual_cost_cny": None,
        "execution": {
            "mode": "real_network_small_sample",
            "serial_sources": True,
            "model_downloaded": False,
            "tokenizer_downloaded": False,
            "training_run": False,
            "final_evaluation_run": False,
        },
        "config_sha256": _sha256_bytes(config_file.read_bytes()),
        "sources_config_sha256": _sha256_bytes(sources_file.read_bytes()),
        "audit_implementation_sha256": _implementation_sha256(
            repo_root, AUDIT_IMPLEMENTATION_INPUTS
        ),
        "download_budget": {
            "per_source_bytes": limits.per_source_bytes,
            "total_bytes": limits.total_bytes,
            "used_bytes": ledger.total_bytes,
            "source_bytes": {
                key: ledger.source_bytes(key) for key in ordered_sources
            },
        },
        "sources": results,
    }
    audit["producer_binding"] = {
        "status": "verified_network_run",
        "git_sha": audit_sha,
        "dirty_worktree": dirty,
        "config_sha256": audit["config_sha256"],
        "sources_config_sha256": audit["sources_config_sha256"],
        "implementation_sha256": audit["audit_implementation_sha256"],
        "raw_evidence_replayed": False,
    }
    if isinstance(config.get("session_transfer_evidence"), Mapping):
        audit["session_transfer_evidence"] = validate_session_transfer_evidence(
            config["session_transfer_evidence"]
        )
    audit = apply_evidence_boundary(sanitize_audit_fingerprints(audit))
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / "upstream_audit_v2.json", audit)
    _write_json(
        report_root / "network_session_evidence_v2.json",
        {
            "audit_version": AUDIT_VERSION,
            "evidence_type": "redacted_network_session_accounting",
            "session_transfer_evidence": audit.get("session_transfer_evidence"),
            "actual_cost_cny": None,
        },
    )
    _write_json(
        report_root / "schema_fingerprint_v2.json",
        {
            "audit_version": AUDIT_VERSION,
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "sources": {
                key: {
                    "schema_fingerprint_sha256": value.get(
                        "schema_fingerprint_sha256"
                    ),
                    "sample_audits": value.get("sample_audits", []),
                }
                for key, value in audit["sources"].items()
            },
        },
    )
    manifest = {
        "audit_version": AUDIT_VERSION,
        "audit_git_sha": audit_sha,
        "audit_dirty_worktree": dirty,
        "audit_time": audit_time,
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "actual_cost_cny": None,
        "config_sha256": audit["config_sha256"],
        "sources_config_sha256": audit["sources_config_sha256"],
        "audit_implementation_sha256": audit["audit_implementation_sha256"],
        "producer_binding": audit["producer_binding"],
        "finalizer_binding": audit.get("finalizer_binding"),
        "evidence_boundary": audit["evidence_boundary"],
        "session_transfer_evidence": audit.get("session_transfer_evidence"),
        "sources": [
            _source_manifest_entry(audit["sources"][key]) for key in ordered_sources
        ],
    }
    _write_json(manifest_root / "source_revision_manifest.json", manifest)
    _write_markdown_reports(report_root=report_root, audit=audit)
    _write_handoff(repo_root, audit)
    return audit


def finalize_existing_audit(config_path: str | Path) -> dict[str, Any]:
    """Replay hash-bound raw evidence and refresh reports without network access."""

    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    repo_root = Path(__file__).resolve().parents[2]
    config = _load_yaml(config_file)
    sources_file = (config_dir / str(config["sources_config"])).resolve()
    sources_config = _load_yaml(sources_file)
    raw_root = (config_dir / str(config["raw_root"])).resolve()
    report_root = (config_dir / str(config["report_root"])).resolve()
    manifest_root = (config_dir / str(config["manifest_root"])).resolve()
    report_path = report_root / "upstream_audit_v2.json"
    if not report_path.is_file():
        raise AuditError(BLOCKED_SCHEMA, "existing upstream audit report is missing")
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("audit_version") != AUDIT_VERSION:
        raise AuditError(BLOCKED_SCHEMA, "existing upstream audit report is incompatible")
    audit = dict(value)
    replayed_sources: dict[str, Any] = {}
    for source_key, raw_source in dict(audit.get("sources", {})).items():
        source = dict(raw_source)
        if source.get("sample_audits"):
            replayed = replay_source_sample_audits(
                source_key=str(source_key),
                source_config=sources_config["sources"][source_key],
                audit_config=config["sources"][source_key],
                source_result=source,
                raw_root=raw_root,
            )
            source = bind_replayed_source(source, replayed)
        elif source.get("status") == "verified":
            raise AuditError(
                BLOCKED_SCHEMA,
                f"verified source {source_key} has no raw sample evidence to replay",
            )
        replayed_sources[str(source_key)] = source
    audit["sources"] = replayed_sources
    audit["audit_git_sha"] = git_sha(repo_root)
    audit["audit_dirty_worktree"] = git_dirty(repo_root)
    audit["config_sha256"] = _sha256_bytes(config_file.read_bytes())
    audit["sources_config_sha256"] = _sha256_bytes(sources_file.read_bytes())
    audit["audit_implementation_sha256"] = _implementation_sha256(
        repo_root, AUDIT_IMPLEMENTATION_INPUTS
    )
    audit["producer_binding"] = {
        "status": "verified_offline_raw_replay",
        "git_sha": audit["audit_git_sha"],
        "dirty_worktree": audit["audit_dirty_worktree"],
        "config_sha256": audit["config_sha256"],
        "sources_config_sha256": audit["sources_config_sha256"],
        "implementation_sha256": audit["audit_implementation_sha256"],
        "raw_evidence_replayed": True,
        "replayed_sources": sorted(
            key
            for key, source in replayed_sources.items()
            if source.get("sample_audits")
        ),
    }
    audit["finalizer_binding"] = {
        "git_sha": audit["audit_git_sha"],
        "implementation_sha256": audit["audit_implementation_sha256"],
    }
    if isinstance(config.get("session_transfer_evidence"), Mapping):
        audit["session_transfer_evidence"] = validate_session_transfer_evidence(
            config["session_transfer_evidence"]
        )
    audit = apply_evidence_boundary(sanitize_audit_fingerprints(audit))
    _write_json(report_path, audit)
    _write_json(
        report_root / "network_session_evidence_v2.json",
        {
            "audit_version": AUDIT_VERSION,
            "evidence_type": "redacted_network_session_accounting",
            "session_transfer_evidence": audit.get("session_transfer_evidence"),
            "actual_cost_cny": None,
        },
    )
    _write_json(
        report_root / "schema_fingerprint_v2.json",
        {
            "audit_version": AUDIT_VERSION,
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "sources": {
                key: {
                    "schema_fingerprint_sha256": value.get(
                        "schema_fingerprint_sha256"
                    ),
                    "sample_audits": value.get("sample_audits", []),
                }
                for key, value in audit["sources"].items()
            },
        },
    )
    ordered_sources = ("medical_o1", "cmb", "medqa_zh", "coig", "ceval")
    manifest = {
        "audit_version": AUDIT_VERSION,
        "audit_git_sha": audit["audit_git_sha"],
        "audit_dirty_worktree": audit["audit_dirty_worktree"],
        "audit_time": audit["audit_time"],
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "actual_cost_cny": None,
        "config_sha256": audit["config_sha256"],
        "sources_config_sha256": audit["sources_config_sha256"],
        "audit_implementation_sha256": audit["audit_implementation_sha256"],
        "producer_binding": audit["producer_binding"],
        "finalizer_binding": audit.get("finalizer_binding"),
        "evidence_boundary": audit["evidence_boundary"],
        "session_transfer_evidence": audit.get("session_transfer_evidence"),
        "sources": [
            _source_manifest_entry(audit["sources"][key]) for key in ordered_sources
        ],
    }
    _write_json(manifest_root / "source_revision_manifest.json", manifest)
    return audit
