"""Streaming, aggregate-only audits for immutable Controller v1 and SFT targets.

This module deliberately has no model, tokenizer, torch, CUDA, vLLM or dataset
dependency. Raw responses and training text are inputs in ignored storage; its
outputs contain only counts, hashes, length summaries and redacted pattern names.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


class AuditIntegrityError(RuntimeError):
    """Raised when frozen evidence is altered or unsafe to summarize."""


_EXPLICIT = re.compile(r"(?:最终答案|答案)\s*[:：是为]?\s*([A-Ea-e])")
_ANSWER_MARKER = re.compile(r"(?:最终答案|答案|综上)")
_RAW_KEYS = frozenset(
    {"question", "options", "answer", "reasoning", "response", "prediction", "gold"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditIntegrityError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise AuditIntegrityError(f"non-object JSONL at {path}:{line_number}")
            yield row


def _length_summary(values: Iterable[int], *, status: str = "available") -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"status": "unavailable", "count": 0}

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "status": status,
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _invalid_class(row: Mapping[str, Any], *, max_new_tokens: int) -> str:
    text = str(row.get("response") or "").strip()
    if not text:
        return "empty_output"
    if row.get("finish_reason") == "length" or (
        type(row.get("generated_token_count")) is int
        and int(row["generated_token_count"]) >= max_new_tokens
    ):
        return "length_truncated"
    if row.get("parse_method") == "ambiguous":
        return "conflicting_answers"
    explicit = [match.upper() for match in _EXPLICIT.findall(text)]
    if len(set(explicit)) > 1:
        return "conflicting_answers"
    if explicit or re.fullmatch(r"[A-Ea-e]", text):
        return "parser_false_negative"
    if _ANSWER_MARKER.search(text):
        return "malformed_explicit_answer"
    if len(text) >= 1:
        return "no_explicit_answer"
    return "other"


def _audit_one(
    path: Path,
    *,
    max_new_tokens: int,
    subject_by_id: Mapping[str, str],
    token_count_fn: Callable[[str], int] | None,
) -> tuple[dict[str, Any], set[str]]:
    total = valid = correct = 0
    ids: set[str] = set()
    domains: dict[str, Counter[str]] = {}
    subjects: dict[str, Counter[str]] = {}
    invalid_classes: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    reached_limit = empty = think_tags = reasoning_without_answer = 0
    explicit_unrecognized = single_letter_unrecognized = 0
    for row in _iter_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in ids:
            raise AuditIntegrityError(f"missing/duplicate sample_id in {path}")
        ids.add(sample_id)
        total += 1
        parsed = row.get("parsed") is True
        is_correct = row.get("correct") is True
        valid += int(parsed)
        correct += int(is_correct)
        domain = str(row.get("domain") or "unknown")
        counter = domains.setdefault(domain, Counter())
        counter.update(total=1, valid=int(parsed), correct=int(is_correct), invalid=int(not parsed))
        if domain == "general":
            subject = str(subject_by_id.get(sample_id, "unknown"))
            subject_counter = subjects.setdefault(subject, Counter())
            subject_counter.update(total=1, valid=int(parsed), correct=int(is_correct), invalid=int(not parsed))
        response = str(row.get("response") or "")
        char_lengths.append(len(response))
        think_tags += int("<think>" in response.casefold() or "</think>" in response.casefold())
        finish_reason = str(row.get("finish_reason") or "unavailable")
        finish_reasons[finish_reason] += 1
        token_count = (
            int(token_count_fn(response))
            if token_count_fn is not None
            else row.get("generated_token_count")
        )
        if type(token_count) is int:
            token_lengths.append(int(token_count))
            reached_limit += int(int(token_count) >= max_new_tokens)
        empty += int(not response.strip())
        if not parsed:
            category = _invalid_class(row, max_new_tokens=max_new_tokens)
            invalid_classes[category] += 1
            explicit_unrecognized += int(category in {"parser_false_negative", "malformed_explicit_answer"})
            single_letter_unrecognized += int(
                category == "parser_false_negative"
                and bool(re.fullmatch(r"[A-Ea-e]", response.strip()))
            )
            reasoning_without_answer += int(
                category in {"no_explicit_answer", "length_truncated"}
                and len(re.findall(r"[\u4e00-\u9fff]", response)) >= 8
                and not _EXPLICIT.search(response)
            )
    if not total:
        raise AuditIntegrityError(f"zero predictions in {path}")

    def summarize_group(counter: Counter[str]) -> dict[str, Any]:
        return {
            "total": counter["total"],
            "valid": counter["valid"],
            "invalid": counter["invalid"],
            "correct": counter["correct"],
            "accuracy": counter["correct"] / counter["total"],
            "parseable_accuracy": (
                counter["correct"] / counter["valid"] if counter["valid"] else None
            ),
        }

    token_status = "available" if len(token_lengths) == total else "partial"
    if not token_lengths:
        token_status = "unavailable"
    return (
        {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "correct": correct,
            "accuracy": correct / total,
            "parseable_accuracy": correct / valid if valid else None,
            "domains": {name: summarize_group(value) for name, value in sorted(domains.items())},
            "subjects": {name: summarize_group(value) for name, value in sorted(subjects.items())},
            "invalid_classes": dict(sorted(invalid_classes.items())),
            "finish_reason_distribution": dict(sorted(finish_reasons.items())),
            "generated_character_length": _length_summary(char_lengths),
            "generated_token_length": _length_summary(token_lengths, status=token_status),
            "token_length_source": (
                "retokenized_response"
                if token_count_fn is not None
                else "artifact_generated_token_count"
                if token_lengths
                else "unavailable"
            ),
            "reached_32_token_limit": reached_limit,
            "empty_output_count": empty,
            "explicit_answer_unrecognized_count": explicit_unrecognized,
            "single_letter_unrecognized_count": single_letter_unrecognized,
            "reasoning_without_explicit_answer_pattern_count": reasoning_without_answer,
            "think_tag_count": think_tags,
        },
        ids,
    )


def audit_controller_pair(
    b0_path: str | Path,
    b1_path: str | Path,
    *,
    expected_sha256: Mapping[str, str],
    subject_by_id: Mapping[str, str] | None = None,
    max_new_tokens: int = 32,
    token_count_fn: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Audit immutable B0/B1 JSONL without retaining response text."""

    paths = {"B0": Path(b0_path), "B1": Path(b1_path)}
    for name, path in paths.items():
        if _sha256(path) != str(expected_sha256.get(name, "")):
            raise AuditIntegrityError(f"{name} prediction SHA-256 mismatch")
    subjects = subject_by_id or {}
    b0, ids0 = _audit_one(
        paths["B0"], max_new_tokens=max_new_tokens,
        subject_by_id=subjects, token_count_fn=token_count_fn,
    )
    b1, ids1 = _audit_one(
        paths["B1"], max_new_tokens=max_new_tokens,
        subject_by_id=subjects, token_count_fn=token_count_fn,
    )
    if ids0 != ids1:
        raise AuditIntegrityError("B0/B1 sample_id sets differ")
    report = {
        "audit_version": "p3.1-v1-audit-v1",
        "protocol": "controller_protocol_v1_strict32",
        "source_prediction_sha256": {
            "B0": expected_sha256["B0"],
            "B1": expected_sha256["B1"],
        },
        "same_sample_ids": True,
        "B0": b0,
        "B1": b1,
        "evidence_boundary": {
            "finish_reason": "available only when present in each record",
            "token_length": (
                "retokenized from stored response with an injected tokenizer; original token IDs unavailable"
                if token_count_fn is not None
                else "available only when generated_token_count is present"
            ),
            "character_length": "complete but is not a token-count substitute",
            "reasoning_without_answer": "text-pattern evidence, not proof of truncation",
        },
    }
    assert_redacted_report(report)
    return report


def audit_sft_target_style(
    records_path: str | Path,
    *,
    seed: int,
    max_samples: int,
    token_count_fn: Callable[[str], int] | None = None,
    input_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically sample bounded SFT rows while streaming the population."""

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    records = Path(records_path)
    actual_records_sha256 = _sha256(records)
    evidence = dict(input_evidence or {})
    declared_records_sha = evidence.get("records_sha256")
    if declared_records_sha is not None and declared_records_sha != actual_records_sha256:
        raise AuditIntegrityError("SFT records SHA-256 differs from declared evidence")
    evidence["records_sha256"] = actual_records_sha256
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    population = 0
    for row in _iter_jsonl(records):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen_ids:
            raise AuditIntegrityError("SFT sample_id is missing or duplicated")
        seen_ids.add(sample_id)
        population += 1
        rank = int.from_bytes(
            hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest(), "big"
        )
        item = (-rank, sample_id, row)
        if len(heap) < max_samples:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    if not selected:
        raise AuditIntegrityError("zero SFT rows")

    response_lengths: list[int] = []
    reasoning_lengths: list[int] = []
    answer_lengths: list[int] = []
    response_token_lengths: list[int] = []
    answer_start_tokens: list[int] = []
    reasoning_before = marker_within_32 = long_reasoning_first = think = duplicate_response = 0
    answer_within_32_tokens = 0
    source_mapping_ok = True
    for row in selected:
        reasoning = str(row.get("reasoning") or "").strip()
        answer = str(row.get("answer") or "").strip()
        completion = "\n".join(value for value in (reasoning, answer) if value)
        response_lengths.append(len(completion))
        reasoning_lengths.append(len(reasoning))
        answer_lengths.append(len(answer))
        if token_count_fn is not None:
            response_token_lengths.append(int(token_count_fn(completion)))
            prefix = reasoning + ("\n" if reasoning and answer else "")
            answer_start = int(token_count_fn(prefix))
            answer_start_tokens.append(answer_start)
            answer_within_32_tokens += int(answer_start < 32)
        elif type(row.get("token_count_response")) is int:
            response_token_lengths.append(int(row["token_count_response"]))
        reasoning_before += int(bool(reasoning and answer and completion.startswith(reasoning)))
        marker = _ANSWER_MARKER.search(completion)
        marker_within_32 += int(bool(marker and marker.start() < 32))
        long_reasoning_first += int(bool(reasoning and len(reasoning) > len(answer)))
        think += int("<think>" in completion.casefold() or "</think>" in completion.casefold())
        duplicate_response += int(any(key in row for key in ("response", "completion")))
        source_mapping_ok = source_mapping_ok and bool(
            row.get("source") == "FreedomIntelligence/medical-o1-reasoning-SFT"
            and reasoning
            and answer
        )
    report = {
        "audit_version": "p3.1-sft-style-v1",
        "seed": seed,
        "population_seen": population,
        "sampled_count": len(selected),
        "assistant_response_character_length": _length_summary(response_lengths),
        "reasoning_character_length": _length_summary(reasoning_lengths),
        "final_answer_character_length": _length_summary(answer_lengths),
        "assistant_response_token_length": _length_summary(response_token_lengths),
        "answer_start_token": _length_summary(answer_start_tokens),
        "reasoning_before_answer_count": reasoning_before,
        "answer_marker_within_first_32_characters_count": marker_within_32,
        "answer_within_first_32_tokens_count": (
            answer_within_32_tokens if token_count_fn is not None else None
        ),
        "long_reasoning_before_answer_count": long_reasoning_first,
        "think_tag_count": think,
        "duplicate_response_field_count": duplicate_response,
        "mapping": {
            "upstream_fields": ["Question", "Complex_CoT", "Response"],
            "normalized_fields": ["question", "reasoning", "answer"],
            "medical_o1_fields_correct": source_mapping_ok,
        },
        "formatter": {
            "mode": "qwen3_nonthinking",
            "completion_order": ["reasoning", "answer", "assistant_eos"],
            "reasoning_is_ordinary_assistant_text": True,
        },
        "tokenizer_audit": (
            "retokenized_with_injected_tokenizer"
            if token_count_fn is not None
            else "formal_record_token_counts_only"
            if len(response_token_lengths) == len(selected)
            else "pending_not_loaded"
        ),
        "input_evidence": evidence,
    }
    assert_redacted_report(report)
    return report


def assert_redacted_report(
    payload: Any, *, forbidden_values: Iterable[str] = ()
) -> None:
    """Fail if a versioned aggregate report contains raw-content fields/values."""

    forbidden = tuple(value for value in forbidden_values if value)

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in _RAW_KEYS:
                    raise AuditIntegrityError(f"raw-text field is forbidden: {key}")
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            if any(item in value for item in forbidden):
                raise AuditIntegrityError("report contains forbidden raw text")

    walk(payload)
