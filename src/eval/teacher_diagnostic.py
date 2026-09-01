"""Label-blind post-hoc diagnostics for frozen Controller v2 generations.

This module never changes Controller v2 parsing or Teacher readiness.  It
recognizes a preregistered set of explicit conclusion forms in already-frozen
generation artifacts, then emits compact prediction rows.  Gold labels belong
to a separate scoring boundary invoked only after the parser manifest is
frozen and model execution has ended.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


METRIC_NAME = "generation_reparse_diagnostic_v1"
_LETTERS = "ABCDE"
_SUPERVISION_FIELDS = frozenset(
    {
        "answer",
        "answer_idx",
        "answer_index",
        "correct",
        "expected_label",
        "gold",
        "label",
        "solution",
    }
)
_EXPLICIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "final_answer_marker",
        re.compile(r"(?:^|\n)\s*最终答案\s*[：:]\s*([A-E])(?=\s|[。\.，,；;]|$)", re.I),
    ),
    (
        "correct_answer_marker",
        re.compile(
            r"正确答案\s*(?:是|为|[：:])\s*(?:选项\s*)?([A-E])(?=\s|[。\.，,；;]|$)",
            re.I,
        ),
    ),
    (
        "correct_option_marker",
        re.compile(
            r"正确选项\s*(?:是|为|[：:])\s*([A-E])(?=\s|[。\.，,；;]|$)",
            re.I,
        ),
    ),
    (
        "answer_is_marker",
        re.compile(
            r"(?<!正确)答案\s*(?:是|为|[：:])\s*(?:选项\s*)?([A-E])(?=\s|[。\.，,；;]|$)",
            re.I,
        ),
    ),
    (
        "choose_option_marker",
        re.compile(
            r"(?:因此|所以|综上|故)(?:，|,)?\s*(?:应|可)?\s*(?:选择|选)\s*(?:选项\s*)?[：:]?\s*([A-E])(?=\s|[。\.，,；;]|$)",
            re.I,
        ),
    ),
)
_OPTION_TEXT_CONCLUSION = re.compile(
    r"(?:^|\n).*?(?:最终答案|正确答案|正确选项|答案)\s*(?:是|为|[：:])\s*(.+?)\s*[。\.]?\s*$",
    re.I,
)


class DiagnosticError(RuntimeError):
    """Fail-closed diagnostic protocol violation."""


@dataclass(frozen=True)
class DiagnosticParse:
    letter: str | None
    method: str
    markers: tuple[str, ...]
    multiple_consistent: bool = False
    conflicting: bool = False
    thinking_then_answer: bool = False

    @property
    def valid(self) -> bool:
        return self.letter is not None and not self.conflicting

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_label_blind_row(row: Mapping[str, Any]) -> None:
    leaked = _SUPERVISION_FIELDS & set(row)
    if leaked:
        raise DiagnosticError(f"diagnostic discovery row contains supervision fields: {sorted(leaked)}")
    sample_id = str(row.get("sample_id") or "")
    role = str(row.get("target_role") or "")
    if not sample_id:
        raise DiagnosticError("diagnostic row requires sample_id")
    if role not in {"medical_controller_dev", "general_controller_dev"} or "final" in role:
        raise DiagnosticError("diagnostic row must use a non-final controller role")
    if "response" not in row:
        raise DiagnosticError("diagnostic discovery row requires the frozen response")


def _normalize_option_text(value: str) -> str:
    return re.sub(r"[\s。\.，,；;：:]", "", value).casefold()


def parse_generation_diagnostic(
    response: str | None,
    *,
    legal_labels: Sequence[str],
    options: Sequence[str] | None = None,
    truncated: bool = False,
) -> DiagnosticParse:
    """Parse only explicit conclusion structures, without a gold label."""

    labels = tuple(str(label).upper() for label in legal_labels)
    if labels not in (tuple("ABCD"), tuple("ABCDE")):
        raise DiagnosticError("diagnostic parser requires the complete ABCD/ABCDE legal set")
    if options is not None and len(options) != len(labels):
        raise DiagnosticError("diagnostic options and legal labels are misaligned")
    text = str(response or "").strip()
    if not text:
        return DiagnosticParse(None, "no_answer_expression", ())

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    found: list[tuple[int, str, str]] = []
    if lines:
        first = re.match(r"^答案\s*[：:]\s*([A-E])(?=\s|[。\.，,；;]|$)", lines[0], re.I)
        if first:
            found.append((text.find(lines[0]), first.group(1).upper(), "first_line_answer"))
    for method, pattern in _EXPLICIT_PATTERNS:
        for match in pattern.finditer(text):
            position = match.start(1)
            label = match.group(1).upper()
            # A first-line answer is a more precise classification than the
            # generic answer marker at the same response position.
            if method == "answer_is_marker" and lines and match.group(0).strip().startswith("答案"):
                first_match = re.match(
                    r"^答案\s*[：:]\s*([A-E])(?=\s|[。\.，,；;]|$)", lines[0], re.I
                )
                if first_match and label == first_match.group(1).upper():
                    continue
            found.append((position, label, method))

    if lines and re.fullmatch(r"[A-E]", lines[-1], re.I):
        found.append((text.rfind(lines[-1]), lines[-1].upper(), "standalone_last_label"))

    if options is not None:
        normalized = [_normalize_option_text(str(option)) for option in options]
        for match in _OPTION_TEXT_CONCLUSION.finditer(text):
            candidate = _normalize_option_text(match.group(1))
            matches = [index for index, option in enumerate(normalized) if candidate == option]
            if len(matches) == 1:
                found.append((match.start(1), labels[matches[0]], "option_text_after_conclusion"))

    # Deduplicate regexes that describe the same marker location and label.
    deduped: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in sorted(found, key=lambda value: value[0]):
        identity = (item[0], item[1])
        if identity not in seen:
            seen.add(identity)
            deduped.append(item)
    illegal = [item for item in deduped if item[1] not in labels]
    if illegal:
        return DiagnosticParse(None, "invalid_option", tuple(item[2] for item in deduped))
    unique = {item[1] for item in deduped}
    if len(unique) > 1:
        return DiagnosticParse(
            None,
            "conflicting_answers",
            tuple(item[2] for item in deduped),
            conflicting=True,
            thinking_then_answer="<think>" in text or "</think>" in text,
        )
    if not deduped:
        return DiagnosticParse(
            None,
            "truncated_before_answer" if truncated else "no_answer_expression",
            (),
            thinking_then_answer=False,
        )
    method = "multiple_consistent_answers" if len(deduped) > 1 else deduped[0][2]
    thinking = ("<think>" in text or "</think>" in text) and deduped[0][0] > text.find("<think>")
    return DiagnosticParse(
        next(iter(unique)),
        method,
        tuple(item[2] for item in deduped),
        multiple_consistent=len(deduped) > 1,
        thinking_then_answer=thinking,
    )


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def select_shared_discovery_ids(
    b0_rows: Iterable[Mapping[str, Any]],
    b1_rows: Iterable[Mapping[str, Any]],
    *,
    count: int = 50,
    seed: int = 42,
) -> tuple[str, ...]:
    def ids(rows: Iterable[Mapping[str, Any]], name: str) -> set[str]:
        result: set[str] = set()
        for row in rows:
            validate_label_blind_row(row)
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise DiagnosticError(f"{name} discovery rows contain duplicate sample_id")
            result.add(sample_id)
        return result

    left, right = ids(b0_rows, "B0"), ids(b1_rows, "B1")
    if left != right:
        raise DiagnosticError("B0/B1 diagnostic discovery sample sets differ")
    if count < 1 or len(left) < count:
        raise DiagnosticError("diagnostic discovery set is too small")
    return tuple(sorted(left, key=lambda item: (_stable_rank(item, seed), item))[:count])


def diagnostic_parser_manifest() -> dict[str, Any]:
    contract = {
        "metric_name": METRIC_NAME,
        "label_blind_discovery": True,
        "teacher_gate_eligible": False,
        "allowed_legal_sets": ["ABCD", "ABCDE"],
        "patterns": [name for name, _ in _EXPLICIT_PATTERNS]
        + ["first_line_answer", "standalone_last_label", "option_text_after_conclusion"],
        "multiple_consistent": "parse_and_disclose",
        "conflicting": "invalid",
        "arbitrary_body_letters": "ignored",
        "seed": 42,
        "discovery_count": 50,
    }
    bound = {
        "contract": contract,
        "implementation": inspect.getsource(parse_generation_diagnostic),
        "patterns": {
            name: pattern.pattern for name, pattern in _EXPLICIT_PATTERNS
        },
        "option_text_pattern": _OPTION_TEXT_CONCLUSION.pattern,
        "supervision_fields": sorted(_SUPERVISION_FIELDS),
    }
    raw = json.dumps(bound, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "parser_sha256": hashlib.sha256(raw).hexdigest()}


def score_diagnostic_rows(
    predictions: Iterable[Mapping[str, Any]], labels: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    def unique(rows: Iterable[Mapping[str, Any]], kind: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in result:
                raise DiagnosticError(f"{kind} contains missing/duplicate sample_id")
            if "final" in str(row.get("target_role") or ""):
                raise DiagnosticError(f"{kind} contains final role")
            result[sample_id] = row
        return result

    pred, gold = unique(predictions, "predictions"), unique(labels, "labels")
    if set(pred) != set(gold):
        raise DiagnosticError("prediction/label sample_id sets differ")
    scored: list[dict[str, Any]] = []
    for sample_id in sorted(pred):
        expected = str(gold[sample_id].get("answer_idx") or "").upper()
        predicted = pred[sample_id].get("predicted_label")
        if expected not in _LETTERS or (predicted is not None and str(predicted) not in _LETTERS):
            raise DiagnosticError("diagnostic prediction/label is not canonical")
        source = pred[sample_id]
        scored.append(
            {
                "sample_id": sample_id,
                "target_role": source.get("target_role"),
                "domain": source.get("domain"),
                "subject": source.get("subject"),
                "predicted_label": predicted,
                "parse_method": source.get("parse_method"),
                "parsed": predicted is not None,
                "correct": predicted == expected if predicted is not None else False,
                "multiple_consistent": bool(source.get("multiple_consistent", False)),
                "conflicting": bool(source.get("conflicting", False)),
                "thinking_then_answer": bool(source.get("thinking_then_answer", False)),
                "truncated": bool(source.get("truncated", False)),
                "thinking_tag": bool(source.get("thinking_tag", False)),
                "finish_reason": str(source.get("finish_reason") or "unknown"),
                "generated_token_count": int(source.get("generated_token_count") or 0),
            }
        )
    return scored


def summarize_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DiagnosticError("diagnostic summary cannot be empty")
    count = len(rows)
    parsed = [row for row in rows if row.get("parsed")]
    domains = sorted({str(row.get("domain") or "unknown") for row in rows})
    methods = Counter(str(row.get("parse_method") or "unknown") for row in rows)
    result: dict[str, Any] = {
        "count": count,
        "parse_coverage": len(parsed) / count,
        "diagnostic_accuracy": sum(bool(row.get("correct")) for row in rows) / count,
        "conditional_accuracy_among_parsed": (
            sum(bool(row.get("correct")) for row in parsed) / len(parsed) if parsed else None
        ),
        "invalid_count": count - len(parsed),
        "conflict_count": sum(bool(row.get("conflicting")) for row in rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in rows),
        "method_distribution": dict(sorted(methods.items())),
        "teacher_gate_eligible": False,
    }
    result["domains"] = {}
    for domain in domains:
        members = [row for row in rows if str(row.get("domain") or "unknown") == domain]
        parsed_members = [row for row in members if row.get("parsed")]
        result["domains"][domain] = {
            "count": len(members),
            "coverage": len(parsed_members) / len(members),
            "accuracy": sum(bool(row.get("correct")) for row in members) / len(members),
            "conditional_accuracy": (
                sum(bool(row.get("correct")) for row in parsed_members) / len(parsed_members)
                if parsed_members else None
            ),
        }
    return result


__all__ = [
    "DiagnosticError",
    "DiagnosticParse",
    "METRIC_NAME",
    "diagnostic_parser_manifest",
    "parse_generation_diagnostic",
    "score_diagnostic_rows",
    "select_shared_discovery_ids",
    "summarize_diagnostic_rows",
    "validate_label_blind_row",
]
