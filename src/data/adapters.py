"""Fail-closed source adapters for Data Protocol v2.

Adapters accept the field shapes published by each upstream source and return a
canonical intermediate record or an auditable drop reason. They never write or
mutate raw input rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import (
    DATA_ROLES_V2,
    DataRecordV2,
    content_hash_v2,
    normalize_options_v2,
    normalize_question_v2,
    stable_sample_id_v2,
)


CEVAL_SUBJECT_ALLOWLIST = frozenset(
    {
        "computer_network",
        "college_programming",
        "advanced_mathematics",
        "discrete_mathematics",
        "college_physics",
        "logic",
        "chinese_language_and_literature",
        "college_economics",
    }
)
UNKNOWN_LICENSES = frozenset({"", "unknown", "unverified", "none", "null"})


@dataclass(frozen=True)
class AdapterContext:
    """Immutable source metadata supplied by a versioned source config."""

    source_type: str
    source: str
    source_revision: str
    source_license: str
    upstream_split: str
    target_role: str
    raw_file_sha256: str
    subsource: str | None = None
    subject: str | None = None

    def __post_init__(self) -> None:
        if self.target_role not in DATA_ROLES_V2:
            raise ValueError(f"unsupported target_role: {self.target_role}")
        if not self.source or not self.source_revision or not self.upstream_split:
            raise ValueError("source, revision and upstream_split are required")
        if len(self.raw_file_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.raw_file_sha256
        ):
            raise ValueError("raw_file_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class AdapterResult:
    """Exactly one accepted record or one privacy-safe drop record."""

    record: DataRecordV2 | None
    drop_reason: str | None
    raw_identity: str

    def __post_init__(self) -> None:
        if (self.record is None) == (self.drop_reason is None):
            raise ValueError("adapter result must contain exactly one outcome")

    def require_record(self) -> DataRecordV2:
        if self.record is None:
            raise ValueError(f"row was filtered: {self.drop_reason}")
        return self.record

    def drop_audit_dict(self) -> dict[str, str]:
        if self.drop_reason is None:
            raise ValueError("accepted rows do not have a drop audit entry")
        return {
            "raw_identity": self.raw_identity,
            "drop_reason": self.drop_reason,
        }


def _canonical_raw_hash(raw: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(raw),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _drop(raw_identity: str, reason: str) -> AdapterResult:
    return AdapterResult(record=None, drop_reason=reason, raw_identity=raw_identity)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _ordered_options(
    raw: Mapping[str, Any],
    *,
    direct_labels: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidate: Any = None
    if direct_labels and all(label in raw for label in ("A", "B", "C", "D")):
        candidate = {label: raw[label] for label in ("A", "B", "C", "D")}
    else:
        for key in ("options", "option", "choices"):
            if key in raw:
                candidate = raw[key]
                break
    if candidate is None:
        return (), ()

    labels: list[str] = []
    options: list[str] = []
    if isinstance(candidate, Mapping):
        keys = list(candidate)
        ordered_keys = sorted(
            keys,
            key=lambda value: (
                0,
                ord(str(value).upper()),
            )
            if re.fullmatch(r"[A-Z]", str(value).upper())
            else (1, str(value)),
        )
        for key in ordered_keys:
            value = _text(candidate[key])
            if value is None:
                continue
            labels.append(str(key).upper())
            options.append(value)
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        for index, item in enumerate(candidate):
            if isinstance(item, Mapping):
                label = _text(
                    item.get("id")
                    or item.get("label")
                    or item.get("key")
                ) or chr(ord("A") + index)
                value = _text(
                    item.get("text")
                    or item.get("value")
                    or item.get("option")
                )
            else:
                label = chr(ord("A") + index)
                value = _text(item)
            if value is None:
                continue
            labels.append(label.upper())
            options.append(value)
    return tuple(options), tuple(labels)


def _answer_fields(
    raw: Mapping[str, Any], labels: Sequence[str]
) -> tuple[str | None, str | None]:
    answer_idx = _text(raw.get("answer_idx") or raw.get("label"))
    answer_value: Any = raw.get("answer")
    if isinstance(answer_value, Sequence) and not isinstance(
        answer_value, (str, bytes)
    ):
        first = answer_value[0] if answer_value else None
        if isinstance(first, Mapping):
            answer_idx = answer_idx or _text(
                first.get("id") or first.get("label") or first.get("key")
            )
            answer = _text(first.get("text") or first.get("value"))
        else:
            answer = _text(first)
    elif isinstance(answer_value, Mapping):
        answer_idx = answer_idx or _text(
            answer_value.get("id")
            or answer_value.get("label")
            or answer_value.get("key")
        )
        answer = _text(answer_value.get("text") or answer_value.get("value"))
    else:
        answer = _text(answer_value)

    if answer_idx is not None:
        answer_idx = answer_idx.upper()
    if answer_idx is None and answer is not None and answer.upper() in labels:
        answer_idx = answer.upper()
    return answer, answer_idx


def _upstream_identity(raw: Mapping[str, Any], raw_identity: str) -> str:
    for key in ("id", "question_id", "index", "uid"):
        value = _text(raw.get(key))
        if value is not None:
            return value
    return raw_identity


def _build_record(
    raw: Mapping[str, Any],
    context: AdapterContext,
    *,
    question: str,
    domain: str,
    options: Sequence[str] = (),
    labels: Sequence[str] = (),
    answer: str | None = None,
    answer_idx: str | None = None,
    reasoning: str | None = None,
    subject: str | None = None,
    category: str | None = None,
    quality_flags: Sequence[str] = (),
) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    normalized_question = normalize_question_v2(question)
    normalized_options = normalize_options_v2(options)
    content_hash = content_hash_v2(question, options)
    resolved_subject = subject or context.subject or context.subsource
    record = DataRecordV2(
        sample_id=stable_sample_id_v2(
            source=context.source,
            source_revision=context.source_revision,
            upstream_split=context.upstream_split,
            upstream_id=_upstream_identity(raw, raw_identity),
            subject=resolved_subject,
        ),
        source=context.source,
        source_revision=context.source_revision,
        source_license=context.source_license,
        upstream_split=context.upstream_split,
        target_role=context.target_role,
        domain=domain,
        subject=resolved_subject,
        category=category,
        question=question,
        options=tuple(str(value) for value in options),
        answer=answer,
        answer_idx=answer_idx,
        reasoning=reasoning,
        normalized_question=normalized_question,
        normalized_options=normalized_options,
        content_hash=content_hash,
        group_id=content_hash,
        token_count_prompt=None,
        token_count_response=None,
        quality_flags=tuple(quality_flags),
        raw_file_sha256=context.raw_file_sha256,
    )
    return AdapterResult(record=record, drop_reason=None, raw_identity=raw_identity)


def _adapt_medical_o1(
    raw: Mapping[str, Any], context: AdapterContext
) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    if context.upstream_split != "train":
        return _drop(raw_identity, "medical_o1_non_train_split")
    question = _text(raw.get("Question"))
    if question is None:
        return _drop(raw_identity, "missing_question")
    return _build_record(
        raw,
        context,
        question=question,
        domain="medical",
        answer=_text(raw.get("Response")),
        reasoning=_text(raw.get("Complex_CoT")),
        quality_flags=("tokenizer_length_pending",),
    )


def _adapt_medqa(raw: Mapping[str, Any], context: AdapterContext) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    expected_role = {
        "dev": "medical_controller_dev",
        "validation": "medical_controller_dev",
        "test": "medical_final_test",
    }.get(context.upstream_split)
    if expected_role is None or context.target_role != expected_role:
        return _drop(raw_identity, "medqa_split_role_violation")
    question = _text(raw.get("question") or raw.get("Question"))
    if question is None:
        return _drop(raw_identity, "missing_question")
    options, labels = _ordered_options(raw)
    if len(options) < 2:
        return _drop(raw_identity, "invalid_options")
    answer, answer_idx = _answer_fields(raw, labels)
    return _build_record(
        raw,
        context,
        question=question,
        domain="medical",
        options=options,
        labels=labels,
        answer=answer,
        answer_idx=answer_idx,
        reasoning=_text(raw.get("explanation") or raw.get("reasoning")),
        subject=_text(raw.get("meta_info")) or context.subject,
    )


def _adapt_cmb(raw: Mapping[str, Any], context: AdapterContext) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    if context.upstream_split != "train" or context.target_role != "medical_opd_cmb":
        return _drop(raw_identity, "cmb_non_train_not_allowed")
    question = _text(raw.get("question"))
    if question is None:
        return _drop(raw_identity, "missing_question")
    options, labels = _ordered_options(raw)
    if len(options) < 2:
        return _drop(raw_identity, "invalid_options")
    answer, answer_idx = _answer_fields(raw, labels)
    category_parts = [
        value
        for value in (
            _text(raw.get("exam_type")),
            _text(raw.get("exam_class")),
            _text(raw.get("exam_subject")),
        )
        if value
    ]
    return _build_record(
        raw,
        context,
        question=question,
        domain="medical",
        options=options,
        labels=labels,
        answer=answer,
        answer_idx=answer_idx,
        reasoning=_text(raw.get("analysis") or raw.get("explanation")),
        subject=_text(raw.get("exam_subject")),
        category="/".join(category_parts) or None,
    )


def _contains_medical_domain(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    folded = text.casefold()
    for explicitly_non_medical in (
        "non-medical",
        "nonmedical",
        "non medical",
        "非医学",
        "非医疗",
    ):
        folded = folded.replace(explicitly_non_medical, "")
    return any(
        marker in folded
        for marker in ("medical", "medicine", "health", "clinical", "医学", "医疗", "健康")
    )


def _adapt_coig(raw: Mapping[str, Any], context: AdapterContext) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    if context.target_role != "general_anchors":
        return _drop(raw_identity, "coig_role_violation")
    if context.source_license.strip().casefold() in UNKNOWN_LICENSES:
        return _drop(raw_identity, "unknown_source_license")
    if _contains_medical_domain(
        [
            raw.get("domain"),
            raw.get("subject"),
            raw.get("category"),
            raw.get("task_name"),
            raw.get("task_name_in_eng"),
        ]
    ):
        return _drop(raw_identity, "coig_medical_domain_excluded")
    use_translated_fields = (
        context.subsource == "translated"
        and _text(raw.get("trans_instruction")) is not None
    )
    instruction = _text(
        raw.get("trans_instruction") if use_translated_fields else raw.get("instruction")
    )
    extra_input = _text(
        raw.get("trans_input") if use_translated_fields else raw.get("input")
    )
    if instruction is not None:
        question = instruction if extra_input is None else f"{instruction}\n{extra_input}"
    else:
        # The pinned COIG exam file uses a distinct, flat textbox schema.  Keep
        # instruction/context/question order and never infer content from the
        # gold answer fields.
        exam_parts = [
            value
            for value in (
                _text(raw.get("textbox_q_instruction")),
                _text(raw.get("textbox_q_context")),
                _text(raw.get("textbox_question")),
            )
            if value is not None
        ]
        if not exam_parts:
            return _drop(raw_identity, "missing_question")
        question = "\n".join(exam_parts)
    subject = (
        _text(raw.get("task_name_in_eng"))
        or _text(raw.get("task_name"))
        or _text(raw.get("subject"))
        or context.subsource
    )
    return _build_record(
        raw,
        context,
        question=question,
        domain="general",
        answer=(
            _text(raw.get("trans_output"))
            if use_translated_fields
            else _text(raw.get("output")) or _text(raw.get("textbox_answer"))
        ),
        reasoning=_text(raw.get("textbox_answer_analysis")),
        subject=subject,
        category=context.subsource,
        quality_flags=("license_verified_by_config",),
    )


def _adapt_ceval(raw: Mapping[str, Any], context: AdapterContext) -> AdapterResult:
    raw_identity = _canonical_raw_hash(raw)
    subject = context.subject or _text(raw.get("subject"))
    if subject not in CEVAL_SUBJECT_ALLOWLIST:
        return _drop(raw_identity, "ceval_subject_not_allowed")
    expected_role = {
        "dev": "ceval_smoke",
        "val": "general_controller_dev",
        "validation": "general_controller_dev",
        "test": "general_final_test",
    }.get(context.upstream_split)
    if expected_role is None or context.target_role != expected_role:
        return _drop(raw_identity, "ceval_split_role_violation")
    question = _text(raw.get("question"))
    if question is None:
        return _drop(raw_identity, "missing_question")
    options, labels = _ordered_options(raw, direct_labels=True)
    if len(options) != 4:
        return _drop(raw_identity, "invalid_options")
    answer, answer_idx = _answer_fields(raw, labels)
    return _build_record(
        raw,
        context,
        question=question,
        domain="general",
        options=options,
        labels=labels,
        answer=answer,
        answer_idx=answer_idx,
        reasoning=_text(raw.get("explanation")),
        subject=subject,
    )


_ADAPTERS = {
    "medical_o1": _adapt_medical_o1,
    "medqa_zh": _adapt_medqa,
    "cmb": _adapt_cmb,
    "coig": _adapt_coig,
    "ceval": _adapt_ceval,
}


def adapt_source_row(
    raw: Mapping[str, Any], context: AdapterContext
) -> AdapterResult:
    """Adapt one raw mapping without mutating it or guessing missing fields."""

    if not isinstance(raw, Mapping):
        raise TypeError("raw source row must be a mapping")
    try:
        adapter = _ADAPTERS[context.source_type]
    except KeyError as error:
        raise ValueError(f"unsupported source adapter: {context.source_type}") from error
    return adapter(raw, context)
