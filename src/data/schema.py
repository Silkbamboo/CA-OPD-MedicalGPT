"""Phase-0 split schema plus the canonical Data Protocol v2 extension.

Design rule: leakage is prevented *structurally*, not by discipline. A sample
belonging to an OPD prompt pool cannot carry an answer field, because
:meth:`Sample.to_record` refuses to serialise fields that the split's policy
declares invisible. So even a buggy training script cannot read a label that was
never written to disk.

Phase-0 v1 compatibility split roles
------------------------------------
======================  ==================================  =================
split                   fields written to disk               may drive control
======================  ==================================  =================
``medical_sft``         question + answer + reasoning        no
``medical_opd_prompts`` question only                        no
``general_anchors``     question + options (no answer)       no
``controller_dev``      question + options + answer          **yes**
``final_test``          question + options + answer          **never**
======================  ==================================  =================

This table documents the retained Phase-0 regression API. Formal P1+ builds use
the explicit Data Protocol v2 roles declared later in this module.
``controller_dev`` is the only v1 split allowed to influence teacher routing,
checkpoint selection and early stopping. ``final_test`` is read exactly once,
after the configuration and checkpoint are frozen.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from src.utils.hashing import content_hash, options_hash, stable_sample_id, text_hash

# -- vocabularies ------------------------------------------------------------

MEDICAL_SFT = "medical_sft"
MEDICAL_OPD_PROMPTS = "medical_opd_prompts"
GENERAL_ANCHORS = "general_anchors"
CONTROLLER_DEV = "controller_dev"
FINAL_TEST = "final_test"

SPLITS: Tuple[str, ...] = (
    MEDICAL_SFT,
    MEDICAL_OPD_PROMPTS,
    GENERAL_ANCHORS,
    CONTROLLER_DEV,
    FINAL_TEST,
)

DOMAIN_MEDICAL = "medical"
DOMAIN_GENERAL = "general"
DOMAINS: Tuple[str, ...] = (DOMAIN_MEDICAL, DOMAIN_GENERAL)

TASK_MCQ = "mcq"
TASK_OPEN_QA = "open_qa"
TASK_REASONING_SFT = "reasoning_sft"
TASKS: Tuple[str, ...] = (TASK_MCQ, TASK_OPEN_QA, TASK_REASONING_SFT)

#: Fields serialised for each split. Anything not listed is dropped on write.
_BASE_FIELDS: FrozenSet[str] = frozenset(
    {"sample_id", "source", "split", "domain", "task", "question", "text_hash", "content_hash", "meta"}
)

SPLIT_VISIBLE_FIELDS: Mapping[str, FrozenSet[str]] = {
    MEDICAL_SFT: _BASE_FIELDS | {"answer", "reasoning"},
    MEDICAL_OPD_PROMPTS: _BASE_FIELDS,  # question only
    GENERAL_ANCHORS: _BASE_FIELDS | {"options"},  # options but no answer
    CONTROLLER_DEV: _BASE_FIELDS | {"options", "answer", "answer_index"},
    FINAL_TEST: _BASE_FIELDS | {"options", "answer", "answer_index"},
}

#: Splits whose evaluation results may drive routing / checkpoint selection.
CONTROL_SPLITS: FrozenSet[str] = frozenset({CONTROLLER_DEV})

#: Fields that constitute supervision and must never leak into an OPD pool.
LABEL_FIELDS: FrozenSet[str] = frozenset({"answer", "answer_index", "reasoning", "options_answer"})


class SchemaError(ValueError):
    """Raised for any sample that violates the schema or its split policy."""


@dataclass
class Sample:
    """One data point in its canonical form.

    ``text_hash`` / ``content_hash`` are always recomputed from the normalised
    question (and options) in :meth:`__post_init__`, so a hand-edited jsonl file
    cannot carry a stale hash that would defeat deduplication.
    """

    source: str
    split: str
    domain: str
    task: str
    question: str
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    answer_index: Optional[int] = None
    reasoning: Optional[str] = None
    raw_id: Optional[str] = None
    sample_id: str = ""
    text_hash: str = ""
    content_hash: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise SchemaError(f"unknown split {self.split!r}; expected one of {SPLITS}")
        if self.domain not in DOMAINS:
            raise SchemaError(f"unknown domain {self.domain!r}; expected one of {DOMAINS}")
        if self.task not in TASKS:
            raise SchemaError(f"unknown task {self.task!r}; expected one of {TASKS}")
        if not self.source:
            raise SchemaError("source must be a non-empty string")
        if not self.question or not self.question.strip():
            raise SchemaError(f"empty question in sample from {self.source!r}")
        if self.options is not None:
            if len(self.options) < 2:
                raise SchemaError(f"MCQ needs >= 2 options, got {self.options!r}")
            if any(not str(o).strip() for o in self.options):
                raise SchemaError("MCQ options must all be non-empty")
        if self.task == TASK_MCQ and self.options is None:
            raise SchemaError("task=mcq requires options")
        if self.answer_index is not None:
            if self.options is None:
                raise SchemaError("answer_index given without options")
            if not 0 <= self.answer_index < len(self.options):
                raise SchemaError(f"answer_index {self.answer_index} out of range for {len(self.options)} options")

        self.text_hash = text_hash(self.question)
        self.content_hash = content_hash(self.question, self.options)
        if not self.sample_id:
            self.sample_id = stable_sample_id(self.source, self.raw_id or self.text_hash[:8], self.question)

    # -- policy -----------------------------------------------------------
    @property
    def visible_fields(self) -> FrozenSet[str]:
        return SPLIT_VISIBLE_FIELDS[self.split]

    def options_hash(self) -> str:
        return options_hash(self.options)

    def to_record(self) -> Dict[str, Any]:
        """Serialise only the fields this split is allowed to expose.

        Dropping happens here on purpose: an OPD prompt sample that *carries* an
        answer in memory (because the loader read it from the raw source) will
        still be written without it.
        """
        allowed = self.visible_fields
        record: Dict[str, Any] = {}
        for key in (
            "sample_id",
            "source",
            "split",
            "domain",
            "task",
            "question",
            "options",
            "answer",
            "answer_index",
            "reasoning",
            "text_hash",
            "content_hash",
        ):
            if key not in allowed:
                continue
            value = getattr(self, key)
            if value is None:
                continue
            record[key] = value
        if "meta" in allowed and self.meta:
            record["meta"] = dict(self.meta)
        return record

    def with_split(self, split: str) -> "Sample":
        """Return a copy assigned to ``split`` (re-validated)."""
        return Sample(
            source=self.source,
            split=split,
            domain=self.domain,
            task=self.task,
            question=self.question,
            options=list(self.options) if self.options else None,
            answer=self.answer,
            answer_index=self.answer_index,
            reasoning=self.reasoning,
            raw_id=self.raw_id,
            sample_id=self.sample_id,
            meta=dict(self.meta),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Sample":
        missing = {"source", "split", "domain", "task", "question"} - set(record)
        if missing:
            raise SchemaError(f"record missing required field(s): {sorted(missing)}")
        return cls(
            source=str(record["source"]),
            split=str(record["split"]),
            domain=str(record["domain"]),
            task=str(record["task"]),
            question=str(record["question"]),
            options=list(record["options"]) if record.get("options") else None,
            answer=record.get("answer"),
            answer_index=record.get("answer_index"),
            reasoning=record.get("reasoning"),
            raw_id=record.get("raw_id"),
            sample_id=str(record.get("sample_id", "")),
            meta=dict(record.get("meta") or {}),
        )


def assert_no_label_leakage(records: Sequence[Mapping[str, Any]], split: str) -> None:
    """Fail if any serialised record of an unlabelled split carries supervision."""
    if split in (MEDICAL_SFT, CONTROLLER_DEV, FINAL_TEST):
        return
    forbidden = LABEL_FIELDS if split == MEDICAL_OPD_PROMPTS else LABEL_FIELDS - {"reasoning"}
    if split == GENERAL_ANCHORS:
        forbidden = {"answer", "answer_index"}
    for i, record in enumerate(records):
        present = set(record) & set(forbidden)
        if present:
            raise SchemaError(
                f"{split} record #{i} ({record.get('sample_id')}) exposes label field(s) {sorted(present)}"
            )


def may_drive_control(split: str) -> bool:
    """Whether evaluation on ``split`` may influence training decisions."""
    if split not in SPLITS:
        raise SchemaError(f"unknown split {split!r}")
    return split in CONTROL_SPLITS


# -- Data Protocol v2 -------------------------------------------------------
#
# The original five-split API above remains available for Phase-0 regression
# and existing OPD/SFT call sites.  Formal P1 data uses the more explicit roles
# below so medical/general controller and final capabilities cannot be confused.

DATA_PROTOCOL_VERSION = "ca-opd-data-v2"
# P1.6 changes formal upstream representations and General Anchor licensing
# without changing the canonical record schema.  Formal manifests must bind
# both versions so pre-P1.6 v2 manifests cannot be reused silently.
SOURCE_POLICY_VERSION = "ca-opd-source-policy-v1"
SCHEMA_VERSION_V2 = 2
DATA_ROLES_V2: Tuple[str, ...] = (
    "medical_sft_train",
    "medical_sft_dev",
    "medical_opd_o1",
    "audit_holdout",
    "medical_opd_cmb",
    "medical_controller_dev",
    "medical_final_test",
    "general_anchors",
    "general_controller_dev",
    "general_final_test",
    "ceval_smoke",
)
TRAINING_ROLES_V2: FrozenSet[str] = frozenset(
    {"medical_sft_train", "medical_opd_o1", "medical_opd_cmb", "general_anchors"}
)
PROMPT_ONLY_ROLES_V2: FrozenSet[str] = frozenset(
    {"medical_opd_o1", "medical_opd_cmb", "general_anchors"}
)
CONTROLLER_ROLES_V2: FrozenSet[str] = frozenset(
    {"medical_controller_dev", "general_controller_dev"}
)
FINAL_ROLES_V2: FrozenSet[str] = frozenset(
    {"medical_final_test", "general_final_test"}
)
SUPERVISION_KEYS: FrozenSet[str] = frozenset(
    {
        "answer",
        "answer_idx",
        "chain_of_thought",
        "cot",
        "rationale",
        "reasoning",
        "response",
        "solution",
        "label",
        "output",
        "completion",
    }
)


def normalize_question_v2(value: str) -> str:
    """Normalize layout without erasing punctuation or medical semantics.

    NFKC canonicalizes full-width forms. Horizontal whitespace is collapsed on
    each line, blank edge lines are removed, and meaningful line boundaries are
    retained. Punctuation, dosage units, polarity/negation and token order are
    never removed.
    """

    if not isinstance(value, str):
        raise TypeError("question text must be a string")
    normalized = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_options_v2(options: Sequence[str]) -> Tuple[str, ...]:
    """Normalize options while preserving their original order."""

    return tuple(normalize_question_v2(str(option)) for option in options)


def content_hash_v2(question: str, options: Sequence[str] = ()) -> str:
    """Build a reconstructable SHA-256 from question and ordered options."""

    payload = {
        "normalized_question": normalize_question_v2(question),
        "normalized_options": list(normalize_options_v2(options)),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_sample_id_v2(
    *,
    source: str,
    source_revision: str,
    upstream_split: str,
    upstream_id: str,
    subject: Optional[str] = None,
) -> str:
    """Return a stable source-qualified ID without embedding raw question text."""

    identity = "\0".join(
        (source, source_revision, upstream_split, subject or "", upstream_id)
    )
    return f"{source}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class DataRecordV2:
    """Canonical intermediate row for Data Protocol v2."""

    sample_id: str
    source: str
    source_revision: str
    source_license: str
    upstream_split: str
    target_role: str
    domain: str
    question: str
    normalized_question: str
    content_hash: str
    group_id: str
    raw_file_sha256: str
    subject: Optional[str] = None
    category: Optional[str] = None
    options: Tuple[str, ...] = ()
    normalized_options: Tuple[str, ...] = ()
    answer: Optional[str] = None
    answer_idx: Optional[str] = None
    reasoning: Optional[str] = None
    token_count_prompt: Optional[int] = None
    token_count_response: Optional[int] = None
    quality_flags: Tuple[str, ...] = ()
    drop_reason: Optional[str] = None
    data_protocol_version: str = DATA_PROTOCOL_VERSION
    schema_version: int = SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if self.target_role not in DATA_ROLES_V2:
            raise ValueError(f"unsupported Data Protocol v2 role: {self.target_role}")
        if self.domain not in {DOMAIN_MEDICAL, DOMAIN_GENERAL}:
            raise ValueError("domain must be medical or general")
        if not self.sample_id or not self.source or not self.source_revision:
            raise ValueError("sample/source/revision must be non-empty")
        if not self.source_license:
            raise ValueError("source_license must be explicit, including unknown")
        if len(self.raw_file_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.raw_file_sha256
        ):
            raise ValueError("raw_file_sha256 must be a lowercase SHA-256")
        if self.normalized_question != normalize_question_v2(self.question):
            raise ValueError("normalized_question is not reconstructable")
        if self.normalized_options != normalize_options_v2(self.options):
            raise ValueError("normalized_options are not reconstructable")
        if self.content_hash != content_hash_v2(self.question, self.options):
            raise ValueError("content_hash is not reconstructable")
        for label, count in (
            ("token_count_prompt", self.token_count_prompt),
            ("token_count_response", self.token_count_response),
        ):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError(f"{label} must be null or a non-negative integer")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        payload["normalized_options"] = list(self.normalized_options)
        payload["quality_flags"] = list(self.quality_flags)
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, (), {}, [])
        }
