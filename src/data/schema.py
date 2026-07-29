"""Canonical data schema for every CA-OPD split.

Design rule: leakage is prevented *structurally*, not by discipline. A sample
belonging to an OPD prompt pool cannot carry an answer field, because
:meth:`Sample.to_record` refuses to serialise fields that the split's policy
declares invisible. So even a buggy training script cannot read a label that was
never written to disk.

Split roles (project data protocol)
-----------------------------------
======================  ==================================  =================
split                   fields written to disk               may drive control
======================  ==================================  =================
``medical_sft``         question + answer + reasoning        no
``medical_opd_prompts`` question only                        no
``general_anchors``     question + options (no answer)       no
``controller_dev``      question + options + answer          **yes**
``final_test``          question + options + answer          **never**
======================  ==================================  =================

``controller_dev`` is the only split allowed to influence teacher routing,
checkpoint selection and early stopping. ``final_test`` is read exactly once,
after the configuration and checkpoint are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
