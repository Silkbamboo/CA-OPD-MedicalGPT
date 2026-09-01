"""Raw-source adapters.

Each public dataset named in the data plan gets one small, unit-tested converter
from its raw record layout to :class:`StagedSample`. Splits are *not* assigned
here - a source can feed several splits (MedQA feeds both controller dev and
final test), so split assignment belongs to ``build_splits``.

Adapters are pure functions over dicts, which means the conversion logic is
testable from tiny inline fixtures with no network access and no dataset
download. Actual downloading happens in :func:`load_hf_source`, which is only
called by the CLI on a machine that has the datasets library and disk budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from src.data.schema import (
    DOMAIN_GENERAL,
    DOMAIN_MEDICAL,
    TASK_MCQ,
    TASK_REASONING_SFT,
    Sample,
    SchemaError,
)
from src.utils.hashing import content_hash
from src.utils.io import iter_jsonl

LETTERS = "ABCDEFGH"


@dataclass
class StagedSample:
    """A converted-but-unassigned sample (no split yet)."""

    source: str
    domain: str
    task: str
    question: str
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    answer_index: Optional[int] = None
    reasoning: Optional[str] = None
    raw_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return content_hash(self.question, self.options)

    def to_sample(self, split: str) -> Sample:
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
            meta=dict(self.meta),
        )


Converter = Callable[[Mapping[str, Any], int], Optional[StagedSample]]


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def from_medical_o1_zh(record: Mapping[str, Any], index: int) -> Optional[StagedSample]:
    """``FreedomIntelligence/medical-o1-reasoning-SFT`` (Chinese subset).

    Raw layout: ``Question`` / ``Complex_CoT`` / ``Response``.
    """
    question = _first_present(record, ("Question", "question"))
    answer = _first_present(record, ("Response", "response", "answer"))
    reasoning = _first_present(record, ("Complex_CoT", "complex_cot", "reasoning"))
    if not question or not answer:
        return None
    return StagedSample(
        source="medical_o1_reasoning_zh",
        domain=DOMAIN_MEDICAL,
        task=TASK_REASONING_SFT,
        question=str(question).strip(),
        answer=str(answer).strip(),
        reasoning=str(reasoning).strip() if reasoning else None,
        raw_id=str(_first_present(record, ("id", "sample_id")) or index),
    )


def _normalise_medqa_options(raw: Any) -> Optional[List[str]]:
    """MedQA options appear as a dict, a list of dicts or a plain list."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return [str(raw[k]).strip() for k in sorted(raw)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        options: List[str] = []
        for item in raw:
            if isinstance(item, Mapping):
                value = _first_present(item, ("value", "text", "option"))
                if value is None:
                    return None
                options.append(str(value).strip())
            else:
                options.append(str(item).strip())
        return options
    return None


def from_medqa_zh(record: Mapping[str, Any], index: int) -> Optional[StagedSample]:
    """``bigbio/med_qa`` MedQA-zh style MCQ.

    Accepts either an explicit ``answer_idx`` (letter or int) or matches the
    ``answer`` string against the option texts. If neither resolves, the sample
    is dropped rather than guessed - a wrong gold label would silently corrupt
    every accuracy number downstream.
    """
    question = _first_present(record, ("question", "Question"))
    options = _normalise_medqa_options(_first_present(record, ("options", "choices")))
    if not question or not options:
        return None

    answer_text = _first_present(record, ("answer", "Answer"))
    answer_index: Optional[int] = None
    raw_idx = _first_present(record, ("answer_idx", "answer_index", "label"))
    if raw_idx is not None:
        if isinstance(raw_idx, int):
            answer_index = raw_idx
        else:
            token = str(raw_idx).strip().upper()
            if token in LETTERS:
                answer_index = LETTERS.index(token)
            elif token.isdigit():
                answer_index = int(token)
    if answer_index is None and answer_text is not None:
        normalised = [o.strip() for o in options]
        target = str(answer_text).strip()
        if target in normalised:
            answer_index = normalised.index(target)
    if answer_index is None or not 0 <= answer_index < len(options):
        return None

    return StagedSample(
        source="medqa_zh",
        domain=DOMAIN_MEDICAL,
        task=TASK_MCQ,
        question=str(question).strip(),
        options=options,
        answer=LETTERS[answer_index],
        answer_index=answer_index,
        raw_id=str(_first_present(record, ("id", "sample_id")) or index),
        meta={"num_options": len(options)},
    )


#: Medical configs in the official 52-subject C-Eval ``subject_mapping.json``.
#: Kept as a hard deny-list even though base.yaml also enumerates only the 48
#: non-medical configs: defense in depth against a future config edit.
MEDICAL_CEVAL_SUBJECTS = frozenset(
    {"clinical_medicine", "basic_medicine", "physician", "veterinary_medicine"}
)


def from_ceval(record: Mapping[str, Any], index: int) -> Optional[StagedSample]:
    """``ceval/ceval-exam`` MCQ with ``A``..``D`` columns.

    Medical subjects are rejected here so a "general" anchor pool can never be
    contaminated with medical content, which would make ``delta`` meaningless.
    """
    subject = str(_first_present(record, ("subject", "subject_name", "category")) or "unknown")
    if subject in MEDICAL_CEVAL_SUBJECTS:
        return None
    question = _first_present(record, ("question", "Question"))
    if not question:
        return None
    options = [str(record[letter]).strip() for letter in "ABCD" if record.get(letter) not in (None, "")]
    if len(options) < 2:
        return None
    answer_letter = _first_present(record, ("answer", "Answer"))
    answer_index: Optional[int] = None
    if answer_letter is not None:
        token = str(answer_letter).strip().upper()
        if token in LETTERS[: len(options)]:
            answer_index = LETTERS.index(token)
    return StagedSample(
        source=f"ceval_{subject}",
        domain=DOMAIN_GENERAL,
        task=TASK_MCQ,
        question=str(question).strip(),
        options=options,
        answer=LETTERS[answer_index] if answer_index is not None else None,
        answer_index=answer_index,
        raw_id=str(_first_present(record, ("id", "sample_id")) or index),
        meta={"subject": subject},
    )


CONVERTERS: Dict[str, Converter] = {
    "medical_o1_zh": from_medical_o1_zh,
    "medqa_zh": from_medqa_zh,
    "ceval": from_ceval,
}


# ---------------------------------------------------------------------------
# source specs + loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """Where a source lives and how to convert it.

    ``hf_name`` is for a single builder config. ``hf_names`` is the explicit
    multi-config form used by C-Eval, where every subject is a separate config
    and there is no ``all`` config. Requiring one of these forms makes a stale or
    invented builder name fail before a multi-GB data build starts.
    """

    name: str
    converter: str
    kind: str = "jsonl"  # jsonl | hf
    path: Optional[str] = None  # for kind=jsonl
    hf_path: Optional[str] = None  # for kind=hf
    hf_name: Optional[str] = None
    hf_names: Tuple[str, ...] = ()
    hf_revision: Optional[str] = None
    hf_trust_remote_code: bool = False
    hf_split: str = "train"
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.converter not in CONVERTERS:
            raise SchemaError(f"unknown converter {self.converter!r}; known: {sorted(CONVERTERS)}")
        if self.kind == "jsonl" and not self.path:
            raise SchemaError(f"source {self.name}: kind=jsonl requires 'path'")
        if self.kind == "hf":
            if not self.hf_path:
                raise SchemaError(f"source {self.name}: kind=hf requires 'hf_path'")
            if bool(self.hf_name) == bool(self.hf_names):
                raise SchemaError(
                    f"source {self.name}: kind=hf requires exactly one of 'hf_name' or 'hf_names'"
                )
            if not self.hf_revision:
                raise SchemaError(f"source {self.name}: kind=hf requires immutable 'hf_revision'")
            if len(set(self.hf_config_names)) != len(self.hf_config_names):
                raise SchemaError(f"source {self.name}: duplicate entries in hf_names")
        if self.kind not in ("jsonl", "hf"):
            raise SchemaError(f"source {self.name}: unknown kind {self.kind!r}")
        if self.max_samples is not None and self.max_samples < 1:
            raise SchemaError(f"source {self.name}: max_samples must be >= 1")

    @property
    def hf_config_names(self) -> Tuple[str, ...]:
        return (str(self.hf_name),) if self.hf_name else tuple(self.hf_names)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceSpec":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise SchemaError(f"unknown source spec keys: {sorted(unknown)}")
        values = dict(data)
        if "hf_names" in values:
            raw = values["hf_names"]
            if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) and v for v in raw):
                raise SchemaError("hf_names must be a non-empty list of config-name strings")
            values["hf_names"] = tuple(raw)
        return cls(**values)


def convert_records(records: Iterable[Mapping[str, Any]], converter: str, max_samples: Optional[int] = None) -> List[StagedSample]:
    """Convert raw records, dropping those the adapter cannot resolve.

    Returns the accepted samples; the number dropped is recoverable as
    ``len(records) - len(result)`` and is reported in the manifest.
    """
    fn = CONVERTERS[converter]
    out: List[StagedSample] = []
    for i, record in enumerate(records):
        staged = fn(record, i)
        if staged is None:
            continue
        out.append(staged)
        if max_samples is not None and len(out) >= max_samples:
            break
    return out


def load_jsonl_source(spec: SourceSpec) -> List[StagedSample]:
    path = Path(str(spec.path))
    if not path.exists():
        raise FileNotFoundError(f"source {spec.name}: file not found: {path}")
    return convert_records(iter_jsonl(path), spec.converter, spec.max_samples)


def load_hf_source(spec: SourceSpec) -> List[StagedSample]:  # pragma: no cover - needs network
    """Load one or more explicit Hugging Face builder configs.

    C-Eval rows do not carry their subject because the subject *is* the builder
    config. We inject that config name before conversion, preserving source IDs
    and providing a second medical-subject rejection guard.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"source {spec.name}: kind=hf requires the 'datasets' package"
        ) from exc

    out: List[StagedSample] = []
    for config_name in spec.hf_config_names:
        remaining = None if spec.max_samples is None else spec.max_samples - len(out)
        if remaining is not None and remaining <= 0:
            break
        load_kwargs: Dict[str, Any] = {
            "split": spec.hf_split,
            "revision": spec.hf_revision,
        }
        if spec.hf_trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        ds = load_dataset(str(spec.hf_path), config_name, **load_kwargs)

        def rows() -> Iterator[Mapping[str, Any]]:
            for row in ds:
                record = dict(row)
                if spec.converter == "ceval":
                    record.setdefault("subject", config_name)
                yield record

        out.extend(convert_records(rows(), spec.converter, remaining))
    return out


def load_source(spec: SourceSpec) -> List[StagedSample]:
    return load_jsonl_source(spec) if spec.kind == "jsonl" else load_hf_source(spec)


def iter_source_names(specs: Iterable[SourceSpec]) -> Iterator[str]:
    for spec in specs:
        yield spec.name
