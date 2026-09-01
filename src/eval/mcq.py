"""Deterministic MCQ evaluation for medical and general ability.

This is the module that produces the numbers the whole project is judged on:
``M_medical``, ``M_general``, ``ΔM``, ``ΔG`` and whether the constraint
``ΔG >= -delta`` holds.

Determinism rules baked in here (data protocol "评测温度、token budget、system
prompt固定并记录"):

* greedy decoding only - the generator callable receives ``temperature=0.0``;
* fixed prompt template and fixed option order (no shuffling);
* fixed max new tokens;
* the prompt/decoding settings are echoed into the result payload so two runs
  can be compared for口径 drift.

The model itself is injected as ``generate_fn(prompts, max_new_tokens) ->
responses``. That keeps this module CPU-testable and lets Phase 1 plug in
transformers ``generate`` or a vLLM server without touching scoring logic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPLATE, ChatTemplate, format_mcq_question
from src.data.schema import DOMAINS, Sample
from src.eval.parsing import parse_with_options

GenerateFn = Callable[[List[str], int], List[str]]


@dataclass(frozen=True)
class DecodeSettings:
    """Everything that can change an accuracy number without changing the model."""

    temperature: float = 0.0
    max_new_tokens: int = 16
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    template_name: str = DEFAULT_TEMPLATE.name
    shuffle_options: bool = False

    def __post_init__(self) -> None:
        if self.temperature != 0.0:
            raise ValueError(
                "MCQ evaluation must be greedy (temperature=0.0); sampling would make "
                "accuracy noisy and non-reproducible"
            )
        if self.shuffle_options:
            raise ValueError("option order must stay fixed so results are comparable across runs")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SampleResult:
    sample_id: str
    domain: str
    correct: bool
    parsed: bool
    predicted: Optional[str]
    gold: str
    parse_method: str
    response: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MCQResult:
    """Aggregate evaluation result for one split at one checkpoint."""

    split: str
    num_samples: int
    accuracy: float
    accuracy_by_domain: Dict[str, float]
    counts_by_domain: Dict[str, int]
    unparsed_rate: float
    parse_methods: Dict[str, int]
    decode: Dict[str, object]
    seconds: float
    samples: List[SampleResult] = field(default_factory=list)

    @property
    def medical_accuracy(self) -> Optional[float]:
        return self.accuracy_by_domain.get("medical")

    @property
    def general_accuracy(self) -> Optional[float]:
        return self.accuracy_by_domain.get("general")

    def as_dict(self, include_samples: bool = False) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "split": self.split,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "accuracy_by_domain": self.accuracy_by_domain,
            "counts_by_domain": self.counts_by_domain,
            "unparsed_rate": self.unparsed_rate,
            "parse_methods": self.parse_methods,
            "decode": self.decode,
            "seconds": round(self.seconds, 3),
        }
        if include_samples:
            payload["samples"] = [s.as_dict() for s in self.samples]
        return payload


def render_mcq_prompt(
    sample: Sample,
    template: ChatTemplate = DEFAULT_TEMPLATE,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    if not sample.options:
        raise ValueError(f"sample {sample.sample_id} has no options; MCQ evaluation requires them")
    return template.render_prompt(format_mcq_question(sample.question, sample.options), system_prompt)


def evaluate_mcq(
    samples: Sequence[Sample],
    generate_fn: GenerateFn,
    split: str,
    decode: DecodeSettings | None = None,
    template: ChatTemplate = DEFAULT_TEMPLATE,
    batch_size: int = 8,
) -> MCQResult:
    """Score ``samples`` with greedy generation and deterministic parsing."""
    if not samples:
        raise ValueError("evaluate_mcq received zero samples")
    settings = decode or DecodeSettings()
    missing_gold = [s.sample_id for s in samples if s.answer_index is None]
    if missing_gold:
        raise ValueError(
            f"{len(missing_gold)} sample(s) have no gold label and cannot be scored "
            f"(first: {missing_gold[0]}); an unlabelled pool must not be used for accuracy"
        )

    t0 = time.time()
    prompts = [render_mcq_prompt(s, template, settings.system_prompt) for s in samples]
    responses: List[str] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        chunk = prompts[start : start + max(1, batch_size)]
        out = generate_fn(chunk, settings.max_new_tokens)
        if len(out) != len(chunk):
            raise RuntimeError(f"generate_fn returned {len(out)} responses for {len(chunk)} prompts")
        responses.extend(out)

    results: List[SampleResult] = []
    per_domain_correct: Dict[str, int] = {d: 0 for d in DOMAINS}
    per_domain_total: Dict[str, int] = {d: 0 for d in DOMAINS}
    parse_methods: Dict[str, int] = {}
    unparsed = 0

    for sample, response in zip(samples, responses):
        parsed = parse_with_options(response, sample.options or [])
        gold_letter = "ABCDEFGH"[int(sample.answer_index)]
        correct = bool(parsed.parsed and parsed.index == sample.answer_index)
        if not parsed.parsed:
            unparsed += 1
        parse_methods[parsed.method] = parse_methods.get(parsed.method, 0) + 1
        per_domain_total[sample.domain] = per_domain_total.get(sample.domain, 0) + 1
        per_domain_correct[sample.domain] = per_domain_correct.get(sample.domain, 0) + int(correct)
        results.append(
            SampleResult(
                sample_id=sample.sample_id,
                domain=sample.domain,
                correct=correct,
                parsed=parsed.parsed,
                predicted=parsed.letter,
                gold=gold_letter,
                parse_method=parsed.method,
                response=str(response),
            )
        )

    total_correct = sum(r.correct for r in results)
    accuracy_by_domain = {
        d: (per_domain_correct[d] / per_domain_total[d]) for d in per_domain_total if per_domain_total[d] > 0
    }
    return MCQResult(
        split=split,
        num_samples=len(results),
        accuracy=total_correct / len(results),
        accuracy_by_domain=accuracy_by_domain,
        counts_by_domain={d: n for d, n in per_domain_total.items() if n > 0},
        unparsed_rate=unparsed / len(results),
        parse_methods=dict(sorted(parse_methods.items())),
        decode=settings.as_dict(),
        seconds=time.time() - t0,
        samples=results,
    )


# ---------------------------------------------------------------------------
# constraint accounting
# ---------------------------------------------------------------------------


@dataclass
class ConstraintReport:
    """``ΔM``, ``ΔG`` and whether the general-ability constraint holds."""

    medical_base: float
    general_base: float
    medical_now: float
    general_now: float
    delta: float
    delta_medical: float
    delta_general: float
    constraint_satisfied: bool
    general_floor: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def constraint_report(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    delta: float,
) -> ConstraintReport:
    """Compare a checkpoint against the base model under the ability constraint.

    ``baseline`` / ``current`` are ``{"medical": acc, "general": acc}`` measured
    on the *same* split with the *same* decode settings. ``delta`` is the allowed
    general-ability degradation, fixed before any final-test evaluation.
    """
    if delta < 0:
        raise ValueError("delta is a magnitude and must be >= 0")
    for name, payload in (("baseline", baseline), ("current", current)):
        missing = {"medical", "general"} - set(payload)
        if missing:
            raise ValueError(f"{name} accuracies missing domain(s): {sorted(missing)}")
    m_base, g_base = float(baseline["medical"]), float(baseline["general"])
    m_now, g_now = float(current["medical"]), float(current["general"])
    floor = g_base - delta
    return ConstraintReport(
        medical_base=m_base,
        general_base=g_base,
        medical_now=m_now,
        general_now=g_now,
        delta=float(delta),
        delta_medical=m_now - m_base,
        delta_general=g_now - g_base,
        constraint_satisfied=g_now >= floor,
        general_floor=floor,
    )
