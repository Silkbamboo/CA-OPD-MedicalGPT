"""Controller Protocol v2 contracts shared by CPU tests and lazy GPU runners.

The primary path scores fixed option-label sequences and therefore does not
depend on free-generation formatting. The secondary path measures answer-first
generation compliance. Execution records intentionally cannot carry labels;
joining predictions to labels belongs to the separate scorer boundary.
"""

from __future__ import annotations

import math
import re
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPLATE


PROTOCOL_VERSION = "controller_protocol_v2"
BASE_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MEDICAL_LORA_SHA256 = "48c36670eb6a682bbb677c1bf1d80744d758928b8133e5810b0db50721ff143d"
CONTROLLER_ROLES = frozenset({"medical_controller_dev", "general_controller_dev"})
_SUPERVISION_FIELDS = frozenset(
    {"answer", "answer_idx", "answer_index", "gold", "label", "solution", "response"}
)
_LETTERS = "ABCDE"
_ANSWER_FIRST_ANY = re.compile(r"^答案：(\S+)$")
_FINAL_LINE_ANY = re.compile(r"^最终答案：(\S+)$")
_PROMPT_POLICY = {
    "choice_instruction": "请根据题意选择唯一正确选项。不要在此处生成解释。 /no_think",
    "choice_assistant_prefix": "最终答案：",
    "generation_instruction": "请先在第一行输出‘答案：X’，其中 X 是合法选项字母；随后可以给出简短解释。 /no_think",
    "option_order": "upstream_order",
    "labels": {"4": "ABCD", "5": "ABCDE"},
}
_PARSER_POLICY = {
    "allowed": ["first_nonempty_line=答案：X", "line=最终答案：X", "entire_output=X"],
    "conflicting_answers": "invalid",
    "arbitrary_body_letters": "ignored",
}
_SCORER_POLICY = {
    "formal_backend": "transformers_direct_logits",
    "legacy_vllm_prompt_logprobs": "diagnostic_only",
    "candidate": "option_label_string",
    "candidate_tokenization": "contextual_suffix_of_prompt_plus_label",
    "formal_candidate_path": "single_token_last_prompt_position",
    "multi_token_policy": "fail_closed_before_formal_execution",
    "score": "float32_log_softmax_model_logits_last_position",
    "tie_break": "frozen_label_order",
    "label_join": "sample_id_after_execution",
    "batch_size": 1,
    "score_repeat_tolerance": 1e-4,
    "micro_smoke_repeat_count": 3,
}


class ControllerV2Error(RuntimeError):
    """Fail-closed Controller v2 contract violation."""


def protocol_component_hashes() -> dict[str, str]:
    # Lazy import avoids importing any model library while binding the actual
    # execution boundary (not merely its declarative configuration) into the
    # frozen protocol identity.
    from src.eval import controller_v2_runtime as runtime
    from src.eval import direct_logit_scorer as direct
    from src.eval import paired_stats
    from src.opd import teacher_gate
    from src.utils import preflight

    def digest(value: Mapping[str, Any], *implementations: Callable[..., Any]) -> str:
        bound = {
            "contract": value,
            "implementation": [inspect.getsource(item) for item in implementations],
        }
        raw = json.dumps(bound, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    components = {
        "prompt_sha256": digest(
            {
                **_PROMPT_POLICY,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "chat_template": asdict(DEFAULT_TEMPLATE),
            },
            _validate_prompt_row,
            _render_question,
            type(DEFAULT_TEMPLATE).render_prompt,
            build_choice_request,
            build_generative_prompt,
        ),
        "parser_sha256": digest(
            {
                **_PARSER_POLICY,
                "answer_first_pattern": _ANSWER_FIRST_ANY.pattern,
                "final_line_pattern": _FINAL_LINE_ANY.pattern,
            },
            parse_generation_v2,
            runtime.run_generation_rows,
            runtime._apply_vllm_v1_generation_policy,
            runtime._make_vllm_generation_backend,
            runtime._release_vllm_engine,
        ),
        "scorer_sha256": digest(
            {
                **_SCORER_POLICY,
                "expected_qwen3_label_token_ids": direct.EXPECTED_QWEN3_LABEL_TOKEN_IDS,
                "direct_candidate_label_order": direct._LETTERS,
                "controller_supervision_fields": sorted(_SUPERVISION_FIELDS),
                "runtime_supervision_fields": sorted(runtime._SUPERVISION_FIELDS),
            },
            candidate_labels,
            conditional_sequence_logprob,
            score_choice_candidates,
            direct.apply_deterministic_runtime,
            direct._ordered_candidate_ids,
            direct._single_token_candidates,
            direct.verify_qwen3_candidate_token_ids,
            direct.direct_logit_model_plan,
            direct.score_last_prompt_position,
            direct.score_direct_request,
            direct.run_direct_choice_rows,
            direct.load_direct_logit_route,
            direct._index_repetition,
            direct.validate_direct_logit_repetitions,
            runtime.run_direct_logit_micro_smoke,
            runtime.run_direct_logit_full_choice,
            runtime.iter_prompt_rows,
            runtime._label_artifact_attestation,
            runtime.verify_medical_lora_identity,
            runtime.release_model_execution,
            runtime.load_controller_v2_config,
            runtime.controller_v2_cpu_preflight,
            runtime.controller_v2_gpu_preflight,
            runtime._authorized_budget,
            preflight._progress_aware_runtime_gate,
            runtime._label_map,
            runtime._compact_score,
            runtime.summarize_controller_tracks,
            runtime.validate_teacher_readiness_evidence,
            runtime.write_standard_run_artifacts,
            runtime.validate_standard_run_artifacts,
            runtime.run_all_gpu,
            paired_stats.paired_comparison,
            paired_stats.teacher_readiness,
            teacher_gate.assert_teacher_ready_for_opd,
        ),
    }
    components["protocol_sha256"] = digest({
        "protocol_version": PROTOCOL_VERSION,
        "components": components,
        "generation_length": {
            "initial": 512,
            "expanded": 1024,
            "truncation_threshold": 0.01,
            "decision_basis": "truncation_only",
        },
        "teacher_gate": {
            "knowledge_true_delta": 0.03,
            "knowledge_false_below": -0.03,
            "generation_invalid_max": 0.05,
            "generation_truncation_max": 0.01,
        },
        "choice_backend": "transformers_direct_logits",
        "vllm_choice_backend_status": "diagnostic_only",
    })
    return components


@dataclass(frozen=True)
class ChoiceCandidate:
    label: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ChoiceRequest:
    sample_id: str
    target_role: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    labels: tuple[str, ...]
    candidates: tuple[ChoiceCandidate, ...]
    protocol_version: str = PROTOCOL_VERSION

    def execution_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "target_role": self.target_role,
            "protocol_version": self.protocol_version,
            "prompt": self.prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "candidate_tokenization": [asdict(item) for item in self.candidates],
        }


@dataclass(frozen=True)
class ChoicePrediction:
    sample_id: str
    predicted_label: str
    candidate_scores: dict[str, float]
    tie_break_rule: str = "frozen_label_order"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationParse:
    letter: str | None
    method: str
    raw_candidates: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.letter is not None


@dataclass(frozen=True)
class LengthFreezeDecision:
    max_new_tokens: int
    b0_max_new_tokens: int
    b1_max_new_tokens: int
    b0_truncation_rate: float
    b1_truncation_rate: float
    threshold: float
    frozen: bool
    decision_basis: str = "truncation_only"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_labels(option_count: int) -> tuple[str, ...]:
    if option_count not in (4, 5):
        raise ControllerV2Error("Controller v2 requires exactly 4 or 5 ordered options")
    return tuple(_LETTERS[:option_count])


def _validate_prompt_row(row: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    leaked = _SUPERVISION_FIELDS & set(row)
    if leaked:
        raise ControllerV2Error(f"prompt execution row contains supervision fields: {sorted(leaked)}")
    role = str(row.get("target_role") or "")
    if "final" in role:
        raise ControllerV2Error("Controller v2 execution cannot read a final role")
    if role not in CONTROLLER_ROLES:
        raise ControllerV2Error(f"unsupported controller role: {role}")
    sample_id = str(row.get("sample_id") or "")
    options = row.get("options")
    if not sample_id or not isinstance(options, list):
        raise ControllerV2Error("controller prompt requires sample_id and ordered options")
    if any(not str(option).strip() for option in options):
        raise ControllerV2Error("controller options must be non-empty")
    candidate_labels(len(options))
    return sample_id, role, [str(option) for option in options]


def _render_question(row: Mapping[str, Any], instruction: str, suffix: str = "") -> str:
    _, _, options = _validate_prompt_row(row)
    lines = [str(row.get("question") or "").strip(), ""]
    lines.extend(f"{label}. {option.strip()}" for label, option in zip(candidate_labels(len(options)), options))
    lines.extend(("", instruction))
    if suffix:
        lines.extend(("", suffix))
    return DEFAULT_TEMPLATE.render_prompt("\n".join(lines), DEFAULT_SYSTEM_PROMPT)


def build_choice_request(
    row: Mapping[str, Any], *, tokenize: Callable[[str], Sequence[int]]
) -> ChoiceRequest:
    sample_id, role, options = _validate_prompt_row(row)
    labels = candidate_labels(len(options))
    prompt = _render_question(
        row,
        _PROMPT_POLICY["choice_instruction"],
    ) + _PROMPT_POLICY["choice_assistant_prefix"]
    prompt_ids = tuple(int(token) for token in tokenize(prompt))
    if not prompt_ids:
        raise ControllerV2Error("choice prompt tokenized to zero tokens")
    candidates = []
    for label in labels:
        full_ids = tuple(int(token) for token in tokenize(prompt + label))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ControllerV2Error(
                f"candidate {label} changes prompt tokenization at the append boundary"
            )
        ids = full_ids[len(prompt_ids) :]
        if not ids:
            raise ControllerV2Error(f"candidate {label} tokenized to zero tokens")
        candidates.append(ChoiceCandidate(label=label, token_ids=ids))
    return ChoiceRequest(
        sample_id=sample_id,
        target_role=role,
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        labels=labels,
        candidates=tuple(candidates),
    )


def build_generative_prompt(row: Mapping[str, Any]) -> str:
    return _render_question(
        row,
        _PROMPT_POLICY["generation_instruction"],
    )


def conditional_sequence_logprob(
    candidate_token_ids: Sequence[int], candidate_step_logits: Sequence[Sequence[float]]
) -> float:
    """Sum frozen autoregressive conditional log probabilities for one label string."""

    if not candidate_token_ids or len(candidate_token_ids) != len(candidate_step_logits):
        raise ControllerV2Error("candidate tokens and conditional logit steps must be non-empty and aligned")
    total = 0.0
    for token, logits in zip(candidate_token_ids, candidate_step_logits, strict=True):
        if not logits or not 0 <= int(token) < len(logits):
            raise ControllerV2Error("candidate token is outside its conditional-logit vocabulary")
        values = [float(value) for value in logits]
        maximum = max(values)
        log_denom = maximum + math.log(sum(math.exp(value - maximum) for value in values))
        total += values[int(token)] - log_denom
    return total


def score_choice_candidates(
    *,
    sample_id: str,
    candidate_token_ids: Mapping[str, Sequence[int]],
    candidate_logits: Mapping[str, Sequence[Sequence[float]]],
) -> ChoicePrediction:
    if not sample_id or set(candidate_token_ids) != set(candidate_logits):
        raise ControllerV2Error("choice candidate tokenization/logits differ")
    ordered = tuple(sorted(candidate_token_ids, key=_LETTERS.index))
    if ordered not in (tuple(_LETTERS[:4]), tuple(_LETTERS[:5])):
        raise ControllerV2Error("choice scores must contain the complete 4/5-label set")
    scores = {
        label: conditional_sequence_logprob(candidate_token_ids[label], candidate_logits[label])
        for label in ordered
    }
    predicted = max(ordered, key=lambda label: scores[label])
    return ChoicePrediction(sample_id=sample_id, predicted_label=predicted, candidate_scores=scores)


def score_choice_logprobs(
    *, sample_id: str, candidate_token_logprobs: Mapping[str, Sequence[float]]
) -> ChoicePrediction:
    """Score vLLM-extracted actual-token logprobs with the same sequence-sum rule."""

    ordered = tuple(sorted(candidate_token_logprobs, key=_LETTERS.index))
    if ordered not in (tuple(_LETTERS[:4]), tuple(_LETTERS[:5])):
        raise ControllerV2Error("choice logprobs must contain the complete 4/5-label set")
    scores: dict[str, float] = {}
    for label in ordered:
        values = [float(value) for value in candidate_token_logprobs[label]]
        if not values or any(not math.isfinite(value) for value in values):
            raise ControllerV2Error("choice token logprobs must be finite and non-empty")
        scores[label] = sum(values)
    predicted = max(ordered, key=lambda label: scores[label])
    return ChoicePrediction(sample_id=sample_id, predicted_label=predicted, candidate_scores=scores)


def parse_generation_v2(response: str | None, *, option_count: int) -> GenerationParse:
    valid = candidate_labels(option_count)
    text = str(response or "").strip()
    if not text:
        return GenerationParse(None, "empty_output")
    if re.fullmatch(r"[A-E]", text):
        return (
            GenerationParse(text, "single_letter", (text,))
            if text in valid
            else GenerationParse(None, "invalid_option", (text,))
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = _ANSWER_FIRST_ANY.fullmatch(lines[0]) if lines else None
    final = [match.group(1) for line in lines if (match := _FINAL_LINE_ANY.fullmatch(line))]
    candidates = ([first.group(1)] if first else []) + final
    if not candidates:
        # Explicit answer-like forms outside the legal set remain invalid rather than guessed.
        if (lines and lines[0].startswith("答案：")) or any(line.startswith("最终答案：") for line in lines):
            return GenerationParse(None, "invalid_option")
        return GenerationParse(None, "no_explicit_answer")
    if any(letter not in valid for letter in candidates):
        return GenerationParse(None, "invalid_option", tuple(candidates))
    if len(set(candidates)) != 1:
        return GenerationParse(None, "conflicting_answers", tuple(candidates))
    letter = candidates[0]
    if first and final:
        method = "answer_first_and_final_consistent"
    elif first:
        method = "answer_first"
    else:
        method = "final_answer_line"
    return GenerationParse(letter, method, tuple(candidates))


def _truncation_rate(rows: Sequence[Mapping[str, Any]], *, limit: int) -> float:
    if len(rows) != 32:
        raise ControllerV2Error("length smoke requires the same 32 fixed controller samples")
    truncated = 0
    for row in rows:
        count = row.get("generated_token_count")
        reason = row.get("finish_reason")
        explicit = row.get("output_truncated") is True
        if reason == "length" or explicit:
            if type(count) is not int or int(count) < limit:
                raise ControllerV2Error("inconsistent truncation evidence in length smoke")
            truncated += 1
    return truncated / len(rows)


def freeze_generation_limit(
    b0_rows: Sequence[Mapping[str, Any]],
    b1_rows: Sequence[Mapping[str, Any]],
    *,
    initial_limit: int = 512,
    expanded_limit: int = 1024,
    threshold: float = 0.01,
) -> LengthFreezeDecision:
    ids0 = [str(row.get("sample_id")) for row in b0_rows if row.get("sample_id") is not None]
    ids1 = [str(row.get("sample_id")) for row in b1_rows if row.get("sample_id") is not None]
    if ids0 or ids1:
        if len(ids0) != 32 or len(ids1) != 32 or set(ids0) != set(ids1) or len(set(ids0)) != 32:
            raise ControllerV2Error("B0/B1 length smoke must use the same 32 unique sample IDs")
    b0_rate = _truncation_rate(b0_rows, limit=initial_limit)
    b1_rate = _truncation_rate(b1_rows, limit=initial_limit)
    selected = expanded_limit if max(b0_rate, b1_rate) > threshold else initial_limit
    return LengthFreezeDecision(
        max_new_tokens=selected,
        b0_max_new_tokens=selected,
        b1_max_new_tokens=selected,
        b0_truncation_rate=b0_rate,
        b1_truncation_rate=b1_rate,
        threshold=threshold,
        frozen=True,
    )


def validate_prediction_artifact(metadata: Mapping[str, Any]) -> None:
    required = {
        "protocol_version", "capability", "base_model_revision", "medical_lora_sha256",
        "final_authorized", "role", "prediction_artifact",
    }
    if not required.issubset(metadata):
        raise ControllerV2Error("Controller v2 prediction metadata is incomplete")
    if metadata["protocol_version"] != PROTOCOL_VERSION:
        raise ControllerV2Error("prediction protocol is not Controller v2")
    if metadata["capability"] != "controller_eval" or "final" in str(metadata["role"]):
        raise ControllerV2Error("Controller v2 artifact cannot use final capability/roles")
    if metadata["role"] not in CONTROLLER_ROLES or metadata["final_authorized"] is not False:
        raise ControllerV2Error("controller role/final authorization is invalid")
    if metadata["base_model_revision"] != BASE_MODEL_REVISION:
        raise ControllerV2Error("base model revision drift")
    if metadata["medical_lora_sha256"] != MEDICAL_LORA_SHA256:
        raise ControllerV2Error("Medical LoRA SHA drift")
    if not Path(str(metadata["prediction_artifact"])).is_file():
        raise ControllerV2Error("prediction artifact is missing")
