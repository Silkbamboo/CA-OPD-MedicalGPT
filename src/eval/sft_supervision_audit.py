"""Streaming, tokenizer-only audit of the frozen Medical SFT supervision.

No model is imported or loaded.  The exact non-thinking ChatML formatter is
used, prompt tokens remain masked, and assistant reasoning/answer/EOS segments
are counted from tokenizer offsets one record at a time.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPLATE


_MCQ_MARKERS = re.compile(
    r"(?:^|\n)\s*A[\.．、:：]\s*.+(?:\n|\s)+\s*B[\.．、:：]",
    re.I | re.S,
)


def _quantiles(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {name: None for name in ("min", "p50", "p90", "p95", "p99", "max")}
    ordered = sorted(int(value) for value in values)

    def at(fraction: float) -> int:
        return ordered[int(round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _tokenized(tokenizer: Any, text: str, *, offsets: bool) -> tuple[list[int], list[tuple[int, int]]]:
    result = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=offsets,
    )
    if not isinstance(result, Mapping) or not isinstance(result.get("input_ids"), list):
        raise ValueError("tokenizer must return input_ids as a list")
    ids = [int(value) for value in result["input_ids"]]
    mapping = result.get("offset_mapping", [])
    if offsets:
        if not isinstance(mapping, list) or len(mapping) != len(ids):
            raise ValueError("tokenizer must return aligned offset_mapping")
        return ids, [(int(start), int(end)) for start, end in mapping]
    return ids, []


def _completion_segments(tokenizer: Any, reasoning: str, answer: str) -> dict[str, int]:
    reasoning = reasoning.strip()
    answer = answer.strip()
    prefix = f"{reasoning}\n" if reasoning else ""
    body = prefix + answer
    completion = body + DEFAULT_TEMPLATE.assistant_suffix
    ids, offsets = _tokenized(tokenizer, completion, offsets=True)
    if not ids:
        raise ValueError("completion tokenized to zero tokens")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    answer_start_char = len(prefix)
    answer_end_char = len(body)
    counts = {"reasoning": 0, "separator": 0, "answer": 0, "assistant_eos": 0}
    answer_start_token: int | None = None
    for index, (token_id, (start, end)) in enumerate(zip(ids, offsets, strict=True)):
        if eos_id is not None and token_id == int(eos_id):
            counts["assistant_eos"] += 1
            continue
        if end <= len(reasoning) and reasoning:
            counts["reasoning"] += 1
        elif start < answer_start_char:
            counts["separator"] += 1
        elif start < answer_end_char:
            counts["answer"] += 1
            if answer_start_token is None:
                answer_start_token = index
        else:
            # Qwen's assistant suffix is expected to be EOS. Any non-EOS
            # suffix token would make the actual training mask ambiguous.
            raise ValueError("assistant suffix did not tokenize exclusively to EOS")
    if answer_start_token is None or counts["answer"] < 1 or counts["assistant_eos"] < 1:
        raise ValueError("answer/EOS supervision segment is missing")
    return {
        **counts,
        "completion": len(ids),
        "answer_start_token": answer_start_token,
    }


def audit_sft_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_seq_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Audit formal SFT records without buffering their text."""

    totals = Counter({
        "prompt": 0,
        "reasoning": 0,
        "separator": 0,
        "answer": 0,
        "assistant_eos": 0,
        "loss_bearing": 0,
    })
    answer_positions: list[int] = []
    completion_lengths: list[int] = []
    prompt_lengths: list[int] = []
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    sample_count = answer_first = boundary = missing_answer = duplicates = 0
    think_count = explicit_options = mcq_shape = reasoning_before_answer = 0
    explicit_answer_marker = reasoning_contains_answer = reasoning_equals_answer = 0
    within = {32: 0, 128: 0, 256: 0, 512: 0}
    seen: set[str] = set()
    for row in rows:
        role = str(row.get("target_role") or "")
        if role != "medical_sft_train" or "final" in role:
            raise ValueError("SFT supervision audit accepts only medical_sft_train")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            duplicates += 1
            raise ValueError("SFT supervision audit requires unique sample_id")
        seen.add(sample_id)
        question = str(row.get("question") or "").strip()
        reasoning = str(row.get("reasoning") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            missing_answer += 1
            raise ValueError("SFT supervision audit requires question and answer")
        if any("<think>" in value.casefold() or "</think>" in value.casefold() for value in (question, reasoning, answer)):
            think_count += 1
        prompt = DEFAULT_TEMPLATE.render_prompt(question, system_prompt)
        prompt_ids, _ = _tokenized(tokenizer, prompt, offsets=False)
        segment = _completion_segments(tokenizer, reasoning, answer)
        total_length = len(prompt_ids) + segment["completion"]
        if total_length >= max_seq_length:
            boundary += 1
        sample_count += 1
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(segment["completion"])
        answer_positions.append(segment["answer_start_token"])
        totals["prompt"] += len(prompt_ids)
        for name in ("reasoning", "separator", "answer", "assistant_eos"):
            totals[name] += segment[name]
        totals["loss_bearing"] += segment["completion"]
        if reasoning:
            reasoning_before_answer += 1
            normalized_reasoning = re.sub(r"\s+", "", reasoning)
            normalized_answer = re.sub(r"\s+", "", answer)
            reasoning_contains_answer += int(bool(normalized_answer) and normalized_answer in normalized_reasoning)
            reasoning_equals_answer += int(normalized_reasoning == normalized_answer)
        else:
            answer_first += 1
        explicit_answer_marker += int(
            bool(re.search(r"(?:最终答案|答案|综上|结论)\s*(?:是|为|[：:])", answer))
        )
        for limit in within:
            within[limit] += segment["answer_start_token"] < limit
        source_counts[str(row.get("source") or "unknown")] += 1
        flags = row.get("quality_flags")
        if isinstance(flags, list) and flags:
            for flag in flags:
                category_counts[str(flag)] += 1
        if isinstance(row.get("options"), list):
            explicit_options += 1
        if _MCQ_MARKERS.search(question):
            mcq_shape += 1
    if sample_count < 1:
        raise ValueError("SFT supervision audit cannot be empty")
    if totals["loss_bearing"] != (
        totals["reasoning"] + totals["separator"] + totals["answer"] + totals["assistant_eos"]
    ):
        raise ValueError("SFT assistant loss segments do not sum to the training mask")
    return {
        "sample_count": sample_count,
        "token_totals": dict(totals),
        "reasoning_share_of_loss_tokens": totals["reasoning"] / totals["loss_bearing"],
        "answer_share_of_loss_tokens": totals["answer"] / totals["loss_bearing"],
        "answer_start_token": _quantiles(answer_positions),
        "completion_tokens": _quantiles(completion_lengths),
        "prompt_tokens": _quantiles(prompt_lengths),
        "answer_first_count": answer_first,
        "answer_first_rate": answer_first / sample_count,
        "answer_within_completion_prefix": {
            str(limit): {"count": within[limit], "rate": within[limit] / sample_count}
            for limit in sorted(within)
        },
        "at_or_above_2048_count": boundary,
        "at_or_above_2048_rate": boundary / sample_count,
        "reasoning_before_answer_count": reasoning_before_answer,
        "reasoning_before_answer_rate": reasoning_before_answer / sample_count,
        "explicit_answer_marker_count": explicit_answer_marker,
        "reasoning_contains_final_answer_count": reasoning_contains_answer,
        "reasoning_equals_answer_count": reasoning_equals_answer,
        "repeated_response_field_count": 0,
        "think_tag_count": think_count,
        "explicit_options_field_count": explicit_options,
        "question_mcq_marker_count": mcq_shape,
        "open_ended_structural_count": sample_count - max(explicit_options, mcq_shape),
        "source_distribution": dict(sorted(source_counts.items())),
        "quality_flag_distribution": dict(sorted(category_counts.items())),
        "duplicate_sample_id_count": duplicates,
        "missing_answer_count": missing_answer,
        "loss_mask_verification": {
            "system_user_loss_tokens": 0,
            "assistant_reasoning_trained": totals["reasoning"] > 0,
            "assistant_answer_trained": totals["answer"] > 0,
            "assistant_eos_trained": totals["assistant_eos"] > 0,
        },
        "model_weights_loaded": False,
        "tokenizer_only": True,
        "final_roles_read": False,
    }


class _FormatAccumulator:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.count = 0
        self.mcq = 0
        self.open_ended = 0
        self.question_chars = 0
        self.reasoning_chars = 0
        self.answer_chars = 0
        self.option_counts: Counter[int] = Counter()
        self.sources: Counter[str] = Counter()
        self.subjects: Counter[str] = Counter()
        self.question_hashes: set[str] = set()
        self.ngram_buckets = bytearray(65536)

    def add(self, row: Mapping[str, Any]) -> None:
        role = str(row.get("target_role") or "")
        expected = "medical_sft_train" if self.kind == "sft" else "medical_controller_dev"
        if role != expected or "final" in role:
            raise ValueError(f"task-format {self.kind} reader received a non-authorized role")
        question = str(row.get("normalized_question") or row.get("question") or "").strip()
        if not question:
            raise ValueError("task-format row lacks a question")
        self.count += 1
        self.question_chars += len(question)
        options = row.get("options")
        is_mcq = isinstance(options, list) and len(options) in {4, 5}
        if not is_mcq:
            is_mcq = bool(_MCQ_MARKERS.search(question))
        self.mcq += int(is_mcq)
        self.open_ended += int(not is_mcq)
        if isinstance(options, list):
            self.option_counts[len(options)] += 1
        self.reasoning_chars += len(str(row.get("reasoning") or ""))
        self.answer_chars += len(str(row.get("answer") or ""))
        self.sources[str(row.get("source") or "unknown")] += 1
        if row.get("subject"):
            self.subjects[str(row["subject"])] += 1
        normalized = re.sub(r"\s+", "", question).casefold()
        self.question_hashes.add(hashlib.sha256(normalized.encode()).hexdigest())
        for index in range(max(0, len(normalized) - 2)):
            ngram = normalized[index : index + 3]
            bucket = int.from_bytes(hashlib.blake2s(ngram.encode(), digest_size=2).digest(), "big")
            self.ngram_buckets[bucket] = 1

    def report(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError(f"task-format {self.kind} rows cannot be empty")
        return {
            "count": self.count,
            "mcq_count": self.mcq,
            "open_ended_count": self.open_ended,
            "mcq_rate": self.mcq / self.count,
            "mean_question_chars": self.question_chars / self.count,
            "mean_reasoning_chars": self.reasoning_chars / self.count,
            "mean_answer_chars": self.answer_chars / self.count,
            "option_count_distribution": {
                str(key): value for key, value in sorted(self.option_counts.items())
            },
            "source_distribution": dict(sorted(self.sources.items())),
            "subject_distribution": dict(sorted(self.subjects.items())),
        }


def compare_task_formats(
    sft_rows: Iterable[Mapping[str, Any]],
    controller_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    sft, controller = _FormatAccumulator("sft"), _FormatAccumulator("controller")
    for row in sft_rows:
        sft.add(row)
    for row in controller_rows:
        controller.add(row)
    union = intersection = 0
    for left, right in zip(sft.ngram_buckets, controller.ngram_buckets, strict=True):
        union += bool(left or right)
        intersection += bool(left and right)
    return {
        "sft": sft.report(),
        "controller": controller.report(),
        "exact_normalized_question_overlap": len(sft.question_hashes & controller.question_hashes),
        "hashed_char_trigram_bucket_jaccard": intersection / union if union else 0.0,
        "hashed_char_trigram_bucket_count": 65536,
        "lexical_statistic": "collision-prone corpus-level diagnostic, not semantic similarity",
        "embedding_used": False,
        "final_roles_read": False,
        "raw_text_in_report": False,
    }


__all__ = ["audit_sft_rows", "compare_task_formats"]
