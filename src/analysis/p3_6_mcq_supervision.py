"""CPU-only P3.6 supervision and controller evidence audits.

The module imports no model runtime and never opens final or confirmation data.
Callers supply already-authorized controller labels only at the independent
scoring boundary. Training rows are rendered through the production SFT-v2
formatter so token weights and boundaries are measured rather than inferred.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from src.eval.controller_v2 import build_choice_request
from src.eval.paired_stats import paired_comparison, score_label_free_predictions
from src.sft.ddp import distributed_sample_indices
from src.sft.weighted import SupervisionWeights, render_sft_v2_row


class P36AuditError(RuntimeError):
    """A frozen P3.6 audit contract was violated."""


def summarize_repeatability(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and compact the frozen direct-logit micro-smoke evidence."""

    valid = (
        payload.get("status") == "PASS"
        and payload.get("choice_backend") == "transformers_direct_logits"
        and int(payload.get("repeat_count", -1)) == 3
        and int(payload.get("sample_count", -1)) == 4
        and float(payload.get("max_abs_score_delta", float("inf"))) == 0.0
        and payload.get("candidate_ordering_deterministic") is True
        and payload.get("labels_opened_during_execution") is False
    )
    if not valid:
        raise ValueError("P3.5 direct-logit repeatability evidence drift")
    return {
        "status": "PASS",
        "backend": "transformers_direct_logits",
        "repeat_count": 3,
        "sample_count": 4,
        "score_repeat_max_abs_diff": 0.0,
        "candidate_ordering_deterministic": True,
        "labels_opened_during_execution": False,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    medical = [row for row in rows if row.get("domain") == "medical"]
    general = [row for row in rows if row.get("domain") == "general"]
    per_subject: dict[str, dict[str, Any]] = {}
    for subject in sorted({str(row.get("subject") or "unknown") for row in general}):
        members = [row for row in general if str(row.get("subject") or "unknown") == subject]
        correct = sum(bool(row["correct"]) for row in members)
        per_subject[subject] = {
            "correct": correct,
            "total": len(members),
            "accuracy": correct / len(members),
        }
    medical_correct = sum(bool(row["correct"]) for row in medical)
    general_correct = sum(bool(row["correct"]) for row in general)
    return {
        "medical_correct": medical_correct,
        "medical_total": len(medical),
        "medical_accuracy": medical_correct / len(medical) if medical else 0.0,
        "general_correct": general_correct,
        "general_total": len(general),
        "general_micro_accuracy": general_correct / len(general) if general else 0.0,
        "general_macro_accuracy": (
            sum(value["accuracy"] for value in per_subject.values()) / len(per_subject)
            if per_subject
            else 0.0
        ),
        "per_subject": per_subject,
    }


def recompute_checkpoint_metrics(
    *,
    b0_predictions: Iterable[Mapping[str, Any]],
    checkpoint_predictions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Join controller labels after execution and recompute paired evidence."""

    label_rows = [dict(row) for row in labels]
    if any("final" in str(row.get("target_role") or "") for row in label_rows):
        raise PermissionError("P3.6 controller audit cannot read final labels")
    b0_scored = score_label_free_predictions(b0_predictions, label_rows)
    checkpoint_scored = score_label_free_predictions(checkpoint_predictions, label_rows)
    paired = paired_comparison(
        b0_scored,
        checkpoint_scored,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    b0_index = {str(row["sample_id"]): row for row in b0_scored}
    checkpoint_index = {str(row["sample_id"]): row for row in checkpoint_scored}
    medical_transitions = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    prediction_changes = 0
    for sample_id in sorted(b0_index):
        before = b0_index[sample_id]
        after = checkpoint_index[sample_id]
        if before.get("predicted_label") != after.get("predicted_label"):
            prediction_changes += 1
        if before.get("domain") != "medical":
            continue
        key = (
            ("correct" if before["correct"] else "wrong")
            + "_to_"
            + ("correct" if after["correct"] else "wrong")
        )
        medical_transitions[key] += 1
    return {
        "b0": _metric_summary(b0_scored),
        "checkpoint": _metric_summary(checkpoint_scored),
        "paired": paired,
        "medical_transitions": medical_transitions,
        "prediction_changes": prediction_changes,
        "sample_ids_equal": set(b0_index) == set(checkpoint_index),
        "labels_joined_after_execution": True,
        "final_authorized": False,
    }


def _kind(row: Mapping[str, Any]) -> str:
    value = str(row.get("sft_v2_kind") or "")
    if value == "medical_o1_answer_first":
        return "medical_o1"
    if value == "cmb_mcq_bridge":
        return "cmb"
    raise P36AuditError(f"unsupported SFT-v2 kind: {value}")


def _empty_source_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "prompt_tokens": 0,
        "answer_tokens": 0,
        "reasoning_tokens": 0,
        "eos_tokens": 0,
        "nonzero_weight_tokens": 0,
        "weighted_denominator": 0.0,
        "answer_weighted_denominator": 0.0,
        "reasoning_weighted_denominator": 0.0,
        "eos_weighted_denominator": 0.0,
    }


def _rate(count: int, total: int) -> dict[str, float | int]:
    return {"count": int(count), "rate": count / total if total else 0.0}


def aggregate_supervision_stats(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    weights: SupervisionWeights,
    max_seq_length: int,
    system_prompt: str,
    seed: int,
    world_size: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    """Measure exact rendered SFT-v2 weights and reconstruct DDP windows."""

    compact: list[dict[str, Any]] = []
    sources = {"medical_o1": _empty_source_stats(), "cmb": _empty_source_stats()}
    seen: set[str] = set()
    for source_row in rows:
        row = dict(source_row)
        role = str(row.get("target_role") or "")
        if "final" in role or "confirmation" in role:
            raise PermissionError("P3.6 supervision audit cannot read final/confirmation roles")
        if role != "medical_sft_train":
            raise P36AuditError("supervision audit requires medical_sft_train")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError("missing or duplicate supervision sample_id")
        seen.add(sample_id)
        example = render_sft_v2_row(
            row,
            tokenizer=tokenizer,
            weights=weights,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
        )
        if example is None:
            raise P36AuditError("frozen SFT-v2 row unexpectedly exceeds max length")
        kind = _kind(row)
        denominator = float(sum(example.loss_weights))
        if not math.isfinite(denominator) or denominator <= 0:
            raise P36AuditError("rendered row has no finite supervision")
        stats = sources[kind]
        stats["rows"] += 1
        stats["prompt_tokens"] += example.prompt_length
        stats["answer_tokens"] += example.segment_token_counts["answer"]
        stats["reasoning_tokens"] += example.segment_token_counts["reasoning"]
        stats["eos_tokens"] += example.segment_token_counts["eos"]
        stats["nonzero_weight_tokens"] += sum(weight > 0 for weight in example.loss_weights)
        stats["weighted_denominator"] += denominator
        stats["answer_weighted_denominator"] += example.segment_weighted_contribution["answer"]
        stats["reasoning_weighted_denominator"] += example.segment_weighted_contribution["reasoning"]
        stats["eos_weighted_denominator"] += example.segment_weighted_contribution["eos"]
        first = next(index for index, weight in enumerate(example.loss_weights) if weight > 0)
        compact.append(
            {
                "sample_id": sample_id,
                "kind": kind,
                "weighted_denominator": denominator,
                "sequence_length": len(example.input_ids),
                "first_supervised_token_id": int(example.input_ids[first]),
                "last_supervised_token_id": int(example.input_ids[-1]),
            }
        )

    if len(compact) % world_size:
        raise P36AuditError("frozen dataset must divide evenly across DDP ranks")
    total_denominator = sum(float(row["weighted_denominator"]) for row in compact)
    for stats in sources.values():
        stats["row_share"] = stats["rows"] / len(compact) if compact else 0.0
        stats["weighted_denominator_share"] = (
            stats["weighted_denominator"] / total_denominator if total_denominator else 0.0
        )
        stats["mean_weighted_denominator_per_row"] = (
            stats["weighted_denominator"] / stats["rows"] if stats["rows"] else 0.0
        )

    rank_indices = [
        distributed_sample_indices(
            len(compact), rank=rank, world_size=world_size, seed=seed, epoch=0
        )
        for rank in range(world_size)
    ]
    flattened = [index for values in rank_indices for index in values]
    duplicates = len(flattened) - len(set(flattened))
    missing = len(compact) - len(set(flattened))
    windows = math.ceil(len(rank_indices[0]) / accumulation_steps)
    window_rows: list[dict[str, Any]] = []
    for window_index in range(windows):
        source_denominator = {"medical_o1": 0.0, "cmb": 0.0}
        source_rows = {"medical_o1": 0, "cmb": 0}
        start = window_index * accumulation_steps
        stop = start + accumulation_steps
        for indices in rank_indices:
            for index in indices[start:stop]:
                item = compact[index]
                source_denominator[item["kind"]] += float(item["weighted_denominator"])
                source_rows[item["kind"]] += 1
        total = sum(source_denominator.values())
        window_rows.append(
            {
                "rows": source_rows,
                "weighted_denominator": source_denominator,
                "cmb_weighted_share": source_denominator["cmb"] / total if total else 0.0,
            }
        )
    shares = [row["cmb_weighted_share"] for row in window_rows]
    mixed = [row for row in window_rows if all(row["rows"][kind] for kind in ("medical_o1", "cmb"))]
    cmb_row_share = sources["cmb"]["row_share"]
    mean_o1 = sources["medical_o1"]["mean_weighted_denominator_per_row"]
    mean_cmb = sources["cmb"]["mean_weighted_denominator_per_row"]

    def schedule(cmb_steps: int, o1_steps: int) -> dict[str, Any]:
        budget = cmb_steps * mean_cmb + o1_steps * mean_o1
        return {
            "task_step_share": {
                "cmb": cmb_steps / (cmb_steps + o1_steps),
                "medical_o1": o1_steps / (cmb_steps + o1_steps),
            },
            "theoretical_weighted_denominator_share": {
                "cmb": cmb_steps * mean_cmb / budget if budget else 0.0,
                "medical_o1": o1_steps * mean_o1 / budget if budget else 0.0,
            },
        }

    return {
        "rows": len(compact),
        "sources": sources,
        "total_weighted_denominator": total_denominator,
        "medical_o1_to_cmb_mean_denominator_ratio": mean_o1 / mean_cmb if mean_cmb else None,
        "sample_coverage": {
            "world_size": world_size,
            "rank_counts": [len(values) for values in rank_indices],
            "duplicates": duplicates,
            "missing": missing,
        },
        "window_dilution": {
            "windows": len(window_rows),
            "mixed_windows": len(mixed),
            "windows_without_cmb": sum(row["rows"]["cmb"] == 0 for row in window_rows),
            "mixed_windows_cmb_weighted_share_below_global_row_share": sum(
                row["cmb_weighted_share"] < cmb_row_share for row in mixed
            ),
            "cmb_weighted_share": {
                "min": min(shares) if shares else 0.0,
                "p50": _quantile(shares, 0.50),
                "p90": _quantile(shares, 0.90),
                "max": max(shares) if shares else 0.0,
            },
        },
        "theoretical_task_schedules": {"3:1": schedule(3, 1), "4:1": schedule(4, 1)},
        "final_authorized": False,
    }


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode([int(token_id)]))


def _label(row: Mapping[str, Any], option_count: int) -> str:
    raw = row.get("answer_idx")
    if isinstance(raw, str) and raw.strip().upper() in "ABCDE":
        value = raw.strip().upper()
    else:
        value = chr(ord("A") + int(raw))
    if value not in "ABCDE"[:option_count]:
        raise P36AuditError("CMB answer label is outside legal candidates")
    return value


def _target_offsets(tokenizer: Any, text: str, token_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Return real fast-tokenizer offsets, with a character-token fixture fallback."""

    if callable(tokenizer):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(start), int(stop)) for start, stop in encoded["offset_mapping"]]
        if encoded_ids != list(token_ids) or len(offsets) != len(token_ids):
            raise P36AuditError("tokenizer offsets differ from production target tokenization")
        return offsets
    offsets = []
    cursor = 0
    for token_id in token_ids:
        piece = _decode_one(tokenizer, int(token_id))
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    if cursor != len(text):
        raise P36AuditError("fixture tokenizer cannot reconstruct target offsets")
    return offsets


def _classify_target_segments(
    tokenizer: Any, target_text: str, token_ids: Sequence[int], answer: str
) -> Counter[str]:
    prefix_stop = len("答案：")
    letter_stop = prefix_stop + len(answer)
    option_start = len(f"答案：{answer}. ")
    regions = {
        "prefix": (0, prefix_stop),
        "answer_letter": (prefix_stop, letter_stop),
        "separator": (letter_stop, option_start),
        "option_text": (option_start, len(target_text)),
    }
    counts: Counter[str] = Counter()
    for start, stop in _target_offsets(tokenizer, target_text, token_ids):
        hits = [
            name
            for name, (region_start, region_stop) in regions.items()
            if start < region_stop and stop > region_start
        ]
        if len(hits) == 1:
            counts[hits[0]] += 1
        else:
            counts["boundary_mixed"] += 1
    return counts


def audit_cmb_first_token_alignment(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    weights: SupervisionWeights,
    max_seq_length: int,
    system_prompt: str,
    snapshot_limit: int = 20,
) -> dict[str, Any]:
    """Compare the real SFT-v2 first supervised token with direct-logit labels."""

    if snapshot_limit < 0 or snapshot_limit > 20:
        raise ValueError("redacted snapshot limit must be between zero and 20")
    total = aligned = prefixed = whitespace = candidate_single = label_position_matches = 0
    multiple_label_tokens = 0
    eos_immediately_after_label = 0
    answer_distribution: Counter[str] = Counter()
    subject_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    first_token_distribution: Counter[str] = Counter()
    target_segments: Counter[str] = Counter()
    candidate_ids_by_label: dict[str, set[int]] = defaultdict(set)
    snapshots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in rows:
        row = dict(source_row)
        if row.get("sft_v2_kind") != "cmb_mcq_bridge":
            raise P36AuditError("first-token audit accepts only CMB bridge rows")
        if row.get("target_role") != "medical_sft_train":
            raise PermissionError("first-token audit cannot read non-training/final roles")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError("missing or duplicate CMB sample_id")
        seen.add(sample_id)
        options = [str(value) for value in row.get("options") or []]
        answer = _label(row, len(options))
        example = render_sft_v2_row(
            row,
            tokenizer=tokenizer,
            weights=weights,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
        )
        if example is None:
            raise P36AuditError("frozen CMB row unexpectedly exceeds max length")
        prompt_row = {
            "sample_id": sample_id,
            "target_role": "medical_controller_dev",
            "question": str(row.get("question") or ""),
            "options": options,
        }
        request = build_choice_request(
            prompt_row, tokenize=lambda text: _tokenize(tokenizer, text)
        )
        candidate = next(item for item in request.candidates if item.label == answer)
        if len(candidate.token_ids) == 1:
            candidate_single += 1
            candidate_id = int(candidate.token_ids[0])
            candidate_ids_by_label[answer].add(candidate_id)
        else:
            multiple_label_tokens += 1
            candidate_id = -1
        first_index = next(index for index, weight in enumerate(example.loss_weights) if weight > 0)
        first_id = int(example.input_ids[first_index])
        first_text = _decode_one(tokenizer, first_id)
        first_token_distribution[first_text] += 1
        is_aligned = candidate_id >= 0 and first_id == candidate_id
        aligned += int(is_aligned)
        starts_prefix = example.target_text.startswith("答案：")
        prefixed += int(starts_prefix)
        whitespace += int(bool(example.target_text[:1].isspace()))
        target_ids = example.input_ids[example.prompt_length : -1]
        answer_target_ids = _tokenize(tokenizer, example.target_text)
        if target_ids != answer_target_ids:
            raise P36AuditError("SFT target tokenization drifted from the production renderer")
        target_segments.update(
            _classify_target_segments(tokenizer, example.target_text, target_ids, answer)
        )
        target_segments["eos"] += 1
        prefix_ids = _tokenize(tokenizer, "答案：")
        if len(target_ids) > len(prefix_ids) and candidate_id >= 0:
            label_position_matches += int(target_ids[len(prefix_ids)] == candidate_id)
            eos_immediately_after_label += int(
                len(target_ids) == len(prefix_ids) + 1
                and int(example.input_ids[-1]) == int(getattr(tokenizer, "eos_token_id"))
            )
        answer_distribution[answer] += 1
        subject_distribution[str(row.get("subject") or "unknown")][answer] += 1
        if len(snapshots) < snapshot_limit:
            snapshots.append(
                {
                    "sample_id_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
                    "answer_label": answer,
                    "first_supervised_token_id": first_id,
                    "first_supervised_text": first_text,
                    "candidate_token_ids": list(candidate.token_ids),
                    "prompt_tokens": example.prompt_length,
                    "target_tokens_excluding_eos": len(target_ids),
                    "alignment": "aligned" if is_aligned else (
                        "prefix_before_candidate" if starts_prefix else "other_mismatch"
                    ),
                }
            )
        total += 1
    frozen_ids: dict[str, int] = {}
    for label, values in sorted(candidate_ids_by_label.items()):
        if len(values) != 1:
            raise P36AuditError(f"candidate token ID is inconsistent for {label}")
        frozen_ids[label] = next(iter(values))
    return {
        "rows": total,
        "first_supervised_is_candidate": _rate(aligned, total),
        "first_supervised_starts_with_answer_prefix": _rate(prefixed, total),
        "first_supervised_starts_with_whitespace": _rate(whitespace, total),
        "candidate_single_token": _rate(candidate_single, total),
        "multi_token_answer_label": _rate(multiple_label_tokens, total),
        "label_position_token_matches_candidate": _rate(label_position_matches, total),
        "eos_immediately_after_answer_label": _rate(eos_immediately_after_label, total),
        "candidate_token_ids": frozen_ids,
        "first_supervised_token_distribution": dict(first_token_distribution.most_common()),
        "training_target_token_segments": {
            name: int(target_segments[name])
            for name in (
                "prefix",
                "answer_letter",
                "separator",
                "option_text",
                "boundary_mixed",
                "eos",
            )
        },
        "answer_label_distribution": dict(sorted(answer_distribution.items())),
        "subject_label_distribution": {
            subject: dict(sorted(values.items()))
            for subject, values in sorted(subject_distribution.items())
        },
        "redacted_snapshots": snapshots,
        "observed_protocol_mismatch": aligned != total,
        "final_authorized": False,
    }


def _unique_index(rows: Iterable[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        role = str(row.get("target_role") or "")
        if "final" in role or "confirmation" in role:
            raise PermissionError(f"P3.6 {name} cannot read final/confirmation roles")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in index:
            raise ValueError(f"missing or duplicate {name} sample_id")
        index[sample_id] = row
    return index


def _question_length_bucket(length: int) -> str:
    if length <= 64:
        return "000-064"
    if length <= 128:
        return "065-128"
    if length <= 256:
        return "129-256"
    return "257+"


def analyze_error_and_coverage(
    *,
    b0_predictions: Iterable[Mapping[str, Any]],
    checkpoint_predictions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    controller_prompts: Iterable[Mapping[str, Any]],
    cmb_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate B0-to-checkpoint transitions and source coverage without raw text."""

    b0 = _unique_index(b0_predictions, name="B0 prediction")
    checkpoint = _unique_index(checkpoint_predictions, name="checkpoint prediction")
    label_index = _unique_index(labels, name="controller label")
    prompt_index = _unique_index(controller_prompts, name="controller prompt")
    if set(b0) != set(checkpoint) or set(b0) != set(label_index):
        raise P36AuditError("B0/checkpoint/label sample sets differ")
    if prompt_index and set(prompt_index) != set(b0):
        raise P36AuditError("controller prompt sample set differs from predictions")

    transitions: Counter[str] = Counter()
    transition_subjects: dict[str, Counter[str]] = defaultdict(Counter)
    b0_errors: Counter[str] = Counter()
    by_answer: dict[str, Counter[str]] = defaultdict(Counter)
    by_length: dict[str, Counter[str]] = defaultdict(Counter)
    by_option_count: dict[str, Counter[str]] = defaultdict(Counter)
    for sample_id in sorted(b0):
        before = b0[sample_id]
        after = checkpoint[sample_id]
        if before.get("domain") != "medical" or after.get("domain") != "medical":
            raise P36AuditError("P3.6 medical error analysis received a non-medical prediction")
        label_row = label_index[sample_id]
        label = str(label_row.get("answer_idx") or "").strip().upper()
        if label not in "ABCDE":
            raise P36AuditError("controller label is outside A-E")
        before_correct = str(before.get("predicted_label")) == label
        after_correct = str(after.get("predicted_label")) == label
        transition = (
            ("correct" if before_correct else "wrong")
            + "_to_"
            + ("correct" if after_correct else "wrong")
        )
        transitions[transition] += 1
        prompt = prompt_index.get(sample_id, before)
        subject = str(prompt.get("subject") or before.get("subject") or "unknown")
        transition_subjects[transition][subject] += 1
        if not before_correct:
            b0_errors[subject] += 1
        by_answer[label][transition] += 1
        question_length = len(str(prompt.get("question") or ""))
        by_length[_question_length_bucket(question_length)][transition] += 1
        option_count = len(prompt.get("options") or [])
        by_option_count[str(option_count)][transition] += 1

    cmb_subjects: Counter[str] = Counter()
    cmb_categories: Counter[str] = Counter()
    cmb_answers: Counter[str] = Counter()
    cmb_rows_count = 0
    for source in cmb_rows:
        row = dict(source)
        role = str(row.get("target_role") or "")
        if "final" in role or "confirmation" in role:
            raise PermissionError("P3.6 CMB coverage cannot read final/confirmation roles")
        if role != "medical_sft_train" or row.get("sft_v2_kind") != "cmb_mcq_bridge":
            raise P36AuditError("CMB coverage requires frozen SFT-v2 bridge rows")
        options = list(row.get("options") or [])
        answer = _label(row, len(options))
        cmb_subjects[str(row.get("subject") or "unknown")] += 1
        cmb_categories[str(row.get("category") or "unknown")] += 1
        cmb_answers[answer] += 1
        cmb_rows_count += 1

    controller_subjects = {
        str((prompt_index.get(sample_id) or b0[sample_id]).get("subject") or "unknown")
        for sample_id in b0
    }
    exact_overlap = sorted(controller_subjects & set(cmb_subjects))

    def plain(counter: Counter[str]) -> dict[str, int]:
        return {key: int(counter[key]) for key in sorted(counter)}

    return {
        "rows": len(b0),
        "transitions": {
            name: int(transitions[name])
            for name in (
                "correct_to_correct",
                "correct_to_wrong",
                "wrong_to_correct",
                "wrong_to_wrong",
            )
        },
        "b0_errors_by_subject": plain(b0_errors),
        "improved_by_subject": plain(transition_subjects["wrong_to_correct"]),
        "regressed_by_subject": plain(transition_subjects["correct_to_wrong"]),
        "transitions_by_answer_label": {
            key: plain(value) for key, value in sorted(by_answer.items())
        },
        "transitions_by_question_length_chars": {
            key: plain(value) for key, value in sorted(by_length.items())
        },
        "transitions_by_option_count": {
            key: plain(value) for key, value in sorted(by_option_count.items())
        },
        "cmb_coverage": {
            "rows": cmb_rows_count,
            "subject": plain(cmb_subjects),
            "category": plain(cmb_categories),
            "answer_label": plain(cmb_answers),
        },
        "controller_subject_count": len(controller_subjects),
        "cmb_subject_count": len(cmb_subjects),
        "exact_subject_overlap": exact_overlap,
        "raw_text_retained": False,
        "final_authorized": False,
    }
