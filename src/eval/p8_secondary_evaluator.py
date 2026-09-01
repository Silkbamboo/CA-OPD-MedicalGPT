"""Frozen CPU-safe helpers for the P8 secondary generative diagnostic."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random
import re
import statistics
from typing import Any, Mapping, Sequence

from src.data.adapters import AdapterContext, adapt_source_row
from src.data.schema import stable_sample_id_v2


FORBIDDEN_LABEL_FIELDS = frozenset(
    {"label", "answer", "answer_idx", "correct", "gold", "target", "solution"}
)
ANSWER_MARKER = re.compile(
    r"(?:最终答案|答案(?:为|是)|the\s+answer\s+is)\s*[:：]?\s*[\(（\[]?\s*([A-Z])\s*[\)）\]]?",
    flags=re.IGNORECASE,
)


def select_label_free_subset(
    rows: Sequence[Mapping[str, Any]], *, medical_count: int, general_count: int
) -> list[dict[str, Any]]:
    for row in rows:
        forbidden = FORBIDDEN_LABEL_FIELDS.intersection(row)
        if forbidden:
            raise ValueError("label-bearing field is forbidden before generation freeze")
    selected: list[dict[str, Any]] = []
    for domain, count in (("medical", medical_count), ("general", general_count)):
        pool = [dict(row) for row in rows if row.get("domain") == domain]
        pool.sort(
            key=lambda row: (
                hashlib.sha256(str(row.get("sample_id", "")).encode()).hexdigest(),
                str(row.get("sample_id", "")),
            )
        )
        if len(pool) < count or any(not row.get("sample_id") for row in pool[:count]):
            raise ValueError(f"insufficient stable label-free {domain} rows")
        selected.extend(pool[:count])
    return selected


def parse_choice_letter_v1(text: str) -> str | None:
    matches = [match.upper() for match in ANSWER_MARKER.findall(str(text))]
    if not matches:
        return None
    return matches[-1]


def legalize_parsed_choice_v1(
    parsed: str | None, *, option_count: int
) -> str | None:
    if option_count not in (4, 5):
        raise ValueError("secondary generation requires four or five options")
    legal = set("ABCDE"[:option_count])
    return parsed if parsed is not None and parsed in legal else None


def redact_generation_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    for row in rows:
        if row.get("parsed") is None:
            errors["parse_failure"] += 1
        if row.get("finish_reason") == "length":
            errors["truncated"] += 1
    return {
        "count": len(rows),
        "parsed_count": sum(row.get("parsed") is not None for row in rows),
        "correct_count": sum(row.get("correct") is True for row in rows),
        "error_types": dict(sorted(errors.items())),
    }


def select_cmb_isolation_subset(
    rows: Sequence[Mapping[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    """Select a stable CMB-train diagnostic without accepting supervision."""

    if count <= 0:
        raise ValueError("CMB isolation count must be positive")
    selected: list[dict[str, Any]] = []
    for row in rows:
        forbidden = FORBIDDEN_LABEL_FIELDS.intersection(row)
        if forbidden:
            raise ValueError("CMB isolation prompt contains supervision")
        if row.get("target_role") != "medical_opd_cmb":
            raise ValueError("CMB isolation accepts only medical_opd_cmb prompts")
        sample_id = str(row.get("sample_id") or "")
        content_hash = str(row.get("content_hash") or "")
        options = row.get("options")
        if not sample_id or len(content_hash) != 64 or not isinstance(options, list):
            raise ValueError("CMB isolation prompt identity differs")
        selected.append(dict(row))
    selected.sort(
        key=lambda row: (
            hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest(),
            str(row["sample_id"]),
        )
    )
    if len(selected) < count:
        raise ValueError("insufficient CMB isolation prompts")
    return selected[:count]


def _canonical_raw_identity(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(raw),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _raw_upstream_identity(raw: Mapping[str, Any], raw_identity: str) -> str:
    for key in ("id", "question_id", "index", "uid"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return raw_identity


def match_cmb_labels_by_stable_sample_id(
    selected_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Join CMB labels through the formal adapter's exact stable identity.

    This function is called only after both model routes have been released. It
    deliberately does not use question-text similarity or fuzzy matching.
    """

    if not selected_rows:
        raise ValueError("CMB isolation selected rows are empty")
    identity_fields = (
        "source",
        "source_revision",
        "source_license",
        "upstream_split",
        "raw_file_sha256",
    )
    identities = {
        tuple(str(row.get(field) or "") for field in identity_fields)
        for row in selected_rows
    }
    if len(identities) != 1:
        raise ValueError("CMB isolation selected source identity differs")
    source, revision, license_name, upstream_split, raw_file_sha256 = next(
        iter(identities)
    )
    if (
        not source
        or not revision
        or not license_name
        or upstream_split != "train"
        or len(raw_file_sha256) != 64
    ):
        raise ValueError("CMB isolation selected source identity is incomplete")
    context = AdapterContext(
        source_type="cmb",
        source=source,
        source_revision=revision,
        source_license=license_name,
        upstream_split=upstream_split,
        target_role="medical_opd_cmb",
        raw_file_sha256=raw_file_sha256,
    )
    wanted = {str(row["sample_id"]): str(row["content_hash"]) for row in selected_rows}
    if "" in wanted or len(wanted) != len(selected_rows):
        raise ValueError("CMB isolation selected stable sample IDs are not unique")
    candidates: dict[str, list[tuple[str, str]]] = {
        sample_id: [] for sample_id in wanted
    }
    for raw in raw_rows:
        raw_identity = _canonical_raw_identity(raw)
        subject_value = raw.get("exam_subject")
        subject = (
            str(subject_value).strip()
            if subject_value is not None and str(subject_value).strip()
            else None
        )
        sample_id = stable_sample_id_v2(
            source=source,
            source_revision=revision,
            upstream_split=upstream_split,
            upstream_id=_raw_upstream_identity(raw, raw_identity),
            subject=subject,
        )
        if sample_id not in wanted:
            continue
        adapted = adapt_source_row(raw, context).require_record()
        if adapted.sample_id != sample_id:
            raise ValueError("CMB isolation adapter stable identity differs")
        answer_idx = str(adapted.answer_idx or "").upper()
        if answer_idx not in "ABCDE"[: len(adapted.options)]:
            raise ValueError("CMB isolation raw label is invalid")
        candidates[sample_id].append((answer_idx, adapted.content_hash))
    result: dict[str, str | None] = {}
    content_hash_equal_count = 0
    missing_answer_idx_count = 0
    non_single_answer_idx_count = 0
    for sample_id, values in candidates.items():
        if len(values) != 1:
            raise ValueError("CMB isolation stable label join is not one-to-one")
        answer_idx, raw_content_hash = values[0]
        if not answer_idx:
            result[sample_id] = None
            missing_answer_idx_count += 1
        elif len(answer_idx) == 1 and answer_idx in set("ABCDE"):
            result[sample_id] = answer_idx
        else:
            result[sample_id] = None
            non_single_answer_idx_count += 1
        content_hash_equal_count += raw_content_hash == wanted[sample_id]
    match_count = sum(len(values) for values in candidates.values())
    audit = {
        "method": "stable_sample_id_v2_exact_adapter_identity",
        "selected_count": len(selected_rows),
        "matched_count": match_count,
        "unique_match_count": len(result),
        "missing_count": len(selected_rows) - len(result),
        "duplicate_match_count": match_count - len(result),
        "content_hash_equal_count": content_hash_equal_count,
        "stable_identity_only_count": len(result) - content_hash_equal_count,
        "exact_adapter_validation_count": len(result),
        "label_resolved_count": sum(label is not None for label in result.values()),
        "label_unresolved_count": sum(label is None for label in result.values()),
        "adapter_answer_idx_missing_count": missing_answer_idx_count,
        "adapter_answer_idx_non_single_count": non_single_answer_idx_count,
        "fuzzy_matching_used": False,
    }
    return result, audit


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile input cannot be empty")
    return ordered[int(round((len(ordered) - 1) * fraction))]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("diagnostic distribution is empty or non-finite")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.pstdev(numeric),
        "min": min(numeric),
        "p50": _quantile(numeric, 0.50),
        "p90": _quantile(numeric, 0.90),
        "p95": _quantile(numeric, 0.95),
        "max": max(numeric),
    }


def summarize_correct_answer_margins(
    route_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str | None],
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Summarize paired correct-token margins without retaining sample rows."""

    if set(route_predictions) != {"B0", "B1"} or bootstrap_samples <= 0:
        raise ValueError("CMB margin diagnostic requires B0/B1 and bootstrap samples")
    route_margins: dict[str, dict[str, float]] = {}
    route_summary: dict[str, dict[str, Any]] = {}
    route_exclusions: dict[str, set[str]] = {}
    for route in ("B0", "B1"):
        indexed: dict[str, float] = {}
        seen: set[str] = set()
        excluded: set[str] = set()
        correct = 0
        for row in route_predictions[route]:
            sample_id = str(row.get("sample_id") or "")
            scores = row.get("candidate_scores")
            if not sample_id:
                raise ValueError("CMB margin prediction sample ID is missing")
            if sample_id in seen:
                raise ValueError("CMB margin prediction sample ID is duplicated")
            seen.add(sample_id)
            if not isinstance(scores, Mapping):
                raise ValueError("CMB margin candidate scores are missing")
            if sample_id not in labels:
                raise ValueError("CMB margin prediction has no joined label identity")
            label = labels[sample_id]
            if label is None:
                excluded.add(sample_id)
                continue
            numeric = {str(key): float(value) for key, value in scores.items()}
            if any(not math.isfinite(value) for value in numeric.values()) or len(numeric) < 2:
                raise ValueError("CMB margin scores are invalid")
            if label not in numeric:
                raise ValueError("CMB margin label is not a legal candidate score key")
            margin = numeric[str(label)] - max(
                value for key, value in numeric.items() if key != label
            )
            indexed[sample_id] = margin
            correct += max(numeric, key=numeric.get) == label
        if seen != set(labels) or set(indexed).union(excluded) != set(labels):
            raise ValueError("CMB margin route sample set differs")
        if not indexed:
            raise ValueError("CMB margin has no single-answer rows")
        route_margins[route] = indexed
        route_exclusions[route] = excluded
        route_summary[route] = {
            "correct_count": correct,
            "accuracy": correct / len(indexed),
            "correct_answer_margin": _distribution(list(indexed.values())),
            "positive_margin_count": sum(value > 0.0 for value in indexed.values()),
        }
    if route_exclusions["B0"] != route_exclusions["B1"]:
        raise ValueError("CMB margin structural exclusion sets differ by route")
    ids = sorted(set(labels) - route_exclusions["B0"])
    if not ids:
        raise ValueError("CMB margin has no single-answer rows")
    delta = [route_margins["B1"][sample_id] - route_margins["B0"][sample_id] for sample_id in ids]
    generator = random.Random(42)
    boot = []
    for _ in range(bootstrap_samples):
        boot.append(statistics.fmean(delta[generator.randrange(len(delta))] for _ in delta))
    return {
        "input_count": len(labels),
        "count": len(ids),
        "structural_exclusions": {
            "ground_truth_answer_idx_unavailable_for_single_token_metric": len(
                route_exclusions["B0"]
            )
        },
        "B0": route_summary["B0"],
        "B1": route_summary["B1"],
        "paired": {
            "mean_margin_delta": statistics.fmean(delta),
            "median_margin_delta": statistics.median(delta),
            "improved": sum(value > 0.0 for value in delta),
            "regressed": sum(value < 0.0 for value in delta),
            "unchanged": sum(value == 0.0 for value in delta),
            "bootstrap_samples": bootstrap_samples,
            "mean_delta_95_ci": [_quantile(boot, 0.025), _quantile(boot, 0.975)],
        },
        "raw_predictions_persisted": False,
        "raw_labels_persisted": False,
    }


def summarize_secondary_generations(
    route_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Aggregate five-route generation diagnostics after the model is released."""

    result: dict[str, Any] = {}
    for route, rows in route_predictions.items():
        indexed = {str(row.get("sample_id") or ""): row for row in rows}
        if "" in indexed or len(indexed) != len(rows) or set(indexed) != set(labels):
            raise ValueError("secondary generation prediction/label identity differs")
        domains: dict[str, Any] = {}
        for domain in ("medical", "general"):
            subset = [row for row in rows if row.get("domain") == domain]
            if not subset:
                raise ValueError("secondary generation domain subset is empty")
            tokens = [int(row.get("generated_token_count") or 0) for row in subset]
            domains[domain] = {
                "count": len(subset),
                "parsed_count": sum(row.get("parsed") is not None for row in subset),
                "parse_failure_count": sum(row.get("parsed") is None for row in subset),
                "correct_count": sum(
                    row.get("parsed") == labels[str(row["sample_id"])] for row in subset
                ),
                "truncated_count": sum(row.get("finish_reason") == "length" for row in subset),
                "generated_tokens": _distribution(tokens),
            }
            domains[domain]["accuracy"] = domains[domain]["correct_count"] / len(subset)
        result[str(route)] = domains
    result["raw_prompts_persisted"] = False
    result["raw_responses_persisted"] = False
    result["raw_labels_persisted"] = False
    return result


__all__ = [
    "FORBIDDEN_LABEL_FIELDS",
    "match_cmb_labels_by_content_hash",
    "parse_choice_letter_v1",
    "redact_generation_diagnostics",
    "select_cmb_isolation_subset",
    "select_label_free_subset",
    "summarize_correct_answer_margins",
    "summarize_secondary_generations",
]
