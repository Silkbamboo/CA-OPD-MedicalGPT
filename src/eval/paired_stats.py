"""CPU-safe paired Controller v2 scoring, uncertainty and Teacher gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class PairedStatsError(RuntimeError):
    """Invalid label/prediction pairing or frozen artifact."""


def _index_unique(rows: Iterable[Mapping[str, Any]], *, kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in indexed:
            raise PairedStatsError(f"{kind} contains a missing/duplicate sample_id")
        if "final" in str(row.get("target_role") or ""):
            raise PairedStatsError(f"{kind} cannot contain a final role")
        indexed[sample_id] = dict(row)
    return indexed


def score_label_free_predictions(
    predictions: Iterable[Mapping[str, Any]], labels: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join physically separated files by ID; never modify the predictions."""

    pred = _index_unique(predictions, kind="predictions")
    gold = _index_unique(labels, kind="labels")
    if set(pred) != set(gold):
        raise PairedStatsError("prediction/label sample_id sets differ")
    scored = []
    for sample_id in sorted(pred):
        expected = str(gold[sample_id].get("answer_idx") or "").upper()
        predicted = str(pred[sample_id].get("predicted_label") or "").upper()
        if expected not in "ABCDE" or predicted not in "ABCDE":
            raise PairedStatsError("prediction/label is not a canonical option letter")
        row = dict(pred[sample_id])
        row["correct"] = predicted == expected
        scored.append(row)
    return scored


def _bootstrap_delta(
    pairs: Sequence[tuple[int, int]], *, seed: int, samples: int
) -> tuple[float, float]:
    if samples < 1 or not pairs:
        raise PairedStatsError("paired bootstrap requires non-empty pairs and samples")
    rng = random.Random(seed)
    count = len(pairs)
    estimates = []
    for _ in range(samples):
        delta = 0
        for _ in range(count):
            b0, b1 = pairs[rng.randrange(count)]
            delta += b1 - b0
        estimates.append(delta / count)
    estimates.sort()
    lower = estimates[int((samples - 1) * 0.025)]
    upper = estimates[int((samples - 1) * 0.975)]
    return lower, upper


def _mcnemar(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    improved = sum(b0 == 0 and b1 == 1 for b0, b1 in pairs)
    regressed = sum(b0 == 1 and b1 == 0 for b0, b1 in pairs)
    discordant = improved + regressed
    correction = max(0, abs(improved - regressed) - 1)
    statistic = (correction ** 2) / discordant if discordant else 0.0
    # Two-sided exact binomial form of McNemar, avoiding a scipy dependency.
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(improved, regressed) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "b0_wrong_b1_right": improved,
        "b0_right_b1_wrong": regressed,
        "discordant": discordant,
        "continuity_corrected_chi_square": statistic,
        "exact_two_sided_p": p_value,
    }


def _summarize_pairs(
    ids: Sequence[str], b0: Mapping[str, Mapping[str, Any]], b1: Mapping[str, Mapping[str, Any]],
    *, seed: int, bootstrap_samples: int,
) -> dict[str, Any]:
    pairs = [(int(bool(b0[item]["correct"])), int(bool(b1[item]["correct"]))) for item in ids]
    count = len(pairs)
    accuracy0 = sum(item[0] for item in pairs) / count
    accuracy1 = sum(item[1] for item in pairs) / count
    return {
        "count": count,
        "accuracy": {"B0": accuracy0, "B1": accuracy1},
        "paired_delta": accuracy1 - accuracy0,
        "bootstrap_95_ci": list(
            _bootstrap_delta(pairs, seed=seed, samples=bootstrap_samples)
        ),
        "mcnemar": _mcnemar(pairs),
        "improved": sum(a == 0 and b == 1 for a, b in pairs),
        "regressed": sum(a == 1 and b == 0 for a, b in pairs),
        "unchanged": sum(a == b for a, b in pairs),
    }


def paired_comparison(
    b0_rows: Iterable[Mapping[str, Any]],
    b1_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    b0 = _index_unique(b0_rows, kind="B0")
    b1 = _index_unique(b1_rows, kind="B1")
    if set(b0) != set(b1):
        raise PairedStatsError("B0/B1 sample sets differ")
    ids = sorted(b0)
    for sample_id in ids:
        if "correct" not in b0[sample_id] or "correct" not in b1[sample_id]:
            raise PairedStatsError("paired rows must be scored before comparison")
        for field in ("domain", "subject", "target_role"):
            if b0[sample_id].get(field) != b1[sample_id].get(field):
                raise PairedStatsError(f"B0/B1 {field} differs for {sample_id}")
    overall = _summarize_pairs(ids, b0, b1, seed=seed, bootstrap_samples=bootstrap_samples)
    report: dict[str, Any] = {
        **overall,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "same_sample_ids": True,
    }
    domains: dict[str, Any] = {}
    for domain in sorted({str(b0[item].get("domain") or "unknown") for item in ids}):
        members = [item for item in ids if str(b0[item].get("domain") or "unknown") == domain]
        domains[domain] = _summarize_pairs(
            members, b0, b1, seed=seed, bootstrap_samples=bootstrap_samples
        )
    subjects: dict[str, Any] = {}
    for subject in sorted(
        {str(b0[item].get("subject") or "unknown") for item in ids if b0[item].get("domain") == "general"}
    ):
        members = [item for item in ids if b0[item].get("domain") == "general" and str(b0[item].get("subject") or "unknown") == subject]
        subjects[subject] = _summarize_pairs(
            members, b0, b1, seed=seed, bootstrap_samples=bootstrap_samples
        )
    report["domains"] = domains
    report["subjects"] = subjects
    report["invalid_rate_delta"] = (
        sum(bool(b1[item].get("invalid")) for item in ids)
        - sum(bool(b0[item].get("invalid")) for item in ids)
    ) / len(ids)
    report["truncation_rate_delta"] = (
        sum(bool(b1[item].get("truncated")) for item in ids)
        - sum(bool(b0[item].get("truncated")) for item in ids)
    ) / len(ids)
    return report


def teacher_readiness(
    *,
    artifact_valid: bool,
    b0_medical_choice_accuracy: float,
    b1_medical_choice_accuracy: float,
    b1_generation_invalid_rate: float,
    b1_generation_truncation_rate: float,
) -> dict[str, Any]:
    delta = b1_medical_choice_accuracy - b0_medical_choice_accuracy
    tolerance = 1e-12
    if delta >= 0.03 - tolerance:
        knowledge: bool | str = True
    elif delta < -0.03 - tolerance:
        knowledge = False
    else:
        knowledge = "ambiguous"
    return {
        "teacher_artifact_valid": bool(artifact_valid),
        "teacher_knowledge_ready": knowledge,
        "teacher_generation_contract_ready": (
            b1_generation_invalid_rate <= 0.05
            and b1_generation_truncation_rate <= 0.01
        ),
        "medical_choice_delta": delta,
        "knowledge_threshold_pp": 3.0,
        "generation_invalid_rate_max": 0.05,
        "generation_truncation_rate_max": 0.01,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_teacher_artifact(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_adapter_sha256: str,
    expected_base_revision: str,
) -> bool:
    path = Path(manifest_path)
    if _sha256(path) != expected_manifest_sha256:
        raise PairedStatsError("Teacher artifact manifest SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    revision = payload.get("model_revision", payload.get("base_model_revision"))
    if revision != expected_base_revision:
        raise PairedStatsError("Teacher base revision mismatch")
    if payload.get("adapter_sha256") != expected_adapter_sha256:
        raise PairedStatsError("Teacher adapter SHA declaration mismatch")
    if payload.get("adapter_file"):
        adapter = path.parent / str(payload["adapter_file"])
        if not adapter.is_file() or _sha256(adapter) != expected_adapter_sha256:
            raise PairedStatsError("Teacher adapter SHA mismatch")
    else:
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise PairedStatsError("Teacher artifact file inventory is missing")
        for item in files:
            artifact = path.parent / str(item.get("path") or "")
            if not artifact.is_file() or _sha256(artifact) != str(item.get("sha256") or ""):
                raise PairedStatsError("Teacher adapter SHA mismatch")
    return True
