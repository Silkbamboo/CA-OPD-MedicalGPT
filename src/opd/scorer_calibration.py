"""Calibration gates and non-reversible trajectory manifest helpers."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CALIBRATION_ARTIFACT_FILES = (
    "config.yaml",
    "metadata.json",
    "trajectory_manifest.json",
    "route_manifest.json",
    "repeatability.json",
    "route_isolation.json",
    "vllm_diagnostic.json",
    "live_rollout_manifest.json",
    "same_model_null.json",
    "one_step_direction.json",
    "null_update.json",
    "sampler_refresh.json",
    "metrics.jsonl",
    "summary.json",
    "stdout.log",
    "cost.json",
)
CROSS_BACKEND_THRESHOLDS = {
    "median_abs_diff_max": 0.005,
    "p95_abs_diff_max": 0.02,
    "max_abs_diff_max": 0.05,
    "correlation_min": 0.99,
    "advantage_sign_agreement_min": 0.99,
    "advantage_sign_gap_floor": 0.05,
}


class CalibrationError(RuntimeError):
    pass


def classify_calibration_failure(phase: str) -> str:
    mapping = {
        "transformers_replay": "blocked_transformers_scorer",
        "route_isolation": "blocked_transformers_scorer",
        "live_rollout": "blocked_live_rollout_alignment",
        "same_model_null": "blocked_live_rollout_alignment",
        "one_step_direction": "blocked_pg_opd_direction",
        "null_update": "blocked_pg_opd_direction",
        "sampler_refresh": "blocked_sampler_refresh",
    }
    return mapping.get(str(phase), "blocked_artifact_contract")


def _stable_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_public_trajectory_manifest(
    fixtures: Sequence[Mapping[str, Any]], *, raw_fixture_sha256: str, seed: int
) -> dict[str, Any]:
    rows = []
    for fixture in sorted(fixtures, key=lambda item: str(item["fixture_id"])):
        rows.append(
            {
                "fixture_id": str(fixture["fixture_id"]),
                "source_role": str(fixture["source_role"]),
                "prompt_sha256": _sha_text(str(fixture["prompt"])),
                "response_sha256": _sha_text(str(fixture["response"])),
                "prompt_length": len(fixture["prompt_ids"]),
                "response_length": len(fixture["response_ids"]),
                "eos": bool(fixture["eos"]),
                "truncated": bool(fixture["truncated"]),
                "tokenizer_revision": str(fixture["tokenizer_revision"]),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "opd_scorer_calibration",
        "seed": int(seed),
        "count": len(rows),
        "contains_labels": False,
        "contains_raw_text": False,
        "contains_token_ids": False,
        "raw_fixture_sha256": raw_fixture_sha256,
        "fixtures": rows,
    }
    return {**payload, "manifest_sha256": hashlib.sha256(_stable_bytes(payload)).hexdigest()}


def validate_repeatability(
    runs: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-4
) -> dict[str, Any]:
    if len(runs) != 3:
        raise CalibrationError("repeatability requires exactly three runs")
    reference = runs[0]
    token_ids = list(reference.get("token_ids", []))
    mask = list(reference.get("response_mask", []))
    if not token_ids or len(token_ids) != len(mask):
        raise CalibrationError("repeatability token/mask mapping is empty or misaligned")
    scores = []
    for run in runs:
        if list(run.get("token_ids", [])) != token_ids or list(run.get("response_mask", [])) != mask:
            raise CalibrationError("repeatability token mapping changed")
        values = [float(value) for value in run.get("token_logprobs", [])]
        if len(values) != len(token_ids) or not all(math.isfinite(value) for value in values):
            raise CalibrationError("repeatability scores must be aligned and finite")
        scores.append(values)
    maximum = max(
        abs(scores[i][j] - scores[0][j])
        for i in range(1, len(scores)) for j in range(len(token_ids))
    )
    if maximum > tolerance:
        raise CalibrationError(
            f"repeatability max drift {maximum:.9g} exceeds frozen tolerance {tolerance:.9g}"
        )
    return {"passed": True, "runs": 3, "max_abs_delta": maximum, "tolerance": tolerance}


def validate_route_isolation(
    sequences: Mapping[str, Sequence[Mapping[str, Any]]], *,
    medical_adapter_sha256: str, tolerance: float = 1e-6,
) -> dict[str, Any]:
    expected_orders = {
        "base_medical_base": ("base", "medical", "base"),
        "medical_base_medical": ("medical", "base", "medical"),
    }
    maxima = []
    route_differences: list[float] = []
    for name, expected in expected_orders.items():
        observations = list(sequences.get(name, ()))
        if len(observations) != 3 or tuple(item.get("route") for item in observations) != expected:
            raise CalibrationError(f"route isolation sequence {name} does not match {expected}")
        for item in observations:
            route = str(item["route"])
            adapter_sha = item.get("adapter_sha")
            if route == "base" and adapter_sha is not None:
                raise CalibrationError("route isolation Base observation retained an adapter")
            if route == "medical" and adapter_sha != medical_adapter_sha256:
                raise CalibrationError("route isolation Medical adapter identity changed")
            token_ids = list(item.get("token_ids", ()))
            mask = list(item.get("response_mask", ()))
            scores = [float(value) for value in item.get("token_logprobs", ())]
            if not token_ids or len(token_ids) != len(mask) or len(token_ids) != len(scores):
                raise CalibrationError("route isolation token/logprob mapping is misaligned")
            if not all(math.isfinite(value) for value in scores):
                raise CalibrationError("route isolation logprobs must be finite")
        repeated = (observations[0], observations[2])
        if (
            list(repeated[0]["token_ids"]) != list(repeated[1]["token_ids"])
            or list(repeated[0]["response_mask"]) != list(repeated[1]["response_mask"])
        ):
            raise CalibrationError("route isolation repeated route changed token mapping")
        maximum = max(
            abs(float(left) - float(right))
            for left, right in zip(
                repeated[0]["token_logprobs"], repeated[1]["token_logprobs"], strict=True
            )
        )
        if maximum > tolerance:
            raise CalibrationError(
                f"route isolation drift {maximum:.9g} exceeds frozen tolerance {tolerance:.9g}"
            )
        maxima.append(maximum)
        base = next(item for item in observations if item["route"] == "base")
        medical = next(item for item in observations if item["route"] == "medical")
        if list(base["token_ids"]) != list(medical["token_ids"]):
            raise CalibrationError("route isolation compared different trajectories")
        route_differences.extend(
            abs(float(left) - float(right))
            for left, right in zip(
                base["token_logprobs"], medical["token_logprobs"], strict=True
            )
        )
    nonzero = sum(value > 0 for value in route_differences)
    if nonzero == 0:
        raise CalibrationError("route isolation Base and Medical routes are all identical")
    return {
        "passed": True,
        "orders": list(expected_orders),
        "tolerance": tolerance,
        "max_same_route_abs_delta": max(maxima),
        "base_adapter_disabled": True,
        "medical_adapter_sha256": medical_adapter_sha256,
        "nonzero_route_difference_fraction": nonzero / len(route_differences),
    }


def summarize_signed_update(
    *, advantage: Sequence[float], logprob_change: Sequence[float],
    near_zero_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if len(advantage) != len(logprob_change) or not advantage:
        raise CalibrationError("signed update vectors must be non-empty and aligned")
    pairs = [(float(a), float(delta)) for a, delta in zip(advantage, logprob_change, strict=True)]
    if not all(math.isfinite(value) for pair in pairs for value in pair):
        raise CalibrationError("signed update values must be finite")
    positive = [delta for value, delta in pairs if value > near_zero_tolerance]
    negative = [delta for value, delta in pairs if value < -near_zero_tolerance]
    near_zero = [delta for value, delta in pairs if abs(value) <= near_zero_tolerance]
    positive_mean = statistics.mean(positive) if positive else None
    negative_mean = statistics.mean(negative) if negative else None
    weighted = statistics.mean(value * delta for value, delta in pairs)
    passed = bool(
        positive and negative and positive_mean is not None and positive_mean > 0
        and negative_mean is not None and negative_mean < 0 and weighted > 0
    )
    return {
        "protocol_version": 2,
        "gate_role": "diagnostic_only",
        "legacy_field_semantics": "passed retains the P4.1 subgroup condition only",
        "passed": passed,
        "near_zero_tolerance": near_zero_tolerance,
        "positive_tokens": len(positive),
        "negative_tokens": len(negative),
        "near_zero_tokens": len(near_zero),
        "positive_advantage_logprob_change_mean": positive_mean,
        "negative_advantage_logprob_change_mean": negative_mean,
        "advantage_weighted_direction_mean": weighted,
    }


def validate_sampler_refresh(
    *, old_adapter_sha256: str, new_adapter_sha256: str,
    old_version: int, new_version: int, exported_files: Sequence[str],
    copied_full_base: bool, teacher_adapter_sha256: str,
    identity_check_finite: bool,
) -> dict[str, Any]:
    expected_files = {"adapter_config.json", "adapter_model.safetensors"}
    actual_files = {Path(name).name for name in exported_files}
    if old_adapter_sha256 == new_adapter_sha256:
        raise CalibrationError("sampler refresh reused the old adapter SHA")
    if new_version != old_version + 1:
        raise CalibrationError("sampler refresh adapter version did not increment exactly once")
    if actual_files != expected_files or copied_full_base:
        raise CalibrationError("sampler refresh must export LoRA-only files without a Base copy")
    if new_adapter_sha256 == teacher_adapter_sha256:
        raise CalibrationError("sampler refresh crossed Student and Teacher adapters")
    if not identity_check_finite:
        raise CalibrationError("sampler refresh identity logprob check was non-finite")
    return {
        "passed": True,
        "old_adapter_sha256": old_adapter_sha256,
        "new_adapter_sha256": new_adapter_sha256,
        "old_version": old_version,
        "new_version": new_version,
        "old_adapter_reused": False,
        "exported_files": sorted(actual_files),
        "copied_full_base": False,
        "teacher_student_adapter_crossline": False,
        "identity_check_finite": True,
    }


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise CalibrationError("correlation requires aligned vectors with at least two values")
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return numerator / denominator if denominator else (1.0 if list(left) == list(right) else 0.0)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + end - 1) / 2.0
        for index, _ in ordered[start:end]:
            result[index] = average
        start = end
    return result


def compare_backends(
    reference: Sequence[float], candidate: Sequence[float], *,
    reference_gaps: Sequence[float], candidate_gaps: Sequence[float] | None = None,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or len(reference) != len(reference_gaps) or not reference:
        raise CalibrationError("backend equivalence vectors are misaligned")
    if candidate_gaps is not None and len(candidate_gaps) != len(reference):
        raise CalibrationError("candidate advantage-gap vector is misaligned")
    if not all(
        math.isfinite(float(x))
        for x in (*reference, *candidate, *reference_gaps, *(candidate_gaps or ()))
    ):
        raise CalibrationError("backend equivalence values must be finite")
    diffs = [abs(float(a) - float(b)) for a, b in zip(reference, candidate, strict=True)]
    pearson = _pearson(reference, candidate)
    spearman = _pearson(_average_ranks(reference), _average_ranks(candidate))
    eligible = [i for i, gap in enumerate(reference_gaps) if abs(float(gap)) >= 0.05]
    sign_agreement = 1.0
    if eligible:
        agreement = 0
        for index in eligible:
            candidate_gap = (
                float(candidate_gaps[index])
                if candidate_gaps is not None
                else float(reference_gaps[index]) + float(candidate[index]) - float(reference[index])
            )
            agreement += (candidate_gap > 0) == (float(reference_gaps[index]) > 0)
        sign_agreement = agreement / len(eligible)
    metrics = {
        "median_abs_diff": statistics.median(diffs),
        "p95_abs_diff": _percentile(diffs, 0.95),
        "max_abs_diff": max(diffs),
        "pearson": pearson,
        "spearman": spearman,
        "advantage_sign_agreement": sign_agreement,
        "sign_positions": len(eligible),
    }
    passed = (
        metrics["median_abs_diff"] <= 0.005
        and metrics["p95_abs_diff"] <= 0.02
        and metrics["max_abs_diff"] <= 0.05
        and pearson >= 0.99
        and spearman >= 0.99
        and sign_agreement >= 0.99
    )
    return {"passed": passed, "metrics": metrics, "thresholds": dict(CROSS_BACKEND_THRESHOLDS)}


def validate_artifact_inventory(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    missing = [name for name in CALIBRATION_ARTIFACT_FILES if not (root / name).is_file()]
    if missing:
        raise CalibrationError(f"calibration artifact inventory is missing: {missing}")
    return {"complete": True, "files": list(CALIBRATION_ARTIFACT_FILES)}


def validate_artifact_size_budget(
    paths: Sequence[str | Path], *, maximum_bytes: int
) -> dict[str, Any]:
    if maximum_bytes <= 0:
        raise CalibrationError("artifact size budget must be positive")
    sizes = {str(Path(path)): Path(path).stat().st_size for path in paths}
    oversized = {path: size for path, size in sizes.items() if size > maximum_bytes}
    if oversized:
        raise CalibrationError(
            f"calibration artifact size budget {maximum_bytes} exceeded: {oversized}"
        )
    return {"passed": True, "maximum_bytes": maximum_bytes, "sizes": sizes}


def write_cpu_mock_calibration_artifacts(
    run_dir: str | Path,
    *,
    request_id: str,
    response_ids: Sequence[int],
    old_logprobs: Sequence[float],
    teacher_logprobs: Sequence[float],
    advantage: Sequence[float],
    loss: float,
) -> dict[str, Any]:
    """Write the standard inventory for a toy, explicitly non-formal CPU run.

    This helper exists only to exercise the entire contract and artifact boundary
    without importing a real model.  Its status fields prevent a mock artifact
    from being confused with the later authorized GPU calibration.
    """

    lengths = {len(response_ids), len(old_logprobs), len(teacher_logprobs), len(advantage)}
    if len(lengths) != 1 or not response_ids:
        raise CalibrationError("CPU mock trajectory vectors must be non-empty and aligned")
    numeric = [*old_logprobs, *teacher_logprobs, *advantage, loss]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise CalibrationError("CPU mock trajectory values must be finite")
    root = Path(run_dir)
    if root.exists():
        raise CalibrationError("CPU mock artifact output already exists")
    root.mkdir(parents=True)
    payloads: dict[str, Any] = {
        "config.yaml": {
            "stage": "opd_scorer_calibration",
            "calibration_only": True,
            "cpu_mock_only": True,
            "formal_opd_training": False,
        },
        "metadata.json": {
            "run_id": "cpu-mock",
            "stage": "opd_scorer_calibration",
            "cpu_contract_verified": True,
            "gpu_runtime_verified": False,
            "formal_opd_authorized": False,
        },
        "trajectory_manifest.json": {
            "count": 1,
            "contains_labels": False,
            "contains_raw_text": False,
            "contains_token_ids": False,
        },
        "route_manifest.json": {
            "routes": ["base", "medical"],
            "teacher_generates": False,
            "shared_backbone_required": True,
        },
        "repeatability.json": {"cpu_mock_only": True, "passed": True},
        "route_isolation.json": {"cpu_mock_only": True, "passed": True},
        "vllm_diagnostic.json": {"backend": "mock", "formal_enabled": False},
        "live_rollout_manifest.json": {"cpu_mock_only": True, "count": 1},
        "same_model_null.json": {"cpu_mock_only": True, "advantage": 0.0},
        "one_step_direction.json": {
            "request_id": request_id,
            "response_length": len(response_ids),
            "loss": float(loss),
            "formal_checkpoint_saved": False,
        },
        "null_update.json": {"cpu_mock_only": True, "update_l2": 0.0},
        "sampler_refresh.json": {"cpu_mock_only": True, "passed": True},
        "summary.json": {
            "status": "cpu_mock_pass",
            "cpu_mock_only": True,
            "gpu_runtime_verified": False,
            "formal_opd_authorized": False,
        },
        "cost.json": {"gpu_seconds": 0, "estimated_cost_cny": 0.0, "actual_cost_cny": None},
    }
    for name, value in payloads.items():
        (root / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
    (root / "metrics.jsonl").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "response_length": len(response_ids),
                "old_logprob_mean": statistics.mean(old_logprobs),
                "teacher_logprob_mean": statistics.mean(teacher_logprobs),
                "advantage_mean": statistics.mean(advantage),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "stdout.log").write_text(
        "CPU toy contract only; no model, GPU, or formal OPD execution.\n", encoding="utf-8"
    )
    return validate_artifact_inventory(root)
