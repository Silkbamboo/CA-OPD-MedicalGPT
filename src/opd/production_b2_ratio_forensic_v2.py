"""Evidence-first P5.1 historical ratio recomputation and forensic package."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence


class RatioForensicV2Error(RuntimeError):
    """Historical evidence is absent, contaminated, or internally inconsistent."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_sha_bound_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write, fsync, rename, fsync directory, reread, and verify payload SHA."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RatioForensicV2Error(f"forensic output already exists: {target}")
    body = dict(payload)
    body["_payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        reread = read_sha_bound_json(target)
        if reread != dict(payload):
            raise RatioForensicV2Error("forensic payload differs after reread")
        return {
            "path": str(target),
            "file_sha256": _sha256_file(target),
            "payload_sha256": body["_payload_sha256"],
            "size_bytes": target.stat().st_size,
            "verified_after_reread": True,
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_sha_bound_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RatioForensicV2Error(f"forensic JSON is invalid: {target}") from error
    if not isinstance(value, dict):
        raise RatioForensicV2Error("forensic JSON is not an object")
    expected = value.pop("_payload_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    if expected != actual:
        raise RatioForensicV2Error("forensic payload SHA differs")
    return value


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RatioForensicV2Error(f"historical JSON invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise RatioForensicV2Error(f"historical JSON is not an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RatioForensicV2Error(f"historical {label} is absent or non-finite")
    return float(value)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise RatioForensicV2Error("historical distribution is empty")
    median = statistics.median(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
    }


def _extract_record(path: Path, *, source: str, expected_step: int) -> dict[str, Any]:
    kernel = _load(path)
    if not (
        kernel.get("step_index") == expected_step - 1
        and kernel.get("from_version") == expected_step - 1
        and kernel.get("to_version") == expected_step
    ):
        raise RatioForensicV2Error(f"historical {source} step/version chain differs at {expected_step}")
    telemetry = kernel.get("reconstruction_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RatioForensicV2Error("historical reconstruction telemetry is absent")
    update = telemetry.get("optimizer_update")
    correction = telemetry.get("q_p_old")
    if not isinstance(update, Mapping) or not isinstance(correction, Mapping):
        raise RatioForensicV2Error("historical optimizer/correction telemetry is absent")
    pre = update.get("ppo_ratio_pre")
    post = update.get("ppo_ratio_post")
    raw_is = correction.get("raw_is")
    log_w = correction.get("log_w")
    prompts = correction.get("per_prompt_ess")
    if not all(isinstance(item, Mapping) for item in (pre, post, raw_is, log_w, prompts)):
        raise RatioForensicV2Error("historical ratio/ESS telemetry is absent")
    post_min = _finite(post.get("min", 1.0), "post ratio min")
    post_max = _finite(post.get("max"), "post ratio max")
    post_p99 = _finite(post.get("p99"), "post ratio p99")
    if min(post_min, post_max, post_p99) <= 0.0:
        raise RatioForensicV2Error("historical post ratio is non-positive")
    long_prompt_ess = [
        _finite(item.get("ess_fraction"), "per-prompt ESS")
        for item in prompts.values()
        if isinstance(item, Mapping) and int(item.get("token_count", 0)) >= 32
    ]
    if not long_prompt_ess:
        raise RatioForensicV2Error("historical long-prompt ESS is absent")
    return {
        "source": source,
        "optimizer_step": expected_step,
        "from_policy_version": expected_step - 1,
        "to_policy_version": expected_step,
        "input_adapter_sha256": kernel.get("input_authority_tensor_sha256"),
        "output_adapter_sha256": kernel.get("trainer_authority_tensor_sha256"),
        "response_tokens_persisted": kernel.get("response_tokens_persisted") is True,
        "pre_identity_max_abs": _finite(correction.get("current_pre_old_max_abs"), "pre identity"),
        "pre_ratio_max": _finite(pre.get("max"), "pre ratio max"),
        "gradient_norm_before_clip": _finite(update.get("gradient_norm_before_clip"), "preclip grad"),
        "relative_update_norm": _finite(update.get("relative_parameter_delta"), "relative update"),
        "post_ratio_max": post_max,
        "post_abs_log_max": max(abs(math.log(post_min)), abs(math.log(post_max))),
        "post_abs_log_p99_proxy": abs(math.log(post_p99)),
        "post_shift_clip_fraction": _finite(post.get("clip_fraction"), "post clip fraction"),
        "backend_weight_p99": _finite(raw_is.get("p99"), "backend p99"),
        "backend_abs_log_max": max(
            abs(_finite(log_w.get("min"), "backend log min")),
            abs(_finite(log_w.get("max"), "backend log max")),
        ),
        "backend_cap_fraction": _finite(correction.get("cap_fraction"), "backend cap fraction"),
        "pooled_ess": _finite(correction.get("ess_fraction"), "pooled ESS"),
        "long_prompt_ess_min": min(long_prompt_ess),
    }


def recompute_historical_ratio_distribution(*, p4_root: str | Path, p5_root: str | Path) -> dict[str, Any]:
    """Recompute 20 P4.8g + 24 accepted P5 steps; never fit step25."""

    roots = (("p4_8g", Path(p4_root), 20), ("p5_accepted", Path(p5_root), 24))
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for source, root, count in roots:
        if root.is_symlink() or not (root / "b2_steps").is_dir():
            raise RatioForensicV2Error(f"historical {source} root is absent")
        for step in range(1, count + 1):
            path = root / "b2_steps" / f"step_{step - 1:02d}_v{step - 1}_to_v{step}.json"
            records.append(_extract_record(path, source=source, expected_step=step))
        source_counts[source] = count
    metrics = {
        key: _distribution([float(record[key]) for record in records])
        for key in (
            "pre_identity_max_abs",
            "pre_ratio_max",
            "gradient_norm_before_clip",
            "relative_update_norm",
            "post_ratio_max",
            "post_abs_log_max",
            "post_abs_log_p99_proxy",
            "post_shift_clip_fraction",
            "backend_weight_p99",
            "backend_abs_log_max",
            "backend_cap_fraction",
            "pooled_ess",
            "long_prompt_ess_min",
        )
    }
    return {
        "schema_version": 2,
        "artifact_kind": "p5_1_historical_ratio_distribution_v2",
        "record_count": len(records),
        "source_counts": source_counts,
        "accepted_policy_chains": {"p4_8g": [0, 20], "p5": [0, 24]},
        "p5_step25_in_threshold_fit": False,
        "response_tokens_persisted": all(record["response_tokens_persisted"] for record in records),
        "metrics": metrics,
        "records": records,
    }


def _round_up(value: float, quantum: float) -> float:
    return math.ceil(value / quantum - 1.0e-12) * quantum


def derive_ratio_health_thresholds_v2(history: Mapping[str, Any]) -> dict[str, Any]:
    if not (
        history.get("record_count") == 44
        and history.get("source_counts") == {"p4_8g": 20, "p5_accepted": 24}
        and history.get("p5_step25_in_threshold_fit") is False
    ):
        raise RatioForensicV2Error("ratio thresholds require uncontaminated 20+24 history")
    metrics = history["metrics"]
    gradient = metrics["gradient_norm_before_clip"]
    median = float(gradient["median"])
    mad = float(gradient["mad"])
    robust_max = 0.0 if mad == 0.0 else (float(gradient["max"]) - median) / mad
    absolute_grad_cap = _round_up(max(100.0, 2.0 * float(gradient["max"])), 10.0)
    robust_z_cap = _round_up(max(300.0, 1.25 * robust_max), 10.0)
    relative_update_cap = _round_up(
        max(0.005, 1.5 * float(metrics["relative_update_norm"]["max"])), 0.001
    )
    backend_p999_cap = _round_up(
        max(0.8, 1.10 * float(metrics["backend_abs_log_max"]["max"])), 0.1
    )
    # Historical artifacts retain P99 and extrema but not the token vector
    # needed to reconstruct P99.9.  The healthy combined envelope nevertheless
    # contains an isolated abs-log shift of 5.938.  A 6.0 hard envelope avoids
    # silently recreating the old raw-max gate; P99 and tail loss/gradient
    # influence remain the sensitive hard checks.
    post_p999_cap = 6.0
    return {
        "schema_version": 2,
        "artifact_kind": "p5_1_ratio_health_thresholds_v2",
        "written_before_new_gpu_results": True,
        "threshold_fit_sources": {"p4_8g_steps": 20, "p5_accepted_steps": 24, "p5_step25": "excluded_validation_only"},
        "ppo_abs_log_p99_max": 1.0e-4,
        "ppo_abs_log_p999_max": 1.0e-4,
        "backend_abs_log_p99_max": 0.35,
        "backend_abs_log_p999_max": backend_p999_cap,
        "backend_clip_fraction_max": 0.05,
        "pooled_ess_floor": 0.95,
        "per_prompt_ess_floor": 0.95,
        "per_prompt_ess_min_tokens": 32,
        "approx_kl_abs_max": 0.05,
        "ppo_clip_fraction_max": 0.20,
        "preclip_grad_norm_absolute_max": absolute_grad_cap,
        "preclip_grad_robust_z_max": robust_z_cap,
        "healthy_grad_median": median,
        "healthy_grad_mad": mad,
        "relative_update_norm_max": relative_update_cap,
        "post_shift_abs_log_p99_max": 0.35,
        "post_shift_abs_log_p999_max": post_p999_cap,
        "post_shift_tail_abs_log_threshold": math.log(5.0),
        "tail_loss_share_max": 0.05,
        "tail_gradient_proxy_share_max": 0.05,
        "raw_post_ratio_max_warning_above": 5.0,
        "raw_post_ratio_max_is_hard_failure_alone": False,
        "consecutive_warning_abort_count": 2,
        "derivation": {
            "historical_metrics": metrics,
            "absolute_grad_cap": "ceil_10(max(100,2*healthy_max))",
            "robust_grad_cap": "ceil_10(max(300,1.25*healthy_max_robust_z))",
            "relative_update_cap": "ceil_0.001(max(0.005,1.5*healthy_max))",
            "post_p999_cap": "healthy_envelope_abs_log_6.0; historical token vectors unavailable; P99/tail influence remain hard",
            "tail_influence_caps": "preregistered_conservative_0.05; token vectors absent from history",
        },
    }


def build_p5_1_forensic_report(*, history: Mapping[str, Any], step25_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if history.get("p5_step25_in_threshold_fit") is not False:
        raise RatioForensicV2Error("step25 contaminated forensic threshold history")
    ratio = step25_evidence.get("ratio")
    if not isinstance(ratio, Mapping) or abs(_finite(ratio.get("max"), "step25 ratio max") - 11.611102) > 1.0e-3:
        raise RatioForensicV2Error("step25 ratio evidence differs")
    return {
        "schema_version": 2,
        "artifact_kind": "p5_1_ratio_forensic_report_v2",
        "historical_p5_status": "blocked_formal_b2_ratio_max",
        "historical_result_rewritten": False,
        "threshold_history": {
            "record_count": history["record_count"],
            "p5_step25_excluded": True,
        },
        "classification": {
            "ratio_11_6111": "post_update_policy_shift_ratio",
            "formula": "exp(log_q_post-log_p_old_canonical)",
            "computed_at": "after_optimizer_candidate",
            "pre_update_ppo_ratio": "identity_1.0_in_persisted_step25_telemetry",
            "same_as_backend_correction": False,
            "ess_uses_same_semantics": False,
            "ess_semantics": "clipped_backend_correction_weight_ess",
            "p5_stop_reason": "v1_post_update_raw_max_gate_after_nontransactional_update",
        },
        "step25_observed": dict(step25_evidence),
        "replay": {
            "exact_historical_token_replay_available": bool(history.get("response_tokens_persisted")),
            "required_action": (
                "exact_fixed_token_replay"
                if history.get("response_tokens_persisted")
                else "schedule_replay_from_step20_new_tokens_ignored_artifact"
            ),
            "step20_usage": "diagnostic_only_not_formal_initialization",
        },
        "unresolved_before_gpu": [
            "anomalous_token_loss_share",
            "anomalous_token_gradient_proxy_share",
            "anomalous_token_advantage_branch",
            "step21_24_replay_identity",
        ],
        "restricted_data_access": {"final": 0, "controller": 0, "confirmation": 0, "labels": 0},
    }
