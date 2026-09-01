"""CPU-safe P4.4 sampler-refresh contract and evidence writer.

This module deliberately imports no model or GPU framework.  GPU callers reduce
their selected-token probes and adapter tensors to the small, privacy-safe values
accepted here, then persist the report before asking the contract to raise.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SAMPLER_REFRESH_MAX_GAP = 1e-4
SAMPLER_REFRESH_ARTIFACT_PROTOCOL = "p4.4-sampler-refresh-contract-v4"


class SamplerRefreshContractError(RuntimeError):
    """Base error for malformed P4.4 sampler evidence."""


class SamplerRefreshGateError(SamplerRefreshContractError):
    """Raised only after a failed sampler report has been persisted."""


class StaleSamplerRequestError(SamplerRefreshContractError):
    """A request targeted a sampler identity other than the active identity."""


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_ADAPTER_NAME = re.compile(
    r"(\.lora_(?:A|B|embedding_A|embedding_B))\.[^.]+(?=(?:\.weight)?$)"
)


def canonical_adapter_tensor_name(name: str) -> str:
    """Remove the runtime PEFT adapter label from a LoRA parameter name."""

    if not isinstance(name, str) or "lora_" not in name:
        raise SamplerRefreshContractError("ordered tensor entry is not a LoRA tensor")
    return _ADAPTER_NAME.sub(r"\1", name)


def ordered_tensor_sha256(
    entries: Mapping[str, tuple[Sequence[int], bytes]],
) -> str:
    """Hash canonical LoRA tensor identity independently of safetensors bytes.

    GPU code supplies each tensor as canonical-name -> (shape, contiguous FP32
    little-endian bytes).  Adapter labels such as ``default`` and ``version1`` are
    removed before sorting, so equivalent loaded adapters can be compared across
    PEFT instances.  This digest is intentionally not a saved-file SHA.
    """

    canonical: dict[str, tuple[tuple[int, ...], bytes]] = {}
    for name, value in entries.items():
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[1], bytes)
        ):
            raise SamplerRefreshContractError("invalid ordered tensor entry")
        canonical_name = canonical_adapter_tensor_name(name)
        shape = tuple(int(item) for item in value[0])
        if not shape or any(item <= 0 for item in shape):
            raise SamplerRefreshContractError("invalid ordered tensor shape")
        if len(value[1]) != 4 * math.prod(shape):
            raise SamplerRefreshContractError(
                "ordered tensor bytes do not match normalized FP32 shape"
            )
        if canonical_name in canonical:
            raise SamplerRefreshContractError("duplicate canonical LoRA tensor name")
        canonical[canonical_name] = (shape, value[1])
    if not canonical:
        raise SamplerRefreshContractError("ordered tensor identity is empty")
    digest = hashlib.sha256()
    for name in sorted(canonical):
        shape, data = canonical[name]
        header = json.dumps(
            {"name": name, "shape": list(shape), "normalized_dtype": "float32_le"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def assert_sampler_request_identity(
    *,
    active_version: int,
    active_ordered_tensor_sha: str,
    active_run_token: str,
    requested_version: int,
    requested_ordered_tensor_sha: str,
    requested_run_token: str,
) -> None:
    """Reject stale/ambiguous routing before a sampler scoring call."""

    if not (
        requested_version == active_version
        and requested_ordered_tensor_sha == active_ordered_tensor_sha
        and requested_run_token == active_run_token
        and _valid_sha256(active_ordered_tensor_sha)
        and isinstance(active_run_token, str)
        and bool(active_run_token)
    ):
        raise StaleSamplerRequestError("stale sampler version/SHA/run token rejected")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SamplerRefreshContractError("cannot summarize an empty finite probe")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


_PAIR_KEYS = {"sample_id", "token_position", "token_id", "left", "right"}
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SCORER_KEYS = {
    "path",
    "backend",
    "dtype",
    "device",
    "mode",
    "attention_backend",
    "use_cache",
    "batch_size",
    "attention_mask",
    "position_ids",
    "log_softmax_dtype",
    "eos_token_ids",
    "pad_token_id",
    "generation_processors_warpers",
}


def _summarize_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    pairs = probe.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise SamplerRefreshContractError("sampler probe pairs are missing")
    finite: list[tuple[float, Mapping[str, Any]]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != _PAIR_KEYS:
            raise SamplerRefreshContractError(
                "probe token must contain only sample_id/position/token_id/numeric pair"
            )
        if not (
            isinstance(pair["sample_id"], str)
            and bool(_SAFE_SAMPLE_ID.fullmatch(pair["sample_id"]))
            and isinstance(pair["token_position"], int)
            and not isinstance(pair["token_position"], bool)
            and pair["token_position"] >= 0
            and isinstance(pair["token_id"], int)
            and not isinstance(pair["token_id"], bool)
            and pair["token_id"] >= 0
        ):
            raise SamplerRefreshContractError("probe token identity is invalid")
        try:
            left = float(pair["left"])
            right = float(pair["right"])
        except (TypeError, ValueError) as exc:
            raise SamplerRefreshContractError("probe logprob is not numeric") from exc
        if math.isfinite(left) and math.isfinite(right):
            finite.append((abs(left - right), pair))
    gaps = [item[0] for item in finite]
    worst = max(finite, key=lambda item: item[0]) if finite else None
    return {
        "count": len(pairs),
        "finite_count": len(finite),
        "finite_rate": len(finite) / len(pairs),
        "mae": sum(gaps) / len(gaps) if gaps else None,
        "p50": _quantile(gaps, 0.50) if gaps else None,
        "p95": _quantile(gaps, 0.95) if gaps else None,
        "p99": _quantile(gaps, 0.99) if gaps else None,
        "max": max(gaps) if gaps else None,
        "worst_token": (
            {
                "sample_id": str(worst[1]["sample_id"]),
                "token_position": int(worst[1]["token_position"]),
                "token_id": int(worst[1]["token_id"]),
                "left": float(worst[1]["left"]),
                "right": float(worst[1]["right"]),
                "abs_gap": float(worst[0]),
            }
            if worst
            else None
        ),
    }


_PROBE_SPECS = {
    "trainer_in_memory_vs_reloaded": "formal_same_path_gate",
    "live_refreshed_vs_fresh_sampler": "formal_same_path_gate",
    "repeated_same_instance_noise": "control_diagnostic",
    "no_op_refresh_gap": "control_diagnostic",
    "generation_raw_vs_direct": "cross_path_diagnostic",
    "processed_generation_vs_raw": "cross_path_diagnostic",
    "direct_cache_vs_no_cache": "cross_path_diagnostic",
    "trainer_direct_vs_sampler_generation": "cross_path_diagnostic",
}

_PROBE_SCORER_CONTRACT = {
    "trainer_in_memory_vs_reloaded": (
        ("direct_forward_raw_logits", "cuda:0", False),
        ("direct_forward_raw_logits", "cuda:0", False),
    ),
    "live_refreshed_vs_fresh_sampler": (
        ("direct_forward_raw_logits", "cuda:1", False),
        ("direct_forward_raw_logits", "cuda:1", False),
    ),
    "repeated_same_instance_noise": (
        ("direct_forward_raw_logits", "cuda:1", False),
        ("direct_forward_raw_logits", "cuda:1", False),
    ),
    "no_op_refresh_gap": (
        ("direct_forward_raw_logits", "cuda:1", False),
        ("direct_forward_raw_logits", "cuda:1", False),
    ),
    "generation_raw_vs_direct": (
        ("generation_raw_logits", "cuda:1", True),
        ("direct_forward_raw_logits", "cuda:1", False),
    ),
    "processed_generation_vs_raw": (
        ("generation_processed_scores", "cuda:1", True),
        ("generation_raw_logits", "cuda:1", True),
    ),
    "direct_cache_vs_no_cache": (
        ("direct_forward_raw_logits", "cuda:1", True),
        ("direct_forward_raw_logits", "cuda:1", False),
    ),
    "trainer_direct_vs_sampler_generation": (
        ("direct_forward_raw_logits", "cuda:0", False),
        ("generation_raw_logits", "cuda:1", True),
    ),
}


def _validated_probe_report(
    *,
    name: str,
    probe: Mapping[str, Any],
    classification: str,
    scorer_contract: tuple[tuple[str, str, bool], tuple[str, str, bool]],
) -> dict[str, Any]:
    if probe.get("name") != name or probe.get("classification") != classification:
        raise SamplerRefreshContractError(f"{name} probe classification drift")
    left = probe.get("left_scorer")
    right = probe.get("right_scorer")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise SamplerRefreshContractError(f"{name} scorer provenance is missing")
    if set(left) != _SCORER_KEYS or set(right) != _SCORER_KEYS:
        raise SamplerRefreshContractError(
            f"{name} scorer provenance has unsafe or missing fields"
        )
    for scorer in (left, right):
        if not (
            scorer.get("path")
            in {
                "direct_forward_raw_logits",
                "generation_raw_logits",
                "generation_processed_scores",
            }
            and isinstance(scorer.get("backend"), str)
            and scorer["backend"].startswith("transformers-")
            and scorer.get("dtype") == "bfloat16"
            and scorer.get("device") in {"cuda:0", "cuda:1"}
            and scorer.get("mode") == "eval"
            and scorer.get("attention_backend") == "eager"
            and isinstance(scorer.get("use_cache"), bool)
            and scorer.get("batch_size") == 1
            and scorer.get("attention_mask") == "all_ones_no_padding"
            and scorer.get("position_ids") == "implicit_from_attention_mask"
            and scorer.get("log_softmax_dtype") == "float32"
            and isinstance(scorer.get("eos_token_ids"), list)
            and bool(scorer["eos_token_ids"])
            and all(isinstance(item, int) for item in scorer["eos_token_ids"])
            and isinstance(scorer.get("pad_token_id"), int)
            and scorer.get("generation_processors_warpers") == []
        ):
            raise SamplerRefreshContractError(
                f"{name} scorer provenance has invalid execution semantics"
            )
    observed_contract = tuple(
        (scorer.get("path"), scorer.get("device"), scorer.get("use_cache"))
        for scorer in (left, right)
    )
    if observed_contract != scorer_contract:
        raise SamplerRefreshContractError(
            f"{name} scorer provenance differs from the frozen path/device/cache contract"
        )
    summary = _summarize_probe(probe)
    return {
        **summary,
        "classification": classification,
        "same_scoring_path": dict(left) == dict(right),
        "left_scorer": dict(left),
        "right_scorer": dict(right),
    }


def _probe_reports(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    probes = observation.get("probes")
    if not isinstance(probes, Mapping) or set(probes) != set(_PROBE_SPECS):
        raise SamplerRefreshContractError("sampler probe set is incomplete")
    reports: dict[str, dict[str, Any]] = {}
    for name, classification in _PROBE_SPECS.items():
        probe = probes[name]
        if not isinstance(probe, Mapping):
            raise SamplerRefreshContractError(f"{name} probe is invalid")
        reports[name] = _validated_probe_report(
            name=name,
            probe=probe,
            classification=classification,
            scorer_contract=_PROBE_SCORER_CONTRACT[name],
        )
    return reports


def _no_op_control_report(
    observation: Mapping[str, Any],
    *,
    run_id: str,
    validated_probes: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the v0 no-op reload identity, guard and fresh-reference evidence."""

    failures: list[str] = []
    control = observation.get("no_op_refresh_control")
    if not isinstance(control, Mapping):
        return None, ["v0 no-op refresh control evidence is missing"]
    fresh_probe = control.get("fresh_reload_probe")
    fresh_summary: dict[str, Any] | None = None
    try:
        if not isinstance(fresh_probe, Mapping):
            raise SamplerRefreshContractError(
                "v0 no-op fresh reload probe is missing"
            )
        fresh_summary = _validated_probe_report(
            name="no_op_fresh_reload_gap",
            probe=fresh_probe,
            classification="control_gate",
            scorer_contract=(
                ("direct_forward_raw_logits", "cuda:1", False),
                ("direct_forward_raw_logits", "cuda:1", False),
            ),
        )
        if fresh_summary["left_scorer"] != dict(
            validated_probes["no_op_refresh_gap"]["left_scorer"]
        ):
            raise SamplerRefreshContractError(
                "v0 no-op fresh reload scorer path differs"
            )
    except SamplerRefreshContractError as error:
        failures.append(str(error))

    before_sha = control.get("ordered_tensor_sha_before")
    after_sha = control.get("ordered_tensor_sha_after")
    fresh_sha = control.get("fresh_ordered_tensor_sha")
    before_file_sha = control.get("saved_adapter_sha_before")
    after_file_sha = control.get("saved_adapter_sha_after")
    fresh_file_sha = control.get("fresh_saved_adapter_sha")
    identity_gate = bool(
        control.get("version_before") == 0
        and control.get("version_after") == 0
        and control.get("fresh_version") == 0
        and _valid_sha256(before_sha)
        and before_sha == after_sha == fresh_sha
        and _valid_sha256(before_file_sha)
        and before_file_sha == after_file_sha == fresh_file_sha
        and control.get("active_adapter_before") == "version0"
        and control.get("active_adapter_after") == "version0_noop"
        and control.get("fresh_active_adapter") == "version0_fresh"
        and control.get("old_adapter_removed") is True
        and control.get("new_adapter_loaded") is True
    )
    if not identity_gate:
        failures.append("v0 no-op version/file/tensor/active-adapter identity mismatch")

    normal = control.get("normal_request")
    stale = control.get("stale_request")
    guard_gate = bool(
        isinstance(normal, Mapping)
        and normal.get("requested_version") == 0
        and normal.get("requested_ordered_tensor_sha") == before_sha
        and normal.get("requested_run_token") == f"{run_id}:adapter-v0"
        and normal.get("accepted") is True
        and normal.get("scoring_executed") is True
        and isinstance(stale, Mapping)
        and stale.get("requested_version") == 0
        and stale.get("requested_ordered_tensor_sha") != before_sha
        and stale.get("requested_run_token") == f"{run_id}:adapter-v0"
        and stale.get("rejected") is True
        and stale.get("scoring_executed") is False
        and stale.get("rejection_phase") == "identity_guard_before_scoring"
        and stale.get("error_type") == "StaleSamplerRequestError"
    )
    if not guard_gate:
        failures.append("v0 no-op normal/stale identity guard evidence mismatch")

    latency = control.get("refresh_latency_seconds")
    latency_gate = bool(
        isinstance(control.get("refresh_start"), str)
        and isinstance(control.get("refresh_end"), str)
        and isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) >= 0
    )
    if not latency_gate:
        failures.append("v0 no-op refresh latency evidence is invalid")

    fresh_gap_gate = bool(
        fresh_summary is not None
        and fresh_summary["finite_rate"] == 1.0
        and fresh_summary["max"] is not None
        and fresh_summary["max"] <= threshold
    )
    if not fresh_gap_gate:
        failures.append("v0 no-op fresh reload max gap exceeds 1e-4")

    public = {
        key: value
        for key, value in control.items()
        if key != "fresh_reload_probe"
    }
    public.update(
        {
            "identity_gate_passed": identity_gate,
            "request_guard_gate_passed": guard_gate,
            "latency_gate_passed": latency_gate,
            "fresh_reload_gap": fresh_summary,
            "fresh_reload_gap_gate_passed": fresh_gap_gate,
        }
    )
    return public, failures


def evaluate_sampler_v0_controls(
    *,
    run_id: str,
    repeated_probe: Mapping[str, Any],
    no_op_probe: Mapping[str, Any],
    no_op_refresh_control: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the preregistered v0 controls before any optimizer update."""

    if threshold != SAMPLER_REFRESH_MAX_GAP:
        raise SamplerRefreshContractError("v0 control threshold differs from 1e-4")
    repeat = _validated_probe_report(
        name="repeated_same_instance_noise",
        probe=repeated_probe,
        classification="control_diagnostic",
        scorer_contract=_PROBE_SCORER_CONTRACT["repeated_same_instance_noise"],
    )
    no_op = _validated_probe_report(
        name="no_op_refresh_gap",
        probe=no_op_probe,
        classification="control_diagnostic",
        scorer_contract=_PROBE_SCORER_CONTRACT["no_op_refresh_gap"],
    )
    public_control, failures = _no_op_control_report(
        {"no_op_refresh_control": no_op_refresh_control},
        run_id=run_id,
        validated_probes={"no_op_refresh_gap": no_op},
        threshold=threshold,
    )
    repeat_passed = bool(
        repeat["same_scoring_path"]
        and repeat["finite_rate"] == 1.0
        and repeat["max"] is not None
        and repeat["max"] <= threshold
    )
    no_op_passed = bool(
        no_op["same_scoring_path"]
        and no_op["finite_rate"] == 1.0
        and no_op["max"] is not None
        and no_op["max"] <= threshold
    )
    if not repeat_passed:
        failures.insert(0, "repeated_same_instance_noise max gap exceeds 1e-4")
    if not no_op_passed:
        failures.append("no_op_refresh_gap max gap exceeds 1e-4")
    failure_status = None
    if not repeat_passed:
        failure_status = "failed_same_instance_repeat"
    elif not no_op_passed or failures:
        failure_status = "failed_no_op_refresh"
    return {
        "schema_version": 4,
        "run_id": run_id,
        "stage": "sampler_v0_controls",
        "status": "pass" if failure_status is None else "fail",
        "hard_gate_passed": failure_status is None,
        "failure_status": failure_status,
        "failure_reasons": failures,
        "threshold": threshold,
        "repeated_same_instance_noise": repeat,
        "no_op_refresh_gap": no_op,
        "no_op_refresh_control": public_control,
    }


def build_sampler_refresh_report(
    observation: Mapping[str, Any], *, threshold: float, threshold_source: str
) -> dict[str, Any]:
    """Return complete sampler evidence without raising on an observed gate fail."""

    if threshold != SAMPLER_REFRESH_MAX_GAP:
        raise SamplerRefreshContractError("sampler threshold differs from frozen 1e-4")
    if not isinstance(threshold_source, str) or not threshold_source:
        raise SamplerRefreshContractError("sampler threshold source is missing")
    run_id = observation.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SamplerRefreshContractError("sampler run_id is missing")
    if observation.get("stage") != "sampler_refresh":
        raise SamplerRefreshContractError("sampler stage is invalid")
    probes = _probe_reports(observation)
    failures: list[str] = []
    no_op_control, no_op_failures = _no_op_control_report(
        observation,
        run_id=run_id,
        validated_probes=probes,
        threshold=threshold,
    )
    failures.extend(no_op_failures)

    required_shas = (
        "trainer_ordered_tensor_sha_before",
        "trainer_ordered_tensor_sha_after",
        "trainer_saved_adapter_sha",
        "trainer_reloaded_ordered_tensor_sha",
        "trainer_reloaded_adapter_sha",
        "sampler_ordered_tensor_sha_before",
        "sampler_ordered_tensor_sha_after",
        "fresh_sampler_ordered_tensor_sha",
        "sampler_loaded_adapter_sha",
    )
    if any(not _valid_sha256(observation.get(field)) for field in required_shas):
        failures.append("adapter ordered tensor SHA or saved adapter SHA is missing")

    trainer_before = observation.get("trainer_version_before")
    trainer_after = observation.get("trainer_version_after")
    sampler_before = observation.get("sampler_version_before")
    sampler_after = observation.get("sampler_version_after")
    if not (
        type(trainer_before) is int
        and type(trainer_after) is int
        and type(sampler_before) is int
        and type(sampler_after) is int
        and trainer_before == sampler_before == 0
        and trainer_after == sampler_after == 1
    ):
        failures.append("trainer/sampler adapter version did not advance together")

    if not (
        observation.get("trainer_ordered_tensor_sha_before")
        == observation.get("sampler_ordered_tensor_sha_before")
        and observation.get("trainer_ordered_tensor_sha_after")
        == observation.get("trainer_reloaded_ordered_tensor_sha")
        == observation.get("sampler_ordered_tensor_sha_after")
        == observation.get("fresh_sampler_ordered_tensor_sha")
        and observation.get("trainer_ordered_tensor_sha_after")
        != observation.get("trainer_ordered_tensor_sha_before")
    ):
        failures.append("trainer/sampler ordered tensor SHA identity mismatch")
    if not (
        observation.get("trainer_saved_adapter_sha")
        == observation.get("trainer_reloaded_adapter_sha")
        == observation.get("sampler_loaded_adapter_sha")
    ):
        failures.append("trainer/sampler saved adapter SHA identity mismatch")
    if not (
        observation.get("new_adapter_name") == "version1"
        and observation.get("active_adapter_name") == "version1"
        and observation.get("old_adapter_name") in {"version0", "version0_noop"}
    ):
        failures.append("active adapter is not the declared v1 adapter")
    if observation.get("old_adapter_removed") is not True:
        failures.append("old adapter was not removed")
    if observation.get("new_adapter_loaded") is not True:
        failures.append("new adapter was not loaded")
    identity_values = [
        observation.get(field)
        for field in (
            "base_revision",
            "trainer_base_revision",
            "sampler_base_revision",
            "tokenizer_revision",
            "trainer_tokenizer_revision",
            "sampler_tokenizer_revision",
        )
    ]
    if not (
        all(isinstance(value, str) and bool(value) for value in identity_values)
        and observation.get("base_revision")
        == observation.get("trainer_base_revision")
        == observation.get("sampler_base_revision")
        and observation.get("tokenizer_revision")
        == observation.get("trainer_tokenizer_revision")
        == observation.get("sampler_tokenizer_revision")
    ):
        failures.append("Base/tokenizer revision mismatch")

    isolation = observation.get("isolation")
    expected_isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if isolation != expected_isolation:
        failures.append("forbidden evaluation or label access")

    same_path_gates: dict[str, bool] = {}
    for name in ("trainer_in_memory_vs_reloaded", "live_refreshed_vs_fresh_sampler"):
        summary = probes[name]
        passed = bool(
            summary["same_scoring_path"]
            and summary["finite_rate"] == 1.0
            and summary["max"] is not None
            and summary["max"] <= threshold
        )
        same_path_gates[name] = passed
        if not passed:
            failures.append(f"{name} same-path max gap exceeds 1e-4 or path differs")

    control_gates: dict[str, bool] = {}
    for name in ("repeated_same_instance_noise", "no_op_refresh_gap"):
        summary = probes[name]
        passed = bool(
            summary["same_scoring_path"]
            and summary["finite_rate"] == 1.0
            and summary["max"] is not None
            and summary["max"] <= threshold
        )
        control_gates[name] = passed
        if not passed:
            failures.append(f"{name} max gap exceeds 1e-4 or path differs")

    if any(summary["finite_rate"] != 1.0 for summary in probes.values()):
        failures.append("one or more sampler diagnostics are nonfinite")

    stale = observation.get("stale_request")
    if not (
        isinstance(stale, Mapping)
        and stale.get("requested_version") == sampler_before
        and stale.get("requested_ordered_tensor_sha")
        == observation.get("sampler_ordered_tensor_sha_before")
        and stale.get("requested_run_token") == f"{run_id}:adapter-v0"
        and stale.get("active_version") == sampler_after
        and stale.get("active_ordered_tensor_sha")
        == observation.get("sampler_ordered_tensor_sha_after")
        and stale.get("active_run_token") == f"{run_id}:adapter-v1"
        and stale.get("rejected") is True
        and stale.get("silent_fallback") is False
        and stale.get("scoring_executed") is False
        and stale.get("routable_adapter_names_after_refresh")
        == [observation.get("new_adapter_name")]
        and stale.get("error_type") == "StaleSamplerRequestError"
        and stale.get("error_code") == "STALE_SAMPLER_IDENTITY"
        and stale.get("rejection_phase") == "identity_guard_before_scoring"
        and isinstance(stale.get("latency_seconds"), (int, float))
        and not isinstance(stale.get("latency_seconds"), bool)
        and math.isfinite(float(stale["latency_seconds"]))
        and float(stale["latency_seconds"]) >= 0
    ):
        failures.append(
            "stale request was not rejected before scoring or stale route remained"
        )
    active_request = observation.get("active_request")
    if not (
        isinstance(active_request, Mapping)
        and active_request.get("requested_version") == 1
        and active_request.get("requested_run_token") == f"{run_id}:adapter-v1"
        and active_request.get("requested_ordered_tensor_sha")
        == observation.get("sampler_ordered_tensor_sha_after")
        and active_request.get("accepted") is True
        and active_request.get("scoring_executed") is True
        and type(active_request.get("guarded_call_count")) is int
        and active_request.get("guarded_call_count", 0) >= 4
        and active_request.get("guarded_request_types")
        == ["fixed_action", "generation", "direct_no_cache", "direct_cache"]
        and type(active_request.get("result_token_count")) is int
        and active_request.get("result_token_count", 0) > 0
        and active_request.get("result_all_finite") is True
    ):
        failures.append("active v1 sampler requests did not use the identity guard")

    latency = observation.get("refresh_latency_seconds")
    if not (
        isinstance(observation.get("refresh_start"), str)
        and isinstance(observation.get("refresh_end"), str)
        and isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) >= 0
    ):
        failures.append("refresh start/end/latency evidence is invalid")

    gate_result = "pass" if not failures else "fail"
    report = {
        "schema_version": 4,
        "artifact_protocol_version": SAMPLER_REFRESH_ARTIFACT_PROTOCOL,
        "run_id": run_id,
        "stage": "sampler_refresh",
        "status": gate_result,
        "gate_result": gate_result,
        "hard_gate_passed": not failures,
        "trainer_version_before": trainer_before,
        "trainer_version_after": trainer_after,
        "sampler_version_before": sampler_before,
        "sampler_version_after": sampler_after,
        "trainer_ordered_tensor_sha_before": observation.get("trainer_ordered_tensor_sha_before"),
        "trainer_ordered_tensor_sha_after": observation.get("trainer_ordered_tensor_sha_after"),
        "trainer_saved_adapter_sha": observation.get("trainer_saved_adapter_sha"),
        "trainer_reloaded_ordered_tensor_sha": observation.get("trainer_reloaded_ordered_tensor_sha"),
        "trainer_reloaded_adapter_sha": observation.get("trainer_reloaded_adapter_sha"),
        "sampler_ordered_tensor_sha_before": observation.get("sampler_ordered_tensor_sha_before"),
        "sampler_ordered_tensor_sha_after": observation.get("sampler_ordered_tensor_sha_after"),
        "fresh_sampler_ordered_tensor_sha": observation.get(
            "fresh_sampler_ordered_tensor_sha"
        ),
        "sampler_loaded_adapter_sha": observation.get("sampler_loaded_adapter_sha"),
        "ordered_tensor_sha_semantics": (
            "canonical_adapter_independent_lora_name_shape_float32_le_bytes"
        ),
        "saved_adapter_sha_semantics": (
            "adapter_config_json_then_adapter_model_safetensors_file_bytes"
        ),
        "active_adapter_name": observation.get("active_adapter_name"),
        "old_adapter_name": observation.get("old_adapter_name"),
        "new_adapter_name": observation.get("new_adapter_name"),
        "old_adapter_removed": observation.get("old_adapter_removed"),
        "new_adapter_loaded": observation.get("new_adapter_loaded"),
        "model_base_tokenizer_identity": {
            field: observation.get(field)
            for field in (
                "base_revision",
                "trainer_base_revision",
                "sampler_base_revision",
                "tokenizer_revision",
                "trainer_tokenizer_revision",
                "sampler_tokenizer_revision",
            )
        },
        "trainer_in_memory_vs_reloaded_gap": probes["trainer_in_memory_vs_reloaded"],
        "live_refreshed_vs_fresh_sampler_gap": probes["live_refreshed_vs_fresh_sampler"],
        "repeated_same_instance_noise": probes["repeated_same_instance_noise"],
        "no_op_refresh_gap": probes["no_op_refresh_gap"],
        "no_op_refresh_control": no_op_control,
        "generation_direct_diagnostic": probes["generation_raw_vs_direct"],
        "processed_generation_raw_diagnostic": probes["processed_generation_vs_raw"],
        "cache_no_cache_diagnostic": probes["direct_cache_vs_no_cache"],
        "trainer_direct_sampler_generation_diagnostic": probes[
            "trainer_direct_vs_sampler_generation"
        ],
        "cross_path_diagnostics_are_hard_gates": False,
        "same_path_gates": same_path_gates,
        "control_gates": control_gates,
        "all_finite": all(summary["finite_rate"] == 1.0 for summary in probes.values()),
        "refresh_start": observation.get("refresh_start"),
        "refresh_end": observation.get("refresh_end"),
        "refresh_latency_seconds": latency,
        "stale_request_test": dict(stale) if isinstance(stale, Mapping) else None,
        "active_request_test": (
            dict(active_request) if isinstance(active_request, Mapping) else None
        ),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "failure_reason": "; ".join(failures) if failures else None,
        "failure_reasons": failures,
        "isolation": dict(isolation) if isinstance(isolation, Mapping) else isolation,
        "raw_prompt_or_response_persisted": False,
        "full_vocab_logits_persisted": False,
    }
    # Prove the public report itself remains JSON-finite before handing it to the
    # writer.  Non-finite input pairs are represented only by finite_rate < 1.
    json.dumps(report, allow_nan=False)
    return report


_PUBLIC_PROBE_KEYS = {
    "trainer_in_memory_vs_reloaded": "trainer_in_memory_vs_reloaded_gap",
    "live_refreshed_vs_fresh_sampler": "live_refreshed_vs_fresh_sampler_gap",
    "repeated_same_instance_noise": "repeated_same_instance_noise",
    "no_op_refresh_gap": "no_op_refresh_gap",
    "generation_raw_vs_direct": "generation_direct_diagnostic",
    "processed_generation_vs_raw": "processed_generation_raw_diagnostic",
    "direct_cache_vs_no_cache": "cache_no_cache_diagnostic",
    "trainer_direct_vs_sampler_generation": (
        "trainer_direct_sampler_generation_diagnostic"
    ),
}


def persisted_sampler_refresh_failures(
    report: Mapping[str, Any], *, expected_run_id: str
) -> list[str]:
    """Independently recompute readiness-relevant gates from public evidence."""

    failures: list[str] = []
    if not (
        report.get("schema_version") == 4
        and report.get("artifact_protocol_version")
        == SAMPLER_REFRESH_ARTIFACT_PROTOCOL
        and report.get("run_id") == expected_run_id
        and report.get("stage") == "sampler_refresh"
        and report.get("status") == report.get("gate_result") == "pass"
        and report.get("hard_gate_passed") is True
    ):
        failures.append("sampler report protocol/run/status mismatch")
    threshold = report.get("threshold")
    if threshold != SAMPLER_REFRESH_MAX_GAP or not isinstance(
        report.get("threshold_source"), str
    ):
        failures.append("sampler threshold identity mismatch")
    if not (
        report.get("trainer_version_before") == 0
        and report.get("sampler_version_before") == 0
        and report.get("trainer_version_after") == 1
        and report.get("sampler_version_after") == 1
        and all(
            type(report.get(field)) is int
            for field in (
                "trainer_version_before",
                "sampler_version_before",
                "trainer_version_after",
                "sampler_version_after",
            )
        )
    ):
        failures.append("persisted sampler version identity mismatch")
    ordered_before = report.get("trainer_ordered_tensor_sha_before")
    ordered_after = report.get("trainer_ordered_tensor_sha_after")
    if not (
        _valid_sha256(ordered_before)
        and _valid_sha256(ordered_after)
        and ordered_before == report.get("sampler_ordered_tensor_sha_before")
        and ordered_after
        == report.get("trainer_reloaded_ordered_tensor_sha")
        == report.get("sampler_ordered_tensor_sha_after")
        == report.get("fresh_sampler_ordered_tensor_sha")
        and ordered_before != ordered_after
    ):
        failures.append("persisted ordered tensor SHA identity mismatch")
    saved_sha = report.get("trainer_saved_adapter_sha")
    if not (
        _valid_sha256(saved_sha)
        and saved_sha
        == report.get("trainer_reloaded_adapter_sha")
        == report.get("sampler_loaded_adapter_sha")
    ):
        failures.append("persisted saved adapter SHA identity mismatch")
    if not (
        report.get("active_adapter_name") == report.get("new_adapter_name") == "version1"
        and report.get("old_adapter_name") in {"version0", "version0_noop"}
        and report.get("old_adapter_removed") is True
        and report.get("new_adapter_loaded") is True
    ):
        failures.append("persisted active/removed adapter identity mismatch")
    model_identity = report.get("model_base_tokenizer_identity")
    if not isinstance(model_identity, Mapping):
        failures.append("persisted Base/tokenizer identity is missing")
    else:
        values = [model_identity.get(field) for field in model_identity]
        base = [model_identity.get(field) for field in (
            "base_revision", "trainer_base_revision", "sampler_base_revision"
        )]
        tokenizer = [model_identity.get(field) for field in (
            "tokenizer_revision", "trainer_tokenizer_revision", "sampler_tokenizer_revision"
        )]
        if not (
            len(model_identity) == 6
            and all(isinstance(value, str) and bool(value) for value in values)
            and len(set(base)) == len(set(tokenizer)) == 1
        ):
            failures.append("persisted Base/tokenizer identity mismatch")

    all_probes_finite = True
    for internal_name, public_name in _PUBLIC_PROBE_KEYS.items():
        probe = report.get(public_name)
        if not isinstance(probe, Mapping):
            failures.append(f"{public_name} is missing")
            all_probes_finite = False
            continue
        expected_classification = _PROBE_SPECS[internal_name]
        left = probe.get("left_scorer")
        right = probe.get("right_scorer")
        scorer_ok = bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == _SCORER_KEYS
            and set(right) == _SCORER_KEYS
            and all(
                isinstance(item.get("backend"), str)
                and item["backend"].startswith("transformers-")
                and item.get("dtype") == "bfloat16"
                and item.get("mode") == "eval"
                and item.get("attention_backend") == "eager"
                and item.get("batch_size") == 1
                and item.get("attention_mask") == "all_ones_no_padding"
                and item.get("position_ids") == "implicit_from_attention_mask"
                and item.get("log_softmax_dtype") == "float32"
                and isinstance(item.get("eos_token_ids"), list)
                and bool(item["eos_token_ids"])
                and isinstance(item.get("pad_token_id"), int)
                and item.get("generation_processors_warpers") == []
                for item in (left, right)
            )
            and tuple(
                (item.get("path"), item.get("device"), item.get("use_cache"))
                for item in (left, right)
            )
            == _PROBE_SCORER_CONTRACT[internal_name]
        )
        numeric = [probe.get(key) for key in ("mae", "p50", "p95", "p99", "max")]
        finite_ok = bool(
            type(probe.get("count")) is int
            and probe.get("count", 0) > 0
            and probe.get("finite_count") == probe.get("count")
            and probe.get("finite_rate") == 1.0
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
                for value in numeric
            )
        )
        worst = probe.get("worst_token")
        worst_ok = bool(
            isinstance(worst, Mapping)
            and set(worst) == {
                "sample_id", "token_position", "token_id", "left", "right", "abs_gap"
            }
            and isinstance(worst.get("sample_id"), str)
            and bool(_SAFE_SAMPLE_ID.fullmatch(worst["sample_id"]))
        )
        same_path_expected = dict(left) == dict(right) if scorer_ok else False
        if not (
            probe.get("classification") == expected_classification
            and scorer_ok
            and finite_ok
            and worst_ok
            and probe.get("same_scoring_path") is same_path_expected
        ):
            failures.append(f"{public_name} persisted statistics/provenance mismatch")
            all_probes_finite = False
        if expected_classification == "formal_same_path_gate" and not (
            same_path_expected
            and isinstance(probe.get("max"), (int, float))
            and float(probe["max"]) <= SAMPLER_REFRESH_MAX_GAP
        ):
            failures.append(f"{public_name} persisted same-path gate failed")
        if internal_name in {
            "repeated_same_instance_noise",
            "no_op_refresh_gap",
        } and not (
            same_path_expected
            and isinstance(probe.get("max"), (int, float))
            and float(probe["max"]) <= SAMPLER_REFRESH_MAX_GAP
        ):
            failures.append(f"{public_name} persisted control gate failed")
    if report.get("all_finite") is not all_probes_finite:
        failures.append("persisted all_finite shortcut differs from recomputation")
    expected_same_path = {
        "trainer_in_memory_vs_reloaded": not any(
            "trainer_in_memory_vs_reloaded_gap" in item for item in failures
        ),
        "live_refreshed_vs_fresh_sampler": not any(
            "live_refreshed_vs_fresh_sampler_gap" in item for item in failures
        ),
    }
    if report.get("same_path_gates") != expected_same_path:
        failures.append("persisted same_path_gates shortcut differs from recomputation")
    expected_controls = {
        "repeated_same_instance_noise": not any(
            "repeated_same_instance_noise" in item for item in failures
        ),
        "no_op_refresh_gap": not any(
            "no_op_refresh_gap" in item for item in failures
        ),
    }
    if report.get("control_gates") != expected_controls:
        failures.append("persisted control_gates shortcut differs from recomputation")
    no_op = report.get("no_op_refresh_control")
    fresh_gap = no_op.get("fresh_reload_gap") if isinstance(no_op, Mapping) else None
    no_op_before_sha = (
        no_op.get("ordered_tensor_sha_before") if isinstance(no_op, Mapping) else None
    )
    no_op_file_sha = (
        no_op.get("saved_adapter_sha_before") if isinstance(no_op, Mapping) else None
    )
    no_op_normal = no_op.get("normal_request") if isinstance(no_op, Mapping) else None
    no_op_stale = no_op.get("stale_request") if isinstance(no_op, Mapping) else None
    if not (
        isinstance(no_op, Mapping)
        and no_op.get("version_before")
        == no_op.get("version_after")
        == no_op.get("fresh_version")
        == 0
        and _valid_sha256(no_op_before_sha)
        and no_op_before_sha
        == no_op.get("ordered_tensor_sha_after")
        == no_op.get("fresh_ordered_tensor_sha")
        and _valid_sha256(no_op_file_sha)
        and no_op_file_sha
        == no_op.get("saved_adapter_sha_after")
        == no_op.get("fresh_saved_adapter_sha")
        and no_op.get("active_adapter_before") == "version0"
        and no_op.get("active_adapter_after") == "version0_noop"
        and no_op.get("fresh_active_adapter") == "version0_fresh"
        and no_op.get("old_adapter_removed") is True
        and no_op.get("new_adapter_loaded") is True
        and no_op.get("identity_gate_passed") is True
        and no_op.get("request_guard_gate_passed") is True
        and no_op.get("latency_gate_passed") is True
        and no_op.get("fresh_reload_gap_gate_passed") is True
        and isinstance(no_op.get("refresh_latency_seconds"), (int, float))
        and not isinstance(no_op.get("refresh_latency_seconds"), bool)
        and math.isfinite(float(no_op["refresh_latency_seconds"]))
        and float(no_op["refresh_latency_seconds"]) >= 0
        and isinstance(fresh_gap, Mapping)
        and fresh_gap.get("finite_rate") == 1.0
        and fresh_gap.get("same_scoring_path") is True
        and isinstance(fresh_gap.get("max"), (int, float))
        and float(fresh_gap["max"]) <= SAMPLER_REFRESH_MAX_GAP
        and isinstance(no_op_normal, Mapping)
        and no_op_normal.get("requested_version") == 0
        and no_op_normal.get("requested_ordered_tensor_sha") == no_op_before_sha
        and no_op_normal.get("requested_run_token")
        == f"{expected_run_id}:adapter-v0"
        and no_op_normal.get("accepted") is True
        and no_op_normal.get("scoring_executed") is True
        and isinstance(no_op_stale, Mapping)
        and no_op_stale.get("requested_version") == 0
        and no_op_stale.get("requested_ordered_tensor_sha") != no_op_before_sha
        and no_op_stale.get("requested_run_token")
        == f"{expected_run_id}:adapter-v0"
        and no_op_stale.get("rejected") is True
        and no_op_stale.get("scoring_executed") is False
        and no_op_stale.get("rejection_phase")
        == "identity_guard_before_scoring"
        and no_op_stale.get("error_type") == "StaleSamplerRequestError"
    ):
        failures.append("persisted v0 no-op control identity/guard/gap mismatch")
    stale = report.get("stale_request_test")
    if not (
        isinstance(stale, Mapping)
        and stale.get("requested_version") == 0
        and stale.get("active_version") == 1
        and stale.get("requested_ordered_tensor_sha") == ordered_before
        and stale.get("active_ordered_tensor_sha") == ordered_after
        and stale.get("requested_run_token") == f"{expected_run_id}:adapter-v0"
        and stale.get("active_run_token") == f"{expected_run_id}:adapter-v1"
        and stale.get("rejected") is True
        and stale.get("silent_fallback") is False
        and stale.get("scoring_executed") is False
        and stale.get("routable_adapter_names_after_refresh") == ["version1"]
        and stale.get("error_type") == "StaleSamplerRequestError"
        and stale.get("error_code") == "STALE_SAMPLER_IDENTITY"
        and stale.get("rejection_phase") == "identity_guard_before_scoring"
        and isinstance(stale.get("latency_seconds"), (int, float))
        and not isinstance(stale.get("latency_seconds"), bool)
        and math.isfinite(float(stale["latency_seconds"]))
        and float(stale["latency_seconds"]) >= 0
    ):
        failures.append("persisted stale request evidence mismatch")
    active_request = report.get("active_request_test")
    if not (
        isinstance(active_request, Mapping)
        and active_request.get("requested_version") == 1
        and active_request.get("requested_ordered_tensor_sha") == ordered_after
        and active_request.get("requested_run_token")
        == f"{expected_run_id}:adapter-v1"
        and active_request.get("accepted") is True
        and active_request.get("scoring_executed") is True
        and type(active_request.get("guarded_call_count")) is int
        and active_request.get("guarded_call_count", 0) >= 4
        and active_request.get("guarded_request_types")
        == ["fixed_action", "generation", "direct_no_cache", "direct_cache"]
        and type(active_request.get("result_token_count")) is int
        and active_request.get("result_token_count", 0) > 0
        and active_request.get("result_all_finite") is True
    ):
        failures.append("persisted active request guard evidence mismatch")
    latency = report.get("refresh_latency_seconds")
    if not (
        isinstance(report.get("refresh_start"), str)
        and isinstance(report.get("refresh_end"), str)
        and isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) >= 0
    ):
        failures.append("persisted refresh latency evidence mismatch")
    if not (
        report.get("cross_path_diagnostics_are_hard_gates") is False
        and report.get("failure_reason") is None
        and report.get("failure_reasons") == []
        and report.get("isolation")
        == {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }
        and report.get("raw_prompt_or_response_persisted") is False
        and report.get("full_vocab_logits_persisted") is False
    ):
        failures.append("persisted isolation/privacy/failure semantics mismatch")
    return failures


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SamplerRefreshContractError("metrics.jsonl cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_index(output: Path) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(output)): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name != "artifact_index.json"
        and not path.name.endswith(".tmp")
    }
    value = {
        "schema_version": 4,
        "artifact_protocol_version": SAMPLER_REFRESH_ARTIFACT_PROTOCOL,
        "status": "preliminary_failure_index",
        "artifacts": artifacts,
    }
    _atomic_json(output / "artifact_index.json", value)
    return value


def _unobserved_probe(classification: str) -> dict[str, Any]:
    return {
        "count": 0,
        "finite_count": 0,
        "finite_rate": 0.0,
        "mae": None,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
        "worst_token": None,
        "classification": classification,
        "same_scoring_path": None,
        "left_scorer": None,
        "right_scorer": None,
        "observation_status": "not_observed_before_runtime_failure",
    }


def persist_sampler_refresh_runtime_failure(
    output_dir: str | Path,
    *,
    run_id: str,
    failed_phase: str,
    error_type: str,
    error: str,
    failure_status: str = "failed_sampler_refresh",
    correction_metrics: Mapping[str, Any] | None = None,
    one_step_metrics: Mapping[str, Any] | None = None,
    null_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed with schema-shaped evidence even before all probes exist.

    Missing observations remain explicit null/not-observed values.  This writer
    never fabricates a numeric result and never raises an experimental gate.
    """

    if not isinstance(run_id, str) or not run_id:
        raise SamplerRefreshContractError("runtime failure run_id is missing")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    same_path = _unobserved_probe("formal_same_path_gate")
    control = _unobserved_probe("control_diagnostic")
    cross_path = _unobserved_probe("cross_path_diagnostic")
    report = {
        "schema_version": 4,
        "artifact_protocol_version": SAMPLER_REFRESH_ARTIFACT_PROTOCOL,
        "run_id": run_id,
        "stage": "sampler_refresh",
        "status": "fail",
        "gate_result": "fail",
        "hard_gate_passed": False,
        "failure_phase": str(failed_phase),
        "trainer_version_before": None,
        "trainer_version_after": None,
        "sampler_version_before": None,
        "sampler_version_after": None,
        "trainer_ordered_tensor_sha_before": None,
        "trainer_ordered_tensor_sha_after": None,
        "trainer_saved_adapter_sha": None,
        "trainer_reloaded_ordered_tensor_sha": None,
        "trainer_reloaded_adapter_sha": None,
        "sampler_ordered_tensor_sha_before": None,
        "sampler_ordered_tensor_sha_after": None,
        "fresh_sampler_ordered_tensor_sha": None,
        "sampler_loaded_adapter_sha": None,
        "ordered_tensor_sha_semantics": (
            "canonical_adapter_independent_lora_name_shape_float32_le_bytes"
        ),
        "saved_adapter_sha_semantics": (
            "adapter_config_json_then_adapter_model_safetensors_file_bytes"
        ),
        "active_adapter_name": None,
        "old_adapter_name": None,
        "new_adapter_name": "version1",
        "old_adapter_removed": False,
        "new_adapter_loaded": False,
        "model_base_tokenizer_identity": None,
        "trainer_in_memory_vs_reloaded_gap": dict(same_path),
        "live_refreshed_vs_fresh_sampler_gap": dict(same_path),
        "repeated_same_instance_noise": dict(control),
        "no_op_refresh_gap": dict(control),
        "generation_direct_diagnostic": dict(cross_path),
        "processed_generation_raw_diagnostic": dict(cross_path),
        "cache_no_cache_diagnostic": dict(cross_path),
        "trainer_direct_sampler_generation_diagnostic": dict(cross_path),
        "scoring_environment": {
            "dtype": None,
            "device": None,
            "attention_backend": None,
            "use_cache": None,
        },
        "refresh_start": None,
        "refresh_end": None,
        "refresh_latency_seconds": None,
        "stale_request_test": {
            "status": "not_observed_before_runtime_failure",
            "rejected": False,
            "silent_fallback": None,
            "scoring_executed": None,
            "routable_adapter_names_after_refresh": None,
        },
        "active_request_test": {
            "status": "not_observed_before_runtime_failure",
            "accepted": False,
            "scoring_executed": None,
            "guarded_call_count": 0,
        },
        "threshold": SAMPLER_REFRESH_MAX_GAP,
        "threshold_source": "ADR-0018 sampler refresh contract v4",
        "failure_reason": f"{error_type}: {error}",
        "failure_reasons": [f"{error_type}: {error}"],
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
        "raw_prompt_or_response_persisted": False,
        "full_vocab_logits_persisted": False,
    }
    sampler_path = output / "sampler_refresh.json"
    _atomic_json(sampler_path, report)
    sampler_sha = _sha256_file(sampler_path)
    aggregate_rows = (
        ("correction_calibration_16", correction_metrics),
        ("corrected_medical_one_step", one_step_metrics),
        ("real_base_teacher_null_update", null_metrics),
    )
    rows = [
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": step,
            "phase": phase,
            "status": "not_run_or_not_observed",
            **dict(metrics or {}),
        }
        for step, (phase, metrics) in enumerate(aggregate_rows, start=2)
    ]
    rows.append(
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 5,
            "phase": "sampler_refresh",
            "status": "fail",
            "gate_result": "fail",
            "failed_phase": str(failed_phase),
        }
    )
    _atomic_jsonl(output / "metrics.jsonl", rows)
    failure = {
        "schema_version": 4,
        "run_id": run_id,
        "status": str(failure_status),
        "phase": str(failed_phase),
        "error_type": str(error_type),
        "error": str(error),
        "sampler_refresh_path": "sampler_refresh.json",
        "sampler_refresh_sha256": sampler_sha,
        "metrics_path": "metrics.jsonl",
        "metrics_sha256": _sha256_file(output / "metrics.jsonl"),
        "B2_authorized": False,
    }
    _atomic_json(output / "failure.json", failure)
    _artifact_index(output)
    return {
        "status": str(failure_status),
        "sampler_refresh_sha256": sampler_sha,
        "metrics_sha256": failure["metrics_sha256"],
    }


def persist_sampler_v0_control_failure(
    output_dir: str | Path,
    control_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist observed v0 controls before raising ahead of optimizer work."""

    if (
        control_report.get("schema_version") != 4
        or control_report.get("stage") != "sampler_v0_controls"
        or control_report.get("hard_gate_passed") is not False
        or control_report.get("failure_status")
        not in {"failed_same_instance_repeat", "failed_no_op_refresh"}
    ):
        raise SamplerRefreshContractError("v0 control failure report is invalid")
    output = Path(output_dir)
    run_id = str(control_report["run_id"])
    failure_status = str(control_report["failure_status"])
    failed_phase = (
        "sampler_v0_repeated_probe"
        if failure_status == "failed_same_instance_repeat"
        else "sampler_v0_noop_unload_reload_control"
    )
    reason = "; ".join(str(item) for item in control_report["failure_reasons"])
    persist_sampler_refresh_runtime_failure(
        output,
        run_id=run_id,
        failed_phase=failed_phase,
        failure_status=failure_status,
        error_type="SamplerRefreshGateError",
        error=reason,
        correction_metrics={"status": "not_run"},
        one_step_metrics={"status": "not_run"},
        null_metrics={"status": "not_run"},
    )
    sampler_path = output / "sampler_refresh.json"
    sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
    sampler.update(
        {
            "status": "fail",
            "gate_result": "fail",
            "hard_gate_passed": False,
            "failure_phase": failed_phase,
            "failure_reason": reason,
            "failure_reasons": list(control_report["failure_reasons"]),
            "threshold": control_report["threshold"],
            "repeated_same_instance_noise": dict(
                control_report["repeated_same_instance_noise"]
            ),
            "no_op_refresh_gap": dict(control_report["no_op_refresh_gap"]),
            "no_op_refresh_control": dict(
                control_report["no_op_refresh_control"]
            ),
            "control_gates": {
                "repeated_same_instance_noise": (
                    control_report["repeated_same_instance_noise"].get("max")
                    is not None
                    and control_report["repeated_same_instance_noise"]["max"]
                    <= SAMPLER_REFRESH_MAX_GAP
                ),
                "no_op_refresh_gap": (
                    control_report["no_op_refresh_gap"].get("max") is not None
                    and control_report["no_op_refresh_gap"]["max"]
                    <= SAMPLER_REFRESH_MAX_GAP
                ),
            },
        }
    )
    _atomic_json(sampler_path, sampler)
    return persist_sampler_refresh_failure_binding(
        output,
        run_id=run_id,
        failed_phase=failed_phase,
        failure_status=failure_status,
        error_type="SamplerRefreshGateError",
        error=reason,
    )


def persist_sampler_refresh_failure_binding(
    output_dir: str | Path,
    *,
    run_id: str,
    failed_phase: str,
    failure_status: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    """Rebuild observed aggregates, then bind a later failure to their SHAs."""

    output = Path(output_dir)
    sampler_path = output / "sampler_refresh.json"
    if not sampler_path.is_file():
        raise SamplerRefreshContractError("sampler artifact is absent for failure binding")

    def read_observed(name: str) -> dict[str, Any] | None:
        path = output / name
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    sampler = read_observed("sampler_refresh.json") or {}
    correction_phase = (
        "correction_calibration_32"
        if (output / "correction_calibration_32.json").is_file()
        else "correction_calibration_16"
    )
    correction = read_observed(f"{correction_phase}.json")
    correction_values = (correction or {}).get("rollout_correction", {})
    medical = read_observed("corrected_medical_one_step.json")
    null = read_observed("real_base_teacher_null_update.json")

    def observed_status(value: Mapping[str, Any] | None, passed: bool) -> str:
        if value is None:
            return "not_run_or_not_observed"
        return "pass" if passed else "fail"

    def numeric_delta(after: Any, before: Any) -> float | None:
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (after, before)
        ):
            return float(after) - float(before)
        return None

    metric_rows = [
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 2,
            "phase": correction_phase,
            "status": observed_status(
                correction,
                (correction or {}).get("calibration_readiness", {}).get(
                    "calibration_ready"
                )
                is True,
            ),
            "ess_fraction": correction_values.get("ess_fraction"),
            "cap_fraction": correction_values.get("cap_fraction"),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 3,
            "phase": "corrected_medical_one_step",
            "status": observed_status(
                medical,
                (medical or {}).get("status") == "pass"
                and (medical or {}).get("hard_gate_passed") is True,
            ),
            "objective_delta": numeric_delta(
                (medical or {}).get("objective_after"),
                (medical or {}).get("objective_before"),
            ),
            "loss_delta": numeric_delta(
                (medical or {}).get("loss_after"),
                (medical or {}).get("loss_before"),
            ),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 4,
            "phase": "real_base_teacher_null_update",
            "status": observed_status(
                null,
                (null or {}).get("status") == "pass"
                and (null or {}).get("hard_gate_passed") is True,
            ),
            "advantage_max_abs": (null or {}).get("advantage_max_abs"),
            "parameter_delta_norm": (null or {}).get("parameter_delta_norm"),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 5,
            "phase": "sampler_refresh",
            "status": str(sampler.get("status", "fail")),
            "gate_result": sampler.get("gate_result"),
            "trainer_reload_max_gap": (
                sampler.get("trainer_in_memory_vs_reloaded_gap") or {}
            ).get("max"),
            "live_fresh_max_gap": (
                sampler.get("live_refreshed_vs_fresh_sampler_gap") or {}
            ).get("max"),
            "generation_direct_max_gap": (
                sampler.get("generation_direct_diagnostic") or {}
            ).get("max"),
            "refresh_latency_seconds": sampler.get("refresh_latency_seconds"),
            "stale_request_rejected": (
                sampler.get("stale_request_test") or {}
            ).get("rejected"),
        },
    ]
    metrics_path = output / "metrics.jsonl"
    _atomic_jsonl(metrics_path, metric_rows)
    failure = {
        "schema_version": 4,
        "run_id": run_id,
        "status": failure_status,
        "phase": failed_phase,
        "error_type": error_type,
        "error": error,
        "sampler_refresh_path": "sampler_refresh.json",
        "sampler_refresh_sha256": _sha256_file(sampler_path),
        "metrics_path": "metrics.jsonl",
        "metrics_sha256": _sha256_file(metrics_path),
        "B2_authorized": False,
    }
    _atomic_json(output / "failure.json", failure)
    _artifact_index(output)
    return failure


def persist_sampler_refresh_evidence(
    output_dir: str | Path,
    report: Mapping[str, Any],
    *,
    correction_metrics: Mapping[str, Any],
    one_step_metrics: Mapping[str, Any],
    null_metrics: Mapping[str, Any],
    correction_phase: str = "correction_calibration_16",
) -> dict[str, Any]:
    """Write all refresh evidence atomically, then assert the hard gate."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if report.get("artifact_protocol_version") != SAMPLER_REFRESH_ARTIFACT_PROTOCOL:
        raise SamplerRefreshContractError("sampler report protocol mismatch")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SamplerRefreshContractError("sampler report run_id is missing")

    sampler_path = output / "sampler_refresh.json"
    _atomic_json(sampler_path, report)
    sampler_sha = _sha256_file(sampler_path)
    metric_rows = [
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 2,
            "phase": correction_phase,
            "status": "pass",
            **dict(correction_metrics),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 3,
            "phase": "corrected_medical_one_step",
            "status": "pass",
            **dict(one_step_metrics),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 4,
            "phase": "real_base_teacher_null_update",
            "status": "pass",
            **dict(null_metrics),
        },
        {
            "schema_version": 4,
            "run_id": run_id,
            "step": 5,
            "phase": "sampler_refresh",
            "status": str(report.get("status")),
            "gate_result": report.get("gate_result"),
            "trainer_reload_max_gap": report.get(
                "trainer_in_memory_vs_reloaded_gap", {}
            ).get("max"),
            "live_fresh_max_gap": report.get(
                "live_refreshed_vs_fresh_sampler_gap", {}
            ).get("max"),
            "generation_direct_max_gap": report.get(
                "generation_direct_diagnostic", {}
            ).get("max"),
            "refresh_latency_seconds": report.get("refresh_latency_seconds"),
            "stale_request_rejected": (
                report.get("stale_request_test") or {}
            ).get("rejected"),
        },
    ]
    _atomic_jsonl(output / "metrics.jsonl", metric_rows)

    if report.get("gate_result") != "pass" or report.get("hard_gate_passed") is not True:
        failure = {
            "schema_version": 4,
            "run_id": run_id,
            "status": "failed_sampler_refresh",
            "phase": "sampler_refresh",
            "failure_reason": report.get("failure_reason"),
            "sampler_refresh_path": "sampler_refresh.json",
            "sampler_refresh_sha256": sampler_sha,
            "metrics_path": "metrics.jsonl",
            "metrics_sha256": _sha256_file(output / "metrics.jsonl"),
            "B2_authorized": False,
        }
        _atomic_json(output / "failure.json", failure)
        _artifact_index(output)
        raise SamplerRefreshGateError(str(report.get("failure_reason")))
    return {
        "status": "pass",
        "sampler_refresh_sha256": sampler_sha,
        "metrics_sha256": _sha256_file(output / "metrics.jsonl"),
    }
