"""CPU-safe semantic contracts for the P4.6 production qualification.

The helpers in this module deliberately operate on JSON primitives only.  GPU
runtime code may build these payloads while tensors are live, but readiness can
later validate the persisted evidence without importing a model backend.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 6
PROBE_SELECTION_RULE = "first_32_valid_response_tokens_per_prompt_v1"
PRODUCTION_SLOT = "student_active"
RUNTIME_TENSOR_MISMATCH = "SAMPLER_RUNTIME_TENSOR_MISMATCH"
STALE_SAMPLER_IDENTITY = "STALE_SAMPLER_IDENTITY"
SAME_PATH_MAX_GAP = 1e-4
NULL_ADVANTAGE_TOLERANCE = 1e-8


class QualificationContractError(ValueError):
    """Persisted qualification evidence is absent, malformed, or contradictory."""


def _fail(message: str) -> None:
    raise QualificationContractError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        _fail(f"{label} fields are not exact; missing={missing}, extra={extra}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _boolean(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        _fail(f"{label} must be {expected}")


def _digest(value: Any, label: str, *, length: int = 64) -> str:
    if not (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be an immutable {length}-hex digest")
    return value


def _json_primitives(value: Any, label: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_primitives(item, f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{label} contains a non-string key")
            result[key] = _json_primitives(item, f"{label}.{key}")
        return result
    _fail(f"{label} contains unsupported value {type(value).__name__}")


def _canonical_sha(value: Mapping[str, Any], *, excluded_field: str | None = None) -> str:
    payload = {
        key: item for key, item in value.items() if excluded_field is None or key != excluded_field
    }
    safe = _json_primitives(payload)
    encoded = json.dumps(
        safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PROBE_SPEC_FIELDS = {
    "schema_version",
    "artifact_kind",
    "selection_rule",
    "run_id",
    "prompt_manifest_sha256",
    "ordered_sample_ids",
    "per_prompt_limit",
    "mask_semantics",
    "attention_semantics",
    "probe_spec_sha256",
}


def build_probe_spec(
    *,
    run_id: str,
    prompt_manifest_sha256: str,
    ordered_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze the selection rule before rollout values are available."""

    run_id = _string(run_id, "run_id")
    prompt_manifest_sha256 = _digest(prompt_manifest_sha256, "prompt manifest SHA")
    if isinstance(ordered_sample_ids, (str, bytes)) or not isinstance(
        ordered_sample_ids, Sequence
    ):
        _fail("ordered sample ids must be a sequence")
    sample_ids = [_string(item, "sample_id") for item in ordered_sample_ids]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        _fail("ordered sample ids must be non-empty and unique")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fixed_action_probe_spec_v6",
        "selection_rule": PROBE_SELECTION_RULE,
        "run_id": run_id,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "ordered_sample_ids": sample_ids,
        "per_prompt_limit": 32,
        "mask_semantics": "valid_response_tokens_only_v1",
        "attention_semantics": "causal_attention_mask_with_prompt_prefix_v1",
    }
    payload["probe_spec_sha256"] = _canonical_sha(payload)
    return payload


def _validate_probe_spec(spec: Mapping[str, Any]) -> None:
    _exact_fields(spec, _PROBE_SPEC_FIELDS, "probe spec")
    if spec.get("schema_version") != SCHEMA_VERSION:
        _fail("probe spec schema version is not v6")
    if spec.get("artifact_kind") != "fixed_action_probe_spec_v6":
        _fail("probe spec kind is invalid")
    if spec.get("selection_rule") != PROBE_SELECTION_RULE:
        _fail("probe selection rule is not frozen")
    _string(spec.get("run_id"), "probe spec run_id")
    _digest(spec.get("prompt_manifest_sha256"), "probe prompt manifest SHA")
    ids = spec.get("ordered_sample_ids")
    if not isinstance(ids, list) or not ids:
        _fail("probe spec ordered sample ids are absent")
    normalized = [_string(item, "probe sample_id") for item in ids]
    if len(normalized) != len(set(normalized)):
        _fail("probe sample ids are not unique")
    if spec.get("per_prompt_limit") != 32:
        _fail("probe per-prompt limit is not frozen at 32")
    if spec.get("mask_semantics") != "valid_response_tokens_only_v1":
        _fail("probe mask semantics changed")
    if spec.get("attention_semantics") != "causal_attention_mask_with_prompt_prefix_v1":
        _fail("probe attention semantics changed")
    expected = _canonical_sha(spec, excluded_field="probe_spec_sha256")
    if _digest(spec.get("probe_spec_sha256"), "probe spec SHA") != expected:
        _fail("probe spec self-hash mismatch")


_PROBE_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_kind",
    "selection_rule",
    "run_id",
    "prompt_manifest_sha256",
    "probe_spec_sha256",
    "frozen_after_rollout_before_optimizer",
    "mask_semantics",
    "attention_semantics",
    "per_prompt_limit",
    "ordered_sample_ids",
    "tokens",
    "per_prompt_count",
    "valid_response_token_count",
    "total_probe_count",
    "manifest_sha256",
}


def build_probe_manifest(
    spec: Mapping[str, Any],
    response_tokens_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Select fixed actions deterministically after rollout and before update."""

    spec = _mapping(spec, "probe spec")
    _validate_probe_spec(spec)
    source = _mapping(response_tokens_by_sample, "response tokens by sample")
    expected_ids = list(spec["ordered_sample_ids"])
    if set(source) != set(expected_ids):
        _fail("response token sample ids do not match the frozen prompt order")

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    valid_counts: dict[str, int] = {}
    for prompt_order, sample_id in enumerate(expected_ids):
        records = source[sample_id]
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            _fail(f"response tokens for {sample_id} are not a sequence")
        last_position = -1
        prompt_selected: list[dict[str, Any]] = []
        valid_count = 0
        for raw in records:
            record = _mapping(raw, f"response token for {sample_id}")
            _exact_fields(
                record,
                {"token_id", "response_token_position", "valid"},
                f"response token for {sample_id}",
            )
            position = _integer(
                record.get("response_token_position"),
                f"response token position for {sample_id}",
            )
            if position <= last_position:
                _fail(f"response token positions for {sample_id} are not strictly increasing")
            last_position = position
            token_id = _integer(record.get("token_id"), f"token id for {sample_id}")
            if not isinstance(record.get("valid"), bool):
                _fail(f"response token validity for {sample_id} is not boolean")
            if record["valid"]:
                valid_token_ordinal = valid_count
                valid_count += 1
                if len(prompt_selected) < 32:
                    prompt_selected.append(
                        {
                            "sample_id": sample_id,
                            "prompt_order": prompt_order,
                            "response_token_position": position,
                            "valid_token_ordinal": valid_token_ordinal,
                            "token_id": token_id,
                        }
                    )
        if valid_count < 1:
            _fail(f"response for {sample_id} has no valid response token")
        selected.extend(prompt_selected)
        counts[sample_id] = len(prompt_selected)
        valid_counts[sample_id] = valid_count

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fixed_action_probe_manifest_v6",
        "selection_rule": PROBE_SELECTION_RULE,
        "run_id": spec["run_id"],
        "prompt_manifest_sha256": spec["prompt_manifest_sha256"],
        "probe_spec_sha256": spec["probe_spec_sha256"],
        "frozen_after_rollout_before_optimizer": True,
        "mask_semantics": spec["mask_semantics"],
        "attention_semantics": spec["attention_semantics"],
        "per_prompt_limit": 32,
        "ordered_sample_ids": expected_ids,
        "tokens": selected,
        "per_prompt_count": counts,
        "valid_response_token_count": valid_counts,
        "total_probe_count": len(selected),
    }
    payload["manifest_sha256"] = _canonical_sha(payload)
    validate_probe_manifest(payload, spec)
    return payload


def validate_probe_manifest(
    manifest: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _mapping(manifest, "probe manifest")
    spec = _mapping(spec, "probe spec")
    _validate_probe_spec(spec)
    _exact_fields(manifest, _PROBE_MANIFEST_FIELDS, "probe manifest")
    for field in (
        "schema_version",
        "selection_rule",
        "run_id",
        "prompt_manifest_sha256",
        "mask_semantics",
        "attention_semantics",
        "per_prompt_limit",
        "ordered_sample_ids",
    ):
        if manifest.get(field) != spec.get(field):
            _fail(f"probe manifest {field} differs from frozen spec")
    if manifest.get("artifact_kind") != "fixed_action_probe_manifest_v6":
        _fail("probe manifest kind is invalid")
    if manifest.get("probe_spec_sha256") != spec.get("probe_spec_sha256"):
        _fail("probe manifest binds a different probe spec")
    _boolean(
        manifest.get("frozen_after_rollout_before_optimizer"),
        True,
        "probe manifest pre-optimizer freeze",
    )

    counts = manifest.get("per_prompt_count")
    if not isinstance(counts, Mapping) or set(counts) != set(spec["ordered_sample_ids"]):
        _fail("probe per-prompt counts do not bind every frozen sample")
    valid_counts = manifest.get("valid_response_token_count")
    if not isinstance(valid_counts, Mapping) or set(valid_counts) != set(
        spec["ordered_sample_ids"]
    ):
        _fail("probe valid response token counts do not bind every frozen sample")
    observed_counts = {sample_id: 0 for sample_id in spec["ordered_sample_ids"]}
    last_prompt_order = -1
    last_position_by_sample: dict[str, int] = {}
    tokens = manifest.get("tokens")
    if not isinstance(tokens, list):
        _fail("probe token list is absent")
    for raw in tokens:
        record = _mapping(raw, "probe token")
        _exact_fields(
            record,
            {
                "sample_id",
                "prompt_order",
                "response_token_position",
                "valid_token_ordinal",
                "token_id",
            },
            "probe token",
        )
        sample_id = _string(record.get("sample_id"), "probe token sample_id")
        if sample_id not in observed_counts:
            _fail("probe token references an unfrozen sample")
        prompt_order = _integer(record.get("prompt_order"), "probe token prompt order")
        if prompt_order >= len(spec["ordered_sample_ids"]) or spec["ordered_sample_ids"][
            prompt_order
        ] != sample_id:
            _fail("probe token prompt order is inconsistent")
        if prompt_order < last_prompt_order:
            _fail("probe tokens are not grouped in frozen prompt order")
        last_prompt_order = prompt_order
        position = _integer(
            record.get("response_token_position"), "probe response token position"
        )
        previous = last_position_by_sample.get(sample_id, -1)
        if position <= previous:
            _fail("probe response token positions are not strictly increasing")
        last_position_by_sample[sample_id] = position
        valid_token_ordinal = _integer(
            record.get("valid_token_ordinal"), "probe valid response token ordinal"
        )
        if valid_token_ordinal != observed_counts[sample_id]:
            _fail("probe valid response token ordinals are not the first prefix")
        _integer(record.get("token_id"), "probe token id")
        observed_counts[sample_id] += 1
        if observed_counts[sample_id] > 32:
            _fail("probe contains more than 32 tokens for one prompt")
    expected_counts: dict[str, int] = {}
    for sample_id in spec["ordered_sample_ids"]:
        expected_counts[sample_id] = _integer(
            counts.get(sample_id), f"probe count for {sample_id}"
        )
        valid_count = _integer(
            valid_counts.get(sample_id),
            f"probe valid response token count for {sample_id}",
            minimum=1,
        )
        if expected_counts[sample_id] != min(32, valid_count):
            _fail("probe count does not contain the first valid response token prefix")
    if expected_counts != observed_counts:
        _fail("probe token counts do not match the manifest")
    total = _integer(manifest.get("total_probe_count"), "total probe count", minimum=1)
    if total != len(tokens):
        _fail("probe total count does not match token records")
    expected_hash = _canonical_sha(manifest, excluded_field="manifest_sha256")
    if _digest(manifest.get("manifest_sha256"), "probe manifest SHA") != expected_hash:
        _fail("probe manifest self-hash mismatch")
    return deepcopy(dict(manifest))


_V0_NORMAL_FIELDS = {
    "run_id",
    "logical_version",
    "accepted",
    "guard_stage",
    "request_expected_tensor_sha256",
    "trainer_authoritative_tensor_sha256",
    "sampler_runtime_tensor_sha256",
    "authority_after_request_sha256",
    "canonical_config_sha256",
    "base_revision",
    "tokenizer_revision",
    "scoring_executed",
    "generation_executed",
    "finite",
    "silent_fallback",
}
_V0_WRONG_FIELDS = _V0_NORMAL_FIELDS | {"error_code", "sampler_self_authority_accepted"}


def _validate_common_guard(value: Mapping[str, Any], label: str) -> None:
    _string(value.get("run_id"), f"{label} run_id")
    if value.get("logical_version") != 0:
        _fail(f"{label} must target logical v0")
    if value.get("guard_stage") != "identity_guard_before_forward":
        _fail(f"{label} was not guarded before forward")
    for field in (
        "request_expected_tensor_sha256",
        "trainer_authoritative_tensor_sha256",
        "sampler_runtime_tensor_sha256",
        "authority_after_request_sha256",
        "canonical_config_sha256",
    ):
        _digest(value.get(field), f"{label} {field}")
    _digest(value.get("base_revision"), f"{label} base revision", length=40)
    _digest(value.get("tokenizer_revision"), f"{label} tokenizer revision", length=40)
    _boolean(value.get("silent_fallback"), False, f"{label} silent fallback")
    if not isinstance(value.get("finite"), bool):
        _fail(f"{label} finite must be boolean")


def validate_v0_guard_evidence(
    normal: Mapping[str, Any], wrong_authority: Mapping[str, Any]
) -> dict[str, Any]:
    normal = _mapping(normal, "v0 normal guard")
    wrong = _mapping(wrong_authority, "v0 wrong-authority guard")
    _exact_fields(normal, _V0_NORMAL_FIELDS, "v0 normal guard")
    _exact_fields(wrong, _V0_WRONG_FIELDS, "v0 wrong-authority guard")
    _validate_common_guard(normal, "v0 normal")
    _validate_common_guard(wrong, "v0 wrong-authority")
    for field in (
        "run_id",
        "canonical_config_sha256",
        "base_revision",
        "tokenizer_revision",
    ):
        if normal.get(field) != wrong.get(field):
            _fail(f"v0 guard evidence disagrees on {field}")

    authority = normal["trainer_authoritative_tensor_sha256"]
    if not (
        normal.get("accepted") is True
        and normal.get("scoring_executed") is True
        and normal.get("generation_executed") is False
        and normal.get("finite") is True
        and normal.get("request_expected_tensor_sha256") == authority
        and normal.get("sampler_runtime_tensor_sha256") == authority
        and normal.get("authority_after_request_sha256") == authority
    ):
        _fail("v0 normal request did not execute exactly once under trainer authority")

    if wrong.get("trainer_authoritative_tensor_sha256") != authority:
        _fail("wrong-authority probe changed trainer authority")
    if not (
        wrong.get("accepted") is False
        and wrong.get("error_code") == RUNTIME_TENSOR_MISMATCH
        and wrong.get("scoring_executed") is False
        and wrong.get("generation_executed") is False
        and wrong.get("finite") is True
        and wrong.get("request_expected_tensor_sha256") != authority
        and wrong.get("sampler_runtime_tensor_sha256") == authority
        and wrong.get("authority_after_request_sha256") == authority
        and wrong.get("sampler_self_authority_accepted") is False
    ):
        _fail("v0 wrong-authority request was not rejected before forward")
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "v0_normal": deepcopy(dict(normal)),
        "v0_wrong_authority": deepcopy(dict(wrong)),
    }


def _refresh_link(
    value: Mapping[str, Any],
    *,
    label: str,
    from_version: int,
    to_version: int,
    expected_sha: str,
) -> str:
    _exact_fields(
        value,
        {
            "from_version",
            "to_version",
            "target_authority_tensor_sha256",
            "published_runtime_tensor_sha256",
            "artifact_sha256",
        },
        label,
    )
    if value.get("from_version") != from_version or value.get("to_version") != to_version:
        _fail(f"{label} logical transition is not v{from_version}->v{to_version}")
    if not (
        _digest(value.get("target_authority_tensor_sha256"), f"{label} target SHA")
        == expected_sha
        == _digest(value.get("published_runtime_tensor_sha256"), f"{label} runtime SHA")
    ):
        _fail(f"{label} did not publish trainer-authoritative tensors")
    return _digest(value.get("artifact_sha256"), f"{label} artifact SHA")


def validate_two_step_chain(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(payload, "two-step chain")
    _exact_fields(
        value,
        {"schema_version", "run_id", "production_slot", "step0", "step1", "stale_v1_after_v2"},
        "two-step chain",
    )
    if value.get("schema_version") != SCHEMA_VERSION:
        _fail("two-step schema version is not v6")
    _string(value.get("run_id"), "two-step run_id")
    if value.get("production_slot") != PRODUCTION_SLOT:
        _fail("two-step chain did not preserve the stable production slot")

    step0 = _mapping(value.get("step0"), "two-step step0")
    _exact_fields(
        step0,
        {
            "rollout_policy_version",
            "q_policy_version",
            "p_old_policy_version",
            "update_from_version",
            "update_to_version",
            "trainer_authority_v1_tensor_sha256",
            "refresh",
        },
        "two-step step0",
    )
    if any(step0.get(field) != 0 for field in ("rollout_policy_version", "q_policy_version", "p_old_policy_version", "update_from_version")):
        _fail("step0 rollout/q/p_old/update source must all be v0")
    if step0.get("update_to_version") != 1:
        _fail("step0 update target must be v1")
    v1_sha = _digest(
        step0.get("trainer_authority_v1_tensor_sha256"), "step0 v1 authority SHA"
    )
    refresh_v1_sha = _refresh_link(
        _mapping(step0.get("refresh"), "step0 refresh"),
        label="step0 refresh",
        from_version=0,
        to_version=1,
        expected_sha=v1_sha,
    )

    step1 = _mapping(value.get("step1"), "two-step step1")
    _exact_fields(step1, {"trajectory", "update"}, "two-step step1")
    trajectory = _mapping(step1.get("trajectory"), "step1 trajectory")
    _exact_fields(
        trajectory,
        {
            "generated_by_policy_version",
            "sampler_runtime_tensor_sha256",
            "trainer_authority_tensor_sha256",
            "p_old_actor_tensor_sha256",
            "refresh_artifact_sha256",
            "prompt_manifest_sha256",
            "seed",
            "q_policy_version",
            "p_old_policy_version",
        },
        "step1 trajectory",
    )
    if any(
        trajectory.get(field) != 1
        for field in ("generated_by_policy_version", "q_policy_version", "p_old_policy_version")
    ):
        _fail("step1 rollout/q/p_old provenance must all be v1")
    for field in (
        "sampler_runtime_tensor_sha256",
        "trainer_authority_tensor_sha256",
        "p_old_actor_tensor_sha256",
    ):
        if _digest(trajectory.get(field), f"step1 trajectory {field}") != v1_sha:
            _fail("step1 rollout runtime, trainer authority, and p_old actor must be v1")
    if _digest(trajectory.get("refresh_artifact_sha256"), "step1 refresh binding") != refresh_v1_sha:
        _fail("step1 trajectory is not bound to the v1 refresh artifact")
    _digest(trajectory.get("prompt_manifest_sha256"), "step1 prompt manifest SHA")
    _integer(trajectory.get("seed"), "step1 seed")

    update = _mapping(step1.get("update"), "step1 update")
    _exact_fields(
        update,
        {
            "from_version",
            "to_version",
            "input_actor_tensor_sha256",
            "trainer_authority_v2_tensor_sha256",
            "fresh_v2_tensor_sha256",
            "runtime_v2_tensor_sha256",
            "registry_count",
            "same_path_max_gap",
            "finite_rate",
            "delta_j",
            "delta_l",
            "alignment",
            "refresh",
        },
        "step1 update",
    )
    if update.get("from_version") != 1 or update.get("to_version") != 2:
        _fail("step1 update must transition v1->v2")
    if _digest(update.get("input_actor_tensor_sha256"), "step1 input actor SHA") != v1_sha:
        _fail("step1 optimizer input is not authoritative v1")
    v2_sha = _digest(
        update.get("trainer_authority_v2_tensor_sha256"), "step1 v2 authority SHA"
    )
    if v2_sha == v1_sha:
        _fail("v2 adapter tensors did not change from v1")
    for field in ("fresh_v2_tensor_sha256", "runtime_v2_tensor_sha256"):
        if _digest(update.get(field), f"step1 {field}") != v2_sha:
            _fail("v2 trainer/runtime/fresh tensor identity is inconsistent")
    if update.get("registry_count") != 1:
        _fail("stable adapter registry did not remain at one slot")
    if _number(update.get("same_path_max_gap"), "v2 same-path max gap") > SAME_PATH_MAX_GAP:
        _fail("v2 same-path fixed-action gap exceeds 1e-4")
    if _number(update.get("finite_rate"), "v2 finite rate") != 1.0:
        _fail("v2 fixed-action finite rate is not 100%")
    if _number(update.get("delta_j"), "step1 delta J") <= 0:
        _fail("step1 surrogate objective did not improve")
    if _number(update.get("delta_l"), "step1 delta L") >= 0:
        _fail("step1 loss did not decrease")
    if _number(update.get("alignment"), "step1 alignment") <= 0:
        _fail("step1 update alignment is not positive")
    _refresh_link(
        _mapping(update.get("refresh"), "step1 refresh"),
        label="step1 refresh",
        from_version=1,
        to_version=2,
        expected_sha=v2_sha,
    )

    stale = _mapping(value.get("stale_v1_after_v2"), "stale v1 evidence")
    _exact_fields(
        stale,
        {
            "logical_version",
            "accepted",
            "guard_stage",
            "error_code",
            "scoring_executed",
            "generation_executed",
        },
        "stale v1 evidence",
    )
    if not (
        stale.get("logical_version") == 1
        and stale.get("accepted") is False
        and stale.get("guard_stage") == "identity_guard_before_forward"
        and stale.get("error_code") == STALE_SAMPLER_IDENTITY
        and stale.get("scoring_executed") is False
        and stale.get("generation_executed") is False
    ):
        _fail("stale v1 request was not rejected before forward after v2")
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "rollout_step1_tensor_sha256": v1_sha,
        "published_v2_tensor_sha256": v2_sha,
    }


_BASE_NULL_FIELDS = {
    "schema_version",
    "run_id",
    "route_is_independent",
    "teacher_is_real_base",
    "old_actor_is_same_base_detached",
    "current_actor_is_base_equivalent_zero_lora",
    "fresh_optimizer",
    "medical_optimizer_state_reused",
    "label_access",
    "final_access",
    "controller_access",
    "current_pre_base_max_abs_gap",
    "advantage_max_abs",
    "objective",
    "loss",
    "gradient_norm_before_clip",
    "gradient_norm_after_clip",
    "parameter_delta_norm",
    "nonzero_update_tensor_count",
    "adapter_tensor_sha256_before",
    "adapter_tensor_sha256_after",
    "base_gradient_tensor_count",
    "teacher_gradient_tensor_count",
    "finite_rate",
}


def validate_base_null(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(payload, "Base null")
    _exact_fields(value, _BASE_NULL_FIELDS, "Base null")
    if value.get("schema_version") != SCHEMA_VERSION:
        _fail("Base null schema version is not v6")
    _string(value.get("run_id"), "Base null run_id")
    for field in (
        "route_is_independent",
        "teacher_is_real_base",
        "old_actor_is_same_base_detached",
        "current_actor_is_base_equivalent_zero_lora",
        "fresh_optimizer",
    ):
        _boolean(value.get(field), True, f"Base null {field}")
    for field in (
        "medical_optimizer_state_reused",
        "label_access",
        "final_access",
        "controller_access",
    ):
        _boolean(value.get(field), False, f"Base null {field}")
    gap = _number(value.get("current_pre_base_max_abs_gap"), "Base null current/Base gap")
    if gap < 0 or gap > SAME_PATH_MAX_GAP:
        _fail("Base null current actor is not Base-equivalent within 1e-4")
    advantage = _number(value.get("advantage_max_abs"), "Base null advantage max abs")
    if advantage < 0 or advantage > NULL_ADVANTAGE_TOLERANCE:
        _fail("Base null advantage is non-zero")
    for field in (
        "objective",
        "loss",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "parameter_delta_norm",
    ):
        if _number(value.get(field), f"Base null {field}") != 0.0:
            _fail(f"Base null {field} is non-zero")
    for field in (
        "nonzero_update_tensor_count",
        "base_gradient_tensor_count",
        "teacher_gradient_tensor_count",
    ):
        if _integer(value.get(field), f"Base null {field}") != 0:
            _fail(f"Base null {field} is non-zero")
    before = _digest(value.get("adapter_tensor_sha256_before"), "Base null before SHA")
    after = _digest(value.get("adapter_tensor_sha256_after"), "Base null after SHA")
    if before != after:
        _fail("Base null adapter tensor SHA changed")
    if _number(value.get("finite_rate"), "Base null finite rate") != 1.0:
        _fail("Base null finite rate is not 100%")
    return {"schema_version": SCHEMA_VERSION, "passed": True, "adapter_tensor_sha256": before}


_LENGTH_TOP_FIELDS = {
    "schema_version",
    "run_id",
    "prompt_manifest_sha256",
    "prompt_count",
    "source_counts",
    "thresholds",
    "candidates",
}
_CANDIDATE_FIELDS = {
    "max_new_tokens",
    "measurement",
    "source_actual_max_new_tokens",
    "finite",
    "invalid_empty_count",
    "thinking_tag_count",
    "oom",
    "cost_gate_passed",
    "disk_gate_passed",
    "sources",
    "overall",
}


def _evaluate_length_candidate(
    raw: Mapping[str, Any], *, length: int, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    value = _mapping(raw, f"length candidate {length}")
    _exact_fields(value, _CANDIDATE_FIELDS, f"length candidate {length}")
    if value.get("max_new_tokens") != length:
        _fail(f"length candidate {length} has the wrong max_new_tokens")
    expected_measurement = "derived_prefix_from_actual_384" if length == 256 else "actual_generation"
    if value.get("measurement") != expected_measurement:
        _fail(f"length candidate {length} has invalid provenance")
    expected_actual = 384 if length == 256 else length
    if value.get("source_actual_max_new_tokens") != expected_actual:
        _fail(f"length candidate {length} has invalid source trajectory provenance")
    if not isinstance(value.get("finite"), bool):
        _fail(f"length candidate {length} finite is not boolean")
    for field in ("oom", "cost_gate_passed", "disk_gate_passed"):
        if not isinstance(value.get(field), bool):
            _fail(f"length candidate {length} {field} is not boolean")
    invalid = _integer(value.get("invalid_empty_count"), f"length {length} invalid/empty")
    thinking = _integer(value.get("thinking_tag_count"), f"length {length} thinking tags")

    sources = _mapping(value.get("sources"), f"length {length} sources")
    if set(sources) != {"medical_o1", "cmb"}:
        _fail(f"length candidate {length} must contain Medical-O1 and CMB")
    rates: dict[str, float] = {}
    truncation_total = 0
    count_total = 0
    source_pass = True
    for source_name in ("medical_o1", "cmb"):
        source = _mapping(sources[source_name], f"length {length} {source_name}")
        _exact_fields(source, {"count", "truncation_count"}, f"length {length} {source_name}")
        count = _integer(source.get("count"), f"length {length} {source_name} count", minimum=1)
        truncated = _integer(
            source.get("truncation_count"), f"length {length} {source_name} truncation"
        )
        if count != 8 or truncated > count:
            _fail(f"length {length} {source_name} count is inconsistent with frozen 8/source")
        rate = truncated / count
        rates[source_name] = rate
        truncation_total += truncated
        count_total += count
        source_pass = source_pass and rate <= thresholds["per_source_truncation_rate_max"]
    overall = _mapping(value.get("overall"), f"length {length} overall")
    _exact_fields(overall, {"count", "truncation_count"}, f"length {length} overall")
    if overall.get("count") != count_total or overall.get("truncation_count") != truncation_total:
        _fail(f"length candidate {length} overall counts do not add up")
    overall_rate = truncation_total / count_total
    reasons: list[str] = []
    if not value["finite"]:
        reasons.append("nonfinite")
    if overall_rate > thresholds["overall_truncation_rate_max"]:
        reasons.append("overall_truncation")
    if not source_pass:
        reasons.append("per_source_truncation")
    if invalid > thresholds["invalid_empty_count_max"]:
        reasons.append("invalid_or_empty")
    if thinking != 0:
        reasons.append("thinking_tag")
    if value["oom"]:
        reasons.append("oom")
    if not value["cost_gate_passed"]:
        reasons.append("cost_gate")
    if not value["disk_gate_passed"]:
        reasons.append("disk_gate")
    return {
        "max_new_tokens": length,
        "passed": not reasons,
        "reasons": reasons,
        "overall_truncation_rate": overall_rate,
        "per_source_truncation_rate": rates,
    }


def decide_response_length(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Select only the shortest frozen candidate; request 512 conditionally."""

    value = _mapping(payload, "length smoke")
    _exact_fields(value, _LENGTH_TOP_FIELDS, "length smoke")
    if value.get("schema_version") != SCHEMA_VERSION:
        _fail("length smoke schema version is not v6")
    run_id = _string(value.get("run_id"), "length smoke run_id")
    prompt_sha = _digest(value.get("prompt_manifest_sha256"), "length prompt manifest SHA")
    if value.get("prompt_count") != 16 or value.get("source_counts") != {
        "medical_o1": 8,
        "cmb": 8,
    }:
        _fail("length smoke must bind exactly 8 Medical-O1 and 8 CMB prompts")
    thresholds = _mapping(value.get("thresholds"), "length thresholds")
    _exact_fields(
        thresholds,
        {
            "overall_truncation_rate_max",
            "per_source_truncation_rate_max",
            "invalid_empty_count_max",
        },
        "length thresholds",
    )
    for field in ("overall_truncation_rate_max", "per_source_truncation_rate_max"):
        if _number(thresholds.get(field), f"length threshold {field}") != 0.20:
            _fail(f"length threshold {field} is not frozen at 0.20")
    _integer(thresholds.get("invalid_empty_count_max"), "invalid/empty threshold")

    candidates = _mapping(value.get("candidates"), "length candidates")
    keys = set(candidates)
    if not {"256", "384"}.issubset(keys) or not keys.issubset({"256", "384", "512"}):
        _fail("length smoke must contain 256/384 and only conditional 512")
    evaluations = {
        "256": _evaluate_length_candidate(candidates["256"], length=256, thresholds=thresholds),
        "384": _evaluate_length_candidate(candidates["384"], length=384, thresholds=thresholds),
    }
    if "512" in candidates:
        if evaluations["384"]["passed"]:
            _fail("512 was run even though actual 384 already passed")
        evaluations["512"] = _evaluate_length_candidate(
            candidates["512"], length=512, thresholds=thresholds
        )

    selected: int | None = None
    status: str
    requires_512 = False
    if evaluations["256"]["passed"]:
        selected = 256
        status = "length_frozen"
    elif evaluations["384"]["passed"]:
        selected = 384
        status = "length_frozen"
    elif "512" not in evaluations:
        status = "requires_actual_512"
        requires_512 = True
    elif evaluations["512"]["passed"]:
        selected = 512
        status = "length_frozen"
    else:
        status = "length_not_frozen"

    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "production_length_decision_v6",
        "run_id": run_id,
        "prompt_manifest_sha256": prompt_sha,
        "status": status,
        "selected_response_length": selected,
        "requires_512": requires_512,
        "evaluations": evaluations,
        "thresholds": deepcopy(dict(thresholds)),
    }
    _json_primitives(result)
    return result
