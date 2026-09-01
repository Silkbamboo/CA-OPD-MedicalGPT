"""CPU-safe orchestration for the P4.6 null, length, and B2 GPU routes.

This module owns the *control contracts* around three expensive operations.  It
does not import torch, Transformers, or PEFT at module import time.  A real GPU
session is constructed lazily (or injected by the combined qualification
runtime), while all prompt selection and returned evidence are validated here.

Only prompt-only OPD sources are accepted.  No response text, prompt text,
token sequence, label, controller, confirmation, or final payload is returned
from these helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping, Protocol, Sequence


PRODUCTION_BACKEND_ID = "custom_transformers_peft_three_policy_v5"
PRODUCTION_SLOT = "student_active"
B2_SELECTION_RULE = "seed42_sha256_rank_first2_per_source_per_step_v1"
P4_7_B2_ALLOWED_RESPONSE_LENGTHS = frozenset(
    {256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096}
)
LENGTH_SOURCES = ("medical_opd_o1", "medical_opd_cmb")
SOURCE_LABELS = {
    "medical_opd_o1": "medical_o1",
    "medical_opd_cmb": "cmb",
}


class ProductionQualificationAuxGPUV6Error(RuntimeError):
    """An auxiliary qualification or B2 gate failed closed."""


class AuxiliarySession(Protocol):
    """Narrow capabilities supplied by the real stable-slot GPU session."""

    def run_base_teacher_null(
        self, *, prompt_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def generate_length_trajectories(
        self,
        *,
        prompt_rows: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> Sequence[Mapping[str, Any]]: ...

    def current_policy_identity(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


SessionFactory = Callable[[Mapping[str, Any], Path], AuxiliarySession]
StepKernel = Callable[..., Mapping[str, Any]]
AuthorizationGate = Callable[[Path], Mapping[str, Any]]


def _fail(message: str) -> None:
    raise ProductionQualificationAuxGPUV6Error(message)


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_isolation(config: Mapping[str, Any], *, allow_b2: bool) -> None:
    isolation = config.get("isolation")
    if not isinstance(isolation, Mapping) or any(
        isolation.get(field) is not False
        for field in ("final_access", "controller_access", "confirmation_access", "label_access")
    ):
        _fail("final/controller/confirmation/label isolation is not fail-closed")
    execution = config.get("execution", {})
    if isinstance(execution, Mapping):
        for field in (
            "automatically_run_idt",
            "automatically_run_sar",
            "automatically_run_ca_opd",
            "automatically_run_controller",
            "automatically_run_confirmation",
            "automatically_run_final",
        ):
            if execution.get(field, False) is not False:
                _fail(f"forbidden downstream execution enabled: {field}")
        if not allow_b2 and execution.get("automatically_start_b2", False) is not False:
            _fail("qualification must not start B2")
    binding = config.get("production_binding", config.get("production_backend", {}))
    backend_id = binding.get("backend_id") if isinstance(binding, Mapping) else None
    if backend_id != PRODUCTION_BACKEND_ID:
        _fail("auxiliary GPU route is not bound to the frozen production backend")


def _frozen_prompt_group(
    config: Mapping[str, Any],
    *,
    group_id: str,
    config_path: Path,
) -> list[dict[str, Any]]:
    """Read exactly the checked-in, pre-run prompt identity selection."""

    from src.opd.production_qualification_prompts_v6 import load_frozen_prompt_group

    root = Path(__file__).resolve().parents[2]
    try:
        rows = load_frozen_prompt_group(config, group_id, repo_root=root)
    except Exception as error:
        raise ProductionQualificationAuxGPUV6Error(
            f"frozen {group_id} prompt selection failed: {error}"
        ) from error
    expected = {"base_null": 4, "length": 16}.get(group_id)
    if expected is None or len(rows) != expected:
        _fail(f"frozen {group_id} prompt group has the wrong count")
    source_counts = {
        source: sum(row.get("target_role") == source for row in rows)
        for source in LENGTH_SOURCES
    }
    if source_counts != {source: expected // 2 for source in LENGTH_SOURCES}:
        _fail(f"frozen {group_id} prompt group source balance drift")
    for row in rows:
        if not (
            isinstance(row.get("sample_id"), str)
            and isinstance(row.get("content_hash"), str)
            and len(row["content_hash"]) == 64
            and isinstance(row.get("question"), str)
            and bool(row["question"].strip())
        ):
            _fail(f"frozen {group_id} prompt row is incomplete")
    return [dict(row) for row in rows]


def _default_session_factory(
    config: Mapping[str, Any], config_path: Path
) -> AuxiliarySession:
    # This import remains inside the explicitly authorized execution path.
    from src.opd.production_qualification_two_step_gpu_v6 import (
        create_production_two_step_session_v6,
    )

    return create_production_two_step_session_v6(config, config_path=config_path)


def _default_b2_session_factory(
    config: Mapping[str, Any], config_path: Path
) -> AuxiliarySession:
    """Restore exactly the v2 checkpoint authorized by full qualification."""

    qualification = config.get("qualification")
    if not isinstance(qualification, Mapping):
        _fail("B2 qualification checkpoint binding is absent")
    checkpoint = Path(str(qualification.get("v2_checkpoint_path", "")))
    if not checkpoint.is_absolute():
        checkpoint = (Path(config_path).resolve().parent / checkpoint).resolve()
    expected = _digest(
        qualification.get("v2_tensor_sha256"), "B2 qualification v2 tensor"
    )
    from src.opd.production_qualification_two_step_gpu_v6 import (
        create_production_b2_session_v6,
    )

    return create_production_b2_session_v6(
        config,
        config_path=Path(config_path),
        checkpoint_v2=checkpoint,
        expected_v2_sha256=expected,
    )


def _default_b2_authorization_gate(
    path: Path,
    *,
    config: Mapping[str, Any] | None = None,
    allow_b2_calibration: bool = False,
    authority_revalidator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Reopen the immutable full graph before any B2 output or model load."""

    try:
        authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionQualificationAuxGPUV6Error(
            f"B2 start authorization rejected: {error}"
        ) from error
    p4_7_config = isinstance(config, Mapping) and "p4_7_start_gate" in config
    p4_7_authorization = isinstance(authorization, Mapping) and authorization.get(
        "artifact_kind"
    ) == "p4_7_b2_20_step_calibration_authorization"
    if p4_7_config and not p4_7_authorization:
        _fail("P4.7 B2 authorization artifact kind mismatch")
    if p4_7_authorization:
        if allow_b2_calibration is not True:
            _fail("P4.7 B2 start requires explicit --allow-b2-calibration")
        if not isinstance(config, Mapping):
            _fail("P4.7 B2 start requires the materialized production config")
        gate = config.get("p4_7_start_gate")
        qualification = config.get("qualification")
        if not isinstance(gate, Mapping) or not isinstance(qualification, Mapping):
            _fail("P4.7 B2 start gate binding is absent")
        root = Path(str(gate.get("formal_run_root", ""))).resolve()
        if root != Path(str(qualification.get("output_path", ""))).resolve():
            _fail("P4.7 B2 formal root binding drift")
        source_path = Path(str(gate.get("qualification_config_path", "")))
        repo_root = Path(__file__).resolve().parents[2]
        if not source_path.is_absolute():
            source_path = (repo_root / source_path).resolve()
        try:
            import yaml

            source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ProductionQualificationAuxGPUV6Error(
                f"B2 start authorization rejected: canonical P4.7 config: {error}"
            ) from error
        if not isinstance(source_config, Mapping):
            _fail("P4.7 source qualification config is invalid")
        revalidator = authority_revalidator
        if revalidator is None:
            from src.opd.production_length_preflight_v7 import (
                reverify_finalization_authority,
            )

            revalidator = reverify_finalization_authority
        try:
            authority = dict(revalidator(source_config))
            from src.opd.production_length_gpu_runtime_v7 import (
                assert_b2_calibration_start_authorized,
            )

            verified = assert_b2_calibration_start_authorized(
                authorization,
                allow_b2_calibration=allow_b2_calibration,
                formal_run_root=root,
                package_dir=Path(path).resolve().parent,
                authority_revalidation=authority,
            )
        except Exception as error:
            raise ProductionQualificationAuxGPUV6Error(
                f"B2 start authorization rejected: {error}"
            ) from error
        bindings = verified.get("bindings")
        if not isinstance(bindings, Mapping):
            _fail("P4.7 B2 authorization bindings are absent")
        return {
            "B2_authorized": verified.get("B2_authorized"),
            "B2_started": verified.get("B2_started"),
            "production_backend_id": bindings.get("backend_id"),
            "optimizer_steps": bindings.get("estimated_steps"),
            "selected_response_length": verified.get("selected_response_length"),
        }

    from src.opd.production_qualification_artifacts_v6 import (
        assert_b2_start_authorized,
    )

    try:
        value = assert_b2_start_authorized(path)
    except Exception as error:
        raise ProductionQualificationAuxGPUV6Error(
            f"B2 start authorization rejected: {error}"
        ) from error
    if not isinstance(value, Mapping):
        _fail("B2 start authorization evidence is absent")
    return value


def _acquire_session(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    session: AuxiliarySession | None,
    session_factory: SessionFactory | None,
) -> tuple[AuxiliarySession, bool]:
    if session is not None and session_factory is not None:
        _fail("provide either an existing session or a session factory, not both")
    if session is not None:
        return session, False
    factory = session_factory or _default_session_factory
    return factory(config, config_path), True


def create_production_auxiliary_session_v6(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    route: str,
    checkpoint_v2: Path | None = None,
    expected_v2_sha256: str | None = None,
) -> AuxiliarySession:
    """Lazily resolve the real source-bound auxiliary GPU session."""

    from src.opd.production_qualification_two_step_gpu_v6 import (
        create_production_auxiliary_session_v6 as create_real_session,
    )

    return create_real_session(
        config,
        config_path=Path(config_path),
        route=route,
        checkpoint_v2=checkpoint_v2,
        expected_v2_sha256=expected_v2_sha256,
    )


def execute_base_teacher_null_v6(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    session: AuxiliarySession | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Run an independent, fresh-optimizer Base=Teacher null route.

    The GPU session performs the real forward/backward/optimizer operations;
    this boundary supplies source-real prompt-only rows and rejects any result
    that does not satisfy the frozen semantic contract.
    """

    from src.opd.production_qualification_contract_v6 import validate_base_null

    _validate_isolation(config, allow_b2=False)
    null_config = config.get("base_null")
    if not isinstance(null_config, Mapping) or not (
        null_config.get("independent_route") is True
        and null_config.get("independent_fresh_optimizer") is True
        and null_config.get("reuse_medical_optimizer_state") is False
        and null_config.get("current_pre_base_max_gap") == 0.0001
        and null_config.get("advantage_max_abs_tolerance") == 1e-8
        and null_config.get("objective_tolerance") == 0.0
        and null_config.get("loss_tolerance") == 0.0
        and null_config.get("gradient_norm_tolerance") == 0.0
        and null_config.get("parameter_delta_tolerance") == 0.0
    ):
        _fail("Base=Teacher null contract drift")
    rows = _frozen_prompt_group(
        config, group_id="base_null", config_path=Path(config_path)
    )
    default_factory: SessionFactory = lambda value, path: create_production_auxiliary_session_v6(
        value, config_path=path, route="base_null"
    )
    selected_factory = (
        session_factory
        if session_factory is not None
        else (default_factory if session is None else None)
    )
    acquired, owned = _acquire_session(
        config,
        Path(config_path),
        session=session,
        session_factory=selected_factory,
    )
    try:
        raw = acquired.run_base_teacher_null(prompt_rows=rows, config=config)
        if not isinstance(raw, Mapping):
            _fail("Base=Teacher null session returned no evidence")
        payload = dict(raw)
        try:
            validate_base_null(payload)
        except Exception as error:
            raise ProductionQualificationAuxGPUV6Error(
                f"Base=Teacher null evidence failed: {error}"
            ) from error
        return payload
    finally:
        if owned:
            acquired.close()


def base_null_artifact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the semantic null contract to the v6 artifact phase view."""

    from src.opd.production_qualification_contract_v6 import validate_base_null

    validate_base_null(payload)
    return {
        "status": "pass",
        "independent_route": payload["route_is_independent"],
        "fresh_optimizer": payload["fresh_optimizer"],
        "teacher_is_base": payload["teacher_is_real_base"],
        "old_actor_base_detached": payload["old_actor_is_same_base_detached"],
        "current_actor_zero_lora": payload["current_actor_is_base_equivalent_zero_lora"],
        "current_pre_base_max_gap": payload["current_pre_base_max_abs_gap"],
        "advantage_max_abs": payload["advantage_max_abs"],
        "objective": payload["objective"],
        "loss": payload["loss"],
        "gradient_norm": payload["gradient_norm_before_clip"],
        "parameter_delta": payload["parameter_delta_norm"],
        "nonzero_update_tensor_count": payload["nonzero_update_tensor_count"],
        "adapter_sha256_before": payload["adapter_tensor_sha256_before"],
        "adapter_sha256_after": payload["adapter_tensor_sha256_after"],
        "teacher_gradient_tensor_count": payload["teacher_gradient_tensor_count"],
        "base_gradient_tensor_count": payload["base_gradient_tensor_count"],
        "finite_rate": payload["finite_rate"],
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        _fail("length percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _normalize_length_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
        _fail(f"actual {max_new_tokens} generation returned no record sequence")
    expected = {row["sample_id"]: row["target_role"] for row in rows}
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            _fail("length generation record is not an object")
        sample_id = raw.get("sample_id")
        source = raw.get("source_role")
        if sample_id not in expected or source != expected.get(sample_id):
            _fail("length generation record does not bind a frozen prompt")
        token_count = raw.get("token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            _fail("length token_count must be an integer")
        if token_count < 0 or token_count > max_new_tokens:
            _fail("length token_count is outside the generation bound")
        eos_position = raw.get("eos_position")
        if eos_position is not None and (
            isinstance(eos_position, bool)
            or not isinstance(eos_position, int)
            or eos_position < 1
            or eos_position > token_count
        ):
            _fail("length eos_position must be one-based within generated tokens")
        finite = raw.get("finite")
        invalid = raw.get("invalid_or_empty")
        thinking = raw.get("thinking_tag_count")
        if not isinstance(finite, bool) or not isinstance(invalid, bool):
            _fail("length finite/invalid evidence must be boolean")
        if isinstance(thinking, bool) or not isinstance(thinking, int) or thinking < 0:
            _fail("length thinking tag count is invalid")
        finish_reason = raw.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            _fail("length finish reason is absent")
        tokens_per_second = _finite(raw.get("tokens_per_second"), "tokens/s")
        wall_time = _finite(raw.get("wall_time_seconds"), "length wall time")
        peak_memory = raw.get("gpu_peak_memory_bytes")
        if (
            tokens_per_second < 0
            or wall_time < 0
            or isinstance(peak_memory, bool)
            or not isinstance(peak_memory, int)
            or peak_memory < 0
        ):
            _fail("length resource telemetry is invalid")
        records.append(
            {
                "sample_id": sample_id,
                "source_role": source,
                "token_count": token_count,
                "eos_position": eos_position,
                "finite": finite,
                "invalid_or_empty": invalid or token_count == 0,
                "thinking_tag_count": thinking,
                "finish_reason": finish_reason,
                "tokens_per_second": tokens_per_second,
                "wall_time_seconds": wall_time,
                "gpu_peak_memory_bytes": peak_memory,
            }
        )
    if len(records) != 16 or len({record["sample_id"] for record in records}) != 16:
        _fail("length generation must return exactly one record for each of 16 prompts")
    if set(expected) != {record["sample_id"] for record in records}:
        _fail("length generation omitted a frozen prompt")
    return records


def _candidate_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_length: int,
    actual_length: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    derived = candidate_length != actual_length
    if derived and not (candidate_length == 256 and actual_length == 384):
        _fail("only the frozen 256 prefix may be derived from actual 384")
    per_source: dict[str, dict[str, Any]] = {}
    contract_sources: dict[str, dict[str, int]] = {}
    lengths: list[int] = []
    eos_count = 0
    truncation_count = 0
    invalid_count = 0
    thinking_count = 0
    finite_count = 0
    finish_reasons: dict[str, int] = {}
    for source in LENGTH_SOURCES:
        source_records = [record for record in records if record["source_role"] == source]
        if len(source_records) != 8:
            _fail("length calibration did not retain exactly 8 prompts per source")
        source_truncated = 0
        for record in source_records:
            eos_position = record["eos_position"]
            completed = eos_position is not None and eos_position <= candidate_length
            observed_length = int(eos_position) if completed else min(
                int(record["token_count"]), candidate_length
            )
            lengths.append(observed_length)
            eos_count += int(completed)
            source_truncated += int(not completed)
            invalid_count += int(record["invalid_or_empty"])
            thinking_count += int(record["thinking_tag_count"])
            finite_count += int(record["finite"])
            finish = "eos" if completed else "length"
            finish_reasons[finish] = finish_reasons.get(finish, 0) + 1
        truncation_count += source_truncated
        per_source[source] = {
            "count": 8,
            "truncation_count": source_truncated,
            "truncation_rate": source_truncated / 8,
        }
        contract_sources[SOURCE_LABELS[source]] = {
            "count": 8,
            "truncation_count": source_truncated,
        }
    finite_rate = finite_count / 16
    artifact = {
        "count": 16,
        "finite_rate": finite_rate,
        "invalid_empty_count": invalid_count,
        "thinking_tag_count": thinking_count,
        "truncation_count": truncation_count,
        "truncation_rate": truncation_count / 16,
        "per_source": per_source,
    }
    contract = {
        "max_new_tokens": candidate_length,
        "measurement": "derived_prefix_from_actual_384" if derived else "actual_generation",
        "source_actual_max_new_tokens": actual_length,
        "finite": finite_rate == 1.0,
        "invalid_empty_count": invalid_count,
        "thinking_tag_count": thinking_count,
        "oom": False,
        "cost_gate_passed": True,
        "disk_gate_passed": True,
        "sources": contract_sources,
        "overall": {"count": 16, "truncation_count": truncation_count},
    }
    telemetry = {
        "max_new_tokens": candidate_length,
        "measurement": contract["measurement"],
        "eos_count": eos_count,
        "eos_rate": eos_count / 16,
        "truncation_count": truncation_count,
        "truncation_rate": truncation_count / 16,
        "length": {
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
        },
        "finish_reason_counts": finish_reasons,
        "invalid_empty_count": invalid_count,
        "thinking_tag_count": thinking_count,
        "finite_rate": finite_rate,
        "tokens_per_second": statistics.fmean(
            float(record["tokens_per_second"]) for record in records
        ),
        "wall_time_seconds": sum(float(record["wall_time_seconds"]) for record in records),
        "gpu_peak_memory_bytes": max(int(record["gpu_peak_memory_bytes"]) for record in records),
        "per_source": per_source,
    }
    return artifact, contract, telemetry


def _artifact_candidate_passes(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate["finite_rate"] == 1.0
        and candidate["invalid_empty_count"] == 0
        and candidate["thinking_tag_count"] == 0
        and candidate["truncation_rate"] <= 0.20
        and all(
            metrics["truncation_count"] <= 1
            for metrics in candidate["per_source"].values()
        )
    )


def execute_length_calibration_v6(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    checkpoint_v2: str | Path,
    v2_tensor_sha256: str,
    authority_v2_artifact_sha256: str,
    session: AuxiliarySession | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Run actual 384 once, derive 256, and run actual 512 only if needed."""

    from src.opd.production_qualification_contract_v6 import decide_response_length

    _validate_isolation(config, allow_b2=False)
    expected_v2 = _digest(v2_tensor_sha256, "v2 trainer authority")
    authority_v2_sha = _digest(
        authority_v2_artifact_sha256, "authority_v2 artifact"
    )
    checkpoint = Path(checkpoint_v2)
    if not checkpoint.is_absolute():
        checkpoint = Path(str(config.get("run", {}).get("output_dir", ""))) / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists() or checkpoint.is_symlink():
        _fail("immutable v2 checkpoint is absent")
    length = config.get("length_smoke")
    if not isinstance(length, Mapping) or not (
        length.get("prompt_count") == 16
        and length.get("prompts_per_source") == 8
        and length.get("actual_initial_max_new_tokens") == 384
        and length.get("derive_256_from_384") is True
        and length.get("conditional_512_only") is True
        and length.get("overall_truncation_rate_max") == 0.20
        and length.get("per_source_truncation_rate_max") == 0.20
        and length.get("invalid_empty_count_max") == 0
        and length.get("thinking_tag_count_max") == 0
        and length.get("full_support_generation") is True
        and length.get("enable_thinking") is False
        and length.get("capability_evaluation") is False
    ):
        _fail("length calibration contract drift")
    rows = _frozen_prompt_group(
        config, group_id="length", config_path=Path(config_path)
    )
    default_factory: SessionFactory = lambda value, path: create_production_auxiliary_session_v6(
        value,
        config_path=path,
        route="length",
        checkpoint_v2=checkpoint,
        expected_v2_sha256=expected_v2,
    )
    selected_factory = (
        session_factory
        if session_factory is not None
        else (default_factory if session is None else None)
    )
    acquired, owned = _acquire_session(
        config,
        Path(config_path),
        session=session,
        session_factory=selected_factory,
    )
    try:
        identity = acquired.current_policy_identity()
        if not isinstance(identity, Mapping) or not (
            identity.get("logical_version") in (2, "v2")
            and identity.get("tensor_sha256") == expected_v2
            and identity.get("active_slot") == PRODUCTION_SLOT
            and identity.get("registry_count") == 1
            and identity.get("checkpoint_path") == str(checkpoint)
        ):
            _fail("length session is not the trainer-authoritative stable-slot v2")
        actual_384_records = _normalize_length_records(
            acquired.generate_length_trajectories(
                prompt_rows=rows, max_new_tokens=384, enable_thinking=False
            ),
            rows=rows,
            max_new_tokens=384,
        )
        derived_256, contract_256, telemetry_256 = _candidate_summary(
            actual_384_records, candidate_length=256, actual_length=384
        )
        actual_384, contract_384, telemetry_384 = _candidate_summary(
            actual_384_records, candidate_length=384, actual_length=384
        )
        artifact_payload: dict[str, Any] = {
            "status": "pass",
            "actual_lengths": [384],
            "conditional_512_executed": False,
            "actual_512_executed": False,
            "derived_256": derived_256,
            "actual_384": actual_384,
        }
        contract_candidates: dict[str, Any] = {
            "256": contract_256,
            "384": contract_384,
        }
        telemetry: dict[str, Any] = {
            "selection_rule": "checked_in_p4_6_prompt_selection_manifest_v1",
            "prompt_count": 16,
            "source_counts": {source: 8 for source in LENGTH_SOURCES},
            "actual_384": telemetry_384,
            "derived_256": telemetry_256,
        }
        if not _artifact_candidate_passes(actual_384):
            actual_512_records = _normalize_length_records(
                acquired.generate_length_trajectories(
                    prompt_rows=rows, max_new_tokens=512, enable_thinking=False
                ),
                rows=rows,
                max_new_tokens=512,
            )
            actual_512, contract_512, telemetry_512 = _candidate_summary(
                actual_512_records, candidate_length=512, actual_length=512
            )
            artifact_payload.update(
                {
                    "actual_lengths": [384, 512],
                    "conditional_512_executed": True,
                    "actual_512_executed": True,
                    "actual_512": actual_512,
                }
            )
            contract_candidates["512"] = contract_512
            telemetry["actual_512"] = telemetry_512
        smoke = {
            "schema_version": 6,
            "run_id": config["run"]["run_id"],
            "prompt_manifest_sha256": config["prompt_selection"]["opd_manifest_sha256"],
            "prompt_count": 16,
            "source_counts": {"medical_o1": 8, "cmb": 8},
            "thresholds": {
                "overall_truncation_rate_max": 0.20,
                "per_source_truncation_rate_max": 0.20,
                "invalid_empty_count_max": 0,
            },
            "candidates": contract_candidates,
        }
        decision = decide_response_length(smoke)
        selected = decision["selected_response_length"]
        evaluated = [256, 384] + ([512] if "512" in contract_candidates else [])
        decision_artifact = {
            "status": "pass" if selected is not None else "fail",
            "selected_response_length": selected,
            "evaluated_candidates": evaluated,
            "decision_rule": "shortest_passing_overall_and_per_source_truncation_v1",
        }
        telemetry["prompt_identity_sha256"] = _canonical_sha(
            [
                {
                    "sample_id": row["sample_id"],
                    "source_role": row["target_role"],
                }
                for row in rows
            ]
        )
        telemetry["selected_response_length"] = selected
        artifact_payload["telemetry"] = telemetry
        artifact_payload["policy_identity"] = {
            "logical_version": "v2",
            "tensor_sha256": identity["tensor_sha256"],
            "checkpoint_path": identity["checkpoint_path"],
            "authority_v2_artifact_sha256": authority_v2_sha,
            "active_slot": identity["active_slot"],
            "registry_count": identity["registry_count"],
        }
        return {
            "length_smoke": smoke,
            "length_decision": decision,
            "artifact_payload": artifact_payload,
            "decision_artifact_payload": decision_artifact,
            "telemetry": telemetry,
            "actual_lengths": artifact_payload["actual_lengths"],
        }
    finally:
        if owned:
            acquired.close()


def _source_real_b2_prompt_batch(
    config: Mapping[str, Any], step_index: int
) -> list[dict[str, Any]]:
    """Resolve real prompt-only OPD rows and take the frozen step window."""

    from src.opd.calibration_data import contains_forbidden_supervision
    from src.opd.production import resolve_opd_source_files

    data = config.get("data")
    if not isinstance(data, Mapping) or not (
        data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
        and data.get("allowed_roles") == list(LENGTH_SOURCES)
        and data.get("selection_rule") == B2_SELECTION_RULE
    ):
        _fail("B2 prompt-only data/selection contract is not frozen")
    schedule_path_value = data.get("schedule_path")
    if schedule_path_value is not None:
        # P4.8b and later use one exact CPU-safe authority/schedule resolver in
        # dry-run, formal preflight, and the production provider.  This branch
        # runs before any generation call and never silently falls back to the
        # legacy dynamic selection path.
        from src.opd.production_b2_data_v2 import (
            B2DataAuthorityV2Error,
            CANONICAL_MANIFEST_PATH,
            resolve_b2_data_authority,
            resolve_b2_schedule_batch,
        )

        schedule_path = Path(str(schedule_path_value))
        if not schedule_path.is_absolute():
            schedule_path = Path(__file__).resolve().parents[2] / schedule_path
        if schedule_path.is_symlink() or not schedule_path.is_file():
            _fail("B2 frozen prompt schedule is absent or a symlink")
        try:
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProductionQualificationAuxGPUV6Error(
                f"B2 frozen prompt schedule is invalid: {type(error).__name__}"
            ) from error
        if not isinstance(schedule, Mapping):
            _fail("B2 frozen prompt schedule is not an object")
        manifest = Path(str(data.get("prompt_manifest_path", "")))
        if not manifest.is_absolute():
            manifest = Path(__file__).resolve().parents[2] / manifest
        try:
            authority = resolve_b2_data_authority(
                manifest,
                expected_manifest_sha256=str(data.get("prompt_manifest_sha256", "")),
                canonical_manifest_path=CANONICAL_MANIFEST_PATH,
            )
            if not (
                data.get("schedule_sha256") == schedule.get("schedule_sha256")
                and data.get("schedule_version") == schedule.get("schedule_version")
                and data.get("provider")
                == "production_b2_data_v2.resolve_b2_schedule_batch"
                and data.get("canonical_manifest_required") is True
            ):
                _fail("B2 package/schedule/provider binding differs")
            return resolve_b2_schedule_batch(
                authority, schedule, step_index=step_index
            )
        except B2DataAuthorityV2Error as error:
            raise ProductionQualificationAuxGPUV6Error(
                f"B2 frozen provider semantic gate failed: {error}"
            ) from error
    manifest = Path(str(data.get("prompt_manifest_path", "")))
    if not manifest.is_absolute():
        manifest = Path(__file__).resolve().parents[2] / manifest
    expected_manifest_sha = data.get("prompt_manifest_sha256")
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or not isinstance(expected_manifest_sha, str)
        or len(expected_manifest_sha) != 64
        or hashlib.sha256(manifest.read_bytes()).hexdigest()
        != expected_manifest_sha
    ):
        _fail("B2 prompt manifest SHA mismatch")
    paths = resolve_opd_source_files(manifest)
    by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in LENGTH_SOURCES}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    _fail("B2 prompt source row is not an object")
                source = raw.get("target_role")
                if contains_forbidden_supervision(raw):
                    _fail("B2 source contains forbidden supervision")
                if not isinstance(source, str) or any(
                    marker in source
                    for marker in ("final", "controller", "confirmation")
                ):
                    _fail("B2 source role is forbidden or invalid")
                if source == "general_anchors":
                    continue
                if source not in by_source:
                    _fail("B2 source role is outside the frozen manifest contract")
                if not (
                    isinstance(raw.get("sample_id"), str)
                    and isinstance(raw.get("content_hash"), str)
                    and len(raw["content_hash"]) == 64
                    and isinstance(raw.get("question"), str)
                    and raw["question"].strip()
                ):
                    _fail("B2 source row lacks stable prompt-only identity")
                by_source[source].append(dict(raw))
    seed = config.get("run", {}).get("seed")
    if seed != 42 or not 0 <= step_index < 20:
        _fail("B2 seed/step is outside the frozen calibration envelope")
    selected: list[dict[str, Any]] = []
    for source in LENGTH_SOURCES:
        ranked = sorted(
            by_source[source],
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}\0b2-calibration\0{row['sample_id']}\0{row['content_hash']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                row["sample_id"],
            ),
        )
        chosen = ranked[step_index * 2 : step_index * 2 + 2]
        if len(chosen) != 2:
            _fail(f"B2 source {source} cannot supply 20 disjoint two-prompt windows")
        selected.extend(chosen)
    return selected


def _default_b2_step_kernel(
    session: Any,
    *,
    step_index: int,
    prompts_or_rows: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
    from_version: int,
    authority_sha256: str,
) -> Mapping[str, Any]:
    method = getattr(session, "run_b2_calibration_step", None)
    if not callable(method):
        _fail("production session lacks the source-real B2 calibration step kernel")
    return method(
        step_index=step_index,
        prompt_rows=prompts_or_rows,
        max_new_tokens=max_new_tokens,
        from_version=from_version,
        authority_sha256=authority_sha256,
    )


def _validate_b2_config(config: Mapping[str, Any]) -> int:
    _validate_isolation(config, allow_b2=True)
    run = config.get("run")
    generation = config.get("generation")
    authorization = config.get("authorization")
    qualification = config.get("qualification")
    execution = config.get("execution")
    data = config.get("data")
    if not all(
        isinstance(value, Mapping)
        for value in (
            run,
            generation,
            authorization,
            qualification,
            execution,
            data,
        )
    ):
        _fail("B2 calibration authorization package is incomplete")
    if data.get("selection_rule") != B2_SELECTION_RULE:
        _fail("B2 calibration data selection_rule contract drift")
    if not (
        run.get("stage") == "b2_medical_opd_calibration"
        and run.get("status") == "authorized_not_started"
        and run.get("optimizer_steps") == 20
        and run.get("seed") == 42
        and run.get("automatically_start") is False
        and execution.get("optimizer_steps") == 20
        and execution.get("calibration_only") is True
        and execution.get("automatically_start_b2") is False
        and authorization.get("production_sampler_refresh_ready") is True
        and authorization.get("OPD_scoring_backend_ready") is True
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_started") is False
        and generation.get("max_new_tokens") in P4_7_B2_ALLOWED_RESPONSE_LENGTHS
        and generation.get("do_sample") is True
        and generation.get("temperature") == 1.0
        and generation.get("top_k") == 0
        and generation.get("top_p") == 1.0
        and generation.get("full_support") is True
        and generation.get("enable_thinking") is False
        and data.get("allowed_roles") == list(LENGTH_SOURCES)
        and data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
    ):
        _fail("B2 calibration execution/authorization contract drift")
    for field in (
        "readiness_sha256",
        "artifact_index_sha256",
        "backend_binding_sha256",
        "protocol_sha256",
        "data_manifest_sha256",
        "authority_v2_sha256",
        "base_null_sha256",
        "length_decision_sha256",
        "cleanup_sha256",
    ):
        _digest(qualification.get(field), f"B2 qualification {field}")
    _digest(
        qualification.get("v2_tensor_sha256"),
        "B2 qualification v2_tensor_sha256",
    )
    for field in ("output_path", "v2_checkpoint_path"):
        value = qualification.get(field)
        if not isinstance(value, str) or not value:
            _fail(f"B2 qualification {field} is absent")
    return int(generation["max_new_tokens"])


def _validate_b2_step(
    raw: Mapping[str, Any],
    *,
    step_index: int,
    from_version: int,
    input_sha256: str,
) -> tuple[int, str, str, dict[str, Any]]:
    required = {
        "step_index",
        "from_version",
        "to_version",
        "generated_by_policy_version",
        "p_old_policy_version",
        "input_authority_tensor_sha256",
        "trainer_authority_tensor_sha256",
        "runtime_tensor_sha256",
        "fresh_tensor_sha256",
        "active_slot",
        "registry_count",
        "same_path_max_gap",
        "finite_rate",
        "delta_j",
        "delta_l",
        "alignment",
        "telemetry_complete",
        "normal_request_accepted",
        "stale_previous_rejected",
        "stale_error_code",
        "checkpoint_path",
        "step_artifact_sha256",
        "final_access",
        "controller_access",
        "confirmation_access",
        "label_access",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        _fail("B2 step result fields are not exact")
    to_version = from_version + 1
    if not (
        raw["step_index"] == step_index
        and raw["from_version"] == from_version
        and raw["to_version"] == to_version
        and raw["generated_by_policy_version"] == from_version
        and raw["p_old_policy_version"] == from_version
        and raw["input_authority_tensor_sha256"] == input_sha256
        and raw["active_slot"] == PRODUCTION_SLOT
        and raw["registry_count"] == 1
        and _finite(raw["same_path_max_gap"], "B2 same-path gap") <= 1e-4
        and _finite(raw["finite_rate"], "B2 finite rate") == 1.0
        and _finite(raw["delta_j"], "B2 delta J") > 0
        and _finite(raw["delta_l"], "B2 delta L") < 0
        and _finite(raw["alignment"], "B2 alignment") > 0
        and raw["telemetry_complete"] is True
        and raw["normal_request_accepted"] is True
        and raw["stale_previous_rejected"] is True
        and raw["stale_error_code"] == "STALE_SAMPLER_IDENTITY"
        and all(
            raw[field] is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
    ):
        _fail(f"B2 step {step_index} on-policy/stable-slot gate failed")
    trainer = _digest(raw["trainer_authority_tensor_sha256"], "B2 trainer authority")
    if trainer == input_sha256 or any(
        _digest(raw[field], f"B2 {field}") != trainer
        for field in ("runtime_tensor_sha256", "fresh_tensor_sha256")
    ):
        _fail(f"B2 step {step_index} trainer/runtime/fresh identity mismatch")
    checkpoint = raw["checkpoint_path"]
    if not isinstance(checkpoint, str) or not checkpoint:
        _fail("B2 step checkpoint path is absent")
    artifact_sha = _digest(raw["step_artifact_sha256"], "B2 step artifact")
    public = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "checkpoint_path",
        }
    }
    return to_version, trainer, checkpoint, public


def execute_b2_calibration_loop_v6(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    session_factory: SessionFactory | None = None,
    step_kernel: StepKernel | None = None,
    prompt_batch_provider: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]]
    | None = None,
    authorization_gate: AuthorizationGate | None = None,
    allow_b2_calibration: bool = False,
) -> dict[str, Any]:
    """Execute exactly 20 source-real calibration steps and nothing downstream."""

    max_new_tokens = _validate_b2_config(config)
    if "p4_7_start_gate" in config and allow_b2_calibration is not True:
        _fail("P4.7 B2 start requires explicit --allow-b2-calibration")
    if "p4_7_start_gate" in config and authorization_gate is not None:
        _fail("P4.7 B2 start forbids an injected authorization gate")
    path = Path(config_path)
    authorization_path = path.resolve().parent / "b2_authorization.json"
    evidence = (
        authorization_gate(authorization_path)
        if authorization_gate is not None
        else _default_b2_authorization_gate(
            authorization_path,
            config=config,
            allow_b2_calibration=allow_b2_calibration,
        )
    )
    if not isinstance(evidence, Mapping) or not (
        evidence.get("B2_authorized") is True
        and evidence.get("B2_started") is False
        and evidence.get("production_backend_id") == PRODUCTION_BACKEND_ID
        and evidence.get("optimizer_steps") == 20
        and evidence.get("selected_response_length") == max_new_tokens
    ):
        _fail("B2 start authorization evidence does not bind this calibration")
    factory = session_factory or _default_b2_session_factory
    kernel = step_kernel or _default_b2_step_kernel
    provider = prompt_batch_provider or _source_real_b2_prompt_batch
    if session_factory is None:
        output = Path(str(config["run"].get("output_dir", "")))
        if output.exists() or output.is_symlink():
            _fail("B2 output directory already exists")
        output.mkdir(parents=True)
    session = factory(config, path)
    try:
        initial = session.current_policy_identity()
        if not isinstance(initial, Mapping) or not (
            isinstance(initial.get("logical_version"), (int, str))
            and initial.get("active_slot") == PRODUCTION_SLOT
            and initial.get("registry_count") == 1
        ):
            _fail("B2 initial production sampler identity is absent")
        raw_version = initial["logical_version"]
        version = int(raw_version[1:]) if isinstance(raw_version, str) else int(raw_version)
        authority_sha = _digest(initial.get("tensor_sha256"), "B2 initial authority")
        qualification = config["qualification"]
        expected_checkpoint = Path(str(qualification["v2_checkpoint_path"]))
        if not expected_checkpoint.is_absolute():
            expected_checkpoint = (path.resolve().parent / expected_checkpoint).resolve()
        actual_checkpoint = Path(str(initial.get("checkpoint_path", "")))
        if not actual_checkpoint.is_absolute():
            actual_checkpoint = (path.resolve().parent / actual_checkpoint).resolve()
        if not (
            version == 2
            and authority_sha == qualification["v2_tensor_sha256"]
            and actual_checkpoint == expected_checkpoint
        ):
            _fail("B2 initial policy is not the authorized qualification v2")
        initial_version = version
        initial_sha = authority_sha
        step_evidence: list[dict[str, Any]] = []
        checkpoint = initial.get("checkpoint_path")
        for step_index in range(20):
            prompt_rows = list(provider(config, step_index))
            if len(prompt_rows) != 4 or any(
                sum(row.get("target_role") == source for row in prompt_rows) != 2
                for source in LENGTH_SOURCES
            ):
                _fail(f"B2 step {step_index} lacks source-real 2+2 prompts")
            result = kernel(
                session,
                step_index=step_index,
                prompts_or_rows=prompt_rows,
                max_new_tokens=max_new_tokens,
                from_version=version,
                authority_sha256=authority_sha,
            )
            version, authority_sha, checkpoint, public = _validate_b2_step(
                result,
                step_index=step_index,
                from_version=version,
                input_sha256=(
                    initial_sha if step_index == 0 else step_evidence[-1]["trainer_authority_tensor_sha256"]
                ),
            )
            step_evidence.append(public)
        return {
            "status": "passed_b2_medical_opd_calibration_20_step",
            "B2_started": True,
            "steps_completed": 20,
            "initial_logical_version": initial_version,
            "final_logical_version": version,
            "initial_tensor_sha256": initial_sha,
            "final_tensor_sha256": authority_sha,
            "final_checkpoint_path": checkpoint,
            "step_artifacts": [item["step_artifact_sha256"] for item in step_evidence],
            "source_selection_rule": B2_SELECTION_RULE,
            "production_backend_id": PRODUCTION_BACKEND_ID,
            "stable_slot": PRODUCTION_SLOT,
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
            "IDT_started": False,
            "SAR_started": False,
            "CA_OPD_started": False,
        }
    finally:
        session.close()
