"""Thin P4.8 adapter over the frozen production three-policy GPU kernel.

The module is deliberately CPU-import safe.  It validates the package-derived
fresh-Student envelope before the legacy kernel is imported or a model is
loaded, and projects privacy-safe kernel evidence into the P4.8 step contract.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from src.opd.production_b2_calibration_contract_v1 import (
    B2_CALIBRATION_STEPS,
    B2CalibrationContractV1Error,
    FRESH_STUDENT_INITIALIZATION,
    SELECTED_RESPONSE_LENGTH,
    SUPPORTED_RESPONSE_LENGTHS,
    validate_step_record,
)


class B2CalibrationBackendV1Error(RuntimeError):
    """The P4.8-to-production-kernel binding failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationBackendV1Error(message)


def build_b2_step_record_v1(
    kernel_evidence: Mapping[str, Any],
    *,
    selected_response_length: int = SELECTED_RESPONSE_LENGTH,
    expected_source_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Add the versioned envelope and validate the complete safe step record."""

    if not isinstance(kernel_evidence, Mapping):
        _fail("kernel step evidence is not an object")
    record = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_step_v1",
        **deepcopy(dict(kernel_evidence)),
    }
    step = record.get("optimizer_step")
    version = record.get("policy_version")
    if isinstance(step, bool) or not isinstance(step, int):
        _fail("kernel optimizer step is invalid")
    if isinstance(version, bool) or not isinstance(version, int):
        _fail("kernel policy version is invalid")
    try:
        return validate_step_record(
            record,
            expected_step=step,
            expected_version=version,
            selected_response_length=selected_response_length,
            expected_source_counts=expected_source_counts,
        )
    except B2CalibrationContractV1Error as error:
        raise B2CalibrationBackendV1Error(str(error)) from error


def _validate_fresh_envelope(config: Mapping[str, Any]) -> Mapping[str, Any]:
    run = config.get("run")
    generation = config.get("generation")
    execution = config.get("execution")
    initialization = config.get("student_initialization")
    qualification = config.get("qualification")
    isolation = config.get("isolation")
    if not all(
        isinstance(value, Mapping)
        for value in (run, generation, execution, initialization, qualification, isolation)
    ):
        _fail("package-derived calibration envelope is incomplete")
    selected_length = generation.get("max_new_tokens")
    package_version = config.get("package_version")
    if not (
        run.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and execution.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and selected_length in SUPPORTED_RESPONSE_LENGTHS
        and (
            selected_length == SELECTED_RESPONSE_LENGTH
            or package_version == "p4_8c_v3"
        )
    ):
        _fail(
            "calibration must remain exactly 20 steps at response length 768 "
            "or the versioned p4_8c_v3 1024 escalation"
        )
    if not (
        initialization.get("mode") == FRESH_STUDENT_INITIALIZATION
        and initialization.get("initial_logical_version") == 0
        and initialization.get("source_adapter_path") is None
        and initialization.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and initialization.get("forbidden_qualification_adapter_path")
        == qualification.get("v2_checkpoint_path")
        and initialization.get("forbidden_qualification_adapter_sha256")
        == qualification.get("v2_tensor_sha256")
    ):
        _fail("qualification v2 cannot initialize the fresh B2 Student")
    if any(
        isolation.get(field) is not False
        for field in (
            "final_access",
            "controller_access",
            "confirmation_access",
            "label_access",
        )
    ):
        _fail("calibration isolation differs from the package")
    return initialization


def create_production_b2_calibration_session_v1(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    _session_constructor: Callable[..., Any] | None = None,
) -> Any:
    """Create a fresh-v0 session while retaining the proven production kernel."""

    initialization = dict(_validate_fresh_envelope(config))
    if _session_constructor is None:
        from src.opd.production_qualification_two_step_gpu_v6 import (
            ProductionTwoStepSessionV6,
            _b2_runtime_config,
        )

        constructor: Callable[..., Any] = ProductionTwoStepSessionV6
        runtime_config = _b2_runtime_config(config)
    else:
        constructor = _session_constructor
        # Tests may use a reduced package-shaped config, but the same immutable
        # fresh-Student envelope is still validated above.
        try:
            from src.opd.production_qualification_two_step_gpu_v6 import (
                _b2_runtime_config,
            )

            runtime_config = _b2_runtime_config(config)
        except Exception:
            runtime_config = deepcopy(dict(config))
    runtime_config["student_initialization"] = initialization
    runtime_config["qualification_evidence"] = deepcopy(
        dict(config["qualification"])
    )
    runtime_config["data"] = deepcopy(dict(config["data"]))
    return constructor(
        runtime_config,
        config_path=Path(config_path),
        route="b2_calibration",
    )


__all__ = [
    "B2CalibrationBackendV1Error",
    "build_b2_step_record_v1",
    "create_production_b2_calibration_session_v1",
]
