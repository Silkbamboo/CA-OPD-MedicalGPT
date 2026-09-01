"""Package-gated factory for the P4.8d memory-balanced production session."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from src.opd.production_b2_calibration_contract_v2 import (
    B2_CALIBRATION_STEPS,
    FRESH_STUDENT_INITIALIZATION,
)
from src.opd.production_b2_memory_execution_v1 import (
    MemoryExecutionV1Error,
    validate_memory_execution_contract,
)


class B2CalibrationBackendV2Error(RuntimeError):
    """The P4.8d package-to-production-memory binding failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationBackendV2Error(message)


def validate_memory_runtime_envelope(config: Mapping[str, Any]) -> dict[str, Any]:
    run = config.get("run")
    generation = config.get("generation")
    execution = config.get("execution")
    student = config.get("student_initialization")
    qualification = config.get("qualification")
    isolation = config.get("isolation")
    if not all(
        isinstance(value, Mapping)
        for value in (run, generation, execution, student, qualification, isolation)
    ):
        _fail("memory calibration envelope is incomplete")
    try:
        memory = validate_memory_execution_contract(
            config.get("memory_execution", {})
        )
    except MemoryExecutionV1Error as error:
        raise B2CalibrationBackendV2Error(str(error)) from error
    if not (
        config.get("schema_id") == "ca-opd/b2-medical-opd-calibration/v4"
        and config.get("schema_version") == 4
        and config.get("package_version")
        in {
            "p4_8d_memory_v4",
            "p4_8e_memory_v5",
            "p4_8e_memory_v6",
            "p4_8e_memory_v7",
            "p4_8f_objective_evidence_v2",
        }
        and run.get("seed") == 42
        and run.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and generation.get("max_new_tokens") == 1024
        and execution.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and execution.get("physical_microbatch_size") == 1
        and execution.get("gradient_accumulation_steps") == 4
        and execution.get("effective_batch_size") == 4
        and execution.get("target_logit_chunk_size") == 128
        and execution.get("checkpoint_strategy")
        == "step5_step10_step15_step20_and_final"
        and (
            config.get("package_version") == "p4_8d_memory_v4"
            or execution.get("scheduler") == "constant_factor_1_no_lr_change"
        )
        and student.get("mode") == FRESH_STUDENT_INITIALIZATION
        and student.get("initial_logical_version") == 0
        and student.get("source_adapter_path") is None
        and student.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and student.get("forbidden_qualification_adapter_path")
        == qualification.get("v2_checkpoint_path")
        and student.get("forbidden_qualification_adapter_sha256")
        == qualification.get("v2_tensor_sha256")
        and all(
            isolation.get(field) is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
    ):
        _fail("memory calibration science/execution envelope differs")
    return memory


def _augment_projected_runtime(
    runtime_config: dict[str, Any],
    *,
    config: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_config["memory_execution"] = deepcopy(dict(memory))
    # The projection kernel predates versioned memory packages and therefore
    # does not carry these discriminators.  Restore them explicitly so the
    # checkpoint/reload gates cannot silently execute an older format.
    runtime_config["package_version"] = str(config["package_version"])
    runtime_config["student_initialization"] = deepcopy(
        dict(config["student_initialization"])
    )
    runtime_config["qualification_evidence"] = deepcopy(
        dict(config["qualification"])
    )
    runtime_config["data"] = deepcopy(dict(config["data"]))
    return runtime_config


def project_production_b2_memory_runtime_v1(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly project a verified package envelope before any model load.

    The package schema binds the three-policy formula by path and SHA instead
    of duplicating its ``algorithm`` mapping.  Differential preparation needs
    those numerical constants before session construction, so reopen the same
    SHA-bound formula and expose the verified mapping on the projected runtime.
    """

    memory = validate_memory_runtime_envelope(config)
    from src.opd.production_qualification_two_step_gpu_v7 import (
        ProductionTwoStepQualificationV6Error,
        _b2_runtime_config,
    )

    try:
        projected = _b2_runtime_config(config)
    except ProductionTwoStepQualificationV6Error as error:
        raise B2CalibrationBackendV2Error(
            f"production B2 runtime projection failed: {error}"
        ) from error
    runtime_config = _augment_projected_runtime(
        projected, config=config, memory=memory
    )
    validation = runtime_config.get("validation")
    if not isinstance(validation, Mapping):
        _fail("projected validation binding is absent")
    formula_path = Path(str(validation.get("config_path", "")))
    if not formula_path.is_absolute():
        formula_path = Path(__file__).resolve().parents[2] / formula_path
    formula_path = formula_path.resolve()
    if formula_path.is_symlink() or not formula_path.is_file():
        _fail("projected formula path is absent or a symlink")
    payload = formula_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != validation.get("config_sha256"):
        _fail("projected formula SHA reread differs")
    try:
        protocol = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise B2CalibrationBackendV2Error(
            f"projected formula is invalid: {type(error).__name__}"
        ) from error
    algorithm = protocol.get("algorithm") if isinstance(protocol, Mapping) else None
    if not isinstance(algorithm, Mapping):
        _fail("projected formula algorithm is absent")
    try:
        beta = float(algorithm["beta"])
        clip_low = float(algorithm["clip_low"])
        clip_high = float(algorithm["clip_high"])
    except (KeyError, TypeError, ValueError) as error:
        raise B2CalibrationBackendV2Error(
            "projected formula algorithm constants are invalid"
        ) from error
    if not (
        all(math.isfinite(value) for value in (beta, clip_low, clip_high))
        and beta > 0.0
        and 0.0 <= clip_low <= clip_high
    ):
        _fail("projected formula algorithm constants differ")
    runtime_config["algorithm"] = deepcopy(dict(algorithm))
    return runtime_config


def create_production_b2_memory_session_v1(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    _session_constructor: Callable[..., Any] | None = None,
    _allow_incomplete_test_projection: bool = False,
) -> Any:
    """Create a fresh-v0 memory session after the narrow CPU envelope gate."""

    memory = validate_memory_runtime_envelope(config)
    if _session_constructor is None or not _allow_incomplete_test_projection:
        runtime_config = project_production_b2_memory_runtime_v1(config)
    else:
        try:
            runtime_config = project_production_b2_memory_runtime_v1(config)
        except B2CalibrationBackendV2Error:
            runtime_config = deepcopy(dict(config))
            _augment_projected_runtime(
                runtime_config, config=config, memory=memory
            )
    if _session_constructor is None:
        from src.opd.production_b2_memory_execution_gpu_v1 import (
            MemoryBalancedProductionTwoStepSessionV1,
        )

        constructor: Callable[..., Any] = MemoryBalancedProductionTwoStepSessionV1
    else:
        constructor = _session_constructor
    return constructor(
        runtime_config,
        config_path=Path(config_path),
        route="b2_calibration",
    )


__all__ = [
    "B2CalibrationBackendV2Error",
    "create_production_b2_memory_session_v1",
    "project_production_b2_memory_runtime_v1",
    "validate_memory_runtime_envelope",
]
