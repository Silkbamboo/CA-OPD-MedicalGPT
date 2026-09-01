"""CPU-safe orchestration for the P4.6 GPU qualification.

The numerical kernels live in two focused GPU-only modules.  This wrapper owns
only fail-stop phase order, immutable artifact commits, runtime release, and the
two authorized top-level entry points.  Importing it does not import Torch,
Transformers, PEFT, CUDA, veRL, Ray, or vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


RUNTIME_PHASES = (
    "probe_manifest",
    "v0_guard",
    "reconstruction_step0",
    "authority_v1",
    "refresh_v1",
    "trajectory_step1_manifest",
    "reconstruction_step1",
    "authority_v2",
    "refresh_v2",
    "base_null",
    "length_smoke",
    "length_decision",
    "runtime_release",
)


class ProductionQualificationRuntimeV6Error(RuntimeError):
    """A scientific or runtime qualification gate failed closed."""


EmitPhase = Callable[[str, Mapping[str, Any], Mapping[str, Any]], str]


class QualificationBackend(Protocol):
    def run_micro(self, emit: EmitPhase) -> Mapping[str, Any]: ...
    def run_two_step(
        self, micro: Mapping[str, Any], emit: EmitPhase
    ) -> Mapping[str, Any]: ...
    def run_base_null(self) -> Mapping[str, Any]: ...
    def run_length_384(self, two_step: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def run_length_512(self, smoke: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def release(self) -> Mapping[str, Any]: ...


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProductionQualificationRuntimeV6Error(f"{label} is not a SHA-256")
    return value


def _validate_null(payload: Mapping[str, Any]) -> None:
    # Accept the strengthened artifact vocabulary.  The artifact writer repeats
    # the complete validation before the phase becomes durable.
    delta = payload.get("parameter_delta", payload.get("parameter_delta_norm"))
    gradient = payload.get("gradient_norm", payload.get("gradient_norm_after_clip"))
    if not (
        payload.get("advantage_max_abs") == 0
        and payload.get("objective") == 0
        and payload.get("loss") == 0
        and gradient == 0
        and delta == 0
        and payload.get("nonzero_update_tensor_count") == 0
    ):
        raise ProductionQualificationRuntimeV6Error("Base null gate failed")


def _candidate_passes(value: Mapping[str, Any]) -> bool:
    if value.get("finite_rate") != 1.0 or value.get("invalid_empty_count") != 0:
        return False
    if value.get("thinking_tag_count") != 0 or value.get("count") != 16:
        return False
    sources = value.get("per_source")
    if not isinstance(sources, Mapping):
        return False
    try:
        return bool(
            value["truncation_count"] <= 3
            and sources["medical_opd_o1"]["count"] == 8
            and sources["medical_opd_o1"]["truncation_count"] <= 1
            and sources["medical_opd_cmb"]["count"] == 8
            and sources["medical_opd_cmb"]["truncation_count"] <= 1
        )
    except (KeyError, TypeError):
        return False


def _length_decision(payload: Mapping[str, Any]) -> tuple[int, list[int]]:
    if _candidate_passes(payload.get("derived_256", {})):
        return 256, [256, 384]
    if _candidate_passes(payload.get("actual_384", {})):
        if payload.get("conditional_512_executed") is True:
            raise ProductionQualificationRuntimeV6Error("512 ran after passing 384")
        return 384, [256, 384]
    if payload.get("conditional_512_executed") is not True:
        raise ProductionQualificationRuntimeV6Error("required conditional 512 absent")
    if _candidate_passes(payload.get("actual_512", {})):
        return 512, [256, 384, 512]
    raise ProductionQualificationRuntimeV6Error("length_not_frozen")


def _validate_length_evidence(
    payload: Mapping[str, Any], two_step: Mapping[str, Any]
) -> None:
    telemetry = payload.get("telemetry")
    identity = payload.get("policy_identity")
    if not isinstance(telemetry, Mapping) or not telemetry:
        raise ProductionQualificationRuntimeV6Error(
            "length telemetry is absent from the committed payload"
        )
    if not isinstance(identity, Mapping):
        raise ProductionQualificationRuntimeV6Error(
            "length v2 policy identity is absent from the committed payload"
        )
    expected_v2 = _digest(two_step.get("v2_tensor_sha256"), "length expected v2")
    expected_authority = _digest(
        two_step.get("authority_v2_artifact_sha256"),
        "length authority_v2 artifact",
    )
    expected_checkpoint = Path(str(two_step.get("checkpoint_v2", "")))
    checkpoint = Path(str(identity.get("checkpoint_path", "")))
    if not (
        identity.get("logical_version") == "v2"
        and _digest(identity.get("tensor_sha256"), "length runtime v2")
        == expected_v2
        and _digest(
            identity.get("authority_v2_artifact_sha256"),
            "length runtime authority_v2 artifact",
        )
        == expected_authority
        and expected_checkpoint.is_absolute()
        and checkpoint.is_absolute()
        and checkpoint == expected_checkpoint
        and identity.get("active_slot") == "student_active"
        and identity.get("registry_count") == 1
    ):
        raise ProductionQualificationRuntimeV6Error(
            "length policy identity is not the committed trainer-authoritative v2"
        )


def execute_qualification_state_machine(
    *, backend: QualificationBackend, emit: EmitPhase
) -> dict[str, Any]:
    """Run ordinal 2..14; post-exit cleanup/terminal remain external."""

    released = False
    try:
        micro = dict(backend.run_micro(emit))
        v1 = _digest(micro.get("v1_tensor_sha256"), "v1 authority")
        _digest(micro.get("refresh_v1_artifact_sha256"), "v1 refresh artifact")
        two_step = dict(backend.run_two_step(micro, emit))
        if _digest(two_step.get("v1_tensor_sha256"), "step1 input v1") != v1:
            raise ProductionQualificationRuntimeV6Error(
                "step1 did not consume trainer-authoritative v1"
            )
        v2 = _digest(two_step.get("v2_tensor_sha256"), "v2 authority")
        if v2 == v1:
            raise ProductionQualificationRuntimeV6Error(
                "v2 tensor identity did not change from v1"
            )

        null_payload = dict(backend.run_base_null())
        _validate_null(null_payload)
        null_payload.setdefault("status", "pass")
        emit(
            "base_null",
            null_payload,
            {"status": "pass", "advantage_max_abs": 0.0, "parameter_delta": 0.0},
        )

        length_payload = dict(backend.run_length_384(two_step))
        # Compatibility with a kernel that returns its evidence envelope.
        if "artifact_payload" in length_payload:
            length_payload = dict(length_payload["artifact_payload"])
        elif "length_smoke" in length_payload and "derived_256" not in length_payload:
            raise ProductionQualificationRuntimeV6Error(
                "length kernel omitted strengthened artifact_payload"
            )
        _validate_length_evidence(length_payload, two_step)
        selected, evaluated = _length_decision(length_payload)
        length_payload.setdefault("status", "pass")
        length_sha = emit(
            "length_smoke",
            length_payload,
            {
                "status": "pass",
                "actual_lengths": length_payload["actual_lengths"],
                "selected_response_length": selected,
            },
        )
        emit(
            "length_decision",
            {
                "status": "pass",
                "selected_response_length": selected,
                "length_smoke_sha256": _digest(length_sha, "length smoke artifact"),
                "evaluated_candidates": evaluated,
                "decision_rule": (
                    "shortest_passing_overall_and_per_source_truncation_v1"
                ),
            },
            {"status": "pass", "selected_response_length": selected},
        )

        release_payload = dict(backend.release())
        released = True
        if release_payload.get("models_released") is not True:
            raise ProductionQualificationRuntimeV6Error("runtime model release failed")
        release_payload.setdefault("status", "pass")
        release_payload.setdefault("B2_started", False)
        emit(
            "runtime_release",
            release_payload,
            {"status": "pass", "models_released": True},
        )
        return {
            "status": "runtime_passed_pending_post_process_cleanup",
            "runtime_exit_code": 0,
            "v1_tensor_sha256": v1,
            "v2_tensor_sha256": v2,
            "selected_response_length": selected,
            "B2_started": False,
        }
    finally:
        if not released:
            try:
                backend.release()
            except Exception:
                pass


@dataclass
class _ArtifactEmitter:
    output: Path
    bindings: Mapping[str, Any]
    mode: str
    next_ordinal: int = 2

    def assert_micro_evidence_prefix_ready(
        self,
        *,
        expected_v1_tensor_sha256: str,
        expected_refresh_v1_sha256: str,
    ) -> Mapping[str, Any]:
        from src.opd.production_qualification_artifacts_v6 import (
            assert_micro_evidence_prefix_ready,
        )

        result = assert_micro_evidence_prefix_ready(
            self.output,
            self.bindings,
            expected_v1_tensor_sha256,
            expected_refresh_v1_sha256,
        )
        if not isinstance(result, Mapping):
            raise ProductionQualificationRuntimeV6Error(
                "artifact-derived micro gate returned no readiness evidence"
            )
        return result

    def __call__(
        self, phase: str, payload: Mapping[str, Any], metric: Mapping[str, Any]
    ) -> str:
        from src.opd.production_qualification_artifacts_v6 import (
            commit_phase,
            sha256_file,
        )

        expected = RUNTIME_PHASES[self.next_ordinal - 2]
        if phase != expected:
            raise ProductionQualificationRuntimeV6Error(
                f"runtime phase order drift: expected {expected}, got {phase}"
            )
        commit_phase(
            self.output,
            bindings=self.bindings,
            mode=self.mode,
            phase_id=phase,
            ordinal=self.next_ordinal,
            payload=payload,
            metric=metric,
        )
        self.next_ordinal += 1
        return sha256_file(self.output / f"{phase}.json")


class RealGPUBackend:
    """Composition adapter for the persistent two-step and auxiliary kernels."""

    def __init__(self, config: Mapping[str, Any], *, config_path: Path) -> None:
        self.config = config
        self.config_path = Path(config_path)
        self.result: dict[str, Any] | None = None
        self.released = False

    def run_micro(self, emit: EmitPhase) -> Mapping[str, Any]:
        from src.opd.production_qualification_two_step_gpu_v6 import (
            execute_two_step_qualification_v6,
        )

        micro_gate = getattr(emit, "assert_micro_evidence_prefix_ready", None)
        if not callable(micro_gate):
            raise ProductionQualificationRuntimeV6Error(
                "artifact-derived micro gate callback is absent"
            )
        self.result = dict(
            execute_two_step_qualification_v6(
                self.config,
                config_path=self.config_path,
                emit=emit,
                micro_gate=micro_gate,
            )
        )
        return {
            "v1_tensor_sha256": self.result["v1_tensor_sha256"],
            "refresh_v1_artifact_sha256": self.result[
                "refresh_v1_artifact_sha256"
            ],
        }

    def run_two_step(
        self, micro: Mapping[str, Any], emit: EmitPhase
    ) -> Mapping[str, Any]:
        if self.result is None:
            raise ProductionQualificationRuntimeV6Error("two-step kernel did not run")
        if self.result["v1_tensor_sha256"] != micro["v1_tensor_sha256"]:
            raise ProductionQualificationRuntimeV6Error("two-step v1 binding changed")
        checkpoint = Path(str(self.result["checkpoint_v2"]))
        if not checkpoint.is_absolute():
            checkpoint = (
                Path(str(self.config["run"]["output_dir"])) / checkpoint
            ).resolve()
        return {
            "v1_tensor_sha256": self.result["v1_tensor_sha256"],
            "v2_tensor_sha256": self.result["v2_tensor_sha256"],
            "authority_v2_artifact_sha256": self.result[
                "authority_v2_artifact_sha256"
            ],
            "checkpoint_v2": str(checkpoint),
        }

    def run_base_null(self) -> Mapping[str, Any]:
        from src.opd.production_qualification_aux_gpu_v6 import (
            base_null_artifact_payload,
            execute_base_teacher_null_v6,
        )

        semantic_payload = execute_base_teacher_null_v6(
            self.config, config_path=self.config_path, session=None
        )
        return base_null_artifact_payload(semantic_payload)

    def run_length_384(self, two_step: Mapping[str, Any]) -> Mapping[str, Any]:
        from src.opd.production_qualification_aux_gpu_v6 import (
            execute_length_calibration_v6,
        )

        if self.result is None or two_step["v2_tensor_sha256"] != self.result[
            "v2_tensor_sha256"
        ]:
            raise ProductionQualificationRuntimeV6Error("length actor is not v2")
        return execute_length_calibration_v6(
            self.config,
            config_path=self.config_path,
            checkpoint_v2=Path(str(two_step["checkpoint_v2"])),
            v2_tensor_sha256=self.result["v2_tensor_sha256"],
            authority_v2_artifact_sha256=two_step[
                "authority_v2_artifact_sha256"
            ],
            session=None,
        )

    def run_length_512(self, smoke: Mapping[str, Any]) -> Mapping[str, Any]:
        raise ProductionQualificationRuntimeV6Error(
            "auxiliary length kernel must perform conditional 512 internally"
        )

    def release(self) -> Mapping[str, Any]:
        if self.released:
            return {
                "status": "pass",
                "models_released": True,
                "post_process_cleanup_required": True,
                "B2_started": False,
            }
        self.released = True
        return {
            "status": "pass",
            "models_released": True,
            "post_process_cleanup_required": True,
            "B2_started": False,
        }


def execute_production_qualification_gpu_protocol_v6(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    artifact_bindings: Mapping[str, Any],
    artifact_mode: str = "full",
) -> dict[str, Any]:
    """Execute real qualification phases; finalizer owns cleanup/readiness."""

    if artifact_mode != "full":
        raise ProductionQualificationRuntimeV6Error("P4.6 production mode must be full")
    prompt_selection = config.get("prompt_selection")
    if not isinstance(prompt_selection, Mapping) or _digest(
        prompt_selection.get("selection_manifest_sha256"),
        "frozen prompt manifest",
    ) != _digest(
        artifact_bindings.get("prompt_manifest_sha256"),
        "artifact prompt manifest",
    ):
        raise ProductionQualificationRuntimeV6Error(
            "frozen prompt manifest binding mismatch"
        )
    output = Path(str(config["run"]["output_dir"]))
    backend = RealGPUBackend(config, config_path=Path(config_path))
    emitter = _ArtifactEmitter(output, artifact_bindings, artifact_mode)
    try:
        result = execute_qualification_state_machine(backend=backend, emit=emitter)
    except BaseException as error:
        from src.opd.production_qualification_artifacts_v6 import record_failure

        if not (output / "failure.json").exists():
            record_failure(
                output,
                bindings=artifact_bindings,
                mode=artifact_mode,
                reason=f"runtime:{type(error).__name__}:{error}",
            )
        raise
    result["B2_started"] = False
    return result


def _execute_b2_calibration_loop(
    config: Mapping[str, Any], *, config_path: Path, allow_b2_calibration: bool = False
) -> Mapping[str, Any]:
    """Resolve and execute the single production B2 calibration kernel."""

    from src.opd.production_qualification_aux_gpu_v6 import (
        execute_b2_calibration_loop_v6,
    )

    return execute_b2_calibration_loop_v6(
        config,
        config_path=Path(config_path),
        allow_b2_calibration=allow_b2_calibration,
    )


def execute_b2_medical_opd_gpu_protocol_v6(
    config: Mapping[str, Any], *, config_path: Path, allow_b2_calibration: bool = False
) -> dict[str, Any]:
    """Execute the source-real 20-step calibration loop after authorization."""

    result = dict(
        _execute_b2_calibration_loop(
            config,
            config_path=Path(config_path),
            allow_b2_calibration=allow_b2_calibration,
        )
    )
    result.setdefault("B2_started", False)
    return result


__all__ = [
    "ProductionQualificationRuntimeV6Error",
    "RUNTIME_PHASES",
    "RealGPUBackend",
    "execute_b2_medical_opd_gpu_protocol_v6",
    "execute_production_qualification_gpu_protocol_v6",
    "execute_qualification_state_machine",
]
