"""P4.8d canary-first, memory-balanced 20-step calibration worker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from src.opd.production_b2_calibration_artifacts_v2 import (
    B2CalibrationArtifactStoreV1,
    B2CalibrationArtifactsV1Error,
)
from src.opd.production_b2_calibration_contract_v2 import (
    B2_CALIBRATION_STEPS,
    FRESH_STUDENT_INITIALIZATION,
    canonical_json_sha256,
    evaluate_latest_calibration_length_window,
)
from src.opd.production_b2_calibration_v1 import (
    B2CalibrationLauncherV1Error,
    _append_safe_log,
    _atomic_json,
    _worker_metadata,
)
from src.opd.production_b2_memory_execution_v1 import (
    MemoryExecutionV1Error,
    assert_canary_isolated,
    evaluate_six_step_memory_drift,
)


def _clear_cublas_workspaces(torch: Any) -> None:
    """Release pinned-version cuBLAS workspaces at an in-process gate."""

    clear = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
    if not callable(clear):
        raise B2CalibrationLauncherV1Error(
            "pinned PyTorch cuBLAS workspace cleanup API is unavailable"
        )
    clear()


def _build_memory_runtime_config(
    package_audit: Mapping[str, Any], *, output_dir: str | Path
) -> dict[str, Any]:
    """Derive the V4 runtime without widening the historical V1 launcher."""

    source = package_audit.get("config")
    if not isinstance(source, Mapping) or not (
        package_audit.get("package_version")
        in {
            "p4_8d_memory_v4",
            "p4_8e_memory_v5",
            "p4_8e_memory_v6",
            "p4_8e_memory_v7",
            "p4_8f_objective_evidence_v2",
        }
        and package_audit.get("selected_response_length") == 1024
        and package_audit.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and package_audit.get("seed") == 42
        and package_audit.get("student_initialization")
        == FRESH_STUDENT_INITIALIZATION
        and package_audit.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
    ):
        raise B2CalibrationLauncherV1Error(
            "verified P4.8d package differs from the frozen runtime contract"
        )
    config = deepcopy(dict(source))
    run = config.get("run")
    generation = config.get("generation")
    execution = config.get("execution")
    memory = config.get("memory_execution")
    if not all(
        isinstance(value, dict)
        for value in (run, generation, execution, memory)
    ):
        raise B2CalibrationLauncherV1Error(
            "verified P4.8d runtime sections are incomplete"
        )
    runtime_run_id = str(package_audit.get("runtime_run_id") or "")
    if not runtime_run_id:
        raise B2CalibrationLauncherV1Error("P4.8d runtime run ID is absent")
    run.update(
        {
            "run_id": runtime_run_id,
            "stage": "b2_calibration",
            "purpose": "memory-balanced Medical OPD 20-step calibration",
            "seed": 42,
            "optimizer_steps": B2_CALIBRATION_STEPS,
            "output_dir": str(Path(output_dir).resolve()),
            "status": "authorized_not_started",
            "automatically_start": False,
        }
    )
    generation["max_new_tokens"] = 1024
    execution.update(
        {
            "optimizer_steps": B2_CALIBRATION_STEPS,
            "calibration_only": True,
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "target_logit_chunk_size": 128,
            "checkpoint_strategy": "step5_step10_step15_step20_and_final",
        }
    )
    config["student_initialization"] = {
        "mode": FRESH_STUDENT_INITIALIZATION,
        "initial_logical_version": 0,
        "source_adapter_path": None,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "forbidden_qualification_adapter_path": str(
            Path(str(package_audit["qualification_v2_path"])).resolve()
        ),
        "forbidden_qualification_adapter_sha256": package_audit[
            "qualification_v2_tensor_sha256"
        ],
    }
    config["p4_8d_start_gate"] = {
        "source_package_content_sha256": package_audit[
            "package_content_sha256"
        ],
        "source_authorization_sha256": package_audit["authorization_sha256"],
        "source_memory_execution_contract_sha256": package_audit[
            "memory_execution_contract_sha256"
        ],
        "requires_environment": "CA_OPD_ALLOW_B2_CALIBRATION_GPU=1",
        "requires_argument": "--allow-b2-calibration",
    }
    if package_audit.get("package_version") in {
        "p4_8e_memory_v5",
        "p4_8e_memory_v6",
        "p4_8e_memory_v7",
        "p4_8f_objective_evidence_v2",
    }:
        gate_name = (
            "p4_8f_start_gate"
            if package_audit.get("package_version")
            == "p4_8f_objective_evidence_v2"
            else "p4_8e_start_gate"
        )
        config[gate_name] = {
            **dict(config.get(gate_name, {})),
            "source_package_content_sha256": package_audit[
                "package_content_sha256"
            ],
            "parent_package_content_sha256": package_audit[
                "parent_package_content_sha256"
            ],
            "requires_gpu_math_differential": True,
            "requires_max_shape_canary": True,
            "requires_formal_fresh_v0": True,
            "requires_environment": "CA_OPD_ALLOW_B2_CALIBRATION_GPU=1",
            "requires_argument": "--allow-b2-calibration",
        }
    return config


def _memory_package_binding(package_audit: Mapping[str, Any]) -> dict[str, Any]:
    package_version = str(package_audit["package_version"])
    return {
        "schema_version": 4,
        "artifact_kind": (
            (
                "p4_8f_objective_evidence_package_binding_v2"
                if package_version == "p4_8f_objective_evidence_v2"
                else "p4_8e_memory_package_binding_v5"
            )
            if package_version
            in {
                "p4_8e_memory_v5",
                "p4_8e_memory_v6",
                "p4_8e_memory_v7",
                "p4_8f_objective_evidence_v2",
            }
            else "p4_8d_memory_package_binding_v4"
        ),
        "package_version": package_version,
        "package_content_sha256": package_audit["package_content_sha256"],
        "authorization_sha256": package_audit["authorization_sha256"],
        "config_sha256": package_audit["config_sha256"],
        "run_card_sha256": package_audit["run_card_sha256"],
        "manifest_sha256": package_audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": package_audit["schedule"]["schedule_sha256"],
        "oom_memory_attestation_sha256": package_audit[
            "oom_memory_attestation_sha256"
        ],
        "memory_execution_contract_sha256": package_audit[
            "memory_execution_contract_sha256"
        ],
        "selected_response_length": 1024,
        "optimizer_steps": B2_CALIBRATION_STEPS,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "qualification_v2_tensor_sha256": package_audit[
            "qualification_v2_tensor_sha256"
        ],
        "B2_authorized": True,
        "B2_started": False,
        "pre_model_semantic_gate": True,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }


def _memory_data_manifest(package_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "artifact_kind": "b2_calibration_data_manifest_v4",
        "manifest_sha256": package_audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": package_audit["schedule"]["schedule_sha256"],
        "schedule_slot_count": package_audit["schedule"]["slot_count"],
        "selection_rule": "frozen_20x4_schedule_seed42_v2",
        "provider": "production_b2_data_v2.resolve_b2_schedule_batch",
        "prompt_only": True,
        "raw_prompts_persisted": False,
        "labels_accessed": False,
        "pre_model_semantic_gate": True,
    }


def _memory_worker_failure_classification(
    error: BaseException,
) -> dict[str, Any]:
    message = str(error).lower()
    if "canary" in message or "memory drift" in message:
        code = "failed_b2_calibration_memory_evidence"
        phase = "runtime_memory_canary_or_drift"
    elif "data_manifest_identity" in message or any(
        marker in message
        for marker in ("manifest", "schedule", "prompt-only", "prompt_only")
    ):
        code = "failed_b2_calibration_data_manifest_identity"
        phase = "pre_model_semantic_gate"
    elif "out of memory" in message or "oom" in message:
        code = "failed_b2_calibration_oom"
        phase = "runtime_memory"
    elif "nan" in message or "inf" in message or "non-finite" in message:
        code = "failed_b2_calibration_nonfinite"
        phase = "optimizer_or_scoring"
    elif "identity" in message or "stale" in message or "authority" in message:
        code = "failed_b2_calibration_student_identity"
        phase = "runtime_identity"
    elif "artifact" in message or "sha" in message or "checkpoint" in message:
        code = "failed_artifact_integrity"
        phase = "artifact_or_checkpoint"
    elif "length" in message and "1024" in message:
        code = "failed_b2_calibration_length_insufficient"
        phase = "rolling_length_gate"
    else:
        code = "failed_b2_calibration_worker"
        phase = "worker_runtime"
    return {
        "primary_failure_code": code,
        "failure_phase": phase,
        "causal_chain": [code, type(error).__name__],
    }


def _default_session_factory(config: Mapping[str, Any], config_path: Path) -> Any:
    from src.opd.production_b2_calibration_backend_v2 import (
        create_production_b2_memory_session_v1,
    )

    return create_production_b2_memory_session_v1(
        config, config_path=config_path
    )


def _default_prompt_provider(
    config: Mapping[str, Any], step_index: int
) -> Sequence[Mapping[str, Any]]:
    from src.opd.production_qualification_aux_gpu_v6 import (
        _source_real_b2_prompt_batch,
    )

    return _source_real_b2_prompt_batch(config, step_index)


def _completed_steps(output: Path) -> int:
    path = output / "metrics.jsonl"
    if not path.is_file() or path.is_symlink():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _atomic_stream_copy(source: Path, target: Path) -> dict[str, Any]:
    """Copy privacy-safe canary telemetry without a non-atomic final file."""

    if source.is_symlink() or not source.is_file():
        raise B2CalibrationLauncherV1Error("memory canary telemetry is absent")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    line_count = 0
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                line_count += block.count(b"\n")
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
        directory = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    if size != target.stat().st_size:
        raise B2CalibrationLauncherV1Error(
            "memory canary telemetry size reread differs"
        )
    reread = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            reread.update(block)
    if reread.hexdigest() != digest.hexdigest():
        raise B2CalibrationLauncherV1Error(
            "memory canary telemetry SHA reread differs"
        )
    return {
        "path": target.name,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "record_count": line_count,
    }


def execute_memory_balanced_calibration_worker_v1(
    *,
    package_audit: Mapping[str, Any],
    output_dir: str | Path,
    execution_mode: str,
    git_commit: str,
    session_factory: Callable[[Mapping[str, Any], Path], Any] | None = None,
    prompt_provider: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]]
    | None = None,
) -> dict[str, Any]:
    if execution_mode not in {"formal_gpu", "mock"}:
        raise B2CalibrationLauncherV1Error("worker execution mode is invalid")
    package_version = package_audit.get("package_version")
    is_p4e = package_version in {
        "p4_8e_memory_v5",
        "p4_8e_memory_v6",
        "p4_8e_memory_v7",
        "p4_8f_objective_evidence_v2",
    }
    if package_version not in {
        "p4_8d_memory_v4",
        "p4_8e_memory_v5",
        "p4_8e_memory_v6",
        "p4_8e_memory_v7",
        "p4_8f_objective_evidence_v2",
    }:
        raise B2CalibrationLauncherV1Error(
            "memory worker accepts only a versioned P4.8d/P4.8e/P4.8f package"
        )
    try:
        if is_p4e:
            if package_version == "p4_8f_objective_evidence_v2":
                from src.opd.production_b2_objective_revalidation_package_v2 import (
                    verify_objective_revalidation_package,
                )

                package_audit = verify_objective_revalidation_package(
                    package_audit["package_dir"],
                    canonical_manifest_path=package_audit["data_authority"][
                        "manifest_path"
                    ],
                )
            else:
                from src.opd.production_b2_memory_revalidation_package_v1 import (
                    verify_revalidation_overlay_package,
                )

                package_audit = verify_revalidation_overlay_package(
                    package_audit["package_dir"],
                    canonical_manifest_path=package_audit["data_authority"][
                        "manifest_path"
                    ],
                )
        else:
            from src.opd.production_b2_calibration_package_v4 import (
                pre_model_semantic_preflight_v4,
            )

            semantic = pre_model_semantic_preflight_v4(
                package_audit["package_dir"],
                canonical_manifest_path=package_audit["data_authority"][
                    "manifest_path"
                ],
            )
            package_audit = semantic["audit"]
    except (KeyError, RuntimeError) as error:
        raise B2CalibrationLauncherV1Error(
            "failed_b2_calibration_data_manifest_identity: "
            f"{type(error).__name__}:{error}"
        ) from error
    output = Path(output_dir).resolve()
    runtime_run_id = str(package_audit["runtime_run_id"])
    runtime_config = _build_memory_runtime_config(
        package_audit, output_dir=output
    )
    runtime_config_sha = canonical_json_sha256(runtime_config)
    store = B2CalibrationArtifactStoreV1(
        output,
        run_id=runtime_run_id,
        config={
            "run_id": runtime_run_id,
            "stage": "b2_calibration",
            "optimizer_steps": 20,
            "selected_response_length": 1024,
            "seed": 42,
            "student_initialization": FRESH_STUDENT_INITIALIZATION,
            "checkpoint_strategy": "step5_step10_step15_step20_and_final",
            "automatically_start_formal_b2": False,
            "runtime_config_sha256": runtime_config_sha,
            "memory_execution_contract_sha256": package_audit[
                "memory_execution_contract_sha256"
            ],
        },
        metadata=_worker_metadata(
            execution_mode=execution_mode,
            git_commit=git_commit,
            run_id=runtime_run_id,
        ),
        package_binding=_memory_package_binding(package_audit),
        data_manifest=_memory_data_manifest(package_audit),
    )
    store.initialize()
    _atomic_json(output / "runtime_config.json", runtime_config)
    _append_safe_log(
        output / "stdout.log",
        "worker_start mode="
        f"{execution_mode} steps=20 length=1024 microbatch=1 accumulation=4",
    )
    factory = session_factory or _default_session_factory
    provider = prompt_provider or _default_prompt_provider
    canary_session = None
    canary_torch = None
    canary_cleanup_before_formal: dict[str, Any] = {}
    formal_session = None
    step_records: list[Mapping[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".p4_8d_memory_canary_", dir=output.parent
        ) as canary_name:
            canary_root = Path(canary_name)
            canary_config = deepcopy(runtime_config)
            canary_config["run"] = deepcopy(runtime_config["run"])
            canary_config["run"]["run_id"] = f"{runtime_run_id}-canary"
            canary_config["run"]["output_dir"] = str(canary_root)
            canary_config_path = canary_root / "runtime_config.json"
            _atomic_json(canary_config_path, canary_config)
            canary_session = factory(canary_config, canary_config_path)
            canary_initial = canary_session.initial_calibration_identity()
            canary_prompts = list(provider(runtime_config, 3))
            fixture_summary: dict[str, Any] = {}
            if is_p4e:
                from src.opd.production_b2_max_shape_canary_v1 import (
                    choose_max_prompt_batch,
                    install_max_shape_rollout_fixture,
                    production_prompt_token_length,
                )

                if execution_mode == "formal_gpu":
                    schedule_batches = [
                        list(provider(runtime_config, step_index))
                        for step_index in range(B2_CALIBRATION_STEPS)
                    ]
                    canary_prompts = choose_max_prompt_batch(
                        schedule_batches,
                        prompt_length=lambda row: production_prompt_token_length(
                            canary_session, row
                        ),
                        risk_step_index=3,
                    )
                    install_max_shape_rollout_fixture(
                        canary_session, valid_token_count=1024
                    )
                else:
                    fixture_summary = {
                        "real_rollout_executed": True,
                        "real_rollout_trajectory_count": 4,
                        "real_rollout_completion_lengths": [1024] * 4,
                        "fixture_indices": [0, 1, 2, 3],
                        "fixture_prompt_tokens": 1,
                        "fixture_prompt_tokens_by_prompt": [1] * 4,
                        "fixture_valid_completion_tokens": 1024,
                        "fixture_valid_completion_tokens_by_prompt": [1024] * 4,
                        "fixture_eos": False,
                        "fixture_source": (
                            "prompt_only_legal_synthetic_token_shape"
                        ),
                        "synthetic_target_scoring_count": 4,
                        "label_access_count": 0,
                        "controller_access_count": 0,
                        "final_access_count": 0,
                        "response_tokens_persisted": False,
                    }
            if len(canary_prompts) != 4:
                raise B2CalibrationLauncherV1Error(
                    "memory canary step-4 prompt batch is not 2+2"
                )
            canary_step_record = canary_session.run_b2_calibration_step_v1(
                step_index=0,
                prompt_rows=canary_prompts,
                max_new_tokens=1024,
            )
            if not (
                int(canary_step_record.get("teacher_gradient_tensor_count", -1))
                == 0
                and int(canary_step_record.get("base_gradient_tensor_count", -1))
                == 0
                and all(
                    isinstance(canary_step_record.get(field), (int, float))
                    and math.isfinite(float(canary_step_record[field]))
                    for field in (
                        "objective",
                        "loss",
                        "gradient_norm",
                        "adapter_delta_norm",
                    )
                )
            ):
                raise B2CalibrationLauncherV1Error(
                    "P4.8e canary gradient/finite gate failed"
                )
            if is_p4e and execution_mode == "formal_gpu":
                fixture_summary = dict(
                    getattr(canary_session, "_p4e_max_shape_fixture", {})
                )
            if is_p4e and not (
                fixture_summary.get("real_rollout_executed") is True
                and fixture_summary.get("fixture_valid_completion_tokens") == 1024
                and fixture_summary.get(
                    "fixture_valid_completion_tokens_by_prompt"
                ) == [1024] * 4
                and fixture_summary.get("synthetic_target_scoring_count") == 4
                and fixture_summary.get("fixture_eos") is False
                and fixture_summary.get("label_access_count") == 0
                and fixture_summary.get("controller_access_count") == 0
                and fixture_summary.get("final_access_count") == 0
                and fixture_summary.get("response_tokens_persisted") is False
            ):
                raise B2CalibrationLauncherV1Error(
                    "P4.8e max-shape canary fixture was not exercised"
                )
            observer = getattr(canary_session, "_memory_phase_observer", None)
            if callable(observer):
                observer("before", "artifact_writer", step=1)
            _atomic_json(
                canary_root / "canary_runtime_artifact.json",
                {
                    "schema_version": 1,
                    "artifact_kind": "p4_8e_canary_runtime_artifact_v1",
                    "run_id": canary_config["run"]["run_id"],
                    "fixture_valid_completion_tokens": int(
                        fixture_summary.get("fixture_valid_completion_tokens", 0)
                    ),
                    "fixture_valid_completion_tokens_by_prompt": list(
                        fixture_summary.get(
                            "fixture_valid_completion_tokens_by_prompt", []
                        )
                    ),
                    "synthetic_target_scoring_count": int(
                        fixture_summary.get("synthetic_target_scoring_count", 0)
                    ),
                    "real_rollout_executed": bool(
                        fixture_summary.get("real_rollout_executed", True)
                    ),
                    "cuda_tensor_persisted": False,
                    "grad_fn_persisted": False,
                },
            )
            if callable(observer):
                observer("after", "artifact_writer")
            runtime_canary = dict(
                canary_session.memory_canary_runtime_summary_v1()
            )
            if is_p4e and execution_mode == "formal_gpu" and not {
                "scheduler_step_count",
                "backbone_forward_calls",
                "backbone_backward_calls",
                "lm_head_chunk_count",
                "retain_graph_calls",
                "adapter_export_count",
                "fresh_identity_verifier_count",
                "policy_version_increment_count",
            } <= set(runtime_canary):
                raise B2CalibrationLauncherV1Error(
                    "P4.8e canary execution topology evidence is absent"
                )
            if is_p4e and execution_mode == "formal_gpu" and not (
                runtime_canary.get("optimizer_steps_executed") == 1
                and runtime_canary.get("scheduler_step_count") == 1
                and runtime_canary.get("backbone_forward_calls") == 4
                and runtime_canary.get("backbone_backward_calls") == 4
                and runtime_canary.get("lm_head_chunk_count") == 32
                and runtime_canary.get("retain_graph_calls") == 0
                and runtime_canary.get("adapter_export_count") == 1
                and runtime_canary.get("sampler_refresh_count") == 1
                and runtime_canary.get("fresh_identity_verifier_count") == 1
                and runtime_canary.get("policy_version_increment_count") == 1
            ):
                raise B2CalibrationLauncherV1Error(
                    "P4.8e 4x1024 canary lifecycle count differs"
                )
            if not (
                runtime_canary.get("initial_registry_count")
                == runtime_canary.get("final_registry_count")
                and runtime_canary.get("initial_model_count")
                == runtime_canary.get("final_model_count")
            ):
                raise B2CalibrationLauncherV1Error(
                    "P4.8e canary runtime/model/adapter registry grew"
                )
            canary_telemetry = _atomic_stream_copy(
                canary_root / "memory_telemetry" / "telemetry.jsonl",
                output / "memory_canary_telemetry.jsonl",
            )
            canary_torch = getattr(canary_session, "torch", None)
            canary_session.close()
            canary_session = None
        if execution_mode == "formal_gpu":
            import gc

            if canary_torch is None:
                raise B2CalibrationLauncherV1Error(
                    "P4.8e canary CUDA runtime is absent before cleanup"
                )
            gc.collect()
            for device in (0, 1):
                with canary_torch.cuda.device(device):
                    canary_torch.cuda.synchronize(device)
            _clear_cublas_workspaces(canary_torch)
            for device in (0, 1):
                with canary_torch.cuda.device(device):
                    canary_torch.cuda.empty_cache()
                    canary_torch.cuda.synchronize(device)
            allocated = [
                int(canary_torch.cuda.memory_allocated(device))
                for device in (0, 1)
            ]
            reserved = [
                int(canary_torch.cuda.memory_reserved(device))
                for device in (0, 1)
            ]
            free = [
                int(canary_torch.cuda.mem_get_info(device)[0])
                for device in (0, 1)
            ]
            canary_cleanup_before_formal = {
                "synchronized": True,
                "gc_collected": True,
                "empty_cache_at_safe_boundary": True,
                "memory_allocated_bytes": allocated,
                "memory_reserved_bytes": reserved,
                "free_bytes": free,
                "runtime_references_released": True,
            }
            if allocated != [0, 0] or reserved != [0, 0]:
                raise B2CalibrationLauncherV1Error(
                    "P4.8e canary memory did not return before formal fresh v0"
                )
        else:
            canary_cleanup_before_formal = {
                "synchronized": True,
                "gc_collected": True,
                "empty_cache_at_safe_boundary": True,
                "memory_allocated_bytes": [0, 0],
                "memory_reserved_bytes": [0, 0],
                "free_bytes": [2 * 1024**3, 2 * 1024**3],
                "runtime_references_released": True,
            }
        formal_session = factory(runtime_config, output / "runtime_config.json")
        formal_initial = formal_session.initial_calibration_identity()
        canary_input = {
            **runtime_canary,
            "status": "passed",
            "session_closed": True,
        }
        canary_isolation = assert_canary_isolated(
            canary_input,
            formal_initial_adapter_sha256=formal_initial["adapter_sha256"],
            formal_policy_version=formal_initial["logical_version"],
        )
        canary_artifact = {
            "schema_version": 1,
            "artifact_kind": "b2_memory_canary_v1",
            "run_id": runtime_run_id,
            "passed": True,
            "risk_schedule_step": 4,
            "prompt_count": 4,
            "selected_response_length": 1024,
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "initial_adapter_sha256": canary_initial["adapter_sha256"],
            "formal_initial_adapter_sha256": formal_initial["adapter_sha256"],
            "minimum_free_bytes_by_gpu": canary_isolation[
                "minimum_free_bytes_by_gpu"
            ],
            "telemetry": canary_telemetry,
            "optimizer_steps_executed_in_throwaway_session": int(
                runtime_canary["optimizer_steps_executed"]
            ),
            "sampler_refresh_count_in_throwaway_session": int(
                runtime_canary["sampler_refresh_count"]
            ),
            "scheduler_steps_executed_in_throwaway_session": int(
                runtime_canary.get("scheduler_step_count", 1)
            ),
            "backbone_backward_calls": int(
                runtime_canary.get("backbone_backward_calls", 4)
            ),
            "backbone_forward_calls": int(
                runtime_canary.get("backbone_forward_calls", 4)
            ),
            "lm_head_chunk_count": int(
                runtime_canary.get("lm_head_chunk_count", 4)
            ),
            "retain_graph_calls": int(runtime_canary.get("retain_graph_calls", 0)),
            "adapter_export_count": int(
                runtime_canary.get("adapter_export_count", 1)
            ),
            "fresh_identity_verifier_count": int(
                runtime_canary.get("fresh_identity_verifier_count", 1)
            ),
            "policy_version_increment_count": int(
                runtime_canary.get("policy_version_increment_count", 1)
            ),
            "canary_session_closed": True,
            "cleanup_before_formal_fresh_v0": canary_cleanup_before_formal,
            "formal_student_rebuilt_fresh_v0": True,
            "canary_state_reused": False,
            "teacher_gradient_tensor_count": int(
                canary_step_record["teacher_gradient_tensor_count"]
            ),
            "base_gradient_tensor_count": int(
                canary_step_record["base_gradient_tensor_count"]
            ),
            "p_old_detached": bool(canary_step_record["p_old_detached"]),
            "initial_registry_count": int(
                runtime_canary["initial_registry_count"]
            ),
            "final_registry_count": int(runtime_canary["final_registry_count"]),
            "initial_model_count": int(runtime_canary["initial_model_count"]),
            "final_model_count": int(runtime_canary["final_model_count"]),
            "real_rollout_executed": bool(
                fixture_summary.get("real_rollout_executed", True)
            ),
            "fixture_valid_completion_tokens": int(
                fixture_summary.get("fixture_valid_completion_tokens", 0)
            ),
            "fixture_valid_completion_tokens_by_prompt": list(
                fixture_summary.get(
                    "fixture_valid_completion_tokens_by_prompt", []
                )
            ),
            "synthetic_target_scoring_count": int(
                fixture_summary.get("synthetic_target_scoring_count", 0)
            ),
            "fixture_prompt_tokens": int(
                fixture_summary.get("fixture_prompt_tokens", 0)
            ),
            "fixture_source": fixture_summary.get("fixture_source"),
            "label_access_count": int(
                fixture_summary.get("label_access_count", 0)
            ),
            "controller_access_count": int(
                fixture_summary.get("controller_access_count", 0)
            ),
            "final_access_count": int(
                fixture_summary.get("final_access_count", 0)
            ),
            "raw_prompt_persisted": False,
            "response_tokens_persisted": False,
            "hidden_gradient_trainable_scope": dict(
                runtime_canary.get("hidden_gradient_trainable_scope", {})
            ),
        }
        _atomic_json(output / "memory_canary.json", canary_artifact)
        store.commit_initial_identity(formal_initial)
        step_end_records: list[Mapping[str, Any]] = []
        for step_index in range(B2_CALIBRATION_STEPS):
            prompts = list(provider(runtime_config, step_index))
            if len(prompts) != 4:
                raise B2CalibrationLauncherV1Error(
                    f"step {step_index + 1} prompt batch is not 2+2"
                )
            record = formal_session.run_b2_calibration_step_v1(
                step_index=step_index,
                prompt_rows=prompts,
                max_new_tokens=1024,
            )
            version = step_index + 1
            if version in {5, 10, 15, 20}:
                checkpoint_started = time.perf_counter()
                formal_session.save_b2_resume_checkpoint_v1(
                    logical_version=version,
                    package_content_sha256=package_audit[
                        "package_content_sha256"
                    ],
                    config_sha256=runtime_config_sha,
                    data_cursor=version * 4,
                )
                if version == 10:
                    reload_identity = formal_session.reload_b2_resume_checkpoint_v1(
                        logical_version=10,
                        package_content_sha256=package_audit[
                            "package_content_sha256"
                        ],
                        config_sha256=runtime_config_sha,
                        data_cursor=40,
                    )
                    store.commit_resume_reload(reload_identity)
                record = deepcopy(dict(record))
                elapsed = time.perf_counter() - checkpoint_started
                timings = deepcopy(dict(record["timings_seconds"]))
                timings["checkpoint"] = float(timings["checkpoint"]) + elapsed
                timings["step"] = float(timings["step"]) + elapsed
                record["timings_seconds"] = timings
            store.commit_step(record)
            step_records.append(record)
            step_end_records.append(formal_session.memory_step_end_record_v1())
            _append_safe_log(
                output / "stdout.log",
                f"optimizer_step={version} policy=v{version} committed=true",
            )
            if version == 6:
                drift = evaluate_six_step_memory_drift(step_end_records)
                _atomic_json(output / "memory_six_step_drift.json", drift)
                if drift["passed"] is not True:
                    raise B2CalibrationLauncherV1Error(
                        "P4.8d six-step memory drift gate failed"
                    )
            if len(step_records) >= 4:
                live_length = evaluate_latest_calibration_length_window(
                    step_records, selected_response_length=1024
                )
                if live_length["passed"] is not True:
                    _atomic_json(
                        output / "length_abort_recommendation.json", live_length
                    )
                    raise B2CalibrationLauncherV1Error(
                        "frozen 1024 calibration length window failed; "
                        "no same-run length switch is allowed"
                    )
        store.commit_final_reload(
            formal_session.final_checkpoint_reload_identity_v1()
        )
        result = {
            "schema_version": 1,
            "artifact_kind": "b2_calibration_worker_status_v1",
            "status": "worker_completed_exactly_20_steps",
            "steps_completed": 20,
            "memory_canary_passed": True,
            "six_step_memory_drift_passed": True,
            "B2_calibration_started": True,
            "B2_calibration_complete": False,
            "B2_formal_authorized": False,
        }
        _atomic_json(output / "worker_status.json", result)
        _append_safe_log(output / "stdout.log", "worker_complete optimizer_steps=20")
        return result
    except Exception as error:
        failure = _memory_worker_failure_classification(error)
        completed = _completed_steps(output)
        _atomic_json(
            output / "worker_status.json",
            {
                "schema_version": 1,
                "artifact_kind": "b2_calibration_worker_status_v1",
                "status": "worker_failed",
                "error_type": type(error).__name__,
                "primary_failure_code": failure["primary_failure_code"],
                "failure_phase": failure["failure_phase"],
                "completed_steps": completed,
                "requested_steps": 20,
                "steps_completed": completed,
                "unmet_success_gates": [
                    "optimizer_step_count_is_not_exactly_20"
                ]
                if completed != 20
                else [],
                "causal_chain": failure["causal_chain"],
                "B2_calibration_complete": False,
                "B2_formal_authorized": False,
            },
        )
        _append_safe_log(
            output / "stdout.log", f"worker_failed type={type(error).__name__}"
        )
        if isinstance(error, B2CalibrationLauncherV1Error):
            raise
        raise B2CalibrationLauncherV1Error(
            f"memory calibration worker failed: {type(error).__name__}:{error}"
        ) from error
    finally:
        if canary_session is not None:
            canary_session.close()
        if formal_session is not None:
            formal_session.close()


__all__ = [
    "_memory_worker_failure_classification",
    "execute_memory_balanced_calibration_worker_v1",
]
