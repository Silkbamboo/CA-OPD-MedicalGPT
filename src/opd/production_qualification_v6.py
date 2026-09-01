"""Fail-stop entrypoint and post-exit finalizer for P4.6.

Importing this module is CPU-safe.  The authorized runtime module is imported
only after formal preflight and the immutable launch/preflight artifacts have
been committed.  Post-exit finalization reopens the artifact graph; it never
accepts a caller-supplied readiness value and never starts B2.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from src.opd import production_qualification_artifacts_v6 as artifact_graph
from src.opd.production_qualification_artifacts_v6 import (
    FULL_PHASES,
    canonical_json_sha256,
    finalize_qualification,
    materialize_b2_calibration_package,
    record_failure,
    record_failure_cleanup,
    sha256_file,
)


QUALIFICATION_PHASES = (
    "formal_preflight",
    "production_backend_binding",
    "micro_v0_normal_guard",
    "micro_v0_wrong_authority_guard",
    "micro_rollout_q",
    "micro_probe_manifest_pre_optimizer",
    "micro_p_old_teacher_scoring",
    "micro_update_v0_to_v1_complete_telemetry",
    "micro_trainer_v1_memory_reload_authority",
    "micro_stable_slot_hotswap_v1",
    "micro_runtime_fresh_trainer_identity",
    "micro_same_path_probe",
    "micro_v1_normal_and_stale_v0",
    "two_step_rollout1_guarded_v1",
    "two_step_p_old_v1_teacher_scoring",
    "two_step_update_v1_to_v2_complete_telemetry",
    "two_step_trainer_v2_memory_reload_authority",
    "two_step_stable_slot_hotswap_v2",
    "two_step_runtime_fresh_trainer_identity",
    "two_step_v2_normal_and_stale_v1",
    "independent_base_teacher_null",
    "length_actual_384_and_derived_256",
    "length_conditional_512",
    "length_decision",
    "runtime_release",
    "resource_cleanup",
    "artifact_graph_readiness",
    "calibration_package_and_b2_authorization_only_if_full_ready",
    "stop_without_starting_b2",
)

_BINDING_FIELDS = (
    "run_id",
    "attempt_id",
    "git_commit",
    "config_sha256",
    "run_card_sha256",
    "schema_sha256",
    "protocol_sha256",
    "backend_binding_sha256",
    "prompt_manifest_sha256",
    "probe_spec_sha256",
    "data_manifest_sha256",
    "isolation",
)


class ProductionQualificationError(RuntimeError):
    """A P4.6 launch/finalization boundary failed closed."""


def qualification_plan(config: Mapping[str, Any]) -> list[str]:
    if (
        config.get("schema_id")
        != "ca-opd/p4.6-combined-production-qualification/v1"
        or config.get("schema_version") != 1
        or config.get("run", {}).get("stage") != "combined_production_qualification"
        or config.get("execution", {}).get("ordered_phases")
        != list(QUALIFICATION_PHASES)
        or config.get("execution", {}).get("fail_stop") is not True
        or config.get("execution", {}).get("automatically_start_b2") is not False
        or config.get("execution", {}).get(
            "generate_calibration_package_after_full_readiness"
        )
        is not True
        or config.get("sampler_refresh", {}).get("candidate_mechanism")
        != "peft_0_17_1_hotswap_stable_slot"
        or config.get("sampler_refresh", {}).get("runtime_slot")
        != "student_active"
    ):
        raise ProductionQualificationError("P4.6 combined qualification plan drift")
    return list(QUALIFICATION_PHASES)


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists() and (candidate / "src/opd").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ProductionQualificationError("cannot bind a lowercase Git commit")
    return result


def _non_placeholder_functions(path: Path) -> set[str]:
    if not path.is_file() or path.is_symlink():
        raise ProductionQualificationError("production runtime source is absent")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ProductionQualificationError("production runtime source is invalid") from error
    implementations: set[str] = set()

    def record_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef, *, prefix: str = ""
    ) -> None:
        meaningful = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        if not meaningful or all(
            isinstance(statement, ast.Pass) for statement in meaningful
        ):
            return
        if len(meaningful) == 1 and isinstance(meaningful[0], ast.Raise):
            raised = meaningful[0].exc
            if (
                isinstance(raised, ast.Call)
                and isinstance(raised.func, ast.Name)
                and raised.func.id in {"NotImplementedError", "RuntimeError"}
            ):
                return
        implementations.add(prefix + node.name)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record_function(node)
        elif isinstance(node, ast.ClassDef):
            implementations.add(node.name)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record_function(member, prefix=f"{node.name}.")
    return implementations


def build_current_backend_binding_manifest(
    config: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    """Recompute the current executable production chain from local sources."""

    root = Path(repo_root).resolve()
    runtime_relative = Path("src/opd/production_qualification_gpu_runtime_v6.py")
    runtime_path = root / runtime_relative
    qualification_symbol = str(
        config["production_binding"]["qualification_runtime_symbol"]
    )
    b2_executor_symbol = "execute_b2_medical_opd_gpu_protocol_v6"
    b2_executor_implementation_symbol = "_execute_b2_calibration_loop"
    source_contracts = (
        (
            runtime_relative,
            (
                qualification_symbol,
                b2_executor_symbol,
                b2_executor_implementation_symbol,
            ),
        ),
        (
            Path("src/opd/production_qualification_v6.py"),
            (
                "build_current_backend_binding_manifest",
                "run_gpu_qualification",
                "finalize_gpu_qualification",
            ),
        ),
        (
            Path("src/opd/production_qualification_artifacts_v6.py"),
            (
                "commit_phase",
                "finalize_qualification",
                "assert_b2_start_authorized",
            ),
        ),
        (
            Path("src/opd/production_qualification_preflight_v6.py"),
            ("preflight",),
        ),
        (
            Path("src/opd/production_qualification_two_step_gpu_v6.py"),
            (
                "execute_two_step_qualification_v6",
                "create_production_two_step_session_v6",
                "create_production_b2_session_v6",
                "create_production_auxiliary_session_v6",
                "ProductionTwoStepSessionV6._action_logprobs",
                "ProductionTwoStepSessionV6.run_b2_calibration_step",
            ),
        ),
        (
            Path("src/opd/production_qualification_aux_gpu_v6.py"),
            (
                "execute_base_teacher_null_v6",
                "execute_length_calibration_v6",
                "execute_b2_calibration_loop_v6",
            ),
        ),
        (
            Path("src/opd/calibration_data.py"),
            ("contains_forbidden_supervision", "render_prompt_text"),
        ),
        (
            Path("src/data/chat.py"),
            ("format_mcq_question",),
        ),
        (
            Path("src/opd/production_backend_binding_v5.py"),
            ("load_production_run_card_v5", "verify_b2_backend_binding"),
        ),
        (
            Path("src/opd/pg_opd_contract.py"),
            (
                "ThreePolicyLogProbBundle",
                "decoupled_corrected_objective",
                "grouped_trajectory_mean",
                "validate_three_policy_bundle",
            ),
        ),
        (
            Path("src/opd/pg_opd_validation.py"),
            ("audit_optimizer_update",),
        ),
        (
            Path("src/opd/production_qualification_contract_v6.py"),
            (
                "build_probe_manifest",
                "build_probe_spec",
                "validate_base_null",
                "decide_response_length",
            ),
        ),
        (
            Path("src/opd/production_qualification_telemetry_v6.py"),
            (
                "build_reconstruction_telemetry",
                "validate_reconstruction_telemetry",
            ),
        ),
        (
            Path("src/opd/production_qualification_prompts_v6.py"),
            ("load_frozen_prompt_group",),
        ),
        (
            Path("src/opd/production_sampler_identity_v5.py"),
            (
                "build_adapter_identity_manifest",
                "guard_sampler_operation",
                "trainer_authority_from_manifest",
            ),
        ),
        (
            Path("src/opd/production_sampler_refresh_v5.py"),
            (
                "adapter_artifact_identity",
                "refresh_stable_slot",
                "runtime_identity_from_peft",
            ),
        ),
        (
            Path("src/opd/rollout_correction_adapter.py"),
            ("native_decoupled_token_is",),
        ),
        (
            Path("src/opd/rollout_probability.py"),
            ("validate_rollout_behavior_provenance",),
        ),
        (
            Path("src/opd/scorer_gpu_calibration.py"),
            ("_apply_determinism", "_release"),
        ),
        (
            Path("src/opd/production.py"),
            ("resolve_opd_source_files",),
        ),
    )
    source_chain: list[dict[str, Any]] = []
    for relative, required_symbols in source_contracts:
        path = root / relative
        implementations = _non_placeholder_functions(path)
        missing = sorted(set(required_symbols) - implementations)
        if missing:
            raise ProductionQualificationError(
                "production executable chain is incomplete: " + ",".join(missing)
            )
        source_chain.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "required_symbols": list(required_symbols),
            }
        )
    b2_config = root / str(config["production_binding"]["b2_config_path"])
    b2_card = root / str(config["production_binding"]["b2_run_card_path"])
    if sha256_file(b2_config) != config["production_binding"]["b2_config_sha256"]:
        raise ProductionQualificationError("production B2 config SHA drift")
    if sha256_file(b2_card) != config["production_binding"]["b2_run_card_sha256"]:
        raise ProductionQualificationError("production B2 run-card SHA drift")
    return {
        "binding_version": "p4.6-current-executable-chain-v3",
        "production_backend_id": config["production_binding"]["backend_id"],
        "b2_config_path": str(config["production_binding"]["b2_config_path"]),
        "b2_config_sha256": sha256_file(b2_config),
        "b2_run_card_path": str(config["production_binding"]["b2_run_card_path"]),
        "b2_run_card_sha256": sha256_file(b2_card),
        "qualification_runtime_path": runtime_relative.as_posix(),
        "qualification_runtime_sha256": sha256_file(runtime_path),
        "qualification_runtime_symbol": qualification_symbol,
        "b2_executor_symbol": b2_executor_symbol,
        "b2_executor_implementation_symbol": b2_executor_implementation_symbol,
        "executable_source_chain": source_chain,
        "refresh_implementation": config["sampler_refresh"]["candidate_mechanism"],
        "runtime_slot": config["sampler_refresh"]["runtime_slot"],
        "model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "dtype": config["model"]["dtype"],
        "attention_implementation": config["model"]["attention_implementation"],
        "protocol_sha256": config["validation"]["config_sha256"],
    }


def build_artifact_bindings(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    preflight_result: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build immutable authorities from checked-in files, never runtime claims."""

    path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root(path)
    card = root / "configs/run_cards" / f"{config['run']['run_id']}.json"
    expected = {
        "config_sha256": sha256_file(path),
        "run_card_sha256": sha256_file(card),
        "artifact_schema_sha256": sha256_file(
            root / str(config["artifacts"]["schema_path"])
        ),
        "protocol_sha256": sha256_file(
            root / str(config["validation"]["config_path"])
        ),
    }
    for key, value in expected.items():
        if preflight_result.get(key) != value:
            raise ProductionQualificationError(f"preflight identity drift: {key}")
    prompt_manifest_sha = str(
        config["prompt_selection"]["selection_manifest_sha256"]
    )
    data_manifest_sha = str(config["prompt_selection"]["opd_manifest_sha256"])
    backend_manifest = build_current_backend_binding_manifest(config, repo_root=root)
    backend_manifest_sha = canonical_json_sha256(backend_manifest)
    if preflight_result.get("backend_binding_sha256") != backend_manifest_sha:
        raise ProductionQualificationError(
            "preflight identity drift: backend_binding_sha256"
        )
    bindings = {
        "run_id": str(config["run"]["run_id"]),
        "attempt_id": str(config["run"]["attempt_id"]),
        "git_commit": _git_commit(root),
        "config_sha256": expected["config_sha256"],
        "run_card_sha256": expected["run_card_sha256"],
        "schema_sha256": expected["artifact_schema_sha256"],
        "protocol_sha256": expected["protocol_sha256"],
        "backend_binding_sha256": backend_manifest_sha,
        "prompt_manifest_sha256": prompt_manifest_sha,
        "probe_spec_sha256": str(
            config["fixed_action_probe"]["probe_spec_sha256"]
        ),
        "data_manifest_sha256": data_manifest_sha,
        "isolation": dict(config["isolation"]),
    }
    if set(bindings) != set(_BINDING_FIELDS):
        raise ProductionQualificationError("artifact binding fields drift")
    return bindings


def _bootstrap_artifact_graph(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    bindings: Mapping[str, Any],
    preflight_result: Mapping[str, Any],
) -> None:
    output = Path(str(config["run"]["output_dir"]))
    root = _repo_root(config_path)
    backend_manifest = build_current_backend_binding_manifest(config, repo_root=root)
    backend_manifest_sha = canonical_json_sha256(backend_manifest)
    if backend_manifest_sha != bindings["backend_binding_sha256"]:
        raise ProductionQualificationError("backend binding changed after preflight")
    artifact_graph.initialize_qualification_artifacts(
        output,
        bindings=bindings,
        mode="full",
        sources={
            "config": config_path,
            "run_card": root
            / "configs/run_cards"
            / f"{config['run']['run_id']}.json",
            "artifact_schema": root / str(config["artifacts"]["schema_path"]),
            "protocol": root / str(config["validation"]["config_path"]),
            "prompt_manifest": root
            / str(config["prompt_selection"]["selection_manifest_path"]),
        },
        backend_binding=backend_manifest,
    )
    launch_payload = {
        "status": "pass",
        "stage": "combined_production_qualification",
        "production_backend_id": config["production_binding"]["backend_id"],
        "refresh_implementation": config["sampler_refresh"][
            "candidate_mechanism"
        ],
        "runtime_slot": config["sampler_refresh"]["runtime_slot"],
        "production_backend_binding_sha256": backend_manifest_sha,
        "production_backend_binding": backend_manifest,
        "gpu_execution_authorized": True,
        "B2_authorized": False,
        "B2_started": False,
        **dict(config["isolation"]),
    }
    artifact_graph.commit_phase(
        output,
        bindings=bindings,
        mode="full",
        phase_id="launch_record",
        ordinal=0,
        payload=launch_payload,
        metric={"status": "pass", "launch_envelope_committed": True},
    )
    preflight_payload = {
        "status": "pass",
        "formal_preflight_passed": True,
        "production_backend_binding_verified": bool(
            preflight_result.get("production_backend_binding_verified", True)
        ),
        "preflight_evidence_sha256": canonical_json_sha256(dict(preflight_result)),
        "gpu_inventory_count": len(preflight_result.get("gpu_inventory", [])),
        "B2_authorized": False,
        "B2_started": False,
        **dict(config["isolation"]),
    }
    artifact_graph.commit_phase(
        output,
        bindings=bindings,
        mode="full",
        phase_id="preflight",
        ordinal=1,
        payload=preflight_payload,
        metric={"status": "pass", "formal_preflight_passed": True},
    )


def _load_bindings(output: Path) -> dict[str, Any]:
    launch = output / "launch_record.json"
    if not launch.is_file() or launch.is_symlink():
        raise ProductionQualificationError("launch_record.json is absent")
    try:
        document = json.loads(launch.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionQualificationError("launch_record.json is invalid") from error
    bindings = {key: document.get(key) for key in _BINDING_FIELDS}
    if any(value is None for value in bindings.values()):
        raise ProductionQualificationError("launch artifact bindings are incomplete")
    return bindings


def run_gpu_qualification(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Bootstrap evidence, then and only then import the authorized GPU runtime."""

    qualification_plan(config)
    authorization = config["authorization"]
    if os.environ.get(authorization["environment_variable"]) != authorization[
        "required_value"
    ]:
        raise ProductionQualificationError("P4.6 GPU authorization is absent")
    from src.opd.production_qualification_preflight_v6 import preflight

    result = preflight(
        config,
        config_path=config_path,
        execute_gpu=True,
        require_clean_git=True,
    )
    bindings = build_artifact_bindings(
        config,
        config_path=config_path,
        preflight_result=result,
    )
    output = Path(str(config["run"]["output_dir"]))
    _bootstrap_artifact_graph(
        config,
        config_path=config_path,
        bindings=bindings,
        preflight_result=result,
    )
    try:
        # This is the first import allowed to import Torch/Transformers/PEFT or
        # touch CUDA.  The two fail-closed bootstrap phases already exist.
        from src.opd.production_qualification_gpu_runtime_v6 import (
            execute_production_qualification_gpu_protocol_v6,
        )

        runtime_result = execute_production_qualification_gpu_protocol_v6(
            config,
            config_path=config_path,
            artifact_bindings=bindings,
            artifact_mode="full",
        )
        if not isinstance(runtime_result, Mapping):
            raise ProductionQualificationError("GPU runtime returned no evidence object")
        if runtime_result.get("B2_started") is not False:
            raise ProductionQualificationError("qualification runtime attempted to start B2")
        return dict(runtime_result)
    except BaseException as error:
        failure_path = output / "failure.json"
        if not failure_path.exists():
            record_failure(
                output,
                bindings=bindings,
                mode="full",
                reason=f"authorized_runtime_error:{type(error).__name__}",
            )
        raise


def _cleanup_observation() -> dict[str, Any]:
    """Observe post-process GPU/worker state without importing a GPU library."""

    memory_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    memory = [
        int(line.strip())
        for line in memory_query.stdout.splitlines()
        if line.strip()
    ]
    compute_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    compute_pids = [
        int(line.strip())
        for line in compute_query.stdout.splitlines()
        if line.strip() and line.strip().isdigit()
    ]
    processes = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    residual_workers: list[int] = []
    markers = ("raylet", "ray::", "vllm", "torchrun", "verl.workers")
    for line in processes:
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        if pid in {os.getpid(), os.getppid()}:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            residual_workers.append(pid)
    return {
        "gpu_memory_used_mib": memory,
        "compute_pids": sorted(set(compute_pids)),
        "residual_workers": sorted(set(residual_workers)),
    }


def _record_failure_once(
    output: Path,
    *,
    bindings: Mapping[str, Any],
    reason: str,
) -> None:
    if not (output / "failure.json").exists():
        record_failure(
            output,
            bindings=bindings,
            mode="full",
            reason=reason,
        )


def _record_failure_cleanup_observation(
    output: Path,
    *,
    bindings: Mapping[str, Any],
    runtime_exit_code: int,
    observation: Mapping[str, Any] | None = None,
    observation_error: str | None = None,
) -> None:
    observed = observation
    error = observation_error
    if observed is None and error is None:
        try:
            observed = _cleanup_observation()
        except (OSError, ValueError, subprocess.SubprocessError) as caught:
            error = f"{type(caught).__name__}:{caught}"
    record_failure_cleanup(
        output,
        bindings=bindings,
        runtime_exit_code=runtime_exit_code,
        observation=observed,
        observation_error=error,
    )


def _commit_post_exit_phases(
    config: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    runtime_exit_code: int,
) -> None:
    output = Path(str(config["run"]["output_dir"]))
    failure_path = output / "failure.json"
    if runtime_exit_code != 0:
        _record_failure_once(
            output,
            bindings=bindings,
            reason=f"runtime_exit_code:{runtime_exit_code}",
        )
        _record_failure_cleanup_observation(
            output,
            bindings=bindings,
            runtime_exit_code=runtime_exit_code,
        )
        return
    if failure_path.exists():
        _record_failure_cleanup_observation(
            output,
            bindings=bindings,
            runtime_exit_code=runtime_exit_code,
        )
        return
    required_before_cleanup = FULL_PHASES[: FULL_PHASES.index("cleanup")]
    missing = [
        phase
        for phase in required_before_cleanup
        if not (output / f"{phase}.json").is_file()
    ]
    if missing:
        _record_failure_once(
            output,
            bindings=bindings,
            reason="runtime_success_missing_phases:" + ",".join(missing),
        )
        _record_failure_cleanup_observation(
            output,
            bindings=bindings,
            runtime_exit_code=runtime_exit_code,
        )
        return
    cleanup_path = output / "cleanup.json"
    if not cleanup_path.exists():
        try:
            observation = _cleanup_observation()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            _record_failure_once(
                output,
                bindings=bindings,
                reason=f"cleanup_probe_error:{type(error).__name__}",
            )
            _record_failure_cleanup_observation(
                output,
                bindings=bindings,
                runtime_exit_code=runtime_exit_code,
                observation_error=f"{type(error).__name__}:{error}",
            )
            return
        clean = (
            observation["gpu_memory_used_mib"] == [0, 0]
            and observation["compute_pids"] == []
            and observation["residual_workers"] == []
        )
        if not clean:
            _record_failure_once(
                output,
                bindings=bindings,
                reason="cleanup_gate_failed:" + canonical_json_sha256(observation),
            )
            _record_failure_cleanup_observation(
                output,
                bindings=bindings,
                runtime_exit_code=runtime_exit_code,
                observation=observation,
            )
            return
        artifact_graph.commit_phase(
            output,
            bindings=bindings,
            mode="full",
            phase_id="cleanup",
            ordinal=FULL_PHASES.index("cleanup"),
            payload={
                "status": "pass",
                "runtime_exit_code": runtime_exit_code,
                **observation,
            },
            metric={"status": "pass", "cleanup_complete": True},
        )
    terminal_path = output / "terminal_summary.json"
    if not terminal_path.exists():
        artifact_graph.commit_phase(
            output,
            bindings=bindings,
            mode="full",
            phase_id="terminal_summary",
            ordinal=FULL_PHASES.index("terminal_summary"),
            payload={
                "status": "pass",
                "qualification_phases_complete": True,
                "artifact_derived_readiness_pending": True,
                "calibration_package_pending_full_readiness": True,
                "B2_started": False,
                **dict(config["isolation"]),
            },
            metric={"status": "pass", "B2_started": False},
        )
    summary_path = output / "summary.json"
    if terminal_path.is_file() and not summary_path.exists():
        artifact_graph.write_terminal_summary_alias(output)


def finalize_gpu_qualification(
    config: Mapping[str, Any], *, runtime_exit_code: int
) -> dict[str, Any]:
    """Finalize from disk, write readiness last, then optionally materialize B2 files."""

    qualification_plan(config)
    output = Path(str(config["run"]["output_dir"]))
    if (output / "readiness.json").is_file():
        sealed = dict(
            artifact_graph.derive_qualification_readiness(output, mode="full")
        )
        sealed["calibration_package_materialized"] = bool(
            sealed.get("ready")
            and sealed.get("OPD_scoring_backend_ready") is True
            and sealed.get("B2_authorized") is True
            and (output / "b2_authorization.json").is_file()
        )
        if sealed["calibration_package_materialized"]:
            sealed["calibration_package_dir"] = str(
                config["run"]["generated_b2_package_dir"]
            )
        sealed["B2_started"] = False
        return sealed
    bindings = _load_bindings(output)
    _commit_post_exit_phases(
        config,
        bindings=bindings,
        runtime_exit_code=runtime_exit_code,
    )
    evidence = finalize_qualification(output, mode="full")
    result = dict(evidence)
    result["calibration_package_materialized"] = False
    result["B2_started"] = False
    if (
        runtime_exit_code == 0
        and evidence.get("ready") is True
        and evidence.get("authorization_eligibility") is True
        and not (output / "failure.json").exists()
    ):
        try:
            package = materialize_b2_calibration_package(
                output, config["run"]["generated_b2_package_dir"]
            )
            if package.get("B2_started") is not False or package.get(
                "authorization", {}
            ).get("B2_authorized") is not True:
                raise ProductionQualificationError(
                    "calibration materialization did not produce a closed B2 authorization"
                )
        except Exception as error:
            _record_failure_once(
                output,
                bindings=bindings,
                reason=(
                    "b2_calibration_materialization_failed:"
                    f"{type(error).__name__}:{error}"
                ),
            )
            final = finalize_qualification(output, mode="full")
            result = dict(final)
            result["calibration_package_materialized"] = False
            result["materialization_error"] = f"{type(error).__name__}:{error}"
            result["B2_authorized"] = False
            result["B2_started"] = False
            return result
        final = finalize_qualification(output, mode="full")
        result = dict(final)
        result["calibration_package_materialized"] = bool(
            final.get("ready")
            and final.get("OPD_scoring_backend_ready") is True
            and final.get("B2_authorized") is True
        )
        if result["calibration_package_materialized"]:
            result["calibration_package_dir"] = str(
                config["run"]["generated_b2_package_dir"]
            )
    else:
        final = finalize_qualification(output, mode="full")
        result = dict(final)
        result["calibration_package_materialized"] = False
        result["B2_authorized"] = False
        result["B2_started"] = False
    return result


def _dry_run(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready_waiting_for_gpu_combined_qualification",
        "phases": qualification_plan(config),
        "gpu_used": False,
        "loaded_real_model": False,
        "B2_authorized": False,
        "B2_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P4.6 combined production qualification"
    )
    parser.add_argument("--config", required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--runtime-exit-code", type=int)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ProductionQualificationError("P4.6 config must be a mapping")
    if args.finalize:
        if args.runtime_exit_code is None:
            parser.error("--finalize requires --runtime-exit-code")
        result = finalize_gpu_qualification(
            config, runtime_exit_code=args.runtime_exit_code
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("B2_authorized") is True else 1
    if args.runtime_exit_code is not None:
        parser.error("--runtime-exit-code is valid only with --finalize")
    if args.execute:
        result = run_gpu_qualification(config, config_path=path)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.preflight:
        from src.opd.production_qualification_preflight_v6 import preflight

        result = preflight(
            config,
            config_path=path,
            execute_gpu=False,
            require_clean_git=True,
        )
    else:
        result = _dry_run(config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
