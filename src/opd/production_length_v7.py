"""P4.7 length-only qualification CLI and formal GPU coordinator.

Imports are CPU safe.  The real model backend is constructed only after the
formal GPU preflight and explicit launcher authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from typing import Callable

import yaml

from src.opd.production_length_artifacts_v7 import (
    LengthArtifactV7Error,
    atomic_write_privacy_safe_json,
    commit_length_qualification,
    derive_length_readiness,
    sha256_file,
    validate_length_telemetry,
)
from src.opd.production_length_contract_v7 import (
    CONDITIONAL_4096_CANDIDATES,
    PRIMARY_CANDIDATES,
    build_explicit_length_telemetry,
    build_length_telemetry,
    canonical_json_sha256,
    evaluate_conditional_4096,
)
from src.opd.production_length_gpu_runtime_v7 import (
    build_b2_calibration_package,
    execute_bounded_generation_plan,
    materialize_b2_calibration_package,
)


class ProductionLengthV7Error(RuntimeError):
    """The formal P4.7 state machine failed closed."""


def _fail(message: str) -> None:
    raise ProductionLengthV7Error(message)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProductionLengthV7Error(
            f"cannot read P4.7 config: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail("P4.7 config is not an object")
    return value


def _load_runtime_prompt_rows(
    config: Mapping[str, Any], *, repo_root: Path
) -> list[dict[str, Any]]:
    from src.opd.production_qualification_prompts_v6 import (
        load_frozen_prompt_group,
    )

    shadow = {
        "run": {"run_id": config["parent_reuse"]["run_id"], "seed": 42},
        "prompt_selection": dict(config["prompt_selection"]),
    }
    raw_rows = load_frozen_prompt_group(shadow, "length", repo_root=repo_root)
    result: list[dict[str, Any]] = []
    for order, row in enumerate(raw_rows):
        result.append(
            {
                "sample_id": str(row["sample_id"]),
                "prompt_hash": str(row["content_hash"]),
                "source": str(row["target_role"]),
                "frozen_order": order,
                "_runtime_source_row": row,
            }
        )
    if len(result) != 16:
        _fail("frozen length prompt group is not 16 rows")
    return result


def _prefix_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in ("medical_opd_o1", "medical_opd_cmb"):
        row = next((item for item in rows if item.get("source") == source), None)
        if row is None:
            _fail("prefix probe source is absent")
        selected.append(dict(row))
    return selected


def _batch_resources(backend: Any, count: int) -> dict[str, Any]:
    history = getattr(backend, "batch_resource_history", None)
    if not isinstance(history, list) or len(history) < count:
        _fail("GPU backend resource telemetry is absent")
    selected = history[-count:]
    elapsed = sum(float(item["elapsed_seconds"]) for item in selected)
    tokens = sum(int(item["actual_generated_tokens"]) for item in selected)
    return {
        "elapsed_seconds": elapsed,
        "actual_generated_tokens": tokens,
        "peak_gpu_memory_bytes": max(
            int(item["peak_gpu_memory_bytes"]) for item in selected
        ),
    }


def _decoding_bundle_sha(config: Mapping[str, Any], caps: Sequence[int]) -> str:
    invariant = dict(config["generation"])
    return canonical_json_sha256(
        {"prefix_invariant_generation": invariant, "actual_caps": list(caps)}
    )


def _build_telemetry(
    generation_values: Mapping[int, Sequence[Mapping[str, Any]]],
    strategy: str,
    *,
    config: Mapping[str, Any],
    backend: Any,
    attestation_sha256: str,
    candidates: Sequence[int],
    actual_cap: int,
) -> dict[str, Any]:
    resource_count = 1 if strategy == "derived_candidates" else len(candidates)
    resources = _batch_resources(backend, resource_count)
    common = {
        "run_id": config["run"]["run_id"],
        "actual_generation_cap": actual_cap,
        "candidates": candidates,
        "parent_p4_6_binding_sha256": attestation_sha256,
        "generation_backend_identity": "transformers_generate_full_support",
        "model_revision": config["model"]["revision"],
        "base_revision": config["model"]["revision"],
        "adapter_revision": "p4.6-v2",
        "student_policy_version": "v2",
        "runtime_adapter_sha256": config["parent_reuse"]["v2"][
            "aggregate_tensor_sha256"
        ],
        "decoding_config_sha256": _decoding_bundle_sha(
            config, sorted(generation_values)
        ),
        "elapsed_seconds": resources["elapsed_seconds"],
        "peak_gpu_memory_bytes": resources["peak_gpu_memory_bytes"],
        "estimated_cost_cny": float(
            config["resources"].get("estimated_cost_cny_upper_bound", 0.0)
        ),
        "actual_cost_cny": None,
    }
    if strategy == "derived_candidates":
        records = generation_values.get(actual_cap)
        if records is None:
            _fail("derived generation does not contain the actual cap")
        return build_length_telemetry(records, **common)
    if strategy != "explicit_independent_generation":
        _fail("unknown generation strategy")
    return build_explicit_length_telemetry(generation_values, **common)


def _schema_validate(config: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    root = Path(__file__).resolve().parents[2]
    schema_path = root / config["artifacts"]["schema_path"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(value)


def _append_metric(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _gpu_idle_evidence() -> dict[str, Any]:
    """Probe GPUs only after the CUDA worker process has exited."""

    used: list[int] = []
    compute_pids: list[str] = []
    try:
        used = [
            int(line.strip())
            for line in subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if line.strip()
        ]
        raw_pids = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        compute_pids = [item.strip() for item in raw_pids if item.strip()]
    except BaseException:
        pass
    residual: list[int] = []
    try:
        for line in subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines():
            lowered = line.lower()
            if any(
                token in lowered
                for token in ("ray::", "vllm", "torchrun", "-m verl", "/verl/")
            ):
                residual.append(int(line.strip().split(None, 1)[0]))
    except BaseException:
        residual = [-1]
    complete = bool(
        len(used) == 2
        and all(value <= 16 for value in used)
        and not compute_pids
        and not residual
    )
    return {
        "gpu_memory_used_mib": used,
        "compute_pids": compute_pids,
        "residual_worker_pids": residual,
        "idle": complete,
    }


def _release_worker_backend(backend: Any | None) -> dict[str, Any]:
    released = backend is not None
    error_code: str | None = None
    if backend is not None:
        try:
            backend.close()
        except BaseException as error:
            released = False
            error_code = type(error).__name__
    return {
        "schema_version": 7,
        "artifact_kind": "p4_7_gpu_worker_release",
        "status": "released" if released else "release_failed",
        "models_released": released,
        "release_error_code": error_code,
        "B2_started": False,
    }


def _root_index(output: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(output).as_posix()
        not in {"artifact_index.json", "readiness.json"}
    )
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "schema_version": 7,
        "artifact_kind": "p4_7_length_final_index",
        "run_id": "qwen3-4b-length-qualification-v7-seed42",
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def _commit_and_reverify_root_index(output: Path) -> str:
    """Atomically seal and independently re-open the complete root index."""

    index = _root_index(output)
    path = output / "artifact_index.json"
    atomic_write_privacy_safe_json(path, index)
    reopened = _read_json_object(path)
    if reopened != index or reopened.get("artifact_count") != len(
        reopened.get("artifacts", [])
    ):
        _fail("root artifact index did not re-open identically")
    seen: set[str] = set()
    for entry in reopened["artifacts"]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            _fail("root artifact index entry is malformed")
        relative = str(entry["path"])
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            _fail("root artifact index path is invalid")
        seen.add(relative)
        item = output / relative
        if (
            item.is_symlink()
            or not item.is_file()
            or entry.get("sha256") != sha256_file(item)
            or entry.get("size_bytes") != item.stat().st_size
        ):
            _fail("root artifact index SHA/size did not revalidate")
    return sha256_file(path)


def _commit_and_reverify_readiness(
    output: Path, readiness: Mapping[str, Any]
) -> dict[str, Any]:
    path = output / "readiness.json"
    atomic_write_privacy_safe_json(path, readiness)
    reopened = _read_json_object(path)
    if reopened != dict(readiness):
        _fail("root readiness did not re-open identically")
    return reopened


def _package_bindings(
    config: Mapping[str, Any], selection_sha256: str
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = config["parent_reuse"]
    return {
        "git_commit": commit,
        "backend_id": config["production_binding"]["backend_id"],
        "protocol_sha256": sha256_file(root / config["protocol"]["decision_path"]),
        "data_manifest_sha256": config["prompt_selection"]["opd_manifest_sha256"],
        "teacher_manifest_sha256": config[
            "teacher_binding_for_future_b2_package"
        ]["manifest_sha256"],
        "teacher_adapter_sha256": config[
            "teacher_binding_for_future_b2_package"
        ]["adapter_sha256"],
        "base_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "parent_evidence_index_sha256": parent["evidence_index_sha256"],
        "parent_final_index_sha256": parent["final_index_sha256"],
        "production_two_step_sha256": parent["core_artifact_sha256"]["refresh_v2"],
        "authority_v1_sha256": parent["core_artifact_sha256"]["authority_v1"],
        "authority_v2_sha256": parent["core_artifact_sha256"]["authority_v2"],
        "base_null_sha256": parent["core_artifact_sha256"]["base_null"],
        "length_decision_sha256": selection_sha256,
        "seed": 42,
        "estimated_steps": 20,
        "checkpoint_strategy": config["b2_package"]["checkpoint_strategy"],
        "estimated_cost_cny": config["resources"].get(
            "estimated_b2_20_step_cost_cny"
        ),
        "actual_cost_cny": None,
    }


def _derive_worker_terminal_code(phase_results: Mapping[str, Any]) -> str:
    primary_status = (
        phase_results.get("primary", {}).get("qualification", {}).get("status")
    )
    conditional_status = (
        phase_results.get("conditional", {}).get("qualification", {}).get("status")
    )
    artifact_failures = {
        "artifact_write_failed",
        "artifact_schema_failed",
        "artifact_sha_failed",
    }
    if conditional_status in artifact_failures:
        return str(conditional_status)
    if primary_status in artifact_failures:
        return str(primary_status)
    if primary_status == "length_frozen" or conditional_status == "length_frozen":
        return "worker_complete"
    if conditional_status == "no_length_candidate_passed":
        return "needs_post_4096_evidence_review"
    if isinstance(primary_status, str):
        return primary_status
    return "worker_incomplete"


def _conditional_resource_preflight(
    config: Mapping[str, Any],
    *,
    output: Path,
    backend: Any,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    """Recompute bounded disk/GPU headroom before the single 4096 attempt."""

    resources = config["resources"]
    free_bytes = int(shutil.disk_usage(output.parent).free)
    projected_increment = int(resources["projected_peak_increment_bytes"])
    required_free = int(
        float(resources["minimum_projected_free_at_peak_gib"]) * 1024**3
    )
    projected_free = free_bytes - projected_increment
    disk_passed = projected_free > required_free
    minimum_gpu_free = int(resources["conditional_4096_min_free_gpu_bytes"])
    gpu: dict[str, Any]
    if disk_passed:
        gpu = dict(
            backend.conditional_4096_resource_preflight(
                max_prompt_tokens=max_prompt_tokens,
                actual_cap=4096,
                minimum_free_bytes=minimum_gpu_free,
            )
        )
    else:
        gpu = {
            "schema_version": 7,
            "artifact_kind": "p4_7_conditional_4096_gpu_resource_preflight",
            "passed": False,
            "not_run_reason": "disk_preflight_failed",
        }
    gpu_passed = gpu.get("passed") is True
    return {
        "schema_version": 7,
        "artifact_kind": "p4_7_conditional_4096_resource_preflight",
        "disk": {
            "free_bytes": free_bytes,
            "projected_increment_bytes": projected_increment,
            "projected_free_bytes": projected_free,
            "required_strictly_greater_than_bytes": required_free,
            "passed": disk_passed,
        },
        "gpu": gpu,
        "disk_preflight_passed": disk_passed,
        "gpu_preflight_passed": gpu_passed,
        "max_prompt_tokens": max_prompt_tokens,
        "actual_generation_cap": 4096,
        "passed": bool(disk_passed and gpu_passed),
    }


def execute_gpu_length_qualification(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """CUDA worker: generate and durably commit phase evidence, then release.

    This process never seals root readiness and never builds a B2 package.
    Those authority-bearing operations belong to the CPU-only finalizer after
    this process (and therefore its CUDA context) has exited.
    """

    from src.opd.production_length_gpu_backend_v7 import (
        ProductionLengthGpuBackendV7,
    )

    from src.opd.production_length_preflight_v7 import preflight

    preflight_result = preflight(config_path, execute_gpu=True)
    output = Path(config["run"]["output_dir"])
    output.mkdir(parents=True)
    metrics_path = output / "metrics.jsonl"
    backend: Any | None = None
    terminal_code: str | None = None
    phase_results: dict[str, Any] = {}
    attestation_path = Path(__file__).resolve().parents[2] / config["parent_reuse"][
        "parent_reuse_attestation_path"
    ]
    attestation_sha = sha256_file(attestation_path)
    atomic_write_privacy_safe_json(output / "preflight.json", preflight_result)
    _append_metric(metrics_path, {"phase": "preflight", "status": "pass"})
    try:
        backend = ProductionLengthGpuBackendV7(
            config, repo_root=Path(__file__).resolve().parents[2]
        )
        identity = backend.identity_evidence()
        atomic_write_privacy_safe_json(output / "v2_reload_identity.json", identity)
        _append_metric(metrics_path, {"phase": "v2_reload_identity", "status": "pass"})
        rows = _load_runtime_prompt_rows(
            config, repo_root=Path(__file__).resolve().parents[2]
        )

        def persist_prefix(evidence: Mapping[str, Any]) -> None:
            atomic_write_privacy_safe_json(output / "prefix_equivalence.json", evidence)
            _append_metric(
                metrics_path,
                {"phase": "prefix_equivalence", "status": "pass", "passed": evidence["passed"]},
            )

        def evaluate_primary(values: Mapping[int, Sequence[Mapping[str, Any]]], strategy: str) -> dict[str, Any]:
            telemetry = _build_telemetry(
                values,
                strategy,
                config=config,
                backend=backend,
                attestation_sha256=attestation_sha,
                candidates=PRIMARY_CANDIDATES,
                actual_cap=2048,
            )
            try:
                _schema_validate(config, telemetry)
            except BaseException as error:
                raise LengthArtifactV7Error("artifact_schema_failed") from error
            qualification = commit_length_qualification(output / "primary", telemetry)
            disk_reverified = False
            gate: dict[str, Any] = {
                "allowed": False,
                "reasons": ["primary_commit_not_reverified_no_candidate"],
                "automatic_further_escalation": False,
            }
            if qualification.get("status") == "no_length_candidate_passed":
                disk_readiness = derive_length_readiness(output / "primary")
                disk_telemetry = validate_length_telemetry(
                    json.loads(
                        (output / "primary" / "length_telemetry.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
                disk_reverified = bool(
                    disk_readiness.get("status") == "no_length_candidate_passed"
                    and disk_readiness.get("ready") is False
                    and disk_readiness.get("failure_reasons")
                    == ["no_length_candidate_passed"]
                    and (output / "primary" / "artifact_index.json").is_file()
                )
                if disk_reverified:
                    maximum_prompt_tokens = max(
                        int(row["prompt_token_count"])
                        for row in values[2048]
                    )
                    resource_preflight = _conditional_resource_preflight(
                        config,
                        output=output,
                        backend=backend,
                        max_prompt_tokens=maximum_prompt_tokens,
                    )
                    atomic_write_privacy_safe_json(
                        output / "conditional_4096_preflight.json",
                        resource_preflight,
                    )
                    gate = evaluate_conditional_4096(
                        disk_telemetry,
                        eos_stop_config_valid=(
                            identity.get("eos_stop_config_verified") is True
                        ),
                        model_context_limit=int(config["model"]["context_limit"]),
                        max_prompt_tokens=maximum_prompt_tokens,
                        disk_preflight_passed=resource_preflight[
                            "disk_preflight_passed"
                        ],
                        gpu_preflight_passed=resource_preflight[
                            "gpu_preflight_passed"
                        ],
                        isolation=config["isolation"],
                    )
                    atomic_write_privacy_safe_json(
                        output / "conditional_4096_eligibility.json", gate
                    )
            qualification = {**qualification, "disk_reverified": disk_reverified}
            phase_results["primary"] = {
                "telemetry": telemetry,
                "qualification": qualification,
                "gate": gate,
            }
            _append_metric(
                metrics_path,
                {"phase": "primary_length", "status": qualification["status"]},
            )
            return {"qualification": qualification, "conditional_4096_gate": gate}

        def evaluate_conditional(values: Mapping[int, Sequence[Mapping[str, Any]]], strategy: str) -> dict[str, Any]:
            telemetry = _build_telemetry(
                values,
                strategy,
                config=config,
                backend=backend,
                attestation_sha256=attestation_sha,
                candidates=CONDITIONAL_4096_CANDIDATES,
                actual_cap=4096,
            )
            try:
                _schema_validate(config, telemetry)
            except BaseException as error:
                raise LengthArtifactV7Error("artifact_schema_failed") from error
            qualification = commit_length_qualification(
                output / "conditional_4096", telemetry
            )
            phase_results["conditional"] = {
                "telemetry": telemetry,
                "qualification": qualification,
            }
            _append_metric(
                metrics_path,
                {"phase": "conditional_4096", "status": qualification["status"]},
            )
            return {"qualification": qualification}

        plan = execute_bounded_generation_plan(
            backend=backend,
            prefix_probe_rows=_prefix_rows(rows),
            qualification_rows=rows,
            base_seed=42,
            evaluate_primary=evaluate_primary,
            evaluate_conditional=evaluate_conditional,
            persist_prefix_evidence=persist_prefix,
        )
        summary = {
            "schema_version": 7,
            "artifact_kind": "p4_7_bounded_generation_summary",
            "generation_strategy": plan["generation_strategy"],
            "prefix_passed": plan["prefix_equivalence"]["passed"],
            "conditional_4096_executed": plan["conditional_4096_executed"],
            "automatic_further_escalation": False,
        }
        atomic_write_privacy_safe_json(output / "generation_summary.json", summary)
    except BaseException as error:
        terminal_code = str(getattr(error, "code", f"runtime_{type(error).__name__}"))
    release = _release_worker_backend(backend)
    atomic_write_privacy_safe_json(output / "worker_release.json", release)
    _append_metric(metrics_path, {"phase": "worker_release", "status": release["status"]})
    if not release["models_released"]:
        terminal_code = terminal_code or "worker_release_failed"

    if terminal_code is None:
        terminal_code = _derive_worker_terminal_code(phase_results)
    status = {
        "schema_version": 7,
        "artifact_kind": "p4_7_gpu_worker_status",
        "status": terminal_code,
        "B2_authorized": False,
        "B2_started": False,
    }
    atomic_write_privacy_safe_json(output / "worker_status.json", status)
    return {**status, "ready": False}


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthV7Error(f"cannot re-open {path.name}") from error
    if not isinstance(value, dict) or not value:
        _fail(f"{path.name} is empty or not an object")
    return value


def _finalizer_failure(
    output: Path,
    *,
    reason: str,
    cleanup_complete: bool,
) -> dict[str, Any]:
    failure_path = output / "failure.json"
    if not failure_path.exists():
        atomic_write_privacy_safe_json(
            failure_path,
            {
                "schema_version": 7,
                "artifact_kind": "p4_7_terminal_failure",
                "status": "fail",
                "reason": reason,
                "production_sampler_refresh_ready": False,
                "OPD_scoring_backend_ready": False,
                "B2_authorized": False,
                "B2_started": False,
            },
        )
    index_sha256 = _commit_and_reverify_root_index(output)
    readiness = {
        "schema_version": 7,
        "artifact_kind": "p4_7_length_final_readiness",
        "run_id": "qwen3-4b-length-qualification-v7-seed42",
        "status": reason,
        "ready": False,
        "selected_response_length": None,
        "artifact_index_sha256": index_sha256,
        "cleanup_complete": cleanup_complete,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
    }
    return _commit_and_reverify_readiness(output, readiness)


def finalize_length_qualification(
    config: Mapping[str, Any],
    *,
    gpu_idle_probe: Callable[[], Mapping[str, Any]] = _gpu_idle_evidence,
    authority_revalidator: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    """CPU-only authority finalizer, invoked after the GPU worker exits."""

    output = Path(str(config["run"]["output_dir"]))
    package_dir = Path(str(config["run"]["generated_b2_package_dir"]))
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        _fail("formal worker output is not a safe directory")
    if not output.exists():
        output.mkdir(parents=True)
    try:
        worker_status = _read_json_object(output / "worker_status.json")
        worker_release = _read_json_object(output / "worker_release.json")
    except ProductionLengthV7Error:
        idle = dict(gpu_idle_probe())
        atomic_write_privacy_safe_json(
            output / "resource_cleanup.json",
            {
                "schema_version": 7,
                "artifact_kind": "p4_7_finalizer_cleanup",
                "cleanup_complete": False,
                "worker_released": False,
                "gpu_memory_used_mib": idle.get("gpu_memory_used_mib", []),
                "compute_pids": idle.get("compute_pids", []),
                "residual_worker_pids": idle.get("residual_worker_pids", []),
                "B2_started": False,
            },
        )
        return _finalizer_failure(
            output, reason="worker_artifact_missing", cleanup_complete=False
        )
    idle = dict(gpu_idle_probe())
    cleanup_complete = bool(
        worker_release.get("status") == "released"
        and worker_release.get("models_released") is True
        and worker_release.get("B2_started") is False
        and idle.get("idle") is True
        and idle.get("gpu_memory_used_mib") == [0, 0]
        and idle.get("compute_pids") == []
        and idle.get("residual_worker_pids") == []
    )
    cleanup = {
        "schema_version": 7,
        "artifact_kind": "p4_7_finalizer_cleanup",
        "cleanup_complete": cleanup_complete,
        "worker_released": worker_release.get("models_released") is True,
        "gpu_memory_used_mib": idle.get("gpu_memory_used_mib", []),
        "compute_pids": idle.get("compute_pids", []),
        "residual_worker_pids": idle.get("residual_worker_pids", []),
        "B2_started": False,
    }
    atomic_write_privacy_safe_json(output / "resource_cleanup.json", cleanup)
    if not cleanup_complete:
        return _finalizer_failure(
            output, reason="cleanup_failed", cleanup_complete=False
        )

    worker_code = worker_status.get("status")
    if worker_code != "worker_complete":
        reason = str(worker_code) if isinstance(worker_code, str) else "worker_incomplete"
        return _finalizer_failure(output, reason=reason, cleanup_complete=True)

    try:
        if authority_revalidator is None:
            from src.opd.production_length_preflight_v7 import (
                reverify_finalization_authority,
            )

            authority = dict(reverify_finalization_authority(config))
        else:
            authority = dict(authority_revalidator(config))
        if not (
            authority.get("parent_core_evidence_verified") is True
            and authority.get("v2_adapter_reusable") is True
            and authority.get("current_bindings_verified") is True
            and authority.get("static_assets_verified") is True
        ):
            _fail("parent finalization authority did not revalidate")
        atomic_write_privacy_safe_json(
            output / "finalizer_authority_revalidation.json", authority
        )
    except BaseException:
        return _finalizer_failure(
            output, reason="parent_revalidation_failed", cleanup_complete=True
        )

    try:
        preflight = _read_json_object(output / "preflight.json")
        identity = _read_json_object(output / "v2_reload_identity.json")
        prefix = _read_json_object(output / "prefix_equivalence.json")
        summary = _read_json_object(output / "generation_summary.json")
        if not (
            preflight.get("run_id") == config["run"]["run_id"]
            and preflight.get("status") == "ready_waiting_for_gpu_length_qualification"
            and preflight.get("parent_core_evidence_verified") is True
            and preflight.get("v2_adapter_reusable") is True
            and isinstance(preflight.get("gpu_host"), Mapping)
            and len(preflight["gpu_host"].get("gpus", [])) == 2
            and preflight.get("isolation") == config["isolation"]
            and preflight.get("git", {}).get("worktree_clean") is True
            and authority.get("worktree_clean") is True
            and preflight.get("git", {}).get("git_commit")
            == authority.get("git_commit")
            and preflight.get("parent_audit_sha256")
            == authority.get("parent_audit_sha256")
            and preflight.get("current_bindings_sha256")
            == authority.get("current_bindings_sha256")
            and preflight.get("static_assets_sha256")
            == authority.get("static_assets_sha256")
            and preflight.get("versions") == authority.get("versions")
            and authority.get("isolation") == config["isolation"]
        ):
            _fail("persisted formal preflight does not revalidate")
        runtime_sha = config["parent_reuse"]["v2"]["aggregate_tensor_sha256"]
        if not (
            identity.get("passed") is True
            and identity.get("runtime_tensor_sha256") == runtime_sha
            and identity.get("checkpoint_tensor_sha256") == runtime_sha
            and identity.get("logical_version") == "v2"
            and identity.get("tensor_count") == 504
            and identity.get("active_slot") == "student_active"
            and identity.get("registry_count") == 1
            and identity.get("eos_stop_config_verified") is True
        ):
            _fail("fresh v2 identity does not revalidate")
        strategy = summary.get("generation_strategy")
        if not (
            prefix.get("artifact_kind") == "production_length_prefix_equivalence_v7"
            and isinstance(prefix.get("passed"), bool)
            and summary.get("automatic_further_escalation") is False
            and ((prefix["passed"] is True and strategy == "derived_candidates") or (prefix["passed"] is False and strategy == "explicit_independent_generation"))
        ):
            _fail("prefix/fallback evidence does not revalidate")

        primary = derive_length_readiness(output / "primary")
        conditional_dir = output / "conditional_4096"
        conditional = (
            derive_length_readiness(conditional_dir)
            if conditional_dir.is_dir()
            else None
        )
        resource_preflight_path = output / "conditional_4096_preflight.json"
        eligibility_path = output / "conditional_4096_eligibility.json"
        if conditional is not None:
            resource_preflight = _read_json_object(resource_preflight_path)
            persisted_gate = _read_json_object(eligibility_path)
            disk_resource = resource_preflight.get("disk")
            gpu_resource = resource_preflight.get("gpu")
            max_prompt_tokens = resource_preflight.get("max_prompt_tokens")
            if not (
                resource_preflight.get("artifact_kind")
                == "p4_7_conditional_4096_resource_preflight"
                and resource_preflight.get("actual_generation_cap") == 4096
                and isinstance(max_prompt_tokens, int)
                and not isinstance(max_prompt_tokens, bool)
                and max_prompt_tokens > 0
                and isinstance(disk_resource, Mapping)
                and isinstance(gpu_resource, Mapping)
                and resource_preflight.get("disk_preflight_passed")
                is (disk_resource.get("passed") is True)
                and resource_preflight.get("gpu_preflight_passed")
                is (gpu_resource.get("passed") is True)
                and resource_preflight.get("passed")
                is bool(
                    disk_resource.get("passed") is True
                    and gpu_resource.get("passed") is True
                )
            ):
                _fail("conditional 4096 resource preflight does not revalidate")
            primary_telemetry = validate_length_telemetry(
                _read_json_object(output / "primary" / "length_telemetry.json")
            )
            expected_gate = evaluate_conditional_4096(
                primary_telemetry,
                eos_stop_config_valid=(
                    identity.get("eos_stop_config_verified") is True
                ),
                model_context_limit=int(config["model"]["context_limit"]),
                max_prompt_tokens=max_prompt_tokens,
                disk_preflight_passed=(
                    resource_preflight.get("disk_preflight_passed") is True
                ),
                gpu_preflight_passed=(
                    resource_preflight.get("gpu_preflight_passed") is True
                ),
                isolation=config["isolation"],
            )
            if not (
                persisted_gate == expected_gate
                and expected_gate.get("allowed") is True
                and summary.get("conditional_4096_executed") is True
            ):
                _fail("conditional 4096 eligibility does not reproduce")
        elif primary.get("ready") is True and (
            resource_preflight_path.exists()
            or eligibility_path.exists()
            or summary.get("conditional_4096_executed") is not False
        ):
            _fail("conditional 4096 evidence exists after primary success")
        selected_phase: str | None = None
        phase_readiness: Mapping[str, Any] | None = None
        if primary.get("ready") is True and conditional is None:
            selected_phase, phase_readiness = "primary", primary
        elif (
            primary.get("status") == "no_length_candidate_passed"
            and conditional is not None
            and conditional.get("ready") is True
            and summary.get("conditional_4096_executed") is True
        ):
            selected_phase, phase_readiness = "conditional", conditional
        if selected_phase is None or phase_readiness is None:
            reason = (
                "needs_post_4096_evidence_review"
                if conditional is not None
                else str(primary.get("status", "phase_evidence_invalid"))
            )
            return _finalizer_failure(output, reason=reason, cleanup_complete=True)

        phase_dir = output / (
            "primary" if selected_phase == "primary" else "conditional_4096"
        )
        selected = phase_readiness["selected_response_length"]
        telemetry_path = phase_dir / "length_telemetry.json"
        selection_path = phase_dir / "length_selection.json"
        formal_evidence = {
            "status": "passed_length_only_qualification",
            "execution_mode": "formal_gpu",
            "formal_run_root": str(output),
            "phase_dir": str(phase_dir),
            "selected_response_length": selected,
            "parent_reuse_attestation_sha256": validate_length_telemetry(
                _read_json_object(telemetry_path)
            )["bindings"]["parent_p4_6_binding_sha256"],
            "runtime_adapter_sha256": runtime_sha,
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "cleanup_path": str(output / "resource_cleanup.json"),
            "cleanup_complete": True,
            "final_index_path": str(phase_dir / "artifact_index.json"),
            "identity_path": str(output / "v2_reload_identity.json"),
            "prefix_path": str(output / "prefix_equivalence.json"),
            "failure_artifact_exists": False,
            "isolation": dict(config["isolation"]),
        }
        package = build_b2_calibration_package(
            formal_evidence,
            _package_bindings(config, formal_evidence["selection_sha256"]),
        )
        from jsonschema import Draft202012Validator

        b2_schema_path = (
            Path(__file__).resolve().parents[2]
            / config["artifacts"]["b2_package_schema_path"]
        )
        Draft202012Validator(
            json.loads(b2_schema_path.read_text(encoding="utf-8"))
        ).validate(package)
        package_result = materialize_b2_calibration_package(
            package_dir,
            package,
            source_length_run_id=config["run"]["run_id"],
            formal_run_root=output,
        )
        atomic_write_privacy_safe_json(
            output / "b2_package_manifest.json",
            {
                "schema_version": 7,
                "artifact_kind": "p4_7_b2_package_manifest",
                "output_dir": str(package_dir),
                "files": package_result["files"],
                "B2_authorized": True,
                "B2_started": False,
            },
        )
        atomic_write_privacy_safe_json(
            output / "length_selection.json",
            {
                "schema_version": 7,
                "artifact_kind": "p4_7_combined_length_selection",
                "selected_response_length": selected,
                "source_phase": selected_phase,
                "source_selection_sha256": formal_evidence["selection_sha256"],
                "B2_started": False,
            },
        )
    except BaseException as error:
        reason = str(getattr(error, "code", "final_artifact_validation_failed"))
        return _finalizer_failure(output, reason=reason, cleanup_complete=True)

    index_sha256 = _commit_and_reverify_root_index(output)
    readiness = {
        "schema_version": 7,
        "artifact_kind": "p4_7_length_final_readiness",
        "run_id": config["run"]["run_id"],
        "status": "passed_length_only_qualification",
        "ready": True,
        "selected_response_length": selected,
        "artifact_index_sha256": index_sha256,
        "cleanup_complete": True,
        "parent_core_evidence_verified": True,
        "v2_reload_identity_verified": True,
        "production_sampler_refresh_ready": True,
        "OPD_scoring_backend_ready": True,
        "B2_authorized": True,
        "B2_started": False,
        "isolation": dict(config["isolation"]),
    }
    return _commit_and_reverify_readiness(output, readiness)


def _dry_run(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready_waiting_for_gpu_length_qualification",
        "run_id": config["run"]["run_id"],
        "gpu_used": False,
        "loaded_real_model": False,
        "parent_core_evidence_verified": True,
        "length_writer_fixed": True,
        "length_protocol_frozen": True,
        "gpu_length_qualification_pending": True,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.7 length-only qualification")
    parser.add_argument("--config", required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--gpu-worker", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = _load_config(path)
    if args.execute or args.gpu_worker:
        if os.environ.get("CA_OPD_ALLOW_LENGTH_QUALIFICATION_V7_GPU") != "1":
            _fail("explicit GPU authorization environment variable is absent")
        if args.execute:
            _fail("direct --execute is forbidden; use the canonical shell launcher")
        if (
            args.gpu_worker
            and os.environ.get("CA_OPD_P4_7_LAUNCHER_WORKER") != "1"
        ):
            _fail("internal GPU execution requires the launcher worker boundary")

        def _raise_interrupt(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt("P4.7 execution interrupted")

        signal.signal(signal.SIGTERM, _raise_interrupt)
        signal.signal(signal.SIGHUP, _raise_interrupt)
        result = execute_gpu_length_qualification(config, config_path=path)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "worker_complete" else 1
    if args.finalize:
        # This process imports no model backend and is launched only after the
        # worker PID has been reaped.
        result = finalize_length_qualification(config)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ready") is True else 1
    if args.preflight:
        from src.opd.production_length_preflight_v7 import preflight

        result = preflight(path, execute_gpu=False)
    else:
        result = _dry_run(config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProductionLengthV7Error",
    "execute_gpu_length_qualification",
    "finalize_length_qualification",
    "main",
]
