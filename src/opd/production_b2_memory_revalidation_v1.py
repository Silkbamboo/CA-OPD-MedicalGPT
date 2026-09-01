"""P4.8e package-bound GPU differential, canary, and 20-step launcher."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

from src.opd.production_b2_calibration_artifacts_v2 import (
    B2CalibrationArtifactsV1Error,
    finalize_calibration_run,
)
from src.opd.production_b2_calibration_v1 import (
    B2CalibrationLauncherV1Error,
    _atomic_json,
    _post_worker_cleanup_observation,
    authorize_gpu_execution,
    install_worker_signal_handlers,
    verify_parent_and_static_assets,
)
from src.opd.production_b2_calibration_v2 import _verify_teacher_assets
from src.opd.production_b2_calibration_worker_v2 import (
    _build_memory_runtime_config,
    _default_prompt_provider,
    execute_memory_balanced_calibration_worker_v1,
)
from src.opd.production_b2_data_v2 import CANONICAL_MANIFEST_PATH
from src.opd.production_b2_gpu_math_differential_v1 import (
    B2GpuMathDifferentialV1Error,
    execute_real_gpu_math_differential_v1,
)
from src.opd.production_b2_memory_revalidation_package_v1 import (
    B2MemoryRevalidationPackageV1Error,
    PACKAGE_VERSION,
    RUN_ID,
    verify_revalidation_overlay_package,
)
from src.opd.production_b2_memory_revalidation_preflight_v1 import (
    B2MemoryRevalidationPreflightV1Error,
    preflight_b2_memory_revalidation_v1,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8e-1024-memory-v3-package"
)
DEFAULT_OUTPUT = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8e-r3-1024-memory-seed42"
)
DEFAULT_GATE_OUTPUT = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8e-1024-memory-gates-v3-seed42"
)
DEFAULT_LAUNCH_SPEC = Path(
    "configs/opd/qwen3_4b_b2_calibration_p4_8e_1024_memory.yaml"
)


class B2MemoryRevalidationLauncherV1Error(RuntimeError):
    """The P4.8e launcher refused an unbound or unsafe action."""


def _fail(message: str) -> None:
    raise B2MemoryRevalidationLauncherV1Error(message)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_cleanup_observation_v1() -> dict[str, Any]:
    memory: list[int] = []
    compute_pids: list[int] = []
    try:
        memory = [
            int(row.strip())
            for row in subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.splitlines()
            if row.strip()
        ]
        compute_pids = [
            int(row.strip())
            for row in subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.splitlines()
            if row.strip()
        ]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_used_mib": memory,
        "compute_pids": compute_pids,
        "cleanup_complete": (
            len(memory) == 2
            and all(0 <= value <= 16 for value in memory)
            and not compute_pids
        ),
    }


def finalize_gate_only_failure_v1(gate_output_dir: str | Path) -> dict[str, Any]:
    """Persist a failed GPU-gate outcome when formal output never started."""

    gates = Path(gate_output_dir).resolve()
    if gates.is_symlink():
        _fail("P4.8e gate failure output may not be a symlink")
    gates.mkdir(parents=True, exist_ok=True)
    failure_path = gates / "gpu_math_failure.json"
    failure = None
    if failure_path.is_file() and not failure_path.is_symlink():
        try:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failure = {"status": "failed_gpu_gate_artifact_invalid"}
    result = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_gate_only_failure_finalization_v1",
        "status": "failed_gpu_gate_before_formal_calibration",
        "gpu_gate_failure": failure,
        "cleanup": _gpu_cleanup_observation_v1(),
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "final_access_count": 0,
        "controller_access_count": 0,
        "label_access_count": 0,
    }
    _atomic_json(gates / "gate_failure_finalization.json", result)
    return result


def load_launch_spec_v1(
    path: str | Path = DEFAULT_LAUNCH_SPEC,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not (
        source.resolve() == (REPO_ROOT / DEFAULT_LAUNCH_SPEC).resolve()
        and source.is_file()
        and not source.is_symlink()
    ):
        _fail("only the canonical P4.8e launch spec is accepted")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise B2MemoryRevalidationLauncherV1Error(
            f"P4.8e launch spec is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail("P4.8e launch spec is not an object")
    required = {
        "schema_id",
        "schema_version",
        "status",
        "run",
        "gates",
        "source_package",
        "parent_package",
        "frozen_contract",
        "authorization",
        "git",
        "resources",
        "isolation",
    }
    if not required <= set(value):
        _fail("P4.8e launch spec sections are incomplete")
    frozen = value["frozen_contract"]
    if not (
        value["schema_id"] == "ca-opd/p4.8e-b2-memory-revalidation-launch/v1"
        and value["schema_version"] == 1
        and value["status"] == "ready_for_p4_8e_gpu_gates"
        and value["run"]["run_id"] == RUN_ID
        and Path(value["run"]["output_dir"]).resolve() == DEFAULT_OUTPUT
        and Path(value["gates"]["output_dir"]).resolve() == DEFAULT_GATE_OUTPUT
        and Path(value["source_package"]["path"]).resolve() == DEFAULT_PACKAGE
        and value["source_package"]["package_version"] == PACKAGE_VERSION
        and frozen
        == {
            "model": "Qwen3-4B",
            "response_length": 1024,
            "seed": 42,
            "optimizer_steps": 20,
            "schedule_slots": 80,
            "effective_batch_size": 4,
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
        }
        and value["authorization"]["B2_calibration_authorized"] is True
        and value["authorization"]["B2_formal_authorized"] is False
        and value["isolation"]
        == {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }
    ):
        _fail("P4.8e launch contract differs")
    return value


def run_package_bound_preflight_v1(
    spec: Mapping[str, Any], *, mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = verify_revalidation_overlay_package(
        spec["source_package"]["path"],
        canonical_manifest_path=CANONICAL_MANIFEST_PATH,
    )
    result = preflight_b2_memory_revalidation_v1(
        audit,
        mode=mode,
        projected_increment_bytes=int(
            spec["resources"]["projected_increment_bytes"]
        ),
    )
    if DEFAULT_GATE_OUTPUT.exists() or DEFAULT_GATE_OUTPUT.is_symlink():
        _fail("P4.8e GPU gate output must be fresh")
    # P4.8e reuses the already frozen P4.7/P4.8d static authority, but streams
    # the actual model/tokenizer assets again before any GPU model load.
    from src.opd.production_b2_calibration_v4 import load_launch_spec_v4

    result["parent_and_static"] = verify_parent_and_static_assets(
        load_launch_spec_v4()
    )
    result["teacher_identity"] = _verify_teacher_assets(audit)
    return result, audit


def execute_revalidation_v1(
    *,
    audit: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output_dir: str | Path,
    gate_output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    gates = Path(gate_output_dir).resolve()
    runtime_config = _build_memory_runtime_config(audit, output_dir=output)
    prompts = list(_default_prompt_provider(runtime_config, 0))
    try:
        differential = execute_real_gpu_math_differential_v1(
            runtime_config=runtime_config,
            prompt_rows=prompts,
            evidence_dir=gates,
        )
    except BaseException as error:
        cleanup: dict[str, Any] = {"torch_cleanup_attempted": False}
        try:
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            cleanup = {
                "torch_cleanup_attempted": True,
                "memory_allocated_bytes": [
                    int(torch.cuda.memory_allocated(device))
                    for device in (0, 1)
                ],
                "memory_reserved_bytes": [
                    int(torch.cuda.memory_reserved(device))
                    for device in (0, 1)
                ],
            }
        except BaseException as cleanup_error:
            cleanup = {
                "torch_cleanup_attempted": True,
                "cleanup_error_type": type(cleanup_error).__name__,
            }
        if gates.is_dir() and not gates.is_symlink():
            _atomic_json(
                gates / "gpu_math_failure.json",
                {
                    "schema_version": 1,
                    "artifact_kind": "p4_8e_gpu_math_gate_failure_v1",
                    "status": "failed_real_gpu_math_differential",
                    "error_type": type(error).__name__,
                    "cleanup": cleanup,
                    "B2_calibration_complete": False,
                    "B2_formal_authorized": False,
                    "final_access_count": 0,
                    "controller_access_count": 0,
                    "label_access_count": 0,
                },
            )
        raise
    if (
        differential.get("passed") is not True
        or differential.get("gpu_cleanup_after_gate") is not True
        or differential.get("gpu_cleanup", {}).get("memory_allocated_bytes")
        != [0, 0]
        or differential.get("gpu_cleanup", {}).get("memory_reserved_bytes")
        != [0, 0]
    ):
        _fail("P4.8e real GPU math differential/cleanup did not pass")
    worker = execute_memory_balanced_calibration_worker_v1(
        package_audit=audit,
        output_dir=output,
        execution_mode="formal_gpu",
        git_commit=str(preflight["git"]["head"]),
    )
    if worker.get("steps_completed") != 20:
        _fail("P4.8e worker did not complete exactly 20 optimizer steps")
    source_report = Path(str(differential["report_path"]))
    target_report = output / "gpu_math_equivalence.json"
    shutil.copyfile(source_report, target_report)
    gate_status = {
        "schema_version": 1,
        "artifact_kind": "p4_8e_gpu_gate_status_v1",
        "run_id": RUN_ID,
        "gpu_math_differential_passed": True,
        "gpu_math_report_sha256": differential["report_sha256"],
        "max_shape_canary_passed": True,
        "max_shape_canary_sha256": _stream_sha256(
            output / "memory_canary.json"
        ),
        "six_step_memory_drift_sha256": _stream_sha256(
            output / "memory_six_step_drift.json"
        ),
        "steps_completed": 20,
        "B2_formal_authorized": False,
    }
    _atomic_json(output / "p4_8e_gate_status.json", gate_status)
    return {**worker, **gate_status}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--host-preflight", action="store_true")
    modes.add_argument("--execute-worker", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--launch-spec", type=Path, default=DEFAULT_LAUNCH_SPEC)
    parser.add_argument("--allow-b2-calibration", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    spec = load_launch_spec_v1(args.launch_spec)
    if not (
        args.package.resolve() == DEFAULT_PACKAGE
        and args.output_root.resolve() == DEFAULT_OUTPUT
    ):
        _fail("loose package/output override differs from P4.8e launch spec")
    if args.dry_run:
        result, _audit = run_package_bound_preflight_v1(spec, mode="dry-run")
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.host_preflight:
        result, _audit = run_package_bound_preflight_v1(
            spec, mode="host-preflight"
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.execute_worker:
        authorize_gpu_execution(
            os.environ, allow_argument=args.allow_b2_calibration
        )
        result, audit = run_package_bound_preflight_v1(spec, mode="execute")
        install_worker_signal_handlers()
        worker = execute_revalidation_v1(
            audit=audit,
            preflight=result,
            output_dir=DEFAULT_OUTPUT,
            gate_output_dir=DEFAULT_GATE_OUTPUT,
        )
        print(json.dumps(worker, sort_keys=True))
        return 0
    if args.finalize:
        if not DEFAULT_OUTPUT.is_dir():
            failure = finalize_gate_only_failure_v1(DEFAULT_GATE_OUTPUT)
            print(json.dumps(failure, sort_keys=True))
            return 2
        verify_revalidation_overlay_package(
            DEFAULT_PACKAGE, canonical_manifest_path=CANONICAL_MANIFEST_PATH
        )
        summary = finalize_calibration_run(
            DEFAULT_OUTPUT,
            cleanup_observation=_post_worker_cleanup_observation(DEFAULT_OUTPUT),
        )
        print(json.dumps(summary, sort_keys=True))
        return (
            0
            if summary.get("status")
            == "b2_calibration_complete_ready_for_b2_formal"
            else 2
        )
    _fail("unknown P4.8e launcher mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        B2MemoryRevalidationLauncherV1Error,
        B2MemoryRevalidationPackageV1Error,
        B2MemoryRevalidationPreflightV1Error,
        B2GpuMathDifferentialV1Error,
        B2CalibrationLauncherV1Error,
        B2CalibrationArtifactsV1Error,
    ) as error:
        print(f"P4.8e B2 memory revalidation refused: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "DEFAULT_GATE_OUTPUT",
    "DEFAULT_LAUNCH_SPEC",
    "DEFAULT_OUTPUT",
    "DEFAULT_PACKAGE",
    "RUN_ID",
    "build_argument_parser",
    "execute_revalidation_v1",
    "finalize_gate_only_failure_v1",
    "load_launch_spec_v1",
    "main",
    "run_package_bound_preflight_v1",
]
