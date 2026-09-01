"""P4.8d package-bound memory canary and B2 calibration launcher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

from src.opd.production_b2_calibration_artifacts_v2 import (
    B2CalibrationArtifactsV1Error,
    finalize_calibration_run,
)
from src.opd.production_b2_calibration_package_v4 import (
    B2CalibrationPackageV4Error,
    RUN_ID,
    verify_memory_execution_package,
)
from src.opd.production_b2_calibration_preflight_v4 import (
    B2CalibrationPreflightV4Error,
    preflight_b2_memory_calibration_v4,
)
from src.opd.production_b2_calibration_v1 import (
    B2CalibrationLauncherV1Error,
    _post_worker_cleanup_observation,
    authorize_gpu_execution,
    install_worker_signal_handlers,
    verify_parent_and_static_assets,
)
from src.opd.production_b2_calibration_v2 import _verify_teacher_assets
from src.opd.production_b2_calibration_worker_v2 import (
    execute_memory_balanced_calibration_worker_v1,
)
from src.opd.production_b2_data_v2 import CANONICAL_MANIFEST_PATH, stream_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8d-1024-memory-v2-package"
)
DEFAULT_OUTPUT = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8d-1024-memory-seed42"
)
DEFAULT_LAUNCH_SPEC = Path(
    "configs/opd/qwen3_4b_b2_calibration_p4_8d_1024_memory.yaml"
)
DEFAULT_RUN_CARD = Path(
    "configs/run_cards/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8d-1024-memory-seed42.json"
)


class B2CalibrationLauncherV4Error(RuntimeError):
    """P4.8d launcher refusal before unsafe work can start."""


def _fail(message: str) -> None:
    raise B2CalibrationLauncherV4Error(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2CalibrationLauncherV4Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def load_launch_spec_v4(path: str | Path = DEFAULT_LAUNCH_SPEC) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    canonical = (REPO_ROOT / DEFAULT_LAUNCH_SPEC).resolve()
    if source.resolve() != canonical or source.is_symlink() or not source.is_file():
        _fail("only the canonical P4.8d launch spec is accepted")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise B2CalibrationLauncherV4Error(
            f"P4.8d launch spec is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail("P4.8d launch spec is not an object")
    required = {
        "schema_id",
        "schema_version",
        "status",
        "run",
        "source_package",
        "frozen_contract",
        "memory_execution",
        "authorization",
        "resources",
        "git",
        "isolation",
        "current_bindings",
        "run_card",
        "p4_7_bindings",
    }
    if not required <= set(value):
        _fail("P4.8d launch spec sections are incomplete")
    run = value["run"]
    frozen = value["frozen_contract"]
    memory = value["memory_execution"]
    package = value["source_package"]
    isolation = value["isolation"]
    if not all(
        isinstance(item, Mapping)
        for item in (run, frozen, memory, package, isolation)
    ) or not (
        value["schema_id"] == "ca-opd/p4.8d-b2-memory-calibration-launch/v4"
        and value["schema_version"] == 4
        and value["status"] == "ready_waiting_for_gpu_b2_memory_revalidation"
        and run.get("run_id") == RUN_ID
        and Path(str(run.get("output_dir", ""))).resolve() == DEFAULT_OUTPUT
        and Path(str(package.get("path", ""))).resolve() == DEFAULT_PACKAGE
        and frozen.get("optimizer_steps") == 20
        and frozen.get("selected_response_length") == 1024
        and frozen.get("seed") == 42
        and frozen.get("effective_batch_size") == 4
        and frozen.get("student_initialization")
        == "fresh_base_plus_fresh_zero_lora_v1"
        and memory.get("physical_microbatch_size") == 1
        and memory.get("gradient_accumulation_steps") == 4
        and memory.get("checkpoint_versions") == [5, 10, 15, 20]
        and memory.get("minimum_canary_headroom_bytes") == 1024**3
        and frozen.get("automatically_start_formal_b2") is False
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
        _fail("P4.8d launch contract differs")
    return value


def verify_current_launch_bindings_v4(spec: Mapping[str, Any]) -> dict[str, str]:
    bindings = spec.get("current_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        _fail("P4.8d executable bindings are absent")
    verified: dict[str, str] = {}
    for name, descriptor in bindings.items():
        if not (
            isinstance(name, str)
            and isinstance(descriptor, Mapping)
            and set(descriptor) == {"path", "sha256"}
        ):
            _fail("P4.8d executable binding descriptor is invalid")
        path = REPO_ROOT / str(descriptor["path"])
        expected = descriptor["sha256"]
        if not (
            path.is_file()
            and not path.is_symlink()
            and isinstance(expected, str)
            and len(expected) == 64
            and stream_sha256(path) == expected
        ):
            _fail(f"P4.8d executable binding differs: {name}")
        verified[name] = expected
    card_path = REPO_ROOT / str(spec["run_card"]["path"])
    if card_path.resolve() != (REPO_ROOT / DEFAULT_RUN_CARD).resolve():
        _fail("P4.8d checked-in run-card path differs")
    card = _read_json(card_path, "P4.8d checked-in run card")
    launch_sha = stream_sha256(REPO_ROOT / DEFAULT_LAUNCH_SPEC)
    package = spec["source_package"]
    if not (
        card.get("schema_id") == "ca-opd/p4.8d-b2-memory-calibration-run-card/v4"
        and card.get("schema_version") == 4
        and card.get("run_id") == RUN_ID
        and card.get("status") == "prepared_not_started"
        and card.get("config_path") == str(DEFAULT_LAUNCH_SPEC)
        and card.get("config_sha256") == launch_sha
        and card.get("source_package_content_sha256")
        == package.get("package_content_sha256")
        and card.get("source_authorization_sha256")
        == package.get("authorization_sha256")
        and card.get("memory_execution_contract_sha256")
        == package.get("memory_execution_contract_sha256")
        and card.get("schedule_sha256") == package.get("schedule_sha256")
        and card.get("manifest_sha256") == package.get("manifest_sha256")
        and card.get("optimizer_steps") == 20
        and card.get("selected_response_length") == 1024
        and card.get("physical_microbatch_size") == 1
        and card.get("gradient_accumulation_steps") == 4
        and card.get("effective_batch_size") == 4
        and card.get("checkpoint_versions") == [5, 10, 15, 20]
        and card.get("B2_calibration_started") is False
        and card.get("B2_formal_authorized") is False
        and card.get("automatically_start_formal_b2") is False
    ):
        _fail("P4.8d checked-in run-card contract differs")
    return verified


def _expected_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    package = spec["source_package"]
    return {
        key: package[key]
        for key in (
            "package_content_sha256",
            "package_index_sha256",
            "authorization_sha256",
            "config_sha256",
            "run_card_sha256",
            "schedule_sha256",
            "manifest_sha256",
            "oom_memory_attestation_sha256",
            "memory_execution_contract_sha256",
        )
    }


def run_package_bound_preflight_v4(
    spec: Mapping[str, Any],
    *,
    mode: str,
    allow_dirty_for_development: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_current_launch_bindings_v4(spec)
    result = preflight_b2_memory_calibration_v4(
        spec["source_package"]["path"],
        output_dir=spec["run"]["output_dir"],
        canonical_manifest_path=CANONICAL_MANIFEST_PATH,
        expected=_expected_from_spec(spec),
        mode=mode,
        expected_branch=str(spec["git"]["branch"]),
        projected_increment_bytes=int(spec["resources"]["projected_increment_bytes"]),
        allow_dirty_for_development=allow_dirty_for_development,
    )
    audit = verify_memory_execution_package(
        spec["source_package"]["path"],
        canonical_manifest_path=CANONICAL_MANIFEST_PATH,
    )
    result["parent_and_static"] = verify_parent_and_static_assets(spec)
    result["teacher_identity"] = _verify_teacher_assets(audit)
    return result, audit


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
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    spec = load_launch_spec_v4(args.launch_spec)
    if not (
        args.package.resolve() == Path(str(spec["source_package"]["path"])).resolve()
        and args.output_root.resolve() == Path(str(spec["run"]["output_dir"])).resolve()
    ):
        _fail("loose package/output override differs from the P4.8d launch spec")
    if args.dry_run:
        result, _audit = run_package_bound_preflight_v4(
            spec,
            mode="dry-run",
            allow_dirty_for_development=args.allow_dirty_for_development,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.host_preflight:
        result, _audit = run_package_bound_preflight_v4(
            spec, mode="host-preflight"
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.execute_worker:
        authorize_gpu_execution(
            os.environ, allow_argument=args.allow_b2_calibration
        )
        result, audit = run_package_bound_preflight_v4(spec, mode="execute")
        install_worker_signal_handlers()
        worker = execute_memory_balanced_calibration_worker_v1(
            package_audit=audit,
            output_dir=args.output_root,
            execution_mode="formal_gpu",
            git_commit=str(result["git"]["head"]),
        )
        print(json.dumps(worker, sort_keys=True))
        return 0
    if args.finalize:
        verify_current_launch_bindings_v4(spec)
        verify_memory_execution_package(
            args.package, canonical_manifest_path=CANONICAL_MANIFEST_PATH
        )
        summary = finalize_calibration_run(
            args.output_root,
            cleanup_observation=_post_worker_cleanup_observation(args.output_root),
        )
        print(json.dumps(summary, sort_keys=True))
        return (
            0
            if summary.get("status")
            == "b2_calibration_complete_ready_for_b2_formal"
            else 2
        )
    _fail("unknown P4.8d launcher mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        B2CalibrationLauncherV4Error,
        B2CalibrationLauncherV1Error,
        B2CalibrationPreflightV4Error,
        B2CalibrationPackageV4Error,
        B2CalibrationArtifactsV1Error,
    ) as error:
        print(f"P4.8d B2 memory calibration refused: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "B2CalibrationLauncherV4Error",
    "DEFAULT_LAUNCH_SPEC",
    "DEFAULT_OUTPUT",
    "DEFAULT_PACKAGE",
    "DEFAULT_RUN_CARD",
    "RUN_ID",
    "build_argument_parser",
    "load_launch_spec_v4",
    "main",
    "run_package_bound_preflight_v4",
    "verify_current_launch_bindings_v4",
]
