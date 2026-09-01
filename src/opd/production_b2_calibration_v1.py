"""Standalone package-bound P4.8 B2 calibration CLI.

All heavyweight imports remain behind the explicitly authorized worker path.
Dry-run, package verification and the post-worker finalizer are CPU-only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from src.opd.production_b2_calibration_artifacts_v1 import (
    B2CalibrationArtifactStoreV1,
    B2CalibrationArtifactsV1Error,
    assert_formal_b2_calibration_candidate,
    finalize_calibration_run,
)
from src.opd.production_b2_calibration_contract_v1 import (
    B2_CALIBRATION_STEPS,
    FRESH_STUDENT_INITIALIZATION,
    SELECTED_RESPONSE_LENGTH,
    SUPPORTED_RESPONSE_LENGTHS,
    canonical_json_sha256,
    evaluate_latest_calibration_length_window,
)
from src.opd.production_b2_calibration_preflight_v1 import (
    B2CalibrationPreflightV1Error,
    preflight_b2_calibration,
    verify_p4_7_package,
)


RUN_ID = "qwen3-4b-b2-medical-opd-calibration-p4-8-seed42"
DEFAULT_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-v1-p4-7-package"
)
DEFAULT_OUTPUT = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8-seed42"
)
DEFAULT_LAUNCH_SPEC = Path(
    "configs/opd/qwen3_4b_b2_calibration_p4_8.yaml"
)
DEFAULT_RUN_CARD = Path(
    "configs/run_cards/"
    "qwen3-4b-b2-medical-opd-calibration-p4-8-seed42.json"
)


class B2CalibrationLauncherV1Error(RuntimeError):
    """The standalone B2 calibration launcher failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationLauncherV1Error(message)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
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


def _append_safe_log(path: Path, message: str) -> None:
    if any(token in message.lower() for token in ("question", "answer", "label")):
        _fail("unsafe calibration log message rejected")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stream_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_launch_spec(path: str | Path) -> dict[str, Any]:
    """Load the checked-in package binding without accepting loose overrides."""

    import yaml

    root = Path(__file__).resolve().parents[2]
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    canonical = (root / DEFAULT_LAUNCH_SPEC).resolve()
    if source.resolve() != canonical:
        _fail("only the canonical P4.8 launch spec is accepted")
    if source.is_symlink() or not source.is_file():
        _fail("canonical P4.8 launch spec is absent or a symlink")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise B2CalibrationLauncherV1Error(
            f"P4.8 launch spec is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail("P4.8 launch spec is not an object")
    source_package = value.get("source_package")
    run = value.get("run")
    frozen = value.get("frozen_contract")
    current = value.get("current_bindings")
    artifact_schema = value.get("artifact_schema")
    protocol = value.get("protocol")
    run_card = value.get("run_card")
    if not all(
        isinstance(item, Mapping)
        for item in (
            source_package,
            run,
            frozen,
            current,
            artifact_schema,
            protocol,
            run_card,
        )
    ):
        _fail("P4.8 launch spec sections are incomplete")
    if not (
        value.get("schema_id") == "ca-opd/p4.8-b2-calibration-launch/v1"
        and value.get("schema_version") == 1
        and run.get("run_id") == RUN_ID
        and Path(str(run.get("output_dir", ""))).resolve() == DEFAULT_OUTPUT
        and frozen.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and frozen.get("selected_response_length") == SELECTED_RESPONSE_LENGTH
        and frozen.get("student_initialization")
        == FRESH_STUDENT_INITIALIZATION
        and frozen.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and frozen.get("automatically_start_formal_b2") is False
    ):
        _fail("P4.8 launch contract drift")
    return value


def verify_checked_in_run_card(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the operator run card and bind it to every executable input."""

    root = Path(__file__).resolve().parents[2]
    descriptor = spec.get("run_card")
    schema = spec.get("artifact_schema")
    protocol = spec.get("protocol")
    current = spec.get("current_bindings")
    source = spec.get("source_package")
    if not all(
        isinstance(value, Mapping)
        for value in (descriptor, schema, protocol, current, source)
    ):
        _fail("P4.8 run-card bindings are incomplete")
    if set(descriptor) != {"path"}:
        _fail("P4.8 run-card descriptor is not canonical")
    card_path = root / str(descriptor["path"])
    if card_path.resolve() != (root / DEFAULT_RUN_CARD).resolve():
        _fail("P4.8 run-card path is not canonical")
    if card_path.is_symlink() or not card_path.is_file():
        _fail("P4.8 run card is absent or a symlink")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2CalibrationLauncherV1Error(
            f"P4.8 run card is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(card, dict):
        _fail("P4.8 run card is not an object")
    config_path = root / DEFAULT_LAUNCH_SPEC
    schema_path = root / str(schema.get("path", ""))
    protocol_path = root / str(protocol.get("path", ""))
    for path, expected, label in (
        (config_path, card.get("config_sha256"), "config"),
        (schema_path, schema.get("sha256"), "artifact schema"),
        (protocol_path, protocol.get("sha256"), "protocol"),
    ):
        if not (
            path.is_file()
            and not path.is_symlink()
            and isinstance(expected, str)
            and _stream_sha256(path) == expected
        ):
            _fail(f"P4.8 {label} run-card binding drift")
    if not (
        card.get("schema_id") == "ca-opd/p4.8-b2-calibration-run-card/v1"
        and card.get("schema_version") == 1
        and card.get("run_id") == RUN_ID
        and card.get("stage") == "b2_calibration"
        and card.get("status") == "prepared_cpu_only_not_started"
        and card.get("config_path") == str(DEFAULT_LAUNCH_SPEC)
        and card.get("artifact_schema_path") == schema.get("path")
        and card.get("artifact_schema_sha256") == schema.get("sha256")
        and card.get("protocol_path") == protocol.get("path")
        and card.get("protocol_sha256") == protocol.get("sha256")
        and card.get("current_bindings_sha256")
        == canonical_json_sha256(current)
        and card.get("source_package_content_sha256")
        == source.get("package_content_sha256")
        and card.get("source_authorization_sha256")
        == source.get("authorization_sha256")
        and card.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and card.get("selected_response_length")
        == SELECTED_RESPONSE_LENGTH
        and card.get("student_initialization")
        == FRESH_STUDENT_INITIALIZATION
        and card.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and card.get("required_environment")
        == "CA_OPD_ALLOW_B2_CALIBRATION_GPU=1"
        and card.get("required_argument") == "--allow-b2-calibration"
        and card.get("B2_authorized") is True
        and card.get("B2_calibration_started") is False
        and card.get("B2_calibration_complete") is False
        and card.get("B2_formal_authorized") is False
        and card.get("automatically_start_formal_b2") is False
    ):
        _fail("P4.8 checked-in run-card contract drift")
    return card


def _expected_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(spec["source_package"])
    source.update(
        {
            "branch": spec["git"]["branch"],
            "git_commit": spec["git"].get("exact_commit"),
            "projected_increment_bytes": int(
                spec["resources"]["projected_increment_bytes"]
            ),
            "student_initialization": FRESH_STUDENT_INITIALIZATION,
        }
    )
    return source


def verify_current_launch_bindings(spec: Mapping[str, Any]) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    bindings = spec.get("current_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        _fail("P4.8 current code/config bindings are absent")
    verified: dict[str, str] = {}
    for name, descriptor in bindings.items():
        if not (
            isinstance(name, str)
            and isinstance(descriptor, Mapping)
            and set(descriptor) == {"path", "sha256"}
        ):
            _fail("P4.8 current binding descriptor is invalid")
        path = root / str(descriptor["path"])
        expected = descriptor["sha256"]
        if not (
            isinstance(expected, str)
            and len(expected) == 64
            and path.is_file()
            and not path.is_symlink()
            and _stream_sha256(path) == expected
        ):
            _fail(f"P4.8 current binding drift: {name}")
        verified[name] = expected
    verify_checked_in_run_card(spec)
    return verified


def verify_parent_and_static_assets(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the P4.6 parent core and stream Base/tokenizer assets."""

    import importlib.metadata
    import yaml

    from src.opd.production_length_parent_v7 import verify_parent_reuse
    from src.opd.production_length_preflight_v7 import (
        _parent_expected,
        verify_static_assets,
    )

    root = Path(__file__).resolve().parents[2]
    p4_7 = spec.get("p4_7_bindings")
    if not isinstance(p4_7, Mapping):
        _fail("P4.7 source bindings are absent")
    config_path = root / str(p4_7["config_path"])
    attestation_path = root / str(p4_7["parent_attestation_path"])
    if not (
        config_path.is_file()
        and not config_path.is_symlink()
        and _stream_sha256(config_path) == p4_7["config_sha256"]
        and attestation_path.is_file()
        and not attestation_path.is_symlink()
        and _stream_sha256(attestation_path)
        == p4_7["parent_attestation_file_sha256"]
    ):
        _fail("P4.7 config/parent attestation SHA drift")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or not isinstance(attestation, Mapping):
        _fail("P4.7 config/parent attestation is invalid")
    parent = verify_parent_reuse(
        config["parent_reuse"]["output_dir"],
        expected=_parent_expected(config, attestation),
    )
    static = verify_static_assets(root, config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "verl", "vllm")
    }
    if versions != {
        key: str(value) for key, value in config.get("versions", {}).items()
    }:
        _fail("installed package versions differ from P4.7 authority")
    if not (
        parent.get("parent_core_evidence_verified") is True
        and parent.get("v2_adapter_reusable") is True
    ):
        _fail("P4.6 parent core or v2 is not reusable")
    return {
        "parent_core_evidence_verified": True,
        "protected_artifact_count": len(parent["protected_artifacts"]),
        "v2_tensor_count": parent["v2_adapter"]["tensor_count"],
        "v2_aggregate_tensor_sha256": parent["v2_adapter"][
            "aggregate_tensor_sha256"
        ],
        "versions": versions,
        "static_assets": static,
    }


def run_package_bound_preflight(
    spec: Mapping[str, Any],
    *,
    mode: str,
    allow_dirty_for_development: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_current_launch_bindings(spec)
    parent = verify_parent_and_static_assets(spec)
    result = preflight_b2_calibration(
        spec["source_package"]["path"],
        output_dir=spec["run"]["output_dir"],
        expected=_expected_from_spec(spec),
        mode=mode,
        allow_dirty_for_development=allow_dirty_for_development,
    )
    result["parent_and_static"] = parent
    audit = verify_p4_7_package(
        spec["source_package"]["path"],
        expected=_expected_from_spec(spec),
    )
    return result, audit


def authorize_gpu_execution(
    environment: Mapping[str, str], *, allow_argument: bool
) -> bool:
    if environment.get("CA_OPD_ALLOW_B2_CALIBRATION_GPU") != "1":
        _fail("B2 GPU execution lacks explicit environment authorization")
    if allow_argument is not True:
        _fail("B2 GPU execution requires --allow-b2-calibration")
    return True


def install_worker_signal_handlers() -> None:
    """Convert launcher signals into exceptions so worker ``finally`` runs."""

    def _controlled_stop(signum: int, _frame: Any) -> None:
        raise B2CalibrationLauncherV1Error(
            f"calibration worker interrupted by signal {signum}"
        )

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _controlled_stop)


def build_package_derived_runtime_config(
    package_audit: Mapping[str, Any], *, output_dir: str | Path
) -> dict[str, Any]:
    """Derive the only runtime config; no caller hyperparameters are accepted."""

    source = package_audit.get("config")
    if not isinstance(source, Mapping):
        _fail("verified P4.7 package config is absent")
    selected_response_length = package_audit.get("selected_response_length")
    package_version = package_audit.get("package_version")
    if not (
        selected_response_length in SUPPORTED_RESPONSE_LENGTHS
        and (
            selected_response_length == SELECTED_RESPONSE_LENGTH
            or package_version == "p4_8c_v3"
        )
        and package_audit.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and package_audit.get("seed") == 42
        and package_audit.get("student_initialization")
        == FRESH_STUDENT_INITIALIZATION
        and package_audit.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
    ):
        _fail("verified package does not bind the frozen calibration contract")
    runtime_run_id = str(package_audit.get("runtime_run_id") or RUN_ID)
    if not runtime_run_id:
        _fail("verified package runtime run ID is absent")
    config = deepcopy(dict(source))
    run = config.setdefault("run", {})
    if not isinstance(run, dict):
        _fail("package run config is invalid")
    run.update(
        {
            "run_id": runtime_run_id,
            "stage": "b2_calibration",
            "purpose": "package-bound Medical OPD 20-step calibration",
            "seed": 42,
            "optimizer_steps": B2_CALIBRATION_STEPS,
            "output_dir": str(Path(output_dir).resolve()),
            "status": "authorized_not_started",
            "automatically_start": False,
        }
    )
    generation = config.setdefault("generation", {})
    execution = config.setdefault("execution", {})
    if not isinstance(generation, dict) or not isinstance(execution, dict):
        _fail("package generation/execution config is invalid")
    generation["max_new_tokens"] = int(selected_response_length)
    execution["optimizer_steps"] = B2_CALIBRATION_STEPS
    execution["calibration_only"] = True
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
    config["p4_8_start_gate"] = {
        "source_package_content_sha256": package_audit["package_content_sha256"],
        "source_authorization_sha256": package_audit["authorization_sha256"],
        "source_length_root": package_audit["formal_run_root"],
        "requires_environment": "CA_OPD_ALLOW_B2_CALIBRATION_GPU=1",
        "requires_argument": "--allow-b2-calibration",
    }
    return config


def _package_binding(package_audit: Mapping[str, Any]) -> dict[str, Any]:
    selected_response_length = int(package_audit["selected_response_length"])
    result = {
        "schema_version": 1,
        "artifact_kind": "p4_8_package_binding_v1",
        "package_content_sha256": package_audit["package_content_sha256"],
        "authorization_sha256": package_audit["authorization_sha256"],
        "config_sha256": package_audit["config_sha256"],
        "run_card_sha256": package_audit["run_card_sha256"],
        "selected_response_length": selected_response_length,
        "optimizer_steps": B2_CALIBRATION_STEPS,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "qualification_v2_tensor_sha256": package_audit[
            "qualification_v2_tensor_sha256"
        ],
        "B2_authorized": True,
        "B2_started": False,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }
    if package_audit.get("package_version") == "p4_8b_v2":
        result.update(
            {
                "schema_version": 2,
                "artifact_kind": "p4_8b_package_binding_v2",
                "package_version": "p4_8b_v2",
                "manifest_sha256": package_audit["data_authority"][
                    "manifest_sha256"
                ],
                "schedule_sha256": package_audit["schedule"][
                    "schedule_sha256"
                ],
                "pre_model_semantic_gate": True,
            }
        )
    elif package_audit.get("package_version") == "p4_8c_v3":
        result.update(
            {
                "schema_version": 3,
                "artifact_kind": "p4_8c_package_binding_v3",
                "package_version": "p4_8c_v3",
                "manifest_sha256": package_audit["data_authority"][
                    "manifest_sha256"
                ],
                "schedule_sha256": package_audit["schedule"][
                    "schedule_sha256"
                ],
                "length_escalation_attestation_sha256": package_audit[
                    "length_escalation_attestation_sha256"
                ],
                "pre_model_semantic_gate": True,
            }
        )
    return result


def _worker_metadata(
    *, execution_mode: str, git_commit: str, run_id: str = RUN_ID
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_metadata_v1",
        "run_id": run_id,
        "execution_mode": execution_mode,
        "git_commit": git_commit,
        "production_backend_id": "custom_transformers_peft_three_policy_v5",
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }


def _data_manifest(package_audit: Mapping[str, Any]) -> dict[str, Any]:
    teacher = package_audit.get("teacher", {})
    result = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_data_manifest_v1",
        "manifest_sha256": teacher.get("data_manifest_sha256"),
        "selection_rule": "seed42_sha256_rank_first2_per_source_per_step_v1",
        "prompt_only": True,
        "raw_prompts_persisted": False,
        "labels_accessed": False,
    }
    if package_audit.get("package_version") in {"p4_8b_v2", "p4_8c_v3"}:
        result.update(
            {
                "schema_version": (
                    3 if package_audit.get("package_version") == "p4_8c_v3" else 2
                ),
                "artifact_kind": (
                    "b2_calibration_data_manifest_v3"
                    if package_audit.get("package_version") == "p4_8c_v3"
                    else "b2_calibration_data_manifest_v2"
                ),
                "manifest_sha256": package_audit["data_authority"][
                    "manifest_sha256"
                ],
                "schedule_sha256": package_audit["schedule"][
                    "schedule_sha256"
                ],
                "schedule_slot_count": package_audit["schedule"]["slot_count"],
                "provider": "production_b2_data_v2.resolve_b2_schedule_batch",
                "pre_model_semantic_gate": True,
            }
        )
    return result


def _default_session_factory(config: Mapping[str, Any], config_path: Path) -> Any:
    # Heavy imports are reachable only from the authorized worker.
    from src.opd.production_b2_calibration_backend_v1 import (
        create_production_b2_calibration_session_v1,
    )

    return create_production_b2_calibration_session_v1(
        config, config_path=config_path
    )


def _default_prompt_provider(
    config: Mapping[str, Any], step_index: int
) -> Sequence[Mapping[str, Any]]:
    from src.opd.production_qualification_aux_gpu_v6 import (
        _source_real_b2_prompt_batch,
    )

    return _source_real_b2_prompt_batch(config, step_index)


def _worker_failure_classification(error: BaseException) -> dict[str, Any]:
    message = str(error).lower()
    if "data_manifest_identity" in message or any(
        marker in message
        for marker in (
            "manifest",
            "schedule",
            "prompt-only",
            "prompt_only",
            "source role",
        )
    ):
        code = "failed_b2_calibration_data_manifest_identity"
        phase = "pre_model_semantic_gate" if "data_manifest_identity" in message else "prompt_provider"
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
        "causal_chain": [code, f"{type(error).__name__}"],
    }


def execute_calibration_worker(
    *,
    package_audit: Mapping[str, Any],
    output_dir: str | Path,
    execution_mode: str,
    session_factory: Callable[[Mapping[str, Any], Path], Any] | None = None,
    prompt_provider: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]]
    | None = None,
    git_commit: str,
) -> dict[str, Any]:
    if execution_mode not in {"formal_gpu", "mock"}:
        _fail("worker execution mode is invalid")
    if package_audit.get("package_version") in {"p4_8b_v2", "p4_8c_v3"}:
        # This is intentionally the first operational action.  Reopen the
        # package, canonical manifest, payloads, and all 80 schedule slots from
        # disk before artifact-store creation or any model/session factory.
        if package_audit.get("package_version") == "p4_8c_v3":
            from src.opd.production_b2_calibration_package_v3 import (
                B2CalibrationPackageV3Error as SemanticPackageError,
                pre_model_semantic_preflight_v3 as semantic_preflight,
            )
        else:
            from src.opd.production_b2_calibration_package_v2 import (
                B2CalibrationPackageV2Error as SemanticPackageError,
                pre_model_semantic_preflight as semantic_preflight,
            )

        try:
            semantic = semantic_preflight(
                package_audit["package_dir"],
                canonical_manifest_path=package_audit["data_authority"][
                    "manifest_path"
                ],
            )
        except (KeyError, SemanticPackageError) as error:
            raise B2CalibrationLauncherV1Error(
                "failed_b2_calibration_data_manifest_identity: "
                f"{type(error).__name__}:{error}"
            ) from error
        package_audit = semantic["audit"]
    output = Path(output_dir).resolve()
    runtime_run_id = str(package_audit.get("runtime_run_id") or RUN_ID)
    runtime_config = build_package_derived_runtime_config(
        package_audit, output_dir=output
    )
    selected_response_length = int(package_audit["selected_response_length"])
    runtime_config_sha = canonical_json_sha256(runtime_config)
    store = B2CalibrationArtifactStoreV1(
        output,
        run_id=runtime_run_id,
        config={
            "run_id": runtime_run_id,
            "stage": "b2_calibration",
            "optimizer_steps": B2_CALIBRATION_STEPS,
            "selected_response_length": selected_response_length,
            "seed": 42,
            "student_initialization": FRESH_STUDENT_INITIALIZATION,
            "checkpoint_strategy": "step10_step20_and_final",
            "automatically_start_formal_b2": False,
            "runtime_config_sha256": runtime_config_sha,
        },
        metadata=_worker_metadata(
            execution_mode=execution_mode,
            git_commit=git_commit,
            run_id=runtime_run_id,
        ),
        package_binding=_package_binding(package_audit),
        data_manifest=_data_manifest(package_audit),
    )
    store.initialize()
    _append_safe_log(
        output / "stdout.log",
        f"worker_start mode={execution_mode} steps={B2_CALIBRATION_STEPS} length={selected_response_length}",
    )
    _atomic_json(output / "runtime_config.json", runtime_config)
    session = None
    step_records: list[Mapping[str, Any]] = []
    try:
        factory = session_factory or _default_session_factory
        provider = prompt_provider or _default_prompt_provider
        session = factory(runtime_config, output / "runtime_config.json")
        identity = session.initial_calibration_identity()
        try:
            store.commit_initial_identity(identity)
        except B2CalibrationArtifactsV1Error as error:
            raise B2CalibrationLauncherV1Error(
                f"fresh Student identity rejected: {error}"
            ) from error
        for step_index in range(B2_CALIBRATION_STEPS):
            prompts = list(provider(runtime_config, step_index))
            if len(prompts) != 4:
                _fail(f"step {step_index + 1} prompt batch is not 2+2")
            record = session.run_b2_calibration_step_v1(
                step_index=step_index,
                prompt_rows=prompts,
                max_new_tokens=selected_response_length,
            )
            logical_version = step_index + 1
            if logical_version in {10, 20}:
                checkpoint_started = time.perf_counter()
                session.save_b2_resume_checkpoint_v1(
                    logical_version=logical_version,
                    package_content_sha256=package_audit[
                        "package_content_sha256"
                    ],
                    config_sha256=runtime_config_sha,
                    data_cursor=logical_version * 4,
                )
                if logical_version == 10:
                    resume_identity = session.reload_b2_resume_checkpoint_v1(
                        logical_version=10,
                        package_content_sha256=package_audit[
                            "package_content_sha256"
                        ],
                        config_sha256=runtime_config_sha,
                        data_cursor=40,
                    )
                    store.commit_resume_reload(resume_identity)
                checkpoint_elapsed = time.perf_counter() - checkpoint_started
                record = deepcopy(dict(record))
                timings = deepcopy(dict(record["timings_seconds"]))
                timings["checkpoint"] = (
                    float(timings["checkpoint"]) + checkpoint_elapsed
                )
                timings["step"] = float(timings["step"]) + checkpoint_elapsed
                record["timings_seconds"] = timings
            store.commit_step(record)
            _append_safe_log(
                output / "stdout.log",
                f"optimizer_step={logical_version} policy=v{logical_version} committed=true",
            )
            step_records.append(record)
            if len(step_records) >= 4:
                live_length = evaluate_latest_calibration_length_window(
                    step_records,
                    selected_response_length=selected_response_length,
                )
                if live_length["passed"] is not True:
                    _atomic_json(
                        output / "length_abort_recommendation.json",
                        live_length,
                    )
                    raise B2CalibrationLauncherV1Error(
                        f"frozen {selected_response_length} calibration length "
                        "window failed; no same-run length switch is allowed"
                    )
        store.commit_final_reload(session.final_checkpoint_reload_identity_v1())
        result = {
            "schema_version": 1,
            "artifact_kind": "b2_calibration_worker_status_v1",
            "status": "worker_completed_exactly_20_steps",
            "steps_completed": B2_CALIBRATION_STEPS,
            "B2_calibration_started": True,
            "B2_calibration_complete": False,
            "B2_formal_authorized": False,
        }
        _atomic_json(output / "worker_status.json", result)
        _append_safe_log(
            output / "stdout.log", "worker_complete optimizer_steps=20"
        )
        return result
    except B2CalibrationLauncherV1Error as error:
        failure = _worker_failure_classification(error)
        completed = sum(
            1 for _ in (output / "metrics.jsonl").open("r", encoding="utf-8")
        )
        _append_safe_log(output / "stdout.log", "worker_failed type=launcher_gate")
        _atomic_json(
            output / "worker_status.json",
            {
                "schema_version": 1,
                "artifact_kind": "b2_calibration_worker_status_v1",
                "status": "worker_failed",
                "primary_failure_code": failure["primary_failure_code"],
                "failure_phase": failure["failure_phase"],
                "completed_steps": completed,
                "requested_steps": B2_CALIBRATION_STEPS,
                "steps_completed": completed,
                "unmet_success_gates": [
                    "optimizer_step_count_is_not_exactly_20"
                ]
                if completed != B2_CALIBRATION_STEPS
                else [],
                "causal_chain": failure["causal_chain"],
                "B2_calibration_complete": False,
                "B2_formal_authorized": False,
            },
        )
        raise
    except Exception as error:
        failure = _worker_failure_classification(error)
        completed = sum(
            1 for _ in (output / "metrics.jsonl").open("r", encoding="utf-8")
        )
        _append_safe_log(
            output / "stdout.log", f"worker_failed type={type(error).__name__}"
        )
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
                "requested_steps": B2_CALIBRATION_STEPS,
                "steps_completed": completed,
                "unmet_success_gates": [
                    "optimizer_step_count_is_not_exactly_20"
                ]
                if completed != B2_CALIBRATION_STEPS
                else [],
                "causal_chain": failure["causal_chain"],
                "B2_calibration_complete": False,
                "B2_formal_authorized": False,
            },
        )
        raise B2CalibrationLauncherV1Error(
            f"calibration worker failed: {type(error).__name__}:{error}"
        ) from error
    finally:
        if session is not None:
            session.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--host-preflight", action="store_true")
    modes.add_argument("--execute-worker", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    modes.add_argument("--mock-20-step", action="store_true")
    modes.add_argument("--formal-b2-shell-dry-run", action="store_true")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--launch-spec", type=Path, default=DEFAULT_LAUNCH_SPEC)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--allow-b2-calibration", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    return parser


def _post_worker_cleanup_observation(output: Path) -> dict[str, Any]:
    metadata_path = output / "metadata.json"
    execution_mode = None
    if metadata_path.is_file() and not metadata_path.is_symlink():
        try:
            execution_mode = json.loads(
                metadata_path.read_text(encoding="utf-8")
            ).get("execution_mode")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            execution_mode = None
    memory: list[int] = []
    compute_pids: list[int] = []
    if execution_mode == "formal_gpu":
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
                ).stdout.splitlines()
                if row.strip()
            ]
        except (OSError, ValueError, subprocess.CalledProcessError):
            memory = []
            compute_pids = []
    else:
        memory = [0, 0]
    residual: list[int] = []
    try:
        rows = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for row in rows:
            fields = row.strip().split(maxsplit=1)
            if len(fields) != 2:
                continue
            pid = int(fields[0])
            command = fields[1]
            if pid == os.getpid():
                continue
            if any(
                marker in command
                for marker in (
                    "--execute-worker",
                    "torchrun",
                    "ray::",
                    "vllm.entrypoints",
                    "verl.trainer",
                )
            ):
                residual.append(pid)
    except (OSError, ValueError, subprocess.CalledProcessError):
        residual = [-1]
    complete = bool(
        len(memory) == 2
        and all(0 <= value <= 16 for value in memory)
        and not compute_pids
        and not residual
    )
    return {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_post_worker_cleanup_v1",
        "worker_exited": True,
        "cleanup_complete": complete,
        "gpu_memory_used_mib": memory,
        "compute_pids": compute_pids,
        "residual_worker_pids": residual,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.formal_b2_shell_dry_run:
        if args.calibration_root is None:
            _fail("formal B2 dry-run requires --calibration-root")
        candidate = assert_formal_b2_calibration_candidate(args.calibration_root)
        print(json.dumps(candidate, sort_keys=True))
        return 0
    spec = load_launch_spec(args.launch_spec)
    if not (
        args.package.resolve()
        == Path(str(spec["source_package"]["path"])).resolve()
        and args.output_root.resolve()
        == Path(str(spec["run"]["output_dir"])).resolve()
    ):
        _fail("loose package/output override differs from the canonical launch spec")
    if args.dry_run:
        result, _audit = run_package_bound_preflight(
            spec,
            mode="dry-run",
            allow_dirty_for_development=args.allow_dirty_for_development,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.host_preflight:
        result, _audit = run_package_bound_preflight(
            spec, mode="host-preflight"
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.execute_worker:
        authorize_gpu_execution(
            os.environ, allow_argument=args.allow_b2_calibration
        )
        result, audit = run_package_bound_preflight(spec, mode="execute")
        git_commit = str(result["git"]["head"])
        install_worker_signal_handlers()
        worker = execute_calibration_worker(
            package_audit=audit,
            output_dir=args.output_root,
            execution_mode="formal_gpu",
            git_commit=git_commit,
        )
        print(json.dumps(worker, sort_keys=True))
        return 0
    if args.finalize:
        verify_current_launch_bindings(spec)
        verify_parent_and_static_assets(spec)
        verify_p4_7_package(
            args.package, expected=_expected_from_spec(spec)
        )
        summary = finalize_calibration_run(
            args.output_root,
            cleanup_observation=_post_worker_cleanup_observation(
                args.output_root
            ),
        )
        print(json.dumps(summary, sort_keys=True))
        return (
            0
            if summary.get("status")
            == "b2_calibration_complete_ready_for_b2_formal"
            else 2
        )
    if args.mock_20_step:
        _fail(
            "CLI mock mode is intentionally test-injected only and cannot emit "
            "formal artifacts"
        )
    _fail("unknown P4.8 launcher mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        B2CalibrationLauncherV1Error,
        B2CalibrationPreflightV1Error,
        B2CalibrationArtifactsV1Error,
    ) as error:
        print(f"P4.8 B2 calibration refused: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "B2CalibrationLauncherV1Error",
    "DEFAULT_OUTPUT",
    "DEFAULT_PACKAGE",
    "DEFAULT_LAUNCH_SPEC",
    "DEFAULT_RUN_CARD",
    "RUN_ID",
    "authorize_gpu_execution",
    "build_argument_parser",
    "build_package_derived_runtime_config",
    "execute_calibration_worker",
    "install_worker_signal_handlers",
    "load_launch_spec",
    "main",
    "run_package_bound_preflight",
    "verify_current_launch_bindings",
    "verify_checked_in_run_card",
    "verify_parent_and_static_assets",
]
