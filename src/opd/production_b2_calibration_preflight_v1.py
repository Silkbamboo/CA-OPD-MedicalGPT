"""CPU-safe package and host preflight for P4.8 B2 calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

from src.opd.production_b2_calibration_contract_v1 import (
    B2_CALIBRATION_STEPS,
    FRESH_STUDENT_INITIALIZATION,
    MINIMUM_DISK_BYTES,
    SELECTED_RESPONSE_LENGTH,
    canonical_json_sha256,
)


EXPECTED_BRANCH = "codex/p4-8-b2-calibration-launcher"
PROJECTED_INCREMENT_BYTES = 4 * 1024**3
PACKAGE_FILES = {
    "b2_20_step_calibration_config.json",
    "b2_20_step_calibration_run_card.json",
    "b2_authorization.json",
}


class B2CalibrationPreflightV1Error(RuntimeError):
    """The immutable package, host or launch identity failed closed."""


def _fail(message: str) -> None:
    raise B2CalibrationPreflightV1Error(message)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise B2CalibrationPreflightV1Error(
            f"cannot stream {path.name}: {type(error).__name__}"
        ) from error
    return digest.hexdigest()


def _safe_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _safe_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2CalibrationPreflightV1Error(
            f"{label} is not valid JSON: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _sha(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _contains_restricted_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            restricted = any(
                token in key for token in ("final", "controller", "confirmation", "label")
            )
            if restricted and (
                (isinstance(child, str) and bool(child))
                or child is True
                or isinstance(child, (list, dict))
            ):
                return True
            if _contains_restricted_path(child):
                return True
    elif isinstance(value, list):
        return any(_contains_restricted_path(child) for child in value)
    return False


def _ordered_adapter_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        item = _safe_file(path / name, f"Teacher {name}")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_static_package_bindings(
    config: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    teacher = config.get("teacher")
    data = config.get("data")
    bindings = authorization.get("bindings")
    if not all(isinstance(value, Mapping) for value in (teacher, data, bindings)):
        _fail("Teacher/data package bindings are incomplete")
    teacher_path = Path(str(teacher.get("adapter_path", "")))
    manifest_path = Path(str(teacher.get("manifest_path", "")))
    data_path = Path(str(data.get("prompt_manifest_path", "")))
    teacher_ordered = _ordered_adapter_sha256(teacher_path)
    teacher_weight = _stream_sha256(
        _safe_file(teacher_path / "adapter_model.safetensors", "Teacher weights")
    )
    teacher_manifest = _stream_sha256(_safe_file(manifest_path, "Teacher manifest"))
    data_manifest = _stream_sha256(_safe_file(data_path, "data manifest"))
    if not (
        teacher.get("role") == "single_frozen_medical_teacher"
        and teacher.get("same_token_scoring") is True
        and teacher.get("adapter_sha256") == teacher_ordered
        and teacher.get("adapter_weight_sha256") == teacher_weight
        and teacher.get("manifest_sha256") == teacher_manifest
        and bindings.get("teacher_adapter_sha256") == teacher_ordered
        and bindings.get("teacher_manifest_sha256") == teacher_manifest
    ):
        _fail("Teacher identity differs from the authorized package")
    if not (
        data.get("prompt_manifest_sha256") == data_manifest
        and bindings.get("data_manifest_sha256") == data_manifest
        and data.get("prompt_only") is True
        and data.get("final_labels_allowed") is False
        and data.get("allowed_roles") == ["medical_opd_o1", "medical_opd_cmb"]
        and data.get("selection_rule")
        == "seed42_sha256_rank_first2_per_source_per_step_v1"
    ):
        _fail("data manifest differs from the authorized package")
    return {
        "teacher_adapter_sha256": teacher_ordered,
        "teacher_weight_sha256": teacher_weight,
        "teacher_manifest_sha256": teacher_manifest,
        "data_manifest_sha256": data_manifest,
    }


def verify_p4_7_package(
    package_dir: str | Path,
    *,
    expected: Mapping[str, Any],
    root_gate: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reopen the exact P4.7 package and sealed length graph from disk."""

    package = Path(package_dir).resolve()
    if package.is_symlink() or not package.is_dir():
        _fail("P4.7 package directory is absent or a symlink")
    actual_names = {path.name for path in package.iterdir()}
    if actual_names != PACKAGE_FILES or any(
        path.is_symlink() or not path.is_file() for path in package.iterdir()
    ):
        _fail("P4.7 package file graph is not exact")
    config_path = package / "b2_20_step_calibration_config.json"
    card_path = package / "b2_20_step_calibration_run_card.json"
    authorization_path = package / "b2_authorization.json"
    config_sha = _stream_sha256(config_path)
    card_sha = _stream_sha256(card_path)
    authorization_sha = _stream_sha256(authorization_path)
    if config_sha != expected.get("config_sha256"):
        _fail("P4.7 B2 config SHA mismatch")
    if card_sha != expected.get("run_card_sha256"):
        _fail("P4.7 B2 run-card SHA mismatch")
    if authorization_sha != expected.get("authorization_sha256"):
        _fail("P4.7 B2 authorization SHA mismatch")
    config = _read_json(config_path, "P4.7 B2 config")
    card = _read_json(card_path, "P4.7 B2 run card")
    authorization = _read_json(authorization_path, "P4.7 B2 authorization")
    content = dict(authorization)
    claimed_content = content.pop("package_content_sha256", None)
    if not (
        claimed_content == expected.get("package_content_sha256")
        and claimed_content == canonical_json_sha256(content)
    ):
        _fail("P4.7 B2 package content SHA mismatch")
    if _contains_restricted_path(config):
        _fail("restricted final/controller/confirmation/label path is present")
    run = config.get("run")
    generation = config.get("generation")
    protocol = config.get("protocol")
    backend = config.get("production_backend")
    qualification = config.get("qualification")
    config_authorization = config.get("authorization")
    execution = config.get("execution")
    isolation = config.get("isolation")
    if not all(
        isinstance(value, Mapping)
        for value in (
            run,
            generation,
            protocol,
            backend,
            qualification,
            config_authorization,
            execution,
            isolation,
        )
    ):
        _fail("P4.7 B2 config sections are incomplete")
    if not (
        authorization.get("status") == "authorized_not_started"
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_started") is False
        and authorization.get("requires_explicit_allow_b2_calibration") is True
        and authorization.get("selected_response_length") == SELECTED_RESPONSE_LENGTH
        and run.get("stage") == "b2_medical_opd_calibration"
        and run.get("seed") == 42
        and run.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and run.get("automatically_start") is False
        and generation.get("max_new_tokens") == SELECTED_RESPONSE_LENGTH
        and generation.get("do_sample") is True
        and generation.get("temperature") == 1.0
        and generation.get("top_k") == 0
        and generation.get("top_p") == 1.0
        and generation.get("full_support") is True
        and generation.get("enable_thinking") is False
        and generation.get("use_cache") is True
        and execution.get("optimizer_steps") == B2_CALIBRATION_STEPS
        and execution.get("calibration_only") is True
        and config_authorization.get("B2_authorized") is True
        and config_authorization.get("B2_started") is False
        and backend.get("backend_id") == "custom_transformers_peft_three_policy_v5"
        and backend.get("refresh_implementation")
        == "peft_0_17_1_hotswap_stable_slot"
        and backend.get("adapter_runtime_slot") == "student_active"
        and backend.get("vllm_used") is False
        and all(isolation.get(field) is False for field in (
            "final_access", "controller_access", "confirmation_access", "label_access"
        ))
    ):
        _fail("P4.7 package is not frozen to 768 and exactly 20 steps")
    if not (
        protocol.get("optimizer") == "AdamW"
        and protocol.get("learning_rate") == 3e-5
        and protocol.get("student_lora_rank") == 16
        and protocol.get("student_lora_alpha") == 32
        and protocol.get("student_lora_target_modules") == "all-linear"
        and protocol.get("correction_upper_threshold") == 2.0
        and protocol.get("ppo_clip_low") == 0.2
        and protocol.get("ppo_clip_high") == 0.28
        and protocol.get("prompt_equal_reduction") is True
    ):
        _fail("P4.7 loss/optimizer/LoRA contract drift")
    if "student_initialization" in config:
        _fail("qualification v2 or another adapter cannot initialize B2")
    v2_path = Path(str(qualification.get("v2_checkpoint_path", ""))).resolve()
    if not (
        v2_path == Path(str(expected.get("qualification_v2_path", ""))).resolve()
        and qualification.get("v2_tensor_sha256")
        == expected.get("qualification_v2_tensor_sha256")
    ):
        _fail("qualification v2 evidence binding drift")
    if expected.get("student_initialization") != FRESH_STUDENT_INITIALIZATION:
        _fail("Student initialization contract is not fresh Base plus zero-LoRA")
    if not (
        card.get("config_sha256") == config_sha
        and card.get("authorization_content_sha256") == claimed_content
        and card.get("selected_response_length") == SELECTED_RESPONSE_LENGTH
        and card.get("steps") == B2_CALIBRATION_STEPS
        and card.get("requires_argument") == "--allow-b2-calibration"
        and card.get("automatically_start") is False
        and card.get("B2_started") is False
    ):
        _fail("P4.7 B2 run-card binding drift")
    static = _validate_static_package_bindings(config, authorization)
    if root_gate is None:
        # The original P4.8 package was internally hash-consistent but bound a
        # pre-freeze manifest.  Formal paths now reuse the production semantic
        # resolver here, before any model/session construction.  Synthetic unit
        # fixtures that inject an explicit root gate remain isolated from real
        # production authority.
        from src.opd.production_b2_data_v2 import (
            B2DataAuthorityV2Error,
            CANONICAL_MANIFEST_PATH,
            resolve_b2_data_authority,
        )

        try:
            resolve_b2_data_authority(
                config["data"]["prompt_manifest_path"],
                expected_manifest_sha256=static["data_manifest_sha256"],
                canonical_manifest_path=CANONICAL_MANIFEST_PATH,
            )
        except B2DataAuthorityV2Error as error:
            raise B2CalibrationPreflightV1Error(
                "P4.7 package data manifest lacks canonical production authority: "
                f"{error}"
            ) from error
    formal_root = Path(str(expected.get("formal_run_root", ""))).resolve()
    if Path(str(qualification.get("output_path", ""))).resolve() != formal_root:
        _fail("P4.7 formal length root binding drift")
    if root_gate is None:
        from src.opd.production_length_gpu_runtime_v7 import (
            assert_b2_calibration_start_authorized,
        )

        root_gate = assert_b2_calibration_start_authorized
        authority_revalidation = _read_json(
            formal_root / "finalizer_authority_revalidation.json",
            "P4.7 finalizer authority",
        )
    else:
        authority_revalidation = {}
    try:
        root_gate(
            authorization,
            allow_b2_calibration=True,
            formal_run_root=formal_root,
            package_dir=package,
            authority_revalidation=authority_revalidation,
        )
    except B2CalibrationPreflightV1Error:
        raise
    except Exception as error:
        raise B2CalibrationPreflightV1Error(
            f"P4.7 sealed root gate failed: {type(error).__name__}:{error}"
        ) from error
    return {
        "package_dir": str(package),
        "formal_run_root": str(formal_root),
        "config_sha256": config_sha,
        "run_card_sha256": card_sha,
        "authorization_sha256": authorization_sha,
        "package_content_sha256": claimed_content,
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "optimizer_steps": B2_CALIBRATION_STEPS,
        "seed": 42,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "qualification_v2_usage": "evidence_only_not_student_init",
        "qualification_v2_path": str(v2_path),
        "qualification_v2_tensor_sha256": qualification["v2_tensor_sha256"],
        "teacher": static,
        "config": config,
        "authorization": authorization,
    }


def reject_runtime_overrides(overrides: Mapping[str, Any] | None) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping) or overrides:
        _fail("runtime hyperparameter override is forbidden")


def _default_git_probe() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return {"branch": branch, "head": head, "clean": not bool(status)}


def _default_gpu_probe() -> dict[str, Any]:
    """Reuse the P4.7 dual-3090 host contract without importing Torch."""

    import yaml

    from src.opd.production_length_preflight_v7 import _gpu_host

    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/opd/qwen3_4b_length_qualification_v7.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise B2CalibrationPreflightV1Error(
            f"cannot load frozen GPU host contract: {type(error).__name__}"
        ) from error
    if not isinstance(config, Mapping):
        _fail("frozen GPU host contract is invalid")
    try:
        return dict(_gpu_host(config))
    except Exception as error:
        raise B2CalibrationPreflightV1Error(
            f"GPU host query failed: {type(error).__name__}:{error}"
        ) from error


def preflight_b2_calibration(
    package_dir: str | Path,
    *,
    output_dir: str | Path,
    expected: Mapping[str, Any],
    mode: str,
    root_gate: Callable[..., Mapping[str, Any]] | None = None,
    git_probe: Callable[[], Mapping[str, Any]] | None = None,
    disk_free_probe: Callable[[Path], int] | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] | None = None,
    allow_dirty_for_development: bool = False,
) -> dict[str, Any]:
    if mode not in {"dry-run", "host-preflight", "execute"}:
        _fail("unknown B2 preflight mode")
    audit = verify_p4_7_package(package_dir, expected=expected, root_gate=root_gate)
    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        _fail("B2 calibration output must be fresh")
    git = dict((git_probe or _default_git_probe)())
    if mode != "dry-run" or not allow_dirty_for_development:
        if not (
            git.get("branch") == expected.get("branch", EXPECTED_BRANCH)
            and git.get("clean") is True
            and isinstance(git.get("head"), str)
            and len(git["head"]) == 40
            and (
                expected.get("git_commit") is None
                or git["head"] == expected.get("git_commit")
            )
        ):
            _fail("formal B2 execution requires the exact clean committed worktree")
    free = int(
        (disk_free_probe or (lambda path: shutil.disk_usage(path.parent).free))(output)
    )
    if free - int(expected.get("projected_increment_bytes", PROJECTED_INCREMENT_BYTES)) <= MINIMUM_DISK_BYTES:
        _fail("projected persistent disk is not strictly above 10 GiB")
    gpu_host: dict[str, Any] | None = None
    gpus: list[dict[str, Any]] | None = None
    if mode in {"host-preflight", "execute"}:
        observed = (gpu_probe or _default_gpu_probe)()
        if not isinstance(observed, Mapping):
            _fail("formal B2 host requires exactly two idle RTX 3090 GPUs")
        gpu_host = dict(observed)
        raw_gpus = gpu_host.get("gpus")
        if not isinstance(raw_gpus, list):
            _fail("formal B2 host requires exactly two idle RTX 3090 GPUs")
        gpus = [dict(item) for item in raw_gpus if isinstance(item, Mapping)]
        if not (
            len(gpus) == 2
            and all(item.get("name") == "NVIDIA GeForce RTX 3090" for item in gpus)
            and all(int(item.get("total_mib", 0)) >= 24000 for item in gpus)
            and all(int(item.get("used_mib", 1_000_000)) <= 16 for item in gpus)
            and isinstance(gpu_host.get("driver_version"), str)
            and bool(gpu_host["driver_version"])
            and isinstance(gpu_host.get("cuda_version"), str)
            and bool(gpu_host["cuda_version"])
            and isinstance(gpu_host.get("topology"), str)
            and "GPU0" in gpu_host["topology"]
            and "GPU1" in gpu_host["topology"]
            and gpu_host.get("compute_processes") == []
            and gpu_host.get("residual_worker_pids") == []
        ):
            _fail("formal B2 host requires exactly two idle RTX 3090 GPUs")
    return {
        "schema_version": 1,
        "artifact_kind": "p4_8_b2_calibration_preflight_v1",
        "status": "ready_waiting_for_gpu_b2_calibration",
        "mode": mode,
        "package_audit_sha256": canonical_json_sha256(
            {key: value for key, value in audit.items() if key not in {"config", "authorization"}}
        ),
        "selected_response_length": SELECTED_RESPONSE_LENGTH,
        "optimizer_steps": B2_CALIBRATION_STEPS,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "disk_free_bytes": free,
        "projected_increment_bytes": int(
            expected.get("projected_increment_bytes", PROJECTED_INCREMENT_BYTES)
        ),
        "git": git,
        "gpus": gpus,
        "gpu_host": gpu_host,
        "B2_authorized": True,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }


__all__ = [
    "B2CalibrationPreflightV1Error",
    "EXPECTED_BRANCH",
    "PACKAGE_FILES",
    "PROJECTED_INCREMENT_BYTES",
    "preflight_b2_calibration",
    "reject_runtime_overrides",
    "verify_p4_7_package",
]
