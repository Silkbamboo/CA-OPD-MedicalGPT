"""Artifact graph, readiness, and B2 authorization for P4.6.

This module is intentionally CPU-only.  It never imports a model runtime and
never trusts a caller-provided ``ready`` value.  A phase is committed once,
metrics are durably appended once, and terminal state is derived by reopening
the files and checking their complete SHA graph.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from src.opd.production_qualification_contract_v6 import (
    QualificationContractError,
    build_probe_spec,
    validate_probe_manifest,
    validate_v0_guard_evidence,
)

from src.opd.production_qualification_preflight_v6 import (
    ProductionQualificationPreflightError,
    validate_base_model_transport as _validate_base_model_transport,
)


SCHEMA_VERSION = 6
ARTIFACT_PROTOCOL_VERSION = "p4.6-combined-production-qualification-v6"
REPO_ROOT = Path(__file__).resolve().parents[2]

B2_CALIBRATION_SCHEMA_ID = "ca-opd/b2-medical-opd-calibration/v1"
B2_CALIBRATION_SCHEMA_VERSION = 1
B2_CALIBRATION_RUN_ID = "qwen3-4b-b2-medical-opd-calibration-v1"
B2_PRODUCTION_BACKEND_ID = "custom_transformers_peft_three_policy_v5"
B2_EXECUTOR_SOURCE_PATH: str | Path = (
    "src/opd/production_qualification_gpu_runtime_v6.py"
)
B2_EXECUTOR_SYMBOL = "execute_b2_medical_opd_gpu_protocol_v6"
B2_TEMPLATE_CONFIG_PATH = Path(
    "configs/runs/b2_medical_opd_qwen3_4b_custom_v5_p4_6.yaml"
)
B2_TEMPLATE_CONFIG_SHA256 = (
    "4ddfb9abd0349bcb1c778fec3742bee12981b079d4e1882a5a22f4fe87479ada"
)
B2_TEMPLATE_RUN_CARD_PATH = Path(
    "configs/run_cards/qwen3-4b-b2-medical-opd-custom-v5-p4-6-seed42.json"
)
B2_TEMPLATE_RUN_CARD_SHA256 = (
    "70d4f7eb7f32478d5753380d9c84437ba59b6b9354e13e1404447dfc64021f9e"
)
B2_CALIBRATION_OPTIMIZER_STEPS = 20
B2_CALIBRATION_SELECTION_RULE = (
    "seed42_sha256_rank_first2_per_source_per_step_v1"
)

_REQUIRED_EXECUTABLE_SOURCE_PATHS = frozenset(
    {
        "src/data/chat.py",
        "src/opd/production_qualification_gpu_runtime_v6.py",
        "src/opd/production_qualification_v6.py",
        "src/opd/production_qualification_artifacts_v6.py",
        "src/opd/production_qualification_preflight_v6.py",
        "src/opd/production_qualification_two_step_gpu_v6.py",
        "src/opd/production_qualification_aux_gpu_v6.py",
        "src/opd/calibration_data.py",
        "src/opd/pg_opd_contract.py",
        "src/opd/pg_opd_validation.py",
        "src/opd/production_qualification_contract_v6.py",
        "src/opd/production_qualification_telemetry_v6.py",
        "src/opd/production_qualification_prompts_v6.py",
        "src/opd/production_sampler_identity_v5.py",
        "src/opd/production_sampler_refresh_v5.py",
        "src/opd/rollout_correction_adapter.py",
        "src/opd/rollout_probability.py",
        "src/opd/scorer_gpu_calibration.py",
        "src/opd/production.py",
        "src/opd/production_backend_binding_v5.py",
    }
)
B2_EXECUTOR_BLOCKER = "B2_PRODUCTION_PACKAGE_NOT_EXECUTABLE"

FULL_PHASES = (
    "launch_record",
    "preflight",
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
    "cleanup",
    "terminal_summary",
)

MICRO_PHASES = (
    "launch_record",
    "preflight",
    "probe_manifest",
    "v0_guard",
    "reconstruction_step0",
    "authority_v1",
    "refresh_v1",
    "runtime_release",
    "cleanup",
    "terminal_summary",
)

STATIC_ARTIFACT_FILES = (
    "metadata.json",
    "config.yaml",
    "run_card.json",
    "artifact_schema.json",
    "protocol.yaml",
    "backend_binding.json",
    "prompt_manifest.json",
)

EVIDENCE_INDEX_FILE = "evidence_artifact_index.json"
EVIDENCE_READINESS_FILE = "evidence_readiness.json"
FINAL_INDEX_FILE = "artifact_index.json"
FINAL_READINESS_FILE = "readiness.json"
ROOT_AUTHORIZATION_FILE = "b2_authorization.json"
FAILURE_CLEANUP_FILE = "failure_cleanup.json"
MICRO_READINESS_FILE = "micro_readiness.json"

_EVIDENCE_INDEX_EXCLUSIONS = frozenset(
    {
        EVIDENCE_INDEX_FILE,
        EVIDENCE_READINESS_FILE,
        FINAL_INDEX_FILE,
        FINAL_READINESS_FILE,
        ROOT_AUTHORIZATION_FILE,
    }
)
_FINAL_INDEX_EXCLUSIONS = frozenset({FINAL_INDEX_FILE, FINAL_READINESS_FILE})

_STATIC_SOURCE_FILES = {
    "config": ("config.yaml", "config_sha256"),
    "run_card": ("run_card.json", "run_card_sha256"),
    "artifact_schema": ("artifact_schema.json", "schema_sha256"),
    "protocol": ("protocol.yaml", "protocol_sha256"),
    "prompt_manifest": ("prompt_manifest.json", "prompt_manifest_sha256"),
}

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

_ENVELOPE_FIELDS = {
    "schema_version",
    "artifact_protocol_version",
    *_BINDING_FIELDS,
    "phase_id",
    "ordinal",
    "previous_phase_path",
    "previous_phase_sha256",
    "probe_manifest_content_sha256",
    "probe_manifest_file_sha256",
    "payload_sha256",
    "payload",
}

_HASH_BINDINGS = {
    "config_sha256",
    "run_card_sha256",
    "schema_sha256",
    "protocol_sha256",
    "backend_binding_sha256",
    "prompt_manifest_sha256",
    "probe_spec_sha256",
    "data_manifest_sha256",
}

_ISOLATION = {
    "final_access": False,
    "controller_access": False,
    "confirmation_access": False,
    "label_access": False,
}


class QualificationArtifactError(RuntimeError):
    """Raised when a P4.6 artifact operation cannot be performed safely."""


def validate_base_model_transport(**kwargs: Any) -> dict[str, Any]:
    """Translate the shared CPU transport verifier into artifact failures."""

    try:
        return _validate_base_model_transport(**kwargs)
    except ProductionQualificationPreflightError as error:
        raise QualificationArtifactError(str(error)) from error


def canonical_json_sha256(value: Any) -> str:
    """Hash the canonical JSON representation used by every v6 binding."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implemented_source_symbols(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise QualificationArtifactError(
            f"backend source cannot be parsed: {path}"
        ) from error
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(f"{node.name}.{member.name}")
    return symbols


def _validate_current_backend_source_chain(output: Path) -> dict[str, Any]:
    """Rehash every frozen runtime dependency before package use or B2 start."""

    binding = _read_json(output / "backend_binding.json")
    if binding.get("binding_version") != "p4.6-current-executable-chain-v3":
        raise QualificationArtifactError("backend source-chain version mismatch")
    chain = binding.get("executable_source_chain")
    if not isinstance(chain, list) or not chain:
        raise QualificationArtifactError("backend executable source chain is absent")
    observed: set[str] = set()
    for item in chain:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "required_symbols",
        }:
            raise QualificationArtifactError("backend source-chain entry is invalid")
        relative = item["path"]
        if not isinstance(relative, str):
            raise QualificationArtifactError("backend source-chain path is invalid")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise QualificationArtifactError("backend source-chain path traversal")
        if relative in observed:
            raise QualificationArtifactError("backend source-chain path is duplicated")
        observed.add(relative)
        source = REPO_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise QualificationArtifactError(
                f"backend source-chain file is absent: {relative}"
            )
        if sha256_file(source) != item["sha256"]:
            raise QualificationArtifactError(
                f"backend source-chain SHA mismatch: {relative}"
            )
        required = item["required_symbols"]
        if not isinstance(required, list) or not required or any(
            not isinstance(symbol, str) or not symbol for symbol in required
        ):
            raise QualificationArtifactError(
                f"backend source-chain symbols are invalid: {relative}"
            )
        missing = sorted(set(required) - _implemented_source_symbols(source))
        if missing:
            raise QualificationArtifactError(
                f"backend source-chain symbols missing: {relative}:{','.join(missing)}"
            )
    if observed != _REQUIRED_EXECUTABLE_SOURCE_PATHS:
        raise QualificationArtifactError("backend executable source inventory mismatch")
    launch = _read_json(output / "launch_record.json")
    if (
        canonical_json_sha256(binding) != launch["backend_binding_sha256"]
        or sha256_file(output / "backend_binding.json")
        != launch["backend_binding_sha256"]
    ):
        raise QualificationArtifactError("backend source-chain artifact binding mismatch")
    return binding


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise QualificationArtifactError(f"artifact already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def _atomic_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise QualificationArtifactError(f"artifact already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_append_jsonl(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise QualificationArtifactError("metrics.jsonl cannot be a symlink")
    previous = path.read_bytes() if path.exists() else b""
    if previous and not previous.endswith(b"\n"):
        raise QualificationArtifactError("metrics.jsonl is not newline terminated")
    line = (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".metrics.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationArtifactError(f"regular artifact is absent: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationArtifactError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise QualificationArtifactError(f"artifact must be an object: {path.name}")
    return value


def _reject_unavailable(value: Any, *, path: str) -> None:
    if value is None:
        raise QualificationArtifactError(f"{path} is null/unavailable")
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise QualificationArtifactError(f"{path} is non-finite")
        return
    if isinstance(value, str):
        if not value or value.lower() in {"unavailable", "unknown", "null"}:
            raise QualificationArtifactError(f"{path} is unavailable")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_unavailable(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_unavailable(child, path=f"{path}[{index}]")
        return
    raise QualificationArtifactError(f"{path} is not JSON-compatible")


def _validate_bindings(bindings: Mapping[str, Any]) -> None:
    if set(bindings) != set(_BINDING_FIELDS):
        missing = sorted(set(_BINDING_FIELDS) - set(bindings))
        extra = sorted(set(bindings) - set(_BINDING_FIELDS))
        raise QualificationArtifactError(
            f"binding fields mismatch; missing={missing}, extra={extra}"
        )
    if not isinstance(bindings["run_id"], str) or not bindings["run_id"]:
        raise QualificationArtifactError("run_id must be non-empty")
    if not isinstance(bindings["attempt_id"], str) or not bindings["attempt_id"]:
        raise QualificationArtifactError("attempt_id must be non-empty")
    if not _is_hex(bindings["git_commit"], 40):
        raise QualificationArtifactError("git_commit must be a lowercase 40-hex value")
    for field in _HASH_BINDINGS:
        if not _is_hex(bindings[field], 64):
            raise QualificationArtifactError(f"{field} must be a lowercase SHA-256")
    if bindings["isolation"] != _ISOLATION:
        raise QualificationArtifactError("final/controller/confirmation/label isolation failed")


def _static_metadata(bindings: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "artifact_kind": "p4_6_qualification_static_identity",
        "mode": mode,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "git_commit": bindings["git_commit"],
        "config_sha256": bindings["config_sha256"],
        "run_card_sha256": bindings["run_card_sha256"],
        "schema_sha256": bindings["schema_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "backend_binding_sha256": bindings["backend_binding_sha256"],
        "prompt_manifest_sha256": bindings["prompt_manifest_sha256"],
        "probe_spec_sha256": bindings["probe_spec_sha256"],
        "data_manifest_sha256": bindings["data_manifest_sha256"],
        "isolation": dict(bindings["isolation"]),
        "B2_started": False,
    }


def initialize_qualification_artifacts(
    output: str | Path,
    *,
    bindings: Mapping[str, Any],
    mode: str,
    sources: Mapping[str, str | Path],
    backend_binding: Mapping[str, Any],
) -> dict[str, str]:
    """Atomically create the immutable static identity package.

    The output directory itself is renamed into place only after every source
    byte and canonical backend identity has been verified and fsynced.
    """

    _validate_bindings(bindings)
    _phases(mode)
    if set(sources) != set(_STATIC_SOURCE_FILES):
        raise QualificationArtifactError("static source inventory mismatch")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise QualificationArtifactError("qualification output must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_bytes: dict[str, bytes] = {}
    for source_name, (_, binding_name) in _STATIC_SOURCE_FILES.items():
        source = Path(sources[source_name])
        if source.is_symlink() or not source.is_file():
            raise QualificationArtifactError(f"static source is absent: {source_name}")
        if sha256_file(source) != bindings[binding_name]:
            raise QualificationArtifactError(f"static source SHA mismatch: {source_name}")
        source_bytes[source_name] = source.read_bytes()
    if canonical_json_sha256(backend_binding) != bindings["backend_binding_sha256"]:
        raise QualificationArtifactError("backend binding canonical SHA mismatch")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.static.", dir=destination.parent)
    )
    try:
        written: dict[str, str] = {}
        for source_name, (file_name, _) in _STATIC_SOURCE_FILES.items():
            written[file_name] = _atomic_bytes(
                temporary / file_name, source_bytes[source_name]
            )
        backend_text = json.dumps(
            backend_binding,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        written["backend_binding.json"] = _atomic_text(
            temporary / "backend_binding.json", backend_text
        )
        written["metadata.json"] = _atomic_json(
            temporary / "metadata.json", _static_metadata(bindings, mode=mode)
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return written
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_qualification_static_artifacts(
    output: str | Path,
    *,
    bindings: Mapping[str, Any],
    mode: str,
    require_summary: bool = False,
) -> list[str]:
    """Reopen every static identity and return fail-closed diagnostics."""

    errors: list[str] = []
    try:
        _validate_bindings(bindings)
        _phases(mode)
    except QualificationArtifactError as error:
        return [str(error)]
    directory = Path(output)
    if not directory.is_dir() or directory.is_symlink():
        return ["static_output_directory_invalid"]
    for source_name, (file_name, binding_name) in _STATIC_SOURCE_FILES.items():
        path = directory / file_name
        if path.is_symlink() or not path.is_file():
            errors.append(f"static_missing:{file_name}")
        elif sha256_file(path) != bindings[binding_name]:
            errors.append(f"static_sha_mismatch:{source_name}")
    backend_path = directory / "backend_binding.json"
    if backend_path.is_symlink() or not backend_path.is_file():
        errors.append("static_missing:backend_binding.json")
    else:
        try:
            backend = _read_json(backend_path)
            if (
                canonical_json_sha256(backend) != bindings["backend_binding_sha256"]
                or sha256_file(backend_path) != bindings["backend_binding_sha256"]
            ):
                errors.append("static_sha_mismatch:backend_binding")
        except QualificationArtifactError as error:
            errors.append(str(error))
    metadata_path = directory / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        errors.append("static_missing:metadata.json")
    else:
        try:
            metadata = _read_json(metadata_path)
            if metadata != _static_metadata(bindings, mode=mode):
                errors.append("static_metadata_mismatch")
        except QualificationArtifactError as error:
            errors.append(str(error))
    summary_path = directory / "summary.json"
    terminal_path = directory / "terminal_summary.json"
    if require_summary:
        if summary_path.is_symlink() or not summary_path.is_file():
            errors.append("static_missing:summary.json")
        elif terminal_path.is_symlink() or not terminal_path.is_file():
            errors.append("terminal_summary")
        elif summary_path.read_bytes() != terminal_path.read_bytes():
            errors.append("summary_alias_mismatch")
    return sorted(set(errors))


def write_terminal_summary_alias(output: str | Path) -> str:
    """Persist the required ``summary.json`` as an exact immutable alias."""

    directory = Path(output)
    terminal = directory / "terminal_summary.json"
    if terminal.is_symlink() or not terminal.is_file():
        raise QualificationArtifactError("terminal_summary.json is absent")
    return _atomic_bytes(directory / "summary.json", terminal.read_bytes())


def _phases(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return FULL_PHASES
    if mode == "micro":
        return MICRO_PHASES
    raise QualificationArtifactError(f"unknown qualification mode: {mode}")


def _probe_spec_from_static(
    output: Path, bindings: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise QualificationArtifactError("probe static config is invalid") from error
    fixed = config.get("fixed_action_probe") if isinstance(config, Mapping) else None
    if not isinstance(fixed, Mapping) or not (
        fixed.get("selection_rule")
        == "first_32_valid_response_tokens_per_prompt_v1"
        and fixed.get("per_prompt_limit") == 32
        and fixed.get("no_cross_prompt_backfill") is True
        and fixed.get("freeze_after_rollout_before_optimizer") is True
    ):
        raise QualificationArtifactError("probe static contract drift")
    try:
        spec = build_probe_spec(
            run_id=str(bindings["run_id"]),
            prompt_manifest_sha256=str(bindings["prompt_manifest_sha256"]),
            ordered_sample_ids=fixed.get("ordered_sample_ids"),
        )
    except QualificationContractError as error:
        raise QualificationArtifactError(f"probe spec invalid: {error}") from error
    if not (
        spec["probe_spec_sha256"] == fixed.get("probe_spec_sha256")
        == bindings["probe_spec_sha256"]
    ):
        raise QualificationArtifactError("probe spec/envelope SHA mismatch")
    return spec


def _validate_probe_payload(
    payload: Mapping[str, Any], *, spec: Mapping[str, Any] | None = None
) -> str:
    if payload.get("status") != "pass":
        raise QualificationArtifactError("probe manifest did not pass")
    manifest = dict(payload)
    manifest.pop("status", None)
    if spec is None:
        try:
            spec = build_probe_spec(
                run_id=str(manifest.get("run_id", "")),
                prompt_manifest_sha256=str(
                    manifest.get("prompt_manifest_sha256", "")
                ),
                ordered_sample_ids=manifest.get("ordered_sample_ids"),
            )
        except QualificationContractError as error:
            raise QualificationArtifactError(f"probe spec invalid: {error}") from error
    try:
        validated = validate_probe_manifest(manifest, spec)
    except QualificationContractError as error:
        raise QualificationArtifactError(f"probe manifest invalid: {error}") from error
    return str(validated["manifest_sha256"])


def _validate_reconstruction(payload: Mapping[str, Any], *, phase: str) -> None:
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise QualificationArtifactError("reconstruction telemetry is absent")
    top_fields = {
        "schema_version", "run_id", "step_id", "q_p_old", "advantage",
        "optimizer_update",
    }
    if not top_fields.issubset(telemetry) or telemetry.get("schema_version") != 6:
        raise QualificationArtifactError("reconstruction telemetry fields are incomplete")
    expected_step = {
        "reconstruction_step0": "step0_v0_to_v1",
        "reconstruction_step1": "step1_v1_to_v2",
    }[phase]
    if telemetry.get("step_id") != expected_step:
        raise QualificationArtifactError(f"{phase} step_id mismatch")
    q_old = telemetry.get("q_p_old")
    advantage = telemetry.get("advantage")
    update = telemetry.get("optimizer_update")
    if not all(isinstance(section, Mapping) for section in (q_old, advantage, update)):
        raise QualificationArtifactError("reconstruction telemetry sections are invalid")
    q_fields = {
        "valid_token_count", "prompt_count", "source_count",
        "signed_difference_semantics", "signed_mean", "absolute_difference",
        "pearson", "spearman", "log_w", "raw_is_semantics", "raw_is",
        "capped_is", "token_ess", "ess_fraction", "per_prompt_ess",
        "per_source_ess", "cap_fraction", "finite_rate",
        "current_pre_old_max_abs", "ppo_ratio_pre",
    }
    advantage_fields = {
        "count", "mean", "std", "min", "max", "quantiles",
        "positive_count", "negative_count", "near_zero_count",
        "near_zero_threshold", "finite_rate", "per_prompt_mean",
        "per_source_mean", "aggregation", "teacher_detached", "old_actor_detached",
    }
    update_fields = {
        "objective_before", "objective_after", "objective_delta", "loss_before",
        "loss_after", "loss_delta", "alignment", "ppo_ratio_pre",
        "ppo_ratio_post", "clip_fraction_pre", "clip_fraction_post",
        "gradient_norm_before_clip", "gradient_norm_after_clip",
        "parameter_delta_norm", "relative_parameter_delta",
        "gradient_dot_parameter_delta", "trainable_tensor_count",
        "trainable_parameter_count", "nonzero_update_tensor_count",
        "zero_update_tensor_count", "update_norm_min", "update_norm_max",
        "teacher_gradient_tensor_count", "base_gradient_tensor_count",
        "correction_weight_detached", "optimizer_config", "optimizer_config_sha256",
    }
    missing = {
        "q_p_old": sorted(q_fields - set(q_old)),
        "advantage": sorted(advantage_fields - set(advantage)),
        "optimizer_update": sorted(update_fields - set(update)),
    }
    missing = {section: fields for section, fields in missing.items() if fields}
    if missing:
        raise QualificationArtifactError(f"reconstruction telemetry missing fields: {missing}")
    nested_fields = (
        (q_old, "absolute_difference", {"mae", "p50", "p95", "p99", "max"}),
        (q_old, "log_w", {"mean", "std", "min", "max"}),
        (q_old, "raw_is", {"mean", "std", "min", "max", "p50", "p95", "p99"}),
        (q_old, "capped_is", {"mean", "std", "min", "max"}),
        (advantage, "quantiles", {"p1", "p5", "p50", "p95", "p99"}),
    )
    for section, field, expected in nested_fields:
        _exact_fields(
            section[field],
            expected,
            label=f"reconstruction {field}",
        )
    _reject_unavailable(telemetry, path="reconstruction.telemetry")
    valid_count = q_old["valid_token_count"]
    if not isinstance(valid_count, int) or isinstance(valid_count, bool) or valid_count <= 0:
        raise QualificationArtifactError("reconstruction valid token count is invalid")
    if advantage["count"] != valid_count or (
        advantage["positive_count"]
        + advantage["negative_count"]
        + advantage["near_zero_count"]
        != valid_count
    ):
        raise QualificationArtifactError("reconstruction advantage counts are invalid")
    if (
        update["nonzero_update_tensor_count"] + update["zero_update_tensor_count"]
        != update["trainable_tensor_count"]
    ):
        raise QualificationArtifactError("reconstruction update tensor counts are invalid")
    try:
        prompt_token_count = sum(
            value["token_count"] for value in q_old["per_prompt_ess"].values()
        )
        source_token_count = sum(
            value["token_count"] for value in q_old["per_source_ess"].values()
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise QualificationArtifactError(
            "reconstruction grouped token evidence is invalid"
        ) from error
    if prompt_token_count != valid_count or source_token_count != valid_count:
        raise QualificationArtifactError(
            "reconstruction grouped token counts are invalid"
        )
    if not (
        q_old["finite_rate"] == 1.0
        and advantage["finite_rate"] == 1.0
        and advantage["teacher_detached"] is True
        and advantage["old_actor_detached"] is True
        and update["correction_weight_detached"] is True
    ):
        raise QualificationArtifactError(
            "reconstruction finite/detach evidence gate failed"
        )
    if update["optimizer_config_sha256"] != canonical_json_sha256(
        update["optimizer_config"]
    ):
        raise QualificationArtifactError("reconstruction optimizer config SHA mismatch")
    if q_old["ess_fraction"] < 0.80 or q_old["cap_fraction"] > 0.05:
        raise QualificationArtifactError("reconstruction correction gate failed")
    if q_old["current_pre_old_max_abs"] > 0.0001:
        raise QualificationArtifactError("reconstruction actor identity gate failed")
    if not (
        update["objective_after"] > update["objective_before"]
        and update["loss_after"] < update["loss_before"]
        and update["alignment"] > 0
        and update["nonzero_update_tensor_count"] > 0
        and update["teacher_gradient_tensor_count"] == 0
        and update["base_gradient_tensor_count"] == 0
    ):
        raise QualificationArtifactError("reconstruction optimizer gate failed")


def _require_payload_fields(payload: Mapping[str, Any], fields: set[str], phase: str) -> None:
    missing = sorted(fields - set(payload))
    if missing:
        raise QualificationArtifactError(f"{phase} missing payload fields: {missing}")


def _validate_per_tensor_digests(
    value: Any, *, tensor_count: Any, total_bytes: Any, label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise QualificationArtifactError(f"{label} per-tensor digests are absent")
    digests: list[dict[str, Any]] = []
    keys: set[str] = set()
    byte_sum = 0
    expected = {"canonical_key", "sha256", "shape", "dtype", "byte_length"}
    for item in value:
        item = dict(_exact_fields(item, expected, label=f"{label} tensor digest"))
        key = item["canonical_key"]
        shape = item["shape"]
        if (
            not isinstance(key, str)
            or not key
            or key in keys
            or not _is_hex(item["sha256"], 64)
            or not isinstance(shape, list)
            or not shape
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in shape
            )
            or not isinstance(item["dtype"], str)
            or not item["dtype"]
            or not isinstance(item["byte_length"], int)
            or isinstance(item["byte_length"], bool)
            or item["byte_length"] <= 0
        ):
            raise QualificationArtifactError(f"{label} tensor digest is invalid")
        keys.add(key)
        byte_sum += item["byte_length"]
        digests.append(item)
    if tensor_count != len(digests) or total_bytes != byte_sum:
        raise QualificationArtifactError(f"{label} tensor count/bytes mismatch")
    return digests


def _validate_authority_payload(phase: str, payload: Mapping[str, Any]) -> None:
    fields = {
        "logical_version",
        "runtime_adapter_name",
        "active_adapter",
        "canonical_config_sha256",
        "aggregate_tensor_sha256",
        "per_tensor_digests",
        "tensor_count",
        "total_bytes",
        "base_revision",
        "tokenizer_revision",
        "immutable_manifest_sha256",
        "trainer_memory_reload_same_path",
        "checkpoint",
    }
    _require_payload_fields(payload, fields, phase)
    expected_version = phase.removeprefix("authority_")
    if not (
        payload["logical_version"] == expected_version
        and payload["runtime_adapter_name"] == "student_active"
        and payload["active_adapter"] == "student_active"
        and _is_hex(payload["canonical_config_sha256"], 64)
        and _is_hex(payload["aggregate_tensor_sha256"], 64)
        and _is_hex(payload["immutable_manifest_sha256"], 64)
        and _is_hex(payload["base_revision"], 40)
        and _is_hex(payload["tokenizer_revision"], 40)
    ):
        raise QualificationArtifactError(f"{phase} identity mismatch")
    _validate_per_tensor_digests(
        payload["per_tensor_digests"],
        tensor_count=payload["tensor_count"],
        total_bytes=payload["total_bytes"],
        label=phase,
    )
    same_path = payload["trainer_memory_reload_same_path"]
    _validate_same_path(same_path, phase=f"{phase} trainer memory/reload")
    checkpoint = _exact_fields(
        payload["checkpoint"],
        {"directory", "transport_manifest_path", "transport_manifest_sha256"},
        label=f"{phase} checkpoint",
    )
    version = phase.removeprefix("authority_")
    if not (
        checkpoint["directory"] == f"checkpoints/{version}"
        and checkpoint["transport_manifest_path"]
        == f"checkpoints/{version}/adapter_transport_manifest.json"
        and _is_hex(checkpoint["transport_manifest_sha256"], 64)
    ):
        raise QualificationArtifactError(f"{phase} checkpoint descriptor mismatch")
    expected_immutable_sha256 = canonical_json_sha256(
        {
            "trainer_tensor_sha256": payload["aggregate_tensor_sha256"],
            "saved_tensor_sha256": payload["aggregate_tensor_sha256"],
            "reload_tensor_sha256": payload["aggregate_tensor_sha256"],
            "transport_manifest_sha256": checkpoint[
                "transport_manifest_sha256"
            ],
            "same_path": same_path,
        }
    )
    if payload["immutable_manifest_sha256"] != expected_immutable_sha256:
        raise QualificationArtifactError(f"{phase} immutable manifest SHA mismatch")


def _validate_checkpoint_binding(
    output: Path, payload: Mapping[str, Any], *, phase: str
) -> None:
    checkpoint = payload["checkpoint"]
    directory_relative = _safe_relative(checkpoint["directory"])
    manifest_relative = _safe_relative(checkpoint["transport_manifest_path"])
    version = phase.removeprefix("authority_")
    if directory_relative != f"checkpoints/{version}" or not manifest_relative.startswith(
        directory_relative + "/"
    ):
        raise QualificationArtifactError(f"{phase} checkpoint path mismatch")
    directory = output / directory_relative
    manifest_path = output / manifest_relative
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != checkpoint["transport_manifest_sha256"]
    ):
        raise QualificationArtifactError(f"{phase} checkpoint manifest binding mismatch")
    manifest = _read_json(manifest_path)
    _exact_fields(
        manifest,
        {
            "schema_version",
            "logical_version",
            "canonical_config_sha256",
            "aggregate_tensor_sha256",
            "files",
        },
        label=f"{phase} checkpoint manifest",
    )
    if not (
        manifest["schema_version"] == 1
        and manifest["logical_version"] == version
        and manifest["canonical_config_sha256"]
        == payload["canonical_config_sha256"]
        and manifest["aggregate_tensor_sha256"]
        == payload["aggregate_tensor_sha256"]
        and isinstance(manifest["files"], list)
        and manifest["files"]
    ):
        raise QualificationArtifactError(f"{phase} checkpoint manifest identity mismatch")
    indexed: set[str] = set()
    roles: set[str] = set()
    for item in manifest["files"]:
        item = _exact_fields(
            item,
            {"path", "role", "sha256", "size_bytes"},
            label=f"{phase} checkpoint file",
        )
        relative = _safe_relative(item["path"])
        if relative in indexed or item["role"] in roles:
            raise QualificationArtifactError(f"{phase} checkpoint file collision")
        if item["role"] not in {"adapter_config", "adapter_weights"}:
            raise QualificationArtifactError(f"{phase} checkpoint role is unsupported")
        path = directory / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or not _is_hex(item["sha256"], 64)
            or sha256_file(path) != item["sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise QualificationArtifactError(f"{phase} checkpoint file binding mismatch")
        indexed.add(relative)
        roles.add(item["role"])
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if roles != {"adapter_config", "adapter_weights"} or actual != indexed:
        raise QualificationArtifactError(f"{phase} checkpoint transport set mismatch")


def _validate_same_path(value: Any, *, phase: str) -> None:
    fields = {
        "mae",
        "p50",
        "p95",
        "p99",
        "max",
        "finite_rate",
        "worst_sample_id",
        "worst_token_position",
        "worst_token_id",
    }
    value = _exact_fields(value, fields, label=f"{phase} same-path")
    numeric = [value[key] for key in ("mae", "p50", "p95", "p99", "max")]
    if not (
        all(isinstance(item, (int, float)) and math.isfinite(float(item)) and item >= 0 for item in numeric)
        and value["p50"] <= value["p95"] <= value["p99"] <= value["max"]
        and value["mae"] <= value["max"]
        and value["max"] <= 0.0001
        and value["finite_rate"] == 1.0
        and isinstance(value["worst_sample_id"], str)
        and bool(value["worst_sample_id"])
        and isinstance(value["worst_token_position"], int)
        and not isinstance(value["worst_token_position"], bool)
        and value["worst_token_position"] >= 0
        and isinstance(value["worst_token_id"], int)
        and not isinstance(value["worst_token_id"], bool)
        and value["worst_token_id"] >= 0
    ):
        raise QualificationArtifactError(f"{phase} same-path gate failed")


def _validate_refresh_payload(phase: str, payload: Mapping[str, Any]) -> None:
    fields = {
        "logical_version",
        "canonical_config_sha256",
        "trainer_tensor_sha256",
        "runtime_tensor_sha256",
        "fresh_tensor_sha256",
        "runtime_per_tensor_digests",
        "fresh_per_tensor_digests",
        "tensor_count",
        "total_bytes",
        "registry_before",
        "registry_after",
        "active_adapter",
        "adapter_enabled",
        "merged",
        "same_path",
        "normal_request",
        "stale_request",
        "refresh_latency_seconds",
    }
    if phase == "refresh_v2":
        fields.add("previous_tensor_sha256")
    _require_payload_fields(payload, fields, phase)
    version = phase.removeprefix("refresh_")
    identities = {
        payload["trainer_tensor_sha256"],
        payload["runtime_tensor_sha256"],
        payload["fresh_tensor_sha256"],
    }
    runtime_digests = _validate_per_tensor_digests(
        payload["runtime_per_tensor_digests"],
        tensor_count=payload["tensor_count"],
        total_bytes=payload["total_bytes"],
        label=f"{phase}.runtime",
    )
    fresh_digests = _validate_per_tensor_digests(
        payload["fresh_per_tensor_digests"],
        tensor_count=payload["tensor_count"],
        total_bytes=payload["total_bytes"],
        label=f"{phase}.fresh",
    )
    if runtime_digests != fresh_digests:
        raise QualificationArtifactError(f"{phase} per-tensor identity mismatch")
    normal = _exact_fields(
        payload["normal_request"],
        {"accepted", "scoring_executed", "generation_executed", "finite_rate"},
        label=f"{phase} normal request",
    )
    stale = _exact_fields(
        payload["stale_request"],
        {
            "logical_version",
            "rejected",
            "error_code",
            "rejection_phase",
            "scoring_executed",
            "generation_executed",
        },
        label=f"{phase} stale request",
    )
    expected_stale = "v0" if version == "v1" else "v1"
    if not (
        payload["logical_version"] == version
        and _is_hex(payload["canonical_config_sha256"], 64)
        and len(identities) == 1
        and all(_is_hex(item, 64) for item in identities)
        and payload["registry_before"] == ["student_active"]
        and payload["registry_after"] == ["student_active"]
        and payload["active_adapter"] == "student_active"
        and payload["adapter_enabled"] is True
        and payload["merged"] is False
        and isinstance(payload["refresh_latency_seconds"], (int, float))
        and math.isfinite(float(payload["refresh_latency_seconds"]))
        and payload["refresh_latency_seconds"] >= 0
        and normal
        == {
            "accepted": True,
            "scoring_executed": True,
            "generation_executed": False,
            "finite_rate": 1.0,
        }
        and stale
        == {
            "logical_version": expected_stale,
            "rejected": True,
            "error_code": "STALE_SAMPLER_IDENTITY",
            "rejection_phase": "identity_guard_before_forward",
            "scoring_executed": False,
            "generation_executed": False,
        }
    ):
        raise QualificationArtifactError(f"{phase} evidence gate failed")
    _validate_same_path(payload["same_path"], phase=phase)


def _validate_trajectory_step1(payload: Mapping[str, Any]) -> None:
    fields = {
        "generated_by_policy_version",
        "logical_version",
        "run_token",
        "sampler_tensor_sha256",
        "trainer_authority_sha256",
        "p_old_actor_tensor_sha256",
        "p_old_policy_version",
        "refresh_artifact_sha256",
        "prompt_manifest_sha256",
        "seed",
        "q_provenance",
        "stale_v0_pre_rollout",
    }
    _require_payload_fields(payload, fields, "trajectory_step1_manifest")
    q = _exact_fields(
        payload["q_provenance"],
        {
            "backend",
            "logical_version",
            "run_token",
            "runtime_tensor_sha256",
            "finite_rate",
        },
        label="trajectory_step1_manifest q provenance",
    )
    stale = _exact_fields(
        payload["stale_v0_pre_rollout"],
        {
            "rejected",
            "error_code",
            "rejection_phase",
            "scoring_executed",
            "generation_executed",
        },
        label="trajectory_step1_manifest stale v0",
    )
    identities = {
        payload["sampler_tensor_sha256"],
        payload["trainer_authority_sha256"],
        payload["p_old_actor_tensor_sha256"],
        q["runtime_tensor_sha256"],
    }
    if not (
        payload["generated_by_policy_version"]
        == payload["logical_version"]
        == payload["p_old_policy_version"]
        == q["logical_version"]
        == "v1"
        and isinstance(payload["run_token"], str)
        and bool(payload["run_token"])
        and payload["run_token"] == q["run_token"]
        and len(identities) == 1
        and all(_is_hex(item, 64) for item in identities)
        and _is_hex(payload["refresh_artifact_sha256"], 64)
        and _is_hex(payload["prompt_manifest_sha256"], 64)
        and payload["seed"] == 42
        and q["backend"] == "transformers_generate_full_support"
        and q["finite_rate"] == 1.0
        and stale
        == {
            "rejected": True,
            "error_code": "STALE_SAMPLER_IDENTITY",
            "rejection_phase": "identity_guard_before_forward",
            "scoring_executed": False,
            "generation_executed": False,
        }
    ):
        raise QualificationArtifactError("trajectory_step1_manifest on-policy gate failed")


def _validate_base_null(payload: Mapping[str, Any]) -> None:
    fields = {
        "independent_route",
        "fresh_optimizer",
        "teacher_is_base",
        "old_actor_base_detached",
        "current_actor_zero_lora",
        "current_pre_base_max_gap",
        "advantage_max_abs",
        "objective",
        "loss",
        "gradient_norm",
        "parameter_delta",
        "nonzero_update_tensor_count",
        "adapter_sha256_before",
        "adapter_sha256_after",
        "teacher_gradient_tensor_count",
        "base_gradient_tensor_count",
        "finite_rate",
    }
    _require_payload_fields(payload, fields, "base_null")
    if not (
        payload["independent_route"] is True
        and payload["fresh_optimizer"] is True
        and payload["teacher_is_base"] is True
        and payload["old_actor_base_detached"] is True
        and payload["current_actor_zero_lora"] is True
        and isinstance(payload["current_pre_base_max_gap"], (int, float))
        and 0 <= payload["current_pre_base_max_gap"] <= 0.0001
        and isinstance(payload["advantage_max_abs"], (int, float))
        and 0 <= payload["advantage_max_abs"] <= 1e-8
        and payload["objective"] == 0
        and payload["loss"] == 0
        and payload["gradient_norm"] == 0
        and payload["parameter_delta"] == 0
        and payload["nonzero_update_tensor_count"] == 0
        and _is_hex(payload["adapter_sha256_before"], 64)
        and payload["adapter_sha256_before"] == payload["adapter_sha256_after"]
        and payload["teacher_gradient_tensor_count"] == 0
        and payload["base_gradient_tensor_count"] == 0
        and payload["finite_rate"] == 1.0
    ):
        raise QualificationArtifactError("base_null gate failed")


def _length_candidate_passes(value: Any, *, label: str) -> bool:
    fields = {
        "count",
        "finite_rate",
        "invalid_empty_count",
        "thinking_tag_count",
        "truncation_count",
        "truncation_rate",
        "per_source",
    }
    value = _exact_fields(value, fields, label=label)
    sources = _exact_fields(
        value["per_source"],
        {"medical_opd_o1", "medical_opd_cmb"},
        label=f"{label} per-source",
    )
    source_passes = True
    truncation_sum = 0
    for source, metrics in sources.items():
        metrics = _exact_fields(
            metrics,
            {"count", "truncation_count", "truncation_rate"},
            label=f"{label}.{source}",
        )
        if not (
            metrics["count"] == 8
            and isinstance(metrics["truncation_count"], int)
            and not isinstance(metrics["truncation_count"], bool)
            and 0 <= metrics["truncation_count"] <= 8
            and math.isclose(
                metrics["truncation_rate"],
                metrics["truncation_count"] / 8,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise QualificationArtifactError(f"{label} per-source statistics invalid")
        truncation_sum += metrics["truncation_count"]
        source_passes = source_passes and metrics["truncation_count"] <= 1
    if not (
        value["count"] == 16
        and value["finite_rate"] == 1.0
        and value["invalid_empty_count"] == 0
        and value["thinking_tag_count"] == 0
        and value["truncation_count"] == truncation_sum
        and math.isclose(
            value["truncation_rate"], truncation_sum / 16, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise QualificationArtifactError(f"{label} aggregate statistics invalid")
    return bool(value["truncation_rate"] <= 0.20 and source_passes)


def _validate_length_smoke(payload: Mapping[str, Any]) -> int:
    fields = {
        "actual_lengths",
        "conditional_512_executed",
        "derived_256",
        "actual_384",
        "actual_512_executed",
        "telemetry",
        "policy_identity",
    }
    _require_payload_fields(payload, fields, "length_smoke")
    actual = payload["actual_lengths"]
    conditional = payload["conditional_512_executed"]
    if actual not in ([384], [384, 512]) or conditional is not (512 in actual):
        raise QualificationArtifactError("length smoke actual generation set invalid")
    if payload["actual_512_executed"] is not conditional:
        raise QualificationArtifactError("length smoke conditional 512 evidence mismatch")
    pass_256 = _length_candidate_passes(
        payload["derived_256"], label="length smoke derived 256"
    )
    pass_384 = _length_candidate_passes(
        payload["actual_384"], label="length smoke actual 384"
    )
    if pass_384 and conditional:
        raise QualificationArtifactError("length smoke ran 512 despite passing 384")
    if not pass_384 and not conditional:
        raise QualificationArtifactError("length smoke omitted required conditional 512")
    pass_512 = False
    if conditional:
        _require_payload_fields(payload, {"actual_512"}, "length_smoke")
        pass_512 = _length_candidate_passes(
            payload["actual_512"], label="length smoke actual 512"
        )
    if pass_256:
        selected = 256
    elif pass_384:
        selected = 384
    elif pass_512:
        selected = 512
    else:
        raise QualificationArtifactError("length smoke has no passing frozen candidate")
    _validate_length_telemetry(
        payload["telemetry"],
        payload=payload,
        selected_response_length=selected,
    )
    _validate_length_policy_identity(payload["policy_identity"])
    return selected


def _validate_length_telemetry(
    telemetry: Any,
    *,
    payload: Mapping[str, Any],
    selected_response_length: int,
) -> None:
    required = {
        "selection_rule",
        "prompt_count",
        "source_counts",
        "actual_384",
        "derived_256",
        "prompt_identity_sha256",
        "selected_response_length",
    }
    if payload["conditional_512_executed"]:
        required.add("actual_512")
    telemetry = _exact_fields(telemetry, required, label="length telemetry")
    if not (
        telemetry["selection_rule"]
        == "checked_in_p4_6_prompt_selection_manifest_v1"
        and telemetry["prompt_count"] == 16
        and telemetry["source_counts"]
        == {"medical_opd_o1": 8, "medical_opd_cmb": 8}
        and _is_hex(telemetry["prompt_identity_sha256"], 64)
        and telemetry["selected_response_length"] == selected_response_length
    ):
        raise QualificationArtifactError("length telemetry identity is invalid")
    candidates = (
        ("derived_256", 256, "derived_prefix_from_actual_384"),
        ("actual_384", 384, "actual_generation"),
    )
    if payload["conditional_512_executed"]:
        candidates += (("actual_512", 512, "actual_generation"),)
    for key, max_new_tokens, measurement in candidates:
        value = _exact_fields(
            telemetry[key],
            {
                "max_new_tokens",
                "measurement",
                "eos_count",
                "eos_rate",
                "truncation_count",
                "truncation_rate",
                "length",
                "finish_reason_counts",
                "invalid_empty_count",
                "thinking_tag_count",
                "finite_rate",
                "tokens_per_second",
                "wall_time_seconds",
                "gpu_peak_memory_bytes",
                "per_source",
            },
            label=f"length telemetry {key}",
        )
        coarse = payload[key]
        length = _exact_fields(
            value["length"],
            {"min", "p50", "p90", "p95", "max", "mean"},
            label=f"length telemetry {key}.length",
        )
        finish = value["finish_reason_counts"]
        length_values = [
            length["min"],
            length["p50"],
            length["p90"],
            length["p95"],
            length["max"],
        ]
        if not (
            value["max_new_tokens"] == max_new_tokens
            and value["measurement"] == measurement
            and isinstance(value["eos_count"], int)
            and not isinstance(value["eos_count"], bool)
            and 0 <= value["eos_count"] <= 16
            and math.isclose(
                value["eos_rate"], value["eos_count"] / 16, rel_tol=0, abs_tol=1e-12
            )
            and value["truncation_count"] == coarse["truncation_count"]
            and value["truncation_rate"] == coarse["truncation_rate"]
            and value["invalid_empty_count"] == coarse["invalid_empty_count"]
            and value["thinking_tag_count"] == coarse["thinking_tag_count"]
            and value["finite_rate"] == coarse["finite_rate"] == 1.0
            and value["per_source"] == coarse["per_source"]
            and isinstance(finish, Mapping)
            and set(finish) == {"eos", "length"}
            and all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in finish.values()
            )
            and sum(finish.values()) == 16
            and finish.get("eos", 0) == value["eos_count"]
            and finish.get("length", 0) == value["truncation_count"]
            and all(
                isinstance(item, (int, float)) and math.isfinite(float(item))
                for item in (*length_values, length["mean"])
            )
            and 0 <= length["min"] <= length["p50"] <= length["p90"]
            <= length["p95"] <= length["max"] <= max_new_tokens
            and length["min"] <= length["mean"] <= length["max"]
            and isinstance(value["tokens_per_second"], (int, float))
            and math.isfinite(float(value["tokens_per_second"]))
            and value["tokens_per_second"] > 0
            and isinstance(value["wall_time_seconds"], (int, float))
            and math.isfinite(float(value["wall_time_seconds"]))
            and value["wall_time_seconds"] > 0
            and isinstance(value["gpu_peak_memory_bytes"], int)
            and not isinstance(value["gpu_peak_memory_bytes"], bool)
            and value["gpu_peak_memory_bytes"] >= 0
        ):
            raise QualificationArtifactError(
                f"length telemetry candidate is invalid: {key}"
            )


def _validate_length_prompt_identity(
    output: Path, payload: Mapping[str, Any]
) -> None:
    manifest = _read_json(output / "prompt_manifest.json")
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise QualificationArtifactError("static prompt manifest groups are invalid")
    length_groups = [
        group
        for group in groups
        if isinstance(group, Mapping) and group.get("group_id") == "length"
    ]
    if len(length_groups) != 1:
        raise QualificationArtifactError("static prompt manifest length group is invalid")
    rows = length_groups[0].get("ordered_samples")
    if not isinstance(rows, list) or len(rows) != 16:
        raise QualificationArtifactError("static prompt manifest length rows are invalid")
    projection: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise QualificationArtifactError("static prompt manifest length row is invalid")
        sample_id = row.get("sample_id")
        source_role = row.get("source_role")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or source_role not in {"medical_opd_o1", "medical_opd_cmb"}
        ):
            raise QualificationArtifactError("static prompt manifest length row is invalid")
        projection.append({"sample_id": sample_id, "source_role": source_role})
    if len({row["sample_id"] for row in projection}) != 16 or {
        source: sum(row["source_role"] == source for row in projection)
        for source in ("medical_opd_o1", "medical_opd_cmb")
    } != {"medical_opd_o1": 8, "medical_opd_cmb": 8}:
        raise QualificationArtifactError("static prompt manifest length group counts mismatch")
    if payload["telemetry"]["prompt_identity_sha256"] != canonical_json_sha256(
        projection
    ):
        raise QualificationArtifactError("length prompt identity SHA mismatch")


def _validate_length_policy_identity(identity: Any) -> None:
    identity = _exact_fields(
        identity,
        {
            "logical_version",
            "tensor_sha256",
            "checkpoint_path",
            "authority_v2_artifact_sha256",
            "active_slot",
            "registry_count",
        },
        label="length policy identity",
    )
    checkpoint = Path(str(identity["checkpoint_path"]))
    if not (
        identity["logical_version"] == "v2"
        and _is_hex(identity["tensor_sha256"], 64)
        and checkpoint.is_absolute()
        and _is_hex(identity["authority_v2_artifact_sha256"], 64)
        and identity["active_slot"] == "student_active"
        and identity["registry_count"] == 1
    ):
        raise QualificationArtifactError("length policy identity is invalid")


def _validate_v0_authority(value: Any) -> Mapping[str, Any]:
    fields = {
        "logical_version",
        "runtime_adapter_name",
        "active_adapter",
        "canonical_config_sha256",
        "aggregate_tensor_sha256",
        "per_tensor_digests",
        "tensor_count",
        "total_bytes",
        "base_revision",
        "tokenizer_revision",
        "immutable_manifest_sha256",
    }
    authority = _exact_fields(value, fields, label="v0 authority")
    if not (
        authority["logical_version"] == "v0"
        and authority["runtime_adapter_name"] == "student_active"
        and authority["active_adapter"] == "student_active"
        and _is_hex(authority["canonical_config_sha256"], 64)
        and _is_hex(authority["aggregate_tensor_sha256"], 64)
        and _is_hex(authority["base_revision"], 40)
        and _is_hex(authority["tokenizer_revision"], 40)
        and _is_hex(authority["immutable_manifest_sha256"], 64)
    ):
        raise QualificationArtifactError("v0 authority identity mismatch")
    _validate_per_tensor_digests(
        authority["per_tensor_digests"],
        tensor_count=authority["tensor_count"],
        total_bytes=authority["total_bytes"],
        label="v0 authority",
    )
    return authority


def _validate_phase_payload(phase: str, payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise QualificationArtifactError(f"{phase} payload must be an object")
    _reject_unavailable(payload, path=f"{phase}.payload")
    if payload.get("status") != "pass":
        raise QualificationArtifactError(f"{phase} status is not pass")
    if phase == "probe_manifest":
        _validate_probe_payload(payload)
    elif phase == "v0_guard":
        _require_payload_fields(
            payload,
            {
                "normal_v0",
                "wrong_authority",
                "same_instance_repeat",
                "v0_authority",
            },
            phase,
        )
        normal = payload["normal_v0"]
        wrong = payload["wrong_authority"]
        if not isinstance(normal, Mapping) or not isinstance(wrong, Mapping):
            raise QualificationArtifactError("v0 guard evidence must be objects")
        authority = _validate_v0_authority(payload["v0_authority"])
        _validate_same_path(
            payload["same_instance_repeat"], phase="v0 guard same-instance repeat"
        )
        try:
            validate_v0_guard_evidence(normal, wrong)
        except QualificationContractError as error:
            raise QualificationArtifactError(f"v0 guard evidence invalid: {error}") from error
        if not (
            normal["trainer_authoritative_tensor_sha256"]
            == normal["sampler_runtime_tensor_sha256"]
            == normal["authority_after_request_sha256"]
            == authority["aggregate_tensor_sha256"]
            and normal["canonical_config_sha256"]
            == authority["canonical_config_sha256"]
            and normal["base_revision"] == authority["base_revision"]
            and normal["tokenizer_revision"] == authority["tokenizer_revision"]
        ):
            raise QualificationArtifactError(
                "v0 guard independent authority evidence gate failed"
            )
    elif phase.startswith("reconstruction_step"):
        _validate_reconstruction(payload, phase=phase)
    elif phase.startswith("authority_v"):
        _validate_authority_payload(phase, payload)
    elif phase == "refresh_v1":
        _validate_refresh_payload(phase, payload)
    elif phase == "trajectory_step1_manifest":
        _validate_trajectory_step1(payload)
    elif phase == "refresh_v2":
        _validate_refresh_payload(phase, payload)
        if payload["previous_tensor_sha256"] in {
            payload["trainer_tensor_sha256"],
            payload["runtime_tensor_sha256"],
            payload["fresh_tensor_sha256"],
        }:
            raise QualificationArtifactError("refresh_v2 did not change from v1")
    elif phase == "base_null":
        _validate_base_null(payload)
    elif phase == "length_smoke":
        _validate_length_smoke(payload)
    elif phase == "length_decision":
        _require_payload_fields(
            payload,
            {
                "selected_response_length",
                "length_smoke_sha256",
                "evaluated_candidates",
                "decision_rule",
            },
            phase,
        )
        if (
            payload["selected_response_length"] not in {256, 384, 512}
            or payload["evaluated_candidates"] not in ([256, 384], [256, 384, 512])
            or payload["decision_rule"]
            != "shortest_passing_overall_and_per_source_truncation_v1"
        ):
            raise QualificationArtifactError("response length is not frozen")
    elif phase == "cleanup":
        if not (
            payload.get("runtime_exit_code") == 0
            and payload.get("gpu_memory_used_mib") == [0, 0]
            and payload.get("compute_pids") == []
            and payload.get("residual_workers") == []
        ):
            raise QualificationArtifactError("cleanup gate failed")
    elif phase == "terminal_summary":
        forbidden_claims = (
            "ready",
            "production_sampler_refresh_ready",
            "OPD_scoring_backend_ready",
            "B2_authorized",
        )
        if any(payload.get(field) is True for field in forbidden_claims):
            raise QualificationArtifactError(
                "terminal_summary cannot claim artifact-derived readiness"
            )
        if payload.get("B2_started") is not False:
            raise QualificationArtifactError("qualification must not start B2")


def _probe_binding(
    output: Path, bindings: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    path = output / "probe_manifest.json"
    if not path.is_file() or path.is_symlink():
        return None, None
    value = _read_json(path)
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        return None, None
    try:
        content_sha = _validate_probe_payload(
            payload, spec=_probe_spec_from_static(output, bindings)
        )
    except QualificationArtifactError:
        return None, None
    return content_sha, sha256_file(path)


def commit_phase(
    output: str | Path,
    *,
    bindings: Mapping[str, Any],
    mode: str,
    phase_id: str,
    ordinal: int,
    payload: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one immutable phase, then durably append exactly one metric row."""

    _validate_bindings(bindings)
    phases = _phases(mode)
    if ordinal < 0 or ordinal >= len(phases) or phases[ordinal] != phase_id:
        raise QualificationArtifactError("phase order/ordinal mismatch")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise QualificationArtifactError("output directory cannot be a symlink")
    if (directory / "failure.json").exists():
        raise QualificationArtifactError("cannot commit after failure")
    path = directory / f"{phase_id}.json"
    if path.exists() or path.is_symlink():
        raise QualificationArtifactError(f"artifact already exists: {path.name}")
    for index, expected in enumerate(phases):
        exists = (directory / f"{expected}.json").is_file()
        if index < ordinal and not exists:
            raise QualificationArtifactError("phase order has a missing predecessor")
        if index > ordinal and exists:
            raise QualificationArtifactError("phase order has a committed successor")
    _validate_phase_payload(phase_id, payload)
    if phase_id == "probe_manifest":
        _validate_probe_payload(
            payload, spec=_probe_spec_from_static(directory, bindings)
        )
    elif phase_id == "v0_guard" and payload["normal_v0"]["run_id"] != bindings[
        "run_id"
    ]:
        raise QualificationArtifactError("v0 guard/envelope run binding mismatch")
    if phase_id in {"authority_v1", "authority_v2"}:
        _validate_checkpoint_binding(directory, payload, phase=phase_id)
    elif phase_id == "length_smoke":
        _validate_length_prompt_identity(directory, payload)
    _reject_unavailable(metric, path=f"metrics.{phase_id}")

    previous_path: str | None = None
    previous_sha: str | None = None
    if ordinal:
        previous_path = f"{phases[ordinal - 1]}.json"
        previous_sha = sha256_file(directory / previous_path)
    probe_content, probe_file = _probe_binding(directory, bindings)
    if phase_id == "probe_manifest":
        probe_content = _validate_probe_payload(
            payload, spec=_probe_spec_from_static(directory, bindings)
        )
        probe_file = None

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        **dict(bindings),
        "phase_id": phase_id,
        "ordinal": ordinal,
        "previous_phase_path": previous_path,
        "previous_phase_sha256": previous_sha,
        "probe_manifest_content_sha256": probe_content,
        "probe_manifest_file_sha256": probe_file,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": dict(payload),
    }
    phase_sha = _atomic_json(path, artifact)
    metric_row = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "phase_id": phase_id,
        "ordinal": ordinal,
        "config_sha256": bindings["config_sha256"],
        "run_card_sha256": bindings["run_card_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "phase_artifact_sha256": phase_sha,
        "metric": dict(metric),
    }
    _atomic_append_jsonl(directory / "metrics.jsonl", metric_row)
    return artifact


def _last_contiguous(output: Path, phases: Sequence[str]) -> tuple[str | None, str | None]:
    last: str | None = None
    for phase in phases:
        if not (output / f"{phase}.json").is_file():
            break
        last = phase
    return (
        (last, sha256_file(output / f"{last}.json"))
        if last is not None
        else (None, None)
    )


def record_failure(
    output: str | Path,
    *,
    bindings: Mapping[str, Any],
    mode: str,
    reason: str,
) -> dict[str, Any]:
    """Write one terminal failure bound to the last contiguous phase and metrics."""

    _validate_bindings(bindings)
    directory = Path(output)
    path = directory / "failure.json"
    if path.exists() or path.is_symlink():
        raise QualificationArtifactError("failure already exists and is immutable")
    if not reason:
        raise QualificationArtifactError("failure reason must be non-empty")
    last, last_sha = _last_contiguous(directory, _phases(mode))
    metrics = directory / "metrics.jsonl"
    failure = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "mode": mode,
        "status": "fail",
        "reason": reason,
        "last_committed_phase": last,
        "last_committed_phase_sha256": last_sha,
        "metrics_sha256": sha256_file(metrics) if metrics.is_file() else None,
        "B2_authorized": False,
        "B2_started": False,
    }
    _atomic_json(path, failure)
    return failure


def record_failure_cleanup(
    output: str | Path,
    *,
    bindings: Mapping[str, Any],
    runtime_exit_code: int,
    observation: Mapping[str, Any] | None,
    observation_error: str | None = None,
) -> dict[str, Any]:
    """Persist an honest cleanup observation when phase ordering has failed."""

    _validate_bindings(bindings)
    directory = Path(output)
    path = directory / FAILURE_CLEANUP_FILE
    if path.exists() or path.is_symlink():
        stored = _read_json(path)
        return stored
    observed = dict(observation or {})
    memory = observed.get("gpu_memory_used_mib")
    pids = observed.get("compute_pids")
    workers = observed.get("residual_workers")
    complete = bool(
        observation_error is None
        and memory == [0, 0]
        and pids == []
        and workers == []
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "runtime_exit_code": int(runtime_exit_code),
        "gpu_memory_used_mib": memory,
        "compute_pids": pids,
        "residual_workers": workers,
        "observation_error": observation_error,
        "cleanup_complete": complete,
        "B2_authorized": False,
        "B2_started": False,
    }
    _atomic_json(path, artifact)
    return artifact


def _load_bindings_from_phase(output: Path, phases: Sequence[str]) -> dict[str, Any]:
    for phase in phases:
        path = output / f"{phase}.json"
        if path.is_file() and not path.is_symlink():
            value = _read_json(path)
            bindings = {key: value.get(key) for key in _BINDING_FIELDS}
            _validate_bindings(bindings)
            return bindings
    raise QualificationArtifactError("no committed phase provides qualification bindings")


def _safe_relative(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise QualificationArtifactError("index path is empty")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != path:
        raise QualificationArtifactError("index path traversal is forbidden")
    return path


def _allowed_paths(phases: Sequence[str]) -> set[str]:
    return {
        *(f"{phase}.json" for phase in phases),
        *STATIC_ARTIFACT_FILES,
        "summary.json",
        "metrics.jsonl",
        "failure.json",
        FAILURE_CLEANUP_FILE,
        MICRO_READINESS_FILE,
        EVIDENCE_INDEX_FILE,
        EVIDENCE_READINESS_FILE,
        FINAL_INDEX_FILE,
        FINAL_READINESS_FILE,
        ROOT_AUTHORIZATION_FILE,
        "checkpoints",
    }


def _filesystem_errors(output: Path, phases: Sequence[str]) -> list[str]:
    errors: list[str] = []
    allowed = _allowed_paths(phases)
    if not output.is_dir() or output.is_symlink():
        return ["output_directory_invalid"]
    for path in sorted(output.iterdir()):
        if path.is_symlink():
            errors.append(f"symlink:{path.name}")
        elif path.is_dir():
            if path.name != "checkpoints":
                errors.append(f"non_file:{path.name}")
        elif path.name not in allowed:
            errors.append(f"extra:{path.name}")
    checkpoint_root = output / "checkpoints"
    if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
        for path in sorted(checkpoint_root.rglob("*")):
            relative = path.relative_to(output).as_posix()
            if path.is_symlink():
                errors.append(f"symlink:{relative}")
            elif not path.is_file() and not path.is_dir():
                errors.append(f"non_file:{relative}")
    return errors


def _inspect_graph(
    output: Path,
    *,
    mode: str,
    verify_index: bool,
    index_name: str = FINAL_INDEX_FILE,
    index_exclusions: frozenset[str] = frozenset(
        {FINAL_INDEX_FILE, FINAL_READINESS_FILE}
    ),
) -> tuple[list[str], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    phases = _phases(mode)
    errors = _filesystem_errors(output, phases)
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] | None = None
    probe_content: str | None = None
    probe_file: str | None = None
    previous_path: str | None = None
    previous_sha: str | None = None

    for ordinal, phase in enumerate(phases):
        path = output / f"{phase}.json"
        if not path.is_file() or path.is_symlink():
            errors.append(phase)
            continue
        try:
            value = _read_json(path)
            documents[phase] = value
            if set(value) != _ENVELOPE_FIELDS:
                raise QualificationArtifactError(f"{phase} envelope fields mismatch")
            if value["schema_version"] != SCHEMA_VERSION:
                raise QualificationArtifactError(f"{phase} schema mismatch")
            if value["artifact_protocol_version"] != ARTIFACT_PROTOCOL_VERSION:
                raise QualificationArtifactError(f"{phase} protocol mismatch")
            current_bindings = {key: value[key] for key in _BINDING_FIELDS}
            _validate_bindings(current_bindings)
            if bindings is None:
                bindings = current_bindings
            elif current_bindings != bindings:
                raise QualificationArtifactError(f"{phase} binding mismatch")
            if value["phase_id"] != phase or value["ordinal"] != ordinal:
                raise QualificationArtifactError(f"{phase} phase/ordinal mismatch")
            if value["previous_phase_path"] != previous_path:
                raise QualificationArtifactError(f"{phase} previous path mismatch")
            if value["previous_phase_sha256"] != previous_sha:
                raise QualificationArtifactError(f"{phase} previous SHA mismatch")
            if value["payload_sha256"] != canonical_json_sha256(value["payload"]):
                raise QualificationArtifactError(f"{phase} payload SHA mismatch")
            _validate_phase_payload(phase, value["payload"])
            if phase in {"authority_v1", "authority_v2"}:
                _validate_checkpoint_binding(output, value["payload"], phase=phase)
            elif phase == "length_smoke":
                _validate_length_prompt_identity(output, value["payload"])
            if phase == "probe_manifest":
                probe_content = _validate_probe_payload(
                    value["payload"],
                    spec=_probe_spec_from_static(output, current_bindings),
                )
                probe_file = sha256_file(path)
                if value["probe_manifest_content_sha256"] != probe_content:
                    raise QualificationArtifactError("probe content binding mismatch")
                if value["probe_manifest_file_sha256"] is not None:
                    raise QualificationArtifactError("probe cannot self-bind its file SHA")
            elif ordinal < phases.index("probe_manifest"):
                if (
                    value["probe_manifest_content_sha256"] is not None
                    or value["probe_manifest_file_sha256"] is not None
                ):
                    raise QualificationArtifactError("pre-probe phase has probe binding")
            elif ordinal > phases.index("probe_manifest"):
                if value["probe_manifest_content_sha256"] != probe_content:
                    raise QualificationArtifactError(f"{phase} probe content binding mismatch")
                if value["probe_manifest_file_sha256"] != probe_file:
                    raise QualificationArtifactError(f"{phase} probe file binding mismatch")
            previous_path = path.name
            previous_sha = sha256_file(path)
        except (QualificationArtifactError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))

    if bindings is not None:
        errors.extend(
            validate_qualification_static_artifacts(
                output,
                bindings=bindings,
                mode=mode,
                require_summary=(output / "terminal_summary.json").is_file(),
            )
        )

    try:
        if bindings is not None and "probe_manifest" in documents:
            probe_payload = documents["probe_manifest"]["payload"]
            if probe_payload["prompt_manifest_sha256"] != bindings[
                "prompt_manifest_sha256"
            ]:
                raise QualificationArtifactError("probe/prompt manifest binding mismatch")
        if "v0_guard" in documents and documents["v0_guard"]["payload"][
            "normal_v0"
        ]["run_id"] != bindings["run_id"]:
            raise QualificationArtifactError("v0 guard/envelope run binding mismatch")
        for phase in ("reconstruction_step0", "reconstruction_step1"):
            if phase in documents and documents[phase]["payload"]["telemetry"][
                "run_id"
            ] != bindings["run_id"]:
                raise QualificationArtifactError(f"{phase} run binding mismatch")
        if {"authority_v1", "refresh_v1"}.issubset(documents):
            authority_payload = documents["authority_v1"]["payload"]
            authority_v1 = authority_payload["aggregate_tensor_sha256"]
            refresh_v1 = documents["refresh_v1"]["payload"]
            if {
                authority_v1,
                refresh_v1["trainer_tensor_sha256"],
                refresh_v1["runtime_tensor_sha256"],
                refresh_v1["fresh_tensor_sha256"],
            } != {authority_v1}:
                raise QualificationArtifactError("v1 authority/refresh binding mismatch")
            if not (
                authority_payload["canonical_config_sha256"]
                == refresh_v1["canonical_config_sha256"]
                and authority_payload["per_tensor_digests"]
                == refresh_v1["runtime_per_tensor_digests"]
                == refresh_v1["fresh_per_tensor_digests"]
                and authority_payload["tensor_count"] == refresh_v1["tensor_count"]
                and authority_payload["total_bytes"] == refresh_v1["total_bytes"]
            ):
                raise QualificationArtifactError("v1 per-tensor/config binding mismatch")
        if "trajectory_step1_manifest" in documents and "refresh_v1" in documents:
            trajectory = documents["trajectory_step1_manifest"]["payload"]
            authority_v1 = documents["authority_v1"]["payload"][
                "aggregate_tensor_sha256"
            ]
            if {
                trajectory["sampler_tensor_sha256"],
                trajectory["trainer_authority_sha256"],
                trajectory["p_old_actor_tensor_sha256"],
            } != {authority_v1}:
                raise QualificationArtifactError("step1 trajectory/v1 authority mismatch")
            if trajectory["refresh_artifact_sha256"] != sha256_file(
                output / "refresh_v1.json"
            ):
                raise QualificationArtifactError("step1 trajectory refresh binding mismatch")
            if bindings is None or trajectory["prompt_manifest_sha256"] != bindings[
                "prompt_manifest_sha256"
            ]:
                raise QualificationArtifactError("step1 trajectory prompt binding mismatch")
        if {"authority_v1", "authority_v2", "refresh_v2"}.issubset(documents):
            authority_v1 = documents["authority_v1"]["payload"][
                "aggregate_tensor_sha256"
            ]
            authority_v2 = documents["authority_v2"]["payload"][
                "aggregate_tensor_sha256"
            ]
            refresh_v2 = documents["refresh_v2"]["payload"]
            authority_v2_payload = documents["authority_v2"]["payload"]
            if authority_v1 == authority_v2 or {
                authority_v2,
                refresh_v2["trainer_tensor_sha256"],
                refresh_v2["runtime_tensor_sha256"],
                refresh_v2["fresh_tensor_sha256"],
            } != {authority_v2} or refresh_v2["previous_tensor_sha256"] != authority_v1:
                raise QualificationArtifactError("v2 authority/refresh binding mismatch")
            if not (
                authority_v2_payload["canonical_config_sha256"]
                == refresh_v2["canonical_config_sha256"]
                and authority_v2_payload["per_tensor_digests"]
                == refresh_v2["runtime_per_tensor_digests"]
                == refresh_v2["fresh_per_tensor_digests"]
                and authority_v2_payload["tensor_count"] == refresh_v2["tensor_count"]
                and authority_v2_payload["total_bytes"] == refresh_v2["total_bytes"]
            ):
                raise QualificationArtifactError("v2 per-tensor/config binding mismatch")
        if {"length_smoke", "length_decision"}.issubset(documents):
            length_payload = documents["length_smoke"]["payload"]
            length_identity = length_payload["policy_identity"]
            authority_v2_payload = documents["authority_v2"]["payload"]
            refresh_v2_payload = documents["refresh_v2"]["payload"]
            expected_checkpoint = (
                output / authority_v2_payload["checkpoint"]["directory"]
            ).resolve()
            if not (
                length_identity["tensor_sha256"]
                == authority_v2_payload["aggregate_tensor_sha256"]
                == refresh_v2_payload["runtime_tensor_sha256"]
                and length_identity["authority_v2_artifact_sha256"]
                == sha256_file(output / "authority_v2.json")
                and Path(length_identity["checkpoint_path"])
                == expected_checkpoint
                and length_identity["active_slot"]
                == refresh_v2_payload["active_adapter"]
                == "student_active"
                and length_identity["registry_count"]
                == len(refresh_v2_payload["registry_after"])
                == 1
            ):
                raise QualificationArtifactError(
                    "length smoke/v2 authority binding mismatch"
                )
            if documents["length_decision"]["payload"][
                "length_smoke_sha256"
            ] != sha256_file(output / "length_smoke.json"):
                raise QualificationArtifactError("length decision/smoke SHA mismatch")
            expected_length = _validate_length_smoke(
                length_payload
            )
            decision = documents["length_decision"]["payload"]
            expected_candidates = (
                [256, 384, 512]
                if documents["length_smoke"]["payload"]["conditional_512_executed"]
                else [256, 384]
            )
            if (
                decision["selected_response_length"] != expected_length
                or decision["evaluated_candidates"] != expected_candidates
            ):
                raise QualificationArtifactError("length decision is not shortest passing")
    except (QualificationArtifactError, KeyError, TypeError) as error:
        errors.append(str(error))

    metrics_path = output / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if not metrics_path.is_file() or metrics_path.is_symlink() or metrics_path.stat().st_size == 0:
        errors.append("metrics.jsonl")
    else:
        try:
            rows = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append("metrics_invalid")
            rows = []
    if len(rows) != len(phases):
        errors.append("metrics_phase_count")
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        try:
            phase = phases[ordinal]
            if not isinstance(row, Mapping):
                raise QualificationArtifactError("metric row is not an object")
            required = {
                "schema_version",
                "artifact_protocol_version",
                "run_id",
                "attempt_id",
                "phase_id",
                "ordinal",
                "config_sha256",
                "run_card_sha256",
                "protocol_sha256",
                "phase_artifact_sha256",
                "metric",
            }
            if set(row) != required:
                raise QualificationArtifactError("metric row fields mismatch")
            if row["phase_id"] in seen:
                raise QualificationArtifactError("duplicate metric phase")
            seen.add(row["phase_id"])
            if row["phase_id"] != phase or row["ordinal"] != ordinal:
                raise QualificationArtifactError("metric phase order mismatch")
            if bindings is None or any(
                row[key] != bindings[key]
                for key in (
                    "run_id",
                    "attempt_id",
                    "config_sha256",
                    "run_card_sha256",
                    "protocol_sha256",
                )
            ):
                raise QualificationArtifactError("metric binding mismatch")
            if row["artifact_protocol_version"] != ARTIFACT_PROTOCOL_VERSION:
                raise QualificationArtifactError("metric protocol mismatch")
            phase_path = output / f"{phase}.json"
            if not phase_path.is_file() or row["phase_artifact_sha256"] != sha256_file(
                phase_path
            ):
                raise QualificationArtifactError("metric phase SHA mismatch")
            _reject_unavailable(row["metric"], path=f"metric.{phase}")
        except (QualificationArtifactError, KeyError, IndexError, TypeError) as error:
            errors.append(str(error))

    if mode == "full" and bindings is not None:
        try:
            _validate_stored_micro_readiness(
                output, bindings=bindings, documents=documents
            )
        except (QualificationArtifactError, KeyError, TypeError) as error:
            errors.append(str(error))

    failure_path = output / "failure.json"
    if failure_path.exists():
        errors.append("failure_present")
        try:
            failure = _read_json(failure_path)
            if bindings is not None and (
                failure.get("run_id") != bindings["run_id"]
                or failure.get("attempt_id") != bindings["attempt_id"]
            ):
                errors.append("failure_binding_mismatch")
            last, last_sha = _last_contiguous(output, phases)
            if (
                failure.get("last_committed_phase") != last
                or failure.get("last_committed_phase_sha256") != last_sha
            ):
                errors.append("failure_last_phase_mismatch")
            metrics_sha = sha256_file(metrics_path) if metrics_path.is_file() else None
            if failure.get("metrics_sha256") != metrics_sha:
                errors.append("failure_metrics_sha_mismatch")
        except QualificationArtifactError as error:
            errors.append(str(error))

    failure_cleanup_path = output / FAILURE_CLEANUP_FILE
    if failure_cleanup_path.exists():
        try:
            cleanup = _read_json(failure_cleanup_path)
            required_cleanup = {
                "schema_version",
                "artifact_protocol_version",
                "run_id",
                "attempt_id",
                "runtime_exit_code",
                "gpu_memory_used_mib",
                "compute_pids",
                "residual_workers",
                "observation_error",
                "cleanup_complete",
                "B2_authorized",
                "B2_started",
            }
            if set(cleanup) != required_cleanup:
                raise QualificationArtifactError("failure cleanup fields mismatch")
            if bindings is None or (
                cleanup["run_id"] != bindings["run_id"]
                or cleanup["attempt_id"] != bindings["attempt_id"]
                or cleanup["schema_version"] != SCHEMA_VERSION
                or cleanup["artifact_protocol_version"]
                != ARTIFACT_PROTOCOL_VERSION
                or cleanup["B2_authorized"] is not False
                or cleanup["B2_started"] is not False
            ):
                raise QualificationArtifactError("failure cleanup binding mismatch")
        except (QualificationArtifactError, KeyError, TypeError) as error:
            errors.append(str(error))

    if verify_index:
        index_path = output / index_name
        if not index_path.is_file() or index_path.is_symlink():
            errors.append(index_name)
        else:
            try:
                index = _read_json(index_path)
                required_index = {
                    "schema_version",
                    "artifact_protocol_version",
                    "run_id",
                    "attempt_id",
                    "mode",
                    "required_phases",
                    "artifact_count",
                    "artifacts",
                }
                if set(index) != required_index:
                    raise QualificationArtifactError("index fields mismatch")
                if bindings is None or index["run_id"] != bindings["run_id"] or index[
                    "attempt_id"
                ] != bindings["attempt_id"]:
                    raise QualificationArtifactError("index binding mismatch")
                if index["mode"] != mode or index["required_phases"] != list(phases):
                    raise QualificationArtifactError("index mode/phase mismatch")
                indexed: set[str] = set()
                for item in index["artifacts"]:
                    if set(item) != {"path", "sha256", "size_bytes"}:
                        raise QualificationArtifactError("index item fields mismatch")
                    relative = _safe_relative(item["path"])
                    if relative in indexed:
                        raise QualificationArtifactError("duplicate index path")
                    indexed.add(relative)
                    artifact_path = output / relative
                    if artifact_path.is_symlink() or not artifact_path.is_file():
                        raise QualificationArtifactError("indexed artifact is not regular")
                    if sha256_file(artifact_path) != item["sha256"]:
                        raise QualificationArtifactError("indexed artifact SHA mismatch")
                    if artifact_path.stat().st_size != item["size_bytes"]:
                        raise QualificationArtifactError("indexed artifact size mismatch")
                expected_indexed = {
                    path.relative_to(output).as_posix()
                    for path in output.rglob("*")
                    if path.is_file()
                    and not path.is_symlink()
                    and path.relative_to(output).as_posix() not in index_exclusions
                }
                if indexed != expected_indexed or index["artifact_count"] != len(indexed):
                    raise QualificationArtifactError("index artifact set mismatch")
            except (QualificationArtifactError, KeyError, TypeError, ValueError) as error:
                errors.append(str(error))

    return sorted(set(errors)), bindings, documents


def _write_index(
    output: Path,
    *,
    mode: str,
    bindings: Mapping[str, Any],
    index_name: str = FINAL_INDEX_FILE,
    exclusions: frozenset[str] = frozenset(
        {FINAL_INDEX_FILE, FINAL_READINESS_FILE}
    ),
) -> dict[str, Any]:
    path = output / index_name
    if path.exists() or path.is_symlink():
        return _read_json(path)
    artifacts = []
    for artifact in sorted(
        output.rglob("*"), key=lambda item: item.relative_to(output).as_posix()
    ):
        relative = artifact.relative_to(output).as_posix()
        if relative in exclusions:
            continue
        if artifact.is_symlink() or not artifact.is_file():
            continue
        artifacts.append(
            {
                "path": relative,
                "sha256": sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        )
    index = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "mode": mode,
        "required_phases": list(_phases(mode)),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _atomic_json(path, index)
    return index


_MICRO_PREFIX_PHASES = FULL_PHASES[: FULL_PHASES.index("trajectory_step1_manifest")]


def _micro_prefix_document(
    output: Path,
    *,
    bindings: Mapping[str, Any],
    expected_v1_tensor_sha256: str,
    expected_refresh_v1_sha256: str,
    require_exact_metric_count: bool,
) -> dict[str, Any]:
    """Reopen the durable Phase-A prefix before any step-1 rollout.

    This deliberately does not reuse the full-run inspector: at the micro
    boundary the later phase files do not exist yet.  Every available envelope,
    checkpoint, metric row, and cross-artifact identity is nevertheless reopened
    and recomputed from disk.
    """

    _validate_bindings(bindings)
    if (output / "failure.json").exists():
        raise QualificationArtifactError("micro evidence has a failure artifact")
    static_errors = validate_qualification_static_artifacts(
        output, bindings=bindings, mode="full", require_summary=False
    )
    if static_errors:
        raise QualificationArtifactError(
            "micro static evidence invalid:" + ",".join(static_errors)
        )

    documents: dict[str, dict[str, Any]] = {}
    phase_hashes: dict[str, str] = {}
    previous_path: str | None = None
    previous_sha: str | None = None
    probe_content: str | None = None
    probe_file: str | None = None
    probe_ordinal = _MICRO_PREFIX_PHASES.index("probe_manifest")
    for ordinal, phase in enumerate(_MICRO_PREFIX_PHASES):
        path = output / f"{phase}.json"
        if path.is_symlink() or not path.is_file():
            raise QualificationArtifactError(f"micro phase is absent: {phase}")
        value = _read_json(path)
        if set(value) != _ENVELOPE_FIELDS:
            raise QualificationArtifactError(f"{phase} envelope fields mismatch")
        if (
            value["schema_version"] != SCHEMA_VERSION
            or value["artifact_protocol_version"] != ARTIFACT_PROTOCOL_VERSION
            or {key: value[key] for key in _BINDING_FIELDS} != dict(bindings)
            or value["phase_id"] != phase
            or value["ordinal"] != ordinal
            or value["previous_phase_path"] != previous_path
            or value["previous_phase_sha256"] != previous_sha
            or value["payload_sha256"]
            != canonical_json_sha256(value["payload"])
        ):
            raise QualificationArtifactError(f"micro phase envelope mismatch: {phase}")
        _validate_phase_payload(phase, value["payload"])
        if phase == "authority_v1":
            _validate_checkpoint_binding(output, value["payload"], phase=phase)
        if phase == "probe_manifest":
            probe_content = _validate_probe_payload(
                value["payload"],
                spec=_probe_spec_from_static(output, bindings),
            )
            probe_file = sha256_file(path)
            if (
                value["probe_manifest_content_sha256"] != probe_content
                or value["probe_manifest_file_sha256"] is not None
            ):
                raise QualificationArtifactError("micro probe self-binding mismatch")
        elif ordinal < probe_ordinal:
            if (
                value["probe_manifest_content_sha256"] is not None
                or value["probe_manifest_file_sha256"] is not None
            ):
                raise QualificationArtifactError("micro pre-probe binding mismatch")
        elif (
            value["probe_manifest_content_sha256"] != probe_content
            or value["probe_manifest_file_sha256"] != probe_file
        ):
            raise QualificationArtifactError(f"micro probe binding mismatch: {phase}")
        documents[phase] = value
        phase_sha = sha256_file(path)
        phase_hashes[phase] = phase_sha
        previous_path = path.name
        previous_sha = phase_sha

    probe_payload = documents["probe_manifest"]["payload"]
    if probe_payload["prompt_manifest_sha256"] != bindings["prompt_manifest_sha256"]:
        raise QualificationArtifactError("micro probe/prompt manifest binding mismatch")
    if documents["v0_guard"]["payload"]["normal_v0"]["run_id"] != bindings[
        "run_id"
    ]:
        raise QualificationArtifactError("micro v0 guard/envelope run binding mismatch")
    reconstruction = documents["reconstruction_step0"]["payload"]["telemetry"]
    if reconstruction["run_id"] != bindings["run_id"]:
        raise QualificationArtifactError("micro reconstruction run binding mismatch")
    authority = documents["authority_v1"]["payload"]
    refresh = documents["refresh_v1"]["payload"]
    v1_tensor = authority["aggregate_tensor_sha256"]
    if not (
        _is_hex(expected_v1_tensor_sha256, 64)
        and v1_tensor == expected_v1_tensor_sha256
        and {
            v1_tensor,
            refresh["trainer_tensor_sha256"],
            refresh["runtime_tensor_sha256"],
            refresh["fresh_tensor_sha256"],
        }
        == {v1_tensor}
        and authority["canonical_config_sha256"]
        == refresh["canonical_config_sha256"]
        and authority["per_tensor_digests"]
        == refresh["runtime_per_tensor_digests"]
        == refresh["fresh_per_tensor_digests"]
        and authority["tensor_count"] == refresh["tensor_count"]
        and authority["total_bytes"] == refresh["total_bytes"]
    ):
        raise QualificationArtifactError("micro v1 authority/refresh binding mismatch")
    refresh_sha = phase_hashes["refresh_v1"]
    if not (
        _is_hex(expected_refresh_v1_sha256, 64)
        and expected_refresh_v1_sha256 == refresh_sha
    ):
        raise QualificationArtifactError("micro refresh_v1 artifact SHA mismatch")

    metrics_path = output / "metrics.jsonl"
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise QualificationArtifactError("micro metrics are absent")
    try:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationArtifactError("micro metrics are invalid") from error
    if len(rows) < len(_MICRO_PREFIX_PHASES) or (
        require_exact_metric_count and len(rows) != len(_MICRO_PREFIX_PHASES)
    ):
        raise QualificationArtifactError("micro metrics prefix count mismatch")
    prefix_rows = rows[: len(_MICRO_PREFIX_PHASES)]
    required_metric_fields = {
        "schema_version",
        "artifact_protocol_version",
        "run_id",
        "attempt_id",
        "phase_id",
        "ordinal",
        "config_sha256",
        "run_card_sha256",
        "protocol_sha256",
        "phase_artifact_sha256",
        "metric",
    }
    for ordinal, row in enumerate(prefix_rows):
        phase = _MICRO_PREFIX_PHASES[ordinal]
        if not isinstance(row, Mapping) or set(row) != required_metric_fields:
            raise QualificationArtifactError("micro metric row fields mismatch")
        if not (
            row["schema_version"] == SCHEMA_VERSION
            and row["artifact_protocol_version"] == ARTIFACT_PROTOCOL_VERSION
            and row["run_id"] == bindings["run_id"]
            and row["attempt_id"] == bindings["attempt_id"]
            and row["config_sha256"] == bindings["config_sha256"]
            and row["run_card_sha256"] == bindings["run_card_sha256"]
            and row["protocol_sha256"] == bindings["protocol_sha256"]
            and row["phase_id"] == phase
            and row["ordinal"] == ordinal
            and row["phase_artifact_sha256"] == phase_hashes[phase]
        ):
            raise QualificationArtifactError("micro metric row binding mismatch")
        _reject_unavailable(row["metric"], path=f"micro.metric.{phase}")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "status": "pass",
        "ready": True,
        "production_sampler_refresh_ready": True,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
        "last_phase": "refresh_v1",
        "phase_artifact_sha256": phase_hashes,
        "metrics_prefix_row_count": len(prefix_rows),
        "metrics_prefix_sha256": canonical_json_sha256(prefix_rows),
        "probe_manifest_content_sha256": probe_content,
        "probe_manifest_file_sha256": probe_file,
        "v1_tensor_sha256": v1_tensor,
        "refresh_v1_artifact_sha256": refresh_sha,
        "bindings_sha256": canonical_json_sha256(dict(bindings)),
    }


def assert_micro_evidence_prefix_ready(
    output: str | Path,
    bindings: Mapping[str, Any],
    expected_v1_tensor_sha256: str,
    expected_refresh_v1_sha256: str,
) -> dict[str, Any]:
    """Persist and return the Phase-A readiness boundary before rollout-1."""

    directory = Path(output)
    document = _micro_prefix_document(
        directory,
        bindings=bindings,
        expected_v1_tensor_sha256=expected_v1_tensor_sha256,
        expected_refresh_v1_sha256=expected_refresh_v1_sha256,
        require_exact_metric_count=not (directory / MICRO_READINESS_FILE).exists(),
    )
    path = directory / MICRO_READINESS_FILE
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _read_json(path) != document:
            raise QualificationArtifactError("stored micro readiness mismatch")
    else:
        _atomic_json(path, document)
    return {**document, "micro_readiness_sha256": sha256_file(path)}


def _validate_stored_micro_readiness(
    output: Path,
    *,
    bindings: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> str:
    path = output / MICRO_READINESS_FILE
    if path.is_symlink() or not path.is_file():
        raise QualificationArtifactError("micro_readiness.json is absent")
    if not {"authority_v1", "refresh_v1"}.issubset(documents) or not (
        output / "refresh_v1.json"
    ).is_file():
        raise QualificationArtifactError("micro v1 authority/refresh evidence is absent")
    expected = _micro_prefix_document(
        output,
        bindings=bindings,
        expected_v1_tensor_sha256=documents["authority_v1"]["payload"][
            "aggregate_tensor_sha256"
        ],
        expected_refresh_v1_sha256=sha256_file(output / "refresh_v1.json"),
        require_exact_metric_count=False,
    )
    if _read_json(path) != expected:
        raise QualificationArtifactError("stored micro readiness mismatch")
    return sha256_file(path)


def _computed_readiness(
    output: Path, *, mode: str, stage: str
) -> dict[str, Any]:
    if stage not in {"evidence", "final"}:
        raise QualificationArtifactError("readiness stage is invalid")
    evidence_stage = stage == "evidence" and mode == "full"
    index_name = EVIDENCE_INDEX_FILE if evidence_stage else FINAL_INDEX_FILE
    exclusions = (
        _EVIDENCE_INDEX_EXCLUSIONS if evidence_stage else _FINAL_INDEX_EXCLUSIONS
    )
    errors, bindings, documents = _inspect_graph(
        output,
        mode=mode,
        verify_index=True,
        index_name=index_name,
        index_exclusions=exclusions,
    )
    if stage == "final" and mode == "full":
        evidence = _derive_evidence_readiness(output)
        if not evidence.get("ready") or not evidence.get(
            "authorization_eligibility"
        ):
            errors.append("evidence_readiness_invalid")
        try:
            _validate_root_authorization_evidence(output)
        except QualificationArtifactError as error:
            errors.append(str(error))
    errors = sorted(set(errors))
    ready = not errors and bindings is not None
    index_path = output / index_name
    summary_path = output / "terminal_summary.json"
    public_summary_path = output / "summary.json"
    cleanup_path = output / "cleanup.json"
    probe_path = output / "probe_manifest.json"
    micro_readiness_path = output / MICRO_READINESS_FILE
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "mode": mode,
        "readiness_stage": stage,
        "run_id": bindings["run_id"] if bindings else "invalid",
        "attempt_id": bindings["attempt_id"] if bindings else "invalid",
        "ready": ready,
        "production_sampler_refresh_ready": ready,
        "authorization_eligibility": bool(ready and mode == "full"),
        "OPD_scoring_backend_ready": bool(
            ready and mode == "full" and stage == "final"
        ),
        "B2_authorized": bool(ready and mode == "full" and stage == "final"),
        "B2_started": False,
        "failure_reasons": errors,
        "artifact_index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        "terminal_summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
        "summary_sha256": (
            sha256_file(public_summary_path) if public_summary_path.is_file() else None
        ),
        "cleanup_sha256": sha256_file(cleanup_path) if cleanup_path.is_file() else None,
        "probe_manifest_file_sha256": sha256_file(probe_path) if probe_path.is_file() else None,
        "probe_manifest_content_sha256": (
            documents.get("probe_manifest", {})
            .get("payload", {})
            .get("manifest_sha256")
        ),
        "micro_readiness_sha256": (
            sha256_file(micro_readiness_path)
            if mode == "full" and micro_readiness_path.is_file()
            else None
        ),
        "evidence_artifact_index_sha256": (
            sha256_file(output / EVIDENCE_INDEX_FILE)
            if (output / EVIDENCE_INDEX_FILE).is_file()
            else None
        ),
        "evidence_readiness_sha256": (
            sha256_file(output / EVIDENCE_READINESS_FILE)
            if stage == "final" and (output / EVIDENCE_READINESS_FILE).is_file()
            else None
        ),
        "b2_authorization_sha256": (
            sha256_file(output / ROOT_AUTHORIZATION_FILE)
            if stage == "final" and (output / ROOT_AUTHORIZATION_FILE).is_file()
            else None
        ),
    }
    required_bindings = [
        result["artifact_index_sha256"],
        result["terminal_summary_sha256"],
        result["summary_sha256"],
        result["cleanup_sha256"],
        result["probe_manifest_file_sha256"],
        result["probe_manifest_content_sha256"],
    ]
    if mode == "full":
        required_bindings.append(result["micro_readiness_sha256"])
    if stage == "final" and mode == "full":
        required_bindings.extend(
            [
                result["evidence_artifact_index_sha256"],
                result["evidence_readiness_sha256"],
                result["b2_authorization_sha256"],
            ]
        )
    if ready and any(value is None for value in required_bindings):
        result["ready"] = False
        result["production_sampler_refresh_ready"] = False
        result["authorization_eligibility"] = False
        result["OPD_scoring_backend_ready"] = False
        result["B2_authorized"] = False
        result["failure_reasons"] = ["readiness_binding_unavailable"]
    return result


def _derive_stored_readiness(
    output: Path, *, mode: str, stage: str, readiness_name: str
) -> dict[str, Any]:
    computed = _computed_readiness(output, mode=mode, stage=stage)
    readiness_path = output / readiness_name
    if readiness_path.exists():
        try:
            stored = _read_json(readiness_path)
            if stored != computed:
                computed["ready"] = False
                computed["production_sampler_refresh_ready"] = False
                computed["authorization_eligibility"] = False
                computed["OPD_scoring_backend_ready"] = False
                computed["B2_authorized"] = False
                computed["failure_reasons"] = sorted(
                    set(computed["failure_reasons"] + ["stored_readiness_mismatch"])
                )
        except QualificationArtifactError as error:
            computed["ready"] = False
            computed["production_sampler_refresh_ready"] = False
            computed["authorization_eligibility"] = False
            computed["OPD_scoring_backend_ready"] = False
            computed["B2_authorized"] = False
            computed["failure_reasons"] = sorted(
                set(computed["failure_reasons"] + [str(error)])
            )
    return computed


def _derive_evidence_readiness(output: Path) -> dict[str, Any]:
    return _derive_stored_readiness(
        output,
        mode="full",
        stage="evidence",
        readiness_name=EVIDENCE_READINESS_FILE,
    )


def derive_qualification_readiness(
    output: str | Path, *, mode: str = "full"
) -> dict[str, Any]:
    """Reopen the indexed graph and recompute readiness without caller booleans."""

    directory = Path(output)
    if mode == "full" and not (directory / FINAL_READINESS_FILE).exists():
        return _derive_evidence_readiness(directory)
    return _derive_stored_readiness(
        directory,
        mode=mode,
        stage="final",
        readiness_name=FINAL_READINESS_FILE,
    )


def finalize_qualification(
    output: str | Path, *, mode: str = "full"
) -> dict[str, Any]:
    """Advance one immutable finalization level and return disk-derived state.

    A full run first seals qualification evidence without granting OPD/B2.  A
    later call, after calibration authorization is copied into the root (or a
    materialization failure is recorded), seals the final index and readiness.
    """

    directory = Path(output)
    phases = _phases(mode)
    bindings = _load_bindings_from_phase(directory, phases)
    pre_errors, _, _ = _inspect_graph(directory, mode=mode, verify_index=False)
    failure_path = directory / "failure.json"
    if pre_errors and not failure_path.exists():
        record_failure(
            directory,
            bindings=bindings,
            mode=mode,
            reason="artifact_graph_invalid:" + ",".join(pre_errors),
        )
    if mode == "full" and not (directory / EVIDENCE_INDEX_FILE).exists():
        _write_index(
            directory,
            mode=mode,
            bindings=bindings,
            index_name=EVIDENCE_INDEX_FILE,
            exclusions=_EVIDENCE_INDEX_EXCLUSIONS,
        )
        result = _computed_readiness(directory, mode=mode, stage="evidence")
        readiness_path = directory / EVIDENCE_READINESS_FILE
    elif mode == "full" and not (directory / EVIDENCE_READINESS_FILE).exists():
        result = _computed_readiness(directory, mode=mode, stage="evidence")
        readiness_path = directory / EVIDENCE_READINESS_FILE
    elif mode == "full" and not (
        (directory / ROOT_AUTHORIZATION_FILE).is_file()
        or (directory / "failure.json").is_file()
        or not _derive_evidence_readiness(directory).get("ready")
    ):
        return _derive_evidence_readiness(directory)
    else:
        if (
            mode == "full"
            and (directory / ROOT_AUTHORIZATION_FILE).is_file()
            and not failure_path.exists()
        ):
            try:
                _validate_root_authorization_evidence(directory)
            except QualificationArtifactError as error:
                record_failure(
                    directory,
                    bindings=bindings,
                    mode=mode,
                    reason=f"b2_authorization_invalid:{error}",
                )
        _write_index(
            directory,
            mode=mode,
            bindings=bindings,
            index_name=FINAL_INDEX_FILE,
            exclusions=_FINAL_INDEX_EXCLUSIONS,
        )
        result = _computed_readiness(directory, mode=mode, stage="final")
        readiness_path = directory / FINAL_READINESS_FILE
    if readiness_path.exists() or readiness_path.is_symlink():
        stored = _read_json(readiness_path)
        if stored != result:
            raise QualificationArtifactError("readiness already exists with different content")
    else:
        _atomic_json(readiness_path, result)
    return result


def _phase_sha(output: Path, phase: str) -> str:
    return sha256_file(output / f"{phase}.json")


def _v2_transport_binding(output: Path) -> dict[str, str]:
    authority = _read_json(output / "authority_v2.json")["payload"]
    checkpoint = (output / authority["checkpoint"]["directory"]).resolve()
    return {
        "output_path": str(output.resolve()),
        "v2_checkpoint_path": str(checkpoint),
        "v2_tensor_sha256": authority["aggregate_tensor_sha256"],
    }


_CALIBRATION_CONFIG_FIELDS = {
    "schema_id",
    "schema_version",
    "run",
    "production_backend",
    "executor",
    "model",
    "teacher",
    "data",
    "protocol",
    "generation",
    "qualification",
    "authorization",
    "isolation",
    "execution",
}
_CALIBRATION_CARD_FIELDS = {
    "schema_id",
    "schema_version",
    "run_id",
    "stage",
    "status",
    "config_path",
    "config_sha256",
    "production_backend_id",
    "executor",
    "selected_response_length",
    "optimizer_steps",
    "qualification",
    "automatically_start_b2",
    "automatically_run_idt_sar_ca_opd",
    "automatically_access_controller_confirmation_final",
    "production_sampler_refresh_ready_now",
    "OPD_scoring_backend_ready_now",
    "B2_authorized_now",
    "B2_started",
}


def _exact_fields(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        missing = sorted(expected - set(value) if isinstance(value, Mapping) else expected)
        extra = sorted(set(value) - expected if isinstance(value, Mapping) else set())
        raise QualificationArtifactError(
            f"{label} fields mismatch; missing={missing}, extra={extra}"
        )
    return value


def _mapping_document(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationArtifactError(f"mapping document is absent: {path}")
    try:
        value = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.suffix.lower() == ".json"
            else yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise QualificationArtifactError(f"mapping document is invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise QualificationArtifactError(f"mapping document is not an object: {path}")
    return dict(value)


def _resolved_asset_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise QualificationArtifactError(f"{label} path is invalid")
    configured = Path(value)
    path = configured if configured.is_absolute() else REPO_ROOT / configured
    if path.is_symlink() or not path.is_file():
        raise QualificationArtifactError(f"{label} is absent or not regular")
    return path


def _validate_asset_sha(value: Any, expected: Any, *, label: str) -> Path:
    path = _resolved_asset_path(value, label=label)
    if not _is_hex(expected, 64) or sha256_file(path) != expected:
        raise QualificationArtifactError(f"{label} SHA mismatch")
    return path


def _ordered_adapter_transport_sha256(directory_value: Any) -> str:
    directory = Path(str(directory_value))
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    if directory.is_symlink() or not directory.is_dir():
        raise QualificationArtifactError("Teacher adapter directory is invalid")
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise QualificationArtifactError(f"Teacher adapter lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _production_executor_descriptor() -> dict[str, Any]:
    """Bind the independently callable B2 executor without importing GPU code."""

    configured = Path(B2_EXECUTOR_SOURCE_PATH)
    source_path = configured if configured.is_absolute() else REPO_ROOT / configured
    if source_path.is_symlink() or not source_path.is_file():
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: executor source is absent"
        )
    try:
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: executor source is invalid"
        ) from error
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == B2_EXECUTOR_SYMBOL
    ]
    if len(definitions) != 1:
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: executor symbol is absent or ambiguous"
        )
    definition = definitions[0]
    parameters = {
        argument.arg
        for argument in (
            list(definition.args.posonlyargs)
            + list(definition.args.args)
            + list(definition.args.kwonlyargs)
        )
    }
    executable_body = list(definition.body)
    if executable_body and isinstance(executable_body[0], ast.Expr) and isinstance(
        executable_body[0].value, ast.Constant
    ) and isinstance(executable_body[0].value.value, str):
        executable_body = executable_body[1:]
    segment = ast.get_source_segment(source_text, definition) or ""
    forbidden = ("notimplemented", "placeholder", "todo", "mock")
    if (
        not {"config", "config_path"}.issubset(parameters)
        or not executable_body
        or all(isinstance(node, (ast.Pass, ast.Raise)) for node in executable_body)
        or not any(isinstance(node, ast.Call) for node in ast.walk(definition))
        or any(marker in segment.lower() for marker in forbidden)
    ):
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: executor symbol is not an executable contract"
        )
    try:
        relative = source_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        # Tests may bind an isolated source. Formal packages always use the
        # frozen repository-relative constant above.
        relative = str(source_path.resolve())
    return {
        "path": relative,
        "symbol": B2_EXECUTOR_SYMBOL,
        "source_sha256": sha256_file(source_path),
    }


def _load_frozen_b2_templates() -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = REPO_ROOT / B2_TEMPLATE_CONFIG_PATH
    card_path = REPO_ROOT / B2_TEMPLATE_RUN_CARD_PATH
    if (
        sha256_file(config_path) != B2_TEMPLATE_CONFIG_SHA256
        or sha256_file(card_path) != B2_TEMPLATE_RUN_CARD_SHA256
    ):
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: frozen B2 template SHA mismatch"
        )
    config = _mapping_document(config_path)
    card = _mapping_document(card_path)
    if (
        config.get("schema_version") != 5
        or card.get("schema_version") != 5
        or config.get("production_backend") != card.get("production_backend")
        or config.get("production_backend", {}).get("backend_id")
        != B2_PRODUCTION_BACKEND_ID
    ):
        raise QualificationArtifactError(
            f"{B2_EXECUTOR_BLOCKER}: frozen B2 template binding mismatch"
        )
    return config, card


def _validate_calibration_documents(
    config: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    config_sha256: str,
    executor: Mapping[str, Any],
    verify_base_weight_payloads: bool = False,
) -> None:
    """Independent CPU package preflight; no caller-authored pass is trusted."""

    _exact_fields(config, _CALIBRATION_CONFIG_FIELDS, label="calibration config")
    _exact_fields(card, _CALIBRATION_CARD_FIELDS, label="calibration run-card")
    run = _exact_fields(
        config["run"],
        {
            "run_id",
            "stage",
            "baseline_id",
            "purpose",
            "seed",
            "status",
            "optimizer_steps",
            "output_dir",
            "automatically_start",
        },
        label="calibration run",
    )
    model = _exact_fields(
        config["model"],
        {
            "base_path",
            "base_manifest_path",
            "base_manifest_sha256",
            "base_weights_manifest_path",
            "base_weights_manifest_sha256",
            "model_revision",
            "tokenizer_revision",
            "dtype",
            "attention_backend",
        },
        label="calibration model",
    )
    teacher = _exact_fields(
        config["teacher"],
        {
            "adapter_path",
            "adapter_sha256",
            "adapter_weight_sha256",
            "manifest_path",
            "manifest_sha256",
            "role",
            "same_token_scoring",
        },
        label="calibration teacher",
    )
    data = _exact_fields(
        config["data"],
        {
            "protocol_version",
            "prompt_manifest_path",
            "prompt_manifest_sha256",
            "selection_rule",
            "allowed_roles",
            "prompt_only",
            "final_labels_allowed",
        },
        label="calibration data",
    )
    protocol = _exact_fields(
        config["protocol"],
        {
            "three_policy_formula_path",
            "three_policy_formula_sha256",
            "correction_upper_threshold",
            "correction_ess_fraction_min",
            "correction_cap_fraction_max",
            "same_path_max_gap",
            "optimizer",
            "learning_rate",
            "ppo_clip_low",
            "ppo_clip_high",
            "student_lora_rank",
            "student_lora_alpha",
            "student_lora_target_modules",
            "prompt_equal_reduction",
            "qualification_protocol_sha256",
        },
        label="calibration protocol",
    )
    generation = _exact_fields(
        config["generation"],
        {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "full_support",
            "enable_thinking",
            "use_cache",
        },
        label="calibration generation",
    )
    qualification = _exact_fields(
        config["qualification"],
        {
            "run_id",
            "attempt_id",
            "readiness_sha256",
            "artifact_index_sha256",
            "qualification_config_sha256",
            "qualification_run_card_sha256",
            "backend_binding_sha256",
            "protocol_sha256",
            "prompt_manifest_sha256",
            "probe_spec_sha256",
            "data_manifest_sha256",
            "git_commit",
            "authority_v1_sha256",
            "authority_v2_sha256",
            "output_path",
            "v2_checkpoint_path",
            "v2_tensor_sha256",
            "two_step_trajectory_sha256",
            "base_null_sha256",
            "length_smoke_sha256",
            "length_decision_sha256",
            "cleanup_sha256",
        },
        label="calibration qualification binding",
    )
    authorization = _exact_fields(
        config["authorization"],
        {
            "source",
            "production_sampler_refresh_ready",
            "OPD_scoring_backend_ready",
            "B2_authorized",
            "B2_started",
        },
        label="calibration authorization",
    )
    execution = _exact_fields(
        config["execution"],
        {
            "optimizer_steps",
            "calibration_only",
            "automatically_start_b2",
            "automatically_run_idt",
            "automatically_run_sar",
            "automatically_run_ca_opd",
            "automatically_run_controller",
            "automatically_run_confirmation",
            "automatically_run_final",
        },
        label="calibration execution",
    )
    selected = generation["max_new_tokens"]
    template, _ = _load_frozen_b2_templates()
    template_backend = template["production_backend"]
    template_models = template["model_paths"]
    template_data = template["data"]
    template_protocol = template["protocol_binding"]
    _validate_asset_sha(
        model["base_manifest_path"],
        model["base_manifest_sha256"],
        label="Base artifact manifest",
    )
    validate_base_model_transport(
        base_path=model["base_path"],
        weights_manifest_path=_resolved_asset_path(
            model["base_weights_manifest_path"],
            label="Base weights manifest",
        ),
        weights_manifest_sha256=model["base_weights_manifest_sha256"],
        expected_revision=model["model_revision"],
        verify_weight_payloads=verify_base_weight_payloads,
    )
    _validate_asset_sha(
        teacher["manifest_path"],
        teacher["manifest_sha256"],
        label="Teacher manifest",
    )
    _validate_asset_sha(
        Path(str(teacher["adapter_path"])) / "adapter_model.safetensors",
        teacher["adapter_weight_sha256"],
        label="Teacher adapter weights",
    )
    if (
        not _is_hex(teacher["adapter_sha256"], 64)
        or _ordered_adapter_transport_sha256(teacher["adapter_path"])
        != teacher["adapter_sha256"]
    ):
        raise QualificationArtifactError("Teacher ordered adapter SHA mismatch")
    _validate_asset_sha(
        data["prompt_manifest_path"],
        data["prompt_manifest_sha256"],
        label="B2 prompt manifest",
    )
    if not (
        config["schema_id"] == B2_CALIBRATION_SCHEMA_ID
        and config["schema_version"] == B2_CALIBRATION_SCHEMA_VERSION
        and card["schema_id"] == B2_CALIBRATION_SCHEMA_ID
        and card["schema_version"] == B2_CALIBRATION_SCHEMA_VERSION
        and run["run_id"] == card["run_id"] == B2_CALIBRATION_RUN_ID
        and run["stage"] == card["stage"] == "b2_medical_opd_calibration"
        and run["status"] == card["status"] == "authorized_not_started"
        and run["seed"] == 42
        and run["optimizer_steps"]
        == execution["optimizer_steps"]
        == card["optimizer_steps"]
        == B2_CALIBRATION_OPTIMIZER_STEPS
        and run["automatically_start"] is False
        and card["config_path"] == "config.yaml"
        and card["config_sha256"] == config_sha256
    ):
        raise QualificationArtifactError("calibration run/config identity failed")
    backend = config["production_backend"]
    if (
        not isinstance(backend, Mapping)
        or backend.get("backend_id") != B2_PRODUCTION_BACKEND_ID
        or dict(backend) != dict(template_backend)
    ):
        raise QualificationArtifactError("calibration production backend mismatch")
    if not (
        card["production_backend_id"] == B2_PRODUCTION_BACKEND_ID
        and dict(config["executor"]) == dict(card["executor"]) == dict(executor)
        and selected in {256, 384, 512}
        and selected == card["selected_response_length"]
        and model["model_revision"] == backend.get("model_revision")
        and model["tokenizer_revision"] == backend.get("tokenizer_revision")
        and model["dtype"] == backend.get("dtype")
        and model["attention_backend"] == backend.get("attention_backend")
        and model["base_path"] == template_models["base"]
        and model["base_manifest_path"] == template_models["base_manifest"]
        and model["base_weights_manifest_path"]
        == template_models["base_weights_manifest"]
        and teacher["adapter_path"] == template_models["teacher_adapter"]
        and teacher["manifest_path"] == template_models["teacher_manifest"]
        and teacher["role"] == "single_frozen_medical_teacher"
        and teacher["same_token_scoring"] is True
        and data["protocol_version"] == template_data["protocol_version"]
        and data["prompt_manifest_path"] == template_data["prompt_manifest"]
        and data["prompt_manifest_sha256"] == qualification["data_manifest_sha256"]
        and data["selection_rule"] == B2_CALIBRATION_SELECTION_RULE
        and data["allowed_roles"] == ["medical_opd_o1", "medical_opd_cmb"]
        and data["prompt_only"] is True
        and data["final_labels_allowed"] is False
        and Path(qualification["output_path"]).is_absolute()
        and Path(qualification["v2_checkpoint_path"])
        == Path(qualification["output_path"]) / "checkpoints" / "v2"
        and isinstance(qualification["v2_tensor_sha256"], str)
        and len(qualification["v2_tensor_sha256"]) == 64
        and {
            key: protocol[key]
            for key in template_protocol
        }
        == template_protocol
        and protocol["qualification_protocol_sha256"]
        == qualification["protocol_sha256"]
        and generation["do_sample"] is True
        and generation["temperature"] == 1.0
        and generation["top_k"] == 0
        and generation["top_p"] == 1.0
        and generation["full_support"] is True
        and generation["enable_thinking"] is False
        and generation["use_cache"] is True
        and config["isolation"] == _ISOLATION
        and authorization
        == {
            "source": "artifact_derived_p4_6_full_qualification",
            "production_sampler_refresh_ready": True,
            "OPD_scoring_backend_ready": True,
            "B2_authorized": True,
            "B2_started": False,
        }
        and execution["calibration_only"] is True
        and execution["automatically_start_b2"] is False
        and all(
            execution[field] is False
            for field in (
                "automatically_run_idt",
                "automatically_run_sar",
                "automatically_run_ca_opd",
                "automatically_run_controller",
                "automatically_run_confirmation",
                "automatically_run_final",
            )
        )
        and card["qualification"] == qualification
        and card["automatically_start_b2"] is False
        and card["automatically_run_idt_sar_ca_opd"] is False
        and card["automatically_access_controller_confirmation_final"] is False
        and card["production_sampler_refresh_ready_now"] is True
        and card["OPD_scoring_backend_ready_now"] is True
        and card["B2_authorized_now"] is True
        and card["B2_started"] is False
    ):
        raise QualificationArtifactError("calibration CPU dry-run contract failed")


def _validate_package_preflight(
    preflight: Mapping[str, Any],
    *,
    config_sha256: str,
    run_card_sha256: str,
    executor: Mapping[str, Any],
    qualification: Mapping[str, Any],
    selected_response_length: int,
) -> None:
    _exact_fields(
        preflight,
        {
            "schema_id",
            "schema_version",
            "status",
            "run_id",
            "config_sha256",
            "run_card_sha256",
            "executor",
            "selected_response_length",
            "optimizer_steps",
            "qualification_readiness_sha256",
            "qualification_artifact_index_sha256",
            "checks",
        },
        label="calibration package preflight",
    )
    checks = _exact_fields(
        preflight["checks"],
        {
            "strict_config_schema",
            "strict_run_card_schema",
            "source_bound_executor",
            "qualification_graph_recomputed",
            "selected_length_artifact_derived",
            "auto_start_disabled",
            "isolation_closed",
        },
        label="calibration package preflight checks",
    )
    if not (
        preflight["schema_id"]
        == "ca-opd/b2-medical-opd-calibration-preflight/v1"
        and preflight["schema_version"] == 1
        and preflight["status"] == "pass"
        and preflight["run_id"] == B2_CALIBRATION_RUN_ID
        and preflight["config_sha256"] == config_sha256
        and preflight["run_card_sha256"] == run_card_sha256
        and dict(preflight["executor"]) == dict(executor)
        and preflight["selected_response_length"] == selected_response_length
        and preflight["optimizer_steps"] == B2_CALIBRATION_OPTIMIZER_STEPS
        and preflight["qualification_readiness_sha256"]
        == qualification["readiness_sha256"]
        and preflight["qualification_artifact_index_sha256"]
        == qualification["artifact_index_sha256"]
        and all(value is True for value in checks.values())
    ):
        raise QualificationArtifactError("calibration package preflight failed")


def _materialize_b2_calibration_package_unchecked(
    qualification_output: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Create a source-bound B2 package, preflight it, then authorize last.

    Full qualification is necessary but not sufficient: if the independent B2
    executor is absent or only a placeholder, no target directory is emitted.
    """

    source = Path(qualification_output).resolve()
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise QualificationArtifactError("calibration destination already exists")
    readiness = _derive_evidence_readiness(source)
    if not readiness.get("ready") or not readiness.get("authorization_eligibility"):
        raise QualificationArtifactError("full qualification eligibility is absent")
    length_document = _read_json(source / "length_decision.json")
    selected = length_document["payload"].get("selected_response_length")
    if selected not in {256, 384, 512}:
        raise QualificationArtifactError("authorized response length is absent")
    first = _read_json(source / "launch_record.json")
    bindings = {key: first[key] for key in _BINDING_FIELDS}
    _validate_bindings(bindings)
    executor = _production_executor_descriptor()
    template, _ = _load_frozen_b2_templates()
    template_backend = dict(template["production_backend"])
    template_model = dict(template["model_paths"])
    template_data = dict(template["data"])
    template_protocol = dict(template["protocol_binding"])
    v2_transport = _v2_transport_binding(source)
    qualification_config = yaml.safe_load(
        (source / "config.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(qualification_config, Mapping):
        raise QualificationArtifactError("qualification config artifact is invalid")
    qualification_model = qualification_config.get("model")
    qualification_teacher = qualification_config.get("teacher")
    if not isinstance(qualification_model, Mapping) or not isinstance(
        qualification_teacher, Mapping
    ):
        raise QualificationArtifactError("qualification model/Teacher binding is absent")

    qualification = {
        "run_id": bindings["run_id"],
        "attempt_id": bindings["attempt_id"],
        "readiness_sha256": sha256_file(source / EVIDENCE_READINESS_FILE),
        "artifact_index_sha256": sha256_file(source / EVIDENCE_INDEX_FILE),
        "qualification_config_sha256": bindings["config_sha256"],
        "qualification_run_card_sha256": bindings["run_card_sha256"],
        "backend_binding_sha256": bindings["backend_binding_sha256"],
        "protocol_sha256": bindings["protocol_sha256"],
        "prompt_manifest_sha256": bindings["prompt_manifest_sha256"],
        "probe_spec_sha256": bindings["probe_spec_sha256"],
        "data_manifest_sha256": bindings["data_manifest_sha256"],
        "git_commit": bindings["git_commit"],
        "authority_v1_sha256": _phase_sha(source, "authority_v1"),
        "authority_v2_sha256": _phase_sha(source, "authority_v2"),
        **v2_transport,
        "two_step_trajectory_sha256": _phase_sha(
            source, "trajectory_step1_manifest"
        ),
        "base_null_sha256": _phase_sha(source, "base_null"),
        "length_smoke_sha256": _phase_sha(source, "length_smoke"),
        "length_decision_sha256": _phase_sha(source, "length_decision"),
        "cleanup_sha256": _phase_sha(source, "cleanup"),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        config = {
            "schema_id": B2_CALIBRATION_SCHEMA_ID,
            "schema_version": B2_CALIBRATION_SCHEMA_VERSION,
            "run": {
                "run_id": B2_CALIBRATION_RUN_ID,
                "stage": "b2_medical_opd_calibration",
                "baseline_id": "B2",
                "purpose": "production-bound single-Medical-Teacher OPD calibration",
                "seed": 42,
                "status": "authorized_not_started",
                "optimizer_steps": B2_CALIBRATION_OPTIMIZER_STEPS,
                "output_dir": (
                    "artifacts/outputs/"
                    + B2_CALIBRATION_RUN_ID
                ),
                "automatically_start": False,
            },
            "production_backend": template_backend,
            "executor": executor,
            "model": {
                "base_path": template_model["base"],
                "base_manifest_path": template_model["base_manifest"],
                "base_manifest_sha256": qualification_model[
                    "artifact_manifest_sha256"
                ],
                "base_weights_manifest_path": template_model[
                    "base_weights_manifest"
                ],
                "base_weights_manifest_sha256": qualification_model[
                    "weights_manifest_sha256"
                ],
                "model_revision": template_backend["model_revision"],
                "tokenizer_revision": template_backend["tokenizer_revision"],
                "dtype": template_backend["dtype"],
                "attention_backend": template_backend["attention_backend"],
            },
            "teacher": {
                "adapter_path": template_model["teacher_adapter"],
                "adapter_sha256": qualification_teacher["adapter_sha256"],
                "adapter_weight_sha256": qualification_teacher[
                    "adapter_weight_sha256"
                ],
                "manifest_path": template_model["teacher_manifest"],
                "manifest_sha256": qualification_teacher["manifest_sha256"],
                "role": "single_frozen_medical_teacher",
                "same_token_scoring": True,
            },
            "data": {
                "protocol_version": template_data["protocol_version"],
                "prompt_manifest_path": template_data["prompt_manifest"],
                "prompt_manifest_sha256": bindings["data_manifest_sha256"],
                "selection_rule": B2_CALIBRATION_SELECTION_RULE,
                "allowed_roles": list(template_data["allowed_roles"]),
                "prompt_only": True,
                "final_labels_allowed": False,
            },
            "protocol": {
                **template_protocol,
                "qualification_protocol_sha256": bindings["protocol_sha256"],
            },
            "generation": {
                "max_new_tokens": selected,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "full_support": True,
                "enable_thinking": False,
                "use_cache": True,
            },
            "qualification": qualification,
            "authorization": {
                "source": "artifact_derived_p4_6_full_qualification",
                "production_sampler_refresh_ready": True,
                "OPD_scoring_backend_ready": True,
                "B2_authorized": True,
                "B2_started": False,
            },
            "isolation": dict(_ISOLATION),
            "execution": {
                "optimizer_steps": B2_CALIBRATION_OPTIMIZER_STEPS,
                "calibration_only": True,
                "automatically_start_b2": False,
                "automatically_run_idt": False,
                "automatically_run_sar": False,
                "automatically_run_ca_opd": False,
                "automatically_run_controller": False,
                "automatically_run_confirmation": False,
                "automatically_run_final": False,
            },
        }
        config_text = yaml.safe_dump(config, sort_keys=True, allow_unicode=False)
        config_sha = _atomic_text(temporary / "config.yaml", config_text)
        card = {
            "schema_id": B2_CALIBRATION_SCHEMA_ID,
            "schema_version": B2_CALIBRATION_SCHEMA_VERSION,
            "run_id": B2_CALIBRATION_RUN_ID,
            "stage": "b2_medical_opd_calibration",
            "status": "authorized_not_started",
            "config_path": "config.yaml",
            "config_sha256": config_sha,
            "production_backend_id": B2_PRODUCTION_BACKEND_ID,
            "executor": executor,
            "selected_response_length": selected,
            "optimizer_steps": B2_CALIBRATION_OPTIMIZER_STEPS,
            "qualification": qualification,
            "automatically_start_b2": False,
            "automatically_run_idt_sar_ca_opd": False,
            "automatically_access_controller_confirmation_final": False,
            "production_sampler_refresh_ready_now": True,
            "OPD_scoring_backend_ready_now": True,
            "B2_authorized_now": True,
            "B2_started": False,
        }
        _validate_calibration_documents(
            config,
            card,
            config_sha256=config_sha,
            executor=executor,
        )
        card_sha = _atomic_json(temporary / "run_card.json", card)
        package_preflight = {
            "schema_id": "ca-opd/b2-medical-opd-calibration-preflight/v1",
            "schema_version": 1,
            "status": "pass",
            "run_id": B2_CALIBRATION_RUN_ID,
            "config_sha256": config_sha,
            "run_card_sha256": card_sha,
            "executor": executor,
            "selected_response_length": selected,
            "optimizer_steps": B2_CALIBRATION_OPTIMIZER_STEPS,
            "qualification_readiness_sha256": qualification["readiness_sha256"],
            "qualification_artifact_index_sha256": qualification[
                "artifact_index_sha256"
            ],
            "checks": {
                "strict_config_schema": True,
                "strict_run_card_schema": True,
                "source_bound_executor": True,
                "qualification_graph_recomputed": True,
                "selected_length_artifact_derived": True,
                "auto_start_disabled": True,
                "isolation_closed": True,
            },
        }
        _validate_package_preflight(
            package_preflight,
            config_sha256=config_sha,
            run_card_sha256=card_sha,
            executor=executor,
            qualification=qualification,
            selected_response_length=selected,
        )
        preflight_sha = _atomic_json(
            temporary / "package_preflight.json", package_preflight
        )
        authorization = {
            "schema_version": SCHEMA_VERSION,
            "authorization_kind": "b2-medical-opd-calibration-v1",
            "qualification_output": str(source),
            "qualification_run_id": bindings["run_id"],
            "qualification_attempt_id": bindings["attempt_id"],
            "calibration_package_dir": str(target.resolve()),
            "evidence_readiness_sha256": sha256_file(
                source / EVIDENCE_READINESS_FILE
            ),
            "evidence_artifact_index_sha256": sha256_file(
                source / EVIDENCE_INDEX_FILE
            ),
            "qualification_config_sha256": bindings["config_sha256"],
            "qualification_run_card_sha256": bindings["run_card_sha256"],
            "calibration_config_sha256": config_sha,
            "calibration_run_card_sha256": card_sha,
            "package_preflight_sha256": preflight_sha,
            "production_backend_id": B2_PRODUCTION_BACKEND_ID,
            "executor_path": executor["path"],
            "executor_symbol": executor["symbol"],
            "executor_source_sha256": executor["source_sha256"],
            "backend_binding_sha256": bindings["backend_binding_sha256"],
            "protocol_sha256": bindings["protocol_sha256"],
            "data_manifest_sha256": bindings["data_manifest_sha256"],
            "git_commit": bindings["git_commit"],
            "authority_v1_sha256": _phase_sha(source, "authority_v1"),
            "authority_v2_sha256": _phase_sha(source, "authority_v2"),
            "two_step_trajectory_sha256": _phase_sha(
                source, "trajectory_step1_manifest"
            ),
            "base_null_sha256": _phase_sha(source, "base_null"),
            "length_smoke_sha256": _phase_sha(source, "length_smoke"),
            "length_decision_sha256": _phase_sha(source, "length_decision"),
            "cleanup_sha256": _phase_sha(source, "cleanup"),
            "selected_response_length": selected,
            "optimizer_steps": B2_CALIBRATION_OPTIMIZER_STEPS,
            "isolation": dict(_ISOLATION),
            "calibration_schema_validated": True,
            "calibration_cpu_dry_run_passed": True,
            "B2_authorized": True,
            "B2_started": False,
        }
        _atomic_json(temporary / "b2_authorization.json", authorization)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        _atomic_bytes(
            source / ROOT_AUTHORIZATION_FILE,
            (target / "b2_authorization.json").read_bytes(),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "config": config,
        "run_card": card,
        "package_preflight": package_preflight,
        "authorization": authorization,
        "B2_started": False,
    }


def materialize_b2_calibration_package(
    qualification_output: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Materialize B2 calibration or durably fail the qualification root."""

    source = Path(qualification_output).resolve()
    try:
        return _materialize_b2_calibration_package_unchecked(source, destination)
    except BaseException as error:
        if (
            (source / EVIDENCE_INDEX_FILE).is_file()
            and (source / EVIDENCE_READINESS_FILE).is_file()
            and not (source / FINAL_READINESS_FILE).exists()
            and not (source / "failure.json").exists()
        ):
            try:
                bindings = _load_bindings_from_phase(source, FULL_PHASES)
                record_failure(
                    source,
                    bindings=bindings,
                    mode="full",
                    reason=(
                        "b2_calibration_materialization_failed:"
                        f"{type(error).__name__}:{error}"
                    ),
                )
            except QualificationArtifactError:
                # Preserve the original materialization exception.  The final
                # readiness recomputation will still fail on the incomplete graph.
                pass
        raise


_AUTHORIZATION_FIELDS = {
    "schema_version",
    "authorization_kind",
    "qualification_output",
    "qualification_run_id",
    "qualification_attempt_id",
    "calibration_package_dir",
    "evidence_readiness_sha256",
    "evidence_artifact_index_sha256",
    "qualification_config_sha256",
    "qualification_run_card_sha256",
    "calibration_config_sha256",
    "calibration_run_card_sha256",
    "package_preflight_sha256",
    "production_backend_id",
    "executor_path",
    "executor_symbol",
    "executor_source_sha256",
    "backend_binding_sha256",
    "protocol_sha256",
    "data_manifest_sha256",
    "git_commit",
    "authority_v1_sha256",
    "authority_v2_sha256",
    "two_step_trajectory_sha256",
    "base_null_sha256",
    "length_smoke_sha256",
    "length_decision_sha256",
    "cleanup_sha256",
    "selected_response_length",
    "optimizer_steps",
    "isolation",
    "calibration_schema_validated",
    "calibration_cpu_dry_run_passed",
    "B2_authorized",
    "B2_started",
}


def _validate_root_authorization_evidence(source: Path) -> dict[str, Any]:
    """Validate root/package authorization against immutable evidence only."""

    root_path = source / ROOT_AUTHORIZATION_FILE
    authorization = _read_json(root_path)
    if set(authorization) != _AUTHORIZATION_FIELDS:
        raise QualificationArtifactError("authorization fields mismatch")
    if (
        authorization["schema_version"] != SCHEMA_VERSION
        or authorization["authorization_kind"]
        != "b2-medical-opd-calibration-v1"
        or authorization["qualification_output"] != str(source.resolve())
        or authorization["production_backend_id"] != B2_PRODUCTION_BACKEND_ID
        or authorization["optimizer_steps"] != B2_CALIBRATION_OPTIMIZER_STEPS
        or authorization["B2_authorized"] is not True
        or authorization["B2_started"] is not False
        or authorization["isolation"] != _ISOLATION
        or authorization["calibration_schema_validated"] is not True
        or authorization["calibration_cpu_dry_run_passed"] is not True
    ):
        raise QualificationArtifactError("authorization state is invalid")
    _validate_current_backend_source_chain(source)
    evidence = _derive_evidence_readiness(source)
    if not evidence.get("ready") or not evidence.get("authorization_eligibility"):
        raise QualificationArtifactError("qualification evidence is no longer ready")
    if (
        authorization["evidence_readiness_sha256"]
        != sha256_file(source / EVIDENCE_READINESS_FILE)
        or authorization["evidence_artifact_index_sha256"]
        != sha256_file(source / EVIDENCE_INDEX_FILE)
    ):
        raise QualificationArtifactError("authorization evidence SHA mismatch")
    if not isinstance(authorization["calibration_package_dir"], str):
        raise QualificationArtifactError("calibration package path is invalid")
    package = Path(authorization["calibration_package_dir"])
    package_authorization_path = package / ROOT_AUTHORIZATION_FILE
    if (
        package_authorization_path.is_symlink()
        or not package_authorization_path.is_file()
        or package_authorization_path.read_bytes() != root_path.read_bytes()
    ):
        raise QualificationArtifactError("root/package authorization mismatch")
    config_path = package / "config.yaml"
    card_path = package / "run_card.json"
    preflight_path = package / "package_preflight.json"
    for package_path in (config_path, card_path, preflight_path):
        if package_path.is_symlink() or not package_path.is_file():
            raise QualificationArtifactError("calibration package file is absent")
    if sha256_file(config_path) != authorization["calibration_config_sha256"]:
        raise QualificationArtifactError("calibration config SHA mismatch")
    if sha256_file(card_path) != authorization["calibration_run_card_sha256"]:
        raise QualificationArtifactError("calibration run-card SHA mismatch")
    if sha256_file(preflight_path) != authorization["package_preflight_sha256"]:
        raise QualificationArtifactError("calibration package preflight SHA mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    card = _read_json(card_path)
    if not isinstance(config, Mapping):
        raise QualificationArtifactError("calibration config is invalid")
    executor = _production_executor_descriptor()
    expected_executor = {
        "path": authorization["executor_path"],
        "symbol": authorization["executor_symbol"],
        "source_sha256": authorization["executor_source_sha256"],
    }
    if executor != expected_executor:
        raise QualificationArtifactError("authorization executor binding mismatch")
    _validate_calibration_documents(
        config,
        card,
        config_sha256=authorization["calibration_config_sha256"],
        executor=executor,
    )
    package_preflight = _read_json(preflight_path)
    _validate_package_preflight(
        package_preflight,
        config_sha256=authorization["calibration_config_sha256"],
        run_card_sha256=authorization["calibration_run_card_sha256"],
        executor=executor,
        qualification=config["qualification"],
        selected_response_length=authorization["selected_response_length"],
    )
    first = _read_json(source / "launch_record.json")
    expected = {
        "qualification_run_id": first["run_id"],
        "qualification_attempt_id": first["attempt_id"],
        "qualification_config_sha256": first["config_sha256"],
        "qualification_run_card_sha256": first["run_card_sha256"],
        "backend_binding_sha256": first["backend_binding_sha256"],
        "protocol_sha256": first["protocol_sha256"],
        "data_manifest_sha256": first["data_manifest_sha256"],
        "git_commit": first["git_commit"],
        "authority_v1_sha256": _phase_sha(source, "authority_v1"),
        "authority_v2_sha256": _phase_sha(source, "authority_v2"),
        "two_step_trajectory_sha256": _phase_sha(
            source, "trajectory_step1_manifest"
        ),
        "base_null_sha256": _phase_sha(source, "base_null"),
        "length_smoke_sha256": _phase_sha(source, "length_smoke"),
        "length_decision_sha256": _phase_sha(source, "length_decision"),
        "cleanup_sha256": _phase_sha(source, "cleanup"),
    }
    for key, value in expected.items():
        if authorization[key] != value:
            raise QualificationArtifactError(
                f"authorization {key} SHA/binding mismatch"
            )
    expected_qualification = {
        "run_id": first["run_id"],
        "attempt_id": first["attempt_id"],
        "readiness_sha256": authorization["evidence_readiness_sha256"],
        "artifact_index_sha256": authorization[
            "evidence_artifact_index_sha256"
        ],
        "qualification_config_sha256": first["config_sha256"],
        "qualification_run_card_sha256": first["run_card_sha256"],
        "backend_binding_sha256": first["backend_binding_sha256"],
        "protocol_sha256": first["protocol_sha256"],
        "prompt_manifest_sha256": first["prompt_manifest_sha256"],
        "probe_spec_sha256": first["probe_spec_sha256"],
        "data_manifest_sha256": first["data_manifest_sha256"],
        "git_commit": first["git_commit"],
        "authority_v1_sha256": expected["authority_v1_sha256"],
        "authority_v2_sha256": expected["authority_v2_sha256"],
        **_v2_transport_binding(source),
        "two_step_trajectory_sha256": expected["two_step_trajectory_sha256"],
        "base_null_sha256": expected["base_null_sha256"],
        "length_smoke_sha256": expected["length_smoke_sha256"],
        "length_decision_sha256": expected["length_decision_sha256"],
        "cleanup_sha256": expected["cleanup_sha256"],
    }
    selected = _read_json(source / "length_decision.json")["payload"][
        "selected_response_length"
    ]
    if config.get("qualification") != expected_qualification or not (
        selected
        == authorization["selected_response_length"]
        == card.get("selected_response_length")
        == config.get("generation", {}).get("max_new_tokens")
    ):
        raise QualificationArtifactError("calibration qualification graph mismatch")
    return authorization


def assert_b2_start_authorized(authorization_path: str | Path) -> dict[str, Any]:
    """Production start gate: recompute all qualification and package bindings."""

    path = Path(authorization_path)
    authorization = _read_json(path)
    source = Path(authorization["qualification_output"]).resolve()
    root_authorization = _validate_root_authorization_evidence(source)
    if authorization != root_authorization:
        raise QualificationArtifactError("root/package authorization mismatch")
    required = _AUTHORIZATION_FIELDS
    if set(authorization) != required:
        raise QualificationArtifactError("authorization fields mismatch")
    if (
        authorization["schema_version"] != SCHEMA_VERSION
        or authorization["authorization_kind"]
        != "b2-medical-opd-calibration-v1"
        or authorization["production_backend_id"] != B2_PRODUCTION_BACKEND_ID
        or authorization["optimizer_steps"] != B2_CALIBRATION_OPTIMIZER_STEPS
        or authorization["B2_authorized"] is not True
        or authorization["B2_started"] is not False
        or authorization["isolation"] != _ISOLATION
        or authorization["calibration_schema_validated"] is not True
        or authorization["calibration_cpu_dry_run_passed"] is not True
    ):
        raise QualificationArtifactError("authorization state is invalid")
    package = path.parent
    config_path = package / "config.yaml"
    card_path = package / "run_card.json"
    preflight_path = package / "package_preflight.json"
    if sha256_file(config_path) != authorization["calibration_config_sha256"]:
        raise QualificationArtifactError("calibration config SHA mismatch")
    if sha256_file(card_path) != authorization["calibration_run_card_sha256"]:
        raise QualificationArtifactError("calibration run-card SHA mismatch")
    if sha256_file(preflight_path) != authorization["package_preflight_sha256"]:
        raise QualificationArtifactError("calibration package preflight SHA mismatch")
    card = _read_json(card_path)
    if card.get("config_sha256") != sha256_file(config_path):
        raise QualificationArtifactError("run-card/config SHA mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise QualificationArtifactError("calibration config is invalid")
    executor = _production_executor_descriptor()
    expected_executor = {
        "path": authorization["executor_path"],
        "symbol": authorization["executor_symbol"],
        "source_sha256": authorization["executor_source_sha256"],
    }
    if executor != expected_executor:
        raise QualificationArtifactError("authorization executor binding mismatch")
    _validate_calibration_documents(
        config,
        card,
        config_sha256=authorization["calibration_config_sha256"],
        executor=executor,
        verify_base_weight_payloads=True,
    )
    package_preflight = _read_json(preflight_path)
    _validate_package_preflight(
        package_preflight,
        config_sha256=authorization["calibration_config_sha256"],
        run_card_sha256=authorization["calibration_run_card_sha256"],
        executor=executor,
        qualification=config["qualification"],
        selected_response_length=authorization["selected_response_length"],
    )

    readiness = derive_qualification_readiness(source, mode="full")
    if not (
        readiness.get("ready")
        and readiness.get("authorization_eligibility")
        and readiness.get("readiness_stage") == "final"
        and readiness.get("OPD_scoring_backend_ready") is True
        and readiness.get("B2_authorized") is True
    ):
        raise QualificationArtifactError("full qualification is no longer ready")
    first = _read_json(source / "launch_record.json")
    expected = {
        "qualification_run_id": first["run_id"],
        "qualification_attempt_id": first["attempt_id"],
        "evidence_readiness_sha256": sha256_file(
            source / EVIDENCE_READINESS_FILE
        ),
        "evidence_artifact_index_sha256": sha256_file(
            source / EVIDENCE_INDEX_FILE
        ),
        "qualification_config_sha256": first["config_sha256"],
        "qualification_run_card_sha256": first["run_card_sha256"],
        "backend_binding_sha256": first["backend_binding_sha256"],
        "protocol_sha256": first["protocol_sha256"],
        "data_manifest_sha256": first["data_manifest_sha256"],
        "git_commit": first["git_commit"],
        "authority_v1_sha256": _phase_sha(source, "authority_v1"),
        "authority_v2_sha256": _phase_sha(source, "authority_v2"),
        "two_step_trajectory_sha256": _phase_sha(
            source, "trajectory_step1_manifest"
        ),
        "base_null_sha256": _phase_sha(source, "base_null"),
        "length_smoke_sha256": _phase_sha(source, "length_smoke"),
        "length_decision_sha256": _phase_sha(source, "length_decision"),
        "cleanup_sha256": _phase_sha(source, "cleanup"),
        "production_backend_id": B2_PRODUCTION_BACKEND_ID,
        "executor_path": executor["path"],
        "executor_symbol": executor["symbol"],
        "executor_source_sha256": executor["source_sha256"],
        "package_preflight_sha256": sha256_file(preflight_path),
    }
    for key, value in expected.items():
        if authorization[key] != value:
            raise QualificationArtifactError(f"authorization {key} SHA/binding mismatch")
    expected_qualification = {
        "run_id": first["run_id"],
        "attempt_id": first["attempt_id"],
        "readiness_sha256": expected["evidence_readiness_sha256"],
        "artifact_index_sha256": expected[
            "evidence_artifact_index_sha256"
        ],
        "qualification_config_sha256": first["config_sha256"],
        "qualification_run_card_sha256": first["run_card_sha256"],
        "backend_binding_sha256": first["backend_binding_sha256"],
        "protocol_sha256": first["protocol_sha256"],
        "prompt_manifest_sha256": first["prompt_manifest_sha256"],
        "probe_spec_sha256": first["probe_spec_sha256"],
        "data_manifest_sha256": first["data_manifest_sha256"],
        "git_commit": first["git_commit"],
        "authority_v1_sha256": expected["authority_v1_sha256"],
        "authority_v2_sha256": expected["authority_v2_sha256"],
        **_v2_transport_binding(source),
        "two_step_trajectory_sha256": expected["two_step_trajectory_sha256"],
        "base_null_sha256": expected["base_null_sha256"],
        "length_smoke_sha256": expected["length_smoke_sha256"],
        "length_decision_sha256": expected["length_decision_sha256"],
        "cleanup_sha256": expected["cleanup_sha256"],
    }
    if config.get("qualification") != expected_qualification:
        raise QualificationArtifactError("calibration qualification graph mismatch")
    selected = _read_json(source / "length_decision.json")["payload"][
        "selected_response_length"
    ]
    if not (
        selected
        == authorization["selected_response_length"]
        == card.get("selected_response_length")
        == config.get("generation", {}).get("max_new_tokens")
        and authorization["optimizer_steps"]
        == card.get("optimizer_steps")
        == config.get("execution", {}).get("optimizer_steps")
        == B2_CALIBRATION_OPTIMIZER_STEPS
    ):
        raise QualificationArtifactError("authorized response length mismatch")
    return authorization


__all__ = [
    "ARTIFACT_PROTOCOL_VERSION",
    "FULL_PHASES",
    "MICRO_PHASES",
    "STATIC_ARTIFACT_FILES",
    "QualificationArtifactError",
    "assert_micro_evidence_prefix_ready",
    "assert_b2_start_authorized",
    "canonical_json_sha256",
    "commit_phase",
    "derive_qualification_readiness",
    "finalize_qualification",
    "initialize_qualification_artifacts",
    "materialize_b2_calibration_package",
    "record_failure",
    "record_failure_cleanup",
    "sha256_file",
    "validate_qualification_static_artifacts",
    "write_terminal_summary_alias",
]
