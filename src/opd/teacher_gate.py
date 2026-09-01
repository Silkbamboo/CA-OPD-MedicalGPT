"""CPU-only gate between Controller v2 evidence and any formal OPD launch."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.eval.controller_v2 import protocol_component_hashes
from src.eval.direct_logit_scorer import DIRECT_LOGIT_BACKEND


PROTOCOL_VERSION = "controller_protocol_v2"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class TeacherGateError(RuntimeError):
    """Raised when immutable Teacher evidence is missing or not ready."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher_gate_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "protocol_version", "protocol_sha256", "choice_backend", "status", "teacher_artifact_valid",
        "teacher_knowledge_ready", "teacher_generation_contract_ready", "current_state",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise TeacherGateError("Teacher gate config schema drift")
    if value.get("protocol_version") != PROTOCOL_VERSION or value.get("status") != "frozen_before_gpu_results":
        raise TeacherGateError("Teacher gate protocol/status is not frozen")
    if value.get("choice_backend") != DIRECT_LOGIT_BACKEND:
        raise TeacherGateError("Teacher gate choice backend is not Transformers direct logits")
    if value.get("protocol_sha256") != protocol_component_hashes()["protocol_sha256"]:
        raise TeacherGateError("Teacher gate protocol SHA is not the trusted frozen implementation")
    artifact = value["teacher_artifact_valid"]
    manifest_path = Path(str(artifact.get("manifest_path", "")))
    if not manifest_path.is_file() or _sha256(manifest_path) != artifact.get("manifest_sha256"):
        raise TeacherGateError("Medical LoRA artifact manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("verification_status") != "files_and_combined_sha_verified"
        or manifest.get("model_revision") != artifact.get("base_revision")
        or manifest.get("adapter_sha256") != artifact.get("adapter_sha256")
    ):
        raise TeacherGateError("Medical LoRA artifact identity/verification mismatch")
    adapter_dir = Path(str(manifest.get("adapter_path", "")))
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        file_path = adapter_dir / name
        if not file_path.is_file():
            raise TeacherGateError(f"Medical LoRA file is missing: {name}")
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    if digest.hexdigest() != artifact.get("adapter_sha256"):
        raise TeacherGateError("Medical LoRA ordered aggregate SHA mismatch")
    return value


def evaluate_opd_static_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    state = config["current_state"]
    return {
        "status": "ready_pending_teacher_gate",
        "teacher_artifact_valid": state["artifact_valid"] is True,
        "teacher_knowledge_ready": state["knowledge_ready"],
        "teacher_generation_contract_ready": state["generation_contract_ready"],
        "gpu_runtime_verified": False,
        "formal_opd_authorized": False,
    }


def assert_teacher_ready_for_opd(
    config: Mapping[str, Any],
    readiness_path: str | Path | None,
    *,
    expected_sha256: str | None = None,
    controller_artifact_manifest_path: str | Path | None = None,
    controller_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed until a Controller v2 result proves the frozen knowledge gate."""

    if readiness_path is None:
        raise TeacherGateError("teacher knowledge readiness is pending Controller v2 GPU evidence")
    path = Path(readiness_path).resolve()
    controller_path = Path(controller_artifact_manifest_path or "").resolve()
    if (
        path.name != "teacher_readiness.json"
        or controller_path.name != "artifact_manifest.json"
        or path.parent != controller_path.parent
        or path != (path.parent / "teacher_readiness.json").resolve()
        or controller_path != (path.parent / "artifact_manifest.json").resolve()
    ):
        raise TeacherGateError("Teacher evidence must use canonical same-run artifact paths")
    if (
        not expected_sha256
        or _HEX64.fullmatch(expected_sha256) is None
        or _HEX64.fullmatch(str(controller_artifact_manifest_sha256 or "")) is None
    ):
        raise TeacherGateError("teacher/controller artifact SHA is missing")
    try:
        from src.eval.controller_v2_runtime import validate_standard_run_artifacts

        validated = validate_standard_run_artifacts(path.parent)
    except (OSError, ValueError, RuntimeError) as error:
        raise TeacherGateError(f"Controller v2 artifact inventory is invalid: {error}") from error
    if _sha256(path) != expected_sha256:
        raise TeacherGateError("teacher readiness artifact SHA mismatch")
    if _sha256(controller_path) != controller_artifact_manifest_sha256:
        raise TeacherGateError("Controller v2 artifact manifest SHA mismatch")
    payload = validated["readiness_payload"]
    controller = validated["manifest_payload"]
    recomputed = validated["recomputed_readiness"]
    if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("final_authorized") is not False:
        raise TeacherGateError("teacher readiness protocol/final state is invalid")
    if payload.get("choice_backend") != DIRECT_LOGIT_BACKEND:
        raise TeacherGateError("teacher readiness choice backend is not authoritative")
    trusted_protocol_sha = config["protocol_sha256"]
    if payload.get("protocol_sha256") != trusted_protocol_sha:
        raise TeacherGateError("teacher readiness protocol SHA is not trusted")
    expected_lora = config["teacher_artifact_valid"]["adapter_sha256"]
    if payload.get("medical_lora_sha256") != expected_lora:
        raise TeacherGateError("teacher readiness Medical LoRA identity mismatch")
    if recomputed.get("teacher_artifact_valid") is not True:
        raise TeacherGateError("teacher artifact is not valid")
    if recomputed.get("teacher_knowledge_ready") is not True:
        raise TeacherGateError("teacher knowledge gate is not ready")
    if payload.get("controller_artifact_manifest_sha256") != controller_artifact_manifest_sha256:
        raise TeacherGateError("teacher readiness is not bound to the Controller v2 artifact manifest")
    if (
        controller.get("protocol_version") != PROTOCOL_VERSION
        or controller.get("protocol_sha256") != trusted_protocol_sha
        or controller.get("choice_backend") != DIRECT_LOGIT_BACKEND
        or controller.get("medical_lora_sha256") != expected_lora
        or controller.get("final_authorized") is not False
    ):
        raise TeacherGateError("Controller v2 artifact identity/final state is invalid")
    # Generation compliance is disclosed but does not block trajectory-logprob OPD.
    return {
        "status": "PASS",
        "teacher_artifact_valid": True,
        "teacher_knowledge_ready": True,
        "teacher_generation_contract_ready": recomputed.get("teacher_generation_contract_ready"),
        "formal_opd_authorized": True,
    }
