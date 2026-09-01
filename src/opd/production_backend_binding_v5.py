"""Fail-closed production backend binding and B2 start gate for P4.5."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PRODUCTION_BACKEND_ID = "custom_transformers_peft_three_policy_v5"
PRODUCTION_REFRESH_IMPLEMENTATION = "peft_0_17_1_hotswap_stable_slot"
REPO_ROOT = Path(__file__).resolve().parents[2]

_BACKEND_FIELDS = (
    "backend_id",
    "trainer_backend",
    "rollout_backend",
    "generation_backend",
    "scoring_backend",
    "sampler_backend",
    "correction_backend",
    "refresh_implementation",
    "adapter_load_implementation",
    "fixed_action_scoring_implementation",
    "checkpoint_writer",
    "model_revision",
    "tokenizer_revision",
    "dtype",
    "attention_backend",
    "generation_cache",
    "fixed_action_cache",
    "sampler_identity_guard",
    "adapter_runtime_slot",
    "vllm_used",
    "verl_usage",
)
_ISOLATION_FIELDS = (
    "final_access",
    "controller_access",
    "confirmation_access",
    "label_access",
)
_READINESS_GATES = {
    "production_sampler_refresh_micro_v5": (
        "production_backend_bound",
        "authoritative_sha_verified",
        "runtime_sha_match",
        "per_tensor_match",
        "same_path_gap_passed",
        "normal_request_passed",
        "stale_request_rejected",
        "v1_reconstruction_evidence_complete",
        "sampler_v0_guard_complete",
        "artifacts_complete",
        "cleanup_complete",
        "isolation_closed",
    ),
    "production_bound_two_step_v5": (
        "production_backend_bound",
        "step0_rollout",
        "v1_update",
        "refresh_verified",
        "step1_used_v1",
        "base_null_passed",
        "artifacts_complete",
        "cleanup_complete",
        "isolation_closed",
    ),
}
_B2_RUN_CARD_FIELDS = {
    "schema_version",
    "run_id",
    "stage",
    "status",
    "config_path",
    "config_sha256",
    "production_backend",
    "required_readiness",
    "automatically_start_b2",
    "automatically_run_idt_sar_ca_opd",
    "automatically_access_controller_confirmation_final",
    "gpu_execution_in_p4_5",
    "B2_authorized_now",
    "OPD_scoring_backend_ready_now",
}
_B2_CONFIG_FIELDS = {
    "schema_version",
    "run",
    "production_backend",
    "call_chain",
    "protocol_binding",
    "model_paths",
    "data",
    "readiness",
    "isolation",
    "historical_backend_disposition",
}
_B2_RUN_FIELDS = {
    "run_id", "baseline_id", "purpose", "seed", "status", "automatically_start"
}
_B2_READINESS_FIELDS = {
    "production_sampler_refresh_ready",
    "production_two_step_ready",
    "OPD_scoring_backend_ready",
    "B2_authorized",
    "required_before_start",
}
_B2_PROTOCOL_FIELDS = {
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
}
_B2_MODEL_PATH_FIELDS = {
    "base", "base_manifest", "teacher_adapter", "teacher_manifest"
}
_P4_6_B2_RUN_ID = "qwen3-4b-b2-medical-opd-custom-v5-p4-6-seed42"
_P4_6_B2_MODEL_PATH_FIELDS = _B2_MODEL_PATH_FIELDS | {"base_weights_manifest"}
_B2_DATA_FIELDS = {
    "protocol_version",
    "prompt_manifest",
    "allowed_roles",
    "final_labels_allowed",
    "response_length_status",
}
_B2_HISTORY_FIELDS = {
    "qwen3_1_7b_verl_vllm_config",
    "p4_4_transformers_delete_load_refresh",
    "transformers_result_may_be_extrapolated_to_vllm",
}


class ProductionBackendBindingError(RuntimeError):
    """The selected B2 stack is not bound to immutable, executable sources."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _mapping_file(path: Path) -> Mapping[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ProductionBackendBindingError(f"cannot read binding file {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ProductionBackendBindingError(f"binding file is not a mapping: {path}")
    return value


def _require_exact_fields(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionBackendBindingError(f"{label} is absent or not a mapping")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ProductionBackendBindingError(f"{label} fields are missing: {missing}")
    if unknown:
        raise ProductionBackendBindingError(f"{label} has unknown fields: {unknown}")
    return value


def load_production_run_card_v5(path: str | Path) -> dict[str, Any]:
    """Load only the frozen B2-v5 production card with strict field dispatch.

    This is intentionally separate from ``src.utils.run_cards``, whose schema
    is the legacy prepared-run family.  A P4.5 micro or P4.6 qualification card
    cannot be silently interpreted as a B2 production card merely because it
    contains an integer schema version.
    """

    card = dict(_require_exact_fields(_mapping_file(Path(path)), _B2_RUN_CARD_FIELDS, label="B2 run-card"))
    if card["schema_version"] != 5:
        raise ProductionBackendBindingError("B2 run-card schema version is invalid")
    if card["stage"] != "b2_medical_opd_custom_v5":
        raise ProductionBackendBindingError("B2 run-card stage is invalid")
    readiness_by_status = {
        "blocked_pending_gpu_refresh_micro_and_two_step": [
            "production_sampler_refresh_micro_v5",
            "production_bound_two_step_v5",
        ],
        "blocked_pending_combined_production_qualification_v6": [
            "combined_production_qualification_v6",
        ],
    }
    expected_readiness = readiness_by_status.get(card["status"])
    if expected_readiness is None:
        raise ProductionBackendBindingError("B2 run-card status is not fail-closed")
    if card["required_readiness"] != expected_readiness:
        raise ProductionBackendBindingError("B2 run-card required readiness drifted")
    for field in (
        "automatically_start_b2",
        "automatically_run_idt_sar_ca_opd",
        "automatically_access_controller_confirmation_final",
        "B2_authorized_now",
        "OPD_scoring_backend_ready_now",
    ):
        if card[field] is not False:
            raise ProductionBackendBindingError(f"B2 run-card is not fail-closed: {field}")
    _validate_backend(card["production_backend"])
    return card


def _validate_production_config_v5(value: Mapping[str, Any]) -> None:
    _require_exact_fields(value, _B2_CONFIG_FIELDS, label="B2 config")
    run = _require_exact_fields(value["run"], _B2_RUN_FIELDS, label="B2 config run")
    _require_exact_fields(value["readiness"], _B2_READINESS_FIELDS, label="B2 config readiness")
    _require_exact_fields(value["isolation"], set(_ISOLATION_FIELDS), label="B2 config isolation")
    _require_exact_fields(value["protocol_binding"], _B2_PROTOCOL_FIELDS, label="B2 config protocol")
    model_fields = (
        _P4_6_B2_MODEL_PATH_FIELDS
        if run.get("run_id") == _P4_6_B2_RUN_ID
        else _B2_MODEL_PATH_FIELDS
    )
    _require_exact_fields(value["model_paths"], model_fields, label="B2 config model paths")
    _require_exact_fields(value["data"], _B2_DATA_FIELDS, label="B2 config data")
    _require_exact_fields(
        value["historical_backend_disposition"], _B2_HISTORY_FIELDS,
        label="B2 config historical backend disposition",
    )


def _validate_backend(value: Any) -> dict[str, Any]:
    value = _require_exact_fields(value, set(_BACKEND_FIELDS), label="production backend")
    result = {field: value[field] for field in _BACKEND_FIELDS}
    if result["backend_id"] != PRODUCTION_BACKEND_ID:
        raise ProductionBackendBindingError("production backend id is not the frozen P4.5 backend")
    if result["refresh_implementation"] != PRODUCTION_REFRESH_IMPLEMENTATION:
        raise ProductionBackendBindingError("production refresh implementation is not frozen")
    if result["adapter_runtime_slot"] != "student_active":
        raise ProductionBackendBindingError("production adapter slot is not stable")
    if result["vllm_used"] is not False:
        raise ProductionBackendBindingError(
            "unverified vLLM backend cannot be extrapolated from Transformers evidence"
        )
    if result["verl_usage"] != "pinned_token_correction_helper_only":
        raise ProductionBackendBindingError("veRL usage exceeds the frozen production evidence")
    if result["generation_cache"] is not True or result["fixed_action_cache"] is not False:
        raise ProductionBackendBindingError("cache semantics differ from the frozen production path")
    for field in _BACKEND_FIELDS:
        if isinstance(result[field], str) and not result[field].strip():
            raise ProductionBackendBindingError(f"production backend field is empty: {field}")
    for field in ("model_revision", "tokenizer_revision"):
        revision = result[field]
        if not (
            isinstance(revision, str)
            and len(revision) == 40
            and all(character in "0123456789abcdef" for character in revision)
        ):
            raise ProductionBackendBindingError(f"{field} is not an immutable 40-hex revision")
    return result


def _validate_call_chain(value: Any, root: Path) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ProductionBackendBindingError("production call-chain is empty")
    result: list[dict[str, str]] = []
    stages: set[str] = set()
    for item in value:
        item = _require_exact_fields(item, {"stage", "path", "symbol"}, label="production call-chain entry")
        stage, raw_path, symbol = item.get("stage"), item.get("path"), item.get("symbol")
        if not all(isinstance(part, str) and part.strip() for part in (stage, raw_path, symbol)):
            raise ProductionBackendBindingError("production call-chain fields are invalid")
        lowered = f"{stage} {raw_path} {symbol}".lower()
        if any(marker in lowered for marker in ("todo", "placeholder", "mock")):
            raise ProductionBackendBindingError("production call-chain contains a TODO/mock/placeholder")
        if stage in stages:
            raise ProductionBackendBindingError(f"production call-chain stage is duplicated: {stage}")
        source = _resolve(raw_path, root)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as error:
            raise ProductionBackendBindingError(
                f"production call-chain source cannot be opened: {source}: {error}"
            ) from error
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if symbol not in definitions:
            raise ProductionBackendBindingError(
                f"production call-chain symbol {symbol!r} is absent from {source}"
            )
        stages.add(stage)
        result.append(
            {
                "stage": stage,
                "path": str(source),
                "path_sha256": sha256_file(source),
                "symbol": symbol,
            }
        )
    return result


def verify_b2_backend_binding(
    config_path: str | Path,
    run_card_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Open and recompute the B2 config/card and every bound source path."""

    root = Path(repo_root)
    config_file = _resolve(config_path, root).resolve()
    card_file = _resolve(run_card_path, root).resolve()
    config = _mapping_file(config_file)
    _validate_production_config_v5(config)
    card = load_production_run_card_v5(card_file)
    config_sha = sha256_file(config_file)
    card_sha = sha256_file(card_file)
    declared_config = _resolve(str(card.get("config_path", "")), root).resolve()
    if declared_config != config_file or card.get("config_sha256") != config_sha:
        raise ProductionBackendBindingError("run card config SHA/path does not match recomputed config SHA")
    if config.get("schema_version") != 5 or card.get("schema_version") != 5:
        raise ProductionBackendBindingError("P4.5 binding schema version is invalid")
    run = config.get("run")
    if not isinstance(run, Mapping) or run.get("run_id") != card.get("run_id"):
        raise ProductionBackendBindingError("config/run-card run id mismatch")
    config_backend = _validate_backend(config.get("production_backend"))
    card_backend = _validate_backend(card.get("production_backend"))
    if config_backend != card_backend:
        raise ProductionBackendBindingError("config/run-card production backend mismatch")
    isolation = config.get("isolation")
    if not isinstance(isolation, Mapping) or any(isolation.get(name) is not False for name in _ISOLATION_FIELDS):
        raise ProductionBackendBindingError("B2 production isolation is not fail-closed")
    chain = _validate_call_chain(config.get("call_chain"), root)
    readiness = config.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ProductionBackendBindingError("B2 readiness declaration is absent")
    if (
        run.get("status") != card.get("status")
        or readiness.get("required_before_start")
        != card.get("required_readiness")
    ):
        raise ProductionBackendBindingError(
            "B2 config/run-card fail-closed readiness mismatch"
        )
    production_ready = bool(
        readiness.get("production_sampler_refresh_ready") is True
        and readiness.get("production_two_step_ready") is True
        and readiness.get("OPD_scoring_backend_ready") is True
        and readiness.get("B2_authorized") is True
    )
    return {
        "schema_version": 5,
        "binding_verified": True,
        "b2_config_path": str(config_file),
        "config_sha256": config_sha,
        "b2_run_card_path": str(card_file),
        "run_card_sha256": card_sha,
        "run_id": run["run_id"],
        "production_backend": config_backend,
        "call_chain": chain,
        "call_chain_complete": True,
        "isolation": {name: False for name in _ISOLATION_FIELDS},
        "implementation_has_todo_mock_or_placeholder": False,
        "production_ready": production_ready,
        "blocking_reason": None if production_ready else "gpu_refresh_micro_and_two_step_not_passed",
    }


def _validate_readiness(
    path: str | Path,
    *,
    kind: str,
    config_sha256: str,
    run_card_sha256: str,
) -> dict[str, Any]:
    value = dict(_mapping_file(Path(path)))
    if value.get("readiness_kind") != kind:
        raise ProductionBackendBindingError(f"{kind} readiness kind mismatch")
    if value.get("production_backend_id") != PRODUCTION_BACKEND_ID:
        raise ProductionBackendBindingError(f"{kind} backend mismatch")
    if (
        value.get("b2_config_sha256") != config_sha256
        or value.get("b2_run_card_sha256") != run_card_sha256
    ):
        raise ProductionBackendBindingError(f"{kind} B2 binding SHA mismatch")
    if any(value.get(name) is not False for name in _ISOLATION_FIELDS):
        raise ProductionBackendBindingError(f"{kind} isolation gate failed")
    required = _READINESS_GATES[kind]
    gates = value.get("gates")
    passed = bool(
        value.get("ready") is True
        and value.get("gate_result") == "pass"
        and value.get("failure_reason") is None
        and isinstance(gates, Mapping)
        and all(gates.get(name) is True for name in required)
    )
    if not passed:
        label = "micro-smoke" if kind == "production_sampler_refresh_micro_v5" else "two-step"
        raise ProductionBackendBindingError(f"{label} readiness did not pass all recomputed gates")
    return value


def assert_b2_start_authorized(
    config_path: str | Path,
    run_card_path: str | Path,
    micro_readiness_path: str | Path,
    two_step_readiness_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Reject the legacy boolean-only B2 gate.

    P4.5 demonstrated that two caller-authored readiness documents do not bind
    the phase artifacts, metrics, cleanup, or deterministic probe.  Keeping the
    old signature makes stale callers fail explicitly while P4.6 installs the
    only supported start gate: a reopened v6 full artifact graph and final
    ``b2_authorization.json``.
    """

    del (
        config_path,
        run_card_path,
        micro_readiness_path,
        two_step_readiness_path,
        repo_root,
    )
    raise ProductionBackendBindingError(
        "legacy B2 readiness booleans are disabled; require the v6 full artifact graph"
    )
