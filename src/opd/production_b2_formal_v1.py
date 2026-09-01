"""CPU-safe primitives and CLI entrypoint for formal B2 v1.

GPU construction is intentionally imported only by the execute path.  Dry-run
and preflight can therefore reject a wrong interpreter or package before any
model/CUDA state exists.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping


PRODUCTION_PYTHON = Path("artifacts/env/bin/python")
PRODUCTION_ENVIRONMENT = PRODUCTION_PYTHON.parent.parent
MILESTONE_STEPS = (30, 60, 90, 120, 150)
PRODUCTION_BACKEND_ID = "custom_transformers_peft_three_policy_v5"
PRODUCTION_REFRESH_MECHANISM = "peft_0_17_1_hotswap_stable_slot"
PRODUCTION_SLOT = "student_active"

FORMAL_MEMORY_EXECUTION_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "artifact_kind": "p5_formal_b2_memory_execution_contract_v1",
    "selected_response_length": 1024,
    "seed": 42,
    "optimizer_steps": 150,
    "stage1_stop_step": 120,
    "source_batch": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
    "schedule_slot_count": 600,
    "stage1_slot_count": 480,
    "physical_microbatch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "target_logit_chunk_size": 128,
    "student_q_backbone_forwards_per_prompt": 1,
    "full_vocabulary_logits_scope": (
        "one_target_position_chunk_per_physical_microbatch_v1"
    ),
    "reduction_contract": (
        "masked_token_mean_per_trajectory_then_group_mean_then_"
        "prompt_mean_then_prompt_batch_mean_v1"
    ),
    "use_cache": False,
    "generation_use_cache": True,
    "gradient_checkpointing": {"enabled": True, "use_reentrant": False},
    "rolling_checkpoint_interval": 10,
    "rolling_checkpoint_keep": 2,
    "milestone_checkpoint_versions": [30, 60, 90, 120, 150],
    "controller_checkpoint_versions": [0, 30, 60, 90, 120, 150],
    "minimum_disk_free_bytes": 10_000_000_000,
    "minimum_canary_headroom_bytes": 1024**3,
    "allocator_policy": "inherit_default_no_override",
    "pytorch_cuda_alloc_conf": None,
    "formal_b2_automatic_start": False,
}


class FormalB2Error(RuntimeError):
    """A formal-B2 gate failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalB2Error(f"formal B2 {label} is absent")
    return value


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise FormalB2Error(f"formal B2 {label} is not a SHA-256")
    return value


def validate_formal_memory_execution_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any drift from the pre-training 120/150 execution envelope."""

    if not isinstance(value, Mapping):
        raise FormalB2Error("formal memory execution contract is not an object")
    for field, expected in FORMAL_MEMORY_EXECUTION_CONTRACT.items():
        if value.get(field) != expected:
            raise FormalB2Error(
                f"formal memory execution {field} differs from the frozen contract"
            )
    if set(value) != set(FORMAL_MEMORY_EXECUTION_CONTRACT):
        raise FormalB2Error("formal memory execution contains unregistered fields")
    return deepcopy(dict(value))


def formal_step_limit(runtime_config: Mapping[str, Any]) -> int:
    """Resolve only the exact formal v1 150-step envelope; never widen calibration."""

    formal = runtime_config.get("formal_b2")
    run = runtime_config.get("run")
    memory = runtime_config.get("memory_execution")
    if not (
        isinstance(formal, Mapping)
        and isinstance(run, Mapping)
        and isinstance(memory, Mapping)
        and formal.get("package_version") == "p5_formal_b2_v1"
        and formal.get("fresh_v0_required") is True
        and formal.get("frozen_max_step") == 150
        and formal.get("stage1_stop_step") == 120
        and run.get("optimizer_steps") == 150
        and run.get("stage1_stop_step") == 120
        and validate_formal_memory_execution_contract(memory)["optimizer_steps"]
        == 150
    ):
        raise FormalB2Error("formal step envelope differs from frozen 120/150")
    return 150


def formal_b2_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the independently authorized formal package onto the proven kernel."""

    if (
        config.get("schema_id") != "ca-opd/formal-b2-medical-opd/v1"
        or config.get("schema_version") != 1
        or config.get("package_version") != "p5_formal_b2_v1"
    ):
        raise FormalB2Error("formal B2 package schema drift")
    run = _mapping(config.get("run"), "run")
    backend = _mapping(config.get("production_backend"), "backend")
    model = _mapping(config.get("model"), "model")
    teacher = _mapping(config.get("teacher"), "teacher")
    protocol = _mapping(config.get("protocol"), "protocol")
    generation = _mapping(config.get("generation"), "generation")
    qualification = _mapping(config.get("qualification"), "qualification")
    initialization = _mapping(
        config.get("student_initialization"), "student initialization"
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    execution = _mapping(config.get("execution"), "execution")
    memory = validate_formal_memory_execution_contract(
        _mapping(config.get("memory_execution"), "memory execution")
    )
    isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if not (
        run.get("seed") == 42
        and run.get("optimizer_steps") == 150
        and run.get("stage1_stop_step") == 120
        and backend.get("backend_id") == PRODUCTION_BACKEND_ID
        and backend.get("refresh_implementation") == PRODUCTION_REFRESH_MECHANISM
        and backend.get("adapter_runtime_slot") == PRODUCTION_SLOT
        and backend.get("dtype") == "bfloat16"
        and backend.get("attention_backend") == "eager"
        and generation
        == {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "full_support": True,
            "enable_thinking": False,
            "use_cache": True,
        }
        and protocol.get("optimizer") == "AdamW"
        and float(protocol.get("learning_rate", -1.0)) == 3e-5
        and int(protocol.get("student_lora_rank", -1)) == 16
        and int(protocol.get("student_lora_alpha", -1)) == 32
        and float(protocol.get("correction_upper_threshold", -1.0)) == 2.0
        and protocol.get("prompt_equal_reduction") is True
        and initialization.get("mode")
        == "fresh_base_plus_fresh_zero_lora_v1"
        and initialization.get("source_adapter_path") is None
        and initialization.get("qualification_v2_usage")
        == "evidence_only_not_student_init"
        and authorization.get("production_sampler_refresh_ready") is True
        and authorization.get("OPD_scoring_backend_ready") is True
        and authorization.get("formal_B2_authorized") is True
        and authorization.get("formal_B2_started") is False
        and execution
        == {
            "optimizer_steps": 150,
            "stage1_stop_step": 120,
            "calibration_only": False,
            "automatically_start_b2": False,
        }
        and config.get("isolation") == isolation
    ):
        raise FormalB2Error("formal B2 production/science/isolation contract drift")
    formula_path = Path(str(protocol.get("three_policy_formula_path", "")))
    if not formula_path.is_absolute():
        formula_path = Path(__file__).resolve().parents[2] / formula_path
    if not (
        formula_path.is_file()
        and not formula_path.is_symlink()
        and _sha256_file(formula_path)
        == _digest(protocol.get("three_policy_formula_sha256"), "formula")
    ):
        raise FormalB2Error("formal B2 formula SHA mismatch")
    for field in ("model_revision", "tokenizer_revision"):
        value = backend.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 40
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise FormalB2Error(f"formal B2 {field} is not immutable")
    _digest(qualification.get("authority_v2_sha256"), "authority v2")
    _digest(qualification.get("v2_tensor_sha256"), "v2 tensor authority")
    output = run.get("output_dir")
    if not isinstance(output, str) or not output:
        raise FormalB2Error("formal B2 output path is absent")
    transformer_generation = {
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "do_sample": True,
        "eos_token_id": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "force_words_ids": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "max_new_tokens": 1024,
        "min_length": 0,
        "min_new_tokens": 0,
        "min_p": None,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "output_logits": True,
        "output_scores": True,
        "pad_token_id": None,
        "renormalize_logits": False,
        "repetition_penalty": 1.0,
        "return_dict_in_generate": True,
        "stop_strings": None,
        "suppress_tokens": None,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "typical_p": 1.0,
        "use_cache": True,
    }
    projected = {
        "schema_id": "ca-opd/p4.6-combined-production-qualification/v1",
        "schema_version": 1,
        "package_version": "p5_formal_b2_v1",
        "run": {
            "run_id": str(run["run_id"]),
            "seed": 42,
            "optimizer_steps": 150,
            "stage1_stop_step": 120,
            "output_dir": output,
        },
        "production_binding": {"backend_id": PRODUCTION_BACKEND_ID},
        "sampler_refresh": {
            "candidate_mechanism": PRODUCTION_REFRESH_MECHANISM,
            "runtime_slot": PRODUCTION_SLOT,
        },
        "model": {
            "id": str(model["base_path"]),
            "revision": str(backend["model_revision"]),
            "tokenizer_revision": str(backend["tokenizer_revision"]),
        },
        "teacher": {"adapter_path": str(teacher["adapter_path"])},
        "validation": {
            "config_path": str(protocol["three_policy_formula_path"]),
            "config_sha256": str(protocol["three_policy_formula_sha256"]),
            "ess_fraction_min": float(protocol["correction_ess_fraction_min"]),
            "cap_fraction_max": float(protocol["correction_cap_fraction_max"]),
            "same_path_max_gap": float(protocol["same_path_max_gap"]),
        },
        "formal_rollout": {
            "backend": "transformers",
            "backend_version": "4.56.2",
            "batch_size": 1,
            "transformers": transformer_generation,
        },
        "micro_replay": {"max_new_tokens": 1024, "prompt_count": 4},
        "reconstruction_telemetry": {"advantage_near_zero_threshold": 1e-6},
        "isolation": isolation,
        "b2_protocol_binding": dict(protocol),
        "student_initialization": dict(initialization),
        "qualification_evidence": dict(
            _mapping(config.get("qualification_evidence"), "qualification evidence")
        ),
        "data": dict(_mapping(config.get("data"), "data")),
        "memory_execution": memory,
        "formal_b2": {
            "package_version": "p5_formal_b2_v1",
            "fresh_v0_required": True,
            "frozen_max_step": 150,
            "stage1_stop_step": 120,
        },
    }
    formal_step_limit(projected)
    return projected


def validate_production_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pinned interpreter metadata before model construction."""

    packages = value.get("packages")
    if not (
        value.get("python_executable") == str(PRODUCTION_PYTHON)
        and value.get("environment_path") == str(PRODUCTION_ENVIRONMENT)
        and value.get("cublas_workspace_config") == ":4096:8"
        and isinstance(packages, Mapping)
        and packages.get("verl") == "0.8.0"
        and packages.get("torch") == "2.8.0+cu128"
        and packages.get("transformers") == "4.56.2"
        and packages.get("peft") == "0.17.1"
    ):
        raise FormalB2Error(
            "formal B2 requires the pinned production Python and dependency set"
        )
    return {"passed": True, **dict(value)}


def checkpoint_retention(
    completed_checkpoint_steps: Iterable[int], *, target_step: int
) -> set[int]:
    """Return milestones plus only the latest two non-milestone rollings."""

    if target_step not in {120, 150}:
        raise FormalB2Error("formal B2 target step must be 120 or 150")
    completed = sorted(
        {
            int(step)
            for step in completed_checkpoint_steps
            if 0 < int(step) <= target_step and int(step) % 10 == 0
        }
    )
    milestones = {
        step for step in MILESTONE_STEPS if step <= target_step and step in completed
    }
    nonmilestones = [step for step in completed if step not in milestones]
    return milestones | set(nonmilestones[-2:])


__all__ = [
    "FORMAL_MEMORY_EXECUTION_CONTRACT",
    "FormalB2Error",
    "MILESTONE_STEPS",
    "PRODUCTION_ENVIRONMENT",
    "PRODUCTION_PYTHON",
    "checkpoint_retention",
    "formal_b2_runtime_config",
    "formal_step_limit",
    "validate_formal_memory_execution_contract",
    "validate_production_environment",
]
