"""Project the verified P7 runtime into P9 without changing training semantics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2


def p9_extension_discriminator() -> dict[str, Any]:
    return {
        "package_version": "p9_b2_adaptive_dose_v1",
        "single_training_semantic_variable": "accepted_optimizer_commit_dose",
        "source_batch": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
        "resume_step": 120,
        "first_review_step": 200,
        "gray_review_step": 240,
        "frozen_max_step": 300,
        "group_size": 1,
        "learning_rate": 1e-5,
        "per_prompt_gradient_clip_norm": 0.25,
        "response_length": 1024,
        "enable_thinking": False,
        "p8_source_mix_loaded": False,
        "final_authorized": False,
        "confirmation_authorized": False,
    }


def validate_p9_runtime_extension(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = p9_extension_discriminator()
    if value.get("source_batch") != expected["source_batch"]:
        raise P9ProtocolError("P9 runtime action is not exact 2 O1 plus 2 CMB")
    if dict(value) != expected:
        raise P9ProtocolError("P9 runtime dose-only discriminator differs")
    return {"passed": True, "absolute_max_step": 300}


def p9_optimizer_step_limit(config: Mapping[str, Any]) -> int:
    """Authorize step300 without changing the SHA-bound P7/P8 kernel source."""

    value = config.get("p9_adaptive_dose")
    if not isinstance(value, Mapping):
        raise P9ProtocolError("P9 runtime dose-only discriminator is absent")
    semantic = {key: item for key, item in value.items() if key != "schedule_sha256"}
    validate_p9_runtime_extension(semantic)
    schedule_sha = value.get("schedule_sha256")
    if not (
        isinstance(schedule_sha, str)
        and len(schedule_sha) == 64
        and all(character in "0123456789abcdef" for character in schedule_sha)
    ):
        raise P9ProtocolError("P9 runtime schedule SHA is absent")
    return 300


def build_p9_runtime_config(
    p7_package_config: Mapping[str, Any],
    *,
    output: Path,
    schedule_sha256: str,
) -> dict[str, Any]:
    if "p8_formal_b2" in p7_package_config:
        raise P9ProtocolError("P8 formal config cannot enter P9")
    runtime = formal_b2_runtime_config_v2(p7_package_config)
    memory = runtime.get("memory_execution", {})
    bounded = runtime.get("bounded_influence_v2", {})
    if not (
        memory.get("source_batch") == {"medical_opd_o1": 2, "medical_opd_cmb": 2}
        and memory.get("effective_batch_size") == 4
        and memory.get("selected_response_length") == 1024
        and bounded.get("per_prompt_gradient_clip_norm") == 0.25
        and runtime.get("run", {}).get("seed") == 42
        and runtime.get("formal_b2_v2", {}).get("selected_common_learning_rate") == 1e-5
    ):
        raise P9ProtocolError("P7 runtime fields differ before P9 projection")
    result = deepcopy(runtime)
    result["run"]["run_id"] = Path(output).name
    result["run"]["output_dir"] = str(Path(output).resolve())
    result["p9_adaptive_dose"] = p9_extension_discriminator()
    result["p9_adaptive_dose"]["schedule_sha256"] = str(schedule_sha256)
    # Validate the registered fields before adding the schedule identity, which
    # is evidence binding rather than a training-semantic field.
    validate_p9_runtime_extension(
        {key: value for key, value in result["p9_adaptive_dose"].items() if key != "schedule_sha256"}
    )
    return result


__all__ = [
    "build_p9_runtime_config",
    "p9_extension_discriminator",
    "p9_optimizer_step_limit",
    "validate_p9_runtime_extension",
]
