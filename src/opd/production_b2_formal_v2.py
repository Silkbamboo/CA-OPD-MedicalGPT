"""Formal B2 v2 configuration projected onto the proven production kernel."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.opd.production_b2_formal_v1 import FormalB2Error, formal_b2_runtime_config


class FormalB2V2Error(FormalB2Error):
    """Formal B2 v2 package or qualification decision differs."""


QUALIFIED_COMMON_LEARNING_RATES = frozenset({3.0e-5, 1.0e-5})


def validate_bounded_formula_v2(
    formula_path: Path, *, selected_learning_rate: float
) -> dict[str, Any]:
    """Validate every formula section consumed by the production GPU kernel."""

    try:
        value = yaml.safe_load(Path(formula_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise FormalB2V2Error("Formal B2 v2 bounded formula is unreadable") from error
    if not isinstance(value, Mapping):
        raise FormalB2V2Error("Formal B2 v2 bounded formula is not a mapping")
    for section in ("algorithm", "optimizer", "calibration_gates"):
        if not isinstance(value.get(section), Mapping):
            raise FormalB2V2Error(
                f"Formal B2 v2 bounded formula lacks {section}"
            )
    algorithm = value["algorithm"]
    optimizer = value["optimizer"]
    gates = value["calibration_gates"]
    try:
        passed = bool(
            float(algorithm["beta"]) == 1.0
            and float(algorithm["clip_low"]) == 0.2
            and float(algorithm["clip_high"]) == 0.28
            and optimizer["type"] == "AdamW"
            and float(optimizer["learning_rate"]) == float(selected_learning_rate)
            and float(optimizer["global_gradient_clip_norm"]) == 1.0
            and float(optimizer["per_prompt_gradient_clip_norm"]) == 0.25
            and int(optimizer["lora_rank"]) == 16
            and int(optimizer["lora_alpha"]) == 32
            and optimizer["target_modules"] == "all-linear"
            and float(gates["current_pre_old_actor_max_abs"]) == 1.0e-4
            and float(gates["ess_fraction_min"]) == 0.80
            and float(gates["cap_fraction_max"]) == 0.05
            and gates["final_access"] is False
            and gates["controller_access"] is False
            and gates["confirmation_access"] is False
            and gates["label_access"] is False
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FormalB2V2Error(
            "Formal B2 v2 bounded formula kernel fields are invalid"
        ) from error
    if not passed:
        raise FormalB2V2Error("Formal B2 v2 bounded formula kernel fields differ")
    return deepcopy(dict(value))


def formal_b2_runtime_config_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    if not (
        config.get("schema_id") == "ca-opd/formal-b2-medical-opd/v2"
        and config.get("schema_version") == 2
        and config.get("package_version") == "p5_1_formal_b2_v2"
    ):
        raise FormalB2V2Error("Formal B2 v2 package schema differs")
    protocol = config.get("protocol")
    ratio = config.get("ratio_health_v2")
    if not isinstance(protocol, Mapping) or not isinstance(ratio, Mapping):
        raise FormalB2V2Error("Formal B2 v2 protocol binding is absent")
    selected_lr = ratio.get("selected_common_learning_rate")
    try:
        selected_lr = float(selected_lr)
        protocol_lr = float(protocol.get("learning_rate"))
    except (TypeError, ValueError) as error:
        raise FormalB2V2Error("Formal B2 v2 learning rate is absent") from error
    if selected_lr not in QUALIFIED_COMMON_LEARNING_RATES or protocol_lr != selected_lr:
        raise FormalB2V2Error(
            "Formal B2 v2 learning rate is not the fixed-token qualified common value"
        )
    thresholds = ratio.get("thresholds")
    if not (
        isinstance(thresholds, Mapping)
        and thresholds.get("schema_version") == 2
        and thresholds.get("written_before_new_gpu_results") is True
    ):
        raise FormalB2V2Error("Ratio Health Protocol v2 thresholds are not frozen")
    formula_path = Path(str(protocol.get("three_policy_formula_path", "")))
    if not formula_path.is_absolute():
        formula_path = Path(__file__).resolve().parents[2] / formula_path
    try:
        formula = yaml.safe_load(formula_path.read_text(encoding="utf-8"))
        formula_lr = float(formula["optimizer"]["learning_rate"])
    except (OSError, UnicodeError, TypeError, KeyError, ValueError, yaml.YAMLError) as error:
        raise FormalB2V2Error("Formal B2 v2 formula learning rate is invalid") from error
    if formula_lr != selected_lr:
        raise FormalB2V2Error("Formal B2 v2 formula/common learning rate differs")
    bounded = config.get("bounded_influence_v2")
    diagnostic_only = ratio.get("diagnostic_only") is True
    if bounded is None and not diagnostic_only:
        raise FormalB2V2Error("Formal B2 v2 common bounded influence is absent")
    if bounded is not None:
        formula = validate_bounded_formula_v2(
            formula_path, selected_learning_rate=selected_lr
        )
        try:
            bounded_ok = bool(
                isinstance(bounded, Mapping)
                and bounded.get("enabled") is True
                and bounded.get("mode") == "per_prompt_gradient_clipping"
                and float(bounded.get("per_prompt_gradient_clip_norm")) == 0.25
                and float(bounded.get("global_gradient_clip_norm")) == 1.0
                and int(bounded.get("effective_batch_size")) == 4
                and set(bounded.get("applies_to", [])) == {"B2", "IDT", "CA-OPD"}
                and float(formula["optimizer"]["per_prompt_gradient_clip_norm"])
                == 0.25
            )
        except (TypeError, ValueError, KeyError) as error:
            raise FormalB2V2Error(
                "Formal B2 v2 bounded influence formula is invalid"
            ) from error
        if not bounded_ok:
            raise FormalB2V2Error("Formal B2 v2 common bounded influence differs")

    # Reuse the previously qualified shape/identity projection.  The v1
    # validator's 3e-5 literal is replaced only after the v2 package proves the
    # selected LR is one of the two preregistered fixed-token outcomes and the
    # versioned formula file binds the same value.
    compatibility = deepcopy(dict(config))
    compatibility["schema_id"] = "ca-opd/formal-b2-medical-opd/v1"
    compatibility["schema_version"] = 1
    compatibility["package_version"] = "p5_formal_b2_v1"
    compatibility["protocol"]["learning_rate"] = 3.0e-5
    runtime = formal_b2_runtime_config(compatibility)
    runtime["b2_protocol_binding"]["learning_rate"] = selected_lr
    runtime["ratio_health_v2"] = deepcopy(dict(ratio))
    candidate_acceptance = config.get("candidate_acceptance_v2_1")
    if candidate_acceptance is not None:
        if not isinstance(candidate_acceptance, Mapping):
            raise FormalB2V2Error("candidate acceptance v2.1 is not a mapping")
        runtime["candidate_acceptance_v2_1"] = deepcopy(
            dict(candidate_acceptance)
        )
    if bounded is not None:
        runtime["bounded_influence_v2"] = deepcopy(dict(bounded))
    runtime["formal_b2_v2"] = {
        "package_version": "p5_1_formal_b2_v2",
        "fresh_v0_required": True,
        "accepted_optimizer_commits_target": 120,
        "rejected_attempts_count_as_steps": False,
        "selected_common_learning_rate": selected_lr,
        "training_backend": "custom_transformers_peft_three_policy_loop",
        "final_authorized": False,
    }
    return runtime


__all__ = [
    "FormalB2V2Error",
    "QUALIFIED_COMMON_LEARNING_RATES",
    "formal_b2_runtime_config_v2",
    "validate_bounded_formula_v2",
]
