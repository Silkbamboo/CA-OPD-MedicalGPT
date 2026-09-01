"""Frozen P8 A→B→C→D→E single-variable decision tree."""

from __future__ import annotations

from typing import Any, Mapping


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def select_single_variable_candidate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if evidence.get("correctness_bug") is True:
        return {
            "selected_branch": "correctness_bug",
            "single_training_semantic_variable": "correctness_repair",
            "candidate": None,
            "initialization": "fresh_base_plus_fresh_zero_effect_lora_v0",
            "training_semantics_changed": True,
            "historical_same_protocol_eligible": False,
            "branch_decisions": {},
        }

    decisions: dict[str, Any] = {}
    a = evidence["branch_a"]
    prompt_share = _number(a.get("weak_source_prompt_share"))
    gap_ratio = _number(a.get("weak_to_strong_answer_gap_ratio"))
    contribution_ratio = _number(a.get("weak_to_strong_effective_contribution_ratio"))
    median_tokens = _number(a.get("weak_median_supervision_tokens"))
    gradient_ratio = _number(a.get("weak_to_strong_median_effective_gradient_ratio"))
    source_large = prompt_share is not None and prompt_share >= 0.25
    direct_weak = bool(
        (gap_ratio is not None and gap_ratio < 0.25)
        or (contribution_ratio is not None and contribution_ratio < 0.25)
    )
    short_and_weak = bool(
        median_tokens is not None
        and median_tokens <= 2.0
        and a.get("answer_related_completion_contract") is True
        and (
            (gap_ratio is not None and gap_ratio <= 0.50)
            or (gradient_ratio is not None and gradient_ratio <= 0.50)
        )
    )
    a_pass = source_large and (direct_weak or short_and_weak)
    reasons: list[str] = []
    if median_tokens is not None and median_tokens <= 2.0 and not short_and_weak:
        reasons.append("short_completion_alone_is_insufficient")
    if not source_large:
        reasons.append("weak_source_below_25_percent_of_medical_prompts")
    if not (direct_weak or short_and_weak):
        reasons.append("answer_related_gap_or_effective_gradient_not_clearly_weak")
    decisions["A"] = {"passed": a_pass, "reasons": reasons}
    if a_pass:
        return {
            "selected_branch": "A",
            "single_training_semantic_variable": "medical_source_mix",
            "candidate": {"medical_opd_o1_prompts": 3, "medical_opd_cmb_prompts": 1},
            "initialization": "fresh_base_plus_fresh_zero_effect_lora_v0",
            "training_semantics_changed": True,
            "historical_same_protocol_eligible": False,
            "branch_decisions": decisions,
        }

    b = evidence["branch_b"]
    trigger_ratio = _number(b.get("high_gap_to_overall_cap_trigger_ratio"))
    median_scale = _number(b.get("median_triggered_cap_scale"))
    global_rate = _number(b.get("global_clip_trigger_rate"))
    b_pass = bool(
        trigger_ratio is not None
        and trigger_ratio >= 1.5
        and median_scale is not None
        and median_scale < 0.5
        and global_rate is not None
        and global_rate < 0.01
        and b.get("medical_high_signal_systematically_compressed") is True
    )
    decisions["B"] = {"passed": b_pass}
    if b_pass:
        return {
            "selected_branch": "B",
            "single_training_semantic_variable": "per_prompt_trust_cap",
            "candidate": {"per_prompt_gradient_clip_norm": 0.5},
            "initialization": "fresh_base_plus_fresh_zero_effect_lora_v0",
            "training_semantics_changed": True,
            "historical_same_protocol_eligible": False,
            "branch_decisions": decisions,
        }

    c = evidence["branch_c"]
    coverage = _number(c.get("group2_answer_coverage_improvement"))
    repeat = _number(c.get("group2_repeat_rate_reduction"))
    c_pass = bool(
        coverage is not None
        and coverage >= 0.25
        and repeat is not None
        and repeat >= 0.20
        and c.get("invalid_truncation_acceptable") is True
        and c.get("resource_cost_acceptable") is True
    )
    decisions["C"] = {"passed": c_pass, "status": "not_run" if coverage is None else "observed"}
    if c_pass:
        return {
            "selected_branch": "C",
            "single_training_semantic_variable": "group_size",
            "candidate": {"group_size": 2},
            "initialization": "fresh_base_plus_fresh_zero_effect_lora_v0",
            "training_semantics_changed": True,
            "historical_same_protocol_eligible": False,
            "branch_decisions": decisions,
        }

    d = evidence["branch_d"]
    d_pass = all(
        d.get(name) is True
        for name in (
            "teacher_signal_normal",
            "trajectory_diversity_normal",
            "cap_source_bias_absent",
            "kl_or_adapter_below_effective_update_threshold",
            "safety_headroom",
        )
    )
    decisions["D"] = {"passed": d_pass}
    if d_pass:
        return {
            "selected_branch": "D",
            "single_training_semantic_variable": "learning_rate",
            "candidate": {"learning_rate": 2e-5},
            "initialization": "fresh_base_plus_fresh_zero_effect_lora_v0",
            "training_semantics_changed": True,
            "historical_same_protocol_eligible": False,
            "branch_decisions": decisions,
        }

    decisions["E"] = {"passed": True, "reason": "no_identifiable_single_variable"}
    return {
        "selected_branch": "E",
        "single_training_semantic_variable": "training_dose_only",
        "candidate": {"target_step": 300},
        "initialization": "resume_verified_historical_b2_step120",
        "training_semantics_changed": False,
        "historical_same_protocol_eligible": True,
        "branch_decisions": decisions,
    }


__all__ = ["select_single_variable_candidate"]
