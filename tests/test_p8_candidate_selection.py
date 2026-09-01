from __future__ import annotations

from src.opd.p8_candidate_selection import select_single_variable_candidate


def _evidence(**overrides):
    value = {
        "correctness_bug": False,
        "branch_a": {
            "weak_source_prompt_share": 0.50,
            "weak_to_strong_answer_gap_ratio": 0.80,
            "weak_to_strong_effective_contribution_ratio": 0.55,
            "weak_median_supervision_tokens": 4.0,
            "weak_to_strong_median_effective_gradient_ratio": 0.70,
            "answer_related_completion_contract": True,
        },
        "branch_b": {
            "high_gap_to_overall_cap_trigger_ratio": 1.0,
            "median_triggered_cap_scale": 0.8,
            "global_clip_trigger_rate": 0.0,
            "medical_high_signal_systematically_compressed": False,
        },
        "branch_c": {
            "group2_answer_coverage_improvement": None,
            "group2_repeat_rate_reduction": None,
            "invalid_truncation_acceptable": None,
            "resource_cost_acceptable": None,
        },
        "branch_d": {
            "teacher_signal_normal": True,
            "trajectory_diversity_normal": True,
            "cap_source_bias_absent": True,
            "kl_or_adapter_below_effective_update_threshold": False,
            "safety_headroom": True,
        },
    }
    value.update(overrides)
    return value


def test_a_is_selected_first_when_short_source_also_has_weak_effective_gradient():
    evidence = _evidence()
    evidence["branch_a"].update(
        {
            "weak_median_supervision_tokens": 2.0,
            "weak_to_strong_median_effective_gradient_ratio": 0.332,
        }
    )
    evidence["branch_b"].update(
        {
            "high_gap_to_overall_cap_trigger_ratio": 2.0,
            "median_triggered_cap_scale": 0.2,
            "medical_high_signal_systematically_compressed": True,
        }
    )
    result = select_single_variable_candidate(evidence)
    assert result["selected_branch"] == "A"
    assert result["single_training_semantic_variable"] == "medical_source_mix"
    assert result["candidate"] == {"medical_opd_o1_prompts": 3, "medical_opd_cmb_prompts": 1}
    assert result["initialization"] == "fresh_base_plus_fresh_zero_effect_lora_v0"


def test_short_completion_alone_never_selects_a():
    evidence = _evidence()
    evidence["branch_a"].update(
        {
            "weak_median_supervision_tokens": 2.0,
            "weak_to_strong_median_effective_gradient_ratio": 0.75,
            "weak_to_strong_answer_gap_ratio": 0.75,
        }
    )
    result = select_single_variable_candidate(evidence)
    assert result["selected_branch"] == "E"
    assert "short_completion_alone_is_insufficient" in result["branch_decisions"]["A"]["reasons"]


def test_b_thresholds_are_all_inclusive_and_global_clip_must_be_below_one_percent():
    evidence = _evidence()
    evidence["branch_b"].update(
        {
            "high_gap_to_overall_cap_trigger_ratio": 1.5,
            "median_triggered_cap_scale": 0.499,
            "global_clip_trigger_rate": 0.0099,
            "medical_high_signal_systematically_compressed": True,
        }
    )
    result = select_single_variable_candidate(evidence)
    assert result["selected_branch"] == "B"
    assert result["candidate"] == {"per_prompt_gradient_clip_norm": 0.5}


def test_c_precedes_d_and_requires_all_four_gpu_sampling_conditions():
    evidence = _evidence()
    evidence["branch_c"].update(
        {
            "group2_answer_coverage_improvement": 0.25,
            "group2_repeat_rate_reduction": 0.20,
            "invalid_truncation_acceptable": True,
            "resource_cost_acceptable": True,
        }
    )
    evidence["branch_d"]["kl_or_adapter_below_effective_update_threshold"] = True
    result = select_single_variable_candidate(evidence)
    assert result["selected_branch"] == "C"
    assert result["candidate"] == {"group_size": 2}


def test_e_is_the_only_continuation_branch_and_never_changes_training_semantics():
    result = select_single_variable_candidate(_evidence())
    assert result["selected_branch"] == "E"
    assert result["candidate"] == {"target_step": 300}
    assert result["initialization"] == "resume_verified_historical_b2_step120"
    assert result["training_semantics_changed"] is False


def test_correctness_bug_preempts_scientific_tree_and_requires_fresh_v0():
    result = select_single_variable_candidate(_evidence(correctness_bug=True))
    assert result["selected_branch"] == "correctness_bug"
    assert result["initialization"] == "fresh_base_plus_fresh_zero_effect_lora_v0"
    assert result["historical_same_protocol_eligible"] is False
