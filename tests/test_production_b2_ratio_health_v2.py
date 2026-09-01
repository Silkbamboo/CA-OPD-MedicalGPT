from __future__ import annotations

import copy
import math

import pytest
import torch

from src.opd.production_b2_ratio_contract_v2 import (
    RatioContractV2Error,
    RatioPoolBindingV2,
    compute_ratio_evidence_v2,
    validate_ratio_evidence_v2,
)
from src.opd.production_b2_ratio_health_v2 import (
    RatioHealthV2Error,
    evaluate_preupdate_backend_health_v2,
    evaluate_ratio_health_v2,
)


def _fixture(*, post_outlier: float = math.log(1.2), token_count: int = 64):
    old = torch.linspace(-4.0, -0.5, token_count).reshape(2, -1)
    q_pre = old.clone()
    mu = old - 0.05
    q_post = old + 0.02
    q_post[0, 0] = old[0, 0] + post_outlier
    mask = torch.ones_like(old, dtype=torch.bool)
    response_ids = torch.arange(token_count).reshape_as(old) + 100
    input_ids = torch.cat([torch.full((2, 2), 7), response_ids], dim=1)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.cat(
        [torch.zeros((2, 2), dtype=torch.bool), mask], dim=1
    )
    binding = RatioPoolBindingV2.from_tensors(
        input_ids=input_ids,
        response_ids=response_ids,
        attention_mask=attention,
        response_mask=response_mask,
        valid_mask=mask,
    )
    advantage = torch.linspace(-0.7, 0.4, token_count).reshape_as(old)
    loss = advantage.abs() + 0.1
    grad = loss * 0.5
    evidence = compute_ratio_evidence_v2(
        log_q_pre=q_pre,
        log_p_old_canonical=old,
        log_mu_sampler=mu,
        log_q_post=q_post,
        valid_mask=mask,
        prompt_ids=("p0", "p1"),
        source_roles=("medical_opd_o1", "medical_opd_cmb"),
        token_ids=response_ids,
        advantage=advantage,
        loss_contribution=loss,
        gradient_proxy=grad,
        pool_binding=binding,
        policy_version=20,
        q_pre_adapter_sha256="a" * 64,
        p_old_adapter_sha256="a" * 64,
        sampler_version=20,
        refresh_version=20,
        backend_log_clip=math.log(2.0),
        post_shift_tail_abs_log_threshold=math.log(5.0),
    )
    return evidence


def _thresholds() -> dict[str, object]:
    return {
        "schema_version": 2,
        "ppo_abs_log_p99_max": 1.0e-4,
        "ppo_abs_log_p999_max": 1.0e-4,
        "backend_abs_log_p99_max": 0.35,
        "backend_abs_log_p999_max": 0.80,
        "backend_clip_fraction_max": 0.05,
        "pooled_ess_floor": 0.95,
        "per_prompt_ess_floor": 0.95,
        "per_prompt_ess_min_tokens": 32,
        "approx_kl_abs_max": 0.05,
        "ppo_clip_fraction_max": 0.20,
        "preclip_grad_norm_absolute_max": 160.0,
        "preclip_grad_robust_z_max": 300.0,
        "healthy_grad_median": 1.0,
        "healthy_grad_mad": 0.5,
        "relative_update_norm_max": 0.006,
        "post_shift_abs_log_p99_max": 0.35,
        "post_shift_abs_log_p999_max": 1.0,
        "tail_loss_share_max": 0.05,
        "tail_gradient_proxy_share_max": 0.05,
        "raw_post_ratio_max_warning_above": 5.0,
        "consecutive_warning_abort_count": 2,
    }


def test_three_ratios_are_physically_separate_and_stage_typed():
    evidence = _fixture()
    assert evidence["ppo_ratio"]["stage"] == "pre_update"
    assert evidence["backend_correction"]["stage"] == "sampler_to_canonical"
    assert evidence["post_update_policy_shift"]["stage"] == "post_update"
    assert evidence["ppo_ratio"]["formula"] == "log_q_pre-log_p_old_canonical"
    assert evidence["backend_correction"]["formula"] == "log_p_old_canonical-log_mu_sampler"
    assert evidence["post_update_policy_shift"]["formula"] == "log_q_post-log_p_old_canonical"

    aliased = copy.deepcopy(evidence)
    aliased["ppo_ratio"]["stage"] = "post_update"
    with pytest.raises(RatioContractV2Error, match="stage"):
        validate_ratio_evidence_v2(aliased)


def test_canonical_pre_and_old_require_same_tokens_mask_policy_and_adapter():
    evidence = _fixture()
    assert validate_ratio_evidence_v2(evidence)["passed"] is True
    assert evidence["ppo_ratio"]["ratio"]["max"] == pytest.approx(1.0)

    stale = copy.deepcopy(evidence)
    stale["identity"]["p_old_policy_version"] = 19
    with pytest.raises(RatioContractV2Error, match="policy version"):
        validate_ratio_evidence_v2(stale)
    stale = copy.deepcopy(evidence)
    stale["identity"]["p_old_adapter_sha256"] = "b" * 64
    with pytest.raises(RatioContractV2Error, match="adapter"):
        validate_ratio_evidence_v2(stale)
    stale = copy.deepcopy(evidence)
    stale["identity"]["refresh_version"] = 19
    with pytest.raises(RatioContractV2Error, match="refresh"):
        validate_ratio_evidence_v2(stale)


def test_ratio_and_ess_cannot_bind_different_token_pools():
    evidence = _fixture()
    mismatched = copy.deepcopy(evidence)
    mismatched["backend_correction"]["ess"]["pool_binding_sha256"] = "f" * 64
    with pytest.raises(RatioContractV2Error, match="token pool"):
        validate_ratio_evidence_v2(mismatched)


def test_backend_correction_records_raw_and_clipped_without_entering_gradient():
    evidence = _fixture()
    correction = evidence["backend_correction"]
    assert correction["raw_log"]["max"] == pytest.approx(0.05, abs=1.0e-6)
    assert correction["clipped_log"]["max"] == pytest.approx(0.05, abs=1.0e-6)
    assert correction["clip_fraction"] == 0.0
    assert correction["detached_from_gradient"] is True
    assert correction["ess"]["formula"] == "(sum(w_clipped)^2)/(n*sum(w_clipped^2))"
    assert correction["ess"]["aggregation"] == "token_pooled_and_per_prompt"
    assert set(correction["ess"]["per_prompt"]) == {"p0", "p1"}


def test_raw_ratio_max_alone_is_warning_not_rejection():
    # One negligible outlier among 20k tokens keeps P99.9 and influence shares low.
    evidence = _fixture(post_outlier=math.log(12.0), token_count=20_000)
    evidence["post_update_policy_shift"]["tail"]["absolute_loss_share"] = 1.0e-5
    evidence["post_update_policy_shift"]["tail"]["gradient_proxy_share"] = 1.0e-5
    result = evaluate_ratio_health_v2(
        evidence,
        thresholds=_thresholds(),
        preclip_grad_norm=1.0,
        relative_update_norm=0.002,
        ppo_clip_fraction=0.0,
        consecutive_warning_count=0,
    )
    assert result["accepted"] is True
    assert result["warnings"] == ["raw_post_ratio_max"]


def test_influential_ratio_tail_is_rejected_even_when_raw_max_is_only_warning():
    evidence = _fixture(post_outlier=math.log(12.0), token_count=20_000)
    evidence["post_update_policy_shift"]["tail"]["absolute_loss_share"] = 0.20
    evidence["post_update_policy_shift"]["tail"]["gradient_proxy_share"] = 0.30
    with pytest.raises(RatioHealthV2Error, match="tail_loss_share.*tail_gradient_proxy_share"):
        evaluate_ratio_health_v2(
            evidence,
            thresholds=_thresholds(),
            preclip_grad_norm=1.0,
            relative_update_norm=0.002,
            ppo_clip_fraction=0.0,
            consecutive_warning_count=0,
        )


def test_short_prompt_ess_is_recorded_but_hard_floor_uses_registered_min_tokens():
    evidence = _fixture()
    evidence["backend_correction"]["ess"]["per_prompt"]["p0"] = {
        "token_count": 2,
        "ess_fraction": 0.89,
    }
    result = evaluate_ratio_health_v2(
        evidence,
        thresholds=_thresholds(),
        preclip_grad_norm=1.0,
        relative_update_norm=0.002,
        ppo_clip_fraction=0.0,
        consecutive_warning_count=0,
    )
    assert result["accepted"] is True
    assert "short_prompt_ess_below_floor:p0" in result["warnings"]
    assert result["next_consecutive_warning_count"] == 0


def test_preupdate_backend_health_uses_pooled_clip_and_minimum_support_prompt_ess():
    evidence = _fixture()
    evidence["backend_correction"]["clip_fraction"] = 1.0 / 827.0
    evidence["backend_correction"]["ess"]["pooled_fraction"] = 0.9965
    evidence["backend_correction"]["ess"]["per_prompt"]["p0"] = {
        "token_count": 6,
        "ess_fraction": 0.9117,
    }
    result = evaluate_preupdate_backend_health_v2(
        evidence, thresholds=_thresholds()
    )
    assert result["accepted"] is True
    assert result["hard_gate_aggregation"] == {
        "backend_clip_fraction": "token_pooled",
        "pooled_ess": "token_pooled",
        "per_prompt_ess": "minimum_32_valid_tokens",
    }
    assert "short_prompt_ess_below_floor:p0" in result["diagnostic_warnings"]


def test_preupdate_backend_health_still_rejects_pooled_or_supported_prompt_risk():
    evidence = _fixture()
    evidence["backend_correction"]["clip_fraction"] = 0.06
    evidence["backend_correction"]["ess"]["per_prompt"]["p0"] = {
        "token_count": 64,
        "ess_fraction": 0.90,
    }
    with pytest.raises(
        RatioHealthV2Error, match="backend_clip_fraction.*per_prompt_ess:p0"
    ):
        evaluate_preupdate_backend_health_v2(evidence, thresholds=_thresholds())


def test_nonfinite_and_pre_post_pool_drift_fail_closed():
    evidence = _fixture()
    evidence["post_update_policy_shift"]["log"]["p99"] = float("nan")
    with pytest.raises(RatioContractV2Error, match="finite"):
        validate_ratio_evidence_v2(evidence)
