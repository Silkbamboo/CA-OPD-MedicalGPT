from __future__ import annotations

from copy import deepcopy

import pytest

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError, validate_p9_resume_manifest
from src.opd.p9_decision import decide_step200, decide_step240, enforce_optimizer_boundary, classify_step300
from src.opd.p9_health import evaluate_p9_common_health
from src.opd.p9_runtime import p9_extension_discriminator, p9_optimizer_step_limit
from src.opd.production_b2_formal_checkpoint_v1 import _formal_checkpoint_upper_bound


def _manifest():
    return {
        "complete": True, "resume_eligible": True, "optimizer_step": 120,
        "scheduler_step": 120, "logical_version": 120, "policy_version": 120,
        "data_cursor": 480,
        "package_content_sha256": "c21ca9acec85bb72014ddfc48b5cf9079f680807cbfed348fe5ce1cc619583e1",
        "config_sha256": "130c91b300aab30d6bbbbf7f7893d77fa7c636c361b01a8dd4fc948f92c44835",
        "manifest_sha256": "9f1d096d06b635737e1b90be3b92d6de32fd64b03fbcd97813e42d0a2ee88a99",
        "schedule_sha256": "ddba16637318580a9f31a938da14d7d6d59e49e50046f3f1faebc1ef38e6382c",
        "adapter_sha256": "6e34e1b9b83064016968dd7d1c9f9c4d70ff87058aa3cab2e2be52bee7570408",
    }


def test_p9_resume_requires_exact_step120_and_no_p8_identity():
    assert validate_p9_resume_manifest(_manifest())["passed"]
    for field, value in (("optimizer_step", 119), ("policy_version", 119), ("data_cursor", 476), ("complete", False)):
        changed = deepcopy(_manifest()); changed[field] = value
        with pytest.raises(P9ProtocolError):
            validate_p9_resume_manifest(changed)
    changed = deepcopy(_manifest()); changed["p8_candidate_weight"] = True
    with pytest.raises(P9ProtocolError, match="P8"):
        validate_p9_resume_manifest(changed)
    changed = deepcopy(_manifest()); changed["adapter_sha256"] = "f" * 64
    with pytest.raises(P9ProtocolError, match="adapter"):
        validate_p9_resume_manifest(changed)


def _health():
    return {"common_health_gate_passed": True}


def test_step200_state_machine_registered_paths():
    positive = decide_step200({120: 217, 150: 225, 180: 222, 200: 221}, ci_low=0.0, slope=1.0, health=_health())
    assert positive["status"] == "positive_at_200" and positive["continue"] is False
    negative = decide_step200({120: 217, 150: 218, 180: 218, 200: 217}, ci_low=-0.1, slope=0.0, health=_health())
    assert negative["status"] == "negative_at_200"
    promising = decide_step200({120: 217, 150: 218, 180: 219, 200: 220}, ci_low=-0.1, slope=1.0, health=_health())
    assert promising["status"] == "promising_at_200_continue_300" and promising["next_max_step"] == 300
    gray = decide_step200({120: 217, 150: 217, 180: 217, 200: 218}, ci_low=-0.1, slope=0.2, health=_health())
    assert gray["status"] == "gray_at_200_continue_240" and gray["next_max_step"] == 240


def test_health_failure_stops_and_step300_is_absolute():
    result = decide_step200({120: 217, 150: 220, 180: 221, 200: 222}, ci_low=-0.1, slope=1.0, health={"common_health_gate_passed": False})
    assert result["status"] == "negative_at_200"
    positive_but_unhealthy = decide_step200(
        {120: 217, 150: 225, 180: 224, 200: 225}, ci_low=0.1,
        slope=1.0, health={"common_health_gate_passed": False},
    )
    assert positive_but_unhealthy["status"] == "negative_at_200"
    assert enforce_optimizer_boundary(300, attempted_next_step=300) == 300
    with pytest.raises(P9ProtocolError, match="step300"):
        enforce_optimizer_boundary(300, attempted_next_step=301)


def test_step240_state_machine():
    assert decide_step240({120: 217, 150: 218, 180: 219, 200: 218, 240: 225}, ci_low=-0.1, slope=1.0, health=_health())["status"] == "positive_at_240"
    assert decide_step240({120: 217, 150: 218, 180: 219, 200: 218, 240: 220}, ci_low=-0.1, slope=1.0, health=_health())["status"] == "promising_at_240_continue_300"
    assert decide_step240({120: 217, 150: 218, 180: 219, 200: 218, 240: 219}, ci_low=-0.1, slope=0.0, health=_health())["status"] == "negative_at_240"
    assert decide_step240(
        {120: 217, 150: 225, 180: 224, 200: 224, 240: 225},
        ci_low=0.1, slope=1.0, health={"common_health_gate_passed": False},
    )["status"] == "negative_at_240"


def test_step300_final_evidence_grades_are_exact():
    assert classify_step300(best_medical_correct=221, paired_ci_low=0.001) == "b2_dose_statistically_supported"
    assert classify_step300(best_medical_correct=225, paired_ci_low=-0.01) == "b2_dose_practical_positive"
    assert classify_step300(best_medical_correct=220, paired_ci_low=-0.01) == "b2_dose_weak_positive_trend"
    assert classify_step300(best_medical_correct=219, paired_ci_low=0.0) == "b2_dose_negative_complete"


def test_kernel_widens_to_300_only_for_exact_p9_dose_discriminator():
    config = {
        "formal_b2": {"package_version": "p5_formal_b2_v1"},
        "p9_adaptive_dose": p9_extension_discriminator(),
    }
    config["p9_adaptive_dose"]["schedule_sha256"] = "f" * 64
    assert p9_optimizer_step_limit(config) == 300
    assert _formal_checkpoint_upper_bound(config) == 300
    config["p9_adaptive_dose"]["source_batch"] = {"medical_opd_o1": 3, "medical_opd_cmb": 1}
    with pytest.raises(Exception, match="P9"):
        p9_optimizer_step_limit(config)


def _health_records(late_gap: float = 1.0):
    records = []
    for step in range(121, 201):
        gap = late_gap if 181 <= step <= 200 else 1.0
        records.append({
            "optimizer_step": step,
            "reverse_kl": {"mean": gap},
            "loss": -0.1,
            "ess_fraction": 0.99,
            "adapter_delta_norm": 0.01,
            "ratio_health_v2": {"accepted": True},
            "ratio_v2": {"bounded_influence_v2": {"prompt_gradients": [
                {"bounded_norm": 0.2, "clip_scale": scale}
                for scale in (0.5, 0.5, 0.5, 1.0)
            ]}},
            "prompt_samples": [
                {"invalid": False, "repetition": False} for _ in range(4)
            ],
        })
    return records


def test_common_health_checks_checkpoint_and_signal_floor():
    passed = evaluate_p9_common_health(
        _health_records(), rejected_transactions=[],
        checkpoint_complete_resume_eligible=True,
        disk_free_bytes=20_000_000_000, projected_cost_cny=20.0,
    )
    assert passed["common_health_gate_passed"]
    incomplete = evaluate_p9_common_health(
        _health_records(), rejected_transactions=[],
        checkpoint_complete_resume_eligible=False,
        disk_free_bytes=20_000_000_000, projected_cost_cny=20.0,
    )
    assert not incomplete["common_health_gate_passed"]
    collapsed = evaluate_p9_common_health(
        _health_records(late_gap=0.19), rejected_transactions=[],
        checkpoint_complete_resume_eligible=True,
        disk_free_bytes=20_000_000_000, projected_cost_cny=20.0,
    )
    assert not collapsed["checks"]["late_teacher_gap_ge_20pct_early"]
