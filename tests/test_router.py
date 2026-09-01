"""CA-OPD router tests (docs/REPRODUCIBILITY.md §6.2 / docs/REPRODUCIBILITY.md §6 "CA-OPD调度").

Coverage map:
* general below constraint -> Base probability rises        -> test_general_below_floor_raises_base_probability
* medical gap grows        -> Medical probability rises     -> test_larger_medical_gap_raises_medical_probability
* p_min / p_max always hold                                -> test_probability_bounds_always_enforced
* EMA + hysteresis absorb single-window noise               -> test_single_noisy_window_does_not_flip_state
* both objectives met -> early stop                        -> test_early_stop_requires_both_objectives
* scheduler cannot accept a final-test evaluator            -> test_router_rejects_final_test_evaluator
* checkpoint resume keeps router state consistent           -> test_state_dict_roundtrip_restores_decisions
"""

from __future__ import annotations

import json

import pytest

from src.opd.router import (
    BASE,
    MEDICAL,
    ConstraintAwareRouter,
    FinalTestLeakageError,
    FixedRatioRouter,
    RouterConfig,
    RouterState,
)


def base_config(**overrides) -> RouterConfig:
    params = dict(
        medical_target=0.60,
        general_baseline=0.50,
        delta=0.01,
        scale_medical=0.05,
        scale_general=0.05,
        rho=0.5,
        tau=1.0,
        p_min=0.2,
        p_max=0.8,
        window_steps=10,
        windows_below_to_recover=2,
        windows_above_to_release=1,
        early_stop_patience=2,
        early_stop_min_improvement=0.002,
        initial_p_medical=0.5,
    )
    params.update(overrides)
    return RouterConfig(**params)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_config_rejects_degenerate_bounds_and_temperature():
    with pytest.raises(ValueError, match="p_min must be > 0"):
        base_config(p_min=0.0)
    with pytest.raises(ValueError, match="p_min <= p_max"):
        base_config(p_min=0.9, p_max=0.2)
    with pytest.raises(ValueError, match="tau must be > 0"):
        base_config(tau=0.0)
    with pytest.raises(ValueError, match="rho must be in"):
        base_config(rho=1.0)
    with pytest.raises(ValueError, match="initial_p_medical"):
        base_config(initial_p_medical=0.95)
    with pytest.raises(ValueError, match="unknown router config keys"):
        RouterConfig.from_mapping({"medical_target": 0.5, "general_baseline": 0.5, "delta": 0.0, "tao": 1.0})


def test_general_floor_is_baseline_minus_delta():
    cfg = base_config(general_baseline=0.62, delta=0.01)
    assert cfg.general_floor == pytest.approx(0.61)


# ---------------------------------------------------------------------------
# gap -> probability direction
# ---------------------------------------------------------------------------


def test_general_below_floor_raises_base_probability():
    """General ability collapses while medical is already at target."""
    router = ConstraintAwareRouter(base_config(), seed=0)
    healthy = router.update(medical_accuracy=0.60, general_accuracy=0.55)
    damaged = router.update(medical_accuracy=0.60, general_accuracy=0.20)
    assert damaged.general_gap > healthy.general_gap
    assert damaged.p_base > healthy.p_base
    assert damaged.p_medical < healthy.p_medical


def test_larger_medical_gap_raises_medical_probability():
    """Same general ability, worse medical ability -> more Medical Teacher."""
    small_gap = ConstraintAwareRouter(base_config(), seed=0).update(0.58, 0.52)
    large_gap = ConstraintAwareRouter(base_config(), seed=0).update(0.30, 0.52)
    assert large_gap.medical_gap > small_gap.medical_gap
    assert large_gap.p_medical > small_gap.p_medical


def test_probability_bounds_always_enforced():
    """Extreme gaps in both directions must still respect [p_min, p_max]."""
    cfg = base_config(p_min=0.15, p_max=0.75, windows_below_to_recover=99)
    router = ConstraintAwareRouter(cfg, seed=0)
    extreme_medical = router.update(medical_accuracy=0.0, general_accuracy=1.0)
    assert extreme_medical.p_medical == pytest.approx(cfg.p_max)
    router2 = ConstraintAwareRouter(cfg, seed=0)
    extreme_general = router2.update(medical_accuracy=1.0, general_accuracy=0.0)
    assert extreme_general.p_medical == pytest.approx(cfg.p_min)
    assert extreme_general.p_base == pytest.approx(1.0 - cfg.p_min)


def test_probabilities_always_sum_to_one():
    router = ConstraintAwareRouter(base_config(), seed=0)
    for m, g in [(0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.0, 0.0), (1.0, 1.0)]:
        d = router.update(m, g)
        assert d.p_medical + d.p_base == pytest.approx(1.0)


def test_update_rejects_non_accuracy_inputs():
    router = ConstraintAwareRouter(base_config(), seed=0)
    with pytest.raises(ValueError, match="accuracy in"):
        router.update(1.2, 0.5)
    with pytest.raises(ValueError, match="accuracy in"):
        router.update(0.5, -0.1)


# ---------------------------------------------------------------------------
# EMA + hysteresis
# ---------------------------------------------------------------------------


def test_ema_initialises_exactly_then_smooths():
    router = ConstraintAwareRouter(base_config(rho=0.5), seed=0)
    first = router.update(0.40, 0.50)
    assert first.medical_ema == pytest.approx(0.40)
    second = router.update(0.60, 0.50)
    assert second.medical_ema == pytest.approx(0.5 * 0.40 + 0.5 * 0.60)


def test_single_noisy_window_does_not_flip_state():
    """One bad general reading must not put the run into recovery mode."""
    router = ConstraintAwareRouter(base_config(rho=0.7), seed=0)
    router.update(0.55, 0.55)  # healthy
    noisy = router.update(0.55, 0.10)  # single outlier
    assert noisy.state is RouterState.PURSUE_MEDICAL, "one noisy window flipped the state machine"
    recovered = router.update(0.55, 0.55)
    assert recovered.state is RouterState.PURSUE_MEDICAL
    # ... but a genuine sustained drop does flip it
    router.update(0.55, 0.05)
    sustained = router.update(0.55, 0.05)
    assert sustained.state is RouterState.RECOVER_GENERAL


def test_recovery_state_pins_medical_probability_to_p_min():
    cfg = base_config(windows_below_to_recover=1)
    router = ConstraintAwareRouter(cfg, seed=0)
    d = router.update(medical_accuracy=0.0, general_accuracy=0.0)  # huge medical gap too
    assert d.state is RouterState.RECOVER_GENERAL
    assert d.p_medical == pytest.approx(cfg.p_min), "recovery must favour the Base Teacher"


def test_release_from_recovery_requires_configured_windows_above():
    cfg = base_config(windows_below_to_recover=1, windows_above_to_release=2, rho=0.0)
    router = ConstraintAwareRouter(cfg, seed=0)
    assert router.update(0.5, 0.10).state is RouterState.RECOVER_GENERAL
    assert router.update(0.5, 0.90).state is RouterState.RECOVER_GENERAL  # only 1 good window
    assert router.update(0.5, 0.90).state is RouterState.PURSUE_MEDICAL  # 2 good windows


def test_state_machine_transitions_are_complete():
    """Walk every transition of the 2-state machine explicitly."""
    cfg = base_config(windows_below_to_recover=2, windows_above_to_release=1, rho=0.0)
    router = ConstraintAwareRouter(cfg, seed=0)
    seq = [
        (0.5, 0.90, RouterState.PURSUE_MEDICAL),  # pursue -> pursue
        (0.5, 0.10, RouterState.PURSUE_MEDICAL),  # 1 below: still pursue
        (0.5, 0.10, RouterState.RECOVER_GENERAL),  # 2 below: pursue -> recover
        (0.5, 0.10, RouterState.RECOVER_GENERAL),  # recover -> recover
        (0.5, 0.90, RouterState.PURSUE_MEDICAL),  # recover -> pursue
    ]
    for medical, general, expected in seq:
        assert router.update(medical, general).state is expected, (medical, general, expected)


# ---------------------------------------------------------------------------
# early stop
# ---------------------------------------------------------------------------


def test_early_stop_requires_both_objectives():
    cfg = base_config(medical_target=0.50, early_stop_patience=2, rho=0.0)
    router = ConstraintAwareRouter(cfg, seed=0)
    # medical at target but general below floor -> never stop
    for _ in range(5):
        d = router.update(0.60, 0.10)
        assert not d.should_stop
    # general recovers and medical plateaus -> stop after patience windows
    router2 = ConstraintAwareRouter(cfg, seed=0)
    decisions = [router2.update(0.60, 0.60) for _ in range(4)]
    assert not decisions[0].should_stop  # first window sets the best value
    assert decisions[-1].should_stop
    assert "no medical improvement" in decisions[-1].reason


def test_improvement_resets_early_stop_counter():
    cfg = base_config(medical_target=0.30, early_stop_patience=2, early_stop_min_improvement=0.01, rho=0.0)
    router = ConstraintAwareRouter(cfg, seed=0)
    router.update(0.40, 0.60)
    router.update(0.40, 0.60)  # no improvement (1)
    improved = router.update(0.80, 0.60)  # improvement -> counter resets
    assert not improved.should_stop
    assert router.windows_without_improvement == 0


# ---------------------------------------------------------------------------
# final-test isolation
# ---------------------------------------------------------------------------


class _FakeControllerDevEvaluator:
    split = "controller_dev"

    def allows_control_decisions(self) -> bool:
        return True


class _FakeFinalTestEvaluator:
    split = "final_test"

    def allows_control_decisions(self) -> bool:
        return False


def test_router_accepts_controller_dev_evaluator():
    router = ConstraintAwareRouter(base_config(), evaluator=_FakeControllerDevEvaluator(), seed=0)
    assert router.evaluator is not None


def test_router_rejects_final_test_evaluator():
    with pytest.raises(FinalTestLeakageError, match="final test"):
        ConstraintAwareRouter(base_config(), evaluator=_FakeFinalTestEvaluator(), seed=0)


def test_router_rejects_evaluator_without_contract():
    class Bare:
        split = "controller_dev"

    with pytest.raises(TypeError, match="allows_control_decisions"):
        ConstraintAwareRouter(base_config(), evaluator=Bare(), seed=0)


# ---------------------------------------------------------------------------
# sampling + windows + persistence
# ---------------------------------------------------------------------------


def test_window_boundary_uses_configured_K():
    router = ConstraintAwareRouter(base_config(window_steps=10), seed=0)
    assert not router.is_window_boundary(0)
    assert not router.is_window_boundary(9)
    assert router.is_window_boundary(10)
    assert router.is_window_boundary(20)


def test_sampling_follows_probability_and_records_realised_ratio():
    router = ConstraintAwareRouter(base_config(p_min=0.2, p_max=0.8), seed=42)
    router.update(0.0, 1.0)  # drives p_medical to p_max = 0.8
    assert router.p_medical == pytest.approx(0.8)
    draws = router.sample_window(2000)
    frac = draws.count(MEDICAL) / len(draws)
    assert 0.75 < frac < 0.85, frac
    assert router.realised_medical_fraction() == pytest.approx(frac)
    assert set(draws) <= {MEDICAL, BASE}


def test_sampling_is_reproducible_from_seed():
    a = ConstraintAwareRouter(base_config(), seed=7)
    b = ConstraintAwareRouter(base_config(), seed=7)
    a.update(0.4, 0.55)
    b.update(0.4, 0.55)
    assert a.sample_window(50) == b.sample_window(50)


def test_state_dict_roundtrip_restores_decisions():
    """Checkpoint resume must not change the router's future behaviour."""
    router = ConstraintAwareRouter(base_config(), seed=3)
    router.update(0.45, 0.52)
    router.update(0.47, 0.51)
    snapshot = json.loads(json.dumps(router.state_dict()))  # survives json round-trip
    expected_next = router.update(0.49, 0.50)
    expected_draws = router.sample_window(20)

    restored = ConstraintAwareRouter(base_config(), seed=999)
    restored.load_state_dict(snapshot)
    got_next = restored.update(0.49, 0.50)
    assert got_next.as_dict() == expected_next.as_dict()
    assert restored.sample_window(20) == expected_draws


def test_state_dict_rejects_config_change_on_resume():
    router = ConstraintAwareRouter(base_config(), seed=0)
    state = router.state_dict()
    other = ConstraintAwareRouter(base_config(tau=2.0), seed=0)
    with pytest.raises(ValueError, match="differs from the current config"):
        other.load_state_dict(state)


def test_decision_metrics_use_frozen_metric_names():
    from src.utils.metrics import METRIC_NAMES

    router = ConstraintAwareRouter(base_config(), seed=0)
    metrics = router.update(0.4, 0.5).as_metrics()
    assert set(metrics) <= METRIC_NAMES


# ---------------------------------------------------------------------------
# fixed-ratio baseline router (B4/B5 + "no dynamic routing" ablation)
# ---------------------------------------------------------------------------


def test_fixed_ratio_router_ignores_ability_gaps():
    router = FixedRatioRouter(p_medical=0.5, window_steps=10, seed=1)
    first = router.update(0.9, 0.9)
    second = router.update(0.1, 0.1)
    assert first.p_medical == second.p_medical == 0.5
    draws = router.sample_window(1000)
    assert 0.45 < draws.count(MEDICAL) / 1000 < 0.55


def test_fixed_ratio_two_to_one_matches_requested_ratio():
    router = FixedRatioRouter(p_medical=2 / 3, seed=1)
    draws = router.sample_window(3000)
    assert 0.63 < draws.count(MEDICAL) / 3000 < 0.70
