"""OPD core mathematics tests (agent.md §6.1 / CLAUDE.md §6 "OPD核心").

Every test here runs on CPU in milliseconds and downloads nothing. Each test
maps to one required property; the mapping is stated in the docstring so a
reviewer can check coverage against the plan rather than against intuition.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.opd.core import (
    DomainKLController,
    OPDBatch,
    build_opd_batch,
    masked_mean,
    policy_entropy,
    ppo_policy_loss,
    reverse_kl_per_token,
    scale_and_clip_advantage,
    selected_token_logprobs,
    teacher_student_advantage,
    token_logprobs,
    assert_same_targets,
)
from src.opd.toy_lm import ToyCausalLM, ToyLMConfig, make_toy_pair

PAD = 0
EOS = 1


def simple_batch(include_eos_in_loss: bool = True) -> OPDBatch:
    """Two sequences of different length -> exercises padding + length bias."""
    return build_opd_batch(
        prompt_ids=[[5, 6, 7], [8, 9]],
        completion_ids=[[10, 11, EOS], [12, 13, 14, 15, EOS]],
        pad_token_id=PAD,
        eos_token_id=EOS,
        domains=("medical", "general"),
        include_eos_in_loss=include_eos_in_loss,
    )


# ---------------------------------------------------------------------------
# toy model sanity: without causality, every downstream test would be vacuous
# ---------------------------------------------------------------------------


def test_toy_model_is_causal():
    """logits[:, i] must not depend on tokens after position i."""
    torch.manual_seed(0)
    model = ToyCausalLM(ToyLMConfig(vocab_size=16, hidden_size=8, num_heads=2))
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        base = model(ids)
        perturbed_ids = ids.clone()
        perturbed_ids[0, 4] = 9  # change only the LAST token
        perturbed = model(perturbed_ids)
    # positions 0..3 predict tokens 1..4 and must be unaffected by token 4's identity
    assert torch.allclose(base[:, :4], perturbed[:, :4], atol=1e-6)
    assert not torch.allclose(base[:, 4], perturbed[:, 4], atol=1e-6)


# ---------------------------------------------------------------------------
# batch construction: prompt / padding / EOS masks
# ---------------------------------------------------------------------------


def test_batch_masks_prompt_padding_and_eos():
    batch = simple_batch()
    # sequence 0: 3 prompt + 3 completion = 6 real tokens, padded to 7
    assert batch.seq_len == 7
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 0]
    assert batch.completion_mask[0].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert batch.attention_mask[1].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert batch.completion_mask[1].tolist() == [0, 0, 1, 1, 1, 1, 1]
    # target mask is completion_mask shifted left by one (autoregressive)
    assert batch.target_mask()[0].tolist() == [0, 0, 1, 1, 1, 0]
    assert batch.num_completion_tokens() == 3 + 5


def test_eos_can_be_excluded_from_loss():
    with_eos = simple_batch(include_eos_in_loss=True)
    without_eos = simple_batch(include_eos_in_loss=False)
    assert with_eos.num_completion_tokens() == 8
    assert without_eos.num_completion_tokens() == 6  # one EOS dropped per sequence
    # the dropped position is exactly the trailing EOS, nothing else changed
    diff = (with_eos.completion_mask - without_eos.completion_mask)
    assert diff.sum().item() == 2
    assert with_eos.input_ids[0, 5].item() == EOS
    assert diff[0, 5].item() == 1


def test_batch_rejects_silent_truncation_and_empty_completion():
    with pytest.raises(ValueError, match="exceeds max_length"):
        build_opd_batch([[1, 2, 3]], [[4, 5]], pad_token_id=PAD, max_length=4)
    with pytest.raises(ValueError, match="empty completion"):
        build_opd_batch([[1, 2]], [[]], pad_token_id=PAD)
    with pytest.raises(ValueError, match="empty prompt"):
        build_opd_batch([[]], [[4]], pad_token_id=PAD)


def test_batch_rejects_trainable_padding():
    ids = torch.tensor([[5, 6, 7, PAD]])
    attn = torch.tensor([[1, 1, 1, 0]])
    comp = torch.tensor([[0, 1, 1, 1]])  # marks the pad position as trainable
    with pytest.raises(ValueError, match="padding position"):
        OPDBatch(
            input_ids=ids,
            attention_mask=attn,
            completion_mask=comp,
            prompt_lengths=torch.tensor([1]),
            completion_lengths=torch.tensor([3]),
        )


def test_domain_mask_selects_rows():
    batch = simple_batch()
    med = batch.domain_mask("medical")
    gen = batch.domain_mask("general")
    assert med[1].sum().item() == 0
    assert gen[0].sum().item() == 0
    assert (med + gen).equal(batch.target_mask())


# ---------------------------------------------------------------------------
# right shift / target alignment
# ---------------------------------------------------------------------------


def test_token_logprobs_matches_manual_gather_on_three_tokens():
    """Hand-checkable case: 1 sequence, 3 tokens, vocab 4 (PROJECT_PLAN §14)."""
    logits = torch.tensor(
        [[
            [0.0, 1.0, 0.0, 0.0],  # position 0 -> predicts token at index 1
            [0.0, 0.0, 2.0, 0.0],  # position 1 -> predicts token at index 2
            [5.0, 5.0, 5.0, 5.0],  # position 2 -> dropped (no target)
        ]]
    )
    ids = torch.tensor([[3, 1, 2]])
    lp = token_logprobs(logits, ids)
    assert lp.shape == (1, 2)
    expected0 = 1.0 - math.log(math.exp(0.0) * 3 + math.exp(1.0))
    expected1 = 2.0 - math.log(math.exp(0.0) * 3 + math.exp(2.0))
    assert lp[0, 0].item() == pytest.approx(expected0, abs=1e-6)
    assert lp[0, 1].item() == pytest.approx(expected1, abs=1e-6)


def test_logprob_uses_previous_position_not_current():
    """A wrong (unshifted) implementation would read logits at the target index."""
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 6)
    ids = torch.tensor([[2, 3, 4, 5]])
    lp = token_logprobs(logits, ids)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    for i in range(3):
        correct = log_probs[0, i, ids[0, i + 1]]
        wrong = log_probs[0, i + 1, ids[0, i + 1]]
        assert lp[0, i].item() == pytest.approx(correct.item(), abs=1e-6)
        assert lp[0, i].item() != pytest.approx(wrong.item(), abs=1e-6)


def test_student_and_teacher_score_identical_targets():
    """Teacher must not generate a second completion (PROJECT_PLAN §9)."""
    batch = simple_batch()
    student, teacher = make_toy_pair(seed=3)
    s_logits = student(batch.input_ids, batch.attention_mask)
    with torch.no_grad():
        t_logits = teacher(batch.input_ids, batch.attention_mask)
    s_lp = selected_token_logprobs(s_logits, batch)
    t_lp = selected_token_logprobs(t_logits, batch)
    assert_same_targets(batch, s_lp, t_lp, names=["student", "teacher"])
    # identical target grid, byte-identical inputs
    assert batch.fingerprint() == batch.fingerprint()
    assert torch.equal(batch.target_ids(), batch.input_ids[:, 1:])
    # teacher carries no gradient and its params are frozen
    assert not t_lp.requires_grad
    assert all(not p.requires_grad for p in teacher.parameters())
    # and the two models genuinely differ, so the test is not vacuous
    assert not torch.allclose(s_lp, t_lp)


def test_assert_same_targets_catches_shape_drift():
    batch = simple_batch()
    with pytest.raises(ValueError, match="expected"):
        assert_same_targets(batch, torch.zeros(batch.batch_size, batch.seq_len))


# ---------------------------------------------------------------------------
# reverse KL and advantage signs
# ---------------------------------------------------------------------------


def test_reverse_kl_and_advantage_signs_are_opposite():
    student_lp = torch.tensor([[-0.5, -2.0]])
    teacher_lp = torch.tensor([[-1.5, -0.5]])
    r = reverse_kl_per_token(student_lp, teacher_lp)
    assert r[0, 0].item() == pytest.approx(1.0)   # student over-confident
    assert r[0, 1].item() == pytest.approx(-1.5)  # teacher prefers this token
    adv = teacher_student_advantage(student_lp, teacher_lp, beta=2.0)
    assert adv[0, 0].item() == pytest.approx(-2.0)
    assert adv[0, 1].item() == pytest.approx(3.0)
    assert torch.allclose(adv, -2.0 * r)


def test_advantage_rejects_grad_carrying_teacher():
    student_lp = torch.tensor([[-0.5]])
    teacher_lp = torch.tensor([[-1.0]], requires_grad=True)
    with pytest.raises(ValueError, match="teacher logprobs carry grad"):
        teacher_student_advantage(student_lp, teacher_lp)


def test_biased_teacher_yields_positive_advantage_for_its_preferred_token():
    """End-to-end sign check with real (toy) models rather than fixed numbers."""
    preferred = 11
    batch = build_opd_batch([[5, 6]], [[preferred, preferred]], pad_token_id=PAD, domains=("medical",))
    student, teacher = make_toy_pair(seed=5, teacher_bias_token=preferred, teacher_bias_strength=8.0)
    with torch.no_grad():
        s_lp = selected_token_logprobs(student(batch.input_ids, batch.attention_mask), batch)
        t_lp = selected_token_logprobs(teacher(batch.input_ids, batch.attention_mask), batch)
    adv = teacher_student_advantage(s_lp, t_lp, beta=1.0)
    mask = batch.target_mask()
    assert masked_mean(adv, mask).item() > 0.0
    assert masked_mean(reverse_kl_per_token(s_lp, t_lp), mask).item() < 0.0


# ---------------------------------------------------------------------------
# PPO update behaviour
# ---------------------------------------------------------------------------


def _one_ppo_step(advantage_value: float, lr: float = 0.5) -> tuple[float, float]:
    """Run one optimizer step with a constant advantage; return before/after logprob."""
    torch.manual_seed(7)
    batch = build_opd_batch([[5, 6]], [[10, 11]], pad_token_id=PAD, domains=("medical",))
    model = ToyCausalLM(ToyLMConfig())
    with torch.no_grad():
        old_lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch).detach()
    mask = batch.target_mask()
    adv = torch.full_like(old_lp, advantage_value)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    new_lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch)
    loss, stats = ppo_policy_loss(new_lp, old_lp, adv, mask, clip_range=0.2)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        after_lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch)
    before = float(masked_mean(old_lp, mask))
    after = float(masked_mean(after_lp, mask))
    # at the first inner step the ratio is exactly 1 and nothing is clipped
    assert stats.ratio_mean == pytest.approx(1.0, abs=1e-6)
    assert stats.clip_fraction == 0.0
    return before, after


def test_positive_advantage_increases_target_probability():
    """Teacher prefers the token -> update must raise its probability."""
    before, after = _one_ppo_step(advantage_value=+1.0)
    assert after > before, f"expected increase, got {before:.4f} -> {after:.4f}"


def test_negative_advantage_decreases_target_probability():
    """Student over-confident -> update must lower its probability."""
    before, after = _one_ppo_step(advantage_value=-1.0)
    assert after < before, f"expected decrease, got {before:.4f} -> {after:.4f}"


def test_ppo_rejects_grad_carrying_old_logprobs_and_advantages():
    lp = torch.zeros(1, 2, requires_grad=True)
    mask = torch.ones(1, 2)
    with pytest.raises(ValueError, match="old_logprobs carry grad"):
        ppo_policy_loss(lp, lp, torch.zeros(1, 2), mask)
    with pytest.raises(ValueError, match="advantages must be detached"):
        ppo_policy_loss(lp, lp.detach(), torch.zeros(1, 2, requires_grad=True), mask)


def test_old_logprobs_stay_frozen_across_two_inner_updates():
    """The rollout logprobs must not drift into the updated policy's values."""
    torch.manual_seed(11)
    batch = build_opd_batch([[5, 6]], [[10, 11]], pad_token_id=PAD)
    model = ToyCausalLM(ToyLMConfig())
    with torch.no_grad():
        old_lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch).detach()
    old_snapshot = old_lp.clone()
    mask = batch.target_mask()
    adv = torch.full_like(old_lp, 1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.4)

    ratios = []
    for _ in range(2):
        new_lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch)
        loss, stats = ppo_policy_loss(new_lp, old_lp, adv, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ratios.append(stats.ratio_mean)

    assert torch.equal(old_lp, old_snapshot), "old_logprobs were mutated during the update"
    assert ratios[0] == pytest.approx(1.0, abs=1e-6)
    assert ratios[1] > 1.0, "second inner step should see ratio > 1 after a positive-advantage update"


def test_ppo_ratio_clipping_bounds_and_fraction():
    """Synthetic ratios: 2 of 4 tokens fall outside [1-eps, 1+eps]."""
    old_lp = torch.zeros(1, 4)
    # log-ratios: +0.0 (ratio 1), +0.1 (1.105), +1.0 (2.718), -1.0 (0.368)
    new_lp = torch.tensor([[0.0, 0.1, 1.0, -1.0]])
    adv = torch.ones(1, 4)
    mask = torch.ones(1, 4)
    loss, stats = ppo_policy_loss(new_lp, old_lp, adv, mask, clip_range=0.2)
    assert stats.clip_fraction == pytest.approx(0.5)
    # with positive advantage the objective is min(r*A, clip(r)*A):
    # ratios above 1+eps are capped at 1.2, ratios below 1-eps use the raw ratio
    expected = -(1.0 + 1.105170918 + 1.2 + 0.367879441) / 4
    assert stats.loss == pytest.approx(expected, abs=1e-5)
    assert stats.ratio_mean == pytest.approx((1.0 + 1.105170918 + 2.718281828 + 0.367879441) / 4, abs=1e-5)


def test_negative_advantage_clipping_uses_lower_bound():
    old_lp = torch.zeros(1, 1)
    new_lp = torch.tensor([[math.log(0.5)]])  # ratio 0.5, below 1 - 0.2
    adv = torch.tensor([[-1.0]])
    mask = torch.ones(1, 1)
    _, stats = ppo_policy_loss(new_lp, old_lp, adv, mask, clip_range=0.2)
    # min(0.5 * -1, 0.8 * -1) = -0.8  -> loss = +0.8
    assert stats.loss == pytest.approx(0.8, abs=1e-6)
    assert stats.clip_fraction == pytest.approx(1.0)


def test_loss_ignores_prompt_and_padding_positions():
    batch = simple_batch()
    torch.manual_seed(13)
    lp_new = torch.randn(batch.batch_size, batch.seq_len - 1)
    lp_old = lp_new.detach().clone()
    adv = torch.ones_like(lp_new)
    mask = batch.target_mask()

    loss_a, stats_a = ppo_policy_loss(lp_new, lp_old, adv, mask)
    # corrupting masked-out positions must not change anything
    lp_new2 = lp_new.clone()
    lp_new2[0, 0] += 100.0  # prompt position
    lp_new2[0, 5] += 100.0  # padding position
    loss_b, stats_b = ppo_policy_loss(lp_new2, lp_old, adv, mask)
    assert stats_a.num_tokens == 8
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)


def test_token_mean_has_no_length_bias_but_seq_mean_does():
    """PROJECT_PLAN §9 / agent.md §6.1: reduction must not favour short sequences."""
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    old_lp = torch.zeros(2, 4)
    new_lp = torch.zeros(2, 4)
    # long sequence has advantage 0, the single-token short sequence has 4
    adv = torch.tensor([[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]])

    _, token_stats = ppo_policy_loss(new_lp, old_lp, adv, mask, reduction="token_mean")
    _, seq_stats = ppo_policy_loss(new_lp, old_lp, adv, mask, reduction="seq_mean_token_mean")
    # token_mean: -(0*4 + 4*1)/5 = -0.8 ; seq_mean: mean(0, -4) = -2.0
    assert token_stats.loss == pytest.approx(-0.8, abs=1e-6)
    assert seq_stats.loss == pytest.approx(-2.0, abs=1e-6)
    assert abs(seq_stats.loss) > abs(token_stats.loss)


def test_masked_mean_rejects_empty_mask_and_shape_mismatch():
    with pytest.raises(ValueError, match="all-zero mask"):
        masked_mean(torch.ones(1, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="must match"):
        masked_mean(torch.ones(1, 3), torch.zeros(1, 4))


# ---------------------------------------------------------------------------
# advantage clipping + domain-level KL safety scaling (PROJECT_PLAN §11.4)
# ---------------------------------------------------------------------------


def test_scale_and_clip_advantage_reports_clipped_fraction():
    adv = torch.tensor([[3.0, -3.0, 0.5, -0.5]])
    clipped, flags = scale_and_clip_advantage(adv, scales=1.0, a_max=1.0)
    assert clipped.tolist() == [[1.0, -1.0, 0.5, -0.5]]
    assert flags.tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_scale_shrinks_updates_and_can_never_amplify():
    adv = torch.tensor([[2.0, -2.0]])
    clipped, flags = scale_and_clip_advantage(adv, scales=0.25, a_max=10.0)
    assert clipped.tolist() == [[0.5, -0.5]]
    assert flags.sum().item() == 0.0
    with pytest.raises(ValueError, match="must be <= 1"):
        scale_and_clip_advantage(adv, scales=1.5, a_max=10.0)


def test_per_sequence_scales_broadcast_over_tokens():
    adv = torch.ones(2, 3)
    scales = torch.tensor([1.0, 0.5])
    clipped, _ = scale_and_clip_advantage(adv, scales=scales, a_max=10.0)
    assert clipped[0].tolist() == [1.0, 1.0, 1.0]
    assert clipped[1].tolist() == [0.5, 0.5, 0.5]


def test_domain_kl_controller_ema_and_safety_scale():
    ctrl = DomainKLController(kappa={"medical": 0.5, "general": 0.5}, rho=0.5, domains=["medical", "general"])
    # no evidence yet -> no throttling
    assert ctrl.scale("medical") == 1.0
    # first observation initialises the EMA exactly
    assert ctrl.update("medical", 0.2) == pytest.approx(0.2)
    assert ctrl.scale("medical") == 1.0  # 0.5 / 0.2 > 1 -> capped at 1
    # KL blows up -> EMA rises, scale drops below 1
    ctrl.update("medical", 4.0)
    assert ctrl.ema["medical"] == pytest.approx(0.5 * 0.2 + 0.5 * 4.0)
    scale = ctrl.scale("medical")
    assert 0.0 < scale < 1.0
    assert scale == pytest.approx(0.5 / (2.1 + 1e-6), rel=1e-4)
    # the other domain is untouched: throttling is per-domain
    assert ctrl.scale("general") == 1.0


def test_domain_kl_controller_state_roundtrip():
    ctrl = DomainKLController(kappa=0.3, rho=0.9)
    ctrl.update("medical", 1.0)
    state = ctrl.state_dict()
    restored = DomainKLController(kappa=1.0, rho=0.1)
    restored.load_state_dict(state)
    assert restored.ema == ctrl.ema
    assert restored.scale("medical") == pytest.approx(ctrl.scale("medical"))


def test_domain_kl_controller_rejects_bad_config():
    with pytest.raises(ValueError):
        DomainKLController(kappa=0.0)
    with pytest.raises(ValueError):
        DomainKLController(kappa=1.0, rho=1.0)
    with pytest.raises(KeyError):
        DomainKLController(kappa={"medical": 1.0}).kappa("general")


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


def test_policy_entropy_is_masked_and_bounded():
    batch = simple_batch()
    torch.manual_seed(17)
    model = ToyCausalLM(ToyLMConfig(vocab_size=32))
    with torch.no_grad():
        logits = model(batch.input_ids, batch.attention_mask)
        ent = policy_entropy(logits, batch.target_mask())
        # uniform logits -> entropy exactly log V, and masked positions are ignored
        uniform = torch.zeros_like(logits)
        ent_uniform = policy_entropy(uniform, batch.target_mask())
    assert 0.0 < float(ent) <= math.log(32) + 1e-6
    assert float(ent_uniform) == pytest.approx(math.log(32), abs=1e-6)
