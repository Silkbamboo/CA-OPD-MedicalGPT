"""P1 gate tests: run planning and veRL config translation.

These run on CPU with no veRL installed - the point is to reject a bad formal-run
configuration before renting GPUs, mirroring the constraints veRL enforces at
startup (documented in docs/decisions/0001, replicated in src/opd/verl_config.py).
"""

from __future__ import annotations

import pytest

from src.opd.verl_config import TeacherSpec, VerlConfigError, VerlOPDConfig
from src.utils.run_plan import CostCapExceeded, RunPlan, ThroughputModel


# ---------------------------------------------------------------------------
# run plan
# ---------------------------------------------------------------------------


def base_plan(**overrides) -> RunPlan:
    params = dict(
        run_id="b2-medical-opd-s42",
        purpose="Phase 1 B2: single Medical Teacher OPD on Qwen3-1.7B",
        baseline_id="B2",
        model="Qwen/Qwen3-1.7B",
        seed=42,
        steps=200,
        prompt_batch_size=2,
        group_size=2,
        max_prompt_tokens=512,
        max_response_tokens=768,
        checkpoint_every_steps=20,
        controller_dev_every_steps=20,
        controller_dev_samples=600,
        num_gpus=2,
        price_per_gpu_hour_rmb=2.5,
        cost_cap_rmb=60.0,
    )
    params.update(overrides)
    return RunPlan(**params)


def test_token_accounting_is_explicit_and_monotone():
    plan = base_plan()
    assert plan.sequences_per_step == 4
    assert plan.generated_tokens == 200 * 4 * 768
    assert plan.teacher_prefill_tokens == 200 * 4 * (512 + 768)
    bigger = base_plan(group_size=4)
    assert bigger.generated_tokens == 2 * plan.generated_tokens
    assert bigger.estimate_gpu_hours() > plan.estimate_gpu_hours()


def test_controller_dev_evaluations_follow_window_size():
    plan = base_plan(steps=200, controller_dev_every_steps=20)
    assert plan.controller_dev_evaluations == 10
    assert plan.controller_dev_generated_tokens == 10 * 600 * 16


def test_wall_clock_assumes_overlap_not_sum():
    plan = base_plan()
    s = plan.estimate_seconds()
    assert s["wall_clock_seconds"] == pytest.approx(
        max(s["rollout_seconds"], s["teacher_seconds"]) + s["optimizer_seconds"]
    )
    assert s["wall_clock_seconds"] < s["rollout_seconds"] + s["teacher_seconds"] + s["optimizer_seconds"]


def test_cost_cap_is_enforced():
    cheap = base_plan(steps=20)
    assert cheap.check_cost_cap() <= cheap.cost_cap_rmb
    expensive = base_plan(steps=20000, cost_cap_rmb=5.0)
    with pytest.raises(CostCapExceeded, match="exceeds cap"):
        expensive.check_cost_cap()


def test_assumed_throughput_is_always_disclosed():
    plan = base_plan()
    assert any("assumed" in a for a in plan.assumptions_used)
    measured = base_plan(
        throughput=ThroughputModel(
            rollout_tokens_per_second=850.0,
            teacher_prefill_tokens_per_second=5200.0,
            optimizer_step_seconds=0.9,
            measured=True,
            source="measured: run b2-smoke-s42 system/* metrics",
        )
    )
    assert not any("assumed" in a for a in measured.assumptions_used)
    # upper-bound and overlap caveats stay in both cases
    assert any("upper bound" in a for a in measured.assumptions_used)


def test_plan_markdown_contains_budget_and_criteria():
    plan = base_plan(
        success_criteria=["controller-dev medical accuracy improves over B0 by >= 2 points"],
        early_stop_criteria=["router reports should_stop"],
        abort_criteria=["policy entropy < 0.1 for 3 consecutive windows", "cost > cap"],
    )
    md = plan.to_markdown()
    for needle in ("Run plan:", "estimated GPU-hours", "within cap", "Assumptions", "Abort"):
        assert needle in md
    assert "policy entropy" in md


def test_plan_rejects_degenerate_values():
    with pytest.raises(ValueError, match="steps must be >= 1"):
        base_plan(steps=0)
    with pytest.raises(ValueError, match="cost_cap_rmb must be > 0"):
        base_plan(cost_cap_rmb=0)
    with pytest.raises(ValueError):
        ThroughputModel(rollout_tokens_per_second=0)


# ---------------------------------------------------------------------------
# veRL config translation
# ---------------------------------------------------------------------------


def single_teacher_config(**overrides) -> VerlOPDConfig:
    params = dict(
        student_model_path="/root/autodl-tmp/ca-opd/models/Qwen3-1.7B",
        teachers=[
            TeacherSpec(
                name="teacher_model",
                routing_key="default",
                model_path="/root/autodl-tmp/ca-opd/models/medical-teacher-merged",
            )
        ],
        teacher_gpus_per_node=1,
        teacher_nnodes=1,
    )
    params.update(overrides)
    return VerlOPDConfig(**params)


def test_single_teacher_overrides_contain_required_keys():
    cfg = single_teacher_config()
    ov = cfg.to_overrides()
    joined = "\n".join(ov)
    # LoRA + vLLM requirements
    assert "actor_rollout_ref.rollout.load_format=safetensors" in ov
    assert "actor_rollout_ref.model.target_modules=all-linear" in ov
    assert "actor_rollout_ref.model.lora_rank=32" in ov
    # OPD replaces reference-KL regularisation
    assert "actor_rollout_ref.actor.use_kl_loss=false" in ov
    assert "algorithm.use_kl_in_reward=false" in ov
    # PG OPD
    assert "distillation.enabled=true" in ov
    assert "distillation.distillation_loss.loss_mode=k1" in ov
    assert "distillation.distillation_loss.use_policy_gradient=true" in ov
    assert "distillation.distillation_loss.policy_loss_mode=vanilla" in ov
    # routing key + teacher entry
    assert "distillation.teacher_key=teacher_route" in ov
    assert "distillation.teacher_models.teacher_model.model_path=" in joined
    # teacher context window covers prompt+response+1
    assert "distillation.teacher_models.teacher_model.inference.max_model_len=1281" in ov
    # shuffling matters for multi-teacher routing
    assert "data.shuffle=true" in ov


def test_command_includes_vllm_v1_env():
    cmd = single_teacher_config().to_command()
    assert cmd.startswith("VLLM_USE_V1=1 python3 -m verl.trainer.main_ppo")


def test_dual_teacher_does_not_fit_two_gpus():
    """The core finding of ADR-0002, encoded as a test."""
    cfg = VerlOPDConfig(
        student_model_path="/models/Qwen3-1.7B",
        teachers=[
            TeacherSpec(name="medical", routing_key="medical", model_path="/models/med"),
            TeacherSpec(name="base", routing_key="base", model_path="/models/base"),
        ],
        teacher_gpus_per_node=1,  # GPU0 is the student, so only 1 GPU is left
        teacher_nnodes=1,
    )
    with pytest.raises(VerlConfigError, match="teacher pool size 1"):
        cfg.to_overrides()
    # with a third GPU it validates
    cfg.teacher_gpus_per_node = 2
    ov = cfg.to_overrides()
    assert "distillation.teacher_models.medical.key=medical" in ov
    assert "distillation.teacher_models.base.key=base" in ov


def test_default_teacher_entry_cannot_coexist_with_named_teachers():
    cfg = VerlOPDConfig(
        student_model_path="/models/s",
        teachers=[
            TeacherSpec(name="teacher_model", routing_key="medical", model_path="/models/med"),
            TeacherSpec(name="base", routing_key="base", model_path="/models/base"),
        ],
        teacher_gpus_per_node=2,
    )
    with pytest.raises(VerlConfigError, match="silently dropped"):
        cfg.validate()


def test_teacher_lora_adapter_is_rejected_with_a_pointer_to_the_adr():
    cfg = single_teacher_config(
        teachers=[
            TeacherSpec(
                name="teacher_model",
                routing_key="default",
                model_path="/models/base",
                lora_adapter_path="/adapters/medical",
            )
        ]
    )
    with pytest.raises(VerlConfigError, match="ADR-0002"):
        cfg.validate()


def test_loss_mode_and_policy_gradient_combinations():
    with pytest.raises(VerlConfigError, match="no gradient through the teacher"):
        single_teacher_config(loss_mode="k1", use_policy_gradient=False).validate()
    with pytest.raises(VerlConfigError, match="discarding the top-k"):
        single_teacher_config(loss_mode="forward_kl_topk", use_policy_gradient=True, topk=128).validate()
    # valid GKD OPD ablation
    gkd = single_teacher_config(loss_mode="forward_kl_topk", use_policy_gradient=False, topk=128)
    ov = gkd.to_overrides()
    assert "distillation.distillation_loss.topk=128" in ov
    assert "distillation.distillation_loss.use_policy_gradient=false" in ov
    with pytest.raises(VerlConfigError, match="requires topk"):
        single_teacher_config(loss_mode="forward_kl_topk", use_policy_gradient=False).validate()


def test_teacher_max_model_len_must_cover_prompt_plus_response():
    cfg = single_teacher_config(
        teachers=[TeacherSpec(name="teacher_model", routing_key="default", model_path="/m", max_model_len=1000)],
        max_prompt_tokens=512,
        max_response_tokens=768,
    )
    with pytest.raises(VerlConfigError, match="max_model_len=1000"):
        cfg.validate()


def test_duplicate_routing_keys_rejected():
    cfg = VerlOPDConfig(
        student_model_path="/models/s",
        teachers=[
            TeacherSpec(name="a", routing_key="same", model_path="/m1"),
            TeacherSpec(name="b", routing_key="same", model_path="/m2"),
        ],
        teacher_gpus_per_node=2,
    )
    with pytest.raises(VerlConfigError, match="distinct routing keys"):
        cfg.validate()


def test_student_lora_adapter_path_is_passed_through():
    cfg = single_teacher_config(lora_adapter_path="/root/autodl-tmp/ca-opd/adapters/medical-sft")
    assert "actor_rollout_ref.model.lora_adapter_path=/root/autodl-tmp/ca-opd/adapters/medical-sft" in cfg.to_overrides()
