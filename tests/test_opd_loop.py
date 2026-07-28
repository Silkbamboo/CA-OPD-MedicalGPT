"""OPD closed-loop integration tests (agent.md §6.4 / CLAUDE.md §6 "系统集成").

Coverage map:
* small model / tiny data completes a full step   -> test_dry_run_completes_and_writes_artifacts
* checkpoint save+resume keeps logprobs identical -> test_resume_reproduces_uninterrupted_run
* rollout logprobs == training-time logprobs      -> test_rollout_logprobs_match_forward_recompute
* teacher performs forward only                   -> test_teacher_is_forward_only
* base/medical routes return the right identity   -> test_teacher_routes_are_distinguishable
* run artifacts required by PROJECT_PLAN §15      -> test_dry_run_completes_and_writes_artifacts
* metric names stay inside the frozen vocabulary  -> test_metrics_file_uses_frozen_metric_names
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.opd.core import build_opd_batch, selected_token_logprobs
from src.opd.loop import (
    BASE,
    MEDICAL,
    SyntheticControllerDev,
    TeacherRegistry,
    build_toy_teachers,
    make_synthetic_prompt_pools,
    run_loop,
    sample_completions,
)
from src.opd.toy_lm import ToyCausalLM, ToyLMConfig
from src.utils.config import ConfigError
from src.utils.io import read_jsonl
from src.utils.metrics import METRIC_NAMES

CONFIG_PATH = Path("configs/opd/dev_cpu.yaml")


def write_config(tmp_path: Path, **overrides) -> Path:
    """Copy the dry-run config with nested overrides like ``{"optim": {"max_steps": 4}}``."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            for k, v in values.items():
                if isinstance(v, dict) and isinstance(cfg[section].get(k), dict):
                    cfg[section][k].update(v)
                else:
                    cfg[section][k] = v
        else:
            cfg[section] = values
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


def test_rollout_logprobs_match_forward_recompute():
    """The sampler's logprob must equal a later forward pass on the same tokens.

    This is the CPU analogue of the vLLM-sampler-vs-trainer mismatch that
    silently breaks on-policy methods (PROJECT_PLAN.md §9: "Student/Teacher 在
    相同 token 和自回归上下文上比较").
    """
    torch.manual_seed(0)
    model = ToyCausalLM(ToyLMConfig())
    prompts = [[5, 6, 7], [8, 9, 10]]
    gen = torch.Generator().manual_seed(123)
    rollout = sample_completions(
        model, prompts, max_new_tokens=5, temperature=1.0, eos_token_id=1, generator=gen
    )
    batch = build_opd_batch(prompts, rollout.completions, pad_token_id=0, eos_token_id=1)
    with torch.no_grad():
        lp = selected_token_logprobs(model(batch.input_ids, batch.attention_mask), batch)

    for i, sampled in enumerate(rollout.sampling_logprobs):
        prompt_len = len(prompts[i])
        for j, expected in enumerate(sampled):
            # completion token j sits at absolute index prompt_len + j, whose
            # logprob lives at shifted index prompt_len + j - 1
            got = float(lp[i, prompt_len + j - 1])
            assert got == pytest.approx(expected, abs=1e-5), (i, j, got, expected)


def test_rollout_stops_at_eos():
    torch.manual_seed(0)
    model = ToyCausalLM(ToyLMConfig(vocab_size=8))
    with torch.no_grad():  # force EOS to dominate
        model.logit_bias[1] += 50.0
    rollout = sample_completions(model, [[3, 4]], max_new_tokens=10, temperature=1.0, eos_token_id=1)
    assert rollout.completions[0] == [1]


def test_rollout_leaves_model_mode_untouched():
    model = ToyCausalLM(ToyLMConfig())
    model.train()
    sample_completions(model, [[3, 4]], max_new_tokens=2, temperature=1.0, eos_token_id=1)
    assert model.training is True


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------


def test_teacher_registry_rejects_trainable_teacher():
    trainable = ToyCausalLM(ToyLMConfig())
    frozen = ToyCausalLM(ToyLMConfig())
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable parameters"):
        TeacherRegistry({BASE: trainable, MEDICAL: frozen})


def test_teacher_registry_requires_both_routes():
    frozen = ToyCausalLM(ToyLMConfig())
    for p in frozen.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="missing route"):
        TeacherRegistry({MEDICAL: frozen})


def test_teacher_routes_are_distinguishable():
    """Base and Medical routes must return different logprobs and identities."""
    cfg = ToyLMConfig()
    teachers = build_toy_teachers(cfg, seed=1, medical_bias_token=17, medical_bias_strength=4.0)
    batch = build_opd_batch([[5, 6]], [[17, 17]], pad_token_id=0, eos_token_id=1, domains=("medical",))
    base_lp = teachers.score(BASE, batch)
    med_lp = teachers.score(MEDICAL, batch)
    assert teachers.identity(MEDICAL) == MEDICAL
    assert teachers.identity(BASE) == BASE
    with pytest.raises(KeyError, match="unknown teacher_id"):
        teachers.identity("radiology")
    # the medical adapter genuinely prefers its token
    assert float(med_lp[0, -1]) > float(base_lp[0, -1])
    assert teachers.call_counts == {BASE: 1, MEDICAL: 1}


def test_teacher_is_forward_only():
    """Scoring must not create gradients or move teacher weights."""
    cfg = ToyLMConfig()
    teachers = build_toy_teachers(cfg, seed=2, medical_bias_token=17, medical_bias_strength=2.0)
    before = copy.deepcopy(teachers._teachers[MEDICAL].state_dict())  # noqa: SLF001 - test introspection
    batch = build_opd_batch([[5, 6]], [[17, 18]], pad_token_id=0, eos_token_id=1, domains=("medical",))
    lp = teachers.score(MEDICAL, batch)
    assert not lp.requires_grad
    assert lp.grad_fn is None
    after = teachers._teachers[MEDICAL].state_dict()  # noqa: SLF001
    for key in before:
        assert torch.equal(before[key], after[key]), f"teacher weight {key} changed during scoring"


# ---------------------------------------------------------------------------
# synthetic data pools
# ---------------------------------------------------------------------------


def test_synthetic_pools_are_disjoint_and_avoid_special_tokens():
    pools = make_synthetic_prompt_pools(
        num_medical=5, num_general=5, prompt_length=4, vocab_size=32, pad_token_id=0, eos_token_id=1, seed=7
    )
    med_tokens = {t for p in pools["medical"] for t in p}
    gen_tokens = {t for p in pools["general"] for t in p}
    assert med_tokens.isdisjoint(gen_tokens)
    assert 0 not in med_tokens | gen_tokens
    assert 1 not in med_tokens | gen_tokens


def test_synthetic_controller_dev_is_marked_and_reactive():
    dev = SyntheticControllerDev(0.4, 0.05, 0.5, 0.03)
    assert dev.allows_control_decisions() is True
    assert dev.is_synthetic is True
    m1, g1 = dev.evaluate(medical_fraction=1.0)
    assert m1 > 0.4 and g1 < 0.5  # all-medical window: medical up, general down
    m2, g2 = dev.evaluate(medical_fraction=0.0)
    assert m2 == pytest.approx(m1) and g2 > g1  # all-base window: general recovers


# ---------------------------------------------------------------------------
# end-to-end dry run
# ---------------------------------------------------------------------------


def test_dry_run_completes_and_writes_artifacts(tmp_path):
    out = tmp_path / "run"
    summary = run_loop(CONFIG_PATH, output_dir=out, max_steps_override=8)

    assert summary["steps_completed"] == 8
    assert summary["implementation"] == "cpu_reference"
    for name in ("config.yaml", "metadata.json", "metrics.jsonl", "summary.json"):
        assert (out / name).exists(), f"missing run artifact {name}"
    ckpts = sorted((out / "checkpoints").iterdir())
    assert ckpts, "no checkpoint written"
    # keep_last=2 discipline
    assert len(ckpts) <= 2

    metadata = json.loads((out / "metadata.json").read_text())
    for key in ("run_id", "git_sha", "seed", "packages", "hardware", "config_path", "baseline_id"):
        assert key in metadata, f"metadata missing §15 field {key}"

    # both teachers were used at least once given p_min > 0
    assert summary["teacher_counts"][MEDICAL] + summary["teacher_counts"][BASE] == 8
    assert summary["controller_dev"]["mode"] == "synthetic"
    assert "NOT a model evaluation" in summary["controller_dev"]["warning"]


def test_metrics_file_uses_frozen_metric_names(tmp_path):
    out = tmp_path / "run"
    run_loop(CONFIG_PATH, output_dir=out, max_steps_override=4)
    reserved = {"step", "wall_time", "run_id", "domain", "phase", "window"}
    rows = read_jsonl(out / "metrics.jsonl")
    assert rows
    for row in rows:
        unknown = set(row) - METRIC_NAMES - reserved
        assert not unknown, f"metrics.jsonl contains unknown keys: {sorted(unknown)}"
    train_rows = [r for r in rows if r.get("phase") == "train"]
    assert len(train_rows) == 4
    for row in train_rows:
        assert "opd/reverse_kl_mean" in row
        assert "ppo/ratio_mean" in row
        assert "opd/kl_scale" in row
        assert row["opd/teacher_id"] in {MEDICAL, BASE}


def test_kl_safety_scale_is_never_above_one(tmp_path):
    out = tmp_path / "run"
    run_loop(CONFIG_PATH, output_dir=out, max_steps_override=8)
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if r.get("phase") == "train"]
    assert all(r["opd/kl_scale"] <= 1.0 + 1e-9 for r in rows)


def test_router_windows_recorded_in_summary(tmp_path):
    out = tmp_path / "run"
    summary = run_loop(CONFIG_PATH, output_dir=out, max_steps_override=8)
    windows = summary["router_windows"]
    assert len(windows) == 2  # window_steps=4 -> two windows in 8 steps
    for w in windows:
        assert 0.2 - 1e-9 <= w["p_medical"] <= 0.8 + 1e-9
        assert w["state"] in {"pursue_medical", "recover_general"}


def test_resume_reproduces_uninterrupted_run(tmp_path):
    """Steps 5-8 of a resumed run must match steps 5-8 of a single 8-step run."""
    full_dir = tmp_path / "full"
    run_loop(CONFIG_PATH, output_dir=full_dir, max_steps_override=8)
    full_rows = {r["step"]: r for r in read_jsonl(full_dir / "metrics.jsonl") if r.get("phase") == "train"}

    part_dir = tmp_path / "part"
    run_loop(CONFIG_PATH, output_dir=part_dir, max_steps_override=4)
    ckpt = part_dir / "checkpoints" / "step-4"
    assert ckpt.exists()

    resumed_dir = tmp_path / "resumed"
    run_loop(CONFIG_PATH, output_dir=resumed_dir, resume_from=ckpt, max_steps_override=8)
    resumed_rows = {r["step"]: r for r in read_jsonl(resumed_dir / "metrics.jsonl") if r.get("phase") == "train"}

    assert set(resumed_rows) == {5, 6, 7, 8}
    for step in (5, 6, 7, 8):
        for key in ("train/loss", "opd/reverse_kl_mean", "opd/advantage_mean", "ppo/ratio_mean", "opd/kl_scale"):
            assert resumed_rows[step][key] == pytest.approx(full_rows[step][key], rel=1e-6, abs=1e-9), (
                f"step {step} metric {key} diverged after resume: "
                f"{resumed_rows[step][key]} vs {full_rows[step][key]}"
            )
        assert resumed_rows[step]["opd/teacher_id"] == full_rows[step]["opd/teacher_id"]


def test_single_teacher_router_uses_only_medical(tmp_path):
    """B2 (Medical OPD) baseline shape: one teacher, same loop."""
    cfg = write_config(tmp_path, router={"kind": "single_teacher", "single_teacher_id": MEDICAL})
    summary = run_loop(cfg, output_dir=tmp_path / "run", max_steps_override=6)
    assert summary["teacher_counts"][MEDICAL] == 6
    assert summary["teacher_counts"][BASE] == 0


def test_fixed_ratio_router_runs_and_records_ratio(tmp_path):
    """B4 (IDT 1:1) baseline shape."""
    cfg = write_config(tmp_path, router={"kind": "fixed_ratio", "fixed_p_medical": 0.5})
    summary = run_loop(cfg, output_dir=tmp_path / "run", max_steps_override=10)
    assert summary["teacher_counts"][MEDICAL] + summary["teacher_counts"][BASE] == 10
    assert summary["realised_medical_fraction"] is not None


def test_early_stop_triggers_and_is_recorded(tmp_path):
    """Both objectives satisfied + no improvement -> stop, with a reason."""
    cfg = write_config(
        tmp_path,
        controller_dev={"medical_start": 0.95, "medical_gain_per_window": 0.0, "general_start": 0.95,
                        "general_loss_per_medical_window": 0.0},
        router={"config": {"medical_target": 0.5, "early_stop_patience": 1, "window_steps": 2}},
        optim={"max_steps": 20},
    )
    summary = run_loop(cfg, output_dir=tmp_path / "run")
    assert summary["stopped_early"] is True
    assert "no medical improvement" in summary["stop_reason"]
    assert summary["steps_completed"] < 20


def test_config_validation_rejects_unknown_and_missing_keys(tmp_path):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["opd"]["bta"] = 1.0  # typo instead of beta
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config keys"):
        run_loop(bad, output_dir=tmp_path / "run")

    cfg2 = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    del cfg2["opd"]["beta"]
    bad2 = tmp_path / "bad2.yaml"
    bad2.write_text(yaml.safe_dump(cfg2), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required config key"):
        run_loop(bad2, output_dir=tmp_path / "run2")


def test_config_rejects_out_of_range_probability(tmp_path):
    cfg = write_config(tmp_path, router={"config": {"p_max": 1.5}})
    with pytest.raises(ConfigError, match="p_max"):
        run_loop(cfg, output_dir=tmp_path / "run")
