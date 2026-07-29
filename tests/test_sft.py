"""SFT tests (CPU only): config schema, rendering, assistant-only mask, length filtering.

The real training path needs transformers >= 4.51 and a GPU, so it is not exercised
here. What *is* exercised is everything that decides whether the GPU run will be
correct: which samples survive, how they are rendered, and where the loss mask falls.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.build_splits import build_splits
from src.data.chat import DEFAULT_TEMPLATE, char_tokenizer
from src.sft.train import SFT_SCHEMA, build_examples, dry_run
from src.utils.config import ConfigError, load_config

DATA_CONFIG = Path("configs/data/fixture_cpu.yaml")
SFT_CONFIG = Path("configs/sft/qwen3_1_7b_medical.yaml")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sft_data")
    build_splits(DATA_CONFIG, output_dir=out)
    return out


def write_sft_config(tmp_path: Path, data_dir: Path, **overrides) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    cfg["data"]["data_dir"] = str(data_dir)
    for section, values in overrides.items():
        cfg[section].update(values)
    path = tmp_path / "sft.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_shipped_config_matches_schema():
    cfg = load_config(SFT_CONFIG, SFT_SCHEMA)
    assert cfg["run"]["baseline_id"] == "B1"
    assert cfg["lora"]["rank"] == 32, "teacher LoRA rank must match the OPD student"
    assert cfg["lora"]["target_modules"] == "all-linear"
    assert cfg["optim"]["save_only_model"] is True, "no optimizer.pt (disk policy)"
    assert cfg["data"]["drop_longer_than_max_seq"] is True


def test_config_rejects_unknown_key(tmp_path, data_dir):
    cfg = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
    cfg["lora"]["ranks"] = 64  # typo
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config keys"):
        load_config(path, SFT_SCHEMA)


def test_dry_run_reports_token_statistics(tmp_path, data_dir):
    report = dry_run(write_sft_config(tmp_path, data_dir))
    assert report.num_samples == 20  # fixture medical_sft allocation
    assert report.num_dropped_too_long == 0
    assert report.total_tokens == report.prompt_tokens + report.trainable_tokens
    assert 0.0 < report.trainable_ratio < 1.0
    for key in ("min", "p50", "p90", "p99", "max", "mean"):
        assert report.length_percentiles[key] > 0
    assert report.length_percentiles["min"] <= report.length_percentiles["max"]


def test_dry_run_example_shows_prompt_masked_and_completion_trained(tmp_path, data_dir):
    report = dry_run(write_sft_config(tmp_path, data_dir))
    assert report.example_prompt.startswith("<|im_start|>system")
    assert report.example_prompt.endswith("<|im_start|>assistant\n")
    assert report.example_completion.endswith("<|im_end|>")
    assert "prompt tokens masked out" in report.example_mask_summary
    assert "completion tokens trained" in report.example_mask_summary


def test_reasoning_toggle_changes_trainable_tokens(tmp_path, data_dir):
    with_cot = dry_run(write_sft_config(tmp_path / "a", data_dir, data={"include_reasoning": True}))
    without = dry_run(write_sft_config(tmp_path / "b", data_dir, data={"include_reasoning": False}))
    assert with_cot.trainable_tokens > without.trainable_tokens
    assert with_cot.prompt_tokens == without.prompt_tokens
    assert "<think>" in with_cot.example_completion
    assert "<think>" not in without.example_completion


def test_over_long_samples_are_dropped_not_truncated(tmp_path, data_dir):
    """Cap at the observed median so some samples are dropped and some survive."""
    full = dry_run(write_sft_config(tmp_path / "full", data_dir))
    cap = full.length_percentiles["p50"]
    report = dry_run(write_sft_config(tmp_path / "capped", data_dir, model={"max_seq_length": cap}))
    assert 0 < report.num_dropped_too_long < full.num_samples
    assert report.num_samples + report.num_dropped_too_long == full.num_samples
    assert report.length_percentiles["max"] <= cap, "no surviving example may exceed the cap"


def test_drop_disabled_turns_over_long_samples_into_an_error(tmp_path, data_dir):
    path = write_sft_config(
        tmp_path / "strict", data_dir, model={"max_seq_length": 64},
        data={"drop_longer_than_max_seq": False},
    )
    with pytest.raises(ValueError, match="max_seq_length"):
        dry_run(path)


def test_examples_use_the_same_template_as_evaluation(tmp_path, data_dir):
    """SFT and eval must render prompts identically, or gains are partly formatting."""
    cfg = load_config(write_sft_config(tmp_path, data_dir), SFT_SCHEMA)
    examples, _ = build_examples(cfg, char_tokenizer)
    rendered = examples[0].prompt_text
    # reconstruct with the eval-side template call
    body = rendered.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    assert rendered == DEFAULT_TEMPLATE.render_prompt(body, cfg["data"]["system_prompt"])


def test_max_samples_limits_dataset(tmp_path, data_dir):
    report = dry_run(write_sft_config(tmp_path, data_dir, data={"max_samples": 5}))
    assert report.num_samples <= 5
