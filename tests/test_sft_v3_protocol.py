from __future__ import annotations

import pytest

from src.sft.v3 import (
    SFTV3Kind,
    build_sft_v3_smoke_rank_rows,
    build_task_balanced_rank_rows,
    render_sft_v3_row,
    sft_v3_task_counts_through_step,
    task_for_optimizer_step,
    validate_sft_v3_schedule,
)
from src.sft.train_ddp import load_ddp_config


class CharTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        assert tokenize is False and enable_thinking is False
        text = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [100 + ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(int(value) - 100) for value in token_ids)


def _cmb(index: int, label: str = "B") -> dict:
    return {
        "sample_id": f"cmb-{index}",
        "target_role": "medical_sft_train",
        "sft_v3_kind": "cmb_mcq_letter",
        "question": "选择",
        "options": ["甲", "乙", "丙", "丁", "戊"],
        "answer_idx": label,
        "content_hash": f"ch-{index}",
        "group_id": f"cg-{index}",
        "upstream_split": "train",
    }


def _o1(index: int) -> dict:
    return {
        "sample_id": f"o1-{index}",
        "target_role": "medical_sft_train",
        "sft_v3_kind": "medical_o1_response",
        "question": "问",
        "answer": "直接回答",
        "content_hash": f"oh-{index}",
        "group_id": f"og-{index}",
        "upstream_split": "train",
    }


def test_cmb_target_is_one_candidate_token_plus_eos_without_prefix_or_option() -> None:
    tokenizer = CharTokenizer()
    example = render_sft_v3_row(
        _cmb(0), tokenizer=tokenizer, max_seq_length=2048, system_prompt="system"
    )

    assert example is not None
    assert example.target_text == "B"
    assert example.input_ids[example.prompt_length :] == [166, tokenizer.eos_token_id]
    assert example.loss_weights[example.prompt_length :] == [1.0, 1.0]
    assert all(value == 0 for value in example.loss_weights[: example.prompt_length])
    assert "答案：" not in example.target_text
    assert "乙" not in example.target_text
    assert example.segment_token_counts == {"answer": 1, "reasoning": 0, "eos": 1}


def test_medical_o1_target_is_response_only_and_rejects_thinking_or_reasoning_export() -> None:
    tokenizer = CharTokenizer()
    example = render_sft_v3_row(
        _o1(0), tokenizer=tokenizer, max_seq_length=2048, system_prompt="system"
    )
    assert example is not None
    assert example.target_text == "直接回答"
    assert "分析" not in example.target_text and "<think>" not in example.target_text

    with pytest.raises(ValueError, match="reasoning"):
        render_sft_v3_row(
            {**_o1(1), "reasoning": "不应导出"},
            tokenizer=tokenizer,
            max_seq_length=2048,
            system_prompt="system",
        )


def test_sft_v3_refuses_final_and_confirmation_roles() -> None:
    for role in ("medical_final_test", "medical_teacher_confirmation_dev"):
        with pytest.raises(PermissionError):
            render_sft_v3_row(
                {**_cmb(0), "target_role": role},
                tokenizer=CharTokenizer(),
                max_seq_length=2048,
                system_prompt="system",
            )


def test_task_schedule_is_strict_three_to_one_for_600_steps() -> None:
    values = [task_for_optimizer_step(step) for step in range(600)]
    assert values[:8] == [
        SFTV3Kind.CMB,
        SFTV3Kind.CMB,
        SFTV3Kind.CMB,
        SFTV3Kind.MEDICAL_O1,
    ] * 2
    assert values.count(SFTV3Kind.CMB) == 450
    assert values.count(SFTV3Kind.MEDICAL_O1) == 150
    assert validate_sft_v3_schedule(total_steps=600, checkpoints=[150, 300, 450, 600])[
        "global_exposures"
    ] == {"cmb": 7200, "medical_o1": 2400}
    assert sft_v3_task_counts_through_step(150) == {"cmb": 113, "medical_o1": 37}
    assert sft_v3_task_counts_through_step(300) == {"cmb": 225, "medical_o1": 75}
    assert sft_v3_task_counts_through_step(450) == {"cmb": 338, "medical_o1": 112}
    assert sft_v3_task_counts_through_step(600) == {"cmb": 450, "medical_o1": 150}


def test_task_balanced_rank_rows_cover_each_source_once_and_resume_without_drift() -> None:
    rows = [_cmb(index) for index in range(48)] + [_o1(index) for index in range(16)]
    rank0 = build_task_balanced_rank_rows(
        rows, rank=0, world_size=2, seed=42, accumulation_steps=2
    )
    rank1 = build_task_balanced_rank_rows(
        rows, rank=1, world_size=2, seed=42, accumulation_steps=2
    )
    all_ids = [row["sample_id"] for row in rank0 + rank1]
    assert len(all_ids) == 64 and len(set(all_ids)) == 64
    assert [row["sft_v3_kind"] for row in rank0[:8]] == [
        "cmb_mcq_letter",
        "cmb_mcq_letter",
        "cmb_mcq_letter",
        "cmb_mcq_letter",
        "cmb_mcq_letter",
        "cmb_mcq_letter",
        "medical_o1_response",
        "medical_o1_response",
    ]
    resumed = build_task_balanced_rank_rows(
        rows,
        rank=0,
        world_size=2,
        seed=42,
        accumulation_steps=2,
        start_optimizer_step=2,
    )
    assert [row["sample_id"] for row in resumed] == [row["sample_id"] for row in rank0[4:]]


def test_gpu_smoke_covers_one_complete_task_period_per_rank() -> None:
    rows = [_cmb(index) for index in range(48)] + [_o1(index) for index in range(16)]
    rank0 = build_sft_v3_smoke_rank_rows(
        rows, rank=0, world_size=2, seed=42, accumulation_steps=2
    )
    rank1 = build_sft_v3_smoke_rank_rows(
        rows, rank=1, world_size=2, seed=42, accumulation_steps=2
    )
    assert len(rank0) == len(rank1) == 8
    for values in (rank0, rank1):
        assert [row["sft_v3_kind"] for row in values] == [
            "cmb_mcq_letter",
            "cmb_mcq_letter",
            "cmb_mcq_letter",
            "cmb_mcq_letter",
            "cmb_mcq_letter",
            "cmb_mcq_letter",
            "medical_o1_response",
            "medical_o1_response",
        ]
    assert {row["sample_id"] for row in rank0}.isdisjoint(
        {row["sample_id"] for row in rank1}
    )


def test_formal_sft_v3_ddp_config_freezes_only_supervision_and_data_changes() -> None:
    config = load_ddp_config("configs/public/sft_v3.recorded.yaml")
    assert config["data"]["supervision_version"] == "mcq_dominant_task_balanced_v3"
    assert config["data"]["include_reasoning"] is False
    assert config["optim"]["max_steps"] == 600
    assert config["optim"]["per_device_batch_size"] == 1
    assert config["optim"]["gradient_accumulation_steps"] == 8
    assert config["distributed"]["world_size"] == 2
    assert config["distributed"]["device_map"] == "none"
    assert config["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "all-linear",
    }


def test_sft_v3_checkpoint_screen_freezes_four_steps_and_no_more_epochs() -> None:
    from src.eval import p3_6_checkpoint_screen as checkpoint_screen

    assert checkpoint_screen.screen.EXPECTED_STEPS == (150, 300, 450, 600)
    assert checkpoint_screen.screen.EXPECTED_OPTIMIZER_STEPS == 600
    assert checkpoint_screen.screen.EXPECTED_RECORDS == 9600
    assert checkpoint_screen.screen.select_candidate({150: 228, 300: 230, 450: 230, 600: 229}) == 300
    assert checkpoint_screen.screen.epoch_two_allowed({150: 220, 300: 221, 450: 222, 600: 227}) is False
