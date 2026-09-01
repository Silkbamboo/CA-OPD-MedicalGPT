"""Chat template + loss-mask tests (docs/REPRODUCIBILITY.md §6.3: "chat template与loss mask快照测试").

The snapshot is written inline rather than to a golden file so a template change
shows up as a readable diff in review.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.data.chat import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    MCQ_INSTRUCTION,
    build_masked_example,
    char_tokenizer,
    format_mcq_question,
    template_snapshot,
)


def test_prompt_snapshot_is_stable():
    prompt = DEFAULT_TEMPLATE.render_prompt("你好")
    assert prompt == (
        "<|im_start|>system\n"
        f"{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        "你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_completion_snapshot_with_and_without_reasoning():
    assert DEFAULT_TEMPLATE.render_completion("答案") == "答案<|im_end|>"
    assert DEFAULT_TEMPLATE.render_completion("答案", reasoning="因为") == (
        "<think>\n因为\n</think>\n答案<|im_end|>"
    )


def test_render_full_is_prompt_plus_completion():
    full = DEFAULT_TEMPLATE.render_full("问", "答")
    assert full == DEFAULT_TEMPLATE.render_prompt("问") + DEFAULT_TEMPLATE.render_completion("答")


def test_system_prompt_can_be_disabled():
    prompt = DEFAULT_TEMPLATE.render_prompt("问", system_prompt=None)
    assert "system" not in prompt
    assert prompt.startswith("<|im_start|>user\n")


def test_mcq_rendering_is_deterministic_and_ordered():
    rendered = format_mcq_question("下列哪项最恰当？", ["甲", "乙", "丙"])
    assert rendered.splitlines()[0] == "下列哪项最恰当？"
    assert "A. 甲" in rendered and "B. 乙" in rendered and "C. 丙" in rendered
    assert rendered.strip().endswith(MCQ_INSTRUCTION)
    assert "/no_think" in rendered
    assert "<think>" not in rendered and "</think>" not in rendered
    # same input -> byte-identical output (no dict ordering, no shuffling)
    assert rendered == format_mcq_question("下列哪项最恰当？", ["甲", "乙", "丙"])


def test_mcq_rejects_more_options_than_letters():
    with pytest.raises(ValueError, match="too many options"):
        format_mcq_question("q", [f"opt{i}" for i in range(9)])


def test_loss_mask_covers_completion_only():
    example = build_masked_example(char_tokenizer, "问题", "答案")
    assert len(example.input_ids) == len(example.loss_mask)
    assert example.prompt_length + example.completion_length == len(example.input_ids)
    assert set(example.loss_mask[: example.prompt_length]) == {0}
    assert set(example.loss_mask[example.prompt_length :]) == {1}
    assert example.trainable_tokens() == example.completion_length
    # the prompt text itself must never appear inside the trainable region
    trainable_text = example.completion_text
    assert "<|im_start|>user" not in trainable_text
    assert DEFAULT_SYSTEM_PROMPT not in trainable_text


def test_loss_mask_includes_reasoning_when_present():
    without = build_masked_example(char_tokenizer, "问题", "答案")
    with_cot = build_masked_example(char_tokenizer, "问题", "答案", reasoning="较长的推理过程")
    assert with_cot.trainable_tokens() > without.trainable_tokens()
    assert with_cot.prompt_length == without.prompt_length


def test_masked_example_boundary_is_exact_for_char_tokenizer():
    example = build_masked_example(char_tokenizer, "A", "B", system_prompt=None)
    prompt_text = DEFAULT_TEMPLATE.render_prompt("A", system_prompt=None)
    assert example.prompt_length == len(prompt_text)
    assert example.input_ids[: example.prompt_length] == char_tokenizer(prompt_text)


def test_template_snapshot_keys():
    snap = template_snapshot()
    assert set(snap) == {"name", "prompt", "completion", "mcq_prompt"}
    assert snap["name"] == "qwen3_chatml"
    assert snap["mcq_prompt"].endswith("<|im_start|>assistant\n")


@pytest.mark.needs_model
def test_matches_real_tokenizer():
    """Cross-check the segment boundary against a real tokenizer.

    Runs when ``CA_OPD_MODEL_PATH`` points at local tokenizer artifacts. P2 can
    satisfy this contract without model weights; a checkout with no audited
    tokenizer still skips rather than silently passing.
    """
    model_path = os.environ.get("CA_OPD_MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        pytest.skip("CA_OPD_MODEL_PATH not set; real-tokenizer artifact is unavailable")
    from transformers import AutoTokenizer  # local import: optional dependency path

    tok = AutoTokenizer.from_pretrained(model_path)

    def encode(text: str):
        return tok(text, add_special_tokens=False)["input_ids"]

    example = build_masked_example(encode, "患者主诉咳嗽三周，应如何处理？", "建议尽快就医。")
    assert example.prompt_length == len(encode(example.prompt_text))
    assert example.input_ids == encode(example.prompt_text) + encode(example.completion_text)
    assert sum(example.loss_mask) == len(encode(example.completion_text))
