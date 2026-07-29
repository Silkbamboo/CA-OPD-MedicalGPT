"""Chat template rendering and assistant-only loss masking.

Two things must be true and stay true across the whole project:

1. SFT, OPD rollout and evaluation all render prompts with the **same** template.
   A prompt-format difference between training and evaluation is one of the
   easiest ways to fake a metric change, so the template lives in one place and
   is snapshot-tested.
2. The SFT loss covers assistant content only. The mask is built from explicit
   ``(text, trainable)`` segments, and the tokenizer is injected, which makes the
   mask verifiable on CPU with a trivial character tokenizer and later
   cross-checked against the real Qwen3 tokenizer
   (``tests/test_chat_template.py::test_matches_real_tokenizer``, skipped unless
   ``CA_OPD_MODEL_PATH`` is set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Tokenizer = Callable[[str], Sequence[int]]

DEFAULT_SYSTEM_PROMPT = "你是一名严谨的中文医疗助手。回答需明确不确定性，并在必要时建议就医。"

#: Fixed instruction used for every MCQ evaluation (medical and general alike).
MCQ_INSTRUCTION = "请阅读题目并从选项中选择唯一正确答案，只回答选项字母。"


@dataclass(frozen=True)
class ChatTemplate:
    """Qwen-style ChatML template.

    Kept as data rather than calling ``tokenizer.apply_chat_template`` directly
    so that (a) Phase 0 runs without any tokenizer installed, and (b) the exact
    string used in every run is recorded in the config instead of depending on
    which transformers version shipped which template.
    """

    name: str = "qwen3_chatml"
    system_prefix: str = "<|im_start|>system\n"
    system_suffix: str = "<|im_end|>\n"
    user_prefix: str = "<|im_start|>user\n"
    user_suffix: str = "<|im_end|>\n"
    assistant_prefix: str = "<|im_start|>assistant\n"
    assistant_suffix: str = "<|im_end|>"
    think_open: str = "<think>"
    think_close: str = "</think>"

    def render_prompt(self, user_content: str, system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT) -> str:
        """Everything up to and including the assistant turn opener."""
        parts: List[str] = []
        if system_prompt:
            parts.append(f"{self.system_prefix}{system_prompt}{self.system_suffix}")
        parts.append(f"{self.user_prefix}{user_content}{self.user_suffix}")
        parts.append(self.assistant_prefix)
        return "".join(parts)

    def render_completion(self, answer: str, reasoning: Optional[str] = None) -> str:
        """Assistant content, optionally wrapped with an explicit thinking block."""
        body = answer.strip()
        if reasoning:
            body = f"{self.think_open}\n{reasoning.strip()}\n{self.think_close}\n{body}"
        return f"{body}{self.assistant_suffix}"

    def render_full(
        self,
        user_content: str,
        answer: str,
        reasoning: Optional[str] = None,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        return self.render_prompt(user_content, system_prompt) + self.render_completion(answer, reasoning)


DEFAULT_TEMPLATE = ChatTemplate()


def format_mcq_question(question: str, options: Sequence[str], letters: str = "ABCDEFGH") -> str:
    """Render an MCQ deterministically: fixed instruction, fixed option order."""
    if len(options) > len(letters):
        raise ValueError(f"too many options ({len(options)}) for letters {letters!r}")
    lines = [question.strip(), ""]
    for letter, option in zip(letters, options):
        lines.append(f"{letter}. {str(option).strip()}")
    lines.append("")
    lines.append(MCQ_INSTRUCTION)
    return "\n".join(lines)


@dataclass
class MaskedExample:
    """Token ids plus the assistant-only training mask."""

    input_ids: List[int]
    loss_mask: List[int]
    prompt_length: int
    completion_length: int
    prompt_text: str
    completion_text: str

    def trainable_tokens(self) -> int:
        return sum(self.loss_mask)


def build_masked_example(
    tokenizer: Tokenizer,
    user_content: str,
    answer: str,
    reasoning: Optional[str] = None,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    template: ChatTemplate = DEFAULT_TEMPLATE,
) -> MaskedExample:
    """Tokenise prompt and completion separately and mask the prompt out.

    Segment-wise tokenisation (rather than tokenising the full string and then
    guessing a boundary) is what makes the boundary exact: the prompt tokens are
    literally a different list from the completion tokens.
    """
    prompt_text = template.render_prompt(user_content, system_prompt)
    completion_text = template.render_completion(answer, reasoning)
    prompt_ids = list(tokenizer(prompt_text))
    completion_ids = list(tokenizer(completion_text))
    if not prompt_ids:
        raise ValueError("prompt tokenised to zero tokens")
    if not completion_ids:
        raise ValueError("completion tokenised to zero tokens")
    return MaskedExample(
        input_ids=prompt_ids + completion_ids,
        loss_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        prompt_text=prompt_text,
        completion_text=completion_text,
    )


def template_snapshot(template: ChatTemplate = DEFAULT_TEMPLATE) -> Dict[str, str]:
    """Canonical rendering used by the snapshot test and recorded in run configs."""
    return {
        "name": template.name,
        "prompt": template.render_prompt("患者主诉咳嗽三周，应如何处理？"),
        "completion": template.render_completion("建议尽快就医。", reasoning="持续咳嗽超过三周属于红旗征象。"),
        "mcq_prompt": template.render_prompt(format_mcq_question("下列哪项最恰当？", ["甲", "乙"])),
    }


def char_tokenizer(text: str) -> List[int]:
    """Deterministic character-level tokenizer for tests and CPU dry-runs."""
    return [ord(ch) % 4096 for ch in text]
