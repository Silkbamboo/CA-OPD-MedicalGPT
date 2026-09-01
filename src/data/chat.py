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
from typing import Mapping

Tokenizer = Callable[[str], Sequence[int]]

DEFAULT_SYSTEM_PROMPT = "你是一名严谨的中文医疗助手。回答需明确不确定性，并在必要时建议就医。"

#: Fixed instruction used for every MCQ evaluation (medical and general alike).
MCQ_INSTRUCTION = "请阅读题目并从选项中选择唯一正确答案，只回答选项字母。 /no_think"


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


def build_masked_example_nonthinking(
    tokenizer: Tokenizer,
    user_content: str,
    answer: str,
    reasoning: Optional[str] = None,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    template: ChatTemplate = DEFAULT_TEMPLATE,
    *,
    enable_thinking: bool,
) -> MaskedExample:
    """Build a Qwen3 Data Protocol v2 SFT example without think markup.

    Medical-O1 reasoning may remain visible as ordinary assistant text, but the
    Qwen3 thinking mode and literal ``<think>`` tags are prohibited. Prompt and
    completion are still tokenized separately so only assistant text plus the
    terminating ``<|im_end|>`` token(s) contribute to the loss.
    """

    if enable_thinking is not False:
        raise ValueError("Data Protocol v2 requires Qwen3 non-thinking mode")
    parts = [value for value in (system_prompt, user_content, reasoning, answer) if value]
    if any("<think>" in value.casefold() or "</think>" in value.casefold() for value in parts):
        raise ValueError("Data Protocol v2 cannot retain <think> tags")
    prompt_text = template.render_prompt(user_content, system_prompt)
    body_parts = [str(value).strip() for value in (reasoning, answer) if value and str(value).strip()]
    completion_text = "\n".join(body_parts) + template.assistant_suffix
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


# -- Data Protocol v2 Qwen3 non-thinking contract --------------------------

_ALLOWED_CHAT_ROLES = frozenset({"assistant", "system", "tool", "user"})


def render_qwen_chat(
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render the frozen Qwen im_start/im_end envelope for CPU fixtures."""

    chunks: List[str] = []
    for message in messages:
        role = str(message["role"])
        if role not in _ALLOWED_CHAT_ROLES:
            raise ValueError(f"unsupported chat role: {role}")
        chunks.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def render_qwen3_nonthinking(
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool = False,
) -> str:
    """Render the Data Protocol v2 non-thinking fixture contract.

    Real tokenizer equivalence remains a cache/GPU-stage audit. This structural
    snapshot refuses thinking mode and pre-existing think tags so no smoke or
    formal v2 input can silently retain chain-of-thought markup.
    """

    if enable_thinking is not False:
        raise ValueError("Data Protocol v2 requires Qwen3 non-thinking mode")
    if any(
        "<think>" in str(message.get("content", "")).casefold()
        or "</think>" in str(message.get("content", "")).casefold()
        for message in messages
    ):
        raise ValueError("Data Protocol v2 cannot retain <think> tags")
    return render_qwen_chat(messages, add_generation_prompt=add_generation_prompt)


def sft_assistant_eos_loss_mask(
    token_roles: Sequence[str],
    *,
    token_ids: Sequence[int],
    eos_token_id: int,
    attention_mask: Optional[Sequence[int]] = None,
) -> List[int]:
    """Mask SFT tokens so only assistant content and its terminating EOS train."""

    if attention_mask is None:
        attention_mask = [1] * len(token_roles)
    if not (len(token_roles) == len(token_ids) == len(attention_mask)):
        raise ValueError("roles, token IDs and attention mask must have equal length")
    mask: List[int] = []
    for role, token_id, attended in zip(
        token_roles, token_ids, attention_mask, strict=True
    ):
        if role == "assistant_eos" and token_id != eos_token_id:
            raise ValueError("assistant_eos role must carry eos_token_id")
        mask.append(
            int(bool(attended) and (role == "assistant" or role == "assistant_eos"))
        )
    return mask
