"""Answer-first segmented supervision for the P3.4 Medical SFT-v2 run.

This module is deliberately model-agnostic.  It renders and tokenizes explicit
segments so prompt, answer, reasoning and EOS weights are auditable before a
GPU model is loaded.  Over-length examples fail closed; no tail truncation is
performed because that could silently remove the final supervision contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, format_mcq_question


@dataclass(frozen=True)
class SupervisionWeights:
    """Frozen SFT-v2 token weights."""

    answer: float
    reasoning: float
    eos: float

    def __post_init__(self) -> None:
        for name, value in (
            ("answer", self.answer),
            ("reasoning", self.reasoning),
            ("eos", self.eos),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} weight must be finite and positive")


@dataclass(frozen=True)
class WeightedExample:
    """One fully rendered SFT-v2 example and its token-level supervision."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    loss_weights: list[float]
    prompt_length: int
    prompt_text: str
    target_text: str
    segment_token_counts: dict[str, int]
    segment_weighted_contribution: dict[str, float]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.labels),
            len(self.loss_weights),
        }
        if len(lengths) != 1:
            raise ValueError("weighted example arrays must have identical lengths")
        if self.prompt_length <= 0 or self.prompt_length >= len(self.input_ids):
            raise ValueError("prompt and completion must both contain tokens")
        if any(weight != 0.0 for weight in self.loss_weights[: self.prompt_length]):
            raise ValueError("prompt tokens must have zero loss weight")
        if any(label != -100 for label in self.labels[: self.prompt_length]):
            raise ValueError("prompt labels must use ignore_index")


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        values = tokenizer.encode(text, add_special_tokens=False)
    elif callable(tokenizer):
        values = tokenizer(text)
    else:
        raise TypeError("tokenizer must expose encode or be callable")
    return [int(value) for value in values]


def _render_prompt(tokenizer: Any, user_content: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    raise TypeError("SFT-v2 tokenizer must expose apply_chat_template")


def _answer_index(row: Mapping[str, Any], option_count: int) -> int:
    value = row.get("answer_idx")
    if isinstance(value, str):
        folded = value.strip().upper()
        if len(folded) == 1 and "A" <= folded <= "Z":
            index = ord(folded) - ord("A")
        else:
            index = int(folded)
    else:
        index = int(value)
    if not 0 <= index < option_count:
        raise ValueError("CMB answer index is outside the legal candidates")
    return index


def render_sft_v2_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    weights: SupervisionWeights,
    max_seq_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> WeightedExample | None:
    """Render one answer-first row or return ``None`` when it is over length."""

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    forbidden_text = " ".join(
        str(row.get(field, "")) for field in ("question", "answer", "reasoning")
    ).casefold()
    if "<think>" in forbidden_text or "</think>" in forbidden_text:
        raise ValueError("SFT-v2 refuses literal thinking tags")

    kind = str(row.get("sft_v2_kind") or "medical_o1_answer_first")
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")

    segments: list[tuple[str, str, float]]
    if kind == "medical_o1_answer_first":
        answer = str(row.get("answer") or "").strip()
        reasoning = str(row.get("reasoning") or "").strip()
        if not answer or not reasoning:
            raise ValueError("Medical-O1 SFT-v2 requires Response and Complex_CoT")
        user_content = question
        segments = [
            ("answer", f"答案：\n{answer}\n\n", weights.answer),
            ("reasoning", f"分析：\n{reasoning}", weights.reasoning),
        ]
    elif kind == "cmb_mcq_bridge":
        options = [str(value).strip() for value in row.get("options") or []]
        if not 2 <= len(options) <= 5 or any(not value for value in options):
            raise ValueError("CMB bridge requires two to five ordered options")
        answer_index = _answer_index(row, len(options))
        letter = chr(ord("A") + answer_index)
        user_content = format_mcq_question(question, options)
        segments = [
            ("answer", f"答案：{letter}. {options[answer_index]}", weights.answer),
        ]
    else:
        raise ValueError(f"unsupported SFT-v2 kind: {kind}")

    prompt_text = _render_prompt(tokenizer, user_content, system_prompt)
    prompt_ids = _encode(tokenizer, prompt_text)
    if not prompt_ids:
        raise ValueError("prompt tokenized to zero tokens")

    segment_ids: list[tuple[str, list[int], float]] = []
    for name, text, weight in segments:
        token_ids = _encode(tokenizer, text)
        if not token_ids:
            raise ValueError(f"{name} segment tokenized to zero tokens")
        segment_ids.append((name, token_ids, weight))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer eos_token_id is required")
    segment_ids.append(("eos", [int(eos_token_id)], weights.eos))

    input_ids = list(prompt_ids)
    labels = [-100] * len(prompt_ids)
    loss_weights = [0.0] * len(prompt_ids)
    segment_counts = {"answer": 0, "reasoning": 0, "eos": 0}
    contributions = {"answer": 0.0, "reasoning": 0.0, "eos": 0.0}
    for name, token_ids, weight in segment_ids:
        input_ids.extend(token_ids)
        labels.extend(token_ids)
        loss_weights.extend([float(weight)] * len(token_ids))
        segment_counts[name] += len(token_ids)
        contributions[name] += len(token_ids) * float(weight)

    if len(input_ids) > max_seq_length:
        return None
    target_text = "".join(text for _, text, _ in segments)
    return WeightedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
        prompt_length=len(prompt_ids),
        prompt_text=prompt_text,
        target_text=target_text,
        segment_token_counts=segment_counts,
        segment_weighted_contribution=contributions,
    )


def weighted_causal_lm_components(
    logits: Any, labels: Any, loss_weights: Any
) -> tuple[Any, Any, dict[str, float | int]]:
    """Compute a local weighted numerator/denominator on one model replica."""

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits[B,T,V], labels[B,T], weights[B,T]")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels and weights must share batch/sequence dimensions")
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].float().contiguous()
    valid = shift_labels.ne(-100)
    if torch.any(shift_weights[~valid] != 0):
        raise ValueError("ignored labels must have zero loss weight")
    effective_weights = shift_weights * valid.float()
    denominator = effective_weights.sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise ValueError("weighted loss has no supervised tokens")
    per_token = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    numerator = (per_token * effective_weights).sum()
    loss = numerator / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("weighted causal LM loss is non-finite")
    return numerator, denominator, {
        "weighted_token_denominator": float(denominator.detach().cpu()),
        "supervised_tokens": int(valid.sum().detach().cpu()),
    }


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> tuple[Any, dict[str, float | int]]:
    """Compute shifted FP32 CE normalized by the frozen token weights."""

    numerator, denominator, stats = weighted_causal_lm_components(
        logits, labels, loss_weights
    )
    return numerator / denominator, stats


def memory_efficient_weighted_causal_lm_components(
    logits: Any,
    labels: Any,
    loss_weights: Any,
    *,
    chunk_tokens: int = 64,
) -> tuple[Any, Any]:
    """FP32 weighted CE with chunk-recomputed backward state.

    The reference implementation retains an FP32 ``[B,T,V]`` tensor until
    backward. Qwen3's vocabulary makes that roughly 0.8 GiB per replica for the
    longest formal sample. This autograd boundary retains only the original
    BF16 logits and recomputes one small FP32 token chunk during backward. The
    objective, causal shift, weights, and global normalization are unchanged.
    """

    import torch

    if logits.ndim != 3 or labels.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits[B,T,V], labels[B,T], weights[B,T]")
    if logits.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError("logits, labels and weights must share batch/sequence dimensions")
    if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    shifted_labels = labels[:, 1:]
    shifted_weights = loss_weights[:, 1:].float()
    valid = shifted_labels.ne(-100)
    if torch.any(shifted_weights[~valid] != 0):
        raise ValueError("ignored labels must have zero loss weight")
    if torch.any(shifted_labels[valid] < 0) or torch.any(shifted_labels[valid] >= logits.shape[-1]):
        raise ValueError("supervised label is outside the vocabulary")
    denominator = (shifted_weights * valid.float()).sum()
    if not torch.isfinite(denominator) or denominator.item() <= 0:
        raise ValueError("weighted loss has no supervised tokens")

    class ChunkedWeightedCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_logits, input_labels, input_weights):
            ctx.save_for_backward(input_logits, input_labels, input_weights)
            ctx.chunk_tokens = chunk_tokens
            numerator = torch.zeros((), device=input_logits.device, dtype=torch.float32)
            batch_size, sequence_length, _ = input_logits.shape
            for batch_index in range(batch_size):
                for start in range(0, sequence_length - 1, chunk_tokens):
                    stop = min(start + chunk_tokens, sequence_length - 1)
                    chunk = input_logits[batch_index, start:stop, :].float()
                    targets = input_labels[batch_index, start + 1 : stop + 1]
                    weights = input_weights[batch_index, start + 1 : stop + 1].float()
                    chunk_valid = targets.ne(-100)
                    safe_targets = targets.masked_fill(~chunk_valid, 0)
                    negative_log_likelihood = (
                        torch.logsumexp(chunk, dim=-1)
                        - chunk.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
                    )
                    numerator.add_((negative_log_likelihood * weights * chunk_valid.float()).sum())
            if not torch.isfinite(numerator):
                raise FloatingPointError("weighted causal LM numerator is non-finite")
            return numerator

        @staticmethod
        def backward(ctx, grad_numerator):
            input_logits, input_labels, input_weights = ctx.saved_tensors
            gradient = torch.zeros_like(input_logits)
            batch_size, sequence_length, _ = input_logits.shape
            scale = grad_numerator.float()
            for batch_index in range(batch_size):
                for start in range(0, sequence_length - 1, ctx.chunk_tokens):
                    stop = min(start + ctx.chunk_tokens, sequence_length - 1)
                    chunk = input_logits[batch_index, start:stop, :].float()
                    targets = input_labels[batch_index, start + 1 : stop + 1]
                    weights = input_weights[batch_index, start + 1 : stop + 1].float()
                    chunk_valid = targets.ne(-100)
                    safe_targets = targets.masked_fill(~chunk_valid, 0)
                    chunk_gradient = torch.softmax(chunk, dim=-1)
                    rows = torch.arange(stop - start, device=input_logits.device)
                    chunk_gradient[rows, safe_targets] -= chunk_valid.float()
                    chunk_gradient.mul_(weights.unsqueeze(1) * chunk_valid.unsqueeze(1) * scale)
                    gradient[batch_index, start:stop, :] = chunk_gradient.to(input_logits.dtype)
            return gradient, None, None

    numerator = ChunkedWeightedCrossEntropy.apply(logits, labels, loss_weights)
    return numerator, denominator


def attach_weighted_loss_forward(model: Any, *, chunk_tokens: int = 64) -> None:
    """Keep full-vocabulary logits and weighted CE inside each GPU replica.

    PyTorch DataParallel gathers a model's returned tensors onto GPU0. Returning
    Qwen's full ``[batch, sequence, vocabulary]`` logits before computing our
    custom loss exhausts GPU0 even though each replica fits. This wrapper removes
    labels/weights before the PEFT forward, computes the exact weighted loss on
    the replica, and returns only two scalar tensors for the cross-device gather.
    """

    if getattr(model, "_ca_opd_weighted_forward_attached", False):
        raise ValueError("weighted loss forward is already attached")
    if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    base_class = model.__class__

    class ReplicaLocalWeightedForward(base_class):
        """Ephemeral PEFT subclass whose method rebinds on every replica."""

        def forward(self, *args, labels=None, loss_weights=None, **kwargs):
            if labels is None or loss_weights is None:
                raise ValueError("weighted SFT forward requires labels and loss_weights")
            # A class-level override is essential: DataParallel shallow-copies
            # instance attributes, so an instance-bound replacement would keep
            # GPU1's method attached to GPU0. ``super`` resolves against the
            # replica-local ``self`` and therefore its replica-local parameters.
            outputs = super().forward(*args, **kwargs)
            numerator, denominator = memory_efficient_weighted_causal_lm_components(
                outputs.logits, labels, loss_weights, chunk_tokens=chunk_tokens
            )
            return {
                "weighted_loss_numerator": numerator.reshape(1),
                "weighted_loss_denominator": denominator.reshape(1),
            }

    ReplicaLocalWeightedForward.__name__ = f"Weighted{base_class.__name__}"
    ReplicaLocalWeightedForward.__qualname__ = ReplicaLocalWeightedForward.__name__
    model.__class__ = ReplicaLocalWeightedForward
    model._ca_opd_weighted_forward_attached = True


class WeightedDataCollator:
    """Right-pad pretokenized weighted examples without altering supervision."""

    def __init__(self, tokenizer: Any) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer pad_token_id is required")
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError("SFT-v2 requires right padding")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            batch["input_ids"].append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            batch["attention_mask"].append(list(feature["attention_mask"]) + [0] * padding)
            batch["labels"].append(list(feature["labels"]) + [-100] * padding)
            batch["loss_weights"].append(list(feature["loss_weights"]) + [0.0] * padding)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }


def make_weighted_trainer_class(base_trainer: type) -> type:
    """Create the TRL subclass lazily so CPU imports never require TRL."""

    class WeightedSFTTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ARG002
            outputs = model(**inputs)
            numerator = outputs["weighted_loss_numerator"].sum()
            denominator = outputs["weighted_loss_denominator"].sum()
            if denominator.item() <= 0:
                raise ValueError("weighted trainer received an empty denominator")
            loss = numerator / denominator
            return (loss, outputs) if return_outputs else loss

    WeightedSFTTrainer.__name__ = "WeightedSFTTrainer"
    return WeightedSFTTrainer
