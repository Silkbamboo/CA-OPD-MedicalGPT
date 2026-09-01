"""A deliberately tiny causal LM for CPU correctness work.

Why this exists: docs/METHOD.md §14 (Phase 0) requires "CPU/小模型跑通一个
optimizer step" *before* any paid GPU time. Downloading Qwen3 is neither
necessary nor sufficient for verifying the OPD math, and the current dev box has
no GPU. A ~10k-parameter model with real causal attention lets every alignment,
mask and gradient-direction test run in milliseconds.

It is **not** a model architecture contribution and is never used for results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ToyLMConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    num_heads: int = 2
    max_position: int = 64
    tie_embeddings: bool = True


class ToyCausalLM(nn.Module):
    """Single-layer decoder-only LM with explicit causal masking.

    ``forward`` returns logits ``[B, T, V]`` where ``logits[:, i]`` may only
    depend on ``input_ids[:, :i + 1]``. ``tests/test_opd_math.py`` asserts that
    causality property directly, so a broken mask cannot hide behind a passing
    loss curve.

    ``logit_bias`` (``[V]``, non-trainable) is added to every position's logits.
    It exists so a test can make a teacher *provably* prefer a specific token
    independently of the hidden state - see :func:`make_toy_pair`.
    """

    def __init__(self, config: ToyLMConfig | None = None):
        super().__init__()
        self.config = config or ToyLMConfig()
        c = self.config
        if c.hidden_size % c.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.pos = nn.Embedding(c.max_position, c.hidden_size)
        self.q = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.k = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.v = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.o = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.norm = nn.LayerNorm(c.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(c.hidden_size, 2 * c.hidden_size),
            nn.GELU(),
            nn.Linear(2 * c.hidden_size, c.hidden_size),
        )
        self.norm2 = nn.LayerNorm(c.hidden_size)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.register_buffer("logit_bias", torch.zeros(c.vocab_size), persistent=True)

    # -- helpers ----------------------------------------------------------
    def _attn(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        b, t, h = x.shape
        heads = self.config.num_heads
        dh = h // heads
        q = self.q(x).view(b, t, heads, dh).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k(x).view(b, t, heads, dh).transpose(1, 2)
        v = self.v(x).view(b, t, heads, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)  # [B, H, T, T]

        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            keep = attention_mask.to(torch.bool)[:, None, None, :]  # [B,1,1,T]
            scores = scores.masked_fill(~keep, float("-inf"))
            # a fully-masked row (padding query) would produce NaN; neutralise it
            fully_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
            scores = scores.masked_fill(fully_masked, 0.0)
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, t, h)
        return self.o(out)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T], got {tuple(input_ids.shape)}")
        b, t = input_ids.shape
        if t > self.config.max_position:
            raise ValueError(f"sequence length {t} exceeds max_position {self.config.max_position}")
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.embed(input_ids) + self.pos(positions)
        x = x + self._attn(self.norm(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        logits = x @ self.embed.weight.T if self.lm_head is None else self.lm_head(x)
        return logits + self.logit_bias


def make_toy_pair(
    seed: int = 0,
    config: ToyLMConfig | None = None,
    teacher_bias_token: Optional[int] = None,
    teacher_bias_strength: float = 4.0,
) -> tuple[ToyCausalLM, ToyCausalLM]:
    """Build a (student, teacher) pair with independent random weights.

    ``teacher_bias_token`` makes the teacher provably prefer one token by adding
    ``teacher_bias_strength`` to that token's logit at every position. Earlier
    attempts biased the (tied) embedding row instead; that also changed the
    teacher's hidden state and the resulting preference depended on the sign of
    the hidden activations, i.e. it was not a preference at all. Biasing the
    logits directly is unambiguous.
    """
    cfg = config or ToyLMConfig()
    torch.manual_seed(seed)
    student = ToyCausalLM(cfg)
    torch.manual_seed(seed + 1)
    teacher = ToyCausalLM(cfg)
    if teacher_bias_token is not None:
        if not 0 <= teacher_bias_token < cfg.vocab_size:
            raise ValueError(f"teacher_bias_token out of range: {teacher_bias_token}")
        with torch.no_grad():
            teacher.logit_bias[teacher_bias_token] += float(teacher_bias_strength)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return student, teacher
