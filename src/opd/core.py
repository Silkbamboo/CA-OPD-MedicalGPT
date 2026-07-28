"""On-Policy Distillation core mathematics (PROJECT_PLAN.md §9, §11.4).

Everything here is framework-agnostic pure ``torch``: it consumes *logits* (or
pre-computed logprobs) and returns losses/statistics. veRL/vLLM plug in later by
supplying those tensors; Phase 0 correctness is therefore testable on CPU with a
toy model and hand-built tensors.

Tensor shape conventions used throughout
----------------------------------------
``B`` batch size, ``T`` padded sequence length, ``V`` vocab size.

* ``input_ids``        ``[B, T]``  ``prompt || completion || pad``  (right padding)
* ``attention_mask``   ``[B, T]``  1 for prompt+completion tokens, 0 for pad
* ``completion_mask``  ``[B, T]``  1 only for student-generated tokens
* ``logits``           ``[B, T, V]``
* per-token logprobs   ``[B, T-1]`` aligned to *targets* ``input_ids[:, 1:]``

The single most important alignment rule (autoregressive right shift):

    logits[:, i, :]  predicts  input_ids[:, i + 1]

so a per-token array ``lp[:, i]`` is the logprob of ``input_ids[:, i + 1]``, and
the mask that selects completion targets is ``completion_mask[:, 1:]``.

Sign conventions (PROJECT_PLAN.md §9)
-------------------------------------
* reverse-KL per-token reward  ``r_t = log pi_S(y_t) - log pi_T(y_t)``
* advantage                    ``A_t = beta * (log pi_T(y_t) - log pi_S(y_t)) = -beta * r_t``

So ``A_t > 0`` exactly when the teacher likes the realised token more than the
student does, and a policy-gradient step then *increases* that token's
probability. This is asserted by tests, not just documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

Reduction = Literal["token_mean", "seq_mean_token_mean"]

__all__ = [
    "OPDBatch",
    "build_opd_batch",
    "token_logprobs",
    "selected_token_logprobs",
    "masked_mean",
    "reverse_kl_per_token",
    "teacher_student_advantage",
    "DomainKLController",
    "scale_and_clip_advantage",
    "ppo_policy_loss",
    "policy_entropy",
    "assert_same_targets",
    "PPOStats",
]


# ---------------------------------------------------------------------------
# batch construction
# ---------------------------------------------------------------------------


@dataclass
class OPDBatch:
    """A batch of ``prompt + student completion`` sequences.

    The batch is the *contract* between rollout and scoring: student and teacher
    must both be forwarded on exactly this ``input_ids`` tensor, which is how the
    project guarantees "Teacher 不重新生成答案" (PROJECT_PLAN.md §9).
    """

    input_ids: Tensor  # [B, T] int64
    attention_mask: Tensor  # [B, T] int64/bool
    completion_mask: Tensor  # [B, T] int64/bool
    prompt_lengths: Tensor  # [B] int64
    completion_lengths: Tensor  # [B] int64
    domains: Tuple[str, ...] = ()
    pad_token_id: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T], got {tuple(self.input_ids.shape)}")
        b, t = self.input_ids.shape
        if t < 2:
            raise ValueError("sequences must have length >= 2 to form an autoregressive target")
        for name in ("attention_mask", "completion_mask"):
            m = getattr(self, name)
            if tuple(m.shape) != (b, t):
                raise ValueError(f"{name} must be {(b, t)}, got {tuple(m.shape)}")
        if self.domains and len(self.domains) != b:
            raise ValueError(f"domains must have one entry per sequence ({b}), got {len(self.domains)}")
        # completion tokens must be a subset of attended (non-pad) tokens
        leaked = (self.completion_mask.long() * (1 - self.attention_mask.long())).sum().item()
        if leaked:
            raise ValueError(f"completion_mask marks {leaked} padding position(s) as trainable")
        if (self.completion_mask.long().sum(dim=1) == 0).any():
            raise ValueError("every sequence must contain at least one completion token")

    # -- derived views ----------------------------------------------------
    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def target_ids(self) -> Tensor:
        """``[B, T-1]`` the tokens being predicted."""
        return self.input_ids[:, 1:]

    def target_mask(self) -> Tensor:
        """``[B, T-1]`` 1 where the predicted token is a student completion token.

        Prompt positions, padding and (optionally) EOS are excluded here, which
        is what makes the loss "assistant-only".
        """
        return self.completion_mask[:, 1:].to(dtype=torch.float32)

    def num_completion_tokens(self) -> int:
        return int(self.target_mask().sum().item())

    def domain_mask(self, domain: str) -> Tensor:
        """``[B, T-1]`` target mask restricted to one domain."""
        if not self.domains:
            raise ValueError("batch has no domain labels")
        rows = torch.tensor(
            [1.0 if d == domain else 0.0 for d in self.domains],
            dtype=torch.float32,
            device=self.input_ids.device,
        ).unsqueeze(1)
        return self.target_mask() * rows

    def fingerprint(self) -> str:
        """Stable hash of ``input_ids`` used to prove teacher/student alignment."""
        import hashlib

        arr = self.input_ids.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()
        return hashlib.sha256(arr).hexdigest()[:16]


def build_opd_batch(
    prompt_ids: Sequence[Sequence[int]],
    completion_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
    domains: Optional[Sequence[str]] = None,
    include_eos_in_loss: bool = True,
    max_length: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> OPDBatch:
    """Right-pad ``prompt || completion`` pairs into an :class:`OPDBatch`.

    Parameters
    ----------
    include_eos_in_loss
        If ``True`` a trailing ``eos_token_id`` inside the completion stays
        trainable (the student must learn *when to stop*). If ``False`` the
        final EOS position is masked out of the loss. Any padding after EOS is
        always masked.
    max_length
        Hard cap; exceeding it raises instead of silently truncating, because a
        silent truncation would change the training signal (agent.md §5:
        "失败要显式报错").
    """
    if len(prompt_ids) != len(completion_ids):
        raise ValueError(f"prompt/completion count mismatch: {len(prompt_ids)} vs {len(completion_ids)}")
    if len(prompt_ids) == 0:
        raise ValueError("cannot build an empty OPD batch")
    if domains is not None and len(domains) != len(prompt_ids):
        raise ValueError("domains must align with prompts")

    lengths = []
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        if len(p) == 0:
            raise ValueError(f"sequence {i}: empty prompt")
        if len(c) == 0:
            raise ValueError(f"sequence {i}: empty completion (nothing to distil)")
        total = len(p) + len(c)
        if max_length is not None and total > max_length:
            raise ValueError(
                f"sequence {i}: prompt+completion = {total} exceeds max_length={max_length}; "
                "shorten the rollout budget instead of truncating silently"
            )
        lengths.append(total)

    t_max = max(lengths)
    b = len(prompt_ids)
    input_ids = torch.full((b, t_max), pad_token_id, dtype=torch.int64)
    attention_mask = torch.zeros((b, t_max), dtype=torch.int64)
    completion_mask = torch.zeros((b, t_max), dtype=torch.int64)
    prompt_lengths = torch.zeros(b, dtype=torch.int64)
    completion_lengths = torch.zeros(b, dtype=torch.int64)

    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        seq = list(p) + list(c)
        n_p, n_c = len(p), len(c)
        input_ids[i, : n_p + n_c] = torch.tensor(seq, dtype=torch.int64)
        attention_mask[i, : n_p + n_c] = 1
        completion_mask[i, n_p : n_p + n_c] = 1
        if not include_eos_in_loss and eos_token_id is not None and c[-1] == eos_token_id:
            completion_mask[i, n_p + n_c - 1] = 0
        prompt_lengths[i] = n_p
        completion_lengths[i] = int(completion_mask[i].sum().item())

    return OPDBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        completion_mask=completion_mask.to(device),
        prompt_lengths=prompt_lengths.to(device),
        completion_lengths=completion_lengths.to(device),
        domains=tuple(domains) if domains is not None else (),
        pad_token_id=int(pad_token_id),
    )


# ---------------------------------------------------------------------------
# logprobs
# ---------------------------------------------------------------------------


def token_logprobs(logits: Tensor, input_ids: Tensor, temperature: float = 1.0) -> Tensor:
    """Per-target logprobs ``[B, T-1]`` with the autoregressive right shift.

    ``logits[:, i]`` scores the distribution over ``input_ids[:, i + 1]``, so we
    drop the last logit column and the first token column.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [B, T], got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(f"logits {tuple(logits.shape)[:2]} and input_ids {tuple(input_ids.shape)} disagree")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    if temperature != 1.0:
        shifted_logits = shifted_logits / temperature
    log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def selected_token_logprobs(logits: Tensor, batch: OPDBatch, temperature: float = 1.0) -> Tensor:
    """``token_logprobs`` for an :class:`OPDBatch` (targets come from the batch)."""
    return token_logprobs(logits, batch.input_ids, temperature=temperature)


def policy_entropy(logits: Tensor, target_mask: Tensor, temperature: float = 1.0) -> Tensor:
    """Mean token-level entropy over masked target positions (scalar).

    Entropy is computed on the *predictive* distribution at each completion
    position, i.e. the same positions that receive gradient.
    """
    shifted = logits[:, :-1, :].float()
    if temperature != 1.0:
        shifted = shifted / temperature
    log_probs = torch.log_softmax(shifted, dim=-1)
    ent = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B, T-1]
    return masked_mean(ent, target_mask)


def masked_mean(values: Tensor, mask: Tensor, dim: Optional[int] = None, eps: float = 1e-8) -> Tensor:
    """Mean of ``values`` over positions where ``mask`` is 1.

    Batch-level ``token_mean``: every token contributes equally, so long and
    short sequences are weighted by their token count (no length bias from an
    accidental per-sequence average).
    """
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} and mask {tuple(mask.shape)} must match")
    mask = mask.to(values.dtype)
    if dim is None:
        denom = mask.sum()
        if denom.item() == 0:
            raise ValueError("masked_mean over an all-zero mask")
        return (values * mask).sum() / denom.clamp_min(eps)
    denom = mask.sum(dim=dim)
    return (values * mask).sum(dim=dim) / denom.clamp_min(eps)


def assert_same_targets(batch: OPDBatch, *logprob_tensors: Tensor, names: Optional[Sequence[str]] = None) -> None:
    """Guard that all logprob tensors were produced on the same target grid."""
    expected = (batch.batch_size, batch.seq_len - 1)
    labels = list(names) if names else [f"tensor[{i}]" for i in range(len(logprob_tensors))]
    for label, lp in zip(labels, logprob_tensors):
        if tuple(lp.shape) != expected:
            raise ValueError(f"{label} has shape {tuple(lp.shape)}, expected {expected} for this batch")


# ---------------------------------------------------------------------------
# reverse KL / advantage
# ---------------------------------------------------------------------------


def reverse_kl_per_token(student_logprobs: Tensor, teacher_logprobs: Tensor) -> Tensor:
    """``r_t = log pi_S(y_t) - log pi_T(y_t)`` (PROJECT_PLAN.md §9).

    Positive where the student is over-confident relative to the teacher. Its
    masked mean is the single-sample estimate of the reverse KL on the student's
    own trajectory distribution.
    """
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            f"student {tuple(student_logprobs.shape)} / teacher {tuple(teacher_logprobs.shape)} shape mismatch"
        )
    return student_logprobs - teacher_logprobs


def teacher_student_advantage(
    student_logprobs: Tensor,
    teacher_logprobs: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """``A_t = beta * (log pi_T - log pi_S)`` = ``-beta * r_t``.

    The teacher tensor must be detached; passing a grad-carrying teacher tensor
    is a hard error because the teacher must never be back-propagated through
    (PROJECT_PLAN.md §9: "不对 Teacher 反向传播").
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if teacher_logprobs.requires_grad:
        raise ValueError("teacher logprobs carry grad; score the teacher under torch.no_grad()")
    r = reverse_kl_per_token(student_logprobs.detach(), teacher_logprobs)
    return (-beta * r).detach()


class DomainKLController:
    """Per-domain reverse-KL EMA and safety scale (PROJECT_PLAN.md §11.4).

    ``s_d = min(1, kappa_d / (EMA(D_KL,d) + eps))`` — when a domain's KL blows
    up, its update magnitude is throttled instead of producing extreme punitive
    gradients. ``s_d`` is never > 1: the mechanism can only damp, never amplify.
    """

    def __init__(
        self,
        kappa: Mapping[str, float] | float,
        rho: float = 0.9,
        eps: float = 1e-6,
        domains: Optional[Iterable[str]] = None,
    ):
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        self.rho = float(rho)
        self.eps = float(eps)
        if isinstance(kappa, Mapping):
            if any(v <= 0 for v in kappa.values()):
                raise ValueError("all kappa values must be > 0")
            self._kappa: Dict[str, float] = {k: float(v) for k, v in kappa.items()}
            self._default_kappa: Optional[float] = None
        else:
            if kappa <= 0:
                raise ValueError("kappa must be > 0")
            self._kappa = {}
            self._default_kappa = float(kappa)
        self.ema: Dict[str, float] = {d: 0.0 for d in (domains or ())}
        self._seen: Dict[str, bool] = {d: False for d in (domains or ())}

    def kappa(self, domain: str) -> float:
        if domain in self._kappa:
            return self._kappa[domain]
        if self._default_kappa is not None:
            return self._default_kappa
        raise KeyError(f"no kappa configured for domain {domain!r}")

    def update(self, domain: str, kl_value: float) -> float:
        """Fold one observation into the domain EMA and return the new EMA.

        The first observation initialises the EMA directly (no warm-up bias
        toward the 0.0 placeholder).
        """
        value = float(kl_value)
        if not self._seen.get(domain, False):
            self.ema[domain] = value
            self._seen[domain] = True
        else:
            self.ema[domain] = self.rho * self.ema[domain] + (1.0 - self.rho) * value
        return self.ema[domain]

    def scale(self, domain: str) -> float:
        ema = self.ema.get(domain)
        if ema is None or not self._seen.get(domain, False):
            return 1.0  # no evidence yet -> do not throttle
        return min(1.0, self.kappa(domain) / (abs(ema) + self.eps))

    def state_dict(self) -> Dict[str, object]:
        return {
            "rho": self.rho,
            "eps": self.eps,
            "kappa": dict(self._kappa),
            "default_kappa": self._default_kappa,
            "ema": dict(self.ema),
            "seen": dict(self._seen),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.rho = float(state["rho"])  # type: ignore[arg-type]
        self.eps = float(state["eps"])  # type: ignore[arg-type]
        self._kappa = dict(state.get("kappa") or {})  # type: ignore[arg-type]
        dk = state.get("default_kappa")
        self._default_kappa = None if dk is None else float(dk)  # type: ignore[arg-type]
        self.ema = dict(state.get("ema") or {})  # type: ignore[arg-type]
        self._seen = dict(state.get("seen") or {})  # type: ignore[arg-type]


def scale_and_clip_advantage(
    advantages: Tensor,
    scales: Tensor | float,
    a_max: float,
) -> Tuple[Tensor, Tensor]:
    """``A_t <- clip(s_d * A_t, -a_max, a_max)``.

    Returns ``(clipped, clipped_flag)`` where ``clipped_flag`` is 1.0 at
    positions whose magnitude was actually cut, so the caller can log
    ``opd/advantage_clip_fraction`` honestly.
    """
    if a_max <= 0:
        raise ValueError(f"a_max must be > 0, got {a_max}")
    if isinstance(scales, Tensor):
        if scales.shape != advantages.shape and scales.dim() != 0:
            if scales.dim() == 1 and scales.shape[0] == advantages.shape[0]:
                scales = scales.unsqueeze(1)
            else:
                raise ValueError(
                    f"scales {tuple(scales.shape)} not broadcastable to advantages {tuple(advantages.shape)}"
                )
        if (scales > 1.0 + 1e-6).any():
            raise ValueError("KL safety scale must be <= 1 (it may only damp updates)")
    else:
        if scales > 1.0 + 1e-6:
            raise ValueError("KL safety scale must be <= 1 (it may only damp updates)")
    scaled = advantages * scales
    clipped = scaled.clamp(min=-a_max, max=a_max)
    flag = (scaled.abs() > a_max).to(dtype=advantages.dtype)
    return clipped, flag


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


@dataclass
class PPOStats:
    loss: float
    ratio_mean: float
    clip_fraction: float
    advantage_mean: float
    advantage_std: float
    advantage_clip_fraction: float
    num_tokens: int
    approx_kl: float

    def as_metrics(self) -> Dict[str, float]:
        """Map to the frozen metric vocabulary (agent.md §8)."""
        return {
            "train/loss": self.loss,
            "ppo/ratio_mean": self.ratio_mean,
            "ppo/clip_fraction": self.clip_fraction,
            "opd/advantage_mean": self.advantage_mean,
            "opd/advantage_std": self.advantage_std,
            "opd/advantage_clip_fraction": self.advantage_clip_fraction,
        }


def ppo_policy_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    target_mask: Tensor,
    clip_range: float = 0.2,
    reduction: Reduction = "token_mean",
    advantage_clip_flags: Optional[Tensor] = None,
) -> Tuple[Tensor, PPOStats]:
    """Clipped importance-ratio policy-gradient loss on completion tokens only.

    ``old_logprobs`` are the rollout-time logprobs and **must be frozen** for the
    whole update: they are required to be detached, which structurally prevents
    the classic bug of re-using the freshly updated policy's logprobs
    (PROJECT_PLAN.md §9: "rollout old logprob 不被误替换").

    ``reduction``
        ``token_mean`` (default) divides by the total number of completion
        tokens in the batch: no length bias.
        ``seq_mean_token_mean`` averages per sequence first, which up-weights
        short sequences; provided for ablation and covered by a test that
        demonstrates the difference.
    """
    if clip_range <= 0:
        raise ValueError(f"clip_range must be > 0, got {clip_range}")
    if old_logprobs.requires_grad:
        raise ValueError(
            "old_logprobs carry grad: they must be the frozen rollout logprobs "
            "(call .detach() at rollout time, not here)"
        )
    if advantages.requires_grad:
        raise ValueError("advantages must be detached before the PPO update")
    for name, tensor in (("old_logprobs", old_logprobs), ("advantages", advantages), ("target_mask", target_mask)):
        if tensor.shape != new_logprobs.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != new_logprobs {tuple(new_logprobs.shape)}"
            )

    mask = target_mask.to(new_logprobs.dtype)
    log_ratio = (new_logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    # maximise min(...) == minimise -min(...)
    per_token_loss = -torch.min(unclipped, clipped)

    if reduction == "token_mean":
        loss = masked_mean(per_token_loss, mask)
    elif reduction == "seq_mean_token_mean":
        per_seq = masked_mean(per_token_loss, mask, dim=1)
        loss = per_seq.mean()
    else:  # pragma: no cover - guarded by Literal + explicit raise
        raise ValueError(f"unknown reduction {reduction!r}")

    with torch.no_grad():
        was_clipped = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).to(mask.dtype)
        adv_flat = advantages[mask > 0]
        stats = PPOStats(
            loss=float(loss.detach()),
            ratio_mean=float(masked_mean(ratio, mask)),
            clip_fraction=float(masked_mean(was_clipped, mask)),
            advantage_mean=float(adv_flat.mean()) if adv_flat.numel() else 0.0,
            advantage_std=float(adv_flat.std(unbiased=False)) if adv_flat.numel() > 1 else 0.0,
            advantage_clip_fraction=(
                float(masked_mean(advantage_clip_flags.to(mask.dtype), mask))
                if advantage_clip_flags is not None
                else 0.0
            ),
            num_tokens=int(mask.sum().item()),
            approx_kl=float(masked_mean(-log_ratio, mask)),
        )
    return loss, stats
