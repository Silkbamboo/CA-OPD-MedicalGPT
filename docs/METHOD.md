# Method

## Research question

The project asks whether on-policy distillation can transfer medical capability from a
specialized teacher while preserving the general capability of the original model, and
whether a constraint-aware teacher router performs better than a fixed router.

The optimization target is treated as a constrained objective:

\[
\max_\theta M_{\mathrm{medical}}(\theta)
\quad\text{subject to}\quad
M_{\mathrm{general}}(\theta) \ge M_{\mathrm{general}}(\theta_0)-\delta.
\]

The final evidence did not support the OPD or CA superiority hypotheses under the frozen
setting. The implementation and negative result are both part of the research artifact.

## Experimental routes

| Route | Initialization | Teacher policy | Purpose |
|---|---|---|---|
| B0 | Qwen3-4B | none | Base reference |
| B1 | B0 + SFT LoRA | none | Frozen medical teacher |
| B2 | fresh B0 + zero-effect LoRA | B1 only | Medical knowledge transfer |
| IDT | fresh B0 + zero-effect LoRA | fixed Medical/Base schedule | Fixed multi-teacher baseline |
| CA-OPD | fresh B0 + zero-effect LoRA | controller-driven Medical/Base routing | Proposed constrained policy |

B2 is deliberately initialized from the Base model, not from the SFT checkpoint. B1 is a
frozen teacher. This separation tests transfer rather than continued supervised training.

## Same-trajectory distillation

For prompt \(x\), the current Student samples a completion:

\[
y \sim \pi_\theta(\cdot\mid x).
\]

The Student and selected Teacher score the exact same token sequence. The Teacher does not
generate a replacement answer:

\[
A_t = \beta\left[\log\pi_T(y_t\mid x,y_{<t})-
\log\pi_\theta(y_t\mid x,y_{<t})\right].
\]

The policy update uses a frozen old-policy log-probability and a clipped importance ratio.
Prompt and padding tokens are masked. The production implementation is in
[`src/opd`](../src/opd); the compact mathematical primitives are in
[`src/opd/core.py`](../src/opd/core.py).

## Prompt-equal updates and trust budgets

Each optimizer step contains four prompt trajectories. Every prompt contributes the same
nominal loss weight, independent of response length. An observed short CMB completion once
produced an extreme gradient. Dropping short CMB responses or imposing a shared minimum
length would have changed the source mixture, so the final protocol instead applies the same
per-prompt gradient trust budget before summing the four contributions, followed by the
existing global gradient clip.

This mechanism is shared by B2, IDT, and CA-OPD. It is an update-stability rule, not evidence
that any route improves capability.

## Fixed and constraint-aware routing

IDT uses a frozen alternation schedule. CA-OPD estimates medical and general capability gaps
on label-isolated controller data, applies an exponential moving average and hysteresis, and
selects the Teacher for the next window subject to probability bounds. Final-test records
cannot be imported by the router or trainer.

The router implementation is in [`src/opd/router.py`](../src/opd/router.py). The experiment
found no advantage over IDT at the shared boundary of 120 accepted steps and four prompts per
step. This controls update count and prompt count, not generated-token count or wall-clock time.

## Scoring and evaluation

Multiple-choice scoring uses the next-token logits for the available labels (A-D or A-E):

\[
\log\operatorname{softmax}(\operatorname{float32}(z_{-1}))[i].
\]

The production scorer uses Transformers direct logits with BF16 model weights and FP32
`log_softmax`. A vLLM LoRA `prompt_logprobs` path showed run-to-run score drift beyond the
pre-registered tolerance and was retained for diagnostics only; the project does not claim a
confirmed upstream vLLM bug.

All route predictions are written and hashed before a separate process opens the label file.
Statistics use paired bootstrap confidence intervals and exact two-sided McNemar tests.

## Production system

- Qwen3-4B, BF16, LoRA rank 16 / alpha 32;
- two RTX 3090 24GB GPUs;
- memory-balanced DDP for SFT;
- custom Transformers/PEFT three-policy OPD loop;
- atomic checkpoints for LoRA, optimizer, scheduler, CPU/CUDA RNG, cursor and sampler state;
- candidate updates committed transactionally or rolled back completely;
- SHA-256 bindings across model, data, config, checkpoint and result identities.

veRL and vLLM adapters are included for compatibility and diagnostics. They were not the
trainer that produced the final OPD evidence.
