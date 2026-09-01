"""Minimal, CPU-runnable OPD / CA-OPD training loop (docs/METHOD.md §14 Phase 0).

Scope and status
----------------
This is the **reference implementation** used to prove the training closed loop
before any GPU time is bought: rollout -> frozen old logprobs -> teacher scoring
on the *same tokens* -> reverse-KL advantage -> domain KL safety scaling -> PPO
update -> window-level teacher routing -> checkpoint save/resume.

The formal Phase 1-3 stack is veRL + vLLM (docs/METHOD.md §6). This module is
not a competing trainer: it exists so that (a) every mathematical property has an
executable end-to-end witness on CPU, and (b) a veRL integration can be diffed
against a known-correct baseline. Results tables must never come from here, and
``summary.json`` records ``implementation: "cpu_reference"`` to make that
impossible to confuse.

What is deliberately *not* here: LoRA, vLLM, Ray, multi-GPU placement, real
tokenizers. Those arrive with the veRL runner in P1.
"""

from __future__ import annotations

import copy
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from src.opd.core import (
    DomainKLController,
    OPDBatch,
    build_opd_batch,
    masked_mean,
    policy_entropy,
    ppo_policy_loss,
    reverse_kl_per_token,
    scale_and_clip_advantage,
    selected_token_logprobs,
    teacher_student_advantage,
)
from src.opd.router import BASE, MEDICAL, ConstraintAwareRouter, FixedRatioRouter, RouterConfig
from src.opd.toy_lm import ToyCausalLM, ToyLMConfig
from src.utils.config import ConfigError, FieldSpec, load_config
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import MetricsLogger
from src.utils.run_meta import RunMetadata, make_run_id
from src.utils.seeding import derive_seed, seed_everything

# Domain label attached to every batch; also the key of the KL controller.
DOMAIN_BY_TEACHER = {MEDICAL: "medical", BASE: "general"}

LOOP_SCHEMA: Dict[str, object] = {
    "run": {
        "name": FieldSpec((str,), doc="run name prefix"),
        "purpose": FieldSpec((str,), doc="why this run exists"),
        "baseline_id": FieldSpec((str,), doc="B0..B5 or O1 (PROJECT_PLAN §10)"),
        "seed": FieldSpec((int,), doc="master seed"),
        "output_root": FieldSpec((str,), doc="where run directories are created"),
        "implementation": FieldSpec((str,), required=False, default="cpu_reference", choices=["cpu_reference"]),
    },
    "model": {
        "kind": FieldSpec((str,), choices=["toy"], doc="toy = tiny CPU model; hf/verl arrive in P1"),
        "vocab_size": FieldSpec((int,), bounds=(4, None)),
        "hidden_size": FieldSpec((int,), bounds=(2, None)),
        "num_heads": FieldSpec((int,), bounds=(1, None)),
        "max_position": FieldSpec((int,), bounds=(8, None)),
        "pad_token_id": FieldSpec((int,), bounds=(0, None)),
        "eos_token_id": FieldSpec((int,), bounds=(0, None)),
        "medical_teacher_bias_token": FieldSpec((int,), required=False, default=None),
        "medical_teacher_bias_strength": FieldSpec((float,), required=False, default=3.0),
    },
    "data": {
        "kind": FieldSpec((str,), choices=["synthetic"], doc="synthetic token pools for the CPU dry-run"),
        "num_medical_prompts": FieldSpec((int,), bounds=(1, None)),
        "num_general_prompts": FieldSpec((int,), bounds=(1, None)),
        "prompt_length": FieldSpec((int,), bounds=(1, None)),
    },
    "rollout": {
        "prompt_batch_size": FieldSpec((int,), bounds=(1, None)),
        "group_size": FieldSpec((int,), bounds=(1, None)),
        "max_new_tokens": FieldSpec((int,), bounds=(1, None)),
        "temperature": FieldSpec((float,), bounds=(0.01, None)),
        "top_k": FieldSpec((int,), required=False, default=0, bounds=(0, None)),
        "include_eos_in_loss": FieldSpec((bool,), required=False, default=True),
    },
    "opd": {
        "beta": FieldSpec((float,), bounds=(0.0001, None), doc="advantage scale"),
        "clip_range": FieldSpec((float,), bounds=(0.0001, None)),
        "advantage_max": FieldSpec((float,), bounds=(0.0001, None)),
        "kl_kappa_medical": FieldSpec((float,), bounds=(0.0001, None)),
        "kl_kappa_general": FieldSpec((float,), bounds=(0.0001, None)),
        "kl_ema_rho": FieldSpec((float,), bounds=(0.0, 0.999)),
        "reduction": FieldSpec((str,), choices=["token_mean", "seq_mean_token_mean"]),
        "ppo_epochs": FieldSpec((int,), required=False, default=1, bounds=(1, None)),
    },
    "optim": {
        "lr": FieldSpec((float,), bounds=(0.0, None)),
        "max_steps": FieldSpec((int,), bounds=(1, None)),
        "gradient_accumulation_steps": FieldSpec((int,), required=False, default=1, bounds=(1, None)),
        "max_grad_norm": FieldSpec((float,), required=False, default=1.0, bounds=(0.0, None)),
    },
    "router": {
        "kind": FieldSpec((str,), choices=["constraint_aware", "fixed_ratio", "single_teacher"]),
        "fixed_p_medical": FieldSpec((float,), required=False, default=0.5, bounds=(0.0, 1.0)),
        "single_teacher_id": FieldSpec((str,), required=False, default=MEDICAL, choices=[MEDICAL, BASE]),
        "config": {
            "medical_target": FieldSpec((float,), bounds=(0.0, 1.0)),
            "general_baseline": FieldSpec((float,), bounds=(0.0, 1.0)),
            "delta": FieldSpec((float,), bounds=(0.0, 1.0)),
            "scale_medical": FieldSpec((float,), bounds=(0.0001, None)),
            "scale_general": FieldSpec((float,), bounds=(0.0001, None)),
            "rho": FieldSpec((float,), bounds=(0.0, 0.999)),
            "tau": FieldSpec((float,), bounds=(0.0001, None)),
            "p_min": FieldSpec((float,), bounds=(0.0001, 1.0)),
            "p_max": FieldSpec((float,), bounds=(0.0001, 1.0)),
            "window_steps": FieldSpec((int,), bounds=(1, None)),
            "windows_below_to_recover": FieldSpec((int,), bounds=(1, None)),
            "windows_above_to_release": FieldSpec((int,), bounds=(1, None)),
            "early_stop_patience": FieldSpec((int,), bounds=(1, None)),
            "early_stop_min_improvement": FieldSpec((float,), bounds=(0.0, 1.0)),
            "initial_p_medical": FieldSpec((float,), bounds=(0.0, 1.0)),
        },
    },
    "controller_dev": {
        "mode": FieldSpec(
            (str,),
            choices=["synthetic"],
            doc="synthetic = deterministic fake accuracies for the CPU dry-run; "
            "real MCQ evaluation is wired in P1 via src.eval.runner",
        ),
        "medical_start": FieldSpec((float,), bounds=(0.0, 1.0)),
        "medical_gain_per_window": FieldSpec((float,), bounds=(-1.0, 1.0)),
        "general_start": FieldSpec((float,), bounds=(0.0, 1.0)),
        "general_loss_per_medical_window": FieldSpec((float,), bounds=(-1.0, 1.0)),
    },
    "checkpoint": {
        "every_steps": FieldSpec((int,), bounds=(1, None)),
        "keep_last": FieldSpec((int,), required=False, default=2, bounds=(1, None)),
    },
}


# ---------------------------------------------------------------------------
# teachers
# ---------------------------------------------------------------------------


class TeacherRegistry:
    """Forward-only teacher models addressed by ``teacher_id``.

    Mirrors the GPU-1 design (docs/METHOD.md §11.5): the *same* backbone object
    can serve both routes, with the medical route differing only by an adapter.
    Here the "adapter" is a logit bias on a shared module, which keeps the
    routing contract testable without LoRA.
    """

    def __init__(self, teachers: Mapping[str, nn.Module]):
        missing = set(DOMAIN_BY_TEACHER) - set(teachers)
        if missing:
            raise ValueError(f"teacher registry is missing route(s): {sorted(missing)}")
        for tid, model in teachers.items():
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            if trainable:
                raise ValueError(f"teacher {tid!r} has trainable parameters: {trainable[:3]}")
            model.eval()
        self._teachers = dict(teachers)
        self.call_counts: Dict[str, int] = {tid: 0 for tid in teachers}
        self.shared_backbone = len({id(m) for m in teachers.values()}) < len(teachers)

    def identity(self, teacher_id: str) -> str:
        if teacher_id not in self._teachers:
            raise KeyError(f"unknown teacher_id {teacher_id!r}; known: {sorted(self._teachers)}")
        return teacher_id

    @torch.no_grad()
    def score(self, teacher_id: str, batch: OPDBatch) -> Tensor:
        """Teacher logprobs ``[B, T-1]`` on the student's own tokens.

        The teacher only ever performs a forward pass over ``batch.input_ids``;
        there is no generate() call anywhere in this class, which is how
        "Teacher 不重新生成答案" is enforced structurally rather than by comment.
        """
        model = self._teachers[self.identity(teacher_id)]
        self.call_counts[teacher_id] += 1
        logits = model(batch.input_ids, batch.attention_mask)
        return selected_token_logprobs(logits, batch).detach()


def build_toy_teachers(
    student_cfg: ToyLMConfig,
    seed: int,
    medical_bias_token: Optional[int],
    medical_bias_strength: float,
) -> TeacherRegistry:
    """Two teacher routes over one shared backbone instance.

    ``base`` is the raw backbone; ``medical`` is the same module with a medical
    "adapter" (logit bias). We deep-copy once so the medical route can hold its
    adapter while both remain frozen.
    """
    torch.manual_seed(seed)
    base = ToyCausalLM(student_cfg)
    medical = copy.deepcopy(base)
    if medical_bias_token is not None:
        with torch.no_grad():
            medical.logit_bias[medical_bias_token] += float(medical_bias_strength)
    for model in (base, medical):
        for p in model.parameters():
            p.requires_grad_(False)
    return TeacherRegistry({BASE: base, MEDICAL: medical})


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    completions: List[List[int]]
    sampling_logprobs: List[List[float]]
    num_tokens: int
    seconds: float


@torch.no_grad()
def sample_completions(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int,
    top_k: int = 0,
    generator: Optional[torch.Generator] = None,
) -> RolloutResult:
    """Autoregressive sampling, one sequence at a time (clarity over speed).

    Returns the sampled tokens *and* the logprob the sampling policy assigned to
    each sampled token. Those two must agree with a later forward pass over the
    assembled batch; ``tests/test_opd_loop.py`` asserts that, which is the CPU
    analogue of "vLLM sampler logprob == training-time logprob".
    """
    was_training = model.training
    model.eval()
    t0 = time.time()
    completions: List[List[int]] = []
    logprobs: List[List[float]] = []
    total = 0
    for prompt in prompts:
        ids = list(prompt)
        comp: List[int] = []
        lps: List[float] = []
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids], dtype=torch.int64)
            logits = model(inp)[0, -1, :].float() / temperature
            if top_k and top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            comp.append(token)
            lps.append(float(log_probs[token]))
            ids.append(token)
            if token == eos_token_id:
                break
        if not comp:  # pragma: no cover - max_new_tokens >= 1 guarantees content
            raise RuntimeError("rollout produced an empty completion")
        completions.append(comp)
        logprobs.append(lps)
        total += len(comp)
    if was_training:
        model.train()
    return RolloutResult(completions, logprobs, total, time.time() - t0)


# ---------------------------------------------------------------------------
# synthetic controller-dev evaluator (CPU dry-run only)
# ---------------------------------------------------------------------------


class SyntheticControllerDev:
    """Deterministic stand-in for the controller-dev evaluation.

    It is **not** a model evaluation: it is a scripted ability trajectory whose
    only purpose is to exercise the router inside the loop on a machine with no
    GPU. It self-identifies as controller-dev so the router accepts it, and every
    run that uses it records ``controller_dev.mode = synthetic`` in
    ``summary.json`` so no number produced here can be mistaken for a result.
    """

    split = "controller_dev"
    is_synthetic = True

    def __init__(
        self,
        medical_start: float,
        medical_gain_per_window: float,
        general_start: float,
        general_loss_per_medical_window: float,
    ):
        self.medical = float(medical_start)
        self.general = float(general_start)
        self.medical_gain = float(medical_gain_per_window)
        self.general_loss = float(general_loss_per_medical_window)

    def allows_control_decisions(self) -> bool:
        return True

    def evaluate(self, medical_fraction: float) -> Tuple[float, float]:
        """Advance the scripted trajectory by one window and return accuracies.

        Medical ability improves with the fraction of medical-teacher batches;
        general ability decays with that same fraction and partially recovers
        when the Base Teacher is used. This mirrors the "seesaw" the project is
        studying, but it is a *stub*, not evidence for it.
        """
        self.medical = min(1.0, max(0.0, self.medical + self.medical_gain * medical_fraction))
        drift = -self.general_loss * medical_fraction + 0.5 * self.general_loss * (1.0 - medical_fraction)
        self.general = min(1.0, max(0.0, self.general + drift))
        return self.medical, self.general


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    teacher_id: str
    domain: str
    loss: float
    reverse_kl: float
    kl_scale: float
    advantage_mean: float
    advantage_clip_fraction: float
    ratio_mean: float
    clip_fraction: float
    entropy: float
    num_completion_tokens: int
    rollout_tokens_per_second: float
    step_seconds: float


@dataclass
class LoopState:
    """Everything that must survive a checkpoint to make resume exact."""

    step: int = 0
    router_state: Dict[str, object] = field(default_factory=dict)
    kl_state: Dict[str, object] = field(default_factory=dict)
    controller_dev: Dict[str, float] = field(default_factory=dict)


def build_router(config: Mapping[str, object], seed: int, evaluator: Optional[object]):
    kind = config["kind"]
    if kind == "constraint_aware":
        return ConstraintAwareRouter(
            RouterConfig.from_mapping(config["config"]),  # type: ignore[arg-type]
            evaluator=evaluator,  # type: ignore[arg-type]
            seed=seed,
        )
    window = int(config["config"]["window_steps"])  # type: ignore[index]
    if kind == "fixed_ratio":
        return FixedRatioRouter(p_medical=float(config["fixed_p_medical"]), window_steps=window, seed=seed)
    if kind == "single_teacher":
        p = 1.0 if config["single_teacher_id"] == MEDICAL else 0.0
        return FixedRatioRouter(p_medical=p, window_steps=window, seed=seed)
    raise ConfigError(f"unknown router kind {kind!r}")


def make_synthetic_prompt_pools(
    num_medical: int,
    num_general: int,
    prompt_length: int,
    vocab_size: int,
    pad_token_id: int,
    eos_token_id: int,
    seed: int,
) -> Dict[str, List[List[int]]]:
    """Two disjoint synthetic token pools standing in for the real prompt pools.

    Reserved ids (pad/eos) are excluded so a "prompt" can never contain EOS,
    which would make the completion mask ambiguous.
    """
    rng = torch.Generator().manual_seed(derive_seed(seed, "synthetic-prompts"))
    reserved = {pad_token_id, eos_token_id}
    usable = [t for t in range(vocab_size) if t not in reserved]
    if len(usable) < 4:
        raise ValueError("vocab too small for synthetic prompts")
    half = len(usable) // 2
    pools = {"medical": usable[:half], "general": usable[half:]}
    out: Dict[str, List[List[int]]] = {}
    for domain, count in (("medical", num_medical), ("general", num_general)):
        tokens = pools[domain]
        prompts = []
        for _ in range(count):
            idx = torch.randint(0, len(tokens), (prompt_length,), generator=rng).tolist()
            prompts.append([tokens[i] for i in idx])
        out[domain] = prompts
    return out


def run_loop(
    config_path: str | Path,
    output_dir: Optional[str | Path] = None,
    resume_from: Optional[str | Path] = None,
    max_steps_override: Optional[int] = None,
) -> Dict[str, object]:
    """Run the CPU reference loop and return its ``summary.json`` payload."""
    cfg = load_config(config_path, LOOP_SCHEMA)
    run_cfg = cfg["run"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    roll_cfg = cfg["rollout"]
    opd_cfg = cfg["opd"]
    optim_cfg = cfg["optim"]
    router_cfg = cfg["router"]
    dev_cfg = cfg["controller_dev"]
    ckpt_cfg = cfg["checkpoint"]

    seed = int(run_cfg["seed"])
    seed_state = seed_everything(seed)
    max_steps = int(max_steps_override or optim_cfg["max_steps"])

    run_id = make_run_id(str(run_cfg["name"]), seed)
    run_dir = ensure_dir(Path(output_dir) if output_dir else Path(str(run_cfg["output_root"])) / run_id)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    logger = MetricsLogger(run_dir, run_id=run_id)

    # -- model / teachers
    toy_cfg = ToyLMConfig(
        vocab_size=int(model_cfg["vocab_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        num_heads=int(model_cfg["num_heads"]),
        max_position=int(model_cfg["max_position"]),
    )
    student = ToyCausalLM(toy_cfg)
    teachers = build_toy_teachers(
        toy_cfg,
        seed=derive_seed(seed, "teacher"),
        medical_bias_token=model_cfg["medical_teacher_bias_token"],
        medical_bias_strength=float(model_cfg["medical_teacher_bias_strength"]),
    )
    optimizer = torch.optim.SGD(student.parameters(), lr=float(optim_cfg["lr"]))

    kl_controller = DomainKLController(
        kappa={"medical": float(opd_cfg["kl_kappa_medical"]), "general": float(opd_cfg["kl_kappa_general"])},
        rho=float(opd_cfg["kl_ema_rho"]),
        domains=["medical", "general"],
    )
    controller_dev = SyntheticControllerDev(
        medical_start=float(dev_cfg["medical_start"]),
        medical_gain_per_window=float(dev_cfg["medical_gain_per_window"]),
        general_start=float(dev_cfg["general_start"]),
        general_loss_per_medical_window=float(dev_cfg["general_loss_per_medical_window"]),
    )
    router = build_router(router_cfg, seed=derive_seed(seed, "router"), evaluator=controller_dev)

    prompt_pools = make_synthetic_prompt_pools(
        num_medical=int(data_cfg["num_medical_prompts"]),
        num_general=int(data_cfg["num_general_prompts"]),
        prompt_length=int(data_cfg["prompt_length"]),
        vocab_size=toy_cfg.vocab_size,
        pad_token_id=int(model_cfg["pad_token_id"]),
        eos_token_id=int(model_cfg["eos_token_id"]),
        seed=seed,
    )

    metadata = RunMetadata(
        run_id=run_id,
        purpose=str(run_cfg["purpose"]),
        baseline_id=str(run_cfg["baseline_id"]),
        config_path=str(config_path),
        seed=seed,
        model=f"toy:{toy_cfg.vocab_size}v/{toy_cfg.hidden_size}h",
        notes="CPU reference implementation; not a source of reportable results",
        extra={"seed_state": seed_state.as_dict(), "implementation": str(run_cfg["implementation"])},
    )
    metadata.save(run_dir)

    rollout_rng = torch.Generator().manual_seed(derive_seed(seed, "rollout"))
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(
            resume_from, student, optimizer, router, kl_controller, controller_dev, rollout_rng=rollout_rng
        )

    records: List[StepRecord] = []
    stopped_early = False
    stop_reason = ""

    for step in range(start_step + 1, max_steps + 1):
        t_step = time.time()
        teacher_id = router.sample_teacher()
        domain = DOMAIN_BY_TEACHER[teacher_id]

        # -- pick prompts for this domain
        pool = prompt_pools[domain]
        n_prompts = min(int(roll_cfg["prompt_batch_size"]), len(pool))
        offset = ((step - 1) * n_prompts) % len(pool)
        chosen = [pool[(offset + i) % len(pool)] for i in range(n_prompts)]
        prompts = [p for p in chosen for _ in range(int(roll_cfg["group_size"]))]

        # -- rollout (student generates; teacher never does)
        rollout = sample_completions(
            student,
            prompts,
            max_new_tokens=int(roll_cfg["max_new_tokens"]),
            temperature=float(roll_cfg["temperature"]),
            eos_token_id=int(model_cfg["eos_token_id"]),
            top_k=int(roll_cfg["top_k"]),
            generator=rollout_rng,
        )
        batch = build_opd_batch(
            prompt_ids=prompts,
            completion_ids=rollout.completions,
            pad_token_id=int(model_cfg["pad_token_id"]),
            eos_token_id=int(model_cfg["eos_token_id"]),
            domains=tuple([domain] * len(prompts)),
            include_eos_in_loss=bool(roll_cfg["include_eos_in_loss"]),
            max_length=toy_cfg.max_position,
        )
        mask = batch.target_mask()

        # -- frozen old logprobs (rollout policy), teacher logprobs (same tokens)
        with torch.no_grad():
            old_logprobs = selected_token_logprobs(student(batch.input_ids, batch.attention_mask), batch).detach()
        teacher_logprobs = teachers.score(teacher_id, batch)

        # -- reverse KL -> domain EMA -> safety scale
        r_kl = reverse_kl_per_token(old_logprobs, teacher_logprobs)
        reverse_kl_mean = float(masked_mean(r_kl, mask))
        kl_controller.update(domain, reverse_kl_mean)
        kl_scale = kl_controller.scale(domain)

        advantages = teacher_student_advantage(old_logprobs, teacher_logprobs, beta=float(opd_cfg["beta"]))
        advantages, adv_clip_flags = scale_and_clip_advantage(
            advantages, scales=kl_scale, a_max=float(opd_cfg["advantage_max"])
        )

        # -- PPO update
        accum = int(optim_cfg["gradient_accumulation_steps"])
        optimizer.zero_grad()
        last_stats = None
        entropy_value = 0.0
        for epoch in range(int(opd_cfg["ppo_epochs"])):
            logits = student(batch.input_ids, batch.attention_mask)
            new_logprobs = selected_token_logprobs(logits, batch)
            loss, stats = ppo_policy_loss(
                new_logprobs,
                old_logprobs,
                advantages,
                mask,
                clip_range=float(opd_cfg["clip_range"]),
                reduction=str(opd_cfg["reduction"]),  # type: ignore[arg-type]
                advantage_clip_flags=adv_clip_flags,
            )
            (loss / accum).backward()
            last_stats = stats
            if epoch == 0:
                with torch.no_grad():
                    entropy_value = float(policy_entropy(logits.detach(), mask))
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(optim_cfg["max_grad_norm"]))
        )
        optimizer.step()
        assert last_stats is not None

        step_seconds = time.time() - t_step
        tokens_per_second = rollout.num_tokens / max(rollout.seconds, 1e-6)

        logger.log(
            step,
            {
                **last_stats.as_metrics(),
                "train/lr": float(optim_cfg["lr"]),
                "train/grad_norm": grad_norm,
                "opd/reverse_kl_mean": reverse_kl_mean,
                "opd/reverse_kl_std": float(r_kl[mask > 0].std(unbiased=False)) if mask.sum() > 1 else 0.0,
                "opd/kl_scale": kl_scale,
                "opd/teacher_id": teacher_id,
                "policy/entropy": entropy_value,
                "system/step_seconds": step_seconds,
                "system/rollout_tokens_per_second": tokens_per_second,
            },
            domain=domain,
            phase="train",
        )
        records.append(
            StepRecord(
                step=step,
                teacher_id=teacher_id,
                domain=domain,
                loss=last_stats.loss,
                reverse_kl=reverse_kl_mean,
                kl_scale=kl_scale,
                advantage_mean=last_stats.advantage_mean,
                advantage_clip_fraction=last_stats.advantage_clip_fraction,
                ratio_mean=last_stats.ratio_mean,
                clip_fraction=last_stats.clip_fraction,
                entropy=entropy_value,
                num_completion_tokens=last_stats.num_tokens,
                rollout_tokens_per_second=tokens_per_second,
                step_seconds=step_seconds,
            )
        )

        # -- window boundary: controller-dev evaluation drives routing
        if router.is_window_boundary(step):
            medical_fraction = router.realised_medical_fraction() or 0.0
            medical_acc, general_acc = controller_dev.evaluate(medical_fraction)
            decision = router.update(medical_acc, general_acc, step=step)
            logger.log(
                step,
                {
                    "eval_dev/medical_accuracy": medical_acc,
                    "eval_dev/general_accuracy": general_acc,
                    **{k: v for k, v in decision.as_metrics().items()},
                },
                window=decision.window,
                phase="controller_dev",
            )
            if decision.should_stop:
                stopped_early = True
                stop_reason = decision.reason
                save_checkpoint(
                    run_dir, step, student, optimizer, router, kl_controller, controller_dev, ckpt_cfg,
                    rollout_rng=rollout_rng,
                )
                break

        if step % int(ckpt_cfg["every_steps"]) == 0:
            save_checkpoint(
                run_dir, step, student, optimizer, router, kl_controller, controller_dev, ckpt_cfg,
                rollout_rng=rollout_rng,
            )

    summary = {
        "run_id": run_id,
        "implementation": str(run_cfg["implementation"]),
        "baseline_id": str(run_cfg["baseline_id"]),
        "seed": seed,
        "steps_completed": records[-1].step if records else start_step,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "teacher_counts": dict(router.teacher_counts),
        "realised_medical_fraction": router.realised_medical_fraction(),
        "final_loss": records[-1].loss if records else None,
        "mean_reverse_kl": (sum(r.reverse_kl for r in records) / len(records)) if records else None,
        "kl_controller_ema": dict(kl_controller.ema),
        "controller_dev": {
            "mode": str(dev_cfg["mode"]),
            "medical_accuracy": controller_dev.medical,
            "general_accuracy": controller_dev.general,
            "warning": "synthetic scripted trajectory - NOT a model evaluation",
        },
        "router_windows": [d.as_dict() for d in getattr(router, "history", [])],
        "metrics_path": str(logger.path),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    router,
    kl_controller: DomainKLController,
    controller_dev: SyntheticControllerDev,
    ckpt_cfg: Mapping[str, object] | None = None,
    rollout_rng: Optional[torch.Generator] = None,
) -> Path:
    """Persist everything needed for an exact resume.

    "Everything" includes the rollout RNG: without it a resumed run samples
    different completions and the resume is only approximately equivalent, which
    would silently break run comparability (docs/METHOD.md §15).
    """
    ckpt_dir = ensure_dir(Path(run_dir) / "checkpoints" / f"step-{step}")
    torch.save(
        {
            "step": step,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "router": router.state_dict(),
            "kl_controller": kl_controller.state_dict(),
            "controller_dev": {"medical": controller_dev.medical, "general": controller_dev.general},
            "torch_rng_state": torch.get_rng_state(),
            "rollout_rng_state": rollout_rng.get_state() if rollout_rng is not None else None,
        },
        ckpt_dir / "state.pt",
    )
    if ckpt_cfg:
        _prune_checkpoints(Path(run_dir) / "checkpoints", int(ckpt_cfg.get("keep_last", 2)))
    return ckpt_dir


def _prune_checkpoints(root: Path, keep_last: int) -> None:
    """Keep only the newest ``keep_last`` checkpoints (disk discipline)."""
    if not root.exists():
        return
    steps = sorted(
        (int(p.name.split("-")[1]), p) for p in root.iterdir() if p.is_dir() and p.name.startswith("step-")
    )
    for _, path in steps[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(path, ignore_errors=True)


def load_checkpoint(
    path: str | Path,
    student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    router,
    kl_controller: DomainKLController,
    controller_dev: Optional[SyntheticControllerDev] = None,
    rollout_rng: Optional[torch.Generator] = None,
) -> int:
    p = Path(path)
    if p.is_dir():
        p = p / "state.pt"
    payload = torch.load(p, map_location="cpu")
    student.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    router.load_state_dict(payload["router"])
    kl_controller.load_state_dict(payload["kl_controller"])
    if controller_dev is not None and "controller_dev" in payload:
        controller_dev.medical = float(payload["controller_dev"]["medical"])
        controller_dev.general = float(payload["controller_dev"]["general"])
    rng = payload.get("torch_rng_state")
    if rng is not None:
        torch.set_rng_state(rng if isinstance(rng, torch.Tensor) else torch.tensor(rng, dtype=torch.uint8))
    roll = payload.get("rollout_rng_state")
    if roll is not None and rollout_rng is not None:
        rollout_rng.set_state(roll if isinstance(roll, torch.Tensor) else torch.tensor(roll, dtype=torch.uint8))
    return int(payload["step"])
