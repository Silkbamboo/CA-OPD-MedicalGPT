"""P4.8e fixed-token real-GPU mathematical differential gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from types import MethodType
from typing import Any, Mapping, Sequence

from src.opd.production_b2_calibration_backend_v2 import (
    project_production_b2_memory_runtime_v1,
)


SCALAR_ATOL = 1e-6
SCALAR_RTOL = 1e-6
GRADIENT_ATOL = 2e-6
GRADIENT_RTOL = 2e-5


class B2GpuMathDifferentialV1Error(RuntimeError):
    """Legacy and memory-balanced GPU math differ."""


def _fail(message: str) -> None:
    raise B2GpuMathDifferentialV1Error(message)


def _fixed_probe_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror the production pre-update identity probe from fixed trajectories."""

    probe_rows = [
        {**dict(row), "response_ids": list(row.get("response_ids", ()))[:32]}
        for row in rows
    ]
    if not probe_rows or any(not row["response_ids"] for row in probe_rows):
        _fail("fixed differential checkpoint probe is empty")
    return probe_rows


def _max_errors(left: Any, right: Any) -> tuple[float, float]:
    import torch

    lhs = left.detach().float().cpu().reshape(-1)
    rhs = right.detach().float().cpu().reshape(-1)
    if lhs.shape != rhs.shape or lhs.numel() == 0:
        _fail("differential tensor shape differs")
    difference = (lhs - rhs).abs()
    denominator = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-30)
    return float(difference.max()), float((difference / denominator).max())


def _allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    import torch

    lhs = left.detach().float().cpu()
    rhs = right.detach().float().cpu()
    return bool(
        lhs.shape == rhs.shape
        and torch.isfinite(lhs).all()
        and torch.isfinite(rhs).all()
        and torch.allclose(lhs, rhs, atol=atol, rtol=rtol)
    )


def compare_differential_snapshots(
    legacy: Mapping[str, Any], balanced: Mapping[str, Any]
) -> dict[str, Any]:
    if legacy.get("fixed_tokens_sha256") != balanced.get("fixed_tokens_sha256"):
        _fail("legacy/new did not reuse the fixed rollout tokens")
    if legacy.get("initial_adapter_sha256") != balanced.get(
        "initial_adapter_sha256"
    ):
        _fail("legacy/new fresh v0 adapter identity differs")
    expected_counts = {
        "optimizer": 1,
        "scheduler": 1,
        "export": 1,
        "refresh": 1,
        "version_increment": 1,
    }
    if not (
        legacy.get("counts") == balanced.get("counts") == expected_counts
    ):
        _fail("legacy/new lifecycle counts differ")

    q_abs, q_rel = _max_errors(
        legacy.get("q_target_logprob"), balanced.get("q_target_logprob")
    )
    if not _allclose(
        legacy["q_target_logprob"],
        balanced["q_target_logprob"],
        atol=SCALAR_ATOL,
        rtol=SCALAR_RTOL,
    ):
        _fail("q target-token logprob exceeds the frozen scalar tolerance")
    scalar_errors: dict[str, dict[str, float]] = {}
    for field in (
        "objective",
        "loss",
        "backward_loss",
        "grad_norm_before_clip",
        "grad_norm",
    ):
        left = float(legacy.get(field))
        right = float(balanced.get(field))
        if not math.isfinite(left) or not math.isfinite(right):
            _fail("differential scalar is non-finite")
        absolute = abs(left - right)
        relative = absolute / max(abs(left), abs(right), 1e-30)
        if absolute > SCALAR_ATOL + SCALAR_RTOL * abs(left):
            _fail(f"{field} exceeds the frozen scalar tolerance")
        scalar_errors[field] = {
            "absolute_error": absolute,
            "relative_error": relative,
        }

    legacy_gradients = legacy.get("gradients")
    balanced_gradients = balanced.get("gradients")
    legacy_deltas = legacy.get("deltas")
    balanced_deltas = balanced.get("deltas")
    if not all(
        isinstance(value, Mapping)
        for value in (
            legacy_gradients,
            balanced_gradients,
            legacy_deltas,
            balanced_deltas,
        )
    ):
        _fail("differential LoRA tensor evidence is absent")
    names = set(legacy_gradients)
    if not names or not (
        names
        == set(balanced_gradients)
        == set(legacy_deltas)
        == set(balanced_deltas)
    ):
        _fail("differential LoRA tensor names differ")
    gradient_abs = 0.0
    gradient_rel = 0.0
    delta_abs = 0.0
    delta_rel = 0.0
    for name in sorted(names):
        current_abs, current_rel = _max_errors(
            legacy_gradients[name], balanced_gradients[name]
        )
        gradient_abs = max(gradient_abs, current_abs)
        gradient_rel = max(gradient_rel, current_rel)
        if not _allclose(
            legacy_gradients[name],
            balanced_gradients[name],
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
        ):
            _fail(f"LoRA gradient tolerance failed: {name}")
        current_abs, current_rel = _max_errors(
            legacy_deltas[name], balanced_deltas[name]
        )
        delta_abs = max(delta_abs, current_abs)
        delta_rel = max(delta_rel, current_rel)
        if not _allclose(
            legacy_deltas[name],
            balanced_deltas[name],
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
        ):
            _fail(f"LoRA delta tolerance failed: {name}")

    return {
        "schema_version": 1,
        "artifact_kind": "p4_8e_gpu_math_equivalence_comparison_v1",
        "passed": True,
        "fixed_tokens_sha256": legacy["fixed_tokens_sha256"],
        "initial_adapter_sha256": legacy["initial_adapter_sha256"],
        "tolerance": {
            "scalar_atol": SCALAR_ATOL,
            "scalar_rtol": SCALAR_RTOL,
            "gradient_atol": GRADIENT_ATOL,
            "gradient_rtol": GRADIENT_RTOL,
        },
        "q_target_logprob_max_abs_error": q_abs,
        "q_target_logprob_max_relative_error": q_rel,
        "objective_abs_error": scalar_errors["objective"]["absolute_error"],
        "loss_abs_error": scalar_errors["loss"]["absolute_error"],
        "backward_loss_abs_error": scalar_errors["backward_loss"][
            "absolute_error"
        ],
        "grad_norm_abs_error": scalar_errors["grad_norm"]["absolute_error"],
        "grad_norm_before_clip_abs_error": scalar_errors[
            "grad_norm_before_clip"
        ]["absolute_error"],
        "gradient_max_abs_error": gradient_abs,
        "gradient_max_relative_error": gradient_rel,
        "delta_max_abs_error": delta_abs,
        "delta_max_relative_error": delta_rel,
        "compared_lora_tensor_count": len(names),
        "compared_gradient_stage": "post_global_clip_pre_optimizer",
        "strict_counts_equal": True,
        "legacy_counts": dict(legacy["counts"]),
        "balanced_counts": dict(balanced["counts"]),
        "legacy_backbone_backward_calls_per_prompt": legacy.get(
            "backbone_backward_calls_per_prompt"
        ),
        "balanced_backbone_backward_calls_per_prompt": balanced.get(
            "backbone_backward_calls_per_prompt"
        ),
        "legacy_lm_head_chunk_count": legacy.get("lm_head_chunk_count"),
        "balanced_lm_head_chunk_count": balanced.get("lm_head_chunk_count"),
        "legacy_retain_graph_calls": legacy.get("retain_graph_calls"),
        "balanced_retain_graph_calls": balanced.get("retain_graph_calls"),
        "legacy_timings_seconds": dict(legacy.get("timings_seconds", {})),
        "balanced_timings_seconds": dict(balanced.get("timings_seconds", {})),
        "legacy_peak_memory_bytes": dict(legacy.get("peak_memory_bytes", {})),
        "balanced_peak_memory_bytes": dict(balanced.get("peak_memory_bytes", {})),
    }


def _fixed_tokens_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = [
        {
            "fixture_id": str(row["fixture_id"]),
            "response_ids": [int(value) for value in row["response_ids"]],
            "rollout_behavior_logprob": [
                float(value) for value in row["rollout_behavior_logprob"]
            ],
        }
        for row in rows
    ]
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    payload = json.dumps(
        [int(value) for value in values], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _differential_sample_identity(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hash token-only identity without persisting raw medical text."""

    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt = [int(value) for value in row["prompt_ids"]]
        completion = [int(value) for value in row["response_ids"]]
        combined = prompt + completion
        attention_mask = [1] * len(combined)
        response_mask = [0] * len(prompt) + [1] * len(completion)
        valid_mask = [1] * len(completion)
        samples.append(
            {
                "sample_id": str(row["fixture_id"]),
                "source": str(row["source_role"]),
                "prompt_token_count": len(prompt),
                "completion_token_count": len(completion),
                "valid_token_count": len(completion),
                "prompt_token_sha256": _integer_sequence_sha256(prompt),
                "completion_token_sha256": _integer_sequence_sha256(completion),
                "combined_token_sequence_sha256": _integer_sequence_sha256(
                    combined
                ),
                "attention_mask_sha256": _integer_sequence_sha256(
                    attention_mask
                ),
                "response_mask_sha256": _integer_sequence_sha256(
                    response_mask
                ),
                "valid_token_mask_sha256": _integer_sequence_sha256(
                    valid_mask
                ),
                "trajectory_weight": 1.0,
                "group_weight": 1.0,
                "prompt_weight": 0.25,
            }
        )
    return samples


def _cpu_rng_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().cpu().clone(),
        "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(torch: Any, state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(list(state["cuda"]))


def _reset_peak_memory_stats_all(torch: Any) -> None:
    """Initialize both CUDA contexts before using the allocator reset API.

    PyTorch 2.8 with the AutoDL 580 driver can reject resetPeakMemoryStats on
    a device whose primary context has not yet been initialized.  Reading
    mem_get_info is a telemetry-only initialization boundary and does not load
    a model or change training math.
    """

    if torch.cuda.device_count() != 2:
        _fail("GPU differential requires exactly two visible CUDA devices")
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)


def _clear_cublas_workspaces(torch: Any) -> None:
    """Release pinned-version cuBLAS workspaces before an in-process gate."""

    clear = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
    if not callable(clear):
        _fail("pinned PyTorch cuBLAS workspace cleanup API is unavailable")
    clear()


def _run_fixed_path(
    *,
    session: Any,
    rows: list[dict[str, Any]],
    provenance: Mapping[str, Any],
    fixed_tokens_sha256: str,
    rng_state: Mapping[str, Any],
    mode: str,
    reused_old_actor: tuple[Any, Any] | None = None,
    reused_teacher: tuple[Any, Any] | None = None,
) -> tuple[dict[str, Any], tuple[Any, Any], tuple[Any, Any]]:
    import torch

    if mode not in {"legacy_prompt_equal", "memory_balanced"}:
        _fail("unknown differential execution mode")
    _restore_rng(torch, rng_state)
    initial = session.initial_calibration_identity()
    # The production worker constructs this fixed-action probe from the same
    # frozen trajectories immediately before run_corrected_step.  The
    # differential enters one level lower so it must mirror that lifecycle
    # input explicitly; otherwise checkpoint export receives an empty probe.
    session.probe_rows = _fixed_probe_rows(rows)
    if len(session.optimizer.state) != 0 or session._optimizer_step_count != 0:
        _fail("differential optimizer did not start fresh")
    if session._scheduler_step_count != 0:
        _fail("differential scheduler did not start fresh")

    capture: dict[str, Any] = {}
    session._p4f_capture_differential = True
    score_calls = 0
    from src.opd.production_qualification_two_step_gpu_v7 import _step_result

    # Keep the bound implementation for this concrete runtime.  Calling the
    # common base method here would make the balanced side reuse the legacy
    # scorer and would only differential-test its backward implementation.
    original_score_rows = session._score_rows

    def score_rows(
        bound_self: Any,
        model: Any,
        score_rows_value: list[Mapping[str, Any]],
        *,
        device: str,
        inference: bool,
    ) -> tuple[Any, Any]:
        nonlocal score_calls
        score_calls += 1
        if mode == "memory_balanced" and score_calls in {1, 3}:
            cached = reused_old_actor if score_calls == 1 else reused_teacher
            if cached is None:
                _fail("memory-balanced path lacks frozen p_old/Teacher targets")
            phases = getattr(bound_self, "_memory_score_phases", None)
            expected_phase = "p_old" if score_calls == 1 else "teacher_same_token"
            if not isinstance(phases, list) or not phases:
                _fail("memory-balanced scorer phase registry is absent")
            if phases.pop(0) != expected_phase:
                _fail("memory-balanced frozen scorer phase order differs")
            # The memory-balanced session deliberately keeps frozen q/p_old/
            # Teacher matrices on CPU and transfers only one prompt slice at
            # backward time.  Preserve that device contract here.
            return cached[0].clone(), cached[1].clone()
        result = original_score_rows(
            model,
            score_rows_value,
            device=device,
            inference=inference,
        )
        if score_calls == 1:
            capture["old_actor"] = (
                result[0].detach().cpu().clone(),
                result[1].detach().cpu().clone(),
            )
        elif score_calls == 2:
            capture["q_pre"] = (
                result[0].detach().cpu().clone(),
                result[1].detach().cpu().clone(),
            )
        elif score_calls == 3:
            capture["teacher"] = (
                result[0].detach().cpu().clone(),
                result[1].detach().cpu().clone(),
            )
        elif mode == "legacy_prompt_equal" and 4 <= score_calls <= 7:
            length = int(result[1][0].sum().cpu())
            capture.setdefault("legacy_backward_q_rows", []).append(
                result[0][0, :length].detach().float().cpu().clone()
            )
        elif (
            mode == "memory_balanced" and score_calls == 4
        ) or (
            mode == "legacy_prompt_equal" and score_calls == 8
        ):
            capture["q_post"] = (
                result[0].detach().cpu().clone(),
                result[1].detach().cpu().clone(),
            )
        return result

    session._score_rows = MethodType(score_rows, session)

    original_objective = session.decoupled_corrected_objective

    def capture_objective(*args: Any, **kwargs: Any) -> Any:
        result = original_objective(*args, **kwargs)
        if mode == "legacy_prompt_equal" and capture.get(
            "inside_legacy_backward", False
        ):
            capture.setdefault("legacy_backward_scaled_losses", []).append(
                float(result.loss.detach().float().cpu()) / float(len(rows))
            )
        return result

    session.decoupled_corrected_objective = capture_objective

    original_backward = session._backward_corrected_rows

    def capture_backward(bound_self: Any, **kwargs: Any) -> Any:
        bundle = kwargs["bundle"]
        before_result = kwargs["before_result"]
        capture["frozen_bundle"] = {
            "rollout_behavior_logprob": (
                bundle.rollout_behavior_logprob.detach().float().cpu().clone()
            ),
            "p_old_target_logprob": (
                bundle.old_actor_logprob.detach().float().cpu().clone()
            ),
            "teacher_target_logprob": (
                bundle.teacher_logprob.detach().float().cpu().clone()
            ),
            "valid_mask": bundle.response_mask.detach().cpu().clone(),
            "correction_weight": (
                before_result.correction.truncated_weight.detach()
                .float().cpu().clone()
            ),
        }
        capture["pre_update_objective"] = float(
            before_result.surrogate.detach().float().cpu()
        )
        capture["pre_update_loss"] = float(
            before_result.loss.detach().float().cpu()
        )
        if mode == "legacy_prompt_equal":
            capture["inside_legacy_backward"] = True
        try:
            return original_backward(**kwargs)
        finally:
            capture["inside_legacy_backward"] = False

    session._backward_corrected_rows = MethodType(capture_backward, session)

    original_audit = session.audit_optimizer_update

    def capture_update(**kwargs: Any) -> Any:
        capture["gradients"] = {
            name: value.detach().cpu().clone()
            for name, value in kwargs["loss_gradients"].items()
        }
        capture["deltas"] = {
            name: (
                kwargs["after"][name].detach().cpu()
                - kwargs["before"][name].detach().cpu()
            ).clone()
            for name in kwargs["declared_trainable_names"]
        }
        return original_audit(**kwargs)

    session.audit_optimizer_update = capture_update
    counts = {"export": 0, "refresh": 0}
    original_export = session._checkpoint_authority

    def capture_export(bound_self: Any, step: int) -> Any:
        counts["export"] += 1
        return original_export(step)

    session._checkpoint_authority = MethodType(capture_export, session)
    original_refresh = session.hotswap_stable_slot

    def capture_refresh(bound_self: Any, **kwargs: Any) -> Any:
        counts["refresh"] += 1
        return original_refresh(**kwargs)

    session.hotswap_stable_slot = MethodType(capture_refresh, session)

    for device in (0, 1):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rollout = {
        "policy_version": "v0",
        "tensor_sha256": session.authorities[0]["aggregate_tensor_sha256"],
        "rows": rows,
        "provenance": dict(provenance),
    }
    update = session.run_corrected_step(0, rollout)
    _reconstruction, _artifact, target, checkpoint = _step_result(update, step=0)
    session.release_step_teacher(0)
    session.hotswap_stable_slot(
        current_authority=session.authorities[0],
        target_authority=target,
        checkpoint=checkpoint,
    )
    elapsed = time.perf_counter() - started
    private = session._last_b2_step_private
    if not isinstance(private, Mapping):
        _fail("differential numeric evidence is absent")
    q_pre_values, q_pre_mask = capture["q_pre"]
    q_post_values, q_post_mask = capture["q_post"]
    frozen_bundle = capture.get("frozen_bundle")
    if not isinstance(frozen_bundle, Mapping):
        _fail("differential frozen objective bundle is absent")
    if mode == "legacy_prompt_equal":
        q_rows = capture.get("legacy_backward_q_rows")
        scaled_losses = capture.get("legacy_backward_scaled_losses")
        if not (
            isinstance(q_rows, list)
            and len(q_rows) == 4
            and isinstance(scaled_losses, list)
            and len(scaled_losses) == 4
        ):
            _fail("legacy prompt-equal backward capture is incomplete")
        q_values, q_mask = session._pad(q_rows)
        q_values = q_values.detach().float().cpu()
        q_mask = q_mask.detach().cpu()
        native_backward_loss = float(sum(scaled_losses))
        native_objective = -native_backward_loss
        backbone_forward_count = 4
        backbone_backward_count = 4
        lm_head_count = 4
        chunk_size = max(len(row["response_ids"]) for row in rows)
        chunk_ranges = [
            [[0, len(row["response_ids"])]] for row in rows
        ]
    else:
        q_rows = getattr(session, "_memory_differential_q_rows", None)
        if not isinstance(q_rows, list) or len(q_rows) != 4:
            _fail("memory-balanced backward q capture is incomplete")
        q_values, q_mask = session._pad(q_rows)
        q_values = q_values.detach().float().cpu()
        q_mask = q_mask.detach().cpu()
        native_backward_loss = float(session._memory_chunk_loss_total)
        native_objective = -native_backward_loss
        backbone_forward_count = 4
        backbone_backward_count = int(session._memory_backbone_backward_count)
        lm_head_count = int(session._memory_lm_head_chunk_count)
        chunk_size = int(session.memory_contract["target_logit_chunk_size"])
        chunk_ranges = [
            [
                [start, min(start + chunk_size, len(row["response_ids"]))]
                for start in range(0, len(row["response_ids"]), chunk_size)
            ]
            for row in rows
        ]
    if q_values is None or q_mask is None or native_objective is None:
        _fail("differential backward q/native objective is absent")
    if not (
        torch.equal(q_mask, frozen_bundle["valid_mask"])
        and torch.equal(q_pre_mask, frozen_bundle["valid_mask"])
        and torch.equal(q_post_mask, frozen_bundle["valid_mask"])
    ):
        _fail("differential q capture masks differ")
    samples = _differential_sample_identity(rows)
    token_payload = {
        "prompt_token_ids": [
            [int(value) for value in row["prompt_ids"]] for row in rows
        ],
        "completion_token_ids": [
            [int(value) for value in row["response_ids"]] for row in rows
        ],
    }
    actual_chunk_count = sum(len(value) for value in chunk_ranges)
    result = {
        "initial_adapter_sha256": initial["adapter_sha256"],
        "fixed_tokens_sha256": fixed_tokens_sha256,
        # Retain the v1 flat field for historical comparator compatibility;
        # P4.8f uses the structured arrays below and distinguishes all three
        # q evaluation points.
        "q_target_logprob": q_values[q_mask],
        "objective": float(private["objective"]),
        "loss": float(private["loss"]),
        "backward_loss": float(native_backward_loss),
        "grad_norm": float(private["gradient_norm"]),
        "grad_norm_before_clip": float(private["gradient_norm_before_clip"]),
        "gradients": capture["gradients"],
        "deltas": capture["deltas"],
        "counts": {
            "optimizer": int(session._optimizer_step_count),
            "scheduler": int(session._scheduler_step_count),
            "export": counts["export"],
            "refresh": counts["refresh"],
            "version_increment": int(session.current_sampler_version),
        },
        "backbone_backward_calls_per_prompt": (
            1
            if mode == "legacy_prompt_equal"
            else float(session._memory_backbone_backward_count) / 4.0
        ),
        "lm_head_chunk_count": (
            4
            if mode == "legacy_prompt_equal"
            else session._memory_lm_head_chunk_count
        ),
        "retain_graph_calls": (
            0
            if mode == "legacy_prompt_equal"
            else session._memory_retain_graph_count
        ),
        "timings_seconds": {
            "total": elapsed,
            "scoring": float(private["scoring_seconds"]),
            "backward": float(private["backward_seconds"]),
        },
        "peak_memory_bytes": {
            "gpu0": int(torch.cuda.max_memory_allocated(0)),
            "gpu1": int(torch.cuda.max_memory_allocated(1)),
        },
        "input_identity": {
            "fixed_tokens_sha256": fixed_tokens_sha256,
            "initial_adapter_sha256": initial["adapter_sha256"],
            "prompt_order": [item["sample_id"] for item in samples],
            "samples": samples,
            "chunk_plan": chunk_ranges,
        },
        "token_payload": token_payload,
        "runtime": {
            "dtype": "bfloat16",
            "autocast_enabled": bool(torch.is_autocast_enabled()),
            "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "use_cache": False,
            "gradient_checkpointing": str(
                session.config.get("execution", {}).get(
                    "gradient_checkpointing", "enabled"
                )
            ),
            "batch_shape": list(q_values.shape),
            "chunk_size": chunk_size,
            "chunk_ranges": chunk_ranges,
            "planned_chunk_count": actual_chunk_count,
            "actual_chunk_count": actual_chunk_count,
            "accumulator_dtype": "float32",
            "backbone_forward_count": backbone_forward_count,
            "backbone_backward_count": backbone_backward_count,
            "lm_head_forward_count": lm_head_count,
            "lm_head_backward_count": lm_head_count,
            "retain_graph_count": int(
                0 if mode == "legacy_prompt_equal"
                else session._memory_retain_graph_count
            ),
            "zero_grad_count": 1,
            "grad_clip_count": 1,
            "optimizer_count": int(session._optimizer_step_count),
            "scheduler_count": int(session._scheduler_step_count),
            "export_count": counts["export"],
            "refresh_count": counts["refresh"],
            "policy_increment_count": int(session.current_sampler_version),
        },
        "arrays": {
            "q_target_logprob": q_values,
            "q_pre_target_logprob": q_pre_values,
            "q_post_target_logprob": q_post_values,
            "p_old_target_logprob": frozen_bundle["p_old_target_logprob"],
            "teacher_target_logprob": frozen_bundle[
                "teacher_target_logprob"
            ],
            "rollout_behavior_logprob": frozen_bundle[
                "rollout_behavior_logprob"
            ],
            "correction_weight": frozen_bundle["correction_weight"],
            "valid_mask": frozen_bundle["valid_mask"],
        },
        "native": {
            "objective": float(native_objective),
            "loss": -float(native_objective),
            "backward_loss": float(native_backward_loss),
            "pre_update_objective": float(capture["pre_update_objective"]),
            "pre_update_loss": float(capture["pre_update_loss"]),
            "post_update_objective": float(private["objective"]),
            "post_update_loss": float(private["loss"]),
            "grad_norm": float(private["gradient_norm"]),
            "grad_norm_before_clip": float(
                private["gradient_norm_before_clip"]
            ),
        },
        "gradients": capture["gradients"],
        "deltas": capture["deltas"],
        "teacher_gradient_tensor_count": int(
            private["teacher_gradient_tensor_count"]
        ),
        "base_gradient_tensor_count": int(
            private["base_gradient_tensor_count"]
        ),
    }
    old_actor = capture.get("old_actor") or reused_old_actor
    teacher = capture.get("teacher") or reused_teacher
    if old_actor is None or teacher is None:
        _fail("differential frozen scorer tensors are absent")
    return result, old_actor, teacher


def execute_real_gpu_math_differential_v1(
    *,
    runtime_config: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    evidence_dir: str | Path,
    legacy_session_factory: Any | None = None,
    balanced_session_factory: Any | None = None,
    _pre_execution_callback: Any | None = None,
    _comparison_callback: Any | None = None,
    _report_artifact_kind: str = "p4_8e_real_qwen3_4b_gpu_math_equivalence_v1",
) -> dict[str, Any]:
    """Run serialized real Qwen3-4B BF16 legacy/new updates on fixed tokens."""

    import gc
    from copy import deepcopy
    import torch

    output = Path(evidence_dir).resolve()
    if output.exists() or output.is_symlink():
        _fail("GPU differential evidence directory must be fresh")
    output.mkdir(parents=True, exist_ok=False)
    if _pre_execution_callback is not None:
        _pre_execution_callback(output)
    if len(prompt_rows) != 4:
        _fail("GPU differential requires frozen step1 four-prompt batch")
    if legacy_session_factory is None or balanced_session_factory is None:
        from src.opd.production_b2_calibration_backend_v2 import (
            create_production_b2_memory_session_v1,
        )
        from src.opd.production_qualification_two_step_gpu_v7 import (
            ProductionTwoStepSessionV6,
        )

        if legacy_session_factory is None:
            legacy_session_factory = lambda config, path: (
                create_production_b2_memory_session_v1(
                    config,
                    config_path=path,
                    _session_constructor=ProductionTwoStepSessionV6,
                )
            )
        if balanced_session_factory is None:
            balanced_session_factory = lambda config, path: (
                create_production_b2_memory_session_v1(config, config_path=path)
            )

    legacy_snapshot: dict[str, Any] | None = None
    balanced_snapshot: dict[str, Any] | None = None
    fixed_rows: list[dict[str, Any]] | None = None
    provenance: Mapping[str, Any] | None = None
    rng_state: Mapping[str, Any] | None = None
    with tempfile.TemporaryDirectory(
        prefix=".p4_8e_gpu_diff_", dir=output.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        legacy_root = temporary / "legacy"
        balanced_root = temporary / "balanced"
        legacy_root.mkdir()
        balanced_root.mkdir()
        differential_run_id = f"{runtime_config['run']['run_id']}-gpu-differential"

        legacy_config = deepcopy(dict(runtime_config))
        legacy_config["run"] = deepcopy(dict(runtime_config["run"]))
        legacy_config["run"].update(
            {"run_id": differential_run_id, "output_dir": str(legacy_root)}
        )
        legacy_path = legacy_root / "runtime_config.json"
        legacy_path.write_text(
            json.dumps(legacy_config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        _reset_peak_memory_stats_all(torch)
        legacy_session = None
        try:
            legacy_load_started = time.perf_counter()
            legacy_session = legacy_session_factory(legacy_config, legacy_path)
            legacy_load_seconds = time.perf_counter() - legacy_load_started
            legacy_load_peak = {
                "gpu0": int(torch.cuda.max_memory_allocated(0)),
                "gpu1": int(torch.cuda.max_memory_allocated(1)),
            }
            tokenized = legacy_session._source_prompt_rows(list(prompt_rows))
            authority = legacy_session.authorities[0]
            generation_started = time.perf_counter()
            with torch.inference_mode():
                fixed_rows = legacy_session._generate_rows(
                    legacy_session.sampler_model,
                    tokenized,
                    device="cuda:1",
                    step_index=0,
                )
            generation_seconds = time.perf_counter() - generation_started
            legacy_pre_update_peak = {
                "gpu0": int(torch.cuda.max_memory_allocated(0)),
                "gpu1": int(torch.cuda.max_memory_allocated(1)),
            }
            provenance = legacy_session._provenance(
                fixed_rows, authority=authority, step_index=0
            )
            fixed_sha = _fixed_tokens_sha256(fixed_rows)
            rng_state = _cpu_rng_snapshot(torch)
            legacy_snapshot, frozen_old, frozen_teacher = _run_fixed_path(
                session=legacy_session,
                rows=fixed_rows,
                provenance=provenance,
                fixed_tokens_sha256=fixed_sha,
                rng_state=rng_state,
                mode="legacy_prompt_equal",
            )
            legacy_snapshot["timings_seconds"].update(
                {
                    "model_load": legacy_load_seconds,
                    "fixed_rollout_generation": generation_seconds,
                }
            )
            for device in ("gpu0", "gpu1"):
                legacy_snapshot["peak_memory_bytes"][device] = max(
                    legacy_snapshot["peak_memory_bytes"][device],
                    legacy_load_peak[device],
                    legacy_pre_update_peak[device],
                )
        finally:
            try:
                if legacy_session is not None:
                    legacy_session.close()
            finally:
                legacy_session = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        balanced_config = deepcopy(dict(runtime_config))
        balanced_config["run"] = deepcopy(dict(runtime_config["run"]))
        balanced_config["run"].update(
            {"run_id": differential_run_id, "output_dir": str(balanced_root)}
        )
        balanced_path = balanced_root / "runtime_config.json"
        balanced_path.write_text(
            json.dumps(balanced_config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        _reset_peak_memory_stats_all(torch)
        balanced_session = None
        try:
            balanced_load_started = time.perf_counter()
            balanced_session = balanced_session_factory(
                balanced_config, balanced_path
            )
            balanced_load_seconds = time.perf_counter() - balanced_load_started
            balanced_load_peak = {
                "gpu0": int(torch.cuda.max_memory_allocated(0)),
                "gpu1": int(torch.cuda.max_memory_allocated(1)),
            }
            if fixed_rows is None or provenance is None or rng_state is None:
                _fail("legacy path did not produce reusable differential state")
            balanced_snapshot, _old, _teacher = _run_fixed_path(
                session=balanced_session,
                rows=fixed_rows,
                provenance=provenance,
                fixed_tokens_sha256=fixed_sha,
                rng_state=rng_state,
                mode="memory_balanced",
                reused_old_actor=frozen_old,
                reused_teacher=frozen_teacher,
            )
            balanced_snapshot["timings_seconds"].update(
                {
                    "model_load": balanced_load_seconds,
                    "fixed_rollout_generation": 0.0,
                }
            )
            for device in ("gpu0", "gpu1"):
                balanced_snapshot["peak_memory_bytes"][device] = max(
                    balanced_snapshot["peak_memory_bytes"][device],
                    balanced_load_peak[device],
                )
        finally:
            try:
                if balanced_session is not None:
                    balanced_session.close()
            finally:
                balanced_session = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if legacy_snapshot is None or balanced_snapshot is None or fixed_rows is None:
        _fail("real GPU differential did not complete both paths")
    # The pinned PyTorch/cuBLAS stack retains one 32 MiB workspace per active
    # handle (two handles on GPU0 in this differential) after every Python CUDA
    # tensor and model has been released.  These blocks count as allocated, so
    # process exit would hide a real in-process canary boundary.  Synchronize,
    # explicitly clear the pinned-version workspaces, then probe both allocators.
    gc.collect()
    for device in (0, 1):
        torch.cuda.synchronize(device)
    _clear_cublas_workspaces(torch)
    torch.cuda.empty_cache()
    for device in (0, 1):
        torch.cuda.synchronize(device)
    cleanup = {
        "memory_allocated_bytes": [
            int(torch.cuda.memory_allocated(device)) for device in (0, 1)
        ],
        "memory_reserved_bytes": [
            int(torch.cuda.memory_reserved(device)) for device in (0, 1)
        ],
        "free_bytes": [
            int(torch.cuda.mem_get_info(device)[0]) for device in (0, 1)
        ],
        "runtime_references_released": True,
    }
    cleanup_passed = (
        cleanup["memory_allocated_bytes"] == [0, 0]
        and cleanup["memory_reserved_bytes"] == [0, 0]
    )
    legacy_snapshot["memory_and_cleanup"] = dict(cleanup)
    balanced_snapshot["memory_and_cleanup"] = dict(cleanup)
    comparator = _comparison_callback or compare_differential_snapshots
    comparison = comparator(legacy_snapshot, balanced_snapshot)
    report = {
        **comparison,
        "artifact_kind": _report_artifact_kind,
        "model": "Qwen3-4B",
        "dtype": "bfloat16",
        "schedule_step": 1,
        "prompt_count": 4,
        "fixed_completion_token_count": sum(
            len(row["response_ids"]) for row in fixed_rows
        ),
        "fixed_completion_lengths": [
            len(row["response_ids"]) for row in fixed_rows
        ],
        "rollout_tokens_persisted": False,
        "throwaway_adapter_reused": False,
        "runtime_loading": "serial",
        "gpu_cleanup_after_gate": cleanup_passed,
        "gpu_cleanup": cleanup,
    }
    report_path = output / "gpu_math_equivalence.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    del legacy_snapshot, balanced_snapshot
    gc.collect()
    return report


def _prepare_differential_runtime_v2(
    runtime_config: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve package-only fields before either differential route loads."""

    projected = project_production_b2_memory_runtime_v1(runtime_config)
    if len(prompt_rows) != 4:
        _fail("GPU differential requires frozen step1 four-prompt batch")
    try:
        prompt_ids = tuple(str(row["sample_id"]) for row in prompt_rows)
    except KeyError as error:
        raise B2GpuMathDifferentialV1Error(
            "canonical differential sample_id is absent"
        ) from error
    if any(not value for value in prompt_ids) or len(set(prompt_ids)) != 4:
        _fail("canonical differential sample IDs are empty or non-unique")
    algorithm = projected.get("algorithm")
    if not isinstance(algorithm, Mapping):
        _fail("projected differential algorithm is absent")
    try:
        constants = tuple(
            float(algorithm[name]) for name in ("beta", "clip_low", "clip_high")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise B2GpuMathDifferentialV1Error(
            "projected differential algorithm constants are invalid"
        ) from error
    if not all(math.isfinite(value) for value in constants):
        _fail("projected differential algorithm constants are nonfinite")
    return {
        "projected_runtime": projected,
        "prompt_ids": prompt_ids,
        "group_ids": ("g0",) * len(prompt_ids),
        "algorithm": dict(algorithm),
    }


def execute_real_gpu_math_differential_v2(
    *,
    runtime_config: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    evidence_dir: str | Path,
    legacy_session_factory: Any | None = None,
    balanced_session_factory: Any | None = None,
) -> dict[str, Any]:
    """Run both throwaway routes, persist every comparison, then decide."""

    from src.opd.production_b2_differential_evidence_v2 import (
        DifferentialEvidenceWriterV2,
        compare_and_persist_differential_v2,
        persist_precomparison_failure_v2,
    )

    output = Path(evidence_dir).resolve()
    preparation: dict[str, Any] = {}

    def prepare_before_execution(runtime_output: Path) -> None:
        writer = DifferentialEvidenceWriterV2(
            runtime_output / "diagnostic",
            run_id="p4_8f_fixed_token_gpu_differential",
        )
        preparation["writer"] = writer
        preparation.update(_prepare_differential_runtime_v2(runtime_config, prompt_rows))

    def evidence_first_comparator(
        legacy: Mapping[str, Any], balanced: Mapping[str, Any]
    ) -> dict[str, Any]:
        # This call is deliberately the only assertion-producing comparison
        # for P4.8f.  It writes all reachable components and full diagnostic
        # payloads before applying the frozen two-layer decision.
        algorithm = preparation["algorithm"]
        return compare_and_persist_differential_v2(
            legacy=legacy,
            balanced=balanced,
            evidence_dir=output / "diagnostic",
            prompt_ids=preparation["prompt_ids"],
            group_ids=preparation["group_ids"],
            beta=float(algorithm["beta"]),
            clip_low=float(algorithm["clip_low"]),
            clip_high=float(algorithm["clip_high"]),
            writer=preparation["writer"],
        )

    try:
        return execute_real_gpu_math_differential_v1(
            runtime_config=runtime_config,
            prompt_rows=prompt_rows,
            evidence_dir=output,
            legacy_session_factory=legacy_session_factory,
            balanced_session_factory=balanced_session_factory,
            _pre_execution_callback=prepare_before_execution,
            _comparison_callback=evidence_first_comparator,
            _report_artifact_kind=(
                "p4_8f_real_qwen3_4b_gpu_math_equivalence_v2"
            ),
        )
    except BaseException as error:
        writer = preparation.get("writer")
        if (
            isinstance(writer, DifferentialEvidenceWriterV2)
            and writer.read_index().get("component_count") == 0
        ):
            try:
                persist_precomparison_failure_v2(writer, error=error)
            except BaseException:
                # A failing storage primitive cannot durably describe itself;
                # preserve the original execution failure and never emit ready.
                pass
        raise


__all__ = [
    "B2GpuMathDifferentialV1Error",
    "GRADIENT_ATOL",
    "GRADIENT_RTOL",
    "SCALAR_ATOL",
    "SCALAR_RTOL",
    "_fixed_probe_rows",
    "_prepare_differential_runtime_v2",
    "compare_differential_snapshots",
    "execute_real_gpu_math_differential_v1",
    "execute_real_gpu_math_differential_v2",
]
