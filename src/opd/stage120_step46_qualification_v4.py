"""P7 fixed-token step46 replay and actual-impact qualification contracts.

The raw ``backend_abs_log_p999=0.9`` threshold remains a diagnostic trigger.
This module never converts a diagnostic-only candidate bypass into a health
acceptance; it only compares the mathematical effect of the two denominators
on the exact same fixed batch before both candidates are rolled back.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.opd.production_b2_correction_gate_forensic_v2 import _comparison
from src.opd.production_b2_fixed_token_qualification_v2 import (
    _diagnostic_runtime,
    _source_index,
)
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint
from src.opd.production_b2_formal_data_v1 import resolve_formal_b2_schedule_batch
from src.opd.production_b2_formal_gpu_v2 import (
    DiagnosticCandidateRollbackV2,
    _canonical_sha256,
    build_ratio_pool_binding_v2,
)
from src.opd.production_b2_ratio_contract_v2 import compute_ratio_evidence_v2
from src.opd.production_b2_transaction_v2 import (
    ordered_trainable_sha256,
    state_tree_sha256,
)
from src.opd.production_b2_formal_worker_v1 import _atomic_json, _json
from src.opd.production_main_method_gpu_v3 import FormalMethodSessionV3
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
)


class Step46QualificationV4Error(RuntimeError):
    """The fixed-token replay or actual-impact evidence is invalid."""


ACTUAL_IMPACT_THRESHOLDS_V4 = {
    "objective_relative_l1_max": 0.05,
    "gradient_relative_l2_max": 0.05,
    "gradient_cosine_min": 0.995,
    "adam_delta_relative_l2_max": 0.05,
    "adam_delta_cosine_min": 0.999,
}


def step46_diagnostic_mode_for_step_v4(mode: str, step_index: int) -> str:
    """Keep all diagnostic bypasses outside the accepted step41--45 replay."""

    if mode not in {
        "raw_reject",
        "production_candidate",
        "canonical_candidate",
    }:
        raise Step46QualificationV4Error("step46 diagnostic mode differs")
    return mode if int(step_index) == 45 else "normal_replay"


def _vector_metrics(left: Tensor, right: Tensor) -> dict[str, float]:
    left = left.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    right = right.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if left.shape != right.shape or left.numel() == 0:
        raise Step46QualificationV4Error("actual-impact vector shape differs")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise Step46QualificationV4Error("actual-impact vector is not finite")
    difference = torch.linalg.vector_norm(left - right)
    denominator = torch.linalg.vector_norm(right).clamp_min(1.0e-30)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        cosine = 1.0 if bool(torch.equal(left, right)) else 0.0
    else:
        cosine = float(torch.dot(left, right) / (left_norm * right_norm))
    return {
        "element_count": int(left.numel()),
        "relative_l2": float(difference / denominator),
        "cosine": cosine,
        "production_l2": float(left_norm),
        "canonical_l2": float(right_norm),
    }


def compare_actual_impact_v4(
    *,
    production_objective: float,
    canonical_objective: float,
    production_gradient: Tensor,
    canonical_gradient: Tensor,
    production_delta: Tensor,
    canonical_delta: Tensor,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare raw-sampler and canonical-denominator candidates."""

    frozen = dict(ACTUAL_IMPACT_THRESHOLDS_V4)
    if thresholds is not None:
        if set(thresholds) != set(frozen):
            raise Step46QualificationV4Error("actual-impact threshold schema differs")
        frozen = {key: float(value) for key, value in thresholds.items()}
    production_objective = float(production_objective)
    canonical_objective = float(canonical_objective)
    if not math.isfinite(production_objective) or not math.isfinite(canonical_objective):
        raise Step46QualificationV4Error("actual-impact objective is not finite")
    objective_relative_l1 = abs(production_objective - canonical_objective) / max(
        abs(canonical_objective), 1.0e-30
    )
    gradient = _vector_metrics(production_gradient, canonical_gradient)
    delta = _vector_metrics(production_delta, canonical_delta)
    failures = []
    if objective_relative_l1 > frozen["objective_relative_l1_max"]:
        failures.append("objective_relative_l1")
    if gradient["relative_l2"] > frozen["gradient_relative_l2_max"]:
        failures.append("gradient_relative_l2")
    if gradient["cosine"] < frozen["gradient_cosine_min"]:
        failures.append("gradient_cosine")
    if delta["relative_l2"] > frozen["adam_delta_relative_l2_max"]:
        failures.append("adam_delta_relative_l2")
    if delta["cosine"] < frozen["adam_delta_cosine_min"]:
        failures.append("adam_delta_cosine")
    return {
        "schema_version": 4,
        "comparison": "raw_sampler_denominator_vs_canonical_old_policy_denominator",
        "production_objective": production_objective,
        "canonical_objective": canonical_objective,
        "objective_relative_l1": objective_relative_l1,
        "gradient": gradient,
        "adam_parameter_delta": delta,
        "thresholds": frozen,
        "hard_failures": failures,
        "passed": not failures,
    }


def validate_step46_replay_summary_v4(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the externally persisted three-repetition qualification."""

    repetitions = value.get("repetitions")
    if not (
        value.get("schema_version") == 4
        and value.get("artifact_kind") == "p7_step46_fixed_token_qualification_v4"
        and isinstance(repetitions, Sequence)
        and not isinstance(repetitions, (str, bytes))
        and len(repetitions) == 3
        and [row.get("mode") for row in repetitions]
        == ["raw_reject", "production_candidate", "canonical_candidate"]
    ):
        raise Step46QualificationV4Error("step46 qualification shape differs")
    expected_sha = repetitions[0].get("completion_token_sha256")
    expected_count = repetitions[0].get("completion_token_count")
    p999 = []
    for row in repetitions:
        if not (
            row.get("output_is_fresh") is True
            and row.get("restored_from_logical_version") == 40
            and row.get("replayed_optimizer_steps") == [41, 42, 43, 44, 45]
            and row.get("replay_all_passed") is True
            and row.get("attempted_optimizer_step") == 46
            and row.get("protected_state_unchanged") is True
            and row.get("route_state_unchanged") is True
            and row.get("final_access_count") == 0
        ):
            raise Step46QualificationV4Error("step46 repetition contract differs")
        if (
            row.get("completion_token_sha256") != expected_sha
            or row.get("completion_token_count") != expected_count
        ):
            raise Step46QualificationV4Error("step46 token identity differs")
        p999.append(float(row.get("raw_abs_log_p999", float("nan"))))
    if not all(math.isfinite(item) for item in p999) or max(p999) - min(p999) > 1.0e-6:
        raise Step46QualificationV4Error("step46 raw P99.9 was not stable")
    if not (
        isinstance(value.get("actual_impact"), Mapping)
        and value["actual_impact"].get("passed") is True
        and value.get("historical_artifacts_modified") is False
        and value.get("diagnostic_threshold_overridden") is False
        and value.get("final_access_count") == 0
    ):
        raise Step46QualificationV4Error("step46 qualification did not pass")
    return {
        "passed": True,
        "completion_token_sha256": expected_sha,
        "completion_token_count": expected_count,
        "raw_abs_log_p999_range": [min(p999), max(p999)],
        "final_access_count": 0,
    }


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_state(session: Any) -> dict[str, Any]:
    state = session._transaction_state_v2
    return {
        "lora_sha256": ordered_trainable_sha256(session.student_model),
        "optimizer_sha256": state_tree_sha256(session.optimizer.state_dict()),
        "scheduler_sha256": state_tree_sha256(session.scheduler.state_dict()),
        "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
        "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        "accepted_optimizer_steps": int(state.accepted_optimizer_steps),
        "data_cursor": int(state.data_cursor),
        "policy_version": int(state.policy_version),
        "sampler_version": int(state.sampler_version),
        "refresh_version": int(state.refresh_version),
        "optimizer_step_count": int(session._optimizer_step_count),
        "scheduler_step_count": int(session._scheduler_step_count),
        "registry_count": int(session._registry_count()),
    }


class Step46DiagnosticSessionV4(FormalMethodSessionV3):
    """Old IDT session instrumented for one fixed step46 diagnostic attempt."""

    def __init__(self, config: Mapping[str, Any], *, diagnostic_mode: str, **kwargs: Any) -> None:
        if diagnostic_mode not in {
            "raw_reject",
            "production_candidate",
            "canonical_candidate",
        }:
            raise Step46QualificationV4Error("step46 diagnostic mode differs")
        self.diagnostic_mode_v4 = diagnostic_mode
        self.fixed_token_detail_v4: dict[str, Any] | None = None
        self.candidate_objective_v4: float | None = None
        self.candidate_loss_v4: float | None = None
        self.candidate_gradient_v4: Tensor | None = None
        self.candidate_delta_v4: Tensor | None = None
        self.gate_protected_before_v4: dict[str, Any] | None = None
        self._step46_active_v4 = False
        super().__init__(config, **kwargs)

    def _fixed_token_detail(
        self,
        *,
        step_index: int,
        rows: list[Mapping[str, Any]],
        bundle: Any,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
    ) -> tuple[Any, Tensor, str, dict[str, Any]]:
        valid = bundle.response_mask.detach().cpu().bool()
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        binding, response_ids = build_ratio_pool_binding_v2(
            rows, valid, pad_token_id=int(pad)
        )
        authority = self.authorities.get(step_index)
        adapter_sha = None if not isinstance(authority, Mapping) else authority.get(
            "aggregate_tensor_sha256"
        )
        if not isinstance(adapter_sha, str) or len(adapter_sha) != 64:
            raise Step46QualificationV4Error("step46 trainer adapter identity differs")
        sampler = bundle.rollout_behavior_logprob.detach().float().cpu()
        canonical = bundle.old_actor_logprob.detach().float().cpu()
        token_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            count = int(valid[row_index].sum().item())
            response = [int(value) for value in row["response_ids"]]
            if len(response) != count:
                raise Step46QualificationV4Error("step46 response boundary differs")
            for position in range(count):
                sampler_value = float(sampler[row_index, position])
                canonical_value = float(canonical[row_index, position])
                token_rows.append(
                    {
                        "sample_id": str(prompt_ids[row_index]),
                        "source": str(source_roles[row_index]),
                        "teacher_route": str(self._current_teacher_routes[row_index]),
                        "response_position": position,
                        "token_id": response[position],
                        "sampler_logprob": sampler_value,
                        "canonical_trainer_logprob": canonical_value,
                        "canonical_minus_sampler": canonical_value - sampler_value,
                        "absolute_difference": abs(canonical_value - sampler_value),
                    }
                )
        generation = copy.deepcopy(dict(bundle.behavior_provenance["generation_config"]))
        difference = (canonical - sampler)[valid]
        detail = {
            "schema_version": 4,
            "artifact_kind": "p7_step46_per_token_backend_detail_v4",
            "attempted_optimizer_step": 46,
            "policy_version": step_index,
            "sampler_version": int(self.current_sampler_version),
            "refresh_version": int(self._transaction_state_v2.refresh_version),
            "prompt_ids": [str(value) for value in prompt_ids],
            "sources": [str(value) for value in source_roles],
            "teacher_routes": list(self._current_teacher_routes),
            "completion_token_sha256": binding.response_token_sha256,
            "completion_token_count": binding.valid_token_count,
            "pool_binding": binding.as_dict(),
            "sampler_policy": {
                "adapter_sha256": bundle.behavior_provenance["sampler_adapter_sha256"],
                "adapter_version": bundle.behavior_provenance["sampler_adapter_version"],
                "score_source": bundle.behavior_provenance["score_source"],
                "score_semantics": bundle.behavior_provenance["score_semantics"],
            },
            "trainer_policy": {
                "adapter_sha256": adapter_sha,
                "adapter_version": step_index,
                "score_source": "canonical_full_sequence_shifted_log_softmax",
                "use_cache": False,
            },
            "base_revision": self.base_revision,
            "medical_teacher": {
                "adapter_ordered_sha256": self.config["teacher"]["adapter_sha256"],
                "adapter_weight_sha256": self.config["teacher"]["adapter_weight_sha256"],
                "manifest_sha256": self.config["teacher"]["manifest_sha256"],
            },
            "generation": {
                "temperature": generation.get("temperature"),
                "top_p": generation.get("top_p"),
                "top_k": generation.get("top_k"),
                "do_sample": generation.get("do_sample"),
                "eos_token_id": generation.get("eos_token_id"),
                "pad_token_id": generation.get("pad_token_id"),
                "use_cache": generation.get("use_cache"),
            },
            "boundaries": [
                {
                    "sample_id": str(prompt_ids[index]),
                    "prompt_token_count": len(row["prompt_ids"]),
                    "completion_token_count": len(row["response_ids"]),
                    "eos_observed": bool(row["eos_observed"]),
                    "response_mask_count": int(valid[index].sum().item()),
                }
                for index, row in enumerate(rows)
            ],
            "backend_difference": {
                "signed_min": float(difference.min()),
                "signed_max": float(difference.max()),
                "abs_max": float(difference.abs().max()),
                "abs_p99": float(torch.quantile(difference.abs(), 0.99)),
                "abs_p999": float(torch.quantile(difference.abs(), 0.999)),
            },
            "token_differences": token_rows,
            "cache_state": {
                "sampler_generation_use_cache": generation.get("use_cache"),
                "trainer_scoring_use_cache": False,
                "persistent_prefix_cache": False,
                "prefix_cache_reuse_across_prompts": False,
            },
            "prompt_text_persisted": False,
            "answer_reasoning_label_access_count": 0,
            "controller_access_count": 0,
            "final_access_count": 0,
        }
        return binding, response_ids, adapter_sha, detail

    def _validate_pre_update_ratio_contract_v2(
        self,
        *,
        step_index: int,
        rows: list[Mapping[str, Any]],
        bundle: Any,
        before_result: Any,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
    ) -> None:
        active_mode = step46_diagnostic_mode_for_step_v4(
            self.diagnostic_mode_v4, step_index
        )
        self._step46_active_v4 = active_mode != "normal_replay"
        if not self._step46_active_v4:
            return super()._validate_pre_update_ratio_contract_v2(
                step_index=step_index,
                rows=rows,
                bundle=bundle,
                before_result=before_result,
                prompt_ids=prompt_ids,
                source_roles=source_roles,
            )
        binding, response_ids, adapter_sha, detail = self._fixed_token_detail(
            step_index=step_index,
            rows=rows,
            bundle=bundle,
            prompt_ids=prompt_ids,
            source_roles=source_roles,
        )
        self.fixed_token_detail_v4 = detail
        self.gate_protected_before_v4 = _protected_state(self)
        if self.diagnostic_mode_v4 == "raw_reject":
            return super()._validate_pre_update_ratio_contract_v2(
                step_index=step_index,
                rows=rows,
                bundle=bundle,
                before_result=before_result,
                prompt_ids=prompt_ids,
                source_roles=source_roles,
            )
        if self._pending_transaction_v2 is not None:
            raise Step46QualificationV4Error("previous transaction remains pending")
        identity_gap = float(
            (
                bundle.current_actor_logprob.detach()
                - bundle.old_actor_logprob.detach()
            )[bundle.response_mask.bool()]
            .abs()
            .max()
            .cpu()
        )
        if not (
            identity_gap <= float(self.ratio_thresholds_v2["ppo_abs_log_p999_max"])
            and step_index == self.current_sampler_version
        ):
            raise Step46QualificationV4Error("step46 canonical PPO identity differs")
        self._ratio_pre_context_v2 = {
            "binding": binding,
            "response_ids": response_ids,
            "adapter_sha256": adapter_sha,
            "prompt_ids": tuple(str(value) for value in prompt_ids),
            "source_roles": tuple(str(value) for value in source_roles),
            "fixed_batch_sha256": _canonical_sha256(
                {
                    "pool_binding_sha256": binding.pool_binding_sha256,
                    "policy_version": step_index,
                    "prompt_ids": list(prompt_ids),
                }
            ),
        }
        self._last_fixed_rollout_v2 = {
            "policy_version": f"v{step_index}",
            "tensor_sha256": adapter_sha,
            "rows": copy.deepcopy(rows),
            "provenance": copy.deepcopy(dict(bundle.behavior_provenance)),
        }
        zeros = torch.zeros_like(bundle.current_actor_logprob.detach())
        evidence = compute_ratio_evidence_v2(
            log_q_pre=bundle.current_actor_logprob,
            log_p_old_canonical=bundle.old_actor_logprob,
            log_mu_sampler=bundle.rollout_behavior_logprob,
            log_q_post=bundle.current_actor_logprob.detach(),
            valid_mask=bundle.response_mask.detach().bool(),
            prompt_ids=prompt_ids,
            source_roles=source_roles,
            token_ids=response_ids,
            advantage=zeros,
            loss_contribution=zeros,
            gradient_proxy=zeros,
            pool_binding=binding,
            policy_version=step_index,
            q_pre_adapter_sha256=adapter_sha,
            p_old_adapter_sha256=adapter_sha,
            sampler_version=self.current_sampler_version,
            refresh_version=self.current_sampler_version,
            backend_log_clip=math.log(2.0),
            post_shift_tail_abs_log_threshold=float(
                self.ratio_thresholds_v2["post_shift_tail_abs_log_threshold"]
            ),
        )
        self._preupdate_backend_evidence_v2 = evidence
        self._preupdate_backend_health_v2 = {
            "accepted": True,
            "diagnostic_candidate_bypass_only": True,
            "raw_p999_trigger_preserved": True,
            "formal_health_acceptance": False,
        }
        objective_bundle = bundle
        if self.diagnostic_mode_v4 == "canonical_candidate":
            objective_bundle = self.ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.old_actor_logprob.detach(),
                old_actor_logprob=bundle.old_actor_logprob,
                current_actor_logprob=bundle.current_actor_logprob,
                teacher_logprob=bundle.teacher_logprob,
                response_mask=bundle.response_mask,
                behavior_provenance=bundle.behavior_provenance,
            )
            before_result = self.decoupled_corrected_objective(
                objective_bundle,
                prompt_ids=prompt_ids,
                group_ids=("g0",) * len(prompt_ids),
                source_roles=source_roles,
                beta=float(self.algorithm["beta"]),
                clip_low=float(self.algorithm["clip_low"]),
                clip_high=float(self.algorithm["clip_high"]),
                rollout_is_threshold=2.0,
                advantage_scale=getattr(self, "_current_advantage_scale", None),
            )
        self.candidate_objective_v4 = float(before_result.surrogate.detach().cpu())
        self.candidate_loss_v4 = float(before_result.loss.detach().cpu())

    def _backward_corrected_rows(self, **kwargs: Any) -> None:
        if self._step46_active_v4 and self.diagnostic_mode_v4 == "canonical_candidate":
            bundle = kwargs["bundle"]
            kwargs["bundle"] = self.ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.old_actor_logprob.detach(),
                old_actor_logprob=bundle.old_actor_logprob,
                current_actor_logprob=bundle.current_actor_logprob,
                teacher_logprob=bundle.teacher_logprob,
                response_mask=bundle.response_mask,
                behavior_provenance=bundle.behavior_provenance,
            )
        return super()._backward_corrected_rows(**kwargs)

    def _prepare_candidate_transaction_v2(self, **kwargs: Any) -> None:
        if step46_diagnostic_mode_for_step_v4(
            self.diagnostic_mode_v4, int(kwargs["step_index"])
        ) == "normal_replay":
            return super()._prepare_candidate_transaction_v2(**kwargs)
        super()._prepare_candidate_transaction_v2(**kwargs)
        self.candidate_gradient_v4 = torch.cat(
            [
                (
                    self.parameters[name].grad.detach().float().cpu().reshape(-1)
                    if self.parameters[name].grad is not None
                    else torch.zeros_like(
                        self.parameters[name], dtype=torch.float32, device="cpu"
                    ).reshape(-1)
                )
                for name in self.trainable_names
            ]
        )

    def _validate_candidate_update_v2(self, **kwargs: Any) -> None:
        if step46_diagnostic_mode_for_step_v4(
            self.diagnostic_mode_v4, int(kwargs["step_index"])
        ) == "normal_replay":
            return super()._validate_candidate_update_v2(**kwargs)
        transaction = self._pending_transaction_v2
        if transaction is None:
            raise Step46QualificationV4Error("step46 candidate transaction is absent")
        snapshot = transaction._load()
        trainable = snapshot["trainable"]
        self.candidate_delta_v4 = torch.cat(
            [
                (
                    self.parameters[name].detach().float().cpu()
                    - trainable[name].detach().float().cpu()
                ).reshape(-1)
                for name in self.trainable_names
            ]
        )
        self._pending_ratio_evidence_v2 = copy.deepcopy(
            self._preupdate_backend_evidence_v2
        )
        transaction.mark_candidate_validated()
        self._abort_candidate_transaction_v2(
            reason="p7_step46_actual_impact_candidate_unconditional_rollback"
        )
        raise DiagnosticCandidateRollbackV2(
            "P7 step46 actual-impact candidate measured and rolled back"
        )


def _runtime_for_repeat(
    source_config: Mapping[str, Any], *, output: Path, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = _diagnostic_runtime(
        source_config,
        output=output,
        thresholds=thresholds,
        selected_learning_rate=1.0e-5,
    )
    runtime["formal_method_v3"] = copy.deepcopy(source_config["formal_method_v3"])
    runtime["teacher"] = copy.deepcopy(source_config["teacher"])
    return runtime


def _fresh_output(output: Path, runtime: Mapping[str, Any]) -> None:
    output.mkdir(parents=True)
    for name in (
        "b2_steps",
        "checkpoints",
        "formal_steps",
        "method_steps_v3",
        "memory_step_audits",
        "memory_telemetry/markers",
        "ratio_evidence_v2",
        "rejected_updates_v2",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "diagnostic_runtime.json", runtime)


def run_step46_fixed_token_qualification_v4(
    *,
    source_package: Path,
    source_output: Path,
    step40_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:  # pragma: no cover - real two-GPU qualification
    source_package = source_package.resolve()
    source_output = source_output.resolve()
    step40_checkpoint = step40_checkpoint.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise Step46QualificationV4Error("step46 qualification output is not fresh")
    index = _source_index(source_package)
    checkpoint = validate_formal_checkpoint(step40_checkpoint)
    if checkpoint.get("logical_version") != 40:
        raise Step46QualificationV4Error("step46 source checkpoint is not step40")
    source_config = _json(source_package / "formal_method_config.json")
    schedule = _json(source_package / "prompt_schedule.json")
    authority = _json(source_package / "data_authority.json")
    thresholds_path = Path("reports/p5_1_ratio_health_thresholds_v2.json").resolve()
    thresholds = _json(thresholds_path)
    immutable_paths = {
        "step40_checkpoint_manifest": step40_checkpoint / "checkpoint_manifest.json",
        "historical_step45": source_output / "formal_steps/step_045.json",
        "historical_step46_rejection": source_output / "rejected_updates_v2/attempt_001.json",
    }
    immutable_before = {key: _sha_file(path) for key, path in immutable_paths.items()}
    output.mkdir(parents=True)
    repetitions: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, Tensor | float]] = {}
    started = time.time()
    for repeat_index, mode in enumerate(
        ("raw_reject", "production_candidate", "canonical_candidate"), start=1
    ):
        repeat_output = output / f"repeat_{repeat_index}_{mode}"
        runtime = _runtime_for_repeat(
            source_config, output=repeat_output, thresholds=thresholds
        )
        _fresh_output(repeat_output, runtime)
        session: Step46DiagnosticSessionV4 | None = None
        try:
            session = Step46DiagnosticSessionV4(
                runtime,
                diagnostic_mode=mode,
                config_path=source_package / "formal_method_config.json",
                route="b2_calibration",
            )
            resume_rows = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=40
            )
            resume = session.restore_formal_checkpoint_v1(
                step40_checkpoint,
                package_content_sha256=str(index["package_content_sha256"]),
                config_sha256=str(index["config_sha256"]),
                manifest_sha256=str(index["manifest_sha256"]),
                schedule_sha256=str(index["schedule_semantic_sha256"]),
                resume_prompt_rows=resume_rows,
            )
            _atomic_json(repeat_output / "step40_resume_identity.json", resume)
            comparisons = []
            for step_index in range(40, 45):
                rows = resolve_formal_b2_schedule_batch(
                    authority, schedule, step_index=step_index
                )
                record = session.run_formal_method_step_v3(
                    step_index=step_index, prompt_rows=rows, max_new_tokens=1024
                )
                historical = _json(
                    source_output / "formal_steps" / f"step_{step_index + 1:03d}.json"
                )
                comparison = _comparison(record, historical)
                comparison["optimizer_step"] = step_index + 1
                comparisons.append(comparison)
                session.release_transient_step_artifacts_v1(step_index + 1)
                if not comparison["passed"]:
                    raise Step46QualificationV4Error(
                        f"step{step_index + 1} deterministic replay differs"
                    )
            route_before = session.route_state.state_dict()
            rows46 = resolve_formal_b2_schedule_batch(
                authority, schedule, step_index=45
            )
            caught: BaseException | None = None
            try:
                session.run_formal_method_step_v3(
                    step_index=45, prompt_rows=rows46, max_new_tokens=1024
                )
            except (ProductionTwoStepQualificationV6Error, DiagnosticCandidateRollbackV2) as error:
                caught = error
            if caught is None:
                raise Step46QualificationV4Error("step46 diagnostic attempt did not stop")
            if mode == "raw_reject" and isinstance(caught, DiagnosticCandidateRollbackV2):
                raise Step46QualificationV4Error("raw step46 did not use the legacy gate")
            if mode != "raw_reject" and not isinstance(caught, DiagnosticCandidateRollbackV2):
                raise Step46QualificationV4Error("step46 candidate did not reach rollback")
            legacy_route_state_changed = session.route_state.state_dict() != route_before
            session.route_state.load_state_dict(route_before)
            route_state_unchanged = session.route_state.state_dict() == route_before
            protected_after = _protected_state(session)
            if session.fixed_token_detail_v4 is None or session.gate_protected_before_v4 is None:
                raise Step46QualificationV4Error("step46 detailed evidence is absent")
            protected_unchanged = (
                protected_after == session.gate_protected_before_v4
            )
            detail_path = repeat_output / "step46_per_token_backend_detail_v4.json"
            _atomic_json(detail_path, session.fixed_token_detail_v4)
            backend = session._preupdate_backend_evidence_v2
            if backend is None:
                attempts = sorted((repeat_output / "rejected_updates_v2").glob("attempt_*.json"))
                if len(attempts) != 1:
                    raise Step46QualificationV4Error("step46 rejection evidence differs")
                backend = _json(attempts[0])["ratio_evidence"]
            raw_p999 = float(backend["backend_correction"]["raw_log"]["abs_p999"])
            repetition = {
                "mode": mode,
                "output_is_fresh": True,
                "restored_from_logical_version": 40,
                "replayed_optimizer_steps": [41, 42, 43, 44, 45],
                "replay_all_passed": all(item["passed"] for item in comparisons),
                "replay_comparisons": comparisons,
                "attempted_optimizer_step": 46,
                "exception_type": type(caught).__name__,
                "exception_message": str(caught),
                "completion_token_sha256": session.fixed_token_detail_v4[
                    "completion_token_sha256"
                ],
                "completion_token_count": session.fixed_token_detail_v4[
                    "completion_token_count"
                ],
                "raw_abs_log_p999": raw_p999,
                "raw_abs_log_p99": float(
                    backend["backend_correction"]["raw_log"]["abs_p99"]
                ),
                "raw_abs_log_max": float(
                    backend["backend_correction"]["raw_log"]["abs_max"]
                ),
                "protected_state_unchanged": protected_unchanged,
                "route_state_unchanged": route_state_unchanged,
                "legacy_route_state_required_explicit_rollback": legacy_route_state_changed,
                "per_token_detail_sha256": _sha_file(detail_path),
                "candidate_bypass_is_diagnostic_only": mode != "raw_reject",
                "formal_health_acceptance": False,
                "final_access_count": 0,
            }
            repetitions.append(repetition)
            _atomic_json(repeat_output / "repetition_summary_v4.json", repetition)
            if mode != "raw_reject":
                if not all(
                    value is not None
                    for value in (
                        session.candidate_objective_v4,
                        session.candidate_gradient_v4,
                        session.candidate_delta_v4,
                    )
                ):
                    raise Step46QualificationV4Error("step46 candidate tensors are absent")
                vectors[mode] = {
                    "objective": float(session.candidate_objective_v4),
                    "gradient": session.candidate_gradient_v4,
                    "delta": session.candidate_delta_v4,
                }
        finally:
            if session is not None:
                session.close()
            torch.cuda.empty_cache()
    actual_impact = compare_actual_impact_v4(
        production_objective=float(vectors["production_candidate"]["objective"]),
        canonical_objective=float(vectors["canonical_candidate"]["objective"]),
        production_gradient=vectors["production_candidate"]["gradient"],
        canonical_gradient=vectors["canonical_candidate"]["gradient"],
        production_delta=vectors["production_candidate"]["delta"],
        canonical_delta=vectors["canonical_candidate"]["delta"],
    )
    immutable_after = {key: _sha_file(path) for key, path in immutable_paths.items()}
    summary = {
        "schema_version": 4,
        "artifact_kind": "p7_step46_fixed_token_qualification_v4",
        "status": "qualified" if actual_impact["passed"] else "rejected",
        "source_package_content_sha256": index["package_content_sha256"],
        "step40_checkpoint_manifest_sha256": immutable_before[
            "step40_checkpoint_manifest"
        ],
        "thresholds_sha256": _sha_file(thresholds_path),
        "repetitions": repetitions,
        "actual_impact": actual_impact,
        "historical_immutable_sha256_before": immutable_before,
        "historical_immutable_sha256_after": immutable_after,
        "historical_artifacts_modified": immutable_before != immutable_after,
        "diagnostic_threshold_overridden": False,
        "candidate_bypass_designation": "actual_impact_measurement_only_never_health_acceptance",
        "elapsed_seconds": time.time() - started,
        "controller_access_count": 0,
        "final_access_count": 0,
    }
    validation = validate_step46_replay_summary_v4(summary)
    summary["validation"] = validation
    _atomic_json(output / "step46_fixed_token_qualification_v4.json", summary)
    return summary


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--step40-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_step46_fixed_token_qualification_v4(
        source_package=args.source_package,
        source_output=args.source_output,
        step40_checkpoint=args.step40_checkpoint,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ACTUAL_IMPACT_THRESHOLDS_V4",
    "Step46QualificationV4Error",
    "compare_actual_impact_v4",
    "run_step46_fixed_token_qualification_v4",
    "step46_diagnostic_mode_for_step_v4",
    "validate_step46_replay_summary_v4",
]


if __name__ == "__main__":
    raise SystemExit(_main())
