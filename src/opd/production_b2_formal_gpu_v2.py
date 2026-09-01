"""Transactional Ratio Health Protocol v2 session for Formal B2.

This module adds candidate validation hooks to the already GPU-qualified v1
kernel.  The underlying objective and three-policy scorer remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.opd.production_b2_formal_gpu_v1 import FormalB2SessionV1
from src.opd.production_b2_ratio_contract_v2 import (
    RatioPoolBindingV2,
    compute_ratio_evidence_v2,
)
from src.opd.production_b2_ratio_health_v2 import (
    RatioHealthV2Error,
    evaluate_preupdate_backend_health_v2,
    evaluate_ratio_health_v2,
)
from src.opd.production_b2_transaction_v2 import (
    OptimizerTransactionV2,
    TransactionStateV2,
    ordered_trainable_sha256,
    state_tree_sha256,
)
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
    _atomic_json,
)


class DiagnosticCandidateRollbackV2(RuntimeError):
    """A GPU qualification candidate was measured and intentionally rolled back."""


def evaluate_candidate_acceptance_v2_1(
    legacy_evidence: Mapping[str, Any], *, accepted_optimizer_steps: int
) -> dict[str, Any]:
    """Separate safety invariants from minibatch-monotonic Adam diagnostics.

    Adam's accumulated moments need not improve the current stochastic batch at
    every step. Direction is still a hard check for the fresh optimizer, while
    ownership/freeze invariants are hard for every candidate.
    """

    if int(accepted_optimizer_steps) < 0:
        raise ProductionTwoStepQualificationV6Error(
            "candidate acceptance optimizer step is invalid"
        )
    directional = (
        "objective_improved",
        "loss_decreased",
        "alignment_positive",
    )
    unconditional = (
        "optimizer_update_audit_passed",
        "teacher_gradient_free",
        "base_gradient_free",
        "frozen_parameter_versions_unchanged",
    )
    for key in (*directional, *unconditional):
        if legacy_evidence.get(key) not in (True, False):
            raise ProductionTwoStepQualificationV6Error(
                f"candidate acceptance component is absent: {key}"
            )
    hard_fields = (*unconditional, *directional) if accepted_optimizer_steps == 0 else unconditional
    hard_failures = sorted(key for key in hard_fields if legacy_evidence[key] is False)
    diagnostic_warnings = sorted(
        key
        for key in directional
        if accepted_optimizer_steps > 0 and legacy_evidence[key] is False
    )
    return {
        "schema_version": 1,
        "protocol_id": "p5_1_candidate_acceptance_v2_1",
        "accepted_optimizer_steps_before_candidate": int(accepted_optimizer_steps),
        "fresh_optimizer_direction_hard_gate": accepted_optimizer_steps == 0,
        "accumulated_adam_same_batch_monotonicity": "diagnostic_only",
        "hard_failures": hard_failures,
        "diagnostic_warnings": diagnostic_warnings,
        "passed": not hard_failures,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_correction_gate_evidence_v2(
    correction: Mapping[str, Any],
    *,
    prompt_ids: Sequence[str],
    source_roles: Sequence[str],
    pool_binding_sha256: str,
    legacy_ess_fraction_min: float,
    legacy_cap_fraction_max: float,
    ratio_v2_per_prompt_ess_floor: float,
    ratio_v2_per_prompt_ess_min_tokens: int,
    ratio_v2_backend_clip_fraction_max: float,
) -> dict[str, Any]:
    """Describe a pre-update correction rejection without storing sample text/IDs."""

    per_prompt = correction.get("per_prompt")
    per_source = correction.get("per_source")
    if not (
        isinstance(per_prompt, Mapping)
        and isinstance(per_source, Mapping)
        and len(prompt_ids) == len(source_roles)
        and isinstance(pool_binding_sha256, str)
        and len(pool_binding_sha256) == 64
    ):
        raise ProductionTwoStepQualificationV6Error(
            "correction gate evidence identity differs"
        )

    def metrics(value: Mapping[str, Any]) -> dict[str, Any]:
        count = value.get("valid_token_count", value.get("token_count"))
        if count is None:
            raise ProductionTwoStepQualificationV6Error(
                "correction gate evidence token count is absent"
            )
        result = {
            "token_count": int(count),
            "ess_fraction": float(value["ess_fraction"]),
            "cap_fraction": float(value["cap_fraction"]),
        }
        if not (
            result["token_count"] > 0
            and all(math.isfinite(result[key]) for key in ("ess_fraction", "cap_fraction"))
        ):
            raise ProductionTwoStepQualificationV6Error(
                "correction gate evidence metric differs"
            )
        return result

    prompt_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    ratio_v2_prompt_ess_passed = True
    for prompt_id, source in zip(prompt_ids, source_roles, strict=True):
        value = per_prompt.get(prompt_id)
        if not isinstance(value, Mapping):
            raise ProductionTwoStepQualificationV6Error(
                "correction gate per-prompt evidence is absent"
            )
        row = {
            "sample_id_sha256": hashlib.sha256(
                str(prompt_id).encode("utf-8")
            ).hexdigest(),
            "source": str(source),
            **metrics(value),
        }
        row["legacy_passed"] = bool(
            row["ess_fraction"] >= float(legacy_ess_fraction_min)
            and row["cap_fraction"] <= float(legacy_cap_fraction_max)
        )
        row["ratio_v2_ess_applicable"] = bool(
            row["token_count"] >= int(ratio_v2_per_prompt_ess_min_tokens)
        )
        row["ratio_v2_ess_passed"] = bool(
            not row["ratio_v2_ess_applicable"]
            or row["ess_fraction"] >= float(ratio_v2_per_prompt_ess_floor)
        )
        ratio_v2_prompt_ess_passed &= row["ratio_v2_ess_passed"]
        prompt_rows.append(row)
        if not row["legacy_passed"]:
            failed.append({"partition": "prompt", **row})

    source_rows: list[dict[str, Any]] = []
    for source, value in per_source.items():
        if not isinstance(value, Mapping):
            raise ProductionTwoStepQualificationV6Error(
                "correction gate per-source evidence differs"
            )
        row = {"source": str(source), **metrics(value)}
        row["legacy_passed"] = bool(
            row["ess_fraction"] >= float(legacy_ess_fraction_min)
            and row["cap_fraction"] <= float(legacy_cap_fraction_max)
        )
        source_rows.append(row)
        if not row["legacy_passed"]:
            failed.append({"partition": "source", **row})

    pooled = {
        "ess_fraction": float(correction["ess_fraction"]),
        "cap_fraction": float(correction["cap_fraction"]),
    }
    pooled["legacy_passed"] = bool(
        pooled["ess_fraction"] >= float(legacy_ess_fraction_min)
        and pooled["cap_fraction"] <= float(legacy_cap_fraction_max)
    )
    if not pooled["legacy_passed"]:
        failed.insert(0, {"partition": "pooled", **pooled})
    ratio_v2_preupdate_passed = bool(
        pooled["ess_fraction"] >= 0.95
        and pooled["cap_fraction"] <= float(ratio_v2_backend_clip_fraction_max)
        and ratio_v2_prompt_ess_passed
    )
    return {
        "schema_version": 2,
        "artifact_kind": "legacy_backend_correction_gate_evidence_v2",
        "pool_binding_sha256": pool_binding_sha256,
        "legacy_thresholds": {
            "ess_fraction_min": float(legacy_ess_fraction_min),
            "cap_fraction_max": float(legacy_cap_fraction_max),
            "aggregation": "pooled_and_all_prompt_and_all_source",
        },
        "ratio_v2_thresholds": {
            "pooled_ess_floor": 0.95,
            "per_prompt_ess_floor": float(ratio_v2_per_prompt_ess_floor),
            "per_prompt_ess_min_tokens": int(ratio_v2_per_prompt_ess_min_tokens),
            "backend_clip_fraction_max": float(ratio_v2_backend_clip_fraction_max),
            "backend_clip_aggregation": "pooled",
        },
        "pooled": pooled,
        "per_prompt": prompt_rows,
        "per_source": source_rows,
        "failed_partitions": failed,
        "legacy_gate_passed": bool(pooled["legacy_passed"] and not failed),
        "ratio_v2_preupdate_backend_gate_passed": ratio_v2_preupdate_passed,
        "raw_prompt_persisted": False,
        "label_access_count": 0,
    }


def build_ratio_pool_binding_v2(
    rows: Sequence[Mapping[str, Any]],
    valid_mask: torch.Tensor,
    *,
    pad_token_id: int,
) -> tuple[RatioPoolBindingV2, torch.Tensor]:
    """Bind the complete input token stream and exact scored response pool."""

    mask = valid_mask.detach().cpu().bool()
    if len(rows) != mask.shape[0] or mask.ndim != 2:
        raise ProductionTwoStepQualificationV6Error("ratio pool row/mask shape differs")
    max_input = max(len(row["prompt_ids"]) + len(row["response_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_input), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_ids = torch.full(mask.shape, int(pad_token_id), dtype=torch.long)
    for index, row in enumerate(rows):
        prompt = [int(value) for value in row["prompt_ids"]]
        response = [int(value) for value in row["response_ids"]]
        if len(response) != int(mask[index].sum().item()):
            raise ProductionTwoStepQualificationV6Error(
                "ratio response token count differs from canonical mask"
            )
        combined = prompt + response
        input_ids[index, : len(combined)] = torch.tensor(combined, dtype=torch.long)
        attention[index, : len(combined)] = True
        response_mask[index, len(prompt) : len(combined)] = True
        response_ids[index, : len(response)] = torch.tensor(response, dtype=torch.long)
    binding = RatioPoolBindingV2.from_tensors(
        input_ids=input_ids,
        response_ids=response_ids,
        attention_mask=attention,
        response_mask=response_mask,
        valid_mask=mask,
    )
    return binding, response_ids


def build_precommit_gradient_evidence_v2(
    *,
    valid_mask: torch.Tensor,
    token_ids: torch.Tensor,
    advantage: torch.Tensor,
    gradient_proxy: torch.Tensor,
    loss_contribution: torch.Tensor,
    prompt_ids: Sequence[str],
    source_roles: Sequence[str],
    pool_binding_sha256: str,
    gradient_norm_before_clip: float,
    robust_z: float,
) -> dict[str, Any]:
    """Summarize a rejected pre-optimizer batch without persisting text."""

    mask = valid_mask.detach().cpu().bool()
    values = (token_ids, advantage, gradient_proxy, loss_contribution)
    if not (
        mask.ndim == 2
        and all(value.shape == mask.shape for value in values)
        and len(prompt_ids) == len(source_roles) == mask.shape[0]
        and isinstance(pool_binding_sha256, str)
        and len(pool_binding_sha256) == 64
        and math.isfinite(float(gradient_norm_before_clip))
        and math.isfinite(float(robust_z))
    ):
        raise ProductionTwoStepQualificationV6Error(
            "precommit gradient evidence identity differs"
        )
    flat_gradient = gradient_proxy.detach().float().cpu()[mask].abs()
    flat_loss = loss_contribution.detach().float().cpu()[mask].abs()
    flat_advantage = advantage.detach().float().cpu()[mask]
    flat_tokens = token_ids.detach().long().cpu()[mask]
    if flat_gradient.numel() == 0 or not all(
        torch.isfinite(value).all() for value in (flat_gradient, flat_loss, flat_advantage)
    ):
        raise ProductionTwoStepQualificationV6Error(
            "precommit gradient evidence is empty or non-finite"
        )
    gradient_total = float(flat_gradient.sum().item())
    loss_total = float(flat_loss.sum().item())
    row_indices, positions = torch.where(mask)
    ranked = torch.argsort(flat_gradient, descending=True)[:20]
    top_tokens: list[dict[str, Any]] = []
    for flat_index in ranked.tolist():
        row_index = int(row_indices[flat_index].item())
        gradient_value = float(flat_gradient[flat_index].item())
        loss_value = float(flat_loss[flat_index].item())
        advantage_value = float(flat_advantage[flat_index].item())
        top_tokens.append(
            {
                "sample_id_sha256": hashlib.sha256(
                    str(prompt_ids[row_index]).encode("utf-8")
                ).hexdigest(),
                "source": str(source_roles[row_index]),
                "response_position": int(positions[flat_index].item()),
                "token_id": int(flat_tokens[flat_index].item()),
                "advantage": advantage_value,
                "advantage_negative": advantage_value < 0.0,
                "gradient_proxy_abs": gradient_value,
                "gradient_proxy_share": (
                    0.0 if gradient_total == 0.0 else gradient_value / gradient_total
                ),
                "loss_contribution_abs": loss_value,
                "loss_contribution_share": (
                    0.0 if loss_total == 0.0 else loss_value / loss_total
                ),
            }
        )
    top_gradient = sum(float(row["gradient_proxy_abs"]) for row in top_tokens)
    return {
        "schema_version": 2,
        "artifact_kind": "precommit_gradient_evidence_v2",
        "pool_binding_sha256": pool_binding_sha256,
        "valid_token_count": int(mask.sum().item()),
        "gradient_norm_before_clip": float(gradient_norm_before_clip),
        "gradient_norm_robust_z": float(robust_z),
        "gradient_proxy_total_abs": gradient_total,
        "gradient_proxy_top1_share": (
            0.0 if gradient_total == 0.0 else float(flat_gradient.max().item()) / gradient_total
        ),
        "gradient_proxy_top20_share": (
            0.0 if gradient_total == 0.0 else top_gradient / gradient_total
        ),
        "loss_contribution_total_abs": loss_total,
        "top_tokens": top_tokens,
        "raw_prompt_persisted": False,
        "label_access_count": 0,
    }


def clip_prompt_gradient_contribution_v2(
    gradients: Mapping[str, torch.Tensor], *, max_norm: float
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Clip one already batch-scaled prompt gradient before accumulation."""

    if not gradients or not math.isfinite(float(max_norm)) or float(max_norm) <= 0.0:
        raise ProductionTwoStepQualificationV6Error(
            "bounded prompt gradient input differs"
        )
    squared = sum(
        float(value.detach().float().square().sum().cpu())
        for value in gradients.values()
    )
    raw_norm = math.sqrt(squared)
    if not math.isfinite(raw_norm):
        raise ProductionTwoStepQualificationV6Error(
            "bounded prompt gradient norm is non-finite"
        )
    scale = 1.0 if raw_norm <= float(max_norm) else float(max_norm) / raw_norm
    clipped = {
        name: value.detach().clone().mul_(scale) for name, value in gradients.items()
    }
    bounded_norm = math.sqrt(
        sum(
            float(value.detach().float().square().sum().cpu())
            for value in clipped.values()
        )
    )
    return clipped, {
        "raw_norm": raw_norm,
        "clip_scale": scale,
        "bounded_norm": bounded_norm,
        "max_norm": float(max_norm),
    }


class FormalB2SessionV2(FormalB2SessionV1):
    """Formal session where only a validated, refreshed candidate can commit."""

    _transactional_commit_v2_enabled = True

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        ratio_config = config.get("ratio_health_v2")
        if not isinstance(ratio_config, Mapping):
            raise ProductionTwoStepQualificationV6Error(
                "Formal B2 v2 ratio health configuration is absent"
            )
        thresholds = ratio_config.get("thresholds")
        if not isinstance(thresholds, Mapping) or thresholds.get("schema_version") != 2:
            raise ProductionTwoStepQualificationV6Error(
                "Formal B2 v2 ratio thresholds are absent"
            )
        self.ratio_thresholds_v2 = copy.deepcopy(dict(thresholds))
        self._pending_transaction_v2: OptimizerTransactionV2 | None = None
        self._transaction_state_v2: TransactionStateV2 | None = None
        self._ratio_pre_context_v2: dict[str, Any] | None = None
        self._pending_ratio_evidence_v2: dict[str, Any] | None = None
        self._last_ratio_evidence_v2: dict[str, Any] | None = None
        self._last_ratio_health_v2: dict[str, Any] | None = None
        self._preupdate_backend_evidence_v2: dict[str, Any] | None = None
        self._preupdate_backend_health_v2: dict[str, Any] | None = None
        self._consecutive_ratio_warning_count_v2 = 0
        self._pending_warning_count_v2 = 0
        self._diagnostic_unconditional_rollback_v2 = False
        self._last_fixed_rollout_v2: dict[str, Any] | None = None
        candidate_protocol = config.get("candidate_acceptance_v2_1")
        self._candidate_acceptance_protocol_v2_1 = (
            copy.deepcopy(dict(candidate_protocol))
            if isinstance(candidate_protocol, Mapping)
            else None
        )
        if self._candidate_acceptance_protocol_v2_1 is not None:
            value = self._candidate_acceptance_protocol_v2_1
            if not (
                value.get("schema_version") == 1
                and value.get("protocol_id")
                == "p5_1_candidate_acceptance_v2_1"
                and value.get("fresh_optimizer_direction_hard_gate") is True
                and value.get("accumulated_adam_same_batch_monotonicity")
                == "diagnostic_only"
                and set(value.get("common_methods", []))
                == {"B2", "IDT", "CA-OPD"}
            ):
                raise ProductionTwoStepQualificationV6Error(
                    "candidate acceptance v2.1 configuration differs"
                )
        bounded = config.get("bounded_influence_v2")
        self._bounded_influence_config_v2 = (
            copy.deepcopy(dict(bounded)) if isinstance(bounded, Mapping) else None
        )
        if self._bounded_influence_config_v2 is not None and not (
            self._bounded_influence_config_v2.get("enabled") is True
            and self._bounded_influence_config_v2.get("mode")
            == "per_prompt_gradient_clipping"
            and float(
                self._bounded_influence_config_v2.get(
                    "per_prompt_gradient_clip_norm", -1.0
                )
            )
            == 0.25
            and float(
                self._bounded_influence_config_v2.get(
                    "global_gradient_clip_norm", -1.0
                )
            )
            == 1.0
            and int(self._bounded_influence_config_v2.get("effective_batch_size", -1))
            == 4
        ):
            raise ProductionTwoStepQualificationV6Error(
                "bounded influence v2 configuration differs"
            )
        self._bounded_prompt_accumulator_v2: dict[str, torch.Tensor] | None = None
        self._bounded_prompt_rows_v2: list[dict[str, Any]] = []
        self._last_bounded_influence_v2: dict[str, Any] | None = None
        run_id = str(config.get("run", {}).get("run_id", "formal-b2-v2"))
        self._transaction_scratch_v2 = Path("/dev/shm") / (
            "ca_opd_" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        )
        super().__init__(config, **kwargs)
        self._sync_transaction_state_v2()
        (self.output / "ratio_evidence_v2").mkdir(parents=True, exist_ok=True)
        (self.output / "rejected_updates_v2").mkdir(parents=True, exist_ok=True)

    def _sync_transaction_state_v2(self) -> None:
        step = int(self.current_sampler_version)
        self._transaction_state_v2 = TransactionStateV2(
            accepted_optimizer_steps=step,
            data_cursor=step * 4,
            policy_version=step,
            sampler_version=step,
            refresh_version=step,
            registry_count=self._registry_count(),
        )

    def restore_formal_checkpoint_v1(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = super().restore_formal_checkpoint_v1(*args, **kwargs)
        self._sync_transaction_state_v2()
        return result

    def set_diagnostic_unconditional_rollback_v2(self, enabled: bool) -> None:
        self._diagnostic_unconditional_rollback_v2 = bool(enabled)

    def enable_bounded_influence_v2(self, config: Mapping[str, Any]) -> None:
        """Enable a versioned repair only at an explicit diagnostic boundary."""

        value = dict(config)
        if not (
            value.get("enabled") is True
            and value.get("mode") == "per_prompt_gradient_clipping"
            and float(value.get("per_prompt_gradient_clip_norm", -1.0)) == 0.25
            and float(value.get("global_gradient_clip_norm", -1.0)) == 1.0
            and int(value.get("effective_batch_size", -1)) == 4
        ):
            raise ProductionTwoStepQualificationV6Error(
                "diagnostic bounded influence configuration differs"
            )
        if self._pending_transaction_v2 is not None:
            raise ProductionTwoStepQualificationV6Error(
                "cannot enable bounded influence during a transaction"
            )
        self._bounded_influence_config_v2 = copy.deepcopy(value)

    def _bound_prompt_gradient_contribution_v2(
        self,
        *,
        row_index: int,
        prompt_id: str,
        source_role: str,
        prompt_count: int,
    ) -> None:
        config = self._bounded_influence_config_v2
        if config is None:
            return
        if prompt_count != 4 or row_index not in range(4):
            raise ProductionTwoStepQualificationV6Error(
                "bounded influence prompt boundary differs"
            )
        gradients = {
            name: self.parameters[name].grad
            for name in self.trainable_names
            if self.parameters[name].grad is not None
        }
        if len(gradients) != len(self.trainable_names):
            raise ProductionTwoStepQualificationV6Error(
                "bounded influence gradient tensor count differs"
            )
        clipped, evidence = clip_prompt_gradient_contribution_v2(
            gradients,
            max_norm=float(config["per_prompt_gradient_clip_norm"]),
        )
        if row_index == 0:
            if self._bounded_prompt_accumulator_v2 is not None:
                raise ProductionTwoStepQualificationV6Error(
                    "bounded influence accumulator was not released"
                )
            self._bounded_prompt_accumulator_v2 = {
                name: value.clone() for name, value in clipped.items()
            }
            self._bounded_prompt_rows_v2 = []
        else:
            if self._bounded_prompt_accumulator_v2 is None:
                raise ProductionTwoStepQualificationV6Error(
                    "bounded influence accumulator is absent"
                )
            for name, value in clipped.items():
                self._bounded_prompt_accumulator_v2[name].add_(value)
        self._bounded_prompt_rows_v2.append(
            {
                "row_index": row_index,
                "sample_id_sha256": hashlib.sha256(
                    str(prompt_id).encode("utf-8")
                ).hexdigest(),
                "source": str(source_role),
                **evidence,
            }
        )
        for name in self.trainable_names:
            self.parameters[name].grad = None
        if row_index == prompt_count - 1:
            accumulator = self._bounded_prompt_accumulator_v2
            if accumulator is None:
                raise ProductionTwoStepQualificationV6Error(
                    "bounded influence final accumulator is absent"
                )
            for name in self.trainable_names:
                self.parameters[name].grad = accumulator[name]
            accumulated_norm = math.sqrt(
                sum(
                    float(value.detach().float().square().sum().cpu())
                    for value in accumulator.values()
                )
            )
            if not math.isfinite(accumulated_norm) or accumulated_norm > 1.000001:
                raise ProductionTwoStepQualificationV6Error(
                    "bounded influence aggregate trust budget exceeded"
                )
            self._last_bounded_influence_v2 = {
                "schema_version": 2,
                "artifact_kind": "per_prompt_gradient_clipping_v2",
                "mode": config["mode"],
                "per_prompt_gradient_clip_norm": float(
                    config["per_prompt_gradient_clip_norm"]
                ),
                "global_gradient_clip_norm": float(
                    config["global_gradient_clip_norm"]
                ),
                "effective_batch_size": 4,
                "prompt_gradients": copy.deepcopy(self._bounded_prompt_rows_v2),
                "accumulated_pre_global_clip_norm": accumulated_norm,
                "prompt_equal_scalar_loss_unchanged": True,
                "raw_prompt_persisted": False,
            }
            self._bounded_prompt_accumulator_v2 = None
            self._bounded_prompt_rows_v2 = []

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
        del before_result
        if self._pending_transaction_v2 is not None:
            raise ProductionTwoStepQualificationV6Error(
                "previous optimizer transaction remains pending"
            )
        valid = bundle.response_mask.detach().cpu().bool()
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        binding, response_ids = build_ratio_pool_binding_v2(
            rows, valid, pad_token_id=int(pad)
        )
        identity_gap = float(
            (
                bundle.current_actor_logprob.detach()
                - bundle.old_actor_logprob.detach()
            )[bundle.response_mask.bool()]
            .abs()
            .max()
            .cpu()
        )
        authority = self.authorities.get(step_index)
        adapter_sha = None if not isinstance(authority, Mapping) else authority.get(
            "aggregate_tensor_sha256"
        )
        if not (
            identity_gap <= float(self.ratio_thresholds_v2["ppo_abs_log_p999_max"])
            and isinstance(adapter_sha, str)
            and len(adapter_sha) == 64
            and step_index == self.current_sampler_version
        ):
            raise ProductionTwoStepQualificationV6Error(
                "pre-update same-version canonical q/p_old identity failed"
            )
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
        try:
            self._preupdate_backend_health_v2 = evaluate_preupdate_backend_health_v2(
                evidence, thresholds=self.ratio_thresholds_v2
            )
        except RatioHealthV2Error as error:
            self._record_preupdate_backend_rejection_v2(
                evidence=evidence, reason=str(error)
            )
            raise ProductionTwoStepQualificationV6Error(str(error)) from error

    def _preupdate_correction_gate_v2_passed(self) -> bool:
        health = self._preupdate_backend_health_v2
        return bool(isinstance(health, Mapping) and health.get("accepted") is True)

    def _record_preupdate_backend_rejection_v2(
        self, *, evidence: Mapping[str, Any], reason: str
    ) -> None:
        state = self._transaction_state_v2
        if not (
            state is not None
            and self._ratio_pre_context_v2 is not None
            and self._pending_transaction_v2 is None
        ):
            raise ProductionTwoStepQualificationV6Error(
                "preupdate backend rejection state differs"
            )
        before = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        after = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        unchanged = before == after
        artifact = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_rejected_update_v2",
            "run_id": self.config["run"]["run_id"],
            "attempted_optimizer_step": state.accepted_optimizer_steps + 1,
            "accepted_optimizer_steps": state.accepted_optimizer_steps,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "reason": "preupdate_backend_health_v2_rejected:" + reason,
            "candidate_lora_sha256": before["lora_sha256"],
            "rollback": {
                "rollback_verified": unchanged,
                "candidate_executed": False,
                "optimizer_executed": False,
                "scheduler_executed": False,
                "rng_restore_required": False,
                "state_before": before,
                "state_after": after,
            },
            "ratio_evidence": copy.deepcopy(dict(evidence)),
            "counts_as_optimizer_commit": False,
            "cursor_advanced": False,
            "sampler_refreshed": False,
            "restricted_access_count": 0,
        }
        if not unchanged:
            raise ProductionTwoStepQualificationV6Error(
                "preupdate backend rejection changed protected state"
            )
        root = self.output / "rejected_updates_v2"
        root.mkdir(parents=True, exist_ok=True)
        suffix = len(list(root.glob("attempt_*.json"))) + 1
        _atomic_json(root / f"attempt_{suffix:03d}.json", artifact)
        self._ratio_pre_context_v2 = None
        self._preupdate_backend_evidence_v2 = None
        self._preupdate_backend_health_v2 = None

    def _record_correction_gate_rejection_v2(
        self,
        *,
        step_index: int,
        correction: Mapping[str, Any],
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
    ) -> None:
        """Persist a fail-closed legacy correction rejection before backward."""

        state = self._transaction_state_v2
        context = self._ratio_pre_context_v2
        if not (
            state is not None
            and context is not None
            and self._pending_transaction_v2 is None
            and state.accepted_optimizer_steps == step_index
        ):
            raise ProductionTwoStepQualificationV6Error(
                "correction gate rejection state differs"
            )
        evidence = build_correction_gate_evidence_v2(
            correction,
            prompt_ids=prompt_ids,
            source_roles=source_roles,
            pool_binding_sha256=context["binding"].pool_binding_sha256,
            legacy_ess_fraction_min=float(
                self.config["validation"]["ess_fraction_min"]
            ),
            legacy_cap_fraction_max=float(
                self.config["validation"]["cap_fraction_max"]
            ),
            ratio_v2_per_prompt_ess_floor=float(
                self.ratio_thresholds_v2["per_prompt_ess_floor"]
            ),
            ratio_v2_per_prompt_ess_min_tokens=int(
                self.ratio_thresholds_v2["per_prompt_ess_min_tokens"]
            ),
            ratio_v2_backend_clip_fraction_max=float(
                self.ratio_thresholds_v2["backend_clip_fraction_max"]
            ),
        )
        before = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        after = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        unchanged = before == after
        artifact = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_rejected_update_v2",
            "run_id": self.config["run"]["run_id"],
            "attempted_optimizer_step": state.accepted_optimizer_steps + 1,
            "accepted_optimizer_steps": state.accepted_optimizer_steps,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "reason": "legacy_backend_correction_gate_rejected",
            "candidate_lora_sha256": before["lora_sha256"],
            "rollback": {
                "rollback_verified": unchanged,
                "candidate_executed": False,
                "optimizer_executed": False,
                "scheduler_executed": False,
                "rng_restore_required": False,
                "state_before": before,
                "state_after": after,
            },
            "ratio_evidence": evidence,
            "counts_as_optimizer_commit": False,
            "cursor_advanced": False,
            "sampler_refreshed": False,
            "restricted_access_count": 0,
        }
        if not unchanged:
            raise ProductionTwoStepQualificationV6Error(
                "correction gate rejection changed protected state"
            )
        root = self.output / "rejected_updates_v2"
        root.mkdir(parents=True, exist_ok=True)
        suffix = len(list(root.glob("attempt_*.json"))) + 1
        _atomic_json(root / f"attempt_{suffix:03d}.json", artifact)
        self._ratio_pre_context_v2 = None

    def _prepare_candidate_transaction_v2(
        self,
        *,
        step_index: int,
        gradient_norm_before_clip: float,
        bundle: Any,
        before_result: Any,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
        **_: Any,
    ) -> None:
        if self._ratio_pre_context_v2 is None or self._transaction_state_v2 is None:
            raise ProductionTwoStepQualificationV6Error(
                "transaction pre-update ratio context is absent"
            )
        grad = float(gradient_norm_before_clip)
        median = float(self.ratio_thresholds_v2["healthy_grad_median"])
        mad = max(float(self.ratio_thresholds_v2["healthy_grad_mad"]), 1.0e-12)
        robust_z = max(0.0, (grad - median) / mad)
        if (
            not math.isfinite(grad)
            or grad > float(self.ratio_thresholds_v2["preclip_grad_norm_absolute_max"])
            or robust_z > float(self.ratio_thresholds_v2["preclip_grad_robust_z_max"])
        ):
            self._record_pre_candidate_rejection_v2(
                reason=(
                    "precommit_gradient_health_v2_rejected:"
                    f"norm={grad:.9g},robust_z={robust_z:.9g}"
                ),
                gradient_norm_before_clip=grad,
                robust_z=robust_z,
                bundle=bundle,
                before_result=before_result,
                prompt_ids=prompt_ids,
                source_roles=source_roles,
            )
            raise ProductionTwoStepQualificationV6Error(
                "pre-commit gradient health gate rejected the fixed batch"
            )
        if self._transaction_state_v2.accepted_optimizer_steps != step_index:
            raise ProductionTwoStepQualificationV6Error(
                "transaction accepted-step identity differs"
            )
        self._pending_transaction_v2 = OptimizerTransactionV2.capture(
            model=self.student_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=self._transaction_state_v2,
            scratch_root=self._transaction_scratch_v2,
            fixed_batch_sha256=self._ratio_pre_context_v2["fixed_batch_sha256"],
        )

    def _record_pre_candidate_rejection_v2(
        self,
        *,
        reason: str,
        gradient_norm_before_clip: float,
        robust_z: float,
        bundle: Any,
        before_result: Any,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
    ) -> None:
        state = self._transaction_state_v2
        context = self._ratio_pre_context_v2
        if state is None or context is None or self._pending_transaction_v2 is not None:
            raise ProductionTwoStepQualificationV6Error(
                "pre-candidate rejection state differs"
            )
        mask = bundle.response_mask.detach().bool()
        counts = mask.sum(dim=1).to(dtype=before_result.token_surrogate.dtype)
        scale = counts[:, None] * float(mask.shape[0])
        loss_contribution = -before_result.token_surrogate.detach() / scale
        gradient_proxy = -(
            before_result.correction.truncated_weight.detach()
            * before_result.advantage.detach()
        ) / scale
        evidence = build_precommit_gradient_evidence_v2(
            valid_mask=mask,
            token_ids=context["response_ids"],
            advantage=before_result.advantage,
            gradient_proxy=gradient_proxy,
            loss_contribution=loss_contribution,
            prompt_ids=prompt_ids,
            source_roles=source_roles,
            pool_binding_sha256=context["binding"].pool_binding_sha256,
            gradient_norm_before_clip=gradient_norm_before_clip,
            robust_z=robust_z,
        )
        before = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        self.optimizer.zero_grad(set_to_none=True)
        after = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        }
        unchanged = before == after
        artifact = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_rejected_update_v2",
            "run_id": self.config["run"]["run_id"],
            "attempted_optimizer_step": state.accepted_optimizer_steps + 1,
            "accepted_optimizer_steps": state.accepted_optimizer_steps,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "reason": reason,
            "candidate_lora_sha256": before["lora_sha256"],
            "rollback": {
                "rollback_verified": unchanged,
                "candidate_executed": False,
                "optimizer_executed": False,
                "scheduler_executed": False,
                "rng_restore_required": False,
                "state_before": before,
                "state_after": after,
            },
            "ratio_evidence": evidence,
            "counts_as_optimizer_commit": False,
            "cursor_advanced": False,
            "sampler_refreshed": False,
            "restricted_access_count": 0,
        }
        if not unchanged:
            raise ProductionTwoStepQualificationV6Error(
                "pre-candidate rejection changed protected state"
            )
        root = self.output / "rejected_updates_v2"
        root.mkdir(parents=True, exist_ok=True)
        suffix = len(list(root.glob("attempt_*.json"))) + 1
        _atomic_json(root / f"attempt_{suffix:03d}.json", artifact)
        self._pending_ratio_evidence_v2 = None
        self._ratio_pre_context_v2 = None

    def _validate_candidate_update_v2(
        self,
        *,
        step_index: int,
        rows: list[Mapping[str, Any]],
        bundle: Any,
        before_result: Any,
        after_result: Any,
        after_mask: torch.Tensor,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
        gradient_norm_before_clip: float,
        telemetry: Mapping[str, Any],
        legacy_candidate_gate_passed: bool,
        legacy_candidate_gate_evidence: Mapping[str, Any],
    ) -> None:
        del rows
        context = self._ratio_pre_context_v2
        transaction = self._pending_transaction_v2
        if context is None or transaction is None:
            raise ProductionTwoStepQualificationV6Error(
                "candidate transaction snapshot is absent"
            )
        mask = bundle.response_mask.detach().bool()
        if not torch.equal(mask, after_mask.detach().bool()):
            self._abort_candidate_transaction_v2(reason="post_update_mask_drift")
            raise ProductionTwoStepQualificationV6Error(
                "candidate post-update mask differs"
            )
        counts = mask.sum(dim=1).to(dtype=before_result.token_surrogate.dtype)
        scale = counts[:, None] * float(mask.shape[0])
        loss_contribution = -before_result.token_surrogate.detach() / scale
        gradient_proxy = -(
            before_result.correction.truncated_weight.detach()
            * before_result.advantage.detach()
        ) / scale
        evidence = compute_ratio_evidence_v2(
            log_q_pre=bundle.current_actor_logprob,
            log_p_old_canonical=bundle.old_actor_logprob,
            log_mu_sampler=bundle.rollout_behavior_logprob,
            log_q_post=after_result.ppo_log_ratio.detach()
            + bundle.old_actor_logprob.detach(),
            valid_mask=mask,
            prompt_ids=prompt_ids,
            source_roles=source_roles,
            token_ids=context["response_ids"],
            advantage=before_result.advantage,
            loss_contribution=loss_contribution,
            gradient_proxy=gradient_proxy,
            pool_binding=context["binding"],
            policy_version=step_index,
            q_pre_adapter_sha256=context["adapter_sha256"],
            p_old_adapter_sha256=context["adapter_sha256"],
            sampler_version=self.current_sampler_version,
            refresh_version=self.current_sampler_version,
            backend_log_clip=math.log(2.0),
            post_shift_tail_abs_log_threshold=float(
                self.ratio_thresholds_v2["post_shift_tail_abs_log_threshold"]
            ),
        )
        if self._bounded_influence_config_v2 is not None:
            if self._last_bounded_influence_v2 is None:
                self._abort_candidate_transaction_v2(
                    reason="bounded_influence_evidence_absent"
                )
                raise ProductionTwoStepQualificationV6Error(
                    "bounded influence evidence is absent"
                )
            evidence["bounded_influence_v2"] = copy.deepcopy(
                self._last_bounded_influence_v2
            )
        evidence["legacy_candidate_gate"] = copy.deepcopy(
            dict(legacy_candidate_gate_evidence)
        )
        candidate_acceptance = None
        if self._candidate_acceptance_protocol_v2_1 is not None:
            candidate_acceptance = evaluate_candidate_acceptance_v2_1(
                legacy_candidate_gate_evidence,
                accepted_optimizer_steps=(
                    transaction.initial_state.accepted_optimizer_steps
                ),
            )
            evidence["candidate_acceptance_v2_1"] = copy.deepcopy(
                candidate_acceptance
            )
        self._pending_ratio_evidence_v2 = evidence
        update = telemetry.get("optimizer_update")
        if not isinstance(update, Mapping):
            self._abort_candidate_transaction_v2(reason="optimizer_telemetry_absent")
            raise ProductionTwoStepQualificationV6Error(
                "candidate optimizer telemetry is absent"
            )
        try:
            if candidate_acceptance is not None and not candidate_acceptance["passed"]:
                raise RatioHealthV2Error(
                    "candidate_acceptance_v2_1:"
                    + ",".join(candidate_acceptance["hard_failures"])
                )
            if candidate_acceptance is None and not legacy_candidate_gate_passed:
                failed_components = sorted(
                    key
                    for key in (
                        "objective_improved",
                        "loss_decreased",
                        "alignment_positive",
                        "optimizer_update_audit_passed",
                        "teacher_gradient_free",
                        "base_gradient_free",
                        "frozen_parameter_versions_unchanged",
                    )
                    if legacy_candidate_gate_evidence.get(key) is not True
                )
                raise RatioHealthV2Error(
                    "legacy_candidate_gate:" + ",".join(failed_components)
                )
            health = evaluate_ratio_health_v2(
                evidence,
                thresholds=self.ratio_thresholds_v2,
                preclip_grad_norm=gradient_norm_before_clip,
                relative_update_norm=float(update["relative_parameter_delta"]),
                ppo_clip_fraction=float(after_result.ppo_clip_fraction),
                consecutive_warning_count=self._consecutive_ratio_warning_count_v2,
            )
            if candidate_acceptance is not None and candidate_acceptance[
                "diagnostic_warnings"
            ]:
                health = dict(health)
                health["warnings"] = sorted(
                    set(health.get("warnings", []))
                    | {
                        "accumulated_adam_current_batch_nonmonotonic:"
                        + ",".join(candidate_acceptance["diagnostic_warnings"])
                    }
                )
        except (RatioHealthV2Error, KeyError, TypeError, ValueError) as error:
            self._abort_candidate_transaction_v2(
                reason="ratio_health_v2_rejected:" + str(error)
            )
            raise ProductionTwoStepQualificationV6Error(str(error)) from error
        self._last_ratio_health_v2 = dict(health)
        self._pending_warning_count_v2 = int(
            health["next_consecutive_warning_count"]
        )
        transaction.mark_candidate_validated()
        if self._diagnostic_unconditional_rollback_v2:
            self._abort_candidate_transaction_v2(
                reason="fixed_token_candidate_unconditional_rollback"
            )
            raise DiagnosticCandidateRollbackV2(
                "fixed-token candidate measured and rollback verified"
            )

    def _abort_candidate_transaction_v2(self, *, reason: str) -> None:
        transaction = self._pending_transaction_v2
        state = self._transaction_state_v2
        if transaction is None or state is None:
            return
        candidate_sha = ordered_trainable_sha256(self.student_model)
        evidence = self._pending_ratio_evidence_v2
        audit = transaction.reject(
            model=self.student_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=state,
            reason=reason,
            observed_registry_count=self._registry_count(),
        )
        self._optimizer_step_count = state.accepted_optimizer_steps
        self._scheduler_step_count = state.accepted_optimizer_steps
        artifact = {
            "schema_version": 2,
            "artifact_kind": "formal_b2_rejected_update_v2",
            "run_id": self.config["run"]["run_id"],
            "attempted_optimizer_step": state.accepted_optimizer_steps + 1,
            "accepted_optimizer_steps": state.accepted_optimizer_steps,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "reason": reason,
            "candidate_lora_sha256": candidate_sha,
            "rollback": audit,
            "ratio_evidence": evidence,
            "counts_as_optimizer_commit": False,
            "cursor_advanced": False,
            "sampler_refreshed": False,
            "restricted_access_count": 0,
        }
        root = self.output / "rejected_updates_v2"
        root.mkdir(parents=True, exist_ok=True)
        suffix = len(list(root.glob("attempt_*.json"))) + 1
        _atomic_json(root / f"attempt_{suffix:03d}.json", artifact)
        self._pending_transaction_v2 = None
        self._ratio_pre_context_v2 = None

    def _commit_candidate_transaction_v2(
        self, *, step_index: int, refresh: Mapping[str, Any]
    ) -> None:
        transaction = self._pending_transaction_v2
        state = self._transaction_state_v2
        evidence = self._pending_ratio_evidence_v2
        if transaction is None or state is None or evidence is None:
            raise ProductionTwoStepQualificationV6Error(
                "validated transaction is absent at commit"
            )
        audit = transaction.commit(
            model=self.student_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=state,
            prompts_per_step=4,
            observed_registry_count=self._registry_count(),
        )
        if not (
            state.accepted_optimizer_steps == step_index + 1
            and state.data_cursor == (step_index + 1) * 4
            and state.policy_version
            == state.sampler_version
            == state.refresh_version
            == self.current_sampler_version
            and int(refresh.get("logical_version", "v0").removeprefix("v"))
            == self.current_sampler_version
        ):
            raise ProductionTwoStepQualificationV6Error(
                "transaction commit/cursor/refresh identity differs"
            )
        self._optimizer_step_count = state.accepted_optimizer_steps
        self._consecutive_ratio_warning_count_v2 = self._pending_warning_count_v2
        self._last_ratio_evidence_v2 = copy.deepcopy(evidence)
        step = state.accepted_optimizer_steps
        _atomic_json(
            self.output / "ratio_evidence_v2" / f"step_{step:03d}.json",
            {
                "schema_version": 2,
                "artifact_kind": "formal_b2_committed_ratio_evidence_v2",
                "run_id": self.config["run"]["run_id"],
                "optimizer_step": step,
                "transaction": audit,
                "health": self._last_ratio_health_v2,
                "ratio_evidence": evidence,
                "accepted_optimizer_commit": True,
                "restricted_access_count": 0,
            },
        )
        self._pending_transaction_v2 = None
        self._pending_ratio_evidence_v2 = None
        self._ratio_pre_context_v2 = None

    def run_formal_step_v2(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        record = dict(
            super().run_formal_step_v1(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=max_new_tokens,
            )
        )
        if self._last_ratio_evidence_v2 is None or self._last_ratio_health_v2 is None:
            raise ProductionTwoStepQualificationV6Error(
                "committed v2 ratio evidence is absent"
            )
        record.update(
            {
                "artifact_kind": "formal_b2_step_v2",
                "accepted_optimizer_commit": True,
                "rejected_attempt_count_for_batch": 0,
                "ratio_v2": copy.deepcopy(self._last_ratio_evidence_v2),
                "ratio_health_v2": copy.deepcopy(self._last_ratio_health_v2),
            }
        )
        _atomic_json(
            self.formal_step_root / f"step_{int(record['optimizer_step']):03d}.json",
            record,
        )
        return record

    def seal_registered_checkpoint_v2(
        self,
        *,
        logical_version: int,
        package_content_sha256: str,
        config_sha256: str,
        manifest_sha256: str,
        schedule_sha256: str,
        environment: Mapping[str, Any],
    ) -> dict[str, Any]:
        from src.opd.production_b2_formal_checkpoint_v2 import (
            seal_formal_checkpoint_v2,
        )

        return seal_formal_checkpoint_v2(
            self,
            logical_version=logical_version,
            data_cursor=logical_version * 4,
            package_content_sha256=package_content_sha256,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            schedule_sha256=schedule_sha256,
            environment=environment,
        )

    def close(self) -> None:
        try:
            if self._pending_transaction_v2 is not None:
                self._abort_candidate_transaction_v2(reason="session_close_pending_candidate")
        finally:
            super().close()


def validate_formal_step_health_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_registry_count: int,
    initial_model_count: int,
) -> dict[str, Any]:
    """Validate committed records without re-aliasing raw post max as PPO ratio."""

    if not records:
        raise ProductionTwoStepQualificationV6Error("Formal B2 v2 health is empty")
    record = records[-1]
    health = record.get("ratio_health_v2")
    isolation = record.get("isolation")
    failures: list[str] = []
    if record.get("accepted_optimizer_commit") is not True:
        failures.append("not_an_accepted_optimizer_commit")
    if not isinstance(health, Mapping) or health.get("accepted") is not True:
        failures.append("ratio_health_v2_not_accepted")
    if not isinstance(record.get("ratio_v2"), Mapping):
        failures.append("ratio_evidence_v2_absent")
    if int(record.get("teacher_gradient_tensor_count", -1)) != 0 or int(
        record.get("base_gradient_tensor_count", -1)
    ) != 0:
        failures.append("teacher_or_base_gradient")
    if int(record.get("nonzero_update_tensor_count", 0)) <= 0 or float(
        record.get("adapter_delta_norm", 0.0)
    ) <= 0.0:
        failures.append("zero_update")
    if int(record.get("registry_count", -1)) != int(initial_registry_count) or int(
        record.get("model_count", -1)
    ) != int(initial_model_count):
        failures.append("registry_or_model_growth")
    if not isinstance(isolation, Mapping) or any(
        isolation.get(field) is not False
        for field in ("final_access", "controller_access", "confirmation_access", "label_access")
    ):
        failures.append("restricted_access")
    window_results: list[dict[str, Any]] = []
    for start in range(max(0, len(records) - 5), max(0, len(records) - 3)):
        window = records[start : start + 4]
        if len(window) != 4:
            continue
        samples = [sample for item in window for sample in item["prompt_samples"]]
        rates = {"overall": sum(bool(row["truncated"]) for row in samples) / len(samples)}
        for source in ("medical_opd_o1", "medical_opd_cmb"):
            source_rows = [row for row in samples if row["source"] == source]
            rates[source] = sum(bool(row["truncated"]) for row in source_rows) / len(source_rows)
        window_results.append(
            {
                "start_step": start + 1,
                "end_step": start + 4,
                "rates": rates,
                "over": any(value > 0.20 for value in rates.values()),
            }
        )
    if len(window_results) == 2 and all(item["over"] for item in window_results):
        failures.append("two_consecutive_truncation_windows")
    if failures:
        raise ProductionTwoStepQualificationV6Error(
            "Formal B2 v2 health failed: " + ",".join(failures)
        )
    return {
        "passed": True,
        "optimizer_step": int(record["optimizer_step"]),
        "window_results": window_results,
        "raw_ratio_max_is_not_a_standalone_abort": True,
    }


__all__ = [
    "DiagnosticCandidateRollbackV2",
    "FormalB2SessionV2",
    "build_ratio_pool_binding_v2",
    "validate_formal_step_health_v2",
]
