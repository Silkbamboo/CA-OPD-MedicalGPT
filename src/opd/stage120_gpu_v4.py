"""Reference-aligned P7 IDT/CA GPU session on the qualified formula-v6 kernel."""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.opd.production_b2_formal_gpu_v2 import (
    FormalB2SessionV2,
    _canonical_sha256,
    build_ratio_pool_binding_v2,
    evaluate_candidate_acceptance_v2_1,
)
from src.opd.production_b2_ratio_contract_v2 import compute_ratio_evidence_v2
from src.opd.production_b2_transaction_v2 import (
    ordered_trainable_sha256,
    state_tree_sha256,
)
from src.opd.production_main_method_v3 import P6FormalMethodError
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
    _atomic_json,
)
from src.opd.router import ConstraintAwareRouter, RouterConfig
from src.opd.stage120_backend_health_v3 import (
    P7BackendHealthError,
    evaluate_backend_health_v3,
)
from src.opd.stage120_protocol_v4 import (
    ActionKLSafetyV4,
    P7Stage120Error,
    Stage120RouteStateV4,
)
from src.opd.stage120_step46_qualification_v4 import compare_actual_impact_v4


class P7ActualImpactMeasurementRollback(RuntimeError):
    """One triggered counterfactual candidate was measured and rolled back."""


class P7CandidateHealthError(RuntimeError):
    """A P7 candidate violated a post-update or ownership contract."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persist_stage120_record_v4(
    output: Path, record: Mapping[str, Any]
) -> Path:
    """Persist the accepted record only after v4 action evidence is attached."""

    step = record.get("optimizer_step")
    evidence = record.get("stage120_v4")
    if not (
        isinstance(step, int)
        and not isinstance(step, bool)
        and step > 0
        and isinstance(evidence, Mapping)
        and evidence.get("schema_version") == 4
        and evidence.get("optimizer_step") == step
        and evidence.get("accepted") is True
        and evidence.get("final_access_count") == 0
    ):
        raise P7Stage120Error("P7 accepted record action evidence differs")
    path = Path(output) / "formal_steps" / f"step_{step:03d}.json"
    _atomic_json(path, dict(record))
    return path


def build_backend_actual_impact_v3(
    *,
    production_objective: float,
    canonical_objective: float,
    production_gradient: Tensor,
    canonical_gradient: Tensor,
    production_delta: Tensor,
    canonical_delta: Tensor,
    production_correction_ess: float,
    counterfactual_correction_ess: float,
    counterfactual_ppo_identity_max_abs: float,
    rollback_verified: bool,
) -> dict[str, Any]:
    """Map exact candidate vectors onto the frozen backend-health v3 schema."""

    if rollback_verified is not True:
        raise P7Stage120Error("P7 actual-impact candidate rollback is unverified")
    comparison = compare_actual_impact_v4(
        production_objective=production_objective,
        canonical_objective=canonical_objective,
        production_gradient=production_gradient,
        canonical_gradient=canonical_gradient,
        production_delta=production_delta,
        canonical_delta=canonical_delta,
    )
    gradient = comparison["gradient"]
    delta = comparison["adam_parameter_delta"]
    return {
        "schema_version": 3,
        "artifact_kind": "p7_backend_actual_impact_v3",
        "fixed_token_identity_verified": True,
        "production_objective": {
            "production": float(production_objective),
            "canonical_counterfactual": float(canonical_objective),
            "relative_l1_change": comparison["objective_relative_l1"],
        },
        "accumulated_gradient": {
            "relative_l2_change": gradient["relative_l2"],
            "cosine_similarity": gradient["cosine"],
            "both_zero": gradient["production_l2"] == gradient["canonical_l2"] == 0.0,
        },
        "adam_parameter_delta": {
            "relative_l2_change": delta["relative_l2"],
            "cosine_similarity": delta["cosine"],
            "both_zero": delta["production_l2"] == delta["canonical_l2"] == 0.0,
        },
        "counterfactual_ppo_identity_max_abs": float(
            counterfactual_ppo_identity_max_abs
        ),
        "production_correction_ess": float(production_correction_ess),
        "counterfactual_correction_ess": float(counterfactual_correction_ess),
        "candidate_committed": False,
        "unconditional_rollback_verified": True,
        "comparison_thresholds": comparison["thresholds"],
        "comparison_passed": comparison["passed"],
    }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P7CandidateHealthError(f"{label} is absent")
    result = float(value)
    if not math.isfinite(result):
        raise P7CandidateHealthError(f"{label} is non-finite")
    return result


def evaluate_p7_candidate_health_v4(
    evidence: Mapping[str, Any],
    *,
    backend_thresholds: Mapping[str, Any],
    legacy_thresholds: Mapping[str, Any],
    actual_impact: Mapping[str, Any] | None,
    preclip_grad_norm: float,
    relative_update_norm: float,
    ppo_clip_fraction: float,
    consecutive_warning_count: int,
) -> dict[str, Any]:
    """Apply v3 pre-health plus unchanged v2 post-update safety gates."""

    preupdate = evaluate_backend_health_v3(
        evidence, thresholds=backend_thresholds, actual_impact=actual_impact
    )
    ppo = evidence["ppo_ratio"]
    post = evidence["post_update_policy_shift"]
    failures: list[str] = []
    warnings = list(preupdate.get("diagnostic_warnings", []))
    checks = (
        ("approx_kl", abs(_number(ppo["approx_kl"], "approx KL")), legacy_thresholds["approx_kl_abs_max"]),
        ("ppo_clip_fraction", ppo_clip_fraction, legacy_thresholds["ppo_clip_fraction_max"]),
        ("relative_update_norm", relative_update_norm, legacy_thresholds["relative_update_norm_max"]),
        ("post_shift_abs_log_p99", post["log"]["abs_p99"], legacy_thresholds["post_shift_abs_log_p99_max"]),
        ("post_shift_abs_log_p999", post["log"]["abs_p999"], legacy_thresholds["post_shift_abs_log_p999_max"]),
        ("tail_loss_share", post["tail"]["absolute_loss_share"], legacy_thresholds["tail_loss_share_max"]),
        ("tail_gradient_proxy_share", post["tail"]["gradient_proxy_share"], legacy_thresholds["tail_gradient_proxy_share_max"]),
    )
    for label, observed, limit in checks:
        if _number(observed, label) > _number(limit, f"{label} limit"):
            failures.append(label)
    grad = _number(preclip_grad_norm, "preclip grad norm")
    if grad > _number(
        legacy_thresholds["preclip_grad_norm_absolute_max"], "absolute grad cap"
    ):
        failures.append("preclip_grad_absolute")
    median = _number(legacy_thresholds["healthy_grad_median"], "healthy grad median")
    mad = max(
        _number(legacy_thresholds["healthy_grad_mad"], "healthy grad MAD"),
        1.0e-12,
    )
    robust_z = max(0.0, (grad - median) / mad)
    if robust_z > _number(
        legacy_thresholds["preclip_grad_robust_z_max"], "grad robust-z cap"
    ):
        failures.append("preclip_grad_robust_z")
    raw_max = _number(post["ratio"]["max"], "raw post ratio max")
    if raw_max > _number(
        legacy_thresholds["raw_post_ratio_max_warning_above"], "raw ratio warning"
    ):
        warnings.append("raw_post_ratio_max")
    composite = "raw_post_ratio_max" in warnings
    next_warning_count = consecutive_warning_count + 1 if composite else 0
    if next_warning_count >= int(
        legacy_thresholds["consecutive_warning_abort_count"]
    ):
        failures.append("consecutive_composite_warnings")
    if failures:
        raise P7CandidateHealthError(
            "P7 candidate health rejected: " + ",".join(failures)
        )
    return {
        "schema_version": 4,
        "protocol_id": "p7_candidate_health_v4",
        "accepted": True,
        "failures": [],
        "warnings": warnings,
        "next_consecutive_warning_count": next_warning_count,
        "preclip_grad_robust_z": robust_z,
        "backend_health_v3": preupdate,
        "raw_post_ratio_max_is_diagnostic_only": True,
    }


class FormalStage120SessionV4(FormalB2SessionV2):
    """Complete-step Medical/General actions with rejection-safe CA state."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        method = config.get("stage120_v4")
        if not isinstance(method, Mapping) or method.get("method_id") not in {
            "IDT-v2",
            "CA-OPD-v2",
        }:
            raise P7Stage120Error("P7 Stage-120 method identity differs")
        self.stage120_config_v4 = copy.deepcopy(dict(method))
        self.method_id_v4 = str(method["method_id"])
        self.route_state_v4 = Stage120RouteStateV4(
            method_id=self.method_id_v4,
            seed=int(config["run"]["seed"]),
        )
        self.current_action_v4: str | None = None
        self.current_reserve_variant_v4 = 0
        self.current_action_evidence_v4: dict[str, Any] = {}
        self._pending_kl_preview_v4 = False
        self._actual_impact_phase_v4 = "normal"
        self._actual_impact_measurements_v4: dict[str, dict[str, Any]] = {}
        self._actual_impact_evidence_v4: dict[str, Any] | None = None
        self._actual_impact_pool_sha_v4: str | None = None
        self._actual_impact_current_objective_v4: float | None = None
        self._actual_impact_current_ess_v4: float | None = None
        self._actual_impact_current_gradient_v4: Tensor | None = None
        self.backend_health_thresholds_v4 = copy.deepcopy(
            dict(config.get("backend_health_v3", {}))
        )
        if self.backend_health_thresholds_v4.get("protocol_id") != "p7_backend_health_v3":
            raise P7Stage120Error("P7 backend health protocol is absent")
        if self.method_id_v4 == "CA-OPD-v2":
            router = method.get("router")
            safety = method.get("domain_kl_safety")
            if not isinstance(router, Mapping) or not isinstance(safety, Mapping):
                raise P7Stage120Error("P7 CA router or KL safety is absent")
            router_config = {
                key: value for key, value in dict(router).items() if key != "random_tape_seed"
            }
            self.ca_router_v4: ConstraintAwareRouter | None = ConstraintAwareRouter(
                RouterConfig.from_mapping(router_config), evaluator=None, seed=42
            )
            self.kl_safety_v4: ActionKLSafetyV4 | None = ActionKLSafetyV4(
                kappa={
                    "medical": float(safety["kappa_medical"]),
                    "general": float(safety["kappa_general"]),
                },
                rho=float(safety["rho"]),
                eps=float(safety["eps"]),
            )
        else:
            self.ca_router_v4 = None
            self.kl_safety_v4 = None
        teacher = config.get("teacher")
        if not isinstance(teacher, Mapping):
            raise P7Stage120Error("P7 Medical Teacher identity is absent")
        adapter = Path(str(teacher["adapter_path"]))
        manifest = Path(str(teacher["manifest_path"]))
        if not (
            adapter.is_dir()
            and _sha_file(adapter / "adapter_model.safetensors")
            == teacher.get("adapter_weight_sha256")
            and manifest.is_file()
            and _sha_file(manifest) == teacher.get("manifest_sha256")
        ):
            raise P7Stage120Error("P7 Medical Teacher SHA differs")
        super().__init__(config, **kwargs)
        (self.output / "stage120_action_steps_v4").mkdir(parents=True, exist_ok=True)
        (self.output / "actual_impact_v3").mkdir(parents=True, exist_ok=True)

    def current_p_medical_v4(self) -> float:
        return 0.5 if self.ca_router_v4 is None else float(self.ca_router_v4.p_medical)

    def select_action_for_attempt_v4(self) -> str:
        action = self.route_state_v4.action_for_slot(
            self.route_state_v4.accepted_steps,
            p_medical=self.current_p_medical_v4(),
        )
        self.current_action_v4 = action
        self.current_reserve_variant_v4 = self.route_state_v4.consecutive_rejections
        return action

    def _source_prompt_rows(
        self, prompt_rows: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Tokenize one complete Medical or General prompt-only action."""

        from src.opd.calibration_data import contains_forbidden_supervision

        if len(prompt_rows) != 4 or self.current_action_v4 not in {
            "medical",
            "general",
        }:
            raise P7Stage120Error("P7 action requires exactly four prompts")
        roles = [str(row.get("target_role", "")) for row in prompt_rows]
        datasets = [str(row.get("source", "")) for row in prompt_rows]
        if self.current_action_v4 == "medical" and not (
            roles.count("medical_opd_o1") == 2
            and roles.count("medical_opd_cmb") == 2
        ):
            raise P7Stage120Error("P7 Medical action is not 2 O1 plus 2 CMB")
        if self.current_action_v4 == "general" and not (
            roles == ["general_anchors"] * 4
            and datasets.count("BAAI/COIG") == 2
            and datasets.count("Instruction-Tuning-with-GPT-4/GPT-4-LLM")
            == 2
        ):
            raise P7Stage120Error("P7 General action stratification differs")
        expected_teacher = (
            "base" if self.current_action_v4 == "general" else "medical"
        )
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in prompt_rows:
            role = str(row.get("target_role", ""))
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            if not (
                not contains_forbidden_supervision(row)
                and all(marker not in role for marker in ("final", "controller", "confirmation"))
                and isinstance(sample_id, str)
                and sample_id
                and sample_id not in seen
                and isinstance(content_hash, str)
                and len(content_hash) == 64
                and all(character in "0123456789abcdef" for character in content_hash)
                and row.get("teacher_route") == expected_teacher
            ):
                raise P7Stage120Error("P7 prompt-only action identity differs")
            seen.add(sample_id)
            prompt = self.render_prompt_text(row)
            result.append(
                {
                    "fixture_id": sample_id,
                    "source_role": role,
                    "source_sample_id": sample_id,
                    "source_dataset": str(row.get("source", "")),
                    "source_subject": str(row.get("subject", "")),
                    "source_category": str(row.get("category", "")),
                    "source_license": str(row.get("source_license", "")),
                    "upstream_split": str(row.get("upstream_split", "")),
                    "teacher_route": expected_teacher,
                    "content_hash": content_hash,
                    "prompt_ids": [
                        int(value)
                        for value in self.tokenizer.apply_chat_template(
                            [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    ],
                }
            )
        return result

    def _score_step_teacher_rows(
        self,
        rows: list[Mapping[str, Any]],
        *,
        step_index: int,
        old_actor: Any,
        old_mask: Any,
    ) -> tuple[Any, Any]:
        if self.current_action_v4 not in {"medical", "general"}:
            raise P7Stage120Error("P7 complete-step action is absent")
        expected_route = "base" if self.current_action_v4 == "general" else "medical"
        if any(str(row.get("teacher_route")) != expected_route for row in rows):
            raise P7Stage120Error("P7 prompt domain and Teacher route differ")
        self._memory_phase_observer(
            "before", "medical_teacher_load", step=step_index + 1
        )
        teacher_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.teacher_model = self.PeftModel.from_pretrained(
            teacher_base,
            self.config["teacher"]["adapter_path"],
            adapter_name="medical",
            is_trainable=False,
        )
        del teacher_base
        self.teacher_model.eval()
        self._memory_phase_observer("after", "medical_teacher_load")
        context = (
            self.teacher_model.disable_adapter()
            if self.current_action_v4 == "general"
            else nullcontext()
        )
        with context:
            teacher, teacher_mask = self._score_rows(
                self.teacher_model, rows, device="cuda:1", inference=True
            )
        if not self.torch.equal(old_mask, teacher_mask):
            raise P7Stage120Error("P7 Teacher/Student response mask differs")
        reverse_kl = float(
            (old_actor - teacher)[teacher_mask].float().mean().detach().cpu()
        )
        scale = 1.0
        preview = None
        if self.kl_safety_v4 is not None:
            preview = self.kl_safety_v4.preview(
                action=self.current_action_v4, reverse_kl=reverse_kl
            )
            self._pending_kl_preview_v4 = True
            scale = float(preview["scale"])
            self._current_advantage_scale = self.torch.full_like(
                old_actor, scale, dtype=self.torch.float32
            ).detach()
        else:
            self._current_advantage_scale = None
        self.current_action_evidence_v4 = {
            "schema_version": 4,
            "method_id": self.method_id_v4,
            "accepted_slot": step_index,
            "attempted_optimizer_step": step_index + 1,
            "reserve_variant": self.current_reserve_variant_v4,
            "selection_probability_medical": self.current_p_medical_v4(),
            "random_tape_value": self.route_state_v4.random_tape[step_index],
            "action": self.current_action_v4,
            "teacher_route": expected_route,
            "source_roles": [str(row["source_role"]) for row in rows],
            "source_datasets": [str(row["source_dataset"]) for row in rows],
            "source_subjects": [str(row["source_subject"]) for row in rows],
            "source_licenses": [str(row["source_license"]) for row in rows],
            "upstream_splits": [str(row["upstream_split"]) for row in rows],
            "teacher_adapter_ordered_sha256": self.config["teacher"]["adapter_sha256"],
            "teacher_adapter_weight_sha256": self.config["teacher"]["adapter_weight_sha256"],
            "teacher_manifest_sha256": self.config["teacher"]["manifest_sha256"],
            "base_revision": self.base_revision,
            "reverse_kl": reverse_kl,
            "kl_safety_preview": preview,
            "kl_safety_scale": scale,
            "controller_state": (
                None if self.ca_router_v4 is None else self.ca_router_v4.state_dict()
            ),
            "final_access_count": 0,
        }
        return teacher, teacher_mask

    def run_stage120_attempt_v4(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if self.current_action_v4 is None:
            raise P7Stage120Error("P7 action was not selected")
        generation_snapshot = self.capture_generation_attempt_state_v4(
            step_index=step_index
        )
        generation_failure = (
            self.output
            / "steps"
            / f"generation_health_failure_step_{step_index + 1:02d}.json"
        )
        failure_sha_before = (
            _sha_file(generation_failure) if generation_failure.is_file() else None
        )
        try:
            record = dict(
                super().run_formal_step_v2(
                    step_index=step_index,
                    prompt_rows=prompt_rows,
                    max_new_tokens=max_new_tokens,
                )
            )
        except ProductionTwoStepQualificationV6Error:
            failure_sha_after = (
                _sha_file(generation_failure)
                if generation_failure.is_file()
                else None
            )
            if (
                failure_sha_after is not None
                and failure_sha_after != failure_sha_before
            ):
                self.record_generation_health_rejection_v4(
                    snapshot=generation_snapshot,
                    failure_path=generation_failure,
                )
            raise
        action = self.current_action_v4
        self.route_state_v4.accept(action=action)
        if self.kl_safety_v4 is not None:
            self.kl_safety_v4.accept_pending()
            self._pending_kl_preview_v4 = False
        evidence = {
            **copy.deepcopy(self.current_action_evidence_v4),
            "optimizer_step": int(record["optimizer_step"]),
            "accepted": True,
            "action_counts": dict(self.route_state_v4.action_counts),
            "route_state": self.route_state_v4.state_dict(),
            "kl_safety_state": (
                None if self.kl_safety_v4 is None else self.kl_safety_v4.state_dict()
            ),
            "backend_actual_impact_v3": copy.deepcopy(
                self._actual_impact_evidence_v4
            ),
            "final_access_count": 0,
        }
        record["stage120_v4"] = evidence
        persist_stage120_record_v4(self.output, record)
        _atomic_json(
            self.output
            / "stage120_action_steps_v4"
            / f"step_{int(record['optimizer_step']):03d}.json",
            evidence,
        )
        self._actual_impact_phase_v4 = "normal"
        self._actual_impact_measurements_v4 = {}
        self._actual_impact_evidence_v4 = None
        self._actual_impact_pool_sha_v4 = None
        self._actual_impact_current_objective_v4 = None
        self._actual_impact_current_ess_v4 = None
        self._actual_impact_current_gradient_v4 = None
        self.current_action_v4 = None
        return record

    def capture_generation_attempt_state_v4(
        self, *, step_index: int
    ) -> dict[str, Any]:
        """Capture the pre-generation state required for atomic rejection."""

        state = self._transaction_state_v2
        if not (
            state is not None
            and step_index == state.accepted_optimizer_steps
            and state.data_cursor == step_index * 4
            and state.policy_version
            == state.sampler_version
            == state.refresh_version
            == self.current_sampler_version
            and self.route_state_v4.accepted_steps == step_index
        ):
            raise P7Stage120Error("P7 generation attempt cursor/version differs")
        cpu_rng = self.torch.get_rng_state().clone()
        cuda_rng = [
            value.cpu().clone() for value in self.torch.cuda.get_rng_state_all()
        ] if self.torch.cuda.is_available() else []
        protected = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(cpu_rng),
            "cuda_rng_sha256": state_tree_sha256(cuda_rng),
        }
        return {
            "schema_version": 4,
            "attempted_optimizer_step": step_index + 1,
            "accepted_optimizer_steps": step_index,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "refresh_version": state.refresh_version,
            "protected_before": protected,
            "cpu_rng": cpu_rng,
            "cuda_rng": cuda_rng,
        }

    def record_generation_health_rejection_v4(
        self,
        *,
        snapshot: Mapping[str, Any],
        failure_path: Path,
    ) -> dict[str, Any]:
        """Restore RNG and register one privacy-safe pre-optimizer rejection."""

        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        state = self._transaction_state_v2
        step = int(snapshot["accepted_optimizer_steps"])
        isolation = failure.get("isolation")
        if not (
            isinstance(failure, Mapping)
            and failure.get("schema_version") == 1
            and failure.get("artifact_kind") == "b2_generation_health_failure_v1"
            and failure.get("run_id") == self.config["run"]["run_id"]
            and failure.get("optimizer_step") == step + 1
            and failure.get("policy_version") == step
            and failure.get("optimizer_executed") is False
            and failure.get("raw_prompt_persisted") is False
            and failure.get("response_tokens_persisted") is False
            and isinstance(isolation, Mapping)
            and all(
                isolation.get(key) is False
                for key in (
                    "confirmation_access",
                    "controller_access",
                    "final_access",
                    "label_access",
                )
            )
            and state is not None
            and state.accepted_optimizer_steps == step
            and state.data_cursor == snapshot["data_cursor"]
            and state.policy_version == snapshot["policy_version"]
            and state.sampler_version == snapshot["sampler_version"]
            and state.refresh_version == snapshot["refresh_version"]
            and self.current_sampler_version == step
            and self.route_state_v4.accepted_steps == step
            and getattr(self, "_pending_transaction_v2", None) is None
        ):
            raise P7Stage120Error("P7 generation rejection identity differs")
        protected_before = dict(snapshot["protected_before"])
        protected_pre_restore = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(self.torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(
                self.torch.cuda.get_rng_state_all()
                if self.torch.cuda.is_available()
                else []
            ),
        }
        for key in ("lora_sha256", "optimizer_sha256", "scheduler_sha256"):
            if protected_pre_restore[key] != protected_before[key]:
                raise P7Stage120Error(
                    "P7 preoptimizer generation rejection changed " + key
                )
        self.torch.set_rng_state(snapshot["cpu_rng"])
        if snapshot["cuda_rng"] and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(list(snapshot["cuda_rng"]))
        protected_after = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(self.torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(
                self.torch.cuda.get_rng_state_all()
                if self.torch.cuda.is_available()
                else []
            ),
        }
        if protected_after != protected_before:
            raise P7Stage120Error(
                "P7 preoptimizer generation rejection rollback differs"
            )
        generation_root = self.output / "generation_health_rejections_v4"
        generation_root.mkdir(parents=True, exist_ok=True)
        rejection_root = self.output / "rejected_updates_v2"
        rejection_root.mkdir(parents=True, exist_ok=True)
        suffix = len(list(rejection_root.glob("attempt_*.json"))) + 1
        generation_artifact = generation_root / f"attempt_{suffix:03d}.json"
        if generation_artifact.exists() or generation_artifact.is_symlink():
            raise P7Stage120Error("P7 generation rejection artifact exists")
        failure_sha256 = _sha_file(failure_path)
        failure_path.replace(generation_artifact)
        rejection = {
            "schema_version": 4,
            "artifact_kind": "formal_p7_preoptimizer_generation_rejection_v4",
            "run_id": self.config["run"]["run_id"],
            "attempted_optimizer_step": step + 1,
            "accepted_optimizer_steps": step,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "reason": "generation_health_v1_rejected",
            "candidate_lora_sha256": protected_before["lora_sha256"],
            "generation_health_artifact": str(generation_artifact),
            "generation_health_sha256": failure_sha256,
            "rollback": {
                "rollback_verified": True,
                "candidate_executed": False,
                "optimizer_executed": False,
                "scheduler_executed": False,
                "cpu_rng_restored": True,
                "cuda_rng_restored": bool(snapshot["cuda_rng"]),
                "state_before": protected_before,
                "state_pre_restore": protected_pre_restore,
                "state_after": protected_after,
            },
            "counts_as_optimizer_commit": False,
            "cursor_advanced": False,
            "sampler_refreshed": False,
            "restricted_access_count": 0,
            "final_access_count": 0,
        }
        _atomic_json(rejection_root / f"attempt_{suffix:03d}.json", rejection)
        return rejection

    def reject_stage120_attempt_v4(self, *, reason: str) -> dict[str, Any]:
        if self.current_action_v4 is None:
            raise P7Stage120Error("P7 rejected action is absent")
        action = self.current_action_v4
        self.route_state_v4.reject(action=action)
        kl = None
        if self.kl_safety_v4 is not None and self._pending_kl_preview_v4:
            kl = self.kl_safety_v4.reject_pending()
            self._pending_kl_preview_v4 = False
        result = {
            "schema_version": 4,
            "artifact_kind": "p7_stage120_action_rejection_v4",
            "accepted_slot": self.route_state_v4.accepted_steps,
            "action": action,
            "reason": reason,
            "route_state": self.route_state_v4.state_dict(),
            "kl_preview_rolled_back": kl,
            "counts_as_accepted_commit": False,
            "final_access_count": 0,
        }
        self._actual_impact_phase_v4 = "normal"
        self._actual_impact_measurements_v4 = {}
        self._actual_impact_evidence_v4 = None
        self._actual_impact_pool_sha_v4 = None
        self._actual_impact_current_objective_v4 = None
        self._actual_impact_current_ess_v4 = None
        self._actual_impact_current_gradient_v4 = None
        self.current_action_v4 = None
        return result

    def formal_route_state(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "method": self.method_id_v4,
            "route_state": self.route_state_v4.state_dict(),
            "kl_safety_state": (
                None if self.kl_safety_v4 is None else self.kl_safety_v4.state_dict()
            ),
            "ca_router_state": (
                None if self.ca_router_v4 is None else self.ca_router_v4.state_dict()
            ),
            "actual_impact_phase": self._actual_impact_phase_v4,
            "pending_action": self.current_action_v4,
            "final_access_count": 0,
        }

    def restore_formal_route_state(self, value: Mapping[str, Any]) -> None:
        if not (
            value.get("schema_version") == 4
            and value.get("method") == self.method_id_v4
            and value.get("actual_impact_phase") == "normal"
            and value.get("pending_action") is None
        ):
            raise P7Stage120Error("P7 route checkpoint boundary differs")
        self.route_state_v4.load_state_dict(value["route_state"])
        if self.kl_safety_v4 is not None:
            self.kl_safety_v4.load_state_dict(value["kl_safety_state"])
            if self.ca_router_v4 is None:
                raise P7Stage120Error("P7 CA router is absent on resume")
            self.ca_router_v4.load_state_dict(value["ca_router_state"])

    def update_ca_controller_v4(
        self,
        *,
        medical_accuracy: float,
        general_accuracy: float,
        completed_step: int,
    ) -> dict[str, Any]:
        if self.ca_router_v4 is None or not self.ca_router_v4.is_window_boundary(
            completed_step
        ):
            raise P7Stage120Error("P7 CA Controller boundary differs")
        return self.ca_router_v4.update(
            medical_accuracy=medical_accuracy,
            general_accuracy=general_accuracy,
            step=completed_step,
        ).as_dict()

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
        if self._pending_transaction_v2 is not None:
            raise P7Stage120Error("P7 previous optimizer transaction remains pending")
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
            identity_gap
            <= float(self.ratio_thresholds_v2["ppo_abs_log_p999_max"])
            and isinstance(adapter_sha, str)
            and len(adapter_sha) == 64
            and step_index == self.current_sampler_version
        ):
            raise P7Stage120Error("P7 canonical q/p_old identity differs")
        if self._actual_impact_pool_sha_v4 is None:
            self._actual_impact_pool_sha_v4 = binding.pool_binding_sha256
        elif self._actual_impact_pool_sha_v4 != binding.pool_binding_sha256:
            raise P7Stage120Error("P7 actual-impact fixed token pool drifted")
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
        actual = (
            self._actual_impact_evidence_v4
            if self._actual_impact_phase_v4 == "commit"
            else None
        )
        try:
            health = evaluate_backend_health_v3(
                evidence,
                thresholds=self.backend_health_thresholds_v4,
                actual_impact=actual,
            )
        except P7BackendHealthError as error:
            if (
                "actual-impact qualification is required" in str(error)
                and self._actual_impact_phase_v4
                in {"normal", "production_measurement", "canonical_measurement"}
            ):
                if self._actual_impact_phase_v4 == "normal":
                    self._actual_impact_phase_v4 = "production_measurement"
                health = {
                    "schema_version": 4,
                    "protocol_id": "p7_backend_health_v3_measurement_pending",
                    "accepted": True,
                    "formal_health_acceptance": False,
                    "measurement_phase": self._actual_impact_phase_v4,
                    "raw_trigger_preserved": True,
                }
            else:
                self._record_preupdate_backend_rejection_v2(
                    evidence=evidence, reason=str(error)
                )
                raise P7Stage120Error(str(error)) from error
        self._preupdate_backend_health_v2 = health
        objective_result = before_result
        if self._actual_impact_phase_v4 == "canonical_measurement":
            canonical_bundle = self.ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.old_actor_logprob.detach(),
                old_actor_logprob=bundle.old_actor_logprob,
                current_actor_logprob=bundle.current_actor_logprob,
                teacher_logprob=bundle.teacher_logprob,
                response_mask=bundle.response_mask,
                behavior_provenance=bundle.behavior_provenance,
            )
            objective_result = self.decoupled_corrected_objective(
                canonical_bundle,
                prompt_ids=prompt_ids,
                group_ids=("g0",) * len(prompt_ids),
                source_roles=source_roles,
                beta=float(self.algorithm["beta"]),
                clip_low=float(self.algorithm["clip_low"]),
                clip_high=float(self.algorithm["clip_high"]),
                rollout_is_threshold=2.0,
                advantage_scale=getattr(self, "_current_advantage_scale", None),
            )
        self._actual_impact_current_objective_v4 = float(
            objective_result.surrogate.detach().cpu()
        )
        self._actual_impact_current_ess_v4 = float(
            objective_result.correction.metrics["ess_fraction"]
        )

    def _backward_corrected_rows(self, **kwargs: Any) -> None:
        if self._actual_impact_phase_v4 == "canonical_measurement":
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

    def _validate_source_roles_for_backward(
        self, source_roles: tuple[str, ...]
    ) -> None:
        if self.current_action_v4 == "medical":
            return super()._validate_source_roles_for_backward(source_roles)
        if self.current_action_v4 == "general" and source_roles == (
            "general_anchors",
            "general_anchors",
            "general_anchors",
            "general_anchors",
        ):
            return
        if self.current_action_v4 == "general":
            raise P7Stage120Error("P7 General action backward source batch differs")
        raise P7Stage120Error("P7 backward action is absent")

    def _expected_step_record_source_counts_v4(self) -> dict[str, int]:
        if self.current_action_v4 == "medical":
            return {"medical_opd_o1": 2, "medical_opd_cmb": 2}
        if self.current_action_v4 == "general":
            return {"general_anchors": 4}
        raise P7Stage120Error("P7 safe-record action is absent")

    def _prepare_candidate_transaction_v2(self, **kwargs: Any) -> None:
        super()._prepare_candidate_transaction_v2(**kwargs)
        if self._actual_impact_phase_v4 in {
            "production_measurement",
            "canonical_measurement",
        }:
            self._actual_impact_current_gradient_v4 = torch.cat(
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

    def _measurement_candidate_v4(self) -> dict[str, Any]:
        transaction = self._pending_transaction_v2
        if transaction is None:
            raise P7Stage120Error("P7 measurement transaction is absent")
        snapshot = transaction._load()
        delta = torch.cat(
            [
                (
                    self.parameters[name].detach().float().cpu()
                    - snapshot["trainable"][name].detach().float().cpu()
                ).reshape(-1)
                for name in self.trainable_names
            ]
        )
        if not (
            self._actual_impact_current_objective_v4 is not None
            and self._actual_impact_current_ess_v4 is not None
            and self._actual_impact_current_gradient_v4 is not None
        ):
            raise P7Stage120Error("P7 measurement candidate tensors are absent")
        return {
            "objective": self._actual_impact_current_objective_v4,
            "ess": self._actual_impact_current_ess_v4,
            "gradient": self._actual_impact_current_gradient_v4,
            "delta": delta,
        }

    def _validate_candidate_update_v2(self, **kwargs: Any) -> None:
        if self._actual_impact_phase_v4 in {
            "production_measurement",
            "canonical_measurement",
        }:
            phase = self._actual_impact_phase_v4
            self._actual_impact_measurements_v4[phase] = self._measurement_candidate_v4()
            transaction = self._pending_transaction_v2
            if transaction is None:
                raise P7Stage120Error("P7 measurement transaction disappeared")
            self._pending_ratio_evidence_v2 = copy.deepcopy(
                self._preupdate_backend_evidence_v2
            )
            transaction.mark_candidate_validated()
            self._abort_candidate_transaction_v2(
                reason=f"p7_{phase}_unconditional_rollback"
            )
            raise P7ActualImpactMeasurementRollback(
                f"P7 {phase} measured and rolled back"
            )
        if self._actual_impact_phase_v4 != "commit":
            return super()._validate_candidate_update_v2(**kwargs)
        self._validate_candidate_commit_v4(**kwargs)

    def _validate_candidate_commit_v4(
        self,
        *,
        step_index: int,
        rows: list[Mapping[str, Any]],
        bundle: Any,
        before_result: Any,
        after_result: Any,
        after_mask: Tensor,
        prompt_ids: Sequence[str],
        source_roles: Sequence[str],
        gradient_norm_before_clip: float,
        telemetry: Mapping[str, Any],
        legacy_candidate_gate_passed: bool,
        legacy_candidate_gate_evidence: Mapping[str, Any],
    ) -> None:
        del rows, legacy_candidate_gate_passed
        context = self._ratio_pre_context_v2
        transaction = self._pending_transaction_v2
        if context is None or transaction is None:
            raise P7Stage120Error("P7 candidate transaction snapshot is absent")
        mask = bundle.response_mask.detach().bool()
        if not torch.equal(mask, after_mask.detach().bool()):
            self._abort_candidate_transaction_v2(reason="p7_post_update_mask_drift")
            raise P7Stage120Error("P7 candidate post-update mask differs")
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
        acceptance = evaluate_candidate_acceptance_v2_1(
            legacy_candidate_gate_evidence,
            accepted_optimizer_steps=transaction.initial_state.accepted_optimizer_steps,
        )
        evidence["candidate_acceptance_v2_1"] = copy.deepcopy(acceptance)
        evidence["backend_actual_impact_v3"] = copy.deepcopy(
            self._actual_impact_evidence_v4
        )
        self._pending_ratio_evidence_v2 = evidence
        update = telemetry.get("optimizer_update")
        if not isinstance(update, Mapping):
            self._abort_candidate_transaction_v2(reason="p7_optimizer_telemetry_absent")
            raise P7Stage120Error("P7 optimizer telemetry is absent")
        try:
            if not acceptance["passed"]:
                raise P7CandidateHealthError(
                    "candidate ownership rejected:" + ",".join(acceptance["hard_failures"])
                )
            health = evaluate_p7_candidate_health_v4(
                evidence,
                backend_thresholds=self.backend_health_thresholds_v4,
                legacy_thresholds=self.ratio_thresholds_v2,
                actual_impact=self._actual_impact_evidence_v4,
                preclip_grad_norm=gradient_norm_before_clip,
                relative_update_norm=float(update["relative_parameter_delta"]),
                ppo_clip_fraction=float(after_result.ppo_clip_fraction),
                consecutive_warning_count=self._consecutive_ratio_warning_count_v2,
            )
        except (P7BackendHealthError, P7CandidateHealthError, KeyError, TypeError, ValueError) as error:
            self._abort_candidate_transaction_v2(
                reason="p7_candidate_health_rejected:" + str(error)
            )
            raise P7Stage120Error(str(error)) from error
        self._last_ratio_health_v2 = health
        self._pending_warning_count_v2 = int(
            health["next_consecutive_warning_count"]
        )
        transaction.mark_candidate_validated()

    def advance_actual_impact_phase_v4(self) -> dict[str, Any]:
        """Advance only after the GPU-owned candidate rollback completed."""

        if self._pending_kl_preview_v4 and self.kl_safety_v4 is not None:
            self.kl_safety_v4.reject_pending()
            self._pending_kl_preview_v4 = False
        phase = self._actual_impact_phase_v4
        if phase == "production_measurement":
            if phase not in self._actual_impact_measurements_v4:
                raise P7Stage120Error("P7 production measurement is absent")
            self._actual_impact_phase_v4 = "canonical_measurement"
        elif phase == "canonical_measurement":
            production = self._actual_impact_measurements_v4.get(
                "production_measurement"
            )
            canonical = self._actual_impact_measurements_v4.get(
                "canonical_measurement"
            )
            if not isinstance(production, Mapping) or not isinstance(
                canonical, Mapping
            ):
                raise P7Stage120Error("P7 paired actual-impact measurements differ")
            impact = build_backend_actual_impact_v3(
                production_objective=float(production["objective"]),
                canonical_objective=float(canonical["objective"]),
                production_gradient=production["gradient"],
                canonical_gradient=canonical["gradient"],
                production_delta=production["delta"],
                canonical_delta=canonical["delta"],
                production_correction_ess=float(production["ess"]),
                counterfactual_correction_ess=float(canonical["ess"]),
                counterfactual_ppo_identity_max_abs=0.0,
                rollback_verified=True,
            )
            try:
                evaluate_backend_health_v3(
                    self._preupdate_backend_evidence_v2,
                    thresholds=self.backend_health_thresholds_v4,
                    actual_impact=impact,
                )
            except P7BackendHealthError as error:
                _atomic_json(
                    self.output
                    / "actual_impact_rejections_v4"
                    / (
                        "attempt_"
                        f"{self.route_state_v4.rejected_attempts + 1:03d}.json"
                    ),
                    {
                        "schema_version": 4,
                        "artifact_kind": "p7_backend_actual_impact_rejection_v4",
                        "accepted_optimizer_step": self.route_state_v4.accepted_steps,
                        "attempted_optimizer_step": (
                            self.route_state_v4.accepted_steps + 1
                        ),
                        "counts_as_accepted_commit": False,
                        "fixed_pool_sha256": self._actual_impact_pool_sha_v4,
                        "backend_ratio_evidence_sha256": _canonical_sha256(
                            self._preupdate_backend_evidence_v2
                        ),
                        "backend_health_thresholds_sha256": _canonical_sha256(
                            self.backend_health_thresholds_v4
                        ),
                        "actual_impact": impact,
                        "health_evaluation": {
                            "accepted": False,
                            "failure": str(error),
                        },
                        "candidate_committed": False,
                        "unconditional_rollback_verified": True,
                        "final_access_count": 0,
                    },
                )
                raise
            self._actual_impact_evidence_v4 = impact
            self._actual_impact_phase_v4 = "commit"
            _atomic_json(
                self.output
                / "actual_impact_v3"
                / f"step_{self.route_state_v4.accepted_steps + 1:03d}.json",
                impact,
            )
        else:
            raise P7Stage120Error("P7 actual-impact phase cannot advance")
        self._actual_impact_current_objective_v4 = None
        self._actual_impact_current_ess_v4 = None
        self._actual_impact_current_gradient_v4 = None
        return {
            "phase_before": phase,
            "phase_after": self._actual_impact_phase_v4,
            "fixed_pool_sha256": self._actual_impact_pool_sha_v4,
            "counts_as_rejection": False,
            "action_redrawn": False,
            "final_access_count": 0,
        }


__all__ = [
    "FormalStage120SessionV4",
    "P7ActualImpactMeasurementRollback",
    "build_backend_actual_impact_v3",
    "persist_stage120_record_v4",
]
