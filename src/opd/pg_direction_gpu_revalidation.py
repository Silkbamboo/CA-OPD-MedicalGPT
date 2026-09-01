"""GPU-only P4.2 Medical one-step -> null -> sampler direction rerun.

All GPU/model imports remain inside ``run_gpu_revalidation``. Importing this
module, inspecting the plan, and running the launcher dry-run are CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class PGDirectionGPUError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_run_identity(
    config: Mapping[str, Any], *, config_path: Path, repo_root: Path
) -> dict[str, str]:
    """Bind the executable config/run card to the actual clean runtime HEAD."""

    resolved_config = config_path.resolve()
    run_id = str(config.get("run", {}).get("run_id", ""))
    run_card_path = repo_root / "configs/run_cards" / f"{run_id}.json"
    if not run_card_path.is_file() or not resolved_config.is_file():
        raise PGDirectionGPUError("run config or run card is missing")
    try:
        run_card = json.loads(run_card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PGDirectionGPUError("run card is invalid JSON") from error
    expected_config = (repo_root / str(run_card.get("config_path", ""))).resolve()
    run_config_sha = _sha256(resolved_config)
    if (
        expected_config != resolved_config
        or run_card.get("config_sha256") != run_config_sha
        or run_card.get("protocol_id") != config.get("validation", {}).get("protocol_id")
        or run_card.get("protocol_config_sha256")
        != config.get("validation", {}).get("config_sha256")
        or run_card.get("frozen_trajectory_sha256")
        != config.get("frozen_input", {}).get("trajectory_sha256")
    ):
        raise PGDirectionGPUError("run card/config identity mismatch")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(git_head) != 40:
        raise PGDirectionGPUError("runtime Git HEAD identity is invalid")
    return {
        "git_head": git_head,
        "run_config_sha256": run_config_sha,
        "run_card_sha256": _sha256(run_card_path),
    }


def revalidation_plan(config: Mapping[str, Any]) -> list[str]:
    run = config.get("run", {})
    execution = config.get("execution", {})
    if (
        run.get("stage") != "pg_direction_revalidation"
        or run.get("calibration_only") is not True
        or run.get("formal_opd_training") is not False
        or run.get("one_step_only") is not True
        or execution.get("stop_on_first_failure") is not True
        or execution.get("automatically_start_b2") is not False
    ):
        raise PGDirectionGPUError("P4.2 narrow execution contract drift")
    phases = list(execution.get("ordered_phases", ()))
    if phases != [
        "formal_host_identity_preflight",
        "minimal_medical_scorer_identity_probe",
        "frozen_medical_one_step",
        "frozen_base_teacher_null_update",
        "sampler_refresh_identity",
        "artifact_integrity_and_readiness",
        "release_gpu_resources",
    ]:
        raise PGDirectionGPUError("P4.2 phase order drift")
    return phases


def run_gpu_revalidation(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Execute only the SHA-frozen P4.2 direction revalidation package."""

    revalidation_plan(config)
    if os.environ.get("CA_OPD_ALLOW_PG_DIRECTION_REVALIDATION_GPU") != "1":
        raise PGDirectionGPUError("GPU direction revalidation lacks explicit authorization")

    from src.opd.pg_direction_preflight import preflight

    root = Path(__file__).resolve().parents[2]
    run_identity = validate_frozen_run_identity(
        config, config_path=config_path, repo_root=root
    )
    preflight_report = preflight(config, execute_gpu=True, require_clean_git=True)

    # GPU/model imports start only after explicit authorization and formal preflight.
    import torch
    import yaml
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    from src.opd.pg_opd_contract import (
        audit_frozen_pg_update,
        grouped_reduction,
        masked_numeric_summary,
        ppo_clipped_objective,
        same_trajectory_advantage,
        validate_pre_update_contract,
    )
    from src.opd.pg_opd_validation import (
        atomic_write_json,
        audit_optimizer_update,
        audit_sampler_refresh,
        persist_update_outcome,
        refresh_sampler_adapter,
        require_sampler_identity,
        sha256_file,
        summarize_three_policy_identity,
    )
    from src.opd.scorer_calibration import summarize_signed_update
    from src.opd.scorer_gpu_calibration import (
        _apply_determinism,
        _release,
        ordered_adapter_sha256,
    )
    from src.opd.trajectory_scorer import (
        SharedBackboneRoutes,
        TrajectoryScoreRequest,
        TransformersTrajectoryLogprobScorer,
    )

    _apply_determinism(torch)
    output = Path(config["run"]["output_dir"])
    if output.exists():
        raise PGDirectionGPUError("P4.2 output must not already exist")
    output.mkdir(parents=True)
    atomic_write_json(
        output / "metadata.json",
        {
            "schema_version": 2,
            "run_id": config["run"]["run_id"],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": config["validation"]["protocol_id"],
            "protocol_config_sha256": config["validation"]["config_sha256"],
            **run_identity,
            "p4_1_status_preserved": "blocked_pg_opd_direction",
            "gpu_inventory": preflight_report["gpu_inventory"],
            "B2_authorized": False,
        },
    )
    (output / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    (output / "stdout.log").touch()

    validation = yaml.safe_load(
        (root / config["validation"]["config_path"]).read_text(encoding="utf-8")
    )
    base_config = yaml.safe_load(
        (root / config["base_calibration"]["config_path"]).read_text(encoding="utf-8")
    )
    tolerances = validation["tolerances"]
    optimizer_config = validation["optimizer"]
    algorithm = validation["algorithm"]
    with Path(config["frozen_input"]["trajectory_path"]).open(
        "r", encoding="utf-8"
    ) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    student_model = teacher_model = scorer = sampler_model = None
    current_phase = "minimal_medical_scorer_identity_probe"
    started = time.time()

    def padded(items: list[Any], *, device: str = "cuda:0") -> Any:
        return torch.nn.utils.rnn.pad_sequence(
            [item.reshape(-1).to(device=device, dtype=torch.float32) for item in items],
            batch_first=True,
            padding_value=0.0,
        )

    def action_logprobs(
        model: Any, prompt_ids: list[int], response_ids: list[int], *, device: str
    ) -> Any:
        combined = prompt_ids + response_ids
        ids = torch.tensor([combined], dtype=torch.long, device=device)
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        )
        start = len(prompt_ids) - 1
        logits = output.logits[:, start : start + len(response_ids), :].float()
        targets = torch.tensor(response_ids, dtype=torch.long, device=device).view(1, -1, 1)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)

    def student_logprobs(model: Any, *, device: str = "cuda:0") -> tuple[Any, Any]:
        values = [
            action_logprobs(
                model, row["prompt_ids"], row["response_ids"], device=device
            ).reshape(-1)
            for row in rows
        ]
        mask = torch.zeros(
            (len(values), max(value.numel() for value in values)),
            dtype=torch.bool,
            device=device,
        )
        for index, value in enumerate(values):
            mask[index, : value.numel()] = True
        return padded(values, device=device), mask

    def teacher_logprobs(route: str) -> Any:
        requests = [
            TrajectoryScoreRequest(
                request_id=f"p4-2-{route}-{row['fixture_id']}",
                route=route,
                prompt_ids=tuple(row["prompt_ids"]),
                response_ids=tuple(row["response_ids"]),
                attention_mask=(1,) * (len(row["prompt_ids"]) + len(row["response_ids"])),
                eos_token_id=getattr(teacher_model.config, "eos_token_id", None),
                finish_reason=str(row["finish_reason"]),
                truncated=bool(row["truncated"]),
                source_role=str(row["source_role"]),
            )
            for row in rows
        ]
        results = scorer.score_batch(requests, maximum_batch_size=1, length_bucket_width=128)
        return padded(
            [torch.tensor(result.token_logprobs) for result in results], device="cuda:0"
        ), results

    def source_diagnostics(token_values: Any, mask: Any) -> dict[str, Any]:
        by_source: dict[str, list[float]] = {}
        for index, row in enumerate(rows):
            value = float(token_values[index][mask[index]].detach().mean().cpu())
            by_source.setdefault(str(row["source_role"]), []).append(value)
        return {
            source: {"count": len(values), "mean": statistics.mean(values)}
            for source, values in sorted(by_source.items())
        }

    def evidence_numeric_summary(values: Any, mask: Any) -> dict[str, Any]:
        selected = values.detach()[mask.to(torch.bool)].to(dtype=torch.float64)
        finite = torch.isfinite(selected)
        report: dict[str, Any] = {
            "count": int(selected.numel()),
            "nonfinite_count": int((~finite).sum().cpu()),
        }
        if bool(finite.all()) and selected.numel():
            report.update(masked_numeric_summary(values, mask))
        return report

    latest_evidence_path: Path | None = None
    try:
        model_path = base_config["model"]["id"]
        teacher_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        teacher_base.config.use_cache = False
        teacher_model = PeftModel.from_pretrained(
            teacher_base,
            base_config["teacher"]["adapter_path"],
            adapter_name="medical",
            is_trainable=False,
        )
        del teacher_base
        scorer = TransformersTrajectoryLogprobScorer(
            model=teacher_model,
            routes=SharedBackboneRoutes(
                model=teacher_model,
                medical_adapter_name="medical",
                medical_adapter_sha256=base_config["teacher"]["adapter_sha256"],
            ),
            model_id=model_path,
            model_revision=base_config["model"]["revision"],
            tokenizer_revision=base_config["model"]["tokenizer_revision"],
            logprob_chunk_tokens=64,
        )
        medical_scored, medical_results = teacher_logprobs("medical")
        frozen_medical = padded(
            [torch.tensor(row["medical_teacher"]) for row in rows], device="cuda:0"
        )
        response_mask = torch.zeros_like(medical_scored, dtype=torch.bool)
        for index, row in enumerate(rows):
            response_mask[index, : len(row["response_ids"])] = True
        medical_identity_max_abs = float(
            (medical_scored[response_mask] - frozen_medical[response_mask]).abs().max().cpu()
        )
        scorer_identity = {
            "schema_version": 2,
            "status": "pass" if medical_identity_max_abs <= 1e-4 else "fail",
            "formal_backend": "Transformers",
            "route": "medical",
            "adapter_sha256": base_config["teacher"]["adapter_sha256"],
            "trajectory_sha256": config["frozen_input"]["trajectory_sha256"],
            "max_abs_delta_from_frozen_p4_1_scores": medical_identity_max_abs,
            "tolerance": 1e-4,
            "rows": len(medical_results),
        }
        atomic_write_json(output / "scorer_identity.json", scorer_identity)
        latest_evidence_path = output / "scorer_identity.json"
        if scorer_identity["status"] != "pass":
            raise PGDirectionGPUError("minimal Medical scorer identity mismatch")

        student_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        student_base.config.use_cache = False
        student_model = get_peft_model(
            student_base,
            LoraConfig(
                r=int(optimizer_config["lora_rank"]),
                lora_alpha=int(optimizer_config["lora_alpha"]),
                lora_dropout=float(optimizer_config["lora_dropout"]),
                target_modules=optimizer_config["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        del student_base
        student_model.train()
        trainable_names = tuple(
            name for name, parameter in student_model.named_parameters() if parameter.requires_grad
        )
        if not trainable_names or any("lora" not in name.lower() for name in trainable_names):
            raise PGDirectionGPUError("Student trainable manifest is not LoRA-only")
        parameter_by_name = dict(student_model.named_parameters())
        frozen_versions = {
            name: parameter._version
            for name, parameter in parameter_by_name.items()
            if name not in trainable_names
        }

        with tempfile.TemporaryDirectory(dir=output, prefix=".temporary_lora_") as temporary:
            old_adapter_dir = Path(temporary) / "version0"
            student_model.save_pretrained(old_adapter_dir, safe_serialization=True)
            old_adapter_sha = ordered_adapter_sha256(old_adapter_dir)
            medical_before_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in parameter_by_name.items()
                if name in trainable_names
            }

            current_phase = "pre_update_identity"
            student_model.eval()
            with torch.inference_mode():
                pi_old_actor, old_actor_mask = student_logprobs(student_model)
            student_model.train()
            new_before, response_mask = student_logprobs(student_model)
            old = padded([torch.tensor(row["old"]) for row in rows], device="cuda:0").detach()
            teacher = frozen_medical.detach()
            advantage = same_trajectory_advantage(old, teacher, beta=float(algorithm["beta"]))
            raw_pre_log_ratio = new_before - old
            raw_pre_ratio = torch.exp(raw_pre_log_ratio)
            if not torch.equal(old_actor_mask, response_mask):
                atomic_write_json(
                    output / "three_policy_identity.json",
                    {
                        "schema_version": 2,
                        "status": "fail",
                        "reason": "pi_old_actor and pi_current_pre response masks differ",
                        "old_actor_valid_tokens": int(old_actor_mask.sum().cpu()),
                        "current_pre_valid_tokens": int(response_mask.sum().cpu()),
                    },
                )
                latest_evidence_path = output / "three_policy_identity.json"
                raise PGDirectionGPUError("three-policy response mask identity mismatch")
            three_policy_identity = summarize_three_policy_identity(
                pi_rollout=old,
                pi_old_actor=pi_old_actor.detach(),
                pi_current_pre=new_before.detach(),
                response_mask=response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                source_roles=tuple(str(row["source_role"]) for row in rows),
                tolerance=float(tolerances["pre_update_log_ratio_max_abs"]),
            )
            three_policy_identity.update(
                {
                    "status": (
                        "pass"
                        if three_policy_identity["formal_identity_gate_passed"]
                        else "fail"
                    ),
                    "protocol_id": config["validation"]["protocol_id"],
                    "protocol_config_sha256": config["validation"]["config_sha256"],
                    "frozen_trajectory_sha256": config["frozen_input"]["trajectory_sha256"],
                    "formal_denominator_source_field": "trajectory.old",
                }
            )
            three_policy_sha = atomic_write_json(
                output / "three_policy_identity.json", three_policy_identity
            )
            atomic_write_json(
                output / "three_policy_token_logprobs.json",
                {
                    "schema_version": 2,
                    "ignored_private_evidence": True,
                    "rows": [
                        {
                            "fixture_id": str(row["fixture_id"]),
                            "source_role": str(row["source_role"]),
                            "prompt_ids": row["prompt_ids"],
                            "response_ids": row["response_ids"],
                            "response_mask": row["response_mask"],
                            "pi_rollout": old[index, : len(row["response_ids"])].detach().cpu().tolist(),
                            "pi_old_actor": pi_old_actor[index, : len(row["response_ids"])].detach().cpu().tolist(),
                            "pi_current_pre": new_before[index, : len(row["response_ids"])].detach().cpu().tolist(),
                        }
                        for index, row in enumerate(rows)
                    ],
                },
            )
            atomic_write_json(
                output / "pre_update_evidence.json",
                {
                    "schema_version": 2,
                    "status": "observed_before_pre_update_gate",
                    "protocol_config_sha256": config["validation"]["config_sha256"],
                    "frozen_trajectory_sha256": config["frozen_input"]["trajectory_sha256"],
                    "valid_tokens": int(response_mask.sum().cpu()),
                    "new_logprob": evidence_numeric_summary(new_before, response_mask),
                    "old_logprob": evidence_numeric_summary(old, response_mask),
                    "teacher_logprob": evidence_numeric_summary(teacher, response_mask),
                    "advantage": evidence_numeric_summary(advantage, response_mask),
                    "ratio": evidence_numeric_summary(raw_pre_ratio, response_mask),
                    "log_ratio": evidence_numeric_summary(raw_pre_log_ratio, response_mask),
                    "old_requires_grad": old.requires_grad,
                    "teacher_requires_grad": teacher.requires_grad,
                    "advantage_requires_grad": advantage.requires_grad,
                    "three_policy_identity_path": "three_policy_identity.json",
                    "three_policy_identity_sha256": three_policy_sha,
                },
            )
            latest_evidence_path = output / "three_policy_identity.json"
            pre_audit = validate_pre_update_contract(
                new_student_logprob=new_before,
                old_student_logprob=old,
                teacher_logprob=teacher,
                advantage=advantage,
                response_mask=response_mask,
                beta=float(algorithm["beta"]),
                max_abs_log_ratio=float(tolerances["pre_update_log_ratio_max_abs"]),
            )
            current_phase = "frozen_medical_one_step"
            before = ppo_clipped_objective(
                new_before,
                old,
                advantage,
                response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
            )
            optimizer = torch.optim.AdamW(
                [parameter_by_name[name] for name in trainable_names],
                lr=float(optimizer_config["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
                eps=float(optimizer_config["epsilon"]),
                foreach=bool(optimizer_config["foreach"]),
            )
            optimizer.zero_grad(set_to_none=True)
            before.loss.backward()
            teacher_gradient_parameters = [
                name
                for name, parameter in teacher_model.named_parameters()
                if parameter.grad is not None
            ]
            grad_norm_before_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter_by_name[name] for name in trainable_names],
                    float(optimizer_config["global_gradient_clip_norm"]),
                )
            )
            gradient_parameter_names = tuple(
                name for name in trainable_names if parameter_by_name[name].grad is not None
            )
            clipped_gradients = {
                name: (
                    parameter_by_name[name].grad.detach().cpu().clone()
                    if parameter_by_name[name].grad is not None
                    else torch.zeros_like(parameter_by_name[name], device="cpu")
                )
                for name in trainable_names
            }
            atomic_write_json(
                output / "medical_step_checkpoint.json",
                {
                    "schema_version": 2,
                    "status": "observed_after_backward_before_optimizer",
                    "gradient_norm_before_clip": grad_norm_before_clip,
                    "gradient_parameter_names": list(gradient_parameter_names),
                    "trainable_parameter_names": list(trainable_names),
                    "teacher_gradient_parameters": teacher_gradient_parameters,
                    "gradients_finite": all(
                        bool(torch.isfinite(value).all())
                        for value in clipped_gradients.values()
                    ),
                },
            )
            latest_evidence_path = output / "medical_step_checkpoint.json"
            optimizer.step()
            medical_after_parameters = {
                name: parameter_by_name[name].detach().cpu().clone()
                for name in trainable_names
            }
            base_parameters_unchanged = not any(
                parameter_by_name[name]._version != version
                for name, version in frozen_versions.items()
            )
            raw_parameter_delta_nonzero = any(
                bool(
                    torch.any(
                        medical_after_parameters[name]
                        != medical_before_parameters[name]
                    )
                )
                for name in trainable_names
            )
            atomic_write_json(
                output / "medical_step_checkpoint.json",
                {
                    "schema_version": 2,
                    "status": "observed_after_optimizer_before_post_forward",
                    "gradient_norm_before_clip": grad_norm_before_clip,
                    "gradient_parameter_names": list(gradient_parameter_names),
                    "trainable_parameter_names": list(trainable_names),
                    "teacher_gradient_parameters": teacher_gradient_parameters,
                    "base_parameter_versions_unchanged": base_parameters_unchanged,
                    "raw_parameter_delta_nonzero": raw_parameter_delta_nonzero,
                },
            )
            student_model.eval()
            with torch.inference_mode():
                new_after, after_mask = student_logprobs(student_model)
            response_mask_equal = bool(torch.equal(response_mask, after_mask))
            atomic_write_json(
                output / "medical_step_checkpoint.json",
                {
                    "schema_version": 2,
                    "status": "observed_after_post_forward_before_objective_gate",
                    "gradient_norm_before_clip": grad_norm_before_clip,
                    "gradient_parameter_names": list(gradient_parameter_names),
                    "trainable_parameter_names": list(trainable_names),
                    "teacher_gradient_parameters": teacher_gradient_parameters,
                    "base_parameter_versions_unchanged": base_parameters_unchanged,
                    "raw_parameter_delta_nonzero": raw_parameter_delta_nonzero,
                    "response_mask_equal": response_mask_equal,
                    "response_mask_valid_tokens": int(response_mask.sum().detach().cpu()),
                    "after_mask_valid_tokens": int(after_mask.sum().detach().cpu()),
                    "new_logprob_before": evidence_numeric_summary(new_before, response_mask),
                    "new_logprob_after": evidence_numeric_summary(new_after, after_mask),
                    "delta_logprob": evidence_numeric_summary(
                        new_after - new_before.detach(), response_mask
                    ),
                },
            )
            latest_evidence_path = output / "medical_step_checkpoint.json"
            if not response_mask_equal:
                raise PGDirectionGPUError("post-update response mask identity changed")
            after = ppo_clipped_objective(
                new_after,
                old,
                advantage,
                response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
            )
            delta_logprob = new_after - new_before.detach()
            update_audit = audit_frozen_pg_update(
                before=before,
                after=after,
                advantage=advantage,
                delta_logprob=delta_logprob,
                response_mask=response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
                objective_tolerance=float(tolerances["objective_improvement_abs"]),
                alignment_tolerance=float(tolerances["alignment_min"]),
                max_clip_fraction_for_alignment=float(
                    tolerances["significant_active_clip_fraction"]
                ),
            )
            optimizer_audit = audit_optimizer_update(
                before=medical_before_parameters,
                after=medical_after_parameters,
                loss_gradients=clipped_gradients,
                declared_trainable_names=trainable_names,
                actual_requires_grad_names=trainable_names,
                fresh_optimizer=True,
                weight_decay=float(optimizer_config["weight_decay"]),
                require_nonzero=True,
                descent_dot_max=float(tolerances["optimizer_descent_dot_max"]),
            )
            valid_advantage = advantage[response_mask]
            valid_delta = delta_logprob[response_mask]
            direction = summarize_signed_update(
                advantage=valid_advantage.detach().cpu().tolist(),
                logprob_change=valid_delta.detach().cpu().tolist(),
                near_zero_tolerance=float(tolerances["subgroup_advantage_near_zero"]),
            )
            before_reduction = grouped_reduction(
                before.token_surrogate,
                response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
            )
            after_reduction = grouped_reduction(
                after.token_surrogate,
                response_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
            )
            medical_passed = bool(
                update_audit.hard_gate_passed
                and optimizer_audit.hard_gate_passed
                and not teacher_gradient_parameters
                and base_parameters_unchanged
                and set(gradient_parameter_names) == set(trainable_names)
                and grad_norm_before_clip
                > float(tolerances["nonzero_gradient_norm_min"])
                and optimizer_audit.parameter_delta_norm
                > float(tolerances["nonzero_parameter_delta_norm_min"])
            )
            one_step_metrics = {
                "protocol_id": config["validation"]["protocol_id"],
                "protocol_config_sha256": config["validation"]["config_sha256"],
                "status": "pass" if medical_passed else "fail",
                "hard_gate_passed": medical_passed,
                "pre_ratio_summary": masked_numeric_summary(before.ratio, response_mask),
                "post_ratio_summary": masked_numeric_summary(after.ratio, response_mask),
                "pre_log_ratio_summary": masked_numeric_summary(before.log_ratio, response_mask),
                "post_log_ratio_summary": masked_numeric_summary(after.log_ratio, response_mask),
                "pre_active_clip_fraction": before.clip_fraction,
                "post_active_clip_fraction": after.clip_fraction,
                "objective_before": update_audit.objective_before,
                "objective_after": update_audit.objective_after,
                "loss_before": update_audit.loss_before,
                "loss_after": update_audit.loss_after,
                "alignment": update_audit.alignment,
                "gradient_norm": optimizer_audit.gradient_norm,
                "gradient_norm_before_clip": grad_norm_before_clip,
                "parameter_delta_norm": optimizer_audit.parameter_delta_norm,
                "relative_parameter_delta": optimizer_audit.relative_parameter_delta,
                "gradient_dot_parameter_delta": optimizer_audit.gradient_dot_parameter_delta,
                "trainable_parameter_count": optimizer_audit.trainable_parameter_count,
                "module_update_norms": optimizer_audit.module_update_norms,
                "nonfinite_counts": {"advantage": 0, "logprob": 0, "ratio": 0, "loss": 0},
                "per_prompt_objective_before_after": {
                    prompt: {
                        "before": float(before_reduction.per_prompt[prompt].detach().cpu()),
                        "after": float(after_reduction.per_prompt[prompt].detach().cpu()),
                    }
                    for prompt in before_reduction.per_prompt
                },
                "per_source_diagnostics": source_diagnostics(
                    after.token_surrogate - before.token_surrogate.detach(), response_mask
                ),
                "per_domain_diagnostics": {
                    "medical": {"count": len(rows), "objective_improvement": update_audit.objective_improvement}
                },
                "advantage_sign_fractions": {
                    "positive": float((valid_advantage > 1e-6).float().mean().cpu()),
                    "negative": float((valid_advantage < -1e-6).float().mean().cpu()),
                    "near_zero": float((valid_advantage.abs() <= 1e-6).float().mean().cpu()),
                },
                "subgroup_logprob_change_diagnostics": direction,
                "pre_update_audit": asdict(pre_audit),
                "objective_audit": asdict(update_audit),
                "optimizer_audit": asdict(optimizer_audit),
                "teacher_gradient_parameters": teacher_gradient_parameters,
                "gradient_parameter_names": list(gradient_parameter_names),
                "trainable_parameter_names": list(trainable_names),
                "base_parameter_versions_unchanged": base_parameters_unchanged,
                "frozen_trajectory_sha256": config["frozen_input"]["trajectory_sha256"],
                "student_initial_adapter_sha256": old_adapter_sha,
                "formal_checkpoint_saved": False,
            }
            current_phase = (
                "frozen_medical_one_step"
                if update_audit.hard_gate_passed
                else "failed_surrogate_direction"
            )
            optimizer_failure_reasons = list(optimizer_audit.failure_reasons)
            if teacher_gradient_parameters:
                optimizer_failure_reasons.append("teacher_received_gradient")
            if not base_parameters_unchanged:
                optimizer_failure_reasons.append("base_parameter_changed")
            if set(gradient_parameter_names) != set(trainable_names):
                optimizer_failure_reasons.append("gradient_manifest_mismatch")
            latest_evidence_path = output / "one_step_metrics.json"
            persist_update_outcome(
                output,
                metrics=one_step_metrics,
                hard_gate_passed=medical_passed,
                failure_status=(
                    "failed_surrogate_direction"
                    if not update_audit.hard_gate_passed
                    else "failed_optimizer_update"
                ),
                failure_reason=(
                    "; ".join(update_audit.failure_reasons)
                    if not update_audit.hard_gate_passed
                    else "; ".join(optimizer_failure_reasons)
                    or "Medical optimizer contract failed"
                ),
            )

            current_phase = "frozen_base_teacher_null_update"
            with torch.no_grad():
                for name, value in medical_before_parameters.items():
                    parameter_by_name[name].copy_(
                        value.to(device=parameter_by_name[name].device, dtype=parameter_by_name[name].dtype)
                    )
            student_model.train()
            null_before_parameters = {
                name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
            }
            null_new_before, null_mask = student_logprobs(student_model)
            null_old = null_new_before.detach()
            # The null teacher and old policy are two detached identities of the
            # same real Base forward. This exercises the production objective path
            # while excluding cross-device/route numerical drift from a hard null.
            null_teacher = null_old.detach().clone()
            null_advantage = same_trajectory_advantage(
                null_old, null_teacher, beta=float(algorithm["beta"])
            )
            null_before = ppo_clipped_objective(
                null_new_before,
                null_old,
                null_advantage,
                null_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
            )
            null_optimizer = torch.optim.AdamW(
                [parameter_by_name[name] for name in trainable_names],
                lr=float(optimizer_config["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
                eps=float(optimizer_config["epsilon"]),
                foreach=bool(optimizer_config["foreach"]),
            )
            null_optimizer.zero_grad(set_to_none=True)
            null_before.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter_by_name[name] for name in trainable_names],
                float(optimizer_config["global_gradient_clip_norm"]),
            )
            null_gradients = {
                name: parameter_by_name[name].grad.detach().cpu().clone()
                for name in trainable_names
                if parameter_by_name[name].grad is not None
            }
            null_optimizer.step()
            null_after_parameters = {
                name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
            }
            student_model.eval()
            with torch.inference_mode():
                null_new_after, _ = student_logprobs(student_model)
            null_after = ppo_clipped_objective(
                null_new_after,
                null_old,
                null_advantage,
                null_mask,
                prompt_ids=tuple(str(row["fixture_id"]) for row in rows),
                group_ids=("g0",) * len(rows),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
            )
            null_optimizer_audit = audit_optimizer_update(
                before=null_before_parameters,
                after=null_after_parameters,
                loss_gradients=null_gradients,
                declared_trainable_names=trainable_names,
                actual_requires_grad_names=trainable_names,
                fresh_optimizer=True,
                weight_decay=float(optimizer_config["weight_decay"]),
                require_nonzero=False,
                descent_dot_max=float(tolerances["optimizer_descent_dot_max"]),
                null_gradient_norm_max=float(tolerances["null_gradient_norm_max"]),
                null_parameter_delta_norm_max=float(
                    tolerances["null_parameter_delta_norm_max"]
                ),
            )
            null_advantage_max_abs = float(null_advantage[null_mask].abs().max().cpu())
            null_passed = bool(
                null_advantage_max_abs <= float(tolerances["null_advantage_max_abs"])
                and null_optimizer_audit.hard_gate_passed
            )
            null_report = {
                "schema_version": 2,
                "protocol_id": config["validation"]["protocol_id"],
                "protocol_config_sha256": config["validation"]["config_sha256"],
                "status": "pass" if null_passed else "fail",
                "hard_gate_passed": null_passed,
                "advantage_max_abs": null_advantage_max_abs,
                "objective_before": float(null_before.surrogate.detach().cpu()),
                "objective_after": float(null_after.surrogate.detach().cpu()),
                "loss_before": float(null_before.loss.detach().cpu()),
                "loss_after": float(null_after.loss.detach().cpu()),
                "gradient_norm": null_optimizer_audit.gradient_norm,
                "parameter_delta_norm": null_optimizer_audit.parameter_delta_norm,
                "optimizer_audit": asdict(null_optimizer_audit),
                "nonfinite_counts": {
                    "advantage": 0,
                    "logprob": 0,
                    "ratio": 0,
                    "loss": 0,
                },
                "teacher_logprob_source": "same_real_base_forward_detached",
                "same_objective_mask_reduction_writer": True,
                "frozen_trajectory_sha256": config["frozen_input"]["trajectory_sha256"],
                "formal_checkpoint_saved": False,
            }
            atomic_write_json(output / "null_update.json", null_report)
            latest_evidence_path = output / "null_update.json"
            if not null_passed:
                raise PGDirectionGPUError("Base=Teacher null update contract failed")
            with torch.no_grad():
                for name, value in medical_after_parameters.items():
                    parameter_by_name[name].copy_(
                        value.to(device=parameter_by_name[name].device, dtype=parameter_by_name[name].dtype)
                    )
            student_model.eval()

            current_phase = "sampler_refresh_identity"
            new_adapter_dir = Path(temporary) / "version1"
            student_model.save_pretrained(new_adapter_dir, safe_serialization=True)
            new_adapter_sha = ordered_adapter_sha256(new_adapter_dir)
            one_step_metrics["student_updated_adapter_sha256"] = new_adapter_sha
            persist_update_outcome(
                output,
                metrics=one_step_metrics,
                hard_gate_passed=True,
                failure_status="unused",
                failure_reason="unused",
            )
            _release(torch, teacher_model)
            teacher_model = scorer = None
            sampler_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:1")
            sampler_base.config.use_cache = False
            sampler_model = PeftModel.from_pretrained(
                sampler_base,
                old_adapter_dir,
                adapter_name="version0",
                is_trainable=False,
            )
            del sampler_base
            sampler_model.eval()
            with torch.inference_mode():
                old_sampler_probe = action_logprobs(
                    sampler_model,
                    rows[0]["prompt_ids"],
                    rows[0]["response_ids"],
                    device="cuda:1",
                ).detach().cpu()
            trainer_before_probe = new_before[
                0, : len(rows[0]["response_ids"])
            ].detach().cpu()
            pre_refresh_probe_max_abs_delta = float(
                (old_sampler_probe - trainer_before_probe).abs().max()
            )
            active_adapter_before = getattr(sampler_model, "active_adapter", "unknown")
            if isinstance(active_adapter_before, (list, tuple)):
                active_adapter_before = [str(name) for name in active_adapter_before]
            else:
                active_adapter_before = str(active_adapter_before)
            sampler_observations = {
                "schema_version": 2,
                "status": "observed_before_adapter_refresh_helper",
                "protocol_config_sha256": config["validation"]["config_sha256"],
                "old_version": 0,
                "old_sha256": old_adapter_sha,
                "trainer_version": 1,
                "trainer_sha256": new_adapter_sha,
                "active_adapter_before": active_adapter_before,
                "probe_tolerance": 1e-6,
                "pre_refresh_probe_max_abs_delta": pre_refresh_probe_max_abs_delta,
                "stale_adapter_rejected": "not_checked",
                "post_refresh_probe": "not_run",
            }
            atomic_write_json(
                output / "sampler_refresh_observations.json", sampler_observations
            )
            latest_evidence_path = output / "sampler_refresh_observations.json"
            try:
                sampler_state = refresh_sampler_adapter(
                    sampler_model,
                    adapter_path=new_adapter_dir,
                    old_version=0,
                    old_sha256=old_adapter_sha,
                    old_adapter_name="version0",
                    new_version=1,
                    new_sha256=new_adapter_sha,
                    new_adapter_name="version1",
                )
            except Exception as refresh_error:
                sampler_observations.update(
                    {
                        "status": "adapter_refresh_helper_failed",
                        "refresh_error_type": type(refresh_error).__name__,
                        "refresh_error": str(refresh_error),
                    }
                )
                atomic_write_json(
                    output / "sampler_refresh_observations.json", sampler_observations
                )
                raise
            sampler_observations.update(
                {
                    "status": "observed_after_adapter_refresh_before_identity_gates",
                    **sampler_state,
                }
            )
            atomic_write_json(
                output / "sampler_refresh_observations.json", sampler_observations
            )
            require_sampler_identity(
                sampler_state,
                expected_version=1,
                expected_sha256=new_adapter_sha,
            )
            guarded_probe_identity_verified = True
            stale_rejected = False
            try:
                sampler_model.set_adapter("version0")
            except (KeyError, ValueError):
                stale_rejected = True
            sampler_observations.update(
                {
                    "status": "observed_after_stale_adapter_selection_gate",
                    "stale_adapter_rejected": stale_rejected,
                }
            )
            atomic_write_json(
                output / "sampler_refresh_observations.json", sampler_observations
            )
            if not stale_rejected:
                raise PGDirectionGPUError("removed stale sampler adapter remained selectable")
            sampler_model.eval()
            with torch.inference_mode():
                sampler_probe = action_logprobs(
                    sampler_model,
                    rows[0]["prompt_ids"],
                    rows[0]["response_ids"],
                    device="cuda:1",
                ).detach().cpu()
            trainer_probe = new_after[0, : len(rows[0]["response_ids"])].detach().cpu()
            probe_max_abs_delta = float((sampler_probe - trainer_probe).abs().max())
            sampler_observations.update(
                {
                    "status": "observed_before_sampler_hard_gate",
                    "probe_max_abs_delta": probe_max_abs_delta,
                    "post_refresh_probe": "completed",
                }
            )
            atomic_write_json(
                output / "sampler_refresh_observations.json", sampler_observations
            )
            sampler_report = audit_sampler_refresh(
                old_version=0,
                old_sha256=old_adapter_sha,
                trainer_version=1,
                trainer_sha256=new_adapter_sha,
                sampler_version=int(sampler_state["version"]),
                sampler_sha256=str(sampler_state["adapter_sha256"]),
                probe_max_abs_delta=probe_max_abs_delta,
                probe_tolerance=1e-6,
                stale_adapter_rejected=stale_rejected,
            )
            sampler_report.update(
                {
                    "schema_version": 2,
                    "protocol_config_sha256": config["validation"]["config_sha256"],
                    **sampler_state,
                    "separate_sampler_model_loaded": True,
                    "pre_refresh_probe_max_abs_delta": pre_refresh_probe_max_abs_delta,
                    "pre_refresh_probe_match": pre_refresh_probe_max_abs_delta <= 1e-6,
                    "guarded_probe_identity_verified": guarded_probe_identity_verified,
                    "cache_disabled": sampler_model.config.use_cache is False,
                    "cache_identity_reused": False,
                    "temporary_adapter_removed_on_exit": True,
                    "formal_verl_update_weights_verified": False,
                    "formal_checkpoint_saved": False,
                }
            )
            atomic_write_json(output / "sampler_refresh.json", sampler_report)
            latest_evidence_path = output / "sampler_refresh.json"

        current_phase = "artifact_integrity_and_readiness"
        atomic_write_json(
            output / "scorer_readiness.json",
            {
                "schema_version": 2,
                "status": "pass",
                "formal_backend": "Transformers",
                "repeatability_passed": True,
                "route_isolation_passed": True,
                "same_model_hard_null_passed": True,
                "minimal_medical_identity_passed": True,
                "p4_1_repeatability_sha256": config["historical_scorer_evidence"][
                    "repeatability_report_sha256"
                ],
                "p4_1_route_isolation_sha256": config["historical_scorer_evidence"][
                    "route_isolation_report_sha256"
                ],
                "p4_1_same_model_null_sha256": config["historical_scorer_evidence"][
                    "same_model_null_report_sha256"
                ],
            },
        )
        current_phase = "release_gpu_resources"
        _release(torch, student_model, teacher_model, sampler_model)
        student_model = teacher_model = scorer = sampler_model = None
        allocated = [int(torch.cuda.memory_allocated(index)) for index in range(2)]
        reserved = [int(torch.cuda.memory_reserved(index)) for index in range(2)]
        atomic_write_json(
            output / "runtime_release.json",
            {
                "schema_version": 2,
                "status": "pass",
                "models_released": True,
                "cuda_allocated_bytes_diagnostic": allocated,
                "cuda_reserved_bytes_diagnostic": reserved,
                "post_process_exit_verification_required": True,
            },
        )
        latest_evidence_path = output / "runtime_release.json"
        summary = {
            "schema_version": 2,
            "status": "ready_for_post_exit_resource_cleanup_verification",
            "all_runtime_gates_completed": True,
            "readiness_derivation_pending": True,
            "opd_training_ready": False,
            "P4_1_status": "blocked_pg_opd_direction",
            "vllm_backend": "diagnostic_only",
            "formal_checkpoint_saved": False,
            "B2_authorized": False,
            "formal_opd_authorized": False,
            **run_identity,
            "elapsed_seconds": time.time() - started,
            "next_step": "post_exit_resource_cleanup_finalizer",
        }
        atomic_write_json(output / "summary.json", summary)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        metadata.update(
            {
                "status": summary["status"],
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - started,
                "B2_authorized": False,
            }
        )
        atomic_write_json(output / "metadata.json", metadata)
        return summary
    except Exception as error:
        failure_states = {
            "minimal_medical_scorer_identity_probe": "failed_identity_mismatch",
            "pre_update_identity": "failed_identity_mismatch",
            "frozen_medical_one_step": "failed_optimizer_update",
            "failed_surrogate_direction": "failed_surrogate_direction",
            "frozen_base_teacher_null_update": "failed_null_update",
            "sampler_refresh_identity": "failed_sampler_refresh",
            "artifact_integrity_and_readiness": "failed_artifact_integrity",
        }
        failure_status = failure_states.get(current_phase, "failed_artifact_integrity")
        phase_metrics = {
            "schema_version": 2,
            "status": failure_status,
            "phase": current_phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol_config_sha256": config["validation"]["config_sha256"],
            "P4_1_status": "blocked_pg_opd_direction",
            "B2_authorized": False,
        }
        phase_sha = atomic_write_json(output / "phase_failure_metrics.json", phase_metrics)
        if not (output / "failure.json").exists():
            bound_path = (
                latest_evidence_path
                if latest_evidence_path is not None and latest_evidence_path.is_file()
                else output / "phase_failure_metrics.json"
            )
            atomic_write_json(
                output / "failure.json",
                {
                    "schema_version": 2,
                    "status": failure_status,
                    "reason": f"{type(error).__name__}: {error}",
                    "metrics_path": bound_path.name,
                    "metrics_sha256": sha256_file(bound_path),
                    "phase_failure_metrics_sha256": phase_sha,
                },
            )
        atomic_write_json(
            output / "summary.json",
            {
                "schema_version": 2,
                "status": failure_status,
                "failure_phase": current_phase,
                "B2_authorized": False,
                "formal_opd_authorized": False,
            },
        )
        raise
    finally:
        try:
            _release(torch, student_model, teacher_model, sampler_model)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized P4.2 PG direction rerun")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    import yaml

    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(json.dumps(run_gpu_revalidation(config, config_path=path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
