"""GPU-only P4.8d session using the frozen production three-policy kernel.

Importing this module does not load a model.  Construction is reachable only
after the package semantic gate and GPU authorization.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from src.opd.production_b2_memory_execution_v1 import (
    MEMORY_EXECUTION_CONTRACT,
    MemoryExecutionV1Error,
    MemoryTelemetryWriterV1,
    assert_prompt_equal_reduction_batch,
    build_target_chunks,
    configure_training_student,
    scaled_prompt_chunk_loss,
    target_logprobs_from_selected_logits,
    validate_memory_execution_contract,
)
from src.opd.production_b2_memory_execution_gpu_v2 import (
    backward_selected_hidden_once,
)
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
    ProductionTwoStepSessionV6,
    _atomic_json,
)


HIDDEN_GRADIENT_LORA_TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)


def _validate_session_memory_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select the exact calibration or P5 formal contract by discriminator."""

    if "formal_b2" in config:
        from src.opd.production_b2_formal_v1 import (
            validate_formal_memory_execution_contract,
        )

        return validate_formal_memory_execution_contract(
            config.get("memory_execution", {})
        )
    return validate_memory_execution_contract(config.get("memory_execution", {}))


def assert_hidden_gradient_trainable_scope(
    model: Any,
    *,
    expected_trainable_tensor_count: int = 504,
    allowed_target_modules: tuple[str, ...] = HIDDEN_GRADIENT_LORA_TARGET_MODULES,
) -> dict[str, Any]:
    """Fail closed unless every trainable tensor is a backbone LoRA tensor."""

    get_base = getattr(model, "get_base_model", None)
    causal_lm = get_base() if callable(get_base) else model
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    get_embeddings = getattr(causal_lm, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    if not all(
        module is not None and callable(getattr(module, "parameters", None))
        for module in (backbone, lm_head, embeddings)
    ):
        raise ProductionTwoStepQualificationV6Error(
            "hidden-gradient trainable scope cannot resolve backbone/head/embedding"
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    lm_head_ids = {id(parameter) for parameter in lm_head.parameters()}
    embedding_ids = {id(parameter) for parameter in embeddings.parameters()}
    trainable_ids = {id(parameter) for _, parameter in trainable}
    after_hidden = trainable_ids - backbone_ids
    lm_head_trainable = trainable_ids & lm_head_ids
    embedding_trainable = trainable_ids & embedding_ids
    allowed = tuple(sorted(set(allowed_target_modules)))
    target_counts = {target: 0 for target in allowed}
    invalid_names: list[str] = []
    for name, _parameter in trainable:
        lowered = name.lower()
        matches = [target for target in allowed if f".{target}." in f".{lowered}."]
        if "lora_" not in lowered or len(matches) != 1:
            invalid_names.append(name)
            continue
        target_counts[matches[0]] += 1
    actual_targets = sorted(
        target for target, count in target_counts.items() if count > 0
    )
    passed = (
        len(trainable) == int(expected_trainable_tensor_count)
        and not after_hidden
        and not lm_head_trainable
        and not embedding_trainable
        and not invalid_names
        and actual_targets == list(allowed)
    )
    if not passed:
        raise ProductionTwoStepQualificationV6Error(
            "hidden-gradient trainable scope differs from the frozen backbone LoRA contract"
        )
    return {
        "schema_version": 1,
        "artifact_kind": "b2_hidden_gradient_trainable_scope_v1",
        "passed": True,
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in trainable
        ),
        "trainable_params_after_hidden": len(after_hidden),
        "lm_head_trainable_tensor_count": len(lm_head_trainable),
        "embedding_trainable_tensor_count": len(embedding_trainable),
        "allowed_target_modules": list(allowed),
        "actual_target_modules": actual_targets,
        "target_module_tensor_counts": target_counts,
    }


class MemoryBalancedProductionTwoStepSessionV1(ProductionTwoStepSessionV6):
    """Same production math with position-chunked q and explicit lifetimes."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        try:
            self.memory_contract = _validate_session_memory_contract(config)
        except (MemoryExecutionV1Error, RuntimeError) as error:
            raise ProductionTwoStepQualificationV6Error(str(error)) from error
        output = Path(str(config["run"]["output_dir"]))
        self._memory_phase_starts: dict[str, tuple[dict[str, Any], float]] = {}
        self._memory_score_phases: list[str] = []
        self._memory_backward_prompt_count = 0
        self._memory_optimizer_count = 0
        self._memory_refresh_count = 0
        self._memory_lm_head_chunk_count = 0
        self._memory_backbone_forward_count = 0
        self._memory_backbone_backward_count = 0
        self._memory_retain_graph_count = 0
        self._memory_export_count = 0
        self._memory_fresh_identity_verifier_count = 0
        self._memory_chunk_loss_total = 0.0
        self._memory_differential_q_rows: list[Any] = []
        self._minimum_free_bytes_by_gpu = [2**63 - 1, 2**63 - 1]
        self._last_step_end_snapshot: dict[str, Any] | None = None
        self._memory_writer = MemoryTelemetryWriterV1(
            output / "memory_telemetry",
            run_id=str(config["run"]["run_id"]),
            gpu_snapshot_provider=self._gpu_memory_snapshot,
        )
        super().__init__(config, **kwargs)
        configure_training_student(self.student_model)
        enable_inputs = getattr(self.student_model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
        self.student_model.eval()
        self._hidden_gradient_trainable_scope = (
            assert_hidden_gradient_trainable_scope(self.student_model)
        )
        _atomic_json(
            self.output / "hidden_gradient_trainable_scope_fresh_v0.json",
            {
                **self._hidden_gradient_trainable_scope,
                "run_id": self.config["run"]["run_id"],
                "logical_version": 0,
            },
        )
        # Preserve the pre-canary v0 identity before the throwaway optimizer
        # step mutates the in-memory Student.  Recomputing "initial" identity
        # after the canary would incorrectly describe v1 as v0.
        self._initial_adapter_sha256 = self.initial_calibration_identity()[
            "adapter_sha256"
        ]
        self._initial_registry_count = self._registry_count()
        self._initial_model_count = self._model_count()

    def _gpu_memory_snapshot(self) -> list[dict[str, Any]]:
        torch = self.torch
        process_memory: dict[int, int | None] = {0: None, 1: None}
        try:
            rows = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.splitlines()
            # UUID-to-index lookup is intentionally omitted when unavailable;
            # torch allocator values remain authoritative and process memory is
            # a nullable diagnostic field.
            if len(rows) == 2:
                for index, row in enumerate(rows):
                    process_memory[index] = int(row.rsplit(",", 1)[-1].strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        result: list[dict[str, Any]] = []
        for device in (0, 1):
            stats = torch.cuda.memory_stats(device)
            free, total = torch.cuda.mem_get_info(device)
            current_reserved = int(torch.cuda.memory_reserved(device))
            max_reserved = int(torch.cuda.max_memory_reserved(device))
            estimated_peak_free = max(
                0, int(free) - max(0, max_reserved - current_reserved)
            )
            result.append(
                {
                    "device": device,
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "memory_reserved_bytes": current_reserved,
                    "max_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "max_memory_reserved_bytes": max_reserved,
                    "active_bytes": int(stats.get("active_bytes.all.current", 0)),
                    "inactive_split_bytes": int(
                        stats.get("inactive_split_bytes.all.current", 0)
                    ),
                    "non_releasable_bytes": int(
                        stats.get("reserved_bytes.all.current", 0)
                        - stats.get("active_bytes.all.current", 0)
                    ),
                    "free_bytes": int(free),
                    "estimated_peak_free_bytes": estimated_peak_free,
                    "total_bytes": int(total),
                    "process_memory_mib": process_memory[device],
                }
            )
            self._minimum_free_bytes_by_gpu[device] = min(
                self._minimum_free_bytes_by_gpu[device], estimated_peak_free
            )
        return result

    def _registry_count(self) -> int:
        model = getattr(self, "sampler_model", None)
        config = getattr(model, "peft_config", None)
        return len(config) if isinstance(config, Mapping) else 0

    def _model_count(self) -> int:
        return sum(
            getattr(self, name, None) is not None
            for name in ("student_model", "teacher_model", "sampler_model")
        )

    def _memory_phase_observer(
        self,
        marker: str,
        phase: str,
        *,
        step: int = 0,
        sequence_shape: list[int] | None = None,
        token_shape: list[int] | None = None,
    ) -> None:
        if marker == "before":
            durable = self._memory_writer.mark_before(
                phase=phase,
                step=step,
                sequence_shape=sequence_shape,
                token_shape=token_shape,
                registry_count=self._registry_count(),
                model_count=self._model_count(),
            )
            self._memory_phase_starts[phase] = (durable, time.perf_counter())
            return
        if marker != "after" or phase not in self._memory_phase_starts:
            raise ProductionTwoStepQualificationV6Error(
                f"memory phase {phase} lacks a durable before marker"
            )
        durable, started = self._memory_phase_starts.pop(phase)
        self._memory_writer.mark_after(
            durable, elapsed_seconds=time.perf_counter() - started
        )

    def _pad(self, values: list[Any], *, device: str = "cpu") -> tuple[Any, Any]:
        del device
        torch = self.torch
        result = torch.nn.utils.rnn.pad_sequence(
            [value.reshape(-1).detach().to(device="cpu", dtype=torch.float32) for value in values],
            batch_first=True,
            padding_value=0.0,
        )
        mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(values):
            mask[index, : value.numel()] = True
        return result, mask

    def _causal_backbone_and_lm_head(self, model: Any) -> tuple[Any, Any]:
        get_base = getattr(model, "get_base_model", None)
        causal_lm = get_base() if callable(get_base) else model
        backbone = getattr(causal_lm, "model", None)
        lm_head = getattr(causal_lm, "lm_head", None)
        if not callable(backbone) or not callable(lm_head):
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d Student/Teacher lacks the frozen causal backbone/LM-head route"
            )
        return backbone, lm_head

    def _selected_chunk_logprobs(
        self,
        model: Any,
        row: Mapping[str, Any],
        *,
        device: str,
        phase: str,
    ) -> Any:
        torch = self.torch
        prompt = [int(value) for value in row["prompt_ids"]]
        response = [int(value) for value in row["response_ids"]]
        ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        positions = torch.arange(
            len(prompt) - 1,
            len(prompt) - 1 + len(response),
            dtype=torch.long,
            device=device,
        )
        backbone, lm_head = self._causal_backbone_and_lm_head(model)
        backbone_phase = f"{phase}_backbone"
        self._memory_phase_observer(
            "before",
            backbone_phase,
            step=self.current_sampler_version + 1,
            sequence_shape=list(ids.shape),
            token_shape=[1, len(response)],
        )
        backbone_result = backbone(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        selected_hidden = backbone_result.last_hidden_state.index_select(1, positions)
        del backbone_result
        self._memory_phase_observer("after", backbone_phase)
        values: list[Any] = []
        for chunk_index, (start, end) in enumerate(
            build_target_chunks(
                len(response),
                chunk_size=int(self.memory_contract["target_logit_chunk_size"]),
            )
        ):
            marker_phase = f"{phase}_target_chunk_{chunk_index + 1}"
            self._memory_phase_observer(
                "before",
                marker_phase,
                step=self.current_sampler_version + 1,
                sequence_shape=list(ids.shape),
                token_shape=[1, end - start],
            )
            logits = lm_head(selected_hidden[:, start:end, :])
            targets = torch.tensor(
                response[start:end], dtype=torch.long, device=device
            ).view(1, -1)
            selected = target_logprobs_from_selected_logits(logits, targets)
            values.append(selected.detach().float().cpu().reshape(-1))
            del selected, targets, logits
            self._memory_phase_observer("after", marker_phase)
        del selected_hidden, positions, attention, ids
        return torch.cat(values, dim=0)

    def _score_rows(
        self,
        model: Any,
        rows: list[Mapping[str, Any]],
        *,
        device: str,
        inference: bool,
    ) -> tuple[Any, Any]:
        if not inference:
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d training q must use immediate-backward target chunks"
            )
        phase = self._memory_score_phases.pop(0) if self._memory_score_phases else "identity_score"
        with self.torch.inference_mode():
            values = [
                self._selected_chunk_logprobs(
                    model,
                    row,
                    device=device,
                    phase=f"{phase}_row_{index + 1}",
                )
                for index, row in enumerate(rows)
            ]
        return self._pad(values)

    def _generate_rows(self, *args: Any, **kwargs: Any) -> Any:
        self._memory_phase_observer(
            "before",
            "rollout_generation",
            step=self.current_sampler_version + 1,
        )
        try:
            with self.torch.inference_mode():
                return super()._generate_rows(*args, **kwargs)
        finally:
            gc.collect()
            self.torch.cuda.empty_cache()
            self._memory_phase_observer("after", "rollout_generation")

    def run_corrected_step(
        self, step_index: int, rollout: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._memory_score_phases = [
            "p_old",
            "student_current_pre",
            "teacher_same_token",
            "student_current_after",
        ]
        self._memory_backward_prompt_count = 0
        self._memory_optimizer_count = 0
        self._memory_refresh_count = 0
        self._memory_lm_head_chunk_count = 0
        self._memory_backbone_forward_count = 0
        self._memory_backbone_backward_count = 0
        self._memory_retain_graph_count = 0
        self._memory_export_count = 0
        self._memory_fresh_identity_verifier_count = 0
        self._memory_chunk_loss_total = 0.0
        self._memory_differential_q_rows = []
        return super().run_corrected_step(step_index, rollout)

    def _validate_source_roles_for_backward(
        self, source_roles: tuple[str, ...]
    ) -> None:
        """Preserve the B2 source-shape gate while allowing versioned overrides."""

        if sorted(source_roles) != [
            "medical_opd_cmb",
            "medical_opd_cmb",
            "medical_opd_o1",
            "medical_opd_o1",
        ]:
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d source batch differs from two O1 plus two CMB"
            )

    def _backward_corrected_rows(
        self,
        *,
        rows: list[Mapping[str, Any]],
        bundle: Any,
        before_result: Any,
        provenance: Mapping[str, Any],
        prompt_ids: tuple[str, ...],
        group_ids: tuple[str, ...],
        source_roles: tuple[str, ...],
    ) -> None:
        del provenance
        if len(rows) != 4:
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d effective prompt batch differs from four"
            )
        assert_prompt_equal_reduction_batch(
            prompt_ids=prompt_ids, group_ids=group_ids
        )
        self._validate_source_roles_for_backward(source_roles)
        torch = self.torch
        self.student_model.train()
        for row_index, row in enumerate(rows):
            prompt = [int(value) for value in row["prompt_ids"]]
            response = [int(value) for value in row["response_ids"]]
            valid_count = len(response)
            ids = torch.tensor(
                [prompt + response], dtype=torch.long, device="cuda:0"
            )
            attention = torch.ones_like(ids)
            positions = torch.arange(
                len(prompt) - 1,
                len(prompt) - 1 + valid_count,
                dtype=torch.long,
                device="cuda:0",
            )
            backbone, lm_head = self._causal_backbone_and_lm_head(
                self.student_model
            )
            backbone_phase = f"student_q_microbatch_{row_index + 1}_backbone"
            self._memory_phase_observer(
                "before",
                backbone_phase,
                step=self.current_sampler_version + 1,
                sequence_shape=list(ids.shape),
                token_shape=[1, valid_count],
            )
            backbone_result = backbone(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            self._memory_backbone_forward_count += 1
            selected_hidden = backbone_result.last_hidden_state.index_select(
                1, positions
            )
            del backbone_result
            self._memory_phase_observer("after", backbone_phase)
            targets = torch.tensor(
                response, dtype=torch.long, device="cuda:0"
            ).view(1, -1)
            old = bundle.old_actor_logprob[row_index, :valid_count].to("cuda:0")
            advantage = before_result.advantage[row_index, :valid_count].to(
                "cuda:0"
            )
            correction = before_result.correction.truncated_weight[
                row_index, :valid_count
            ].to("cuda:0")

            def observe_chunk(marker: str, chunk_index: int, start: int, end: int) -> None:
                phase = (
                    f"student_q_microbatch_{row_index + 1}_"
                    f"target_chunk_{chunk_index + 1}"
                )
                self._memory_phase_observer(
                    marker,
                    phase,
                    step=self.current_sampler_version + 1,
                    sequence_shape=list(ids.shape),
                    token_shape=[1, end - start],
                )

            backward_audit = backward_selected_hidden_once(
                selected_hidden_states=selected_hidden,
                lm_head=lm_head,
                target_ids=targets,
                old_logprob=old,
                advantage=advantage,
                correction_weight=correction,
                prompt_valid_token_count=valid_count,
                effective_batch_size=4,
                clip_low=float(self.algorithm["clip_low"]),
                clip_high=float(self.algorithm["clip_high"]),
                chunk_size=int(self.memory_contract["target_logit_chunk_size"]),
                lm_head_layout_rows=int(ids.shape[1]),
                target_position_offset=len(prompt) - 1,
                chunk_observer=observe_chunk,
                capture_per_token=bool(
                    getattr(self, "_p4f_capture_differential", False)
                ),
            )
            self._memory_lm_head_chunk_count += int(
                backward_audit["lm_head_chunk_count"]
            )
            self._memory_backbone_backward_count += int(
                backward_audit["backbone_backward_calls"]
            )
            self._memory_retain_graph_count += int(
                backward_audit["retain_graph_calls"]
            )
            self._memory_chunk_loss_total += float(backward_audit["loss"])
            if "q_target_logprob" in backward_audit:
                self._memory_differential_q_rows.append(
                    backward_audit["q_target_logprob"]
                )
            bounded_prompt_hook = getattr(
                self, "_bound_prompt_gradient_contribution_v2", None
            )
            if callable(bounded_prompt_hook):
                bounded_prompt_hook(
                    row_index=row_index,
                    prompt_id=prompt_ids[row_index],
                    source_role=source_roles[row_index],
                    prompt_count=len(rows),
                )
            del (
                correction,
                advantage,
                old,
                targets,
                selected_hidden,
                positions,
                attention,
                ids,
            )
            self._memory_backward_prompt_count += 1
            gc.collect()
            torch.cuda.empty_cache()
        self.student_model.eval()

    def _before_optimizer_step(self, *, step_index: int) -> None:
        if self._memory_backward_prompt_count != 4 or self._memory_optimizer_count != 0:
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d optimizer boundary does not follow four prompt backwards"
            )
        self._memory_phase_observer(
            "before", "optimizer_step", step=step_index + 1
        )

    def _after_optimizer_step(self, *, step_index: int) -> None:
        self._memory_optimizer_count += 1
        self._memory_phase_observer("after", "optimizer_step")

    def _checkpoint_authority(self, step: int) -> Any:
        phase = "adapter_export"
        self._memory_phase_observer("before", phase, step=step)
        try:
            result = super()._checkpoint_authority(step)
            self._memory_export_count += 1
            return result
        finally:
            self._memory_phase_observer("after", phase)

    def hotswap_stable_slot(self, **kwargs: Any) -> Mapping[str, Any]:
        phase = "sampler_refresh_and_fresh_identity"
        self._memory_phase_observer(
            "before", phase, step=self.current_sampler_version + 1
        )
        try:
            result = super().hotswap_stable_slot(**kwargs)
            self._memory_refresh_count += 1
            self._memory_fresh_identity_verifier_count += 1
            return result
        finally:
            self._memory_phase_observer("after", phase)

    def run_b2_calibration_step_v1(self, **kwargs: Any) -> Mapping[str, Any]:
        record = super().run_b2_calibration_step_v1(**kwargs)
        if not (
            self._memory_backward_prompt_count == 4
            and self._memory_backbone_forward_count == 4
            and self._memory_backbone_backward_count == 4
            and self._memory_retain_graph_count == 0
            and self._memory_lm_head_chunk_count >= 4
            and self._memory_optimizer_count == 1
            and self._memory_export_count == 1
            and self._memory_refresh_count == 1
            and self._memory_fresh_identity_verifier_count == 1
            and self._scheduler_step_count == record["optimizer_step"]
            and record["next_policy_version"] == record["policy_version"] + 1
        ):
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d microbatch/optimizer/refresh contract drift"
            )
        audit = {
            "schema_version": 1,
            "artifact_kind": "b2_memory_step_execution_audit_v1",
            "run_id": self.config["run"]["run_id"],
            "optimizer_step": record["optimizer_step"],
            "from_policy_version": record["policy_version"],
            "to_policy_version": record["next_policy_version"],
            "physical_microbatch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "backward_prompt_count": self._memory_backward_prompt_count,
            "backbone_forward_calls": self._memory_backbone_forward_count,
            "backbone_backward_calls_per_prompt": (
                self._memory_backbone_backward_count
                / self._memory_backward_prompt_count
            ),
            "lm_head_chunk_count": self._memory_lm_head_chunk_count,
            "retain_graph_calls": self._memory_retain_graph_count,
            "chunked_backward_loss": self._memory_chunk_loss_total,
            "optimizer_step_count": self._memory_optimizer_count,
            "scheduler_step_count": 1,
            "adapter_export_count": self._memory_export_count,
            "sampler_refresh_count": self._memory_refresh_count,
            "fresh_identity_verifier_count": (
                self._memory_fresh_identity_verifier_count
            ),
            "policy_version_increment_count": 1,
            "rollout_resampled_during_accumulation": False,
            "p_old_detached": True,
            "teacher_same_token_scoring": True,
            "teacher_generated_completion": False,
            "raw_prompt_persisted": False,
            "response_tokens_persisted": False,
            "hidden_gradient_trainable_scope": dict(
                self._hidden_gradient_trainable_scope
            ),
        }
        _atomic_json(
            self.output
            / "memory_step_audits"
            / f"step_{record['optimizer_step']:02d}.json",
            audit,
        )
        self._memory_phase_observer(
            "before", "step_end", step=record["optimizer_step"]
        )
        self._memory_phase_observer("after", "step_end")
        self._last_step_end_snapshot = {
            "step": record["optimizer_step"],
            "registry_count": self._registry_count(),
            "model_count": self._model_count(),
            "gpus": self._gpu_memory_snapshot(),
        }
        return record

    def memory_step_end_record_v1(self) -> Mapping[str, Any]:
        if self._last_step_end_snapshot is None:
            raise ProductionTwoStepQualificationV6Error(
                "memory step-end snapshot is absent"
            )
        return dict(self._last_step_end_snapshot)

    def memory_canary_runtime_summary_v1(self) -> Mapping[str, Any]:
        return {
            "initial_adapter_sha256": self._initial_adapter_sha256,
            "minimum_free_bytes_by_gpu": [
                int(value) for value in self._minimum_free_bytes_by_gpu
            ],
            "oom": False,
            "non_finite": False,
            "optimizer_steps_executed": self._memory_optimizer_count,
            "scheduler_step_count": int(
                getattr(self, "_scheduler_step_count", 0)
            ),
            "sampler_refresh_count": self._memory_refresh_count,
            "backbone_backward_calls": int(
                getattr(self, "_memory_backbone_backward_count", 0)
            ),
            "backbone_forward_calls": int(
                getattr(self, "_memory_backbone_forward_count", 0)
            ),
            "lm_head_chunk_count": int(
                getattr(self, "_memory_lm_head_chunk_count", 0)
            ),
            "retain_graph_calls": int(
                getattr(self, "_memory_retain_graph_count", 0)
            ),
            "adapter_export_count": int(
                getattr(self, "_memory_export_count", 0)
            ),
            "fresh_identity_verifier_count": int(
                getattr(self, "_memory_fresh_identity_verifier_count", 0)
            ),
            "policy_version_increment_count": 1,
            "hidden_gradient_trainable_scope": dict(
                getattr(self, "_hidden_gradient_trainable_scope", {})
            ),
            "initial_registry_count": int(
                getattr(self, "_initial_registry_count", self._registry_count())
            ),
            "final_registry_count": self._registry_count(),
            "initial_model_count": int(
                getattr(self, "_initial_model_count", self._model_count())
            ),
            "final_model_count": self._model_count(),
        }

    def save_b2_resume_checkpoint_v1(self, **kwargs: Any) -> Mapping[str, Any]:
        version = int(kwargs.get("logical_version", -1))
        if version not in {5, 10, 15, 20}:
            raise ProductionTwoStepQualificationV6Error(
                "P4.8d resume checkpoint must be v5/v10/v15/v20"
            )
        # The parent method owns the exact atomic optimizer/RNG/state format;
        # it is generalized in the production kernel for the package-declared
        # checkpoint set.
        phase = f"resume_checkpoint_v{version}_save"
        self._memory_phase_observer("before", phase, step=version)
        try:
            return super().save_b2_resume_checkpoint_v1(**kwargs)
        finally:
            self._memory_phase_observer("after", phase)

    def reload_b2_resume_checkpoint_v1(self, **kwargs: Any) -> Mapping[str, Any]:
        version = int(kwargs.get("logical_version", -1))
        phase = f"resume_checkpoint_v{version}_fresh_reload"
        self._memory_phase_observer("before", phase, step=version)
        try:
            result = super().reload_b2_resume_checkpoint_v1(**kwargs)
        finally:
            self._memory_phase_observer("after", phase)
        configure_training_student(self.student_model)
        enable_inputs = getattr(self.student_model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
        self.student_model.eval()
        self._hidden_gradient_trainable_scope = (
            assert_hidden_gradient_trainable_scope(self.student_model)
        )
        _atomic_json(
            self.output / "hidden_gradient_trainable_scope_v10_reload.json",
            {
                **self._hidden_gradient_trainable_scope,
                "run_id": self.config["run"]["run_id"],
                "logical_version": 10,
            },
        )
        return result

    def final_checkpoint_reload_identity_v1(self) -> Mapping[str, Any]:
        phase = "final_checkpoint_v20_fresh_reload"
        self._memory_phase_observer("before", phase, step=20)
        try:
            return super().final_checkpoint_reload_identity_v1()
        finally:
            self._memory_phase_observer("after", phase)


__all__ = [
    "HIDDEN_GRADIENT_LORA_TARGET_MODULES",
    "MemoryBalancedProductionTwoStepSessionV1",
    "assert_hidden_gradient_trainable_scope",
]
