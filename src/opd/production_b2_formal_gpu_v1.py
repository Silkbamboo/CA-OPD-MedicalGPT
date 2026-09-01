"""GPU-only formal B2 session built on the P4.8g validated kernel."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from src.opd.production_b2_formal_checkpoint_v1 import (
    prune_formal_checkpoints,
    seal_formal_checkpoint,
    validate_formal_checkpoint,
)
from src.opd.production_b2_formal_v1 import FormalB2Error, formal_step_limit
from src.opd.production_b2_memory_execution_gpu_v1 import (
    MemoryBalancedProductionTwoStepSessionV1,
)
from src.opd.production_qualification_two_step_gpu_v7 import _atomic_json


class FormalB2SessionV1(MemoryBalancedProductionTwoStepSessionV1):
    """Fresh-v0 B2 session with formal metrics and bounded transient storage."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        formal_step_limit(config)
        super().__init__(config, **kwargs)
        self.formal_step_root = self.output / "formal_steps"
        self.formal_step_root.mkdir(parents=True, exist_ok=True)

    def _build_resume_identity_probe_v1(
        self, prompt_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a deterministic label-free fixed-action probe for resume smoke."""

        if len(prompt_rows) != 4:
            raise FormalB2Error("formal resume identity requires the next frozen 2+2 batch")
        response_ids = [
            int(value)
            for value in self.tokenizer.encode(
                "This is a deterministic label-free resume identity probe.",
                add_special_tokens=False,
            )
        ][:32]
        if not response_ids:
            raise FormalB2Error("formal resume fixed response tokenization is empty")
        result: list[dict[str, Any]] = []
        for index, row in enumerate(prompt_rows):
            prompt_ids = [
                int(value)
                for value in self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.render_prompt_text(row)},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            ]
            if not prompt_ids:
                raise FormalB2Error("formal resume prompt tokenization is empty")
            result.append(
                {
                    "fixture_id": str(row.get("sample_id", f"resume-{index}")),
                    "source_role": str(row.get("target_role", "unknown")),
                    "prompt_ids": prompt_ids,
                    "response_ids": list(response_ids),
                }
            )
        return result

    def restore_formal_checkpoint_v1(
        self,
        checkpoint: Path,
        *,
        package_content_sha256: str,
        config_sha256: str,
        manifest_sha256: str,
        schedule_sha256: str,
        resume_prompt_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Restore exact model/optimizer/scheduler/RNG/cursor and smoke identity."""

        checkpoint = Path(checkpoint).resolve()
        manifest = validate_formal_checkpoint(checkpoint)
        step = int(manifest["logical_version"])
        expected = {
            "package_content_sha256": package_content_sha256,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "schedule_sha256": schedule_sha256,
        }
        if any(manifest.get(field) != value for field, value in expected.items()):
            raise FormalB2Error("formal resume package/config/data identity differs")
        training_state = json.loads(
            (checkpoint / "training_state.json").read_text(encoding="utf-8")
        )
        sampler_state = json.loads(
            (checkpoint / "sampler_state.json").read_text(encoding="utf-8")
        )
        if not (
            training_state.get("optimizer_step") == step
            and training_state.get("scheduler_step") == step
            and training_state.get("policy_version") == step
            and training_state.get("data_cursor") == step * 4
            and all(training_state.get(field) == value for field, value in expected.items())
            and sampler_state.get("policy_version") == step
            and sampler_state.get("runtime_adapter_sha256")
            == manifest["adapter_sha256"]
            and sampler_state.get("active_adapter") == "student_active"
            and sampler_state.get("registry_count") == 1
        ):
            raise FormalB2Error("formal resume state/cursor/sampler differs")
        route_state = json.loads(
            (checkpoint / "route_state.json").read_text(encoding="utf-8")
        )
        restore_route = getattr(self, "restore_formal_route_state", None)
        if callable(restore_route):
            restore_route(route_state)

        self._release(
            self.torch,
            self.teacher_model,
            self.student_model,
            self.sampler_model,
        )
        self.teacher_model = None
        self.student_model = None
        self.sampler_model = None
        self.optimizer = None
        self.scheduler = None
        trainer_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        self.student_model = self.PeftModel.from_pretrained(
            trainer_base,
            checkpoint,
            adapter_name="default",
            is_trainable=True,
        )
        del trainer_base
        from src.opd.production_b2_memory_execution_v1 import (
            configure_training_student,
        )
        from src.opd.production_b2_memory_execution_gpu_v1 import (
            assert_hidden_gradient_trainable_scope,
        )

        configure_training_student(self.student_model)
        enable_inputs = getattr(self.student_model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
        self.student_model.eval()
        self.parameters = dict(self.student_model.named_parameters())
        self.trainable_names = tuple(
            name
            for name, parameter in self.parameters.items()
            if parameter.requires_grad
        )
        self._hidden_gradient_trainable_scope = assert_hidden_gradient_trainable_scope(
            self.student_model
        )
        self.frozen_versions = {
            name: parameter._version
            for name, parameter in self.parameters.items()
            if name not in self.trainable_names
        }
        optimizer = self.optimizer_config
        self.optimizer = self.torch.optim.AdamW(
            [self.parameters[name] for name in self.trainable_names],
            lr=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            betas=(float(optimizer["beta1"]), float(optimizer["beta2"])),
            eps=float(optimizer["epsilon"]),
            foreach=bool(optimizer["foreach"]),
        )
        from src.opd.production_b2_memory_checkpoint_v2 import (
            build_constant_lr_scheduler,
        )

        self.scheduler = build_constant_lr_scheduler(self.torch, self.optimizer)
        self.optimizer.load_state_dict(
            self.torch.load(
                checkpoint / "optimizer_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.scheduler.load_state_dict(
            self.torch.load(
                checkpoint / "scheduler_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        if int(self.scheduler.state_dict().get("last_epoch", -1)) != step:
            raise FormalB2Error("formal resume scheduler step differs")
        sampler_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.sampler_model = self.PeftModel.from_pretrained(
            sampler_base,
            checkpoint,
            adapter_name="student_active",
            is_trainable=False,
        )
        del sampler_base
        self.sampler_model.eval()
        trainer = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=step,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        saved = self.adapter_artifact_identity(
            checkpoint,
            logical_version=step,
            runtime_name="student_active",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=step,
            runtime_name="student_active",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not (
            trainer["aggregate_tensor_sha256"]
            == saved["aggregate_tensor_sha256"]
            == runtime["aggregate_tensor_sha256"]
            == manifest["adapter_sha256"]
        ):
            raise FormalB2Error("formal resume trainer/checkpoint/sampler identity differs")
        authority = self.trainer_authority_from_manifest(
            saved,
            artifact_manifest_sha256=manifest["files"]["adapter_transport_manifest.json"]["sha256"],
            trainer_memory_reload_gate_passed=True,
            run_token=f"{self.config['run']['run_id']}:formal-resume-v{step}",
        )
        rng = self.torch.load(
            checkpoint / "rng_state.pt", map_location="cpu", weights_only=True
        )
        if not isinstance(rng, Mapping) or not isinstance(rng.get("cuda"), list) or len(rng["cuda"]) != 2:
            raise FormalB2Error("formal resume RNG state is incomplete")
        self.torch.set_rng_state(rng["cpu"])
        self.torch.cuda.set_rng_state_all(rng["cuda"])
        self.authorities = {step: authority}
        self.current_sampler_version = step
        self.current_sampler_runtime = runtime
        self._current_checkpoint_path = checkpoint
        self._optimizer_step_count = step
        self._scheduler_step_count = step
        self.probe_rows = self._build_resume_identity_probe_v1(resume_prompt_rows)
        with self.torch.inference_mode():
            trainer_values, _ = self._score_rows(
                self.student_model, self.probe_rows, device="cuda:0", inference=True
            )
            sampler_values, _ = self._score_rows(
                self.sampler_model, self.probe_rows, device="cuda:1", inference=True
            )
        rows_left = [
            trainer_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        rows_right = [
            sampler_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        gap = self._gap_metrics(rows_left, rows_right, self.probe_rows)
        if gap["finite_rate"] != 1.0 or gap["max"] > float(
            self.config["validation"]["same_path_max_gap"]
        ):
            raise FormalB2Error("formal resume identity smoke differs")
        self._initial_registry_count = self._registry_count()
        self._initial_model_count = self._model_count()
        probe_binding = hashlib.sha256(
            json.dumps(
                [
                    {
                        "fixture_id": row["fixture_id"],
                        "source_role": row["source_role"],
                        "prompt_ids": row["prompt_ids"],
                        "response_ids": row["response_ids"],
                    }
                    for row in self.probe_rows
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            "artifact_kind": "formal_b2_resume_identity_smoke_v1",
            "logical_version": step,
            "data_cursor": step * 4,
            "adapter_sha256": manifest["adapter_sha256"],
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
            "sampler_state_restored": True,
            "same_path": dict(gap),
            "resume_probe_binding_sha256": probe_binding,
            "resume_probe_prompt_count": len(self.probe_rows),
            "resume_probe_label_access_count": 0,
            "passed": True,
        }

    def run_formal_step_v1(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        record = dict(
            super().run_b2_calibration_step_v1(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=max_new_tokens,
            )
        )
        step = int(record["optimizer_step"])
        b2_path = (
            self.output
            / "b2_steps"
            / f"step_{step_index:02d}_v{step_index}_to_v{step_index + 1}.json"
        )
        try:
            kernel = json.loads(b2_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FormalB2Error(
                f"formal step reconstruction evidence invalid: {type(error).__name__}"
            ) from error
        reconstruction = kernel.get("reconstruction_telemetry")
        if not isinstance(reconstruction, Mapping):
            raise FormalB2Error("formal step reconstruction telemetry is absent")
        ratio = reconstruction.get("optimizer_update", {}).get("ppo_ratio_post")
        private = self._last_b2_step_private
        if not isinstance(ratio, Mapping) or not isinstance(private, Mapping):
            raise FormalB2Error("formal step ratio/private telemetry is absent")
        step_end = self.memory_step_end_record_v1()
        formal = {
            **record,
            "artifact_kind": "formal_b2_step_v1",
            "formal_phase": "stage1" if step <= 120 else "registered_extension",
            "per_source_objective": dict(private["per_source_objective"]),
            "ratio": {
                key: float(ratio[key])
                for key in ("mean", "max", "p95", "p99", "clip_fraction")
            },
            "gradient_norm_before_clip": float(
                reconstruction["optimizer_update"]["gradient_norm_before_clip"]
            ),
            "registry_count": int(step_end["registry_count"]),
            "model_count": int(step_end["model_count"]),
            "gpu_step_end": list(step_end["gpus"]),
            "kernel_step_artifact_sha256": str(kernel["reconstruction_telemetry"]["artifact_sha256"])
            if "artifact_sha256" in kernel["reconstruction_telemetry"]
            else str(kernel.get("step_artifact_sha256", record.get("step_artifact_sha256", ""))),
        }
        path = self.formal_step_root / f"step_{step:03d}.json"
        if path.exists() or path.is_symlink():
            raise FormalB2Error("formal step artifact already exists")
        _atomic_json(path, formal)
        return formal

    def seal_registered_checkpoint_v1(
        self,
        *,
        logical_version: int,
        package_content_sha256: str,
        config_sha256: str,
        manifest_sha256: str,
        schedule_sha256: str,
        environment: Mapping[str, Any],
        target_step: int,
    ) -> dict[str, Any]:
        manifest = seal_formal_checkpoint(
            self,
            logical_version=logical_version,
            data_cursor=logical_version * 4,
            package_content_sha256=package_content_sha256,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            schedule_sha256=schedule_sha256,
            environment=environment,
        )
        rotation = prune_formal_checkpoints(self.output, target_step=target_step)
        return {"manifest": manifest, "rotation": rotation}

    def release_transient_step_artifacts_v1(self, logical_version: int) -> list[str]:
        """Delete only this session's already-verified transient adapter copies."""

        step = int(logical_version)
        roots_and_targets = (
            (self.output / "checkpoints", self.output / "checkpoints" / f"v{step}"),
            (self.scratch, self.scratch / f"save_v{step}"),
        )
        removed: list[str] = []
        for root, target in roots_and_targets:
            root = root.resolve()
            if target.exists():
                if target.is_symlink() or target.resolve().parent != root:
                    raise FormalB2Error("transient cleanup target escaped owned root")
                shutil.rmtree(target)
                removed.append(str(target))
        return removed


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_formal_step_health(
    records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    initial_registry_count: int,
    initial_model_count: int,
    minimum_disk_free_bytes: int = 10_000_000_000,
) -> dict[str, Any]:
    """Apply preregistered per-step and two-consecutive-window abort gates."""

    if not records:
        raise FormalB2Error("formal health requires at least one step")
    record = records[-1]
    ratio = record.get("ratio")
    isolation = record.get("isolation")
    scalar_fields = (
        "loss",
        "objective",
        "ess_fraction",
        "ppo_clip_fraction",
        "gradient_norm",
        "adapter_delta_norm",
    )
    if not (
        isinstance(ratio, Mapping)
        and isinstance(isolation, Mapping)
        and all(_finite(record.get(field)) for field in scalar_fields)
        and all(_finite(ratio.get(field)) for field in ("mean", "max", "p95", "p99", "clip_fraction"))
    ):
        raise FormalB2Error("formal health found NaN/Inf or missing scalar")
    failures: list[str] = []
    if int(record.get("teacher_gradient_tensor_count", -1)) != 0 or int(record.get("base_gradient_tensor_count", -1)) != 0:
        failures.append("teacher_or_base_gradient")
    if int(record.get("nonzero_update_tensor_count", 0)) <= 0 or float(record.get("adapter_delta_norm", 0.0)) <= 0:
        failures.append("zero_update")
    if any(isolation.get(field) is not False for field in ("final_access", "controller_access", "confirmation_access", "label_access")):
        failures.append("restricted_access")
    if int(record.get("disk_remaining_bytes", 0)) < minimum_disk_free_bytes:
        failures.append("disk_below_10gb")
    if float(record["ess_fraction"]) < float(gates["ess_fraction"]["abort_below"]):
        failures.append("ess")
    if float(ratio["max"]) > float(gates["ratio_max"]["abort_above"]):
        failures.append("ratio_max")
    if float(ratio["p99"]) > float(gates["ratio_p99"]["abort_above"]):
        failures.append("ratio_p99")
    if float(record["ppo_clip_fraction"]) > float(gates["clip_fraction"]["abort_above"]):
        failures.append("clip_fraction")
    if float(record["gradient_norm"]) > float(
        gates["gradient_norm_before_clip"]["after_clip_abort_above"]
    ):
        failures.append("gradient_norm")
    if int(record.get("registry_count", -1)) != initial_registry_count or int(record.get("model_count", -1)) != initial_model_count:
        failures.append("registry_or_model_growth")

    window_results: list[dict[str, Any]] = []
    for start in range(max(0, len(records) - 5), max(0, len(records) - 3)):
        window = records[start : start + 4]
        if len(window) != 4:
            continue
        samples = [sample for item in window for sample in item["prompt_samples"]]
        rates = {
            "overall": sum(bool(sample["truncated"]) for sample in samples) / len(samples)
        }
        for source in ("medical_opd_o1", "medical_opd_cmb"):
            source_rows = [sample for sample in samples if sample["source"] == source]
            rates[source] = sum(bool(sample["truncated"]) for sample in source_rows) / len(source_rows)
        window_results.append({"start_step": start + 1, "end_step": start + 4, "rates": rates, "over": any(value > 0.20 for value in rates.values())})
    if len(window_results) == 2 and all(item["over"] for item in window_results):
        failures.append("two_consecutive_truncation_windows")
    if failures:
        raise FormalB2Error("formal health gate failed: " + ",".join(failures))
    return {
        "passed": True,
        "optimizer_step": int(record["optimizer_step"]),
        "window_results": window_results,
    }


__all__ = [
    "FormalB2SessionV1",
    "validate_formal_step_health",
]
