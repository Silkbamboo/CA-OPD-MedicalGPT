"""Persistent-sampler Phase A/B kernel for P4.6.

Importing this module is CPU safe.  The public executor owns the scientific
ordering and accepts a session factory for CPU contract tests.  The default
factory is the only place that imports the real Torch/Transformers/PEFT
implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol


PRODUCTION_BACKEND_ID = "custom_transformers_peft_three_policy_v5"
PRODUCTION_REFRESH_MECHANISM = "peft_0_17_1_hotswap_stable_slot"
PRODUCTION_SLOT = "student_active"
P4_7_B2_ALLOWED_RESPONSE_LENGTHS = frozenset(
    {256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096}
)


class ProductionTwoStepQualificationV6Error(RuntimeError):
    """The production two-step state machine failed closed."""


class EmitPhase(Protocol):
    def __call__(
        self,
        phase: str,
        payload: Mapping[str, Any],
        metric: Mapping[str, Any],
    ) -> str: ...


class MicroEvidenceGate(Protocol):
    def __call__(
        self,
        *,
        expected_v1_tensor_sha256: str,
        expected_refresh_v1_sha256: str,
    ) -> Mapping[str, Any]: ...


class TwoStepSessionV6(Protocol):
    sampler_id: int

    def prepare_micro_evidence(self) -> Mapping[str, Any]: ...

    def run_corrected_step(
        self, step_index: int, rollout: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def hotswap_stable_slot(
        self,
        *,
        current_authority: Mapping[str, Any],
        target_authority: Mapping[str, Any],
        checkpoint: str,
    ) -> Mapping[str, Any]: ...

    def generate_guarded_rollout(
        self,
        step_index: int,
        *,
        authority: Mapping[str, Any],
        refresh_artifact_sha256: str,
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


SessionFactory = Callable[..., TwoStepSessionV6]


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _sha_file(path)


def _digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProductionTwoStepQualificationV6Error(f"{label} is not a SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionTwoStepQualificationV6Error(f"{label} is absent")
    return value


def _validate_contract(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_id")
        != "ca-opd/p4.6-combined-production-qualification/v1"
        or config.get("schema_version") != 1
    ):
        raise ProductionTwoStepQualificationV6Error("P4.6 schema contract drift")
    if (
        config.get("production_binding", {}).get("backend_id")
        != PRODUCTION_BACKEND_ID
    ):
        raise ProductionTwoStepQualificationV6Error("production backend mismatch")
    refresh = config.get("sampler_refresh", {})
    if (
        refresh.get("candidate_mechanism") != PRODUCTION_REFRESH_MECHANISM
        or refresh.get("runtime_slot") != PRODUCTION_SLOT
    ):
        raise ProductionTwoStepQualificationV6Error(
            "stable-slot refresh contract drift"
        )
    two_step = config.get("two_step", {})
    if (
        two_step.get("optimizer_steps") != 2
        or two_step.get("rollout1_must_be_generated_by") != "v1"
        or two_step.get("p_old1_must_be") != "v1"
        or two_step.get("final_logical_version") != "v2"
        or two_step.get("stable_registry_count") != 1
    ):
        raise ProductionTwoStepQualificationV6Error("two-step contract drift")
    if config.get("isolation") != {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }:
        raise ProductionTwoStepQualificationV6Error("evaluation isolation drift")


def _emit(
    emit: EmitPhase,
    phase: str,
    payload: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> str:
    if payload.get("status") != "pass":
        raise ProductionTwoStepQualificationV6Error(
            f"{phase} did not pass before persistence"
        )
    return _digest(emit(phase, payload, metric), f"{phase} artifact")


def _step_result(value: Any, *, step: int) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str
]:
    result = _mapping(value, f"step{step} result")
    reconstruction = _mapping(result.get("reconstruction"), "reconstruction")
    authority = _mapping(result.get("authority"), "authority artifact")
    authority_state = _mapping(result.get("authority_state"), "authority state")
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ProductionTwoStepQualificationV6Error("authority checkpoint is absent")
    expected_version = step + 1
    authority_state_version = authority_state.get(
        "adapter_logical_version", authority_state.get("logical_version")
    )
    if (
        authority.get("logical_version") != f"v{expected_version}"
        or authority_state_version != expected_version
    ):
        raise ProductionTwoStepQualificationV6Error(
            f"step{step} authority logical version mismatch"
        )
    artifact_sha = _digest(
        authority.get("aggregate_tensor_sha256"), f"v{expected_version} artifact"
    )
    runtime_sha = _digest(
        authority_state.get("aggregate_tensor_sha256"),
        f"v{expected_version} authority",
    )
    if artifact_sha != runtime_sha:
        raise ProductionTwoStepQualificationV6Error(
            f"v{expected_version} authority payload/state mismatch"
        )
    return reconstruction, authority, authority_state, checkpoint


class ProductionTwoStepSessionV6:
    """One trainer and one long-lived PEFT sampler for the two formal steps."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: Path,
        route: str = "two_step",
        checkpoint_v2: Path | None = None,
        expected_v2_sha256: str | None = None,
        qualification_authority_artifact_sha256: str | None = None,
    ) -> None:
        # These imports are deliberately inside the explicitly authorized
        # constructor.  Importing this module remains CPU/CUDA safe.
        import torch
        import yaml
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.opd.calibration_data import render_prompt_text
        from src.opd.pg_opd_contract import (
            ThreePolicyLogProbBundle,
            decoupled_corrected_objective,
            grouped_trajectory_mean,
            validate_three_policy_bundle,
        )
        from src.opd.pg_opd_validation import audit_optimizer_update
        from src.opd.production_qualification_contract_v6 import (
            build_probe_manifest,
            build_probe_spec,
            validate_v0_guard_evidence,
        )
        from src.opd.production_qualification_telemetry_v6 import (
            build_reconstruction_telemetry,
            validate_reconstruction_telemetry,
        )
        from src.opd.production_sampler_identity_v5 import (
            SamplerIdentityGuardError,
            build_adapter_identity_manifest,
            guard_sampler_operation,
            trainer_authority_from_manifest,
        )
        from src.opd.production_sampler_refresh_v5 import (
            adapter_artifact_identity,
            refresh_stable_slot,
            runtime_identity_from_peft,
        )
        from src.opd.rollout_probability import validate_rollout_behavior_provenance
        from src.opd.scorer_gpu_calibration import _apply_determinism, _release

        self.config = deepcopy(dict(config))
        if route not in {
            "two_step",
            "base_null",
            "length",
            "b2",
            "b2_calibration",
        }:
            raise ProductionTwoStepQualificationV6Error("unknown session route")
        self.route = route
        self.config_path = Path(config_path)
        self.root = Path(__file__).resolve().parents[2]
        self.output = Path(str(self.config["run"]["output_dir"]))
        if not self.output.is_dir():
            raise ProductionTwoStepQualificationV6Error(
                "qualification output envelope does not exist"
            )
        self.torch = torch
        self.yaml = yaml
        self.LoraConfig = LoraConfig
        self.PeftModel = PeftModel
        self.get_peft_model = get_peft_model
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.render_prompt_text = render_prompt_text
        self.ThreePolicyLogProbBundle = ThreePolicyLogProbBundle
        self.decoupled_corrected_objective = decoupled_corrected_objective
        self.grouped_trajectory_mean = grouped_trajectory_mean
        self.validate_three_policy_bundle = validate_three_policy_bundle
        self.audit_optimizer_update = audit_optimizer_update
        self.build_probe_manifest = build_probe_manifest
        self.build_probe_spec = build_probe_spec
        self.validate_v0_guard_evidence = validate_v0_guard_evidence
        self.build_reconstruction_telemetry = build_reconstruction_telemetry
        self.validate_reconstruction_telemetry = validate_reconstruction_telemetry
        self.SamplerIdentityGuardError = SamplerIdentityGuardError
        self.build_adapter_identity_manifest = build_adapter_identity_manifest
        self.guard_sampler_operation = guard_sampler_operation
        self.trainer_authority_from_manifest = trainer_authority_from_manifest
        self.adapter_artifact_identity = adapter_artifact_identity
        self.refresh_stable_slot = refresh_stable_slot
        self.runtime_identity_from_peft = runtime_identity_from_peft
        self.validate_rollout_behavior_provenance = validate_rollout_behavior_provenance
        self._release = _release
        _apply_determinism(torch)

        protocol_path = self.root / str(self.config["validation"]["config_path"])
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        self.algorithm = dict(protocol["algorithm"])
        self.optimizer_config = dict(protocol["optimizer"])
        self.gates = dict(protocol["calibration_gates"])
        if route in {"b2", "b2_calibration"}:
            binding = _mapping(
                self.config.get("b2_protocol_binding"), "B2 protocol binding"
            )
            if not (
                binding.get("optimizer") == self.optimizer_config.get("type")
                and float(binding.get("learning_rate"))
                == float(self.optimizer_config.get("learning_rate"))
                and float(binding.get("ppo_clip_low"))
                == float(self.algorithm.get("clip_low"))
                and float(binding.get("ppo_clip_high"))
                == float(self.algorithm.get("clip_high"))
                and int(binding.get("student_lora_rank"))
                == int(self.optimizer_config.get("lora_rank"))
                and int(binding.get("student_lora_alpha"))
                == int(self.optimizer_config.get("lora_alpha"))
                and binding.get("student_lora_target_modules")
                == self.optimizer_config.get("target_modules")
                and float(binding.get("correction_upper_threshold")) == 2.0
                and float(binding.get("correction_ess_fraction_min"))
                == float(self.gates.get("ess_fraction_min"))
                and float(binding.get("correction_cap_fraction_max"))
                == float(self.gates.get("cap_fraction_max"))
            ):
                raise ProductionTwoStepQualificationV6Error(
                    "B2 protocol differs from the frozen formula"
                )
        self.model_path = str(self.config["model"]["id"])
        self.base_revision = str(self.config["model"]["revision"])
        self.tokenizer_revision = str(self.config["model"]["tokenizer_revision"])
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".p4_6_two_step_", dir=self.output.parent
        )
        self.scratch = Path(self.temporary.name)
        self.student_model: Any = None
        self.sampler_model: Any = None
        self.teacher_model: Any = None
        self.tokenizer: Any = None
        self.optimizer: Any = None
        self.parameters: dict[str, Any] = {}
        self.trainable_names: tuple[str, ...] = ()
        self.frozen_versions: dict[str, int] = {}
        self.probe_rows: list[dict[str, Any]] = []
        self.authorities: dict[int, Mapping[str, Any]] = {}
        self.current_sampler_version = 0
        self.current_sampler_runtime: Mapping[str, Any] | None = None
        self._step_prompt_ids: dict[int, set[str]] = {}
        self._optimizer_step_count = 0
        self._current_checkpoint_path: Path | None = None
        self._last_b2_step_private: dict[str, Any] | None = None
        self._last_b2_refresh: dict[str, Any] | None = None
        self._closed = False
        if route == "b2":
            if (
                checkpoint_v2 is None
                or expected_v2_sha256 is None
                or qualification_authority_artifact_sha256 is None
            ):
                raise ProductionTwoStepQualificationV6Error(
                    "B2 route requires qualification v2 checkpoint authority"
                )
            self._initialize_b2_checkpoint(
                Path(checkpoint_v2),
                expected_v2_sha256=expected_v2_sha256,
                qualification_authority_artifact_sha256=(
                    qualification_authority_artifact_sha256
                ),
            )
        else:
            self._initialize_models()
        if route == "length":
            if checkpoint_v2 is None or expected_v2_sha256 is None:
                raise ProductionTwoStepQualificationV6Error(
                    "length route requires trainer-authoritative v2"
                )
            self._load_length_checkpoint_v2(
                Path(checkpoint_v2), expected_v2_sha256=expected_v2_sha256
            )

    @property
    def sampler_id(self) -> int:
        if self.sampler_model is None:
            raise ProductionTwoStepQualificationV6Error("sampler is unavailable")
        return id(self.sampler_model)

    def _initialize_models(self) -> None:
        torch = self.torch
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.tokenizer_revision,
        )
        base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        optimizer = self.optimizer_config
        self.student_model = self.get_peft_model(
            base,
            self.LoraConfig(
                r=int(optimizer["lora_rank"]),
                lora_alpha=int(optimizer["lora_alpha"]),
                lora_dropout=float(optimizer["lora_dropout"]),
                target_modules=optimizer["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        del base
        self.student_model.eval()
        self.parameters = dict(self.student_model.named_parameters())
        self.trainable_names = tuple(
            name for name, parameter in self.parameters.items() if parameter.requires_grad
        )
        if not self.trainable_names or any(
            "lora" not in name.lower() for name in self.trainable_names
        ):
            raise ProductionTwoStepQualificationV6Error(
                "Student trainable scope is not LoRA-only"
            )
        self.frozen_versions = {
            name: parameter._version
            for name, parameter in self.parameters.items()
            if name not in self.trainable_names
        }
        self.optimizer = torch.optim.AdamW(
            [self.parameters[name] for name in self.trainable_names],
            lr=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            betas=(float(optimizer["beta1"]), float(optimizer["beta2"])),
            eps=float(optimizer["epsilon"]),
            foreach=bool(optimizer["foreach"]),
        )

        v0_path = self.scratch / "v0"
        self.student_model.save_pretrained(v0_path, safe_serialization=True)
        sampler_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.sampler_model = self.PeftModel.from_pretrained(
            sampler_base,
            v0_path,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del sampler_base
        self.sampler_model.eval()

        trainer_v0 = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=0,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        saved_v0 = self.adapter_artifact_identity(
            v0_path,
            logical_version=0,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        runtime_v0 = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=0,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not (
            self._tensor_identity_equal(trainer_v0, saved_v0)
            and self._identity_equal(saved_v0, runtime_v0)
        ):
            raise ProductionTwoStepQualificationV6Error(
                "initial trainer/saved/runtime v0 identity mismatch"
            )
        authority = self.trainer_authority_from_manifest(
            saved_v0,
            artifact_manifest_sha256=_canonical_sha(saved_v0),
            trainer_memory_reload_gate_passed=True,
            run_token=f"{self.config['run']['run_id']}:adapter-v0",
        )
        self.authorities[0] = authority
        self.current_sampler_runtime = runtime_v0

    def _initialize_b2_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_v2_sha256: str,
        qualification_authority_artifact_sha256: str,
    ) -> None:
        """Restore trainer and one long-lived sampler from qualified v2."""

        torch = self.torch
        checkpoint = checkpoint.resolve()
        expected = _digest(expected_v2_sha256, "B2 v2 authority")
        authority_artifact = _digest(
            qualification_authority_artifact_sha256,
            "qualification authority_v2 artifact",
        )
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise ProductionTwoStepQualificationV6Error(
                "qualified B2 v2 checkpoint is absent"
            )
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.tokenizer_revision,
        )
        saved = self.adapter_artifact_identity(
            checkpoint,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if saved["aggregate_tensor_sha256"] != expected:
            raise ProductionTwoStepQualificationV6Error(
                "B2 checkpoint differs from qualification v2 authority"
            )

        trainer_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
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
        self.student_model.eval()
        self.parameters = dict(self.student_model.named_parameters())
        self.trainable_names = tuple(
            name
            for name, parameter in self.parameters.items()
            if parameter.requires_grad
        )
        if not self.trainable_names or any(
            "lora" not in name.lower() for name in self.trainable_names
        ):
            raise ProductionTwoStepQualificationV6Error(
                "B2 restored Student trainable scope is not LoRA-only"
            )
        self.frozen_versions = {
            name: parameter._version
            for name, parameter in self.parameters.items()
            if name not in self.trainable_names
        }
        optimizer = self.optimizer_config
        self.optimizer = torch.optim.AdamW(
            [self.parameters[name] for name in self.trainable_names],
            lr=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            betas=(float(optimizer["beta1"]), float(optimizer["beta2"])),
            eps=float(optimizer["epsilon"]),
            foreach=bool(optimizer["foreach"]),
        )

        sampler_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.sampler_model = self.PeftModel.from_pretrained(
            sampler_base,
            checkpoint,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del sampler_base
        self.sampler_model.eval()
        trainer = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=2,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not (
            self._identity_equal(trainer, saved)
            and self._identity_equal(saved, runtime)
        ):
            raise ProductionTwoStepQualificationV6Error(
                "B2 initial trainer/checkpoint/sampler v2 identity mismatch"
            )
        authority = self.trainer_authority_from_manifest(
            saved,
            artifact_manifest_sha256=authority_artifact,
            trainer_memory_reload_gate_passed=True,
            run_token=f"{self.config['run']['run_id']}:adapter-v2",
        )
        self.authorities = {2: authority}
        self.current_sampler_version = 2
        self.current_sampler_runtime = runtime
        self._current_checkpoint_path = checkpoint

    @staticmethod
    def _tensor_records(identity: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "canonical_key": item["canonical_key"],
                "sha256": item["sha256"],
                "shape": list(item["shape"]),
                "dtype": item["canonical_dtype"],
                "byte_length": int(item["canonical_byte_length"]),
            }
            for item in identity["tensors"]
        ]

    def _tensor_identity_equal(
        self, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> bool:
        return bool(
            left.get("aggregate_tensor_sha256")
            == right.get("aggregate_tensor_sha256")
            and self._tensor_records(left) == self._tensor_records(right)
        )

    def _identity_equal(
        self, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> bool:
        return bool(
            self._tensor_identity_equal(left, right)
            and left.get("canonical_config_sha256")
            == right.get("canonical_config_sha256")
            and left.get("base_revision") == right.get("base_revision")
            and left.get("tokenizer_revision") == right.get("tokenizer_revision")
        )

    def _pad(self, values: list[Any], *, device: str = "cuda:0") -> tuple[Any, Any]:
        torch = self.torch
        result = torch.nn.utils.rnn.pad_sequence(
            [value.reshape(-1).to(device=device, dtype=torch.float32) for value in values],
            batch_first=True,
            padding_value=0.0,
        )
        mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(values):
            mask[index, : value.numel()] = True
        return result, mask

    def _action_logprobs(
        self, model: Any, row: Mapping[str, Any], *, device: str
    ) -> Any:
        torch = self.torch
        prompt = [int(value) for value in row["prompt_ids"]]
        response = [int(value) for value in row["response_ids"]]
        ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
        result = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        )
        logits = result.logits[:, len(prompt) - 1 : len(prompt) - 1 + len(response), :].float()
        targets = torch.tensor(response, dtype=torch.long, device=device).view(1, -1, 1)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)

    def _score_rows(
        self,
        model: Any,
        rows: list[Mapping[str, Any]],
        *,
        device: str,
        inference: bool,
    ) -> tuple[Any, Any]:
        context = self.torch.inference_mode() if inference else nullcontext()
        with context:
            values = [
                self._action_logprobs(model, row, device=device).reshape(-1)
                for row in rows
            ]
        return self._pad(values, device="cuda:0")

    def _frozen_prompt_rows(self, step_index: int) -> tuple[list[dict[str, Any]], str]:
        from src.opd.production_qualification_prompts_v6 import (
            load_frozen_prompt_group,
        )

        selected = load_frozen_prompt_group(
            self.config, f"step{step_index}", repo_root=self.root
        )
        if len(selected) != 4:
            raise ProductionTwoStepQualificationV6Error("two-step prompt count drift")
        result: list[dict[str, Any]] = []
        projection: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            sample_id = str(row["sample_id"])
            role = str(row["target_role"])
            content_hash = str(row["content_hash"])
            prompt = self.render_prompt_text(row)
            result.append(
                {
                    "fixture_id": sample_id,
                    "source_role": role,
                    "source_sample_id": sample_id,
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
            projection.append(
                {
                    "order": index,
                    "sample_id": sample_id,
                    "source_role": role,
                    "content_hash": content_hash,
                }
            )
        if [item["source_role"] for item in result] != [
            "medical_opd_o1",
            "medical_opd_cmb",
            "medical_opd_o1",
            "medical_opd_cmb",
        ]:
            raise ProductionTwoStepQualificationV6Error(
                "frozen two-step prompt order must alternate O1/CMB"
            )
        return result, _canonical_sha(projection)

    def _generation_config(self, model: Any, *, max_new_tokens: int) -> dict[str, Any]:
        generation = deepcopy(dict(self.config["formal_rollout"]["transformers"]))
        generation["max_new_tokens"] = int(max_new_tokens)
        generation["eos_token_id"] = model.generation_config.eos_token_id
        generation["pad_token_id"] = model.generation_config.pad_token_id
        return generation

    def _generate_rows(
        self,
        model: Any,
        rows: list[dict[str, Any]],
        *,
        device: str,
        step_index: int,
    ) -> list[dict[str, Any]]:
        torch = self.torch
        generation = self._generation_config(
            model, max_new_tokens=int(self.config["micro_replay"]["max_new_tokens"])
        )
        result: list[dict[str, Any]] = []
        device_index = int(device.split(":", 1)[1])
        for index, row in enumerate(rows):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            seed = (
                int(self.config["run"]["seed"]) * 100_000
                + step_index * 10_000
                + index
            )
            with torch.random.fork_rng(devices=[device_index]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    **generation,
                )
            response = [
                int(value)
                for value in generated.sequences[0, len(row["prompt_ids"]) :].tolist()
            ]
            if not response or len(generated.scores) != len(response):
                raise ProductionTwoStepQualificationV6Error(
                    "generation/token score alignment failed"
                )
            behavior = [
                float(torch.log_softmax(score[0].float(), dim=-1)[token].detach().cpu())
                for token, score in zip(response, generated.scores, strict=True)
            ]
            raw = [
                float(torch.log_softmax(logit[0].float(), dim=-1)[token].detach().cpu())
                for token, logit in zip(response, generated.logits, strict=True)
            ]
            eos = generation["eos_token_id"]
            eos_ids = {int(eos)} if isinstance(eos, int) else {int(item) for item in eos or []}
            result.append(
                {
                    **row,
                    "response_ids": response,
                    "rollout_behavior_logprob": behavior,
                    "raw_generation_logprob": raw,
                    "seed": seed,
                    "eos_observed": response[-1] in eos_ids,
                }
            )
            del generated
        return result

    def _provenance(
        self,
        rows: list[Mapping[str, Any]],
        *,
        authority: Mapping[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        generation = self._generation_config(
            self.student_model,
            max_new_tokens=int(self.config["micro_replay"]["max_new_tokens"]),
        )
        identity = [
            {
                "fixture_id": row["fixture_id"],
                "prompt_ids": row["prompt_ids"],
                "response_ids": row["response_ids"],
            }
            for row in rows
        ]
        return {
            "artifact_protocol_version": "p4.3-full-support-trajectory-v1",
            "trajectory_run_id": self.config["run"]["run_id"],
            "trajectory_kind": "fresh_full_support",
            "backend": "transformers",
            "backend_version": "4.56.2",
            "model_version": self.base_revision,
            "adapter_version": authority["aggregate_tensor_sha256"],
            "generation_config": generation,
            "processor_warper_provenance": {
                "all_support_changing_processors_disabled": True,
                "active_logits_processor_warper_classes": [],
                "active_stopping_criteria_classes": [
                    "EosTokenCriteria",
                    "MaxLengthCriteria",
                ],
                "selected_token_score_stage": "processed_pre_softmax",
                "source": "local_transformers_4.56.2_generation_utils._sample",
                "identity_source": "effective_generation_config_plus_local_transformers_4.56.2_source",
            },
            "score_source": "generate.scores_manual_log_softmax_selected_token",
            "score_semantics": "normalized_behavior_logprob",
            "behavior_selected_token_logprob_saved": True,
            "raw_selected_token_logprob_saved": True,
            "token_identity_sha256": _canonical_sha(identity),
            "eos_and_truncation_saved": True,
            "seed": int(self.config["run"]["seed"]),
            "generator": "torch_cuda_default_generator_scoped_manual_seed_per_prompt",
            "sampler_adapter_version": step_index,
            "sampler_adapter_sha256": authority["aggregate_tensor_sha256"],
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }

    def _gap_metrics(
        self,
        left: list[Any],
        right: list[Any],
        rows: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for row_index, (first, second, row) in enumerate(
            zip(left, right, rows, strict=True)
        ):
            first_values = first.detach().float().cpu().reshape(-1)
            second_values = second.detach().float().cpu().reshape(-1)
            if first_values.shape != second_values.shape:
                raise ProductionTwoStepQualificationV6Error(
                    "fixed-action shape mismatch"
                )
            for position, (a, b) in enumerate(
                zip(first_values.tolist(), second_values.tolist(), strict=True)
            ):
                records.append(
                    {
                        "sample_id": str(row["fixture_id"]),
                        "token_position": position,
                        "token_id": int(row["response_ids"][position]),
                        "gap": abs(float(a) - float(b)),
                    }
                )
        if not records:
            raise ProductionTwoStepQualificationV6Error("fixed-action probe is empty")
        gaps = self.torch.tensor(
            [record["gap"] for record in records], dtype=self.torch.float64
        )
        worst = max(records, key=lambda record: record["gap"])
        return {
            "mae": float(gaps.mean()),
            "p50": float(self.torch.quantile(gaps, 0.50)),
            "p95": float(self.torch.quantile(gaps, 0.95)),
            "p99": float(self.torch.quantile(gaps, 0.99)),
            "max": float(gaps.max()),
            "finite_rate": float(self.torch.isfinite(gaps).double().mean()),
            "worst_sample_id": worst["sample_id"],
            "worst_token_position": worst["token_position"],
            "worst_token_id": worst["token_id"],
        }

    def _masked_distribution(self, values: Any, mask: Any) -> dict[str, float]:
        selected = values.detach().float()[mask.to(dtype=self.torch.bool)]
        if selected.numel() <= 0 or not bool(self.torch.isfinite(selected).all()):
            raise ProductionTwoStepQualificationV6Error(
                "B2 metric distribution is empty or non-finite"
            )
        return {
            "mean": float(selected.mean().cpu()),
            "std": float(selected.std(unbiased=False).cpu()),
        }

    def _checkpoint_authority(
        self, version: int
    ) -> tuple[dict[str, Any], Mapping[str, Any], str]:
        torch = self.torch
        checkpoint = self.output / "checkpoints" / f"v{version}"
        if checkpoint.exists():
            raise ProductionTwoStepQualificationV6Error(
                f"checkpoint v{version} already exists"
            )
        staging = self.scratch / f"save_v{version}"
        self.student_model.save_pretrained(staging, safe_serialization=True)
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source = staging / name
            if not source.is_file():
                raise ProductionTwoStepQualificationV6Error(
                    f"checkpoint transport file absent: {name}"
                )
            shutil.copy2(source, checkpoint / name)

        trainer_identity = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=version,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        target_identity = self.adapter_artifact_identity(
            checkpoint,
            logical_version=version,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        with torch.inference_mode():
            trainer_values, _ = self._score_rows(
                self.student_model,
                self.probe_rows,
                device="cuda:0",
                inference=True,
            )
        trainer_rows = [
            trainer_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        fresh = self.PeftModel.from_pretrained(
            base,
            checkpoint,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del base
        fresh.eval()
        reload_identity = self.runtime_identity_from_peft(
            fresh,
            logical_version=version,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        with torch.inference_mode():
            reload_values, _ = self._score_rows(
                fresh, self.probe_rows, device="cuda:0", inference=True
            )
        reload_rows = [
            reload_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        same_path = self._gap_metrics(trainer_rows, reload_rows, self.probe_rows)
        self._release(torch, fresh)
        del fresh
        if not (
            self._tensor_identity_equal(trainer_identity, target_identity)
            and self._identity_equal(target_identity, reload_identity)
            and same_path["finite_rate"] == 1.0
            and same_path["max"] <= float(
                self.config["validation"]["same_path_max_gap"]
            )
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"trainer memory/reload v{version} authority gate failed"
            )

        files = [
            {
                "path": name,
                "role": role,
                "sha256": _sha_file(checkpoint / name),
                "size_bytes": (checkpoint / name).stat().st_size,
            }
            for name, role in (
                ("adapter_config.json", "adapter_config"),
                ("adapter_model.safetensors", "adapter_weights"),
            )
        ]
        transport = {
            "schema_version": 1,
            "logical_version": f"v{version}",
            "canonical_config_sha256": target_identity[
                "canonical_config_sha256"
            ],
            "aggregate_tensor_sha256": target_identity[
                "aggregate_tensor_sha256"
            ],
            "files": files,
        }
        transport_name = "adapter_transport_manifest.json"
        transport_sha = _atomic_json(checkpoint / transport_name, transport)
        immutable_sha = _canonical_sha(
            {
                "trainer_tensor_sha256": trainer_identity[
                    "aggregate_tensor_sha256"
                ],
                "saved_tensor_sha256": target_identity[
                    "aggregate_tensor_sha256"
                ],
                "reload_tensor_sha256": reload_identity[
                    "aggregate_tensor_sha256"
                ],
                "transport_manifest_sha256": transport_sha,
                "same_path": same_path,
            }
        )
        authority = self.trainer_authority_from_manifest(
            target_identity,
            artifact_manifest_sha256=immutable_sha,
            trainer_memory_reload_gate_passed=True,
            run_token=f"{self.config['run']['run_id']}:adapter-v{version}",
        )
        payload = {
            "status": "pass",
            "logical_version": f"v{version}",
            "runtime_adapter_name": PRODUCTION_SLOT,
            "active_adapter": PRODUCTION_SLOT,
            "canonical_config_sha256": target_identity[
                "canonical_config_sha256"
            ],
            "aggregate_tensor_sha256": target_identity[
                "aggregate_tensor_sha256"
            ],
            "per_tensor_digests": self._tensor_records(target_identity),
            "tensor_count": target_identity["tensor_count"],
            "total_bytes": target_identity["total_canonical_bytes"],
            "base_revision": self.base_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "immutable_manifest_sha256": immutable_sha,
            "trainer_memory_reload_same_path": same_path,
            "checkpoint": {
                "directory": f"checkpoints/v{version}",
                "transport_manifest_path": (
                    f"checkpoints/v{version}/{transport_name}"
                ),
                "transport_manifest_sha256": transport_sha,
            },
        }
        self.authorities[version] = authority
        return payload, authority, f"checkpoints/v{version}"

    def current_policy_identity(self) -> dict[str, Any]:
        authority = self.authorities[self.current_sampler_version]
        runtime = self.current_sampler_runtime or {}
        registry = runtime.get("registry_snapshot", {})
        return {
            "logical_version": f"v{self.current_sampler_version}",
            "tensor_sha256": authority["aggregate_tensor_sha256"],
            "active_slot": runtime.get("active_adapter"),
            "registry_count": registry.get("adapter_count"),
            "checkpoint_path": (
                None
                if self.current_sampler_version == 0
                else str(
                    (
                        self._current_checkpoint_path
                        or (
                            self.output
                            / "checkpoints"
                            / f"v{self.current_sampler_version}"
                        )
                    ).resolve()
                )
            ),
        }

    def initial_calibration_identity(self) -> dict[str, Any]:
        """Prove that P4.8 starts from fresh Base plus a zero-effect LoRA."""

        if self.route != "b2_calibration" or self.current_sampler_version != 0:
            raise ProductionTwoStepQualificationV6Error(
                "fresh B2 calibration identity is unavailable on this route"
            )
        initialization = _mapping(
            self.config.get("student_initialization"),
            "fresh Student initialization",
        )
        qualification = _mapping(
            self.config.get("qualification_evidence"),
            "qualification evidence",
        )
        lora_b = [
            parameter
            for name, parameter in self.parameters.items()
            if name in self.trainable_names and "lora_b" in name.lower()
        ]
        nonzero = sum(
            int(parameter.detach().count_nonzero().cpu()) for parameter in lora_b
        )
        authority = self.authorities[0]
        initial_sha = _digest(
            authority.get("aggregate_tensor_sha256"), "fresh v0 authority"
        )
        qualification_sha = _digest(
            initialization.get("forbidden_qualification_adapter_sha256"),
            "qualification v2 adapter",
        )
        if not (
            initialization.get("mode")
            == "fresh_base_plus_fresh_zero_lora_v1"
            and initialization.get("source_adapter_path") is None
            and initialization.get("qualification_v2_usage")
            == "evidence_only_not_student_init"
            and qualification.get("v2_tensor_sha256") == qualification_sha
            and len(lora_b) == 252
            and nonzero == 0
            and initial_sha != qualification_sha
            and authority.get("tensor_count") == 504
        ):
            raise ProductionTwoStepQualificationV6Error(
                "fresh zero-effect Student identity gate failed"
            )
        return {
            "schema_version": 1,
            "artifact_kind": "b2_fresh_student_initial_identity_v1",
            "run_id": self.config["run"]["run_id"],
            "initialization": "fresh_base_plus_fresh_zero_lora_v1",
            "logical_version": 0,
            "adapter_sha256": initial_sha,
            "qualification_v2_sha256": qualification_sha,
            "differs_from_qualification_v2": True,
            "source_adapter_path": None,
            "tensor_count": 504,
            "lora_b_tensor_count": len(lora_b),
            "nonzero_lora_b_value_count": nonzero,
            "zero_effect_verified": True,
            "base_gradient_tensor_count": 0,
        }

    def _load_length_checkpoint_v2(
        self, checkpoint: Path, *, expected_v2_sha256: str
    ) -> None:
        """Replace the bootstrap v0 objects with an immutable fresh v2 sampler."""

        expected = _digest(expected_v2_sha256, "length v2 authority")
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise ProductionTwoStepQualificationV6Error(
                "length v2 checkpoint is absent"
            )
        saved = self.adapter_artifact_identity(
            checkpoint,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if saved["aggregate_tensor_sha256"] != expected:
            raise ProductionTwoStepQualificationV6Error(
                "length checkpoint differs from trainer-authoritative v2"
            )
        self._release(self.torch, self.student_model, self.sampler_model)
        self.student_model = None
        self.sampler_model = None
        self.optimizer = None
        self.parameters = {}
        self.trainable_names = ()
        base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.sampler_model = self.PeftModel.from_pretrained(
            base,
            checkpoint,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del base
        self.sampler_model.eval()
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=2,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._identity_equal(saved, runtime):
            raise ProductionTwoStepQualificationV6Error(
                "fresh length sampler runtime differs from v2 checkpoint"
            )
        authority = self.trainer_authority_from_manifest(
            saved,
            artifact_manifest_sha256=_canonical_sha(saved),
            trainer_memory_reload_gate_passed=True,
            run_token=f"{self.config['run']['run_id']}:adapter-v2",
        )
        self.authorities = {2: authority}
        self.current_sampler_version = 2
        self.current_sampler_runtime = runtime
        self._current_checkpoint_path = checkpoint.resolve()

    def run_base_teacher_null(
        self,
        *,
        prompt_rows: list[Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute a fresh zero-LoRA Base=Teacher update on real forwards."""

        if self.route != "base_null" or self.current_sampler_version != 0:
            raise ProductionTwoStepQualificationV6Error(
                "Base-null route is not an independent fresh zero-LoRA session"
            )
        if config is not self.config and dict(config) != self.config:
            raise ProductionTwoStepQualificationV6Error("Base-null config drift")
        if len(prompt_rows) != 4:
            raise ProductionTwoStepQualificationV6Error(
                "Base-null requires frozen 2+2 prompts"
            )
        torch = self.torch
        rows: list[dict[str, Any]] = []
        for index, source in enumerate(prompt_rows):
            prompt = self.render_prompt_text(source)
            prompt_ids = [
                int(value)
                for value in self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            ]
            ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
            generation = self._generation_config(self.student_model, max_new_tokens=128)
            generation["output_scores"] = False
            generation["output_logits"] = False
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(int(self.config["run"]["seed"]) * 100_000 + index)
                torch.cuda.manual_seed_all(
                    int(self.config["run"]["seed"]) * 100_000 + index
                )
                generated = self.student_model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    **generation,
                )
            response_ids = [
                int(value)
                for value in generated.sequences[0, len(prompt_ids) :].tolist()
            ]
            if not response_ids:
                raise ProductionTwoStepQualificationV6Error(
                    "Base-null generated an empty fixed action"
                )
            rows.append(
                {
                    "fixture_id": str(source["sample_id"]),
                    "source_role": str(source["target_role"]),
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                }
            )

        before_identity = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=0,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        with self.student_model.disable_adapter():
            base_scores, mask = self._score_rows(
                self.student_model, rows, device="cuda:0", inference=True
            )
        current_scores, current_mask = self._score_rows(
            self.student_model, rows, device="cuda:0", inference=False
        )
        if not torch.equal(mask, current_mask):
            raise ProductionTwoStepQualificationV6Error(
                "Base-null fixed-action masks differ"
            )
        valid_gap = (current_scores.detach() - base_scores).abs()[mask]
        advantage = base_scores.detach() - base_scores.detach()
        objective = ((current_scores * advantage) * mask).sum() / mask.sum()
        loss = -objective
        before_parameters = {
            name: self.parameters[name].detach().cpu().clone()
            for name in self.trainable_names
        }
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_before = float(
            torch.nn.utils.clip_grad_norm_(
                [self.parameters[name] for name in self.trainable_names],
                float(self.optimizer_config["global_gradient_clip_norm"]),
            )
        )
        gradient_after = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum().cpu())
                for name, parameter in self.parameters.items()
                if name in self.trainable_names and parameter.grad is not None
            )
        )
        self.optimizer.step()
        after_parameters = {
            name: self.parameters[name].detach().cpu().clone()
            for name in self.trainable_names
        }
        deltas = {
            name: float((after_parameters[name] - before_parameters[name]).norm())
            for name in self.trainable_names
        }
        after_identity = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=0,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        base_gradient_count = sum(
            parameter.grad is not None
            for name, parameter in self.student_model.named_parameters()
            if name not in self.trainable_names
        )
        finite = bool(
            torch.isfinite(base_scores).all()
            and torch.isfinite(current_scores).all()
            and torch.isfinite(advantage).all()
        )
        payload = {
            "schema_version": 6,
            "run_id": self.config["run"]["run_id"],
            "route_is_independent": True,
            "teacher_is_real_base": True,
            "old_actor_is_same_base_detached": True,
            "current_actor_is_base_equivalent_zero_lora": True,
            "fresh_optimizer": True,
            "medical_optimizer_state_reused": False,
            "label_access": False,
            "final_access": False,
            "controller_access": False,
            "current_pre_base_max_abs_gap": float(valid_gap.max().cpu()),
            "advantage_max_abs": float(advantage.abs().max().cpu()),
            "objective": float(objective.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": gradient_before,
            "gradient_norm_after_clip": gradient_after,
            "parameter_delta_norm": math.sqrt(sum(value * value for value in deltas.values())),
            "nonzero_update_tensor_count": sum(value != 0.0 for value in deltas.values()),
            "adapter_tensor_sha256_before": before_identity["aggregate_tensor_sha256"],
            "adapter_tensor_sha256_after": after_identity["aggregate_tensor_sha256"],
            "base_gradient_tensor_count": base_gradient_count,
            "teacher_gradient_tensor_count": 0,
            "finite_rate": 1.0 if finite else 0.0,
        }
        if not (
            payload["current_pre_base_max_abs_gap"]
            <= float(self.config["base_null"]["current_pre_base_max_gap"])
            and payload["advantage_max_abs"] == 0.0
            and payload["objective"] == 0.0
            and payload["loss"] == 0.0
            and payload["gradient_norm_before_clip"] == 0.0
            and payload["gradient_norm_after_clip"] == 0.0
            and payload["parameter_delta_norm"] == 0.0
            and payload["nonzero_update_tensor_count"] == 0
            and payload["adapter_tensor_sha256_before"]
            == payload["adapter_tensor_sha256_after"]
            and payload["base_gradient_tensor_count"] == 0
            and payload["finite_rate"] == 1.0
        ):
            raise ProductionTwoStepQualificationV6Error(
                "real Base=Teacher null gate failed"
            )
        return payload

    def generate_length_trajectories(
        self,
        *,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> list[dict[str, Any]]:
        """Generate source-real v2 length records without persisting text."""

        if self.route != "length" or self.current_sampler_version != 2:
            raise ProductionTwoStepQualificationV6Error(
                "length generation is not running from immutable v2"
            )
        if enable_thinking is not False or max_new_tokens not in {384, 512}:
            raise ProductionTwoStepQualificationV6Error(
                "length generation envelope drift"
            )
        torch = self.torch
        generation = self._generation_config(
            self.sampler_model, max_new_tokens=max_new_tokens
        )
        generation["output_scores"] = False
        generation["output_logits"] = False
        records: list[dict[str, Any]] = []
        eos_ids = generation.get("eos_token_id")
        eos_set = {
            int(value) for value in (eos_ids if isinstance(eos_ids, list) else [eos_ids])
        }
        for index, row in enumerate(prompt_rows):
            prompt = self.render_prompt_text(row)
            prompt_ids = [
                int(value)
                for value in self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            ]
            ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:1")
            torch.cuda.reset_peak_memory_stats(1)
            started = time.perf_counter()
            with torch.random.fork_rng(devices=[1]):
                seed = int(self.config["run"]["seed"]) * 100_000 + max_new_tokens * 100 + index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                generated = self.sampler_model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    **generation,
                )
            elapsed = time.perf_counter() - started
            response = [
                int(value)
                for value in generated.sequences[0, len(prompt_ids) :].tolist()
            ]
            if not response:
                eos_position = None
                finite = True
            else:
                positions = [
                    position + 1
                    for position, token in enumerate(response)
                    if token in eos_set
                ]
                eos_position = positions[0] if positions else None
                probe = {
                    "prompt_ids": prompt_ids,
                    "response_ids": response,
                }
                with torch.inference_mode():
                    scored = self._action_logprobs(
                        self.sampler_model, probe, device="cuda:1"
                    )
                finite = bool(torch.isfinite(scored).all())
            decoded = self.tokenizer.decode(response, skip_special_tokens=False)
            records.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "source_role": str(row["target_role"]),
                    "token_count": len(response),
                    "eos_position": eos_position,
                    "finite": finite,
                    "invalid_or_empty": not response,
                    "thinking_tag_count": decoded.count("<think>"),
                    "finish_reason": "eos" if eos_position is not None else "length",
                    "tokens_per_second": len(response) / elapsed if elapsed > 0 else 0.0,
                    "wall_time_seconds": elapsed,
                    "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(1)),
                }
            )
        return records

    @staticmethod
    def _request(authority: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "run_token": authority["run_token"],
            "logical_version": authority["adapter_logical_version"],
            "authoritative_tensor_sha256": authority[
                "aggregate_tensor_sha256"
            ],
            "canonical_config_sha256": authority["canonical_config_sha256"],
            "base_revision": authority["base_revision"],
            "tokenizer_revision": authority["tokenizer_revision"],
        }

    def _stale_evidence(
        self,
        *,
        authority: Mapping[str, Any],
        runtime: Mapping[str, Any],
        stale_authority: Mapping[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        request = self._request(stale_authority)
        try:
            self.guard_sampler_operation(
                authority=authority,
                runtime_identity=runtime,
                request_identity=request,
                operation=operation,
                callback=lambda: (_ for _ in ()).throw(
                    AssertionError("stale request reached model forward")
                ),
            )
        except self.SamplerIdentityGuardError as error:
            return {
                "logical_version": f"v{stale_authority['adapter_logical_version']}",
                "rejected": True,
                "error_code": error.code,
                "rejection_phase": error.evidence["guard_stage"],
                "scoring_executed": error.evidence["scoring_executed"],
                "generation_executed": error.evidence["generation_executed"],
            }
        raise ProductionTwoStepQualificationV6Error("stale request was accepted")

    def prepare_micro_evidence(self) -> Mapping[str, Any]:
        torch = self.torch
        prompt_rows, _selection_sha = self._frozen_prompt_rows(0)
        trajectories = self._generate_rows(
            self.student_model,
            prompt_rows,
            device="cuda:0",
            step_index=0,
        )
        provenance = self._provenance(
            trajectories, authority=self.authorities[0], step_index=0
        )
        self.validate_rollout_behavior_provenance(
            provenance,
            expected_sampler_adapter_sha256=self.authorities[0][
                "aggregate_tensor_sha256"
            ],
            expected_trajectory_run_id=self.config["run"]["run_id"],
        )

        sample_ids = tuple(str(row["fixture_id"]) for row in trajectories)
        probe_spec = self.build_probe_spec(
            run_id=self.config["run"]["run_id"],
            prompt_manifest_sha256=self.config["prompt_selection"][
                "selection_manifest_sha256"
            ],
            ordered_sample_ids=sample_ids,
        )
        if probe_spec["probe_spec_sha256"] != self.config["fixed_action_probe"][
            "probe_spec_sha256"
        ]:
            raise ProductionTwoStepQualificationV6Error("probe spec SHA drift")
        probe = self.build_probe_manifest(
            probe_spec,
            {
                str(row["fixture_id"]): [
                    {
                        "token_id": int(token_id),
                        "response_token_position": position,
                        "valid": True,
                    }
                    for position, token_id in enumerate(row["response_ids"])
                ]
                for row in trajectories
            },
        )
        counts = probe["per_prompt_count"]
        self.probe_rows = [
            {
                **row,
                "response_ids": row["response_ids"][
                    : int(counts[str(row["fixture_id"])])
                ],
            }
            for row in trajectories
        ]
        official_probe: dict[str, Any] = {"status": "pass", **dict(probe)}

        sampler_id_before = self.sampler_id
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=0,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        with torch.inference_mode():
            first, _ = self._score_rows(
                self.sampler_model,
                self.probe_rows,
                device="cuda:1",
                inference=True,
            )
            second, _ = self._score_rows(
                self.sampler_model,
                self.probe_rows,
                device="cuda:1",
                inference=True,
            )
        first_rows = [
            first[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        second_rows = [
            second[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        repeat = self._gap_metrics(first_rows, second_rows, self.probe_rows)
        request = self._request(self.authorities[0])
        normal_values, normal_execution = self.guard_sampler_operation(
            authority=self.authorities[0],
            runtime_identity=runtime,
            request_identity=request,
            operation="fixed_action",
            callback=lambda: self._score_rows(
                self.sampler_model,
                self.probe_rows,
                device="cuda:1",
                inference=True,
            )[0],
        )
        wrong_request = dict(request)
        wrong_request["authoritative_tensor_sha256"] = (
            "0" * 64
            if request["authoritative_tensor_sha256"] != "0" * 64
            else "f" * 64
        )
        wrong: dict[str, Any]
        try:
            self.guard_sampler_operation(
                authority=self.authorities[0],
                runtime_identity=runtime,
                request_identity=wrong_request,
                operation="fixed_action",
                callback=lambda: (_ for _ in ()).throw(
                    AssertionError("wrong authority reached model forward")
                ),
            )
        except self.SamplerIdentityGuardError as error:
            wrong_execution = dict(error.evidence)
            wrong_error_code = error.code
        else:
            raise ProductionTwoStepQualificationV6Error(
                "wrong v0 authority request was accepted"
            )
        if not (
            sampler_id_before == self.sampler_id
            and self._identity_equal(runtime, self.authorities[0])
            and runtime["registry_snapshot"]["peft_config_names"]
            == [PRODUCTION_SLOT]
            and repeat["finite_rate"] == 1.0
            and repeat["max"]
            <= float(self.config["validation"]["same_path_max_gap"])
            and bool(torch.isfinite(normal_values).all())
            and normal_execution["accepted"] is True
            and wrong_error_code == "SAMPLER_RUNTIME_TENSOR_MISMATCH"
        ):
            raise ProductionTwoStepQualificationV6Error("v0 identity/guard gate failed")
        authority_after = self.authorities[0]["aggregate_tensor_sha256"]
        common_guard = {
            "run_id": self.config["run"]["run_id"],
            "logical_version": 0,
            "guard_stage": "identity_guard_before_forward",
            "trainer_authoritative_tensor_sha256": authority_after,
            "sampler_runtime_tensor_sha256": runtime[
                "aggregate_tensor_sha256"
            ],
            "authority_after_request_sha256": authority_after,
            "canonical_config_sha256": self.authorities[0][
                "canonical_config_sha256"
            ],
            "base_revision": self.base_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "generation_executed": False,
            "finite": True,
            "silent_fallback": False,
        }
        normal = {
            **common_guard,
            "accepted": normal_execution["accepted"],
            "guard_stage": normal_execution["guard_stage"],
            "request_expected_tensor_sha256": request[
                "authoritative_tensor_sha256"
            ],
            "scoring_executed": normal_execution["scoring_executed"],
        }
        wrong = {
            **common_guard,
            "accepted": wrong_execution["accepted"],
            "guard_stage": wrong_execution["guard_stage"],
            "request_expected_tensor_sha256": wrong_request[
                "authoritative_tensor_sha256"
            ],
            "scoring_executed": wrong_execution["scoring_executed"],
            "generation_executed": wrong_execution["generation_executed"],
            "error_code": wrong_error_code,
            "sampler_self_authority_accepted": False,
        }
        self.validate_v0_guard_evidence(normal, wrong)
        v0_guard = {
            "status": "pass",
            "normal_v0": normal,
            "wrong_authority": wrong,
            "same_instance_repeat": repeat,
        }
        self.current_sampler_runtime = runtime
        return {
            "probe_manifest": official_probe,
            "v0_guard": v0_guard,
            "v0_authority": self.authorities[0],
            "rollout": {
                "policy_version": "v0",
                "tensor_sha256": self.authorities[0]["aggregate_tensor_sha256"],
                "rows": trajectories,
                "provenance": provenance,
            },
        }

    def run_corrected_step(
        self, step_index: int, rollout: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index < 0
            or (
                self.route not in {"b2", "b2_calibration"}
                and step_index not in {0, 1}
            )
            or (
                self.route in {"b2", "b2_calibration"}
                and step_index != self.current_sampler_version
            )
        ):
            raise ProductionTwoStepQualificationV6Error(
                "corrected step is not the current authoritative version"
            )
        expected_version = f"v{step_index}"
        expected_sha = self.authorities[step_index]["aggregate_tensor_sha256"]
        if (
            rollout.get("policy_version") != expected_version
            or rollout.get("tensor_sha256") != expected_sha
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"step{step_index} rollout identity mismatch"
            )
        corrected_started = time.perf_counter()
        rows = rollout.get("rows")
        provenance = rollout.get("provenance")
        if not isinstance(rows, list) or len(rows) != 4 or not isinstance(
            provenance, Mapping
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"step{step_index} rollout evidence is incomplete"
            )
        self.validate_rollout_behavior_provenance(
            provenance,
            expected_sampler_adapter_sha256=expected_sha,
            expected_trajectory_run_id=self.config["run"]["run_id"],
        )
        torch = self.torch
        self.student_model.to("cuda:0")
        self.student_model.eval()
        step_frozen_versions = {
            name: parameter._version
            for name, parameter in self.parameters.items()
            if name not in self.trainable_names
        }
        live_identity = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=step_index,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._tensor_identity_equal(live_identity, self.authorities[step_index]):
            raise ProductionTwoStepQualificationV6Error(
                f"trainer is not authoritative v{step_index} before p_old"
            )

        scoring_started = time.perf_counter()
        old_actor, old_mask = self._score_rows(
            self.student_model, rows, device="cuda:0", inference=True
        )
        current_pre, current_mask = self._score_rows(
            self.student_model, rows, device="cuda:0", inference=True
        )
        behavior, behavior_mask = self._pad(
            [torch.tensor(row["rollout_behavior_logprob"]) for row in rows]
        )
        teacher_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
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
        teacher, teacher_mask = self._score_rows(
            self.teacher_model, rows, device="cuda:1", inference=True
        )
        if not (
            torch.equal(behavior_mask, old_mask)
            and torch.equal(behavior_mask, current_mask)
            and torch.equal(behavior_mask, teacher_mask)
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"step{step_index} three-policy response masks differ"
            )
        prompt_ids = tuple(str(row["fixture_id"]) for row in rows)
        source_roles = tuple(str(row["source_role"]) for row in rows)
        group_ids = ("g0",) * len(rows)
        bundle = self.ThreePolicyLogProbBundle(
            rollout_behavior_logprob=behavior.detach(),
            old_actor_logprob=old_actor.detach(),
            current_actor_logprob=current_pre.detach().requires_grad_(),
            teacher_logprob=teacher.detach(),
            response_mask=behavior_mask,
            behavior_provenance=provenance,
        )
        self.validate_three_policy_bundle(
            bundle,
            require_pre_update_identity=True,
            identity_tolerance=float(self.gates["current_pre_old_actor_max_abs"]),
        )
        before_result = self.decoupled_corrected_objective(
            bundle,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            source_roles=source_roles,
            beta=float(self.algorithm["beta"]),
            clip_low=float(self.algorithm["clip_low"]),
            clip_high=float(self.algorithm["clip_high"]),
            rollout_is_threshold=2.0,
        )
        correction = before_result.correction.metrics
        partitions = list(correction["per_prompt"].values()) + list(
            correction["per_source"].values()
        )
        if not (
            before_result.correction.truncated_weight.requires_grad is False
            and correction["ess_fraction"]
            >= float(self.config["validation"]["ess_fraction_min"])
            and correction["cap_fraction"]
            <= float(self.config["validation"]["cap_fraction_max"])
            and all(
                item["ess_fraction"]
                >= float(self.config["validation"]["ess_fraction_min"])
                and item["cap_fraction"]
                <= float(self.config["validation"]["cap_fraction_max"])
                for item in partitions
            )
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"step{step_index} correction gate failed"
            )
        pre_update_scoring_seconds = time.perf_counter() - scoring_started

        before_parameters = {
            name: self.parameters[name].detach().cpu().clone()
            for name in self.trainable_names
        }
        self.optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        for row_index, row in enumerate(rows):
            current_row, row_mask = self._score_rows(
                self.student_model, [row], device="cuda:0", inference=False
            )
            length = int(row_mask[0].sum().cpu())
            row_bundle = self.ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.rollout_behavior_logprob[
                    row_index : row_index + 1, :length
                ],
                old_actor_logprob=bundle.old_actor_logprob[
                    row_index : row_index + 1, :length
                ],
                current_actor_logprob=current_row[:, :length],
                teacher_logprob=bundle.teacher_logprob[
                    row_index : row_index + 1, :length
                ],
                response_mask=row_mask[:, :length],
                behavior_provenance=provenance,
            )
            row_result = self.decoupled_corrected_objective(
                row_bundle,
                prompt_ids=(prompt_ids[row_index],),
                group_ids=(group_ids[row_index],),
                source_roles=(source_roles[row_index],),
                beta=float(self.algorithm["beta"]),
                clip_low=float(self.algorithm["clip_low"]),
                clip_high=float(self.algorithm["clip_high"]),
                rollout_is_threshold=2.0,
            )
            (row_result.loss / float(len(rows))).backward()
        gradient_before_clip = float(
            torch.nn.utils.clip_grad_norm_(
                [self.parameters[name] for name in self.trainable_names],
                float(self.optimizer_config["global_gradient_clip_norm"]),
            )
        )
        gradient_after_clip = math.sqrt(
            sum(
                float(
                    self.parameters[name]
                    .grad.detach()
                    .float()
                    .square()
                    .sum()
                    .cpu()
                )
                for name in self.trainable_names
                if self.parameters[name].grad is not None
            )
        )
        gradients = {
            name: (
                self.parameters[name].grad.detach().cpu().clone()
                if self.parameters[name].grad is not None
                else torch.zeros_like(self.parameters[name], device="cpu")
            )
            for name in self.trainable_names
        }
        self.optimizer.step()
        backward_seconds = time.perf_counter() - backward_started
        after_parameters = {
            name: self.parameters[name].detach().cpu().clone()
            for name in self.trainable_names
        }
        update_audit = self.audit_optimizer_update(
            before=before_parameters,
            after=after_parameters,
            loss_gradients=gradients,
            declared_trainable_names=self.trainable_names,
            actual_requires_grad_names=tuple(
                name
                for name, parameter in self.student_model.named_parameters()
                if parameter.requires_grad
            ),
            fresh_optimizer=self._optimizer_step_count == 0,
            weight_decay=float(self.optimizer_config["weight_decay"]),
            require_nonzero=True,
            descent_dot_max=0.0,
        )
        self.student_model.eval()
        post_update_scoring_started = time.perf_counter()
        current_after, after_mask = self._score_rows(
            self.student_model, rows, device="cuda:0", inference=True
        )
        after_bundle = self.ThreePolicyLogProbBundle(
            rollout_behavior_logprob=bundle.rollout_behavior_logprob,
            old_actor_logprob=bundle.old_actor_logprob,
            current_actor_logprob=current_after.detach().requires_grad_(),
            teacher_logprob=bundle.teacher_logprob,
            response_mask=after_mask,
            behavior_provenance=provenance,
        )
        after_result = self.decoupled_corrected_objective(
            after_bundle,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            source_roles=source_roles,
            beta=float(self.algorithm["beta"]),
            clip_low=float(self.algorithm["clip_low"]),
            clip_high=float(self.algorithm["clip_high"]),
            rollout_is_threshold=2.0,
        )
        alignment = float(
            self.grouped_trajectory_mean(
                before_result.correction.truncated_weight
                * before_result.advantage
                * (
                    current_after.detach()
                    - bundle.current_actor_logprob.detach()
                ),
                behavior_mask,
                prompt_ids=prompt_ids,
                group_ids=group_ids,
            ).cpu()
        )
        objective_before = float(before_result.surrogate.detach().cpu())
        objective_after = float(after_result.surrogate.detach().cpu())
        loss_before = float(before_result.loss.detach().cpu())
        loss_after = float(after_result.loss.detach().cpu())
        teacher_gradient_count = sum(
            parameter.grad is not None for parameter in self.teacher_model.parameters()
        )
        base_gradient_count = sum(
            parameter.grad is not None
            for name, parameter in self.student_model.named_parameters()
            if name not in self.trainable_names
        )
        post_update_scoring_seconds = (
            time.perf_counter() - post_update_scoring_started
        )
        telemetry = self.build_reconstruction_telemetry(
            run_id=self.config["run"]["run_id"],
            step_id=(
                f"step{step_index}_v{step_index}_to_v{step_index + 1}"
                if self.route not in {"b2", "b2_calibration"}
                else (
                    f"b2_step{self._optimizer_step_count:02d}_"
                    f"v{step_index}_to_v{step_index + 1}"
                )
            ),
            rollout_logprobs=bundle.rollout_behavior_logprob,
            old_logprobs=bundle.old_actor_logprob,
            current_pre_logprobs=bundle.current_actor_logprob,
            advantages=before_result.advantage,
            response_mask=bundle.response_mask,
            prompt_ids=prompt_ids,
            source_roles=source_roles,
            objective_before=objective_before,
            objective_after=objective_after,
            loss_before=loss_before,
            loss_after=loss_after,
            alignment=alignment,
            ppo_ratio_post=after_result.ppo_ratio,
            gradient_norm_before_clip=gradient_before_clip,
            gradient_norm_after_clip=gradient_after_clip,
            before_parameters=before_parameters,
            after_parameters=after_parameters,
            loss_gradients=gradients,
            teacher_gradient_tensor_count=teacher_gradient_count,
            base_gradient_tensor_count=base_gradient_count,
            optimizer_config={
                "name": str(self.optimizer_config["type"]).lower(),
                "learning_rate": float(self.optimizer_config["learning_rate"]),
                "weight_decay": float(self.optimizer_config["weight_decay"]),
                "max_grad_norm": float(
                    self.optimizer_config["global_gradient_clip_norm"]
                ),
                "ppo_clip_low": 1.0 - float(self.algorithm["clip_low"]),
                "ppo_clip_high": 1.0 + float(self.algorithm["clip_high"]),
                "importance_cap": 2.0,
            },
            near_zero_threshold=float(
                self.config["reconstruction_telemetry"][
                    "advantage_near_zero_threshold"
                ]
            ),
            teacher_detached=not bundle.teacher_logprob.requires_grad,
            old_actor_detached=not bundle.old_actor_logprob.requires_grad,
            correction_weight_detached=(
                not before_result.correction.truncated_weight.requires_grad
            ),
        )
        self.validate_reconstruction_telemetry(telemetry)
        if not (
            objective_after > objective_before
            and loss_after < loss_before
            and alignment > 0
            and update_audit.hard_gate_passed
            and teacher_gradient_count == 0
            and base_gradient_count == 0
            and all(
                self.parameters[name]._version == version
                for name, version in step_frozen_versions.items()
            )
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"step{step_index} optimizer direction/ownership gate failed"
            )
        checkpoint_started = time.perf_counter()
        authority_payload, authority, checkpoint = self._checkpoint_authority(
            step_index + 1
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        if self.route in {"b2", "b2_calibration"}:
            self._last_b2_step_private = {
                "q_logprob": self._masked_distribution(
                    bundle.rollout_behavior_logprob, bundle.response_mask
                ),
                "p_old_logprob": self._masked_distribution(
                    bundle.old_actor_logprob, bundle.response_mask
                ),
                "teacher_logprob": self._masked_distribution(
                    bundle.teacher_logprob, bundle.response_mask
                ),
                "reverse_kl": self._masked_distribution(
                    bundle.old_actor_logprob - bundle.teacher_logprob,
                    bundle.response_mask,
                ),
                "advantage": {
                    **self._masked_distribution(
                        before_result.advantage, bundle.response_mask
                    ),
                    "clip_fraction": 0.0,
                },
                "importance_ratio": self._masked_distribution(
                    after_result.ppo_ratio, bundle.response_mask
                ),
                "valid_token_count": int(bundle.response_mask.sum().cpu()),
                "ppo_clip_fraction": float(after_result.ppo_clip_fraction),
                "ess_fraction": float(correction["ess_fraction"]),
                "objective": objective_after,
                "loss": loss_after,
                "gradient_norm": gradient_after_clip,
                "nonzero_update_tensor_count": int(
                    telemetry["optimizer_update"]["nonzero_update_tensor_count"]
                ),
                "zero_update_tensor_count": int(
                    telemetry["optimizer_update"]["zero_update_tensor_count"]
                ),
                "teacher_gradient_tensor_count": teacher_gradient_count,
                "base_gradient_tensor_count": base_gradient_count,
                "adapter_delta_norm": float(
                    telemetry["optimizer_update"]["parameter_delta_norm"]
                ),
                "scoring_seconds": (
                    pre_update_scoring_seconds + post_update_scoring_seconds
                ),
                "backward_seconds": backward_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "corrected_step_seconds": time.perf_counter() - corrected_started,
            }
        self._optimizer_step_count += 1
        return {
            "reconstruction": {"status": "pass", "telemetry": telemetry},
            "authority": authority_payload,
            "authority_state": authority,
            "checkpoint": checkpoint,
        }

    def release_step_teacher(self, step_index: int) -> None:
        del step_index
        if self.teacher_model is not None:
            self._release(self.torch, self.teacher_model)
            self.teacher_model = None

    def hotswap_stable_slot(
        self,
        *,
        current_authority: Mapping[str, Any],
        target_authority: Mapping[str, Any],
        checkpoint: str,
    ) -> Mapping[str, Any]:
        torch = self.torch
        current_version = int(current_authority["adapter_logical_version"])
        target_version = int(target_authority["adapter_logical_version"])
        if (
            current_version != self.current_sampler_version
            or target_version != current_version + 1
            or current_authority["aggregate_tensor_sha256"]
            != self.authorities[current_version]["aggregate_tensor_sha256"]
            or target_authority["aggregate_tensor_sha256"]
            != self.authorities[target_version]["aggregate_tensor_sha256"]
        ):
            raise ProductionTwoStepQualificationV6Error(
                "hotswap authority transition is not contiguous"
            )
        checkpoint_path = self.output / checkpoint
        object_id = self.sampler_id
        self.sampler_model.to("cuda:1")
        refresh = self.refresh_stable_slot(
            self.sampler_model,
            adapter_path=checkpoint_path,
            current_authority=current_authority,
            target_authority=target_authority,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        runtime = refresh["runtime_identity"]
        request = self._request(target_authority)
        with torch.inference_mode():
            live_values, normal = self.guard_sampler_operation(
                authority=target_authority,
                runtime_identity=runtime,
                request_identity=request,
                operation="fixed_action",
                callback=lambda: self._score_rows(
                    self.sampler_model,
                    self.probe_rows,
                    device="cuda:1",
                    inference=True,
                )[0],
            )
        live_rows = [
            live_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        fresh = self.PeftModel.from_pretrained(
            base,
            checkpoint_path,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del base
        fresh.eval()
        fresh_identity = self.runtime_identity_from_peft(
            fresh,
            logical_version=target_version,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        with torch.inference_mode():
            fresh_values, _ = self._score_rows(
                fresh, self.probe_rows, device="cuda:1", inference=True
            )
        fresh_rows = [
            fresh_values[index, : len(row["response_ids"])]
            for index, row in enumerate(self.probe_rows)
        ]
        self._release(torch, fresh)
        del fresh
        same_path = self._gap_metrics(live_rows, fresh_rows, self.probe_rows)
        stale = self._stale_evidence(
            authority=target_authority,
            runtime=runtime,
            stale_authority=current_authority,
            operation="fixed_action",
        )
        if not (
            object_id == self.sampler_id
            and self._identity_equal(runtime, target_authority)
            and self._identity_equal(fresh_identity, target_authority)
            and runtime["registry_snapshot"]["peft_config_names"]
            == [PRODUCTION_SLOT]
            and refresh["registry_before"]["peft_config_names"]
            == [PRODUCTION_SLOT]
            and refresh["registry_after"]["peft_config_names"]
            == [PRODUCTION_SLOT]
            and same_path["finite_rate"] == 1.0
            and same_path["max"]
            <= float(self.config["validation"]["same_path_max_gap"])
            and normal["accepted"] is True
            and normal["scoring_executed"] is True
            and stale["error_code"] == "STALE_SAMPLER_IDENTITY"
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"refreshed v{target_version} identity/same-path gate failed"
            )
        self.current_sampler_version = target_version
        self.current_sampler_runtime = runtime
        self._current_checkpoint_path = checkpoint_path.resolve()
        self.sampler_model.to("cuda:1")
        payload: dict[str, Any] = {
            "status": "pass",
            "logical_version": f"v{target_version}",
            "canonical_config_sha256": target_authority[
                "canonical_config_sha256"
            ],
            "trainer_tensor_sha256": target_authority[
                "aggregate_tensor_sha256"
            ],
            "runtime_tensor_sha256": runtime["aggregate_tensor_sha256"],
            "fresh_tensor_sha256": fresh_identity["aggregate_tensor_sha256"],
            "runtime_per_tensor_digests": self._tensor_records(runtime),
            "fresh_per_tensor_digests": self._tensor_records(fresh_identity),
            "tensor_count": runtime["tensor_count"],
            "total_bytes": runtime["total_canonical_bytes"],
            "registry_before": list(
                refresh["registry_before"]["peft_config_names"]
            ),
            "registry_after": list(
                refresh["registry_after"]["peft_config_names"]
            ),
            "active_adapter": runtime["active_adapter"],
            "adapter_enabled": runtime["adapters_enabled"],
            "merged": runtime["merged"],
            "same_path": same_path,
            "normal_request": {
                "accepted": True,
                "scoring_executed": True,
                "generation_executed": False,
                "finite_rate": 1.0,
            },
            "stale_request": stale,
            "refresh_latency_seconds": float(refresh["refresh_latency_seconds"]),
            "sampler_object_id": str(object_id),
        }
        if target_version == 2:
            payload["previous_tensor_sha256"] = current_authority[
                "aggregate_tensor_sha256"
            ]
        return payload

    def generate_guarded_rollout(
        self,
        step_index: int,
        *,
        authority: Mapping[str, Any],
        refresh_artifact_sha256: str,
    ) -> Mapping[str, Any]:
        if step_index != 1 or self.current_sampler_version != 1:
            raise ProductionTwoStepQualificationV6Error(
                "only guarded rollout1 from refreshed v1 is valid"
            )
        _digest(refresh_artifact_sha256, "v1 refresh artifact")
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=1,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._identity_equal(runtime, authority):
            raise ProductionTwoStepQualificationV6Error(
                "sampler is not trainer-authoritative v1 before rollout1"
            )
        stale = self._stale_evidence(
            authority=authority,
            runtime=runtime,
            stale_authority=self.authorities[0],
            operation="generation",
        )
        prompt_rows, _projection_sha = self._frozen_prompt_rows(1)
        rows, generation_evidence = self.guard_sampler_operation(
            authority=authority,
            runtime_identity=runtime,
            request_identity=self._request(authority),
            operation="generation",
            callback=lambda: self._generate_rows(
                self.sampler_model,
                prompt_rows,
                device="cuda:1",
                step_index=1,
            ),
        )
        if not isinstance(rows, list) or not rows:
            raise ProductionTwoStepQualificationV6Error("rollout1 is empty")
        q_values = [
            value
            for row in rows
            for value in row["rollout_behavior_logprob"]
        ]
        if not q_values or not all(math.isfinite(float(value)) for value in q_values):
            raise ProductionTwoStepQualificationV6Error("rollout1 q is non-finite")
        trainer_identity = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=1,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._tensor_identity_equal(trainer_identity, authority):
            raise ProductionTwoStepQualificationV6Error(
                "rollout1 p_old trainer is not authoritative v1"
            )
        provenance = self._provenance(rows, authority=authority, step_index=1)
        self.validate_rollout_behavior_provenance(
            provenance,
            expected_sampler_adapter_sha256=authority[
                "aggregate_tensor_sha256"
            ],
            expected_trajectory_run_id=self.config["run"]["run_id"],
        )
        manifest = {
            "status": "pass",
            "generated_by_policy_version": "v1",
            "logical_version": "v1",
            "run_token": authority["run_token"],
            "sampler_tensor_sha256": authority["aggregate_tensor_sha256"],
            "trainer_authority_sha256": authority["aggregate_tensor_sha256"],
            "p_old_actor_tensor_sha256": authority["aggregate_tensor_sha256"],
            "p_old_policy_version": "v1",
            "refresh_artifact_sha256": refresh_artifact_sha256,
            "prompt_manifest_sha256": self.config["prompt_selection"][
                "selection_manifest_sha256"
            ],
            "seed": int(self.config["run"]["seed"]),
            "q_provenance": {
                "backend": "transformers_generate_full_support",
                "logical_version": "v1",
                "run_token": authority["run_token"],
                "runtime_tensor_sha256": runtime[
                    "aggregate_tensor_sha256"
                ],
                "finite_rate": 1.0,
            },
            "stale_v0_pre_rollout": {
                key: stale[key]
                for key in (
                    "rejected",
                    "error_code",
                    "rejection_phase",
                    "scoring_executed",
                    "generation_executed",
                )
            },
            "sampler_object_id": str(self.sampler_id),
            "normal_generation_guard": generation_evidence,
        }
        return {
            "manifest": manifest,
            "rollout": {
                "policy_version": "v1",
                "tensor_sha256": authority["aggregate_tensor_sha256"],
                "rows": rows,
                "provenance": provenance,
            },
        }

    def _source_prompt_rows(
        self, prompt_rows: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Tokenize one source-real prompt-only 2+2 B2 batch."""

        from src.opd.calibration_data import contains_forbidden_supervision

        if len(prompt_rows) != 4:
            raise ProductionTwoStepQualificationV6Error(
                "B2 requires exactly four source-real prompts"
            )
        counts = {
            source: sum(row.get("target_role") == source for row in prompt_rows)
            for source in ("medical_opd_o1", "medical_opd_cmb")
        }
        if counts != {"medical_opd_o1": 2, "medical_opd_cmb": 2}:
            raise ProductionTwoStepQualificationV6Error(
                "B2 prompt batch is not 2 O1 plus 2 CMB"
            )
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in prompt_rows:
            source_role = row.get("target_role")
            if contains_forbidden_supervision(row) or not isinstance(
                source_role, str
            ) or any(
                marker in source_role
                for marker in ("final", "controller", "confirmation")
            ):
                raise ProductionTwoStepQualificationV6Error(
                    "B2 prompt row contains forbidden supervision"
                )
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ProductionTwoStepQualificationV6Error(
                    "B2 prompt identity is absent or duplicated"
                )
            _digest(content_hash, "B2 prompt content hash")
            seen.add(sample_id)
            prompt = self.render_prompt_text(row)
            result.append(
                {
                    "fixture_id": sample_id,
                    "source_role": source_role,
                    "source_sample_id": sample_id,
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

    def run_b2_calibration_step(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
        from_version: int,
        authority_sha256: str,
    ) -> Mapping[str, Any]:
        """Run one on-policy B2 update through the same stable-slot sampler."""

        step_started = time.perf_counter()
        if not (
            self.route in {"b2", "b2_calibration"}
            and isinstance(step_index, int)
            and not isinstance(step_index, bool)
            and 0 <= step_index < 20
            and from_version
            == self.current_sampler_version
            == step_index + (2 if self.route == "b2" else 0)
            and max_new_tokens == int(self.config["micro_replay"]["max_new_tokens"])
        ):
            raise ProductionTwoStepQualificationV6Error(
                "B2 step/version/generation envelope drift"
            )
        current = self.authorities[from_version]
        if _digest(authority_sha256, "B2 input authority") != current.get(
            "aggregate_tensor_sha256"
        ):
            raise ProductionTwoStepQualificationV6Error(
                "B2 caller authority differs from trainer authority"
            )
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=from_version,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._identity_equal(runtime, current):
            raise ProductionTwoStepQualificationV6Error(
                "B2 sampler differs from trainer authority before generation"
            )
        for device_index in (0, 1):
            self.torch.cuda.reset_peak_memory_stats(device_index)
        tokenized = self._source_prompt_rows(list(prompt_rows))
        generation_started = time.perf_counter()
        generated, generation_evidence = self.guard_sampler_operation(
            authority=current,
            runtime_identity=runtime,
            request_identity=self._request(current),
            operation="generation",
            callback=lambda: self._generate_rows(
                self.sampler_model,
                tokenized,
                device="cuda:1",
                step_index=(step_index + 2 if self.route == "b2" else step_index),
            ),
        )
        generation_seconds = time.perf_counter() - generation_started
        if not isinstance(generated, list) or len(generated) != 4:
            raise ProductionTwoStepQualificationV6Error(
                "B2 generation did not return four trajectories"
            )
        q_values = [
            float(value)
            for row in generated
            for value in row.get("rollout_behavior_logprob", ())
        ]
        if not q_values or not all(math.isfinite(value) for value in q_values):
            raise ProductionTwoStepQualificationV6Error(
                "B2 behavior q is empty or non-finite"
            )
        from src.opd.production_length_gpu_backend_v7 import (
            detect_repetition_v7,
            validate_decoded_output_contract_v7,
        )

        safe_samples: list[dict[str, Any]] = []
        for row in generated:
            response_ids = [int(value) for value in row["response_ids"]]
            eos_seen = bool(row["eos_observed"])
            decoded = self.tokenizer.decode(
                response_ids, skip_special_tokens=False
            )
            unexpected_think = "<think>" in decoded.lower() or (
                "</think>" in decoded.lower()
            )
            finite = all(
                math.isfinite(float(value))
                for value in row["rollout_behavior_logprob"]
            )
            empty = not response_ids
            truncated = bool(
                not eos_seen and len(response_ids) == max_new_tokens
            )
            output_valid = validate_decoded_output_contract_v7(
                decoded, eos_seen=eos_seen
            )
            repetition = detect_repetition_v7(response_ids)
            safe_samples.append(
                {
                    "sample_id": str(row["fixture_id"]),
                    "content_hash": str(row["content_hash"]),
                    "source": str(row["source_role"]),
                    "prompt_tokens": len(row["prompt_ids"]),
                    "generated_tokens": len(response_ids),
                    "eos": eos_seen,
                    "truncated": truncated,
                    "finish_reason": "eos" if eos_seen else "length",
                    "invalid": bool(
                        not output_valid
                        or (not eos_seen and not truncated)
                    ),
                    "empty": empty,
                    "non_finite": not finite,
                    "unexpected_think_tag": unexpected_think,
                    "repetition": repetition,
                }
            )
        unhealthy = [
            row
            for row in safe_samples
            if any(
                bool(row[field])
                for field in (
                    "invalid",
                    "empty",
                    "non_finite",
                    "unexpected_think_tag",
                    "repetition",
                )
            )
        ]
        if unhealthy:
            _atomic_json(
                self.output
                / "steps"
                / f"generation_health_failure_step_{step_index + 1:02d}.json",
                {
                    "schema_version": 1,
                    "artifact_kind": "b2_generation_health_failure_v1",
                    "run_id": self.config["run"]["run_id"],
                    "optimizer_step": step_index + 1,
                    "policy_version": from_version,
                    "prompt_samples": safe_samples,
                    "optimizer_executed": False,
                    "raw_prompt_persisted": False,
                    "response_tokens_persisted": False,
                    "isolation": dict(self.config["isolation"]),
                },
            )
            raise ProductionTwoStepQualificationV6Error(
                "B2 generation health contract failed before optimizer"
            )
        self.probe_rows = [
            {
                **row,
                "response_ids": list(row["response_ids"][:32]),
            }
            for row in generated
        ]
        if any(not row["response_ids"] for row in self.probe_rows):
            raise ProductionTwoStepQualificationV6Error(
                "B2 pre-update fixed-action probe is empty"
            )
        provenance = self._provenance(
            generated,
            authority=current,
            step_index=from_version,
        )
        self.validate_rollout_behavior_provenance(
            provenance,
            expected_sampler_adapter_sha256=authority_sha256,
            expected_trajectory_run_id=self.config["run"]["run_id"],
        )
        rollout = {
            "policy_version": f"v{from_version}",
            "tensor_sha256": authority_sha256,
            "rows": generated,
            "provenance": provenance,
        }
        updated = self.run_corrected_step(from_version, rollout)
        reconstruction, authority_artifact, target, checkpoint = _step_result(
            updated,
            step=from_version,
        )
        self.release_step_teacher(from_version)
        target_sha = _digest(
            target.get("aggregate_tensor_sha256"), "B2 target authority"
        )
        if target_sha == authority_sha256:
            raise ProductionTwoStepQualificationV6Error(
                "B2 optimizer did not change adapter identity"
            )
        refresh = _mapping(
            self.hotswap_stable_slot(
                current_authority=current,
                target_authority=target,
                checkpoint=checkpoint,
            ),
            "B2 refresh",
        )
        same_path = _mapping(refresh.get("same_path"), "B2 same-path evidence")
        normal = _mapping(refresh.get("normal_request"), "B2 normal request")
        stale = _mapping(refresh.get("stale_request"), "B2 stale request")
        runtime_sha = _digest(
            refresh.get("runtime_tensor_sha256"), "B2 runtime target"
        )
        fresh_sha = _digest(
            refresh.get("fresh_tensor_sha256"), "B2 fresh target"
        )
        if not (
            runtime_sha == fresh_sha == target_sha
            and float(same_path.get("max"))
            <= float(self.config["validation"]["same_path_max_gap"])
            and float(same_path.get("finite_rate")) == 1.0
            and normal.get("accepted") is True
            and stale.get("rejected") is True
            and stale.get("error_code") == "STALE_SAMPLER_IDENTITY"
            and refresh.get("registry_after") == [PRODUCTION_SLOT]
        ):
            raise ProductionTwoStepQualificationV6Error(
                "B2 refreshed identity/same-path/guard gate failed"
            )
        self._last_b2_refresh = dict(refresh)
        telemetry = _mapping(
            reconstruction.get("telemetry"), "B2 reconstruction telemetry"
        )
        update = _mapping(
            telemetry.get("optimizer_update"), "B2 optimizer telemetry"
        )
        artifact = {
            "schema_version": 1,
            "run_id": self.config["run"]["run_id"],
            "step_index": step_index,
            "from_version": from_version,
            "to_version": from_version + 1,
            "input_authority_tensor_sha256": authority_sha256,
            "trainer_authority_tensor_sha256": target_sha,
            "authority_artifact": authority_artifact,
            "reconstruction_telemetry": telemetry,
            "refresh": {
                "runtime_tensor_sha256": runtime_sha,
                "fresh_tensor_sha256": fresh_sha,
                "same_path": dict(same_path),
                "normal_request": dict(normal),
                "stale_request": dict(stale),
                "registry_after": list(refresh["registry_after"]),
            },
            "generation_guard": dict(generation_evidence),
            "prompt_count": 4,
            "source_counts": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
            "raw_prompt_persisted": False,
            "response_tokens_persisted": False,
            "full_logits_persisted": False,
            "isolation": dict(self.config["isolation"]),
        }
        artifact_path = (
            self.output
            / "b2_steps"
            / f"step_{step_index:02d}_v{from_version}_to_v{from_version + 1}.json"
        )
        if artifact_path.exists() or artifact_path.is_symlink():
            raise ProductionTwoStepQualificationV6Error(
                "B2 step artifact already exists"
            )
        artifact_sha = _atomic_json(artifact_path, artifact)
        private = self._last_b2_step_private
        if not isinstance(private, dict):
            raise ProductionTwoStepQualificationV6Error(
                "B2 numeric/timing evidence is absent"
            )
        private.update(
            {
                "prompt_samples": safe_samples,
                "rollout_provenance_sha256": _canonical_sha(provenance),
                "generation_seconds": generation_seconds,
                "sampler_refresh_seconds": float(
                    refresh["refresh_latency_seconds"]
                ),
                "step_seconds": time.perf_counter() - step_started,
                "rollout_tokens_per_second": (
                    len(q_values) / generation_seconds
                    if generation_seconds > 0
                    else 0.0
                ),
                "scorer_tokens_per_second": (
                    int(private["valid_token_count"])
                    / float(private["scoring_seconds"])
                    if float(private["scoring_seconds"]) > 0
                    else 0.0
                ),
                "gpu_memory_bytes": {
                    "gpu0_allocated": int(self.torch.cuda.memory_allocated(0)),
                    "gpu0_reserved": int(self.torch.cuda.memory_reserved(0)),
                    "gpu0_peak": int(self.torch.cuda.max_memory_allocated(0)),
                    "gpu1_allocated": int(self.torch.cuda.memory_allocated(1)),
                    "gpu1_reserved": int(self.torch.cuda.memory_reserved(1)),
                    "gpu1_peak": int(self.torch.cuda.max_memory_allocated(1)),
                },
                "disk_remaining_bytes": int(shutil.disk_usage(self.output).free),
            }
        )
        return {
            "step_index": step_index,
            "from_version": from_version,
            "to_version": from_version + 1,
            "generated_by_policy_version": from_version,
            "p_old_policy_version": from_version,
            "input_authority_tensor_sha256": authority_sha256,
            "trainer_authority_tensor_sha256": target_sha,
            "runtime_tensor_sha256": runtime_sha,
            "fresh_tensor_sha256": fresh_sha,
            "active_slot": PRODUCTION_SLOT,
            "registry_count": len(refresh["registry_after"]),
            "same_path_max_gap": float(same_path["max"]),
            "finite_rate": float(same_path["finite_rate"]),
            "delta_j": float(update["objective_delta"]),
            "delta_l": float(update["loss_delta"]),
            "alignment": float(update["alignment"]),
            "telemetry_complete": True,
            "normal_request_accepted": True,
            "stale_previous_rejected": True,
            "stale_error_code": "STALE_SAMPLER_IDENTITY",
            "checkpoint_path": str((self.output / checkpoint).resolve()),
            "step_artifact_sha256": artifact_sha,
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }

    def run_b2_calibration_step_v1(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> Mapping[str, Any]:
        """Return one strict, privacy-safe P4.8 step record."""

        if self.route != "b2_calibration":
            raise ProductionTwoStepQualificationV6Error(
                "P4.8 step API is unavailable on the qualification-v2 route"
            )
        from src.opd.production_b2_calibration_backend_v1 import (
            build_b2_step_record_v1,
        )

        version = self.current_sampler_version
        authority_sha = _digest(
            self.authorities[version].get("aggregate_tensor_sha256"),
            "P4.8 input authority",
        )
        raw = self.run_b2_calibration_step(
            step_index=step_index,
            prompt_rows=prompt_rows,
            max_new_tokens=max_new_tokens,
            from_version=version,
            authority_sha256=authority_sha,
        )
        private = self._last_b2_step_private
        if not isinstance(private, Mapping):
            raise ProductionTwoStepQualificationV6Error(
                "P4.8 step metrics were not retained"
            )
        evidence = {
            "run_id": self.config["run"]["run_id"],
            "optimizer_step": step_index + 1,
            "policy_version": version,
            "next_policy_version": version + 1,
            "generated_by_policy_version": version,
            "p_old_policy_version": version,
            "sampler_adapter_sha256": authority_sha,
            "input_trainer_authority_sha256": authority_sha,
            "trainer_authority_sha256": raw[
                "trainer_authority_tensor_sha256"
            ],
            "runtime_adapter_sha256": raw["runtime_tensor_sha256"],
            "fresh_adapter_sha256": raw["fresh_tensor_sha256"],
            "rollout_provenance_sha256": private[
                "rollout_provenance_sha256"
            ],
            "prompt_samples": list(private["prompt_samples"]),
            "q_logprob": dict(private["q_logprob"]),
            "p_old_logprob": dict(private["p_old_logprob"]),
            "teacher_logprob": dict(private["teacher_logprob"]),
            "valid_token_count": int(private["valid_token_count"]),
            "reverse_kl": dict(private["reverse_kl"]),
            "advantage": dict(private["advantage"]),
            "importance_ratio": dict(private["importance_ratio"]),
            "ppo_clip_fraction": float(private["ppo_clip_fraction"]),
            "ess_fraction": float(private["ess_fraction"]),
            "objective": float(private["objective"]),
            "loss": float(private["loss"]),
            "gradient_norm": float(private["gradient_norm"]),
            "nonzero_update_tensor_count": int(
                private["nonzero_update_tensor_count"]
            ),
            "zero_update_tensor_count": int(private["zero_update_tensor_count"]),
            "teacher_gradient_tensor_count": int(
                private["teacher_gradient_tensor_count"]
            ),
            "base_gradient_tensor_count": int(
                private["base_gradient_tensor_count"]
            ),
            "adapter_delta_norm": float(private["adapter_delta_norm"]),
            "teacher_same_token_scoring": True,
            "teacher_generated_completion": False,
            "p_old_detached": True,
            "normal_request_accepted": bool(raw["normal_request_accepted"]),
            "stale_policy_rejected_before_forward": bool(
                raw["stale_previous_rejected"]
            ),
            "stale_error_code": str(raw["stale_error_code"]),
            "sampler_refresh_seconds": float(
                private["sampler_refresh_seconds"]
            ),
            "timings_seconds": {
                "generation": float(private["generation_seconds"]),
                "scoring": float(private["scoring_seconds"]),
                "backward": float(private["backward_seconds"]),
                "checkpoint": float(private["checkpoint_seconds"]),
                "step": float(private["step_seconds"]),
            },
            "throughput": {
                "rollout_tokens_per_second": float(
                    private["rollout_tokens_per_second"]
                ),
                "scorer_tokens_per_second": float(
                    private["scorer_tokens_per_second"]
                ),
            },
            "gpu_memory_bytes": dict(private["gpu_memory_bytes"]),
            "disk_remaining_bytes": int(private["disk_remaining_bytes"]),
            "checkpoint": {
                "logical_version": version + 1,
                "path": str(raw["checkpoint_path"]),
                "adapter_sha256": raw["trainer_authority_tensor_sha256"],
                "complete": True,
                "resume_eligible": version + 1 in {10, 20},
            },
            "isolation": dict(self.config["isolation"]),
        }
        return build_b2_step_record_v1(
            evidence, selected_response_length=max_new_tokens
        )

    def _atomic_torch_save(self, path: Path, value: Any) -> None:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            self.torch.save(value, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def save_b2_resume_checkpoint_v1(
        self,
        *,
        logical_version: int,
        package_content_sha256: str,
        config_sha256: str,
        data_cursor: int,
    ) -> Mapping[str, Any]:
        """Add optimizer/RNG/cursor state to a complete v10 or v20 adapter."""

        if not (
            self.route == "b2_calibration"
            and logical_version in {10, 20}
            and logical_version == self.current_sampler_version
            and data_cursor == logical_version * 4
        ):
            raise ProductionTwoStepQualificationV6Error(
                "resume checkpoint version/cursor is not eligible"
            )
        package_sha = _digest(package_content_sha256, "P4.8 package content")
        runtime_config_sha = _digest(config_sha256, "P4.8 runtime config")
        checkpoint = self.output / "checkpoints" / f"v{logical_version}"
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ProductionTwoStepQualificationV6Error(
                "adapter checkpoint is absent before resume sealing"
            )
        self._atomic_torch_save(
            checkpoint / "optimizer_state.pt", self.optimizer.state_dict()
        )
        self._atomic_torch_save(
            checkpoint / "rng_state.pt",
            {
                "cpu": self.torch.get_rng_state(),
                "cuda": self.torch.cuda.get_rng_state_all(),
            },
        )
        state = {
            "optimizer_step": logical_version,
            "policy_version": logical_version,
            "data_cursor": data_cursor,
            "package_content_sha256": package_sha,
            "config_sha256": runtime_config_sha,
        }
        _atomic_json(checkpoint / "calibration_state.json", state)
        file_names = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer_state.pt",
            "rng_state.pt",
            "calibration_state.json",
        )
        files = {
            name: {
                "sha256": _sha_file(checkpoint / name),
                "size_bytes": (checkpoint / name).stat().st_size,
            }
            for name in file_names
        }
        authority_sha = _digest(
            self.authorities[logical_version].get("aggregate_tensor_sha256"),
            "resume authority",
        )
        manifest = {
            "schema_version": 1,
            "artifact_kind": "b2_resume_checkpoint_manifest_v1",
            "run_id": self.config["run"]["run_id"],
            "logical_version": logical_version,
            "adapter_sha256": authority_sha,
            "optimizer_step": logical_version,
            "sampler_policy_version": logical_version,
            "package_content_sha256": package_sha,
            "config_sha256": runtime_config_sha,
            "complete": True,
            "resume_eligible": True,
            "files": files,
        }
        _atomic_json(checkpoint / "checkpoint_manifest.json", manifest)
        return manifest

    def reload_b2_resume_checkpoint_v1(
        self,
        *,
        logical_version: int,
        package_content_sha256: str,
        config_sha256: str,
        data_cursor: int,
    ) -> Mapping[str, Any]:
        """Exercise a real v10 adapter/optimizer/RNG save-and-reload boundary."""

        if not (
            self.route == "b2_calibration"
            and logical_version == self.current_sampler_version == 10
            and data_cursor == 40
        ):
            raise ProductionTwoStepQualificationV6Error(
                "only the frozen v10 midpoint is reload eligible"
            )
        checkpoint = self.output / "checkpoints" / "v10"
        manifest_path = checkpoint / "checkpoint_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = json.loads(
                (checkpoint / "calibration_state.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProductionTwoStepQualificationV6Error(
                f"resume checkpoint metadata invalid: {type(error).__name__}"
            ) from error
        files = _mapping(manifest.get("files"), "resume checkpoint files")
        required = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer_state.pt",
            "rng_state.pt",
            "calibration_state.json",
        }
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("artifact_kind")
            == "b2_resume_checkpoint_manifest_v1"
            and manifest.get("run_id") == self.config["run"]["run_id"]
            and manifest.get("logical_version") == 10
            and manifest.get("optimizer_step") == 10
            and manifest.get("sampler_policy_version") == 10
            and manifest.get("complete") is True
            and manifest.get("resume_eligible") is True
            and set(files) == required
            and state
            == {
                "optimizer_step": 10,
                "policy_version": 10,
                "data_cursor": 40,
                "package_content_sha256": _digest(
                    package_content_sha256, "resume package"
                ),
                "config_sha256": _digest(config_sha256, "resume config"),
            }
        ):
            raise ProductionTwoStepQualificationV6Error(
                "resume checkpoint manifest/state contract drift"
            )
        for name, descriptor in files.items():
            path = checkpoint / name
            if not (
                isinstance(descriptor, Mapping)
                and path.is_file()
                and not path.is_symlink()
                and descriptor.get("sha256") == _sha_file(path)
                and descriptor.get("size_bytes") == path.stat().st_size
            ):
                raise ProductionTwoStepQualificationV6Error(
                    "resume checkpoint file SHA/size mismatch"
                )
        target = self.authorities[10]
        expected_sha = _digest(
            target.get("aggregate_tensor_sha256"), "v10 resume authority"
        )
        if manifest.get("adapter_sha256") != expected_sha:
            raise ProductionTwoStepQualificationV6Error(
                "resume checkpoint adapter differs from trainer authority"
            )
        saved = self.adapter_artifact_identity(
            checkpoint,
            logical_version=10,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not self._identity_equal(saved, target):
            raise ProductionTwoStepQualificationV6Error(
                "resume checkpoint transport identity differs"
            )
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
        self.parameters = {}
        self.trainable_names = ()
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
        self.student_model.eval()
        self.parameters = dict(self.student_model.named_parameters())
        self.trainable_names = tuple(
            name
            for name, parameter in self.parameters.items()
            if parameter.requires_grad
        )
        if len(self.trainable_names) != 504 or any(
            "lora" not in name.lower() for name in self.trainable_names
        ):
            raise ProductionTwoStepQualificationV6Error(
                "resumed Student trainable scope differs"
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
        optimizer_state = self.torch.load(
            checkpoint / "optimizer_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.optimizer.load_state_dict(optimizer_state)
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
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del sampler_base
        self.sampler_model.eval()
        trainer = self.build_adapter_identity_manifest(
            {name: self.parameters[name] for name in self.trainable_names},
            adapter_config=self.student_model.peft_config["default"],
            adapter_logical_version=10,
            adapter_runtime_name="default",
            active_adapter="default",
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        runtime = self.runtime_identity_from_peft(
            self.sampler_model,
            logical_version=10,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        if not (
            self._identity_equal(trainer, target)
            and self._identity_equal(runtime, target)
        ):
            raise ProductionTwoStepQualificationV6Error(
                "v10 resume trainer/runtime identity differs"
            )
        rng = self.torch.load(
            checkpoint / "rng_state.pt", map_location="cpu", weights_only=True
        )
        if not (
            isinstance(rng, Mapping)
            and isinstance(rng.get("cuda"), list)
            and len(rng["cuda"]) == 2
        ):
            raise ProductionTwoStepQualificationV6Error(
                "resume RNG state is incomplete"
            )
        self.torch.set_rng_state(rng["cpu"])
        self.torch.cuda.set_rng_state_all(rng["cuda"])
        self.current_sampler_runtime = runtime
        self._current_checkpoint_path = checkpoint.resolve()
        return {
            "schema_version": 1,
            "artifact_kind": "b2_resume_reload_identity_v1",
            "run_id": self.config["run"]["run_id"],
            "logical_version": 10,
            "trainer_adapter_sha256": trainer["aggregate_tensor_sha256"],
            "runtime_adapter_sha256": runtime["aggregate_tensor_sha256"],
            "checkpoint_adapter_sha256": saved["aggregate_tensor_sha256"],
            "optimizer_state_restored": True,
            "rng_state_restored": True,
            "data_cursor": 40,
            "tensor_count": int(runtime["tensor_count"]),
        }

    def final_checkpoint_reload_identity_v1(self) -> Mapping[str, Any]:
        """Release live runtime and independently reload the final v20 adapter."""

        if self.route != "b2_calibration" or self.current_sampler_version != 20:
            raise ProductionTwoStepQualificationV6Error(
                "final calibration reload requires authoritative v20"
            )
        target = self.authorities[20]
        target_sha = _digest(
            target.get("aggregate_tensor_sha256"), "final trainer authority"
        )
        runtime = _mapping(self.current_sampler_runtime, "final runtime identity")
        refresh = _mapping(self._last_b2_refresh, "final refresh evidence")
        checkpoint = self.output / "checkpoints" / "v20"
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
        self.parameters = {}
        self.trainable_names = ()
        base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        fresh = self.PeftModel.from_pretrained(
            base,
            checkpoint,
            adapter_name=PRODUCTION_SLOT,
            is_trainable=False,
        )
        del base
        fresh.eval()
        fresh_identity = self.runtime_identity_from_peft(
            fresh,
            logical_version=20,
            runtime_name=PRODUCTION_SLOT,
            base_revision=self.base_revision,
            tokenizer_revision=self.tokenizer_revision,
        )
        self._release(self.torch, fresh)
        if not (
            self._identity_equal(target, runtime)
            and self._identity_equal(target, fresh_identity)
            and target_sha == runtime.get("aggregate_tensor_sha256")
            == fresh_identity.get("aggregate_tensor_sha256")
        ):
            raise ProductionTwoStepQualificationV6Error(
                "final v20 trainer/runtime/fresh reload identity mismatch"
            )
        same_path = _mapping(refresh.get("same_path"), "final same-path evidence")
        return {
            "schema_version": 1,
            "artifact_kind": "b2_final_checkpoint_reload_identity_v1",
            "run_id": self.config["run"]["run_id"],
            "logical_version": 20,
            "trainer_adapter_sha256": target_sha,
            "runtime_adapter_sha256": runtime["aggregate_tensor_sha256"],
            "fresh_adapter_sha256": fresh_identity["aggregate_tensor_sha256"],
            "tensor_count": int(fresh_identity["tensor_count"]),
            "same_path_max_gap": float(same_path["max"]),
            "finite_rate": float(same_path["finite_rate"]),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._release(
                self.torch,
                self.teacher_model,
                self.student_model,
                self.sampler_model,
            )
        finally:
            self.teacher_model = None
            self.student_model = None
            self.sampler_model = None
            self.optimizer = None
            self.parameters = {}
            self.trainable_names = ()
            try:
                self.temporary.cleanup()
            except FileNotFoundError:
                pass
            self._closed = True


def _v0_authority_artifact_projection(value: Any) -> dict[str, Any]:
    """Project the trainer-derived v0 authority into durable guard evidence."""

    authority = _mapping(value, "v0 authority")
    logical_version = authority.get(
        "adapter_logical_version", authority.get("logical_version")
    )
    tensors = authority.get("tensors")
    if logical_version not in {0, "v0"} or not isinstance(tensors, list) or not tensors:
        raise ProductionTwoStepQualificationV6Error("v0 authority identity is incomplete")
    per_tensor: list[dict[str, Any]] = []
    for item in tensors:
        tensor = _mapping(item, "v0 authority tensor")
        per_tensor.append(
            {
                "canonical_key": tensor.get("canonical_key"),
                "sha256": _digest(tensor.get("sha256"), "v0 authority tensor"),
                "shape": list(tensor.get("shape", [])),
                "dtype": tensor.get("source_dtype"),
                "byte_length": tensor.get("canonical_byte_length"),
            }
        )
    runtime_name = authority.get(
        "production_runtime_name", authority.get("adapter_runtime_name")
    )
    revisions = {
        "base_revision": authority.get("base_revision"),
        "tokenizer_revision": authority.get("tokenizer_revision"),
    }
    if any(
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in revisions.values()
    ):
        raise ProductionTwoStepQualificationV6Error(
            "v0 authority base/tokenizer revision is invalid"
        )
    return {
        "logical_version": "v0",
        "runtime_adapter_name": runtime_name,
        "active_adapter": authority.get("active_adapter"),
        "canonical_config_sha256": _digest(
            authority.get("canonical_config_sha256"), "v0 authority config"
        ),
        "aggregate_tensor_sha256": _digest(
            authority.get("aggregate_tensor_sha256"), "v0 authority tensor aggregate"
        ),
        "per_tensor_digests": per_tensor,
        "tensor_count": authority.get("tensor_count"),
        "total_bytes": authority.get("total_canonical_bytes"),
        "base_revision": revisions["base_revision"],
        "tokenizer_revision": revisions["tokenizer_revision"],
        "immutable_manifest_sha256": _digest(
            authority.get("artifact_manifest_sha256"),
            "v0 authority immutable manifest",
        ),
    }


def execute_two_step_qualification_v6(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    emit: EmitPhase,
    micro_gate: MicroEvidenceGate | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Execute exactly v0 -> v1 -> guarded rollout-1 -> v2.

    The same session owns one long-lived sampler for both hotswaps and the
    intervening generation.  A failed identity edge stops before the next
    optimizer step.  The caller owns the outer failure/index/cleanup graph;
    this function always releases its model session.
    """

    _validate_contract(config)
    factory = session_factory or create_production_two_step_session_v6
    session = factory(config, config_path=Path(config_path))
    try:
        prepared = _mapping(session.prepare_micro_evidence(), "micro evidence")
        probe = _mapping(prepared.get("probe_manifest"), "probe manifest")
        v0_guard = dict(_mapping(prepared.get("v0_guard"), "v0 guard"))
        v0_guard["v0_authority"] = _v0_authority_artifact_projection(
            prepared.get("v0_authority")
        )
        rollout0 = _mapping(prepared.get("rollout"), "rollout0")
        if rollout0.get("policy_version") not in {0, "v0"}:
            raise ProductionTwoStepQualificationV6Error(
                "rollout0 was not generated by v0"
            )
        _emit(
            emit,
            "probe_manifest",
            probe,
            {"status": "pass", "total_probe_count": probe.get("total_probe_count", 1)},
        )
        _emit(
            emit,
            "v0_guard",
            v0_guard,
            {"status": "pass", "normal_v0": True, "wrong_authority_rejected": True},
        )

        step0 = session.run_corrected_step(0, rollout0)
        reconstruction0, authority1_artifact, authority1, checkpoint1 = _step_result(
            step0, step=0
        )
        v1_sha = _digest(
            authority1["aggregate_tensor_sha256"], "trainer-authoritative v1"
        )
        _emit(
            emit,
            "reconstruction_step0",
            reconstruction0,
            {"status": "pass", "logical_transition": "v0_to_v1"},
        )
        release_teacher = getattr(session, "release_step_teacher", None)
        if callable(release_teacher):
            release_teacher(0)
        _emit(
            emit,
            "authority_v1",
            authority1_artifact,
            {"status": "pass", "aggregate_tensor_sha256": v1_sha},
        )
        refresh1 = _mapping(
            session.hotswap_stable_slot(
                current_authority=_mapping(
                    prepared.get("v0_authority"), "v0 authority"
                ),
                target_authority=authority1,
                checkpoint=checkpoint1,
            ),
            "refresh v1",
        )
        if _digest(refresh1.get("runtime_tensor_sha256"), "runtime v1") != v1_sha:
            raise ProductionTwoStepQualificationV6Error(
                "runtime v1 differs from trainer authority"
            )
        refresh1_sha = _emit(
            emit,
            "refresh_v1",
            refresh1,
            {"status": "pass", "runtime_tensor_sha256": v1_sha},
        )

        if micro_gate is None:
            raise ProductionTwoStepQualificationV6Error(
                "artifact-derived micro gate callback is absent"
            )
        try:
            micro_readiness = _mapping(
                micro_gate(
                    expected_v1_tensor_sha256=v1_sha,
                    expected_refresh_v1_sha256=refresh1_sha,
                ),
                "artifact-derived micro gate",
            )
        except ProductionTwoStepQualificationV6Error:
            raise
        except Exception as error:
            raise ProductionTwoStepQualificationV6Error(
                f"artifact-derived micro gate rejected refresh_v1: {error}"
            ) from error
        if not (
            micro_readiness.get("ready") is True
            and _digest(
                micro_readiness.get("v1_tensor_sha256"),
                "artifact-derived micro v1",
            )
            == v1_sha
            and _digest(
                micro_readiness.get("refresh_v1_artifact_sha256"),
                "artifact-derived micro refresh_v1",
            )
            == refresh1_sha
        ):
            raise ProductionTwoStepQualificationV6Error(
                "artifact-derived micro gate did not pass"
            )
        _digest(
            micro_readiness.get("micro_readiness_sha256"),
            "artifact-derived micro readiness",
        )

        generated = _mapping(
            session.generate_guarded_rollout(
                1,
                authority=authority1,
                refresh_artifact_sha256=refresh1_sha,
            ),
            "guarded rollout1",
        )
        trajectory1 = _mapping(generated.get("manifest"), "trajectory1 manifest")
        rollout1 = _mapping(generated.get("rollout"), "rollout1")
        if not (
            trajectory1.get("generated_by_policy_version") == "v1"
            and trajectory1.get("p_old_policy_version") == "v1"
            and trajectory1.get("sampler_tensor_sha256") == v1_sha
            and trajectory1.get("p_old_actor_tensor_sha256") == v1_sha
            and trajectory1.get("refresh_artifact_sha256") == refresh1_sha
            and rollout1.get("policy_version") == "v1"
            and rollout1.get("tensor_sha256") == v1_sha
        ):
            raise ProductionTwoStepQualificationV6Error(
                "rollout1/p_old identity is not trainer-authoritative v1"
            )
        _emit(
            emit,
            "trajectory_step1_manifest",
            trajectory1,
            {"status": "pass", "generated_by_policy_version": "v1"},
        )

        step1 = session.run_corrected_step(1, rollout1)
        reconstruction1, authority2_artifact, authority2, checkpoint2 = _step_result(
            step1, step=1
        )
        v2_sha = _digest(
            authority2["aggregate_tensor_sha256"], "trainer-authoritative v2"
        )
        _emit(
            emit,
            "reconstruction_step1",
            reconstruction1,
            {"status": "pass", "logical_transition": "v1_to_v2"},
        )
        if callable(release_teacher):
            release_teacher(1)
        if v2_sha == v1_sha:
            raise ProductionTwoStepQualificationV6Error(
                "v2 tensor identity did not change from v1"
            )
        authority2_sha = _emit(
            emit,
            "authority_v2",
            authority2_artifact,
            {"status": "pass", "aggregate_tensor_sha256": v2_sha},
        )
        refresh2 = _mapping(
            session.hotswap_stable_slot(
                current_authority=authority1,
                target_authority=authority2,
                checkpoint=checkpoint2,
            ),
            "refresh v2",
        )
        if _digest(refresh2.get("runtime_tensor_sha256"), "runtime v2") != v2_sha:
            raise ProductionTwoStepQualificationV6Error(
                "runtime v2 differs from trainer authority"
            )
        _emit(
            emit,
            "refresh_v2",
            refresh2,
            {"status": "pass", "runtime_tensor_sha256": v2_sha},
        )
        sampler_id = getattr(session, "sampler_id", None)
        if not isinstance(sampler_id, int):
            raise ProductionTwoStepQualificationV6Error(
                "long-lived sampler diagnostic identity is absent"
            )
        return {
            "v1_tensor_sha256": v1_sha,
            "v2_tensor_sha256": v2_sha,
            "refresh_v1_artifact_sha256": refresh1_sha,
            "authority_v2_artifact_sha256": authority2_sha,
            "checkpoint_v1": checkpoint1,
            "checkpoint_v2": checkpoint2,
            "sampler_session_proof": {
                "same_long_lived_sampler": True,
                "sampler_object_id_diagnostic": str(sampler_id),
                "rollout1_generated_by": "v1",
                "p_old1_policy_version": "v1",
                "stable_slot": PRODUCTION_SLOT,
            },
        }
    finally:
        session.close()


def create_production_two_step_session_v6(
    config: Mapping[str, Any], *, config_path: Path
) -> TwoStepSessionV6:
    """Create the authorized real session with imports delayed to this call."""

    return ProductionTwoStepSessionV6(config, config_path=Path(config_path))


def _b2_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the authorized calibration package onto the proven GPU kernel."""

    schema = (config.get("schema_id"), config.get("schema_version"))
    if schema not in {
        ("ca-opd/b2-medical-opd-calibration/v1", 1),
        ("ca-opd/b2-medical-opd-calibration/v2", 2),
        ("ca-opd/b2-medical-opd-calibration/v3", 3),
    }:
        raise ProductionTwoStepQualificationV6Error("B2 calibration schema drift")
    run = _mapping(config.get("run"), "B2 run")
    backend = _mapping(config.get("production_backend"), "B2 backend")
    model = _mapping(config.get("model"), "B2 model")
    teacher = _mapping(config.get("teacher"), "B2 teacher")
    protocol = _mapping(config.get("protocol"), "B2 protocol")
    generation = _mapping(config.get("generation"), "B2 generation")
    qualification = _mapping(config.get("qualification"), "B2 qualification")
    authorization = _mapping(config.get("authorization"), "B2 authorization")
    execution = _mapping(config.get("execution"), "B2 execution")
    if not (
        run.get("seed") == 42
        and run.get("optimizer_steps") == 20
        and backend.get("backend_id") == PRODUCTION_BACKEND_ID
        and backend.get("refresh_implementation") == PRODUCTION_REFRESH_MECHANISM
        and backend.get("adapter_runtime_slot") == PRODUCTION_SLOT
        and backend.get("dtype") == "bfloat16"
        and backend.get("attention_backend") == "eager"
        and generation.get("max_new_tokens") in P4_7_B2_ALLOWED_RESPONSE_LENGTHS
        and generation.get("do_sample") is True
        and generation.get("temperature") == 1.0
        and generation.get("top_k") == 0
        and generation.get("top_p") == 1.0
        and generation.get("full_support") is True
        and generation.get("enable_thinking") is False
        and generation.get("use_cache") is True
        and protocol.get("correction_upper_threshold") == 2.0
        and protocol.get("prompt_equal_reduction") is True
        and authorization.get("production_sampler_refresh_ready") is True
        and authorization.get("OPD_scoring_backend_ready") is True
        and authorization.get("B2_authorized") is True
        and authorization.get("B2_started") is False
        and execution.get("optimizer_steps") == 20
        and execution.get("calibration_only") is True
        and execution.get("automatically_start_b2") is False
        and config.get("isolation")
        == {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }
    ):
        raise ProductionTwoStepQualificationV6Error(
            "B2 production/correction/isolation contract drift"
        )
    formula_path = str(protocol.get("three_policy_formula_path", ""))
    formula_sha = _digest(
        protocol.get("three_policy_formula_sha256"), "B2 formula"
    )
    source = Path(formula_path)
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[2] / source
    if not source.is_file() or source.is_symlink() or _sha_file(source) != formula_sha:
        raise ProductionTwoStepQualificationV6Error("B2 formula SHA mismatch")
    for field, expected in (
        ("model_revision", 40),
        ("tokenizer_revision", 40),
    ):
        value = backend.get(field)
        if not (
            isinstance(value, str)
            and len(value) == expected
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ProductionTwoStepQualificationV6Error(
                f"B2 {field} is not immutable"
            )
    _digest(qualification.get("authority_v2_sha256"), "authority_v2 artifact")
    _digest(qualification.get("v2_tensor_sha256"), "v2 tensor authority")
    output = run.get("output_dir")
    if not isinstance(output, str) or not output:
        raise ProductionTwoStepQualificationV6Error("B2 output path is absent")
    transformer_generation = {
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "do_sample": True,
        "eos_token_id": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "force_words_ids": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "max_new_tokens": int(generation["max_new_tokens"]),
        "min_length": 0,
        "min_new_tokens": 0,
        "min_p": None,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "output_logits": True,
        "output_scores": True,
        "pad_token_id": None,
        "renormalize_logits": False,
        "repetition_penalty": 1.0,
        "return_dict_in_generate": True,
        "stop_strings": None,
        "suppress_tokens": None,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "typical_p": 1.0,
        "use_cache": True,
    }
    return {
        "schema_id": "ca-opd/p4.6-combined-production-qualification/v1",
        "schema_version": 1,
        "run": {
            "run_id": str(run["run_id"]),
            "seed": 42,
            "optimizer_steps": 20,
            "output_dir": output,
        },
        "production_binding": {"backend_id": PRODUCTION_BACKEND_ID},
        "sampler_refresh": {
            "candidate_mechanism": PRODUCTION_REFRESH_MECHANISM,
            "runtime_slot": PRODUCTION_SLOT,
        },
        "model": {
            "id": str(model["base_path"]),
            "revision": str(backend["model_revision"]),
            "tokenizer_revision": str(backend["tokenizer_revision"]),
        },
        "teacher": {"adapter_path": str(teacher["adapter_path"])},
        "validation": {
            "config_path": formula_path,
            "config_sha256": formula_sha,
            "ess_fraction_min": float(protocol["correction_ess_fraction_min"]),
            "cap_fraction_max": float(protocol["correction_cap_fraction_max"]),
            "same_path_max_gap": float(protocol["same_path_max_gap"]),
        },
        "formal_rollout": {
            "backend": "transformers",
            "backend_version": "4.56.2",
            "batch_size": 1,
            "transformers": transformer_generation,
        },
        "micro_replay": {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "prompt_count": 4,
        },
        "reconstruction_telemetry": {
            "advantage_near_zero_threshold": 1e-6,
        },
        "isolation": dict(config["isolation"]),
        "b2_protocol_binding": dict(protocol),
    }


def create_production_b2_session_v6(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    checkpoint_v2: Path,
    expected_v2_sha256: str,
) -> ProductionTwoStepSessionV6:
    """Restore the authorized qualification v2 into the production B2 loop."""

    runtime_config = _b2_runtime_config(config)
    qualification = _mapping(config.get("qualification"), "B2 qualification")
    configured_checkpoint = Path(str(qualification.get("v2_checkpoint_path", "")))
    checkpoint = Path(checkpoint_v2).resolve()
    if not configured_checkpoint.is_absolute():
        configured_checkpoint = (
            Path(config_path).resolve().parent / configured_checkpoint
        ).resolve()
    if checkpoint != configured_checkpoint or expected_v2_sha256 != qualification.get(
        "v2_tensor_sha256"
    ):
        raise ProductionTwoStepQualificationV6Error(
            "B2 factory input differs from the authorized v2 binding"
        )
    return ProductionTwoStepSessionV6(
        runtime_config,
        config_path=Path(config_path),
        route="b2",
        checkpoint_v2=checkpoint,
        expected_v2_sha256=expected_v2_sha256,
        qualification_authority_artifact_sha256=str(
            qualification["authority_v2_sha256"]
        ),
    )


def create_production_auxiliary_session_v6(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    route: str,
    checkpoint_v2: Path | None = None,
    expected_v2_sha256: str | None = None,
) -> ProductionTwoStepSessionV6:
    """Create one isolated Base-null or immutable-v2 length session.

    Heavy imports remain confined to ``ProductionTwoStepSessionV6.__init__``;
    merely importing either the qualification launcher or auxiliary contract
    stays CPU-only.
    """

    if route not in {"base_null", "length"}:
        raise ProductionTwoStepQualificationV6Error(
            "auxiliary session route must be base_null or length"
        )
    return ProductionTwoStepSessionV6(
        config,
        config_path=Path(config_path),
        route=route,
        checkpoint_v2=(None if checkpoint_v2 is None else Path(checkpoint_v2)),
        expected_v2_sha256=expected_v2_sha256,
    )


__all__ = [
    "PRODUCTION_BACKEND_ID",
    "PRODUCTION_REFRESH_MECHANISM",
    "PRODUCTION_SLOT",
    "ProductionTwoStepSessionV6",
    "ProductionTwoStepQualificationV6Error",
    "TwoStepSessionV6",
    "create_production_auxiliary_session_v6",
    "create_production_b2_session_v6",
    "create_production_two_step_session_v6",
    "execute_two_step_qualification_v6",
]
