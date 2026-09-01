"""GPU-only implementation of the frozen P4.3 three-policy protocol.

This module is imported only after the launcher authorization and CPU-safe
preflight. Model-runtime imports stay inside ``execute_three_policy_gpu_protocol``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class ThreePolicyGPURuntimeError(RuntimeError):
    pass


def execute_sampler_refresh_gpu_protocol_v4(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Execute the shared three-policy runtime under the P4.4 refresh contract.

    The heavy implementation remains below so CPU imports stay framework-free.
    These named checkpoints are the frozen v4 ordering consumed by source audit:
    sampler_v0_repeated_probe -> sampler_v0_noop_unload_reload_control ->
    trainer_v1_in_memory_vs_fresh_reload -> long_lived_sampler_v0_to_v1_refresh ->
    fresh_sampler_v1_reference -> live_refreshed_vs_fresh_same_path ->
    stale_v0_request_rejection -> generation_direct_cross_path_diagnostics.
    """

    if config.get("schema_version") != 4:
        raise ThreePolicyGPURuntimeError("P4.4 runtime requires schema version 4")
    from src.opd.sampler_refresh_contract import (
        assert_sampler_request_identity,
        build_sampler_refresh_report,
        persist_sampler_refresh_evidence,
    )

    # The imports are asserted here so the authorized runtime cannot silently
    # fall back to the P4.3 combined gate. The shared executor invokes all three
    # functions in its schema-v4 sampler branch before readiness.
    required_v4_functions = (
        assert_sampler_request_identity,
        build_sampler_refresh_report,
        persist_sampler_refresh_evidence,
    )
    if not all(callable(function) for function in required_v4_functions):
        raise ThreePolicyGPURuntimeError("P4.4 sampler contract is unavailable")
    return execute_three_policy_gpu_protocol(config, config_path=config_path)


def execute_production_sampler_micro_gpu_protocol_v5(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Run the frozen four-prompt v1 regeneration and production refresh micro.

    Heavy framework imports stay inside this authorized entrypoint.  The function
    deliberately does not run the 16/32 prompt ladder, Base null, B2, controller,
    confirmation, or final evaluation.
    """

    if (
        config.get("schema_version") != 5
        or config.get("run", {}).get("stage") != "production_sampler_refresh_micro_v5"
        or config.get("prompt_selection", {}).get("prompts") != 4
        or config.get("historical_v1", {}).get(
            "regenerate_with_minimal_four_prompt_one_step"
        )
        is not True
        or config.get("sampler_refresh", {}).get("candidate_mechanism")
        != "peft_0_17_1_hotswap_stable_slot"
    ):
        raise ThreePolicyGPURuntimeError("P4.5 production micro contract drift")

    import torch
    import yaml
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.opd.calibration_data import render_prompt_text, select_prompt_rows
    from src.opd.pg_opd_contract import (
        ThreePolicyLogProbBundle,
        decoupled_corrected_objective,
        grouped_trajectory_mean,
        validate_three_policy_bundle,
    )
    from src.opd.pg_opd_validation import atomic_write_json, audit_optimizer_update
    from src.opd.production_backend_binding_v5 import verify_b2_backend_binding
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
    from src.opd.scorer_gpu_calibration import (
        _apply_determinism,
        _release,
        ordered_adapter_sha256,
    )

    root = Path(__file__).resolve().parents[2]
    output = Path(config["run"]["output_dir"])
    required = {
        "launch_record.json",
        "config.json",
        "metadata.json",
        "metrics.jsonl",
        "stdout.log",
        "cost.json",
        "summary.json",
    }
    observed = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    if not output.is_dir() or not required.issubset(observed):
        raise ThreePolicyGPURuntimeError("P4.5 launch artifact envelope is incomplete")
    binding = verify_b2_backend_binding(
        config["production_binding"]["b2_config_path"],
        config["production_binding"]["b2_run_card_path"],
        repo_root=root,
    )
    if binding["production_backend"]["backend_id"] != config["production_binding"][
        "backend_id"
    ]:
        raise ThreePolicyGPURuntimeError("P4.5 production backend mismatch")
    protocol = yaml.safe_load(
        (root / str(config["validation"]["config_path"])).read_text(encoding="utf-8")
    )
    algorithm = protocol["algorithm"]
    optimizer_config = protocol["optimizer"]
    gates = protocol["calibration_gates"]
    _apply_determinism(torch)
    started = time.time()
    student_model = teacher_model = sampler_model = tokenizer = None

    def append_metric(step: int, phase: str, status: str, **values: Any) -> None:
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"step": step, "phase": phase, "status": status, **values},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def pad(values: Sequence[Any]) -> tuple[Any, Any]:
        result = torch.nn.utils.rnn.pad_sequence(
            [value.reshape(-1).to(device="cuda:0", dtype=torch.float32) for value in values],
            batch_first=True,
            padding_value=0.0,
        )
        mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(values):
            mask[index, : value.numel()] = True
        return result, mask

    def action_logprobs(
        model: Any,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        *,
        device: str,
    ) -> Any:
        combined = [int(value) for value in prompt_ids] + [int(value) for value in response_ids]
        ids = torch.tensor([combined], dtype=torch.long, device=device)
        result = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        )
        start = len(prompt_ids) - 1
        logits = result.logits[:, start : start + len(response_ids), :].float()
        targets = torch.tensor(response_ids, dtype=torch.long, device=device).view(1, -1, 1)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)

    def score_rows(
        model: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        device: str,
        inference: bool,
    ) -> tuple[Any, Any]:
        context = torch.inference_mode() if inference else nullcontext()
        with context:
            values = [
                action_logprobs(
                    model,
                    row["prompt_ids"],
                    row["response_ids"],
                    device=device,
                ).reshape(-1)
                for row in rows
            ]
        return pad(values)

    def generate_one(row: Mapping[str, Any], *, row_index: int) -> dict[str, Any]:
        prompt_ids = [int(value) for value in row["prompt_ids"]]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
        generation = dict(config["formal_rollout"]["transformers"])
        generation["eos_token_id"] = student_model.generation_config.eos_token_id
        generation["pad_token_id"] = student_model.generation_config.pad_token_id
        seed = int(config["run"]["seed"]) * 100_000 + int(row_index)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = student_model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                **generation,
            )
        response_ids = [
            int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()
        ]
        if not response_ids or len(generated.scores) != len(response_ids):
            raise ThreePolicyGPURuntimeError("P4.5 generation/token score alignment failed")
        behavior = [
            float(torch.log_softmax(score[0].float(), dim=-1)[token].detach().cpu())
            for token, score in zip(response_ids, generated.scores, strict=True)
        ]
        raw = [
            float(torch.log_softmax(logit[0].float(), dim=-1)[token].detach().cpu())
            for token, logit in zip(response_ids, generated.logits, strict=True)
        ]
        eos = generation["eos_token_id"]
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        del generated
        return {
            **dict(row),
            "response_ids": response_ids,
            "rollout_behavior_logprob": behavior,
            "raw_generation_logprob": raw,
            "seed": seed,
            "eos_observed": bool(response_ids[-1] in set(int(item) for item in eos)),
        }

    def provenance(rows: Sequence[Mapping[str, Any]], adapter_sha: str) -> dict[str, Any]:
        generation = dict(config["formal_rollout"]["transformers"])
        generation["eos_token_id"] = student_model.generation_config.eos_token_id
        generation["pad_token_id"] = student_model.generation_config.pad_token_id
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
            "trajectory_run_id": config["run"]["run_id"],
            "trajectory_kind": "fresh_full_support",
            "backend": "transformers",
            "backend_version": "4.56.2",
            "model_version": config["model"]["revision"],
            "adapter_version": adapter_sha,
            "generation_config": generation,
            "processor_warper_provenance": {
                "all_support_changing_processors_disabled": True,
                "active_logits_processor_warper_classes": [],
                "active_stopping_criteria_classes": ["EosTokenCriteria", "MaxLengthCriteria"],
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
            "seed": int(config["run"]["seed"]),
            "generator": "torch_cuda_default_generator_scoped_manual_seed_per_prompt",
            "sampler_adapter_version": 0,
            "sampler_adapter_sha256": adapter_sha,
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }

    def tensor_manifest_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        fields = (
            "aggregate_tensor_sha256",
            "tensor_count",
            "total_canonical_bytes",
            "base_revision",
            "tokenizer_revision",
        )
        tensor_fields = (
            "canonical_key",
            "sha256",
            "shape",
            "canonical_dtype",
            "canonical_byte_length",
        )
        return bool(
            all(left.get(field) == right.get(field) for field in fields)
            and [
                {field: item.get(field) for field in tensor_fields}
                for item in left.get("tensors", [])
            ]
            == [
                {field: item.get(field) for field in tensor_fields}
                for item in right.get("tensors", [])
            ]
        )

    def structural_config_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_config = dict(left.get("canonical_config", {}))
        right_config = dict(right.get("canonical_config", {}))
        # A trainable trainer and an immutable inference artifact legitimately
        # differ here; all structural/scaling fields remain independently bound.
        left_config.pop("inference_mode", None)
        right_config.pop("inference_mode", None)
        return left_config == right_config

    def manifest_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return bool(
            tensor_manifest_equal(left, right)
            and left.get("canonical_config_sha256")
            == right.get("canonical_config_sha256")
        )

    def gap_metrics(
        left: Sequence[Any], right: Sequence[Any], rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        records = []
        for row_index, (left_row, right_row, row) in enumerate(
            zip(left, right, rows, strict=True)
        ):
            left_values = left_row.detach().float().cpu().reshape(-1)
            right_values = right_row.detach().float().cpu().reshape(-1)
            if left_values.shape != right_values.shape:
                raise ThreePolicyGPURuntimeError("P4.5 fixed-action probe shape mismatch")
            for token_position, (first, second) in enumerate(
                zip(left_values.tolist(), right_values.tolist(), strict=True)
            ):
                records.append(
                    {
                        "sample_id": str(row["fixture_id"]),
                        "sample_index": row_index,
                        "token_position": token_position,
                        "left": float(first),
                        "right": float(second),
                        "abs_gap": abs(float(first) - float(second)),
                    }
                )
        gaps = torch.tensor([item["abs_gap"] for item in records], dtype=torch.float64)
        worst = max(records, key=lambda item: item["abs_gap"])
        return {
            "mae": float(gaps.mean()),
            "p50": float(torch.quantile(gaps, 0.50)),
            "p95": float(torch.quantile(gaps, 0.95)),
            "p99": float(torch.quantile(gaps, 0.99)),
            "max": float(gaps.max()),
            "finite_rate": float(torch.isfinite(gaps).to(torch.float64).mean()),
            "worst_token": worst,
            "threshold": float(config["sampler_refresh"]["max_same_path_gap"]),
        }

    try:
        prompt_config = config["prompt_selection"]
        o1 = select_prompt_rows(
            prompt_config["medical_opd_o1_path"],
            role="medical_opd_o1",
            count=2,
            seed=int(config["run"]["seed"]),
        )
        cmb = select_prompt_rows(
            prompt_config["medical_opd_cmb_path"],
            role="medical_opd_cmb",
            count=2,
            seed=int(config["run"]["seed"]),
        )
        selected = [item for pair in zip(o1, cmb, strict=True) for item in pair]
        tokenizer = AutoTokenizer.from_pretrained(
            str(config["model"]["id"]),
            local_files_only=True,
            revision=config["model"]["tokenizer_revision"],
        )
        prompt_rows = []
        sample_id_prefix = str(
            config.get("fixed_action_probe", {}).get("sample_id_prefix", "p4-5")
        )
        for index, row in enumerate(selected):
            prompt = render_prompt_text(row)
            prompt_rows.append(
                {
                    "fixture_id": f"{sample_id_prefix}-{index:02d}",
                    "source_role": row["target_role"],
                    "prompt_ids": [
                        int(value)
                        for value in tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    ],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
        probe_config = config.get("fixed_action_probe", {})
        if not isinstance(probe_config, Mapping):
            raise ThreePolicyGPURuntimeError("P4.6 fixed-action probe config is absent")
        probe_spec = build_probe_spec(
            run_id=config["run"]["run_id"],
            prompt_manifest_sha256=config["prompt_selection"]["opd_manifest_sha256"],
            ordered_sample_ids=tuple(str(row["fixture_id"]) for row in prompt_rows),
        )
        if (
            probe_config.get("selection_rule")
            != "first_32_valid_response_tokens_per_prompt_v1"
            or probe_config.get("probe_spec_sha256") != probe_spec["probe_spec_sha256"]
        ):
            raise ThreePolicyGPURuntimeError("P4.6 fixed-action probe spec drift")
        atomic_write_json(output / "probe_spec.json", probe_spec)

        model_path = str(config["model"]["id"])
        student_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
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
        student_model.eval()
        trainable_names = tuple(
            name for name, parameter in student_model.named_parameters() if parameter.requires_grad
        )
        if not trainable_names or any("lora" not in name.lower() for name in trainable_names):
            raise ThreePolicyGPURuntimeError("P4.5 Student trainable scope is not LoRA-only")
        parameters = dict(student_model.named_parameters())
        frozen_versions = {
            name: parameter._version
            for name, parameter in parameters.items()
            if name not in trainable_names
        }

        with tempfile.TemporaryDirectory(dir=output, prefix=".p4_5_v0_") as temporary:
            v0_path = Path(temporary) / "v0"
            student_model.save_pretrained(v0_path, safe_serialization=True)
            v0_file_sha = ordered_adapter_sha256(v0_path)
            trajectories = [
                generate_one(row, row_index=index) for index, row in enumerate(prompt_rows)
            ]
            probe_manifest = build_probe_manifest(
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
            atomic_write_json(output / "probe_manifest.json", probe_manifest)
            probe_counts = probe_manifest["per_prompt_count"]
            probe_rows = [
                {
                    **row,
                    "response_ids": row["response_ids"][
                        : int(probe_counts[str(row["fixture_id"])])
                    ],
                }
                for row in trajectories
            ]
            old_actor, old_mask = score_rows(
                student_model, trajectories, device="cuda:0", inference=True
            )
            for index, row in enumerate(trajectories):
                row["old_actor_logprob"] = old_actor[index, : len(row["response_ids"])].cpu().tolist()
            rollout_provenance = provenance(trajectories, v0_file_sha)
            validate_rollout_behavior_provenance(
                rollout_provenance,
                expected_sampler_adapter_sha256=v0_file_sha,
                expected_trajectory_run_id=config["run"]["run_id"],
            )

            sampler_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:1")
            sampler_model = PeftModel.from_pretrained(
                sampler_base,
                v0_path,
                adapter_name="student_active",
                is_trainable=False,
            )
            del sampler_base
            sampler_model.eval()
            with torch.inference_mode():
                v0_first, _ = score_rows(
                    sampler_model, probe_rows, device="cuda:1", inference=True
                )
                v0_second, _ = score_rows(
                    sampler_model, probe_rows, device="cuda:1", inference=True
                )
            v0_rows = [
                v0_first[index, : len(row["response_ids"])] for index, row in enumerate(probe_rows)
            ]
            v0_repeat_rows = [
                v0_second[index, : len(row["response_ids"])] for index, row in enumerate(probe_rows)
            ]
            v0_repeat = gap_metrics(v0_rows, v0_repeat_rows, probe_rows)
            v0_identity = adapter_artifact_identity(
                v0_path,
                logical_version=0,
                runtime_name="student_active",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            student_v0_identity = build_adapter_identity_manifest(
                {name: parameters[name] for name in trainable_names},
                adapter_config=student_model.peft_config["default"],
                adapter_logical_version=0,
                adapter_runtime_name="default",
                active_adapter="default",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            v0_runtime = runtime_identity_from_peft(
                sampler_model,
                logical_version=0,
                runtime_name="student_active",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            v0_identity_gate = bool(
                tensor_manifest_equal(student_v0_identity, v0_identity)
                and structural_config_equal(student_v0_identity, v0_identity)
                and manifest_equal(v0_identity, v0_runtime)
            )
            v0_manifest_path = output / "trainer_v0_authority_manifest.json"
            v0_manifest_sha = atomic_write_json(
                v0_manifest_path,
                {
                    "schema_version": 5,
                    "trainer_identity": student_v0_identity,
                    "identity": v0_identity,
                    "fresh_runtime_identity": v0_runtime,
                    "identity_gate_passed": v0_identity_gate,
                },
            )
            v0_authority = trainer_authority_from_manifest(
                v0_identity,
                artifact_manifest_sha256=v0_manifest_sha,
                trainer_memory_reload_gate_passed=v0_identity_gate,
                run_token=f"{config['run']['run_id']}:adapter-v0",
            )
            sampler_registry_before = v0_runtime["registry_snapshot"]
            v0_request = {
                "run_token": v0_authority["run_token"],
                "logical_version": 0,
                "authoritative_tensor_sha256": v0_authority[
                    "aggregate_tensor_sha256"
                ],
                "canonical_config_sha256": v0_authority[
                    "canonical_config_sha256"
                ],
                "base_revision": v0_authority["base_revision"],
                "tokenizer_revision": v0_authority["tokenizer_revision"],
            }
            with torch.inference_mode():
                v0_normal_values, v0_normal_execution = guard_sampler_operation(
                    authority=v0_authority,
                    runtime_identity=v0_runtime,
                    request_identity=v0_request,
                    operation="fixed_action",
                    callback=lambda: score_rows(
                        sampler_model, probe_rows, device="cuda:1", inference=True
                    )[0],
                )
            v0_normal = {
                "run_id": config["run"]["run_id"],
                "logical_version": 0,
                "accepted": v0_normal_execution["accepted"],
                "guard_stage": v0_normal_execution["guard_stage"],
                "request_expected_tensor_sha256": v0_request[
                    "authoritative_tensor_sha256"
                ],
                "trainer_authoritative_tensor_sha256": v0_authority[
                    "aggregate_tensor_sha256"
                ],
                "sampler_runtime_tensor_sha256": v0_runtime[
                    "aggregate_tensor_sha256"
                ],
                "authority_after_request_sha256": v0_authority[
                    "aggregate_tensor_sha256"
                ],
                "canonical_config_sha256": v0_authority[
                    "canonical_config_sha256"
                ],
                "base_revision": v0_authority["base_revision"],
                "tokenizer_revision": v0_authority["tokenizer_revision"],
                "scoring_executed": v0_normal_execution["scoring_executed"],
                "generation_executed": v0_normal_execution["generation_executed"],
                "finite": bool(torch.isfinite(v0_normal_values).all()),
                "silent_fallback": False,
            }
            wrong_request = dict(v0_request)
            wrong_request["authoritative_tensor_sha256"] = (
                "0" * 64
                if v0_authority["aggregate_tensor_sha256"] != "0" * 64
                else "f" * 64
            )
            v0_wrong = {
                "run_id": config["run"]["run_id"],
                "logical_version": 0,
                "accepted": False,
                "guard_stage": "identity_guard_before_forward",
                "request_expected_tensor_sha256": wrong_request[
                    "authoritative_tensor_sha256"
                ],
                "trainer_authoritative_tensor_sha256": v0_authority[
                    "aggregate_tensor_sha256"
                ],
                "sampler_runtime_tensor_sha256": v0_runtime[
                    "aggregate_tensor_sha256"
                ],
                "authority_after_request_sha256": v0_authority[
                    "aggregate_tensor_sha256"
                ],
                "canonical_config_sha256": v0_authority[
                    "canonical_config_sha256"
                ],
                "base_revision": v0_authority["base_revision"],
                "tokenizer_revision": v0_authority["tokenizer_revision"],
                "scoring_executed": False,
                "generation_executed": False,
                "finite": True,
                "silent_fallback": False,
                "error_code": None,
                "sampler_self_authority_accepted": False,
            }
            try:
                guard_sampler_operation(
                    authority=v0_authority,
                    runtime_identity=v0_runtime,
                    request_identity=wrong_request,
                    operation="fixed_action",
                    callback=lambda: (_ for _ in ()).throw(
                        AssertionError("wrong-authority v0 request reached forward")
                    ),
                )
            except SamplerIdentityGuardError as error:
                v0_wrong.update(
                    {
                        "error_code": error.code,
                        "guard_stage": error.evidence["guard_stage"],
                        "scoring_executed": error.evidence["scoring_executed"],
                        "generation_executed": error.evidence[
                            "generation_executed"
                        ],
                    }
                )
            v0_guard = validate_v0_guard_evidence(v0_normal, v0_wrong)
            atomic_write_json(output / "v0_guard.json", v0_guard)
            atomic_write_json(
                output / "v0_repeat_control.json",
                {
                    "schema_version": 5,
                    "status": (
                        "pass"
                        if v0_repeat["finite_rate"] == 1.0
                        and v0_repeat["max"] <= 0.0001
                        and v0_identity_gate
                        else "fail"
                    ),
                    "same_instance_repeat": v0_repeat,
                    "trainer_saved_runtime_identity_gate": v0_identity_gate,
                },
            )
            if not (
                v0_repeat["finite_rate"] == 1.0
                and v0_repeat["max"] <= 0.0001
                and v0_identity_gate
            ):
                raise ThreePolicyGPURuntimeError("P4.5 v0 repeat/identity gate failed")
            sampler_model.to("cpu")
            torch.cuda.empty_cache()

            teacher_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:1")
            teacher_model = PeftModel.from_pretrained(
                teacher_base,
                config["teacher"]["adapter_path"],
                adapter_name="medical",
                is_trainable=False,
            )
            del teacher_base
            teacher_model.eval()
            behavior, mask = pad(
                [torch.tensor(row["rollout_behavior_logprob"]) for row in trajectories]
            )
            current_pre, current_mask = score_rows(
                student_model, trajectories, device="cuda:0", inference=True
            )
            teacher, teacher_mask = score_rows(
                teacher_model, trajectories, device="cuda:1", inference=True
            )
            teacher_gradient_parameters = [
                name
                for name, parameter in teacher_model.named_parameters()
                if parameter.grad is not None
            ]
            if not (torch.equal(mask, old_mask) and torch.equal(mask, current_mask) and torch.equal(mask, teacher_mask)):
                raise ThreePolicyGPURuntimeError("P4.5 three-policy response masks differ")
            prompt_ids = tuple(str(row["fixture_id"]) for row in trajectories)
            group_ids = ("g0",) * len(trajectories)
            source_roles = tuple(str(row["source_role"]) for row in trajectories)
            bundle = ThreePolicyLogProbBundle(
                rollout_behavior_logprob=behavior.detach(),
                old_actor_logprob=old_actor.detach(),
                current_actor_logprob=current_pre.detach().requires_grad_(),
                teacher_logprob=teacher.detach(),
                response_mask=mask,
                behavior_provenance=rollout_provenance,
            )
            validate_three_policy_bundle(
                bundle,
                require_pre_update_identity=True,
                identity_tolerance=float(gates["current_pre_old_actor_max_abs"]),
            )
            before_result = decoupled_corrected_objective(
                bundle,
                prompt_ids=prompt_ids,
                group_ids=group_ids,
                source_roles=source_roles,
                beta=float(algorithm["beta"]),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
                rollout_is_threshold=2.0,
            )
            correction = before_result.correction.metrics
            partition_values = list(correction["per_prompt"].values()) + list(
                correction["per_source"].values()
            )
            correction_pass = bool(
                not before_result.correction.truncated_weight.requires_grad
                and correction["ess_fraction"] >= float(gates["ess_fraction_min"])
                and correction["cap_fraction"] <= float(gates["cap_fraction_max"])
                and all(
                    item["ess_fraction"] >= float(gates["ess_fraction_min"])
                    and item["cap_fraction"] <= float(gates["cap_fraction_max"])
                    for item in partition_values
                )
            )
            atomic_write_json(
                output / "four_prompt_correction.json",
                {
                    "schema_version": 5,
                    "status": "pass" if correction_pass else "fail",
                    "prompt_count": 4,
                    "ess_fraction": correction["ess_fraction"],
                    "cap_fraction": correction["cap_fraction"],
                    "per_prompt": correction["per_prompt"],
                    "per_source": correction["per_source"],
                    "correction_weight_requires_grad": before_result.correction.truncated_weight.requires_grad,
                    "final_access": False,
                    "label_access": False,
                },
            )
            if not correction_pass:
                raise ThreePolicyGPURuntimeError("P4.5 four-prompt correction gate failed")

            before_parameters = {
                name: parameters[name].detach().cpu().clone() for name in trainable_names
            }
            optimizer = torch.optim.AdamW(
                [parameters[name] for name in trainable_names],
                lr=float(optimizer_config["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
                eps=float(optimizer_config["epsilon"]),
                foreach=bool(optimizer_config["foreach"]),
            )
            optimizer.zero_grad(set_to_none=True)
            for index, row in enumerate(trajectories):
                current_row, row_mask = score_rows(
                    student_model, [row], device="cuda:0", inference=False
                )
                length = int(row_mask[0].sum().cpu())
                row_bundle = ThreePolicyLogProbBundle(
                    rollout_behavior_logprob=bundle.rollout_behavior_logprob[index : index + 1, :length],
                    old_actor_logprob=bundle.old_actor_logprob[index : index + 1, :length],
                    current_actor_logprob=current_row[:, :length],
                    teacher_logprob=bundle.teacher_logprob[index : index + 1, :length],
                    response_mask=row_mask[:, :length],
                    behavior_provenance=rollout_provenance,
                )
                row_result = decoupled_corrected_objective(
                    row_bundle,
                    prompt_ids=(prompt_ids[index],),
                    group_ids=(group_ids[index],),
                    source_roles=(source_roles[index],),
                    beta=float(algorithm["beta"]),
                    clip_low=float(algorithm["clip_low"]),
                    clip_high=float(algorithm["clip_high"]),
                    rollout_is_threshold=2.0,
                )
                (row_result.loss / 4.0).backward()
            gradient_before_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameters[name] for name in trainable_names],
                    float(optimizer_config["global_gradient_clip_norm"]),
                )
            )
            gradient_after_clip = math.sqrt(
                sum(
                    float(parameters[name].grad.detach().float().square().sum().cpu())
                    for name in trainable_names
                    if parameters[name].grad is not None
                )
            )
            gradients = {
                name: parameters[name].grad.detach().cpu().clone()
                for name in trainable_names
                if parameters[name].grad is not None
            }
            optimizer.step()
            after_parameters = {
                name: parameters[name].detach().cpu().clone() for name in trainable_names
            }
            optimizer_audit = audit_optimizer_update(
                before=before_parameters,
                after=after_parameters,
                loss_gradients=gradients,
                declared_trainable_names=trainable_names,
                actual_requires_grad_names=tuple(
                    name for name, parameter in student_model.named_parameters() if parameter.requires_grad
                ),
                fresh_optimizer=True,
                weight_decay=float(optimizer_config["weight_decay"]),
                require_nonzero=True,
                descent_dot_max=0.0,
            )
            student_model.eval()
            current_after, after_mask = score_rows(
                student_model, trajectories, device="cuda:0", inference=True
            )
            after_bundle = ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.rollout_behavior_logprob,
                old_actor_logprob=bundle.old_actor_logprob,
                current_actor_logprob=current_after.detach().requires_grad_(),
                teacher_logprob=bundle.teacher_logprob,
                response_mask=after_mask,
                behavior_provenance=rollout_provenance,
            )
            after_result = decoupled_corrected_objective(
                after_bundle,
                prompt_ids=prompt_ids,
                group_ids=group_ids,
                source_roles=source_roles,
                beta=float(algorithm["beta"]),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
                rollout_is_threshold=2.0,
            )
            alignment = float(
                grouped_trajectory_mean(
                    before_result.correction.truncated_weight
                    * before_result.advantage
                    * (current_after.detach() - bundle.current_actor_logprob.detach()),
                    mask,
                    prompt_ids=prompt_ids,
                    group_ids=group_ids,
                ).cpu()
            )
            objective_before = float(before_result.surrogate.detach().cpu())
            objective_after = float(after_result.surrogate.detach().cpu())
            loss_before = float(before_result.loss.detach().cpu())
            loss_after = float(after_result.loss.detach().cpu())
            base_gradient_parameters = [
                name
                for name, parameter in student_model.named_parameters()
                if name not in trainable_names and parameter.grad is not None
            ]
            teacher_gradient_parameters = [
                name
                for name, parameter in teacher_model.named_parameters()
                if parameter.grad is not None
            ]
            telemetry_config = config.get("reconstruction_telemetry", {})
            if not isinstance(telemetry_config, Mapping):
                raise ThreePolicyGPURuntimeError(
                    "reconstruction telemetry config is absent"
                )
            reconstruction = build_reconstruction_telemetry(
                run_id=config["run"]["run_id"],
                step_id=str(telemetry_config.get("step_id", "step0_v0_to_v1")),
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
                teacher_gradient_tensor_count=len(teacher_gradient_parameters),
                base_gradient_tensor_count=len(base_gradient_parameters),
                optimizer_config={
                    "name": str(optimizer_config["type"]).lower(),
                    "learning_rate": float(optimizer_config["learning_rate"]),
                    "weight_decay": float(optimizer_config["weight_decay"]),
                    "max_grad_norm": float(
                        optimizer_config["global_gradient_clip_norm"]
                    ),
                    "ppo_clip_low": 1.0 - float(algorithm["clip_low"]),
                    "ppo_clip_high": 1.0 + float(algorithm["clip_high"]),
                    "importance_cap": 2.0,
                },
                near_zero_threshold=float(
                    telemetry_config["advantage_near_zero_threshold"]
                ),
                teacher_detached=not bundle.teacher_logprob.requires_grad,
                old_actor_detached=not bundle.old_actor_logprob.requires_grad,
                correction_weight_detached=(
                    not before_result.correction.truncated_weight.requires_grad
                ),
            )
            validate_reconstruction_telemetry(reconstruction)
            reconstruction_name = str(
                telemetry_config.get("artifact_name", "reconstruction_step0.json")
            )
            if Path(reconstruction_name).name != reconstruction_name:
                raise ThreePolicyGPURuntimeError(
                    "reconstruction telemetry artifact name is invalid"
                )
            reconstruction_sha = atomic_write_json(
                output / reconstruction_name, reconstruction
            )
            append_metric(
                0,
                "reconstruction_step0",
                "pass",
                run_id=config["run"]["run_id"],
                artifact_sha256=reconstruction_sha,
                valid_token_count=reconstruction["q_p_old"]["valid_token_count"],
                ess_fraction=reconstruction["q_p_old"]["ess_fraction"],
                cap_fraction=reconstruction["q_p_old"]["cap_fraction"],
                objective_delta=reconstruction["optimizer_update"]["objective_delta"],
                loss_delta=reconstruction["optimizer_update"]["loss_delta"],
            )
            _release(torch, teacher_model)
            teacher_model = None
            update_pass = bool(
                objective_after > objective_before + 1e-6
                and loss_after < loss_before - 1e-6
                and alignment > 0
                and optimizer_audit.hard_gate_passed
                and not teacher_gradient_parameters
                and not base_gradient_parameters
                and all(parameters[name]._version == value for name, value in frozen_versions.items())
            )
            step_artifact = {
                "schema_version": 5,
                "status": "pass" if update_pass else "fail",
                "prompt_count": 4,
                "objective_before": objective_before,
                "objective_after": objective_after,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "alignment": alignment,
                "gradient_norm_before_clip": gradient_before_clip,
                "gradient_norm_after_clip": gradient_after_clip,
                "parameter_delta_norm": optimizer_audit.parameter_delta_norm,
                "trainable_tensor_count": len(trainable_names),
                "teacher_gradient_parameters": teacher_gradient_parameters,
                "base_gradient_parameters": base_gradient_parameters,
                "base_parameter_versions_unchanged": all(
                    parameters[name]._version == value for name, value in frozen_versions.items()
                ),
                "formal_b2_checkpoint": False,
                "medical_capability_claim": False,
            }
            atomic_write_json(output / "four_prompt_corrected_one_step.json", step_artifact)
            if not update_pass:
                raise ThreePolicyGPURuntimeError("P4.5 four-prompt one-step gate failed")

            checkpoint = output / "checkpoints" / "trainer_v1_adapter"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            student_model.save_pretrained(checkpoint, safe_serialization=True)
            v1_file_sha = ordered_adapter_sha256(checkpoint)
            target_identity = adapter_artifact_identity(
                checkpoint,
                logical_version=1,
                runtime_name="student_active",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            trainer_identity = build_adapter_identity_manifest(
                {name: parameters[name] for name in trainable_names},
                adapter_config=student_model.peft_config["default"],
                adapter_logical_version=1,
                adapter_runtime_name="default",
                active_adapter="default",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            with torch.inference_mode():
                trainer_values, _ = score_rows(
                    student_model, probe_rows, device="cuda:0", inference=True
                )
            trainer_rows = [
                trainer_values[index, : len(row["response_ids"])]
                for index, row in enumerate(probe_rows)
            ]
            student_model.to("cpu")
            torch.cuda.empty_cache()
            reload_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:0")
            trainer_reload = PeftModel.from_pretrained(
                reload_base,
                checkpoint,
                adapter_name="student_active",
                is_trainable=False,
            )
            del reload_base
            trainer_reload.eval()
            reload_identity = runtime_identity_from_peft(
                trainer_reload,
                logical_version=1,
                runtime_name="student_active",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            with torch.inference_mode():
                reload_values, _ = score_rows(
                    trainer_reload, probe_rows, device="cuda:0", inference=True
                )
            reload_rows = [
                reload_values[index, : len(row["response_ids"])]
                for index, row in enumerate(probe_rows)
            ]
            trainer_reload_gap = gap_metrics(trainer_rows, reload_rows, probe_rows)
            _release(torch, trainer_reload)
            del trainer_reload
            trainer_gate = bool(
                tensor_manifest_equal(trainer_identity, target_identity)
                and structural_config_equal(trainer_identity, target_identity)
                and manifest_equal(reload_identity, target_identity)
                and trainer_reload_gap["finite_rate"] == 1.0
                and trainer_reload_gap["max"] <= 0.0001
            )
            trainer_manifest_path = output / "trainer_v1_authority_manifest.json"
            trainer_manifest_sha = atomic_write_json(
                trainer_manifest_path,
                {
                    "schema_version": 5,
                    "saved_adapter_file_sha256": v1_file_sha,
                    "trainer_identity": trainer_identity,
                    "saved_artifact_identity": target_identity,
                    "fresh_reload_identity": reload_identity,
                    "trainer_memory_reload_metrics": trainer_reload_gap,
                    "trainer_saved_tensor_identity_match": tensor_manifest_equal(
                        trainer_identity, target_identity
                    ),
                    "trainer_saved_structural_config_match": structural_config_equal(
                        trainer_identity, target_identity
                    ),
                    "trainer_inference_mode": trainer_identity["canonical_config"][
                        "inference_mode"
                    ],
                    "saved_inference_mode": target_identity["canonical_config"][
                        "inference_mode"
                    ],
                    "trainer_memory_reload_gate_passed": trainer_gate,
                },
            )
            authority = trainer_authority_from_manifest(
                target_identity,
                artifact_manifest_sha256=trainer_manifest_sha,
                trainer_memory_reload_gate_passed=trainer_gate,
                run_token=f"{config['run']['run_id']}:adapter-v1",
            )

            sampler_model.to("cuda:1")
            refresh_result = refresh_stable_slot(
                sampler_model,
                adapter_path=checkpoint,
                current_authority=v0_authority,
                target_authority=authority,
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            runtime_identity = refresh_result["runtime_identity"]
            request = {
                "run_token": authority["run_token"],
                "logical_version": 1,
                "authoritative_tensor_sha256": authority["aggregate_tensor_sha256"],
                "canonical_config_sha256": authority["canonical_config_sha256"],
                "base_revision": authority["base_revision"],
                "tokenizer_revision": authority["tokenizer_revision"],
            }
            with torch.inference_mode():
                normal_values, normal_evidence = guard_sampler_operation(
                    authority=authority,
                    runtime_identity=runtime_identity,
                    request_identity=request,
                    operation="fixed_action",
                    callback=lambda: score_rows(
                        sampler_model, probe_rows, device="cuda:1", inference=True
                    )[0],
                )
            live_rows = [
                normal_values[index, : len(row["response_ids"])]
                for index, row in enumerate(probe_rows)
            ]
            sampler_model.to("cpu")
            torch.cuda.empty_cache()
            fresh_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:1")
            fresh = PeftModel.from_pretrained(
                fresh_base,
                checkpoint,
                adapter_name="student_active",
                is_trainable=False,
            )
            del fresh_base
            fresh.eval()
            fresh_identity = runtime_identity_from_peft(
                fresh,
                logical_version=1,
                runtime_name="student_active",
                base_revision=config["model"]["revision"],
                tokenizer_revision=config["model"]["tokenizer_revision"],
            )
            with torch.inference_mode():
                fresh_values, _ = score_rows(
                    fresh, probe_rows, device="cuda:1", inference=True
                )
            fresh_rows = [
                fresh_values[index, : len(row["response_ids"])]
                for index, row in enumerate(probe_rows)
            ]
            _release(torch, fresh)
            del fresh
            same_path = gap_metrics(live_rows, fresh_rows, probe_rows)
            stale = dict(request)
            stale.update(
                {
                    "run_token": v0_authority["run_token"],
                    "logical_version": 0,
                    "authoritative_tensor_sha256": v0_authority[
                        "aggregate_tensor_sha256"
                    ],
                    "canonical_config_sha256": v0_authority["canonical_config_sha256"],
                }
            )
            stale_evidence = {
                "rejected": False,
                "scoring_executed": False,
                "generation_executed": False,
                "error_code": None,
                "rejection_phase": None,
            }
            try:
                guard_sampler_operation(
                    authority=authority,
                    runtime_identity=runtime_identity,
                    request_identity=stale,
                    operation="fixed_action",
                    callback=lambda: (_ for _ in ()).throw(
                        AssertionError("stale request reached forward")
                    ),
                )
            except SamplerIdentityGuardError as error:
                stale_evidence.update(
                    {
                        "rejected": True,
                        "error_code": error.code,
                        "rejection_phase": error.evidence["guard_stage"],
                        "scoring_executed": error.evidence["scoring_executed"],
                        "generation_executed": error.evidence["generation_executed"],
                    }
                )
            normal_evidence["silent_fallback"] = False
            stale_evidence["silent_fallback"] = False
            identity_gate = bool(
                manifest_equal(target_identity, runtime_identity)
                and manifest_equal(target_identity, fresh_identity)
            )
            hard_gate = bool(
                v0_repeat["finite_rate"] == 1.0
                and v0_repeat["max"] <= 0.0001
                and trainer_gate
                and identity_gate
                and same_path["finite_rate"] == 1.0
                and same_path["max"] <= 0.0001
                and normal_evidence["accepted"] is True
                and normal_evidence["scoring_executed"] is True
                and stale_evidence["rejected"] is True
                and stale_evidence["error_code"] == "STALE_SAMPLER_IDENTITY"
                and stale_evidence["scoring_executed"] is False
            )
            refresh_artifact = {
                "artifact_protocol_version": "p4.5-production-sampler-refresh-v5",
                "run_id": config["run"]["run_id"],
                "status": "pass" if hard_gate else "fail",
                "production_backend_binding": binding,
                "candidate_mechanism": config["sampler_refresh"]["candidate_mechanism"],
                "authoritative_manifest_sha256": trainer_manifest_sha,
                "adapter_config_sha256": authority["canonical_config_sha256"],
                "trainer_identity": target_identity,
                "runtime_identity": runtime_identity,
                "fresh_identity": fresh_identity,
                "logical_versions": {"before": 0, "after": 1},
                "runtime_slot": "student_active",
                "active_adapter": runtime_identity["active_adapter"],
                "registry_before": sampler_registry_before,
                "registry_after": refresh_result["registry_after"],
                "trainer_memory_reload_metrics": trainer_reload_gap,
                "v0_repeat_metrics": v0_repeat,
                "same_path_metrics": same_path,
                "normal_request": normal_evidence,
                "stale_request": stale_evidence,
                "refresh_latency_seconds": refresh_result["refresh_latency_seconds"],
                "failure_layer": None if hard_gate else "production_sampler_refresh_gate",
                "gate_result": "pass" if hard_gate else "fail",
                "failure_reason": None if hard_gate else "one_or_more_frozen_micro_gates_failed",
                "isolation": dict(config["isolation"]),
                "full_prompt_or_response_persisted": False,
                "full_logits_persisted": False,
                "B2_authorized": False,
            }
            atomic_write_json(output / "production_sampler_refresh.json", refresh_artifact)
            append_metric(
                0,
                "four_prompt_corrected_one_step",
                "pass",
                objective_delta=step_artifact["objective_after"] - step_artifact["objective_before"],
                loss_delta=step_artifact["loss_after"] - step_artifact["loss_before"],
            )
            append_metric(
                1,
                "production_sampler_refresh",
                "pass" if hard_gate else "fail",
                max_same_path_gap=same_path["max"],
                finite_rate=same_path["finite_rate"],
                refresh_latency_seconds=refresh_result["refresh_latency_seconds"],
            )
            if not hard_gate:
                raise ThreePolicyGPURuntimeError("P4.5 production sampler refresh gate failed")

        _release(torch, student_model, teacher_model, sampler_model)
        student_model = teacher_model = sampler_model = None
        atomic_write_json(
            output / "runtime_release.json",
            {
                "schema_version": 5,
                "status": "pass",
                "models_released": True,
                "cuda_allocated_bytes_diagnostic": [
                    int(torch.cuda.memory_allocated(index)) for index in range(2)
                ],
                "cuda_reserved_bytes_diagnostic": [
                    int(torch.cuda.memory_reserved(index)) for index in range(2)
                ],
                "post_process_exit_verification_required": True,
            },
        )
        summary = {
            "schema_version": 5,
            "status": "production_sampler_refresh_runtime_passed_pending_post_exit_cleanup",
            "elapsed_seconds": time.time() - started,
            "production_sampler_refresh_ready": False,
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "B2_started": False,
            "next_step": "post_exit_cleanup_and_artifact_derived_readiness",
        }
        atomic_write_json(output / "summary.json", summary)
        return summary
    finally:
        try:
            _release(torch, student_model, teacher_model, sampler_model)
        except Exception:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observed_aggregate_metrics(output: Path) -> dict[str, dict[str, Any]]:
    """Read only already-committed aggregate artifacts for a failure envelope."""

    def read(name: str) -> dict[str, Any] | None:
        path = output / name
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    # A rung-32 calibration artifact is committed inside ``calibrate`` before
    # its gate can raise.  Prefer that observed evidence even when the optional
    # rung marker has not yet been written.
    correction = read("correction_calibration_32.json") or read(
        "correction_calibration_16.json"
    )
    medical = read("corrected_medical_one_step.json")
    null = read("real_base_teacher_null_update.json")
    correction_values = (correction or {}).get("rollout_correction", {})

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    correction_metrics: dict[str, Any] = {"status": "not_run_or_not_observed"}
    if correction is not None:
        correction_metrics = {
            "status": (
                "pass"
                if correction.get("calibration_readiness", {}).get(
                    "calibration_ready"
                )
                is True
                else "fail"
            ),
            "ess_fraction": correction_values.get("ess_fraction"),
            "cap_fraction": correction_values.get("cap_fraction"),
        }

    one_step_metrics: dict[str, Any] = {"status": "not_run_or_not_observed"}
    if medical is not None:
        one_step_metrics = {
            "status": (
                "pass"
                if medical.get("status") == "pass"
                and medical.get("hard_gate_passed") is True
                else "fail"
            )
        }
        if all(
            is_number(medical.get(field))
            for field in (
                "objective_after",
                "objective_before",
                "loss_after",
                "loss_before",
            )
        ):
            one_step_metrics.update(
                {
                    "objective_delta": medical.get("objective_after")
                    - medical.get("objective_before"),
                    "loss_delta": medical.get("loss_after")
                    - medical.get("loss_before"),
                }
            )

    null_metrics: dict[str, Any] = {"status": "not_run_or_not_observed"}
    if null is not None:
        null_metrics = {
            "status": (
                "pass"
                if null.get("status") == "pass"
                and null.get("hard_gate_passed") is True
                else "fail"
            ),
            "advantage_max_abs": null.get("advantage_max_abs"),
            "parameter_delta_norm": null.get("parameter_delta_norm"),
        }
    return {
        "correction": correction_metrics,
        "one_step": one_step_metrics,
        "null": null_metrics,
    }


def optional_32_instability_decision(
    calibration_metrics: Mapping[str, Any], *, ess_fraction_min: float, cap_fraction_max: float
) -> dict[str, Any]:
    """Use the frozen ESS/cap thresholds; never tune after observing rung 16."""

    correction = calibration_metrics.get("rollout_correction")
    if not isinstance(correction, Mapping):
        raise ThreePolicyGPURuntimeError("rung-16 correction metrics are missing")
    reasons: list[str] = []
    for scope in ("per_prompt", "per_source"):
        values = correction.get(scope)
        if not isinstance(values, Mapping) or not values:
            raise ThreePolicyGPURuntimeError(f"rung-16 {scope} metrics are missing")
        for identity, item in values.items():
            if not isinstance(item, Mapping):
                raise ThreePolicyGPURuntimeError(f"rung-16 {scope} metric is malformed")
            ess = item.get("ess_fraction")
            cap = item.get("cap_fraction")
            if not (
                isinstance(ess, (int, float))
                and isinstance(cap, (int, float))
                and math.isfinite(float(ess))
                and math.isfinite(float(cap))
            ):
                raise ThreePolicyGPURuntimeError(f"rung-16 {scope} metric is nonfinite")
            if float(ess) < ess_fraction_min:
                reasons.append(f"{scope}:{identity}:ess_fraction_below_frozen_gate")
            if float(cap) > cap_fraction_max:
                reasons.append(f"{scope}:{identity}:cap_fraction_above_frozen_gate")
    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "ess_fraction_min": float(ess_fraction_min),
        "cap_fraction_max": float(cap_fraction_max),
        "threshold_changed": False,
    }


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
    temporary.replace(path)
    return digest.hexdigest()


def _validate_runtime_card(config: Mapping[str, Any], config_path: Path, root: Path) -> dict[str, str]:
    run_id = str(config.get("run", {}).get("run_id", ""))
    card_path = root / "configs/run_cards" / f"{run_id}.json"
    if not card_path.is_file() or not config_path.is_file():
        raise ThreePolicyGPURuntimeError("run config or run card is missing")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    if (
        (root / str(card.get("config_path", ""))).resolve() != config_path.resolve()
        or card.get("config_sha256") != config_sha
        or card.get("protocol_config_sha256")
        != config.get("validation", {}).get("config_sha256")
        or card.get("artifact_schema_sha256")
        != config.get("artifacts", {}).get("schema_sha256")
        or card.get("P4_2_immutable_status") != "failed_identity_mismatch"
    ):
        raise ThreePolicyGPURuntimeError("run card/config identity mismatch")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(git_head) != 40:
        raise ThreePolicyGPURuntimeError("runtime Git HEAD is invalid")
    return {
        "git_head": git_head,
        "run_config_sha256": config_sha,
        "run_card_sha256": _sha256(card_path),
    }


def execute_three_policy_gpu_protocol(
    config: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    """Run probe -> fresh rollout -> correction -> step/null/refresh once."""

    # GPU/model imports are intentionally below launcher authorization/preflight.
    import torch
    import yaml
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.opd.calibration_data import render_prompt_text, select_prompt_rows
    from src.opd.pg_opd_contract import (
        ThreePolicyLogProbBundle,
        decoupled_corrected_objective,
        grouped_trajectory_mean,
        masked_numeric_summary,
        validate_three_policy_bundle,
    )
    from src.opd.pg_opd_validation import (
        atomic_write_json,
        audit_optimizer_update,
        audit_sampler_refresh,
        refresh_sampler_adapter,
        require_sampler_identity,
    )
    from src.opd.rollout_probability import validate_rollout_behavior_provenance
    from src.opd.sampler_refresh_contract import (
        SAMPLER_REFRESH_MAX_GAP,
        StaleSamplerRequestError,
        assert_sampler_request_identity,
        build_sampler_refresh_report,
        evaluate_sampler_v0_controls,
        ordered_tensor_sha256,
        persist_sampler_refresh_evidence,
        persist_sampler_refresh_failure_binding,
        persist_sampler_refresh_runtime_failure,
        persist_sampler_v0_control_failure,
    )
    from src.opd.scorer_gpu_calibration import (
        _apply_determinism,
        _release,
        ordered_adapter_sha256,
    )
    from src.opd.three_policy_readiness import evaluate_three_policy_calibration

    root = Path(__file__).resolve().parents[2]
    identity = _validate_runtime_card(config, config_path, root)
    _apply_determinism(torch)
    output = Path(str(config["run"]["output_dir"]))
    required_bootstrap = {
        "launch_record.json",
        "config.yaml",
        "metadata.json",
        "data_manifest.json",
        "metrics.jsonl",
        "stdout.log",
        "cost.json",
        "summary.json",
        "checkpoints/index.json",
    }
    observed_bootstrap = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    }
    if not output.is_dir() or not required_bootstrap.issubset(observed_bootstrap):
        raise ThreePolicyGPURuntimeError("P4.3 launch artifact envelope is incomplete")
    started = time.time()
    current_phase = "formal_host_sha_preflight"
    latest_evidence: Path | None = None
    student_model = teacher_model = sampler_model = tokenizer = None

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(identity)
    atomic_write_json(output / "metadata.json", metadata)

    protocol = yaml.safe_load(
        (root / str(config["validation"]["config_path"])).read_text(encoding="utf-8")
    )
    algorithm = protocol["algorithm"]
    optimizer_config = protocol["optimizer"]
    gates = protocol["calibration_gates"]
    is_sampler_refresh_v4 = config.get("schema_version") == 4

    def pad(values: Sequence[Any], *, device: str = "cuda:0") -> tuple[Any, Any]:
        if not values:
            raise ThreePolicyGPURuntimeError("cannot pad an empty trajectory batch")
        result = torch.nn.utils.rnn.pad_sequence(
            [value.reshape(-1).to(device=device, dtype=torch.float32) for value in values],
            batch_first=True,
            padding_value=0.0,
        )
        mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(values):
            mask[index, : value.numel()] = True
        return result, mask

    def action_logprobs(
        model: Any,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        *,
        device: str,
        use_cache: bool = False,
        disable_adapter: bool = False,
    ) -> Any:
        combined = [int(value) for value in prompt_ids] + [int(value) for value in response_ids]
        ids = torch.tensor([combined], dtype=torch.long, device=device)
        context = model.disable_adapter() if disable_adapter else nullcontext()
        with context:
            result = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=use_cache,
                return_dict=True,
            )
        start = len(prompt_ids) - 1
        logits = result.logits[:, start : start + len(response_ids), :].float()
        targets = torch.tensor(response_ids, dtype=torch.long, device=device).view(1, -1, 1)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)

    def score_rows(
        model: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        device: str,
        inference: bool,
        disable_adapter: bool = False,
    ) -> tuple[Any, Any]:
        context = torch.inference_mode() if inference else nullcontext()
        with context:
            values = [
                action_logprobs(
                    model,
                    row["prompt_ids"],
                    row["response_ids"],
                    device=device,
                    use_cache=False,
                    disable_adapter=disable_adapter,
                ).reshape(-1)
                for row in rows
            ]
        return pad(values, device="cuda:0")

    def evidence(values: Any, mask: Any, *, semantic_name: str, requires_grad: bool) -> dict[str, Any]:
        selected = values.detach()[mask]
        finite_count = int(torch.isfinite(selected).sum().cpu())
        return {
            "semantic_name": semantic_name,
            "requires_grad": requires_grad,
            "finite_count": finite_count,
            "valid_token_count": int(selected.numel()),
            "summary": masked_numeric_summary(values, mask),
        }

    def nonfinite_count(values: Any, mask: Any) -> int:
        return int((~torch.isfinite(values.detach()[mask])).sum().cpu())

    def tensor_dict_l2(values: Mapping[str, Any]) -> float:
        squared = sum(
            float(value.detach().float().square().sum().cpu())
            for value in values.values()
        )
        return math.sqrt(squared)

    def tensor_dict_delta_l2(
        before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> float:
        squared = sum(
            float((after[name].float() - before[name].float()).square().sum())
            for name in before
        )
        return math.sqrt(squared)

    def adapter_tensor_sha(values: Mapping[str, Any] | Any) -> str:
        parameters = (
            values
            if isinstance(values, Mapping)
            else {
                name: parameter
                for name, parameter in values.named_parameters()
                if "lora_" in name
            }
        )
        entries: dict[str, tuple[tuple[int, ...], bytes]] = {}
        for name, parameter in parameters.items():
            if "lora_" not in name:
                continue
            normalized = parameter.detach().float().cpu().contiguous().numpy()
            entries[str(name)] = (
                tuple(int(item) for item in normalized.shape),
                normalized.astype("<f4", copy=False).tobytes(order="C"),
            )
        return ordered_tensor_sha256(entries)

    def probe_pairs(
        left: Any,
        right: Any,
        *,
        row: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        left_values = left.detach().float().cpu().reshape(-1).tolist()
        right_values = right.detach().float().cpu().reshape(-1).tolist()
        token_ids = [int(value) for value in row["response_ids"]]
        if not (len(left_values) == len(right_values) == len(token_ids)):
            raise ThreePolicyGPURuntimeError("sampler probe token shape mismatch")
        return [
            {
                "sample_id": str(row["fixture_id"]),
                "token_position": index,
                "token_id": token,
                "left": float(left_values[index]),
                "right": float(right_values[index]),
            }
            for index, token in enumerate(token_ids)
        ]

    def scorer_provenance(
        *, device: str, path: str, use_cache: bool
    ) -> dict[str, Any]:
        return {
            "path": path,
            "backend": f"transformers-{config['formal_rollout']['backend_version']}",
            "dtype": str(config["model"]["dtype"]),
            "device": device,
            "mode": "eval",
            "attention_backend": str(config["model"]["attention_implementation"]),
            "use_cache": use_cache,
            "batch_size": 1,
            "attention_mask": "all_ones_no_padding",
            "position_ids": "implicit_from_attention_mask",
            "log_softmax_dtype": "float32",
            "eos_token_ids": list(config["formal_rollout"]["transformers"]["eos_token_id"]),
            "pad_token_id": int(config["formal_rollout"]["transformers"]["pad_token_id"]),
            "generation_processors_warpers": [],
        }

    def guarded_sampler_action_logprobs(
        model: Any,
        row: Mapping[str, Any],
        *,
        device: str,
        active_version: int,
        active_ordered_tensor_sha: str,
        active_run_token: str,
        requested_version: int,
        requested_ordered_tensor_sha: str,
        requested_run_token: str,
        use_cache: bool = False,
    ) -> Any:
        assert_sampler_request_identity(
            active_version=active_version,
            active_ordered_tensor_sha=active_ordered_tensor_sha,
            active_run_token=active_run_token,
            requested_version=requested_version,
            requested_ordered_tensor_sha=requested_ordered_tensor_sha,
            requested_run_token=requested_run_token,
        )
        return action_logprobs(
            model,
            row["prompt_ids"],
            row["response_ids"],
            device=device,
            use_cache=use_cache,
        )

    def guarded_sampler_generate(
        model: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        generation: Mapping[str, Any],
        active_version: int,
        active_ordered_tensor_sha: str,
        active_run_token: str,
        requested_version: int,
        requested_ordered_tensor_sha: str,
        requested_run_token: str,
    ) -> Any:
        assert_sampler_request_identity(
            active_version=active_version,
            active_ordered_tensor_sha=active_ordered_tensor_sha,
            active_run_token=active_run_token,
            requested_version=requested_version,
            requested_ordered_tensor_sha=requested_ordered_tensor_sha,
            requested_run_token=requested_run_token,
        )
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **dict(generation),
        )

    def probe_payload(
        name: str,
        *,
        classification: str,
        left: Any,
        right: Any,
        row: Mapping[str, Any],
        left_scorer: Mapping[str, Any],
        right_scorer: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "classification": classification,
            "left_scorer": dict(left_scorer),
            "right_scorer": dict(right_scorer),
            "pairs": probe_pairs(left, right, row=row),
        }

    def generate_one(row: Mapping[str, Any], *, row_index: int, max_new_tokens: int) -> dict[str, Any]:
        prompt_ids = [int(value) for value in row["prompt_ids"]]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
        generation = dict(config["formal_rollout"]["transformers"])
        generation["max_new_tokens"] = int(max_new_tokens)
        generation["eos_token_id"] = student_model.generation_config.eos_token_id
        generation["pad_token_id"] = student_model.generation_config.pad_token_id
        seed = int(config["run"]["seed"]) * 100_000 + int(row_index)
        # Transformers 4.56.2 `_sample` calls torch.multinomial without a
        # per-call generator parameter. Scoped default-generator seeding gives
        # deterministic, isolated sampling without passing an unsupported kwarg.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = student_model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                **generation,
            )
        response_ids = [int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()]
        if not response_ids or len(generated.scores) != len(response_ids):
            raise ThreePolicyGPURuntimeError("generate output score/token alignment failed")
        if generated.logits is None or len(generated.logits) != len(response_ids):
            raise ThreePolicyGPURuntimeError("raw generation logits were not returned")
        behavior = []
        raw_generation = []
        for token, processed, raw in zip(
            response_ids, generated.scores, generated.logits, strict=True
        ):
            behavior.append(
                float(torch.log_softmax(processed[0].float(), dim=-1)[token].detach().cpu())
            )
            raw_generation.append(
                float(torch.log_softmax(raw[0].float(), dim=-1)[token].detach().cpu())
            )
        eos_values = generation["eos_token_id"]
        if isinstance(eos_values, int):
            eos_values = [eos_values]
        eos_observed = bool(response_ids[-1] in set(int(value) for value in eos_values or []))
        del generated
        return {
            **dict(row),
            "response_ids": response_ids,
            "response_mask": [1] * len(response_ids),
            "rollout_behavior_logprob": behavior,
            "raw_generation_logprob": raw_generation,
            "seed": seed,
            "generator": "torch_cuda_default_generator_scoped_manual_seed",
            "finish_reason": "eos" if eos_observed else "length",
            "eos_observed": eos_observed,
            "truncated": not eos_observed,
        }

    def trajectory_provenance(rows: Sequence[Mapping[str, Any]], *, adapter_sha: str) -> dict[str, Any]:
        token_identity = [
            {
                "fixture_id": row["fixture_id"],
                "prompt_ids": row["prompt_ids"],
                "response_ids": row["response_ids"],
            }
            for row in rows
        ]
        generation = dict(config["formal_rollout"]["transformers"])
        generation["eos_token_id"] = student_model.generation_config.eos_token_id
        generation["pad_token_id"] = student_model.generation_config.pad_token_id
        return {
            "artifact_protocol_version": "p4.3-full-support-trajectory-v1",
            "trajectory_run_id": config["run"]["run_id"],
            "trajectory_kind": "fresh_full_support",
            "backend": "transformers",
            "backend_version": "4.56.2",
            "model_version": config["model"]["revision"],
            "adapter_version": adapter_sha,
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
            "token_identity_sha256": _canonical_sha(token_identity),
            "eos_and_truncation_saved": True,
            "seed": int(config["run"]["seed"]),
            "generator": "torch_cuda_default_generator_scoped_manual_seed_per_prompt",
            "sampler_adapter_version": 0,
            "sampler_adapter_sha256": adapter_sha,
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }

    def correction_artifact(result: Any) -> dict[str, Any]:
        pooled = result.correction.metrics["token_pooled"]
        return {
            "rollout_is_threshold": 2.0,
            "rollout_actor_log_ratio": pooled["rollout_actor_log_ratio"],
            "raw_is_weight": pooled["raw_is_weight"],
            "truncated_is_weight": pooled["truncated_is_weight"],
            "ess": result.correction.metrics["ess"],
            "ess_fraction": result.correction.metrics["ess_fraction"],
            "cap_fraction": result.correction.metrics["cap_fraction"],
            "per_prompt": result.correction.metrics["per_prompt"],
            "per_source": result.correction.metrics["per_source"],
            "token_pooled": pooled,
            "prompt_equal": result.correction.metrics["prompt_equal"],
        }

    def calibrate(
        rows: Sequence[Mapping[str, Any]],
        *,
        provenance: Mapping[str, Any],
        trajectory_sha256: str,
        rung: int,
    ) -> dict[str, Any]:
        selected = list(rows[:rung])
        behavior, response_mask = pad(
            [torch.tensor(row["rollout_behavior_logprob"]) for row in selected]
        )
        old_actor, old_mask = pad(
            [torch.tensor(row["old_actor_logprob"]) for row in selected]
        )
        if not torch.equal(response_mask, old_mask):
            raise ThreePolicyGPURuntimeError("behavior/old actor mask mismatch")
        student_model.eval()
        # Calibration needs actor identity/numerics, not 16 retained 4B graphs.
        current_scored, current_mask = score_rows(
            student_model, selected, device="cuda:0", inference=True
        )
        current_actor = current_scored.detach().requires_grad_()
        if not torch.equal(response_mask, current_mask):
            raise ThreePolicyGPURuntimeError("current actor mask mismatch")
        teacher_model.eval()
        teacher, teacher_mask = score_rows(
            teacher_model, selected, device="cuda:1", inference=True
        )
        if not torch.equal(response_mask, teacher_mask):
            raise ThreePolicyGPURuntimeError("Teacher mask mismatch")
        prompt_ids = tuple(str(row["fixture_id"]) for row in selected)
        source_roles = tuple(str(row["source_role"]) for row in selected)
        group_ids = ("g0",) * len(selected)
        bundle = ThreePolicyLogProbBundle(
            rollout_behavior_logprob=behavior.detach(),
            old_actor_logprob=old_actor.detach(),
            current_actor_logprob=current_actor,
            teacher_logprob=teacher.detach(),
            response_mask=response_mask,
            behavior_provenance=dict(provenance),
        )
        validate_three_policy_bundle(
            bundle,
            require_pre_update_identity=True,
            identity_tolerance=float(gates["current_pre_old_actor_max_abs"]),
        )
        result = decoupled_corrected_objective(
            bundle,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            source_roles=source_roles,
            beta=float(algorithm["beta"]),
            clip_low=float(algorithm["clip_low"]),
            clip_high=float(algorithm["clip_high"]),
            rollout_is_threshold=2.0,
        )
        valid = response_mask
        identity_max = float(
            (current_actor.detach()[valid] - old_actor[valid]).abs().max().cpu()
        )
        q_old_absolute = masked_numeric_summary(
            (old_actor - behavior).detach().abs(), valid
        )
        ppo_ratio = result.ppo_ratio.detach()
        correction = correction_artifact(result)
        metrics = {
            "schema_version": 3,
            "artifact_protocol_version": "p4.3-three-policy-correction-v3",
            "trajectory_protocol_version": "p4.3-full-support-trajectory-v1",
            "trajectory_kind": "fresh_full_support",
            "p4_2_historical_status": "failed_identity_mismatch",
            "rung_prompts": rung,
            "trajectory_sha256": trajectory_sha256,
            "token_identity_sha256": provenance["token_identity_sha256"],
            "policy_semantics": {
                "rollout_behavior_logprob": "log_q_detached",
                "old_actor_logprob": "log_p_old_direct_forward_detached",
                "current_actor_logprob": "log_p_theta_with_gradient",
                "teacher_logprob": "same_tokens_raw_teacher_detached",
            },
            "ratio_semantics": {
                "rollout_correction": "old_actor_minus_rollout_behavior",
                "ppo_ratio": "current_actor_minus_old_actor",
            },
            "rollout_behavior_logprob": evidence(
                behavior, valid, semantic_name="log_q", requires_grad=False
            ),
            "old_actor_logprob": evidence(
                old_actor, valid, semantic_name="log_p_old", requires_grad=False
            ),
            "current_actor_logprob": evidence(
                current_actor, valid, semantic_name="log_p_theta", requires_grad=True
            ),
            "calibration_current_gradient_connectivity": (
                "detached_leaf_for_numeric_identity_and_loss_boundary_only; "
                "model_parameter_connectivity_is_tested_by_streamed_medical_optimizer_audit"
            ),
            "teacher_logprob": evidence(
                teacher, valid, semantic_name="log_p_teacher", requires_grad=False
            ),
            "behavior_provenance": dict(provenance),
            "q_vs_old_actor": {
                "mae": q_old_absolute["mean"],
                "abs_p95": q_old_absolute["p95"],
                "max_abs": q_old_absolute["max"],
            },
            "current_pre_vs_old_actor": {"max_abs": identity_max},
            "rollout_correction": correction,
            "ppo": {
                "pre_ratio": masked_numeric_summary(ppo_ratio, valid),
                "pre_clip_fraction": result.ppo_clip_fraction,
            },
            "nonfinite_counts": {
                "rollout_behavior_logprob": nonfinite_count(behavior, valid),
                "old_actor_logprob": nonfinite_count(old_actor, valid),
                "current_actor_logprob": nonfinite_count(current_actor, valid),
                "teacher_logprob": nonfinite_count(teacher, valid),
                "raw_is_weight": nonfinite_count(result.correction.raw_weight, valid),
                "truncated_is_weight": nonfinite_count(result.correction.truncated_weight, valid),
                "ppo_ratio": nonfinite_count(ppo_ratio, valid),
            },
            "trainable_parameter_names": [
                name for name, parameter in student_model.named_parameters() if parameter.requires_grad
            ],
            "base_parameter_versions_unchanged": True,
            "isolation": dict(config["isolation"]),
        }
        readiness = evaluate_three_policy_calibration(metrics, provenance)
        metrics["calibration_readiness"] = asdict(readiness)
        path = output / f"correction_calibration_{rung}.json"
        atomic_write_json(path, metrics)
        nonlocal latest_evidence
        latest_evidence = path
        if not readiness.calibration_ready:
            raise ThreePolicyGPURuntimeError(
                "correction calibration failed: " + "; ".join(readiness.failure_reasons)
            )
        return {
            "rows": selected,
            "bundle": bundle,
            "result": result,
            "prompt_ids": prompt_ids,
            "group_ids": group_ids,
            "source_roles": source_roles,
            "metrics": metrics,
        }

    try:
        prompt_config = config["prompt_selection"]
        o1_rows = select_prompt_rows(
            prompt_config["medical_opd_o1_path"],
            role="medical_opd_o1",
            count=16,
            seed=int(config["run"]["seed"]),
        )
        cmb_rows = select_prompt_rows(
            prompt_config["medical_opd_cmb_path"],
            role="medical_opd_cmb",
            count=16,
            seed=int(config["run"]["seed"]),
        )
        selected_source_rows = [
            item for pair in zip(o1_rows, cmb_rows, strict=True) for item in pair
        ]

        model_path = str(config["model"]["id"])
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, revision=config["model"]["tokenizer_revision"]
        )
        prompt_rows = []
        for index, row in enumerate(selected_source_rows):
            prompt = render_prompt_text(row)
            prompt_ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_rows.append(
                {
                    "fixture_id": f"p4-3-{index:02d}",
                    "source_role": row["target_role"],
                    "source_sample_id": row["sample_id"],
                    "source_content_hash": row["content_hash"],
                    "prompt_ids": [int(value) for value in prompt_ids],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )

        student_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
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
        student_model.eval()
        trainable_names = tuple(
            name for name, parameter in student_model.named_parameters() if parameter.requires_grad
        )
        if not trainable_names or any("lora" not in name.lower() for name in trainable_names):
            raise ThreePolicyGPURuntimeError("Student trainable parameters are not LoRA-only")
        parameter_by_name = dict(student_model.named_parameters())
        frozen_versions = {
            name: parameter._version
            for name, parameter in parameter_by_name.items()
            if name not in trainable_names
        }

        teacher_model = None
        if not is_sampler_refresh_v4:
            teacher_base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to("cuda:1")
            teacher_model = PeftModel.from_pretrained(
                teacher_base,
                config["teacher"]["adapter_path"],
                adapter_name="medical",
                is_trainable=False,
            )
            del teacher_base
            teacher_model.eval()

        with tempfile.TemporaryDirectory(dir=output, prefix=".temporary_adapters_") as temporary:
            initial_adapter_dir = Path(temporary) / "version0"
            student_model.save_pretrained(initial_adapter_dir, safe_serialization=True)
            initial_adapter_sha = ordered_adapter_sha256(initial_adapter_dir)
            initial_parameters = {
                name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
            }

            current_phase = "generation_score_provenance_micro_probe"
            probe = generate_one(prompt_rows[0], row_index=90_000, max_new_tokens=4)
            with torch.inference_mode():
                direct_no_cache = action_logprobs(
                    student_model,
                    probe["prompt_ids"],
                    probe["response_ids"],
                    device="cuda:0",
                    use_cache=False,
                ).reshape(-1)
                direct_cache = action_logprobs(
                    student_model,
                    probe["prompt_ids"],
                    probe["response_ids"],
                    device="cuda:0",
                    use_cache=True,
                ).reshape(-1)
            processed = torch.tensor(probe["rollout_behavior_logprob"])
            raw_generation = torch.tensor(probe["raw_generation_logprob"])
            probe_report = {
                "schema_version": 3,
                "status": "pass",
                "backend": "transformers",
                "backend_version": "4.56.2",
                "tokens": len(probe["response_ids"]),
                "processed_vs_raw_selected_logprob_max_abs": float(
                    (processed - raw_generation).abs().max()
                ),
                "raw_generation_vs_direct_no_cache_max_abs": float(
                    (raw_generation - direct_no_cache.cpu()).abs().max()
                ),
                "direct_cache_vs_no_cache_max_abs": float(
                    (direct_cache.cpu() - direct_no_cache.cpu()).abs().max()
                ),
                "scores_semantics": "processed_pre_softmax_then_manual_log_softmax_selected_token",
                "logits_semantics": "unprocessed_lm_head_logits_then_manual_log_softmax_selected_token",
                "sampling_semantics": "softmax_processed_scores_then_torch_multinomial",
                "generator_semantics": "scoped_default_cuda_generator_manual_seed_local_4.56.2_has_no_generate_generator_parameter",
                "per_token_values_persisted": False,
                "GPU_observed": True,
            }
            if not all(
                math.isfinite(value)
                for key, value in probe_report.items()
                if key.endswith("max_abs")
            ):
                probe_report["status"] = "fail"
            atomic_write_json(output / "generation_provenance_probe.json", probe_report)
            latest_evidence = output / "generation_provenance_probe.json"
            if probe_report["status"] != "pass":
                raise ThreePolicyGPURuntimeError("generation probability micro-probe is nonfinite")

            # P4.4 freezes the v0 controls before rollout/training so the exact
            # long-lived sampler instance can later be refreshed to v1.
            if is_sampler_refresh_v4:
                current_phase = "sampler_v0_repeated_probe"
                sampler_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:1")
                sampler_model = PeftModel.from_pretrained(
                    sampler_base,
                    initial_adapter_dir,
                    adapter_name="version0",
                    is_trainable=False,
                )
                del sampler_base
                sampler_model.eval()
                sampler_before_ordered_sha = adapter_tensor_sha(sampler_model)
                direct_cuda1 = scorer_provenance(
                    device="cuda:1", path="direct_forward_raw_logits", use_cache=False
                )
                with torch.inference_mode():
                    sampler_v0_first = action_logprobs(
                        sampler_model,
                        probe["prompt_ids"],
                        probe["response_ids"],
                        device="cuda:1",
                    ).detach().cpu()
                    sampler_v0_second = action_logprobs(
                        sampler_model,
                        probe["prompt_ids"],
                        probe["response_ids"],
                        device="cuda:1",
                    ).detach().cpu()

                current_phase = "sampler_v0_noop_unload_reload_control"
                noop_refresh_started_at = datetime.now(timezone.utc).isoformat()
                noop_refresh_started_clock = time.perf_counter()
                sampler_model.load_adapter(
                    str(initial_adapter_dir),
                    adapter_name="version0_noop",
                    is_trainable=False,
                )
                sampler_model.set_adapter("version0_noop")
                sampler_model.delete_adapter("version0")
                if (
                    sampler_model.active_adapter != "version0_noop"
                    or "version0" in sampler_model.peft_config
                ):
                    raise ThreePolicyGPURuntimeError("v0 no-op adapter control failed")
                noop_refresh_latency = time.perf_counter() - noop_refresh_started_clock
                noop_refresh_ended_at = datetime.now(timezone.utc).isoformat()
                sampler_noop_ordered_sha = adapter_tensor_sha(sampler_model)
                v0_run_token = f"{config['run']['run_id']}:adapter-v0"
                with torch.inference_mode():
                    sampler_v0_noop = guarded_sampler_action_logprobs(
                        sampler_model,
                        probe,
                        device="cuda:1",
                        active_version=0,
                        active_ordered_tensor_sha=sampler_noop_ordered_sha,
                        active_run_token=v0_run_token,
                        requested_version=0,
                        requested_ordered_tensor_sha=sampler_noop_ordered_sha,
                        requested_run_token=v0_run_token,
                    ).detach().cpu()
                noop_stale_requested_sha = hashlib.sha256(
                    b"p4.4-v0-noop-stale-control"
                ).hexdigest()
                noop_stale_rejected = False
                noop_stale_scoring_executed = False
                try:
                    guarded_sampler_action_logprobs(
                        sampler_model,
                        probe,
                        device="cuda:1",
                        active_version=0,
                        active_ordered_tensor_sha=sampler_noop_ordered_sha,
                        active_run_token=v0_run_token,
                        requested_version=0,
                        requested_ordered_tensor_sha=noop_stale_requested_sha,
                        requested_run_token=v0_run_token,
                    )
                    noop_stale_scoring_executed = True
                except StaleSamplerRequestError:
                    noop_stale_rejected = True
                sampler_model.to("cpu")
                torch.cuda.empty_cache()

                fresh_v0_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:1")
                fresh_v0_sampler = PeftModel.from_pretrained(
                    fresh_v0_base,
                    initial_adapter_dir,
                    adapter_name="version0_fresh",
                    is_trainable=False,
                )
                del fresh_v0_base
                fresh_v0_sampler.eval()
                fresh_v0_ordered_sha = adapter_tensor_sha(fresh_v0_sampler)
                with torch.inference_mode():
                    fresh_v0_probe = action_logprobs(
                        fresh_v0_sampler,
                        probe["prompt_ids"],
                        probe["response_ids"],
                        device="cuda:1",
                    ).detach().cpu()
                _release(torch, fresh_v0_sampler)
                del fresh_v0_sampler

                repeated_v0_probe_payload = probe_payload(
                    "repeated_same_instance_noise",
                    classification="control_diagnostic",
                    left=sampler_v0_first,
                    right=sampler_v0_second,
                    row=probe,
                    left_scorer=direct_cuda1,
                    right_scorer=direct_cuda1,
                )
                noop_v0_probe_payload = probe_payload(
                    "no_op_refresh_gap",
                    classification="control_diagnostic",
                    left=sampler_v0_first,
                    right=sampler_v0_noop,
                    row=probe,
                    left_scorer=direct_cuda1,
                    right_scorer=direct_cuda1,
                )
                no_op_refresh_control = {
                    "version_before": 0,
                    "version_after": 0,
                    "fresh_version": 0,
                    "ordered_tensor_sha_before": sampler_before_ordered_sha,
                    "ordered_tensor_sha_after": sampler_noop_ordered_sha,
                    "fresh_ordered_tensor_sha": fresh_v0_ordered_sha,
                    "saved_adapter_sha_before": initial_adapter_sha,
                    "saved_adapter_sha_after": initial_adapter_sha,
                    "fresh_saved_adapter_sha": initial_adapter_sha,
                    "active_adapter_before": "version0",
                    "active_adapter_after": "version0_noop",
                    "fresh_active_adapter": "version0_fresh",
                    "old_adapter_removed": "version0" not in sampler_model.peft_config,
                    "new_adapter_loaded": "version0_noop" in sampler_model.peft_config,
                    "refresh_start": noop_refresh_started_at,
                    "refresh_end": noop_refresh_ended_at,
                    "refresh_latency_seconds": noop_refresh_latency,
                    "fresh_reload_probe": probe_payload(
                        "no_op_fresh_reload_gap",
                        classification="control_gate",
                        left=sampler_v0_noop,
                        right=fresh_v0_probe,
                        row=probe,
                        left_scorer=direct_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "normal_request": {
                        "requested_version": 0,
                        "requested_ordered_tensor_sha": sampler_noop_ordered_sha,
                        "requested_run_token": v0_run_token,
                        "accepted": True,
                        "scoring_executed": True,
                    },
                    "stale_request": {
                        "requested_version": 0,
                        "requested_ordered_tensor_sha": noop_stale_requested_sha,
                        "requested_run_token": v0_run_token,
                        "rejected": noop_stale_rejected,
                        "scoring_executed": noop_stale_scoring_executed,
                        "rejection_phase": "identity_guard_before_scoring",
                        "error_type": "StaleSamplerRequestError",
                    },
                }
                v0_control_report = evaluate_sampler_v0_controls(
                    run_id=config["run"]["run_id"],
                    repeated_probe=repeated_v0_probe_payload,
                    no_op_probe=noop_v0_probe_payload,
                    no_op_refresh_control=no_op_refresh_control,
                    threshold=SAMPLER_REFRESH_MAX_GAP,
                )
                atomic_write_json(
                    output / "sampler_v0_controls.json", v0_control_report
                )
                latest_evidence = output / "sampler_v0_controls.json"
                if not v0_control_report["hard_gate_passed"]:
                    persist_sampler_v0_control_failure(output, v0_control_report)
                    raise ThreePolicyGPURuntimeError(
                        "; ".join(v0_control_report["failure_reasons"])
                    )

                teacher_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:1")
                teacher_model = PeftModel.from_pretrained(
                    teacher_base,
                    config["teacher"]["adapter_path"],
                    adapter_name="medical",
                    is_trainable=False,
                )
                del teacher_base
                teacher_model.eval()

            current_phase = "fresh_full_support_rollout_4"
            trajectories: list[dict[str, Any]] = [
                generate_one(row, row_index=index, max_new_tokens=128)
                for index, row in enumerate(prompt_rows[:4])
            ]
            with torch.inference_mode():
                old_actor_4, old_mask_4 = score_rows(
                    student_model, trajectories, device="cuda:0", inference=True
                )
            for index, row in enumerate(trajectories):
                length = len(row["response_ids"])
                row["old_actor_logprob"] = old_actor_4[index, :length].cpu().tolist()
            provenance_4 = trajectory_provenance(
                trajectories, adapter_sha=initial_adapter_sha
            )
            validate_rollout_behavior_provenance(
                provenance_4,
                expected_sampler_adapter_sha256=initial_adapter_sha,
                expected_trajectory_run_id=config["run"]["run_id"],
            )
            trajectory_4_sha = _atomic_jsonl(
                output / "fresh_full_support_trajectory_4.jsonl", trajectories
            )
            manifest_4 = {
                "schema_version": 3,
                "status": "fresh_full_support_micro_smoke",
                "run_id": config["run"]["run_id"],
                "rows": 4,
                "response_tokens": sum(len(row["response_ids"]) for row in trajectories),
                "trajectory_sha256": trajectory_4_sha,
                "behavior_provenance": provenance_4,
                "P4_2_status_preserved": "failed_identity_mismatch",
            }
            atomic_write_json(output / "fresh_trajectory_manifest_4.json", manifest_4)
            latest_evidence = output / "fresh_trajectory_manifest_4.json"

            current_phase = "correction_calibration_4"
            calibrate(
                trajectories,
                provenance=provenance_4,
                trajectory_sha256=trajectory_4_sha,
                rung=4,
            )

            current_phase = "fresh_full_support_rollout_16"
            trajectories.extend(
                generate_one(row, row_index=index, max_new_tokens=128)
                for index, row in enumerate(prompt_rows[4:16], start=4)
            )
            with torch.inference_mode():
                old_actor, old_mask = score_rows(
                    student_model, trajectories, device="cuda:0", inference=True
                )
            for index, row in enumerate(trajectories):
                length = len(row["response_ids"])
                row["old_actor_logprob"] = old_actor[index, :length].cpu().tolist()
            provenance = trajectory_provenance(trajectories, adapter_sha=initial_adapter_sha)
            validate_rollout_behavior_provenance(
                provenance,
                expected_sampler_adapter_sha256=initial_adapter_sha,
                expected_trajectory_run_id=config["run"]["run_id"],
            )
            trajectory_sha = _atomic_jsonl(output / "fresh_full_support_trajectory.jsonl", trajectories)
            manifest = {
                "schema_version": 3,
                "status": "fresh_full_support",
                "run_id": config["run"]["run_id"],
                "rows": len(trajectories),
                "response_tokens": sum(len(row["response_ids"]) for row in trajectories),
                "source_counts": {
                    role: sum(row["source_role"] == role for row in trajectories)
                    for role in ("medical_opd_o1", "medical_opd_cmb")
                },
                "trajectory_sha256": trajectory_sha,
                "behavior_provenance": provenance,
                "old_p4_1_trajectory_sha256": config["historical"]["p4_1_trajectory_sha256"],
                "old_p4_1_trajectory_used_as_formal_evidence": False,
                "P4_2_status_preserved": "failed_identity_mismatch",
            }
            atomic_write_json(output / "fresh_trajectory_manifest.json", manifest)
            latest_evidence = output / "fresh_trajectory_manifest.json"

            current_phase = "correction_calibration_16"
            calibration = calibrate(
                trajectories,
                provenance=provenance,
                trajectory_sha256=trajectory_sha,
                rung=16,
            )
            if is_sampler_refresh_v4:
                optional_32 = optional_32_instability_decision(
                    calibration["metrics"],
                    ess_fraction_min=float(gates["ess_fraction_min"]),
                    cap_fraction_max=float(gates["cap_fraction_max"]),
                )
                if optional_32["triggered"]:
                    current_phase = "fresh_full_support_rollout_32"
                    trajectories.extend(
                        generate_one(row, row_index=index, max_new_tokens=128)
                        for index, row in enumerate(prompt_rows[16:32], start=16)
                    )
                    with torch.inference_mode():
                        old_actor, old_mask = score_rows(
                            student_model,
                            trajectories,
                            device="cuda:0",
                            inference=True,
                        )
                    for index, row in enumerate(trajectories):
                        length = len(row["response_ids"])
                        row["old_actor_logprob"] = old_actor[
                            index, :length
                        ].cpu().tolist()
                    provenance = trajectory_provenance(
                        trajectories, adapter_sha=initial_adapter_sha
                    )
                    validate_rollout_behavior_provenance(
                        provenance,
                        expected_sampler_adapter_sha256=initial_adapter_sha,
                        expected_trajectory_run_id=config["run"]["run_id"],
                    )
                    trajectory_sha = _atomic_jsonl(
                        output / "fresh_full_support_trajectory.jsonl", trajectories
                    )
                    manifest.update(
                        {
                            "rows": len(trajectories),
                            "response_tokens": sum(
                                len(row["response_ids"]) for row in trajectories
                            ),
                            "source_counts": {
                                role: sum(
                                    row["source_role"] == role for row in trajectories
                                )
                                for role in ("medical_opd_o1", "medical_opd_cmb")
                            },
                            "trajectory_sha256": trajectory_sha,
                            "behavior_provenance": provenance,
                        }
                    )
                    atomic_write_json(
                        output / "fresh_trajectory_manifest.json", manifest
                    )
                    current_phase = "correction_calibration_32"
                    calibration = calibrate(
                        trajectories,
                        provenance=provenance,
                        trajectory_sha256=trajectory_sha,
                        rung=32,
                    )
                    optional_32[
                        "status"
                    ] = "run_completed_due_preregistered_instability"
                else:
                    optional_32["status"] = "not_run_stable_at_16"
                atomic_write_json(
                    output / "optional_32_prompt_rung.json",
                    {
                        "schema_version": 4,
                        **optional_32,
                    },
                )
            else:
                atomic_write_json(
                    output / "optional_32_prompt_rung.json",
                    {
                        "schema_version": 3,
                        "status": "not_run_frozen_default_stop_after_16",
                        "reason": (
                            "16-prompt calibration passed; no low-cost distribution "
                            "uncertainty declared before execution"
                        ),
                        "threshold_changed": False,
                    },
                )

            current_phase = "corrected_medical_one_step"
            bundle = calibration["bundle"]
            before_result = calibration["result"]
            advantage_values = before_result.advantage.detach()[bundle.response_mask]
            near_zero_threshold = 1.0e-8
            advantage_evidence = {
                **masked_numeric_summary(
                    before_result.advantage.detach(), bundle.response_mask
                ),
                "positive_count": int(
                    (advantage_values > near_zero_threshold).sum().cpu()
                ),
                "negative_count": int(
                    (advantage_values < -near_zero_threshold).sum().cpu()
                ),
                "near_zero_count": int(
                    (advantage_values.abs() <= near_zero_threshold).sum().cpu()
                ),
                "near_zero_threshold": near_zero_threshold,
            }
            before_parameters = {
                name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
            }
            optimizer = torch.optim.AdamW(
                [parameter_by_name[name] for name in trainable_names],
                lr=float(optimizer_config["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
                eps=float(optimizer_config["epsilon"]),
                foreach=bool(optimizer_config["foreach"]),
            )
            optimizer.zero_grad(set_to_none=True)
            # Exact prompt-equal gradient accumulation: this run has one
            # trajectory/group per unique prompt, so each prompt contributes
            # loss_i / N. Backward immediately frees each 4B activation graph.
            if len(set(calibration["prompt_ids"])) != len(calibration["rows"]):
                raise ThreePolicyGPURuntimeError(
                    "streamed backward requires one trajectory per unique prompt"
                )
            medical_prompt_count = len(calibration["rows"])
            for index, row in enumerate(calibration["rows"]):
                current_row, current_row_mask = score_rows(
                    student_model, [row], device="cuda:0", inference=False
                )
                valid_length = int(current_row_mask[0].sum().cpu())
                row_bundle = ThreePolicyLogProbBundle(
                    rollout_behavior_logprob=bundle.rollout_behavior_logprob[
                        index : index + 1, :valid_length
                    ],
                    old_actor_logprob=bundle.old_actor_logprob[
                        index : index + 1, :valid_length
                    ],
                    current_actor_logprob=current_row[:, :valid_length],
                    teacher_logprob=bundle.teacher_logprob[
                        index : index + 1, :valid_length
                    ],
                    response_mask=current_row_mask[:, :valid_length],
                    behavior_provenance=bundle.behavior_provenance,
                )
                validate_three_policy_bundle(
                    row_bundle,
                    require_pre_update_identity=True,
                    identity_tolerance=float(gates["current_pre_old_actor_max_abs"]),
                )
                row_result = decoupled_corrected_objective(
                    row_bundle,
                    prompt_ids=(calibration["prompt_ids"][index],),
                    group_ids=(calibration["group_ids"][index],),
                    source_roles=(calibration["source_roles"][index],),
                    beta=float(algorithm["beta"]),
                    clip_low=float(algorithm["clip_low"]),
                    clip_high=float(algorithm["clip_high"]),
                    rollout_is_threshold=2.0,
                )
                (row_result.loss / float(medical_prompt_count)).backward()
            gradient_norm_before_clip = float(
                torch.nn.utils.clip_grad_norm_(
                [parameter_by_name[name] for name in trainable_names],
                float(optimizer_config["global_gradient_clip_norm"]),
            )
            )
            gradients = {
                name: parameter_by_name[name].grad.detach().cpu().clone()
                for name in trainable_names
                if parameter_by_name[name].grad is not None
            }
            gradient_norm_after_clip = tensor_dict_l2(gradients)
            optimizer.step()
            after_parameters = {
                name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
            }
            medical_checkpoint = {
                "schema_version": 3,
                "status": "optimizer_step_observed_before_post_forward_audit",
                "loss_before": float(before_result.loss.detach().cpu()),
                "objective_before": float(before_result.surrogate.detach().cpu()),
                "gradient_norm": tensor_dict_l2(gradients),
                "gradient_norm_before_clip": gradient_norm_before_clip,
                "gradient_norm_after_clip": gradient_norm_after_clip,
                "parameter_delta_norm": tensor_dict_delta_l2(
                    before_parameters, after_parameters
                ),
                "gradient_parameter_names": sorted(gradients),
                "trainable_parameter_names": sorted(trainable_names),
                "formal_checkpoint_saved": False,
            }
            atomic_write_json(output / "medical_step_checkpoint.json", medical_checkpoint)
            latest_evidence = output / "medical_step_checkpoint.json"
            student_model.eval()
            current_after_scored, after_mask = score_rows(
                student_model, calibration["rows"], device="cuda:0", inference=True
            )
            current_after = current_after_scored.detach().requires_grad_()
            after_bundle = ThreePolicyLogProbBundle(
                rollout_behavior_logprob=bundle.rollout_behavior_logprob,
                old_actor_logprob=bundle.old_actor_logprob,
                current_actor_logprob=current_after,
                teacher_logprob=bundle.teacher_logprob,
                response_mask=after_mask,
                behavior_provenance=bundle.behavior_provenance,
            )
            after_result = decoupled_corrected_objective(
                after_bundle,
                prompt_ids=calibration["prompt_ids"],
                group_ids=calibration["group_ids"],
                source_roles=calibration["source_roles"],
                beta=float(algorithm["beta"]),
                clip_low=float(algorithm["clip_low"]),
                clip_high=float(algorithm["clip_high"]),
                rollout_is_threshold=2.0,
            )
            medical_checkpoint.update(
                {
                    "status": "post_forward_observed_before_optimizer_audit",
                    "loss_after": float(after_result.loss.detach().cpu()),
                    "objective_after": float(after_result.surrogate.detach().cpu()),
                    "ppo_post_ratio": masked_numeric_summary(
                        after_result.ppo_ratio.detach(), bundle.response_mask
                    ),
                    "ppo_post_clip_fraction": after_result.ppo_clip_fraction,
                }
            )
            atomic_write_json(output / "medical_step_checkpoint.json", medical_checkpoint)
            optimizer_audit = audit_optimizer_update(
                before=before_parameters,
                after=after_parameters,
                loss_gradients=gradients,
                declared_trainable_names=trainable_names,
                actual_requires_grad_names=tuple(
                    name for name, parameter in student_model.named_parameters() if parameter.requires_grad
                ),
                fresh_optimizer=True,
                weight_decay=float(optimizer_config["weight_decay"]),
                require_nonzero=True,
                descent_dot_max=0.0,
            )
            delta_logprob = current_after.detach() - bundle.current_actor_logprob.detach()
            alignment = float(
                grouped_trajectory_mean(
                    before_result.correction.truncated_weight
                    * before_result.advantage
                    * delta_logprob,
                    bundle.response_mask,
                    prompt_ids=calibration["prompt_ids"],
                    group_ids=calibration["group_ids"],
                ).cpu()
            )
            update_passed = bool(
                float(after_result.surrogate.detach())
                > float(before_result.surrogate.detach()) + 1e-6
                and float(after_result.loss.detach())
                < float(before_result.loss.detach()) - 1e-6
                and alignment > 0
                and optimizer_audit.hard_gate_passed
                and all(parameter_by_name[name]._version == version for name, version in frozen_versions.items())
            )
            parameter_delta_nonzero_tensor_count = sum(
                int(bool(torch.any(after_parameters[name] != before_parameters[name])))
                for name in trainable_names
            )
            gradient_nonzero_tensor_count = sum(
                int(bool(torch.any(gradients[name] != 0))) for name in gradients
            )
            trainer_v1_ordered_tensor_sha = adapter_tensor_sha(after_parameters)
            medical_metrics = {
                "schema_version": 3,
                "status": "pass" if update_passed else "fail",
                "hard_gate_passed": update_passed,
                "finite": all(
                    math.isfinite(float(value))
                    for value in (
                        before_result.surrogate.detach(),
                        after_result.surrogate.detach(),
                        before_result.loss.detach(),
                        after_result.loss.detach(),
                        alignment,
                        gradient_norm_before_clip,
                        gradient_norm_after_clip,
                        optimizer_audit.parameter_delta_norm,
                    )
                ),
                "advantage": advantage_evidence,
                "objective_before": float(before_result.surrogate.detach().cpu()),
                "objective_after": float(after_result.surrogate.detach().cpu()),
                "loss_before": float(before_result.loss.detach().cpu()),
                "loss_after": float(after_result.loss.detach().cpu()),
                "alignment": alignment,
                "ppo_pre_ratio": masked_numeric_summary(
                    before_result.ppo_ratio.detach(), bundle.response_mask
                ),
                "ppo_post_ratio": masked_numeric_summary(
                    after_result.ppo_ratio.detach(), bundle.response_mask
                ),
                "ppo_pre_clip_fraction": before_result.ppo_clip_fraction,
                "ppo_post_clip_fraction": after_result.ppo_clip_fraction,
                "rollout_correction": correction_artifact(before_result),
                "optimizer_audit": asdict(optimizer_audit),
                "gradient_norm_before_clip": gradient_norm_before_clip,
                "gradient_norm_after_clip": gradient_norm_after_clip,
                "gradient_nonzero_tensor_count": gradient_nonzero_tensor_count,
                "parameter_delta_norm": optimizer_audit.parameter_delta_norm,
                "parameter_delta_nonzero_tensor_count": (
                    parameter_delta_nonzero_tensor_count
                ),
                "trainable_tensor_count": len(trainable_names),
                "trainable_parameter_names": list(trainable_names),
                "teacher_gradient_parameters": [
                    name
                    for name, parameter in teacher_model.named_parameters()
                    if parameter.grad is not None
                ],
                "base_gradient_parameters": [
                    name
                    for name, parameter in student_model.named_parameters()
                    if name not in trainable_names and parameter.grad is not None
                ],
                "base_parameter_versions_unchanged": all(
                    parameter_by_name[name]._version == version
                    for name, version in frozen_versions.items()
                ),
                "trajectory_sha256": trajectory_sha,
                "token_identity_sha256": provenance["token_identity_sha256"],
                "run_id": config["run"]["run_id"],
                "formal_checkpoint_saved": False,
                "P4_2_status_preserved": "failed_identity_mismatch",
                "trainer_v1_ordered_tensor_sha": trainer_v1_ordered_tensor_sha,
                "saved_adapter_file_sha": None,
                "saved_reload_ordered_tensor_sha": None,
            }
            atomic_write_json(output / "corrected_medical_one_step.json", medical_metrics)
            latest_evidence = output / "corrected_medical_one_step.json"
            if not update_passed:
                raise ThreePolicyGPURuntimeError("corrected Medical one-step hard gate failed")
            medical_after_parameters = {
                name: value.clone() for name, value in after_parameters.items()
            }

            def run_real_base_teacher_null_update() -> dict[str, Any]:
                nonlocal current_phase, latest_evidence
                current_phase = "real_base_teacher_null_update"
                with torch.no_grad():
                    for name, value in initial_parameters.items():
                        parameter_by_name[name].copy_(
                            value.to(parameter_by_name[name].device, parameter_by_name[name].dtype)
                        )
                null_before_parameters = {
                    name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
                }
                null_adapter_sha_before = adapter_tensor_sha(null_before_parameters)
                student_model.eval()
                real_base, real_base_mask = score_rows(
                    student_model,
                    calibration["rows"],
                    device="cuda:0",
                    inference=True,
                    disable_adapter=True,
                )
                null_old = real_base.detach()
                # Null Teacher and old actor are two detached identities of one
                # observed real-Base forward. Current is independently checked.
                null_teacher = null_old.detach().clone()
                null_mask = real_base_mask
                null_current_leaf = null_old.clone().requires_grad_()
                null_bundle = ThreePolicyLogProbBundle(
                    rollout_behavior_logprob=bundle.rollout_behavior_logprob,
                    old_actor_logprob=null_old,
                    current_actor_logprob=null_current_leaf,
                    teacher_logprob=null_teacher,
                    response_mask=null_mask,
                    behavior_provenance=bundle.behavior_provenance,
                )
                validate_three_policy_bundle(
                    null_bundle,
                    require_pre_update_identity=True,
                    identity_tolerance=float(gates["current_pre_old_actor_max_abs"]),
                )
                null_before = decoupled_corrected_objective(
                    null_bundle,
                    prompt_ids=calibration["prompt_ids"],
                    group_ids=calibration["group_ids"],
                    source_roles=calibration["source_roles"],
                    beta=float(algorithm["beta"]),
                    clip_low=float(algorithm["clip_low"]),
                    clip_high=float(algorithm["clip_high"]),
                    rollout_is_threshold=2.0,
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
                null_current_base_max_abs = 0.0
                null_prompt_count = len(calibration["rows"])
                for index, row in enumerate(calibration["rows"]):
                    null_current_row, null_current_mask = score_rows(
                        student_model, [row], device="cuda:0", inference=False
                    )
                    valid_length = int(null_current_mask[0].sum().cpu())
                    if not torch.equal(
                        null_current_mask[:, :valid_length],
                        real_base_mask[index : index + 1, :valid_length],
                    ):
                        raise ThreePolicyGPURuntimeError("Base null response mask mismatch")
                    row_old = null_old[index : index + 1, :valid_length]
                    row_max = float(
                        (null_current_row.detach()[:, :valid_length] - row_old)
                        .abs()
                        .max()
                        .cpu()
                    )
                    null_current_base_max_abs = max(null_current_base_max_abs, row_max)
                    row_null_bundle = ThreePolicyLogProbBundle(
                        rollout_behavior_logprob=bundle.rollout_behavior_logprob[
                            index : index + 1, :valid_length
                        ],
                        old_actor_logprob=row_old,
                        current_actor_logprob=null_current_row[:, :valid_length],
                        teacher_logprob=row_old.detach().clone(),
                        response_mask=null_current_mask[:, :valid_length],
                        behavior_provenance=bundle.behavior_provenance,
                    )
                    validate_three_policy_bundle(
                        row_null_bundle,
                        require_pre_update_identity=True,
                        identity_tolerance=float(gates["current_pre_old_actor_max_abs"]),
                    )
                    row_null = decoupled_corrected_objective(
                        row_null_bundle,
                        prompt_ids=(calibration["prompt_ids"][index],),
                        group_ids=(calibration["group_ids"][index],),
                        source_roles=(calibration["source_roles"][index],),
                        beta=float(algorithm["beta"]),
                        clip_low=float(algorithm["clip_low"]),
                        clip_high=float(algorithm["clip_high"]),
                        rollout_is_threshold=2.0,
                    )
                    (row_null.loss / float(null_prompt_count)).backward()
                null_gradients = {
                    name: parameter_by_name[name].grad.detach().cpu().clone()
                    for name in trainable_names
                    if parameter_by_name[name].grad is not None
                }
                null_optimizer.step()
                null_after_parameters = {
                    name: parameter_by_name[name].detach().cpu().clone() for name in trainable_names
                }
                null_adapter_sha_after = adapter_tensor_sha(null_after_parameters)
                student_model.eval()
                null_after_scored, null_after_mask = score_rows(
                    student_model,
                    calibration["rows"],
                    device="cuda:0",
                    inference=True,
                )
                null_after_bundle = ThreePolicyLogProbBundle(
                    rollout_behavior_logprob=bundle.rollout_behavior_logprob,
                    old_actor_logprob=null_old,
                    current_actor_logprob=null_after_scored.detach().requires_grad_(),
                    teacher_logprob=null_teacher,
                    response_mask=null_after_mask,
                    behavior_provenance=bundle.behavior_provenance,
                )
                null_after_result = decoupled_corrected_objective(
                    null_after_bundle,
                    prompt_ids=calibration["prompt_ids"],
                    group_ids=calibration["group_ids"],
                    source_roles=calibration["source_roles"],
                    beta=float(algorithm["beta"]),
                    clip_low=float(algorithm["clip_low"]),
                    clip_high=float(algorithm["clip_high"]),
                    rollout_is_threshold=2.0,
                )
                null_checkpoint = {
                    "schema_version": 3,
                    "status": "null_optimizer_step_observed_before_audit",
                    "advantage_max_abs": float(
                        null_before.advantage[null_mask].abs().max().cpu()
                    ),
                    "gradient_norm": tensor_dict_l2(null_gradients),
                    "parameter_delta_norm": tensor_dict_delta_l2(
                        null_before_parameters, null_after_parameters
                    ),
                    "current_pre_vs_real_base_max_abs": null_current_base_max_abs,
                    "gradient_parameter_names": sorted(null_gradients),
                    "formal_checkpoint_saved": False,
                }
                atomic_write_json(output / "null_update_checkpoint.json", null_checkpoint)
                latest_evidence = output / "null_update_checkpoint.json"
                null_audit = audit_optimizer_update(
                    before=null_before_parameters,
                    after=null_after_parameters,
                    loss_gradients=null_gradients,
                    declared_trainable_names=trainable_names,
                    actual_requires_grad_names=trainable_names,
                    fresh_optimizer=True,
                    weight_decay=float(optimizer_config["weight_decay"]),
                    require_nonzero=False,
                    descent_dot_max=0.0,
                    null_gradient_norm_max=1e-10,
                    null_parameter_delta_norm_max=1e-12,
                )
                null_advantage_max = float(null_before.advantage[null_mask].abs().max().cpu())
                null_passed = bool(
                    null_advantage_max == 0.0
                    and float(null_before.surrogate.detach().cpu()) == 0.0
                    and float(null_after_result.surrogate.detach().cpu()) == 0.0
                    and float(null_before.loss.detach().cpu()) == 0.0
                    and float(null_after_result.loss.detach().cpu()) == 0.0
                    and null_adapter_sha_before == null_adapter_sha_after
                    and null_audit.hard_gate_passed
                )
                null_report = {
                    "schema_version": 3,
                    "status": "pass" if null_passed else "fail",
                    "hard_gate_passed": null_passed,
                    "finite": all(
                        math.isfinite(float(value))
                        for value in (
                            null_before.surrogate.detach(),
                            null_after_result.surrogate.detach(),
                            null_before.loss.detach(),
                            null_after_result.loss.detach(),
                            null_advantage_max,
                            null_audit.gradient_norm,
                            null_audit.parameter_delta_norm,
                        )
                    ),
                    "teacher_logprob_source": "real_base_forward_with_student_adapter_disabled",
                    "old_actor_logprob_source": "same_real_base_forward_detached",
                    "current_actor_logprob_source": "zero_lora_actor_independently_base_equivalent",
                    "current_pre_vs_real_base_max_abs": null_current_base_max_abs,
                    "advantage_max_abs": null_advantage_max,
                    "objective_before": float(
                        null_before.surrogate.detach().cpu()
                    ),
                    "objective_after": float(
                        null_after_result.surrogate.detach().cpu()
                    ),
                    "loss_before": float(null_before.loss.detach().cpu()),
                    "loss_after": float(null_after_result.loss.detach().cpu()),
                    "gradient_norm": null_audit.gradient_norm,
                    "parameter_delta_norm": null_audit.parameter_delta_norm,
                    "adapter_ordered_tensor_sha_before": null_adapter_sha_before,
                    "adapter_ordered_tensor_sha_after": null_adapter_sha_after,
                    "teacher_gradient_parameters": [],
                    "base_gradient_parameters": [
                        name
                        for name, parameter in student_model.named_parameters()
                        if name not in trainable_names and parameter.grad is not None
                    ],
                    "optimizer_audit": asdict(null_audit),
                    "same_correction_objective_reduction_writer": True,
                    "rollout_correction": correction_artifact(null_before),
                    "formal_checkpoint_saved": False,
                }
                atomic_write_json(output / "real_base_teacher_null_update.json", null_report)
                latest_evidence = output / "real_base_teacher_null_update.json"
                if not null_passed:
                    raise ThreePolicyGPURuntimeError("real Base=Teacher null update failed")
                return null_report

            null_report: dict[str, Any] | None = None
            if not is_sampler_refresh_v4:
                null_report = run_real_base_teacher_null_update()

            with torch.no_grad():
                for name, value in medical_after_parameters.items():
                    parameter_by_name[name].copy_(
                        value.to(parameter_by_name[name].device, parameter_by_name[name].dtype)
                    )
            student_model.eval()
            updated_adapter_dir = Path(temporary) / "version1"
            student_model.save_pretrained(updated_adapter_dir, safe_serialization=True)
            updated_adapter_sha = ordered_adapter_sha256(updated_adapter_dir)
            medical_metrics["saved_adapter_file_sha"] = updated_adapter_sha
            atomic_write_json(
                output / "corrected_medical_one_step.json", medical_metrics
            )

            current_phase = "sampler_refresh"
            _release(torch, teacher_model)
            teacher_model = None
            if is_sampler_refresh_v4:
                probe_row = trajectories[0]
                direct_cuda0 = scorer_provenance(
                    device="cuda:0", path="direct_forward_raw_logits", use_cache=False
                )
                direct_cuda1 = scorer_provenance(
                    device="cuda:1", path="direct_forward_raw_logits", use_cache=False
                )

                current_phase = "trainer_v1_in_memory_vs_fresh_reload"
                trainer_ordered_sha = adapter_tensor_sha(student_model)
                trainer_before_ordered_sha = adapter_tensor_sha(initial_parameters)
                with torch.inference_mode():
                    trainer_in_memory_probe = action_logprobs(
                        student_model,
                        probe_row["prompt_ids"],
                        probe_row["response_ids"],
                        device="cuda:0",
                    ).detach().cpu()
                student_model.to("cpu")
                torch.cuda.empty_cache()
                trainer_reload_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:0")
                trainer_reload = PeftModel.from_pretrained(
                    trainer_reload_base,
                    updated_adapter_dir,
                    adapter_name="version1",
                    is_trainable=False,
                )
                del trainer_reload_base
                trainer_reload.eval()
                trainer_reload_ordered_sha = adapter_tensor_sha(trainer_reload)
                with torch.inference_mode():
                    trainer_reload_probe = action_logprobs(
                        trainer_reload,
                        probe_row["prompt_ids"],
                        probe_row["response_ids"],
                        device="cuda:0",
                    ).detach().cpu()
                _release(torch, trainer_reload)
                del trainer_reload
                medical_metrics[
                    "saved_reload_ordered_tensor_sha"
                ] = trainer_reload_ordered_sha
                atomic_write_json(
                    output / "corrected_medical_one_step.json", medical_metrics
                )

                current_phase = "long_lived_sampler_v0_to_v1_refresh"
                sampler_model.to("cuda:1")
                refresh_started_at = datetime.now(timezone.utc).isoformat()
                refresh_started_clock = time.perf_counter()
                sampler_state = refresh_sampler_adapter(
                    sampler_model,
                    adapter_path=updated_adapter_dir,
                    old_version=0,
                    old_sha256=initial_adapter_sha,
                    old_adapter_name="version0_noop",
                    new_version=1,
                    new_sha256=updated_adapter_sha,
                    new_adapter_name="version1",
                )
                refresh_latency = time.perf_counter() - refresh_started_clock
                refresh_ended_at = datetime.now(timezone.utc).isoformat()
                require_sampler_identity(
                    sampler_state,
                    expected_version=1,
                    expected_sha256=updated_adapter_sha,
                )
                sampler_after_ordered_sha = adapter_tensor_sha(sampler_model)
                active_run_token = f"{config['run']['run_id']}:adapter-v1"
                stale_run_token = f"{config['run']['run_id']}:adapter-v0"
                with torch.inference_mode():
                    live_refreshed_probe = guarded_sampler_action_logprobs(
                        sampler_model,
                        probe,
                        device="cuda:1",
                        active_version=1,
                        active_ordered_tensor_sha=sampler_after_ordered_sha,
                        active_run_token=active_run_token,
                        requested_version=1,
                        requested_ordered_tensor_sha=sampler_after_ordered_sha,
                        requested_run_token=active_run_token,
                    ).detach().cpu()
                active_guarded_call_count = 1

                # The fresh reference uses the same backend, GPU, dtype, scorer,
                # prompt/action tokens and cache setting. Only construction
                # history differs from the long-lived refreshed instance.
                sampler_model.to("cpu")
                torch.cuda.empty_cache()
                current_phase = "fresh_sampler_v1_reference"
                fresh_sampler_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:1")
                fresh_sampler = PeftModel.from_pretrained(
                    fresh_sampler_base,
                    updated_adapter_dir,
                    adapter_name="version1",
                    is_trainable=False,
                )
                del fresh_sampler_base
                fresh_sampler.eval()
                fresh_sampler_ordered_sha = adapter_tensor_sha(fresh_sampler)
                with torch.inference_mode():
                    fresh_sampler_probe = action_logprobs(
                        fresh_sampler,
                        probe["prompt_ids"],
                        probe["response_ids"],
                        device="cuda:1",
                    ).detach().cpu()
                _release(torch, fresh_sampler)
                del fresh_sampler
                sampler_model.to("cuda:1")
                current_phase = "live_refreshed_vs_fresh_same_path"

                current_phase = "stale_v0_request_rejection"
                stale_rejected = False
                stale_scoring_executed = False
                stale_request_started = time.perf_counter()
                try:
                    guarded_sampler_action_logprobs(
                        sampler_model,
                        probe,
                        device="cuda:1",
                        active_version=1,
                        active_ordered_tensor_sha=sampler_after_ordered_sha,
                        active_run_token=active_run_token,
                        requested_version=0,
                        requested_ordered_tensor_sha=sampler_before_ordered_sha,
                        requested_run_token=stale_run_token,
                    )
                    stale_scoring_executed = True
                except StaleSamplerRequestError:
                    stale_rejected = True
                stale_request_latency = time.perf_counter() - stale_request_started

                current_phase = "generation_direct_cross_path_diagnostics"
                generation = dict(config["formal_rollout"]["transformers"])
                generation["max_new_tokens"] = 4
                ids = torch.tensor(
                    [[int(value) for value in probe_row["prompt_ids"]]],
                    dtype=torch.long,
                    device="cuda:1",
                )
                with torch.random.fork_rng(devices=[1]):
                    torch.manual_seed(int(config["run"]["seed"]) + 400_000)
                    torch.cuda.manual_seed_all(int(config["run"]["seed"]) + 400_000)
                    generated = guarded_sampler_generate(
                        sampler_model,
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        generation=generation,
                        active_version=1,
                        active_ordered_tensor_sha=sampler_after_ordered_sha,
                        active_run_token=active_run_token,
                        requested_version=1,
                        requested_ordered_tensor_sha=sampler_after_ordered_sha,
                        requested_run_token=active_run_token,
                    )
                active_guarded_call_count += 1
                diagnostic_ids = [
                    int(value)
                    for value in generated.sequences[
                        0, len(probe_row["prompt_ids"]) :
                    ].tolist()
                ]
                processed_values = []
                raw_generation_values = []
                for token, processed_score, raw_logit in zip(
                    diagnostic_ids,
                    generated.scores,
                    generated.logits,
                    strict=True,
                ):
                    processed_values.append(
                        torch.log_softmax(processed_score[0].float(), dim=-1)[token]
                    )
                    raw_generation_values.append(
                        torch.log_softmax(raw_logit[0].float(), dim=-1)[token]
                    )
                processed_tensor = torch.stack(processed_values).cpu()
                raw_generation_tensor = torch.stack(raw_generation_values).cpu()
                diagnostic_row = {
                    "fixture_id": str(probe_row["fixture_id"]) + "-cross-path",
                    "prompt_ids": probe_row["prompt_ids"],
                    "response_ids": diagnostic_ids,
                }
                with torch.inference_mode():
                    sampler_direct_no_cache = guarded_sampler_action_logprobs(
                        sampler_model,
                        diagnostic_row,
                        device="cuda:1",
                        active_version=1,
                        active_ordered_tensor_sha=sampler_after_ordered_sha,
                        active_run_token=active_run_token,
                        requested_version=1,
                        requested_ordered_tensor_sha=sampler_after_ordered_sha,
                        requested_run_token=active_run_token,
                        use_cache=False,
                    ).detach().cpu()
                    sampler_direct_cache = guarded_sampler_action_logprobs(
                        sampler_model,
                        diagnostic_row,
                        device="cuda:1",
                        active_version=1,
                        active_ordered_tensor_sha=sampler_after_ordered_sha,
                        active_run_token=active_run_token,
                        requested_version=1,
                        requested_ordered_tensor_sha=sampler_after_ordered_sha,
                        requested_run_token=active_run_token,
                        use_cache=True,
                    ).detach().cpu()
                active_guarded_call_count += 2
                student_model.to("cuda:0")
                student_model.eval()
                with torch.inference_mode():
                    trainer_cross_probe = action_logprobs(
                        student_model,
                        diagnostic_row["prompt_ids"],
                        diagnostic_ids,
                        device="cuda:0",
                        use_cache=False,
                    ).detach().cpu()
                del generated

                generation_raw_cuda1 = scorer_provenance(
                    device="cuda:1", path="generation_raw_logits", use_cache=True
                )
                generation_processed_cuda1 = scorer_provenance(
                    device="cuda:1",
                    path="generation_processed_scores",
                    use_cache=True,
                )
                direct_cache_cuda1 = scorer_provenance(
                    device="cuda:1", path="direct_forward_raw_logits", use_cache=True
                )
                probes = {
                    "trainer_in_memory_vs_reloaded": probe_payload(
                        "trainer_in_memory_vs_reloaded",
                        classification="formal_same_path_gate",
                        left=trainer_in_memory_probe,
                        right=trainer_reload_probe,
                        row=probe_row,
                        left_scorer=direct_cuda0,
                        right_scorer=direct_cuda0,
                    ),
                    "live_refreshed_vs_fresh_sampler": probe_payload(
                        "live_refreshed_vs_fresh_sampler",
                        classification="formal_same_path_gate",
                        left=live_refreshed_probe,
                        right=fresh_sampler_probe,
                        row=probe,
                        left_scorer=direct_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "repeated_same_instance_noise": probe_payload(
                        "repeated_same_instance_noise",
                        classification="control_diagnostic",
                        left=sampler_v0_first,
                        right=sampler_v0_second,
                        row=probe,
                        left_scorer=direct_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "no_op_refresh_gap": probe_payload(
                        "no_op_refresh_gap",
                        classification="control_diagnostic",
                        left=sampler_v0_first,
                        right=sampler_v0_noop,
                        row=probe,
                        left_scorer=direct_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "generation_raw_vs_direct": probe_payload(
                        "generation_raw_vs_direct",
                        classification="cross_path_diagnostic",
                        left=raw_generation_tensor,
                        right=sampler_direct_no_cache,
                        row=diagnostic_row,
                        left_scorer=generation_raw_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "processed_generation_vs_raw": probe_payload(
                        "processed_generation_vs_raw",
                        classification="cross_path_diagnostic",
                        left=processed_tensor,
                        right=raw_generation_tensor,
                        row=diagnostic_row,
                        left_scorer=generation_processed_cuda1,
                        right_scorer=generation_raw_cuda1,
                    ),
                    "direct_cache_vs_no_cache": probe_payload(
                        "direct_cache_vs_no_cache",
                        classification="cross_path_diagnostic",
                        left=sampler_direct_cache,
                        right=sampler_direct_no_cache,
                        row=diagnostic_row,
                        left_scorer=direct_cache_cuda1,
                        right_scorer=direct_cuda1,
                    ),
                    "trainer_direct_vs_sampler_generation": probe_payload(
                        "trainer_direct_vs_sampler_generation",
                        classification="cross_path_diagnostic",
                        left=trainer_cross_probe,
                        right=raw_generation_tensor,
                        row=diagnostic_row,
                        left_scorer=direct_cuda0,
                        right_scorer=generation_raw_cuda1,
                    ),
                }
                observation = {
                    "run_id": config["run"]["run_id"],
                    "stage": "sampler_refresh",
                    "trainer_version_before": 0,
                    "trainer_version_after": 1,
                    "sampler_version_before": 0,
                    "sampler_version_after": int(sampler_state["version"]),
                    "trainer_ordered_tensor_sha_before": trainer_before_ordered_sha,
                    "trainer_ordered_tensor_sha_after": trainer_ordered_sha,
                    "trainer_saved_adapter_sha": updated_adapter_sha,
                    "trainer_reloaded_ordered_tensor_sha": trainer_reload_ordered_sha,
                    "trainer_reloaded_adapter_sha": ordered_adapter_sha256(
                        updated_adapter_dir
                    ),
                    "sampler_ordered_tensor_sha_before": sampler_before_ordered_sha,
                    "sampler_ordered_tensor_sha_after": sampler_after_ordered_sha,
                    "fresh_sampler_ordered_tensor_sha": fresh_sampler_ordered_sha,
                    "sampler_loaded_adapter_sha": updated_adapter_sha,
                    "active_adapter_name": sampler_state["active_adapter_name"],
                    "old_adapter_name": "version0_noop",
                    "new_adapter_name": "version1",
                    "old_adapter_removed": sampler_state["old_adapter_removed"],
                    "new_adapter_loaded": sampler_state["load_adapter_called"],
                    "base_revision": config["model"]["revision"],
                    "trainer_base_revision": config["model"]["revision"],
                    "sampler_base_revision": config["model"]["revision"],
                    "tokenizer_revision": config["model"]["tokenizer_revision"],
                    "trainer_tokenizer_revision": config["model"]["tokenizer_revision"],
                    "sampler_tokenizer_revision": config["model"]["tokenizer_revision"],
                    "refresh_start": refresh_started_at,
                    "refresh_end": refresh_ended_at,
                    "refresh_latency_seconds": refresh_latency,
                    "stale_request": {
                        "requested_version": 0,
                        "requested_run_token": stale_run_token,
                        "requested_ordered_tensor_sha": sampler_before_ordered_sha,
                        "active_version": 1,
                        "active_run_token": active_run_token,
                        "active_ordered_tensor_sha": sampler_after_ordered_sha,
                        "rejected": stale_rejected,
                        "silent_fallback": False,
                        "scoring_executed": stale_scoring_executed,
                        "routable_adapter_names_after_refresh": sorted(
                            str(name) for name in sampler_model.peft_config
                        ),
                        "error_type": "StaleSamplerRequestError",
                        "error_code": "STALE_SAMPLER_IDENTITY",
                        "rejection_phase": "identity_guard_before_scoring",
                        "latency_seconds": stale_request_latency,
                    },
                    "no_op_refresh_control": no_op_refresh_control,
                    "active_request": {
                        "requested_version": 1,
                        "requested_run_token": active_run_token,
                        "requested_ordered_tensor_sha": sampler_after_ordered_sha,
                        "accepted": True,
                        "scoring_executed": True,
                        "guarded_call_count": active_guarded_call_count,
                        "guarded_request_types": [
                            "fixed_action",
                            "generation",
                            "direct_no_cache",
                            "direct_cache",
                        ],
                        "result_token_count": len(diagnostic_ids),
                        "result_all_finite": bool(
                            torch.isfinite(raw_generation_tensor).all()
                        ),
                    },
                    "probes": probes,
                    "isolation": dict(config["isolation"]),
                }
                refresh_report = build_sampler_refresh_report(
                    observation,
                    threshold=SAMPLER_REFRESH_MAX_GAP,
                    threshold_source=config["sampler_refresh"]["threshold_source"],
                )
                latest_evidence = output / "sampler_refresh.json"
                persist_sampler_refresh_evidence(
                    output,
                    refresh_report,
                    correction_metrics={
                        "ess_fraction": calibration["metrics"]["rollout_correction"][
                            "ess_fraction"
                        ],
                        "cap_fraction": calibration["metrics"]["rollout_correction"][
                            "cap_fraction"
                        ],
                    },
                    one_step_metrics={
                        "objective_delta": medical_metrics["objective_after"]
                        - medical_metrics["objective_before"],
                        "loss_delta": medical_metrics["loss_after"]
                        - medical_metrics["loss_before"],
                    },
                    null_metrics={
                        "status": "not_run_pending_sampler_refresh_gate",
                    },
                    correction_phase=(
                        f"correction_calibration_{calibration['metrics']['rung_prompts']}"
                    ),
                )
                atomic_write_json(
                    output / "sampler_refresh_observations.json",
                    {
                        "schema_version": 4,
                        "status": "complete_report_persisted_before_assertion",
                        "sampler_refresh_sha256": _sha256(
                            output / "sampler_refresh.json"
                        ),
                        "raw_prompt_or_response_persisted": False,
                    },
                )

                # The sampler gate has now persisted and passed. Run the real
                # Base=Teacher null only afterward, as preregistered for v4.
                null_report = run_real_base_teacher_null_update()
                with torch.no_grad():
                    for name, value in medical_after_parameters.items():
                        parameter_by_name[name].copy_(
                            value.to(
                                parameter_by_name[name].device,
                                parameter_by_name[name].dtype,
                            )
                        )
                student_model.eval()
                persist_sampler_refresh_evidence(
                    output,
                    refresh_report,
                    correction_metrics={
                        "ess_fraction": calibration["metrics"]["rollout_correction"][
                            "ess_fraction"
                        ],
                        "cap_fraction": calibration["metrics"]["rollout_correction"][
                            "cap_fraction"
                        ],
                    },
                    one_step_metrics={
                        "objective_delta": medical_metrics["objective_after"]
                        - medical_metrics["objective_before"],
                        "loss_delta": medical_metrics["loss_after"]
                        - medical_metrics["loss_before"],
                    },
                    null_metrics={
                        "status": null_report["status"],
                        "advantage_max_abs": null_report["advantage_max_abs"],
                        "parameter_delta_norm": null_report["parameter_delta_norm"],
                    },
                    correction_phase=(
                        f"correction_calibration_{calibration['metrics']['rung_prompts']}"
                    ),
                )
            else:
                sampler_base = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to("cuda:1")
                sampler_model = PeftModel.from_pretrained(
                    sampler_base,
                    initial_adapter_dir,
                    adapter_name="version0",
                    is_trainable=False,
                )
                del sampler_base
                sampler_model.eval()
                sampler_state = refresh_sampler_adapter(
                    sampler_model,
                    adapter_path=updated_adapter_dir,
                    old_version=0,
                    old_sha256=initial_adapter_sha,
                    old_adapter_name="version0",
                    new_version=1,
                    new_sha256=updated_adapter_sha,
                    new_adapter_name="version1",
                )
                atomic_write_json(
                    output / "sampler_refresh_observations.json",
                    {
                        "schema_version": 3,
                        "status": "adapter_refresh_observed_before_identity_audit",
                        "initial_adapter_version": 0,
                        "initial_adapter_sha256": initial_adapter_sha,
                        "trainer_adapter_version": 1,
                        "trainer_adapter_sha256": updated_adapter_sha,
                        "sampler_state": sampler_state,
                    },
                )
                latest_evidence = output / "sampler_refresh_observations.json"
                require_sampler_identity(
                    sampler_state, expected_version=1, expected_sha256=updated_adapter_sha
                )
                stale_rejected = False
                try:
                    sampler_model.set_adapter("version0")
                except (KeyError, ValueError):
                    stale_rejected = True
                with torch.inference_mode():
                    sampler_probe = action_logprobs(
                        sampler_model,
                        trajectories[0]["prompt_ids"],
                        trajectories[0]["response_ids"],
                        device="cuda:1",
                    ).detach().cpu()
                    trainer_probe = action_logprobs(
                        student_model,
                        trajectories[0]["prompt_ids"],
                        trajectories[0]["response_ids"],
                        device="cuda:0",
                    ).detach().cpu()
                probe_delta = float((sampler_probe - trainer_probe).abs().max())
                refresh_audit = audit_sampler_refresh(
                    old_version=0,
                    old_sha256=initial_adapter_sha,
                    trainer_version=1,
                    trainer_sha256=updated_adapter_sha,
                    sampler_version=int(sampler_state["version"]),
                    sampler_sha256=str(sampler_state["adapter_sha256"]),
                    probe_max_abs_delta=probe_delta,
                    probe_tolerance=1e-4,
                    stale_adapter_rejected=stale_rejected,
                )
                refresh_report = {
                    **refresh_audit,
                    **sampler_state,
                    "schema_version": 3,
                    "status": "pass",
                    "actual_adapter_refresh_verified": True,
                    "cache_identity_reused": False,
                    "formal_checkpoint_saved": False,
                }
                atomic_write_json(output / "sampler_refresh.json", refresh_report)
                latest_evidence = output / "sampler_refresh.json"

        current_phase = "artifact_readiness"
        readiness = {
            "schema_version": 4 if is_sampler_refresh_v4 else 3,
            "status": "three_policy_revalidation_runtime_passed_pending_post_exit_cleanup",
            "generation_provenance_ready": True,
            "fresh_full_support_rollout_ready": True,
            "correction_calibration_ready": True,
            "corrected_medical_one_step_ready": True,
            "real_null_update_ready": True,
            "sampler_refresh_ready": True,
            "opd_backend_ready": False,
            "B2_authorized": False,
            "P4_2_status_preserved": "failed_identity_mismatch",
        }
        atomic_write_json(output / "readiness.json", readiness)
        current_phase = "release_gpu_resources"
        _release(torch, student_model, teacher_model, sampler_model)
        student_model = teacher_model = sampler_model = None
        release = {
            "schema_version": 3,
            "status": "pass",
            "models_released": True,
            "cuda_allocated_bytes_diagnostic": [
                int(torch.cuda.memory_allocated(index)) for index in range(2)
            ],
            "cuda_reserved_bytes_diagnostic": [
                int(torch.cuda.memory_reserved(index)) for index in range(2)
            ],
            "post_process_exit_verification_required": True,
        }
        atomic_write_json(output / "runtime_release.json", release)
        if not is_sampler_refresh_v4:
            _atomic_jsonl(
                output / "metrics.jsonl",
                [
                    {"step": 0, "phase": "generation_provenance", "status": "pass"},
                    {"step": 1, "phase": "correction_calibration_4", "status": "pass"},
                    {"step": 2, "phase": "correction_calibration_16", "status": "pass"},
                    {"step": 3, "phase": "corrected_medical_one_step", "status": "pass"},
                    {"step": 4, "phase": "real_base_teacher_null_update", "status": "pass"},
                    {"step": 5, "phase": "sampler_refresh", "status": "pass"},
                ],
            )
        summary = {
            "schema_version": 3,
            "status": "ready_for_post_exit_resource_cleanup_verification",
            "elapsed_seconds": time.time() - started,
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
            "formal_opd_authorized": False,
            "next_step": "post_exit_cleanup_finalizer",
            **identity,
        }
        atomic_write_json(output / "summary.json", summary)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        metadata.update(
            {
                "status": summary["status"],
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
        atomic_write_json(output / "metadata.json", metadata)
        return summary
    except Exception as error:
        status_by_phase = {
            "generation_score_provenance_micro_probe": "failed_generation_provenance",
            "fresh_full_support_rollout_4": "failed_fresh_rollout",
            "fresh_full_support_rollout_16": "failed_fresh_rollout",
            "fresh_full_support_rollout_32": "failed_fresh_rollout",
            "correction_calibration_4": "failed_three_policy_calibration",
            "correction_calibration_16": "failed_three_policy_calibration",
            "correction_calibration_32": "failed_three_policy_calibration",
            "corrected_medical_one_step": "failed_corrected_medical_one_step",
            "real_base_teacher_null_update": "failed_null_update",
            "sampler_refresh": "failed_sampler_refresh",
            "sampler_v0_repeated_probe": "failed_same_instance_repeat",
            "sampler_v0_noop_unload_reload_control": "failed_no_op_refresh",
            "trainer_v1_in_memory_vs_fresh_reload": "failed_sampler_refresh",
            "long_lived_sampler_v0_to_v1_refresh": "failed_sampler_refresh",
            "fresh_sampler_v1_reference": "failed_sampler_refresh",
            "live_refreshed_vs_fresh_same_path": "failed_sampler_refresh",
            "stale_v0_request_rejection": "failed_sampler_refresh",
            "generation_direct_cross_path_diagnostics": "failed_sampler_refresh",
            "artifact_readiness": "failed_artifact_integrity",
            "release_gpu_resources": "failed_resource_release",
        }
        failure_status = status_by_phase.get(current_phase, "failed_artifact_integrity")
        failure = {
            "schema_version": 3,
            "status": failure_status,
            "phase": current_phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "latest_evidence": latest_evidence.name if latest_evidence else None,
            "P4_2_status_preserved": "failed_identity_mismatch",
            "B2_authorized": False,
        }
        if is_sampler_refresh_v4 and not (output / "failure.json").is_file():
            if (output / "sampler_refresh.json").is_file():
                persist_sampler_refresh_failure_binding(
                    output,
                    run_id=config["run"]["run_id"],
                    failed_phase=current_phase,
                    failure_status=failure_status,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            else:
                observed_aggregates = _observed_aggregate_metrics(output)
                persist_sampler_refresh_runtime_failure(
                    output,
                    run_id=config["run"]["run_id"],
                    failed_phase=current_phase,
                    failure_status=failure_status,
                    error_type=type(error).__name__,
                    error=str(error),
                    correction_metrics=observed_aggregates["correction"],
                    one_step_metrics=observed_aggregates["one_step"],
                    null_metrics=observed_aggregates["null"],
                )
        if not (output / "failure.json").is_file():
            atomic_write_json(output / "failure.json", failure)
        atomic_write_json(
            output / "summary.json",
            {
                "schema_version": 3,
                "status": failure_status,
                "failure_phase": current_phase,
                "return_to_cpu_decision": True,
                "B2_authorized": False,
            },
        )
        raise
    finally:
        try:
            _release(torch, student_model, teacher_model, sampler_model)
        except Exception:
            pass
