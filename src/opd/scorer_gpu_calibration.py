"""Authorized GPU runtime for the narrow P4.0 scorer calibration only.

Imports of torch/Transformers/PEFT occur only inside ``run_gpu_calibration``.
CPU preflight and unit tests can therefore import this module without touching
CUDA or resolving a real model.
"""

from __future__ import annotations

import argparse
import gc
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
from typing import Any, Mapping, Sequence


class GPUCalibrationError(RuntimeError):
    pass


def calibration_runtime_contract() -> dict[str, Any]:
    return {
        "student_gpu": 0,
        "teacher_gpu": 1,
        "teacher_base_backbone_instances": 1,
        "medical_adapter_merged": False,
        "teacher_generates": False,
        "teacher_retokenizes": False,
        "full_opd_allowed": False,
        "formal_student_checkpoint_allowed": False,
        "formal_backend": "transformers_direct_trajectory_logits",
        "sampler_refresh": "temporary_lora_adapter_version_only",
    }


def calibration_plan(config: Mapping[str, Any]) -> list[str]:
    run = config.get("run", {})
    resources = config.get("resources", {})
    algorithm = config.get("algorithm", {})
    scoring = config.get("scoring", {})
    if run.get("calibration_only") is not True or run.get("formal_opd_training") is not False:
        raise GPUCalibrationError("formal OPD is forbidden in scorer calibration")
    if algorithm.get("one_step_only") is not True or algorithm.get("save_student_checkpoint") is not False:
        raise GPUCalibrationError("formal checkpoint or multi-step update is forbidden")
    if (
        algorithm.get("beta") != 1.0
        or algorithm.get("use_task_rewards") is not False
        or algorithm.get("reference_policy_kl") is not False
        or algorithm.get("old_logprob_source") != "sampling_time_policy"
    ):
        raise GPUCalibrationError("GPU calibration algorithm contract drift")
    if (
        resources.get("required_gpus") != 2
        or resources.get("student_gpu") != 0
        or resources.get("teacher_gpu") != 1
    ):
        raise GPUCalibrationError("calibration requires GPU0 Student and GPU1 Teacher")
    if scoring.get("formal_backend") != "transformers_direct_trajectory_logits":
        raise GPUCalibrationError("Transformers must be the formal reference scorer")
    vllm = scoring.get("vllm", {})
    if vllm.get("formal_enabled") is not False or vllm.get("diagnostic_only") is not True:
        raise GPUCalibrationError("vLLM must remain diagnostic_only")
    return [
        "transformers_teacher_repeatability",
        "route_state_isolation",
        "vllm_diagnostic_equivalence",
        "live_rollout_same_model_null",
        "one_step_direction_no_checkpoint",
        "null_one_step",
        "sampler_adapter_refresh",
        "write_calibration_artifacts",
        "release_all_gpu_resources",
    ]


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 1.0 if list(left) == list(right) else 0.0
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else (1.0 if list(left) == list(right) else 0.0)


def same_model_null_metrics(
    *, rollout_logprobs: Sequence[float], transformers_base_logprobs: Sequence[float]
) -> dict[str, float]:
    if len(rollout_logprobs) != len(transformers_base_logprobs) or not rollout_logprobs:
        raise GPUCalibrationError("same-model null vectors must be non-empty and aligned")
    values = [
        float(teacher) - float(old)
        for old, teacher in zip(rollout_logprobs, transformers_base_logprobs, strict=True)
    ]
    if not all(math.isfinite(value) for value in values):
        raise GPUCalibrationError("same-model null values must be finite")
    absolute = [abs(value) for value in values]
    near_zero_tolerance = 1e-6
    standard_deviation = statistics.pstdev(values)
    return {
        "advantage_mean": statistics.mean(values),
        "advantage_std": standard_deviation,
        "advantage_min": min(values),
        "advantage_max": max(values),
        "mean_absolute": statistics.mean(absolute),
        "p50_absolute": _percentile(absolute, 0.50),
        "p95_absolute": _percentile(absolute, 0.95),
        "max_absolute": max(absolute),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "negative_fraction": sum(value < 0 for value in values) / len(values),
        "near_zero_fraction": sum(abs(value) <= near_zero_tolerance for value in values) / len(values),
        "pearson": _pearson(rollout_logprobs, transformers_base_logprobs),
        "constant_offset_candidate": bool(
            abs(statistics.mean(values)) > near_zero_tolerance
            and standard_deviation <= max(1e-4, abs(statistics.mean(values)) * 0.1)
        ),
    }


def lora_update_summary(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if set(before) != set(after) or not before:
        raise GPUCalibrationError("LoRA parameter snapshots must have identical non-empty keys")
    delta_squared = 0.0
    before_squared = 0.0
    maximum = 0.0
    updated = 0
    finite = True
    for name in sorted(before):
        left = before[name].detach().float().cpu()
        right = after[name].detach().float().cpu()
        if tuple(left.shape) != tuple(right.shape):
            raise GPUCalibrationError(f"LoRA parameter shape changed: {name}")
        delta = right - left
        if not bool(delta.isfinite().all()) or not bool(left.isfinite().all()) or not bool(right.isfinite().all()):
            finite = False
        squared = float((delta * delta).sum())
        delta_squared += squared
        before_squared += float((left * left).sum())
        maximum = max(maximum, float(delta.abs().max()))
        updated += int(squared > 0)
    update_l2 = math.sqrt(delta_squared)
    before_l2 = math.sqrt(before_squared)
    return {
        "finite": finite,
        "parameter_tensors": len(before),
        "updated_tensors": updated,
        "update_l2": update_l2,
        "update_max_abs": maximum,
        "relative_update_l2": update_l2 / before_l2 if before_l2 else None,
    }


def ordered_adapter_sha256(root: str | Path) -> str:
    digest = hashlib.sha256()
    base = Path(root)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = base / name
        if not path.is_file():
            raise GPUCalibrationError(f"temporary Student adapter lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def public_live_rollout_manifest(
    rows: Sequence[Mapping[str, Any]], *, model_revision: str,
    tokenizer_revision: str, rollout_backend: str,
) -> dict[str, Any]:
    public_rows = []
    for row in rows:
        prompt_ids = [int(value) for value in row["prompt_ids"]]
        response_ids = [int(value) for value in row["response_ids"]]
        response_mask = [int(value) for value in row["response_mask"]]
        identity = {
            "prompt": prompt_ids,
            "response": response_ids,
            "mask": response_mask,
            "finish_reason": row["finish_reason"],
            "truncated": bool(row["truncated"]),
        }
        trajectory_sha = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        public_rows.append({
            "fixture_id": str(row["fixture_id"]),
            "source_role": str(row["source_role"]),
            "prompt_length": len(prompt_ids),
            "response_length": len(response_ids),
            "response_mask_count": sum(response_mask),
            "eos_generated": bool(row["eos_generated"]),
            "eos_position": row.get("eos_position"),
            "finish_reason": str(row["finish_reason"]),
            "truncated": bool(row["truncated"]),
            "seed": int(row["seed"]),
            "trajectory_sha256": trajectory_sha,
        })
    return {
        "schema_version": 1,
        "count": len(public_rows),
        "contains_labels": False,
        "contains_raw_text": False,
        "contains_token_ids": False,
        "contains_per_token_logprobs": False,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "rollout_backend": rollout_backend,
        "rows": public_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    from src.opd.pg_opd_validation import atomic_write_json

    atomic_write_json(path, value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _apply_determinism(torch: Any) -> None:
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def reset_peak_memory_stats(torch: Any, device_index: int) -> None:
    """Reset metrics only after PyTorch has initialized its CUDA device table."""

    if not torch.cuda.is_initialized():
        torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device_index)


def _student_action_logprobs(model: Any, prompt_ids: list[int], response_ids: list[int], torch: Any):
    combined = prompt_ids + response_ids
    ids = torch.tensor([combined], dtype=torch.long, device="cuda:0")
    attention = torch.ones_like(ids)
    output = model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
    start = len(prompt_ids) - 1
    logits = output.logits[:, start : start + len(response_ids), :].float()
    targets = torch.tensor(response_ids, dtype=torch.long, device="cuda:0").view(1, -1, 1)
    return torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)


def _release(torch: Any, *models: Any) -> None:
    for model in models:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
    del models
    gc.collect()
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_index)


def _synchronize_same_object_sampler(actor: Any, rollout_model: Any) -> dict[str, Any]:
    """Exercise the project-local sampler synchronization boundary.

    The narrow calibration deliberately time-shares one Student object between
    rollout and update, so no weight copy is required.  Formal veRL execution
    will separately exercise ``engine_workers.update_weights``.
    """

    if actor is not rollout_model:
        raise GPUCalibrationError("project-local sampler no longer shares the actor object")
    return {
        "callable": True,
        "mode": "same_process_shared_actor_object",
        "formal_verl_update_weights_verified": False,
    }


def run_gpu_calibration(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Run the bounded repeatability/null/direction calibration.

    This function is unreachable from the CPU default launcher and is called
    only after the separate execute-gate preflight succeeds.
    """

    plan = calibration_plan(config)
    if os.environ.get("CA_OPD_ALLOW_OPD_SCORER_CALIBRATION_GPU") != "1":
        raise GPUCalibrationError("GPU calibration lacks explicit authorization")
    from src.opd.scorer_preflight import preflight

    preflight_report = preflight(config, execute_gpu=True, require_clean_git=True)
    # Lazy GPU imports begin here.
    import torch
    import yaml
    from importlib.metadata import version
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    from src.opd.pg_opd_contract import (
        ppo_clipped_objective,
        same_trajectory_advantage,
    )
    from src.opd.scorer_calibration import (
        classify_calibration_failure,
        compare_backends,
        summarize_signed_update,
        validate_artifact_inventory,
        validate_artifact_size_budget,
        validate_repeatability,
        validate_route_isolation,
        validate_sampler_refresh,
    )
    from src.opd.trajectory_scorer import (
        SharedBackboneRoutes,
        TrajectoryScoreRequest,
        TransformersTrajectoryLogprobScorer,
    )

    _apply_determinism(torch)
    repo = Path(__file__).resolve().parents[2]
    output = Path(config["run"]["output_dir"])
    if output.exists():
        raise GPUCalibrationError("calibration output must not already exist")
    output.mkdir(parents=True)
    stdout_path = output / "stdout.log"
    stdout_path.touch()
    (output / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    current_phase = "transformers_replay"
    phase_timings: dict[str, float] = {}
    phase_memory: dict[str, Any] = {}

    def phase_log(phase: str, status: str, **values: Any) -> None:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "status": status,
            **values,
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def numeric_summary(values: Sequence[float]) -> dict[str, float]:
        numeric = [float(value) for value in values]
        if not numeric or not all(math.isfinite(value) for value in numeric):
            raise GPUCalibrationError("numeric summary requires non-empty finite values")
        return {
            "mean": statistics.mean(numeric),
            "std": statistics.pstdev(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "p50": _percentile(numeric, 0.50),
            "p95": _percentile(numeric, 0.95),
        }

    metadata = {
        "run_id": config["run"]["run_id"],
        "stage": "opd_scorer_calibration",
        "calibration_only": True,
        "formal_opd_training": False,
        "git_sha": _git_sha(repo),
        "start": started_iso,
        "status": "running",
        "failure_reason": None,
        "runtime_contract": calibration_runtime_contract(),
        "plan": plan,
        "actual_cost_cny": None,
        "price_cny_per_hour": 2.96,
        "user_instance_start_time": os.environ.get("CA_OPD_INSTANCE_START_TIME", "unknown"),
        "codex_takeover_time": os.environ.get("CA_OPD_CODEX_TAKEOVER_TIME", "unknown"),
        "model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "base_artifact_manifest_sha256": config["model"]["artifact_manifest_sha256"],
        "teacher_manifest_sha256": config["teacher"]["manifest_sha256"],
        "medical_adapter_sha256": config["teacher"]["adapter_sha256"],
        "medical_adapter_weight_sha256": config["teacher"]["adapter_weight_sha256"],
        "trajectory_manifest_sha256": config["data"]["trajectory_manifest_file_sha256"],
        "live_prompt_manifest_sha256": config["data"]["live_prompt_manifest_file_sha256"],
        "seed": int(config["run"]["seed"]),
        "backend": "transformers_direct_trajectory_logits",
        "versions": {
            name: version(name) for name in ("torch", "transformers", "peft", "vllm", "verl")
        },
        "gpus": preflight_report["gpu_inventory"],
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
    }
    _atomic_json(output / "metadata.json", metadata)
    teacher_model = student_model = scorer = None
    diagnostic_output = output / "vllm_token_logprobs.jsonl"
    raw_transformers = output / "transformers_token_logprobs.jsonl"
    raw_live = output / "live_trajectory_logprobs.jsonl"
    try:
        replay = _jsonl(Path(config["data"]["private_replay_path"]))
        live_prompts = _jsonl(Path(config["data"]["private_live_prompt_path"]))
        if len(replay) != 12 or len(live_prompts) != 4:
            raise GPUCalibrationError("runtime fixture count differs from frozen 12/4 contract")
        model_path = config["model"]["id"]

        # Phase A: authoritative Transformers replay and explicit route-state isolation.
        phase_started = time.time()
        phase_log(current_phase, "started", replay_count=12, repeats=3)
        reset_peak_memory_stats(torch, 1)
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
            config["teacher"]["adapter_path"],
            adapter_name="medical",
            is_trainable=False,
        )
        del teacher_base
        routes = SharedBackboneRoutes(
            model=teacher_model,
            medical_adapter_name="medical",
            medical_adapter_sha256=config["teacher"]["adapter_sha256"],
        )
        scorer = TransformersTrajectoryLogprobScorer(
            model=teacher_model,
            routes=routes,
            model_id=model_path,
            model_revision=config["model"]["revision"],
            tokenizer_revision=config["model"]["tokenizer_revision"],
            logprob_chunk_tokens=64,
        )
        all_scores: list[dict[str, Any]] = []
        repeat_reports: dict[str, Any] = {}
        route_first: dict[tuple[str, str], dict[str, Any]] = {}
        call_metrics: list[dict[str, Any]] = []
        results_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {
            (row["fixture_id"], route): []
            for row in replay for route in ("base", "medical")
        }
        for route in ("base", "medical"):
            for repeat in range(3):
                requests = [
                    TrajectoryScoreRequest(
                        request_id=f"{row['fixture_id']}-{route}-{repeat}",
                        route=route,
                        prompt_ids=tuple(row["prompt_ids"]),
                        response_ids=tuple(row["response_ids"]),
                        attention_mask=(1,) * (len(row["prompt_ids"]) + len(row["response_ids"])),
                        eos_token_id=getattr(teacher_model.config, "eos_token_id", None),
                        finish_reason="length" if row["truncated"] else "stop",
                        truncated=bool(row["truncated"]),
                        source_role=row["source_role"],
                    )
                    for row in replay
                ]
                call_started = time.perf_counter()
                results = scorer.score_batch(
                    requests,
                    maximum_batch_size=int(config["scoring"]["maximum_batch_size"]),
                    length_bucket_width=int(config["scoring"]["length_bucket_width"]),
                )
                call_elapsed = time.perf_counter() - call_started
                call_metrics.append({
                    "route": route,
                    "repeat": repeat,
                    "elapsed_seconds": call_elapsed,
                    "response_tokens": sum(len(row["response_ids"]) for row in replay),
                })
                for row, value in zip(replay, results, strict=True):
                    result = asdict(value)
                    results_by_key[(row["fixture_id"], route)].append(result)
                    all_scores.append(result)
        for row in replay:
            for route in ("base", "medical"):
                runs = results_by_key[(row["fixture_id"], route)]
                repeat_reports[f"{row['fixture_id']}:{route}"] = validate_repeatability(
                    runs, tolerance=1e-4
                )
                route_first[(row["fixture_id"], route)] = runs[0]
        transformers_repeat_max = max(
            item["max_abs_delta"] for item in repeat_reports.values()
        )
        base_repeat_max = max(
            item["max_abs_delta"]
            for key, item in repeat_reports.items() if key.endswith(":base")
        )
        if base_repeat_max > 1e-6:
            raise GPUCalibrationError(
                f"same-Transformers Base null drift {base_repeat_max:.9g} exceeds 1e-6"
            )
        base_repeat_0 = []
        base_repeat_1 = []
        for row in replay:
            base_repeat_0.extend(results_by_key[(row["fixture_id"], "base")][0]["token_logprobs"])
            base_repeat_1.extend(results_by_key[(row["fixture_id"], "base")][1]["token_logprobs"])
        hard_null = same_model_null_metrics(
            rollout_logprobs=base_repeat_0,
            transformers_base_logprobs=base_repeat_1,
        )
        hard_null.update({
            "status": "pass",
            "implementation": "same_transformers_base_scorer",
            "max_abs_tolerance": 1e-6,
            "max_abs_delta": base_repeat_max,
        })
        route_diffs = []
        for row in replay:
            base_values = route_first[(row["fixture_id"], "base")]["token_logprobs"]
            medical_values = route_first[(row["fixture_id"], "medical")]["token_logprobs"]
            route_diffs.extend(
                abs(float(base) - float(medical))
                for base, medical in zip(base_values, medical_values, strict=True)
            )
        if not any(value > 0 for value in route_diffs):
            raise GPUCalibrationError("Base and Medical Teacher routes are all identical")

        current_phase = "route_isolation"
        isolation_started = time.time()
        phase_log(current_phase, "started")
        switch_orders = {
            "base_medical_base": ("base", "medical", "base"),
            "medical_base_medical": ("medical", "base", "medical"),
        }
        isolation_sequences: dict[str, list[dict[str, Any]]] = {}
        isolation_per_fixture: dict[str, Any] = {}
        route_switch_seconds: list[float] = []
        for order_name, order in switch_orders.items():
            aggregate_observations = []
            step_results: list[list[dict[str, Any]]] = []
            for step_index, route in enumerate(order):
                requests = [
                    TrajectoryScoreRequest(
                        request_id=f"isolation-{order_name}-{step_index}-{row['fixture_id']}",
                        route=route,
                        prompt_ids=tuple(row["prompt_ids"]),
                        response_ids=tuple(row["response_ids"]),
                        attention_mask=(1,) * (len(row["prompt_ids"]) + len(row["response_ids"])),
                        eos_token_id=getattr(teacher_model.config, "eos_token_id", None),
                        finish_reason="length" if row["truncated"] else "stop",
                        truncated=bool(row["truncated"]),
                        source_role=row["source_role"],
                    )
                    for row in replay
                ]
                switch_started = time.perf_counter()
                values = scorer.score_batch(
                    requests,
                    maximum_batch_size=int(config["scoring"]["maximum_batch_size"]),
                    length_bucket_width=int(config["scoring"]["length_bucket_width"]),
                )
                route_switch_seconds.append(time.perf_counter() - switch_started)
                serialized = [asdict(value) for value in values]
                step_results.append(serialized)
                aggregate_observations.append({
                    "route": route,
                    "adapter_sha": serialized[0]["adapter_sha"],
                    "token_ids": [token for result in serialized for token in result["token_ids"]],
                    "token_logprobs": [
                        score for result in serialized for score in result["token_logprobs"]
                    ],
                    "response_mask": [
                        mask for result in serialized for mask in result["response_mask"]
                    ],
                })
            isolation_sequences[order_name] = aggregate_observations
            repeat_left, repeat_right = step_results[0], step_results[2]
            for row, left, right in zip(replay, repeat_left, repeat_right, strict=True):
                maximum = max(
                    abs(float(a) - float(b))
                    for a, b in zip(
                        left["token_logprobs"], right["token_logprobs"], strict=True
                    )
                )
                isolation_per_fixture[f"{order_name}:{row['fixture_id']}"] = maximum
        route_isolation = validate_route_isolation(
            isolation_sequences,
            medical_adapter_sha256=config["teacher"]["adapter_sha256"],
            tolerance=1e-6,
        )
        route_isolation.update({
            "status": "pass",
            "per_fixture_max_same_route_abs_delta": isolation_per_fixture,
            "route_switch_seconds": {
                "mean": statistics.mean(route_switch_seconds),
                "p95": _percentile(route_switch_seconds, 0.95),
                "max": max(route_switch_seconds),
            },
        })
        _atomic_json(output / "route_isolation.json", route_isolation)
        phase_timings["route_isolation_seconds"] = time.time() - isolation_started
        phase_memory["transformers_replay_gpu1"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(1)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(1)),
        }
        phase_timings["transformers_scorer_seconds"] = time.time() - phase_started
        total_scored_tokens = sum(item["response_tokens"] for item in call_metrics)
        total_call_seconds = sum(item["elapsed_seconds"] for item in call_metrics)
        length_buckets: dict[str, list[dict[str, float]]] = {}
        for result in all_scores:
            bucket = str(
                ((int(result["response_length"]) - 1) // 128 + 1) * 128
            )
            length_buckets.setdefault(bucket, []).append({
                "tokens": float(result["response_length"]),
                "seconds": float(result["elapsed_seconds"]),
            })
        repeatability_report = {
            "status": "pass",
            "runs_per_route_per_trajectory": 3,
            "tolerance": 1e-4,
            "same_transformers_base_tolerance": 1e-6,
            "max_abs_delta": transformers_repeat_max,
            "base_max_abs_delta": base_repeat_max,
            "finite": True,
            "token_alignment": True,
            "routes": repeat_reports,
            "route_difference": {
                **numeric_summary(route_diffs),
                "mean_absolute": statistics.mean(route_diffs),
                "nonzero_token_fraction": sum(value > 0 for value in route_diffs) / len(route_diffs),
            },
            "tokens_per_second": total_scored_tokens / total_call_seconds,
            "length_buckets": {
                bucket: {
                    "observations": len(items),
                    "tokens": int(sum(item["tokens"] for item in items)),
                    "mean_elapsed_seconds": statistics.mean(item["seconds"] for item in items),
                }
                for bucket, items in sorted(length_buckets.items(), key=lambda item: int(item[0]))
            },
            "gpu1_memory": phase_memory["transformers_replay_gpu1"],
        }
        _atomic_json(output / "repeatability.json", repeatability_report)
        _atomic_json(output / "route_manifest.json", {
            "backend": scorer.backend,
            "integration": "verl_integrated_custom_teacher_scorer",
            "shared_backbone_instances": 1,
            "base_adapter": "explicitly_disabled",
            "medical_adapter_name": "medical",
            "medical_adapter_sha256": config["teacher"]["adapter_sha256"],
            "medical_adapter_merged": False,
            "teacher_generates": False,
            "teacher_retokenizes": False,
            "route_isolation_passed": True,
        })
        with raw_transformers.open("w", encoding="utf-8") as handle:
            for value in all_scores:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        phase_log(
            "transformers_replay", "passed",
            max_abs_delta=transformers_repeat_max,
            base_null_max_abs_delta=base_repeat_max,
            tokens_per_second=repeatability_report["tokens_per_second"],
        )
        phase_log("route_isolation", "passed", max_abs_delta=route_isolation["max_same_route_abs_delta"])

        # Phase B: release Transformers then run one bounded diagnostic-only vLLM attempt.
        _release(torch, teacher_model)
        teacher_model = scorer = None
        current_phase = "vllm_diagnostic"
        phase_started = time.time()
        phase_log(current_phase, "started", diagnostic_only=True)
        diagnostic_env = dict(os.environ)
        diagnostic_env.update({
            "CUDA_VISIBLE_DEVICES": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V1": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        diagnostic_process_log = output / "vllm_process.log"
        with diagnostic_process_log.open("w", encoding="utf-8") as log_handle:
            diagnostic = subprocess.run(
                [
                    os.environ.get("CA_OPD_PYTHON", "artifacts/env/bin/python"),
                    "-m", "src.opd.vllm_trajectory_diagnostic",
                    "--config", str(config_path),
                    "--output", str(diagnostic_output),
                ],
                cwd=repo,
                env=diagnostic_env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            if diagnostic.returncode != 0:
                tail = diagnostic_process_log.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise GPUCalibrationError(tail or "vLLM diagnostic failed")
            vllm_rows = _jsonl(diagnostic_output)
            diagnostic_repeatability = {}
            for row in replay:
                fixture_id = row["fixture_id"]
                for route in ("base", "medical"):
                    repetitions = [
                        item for item in vllm_rows
                        if item["fixture_id"] == fixture_id and item["route"] == route
                    ]
                    try:
                        diagnostic_repeatability[f"{fixture_id}:{route}"] = validate_repeatability(
                            repetitions, tolerance=1e-4
                        )
                    except Exception as error:
                        diagnostic_repeatability[f"{fixture_id}:{route}"] = {
                            "passed": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
            vllm_by_key = {
                (row["fixture_id"], row["route"]): row
                for row in vllm_rows if row["repeat"] == 0
            }
            reference_values: list[float] = []
            candidate_values: list[float] = []
            reference_gaps: list[float] = []
            candidate_gaps: list[float] = []
            for row in replay:
                fixture_id = row["fixture_id"]
                base_ref = route_first[(fixture_id, "base")]["token_logprobs"]
                medical_ref = route_first[(fixture_id, "medical")]["token_logprobs"]
                base_can = vllm_by_key[(fixture_id, "base")]["token_logprobs"]
                medical_can = vllm_by_key[(fixture_id, "medical")]["token_logprobs"]
                ref_gap = [
                    float(medical) - float(base)
                    for base, medical in zip(base_ref, medical_ref, strict=True)
                ]
                can_gap = [
                    float(medical) - float(base)
                    for base, medical in zip(base_can, medical_can, strict=True)
                ]
                for ref, candidate in ((base_ref, base_can), (medical_ref, medical_can)):
                    reference_values.extend(ref)
                    candidate_values.extend(candidate)
                    reference_gaps.extend(ref_gap)
                    candidate_gaps.extend(can_gap)
            equivalence = compare_backends(
                reference_values,
                candidate_values,
                reference_gaps=reference_gaps,
                candidate_gaps=candidate_gaps,
            )
            all_repeatability_passed = all(
                item.get("passed") is True for item in diagnostic_repeatability.values()
            )
            equivalence["passed"] = bool(equivalence["passed"] and all_repeatability_passed)
            equivalence.update({
                "status": "pass" if equivalence["passed"] else "failed_nonblocking",
                "vllm_diagnostic_pass": bool(equivalence["passed"]),
                "backend_status": "diagnostic_only",
                "formal_enabled": False,
                "diagnostic_only": True,
                "repeatability": diagnostic_repeatability,
                "all_repeatability_passed": all_repeatability_passed,
                "transformers_formal_gate_affected": False,
                "bounded_attempts": 1,
            })
        except Exception as diagnostic_error:
            equivalence = {
                "status": "failed_nonblocking",
                "passed": False,
                "vllm_diagnostic_pass": False,
                "backend_status": "diagnostic_only",
                "formal_enabled": False,
                "diagnostic_only": True,
                "diagnostic_error": f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                "transformers_formal_gate_affected": False,
                "bounded_attempts": 1,
            }
        _atomic_json(output / "vllm_diagnostic.json", equivalence)
        phase_timings["vllm_diagnostic_seconds"] = time.time() - phase_started
        phase_log(
            current_phase,
            "passed" if equivalence["passed"] else "failed_nonblocking",
            diagnostic_only=True,
        )
        torch.cuda.empty_cache()

        # Phase C/D: reload the authoritative shared Teacher and make four live Base rollouts.
        current_phase = "live_rollout"
        phase_started = time.time()
        phase_log(current_phase, "started", prompt_count=4)
        reset_peak_memory_stats(torch, 0)
        reset_peak_memory_stats(torch, 1)
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
            config["teacher"]["adapter_path"],
            adapter_name="medical",
            is_trainable=False,
        )
        del teacher_base
        scorer = TransformersTrajectoryLogprobScorer(
            model=teacher_model,
            routes=SharedBackboneRoutes(
                model=teacher_model,
                medical_adapter_name="medical",
                medical_adapter_sha256=config["teacher"]["adapter_sha256"],
            ),
            model_id=model_path,
            model_revision=config["model"]["revision"],
            tokenizer_revision=config["model"]["tokenizer_revision"],
            logprob_chunk_tokens=64,
        )
        student_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        student_base.config.use_cache = False
        student_lora = config["algorithm"]["student_lora"]
        student_model = get_peft_model(
            student_base,
            LoraConfig(
                r=int(student_lora["rank"]),
                lora_alpha=int(student_lora["alpha"]),
                lora_dropout=float(student_lora["dropout"]),
                target_modules=student_lora["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        del student_base
        student_model.eval()
        live_trajectories: list[dict[str, Any]] = []
        old_flat: list[float] = []
        base_flat: list[float] = []
        for index, row in enumerate(live_prompts):
            rollout_seed = 42 + index
            torch.manual_seed(rollout_seed)
            prompt = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                generated = student_model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )
            response_ids = [
                int(value) for value in generated.sequences[0, prompt.shape[1]:].tolist()
            ]
            if not response_ids or len(generated.scores) != len(response_ids):
                raise GPUCalibrationError("live Student rollout token/old-logprob length mismatch")
            student_eos = getattr(student_model.config, "eos_token_id", None)
            if isinstance(student_eos, (list, tuple)):
                eos_ids = {int(value) for value in student_eos}
                request_eos = next(iter(eos_ids)) if len(eos_ids) == 1 else None
            else:
                eos_ids = {int(student_eos)} if student_eos is not None else set()
                request_eos = int(student_eos) if student_eos is not None else None
            eos_generated = bool(eos_ids and response_ids[-1] in eos_ids)
            eos_position = len(response_ids) - 1 if eos_generated else None
            truncated = len(response_ids) == 128 and not eos_generated
            finish_reason = "stop" if eos_generated else "length" if truncated else "rollout"
            old = [
                float(torch.log_softmax(step[0].float(), dim=-1)[token].cpu())
                for step, token in zip(generated.scores, response_ids, strict=True)
            ]
            if not all(math.isfinite(value) for value in old):
                raise GPUCalibrationError("live Student old logprob is non-finite")
            req_common = dict(
                prompt_ids=tuple(row["prompt_ids"]),
                response_ids=tuple(response_ids),
                attention_mask=(1,) * (len(row["prompt_ids"]) + len(response_ids)),
                eos_token_id=request_eos,
                finish_reason=finish_reason,
                truncated=truncated,
                source_role=row["source_role"],
            )
            base_score = scorer.score(TrajectoryScoreRequest(
                request_id=f"{row['fixture_id']}-base", route="base", **req_common
            ))
            medical_score = scorer.score(TrajectoryScoreRequest(
                request_id=f"{row['fixture_id']}-medical", route="medical", **req_common
            ))
            if (
                list(base_score.token_ids) != response_ids
                or list(medical_score.token_ids) != response_ids
                or len(old) != len(base_score.token_logprobs)
                or len(old) != len(medical_score.token_logprobs)
            ):
                raise GPUCalibrationError("live rollout and Teacher token alignment mismatch")
            old_flat.extend(old)
            base_flat.extend(base_score.token_logprobs)
            live_trajectories.append({
                "fixture_id": row["fixture_id"],
                "source_role": row["source_role"],
                "prompt_ids": row["prompt_ids"],
                "response_ids": response_ids,
                "response_mask": [1] * len(response_ids),
                "old": old,
                "eos_generated": eos_generated,
                "eos_position": eos_position,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "seed": rollout_seed,
                "base_teacher": list(base_score.token_logprobs),
                "medical_teacher": list(medical_score.token_logprobs),
            })
        live_manifest = public_live_rollout_manifest(
            live_trajectories,
            model_revision=config["model"]["revision"],
            tokenizer_revision=config["model"]["tokenizer_revision"],
            rollout_backend="transformers_generate_sampling_scores",
        )
        live_manifest.update({
            "status": "pass",
            "sampling": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 128},
            "old_logprob_source": "sampling_time_policy",
            "teacher_same_trajectory": True,
        })
        _atomic_json(output / "live_rollout_manifest.json", live_manifest)
        with raw_live.open("w", encoding="utf-8") as handle:
            for row in live_trajectories:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        rollout_null = same_model_null_metrics(
            rollout_logprobs=old_flat,
            transformers_base_logprobs=base_flat,
        )
        per_trajectory_null = {
            row["fixture_id"]: same_model_null_metrics(
                rollout_logprobs=row["old"],
                transformers_base_logprobs=row["base_teacher"],
            )
            for row in live_trajectories
        }
        length_bucket_null: dict[str, list[tuple[list[float], list[float]]]] = {}
        shift_diagnostics = []
        for row in live_trajectories:
            bucket = str(((len(row["response_ids"]) - 1) // 64 + 1) * 64)
            length_bucket_null.setdefault(bucket, []).append((row["old"], row["base_teacher"]))
            if len(row["old"]) >= 3:
                aligned = _pearson(row["old"], row["base_teacher"])
                left_shift = _pearson(row["old"][1:], row["base_teacher"][:-1])
                right_shift = _pearson(row["old"][:-1], row["base_teacher"][1:])
                shift_diagnostics.append({
                    "fixture_id": row["fixture_id"],
                    "aligned_pearson": aligned,
                    "left_shift_pearson": left_shift,
                    "right_shift_pearson": right_shift,
                    "shift_better_than_aligned": max(left_shift, right_shift) > aligned + 0.1,
                })
        length_bucket_report = {}
        for bucket, groups in length_bucket_null.items():
            left = [value for group, _ in groups for value in group]
            right = [value for _, group in groups for value in group]
            length_bucket_report[bucket] = same_model_null_metrics(
                rollout_logprobs=left,
                transformers_base_logprobs=right,
            )
        obvious_shift = bool(
            rollout_null["mean_absolute"] > 0.1
            and any(item["shift_better_than_aligned"] for item in shift_diagnostics)
        )
        if obvious_shift or not all(math.isfinite(value) for value in (*old_flat, *base_flat)):
            raise GPUCalibrationError("rollout-vs-Transformers Base shows obvious autoregressive misalignment")
        same_model_null_report = {
            "status": "pass",
            "hard_same_transformers_null": hard_null,
            "rollout_vs_transformers_base_diagnostic": {
                **rollout_null,
                "per_trajectory": per_trajectory_null,
                "length_buckets": length_bucket_report,
                "autoregressive_shift": shift_diagnostics,
                "obvious_shift_detected": obvious_shift,
                "hard_gate": False,
            },
        }
        _atomic_json(output / "same_model_null.json", same_model_null_report)
        phase_memory["live_rollout_student_gpu0"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        }
        phase_memory["live_teacher_gpu1"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(1)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(1)),
        }
        phase_timings["live_rollout_seconds"] = time.time() - phase_started
        phase_log(current_phase, "passed", response_tokens=len(old_flat))

        trainable_parameters = [
            parameter for parameter in student_model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise GPUCalibrationError("temporary Student LoRA has no trainable parameters")
        with tempfile.TemporaryDirectory(
            dir=output, prefix=".temporary_student_adapter_"
        ) as temporary_root:
            temporary_root_path = Path(temporary_root)
            old_adapter_dir = temporary_root_path / "version0"
            student_model.save_pretrained(old_adapter_dir, safe_serialization=True)
            old_adapter_sha = ordered_adapter_sha256(old_adapter_dir)

            # Phase E: exactly one Medical-Teacher PPO update.
            current_phase = "one_step_direction"
            one_step_started = time.time()
            phase_log(current_phase, "started")
            medical_before_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in student_model.named_parameters() if parameter.requires_grad
            }
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=float(student_lora["calibration_lr"]),
                weight_decay=0.0,
                foreach=False,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_values: list[float] = []
            before_logprobs = []
            advantages = []
            reverse_kls: list[float] = []
            ratio_values: list[float] = []
            clip_values: list[bool] = []
            for row in live_trajectories:
                new = _student_action_logprobs(
                    student_model, row["prompt_ids"], row["response_ids"], torch
                )
                old = torch.tensor(
                    [row["old"]], dtype=torch.float32, device="cuda:0"
                ).detach()
                teacher = torch.tensor(
                    [row["medical_teacher"]], dtype=torch.float32, device="cuda:0"
                ).detach()
                advantage = same_trajectory_advantage(old, teacher, beta=1.0)
                result = ppo_clipped_objective(
                    new,
                    old,
                    advantage,
                    torch.ones_like(new),
                    prompt_ids=(row["fixture_id"],),
                    group_ids=("g0",),
                    clip_low=0.2,
                    clip_high=0.28,
                )
                ratio = torch.exp(new.detach() - old)
                if not bool(ratio.isfinite().all()):
                    raise GPUCalibrationError("one-step PPO ratio is non-finite")
                loss_values.append(float(result.loss.detach().cpu()))
                (result.loss / len(live_trajectories)).backward()
                before_logprobs.append(new.detach())
                advantages.append(advantage.detach())
                reverse_kls.extend((old - teacher).detach().cpu().flatten().tolist())
                ratio_values.extend(ratio.cpu().flatten().tolist())
                clip_values.extend(
                    ((ratio < 0.8) | (ratio > 1.28)).cpu().flatten().tolist()
                )
            loss_value = statistics.mean(loss_values)
            gradient_tensors = [
                parameter.grad for parameter in trainable_parameters
                if parameter.grad is not None
            ]
            nonzero_gradient_tensors = sum(
                int(float(gradient.detach().abs().max()) > 0) for gradient in gradient_tensors
            )
            gradients_finite = all(bool(gradient.isfinite().all()) for gradient in gradient_tensors)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if (
                not math.isfinite(loss_value)
                or not math.isfinite(float(grad_norm))
                or not gradients_finite
                or nonzero_gradient_tensors == 0
            ):
                raise GPUCalibrationError("one-step loss/gradient is non-finite or zero")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            medical_after_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in student_model.named_parameters() if parameter.requires_grad
            }
            medical_update = lora_update_summary(
                medical_before_parameters, medical_after_parameters
            )
            if not medical_update["finite"] or medical_update["update_l2"] <= 0:
                raise GPUCalibrationError("Medical one-step LoRA update is zero or non-finite")
            student_model.eval()
            after_logprobs = []
            with torch.inference_mode():
                for row in live_trajectories:
                    after_logprobs.append(_student_action_logprobs(
                        student_model, row["prompt_ids"], row["response_ids"], torch
                    ).detach())
            advantage_flat = [
                value for tensor in advantages for value in tensor.cpu().flatten().tolist()
            ]
            before_flat = [
                value for tensor in before_logprobs for value in tensor.cpu().flatten().tolist()
            ]
            after_flat = [
                value for tensor in after_logprobs for value in tensor.cpu().flatten().tolist()
            ]
            changes = [
                after - before for before, after in zip(before_flat, after_flat, strict=True)
            ]
            direction = summarize_signed_update(
                advantage=advantage_flat,
                logprob_change=changes,
                near_zero_tolerance=1e-6,
            )
            one_step_report = {
                "status": "pass" if direction["passed"] else "fail",
                "beta": 1.0,
                "use_task_rewards": False,
                "reference_policy_kl": False,
                "reverse_kl": numeric_summary(reverse_kls),
                "advantage": {
                    **numeric_summary(advantage_flat),
                    "positive_fraction": sum(value > 1e-6 for value in advantage_flat) / len(advantage_flat),
                    "negative_fraction": sum(value < -1e-6 for value in advantage_flat) / len(advantage_flat),
                    "near_zero_fraction": sum(abs(value) <= 1e-6 for value in advantage_flat) / len(advantage_flat),
                },
                "ratio": {
                    **numeric_summary(ratio_values),
                    "clip_fraction": sum(clip_values) / len(clip_values),
                },
                "policy_loss": loss_value,
                "grad_norm_before_clip": float(grad_norm),
                "nonzero_gradient_tensors": nonzero_gradient_tensors,
                "lora_update": medical_update,
                "student_logprob_before": numeric_summary(before_flat),
                "student_logprob_after": numeric_summary(after_flat),
                "student_logprob_change": numeric_summary(changes),
                "advantage_change_pearson": _pearson(advantage_flat, changes),
                "aggregate_direction": direction,
                "elapsed_seconds": time.time() - one_step_started,
                "gpu0_memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                },
                "formal_checkpoint_saved": False,
            }
            _atomic_json(output / "one_step_direction.json", one_step_report)
            if not direction["passed"]:
                raise GPUCalibrationError(
                    "one-step aggregate positive/negative direction disagrees with advantage"
                )
            phase_timings["one_step_seconds"] = time.time() - one_step_started
            phase_memory["one_step_student_gpu0"] = one_step_report["gpu0_memory"]
            phase_log(
                current_phase,
                "passed",
                loss=one_step_report["policy_loss"],
                grad_norm=one_step_report["grad_norm_before_clip"],
                update_l2=medical_update["update_l2"],
            )

            # Phase F: restore the pre-Medical LoRA and execute an independent
            # same-implementation Base=Teacher null update, then restore the
            # Medical-updated LoRA for the sampler-refresh contract.
            current_phase = "null_update"
            null_started = time.time()
            phase_log(current_phase, "started")
            parameter_by_name = dict(student_model.named_parameters())
            with torch.no_grad():
                for name, value in medical_before_parameters.items():
                    parameter_by_name[name].copy_(
                        value.to(
                            device=parameter_by_name[name].device,
                            dtype=parameter_by_name[name].dtype,
                        )
                    )
            student_model.train()
            null_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in student_model.named_parameters() if parameter.requires_grad
            }
            null_optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=float(student_lora["calibration_lr"]),
                weight_decay=0.0,
                foreach=False,
            )
            null_optimizer.zero_grad(set_to_none=True)
            null_loss_values: list[float] = []
            for row in live_trajectories:
                new_null = _student_action_logprobs(
                    student_model, row["prompt_ids"], row["response_ids"], torch
                )
                null_old = new_null.detach()
                null_advantage = same_trajectory_advantage(
                    null_old, null_old, beta=1.0
                )
                null_result = ppo_clipped_objective(
                    new_null,
                    null_old,
                    null_advantage,
                    torch.ones_like(new_null),
                    prompt_ids=(row["fixture_id"],),
                    group_ids=("g0",),
                    clip_low=0.2,
                    clip_high=0.28,
                )
                null_loss_values.append(float(null_result.loss.detach().cpu()))
                (null_result.loss / len(live_trajectories)).backward()
            null_loss_value = statistics.mean(null_loss_values)
            null_grad_max = max(
                (
                    float(parameter.grad.detach().abs().max())
                    for parameter in trainable_parameters if parameter.grad is not None
                ),
                default=0.0,
            )
            null_optimizer.step()
            null_optimizer.zero_grad(set_to_none=True)
            null_after = {
                name: parameter.detach().cpu().clone()
                for name, parameter in student_model.named_parameters() if parameter.requires_grad
            }
            null_update = lora_update_summary(null_before, null_after)
            null_report = {
                "status": "pass",
                "independent_from_medical_step": True,
                "advantage_mean": 0.0,
                "policy_loss": null_loss_value,
                "grad_max_abs": null_grad_max,
                **null_update,
                "medical_update_l2": medical_update["update_l2"],
                "relative_to_medical_update_l2": (
                    null_update["update_l2"] / medical_update["update_l2"]
                ),
                "formal_checkpoint_saved": False,
            }
            if (
                not null_update["finite"]
                or abs(null_report["policy_loss"]) > 1e-8
                or null_grad_max > 1e-8
                or null_update["update_l2"] > 1e-12
                or null_update["update_l2"] >= medical_update["update_l2"]
            ):
                raise GPUCalibrationError("same-implementation null one-step drift exceeded tolerance")
            _atomic_json(output / "null_update.json", null_report)
            phase_timings["null_update_seconds"] = time.time() - null_started
            phase_log(current_phase, "passed", update_l2=null_update["update_l2"])
            with torch.no_grad():
                for name, value in medical_after_parameters.items():
                    parameter_by_name[name].copy_(
                        value.to(
                            device=parameter_by_name[name].device,
                            dtype=parameter_by_name[name].dtype,
                        )
                    )
            student_model.eval()

            # Sampler refresh: export only the temporary Student LoRA, bump identity once,
            # and run one label-free read-only identity check through the shared actor object.
            current_phase = "sampler_refresh"
            refresh_started = time.time()
            phase_log(current_phase, "started")
            new_adapter_dir = temporary_root_path / "version1"
            student_model.save_pretrained(new_adapter_dir, safe_serialization=True)
            new_adapter_sha = ordered_adapter_sha256(new_adapter_dir)
            exported_names = sorted(path.name for path in new_adapter_dir.iterdir() if path.is_file())
            copied_full_base = any(
                name == "model.safetensors" or name.startswith("model-")
                for name in exported_names
            )
            sampler_state = {"version": 0, "adapter_sha256": old_adapter_sha}
            sampler_state.update({"version": 1, "adapter_sha256": new_adapter_sha})
            with torch.inference_mode():
                identity_scores = _student_action_logprobs(
                    student_model,
                    live_trajectories[0]["prompt_ids"],
                    live_trajectories[0]["response_ids"],
                    torch,
                ).detach().cpu()
            identity_max_abs = float(
                (identity_scores - after_logprobs[0].detach().cpu()).abs().max()
            )
            identity_finite = bool(identity_scores.isfinite().all()) and identity_max_abs <= 1e-6
            sampler_refresh = validate_sampler_refresh(
                old_adapter_sha256=old_adapter_sha,
                new_adapter_sha256=new_adapter_sha,
                old_version=0,
                new_version=int(sampler_state["version"]),
                exported_files=[
                    name for name in exported_names
                    if name in {"adapter_config.json", "adapter_model.safetensors"}
                ],
                copied_full_base=copied_full_base,
                teacher_adapter_sha256=config["teacher"]["adapter_sha256"],
                identity_check_finite=identity_finite,
            )
            sampler_refresh.update({
                "status": "pass",
                "callable": True,
                "mode": "project_local_time_shared_actor_lora_version_refresh",
                "integration": "verl_integrated_custom_teacher_scorer",
                "formal_verl_update_weights_verified": False,
                "old_adapter_continued": False,
                "identity_check_max_abs_delta": identity_max_abs,
                "refresh_elapsed_seconds": time.time() - refresh_started,
                "temporary_adapter_removed_on_exit": True,
            })
            _atomic_json(output / "sampler_refresh.json", sampler_refresh)
            phase_timings["sampler_refresh_seconds"] = time.time() - refresh_started
            phase_log(current_phase, "passed", new_version=1)

        # All temporary Student adapter files are gone here; no formal checkpoint remains.
        if any(path.name.startswith(".temporary_student_adapter_") for path in output.iterdir()):
            raise GPUCalibrationError("temporary Student adapter cleanup failed")
        _release(torch, student_model, teacher_model)
        student_model = teacher_model = scorer = None

        _atomic_json(output / "trajectory_manifest.json", {
            "replay_public_manifest_path": config["data"]["trajectory_manifest"],
            "replay_public_manifest_sha256": config["data"]["trajectory_manifest_file_sha256"],
            "replay_private_sha256": config["data"]["private_replay_sha256"],
            "live_public_manifest_path": config["data"]["live_prompt_manifest"],
            "live_public_manifest_sha256": config["data"]["live_prompt_manifest_file_sha256"],
            "live_private_sha256": config["data"]["private_live_prompt_sha256"],
            "replay_count": 12,
            "live_count": 4,
            "contains_labels": False,
            "contains_controller": False,
            "contains_confirmation": False,
            "contains_final": False,
        })
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            metrics = {
                "transformers_repeatability_max": transformers_repeat_max,
                "route_isolation_max": route_isolation["max_same_route_abs_delta"],
                "route_difference_mean_absolute": statistics.mean(route_diffs),
                "same_model_hard_null_max": base_repeat_max,
                "rollout_base_mean_absolute": rollout_null["mean_absolute"],
                "one_step_policy_loss": one_step_report["policy_loss"],
                "one_step_grad_norm": one_step_report["grad_norm_before_clip"],
                "one_step_update_l2": medical_update["update_l2"],
                "null_update_l2": null_update["update_l2"],
                "sampler_refresh_seconds": sampler_refresh["refresh_elapsed_seconds"],
                "vllm_diagnostic_pass": equivalence["passed"],
            }
            for name, value in metrics.items():
                handle.write(json.dumps({"metric": name, "value": value}, sort_keys=True) + "\n")
        status = "opd_scorer_calibration_passed_transformers"
        elapsed = time.time() - started
        summary = {
            "status": status,
            "formal_teacher_backend": "Transformers",
            "formal_backend": "transformers_direct_trajectory_logits",
            "transformers_repeatability": "pass",
            "route_isolation": "pass",
            "token_alignment": "pass",
            "same_model_hard_null": "pass",
            "live_rollout_alignment": "pass",
            "one_step_direction": "pass",
            "null_update": "pass",
            "sampler_refresh": "pass",
            "vllm_diagnostic_pass": bool(equivalence["passed"]),
            "vllm_backend": "diagnostic_only",
            "OPD_scoring_backend_ready": True,
            "B2_authorized": False,
            "formal_opd_authorized": False,
            "phase_timings": phase_timings,
            "phase_memory": phase_memory,
            "next_step": "freeze_B2_Medical_OPD_short_run_card_and_request_separate_authorization",
        }
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "cost.json", {
            "user_instance_start_time": metadata["user_instance_start_time"],
            "codex_takeover_time": metadata["codex_takeover_time"],
            "calibration_start_time": started_iso,
            "calibration_end_time": datetime.now(timezone.utc).isoformat(),
            "phase_timings": phase_timings,
            "elapsed_seconds": elapsed,
            "price_cny_per_hour": 2.96,
            "estimated_cost_cny": elapsed / 3600 * 2.96,
            "actual_cost_cny": None,
        })
        size_budget = int(config["artifacts"]["max_per_token_artifact_mib"]) * 2**20
        raw_paths = [raw_transformers, raw_live]
        if diagnostic_output.is_file():
            raw_paths.append(diagnostic_output)
        validate_artifact_size_budget(raw_paths, maximum_bytes=size_budget)
        inventory = validate_artifact_inventory(output)
        summary["artifact_inventory_complete"] = inventory["complete"]
        _atomic_json(output / "summary.json", summary)
        metadata.update({
            "status": status,
            "failure_reason": None,
            "end": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "estimated_cost_cny": elapsed / 3600 * 2.96,
            "OPD_scoring_backend_ready": True,
            "vllm_diagnostic_pass": bool(equivalence["passed"]),
            "B2_authorized": False,
        })
        _atomic_json(output / "metadata.json", metadata)
        phase_log("calibration", "passed", terminal_status=status)
        return summary
    except Exception as error:
        failure_status = classify_calibration_failure(current_phase)
        elapsed = time.time() - started
        failure_reason = f"{type(error).__name__}: {error}"
        phase_log(current_phase, "failed", failure_status=failure_status, reason=failure_reason)
        metadata.update({
            "status": failure_status,
            "failure_reason": failure_reason,
            "end": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "estimated_cost_cny": elapsed / 3600 * 2.96,
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
        })
        _atomic_json(output / "metadata.json", metadata)
        _atomic_json(output / "summary.json", {
            "status": failure_status,
            "failure_phase": current_phase,
            "failure_reason": failure_reason,
            "formal_teacher_backend": "Transformers",
            "OPD_scoring_backend_ready": False,
            "B2_authorized": False,
            "formal_opd_authorized": False,
        })
        _atomic_json(output / "cost.json", {
            "elapsed_seconds": elapsed,
            "price_cny_per_hour": 2.96,
            "estimated_cost_cny": elapsed / 3600 * 2.96,
            "actual_cost_cny": None,
            "phase_timings": phase_timings,
        })
        placeholder_names = {
            "trajectory_manifest.json": {"status": "blocked", "contains_labels": False},
            "route_manifest.json": {"status": "blocked", "formal_backend": "Transformers"},
            "repeatability.json": {"status": "not_completed"},
            "route_isolation.json": {"status": "not_completed"},
            "vllm_diagnostic.json": {
                "status": "not_completed", "diagnostic_only": True, "formal_enabled": False
            },
            "live_rollout_manifest.json": {"status": "not_completed", "contains_labels": False},
            "same_model_null.json": {"status": "not_completed"},
            "one_step_direction.json": {"status": "not_completed"},
            "null_update.json": {"status": "not_completed"},
            "sampler_refresh.json": {"status": "not_completed"},
        }
        for name, payload in placeholder_names.items():
            if not (output / name).exists():
                _atomic_json(output / name, payload)
        if not (output / "metrics.jsonl").exists():
            (output / "metrics.jsonl").write_text(
                json.dumps({"metric": "failure", "value": failure_status}) + "\n",
                encoding="utf-8",
            )
        raise
    finally:
        try:
            _release(torch, student_model, teacher_model)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorized P4.0 GPU scorer calibration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    import yaml

    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(json.dumps(run_gpu_calibration(config, config_path=path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
