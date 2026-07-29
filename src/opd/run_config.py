"""One run file -> (RunPlan, VerlOPDConfig).

A single YAML per formal run describes budget, criteria and the veRL wiring. Both
the declared plan and the executed command are generated from that one file, so
"the plan said 200 steps but the command ran 2000" cannot happen.

Used by ``scripts/preflight.py`` (the gate that must pass before any paid run).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.opd.verl_config import TeacherSpec, VerlOPDConfig
from src.utils.config import ConfigError, FieldSpec, load_yaml, validate
from src.utils.run_plan import RunPlan, ThroughputModel

RUN_SCHEMA: Dict[str, object] = {
    "run": {
        "run_id": FieldSpec((str,), doc="unique run id"),
        "purpose": FieldSpec((str,)),
        "baseline_id": FieldSpec((str,), choices=["B0", "B1", "B2", "B3", "B4", "B5", "O1"]),
        "model": FieldSpec((str,)),
        "seed": FieldSpec((int,), bounds=(0, None)),
        "output_root": FieldSpec((str,)),
    },
    "data": {
        "manifest": FieldSpec((str,), required=False, default=None, doc="data_manifest.json path"),
        "controller_dev_samples": FieldSpec((int,), bounds=(1, None)),
    },
    "budget": {
        "steps": FieldSpec((int,), bounds=(1, None)),
        "prompt_batch_size": FieldSpec((int,), bounds=(1, None)),
        "group_size": FieldSpec((int,), bounds=(1, None)),
        "max_prompt_tokens": FieldSpec((int,), bounds=(1, None)),
        "max_response_tokens": FieldSpec((int,), bounds=(1, None)),
        "checkpoint_every_steps": FieldSpec((int,), bounds=(1, None)),
        "controller_dev_every_steps": FieldSpec((int,), bounds=(1, None)),
        "num_gpus": FieldSpec((int,), bounds=(1, None)),
        "price_per_gpu_hour_rmb": FieldSpec((float,), bounds=(0.0, None)),
        "cost_cap_rmb": FieldSpec((float,), bounds=(0.0001, None)),
    },
    "throughput": {
        "rollout_tokens_per_second": FieldSpec((float,), bounds=(0.0001, None)),
        "teacher_prefill_tokens_per_second": FieldSpec((float,), bounds=(0.0001, None)),
        "optimizer_step_seconds": FieldSpec((float,), bounds=(0.0001, None)),
        "measured": FieldSpec((bool,)),
        "source": FieldSpec((str,)),
    },
    "criteria": {
        "success": FieldSpec((list,)),
        "early_stop": FieldSpec((list,)),
        "abort": FieldSpec((list,)),
    },
    "verl": {
        "student_model_path": FieldSpec((str,)),
        "teacher_key": FieldSpec((str,)),
        "teacher_gpus_per_node": FieldSpec((int,), bounds=(1, None)),
        "teacher_nnodes": FieldSpec((int,), bounds=(1, None)),
        "teachers": FieldSpec((list,), doc="list of teacher entries"),
        "lora_rank": FieldSpec((int,), bounds=(1, None)),
        "lora_alpha": FieldSpec((int,), bounds=(1, None)),
        "lora_target_modules": FieldSpec((str,)),
        "lora_adapter_path": FieldSpec((str,), required=False, default=None),
        "layered_summon": FieldSpec((bool,)),
        "loss_mode": FieldSpec((str,)),
        "use_policy_gradient": FieldSpec((bool,)),
        "use_task_rewards": FieldSpec((bool,)),
        "clip_ratio_low": FieldSpec((float,), bounds=(0.0001, None)),
        "clip_ratio_high": FieldSpec((float,), bounds=(0.0001, None)),
        "topk": FieldSpec((int,), required=False, default=None, bounds=(1, None)),
        "loss_max_clamp": FieldSpec((float,), required=False, default=None),
        "rollout_temperature": FieldSpec((float,), bounds=(0.0001, None)),
        "lr": FieldSpec((float,), bounds=(0.0, None)),
    },
    "router": {
        "kind": FieldSpec((str,), choices=["constraint_aware", "fixed_ratio", "single_teacher"]),
        "config_path": FieldSpec((str,), required=False, default=None),
        "fixed_p_medical": FieldSpec((float,), required=False, default=None, bounds=(0.0, 1.0)),
        "single_teacher_id": FieldSpec((str,), required=False, default=None),
    },
}

_TEACHER_KEYS = {
    "name",
    "routing_key",
    "model_path",
    "num_replicas",
    "tensor_model_parallel_size",
    "data_parallel_size",
    "pipeline_model_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "lora_adapter_path",
}


@dataclass
class LoadedRun:
    config_path: str
    raw: Dict[str, Any]
    plan: RunPlan
    verl: VerlOPDConfig

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "plan": self.plan.as_dict(),
            "verl_overrides": self.verl.to_overrides(),
            "router": self.raw["router"],
        }


def _teacher_specs(entries: List[Mapping[str, Any]]) -> List[TeacherSpec]:
    specs: List[TeacherSpec] = []
    for i, entry in enumerate(entries):
        unknown = set(entry) - _TEACHER_KEYS
        if unknown:
            raise ConfigError(f"verl.teachers[{i}] has unknown keys: {sorted(unknown)}")
        missing = {"name", "routing_key", "model_path"} - set(entry)
        if missing:
            raise ConfigError(f"verl.teachers[{i}] missing {sorted(missing)}")
        specs.append(TeacherSpec(**dict(entry)))
    return specs


def load_run(config_path: str | Path) -> LoadedRun:
    raw = validate(load_yaml(config_path), RUN_SCHEMA)
    run, budget, thr, crit, v, data = (
        raw["run"], raw["budget"], raw["throughput"], raw["criteria"], raw["verl"], raw["data"],
    )

    plan = RunPlan(
        run_id=str(run["run_id"]),
        purpose=str(run["purpose"]),
        baseline_id=str(run["baseline_id"]),
        model=str(run["model"]),
        seed=int(run["seed"]),
        steps=int(budget["steps"]),
        prompt_batch_size=int(budget["prompt_batch_size"]),
        group_size=int(budget["group_size"]),
        max_prompt_tokens=int(budget["max_prompt_tokens"]),
        max_response_tokens=int(budget["max_response_tokens"]),
        checkpoint_every_steps=int(budget["checkpoint_every_steps"]),
        controller_dev_every_steps=int(budget["controller_dev_every_steps"]),
        controller_dev_samples=int(data["controller_dev_samples"]),
        num_gpus=int(budget["num_gpus"]),
        price_per_gpu_hour_rmb=float(budget["price_per_gpu_hour_rmb"]),
        cost_cap_rmb=float(budget["cost_cap_rmb"]),
        throughput=ThroughputModel(
            rollout_tokens_per_second=float(thr["rollout_tokens_per_second"]),
            teacher_prefill_tokens_per_second=float(thr["teacher_prefill_tokens_per_second"]),
            optimizer_step_seconds=float(thr["optimizer_step_seconds"]),
            measured=bool(thr["measured"]),
            source=str(thr["source"]),
        ),
        data_manifest_path=data["manifest"],
        success_criteria=list(crit["success"]),
        early_stop_criteria=list(crit["early_stop"]),
        abort_criteria=list(crit["abort"]),
    )

    verl = VerlOPDConfig(
        student_model_path=str(v["student_model_path"]),
        teachers=_teacher_specs(list(v["teachers"])),
        teacher_key=str(v["teacher_key"]),
        teacher_gpus_per_node=int(v["teacher_gpus_per_node"]),
        teacher_nnodes=int(v["teacher_nnodes"]),
        max_prompt_tokens=int(budget["max_prompt_tokens"]),
        max_response_tokens=int(budget["max_response_tokens"]),
        prompt_batch_size=int(budget["prompt_batch_size"]),
        group_size=int(budget["group_size"]),
        rollout_temperature=float(v["rollout_temperature"]),
        lora_rank=int(v["lora_rank"]),
        lora_alpha=int(v["lora_alpha"]),
        lora_target_modules=str(v["lora_target_modules"]),
        lora_adapter_path=v["lora_adapter_path"],
        layered_summon=bool(v["layered_summon"]),
        loss_mode=str(v["loss_mode"]),
        use_policy_gradient=bool(v["use_policy_gradient"]),
        use_task_rewards=bool(v["use_task_rewards"]),
        clip_ratio_low=float(v["clip_ratio_low"]),
        clip_ratio_high=float(v["clip_ratio_high"]),
        topk=v["topk"],
        loss_max_clamp=v["loss_max_clamp"],
        lr=float(v["lr"]),
        total_steps=int(budget["steps"]),
        save_freq=int(budget["checkpoint_every_steps"]),
    )

    # cross-file consistency: a constraint-aware run must point at a router config
    router = raw["router"]
    if router["kind"] == "constraint_aware" and not router["config_path"]:
        raise ConfigError("router.kind=constraint_aware requires router.config_path")
    if router["kind"] == "fixed_ratio" and router["fixed_p_medical"] is None:
        raise ConfigError("router.kind=fixed_ratio requires router.fixed_p_medical")
    if router["kind"] == "single_teacher" and not router["single_teacher_id"]:
        raise ConfigError("router.kind=single_teacher requires router.single_teacher_id")
    if router["kind"] != "single_teacher" and len(verl.teachers) < 2:
        raise ConfigError(
            f"router.kind={router['kind']} needs two teachers, but verl.teachers has "
            f"{len(verl.teachers)}"
        )

    verl.validate()
    return LoadedRun(config_path=str(config_path), raw=raw, plan=plan, verl=verl)
