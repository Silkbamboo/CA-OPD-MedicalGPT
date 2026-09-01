"""Target-environment smoke checks for the Qwen3 production stack.

This module is intentionally not imported by normal CPU tests: target packages
are loaded lazily inside checks. The pure requirement/report helpers are tested
on CPU; the actual imports run only in ``scripts/target_env_smoke.py`` on the
isolated OPD environment.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.data.chat import DEFAULT_TEMPLATE, build_masked_example
from src.sft.train import (
    SFT_SCHEMA,
    lora_config_kwargs,
    sft_config_kwargs,
    sft_trainer_kwargs,
)
from src.utils.config import load_config
from src.utils.io import write_json

_EXACT_PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)$")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_exact_pins(path: str | Path) -> Dict[str, str]:
    """Parse a direct-pin file and reject ranges/unpinned requirements."""
    pins: Dict[str, str] = {}
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_PIN.fullmatch(line)
        if not match:
            raise ValueError(f"{path}:{lineno}: expected exact package==version pin, got {line!r}")
        name, expected = canonical_name(match.group(1)), match.group(2)
        if name in pins:
            raise ValueError(f"{path}:{lineno}: duplicate pin for {name}")
        pins[name] = expected
    if not pins:
        raise ValueError(f"{path}: no package pins found")
    return pins


@dataclass
class SmokeCheck:
    name: str
    status: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


@dataclass
class SmokeReport:
    model: str
    requirements: str
    checks: List[SmokeCheck] = field(default_factory=list)
    revision: Optional[str] = None

    @property
    def passed(self) -> bool:
        return not any(check.failed for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": "PASS" if self.passed else "FAIL",
            "model": self.model,
            "revision": self.revision,
            "requirements": self.requirements,
            "checks": [asdict(c) for c in self.checks],
        }


def check_package_pins(
    requirements: str | Path,
    installed: Optional[Mapping[str, str]] = None,
) -> SmokeCheck:
    pins = read_exact_pins(requirements)
    actual: Dict[str, str] = {}
    missing: List[str] = []
    mismatched: Dict[str, Dict[str, str]] = {}
    for name, expected in pins.items():
        if installed is not None:
            value = installed.get(name)
        else:
            try:
                value = dist_version(name)
            except PackageNotFoundError:
                value = None
        if value is None:
            missing.append(name)
            continue
        actual[name] = value
        if value != expected:
            mismatched[name] = {"expected": expected, "actual": value}
    if missing or mismatched:
        return SmokeCheck(
            "exact package pins",
            "FAIL",
            f"missing={missing}; mismatched={mismatched}",
            {"expected": pins, "actual": actual},
        )
    return SmokeCheck("exact package pins", "PASS", f"{len(pins)} direct pins match", {"versions": actual})


def check_pip_consistency() -> SmokeCheck:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120, check=False
    )
    detail = (proc.stdout or proc.stderr).strip()
    return SmokeCheck("pip dependency consistency", "PASS" if proc.returncode == 0 else "FAIL", detail)


def check_cuda(require_gpu: bool, require_two_gpus: bool) -> SmokeCheck:
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        evidence = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_count": count,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "bf16": bool(torch.cuda.is_bf16_supported()),
                    "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                }
                for i in range(count)
            ],
        }
        required = 2 if require_two_gpus else (1 if require_gpu else 0)
        if count < required:
            return SmokeCheck("CUDA/BF16", "FAIL", f"found {count} GPU(s), require {required}", evidence)
        if count and (torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 8)):
            return SmokeCheck("CUDA/BF16", "FAIL", f"torch CUDA runtime {torch.version.cuda}; require >=12.8", evidence)
        if count and not torch.cuda.is_bf16_supported():
            return SmokeCheck("CUDA/BF16", "FAIL", "BF16 is not supported", evidence)
        status = "PASS" if count else "SKIP"
        return SmokeCheck("CUDA/BF16", status, f"{count} compatible GPU(s)" if count else "GPU not required", evidence)
    except Exception as exc:  # noqa: BLE001
        return SmokeCheck("CUDA/BF16", "FAIL", f"{type(exc).__name__}: {exc}")


def check_qwen3_and_mask(model: str, revision: Optional[str], local_files_only: bool) -> tuple[SmokeCheck, Any, Any]:
    try:
        if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Qwen smoke requires an immutable 40-hex --revision")
        from transformers import AutoConfig, AutoTokenizer

        kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
        if revision:
            kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(model, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        if getattr(config, "model_type", None) != "qwen3":
            raise AssertionError(f"model_type={getattr(config, 'model_type', None)!r}, expected 'qwen3'")

        messages = [
            {"role": "system", "content": "你是一个严谨、诚实的中文医疗助手。"},
            {"role": "user", "content": "患者主诉咳嗽三周，应如何处理？"},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        project_prompt = DEFAULT_TEMPLATE.render_prompt(messages[1]["content"], messages[0]["content"])
        if rendered != project_prompt:
            raise AssertionError("project ChatML prompt differs from Qwen3 tokenizer.apply_chat_template")

        def encode(text: str) -> List[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        example = build_masked_example(
            encode, messages[1]["content"], "建议先评估红旗征象并尽快就医。", system_prompt=messages[0]["content"]
        )
        full_ids = encode(example.prompt_text + example.completion_text)
        if full_ids != example.input_ids:
            raise AssertionError("segment-wise tokenization differs from full prompt+completion tokenization")
        if any(example.loss_mask[: example.prompt_length]) or not all(example.loss_mask[example.prompt_length :]):
            raise AssertionError("assistant-only mask boundary is incorrect")
        evidence = {
            "model_type": config.model_type,
            "architectures": getattr(config, "architectures", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "tokenizer_class": type(tokenizer).__name__,
            "prompt_tokens": example.prompt_length,
            "completion_tokens": example.completion_length,
            "chat_template_match": True,
            "segment_boundary_match": True,
        }
        return SmokeCheck("Qwen3 config/tokenizer/mask", "PASS", "real tokenizer contract verified", evidence), config, tokenizer
    except Exception as exc:  # noqa: BLE001
        return SmokeCheck("Qwen3 config/tokenizer/mask", "FAIL", f"{type(exc).__name__}: {exc}"), None, None


def check_trl_peft(sft_config_path: str | Path, model_config: Any, tokenizer: Any) -> SmokeCheck:
    if model_config is None or tokenizer is None:
        return SmokeCheck("TRL/PEFT SFT API", "FAIL", "Qwen3 config/tokenizer prerequisite failed")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM
        from trl import SFTConfig, SFTTrainer

        project_config = load_config(sft_config_path, SFT_SCHEMA)
        trl_args = SFTConfig(**sft_config_kwargs(project_config, "/tmp/ca-opd-sft-smoke"))
        if not hasattr(trl_args, "max_length") or trl_args.max_length != project_config["model"]["max_seq_length"]:
            raise AssertionError("TRL SFTConfig.max_length contract mismatch")
        peft_args = LoraConfig(**lora_config_kwargs(project_config))
        if peft_args.target_modules != "all-linear":
            raise AssertionError(f"target_modules={peft_args.target_modules!r}")

        # Instantiate a tiny model of the *same Qwen3 architecture* so all-linear
        # module discovery and SFTTrainer's pretokenized-label path are exercised
        # without allocating the 1.7B weights.
        tiny_dict = model_config.to_dict()
        tiny_dict.update(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        tiny_config = type(model_config).from_dict(tiny_dict)
        tiny_model = AutoModelForCausalLM.from_config(tiny_config)
        data = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3, 4]],
                "attention_mask": [[1, 1, 1, 1]],
                "labels": [[-100, -100, 3, 4]],
            }
        )
        trainer = SFTTrainer(
            **sft_trainer_kwargs(
                model=tiny_model,
                args=trl_args,
                train_dataset=data,
                peft_config=peft_args,
                tokenizer=tokenizer,
            )
        )
        batch = trainer.data_collator([data[0]])
        if batch["labels"].tolist()[0] != [-100, -100, 3, 4]:
            raise AssertionError(f"SFT collator changed assistant-only labels: {batch['labels'].tolist()}")
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        if trainable <= 0:
            raise AssertionError("PEFT all-linear produced no trainable parameters")
        del trainer, tiny_model, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return SmokeCheck(
            "TRL/PEFT SFT API",
            "PASS",
            "SFTConfig + pretokenized labels + Qwen3 all-linear LoRA verified",
            {"max_length": trl_args.max_length, "trainable_tiny_parameters": trainable},
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheck("TRL/PEFT SFT API", "FAIL", f"{type(exc).__name__}: {exc}")


def check_vllm_api() -> SmokeCheck:
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        sampling = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0)
        lora_signature = inspect.signature(LoRARequest)
        names = set(lora_signature.parameters)
        required = {"lora_name", "lora_int_id", "lora_path"}
        if not required.issubset(names):
            raise AssertionError(f"LoRARequest signature lacks {sorted(required - names)}: {lora_signature}")
        generate_signature = inspect.signature(LLM.generate)
        generate_names = set(generate_signature.parameters)
        generate_required = {"prompts", "sampling_params", "use_tqdm", "lora_request"}
        if not generate_required.issubset(generate_names):
            raise AssertionError(
                f"LLM.generate signature lacks {sorted(generate_required - generate_names)}: {generate_signature}"
            )
        return SmokeCheck(
            "vLLM prompt_logprobs + LoRA API",
            "PASS",
            "scoring request primitives available",
            {
                "version": vllm.__version__,
                "sampling": str(sampling),
                "lora_signature": str(lora_signature),
                "generate_signature": str(generate_signature),
                "input_shape": "TokensPrompt(prompt_token_ids=...)",
            },
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheck("vLLM prompt_logprobs + LoRA API", "FAIL", f"{type(exc).__name__}: {exc}")


def check_verl_ray_api() -> SmokeCheck:
    try:
        import ray
        from verl.trainer.distillation.losses import distillation_ppo_loss
        from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig

        if not callable(distillation_ppo_loss):
            raise AssertionError("distillation_ppo_loss is not callable")
        return SmokeCheck(
            "veRL OPD + Ray API",
            "PASS",
            "distillation config/loss and Ray import verified",
            {
                "ray": ray.__version__,
                "distillation_config": f"{DistillationConfig.__module__}.{DistillationConfig.__name__}",
                "loss_config": f"{DistillationLossConfig.__module__}.{DistillationLossConfig.__name__}",
            },
        )
    except Exception as exc:  # noqa: BLE001
        return SmokeCheck("veRL OPD + Ray API", "FAIL", f"{type(exc).__name__}: {exc}")


def run_target_smoke(
    model: str = "Qwen/Qwen3-1.7B",
    requirements: str | Path = "env/requirements-opd.txt",
    sft_config: str | Path = "configs/sft/qwen3_1_7b_medical.yaml",
    revision: Optional[str] = None,
    require_gpu: bool = False,
    require_two_gpus: bool = False,
    local_files_only: bool = False,
) -> SmokeReport:
    report = SmokeReport(model=model, requirements=str(requirements), revision=revision)
    report.checks.append(check_package_pins(requirements))
    report.checks.append(check_pip_consistency())
    report.checks.append(check_cuda(require_gpu or require_two_gpus, require_two_gpus))
    model_check, model_config, tokenizer = check_qwen3_and_mask(model, revision, local_files_only)
    report.checks.append(model_check)
    report.checks.append(check_trl_peft(sft_config, model_config, tokenizer))
    report.checks.append(check_vllm_api())
    report.checks.append(check_verl_ray_api())
    return report


def write_smoke_report(path: str | Path, report: SmokeReport) -> Path:
    return write_json(path, report.as_dict())
