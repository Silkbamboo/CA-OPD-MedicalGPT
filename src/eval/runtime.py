"""Production Qwen3-4B evaluator configuration and lazy vLLM execution boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

from src.data.schema import CONTROLLER_DEV, FINAL_TEST, TASK_MCQ, Sample
from src.data.chat import template_snapshot
from src.eval.mcq import DecodeSettings, MCQResult, evaluate_mcq, render_mcq_prompt


MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CONTROLLER_ROLES = ["medical_controller_dev", "general_controller_dev"]
FINAL_ROLES = ["medical_final_test", "general_final_test"]
_HEX64 = re.compile(r"[0-9a-f]{64}")
REPO_ROOT = Path(__file__).resolve().parents[2]


class EvalRuntimeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(manifest_path: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, manifest_path.parent / path, manifest_path.parent / path.name):
        if candidate.is_file():
            return candidate
    return REPO_ROOT / path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvalRuntimeError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise EvalRuntimeError(f"non-object JSONL at {path}:{line_number}")
            yield value


def _answer_index(value: Any, option_count: int) -> int:
    if type(value) is int:
        index = value
    elif isinstance(value, str) and len(value.strip()) == 1 and value.strip().upper() in "ABCDEFGH":
        index = ord(value.strip().upper()) - ord("A")
    else:
        raise EvalRuntimeError("label answer_idx is not a canonical integer/letter")
    if not 0 <= index < option_count:
        raise EvalRuntimeError("label answer_idx is outside the ordered options")
    return index


def load_frozen_mcq_samples(
    manifest_path: str | Path, roles: Sequence[str]
) -> list[Sample]:
    """Join frozen prompt/label artifacts only at the evaluator boundary.

    Neither artifact is rewritten and no question/option/answer text is logged.
    Artifact hashes, IDs, roles and counts are checked before a model can run.
    """

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("data_protocol_version") != "ca-opd-data-v2"
        or not isinstance(manifest.get("roles"), dict)
    ):
        raise EvalRuntimeError("evaluator manifest is not Data Protocol v2")
    samples: list[Sample] = []
    seen: set[str] = set()
    for role in roles:
        role_meta = manifest["roles"].get(role)
        if not isinstance(role_meta, dict):
            raise EvalRuntimeError(f"manifest lacks evaluator role: {role}")
        files = role_meta.get("files")
        if not isinstance(files, list) or len(files) != 2:
            raise EvalRuntimeError(f"{role} must bind exactly one prompt and one label artifact")
        prompt_meta = next((item for item in files if str(item.get("path", "")).endswith(".prompts.jsonl")), None)
        label_meta = next((item for item in files if str(item.get("path", "")).endswith(".labels.jsonl")), None)
        if not isinstance(prompt_meta, dict) or not isinstance(label_meta, dict):
            raise EvalRuntimeError(f"{role} prompt/label artifacts are not physically separated")
        prompt_path = _artifact_path(path, str(prompt_meta["path"]))
        label_path = _artifact_path(path, str(label_meta["path"]))
        for artifact, metadata in ((prompt_path, prompt_meta), (label_path, label_meta)):
            if not artifact.is_file() or _sha256(artifact) != str(metadata.get("sha256", "")):
                raise EvalRuntimeError(f"{role} artifact SHA mismatch")
        labels: dict[str, dict[str, Any]] = {}
        for item in _iter_jsonl(label_path):
            sample_id = str(item.get("sample_id", ""))
            if not sample_id or sample_id in labels or item.get("target_role") != role:
                raise EvalRuntimeError(f"{role} label identity/role is invalid")
            labels[sample_id] = item
        prompt_ids: set[str] = set()
        role_samples: list[Sample] = []
        expected_split = FINAL_TEST if "final" in role else CONTROLLER_DEV
        for item in _iter_jsonl(prompt_path):
            sample_id = str(item.get("sample_id", ""))
            if not sample_id or sample_id in prompt_ids or sample_id in seen:
                raise EvalRuntimeError(f"{role} prompt sample_id is missing or duplicated")
            prompt_ids.add(sample_id)
            if item.get("target_role") != role or sample_id not in labels:
                raise EvalRuntimeError(f"{role} prompt/label IDs or roles differ")
            options = item.get("options")
            if not isinstance(options, list) or not 4 <= len(options) <= 5:
                raise EvalRuntimeError(f"{role} does not have a supported 4/5-option schema")
            label = labels[sample_id]
            index = _answer_index(label.get("answer_idx"), len(options))
            role_samples.append(
                Sample(
                    source=str(item.get("source", "unknown")),
                    split=expected_split,
                    domain=str(item.get("domain", "")),
                    task=TASK_MCQ,
                    question=str(item.get("question", "")),
                    options=[str(option) for option in options],
                    answer=str(label.get("answer", options[index])),
                    answer_index=index,
                    sample_id=sample_id,
                    meta={
                        "target_role": role,
                        "upstream_split": item.get("upstream_split"),
                        "subject": item.get("subject"),
                    },
                )
            )
        if prompt_ids != set(labels):
            raise EvalRuntimeError(f"{role} prompt/label IDs differ")
        if len(role_samples) != int(role_meta.get("actual_count", len(role_samples))):
            raise EvalRuntimeError(f"{role} manifest count differs from evaluator artifacts")
        samples.extend(role_samples)
        seen.update(prompt_ids)
    return samples


def load_eval_runtime_config(
    path: str | Path,
    *,
    allow_final_eval: bool = False,
    for_execution: bool = False,
) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvalRuntimeError("evaluator config root must be a mapping")
    required = {
        "run_id", "capability", "model", "data", "decode", "output_root",
        "allow_final_eval", "primary_final_frozen",
    }
    if set(payload) != required:
        raise EvalRuntimeError(f"evaluator config fields differ: missing={sorted(required-set(payload))}")
    model = payload["model"]
    data = payload["data"]
    decode = payload["decode"]
    if model.get("revision") != MODEL_REVISION:
        raise EvalRuntimeError("evaluator model revision is not the approved Qwen3-4B commit")
    if set(decode) != {
        "temperature", "do_sample", "seed", "max_prompt_tokens",
        "max_new_tokens", "enable_thinking",
    }:
        raise EvalRuntimeError("decode contract is incomplete")
    if decode != {
        "temperature": 0.0,
        "do_sample": False,
        "seed": 42,
        "max_prompt_tokens": 512,
        "max_new_tokens": 32,
        "enable_thinking": False,
    }:
        raise EvalRuntimeError("evaluation decoding must be deterministic and non-thinking")
    roles = data.get("roles")
    capability = payload["capability"]
    if capability == "controller_eval":
        if roles != CONTROLLER_ROLES or any("final" in str(role) for role in roles or []):
            raise EvalRuntimeError("controller evaluator cannot bind a final role")
        if payload["allow_final_eval"] is not False:
            raise EvalRuntimeError("controller evaluator cannot authorize final")
    elif capability == "final":
        if roles != FINAL_ROLES or payload["primary_final_frozen"] is not True:
            raise EvalRuntimeError("final evaluator requires the frozen final roles")
        if for_execution and not allow_final_eval:
            raise EvalRuntimeError("final execution requires explicit --allow-final-eval")
    else:
        raise EvalRuntimeError("unsupported evaluator capability")
    manifest = Path(str(data.get("manifest_path", "")))
    expected = str(data.get("manifest_sha256", ""))
    if _HEX64.fullmatch(expected) is None:
        raise EvalRuntimeError("evaluator manifest SHA is invalid")
    if manifest.is_file() and _sha256(manifest) != expected:
        raise EvalRuntimeError("evaluator manifest SHA mismatch")
    return payload


def select_smoke_samples(samples: Sequence[Sample], *, per_role: int) -> list[Sample]:
    """Select a deterministic, role-balanced evaluator smoke without touching final."""

    if per_role < 1:
        raise EvalRuntimeError("smoke per-role count must be positive")
    selected: list[Sample] = []
    for role in CONTROLLER_ROLES:
        members = sorted(
            (sample for sample in samples if sample.meta.get("target_role") == role),
            key=lambda sample: sample.sample_id,
        )
        if len(members) < per_role:
            raise EvalRuntimeError(f"{role} has fewer than {per_role} smoke samples")
        selected.extend(members[:per_role])
    return selected


def controller_metrics(result: MCQResult, samples: Sequence[Sample]) -> dict[str, Any]:
    """Compute the frozen medical/general and per-subject controller metrics."""

    by_id = {sample.sample_id: sample for sample in samples}
    subject_total: dict[str, int] = {}
    subject_correct: dict[str, int] = {}
    for item in result.samples:
        sample = by_id[item.sample_id]
        if sample.domain != "general":
            continue
        subject = str(sample.meta.get("subject") or "unknown")
        subject_total[subject] = subject_total.get(subject, 0) + 1
        subject_correct[subject] = subject_correct.get(subject, 0) + int(item.correct)
    subject_accuracy = {
        subject: subject_correct[subject] / count
        for subject, count in sorted(subject_total.items())
    }
    general_macro = (
        sum(subject_accuracy.values()) / len(subject_accuracy) if subject_accuracy else None
    )
    return {
        "medical_accuracy": result.accuracy_by_domain.get("medical"),
        "general_micro_accuracy": result.accuracy_by_domain.get("general"),
        "general_macro_accuracy": general_macro,
        "general_subject_accuracy": subject_accuracy,
        "general_subject_counts": dict(sorted(subject_total.items())),
    }


def make_vllm_generate_fn(
    *,
    model_path: str,
    adapter_path: str | None = None,
    max_model_len: int,
    seed: int,
) -> Callable[[list[str], int], list[str]]:  # pragma: no cover - GPU only
    """Construct the real greedy vLLM generator lazily; never called by CPU tests."""

    from vllm import LLM, SamplingParams

    if max_model_len < 1 or seed < 0:
        raise EvalRuntimeError("vLLM context length and seed must be explicit")
    kwargs: dict[str, Any] = {
        "model": model_path,
        "enable_lora": bool(adapter_path),
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "seed": seed,
    }
    engine = LLM(**kwargs)
    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("evaluation_adapter", 1, adapter_path)

    def generate(prompts: list[str], max_new_tokens: int) -> list[str]:
        sampling = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, seed=seed
        )
        outputs = engine.generate(
            prompts=prompts,
            sampling_params=sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )
        return [str(item.outputs[0].text) for item in outputs]

    return generate


def run_evaluation(
    config_path: str | Path,
    *,
    allow_final_eval: bool = False,
    smoke_per_role: int | None = None,
) -> dict[str, Any]:  # pragma: no cover - GPU only
    """Run the frozen evaluator and atomically persist aggregate/per-sample records."""

    started_at = datetime.now(timezone.utc)
    config = load_eval_runtime_config(
        config_path, allow_final_eval=allow_final_eval, for_execution=True
    )
    manifest_path = Path(str(config["data"]["manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    samples = load_frozen_mcq_samples(manifest_path, config["data"]["roles"])
    if smoke_per_role is not None:
        if config["capability"] != "controller_eval":
            raise EvalRuntimeError("smoke selection is only available for controller_eval")
        samples = select_smoke_samples(samples, per_role=smoke_per_role)
    generator = make_vllm_generate_fn(
        model_path=str(config["model"]["path"]),
        adapter_path=config["model"].get("adapter_path"),
        max_model_len=(
            int(config["decode"]["max_prompt_tokens"])
            + int(config["decode"]["max_new_tokens"])
        ),
        seed=int(config["decode"]["seed"]),
    )
    repeat_prompt = render_mcq_prompt(samples[0])
    repeated = generator(
        [repeat_prompt, repeat_prompt], int(config["decode"]["max_new_tokens"])
    )
    if len(repeated) != 2 or repeated[0] != repeated[1]:
        raise EvalRuntimeError("greedy evaluator repeat-determinism check failed")
    decode = DecodeSettings(
        temperature=0.0,
        max_new_tokens=int(config["decode"]["max_new_tokens"]),
        shuffle_options=False,
    )
    result = evaluate_mcq(
        samples, generator, split=str(config["capability"]), decode=decode, batch_size=8
    )
    run_id = str(config["run_id"]) + ("-smoke" if smoke_per_role is not None else "")
    run_dir = Path(str(config["output_root"])) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise EvalRuntimeError(f"evaluator output directory is not new/empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    aggregate = result.as_dict(include_samples=False)
    # Persist the production config contract (including do_sample/seed), not
    # only the narrower scorer dataclass view.
    aggregate["decode"] = dict(config["decode"])
    aggregate["run_id"] = config["run_id"]
    aggregate["capability"] = config["capability"]
    aggregate["data_manifest_sha256"] = config["data"]["manifest_sha256"]
    aggregate["model_revision"] = config["model"]["revision"]
    aggregate["adapter_path"] = config["model"].get("adapter_path")
    aggregate["invalid_count"] = sum(not item.parsed for item in result.samples)
    aggregate["invalid_rate"] = aggregate["invalid_count"] / result.num_samples
    aggregate.update(controller_metrics(result, samples))
    aggregate["smoke_per_role"] = smoke_per_role
    aggregate["repeat_determinism_passed"] = True
    aggregate["final_authorized_for_this_invocation"] = bool(allow_final_eval)
    template_payload = json.dumps(template_snapshot(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    aggregate["prompt_template_sha256"] = hashlib.sha256(template_payload).hexdigest()
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    aggregate["evaluator_git_sha"] = git_sha
    aggregate["run_id"] = run_id
    ended_at = datetime.now(timezone.utc)
    metadata = {
        "run_id": run_id,
        "stage": "controller_eval",
        "status": "completed",
        "git_sha": git_sha,
        "model_path": config["model"]["path"],
        "model_revision": config["model"]["revision"],
        "adapter_path": config["model"].get("adapter_path"),
        "data_manifest_sha256": config["data"]["manifest_sha256"],
        "prompt_template_sha256": aggregate["prompt_template_sha256"],
        "decode": dict(config["decode"]),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "actual_cost_cny": None,
        "final_authorized": False,
    }
    metrics_record = {
        "step": 0,
        "medical_accuracy": aggregate["medical_accuracy"],
        "general_macro_accuracy": aggregate["general_macro_accuracy"],
        "general_micro_accuracy": aggregate["general_micro_accuracy"],
        "invalid_count": aggregate["invalid_count"],
        "invalid_rate": aggregate["invalid_rate"],
        "seconds": aggregate["seconds"],
    }
    def atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    atomic_json(run_dir / "aggregate.json", aggregate)
    atomic_json(run_dir / "summary.json", aggregate)
    atomic_json(run_dir / "metadata.json", metadata)
    metrics_tmp = run_dir / "metrics.jsonl.tmp"
    metrics_tmp.write_text(json.dumps(metrics_record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(metrics_tmp, run_dir / "metrics.jsonl")
    per_sample_tmp = run_dir / "per_sample.jsonl.tmp"
    with per_sample_tmp.open("w", encoding="utf-8") as handle:
        for item in result.samples:
            # Model response and IDs are allowed in ignored run outputs; raw prompts are never copied.
            handle.write(json.dumps(item.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(per_sample_tmp, run_dir / "per_sample.jsonl")
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI/GPU only
    import argparse

    parser = argparse.ArgumentParser(description="Frozen deterministic Qwen3-4B MCQ evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-final-eval", action="store_true")
    parser.add_argument("--smoke-per-role", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_eval_runtime_config(
        args.config, allow_final_eval=args.allow_final_eval, for_execution=args.execute
    )
    if not args.execute:
        print(json.dumps({"status": "config_valid", "run_id": config["run_id"]}, sort_keys=True))
        return 0
    result = run_evaluation(
        args.config,
        allow_final_eval=args.allow_final_eval,
        smoke_per_role=args.smoke_per_role,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
