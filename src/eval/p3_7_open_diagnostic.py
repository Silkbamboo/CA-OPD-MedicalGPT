"""Bounded, label-free Medical-O1 open-prompt behavior diagnostic for P3.7."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PERSIST_ROOT = Path("artifacts")
RUN_ID = "qwen3-4b-medical-sft-v3-open-diagnostic-step450-retry1"
CANDIDATE = PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-mcq-dominant-seed42/checkpoint-450"
AUDIT = PERSIST_ROOT / "data/formal_v2/audit_holdout.jsonl"
MAX_NEW_TOKENS = 512
EXPECTED_CANDIDATE_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"


class OpenDiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize_open_outputs(rows: Sequence[Mapping[str, Any]], *, max_new_tokens: int) -> dict[str, Any]:
    count = len(rows)
    if count <= 0:
        raise OpenDiagnosticError("open diagnostic cannot be empty")
    nonempty = sum(bool(str(row.get("text") or "").strip()) for row in rows)
    thinking = sum(
        "<think>" in str(row.get("text") or "") or "</think>" in str(row.get("text") or "")
        for row in rows
    )
    letter_only = sum(bool(re.fullmatch(r"\s*[A-E]\s*", str(row.get("text") or ""))) for row in rows)
    truncated = sum(
        str(row.get("finish_reason") or "") == "length"
        and len(row.get("token_ids") or []) >= max_new_tokens
        for row in rows
    )
    echo = sum(bool(row.get("prompt_echo")) for row in rows)
    repeated = sum(bool(row.get("repetition")) for row in rows)
    summary = {
        "count": count,
        "non_empty_count": nonempty,
        "non_empty_rate": nonempty / count,
        "thinking_tag_count": thinking,
        "thinking_tag_rate": thinking / count,
        "letter_only_count": letter_only,
        "letter_only_rate": letter_only / count,
        "truncation_count": truncated,
        "truncation_rate": truncated / count,
        "prompt_echo_count": echo,
        "prompt_echo_rate": echo / count,
        "repetition_count": repeated,
        "repetition_rate": repeated / count,
        "finish_reason_distribution": dict(sorted(Counter(str(r.get("finish_reason")) for r in rows).items())),
        "generated_token_count": {
            "min": min(len(r.get("token_ids") or []) for r in rows),
            "max": max(len(r.get("token_ids") or []) for r in rows),
            "mean": sum(len(r.get("token_ids") or []) for r in rows) / count,
        },
    }
    summary["open_prompt_contract_ready"] = (
        summary["non_empty_rate"] >= 0.95
        and summary["thinking_tag_rate"] == 0
        and summary["letter_only_rate"] <= 0.05
        and summary["truncation_rate"] <= 0.10
        and max(summary["prompt_echo_rate"], summary["repetition_rate"]) <= 0.05
    )
    return summary


def run() -> dict[str, Any]:  # pragma: no cover - authorized GPU only
    if (
        os.environ.get("CA_OPD_ALLOW_P3_7_OPEN_DIAGNOSTIC_GPU") != "1"
        or os.environ.get("CA_OPD_CONFIRM_RUN") != RUN_ID
    ):
        raise OpenDiagnosticError("open diagnostic GPU execution is not explicitly authorized")
    output = PERSIST_ROOT / "outputs" / RUN_ID
    if output.exists():
        raise OpenDiagnosticError("open diagnostic output must be new")
    confirmation = json.loads(
        (PERSIST_ROOT / "outputs/qwen3-4b-medical-sft-v3-confirmation-step450/summary.json").read_text()
    )
    if confirmation.get("outcome", {}).get("status") != "confirmed":
        raise OpenDiagnosticError("open diagnostic requires a confirmed screen candidate")
    rows = []
    for line in AUDIT.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("target_role") != "audit_holdout" or "final" in str(row.get("target_role")):
            raise OpenDiagnosticError("audit holdout role drift")
        # Deliberately discard the held-out answer/reasoning before selection or
        # prompt rendering. This diagnostic measures behavior, never accuracy.
        rows.append({
            "sample_id": row["sample_id"],
            "source": row["source"],
            "target_role": row["target_role"],
            "question": row["question"],
        })
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"42:{row['sample_id']}".encode()).hexdigest(), row["sample_id"]
        ),
    )[:32]
    if len(ordered) != 32 or any(row.get("source") != "FreedomIntelligence/medical-o1-reasoning-SFT" for row in ordered):
        raise OpenDiagnosticError("Medical-O1 diagnostic selection failed")
    from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256

    if _ordered_adapter_sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA256:
        raise OpenDiagnosticError("candidate adapter identity drift")
    if confirmation.get("candidate_adapter_sha256") != EXPECTED_CANDIDATE_SHA256:
        raise OpenDiagnosticError("confirmation candidate identity drift")
    output.mkdir(parents=True, exist_ok=False)
    selected_sha = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in ordered).encode()
    ).hexdigest()

    from transformers import AutoTokenizer
    from src.eval.controller_v2_runtime import _make_vllm_generation_backend, _release_vllm_engine

    config = yaml.safe_load((REPO_ROOT / "configs/eval/qwen3_4b/controller_v2.yaml").read_text())
    config["model"]["medical_lora_path"] = str(CANDIDATE)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"], revision=config["model"]["tokenizer_revision"], local_files_only=True
    )
    prompts = []
    for row in ordered:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是一名严谨的中文医疗助手。回答需明确不确定性，并在必要时建议就医。"},
                {"role": "user", "content": str(row["question"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    if any(length + MAX_NEW_TOKENS > 1536 for length in prompt_lengths):
        raise OpenDiagnosticError("frozen diagnostic prompt exceeds vLLM context budget")
    rendered_prompts_sha = hashlib.sha256(
        "".join(hashlib.sha256(prompt.encode()).hexdigest() + "\n" for prompt in prompts).encode()
    ).hexdigest()
    _atomic_json(output / "diagnostic_manifest.json", {
        "status": "frozen_before_model_execution",
        "selection": "sha256(42:sample_id)_ascending",
        "count": 32,
        "selected_sample_ids_sha256": selected_sha,
        "rendered_prompts_sha256": rendered_prompts_sha,
        "prompt_token_length_min": min(prompt_lengths),
        "prompt_token_length_max": max(prompt_lengths),
        "audit_artifact_sha256": _sha256(AUDIT),
        "candidate_adapter_sha256": EXPECTED_CANDIDATE_SHA256,
        "label_or_answer_fields_discarded_before_selection": True,
        "label_or_answer_used": False,
        "final_authorized": False,
    })
    engine = None
    try:
        engine, _, generate0, generate1 = _make_vllm_generation_backend(config)
        summaries = {}
        for route, generate in (("B0", generate0), ("B1", generate1)):
            outputs = generate(prompts, MAX_NEW_TOKENS)
            raw_path = output / f"{route.lower()}_outputs.jsonl"
            with raw_path.open("w", encoding="utf-8") as handle:
                enriched = []
                for source, prompt, value in zip(ordered, prompts, outputs, strict=True):
                    text = str(value.get("text") or "")
                    item = {
                        "sample_id": source["sample_id"],
                        **value,
                        "prompt_echo": bool(str(source["question"])[:40] in text),
                        "repetition": bool(len(text) >= 80 and text[-40:] == text[-80:-40]),
                    }
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    enriched.append(item)
            summaries[route] = summarize_open_outputs(enriched, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        if engine is not None:
            _release_vllm_engine(engine)
        # vLLM V1 may initialize a one-rank process group even without tensor
        # parallelism; explicitly close it to satisfy the P3.7 cleanup contract.
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except ImportError:
            pass
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "complete",
        "candidate_step": 450,
        "candidate_adapter_sha256": EXPECTED_CANDIDATE_SHA256,
        "selected_sample_ids_sha256": selected_sha,
        "rendered_prompts_sha256": rendered_prompts_sha,
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": False,
        "deterministic": True,
        "routes": summaries,
        "open_prompt_contract_ready": bool(summaries["B1"]["open_prompt_contract_ready"]),
        "knowledge_metric": False,
        "labels_opened": False,
        "final_authorized": False,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "summary.json", result)
    _atomic_json(output / "artifact_manifest.json", {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "p3_7_open_prompt_diagnostic",
        "files": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"
        ],
        "final_authorized": False,
    })
    return result


def main() -> int:
    try:
        result = run()
    except Exception as exc:
        output = PERSIST_ROOT / "outputs" / RUN_ID
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "failure.json", {
            "run_id": RUN_ID,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "final_authorized": False,
        })
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
