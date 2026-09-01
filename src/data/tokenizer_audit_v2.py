"""Real, local-only Qwen3 tokenizer auditing without importing model weights."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.data.chat import DEFAULT_SYSTEM_PROMPT, format_mcq_question


_HEX40 = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN = (
    "*.safetensors",
    "pytorch_model*",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
)
_QWEN_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


class TokenizerArtifactError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound_tokenizer(
    directory: str | Path,
    *,
    expected_id: str,
    expected_revision: str,
):
    """Verify a tokenizer-only artifact manifest, then load strictly locally."""

    root = Path(directory)
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise TokenizerArtifactError("tokenizer artifact_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tokenizer_id") != expected_id:
        raise TokenizerArtifactError("tokenizer ID mismatch")
    if _HEX40.fullmatch(expected_revision) is None or manifest.get(
        "tokenizer_revision"
    ) != expected_revision:
        raise TokenizerArtifactError("tokenizer revision mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise TokenizerArtifactError("tokenizer manifest files are missing")
    for name, metadata in files.items():
        basename = Path(name).name.casefold()
        if any(fnmatch.fnmatch(basename, pattern) for pattern in _FORBIDDEN):
            raise TokenizerArtifactError(f"model weight appears in tokenizer manifest: {name}")
        path = root / name
        if not path.is_file():
            raise TokenizerArtifactError(f"tokenizer artifact is missing: {name}")
        if _sha256(path) != metadata.get("sha256"):
            raise TokenizerArtifactError(f"tokenizer artifact SHA-256 mismatch: {name}")
        if path.stat().st_size != metadata.get("bytes"):
            raise TokenizerArtifactError(f"tokenizer artifact size mismatch: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise TokenizerArtifactError("tokenizer does not provide a chat template")
    binding = {
        "tokenizer_id": expected_id,
        "tokenizer_revision": expected_revision,
        "artifact_manifest_sha256": _sha256(manifest_path),
        "files": files,
    }
    return tokenizer, binding


def _render(tokenizer, messages: Sequence[dict[str, str]], *, generation: bool) -> str:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TokenizerArtifactError("chat template did not render text")
    rendered = rendered.replace(_QWEN_EMPTY_THINK_BLOCK, "")
    folded = rendered.casefold()
    if "<think>" in folded or "</think>" in folded:
        raise TokenizerArtifactError("non-thinking chat template emitted think tags")
    return rendered


def nonthinking_template_evidence(tokenizer) -> dict[str, Any]:
    """Probe Qwen3 non-thinking behavior with synthetic text only."""

    raw = tokenizer.apply_chat_template(
        [{"role": "user", "content": "SYNTHETIC_TEMPLATE_PROBE"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(raw, str):
        raise TokenizerArtifactError("chat template probe did not render text")
    count = raw.count(_QWEN_EMPTY_THINK_BLOCK)
    sanitized = raw.replace(_QWEN_EMPTY_THINK_BLOCK, "")
    if "<think>" in sanitized.casefold() or "</think>" in sanitized.casefold():
        raise TokenizerArtifactError("non-thinking probe contains non-empty think tags")
    return {
        "enable_thinking": False,
        "upstream_empty_think_block_count": count,
        "postprocess": "strip_exact_empty_qwen_think_block" if count else "none",
        "think_tags_after_postprocess": 0,
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
    }


def _encode(tokenizer, rendered: str) -> list[int]:
    return list(tokenizer(rendered, add_special_tokens=False)["input_ids"])


@dataclass(frozen=True)
class SFTTokenAudit:
    admitted: bool
    drop_reason: str | None
    truncated: bool
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    prompt_input_ids: list[int]
    input_ids: list[int]
    loss_mask: list[int]
    rendered: str


@dataclass(frozen=True)
class PromptTokenAudit:
    admitted: bool
    drop_reason: str | None
    prompt_tokens: int
    rendered: str


def audit_sft_record(
    tokenizer,
    *,
    question: str,
    reasoning: str | None,
    answer: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_length: int = 2048,
) -> SFTTokenAudit:
    values = [value for value in (question, reasoning, answer, system_prompt) if value]
    if any("<think>" in value.casefold() or "</think>" in value.casefold() for value in values):
        raise TokenizerArtifactError("SFT source text contains forbidden think tags")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    prompt = _render(tokenizer, messages, generation=True)
    assistant = "\n".join(
        value.strip() for value in (reasoning, answer) if value and value.strip()
    )
    full_messages = [*messages, {"role": "assistant", "content": assistant}]
    full = _render(tokenizer, full_messages, generation=False)
    prompt_ids = _encode(tokenizer, prompt)
    full_ids = _encode(tokenizer, full)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TokenizerArtifactError("Qwen3 full chat does not preserve the prompt token prefix")
    response_tokens = len(full_ids) - len(prompt_ids)
    if response_tokens <= 0:
        raise TokenizerArtifactError("assistant response and EOS tokenized to zero tokens")
    admitted = len(full_ids) <= max_length
    return SFTTokenAudit(
        admitted=admitted,
        drop_reason=None if admitted else "sft_full_length_exceeds_2048",
        truncated=False,
        prompt_tokens=len(prompt_ids),
        response_tokens=response_tokens,
        total_tokens=len(full_ids),
        prompt_input_ids=prompt_ids,
        input_ids=full_ids,
        loss_mask=[0] * len(prompt_ids) + [1] * response_tokens,
        rendered=full,
    )


def _audit_prompt(
    tokenizer, messages: Sequence[dict[str, str]], *, max_length: int, reason: str
) -> PromptTokenAudit:
    rendered = _render(tokenizer, messages, generation=True)
    count = len(_encode(tokenizer, rendered))
    return PromptTokenAudit(
        admitted=count <= max_length,
        drop_reason=None if count <= max_length else reason,
        prompt_tokens=count,
        rendered=rendered,
    )


def audit_opd_prompt(tokenizer, *, question: str, max_length: int = 512) -> PromptTokenAudit:
    return _audit_prompt(
        tokenizer,
        [{"role": "user", "content": question}],
        max_length=max_length,
        reason="opd_prompt_length_exceeds_512",
    )


def audit_eval_prompt(
    tokenizer,
    *,
    question: str,
    options: Sequence[str],
    max_length: int = 2048,
) -> PromptTokenAudit:
    content = format_mcq_question(question, options)
    return _audit_prompt(
        tokenizer,
        [{"role": "user", "content": content}],
        max_length=max_length,
        reason="eval_prompt_length_exceeds_limit",
    )


def length_summary(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(int(value) for value in values)

    def percentile(p: float) -> int:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }
