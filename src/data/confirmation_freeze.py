"""Freeze a one-use, non-final MedQA validation confirmation role.

Selection is independent of model output and stores raw prompts/labels only in the caller-selected
ignored artifact directory. The returned/tracked manifest is redacted and hash-only.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class ConfirmationFreezeError(RuntimeError):
    pass


ROLE = "medical_teacher_confirmation_dev"
SOURCE_ROLE = "medical_controller_dev"
FORBIDDEN_PROMPT_FIELDS = frozenset(
    {"answer", "answer_idx", "label", "solution", "reasoning", "explanation"}
)
FORBIDDEN_LABEL_FIELDS = frozenset({"question", "options", "normalized_question", "normalized_options"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ConfirmationFreezeError(f"JSONL row {line_number} is not an object: {path}")
            yield value


def _sample_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("sample_id") or "").strip()
    if not value:
        raise ConfirmationFreezeError("confirmation row lacks sample_id")
    return value


def _assert_non_final(row: Mapping[str, Any]) -> None:
    role = str(row.get("target_role") or "")
    if "final" in role.casefold():
        raise ConfirmationFreezeError("final role is forbidden from confirmation freeze")
    if role != SOURCE_ROLE:
        raise ConfirmationFreezeError(f"confirmation source role must be {SOURCE_ROLE}")


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        _assert_non_final(row)
        sample_id = _sample_id(row)
        if sample_id in labels:
            raise ConfirmationFreezeError(f"duplicate label sample_id: {sample_id}")
        if FORBIDDEN_LABEL_FIELDS & set(row):
            raise ConfirmationFreezeError("label artifact contains prompt fields")
        answer = str(row.get("answer_idx") or "").strip().upper()
        if answer not in set("ABCDE"):
            raise ConfirmationFreezeError("confirmation label must be A/B/C/D/E")
        labels[sample_id] = dict(row)
    return labels


def _controller_exclusions(prompt_path: Path, label_path: Path) -> tuple[set[str], set[str], set[str]]:
    label_ids = set(_read_labels(label_path))
    ids: set[str] = set()
    content_hashes: set[str] = set()
    group_ids: set[str] = set()
    for row in _iter_jsonl(prompt_path):
        _assert_non_final(row)
        sample_id = _sample_id(row)
        if sample_id in ids:
            raise ConfirmationFreezeError(f"duplicate controller sample_id: {sample_id}")
        ids.add(sample_id)
        content_hashes.add(str(row.get("content_hash") or ""))
        group_ids.add(str(row.get("group_id") or ""))
    if ids != label_ids:
        raise ConfirmationFreezeError("controller prompt/label sample sets differ")
    return ids, content_hashes - {""}, group_ids - {""}


def _priority(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            )
            count += 1
    os.replace(temporary, path)
    return count, path.stat().st_size, _sha256(path)


def freeze_medical_teacher_confirmation(
    *,
    source_prompts: str | Path,
    source_labels: str | Path,
    controller_prompts: str | Path,
    controller_labels: str | Path,
    output_dir: str | Path,
    count: int,
    seed: int,
    source_revision: str,
    raw_file_sha256: str,
) -> dict[str, Any]:
    """Select a hash-stable validation subset without opening any final artifact."""

    if count <= 0:
        raise ConfirmationFreezeError("confirmation count must be positive")
    if len(source_revision) != 40 or len(raw_file_sha256) != 64:
        raise ConfirmationFreezeError("immutable source revision and raw SHA are required")
    prompt_path, label_path = Path(source_prompts), Path(source_labels)
    controller_prompt_path, controller_label_path = Path(controller_prompts), Path(controller_labels)
    for path in (prompt_path, label_path, controller_prompt_path, controller_label_path):
        if not path.is_file():
            raise ConfirmationFreezeError(f"required confirmation input missing: {path}")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ConfirmationFreezeError("confirmation output directory must be empty or new")
    destination.mkdir(parents=True, exist_ok=True)

    controller_ids, controller_hashes, controller_groups = _controller_exclusions(
        controller_prompt_path, controller_label_path
    )
    labels = _read_labels(label_path)
    seen: set[str] = set()
    heap: list[tuple[int, str, dict[str, Any]]] = []
    eligible = 0
    for row in _iter_jsonl(prompt_path):
        _assert_non_final(row)
        if FORBIDDEN_PROMPT_FIELDS & set(row):
            raise ConfirmationFreezeError("prompt artifact contains supervision fields")
        sample_id = _sample_id(row)
        if sample_id in seen:
            raise ConfirmationFreezeError(f"duplicate prompt sample_id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in labels:
            raise ConfirmationFreezeError(f"prompt has no physical label row: {sample_id}")
        content_hash = str(row.get("content_hash") or "")
        group_id = str(row.get("group_id") or "")
        if (
            sample_id in controller_ids
            or (content_hash and content_hash in controller_hashes)
            or (group_id and group_id in controller_groups)
        ):
            continue
        eligible += 1
        priority = int(_priority(sample_id, seed), 16)
        candidate = (-priority, sample_id, dict(row))
        if len(heap) < count:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    if seen != set(labels):
        raise ConfirmationFreezeError("source prompt/label sample sets differ")
    if len(heap) != count:
        raise ConfirmationFreezeError(
            f"only {len(heap)} eligible validation rows remain; required {count}"
        )

    selected = sorted(
        ((-negative, sample_id, row) for negative, sample_id, row in heap),
        key=lambda item: (item[0], item[1]),
    )
    selected_ids = [sample_id for _, sample_id, _ in selected]
    prompt_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for _, sample_id, raw_prompt in selected:
        prompt = dict(raw_prompt)
        prompt["target_role"] = ROLE
        prompt_rows.append(prompt)
        label = dict(labels[sample_id])
        label["target_role"] = ROLE
        label_rows.append(label)

    prompt_output = destination / f"{ROLE}.prompts.jsonl"
    label_output = destination / f"{ROLE}.labels.jsonl"
    prompt_count, prompt_bytes, prompt_sha = _atomic_jsonl(prompt_output, prompt_rows)
    label_count, label_bytes, label_sha = _atomic_jsonl(label_output, label_rows)
    selected_ids_sha = hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in selected_ids).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "data_protocol_version": "ca-opd-data-v2",
        "role": ROLE,
        "status": "frozen_before_candidate_results",
        "source": "bigbio/med_qa",
        "source_revision": source_revision,
        "source_upstream_split": "validation",
        "source_raw_file_sha256": raw_file_sha256,
        "source_prompt_artifact_sha256": _sha256(prompt_path),
        "source_label_artifact_sha256": _sha256(label_path),
        "selection": "sha256(seed:sample_id)_ascending",
        "seed": seed,
        "requested_count": count,
        "actual_count": len(selected_ids),
        "eligible_after_controller_exclusion": eligible,
        "selected_sample_ids_sha256": selected_ids_sha,
        "controller_overlap": {"sample_id": 0, "content_hash": 0, "group_id": 0},
        "prompt_label_separated": True,
        "final_authorized": False,
        "final_artifacts_opened": False,
        "one_use_confirmation": True,
        "artifacts": [
            {"kind": "prompts", "path": str(prompt_output), "count": prompt_count,
             "bytes": prompt_bytes, "sha256": prompt_sha, "supervision_fields": 0},
            {"kind": "labels", "path": str(label_output), "count": label_count,
             "bytes": label_bytes, "sha256": label_sha},
        ],
    }
    manifest_path = destination / f"{ROLE}.manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path)}
