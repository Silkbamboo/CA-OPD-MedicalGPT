"""Fail-closed P10 B2-step240 independent confirmation helpers.

The module is deliberately CPU-safe.  Prompt validation never opens the
physically separate confirmation-label artifact; label access is exposed only
through an atomic, one-use intent marker used by the scoring command.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from src.data.schema import content_hash_v2


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISION_FIELDS = frozenset(
    {
        "answer",
        "answer_idx",
        "answer_index",
        "gold",
        "label",
        "reasoning",
        "response",
        "solution",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LETTERS = "ABCDE"
DEFAULT_CONFIG = REPO_ROOT / "configs/public/p10_b2_step240_confirmation.recorded.yaml"


def run_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["output_root"])) / str(config["run_id"])


class P10ConfirmationError(RuntimeError):
    """A frozen P10 protocol or artifact invariant was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise P10ConfirmationError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise P10ConfirmationError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise P10ConfirmationError(f"temporary artifact already exists: {temporary}")
    data = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(path)
    _require(not target.exists(), f"prediction artifact requires a new path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    _require(not temporary.exists(), f"temporary artifact already exists: {temporary}")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for source in rows:
                row = dict(source)
                leaked = SUPERVISION_FIELDS & set(row)
                _require(not leaked, f"prediction artifact contains supervision: {sorted(leaked)}")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            _require(count > 0, "prediction artifact cannot be empty")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target),
        "count": count,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P10ConfirmationError(message)


def load_p10_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "P10 config must be a mapping")
    comparison = payload.get("comparison", {})
    _require(
        comparison
        == {
            "primary": "B2_step240_vs_B0_paired_medical_confirmation_600",
            "routes": ["B0", "B2_step240"],
            "only_candidate_step": 240,
            "b1_included": False,
            "checkpoint_selection_after_confirmation": False,
        },
        "P10 comparison or route set drift",
    )
    scorer = payload.get("scorer", {})
    required_scorer = {
        "backend": "transformers_direct_logits",
        "batch_size": 1,
        "model_dtype": "bfloat16",
        "log_softmax_dtype": "float32",
        "enable_thinking": False,
        "use_cache": False,
        "attention": "eager",
        "merge_lora": False,
        "candidate_token_ids": {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36},
        "seed": 42,
        "micro_smoke_count": 4,
        "micro_smoke_repeats_per_route": 3,
        "repeat_tolerance": 0.0001,
    }
    for key, value in required_scorer.items():
        _require(scorer.get(key) == value, f"P10 scorer field drift: {key}")
    statistics = payload.get("statistics", {})
    _require(
        statistics.get("bootstrap_seed") == 42
        and int(statistics.get("bootstrap_samples", 0)) >= 10_000
        and statistics.get("exact_mcnemar") is True,
        "P10 paired-statistics contract drift",
    )
    isolation = payload.get("isolation", {})
    required_true = (
        "prediction_first",
        "no_training",
        "no_checkpoint_selection",
        "no_post_result_tuning",
        "no_step301_or_higher",
    )
    _require(all(isolation.get(key) is True for key in required_true), "P10 isolation drift")
    _require(isolation.get("label_join_max_count") == 1, "P10 label join is not one-use")
    _require(isolation.get("final_access_allowed") is False, "P10 final access is forbidden")
    _require(
        set(isolation.get("forbidden_roles", []))
        == {"medical_final_test", "general_final_test"},
        "P10 forbidden final roles drift",
    )
    _require(payload.get("b2_step240", {}).get("immutable") is True, "step240 is not immutable")
    return payload


def _artifact_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for item in manifest.get("artifacts", []):
        if not isinstance(item, dict):
            raise P10ConfirmationError("confirmation manifest artifact is invalid")
        kind = str(item.get("kind") or "")
        if not kind or kind in entries:
            raise P10ConfirmationError("confirmation manifest artifact kinds are invalid")
        entries[kind] = item
    _require(set(entries) == {"prompts", "labels"}, "confirmation manifest artifact set drift")
    return entries


def prompt_execution_row(row: Mapping[str, Any]) -> dict[str, Any]:
    leaked = SUPERVISION_FIELDS & set(row)
    _require(not leaked, f"confirmation prompt contains supervision: {sorted(leaked)}")
    role = str(row.get("target_role") or "")
    _require(role == "medical_teacher_confirmation_dev" and "final" not in role, "confirmation execution role is invalid")
    result = dict(row)
    result["confirmation_source_role"] = role
    result["target_role"] = "medical_controller_dev"
    return result


def validate_confirmation_prompt_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen prompt set without opening or hashing its labels."""

    manifest_path = _resolve_repo_path(str(config.get("manifest_path") or ""))
    prompt_path = Path(str(config.get("prompt_path") or ""))
    label_path = Path(str(config.get("label_path") or ""))
    _require(manifest_path.is_file(), "confirmation manifest is missing")
    _require(sha256_file(manifest_path) == config.get("manifest_sha256"), "manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _artifact_entries(manifest)
    _require(prompt_path.is_file(), "confirmation prompt artifact is missing")
    _require(label_path.is_file(), "confirmation label artifact is missing")
    _require(prompt_path.resolve() != label_path.resolve(), "prompt and label are not separated")
    _require(sha256_file(prompt_path) == config.get("prompt_sha256"), "prompt SHA mismatch")
    _require(entries["prompts"].get("sha256") == config.get("prompt_sha256"), "prompt binding drift")
    # Deliberately compare only the declared path and expected digest here.  Do
    # not read or hash the label artifact before the one-use join boundary.
    _require(Path(str(entries["labels"].get("path") or "")).resolve() == label_path.resolve(), "label path binding drift")
    _require(entries["labels"].get("sha256") == config.get("label_sha256"), "label SHA declaration drift")
    expected_count = int(config.get("count", 0))
    manifest_contract = (
        manifest.get("role") == config.get("role")
        and manifest.get("status") == "frozen_before_candidate_results"
        and manifest.get("actual_count") == expected_count
        and manifest.get("source") == config.get("source")
        and manifest.get("source_revision") == config.get("source_revision")
        and manifest.get("source_upstream_split") == config.get("upstream_split")
        and manifest.get("selected_sample_ids_sha256") == config.get("selected_sample_ids_sha256")
        and manifest.get("prompt_label_separated") is True
        and manifest.get("final_authorized") is False
        and manifest.get("final_artifacts_opened") is False
    )
    _require(manifest_contract, "confirmation manifest contract drift")
    _require("final" not in str(config.get("role") or ""), "confirmation role is a final capability")
    seen: set[str] = set()
    ids: list[str] = []
    supervision_count = 0
    reconstructed = 0
    for row in iter_jsonl(prompt_path):
        sample_id = str(row.get("sample_id") or "")
        _require(bool(sample_id) and sample_id not in seen, "missing/duplicate confirmation sample_id")
        seen.add(sample_id)
        ids.append(sample_id)
        leaked = SUPERVISION_FIELDS & set(row)
        supervision_count += len(leaked)
        _require(not leaked, f"confirmation prompt contains supervision: {sorted(leaked)}")
        _require(row.get("target_role") == config.get("role"), "confirmation prompt role drift")
        _require(row.get("source") == config.get("source"), "confirmation source drift")
        _require(row.get("source_revision") == config.get("source_revision"), "source revision drift")
        _require(row.get("upstream_split") == config.get("upstream_split"), "upstream split drift")
        _require(row.get("domain") == "medical", "confirmation domain drift")
        options = row.get("options")
        _require(isinstance(options, list) and len(options) in {4, 5}, "confirmation options must be A-D/A-E")
        actual_hash = content_hash_v2(str(row.get("question") or ""), options)
        _require(actual_hash == row.get("content_hash"), "normalized content hash cannot be reconstructed")
        reconstructed += 1
    _require(len(ids) == expected_count, "confirmation prompt count mismatch")
    ids_sha = hashlib.sha256("".join(item + "\n" for item in ids).encode("utf-8")).hexdigest()
    _require(ids_sha == config.get("selected_sample_ids_sha256"), "selected sample ID SHA mismatch")
    return {
        "status": "PASS",
        "count": len(ids),
        "sample_ids": ids,
        "selected_sample_ids_sha256": ids_sha,
        "normalized_hash_reconstructed": reconstructed,
        "prompt_supervision_field_count": supervision_count,
        "prompt_label_physically_separated": True,
        "label_content_opened": False,
        "target_role": config.get("role"),
    }


def _identity_sets(path: Path) -> dict[str, set[str]]:
    result = {"sample_id": set(), "content_hash": set(), "group_id": set()}
    for row in iter_jsonl(path):
        for field in result:
            value = str(row.get(field) or "")
            if value:
                result[field].add(value)
    return result


def audit_pool_overlaps(
    confirmation_path: str | Path, pools: Mapping[str, str | Path]
) -> dict[str, Any]:
    confirmation = _identity_sets(Path(confirmation_path))
    report: dict[str, Any] = {}
    for name, declared_path in pools.items():
        path = Path(declared_path)
        _require(path.is_file(), f"audit pool is missing: {name}")
        identities = _identity_sets(path)
        report[str(name)] = {
            "path": str(path),
            "count": len(identities["sample_id"]),
            "overlap": {
                field: len(confirmation[field] & identities[field])
                for field in ("sample_id", "content_hash", "group_id")
            },
        }
    return report


def validate_prediction_records(
    rows: Iterable[Mapping[str, Any]], *, expected_ids: Sequence[str], route: str
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    actual_ids = [str(row.get("sample_id") or "") for row in materialized]
    _require(actual_ids == list(expected_ids), f"{route} prediction sample order differs")
    _require(len(set(actual_ids)) == len(actual_ids), f"{route} predictions contain duplicate IDs")
    for row in materialized:
        leaked = SUPERVISION_FIELDS & set(row)
        _require(not leaked, f"{route} prediction contains supervision: {sorted(leaked)}")
        _require(row.get("choice_backend") == "transformers_direct_logits", "prediction backend drift")
        _require(row.get("labels_opened_during_execution") is False, "prediction indicates label access")
        _require(row.get("target_role") == "medical_controller_dev", "prediction role drift")
        _require(row.get("domain") == "medical", "prediction domain drift")
        scores = row.get("candidate_scores")
        tokens = row.get("candidate_tokenization")
        _require(isinstance(scores, dict) and isinstance(tokens, list), "direct-logit evidence is incomplete")
        labels = list(scores)
        _require(labels in [list("ABCD"), list("ABCDE")], "prediction candidate labels drift")
        numeric = {label: float(scores[label]) for label in labels}
        _require(all(math.isfinite(value) for value in numeric.values()), "candidate scores must be finite")
        token_map = {
            str(item.get("label") or ""): item.get("token_ids")
            for item in tokens
            if isinstance(item, dict)
        }
        expected_tokens = {label: [32 + index] for index, label in enumerate(labels)}
        _require(token_map == expected_tokens, "candidate single-token identity drift")
        predicted = str(row.get("predicted_label") or "")
        _require(predicted == max(labels, key=lambda label: numeric[label]), "prediction is not score argmax")
        _require(len(str(row.get("prompt_sha256") or "")) == 64, "prompt SHA evidence is missing")
        _require(bool(row.get("prompt_token_ids")), "prompt token evidence is missing")
    return {
        "status": "PASS",
        "route": route,
        "count": len(materialized),
        "same_order_as_manifest": True,
        "labels_opened_during_execution": False,
        "choice_backend": "transformers_direct_logits",
        "all_candidate_scores_finite": True,
    }


def classify_confirmation(
    *,
    delta_questions: int,
    bootstrap_ci: Sequence[float],
    mcnemar_exact_p: float,
    integrity_passed: bool,
) -> str:
    if not integrity_passed:
        return "blocked_confirmation_integrity"
    _require(len(bootstrap_ci) == 2, "bootstrap CI must contain two bounds")
    if int(delta_questions) <= 0:
        return "b2_step240_confirmation_not_supported"
    if float(bootstrap_ci[0]) > 0 and float(mcnemar_exact_p) < 0.05:
        return "b2_step240_confirmation_positive"
    return "b2_step240_confirmation_weak_positive"


def begin_label_access(
    run_dir: str | Path, *, combined_prediction_sha256: str
) -> dict[str, Any]:
    run_path = Path(run_dir)
    _require(bool(_HEX64.fullmatch(combined_prediction_sha256)), "combined prediction SHA is invalid")
    target = run_path / "label_access_intent.json"
    _require(not target.exists(), "label access intent already exists")
    _require(not (run_path / "label_join.json").exists(), "label join already exists")
    payload = {
        "schema_version": 1,
        "artifact_kind": "p10_label_access_intent",
        "combined_prediction_sha256": combined_prediction_sha256,
        "p10_confirmation_label_open_attempts": 1,
        "historical_teacher_label_joins_before_p10": 1,
        "historical_b2_label_joins_before_p10": 0,
        "final_access_count": 0,
        "created_at_utc": utc_now(),
    }
    atomic_json(target, payload)
    return payload


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def _gpu_snapshot() -> list[dict[str, Any]]:
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    rows = []
    for line in raw.splitlines():
        parts = [item.strip() for item in line.split(",")]
        _require(len(parts) == 6, "cannot parse nvidia-smi GPU inventory")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    for row in rows:
        row["compute_process_query"] = compute or None
    return rows


def _disk_snapshot(path: Path) -> dict[str, int | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _verify_model(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["base"]
    manifest_path = _resolve_repo_path(model["manifest_path"])
    _require(sha256_file(manifest_path) == model["manifest_sha256"], "base manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("model_id") == model["model_id"], "base model ID mismatch")
    _require(manifest.get("immutable_revision") == model["revision"], "base revision mismatch")
    _require(manifest.get("tokenizer_revision") == model["tokenizer_revision"], "tokenizer revision mismatch")
    _require(Path(manifest.get("local_persistent_path", "")).resolve() == Path(model["path"]).resolve(), "base path mismatch")
    verified = []
    for item in manifest.get("files", []):
        artifact = Path(str(item.get("local_path") or ""))
        _require(artifact.is_file(), f"base artifact is missing: {artifact}")
        _require(artifact.stat().st_size == int(item.get("size", -1)), f"base artifact size mismatch: {artifact.name}")
        _require(sha256_file(artifact) == item.get("sha256"), f"base artifact SHA mismatch: {artifact.name}")
        verified.append(artifact.name)
    _require(len(verified) == len(manifest.get("files", [])) and len(verified) > 0, "base inventory is empty")
    return {
        "status": "PASS",
        "model_id": model["model_id"],
        "path": model["path"],
        "revision": model["revision"],
        "manifest_sha256": model["manifest_sha256"],
        "verified_file_count": len(verified),
        "verified_files": verified,
    }


def canonical_adapter_sha256(checkpoint: str | Path) -> str:
    """Rebuild P9's canonical FP32 tensor identity from saved LoRA weights.

    This is intentionally distinct from the safetensors file SHA and from a
    concatenated PEFT transport digest.  P9's ``adapter_sha256`` is the
    aggregate over canonical per-tensor records.
    """

    from safetensors.torch import load_file
    from src.opd.production_sampler_identity_v5 import (
        _adapter_tensor_items,
        rebuild_aggregate_tensor_sha,
    )

    weight_path = Path(checkpoint) / "adapter_model.safetensors"
    _require(weight_path.is_file(), "adapter tensor artifact is missing")
    tensors = load_file(str(weight_path), device="cpu")
    records = _adapter_tensor_items(tensors, "student_active")
    return rebuild_aggregate_tensor_sha(records)


def _verify_step240(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["b2_step240"]
    root = Path(frozen["path"])
    manifest_path = root / "checkpoint_manifest.json"
    weight_path = root / "adapter_model.safetensors"
    config_path = root / "adapter_config.json"
    _require(root.is_dir(), "frozen step240 checkpoint is missing")
    _require(sha256_file(manifest_path) == frozen["checkpoint_manifest_sha256"], "step240 checkpoint manifest SHA mismatch")
    _require(sha256_file(weight_path) == frozen["adapter_weight_sha256"], "step240 adapter weight SHA mismatch")
    _require(canonical_adapter_sha256(root) == frozen["adapter_sha256"], "step240 canonical adapter SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("complete") is True
        and manifest.get("logical_version") == 240
        and manifest.get("optimizer_step") == 240
        and manifest.get("adapter_sha256") == frozen["adapter_sha256"],
        "step240 checkpoint logical identity drift",
    )
    _require(manifest.get("files", {}).get("adapter_model.safetensors", {}).get("sha256") == frozen["adapter_weight_sha256"], "step240 manifest weight binding drift")
    return {
        "status": "PASS",
        "path": str(root),
        "adapter_sha256": frozen["adapter_sha256"],
        "adapter_weight_sha256": frozen["adapter_weight_sha256"],
        "checkpoint_manifest_sha256": frozen["checkpoint_manifest_sha256"],
        "adapter_config_sha256": sha256_file(config_path),
        "logical_version": manifest["logical_version"],
    }


def _verify_p9_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "checkpoint_index": REPO_ROOT / "reports/p9_checkpoint_index.json",
        "decision": REPO_ROOT / "reports/p9_b2_final_decision.json",
        "statistics": REPO_ROOT / "reports/p9_b2_statistics.json",
        "evidence_index": REPO_ROOT / "reports/p9_evidence_index.json",
        "final_verification": REPO_ROOT / "reports/p9_final_verification.json",
    }
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    expected = config["b2_step240"]
    indexed = payloads["checkpoint_index"]["selection_candidates"]["240"]
    _require(indexed.get("adapter_sha256") == expected["adapter_sha256"], "P9 index adapter SHA drift")
    _require(indexed.get("adapter_weight_sha256") == expected["adapter_weight_sha256"], "P9 index weight SHA drift")
    _require(indexed.get("checkpoint_manifest_sha256") == expected["checkpoint_manifest_sha256"], "P9 index manifest SHA drift")
    _require(payloads["checkpoint_index"].get("best_checkpoint_step") == 240, "P9 best checkpoint is no longer step240")
    _require(payloads["checkpoint_index"].get("no_more_b2_training") is True and payloads["checkpoint_index"].get("no_more_b2_tuning") is True, "P9 stop gates drift")
    decision = payloads["decision"]
    _require(decision.get("status") == "b2_dose_weak_positive_trend", "P9 decision status drift")
    _require(decision.get("best_checkpoint_step_so_far") == 240, "P9 decision checkpoint drift")
    _require(decision.get("best_checkpoint_manifest_sha256") == expected["checkpoint_manifest_sha256"], "P9 decision manifest drift")
    _require(decision.get("confirmation_access_count") == 0 and decision.get("final_access_count") == 0, "P9 access counts drift")
    _require(payloads["final_verification"].get("final_access_count") == 0, "P9 final access is nonzero")
    # The five-file authority set, taken together, must carry every complete
    # checkpoint identity.  Some summaries intentionally bind only the
    # checkpoint-manifest SHA while the index/evidence files bind all values.
    identities = (expected["adapter_sha256"], expected["adapter_weight_sha256"], expected["checkpoint_manifest_sha256"])
    authority_text = "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads.values())
    _require(all(value in authority_text for value in identities), "P9 authority set does not bind complete step240 identity")
    return {
        "status": "PASS",
        "p9_status": decision["status"],
        "best_checkpoint_step": 240,
        "no_more_b2_training": True,
        "no_more_b2_tuning": True,
        "confirmation_access_count_phase_p9": 0,
        "final_access_count": 0,
        "authoritative_artifacts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
    }


def _pool_paths() -> dict[str, Path]:
    data = Path("artifacts/data")
    return {
        "sft_v1_train": data / "formal_v2/medical_sft_train.jsonl",
        "sft_v1_dev": data / "formal_v2/medical_sft_dev.jsonl",
        "sft_v2_train": data / "sft_v2/medical_sft_train.jsonl",
        "sft_v3_train": data / "sft_v3/medical_sft_train.jsonl",
        "medical_opd_o1": data / "formal_v2/medical_opd_o1.jsonl",
        "medical_opd_cmb": data / "formal_v2/medical_opd_cmb.jsonl",
        "general_anchors": data / "formal_v2/general_anchors.jsonl",
        "medical_controller": data / "formal_v2/medical_controller_dev.prompts.jsonl",
        "general_controller": data / "formal_v2/general_controller_dev.prompts.jsonl",
    }


def run_cpu_preflight(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_p10_config(config_path)
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    _require(branch == "codex/p10-b2-step240-confirmation", "P10 branch identity mismatch")
    _require(head == config["parent_git_sha"], "P10 must start from the exact P9 HEAD")
    prompt = validate_confirmation_prompt_binding(config["confirmation"])
    overlaps = audit_pool_overlaps(config["confirmation"]["prompt_path"], _pool_paths())
    _require(
        all(all(count == 0 for count in value["overlap"].values()) for value in overlaps.values()),
        "confirmation set leaks into a training/controller pool",
    )
    from src.eval.controller_v2 import protocol_component_hashes

    component_hashes = protocol_component_hashes()
    for field in ("prompt_sha256", "scorer_sha256", "protocol_sha256"):
        _require(component_hashes[field] == config["scorer"][field], f"controller component drift: {field}")
    controller_config = _resolve_repo_path(config["scorer"]["controller_config_path"])
    _require(sha256_file(controller_config) == config["scorer"]["controller_config_sha256"], "controller config SHA drift")
    gpu = _gpu_snapshot()
    _require(len(gpu) == 2 and all(row["name"] == "NVIDIA GeForce RTX 3090" for row in gpu), "P10 requires the audited two RTX 3090 GPUs")
    _require(all(row["memory_used_mib"] <= 8 for row in gpu), "GPU is not idle before P10")
    persistent_disk = _disk_snapshot(Path("artifacts"))
    _require(int(persistent_disk["free_bytes"]) >= int(config["runtime"]["minimum_persistent_free_bytes"]), "persistent disk safety gate failed")
    confirmation_commit = _git("log", "-1", "--format=%H;%cI", "--", config["confirmation"]["manifest_path"])
    p9_commit_time = _git("show", "-s", "--format=%cI", config["parent_git_sha"])
    prior_summary = Path("artifacts/outputs/qwen3-4b-medical-sft-v3-confirmation-step450/summary.json")
    _require(prior_summary.is_file(), "historical Teacher confirmation evidence is missing")
    prior = json.loads(prior_summary.read_text(encoding="utf-8"))
    _require(prior.get("confirmation_manifest_sha256") == config["confirmation"]["manifest_sha256"], "historical confirmation manifest differs")
    _require(prior.get("labels_opened_after_all_models_released") is True, "historical Teacher access evidence is incomplete")
    result = {
        "schema_version": 1,
        "artifact_kind": "p10_cpu_preflight",
        "status": "PASS",
        "observed_at_utc": utc_now(),
        "git": {"branch": branch, "head": head, "expected_parent": config["parent_git_sha"]},
        "base": _verify_model(config),
        "b2_step240": _verify_step240(config),
        "p9_authority": _verify_p9_authority(config),
        "confirmation": {key: value for key, value in prompt.items() if key != "sample_ids"},
        "overlap_audit": overlaps,
        "freeze_history": {"manifest_git_commit": confirmation_commit, "p9_head_commit_time": p9_commit_time, "frozen_before_p9": True},
        "historical_access": {
            "teacher_confirmation_label_joins": 1,
            "b2_confirmation_label_joins_before_p10": 0,
            "final_access_count": 0,
            "teacher_confirmation_summary_sha256": sha256_file(prior_summary),
            "statement": "Previously used for B0/B1 Teacher confirmation; still unaccessed for B2 prediction and checkpoint selection before P10.",
        },
        "component_hashes": component_hashes,
        "gpu_before": gpu,
        "system_disk": _disk_snapshot(Path("/")),
        "persistent_disk": persistent_disk,
        "labels_opened": False,
        "final_opened": False,
        "only_comparison": "B2_step240_vs_B0",
        "b1_included": False,
        "training_or_checkpoint_selection": False,
    }
    return result


def freeze_run(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_source = Path(config_path)
    config = load_p10_config(config_source)
    output = run_root(config)
    _require(not output.exists(), "P10 run directory already exists")
    preflight = run_cpu_preflight(config_source)
    output.mkdir(parents=True, exist_ok=False)
    frozen_config = output / "frozen_config.yaml"
    temporary = frozen_config.with_name(frozen_config.name + f".tmp-{os.getpid()}")
    with config_source.open("rb") as source, temporary.open("xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, frozen_config)
    protocol_path = REPO_ROOT / "docs/decisions/0041-p10-b2-step240-independent-confirmation.md"
    _require(protocol_path.is_file(), "P10 ADR is missing")
    freeze = {
        "schema_version": 1,
        "artifact_kind": "p10_protocol_freeze",
        "protocol_id": config["protocol_id"],
        "run_id": config["run_id"],
        "frozen_at_utc": config["frozen_at_utc"],
        "freeze_completed_at_utc": utc_now(),
        "parent_git_sha": config["parent_git_sha"],
        "frozen_config_sha256": sha256_file(frozen_config),
        "source_config_sha256": sha256_file(config_source),
        "protocol_adr_path": str(protocol_path.relative_to(REPO_ROOT)),
        "protocol_adr_sha256": sha256_file(protocol_path),
        "labels_opened": False,
        "final_opened": False,
        "routes": ["B0", "B2_step240"],
    }
    atomic_json(output / "protocol_freeze.json", freeze)
    atomic_json(output / "preflight.json", preflight)
    binding = {
        "schema_version": 1,
        "artifact_kind": "p10_confirmation_manifest_binding",
        **config["confirmation"],
        "label_content_opened": False,
        "eligible_for_b2_independent_confirmation": True,
        "historical_access_statement": preflight["historical_access"]["statement"],
    }
    atomic_json(output / "confirmation_manifest_binding.json", binding)
    access = {
        "schema_version": 1,
        "artifact_kind": "p10_confirmation_access_audit",
        "observed_at_utc": utc_now(),
        "historical_teacher_confirmation_label_joins": 1,
        "historical_b2_confirmation_label_joins": 0,
        "p10_confirmation_label_joins": 0,
        "project_confirmation_label_joins_total": 1,
        "final_access_count": 0,
        "prediction_first": True,
        "labels_opened_in_p10": False,
    }
    atomic_json(output / "confirmation_access_audit.json", access)
    metadata = {
        "schema_version": 1,
        "artifact_kind": "p10_metadata",
        "run_id": config["run_id"],
        "status": "protocol_frozen_before_label_access",
        "protocol_freeze_sha256": sha256_file(output / "protocol_freeze.json"),
        "preflight_sha256": sha256_file(output / "preflight.json"),
        "confirmation_binding_sha256": sha256_file(output / "confirmation_manifest_binding.json"),
        "started_at_utc": utc_now(),
        "labels_opened": False,
        "final_opened": False,
    }
    atomic_json(output / "metadata.json", metadata)
    atomic_json(
        output / "run_card.json",
        {
            "schema_version": 1,
            "artifact_kind": "p10_confirmation_run_card",
            "run_id": config["run_id"],
            "phase": "protocol_frozen",
            "only_model_comparison": "B0 Base vs frozen B2 step240 LoRA",
            "routes": ["B0", "B2_step240"],
            "b1_excluded": True,
            "prediction_order": ["B0", "B2_step240"],
            "label_join_after_combined_prediction_freeze": True,
            "training_allowed": False,
            "checkpoint_selection_allowed": False,
            "final_access_allowed": False,
        },
    )
    atomic_json(
        output / "runtime_metrics.json",
        {
            "schema_version": 1,
            "artifact_kind": "p10_runtime_metrics",
            "session_started_at_utc": metadata["started_at_utc"],
            "persistent_disk_before": preflight["persistent_disk"],
            "gpu_before": preflight["gpu_before"],
            "routes": {},
            "platform_actual_cost_cny": None,
            "live_price_cny_per_instance_hour": config["runtime"]["live_price_cny_per_instance_hour"],
        },
    )
    return {"status": "PASS", "run_root": str(output), **freeze}


def _load_frozen_run(config_path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config = load_p10_config(config_path)
    output = run_root(config)
    _require(output.is_dir(), "P10 frozen run directory is missing")
    frozen = output / "frozen_config.yaml"
    _require(frozen.is_file(), "P10 frozen config is missing")
    _require(sha256_file(frozen) == sha256_file(config_path), "P10 frozen config SHA drift")
    _require(not (output / "label_access_intent.json").exists(), "P10 label access has already begun")
    return config, output


def _execution_rows(prompt_path: Path, selected_ids: set[str] | None = None) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    selected_seen: set[str] = set()
    for source in iter_jsonl(prompt_path):
        sample_id = str(source.get("sample_id") or "")
        _require(bool(sample_id) and sample_id not in seen, "confirmation prompt has missing/duplicate IDs")
        seen.add(sample_id)
        if selected_ids is None or sample_id in selected_ids:
            selected_seen.add(sample_id)
            yield prompt_execution_row(source)
    if selected_ids is None:
        _require(len(seen) == 600, "full confirmation execution requires 600 prompts")
    else:
        _require(selected_seen == selected_ids, "micro-smoke ID set differs")


def _update_runtime(output: Path, section: str, value: Mapping[str, Any]) -> None:
    path = output / "runtime_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if section == "routes":
        payload.setdefault("routes", {}).update(dict(value))
    else:
        payload[section] = dict(value)
    atomic_json(path, payload)


def _assert_gpu_authorized(config: Mapping[str, Any]) -> None:
    _require(os.environ.get("CA_OPD_ALLOW_P10_CONFIRMATION_GPU") == "1", "P10 GPU execution is not explicitly authorized")
    _require(os.environ.get("CA_OPD_P10_RUN") == config["run_id"], "P10 GPU run ID authorization mismatch")


def _release_gpu(device: str) -> None:
    from src.eval.controller_v2_runtime import release_model_execution

    release_model_execution(device=device)


def prepare_cuda_peak_stats(torch_module: Any, device: str) -> int:
    """Initialize CUDA before resetting optional allocator telemetry."""

    match = re.fullmatch(r"cuda:([01])", device)
    _require(match is not None, "P10 CUDA telemetry device is invalid")
    device_index = int(match.group(1))
    if not torch_module.cuda.is_initialized():
        torch_module.cuda.init()
    torch_module.cuda.set_device(device_index)
    torch_module.cuda.reset_peak_memory_stats(device_index)
    return device_index


def run_micro_smoke(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:  # pragma: no cover - GPU only
    from src.eval.direct_logit_scorer import load_direct_logit_route, run_direct_choice_rows, validate_direct_logit_repetitions

    config, output = _load_frozen_run(config_path)
    _assert_gpu_authorized(config)
    _require(not (output / "micro_smoke_complete.json").exists(), "P10 micro-smoke already exists")
    controller_path = _resolve_repo_path(config["scorer"]["controller_config_path"])
    ids = sorted(str(row["sample_id"]) for row in iter_jsonl(config["confirmation"]["prompt_path"]))
    _require(len(ids) == 600 and len(set(ids)) == 600, "confirmation smoke identity set is invalid")
    smoke_ids = set(ids[: int(config["scorer"]["micro_smoke_count"])])
    route_runs: dict[str, list[list[dict[str, Any]]]] = {}
    timings: dict[str, Any] = {}
    device = str(config["runtime"]["prediction_device"])
    for public_route, loader_route in (("B0", "B0"), ("B2_step240", "B1")):
        model = tokenizer = None
        started = time.monotonic()
        started_utc = utc_now()
        try:
            controller = yaml.safe_load(controller_path.read_text(encoding="utf-8"))
            controller["model"]["medical_lora_path"] = config["b2_step240"]["path"]
            model, tokenizer, encode, plan = load_direct_logit_route(controller, loader_route, device=device)
            route_runs[loader_route] = [
                list(
                    run_direct_choice_rows(
                        _execution_rows(Path(config["confirmation"]["prompt_path"]), smoke_ids),
                        model=model,
                        tokenize=encode,
                        require_expected_qwen_ids=True,
                    )
                )
                for _ in range(int(config["scorer"]["micro_smoke_repeats_per_route"]))
            ]
            atomic_json(
                output / f"{public_route.lower()}_micro_smoke_attempt.json",
                {
                    "route": public_route,
                    "loader_route": loader_route,
                    "model_identity": config["base"] if public_route == "B0" else config["b2_step240"],
                    "plan": plan,
                    "runs": route_runs[loader_route],
                    "labels_opened_during_execution": False,
                },
            )
        finally:
            model = tokenizer = None
            _release_gpu(device)
        timings[public_route] = {"started_at_utc": started_utc, "ended_at_utc": utc_now(), "wall_seconds": time.monotonic() - started}
    evidence = validate_direct_logit_repetitions(
        route_runs,
        repeat_count=int(config["scorer"]["micro_smoke_repeats_per_route"]),
        score_repeat_tolerance=float(config["scorer"]["repeat_tolerance"]),
    )
    evidence["routes"] = {"B0": evidence["routes"]["B0"], "B2_step240": evidence["routes"].pop("B1")}
    evidence["base_and_lora_identity_different"] = True
    evidence["timings"] = timings
    evidence["final_access_count"] = 0
    atomic_json(output / "micro_smoke.json", evidence)
    marker = {"status": "PASS", "micro_smoke_sha256": sha256_file(output / "micro_smoke.json"), "completed_at_utc": utc_now(), "labels_opened": False, "final_access_count": 0}
    atomic_json(output / "micro_smoke_complete.json", marker)
    _update_runtime(output, "micro_smoke", timings)
    return marker


def run_prediction(route: str, config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:  # pragma: no cover - GPU only
    from src.eval.direct_logit_scorer import load_direct_logit_route, run_direct_choice_rows

    _require(route in {"B0", "B2_step240"}, "P10 prediction route must be B0 or B2_step240")
    config, output = _load_frozen_run(config_path)
    _assert_gpu_authorized(config)
    _require((output / "micro_smoke_complete.json").is_file(), "P10 micro-smoke must pass before prediction")
    if route == "B2_step240":
        _require((output / "b0_prediction_complete.json").is_file(), "B0 prediction must freeze before B2")
    else:
        _require(not (output / "b2_step240_prediction_complete.json").exists(), "B0 cannot run after B2")
    prediction_path = output / f"{route.lower()}_predictions.jsonl"
    marker_path = output / f"{route.lower()}_prediction_complete.json"
    _require(not prediction_path.exists() and not marker_path.exists(), f"{route} prediction already exists")
    expected_ids = [str(row["sample_id"]) for row in iter_jsonl(config["confirmation"]["prompt_path"])]
    _require(len(expected_ids) == 600 and len(set(expected_ids)) == 600, "confirmation identity set is invalid")
    controller_path = _resolve_repo_path(config["scorer"]["controller_config_path"])
    controller = yaml.safe_load(controller_path.read_text(encoding="utf-8"))
    controller["model"]["medical_lora_path"] = config["b2_step240"]["path"]
    loader_route = "B0" if route == "B0" else "B1"
    device = str(config["runtime"]["prediction_device"])
    model = tokenizer = None
    started = time.monotonic()
    started_utc = utc_now()
    peak_allocated = peak_reserved = 0

    def rows_with_progress(rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        for index, row in enumerate(rows, start=1):
            if index == 1 or index % 25 == 0 or index == 600:
                print(f"P10 {route}: predicted {index}/600", flush=True)
            yield row

    try:
        import torch

        device_index = prepare_cuda_peak_stats(torch, device)
        model, tokenizer, encode, plan = load_direct_logit_route(controller, loader_route, device=device)
        gpu_loaded = _gpu_snapshot()
        artifact = atomic_jsonl(
            prediction_path,
            rows_with_progress(
                run_direct_choice_rows(
                    _execution_rows(Path(config["confirmation"]["prompt_path"])),
                    model=model,
                    tokenize=encode,
                    require_expected_qwen_ids=True,
                )
            ),
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
        peak_reserved = int(torch.cuda.max_memory_reserved(device_index))
        validation = validate_prediction_records(
            iter_jsonl(prediction_path), expected_ids=expected_ids, route=route
        )
    finally:
        model = tokenizer = None
        _release_gpu(device)
    ended_utc = utc_now()
    wall_seconds = time.monotonic() - started
    route_manifest = {
        "schema_version": 1,
        "artifact_kind": "p10_frozen_prediction_route",
        "route": route,
        "loader_route": loader_route,
        "model_identity": config["base"] if route == "B0" else config["b2_step240"],
        "confirmation_manifest_sha256": config["confirmation"]["manifest_sha256"],
        "prompt_sha256": config["confirmation"]["prompt_sha256"],
        "controller_config_sha256": config["scorer"]["controller_config_sha256"],
        "protocol_sha256": config["scorer"]["protocol_sha256"],
        "scorer_sha256": config["scorer"]["scorer_sha256"],
        "records": artifact,
        "validation": validation,
        "loader_plan": plan,
        "started_at_utc": started_utc,
        "ended_at_utc": ended_utc,
        "wall_seconds": wall_seconds,
        "samples_per_second": 600 / wall_seconds,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
        "gpu_after_model_load": gpu_loaded,
        "labels_opened_during_execution": False,
        "final_access_count": 0,
    }
    manifest_path = output / f"{route.lower()}_prediction_manifest.json"
    atomic_json(manifest_path, route_manifest)
    marker = {
        "status": "complete_read_only",
        "route": route,
        "records_sha256": artifact["sha256"],
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "count": 600,
        "completed_at_utc": ended_utc,
        "labels_opened": False,
        "final_access_count": 0,
    }
    atomic_json(marker_path, marker)
    for path in (prediction_path, manifest_path, marker_path):
        path.chmod(0o444)
    _update_runtime(
        output,
        "routes",
        {
            route: {
                "started_at_utc": started_utc,
                "ended_at_utc": ended_utc,
                "wall_seconds": wall_seconds,
                "samples_per_second": 600 / wall_seconds,
                "gpu_peak_allocated_bytes": peak_allocated,
                "gpu_peak_reserved_bytes": peak_reserved,
            }
        },
    )
    return marker


def freeze_combined_predictions(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, output = _load_frozen_run(config_path)
    _require((output / "b0_prediction_complete.json").is_file(), "B0 prediction is incomplete")
    _require((output / "b2_step240_prediction_complete.json").is_file(), "B2 step240 prediction is incomplete")
    combined_path = output / "combined_prediction_manifest.json"
    marker_path = output / "combined_predictions_complete.json"
    _require(not combined_path.exists() and not marker_path.exists(), "combined prediction freeze already exists")
    expected_ids = [str(row["sample_id"]) for row in iter_jsonl(config["confirmation"]["prompt_path"])]
    routes: dict[str, Any] = {}
    for route in ("B0", "B2_step240"):
        prediction = output / f"{route.lower()}_predictions.jsonl"
        manifest = output / f"{route.lower()}_prediction_manifest.json"
        marker = json.loads((output / f"{route.lower()}_prediction_complete.json").read_text(encoding="utf-8"))
        _require(sha256_file(prediction) == marker["records_sha256"], f"{route} frozen prediction SHA drift")
        _require(sha256_file(manifest) == marker["prediction_manifest_sha256"], f"{route} manifest SHA drift")
        _require(prediction.stat().st_mode & 0o222 == 0, f"{route} prediction is not read-only")
        validation = validate_prediction_records(iter_jsonl(prediction), expected_ids=expected_ids, route=route)
        routes[route] = {
            "prediction_path": str(prediction),
            "records_sha256": marker["records_sha256"],
            "prediction_manifest_path": str(manifest),
            "prediction_manifest_sha256": marker["prediction_manifest_sha256"],
            "count": validation["count"],
            "read_only": True,
        }
    payload = {
        "schema_version": 1,
        "artifact_kind": "p10_combined_prediction_manifest",
        "run_id": config["run_id"],
        "route_order": ["B0", "B2_step240"],
        "confirmation_manifest_sha256": config["confirmation"]["manifest_sha256"],
        "selected_sample_ids_sha256": config["confirmation"]["selected_sample_ids_sha256"],
        "routes": routes,
        "combined_before_label_access": True,
        "labels_opened": False,
        "final_access_count": 0,
        "frozen_at_utc": utc_now(),
    }
    atomic_json(combined_path, payload)
    marker = {
        "status": "complete_read_only",
        "combined_prediction_manifest_sha256": sha256_file(combined_path),
        "routes": ["B0", "B2_step240"],
        "count_per_route": 600,
        "completed_at_utc": utc_now(),
        "labels_opened": False,
        "final_access_count": 0,
    }
    atomic_json(marker_path, marker)
    combined_path.chmod(0o444)
    marker_path.chmod(0o444)
    return marker


def _read_labels_once(path: Path) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if raw.strip():
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise P10ConfirmationError(f"invalid confirmation label JSONL line {line_number}") from error
                _require(isinstance(value, dict), "confirmation label row is not an object")
                rows.append(value)
    return rows, digest.hexdigest()


def _subject_breakdown(b0: Sequence[Mapping[str, Any]], b2: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = {str(row["sample_id"]): row for row in b0}
    second = {str(row["sample_id"]): row for row in b2}
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id, row in first.items():
        grouped[str(row.get("subject") or "unknown")].append(sample_id)
    report = {}
    for subject, ids in sorted(grouped.items()):
        b0_correct = sum(bool(first[item]["correct"]) for item in ids)
        b2_correct = sum(bool(second[item]["correct"]) for item in ids)
        improved = sum(not bool(first[item]["correct"]) and bool(second[item]["correct"]) for item in ids)
        regressed = sum(bool(first[item]["correct"]) and not bool(second[item]["correct"]) for item in ids)
        report[subject] = {
            "count": len(ids),
            "b0_correct": b0_correct,
            "b2_step240_correct": b2_correct,
            "delta_questions": b2_correct - b0_correct,
            "improved": improved,
            "regressed": regressed,
            "unchanged": len(ids) - improved - regressed,
            "diagnostic_only": True,
        }
    return report


def run_label_join(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    from src.eval.paired_stats import paired_comparison, score_label_free_predictions

    config = load_p10_config(config_path)
    output = run_root(config)
    _require(output.is_dir(), "P10 frozen run is missing")
    combined_marker_path = output / "combined_predictions_complete.json"
    _require(combined_marker_path.is_file(), "all predictions must freeze before label access")
    _require(not (output / "label_access_intent.json").exists(), "P10 label access already attempted")
    combined_marker = json.loads(combined_marker_path.read_text(encoding="utf-8"))
    combined_path = output / "combined_prediction_manifest.json"
    combined_sha = sha256_file(combined_path)
    _require(combined_sha == combined_marker["combined_prediction_manifest_sha256"], "combined prediction SHA drift")
    expected_ids = [str(row["sample_id"]) for row in iter_jsonl(config["confirmation"]["prompt_path"])]
    predictions: dict[str, list[dict[str, Any]]] = {}
    for route in ("B0", "B2_step240"):
        path = output / f"{route.lower()}_predictions.jsonl"
        rows = list(iter_jsonl(path))
        validate_prediction_records(rows, expected_ids=expected_ids, route=route)
        predictions[route] = rows
    begin_label_access(output, combined_prediction_sha256=combined_sha)
    label_path = Path(config["confirmation"]["label_path"])
    labels, label_sha = _read_labels_once(label_path)
    _require(label_sha == config["confirmation"]["label_sha256"], "confirmation label SHA mismatch after access")
    _require(len(labels) == 600, "confirmation label count mismatch")
    label_ids = [str(row.get("sample_id") or "") for row in labels]
    _require(len(set(label_ids)) == 600 and set(label_ids) == set(expected_ids), "confirmation label ID set mismatch")
    _require(all("final" not in str(row.get("target_role") or "") for row in labels), "label artifact contains a final role")
    scored_b0 = score_label_free_predictions(predictions["B0"], labels)
    scored_b2 = score_label_free_predictions(predictions["B2_step240"], labels)
    paired = paired_comparison(
        scored_b0,
        scored_b2,
        seed=int(config["statistics"]["bootstrap_seed"]),
        bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
    )
    b0_correct = sum(bool(row["correct"]) for row in scored_b0)
    b2_correct = sum(bool(row["correct"]) for row in scored_b2)
    delta_questions = b2_correct - b0_correct
    ci = [float(value) for value in paired["bootstrap_95_ci"]]
    p_value = float(paired["mcnemar"]["exact_two_sided_p"])
    status = classify_confirmation(
        delta_questions=delta_questions,
        bootstrap_ci=ci,
        mcnemar_exact_p=p_value,
        integrity_passed=True,
    )
    statistics = {
        "schema_version": 1,
        "artifact_kind": "p10_b2_step240_paired_statistics",
        "status": status,
        "total": 600,
        "B0": {"correct": b0_correct, "accuracy": b0_correct / 600},
        "B2_step240": {"correct": b2_correct, "accuracy": b2_correct / 600},
        "delta_questions": delta_questions,
        "delta_percentage_points": 100.0 * delta_questions / 600,
        "improved": paired["improved"],
        "regressed": paired["regressed"],
        "unchanged": paired["unchanged"],
        "discordant_pairs": paired["mcnemar"]["discordant"],
        "paired_bootstrap_95_ci": ci,
        "paired_bootstrap_95_ci_percentage_points": [100.0 * value for value in ci],
        "bootstrap_seed": paired["seed"],
        "bootstrap_samples": paired["bootstrap_samples"],
        "exact_mcnemar_two_sided_p": p_value,
        "mcnemar": paired["mcnemar"],
        "subject_breakdown": _subject_breakdown(scored_b0, scored_b2),
        "subject_breakdown_diagnostic_only": True,
        "b1_included": False,
    }
    atomic_json(output / "statistics.json", statistics)
    label_join = {
        "schema_version": 1,
        "artifact_kind": "p10_one_time_label_join",
        "combined_prediction_manifest_sha256": combined_sha,
        "confirmation_label_sha256": label_sha,
        "label_count": 600,
        "joined_routes": ["B0", "B2_step240"],
        "prediction_first": True,
        "labels_opened_after_all_predictions_frozen": True,
        "models_released_before_label_access": True,
        "p10_confirmation_label_join_count": 1,
        "project_confirmation_label_join_count_after_p10": 2,
        "final_access_count": 0,
        "completed_at_utc": utc_now(),
        "statistics_sha256": sha256_file(output / "statistics.json"),
    }
    atomic_json(output / "label_join.json", label_join)
    decision = {
        "schema_version": 1,
        "artifact_kind": "p10_b2_step240_final_decision",
        "status": status,
        "only_candidate_step": 240,
        "only_comparison": "B2_step240_vs_B0_paired_medical_confirmation_600",
        "preregistered_rule_applied_without_change": True,
        "integrity_gate_passed": True,
        "statistics_sha256": label_join["statistics_sha256"],
        "label_join_sha256": sha256_file(output / "label_join.json"),
        "no_training": True,
        "no_checkpoint_selection": True,
        "checkpoint_unchanged": True,
        "no_post_result_tuning": True,
        "final_access_count": 0,
        "decided_at_utc": utc_now(),
    }
    atomic_json(output / "final_decision.json", decision)
    access = {
        "schema_version": 1,
        "artifact_kind": "p10_confirmation_access_audit",
        "observed_at_utc": utc_now(),
        "historical_teacher_confirmation_label_joins": 1,
        "historical_b2_confirmation_label_joins_before_p10": 0,
        "p10_confirmation_label_joins": 1,
        "b2_confirmation_label_joins_after_p10": 1,
        "project_confirmation_label_joins_total": 2,
        "final_access_count": 0,
        "prediction_first": True,
        "labels_opened_in_p10": True,
        "combined_prediction_manifest_sha256": combined_sha,
        "label_access_intent_sha256": sha256_file(output / "label_access_intent.json"),
        "label_join_sha256": sha256_file(output / "label_join.json"),
    }
    atomic_json(output / "confirmation_access_audit.json", access)
    runtime_path = output / "runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(runtime["session_started_at_utc"])
    total_seconds = (datetime.now(timezone.utc) - start).total_seconds()
    runtime.update(
        {
            "session_ended_at_utc": utc_now(),
            "total_wall_seconds_through_label_join": total_seconds,
            "derived_cost_cny": total_seconds / 3600 * float(config["runtime"]["live_price_cny_per_instance_hour"]),
            "platform_actual_cost_cny": None,
            "persistent_disk_after_label_join": _disk_snapshot(Path("artifacts")),
            "label_join_count": 1,
            "final_access_count": 0,
        }
    )
    atomic_json(runtime_path, runtime)
    return decision


def record_gpu_cleanup(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_p10_config(config_path)
    output = run_root(config)
    _release_gpu(str(config["runtime"]["prediction_device"]))
    gpu = _gpu_snapshot()
    clean = all(row["memory_used_mib"] <= 8 for row in gpu) and all(row["compute_process_query"] is None for row in gpu)
    runtime_path = output / "runtime_metrics.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(runtime["session_started_at_utc"])
    total_seconds = (datetime.now(timezone.utc) - start).total_seconds()
    derived_cost = total_seconds / 3600 * float(config["runtime"]["live_price_cny_per_instance_hour"])
    final_disk = _disk_snapshot(Path("artifacts"))
    report = {
        "schema_version": 1,
        "artifact_kind": "p10_gpu_cleanup",
        "status": "PASS" if clean else "FAIL",
        "observed_at_utc": utc_now(),
        "gpus": gpu,
        "p10_compute_pids": [],
        "models_released": ["B0", "B2_step240"],
        "b1_loaded": False,
        "training_process_started": False,
        "final_process_started": False,
        "safe_to_close_instance": clean,
        "total_wall_seconds_through_gpu_cleanup": total_seconds,
        "derived_cost_cny": derived_cost,
        "platform_actual_cost_cny": None,
        "persistent_disk_final": final_disk,
    }
    atomic_json(output / "gpu_cleanup.json", report)
    runtime.update(
        {
            "gpu_cleanup_at_utc": report["observed_at_utc"],
            "gpu_final": gpu,
            "persistent_disk_final": final_disk,
            "total_wall_seconds_through_gpu_cleanup": total_seconds,
            "derived_cost_cny": derived_cost,
            "platform_actual_cost_cny": None,
        }
    )
    atomic_json(runtime_path, runtime)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled P10 B2 step240 confirmation")
    parser.add_argument(
        "command",
        choices=("preflight", "freeze", "smoke", "predict", "combine", "join", "cleanup"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--route", choices=("B0", "B2_step240"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_cpu_preflight(args.config)
    elif args.command == "freeze":
        result = freeze_run(args.config)
    elif args.command == "smoke":
        result = run_micro_smoke(args.config)
    elif args.command == "predict":
        _require(args.route is not None, "predict requires --route")
        result = run_prediction(args.route, args.config)
    elif args.command == "combine":
        result = freeze_combined_predictions(args.config)
    elif args.command == "join":
        result = run_label_join(args.config)
    else:
        result = record_gpu_cleanup(args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


__all__ = [
    "P10ConfirmationError",
    "atomic_json",
    "atomic_jsonl",
    "audit_pool_overlaps",
    "begin_label_access",
    "classify_confirmation",
    "iter_jsonl",
    "load_p10_config",
    "prompt_execution_row",
    "sha256_file",
    "utc_now",
    "validate_confirmation_prompt_binding",
    "validate_prediction_records",
]


if __name__ == "__main__":
    raise SystemExit(main())
