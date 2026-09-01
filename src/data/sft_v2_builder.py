"""Deterministic Medical-O1 + CMB-train bridge construction for SFT-v2.

The formal command streams the 148k-row CMB JSON array.  This core selection
function only retains eligible canonical candidates and is input-order
independent, which makes it suitable for both the builder and small fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.schema import content_hash_v2, normalize_options_v2, normalize_question_v2, stable_sample_id_v2
from src.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl


DEFAULT_CMB_REVISION = "935fbc09edf1303d89872b21265ff597f426ac0d"
DEFAULT_CMB_RAW_SHA256 = "6539cb96d81e0672d18b0ea255b2f040dda2ec90095916d7620969d0ce8eb01d"


def _canonical_raw_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_options(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    raw = row.get("option") or row.get("options")
    if not isinstance(raw, Mapping):
        return [], []
    labels = sorted(str(key).strip().upper() for key in raw)
    if labels != [chr(ord("A") + index) for index in range(len(labels))]:
        return [], []
    options = [str(raw[label]).strip() for label in labels]
    if any(not value for value in options):
        return [], []
    return labels, options


def _cmb_record(
    row: Mapping[str, Any],
    *,
    source_revision: str,
    raw_file_sha256: str,
    source_license: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if str(row.get("upstream_split") or "train").strip().casefold() != "train":
        return None, "non_train_split"
    question = str(row.get("question") or "").strip()
    if not question:
        return None, "missing_question"
    labels, options = _ordered_options(row)
    if not 2 <= len(options) <= 5:
        return None, "invalid_options"
    answer_label = str(row.get("answer") or "").strip().upper()
    if answer_label not in labels:
        return None, "invalid_answer"
    answer_index = labels.index(answer_label)
    content_hash = content_hash_v2(question, options)
    raw_identity = _canonical_raw_hash(row)
    subject = str(row.get("exam_subject") or "unknown").strip() or "unknown"
    category_values = [
        str(row.get(key) or "").strip()
        for key in ("exam_type", "exam_class", "exam_subject", "question_type")
    ]
    category = "/".join(value for value in category_values if value) or "unknown"
    return {
        "sample_id": stable_sample_id_v2(
            source="cmb_exam",
            source_revision=source_revision,
            upstream_split="train",
            upstream_id=raw_identity,
            subject=subject,
        ),
        "source": "cmb_exam",
        "source_revision": source_revision,
        "source_license": source_license,
        "license": source_license,
        "upstream_split": "train",
        "target_role": "medical_sft_train",
        "domain": "medical",
        "subject": subject,
        "category": category,
        "question": question,
        "normalized_question": normalize_question_v2(question),
        "options": options,
        "normalized_options": list(normalize_options_v2(options)),
        "answer": options[answer_index],
        "answer_idx": answer_label,
        "content_hash": content_hash,
        "group_id": content_hash,
        "raw_file_sha256": raw_file_sha256,
        "data_protocol_version": "ca-opd-data-v2",
        "schema_version": 2,
        "sft_v2_kind": "cmb_mcq_bridge",
        "quality_flags": ["license_verified_by_p1_6", "cmb_opd_disjoint"],
    }, None


def _priority(seed: int, record: Mapping[str, Any]) -> str:
    identity = "\0".join(
        (
            str(seed),
            str(record.get("category") or "unknown"),
            str(record["sample_id"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _stratified_select(records: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        stratum = str(record.get("category") or "unknown")
        by_stratum.setdefault(stratum, []).append(record)
    for values in by_stratum.values():
        values.sort(key=lambda row: (_priority(seed, row), str(row["sample_id"])))
    selected: list[dict[str, Any]] = []
    strata = sorted(by_stratum)
    offset = 0
    while len(selected) < target:
        progressed = False
        for stratum in strata:
            values = by_stratum[stratum]
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        offset += 1
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def build_sft_v2_records(
    *,
    medical_rows: Iterable[Mapping[str, Any]],
    cmb_rows: Iterable[Mapping[str, Any]],
    protected_content_hashes: set[str],
    protected_group_ids: set[str],
    target_cmb_count: int,
    seed: int,
    cmb_source_revision: str = DEFAULT_CMB_REVISION,
    cmb_raw_file_sha256: str = DEFAULT_CMB_RAW_SHA256,
    cmb_source_license: str = "Apache-2.0",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the deterministic mixture without altering either input source."""

    if target_cmb_count < 0:
        raise ValueError("target_cmb_count must be non-negative")
    medical: list[dict[str, Any]] = []
    training_hashes: set[str] = set()
    training_groups: set[str] = set()
    for source_row in medical_rows:
        row = dict(source_row)
        if row.get("target_role") != "medical_sft_train":
            raise ValueError("Medical-O1 source row is not formal medical_sft_train")
        if not row.get("answer") or not row.get("reasoning"):
            raise ValueError("Medical-O1 source row lacks Response or Complex_CoT")
        if row["content_hash"] in training_hashes:
            raise ValueError("duplicate Medical-O1 content hash")
        row["sft_v2_kind"] = "medical_o1_answer_first"
        medical.append(row)
        training_hashes.add(str(row["content_hash"]))
        training_groups.add(str(row["group_id"]))

    dropped: Counter[str] = Counter()
    accepted_by_content: dict[str, dict[str, Any]] = {}
    for raw_row in cmb_rows:
        record, reason = _cmb_record(
            raw_row,
            source_revision=cmb_source_revision,
            raw_file_sha256=cmb_raw_file_sha256,
            source_license=cmb_source_license,
        )
        if record is None:
            dropped[str(reason)] += 1
            continue
        if (
            record["content_hash"] in protected_content_hashes
            or record["group_id"] in protected_group_ids
            or record["content_hash"] in training_hashes
            or record["group_id"] in training_groups
        ):
            dropped["protected_overlap"] += 1
            continue
        previous = accepted_by_content.get(str(record["content_hash"]))
        if previous is not None:
            dropped["duplicate_content"] += 1
            if str(record["sample_id"]) < str(previous["sample_id"]):
                accepted_by_content[str(record["content_hash"])] = record
            continue
        accepted_by_content[str(record["content_hash"])] = record

    selected = _stratified_select(
        list(accepted_by_content.values()),
        min(target_cmb_count, len(accepted_by_content)),
        seed,
    )
    report = {
        "seed": seed,
        "medical_o1_count": len(medical),
        "cmb_target_count": target_cmb_count,
        "cmb_eligible_count": len(accepted_by_content),
        "cmb_selected_count": len(selected),
        "dropped": dict(sorted(dropped.items())),
        "selection": "stable_hash_round_robin_by_exam_category",
        "source_revision": cmb_source_revision,
        "source_license": cmb_source_license,
        "upstream_split": "train",
    }
    return sorted(medical, key=lambda row: str(row["sample_id"])) + selected, report


def _load_protected_identities(
    jsonl_paths: Iterable[str | Path], hash_manifest_paths: Iterable[str | Path]
) -> tuple[set[str], set[str], dict[str, int]]:
    """Read prompt/train identities and hash-only final manifests, never labels."""

    content_hashes: set[str] = set()
    group_ids: set[str] = set()
    counts: dict[str, int] = {}
    for value in jsonl_paths:
        path = Path(value)
        count = 0
        for row in iter_jsonl(path):
            if row.get("content_hash"):
                content_hashes.add(str(row["content_hash"]))
            if row.get("group_id"):
                group_ids.add(str(row["group_id"]))
            count += 1
        counts[str(path)] = count
    for value in hash_manifest_paths:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if items is None:
            items = payload.get("records")
        if not isinstance(items, list):
            raise ValueError(f"hash manifest lacks items: {path}")
        for item in items:
            if not isinstance(item, Mapping) or not item.get("content_hash"):
                raise ValueError(f"hash manifest has invalid item: {path}")
            content_hashes.add(str(item["content_hash"]))
        counts[str(path)] = len(items)
    return content_hashes, group_ids, counts


def build_formal_sft_v2(config_path: str | Path) -> dict[str, Any]:
    """Build the persistent weighted mixture and its small versioned manifest."""

    import yaml
    from transformers import AutoTokenizer

    from src.data.readers_v2 import iter_json_array
    from src.sft.weighted import SupervisionWeights, render_sft_v2_row

    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    required = {
        "run_id",
        "seed",
        "medical_records_path",
        "medical_manifest_path",
        "cmb_raw_path",
        "cmb_raw_sha256",
        "cmb_source_revision",
        "cmb_source_license",
        "target_cmb_count",
        "protected_jsonl_paths",
        "protected_hash_manifest_paths",
        "tokenizer_path",
        "tokenizer_revision",
        "max_seq_length",
        "system_prompt",
        "weights",
        "output_records_path",
        "output_manifest_path",
        "output_report_path",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError(
            f"SFT-v2 data config keys mismatch: missing={sorted(required - set(config or {}))}, "
            f"extra={sorted(set(config or {}) - required)}"
        )
    output_records = Path(str(config["output_records_path"]))
    output_manifest = Path(str(config["output_manifest_path"]))
    output_report = Path(str(config["output_report_path"]))
    occupied = [path for path in (output_records, output_manifest, output_report) if path.exists()]
    if occupied:
        raise FileExistsError(
            "SFT-v2 artifacts are immutable once written: "
            + ", ".join(str(path) for path in occupied)
        )
    medical_path = Path(str(config["medical_records_path"]))
    medical_manifest_path = Path(str(config["medical_manifest_path"]))
    cmb_path = Path(str(config["cmb_raw_path"]))
    for path in (medical_path, medical_manifest_path, cmb_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(cmb_path) != str(config["cmb_raw_sha256"]):
        raise ValueError("CMB train raw SHA mismatch")
    medical_manifest = json.loads(medical_manifest_path.read_text(encoding="utf-8"))
    protected_content, protected_groups, protected_counts = _load_protected_identities(
        config["protected_jsonl_paths"], config["protected_hash_manifest_paths"]
    )
    records, selection_report = build_sft_v2_records(
        medical_rows=iter_jsonl(medical_path),
        cmb_rows=iter_json_array(cmb_path),
        protected_content_hashes=protected_content,
        protected_group_ids=protected_groups,
        target_cmb_count=int(config["target_cmb_count"]),
        seed=int(config["seed"]),
        cmb_source_revision=str(config["cmb_source_revision"]),
        cmb_raw_file_sha256=str(config["cmb_raw_sha256"]),
        cmb_source_license=str(config["cmb_source_license"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["tokenizer_path"]),
        revision=str(config["tokenizer_revision"]),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    weights_cfg = config["weights"]
    if not isinstance(weights_cfg, Mapping) or set(weights_cfg) != {"answer", "reasoning", "eos"}:
        raise ValueError("weights must contain exactly answer/reasoning/eos")
    weights = SupervisionWeights(
        answer=float(weights_cfg["answer"]),
        reasoning=float(weights_cfg["reasoning"]),
        eos=float(weights_cfg["eos"]),
    )
    kept: list[dict[str, Any]] = []
    dropped_too_long: Counter[str] = Counter()
    token_counts = {"prompt": 0, "answer": 0, "reasoning": 0, "eos": 0}
    weighted_contribution = {"answer": 0.0, "reasoning": 0.0, "eos": 0.0}
    for record in records:
        example = render_sft_v2_row(
            record,
            tokenizer=tokenizer,
            weights=weights,
            max_seq_length=int(config["max_seq_length"]),
            system_prompt=str(config["system_prompt"]),
        )
        if example is None:
            dropped_too_long[str(record["sft_v2_kind"])] += 1
            continue
        row = dict(record)
        row["token_count_prompt"] = example.prompt_length
        row["token_count_response"] = len(example.input_ids) - example.prompt_length
        kept.append(row)
        token_counts["prompt"] += example.prompt_length
        for name in ("answer", "reasoning", "eos"):
            token_counts[name] += example.segment_token_counts[name]
            weighted_contribution[name] += example.segment_weighted_contribution[name]

    written = write_jsonl(output_records, kept)
    records_sha = file_sha256(output_records)
    records_bytes = output_records.stat().st_size
    counts_by_kind = Counter(str(row["sft_v2_kind"]) for row in kept)
    manifest = {
        "schema_version": 2,
        "data_protocol_version": "ca-opd-data-v2",
        "source_policy_version": "ca-opd-source-policy-v1",
        "build_version": "p3-4-medical-sft-v2-answer-first-bridge-v1",
        "run_id": str(config["run_id"]),
        "seed": int(config["seed"]),
        "final_authorized": False,
        "primary_final_frozen": True,
        "supervision_version": "answer_first_weighted_v2",
        "enable_thinking": False,
        "tail_truncation_allowed": False,
        "max_seq_length": int(config["max_seq_length"]),
        "weights": dict(weights_cfg),
        "original_medical_sft_manifest_sha256": file_sha256(medical_manifest_path),
        "original_medical_sft_records_sha256": file_sha256(medical_path),
        "cmb_train_raw_sha256": str(config["cmb_raw_sha256"]),
        "cmb_source_revision": str(config["cmb_source_revision"]),
        "cmb_source_license": str(config["cmb_source_license"]),
        "protected_identity_sources": protected_counts,
        "roles": {
            "medical_sft_train": {
                "actual_count": written,
                "count": written,
                "files": [
                    {
                        "path": str(output_records),
                        "count": written,
                        "bytes": records_bytes,
                        "sha256": records_sha,
                        "supervision_fields": written * 2,
                    }
                ],
                "upstream_splits": ["train"],
            }
        },
        "sources": {
            "medical_o1": medical_manifest.get("sources", {}).get("medical_o1"),
            "cmb": medical_manifest.get("sources", {}).get("cmb"),
        },
    }
    for key in (
        "build_mode",
        "build_status",
        "synthetic_fixture",
        "conflict_policy_version",
        "manual_audit_pending",
        "human_reviewed",
        "manual_audit_waived_by_user",
        "waiver_reason",
        "cross_role_candidates_conservatively_resolved",
        "unresolved_cross_role_candidates",
    ):
        if key not in medical_manifest:
            raise ValueError(f"original formal SFT manifest lacks required gate: {key}")
        manifest[key] = medical_manifest[key]
    write_json(output_manifest, manifest)
    report = {
        "status": "formal_sft_v2_frozen",
        "records_path": str(output_records),
        "records_sha256": records_sha,
        "manifest_path": str(output_manifest),
        "manifest_sha256": file_sha256(output_manifest),
        "actual_count": written,
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "selection": selection_report,
        "dropped_too_long": dict(sorted(dropped_too_long.items())),
        "token_counts": token_counts,
        "weighted_contribution": weighted_contribution,
        "weighted_contribution_total": sum(weighted_contribution.values()),
        "final_authorized": False,
        "final_labels_opened": False,
        "tail_truncation_used": False,
    }
    write_json(output_report, report)
    return report


def main() -> int:  # pragma: no cover - exercised through the CLI integration gate
    import argparse

    parser = argparse.ArgumentParser(description="Build the frozen P3.4 Medical SFT-v2 mixture")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_formal_sft_v2(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
