"""Deterministic, bounded-memory MCQ-dominant SFT-v3 construction."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.sft_v2_builder import (
    DEFAULT_CMB_RAW_SHA256,
    DEFAULT_CMB_REVISION,
    _cmb_record,
    _load_protected_identities,
)
from src.sft.v3 import SFTV3Kind, render_sft_v3_row, validate_sft_v3_schedule
from src.utils.io import file_sha256, iter_jsonl, write_json, write_jsonl


def _priority(seed: int, namespace: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{sample_id}".encode("utf-8")).hexdigest()


def _medical_record(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("target_role") != "medical_sft_train":
        raise ValueError("Medical-O1 SFT-v3 source role drift")
    answer = str(row.get("answer") or "").strip()
    if not answer:
        raise ValueError("Medical-O1 SFT-v3 source lacks Response")
    result = {
        key: value
        for key, value in dict(row).items()
        if str(key).casefold() not in {"reasoning", "complex_cot"}
    }
    result["answer"] = answer
    result.pop("sft_v2_kind", None)
    result["sft_v3_kind"] = SFTV3Kind.MEDICAL_O1.value
    result["quality_flags"] = sorted(
        set(str(value) for value in result.get("quality_flags") or [])
        | {"response_only_supervision", "sft_v3_mcq_dominant"}
    )
    return result


def _v3_cmb_record(
    row: Mapping[str, Any],
    *,
    source_revision: str,
    raw_file_sha256: str,
    source_license: str,
) -> tuple[dict[str, Any] | None, str | None]:
    record, reason = _cmb_record(
        row,
        source_revision=source_revision,
        raw_file_sha256=raw_file_sha256,
        source_license=source_license,
    )
    if record is None:
        return None, reason
    record.pop("answer", None)
    record.pop("sft_v2_kind", None)
    record["sft_v3_kind"] = SFTV3Kind.CMB.value
    record["quality_flags"] = sorted(
        set(str(value) for value in record.get("quality_flags") or [])
        | {"single_letter_supervision", "sft_v3_mcq_dominant"}
    )
    return record, None


def _stratum(record: Mapping[str, Any]) -> str:
    return "\0".join(
        (
            str(record.get("category") or "unknown"),
            str(record.get("subject") or "unknown"),
            str(record.get("answer_idx") or "unknown"),
        )
    )


def _round_robin(records: list[dict[str, Any]], *, target: int, seed: int) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_stratum.setdefault(_stratum(record), []).append(record)
    for stratum, values in by_stratum.items():
        values.sort(key=lambda row: (_priority(seed, stratum, str(row["sample_id"])), row["sample_id"]))
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < target:
        progressed = False
        for stratum in sorted(by_stratum):
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


def build_sft_v3_records(
    *,
    medical_rows: Iterable[Mapping[str, Any]],
    cmb_rows: Iterable[Mapping[str, Any]],
    protected_content_hashes: set[str],
    protected_group_ids: set[str],
    target_medical_count: int,
    target_cmb_count: int,
    seed: int,
    cmb_source_revision: str = DEFAULT_CMB_REVISION,
    cmb_raw_file_sha256: str = DEFAULT_CMB_RAW_SHA256,
    cmb_source_license: str = "Apache-2.0",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Small-fixture implementation of the same frozen deterministic selection."""

    if target_medical_count < 0 or target_cmb_count < 0:
        raise ValueError("SFT-v3 source targets must be non-negative")
    medical_candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_groups: set[str] = set()
    dropped: Counter[str] = Counter()
    for source in medical_rows:
        record = _medical_record(source)
        content_hash = str(record.get("content_hash") or "")
        group_id = str(record.get("group_id") or "")
        if not content_hash or not group_id:
            raise ValueError("Medical-O1 SFT-v3 identity is incomplete")
        if content_hash in protected_content_hashes or group_id in protected_group_ids:
            dropped["protected_overlap"] += 1
            continue
        if content_hash in seen_hashes or group_id in seen_groups:
            dropped["duplicate_identity"] += 1
            continue
        seen_hashes.add(content_hash)
        seen_groups.add(group_id)
        medical_candidates.append(record)
    medical_candidates.sort(
        key=lambda row: (_priority(seed, "medical_o1", str(row["sample_id"])), row["sample_id"])
    )
    medical = sorted(medical_candidates[:target_medical_count], key=lambda row: row["sample_id"])

    cmb_by_content: dict[str, dict[str, Any]] = {}
    for source in cmb_rows:
        record, reason = _v3_cmb_record(
            source,
            source_revision=cmb_source_revision,
            raw_file_sha256=cmb_raw_file_sha256,
            source_license=cmb_source_license,
        )
        if record is None:
            dropped[str(reason)] += 1
            continue
        content_hash = str(record["content_hash"])
        group_id = str(record["group_id"])
        if (
            content_hash in protected_content_hashes
            or group_id in protected_group_ids
            or content_hash in seen_hashes
            or group_id in seen_groups
        ):
            dropped["protected_overlap"] += 1
            continue
        previous = cmb_by_content.get(content_hash)
        if previous is not None:
            dropped["duplicate_content"] += 1
            if str(record["sample_id"]) < str(previous["sample_id"]):
                cmb_by_content[content_hash] = record
        else:
            cmb_by_content[content_hash] = record
    cmb = _round_robin(list(cmb_by_content.values()), target=target_cmb_count, seed=seed)
    report = {
        "seed": seed,
        "medical_o1_target_count": target_medical_count,
        "medical_o1_eligible_count": len(medical_candidates),
        "medical_o1_selected_count": len(medical),
        "medical_o1_shortfall": max(0, target_medical_count - len(medical)),
        "cmb_target_count": target_cmb_count,
        "cmb_eligible_count": len(cmb_by_content),
        "cmb_selected_count": len(cmb),
        "cmb_shortfall": max(0, target_cmb_count - len(cmb)),
        "selection": "stable_hash_round_robin_by_category_subject_answer_label",
        "dropped": dict(sorted(dropped.items())),
        "source_revision": cmb_source_revision,
        "source_license": cmb_source_license,
        "upstream_split": "train",
        "controller_performance_used_for_selection": False,
    }
    return medical + cmb, report


class _DiskCandidateIndex:
    """SQLite-backed CMB candidate deduplication under the 2GB cgroup limit."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            "CREATE TABLE candidates (content_hash TEXT PRIMARY KEY, sample_id TEXT NOT NULL, "
            "stratum TEXT NOT NULL, priority TEXT NOT NULL, row_json TEXT NOT NULL)"
        )

    def add(self, record: Mapping[str, Any], *, seed: int) -> bool:
        content_hash = str(record["content_hash"])
        sample_id = str(record["sample_id"])
        stratum = _stratum(record)
        priority = _priority(seed, stratum, sample_id)
        payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        previous = self.connection.execute(
            "SELECT sample_id FROM candidates WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if previous is None:
            self.connection.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?)",
                (content_hash, sample_id, stratum, priority, payload),
            )
            return True
        if sample_id < str(previous[0]):
            self.connection.execute(
                "UPDATE candidates SET sample_id=?,stratum=?,priority=?,row_json=? WHERE content_hash=?",
                (sample_id, stratum, priority, payload, content_hash),
            )
        return False

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def select(self, target: int) -> list[dict[str, Any]]:
        query = """
            SELECT row_json FROM (
              SELECT row_json, stratum, sample_id,
                     ROW_NUMBER() OVER (PARTITION BY stratum ORDER BY priority, sample_id) AS rn
              FROM candidates
            ) ORDER BY rn, stratum, sample_id LIMIT ?
        """
        rows = [json.loads(value[0]) for value in self.connection.execute(query, (target,))]
        return sorted(rows, key=lambda row: str(row["sample_id"]))

    def close(self) -> None:
        self.connection.close()


def _manifest_items_sha(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["sample_id"])):
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["content_hash"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_formal_sft_v3(config_path: str | Path) -> dict[str, Any]:
    """Build formal SFT-v3 records using a disk-backed raw CMB scan."""

    import yaml
    from transformers import AutoTokenizer

    from src.data.readers_v2 import iter_json_array

    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    required = {
        "run_id", "seed", "medical_records_path", "medical_manifest_path", "cmb_raw_path",
        "cmb_raw_sha256", "cmb_source_revision", "cmb_source_license", "target_medical_count",
        "target_cmb_count", "protected_jsonl_paths", "protected_hash_manifest_paths",
        "tokenizer_path", "tokenizer_revision", "max_seq_length", "system_prompt",
        "output_records_path", "output_manifest_path", "output_report_path",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError("SFT-v3 data config keys mismatch")
    outputs = [Path(str(config[key])) for key in (
        "output_records_path", "output_manifest_path", "output_report_path"
    )]
    occupied = [path for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError("SFT-v3 artifacts are immutable: " + ", ".join(map(str, occupied)))
    medical_path = Path(str(config["medical_records_path"]))
    medical_manifest_path = Path(str(config["medical_manifest_path"]))
    cmb_path = Path(str(config["cmb_raw_path"]))
    if file_sha256(cmb_path) != str(config["cmb_raw_sha256"]):
        raise ValueError("CMB train raw SHA mismatch")
    protected_content, protected_groups, protected_counts = _load_protected_identities(
        config["protected_jsonl_paths"], config["protected_hash_manifest_paths"]
    )

    # Medical-O1 is only 8k rows and remains bounded; select by stable hash with no controller signal.
    medical_candidates: list[dict[str, Any]] = []
    for row in iter_jsonl(medical_path):
        record = _medical_record(row)
        if record["content_hash"] in protected_content or record["group_id"] in protected_groups:
            continue
        medical_candidates.append(record)
    medical_candidates.sort(
        key=lambda row: (_priority(int(config["seed"]), "medical_o1", row["sample_id"]), row["sample_id"])
    )
    medical = sorted(
        medical_candidates[: int(config["target_medical_count"])], key=lambda row: row["sample_id"]
    )
    medical_hashes = {str(row["content_hash"]) for row in medical_candidates}
    medical_groups = {str(row["group_id"]) for row in medical_candidates}

    dropped: Counter[str] = Counter()
    temporary = tempfile.NamedTemporaryFile(prefix="p3_6_cmb_", suffix=".sqlite", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    index = _DiskCandidateIndex(temporary_path)
    try:
        for number, source in enumerate(iter_json_array(cmb_path), 1):
            record, reason = _v3_cmb_record(
                source,
                source_revision=str(config["cmb_source_revision"]),
                raw_file_sha256=str(config["cmb_raw_sha256"]),
                source_license=str(config["cmb_source_license"]),
            )
            if record is None:
                dropped[str(reason)] += 1
                continue
            if (
                record["content_hash"] in protected_content
                or record["group_id"] in protected_groups
                or record["content_hash"] in medical_hashes
                or record["group_id"] in medical_groups
            ):
                dropped["protected_overlap"] += 1
                continue
            if not index.add(record, seed=int(config["seed"])):
                dropped["duplicate_content"] += 1
            if number % 250 == 0:
                index.commit()
        index.commit()
        eligible = index.count()
        cmb = index.select(int(config["target_cmb_count"]))
    finally:
        index.close()
        temporary_path.unlink(missing_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["tokenizer_path"]),
        revision=str(config["tokenizer_revision"]),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    candidate_ids: dict[str, set[int]] = {label: set() for label in "ABCDE"}
    for record in medical + cmb:
        example = render_sft_v3_row(
            record,
            tokenizer=tokenizer,
            max_seq_length=int(config["max_seq_length"]),
            system_prompt=str(config["system_prompt"]),
        )
        if example is None:
            dropped[f"too_long_{record['sft_v3_kind']}"] += 1
            continue
        row = dict(record)
        row["token_count_prompt"] = example.prompt_length
        row["token_count_response"] = len(example.input_ids) - example.prompt_length
        row["first_supervised_token_id"] = int(example.input_ids[example.prompt_length])
        if row["sft_v3_kind"] == SFTV3Kind.CMB.value:
            candidate_ids[str(row["answer_idx"])].add(row["first_supervised_token_id"])
        records.append(row)
        by_kind[row["sft_v3_kind"]] += 1
        token_counts["prompt"] += example.prompt_length
        token_counts["answer"] += example.segment_token_counts["answer"]
        token_counts["reasoning"] += example.segment_token_counts["reasoning"]
        token_counts["eos"] += example.segment_token_counts["eos"]

    counts = {
        "cmb": by_kind[SFTV3Kind.CMB.value],
        "medical_o1": by_kind[SFTV3Kind.MEDICAL_O1.value],
    }
    if counts != {
        "cmb": int(config["target_cmb_count"]),
        "medical_o1": int(config["target_medical_count"]),
    }:
        raise RuntimeError(f"SFT-v3 source shortfall after rendering: {counts}")
    schedule = validate_sft_v3_schedule(total_steps=600, checkpoints=[150, 300, 450, 600])
    if schedule["global_exposures"] != counts:
        raise RuntimeError("SFT-v3 source counts do not equal one frozen schedule exposure")
    frozen_candidate_ids = {}
    for label, values in candidate_ids.items():
        if len(values) != 1:
            raise RuntimeError(f"SFT-v3 contextual candidate token drift for {label}")
        frozen_candidate_ids[label] = next(iter(values))

    output_records, output_manifest, output_report = outputs
    written = write_jsonl(output_records, sorted(records, key=lambda row: row["sample_id"]))
    records_sha = file_sha256(output_records)
    original_manifest = json.loads(medical_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 2,
        "data_protocol_version": "ca-opd-data-v2",
        "source_policy_version": "ca-opd-source-policy-v1",
        "build_version": "p3-6-sft-v3-mcq-dominant-v1",
        "run_id": str(config["run_id"]),
        "seed": int(config["seed"]),
        "target_role": "medical_sft_train",
        "final_authorized": False,
        "primary_final_frozen": True,
        "confirmation_authorized": False,
        "supervision_version": "mcq_dominant_task_balanced_v3",
        "enable_thinking": False,
        "tail_truncation_allowed": False,
        "max_seq_length": int(config["max_seq_length"]),
        "source_counts": counts,
        "source_licenses": {"cmb": str(config["cmb_source_license"]), "medical_o1": "Apache-2.0"},
        "upstream_splits": {"cmb": ["train"], "medical_o1": ["train"]},
        "task_schedule": schedule,
        "loss_weights": {"target": 1.0, "eos": 1.0, "prompt_padding": 0.0},
        "candidate_token_ids": frozen_candidate_ids,
        "selected_identity_sha256": _manifest_items_sha(records),
        "protected_identity_sources": protected_counts,
        "cmb_train_raw_sha256": str(config["cmb_raw_sha256"]),
        "cmb_source_revision": str(config["cmb_source_revision"]),
        "original_medical_sft_manifest_sha256": file_sha256(medical_manifest_path),
        "original_medical_sft_records_sha256": file_sha256(medical_path),
        "roles": {
            "medical_sft_train": {
                "count": written,
                "actual_count": written,
                "files": [{
                    "path": str(output_records), "count": written,
                    "bytes": output_records.stat().st_size, "sha256": records_sha,
                    "supervision_fields": written,
                }],
                "upstream_splits": ["train"],
            }
        },
    }
    for key in (
        "build_mode", "build_status", "synthetic_fixture", "conflict_policy_version",
        "manual_audit_pending", "human_reviewed", "manual_audit_waived_by_user",
        "waiver_reason", "cross_role_candidates_conservatively_resolved",
        "unresolved_cross_role_candidates",
    ):
        manifest[key] = original_manifest[key]
    write_json(output_manifest, manifest)
    report = {
        "status": "formal_sft_v3_frozen_cpu_only",
        "records_path": str(output_records),
        "records_sha256": records_sha,
        "manifest_path": str(output_manifest),
        "manifest_sha256": file_sha256(output_manifest),
        "actual_count": written,
        "source_counts": counts,
        "cmb_eligible_count": eligible,
        "dropped": dict(sorted(dropped.items())),
        "token_counts": dict(token_counts),
        "candidate_token_ids": frozen_candidate_ids,
        "selection_identity_sha256": manifest["selected_identity_sha256"],
        "protected_identity_sources": protected_counts,
        "raw_questions_committed": False,
        "model_loaded": False,
        "cuda_used": False,
        "final_authorized": False,
    }
    write_json(output_report, report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_formal_sft_v3(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
