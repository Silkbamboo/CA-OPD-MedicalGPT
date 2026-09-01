"""Resumable single-process runtime for the P2 formal streaming build."""

from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from src.data.adapters import AdapterContext, adapt_source_row
from src.data.chat import DEFAULT_SYSTEM_PROMPT, format_mcq_question
from src.data.dedup_v2 import DiskNearDuplicateIndex
from src.data.download_v2 import (
    DownloadIncompleteError,
    DownloadPolicyError,
    DownloadResult,
    ExactFileSpec,
    download_exact,
    sha256_file,
)
from src.data.formal_builder_v2 import (
    ConfiguredSourceFile,
    atomic_jsonl_export,
    atomic_write_json,
    build_protected_denylist,
    configured_source_files,
    deterministic_group_allocation,
    load_formal_config,
    medqa_option_stats,
    split_prompt_label,
)
from src.data.formal_store_v2 import FormalStore, release_file_cache_paths
from src.data.medqa_conflicts_v2 import (
    CONFLICT_POLICY_VERSION,
    ConflictDecisionIndex,
    build_medqa_conflict_audit,
)
from src.data.readers_v2 import iter_records
from src.data.schema import (
    CONTROLLER_ROLES_V2,
    DATA_PROTOCOL_VERSION,
    FINAL_ROLES_V2,
    PROMPT_ONLY_ROLES_V2,
    SCHEMA_VERSION_V2,
    SOURCE_POLICY_VERSION,
    DataRecordV2,
    content_hash_v2,
    normalize_options_v2,
    normalize_question_v2,
    stable_sample_id_v2,
)
from src.data.taxonomy_v2 import classify_general_prompt, load_taxonomy
from src.data.tokenizer_audit_v2 import (
    audit_eval_prompt,
    audit_opd_prompt,
    audit_sft_record,
    length_summary,
    load_bound_tokenizer,
    nonthinking_template_evidence,
)


class FormalBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceSnapshot:
    time_unix: float
    memory_current: int | None
    memory_max: int | None
    memory_peak: int | None
    memory_events: dict[str, int]
    disk_free_bytes: int


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None if value == "max" else int(value)


def resource_snapshot(path: str | Path = ".") -> ResourceSnapshot:
    root = Path("/sys/fs/cgroup")
    events: dict[str, int] = {}
    try:
        for line in (root / "memory.events").read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            events[key] = int(value)
    except OSError:
        pass
    usage = shutil.disk_usage(Path(path))
    return ResourceSnapshot(
        time_unix=time.time(),
        memory_current=_read_int(root / "memory.current"),
        memory_max=_read_int(root / "memory.max"),
        memory_peak=_read_int(root / "memory.peak"),
        memory_events=events,
        disk_free_bytes=usage.free,
    )


def _sha(path: str | Path) -> str:
    return sha256_file(Path(path))


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


class FormalPipeline:
    """Every public phase is restartable and checks the cgroup before work."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_formal_config(self.config_path)
        repo_root = self.config_path.parents[2]
        self.repo_root = repo_root
        paths = self.config["paths"]
        self.raw_root = (repo_root / paths["raw_root"]).resolve()
        self.interim_root = (repo_root / paths["interim_root"]).resolve()
        self.processed_root = (repo_root / paths["processed_root"]).resolve()
        self.manifest_root = (repo_root / paths["manifest_root"]).resolve()
        self.report_root = (repo_root / paths["report_root"]).resolve()
        self.store_path = self.interim_root / "formal.sqlite3"
        self.near_path = self.interim_root / "near_duplicates.sqlite3"
        self.medqa_conflict_path = self.interim_root / "medqa_conflicts.sqlite3"
        self.observed_memory_peak = 0

    def snapshot(self, phase: str) -> ResourceSnapshot:
        snapshot = resource_snapshot(self.repo_root)
        current = snapshot.memory_current or 0
        if current > int(self.config["resource_limits"]["abort_memory_bytes"]):
            self._release_known_file_cache()
            snapshot = resource_snapshot(self.repo_root)
            current = snapshot.memory_current or 0
        self.observed_memory_peak = max(self.observed_memory_peak, current)
        if current > int(self.config["resource_limits"]["abort_memory_bytes"]):
            raise FormalBuildError(
                f"{phase}: memory.current={current} exceeds configured 1.80 GiB stop gate"
            )
        if snapshot.disk_free_bytes < int(
            self.config["resource_limits"]["minimum_free_disk_bytes"]
        ):
            raise FormalBuildError(
                f"{phase}: free disk {snapshot.disk_free_bytes} is below configured gate"
            )
        return snapshot

    def _release_known_file_cache(self) -> None:
        paths: list[Path] = [
            self.store_path,
            Path(f"{self.store_path}-wal"),
            self.near_path,
            Path(f"{self.near_path}-wal"),
            self.medqa_conflict_path,
        ]
        manifest_path = self._download_manifest_path()
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                paths.extend(
                    Path(str(item["local_path"]))
                    for item in manifest.get("artifacts", ())
                    if item.get("local_path")
                )
            except (OSError, ValueError, TypeError):
                pass
        release_file_cache_paths(*paths)

    def _raw_path(self, entry: ConfiguredSourceFile) -> Path:
        return self.raw_root / entry.source_key / entry.path

    def _download_manifest_path(self) -> Path:
        return self.raw_root / "download_manifest.json"

    def _load_download_manifest(self) -> dict[str, Any]:
        path = self._download_manifest_path()
        if not path.is_file():
            raise FormalBuildError("download phase has not produced download_manifest.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def _download_with_retries(
        self, spec: ExactFileSpec, destination: Path
    ) -> DownloadResult:
        retries = int(self.config["resource_limits"]["max_retries"])
        for attempt in range(retries + 1):
            try:
                return download_exact(
                    spec,
                    destination,
                    chunk_size=int(self.config["resource_limits"]["download_chunk_bytes"]),
                )
            except DownloadIncompleteError as error:
                if attempt >= retries:
                    raise FormalBuildError(
                        f"download remained incomplete after {retries} retries: "
                        f"{spec.repository}/{spec.path}: {error}"
                    ) from error
                time.sleep(min(2 ** attempt, 4))
            except DownloadPolicyError:
                raise
            except Exception as error:
                if attempt >= retries:
                    raise FormalBuildError(
                        f"download failed after {retries} retries: {spec.repository}/{spec.path}: {error}"
                    ) from error
                time.sleep(min(2 ** attempt, 4))
        raise AssertionError("unreachable")

    def download_sources(self) -> dict[str, Any]:
        before = self.snapshot("download")
        self.raw_root.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        total_bytes = 0
        for entry in configured_source_files(self.config):
            source = self.config["sources"][entry.source_key]
            spec = ExactFileSpec(
                repository=entry.repository,
                revision=entry.revision,
                path=entry.path,
                allowed_paths=tuple(source["exact_file_allowlist"]),
                max_bytes=int(source["max_download_bytes"]),
                host=str(source.get("host", "huggingface")),
                timeout_seconds=int(self.config["resource_limits"]["timeout_seconds"]),
            )
            result = self._download_with_retries(spec, self._raw_path(entry))
            total_bytes += result.bytes
            artifacts.append(
                {
                    **asdict(entry),
                    "local_path": str(Path(result.path).resolve()),
                    "bytes": result.bytes,
                    "sha256": result.sha256,
                    "etag": result.etag,
                }
            )
            self.snapshot(f"download:{entry.source_key}")
        payload = {
            "build_version": self.config["build_version"],
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "conflict_policy_version": self.config["medqa_conflict_policy"][
                "version"
            ],
            "config_sha256": _sha(self.config_path),
            "downloaded_bytes": total_bytes,
            "artifacts": artifacts,
            "resource_before": asdict(before),
            "resource_after": asdict(self.snapshot("download-complete")),
            "model_weights_downloaded": False,
        }
        atomic_write_json(self._download_manifest_path(), payload)
        return payload

    def download_tokenizer(self) -> dict[str, Any]:
        self.snapshot("tokenizer-download")
        tokenizer_cfg = self.config["tokenizer"]
        directory = self.raw_root / "tokenizer" / _safe_component(tokenizer_cfg["id"])
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for name in tokenizer_cfg["exact_file_allowlist"]:
            spec = ExactFileSpec(
                repository=str(tokenizer_cfg["id"]),
                revision=str(tokenizer_cfg["revision"]),
                path=str(name),
                allowed_paths=tuple(tokenizer_cfg["exact_file_allowlist"]),
                max_bytes=int(tokenizer_cfg["max_download_bytes"]),
                host="huggingface_model",
                timeout_seconds=int(self.config["resource_limits"]["timeout_seconds"]),
            )
            result = self._download_with_retries(spec, directory / name)
            total += result.bytes
            if total > int(tokenizer_cfg["max_download_bytes"]):
                raise FormalBuildError("tokenizer artifacts exceed aggregate byte budget")
            files[name] = {"sha256": result.sha256, "bytes": result.bytes}
        manifest = {
            "tokenizer_id": tokenizer_cfg["id"],
            "tokenizer_revision": tokenizer_cfg["revision"],
            "files": dict(sorted(files.items())),
            "total_bytes": total,
            "model_weights_downloaded": False,
        }
        atomic_write_json(directory / "artifact_manifest.json", manifest)
        # This verifies all SHA bindings and loads with local_files_only=True.
        load_bound_tokenizer(
            directory,
            expected_id=str(tokenizer_cfg["id"]),
            expected_revision=str(tokenizer_cfg["revision"]),
        )
        return {**manifest, "local_path": str(directory)}

    def _iter_medqa_records(
        self,
        *,
        split: str,
        artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> Iterator[DataRecordV2]:
        entries = [
            entry
            for entry in configured_source_files(self.config)
            if entry.source_key == "medqa_zh" and entry.upstream_split == split
        ]
        if len(entries) != 1:
            raise FormalBuildError(f"expected one MedQA {split} artifact")
        entry = entries[0]
        artifact = artifacts.get((entry.source_key, entry.path))
        if artifact is None:
            raise FormalBuildError(f"download manifest lacks MedQA {split}")
        raw_path = Path(str(artifact["local_path"]))
        if not raw_path.is_file() or _sha(raw_path) != artifact["sha256"]:
            raise FormalBuildError(f"MedQA {split} raw artifact SHA mismatch")
        source = self.config["sources"]["medqa_zh"]
        context = AdapterContext(
            source_type="medqa_zh",
            source=str(source["repository"]),
            source_revision=str(source["revision"]),
            source_license=str(source["source_license"]),
            upstream_split=split,
            target_role=(
                "medical_controller_dev"
                if split == "validation"
                else "medical_final_test"
            ),
            raw_file_sha256=str(artifact["sha256"]),
        )
        for row in iter_records(
            raw_path,
            entry.file_format,
            batch_size=int(self.config["resource_limits"]["batch_size"]),
        ):
            result = adapt_source_row(row, context)
            if result.record is None:
                raise FormalBuildError(
                    f"MedQA conflict audit adapter drop: {result.drop_reason}"
                )
            yield result.record

    def audit_medqa_conflicts(self) -> dict[str, Any]:
        """Audit the fixed MedQA validation/test streams before normalization."""

        self.snapshot("medqa-conflicts")
        downloads = self._load_download_manifest()
        artifacts = {
            (str(item["source_key"]), str(item["path"])): item
            for item in downloads["artifacts"]
        }
        report = build_medqa_conflict_audit(
            self._iter_medqa_records(split="validation", artifacts=artifacts),
            self._iter_medqa_records(split="test", artifacts=artifacts),
            sqlite_path=self.medqa_conflict_path,
            config_sha256=_sha(self.config_path),
        )
        if report["conflict_policy_version"] != CONFLICT_POLICY_VERSION:
            raise FormalBuildError("MedQA conflict policy version mismatch")
        self.report_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.report_root / "p2_capability_overlap_conflict.json", report
        )
        self._write_medqa_conflict_report(report)
        return report

    def normalize(self) -> dict[str, Any]:
        self.snapshot("normalize")
        conflict_report = (
            self.audit_medqa_conflicts()
            if "medqa_zh" in self.config["sources"]
            else None
        )
        downloads = self._load_download_manifest()
        artifacts = {
            (str(item["source_key"]), str(item["path"])): item
            for item in downloads["artifacts"]
        }
        entries = configured_source_files(self.config)
        priority = {
            "medqa_zh": 0,
            "ceval": 1,
            "cmb": 2,
            "medical_o1": 3,
            "coig_leetcode": 4,
            "gpt4_llm_alpaca_zh": 5,
        }
        entries.sort(
            key=lambda item: (
                0 if item.source_key in {"medqa_zh", "ceval"} or item.denylist_only else 1,
                priority.get(item.source_key, 99),
                item.path,
            )
        )
        source_input: Counter[str] = Counter()
        source_accepted: Counter[str] = Counter()
        source_dropped: Counter[str] = Counter()
        medqa_option_counts: Counter[str] = Counter()
        medqa_invalid = medqa_empty = medqa_duplicate = medqa_total = 0
        self.interim_root.mkdir(parents=True, exist_ok=True)
        conflict_context = (
            ConflictDecisionIndex(self.medqa_conflict_path)
            if conflict_report is not None
            else contextlib.nullcontext(None)
        )
        with conflict_context as conflict_index, FormalStore(
            self.store_path, seed=int(self.config["seed"])
        ) as store:
            prior = store.phase("normalize")
            if prior is not None:
                if conflict_report is not None:
                    prior = {
                        **prior,
                        "medqa_conflict_policy_version": conflict_report[
                            "conflict_policy_version"
                        ],
                        "medqa_conflict_report_payload_sha256": conflict_report[
                            "report_payload_sha256"
                        ],
                        "medqa_cleaned_controller_candidate_count": conflict_report[
                            "cleaned_controller_candidate_count"
                        ],
                        "medqa_cleaned_final_candidate_count": conflict_report[
                            "cleaned_final_candidate_count"
                        ],
                    }
                    store.mark_phase("normalize", prior)
                    atomic_write_json(
                        self.interim_root / "normalize_summary.json", prior
                    )
                return prior
            if conflict_index is not None:
                for content_hash in conflict_index.iter_training_denylist():
                    store.protect_hash(
                        content_hash,
                        sample_id=f"medqa-conflict:{content_hash[:24]}",
                        target_role="medical_final_test",
                    )
            for entry in entries:
                source_cfg = self.config["sources"][entry.source_key]
                artifact = artifacts.get((entry.source_key, entry.path))
                if artifact is None:
                    raise FormalBuildError(f"download manifest lacks {entry.source_key}:{entry.path}")
                raw_path = Path(str(artifact["local_path"]))
                if not raw_path.is_file() or _sha(raw_path) != artifact["sha256"]:
                    raise FormalBuildError(f"raw artifact SHA mismatch: {raw_path}")
                entry_key = f"{entry.source_key}:{entry.path}"
                progress = store.source_progress(entry_key)
                local = dict(
                    progress["payload"]
                    if progress is not None
                    else {
                        "input": 0,
                        "accepted": 0,
                        "dropped": 0,
                        "medqa_total": 0,
                        "medqa_option_counts": {},
                        "medqa_invalid": 0,
                        "medqa_empty": 0,
                        "medqa_duplicate": 0,
                    }
                )
                start_row = int(progress["next_row"]) if progress is not None else 0
                if progress is None or progress["status"] != "complete":
                    for row_index, row in enumerate(iter_records(
                    raw_path,
                    entry.file_format,
                    batch_size=int(self.config["resource_limits"]["batch_size"]),
                    )):
                        if row_index < start_row:
                            continue
                        local["input"] = int(local["input"]) + 1
                        if entry.source_key == "medqa_zh":
                            one = medqa_option_stats((row,))
                            local["medqa_total"] = int(local["medqa_total"]) + int(
                                one["total_rows"]
                            )
                            option_counts = Counter(local["medqa_option_counts"])
                            option_counts.update(one["option_count_distribution"])
                            local["medqa_option_counts"] = dict(option_counts)
                            local["medqa_invalid"] = int(local["medqa_invalid"]) + int(
                                one["invalid_label_count"]
                            )
                            local["medqa_empty"] = int(local["medqa_empty"]) + int(
                                one["empty_option_rows"]
                            )
                            local["medqa_duplicate"] = int(
                                local["medqa_duplicate"]
                            ) + int(one["duplicate_option_rows"])
                        if entry.denylist_only:
                            record = _denylist_record(
                                row, entry, source_cfg, str(artifact["sha256"])
                            )
                            if record is None:
                                store.add_drop(
                                    _raw_identity(row),
                                    entry.source_key,
                                    "denylist_missing_question",
                                )
                                local["dropped"] = int(local["dropped"]) + 1
                            else:
                                outcome = store.stage(
                                    record, source_key=entry.source_key, protected=True
                                )
                                if outcome == "accepted":
                                    local["accepted"] = int(local["accepted"]) + 1
                                    store.deactivate(
                                        record.sample_id,
                                        "denylist_only_nontraining_split",
                                    )
                                else:
                                    local["dropped"] = int(local["dropped"]) + 1
                        else:
                            target_role = entry.target_role
                            if target_role is None:
                                target_role = {
                                    "medical_o1": "medical_sft_train",
                                    "cmb": "medical_opd_cmb",
                                    "coig_leetcode": "general_anchors",
                                    "gpt4_llm_alpaca_zh": "general_anchors",
                                }[entry.source_key]
                            subsource = {
                                "coig_leetcode": "leetcode",
                                "gpt4_llm_alpaca_zh": "gpt4_llm_alpaca_zh",
                            }.get(entry.source_key)
                            context = AdapterContext(
                                source_type=str(entry.adapter),
                                source=str(source_cfg["repository"]),
                                source_revision=str(source_cfg["revision"]),
                                source_license=str(source_cfg["source_license"]),
                                upstream_split=entry.upstream_split,
                                target_role=target_role,
                                raw_file_sha256=str(artifact["sha256"]),
                                subsource=subsource,
                                subject=entry.subject,
                            )
                            result = adapt_source_row(row, context)
                            admitted_record = result.record
                            if admitted_record is None:
                                store.add_drop(
                                    result.raw_identity,
                                    entry.source_key,
                                    str(result.drop_reason),
                                )
                                local["dropped"] = int(local["dropped"]) + 1
                            else:
                                if entry.source_key == "medqa_zh":
                                    if conflict_index is None:
                                        raise FormalBuildError(
                                            "MedQA conflict index is unavailable"
                                        )
                                    decision = conflict_index.decision(
                                        admitted_record.sample_id
                                    )
                                    if (
                                        decision.content_hash
                                        != admitted_record.content_hash
                                        or decision.upstream_split
                                        != admitted_record.upstream_split
                                        or decision.target_role
                                        != admitted_record.target_role
                                    ):
                                        raise FormalBuildError(
                                            "MedQA conflict decision provenance mismatch"
                                        )
                                    if decision.action != "keep":
                                        store.add_drop(
                                            admitted_record.sample_id,
                                            entry.source_key,
                                            str(decision.drop_reason),
                                        )
                                        local["dropped"] = int(local["dropped"]) + 1
                                        admitted_record = None
                                if admitted_record is not None:
                                    protected = (
                                        target_role in CONTROLLER_ROLES_V2
                                        or target_role in FINAL_ROLES_V2
                                        or target_role == "ceval_smoke"
                                    )
                                    outcome = store.stage(
                                        admitted_record,
                                        source_key=entry.source_key,
                                        protected=protected,
                                    )
                                    if outcome == "accepted":
                                        local["accepted"] = int(local["accepted"]) + 1
                                    else:
                                        local["dropped"] = int(local["dropped"]) + 1
                        next_row = row_index + 1
                        if next_row % 256 == 0:
                            store.checkpoint_source(
                                entry_key,
                                next_row=next_row,
                                status="in_progress",
                                payload=local,
                            )
                        if next_row % 1024 == 0:
                            store.release_file_cache(raw_path)
                            self.snapshot(f"normalize:{entry.source_key}")
                    store.checkpoint_source(
                        entry_key,
                        next_row=int(local["input"]),
                        status="complete",
                        payload=local,
                    )
                source_input[entry.source_key] += int(local["input"])
                source_accepted[entry.source_key] += int(local["accepted"])
                source_dropped[entry.source_key] += int(local["dropped"])
                medqa_total += int(local["medqa_total"])
                medqa_option_counts.update(local["medqa_option_counts"])
                medqa_invalid += int(local["medqa_invalid"])
                medqa_empty += int(local["medqa_empty"])
                medqa_duplicate += int(local["medqa_duplicate"])
                store.release_file_cache(raw_path)
                self.snapshot(f"normalize:{entry.source_key}:complete")
            payload = {
                "status": "complete",
                "input_counts": dict(sorted(source_input.items())),
                "accepted_before_quota": dict(sorted(source_accepted.items())),
                "adapter_or_exact_drops": dict(sorted(source_dropped.items())),
                "active_records": store.count(active_only=True),
                "drop_reasons": store.drop_counts(),
                "medqa_option_stats": {
                    "total_rows": medqa_total,
                    "option_count_distribution": dict(
                        sorted(medqa_option_counts.items(), key=lambda item: int(item[0]))
                    ),
                    "empty_option_rows": medqa_empty,
                    "duplicate_option_rows": medqa_duplicate,
                    "invalid_label_count": medqa_invalid,
                    "filtered_to_four_options": False,
                },
                "medqa_conflict_policy_version": (
                    conflict_report["conflict_policy_version"]
                    if conflict_report is not None
                    else "not_applicable_fixture_without_medqa"
                ),
                "medqa_conflict_report_payload_sha256": (
                    conflict_report["report_payload_sha256"]
                    if conflict_report is not None
                    else None
                ),
                "medqa_cleaned_controller_candidate_count": (
                    conflict_report["cleaned_controller_candidate_count"]
                    if conflict_report is not None
                    else 0
                ),
                "medqa_cleaned_final_candidate_count": (
                    conflict_report["cleaned_final_candidate_count"]
                    if conflict_report is not None
                    else 0
                ),
                "resource": asdict(self.snapshot("normalize-complete")),
            }
            store.mark_phase("normalize", payload)
        atomic_write_json(self.interim_root / "normalize_summary.json", payload)
        return payload

    def build_near_duplicates(self) -> dict[str, Any]:
        self.snapshot("near-duplicate")
        self.preselect_taxonomy_and_source_quotas()
        near_cfg_path = self.config_path.parent / self.config["near_duplicate"]["config"]
        import yaml

        near_cfg = yaml.safe_load(near_cfg_path.read_text(encoding="utf-8"))
        with FormalStore(self.store_path, seed=int(self.config["seed"])) as store:
            prior = store.phase("near_duplicate")
            if prior is not None:
                return prior
            with DiskNearDuplicateIndex(
                self.near_path,
                ngram_size=int(near_cfg["ngram_size"]),
                signature_size=int(near_cfg["signature_size"]),
                bands=int(near_cfg["bands"]),
            ) as index:
                indexed = 0
                for row in store.iter_index_rows(active_only=True):
                    record = store.get_record(str(row["sample_id"]))
                    text = record.normalized_question
                    if record.normalized_options:
                        text += "\n" + "\n".join(record.normalized_options)
                    index.add(
                        record.sample_id,
                        record.target_role,
                        str(row["source_key"]),
                        text,
                    )
                    indexed += 1
                    if indexed % 256 == 0:
                        index.commit()
                    if indexed % 2048 == 0:
                        self.snapshot("near-duplicate:index")
                index.commit()
                candidate_count = index.build_candidates(
                    threshold=float(near_cfg["candidate_threshold"])
                )
                high_threshold = float(near_cfg["high_confidence_cross_role_threshold"])
                parents: dict[str, str] = {}

                def find(item: str) -> str:
                    parents.setdefault(item, item)
                    while parents[item] != item:
                        parents[item] = parents[parents[item]]
                        item = parents[item]
                    return item

                def union(left: str, right: str) -> None:
                    left_root, right_root = find(left), find(right)
                    if left_root != right_root:
                        parents[max(left_root, right_root)] = min(left_root, right_root)

                cross_role = 0
                cross_protected = 0
                high_confidence_excluded = 0
                source_priority = {
                    "medical_o1": 0,
                    "cmb": 1,
                    "coig_leetcode": 2,
                    "gpt4_llm_alpaca_zh": 3,
                }
                for candidate in index.iter_candidates():
                    score = float(candidate["similarity"])
                    left_id = str(candidate["left_sample_id"])
                    right_id = str(candidate["right_sample_id"])
                    left_source = str(candidate["left_source"])
                    right_source = str(candidate["right_source"])
                    left_role = str(candidate["left_role"])
                    right_role = str(candidate["right_role"])
                    if left_source == right_source == "medical_o1" and score >= high_threshold:
                        union(left_id, right_id)
                    if left_role == right_role or score < high_threshold:
                        continue
                    cross_role += 1
                    left_protected = store.is_protected(left_id)
                    right_protected = store.is_protected(right_id)
                    if left_protected and right_protected:
                        cross_protected += 1
                        continue
                    if left_protected != right_protected:
                        loser = right_id if left_protected else left_id
                    else:
                        left_priority = source_priority.get(left_source, 99)
                        right_priority = source_priority.get(right_source, 99)
                        loser = right_id if left_priority <= right_priority else left_id
                    store.deactivate(loser, "high_confidence_cross_role_near_duplicate")
                    high_confidence_excluded += 1

                clusters: dict[str, list[str]] = defaultdict(list)
                for sample_id in parents:
                    clusters[find(sample_id)].append(sample_id)
                for members in clusters.values():
                    hashes = sorted(store.get_record(member).content_hash for member in members)
                    group_id = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
                    for member in members:
                        store.assign_group(member, group_id)
                store.commit()
                audit_target = int(near_cfg["manual_audit_target_pairs"])
                audit_rows = index.audit_rows(limit=audit_target)
                payload = {
                    "status": "complete",
                    "near_duplicate_version": near_cfg["near_duplicate_version"],
                    "indexed_records": indexed,
                    "candidate_threshold": near_cfg["candidate_threshold"],
                    "high_confidence_threshold": high_threshold,
                    "candidate_count": candidate_count,
                    "cluster_count": len(clusters),
                    "cross_role_candidate_count": cross_role,
                    "cross_protected_candidate_count": cross_protected,
                    "high_confidence_excluded": high_confidence_excluded,
                    "manual_audit_target": audit_target,
                    "manual_audit_rows": len(audit_rows),
                    "manual_audit_pending": True,
                    "human_reviewed": False,
                    "audit_rows": audit_rows,
                    "resource": asdict(self.snapshot("near-duplicate-complete")),
                }
                store.mark_phase("near_duplicate", payload)
        atomic_write_json(self.interim_root / "near_duplicate_summary.json", payload)
        return payload

    def preselect_taxonomy_and_source_quotas(self) -> dict[str, Any]:
        """Bound CMB/general pools before materializing the disk LSH index."""

        self.snapshot("taxonomy-and-source-preselection")
        taxonomy_path = self.config_path.parent / self.config["taxonomy"]["config"]
        taxonomy = load_taxonomy(taxonomy_path)
        taxonomy_counts: Counter[str] = Counter()
        with FormalStore(self.store_path, seed=int(self.config["seed"])) as store:
            prior = store.phase("preselection")
            if prior is not None:
                return prior
            processed = 0
            for record in store.iter_records(role="general_anchors"):
                decision = classify_general_prompt(record.question, taxonomy)
                taxonomy_counts[decision.status] += 1
                store.set_taxonomy(
                    record.sample_id,
                    status=decision.status,
                    capability=decision.capability,
                    rule_ids=decision.rule_ids,
                )
                if not decision.admitted:
                    store.deactivate(
                        record.sample_id,
                        "general_taxonomy_medical_excluded"
                        if decision.status == "rejected"
                        else "general_taxonomy_uncertain_fail_closed",
                    )
                processed += 1
                if processed % 256 == 0:
                    store.commit()
                if processed % 2048 == 0:
                    store.release_file_cache()
                    self.snapshot("taxonomy")
            store.commit()
            cmb_selected = store.selected_ids_for_source(
                "cmb",
                limit=int(self.config["quotas"]["medical_opd_cmb"]),
                stratified=True,
            )
            store.deactivate_unselected_source(
                "cmb", cmb_selected, "cmb_stratified_target_capacity_exhausted"
            )

            for source_key, limit in self.config["quotas"]["general_anchors"].items():
                selected = store.selected_ids_for_source(
                    source_key, limit=int(limit), stratified=False
                )
                store.deactivate_unselected_source(
                    source_key, selected, "general_anchor_source_quota_exhausted"
                )
            store.commit()
            payload = {
                "status": "complete",
                "taxonomy_version": taxonomy["taxonomy_version"],
                "taxonomy_config_sha256": _sha(taxonomy_path),
                "taxonomy_decisions": dict(sorted(taxonomy_counts.items())),
                "role_counts": store.role_counts(),
                "source_counts": store.source_counts(),
                "capability_counts": store.capability_counts(),
                "drop_reasons": store.drop_counts(),
                "duplicate_to_fill": False,
                "preserve_actual_if_short": True,
                "near_duplicate_scope": (
                    "all protected controller/final plus full Medical-O1 and "
                    "source-policy/taxonomy/quota-admitted CMB/general candidates"
                ),
                "resource": asdict(
                    self.snapshot("taxonomy-and-source-preselection-complete")
                ),
            }
            store.mark_phase("preselection", payload)
        atomic_write_json(self.interim_root / "preselection_summary.json", payload)
        return payload

    def apply_taxonomy_and_quotas(self) -> dict[str, Any]:
        """Allocate Medical-O1 near-duplicate groups after disk LSH grouping."""

        preselection = self.preselect_taxonomy_and_source_quotas()
        self.snapshot("medical-o1-group-allocation")
        with FormalStore(self.store_path, seed=int(self.config["seed"])) as store:
            prior = store.phase("allocation")
            if prior is not None:
                return prior
            medical_targets = self.config["quotas"]["medical_o1"]
            allocation = deterministic_group_allocation(
                (
                    {"sample_id": record.sample_id, "group_id": record.group_id}
                    for record in store.iter_records(source_key="medical_o1")
                ),
                targets=medical_targets,
                seed=int(self.config["seed"]),
            )
            selected_medical = set(allocation)
            for sample_id, role in allocation.items():
                store.assign_role(sample_id, role)
            store.deactivate_unselected_source(
                "medical_o1",
                selected_medical,
                "medical_o1_target_capacity_exhausted",
            )
            store.commit()
            payload = {
                "status": "complete",
                "taxonomy_version": preselection["taxonomy_version"],
                "taxonomy_config_sha256": preselection[
                    "taxonomy_config_sha256"
                ],
                "taxonomy_decisions": preselection["taxonomy_decisions"],
                "role_counts": store.role_counts(),
                "source_counts": store.source_counts(),
                "capability_counts": store.capability_counts(),
                "drop_reasons": store.drop_counts(),
                "duplicate_to_fill": False,
                "preserve_actual_if_short": True,
                "resource": asdict(
                    self.snapshot("medical-o1-group-allocation-complete")
                ),
            }
            store.mark_phase("allocation", payload)
        atomic_write_json(self.interim_root / "allocation_summary.json", payload)
        return payload

    def audit_token_lengths(self) -> dict[str, Any]:
        self.snapshot("tokenizer-audit")
        tokenizer_cfg = self.config["tokenizer"]
        directory = self.raw_root / "tokenizer" / _safe_component(tokenizer_cfg["id"])
        tokenizer, binding = load_bound_tokenizer(
            directory,
            expected_id=str(tokenizer_cfg["id"]),
            expected_revision=str(tokenizer_cfg["revision"]),
        )
        limits = self.config["length_limits"]
        template_evidence = nonthinking_template_evidence(tokenizer)
        prompt_lengths: dict[str, list[int]] = defaultdict(list)
        response_lengths: dict[str, list[int]] = defaultdict(list)
        length_drops: Counter[str] = Counter()
        with FormalStore(self.store_path, seed=int(self.config["seed"])) as store:
            prior = store.phase("tokenizer")
            if prior is not None:
                return prior
            processed = 0
            for record in store.iter_records():
                if record.target_role in {
                    "medical_sft_train",
                    "medical_sft_dev",
                    "audit_holdout",
                } and record.source == "FreedomIntelligence/medical-o1-reasoning-SFT":
                    audit = audit_sft_record(
                        tokenizer,
                        question=record.question,
                        reasoning=record.reasoning,
                        answer=str(record.answer or ""),
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        max_length=int(limits["medical_sft_full"]),
                    )
                    prompt_count = audit.prompt_tokens
                    response_count = audit.response_tokens
                    admitted = audit.admitted
                    drop_reason = audit.drop_reason
                elif record.target_role in PROMPT_ONLY_ROLES_V2:
                    question = (
                        format_mcq_question(record.question, record.options)
                        if record.options
                        else record.question
                    )
                    audit_prompt = audit_opd_prompt(
                        tokenizer,
                        question=question,
                        max_length=int(limits["opd_prompt"]),
                    )
                    prompt_count = audit_prompt.prompt_tokens
                    response_count = None
                    admitted = audit_prompt.admitted
                    drop_reason = audit_prompt.drop_reason
                elif record.options:
                    audit_prompt = audit_eval_prompt(
                        tokenizer,
                        question=record.question,
                        options=record.options,
                        max_length=int(limits["eval_prompt"]),
                    )
                    prompt_count = audit_prompt.prompt_tokens
                    response_count = None
                    admitted = audit_prompt.admitted
                    drop_reason = audit_prompt.drop_reason
                else:
                    audit_prompt = audit_opd_prompt(
                        tokenizer,
                        question=record.question,
                        max_length=int(limits["eval_prompt"]),
                    )
                    prompt_count = audit_prompt.prompt_tokens
                    response_count = None
                    admitted = audit_prompt.admitted
                    drop_reason = audit_prompt.drop_reason
                flags = tuple(
                    sorted(
                        (set(record.quality_flags) - {"tokenizer_length_pending"})
                        | {"qwen3_tokenizer_audited", "enable_thinking_false"}
                    )
                )
                updated = replace(
                    record,
                    token_count_prompt=prompt_count,
                    token_count_response=response_count,
                    quality_flags=flags,
                )
                store.update_record(updated)
                prompt_lengths[record.target_role].append(prompt_count)
                if response_count is not None:
                    response_lengths[record.target_role].append(response_count)
                if not admitted:
                    reason = str(drop_reason or "token_length_violation")
                    length_drops[reason] += 1
                    if record.target_role in CONTROLLER_ROLES_V2 or record.target_role in FINAL_ROLES_V2:
                        raise FormalBuildError(
                            f"official controller/final row exceeds configured eval length: {record.sample_id}"
                        )
                    store.deactivate(record.sample_id, reason)
                processed += 1
                if processed % 256 == 0:
                    store.commit()
                if processed % 1024 == 0:
                    self.snapshot("tokenizer-audit:records")
            store.commit()
            payload = {
                "status": "complete",
                "tokenizer_id": tokenizer_cfg["id"],
                "tokenizer_revision": tokenizer_cfg["revision"],
                "tokenizer_artifact_manifest_sha256": binding[
                    "artifact_manifest_sha256"
                ],
                "enable_thinking": False,
                "think_tags_emitted": 0,
                "nonthinking_template_evidence": template_evidence,
                "length_limits": dict(limits),
                "prompt_lengths": {
                    role: length_summary(values)
                    for role, values in sorted(prompt_lengths.items())
                },
                "response_lengths": {
                    role: length_summary(values)
                    for role, values in sorted(response_lengths.items())
                },
                "length_drop_reasons": dict(sorted(length_drops.items())),
                "role_counts_after_length_filter": store.role_counts(),
                "resource": asdict(self.snapshot("tokenizer-audit-complete")),
            }
            store.mark_phase("tokenizer", payload)
        atomic_write_json(self.interim_root / "token_lengths_summary.json", payload)
        return payload

    def _common_manifest(
        self,
        *,
        roles: Mapping[str, Any],
        sources: Mapping[str, Any],
        overlap_sha: str,
        status: str,
    ) -> dict[str, Any]:
        import datetime

        tokenizer_manifest = (
            self.raw_root
            / "tokenizer"
            / _safe_component(self.config["tokenizer"]["id"])
            / "artifact_manifest.json"
        )
        taxonomy_path = self.config_path.parent / self.config["taxonomy"]["config"]
        near_path = self.config_path.parent / self.config["near_duplicate"]["config"]
        source_config = self.repo_root / "configs/data/sources_v2.yaml"
        split_config = self.repo_root / "configs/data/splits_v2.yaml"
        filter_config = self.repo_root / "configs/data/filters_v2.yaml"
        return {
            "build_version": self.config["build_version"],
            "build_mode": "formal",
            "build_status": status,
            "synthetic_fixture": False,
            "schema_version": SCHEMA_VERSION_V2,
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "conflict_policy_version": self.config["medqa_conflict_policy"][
                "version"
            ],
            "medqa_conflict_report_sha256": _sha(
                self.report_root / "p2_capability_overlap_conflict.json"
            ),
            "seed": int(self.config["seed"]),
            "source_config_sha256": _sha(source_config),
            "split_config_sha256": _sha(split_config),
            "filter_config_sha256": _sha(filter_config),
            "formal_config_sha256": _sha(self.config_path),
            "taxonomy_version": load_taxonomy(taxonomy_path)["taxonomy_version"],
            "taxonomy_config_sha256": _sha(taxonomy_path),
            "normalization_version": self.config["normalization"]["version"],
            "normalization_implementation_sha256": _sha(
                self.repo_root / "src/data/schema.py"
            ),
            "near_duplicate_version": json.loads(
                (self.interim_root / "near_duplicate_summary.json").read_text(encoding="utf-8")
            )["near_duplicate_version"],
            "near_duplicate_config_sha256": _sha(near_path),
            "tokenizer_id": self.config["tokenizer"]["id"],
            "tokenizer_revision": self.config["tokenizer"]["revision"],
            "tokenizer_artifact_sha256": _sha(tokenizer_manifest),
            "overlap_report_sha256": overlap_sha,
            "manual_audit_pending": True,
            "human_reviewed": False,
            "primary_final_frozen": False,
            "prompt_label_separated": True,
            "public_checkpoint_release": False,
            "actual_cost_cny": None,
            "sources": dict(sources),
            "roles": dict(roles),
            "build_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def export(self) -> dict[str, Any]:
        self.snapshot("export")
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        downloads = self._load_download_manifest()
        normalize_summary = json.loads(
            (self.interim_root / "normalize_summary.json").read_text(encoding="utf-8")
        )
        allocation_summary = json.loads(
            (self.interim_root / "allocation_summary.json").read_text(encoding="utf-8")
        )
        token_summary = json.loads(
            (self.interim_root / "token_lengths_summary.json").read_text(encoding="utf-8")
        )
        near_summary = json.loads(
            (self.interim_root / "near_duplicate_summary.json").read_text(encoding="utf-8")
        )
        all_roles = (
            "medical_sft_train",
            "medical_sft_dev",
            "medical_opd_o1",
            "audit_holdout",
            "medical_opd_cmb",
            "medical_controller_dev",
            "medical_final_test",
            "general_anchors",
            "general_controller_dev",
            "general_final_test",
            "ceval_smoke",
        )
        role_metadata: dict[str, Any] = {}
        supervision_in_opd = 0
        with FormalStore(self.store_path, seed=int(self.config["seed"])) as store:
            for role in all_roles:
                count = store.count(active_only=True, role=role)
                if count == 0:
                    continue
                if role in CONTROLLER_ROLES_V2 or role in FINAL_ROLES_V2 or role == "ceval_smoke":
                    prompt_path = (self.processed_root / f"{role}.prompts.jsonl").resolve()
                    label_path = (self.processed_root / f"{role}.labels.jsonl").resolve()
                    prompt_meta = atomic_jsonl_export(
                        prompt_path,
                        (
                            split_prompt_label(record.to_dict())[0]
                            for record in store.iter_records(role=role)
                        ),
                        role=role,
                    )
                    label_meta = atomic_jsonl_export(
                        label_path,
                        (
                            split_prompt_label(record.to_dict())[1]
                            for record in store.iter_records(role=role)
                        ),
                        role=role,
                    )
                    files = [prompt_meta, label_meta]
                else:
                    records_path = (self.processed_root / f"{role}.jsonl").resolve()
                    record_meta = atomic_jsonl_export(
                        records_path,
                        (record.to_dict() for record in store.iter_records(role=role)),
                        role=role,
                    )
                    files = [record_meta]
                    if role in PROMPT_ONLY_ROLES_V2:
                        supervision_in_opd += int(record_meta["supervision_fields"])
                for file_metadata in files:
                    file_metadata["path"] = str(
                        Path(str(file_metadata["path"])).relative_to(self.repo_root)
                    )
                role_metadata[role] = {
                    "actual_count": count,
                    "upstream_splits": sorted(
                        {record.upstream_split for record in store.iter_records(role=role)}
                    ),
                    "files": files,
                }

            duplicate_ids = int(
                store.connection.execute(
                    "SELECT COUNT(*)-COUNT(DISTINCT sample_id) FROM records WHERE active=1"
                ).fetchone()[0]
            )
            duplicate_hashes = int(
                store.connection.execute(
                    "SELECT COUNT(*)-COUNT(DISTINCT content_hash) FROM records WHERE active=1"
                ).fetchone()[0]
            )
            group_conflicts = int(
                store.connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT group_id FROM records WHERE active=1
                      GROUP BY group_id HAVING COUNT(DISTINCT target_role)>1
                    )
                    """
                ).fetchone()[0]
            )
            final_overlap = int(
                store.connection.execute(
                    """
                    SELECT COUNT(*) FROM records AS final
                    JOIN records AS other ON final.content_hash=other.content_hash
                    WHERE final.active=1 AND other.active=1
                      AND final.target_role IN ('medical_final_test','general_final_test')
                      AND other.target_role NOT IN ('medical_final_test','general_final_test')
                    """
                ).fetchone()[0]
            )
            role_counts = store.role_counts()
            source_counts = store.source_counts()
            drop_counts = store.drop_counts()
            capability_counts = store.capability_counts()
            controller_prompt_tokens = {
                str(role): int(total or 0)
                for role, total in store.connection.execute(
                    """
                    SELECT target_role, SUM(
                        CAST(json_extract(record_json, '$.token_count_prompt') AS INTEGER)
                    )
                    FROM records
                    WHERE active=1 AND target_role IN (
                        'medical_controller_dev','general_controller_dev'
                    )
                    GROUP BY target_role ORDER BY target_role
                    """
                )
            }
            final_records = [
                record.to_dict()
                for final_role in sorted(FINAL_ROLES_V2)
                for record in store.iter_records(role=final_role)
            ]

        leakage = {
            "status": "PASS"
            if not any((duplicate_ids, duplicate_hashes, group_conflicts, final_overlap, supervision_in_opd))
            else "FAIL",
            "global_duplicate_sample_id_count": duplicate_ids,
            "global_duplicate_content_hash_count": duplicate_hashes,
            "cross_role_group_conflict_count": group_conflicts,
            "final_hash_outside_final_count": final_overlap,
            "opd_supervision_field_count": supervision_in_opd,
            "manual_audit_pending": True,
            "near_duplicate_candidate_count": near_summary["candidate_count"],
            "near_duplicate_cross_role_candidate_count": near_summary[
                "cross_role_candidate_count"
            ],
        }
        leakage_path = self.report_root / "leakage_formal_v2.json"
        atomic_write_json(leakage_path, leakage)
        if leakage["status"] != "PASS":
            raise FormalBuildError("formal leakage report failed")

        grouped_downloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in downloads["artifacts"]:
            grouped_downloads[str(item["source_key"])].append(item)
        sources_manifest: dict[str, Any] = {}
        for source_key, items in sorted(grouped_downloads.items()):
            source = self.config["sources"][source_key]
            composite = hashlib.sha256()
            for item in sorted(items, key=lambda value: value["path"]):
                composite.update(str(item["path"]).encode("utf-8"))
                composite.update(b"\0")
                composite.update(str(item["sha256"]).encode("ascii"))
                composite.update(b"\n")
            sources_manifest[source_key] = {
                "repository": source["repository"],
                "revision": source["revision"],
                "declared_license": source["source_license"],
                "usage_scope": source["usage_scope"],
                "redistribution_allowed": bool(source.get("redistribution_allowed", False)),
                "raw_file_sha256": composite.hexdigest(),
                "files": [
                    {
                        "path": item["path"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                        "upstream_split": item["upstream_split"],
                    }
                    for item in sorted(items, key=lambda value: value["path"])
                ],
            }

        manual_target = int(near_summary["manual_audit_target"])
        manual_count = int(near_summary["manual_audit_rows"])
        build_status = (
            "built_pending_manual_audit"
            if manual_count >= manual_target
            else "blocked_manual_audit_pair_shortfall"
        )
        common = self._common_manifest(
            roles=role_metadata,
            sources=sources_manifest,
            overlap_sha=_sha(leakage_path),
            status=build_status,
        )
        manifest_variants = {
            "formal_manifest.json": set(role_metadata),
            "medical_sft_manifest.json": {"medical_sft_train", "medical_sft_dev"},
            "opd_manifest.json": set(PROMPT_ONLY_ROLES_V2),
            "controller_manifest.json": set(CONTROLLER_ROLES_V2),
            "final_candidate_manifest.json": set(FINAL_ROLES_V2),
        }
        manifest_paths: dict[str, str] = {}
        for name, admitted_roles in manifest_variants.items():
            payload = {
                **common,
                "roles": {
                    role: metadata
                    for role, metadata in role_metadata.items()
                    if role in admitted_roles
                },
            }
            if name == "final_candidate_manifest.json":
                payload.update({"frozen": False, "primary_final_frozen": False})
            path = self.manifest_root / name
            atomic_write_json(path, payload)
            manifest_paths[name] = str(path)

        denylist = {
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "conflict_policy_version": self.config["medqa_conflict_policy"][
                "version"
            ],
            "primary_final_frozen": False,
            "formal_final_manifest": False,
            "records": build_protected_denylist(final_records),
            "medqa_cross_split_denylist": [
                {
                    "content_hash": group["content_hash"],
                    "conflict_class": group["conflict_class"],
                    "action": group["action"],
                }
                for group in json.loads(
                    (
                        self.report_root / "p2_capability_overlap_conflict.json"
                    ).read_text(encoding="utf-8")
                )["groups"]
            ],
        }
        atomic_write_json(
            self.manifest_root / "final_candidate_denylist_manifest.json", denylist
        )

        atomic_write_json(
            self.report_root / "data_stats_formal_v2.json",
            {
                "build_status": build_status,
                "targets": self.config["quotas"],
                "role_counts": role_counts,
                "source_counts": source_counts,
                "drop_reason_counts": drop_counts,
                "controller_prompt_tokens": controller_prompt_tokens,
                "controller_prompt_tokens_total": sum(
                    controller_prompt_tokens.values()
                ),
                "general_anchors_status": (
                    "provisional_coverage_review_pending"
                    if role_counts.get("general_anchors", 0)
                    < sum(self.config["quotas"]["general_anchors"].values())
                    else "built_pending_manual_audit"
                ),
                "downloaded_bytes": downloads["downloaded_bytes"],
                "processed_bytes": sum(
                    int(file["bytes"])
                    for metadata in role_metadata.values()
                    for file in metadata["files"]
                ),
            },
        )
        atomic_write_json(
            self.report_root / "taxonomy_formal_v2.json",
            {
                "taxonomy_version": allocation_summary["taxonomy_version"],
                "taxonomy_config_sha256": allocation_summary[
                    "taxonomy_config_sha256"
                ],
                "decisions": allocation_summary["taxonomy_decisions"],
                "capability_counts": capability_counts,
                "uncertain_fail_closed": True,
                "codex_review_is_human_review": False,
            },
        )
        atomic_write_json(
            self.report_root / "token_lengths_formal_v2.json", token_summary
        )
        atomic_write_json(
            self.report_root / "near_duplicate_formal_v2.json",
            {key: value for key, value in near_summary.items() if key != "audit_rows"},
        )
        self._write_manual_audit(near_summary)
        self._write_medqa_report(normalize_summary["medqa_option_stats"])
        self._write_quality_report(
            build_status,
            role_counts,
            drop_counts,
            capability_counts,
            leakage,
            controller_prompt_tokens,
        )
        self._write_license_report(sources_manifest)
        result = {
            "build_status": build_status,
            "role_counts": role_counts,
            "drop_reason_counts": drop_counts,
            "manual_audit_rows": manual_count,
            "manual_audit_pending": True,
            "primary_final_frozen": False,
            "manifest_paths": manifest_paths,
            "resource": asdict(self.snapshot("export-complete")),
        }
        atomic_write_json(self.interim_root / "export_summary.json", result)
        return result

    def _write_manual_audit(self, near_summary: Mapping[str, Any]) -> None:
        rows = list(near_summary["audit_rows"])
        import yaml

        near_cfg = yaml.safe_load(
            (
                self.config_path.parent
                / self.config["near_duplicate"]["config"]
            ).read_text(encoding="utf-8")
        )
        public_path = self.report_root / "manual_audit_formal_v2.csv"
        temporary = public_path.with_suffix(public_path.suffix + ".tmp")
        fieldnames = [
            "left_sample_id",
            "right_sample_id",
            "left_role",
            "right_role",
            "left_source",
            "right_source",
            "left_text_sha256",
            "right_text_sha256",
            "similarity",
            "human_reviewed",
        ]
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, public_path)

        private_path = self.interim_root / "manual_audit_private.csv"
        private_tmp = private_path.with_suffix(private_path.suffix + ".tmp")
        with DiskNearDuplicateIndex(
            self.near_path,
            ngram_size=int(near_cfg["ngram_size"]),
            signature_size=int(near_cfg["signature_size"]),
            bands=int(near_cfg["bands"]),
        ) as index, private_tmp.open("w", encoding="utf-8", newline="") as handle:
            private_fields = [
                "left_sample_id",
                "right_sample_id",
                "similarity",
                "left_role",
                "right_role",
                "left_source",
                "right_source",
                "left_text",
                "right_text",
            ]
            writer = csv.DictWriter(
                handle,
                fieldnames=private_fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                index.private_audit_rows(limit=int(near_summary["manual_audit_target"]))
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(private_tmp, private_path)

    def _write_medqa_report(self, stats: Mapping[str, Any]) -> None:
        distribution = ", ".join(
            f"{count} options: {rows}"
            for count, rows in stats["option_count_distribution"].items()
        )
        text = f"""# MedQA option protocol v2

## Formal source evidence

- Immutable revision: `ca00e01fbf688d59b54730857759059b2faafc57`.
- Representation: `med_qa_zh_source`; splits: official validation and test.
- Full validation/test rows scanned: {stats['total_rows']}.
- Option-count distribution: {distribution or 'none'}.
- Rows with empty options: {stats['empty_option_rows']}; duplicate options: {stats['duplicate_option_rows']}; invalid labels: {stats['invalid_label_count']}.
- No row was filtered or rewritten to manufacture four options.
- License remains `unknown`, usage is local evaluation only, redistribution is disabled.

## Representation and reference-protocol difference

The selected source representation exposes ordered `question`, `options`,
`answer`, and `answer_idx` fields.  The BigBio QA representation wraps the same
upstream material in a generic QA schema and is not selected.  The pinned
parquet revision does not expose the audited `med_qa_zh_4options_source`
representation.  The historical workspace contains no canonical script that
legally defines deletion of a fifth option; therefore the reference claim of
"4 options / 600" is not silently reproduced here.

## Protocol choices awaiting a decision

1. **A — official five-option schema:** retains all official rows and a random
   baseline of 20%, but reference four-option numbers are not directly comparable.
2. **B — naturally four-option subset only:** allowed only if such upstream rows
   exist; reduces and changes the sample population, with a 25% random baseline.
3. **C — different fixed representation/revision:** requires a new immutable
   source audit and ADR before use.

No model result was used to choose among these protocols.  P2 builds the full
test candidate denylist only; `primary_final_frozen=false` and no 600-row final
subset is frozen.
"""
        path = self.report_root / "medqa_option_protocol_v2.md"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_medqa_conflict_report(self, report: Mapping[str, Any]) -> None:
        classes = report["conflict_class_counts"]
        class_lines = [
            f"- `{name}`: {count}"
            for name, count in sorted(classes.items())
        ] or ["- none"]
        text = "\n".join(
            [
                "# P2 MedQA validation/test exact-overlap policy",
                "",
                "Status: `resolved_bplus_before_any_model_output`",
                "",
                "## Evidence boundary",
                "",
                "This report contains counts and hashes only. It contains no MedQA question, option, answer, or model output.",
                "",
                f"- Revision: `{report['source_revision']}`",
                f"- Representation: `{report['representation']}`",
                f"- Policy: `{report['conflict_policy_version']}`",
                f"- Validation rows: {report['validation_original_count']}",
                f"- Test rows: {report['test_original_count']}",
                f"- Shared ordered prompt/options hashes: {report['shared_hash_count']}",
                f"- Config SHA-256: `{report['config_sha256']}`",
                f"- Report payload SHA-256: `{report['report_payload_sha256']}`",
                "",
                "## Deterministic classification",
                "",
                *class_lines,
                "",
                "B+ keeps the test record only for a one-to-one exact-consistent group. Validation is dropped with `overlap_with_final_test`. Label/option mismatch, normalization collision, parse error, ambiguity, or duplicate multiplicity quarantines both sides. Every shared hash remains in the global training denylist.",
                "",
                "## Result",
                "",
                f"- Validation records removed (including deterministic within-split duplicate cleanup): {report['validation_removed_records']}",
                f"- Test records retained after cleanup: {report['test_retained_records']}",
                f"- Consistent cross-split test records retained: {report['consistent_test_records_retained']}",
                f"- Rows quarantined on both sides: {report['both_sides_quarantined_records']}",
                f"- Clean controller candidates: {report['cleaned_controller_candidate_count']}",
                f"- Clean final candidates: {report['cleaned_final_candidate_count']}",
                f"- Controller/final exact overlap: {report['controller_final_exact_overlap']}",
                f"- Training-denylisted shared hashes: {report['training_denylist_hash_count']}",
                "",
                "MedQA remains license `unknown`, `usage_scope=local_evaluation_only`, and `redistribution_allowed=false`. The primary 600-row final is not frozen. This policy was fixed before any model result; no item was removed based on model performance.",
                "",
            ]
        )
        path = self.report_root / "p2_capability_overlap_conflict.md"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def _write_quality_report(
        self,
        status: str,
        roles: Mapping[str, int],
        drops: Mapping[str, int],
        capabilities: Mapping[str, int],
        leakage: Mapping[str, Any],
        controller_prompt_tokens: Mapping[str, int],
    ) -> None:
        lines = [
            "# Formal data quality v2",
            "",
            f"Build status: `{status}`. This is not a model-training or capability-evaluation result.",
            "",
            "## Role counts",
            "",
            *[f"- `{key}`: {value}" for key, value in sorted(roles.items())],
            "",
            "## Drop reasons",
            "",
            *[f"- `{key}`: {value}" for key, value in sorted(drops.items())],
            "",
            "## General-anchor capability diagnostics",
            "",
            *[f"- `{key}`: {value}" for key, value in sorted(capabilities.items())],
            "",
            "## Controller candidate cost evidence",
            "",
            *[
                f"- `{key}` total prompt tokens: {value}"
                for key, value in sorted(controller_prompt_tokens.items())
            ],
            f"- Combined single-pass controller prompt tokens: {sum(controller_prompt_tokens.values())}",
            "- No model was run. Full-controller use every K optimizer steps is not approved; any smaller stratified controller requires a new pre-model-results ADR.",
            "",
            "## Evidence boundary",
            "",
            f"- Exact leakage status: `{leakage['status']}`.",
            "- Near-duplicate manual audit is pending; Codex review is not human review.",
            "- No tokenizer/model capability result, SFT, OPD, controller evaluation, or final evaluation was run.",
            "",
        ]
        path = self.report_root / "data_quality_formal_v2.md"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp, path)

    def _write_license_report(self, sources: Mapping[str, Any]) -> None:
        lines = [
            "# Formal data licence and usage report",
            "",
            "| Source | Revision | Licence | Usage scope | Raw redistribution |",
            "|---|---|---|---|---:|",
        ]
        for key, source in sorted(sources.items()):
            lines.append(
                f"| {key} | `{source['revision']}` | {source['declared_license']} | "
                f"{source['usage_scope']} | {str(source['redistribution_allowed']).lower()} |"
            )
        lines.extend(
            [
                "",
                "MedQA remains licence-unknown and local-evaluation-only. GPT4-LLM and C-Eval are non-commercial; COIG LeetCode requires attribution/share-alike notice. Public checkpoint release remains disabled pending a separate licence review.",
                "",
            ]
        )
        path = self.report_root / "license_formal_v2.md"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp, path)


def _raw_identity(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_options(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("options") or row.get("option") or row.get("choices")
    if value is None and all(label in row for label in "ABCD"):
        value = {label: row[label] for label in "ABCD"}
    if isinstance(value, Mapping):
        keys = sorted(value, key=lambda key: str(key))
        return tuple(str(value[key]).strip() for key in keys if str(value[key]).strip())
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("value") or item.get("text") or item.get("option")
            else:
                text = item
            if text is not None and str(text).strip():
                result.append(str(text).strip())
        return tuple(result)
    return ()


def _denylist_record(
    row: Mapping[str, Any], entry: ConfiguredSourceFile, source: Mapping[str, Any], raw_sha: str
) -> DataRecordV2 | None:
    question = str(row.get("question") or row.get("Question") or "").strip()
    if not question:
        return None
    options = _extract_options(row)
    raw_id = str(row.get("id") or row.get("question_id") or _raw_identity(row))
    digest = content_hash_v2(question, options)
    return DataRecordV2(
        sample_id=stable_sample_id_v2(
            source=str(source["repository"]),
            source_revision=str(source["revision"]),
            upstream_split=entry.upstream_split,
            upstream_id=raw_id,
        ),
        source=str(source["repository"]),
        source_revision=str(source["revision"]),
        source_license=str(source["source_license"]),
        upstream_split=entry.upstream_split,
        target_role="audit_holdout",
        domain="medical",
        question=question,
        options=options,
        normalized_question=normalize_question_v2(question),
        normalized_options=normalize_options_v2(options),
        content_hash=digest,
        group_id=digest,
        raw_file_sha256=raw_sha,
        quality_flags=("denylist_only_nontraining_split",),
    )
