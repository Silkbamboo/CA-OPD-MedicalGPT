"""Build the mutually exclusive CA-OPD splits with a versioned manifest.

Guarantees this module provides (each has a test in ``tests/test_data_splits.py``):

1. **Mutual exclusivity by construction.** Splits are filled in protection order
   - ``final_test`` first, then ``controller_dev``, then the training pools - and
   a sample whose ``content_hash`` has already been consumed is skipped. No
   post-hoc "leakage check and hope" step is required.
2. **No labels in OPD pools on disk.** Writing goes through
   ``Sample.to_record()``, which drops fields the split may not expose, and is
   double-checked by ``assert_no_label_leakage``.
3. **Reproducibility.** Given the same config and seed the byte content of every
   output file is identical; the manifest records the seed, source counts, per
   file sha256 and the git SHA of the code that produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.data.schema import (
    CONTROLLER_DEV,
    DOMAIN_GENERAL,
    DOMAIN_MEDICAL,
    FINAL_TEST,
    GENERAL_ANCHORS,
    MEDICAL_OPD_PROMPTS,
    MEDICAL_SFT,
    SPLITS,
    TASK_MCQ,
    Sample,
    SchemaError,
    assert_no_label_leakage,
)
from src.data.sources import SourceSpec, StagedSample, load_source
from src.utils.config import FieldSpec, load_config
from src.utils.io import ensure_dir, file_sha256, write_json, write_jsonl
from src.utils.run_meta import git_sha
from src.utils.seeding import new_rng

SCHEMA_VERSION = "1.0.0"

#: Destination -> (split, domain). Order = protection priority (first picks first).
ALLOCATION_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("final_test_medical", FINAL_TEST, DOMAIN_MEDICAL),
    ("final_test_general", FINAL_TEST, DOMAIN_GENERAL),
    ("controller_dev_medical", CONTROLLER_DEV, DOMAIN_MEDICAL),
    ("controller_dev_general", CONTROLLER_DEV, DOMAIN_GENERAL),
    ("medical_sft", MEDICAL_SFT, DOMAIN_MEDICAL),
    ("medical_opd_prompts", MEDICAL_OPD_PROMPTS, DOMAIN_MEDICAL),
    ("general_anchors", GENERAL_ANCHORS, DOMAIN_GENERAL),
)

#: Which source pool each destination draws from. General evaluation and
#: anchors are separate when ``sources.general_anchor`` is declared. The v1
#: compatibility path below falls back to ``general_mcq`` only for older
#: configs that predate this field; Data Protocol v2 never uses this builder.
DESTINATION_POOL: Mapping[str, str] = {
    "final_test_medical": "medical_mcq",
    "final_test_general": "general_mcq",
    "controller_dev_medical": "medical_mcq",
    "controller_dev_general": "general_mcq",
    "medical_sft": "medical_reasoning",
    "medical_opd_prompts": "medical_reasoning",
    "general_anchors": "general_anchor",
}

POOLS: Tuple[str, ...] = ("medical_reasoning", "medical_mcq", "general_mcq", "general_anchor")

DATA_SCHEMA: Dict[str, object] = {
    "version": FieldSpec((str,), doc="dataset version tag, e.g. v1"),
    "seed": FieldSpec((int,), bounds=(0, None)),
    "output_dir": FieldSpec((str,)),
    "strict": FieldSpec(
        (bool,),
        required=False,
        default=True,
        doc="if true, a destination that cannot be filled to its requested count is an error",
    ),
    "sources": {
        "medical_reasoning": FieldSpec((list,), doc="SFT/OPD-prompt source specs"),
        "medical_mcq": FieldSpec((list,), doc="labeled medical MCQ for controller dev + final test"),
        "general_mcq": FieldSpec((list,), doc="labeled general MCQ for controller dev + final test"),
        "general_anchor": FieldSpec(
            (list,),
            required=False,
            default=None,
            doc="general training anchors; omitted only by legacy v1 configs",
        ),
    },
    "allocation": {
        "final_test_medical": FieldSpec((int,), bounds=(0, None)),
        "final_test_general": FieldSpec((int,), bounds=(0, None)),
        "controller_dev_medical": FieldSpec((int,), bounds=(0, None)),
        "controller_dev_general": FieldSpec((int,), bounds=(0, None)),
        "medical_sft": FieldSpec((int,), bounds=(0, None)),
        "medical_opd_prompts": FieldSpec((int,), bounds=(0, None)),
        "general_anchors": FieldSpec((int,), bounds=(0, None)),
    },
}


@dataclass
class PoolStats:
    raw_records: int
    converted: int
    unique: int
    duplicates_within_pool: int
    sources: Dict[str, int]


@dataclass
class BuildResult:
    output_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    samples_by_split: Dict[str, List[Sample]]


def _load_pool(specs: Sequence[Mapping[str, Any]]) -> Tuple[List[StagedSample], PoolStats]:
    staged: List[StagedSample] = []
    per_source: Dict[str, int] = {}
    for raw_spec in specs:
        spec = SourceSpec.from_mapping(raw_spec)
        items = load_source(spec)
        per_source[spec.name] = len(items)
        staged.extend(items)
    converted = len(staged)

    seen: set[str] = set()
    unique: List[StagedSample] = []
    for item in staged:
        h = item.content_hash
        if h in seen:
            continue
        seen.add(h)
        unique.append(item)
    stats = PoolStats(
        raw_records=converted,
        converted=converted,
        unique=len(unique),
        duplicates_within_pool=converted - len(unique),
        sources=dict(sorted(per_source.items())),
    )
    return unique, stats


def _shuffled(pool: List[StagedSample], seed: int, pool_name: str) -> List[StagedSample]:
    """Deterministic shuffle: same seed and pool -> same order, always."""
    ordered = sorted(pool, key=lambda s: s.content_hash)  # canonical starting order
    rng = new_rng(seed, "split-shuffle", pool_name)
    rng.shuffle(ordered)
    return ordered


def build_splits(config_path: str | Path, output_dir: Optional[str | Path] = None) -> BuildResult:
    cfg = load_config(config_path, DATA_SCHEMA)
    seed = int(cfg["seed"])
    strict = bool(cfg["strict"])
    out_dir = ensure_dir(Path(output_dir) if output_dir else Path(str(cfg["output_dir"])))

    # -- load and dedup each pool
    pools: Dict[str, List[StagedSample]] = {}
    pool_stats: Dict[str, PoolStats] = {}
    for pool_name in POOLS:
        specs = cfg["sources"][pool_name]  # type: ignore[index]
        if pool_name == "general_anchor" and specs is None:
            specs = cfg["sources"]["general_mcq"]  # v1 backward compatibility only
        items, stats = _load_pool(specs)
        pools[pool_name] = _shuffled(items, seed, pool_name)
        pool_stats[pool_name] = stats

    # -- allocate in protection order, skipping globally consumed content hashes
    cursors: Dict[str, int] = {name: 0 for name in POOLS}
    consumed_hashes: set[str] = set()
    consumed_ids: set[str] = set()
    allocated: Dict[str, List[Sample]] = {name: [] for name, _, _ in ALLOCATION_ORDER}
    shortfalls: Dict[str, int] = {}

    for destination, split, domain in ALLOCATION_ORDER:
        requested = int(cfg["allocation"][destination])  # type: ignore[index]
        pool_name = DESTINATION_POOL[destination]
        pool = pools[pool_name]
        picked: List[Sample] = []
        while len(picked) < requested and cursors[pool_name] < len(pool):
            staged = pool[cursors[pool_name]]
            cursors[pool_name] += 1
            if staged.content_hash in consumed_hashes:
                continue
            sample = staged.to_sample(split)
            if sample.domain != domain:
                raise SchemaError(
                    f"destination {destination} expects domain={domain} but source "
                    f"{staged.source} produced domain={sample.domain}"
                )
            if split in (CONTROLLER_DEV, FINAL_TEST) and sample.task == TASK_MCQ:
                if sample.answer_index is None or sample.answer is None:
                    raise SchemaError(
                        f"destination {destination} requires a gold-labeled MCQ, but "
                        f"source {sample.source} sample {sample.raw_id} has no answer"
                    )
                expected = "ABCDEFGH"[sample.answer_index]
                if str(sample.answer).strip().upper() != expected:
                    raise SchemaError(
                        f"destination {destination} has inconsistent MCQ gold for "
                        f"{sample.source}/{sample.raw_id}: answer={sample.answer!r}, "
                        f"answer_index={sample.answer_index}"
                    )
            if sample.sample_id in consumed_ids:
                continue
            consumed_hashes.add(sample.content_hash)
            consumed_ids.add(sample.sample_id)
            picked.append(sample)
        if len(picked) < requested:
            shortfalls[destination] = requested - len(picked)
            if strict:
                raise SchemaError(
                    f"destination {destination} requested {requested} samples but only "
                    f"{len(picked)} unique samples were available in pool {pool_name!r}; "
                    "reduce the allocation or add sources (set strict: false to allow partial fills)"
                )
        allocated[destination] = picked

    # -- group by split and write
    samples_by_split: Dict[str, List[Sample]] = {split: [] for split in SPLITS}
    for destination, split, _ in ALLOCATION_ORDER:
        samples_by_split[split].extend(allocated[destination])

    split_files: Dict[str, Dict[str, Any]] = {}
    for split, samples in samples_by_split.items():
        records = [s.to_record() for s in samples]
        assert_no_label_leakage(records, split)
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        split_files[split] = {
            "path": str(path),
            "count": len(records),
            "sha256": file_sha256(path),
            "domains": _count_by(samples, lambda s: s.domain),
            "sources": _count_by(samples, lambda s: s.source),
            "tasks": _count_by(samples, lambda s: s.task),
            "fields_written": sorted({k for r in records for k in r}),
        }

    leakage = leakage_report(samples_by_split)
    if leakage["max_pairwise_overlap"] != 0:
        raise SchemaError(f"split construction produced overlapping samples: {leakage}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": str(cfg["version"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "seed": seed,
        "config_path": str(config_path),
        "config": _plain(cfg),
        "allocation_order": [d for d, _, _ in ALLOCATION_ORDER],
        "pools": {
            name: {
                "converted": stats.converted,
                "unique": stats.unique,
                "duplicates_within_pool": stats.duplicates_within_pool,
                "sources": stats.sources,
            }
            for name, stats in pool_stats.items()
        },
        "destinations": {
            destination: {"split": split, "domain": domain, "count": len(allocated[destination])}
            for destination, split, domain in ALLOCATION_ORDER
        },
        "shortfalls": shortfalls,
        "splits": split_files,
        "leakage_report": leakage,
        "control_policy": {
            "may_drive_control": [CONTROLLER_DEV],
            "never_reads_during_training": [FINAL_TEST],
        },
    }
    manifest_path = write_json(out_dir / "data_manifest.json", manifest)
    return BuildResult(out_dir, manifest_path, manifest, samples_by_split)


def _count_by(samples: Iterable[Sample], key) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        k = key(sample)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _plain(value: Any) -> Any:
    """Recursively convert config objects to plain json-serialisable values."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def leakage_report(samples_by_split: Mapping[str, Sequence[Sample]]) -> Dict[str, Any]:
    """Pairwise overlap on ``sample_id`` and ``content_hash`` between all splits."""
    ids = {split: {s.sample_id for s in samples} for split, samples in samples_by_split.items()}
    hashes = {split: {s.content_hash for s in samples} for split, samples in samples_by_split.items()}
    pairs: Dict[str, Dict[str, int]] = {}
    worst = 0
    names = list(samples_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            id_overlap = len(ids[a] & ids[b])
            hash_overlap = len(hashes[a] & hashes[b])
            worst = max(worst, id_overlap, hash_overlap)
            pairs[f"{a}|{b}"] = {"sample_id_overlap": id_overlap, "content_hash_overlap": hash_overlap}
    duplicate_within = {
        split: len(samples) - len(hashes[split]) for split, samples in samples_by_split.items()
    }
    return {
        "pairwise": pairs,
        "max_pairwise_overlap": worst,
        "duplicates_within_split": duplicate_within,
        "total_samples": sum(len(s) for s in samples_by_split.values()),
    }


def verify_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Re-check a manifest against the files on disk.

    Returns a report; raises :class:`SchemaError` if a file is missing or its
    sha256 no longer matches, which is how a silently hand-edited split file gets
    caught before it reaches training.
    """
    from src.utils.io import read_json

    manifest = read_json(manifest_path)
    problems: List[str] = []
    for split, info in manifest["splits"].items():
        path = Path(info["path"])
        if not path.exists():
            problems.append(f"{split}: missing file {path}")
            continue
        digest = file_sha256(path)
        if digest != info["sha256"]:
            problems.append(f"{split}: sha256 mismatch (manifest {info['sha256'][:12]}, file {digest[:12]})")
    if problems:
        raise SchemaError("manifest verification failed:\n  - " + "\n  - ".join(problems))
    return {"manifest": str(manifest_path), "splits_verified": sorted(manifest["splits"]), "ok": True}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build CA-OPD data splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verify-only", default=None, help="path to an existing data_manifest.json")
    args = parser.parse_args(argv)

    if args.verify_only:
        print(json.dumps(verify_manifest(args.verify_only), ensure_ascii=False, indent=2))
        return 0
    result = build_splits(args.config, args.output_dir)
    summary = {
        "output_dir": str(result.output_dir),
        "manifest": str(result.manifest_path),
        "counts": {k: v["count"] for k, v in result.manifest["splits"].items()},
        "max_pairwise_overlap": result.manifest["leakage_report"]["max_pairwise_overlap"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
