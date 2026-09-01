import json
from pathlib import Path

import pytest
import yaml

from src.utils.artifacts import (
    ArtifactError,
    RunMetadata,
    append_metric,
    initialize_run_artifacts,
)


def metadata(**overrides):
    values = {
        "run_id": "sft-dry-fixture",
        "stage": "sft",
        "git_sha": "a" * 40,
        "dirty_worktree": True,
        "model_id": "mock-model",
        "model_revision": "mock-revision",
        "tokenizer_revision": "mock-tokenizer",
        "data_protocol_version": "ca-opd-data-v2",
        "data_manifest_sha256": "b" * 64,
        "seed": 42,
        "package_versions": {"python": "3.12.3"},
        "cuda_info": None,
        "gpu_info": [],
        "hardware_status": "not_available_in_cpu_dry_run",
        "start_time": "2026-08-03T00:00:00Z",
        "end_time": None,
        "status": "dry_run",
        "failure_reason": None,
        "estimated_cost_cny": 0.0,
        "actual_cost_cny": None,
    }
    values.update(overrides)
    return RunMetadata(**values)


def test_run_metadata_freezes_required_fields_and_null_actual_cost():
    value = metadata()
    payload = value.to_dict()

    assert payload["actual_cost_cny"] is None
    assert payload["cuda_info"] is None
    assert payload["gpu_info"] == []
    assert payload["hardware_status"] == "not_available_in_cpu_dry_run"
    assert payload["data_protocol_version"] == "ca-opd-data-v2"


@pytest.mark.parametrize("status", ["planned", "dry_run", "running"])
def test_unrun_or_incomplete_metadata_cannot_claim_actual_cost(status):
    with pytest.raises(ArtifactError, match="actual_cost_cny"):
        metadata(status=status, actual_cost_cny=12.5)


def test_writer_creates_complete_standard_run_inventory(tmp_path):
    data_manifest = {
        "schema_version": 2,
        "data_protocol_version": "ca-opd-data-v2",
        "roles": {"medical_sft_train": {"count": 1}},
    }
    paths = initialize_run_artifacts(
        tmp_path,
        config={"stage": "sft", "seed": 42},
        metadata=metadata(),
        data_manifest=data_manifest,
    )

    expected = {
        "config.yaml",
        "metadata.json",
        "data_manifest.json",
        "metrics.jsonl",
        "summary.json",
        "stdout.log",
        "cost.json",
        "checkpoints/index.json",
    }
    actual = {
        str(path.relative_to(paths.run_dir))
        for path in paths.run_dir.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert yaml.safe_load(paths.config.read_text(encoding="utf-8")) == {
        "seed": 42,
        "stage": "sft",
    }
    assert json.loads(paths.data_manifest.read_text(encoding="utf-8")) == data_manifest
    assert json.loads(paths.cost.read_text(encoding="utf-8")) == {
        "currency": "CNY",
        "estimated_cost_cny": 0.0,
        "actual_cost_cny": None,
        "cost_status": "not_run",
    }
    assert paths.metrics.read_text(encoding="utf-8") == ""
    assert paths.stdout.read_text(encoding="utf-8") == ""


def test_writer_refuses_to_overwrite_existing_run_directory(tmp_path):
    initialize_run_artifacts(
        tmp_path,
        config={"stage": "sft"},
        metadata=metadata(),
        data_manifest={
            "schema_version": 2,
            "data_protocol_version": "ca-opd-data-v2",
            "roles": {"medical_sft_train": {"count": 1}},
        },
    )
    with pytest.raises(ArtifactError, match="already exists"):
        initialize_run_artifacts(
            tmp_path,
            config={"stage": "sft"},
            metadata=metadata(),
            data_manifest={
                "schema_version": 2,
                "data_protocol_version": "ca-opd-data-v2",
                "roles": {"medical_sft_train": {"count": 1}},
            },
        )


def test_metrics_writer_appends_canonical_jsonl_and_rejects_nan(tmp_path):
    paths = initialize_run_artifacts(
        tmp_path,
        config={"stage": "sft"},
        metadata=metadata(),
        data_manifest={
            "schema_version": 2,
            "data_protocol_version": "ca-opd-data-v2",
            "roles": {"medical_sft_train": {"count": 1}},
        },
    )
    append_metric(paths.metrics, {"step": 0, "train/loss": 1.25})
    append_metric(paths.metrics, {"step": 1, "train/loss": 1.0})
    assert paths.metrics.read_text(encoding="utf-8").splitlines() == [
        '{"step":0,"train/loss":1.25}',
        '{"step":1,"train/loss":1.0}',
    ]
    with pytest.raises(ArtifactError, match="finite JSON"):
        append_metric(paths.metrics, {"step": 2, "train/loss": float("nan")})
