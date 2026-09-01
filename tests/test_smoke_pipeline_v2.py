import json
from pathlib import Path

from src.data.pipeline_v2 import (
    FORBIDDEN_SUPERVISION_KEYS,
    build_smoke_pipeline,
    iter_jsonl,
)
from src.utils.run_meta import git_dirty, git_sha


ROOT = Path(__file__).parents[1]
SMOKE_CONFIG = ROOT / "configs" / "data" / "smoke_v2.yaml"


def contains_forbidden(value):
    if isinstance(value, dict):
        return any(
            str(key).casefold() in FORBIDDEN_SUPERVISION_KEYS
            or contains_forbidden(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(contains_forbidden(child) for child in value)
    return False


def test_smoke_pipeline_processes_exactly_twenty_rows_per_source(tmp_path):
    result = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "data",
        report_root=tmp_path / "reports",
    )

    assert result.source_input_counts == {
        "medical_o1": 20,
        "cmb": 20,
        "medqa_zh": 20,
        "coig": 20,
        "ceval": 20,
    }
    assert result.source_accepted_counts == result.source_input_counts
    assert result.tokenizer_audit_status == "pending_qwen3_tokenizer"
    assert result.supervision_fields_in_opd == 0


def test_prompt_only_and_final_prompt_files_are_physically_separated(tmp_path):
    result = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "data",
        report_root=tmp_path / "reports",
    )

    for role in ("medical_opd_o1", "medical_opd_cmb", "general_anchors"):
        rows = list(iter_jsonl(result.output_root / f"{role}.jsonl"))
        assert rows
        assert all(not contains_forbidden(row) for row in rows)

    for role in ("medical_final_test", "general_final_test"):
        prompts = list(iter_jsonl(result.output_root / f"{role}.prompts.jsonl"))
        labels = list(iter_jsonl(result.output_root / f"{role}.labels.jsonl"))
        assert prompts and len(prompts) == len(labels)
        assert all(not contains_forbidden(row) for row in prompts)
        assert all("question" not in row and "options" not in row for row in labels)
        assert {row["sample_id"] for row in prompts} == {
            row["sample_id"] for row in labels
        }


def test_manifest_contains_only_ids_hashes_and_statistics(tmp_path):
    result = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "data",
        report_root=tmp_path / "reports",
    )
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["data_protocol_version"] == "ca-opd-data-v2"
    assert manifest["build_mode"] == "smoke"
    assert manifest["schema_version"] == 2
    assert manifest["tokenizer_audit_status"] == "pending_qwen3_tokenizer"
    assert len(manifest["script_git_sha"]) == 40
    assert set(manifest["script_git_sha"]) <= set("0123456789abcdef")
    expected_dirty = git_dirty(ROOT)
    assert manifest["script_git_sha"] == git_sha(ROOT)
    assert manifest["dirty_worktree"] is expected_dirty
    assert manifest["script_revision_status"] == (
        "worktree_uncommitted" if expected_dirty else "committed"
    )
    assert manifest["synthetic_fixture"] is True
    for key in (
        "source_config_sha256",
        "split_config_sha256",
        "filter_config_sha256",
        "overlap_report_sha256",
    ):
        assert len(manifest[key]) == 64
    assert "患者合成病例" not in manifest_text
    assert "网络合成题" not in manifest_text
    assert all(
        set(role_meta) >= {"count", "sample_ids", "content_hashes", "files"}
        for role_meta in manifest["roles"].values()
    )


def test_smoke_build_is_byte_deterministic_for_same_config_and_seed(tmp_path):
    left = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "left-data",
        report_root=tmp_path / "left-reports",
    )
    right = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "right-data",
        report_root=tmp_path / "right-reports",
    )

    left_files = {
        path.relative_to(left.output_root): path.read_bytes()
        for path in left.output_root.rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right.output_root): path.read_bytes()
        for path in right.output_root.rglob("*")
        if path.is_file()
    }
    assert left_files == right_files
    assert left.manifest_path.read_bytes() == right.manifest_path.read_bytes()
    assert left.stats_path.read_bytes() == right.stats_path.read_bytes()
    assert left.leakage_path.read_bytes() == right.leakage_path.read_bytes()


def test_near_duplicate_groups_never_cross_roles(tmp_path):
    result = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "data",
        report_root=tmp_path / "reports",
    )
    owners = {}
    for role, metadata in result.manifest["roles"].items():
        for group_id in metadata["group_ids"]:
            assert group_id not in owners or owners[group_id] == role
            owners[group_id] = role
    assert result.leakage_report["status"] == "PASS"
    assert result.leakage_report["cross_role_group_overlap_count"] == 0
