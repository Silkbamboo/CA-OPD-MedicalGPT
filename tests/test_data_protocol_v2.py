import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.data.access import (
    FinalManifestAccessError,
    load_manifest_for_scheduler,
    load_manifest_for_trainer,
)
from src.data.adapters import AdapterContext, adapt_source_row
from src.data.chat import render_qwen3_nonthinking, sft_assistant_eos_loss_mask
from src.data.pipeline_v2 import (
    DataProtocolError,
    _stratified_take,
    build_smoke_pipeline,
    iter_jsonl,
    validate_role_isolation,
)
from src.data.schema import (
    FINAL_ROLES_V2,
    TRAINING_ROLES_V2,
    content_hash_v2,
)


ROOT = Path(__file__).parents[1]
SMOKE_CONFIG = ROOT / "configs" / "data" / "smoke_v2.yaml"


@pytest.fixture()
def smoke(tmp_path):
    return build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "data",
        report_root=tmp_path / "reports",
    )


def test_01_sample_ids_are_globally_unique(smoke):
    ids = [
        sample_id
        for metadata in smoke.manifest["roles"].values()
        for sample_id in metadata["sample_ids"]
    ]
    assert len(ids) == len(set(ids))


def test_01b_role_isolation_gate_rejects_duplicate_sample_id_in_same_role():
    record = adapt_source_row(
        {"Question": "合成问题", "Complex_CoT": "推理", "Response": "回答"},
        AdapterContext(
            source_type="medical_o1",
            source="fixture/medical-o1",
            source_revision="a" * 40,
            source_license="Apache-2.0",
            upstream_split="train",
            target_role="medical_sft_train",
            raw_file_sha256="b" * 64,
        ),
    ).require_record()
    with pytest.raises(DataProtocolError, match="duplicate sample_id"):
        validate_role_isolation([record, record])


def test_01c_stratified_sampling_records_every_unselected_row_drop_reason():
    records = []
    for index in range(2):
        records.append(
            adapt_source_row(
                {
                    "id": index,
                    "exam_type": "医师考试",
                    "exam_class": "执业医师",
                    "exam_subject": "内科",
                    "question": f"合成 CMB 问题 {index}",
                    "option": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
                    "answer": "B",
                },
                AdapterContext(
                    source_type="cmb",
                    source="fixture/cmb",
                    source_revision="a" * 40,
                    source_license="Apache-2.0",
                    upstream_split="train",
                    target_role="medical_opd_cmb",
                    raw_file_sha256="b" * 64,
                ),
            ).require_record()
        )
    drops = []
    selected = _stratified_take(records, target=1, seed=42, drops=drops)

    assert len(selected) == 1
    assert drops == [
        {
            "raw_identity": next(
                record.sample_id
                for record in records
                if record.sample_id != selected[0].sample_id
            ),
            "drop_reason": "cmb_stratified_target_capacity_exhausted",
        }
    ]


def test_02_content_hashes_are_reconstructable_from_exported_prompts(smoke):
    for path in smoke.output_root.glob("*.jsonl"):
        if path.name.endswith(".labels.jsonl"):
            continue
        for row in iter_jsonl(path):
            assert row["content_hash"] == content_hash_v2(
                row["question"], row.get("options", [])
            )


def test_03_group_id_never_crosses_target_roles(smoke):
    owners = {}
    for role, metadata in smoke.manifest["roles"].items():
        for group_id in metadata["group_ids"]:
            assert owners.setdefault(group_id, role) == role


def test_04_medical_sft_train_and_medical_opd_o1_are_disjoint(smoke):
    roles = smoke.manifest["roles"]
    assert set(roles["medical_sft_train"]["sample_ids"]).isdisjoint(
        roles["medical_opd_o1"]["sample_ids"]
    )
    assert set(roles["medical_sft_train"]["content_hashes"]).isdisjoint(
        roles["medical_opd_o1"]["content_hashes"]
    )


def test_05_medical_o1_audit_is_not_a_training_role(smoke):
    assert "audit_holdout" not in TRAINING_ROLES_V2
    assert set(smoke.manifest["roles"]["audit_holdout"]["sample_ids"]).isdisjoint(
        {
            sample_id
            for role in TRAINING_ROLES_V2
            for sample_id in smoke.manifest["roles"].get(role, {}).get(
                "sample_ids", []
            )
        }
    )


def test_06_medqa_official_dev_and_test_never_map_to_training_roles(smoke):
    roles = smoke.manifest["roles"]
    assert roles["medical_controller_dev"]["count"] == 10
    assert roles["medical_final_test"]["count"] == 10
    assert all(
        role not in TRAINING_ROLES_V2
        for role in ("medical_controller_dev", "medical_final_test")
    )


def test_07_ceval_val_and_test_never_enter_general_anchors(smoke):
    roles = smoke.manifest["roles"]
    anchor_ids = set(roles["general_anchors"]["sample_ids"])
    assert anchor_ids.isdisjoint(roles["general_controller_dev"]["sample_ids"])
    assert anchor_ids.isdisjoint(roles["general_final_test"]["sample_ids"])


def test_08_ceval_test_never_enters_controller(smoke):
    roles = smoke.manifest["roles"]
    assert set(roles["general_final_test"]["sample_ids"]).isdisjoint(
        roles["general_controller_dev"]["sample_ids"]
    )
    assert "general_final_test" in FINAL_ROLES_V2


@pytest.mark.parametrize(
    "key",
    [
        "answer",
        "answer_idx",
        "label",
        "reasoning",
        "response",
        "solution",
        "output",
        "completion",
    ],
)
def test_09_opd_export_contains_no_supervision_key(smoke, key):
    def keys(value):
        if isinstance(value, dict):
            return {str(name).casefold() for name in value} | {
                nested
                for child in value.values()
                for nested in keys(child)
            }
        if isinstance(value, list):
            return {nested for child in value for nested in keys(child)}
        return set()

    for role in ("medical_opd_o1", "medical_opd_cmb", "general_anchors"):
        for row in iter_jsonl(smoke.output_root / f"{role}.jsonl"):
            assert key not in keys(row)


def test_10_scheduler_rejects_a_manifest_with_final_roles(smoke):
    with pytest.raises(FinalManifestAccessError, match="final"):
        load_manifest_for_scheduler(smoke.manifest)


def test_11_trainer_rejects_a_manifest_with_final_roles(smoke):
    with pytest.raises(FinalManifestAccessError, match="final"):
        load_manifest_for_trainer(smoke.manifest, stage="sft")


def test_pre_source_policy_manifest_is_rejected_by_all_training_consumers(smoke):
    stale = dict(smoke.manifest)
    stale.pop("source_policy_version")
    with pytest.raises(ValueError, match="source policy"):
        load_manifest_for_trainer(stale, stage="sft")
    with pytest.raises(ValueError, match="source policy"):
        load_manifest_for_scheduler(stale)


def test_12_ceval_subject_outside_allowlist_is_filtered():
    result = adapt_source_row(
        {
            "id": 1,
            "question": "合成题",
            "A": "甲",
            "B": "乙",
            "C": "丙",
            "D": "丁",
            "answer": "A",
        },
        AdapterContext(
            source_type="ceval",
            source="ceval/ceval-exam",
            source_revision="a" * 40,
            source_license="CC BY-NC-SA 4.0",
            upstream_split="val",
            target_role="general_controller_dev",
            raw_file_sha256="b" * 64,
            subject="clinical_medicine",
        ),
    )
    assert result.drop_reason == "ceval_subject_not_allowed"


def test_13_same_config_and_seed_rebuild_same_ids_and_hashes(tmp_path):
    left = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "left",
        report_root=tmp_path / "left-reports",
    )
    right = build_smoke_pipeline(
        SMOKE_CONFIG,
        output_root=tmp_path / "right",
        report_root=tmp_path / "right-reports",
    )
    assert {
        role: (metadata["sample_ids"], metadata["content_hashes"])
        for role, metadata in left.manifest["roles"].items()
    } == {
        role: (metadata["sample_ids"], metadata["content_hashes"])
        for role, metadata in right.manifest["roles"].items()
    }


def test_14_qwen3_non_thinking_snapshot_has_no_think_tag():
    rendered = render_qwen3_nonthinking(
        [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "答案"},
        ],
        enable_thinking=False,
    )
    assert rendered == (
        "<|im_start|>user\n问题<|im_end|>\n"
        "<|im_start|>assistant\n答案<|im_end|>\n"
    )
    assert "<think>" not in rendered
    with pytest.raises(ValueError, match="non-thinking"):
        render_qwen3_nonthinking(
            [{"role": "user", "content": "问题"}], enable_thinking=True
        )


def test_15_sft_loss_mask_covers_only_assistant_content_and_eos():
    assert sft_assistant_eos_loss_mask(
        ["system", "user", "assistant", "assistant_eos", "padding"],
        token_ids=[10, 11, 12, 2, 0],
        eos_token_id=2,
        attention_mask=[1, 1, 1, 1, 0],
    ) == [0, 0, 1, 1, 0]


def test_16_final_prompt_and_label_are_physically_separate(smoke):
    prompts = list(
        iter_jsonl(smoke.output_root / "medical_final_test.prompts.jsonl")
    )
    labels = list(
        iter_jsonl(smoke.output_root / "medical_final_test.labels.jsonl")
    )
    assert all("answer" not in row and "answer_idx" not in row for row in prompts)
    assert all("question" not in row and "options" not in row for row in labels)


def test_17_medqa_license_remains_unknown(smoke):
    medqa = smoke.manifest["sources"]["medqa_zh"]
    assert medqa["declared_license"] == "unknown"
    assert medqa["declared_license"] not in {"Apache-2.0", "MIT"}


def test_18_coig_unknown_license_is_rejected():
    result = adapt_source_row(
        {"instruction": "合成任务", "output": "合成回答", "domain": "general"},
        AdapterContext(
            source_type="coig",
            source="BAAI/COIG",
            source_revision="a" * 40,
            source_license="unknown",
            upstream_split="train",
            target_role="general_anchors",
            raw_file_sha256="b" * 64,
            subsource="exam",
        ),
    )
    assert result.drop_reason == "unknown_source_license"


def test_19_final_hash_in_training_data_fails_the_build():
    base = adapt_source_row(
        {"Question": "相同合成问题", "Complex_CoT": "推理", "Response": "回答"},
        AdapterContext(
            source_type="medical_o1",
            source="fixture/medical-o1",
            source_revision="a" * 40,
            source_license="Apache-2.0",
            upstream_split="train",
            target_role="medical_sft_train",
            raw_file_sha256="b" * 64,
        ),
    ).require_record()
    leaked_final = replace(
        base,
        sample_id="fixture/final:different",
        target_role="medical_final_test",
    )
    with pytest.raises(DataProtocolError, match="content_hash crosses target roles"):
        validate_role_isolation([base, leaked_final])
