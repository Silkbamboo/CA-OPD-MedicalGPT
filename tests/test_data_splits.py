"""Data schema / split tests (docs/REPRODUCIBILITY.md §6.3 / docs/REPRODUCIBILITY.md §6 "数据与评测").

Coverage map:
* split mutual exclusivity + hash dedup     -> test_splits_are_mutually_exclusive*
* no answer field reaches an OPD pool       -> test_opd_prompt_file_has_no_label_fields
* evaluator must declare split, test locked -> test_load_split_blocks_final_test*
* manifest rebuildable from a fixed seed    -> test_same_seed_produces_identical_files
* converter robustness on raw formats       -> test_from_* / test_ceval_rejects_medical_subjects
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.access import (
    FinalTestAccessError,
    describe_split,
    final_test_access_count,
    load_split,
)
from src.data.build_splits import build_splits, leakage_report, verify_manifest
from src.data.schema import (
    CONTROLLER_DEV,
    FINAL_TEST,
    GENERAL_ANCHORS,
    MEDICAL_OPD_PROMPTS,
    MEDICAL_SFT,
    SPLITS,
    Sample,
    SchemaError,
    may_drive_control,
)
from src.data.sources import (
    MEDICAL_CEVAL_SUBJECTS,
    SourceSpec,
    convert_records,
    from_ceval,
    from_medical_o1_zh,
    from_medqa_zh,
)
from src.utils.hashing import content_hash, normalize_text
from src.utils.io import read_json, read_jsonl

CONFIG = Path("configs/data/fixture_cpu.yaml")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("splits")
    return build_splits(CONFIG, output_dir=out)


# ---------------------------------------------------------------------------
# normalisation / hashing
# ---------------------------------------------------------------------------


def test_normalisation_folds_width_case_punctuation_and_whitespace():
    assert normalize_text("ＡＢＣ　ａ") == normalize_text("abc a")
    assert normalize_text("患者，咳嗽三周。") == normalize_text("患者 咳嗽三周")
    assert normalize_text("A. 甲") == normalize_text("a甲")


def test_normalisation_does_not_merge_different_questions():
    assert normalize_text("血压 120") != normalize_text("血压 130")
    assert normalize_text("是否需要就医") != normalize_text("是否需要用药")


def test_content_hash_is_option_order_insensitive_but_content_sensitive():
    a = content_hash("题目", ["甲", "乙", "丙"])
    b = content_hash("题目", ["丙", "乙", "甲"])
    c = content_hash("题目", ["甲", "乙", "丁"])
    d = content_hash("题目", None)
    assert a == b
    assert a != c
    assert a != d


# ---------------------------------------------------------------------------
# schema policy
# ---------------------------------------------------------------------------


def test_sample_rejects_unknown_split_domain_task():
    kwargs = dict(source="s", domain="medical", task="mcq", question="q", options=["a", "b"])
    with pytest.raises(SchemaError, match="unknown split"):
        Sample(split="train", **kwargs)
    with pytest.raises(SchemaError, match="unknown domain"):
        Sample(split=CONTROLLER_DEV, **{**kwargs, "domain": "surgery"})
    with pytest.raises(SchemaError, match="unknown task"):
        Sample(split=CONTROLLER_DEV, **{**kwargs, "task": "chat"})


def test_sample_rejects_answer_index_out_of_range_and_mcq_without_options():
    with pytest.raises(SchemaError, match="answer_index"):
        Sample(source="s", split=FINAL_TEST, domain="medical", task="mcq",
               question="q", options=["a", "b"], answer_index=5)
    with pytest.raises(SchemaError, match="task=mcq requires options"):
        Sample(source="s", split=FINAL_TEST, domain="medical", task="mcq", question="q")


def test_to_record_strips_fields_the_split_may_not_expose():
    """A sample can hold an answer in memory and still be written without it."""
    s = Sample(
        source="src", split=MEDICAL_OPD_PROMPTS, domain="medical", task="open_qa",
        question="患者主诉咳嗽三周，应如何处理？", answer="建议尽快就医", reasoning="红旗征象",
    )
    record = s.to_record()
    assert "answer" not in record and "reasoning" not in record
    assert record["question"].startswith("患者主诉")
    assert record["content_hash"] == s.content_hash

    anchor = Sample(
        source="src", split=GENERAL_ANCHORS, domain="general", task="mcq",
        question="下列说法正确的是？", options=["甲", "乙"], answer="A", answer_index=0,
    )
    anchor_record = anchor.to_record()
    assert "options" in anchor_record
    assert "answer" not in anchor_record and "answer_index" not in anchor_record


def test_hashes_are_recomputed_not_trusted_from_input():
    record = {
        "source": "src", "split": FINAL_TEST, "domain": "medical", "task": "mcq",
        "question": "题目", "options": ["甲", "乙"], "answer": "A", "answer_index": 0,
        "content_hash": "deadbeef", "text_hash": "deadbeef",
    }
    sample = Sample.from_record(record)
    assert sample.content_hash != "deadbeef"
    assert sample.content_hash == content_hash("题目", ["甲", "乙"])


def test_only_controller_dev_may_drive_control():
    assert may_drive_control(CONTROLLER_DEV) is True
    for split in SPLITS:
        if split != CONTROLLER_DEV:
            assert may_drive_control(split) is False


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------


def test_from_medical_o1_zh_maps_cot_and_response():
    staged = from_medical_o1_zh(
        {"Question": "问题", "Complex_CoT": "推理", "Response": "回答", "id": "x1"}, 0
    )
    assert staged is not None
    assert staged.task == "reasoning_sft"
    assert staged.reasoning == "推理" and staged.answer == "回答"
    assert from_medical_o1_zh({"Question": "问题", "Response": ""}, 0) is None


@pytest.mark.parametrize(
    "record,expected_index",
    [
        ({"question": "q", "options": ["甲", "乙", "丙", "丁"], "answer_idx": "C"}, 2),
        ({"question": "q", "options": ["甲", "乙", "丙", "丁"], "answer_idx": 1}, 1),
        ({"question": "q", "options": ["甲", "乙", "丙", "丁"], "answer": "丁"}, 3),
        ({"question": "q", "options": {"A": "甲", "B": "乙"}, "answer_idx": "B"}, 1),
        ({"question": "q", "options": [{"key": "A", "value": "甲"}, {"key": "B", "value": "乙"}], "answer": "乙"}, 1),
    ],
)
def test_from_medqa_zh_resolves_every_supported_label_encoding(record, expected_index):
    staged = from_medqa_zh(record, 0)
    assert staged is not None
    assert staged.answer_index == expected_index
    assert staged.answer == "ABCD"[expected_index]


def test_from_medqa_zh_drops_unresolvable_labels_instead_of_guessing():
    assert from_medqa_zh({"question": "q", "options": ["甲", "乙"]}, 0) is None
    assert from_medqa_zh({"question": "q", "options": ["甲", "乙"], "answer": "丙"}, 0) is None
    assert from_medqa_zh({"question": "q", "options": ["甲", "乙"], "answer_idx": 9}, 0) is None


def test_ceval_rejects_medical_subjects():
    for subject in MEDICAL_CEVAL_SUBJECTS:
        record = {"subject": subject, "question": "q", "A": "甲", "B": "乙", "C": "丙", "D": "丁", "answer": "A"}
        assert from_ceval(record, 0) is None, f"{subject} leaked into the general pool"
    ok = from_ceval({"subject": "law", "question": "q", "A": "甲", "B": "乙", "C": "丙", "D": "丁", "answer": "B"}, 0)
    assert ok is not None and ok.domain == "general" and ok.answer_index == 1


def test_convert_records_respects_max_samples():
    records = [{"Question": f"q{i}", "Response": "a"} for i in range(10)]
    assert len(convert_records(records, "medical_o1_zh", max_samples=3)) == 3


def test_source_spec_validation():
    with pytest.raises(SchemaError, match="unknown converter"):
        SourceSpec(name="x", converter="nope")
    with pytest.raises(SchemaError, match="requires 'path'"):
        SourceSpec(name="x", converter="ceval", kind="jsonl")
    with pytest.raises(SchemaError, match="requires 'hf_path'"):
        SourceSpec(name="x", converter="ceval", kind="hf")
    with pytest.raises(SchemaError, match="unknown source spec keys"):
        SourceSpec.from_mapping({"name": "x", "converter": "ceval", "kind": "jsonl", "path": "p", "typo": 1})


# ---------------------------------------------------------------------------
# split construction
# ---------------------------------------------------------------------------


def test_all_splits_are_created_with_requested_counts(built):
    counts = {split: info["count"] for split, info in built.manifest["splits"].items()}
    assert counts[FINAL_TEST] == 16  # 8 medical + 8 general
    assert counts[CONTROLLER_DEV] == 12
    assert counts[MEDICAL_SFT] == 20
    assert counts[MEDICAL_OPD_PROMPTS] == 10
    assert counts[GENERAL_ANCHORS] == 10
    assert built.manifest["shortfalls"] == {}


def test_splits_are_mutually_exclusive_by_id_and_content_hash(built):
    report = built.manifest["leakage_report"]
    assert report["max_pairwise_overlap"] == 0
    for pair, overlap in report["pairwise"].items():
        assert overlap["sample_id_overlap"] == 0, pair
        assert overlap["content_hash_overlap"] == 0, pair
    assert all(v == 0 for v in report["duplicates_within_split"].values())


def test_leakage_report_detects_a_planted_overlap(built):
    """The检查 itself must be able to fail, otherwise it proves nothing."""
    medical = built.samples_by_split[MEDICAL_SFT]
    planted = {
        MEDICAL_SFT: medical,
        FINAL_TEST: [medical[0].with_split(FINAL_TEST)],
    }
    report = leakage_report(planted)
    assert report["max_pairwise_overlap"] == 1
    assert report["pairwise"][f"{MEDICAL_SFT}|{FINAL_TEST}"]["content_hash_overlap"] == 1


def test_opd_prompt_file_has_no_label_fields(built):
    records = read_jsonl(built.output_dir / f"{MEDICAL_OPD_PROMPTS}.jsonl")
    assert records
    for record in records:
        assert "answer" not in record
        assert "answer_index" not in record
        assert "reasoning" not in record
        assert "options" not in record
    fields = built.manifest["splits"][MEDICAL_OPD_PROMPTS]["fields_written"]
    assert "answer" not in fields and "reasoning" not in fields


def test_general_anchor_file_has_options_but_no_answers(built):
    records = read_jsonl(built.output_dir / f"{GENERAL_ANCHORS}.jsonl")
    assert records
    for record in records:
        assert record["options"]
        assert "answer" not in record and "answer_index" not in record


def test_dev_and_test_keep_labels(built):
    for split in (CONTROLLER_DEV, FINAL_TEST):
        records = read_jsonl(built.output_dir / f"{split}.jsonl")
        assert records
        for record in records:
            assert record["answer"] in list("ABCD")
            assert isinstance(record["answer_index"], int)


def test_general_splits_contain_no_medical_ceval_subjects(built):
    for split in (GENERAL_ANCHORS, CONTROLLER_DEV, FINAL_TEST):
        for sample in built.samples_by_split[split]:
            if sample.domain != "general":
                continue
            subject = (sample.meta or {}).get("subject", "")
            assert subject not in MEDICAL_CEVAL_SUBJECTS, (split, subject)


def test_manifest_records_reproducibility_fields(built):
    manifest = built.manifest
    for key in ("schema_version", "dataset_version", "seed", "git_sha", "config", "pools", "splits", "leakage_report"):
        assert key in manifest, f"manifest missing {key}"
    assert manifest["seed"] == 42
    for info in manifest["splits"].values():
        assert len(info["sha256"]) == 64
    assert manifest["pools"]["medical_reasoning"]["duplicates_within_pool"] >= 1  # the planted duplicate


def test_same_seed_produces_identical_files(tmp_path):
    a = build_splits(CONFIG, output_dir=tmp_path / "a")
    b = build_splits(CONFIG, output_dir=tmp_path / "b")
    for split in SPLITS:
        assert (tmp_path / "a" / f"{split}.jsonl").read_bytes() == (tmp_path / "b" / f"{split}.jsonl").read_bytes()
        assert a.manifest["splits"][split]["sha256"] == b.manifest["splits"][split]["sha256"]


def test_different_seed_changes_assignment(tmp_path):
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["seed"] = 4242
    path = tmp_path / "seed2.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    a = build_splits(CONFIG, output_dir=tmp_path / "a")
    b = build_splits(path, output_dir=tmp_path / "b")
    assert a.manifest["splits"][FINAL_TEST]["sha256"] != b.manifest["splits"][FINAL_TEST]["sha256"]
    # but both remain internally leak-free
    assert b.manifest["leakage_report"]["max_pairwise_overlap"] == 0


def test_strict_mode_refuses_impossible_allocation(tmp_path):
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["allocation"]["medical_sft"] = 10_000
    path = tmp_path / "too_big.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(SchemaError, match="requested 10000"):
        build_splits(path, output_dir=tmp_path / "out")

    cfg["strict"] = False
    path2 = tmp_path / "not_strict.yaml"
    path2.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    result = build_splits(path2, output_dir=tmp_path / "out2")
    assert result.manifest["shortfalls"]["medical_sft"] > 0
    assert result.manifest["leakage_report"]["max_pairwise_overlap"] == 0


def test_verify_manifest_detects_tampering(built, tmp_path):
    result = build_splits(CONFIG, output_dir=tmp_path / "v")
    assert verify_manifest(result.manifest_path)["ok"] is True
    target = tmp_path / "v" / f"{CONTROLLER_DEV}.jsonl"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sample_id": "hand-edited"}) + "\n")
    with pytest.raises(SchemaError, match="sha256 mismatch"):
        verify_manifest(result.manifest_path)


# ---------------------------------------------------------------------------
# access control
# ---------------------------------------------------------------------------


def test_load_split_requires_explicit_split_name(built):
    samples = load_split(built.output_dir, CONTROLLER_DEV)
    assert len(samples) == 12
    with pytest.raises(SchemaError, match="unknown split"):
        load_split(built.output_dir, "dev")


def test_load_split_blocks_final_test_by_default(built):
    with pytest.raises(FinalTestAccessError, match="allow_final_test=True"):
        load_split(built.output_dir, FINAL_TEST)
    with pytest.raises(FinalTestAccessError, match="reason"):
        load_split(built.output_dir, FINAL_TEST, allow_final_test=True)


def test_final_test_access_is_logged(tmp_path):
    result = build_splits(CONFIG, output_dir=tmp_path / "audit")
    assert final_test_access_count(result.output_dir) == 0
    samples = load_split(
        result.output_dir, FINAL_TEST, allow_final_test=True, reason="unit test: frozen checkpoint evaluation"
    )
    assert len(samples) == 16
    assert final_test_access_count(result.output_dir) == 1
    log = (result.output_dir / "final_test_access.log").read_text(encoding="utf-8")
    assert "reason=unit test" in log
    assert "test_data_splits.py" in log


def test_describe_split_reports_control_permission(built):
    dev = describe_split(built.output_dir, CONTROLLER_DEV)
    assert dev["may_drive_control"] is True
    assert dev["count"] == 12
    assert set(dev["domains"]) == {"medical", "general"}
    opd = describe_split(built.output_dir, MEDICAL_OPD_PROMPTS)
    assert opd["may_drive_control"] is False


# ---------------------------------------------------------------------------
# remote source declarations / metadata gate (network-free)
# ---------------------------------------------------------------------------


def test_source_spec_requires_exactly_one_hf_config_form():
    base = dict(
        name="x", converter="ceval", kind="hf", hf_path="ceval/ceval-exam", hf_revision="abc123"
    )
    with pytest.raises(SchemaError, match="exactly one"):
        SourceSpec.from_mapping(base)
    with pytest.raises(SchemaError, match="exactly one"):
        SourceSpec.from_mapping({**base, "hf_name": "law", "hf_names": ["logic"]})
    with pytest.raises(SchemaError, match="duplicate"):
        SourceSpec.from_mapping({**base, "hf_names": ["law", "law"]})


def test_hf_metadata_gate_rejects_missing_config_and_split(tmp_path):
    from src.data.hf_metadata import verify_hf_metadata

    config = tmp_path / "data.yaml"
    config.write_text(
        """version: x
seed: 1
output_dir: out
strict: true
sources:
  medical_reasoning: []
  medical_mcq: []
  general_mcq:
    - {name: bad, converter: ceval, kind: hf, hf_path: ceval/ceval-exam, hf_revision: abc123, hf_name: all, hf_split: val}
  general_anchor: []
allocation:
  final_test_medical: 0
  final_test_general: 0
  controller_dev_medical: 0
  controller_dev_general: 0
  medical_sft: 0
  medical_opd_prompts: 0
  general_anchors: 0
"""
    )
    with pytest.raises(SchemaError, match="missing declared config"):
        verify_hf_metadata(
            config,
            lambda path, revision, trust: ["law"],
            lambda path, name, revision, trust: ["val"],
        )

    config.write_text(config.read_text().replace("hf_name: all", "hf_name: law"))
    with pytest.raises(SchemaError, match="split 'val' does not exist"):
        verify_hf_metadata(
            config,
            lambda path, revision, trust: ["law"],
            lambda path, name, revision, trust: ["dev"],
        )


def test_multi_config_ceval_loader_injects_subject_and_filters_medical(monkeypatch):
    import sys
    import types

    from src.data.sources import load_hf_source

    calls = []

    def fake_load_dataset(path, name, **kwargs):
        calls.append((path, name, kwargs))
        return [{"id": 1, "question": f"{name}?", "A": "甲", "B": "乙", "C": "丙", "D": "丁", "answer": "A"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    spec = SourceSpec(
        name="ceval_mix",
        converter="ceval",
        kind="hf",
        hf_path="ceval/ceval-exam",
        hf_names=("law", "clinical_medicine"),
        hf_revision="ceval-rev",
        hf_split="val",
    )
    rows = load_hf_source(spec)
    assert calls == [
        ("ceval/ceval-exam", "law", {"split": "val", "revision": "ceval-rev"}),
        ("ceval/ceval-exam", "clinical_medicine", {"split": "val", "revision": "ceval-rev"}),
    ]
    assert len(rows) == 1
    assert rows[0].source == "ceval_law"
    assert rows[0].meta["subject"] == "law"


def test_split_builder_rejects_unlabeled_evaluator_mcq(tmp_path):
    raw = tmp_path / "unlabeled.jsonl"
    raw.write_text(
        json.dumps({"subject": "law", "id": 1, "question": "q", "A": "甲", "B": "乙", "C": "丙", "D": "丁"}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "unlabeled-eval.yaml"
    config.write_text(
        f"""version: label-guard
seed: 1
output_dir: {tmp_path / 'out'}
strict: true
sources:
  medical_reasoning: []
  medical_mcq: []
  general_mcq:
    - name: unlabeled
      converter: ceval
      kind: jsonl
      path: {raw}
  general_anchor: []
allocation:
  final_test_medical: 0
  final_test_general: 1
  controller_dev_medical: 0
  controller_dev_general: 0
  medical_sft: 0
  medical_opd_prompts: 0
  general_anchors: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="requires a gold-labeled MCQ"):
        build_splits(config)
