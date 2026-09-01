from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.data.medqa_conflicts_v2 import (
    CONFLICT_POLICY_VERSION,
    ConflictMember,
    build_medqa_conflict_audit,
    classify_conflict_group,
    load_conflict_decisions,
    load_training_denylist,
)
from src.data.formal_store_v2 import FormalStore
from src.data.schema import (
    DataRecordV2,
    content_hash_v2,
    normalize_options_v2,
    normalize_question_v2,
)


REVISION = "a" * 40
RAW_SHA = "b" * 64


def _record(
    sample_id: str,
    split: str,
    *,
    question: str = "同一问题",
    options: tuple[str, ...] = ("选项甲", "选项乙"),
    label: str = "A",
) -> DataRecordV2:
    role = "medical_controller_dev" if split == "validation" else "medical_final_test"
    return DataRecordV2(
        sample_id=sample_id,
        source="bigbio/med_qa",
        source_revision=REVISION,
        source_license="unknown",
        upstream_split=split,
        target_role=role,
        domain="medical",
        question=question,
        options=options,
        answer_idx=label,
        normalized_question=normalize_question_v2(question),
        normalized_options=normalize_options_v2(options),
        content_hash=content_hash_v2(question, options),
        group_id=content_hash_v2(question, options),
        raw_file_sha256=RAW_SHA,
    )


def _member(
    sample_id: str,
    split: str,
    *,
    label: str = "A",
    normalized_question_sha256: str = "1" * 64,
    normalized_options_sha256: str = "2" * 64,
    raw_question_sha256: str = "3" * 64,
    raw_options_sha256: str = "4" * 64,
    option_count: int = 2,
    parse_ok: bool = True,
) -> ConflictMember:
    return ConflictMember(
        sample_id=sample_id,
        split=split,
        content_hash="f" * 64,
        normalized_question_sha256=normalized_question_sha256,
        normalized_options_sha256=normalized_options_sha256,
        raw_question_sha256=raw_question_sha256,
        raw_options_sha256=raw_options_sha256,
        option_count=option_count,
        canonical_label=label,
        parse_ok=parse_ok,
    )


def test_exact_consistent_keeps_test_and_removes_validation(tmp_path: Path):
    validation = [_record("validation-1", "validation")]
    test = [_record("test-1", "test")]
    report = build_medqa_conflict_audit(
        validation,
        test,
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    decisions = load_conflict_decisions(tmp_path / "audit.sqlite3")

    assert report["conflict_class_counts"] == {"exact_consistent": 1}
    assert decisions["validation-1"].action == "drop"
    assert decisions["validation-1"].drop_reason == "overlap_with_final_test"
    assert decisions["test-1"].action == "keep"
    assert report["cleaned_controller_candidate_count"] == 0
    assert report["cleaned_final_candidate_count"] == 1
    assert report["controller_final_exact_overlap"] == 0


def test_label_mismatch_quarantines_both_sides(tmp_path: Path):
    report = build_medqa_conflict_audit(
        [_record("validation-1", "validation", label="A")],
        [_record("test-1", "test", label="B")],
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    decisions = load_conflict_decisions(tmp_path / "audit.sqlite3")
    assert report["conflict_class_counts"] == {"label_mismatch": 1}
    assert {item.action for item in decisions.values()} == {"quarantine"}
    assert {item.drop_reason for item in decisions.values()} == {
        "cross_split_label_mismatch_quarantine"
    }


def test_option_mismatch_and_normalization_collision_are_anomalies():
    validation = [_member("v", "validation")]
    assert classify_conflict_group(
        validation,
        [_member("t", "test", normalized_options_sha256="9" * 64)],
    ) == "option_mismatch"
    assert classify_conflict_group(
        validation,
        [_member("t", "test", raw_question_sha256="8" * 64)],
    ) == "normalization_collision"


def test_duplicate_multiplicity_is_quarantined_and_order_independent(tmp_path: Path):
    validation = [
        _record("validation-z", "validation"),
        _record("validation-a", "validation"),
    ]
    test = [_record("test-1", "test")]
    first = build_medqa_conflict_audit(
        validation,
        test,
        sqlite_path=tmp_path / "first.sqlite3",
        config_sha256="c" * 64,
    )
    second = build_medqa_conflict_audit(
        list(reversed(validation)),
        list(reversed(test)),
        sqlite_path=tmp_path / "second.sqlite3",
        config_sha256="c" * 64,
    )
    assert first == second
    assert first["conflict_class_counts"] == {"duplicate_multiplicity": 1}
    assert first["both_sides_quarantined_records"] == 3


def test_nonshared_within_split_duplicate_uses_stable_sample_id(tmp_path: Path):
    validation = [
        _record("validation-z", "validation", question="仅验证重复"),
        _record("validation-a", "validation", question="仅验证重复"),
    ]
    report = build_medqa_conflict_audit(
        list(reversed(validation)),
        [_record("test-1", "test", question="独立测试")],
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    decisions = load_conflict_decisions(tmp_path / "audit.sqlite3")
    assert decisions["validation-a"].action == "keep"
    assert decisions["validation-z"].drop_reason == "within_split_exact_duplicate"
    assert report["within_split_duplicate_drops"] == {"test": 0, "validation": 1}


def test_all_shared_hashes_are_training_denylisted(tmp_path: Path):
    report = build_medqa_conflict_audit(
        [_record("validation-1", "validation")],
        [_record("test-1", "test")],
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    assert report["shared_hash_count"] == report["training_denylist_hash_count"] == 1
    assert report["groups"][0]["training_denylisted"] is True
    denylist = load_training_denylist(tmp_path / "audit.sqlite3")
    shared_hash = next(iter(denylist))
    with FormalStore(tmp_path / "formal.sqlite3", seed=42) as store:
        store.protect_hash(
            shared_hash,
            sample_id="medqa-conflict",
            target_role="medical_final_test",
        )
        for role in ("medical_sft_train", "medical_opd_o1", "general_anchors"):
            training = _record(
                f"training-{role}",
                "validation",
            )
            training = dataclasses.replace(training, target_role=role)
            assert store.stage(training, source_key="training", protected=False) == (
                "protected_hash_overlap"
            )


def test_provenance_license_and_report_redaction(tmp_path: Path):
    question = "SENSITIVE_MEDQA_QUESTION"
    option = "SENSITIVE_MEDQA_OPTION"
    report = build_medqa_conflict_audit(
        [_record("validation-1", "validation", question=question, options=(option, "B"))],
        [_record("test-1", "test", question=question, options=(option, "B"))],
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert question not in encoded and option not in encoded
    assert report["source_license"] == "unknown"
    assert report["usage_scope"] == "local_evaluation_only"
    assert report["redistribution_allowed"] is False
    assert report["primary_final_frozen"] is False
    decisions = load_conflict_decisions(tmp_path / "audit.sqlite3")
    assert decisions["test-1"].upstream_split == "test"
    assert decisions["test-1"].target_role == "medical_final_test"


def test_report_binds_policy_and_payload_digest(tmp_path: Path):
    report = build_medqa_conflict_audit(
        [_record("validation-1", "validation")],
        [_record("test-1", "test")],
        sqlite_path=tmp_path / "audit.sqlite3",
        config_sha256="c" * 64,
    )
    assert report["conflict_policy_version"] == CONFLICT_POLICY_VERSION
    assert report["config_sha256"] == "c" * 64
    assert len(report["report_payload_sha256"]) == 64
