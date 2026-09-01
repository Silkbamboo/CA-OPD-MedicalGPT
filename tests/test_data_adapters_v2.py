import hashlib

import pytest

from src.data.adapters import AdapterContext, adapt_source_row
from src.data.schema import (
    DATA_PROTOCOL_VERSION,
    content_hash_v2,
    normalize_options_v2,
    normalize_question_v2,
)


RAW_SHA = hashlib.sha256(b"synthetic-real-field-fixture").hexdigest()
REVISION = "a" * 40


def context(source_type, target_role, *, license_name="fixture-only"):
    return AdapterContext(
        source_type=source_type,
        source=f"fixture/{source_type}",
        source_revision=REVISION,
        source_license=license_name,
        upstream_split="train",
        target_role=target_role,
        raw_file_sha256=RAW_SHA,
        subsource="fixture-subsource",
    )


def test_v2_normalization_preserves_medical_semantics_and_option_order():
    question = "  患者用药 ５ mg，结果为阴性；以下哪项不正确？  "
    normalized = normalize_question_v2(question)

    assert normalized == "患者用药 5 mg,结果为阴性;以下哪项不正确?"
    assert "mg" in normalized
    assert "阴性" in normalized
    assert "不正确" in normalized
    assert content_hash_v2(question, ["A. 给药", "B. 停药"]) != content_hash_v2(
        question, ["B. 停药", "A. 给药"]
    )
    assert normalize_options_v2(["Ａ．给药", "B. 停药"]) == (
        "A.给药",
        "B. 停药",
    )


def test_medical_o1_adapter_uses_actual_capitalized_fields():
    result = adapt_source_row(
        {
            "Question": "患者发热应如何处理？",
            "Complex_CoT": "先评估危险信号。",
            "Response": "建议及时就医。",
        },
        context("medical_o1", "medical_sft_train", license_name="Apache-2.0"),
    )

    record = result.require_record()
    assert record.question == "患者发热应如何处理？"
    assert record.reasoning == "先评估危险信号。"
    assert record.answer == "建议及时就医。"
    assert record.source_license == "Apache-2.0"
    assert record.data_protocol_version == DATA_PROTOCOL_VERSION
    assert record.content_hash == content_hash_v2(record.question, record.options)


@pytest.mark.parametrize(
    ("raw", "expected_options", "expected_answer_idx"),
    [
        (
            {
                "id": "official-1",
                "question": "首选处理是？",
                "options": {"A": "观察", "B": "就医", "C": "运动", "D": "停水"},
                "answer": "就医",
                "answer_idx": "B",
            },
            ("观察", "就医", "运动", "停水"),
            "B",
        ),
        (
            {
                "id": "bigbio-1",
                "question": "BigBio 形式？",
                "choices": [
                    {"id": "A", "text": "甲"},
                    {"id": "B", "text": "乙"},
                    {"id": "C", "text": "丙"},
                    {"id": "D", "text": "丁"},
                ],
                "answer": [{"id": "B", "text": "乙"}],
            },
            ("甲", "乙", "丙", "丁"),
            "B",
        ),
    ],
)
def test_medqa_adapter_supports_official_and_bigbio_field_shapes(
    raw, expected_options, expected_answer_idx
):
    ctx = context("medqa_zh", "medical_controller_dev", license_name="unknown")
    ctx = AdapterContext(**{**ctx.__dict__, "upstream_split": "dev"})
    record = adapt_source_row(raw, ctx).require_record()

    assert record.options == expected_options
    assert record.answer_idx == expected_answer_idx
    assert record.source_license == "unknown"
    assert record.target_role == "medical_controller_dev"


def test_cmb_adapter_preserves_category_and_option_order():
    raw = {
        "id": 7,
        "exam_type": "医师考试",
        "exam_class": "执业医师",
        "exam_subject": "内科",
        "question": "哪项处理正确？",
        "option": {"A": "方案甲", "B": "方案乙", "C": "方案丙", "D": "方案丁"},
        "answer": "B",
        "analysis": "乙更合适。",
    }
    record = adapt_source_row(
        raw, context("cmb", "medical_opd_cmb", license_name="Apache-2.0")
    ).require_record()

    assert record.options == ("方案甲", "方案乙", "方案丙", "方案丁")
    assert record.answer_idx == "B"
    assert record.reasoning == "乙更合适。"
    assert record.category == "医师考试/执业医师/内科"


def test_coig_adapter_uses_instruction_input_output_and_rejects_unknown_license():
    raw = {
        "index": "coig-1",
        "instruction": "编写一个排序函数。",
        "input": "输入为整数列表。",
        "output": "可以使用归并排序。",
        "domain": "code",
        "task_name_in_eng": "leetcode",
    }
    accepted = adapt_source_row(
        raw,
        context(
            "coig",
            "general_anchors",
            license_name="CC-BY-SA-4.0",
        ),
    ).require_record()

    assert accepted.question == "编写一个排序函数。\n输入为整数列表。"
    assert accepted.answer == "可以使用归并排序。"
    assert accepted.subject == "leetcode"

    rejected = adapt_source_row(
        raw, context("coig", "general_anchors", license_name="unknown")
    )
    assert rejected.record is None
    assert rejected.drop_reason == "unknown_source_license"


def test_coig_exam_adapter_supports_pinned_textbox_field_shape():
    raw = {
        "subject": "non-medical-exam",
        "textbox_q_instruction": "请完成这道非医疗题。",
        "textbox_q_context": "这是合成上下文。",
        "textbox_question": "合成问题是什么？",
        "textbox_answer_analysis": "这是合成解析。",
        "textbox_answer": "这是合成答案。",
    }
    record = adapt_source_row(
        raw,
        context(
            "coig",
            "general_anchors",
            license_name="fixture-permissive-license",
        ),
    ).require_record()

    assert record.question == (
        "请完成这道非医疗题。\n这是合成上下文。\n合成问题是什么？"
    )
    assert record.answer == "这是合成答案。"
    assert record.reasoning == "这是合成解析。"
    assert record.subject == "non-medical-exam"


def test_coig_translated_adapter_prefers_pinned_translated_fields():
    raw = {
        "instruction": "Synthetic English instruction.",
        "input": "Synthetic English input.",
        "output": "Synthetic English output.",
        "trans_instruction": "合成中文指令。",
        "trans_input": "合成中文输入。",
        "trans_output": "合成中文输出。",
    }
    translated_context = context(
        "coig",
        "general_anchors",
        license_name="fixture-permissive-license",
    )
    translated_context = AdapterContext(
        **{**translated_context.__dict__, "subsource": "translated"}
    )
    record = adapt_source_row(raw, translated_context).require_record()

    assert record.question == "合成中文指令。\n合成中文输入。"
    assert record.answer == "合成中文输出。"


def test_coig_adapter_rejects_medical_subject_without_domain_field():
    result = adapt_source_row(
        {
            "instruction": "合成指令。",
            "output": "合成输出。",
            "subject": "clinical_medicine",
        },
        context(
            "coig",
            "general_anchors",
            license_name="fixture-permissive-license",
        ),
    )

    assert result.record is None
    assert result.drop_reason == "coig_medical_domain_excluded"


def test_ceval_adapter_uses_a_b_c_d_fields_and_filters_subject_whitelist():
    raw = {
        "id": 3,
        "question": "网络层协议是？",
        "A": "HTTP",
        "B": "IP",
        "C": "HTML",
        "D": "CSS",
        "answer": "B",
        "explanation": "IP 位于网络层。",
    }
    allowed_context = context(
        "ceval", "general_controller_dev", license_name="CC BY-NC-SA 4.0"
    )
    allowed_context = AdapterContext(
        **{
            **allowed_context.__dict__,
            "upstream_split": "val",
            "subject": "computer_network",
        }
    )
    record = adapt_source_row(raw, allowed_context).require_record()
    assert record.options == ("HTTP", "IP", "HTML", "CSS")
    assert record.answer_idx == "B"
    assert record.reasoning == "IP 位于网络层。"

    blocked_context = AdapterContext(
        **{**allowed_context.__dict__, "subject": "clinical_medicine"}
    )
    blocked = adapt_source_row(raw, blocked_context)
    assert blocked.record is None
    assert blocked.drop_reason == "ceval_subject_not_allowed"


def test_missing_required_field_returns_auditable_drop_reason():
    dropped = adapt_source_row(
        {"Response": "没有问题字段"},
        context("medical_o1", "medical_sft_train", license_name="Apache-2.0"),
    )

    assert dropped.record is None
    assert dropped.drop_reason == "missing_question"
    assert dropped.raw_identity
