from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.paired_stats import (
    PairedStatsError,
    paired_comparison,
    score_label_free_predictions,
    teacher_readiness,
    verify_teacher_artifact,
)


def _pred(sample_id: str, prediction: str, *, domain: str = "medical", subject: str = "medical",
          role: str | None = None, invalid: bool = False, truncated: bool = False) -> dict:
    return {
        "sample_id": sample_id,
        "predicted_label": prediction,
        "domain": domain,
        "subject": subject,
        "target_role": role or ("medical_controller_dev" if domain == "medical" else "general_controller_dev"),
        "invalid": invalid,
        "truncated": truncated,
    }


def _label(sample_id: str, answer: str, *, role: str = "medical_controller_dev") -> dict:
    return {"sample_id": sample_id, "answer_idx": answer, "target_role": role}


def test_label_free_predictions_join_by_id_independent_of_file_order():
    predictions = [_pred("b", "B"), _pred("a", "A")]
    labels = [_label("a", "A"), _label("b", "A")]
    scored = score_label_free_predictions(predictions, labels)
    assert [row["sample_id"] for row in scored] == ["a", "b"]
    assert [row["correct"] for row in scored] == [True, False]
    assert all("answer_idx" not in row and "gold" not in row for row in predictions)


def test_join_fails_on_duplicate_missing_or_final_ids():
    with pytest.raises(PairedStatsError, match="duplicate"):
        score_label_free_predictions([_pred("a", "A"), _pred("a", "B")], [_label("a", "A")])
    with pytest.raises(PairedStatsError, match="sets differ"):
        score_label_free_predictions([_pred("a", "A")], [_label("b", "A")])
    with pytest.raises(PairedStatsError, match="final"):
        score_label_free_predictions(
            [_pred("a", "A", role="medical_final_test")],
            [_label("a", "A", role="medical_final_test")],
        )


def test_paired_bootstrap_and_mcnemar_are_deterministic_and_order_independent():
    labels = [_label(f"m{i}", "A") for i in range(6)]
    b0 = [_pred(f"m{i}", "A" if i < 3 else "B") for i in range(6)]
    b1 = [_pred(f"m{i}", "A" if i in {0, 1, 3, 4} else "B") for i in range(6)]
    scored0 = score_label_free_predictions(b0, labels)
    scored1 = score_label_free_predictions(list(reversed(b1)), list(reversed(labels)))
    first = paired_comparison(scored0, scored1, seed=42, bootstrap_samples=500)
    second = paired_comparison(list(reversed(scored0)), scored1, seed=42, bootstrap_samples=500)
    assert first == second
    assert first["accuracy"]["B0"] == 0.5
    assert first["accuracy"]["B1"] == pytest.approx(4 / 6)
    assert first["paired_delta"] == pytest.approx(1 / 6)
    assert first["improved"] == 2
    assert first["regressed"] == 1
    assert first["unchanged"] == 3
    assert first["mcnemar"]["b0_wrong_b1_right"] == 2
    assert first["mcnemar"]["b0_right_b1_wrong"] == 1
    assert first["bootstrap_95_ci"][0] <= first["paired_delta"] <= first["bootstrap_95_ci"][1]


def test_mcnemar_continuity_correction_is_zero_for_equal_discordant_counts():
    rows0 = [
        {**_pred("improved", "A"), "correct": False},
        {**_pred("regressed", "A"), "correct": True},
    ]
    rows1 = [
        {**_pred("improved", "A"), "correct": True},
        {**_pred("regressed", "A"), "correct": False},
    ]
    report = paired_comparison(rows0, rows1, seed=42, bootstrap_samples=50)
    assert report["mcnemar"]["continuity_corrected_chi_square"] == 0.0


def test_paired_comparison_requires_identical_samples_and_rejects_final():
    row = {**_pred("a", "A"), "correct": True}
    with pytest.raises(PairedStatsError, match="sample sets"):
        paired_comparison([row], [{**row, "sample_id": "b"}])
    with pytest.raises(PairedStatsError, match="final"):
        paired_comparison(
            [{**row, "target_role": "medical_final_test"}],
            [{**row, "target_role": "medical_final_test"}],
        )


def test_paired_comparison_reports_domain_subject_invalid_and_truncation_deltas():
    rows0 = [
        {**_pred("m", "A"), "correct": False},
        {**_pred("g", "A", domain="general", subject="logic", invalid=True), "correct": True},
    ]
    rows1 = [
        {**_pred("m", "A"), "correct": True},
        {**_pred("g", "A", domain="general", subject="logic", truncated=True), "correct": False},
    ]
    report = paired_comparison(rows0, rows1, seed=42, bootstrap_samples=100)
    assert report["domains"]["medical"]["paired_delta"] == 1.0
    assert report["subjects"]["logic"]["paired_delta"] == -1.0
    assert report["invalid_rate_delta"] == -0.5
    assert report["truncation_rate_delta"] == 0.5


@pytest.mark.parametrize(
    "delta,expected",
    [(0.031, True), (0.03, True), (0.029, "ambiguous"), (-0.03, "ambiguous"), (-0.031, False)],
)
def test_teacher_readiness_gate_is_frozen_before_gpu_results(delta, expected):
    decision = teacher_readiness(
        artifact_valid=True,
        b0_medical_choice_accuracy=0.60,
        b1_medical_choice_accuracy=0.60 + delta,
        b1_generation_invalid_rate=0.04,
        b1_generation_truncation_rate=0.01,
    )
    assert decision["teacher_artifact_valid"] is True
    assert decision["teacher_knowledge_ready"] == expected
    assert decision["teacher_generation_contract_ready"] is True


def test_generation_gate_is_separate_from_knowledge_gate():
    decision = teacher_readiness(
        artifact_valid=True,
        b0_medical_choice_accuracy=0.50,
        b1_medical_choice_accuracy=0.55,
        b1_generation_invalid_rate=0.051,
        b1_generation_truncation_rate=0.011,
    )
    assert decision["teacher_knowledge_ready"] is True
    assert decision["teacher_generation_contract_ready"] is False


def test_teacher_artifact_verification_recomputes_manifest_and_adapter_sha(tmp_path: Path):
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter-fixture")
    adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps({
        "base_model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "adapter_file": adapter.name,
        "adapter_sha256": adapter_sha,
    }), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_teacher_artifact(
        manifest,
        expected_manifest_sha256=manifest_sha,
        expected_adapter_sha256=adapter_sha,
        expected_base_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    ) is True
    adapter.write_bytes(b"tampered")
    with pytest.raises(PairedStatsError, match="adapter SHA"):
        verify_teacher_artifact(
            manifest,
            expected_manifest_sha256=manifest_sha,
            expected_adapter_sha256=adapter_sha,
            expected_base_revision="1cfa9a7208912126459214e8b04321603b3df60c",
        )
