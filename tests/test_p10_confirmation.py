from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from src.data.schema import content_hash_v2
from src.eval.p10_confirmation import (
    P10ConfirmationError,
    atomic_jsonl,
    audit_pool_overlaps,
    begin_label_access,
    canonical_adapter_sha256,
    classify_confirmation,
    load_p10_config,
    prompt_execution_row,
    prepare_cuda_peak_stats,
    validate_confirmation_prompt_binding,
    validate_prediction_records,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/public/p10_b2_step240_confirmation.recorded.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt(sample_id: str, question: str, *, role: str = "medical_teacher_confirmation_dev") -> dict:
    options = ["one", "two", "three", "four", "five"]
    content_hash = content_hash_v2(question, options)
    return {
        "sample_id": sample_id,
        "target_role": role,
        "source": "bigbio/med_qa",
        "source_revision": "a" * 40,
        "upstream_split": "validation",
        "domain": "medical",
        "subject": "fixture-subject",
        "question": question,
        "options": options,
        "normalized_question": question,
        "normalized_options": options,
        "content_hash": content_hash,
        "group_id": content_hash,
    }


def test_p10_config_freezes_the_only_candidate_and_forbids_training_and_final() -> None:
    config = load_p10_config(CONFIG)
    assert config["comparison"] == {
        "primary": "B2_step240_vs_B0_paired_medical_confirmation_600",
        "routes": ["B0", "B2_step240"],
        "only_candidate_step": 240,
        "b1_included": False,
        "checkpoint_selection_after_confirmation": False,
    }
    assert config["b2_step240"]["adapter_sha256"] == (
        "d9829cdc3382eb04846c61a608bac82722039efbcfec5797b2bcea1104ed534b"
    )
    assert config["b2_step240"]["adapter_weight_sha256"] == (
        "3a038ba0c1f036bcaf95e82f985314964b31b0234bc6c5cb214a80e8b7948842"
    )
    assert config["b2_step240"]["checkpoint_manifest_sha256"] == (
        "a3fafacaba237209db80fb105dcbe9f60f3f032aabf5e0032276b723ae4c3332"
    )
    assert config["isolation"]["no_training"] is True
    assert config["isolation"]["no_checkpoint_selection"] is True
    assert config["isolation"]["final_access_allowed"] is False
    assert set(config["isolation"]["forbidden_roles"]) == {
        "medical_final_test",
        "general_final_test",
    }


@pytest.mark.parametrize(
    ("delta", "ci", "p_value", "expected"),
    [
        (12, (0.001, 0.04), 0.049, "b2_step240_confirmation_positive"),
        (1, (-0.01, 0.03), 0.01, "b2_step240_confirmation_weak_positive"),
        (1, (0.001, 0.03), 0.05, "b2_step240_confirmation_weak_positive"),
        (0, (-0.02, 0.02), 1.0, "b2_step240_confirmation_not_supported"),
        (-1, (-0.03, 0.01), 1.0, "b2_step240_confirmation_not_supported"),
    ],
)
def test_confirmation_state_is_derived_only_from_preregistered_rules(
    delta: int, ci: tuple[float, float], p_value: float, expected: str
) -> None:
    assert classify_confirmation(
        delta_questions=delta,
        bootstrap_ci=ci,
        mcnemar_exact_p=p_value,
        integrity_passed=True,
    ) == expected
    assert classify_confirmation(
        delta_questions=delta,
        bootstrap_ci=ci,
        mcnemar_exact_p=p_value,
        integrity_passed=False,
    ) == "blocked_confirmation_integrity"


def test_prompt_binding_is_reconstructable_and_never_opens_label_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = tmp_path / "confirmation.prompts.jsonl"
    labels = tmp_path / "confirmation.labels.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [_prompt("bigbio/med_qa:" + f"{index:024x}", f"question {index}") for index in range(3)]
    _write_jsonl(prompts, rows)
    labels.write_bytes(b"do-not-open-before-join")
    ids_sha = hashlib.sha256(
        "".join(str(row["sample_id"]) + "\n" for row in rows).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "data_protocol_version": "ca-opd-data-v2",
        "role": "medical_teacher_confirmation_dev",
        "status": "frozen_before_candidate_results",
        "actual_count": 3,
        "source": "bigbio/med_qa",
        "source_revision": "a" * 40,
        "source_upstream_split": "validation",
        "selected_sample_ids_sha256": ids_sha,
        "prompt_label_separated": True,
        "final_authorized": False,
        "final_artifacts_opened": False,
        "one_use_confirmation": True,
        "artifacts": [
            {"kind": "prompts", "path": str(prompts), "count": 3, "sha256": _sha(prompts)},
            {"kind": "labels", "path": str(labels), "count": 3, "sha256": "f" * 64},
        ],
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    config = {
        "role": "medical_teacher_confirmation_dev",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "prompt_path": str(prompts),
        "prompt_sha256": _sha(prompts),
        "label_path": str(labels),
        "label_sha256": "f" * 64,
        "count": 3,
        "source": "bigbio/med_qa",
        "source_revision": "a" * 40,
        "upstream_split": "validation",
        "selected_sample_ids_sha256": ids_sha,
    }
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == labels.resolve():
            raise AssertionError("label file was opened before the join")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = validate_confirmation_prompt_binding(config)
    assert result["count"] == 3
    assert result["normalized_hash_reconstructed"] == 3
    assert result["label_content_opened"] is False
    assert result["prompt_supervision_field_count"] == 0


def test_pool_audit_reports_exact_hash_and_group_leakage(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirmation.jsonl"
    clean = tmp_path / "clean.jsonl"
    leaked = tmp_path / "leaked.jsonl"
    row = _prompt("bigbio/med_qa:" + "1" * 24, "confirmation")
    _write_jsonl(confirmation, [row])
    _write_jsonl(clean, [_prompt("bigbio/med_qa:" + "2" * 24, "different")])
    _write_jsonl(leaked, [{**_prompt("other:" + "3" * 24, "other"), "group_id": row["group_id"]}])
    result = audit_pool_overlaps(confirmation, {"clean": clean, "leaked": leaked})
    assert result["clean"]["overlap"] == {"sample_id": 0, "content_hash": 0, "group_id": 0}
    assert result["leaked"]["overlap"]["group_id"] == 1


def _prediction(sample_id: str, *, predicted: str = "B") -> dict:
    scores = {"A": -2.0, "B": -0.1, "C": -1.0, "D": -3.0, "E": -4.0}
    return {
        "sample_id": sample_id,
        "target_role": "medical_controller_dev",
        "domain": "medical",
        "subject": "fixture-subject",
        "predicted_label": predicted,
        "candidate_scores": scores,
        "candidate_tokenization": [
            {"label": label, "token_ids": [token]}
            for label, token in zip("ABCDE", (32, 33, 34, 35, 36), strict=True)
        ],
        "prompt_sha256": "a" * 64,
        "prompt_token_ids": [1, 2, 3],
        "choice_backend": "transformers_direct_logits",
        "labels_opened_during_execution": False,
    }


def test_prediction_freeze_requires_ordered_finite_label_free_records() -> None:
    rows = [_prediction("s1"), _prediction("s2")]
    summary = validate_prediction_records(rows, expected_ids=["s1", "s2"], route="B0")
    assert summary["count"] == 2
    assert summary["labels_opened_during_execution"] is False
    with pytest.raises(P10ConfirmationError, match="order"):
        validate_prediction_records(list(reversed(rows)), expected_ids=["s1", "s2"], route="B0")
    with pytest.raises(P10ConfirmationError, match="supervision"):
        validate_prediction_records([{**rows[0], "answer_idx": "B"}, rows[1]], expected_ids=["s1", "s2"], route="B0")
    bad = _prediction("s1")
    bad["candidate_scores"]["A"] = math.inf
    with pytest.raises(P10ConfirmationError, match="finite"):
        validate_prediction_records([bad, rows[1]], expected_ids=["s1", "s2"], route="B0")


def test_label_access_intent_is_atomic_and_cannot_be_created_twice(tmp_path: Path) -> None:
    first = begin_label_access(tmp_path, combined_prediction_sha256="a" * 64)
    assert first["p10_confirmation_label_open_attempts"] == 1
    assert first["final_access_count"] == 0
    assert json.loads((tmp_path / "label_access_intent.json").read_text())["combined_prediction_sha256"] == "a" * 64
    with pytest.raises(P10ConfirmationError, match="already exists"):
        begin_label_access(tmp_path, combined_prediction_sha256="a" * 64)


def test_prompt_execution_changes_only_the_capability_role() -> None:
    source = _prompt("s1", "question")
    execution = prompt_execution_row(source)
    assert execution["confirmation_source_role"] == "medical_teacher_confirmation_dev"
    assert execution["target_role"] == "medical_controller_dev"
    assert execution["question"] == source["question"]
    with pytest.raises(P10ConfirmationError, match="supervision"):
        prompt_execution_row({**source, "answer_idx": "A"})


def test_atomic_jsonl_refuses_overwrite_and_preserves_label_free_rows(tmp_path: Path) -> None:
    target = tmp_path / "predictions.jsonl"
    evidence = atomic_jsonl(target, [_prediction("s1"), _prediction("s2")])
    assert evidence["count"] == 2
    assert evidence["sha256"] == _sha(target)
    with pytest.raises(P10ConfirmationError, match="new path"):
        atomic_jsonl(target, [_prediction("s1")])
    with pytest.raises(P10ConfirmationError, match="supervision"):
        atomic_jsonl(tmp_path / "leaked.jsonl", [{**_prediction("s1"), "gold": "B"}])


def test_b2_adapter_identity_uses_canonical_tensors_not_transport_bytes(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"irrelevant":"transport bytes"}\n')
    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "base_model.model.layer.lora_B.weight": torch.tensor([[5.0], [6.0]]),
        },
        checkpoint / "adapter_model.safetensors",
    )
    assert canonical_adapter_sha256(checkpoint) == (
        "b3fa0731b69cb17b4be88149f4bd3edaa3fc2f4778fe1e0d3d667d9345055fe3"
    )


def test_peak_memory_telemetry_initializes_cuda_before_reset() -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeCuda:
        def is_initialized(self) -> bool:
            return False

        def init(self) -> None:
            calls.append(("init", None))

        def set_device(self, value: int) -> None:
            calls.append(("set_device", value))

        def reset_peak_memory_stats(self, value: int) -> None:
            calls.append(("reset", value))

    class FakeTorch:
        cuda = FakeCuda()

    assert prepare_cuda_peak_stats(FakeTorch(), "cuda:0") == 0
    assert calls == [("init", None), ("set_device", 0), ("reset", 0)]
