from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.p7_stage120_controller import checkpoint_route
from src.eval.p7_stage120_statistics import (
    execute_decision_state_machine,
    pareto_frontier,
    summarize_stage120,
)


def _rows(correct: tuple[int, ...]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"m-{index}",
            "domain": "medical",
            "subject": "medical",
            "target_role": "medical_controller_dev",
            "correct": bool(value),
            "invalid": False,
            "truncated": False,
        }
        for index, value in enumerate(correct)
    ]


def test_checkpoint_route_binds_complete_resume_eligible_adapter(tmp_path: Path):
    checkpoint = tmp_path / "step_060"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "resume_eligible": True,
                "optimizer_step": 60,
                "policy_version": 60,
                "package_content_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    route = checkpoint_route("IDT-v2", 60, checkpoint)
    assert route["name"] == "IDT_step60"
    assert route["step"] == 60
    assert route["adapter_path"] == str(checkpoint.resolve())
    assert len(route["adapter_ordered_sha256"]) == 64
    assert len(route["adapter_weight_sha256"]) == 64
    assert len(route["adapter_manifest_sha256"]) == 64

    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    manifest["resume_eligible"] = False
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="resume eligible"):
        checkpoint_route("IDT-v2", 60, checkpoint)


def test_pareto_frontier_removes_strictly_dominated_checkpoints():
    metrics = {
        "IDT_step60": {"medical_accuracy": 0.73, "general_micro_accuracy": 0.61},
        "IDT_step90": {"medical_accuracy": 0.72, "general_micro_accuracy": 0.60},
        "CA_step60": {"medical_accuracy": 0.74, "general_micro_accuracy": 0.60},
        "CA_step90": {"medical_accuracy": 0.73, "general_micro_accuracy": 0.62},
    }
    assert pareto_frontier(metrics, tuple(metrics)) == ["CA_step60", "CA_step90"]


def test_stage120_summary_has_feasibility_pairing_and_same_step_ca_idt():
    metrics = {
        "B0": {"medical_accuracy": 0.5, "general_micro_accuracy": 0.6},
        "IDT_step60": {"medical_accuracy": 0.5, "general_micro_accuracy": 0.6},
        "IDT_step90": {"medical_accuracy": 0.75, "general_micro_accuracy": 0.58},
        "IDT_step120": {"medical_accuracy": 0.5, "general_micro_accuracy": 0.6},
        "CA_step60": {"medical_accuracy": 0.75, "general_micro_accuracy": 0.6},
        "CA_step90": {"medical_accuracy": 0.5, "general_micro_accuracy": 0.6},
        "CA_step120": {"medical_accuracy": 0.5, "general_micro_accuracy": 0.6},
    }
    scored = {
        "IDT_step60": _rows((1, 0, 0, 1)),
        "CA_step60": _rows((1, 1, 0, 1)),
        "IDT_step90": _rows((1, 1, 0, 1)),
        "CA_step90": _rows((1, 0, 0, 1)),
        "IDT_step120": _rows((1, 0, 1, 0)),
        "CA_step120": _rows((1, 1, 1, 0)),
    }
    report = summarize_stage120(
        metrics=metrics,
        scored_rows=scored,
        gpu_hours={name: float(index + 1) for index, name in enumerate(metrics)},
        general_constraint_delta=0.01,
        bootstrap_seed=42,
        bootstrap_samples=100,
    )
    assert report["selection"]["IDT-v2"]["selected_step"] == 60
    assert report["selection"]["CA-OPD-v2"]["selected_step"] == 60
    assert report["feasible_checkpoint_ratio"] == {"CA-OPD-v2": 1.0, "IDT-v2": 2 / 3}
    assert report["ca_minus_idt_same_step"]["step60"]["medical"]["paired_delta"] == 0.25
    assert report["first_feasible"]["CA-OPD-v2"]["step"] == 60


def test_decision_state_machine_preserves_repair_precedence_and_mixed_scale():
    repair = execute_decision_state_machine(
        {
            "repair_before_scale": True,
            "close_at_120": True,
            "recommend_b2_scale_to_300": True,
            "recommend_idt_ca_scale_to_300": True,
            "stop_no_scale": False,
        }
    )
    assert repair["primary_state"] == "repair_before_scale"
    assert repair["automatic_300_launch"] is False

    mixed = execute_decision_state_machine(
        {
            "repair_before_scale": False,
            "close_at_120": False,
            "recommend_b2_scale_to_300": True,
            "recommend_idt_ca_scale_to_300": True,
            "stop_no_scale": False,
        }
    )
    assert mixed["primary_state"] == "recommend_b2_scale_to_300"
    assert mixed["recommended_scale_methods"] == ["B2", "IDT-v2", "CA-OPD-v2"]
    assert mixed["completion_status"] == "stage120_complete_mixed_recommendation"
    assert mixed["automatic_300_launch"] is False
