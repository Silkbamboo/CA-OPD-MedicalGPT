from __future__ import annotations

import pytest

from src.eval.p9_statistics import (
    select_best_medical_checkpoint,
    theil_sen_slope,
    validate_frozen_controller_anchors,
)
from src.opd.p9_adaptive_dose_protocol import P9ProtocolError


def test_theil_sen_uses_all_registered_medical_points():
    assert theil_sen_slope({120: 217, 150: 218, 180: 219, 200: 220}) > 0
    assert theil_sen_slope({120: 217, 150: 217, 180: 217, 200: 217}) == 0


def test_best_checkpoint_uses_medical_only_and_earlier_tie_break():
    curves = {
        120: {"medical_correct": 217, "general_correct": 128},
        150: {"medical_correct": 225, "general_correct": 100},
        180: {"medical_correct": 225, "general_correct": 140},
        200: {"medical_correct": 224, "general_correct": 150},
    }
    assert select_best_medical_checkpoint(curves) == 150


def test_frozen_controller_anchors_must_reproduce_before_decision():
    metrics = {
        "B0": {"medical_accuracy": 219 / 300, "general_micro_accuracy": 128 / 209},
        "B1": {"medical_accuracy": 240 / 300, "general_micro_accuracy": 139 / 209},
        "B2_step120": {
            "medical_accuracy": 217 / 300,
            "general_micro_accuracy": 128 / 209,
        },
    }
    assert validate_frozen_controller_anchors(metrics)["passed"]
    metrics["B0"]["medical_accuracy"] = 218 / 300
    with pytest.raises(P9ProtocolError, match="anchors"):
        validate_frozen_controller_anchors(metrics)
