#!/usr/bin/env python3
"""Validate the arithmetic and claim boundaries of public aggregate results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"


def load(name: str) -> dict:
    value = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {name}")
    return value


def main() -> int:
    sft = load("sft_v3_confirmation.json")
    stage120 = load("stage120_controller.json")
    curve = load("b2_dose_curve.json")
    p10 = load("p10_confirmation.json")

    assert sft["total"] == 600
    assert sft["sft_v3_step450"]["correct"] - sft["base"]["correct"] == 24
    assert abs(sft["delta_percentage_points"] - 4.0) < 1e-12
    assert sft["paired_bootstrap_95_ci"][0] > 0
    assert sft["mcnemar_exact_two_sided_p"] < 0.05

    assert stage120["routes"]["CA_step120"]["medical_correct"] <= stage120["routes"]["IDT_step120"]["medical_correct"]
    assert stage120["routes"]["CA_step120"]["general_correct"] <= stage120["routes"]["IDT_step120"]["general_correct"]
    assert stage120["b1_vs_b0"]["general_improved"] == 20
    assert stage120["b1_vs_b0"]["general_regressed"] == 9
    assert stage120["b1_vs_b0"]["general_mcnemar_exact_two_sided_p"] > 0.05

    points = {row["step"]: row for row in curve["points"]}
    assert points[240]["medical_correct"] - curve["base"]["medical_correct"] == 4
    assert points[270]["medical_correct"] < points[240]["medical_correct"]
    assert points[300]["medical_correct"] < points[240]["medical_correct"]
    assert curve["selected_medical_paired_bootstrap_95_ci"][0] <= 0

    assert p10["base"]["correct"] == p10["b2_step240"]["correct"] == 443
    assert p10["improved"] == p10["regressed"] == 10
    assert p10["status"] == "b2_step240_confirmation_not_supported"
    assert p10["mcnemar_exact_two_sided_p"] == 1.0

    for payload in (sft, stage120, curve, p10):
        assert payload["final_access_count"] == 0
        forbidden = {"question", "prompt", "label", "answer", "prediction", "rollout"}
        assert not (forbidden & set(payload))

    for line in (RESULTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        actual = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        assert actual == expected, f"SHA-256 mismatch: {name}"

    print("PUBLIC RESULT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
