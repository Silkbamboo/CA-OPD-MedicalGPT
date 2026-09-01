"""Fail-closed, JSON run-card contracts for the prepared Qwen3-4B experiment family."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_REQUIRED = {
    "run_id",
    "stage",
    "method",
    "status",
    "git_sha",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "data_manifest_path",
    "data_manifest_sha256",
    "seed",
    "lora",
    "max_lengths",
    "batch",
    "optimizer",
    "max_steps",
    "candidate_max_steps",
    "checkpoint_policy",
    "controller_window",
    "controller_plan",
    "final_plan",
    "software_stack_id",
    "maximum_runtime_hours",
    "maximum_cost_cny",
    "price_cny_per_hour",
    "estimated_cost_cny",
    "actual_cost_cny",
    "disk_gate_gib",
    "success_conditions",
    "abort_conditions",
    "hardware_calibration_status",
}
_FAIRNESS_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "data_manifest_sha256",
    "seed",
    "lora",
    "max_lengths",
    "batch",
    "optimizer",
    "checkpoint_policy",
    "controller_window",
    "software_stack_id",
)


class RunCardError(ValueError):
    pass


def _validate_card(card: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(_REQUIRED - set(card))
    if missing:
        raise RunCardError(f"run card missing fields: {missing}")
    if card["status"] != "prepared_not_started":
        raise RunCardError("run card status must be prepared_not_started")
    if card["model_id"] != MODEL_ID or card["model_revision"] != MODEL_REVISION:
        raise RunCardError("run card is not bound to the approved Qwen3-4B revision")
    if card["tokenizer_revision"] != MODEL_REVISION:
        raise RunCardError("tokenizer revision must equal the Qwen3-4B revision")
    if _HEX40.fullmatch(str(card["git_sha"])) is None:
        raise RunCardError("git_sha must be a 40-hex preparation commit")
    if _HEX64.fullmatch(str(card["data_manifest_sha256"])) is None:
        raise RunCardError("data manifest SHA must be immutable")
    if card["seed"] != 42:
        raise RunCardError("the prepared experiment family uses seed=42")
    if any(card[key] is not None for key in ("price_cny_per_hour", "estimated_cost_cny", "actual_cost_cny")):
        raise RunCardError(
            "price_cny_per_hour, estimated_cost_cny, and actual_cost_cny must stay null before the GPU run"
        )
    if card["hardware_calibration_status"] != "candidate_pending_gpu_calibration":
        raise RunCardError("unstarted run must remain candidate_pending_gpu_calibration")
    if not isinstance(card["success_conditions"], list) or not card["success_conditions"]:
        raise RunCardError("success conditions must be explicit")
    if not isinstance(card["abort_conditions"], list) or not card["abort_conditions"]:
        raise RunCardError("abort conditions must be explicit")
    return dict(card)


def load_run_card(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunCardError("run card root must be an object")
    return _validate_card(payload)


def validate_run_card_family(cards: Sequence[Mapping[str, Any]]) -> None:
    normalized = [_validate_card(card) for card in cards]
    run_ids = [str(card["run_id"]) for card in normalized]
    if len(run_ids) != len(set(run_ids)):
        raise RunCardError("run IDs must be unique")
    methods = [card for card in normalized if card["method"] in {"sar", "idt_1to1", "ca_opd"}]
    if len(methods) != 3:
        raise RunCardError("fairness family requires SAR, IDT 1:1, and CA-OPD")
    reference = methods[0]
    for card in methods[1:]:
        for field in _FAIRNESS_FIELDS:
            if card[field] != reference[field]:
                raise RunCardError(f"fairness drift in {field}")
    if any(card["max_steps"] is not None or card["candidate_max_steps"] != [120, 150] for card in methods):
        raise RunCardError("formal OPD step count must remain pending the 20-step GPU calibration")
