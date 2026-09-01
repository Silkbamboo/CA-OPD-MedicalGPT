"""Persistent controller-result boundary for windowed CA-OPD execution.

This module never evaluates a model. It accepts only deterministic controller
artifacts, derives the frozen B0/B1 targets, and persists enough JSON state for
one GPU training window at a time. Final capability artifacts fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.opd.router import ConstraintAwareRouter, RouterConfig


class CARuntimeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_controller_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    if value.get("capability") != "controller_eval":
        raise CARuntimeError("CA routing accepts controller_eval artifacts only; final is forbidden")
    decode = value.get("decode")
    if (
        not isinstance(decode, Mapping)
        or float(decode.get("temperature", -1)) != 0.0
        or decode.get("do_sample") is not False
        or type(decode.get("seed")) is not int
    ):
        raise CARuntimeError("controller artifact must use deterministic decoding")
    abilities = value.get("accuracy_by_domain")
    if not isinstance(abilities, Mapping) or set(abilities) != {"medical", "general"}:
        raise CARuntimeError("controller artifact must contain medical/general accuracies")
    if any(type(abilities[key]) not in (int, float) or not 0 <= float(abilities[key]) <= 1 for key in abilities):
        raise CARuntimeError("controller accuracies must be in [0,1]")
    manifest_sha = str(value.get("data_manifest_sha256", ""))
    if len(manifest_sha) != 64:
        raise CARuntimeError("controller artifact lacks its manifest SHA")
    return {**value, "_artifact_sha256": _sha256(artifact_path)}


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def initialize_ca_state(
    b0_artifact: str | Path,
    b1_artifact: str | Path,
    router_template: str | Path,
    state_path: str | Path,
    *,
    total_steps: int,
) -> dict[str, Any]:
    """Freeze CA targets from B0 general and B1 medical controller evidence."""

    b0, b1 = _load_controller_artifact(b0_artifact), _load_controller_artifact(b1_artifact)
    if b0["data_manifest_sha256"] != b1["data_manifest_sha256"]:
        raise CARuntimeError("B0/B1 controller manifest SHA differs")
    template = yaml.safe_load(Path(router_template).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise CARuntimeError("router template must be a mapping")
    template.pop("configuration_status", None)
    template["medical_target"] = float(b1["accuracy_by_domain"]["medical"])
    template["general_baseline"] = float(b0["accuracy_by_domain"]["general"])
    config = RouterConfig.from_mapping(template)
    if total_steps < config.window_steps:
        raise CARuntimeError("CA total steps must cover at least one controller window")
    state = {
        "schema_version": 1,
        "status": "ready_for_window",
        "total_steps": int(total_steps),
        "window_steps": config.window_steps,
        "router_config": template,
        "controller_manifest_sha256": b0["data_manifest_sha256"],
        "b0_artifact_sha256": b0["_artifact_sha256"],
        "b1_artifact_sha256": b1["_artifact_sha256"],
        "observations": [],
        "final_artifacts_used": False,
    }
    _atomic(Path(state_path), state)
    return state


def _load_state(state_path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("final_artifacts_used") is not False:
        raise CARuntimeError("invalid or final-contaminated CA state")
    return value


def _replay(state: Mapping[str, Any]) -> ConstraintAwareRouter:
    router = ConstraintAwareRouter(RouterConfig.from_mapping(state["router_config"]), seed=42)
    for observation in state["observations"]:
        router.update(
            float(observation["medical"]),
            float(observation["general"]),
            step=int(observation["completed_step"]),
        )
    return router


def next_ca_window(state_path: str | Path) -> dict[str, Any]:
    state = _load_state(state_path)
    start = len(state["observations"]) * int(state["window_steps"])
    if start >= int(state["total_steps"]):
        raise CARuntimeError("CA run already reached its total step budget")
    router = _replay(state)
    end = min(int(state["total_steps"]), start + int(state["window_steps"]))
    return {
        "start_step": start,
        "end_step": end,
        "p_medical": router.p_medical,
        "p_base": 1.0 - router.p_medical,
    }


def record_ca_controller_result(
    state_path: str | Path,
    controller_artifact: str | Path,
    *,
    completed_step: int,
) -> dict[str, Any]:
    path = Path(state_path)
    state = _load_state(path)
    expected = next_ca_window(path)["end_step"]
    if completed_step != expected:
        raise CARuntimeError(f"controller result step {completed_step} does not close window {expected}")
    artifact = _load_controller_artifact(controller_artifact)
    if artifact["data_manifest_sha256"] != state["controller_manifest_sha256"]:
        raise CARuntimeError("controller result manifest SHA differs from B0/B1")
    state["observations"].append({
        "completed_step": completed_step,
        "medical": float(artifact["accuracy_by_domain"]["medical"]),
        "general": float(artifact["accuracy_by_domain"]["general"]),
        "artifact_sha256": artifact["_artifact_sha256"],
    })
    state["status"] = (
        "completed" if completed_step >= int(state["total_steps"]) else "ready_for_window"
    )
    _atomic(path, state)
    return state


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - operator CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--b0", required=True)
    init.add_argument("--b1", required=True)
    init.add_argument("--router-template", required=True)
    init.add_argument("--state", required=True)
    init.add_argument("--total-steps", required=True, type=int)
    nxt = sub.add_parser("next")
    nxt.add_argument("--state", required=True)
    record = sub.add_parser("record")
    record.add_argument("--state", required=True)
    record.add_argument("--controller-artifact", required=True)
    record.add_argument("--completed-step", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        result = initialize_ca_state(args.b0, args.b1, args.router_template, args.state, total_steps=args.total_steps)
    elif args.command == "next":
        result = next_ca_window(args.state)
    else:
        result = record_ca_controller_result(args.state, args.controller_artifact, completed_step=args.completed_step)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
