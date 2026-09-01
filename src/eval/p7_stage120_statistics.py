"""CPU-only statistics, Pareto analysis and preregistered P7 scale decision."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.eval.paired_stats import paired_comparison


METHOD_ROUTES = {
    "IDT-v2": ("IDT_step60", "IDT_step90", "IDT_step120"),
    "CA-OPD-v2": ("CA_step60", "CA_step90", "CA_step120"),
}


def pareto_frontier(
    metrics: Mapping[str, Mapping[str, Any]], routes: Sequence[str]
) -> list[str]:
    """Return deterministic non-dominated Medical/General route names."""

    frontier: list[str] = []
    for route in sorted(routes):
        medical = float(metrics[route]["medical_accuracy"])
        general = float(metrics[route]["general_micro_accuracy"])
        dominated = False
        for other in routes:
            if other == route:
                continue
            other_medical = float(metrics[other]["medical_accuracy"])
            other_general = float(metrics[other]["general_micro_accuracy"])
            if (
                other_medical >= medical
                and other_general >= general
                and (other_medical > medical or other_general > general)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(route)
    return frontier


def _select(
    metrics: Mapping[str, Mapping[str, Any]], routes: Sequence[str], threshold: float
) -> dict[str, Any]:
    feasible = [
        route
        for route in routes
        if float(metrics[route]["general_micro_accuracy"]) >= threshold
    ]
    if not feasible:
        return {
            "status": "constraint_not_met",
            "selected_checkpoint": None,
            "feasible_steps": [],
            "general_constraint_threshold": threshold,
        }
    selected = min(
        feasible,
        key=lambda route: (
            -float(metrics[route]["medical_accuracy"]),
            int(route.rsplit("step", 1)[1]),
        ),
    )
    return {
        "status": "selected",
        "selected_checkpoint": selected,
        "selected_step": int(selected.rsplit("step", 1)[1]),
        "feasible_steps": sorted(int(route.rsplit("step", 1)[1]) for route in feasible),
        "general_constraint_threshold": threshold,
        "tie_break": "earlier_checkpoint",
    }


def summarize_stage120(
    *,
    metrics: Mapping[str, Mapping[str, Any]],
    scored_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    gpu_hours: Mapping[str, float],
    general_constraint_delta: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Build all result-dependent Stage-120 analyses without final access."""

    b0 = metrics["B0"]
    threshold = float(b0["general_micro_accuracy"]) - float(general_constraint_delta)
    selection = {
        method: _select(metrics, routes, threshold)
        for method, routes in METHOD_ROUTES.items()
    }
    trajectories: dict[str, list[dict[str, Any]]] = {}
    feasible_ratio: dict[str, float] = {}
    first_feasible: dict[str, Any] = {}
    for method, routes in METHOD_ROUTES.items():
        trajectories[method] = [
            {
                "route": route,
                "step": int(route.rsplit("step", 1)[1]),
                "medical_accuracy": float(metrics[route]["medical_accuracy"]),
                "general_micro_accuracy": float(metrics[route]["general_micro_accuracy"]),
                "medical_delta_vs_b0": float(metrics[route]["medical_accuracy"])
                - float(b0["medical_accuracy"]),
                "general_delta_vs_b0": float(metrics[route]["general_micro_accuracy"])
                - float(b0["general_micro_accuracy"]),
                "feasible": float(metrics[route]["general_micro_accuracy"]) >= threshold,
                "gpu_hours": float(gpu_hours[route]),
            }
            for route in routes
        ]
        feasible = [item for item in trajectories[method] if item["feasible"]]
        feasible_ratio[method] = len(feasible) / len(routes)
        first_feasible[method] = (
            min(feasible, key=lambda item: item["step"])
            if feasible
            else {"step": None, "gpu_hours": None, "route": None}
        )

    ca_minus_idt: dict[str, Any] = {}
    for step in (60, 90, 120):
        comparison = paired_comparison(
            scored_rows[f"IDT_step{step}"],
            scored_rows[f"CA_step{step}"],
            seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        ca_minus_idt[f"step{step}"] = {
            "medical": comparison["domains"]["medical"],
            "general": comparison["domains"].get("general"),
            "same_sample_ids": comparison["same_sample_ids"],
        }

    all_method_routes = tuple(route for routes in METHOD_ROUTES.values() for route in routes)
    return {
        "schema_version": 1,
        "artifact_kind": "p7_stage120_statistics",
        "general_constraint_threshold": threshold,
        "selection": selection,
        "trajectories": trajectories,
        "pareto_frontier": pareto_frontier(metrics, all_method_routes),
        "feasible_checkpoint_ratio": feasible_ratio,
        "first_feasible": first_feasible,
        "ca_minus_idt_same_step": ca_minus_idt,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }


def execute_decision_state_machine(inputs: Mapping[str, bool]) -> dict[str, Any]:
    """Apply the preregistered precedence without launching any training."""

    required = {
        "repair_before_scale",
        "close_at_120",
        "recommend_b2_scale_to_300",
        "recommend_idt_ca_scale_to_300",
        "stop_no_scale",
    }
    if set(inputs) != required or not all(isinstance(inputs[key], bool) for key in required):
        raise ValueError("P7 decision inputs differ from the preregistered state set")
    if inputs["repair_before_scale"]:
        primary = "repair_before_scale"
    elif inputs["close_at_120"]:
        primary = "close_at_120"
    elif inputs["recommend_b2_scale_to_300"]:
        primary = "recommend_b2_scale_to_300"
    elif inputs["recommend_idt_ca_scale_to_300"]:
        primary = "recommend_idt_ca_scale_to_300"
    else:
        primary = "stop_no_scale"

    scale: list[str] = []
    if not inputs["repair_before_scale"] and not inputs["close_at_120"]:
        if inputs["recommend_b2_scale_to_300"]:
            scale.append("B2")
        if inputs["recommend_idt_ca_scale_to_300"]:
            scale.extend(("IDT-v2", "CA-OPD-v2"))
    if len(scale) == 3:
        completion = "stage120_complete_mixed_recommendation"
    elif scale == ["B2"]:
        completion = "stage120_complete_scale_b2_recommended"
    elif scale == ["IDT-v2", "CA-OPD-v2"]:
        completion = "stage120_complete_scale_idt_ca_recommended"
    elif primary in {"close_at_120", "stop_no_scale"}:
        completion = "stage120_complete_close_recommended"
    else:
        completion = "blocked_training_integrity"
    return {
        "primary_state": primary,
        "all_true_states": [key for key in required if inputs[key]],
        "recommended_scale_methods": scale,
        "completion_status": completion,
        "automatic_300_launch": False,
        "automatic_final_access": False,
    }


__all__ = ["execute_decision_state_machine", "pareto_frontier", "summarize_stage120"]
