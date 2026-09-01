"""Identity-bound unified Controller entry point for P7 Stage-120."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p6_controller_runtime import run_p6_controller


REQUIRED_STEPS = (60, 90, 120)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_route(method_id: str, step: int, checkpoint: Path) -> dict[str, Any]:
    """Return a route only for a complete, resumable, identity-bound checkpoint."""

    if method_id not in {"IDT-v2", "CA-OPD-v2"} or step not in REQUIRED_STEPS:
        raise ValueError("P7 method or registered Controller step differs")
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint / "checkpoint_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest is absent or invalid") from error
    if not (
        manifest.get("complete") is True
        and manifest.get("resume_eligible") is True
        and manifest.get("optimizer_step") == step
        and manifest.get("policy_version") == step
    ):
        raise ValueError("checkpoint is not complete and resume eligible at the registered step")
    weight_path = checkpoint / "adapter_model.safetensors"
    if not weight_path.is_file():
        raise ValueError("checkpoint adapter weights are absent")
    prefix = "IDT" if method_id == "IDT-v2" else "CA"
    return {
        "name": f"{prefix}_step{step}",
        "method_id": method_id,
        "step": step,
        "adapter_path": str(checkpoint),
        "adapter_manifest_path": str(manifest_path),
        "adapter_ordered_sha256": _ordered_adapter_sha256(checkpoint),
        "adapter_weight_sha256": _sha_file(weight_path),
        "adapter_manifest_sha256": _sha_file(manifest_path),
        "checkpoint_package_content_sha256": manifest.get("package_content_sha256"),
        "checkpoint_resume_eligible": True,
    }


def build_route_spec(
    *,
    baseline_routes_path: Path,
    idt_output: Path,
    ca_output: Path,
) -> dict[str, Any]:
    """Combine verified B0/B1/B2 routes with P7 Stage-120 checkpoints."""

    payload = json.loads(Path(baseline_routes_path).read_text(encoding="utf-8"))
    routes = payload.get("routes") if isinstance(payload, Mapping) else None
    if not isinstance(routes, list):
        raise ValueError("baseline Controller routes are absent")
    required_baselines = {"B0", "B1", "B2_step60", "B2_step90", "B2_step120"}
    baseline = [dict(route) for route in routes if route.get("name") in required_baselines]
    if {route["name"] for route in baseline} != required_baselines:
        raise ValueError("required B0/B1/B2 Controller routes differ")
    combined = baseline
    for method_id, output in (("IDT-v2", idt_output), ("CA-OPD-v2", ca_output)):
        for step in REQUIRED_STEPS:
            combined.append(
                checkpoint_route(
                    method_id,
                    step,
                    Path(output) / "formal_checkpoints" / f"step_{step:03d}",
                )
            )
    return {
        "schema_version": 1,
        "artifact_kind": "p7_stage120_controller_route_spec",
        "reference_route": "B0",
        "routes": combined,
        "evaluated_only_after_both_formal_runs": True,
        "controller_access_count": 1,
        "confirmation_access_count": 0,
        "final_access_count": 0,
    }


def run_p7_controller(
    *,
    config_path: Path,
    routes_path: Path,
    output: Path,
    cache_root: Path,
    allow_cache: bool,
) -> dict[str, Any]:  # pragma: no cover - GPU
    """Run the established direct-logit scorer and bind the P7 route set."""

    result = run_p6_controller(
        config_path=config_path,
        routes_path=routes_path,
        output=output,
        cache_root=cache_root,
        allow_cache=allow_cache,
    )
    expected = {
        "B0",
        "B1",
        "B2_step60",
        "B2_step90",
        "B2_step120",
        "IDT_step60",
        "IDT_step90",
        "IDT_step120",
        "CA_step60",
        "CA_step90",
        "CA_step120",
    }
    if set(result["metrics"]) != expected:
        raise ValueError("P7 unified Controller route set differs")
    return result


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--allow-cache", action="store_true")
    args = parser.parse_args(argv)
    result = run_p7_controller(
        config_path=args.config,
        routes_path=args.routes,
        output=args.output,
        cache_root=args.cache_root,
        allow_cache=args.allow_cache,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())


__all__ = ["build_route_spec", "checkpoint_route", "run_p7_controller"]
