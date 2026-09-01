"""Identity-bound P9 Controller route builder and runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.eval.p6_controller_runtime import run_p6_controller
from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint


REGISTERED_STEPS = frozenset({120, 150, 180, 200, 240, 270, 300})
P6_BASELINE_ROUTES_SHA256 = "236c82a435bc2fb5de2f52ccdefae36666e483812ac62f76ac38755fe6bc234e"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checkpoint_route(step: int, checkpoint: Path) -> dict[str, Any]:
    if step not in REGISTERED_STEPS or step == 120:
        raise ValueError("P9 checkpoint route step is not a P9 registered extension")
    manifest = validate_formal_checkpoint(checkpoint)
    if manifest["logical_version"] != step:
        raise ValueError("P9 checkpoint route logical step differs")
    return {
        "name": f"B2_step{step}",
        "method_id": "B2-P9-dose",
        "step": step,
        "adapter_path": str(Path(checkpoint).resolve()),
        "adapter_manifest_path": str((Path(checkpoint) / "checkpoint_manifest.json").resolve()),
        "adapter_ordered_sha256": _ordered_adapter_sha256(checkpoint),
        "adapter_weight_sha256": _sha_file(Path(checkpoint) / "adapter_model.safetensors"),
        "adapter_manifest_sha256": _sha_file(Path(checkpoint) / "checkpoint_manifest.json"),
        "checkpoint_package_content_sha256": manifest["package_content_sha256"],
        "checkpoint_resume_eligible": True,
    }


def build_p9_route_spec(
    *, baseline_routes_path: Path, p9_output: Path, steps: Sequence[int]
) -> dict[str, Any]:
    registered = sorted(set(int(step) for step in steps))
    if 120 not in registered or any(step not in REGISTERED_STEPS for step in registered):
        raise ValueError("P9 requested Controller steps differ")
    if _sha_file(Path(baseline_routes_path)) != P6_BASELINE_ROUTES_SHA256:
        raise ValueError("P9 frozen P6 baseline route source identity differs")
    baseline_payload = json.loads(Path(baseline_routes_path).read_text(encoding="utf-8"))
    baseline = [
        dict(route) for route in baseline_payload["routes"]
        if route.get("name") in {"B0", "B1", "B2_step120"}
    ]
    if {route["name"] for route in baseline} != {"B0", "B1", "B2_step120"}:
        raise ValueError("P9 baseline routes are absent")
    routes = baseline + [
        checkpoint_route(step, Path(p9_output) / "formal_checkpoints" / f"step_{step:03d}")
        for step in registered if step != 120
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "p9_b2_controller_route_spec",
        "reference_route": "B0",
        "routes": routes,
        "baseline_routes_source_sha256": P6_BASELINE_ROUTES_SHA256,
        "route_identity_sha256": _canonical_sha(routes),
        "registered_steps": registered,
        "medical_only_checkpoint_selection": True,
        "confirmation_access_count": 0,
        "final_access_count": 0,
    }


def run_p9_controller(
    *, config_path: Path, routes_path: Path, output: Path, cache_root: Path
) -> dict[str, Any]:  # pragma: no cover - GPU
    result = run_p6_controller(
        config_path=config_path,
        routes_path=routes_path,
        output=output,
        cache_root=cache_root,
        allow_cache=True,
    )
    routes = json.loads(Path(routes_path).read_text(encoding="utf-8"))["routes"]
    if set(result["metrics"]) != {str(route["name"]) for route in routes}:
        raise ValueError("P9 Controller result route set differs")
    return result


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_p9_controller(config_path=args.config, routes_path=args.routes, output=args.output, cache_root=args.cache_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["REGISTERED_STEPS", "build_p9_route_spec", "checkpoint_route", "run_p9_controller"]
