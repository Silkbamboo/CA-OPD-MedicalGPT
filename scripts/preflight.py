#!/usr/bin/env python3
"""Preflight gate for a formal (paid) run.

    python scripts/preflight.py --run-config configs/runs/b2_medical_opd_qwen3_1_7b.yaml
    python scripts/preflight.py --run-config ... --with-tests --emit-plan outputs/plans

Checks, in order:

1. **run config schema** - unknown/missing keys, ranges, cross-file consistency;
2. **veRL constraints** - teacher pool sizing, LoRA rollout format, loss-mode
   combinations (replicated locally so a bad config fails here, not after
   ``docker run`` on a rented box);
3. **router config** - loaded and validated if the run uses the constraint-aware router;
4. **data manifest** - file sha256s match, zero pairwise split overlap, and
   ``final_test_access.log`` shows no read during training;
5. **environment** - reports torch/transformers/vllm/verl/GPU and flags anything
   incompatible with the stack ADR (transformers >= 4.51 for Qwen3, etc.);
6. **cost cap** - the estimated cost must be under the declared ceiling;
7. optional **CPU test suite** (``--with-tests``).

Exit code is 0 only if every blocking check passes. Environment findings are
warnings when the box has no GPU (i.e. during planning on the dev machine) and
blocking when a GPU is present (i.e. right before the real run).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.opd.router import RouterConfig  # noqa: E402
from src.opd.run_config import load_run  # noqa: E402
from src.utils.config import ConfigError, load_yaml  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402
from src.utils.run_plan import CostCapExceeded  # noqa: E402

MIN_TRANSFORMERS_FOR_QWEN3 = (4, 51, 0)


class Check:
    def __init__(self, name: str):
        self.name = name
        self.status = "SKIP"
        self.detail = ""
        self.blocking = True

    def ok(self, detail: str = "") -> "Check":
        self.status, self.detail = "PASS", detail
        return self

    def fail(self, detail: str) -> "Check":
        self.status, self.detail = "FAIL", detail
        return self

    def warn(self, detail: str) -> "Check":
        self.status, self.detail = "WARN", detail
        self.blocking = False
        return self

    def __str__(self) -> str:
        return f"{self.status:4}  {self.name}: {self.detail}" if self.detail else f"{self.status:4}  {self.name}"


def check_run_config(path: str) -> Tuple[Check, Optional[Any]]:
    c = Check("run config schema + veRL constraints")
    try:
        loaded = load_run(path)
    except (ConfigError, ValueError) as exc:
        return c.fail(f"{type(exc).__name__}: {exc}"), None
    n_overrides = len(loaded.verl.to_overrides())
    return c.ok(
        f"{loaded.plan.run_id} ({loaded.plan.baseline_id}), "
        f"{len(loaded.verl.teachers)} teacher(s), {n_overrides} veRL overrides"
    ), loaded


def check_router_config(loaded: Any) -> Check:
    c = Check("router config")
    router = loaded.raw["router"]
    if router["kind"] != "constraint_aware":
        return c.warn(f"kind={router['kind']} (no CA router config needed)")
    path = REPO_ROOT / str(router["config_path"])
    if not path.exists():
        return c.fail(f"missing {path}")
    try:
        cfg = RouterConfig.from_mapping(load_yaml(path))
    except (ConfigError, ValueError, TypeError) as exc:
        return c.fail(f"{type(exc).__name__}: {exc}")
    if cfg.window_steps != int(loaded.raw["budget"]["controller_dev_every_steps"]):
        return c.fail(
            f"router.window_steps={cfg.window_steps} disagrees with "
            f"budget.controller_dev_every_steps={loaded.raw['budget']['controller_dev_every_steps']}"
        )
    return c.ok(
        f"K={cfg.window_steps}, p in [{cfg.p_min}, {cfg.p_max}], "
        f"floor=B_G-delta={cfg.general_floor:.3f}"
    )


def check_data_manifest(loaded: Any) -> List[Check]:
    checks: List[Check] = []
    manifest_path = loaded.plan.data_manifest_path
    c = Check("data manifest")
    if not manifest_path:
        checks.append(c.fail("data.manifest is not set; a formal run must pin its data"))
        return checks
    path = Path(manifest_path)
    if not path.exists():
        checks.append(
            c.warn(f"{path} not found - build splits first: python -m src.data.build_splits --config configs/data/base.yaml")
        )
        return checks

    try:
        from src.data.build_splits import verify_manifest

        verify_manifest(path)
        manifest = read_json(path)
    except Exception as exc:  # noqa: BLE001 - report any verification failure
        checks.append(c.fail(f"{type(exc).__name__}: {exc}"))
        return checks
    checks.append(
        c.ok(
            f"seed={manifest['seed']}, splits="
            + ", ".join(f"{k}:{v['count']}" for k, v in sorted(manifest["splits"].items()))
        )
    )

    leak = Check("split isolation")
    overlap = manifest["leakage_report"]["max_pairwise_overlap"]
    if overlap != 0:
        leak.fail(f"max pairwise split overlap = {overlap} (must be 0)")
    else:
        leak.ok("zero pairwise overlap on sample_id and content_hash")
    checks.append(leak)

    audit = Check("final-test isolation")
    log = path.parent / "final_test_access.log"
    if not log.exists():
        audit.ok("final_test has never been read")
    else:
        lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        audit.fail(
            f"final_test_access.log has {len(lines)} entry/entries; final test must not be read "
            f"before the checkpoint is frozen. Last: {lines[-1][:120]}"
        ) if lines else audit.ok("log exists but is empty")
    checks.append(audit)
    return checks


def check_environment(strict: bool) -> List[Check]:
    """Report the stack versions without importing the heavy packages.

    ``importlib.metadata`` reads dist-info, and ``nvidia-smi`` answers the GPU
    question, so preflight costs a few MiB instead of the ~350 MiB a torch import
    needs. That matters here: this container shares a 2 GiB cgroup with the editor
    runtime, and an unnecessary torch import was enough to get the gate OOM-killed.
    """
    from importlib.metadata import PackageNotFoundError, version as dist_version

    checks: List[Check] = []
    versions: Dict[str, str] = {}
    for pkg in ("torch", "transformers", "vllm", "verl", "ray", "peft", "trl"):
        try:
            versions[pkg] = dist_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
        except Exception as exc:  # noqa: BLE001
            versions[pkg] = f"unknown ({type(exc).__name__})"

    gpu = Check("GPU availability")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if proc.returncode == 0 and lines:
            gpu.ok(f"{len(lines)} GPU(s): {'; '.join(lines)}")
        else:
            gpu.warn("no CUDA device (planning mode; the real run needs 2 GPUs)")
    except (OSError, subprocess.SubprocessError) as exc:
        gpu.warn(f"nvidia-smi unavailable: {type(exc).__name__}")
    checks.append(gpu)

    tf = Check("transformers >= 4.51 (Qwen3)")
    raw = versions.get("transformers", "not-installed")
    if raw.startswith("not-installed") or raw.startswith("unknown"):
        (tf.fail if strict else tf.warn)("transformers is not installed")
    else:
        parts = []
        for token in raw.split(".")[:3]:
            digits = "".join(ch for ch in token if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) >= MIN_TRANSFORMERS_FOR_QWEN3:
            tf.ok(raw)
        else:
            (tf.fail if strict else tf.warn)(
                f"{raw} cannot load Qwen3 (config declares transformers_version 4.51.0)"
            )
    checks.append(tf)

    for pkg in ("vllm", "verl", "ray"):
        c = Check(f"{pkg} installed")
        if versions[pkg] == "not-installed":
            (c.fail if strict else c.warn)("not installed (required for the formal run)")
        else:
            c.ok(versions[pkg])
        checks.append(c)

    checks.append(Check("package versions").ok(json.dumps(versions, sort_keys=True)))
    return checks


def check_cost(loaded: Any) -> Check:
    c = Check("cost cap")
    try:
        cost = loaded.plan.check_cost_cap()
    except CostCapExceeded as exc:
        return c.fail(str(exc))
    d = loaded.plan.as_dict()["derived"]
    return c.ok(
        f"{cost:.2f} RMB <= cap {loaded.plan.cost_cap_rmb:.2f} "
        f"({d['estimated_gpu_hours']} GPU-h, {d['generated_tokens']:,} generated tokens)"
    )


def check_throughput_provenance(loaded: Any) -> Check:
    c = Check("throughput provenance")
    if loaded.plan.throughput.measured:
        return c.ok(loaded.plan.throughput.source)
    return c.warn(f"estimate uses assumed throughput ({loaded.plan.throughput.source})")


def run_cpu_tests() -> Check:
    c = Check("CPU test suite")
    script = REPO_ROOT / "scripts" / "run_cpu_checks.sh"
    if not script.exists():
        return c.fail("scripts/run_cpu_checks.sh missing")
    proc = subprocess.run(["bash", str(script), "--quick"], capture_output=True, text=True, cwd=REPO_ROOT)
    tail = [l for l in proc.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
    if proc.returncode == 0:
        return c.ok(f"{len(tail)} groups passed")
    return c.fail("; ".join(l for l in tail if l.startswith("FAIL")) or proc.stdout[-300:])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight gate for a formal run")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--with-tests", action="store_true", help="also run the CPU test suite")
    parser.add_argument("--emit-plan", default=None, help="directory to write run_plan.md / run_plan.json")
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="treat missing vllm/verl/transformers as blocking (use on the training box)",
    )
    args = parser.parse_args(argv)

    checks: List[Check] = []
    cfg_check, loaded = check_run_config(args.run_config)
    checks.append(cfg_check)

    if loaded is not None:
        checks.append(check_router_config(loaded))
        checks.extend(check_data_manifest(loaded))
        checks.append(check_throughput_provenance(loaded))
        checks.append(check_cost(loaded))
    strict = args.strict_env or bool(os.environ.get("CA_OPD_STRICT_ENV"))
    checks.extend(check_environment(strict))
    if args.with_tests:
        checks.append(run_cpu_tests())

    print("=== CA-OPD preflight ==================================================")
    print(f"run config: {args.run_config}")
    print()
    for c in checks:
        print(c)
    blocking_failures = [c for c in checks if c.status == "FAIL" and c.blocking]
    warnings = [c for c in checks if c.status == "WARN"]
    print()
    print(f"{len(checks)} checks: "
          f"{sum(c.status == 'PASS' for c in checks)} pass, {len(warnings)} warn, "
          f"{sum(c.status == 'FAIL' for c in checks)} fail")

    if loaded is not None and args.emit_plan:
        out = Path(args.emit_plan) / loaded.plan.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_plan.md").write_text(loaded.plan.to_markdown(), encoding="utf-8")
        write_json(out / "run_plan.json", loaded.as_dict())
        (out / "verl_command.sh").write_text(loaded.verl.to_command(), encoding="utf-8")
        print(f"\nplan written to {out}/ (run_plan.md, run_plan.json, verl_command.sh)")

    if blocking_failures:
        print("\nGATE: BLOCKED - fix the FAIL items above before starting a paid run")
        return 1
    if warnings:
        print("\nGATE: PASSED WITH WARNINGS - review them, then get explicit approval before spending")
        return 0
    print("\nGATE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
