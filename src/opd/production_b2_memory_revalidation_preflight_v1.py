"""Semantic-first host preflight for the P4.8e revalidation overlay."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

from src.opd.production_b2_calibration_contract_v1 import MINIMUM_DISK_BYTES
from src.opd.production_b2_calibration_preflight_v1 import (
    _default_gpu_probe,
)
from src.opd.production_b2_memory_revalidation_package_v1 import (
    EXPECTED_PARENT_CONTENT_SHA256,
    PACKAGE_VERSION,
    RUN_ID,
)


EXPECTED_BRANCH = "codex/p4-8e-b2-memory-revalidation"
PROJECTED_INCREMENT_BYTES = 5 * 1024**3


class B2MemoryRevalidationPreflightV1Error(RuntimeError):
    """P4.8e is unsafe to start before any model load."""


def _fail(message: str) -> None:
    raise B2MemoryRevalidationPreflightV1Error(message)


def _default_git_probe() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"branch": branch, "head": head, "clean": not bool(status)}


def _is_ancestor(commit: str, head: str) -> bool:
    root = Path(__file__).resolve().parents[2]
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def preflight_b2_memory_revalidation_v1(
    package_audit: Mapping[str, Any],
    *,
    mode: str,
    git_probe: Callable[[], Mapping[str, Any]] | None = None,
    disk_free_probe: Callable[[Path], int] | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] | None = None,
    projected_increment_bytes: int = PROJECTED_INCREMENT_BYTES,
    allow_dirty_for_development: bool = False,
) -> dict[str, Any]:
    if mode not in {"dry-run", "host-preflight", "execute"}:
        _fail("unknown P4.8e preflight mode")
    schedule = package_audit.get("schedule")
    authority = package_audit.get("data_authority")
    contract = package_audit.get("memory_revalidation_contract")
    if not all(isinstance(value, Mapping) for value in (schedule, authority, contract)):
        _fail("P4.8e verified package audit is incomplete")
    if not (
        package_audit.get("package_version") == PACKAGE_VERSION
        and package_audit.get("parent_package_content_sha256")
        == EXPECTED_PARENT_CONTENT_SHA256
        and package_audit.get("runtime_run_id") == RUN_ID
        and schedule.get("slot_count") == 80
        and authority.get("manifest_sha256")
        == "9f1d096d06b635737e1b90be3b92d6de32fd64b03fbcd97813e42d0a2ee88a99"
        and schedule.get("schedule_sha256")
        == "4567ebb38972c1d37936a77590b5b6d28a6b6c234297fd85c82dacdad1926d88"
        and contract.get("minimum_headroom_bytes") == 1024**3
        and contract.get("B2_formal_authorized") is False
    ):
        _fail("P4.8e package/science identity differs")

    output = Path(str(package_audit.get("runtime_output_dir", ""))).resolve()
    if output.exists() or output.is_symlink():
        _fail("P4.8e GPU output must be fresh")
    git = dict((git_probe or _default_git_probe)())
    ancestor = (
        bool(git.get("package_code_commit_is_ancestor"))
        if git_probe is not None
        else _is_ancestor(str(package_audit.get("code_git_commit", "")), str(git.get("head", "")))
    )
    if not allow_dirty_for_development and not (
        git.get("branch") == EXPECTED_BRANCH
        and git.get("clean") is True
        and isinstance(git.get("head"), str)
        and len(git["head"]) == 40
        and ancestor
    ):
        _fail("P4.8e requires a clean P4.8e branch containing the package code commit")

    free = int(
        (disk_free_probe or (lambda path: shutil.disk_usage(path.parent).free))(
            output
        )
    )
    if free - int(projected_increment_bytes) <= MINIMUM_DISK_BYTES:
        _fail("projected persistent disk is not strictly above 10 GiB")

    gpu_host: dict[str, Any] | None = None
    gpus: list[dict[str, Any]] | None = None
    if mode in {"host-preflight", "execute"}:
        observed = (gpu_probe or _default_gpu_probe)()
        if not isinstance(observed, Mapping):
            _fail("P4.8e host requires exactly two idle RTX 3090 GPUs")
        gpu_host = dict(observed)
        raw_gpus = gpu_host.get("gpus")
        gpus = [dict(value) for value in raw_gpus] if isinstance(raw_gpus, list) else []
        if not (
            len(gpus) == 2
            and all(gpu.get("name") == "NVIDIA GeForce RTX 3090" for gpu in gpus)
            and all(int(gpu.get("total_mib", 0)) >= 24000 for gpu in gpus)
            and all(int(gpu.get("used_mib", 1_000_000)) <= 16 for gpu in gpus)
            and gpu_host.get("compute_processes") == []
            and gpu_host.get("residual_worker_pids") == []
            and isinstance(gpu_host.get("driver_version"), str)
            and bool(gpu_host["driver_version"])
            and isinstance(gpu_host.get("cuda_version"), str)
            and bool(gpu_host["cuda_version"])
            and isinstance(gpu_host.get("topology"), str)
            and "GPU0" in gpu_host["topology"]
            and "GPU1" in gpu_host["topology"]
        ):
            _fail("P4.8e host requires exactly two idle RTX 3090 GPUs")

    return {
        "schema_version": 1,
        "artifact_kind": "p4_8e_b2_memory_revalidation_preflight_v1",
        "status": "ready_for_p4_8e_gpu_gates",
        "mode": mode,
        "package_version": PACKAGE_VERSION,
        "package_content_sha256": package_audit["package_content_sha256"],
        "package_index_sha256": package_audit["package_index_sha256"],
        "parent_package_content_sha256": package_audit[
            "parent_package_content_sha256"
        ],
        "manifest_sha256": authority["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_slot_count": 80,
        "selected_response_length": 1024,
        "optimizer_steps": 20,
        "physical_microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 4,
        "git": {**git, "package_code_commit_is_ancestor": ancestor},
        "disk_free_bytes": free,
        "projected_increment_bytes": int(projected_increment_bytes),
        "gpus": gpus,
        "gpu_host": gpu_host,
        "gpu_assignment": {"student": 0, "teacher": 1},
        "isolation_access_counts": {
            "final": 0,
            "controller": 0,
            "confirmation": 0,
            "labels": 0,
        },
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
    }


__all__ = [
    "B2MemoryRevalidationPreflightV1Error",
    "EXPECTED_BRANCH",
    "PROJECTED_INCREMENT_BYTES",
    "preflight_b2_memory_revalidation_v1",
]
