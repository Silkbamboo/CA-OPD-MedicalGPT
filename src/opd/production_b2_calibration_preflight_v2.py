"""P4.8b semantic-first preflight for the replacement B2 package."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping

from src.opd.production_b2_calibration_contract_v1 import MINIMUM_DISK_BYTES
from src.opd.production_b2_calibration_package_v2 import (
    B2CalibrationPackageV2Error,
    pre_model_semantic_preflight,
)


EXPECTED_BRANCH = "codex/p4-8b-frozen-manifest-binding"
PROJECTED_INCREMENT_BYTES = 4 * 1024**3


class B2CalibrationPreflightV2Error(RuntimeError):
    """Raised before GPU/model/session construction when P4.8b is unsafe."""


def _fail(message: str) -> None:
    raise B2CalibrationPreflightV2Error(message)


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


def _default_gpu_probe() -> Mapping[str, Any]:
    # Imported only after the semantic, package, git, output, and disk gates.
    from src.opd.production_b2_calibration_preflight_v1 import _default_gpu_probe as probe

    return probe()


def _verify_expected(audit: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    fields = (
        "package_content_sha256",
        "package_index_sha256",
        "authorization_sha256",
        "config_sha256",
        "run_card_sha256",
    )
    for field in fields:
        if audit.get(field) != expected.get(field):
            _fail(f"replacement package expected {field} differs")
    if audit.get("schedule", {}).get("schedule_sha256") != expected.get(
        "schedule_sha256"
    ):
        _fail("replacement package expected schedule SHA differs")
    if audit.get("data_authority", {}).get("manifest_sha256") != expected.get(
        "manifest_sha256"
    ):
        _fail("replacement package expected manifest SHA differs")


def preflight_b2_calibration_v2(
    package_dir: str | Path,
    *,
    output_dir: str | Path,
    canonical_manifest_path: str | Path,
    expected: Mapping[str, Any],
    mode: str,
    git_probe: Callable[[], Mapping[str, Any]] | None = None,
    disk_free_probe: Callable[[Path], int] | None = None,
    gpu_probe: Callable[[], Mapping[str, Any]] | None = None,
    expected_branch: str = EXPECTED_BRANCH,
    expected_commit: str | None = None,
    projected_increment_bytes: int = PROJECTED_INCREMENT_BYTES,
    allow_dirty_for_development: bool = False,
) -> dict[str, Any]:
    """Run the exact production semantic gate before every host/runtime probe."""

    if mode not in {"dry-run", "host-preflight", "execute"}:
        _fail("unknown P4.8b preflight mode")
    try:
        semantic = pre_model_semantic_preflight(
            package_dir,
            canonical_manifest_path=canonical_manifest_path,
        )
    except B2CalibrationPackageV2Error as error:
        raise B2CalibrationPreflightV2Error(
            f"replacement package semantic gate failed: {error}"
        ) from error
    audit = semantic["audit"]
    _verify_expected(audit, expected)
    output = Path(output_dir).resolve()
    if output.exists() or output.is_symlink():
        _fail("P4.8b GPU output must be fresh")
    git = dict((git_probe or _default_git_probe)())
    if not allow_dirty_for_development and not (
        git.get("branch") == expected_branch
        and git.get("clean") is True
        and isinstance(git.get("head"), str)
        and len(git["head"]) == 40
        and (expected_commit is None or git["head"] == expected_commit)
    ):
        _fail("P4.8b requires the exact clean committed worktree")
    free = int(
        (disk_free_probe or (lambda path: shutil.disk_usage(path.parent).free))(output)
    )
    if free - int(projected_increment_bytes) <= MINIMUM_DISK_BYTES:
        _fail("projected persistent disk is not strictly above 10 GiB")
    gpu_host: dict[str, Any] | None = None
    gpus: list[dict[str, Any]] | None = None
    if mode in {"host-preflight", "execute"}:
        observed = (gpu_probe or _default_gpu_probe)()
        if not isinstance(observed, Mapping):
            _fail("P4.8b host requires exactly two idle RTX 3090 GPUs")
        gpu_host = dict(observed)
        raw_gpus = gpu_host.get("gpus")
        if not isinstance(raw_gpus, list):
            _fail("P4.8b host requires exactly two idle RTX 3090 GPUs")
        gpus = [dict(item) for item in raw_gpus if isinstance(item, Mapping)]
        if not (
            len(gpus) == 2
            and all(item.get("name") == "NVIDIA GeForce RTX 3090" for item in gpus)
            and all(int(item.get("total_mib", 0)) >= 24000 for item in gpus)
            and all(int(item.get("used_mib", 1_000_000)) <= 16 for item in gpus)
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
            _fail("P4.8b host requires exactly two idle RTX 3090 GPUs")
    semantic_public = {
        key: value for key, value in semantic.items() if key != "audit"
    }
    return {
        "schema_version": 2,
        "artifact_kind": "p4_8b_b2_calibration_preflight_v2",
        "status": "ready_waiting_for_gpu_b2_calibration_revalidation",
        "mode": mode,
        "package_content_sha256": audit["package_content_sha256"],
        "package_index_sha256": audit["package_index_sha256"],
        "manifest_sha256": audit["data_authority"]["manifest_sha256"],
        "schedule_sha256": audit["schedule"]["schedule_sha256"],
        "semantic_gate": semantic_public,
        "selected_response_length": 768,
        "optimizer_steps": 20,
        "seed": 42,
        "student_initialization": "fresh_base_plus_fresh_zero_lora_v1",
        "git": git,
        "disk_free_bytes": free,
        "projected_increment_bytes": projected_increment_bytes,
        "gpus": gpus,
        "gpu_host": gpu_host,
        "B2_authorized": True,
        "B2_calibration_started": False,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }


__all__ = [
    "B2CalibrationPreflightV2Error",
    "EXPECTED_BRANCH",
    "PROJECTED_INCREMENT_BYTES",
    "preflight_b2_calibration_v2",
]
