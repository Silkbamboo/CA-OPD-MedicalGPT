"""Run identity and metadata capture.

docs/METHOD.md §15 lists exactly what every run must persist. This module
produces ``metadata.json`` so no run can be reported without git SHA, config
path, seed, package versions and hardware info.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .io import ensure_dir, write_json

_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "vllm",
    "verl",
    "ray",
    "numpy",
)


def git_sha(repo_root: str | Path | None = None) -> str:
    """Current commit SHA, or ``unknown`` when git is unavailable."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def git_dirty(repo_root: str | Path | None = None) -> Optional[bool]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - absence is information, not an error
            versions[name] = "not-installed"
    return versions


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": None,
        "gpus": [],
        "cuda_available": False,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:  # noqa: BLE001
        pass
    return info


def make_run_id(prefix: str, seed: int, timestamp: Optional[float] = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp or time.time()))
    return f"{prefix}-s{seed}-{ts}"


@dataclass
class RunMetadata:
    """Everything §15 demands, minus the metrics themselves."""

    run_id: str
    purpose: str
    baseline_id: str
    config_path: str
    seed: int
    model: str
    data_manifest_path: Optional[str] = None
    data_manifest_hash: Optional[str] = None
    notes: str = ""
    cost_limit_rmb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self, repo_root: str | Path | None = None) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "purpose": self.purpose,
            "baseline_id": self.baseline_id,
            "config_path": self.config_path,
            "seed": self.seed,
            "model": self.model,
            "data_manifest_path": self.data_manifest_path,
            "data_manifest_hash": self.data_manifest_hash,
            "notes": self.notes,
            "cost_limit_rmb": self.cost_limit_rmb,
            "git_sha": git_sha(repo_root),
            "git_dirty": git_dirty(repo_root),
            "packages": package_versions(),
            "hardware": hardware_info(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **self.extra,
        }

    def save(self, run_dir: str | Path, repo_root: str | Path | None = None) -> Path:
        ensure_dir(run_dir)
        return write_json(Path(run_dir) / "metadata.json", self.as_dict(repo_root))


def write_run_metadata(run_dir: str | Path, metadata: RunMetadata | Mapping[str, Any]) -> Path:
    if isinstance(metadata, RunMetadata):
        return metadata.save(run_dir)
    return write_json(Path(run_dir) / "metadata.json", dict(metadata))
