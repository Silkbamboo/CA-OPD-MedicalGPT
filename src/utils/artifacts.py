"""Standard run artifact schema and side-effect-limited writer."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.schema import DATA_PROTOCOL_VERSION


RUN_STAGES = frozenset({"data", "sft", "controller_eval", "opd", "final"})
RUN_STATUSES = frozenset({"planned", "dry_run", "running", "completed", "failed"})
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactError(ValueError):
    """Raised when run metadata would be incomplete or misleading."""


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    stage: str
    git_sha: str
    dirty_worktree: bool
    model_id: str
    model_revision: str
    tokenizer_revision: str
    data_protocol_version: str
    data_manifest_sha256: str
    seed: int
    package_versions: Mapping[str, str]
    cuda_info: Mapping[str, Any] | None
    gpu_info: list[Mapping[str, Any]]
    hardware_status: str
    start_time: str
    end_time: str | None
    status: str
    failure_reason: str | None
    estimated_cost_cny: float
    actual_cost_cny: float | None

    def __post_init__(self) -> None:
        if not _SAFE_RUN_ID.fullmatch(self.run_id):
            raise ArtifactError("run_id is empty, unsafe, or too long")
        if self.stage not in RUN_STAGES:
            raise ArtifactError(f"unsupported run stage: {self.stage}")
        if self.status not in RUN_STATUSES:
            raise ArtifactError(f"unsupported run status: {self.status}")
        if len(self.git_sha) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_sha
        ):
            raise ArtifactError("git_sha must be a 40-character lowercase hex SHA")
        if type(self.dirty_worktree) is not bool:
            raise ArtifactError("dirty_worktree must be boolean")
        if not self.model_id or not self.model_revision or not self.tokenizer_revision:
            raise ArtifactError("model/tokenizer identities must be explicit")
        if self.data_protocol_version != DATA_PROTOCOL_VERSION:
            raise ArtifactError("data_protocol_version must be ca-opd-data-v2")
        if len(self.data_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.data_manifest_sha256
        ):
            raise ArtifactError("data_manifest_sha256 must be a lowercase SHA-256")
        if type(self.seed) is not int:
            raise ArtifactError("seed must be an integer")
        if not isinstance(self.package_versions, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.package_versions.items()
        ):
            raise ArtifactError("package_versions must be a string mapping")
        if not isinstance(self.gpu_info, list) or not self.hardware_status:
            raise ArtifactError("GPU inventory and hardware_status must be explicit")
        if not self.gpu_info and self.cuda_info is None and "not_available" not in self.hardware_status:
            raise ArtifactError("CPU/dry-run hardware absence needs an explicit reason")
        if not self.start_time:
            raise ArtifactError("start_time is required")
        if self.status in {"completed", "failed"} and not self.end_time:
            raise ArtifactError("completed/failed runs require end_time")
        if self.status == "failed" and not self.failure_reason:
            raise ArtifactError("failed runs require failure_reason")
        if self.status != "failed" and self.failure_reason is not None:
            raise ArtifactError("failure_reason is only valid for failed runs")
        if type(self.estimated_cost_cny) not in (int, float) or self.estimated_cost_cny < 0:
            raise ArtifactError("estimated_cost_cny must be non-negative")
        if self.actual_cost_cny is not None and (
            type(self.actual_cost_cny) not in (int, float)
            or self.actual_cost_cny < 0
        ):
            raise ArtifactError("actual_cost_cny must be null or non-negative")
        if self.status in {"planned", "dry_run", "running"} and self.actual_cost_cny is not None:
            raise ArtifactError(
                "actual_cost_cny must be null before a real run has ended"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["package_versions"] = dict(sorted(self.package_versions.items()))
        return payload


@dataclass(frozen=True)
class RunArtifactPaths:
    run_dir: Path
    config: Path
    metadata: Path
    data_manifest: Path
    metrics: Path
    summary: Path
    stdout: Path
    checkpoint_index: Path
    cost: Path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactError("artifact must be finite JSON") from error


def initialize_run_artifacts(
    root: str | Path,
    *,
    config: Mapping[str, Any],
    metadata: RunMetadata,
    data_manifest: Mapping[str, Any],
) -> RunArtifactPaths:
    """Create the complete empty inventory for one planned/dry run.

    The writer refuses any existing run directory and never starts a model,
    trainer, evaluator, GPU service, or paid resource.
    """

    if not isinstance(config, Mapping) or not config:
        raise ArtifactError("config must be a non-empty mapping")
    if (
        data_manifest.get("schema_version") != 2
        or data_manifest.get("data_protocol_version") != DATA_PROTOCOL_VERSION
        or not isinstance(data_manifest.get("roles"), Mapping)
        or not data_manifest["roles"]
    ):
        raise ArtifactError("data_manifest must be a non-empty Data Protocol v2 manifest")
    run_dir = Path(root) / metadata.run_id
    if run_dir.exists():
        raise ArtifactError(f"run directory already exists: {run_dir}")
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    paths = RunArtifactPaths(
        run_dir=run_dir,
        config=run_dir / "config.yaml",
        metadata=run_dir / "metadata.json",
        data_manifest=run_dir / "data_manifest.json",
        metrics=run_dir / "metrics.jsonl",
        summary=run_dir / "summary.json",
        stdout=run_dir / "stdout.log",
        checkpoint_index=checkpoints / "index.json",
        cost=run_dir / "cost.json",
    )

    import yaml

    paths.config.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    paths.metadata.write_bytes(_json_bytes(metadata.to_dict()))
    paths.data_manifest.write_bytes(_json_bytes(data_manifest))
    paths.metrics.write_bytes(b"")
    paths.stdout.write_bytes(b"")
    paths.summary.write_bytes(
        _json_bytes(
            {
                "run_id": metadata.run_id,
                "stage": metadata.stage,
                "status": metadata.status,
                "metrics_summary": None,
                "result_claims": [],
            }
        )
    )
    paths.checkpoint_index.write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "run_id": metadata.run_id,
                "checkpoints": [],
            }
        )
    )
    cost_status = (
        "measured"
        if metadata.actual_cost_cny is not None
        else "not_run"
        if metadata.status in {"planned", "dry_run", "running"}
        else "not_recorded"
    )
    paths.cost.write_bytes(
        _json_bytes(
            {
                "currency": "CNY",
                "estimated_cost_cny": float(metadata.estimated_cost_cny),
                "actual_cost_cny": metadata.actual_cost_cny,
                "cost_status": cost_status,
            }
        )
    )
    return paths


def append_metric(path: str | Path, metric: Mapping[str, Any]) -> None:
    """Append one canonical, finite metric record."""

    if not isinstance(metric, Mapping) or not metric:
        raise ArtifactError("metric must be a non-empty mapping")
    if type(metric.get("step")) is not int or int(metric["step"]) < 0:
        raise ArtifactError("metric step must be a non-negative integer")
    payload = _json_bytes(metric)
    with Path(path).open("ab") as handle:
        handle.write(payload)
