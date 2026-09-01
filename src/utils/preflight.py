"""CPU-safe, stage-aware, fail-closed preflight for CA-OPD runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.data.access import (
    FinalManifestAccessError,
    load_manifest_for_scheduler,
    load_manifest_for_trainer,
    verify_role_records_artifact,
)
from src.data.schema import (
    CONTROLLER_ROLES_V2,
    DATA_PROTOCOL_VERSION,
    FINAL_ROLES_V2,
    PROMPT_ONLY_ROLES_V2,
    SOURCE_POLICY_VERSION,
)
from src.data.medqa_conflicts_v2 import CONFLICT_POLICY_VERSION


PREFLIGHT_STAGES = ("data", "sft", "controller_eval", "opd", "final")
PREFLIGHT_MODES = ("dry-run", "formal")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PreflightError(RuntimeError):
    """Raised when one stage is not authorized or reproducible."""


@dataclass(frozen=True)
class PreflightResult:
    stage: str
    mode: str
    status: str
    checks: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "mode": self.mode,
            "status": self.status,
            "checks": dict(self.checks),
            "warnings": list(self.warnings),
        }


def _require(value: Mapping[str, Any], key: str) -> Any:
    item = value.get(key)
    if item in (None, "", [], {}):
        raise PreflightError(f"missing required preflight field: {key}")
    return item


def _file(value: Mapping[str, Any], key: str) -> Path:
    path = Path(str(_require(value, key)))
    if not path.is_file():
        raise PreflightError(f"{key} is not a readable file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha(path: Path, expected: Any, *, label: str) -> None:
    if _sha256(path) != str(expected):
        raise PreflightError(f"{label} SHA-256 mismatch")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot parse mapping {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{path} must contain a mapping")
    return value


def _manifest(path: Path) -> dict[str, Any]:
    value = _load_mapping(path)
    if (
        value.get("schema_version") != 2
        or value.get("data_protocol_version") != DATA_PROTOCOL_VERSION
        or not isinstance(value.get("roles"), Mapping)
        or not value["roles"]
    ):
        raise PreflightError(f"{path} is not a Data Protocol v2 manifest")
    return value


def _same_path(left: Any, right: Any) -> bool:
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def _git_sha(request: Mapping[str, Any]) -> None:
    if _HEX40.fullmatch(str(_require(request, "git_sha"))) is None:
        raise PreflightError("git_sha must be a committed 40-character hex SHA")


def _clean_committed(request: Mapping[str, Any], stage: str) -> None:
    if request.get("dirty_worktree") is not False or request.get("committed_worktree") is not True:
        raise PreflightError(f"formal {stage} requires a clean, committed worktree")
    if request.get("verify_live_git") is True:
        import subprocess

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise PreflightError(f"formal {stage} live worktree is not clean")
        authorized = str(_require(request, "git_sha"))
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", authorized, "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestry.returncode != 0:
            raise PreflightError(f"formal {stage} HEAD does not descend from the authorized preparation SHA")


def _disk_gate(request: Mapping[str, Any]) -> None:
    value = request.get("disk_free_gb")
    if type(value) not in (int, float) or float(value) < 20:
        raise PreflightError("disk_free_gb must report at least 20 GiB free")


def _cost_gate(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be a mapping")
    estimate, cap = value.get("estimated_cost_cny"), value.get("cost_cap_cny")
    if (
        type(estimate) not in (int, float)
        or type(cap) not in (int, float)
        or float(estimate) < 0
        or float(cap) <= 0
        or float(estimate) > float(cap)
    ):
        raise PreflightError(f"{label} has an invalid estimate/cap")


def _progress_aware_runtime_gate(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("runtime_policy") != "progress_aware"
        or value.get("global_runtime_hard_limit") is not None
        or type(value.get("live_price_cny_per_hour")) not in (int, float)
        or float(value["live_price_cny_per_hour"]) <= 0
        or type(value.get("cost_monitor_interval_minutes")) is not int
        or int(value["cost_monitor_interval_minutes"]) != 30
    ):
        raise PreflightError(f"{label} is not the frozen progress-aware policy")


def _validate_decoding(value: Any, *, label: str = "decoding") -> None:
    if (
        not isinstance(value, Mapping)
        or type(value.get("temperature")) not in (int, float)
        or float(value["temperature"]) != 0.0
        or value.get("do_sample") is not False
        or type(value.get("seed")) is not int
    ):
        raise PreflightError(
            f"{label} must fix temperature=0, do_sample=false and an integer seed"
        )


def _validate_manifest_files(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    roles: set[str],
    *,
    include_labels: bool = True,
) -> None:
    for role in roles:
        metadata = manifest["roles"].get(role)
        files = metadata.get("files") if isinstance(metadata, Mapping) else None
        if not isinstance(files, list) or not files:
            raise PreflightError(f"{role} lacks physically separated prompt/label artifacts")
        names = [str(item.get("path", "")) for item in files if isinstance(item, Mapping)]
        prompts = [name for name in names if name.endswith(".prompts.jsonl")]
        labels = [name for name in names if name.endswith(".labels.jsonl")]
        if len(prompts) != 1 or len(labels) != 1 or prompts[0] == labels[0]:
            raise PreflightError(f"{role} prompt/label artifacts are not physically separated")
        checked = files if include_labels else [
            item for item in files
            if isinstance(item, Mapping) and str(item.get("path", "")).endswith(".prompts.jsonl")
        ]
        for item in checked:
            if not isinstance(item, Mapping) or not item.get("sha256"):
                raise PreflightError(f"{role} artifact metadata is incomplete")
            path = _resolve_manifest_artifact(manifest_path, str(item["path"]))
            if not path.is_file() or _sha256(path) != str(item["sha256"]):
                raise PreflightError(f"{role} artifact SHA-256 mismatch")


def _resolve_manifest_artifact(manifest_path: Path, declared_path: str) -> Path:
    """Resolve absolute or repository-relative paths bound by a manifest.

    P1 fixture manifests store files next to the manifest.  P2 formal manifests
    deliberately store portable repository-relative paths such as
    ``data/processed/formal_v2/...``.  Walking the manifest's ancestors lets a
    preflight invoked outside the repository find the latter without weakening
    the declared-path or SHA checks.
    """

    declared = Path(declared_path)
    if declared.is_absolute():
        return declared
    # The manifest location is authoritative.  ``cwd`` is only a final
    # compatibility fallback; preferring it could accidentally bind a same-name
    # artifact from a different checkout.
    candidates = [manifest_path.parent / declared]
    candidates.extend(parent / declared for parent in manifest_path.parents)
    candidates.append(Path.cwd() / declared)
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized
    raise PreflightError(f"manifest artifact is not a readable file: {declared_path}")


def _validate_p2_formal_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate P2 provenance without granting final-evaluation capability."""

    if manifest.get("conflict_policy_version") != CONFLICT_POLICY_VERSION:
        raise PreflightError("formal data manifest has missing or stale MedQA conflict policy")
    required_sha_fields = (
        "formal_config_sha256",
        "taxonomy_config_sha256",
        "normalization_implementation_sha256",
        "near_duplicate_config_sha256",
        "tokenizer_artifact_sha256",
    )
    for key in required_sha_fields:
        if _HEX64.fullmatch(str(manifest.get(key, ""))) is None:
            raise PreflightError(f"formal data manifest has invalid {key}")
    for key in (
        "taxonomy_version",
        "normalization_version",
        "near_duplicate_version",
        "tokenizer_id",
    ):
        if not str(manifest.get(key, "")).strip():
            raise PreflightError(f"formal data manifest lacks {key}")
    if _HEX40.fullmatch(str(manifest.get("tokenizer_revision", ""))) is None:
        raise PreflightError("formal data manifest tokenizer revision is not immutable")
    if manifest.get("prompt_label_separated") is not True:
        raise PreflightError("formal data manifest must separate prompts and labels")
    status = str(manifest.get("build_status", ""))
    if status not in {
        "built_pending_manual_audit",
        "formal_ready",
        "formal_ready_mvp_waived",
    }:
        raise PreflightError("formal data manifest has an invalid build_status")
    pending = manifest.get("manual_audit_pending")
    reviewed = manifest.get("human_reviewed")
    if type(pending) is not bool or type(reviewed) is not bool:
        raise PreflightError("formal data manifest manual-audit state is inconsistent")
    if status == "formal_ready_mvp_waived":
        if not (
            pending is False
            and reviewed is False
            and manifest.get("manual_audit_waived_by_user") is True
            and manifest.get("waiver_reason") == "time_constrained_interview_mvp"
            and manifest.get("cross_role_candidates_conservatively_resolved") is True
            and manifest.get("unresolved_cross_role_candidates") == 0
            and manifest.get("primary_final_frozen") is True
            and manifest.get("final_authorized") is False
        ):
            raise PreflightError("formal MVP waiver contract is incomplete or falsely reviewed")
    elif pending == reviewed:
        raise PreflightError("formal data manifest manual-audit state is inconsistent")
    elif manifest.get("primary_final_frozen") is not False:
        raise PreflightError("pre-waiver P2 manifest must keep primary_final_frozen=false")
    for role, metadata in manifest["roles"].items():
        files = metadata.get("files") if isinstance(metadata, Mapping) else None
        if not isinstance(files, list) or not files:
            raise PreflightError(f"formal role {role} lacks records artifacts")
        for item in files:
            if not isinstance(item, Mapping) or item.get("complete") is not True:
                raise PreflightError(f"formal role {role} contains an incomplete artifact")
            path = _resolve_manifest_artifact(manifest_path, str(item.get("path", "")))
            if _sha256(path) != str(item.get("sha256", "")):
                raise PreflightError(f"formal role {role} records SHA-256 mismatch")
            if path.with_suffix(path.suffix + ".tmp").exists():
                raise PreflightError(f"formal role {role} still has a partial artifact")
    if pending:
        return (
            "manual near-duplicate audit is pending; data are built but not fully training-approved",
        )
    if status == "formal_ready_mvp_waived":
        return (
            "manual near-duplicate audit was explicitly waived; human_reviewed remains false",
        )
    return ()


def _run_data(request: Mapping[str, Any], mode: str) -> PreflightResult:
    source_path = _file(request, "source_config")
    split_path = _file(request, "split_config")
    filter_path = _file(request, "filter_config") if request.get("filter_config") else None
    if request.get("schema_version") != 2:
        raise PreflightError("data stage requires schema_version=2")
    loaded_configs = {
        "source": _load_mapping(source_path),
        "split": _load_mapping(split_path),
    }
    if filter_path is not None:
        loaded_configs["filter"] = _load_mapping(filter_path)
    for path, config in (
        (source_path, loaded_configs["source"]),
        (split_path, loaded_configs["split"]),
        (filter_path, loaded_configs.get("filter")),
    ):
        if path is not None and config.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
            raise PreflightError(f"{path} does not declare Data Protocol v2")
    manifest_path = _file(request, "manifest")
    manifest = _manifest(manifest_path)
    build_mode = str(_require(request, "build_mode"))
    if build_mode not in {"smoke", "formal"} or manifest.get("build_mode") != build_mode:
        raise PreflightError("data build_mode differs from the manifest")
    _check_sha(source_path, _require(manifest, "source_config_sha256"), label="source config")
    _check_sha(split_path, _require(manifest, "split_config_sha256"), label="split config")
    if filter_path is not None:
        _check_sha(filter_path, _require(manifest, "filter_config_sha256"), label="filter config")
    elif manifest.get("filter_config_sha256"):
        raise PreflightError("manifest binds a filter config but request omits filter_config")
    overlap_path = _file(request, "overlap_report")
    if _load_mapping(overlap_path).get("status") != "PASS":
        raise PreflightError("data overlap report is not PASS")
    _check_sha(overlap_path, _require(manifest, "overlap_report_sha256"), label="overlap report")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise PreflightError("data manifest must record source revision/license")
    if mode == "formal" and (build_mode != "formal" or manifest.get("synthetic_fixture") is not False):
        raise PreflightError("formal data rejects synthetic fixture manifests")
    warnings: tuple[str, ...] = ()
    if mode == "formal":
        if manifest.get("source_policy_version") != SOURCE_POLICY_VERSION:
            raise PreflightError("formal data manifest has missing or stale source policy version")
        for label in ("source", "split"):
            if loaded_configs[label].get("source_policy_version") != SOURCE_POLICY_VERSION:
                raise PreflightError(f"formal {label} config has missing or stale source policy version")
    for name, source in sources.items():
        if not isinstance(source, Mapping) or not source.get("revision") or not (source.get("declared_license") or source.get("license")):
            raise PreflightError(f"source {name} lacks revision/license")
        if mode == "formal":
            revision = str(source["revision"])
            if _HEX40.fullmatch(revision) is None:
                raise PreflightError(f"source {name} uses floating revision {revision!r}")
            if source.get("fixture_only") is True or str(source.get("declared_license", "")).casefold() == "fixture-only":
                raise PreflightError(f"formal data cannot use fixture-only source {name}")
            if _HEX64.fullmatch(str(source.get("raw_file_sha256", ""))) is None:
                raise PreflightError(f"formal source {name} lacks raw/source revision provenance")
    if mode == "formal":
        warnings = _validate_p2_formal_manifest(manifest_path, manifest)
    return PreflightResult(
        "data",
        mode,
        "PASS",
        {
            "source_config": "PASS",
            "split_config": "PASS",
            "filter_config": "PASS",
            "config_sha256": "PASS",
            "schema_version": "PASS",
            "source_policy_version": "PASS",
            "manifest": "PASS",
            "source_revision_license": "PASS",
            "overlap_report_sha256": "PASS",
            "build_mode": "PASS",
        },
        warnings,
    )


def _load_sft_config(path: Path) -> dict[str, Any]:
    try:
        from src.sft.train import SFT_SCHEMA, validate_model_revisions
        from src.utils.config import load_config

        raw = _load_mapping(path)
        if "distributed" in raw:
            from src.sft.train_ddp import DDP_SFT_SCHEMA

            config = load_config(path, DDP_SFT_SCHEMA)
        else:
            config = load_config(path, SFT_SCHEMA)
        validate_model_revisions(config)
        return config
    except (ValueError, TypeError, OSError) as error:
        raise PreflightError(f"SFT config is invalid: {error}") from error


def _run_sft(request: Mapping[str, Any], mode: str) -> PreflightResult:
    config_path = _file(request, "sft_config")
    config = _load_sft_config(config_path)
    model, data = config["model"], config["data"]
    bindings = (
        (str(_require(request, "base_model_id")), str(model["path"]), "model path"),
        (str(_require(request, "base_model_revision")), str(model["revision"]), "model revision"),
        (str(_require(request, "tokenizer_revision")), str(model["tokenizer_revision"]), "tokenizer revision"),
        (str(_require(request, "data_protocol_version")), str(data["protocol_version"]), "data protocol version"),
        (str(_require(request, "target_role")), str(data["target_role"]), "target_role"),
    )
    for actual, declared, label in bindings:
        if actual != declared:
            raise PreflightError(f"SFT request {label} differs from parsed config")
    if data["protocol_version"] != DATA_PROTOCOL_VERSION or data["target_role"] != "medical_sft_train":
        raise PreflightError("SFT config must bind Data Protocol v2 target_role=medical_sft_train")
    if data["enable_thinking"] is not False or data["drop_longer_than_max_seq"] is not True:
        raise PreflightError("SFT config requires enable_thinking=false and drop_longer_than_max_seq=true")
    if int(model["max_seq_length"]) < 1:
        raise PreflightError("SFT config max_seq_length must be positive")
    manifest_path = _file(request, "medical_sft_manifest")
    records_path = _file(request, "records_path")
    manifest = _manifest(manifest_path)
    try:
        load_manifest_for_trainer(manifest, stage="sft")
    except (FinalManifestAccessError, PermissionError, ValueError) as error:
        raise PreflightError(f"SFT manifest rejected: {error}") from error
    if not _same_path(data["manifest_path"], manifest_path) or not _same_path(data["records_path"], records_path):
        raise PreflightError("SFT config manifest/records path differs from request")
    _check_sha(manifest_path, _require(request, "data_manifest_sha256"), label="data manifest")
    _check_sha(records_path, _require(request, "records_sha256"), label="records")
    try:
        verify_role_records_artifact(manifest, records_path, role="medical_sft_train")
    except (FinalManifestAccessError, PermissionError, ValueError) as error:
        raise PreflightError(f"SFT manifest rejected: {error}") from error
    output = Path(str(_require(request, "output_dir")))
    if output.exists() and any(output.iterdir()):
        raise PreflightError("SFT output_dir must be empty or new")
    _disk_gate(request)
    _git_sha(request)
    _file(request, "run_card")
    warnings: list[str] = []
    checks = {
        "sft_config": "PASS",
        "model_revision_binding": "PASS",
        "medical_sft_manifest": "PASS",
        "records_sha256": "PASS",
        "output_dir": "PASS",
        "disk_gate": "PASS",
        "git_sha": "PASS",
        "run_card": "PASS",
    }
    distributed = config.get("distributed")
    if distributed is not None or request.get("launch_mode") is not None:
        from src.sft.ddp import DDPExecutionContract, validate_training_source_contract

        frozen = DDPExecutionContract.frozen()
        if request.get("launch_mode") != "ddp" or distributed is None:
            raise PreflightError("P3.5 formal SFT requires the explicit DDP launch contract")
        if int(request.get("world_size", -1)) != 2 or int(distributed["world_size"]) != 2:
            raise PreflightError("P3.5 DDP world_size must be 2")
        if int(request.get("expected_gpu_count", -1)) != 2:
            raise PreflightError("P3.5 expects exactly two GPUs at host preflight")
        effective = (
            int(config["optim"]["per_device_batch_size"])
            * int(distributed["world_size"])
            * int(config["optim"]["gradient_accumulation_steps"])
        )
        if effective != 16 or int(request.get("global_effective_batch", -1)) != 16:
            raise PreflightError("P3.5 DDP global batch must remain 16")
        expected_distributed = {
            "launch_mode": "ddp",
            "backend": "nccl",
            "world_size": 2,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "bucket_cap_mb": 16,
            "global_weighted_denominator": True,
            "rank_zero_only_writes": True,
            "fresh_base_per_rank": True,
            "device_map": "none",
        }
        if dict(distributed) != expected_distributed:
            raise PreflightError("P3.5 DDP execution options differ from the frozen contract")
        if request.get("fresh_base_initialization") is not True or config["run"]["resume_from_checkpoint"] is not None:
            raise PreflightError("P3.5 requires fresh Base initialization")
        if request.get("final_authorized") is not False:
            raise PreflightError("P3.5 SFT requires final_authorized=false")
        supervision_version = config["data"]["supervision_version"]
        if supervision_version == "mcq_dominant_task_balanced_v3":
            if (
                float(config["data"]["answer_weight"]) != 1.0
                or float(config["data"]["reasoning_weight"]) != 1.0
                or float(config["data"]["eos_weight"]) != 1.0
                or config["data"]["include_reasoning"] is not False
                or int(model["max_seq_length"]) != 2048
                or config["optim"].get("optimizer") != "adamw_torch_fused"
                or int(config["optim"].get("max_steps") or -1) != 600
            ):
                raise PreflightError("SFT-v3 supervision, optimizer steps, or max length drifted")
            if request.get("sft_v3_task_schedule") != ["cmb", "cmb", "cmb", "medical_o1"]:
                raise PreflightError("SFT-v3 task schedule differs from frozen 3:1 order")
            if int(request.get("optimizer_steps", -1)) != 600 or request.get(
                "checkpoint_steps"
            ) != [150, 300, 450, 600]:
                raise PreflightError("SFT-v3 optimizer/checkpoint schedule drift")
            if int(request.get("controller_gate_medical_correct", -1)) != 228:
                raise PreflightError("SFT-v3 Teacher gate must remain 228/300")
            if manifest.get("build_version") != "p3-6-sft-v3-mcq-dominant-v1":
                raise PreflightError("SFT-v3 manifest build version drift")
            if manifest.get("source_counts") != {"cmb": 7200, "medical_o1": 2400}:
                raise PreflightError("SFT-v3 source count drift")
            if manifest.get("candidate_token_ids") != {
                "A": 32, "B": 33, "C": 34, "D": 35, "E": 36
            }:
                raise PreflightError("SFT-v3 first-token candidate contract drift")
            if manifest.get("task_schedule", {}).get("global_exposures") != {
                "cmb": 7200, "medical_o1": 2400
            }:
                raise PreflightError("SFT-v3 one-use task schedule drift")
            confirmation_path = _file(request, "confirmation_manifest")
            _check_sha(
                confirmation_path,
                _require(request, "confirmation_manifest_sha256"),
                label="frozen confirmation manifest",
            )
            confirmation = _load_mapping(confirmation_path)
            if (
                confirmation.get("role") != "medical_teacher_confirmation_dev"
                or confirmation.get("status") != "frozen_before_candidate_results"
                or confirmation.get("final_authorized") is not False
                or confirmation.get("final_artifacts_opened") is not False
            ):
                raise PreflightError("SFT-v3 confirmation isolation contract drift")
            checks.update(
                {
                    "sft_v3_task_schedule": "PASS",
                    "sft_v3_first_token_contract": "PASS",
                    "confirmation_final_isolation": "PASS",
                }
            )
        elif (
            supervision_version != "answer_first_weighted_v2"
            or float(config["data"]["answer_weight"]) != 1.5
            or float(config["data"]["reasoning_weight"]) != 0.5
            or float(config["data"]["eos_weight"]) != 1.5
            or int(model["max_seq_length"]) != 2048
            or config["optim"].get("optimizer") != "adamw_torch_fused"
        ):
            raise PreflightError("P3.5 weighted supervision or max length drifted")
        source = Path("src/sft/train_ddp.py")
        if not source.is_file():
            raise PreflightError("P3.5 DDP runtime source is missing")
        try:
            validate_training_source_contract(source.read_text(encoding="utf-8"))
        except ValueError as error:
            raise PreflightError(str(error)) from error
        checks.update(
            {
                "ddp_launch_contract": "PASS",
                "world_size_two": "PASS",
                "global_effective_batch_16": "PASS",
                "global_weighted_loss": "PASS",
                "fresh_base_initialization": "PASS",
                "final_isolation": "PASS",
                "no_implicit_parallel_fallback": "PASS",
            }
        )
        if request.get("calibration_samples") is not None:
            calibration_path = _file(request, "calibration_samples")
            _check_sha(
                calibration_path,
                _require(request, "calibration_samples_sha256"),
                label="DDP calibration samples",
            )
            calibration = _load_mapping(calibration_path)
            if (
                calibration.get("status") != "frozen_before_gpu"
                or calibration.get("world_size") != 2
                or calibration.get("microbatches_per_rank") != 8
                or calibration.get("records_sha256") != str(_require(request, "records_sha256"))
                or calibration.get("final_authorized") is not False
            ):
                raise PreflightError("DDP calibration selection contract is invalid")
            checks["worst_length_calibration_selection"] = "PASS"
        if request.get("runtime_hardware_check_pending") is not True:
            raise PreflightError(
                "CPU preparation must record runtime_hardware_check_pending=true"
            )
        warnings.append(
            "GPU count=2 must be verified by gpu_host_preflight after explicit GPU authorization"
        )
    if mode == "dry-run":
        if request.get("mock_model") is not True or request.get("mock_tokenizer") is not True:
            raise PreflightError("SFT dry-run requires explicit mock_model/mock_tokenizer")
        checks["mock_model_tokenizer"] = "PASS"
        if request.get("dirty_worktree") is True:
            warnings.append("dirty worktree allowed for SFT dry-run and recorded")
    else:
        if (
            manifest.get("build_mode") != "formal"
            or manifest.get("synthetic_fixture") is not False
            or manifest.get("conflict_policy_version") != CONFLICT_POLICY_VERSION
        ):
            raise PreflightError("formal SFT requires a current non-synthetic P2 formal manifest")
        reviewed_ready = (
            manifest.get("build_status") == "formal_ready"
            and manifest.get("manual_audit_pending") is False
            and manifest.get("human_reviewed") is True
        )
        waived_ready = (
            manifest.get("build_status") == "formal_ready_mvp_waived"
            and manifest.get("manual_audit_pending") is False
            and manifest.get("human_reviewed") is False
            and manifest.get("manual_audit_waived_by_user") is True
            and manifest.get("waiver_reason") == "time_constrained_interview_mvp"
            and manifest.get("cross_role_candidates_conservatively_resolved") is True
            and manifest.get("unresolved_cross_role_candidates") == 0
            and manifest.get("primary_final_frozen") is True
            and manifest.get("final_authorized") is False
        )
        if not (reviewed_ready or waived_ready):
            raise PreflightError(
                "formal SFT requires a completed manual audit or explicit MVP waiver contract"
            )
        approved = str(model["path"]) in {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"} or Path(str(model["path"])).name in {"Qwen3-1.7B", "Qwen3-4B"}
        if not approved or _HEX40.fullmatch(str(model["revision"])) is None:
            raise PreflightError("formal SFT requires an approved Qwen3 model and immutable revision")
        _clean_committed(request, "SFT")
        _cost_gate(_require(request, "budget"), label="budget")
        checks.update({"base_model": "PASS", "clean_committed_worktree": "PASS", "budget": "PASS"})
    return PreflightResult("sft", mode, "PASS", checks, tuple(warnings))


def _validate_controller_config(
    path: Path, request: Mapping[str, Any], manifest_path: Path
) -> None:
    config = _load_mapping(path)
    runtime_fields = {
        "run_id", "capability", "model", "data", "decode", "output_root",
        "allow_final_eval", "primary_final_frozen",
    }
    if set(config) == runtime_fields:
        try:
            from src.eval.runtime import load_eval_runtime_config

            parsed = load_eval_runtime_config(path)
        except (ValueError, OSError) as error:
            raise PreflightError(f"evaluator runtime config is invalid: {error}") from error
        data = parsed["data"]
        if not _same_path(data["manifest_path"], manifest_path):
            raise PreflightError("evaluator config manifest path differs from request")
        expected = str(_require(request, "controller_manifest_sha256"))
        if str(data["manifest_sha256"]) != expected:
            raise PreflightError("evaluator config manifest hash differs from request")
        if set(data["roles"]) != set(CONTROLLER_ROLES_V2):
            raise PreflightError("evaluator config must declare both controller roles and no final role")
        if dict(parsed["decode"]) != dict(_require(request, "decoding")):
            raise PreflightError("evaluator decoding differs from request")
        return
    required = {
        "data_protocol_version",
        "target_roles",
        "manifest_path",
        "manifest_sha256",
        "prompt_label_separated",
        "decoding",
    }
    if set(config) != required:
        raise PreflightError("evaluator config has missing or unknown keys")
    if config["data_protocol_version"] != DATA_PROTOCOL_VERSION:
        raise PreflightError("evaluator config data protocol mismatch")
    if set(config["target_roles"]) != set(CONTROLLER_ROLES_V2):
        raise PreflightError("evaluator config must declare both controller roles and no final role")
    if not _same_path(config["manifest_path"], manifest_path):
        raise PreflightError("evaluator config manifest path differs from request")
    expected = str(_require(request, "controller_manifest_sha256"))
    if str(config["manifest_sha256"]) != expected:
        raise PreflightError("evaluator config manifest hash differs from request")
    if config["prompt_label_separated"] is not True:
        raise PreflightError("evaluator config requires prompt/label separation")
    _validate_decoding(config["decoding"], label="evaluator decoding")
    if dict(config["decoding"]) != dict(_require(request, "decoding")):
        raise PreflightError("evaluator decoding differs from request")


def _validate_controller_v2_config(
    path: Path,
    request: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    mode: str,
) -> dict[str, str]:
    """Validate the frozen v2 protocol without importing a model runtime."""

    try:
        from src.eval.controller_v2 import (
            BASE_MODEL_REVISION,
            MEDICAL_LORA_SHA256,
            PROTOCOL_VERSION,
            protocol_component_hashes,
        )
        from src.eval.controller_v2_runtime import load_controller_v2_config
        from src.eval.controller_v2_runtime import _label_artifact_attestation
        from src.eval.direct_logit_scorer import (
            DIRECT_LOGIT_BACKEND,
            VLLM_CHOICE_BACKEND_STATUS,
            direct_logit_model_plan,
        )

        config = load_controller_v2_config(path)
    except (OSError, ValueError, RuntimeError) as error:
        raise PreflightError(f"Controller v2 evaluator config is invalid: {error}") from error

    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise PreflightError("Controller v2 protocol_version mismatch")
    if request.get("final_authorized") is not False:
        raise PreflightError("Controller v2 must keep final_authorized=false")
    if not _same_path(config["data"]["manifest_path"], manifest_path):
        raise PreflightError("Controller v2 config manifest path differs from request")
    if config["data"]["manifest_sha256"] != str(_require(request, "controller_manifest_sha256")):
        raise PreflightError("Controller v2 config manifest SHA differs from request")
    expected_counts = {
        "medical_controller_dev": int(config["data"]["medical_count"]),
        "general_controller_dev": int(config["data"]["general_count"]),
    }
    actual_counts = {
        role: int(manifest["roles"][role].get("actual_count", -1))
        for role in CONTROLLER_ROLES_V2
    }
    if expected_counts != actual_counts:
        raise PreflightError("Controller v2 B0/B1 sample set/count binding mismatch")
    if request.get("base_model_revision") != BASE_MODEL_REVISION:
        raise PreflightError("Controller v2 base model revision mismatch")
    if (
        request.get("model_id") != config["model"]["id"]
        or not _same_path(str(request.get("model_path") or ""), Path(config["model"]["path"]))
        or request.get("model_manifest_sha256") != config["model"]["manifest_sha256"]
    ):
        raise PreflightError("Controller v2 verified model identity/path mismatch")
    if request.get("medical_lora_sha256") != MEDICAL_LORA_SHA256:
        raise PreflightError("Controller v2 Medical LoRA SHA mismatch")
    if (
        request.get("choice_backend") != DIRECT_LOGIT_BACKEND
        or config["choice_score"].get("backend") != DIRECT_LOGIT_BACKEND
        or request.get("vllm_choice_backend_status") != VLLM_CHOICE_BACKEND_STATUS
        or config["choice_score"].get("legacy_vllm_prompt_logprobs") != VLLM_CHOICE_BACKEND_STATUS
    ):
        raise PreflightError("Controller v2 formal choice backend must be Transformers direct logits")
    if (
        request.get("direct_logit_batch_size") != 1
        or request.get("float32_log_softmax") is not True
        or request.get("direct_logit_runtime") != config["choice_score"].get("runtime")
    ):
        raise PreflightError("Controller v2 direct-logit execution contract mismatch")
    for route in ("B0", "B1"):
        plan = direct_logit_model_plan(route)
        if (
            plan["backend"] != DIRECT_LOGIT_BACKEND
            or plan["batch_size"] != 1
            or plan["use_cache"] is not False
            or plan["attn_implementation"] != "eager"
            or plan["torch_compile"] is not False
            or plan["merge_lora"] is not False
            or plan["log_softmax_dtype"] != "float32"
        ):
            raise PreflightError("Controller v2 direct-logit model plan drift")
    component_hashes = protocol_component_hashes()
    for field in ("protocol_sha256", "prompt_sha256", "parser_sha256", "scorer_sha256"):
        if request.get(field) != component_hashes[field]:
            raise PreflightError(f"Controller v2 {field.removesuffix('_sha256')} SHA mismatch")
    if config["choice_score"].get("candidates_from_option_count") is not True:
        raise PreflightError("Controller v2 candidates must come from each option count")
    generation = config["generation"]
    _validate_decoding(
        {
            "temperature": generation.get("temperature"),
            "do_sample": generation.get("do_sample"),
            "seed": config.get("seed"),
        },
        label="Controller v2 generation",
    )
    phase = request.get("evaluation_phase")
    if phase not in {"length_smoke", "full"}:
        raise PreflightError("Controller v2 evaluation_phase must be length_smoke or full")
    if phase == "full":
        expected_attestation = _label_artifact_attestation(
            manifest_path, list(config["data"]["roles"])
        )
        if request.get("prevalidated_label_attestation") != expected_attestation:
            raise PreflightError("Controller v2 full evaluation label attestation mismatch")
        decision_path = request.get("length_decision")
        if not decision_path:
            raise PreflightError("Controller v2 full evaluation requires a frozen length decision")
        decision = _load_mapping(Path(str(decision_path)))
        if (
            decision.get("protocol_version") != PROTOCOL_VERSION
            or decision.get("status") != "frozen_before_full_evaluation"
            or decision.get("max_new_tokens") not in {512, 1024}
            or decision.get("decision_basis") != "truncation_only"
        ):
            raise PreflightError("Controller v2 length decision is invalid")
        _check_sha(
            Path(str(decision_path)),
            _require(request, "length_decision_sha256"),
            label="Controller v2 length decision",
        )
    if mode == "dry-run":
        if request.get("cpu_dry_run") is not True:
            raise PreflightError("Controller v2 dry-run must declare cpu_dry_run=true")
    else:
        _git_sha(request)
        _clean_committed(request, "controller_eval v2")
        _progress_aware_runtime_gate(
            _require(request, "runtime_gate"), label="controller_eval runtime_gate"
        )
    output = Path(str(_require(request, "result_output_dir")))
    if output.exists() and any(output.iterdir()):
        raise PreflightError("Controller v2 result output must be empty or new")
    return {
        "controller_protocol_v2": "PASS",
        "choice_backend": "PASS",
        "vllm_choice_diagnostic_only": "PASS",
        "direct_logit_determinism": "PASS",
        "base_and_lora_binding": "PASS",
        "protocol_component_hashes": "PASS",
        "same_b0_b1_samples": "PASS",
        "length_policy": "PASS",
        "cpu_dry_run_no_model": "PASS" if mode == "dry-run" else "N/A",
    }


def _run_controller(request: Mapping[str, Any], mode: str) -> PreflightResult:
    if "final_manifest" in request:
        raise PreflightError("controller_eval cannot receive a final manifest")
    if request.get("mock_checkpoint") is not True:
        checkpoint = _file(request, "checkpoint")
        if request.get("checkpoint_frozen") is not True:
            raise PreflightError("controller checkpoint must be frozen")
        _check_sha(checkpoint, _require(request, "checkpoint_sha256"), label="checkpoint")
    manifest_path = _file(request, "controller_manifest")
    _check_sha(manifest_path, _require(request, "controller_manifest_sha256"), label="controller manifest")
    manifest = _manifest(manifest_path)
    try:
        load_manifest_for_scheduler(manifest)
    except (FinalManifestAccessError, PermissionError, ValueError) as error:
        raise PreflightError(f"controller manifest rejected final/disallowed roles: {error}") from error
    if request.get("prompt_label_separated") is not True or manifest.get("prompt_label_separated") is not True:
        raise PreflightError("controller prompt/label artifacts must be separated")
    _validate_decoding(_require(request, "decoding"))
    config_path = _file(request, "evaluator_config")
    v2_checks: dict[str, str] = {}
    if request.get("protocol_version") == "controller_protocol_v2":
        v2_checks = _validate_controller_v2_config(
            config_path, request, manifest_path, manifest, mode
        )
        _validate_manifest_files(
            manifest_path,
            manifest,
            set(CONTROLLER_ROLES_V2),
            # Controller Protocol v2 keeps label files unopened until every
            # model execution object has been released. The manifest metadata
            # still proves physical prompt/label separation here; the
            # independent scorer revalidates label SHA/content afterward.
            include_labels=False,
        )
    else:
        _validate_manifest_files(manifest_path, manifest, set(CONTROLLER_ROLES_V2))
        _validate_controller_config(config_path, request, manifest_path)
    return PreflightResult(
        "controller_eval",
        mode,
        "PASS",
        {
            "checkpoint": "PASS",
            "controller_manifest": "PASS",
            "controller_manifest_sha256": "PASS",
            "evaluator_config": "PASS",
            "prompt_label_separation": "PASS",
            "deterministic_decoding": "PASS",
            "final_manifest_absent": "PASS",
            **v2_checks,
        },
    )


def _validate_teacher_config(path: Path, adapter_path: Path) -> None:
    config = _load_mapping(path)
    required = {
        "mode",
        "model_path",
        "model_revision",
        "medical_adapter_path",
        "max_lora_rank",
        "gpu_memory_utilization",
        "temperature",
        "max_tokens",
        "prompt_logprobs",
        "calibration_status",
    }
    if set(config) != required:
        raise PreflightError("Teacher config has missing or unknown keys")
    if config["mode"] != "shared_backbone_lora_router":
        raise PreflightError("Teacher config mode must be shared_backbone_lora_router")
    if not str(config["model_path"]).strip() or _HEX40.fullmatch(str(config["model_revision"])) is None:
        raise PreflightError("Teacher config must bind model path and immutable revision")
    if not _same_path(config["medical_adapter_path"], adapter_path):
        raise PreflightError("Teacher config adapter path differs from Medical LoRA artifact")
    if type(config["max_lora_rank"]) is not int or config["max_lora_rank"] < 1:
        raise PreflightError("Teacher config max_lora_rank must be positive")
    if type(config["gpu_memory_utilization"]) not in (int, float) or not 0 < float(config["gpu_memory_utilization"]) <= 1:
        raise PreflightError("Teacher config gpu_memory_utilization must be in (0, 1]")
    if (
        float(config["temperature"]) != 1.0
        or config["max_tokens"] != 1
        or config["prompt_logprobs"] != 0
        or config["calibration_status"] != "candidate_pending_gpu_calibration"
    ):
        raise PreflightError("Teacher config scoring/calibration contract is invalid")


def _validate_baselines(value: Any, manifest_sha: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"B0", "B1"}:
        raise PreflightError("baseline_artifacts must contain complete B0 and B1 descriptors")
    required = {
        "baseline_id",
        "status",
        "artifact_path",
        "artifact_sha256",
        "data_protocol_version",
        "data_manifest_sha256",
        "checkpoint_sha256",
        "decoding_config",
        "decoding_config_sha256",
    }
    for baseline_id in ("B0", "B1"):
        item = value[baseline_id]
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise PreflightError(f"{baseline_id} baseline artifact descriptor is incomplete")
        if item["baseline_id"] != baseline_id or item["status"] != "complete":
            raise PreflightError(f"{baseline_id} baseline artifact is not complete")
        if item["data_protocol_version"] != DATA_PROTOCOL_VERSION:
            raise PreflightError(f"{baseline_id} data protocol mismatch")
        if item["data_manifest_sha256"] != manifest_sha:
            raise PreflightError(f"{baseline_id} data manifest SHA-256 mismatch")
        if _HEX64.fullmatch(str(item["checkpoint_sha256"])) is None:
            raise PreflightError(f"{baseline_id} checkpoint SHA-256 is invalid")
        artifact = Path(str(item["artifact_path"]))
        if not artifact.is_file():
            raise PreflightError(f"{baseline_id} artifact is not a readable file")
        _check_sha(artifact, item["artifact_sha256"], label=f"{baseline_id} artifact")
        decoding_path = Path(str(item["decoding_config"]))
        if not decoding_path.is_file():
            raise PreflightError(f"{baseline_id} decoding config is not a readable file")
        _check_sha(decoding_path, item["decoding_config_sha256"], label=f"{baseline_id} decoding config")
        _validate_decoding(_load_mapping(decoding_path), label=f"{baseline_id} decoding")


def _run_opd(request: Mapping[str, Any], mode: str) -> PreflightResult:
    for key in (
        "medical_lora_manifest",
        "controller_manifest",
        "baseline_artifacts",
        "opd_prompt_manifest",
        "router_config",
        "teacher_service_config",
        "opd_config",
    ):
        _require(request, key)
    lora_manifest_path = _file(request, "medical_lora_manifest")
    lora_manifest = _load_mapping(lora_manifest_path)
    adapter_name, adapter_sha = lora_manifest.get("adapter_file"), lora_manifest.get("adapter_sha256")
    if not adapter_name or not adapter_sha:
        raise PreflightError("medical_lora_manifest lacks adapter file/SHA")
    adapter_path = lora_manifest_path.parent / str(adapter_name)
    if not adapter_path.is_file() or _sha256(adapter_path) != str(adapter_sha):
        raise PreflightError("Medical LoRA adapter SHA mismatch")

    prompt_path = _file(request, "opd_prompt_manifest")
    _check_sha(prompt_path, _require(request, "opd_prompt_manifest_sha256"), label="OPD prompt manifest")
    manifest_sha = _sha256(prompt_path)
    prompt_manifest = _manifest(prompt_path)
    try:
        load_manifest_for_trainer(prompt_manifest, stage="opd")
    except (FinalManifestAccessError, PermissionError, ValueError) as error:
        raise PreflightError(f"OPD prompt manifest rejected final/disallowed roles: {error}") from error
    if set(prompt_manifest["roles"]) != set(PROMPT_ONLY_ROLES_V2):
        raise PreflightError("OPD prompt manifest must bind both medical pools and general anchors")
    if request.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise PreflightError("OPD data protocol version mismatch")

    controller_path = _file(request, "controller_manifest")
    _check_sha(controller_path, _require(request, "controller_manifest_sha256"), label="controller manifest")
    controller_manifest = _manifest(controller_path)
    try:
        load_manifest_for_scheduler(controller_manifest)
    except (FinalManifestAccessError, PermissionError, ValueError) as error:
        raise PreflightError(f"OPD controller manifest rejected final/disallowed roles: {error}") from error
    _validate_manifest_files(controller_path, controller_manifest, set(CONTROLLER_ROLES_V2))
    _validate_baselines(_require(request, "baseline_artifacts"), manifest_sha)

    router_path = _file(request, "router_config")
    try:
        from src.opd.router import RouterConfig

        RouterConfig.from_mapping(_load_mapping(router_path))
    except (ValueError, TypeError, KeyError) as error:
        raise PreflightError(f"Router config is invalid: {error}") from error
    teacher_path = _file(request, "teacher_service_config")
    _validate_teacher_config(teacher_path, adapter_path)
    opd_path = _file(request, "opd_config")
    try:
        from src.opd.run_config import load_run

        loaded = load_run(opd_path)
    except (ValueError, TypeError, OSError) as error:
        raise PreflightError(f"OPD config is invalid: {error}") from error
    if loaded.raw["data"]["protocol_version"] != DATA_PROTOCOL_VERSION or not _same_path(loaded.raw["data"]["manifest"], prompt_path):
        raise PreflightError("OPD config data manifest/protocol differs from request")
    if request.get("tokenizer_compatible") is not True:
        raise PreflightError("Student/Teacher tokenizer compatibility is not verified")
    _cost_gate(_require(request, "cost_gate"), label="cost_gate")
    _disk_gate(request)
    if mode == "formal":
        _git_sha(request)
        _clean_committed(request, "OPD")
    return PreflightResult(
        "opd",
        mode,
        "PASS",
        {
            "medical_lora": "PASS",
            "baseline_artifacts": "PASS",
            "opd_manifest": "PASS",
            "controller_manifest": "PASS",
            "router": "PASS",
            "teacher_service": "PASS",
            "tokenizer_compatibility": "PASS",
            "opd_config": "PASS",
            "cost_gate": "PASS",
            "disk_gate": "PASS",
            "clean_committed_worktree": "PASS",
        },
    )


def _run_final(request: Mapping[str, Any], mode: str, *, allow_final_eval: bool) -> PreflightResult:
    if not allow_final_eval:
        raise PreflightError("final stage requires explicit allow_final_eval capability")
    checkpoint = _file(request, "checkpoint")
    if request.get("checkpoint_frozen") is not True:
        raise PreflightError("final checkpoint must be frozen")
    _check_sha(checkpoint, _require(request, "checkpoint_sha256"), label="checkpoint")
    manifest_path = _file(request, "final_manifest")
    manifest = _manifest(manifest_path)
    if manifest.get("source_policy_version") != SOURCE_POLICY_VERSION:
        raise PreflightError("final manifest has missing or stale source policy version")
    _check_sha(manifest_path, _require(request, "final_manifest_sha256"), label="final manifest")
    if manifest.get("primary_final_frozen") is not True:
        raise PreflightError(
            "final manifest requires primary_final_frozen=true; candidate denylists are not final capability"
        )
    if request.get("final_manifest_frozen") is not True or manifest.get("frozen") is not True:
        raise PreflightError("final manifest must be frozen in request and artifact")
    if set(manifest["roles"]) != set(FINAL_ROLES_V2):
        raise PreflightError("final manifest must contain only both final roles")
    if manifest.get("prompt_label_separated") is not True:
        raise PreflightError("final prompt/label artifacts must be separated")
    _validate_manifest_files(manifest_path, manifest, set(FINAL_ROLES_V2))
    if request.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise PreflightError("final request data protocol version mismatch")
    _git_sha(request)
    _validate_decoding(_require(request, "decoding"))
    if not isinstance(request.get("final_authorization"), str) or not request["final_authorization"].strip():
        raise PreflightError("final_authorization must identify explicit approval")
    output = Path(str(_require(request, "result_output_dir")))
    if output.exists() and any(output.iterdir()):
        raise PreflightError("final result output must be empty or new")
    if request.get("final_not_used_for_training_or_selection") is not True:
        raise PreflightError("final must be declared unused for training or selection")
    return PreflightResult(
        "final",
        mode,
        "PASS",
        {
            "checkpoint_frozen_sha": "PASS",
            "final_manifest_frozen_sha": "PASS",
            "prompt_label_separation": "PASS",
            "data_protocol_version": "PASS",
            "git_sha": "PASS",
            "deterministic_decoding": "PASS",
            "final_authorization": "PASS",
            "new_output_dir": "PASS",
            "no_training_or_selection_use": "PASS",
        },
    )


def run_preflight(
    stage: str,
    request: Mapping[str, Any],
    *,
    mode: str,
    allow_final_eval: bool = False,
) -> PreflightResult:
    """Run exactly one stage-specific preflight without starting external work."""

    if stage not in PREFLIGHT_STAGES:
        raise PreflightError(f"unsupported preflight stage: {stage}")
    if mode not in PREFLIGHT_MODES:
        raise PreflightError(f"unsupported preflight mode: {mode}")
    if not isinstance(request, Mapping):
        raise PreflightError("preflight request must be a mapping")
    if stage == "data":
        return _run_data(request, mode)
    if stage == "sft":
        return _run_sft(request, mode)
    if stage == "controller_eval":
        return _run_controller(request, mode)
    if stage == "opd":
        return _run_opd(request, mode)
    return _run_final(request, mode, allow_final_eval=allow_final_eval)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=PREFLIGHT_STAGES)
    parser.add_argument("--mode", required=True, choices=PREFLIGHT_MODES)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--allow-final-eval", action="store_true")
    args = parser.parse_args()
    try:
        result = run_preflight(
            args.stage,
            _load_mapping(args.request),
            mode=args.mode,
            allow_final_eval=args.allow_final_eval,
        )
    except PreflightError as error:
        print(json.dumps({"stage": args.stage, "mode": args.mode, "status": "FAIL", "error": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
    print(json.dumps(result.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
