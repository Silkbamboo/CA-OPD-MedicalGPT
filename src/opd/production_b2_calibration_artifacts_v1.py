"""Atomic artifact store and disk-derived finalizer for P4.8 calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from src.opd.production_b2_calibration_contract_v1 import (
    B2_CALIBRATION_STEPS,
    B2CalibrationContractV1Error,
    FRESH_STUDENT_INITIALIZATION,
    SELECTED_RESPONSE_LENGTH,
    SUPPORTED_RESPONSE_LENGTHS,
    canonical_json_sha256,
    evaluate_calibration_length_gate,
    validate_calibration_chain,
    validate_step_record,
)


_INITIAL_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "initialization",
    "logical_version",
    "adapter_sha256",
    "qualification_v2_sha256",
    "differs_from_qualification_v2",
    "source_adapter_path",
    "tensor_count",
    "lora_b_tensor_count",
    "nonzero_lora_b_value_count",
    "zero_effect_verified",
    "base_gradient_tensor_count",
}
_RELOAD_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "logical_version",
    "trainer_adapter_sha256",
    "runtime_adapter_sha256",
    "fresh_adapter_sha256",
    "tensor_count",
    "same_path_max_gap",
    "finite_rate",
}
_RESUME_RELOAD_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "logical_version",
    "trainer_adapter_sha256",
    "runtime_adapter_sha256",
    "checkpoint_adapter_sha256",
    "optimizer_state_restored",
    "rng_state_restored",
    "data_cursor",
    "tensor_count",
}
_CLEANUP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "worker_exited",
    "cleanup_complete",
    "gpu_memory_used_mib",
    "compute_pids",
    "residual_worker_pids",
    "isolation",
}
_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer_state.pt",
    "rng_state.pt",
    "calibration_state.json",
}
_FORBIDDEN_PRIVACY_TOKENS = (
    '"question"',
    '"answer"',
    '"label"',
    '"response"',
    '"completion"',
    '"prompt_text"',
    '"response_text"',
)


class B2CalibrationArtifactsV1Error(RuntimeError):
    """A durable calibration artifact or finalizer gate failed."""


def _fail(message: str) -> None:
    raise B2CalibrationArtifactsV1Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _atomic_bytes(path, _canonical_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is absent or a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2CalibrationArtifactsV1Error(
            f"{label} is invalid: {type(error).__name__}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _sha(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} is not a lowercase SHA-256")
    return value


def _privacy_bytes(payload: bytes) -> None:
    lowered = payload.decode("utf-8", errors="strict").lower()
    if any(token in lowered for token in _FORBIDDEN_PRIVACY_TOKENS):
        _fail("privacy-sensitive raw medical content field is forbidden")


def _selected_length(
    config: Mapping[str, Any], binding: Mapping[str, Any] | None = None
) -> int:
    value = config.get("selected_response_length")
    if value not in SUPPORTED_RESPONSE_LENGTHS:
        _fail("selected response length is not a registered package value")
    if (
        binding is not None
        and "selected_response_length" in binding
        and binding.get("selected_response_length") != value
    ):
        _fail("config/package selected response length differs")
    return int(value)


def _file_entry(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"indexed artifact is absent or a symlink: {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _stream_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_index(
    root: Path,
    name: str,
    *,
    kind: str,
    run_id: str,
    excluded: set[str],
) -> dict[str, Any]:
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).as_posix() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    value = {
        "schema_version": 1,
        "artifact_kind": kind,
        "run_id": run_id,
        "artifact_count": len(paths),
        "artifacts": [_file_entry(root, path) for path in paths],
    }
    metadata = _atomic_json(root / name, value)
    return {**value, **metadata}


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail("metrics.jsonl is absent or a symlink")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"metrics.jsonl line {line_number} is empty")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise B2CalibrationArtifactsV1Error(
                    f"metrics.jsonl line {line_number} is invalid"
                ) from error
            if not isinstance(value, dict):
                _fail(f"metrics.jsonl line {line_number} is not an object")
            records.append(value)
    return records


def _validate_initial_identity(
    raw: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _INITIAL_FIELDS:
        _fail("initial Student identity fields are not exact")
    value = dict(raw)
    if not (
        value["schema_version"] == 1
        and value["artifact_kind"] == "b2_fresh_student_initial_identity_v1"
        and value["run_id"] == run_id
        and value["initialization"] == FRESH_STUDENT_INITIALIZATION
        and value["logical_version"] == 0
        and value["source_adapter_path"] is None
        and value["differs_from_qualification_v2"] is True
        and value["adapter_sha256"] != value["qualification_v2_sha256"]
        and value["tensor_count"] == 504
        and value["lora_b_tensor_count"] == 252
        and value["nonzero_lora_b_value_count"] == 0
        and value["zero_effect_verified"] is True
        and value["base_gradient_tensor_count"] == 0
    ):
        _fail("fresh Student initialization identity failed closed")
    _sha(value["adapter_sha256"], "initial adapter")
    _sha(value["qualification_v2_sha256"], "qualification v2 adapter")
    return value


class B2CalibrationArtifactStoreV1:
    """Append-only worker store; terminal claims are owned by the finalizer."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        package_binding: Mapping[str, Any],
        data_manifest: Mapping[str, Any],
    ) -> None:
        self.output = Path(output_dir).resolve()
        self.run_id = run_id
        self.config = dict(config)
        self.metadata = dict(metadata)
        self.package_binding = dict(package_binding)
        self.data_manifest = dict(data_manifest)
        self.selected_response_length = _selected_length(
            self.config, self.package_binding
        )

    def initialize(self) -> None:
        if self.output.exists() or self.output.is_symlink():
            _fail("calibration output is not fresh")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output.name}.staging.", dir=self.output.parent
            )
        )
        try:
            for name, value in (
                ("config.yaml", self.config),
                ("metadata.json", self.metadata),
                ("package_binding.json", self.package_binding),
                ("data_manifest.json", self.data_manifest),
            ):
                payload = _canonical_bytes(value)
                _privacy_bytes(payload)
                _atomic_bytes(staging / name, payload)
            _atomic_json(
                staging / "cost.json",
                {
                    "schema_version": 1,
                    "artifact_kind": "b2_calibration_cost_v1",
                    "run_id": self.run_id,
                    "estimated_cost_cny": {
                        "lower": 4.44,
                        "upper": 11.84,
                        "basis": (
                            "90-240 GPU minutes at historical 2.96 CNY/hour"
                        ),
                    },
                    "actual_cost_cny": None,
                    "actual_cost_source": "platform_bill_unavailable",
                },
            )
            _atomic_bytes(staging / "metrics.jsonl", b"")
            _atomic_bytes(staging / "stdout.log", b"")
            (staging / "steps").mkdir()
            (staging / "checkpoints").mkdir()
            os.replace(staging, self.output)
            directory = os.open(
                self.output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
            raise

    def commit_initial_identity(self, raw: Mapping[str, Any]) -> None:
        value = _validate_initial_identity(raw, run_id=self.run_id)
        _atomic_json(self.output / "initial_identity.json", value)

    def commit_step(self, raw: Mapping[str, Any]) -> None:
        records = _read_metrics(self.output / "metrics.jsonl")
        try:
            value = validate_step_record(
                raw,
                expected_step=len(records) + 1,
                expected_version=len(records),
                selected_response_length=self.selected_response_length,
            )
        except B2CalibrationContractV1Error as error:
            raise B2CalibrationArtifactsV1Error(str(error)) from error
        if value["run_id"] != self.run_id:
            _fail("B2 step run identity differs")
        payload = _canonical_bytes(value)
        _privacy_bytes(payload)
        step_path = self.output / "steps" / f"step_{len(records) + 1:02d}.json"
        if step_path.exists() or step_path.is_symlink():
            _fail("B2 step artifact already exists")
        _atomic_bytes(step_path, payload)
        with (self.output / "metrics.jsonl").open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def commit_final_reload(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping) or set(raw) != _RELOAD_FIELDS:
            _fail("final reload identity fields are not exact")
        value = dict(raw)
        if not (
            value["schema_version"] == 1
            and value["artifact_kind"] == "b2_final_checkpoint_reload_identity_v1"
            and value["run_id"] == self.run_id
            and value["logical_version"] == B2_CALIBRATION_STEPS
            and value["tensor_count"] == 504
            and isinstance(value["same_path_max_gap"], (int, float))
            and isinstance(value["finite_rate"], (int, float))
        ):
            _fail("final reload identity envelope is invalid")
        for field in (
            "trainer_adapter_sha256",
            "runtime_adapter_sha256",
            "fresh_adapter_sha256",
        ):
            _sha(value[field], field)
        _atomic_json(self.output / "final_reload_identity.json", value)

    def commit_resume_reload(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping) or set(raw) != _RESUME_RELOAD_FIELDS:
            _fail("midpoint resume reload identity fields are not exact")
        value = dict(raw)
        if not (
            value["schema_version"] == 1
            and value["artifact_kind"] == "b2_resume_reload_identity_v1"
            and value["run_id"] == self.run_id
            and value["logical_version"] == 10
            and value["trainer_adapter_sha256"]
            == value["runtime_adapter_sha256"]
            == value["checkpoint_adapter_sha256"]
            and value["optimizer_state_restored"] is True
            and value["rng_state_restored"] is True
            and value["data_cursor"] == 40
            and value["tensor_count"] == 504
        ):
            _fail("midpoint resume reload identity failed closed")
        _sha(value["trainer_adapter_sha256"], "resume reload adapter")
        _atomic_json(self.output / "resume_reload_identity_v10.json", value)


def _validate_cleanup(raw: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw, Mapping) or set(raw) != _CLEANUP_FIELDS:
        _fail("cleanup observation fields are not exact")
    value = dict(raw)
    isolation = value.get("isolation")
    valid = bool(
        value.get("schema_version") == 1
        and value.get("artifact_kind") == "b2_calibration_post_worker_cleanup_v1"
        and value.get("worker_exited") is True
        and value.get("cleanup_complete") is True
        and value.get("gpu_memory_used_mib") == [0, 0]
        and value.get("compute_pids") == []
        and value.get("residual_worker_pids") == []
        and isinstance(isolation, Mapping)
        and all(
            isolation.get(field) is False
            for field in (
                "final_access",
                "controller_access",
                "confirmation_access",
                "label_access",
            )
        )
    )
    return value, valid


def _validate_checkpoints(
    output: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entries = []
    by_version = {
        int(record["next_policy_version"]): record for record in records
    }
    for version in (10, 20):
        directory = output / "checkpoints" / f"v{version}"
        manifest_path = directory / "checkpoint_manifest.json"
        manifest = _read_json(manifest_path, f"checkpoint v{version} manifest")
        transport_path = directory / "adapter_transport_manifest.json"
        transport = _read_json(
            transport_path, f"checkpoint v{version} adapter transport"
        )
        files = manifest.get("files")
        actual = {
            path.name
            for path in directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("artifact_kind") == "b2_resume_checkpoint_manifest_v1"
            and manifest.get("logical_version") == version
            and manifest.get("optimizer_step") == version
            and manifest.get("sampler_policy_version") == version
            and manifest.get("complete") is True
            and manifest.get("resume_eligible") is True
            and isinstance(files, Mapping)
            and set(files) == _CHECKPOINT_FILES
            and actual
            == _CHECKPOINT_FILES
            | {"checkpoint_manifest.json", "adapter_transport_manifest.json"}
            and version in by_version
            and manifest.get("adapter_sha256")
            == by_version[version]["trainer_authority_sha256"]
        ):
            _fail(f"checkpoint v{version} is partial, stale or ineligible")
        transport_files = transport.get("files")
        if not (
            transport.get("schema_version") == 1
            and transport.get("logical_version") == f"v{version}"
            and transport.get("aggregate_tensor_sha256")
            == manifest["adapter_sha256"]
            and isinstance(transport_files, list)
            and [item.get("path") for item in transport_files]
            == ["adapter_config.json", "adapter_model.safetensors"]
        ):
            _fail(f"checkpoint v{version} adapter transport differs")
        for item in transport_files:
            path = directory / item["path"]
            if not (
                isinstance(item, Mapping)
                and item.get("sha256") == _stream_sha256(path)
                and item.get("size_bytes") == path.stat().st_size
            ):
                _fail(f"checkpoint v{version} transport file binding differs")
        for name, metadata in files.items():
            path = directory / name
            if not (
                isinstance(metadata, Mapping)
                and not path.is_symlink()
                and path.is_file()
                and metadata.get("sha256") == _stream_sha256(path)
                and metadata.get("size_bytes") == path.stat().st_size
            ):
                _fail(f"checkpoint v{version} file binding differs")
        entries.append(
            {
                "logical_version": version,
                "path": f"checkpoints/v{version}",
                "adapter_sha256": manifest["adapter_sha256"],
                "manifest_sha256": _stream_sha256(manifest_path),
                "resume_eligible": True,
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_checkpoint_index_v1",
        "run_id": records[0]["run_id"],
        "checkpoint_strategy": "step10_step20_and_final",
        "checkpoints": entries,
        "final_checkpoint_version": 20,
    }


def _validate_final_reload(output: Path, final_sha: str) -> dict[str, Any]:
    value = _read_json(output / "final_reload_identity.json", "final reload identity")
    if not (
        set(value) == _RELOAD_FIELDS
        and value.get("logical_version") == 20
        and value.get("trainer_adapter_sha256")
        == value.get("runtime_adapter_sha256")
        == value.get("fresh_adapter_sha256")
        == final_sha
        and value.get("tensor_count") == 504
        and float(value.get("same_path_max_gap", 1.0)) <= 1e-4
        and float(value.get("finite_rate", 0.0)) == 1.0
    ):
        _fail("final checkpoint fresh reload identity differs")
    return value


def _validate_resume_reload(output: Path, expected_sha: str) -> dict[str, Any]:
    value = _read_json(
        output / "resume_reload_identity_v10.json", "midpoint resume reload"
    )
    if not (
        set(value) == _RESUME_RELOAD_FIELDS
        and value.get("schema_version") == 1
        and value.get("artifact_kind") == "b2_resume_reload_identity_v1"
        and value.get("logical_version") == 10
        and value.get("trainer_adapter_sha256")
        == value.get("runtime_adapter_sha256")
        == value.get("checkpoint_adapter_sha256")
        == expected_sha
        and value.get("optimizer_state_restored") is True
        and value.get("rng_state_restored") is True
        and value.get("data_cursor") == 40
        and value.get("tensor_count") == 504
    ):
        _fail("midpoint resume reload identity differs")
    return value


def _failure(
    status: str,
    reason: str,
    recommendation: Any = None,
    *,
    classification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_failure_v1",
        "status": status,
        "reason": reason,
        "recommendation": recommendation,
        "B2_calibration_complete": False,
        "B2_formal_authorized": False,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    if classification is not None:
        result.update(
            {
                "primary_failure_code": classification["primary_failure_code"],
                "failure_phase": classification["failure_phase"],
                "completed_steps": classification["completed_steps"],
                "requested_steps": classification["requested_steps"],
                "unmet_success_gates": list(
                    classification["unmet_success_gates"]
                ),
                "causal_chain": list(classification["causal_chain"]),
            }
        )
    return result


def classify_calibration_failure(
    *,
    primary_failure_code: str | None,
    failure_phase: str | None,
    completed_steps: int,
    requested_steps: int,
    causal_chain: Sequence[str],
) -> dict[str, Any]:
    """Preserve a precise disk-recorded cause ahead of generic success gates."""

    if not (
        isinstance(completed_steps, int)
        and isinstance(requested_steps, int)
        and 0 <= completed_steps <= requested_steps
        and requested_steps > 0
        and isinstance(causal_chain, Sequence)
        and not isinstance(causal_chain, (str, bytes))
        and all(isinstance(item, str) and item for item in causal_chain)
    ):
        raise ValueError("failure classification inputs are invalid")
    unmet: list[str] = []
    if completed_steps != requested_steps:
        unmet.append("optimizer_step_count_is_not_exactly_20")
    if primary_failure_code is not None:
        if not (
            isinstance(primary_failure_code, str)
            and primary_failure_code.startswith("failed_")
            and isinstance(failure_phase, str)
            and failure_phase
        ):
            raise ValueError("primary failure code/phase is invalid")
        status = primary_failure_code
        chain = list(causal_chain)
        if not chain or chain[0] != primary_failure_code:
            chain.insert(0, primary_failure_code)
    elif completed_steps != requested_steps:
        status = "failed_b2_calibration_step_count"
        chain = list(causal_chain) or [status]
    else:
        raise ValueError("no failure is present")
    return {
        "primary_failure_code": status,
        "status": status,
        "failure_phase": failure_phase,
        "completed_steps": completed_steps,
        "requested_steps": requested_steps,
        "unmet_success_gates": unmet,
        "causal_chain": chain,
    }


def finalize_calibration_run(
    output_dir: str | Path,
    *,
    cleanup_observation: Mapping[str, Any],
    caller_ready: bool | None = None,
) -> dict[str, Any]:
    """Seal success or failure after the GPU worker has actually exited."""

    del caller_ready
    output = Path(output_dir).resolve()
    if output.is_symlink() or not output.is_dir():
        _fail("calibration output directory is absent")
    metadata = _read_json(output / "metadata.json", "metadata")
    config = _read_json(output / "config.yaml", "config")
    binding = _read_json(output / "package_binding.json", "package binding")
    selected_response_length = _selected_length(config, binding)
    run_id = str(config.get("run_id", ""))
    initial: dict[str, Any] | None = None
    initialization_error: str | None = None
    try:
        initial = _read_json(output / "initial_identity.json", "initial identity")
        initial = _validate_initial_identity(initial, run_id=run_id)
    except (B2CalibrationArtifactsV1Error, KeyError, OSError) as error:
        initialization_error = type(error).__name__
    cleanup_value, cleanup_ok = _validate_cleanup(cleanup_observation)
    _atomic_json(output / "cleanup.json", cleanup_value)
    metrics_error: str | None = None
    try:
        records = _read_metrics(output / "metrics.jsonl")
    except (B2CalibrationArtifactsV1Error, OSError, UnicodeError) as error:
        records = []
        metrics_error = type(error).__name__
    worker_classification: dict[str, Any] | None = None
    worker_status_path = output / "worker_status.json"
    if worker_status_path.exists() or worker_status_path.is_symlink():
        worker_status = _read_json(worker_status_path, "worker status")
        if worker_status.get("status") == "worker_failed" and worker_status.get(
            "primary_failure_code"
        ) is not None:
            try:
                worker_classification = classify_calibration_failure(
                    primary_failure_code=worker_status.get(
                        "primary_failure_code"
                    ),
                    failure_phase=worker_status.get("failure_phase"),
                    completed_steps=int(worker_status.get("completed_steps")),
                    requested_steps=int(worker_status.get("requested_steps")),
                    causal_chain=worker_status.get("causal_chain", []),
                )
            except (TypeError, ValueError):
                worker_classification = classify_calibration_failure(
                    primary_failure_code="failed_artifact_integrity",
                    failure_phase="worker_failure_artifact",
                    completed_steps=len(records),
                    requested_steps=B2_CALIBRATION_STEPS,
                    causal_chain=[
                        "failed_artifact_integrity",
                        "invalid_worker_failure_classification",
                    ],
                )
    status = "b2_calibration_complete_ready_for_b2_formal"
    reason = "all_exact_20_step_disk_gates_passed"
    recommendation = None
    live_abort = None
    live_abort_path = output / "length_abort_recommendation.json"
    if live_abort_path.exists() or live_abort_path.is_symlink():
        live_abort = _read_json(live_abort_path, "live length abort")
    chain: dict[str, Any] | None = None
    checkpoint_index: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_checkpoint_index_v1",
        "run_id": run_id,
        "checkpoint_strategy": "step10_step20_and_final",
        "checkpoints": [],
        "final_checkpoint_version": None,
    }
    if worker_classification is not None:
        status = worker_classification["status"]
        reason = "disk_recorded_primary_worker_failure"
    elif not cleanup_ok:
        status = "failed_b2_calibration_cleanup"
        reason = "post_worker_cleanup_not_complete"
    elif metadata.get("execution_mode") != "formal_gpu":
        status = "failed_non_formal_execution_mode"
        reason = "mock_or_nonformal_execution_cannot_authorize"
    elif initialization_error is not None or initial is None:
        status = "failed_b2_calibration_initialization"
        reason = f"fresh_student_identity_absent_or_invalid:{initialization_error}"
    elif metrics_error is not None:
        status = "failed_artifact_integrity"
        reason = f"metrics_jsonl_absent_or_invalid:{metrics_error}"
    elif live_abort is not None:
        recommendation_valid = (
            isinstance(live_abort.get("escalation_recommendation"), Mapping)
            and live_abort["escalation_recommendation"].get(
                "recommended_response_length"
            )
            == 1024
            and live_abort["escalation_recommendation"].get(
                "same_run_switch_allowed"
            )
            is False
        ) if selected_response_length == 768 else (
            live_abort.get("escalation_recommendation") is None
        )
        if not (
            live_abort.get("status")
            == "failed_b2_calibration_length_insufficient"
            and live_abort.get("passed") is False
            and live_abort.get("selected_response_length")
            == selected_response_length
            and recommendation_valid
        ):
            status = "failed_b2_calibration_length_contract"
            reason = "live_length_abort_artifact_invalid"
        else:
            status = "failed_b2_calibration_length_insufficient"
            reason = (
                f"frozen_{selected_response_length}_rolling_window_"
                "truncation_gate_failed"
            )
            recommendation = live_abort["escalation_recommendation"]
    elif len(records) != B2_CALIBRATION_STEPS:
        status = "failed_b2_calibration_step_count"
        reason = "optimizer_step_count_is_not_exactly_20"
    else:
        try:
            chain = validate_calibration_chain(
                records,
                initial_adapter_sha256=initial["adapter_sha256"],
                final_reload_adapter_sha256=records[-1][
                    "trainer_authority_sha256"
                ],
                forbidden_qualification_adapter_sha256=binding[
                    "qualification_v2_tensor_sha256"
                ],
                selected_response_length=selected_response_length,
            )
        except (B2CalibrationContractV1Error, KeyError) as error:
            status = "failed_b2_calibration_step_chain"
            reason = f"step_chain_gate:{type(error).__name__}"
        if chain is not None:
            try:
                checkpoint_index = _validate_checkpoints(output, records)
            except B2CalibrationArtifactsV1Error:
                status = "failed_b2_calibration_checkpoint_integrity"
                reason = "checkpoint_index_or_resume_state_invalid"
            else:
                try:
                    _validate_resume_reload(
                        output, records[9]["trainer_authority_sha256"]
                    )
                except B2CalibrationArtifactsV1Error:
                    status = "failed_b2_calibration_checkpoint_integrity"
                    reason = "midpoint_resume_reload_identity_invalid"
                    chain = None
                try:
                    if chain is None:
                        raise B2CalibrationArtifactsV1Error(
                            "midpoint resume reload failed"
                        )
                    _validate_final_reload(output, chain["final_adapter_sha256"])
                except B2CalibrationArtifactsV1Error:
                    if status != "failed_b2_calibration_checkpoint_integrity":
                        status = "failed_b2_calibration_final_reload_identity"
                        reason = "step20_trainer_runtime_fresh_identity_mismatch"
                else:
                    try:
                        length = evaluate_calibration_length_gate(
                            records,
                            selected_response_length=selected_response_length,
                        )
                    except B2CalibrationContractV1Error as error:
                        status = "failed_b2_calibration_length_contract"
                        reason = f"length_gate:{type(error).__name__}"
                    else:
                        if not length["passed"]:
                            status = "failed_b2_calibration_length_insufficient"
                            reason = (
                                f"frozen_{selected_response_length}_window_"
                                "truncation_gate_failed"
                            )
                            recommendation = length["escalation_recommendation"]
    _atomic_json(output / "checkpoints" / "index.json", checkpoint_index)
    length_summary = None
    if len(records) == B2_CALIBRATION_STEPS:
        try:
            length_summary = evaluate_calibration_length_gate(
                records,
                selected_response_length=selected_response_length,
            )
        except B2CalibrationContractV1Error:
            length_summary = None
    cost = _read_json(output / "cost.json", "cost")
    resources = {
        "wall_time_seconds": sum(
            float(record["timings_seconds"]["step"]) for record in records
        ),
        "mean_rollout_tokens_per_second": (
            sum(
                float(record["throughput"]["rollout_tokens_per_second"])
                for record in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "mean_scorer_tokens_per_second": (
            sum(
                float(record["throughput"]["scorer_tokens_per_second"])
                for record in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "gpu0_peak_bytes": max(
            (int(record["gpu_memory_bytes"]["gpu0_peak"]) for record in records),
            default=0,
        ),
        "gpu1_peak_bytes": max(
            (int(record["gpu_memory_bytes"]["gpu1_peak"]) for record in records),
            default=0,
        ),
        "minimum_disk_remaining_bytes": min(
            (int(record["disk_remaining_bytes"]) for record in records),
            default=0,
        ),
        "estimated_cost_cny": cost.get("estimated_cost_cny"),
        "actual_cost_cny": cost.get("actual_cost_cny"),
    }
    summary = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_summary_v1",
        "run_id": run_id,
        "status": status,
        "steps_completed": len(records),
        "initial_policy_version": 0,
        "final_policy_version": (records[-1]["next_policy_version"] if records else 0),
        "selected_response_length": selected_response_length,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "initial_adapter_sha256": (
            initial.get("adapter_sha256") if initial is not None else None
        ),
        "final_adapter_sha256": (
            records[-1].get("trainer_authority_sha256") if records else None
        ),
        "step_chain_sha256": (chain.get("step_chain_sha256") if chain else None),
        "length_gate": length_summary,
        "resources": resources,
        "cleanup_complete": cleanup_ok,
        "primary_failure_code": (
            worker_classification["primary_failure_code"]
            if worker_classification is not None
            else (None if status == "b2_calibration_complete_ready_for_b2_formal" else status)
        ),
        "failure_phase": (
            worker_classification["failure_phase"]
            if worker_classification is not None
            else None
        ),
        "requested_steps": B2_CALIBRATION_STEPS,
        "unmet_success_gates": (
            list(worker_classification["unmet_success_gates"])
            if worker_classification is not None
            else (
                ["optimizer_step_count_is_not_exactly_20"]
                if len(records) != B2_CALIBRATION_STEPS
                else []
            )
        ),
        "causal_chain": (
            list(worker_classification["causal_chain"])
            if worker_classification is not None
            else ([] if status == "b2_calibration_complete_ready_for_b2_formal" else [status])
        ),
        "B2_calibration_complete": status
        == "b2_calibration_complete_ready_for_b2_formal",
        "B2_formal_authorization_candidate": status
        == "b2_calibration_complete_ready_for_b2_formal",
        "B2_formal_authorized": False,
        "full_B2_started": False,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }
    _atomic_json(output / "summary.json", summary)
    success = status == "b2_calibration_complete_ready_for_b2_formal"
    formal_candidate_path = output / "b2_formal_run_card_candidate.json"
    if success:
        seconds_per_step = resources["wall_time_seconds"] / B2_CALIBRATION_STEPS
        _atomic_json(
            formal_candidate_path,
            {
                "schema_version": 1,
                "artifact_kind": "b2_formal_run_card_candidate_v1",
                "run_id": run_id,
                "source_calibration_status": status,
                "source_calibration_final_adapter_sha256": summary[
                    "final_adapter_sha256"
                ],
                "student_initialization": FRESH_STUDENT_INITIALIZATION,
                "selected_response_length": selected_response_length,
                "recommended_optimizer_steps": {
                    "minimum": 120,
                    "maximum": 150,
                    "status": "candidate_pending_separate_user_authorization",
                },
                "estimated_runtime_seconds": {
                    "for_120_steps": seconds_per_step * 120.0,
                    "for_150_steps": seconds_per_step * 150.0,
                    "source": "measured_calibration_mean_step_time",
                },
                "B2_formal_authorized": False,
                "automatically_start": False,
                "isolation": dict(summary["isolation"]),
            },
        )
    failure_path = output / "failure.json"
    if success:
        if failure_path.exists() or failure_path.is_symlink():
            _fail("success output contains a failure artifact")
    else:
        classification = worker_classification
        if classification is None:
            classification = {
                "primary_failure_code": status,
                "failure_phase": None,
                "completed_steps": len(records),
                "requested_steps": B2_CALIBRATION_STEPS,
                "unmet_success_gates": (
                    ["optimizer_step_count_is_not_exactly_20"]
                    if len(records) != B2_CALIBRATION_STEPS
                    else []
                ),
                "causal_chain": [status],
            }
        _atomic_json(
            output / "failure.json",
            _failure(
                status,
                reason,
                recommendation,
                classification=classification,
            ),
        )
    evidence = _write_index(
        output,
        "evidence_index.json",
        kind="b2_calibration_evidence_index_v1",
        run_id=run_id,
        excluded={"evidence_index.json", "final_index.json", "readiness.json"},
    )
    final_index = _write_index(
        output,
        "final_index.json",
        kind="b2_calibration_final_index_v1",
        run_id=run_id,
        excluded={"final_index.json", "readiness.json"},
    )
    readiness = {
        "schema_version": 1,
        "artifact_kind": "b2_calibration_readiness_v1",
        "run_id": run_id,
        "status": status,
        "ready": success,
        "steps_completed": len(records),
        "selected_response_length": selected_response_length,
        "student_initialization": FRESH_STUDENT_INITIALIZATION,
        "metrics_sha256": _stream_sha256(output / "metrics.jsonl"),
        "evidence_index_sha256": evidence["sha256"],
        "final_index_sha256": final_index["sha256"],
        "cleanup_complete": cleanup_ok,
        "primary_failure_code": summary["primary_failure_code"],
        "failure_phase": summary["failure_phase"],
        "requested_steps": B2_CALIBRATION_STEPS,
        "unmet_success_gates": summary["unmet_success_gates"],
        "causal_chain": summary["causal_chain"],
        "B2_authorized": True,
        "B2_calibration_started": bool(records),
        "B2_calibration_complete": success,
        "B2_formal_authorization_candidate": success,
        "B2_formal_authorized": False,
        "full_B2_started": False,
        "isolation": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }
    _atomic_json(output / "readiness.json", readiness)
    return dict(summary)


def _revalidate_index(root: Path, path: Path, *, expected_kind: str) -> dict[str, Any]:
    value = _read_json(path, expected_kind)
    entries = value.get("artifacts")
    if not (
        value.get("schema_version") == 1
        and value.get("artifact_kind") == expected_kind
        and isinstance(entries, list)
        and value.get("artifact_count") == len(entries)
    ):
        _fail("artifact index envelope is invalid")
    seen: set[str] = set()
    for entry in entries:
        if not (
            isinstance(entry, Mapping)
            and set(entry) == {"path", "sha256", "size_bytes"}
            and isinstance(entry.get("path"), str)
            and entry["path"] not in seen
        ):
            _fail("artifact index entry is invalid")
        seen.add(entry["path"])
        item = root / entry["path"]
        if not (
            not item.is_symlink()
            and item.is_file()
            and entry["sha256"] == _stream_sha256(item)
            and entry["size_bytes"] == item.stat().st_size
        ):
            _fail("artifact index SHA/size differs from disk")
    return value


def recompute_calibration_readiness(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    false = {
        "status": "failed_artifact_revalidation",
        "ready": False,
        "B2_calibration_complete": False,
        "B2_formal_authorization_candidate": False,
        "B2_formal_authorized": False,
    }
    try:
        readiness = _read_json(output / "readiness.json", "readiness")
        final_index_path = output / "final_index.json"
        final_index = _revalidate_index(
            output,
            final_index_path,
            expected_kind="b2_calibration_final_index_v1",
        )
        summary = _read_json(output / "summary.json", "summary")
        config = _read_json(output / "config.yaml", "config")
        cleanup = _read_json(output / "cleanup.json", "cleanup")
        cleanup_value, cleanup_ok = _validate_cleanup(cleanup)
        evidence_index_path = output / "evidence_index.json"
        evidence_index = _revalidate_index(
            output,
            evidence_index_path,
            expected_kind="b2_calibration_evidence_index_v1",
        )
        initial = _read_json(output / "initial_identity.json", "initial identity")
        binding = _read_json(output / "package_binding.json", "package binding")
        selected_response_length = _selected_length(config, binding)
        initial = _validate_initial_identity(
            initial, run_id=str(readiness.get("run_id", ""))
        )
        records = _read_metrics(output / "metrics.jsonl")
        chain = validate_calibration_chain(
            records,
            initial_adapter_sha256=initial["adapter_sha256"],
            final_reload_adapter_sha256=records[-1]["trainer_authority_sha256"],
            forbidden_qualification_adapter_sha256=binding[
                "qualification_v2_tensor_sha256"
            ],
            selected_response_length=selected_response_length,
        )
        _validate_checkpoints(output, records)
        _validate_resume_reload(output, records[9]["trainer_authority_sha256"])
        _validate_final_reload(output, chain["final_adapter_sha256"])
        length = evaluate_calibration_length_gate(
            records, selected_response_length=selected_response_length
        )
        success = bool(
            readiness.get("artifact_kind") == "b2_calibration_readiness_v1"
            and readiness.get("status")
            == "b2_calibration_complete_ready_for_b2_formal"
            and readiness.get("ready") is True
            and readiness.get("steps_completed") == B2_CALIBRATION_STEPS
            and readiness.get("selected_response_length")
            == selected_response_length
            and summary.get("selected_response_length")
            == selected_response_length
            and readiness.get("student_initialization")
            == FRESH_STUDENT_INITIALIZATION
            and readiness.get("metrics_sha256")
            == _stream_sha256(output / "metrics.jsonl")
            and readiness.get("evidence_index_sha256")
            == _stream_sha256(evidence_index_path)
            and readiness.get("final_index_sha256")
            == _stream_sha256(final_index_path)
            and readiness.get("cleanup_complete") is True
            and readiness.get("B2_calibration_complete") is True
            and readiness.get("B2_formal_authorization_candidate") is True
            and readiness.get("B2_formal_authorized") is False
            and readiness.get("full_B2_started") is False
            and summary.get("status") == readiness.get("status")
            and summary.get("steps_completed") == B2_CALIBRATION_STEPS
            and cleanup_ok
            and cleanup_value.get("cleanup_complete") is True
            and len(records) == B2_CALIBRATION_STEPS
            and chain.get("steps_completed") == B2_CALIBRATION_STEPS
            and length.get("passed") is True
            and (output / "b2_formal_run_card_candidate.json").is_file()
            and not (output / "failure.json").exists()
            and evidence_index.get("artifact_count", 0) > 0
            and final_index.get("artifact_count", 0) > 0
        )
        if not success:
            return {**false, "status": str(readiness.get("status", false["status"]))}
        return readiness
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        IndexError,
        B2CalibrationArtifactsV1Error,
        B2CalibrationContractV1Error,
    ):
        return false


def assert_formal_b2_calibration_candidate(output_dir: str | Path) -> dict[str, Any]:
    readiness = recompute_calibration_readiness(output_dir)
    if not (
        readiness.get("ready") is True
        and readiness.get("B2_calibration_complete") is True
        and readiness.get("B2_formal_authorization_candidate") is True
        and readiness.get("B2_formal_authorized") is False
    ):
        _fail("formal B2 requires disk-derived calibration success")
    summary = _read_json(Path(output_dir) / "summary.json", "summary")
    formal = _read_json(
        Path(output_dir) / "b2_formal_run_card_candidate.json",
        "formal B2 run-card candidate",
    )
    if not (
        formal.get("artifact_kind") == "b2_formal_run_card_candidate_v1"
        and formal.get("source_calibration_status") == readiness.get("status")
        and formal.get("selected_response_length")
        == summary.get("selected_response_length")
        and formal.get("B2_formal_authorized") is False
        and formal.get("automatically_start") is False
    ):
        _fail("formal B2 run-card candidate binding differs")
    return {
        "status": "formal_b2_candidate_requires_separate_user_authorization",
        "steps_completed": summary["steps_completed"],
        "selected_response_length": summary["selected_response_length"],
        "final_adapter_sha256": summary["final_adapter_sha256"],
        "recommended_optimizer_steps": formal["recommended_optimizer_steps"],
        "estimated_runtime_seconds": formal["estimated_runtime_seconds"],
        "B2_formal_authorized": False,
        "full_B2_started": False,
    }


__all__ = [
    "B2CalibrationArtifactStoreV1",
    "B2CalibrationArtifactsV1Error",
    "assert_formal_b2_calibration_candidate",
    "finalize_calibration_run",
    "recompute_calibration_readiness",
]
