"""Fail-closed P4.6 parent evidence verification for P4.7 length continuation.

The verifier deliberately uses only the Python standard library.  Adapter
weights are hashed and inspected as a safetensors byte stream; no model,
torch, CUDA, tokenizer, or training runtime is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CORE_PHASES = (
    "launch_record",
    "preflight",
    "probe_manifest",
    "v0_guard",
    "reconstruction_step0",
    "authority_v1",
    "refresh_v1",
    "trajectory_step1_manifest",
    "reconstruction_step1",
    "authority_v2",
    "refresh_v2",
    "base_null",
)

_BINDING_FIELDS = (
    "schema_version",
    "artifact_protocol_version",
    "run_id",
    "attempt_id",
    "git_commit",
    "config_sha256",
    "protocol_sha256",
    "schema_sha256",
    "run_card_sha256",
    "backend_binding_sha256",
    "prompt_manifest_sha256",
    "data_manifest_sha256",
    "probe_spec_sha256",
)
_HEX = re.compile(r"^[0-9a-f]+$")
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


class ParentReuseVerificationError(RuntimeError):
    """A stable, structured failure raised before parent reuse is allowed."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def canonical_json_sha256(value: Any) -> str:
    """Return the deterministic canonical JSON digest used by P4.6/P4.7."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file_stream(path: Path, *, chunk_bytes: int = _STREAM_CHUNK_BYTES) -> str:
    """Hash one regular file with bounded memory."""

    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fail(code: str, detail: str) -> None:
    raise ParentReuseVerificationError(code, detail)


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and bool(_HEX.fullmatch(value))


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("parent_spec_invalid", f"{label} path is empty")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("parent_spec_invalid", f"{label} path is unsafe")
    return pure.as_posix()


def _assert_no_symlink(root: Path, relative: str, *, code: str) -> Path:
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            _fail(code, f"symlink rejected at {relative}")
    return path


def _regular_file(
    root: Path,
    relative: str,
    *,
    missing_code: str,
    symlink_code: str,
) -> Path:
    relative = _safe_relative(relative, label="artifact")
    path = _assert_no_symlink(root, relative, code=symlink_code)
    if not path.is_file():
        _fail(missing_code, f"regular file is absent: {relative}")
    return path


def _verify_file(
    root: Path,
    relative: str,
    expected_sha256: Any,
    *,
    expected_size: Any | None = None,
    missing_code: str,
    symlink_code: str,
    mismatch_code: str,
) -> dict[str, Any]:
    if not _is_hex(expected_sha256, 64):
        _fail("parent_spec_invalid", f"invalid expected SHA for {relative}")
    path = _regular_file(
        root,
        relative,
        missing_code=missing_code,
        symlink_code=symlink_code,
    )
    before = path.stat()
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or before.st_size != expected_size
    ):
        _fail(mismatch_code, f"size mismatch: {relative}")
    actual = sha256_file_stream(path)
    after = path.stat()
    if (
        actual != expected_sha256
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        _fail(mismatch_code, f"SHA or file identity mismatch: {relative}")
    return {
        "path": relative,
        "sha256": actual,
        "size_bytes": after.st_size,
    }


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(code, f"invalid JSON: {path.name}: {type(error).__name__}")
    if not isinstance(value, dict):
        _fail(code, f"JSON root is not an object: {path.name}")
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_status(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("status") != "pass":
        _fail("parent_core_gate_failed", f"{label} status is not pass")


def _verify_reconstruction(payload: Mapping[str, Any], *, label: str) -> None:
    _require_status(payload, label=label)
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping):
        _fail("parent_core_gate_failed", f"{label} telemetry is absent")
    q = telemetry.get("q_p_old")
    advantage = telemetry.get("advantage")
    update = telemetry.get("optimizer_update")
    if not all(isinstance(value, Mapping) for value in (q, advantage, update)):
        _fail("parent_core_gate_failed", f"{label} telemetry sections are absent")
    valid_count = q.get("valid_token_count")
    if not (
        isinstance(valid_count, int)
        and not isinstance(valid_count, bool)
        and valid_count > 0
        and q.get("finite_rate") == 1.0
        and _finite_number(q.get("ess_fraction"))
        and q["ess_fraction"] >= 0.80
        and _finite_number(q.get("cap_fraction"))
        and q["cap_fraction"] <= 0.05
        and _finite_number(q.get("current_pre_old_max_abs"))
        and q["current_pre_old_max_abs"] <= 0.0001
    ):
        _fail("parent_core_gate_failed", f"{label} q/p_old gate failed")
    counts = [
        advantage.get("positive_count"),
        advantage.get("negative_count"),
        advantage.get("near_zero_count"),
    ]
    if not (
        advantage.get("count") == valid_count
        and advantage.get("finite_rate") == 1.0
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts)
        and sum(counts) == valid_count
    ):
        _fail("parent_core_gate_failed", f"{label} advantage gate failed")
    trainable = update.get("trainable_tensor_count")
    nonzero = update.get("nonzero_update_tensor_count")
    zero = update.get("zero_update_tensor_count")
    if not (
        _finite_number(update.get("objective_delta"))
        and update["objective_delta"] > 0
        and _finite_number(update.get("loss_delta"))
        and update["loss_delta"] < 0
        and _finite_number(update.get("alignment"))
        and update["alignment"] > 0
        and isinstance(trainable, int)
        and not isinstance(trainable, bool)
        and trainable > 0
        and isinstance(nonzero, int)
        and not isinstance(nonzero, bool)
        and nonzero > 0
        and isinstance(zero, int)
        and not isinstance(zero, bool)
        and zero >= 0
        and nonzero + zero == trainable
        and update.get("teacher_gradient_tensor_count") == 0
        and update.get("base_gradient_tensor_count") == 0
    ):
        _fail("parent_core_gate_failed", f"{label} optimizer gate failed")


def _validate_tensor_records(
    value: Any, *, tensor_count: Any, total_bytes: Any, label: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("parent_core_gate_failed", f"{label} tensor inventory is absent")
    records: list[dict[str, Any]] = []
    keys: set[str] = set()
    byte_sum = 0
    for raw in value:
        if not isinstance(raw, Mapping):
            _fail("parent_core_gate_failed", f"{label} tensor record is invalid")
        key = raw.get("canonical_key")
        shape = raw.get("shape")
        byte_length = raw.get("byte_length")
        if not (
            isinstance(key, str)
            and key
            and key not in keys
            and _is_hex(raw.get("sha256"), 64)
            and isinstance(shape, list)
            and shape
            and all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in shape)
            and isinstance(raw.get("dtype"), str)
            and bool(raw["dtype"])
            and isinstance(byte_length, int)
            and not isinstance(byte_length, bool)
            and byte_length == 4 * math.prod(shape)
        ):
            _fail("parent_core_gate_failed", f"{label} tensor record is invalid")
        keys.add(key)
        byte_sum += byte_length
        records.append(dict(raw))
    if tensor_count != len(records) or total_bytes != byte_sum:
        _fail("parent_core_gate_failed", f"{label} tensor count/bytes mismatch")
    return records


def _rebuild_aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    canonical: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = {
            "canonical_key": raw["canonical_key"],
            "sha256": raw["sha256"],
            "shape": list(raw["shape"]),
            "canonical_dtype": "float32_le",
            "canonical_byte_length": raw["byte_length"],
        }
        canonical[str(record["canonical_key"])] = record
    for key in sorted(canonical):
        encoded = json.dumps(
            canonical[key],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _verify_authority(payload: Mapping[str, Any], *, version: int) -> list[dict[str, Any]]:
    _require_status(payload, label=f"authority_v{version}")
    if not (
        payload.get("logical_version") == f"v{version}"
        and payload.get("runtime_adapter_name") == "student_active"
        and payload.get("active_adapter") == "student_active"
        and _is_hex(payload.get("aggregate_tensor_sha256"), 64)
        and _is_hex(payload.get("canonical_config_sha256"), 64)
    ):
        _fail("parent_core_gate_failed", f"authority_v{version} identity failed")
    records = _validate_tensor_records(
        payload.get("per_tensor_digests"),
        tensor_count=payload.get("tensor_count"),
        total_bytes=payload.get("total_bytes"),
        label=f"authority_v{version}",
    )
    if _rebuild_aggregate(records) != payload["aggregate_tensor_sha256"]:
        _fail("parent_core_gate_failed", f"authority_v{version} aggregate mismatch")
    same_path = payload.get("trainer_memory_reload_same_path")
    if not (
        isinstance(same_path, Mapping)
        and same_path.get("finite_rate") == 1.0
        and _finite_number(same_path.get("max"))
        and same_path["max"] <= 0.0001
    ):
        _fail("parent_core_gate_failed", f"authority_v{version} same-path failed")
    return records


def _verify_refresh(
    payload: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    version: int,
    previous: str | None,
) -> None:
    _require_status(payload, label=f"refresh_v{version}")
    aggregate = authority["aggregate_tensor_sha256"]
    same_path = payload.get("same_path")
    normal = payload.get("normal_request")
    stale = payload.get("stale_request")
    if not (
        payload.get("logical_version") == f"v{version}"
        and payload.get("trainer_tensor_sha256") == aggregate
        and payload.get("runtime_tensor_sha256") == aggregate
        and payload.get("fresh_tensor_sha256") == aggregate
        and payload.get("runtime_per_tensor_digests") == authority.get("per_tensor_digests")
        and payload.get("fresh_per_tensor_digests") == authority.get("per_tensor_digests")
        and payload.get("tensor_count") == authority.get("tensor_count")
        and payload.get("total_bytes") == authority.get("total_bytes")
        and payload.get("registry_before") == ["student_active"]
        and payload.get("registry_after") == ["student_active"]
        and payload.get("active_adapter") == "student_active"
        and payload.get("adapter_enabled") is True
        and payload.get("merged") is False
        and isinstance(same_path, Mapping)
        and same_path.get("finite_rate") == 1.0
        and _finite_number(same_path.get("max"))
        and same_path["max"] <= 0.0001
        and isinstance(normal, Mapping)
        and normal.get("accepted") is True
        and normal.get("finite_rate") == 1.0
        and normal.get("scoring_executed") is True
        and isinstance(stale, Mapping)
        and stale.get("rejected") is True
        and stale.get("rejection_phase") == "identity_guard_before_forward"
        and stale.get("scoring_executed") is False
        and stale.get("generation_executed") is False
    ):
        _fail("parent_core_gate_failed", f"refresh_v{version} identity gate failed")
    if previous is not None and payload.get("previous_tensor_sha256") != previous:
        _fail("parent_core_gate_failed", f"refresh_v{version} previous SHA failed")


def _verify_v0_guard(payload: Mapping[str, Any]) -> None:
    _require_status(payload, label="v0_guard")
    normal = payload.get("normal_v0")
    wrong = payload.get("wrong_authority")
    if not (
        isinstance(normal, Mapping)
        and normal.get("accepted") is True
        and normal.get("scoring_executed") is True
        and normal.get("generation_executed") is False
        and normal.get("silent_fallback") is False
        and normal.get("sampler_runtime_tensor_sha256")
        == normal.get("trainer_authoritative_tensor_sha256")
        and isinstance(wrong, Mapping)
        and wrong.get("accepted") is False
        and wrong.get("error_code") == "SAMPLER_RUNTIME_TENSOR_MISMATCH"
        and wrong.get("scoring_executed") is False
        and wrong.get("generation_executed") is False
        and wrong.get("silent_fallback") is False
    ):
        _fail("parent_core_gate_failed", "v0 guard failed")


def _verify_trajectory(payload: Mapping[str, Any], v1_sha: str) -> None:
    _require_status(payload, label="trajectory_step1_manifest")
    guard = payload.get("normal_generation_guard")
    stale = payload.get("stale_v0_pre_rollout")
    if not (
        payload.get("generated_by_policy_version") == "v1"
        and payload.get("p_old_policy_version") == "v1"
        and payload.get("logical_version") == "v1"
        and payload.get("sampler_tensor_sha256") == v1_sha
        and payload.get("trainer_authority_sha256") == v1_sha
        and payload.get("p_old_actor_tensor_sha256") == v1_sha
        and isinstance(guard, Mapping)
        and guard.get("accepted") is True
        and guard.get("generation_executed") is True
        and isinstance(stale, Mapping)
        and stale.get("rejected") is True
        and stale.get("generation_executed") is False
        and stale.get("scoring_executed") is False
    ):
        _fail("parent_core_gate_failed", "rollout-1 identity gate failed")


def _verify_base_null(payload: Mapping[str, Any]) -> None:
    _require_status(payload, label="base_null")
    if not (
        payload.get("teacher_is_base") is True
        and payload.get("old_actor_base_detached") is True
        and payload.get("current_actor_zero_lora") is True
        and payload.get("fresh_optimizer") is True
        and payload.get("independent_route") is True
        and _finite_number(payload.get("current_pre_base_max_gap"))
        and payload["current_pre_base_max_gap"] <= 0.0001
        and _finite_number(payload.get("advantage_max_abs"))
        and abs(float(payload["advantage_max_abs"])) <= 0.00000001
        and payload.get("objective") == 0
        and payload.get("loss") == 0
        and payload.get("gradient_norm") == 0
        and payload.get("parameter_delta") == 0
        and payload.get("nonzero_update_tensor_count") == 0
        and payload.get("adapter_sha256_before") == payload.get("adapter_sha256_after")
        and payload.get("teacher_gradient_tensor_count") == 0
        and payload.get("base_gradient_tensor_count") == 0
        and payload.get("finite_rate") == 1.0
    ):
        _fail("parent_core_gate_failed", "Base=Teacher null gate failed")


def _verify_index(root: Path, relative: str, *, run_id: str) -> dict[str, Any]:
    path = _regular_file(
        root,
        relative,
        missing_code="parent_artifact_missing",
        symlink_code="parent_artifact_symlink",
    )
    value = _read_json(path, code="parent_index_invalid")
    entries = value.get("artifacts")
    if not (
        value.get("run_id") == run_id
        and isinstance(entries, list)
        and value.get("artifact_count") == len(entries)
        and isinstance(value.get("required_phases"), list)
        and all(phase in value["required_phases"] for phase in CORE_PHASES)
    ):
        _fail("parent_index_invalid", f"index header invalid: {relative}")
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            _fail("parent_index_invalid", f"index entry invalid: {relative}")
        indexed = _safe_relative(raw.get("path"), label="indexed artifact")
        if indexed in seen:
            _fail("parent_index_invalid", f"index duplicate: {indexed}")
        seen.add(indexed)
        _verify_file(
            root,
            indexed,
            raw.get("sha256"),
            expected_size=raw.get("size_bytes"),
            missing_code="indexed_artifact_missing",
            symlink_code="indexed_artifact_symlink",
            mismatch_code="indexed_artifact_sha_mismatch",
        )
    return value


def _verify_safetensors(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    by_key = {str(item["canonical_key"]): item for item in records}
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                _fail("v2_adapter_inventory_mismatch", "safetensors header is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > _MAX_SAFETENSORS_HEADER_BYTES:
                _fail("v2_adapter_inventory_mismatch", "safetensors header length is invalid")
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                _fail("v2_adapter_inventory_mismatch", "safetensors header is truncated")
            try:
                header = json.loads(header_bytes)
            except (UnicodeError, json.JSONDecodeError):
                _fail("v2_adapter_inventory_mismatch", "safetensors header JSON is invalid")
            if not isinstance(header, dict):
                _fail("v2_adapter_inventory_mismatch", "safetensors header is not an object")
            tensors = {key: value for key, value in header.items() if key != "__metadata__"}
            if set(tensors) != set(by_key):
                _fail("v2_adapter_inventory_mismatch", "safetensors tensor keys differ")
            spans: list[tuple[int, int, str]] = []
            for key, raw in tensors.items():
                if not isinstance(raw, Mapping):
                    _fail("v2_adapter_inventory_mismatch", "safetensors tensor metadata is invalid")
                offsets = raw.get("data_offsets")
                record = by_key[key]
                if not (
                    raw.get("dtype") == "F32"
                    and raw.get("shape") == record["shape"]
                    and isinstance(offsets, list)
                    and len(offsets) == 2
                    and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in offsets)
                    and offsets[1] >= offsets[0]
                    and offsets[1] - offsets[0] == record["byte_length"]
                ):
                    _fail("v2_adapter_inventory_mismatch", f"safetensors metadata differs: {key}")
                spans.append((offsets[0], offsets[1], key))
            spans.sort()
            expected_start = 0
            for start, end, _ in spans:
                if start != expected_start:
                    _fail("v2_adapter_inventory_mismatch", "safetensors data offsets are not contiguous")
                expected_start = end
            data_start = 8 + header_length
            if path.stat().st_size - data_start != expected_start:
                _fail("v2_adapter_inventory_mismatch", "safetensors data span differs")
            checked = 0
            for start, end, key in spans:
                handle.seek(data_start + start)
                remaining = end - start
                digest = hashlib.sha256()
                while remaining:
                    chunk = handle.read(min(_STREAM_CHUNK_BYTES, remaining))
                    if not chunk:
                        _fail("v2_adapter_inventory_mismatch", "safetensors tensor data is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
                    checked += len(chunk)
                if digest.hexdigest() != by_key[key]["sha256"]:
                    _fail("v2_adapter_inventory_mismatch", f"tensor SHA differs: {key}")
    except OSError as error:
        _fail("v2_adapter_inventory_mismatch", f"safetensors read failed: {type(error).__name__}")
    return {"tensor_count": len(by_key), "tensor_bytes": checked}


def _verify_v2_adapter(
    root: Path,
    expected_v2: Mapping[str, Any],
    authority: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint_relative = _safe_relative(
        expected_v2.get("checkpoint_directory"), label="v2 checkpoint"
    )
    if checkpoint_relative != "checkpoints/v2":
        _fail("parent_spec_invalid", "v2 checkpoint must be checkpoints/v2")
    checkpoint = _assert_no_symlink(root, checkpoint_relative, code="v2_adapter_symlink")
    if not checkpoint.is_dir():
        _fail("v2_adapter_missing", "v2 checkpoint directory is absent")
    weights = _verify_file(
        root,
        f"{checkpoint_relative}/adapter_model.safetensors",
        expected_v2.get("adapter_weights_sha256"),
        expected_size=expected_v2.get("adapter_weights_size_bytes"),
        missing_code="v2_adapter_missing",
        symlink_code="v2_adapter_symlink",
        mismatch_code="v2_adapter_sha_mismatch",
    )
    config = _verify_file(
        root,
        f"{checkpoint_relative}/adapter_config.json",
        expected_v2.get("adapter_config_sha256"),
        expected_size=expected_v2.get("adapter_config_size_bytes"),
        missing_code="v2_adapter_missing",
        symlink_code="v2_adapter_symlink",
        mismatch_code="v2_adapter_sha_mismatch",
    )
    manifest = _verify_file(
        root,
        f"{checkpoint_relative}/adapter_transport_manifest.json",
        expected_v2.get("transport_manifest_sha256"),
        expected_size=expected_v2.get("transport_manifest_size_bytes"),
        missing_code="v2_adapter_missing",
        symlink_code="v2_adapter_symlink",
        mismatch_code="v2_adapter_sha_mismatch",
    )
    descriptor = authority.get("checkpoint")
    if not (
        isinstance(descriptor, Mapping)
        and descriptor.get("directory") == checkpoint_relative
        and descriptor.get("transport_manifest_path")
        == f"{checkpoint_relative}/adapter_transport_manifest.json"
        and descriptor.get("transport_manifest_sha256") == manifest["sha256"]
        and authority.get("aggregate_tensor_sha256")
        == expected_v2.get("aggregate_tensor_sha256")
    ):
        _fail("v2_adapter_binding_mismatch", "authority checkpoint descriptor differs")
    manifest_value = _read_json(
        root / manifest["path"], code="v2_adapter_manifest_invalid"
    )
    files = manifest_value.get("files")
    if not (
        manifest_value.get("schema_version") == 1
        and manifest_value.get("logical_version") == "v2"
        and manifest_value.get("canonical_config_sha256")
        == authority.get("canonical_config_sha256")
        and manifest_value.get("aggregate_tensor_sha256")
        == authority.get("aggregate_tensor_sha256")
        and isinstance(files, list)
        and len(files) == 2
    ):
        _fail("v2_adapter_manifest_invalid", "v2 transport manifest identity differs")
    roles: dict[str, Mapping[str, Any]] = {}
    indexed: set[str] = set()
    for raw in files:
        if not isinstance(raw, Mapping):
            _fail("v2_adapter_manifest_invalid", "v2 transport entry is invalid")
        role = raw.get("role")
        relative = _safe_relative(raw.get("path"), label="v2 transport file")
        if role not in {"adapter_config", "adapter_weights"} or role in roles or relative in indexed:
            _fail("v2_adapter_manifest_invalid", "v2 transport roles collide")
        roles[str(role)] = raw
        indexed.add(relative)
    if set(roles) != {"adapter_config", "adapter_weights"}:
        _fail("v2_adapter_manifest_invalid", "v2 transport roles are incomplete")
    for item, verified in ((roles["adapter_config"], config), (roles["adapter_weights"], weights)):
        if item.get("sha256") != verified["sha256"] or item.get("size_bytes") != verified["size_bytes"]:
            _fail("v2_adapter_manifest_invalid", "v2 transport file binding differs")
    entries = list(checkpoint.iterdir())
    if any(path.is_symlink() for path in entries):
        _fail("v2_adapter_symlink", "v2 checkpoint contains a symlink")
    if any(not path.is_file() for path in entries):
        _fail("v2_adapter_binding_mismatch", "v2 checkpoint contains a non-file entry")
    actual = {path.relative_to(checkpoint).as_posix() for path in entries}
    if actual != {"adapter_config.json", "adapter_model.safetensors", "adapter_transport_manifest.json"}:
        _fail("v2_adapter_binding_mismatch", "v2 checkpoint contains an unexpected transport set")
    inventory = _verify_safetensors(root / weights["path"], records)
    return {
        "canonical_path": str(checkpoint),
        "aggregate_tensor_sha256": authority["aggregate_tensor_sha256"],
        "authority_artifact_sha256": expected_v2["authority_artifact_sha256"],
        "transport_manifest_sha256": manifest["sha256"],
        "adapter_config_sha256": config["sha256"],
        "adapter_config_size_bytes": config["size_bytes"],
        "adapter_weights_sha256": weights["sha256"],
        "adapter_weights_size_bytes": weights["size_bytes"],
        "transport_manifest_size_bytes": manifest["size_bytes"],
        "transport_total_size_bytes": (
            config["size_bytes"] + weights["size_bytes"] + manifest["size_bytes"]
        ),
        **inventory,
    }


def _verify_protected(expected: Any) -> list[dict[str, Any]]:
    if not isinstance(expected, list) or not expected:
        _fail("parent_spec_invalid", "protected artifact list is empty")
    verified: list[dict[str, Any]] = []
    for raw in expected:
        if not isinstance(raw, Mapping):
            _fail("parent_spec_invalid", "protected artifact entry is invalid")
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            _fail("parent_spec_invalid", "protected artifact path is empty")
        path = Path(path_value)
        if path.is_symlink():
            _fail("protected_artifact_symlink", f"protected symlink rejected: {path.name}")
        if not path.is_file():
            _fail("protected_artifact_missing", f"protected file absent: {path.name}")
        size = path.stat().st_size
        if size != raw.get("size_bytes"):
            _fail("protected_artifact_sha_mismatch", f"protected size differs: {path.name}")
        digest = sha256_file_stream(path)
        if digest != raw.get("sha256"):
            _fail("protected_artifact_sha_mismatch", f"protected SHA differs: {path.name}")
        stage = raw.get("stage")
        if not isinstance(stage, str) or not stage:
            _fail("parent_spec_invalid", "protected artifact stage is empty")
        verified.append(
            {
                "stage": stage,
                "path": str(path),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    verified.sort(key=lambda item: (item["stage"], item["path"]))
    return verified


def verify_parent_reuse(
    output_dir: Path | str, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute reusable P4.6 core evidence from independent frozen SHA bindings.

    The P4.6 final readiness is intentionally not used as an authority: it is
    expected to be false because length qualification failed.  Conversely, a
    caller-provided or on-disk ``ready=true`` cannot override any failed gate.
    """

    root = Path(output_dir)
    if root.is_symlink():
        _fail("parent_output_symlink", "parent output directory is a symlink")
    if not root.is_dir():
        _fail("parent_output_missing", "parent output directory is absent")
    if not isinstance(expected, Mapping):
        _fail("parent_spec_invalid", "expected parent binding is not an object")
    run_id = expected.get("run_id")
    formal_commit = expected.get("formal_git_commit")
    if not isinstance(run_id, str) or not run_id or not _is_hex(formal_commit, 40):
        _fail("parent_spec_invalid", "run ID or formal Git commit is invalid")

    core_expected = expected.get("core_artifact_sha256")
    if not isinstance(core_expected, Mapping) or set(core_expected) != set(CORE_PHASES):
        _fail("parent_spec_invalid", "core artifact SHA map is incomplete")
    expected_v2 = expected.get("v2")
    if not isinstance(expected_v2, Mapping):
        _fail("parent_spec_invalid", "v2 binding is absent")
    if (
        not _is_hex(expected_v2.get("authority_artifact_sha256"), 64)
        or expected_v2["authority_artifact_sha256"]
        != core_expected["authority_v2"]
    ):
        _fail(
            "v2_adapter_binding_mismatch",
            "v2 authority artifact SHA differs from the core binding",
        )
    checkpoint_relative = _safe_relative(
        expected_v2.get("checkpoint_directory"), label="v2 checkpoint"
    )
    _verify_file(
        root,
        f"{checkpoint_relative}/adapter_model.safetensors",
        expected_v2.get("adapter_weights_sha256"),
        expected_size=expected_v2.get("adapter_weights_size_bytes"),
        missing_code="v2_adapter_missing",
        symlink_code="v2_adapter_symlink",
        mismatch_code="v2_adapter_sha_mismatch",
    )

    for phase in CORE_PHASES:
        _verify_file(
            root,
            f"{phase}.json",
            core_expected[phase],
            missing_code="parent_artifact_missing",
            symlink_code="parent_artifact_symlink",
            mismatch_code="parent_sha_mismatch",
        )
    scalar_files = {
        "evidence_artifact_index.json": expected.get("evidence_index_sha256"),
        "artifact_index.json": expected.get("final_index_sha256"),
        "failure.json": expected.get("failure_sha256"),
        "metrics.jsonl": expected.get("metrics_sha256"),
        "readiness.json": expected.get("readiness_sha256"),
        "evidence_readiness.json": expected.get("evidence_readiness_sha256"),
        "micro_readiness.json": expected.get("micro_readiness_sha256"),
    }
    for relative, digest in scalar_files.items():
        _verify_file(
            root,
            relative,
            digest,
            missing_code="parent_artifact_missing",
            symlink_code="parent_artifact_symlink",
            mismatch_code="parent_sha_mismatch",
        )

    documents: dict[str, dict[str, Any]] = {}
    previous_path: str | None = None
    previous_sha: str | None = None
    common_bindings: dict[str, Any] | None = None
    for ordinal, phase in enumerate(CORE_PHASES):
        path = root / f"{phase}.json"
        value = _read_json(path, code="parent_core_artifact_invalid")
        documents[phase] = value
        bindings = {field: value.get(field) for field in _BINDING_FIELDS}
        if common_bindings is None:
            common_bindings = bindings
        elif bindings != common_bindings:
            _fail("parent_core_binding_mismatch", f"binding drift at {phase}")
        if not (
            value.get("phase_id") == phase
            and value.get("ordinal") == ordinal
            and value.get("previous_phase_path") == previous_path
            and value.get("previous_phase_sha256") == previous_sha
            and value.get("payload_sha256") == canonical_json_sha256(value.get("payload"))
            and value.get("isolation")
            == {
                "final_access": False,
                "controller_access": False,
                "confirmation_access": False,
                "label_access": False,
            }
            and isinstance(value.get("payload"), Mapping)
        ):
            _fail("parent_core_binding_mismatch", f"phase chain drift at {phase}")
        previous_path = f"{phase}.json"
        previous_sha = str(core_expected[phase])
    assert common_bindings is not None
    if common_bindings.get("run_id") != run_id or common_bindings.get("git_commit") != formal_commit:
        _fail("parent_core_binding_mismatch", "run ID or formal Git commit drift")

    for phase in ("launch_record", "preflight", "probe_manifest"):
        _require_status(documents[phase]["payload"], label=phase)
    _verify_v0_guard(documents["v0_guard"]["payload"])
    _verify_reconstruction(documents["reconstruction_step0"]["payload"], label="reconstruction_step0")
    authority1 = documents["authority_v1"]["payload"]
    authority2 = documents["authority_v2"]["payload"]
    _verify_authority(authority1, version=1)
    _verify_refresh(documents["refresh_v1"]["payload"], authority1, version=1, previous=None)
    v1_sha = str(authority1["aggregate_tensor_sha256"])
    _verify_trajectory(documents["trajectory_step1_manifest"]["payload"], v1_sha)
    _verify_reconstruction(documents["reconstruction_step1"]["payload"], label="reconstruction_step1")
    records2 = _verify_authority(authority2, version=2)
    v2_sha = str(authority2["aggregate_tensor_sha256"])
    if v2_sha == v1_sha:
        _fail("parent_core_gate_failed", "v1 and v2 tensor identities are equal")
    _verify_refresh(documents["refresh_v2"]["payload"], authority2, version=2, previous=v1_sha)
    _verify_base_null(documents["base_null"]["payload"])

    metrics_phases: list[str] = []
    try:
        with (root / "metrics.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    _fail("parent_metrics_invalid", "metrics contains an empty record")
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("phase_id"), str):
                    _fail("parent_metrics_invalid", "metrics row is invalid")
                metrics_phases.append(row["phase_id"])
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("parent_metrics_invalid", f"metrics parse failed: {type(error).__name__}")
    if tuple(metrics_phases) != CORE_PHASES:
        _fail("parent_metrics_invalid", "metrics core phase sequence differs")

    failure = _read_json(root / "failure.json", code="parent_failure_invalid")
    if not (
        failure.get("run_id") == run_id
        and failure.get("status") == "fail"
        and isinstance(failure.get("reason"), str)
        and "length_not_frozen" in failure["reason"]
        and failure.get("last_committed_phase") == "base_null"
        and failure.get("last_committed_phase_sha256") == core_expected["base_null"]
        and failure.get("metrics_sha256") == expected["metrics_sha256"]
        and failure.get("B2_authorized") is False
        and failure.get("B2_started") is False
    ):
        _fail("parent_failure_invalid", "P4.6 failure binding differs")

    micro = _read_json(root / "micro_readiness.json", code="parent_micro_invalid")
    if not (
        micro.get("run_id") == run_id
        and micro.get("ready") is True
        and micro.get("status") == "pass"
        and micro.get("production_sampler_refresh_ready") is True
        and micro.get("B2_authorized") is False
        and micro.get("B2_started") is False
        and micro.get("v1_tensor_sha256") == v1_sha
        and micro.get("refresh_v1_artifact_sha256") == core_expected["refresh_v1"]
    ):
        _fail("parent_micro_invalid", "micro readiness is not artifact-supported")

    evidence_index = _verify_index(root, "evidence_artifact_index.json", run_id=run_id)
    final_index = _verify_index(root, "artifact_index.json", run_id=run_id)
    evidence_readiness = _read_json(
        root / "evidence_readiness.json", code="parent_readiness_invalid"
    )
    readiness = _read_json(root / "readiness.json", code="parent_readiness_invalid")
    if not (
        evidence_readiness.get("run_id") == run_id
        and evidence_readiness.get("artifact_index_sha256")
        == expected["evidence_index_sha256"]
        and evidence_readiness.get("micro_readiness_sha256")
        == expected["micro_readiness_sha256"]
        and readiness.get("run_id") == run_id
        and readiness.get("artifact_index_sha256") == expected["final_index_sha256"]
        and readiness.get("evidence_artifact_index_sha256")
        == expected["evidence_index_sha256"]
        and readiness.get("evidence_readiness_sha256")
        == expected["evidence_readiness_sha256"]
        and readiness.get("micro_readiness_sha256")
        == expected["micro_readiness_sha256"]
        and readiness.get("B2_started") is False
    ):
        _fail("parent_readiness_invalid", "readiness SHA graph differs")

    v2_adapter = _verify_v2_adapter(root, expected_v2, authority2, records2)
    protected = _verify_protected(expected.get("protected_artifacts"))
    source_branch = expected.get("source_branch")
    if not isinstance(source_branch, str) or not source_branch:
        _fail("parent_spec_invalid", "source branch is empty")
    return {
        "schema_version": 1,
        "audit_kind": "p4_6_parent_core_reuse_audit_v1",
        "source_branch": source_branch,
        "formal_git_commit": formal_commit,
        "run_id": run_id,
        "parent_output_path": str(root),
        "parent_core_evidence_verified": True,
        "v2_adapter_reusable": True,
        "observed_parent_final_ready": readiness.get("ready") is True,
        "observed_parent_failure": {
            "status": failure["status"],
            "reason": failure["reason"],
        },
        "index_bindings": {
            "evidence_index_sha256": expected["evidence_index_sha256"],
            "evidence_artifact_count": evidence_index["artifact_count"],
            "final_index_sha256": expected["final_index_sha256"],
            "final_artifact_count": final_index["artifact_count"],
            "failure_sha256": expected["failure_sha256"],
            "metrics_sha256": expected["metrics_sha256"],
            "readiness_sha256": expected["readiness_sha256"],
            "evidence_readiness_sha256": expected["evidence_readiness_sha256"],
            "micro_readiness_sha256": expected["micro_readiness_sha256"],
        },
        "core_artifact_sha256": {phase: core_expected[phase] for phase in CORE_PHASES},
        "policy_identity": {
            "v1_tensor_sha256": v1_sha,
            "v2_tensor_sha256": v2_sha,
            "v1_differs_from_v2": True,
        },
        "v2_adapter": v2_adapter,
        "protected_artifacts": protected,
        "access": {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        },
    }


def build_parent_reuse_attestation(
    audit: Mapping[str, Any], *, current_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a deterministic, self-digesting attestation from a verified audit."""

    if not isinstance(audit, Mapping) or audit.get("parent_core_evidence_verified") is not True:
        _fail("parent_audit_unverified", "attestation requires a verified audit")
    if audit.get("v2_adapter_reusable") is not True:
        _fail("parent_audit_unverified", "attestation requires a reusable v2 adapter")
    if not isinstance(current_bindings, Mapping) or not current_bindings:
        _fail("parent_spec_invalid", "current code/config bindings are absent")
    normalized: dict[str, str] = {}
    for key in sorted(current_bindings):
        value = current_bindings[key]
        expected_length = 40 if key.endswith("git_commit") else 64
        if not isinstance(key, str) or not key or not _is_hex(value, expected_length):
            _fail("parent_spec_invalid", f"current binding is invalid: {key}")
        normalized[key] = value
    stable_audit = json.loads(
        json.dumps(audit, ensure_ascii=True, allow_nan=False, sort_keys=True)
    )
    value = {
        "schema_version": 1,
        "artifact_kind": "p4_7_parent_reuse_attestation_v1",
        "status": "verified_parent_core_reuse",
        "parent_core_evidence_verified": True,
        "v2_adapter_reusable": True,
        "gpu_length_qualification_pending": True,
        "production_sampler_refresh_ready": False,
        "OPD_scoring_backend_ready": False,
        "B2_authorized": False,
        "B2_started": False,
        "current_bindings": normalized,
        "parent_audit": stable_audit,
    }
    value["attestation_sha256"] = canonical_json_sha256(value)
    return value


__all__ = [
    "CORE_PHASES",
    "ParentReuseVerificationError",
    "build_parent_reuse_attestation",
    "canonical_json_sha256",
    "sha256_file_stream",
    "verify_parent_reuse",
]
