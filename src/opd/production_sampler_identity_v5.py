"""CPU-safe canonical adapter identity and production request guard for P4.5."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import torch


IDENTITY_SCHEMA_VERSION = 5
AUTHORITY_SOURCE = "trainer_memory_reload_verified_immutable_artifact"
PRODUCTION_ADAPTER_SLOT = "student_active"
_ADAPTER_ROLES = {
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
    "modules_to_save",
}
_CONFIG_FIELDS = (
    "peft_type",
    "r",
    "rank_pattern",
    "lora_alpha",
    "alpha_pattern",
    "lora_dropout",
    "target_modules",
    "modules_to_save",
    "bias",
    "use_dora",
    "use_rslora",
    "task_type",
    "inference_mode",
)


class AdapterIdentityError(RuntimeError):
    """An adapter identity cannot be made canonical or authoritative."""


class SamplerIdentityGuardError(AdapterIdentityError):
    """A request or runtime failed closed before model execution."""

    def __init__(self, code: str, message: str, evidence: Mapping[str, Any]):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.evidence = dict(evidence)


def _valid_digest(value: Any, *, length: int = 64) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _canonical_json_value(value: Any) -> Any:
    value = _enum_value(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_json_value(item) for item in value)
    if isinstance(value, tuple):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, list):
        normalized = [_canonical_json_value(item) for item in value]
        return sorted(normalized) if all(isinstance(item, str) for item in normalized) else normalized
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise AdapterIdentityError("adapter config contains a non-finite value")
        return value
    raise AdapterIdentityError(f"adapter config contains unsupported value {type(value).__name__}")


def _config_mapping(config: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise AdapterIdentityError("adapter config must be a mapping or expose to_dict")
    value = to_dict()
    if not isinstance(value, Mapping):
        raise AdapterIdentityError("adapter config to_dict did not return a mapping")
    return value


def canonical_adapter_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Bind every LoRA structural/scaling field used by the production slot."""

    source = _config_mapping(config)
    missing = [field for field in _CONFIG_FIELDS if field not in source]
    if missing:
        raise AdapterIdentityError(f"adapter config is missing identity fields: {missing}")
    payload = {field: _canonical_json_value(source[field]) for field in _CONFIG_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def canonical_adapter_tensor_key(name: str, runtime_adapter_name: str) -> str:
    """Strip one exact PEFT runtime-label segment while preserving module semantics."""

    if not isinstance(name, str) or not name or not isinstance(runtime_adapter_name, str):
        raise AdapterIdentityError("adapter tensor key/runtime name is invalid")
    parts = name.split(".")
    role_indices = [index for index, part in enumerate(parts) if part in _ADAPTER_ROLES]
    if len(role_indices) != 1:
        raise AdapterIdentityError(f"adapter tensor key has ambiguous adapter role: {name}")
    role_index = role_indices[0]
    if role_index + 1 < len(parts) and parts[role_index + 1] == runtime_adapter_name:
        del parts[role_index + 1]
    return ".".join(parts)


def _adapter_tensor_items(
    tensors: Mapping[str, torch.Tensor], runtime_adapter_name: str
) -> list[dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    for source_key, tensor in tensors.items():
        if not any(f".{role}." in f".{source_key}." for role in _ADAPTER_ROLES):
            continue
        if not isinstance(tensor, torch.Tensor):
            raise AdapterIdentityError(f"adapter tensor {source_key} is not a torch.Tensor")
        canonical_key = canonical_adapter_tensor_key(str(source_key), runtime_adapter_name)
        if canonical_key in canonical:
            raise AdapterIdentityError(f"canonical adapter tensor key collision: {canonical_key}")
        normalized = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        shape = [int(item) for item in normalized.shape]
        if not shape or any(item <= 0 for item in shape):
            raise AdapterIdentityError(f"adapter tensor {source_key} has invalid shape")
        data = normalized.numpy().astype("<f4", copy=False).tobytes(order="C")
        canonical[canonical_key] = {
            "canonical_key": canonical_key,
            "sha256": hashlib.sha256(data).hexdigest(),
            "shape": shape,
            "source_dtype": str(tensor.dtype),
            "canonical_dtype": "float32_le",
            "canonical_byte_length": len(data),
        }
    if not canonical:
        raise AdapterIdentityError("canonical adapter identity contains no adapter tensors")
    return [canonical[key] for key in sorted(canonical)]


def rebuild_aggregate_tensor_sha(
    tensor_records: Sequence[Mapping[str, Any]],
) -> str:
    """Rebuild the tensor-only aggregate from canonical per-tensor records."""

    canonical: dict[str, dict[str, Any]] = {}
    for raw in tensor_records:
        record = {
            "canonical_key": raw.get("canonical_key"),
            "sha256": raw.get("sha256"),
            "shape": list(raw.get("shape", [])),
            "canonical_dtype": raw.get("canonical_dtype"),
            "canonical_byte_length": raw.get("canonical_byte_length"),
        }
        key = record["canonical_key"]
        shape = record["shape"]
        if not isinstance(key, str) or not key or key in canonical:
            raise AdapterIdentityError("aggregate tensor records contain a key collision")
        if not _valid_digest(record["sha256"]):
            raise AdapterIdentityError(f"per-tensor SHA is invalid for {key}")
        if not shape or any(not isinstance(item, int) or item <= 0 for item in shape):
            raise AdapterIdentityError(f"per-tensor shape is invalid for {key}")
        expected_bytes = 4 * math.prod(shape)
        if record["canonical_dtype"] != "float32_le" or record["canonical_byte_length"] != expected_bytes:
            raise AdapterIdentityError(f"per-tensor canonical byte metadata is invalid for {key}")
        canonical[key] = record
    if not canonical:
        raise AdapterIdentityError("aggregate tensor records are empty")
    digest = hashlib.sha256()
    for key in sorted(canonical):
        encoded = json.dumps(
            canonical[key], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_adapter_identity_manifest(
    tensors: Mapping[str, torch.Tensor],
    *,
    adapter_config: Mapping[str, Any] | Any,
    adapter_logical_version: int,
    adapter_runtime_name: str,
    active_adapter: str,
    base_revision: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    if not isinstance(adapter_logical_version, int) or isinstance(adapter_logical_version, bool) or adapter_logical_version < 0:
        raise AdapterIdentityError("adapter logical version must be a non-negative integer")
    if not isinstance(adapter_runtime_name, str) or not adapter_runtime_name:
        raise AdapterIdentityError("adapter runtime name is empty")
    if not isinstance(active_adapter, str) or not active_adapter:
        raise AdapterIdentityError("active adapter is empty")
    if not (_valid_digest(base_revision, length=40) and _valid_digest(tokenizer_revision, length=40)):
        raise AdapterIdentityError("base/tokenizer revision must be immutable 40-hex values")
    config_identity = canonical_adapter_config(adapter_config)
    tensor_records = _adapter_tensor_items(tensors, adapter_runtime_name)
    aggregate = rebuild_aggregate_tensor_sha(tensor_records)
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_kind": "canonical_adapter_runtime_tensors",
        "adapter_logical_version": adapter_logical_version,
        "adapter_runtime_name": adapter_runtime_name,
        "active_adapter": active_adapter,
        "canonical_config_sha256": config_identity["sha256"],
        "canonical_config": config_identity["payload"],
        "tensors": tensor_records,
        "aggregate_tensor_sha256": aggregate,
        "tensor_count": len(tensor_records),
        "total_canonical_bytes": sum(item["canonical_byte_length"] for item in tensor_records),
        "base_revision": base_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def trainer_authority_from_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_manifest_sha256: str,
    trainer_memory_reload_gate_passed: bool,
    run_token: str,
    production_runtime_name: str = PRODUCTION_ADAPTER_SLOT,
) -> dict[str, Any]:
    if trainer_memory_reload_gate_passed is not True:
        raise AdapterIdentityError("trainer memory/reload gate did not pass")
    if not _valid_digest(artifact_manifest_sha256):
        raise AdapterIdentityError("immutable artifact manifest SHA is invalid")
    if not isinstance(run_token, str) or not run_token:
        raise AdapterIdentityError("authoritative run token is empty")
    if not isinstance(production_runtime_name, str) or not production_runtime_name:
        raise AdapterIdentityError("production runtime slot is empty")
    result = deepcopy(dict(manifest))
    if not _valid_digest(result.get("canonical_config_sha256")):
        raise AdapterIdentityError("trainer artifact config SHA is invalid")
    rebuilt = rebuild_aggregate_tensor_sha(result.get("tensors", []))
    if rebuilt != result.get("aggregate_tensor_sha256"):
        raise AdapterIdentityError("trainer artifact aggregate cannot be rebuilt")
    result.update(
        {
            "authority_source": AUTHORITY_SOURCE,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "run_token": run_token,
            "production_runtime_name": production_runtime_name,
        }
    )
    return result


def _execution_evidence(operation: str) -> dict[str, Any]:
    return {
        "operation": operation,
        "accepted": False,
        "guard_stage": "identity_guard_before_forward",
        "scoring_executed": False,
        "generation_executed": False,
    }


def _fail(code: str, message: str, evidence: Mapping[str, Any]) -> None:
    raise SamplerIdentityGuardError(code, message, evidence)


def _comparable_tensors(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "canonical_key",
        "sha256",
        "shape",
        "canonical_dtype",
        "canonical_byte_length",
    )
    return [{field: item.get(field) for field in fields} for item in value.get("tensors", [])]


def guard_sampler_operation(
    *,
    authority: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    operation: str,
    callback: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    """Verify trainer authority against runtime and request before any forward."""

    allowed = {"fixed_action", "generation", "direct_no_cache", "direct_cache"}
    if operation not in allowed:
        raise AdapterIdentityError(f"unsupported sampler operation: {operation}")
    evidence = _execution_evidence(operation)
    if authority.get("authority_source") != AUTHORITY_SOURCE or not _valid_digest(
        authority.get("artifact_manifest_sha256")
    ):
        _fail("INVALID_TRAINER_AUTHORITY", "authority is not an immutable trainer artifact", evidence)

    request_pairs = (
        ("run_token", "run_token"),
        ("logical_version", "adapter_logical_version"),
        ("canonical_config_sha256", "canonical_config_sha256"),
        ("base_revision", "base_revision"),
        ("tokenizer_revision", "tokenizer_revision"),
    )
    if any(request_identity.get(left) != authority.get(right) for left, right in request_pairs):
        _fail("STALE_SAMPLER_IDENTITY", "request does not match trainer authority", evidence)
    if request_identity.get("authoritative_tensor_sha256") != authority.get(
        "aggregate_tensor_sha256"
    ):
        _fail(
            "SAMPLER_RUNTIME_TENSOR_MISMATCH",
            "request expected tensor SHA does not match trainer authority",
            evidence,
        )

    if runtime_identity.get("active_adapter") != authority.get("production_runtime_name"):
        _fail("SAMPLER_ACTIVE_ADAPTER_MISMATCH", "active adapter is not the production slot", evidence)
    if runtime_identity.get("adapters_enabled") is not True:
        _fail("SAMPLER_ADAPTER_DISABLED", "adapter layers are disabled", evidence)
    if runtime_identity.get("merged") is not False:
        _fail("SAMPLER_ADAPTER_MERGED", "adapter is merged into the base", evidence)
    for field, code in (
        ("base_revision", "SAMPLER_BASE_REVISION_MISMATCH"),
        ("tokenizer_revision", "SAMPLER_TOKENIZER_REVISION_MISMATCH"),
        ("canonical_config_sha256", "SAMPLER_CONFIG_MISMATCH"),
    ):
        if runtime_identity.get(field) != authority.get(field):
            _fail(code, f"runtime {field} does not match trainer authority", evidence)
    if runtime_identity.get("adapter_logical_version") != authority.get("adapter_logical_version"):
        _fail("SAMPLER_LOGICAL_VERSION_MISMATCH", "runtime logical version is stale", evidence)
    if (
        runtime_identity.get("aggregate_tensor_sha256")
        != authority.get("aggregate_tensor_sha256")
        or _comparable_tensors(runtime_identity) != _comparable_tensors(authority)
    ):
        _fail(
            "SAMPLER_RUNTIME_TENSOR_MISMATCH",
            "runtime adapter tensors do not match trainer authority",
            evidence,
        )

    result = callback()
    evidence["accepted"] = True
    if operation == "generation":
        evidence["generation_executed"] = True
    else:
        evidence["scoring_executed"] = True
    return result, evidence
