"""Production-bound PEFT 0.17.1 stable-slot refresh state machine for P4.5."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_sampler_identity_v5 import (
    AUTHORITY_SOURCE,
    PRODUCTION_ADAPTER_SLOT,
    build_adapter_identity_manifest,
)


PRODUCTION_REFRESH_MECHANISM = "peft_0_17_1_hotswap_stable_slot"


class ProductionSamplerRefreshError(RuntimeError):
    """The production refresh failed closed without publishing target identity."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def validate_refresh_mechanism(mechanism: str) -> None:
    if "delete_first" in str(mechanism):
        raise ProductionSamplerRefreshError(
            "DELETE_FIRST_REFRESH_FORBIDDEN",
            "delete-first refresh is forbidden because it destroys the last known-good route",
        )
    if mechanism != PRODUCTION_REFRESH_MECHANISM:
        raise ProductionSamplerRefreshError(
            "NON_PRODUCTION_REFRESH_MECHANISM",
            f"{mechanism!r} is not production; only stable-slot hotswap is frozen",
        )


def _active_adapter(model: Any) -> str | None:
    value = getattr(model, "active_adapter", None)
    if isinstance(value, (list, tuple)):
        return str(value[0]) if len(value) == 1 else None
    return str(value) if isinstance(value, str) else None


def _layer_state(model: Any) -> tuple[bool, bool, list[dict[str, Any]]]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    enabled = True
    merged = False
    layers: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        disabled = bool(
            getattr(module, "disable_adapters", False)
            or getattr(module, "_disable_adapters", False)
        )
        module_merged = bool(
            getattr(module, "merged", False)
            or getattr(module, "merged_adapters", [])
        )
        available = (
            list(module._all_available_adapter_names())
            if hasattr(module, "_all_available_adapter_names")
            else []
        )
        active = [str(item) for item in getattr(module, "active_adapters", [])]
        layers.append(
            {
                "module": str(name),
                "available_adapters": sorted(str(item) for item in available),
                "active_adapters": active,
                "enabled": not disabled,
                "merged": module_merged,
            }
        )
        enabled = enabled and not disabled
        merged = merged or module_merged
    if not layers:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_REGISTRY_INVALID", "no PEFT tuner layers were found"
        )
    return enabled, merged, layers


def registry_snapshot(model: Any) -> dict[str, Any]:
    enabled, merged, layers = _layer_state(model)
    peft_config = getattr(model, "peft_config", None)
    names = sorted(str(name) for name in peft_config) if isinstance(peft_config, Mapping) else []
    return {
        "active_adapter": _active_adapter(model),
        "peft_config_names": names,
        "adapter_count": len(names),
        "adapters_enabled": enabled,
        "merged": merged,
        "layers": layers,
    }


def runtime_identity_from_peft(
    model: Any,
    *,
    logical_version: int,
    runtime_name: str,
    base_revision: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    config = getattr(model, "peft_config", None)
    if not isinstance(config, Mapping) or runtime_name not in config:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_REGISTRY_INVALID", "production adapter slot is absent"
        )
    snapshot = registry_snapshot(model)
    manifest = build_adapter_identity_manifest(
        dict(model.named_parameters()),
        adapter_config=config[runtime_name],
        adapter_logical_version=logical_version,
        adapter_runtime_name=runtime_name,
        active_adapter=snapshot["active_adapter"] or "",
        base_revision=base_revision,
        tokenizer_revision=tokenizer_revision,
    )
    manifest.update(
        {
            "adapters_enabled": snapshot["adapters_enabled"],
            "merged": snapshot["merged"],
            "registry_snapshot": snapshot,
        }
    )
    return manifest


def adapter_artifact_identity(
    adapter_path: str | Path,
    *,
    logical_version: int,
    runtime_name: str,
    base_revision: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    """Build target identity directly from the immutable saved adapter bytes."""

    from peft import PeftConfig
    from peft.utils.save_and_load import load_peft_weights

    path = Path(adapter_path)
    if not (path / "adapter_config.json").is_file():
        raise ProductionSamplerRefreshError(
            "TARGET_ADAPTER_ARTIFACT_INVALID", "adapter_config.json is absent"
        )
    try:
        config = PeftConfig.from_pretrained(path)
        tensors = load_peft_weights(path, device="cpu")
        return build_adapter_identity_manifest(
            tensors,
            adapter_config=config,
            adapter_logical_version=logical_version,
            adapter_runtime_name=runtime_name,
            active_adapter=runtime_name,
            base_revision=base_revision,
            tokenizer_revision=tokenizer_revision,
        )
    except ProductionSamplerRefreshError:
        raise
    except Exception as error:
        raise ProductionSamplerRefreshError(
            "TARGET_ADAPTER_ARTIFACT_INVALID", f"cannot inspect target adapter: {error}"
        ) from error


def _comparable_tensors(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "canonical_key",
        "sha256",
        "shape",
        "canonical_dtype",
        "canonical_byte_length",
    )
    return [{field: item.get(field) for field in fields} for item in value.get("tensors", [])]


def _manifest_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("adapter_logical_version") == right.get("adapter_logical_version")
        and left.get("canonical_config_sha256") == right.get("canonical_config_sha256")
        and left.get("aggregate_tensor_sha256") == right.get("aggregate_tensor_sha256")
        and left.get("base_revision") == right.get("base_revision")
        and left.get("tokenizer_revision") == right.get("tokenizer_revision")
        and _comparable_tensors(left) == _comparable_tensors(right)
    )


def _require_authority(authority: Mapping[str, Any], label: str) -> None:
    if authority.get("authority_source") != AUTHORITY_SOURCE:
        raise ProductionSamplerRefreshError(
            "INVALID_TRAINER_AUTHORITY", f"{label} does not come from trainer memory/reload evidence"
        )


def _require_production_runtime(identity: Mapping[str, Any]) -> None:
    if identity.get("active_adapter") != PRODUCTION_ADAPTER_SLOT:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ACTIVE_ADAPTER_MISMATCH", "stable production slot is not active"
        )
    if identity.get("adapters_enabled") is not True:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_DISABLED", "adapter layers are disabled"
        )
    if identity.get("merged") is not False:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_MERGED", "adapter layers are merged"
        )
    snapshot = identity.get("registry_snapshot", {})
    if snapshot.get("peft_config_names") != [PRODUCTION_ADAPTER_SLOT]:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_REGISTRY_INVALID", "production requires exactly one stable adapter slot"
        )


def refresh_stable_slot(
    model: Any,
    *,
    adapter_path: str | Path,
    current_authority: Mapping[str, Any],
    target_authority: Mapping[str, Any],
    base_revision: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    """Verify target bytes, hotswap one slot, then publish only exact authority."""

    validate_refresh_mechanism(PRODUCTION_REFRESH_MECHANISM)
    _require_authority(current_authority, "current authority")
    _require_authority(target_authority, "target authority")
    current_version = current_authority.get("adapter_logical_version")
    target_version = target_authority.get("adapter_logical_version")
    if not isinstance(current_version, int) or not isinstance(target_version, int):
        raise ProductionSamplerRefreshError(
            "INVALID_REFRESH_VERSION", "logical versions must be integers"
        )
    noop = target_version == current_version
    if target_version not in {current_version, current_version + 1}:
        raise ProductionSamplerRefreshError(
            "INVALID_REFRESH_VERSION", "target must be the current or next logical version"
        )
    if noop and target_authority.get("aggregate_tensor_sha256") != current_authority.get(
        "aggregate_tensor_sha256"
    ):
        raise ProductionSamplerRefreshError(
            "INVALID_REFRESH_VERSION", "same-version refresh must be a tensor no-op"
        )
    if not noop and target_authority.get("run_token") == current_authority.get("run_token"):
        raise ProductionSamplerRefreshError(
            "INVALID_REFRESH_VERSION", "advanced logical version requires a new run token"
        )

    registry_before = registry_snapshot(model)
    current_runtime = runtime_identity_from_peft(
        model,
        logical_version=current_version,
        runtime_name=PRODUCTION_ADAPTER_SLOT,
        base_revision=base_revision,
        tokenizer_revision=tokenizer_revision,
    )
    _require_production_runtime(current_runtime)
    if not _manifest_matches(current_runtime, current_authority):
        raise ProductionSamplerRefreshError(
            "CURRENT_RUNTIME_AUTHORITY_MISMATCH",
            "current sampler does not match the last trainer authority",
        )

    candidate = adapter_artifact_identity(
        adapter_path,
        logical_version=target_version,
        runtime_name=PRODUCTION_ADAPTER_SLOT,
        base_revision=base_revision,
        tokenizer_revision=tokenizer_revision,
    )
    modules_to_save = candidate.get("canonical_config", {}).get("modules_to_save")
    if modules_to_save:
        raise ProductionSamplerRefreshError(
            "HOTSWAP_MODULES_TO_SAVE_UNSUPPORTED",
            "stable-slot hotswap is frozen only for LoRA tensors without modules_to_save",
        )
    if not _manifest_matches(candidate, target_authority):
        raise ProductionSamplerRefreshError(
            "TARGET_ARTIFACT_AUTHORITY_MISMATCH",
            "saved target adapter does not match trainer authority",
        )

    from peft.utils.hotswap import hotswap_adapter

    adapter_devices = {
        str(parameter.device)
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    if len(adapter_devices) != 1:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_DEVICE_AMBIGUOUS", "adapter tensors span multiple devices"
        )
    started = time.perf_counter()
    try:
        hotswap_adapter(
            model,
            str(adapter_path),
            adapter_name=PRODUCTION_ADAPTER_SLOT,
            torch_device=next(iter(adapter_devices)),
        )
    except Exception as error:
        raise ProductionSamplerRefreshError(
            "PEFT_HOTSWAP_FAILED", f"PEFT stable-slot hotswap failed: {error}"
        ) from error
    latency = time.perf_counter() - started

    runtime = runtime_identity_from_peft(
        model,
        logical_version=target_version,
        runtime_name=PRODUCTION_ADAPTER_SLOT,
        base_revision=base_revision,
        tokenizer_revision=tokenizer_revision,
    )
    _require_production_runtime(runtime)
    if not _manifest_matches(runtime, target_authority):
        raise ProductionSamplerRefreshError(
            "SAMPLER_RUNTIME_TENSOR_MISMATCH",
            "hotswapped runtime tensors do not match trainer authority",
        )
    registry_after = registry_snapshot(model)
    if registry_after["adapter_count"] != registry_before["adapter_count"]:
        raise ProductionSamplerRefreshError(
            "SAMPLER_ADAPTER_REGISTRY_LEAK", "stable-slot refresh changed registry size"
        )
    return {
        "schema_version": 5,
        "refresh_mechanism": PRODUCTION_REFRESH_MECHANISM,
        "noop": noop,
        "refresh_latency_seconds": latency,
        "registry_before": registry_before,
        "registry_after": registry_after,
        "runtime_identity": runtime,
        "published_authority": deepcopy(dict(target_authority)),
    }
