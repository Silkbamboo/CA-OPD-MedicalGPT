"""Cross-method common-protocol identity for B2, IDT, and CA-OPD."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


class MainProtocolV2Error(RuntimeError):
    """A method package diverges from the compute-matched protocol."""


METHOD_IDS = ("B2", "IDT", "CA-OPD")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def common_protocol_sha256_v2(package: Mapping[str, Any]) -> str:
    common = package.get("common")
    if not isinstance(common, Mapping) or not common:
        raise MainProtocolV2Error("method package common protocol is absent")
    return _canonical_sha256(dict(common))


def validate_method_packages_v2(
    packages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(packages) != 3:
        raise MainProtocolV2Error("exactly three method packages are required")
    by_method: dict[str, str] = {}
    for package in packages:
        method = package.get("method_id")
        if method not in METHOD_IDS or method in by_method:
            raise MainProtocolV2Error("method package identity differs")
        by_method[str(method)] = common_protocol_sha256_v2(package)
    if set(by_method) != set(METHOD_IDS) or len(set(by_method.values())) != 1:
        raise MainProtocolV2Error("B2/IDT/CA common protocol hash differs")
    return {
        "schema_version": 2,
        "artifact_kind": "main_experiment_common_protocol_attestation_v2",
        "passed": True,
        "common_protocol_sha256": next(iter(by_method.values())),
        "common_protocol_sha256_by_method": by_method,
        "allowed_method_differences_only": True,
    }


__all__ = [
    "MainProtocolV2Error",
    "common_protocol_sha256_v2",
    "validate_method_packages_v2",
]
