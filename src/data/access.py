"""Split access with an auditable final-test guard.

Reading ``final_test`` requires three things at once: the explicit split name,
``allow_final_test=True`` and a human-readable ``reason``. Every such access is
appended to ``final_test_access.log`` next to the data, so "final test was only
read once, after the checkpoint was frozen" becomes a checkable claim instead of
a promise.
"""

from __future__ import annotations

import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.data.schema import FINAL_TEST, SPLITS, Sample, SchemaError, may_drive_control
from src.utils.io import iter_jsonl

ACCESS_LOG_NAME = "final_test_access.log"


class FinalTestAccessError(PermissionError):
    """Raised when final test is read without explicit, justified permission."""


def split_path(data_dir: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise SchemaError(f"unknown split {split!r}; expected one of {SPLITS}")
    return Path(data_dir) / f"{split}.jsonl"


def load_split(
    data_dir: str | Path,
    split: str,
    allow_final_test: bool = False,
    reason: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Sample]:
    """Load one split as :class:`Sample` objects.

    ``split`` is always explicit - there is no default - so no code path can
    "accidentally read test" (data protocol: evaluator must declare its split).
    """
    path = split_path(data_dir, split)
    if split == FINAL_TEST:
        if not allow_final_test:
            raise FinalTestAccessError(
                "final_test may only be read with allow_final_test=True and a reason; "
                "it must never influence routing, hyper-parameters or checkpoint selection"
            )
        if not reason or not reason.strip():
            raise FinalTestAccessError("reading final_test requires a non-empty reason for the audit log")
        _log_final_test_access(Path(data_dir), reason)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")

    samples: List[Sample] = []
    for record in iter_jsonl(path):
        samples.append(Sample.from_record(record))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _log_final_test_access(data_dir: Path, reason: str) -> None:
    import inspect

    caller = "unknown"
    stack = inspect.stack()
    for frame in stack[2:]:
        if "src/data/access.py" not in frame.filename:
            caller = f"{frame.filename}:{frame.lineno} in {frame.function}"
            break
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{caller}\treason={reason.strip()}\n"
    log_path = data_dir / ACCESS_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def final_test_access_count(data_dir: str | Path) -> int:
    log_path = Path(data_dir) / ACCESS_LOG_NAME
    if not log_path.exists():
        return 0
    return sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())


def describe_split(data_dir: str | Path, split: str, **kwargs: Any) -> Dict[str, Any]:
    samples = load_split(data_dir, split, **kwargs)
    domains: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for s in samples:
        domains[s.domain] = domains.get(s.domain, 0) + 1
        sources[s.source] = sources.get(s.source, 0) + 1
    return {
        "split": split,
        "count": len(samples),
        "domains": dict(sorted(domains.items())),
        "sources": dict(sorted(sources.items())),
        "may_drive_control": may_drive_control(split),
    }


# -- Data Protocol v2 consumer capabilities --------------------------------


class FinalManifestAccessError(PermissionError):
    """Raised when a training-time consumer is handed final capability data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_role_records_artifact(
    manifest: Mapping[str, Any], records_path: str | Path, *, role: str
) -> None:
    """Verify a role's declared records artifact before a consumer reads it."""

    roles = manifest.get("roles")
    metadata = roles.get(role) if isinstance(roles, Mapping) else None
    files = metadata.get("files") if isinstance(metadata, Mapping) else None
    if not isinstance(files, list) or not files:
        raise ValueError(f"manifest role {role} lacks records artifact metadata")
    path = Path(records_path)
    if not path.is_file():
        raise ValueError(f"records artifact is not a readable file: {path}")
    resolved = path.resolve()

    def declared_matches(item: Mapping[str, Any]) -> bool:
        declared_text = str(item.get("path", ""))
        if not declared_text:
            return False
        declared = Path(declared_text)
        if ".." in declared.parts:
            return False
        if declared.is_absolute():
            return declared.resolve() == resolved
        # P1 manifests commonly use a basename; P2 manifests use a portable
        # repository-relative path.  Match the latter as a complete suffix of
        # the consumer's absolute path, never as a substring.
        parts = declared.parts
        return (
            (len(parts) == 1 and declared.name == path.name)
            or (len(parts) <= len(resolved.parts) and resolved.parts[-len(parts) :] == parts)
        )

    matches = [item for item in files if isinstance(item, Mapping) and declared_matches(item)]
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise ValueError(f"manifest role {role} does not uniquely bind records artifact {path.name}")
    if _sha256(path) != str(matches[0]["sha256"]):
        raise ValueError("records SHA-256 mismatch")


def _protocol_v2_manifest(value: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    from src.data.schema import DATA_PROTOCOL_VERSION, SOURCE_POLICY_VERSION

    if isinstance(value, Mapping):
        manifest = dict(value)
    else:
        manifest = json.loads(Path(value).read_text(encoding="utf-8"))
    if manifest.get("data_protocol_version") != DATA_PROTOCOL_VERSION:
        raise ValueError("manifest does not declare Data Protocol v2")
    if manifest.get("source_policy_version") != SOURCE_POLICY_VERSION:
        raise ValueError("manifest does not declare the current source policy version")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError("manifest roles must be a non-empty mapping")
    return manifest


def _reject_final_roles(roles: set[str], *, consumer: str) -> None:
    from src.data.schema import FINAL_ROLES_V2

    final_roles = roles & set(FINAL_ROLES_V2)
    if final_roles:
        raise FinalManifestAccessError(
            f"{consumer} cannot receive final manifest roles: {sorted(final_roles)}"
        )


def load_manifest_for_trainer(
    value: Mapping[str, Any] | str | Path,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Validate a role-scoped SFT or OPD manifest for a trainer."""

    from src.data.schema import PROMPT_ONLY_ROLES_V2

    manifest = _protocol_v2_manifest(value)
    roles = set(manifest["roles"])
    _reject_final_roles(roles, consumer="trainer")
    if stage == "sft":
        allowed = {"medical_sft_train", "medical_sft_dev"}
    elif stage == "opd":
        allowed = set(PROMPT_ONLY_ROLES_V2)
    else:
        raise ValueError("trainer stage must be sft or opd")
    disallowed = roles - allowed
    if disallowed or not roles:
        raise PermissionError(
            f"{stage} trainer manifest contains disallowed roles: {sorted(disallowed)}"
        )
    return manifest


def load_manifest_for_scheduler(
    value: Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Validate a controller-only manifest for the CA-OPD scheduler."""

    from src.data.schema import CONTROLLER_ROLES_V2

    manifest = _protocol_v2_manifest(value)
    roles = set(manifest["roles"])
    _reject_final_roles(roles, consumer="scheduler")
    disallowed = roles - set(CONTROLLER_ROLES_V2)
    if disallowed or not set(CONTROLLER_ROLES_V2).issubset(roles):
        raise PermissionError(
            "scheduler requires exactly medical/general controller roles; "
            f"disallowed={sorted(disallowed)}"
        )
    return manifest
