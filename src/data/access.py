"""Split access with an auditable final-test guard.

Reading ``final_test`` requires three things at once: the explicit split name,
``allow_final_test=True`` and a human-readable ``reason``. Every such access is
appended to ``final_test_access.log`` next to the data, so "final test was only
read once, after the checkpoint was frozen" becomes a checkable claim instead of
a promise.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
