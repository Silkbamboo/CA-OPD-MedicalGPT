"""Deterministic IO helpers.

Deterministic serialisation matters for docs/METHOD.md §8.3 ("同 seed 两次构建
产物一致"): we always write UTF-8, ``ensure_ascii=False``, sorted keys for
manifests and a trailing newline per record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]], sort_keys: bool = True) -> int:
    """Write records as JSONL; returns the number of records written."""
    p = Path(path)
    ensure_dir(p.parent)
    count = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=sort_keys) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # fail loudly, never skip silently
                raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc


def write_json(path: str | Path, payload: Mapping[str, Any], sort_keys: bool = True) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, indent=2) + "\n",
        encoding="utf-8",
    )
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    """Hash of a file's bytes - used to pin data files inside manifests."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
