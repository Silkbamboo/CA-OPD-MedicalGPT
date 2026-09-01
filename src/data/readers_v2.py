"""Bounded-memory row iterators used by the formal builder."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield dict(value)


def iter_json_array(
    path: str | Path, *, chunk_bytes: int = 1024 * 1024
) -> Iterator[dict[str, Any]]:
    """Incrementally decode a top-level JSON array without materializing it."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            piece = handle.read(chunk_bytes)
            if piece == "":
                eof = True
                return False
            buffer += piece
            return True

        while not buffer and fill():
            pass
        while True:
            while position >= len(buffer) and not eof:
                fill()
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError("empty JSON document")
        if buffer[position] != "[":
            raise ValueError("expected a top-level JSON array")
        position += 1

        expect_value = True
        while True:
            while True:
                while position >= len(buffer) and not eof:
                    fill()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break
            if position >= len(buffer):
                raise ValueError("unterminated JSON array")
            character = buffer[position]
            if character == "]":
                return
            if not expect_value:
                if character != ",":
                    raise ValueError("expected comma between JSON array rows")
                position += 1
                expect_value = True
                continue
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            if not isinstance(value, Mapping):
                raise ValueError("JSON array row is not an object")
            position = end
            expect_value = False
            yield dict(value)


def iter_parquet(
    path: str | Path, *, batch_size: int = 64
) -> Iterator[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {name: values[index] for name, values in columns.items()}


def iter_records(
    path: str | Path, file_format: str, *, batch_size: int = 64
) -> Iterator[dict[str, Any]]:
    if file_format == "jsonl":
        return iter_jsonl(path)
    if file_format == "json_array":
        return iter_json_array(path)
    if file_format == "parquet":
        return iter_parquet(path, batch_size=batch_size)
    raise ValueError(f"unsupported streaming file format: {file_format}")
