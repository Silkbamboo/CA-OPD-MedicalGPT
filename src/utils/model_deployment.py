"""Immutable, resumable Qwen3-4B deployment without loading model tensors."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol, Sequence


MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_LICENSE = "Apache-2.0"
MODELSCOPE_REVISION = "2c54d5a09e7e92d4f5126b92a5a457448c9593e6"

REQUIRED_MODEL_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
TOKENIZER_ARTIFACT_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")


class ModelDeploymentError(RuntimeError):
    pass


def parse_range_response(status: int, content_range: str, start: int, end: int) -> int:
    """Validate the exact byte window even when ModelScope uses HTTP 200 for Git blobs."""

    if status not in {200, 206}:
        raise ModelDeploymentError(f"unexpected HTTP range status: {status}")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
    if match is None or (int(match.group(1)), int(match.group(2))) != (start, end):
        raise ModelDeploymentError("transfer source returned an invalid Content-Range")
    total = int(match.group(3))
    if total <= end:
        raise ModelDeploymentError("transfer source Content-Range total is invalid")
    return total


class ModelTransport(Protocol):
    def metadata(self, url: str, timeout: int) -> dict[str, object]: ...

    def chunks(
        self, url: str, start: int, chunk_size: int, timeout: int
    ) -> Iterable[bytes]: ...


class RangeModelTransport(Protocol):
    def metadata(self, url: str, timeout: int) -> dict[str, object]: ...

    def range_chunks(
        self, url: str, start: int, end: int, chunk_size: int, timeout: int
    ) -> Iterable[bytes]: ...


class UrllibModelTransport:
    user_agent = "CA-OPD-Qwen3-4B-deployment/1"

    def metadata(self, url: str, timeout: int) -> dict[str, object]:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": self.user_agent}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return {
                "content_length": int(length) if length is not None else None,
                "etag": response.headers.get("ETag"),
                "resolved_url": response.geturl(),
            }

    def chunks(
        self, url: str, start: int, chunk_size: int, timeout: int
    ) -> Iterable[bytes]:
        headers = {"User-Agent": self.user_agent}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if start and getattr(response, "status", None) != 206:
                raise ModelDeploymentError("server ignored Range resume request")
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk


class UrllibRangeModelTransport:
    """Bounded HTTP Range reader; it never holds a complete block in memory."""

    user_agent = "CA-OPD-Qwen3-4B-deployment/1"

    def metadata(self, url: str, timeout: int) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Range": "bytes=0-0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range", "")
            total = parse_range_response(
                int(getattr(response, "status", 0)), content_range, 0, 0
            )
            return {
                "content_length": total,
                "etag": response.headers.get("ETag"),
                "resolved_url": response.geturl(),
            }

    def range_chunks(
        self, url: str, start: int, end: int, chunk_size: int, timeout: int
    ) -> Iterable[bytes]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Range": f"bytes={start}-{end}",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = parse_range_response(
                int(getattr(response, "status", 0)),
                response.headers.get("Content-Range", ""),
                start,
                end,
            )
            if total <= end:
                raise ModelDeploymentError("transfer source range total is too small")
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        offset = 0
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            offset += len(chunk)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    handle.fileno(),
                    max(0, offset - len(chunk)),
                    len(chunk),
                    os.POSIX_FADV_DONTNEED,
                )
    return digest.hexdigest()


def validate_model_file_plan(
    model_id: str, revision: str, files: Sequence[ModelFile]
) -> None:
    if model_id != MODEL_ID:
        raise ModelDeploymentError("only Qwen/Qwen3-4B is admitted")
    if _HEX40.fullmatch(revision) is None or revision != MODEL_REVISION:
        raise ModelDeploymentError("model revision must equal the approved immutable commit")
    paths = [file.path for file in files]
    if len(paths) != len(set(paths)) or set(paths) != set(REQUIRED_MODEL_FILES):
        raise ModelDeploymentError("model file plan differs from the exact allowlist")
    for file in files:
        pure = PurePosixPath(file.path)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            raise ModelDeploymentError("model allowlist paths must be safe basenames")
        if type(file.size) is not int or file.size <= 0:
            raise ModelDeploymentError("model file sizes must be positive integers")
        if _HEX64.fullmatch(file.sha256) is None:
            raise ModelDeploymentError("model files require lowercase SHA-256")


def check_disk_budget(*, free_bytes: int, planned_bytes: int, safety_bytes: int) -> None:
    if min(free_bytes, planned_bytes, safety_bytes) < 0:
        raise ModelDeploymentError("disk budget values must be non-negative")
    if free_bytes - planned_bytes < safety_bytes:
        raise ModelDeploymentError("planned deployment violates the post-download safety margin")


def validate_transfer_source(repository: str, transfer_revision: str) -> None:
    if repository != MODEL_ID:
        raise ModelDeploymentError("transfer source must be the official Qwen/Qwen3-4B repository")
    if _HEX40.fullmatch(transfer_revision) is None or transfer_revision != MODELSCOPE_REVISION:
        raise ModelDeploymentError("transfer source requires the approved immutable revision")


def _url(repository: str, revision: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}?download=true"


def _modelscope_url(repository: str, revision: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://www.modelscope.cn/models/{repository}/resolve/{revision}/{quoted}"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def download_model_file_ranges(
    *,
    repository: str,
    canonical_revision: str,
    transfer_revision: str,
    spec: ModelFile,
    destination: str | Path,
    transport: RangeModelTransport | None = None,
    block_size: int = 64 * 1024 * 1024,
    io_chunk_size: int = 1024 * 1024,
    max_workers: int = 4,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Download disjoint ranges into one sparse partial file with durable block state.

    The ModelScope repository is only a transfer path. The admitted bytes remain
    bound to the canonical Hugging Face revision through ``spec.size`` and
    ``spec.sha256``; the file is published only after a full SHA-256 pass.
    """

    if canonical_revision != MODEL_REVISION:
        raise ModelDeploymentError("canonical revision is not the approved Qwen3-4B commit")
    validate_transfer_source(repository, transfer_revision)
    if spec.path not in REQUIRED_MODEL_FILES:
        raise ModelDeploymentError("download path is outside the model allowlist")
    if _HEX64.fullmatch(spec.sha256) is None or spec.size <= 0:
        raise ModelDeploymentError("range download requires pinned size and SHA-256")
    if block_size <= 0 or io_chunk_size <= 0 or not 1 <= max_workers <= 8:
        raise ValueError("invalid bounded range download resource limits")

    client = transport or UrllibRangeModelTransport()
    url = _modelscope_url(repository, transfer_revision, spec.path)
    metadata = client.metadata(url, timeout_seconds)
    if metadata.get("content_length") != spec.size:
        raise ModelDeploymentError(
            f"transfer Content-Length for {spec.path} differs from pinned inventory"
        )

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    block_count = (spec.size + block_size - 1) // block_size
    if target.is_file():
        if target.stat().st_size != spec.size or sha256_file(target) != spec.sha256:
            raise ModelDeploymentError(f"existing model file is corrupt: {spec.path}")
        return {
            "path": str(target),
            "bytes": spec.size,
            "sha256": spec.sha256,
            "resumed_blocks": block_count,
            "canonical_revision": canonical_revision,
            "transfer_revision": transfer_revision,
            "etag": metadata.get("etag"),
        }

    partial = target.with_suffix(target.suffix + ".part")
    state_path = target.with_suffix(target.suffix + ".ranges.json")
    state_contract = {
        "path": spec.path,
        "size": spec.size,
        "sha256": spec.sha256,
        "block_size": block_size,
        "canonical_revision": canonical_revision,
        "transfer_revision": transfer_revision,
    }
    completed: set[int] = set()
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        values = state.get("completed_blocks", [])
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise ModelDeploymentError(f"invalid range resume state: {spec.path}")
        contract_mismatch = any(
            state.get(key) != value for key, value in state_contract.items()
        )
        if contract_mismatch and values:
            raise ModelDeploymentError(f"range resume state contract mismatch: {spec.path}")
        completed = set() if contract_mismatch else set(values)
    resumed_blocks = len(completed)

    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    lock = threading.Lock()
    try:
        os.ftruncate(fd, spec.size)
        _atomic_json(state_path, {**state_contract, "completed_blocks": sorted(completed)})

        def fetch_block(index: int) -> int:
            start = index * block_size
            end = min(spec.size, start + block_size) - 1
            cursor = start
            for chunk in client.range_chunks(
                url, start, end, io_chunk_size, timeout_seconds
            ):
                if cursor + len(chunk) > end + 1:
                    raise ModelDeploymentError(f"range response exceeded block {index}")
                os.pwrite(fd, chunk, cursor)
                cursor += len(chunk)
            if cursor != end + 1:
                raise ModelDeploymentError(
                    f"incomplete range block {index}: {cursor - start}/{end - start + 1}"
                )
            os.fsync(fd)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, start, end - start + 1, os.POSIX_FADV_DONTNEED)
            with lock:
                completed.add(index)
                _atomic_json(
                    state_path,
                    {**state_contract, "completed_blocks": sorted(completed)},
                )
            return index

        missing = [index for index in range(block_count) if index not in completed]
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(missing)))) as pool:
            futures = [pool.submit(fetch_block, index) for index in missing]
            for future in as_completed(futures):
                future.result()
    finally:
        os.close(fd)

    digest = sha256_file(partial)
    if digest != spec.sha256:
        state_path.unlink(missing_ok=True)
        raise ModelDeploymentError(f"SHA-256 mismatch for {spec.path}")
    os.replace(partial, target)
    state_path.unlink(missing_ok=True)
    return {
        "path": str(target),
        "bytes": spec.size,
        "sha256": digest,
        "resumed_blocks": resumed_blocks,
        "canonical_revision": canonical_revision,
        "transfer_revision": transfer_revision,
        "etag": metadata.get("etag"),
    }


def download_model_file(
    *,
    repository: str,
    revision: str,
    spec: ModelFile,
    destination: str | Path,
    transport: ModelTransport | None = None,
    chunk_size: int = 4 * 1024 * 1024,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    if repository != MODEL_ID or revision != MODEL_REVISION:
        raise ModelDeploymentError("download target is not the approved immutable Qwen3-4B")
    if spec.path not in REQUIRED_MODEL_FILES:
        raise ModelDeploymentError("download path is outside the model allowlist")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    client = transport or UrllibModelTransport()
    url = _url(repository, revision, spec.path)
    metadata = client.metadata(url, timeout_seconds)
    declared = metadata.get("content_length")
    if declared != spec.size:
        raise ModelDeploymentError(
            f"remote Content-Length for {spec.path} differs from pinned inventory"
        )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size != spec.size or sha256_file(target) != spec.sha256:
            raise ModelDeploymentError(f"existing model file is corrupt: {spec.path}")
        return {
            "path": str(target),
            "bytes": spec.size,
            "sha256": spec.sha256,
            "resumed_from_bytes": spec.size,
            "etag": metadata.get("etag"),
        }
    partial = target.with_suffix(target.suffix + ".part")
    resumed = partial.stat().st_size if partial.exists() else 0
    if resumed > spec.size:
        raise ModelDeploymentError(f"partial file exceeds pinned size: {spec.path}")
    written = resumed
    with partial.open("ab" if resumed else "wb") as handle:
        for chunk in client.chunks(url, resumed, chunk_size, timeout_seconds):
            written += len(chunk)
            if written > spec.size:
                raise ModelDeploymentError(f"download exceeded pinned size: {spec.path}")
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if written != spec.size:
        raise ModelDeploymentError(
            f"incomplete model download for {spec.path}: {written}/{spec.size}"
        )
    digest = sha256_file(partial)
    if digest != spec.sha256:
        raise ModelDeploymentError(f"SHA-256 mismatch for {spec.path}")
    os.replace(partial, target)
    return {
        "path": str(target),
        "bytes": spec.size,
        "sha256": digest,
        "resumed_from_bytes": resumed,
        "etag": metadata.get("etag"),
    }


def verify_shard_index(index_path: str | Path, model_root: str | Path) -> list[str]:
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelDeploymentError("model shard index lacks weight_map")
    shards = sorted({str(value) for value in weight_map.values()})
    if len(shards) != 3 or any(_SHARD.fullmatch(shard) is None for shard in shards):
        raise ModelDeploymentError("model shard index does not reference the three BF16 shards")
    missing = [shard for shard in shards if not (Path(model_root) / shard).is_file()]
    if missing:
        raise ModelDeploymentError(f"model shard index references missing files: {missing}")
    return shards


def validate_safetensors_header(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    with file.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ModelDeploymentError(f"invalid safetensors header prefix: {file.name}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 1 or header_size > min(file.stat().st_size - 8, 128 * 1024 * 1024):
            raise ModelDeploymentError(f"invalid safetensors header length: {file.name}")
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDeploymentError(f"invalid safetensors JSON header: {file.name}") from error
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors or any(not isinstance(value, dict) for value in tensors):
        raise ModelDeploymentError(f"safetensors file has no tensor metadata: {file.name}")
    dtypes = sorted({str(value.get("dtype", "")) for value in tensors})
    return {
        "path": str(file),
        "header_bytes": header_size,
        "tensor_count": len(tensors),
        "dtypes": dtypes,
        "full_tensor_payload_loaded": False,
    }


def build_model_manifest(
    *,
    model_root: str | Path,
    files: Sequence[ModelFile],
    verification_status: str,
    download_started_at: str,
    download_ended_at: str,
) -> dict[str, Any]:
    root = Path(model_root)
    inventory = []
    for spec in files:
        path = root / spec.path
        if not path.is_file() or path.stat().st_size != spec.size:
            raise ModelDeploymentError(f"model manifest file is missing or wrong-sized: {spec.path}")
        digest = sha256_file(path)
        if digest != spec.sha256:
            raise ModelDeploymentError(f"model manifest SHA mismatch: {spec.path}")
        inventory.append({**asdict(spec), "local_path": str(path)})
    return {
        "model_id": MODEL_ID,
        "immutable_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "files": inventory,
        "total_bytes": sum(file.size for file in files),
        "local_persistent_path": str(root),
        "download_started_at": download_started_at,
        "download_ended_at": download_ended_at,
        "verification_status": verification_status,
        "full_model_loaded": False,
        "actual_cost_cny": None,
    }


def build_tokenizer_artifact_manifest(model_root: str | Path) -> dict[str, Any]:
    """Bind the local Qwen3 tokenizer/config files without treating weights as tokenizer assets."""

    root = Path(model_root)
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(TOKENIZER_ARTIFACT_FILES):
        path = root / name
        if not path.is_file():
            raise ModelDeploymentError(f"required tokenizer artifact is missing: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "files": files,
        "full_model_loaded": False,
    }


def validate_qwen3_lora_targets(
    config_path: str | Path,
    shard_index_path: str | Path,
    target_modules: str,
) -> dict[str, Any]:
    """Validate Qwen3 linear-module suffixes from metadata only.

    The shard index names every parameter, so this proves module naming without
    importing Transformers or reading any safetensors payload.
    """

    if target_modules != "all-linear":
        raise ModelDeploymentError("Qwen3-4B MVP requires target_modules=all-linear")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in config.get("architectures", []):
        raise ModelDeploymentError("model config is not Qwen3ForCausalLM")
    index = json.loads(Path(shard_index_path).read_text(encoding="utf-8"))
    keys = set((index.get("weight_map") or {}).keys())
    expected = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    missing = [name for name in expected if not any(key.endswith(f".{name}.weight") for key in keys)]
    if missing:
        raise ModelDeploymentError(f"Qwen3 shard index lacks LoRA target module(s): {missing}")
    return {
        "architecture": "Qwen3ForCausalLM",
        "target_modules": target_modules,
        "expected_linear_suffixes": list(expected),
        "model_weights_loaded": False,
        "pending_gpu_structure_check": False,
    }
