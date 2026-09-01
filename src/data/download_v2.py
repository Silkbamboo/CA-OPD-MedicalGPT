"""Exact, resumable and fail-closed downloads for the P2 formal builder."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Protocol


_HEX40 = re.compile(r"[0-9a-f]{40}")
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
)


class DownloadPolicyError(RuntimeError):
    """The request violates the immutable-file or resource policy."""


class DownloadIncompleteError(DownloadPolicyError):
    """A retryable early EOF; the verified partial file is preserved."""


class Transport(Protocol):
    def metadata(self, url: str, timeout: int) -> dict[str, object]: ...

    def chunks(
        self, url: str, start: int, chunk_size: int, timeout: int
    ) -> Iterable[bytes]: ...


class UrllibTransport:
    """Small HTTP transport with explicit timeouts and Range support."""

    user_agent = "CA-OPD-P2-formal-builder/1"

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
                raise DownloadPolicyError("server ignored Range resume request")
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk


@dataclass(frozen=True)
class ExactFileSpec:
    repository: str
    revision: str
    path: str
    allowed_paths: tuple[str, ...]
    max_bytes: int
    expected_sha256: str | None = None
    host: str = "huggingface"
    timeout_seconds: int = 30

    def url(self) -> str:
        quoted_path = urllib.parse.quote(self.path, safe="/")
        if self.host == "huggingface":
            return (
                f"https://huggingface.co/datasets/{self.repository}/resolve/"
                f"{self.revision}/{quoted_path}?download=true"
            )
        if self.host == "huggingface_model":
            return (
                f"https://huggingface.co/{self.repository}/resolve/"
                f"{self.revision}/{quoted_path}?download=true"
            )
        if self.host == "github":
            return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{quoted_path}"
        raise DownloadPolicyError(f"unsupported download host: {self.host}")


@dataclass(frozen=True)
class DownloadResult:
    path: str
    url: str
    bytes: int
    sha256: str
    resumed_from_bytes: int
    etag: str | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_spec(spec: ExactFileSpec) -> None:
    if _HEX40.fullmatch(spec.revision) is None:
        raise DownloadPolicyError("revision must be an immutable 40-hex commit")
    pure_path = PurePosixPath(spec.path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise DownloadPolicyError("download path must be a safe repository-relative path")
    if spec.path not in spec.allowed_paths:
        raise DownloadPolicyError(f"path is not in exact file allowlist: {spec.path}")
    basename = pure_path.name.casefold()
    if any(fnmatch.fnmatch(basename, pattern) for pattern in _WEIGHT_PATTERNS):
        raise DownloadPolicyError(f"model weight download is forbidden: {spec.path}")
    if type(spec.max_bytes) is not int or spec.max_bytes <= 0:
        raise DownloadPolicyError("max_bytes must be a positive integer")
    if spec.expected_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", spec.expected_sha256
    ) is None:
        raise DownloadPolicyError("expected_sha256 must be lowercase SHA-256")


def download_exact(
    spec: ExactFileSpec,
    destination: str | Path,
    *,
    transport: Transport | None = None,
    chunk_size: int = 1024 * 1024,
) -> DownloadResult:
    """Download one exact file and atomically publish it after SHA verification."""

    _validate_spec(spec)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    client = transport or UrllibTransport()
    url = spec.url()
    metadata = client.metadata(url, spec.timeout_seconds)
    declared = metadata.get("content_length")
    if not isinstance(declared, int) or declared < 0:
        raise DownloadPolicyError("server did not provide a valid Content-Length")
    if declared > spec.max_bytes:
        raise DownloadPolicyError(
            f"declared size {declared} exceeds configured max_bytes {spec.max_bytes}"
        )

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        existing_size = target.stat().st_size
        existing_sha = sha256_file(target)
        if existing_size != declared:
            raise DownloadPolicyError("existing final artifact size differs from metadata")
        if spec.expected_sha256 and existing_sha != spec.expected_sha256:
            raise DownloadPolicyError("existing final artifact SHA-256 mismatch")
        return DownloadResult(
            path=str(target),
            url=url,
            bytes=existing_size,
            sha256=existing_sha,
            resumed_from_bytes=existing_size,
            etag=str(metadata.get("etag")) if metadata.get("etag") else None,
        )

    part = target.with_suffix(target.suffix + ".part")
    resumed = part.stat().st_size if part.exists() else 0
    if resumed > declared:
        raise DownloadPolicyError("partial file exceeds declared remote size")
    mode = "ab" if resumed else "wb"
    written = resumed
    with part.open(mode) as handle:
        for chunk in client.chunks(url, resumed, chunk_size, spec.timeout_seconds):
            written += len(chunk)
            if written > declared or written > spec.max_bytes:
                raise DownloadPolicyError("download exceeded admitted byte budget")
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if written != declared:
        raise DownloadIncompleteError(
            f"download incomplete: wrote {written} bytes, expected {declared}"
        )
    digest = sha256_file(part)
    if spec.expected_sha256 and digest != spec.expected_sha256:
        raise DownloadPolicyError("downloaded artifact SHA-256 mismatch")
    os.replace(part, target)
    return DownloadResult(
        path=str(target),
        url=url,
        bytes=written,
        sha256=digest,
        resumed_from_bytes=resumed,
        etag=str(metadata.get("etag")) if metadata.get("etag") else None,
    )
