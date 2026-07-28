"""Text normalisation and stable hashing.

PROJECT_PLAN.md §8.3 requires deduplication "以原始样本 ID、规范化问题文本与选项哈希".
This module is the single definition of "normalised text" in the project; the
data builder, the leakage tests and the evaluators all import from here so a
hash computed in one place is comparable everywhere.

Normalisation rules (deliberately conservative - we must not merge two
genuinely different questions):

1. Unicode NFKC (folds full-width ASCII/digits to half-width).
2. Lowercase (safe for zh + en MCQ text).
3. Strip a fixed set of CJK/ASCII punctuation and all whitespace.
4. Nothing else: no stemming, no synonym folding, no number rewriting.

Option hashing sorts the option *texts* so that a shuffled-option duplicate of
the same question still collides.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Sequence

# Punctuation removed before hashing. Kept explicit (not a unicode category
# sweep) so the rule is auditable and stable across python versions.
_PUNCT = (
    "，。、；：？！“”‘’（）《》〈〉【】〖〗「」『』…—～·"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
_PUNCT_TABLE = {ord(ch): None for ch in _PUNCT}
_WS_RE = re.compile(r"\s+")

HASH_ALGORITHM = "sha256"
HASH_PREFIX_LEN = 32  # 128 bits of sha256 hex is ample for <1e6 samples


def normalize_text(text: str) -> str:
    """Return the canonical form of ``text`` used for hashing/dedup."""
    if text is None:
        raise ValueError("normalize_text() received None")
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.lower()
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _WS_RE.sub("", normalized)
    return normalized


def _digest(payload: str) -> str:
    return hashlib.new(HASH_ALGORITHM, payload.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]


def text_hash(text: str) -> str:
    """Hash of the normalised question/prompt text."""
    return _digest(normalize_text(text))


def options_hash(options: Sequence[str] | None) -> str:
    """Order-insensitive hash of MCQ option texts.

    Empty/None options hash to a fixed sentinel so open-ended samples still get
    a well-defined value instead of ``None`` leaking into manifests.
    """
    if not options:
        return _digest("<no-options>")
    normalized = sorted(normalize_text(opt) for opt in options)
    return _digest("\u0001".join(normalized))


def content_hash(text: str, options: Sequence[str] | None = None) -> str:
    """Combined question+options hash: the project-wide dedup key."""
    return _digest(f"{text_hash(text)}:{options_hash(options)}")


def stable_sample_id(source: str, raw_id: str | int, text: str) -> str:
    """Build a stable, human-debuggable ``sample_id``.

    Format ``<source>-<raw_id>-<h8>`` where ``h8`` are the first 8 hex chars of
    the normalised text hash. Including the text hash means a source that
    silently re-indexes its rows cannot alias two different samples onto one id.
    """
    if not source:
        raise ValueError("source must be a non-empty string")
    short = text_hash(text)[:8]
    raw = str(raw_id).strip().replace(" ", "_")
    return f"{source}-{raw}-{short}"


def iter_unique(hashes: Iterable[str]) -> set[str]:
    return set(hashes)
