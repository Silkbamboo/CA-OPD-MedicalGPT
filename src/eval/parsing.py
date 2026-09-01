"""MCQ answer extraction.

Requirement (docs/REPRODUCIBILITY.md §6.3): "选择题解析对格式变化鲁棒但不猜测答案". Those two
goals conflict, so the resolution is explicit: strategies are tried in order of
decreasing evidence strength, and if no strategy yields a *unique* letter the
parse fails (``None``). A failed parse counts as an incorrect answer and is
reported separately as ``unparsed_rate`` - never silently mapped to option A,
and never resolved by picking "the first letter that appears".

Strategy order
--------------
1. ``explicit``     an answer statement such as ``答案是B`` / ``正确选项：C`` /
                    ``answer: D`` (last occurrence wins - models often restate).
2. ``boxed``        ``\\boxed{B}`` (common in reasoning-style outputs).
3. ``leading``      the response begins with ``B``, ``B.``, ``(B)`` ...
4. ``unique_letter`` exactly one distinct standalone option letter in the text.
5. ``option_text``  exactly one option's full text appears and no letter does.

Note on false positives: a standalone letter must be surrounded by non-word
characters, so "维生素A" does not register as an answer while "维生素 A" would.
That residual ambiguity is why ``option_text`` runs last and why unparsed
responses are reported rather than hidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

LETTERS = "ABCDEFGH"

_EXPLICIT_RE = re.compile(
    r"(?:答案|答桉|正确答案|正确选项|应选|选择|故选|选|answer|Answer|ANSWER)"
    r"\s*(?:是|为|应该是|应为|选|:|：|＝|=|is|are)?\s*"
    r"[（(\[【]?\s*([A-Ha-h])\s*[)）\]】.。、,，:：]?",
)
_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Ha-h])\s*\}")
_LEADING_RE = re.compile(r"^\s*[（(\[【]?\s*([A-Ha-h])\s*[)）\]】]?\s*(?:[.。、,，:：]|$|\s)")


def _standalone_letters(text: str, valid: str) -> List[str]:
    found: List[str] = []
    for match in re.finditer(r"[A-Ha-h]", text):
        letter = match.group(0).upper()
        if letter not in valid:
            continue
        start, end = match.start(), match.end()
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if before.isalnum() and before.isascii():
            continue
        if after.isalnum() and after.isascii():
            continue
        # a CJK character immediately adjacent means the letter is part of a term
        if _is_cjk(before) or _is_cjk(after):
            continue
        found.append(letter)
    return found


def _is_cjk(ch: str) -> bool:
    return bool(ch) and "\u4e00" <= ch <= "\u9fff"


@dataclass(frozen=True)
class ParsedAnswer:
    """Result of parsing one model response."""

    letter: Optional[str]
    index: Optional[int]
    method: str
    raw_candidates: Sequence[str] = ()

    @property
    def parsed(self) -> bool:
        return self.letter is not None

    def as_dict(self) -> Dict[str, object]:
        return {
            "letter": self.letter,
            "index": self.index,
            "method": self.method,
            "candidates": list(self.raw_candidates),
        }


def parse_mcq_answer(response: str, num_options: int = 4) -> ParsedAnswer:
    """Extract the chosen option letter from a model response."""
    if num_options < 2 or num_options > len(LETTERS):
        raise ValueError(f"num_options must be in [2, {len(LETTERS)}], got {num_options}")
    valid = LETTERS[:num_options]
    if response is None:
        return ParsedAnswer(None, None, "empty")
    text = str(response).strip()
    if not text:
        return ParsedAnswer(None, None, "empty")

    # 1. explicit answer statement (last one wins)
    explicit = [m.group(1).upper() for m in _EXPLICIT_RE.finditer(text)]
    explicit = [c for c in explicit if c in valid]
    if explicit:
        letter = explicit[-1]
        return ParsedAnswer(letter, valid.index(letter), "explicit", explicit)

    # 2. \boxed{}
    boxed = [m.group(1).upper() for m in _BOXED_RE.finditer(text) if m.group(1).upper() in valid]
    if boxed:
        letter = boxed[-1]
        return ParsedAnswer(letter, valid.index(letter), "boxed", boxed)

    # 3. leading letter
    lead = _LEADING_RE.match(text)
    if lead and lead.group(1).upper() in valid:
        letter = lead.group(1).upper()
        return ParsedAnswer(letter, valid.index(letter), "leading", [letter])

    # 4. exactly one distinct standalone letter
    letters = _standalone_letters(text, valid)
    distinct = sorted(set(letters))
    if len(distinct) == 1:
        letter = distinct[0]
        return ParsedAnswer(letter, valid.index(letter), "unique_letter", letters)

    return ParsedAnswer(None, None, "ambiguous" if distinct else "no_answer", distinct)


def parse_with_options(response: str, options: Sequence[str]) -> ParsedAnswer:
    """Like :func:`parse_mcq_answer` but can also match a unique option text."""
    parsed = parse_mcq_answer(response, num_options=len(options))
    if parsed.parsed:
        return parsed
    text = str(response or "")
    hits = [i for i, opt in enumerate(options) if opt and str(opt).strip() and str(opt).strip() in text]
    if len(hits) == 1:
        index = hits[0]
        return ParsedAnswer(LETTERS[index], index, "option_text", [LETTERS[index]])
    return parsed
