"""Deterministic, versioned medical exclusion and capability taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.schema import normalize_question_v2


@dataclass(frozen=True)
class TaxonomyDecision:
    status: str
    admitted: bool
    capability: str | None
    rule_ids: tuple[str, ...]


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload.get("taxonomy_version"):
        raise ValueError("taxonomy config must declare taxonomy_version")
    if payload.get("policy") != "uncertain_fail_closed":
        raise ValueError("formal taxonomy must be uncertain_fail_closed")
    return dict(payload)


def _matches(text: str, rule: Mapping[str, Any]) -> bool:
    folded = text.casefold()
    terms = rule.get("terms", ())
    if any(str(term).casefold() in folded for term in terms):
        return True
    pattern = rule.get("regex")
    return bool(pattern and re.search(str(pattern), text))


def classify_general_prompt(
    text: str, taxonomy: Mapping[str, Any]
) -> TaxonomyDecision:
    normalized = normalize_question_v2(text)
    medical = tuple(
        str(rule["id"])
        for rule in taxonomy.get("medical_rules", ())
        if _matches(normalized, rule)
    )
    if medical:
        return TaxonomyDecision("rejected", False, None, medical)
    uncertain = tuple(
        str(rule["id"])
        for rule in taxonomy.get("uncertain_rules", ())
        if _matches(normalized, rule)
    )
    if uncertain:
        return TaxonomyDecision("uncertain", False, None, uncertain)
    for capability, rule in taxonomy.get("capabilities", {}).items():
        if capability != "other" and _matches(normalized, rule):
            return TaxonomyDecision(
                "accepted", True, str(capability), (f"capability:{capability}",)
            )
    return TaxonomyDecision("accepted", True, "other", ("capability:other",))
