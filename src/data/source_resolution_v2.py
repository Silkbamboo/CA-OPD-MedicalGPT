"""Narrow, fail-closed source policy for the P1.6 audit.

This module contains deterministic policy and row-shape logic only.  Network
access and bounded raw-file acquisition live in the P1.6 audit script so unit
tests never contact upstream services.  It deliberately reuses the canonical
Data Protocol v2 adapter/schema instead of creating a second data model.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .adapters import AdapterContext, adapt_source_row
from .schema import SOURCE_POLICY_VERSION, SUPERVISION_KEYS


MEDQA_LICENSE = "unknown"
GPT4_LLM_DATA_LICENSE: Mapping[str, Any] = {
    "source_license": "CC-BY-NC-4.0",
    "license_status": "verified",
    "usage_scope": "noncommercial_research_only",
    "research_training_allowed": True,
    "public_checkpoint_release": False,
    "raw_data_redistribution": False,
    "attribution_required": True,
}

_HEX40 = re.compile(r"[0-9a-f]{40}")
_MEDICAL_MARKERS = (
    "医学",
    "医疗",
    "临床",
    "患者",
    "诊断",
    "治疗",
    "症状",
    "病因",
    "疾病",
    "处方",
    "药物",
    "药品",
    "医生",
    "医师",
    "护士",
    "医院",
    "手术",
    "疫苗",
    "感染",
    "肿瘤",
    "癌症",
    "糖尿病",
    "高血压",
    "心肌梗死",
    "脑卒中",
    "剂量",
    "用药",
    "疾病",
    "medical",
    "medicine",
    "clinical",
    "patient",
    "diagnosis",
    "treatment",
    "symptom",
    "disease",
    "prescription",
    "medication",
    "doctor",
    "nurse",
    "hospital",
    "surgery",
    "vaccine",
    "infection",
    "cancer",
    "diabetes",
    "hypertension",
)


class SourceResolutionError(ValueError):
    """A pinned representation or source-policy contract was violated."""


def _require_exact_revision(configured: str, resolved: str, *, label: str) -> str:
    if not _HEX40.fullmatch(str(configured)):
        raise SourceResolutionError(f"{label} revision must be exact lowercase 40-hex")
    if not _HEX40.fullmatch(str(resolved)):
        raise SourceResolutionError(f"{label} resolved revision must be lowercase 40-hex")
    if configured != resolved:
        raise SourceResolutionError(
            f"{label} resolved revision {resolved} does not match configured revision {configured}"
        )
    return configured


def resolve_medqa_representation(
    *,
    configured_revision: str,
    resolved_revision: str,
    available_configs: Mapping[str, Iterable[str]],
    evaluator_requires_four_options: bool = False,
) -> dict[str, Any]:
    """Select the closest parquet representation without changing semantics."""

    revision = _require_exact_revision(
        configured_revision, resolved_revision, label="MedQA"
    )
    normalized = {name: set(splits) for name, splits in available_configs.items()}
    required_splits = {"validation", "test"}
    preferred = (
        "med_qa_zh_4options_source"
        if evaluator_requires_four_options
        else "med_qa_zh_source"
    )
    if required_splits <= normalized.get(preferred, set()):
        selected = preferred
    elif required_splits <= normalized.get("med_qa_zh_source", set()):
        selected = "med_qa_zh_source"
    else:
        raise SourceResolutionError(
            "MedQA source representation lacks validation/test parquet splits"
        )
    return {
        "configured_revision": revision,
        "resolved_revision": revision,
        "selected_config": selected,
        "bigbio_qa_selected": selected == "med_qa_zh_bigbio_qa",
        "source_license": MEDQA_LICENSE,
        "usage_scope": "local_evaluation_only",
        "redistribution_allowed": False,
        "raw_questions_committed": False,
        "primary_final_frozen": False,
        "source_policy_version": SOURCE_POLICY_VERSION,
    }


def selected_medqa_audit_revision(config: Mapping[str, Any]) -> str:
    """Return the frozen P1.6 MedQA revision; never auto-fallback."""

    selected = str(config.get("selected_revision") or "")
    _require_exact_revision(selected, selected, label="MedQA selected")
    candidates = {str(value) for value in config.get("candidate_revisions", ())}
    if selected not in candidates:
        raise SourceResolutionError("MedQA selected revision is not in the audited candidate set")
    return selected


def validate_resolution_limits(value: Mapping[str, Any]) -> dict[str, int]:
    """Enforce P1.6 ceilings independently of the broader P1.5 audit engine."""

    try:
        result = {
            "per_source_mib": int(value["per_source_mib"]),
            "total_mib": int(value["total_mib"]),
            "max_records_per_candidate": int(value["max_records_per_candidate"]),
            "timeout_seconds": int(value["timeout_seconds"]),
            "max_retries": int(value["max_retries"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise SourceResolutionError("P1.6 network limits are incomplete") from error
    if not 0 < result["per_source_mib"] <= 25:
        raise SourceResolutionError("P1.6 per-source budget exceeds 25 MiB")
    if not 0 < result["total_mib"] <= 75:
        raise SourceResolutionError("P1.6 total budget exceeds 75 MiB")
    if not 0 < result["max_records_per_candidate"] <= 50:
        raise SourceResolutionError("P1.6 candidate row cap exceeds 50")
    if result["timeout_seconds"] <= 0:
        raise SourceResolutionError("P1.6 timeout must be positive")
    if not 0 <= result["max_retries"] <= 2:
        raise SourceResolutionError("P1.6 retry cap exceeds two")
    return result


def audit_medqa_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    revision: str,
    upstream_split: str,
    raw_file_sha256: str,
) -> dict[str, Any]:
    """Pass bounded MedQA rows through the canonical adapter and return stats."""

    if upstream_split not in {"validation", "test"}:
        raise SourceResolutionError("MedQA audit only permits validation/test")
    if len(rows) > 50:
        raise SourceResolutionError("MedQA audit is capped at 50 records")
    role = {
        "validation": "medical_controller_dev",
        "test": "medical_final_test",
    }[upstream_split]
    context = AdapterContext(
        source_type="medqa_zh",
        source="medqa_zh",
        source_revision=revision,
        source_license=MEDQA_LICENSE,
        upstream_split=upstream_split,
        target_role=role,
        raw_file_sha256=raw_file_sha256,
    )
    accepted = 0
    reasons: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    for row in rows:
        result = adapt_source_row(row, context)
        if result.record is None:
            reasons[str(result.drop_reason)] += 1
            continue
        accepted += 1
        option_counts[str(len(result.record.options))] += 1
    return {
        "upstream_split": upstream_split,
        "target_role": role,
        "sampled": len(rows),
        "accepted": accepted,
        "dropped": len(rows) - accepted,
        "drop_reasons": dict(sorted(reasons.items())),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "source_license": MEDQA_LICENSE,
        "primary_final_frozen": False,
    }


def validate_coig_split(upstream_split: str) -> str:
    """Allow only the fixed-revision ``Default`` representation as anchors."""

    if upstream_split == "NoTranslate":
        raise SourceResolutionError("COIG NoTranslate must not enter general anchors")
    if upstream_split != "Default":
        raise SourceResolutionError("COIG formal upstream split must be Default")
    return upstream_split


def bind_cached_coig_evidence(
    *,
    revision: str,
    readme_sha256: str,
    raw_file_sha256s: Sequence[str],
    prior_source_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind reused ignored raw artifacts to the verified P1.5 producer result."""

    _require_exact_revision(revision, revision, label="COIG")
    if (
        prior_source_result.get("configured_revision") != revision
        or prior_source_result.get("resolved_revision") != revision
    ):
        raise SourceResolutionError("cached COIG evidence revision differs from P1.5")
    prior_raw = {str(value) for value in prior_source_result.get("raw_file_sha256s", ())}
    current_raw = {str(value) for value in raw_file_sha256s}
    if not current_raw or not current_raw <= prior_raw:
        raise SourceResolutionError("cached COIG raw artifact SHA is not bound to P1.5")
    license_evidence = prior_source_result.get("license_evidence", ())
    readme_bound = any(
        isinstance(item, Mapping)
        and item.get("evidence_revision") == revision
        and item.get("evidence_file_sha256") == readme_sha256
        for item in license_evidence
    )
    if not readme_bound:
        raise SourceResolutionError("cached COIG README evidence is not bound to P1.5")
    return {
        "status": "bound_to_p1_5_evidence",
        "revision": revision,
        "readme_sha256": readme_sha256,
        "raw_artifact_count": len(current_raw),
    }


def classify_coig_sources(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Classify the four frozen COIG categories from file-scoped evidence.

    An umbrella repository license never proves provenance for a mixed file.
    Human Value and Translated therefore require explicit evidence beyond the
    README header; absent that evidence they fail closed.
    """

    readme = str(evidence.get("readme_text") or "")
    expected_sha = hashlib.sha256(readme.encode("utf-8")).hexdigest()
    if evidence.get("readme_sha256") != expected_sha:
        raise SourceResolutionError("COIG README SHA-256 does not match evidence")
    fields = evidence.get("files")
    if not isinstance(fields, Mapping):
        raise SourceResolutionError("COIG evidence must bind source files")

    leetcode_verified = (
        "CC-BY-SA-4.0" in readme
        and "github.com/doocs/leetcode" in readme
        and fields.get("leetcode") == "leetcode_instructions.jsonl"
    )
    leetcode = {
        "decision": "include" if leetcode_verified else "exclude",
        "source_license": "CC-BY-SA-4.0" if leetcode_verified else "unknown",
        "license_status": "verified" if leetcode_verified else "blocked_license_evidence",
        "research_training_allowed": leetcode_verified,
        "redistribution_requires_attribution": leetcode_verified,
        "share_alike_notice_required": leetcode_verified,
        "max_quota": 800 if leetcode_verified else 0,
        "evidence_file_sha256": expected_sha,
        "upstream_source_url": "https://github.com/doocs/leetcode",
    }

    human_file_provenance = bool(evidence.get("human_file_authorship_verified"))
    human = {
        "decision": "include" if human_file_provenance else "exclude",
        "source_license": "Apache-2.0" if human_file_provenance else "unknown",
        "license_status": "verified" if human_file_provenance else "blocked_file_provenance",
        "max_quota": 500 if human_file_provenance else 0,
        "evidence_file_sha256": expected_sha,
    }

    provenance_names = {
        "source",
        "source_name",
        "dataset",
        "dataset_name",
        "origin",
        "provenance",
        "instance_license",
    }
    translated_fields = {
        str(value) for value in evidence.get("translated_row_fields", ())
    }
    has_row_provenance = bool(translated_fields & provenance_names)
    translated = {
        # A provenance column only makes per-row license resolution possible;
        # it is not itself license evidence.  A later builder may include only
        # rows whose referenced upstream license has separately been verified.
        "decision": "blocked" if has_row_provenance else "exclude",
        "source_license": "per-row" if has_row_provenance else "unknown",
        "license_status": "requires_row_license_validation" if has_row_provenance else "blocked_row_provenance",
        "row_level_provenance": has_row_provenance,
        # Merely having a provenance column is not enough to allocate formal
        # rows; each referenced upstream license still needs validation.
        "max_quota": 0,
        "evidence_file_sha256": expected_sha,
    }
    exam = {
        "decision": "exclude",
        "source_license": "unknown",
        "license_status": "unknown",
        "max_quota": 0,
        "evidence_file_sha256": expected_sha,
    }
    return {
        "leetcode": leetcode,
        "human_value": human,
        "translated": translated,
        "exam": exam,
    }


def validate_replacement_revision(configured: str, resolved: str) -> str:
    """Bind the official replacement repository to one immutable commit."""

    return _require_exact_revision(configured, resolved, label="replacement")


def _is_medical_prompt(text: str) -> bool:
    folded = text.casefold()
    return any(marker in folded for marker in _MEDICAL_MARKERS)


def _prompt_only(payload: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = set(SUPERVISION_KEYS) | {"answer_index", "chain_of_thought"}
    return {key: value for key, value in payload.items() if key not in forbidden}


def build_replacement_anchors(
    *,
    rows: Sequence[Mapping[str, Any]],
    revision: str,
    raw_file_sha256: str,
    max_records: int,
    protected_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Adapt bounded GPT4-LLM rows and emit physically prompt-only anchors."""

    if not _HEX40.fullmatch(revision):
        raise SourceResolutionError("replacement revision must be exact 40-hex")
    if max_records < 0 or max_records > 50:
        raise SourceResolutionError("replacement audit max_records must be in [0, 50]")
    protected = set(protected_hashes)
    context = AdapterContext(
        source_type="coig",
        source="gpt4_llm_alpaca_zh",
        source_revision=revision,
        source_license=str(GPT4_LLM_DATA_LICENSE["source_license"]),
        upstream_split="train",
        target_role="general_anchors",
        raw_file_sha256=raw_file_sha256,
        subsource="translated_general_instructions",
    )
    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for raw in rows:
        if len(records) >= max_records:
            break
        instruction = str(raw.get("instruction") or "").strip()
        extra = str(raw.get("input") or "").strip()
        prompt = instruction if not extra else f"{instruction}\n{extra}"
        if _is_medical_prompt(prompt):
            reasons["medical_prompt_excluded"] += 1
            continue
        result = adapt_source_row(raw, context)
        if result.record is None:
            reasons[str(result.drop_reason)] += 1
            continue
        if result.record.content_hash in protected:
            reasons["protected_hash_overlap"] += 1
            continue
        records.append(_prompt_only(result.record.to_dict()))
    sampled = min(len(rows), max_records + sum(reasons.values()))
    return {
        "source_policy_version": SOURCE_POLICY_VERSION,
        "source_license": GPT4_LLM_DATA_LICENSE["source_license"],
        "sampled": sampled,
        "accepted": len(records),
        "dropped": sum(reasons.values()),
        "drop_reasons": dict(sorted(reasons.items())),
        "records": records,
        "medical_filter_scope": "bounded_audit_keyword_screen_not_formal_taxonomy",
    }


def allocate_general_anchors(
    *, available_counts: Mapping[str, int], target_total: int = 4000
) -> dict[str, Any]:
    """Allocate without duplication or license relaxation.

    Priority is frozen as COIG LeetCode (up to 800), then the audited Chinese
    general-instruction replacement.  Human Value remains unavailable until
    file-scoped authorship evidence closes its license.
    """

    if target_total < 0:
        raise SourceResolutionError("target_total must be non-negative")
    leetcode = min(max(int(available_counts.get("coig_leetcode", 0)), 0), 800, target_total)
    remaining = target_total - leetcode
    replacement = min(max(int(available_counts.get("gpt4_zh", 0)), 0), remaining)
    actual = leetcode + replacement
    return {
        "target_total": target_total,
        "allocated": {"coig_leetcode": leetcode, "gpt4_zh": replacement},
        "actual_total": actual,
        "shortfall": target_total - actual,
        "duplicates_added": 0,
        "source_policy_version": SOURCE_POLICY_VERSION,
    }
