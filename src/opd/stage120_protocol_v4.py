"""Pure P7 Stage-120 action, data-audit and route-state contracts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class P7Stage120Error(RuntimeError):
    """A P7 Stage-120 protocol or replay invariant differs."""


METHODS = ("IDT-v2", "CA-OPD-v2")
ACTIONS = ("medical", "general")
GENERAL_SOURCES = {
    "BAAI/COIG": ("Default", "CC-BY-SA-4.0"),
    "Instruction-Tuning-with-GPT-4/GPT-4-LLM": (
        "train",
        "CC-BY-NC-4.0",
    ),
}
SUPERVISION_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_idx",
        "label",
        "labels",
        "reasoning",
        "response",
        "responses",
        "solution",
        "solutions",
        "output",
        "outputs",
        "completion",
        "completions",
        "target",
        "targets",
    }
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_supervision(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in SUPERVISION_KEYS or _has_supervision(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_supervision(item) for item in value)
    return False


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P7Stage120Error(f"{label} is unreadable") from error
    if not isinstance(value, Mapping):
        raise P7Stage120Error(f"{label} is not an object")
    return dict(value)


def audit_general_anchor_records_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    leakage_report_path: Path | None = None,
) -> dict[str, Any]:
    """Audit General Anchors without opening Controller/final label records."""

    if expected_count <= 0 or len(rows) != expected_count:
        raise P7Stage120Error("General Anchor record count differs")
    sources: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    sample_ids: set[str] = set()
    group_ids: set[str] = set()
    content_hashes: set[str] = set()
    for row in rows:
        if _has_supervision(row):
            raise P7Stage120Error("General Anchor supervision field is present")
        source = str(row.get("source", ""))
        split = str(row.get("upstream_split", ""))
        licence = str(row.get("source_license", ""))
        sample_id = str(row.get("sample_id", ""))
        group_id = str(row.get("group_id", ""))
        content_hash = str(row.get("content_hash", ""))
        if not (
            row.get("target_role") == "general_anchors"
            and source in GENERAL_SOURCES
            and (split, licence) == GENERAL_SOURCES[source]
            and sample_id
            and group_id
            and len(content_hash) == 64
            and all(character in "0123456789abcdef" for character in content_hash)
        ):
            raise P7Stage120Error("General Anchor source/identity differs")
        if (
            sample_id in sample_ids
            or group_id in group_ids
            or content_hash in content_hashes
        ):
            raise P7Stage120Error("General Anchor identity is duplicated")
        sample_ids.add(sample_id)
        group_ids.add(group_id)
        content_hashes.add(content_hash)
        sources[source] += 1
        licences[licence] += 1
        splits[f"{source}:{split}"] += 1

    overlap: dict[str, Any] | None = None
    if leakage_report_path is not None:
        report = _read_json_object(leakage_report_path, "frozen leakage report")
        if not (
            report.get("status") == "PASS"
            and int(report.get("exact_overlap_count", -1)) == 0
            and int(report.get("final_hash_in_training_count", -1)) == 0
            and int(report.get("opd_supervision_field_count", -1)) == 0
            and int(report.get("unresolved_cross_role_candidate_count", -1)) == 0
        ):
            raise P7Stage120Error("frozen Controller/final overlap authority differs")
        overlap = {
            "passed": True,
            "report_sha256": _sha_file(leakage_report_path),
            "exact_overlap_count": 0,
            "final_hash_in_training_count": 0,
            "unresolved_cross_role_candidate_count": 0,
        }
    return {
        "schema_version": 4,
        "artifact_kind": "p7_general_anchor_prompt_only_audit_v4",
        "record_count": len(rows),
        "supervision_field_count": 0,
        "sources": dict(sorted(sources.items())),
        "source_licences": dict(sorted(licences.items())),
        "source_splits": dict(sorted(splits.items())),
        "unique_sample_ids": len(sample_ids),
        "unique_group_ids": len(group_ids),
        "unique_content_hashes": len(content_hashes),
        "overlap_authority": overlap,
        "final_records_opened": False,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }


def _uniform(seed: int, slot: int) -> float:
    digest = hashlib.sha256(f"p7-ca-tape:{seed}:{slot}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


class ActionKLSafetyV4:
    """Damp-only per-action reverse-KL EMA with transactional preview."""

    def __init__(
        self, *, kappa: Mapping[str, float], rho: float, eps: float = 1.0e-6
    ) -> None:
        if not (
            set(kappa) == set(ACTIONS)
            and all(float(value) > 0.0 for value in kappa.values())
            and 0.0 <= float(rho) < 1.0
            and float(eps) > 0.0
        ):
            raise P7Stage120Error("P7 action KL safety configuration differs")
        self.kappa = {str(key): float(value) for key, value in kappa.items()}
        self.rho = float(rho)
        self.eps = float(eps)
        self.ema = {action: 0.0 for action in ACTIONS}
        self.seen = {action: False for action in ACTIONS}
        self.trigger_count = {action: 0 for action in ACTIONS}
        self._pending: dict[str, Any] | None = None

    def preview(self, *, action: str, reverse_kl: float) -> dict[str, Any]:
        if self._pending is not None or action not in ACTIONS:
            raise P7Stage120Error("P7 action KL preview boundary differs")
        value = float(reverse_kl)
        if not math.isfinite(value):
            raise P7Stage120Error("P7 action reverse KL is non-finite")
        next_ema = (
            value
            if not self.seen[action]
            else self.rho * self.ema[action] + (1.0 - self.rho) * value
        )
        scale = min(1.0, self.kappa[action] / (abs(next_ema) + self.eps))
        self._pending = {
            "action": action,
            "reverse_kl": value,
            "next_ema": next_ema,
            "scale": scale,
            "triggered": scale < 1.0,
        }
        return deepcopy(self._pending)

    def accept_pending(self) -> dict[str, Any]:
        if self._pending is None:
            raise P7Stage120Error("P7 action KL acceptance has no preview")
        value = deepcopy(self._pending)
        action = str(value["action"])
        self.ema[action] = float(value["next_ema"])
        self.seen[action] = True
        if value["triggered"] is True:
            self.trigger_count[action] += 1
        self._pending = None
        return value

    def reject_pending(self) -> dict[str, Any]:
        if self._pending is None:
            raise P7Stage120Error("P7 action KL rejection has no preview")
        value = deepcopy(self._pending)
        self._pending = None
        return value

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise P7Stage120Error("P7 action KL state cannot checkpoint a pending preview")
        return {
            "schema_version": 4,
            "kappa": dict(self.kappa),
            "rho": self.rho,
            "eps": self.eps,
            "ema": dict(self.ema),
            "seen": dict(self.seen),
            "trigger_count": dict(self.trigger_count),
            "pending": None,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not (
            value.get("schema_version") == 4
            and dict(value.get("kappa", {})) == self.kappa
            and float(value.get("rho", -1.0)) == self.rho
            and float(value.get("eps", -1.0)) == self.eps
            and value.get("pending") is None
        ):
            raise P7Stage120Error("P7 action KL resume identity differs")
        ema = {str(key): float(item) for key, item in dict(value.get("ema", {})).items()}
        seen = {str(key): bool(item) for key, item in dict(value.get("seen", {})).items()}
        triggers = {
            str(key): int(item)
            for key, item in dict(value.get("trigger_count", {})).items()
        }
        if not (
            set(ema) == set(seen) == set(triggers) == set(ACTIONS)
            and all(math.isfinite(item) for item in ema.values())
            and all(item >= 0 for item in triggers.values())
        ):
            raise P7Stage120Error("P7 action KL resume state differs")
        self.ema = ema
        self.seen = seen
        self.trigger_count = triggers
        self._pending = None


class Stage120RouteStateV4:
    """Deterministic accepted-slot action state with rejection-safe replay."""

    def __init__(self, *, method_id: str, seed: int, accepted_target: int = 120) -> None:
        if method_id not in METHODS or seed != 42 or accepted_target != 120:
            raise P7Stage120Error("P7 route identity differs")
        self.method_id = method_id
        self.seed = int(seed)
        self.accepted_target = int(accepted_target)
        self.random_tape = tuple(_uniform(self.seed, slot) for slot in range(accepted_target))
        self.accepted_steps = 0
        self.rejected_attempts = 0
        self.consecutive_rejections = 0
        self.action_counts = {action: 0 for action in ACTIONS}
        self._pending_action: str | None = None

    def action_for_slot(self, slot: int, *, p_medical: float = 0.5) -> str:
        if slot != self.accepted_steps or not 0 <= slot < self.accepted_target:
            raise P7Stage120Error("P7 action request escaped accepted cursor")
        if not math.isfinite(float(p_medical)) or not 0.0 <= float(p_medical) <= 1.0:
            raise P7Stage120Error("P7 route probability differs")
        action = (
            "medical"
            if self.method_id == "IDT-v2" and slot % 2 == 0
            else "general"
            if self.method_id == "IDT-v2"
            else "medical"
            if self.random_tape[slot] < float(p_medical)
            else "general"
        )
        if self._pending_action is not None and self._pending_action != action:
            raise P7Stage120Error("rejected action was redrawn")
        self._pending_action = action
        return action

    def accept(self, *, action: str) -> None:
        if action not in ACTIONS or self._pending_action != action:
            raise P7Stage120Error("accepted P7 action differs from frozen slot")
        self.action_counts[action] += 1
        self.accepted_steps += 1
        self.consecutive_rejections = 0
        self._pending_action = None

    def reject(self, *, action: str) -> None:
        if action not in ACTIONS or self._pending_action != action:
            raise P7Stage120Error("rejected P7 action differs from frozen slot")
        self.rejected_attempts += 1
        self.consecutive_rejections += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "method_id": self.method_id,
            "seed": self.seed,
            "accepted_target": self.accepted_target,
            "random_tape": list(self.random_tape),
            "accepted_steps": self.accepted_steps,
            "rejected_attempts": self.rejected_attempts,
            "consecutive_rejections": self.consecutive_rejections,
            "action_counts": dict(self.action_counts),
            "pending_action": self._pending_action,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if not (
            value.get("schema_version") == 4
            and value.get("method_id") == self.method_id
            and int(value.get("seed", -1)) == self.seed
            and int(value.get("accepted_target", -1)) == self.accepted_target
            and tuple(value.get("random_tape", ())) == self.random_tape
        ):
            raise P7Stage120Error("P7 route resume identity differs")
        accepted = int(value.get("accepted_steps", -1))
        rejected = int(value.get("rejected_attempts", -1))
        consecutive = int(value.get("consecutive_rejections", -1))
        counts = {str(key): int(item) for key, item in dict(value.get("action_counts", {})).items()}
        pending = value.get("pending_action")
        if not (
            0 <= accepted <= self.accepted_target
            and rejected >= 0
            and consecutive >= 0
            and set(counts) == set(ACTIONS)
            and sum(counts.values()) == accepted
            and (pending is None or pending in ACTIONS)
        ):
            raise P7Stage120Error("P7 route resume state differs")
        self.accepted_steps = accepted
        self.rejected_attempts = rejected
        self.consecutive_rejections = consecutive
        self.action_counts = counts
        self._pending_action = None if pending is None else str(pending)


__all__ = [
    "ACTIONS",
    "ActionKLSafetyV4",
    "GENERAL_SOURCES",
    "METHODS",
    "P7Stage120Error",
    "Stage120RouteStateV4",
    "audit_general_anchor_records_v4",
]
