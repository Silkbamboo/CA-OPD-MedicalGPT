"""CPU-safe bounded rejection and checkpoint metadata for P7 Stage-120."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


class P7TransactionError(RuntimeError):
    """A P7 attempt transaction or resume invariant failed closed."""


PROTECTED_KEYS = frozenset(
    {
        "lora_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "cpu_rng_sha256",
        "cuda_rng_sha256",
    }
)
CURSOR_KEYS = frozenset(
    {
        "accepted_steps",
        "data_cursor",
        "policy_version",
        "sampler_version",
        "scheduler_step",
        "controller_step",
        "action_occurrences",
    }
)
PACKAGE_KEYS = frozenset(
    {
        "package_sha256",
        "formula_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "health_sha256",
    }
)


def _is_sha(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_protected(value: Mapping[str, Any]) -> dict[str, str]:
    result = {str(key): str(item) for key, item in dict(value).items()}
    if set(result) != set(PROTECTED_KEYS) or not all(_is_sha(item) for item in result.values()):
        raise P7TransactionError("P7 protected state hashes differ")
    return result


def _validate_cursors(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if set(result) != set(CURSOR_KEYS):
        raise P7TransactionError("P7 transaction cursor fields differ")
    step = int(result["accepted_steps"])
    occurrences = {str(key): int(item) for key, item in dict(result["action_occurrences"]).items()}
    if not (
        0 <= step <= 120
        and int(result["data_cursor"]) == step * 4
        and int(result["policy_version"]) == step
        and int(result["sampler_version"]) == step
        and int(result["scheduler_step"]) == step
        and 0 <= int(result["controller_step"]) <= step
        and set(occurrences) == {"medical", "general"}
        and sum(occurrences.values()) == step
    ):
        raise P7TransactionError("P7 transaction cursor/version state differs")
    result["action_occurrences"] = occurrences
    return result


class Stage120TransactionStateV4:
    """Bounded, pre-commit attempt state; rejections never advance cursors."""

    def __init__(self) -> None:
        self.accepted_steps = 0
        self.rejected_attempts = 0
        self.consecutive_rejections = 0
        self.action_counts = {"medical": 0, "general": 0}
        self.pending_attempt: dict[str, Any] | None = None

    def begin_attempt(
        self,
        *,
        action: str,
        protected: Mapping[str, Any],
        cursors: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.pending_attempt is not None:
            raise P7TransactionError("P7 transaction attempt is already pending")
        if action not in {"medical", "general"}:
            raise P7TransactionError("P7 transaction action differs")
        cursor_state = _validate_cursors(cursors)
        if int(cursor_state["accepted_steps"]) != self.accepted_steps:
            raise P7TransactionError("P7 transaction accepted cursor differs")
        token = {
            "schema_version": 4,
            "attempt_ordinal": self.accepted_steps + self.rejected_attempts + 1,
            "accepted_slot": self.accepted_steps,
            "action": action,
            "reserve_variant": self.consecutive_rejections,
            "protected_before": _validate_protected(protected),
            "cursors_before": cursor_state,
        }
        if int(token["reserve_variant"]) > 3:
            raise P7TransactionError("P7 reserve schedule was exhausted")
        self.pending_attempt = deepcopy(token)
        return deepcopy(token)

    def reject_attempt(
        self,
        token: Mapping[str, Any],
        *,
        protected_after: Mapping[str, Any],
        cursors_after: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        if self.pending_attempt != dict(token) or not reason:
            raise P7TransactionError("P7 rejected attempt token/reason differs")
        protected = _validate_protected(protected_after)
        cursors = _validate_cursors(cursors_after)
        if protected != token["protected_before"]:
            raise P7TransactionError("P7 protected state changed on rejection")
        if cursors != token["cursors_before"]:
            raise P7TransactionError("P7 cursor/version state changed on rejection")
        self.rejected_attempts += 1
        self.consecutive_rejections += 1
        self.pending_attempt = None
        if self.consecutive_rejections > 2:
            raise P7TransactionError("P7 consecutive rejection maximum exceeded")
        if self.rejected_attempts > 3:
            raise P7TransactionError("P7 total rejection maximum exceeded")
        return {
            "schema_version": 4,
            "artifact_kind": "p7_stage120_rejected_attempt_v4",
            "attempt_ordinal": token["attempt_ordinal"],
            "accepted_slot": token["accepted_slot"],
            "action": token["action"],
            "reserve_variant_used": token["reserve_variant"],
            "reserve_variant_next": self.consecutive_rejections,
            "reason": reason,
            "counts_as_accepted_commit": False,
            "atomic_rollback_verified": True,
            "cursor_advanced": False,
            "optimizer_executed": False,
            "scheduler_executed": False,
            "sampler_refreshed": False,
        }

    def record_external_accepted_commit(
        self, *, action: str, cursors_after: Mapping[str, Any]
    ) -> None:
        """Record a GPU-owned commit after its candidate transaction succeeds."""

        if self.pending_attempt is not None or action not in {"medical", "general"}:
            raise P7TransactionError("P7 accepted transaction boundary differs")
        cursors = _validate_cursors(cursors_after)
        if int(cursors["accepted_steps"]) != self.accepted_steps + 1:
            raise P7TransactionError("P7 accepted cursor did not advance exactly once")
        self.accepted_steps += 1
        self.action_counts[action] += 1
        self.consecutive_rejections = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "accepted_steps": self.accepted_steps,
            "rejected_attempts": self.rejected_attempts,
            "consecutive_rejections": self.consecutive_rejections,
            "action_counts": dict(self.action_counts),
            "pending_attempt": deepcopy(self.pending_attempt),
            "total_rejection_max": 3,
            "consecutive_rejection_max": 2,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        counts = {str(key): int(item) for key, item in dict(value.get("action_counts", {})).items()}
        if not (
            value.get("schema_version") == 4
            and int(value.get("total_rejection_max", -1)) == 3
            and int(value.get("consecutive_rejection_max", -1)) == 2
            and set(counts) == {"medical", "general"}
        ):
            raise P7TransactionError("P7 transaction resume identity differs")
        accepted = int(value.get("accepted_steps", -1))
        rejected = int(value.get("rejected_attempts", -1))
        consecutive = int(value.get("consecutive_rejections", -1))
        pending = value.get("pending_attempt")
        if not (
            0 <= accepted <= 120
            and 0 <= rejected <= 3
            and 0 <= consecutive <= 2
            and sum(counts.values()) == accepted
            and (pending is None or isinstance(pending, Mapping))
        ):
            raise P7TransactionError("P7 transaction resume state differs")
        self.accepted_steps = accepted
        self.rejected_attempts = rejected
        self.consecutive_rejections = consecutive
        self.action_counts = counts
        self.pending_attempt = None if pending is None else deepcopy(dict(pending))


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_stage120_checkpoint_metadata_v4(
    *,
    method_id: str,
    logical_version: int,
    policy_version: int,
    sampler_version: int,
    data_cursor: int,
    scheduler_step: int,
    accepted_steps: int,
    rejected_attempts: int,
    route_state: Mapping[str, Any],
    controller_state: Mapping[str, Any] | None,
    action_occurrences: Mapping[str, int],
    rng_sha256: Mapping[str, str],
    package_identities: Mapping[str, str],
) -> dict[str, Any]:
    identities = {str(key): str(item) for key, item in dict(package_identities).items()}
    rng = {str(key): str(item) for key, item in dict(rng_sha256).items()}
    occurrences = {str(key): int(item) for key, item in dict(action_occurrences).items()}
    if not (
        method_id in {"IDT-v2", "CA-OPD-v2"}
        and logical_version in {30, 60, 90, 120}
        and logical_version == policy_version == sampler_version == scheduler_step == accepted_steps
        and data_cursor == accepted_steps * 4
        and 0 <= rejected_attempts <= 3
        and set(identities) == set(PACKAGE_KEYS)
        and all(_is_sha(item) for item in identities.values())
        and set(rng) == {"cpu", "cuda"}
        and all(_is_sha(item) for item in rng.values())
        and set(occurrences) == {"medical", "general"}
        and sum(occurrences.values()) == accepted_steps
        and int(route_state.get("accepted_steps", -1)) == accepted_steps
        and (method_id != "CA-OPD-v2" or isinstance(controller_state, Mapping))
    ):
        raise P7TransactionError("P7 checkpoint metadata inputs differ")
    value = {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_checkpoint_metadata_v4",
        "method_id": method_id,
        "logical_version": logical_version,
        "optimizer_step": accepted_steps,
        "policy_version": policy_version,
        "sampler_version": sampler_version,
        "scheduler_step": scheduler_step,
        "data_cursor": data_cursor,
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "action_occurrences": occurrences,
        "route_state": deepcopy(dict(route_state)),
        "controller_state": None if controller_state is None else deepcopy(dict(controller_state)),
        "rng_sha256": rng,
        "package_identities": identities,
        "complete": True,
        "resume_eligible": True,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }
    value["metadata_sha256"] = _canonical_sha(value)
    return value


def validate_stage120_checkpoint_metadata_v4(
    value: Mapping[str, Any],
    *,
    expected_method_id: str,
    expected_package_identities: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(value)
    digest = payload.pop("metadata_sha256", None)
    if not _is_sha(digest) or _canonical_sha(payload) != digest:
        raise P7TransactionError("P7 checkpoint metadata SHA differs")
    step = int(payload.get("accepted_steps", -1))
    if not (
        payload.get("schema_version") == 4
        and payload.get("artifact_kind") == "p7_stage120_checkpoint_metadata_v4"
        and payload.get("method_id") == expected_method_id
        and payload.get("complete") is True
        and payload.get("resume_eligible") is True
        and int(payload.get("logical_version", -1))
        == int(payload.get("optimizer_step", -1))
        == int(payload.get("policy_version", -1))
        == int(payload.get("sampler_version", -1))
        == int(payload.get("scheduler_step", -1))
        == step
        and int(payload.get("data_cursor", -1)) == step * 4
        and payload.get("package_identities") == dict(expected_package_identities)
        and payload.get("final_access_count") == 0
        and payload.get("confirmation_access_count") == 0
    ):
        raise P7TransactionError("P7 checkpoint cursor/version/package identity differs")
    return {
        "passed": True,
        "method_id": expected_method_id,
        "logical_version": step,
        "metadata_sha256": digest,
        "resume_eligible": True,
    }


__all__ = [
    "P7TransactionError",
    "Stage120TransactionStateV4",
    "build_stage120_checkpoint_metadata_v4",
    "validate_stage120_checkpoint_metadata_v4",
]
