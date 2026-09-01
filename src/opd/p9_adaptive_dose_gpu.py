"""P9 wrapper around the unchanged P7 Formal B2 v2 GPU session."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError
from src.opd.p9_runtime import p9_optimizer_step_limit
from src.opd.production_b2_formal_gpu_v2 import FormalB2SessionV2
from src.opd.production_b2_formal_gpu_v2 import DiagnosticCandidateRollbackV2
import src.opd.production_qualification_two_step_gpu_v7 as p7_kernel
from src.opd.production_b2_transaction_v2 import (
    ordered_trainable_sha256,
    state_tree_sha256,
)
from src.opd.production_b2_memory_execution_v1 import MemoryTelemetryWriterV1
import src.opd.production_b2_memory_execution_gpu_v1 as memory_gpu


class P9RejectedAttempt(RuntimeError):
    """A batch was rejected after a verified full pre-generation rollback."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__(str(evidence["reason"]))
        self.evidence = dict(evidence)


class P9EngineeringAttemptError(RuntimeError):
    """An engineering/protocol failure rolled back but must not consume a reserve."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__(str(evidence["reason"]))
        self.evidence = dict(evidence)


_SCIENTIFIC_REJECTION_PREFIXES = (
    "preupdate_backend_health_v2_rejected:",
    "legacy_backend_correction_gate_rejected",
    "precommit_gradient_health_v2_rejected:",
    "ratio_health_v2_rejected:",
)


def is_scientific_rejection_artifact(path: Any) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    reason = str(value.get("reason", ""))
    return bool(
        value.get("artifact_kind") == "formal_b2_rejected_update_v2"
        and value.get("counts_as_optimizer_commit") is False
        and any(reason.startswith(prefix) for prefix in _SCIENTIFIC_REJECTION_PREFIXES)
    )


def is_generation_health_rejection_artifact(path: Any) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    samples = value.get("prompt_samples")
    return bool(
        value.get("artifact_kind") == "b2_generation_health_failure_v1"
        and value.get("optimizer_executed") is False
        and isinstance(samples, list)
        and samples
        and any(
            any(
                bool(sample.get(field))
                for field in (
                    "invalid", "empty", "non_finite",
                    "unexpected_think_tag", "repetition",
                )
            )
            for sample in samples
        )
    )


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed_artifact_paths(
    before: Mapping[Path, str], current: list[Path]
) -> list[Path]:
    """Detect both newly created artifacts and atomic replacement at fixed paths."""

    return sorted(
        path for path in current if before.get(path) != _artifact_sha256(path)
    )


def ratio_evidence_sha256_from_rejections(paths: list[Path]) -> str | None:
    """Bind diagnostic ratio evidence after the kernel clears its live field."""

    evidence: list[Mapping[str, Any]] = []
    for path in paths:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise P9ProtocolError("P9 kernel rejection evidence is unreadable") from error
        ratio = value.get("ratio_evidence")
        if isinstance(ratio, Mapping):
            evidence.append(ratio)
    if not evidence:
        return None
    if len(evidence) != 1:
        raise P9ProtocolError("P9 kernel rejection ratio evidence is ambiguous")
    return hashlib.sha256(
        json.dumps(
            evidence[0], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _ratio_evidence_from_rejections(paths: list[Path]) -> Mapping[str, Any] | None:
    found = []
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(value.get("ratio_evidence"), Mapping):
            found.append(value["ratio_evidence"])
    if not found:
        return None
    if len(found) != 1:
        raise P9ProtocolError("P9 diagnostic ratio evidence is ambiguous")
    return found[0]


def build_p9_qualification_observation(
    *,
    private: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    rollout: Mapping[str, Any] | None,
    kernel_rejection_artifacts: list[Path],
    ratio_health: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    telemetry = private if isinstance(private, Mapping) else candidate
    if not isinstance(telemetry, Mapping) or not isinstance(rollout, Mapping):
        return None
    response_ids = [
        list(row.get("response_ids", [])) for row in rollout.get("rows", [])
    ]
    ratio_evidence = _ratio_evidence_from_rejections(kernel_rejection_artifacts)
    backend_ess = (
        ratio_evidence.get("backend_correction", {}).get("ess", {}).get(
            "pooled_fraction"
        )
        if isinstance(ratio_evidence, Mapping)
        else None
    )
    ess_fraction = telemetry.get("ess_fraction", backend_ess)
    return {
        "sample_ids": [str(row.get("fixture_id")) for row in rollout.get("rows", [])],
        "completion_token_counts": [len(row) for row in response_ids],
        "completion_token_sha256": hashlib.sha256(
            json.dumps(response_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "student_score": copy.deepcopy(telemetry.get("q_logprob")),
        "teacher_score": copy.deepcopy(telemetry.get("teacher_logprob")),
        "loss": telemetry.get("loss"),
        "objective": telemetry.get("objective"),
        "reverse_kl": copy.deepcopy(telemetry.get("reverse_kl")),
        "advantage": copy.deepcopy(telemetry.get("advantage")),
        "ess_fraction": ess_fraction,
        "ratio": copy.deepcopy(
            ratio_evidence.get("ppo_ratio")
            if isinstance(ratio_evidence, Mapping)
            else None
        ),
        "ratio_health": copy.deepcopy(ratio_health),
        "ratio_evidence_sha256": ratio_evidence_sha256_from_rejections(
            kernel_rejection_artifacts
        ),
    }


class P9MemoryTelemetryWriter(MemoryTelemetryWriterV1):
    """Continue the existing authoritative marker sequence after a legal resume."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._p9_candidate_observation: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)
        existing = sorted((self.root / "markers").glob("*.json"))
        if existing:
            sequences = []
            for path in existing:
                value = json.loads(path.read_text(encoding="utf-8"))
                if int(path.stem) != int(value.get("sequence", -1)):
                    raise P9ProtocolError("P9 memory telemetry sequence identity differs")
                sequences.append(int(value["sequence"]))
            if sequences != list(range(min(sequences), max(sequences) + 1)):
                raise P9ProtocolError("P9 memory telemetry sequence is not contiguous")
            self._sequence = max(sequences) + 1


class P9FormalB2Session(FormalB2SessionV2):
    """Exact P7 2:2 semantics plus whole-attempt rejection rollback evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        original_writer = memory_gpu.MemoryTelemetryWriterV1
        memory_gpu.MemoryTelemetryWriterV1 = P9MemoryTelemetryWriter
        try:
            super().__init__(*args, **kwargs)
        finally:
            memory_gpu.MemoryTelemetryWriterV1 = original_writer

    def _p9_distribution(self, values: Any, mask: Any) -> dict[str, float]:
        """Add P9-only telemetry without changing the P7 record schema hook."""

        result = super()._masked_distribution(values, mask)
        selected = values.detach().float()[mask.to(dtype=self.torch.bool)]
        result["positive_fraction"] = float((selected > 0).float().mean().cpu())
        return result

    def _validate_candidate_update_v2(self, **kwargs: Any) -> None:
        """Capture read-only candidate telemetry before diagnostic rollback."""

        bundle = kwargs["bundle"]
        before_result = kwargs["before_result"]
        gate = kwargs["legacy_candidate_gate_evidence"]
        self._p9_candidate_observation = {
            "q_logprob": self._p9_distribution(
                bundle.current_actor_logprob, bundle.response_mask
            ),
            "teacher_logprob": self._p9_distribution(
                bundle.teacher_logprob, bundle.response_mask
            ),
            "reverse_kl": self._p9_distribution(
                bundle.old_actor_logprob - bundle.teacher_logprob,
                bundle.response_mask,
            ),
            "advantage": {
                **self._p9_distribution(
                    before_result.advantage, bundle.response_mask
                ),
                "clip_fraction": 0.0,
            },
            "loss": float(gate["loss_after"]),
            "objective": float(gate["objective_after"]),
        }
        return super()._validate_candidate_update_v2(**kwargs)

    def _p9_attempt_snapshot(self, *, step_index: int) -> dict[str, Any]:
        state = self._transaction_state_v2
        if not (
            state is not None
            and state.accepted_optimizer_steps == step_index
            and state.data_cursor == step_index * 4
            and state.policy_version == state.sampler_version == state.refresh_version == step_index
            and self.current_sampler_version == step_index
        ):
            raise P9ProtocolError("P9 pre-generation cursor/version differs")
        cpu_rng = self.torch.get_rng_state().clone()
        cuda_rng = [value.cpu().clone() for value in self.torch.cuda.get_rng_state_all()]
        return {
            "accepted_optimizer_steps": step_index,
            "data_cursor": state.data_cursor,
            "policy_version": state.policy_version,
            "sampler_version": state.sampler_version,
            "refresh_version": state.refresh_version,
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng": cpu_rng,
            "cuda_rng": cuda_rng,
            "cpu_rng_sha256": state_tree_sha256(cpu_rng),
            "cuda_rng_sha256": state_tree_sha256(cuda_rng),
        }

    def _p9_rollback_failed_attempt(
        self,
        *,
        snapshot: Mapping[str, Any],
        reason: BaseException,
        kernel_rejection_artifacts: list[Path],
    ) -> dict[str, Any]:
        if self._pending_transaction_v2 is not None:
            self._abort_candidate_transaction_v2(reason="p9_attempt_exception")
        self.torch.set_rng_state(snapshot["cpu_rng"])
        self.torch.cuda.set_rng_state_all(list(snapshot["cuda_rng"]))
        state = self._transaction_state_v2
        after = {
            "lora_sha256": ordered_trainable_sha256(self.student_model),
            "optimizer_sha256": state_tree_sha256(self.optimizer.state_dict()),
            "scheduler_sha256": state_tree_sha256(self.scheduler.state_dict()),
            "cpu_rng_sha256": state_tree_sha256(self.torch.get_rng_state()),
            "cuda_rng_sha256": state_tree_sha256(self.torch.cuda.get_rng_state_all()),
        }
        expected = {key: snapshot[key] for key in after}
        if not (
            after == expected
            and state is not None
            and state.accepted_optimizer_steps == snapshot["accepted_optimizer_steps"]
            and state.data_cursor == snapshot["data_cursor"]
            and state.policy_version == snapshot["policy_version"]
            and state.sampler_version == snapshot["sampler_version"]
            and state.refresh_version == snapshot["refresh_version"]
            and self.current_sampler_version == snapshot["sampler_version"]
        ):
            raise P9ProtocolError("P9 rejected attempt rollback could not be proven") from reason
        qualification = build_p9_qualification_observation(
            private=self._last_b2_step_private,
            candidate=self._p9_candidate_observation,
            rollout=self._last_fixed_rollout_v2,
            kernel_rejection_artifacts=kernel_rejection_artifacts,
            ratio_health=self._last_ratio_health_v2,
        )
        return {
            "schema_version": 1,
            "artifact_kind": "p9_rejected_attempt_rollback",
            "attempted_optimizer_step": int(snapshot["accepted_optimizer_steps"]) + 1,
            "accepted_optimizer_steps": int(snapshot["accepted_optimizer_steps"]),
            "reason_type": type(reason).__name__,
            "reason": str(reason),
            "protected_before": expected,
            "protected_after": after,
            "adapter_rollback_verified": True,
            "optimizer_rollback_verified": True,
            "scheduler_rollback_verified": True,
            "rng_rollback_verified": True,
            "cursor_advanced": False,
            "sampler_advanced": False,
            "counts_as_optimizer_commit": False,
            "qualification_observations": qualification,
            "final_access_count": 0,
        }

    def run_p9_attempt(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        snapshot = self._p9_attempt_snapshot(step_index=step_index)
        rejection_root = self.output / "rejected_updates_v2"
        rejection_before = set(rejection_root.glob("attempt_*.json"))
        generation_root = self.output / "steps"
        generation_before = {
            path: _artifact_sha256(path)
            for path in generation_root.glob("generation_health_failure_step_*.json")
        }
        p9_optimizer_step_limit(self.config)
        original_step_limit = p7_kernel._b2_optimizer_step_limit
        p7_kernel._b2_optimizer_step_limit = p9_optimizer_step_limit
        try:
            record = dict(
                super().run_formal_step_v2(
                    step_index=step_index,
                    prompt_rows=prompt_rows,
                    max_new_tokens=max_new_tokens,
                )
            )
        except Exception as error:
            rejection_after = set(rejection_root.glob("attempt_*.json"))
            new_rejections = sorted(rejection_after - rejection_before)
            new_generation_failures = changed_artifact_paths(
                generation_before,
                list(generation_root.glob("generation_health_failure_step_*.json")),
            )
            evidence = self._p9_rollback_failed_attempt(
                snapshot=snapshot,
                reason=error,
                kernel_rejection_artifacts=new_rejections,
            )
            scientific = (
                isinstance(error, DiagnosticCandidateRollbackV2)
                or bool(new_rejections or new_generation_failures)
                and all(is_scientific_rejection_artifact(path) for path in new_rejections)
                and all(
                    is_generation_health_rejection_artifact(path)
                    for path in new_generation_failures
                )
            )
            evidence["failure_classification"] = (
                "scientific_health_rejection" if scientific else "engineering_or_protocol_failure"
            )
            evidence["kernel_rejection_artifacts"] = [str(path) for path in new_rejections]
            evidence["generation_health_artifacts"] = [
                str(path) for path in new_generation_failures
            ]
            evidence["generation_health_evidence"] = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in new_generation_failures
            ]
            if scientific:
                raise P9RejectedAttempt(evidence) from error
            raise P9EngineeringAttemptError(evidence) from error
        finally:
            p7_kernel._b2_optimizer_step_limit = original_step_limit
        record["p9"] = {
            "schema_version": 1,
            "single_training_semantic_variable": "accepted_optimizer_commit_dose",
            "source_batch": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
            "absolute_max_step": 300,
            "final_access_count": 0,
            "advantage_positive_fraction": (
                self._p9_candidate_observation.get("advantage", {}).get(
                    "positive_fraction"
                )
                if isinstance(self._p9_candidate_observation, Mapping)
                else None
            ),
        }
        rollout = self._last_fixed_rollout_v2
        if not isinstance(rollout, Mapping):
            raise P9ProtocolError("P9 completion-token evidence is absent")
        response_ids = [list(row.get("response_ids", [])) for row in rollout.get("rows", [])]
        record["p9"]["completion_token_counts"] = [len(row) for row in response_ids]
        record["p9"]["completion_token_sha256"] = hashlib.sha256(
            json.dumps(response_ids, separators=(",", ":")).encode()
        ).hexdigest()
        record["p9"]["raw_completion_tokens_persisted"] = False
        return record


__all__ = [
    "P9EngineeringAttemptError", "P9FormalB2Session", "P9RejectedAttempt",
    "P9MemoryTelemetryWriter", "is_scientific_rejection_artifact",
    "is_generation_health_rejection_artifact",
    "changed_artifact_paths",
    "ratio_evidence_sha256_from_rejections",
    "build_p9_qualification_observation",
]
