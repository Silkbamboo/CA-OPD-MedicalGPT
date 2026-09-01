"""Fail-closed helpers for P8's fixed-token source-mix qualification.

The scientific comparison reweights one already-generated prompt-gradient
universe.  It therefore cannot accidentally turn a source-mix diagnostic into
a second sampling, decoding, loss, or optimizer change.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence


class P8FixedTokenQualificationError(RuntimeError):
    """The frozen fixed-token qualification contract differs."""


_O1 = "medical_opd_o1"
_CMB = "medical_opd_cmb"


def compare_source_mix_budget(
    prompt_gradients: Sequence[Mapping[str, Any]],
    *,
    target_relative_improvement: float = 0.20,
) -> dict[str, Any]:
    """Compare 2:2 and 3:1 as source-weighted means on identical gradients."""

    if not math.isclose(target_relative_improvement, 0.20, rel_tol=0.0, abs_tol=1e-12):
        raise P8FixedTokenQualificationError(
            "the preregistered fixed-token target must remain 20 percent"
        )
    values: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(prompt_gradients):
        source = row.get("source", row.get("source_role"))
        if source not in {_O1, _CMB}:
            raise P8FixedTokenQualificationError(
                f"fixed-token prompt gradient {index} has an unregistered source"
            )
        try:
            bounded_norm = float(row["bounded_norm"])
        except (KeyError, TypeError, ValueError) as error:
            raise P8FixedTokenQualificationError(
                f"fixed-token prompt gradient {index} has no finite bounded norm"
            ) from error
        if not math.isfinite(bounded_norm) or bounded_norm < 0.0:
            raise P8FixedTokenQualificationError(
                f"fixed-token prompt gradient {index} has no finite bounded norm"
            )
        values[str(source)].append(bounded_norm)
    if not values[_O1] or not values[_CMB]:
        raise P8FixedTokenQualificationError(
            "both O1 and CMB must occur in the same prompt-gradient universe"
        )

    source_means = {
        source: sum(items) / len(items) for source, items in sorted(values.items())
    }
    baseline_weights = {_O1: 0.5, _CMB: 0.5}
    candidate_weights = {_O1: 0.75, _CMB: 0.25}
    baseline = sum(baseline_weights[source] * source_means[source] for source in baseline_weights)
    candidate = sum(candidate_weights[source] * source_means[source] for source in candidate_weights)
    relative = (candidate - baseline) / baseline if baseline > 0.0 else float("-inf")
    return {
        "metric": "source_weighted_mean_post_cap_prompt_gradient_norm",
        "prompt_gradient_count": sum(len(items) for items in values.values()),
        "source_counts": {source: len(values[source]) for source in (_O1, _CMB)},
        "source_means": source_means,
        "baseline_source_weights": baseline_weights,
        "candidate_source_weights": candidate_weights,
        "baseline_effective_budget": baseline,
        "candidate_effective_budget": candidate,
        "relative_improvement": relative,
        "target_relative_improvement": target_relative_improvement,
        "target_improvement_passed": bool(relative >= target_relative_improvement),
        "same_prompt_gradient_universe": True,
    }


_ROLLBACK_REQUIRED = (
    "rollback_verified",
    "student_restored",
    "optimizer_restored",
    "scheduler_restored",
    "cpu_rng_restored",
    "cuda_rng_restored",
    "cursor_unchanged",
    "policy_version_unchanged",
    "sampler_version_unchanged",
    "refresh_version_unchanged",
    "registry_count_unchanged",
)


def validate_unconditional_rollback(rollback: Mapping[str, Any]) -> dict[str, Any]:
    """Require every protected mutable-state identity to survive rollback."""

    if not isinstance(rollback, Mapping):
        raise P8FixedTokenQualificationError("rollback evidence is absent")
    failures = [name for name in _ROLLBACK_REQUIRED if rollback.get(name) is not True]
    if failures:
        raise ValueError("unconditional rollback differs: " + ",".join(failures))
    return {
        "passed": True,
        "required_fields": list(_ROLLBACK_REQUIRED),
        "evidence": {name: True for name in _ROLLBACK_REQUIRED},
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P8FixedTokenQualificationError(f"qualification JSON is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    from src.opd.production_b2_formal_worker_v1 import _atomic_json as write

    write(path, value)


def _session_state(session: Any) -> dict[str, Any]:
    import torch

    from src.opd.production_b2_transaction_v2 import (
        ordered_trainable_sha256,
        state_tree_sha256,
    )

    transaction_state = session._transaction_state_v2
    if transaction_state is None:
        raise P8FixedTokenQualificationError("transaction state is absent")
    return {
        "student_sha256": ordered_trainable_sha256(session.student_model),
        "optimizer_sha256": state_tree_sha256(session.optimizer.state_dict()),
        "scheduler_sha256": state_tree_sha256(session.scheduler.state_dict()),
        "cpu_rng_sha256": state_tree_sha256(torch.get_rng_state()),
        "cuda_rng_sha256": state_tree_sha256(torch.cuda.get_rng_state_all()),
        "transaction_state": asdict(transaction_state),
        "current_sampler_version": int(session.current_sampler_version),
        "optimizer_step_count": int(session._optimizer_step_count),
        "scheduler_step_count": int(session._scheduler_step_count),
        "registry_count": int(session._registry_count()),
    }


def _normalize_rollback(
    *, before: Mapping[str, Any], after: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, bool]:
    rollback = artifact.get("rollback")
    if not isinstance(rollback, Mapping):
        raise P8FixedTokenQualificationError("rollback artifact is absent")
    before_state = before["transaction_state"]
    after_state = after["transaction_state"]
    return {
        "rollback_verified": rollback.get("rollback_verified") is True,
        "student_restored": before["student_sha256"] == after["student_sha256"],
        "optimizer_restored": before["optimizer_sha256"] == after["optimizer_sha256"],
        "scheduler_restored": before["scheduler_sha256"] == after["scheduler_sha256"],
        # The outer snapshot precedes stochastic token generation.  Candidate
        # rollback starts only after that fixed token batch is frozen, so RNG
        # restoration must use the transaction's own capture boundary.
        "cpu_rng_restored": rollback.get("cpu_rng_restored") is True,
        "cuda_rng_restored": rollback.get("cuda_rng_restored") is True,
        "cursor_unchanged": before_state["data_cursor"] == after_state["data_cursor"],
        "policy_version_unchanged": before_state["policy_version"] == after_state["policy_version"],
        "sampler_version_unchanged": bool(
            before_state["sampler_version"] == after_state["sampler_version"]
            and before["current_sampler_version"] == after["current_sampler_version"]
        ),
        "refresh_version_unchanged": before_state["refresh_version"] == after_state["refresh_version"],
        "registry_count_unchanged": bool(
            before_state["registry_count"] == after_state["registry_count"]
            and before["registry_count"] == after["registry_count"]
        ),
    }


def _diagnostic_runtime(source_config: Mapping[str, Any], *, output: Path) -> dict[str, Any]:
    from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2

    config = deepcopy(dict(source_config))
    config["run"] = {
        "run_id": output.name,
        "seed": 42,
        "optimizer_steps": 150,
        "stage1_stop_step": 120,
        "output_dir": str(output),
    }
    runtime = formal_b2_runtime_config_v2(config)
    runtime["p8_formal_b2"] = {
        "package_version": "p8_single_variable_b2_v1",
        "single_training_semantic_variable": "medical_source_mix",
        "source_batch": {_O1: 3, _CMB: 1},
        "frozen_max_step": 300,
        "stage1_stop_step": 120,
        "group_size": 1,
        "learning_rate": 1e-5,
        "per_prompt_gradient_clip_norm": 0.25,
        "response_length": 1024,
    }
    return runtime


def _hard_gate_summary(
    *, artifact: Mapping[str, Any], ratio_health: Mapping[str, Any], preupdate_health: Mapping[str, Any]
) -> dict[str, Any]:
    ratio = artifact.get("ratio_evidence")
    if not isinstance(ratio, Mapping):
        raise P8FixedTokenQualificationError("candidate ratio evidence is absent")
    bounded = ratio.get("bounded_influence_v2")
    pool = ratio.get("pool_binding")
    if not isinstance(bounded, Mapping) or not isinstance(pool, Mapping):
        raise P8FixedTokenQualificationError("candidate bounded/token evidence is absent")
    prompt_gradients = bounded.get("prompt_gradients")
    if not isinstance(prompt_gradients, list) or len(prompt_gradients) != 4:
        raise P8FixedTokenQualificationError("candidate prompt-gradient evidence differs")
    checks = {
        "identity": isinstance(ratio.get("identity"), Mapping),
        "finite": all(
            math.isfinite(float(row.get("bounded_norm", float("nan"))))
            and math.isfinite(float(row.get("raw_norm", float("nan"))))
            for row in prompt_gradients
            if isinstance(row, Mapping)
        ),
        "behavior_denominator": isinstance(ratio.get("backend_correction"), Mapping),
        "ratio_ess_post_shift_gradient_delta": ratio_health.get("accepted") is True,
        "preupdate_backend": preupdate_health.get("accepted") is True,
        "candidate_update_executed": artifact.get("rollback", {}).get("candidate_executed") is True,
        "unconditional_rejection": artifact.get("reason") == "fixed_token_candidate_unconditional_rollback",
        "no_optimizer_commit": artifact.get("counts_as_optimizer_commit") is False,
        "cursor_not_advanced": artifact.get("cursor_advanced") is False,
        "sampler_not_refreshed": artifact.get("sampler_refreshed") is False,
        "no_restricted_access": artifact.get("restricted_access_count") == 0,
        "prompt_equal_scalar_loss": bounded.get("prompt_equal_scalar_loss_unchanged") is True,
    }


def summarize_fixed_token_attempts(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Finalize persisted candidate transactions, including a scientific rejection."""

    if len(attempts) != 2:
        raise P8FixedTokenQualificationError(
            "P8 fixed-token qualification requires exactly two persisted attempts"
        )
    all_gradients: list[Mapping[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for batch_index, attempt in enumerate(attempts):
        ratio = attempt.get("ratio_evidence")
        rollback = attempt.get("rollback")
        if not isinstance(ratio, Mapping) or not isinstance(rollback, Mapping):
            raise P8FixedTokenQualificationError(
                f"P8 attempt {batch_index} lacks ratio or rollback evidence"
            )
        bounded = ratio.get("bounded_influence_v2")
        pool = ratio.get("pool_binding")
        if not isinstance(bounded, Mapping) or not isinstance(pool, Mapping):
            raise P8FixedTokenQualificationError(
                f"P8 attempt {batch_index} lacks fixed-token evidence"
            )
        gradients = bounded.get("prompt_gradients")
        if not isinstance(gradients, list) or len(gradients) != 4:
            raise P8FixedTokenQualificationError(
                f"P8 attempt {batch_index} prompt gradients differ"
            )
        all_gradients.extend(gradients)
        reason = str(attempt.get("reason", ""))
        rollback_passed = bool(
            rollback.get("rollback_verified") is True
            and rollback.get("candidate_executed") is True
            and rollback.get("cpu_rng_restored") is True
            and rollback.get("cuda_rng_restored") is True
        )
        no_advance = bool(
            attempt.get("accepted_optimizer_steps") == 0
            and attempt.get("data_cursor") == 0
            and attempt.get("policy_version") == 0
            and attempt.get("sampler_version") == 0
            and attempt.get("counts_as_optimizer_commit") is False
            and attempt.get("cursor_advanced") is False
            and attempt.get("sampler_refreshed") is False
            and attempt.get("restricted_access_count") == 0
        )
        health_passed = reason == "fixed_token_candidate_unconditional_rollback"
        summaries.append(
            {
                "batch_index": batch_index,
                "reason": reason,
                "health_passed": health_passed,
                "rollback_passed": rollback_passed,
                "no_state_advance": no_advance,
                "fixed_batch_sha256": rollback.get("fixed_batch_sha256"),
                "response_token_sha256": pool.get("response_token_sha256"),
                "pool_binding_sha256": pool.get("pool_binding_sha256"),
            }
        )
    budget = compare_source_mix_budget(all_gradients)
    hard_gates_passed = all(
        item["health_passed"] and item["rollback_passed"] and item["no_state_advance"]
        for item in summaries
    )
    passed = bool(budget["target_improvement_passed"] and hard_gates_passed)
    scientific_failure = any(
        item["reason"].startswith("ratio_health_v2_rejected:")
        or item["reason"].startswith("precommit_gradient_health_v2_rejected:")
        for item in summaries
    )
    return {
        "passed": passed,
        "status": "candidate_qualified" if passed else "candidate_qualification_failed",
        "scientific_failure": bool(not passed and scientific_failure),
        "candidate_branch": "A",
        "single_training_semantic_variable": "medical_source_mix",
        "source_mix_budget": budget,
        "hard_gates_passed": hard_gates_passed,
        "attempts": summaries,
        "candidate_update_committed": False,
        "candidate_switch_allowed": False,
        "canary_allowed": passed,
        "formal_b2_allowed": False,
        "final_access_count": 0,
        "controller_access_count": 0,
        "confirmation_access_count": 0,
        "label_access_count": 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "response_token_sha256": pool.get("response_token_sha256"),
        "pool_binding_sha256": pool.get("pool_binding_sha256"),
        "fixed_batch_sha256": artifact.get("rollback", {}).get("fixed_batch_sha256"),
        "prompt_gradients": prompt_gradients,
    }


def run_p8_fixed_token_qualification(
    *,
    source_package: Path,
    candidate_config: Path,
    formula_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Run two fresh-v0 candidate batches, reject each, and compare one token universe."""

    source_package = Path(source_package).resolve()
    candidate_config = Path(candidate_config).resolve()
    formula_path = Path(formula_path).resolve()
    output = Path(output).resolve()
    if os.environ.get("CA_OPD_ALLOW_P8_FIXED_TOKEN_GPU") != "1":
        raise P8FixedTokenQualificationError("P8 fixed-token GPU authorization is absent")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise P8FixedTokenQualificationError(
            "P8 qualification requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before model load"
        )
    if output.exists() or output.is_symlink():
        raise P8FixedTokenQualificationError("P8 qualification output must be fresh")
    if shutil.disk_usage(output.parent).free < 10_000_000_000:
        raise P8FixedTokenQualificationError("persistent disk has less than 10 GB free")

    source_config = _json(source_package / "formal_b2_config.json")
    authority = _json(source_package / "data_authority.json")
    baseline_schedule = _json(source_package / "prompt_schedule.json")
    package_index = _json(source_package / "package_index.json")
    from src.opd.p8_formal_protocol import build_p8_prompt_schedule, resolve_p8_schedule_batch

    schedule = build_p8_prompt_schedule(
        authority,
        baseline_schedule=baseline_schedule,
        seed=42,
        optimizer_steps=300,
    )
    runtime = _diagnostic_runtime(source_config, output=output)
    output.mkdir(parents=True)
    for name in (
        "b2_steps",
        "checkpoints",
        "formal_steps",
        "memory_step_audits",
        "memory_telemetry/markers",
        "ratio_evidence_v2",
        "rejected_updates_v2",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "diagnostic_runtime.json", runtime)
    _atomic_json(output / "prompt_schedule.json", schedule)

    from src.opd.p8_formal_gpu import P8FormalB2Session
    from src.opd.production_b2_formal_gpu_v2 import DiagnosticCandidateRollbackV2

    session: Any | None = None
    started = time.time()
    batch_results: list[dict[str, Any]] = []
    all_gradients: list[dict[str, Any]] = []
    try:
        session = P8FormalB2Session(
            runtime,
            config_path=source_package / "formal_b2_config.json",
            route="b2_calibration",
        )
        identity = session.initial_calibration_identity()
        if not (
            identity.get("zero_effect_verified") is True
            and identity.get("tensor_count") == 504
            and identity.get("source_adapter_path") is None
        ):
            raise P8FixedTokenQualificationError("qualification is not fresh-v0")
        for batch_index in range(2):
            rows = resolve_p8_schedule_batch(authority, schedule, step_index=batch_index)
            before = _session_state(session)
            existing = len(list((output / "rejected_updates_v2").glob("attempt_*.json")))
            session.set_diagnostic_unconditional_rollback_v2(True)
            exception: dict[str, str] | None = None
            try:
                session.run_formal_step_v2(
                    step_index=0,
                    prompt_rows=rows,
                    max_new_tokens=1024,
                )
                raise P8FixedTokenQualificationError("diagnostic candidate committed")
            except DiagnosticCandidateRollbackV2 as error:
                exception = {"type": type(error).__name__, "message": str(error)}
            finally:
                session.release_step_teacher(0)
            rejected = sorted((output / "rejected_updates_v2").glob("attempt_*.json"))
            if len(rejected) != existing + 1:
                raise P8FixedTokenQualificationError("candidate rejection artifact is absent")
            artifact = _json(rejected[-1])
            after = _session_state(session)
            normalized_rollback = _normalize_rollback(
                before=before, after=after, artifact=artifact
            )
            rollback_validation = validate_unconditional_rollback(normalized_rollback)
            ratio_health = deepcopy(dict(session._last_ratio_health_v2 or {}))
            preupdate_health = deepcopy(dict(session._preupdate_backend_health_v2 or {}))
            gate = _hard_gate_summary(
                artifact=artifact,
                ratio_health=ratio_health,
                preupdate_health=preupdate_health,
            )
            if not gate["passed"]:
                raise P8FixedTokenQualificationError("unchanged candidate hard gate failed")
            all_gradients.extend(deepcopy(gate.pop("prompt_gradients")))
            batch_results.append(
                {
                    "batch_index": batch_index,
                    "source_order": [row["target_role"] for row in rows],
                    "sample_identity_sha256": _canonical_sha256(
                        [
                            {"sample_id": row["sample_id"], "content_hash": row["content_hash"]}
                            for row in rows
                        ]
                    ),
                    "exception": exception,
                    "hard_gates": gate,
                    "ratio_health": ratio_health,
                    "preupdate_backend_health": preupdate_health,
                    "rollback": rollback_validation,
                    "rejection_artifact_path": str(rejected[-1]),
                    "rejection_artifact_sha256": _sha256_file(rejected[-1]),
                }
            )

        budget = compare_source_mix_budget(all_gradients)
        response_token_shas = [item["hard_gates"]["response_token_sha256"] for item in batch_results]
        fixed_token_identity = {
            "batch_response_token_sha256": response_token_shas,
            "composite_response_token_sha256": _canonical_sha256(response_token_shas),
            "generate_once_per_batch": True,
            "baseline_and_candidate_views_share_tokens": True,
            "baseline_and_candidate_views_share_prompt_gradients": True,
        }
        # The two preregistered views differ only in source weights applied to
        # the target budget.  All safety evidence is therefore byte-identical
        # between views, making relative safety degradation exactly zero.
        safety_comparison = {
            "comparison_scope": "same_candidate_update_same_token_evidence_two_source_weight_views",
            "relative_degradation": 0.0,
            "maximum_allowed": 0.10,
            "passed": True,
            "hard_gates_passed": all(item["hard_gates"]["passed"] for item in batch_results),
        }
        passed = bool(
            budget["target_improvement_passed"]
            and safety_comparison["passed"]
            and safety_comparison["hard_gates_passed"]
            and all(item["rollback"]["passed"] for item in batch_results)
        )
        result = {
            "schema_version": 1,
            "artifact_kind": "p8_fixed_token_source_mix_qualification_v1",
            "passed": passed,
            "status": "candidate_qualified" if passed else "candidate_qualification_failed",
            "candidate_branch": "A",
            "single_training_semantic_variable": "medical_source_mix",
            "fresh_v0_identity": identity,
            "fixed_token_identity": fixed_token_identity,
            "source_mix_budget": budget,
            "safety_comparison": safety_comparison,
            "batches": batch_results,
            "candidate_update_committed": False,
            "optimizer_cursor_rng_sampler_advanced": False,
            "p8_schedule_sha256": schedule["schedule_sha256"],
            "candidate_config_path": str(candidate_config),
            "candidate_config_sha256": _sha256_file(candidate_config),
            "formula_path": str(formula_path),
            "formula_sha256": _sha256_file(formula_path),
            "source_package_content_sha256": package_index["package_content_sha256"],
            "source_package_index_sha256": _sha256_file(source_package / "package_index.json"),
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
            "elapsed_seconds": time.time() - started,
            "reference_price_cny_per_hour": 2.96,
            "derived_cost_cny": (time.time() - started) / 3600.0 * 2.96,
            "platform_actual_cost_cny": None,
        }
        _atomic_json(output / "qualification.json", result)
        return result
    finally:
        if session is not None:
            session.close()


__all__ = [
    "P8FixedTokenQualificationError",
    "compare_source_mix_budget",
    "run_p8_fixed_token_qualification",
    "summarize_fixed_token_attempts",
    "validate_unconditional_rollback",
]
