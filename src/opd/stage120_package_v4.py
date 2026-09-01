"""CPU-safe P7 Stage-120 method definitions and immutable package helpers."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class P7PackageError(RuntimeError):
    """A P7 package definition or immutable package differs."""


BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TEACHER_ORDERED_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
TEACHER_WEIGHT_SHA256 = "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63"
TEACHER_MANIFEST_SHA256 = "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67"


def _is_sha(value: Any, *, length: int = 64) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def freeze_stage120_method_definitions_v4(
    *,
    seed: int,
    capability_results_opened: bool,
    b0_general_micro: float,
    b1_medical: float,
) -> dict[str, dict[str, Any]]:
    """Freeze IDT and CA definitions together before either result exists."""

    if capability_results_opened:
        raise P7PackageError("P7 method definitions must freeze before capability results")
    if seed != 42 or not (0.0 <= b0_general_micro <= 1.0 and 0.0 <= b1_medical <= 1.0):
        raise P7PackageError("P7 method definition inputs differ")
    common = {
        "schema_version": 4,
        "actions": ["medical", "general"],
        "action_definitions": {
            "medical": {
                "teacher_route": "sft_v3_medical_teacher",
                "prompt_count": 4,
                "strata": {"medical_opd_o1": 2, "medical_opd_cmb": 2},
                "prompt_only": True,
                "forbidden_fields": [
                    "answer",
                    "output",
                    "solution",
                    "reasoning",
                    "label",
                ],
            },
            "general": {
                "teacher_route": "qwen3_4b_base",
                "prompt_count": 4,
                "strata": {
                    "BAAI/COIG": 2,
                    "Instruction-Tuning-with-GPT-4/GPT-4-LLM": 2,
                },
                "target_role": "general_anchors",
                "prompt_only": True,
                "forbidden_fields": [
                    "answer",
                    "output",
                    "solution",
                    "reasoning",
                    "label",
                ],
                "controller_or_final_sources_allowed": False,
            },
        },
        "fresh_v0_required": True,
        "accepted_optimizer_commits": 120,
        "prompt_batch": 4,
        "seed": 42,
        "formula": "formula-v6-or-strict-qualified-successor",
        "checkpoint_steps": [30, 60, 90, 120],
        "capability_steps": [60, 90, 120],
        "bounded_rejection": {"total_max": 3, "consecutive_max": 2},
        "final_authorized": False,
        "confirmation_authorized": False,
        "written_before_idt_or_ca_results": True,
    }
    idt = {
        **deepcopy(common),
        "method_id": "IDT-v2",
        "fixed_order": "odd_medical_even_general",
        "adaptive_routing": False,
    }
    ca = {
        **deepcopy(common),
        "method_id": "CA-OPD-v2",
        "adaptive_routing": True,
        "router": {
            "medical_target": float(b1_medical),
            "general_baseline": float(b0_general_micro),
            "delta": 0.01,
            "scale_medical": 0.05,
            "scale_general": 0.05,
            "rho": 0.7,
            "tau": 1.0,
            "p_min": 0.2,
            "p_max": 0.8,
            "window_steps": 30,
            "windows_below_to_recover": 2,
            "windows_above_to_release": 1,
            "initial_p_medical": 0.5,
        },
        "domain_kl_safety": {
            "kappa_medical": 1.0,
            "kappa_general": 1.0,
            "rho": 0.9,
            "eps": 1.0e-6,
            "damp_only": True,
        },
    }
    return {"IDT-v2": idt, "CA-OPD-v2": ca}


def freeze_stage120_decision_rules_v4(
    *, capability_results_opened: bool
) -> dict[str, Any]:
    """Freeze the post-120 recommendation state machine before training."""

    if capability_results_opened:
        raise P7PackageError("P7 scale rules must freeze before capability results")
    return {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_to_300_decision_rules_v4",
        "states": [
            "close_at_120",
            "recommend_b2_scale_to_300",
            "recommend_idt_ca_scale_to_300",
            "repair_before_scale",
            "stop_no_scale",
        ],
        "precedence": [
            "repair_before_scale",
            "close_at_120",
            "recommend_b2_scale_to_300",
            "recommend_idt_ca_scale_to_300",
            "stop_no_scale",
        ],
        "close_at_120": {
            "all": [
                "ca_general_ge_b0_minus_0.01",
                "ca_medical_gt_b0",
                "ca_medical_gt_same_budget_idt",
                "identity_health_data_checkpoint_valid",
            ],
            "statistical_significance_not_required_for_point_estimate_claim": True,
        },
        "recommend_b2_scale_to_300": {
            "all": [
                "b2_medical_le_b0",
                "b2_training_healthy",
                "teacher_student_gap_nonzero",
                "advantage_gradient_not_collapsed",
                "legal_late_checkpoint_improvement_trend",
            ],
            "equal_compute_endpoint_claim_allowed": False,
        },
        "recommend_idt_ca_scale_to_300": {
            "all": [
                "idt_and_ca_healthy",
                "ca_not_clear_winner",
                "late_checkpoints_still_changing",
                "best_medical_at_90_or_120_or_medical_60_to_120_ge_0.01",
                "teacher_signal_not_collapsed",
                "undertraining_or_statistical_uncertainty_not_implementation_error",
            ],
            "methods_must_scale_together": ["IDT-v2", "CA-OPD-v2"],
        },
        "repair_before_scale": {
            "any": [
                "common_loss_mask_old_logprob_sampler_error",
                "data_route_error",
                "teacher_identity_error",
                "final_or_controller_leakage",
                "ca_tuned_after_idt_result",
                "checkpoint_not_recoverable",
            ]
        },
        "stop_no_scale": {
            "any": [
                "healthy_medical_plateau_or_worsening",
                "teacher_student_gap_near_zero",
                "advantage_or_gradient_collapsed",
                "general_severely_declining",
                "no_mechanistic_basis_for_more_steps",
            ]
        },
        "automatic_300_launch": False,
        "automatic_final_access": False,
        "written_before_idt_or_ca_results": True,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }


def build_stage120_package_payloads_v4(
    *,
    method_id: str,
    output_dir: str,
    source_config: Mapping[str, Any],
    authority: Mapping[str, Any],
    schedule: Mapping[str, Any],
    health_protocol: Mapping[str, Any],
    method_definitions: Mapping[str, Mapping[str, Any]],
    step46_qualification_sha256: str,
    step46_qualification_status: str,
    git_head: str,
    capability_results_opened: bool,
) -> dict[str, dict[str, Any]]:
    """Build the shared, pre-result payload set for one immutable method package."""

    if capability_results_opened:
        raise P7PackageError("P7 packages must freeze before capability results")
    model_identity = source_config.get("model", {})
    source_model_revision = model_identity.get(
        "model_revision", model_identity.get("base_revision")
    )
    aliases_consistent = not (
        model_identity.get("model_revision") is not None
        and model_identity.get("base_revision") is not None
        and model_identity.get("model_revision") != model_identity.get("base_revision")
    )
    if not (
        method_id in {"IDT-v2", "CA-OPD-v2"}
        and method_id in method_definitions
        and output_dir
        and _is_sha(step46_qualification_sha256)
        and step46_qualification_status == "qualified"
        and _is_sha(git_head, length=40)
        and source_model_revision == BASE_REVISION
        and aliases_consistent
        and model_identity.get("tokenizer_revision") == BASE_REVISION
        and source_config.get("teacher", {}).get("adapter_sha256")
        == TEACHER_ORDERED_SHA256
        and source_config.get("teacher", {}).get("adapter_weight_sha256")
        == TEACHER_WEIGHT_SHA256
        and source_config.get("teacher", {}).get("manifest_sha256")
        == TEACHER_MANIFEST_SHA256
        and _is_sha(
            source_config.get("protocol", {}).get("three_policy_formula_sha256")
        )
        and _is_sha(authority.get("manifest_sha256"))
        and authority.get("final_authorized") is False
        and _is_sha(schedule.get("schedule_sha256"))
        and schedule.get("final_access_count") == 0
        and health_protocol.get("protocol_id") == "p7_backend_health_v3"
        and health_protocol.get("final_access_allowed") is False
    ):
        raise P7PackageError("P7 package identity inputs differ")
    decision = freeze_stage120_decision_rules_v4(
        capability_results_opened=False
    )
    config = deepcopy(dict(source_config))
    config.pop("formal_method_v3", None)
    config["run"] = {
        "run_id": output_dir.rstrip("/").split("/")[-1],
        "seed": 42,
        "optimizer_steps": 150,
        "stage1_stop_step": 120,
        "output_dir": output_dir,
    }
    config["stage120_v4"] = deepcopy(dict(method_definitions[method_id]))
    config["backend_health_v3"] = deepcopy(dict(health_protocol))
    config["stage120_v4"]["schedule_file"] = "stage120_schedule.json"
    config["stage120_v4"]["action_route_authority"] = "stage120_schedule.json"
    config["stage120_v4"]["inherited_data_config_status"] = (
        "runtime_shape_validation_only_not_stage120_route_authority"
    )
    config["stage120_v4"]["health_protocol_file"] = "health_protocol.json"
    config["stage120_v4"]["decision_rules_file"] = "decision_rules.json"
    config["stage120_v4"]["step46_qualification_sha256"] = (
        step46_qualification_sha256
    )
    config["stage120_v4"]["capability_results_opened_at_freeze"] = False
    config["stage120_v4"]["final_authorized"] = False
    formula_sha = str(config["protocol"]["three_policy_formula_sha256"])
    provenance = {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_package_provenance_v4",
        "method_id": method_id,
        "git_head": git_head,
        "base_revision": BASE_REVISION,
        "teacher_ordered_sha256": TEACHER_ORDERED_SHA256,
        "teacher_weight_sha256": TEACHER_WEIGHT_SHA256,
        "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
        "manifest_sha256": authority["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "formula_sha256": formula_sha,
        "health_protocol_id": "p7_backend_health_v3",
        "step46_qualification_sha256": step46_qualification_sha256,
        "step46_qualification_status": "qualified",
        "written_before_idt_or_ca_results": True,
        "final_access_count": 0,
        "confirmation_access_count": 0,
    }
    authorization = {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_authorization_v4",
        "method_id": method_id,
        "git_head": git_head,
        "formal_training_authorized": True,
        "target_accepted_steps": 120,
        "automatic_300_authorized": False,
        "final_authorized": False,
        "confirmation_600_authorized": False,
    }
    return {
        "formal_method_config.json": config,
        "data_authority.json": deepcopy(dict(authority)),
        "stage120_schedule.json": deepcopy(dict(schedule)),
        "method_definition.json": deepcopy(dict(method_definitions[method_id])),
        "health_protocol.json": deepcopy(dict(health_protocol)),
        "decision_rules.json": decision,
        "provenance.json": provenance,
        "authorization.json": authorization,
    }


def write_stage120_package_v4(
    package: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Atomically write a fresh content-addressed package directory."""

    package = Path(package).resolve()
    expected = {
        "formal_method_config.json",
        "data_authority.json",
        "stage120_schedule.json",
        "method_definition.json",
        "health_protocol.json",
        "decision_rules.json",
        "provenance.json",
        "authorization.json",
    }
    if package.exists() or package.is_symlink() or set(payloads) != expected:
        raise P7PackageError("P7 package output/payload set differs")
    package.mkdir(parents=True)
    for name in sorted(expected):
        value = payloads[name]
        if not isinstance(value, Mapping):
            raise P7PackageError("P7 package payload is not an object")
        _atomic_json(package / name, value)
    files = {
        name: {
            "sha256": _sha_file(package / name),
            "size_bytes": (package / name).stat().st_size,
        }
        for name in sorted(expected)
    }
    config = payloads["formal_method_config.json"]
    method_id = config.get("stage120_v4", {}).get("method_id")
    provenance = payloads["provenance.json"]
    if not (
        method_id in {"IDT-v2", "CA-OPD-v2"}
        and provenance.get("method_id") == method_id
    ):
        raise P7PackageError("P7 package method identity differs")
    index = {
        "schema_version": 4,
        "artifact_kind": "p7_stage120_package_index_v4",
        "method_id": method_id,
        "files": files,
        "package_content_sha256": _canonical_sha(files),
        "config_sha256": files["formal_method_config.json"]["sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "schedule_sha256": provenance["schedule_sha256"],
        "formula_sha256": provenance["formula_sha256"],
        "health_protocol_sha256": files["health_protocol.json"]["sha256"],
        "step46_qualification_sha256": provenance[
            "step46_qualification_sha256"
        ],
        "written_before_idt_or_ca_results": True,
        "final_access_count": 0,
    }
    _atomic_json(package / "package_index.json", index)
    return index


def verify_stage120_package_v4(package: Path) -> dict[str, Any]:
    """Verify every immutable package payload and its cross-file identity."""

    package = Path(package).resolve()
    try:
        index = json.loads((package / "package_index.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P7PackageError("P7 package index is unreadable") from error
    files = index.get("files") if isinstance(index, Mapping) else None
    if not (
        isinstance(index, Mapping)
        and index.get("schema_version") == 4
        and index.get("artifact_kind") == "p7_stage120_package_index_v4"
        and index.get("method_id") in {"IDT-v2", "CA-OPD-v2"}
        and isinstance(files, Mapping)
        and _is_sha(index.get("package_content_sha256"))
        and index.get("final_access_count") == 0
    ):
        raise P7PackageError("P7 package index contract differs")
    actual: dict[str, Any] = {}
    for name, descriptor in files.items():
        path = package / str(name)
        if not (
            isinstance(descriptor, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and descriptor.get("sha256") == _sha_file(path)
            and descriptor.get("size_bytes") == path.stat().st_size
        ):
            raise P7PackageError("P7 package file SHA differs")
        actual[str(name)] = dict(descriptor)
    if _canonical_sha(actual) != index["package_content_sha256"]:
        raise P7PackageError("P7 package content SHA differs")
    config = json.loads(
        (package / "formal_method_config.json").read_text(encoding="utf-8")
    )
    authorization = json.loads(
        (package / "authorization.json").read_text(encoding="utf-8")
    )
    provenance = json.loads((package / "provenance.json").read_text(encoding="utf-8"))
    method_id = str(index["method_id"])
    if not (
        config.get("stage120_v4", {}).get("method_id") == method_id
        and authorization.get("method_id") == method_id
        and authorization.get("formal_training_authorized") is True
        and authorization.get("target_accepted_steps") == 120
        and authorization.get("automatic_300_authorized") is False
        and authorization.get("final_authorized") is False
        and authorization.get("confirmation_600_authorized") is False
        and provenance.get("method_id") == method_id
        and provenance.get("written_before_idt_or_ca_results") is True
        and provenance.get("formula_sha256") == index.get("formula_sha256")
        and provenance.get("manifest_sha256") == index.get("manifest_sha256")
        and provenance.get("schedule_sha256") == index.get("schedule_sha256")
        and provenance.get("step46_qualification_sha256")
        == index.get("step46_qualification_sha256")
    ):
        raise P7PackageError("P7 package cross-file identity differs")
    return {
        "passed": True,
        "method_id": method_id,
        "package_content_sha256": index["package_content_sha256"],
        "config_sha256": index["config_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "schedule_sha256": index["schedule_sha256"],
        "formula_sha256": index["formula_sha256"],
        "health_protocol_sha256": index["health_protocol_sha256"],
        "step46_qualification_sha256": index["step46_qualification_sha256"],
        "final_access_count": 0,
    }


__all__ = [
    "P7PackageError",
    "build_stage120_package_payloads_v4",
    "freeze_stage120_decision_rules_v4",
    "freeze_stage120_method_definitions_v4",
    "verify_stage120_package_v4",
    "write_stage120_package_v4",
]
