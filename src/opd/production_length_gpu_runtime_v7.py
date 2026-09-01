"""CPU-safe orchestration boundary for the P4.7 GPU continuation.

This module deliberately contains no model-framework import.  A real GPU
backend is injected only by the explicitly authorized launcher; unit tests use
synthetic backends.  The coordinator permits exactly one primary envelope and
one conditional 4096 envelope, and the B2 helper materializes authorization
metadata without starting training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.opd.production_length_contract_v7 import (
    CONDITIONAL_4096_CANDIDATES,
    PRIMARY_CANDIDATES,
    SOURCES,
    canonical_json_sha256,
    compare_prefix_equivalence,
    derive_per_sample_seed,
    select_shortest_passing_length,
)
from src.opd.production_length_artifacts_v7 import (
    derive_length_readiness,
    validate_length_telemetry,
)


class ProductionLengthGpuRuntimeV7Error(RuntimeError):
    """The bounded length continuation or B2 boundary failed closed."""


class GenerationBackendV7(Protocol):
    def generate(
        self,
        rows: list[dict[str, object]],
        *,
        actual_cap: int,
        per_sample_seeds: list[int],
        capture_prefix_provenance: bool,
        candidate_health_caps: Sequence[int],
    ) -> list[dict[str, object]]: ...


PrimaryEvaluator = Callable[
    [Mapping[int, Sequence[Mapping[str, Any]]], str], Mapping[str, Any]
]
ConditionalEvaluator = Callable[
    [Mapping[int, Sequence[Mapping[str, Any]]], str], Mapping[str, Any]
]
PrefixEvidenceWriter = Callable[[Mapping[str, Any]], Any]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_P4_7_CONFIG_PATH = Path("configs/opd/qwen3_4b_length_qualification_v7.yaml")
_B2_TEMPLATE_PATH = Path("configs/runs/b2_medical_opd_qwen3_4b_custom_v5_p4_6.yaml")
_P4_7_RUN_ID = "qwen3-4b-length-qualification-v7-seed42"
_B2_RUN_ID = "qwen3-4b-b2-medical-opd-calibration-p4-7-seed42"
_ROOT_READINESS_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "status",
    "ready",
    "selected_response_length",
    "artifact_index_sha256",
    "cleanup_complete",
    "parent_core_evidence_verified",
    "v2_reload_identity_verified",
    "production_sampler_refresh_ready",
    "OPD_scoring_backend_ready",
    "B2_authorized",
    "B2_started",
    "isolation",
}
_ROOT_REQUIRED_FILES = {
    "preflight.json",
    "metrics.jsonl",
    "v2_reload_identity.json",
    "prefix_equivalence.json",
    "generation_summary.json",
    "worker_release.json",
    "worker_status.json",
    "resource_cleanup.json",
    "finalizer_authority_revalidation.json",
    "length_selection.json",
    "b2_package_manifest.json",
}
_PHASE_REQUIRED_FILES = {
    "length_telemetry.json",
    "length_evidence_index.json",
    "length_selection.json",
    "artifact_index.json",
    "readiness.json",
}


def _fail(message: str) -> None:
    raise ProductionLengthGpuRuntimeV7Error(message)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProductionLengthGpuRuntimeV7Error(
            f"{label} cannot be reopened"
        ) from error
    if not isinstance(value, Mapping):
        _fail(f"{label} is not a mapping")
    return dict(value)


def _canonical_length_config() -> dict[str, Any]:
    return _read_yaml_mapping(_REPO_ROOT / _P4_7_CONFIG_PATH, "canonical P4.7 config")


def _phase_materialization_bindings(
    root: Path, package: Mapping[str, Any]
) -> dict[str, str]:
    """Reopen the selected phase before emitting an executable B2 config."""

    selected = package.get("selected_response_length")
    matching: list[Path] = []
    for name in ("primary", "conditional_4096"):
        phase = root / name
        selection_path = phase / "length_selection.json"
        if not selection_path.is_file() or selection_path.is_symlink():
            continue
        selection = _read_formal_json(selection_path, f"{name} selection")
        if (
            selection.get("status") == "length_frozen"
            and selection.get("selected_response_length") == selected
        ):
            matching.append(phase)
    if len(matching) != 1:
        _fail("B2 materialization cannot identify one selected formal phase")
    phase = matching[0]
    telemetry = _safe_file(phase / "length_telemetry.json")
    selection = _safe_file(phase / "length_selection.json")
    index = _safe_file(phase / "artifact_index.json")
    readiness = _safe_file(phase / "readiness.json")
    cleanup = _safe_file(root / "resource_cleanup.json")
    if not (
        _stream_sha256(telemetry) == package.get("length_telemetry_sha256")
        and _stream_sha256(selection) == package.get("length_selection_sha256")
        and _stream_sha256(index) == package.get("length_final_index_sha256")
    ):
        _fail("B2 materialization phase hashes disagree with authorization")
    phase_readiness = derive_length_readiness(phase)
    if not (
        phase_readiness.get("ready") is True
        and phase_readiness.get("selected_response_length") == selected
    ):
        _fail("B2 materialization phase readiness does not revalidate")
    return {
        "phase_dir": str(phase.resolve()),
        "readiness_sha256": _stream_sha256(readiness),
        "artifact_index_sha256": _stream_sha256(index),
        "cleanup_sha256": _stream_sha256(cleanup),
    }


def _safe_file(path_value: Any, expected_sha: Any | None = None) -> Path:
    path = Path(str(path_value))
    if path.is_symlink() or not path.is_file():
        _fail(f"required formal artifact is absent: {path}")
    if expected_sha is not None and (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or _stream_sha256(path) != expected_sha
    ):
        _fail(f"formal artifact SHA mismatch: {path}")
    return path


def _generation_seeds(rows: Sequence[Mapping[str, Any]], base_seed: int) -> list[int]:
    result: list[int] = []
    identities: set[tuple[str, int]] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        frozen_order = row.get("frozen_order")
        if not isinstance(sample_id, str) or not isinstance(frozen_order, int):
            _fail("generation row lacks frozen sample identity")
        identity = (sample_id, frozen_order)
        if identity in identities:
            _fail("generation rows contain a duplicate frozen identity")
        identities.add(identity)
        result.append(derive_per_sample_seed(base_seed, sample_id, frozen_order))
    return result


def _generate(
    backend: GenerationBackendV7,
    rows: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    base_seed: int,
    capture_prefix: bool,
    candidate_health_caps: Sequence[int],
) -> list[dict[str, object]]:
    copied = [dict(item) for item in rows]
    result = backend.generate(
        copied,
        actual_cap=cap,
        per_sample_seeds=_generation_seeds(copied, base_seed),
        capture_prefix_provenance=capture_prefix,
        candidate_health_caps=candidate_health_caps,
    )
    if not isinstance(result, list) or len(result) != len(copied):
        _fail("generation backend returned an incomplete batch")
    return result


def execute_bounded_generation_plan(
    *,
    backend: GenerationBackendV7,
    prefix_probe_rows: Sequence[Mapping[str, Any]],
    qualification_rows: Sequence[Mapping[str, Any]],
    base_seed: int,
    evaluate_primary: PrimaryEvaluator,
    evaluate_conditional: ConditionalEvaluator,
    persist_prefix_evidence: PrefixEvidenceWriter | None = None,
) -> dict[str, Any]:
    """Execute the finite prefix/2048/(optional 4096) generation graph.

    The function does not decide scientific gates itself.  Evaluators receive
    the real generation batches and must produce evidence-derived decisions.
    A prefix mismatch switches visibly to independent actual generation for
    every candidate instead of silently deriving prefixes.
    """

    prefix_sources = {
        str(row.get("source")) for row in prefix_probe_rows if isinstance(row, Mapping)
    }
    if not set(SOURCES).issubset(prefix_sources):
        _fail("prefix gate requires at least one frozen sample from each source")
    short = _generate(
        backend,
        prefix_probe_rows,
        cap=1024,
        base_seed=base_seed,
        capture_prefix=True,
        candidate_health_caps=(1024,),
    )
    long = _generate(
        backend,
        prefix_probe_rows,
        cap=2048,
        base_seed=base_seed,
        capture_prefix=True,
        candidate_health_caps=(2048,),
    )
    prefix = compare_prefix_equivalence(short, long, prefix_length=1024)
    if persist_prefix_evidence is not None:
        persist_prefix_evidence(prefix)
    if prefix["passed"] is True:
        strategy = "derived_candidates"
        primary_values: dict[int, Sequence[Mapping[str, Any]]] = {
            2048: _generate(
                backend,
                qualification_rows,
                cap=2048,
                base_seed=base_seed,
                capture_prefix=False,
                candidate_health_caps=PRIMARY_CANDIDATES,
            )
        }
    else:
        strategy = "explicit_independent_generation"
        primary_values = {
            candidate: _generate(
                backend,
                qualification_rows,
                cap=candidate,
                base_seed=base_seed,
                capture_prefix=False,
                candidate_health_caps=(candidate,),
            )
            for candidate in PRIMARY_CANDIDATES
        }
    primary = dict(evaluate_primary(primary_values, strategy))
    gate = primary.get("conditional_4096_gate")
    qualification = primary.get("qualification")
    conditional_executed = bool(
        isinstance(qualification, Mapping)
        and qualification.get("status") == "no_length_candidate_passed"
        and qualification.get("ready") is False
        and qualification.get("disk_reverified") is True
        and isinstance(gate, Mapping)
        and gate.get("allowed") is True
    )
    conditional: dict[str, Any] | None = None
    if conditional_executed:
        if strategy == "derived_candidates":
            conditional_strategy = "derived_candidates"
            conditional_values: dict[int, Sequence[Mapping[str, Any]]] = {
                4096: _generate(
                    backend,
                    qualification_rows,
                    cap=4096,
                    base_seed=base_seed,
                    capture_prefix=False,
                    candidate_health_caps=CONDITIONAL_4096_CANDIDATES,
                )
            }
        else:
            conditional_strategy = "explicit_independent_generation"
            conditional_values = {2048: primary_values[2048]}
            for candidate in CONDITIONAL_4096_CANDIDATES[1:]:
                conditional_values[candidate] = _generate(
                    backend,
                    qualification_rows,
                    cap=candidate,
                    base_seed=base_seed,
                    capture_prefix=False,
                    candidate_health_caps=(candidate,),
                )
        conditional = dict(
            evaluate_conditional(conditional_values, conditional_strategy)
        )
    return {
        "schema_version": 7,
        "artifact_kind": "bounded_length_generation_plan_v7",
        "prefix_equivalence": prefix,
        "generation_strategy": strategy,
        "primary": primary,
        "conditional_4096_executed": conditional_executed,
        "conditional": conditional,
        "automatic_further_escalation": False,
    }


_B2_BINDING_FIELDS = {
    "git_commit",
    "backend_id",
    "protocol_sha256",
    "data_manifest_sha256",
    "teacher_manifest_sha256",
    "teacher_adapter_sha256",
    "base_revision",
    "tokenizer_revision",
    "parent_evidence_index_sha256",
    "parent_final_index_sha256",
    "production_two_step_sha256",
    "authority_v1_sha256",
    "authority_v2_sha256",
    "base_null_sha256",
    "length_decision_sha256",
    "seed",
    "estimated_steps",
    "checkpoint_strategy",
    "estimated_cost_cny",
    "actual_cost_cny",
}


def _read_formal_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLengthGpuRuntimeV7Error(
            f"formal {label} artifact cannot be reopened"
        ) from error
    if not isinstance(value, dict) or not value:
        _fail(f"formal {label} artifact is empty or not an object")
    return value


def _reverify_formal_length_evidence(
    formal_length_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute B2 inputs from phase files; caller booleans are non-authoritative."""

    formal_root = Path(str(formal_length_evidence.get("formal_run_root", ""))).resolve()
    phase_dir = Path(str(formal_length_evidence.get("phase_dir", ""))).resolve()
    if (
        not formal_root.is_dir()
        or not phase_dir.is_dir()
        or formal_root not in phase_dir.parents
    ):
        _fail("B2 formal phase directory escapes the length run")
    telemetry_path = _safe_file(
        formal_length_evidence.get("telemetry_path"),
        formal_length_evidence.get("telemetry_sha256"),
    )
    selection_path = _safe_file(
        formal_length_evidence.get("selection_path"),
        formal_length_evidence.get("selection_sha256"),
    )
    phase_index_path = _safe_file(formal_length_evidence.get("final_index_path"))
    cleanup_path = _safe_file(formal_length_evidence.get("cleanup_path"))
    identity_path = _safe_file(formal_length_evidence.get("identity_path"))
    prefix_path = _safe_file(formal_length_evidence.get("prefix_path"))
    if any(
        formal_root not in path.resolve().parents
        for path in (
            telemetry_path,
            selection_path,
            phase_index_path,
            cleanup_path,
            identity_path,
            prefix_path,
        )
    ):
        _fail("B2 evidence paths escape the formal length run")
    if any(
        phase_dir not in path.resolve().parents
        for path in (telemetry_path, selection_path, phase_index_path)
    ):
        _fail("B2 phase evidence is not co-located")

    telemetry = validate_length_telemetry(
        _read_formal_json(telemetry_path, "telemetry")
    )
    selection = _read_formal_json(selection_path, "selection")
    expected_selection = dict(select_shortest_passing_length(telemetry))
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        _fail("formal length selection does not reproduce from telemetry")
    phase_readiness = derive_length_readiness(phase_dir)
    selected = expected_selection.get("selected_response_length")
    if not (
        phase_readiness.get("ready") is True
        and phase_readiness.get("selected_response_length") == selected
        and isinstance(selected, int)
    ):
        _fail("formal length phase/index does not revalidate as success")

    cleanup = _read_formal_json(cleanup_path, "cleanup")
    if not (
        cleanup.get("cleanup_complete") is True
        and cleanup.get("worker_released") is True
        and cleanup.get("gpu_memory_used_mib") == [0, 0]
        and cleanup.get("compute_pids") == []
        and cleanup.get("residual_worker_pids") == []
        and cleanup.get("B2_started") is False
    ):
        _fail("formal cleanup does not prove both GPUs idle after worker exit")

    identity = _read_formal_json(identity_path, "v2 identity")
    runtime_sha = identity.get("runtime_tensor_sha256")
    if not (
        identity.get("artifact_kind") == "p4_7_fresh_v2_reload_identity"
        and identity.get("passed") is True
        and identity.get("logical_version") == "v2"
        and identity.get("checkpoint_tensor_sha256") == runtime_sha
        and runtime_sha == telemetry["bindings"]["runtime_adapter_sha256"]
        and identity.get("tensor_count") == 504
        and identity.get("active_slot") == "student_active"
        and identity.get("registry_count") == 1
        and identity.get("eos_stop_config_verified") is True
    ):
        _fail("formal v2 identity does not reproduce")

    prefix = _read_formal_json(prefix_path, "prefix equivalence")
    counts = prefix.get("per_source_probe_count")
    if not (
        prefix.get("artifact_kind") == "production_length_prefix_equivalence_v7"
        and isinstance(prefix.get("passed"), bool)
        and isinstance(counts, Mapping)
        and all(isinstance(counts.get(source), int) and counts[source] >= 1 for source in SOURCES)
    ):
        _fail("formal prefix evidence is incomplete")
    derived = telemetry.get("generation_mode") == "single_actual_trajectory_derived_candidates"
    explicit = telemetry.get("generation_mode") == "explicit_independent_generation"
    if (prefix["passed"] is True and not derived) or (
        prefix["passed"] is False and not explicit
    ):
        _fail("prefix result and generation fallback strategy disagree")

    parent_sha = telemetry["bindings"]["parent_p4_6_binding_sha256"]
    if not (
        formal_length_evidence.get("parent_reuse_attestation_sha256") == parent_sha
        and formal_length_evidence.get("runtime_adapter_sha256") == runtime_sha
        and formal_length_evidence.get("selected_response_length") == selected
        and formal_length_evidence.get("failure_artifact_exists") is False
    ):
        _fail("caller evidence binding disagrees with reopened formal artifacts")
    return {
        "selected_response_length": selected,
        "parent_reuse_attestation_sha256": parent_sha,
        "runtime_adapter_sha256": runtime_sha,
        "telemetry_sha256": _stream_sha256(telemetry_path),
        "selection_sha256": _stream_sha256(selection_path),
        "phase_index_sha256": _stream_sha256(phase_index_path),
    }


def build_b2_calibration_package(
    formal_length_evidence: Mapping[str, Any], bindings: Mapping[str, Any]
) -> dict[str, Any]:
    """Build, but never start, the 20-step package after formal GPU success."""

    isolation = {
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }
    verified = _reverify_formal_length_evidence(formal_length_evidence)
    selected = verified["selected_response_length"]
    allowed_lengths = set(PRIMARY_CANDIDATES) | set(CONDITIONAL_4096_CANDIDATES)
    if not (
        formal_length_evidence.get("status") == "passed_length_only_qualification"
        and formal_length_evidence.get("execution_mode") == "formal_gpu"
        and isinstance(selected, int)
        and not isinstance(selected, bool)
        and selected in allowed_lengths
        and formal_length_evidence.get("failure_artifact_exists") is False
        and formal_length_evidence.get("cleanup_complete") is True
        and formal_length_evidence.get("isolation") == isolation
    ):
        _fail("B2 package requires successful isolated formal GPU length evidence")
    if set(bindings) != _B2_BINDING_FIELDS:
        _fail("B2 package binding fields are not exact")
    if (
        not isinstance(bindings["git_commit"], str)
        or len(str(bindings["git_commit"])) != 40
        or bindings["backend_id"] != "custom_transformers_peft_three_policy_v5"
        or bindings["estimated_steps"] != 20
        or bindings["actual_cost_cny"] is not None
        or bindings["length_decision_sha256"]
        != verified["selection_sha256"]
    ):
        _fail("B2 calibration binding drift")
    for field in (
        "protocol_sha256",
        "data_manifest_sha256",
        "teacher_manifest_sha256",
        "teacher_adapter_sha256",
        "parent_evidence_index_sha256",
        "parent_final_index_sha256",
        "production_two_step_sha256",
        "authority_v1_sha256",
        "authority_v2_sha256",
        "base_null_sha256",
        "length_decision_sha256",
    ):
        if not isinstance(bindings[field], str) or len(bindings[field]) != 64:
            _fail(f"B2 calibration binding lacks {field}")
    package = {
        "schema_version": 1,
        "artifact_kind": "p4_7_b2_20_step_calibration_authorization",
        "status": "authorized_not_started",
        "selected_response_length": selected,
        "parent_reuse_attestation_sha256": verified[
            "parent_reuse_attestation_sha256"
        ],
        "runtime_adapter_sha256": verified["runtime_adapter_sha256"],
        "length_telemetry_sha256": verified["telemetry_sha256"],
        "length_selection_sha256": verified["selection_sha256"],
        "length_final_index_sha256": verified["phase_index_sha256"],
        "bindings": dict(bindings),
        "isolation": isolation,
        "B2_authorized": True,
        "B2_started": False,
        "requires_explicit_allow_b2_calibration": True,
    }
    package["package_content_sha256"] = canonical_json_sha256(package)
    return package


def assert_b2_calibration_start_authorized(
    package: Mapping[str, Any],
    *,
    allow_b2_calibration: bool,
    formal_run_root: str | Path | None = None,
    package_dir: str | Path | None = None,
    authority_revalidation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate start gate; require the later sealed disk authorization graph."""

    content = dict(package)
    claimed = content.pop("package_content_sha256", None)
    if not (
        allow_b2_calibration is True
        and package.get("status") == "authorized_not_started"
        and package.get("B2_authorized") is True
        and package.get("B2_started") is False
        and package.get("requires_explicit_allow_b2_calibration") is True
        and package.get("isolation")
        == {
            "final_access": False,
            "controller_access": False,
            "confirmation_access": False,
            "label_access": False,
        }
        and isinstance(claimed, str)
        and claimed == canonical_json_sha256(content)
    ):
        _fail("B2 calibration start is not explicitly artifact-authorized")
    if formal_run_root is None or package_dir is None:
        _fail("B2 start requires sealed formal root and materialized package")
    root = Path(formal_run_root).resolve()
    materialized = Path(package_dir).resolve()
    if not root.is_dir() or root.is_symlink() or not materialized.is_dir() or materialized.is_symlink():
        _fail("B2 disk authorization directories are absent")
    readiness_path = _safe_file(root / "readiness.json")
    index_path = _safe_file(root / "artifact_index.json")
    readiness = _read_formal_json(readiness_path, "root readiness")
    index = _read_formal_json(index_path, "root index")
    if not (
        set(readiness) == _ROOT_READINESS_FIELDS
        and readiness.get("schema_version") == 7
        and readiness.get("artifact_kind") == "p4_7_length_final_readiness"
        and readiness.get("run_id") == _P4_7_RUN_ID
        and readiness.get("status") == "passed_length_only_qualification"
        and readiness.get("ready") is True
        and readiness.get("selected_response_length")
        == package.get("selected_response_length")
        and readiness.get("artifact_index_sha256") == _stream_sha256(index_path)
        and readiness.get("cleanup_complete") is True
        and readiness.get("parent_core_evidence_verified") is True
        and readiness.get("v2_reload_identity_verified") is True
        and readiness.get("production_sampler_refresh_ready") is True
        and readiness.get("OPD_scoring_backend_ready") is True
        and readiness.get("B2_authorized") is True
        and readiness.get("B2_started") is False
        and readiness.get("isolation") == package.get("isolation")
    ):
        _fail("B2 root readiness is not an exact sealed success")
    if not (
        set(index)
        == {"schema_version", "artifact_kind", "run_id", "artifact_count", "artifacts"}
        and index.get("schema_version") == 7
        and index.get("artifact_kind") == "p4_7_length_final_index"
        and index.get("run_id") == _P4_7_RUN_ID
    ):
        _fail("B2 root index identity is not exact")
    entries = index.get("artifacts")
    if not (
        isinstance(entries, list)
        and entries
        and index.get("artifact_count") == len(entries)
    ):
        _fail("B2 root index is incomplete")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not (
            isinstance(entry, Mapping)
            and set(entry) == {"path", "sha256", "size_bytes"}
            and isinstance(entry.get("path"), str)
        ):
            _fail("B2 root index entry is invalid")
        relative = str(entry["path"])
        if relative in indexed or relative.startswith("/") or ".." in Path(relative).parts:
            _fail("B2 root index path is invalid")
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or entry.get("sha256") != _stream_sha256(path)
            or entry.get("size_bytes") != path.stat().st_size
        ):
            _fail("B2 root index entry does not revalidate")
        indexed[relative] = entry
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix()
        not in {"artifact_index.json", "readiness.json"}
    }
    if set(indexed) != actual:
        _fail("B2 root index is not an exact graph of disk artifacts")
    if not _ROOT_REQUIRED_FILES.issubset(indexed):
        _fail("B2 root index lacks required production phase evidence")
    forbidden = {
        path
        for path in indexed
        if any(
            token in Path(path).name.lower()
            for token in (
                "controller",
                "confirmation",
                "final_manifest",
                "final_label",
                "label_manifest",
            )
        )
    }
    if forbidden or (root / "failure.json").exists():
        _fail("B2 root graph contains forbidden or failed evidence")

    persisted_authority = _read_formal_json(
        root / "finalizer_authority_revalidation.json", "final authority"
    )
    if not (
        isinstance(authority_revalidation, Mapping)
        and dict(authority_revalidation) == persisted_authority
        and persisted_authority.get("schema_version") == 7
        and persisted_authority.get("artifact_kind")
        == "p4_7_finalizer_authority_revalidation"
        and persisted_authority.get("parent_core_evidence_verified") is True
        and persisted_authority.get("v2_adapter_reusable") is True
        and persisted_authority.get("current_bindings_verified") is True
        and persisted_authority.get("static_assets_verified") is True
        and persisted_authority.get("worktree_clean") is True
        and persisted_authority.get("B2_started") is False
        and persisted_authority.get("isolation") == package.get("isolation")
    ):
        _fail("B2 parent/static authority was not freshly revalidated")
    preflight = _read_formal_json(root / "preflight.json", "root preflight")
    if not (
        preflight.get("run_id") == _P4_7_RUN_ID
        and preflight.get("parent_core_evidence_verified") is True
        and preflight.get("v2_adapter_reusable") is True
        and preflight.get("git", {}).get("git_commit")
        == persisted_authority.get("git_commit")
        and preflight.get("git", {}).get("worktree_clean") is True
        and preflight.get("parent_audit_sha256")
        == persisted_authority.get("parent_audit_sha256")
        and preflight.get("current_bindings_sha256")
        == persisted_authority.get("current_bindings_sha256")
        and preflight.get("static_assets_sha256")
        == persisted_authority.get("static_assets_sha256")
        and preflight.get("versions") == persisted_authority.get("versions")
        and preflight.get("isolation") == package.get("isolation")
    ):
        _fail("B2 root preflight/authority chain drifted")

    combined = _read_formal_json(root / "length_selection.json", "combined selection")
    selected_phase = combined.get("source_phase")
    phase_name = (
        "primary" if selected_phase == "primary" else "conditional_4096"
        if selected_phase == "conditional"
        else None
    )
    if phase_name is None:
        _fail("B2 combined selection does not identify a bounded phase")
    phase_dir = root / phase_name
    phase_required = {f"{phase_name}/{name}" for name in _PHASE_REQUIRED_FILES}
    if not phase_required.issubset(indexed):
        _fail("B2 root index lacks exact selected phase evidence")
    source_selection = _safe_file(phase_dir / "length_selection.json")
    if not (
        combined.get("schema_version") == 7
        and combined.get("artifact_kind") == "p4_7_combined_length_selection"
        and combined.get("selected_response_length")
        == package.get("selected_response_length")
        and combined.get("source_selection_sha256")
        == _stream_sha256(source_selection)
        and combined.get("B2_started") is False
    ):
        _fail("B2 combined selection is not bound to the selected phase")
    summary = _read_formal_json(root / "generation_summary.json", "generation summary")
    if not (
        summary.get("artifact_kind") == "p4_7_bounded_generation_summary"
        and summary.get("automatic_further_escalation") is False
        and (
            (phase_name == "primary" and summary.get("conditional_4096_executed") is False)
            or (
                phase_name == "conditional_4096"
                and summary.get("conditional_4096_executed") is True
                and "conditional_4096_preflight.json" in indexed
                and "conditional_4096_eligibility.json" in indexed
            )
        )
    ):
        _fail("B2 generation summary does not bind the bounded selected phase")
    telemetry_path = phase_dir / "length_telemetry.json"
    verified = _reverify_formal_length_evidence(
        {
            "formal_run_root": str(root),
            "phase_dir": str(phase_dir),
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": package.get("length_telemetry_sha256"),
            "selection_path": str(source_selection),
            "selection_sha256": package.get("length_selection_sha256"),
            "final_index_path": str(phase_dir / "artifact_index.json"),
            "cleanup_path": str(root / "resource_cleanup.json"),
            "identity_path": str(root / "v2_reload_identity.json"),
            "prefix_path": str(root / "prefix_equivalence.json"),
            "parent_reuse_attestation_sha256": package.get(
                "parent_reuse_attestation_sha256"
            ),
            "runtime_adapter_sha256": package.get("runtime_adapter_sha256"),
            "selected_response_length": package.get("selected_response_length"),
            "failure_artifact_exists": False,
        }
    )
    if not (
        verified["telemetry_sha256"] == package.get("length_telemetry_sha256")
        and verified["selection_sha256"] == package.get("length_selection_sha256")
        and verified["phase_index_sha256"] == package.get("length_final_index_sha256")
    ):
        _fail("B2 package differs from independently reopened phase evidence")
    manifest = _read_formal_json(
        root / "b2_package_manifest.json", "B2 package manifest"
    )
    if not (
        Path(str(manifest.get("output_dir", ""))).resolve() == materialized
        and manifest.get("B2_authorized") is True
        and manifest.get("B2_started") is False
        and isinstance(manifest.get("files"), Mapping)
    ):
        _fail("B2 package manifest is invalid")
    expected_names = {
        "b2_20_step_calibration_config.json",
        "b2_20_step_calibration_run_card.json",
        "b2_authorization.json",
    }
    if set(manifest["files"]) != expected_names or {
        item.name for item in materialized.iterdir()
    } != expected_names:
        _fail("B2 materialized package file set is not exact")
    for name in expected_names:
        path = materialized / name
        metadata = manifest["files"][name]
        if not (
            isinstance(metadata, Mapping)
            and path.is_file()
            and not path.is_symlink()
            and metadata.get("sha256") == _stream_sha256(path)
            and metadata.get("size_bytes") == path.stat().st_size
        ):
            _fail("B2 materialized package SHA/size mismatch")
    authorization = _read_formal_json(
        materialized / "b2_authorization.json", "B2 authorization"
    )
    config = _read_formal_json(
        materialized / "b2_20_step_calibration_config.json", "B2 config"
    )
    card = _read_formal_json(
        materialized / "b2_20_step_calibration_run_card.json", "B2 run card"
    )
    if not (
        authorization == dict(package)
        and config.get("generation", {}).get("max_new_tokens")
        == package.get("selected_response_length")
        and config.get("run", {}).get("automatically_start") is False
        and config.get("authorization", {}).get("B2_started") is False
        and Path(str(config.get("qualification", {}).get("output_path", ""))).resolve()
        == root
        and config.get("qualification", {}).get("length_decision_sha256")
        == package.get("length_selection_sha256")
        and config.get("qualification", {}).get("v2_tensor_sha256")
        == package.get("runtime_adapter_sha256")
        and card.get("selected_response_length") == package.get("selected_response_length")
        and card.get("status") == "authorized_not_started"
        and card.get("requires_argument") == "--allow-b2-calibration"
        and card.get("B2_started") is False
    ):
        _fail("B2 materialized authorization contents drift")
    return dict(package)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def materialize_b2_calibration_package(
    output_dir: str | Path,
    package: Mapping[str, Any],
    *,
    source_length_run_id: str,
    formal_run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Durably stage the authorized config/card without launching training."""

    content = dict(package)
    claimed = content.pop("package_content_sha256", None)
    if not (
        package.get("status") == "authorized_not_started"
        and package.get("B2_authorized") is True
        and package.get("B2_started") is False
        and package.get("requires_explicit_allow_b2_calibration") is True
        and claimed == canonical_json_sha256(content)
        and isinstance(source_length_run_id, str)
        and source_length_run_id
    ):
        _fail("cannot materialize an unverified B2 authorization package")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        _fail("B2 calibration package output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    config_name = "b2_20_step_calibration_config.json"
    card_name = "b2_20_step_calibration_run_card.json"
    authorization_name = "b2_authorization.json"
    source_config = _canonical_length_config()
    template = _read_yaml_mapping(_REPO_ROOT / _B2_TEMPLATE_PATH, "frozen B2 template")
    root = Path(
        formal_run_root
        if formal_run_root is not None
        else source_config.get("run", {}).get("output_dir", "")
    ).resolve()
    if source_length_run_id != source_config.get("run", {}).get("run_id"):
        _fail("B2 source run identity differs from the canonical P4.7 run")
    phase_bindings = _phase_materialization_bindings(root, package)
    backend = dict(template.get("production_backend", {}))
    protocol = dict(template.get("protocol_binding", {}))
    source_model = source_config.get("model", {})
    source_teacher = source_config.get("teacher_binding_for_future_b2_package", {})
    source_data = source_config.get("prompt_selection", {})
    parent = source_config.get("parent_reuse", {})
    v2 = parent.get("v2", {}) if isinstance(parent, Mapping) else {}
    if not all(
        isinstance(value, Mapping)
        for value in (source_model, source_teacher, source_data, parent, v2)
    ) or not backend or not protocol:
        _fail("canonical P4.7/B2 production bindings are incomplete")
    bindings = package["bindings"]
    qualification = {
        "run_id": source_length_run_id,
        "attempt_id": "formal-attempt-1",
        "readiness_sha256": phase_bindings["readiness_sha256"],
        "artifact_index_sha256": phase_bindings["artifact_index_sha256"],
        "backend_binding_sha256": canonical_json_sha256(backend),
        "protocol_sha256": bindings["protocol_sha256"],
        "data_manifest_sha256": bindings["data_manifest_sha256"],
        "authority_v2_sha256": bindings["authority_v2_sha256"],
        "base_null_sha256": bindings["base_null_sha256"],
        "length_decision_sha256": package["length_selection_sha256"],
        "cleanup_sha256": phase_bindings["cleanup_sha256"],
        "output_path": str(root),
        "v2_checkpoint_path": str(v2.get("canonical_absolute_path", "")),
        "v2_tensor_sha256": package["runtime_adapter_sha256"],
    }
    config = {
        "schema_id": "ca-opd/b2-medical-opd-calibration/v1",
        "schema_version": 1,
        "run": {
            "run_id": _B2_RUN_ID,
            "stage": "b2_medical_opd_calibration",
            "baseline_id": "B2",
            "purpose": "P4.7-qualified production Medical OPD calibration",
            "seed": bindings["seed"],
            "status": "authorized_not_started",
            "optimizer_steps": bindings["estimated_steps"],
            "output_dir": (
                "artifacts/outputs/" + _B2_RUN_ID
            ),
            "automatically_start": False,
        },
        "production_backend": backend,
        "executor": {
            "path": "src/opd/production_qualification_gpu_runtime_v6.py",
            "symbol": "execute_b2_medical_opd_gpu_protocol_v6",
        },
        "model": {
            "base_path": source_model.get("id"),
            "base_manifest_path": source_model.get("artifact_manifest_path"),
            "base_manifest_sha256": source_model.get("artifact_manifest_sha256"),
            "base_weights_manifest_path": source_model.get("weights_manifest_path"),
            "base_weights_manifest_sha256": source_model.get("weights_manifest_sha256"),
            "model_revision": source_model.get("revision"),
            "tokenizer_revision": source_model.get("tokenizer_revision"),
            "dtype": source_model.get("dtype"),
            "attention_backend": source_model.get("attention_implementation"),
        },
        "teacher": {
            "adapter_path": source_teacher.get("adapter_path"),
            "adapter_sha256": source_teacher.get("adapter_sha256"),
            "adapter_weight_sha256": source_teacher.get("adapter_weight_sha256"),
            "manifest_path": source_teacher.get("manifest_path"),
            "manifest_sha256": source_teacher.get("manifest_sha256"),
            "role": "single_frozen_medical_teacher",
            "same_token_scoring": True,
        },
        "data": {
            "protocol_version": "ca-opd-data-v2",
            "prompt_manifest_path": source_data.get("opd_manifest_path"),
            "prompt_manifest_sha256": bindings["data_manifest_sha256"],
            "selection_rule": "seed42_sha256_rank_first2_per_source_per_step_v1",
            "allowed_roles": list(SOURCES),
            "prompt_only": True,
            "final_labels_allowed": False,
        },
        "protocol": {
            **protocol,
            "qualification_protocol_sha256": bindings["protocol_sha256"],
        },
        "generation": {
            "max_new_tokens": package["selected_response_length"],
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "full_support": True,
            "enable_thinking": False,
            "use_cache": True,
        },
        "qualification": qualification,
        "authorization": {
            "source": "artifact_derived_p4_7_length_qualification",
            "production_sampler_refresh_ready": True,
            "OPD_scoring_backend_ready": True,
            "B2_authorized": True,
            "B2_started": False,
        },
        "isolation": dict(package["isolation"]),
        "execution": {
            "optimizer_steps": bindings["estimated_steps"],
            "calibration_only": True,
            "automatically_start_b2": False,
            "automatically_run_idt": False,
            "automatically_run_sar": False,
            "automatically_run_ca_opd": False,
            "automatically_run_controller": False,
            "automatically_run_confirmation": False,
            "automatically_run_final": False,
        },
        "p4_7_start_gate": {
            "qualification_config_path": _P4_7_CONFIG_PATH.as_posix(),
            "formal_run_root": str(root),
            "source_length_run_id": source_length_run_id,
            "package_content_sha256": claimed,
            "requires_explicit_allow_b2_calibration": True,
        },
    }
    try:
        config_metadata = _atomic_json(staging / config_name, config)
        card = {
        "schema_version": 1,
        "run_id": config["run"]["run_id"],
        "status": "authorized_not_started",
        "config_path": config_name,
        "config_sha256": config_metadata["sha256"],
        "authorization_path": authorization_name,
        "authorization_content_sha256": claimed,
        "selected_response_length": package["selected_response_length"],
        "steps": 20,
        "requires_argument": "--allow-b2-calibration",
        "automatically_start": False,
        "B2_started": False,
        }
        card_metadata = _atomic_json(staging / card_name, card)
        authorization_metadata = _atomic_json(
            staging / authorization_name, dict(package)
        )
        os.replace(staging, output)
        directory = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "materialized_authorized_not_started",
        "output_dir": str(output),
        "files": {
            config_name: config_metadata,
            card_name: card_metadata,
            authorization_name: authorization_metadata,
        },
        "B2_authorized": True,
        "B2_started": False,
    }


__all__ = [
    "GenerationBackendV7",
    "ProductionLengthGpuRuntimeV7Error",
    "assert_b2_calibration_start_authorized",
    "build_b2_calibration_package",
    "execute_bounded_generation_plan",
    "materialize_b2_calibration_package",
]
