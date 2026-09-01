"""P4.8f evidence-first differential state machine.

Every component is atomically written, fsynced, reread, SHA-verified and
indexed before its pass/fail result can affect control flow.  Full token and
tensor payloads live only under the ignored diagnostic output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from src.opd.production_b2_objective_reducer_v2 import (
    B2ObjectiveReducerV2Error,
    CanonicalObjectiveResult,
    canonical_corrected_objective,
    canonical_fp64_replay_pair,
    locate_first_divergence,
)


STRICT_SCALAR_ATOL = 1e-6
STRICT_SCALAR_RTOL = 1e-6
STRICT_GRADIENT_ATOL = 2e-6
STRICT_GRADIENT_RTOL = 2e-5
OPERATIONAL_SCALAR_ATOL = 1e-5
OPERATIONAL_SCALAR_RTOL = 1e-4
OPERATIONAL_COSINE_MIN = 0.9999
OPERATIONAL_RELATIVE_L2_MAX = 1e-3

DIFFERENTIAL_STAGES = (
    "input_identity",
    "token_and_mask_identity",
    "legacy_forward",
    "new_forward",
    "q_comparison",
    "p_old_comparison",
    "teacher_comparison",
    "advantage_comparison",
    "ratio_clip_comparison",
    "per_token_objective_comparison",
    "hierarchical_reduction_comparison",
    "loss_comparison",
    "backward_gradient_comparison",
    "optimizer_delta_comparison",
    "lifecycle_count_comparison",
    "memory_and_cleanup",
    "final_decision",
)


class B2DifferentialEvidenceV2Error(RuntimeError):
    """Evidence writing or the frozen equivalence contracts failed."""


def _fail(message: str) -> None:
    raise B2DifferentialEvidenceV2Error(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B2DifferentialEvidenceV2Error(
            f"differential component is not canonical JSON: {type(error).__name__}"
        ) from error


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    reread = path.read_bytes()
    if reread != payload:
        _fail("differential atomic write reread differs")
    return {
        "path": path.as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": hashlib.sha256(reread).hexdigest(),
        "size_bytes": len(reread),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return _atomic_bytes(path, _canonical_bytes(value))


class DifferentialEvidenceWriterV2:
    """Append-only, ordered writer for the 17 registered stages."""

    def __init__(self, root: str | Path, *, run_id: str) -> None:
        self.root = Path(root).resolve()
        if self.root.exists() or self.root.is_symlink():
            _fail("differential evidence directory must be fresh")
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "components").mkdir()
        (self.root / "per_token").mkdir()
        (self.root / "gradients").mkdir()
        self.run_id = str(run_id)
        self._components: list[dict[str, Any]] = []
        self._binary_artifacts: list[dict[str, Any]] = []
        self._write_index()

    def _write_index(self) -> None:
        value = {
            "schema_version": 2,
            "artifact_kind": "b2_differential_evidence_index_v2",
            "run_id": self.run_id,
            "component_count": len(self._components),
            "components": self._components,
            "binary_artifact_count": len(self._binary_artifacts),
            "binary_artifacts": self._binary_artifacts,
            "ready": bool(
                len(self._components) == len(DIFFERENTIAL_STAGES)
                and self._components[-1].get("status") == "complete"
            ),
            "B2_formal_authorized": False,
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
        }
        _atomic_json(self.root / "evidence_index.json", value)

    def _atomic_component(
        self, *, stage: str, sequence: int, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _atomic_json(
            self.root / "components" / f"{sequence:03d}_{stage}.json",
            payload,
        )

    def record(
        self,
        stage: str,
        *,
        status: str,
        value: Any | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        sequence = len(self._components) + 1
        if sequence > len(DIFFERENTIAL_STAGES) or stage != DIFFERENTIAL_STAGES[
            sequence - 1
        ]:
            _fail("differential stage order differs")
        if status not in {"complete", "failed", "not_executed"}:
            _fail("differential component status is invalid")
        if status in {"complete", "failed"} and value is None:
            _fail("executed differential component cannot have a null value")
        if status in {"failed", "not_executed"} and not reason:
            _fail("failed/not-executed component requires a reason")
        payload: dict[str, Any] = {
            "schema_version": 2,
            "artifact_kind": "b2_differential_component_v2",
            "stage": stage,
            "status": status,
        }
        if status != "not_executed":
            payload["value"] = value
        if reason is not None:
            payload["reason"] = str(reason)
        metadata = self._atomic_component(
            stage=stage, sequence=sequence, payload=payload
        )
        reread = Path(metadata["absolute_path"]).read_bytes()
        if hashlib.sha256(reread).hexdigest() != metadata["sha256"]:
            _fail("differential component SHA reread failed")
        try:
            decoded = json.loads(reread)
        except json.JSONDecodeError as error:
            raise B2DifferentialEvidenceV2Error(
                "differential component JSON reread failed"
            ) from error
        if decoded != payload:
            _fail("differential component schema reread failed")
        entry = {
            **metadata,
            "stage": stage,
            "status": status,
        }
        self._components.append(entry)
        self._write_index()
        return payload

    def complete_remaining(self, *, reason: str) -> None:
        while len(self._components) < len(DIFFERENTIAL_STAGES):
            self.record(
                DIFFERENTIAL_STAGES[len(self._components)],
                status="not_executed",
                reason=reason,
            )

    def register_binary(self, metadata: Mapping[str, Any], *, kind: str) -> None:
        entry = {**dict(metadata), "artifact_kind": str(kind)}
        self._binary_artifacts.append(entry)
        self._write_index()

    def read_index(self) -> dict[str, Any]:
        value = json.loads(
            (self.root / "evidence_index.json").read_text(encoding="utf-8")
        )
        if not isinstance(value, dict):
            _fail("differential evidence index is not an object")
        return value

    def component_values(self) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for entry in self._components:
            values[entry["stage"]] = json.loads(
                Path(entry["absolute_path"]).read_text(encoding="utf-8")
            )
        return values


def _atomic_npz(
    writer: DifferentialEvidenceWriterV2,
    relative: str,
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
) -> dict[str, Any]:
    path = writer.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with np.load(path, allow_pickle=False) as reread:
        if set(reread.files) != set(arrays):
            _fail("differential NPZ field reread differs")
        for name, expected in arrays.items():
            if not np.array_equal(reread[name], expected, equal_nan=False):
                _fail("differential NPZ value reread differs")
    metadata = {
        "path": relative,
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    writer.register_binary(metadata, kind=kind)
    return metadata


def _atomic_torch_payload(
    writer: DifferentialEvidenceWriterV2,
    relative: str,
    value: Mapping[str, Tensor],
    *,
    kind: str,
) -> dict[str, Any]:
    path = writer.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in value.items()},
            temporary,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    reread = torch.load(path, map_location="cpu", weights_only=True)
    if set(reread) != set(value) or any(
        not torch.equal(reread[name], value[name].detach().cpu())
        for name in value
    ):
        _fail("differential gradient payload reread differs")
    metadata = {
        "path": relative,
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    writer.register_binary(metadata, kind=kind)
    return metadata


def _tensor(value: Any, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        _fail(f"{label} is not a tensor")
    return value.detach().cpu()


def _tensor_comparison(
    left: Tensor,
    right: Tensor,
    *,
    mask: Tensor | None = None,
    atol: float = STRICT_SCALAR_ATOL,
    rtol: float = STRICT_SCALAR_RTOL,
) -> dict[str, Any]:
    lhs = left.detach().to(dtype=torch.float64, device="cpu")
    rhs = right.detach().to(dtype=torch.float64, device="cpu")
    if lhs.shape != rhs.shape:
        return {
            "shape_equal": False,
            "left_shape": list(lhs.shape),
            "right_shape": list(rhs.shape),
            "finite": False,
            "pass": False,
        }
    selected_mask = (
        torch.ones_like(lhs, dtype=torch.bool)
        if mask is None
        else mask.detach().to(dtype=torch.bool, device="cpu")
    )
    if selected_mask.shape != lhs.shape or not bool(selected_mask.any()):
        _fail("comparison mask shape/nonempty contract failed")
    left_values = lhs[selected_mask]
    right_values = rhs[selected_mask]
    finite = bool(
        torch.isfinite(left_values).all()
        and torch.isfinite(right_values).all()
    )
    difference = (left_values - right_values).abs()
    denominator = torch.maximum(
        left_values.abs(), right_values.abs()
    ).clamp_min(1e-30)
    maximum_index = int(difference.argmax())
    flat_positions = selected_mask.nonzero(as_tuple=False)
    position = [int(value) for value in flat_positions[maximum_index].tolist()]
    return {
        "shape_equal": True,
        "shape": list(lhs.shape),
        "dtype_left": str(left.dtype).replace("torch.", ""),
        "dtype_right": str(right.dtype).replace("torch.", ""),
        "finite": finite,
        "finite_count": int(
            torch.isfinite(left_values).sum()
            + torch.isfinite(right_values).sum()
        ),
        "value_count": int(left_values.numel()),
        "max_abs_error": float(difference.max()),
        "max_relative_error": float((difference / denominator).max()),
        "rms_error": float(difference.square().mean().sqrt()),
        "max_error_position": position,
        "max_error_sample_index": position[0],
        "max_error_token_position": position[1] if len(position) > 1 else 0,
        "atol": float(atol),
        "rtol": float(rtol),
        "pass": bool(
            finite
            and torch.allclose(left_values, right_values, atol=atol, rtol=rtol)
        ),
    }


def _array_summary(value: Tensor, mask: Tensor) -> dict[str, Any]:
    selected = value.detach().to(dtype=torch.float64, device="cpu")[
        mask.to(dtype=torch.bool, device="cpu")
    ]
    finite = selected[torch.isfinite(selected)]
    if finite.numel() == 0:
        _fail("per-token summary contains no finite value")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "finite_count": int(finite.numel()),
        "value_count": int(selected.numel()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "p50": float(torch.quantile(finite, 0.50)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
    }


def _route_canonical(
    snapshot: Mapping[str, Any],
    *,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
) -> CanonicalObjectiveResult:
    arrays = snapshot.get("arrays")
    if not isinstance(arrays, Mapping):
        _fail("differential route arrays are absent")
    return canonical_corrected_objective(
        q_target_logprob=_tensor(arrays.get("q_target_logprob"), "q"),
        p_old_target_logprob=_tensor(
            arrays.get("p_old_target_logprob"), "p_old"
        ),
        teacher_target_logprob=_tensor(
            arrays.get("teacher_target_logprob"), "Teacher"
        ),
        correction_weight=_tensor(
            arrays.get("correction_weight"), "correction"
        ),
        valid_mask=_tensor(arrays.get("valid_mask"), "valid mask"),
        prompt_ids=prompt_ids,
        group_ids=group_ids,
        beta=beta,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def _npz_arrays(
    result: CanonicalObjectiveResult,
    snapshot: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    fields = {
        "q_target_logprob": result.q_target_logprob,
        "p_old_target_logprob": result.p_old_target_logprob,
        "teacher_target_logprob": result.teacher_target_logprob,
        "raw_advantage": result.raw_advantage,
        "scaled_advantage": result.scaled_advantage,
        "clipped_advantage": result.clipped_advantage,
        "raw_ppo_ratio": result.raw_ppo_ratio,
        "clipped_ratio": result.clipped_ratio,
        "unclipped_objective": result.unclipped_objective,
        "clipped_objective": result.clipped_objective,
        "selected_objective": result.selected_objective,
        "corrected_selected_objective": result.corrected_selected_objective,
        "clip_boundary_mask": result.clip_boundary_mask,
        "valid_mask": result.valid_mask,
    }
    arrays = snapshot.get("arrays")
    if isinstance(arrays, Mapping):
        for name in (
            "q_pre_target_logprob",
            "q_post_target_logprob",
            "rollout_behavior_logprob",
            "correction_weight",
        ):
            value = arrays.get(name)
            if isinstance(value, Tensor):
                fields[name] = value
    return {
        name: value.detach().cpu().numpy() for name, value in fields.items()
    }


def _scalar_comparison(left: float, right: float) -> dict[str, Any]:
    finite = math.isfinite(left) and math.isfinite(right)
    absolute = abs(left - right) if finite else math.inf
    relative = absolute / max(abs(left), abs(right), 1e-30)
    return {
        "legacy": left,
        "balanced": right,
        "finite": finite,
        "sign_equal": bool(
            finite and (left == 0.0 or right == 0.0 or math.copysign(1, left) == math.copysign(1, right))
        ),
        "absolute_error": absolute,
        "relative_error": relative,
        "strict_1e6_pass": bool(
            finite
            and absolute
            <= STRICT_SCALAR_ATOL + STRICT_SCALAR_RTOL * abs(left)
        ),
        "operational_pass": bool(
            finite
            and absolute
            <= OPERATIONAL_SCALAR_ATOL + OPERATIONAL_SCALAR_RTOL * abs(left)
        ),
    }


def _reduction_value(
    result: CanonicalObjectiveResult,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    reduction = result.reduction
    runtime = snapshot.get("runtime")
    ranges = runtime.get("chunk_ranges") if isinstance(runtime, Mapping) else []
    chunks: list[dict[str, Any]] = []
    if isinstance(ranges, list):
        for sample_index, sample_ranges in enumerate(ranges):
            if not isinstance(sample_ranges, list):
                continue
            for chunk_index, interval in enumerate(sample_ranges):
                if not (
                    isinstance(interval, list)
                    and len(interval) == 2
                    and all(isinstance(value, int) for value in interval)
                ):
                    continue
                start, end = interval
                mask = result.valid_mask[sample_index, start:end]
                values = result.corrected_selected_objective[
                    sample_index, start:end
                ]
                chunks.append(
                    {
                        "sample_index": sample_index,
                        "chunk_index": chunk_index,
                        "start": start,
                        "end": end,
                        "numerator": float(values[mask].sum().detach()),
                        "valid_count": int(mask.sum()),
                    }
                )
    return {
        "chunk_numerator_contract": "sum_valid_token_contributions_only",
        "chunks": chunks,
        "trajectory_sums": [float(value) for value in reduction.trajectory_sums.detach()],
        "trajectory_valid_counts": [
            int(value) for value in reduction.valid_token_counts.detach()
        ],
        "trajectory_means": [
            float(value) for value in reduction.trajectory_means.detach()
        ],
        "per_group": {
            f"{prompt}::{group}": float(value.detach())
            for (prompt, group), value in reduction.per_group.items()
        },
        "per_prompt": {
            prompt: float(value.detach())
            for prompt, value in reduction.per_prompt.items()
        },
        "per_prompt_scaled_objective": {
            prompt: float(value.detach()) / len(reduction.per_prompt)
            for prompt, value in reduction.per_prompt.items()
        },
        "scaled_prompt_sum": float(result.objective.detach()),
        "batch_objective": float(result.objective.detach()),
        "loss": float(result.loss.detach()),
        "accumulator_dtype": str(result.accumulator_dtype).replace("torch.", ""),
    }


def _vector_metrics(left: Tensor, right: Tensor) -> dict[str, Any]:
    lhs = left.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    rhs = right.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if lhs.shape != rhs.shape or lhs.numel() == 0:
        return {"shape_equal": False, "finite": False}
    difference = lhs - rhs
    left_norm = torch.linalg.vector_norm(lhs)
    right_norm = torch.linalg.vector_norm(rhs)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(float(left_norm), 1e-30)
    if float(left_norm) == 0.0 and float(right_norm) == 0.0:
        cosine = 1.0
    elif float(left_norm) == 0.0 or float(right_norm) == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(lhs, rhs) / (left_norm * right_norm))
    return {
        "shape_equal": True,
        "shape": list(left.shape),
        "finite": bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()),
        "legacy_norm": float(left_norm),
        "balanced_norm": float(right_norm),
        "max_abs_error": float(difference.abs().max()),
        "relative_l2": float(difference_norm) / denominator,
        "cosine_similarity": cosine,
        "strict_allclose_pass": bool(
            torch.allclose(
                lhs,
                rhs,
                atol=STRICT_GRADIENT_ATOL,
                rtol=STRICT_GRADIENT_RTOL,
            )
        ),
        "operational_pass": bool(
            cosine >= OPERATIONAL_COSINE_MIN
            and float(difference_norm) / denominator
            <= OPERATIONAL_RELATIVE_L2_MAX
        ),
    }


def _parameter_comparison(
    legacy: Mapping[str, Tensor], balanced: Mapping[str, Tensor]
) -> dict[str, Any]:
    names_equal = bool(legacy and set(legacy) == set(balanced))
    per_tensor: dict[str, Any] = {}
    if names_equal:
        for name in sorted(legacy):
            per_tensor[name] = {
                "name": name,
                **_vector_metrics(legacy[name], balanced[name]),
            }
    flattened_left = (
        torch.cat([legacy[name].detach().cpu().reshape(-1) for name in sorted(legacy)])
        if names_equal
        else torch.empty(0)
    )
    flattened_right = (
        torch.cat([balanced[name].detach().cpu().reshape(-1) for name in sorted(balanced)])
        if names_equal
        else torch.empty(0)
    )
    aggregate = (
        _vector_metrics(flattened_left, flattened_right)
        if names_equal
        else {"shape_equal": False, "finite": False, "operational_pass": False, "strict_allclose_pass": False}
    )
    return {
        "tensor_names_equal": names_equal,
        "tensor_count": len(per_tensor),
        "per_tensor": per_tensor,
        "aggregate": aggregate,
        "strict_pass": bool(
            names_equal
            and all(item.get("strict_allclose_pass") for item in per_tensor.values())
        ),
        "operational_pass": bool(
            names_equal and aggregate.get("operational_pass") is True
        ),
    }


def _runtime_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    runtime = snapshot.get("runtime")
    if not isinstance(runtime, Mapping):
        _fail("differential runtime facts are absent")
    fields = (
        "backbone_forward_count",
        "backbone_backward_count",
        "lm_head_forward_count",
        "lm_head_backward_count",
        "retain_graph_count",
        "zero_grad_count",
        "grad_clip_count",
        "optimizer_count",
        "scheduler_count",
        "export_count",
        "refresh_count",
        "policy_increment_count",
    )
    return {name: int(runtime[name]) for name in fields}


def _persist_token_payload(
    writer: DifferentialEvidenceWriterV2,
    legacy: Mapping[str, Any],
    balanced: Mapping[str, Any],
) -> None:
    value = {
        "schema_version": 2,
        "artifact_kind": "b2_differential_fixed_token_payload_v2",
        "legacy": legacy.get("token_payload"),
        "balanced": balanced.get("token_payload"),
        "raw_medical_text_persisted": False,
    }
    metadata = _atomic_json(writer.root / "fixed_token_ids.json", value)
    metadata["path"] = "fixed_token_ids.json"
    writer.register_binary(metadata, kind=value["artifact_kind"])


def _persist_blocked_cleanup_and_decision(
    writer: DifferentialEvidenceWriterV2,
    *,
    legacy: Mapping[str, Any],
    balanced: Mapping[str, Any],
    reason: str,
    first_divergence: str,
) -> None:
    """Seal known cleanup and a failed decision after an allowed early stop."""

    memory_index = DIFFERENTIAL_STAGES.index("memory_and_cleanup")
    while len(writer._components) < memory_index:
        writer.record(
            DIFFERENTIAL_STAGES[len(writer._components)],
            status="not_executed",
            reason=reason,
        )
    if len(writer._components) == memory_index:
        left = legacy.get("memory_and_cleanup")
        right = balanced.get("memory_and_cleanup")
        passed = bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.get("runtime_references_released") is True
            and right.get("runtime_references_released") is True
        )
        writer.record(
            "memory_and_cleanup",
            status="complete" if passed else "failed",
            value={
                "legacy": dict(left) if isinstance(left, Mapping) else {},
                "balanced": dict(right) if isinstance(right, Mapping) else {},
                "pass": passed,
            },
            reason=None if passed else "cleanup evidence is incomplete",
        )
    if len(writer._components) == DIFFERENTIAL_STAGES.index("final_decision"):
        writer.record(
            "final_decision",
            status="failed",
            value={
                "legacy_strict_1e6_pass": False,
                "algebraic_semantic_equivalence_pass": False,
                "gpu_bf16_operational_equivalence_pass": False,
                "first_divergence": first_divergence,
                "failures": [first_divergence],
                "can_enter_max_shape_canary": False,
                "B2_formal_authorized": False,
                "final_access_count": 0,
                "controller_access_count": 0,
                "confirmation_access_count": 0,
                "label_access_count": 0,
            },
            reason=reason,
        )


def compare_and_persist_differential_v2(
    *,
    legacy: Mapping[str, Any],
    balanced: Mapping[str, Any],
    evidence_dir: str | Path,
    prompt_ids: Sequence[str],
    group_ids: Sequence[str],
    beta: float,
    clip_low: float,
    clip_high: float,
    writer: DifferentialEvidenceWriterV2 | None = None,
) -> dict[str, Any]:
    """Persist every reachable stage, then enforce the two-layer contract."""

    evidence_root = Path(evidence_dir).resolve()
    if writer is None:
        writer = DifferentialEvidenceWriterV2(
            evidence_root, run_id="p4_8f_fixed_token_gpu_differential"
        )
    elif not (
        writer.root == evidence_root
        and writer.read_index().get("component_count") == 0
    ):
        _fail("precreated differential writer is nonempty or targets another root")
    failures: list[str] = []
    try:
        left_identity = legacy.get("input_identity")
        right_identity = balanced.get("input_identity")
        identity_pass = bool(
            isinstance(left_identity, Mapping)
            and isinstance(right_identity, Mapping)
            and left_identity.get("fixed_tokens_sha256")
            == right_identity.get("fixed_tokens_sha256")
            and left_identity.get("initial_adapter_sha256")
            == right_identity.get("initial_adapter_sha256")
            and left_identity.get("prompt_order") == right_identity.get("prompt_order")
        )
        identity_value = {
            "legacy_fixed_tokens_sha256": (
                left_identity.get("fixed_tokens_sha256")
                if isinstance(left_identity, Mapping)
                else "absent"
            ),
            "balanced_fixed_tokens_sha256": (
                right_identity.get("fixed_tokens_sha256")
                if isinstance(right_identity, Mapping)
                else "absent"
            ),
            "legacy_initial_adapter_sha256": (
                left_identity.get("initial_adapter_sha256")
                if isinstance(left_identity, Mapping)
                else "absent"
            ),
            "balanced_initial_adapter_sha256": (
                right_identity.get("initial_adapter_sha256")
                if isinstance(right_identity, Mapping)
                else "absent"
            ),
            "pass": identity_pass,
        }
        writer.record(
            "input_identity",
            status="complete" if identity_pass else "failed",
            value=identity_value,
            reason=None if identity_pass else "fixed token or initial adapter identity differs",
        )

        left_arrays = legacy.get("arrays")
        right_arrays = balanced.get("arrays")
        left_mask = (
            _tensor(left_arrays.get("valid_mask"), "legacy valid mask")
            if isinstance(left_arrays, Mapping)
            else torch.empty(0, dtype=torch.bool)
        )
        right_mask = (
            _tensor(right_arrays.get("valid_mask"), "balanced valid mask")
            if isinstance(right_arrays, Mapping)
            else torch.empty(0, dtype=torch.bool)
        )
        samples_equal = bool(
            isinstance(left_identity, Mapping)
            and isinstance(right_identity, Mapping)
            and left_identity.get("samples") == right_identity.get("samples")
        )
        token_identity_pass = bool(
            identity_pass
            and samples_equal
            and left_mask.shape == right_mask.shape
            and torch.equal(left_mask, right_mask)
        )
        token_value = {
            "samples": (
                list(left_identity.get("samples", []))
                if isinstance(left_identity, Mapping)
                else []
            ),
            "prompt_order": (
                list(left_identity.get("prompt_order", []))
                if isinstance(left_identity, Mapping)
                else []
            ),
            # Chunk plans are execution facts, not input identity: legacy uses
            # one full-vocabulary range while the balanced route uses exact
            # target-position ranges over the same token/mask payload.
            "legacy_chunk_plan": (
                list(left_identity.get("chunk_plan", []))
                if isinstance(left_identity, Mapping)
                else []
            ),
            "balanced_chunk_plan": (
                list(right_identity.get("chunk_plan", []))
                if isinstance(right_identity, Mapping)
                else []
            ),
            "valid_mask_shape": list(left_mask.shape),
            "valid_mask_equal": bool(
                left_mask.shape == right_mask.shape
                and torch.equal(left_mask, right_mask)
            ),
            "pass": token_identity_pass,
        }
        writer.record(
            "token_and_mask_identity",
            status="complete" if token_identity_pass else "failed",
            value=token_value,
            reason=None if token_identity_pass else "token_and_mask_identity differs",
        )
        _persist_token_payload(writer, legacy, balanced)
        if not token_identity_pass:
            _persist_blocked_cleanup_and_decision(
                writer,
                legacy=legacy,
                balanced=balanced,
                reason="blocked by token_and_mask_identity",
                first_divergence="identity_or_chunk_boundary",
            )
            _fail("differential equivalence failed at token_and_mask_identity")

        left_result = _route_canonical(
            legacy,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            beta=beta,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        right_result = _route_canonical(
            balanced,
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            beta=beta,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        for route_name, snapshot, result, stage in (
            ("legacy", legacy, left_result, "legacy_forward"),
            ("balanced", balanced, right_result, "new_forward"),
        ):
            runtime = snapshot.get("runtime")
            native = snapshot.get("native")
            complete = isinstance(runtime, Mapping) and isinstance(native, Mapping)
            value = {
                "route": route_name,
                "runtime": dict(runtime) if isinstance(runtime, Mapping) else {},
                "native": dict(native) if isinstance(native, Mapping) else {},
                "q_summary": _array_summary(result.q_target_logprob, result.valid_mask),
                "p_old_summary": _array_summary(result.p_old_target_logprob, result.valid_mask),
                "teacher_summary": _array_summary(result.teacher_target_logprob, result.valid_mask),
                "finite": bool(
                    torch.isfinite(result.q_target_logprob[result.valid_mask]).all()
                    and torch.isfinite(result.p_old_target_logprob[result.valid_mask]).all()
                    and torch.isfinite(result.teacher_target_logprob[result.valid_mask]).all()
                ),
            }
            writer.record(
                stage,
                status="complete" if complete and value["finite"] else "failed",
                value=value,
                reason=None if complete and value["finite"] else f"{route_name} forward evidence is incomplete/nonfinite",
            )
            if not (complete and value["finite"]):
                _persist_blocked_cleanup_and_decision(
                    writer,
                    legacy=legacy,
                    balanced=balanced,
                    reason=f"blocked by {stage}",
                    first_divergence="forward_or_scorer",
                )
                _fail(f"differential equivalence failed at {stage}")

        left_npz = _atomic_npz(
            writer,
            "per_token/legacy.npz",
            _npz_arrays(left_result, legacy),
            kind="b2_differential_per_token_route_v2",
        )
        right_npz = _atomic_npz(
            writer,
            "per_token/balanced.npz",
            _npz_arrays(right_result, balanced),
            kind="b2_differential_per_token_route_v2",
        )
        left_gradients = legacy.get("gradients")
        right_gradients = balanced.get("gradients")
        left_deltas = legacy.get("deltas")
        right_deltas = balanced.get("deltas")
        if not all(
            isinstance(value, Mapping)
            for value in (left_gradients, right_gradients, left_deltas, right_deltas)
        ):
            _fail("differential gradient/delta payload is absent")
        left_gradient_payload = _atomic_torch_payload(
            writer,
            "gradients/legacy_gradients.pt",
            left_gradients,
            kind="b2_differential_full_gradient_payload_v2",
        )
        right_gradient_payload = _atomic_torch_payload(
            writer,
            "gradients/balanced_gradients.pt",
            right_gradients,
            kind="b2_differential_full_gradient_payload_v2",
        )
        left_delta_payload = _atomic_torch_payload(
            writer,
            "gradients/legacy_deltas.pt",
            left_deltas,
            kind="b2_differential_full_delta_payload_v2",
        )
        right_delta_payload = _atomic_torch_payload(
            writer,
            "gradients/balanced_deltas.pt",
            right_deltas,
            kind="b2_differential_full_delta_payload_v2",
        )

        comparisons: dict[str, dict[str, Any]] = {}
        left_snapshot_arrays = legacy["arrays"]
        right_snapshot_arrays = balanced["arrays"]
        q_backward = _tensor_comparison(
            left_result.q_target_logprob,
            right_result.q_target_logprob,
            mask=left_result.valid_mask,
        )
        left_q_pre = left_snapshot_arrays.get(
            "q_pre_target_logprob", left_result.q_target_logprob
        )
        right_q_pre = right_snapshot_arrays.get(
            "q_pre_target_logprob", right_result.q_target_logprob
        )
        left_q_post = left_snapshot_arrays.get(
            "q_post_target_logprob", left_result.q_target_logprob
        )
        right_q_post = right_snapshot_arrays.get(
            "q_post_target_logprob", right_result.q_target_logprob
        )
        q_pre = _tensor_comparison(
            _tensor(left_q_pre, "legacy q_pre"),
            _tensor(right_q_pre, "balanced q_pre"),
            mask=left_result.valid_mask,
        )
        q_post = _tensor_comparison(
            _tensor(left_q_post, "legacy q_post"),
            _tensor(right_q_post, "balanced q_post"),
            mask=left_result.valid_mask,
        )
        q_value = {
            **q_backward,
            "pre_update_inference": q_pre,
            "backward_training": q_backward,
            "post_update_inference": q_post,
            # Post-update q is diagnostic for delta propagation.  The frozen
            # q gate applies before/update-backward; optimizer delta has its
            # own operational vector contract.
            "post_update_is_diagnostic_only": True,
            "pass": bool(q_pre["pass"] and q_backward["pass"]),
        }
        comparisons["q_comparison"] = q_value
        writer.record(
            "q_comparison",
            status="complete" if q_value["pass"] else "failed",
            value=q_value,
            reason=(
                None
                if q_value["pass"]
                else "q pre-update/backward exceeds strict tolerance"
            ),
        )
        if not q_value["pass"]:
            failures.append("q_comparison")

        for stage, left_value, right_value in (
            ("p_old_comparison", left_result.p_old_target_logprob, right_result.p_old_target_logprob),
            ("teacher_comparison", left_result.teacher_target_logprob, right_result.teacher_target_logprob),
            ("advantage_comparison", left_result.raw_advantage, right_result.raw_advantage),
        ):
            value = _tensor_comparison(
                left_value, right_value, mask=left_result.valid_mask
            )
            comparisons[stage] = value
            writer.record(
                stage,
                status="complete" if value["pass"] else "failed",
                value=value,
                reason=None if value["pass"] else f"{stage} exceeds strict tolerance",
            )
            if not value["pass"]:
                failures.append(stage)

        ratio = {
            "raw_ratio": _tensor_comparison(
                left_result.raw_ppo_ratio,
                right_result.raw_ppo_ratio,
                mask=left_result.valid_mask,
            ),
            "clipped_ratio": _tensor_comparison(
                left_result.clipped_ratio,
                right_result.clipped_ratio,
                mask=left_result.valid_mask,
            ),
            "clip_boundary_crossing_count": int(
                torch.logical_xor(
                    left_result.clip_boundary_mask,
                    right_result.clip_boundary_mask,
                )[left_result.valid_mask].sum()
            ),
        }
        ratio["pass"] = bool(
            ratio["raw_ratio"]["pass"]
            and ratio["clipped_ratio"]["pass"]
            and ratio["clip_boundary_crossing_count"] == 0
        )
        comparisons["ratio_clip_comparison"] = ratio
        writer.record(
            "ratio_clip_comparison",
            status="complete" if ratio["pass"] else "failed",
            value=ratio,
            reason=None if ratio["pass"] else "ratio/clip semantics differ",
        )
        if not ratio["pass"]:
            failures.append("ratio_clip_comparison")

        per_token = {
            "unclipped": _tensor_comparison(
                left_result.unclipped_objective,
                right_result.unclipped_objective,
                mask=left_result.valid_mask,
            ),
            "clipped": _tensor_comparison(
                left_result.clipped_objective,
                right_result.clipped_objective,
                mask=left_result.valid_mask,
            ),
            "selected": _tensor_comparison(
                left_result.corrected_selected_objective,
                right_result.corrected_selected_objective,
                mask=left_result.valid_mask,
            ),
            "legacy_npz": left_npz,
            "balanced_npz": right_npz,
        }
        per_token["pass"] = all(
            per_token[name]["pass"] for name in ("unclipped", "clipped", "selected")
        )
        comparisons["per_token_objective_comparison"] = per_token
        writer.record(
            "per_token_objective_comparison",
            status="complete" if per_token["pass"] else "failed",
            value=per_token,
            reason=None if per_token["pass"] else "per-token formula/clip/dtype differs",
        )
        if not per_token["pass"]:
            failures.append("per_token_objective_comparison")

        native_left = legacy["native"]
        native_right = balanced["native"]
        replay = canonical_fp64_replay_pair(
            legacy={
                **dict(legacy["arrays"]),
                "native_objective": float(native_left["objective"]),
            },
            balanced={
                **dict(balanced["arrays"]),
                "native_objective": float(native_right["objective"]),
            },
            prompt_ids=prompt_ids,
            group_ids=group_ids,
            beta=beta,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        reduction_cross = _scalar_comparison(
            float(left_result.objective.detach()),
            float(right_result.objective.detach()),
        )
        trajectory_cross = _tensor_comparison(
            left_result.reduction.trajectory_means,
            right_result.reduction.trajectory_means,
        )
        group_cross = {
            f"{prompt}::{group}": _scalar_comparison(
                float(left_result.reduction.per_group[(prompt, group)].detach()),
                float(right_result.reduction.per_group[(prompt, group)].detach()),
            )
            for prompt, group in left_result.reduction.per_group
        }
        prompt_cross = {
            prompt: _scalar_comparison(
                float(left_result.reduction.per_prompt[prompt].detach()),
                float(right_result.reduction.per_prompt[prompt].detach()),
            )
            for prompt in left_result.reduction.per_prompt
        }
        native_left_vs_canonical = _scalar_comparison(
            float(native_left["objective"]),
            float(left_result.objective.detach()),
        )
        native_right_vs_canonical = _scalar_comparison(
            float(native_right["objective"]),
            float(right_result.objective.detach()),
        )
        hierarchical_pass = bool(
            reduction_cross["strict_1e6_pass"]
            and trajectory_cross["pass"]
            and all(value["strict_1e6_pass"] for value in group_cross.values())
            and all(value["strict_1e6_pass"] for value in prompt_cross.values())
            and native_left_vs_canonical["strict_1e6_pass"]
            and native_right_vs_canonical["strict_1e6_pass"]
            and replay["canonical_cross_path_error"]
            <= replay["objective_error_bound"] + 1e-12
        )
        hierarchical = {
            "legacy": _reduction_value(left_result, legacy),
            "balanced": _reduction_value(right_result, balanced),
            "cross_path": {
                "trajectory": trajectory_cross,
                "group": group_cross,
                "prompt": prompt_cross,
                "batch": reduction_cross,
                "legacy_native_vs_canonical": native_left_vs_canonical,
                "balanced_native_vs_canonical": native_right_vs_canonical,
            },
            "fp64_replay": replay,
            "pass": hierarchical_pass,
        }
        comparisons["hierarchical_reduction_comparison"] = hierarchical
        writer.record(
            "hierarchical_reduction_comparison",
            status="complete" if hierarchical_pass else "failed",
            value=hierarchical,
            reason=None if hierarchical_pass else "hierarchical/native reduction differs",
        )
        if not hierarchical_pass:
            failures.append("hierarchical_reduction_comparison")

        pre_update_scalar_fields = (
            "objective",
            "loss",
            "backward_loss",
        )
        gradient_norm_probe_fields = (
            "grad_norm_before_clip",
            "grad_norm",
        )
        post_update_probe_fields = (
            "post_update_objective",
            "post_update_loss",
        )
        scalar_fields = (
            *pre_update_scalar_fields,
            *gradient_norm_probe_fields,
            *post_update_probe_fields,
        )
        scalar_values = {
            name: _scalar_comparison(
                float(native_left[name]), float(native_right[name])
            )
            for name in scalar_fields
        }
        # Preserve the historical P4.8e strict scalar surface (pre-update
        # objective/loss plus gradient norms), but do not add the separately
        # requested post-update probe to that historical result.
        scalar_strict = all(
            scalar_values[name]["strict_1e6_pass"]
            for name in (*pre_update_scalar_fields, *gradient_norm_probe_fields)
        )
        scalar_operational = all(
            scalar_values[name]["operational_pass"]
            and scalar_values[name]["sign_equal"]
            for name in pre_update_scalar_fields
        )
        post_update_operational = all(
            scalar_values[name]["operational_pass"]
            and scalar_values[name]["sign_equal"]
            for name in post_update_probe_fields
        )
        loss_value = {
            "fields": scalar_values,
            "pre_update": {
                "fields": {
                    name: scalar_values[name]
                    for name in pre_update_scalar_fields
                },
                "gating": True,
                "operational_pass": scalar_operational,
            },
            "gradient_norm_probe": {
                "fields": {
                    name: scalar_values[name]
                    for name in gradient_norm_probe_fields
                },
                "gating": False,
            },
            "post_update_probe": {
                "fields": {
                    name: scalar_values[name]
                    for name in post_update_probe_fields
                },
                "gating": False,
                "operational_pass": post_update_operational,
            },
            "strict_1e6_pass": scalar_strict,
            "operational_pass": scalar_operational,
        }
        comparisons["loss_comparison"] = loss_value
        writer.record(
            "loss_comparison",
            status="complete" if scalar_operational else "failed",
            value=loss_value,
            reason=None if scalar_operational else "objective/loss operational contract failed",
        )
        if not scalar_operational:
            failures.append("loss_comparison")

        gradient_value = _parameter_comparison(left_gradients, right_gradients)
        gradient_value.update(
            {
                "legacy_payload": left_gradient_payload,
                "balanced_payload": right_gradient_payload,
                "legacy_teacher_gradient_tensor_count": int(
                    legacy.get("teacher_gradient_tensor_count", -1)
                ),
                "balanced_teacher_gradient_tensor_count": int(
                    balanced.get("teacher_gradient_tensor_count", -1)
                ),
                "legacy_base_gradient_tensor_count": int(
                    legacy.get("base_gradient_tensor_count", -1)
                ),
                "balanced_base_gradient_tensor_count": int(
                    balanced.get("base_gradient_tensor_count", -1)
                ),
            }
        )
        ownership_pass = all(
            gradient_value[name] == 0
            for name in (
                "legacy_teacher_gradient_tensor_count",
                "balanced_teacher_gradient_tensor_count",
                "legacy_base_gradient_tensor_count",
                "balanced_base_gradient_tensor_count",
            )
        )
        gradient_value["ownership_pass"] = ownership_pass
        writer.record(
            "backward_gradient_comparison",
            status="complete" if gradient_value["operational_pass"] and ownership_pass else "failed",
            value=gradient_value,
            reason=None if gradient_value["operational_pass"] and ownership_pass else "gradient/ownership operational contract failed",
        )
        if not (gradient_value["operational_pass"] and ownership_pass):
            failures.append("backward_gradient_comparison")

        delta_value = _parameter_comparison(left_deltas, right_deltas)
        delta_value.update(
            {
                "legacy_payload": left_delta_payload,
                "balanced_payload": right_delta_payload,
            }
        )
        delta_value["legacy_nonzero_update_tensor_count"] = sum(
            int(torch.count_nonzero(value)) > 0 for value in left_deltas.values()
        )
        delta_value["balanced_nonzero_update_tensor_count"] = sum(
            int(torch.count_nonzero(value)) > 0 for value in right_deltas.values()
        )
        writer.record(
            "optimizer_delta_comparison",
            status="complete" if delta_value["operational_pass"] else "failed",
            value=delta_value,
            reason=None if delta_value["operational_pass"] else "optimizer delta operational contract failed",
        )
        if not delta_value["operational_pass"]:
            failures.append("optimizer_delta_comparison")

        left_counts = _runtime_counts(legacy)
        right_counts = _runtime_counts(balanced)
        lifecycle_fields = (
            "zero_grad_count",
            "grad_clip_count",
            "optimizer_count",
            "scheduler_count",
            "export_count",
            "refresh_count",
            "policy_increment_count",
        )
        lifecycle_pass = bool(
            all(left_counts[name] == right_counts[name] == 1 for name in lifecycle_fields)
            # The legacy full-batch route enters the backbone graph once for
            # the four-prompt batch.  The memory-balanced route enters it once
            # per prompt.  Both are expected facts, not equal call counts.
            and left_counts["backbone_backward_count"] in {1, 4}
            and right_counts["backbone_backward_count"] == 4
            and left_counts["retain_graph_count"] == 0
            and right_counts["retain_graph_count"] == 0
        )
        lifecycle_value = {
            "legacy": left_counts,
            "balanced": right_counts,
            "strict_equal_once": lifecycle_pass,
        }
        writer.record(
            "lifecycle_count_comparison",
            status="complete" if lifecycle_pass else "failed",
            value=lifecycle_value,
            reason=None if lifecycle_pass else "lifecycle count contract differs",
        )
        if not lifecycle_pass:
            failures.append("lifecycle_count_comparison")

        memory_left = legacy.get("memory_and_cleanup")
        memory_right = balanced.get("memory_and_cleanup")
        memory_pass = bool(
            isinstance(memory_left, Mapping)
            and isinstance(memory_right, Mapping)
            and memory_left.get("runtime_references_released") is True
            and memory_right.get("runtime_references_released") is True
        )
        memory_value = {
            "legacy": dict(memory_left) if isinstance(memory_left, Mapping) else {},
            "balanced": dict(memory_right) if isinstance(memory_right, Mapping) else {},
            "pass": memory_pass,
        }
        writer.record(
            "memory_and_cleanup",
            status="complete" if memory_pass else "failed",
            value=memory_value,
            reason=None if memory_pass else "memory/cleanup evidence differs",
        )
        if not memory_pass:
            failures.append("memory_and_cleanup")

        signals = {
            "token_identity_pass": token_identity_pass,
            "q_pass": comparisons["q_comparison"]["pass"],
            "frozen_scorer_pass": bool(
                comparisons["p_old_comparison"]["pass"]
                and comparisons["teacher_comparison"]["pass"]
            ),
            "per_token_objective_pass": per_token["pass"],
            "hierarchical_reduction_pass": hierarchical_pass,
            "native_scalar_pass": scalar_operational,
            "canonical_within_error_bound": bool(
                replay["canonical_cross_path_error"]
                <= replay["objective_error_bound"] + 1e-12
            ),
            "gradient_pass": bool(
                gradient_value["operational_pass"] and ownership_pass
            ),
            "delta_pass": bool(delta_value["operational_pass"]),
        }
        legacy_strict = bool(
            comparisons["q_comparison"]["pass"]
            and comparisons["p_old_comparison"]["pass"]
            and comparisons["teacher_comparison"]["pass"]
            and scalar_strict
            and gradient_value["strict_pass"]
            and delta_value["strict_pass"]
            and lifecycle_pass
        )
        algebraic = bool(
            token_identity_pass
            and per_token["pass"]
            and hierarchical_pass
            and signals["canonical_within_error_bound"]
        )
        operational = bool(
            algebraic
            and all(
                comparisons[name]["pass"]
                for name in (
                    "q_comparison",
                    "p_old_comparison",
                    "teacher_comparison",
                )
            )
            and scalar_operational
            and gradient_value["operational_pass"]
            and delta_value["operational_pass"]
            and ownership_pass
            and lifecycle_pass
            and memory_pass
        )
        final_value = {
            "legacy_strict_1e6_pass": legacy_strict,
            "algebraic_semantic_equivalence_pass": algebraic,
            "gpu_bf16_operational_equivalence_pass": operational,
            "first_divergence": locate_first_divergence(signals),
            "failures": failures,
            "strict_tolerance": {
                "scalar_atol": STRICT_SCALAR_ATOL,
                "scalar_rtol": STRICT_SCALAR_RTOL,
                "gradient_atol": STRICT_GRADIENT_ATOL,
                "gradient_rtol": STRICT_GRADIENT_RTOL,
            },
            "operational_tolerance": {
                "scalar_atol": OPERATIONAL_SCALAR_ATOL,
                "scalar_rtol": OPERATIONAL_SCALAR_RTOL,
                "gradient_cosine_min": OPERATIONAL_COSINE_MIN,
                "gradient_relative_l2_max": OPERATIONAL_RELATIVE_L2_MAX,
                "delta_cosine_min": OPERATIONAL_COSINE_MIN,
                "delta_relative_l2_max": OPERATIONAL_RELATIVE_L2_MAX,
            },
            "can_enter_max_shape_canary": bool(algebraic and operational),
            "B2_formal_authorized": False,
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
        }
        writer.record(
            "final_decision",
            status="complete" if algebraic and operational else "failed",
            value=final_value,
            reason=None if algebraic and operational else "two-layer equivalence contract failed",
        )
        report = {
            "schema_version": 2,
            "artifact_kind": "p4_8f_gpu_differential_report_v2",
            "status": "passed" if algebraic and operational else "failed",
            "components": writer.component_values(),
            "evidence_index_sha256_before_report_registration": hashlib.sha256(
                (writer.root / "evidence_index.json").read_bytes()
            ).hexdigest(),
            **final_value,
        }
        report_metadata = _atomic_json(writer.root / "comparison.json", report)
        report_metadata["path"] = "comparison.json"
        writer.register_binary(
            report_metadata, kind="p4_8f_gpu_differential_report_v2"
        )
        if algebraic and operational:
            ready = {
                "schema_version": 2,
                "artifact_kind": "p4_8f_gpu_differential_ready_v2",
                "algebraic_semantic_equivalence_pass": True,
                "gpu_bf16_operational_equivalence_pass": True,
                "can_enter_max_shape_canary": True,
                "B2_formal_authorized": False,
            }
            _atomic_json(writer.root / "ready.json", ready)
            return report
        _fail("differential equivalence contracts failed")
    except B2DifferentialEvidenceV2Error:
        try:
            _persist_blocked_cleanup_and_decision(
                writer,
                legacy=legacy,
                balanced=balanced,
                reason="blocked by evidence writer or differential runtime failure",
                first_divergence="runtime_or_evidence_writer",
            )
        except B2DifferentialEvidenceV2Error:
            # A failing storage primitive cannot be used to durably describe
            # its own failure.  Never manufacture ready state in that case.
            pass
        raise
    except (B2ObjectiveReducerV2Error, KeyError, TypeError, ValueError) as error:
        _persist_blocked_cleanup_and_decision(
            writer,
            legacy=legacy,
            balanced=balanced,
            reason=f"blocked by runtime/schema error: {type(error).__name__}",
            first_divergence="runtime_or_schema",
        )
        raise B2DifferentialEvidenceV2Error(
            f"differential evidence/schema failure: {type(error).__name__}:{error}"
        ) from error


def persist_precomparison_failure_v2(
    writer: DifferentialEvidenceWriterV2,
    *,
    error: BaseException,
    first_divergence: str = "runtime_projection_or_session",
) -> None:
    """Durably seal all 17 stages when execution fails before comparison."""

    if writer.read_index().get("component_count") != 0:
        _fail("precomparison failure writer is not empty")
    reason = (
        "blocked before route comparison: "
        f"{type(error).__name__}:{str(error) or 'no error message'}"
    )
    for stage in DIFFERENTIAL_STAGES[:-1]:
        writer.record(stage, status="not_executed", reason=reason)
    writer.record(
        "final_decision",
        status="failed",
        value={
            "legacy_strict_1e6_pass": False,
            "algebraic_semantic_equivalence_pass": False,
            "gpu_bf16_operational_equivalence_pass": False,
            "first_divergence": first_divergence,
            "failures": [first_divergence],
            "can_enter_max_shape_canary": False,
            "B2_formal_authorized": False,
            "final_access_count": 0,
            "controller_access_count": 0,
            "confirmation_access_count": 0,
            "label_access_count": 0,
        },
        reason=reason,
    )


__all__ = [
    "B2DifferentialEvidenceV2Error",
    "DIFFERENTIAL_STAGES",
    "DifferentialEvidenceWriterV2",
    "OPERATIONAL_COSINE_MIN",
    "OPERATIONAL_RELATIVE_L2_MAX",
    "OPERATIONAL_SCALAR_ATOL",
    "OPERATIONAL_SCALAR_RTOL",
    "STRICT_GRADIENT_ATOL",
    "STRICT_GRADIENT_RTOL",
    "STRICT_SCALAR_ATOL",
    "STRICT_SCALAR_RTOL",
    "compare_and_persist_differential_v2",
    "persist_precomparison_failure_v2",
]
