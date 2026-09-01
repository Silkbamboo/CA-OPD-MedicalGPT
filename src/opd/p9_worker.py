"""P9 formal worker with decision-bound resume and accepted-commit counting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping
import argparse
import yaml

from src.opd.p9_adaptive_dose_protocol import P9ProtocolError


ALLOWED_TARGETS = frozenset({200, 240, 300})
REPO = Path(__file__).resolve().parents[2]
CHECKED_IN_CONFIG = REPO / "configs/runs/qwen3_4b_b2_p9_adaptive_dose.yaml"
P7_PACKAGE = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-formal-p5-1-v2-2-r1-package"
)
P7_STEP120 = Path(
    "artifacts/outputs/"
    "qwen3-4b-b2-medical-opd-formal-p5-1-v2-2-r1-seed42/"
    "formal_checkpoints/step_120"
)


class P9TrainingHealthFailure(RuntimeError):
    """A committed in-memory candidate failed a terminal historical health gate."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _bind_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise P9ProtocolError(f"P9 authoritative artifact drifted: {path.name}")
        return
    if path.exists() or path.is_symlink():
        raise P9ProtocolError(f"P9 authoritative artifact target is unsafe: {path.name}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit_p9_launch_assets(p7_package: Path) -> dict[str, Any]:
    from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256

    package = Path(p7_package).resolve()
    if package != P7_PACKAGE or p7_package.is_symlink():
        raise P9ProtocolError("P9 P7 package path differs")
    try:
        index = json.loads((package / "package_index.json").read_text(encoding="utf-8"))
        config = json.loads((package / "formal_b2_config.json").read_text(encoding="utf-8"))
        schedule = json.loads((package / "prompt_schedule.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P9ProtocolError("P9 P7 package is unreadable") from error
    exact_index = {
        "package_content_sha256": "c21ca9acec85bb72014ddfc48b5cf9079f680807cbfed348fe5ce1cc619583e1",
        "config_sha256": "130c91b300aab30d6bbbbf7f7893d77fa7c636c361b01a8dd4fc948f92c44835",
        "manifest_sha256": "9f1d096d06b635737e1b90be3b92d6de32fd64b03fbcd97813e42d0a2ee88a99",
        "schedule_file_sha256": "5ef4eef22acc7d291d1ebd9c4916313b8ff80613e81bc7cb8dbc5af79aa32fb5",
        "schedule_semantic_sha256": "ddba16637318580a9f31a938da14d7d6d59e49e50046f3f1faebc1ef38e6382c",
    }
    if any(index.get(key) != value for key, value in exact_index.items()):
        raise P9ProtocolError("P9 P7 package index identity differs")
    for name, descriptor in index.get("files", {}).items():
        path = package / name
        if not (
            path.is_file() and not path.is_symlink()
            and _sha_file(path) == descriptor.get("sha256")
            and path.stat().st_size == int(descriptor.get("size_bytes", -1))
        ):
            raise P9ProtocolError(f"P9 P7 package file differs: {name}")
    if not (
        _sha_file(package / "formal_b2_config.json") == exact_index["config_sha256"]
        and _sha_file(package / "prompt_schedule.json") == exact_index["schedule_file_sha256"]
        and schedule.get("schedule_sha256") == exact_index["schedule_semantic_sha256"]
    ):
        raise P9ProtocolError("P9 P7 config/schedule identity differs")
    model = config.get("model", {})
    teacher = config.get("teacher", {})
    base_manifest = Path(str(model.get("base_manifest_path", "")))
    teacher_manifest = Path(str(teacher.get("manifest_path", "")))
    adapter = Path(str(teacher.get("adapter_path", "")))
    exact_model = {
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "base_manifest_sha256": "c796c078afd35849b59017582eb7dd0e1553be43bdbd7ce0eed441fda889a213",
    }
    exact_teacher = {
        "adapter_sha256": "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2",
        "adapter_weight_sha256": "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63",
        "manifest_sha256": "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67",
    }
    if not (
        all(model.get(key) == value for key, value in exact_model.items())
        and base_manifest.is_file()
        and _sha_file(base_manifest) == exact_model["base_manifest_sha256"]
        and all(teacher.get(key) == value for key, value in exact_teacher.items())
        and teacher_manifest.is_file()
        and _sha_file(teacher_manifest) == exact_teacher["manifest_sha256"]
        and adapter.is_dir()
        and _ordered_adapter_sha256(adapter) == exact_teacher["adapter_sha256"]
        and _sha_file(adapter / "adapter_model.safetensors")
        == exact_teacher["adapter_weight_sha256"]
    ):
        raise P9ProtocolError("P9 Base/Teacher launch identity differs")
    compatibility = REPO / "reports/p9_compatibility_audit.json"
    resume_audit = REPO / "reports/p9_step120_resume_audit.json"
    try:
        compatibility_value = json.loads(compatibility.read_text(encoding="utf-8"))
        resume_value = json.loads(resume_audit.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P9ProtocolError("P9 launch audit artifacts are absent") from error
    if not (
        compatibility_value.get("status") == "cpu_compatibility_passed_gpu_reload_pending"
        and resume_value.get("status") == "cpu_resume_audit_passed_gpu_reload_pending"
        and resume_value.get("checkpoint") == str(P7_STEP120)
    ):
        raise P9ProtocolError("P9 launch audit artifact status differs")
    return {
        "passed": True,
        "p7_package_content_sha256": exact_index["package_content_sha256"],
        "base_manifest_sha256": exact_model["base_manifest_sha256"],
        "teacher_ordered_sha256": exact_teacher["adapter_sha256"],
        "teacher_weight_sha256": exact_teacher["adapter_weight_sha256"],
        "teacher_manifest_sha256": exact_teacher["manifest_sha256"],
        "compatibility_audit_sha256": _sha_file(compatibility),
        "resume_audit_sha256": _sha_file(resume_audit),
        "final_access_count": 0,
    }


def _attempt_intent_path(output: Path, *, step: int, variant: int) -> Path:
    return (
        Path(output) / "transaction_intents"
        / f"step_{step:03d}_variant_{variant:02d}.json"
    )


def begin_p9_attempt_intent(
    output: Path,
    *,
    step: int,
    variant: int,
    rows: list[Mapping[str, Any]],
) -> Path:
    path = _attempt_intent_path(output, step=step, variant=variant)
    if path.exists() or path.is_symlink():
        raise P9ProtocolError("P9 attempt intent already exists")
    _atomic_json(path, {
        "schema_version": 1,
        "artifact_kind": "p9_attempt_intent",
        "attempted_optimizer_step": step,
        "reserve_variant": variant,
        "pre_policy_version": step - 1,
        "sample_ids": [row["sample_id"] for row in rows],
        "content_hashes": [row["content_hash"] for row in rows],
        "source_roles": [row["target_role"] for row in rows],
        "outcome": "prepared",
        "final_access_count": 0,
    })
    return path


def complete_p9_attempt_intent(
    path: Path,
    *,
    outcome: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("outcome") != "prepared":
        raise P9ProtocolError("P9 attempt intent is not prepared")
    descriptors = []
    for field in ("kernel_rejection_artifacts", "generation_health_artifacts"):
        for raw in evidence.get(field, []):
            artifact = Path(str(raw))
            if artifact.is_file():
                descriptors.append({"path": str(artifact), "sha256": _sha_file(artifact)})
    value.update({
        "outcome": outcome,
        "evidence_sha256": _canonical_sha(evidence),
        "evidence": dict(evidence),
        "artifact_descriptors": descriptors,
    })
    _atomic_json(Path(path), value)
    return value


def recover_orphan_scientific_intent(output: Path, *, start_step: int) -> dict[str, Any]:
    """Close the kernel-artifact→P9-ledger crash window without replaying a bad batch."""

    output = Path(output)
    prepared = []
    claimed: set[tuple[str, str]] = set()
    scientific_intents: list[dict[str, Any]] = []
    intent_root = output / "transaction_intents"
    for path in sorted(intent_root.glob("*.json")) if intent_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        for descriptor in value.get("artifact_descriptors", []):
            claimed.add((str(descriptor.get("path")), str(descriptor.get("sha256"))))
        if (
            int(value.get("attempted_optimizer_step", -1)) > start_step
            and value.get("outcome") == "prepared"
        ):
            prepared.append((path, value))
        if value.get("outcome") in {
            "scientific_health_rejection",
            "scientific_health_rejection_recovered",
        }:
            scientific_intents.append(value)

    def backfill_ledgers() -> int:
        ledger = output / "transaction_ledger.jsonl"
        existing = set()
        if ledger.is_file():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("artifact_kind") == "p9_rejected_attempt_rollback":
                    existing.add((
                        int(row.get("attempted_optimizer_step", -1)),
                        int(row.get("reserve_variant", -1)),
                    ))
        added = 0
        for intent in scientific_intents:
            key = (
                int(intent.get("attempted_optimizer_step", -1)),
                int(intent.get("reserve_variant", -1)),
            )
            if key in existing:
                continue
            evidence = dict(intent.get("evidence", {}))
            _append_jsonl(ledger, {
                **evidence,
                "artifact_kind": "p9_rejected_attempt_rollback",
                "attempted_optimizer_step": key[0],
                "reserve_variant": key[1],
                "failure_classification": "scientific_health_rejection",
                "recovered_ledger_from_intent": True,
                "final_access_count": 0,
            })
            existing.add(key)
            added += 1
        return added

    backfilled = backfill_ledgers()
    if not prepared:
        return {"recovered": False, "ledger_rows_backfilled": backfilled}
    if len(prepared) != 1:
        raise P9ProtocolError("P9 has multiple orphan prepared intents")
    intent_path, intent_value = prepared[0]
    orphan_step = int(intent_value["attempted_optimizer_step"])

    scientific: list[Path] = []
    for path in sorted((output / "rejected_updates_v2").glob("attempt_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rollback = value.get("rollback")
        descriptor = (str(path), _sha_file(path))
        candidate_rollback = isinstance(rollback, Mapping) and (
            (
                rollback.get("rollback_verified") is True
                and rollback.get("cpu_rng_restored") is True
                and rollback.get("cuda_rng_restored") is True
            )
            or (
                rollback.get("rollback_verified") is True
                and rollback.get("candidate_executed") is False
                and rollback.get("optimizer_executed") is False
                and rollback.get("scheduler_executed") is False
                and rollback.get("rng_restore_required") is False
                and rollback.get("state_before") == rollback.get("state_after")
            )
        )
        if (
            descriptor not in claimed
            and int(value.get("attempted_optimizer_step", -1)) == orphan_step
            and value.get("counts_as_optimizer_commit") is False
            and value.get("cursor_advanced") is False
            and value.get("sampler_refreshed") is False
            and candidate_rollback
            and str(value.get("reason", "")).startswith((
                "preupdate_backend_health_v2_rejected:",
                "legacy_backend_correction_gate_rejected",
                "precommit_gradient_health_v2_rejected:",
                "ratio_health_v2_rejected:",
            ))
        ):
            scientific.append(path)
    for path in sorted((output / "steps").glob("generation_health_failure_step_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        samples = value.get("prompt_samples")
        descriptor = (str(path), _sha_file(path))
        if (
            descriptor not in claimed
            and int(value.get("optimizer_step", -1)) == orphan_step
            and value.get("artifact_kind") == "b2_generation_health_failure_v1"
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
        ):
            scientific.append(path)
    if not scientific:
        return {"recovered": False}
    evidence = {
        "failure_classification": "scientific_health_rejection",
        "recovered_after_process_exit": True,
        "rollback_reloaded_from_complete_checkpoint": True,
        "adapter_rollback_verified": True,
        "optimizer_rollback_verified": True,
        "scheduler_rollback_verified": True,
        "rng_rollback_verified": True,
        "cursor_advanced": False,
        "sampler_advanced": False,
        "kernel_rejection_artifacts": [
            str(path) for path in scientific
            if path.parent.name == "rejected_updates_v2"
        ],
        "generation_health_artifacts": [
            str(path) for path in scientific if path.parent.name == "steps"
        ],
        "kernel_rejection_evidence": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in scientific if path.parent.name == "rejected_updates_v2"
        ],
        "generation_health_evidence": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in scientific if path.parent.name == "steps"
        ],
        "counts_as_optimizer_commit": False,
    }
    complete_p9_attempt_intent(
        intent_path,
        outcome="scientific_health_rejection_recovered",
        evidence=evidence,
    )
    scientific_intents.append(
        json.loads(intent_path.read_text(encoding="utf-8"))
    )
    backfilled += backfill_ledgers()
    return {
        "recovered": True,
        "intent": str(intent_path),
        "ledger_rows_backfilled": backfilled,
    }


def reconcile_recovery_tail(output: Path, *, start_step: int) -> dict[str, Any]:
    output = Path(output)
    orphan_recovery = recover_orphan_scientific_intent(
        output, start_step=start_step
    )
    quarantine_root = output / "recovery_quarantine"
    attempt = len(list(quarantine_root.glob(f"resume_from_{start_step:03d}_attempt_*"))) + 1
    quarantine = quarantine_root / f"resume_from_{start_step:03d}_attempt_{attempt:02d}"
    moved: list[str] = []

    def move(path: Path) -> None:
        relative = path.relative_to(output)
        target = quarantine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        moved.append(str(relative))

    for root_name in ("formal_steps", "ratio_evidence_v2", "memory_step_audits"):
        root = output / root_name
        for path in sorted(root.glob("step_*.json")) if root.is_dir() else []:
            step = int(path.stem.rsplit("_", 1)[1])
            if step > start_step:
                move(path)
    b2_root = output / "b2_steps"
    for path in sorted(b2_root.glob("step_*.json")) if b2_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("to_version", -1)) > start_step:
            move(path)
    rejected_root = output / "rejected_updates_v2"
    for path in sorted(rejected_root.glob("attempt_*.json")) if rejected_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        attempted = int(value.get("attempted_optimizer_step", start_step + 1))
        if attempted > start_step:
            move(path)
    steps_root = output / "steps"
    for path in sorted(steps_root.glob("generation_health_failure_step_*.json")) if steps_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("optimizer_step", -1)) > start_step:
            move(path)
    intent_root = output / "transaction_intents"
    for path in sorted(intent_root.glob("*.json")) if intent_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        attempted = int(value.get("attempted_optimizer_step", -1))
        scientific_next = (
            attempted > start_step
            and value.get("outcome") in {
                "scientific_health_rejection",
                "scientific_health_rejection_recovered",
            }
        )
        if attempted > start_step and not scientific_next:
            move(path)
    transient_root = output / "checkpoints"
    for path in sorted(transient_root.glob("v*")) if transient_root.is_dir() else []:
        try:
            version = int(path.name[1:])
        except ValueError:
            version = start_step + 1
        if version >= start_step:
            move(path)
    formal_checkpoint_root = output / "formal_checkpoints"
    if formal_checkpoint_root.is_dir():
        for path in sorted(formal_checkpoint_root.iterdir()):
            if path.name.startswith("step_"):
                continue
            move(path)

    for filename, step_key in (
        ("metrics.jsonl", "accepted_optimizer_commits"),
        ("transaction_ledger.jsonl", None),
    ):
        path = output / filename
        if not path.is_file():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        kept: list[str] = []
        for line in lines:
            value = json.loads(line)
            if step_key is not None:
                step = int(value.get(step_key, -1))
            elif value.get("artifact_kind") == "p9_accepted_transaction":
                step = int(value.get("optimizer_step", -1))
            else:
                step = int(value.get("attempted_optimizer_step", start_step + 1))
            scientific_next_reserve = (
                value.get("artifact_kind") == "p9_rejected_attempt_rollback"
                and value.get("failure_classification") == "scientific_health_rejection"
                and step > start_step
            )
            if step <= start_step or scientific_next_reserve:
                kept.append(json.dumps(value, sort_keys=True, allow_nan=False))
        if len(kept) != len(lines):
            quarantine.mkdir(parents=True, exist_ok=True)
            _bind_bytes(quarantine / filename, path.read_bytes())
            _atomic_replace_bytes(
                path, (("\n".join(kept) + "\n") if kept else "").encode()
            )
            moved.append(filename)
    health = output / "health_summary.json"
    if health.is_file():
        value = json.loads(health.read_text(encoding="utf-8"))
        if int(value.get("optimizer_step", start_step + 1)) > start_step:
            move(health)
    telemetry_root = output / "memory_telemetry"
    marker_root = telemetry_root / "markers"
    for path in sorted(marker_root.glob("*.json")) if marker_root.is_dir() else []:
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("step", start_step + 1)) > start_step:
            move(path)
    telemetry = telemetry_root / "telemetry.jsonl"
    if telemetry.is_file():
        lines = [line for line in telemetry.read_text(encoding="utf-8").splitlines() if line]
        kept = [
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for value in (json.loads(line) for line in lines)
            if int(value.get("step", start_step + 1)) <= start_step
        ]
        if len(kept) != len(lines):
            quarantine.mkdir(parents=True, exist_ok=True)
            _bind_bytes(quarantine / "memory_telemetry/telemetry.jsonl", telemetry.read_bytes())
            _atomic_replace_bytes(
                telemetry, (("\n".join(kept) + "\n") if kept else "").encode()
            )
            moved.append("memory_telemetry/telemetry.jsonl")
    result = {
        "schema_version": 1, "artifact_kind": "p9_recovery_reconciliation",
        "resume_step": start_step, "quarantined": bool(moved),
        "quarantine": str(quarantine) if moved else None,
        "moved_or_rebuilt": moved,
        "orphan_scientific_intent_recovery": orphan_recovery,
        "final_access_count": 0,
    }
    if moved:
        _atomic_json(quarantine / "reconciliation.json", result)
    return result


def resume_reserve_variant(output: Path, *, attempted_step: int) -> int:
    ledger = Path(output) / "transaction_ledger.jsonl"
    variants: list[int] = []
    values: list[Mapping[str, Any]] = []
    if ledger.is_file():
        values.extend(
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line
        )
    intent_root = Path(output) / "transaction_intents"
    for path in sorted(intent_root.glob("*.json")) if intent_root.is_dir() else []:
        intent = json.loads(path.read_text(encoding="utf-8"))
        if intent.get("outcome") in {
            "scientific_health_rejection",
            "scientific_health_rejection_recovered",
        }:
            values.append({
                **dict(intent.get("evidence", {})),
                "artifact_kind": "p9_rejected_attempt_rollback",
                "attempted_optimizer_step": intent.get("attempted_optimizer_step"),
                "reserve_variant": intent.get("reserve_variant"),
                "failure_classification": "scientific_health_rejection",
            })
    for value in values:
        if (
            value.get("artifact_kind") == "p9_rejected_attempt_rollback"
            and value.get("failure_classification") == "scientific_health_rejection"
            and int(value.get("attempted_optimizer_step", -1)) == attempted_step
        ):
            if not all(
                value.get(field) is True
                for field in (
                    "adapter_rollback_verified", "optimizer_rollback_verified",
                    "scheduler_rollback_verified", "rng_rollback_verified",
                )
            ):
                raise P9ProtocolError("P9 prior reserve rejection rollback is untrusted")
            variants.append(int(value.get("reserve_variant", -1)))
    if not variants:
        return 0
    ordered = sorted(set(variants))
    if ordered != list(range(0, max(ordered) + 1)):
        raise P9ProtocolError("P9 prior reserve variants are not contiguous")
    return max(ordered) + 1


def validate_resume_checkpoint_for_launch(
    output: Path, resume_checkpoint: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint

    output = Path(output).resolve()
    checkpoint = Path(resume_checkpoint).resolve()
    step = int(manifest.get("logical_version", -1))
    if step == 120:
        if checkpoint != P7_STEP120 or resume_checkpoint.is_symlink():
            raise P9ProtocolError("P9 initial resume checkpoint path differs")
        complete_p9: list[int] = []
        for path in sorted((output / "formal_checkpoints").glob("step_*")):
            try:
                value = validate_formal_checkpoint(path)
            except Exception:
                continue
            complete_p9.append(int(value["logical_version"]))
        if complete_p9:
            raise P9ProtocolError(
                "P9 step120 resume is forbidden after a complete P9 checkpoint exists"
            )
    else:
        expected = output / "formal_checkpoints" / f"step_{step:03d}"
        if not (130 <= step <= 300 and step % 10 == 0 and checkpoint == expected):
            raise P9ProtocolError("P9 recovery checkpoint path/step differs")
        complete: list[int] = []
        for path in sorted((output / "formal_checkpoints").glob("step_*")):
            value = validate_formal_checkpoint(path)
            complete.append(int(value["logical_version"]))
        if not complete or max(complete) != step:
            raise P9ProtocolError("P9 recovery must use the latest complete checkpoint")
    if not (
        manifest.get("complete") is True
        and manifest.get("resume_eligible") is True
        and manifest.get("optimizer_step") == step
        and manifest.get("scheduler_step") == step
        and manifest.get("policy_version") == step
        and manifest.get("data_cursor") == step * 4
    ):
        raise P9ProtocolError("P9 recovery checkpoint state counters differ")
    return {"passed": True, "resume_step": step, "checkpoint": str(checkpoint)}


def audit_p9_reload_qualification(output: Path) -> dict[str, Any]:
    path = Path(output) / "p9_step120_reload_qualification.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P9ProtocolError("P9 two-process reload qualification is absent") from error
    checks = value.get("checks")
    attempts = value.get("attempts")
    required_checks = {
        "sample_ids_same",
        "completion_token_counts_same",
        "completion_token_sha256_same",
        "ratio_evidence_sha256_same",
        "student_score_same_within_1e_6",
        "teacher_score_same_within_1e_6",
        "loss_same_within_1e_6",
        "objective_same_within_1e_6",
        "reverse_kl_same_within_1e_6",
        "advantage_same_within_1e_6",
        "ess_fraction_same_within_1e_6",
        "ratio_same_within_1e_6",
        "health_classification_same_within_1e_6",
        "first_rollback_complete",
        "second_rollback_complete",
        "p7_checkpoint_adapter_identity_exact",
        "p7_checkpoint_weight_unchanged",
        "restore_identity_exact",
        "base_teacher_launch_identity_exact",
        "ratio_evidence_sha256_bound",
    }
    expected_launch = {
        "passed": True,
        "p7_package_content_sha256": "c21ca9acec85bb72014ddfc48b5cf9079f680807cbfed348fe5ce1cc619583e1",
        "base_manifest_sha256": "c796c078afd35849b59017582eb7dd0e1553be43bdbd7ce0eed441fda889a213",
        "teacher_ordered_sha256": "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2",
        "teacher_weight_sha256": "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63",
        "teacher_manifest_sha256": "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67",
        "final_access_count": 0,
    }
    expected_resume = {
        "artifact_kind": "formal_b2_resume_identity_smoke_v1",
        "logical_version": 120,
        "data_cursor": 480,
        "adapter_sha256": "6e34e1b9b83064016968dd7d1c9f9c4d70ff87058aa3cab2e2be52bee7570408",
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "rng_state_restored": True,
        "sampler_state_restored": True,
        "resume_probe_label_access_count": 0,
        "passed": True,
    }
    try:
        schedule = json.loads((Path(output) / "schedule.json").read_text(encoding="utf-8"))
        expected_sample_ids = [
            row["sample_id"] for row in schedule["slots"] if row["step"] == 121
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise P9ProtocolError("P9 reload qualification schedule binding is absent") from error
    attempts_bound = (
        isinstance(attempts, list)
        and len(attempts) == 2
        and all(
            isinstance(attempt, Mapping)
            and isinstance(attempt.get("launch_asset_audit"), Mapping)
            and isinstance(attempt.get("resume"), Mapping)
            and all(attempt.get("launch_asset_audit", {}).get(key) == expected for key, expected in expected_launch.items())
            and all(attempt.get("resume", {}).get(key) == expected for key, expected in expected_resume.items())
            and attempt.get("checkpoint_adapter_sha256_before") == expected_resume["adapter_sha256"]
            and attempt.get("checkpoint_adapter_weight_sha256_after") == "bd6bfe2597c82113c2a878f31abc0b7a7e99a05e7221b888f0a86220404d64f9"
            and attempt.get("sample_ids") == expected_sample_ids
            and isinstance(attempt.get("completion_token_counts"), list)
            and len(attempt["completion_token_counts"]) == 4
            and all(isinstance(count, int) and count > 0 for count in attempt["completion_token_counts"])
            and isinstance(attempt.get("completion_token_sha256"), str)
            and len(attempt["completion_token_sha256"]) == 64
            and isinstance(attempt.get("ratio_evidence_sha256"), str)
            and len(attempt["ratio_evidence_sha256"]) == 64
            and isinstance(attempt.get("rollback"), Mapping)
            and attempt["rollback"].get("counts_as_optimizer_commit") is False
            and all(
                attempt["rollback"].get(field) is True
                for field in (
                    "adapter_rollback_verified", "optimizer_rollback_verified",
                    "scheduler_rollback_verified", "rng_rollback_verified",
                )
            )
            and attempt.get("cursor_rng_sampler_version_advanced") is False
            and attempt.get("formal_commit_count") == 0
            and attempt.get("final_access_count") == 0
            for attempt in attempts
        )
    )
    if not (
        value.get("status") == "passed"
        and value.get("passed") is True
        and value.get("fresh_process_count") == 2
        and value.get("formal_commit_count") == 0
        and value.get("controller_access_count") == 0
        and value.get("final_access_count") == 0
        and isinstance(checks, Mapping)
        and required_checks <= set(checks)
        and all(item is True for item in checks.values())
        and attempts_bound
    ):
        raise P9ProtocolError("P9 two-process reload qualification did not pass")
    return {"passed": True, "path": str(path), "sha256": _sha_file(path)}


def initialize_authoritative_run_artifacts(
    output: Path,
    *,
    schedule: Mapping[str, Any],
    authority: Mapping[str, Any],
    p7_config: Mapping[str, Any],
    schedule_path: Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    schedule_path = Path(schedule_path or output / "schedule.json")
    config_bytes = CHECKED_IN_CONFIG.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not (
        config["schedule"]["p9_schedule_semantic_sha256"] == schedule.get("schedule_sha256")
        and schedule_path.is_file()
        and config["schedule"]["p9_schedule_file_sha256"] == _sha_file(schedule_path)
        and config["training"]["action"] == {"medical_opd_o1": 2, "medical_opd_cmb": 2}
    ):
        raise P9ProtocolError("P9 checked-in config differs from frozen schedule/action")
    protocol = p7_config.get("protocol", {})
    formula_path = Path(str(protocol.get("three_policy_formula_path", "")))
    formula_sha = str(protocol.get("three_policy_formula_sha256", ""))
    if not formula_path.is_file() or _sha_file(formula_path) != formula_sha:
        raise P9ProtocolError("P9 P7 formula identity differs")
    _bind_bytes(output / "config.yaml", config_bytes)
    _atomic_json(output / "data_manifest.json", {
        "schema_version": 1, "artifact_kind": "p9_data_manifest",
        "manifest_sha256": authority.get("manifest_sha256"),
        "payloads": authority.get("payloads"),
        "controller_access_count": 0, "final_access_count": 0,
    })
    _atomic_json(output / "formula_manifest.json", {
        "schema_version": 1, "artifact_kind": "p9_formula_manifest",
        "path": str(formula_path), "sha256": formula_sha,
        "semantic_source": "P7_formula_v6_unchanged", "final_access_count": 0,
    })
    _atomic_json(output / "schedule_manifest.json", {
        "schema_version": 1, "artifact_kind": "p9_run_schedule_manifest",
        "path": str(schedule_path.resolve()),
        "schedule_file_sha256": _sha_file(schedule_path),
        "schedule_sha256": schedule.get("schedule_sha256"),
        "source_counts": schedule.get("source_counts"),
        "slot_count": len(schedule.get("slots", [])),
        "reserve_slot_count": len(schedule.get("reserves", [])),
        "controller_access_count": 0, "final_access_count": 0,
    })
    return {"passed": True, "artifacts": [
        "config.yaml", "data_manifest.json", "formula_manifest.json", "schedule_manifest.json"
    ]}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _timed_process_records(output: Path) -> list[dict[str, Any]]:
    selected: dict[str, tuple[int, dict[str, Any]]] = {}
    sources = [
        (0, path) for path in sorted(Path(output).glob("training_status_step*.json"))
    ] + [
        (1, path)
        for path in sorted((Path(output) / "failures").glob("failure_*.json"))
    ]
    for priority, path in sources:
        value = json.loads(path.read_text(encoding="utf-8"))
        key = str(value.get("process_id", path))
        previous = selected.get(key)
        if previous is None or priority >= previous[0]:
            selected[key] = (priority, value)
    return [value for _priority, value in selected.values()]


def write_p9_cost_artifact(output: Path) -> dict[str, Any]:
    timed = _timed_process_records(output)
    value = {
        "schema_version": 1,
        "artifact_kind": "p9_cost",
        "gpu_wall_time_seconds": sum(
            float(item["elapsed_seconds_this_process"]) for item in timed
        ),
        "gpu_hours": sum(float(item["gpu_hours_this_process"]) for item in timed),
        "reference_price_cny_per_hour": 2.96,
        "derived_cost_cny": sum(float(item["derived_cost_cny"]) for item in timed),
        "platform_actual_cost_cny": None,
        "timed_process_count": len(timed),
        "final_access_count": 0,
    }
    _atomic_json(Path(output) / "cost.json", value)
    return value


def validate_launch_scope(*, mode: str, target_step: int) -> dict[str, Any]:
    if mode != "b2_medical":
        raise P9ProtocolError("P9 launcher forbidden mode requested")
    if target_step not in ALLOWED_TARGETS:
        raise P9ProtocolError("P9 launcher target is not 200, 240, or 300")
    return {"passed": True, "mode": mode, "target_step": target_step, "forbidden_runs_started": []}


def validate_p9_execution_environment() -> dict[str, Any]:
    expected = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise P9ProtocolError(f"P9 {name} launch binding differs")
    return {"passed": True, "environment": expected}


def require_decision_before_resume(
    output: Path,
    *,
    current_step: int,
    target_step: int,
    boundary_checkpoint_sha256: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    if current_step % 10 or not 120 <= current_step <= target_step <= 300:
        raise P9ProtocolError("P9 resume checkpoint/target step boundary differs")
    if current_step == target_step == 200:
        return {
            "passed": True, "decision_required": False,
            "boundary_finalize_only": True,
        }
    if current_step < 200:
        if target_step != 200:
            raise P9ProtocolError("P9 pre-step200 recovery must still target step200")
        return {"passed": True, "decision_required": False}

    def verified(decision_step: int) -> dict[str, Any]:
        expected_sha = None
        if boundary_checkpoint_sha256 is not None:
            expected_sha = boundary_checkpoint_sha256.get(decision_step)
        if expected_sha is None:
            from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint

            checkpoint = output / "formal_checkpoints" / f"step_{decision_step:03d}"
            try:
                manifest = validate_formal_checkpoint(checkpoint)
            except Exception as error:
                raise P9ProtocolError(
                    "P9 decision boundary checkpoint is absent or incomplete"
                ) from error
            if int(manifest.get("logical_version", -1)) != decision_step:
                raise P9ProtocolError("P9 decision boundary checkpoint step differs")
            expected_sha = _sha_file(checkpoint / "checkpoint_manifest.json")
        path = output / f"decision_step{decision_step}.json"
        marker_path = output / f"decision_step{decision_step}_complete_marker.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise P9ProtocolError("P9 decision artifact or complete marker is absent") from error
        claimed = value.get("artifact_sha256")
        unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if not (
            isinstance(claimed, str)
            and claimed == _canonical_sha(unsigned)
            and value.get("checkpoint_manifest_sha256") == expected_sha
            and marker.get("decision_step") == decision_step
            and marker.get("decision_sha256") == _sha_file(path)
            and marker.get("complete") is True
            and marker.get("continue") is value.get("continue")
            and marker.get("next_max_step") == value.get("next_max_step")
            and marker.get("final_access_count") == 0
        ):
            raise P9ProtocolError("P9 decision/checkpoint/complete-marker SHA binding differs")
        return value

    decision200 = verified(200)
    if (
        decision200.get("status") == "promising_at_200_continue_300"
        and decision200.get("continue") is True
        and decision200.get("next_max_step") == 300
        and target_step == 300
    ):
        return {
            "passed": True, "decision_required": True, "decision_step": 200,
            "status": decision200["status"],
        }
    if not (
        decision200.get("status") == "gray_at_200_continue_240"
        and decision200.get("continue") is True
        and decision200.get("next_max_step") == 240
    ):
        raise P9ProtocolError("P9 step200 decision does not authorize recovery")
    if current_step <= 240 and target_step == 240:
        return {
            "passed": True, "decision_required": True, "decision_step": 200,
            "status": decision200["status"],
            "boundary_finalize_only": current_step == target_step,
        }
    decision_step = 240
    value = verified(decision_step)
    if not (
        value.get("status") == "promising_at_240_continue_300"
        and value.get("continue") is True
        and value.get("next_max_step") == 300
        and target_step == 300
    ):
        raise P9ProtocolError("P9 decision artifact does not authorize this resume")
    return {
        "passed": True, "decision_required": True, "decision_step": decision_step,
        "status": value["status"],
        "boundary_finalize_only": current_step == target_step,
    }


def checkpoint_index(output: Path) -> dict[str, Any]:
    from src.opd.production_b2_formal_checkpoint_v1 import validate_formal_checkpoint

    checkpoints = []
    root = Path(output) / "formal_checkpoints"
    if root.is_dir():
        for path in sorted(root.glob("step_*")):
            manifest = validate_formal_checkpoint(path)
            checkpoints.append({
                "step": manifest["logical_version"],
                "path": str(path),
                "adapter_sha256": manifest["adapter_sha256"],
                "adapter_weight_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
                "checkpoint_manifest_sha256": _sha_file(path / "checkpoint_manifest.json"),
                "complete": True,
                "resume_eligible": True,
                "permanent": manifest["logical_version"] in {150, 180, 200, 240, 270, 300},
            })
    result = {
        "schema_version": 1,
        "artifact_kind": "p9_checkpoint_index",
        "retention": "all_complete_p9_checkpoints_retained_within_disk_budget",
        "p7_step120_read_only": True,
        "checkpoints": checkpoints,
    }
    _atomic_json(Path(output) / "checkpoint_index.json", result)
    return result


def _hydrate_rows(authority: Mapping[str, Any], scheduled: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from src.opd.production_b2_data_v2 import _iter_jsonl, _validate_prompt_row

    wanted = {(row["sample_id"], row["content_hash"]): row for row in scheduled}
    hydrated: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in authority["payloads"]:
        role = payload["target_role"]
        if role not in {"medical_opd_o1", "medical_opd_cmb"}:
            continue
        for _line, raw in _iter_jsonl(Path(payload["resolved_path"])):
            key = (raw.get("sample_id"), raw.get("content_hash"))
            if key in wanted:
                _validate_prompt_row(raw, role=role); hydrated[key] = dict(raw)
    if set(hydrated) != set(wanted):
        raise P9ProtocolError("P9 worker could not hydrate frozen batch")
    return [hydrated[(row["sample_id"], row["content_hash"])] for row in scheduled]


def _batch(schedule: Mapping[str, Any], *, step: int, reserve_variant: int) -> list[dict[str, Any]]:
    if reserve_variant == 0:
        return [dict(row) for row in schedule["slots"] if row["step"] == step]
    return [
        dict(row) for row in schedule["reserves"]
        if row["accepted_step"] == step and row["reserve_variant"] == reserve_variant
    ]


def run_p9_worker(
    *,
    output: Path,
    p7_package: Path,
    resume_checkpoint: Path,
    target_step: int,
) -> dict[str, Any]:  # pragma: no cover - GPU
    validate_launch_scope(mode="b2_medical", target_step=target_step)
    validate_p9_execution_environment()
    if os.environ.get("CA_OPD_ALLOW_P9_TRAINING") != "1":
        raise P9ProtocolError("P9 paid-training authorization environment is absent")
    output = Path(output).resolve(); p7_package = Path(p7_package).resolve(); resume_checkpoint = Path(resume_checkpoint).resolve()
    terminal_failure = output / "failure.json"
    if terminal_failure.is_file():
        failure_value = json.loads(terminal_failure.read_text(encoding="utf-8"))
        if failure_value.get("terminal_status") == "failed_training_health":
            raise P9ProtocolError("P9 terminal training-health failure forbids resume")
    launch_asset_audit = audit_p9_launch_assets(p7_package)
    reload_qualification_audit = audit_p9_reload_qualification(output)
    schedule = json.loads((output / "schedule.json").read_text(encoding="utf-8"))
    authority = json.loads((output / "data_authority.json").read_text(encoding="utf-8"))
    p7_config = json.loads((p7_package / "formal_b2_config.json").read_text(encoding="utf-8"))
    p7_index = json.loads((p7_package / "package_index.json").read_text(encoding="utf-8"))
    environment = json.loads((p7_package / "environment.json").read_text(encoding="utf-8"))
    initialize_authoritative_run_artifacts(
        output, schedule=schedule, authority=authority, p7_config=p7_config
    )
    from src.opd.p9_adaptive_dose_protocol import validate_p9_resume_manifest
    from src.opd.p9_runtime import build_p9_runtime_config
    from src.opd.production_b2_formal_checkpoint_v1 import seal_formal_checkpoint, validate_formal_checkpoint
    from src.opd.production_b2_formal_gpu_v2 import validate_formal_step_health_v2
    from src.opd.p9_adaptive_dose_gpu import (
        P9EngineeringAttemptError, P9FormalB2Session, P9RejectedAttempt,
    )

    resume_manifest = validate_formal_checkpoint(resume_checkpoint)
    start_step = int(resume_manifest["logical_version"])
    validate_resume_checkpoint_for_launch(output, resume_checkpoint, resume_manifest)
    if start_step == 120:
        validate_p9_resume_manifest(resume_manifest)
    require_decision_before_resume(output, current_step=start_step, target_step=target_step)
    runtime = build_p9_runtime_config(p7_config, output=output, schedule_sha256=schedule["schedule_sha256"])
    runtime_path = output / "runtime_config.json"
    if runtime_path.is_file():
        existing = json.loads(runtime_path.read_text(encoding="utf-8"))
        if existing != runtime:
            raise P9ProtocolError("P9 runtime config drifted across process boundary")
    else:
        _atomic_json(runtime_path, runtime)
    config_sha = _sha_file(runtime_path)
    package_content_sha = _canonical_sha({
        "p7_package_content_sha256": p7_index["package_content_sha256"],
        "runtime_config_sha256": config_sha,
        "schedule_sha256": schedule["schedule_sha256"],
    })
    if start_step > 120 and not (
        resume_manifest.get("package_content_sha256") == package_content_sha
        and resume_manifest.get("config_sha256") == config_sha
        and resume_manifest.get("manifest_sha256") == authority["manifest_sha256"]
        and resume_manifest.get("schedule_sha256") == schedule["schedule_sha256"]
    ):
        raise P9ProtocolError("P9 recovery checkpoint runtime/package identities differ")
    recovery_reconciliation = reconcile_recovery_tail(output, start_step=start_step)
    metadata = {
        "schema_version": 1, "artifact_kind": "p9_run_metadata",
        "run_id": output.name, "formal_training_git_sha": os.environ.get("P9_FORMAL_GIT_SHA"),
        "p7_package_content_sha256": p7_index["package_content_sha256"],
        "p9_package_content_sha256": package_content_sha,
        "runtime_config_sha256": config_sha, "schedule_sha256": schedule["schedule_sha256"],
        "resume_checkpoint": str(resume_checkpoint), "target_step_this_process": target_step,
        "launch_asset_audit": launch_asset_audit,
        "reload_qualification_audit": reload_qualification_audit,
        "recovery_reconciliation": recovery_reconciliation,
        "final_access_count": 0,
    }
    _atomic_json(output / "metadata.json", metadata)
    for name in (
        "b2_steps", "checkpoints", "formal_steps", "memory_step_audits",
        "memory_telemetry/markers", "ratio_evidence_v2", "rejected_updates_v2",
        "steps", "transaction_intents",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((output / "formal_steps").glob("step_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value["optimizer_step"]) <= start_step:
            records.append(value)
    started = time.time()
    process_id = f"{time.time_ns()}-{os.getpid()}"
    if start_step == target_step:
        status_path = output / f"training_status_step{target_step}.json"
        if status_path.is_file():
            result = json.loads(status_path.read_text(encoding="utf-8"))
            if not (
                result.get("artifact_kind") == "p9_training_boundary_status"
                and result.get("status")
                == f"trained_to_{target_step}_controller_pending"
                and int(result.get("accepted_optimizer_commits", -1)) == target_step
                and result.get("final_access_count") == 0
            ):
                raise P9ProtocolError("P9 existing boundary status is untrusted")
        else:
            result = {
                "schema_version": 1,
                "artifact_kind": "p9_training_boundary_status",
                "status": f"trained_to_{target_step}_controller_pending",
                "accepted_optimizer_commits": target_step,
                "new_accepted_commits_this_process": 0,
                "boundary_recovered_from_complete_checkpoint": True,
                "rejected_attempts_total": len(
                    list((output / "rejected_updates_v2").glob("attempt_*.json"))
                ),
                "checkpoint_index": checkpoint_index(output),
                "process_id": process_id,
                "elapsed_seconds_this_process": time.time() - started,
                "gpu_hours_this_process": 0.0,
                "derived_cost_cny": 0.0,
                "platform_actual_cost_cny": None,
                "confirmation_access_count": 0,
                "final_access_count": 0,
            }
            _atomic_json(status_path, result)
        write_p9_cost_artifact(output)
        return result
    session: P9FormalB2Session | None = None
    try:
        session = P9FormalB2Session(runtime, config_path=runtime_path, route="b2_calibration")
        resume_rows = _hydrate_rows(authority, _batch(schedule, step=start_step + 1, reserve_variant=0))
        if start_step == 120:
            restore_ids = {
                "package_content_sha256": p7_index["package_content_sha256"],
                "config_sha256": p7_index["config_sha256"],
                "manifest_sha256": p7_index["manifest_sha256"],
                "schedule_sha256": p7_index["schedule_semantic_sha256"],
            }
        else:
            restore_ids = {
                "package_content_sha256": package_content_sha,
                "config_sha256": config_sha,
                "manifest_sha256": authority["manifest_sha256"],
                "schedule_sha256": schedule["schedule_sha256"],
            }
        resume_evidence = session.restore_formal_checkpoint_v1(
            resume_checkpoint, resume_prompt_rows=resume_rows, **restore_ids
        )
        _atomic_json(output / f"resume_identity_step_{start_step:03d}.json", resume_evidence)
        initial_registry = session._registry_count(); initial_models = session._model_count()
        for step_index in range(start_step, target_step):
            next_step = step_index + 1
            if shutil.disk_usage(output).free < 10_000_000_000:
                raise P9ProtocolError("P9 disk fell below the 10GB safety floor")
            record = None
            first_variant = resume_reserve_variant(output, attempted_step=next_step)
            for variant in range(first_variant, int(schedule["reserve_variants_per_step"]) + 1):
                rows = _hydrate_rows(authority, _batch(schedule, step=next_step, reserve_variant=variant))
                intent_path = begin_p9_attempt_intent(
                    output, step=next_step, variant=variant, rows=rows
                )
                try:
                    record = session.run_p9_attempt(step_index=step_index, prompt_rows=rows, max_new_tokens=1024)
                    record["p9"]["reserve_variant"] = variant
                    break
                except P9RejectedAttempt as error:
                    rejection = {
                        **error.evidence,
                        "reserve_variant": variant,
                        "source_roles": [row["target_role"] for row in rows],
                        "next_reserve_variant": variant + 1,
                    }
                    complete_p9_attempt_intent(
                        intent_path,
                        outcome="scientific_health_rejection",
                        evidence=rejection,
                    )
                    _append_jsonl(output / "transaction_ledger.jsonl", rejection)
            if record is None:
                raise P9TrainingHealthFailure(
                    "P9 exhausted all frozen same-action reserves after scientific rejection"
                )
            records.append(record)
            try:
                health = validate_formal_step_health_v2(
                    records,
                    initial_registry_count=initial_registry,
                    initial_model_count=initial_models,
                )
            except Exception as error:
                if str(error).startswith("Formal B2 v2 health failed:"):
                    raise P9TrainingHealthFailure(str(error)) from error
                raise
            checkpoint_due = next_step % 10 == 0
            metrics = {
                "accepted_optimizer_commits": next_step,
                "rejected_attempts": len(list((output / "rejected_updates_v2").glob("attempt_*.json"))),
                "loss": record["loss"], "reverse_kl": record["reverse_kl"],
                "advantage": {
                    **record["advantage"],
                    "positive_fraction": record["p9"]["advantage_positive_fraction"],
                }, "ratio_v2": record["ratio_v2"],
                "ess_fraction": record["ess_fraction"], "gradient_norm": record["gradient_norm"],
                "gradient_norm_before_clip": record["gradient_norm_before_clip"],
                "bounded_influence_v2": record.get("bounded_influence_v2"),
                "adapter_delta_norm": record["adapter_delta_norm"], "prompt_samples": record["prompt_samples"],
                "completion_token_counts": record["p9"]["completion_token_counts"],
                "completion_token_sha256": record["p9"]["completion_token_sha256"],
                "raw_completion_tokens_persisted": False,
                "timings_seconds": record["timings_seconds"], "throughput": record["throughput"],
                "gpu_memory_bytes": record["gpu_memory_bytes"], "disk_free_bytes": shutil.disk_usage(output).free,
                "checkpoint_due": checkpoint_due,
                "checkpoint_complete_on_successful_step_return": checkpoint_due,
                "health": health,
                "transient_cleanup_after_ledger": True, "final_access_count": 0,
            }
            _append_jsonl(output / "metrics.jsonl", metrics)
            _append_jsonl(output / "transaction_ledger.jsonl", {
                "artifact_kind": "p9_accepted_transaction", "optimizer_step": next_step,
                "reserve_variant": record["p9"]["reserve_variant"], "accepted": True,
                "adapter_sha256": record["checkpoint"]["adapter_sha256"], "final_access_count": 0,
            })
            complete_p9_attempt_intent(
                _attempt_intent_path(
                    output,
                    step=next_step,
                    variant=int(record["p9"]["reserve_variant"]),
                ),
                outcome="accepted",
                evidence={
                    "optimizer_step": next_step,
                    "adapter_sha256": record["checkpoint"]["adapter_sha256"],
                    "counts_as_optimizer_commit": True,
                },
            )
            _atomic_json(output / "health_summary.json", health)
            if checkpoint_due:
                seal_formal_checkpoint(
                    session, logical_version=next_step, data_cursor=next_step * 4,
                    package_content_sha256=package_content_sha, config_sha256=config_sha,
                    manifest_sha256=authority["manifest_sha256"], schedule_sha256=schedule["schedule_sha256"],
                    environment=environment,
                )
                checkpoint_index(output)
            session.release_transient_step_artifacts_v1(next_step)
            print(json.dumps({"event": "p9_progress", "step": next_step, "loss": record["loss"], "ess": record["ess_fraction"], "elapsed_seconds": time.time() - started}), flush=True)
        result = {
            "schema_version": 1, "artifact_kind": "p9_training_boundary_status",
            "status": f"trained_to_{target_step}_controller_pending",
            "accepted_optimizer_commits": target_step,
            "new_accepted_commits_this_process": target_step - start_step,
            "rejected_attempts_total": len(list((output / "rejected_updates_v2").glob("attempt_*.json"))),
            "checkpoint_index": checkpoint_index(output),
            "process_id": process_id,
            "elapsed_seconds_this_process": time.time() - started,
            "gpu_hours_this_process": 2 * (time.time() - started) / 3600,
            "derived_cost_cny": (time.time() - started) / 3600 * 2.96,
            "platform_actual_cost_cny": None,
            "confirmation_access_count": 0, "final_access_count": 0,
        }
        _atomic_json(output / f"training_status_step{target_step}.json", result)
        write_p9_cost_artifact(output)
        return result
    except Exception as error:
        evidence = error.evidence if isinstance(error, P9EngineeringAttemptError) else None
        failure = {
            "schema_version": 1, "artifact_kind": "p9_training_failure",
            "failure_type": type(error).__name__, "reason": str(error),
            "failure_classification": (
                "scientific_training_health"
                if isinstance(error, P9TrainingHealthFailure)
                else "engineering_or_protocol_failure"
            ),
            "terminal_status": (
                "failed_training_health"
                if isinstance(error, P9TrainingHealthFailure)
                else None
            ),
            "resume_step_this_process": start_step,
            "target_step_this_process": target_step,
            "elapsed_seconds_this_process": time.time() - started,
            "gpu_hours_this_process": 2 * (time.time() - started) / 3600,
            "derived_cost_cny": (time.time() - started) / 3600 * 2.96,
            "platform_actual_cost_cny": None,
            "engineering_rollback_evidence": evidence,
            "requires_resume_from_latest_complete_checkpoint": not isinstance(
                error, P9TrainingHealthFailure
            ),
            "in_memory_candidate_discarded_on_process_close": isinstance(
                error, P9TrainingHealthFailure
            ),
            "counts_as_optimizer_commit": False,
            "process_id": process_id,
            "confirmation_access_count": 0, "final_access_count": 0,
        }
        failure_root = output / "failures"
        suffix = len(list(failure_root.glob("failure_*.json"))) + 1
        _atomic_json(failure_root / f"failure_{suffix:03d}.json", failure)
        _atomic_json(output / "failure.json", failure)
        write_p9_cost_artifact(output)
        raise
    finally:
        if session is not None:
            session.close()


__all__ = [
    "audit_p9_launch_assets", "audit_p9_reload_qualification", "checkpoint_index",
    "initialize_authoritative_run_artifacts", "reconcile_recovery_tail",
    "resume_reserve_variant",
    "require_decision_before_resume", "run_p9_worker", "validate_launch_scope",
    "validate_p9_execution_environment", "validate_resume_checkpoint_for_launch",
    "write_p9_cost_artifact",
]


def _main() -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p7-package", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--mode", default="b2_medical")
    args = parser.parse_args()
    validate_launch_scope(mode=args.mode, target_step=args.target_step)
    print(json.dumps(run_p9_worker(output=args.output, p7_package=args.p7_package, resume_checkpoint=args.resume_checkpoint, target_step=args.target_step), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
