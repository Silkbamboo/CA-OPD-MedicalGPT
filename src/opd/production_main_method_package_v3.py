"""Build and verify immutable P6 IDT/CA formal packages."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from src.eval.p3_5_checkpoint_screen import _ordered_adapter_sha256
from src.opd.production_b2_disk_policy_v2 import validate_disk_safety_v2
from src.opd.production_b2_formal_package_v2 import (
    MEASURED_FULL_CHECKPOINT_BYTES,
    PREDICTED_LOG_GROWTH_BYTES,
)
from src.opd.production_b2_formal_v2 import formal_b2_runtime_config_v2
from src.opd.production_main_protocol_v2 import common_protocol_sha256_v2
from src.opd.production_main_method_v3 import P6FormalMethodError


REPO = Path(__file__).resolve().parents[2]
TEACHER_ORDERED_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
TEACHER_WEIGHT_SHA256 = "7866a15f1e3308cab5c4814d974c8a5368785013a137e66adb0ff0e6828adf63"
TEACHER_MANIFEST_SHA256 = "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67"
COMMON_SEMANTIC_SHA256 = "40b3922d7f916eba9ab41408caccc2856067b97dc59e3721629f8aa1fdf7f2d3"


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


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6FormalMethodError(f"formal method JSON invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise P6FormalMethodError("formal method JSON is not an object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def _method_template(method_id: str) -> dict[str, Any]:
    path = REPO / "configs/runs" / (
        "qwen3_4b_idt_formal_package_v2.json"
        if method_id == "IDT"
        else "qwen3_4b_ca_opd_formal_package_v2.json"
    )
    value = _json(path)
    if common_protocol_sha256_v2(value) != COMMON_SEMANTIC_SHA256:
        raise P6FormalMethodError("formal method common semantic SHA differs")
    return value


def verify_method_package_v3(
    package: Path, *, require_clean_git: bool = False
) -> dict[str, Any]:
    package = Path(package).resolve()
    index = _json(package / "package_index.json")
    authorization = _json(package / "authorization.json")
    config = _json(package / "formal_method_config.json")
    files = index.get("files")
    if not (
        index.get("schema_version") == 3
        and index.get("artifact_kind") == "p6_formal_method_package_index_v3"
        and isinstance(files, Mapping)
        and authorization.get("package_content_sha256")
        == index.get("package_content_sha256")
        and authorization.get("final_authorized") is False
        and authorization.get("formal_training_authorized") is True
    ):
        raise P6FormalMethodError("formal method package index/authorization differs")
    actual_files: dict[str, Any] = {}
    for name, descriptor in files.items():
        path = package / str(name)
        if not (
            path.is_file()
            and not path.is_symlink()
            and _sha_file(path) == descriptor.get("sha256")
            and path.stat().st_size == descriptor.get("size_bytes")
        ):
            raise P6FormalMethodError("formal method package file SHA differs")
        actual_files[str(name)] = dict(descriptor)
    if _canonical_sha(actual_files) != index.get("package_content_sha256"):
        raise P6FormalMethodError("formal method package content SHA differs")
    teacher = config.get("teacher")
    adapter = Path(str(teacher.get("adapter_path", ""))) if isinstance(teacher, Mapping) else Path()
    manifest = Path(str(teacher.get("manifest_path", ""))) if isinstance(teacher, Mapping) else Path()
    if not (
        config.get("formal_method_v3", {}).get("method_id") in {"IDT", "CA-OPD"}
        and adapter.is_dir()
        and _ordered_adapter_sha256(adapter) == TEACHER_ORDERED_SHA256
        and _sha_file(adapter / "adapter_model.safetensors") == TEACHER_WEIGHT_SHA256
        and manifest.is_file()
        and _sha_file(manifest) == TEACHER_MANIFEST_SHA256
        and teacher.get("adapter_sha256") == TEACHER_ORDERED_SHA256
        and teacher.get("adapter_weight_sha256") == TEACHER_WEIGHT_SHA256
        and teacher.get("manifest_sha256") == TEACHER_MANIFEST_SHA256
    ):
        raise P6FormalMethodError("formal method Teacher binding differs")
    formal_b2_runtime_config_v2(config)
    if require_clean_git and _git("status", "--porcelain"):
        raise P6FormalMethodError("formal method package requires clean Git")
    if require_clean_git:
        package_head = str(authorization.get("git_head") or "")
        current_head = _git("rev-parse", "HEAD")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", package_head, current_head],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if not package_head or ancestry.returncode != 0:
            raise P6FormalMethodError(
                "formal method package Git authorization is not an ancestor"
            )
    disk = validate_disk_safety_v2(
        free_bytes=shutil.disk_usage(Path(str(config["run"]["output_dir"])).parent).free,
        full_checkpoint_bytes=MEASURED_FULL_CHECKPOINT_BYTES,
        predicted_log_growth_bytes=PREDICTED_LOG_GROWTH_BYTES,
    )
    return {
        "passed": True,
        "method_id": config["formal_method_v3"]["method_id"],
        "package_content_sha256": index["package_content_sha256"],
        "config_sha256": files["formal_method_config.json"]["sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "schedule_semantic_sha256": index["schedule_semantic_sha256"],
        "disk": disk,
        "final_access_count": 0,
        "model_loader_calls": 0,
        "teacher_loader_calls": 0,
    }


def build_method_package_v3(
    *,
    method_id: str,
    package: Path,
    output: Path,
    b2_package: Path,
    controller_report: Path,
    selected_b2_snapshot: Path,
) -> dict[str, Any]:
    if method_id not in {"IDT", "CA-OPD"}:
        raise P6FormalMethodError("formal method must be IDT or CA-OPD")
    package = Path(package).resolve()
    output = Path(output).resolve()
    b2_package = Path(b2_package).resolve()
    controller_report = Path(controller_report).resolve()
    selected_b2_snapshot = Path(selected_b2_snapshot).resolve()
    if package.exists() or package.is_symlink() or output.exists() or output.is_symlink():
        raise P6FormalMethodError("formal method package/output must be fresh")
    if _git("status", "--porcelain"):
        raise P6FormalMethodError("formal method package build requires clean committed Git")
    source_index = _json(b2_package / "package_index.json")
    source_config = _json(b2_package / "formal_b2_config.json")
    schedule = _json(b2_package / "prompt_schedule.json")
    authority = _json(b2_package / "data_authority.json")
    controller = _json(controller_report)
    snapshot = _json(selected_b2_snapshot / "snapshot_manifest.json")
    selection = controller.get("selection", {}).get("B2")
    if not (
        controller.get("artifact_kind") == "p6_identity_bound_controller_report"
        and controller.get("final_access_count") == 0
        and isinstance(selection, Mapping)
        and selection.get("status") == "selected"
        and int(selection.get("selected_step")) == int(snapshot["logical_version"])
    ):
        raise P6FormalMethodError("formal method B2/controller freeze differs")
    template = _method_template(method_id)
    metrics = controller["metrics"]
    config = deepcopy(source_config)
    config["run"] = {
        "run_id": output.name,
        "seed": 42,
        # Keep the previously qualified compatibility envelope (150 maximum,
        # frozen stage-1 stop at 120).  formal_method_v3 and the worker both
        # fail closed at exactly 120 accepted commits.
        "optimizer_steps": 150,
        "stage1_stop_step": 120,
        "output_dir": str(output),
    }
    config["data"]["schedule_path"] = str(package / "prompt_schedule.json")
    method: dict[str, Any] = {
        "schema_version": 3,
        "method_id": method_id,
        "teacher_route": (
            "fixed_medical_base_1_to_1"
            if method_id == "IDT"
            else "adaptive_medical_base"
        ),
        "window_steps": 30,
        "source_route_orthogonality_required": True,
        "controller_steps": [30, 60, 90, 120],
        "accepted_optimizer_commits": 120,
        "fresh_v0_required": True,
        "common_protocol_sha256": COMMON_SEMANTIC_SHA256,
        "final_authorized": False,
    }
    if method_id == "CA-OPD":
        method["router"] = {
            "medical_target": float(metrics["B1"]["medical_accuracy"]),
            "general_baseline": float(metrics["B0"]["general_micro_accuracy"]),
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
            "early_stop_patience": 3,
            "early_stop_min_improvement": 0.002,
            "initial_p_medical": 0.5,
        }
        method["domain_kl_safety"] = {
            "protocol": "phase0_frozen_domain_kl_scale",
            "kappa_medical": 1.0,
            "kappa_base": 1.0,
            "rho": 0.9,
            "eps": 1.0e-6,
            "scale_formula": "min(1,kappa/(abs(ema_reverse_kl)+eps))",
            "amplification_forbidden": True,
        }
    config["formal_method_v3"] = method
    formal_b2_runtime_config_v2(config)
    provenance = {
        "schema_version": 3,
        "artifact_kind": "p6_formal_method_provenance",
        "method_id": method_id,
        "source_b2_package_content_sha256": source_index["package_content_sha256"],
        "controller_report_sha256": _sha_file(controller_report),
        "selected_b2_snapshot": str(selected_b2_snapshot),
        "selected_b2_adapter_sha256": snapshot["adapter_sha256"],
        "correct_b1_adapter_sha256": controller["route_identities"]["B1"]["adapter_ordered_sha256"],
        "common_protocol_sha256": COMMON_SEMANTIC_SHA256,
        "data_schedule_sha256": schedule["schedule_sha256"],
        "teacher_ordered_sha256": TEACHER_ORDERED_SHA256,
        "teacher_weight_sha256": TEACHER_WEIGHT_SHA256,
        "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
        "written_before_idt_or_ca_results": True,
        "final_access_count": 0,
    }
    common = deepcopy(template)
    common["status"] = "authorized"
    common["authorization"] = {
        "formal_training_authorized": True,
        "launch_in_p6": True,
    }
    core = {
        "formal_method_config.json": config,
        "prompt_schedule.json": schedule,
        "data_authority.json": authority,
        "method_common_protocol.json": common,
        "provenance.json": provenance,
    }
    package.mkdir(parents=True)
    for name, value in core.items():
        _atomic_json(package / name, value)
    files = {
        name: {
            "sha256": _sha_file(package / name),
            "size_bytes": (package / name).stat().st_size,
        }
        for name in sorted(core)
    }
    index = {
        "schema_version": 3,
        "artifact_kind": "p6_formal_method_package_index_v3",
        "method_id": method_id,
        "run_id": output.name,
        "files": files,
        "package_content_sha256": _canonical_sha(files),
        "config_sha256": files["formal_method_config.json"]["sha256"],
        "manifest_sha256": authority["manifest_sha256"],
        "schedule_semantic_sha256": schedule["schedule_sha256"],
        "common_protocol_sha256": COMMON_SEMANTIC_SHA256,
    }
    _atomic_json(package / "package_index.json", index)
    authorization = {
        "schema_version": 3,
        "artifact_kind": "p6_formal_method_authorization_v3",
        "method_id": method_id,
        "formal_training_authorized": True,
        "user_authorized_this_turn": True,
        "fresh_v0_required": True,
        "accepted_optimizer_commits_target": 120,
        "git_clean_committed": True,
        "git_branch": _git("branch", "--show-current"),
        "git_head": _git("rev-parse", "HEAD"),
        "package_content_sha256": index["package_content_sha256"],
        "final_authorized": False,
    }
    _atomic_json(package / "authorization.json", authorization)
    return verify_method_package_v3(package, require_clean_git=True)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("IDT", "CA-OPD"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b2-package", type=Path, required=True)
    parser.add_argument("--controller-report", type=Path, required=True)
    parser.add_argument("--selected-b2-snapshot", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_method_package_v3(
        method_id=args.method,
        package=args.package,
        output=args.output,
        b2_package=args.b2_package,
        controller_report=args.controller_report,
        selected_b2_snapshot=args.selected_b2_snapshot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "build_method_package_v3",
    "verify_method_package_v3",
]
