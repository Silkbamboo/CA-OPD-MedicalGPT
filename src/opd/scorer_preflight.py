"""CPU-safe preflight for the narrow P4.0 GPU scorer calibration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import argparse
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MEDICAL_ADAPTER_SHA256 = "a0951a7d854f4907e6779408353455dc8e0f4af14b05fa505aba68bee70f8bc2"
TEACHER_MANIFEST_SHA256 = "80670f6e32e02bc1ea619bc94b561fedd872418b55621cbbfafbd2d0d36b5a67"
_SUPERVISION_FIELDS = {
    "answer", "answer_idx", "answer_index", "gold", "label", "labels",
    "solution", "reference_answer", "ground_truth", "reward",
}


class PreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: str, repo_root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _ordered_adapter_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = root / name
        if not path.is_file():
            raise PreflightError(f"Teacher adapter lacks {name}")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _assert_no_forbidden_data_reference(data: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(data), ensure_ascii=False).lower()
    if "final" in encoded or "confirmation" in encoded:
        raise PreflightError("final/confirmation manifests are forbidden in scorer calibration")
    if "controller" in encoded:
        raise PreflightError("controller manifests are forbidden in scorer calibration")


def _find_supervision(value: Any, path: str = "row") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).lower() in _SUPERVISION_FIELDS:
                return child
            found = _find_supervision(item, child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_supervision(item, f"{path}[{index}]")
            if found:
                return found
    return None


def _validate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"trajectory manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contains_labels") is not False:
        raise PreflightError("trajectory manifest must attest contains_labels=false")
    if payload.get("contains_raw_text") is not False:
        raise PreflightError("versioned trajectory manifest must not contain raw text")
    if int(payload.get("count", 0)) < 1:
        raise PreflightError("trajectory manifest is empty")
    return payload


def _validate_private_jsonl(path: Path) -> dict[str, Any]:
    count = 0
    role_counts: dict[str, int] = {}
    response_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = _find_supervision(row)
            if forbidden:
                raise PreflightError(
                    f"private trajectory contains supervision at line {line_number}: {forbidden}"
                )
            if row.get("contains_labels") is not False:
                raise PreflightError("private trajectory must attest contains_labels=false")
            source_role = str(row.get("source_role", "")).lower()
            if "final" in source_role or "confirmation" in source_role:
                raise PreflightError("private trajectory references final/confirmation")
            if not isinstance(row.get("prompt_ids"), list) or not row["prompt_ids"]:
                raise PreflightError("private trajectory lacks prompt token IDs")
            role = str(row.get("source_role", ""))
            role_counts[role] = role_counts.get(role, 0) + 1
            if isinstance(row.get("response_ids"), list) and row["response_ids"]:
                response_rows += 1
            count += 1
    if count < 1:
        raise PreflightError("private trajectory artifact is empty")
    return {
        "count": count,
        "contains_labels": False,
        "role_counts": role_counts,
        "response_rows": response_rows,
    }


def preflight(
    config: Mapping[str, Any], *, execute_gpu: bool, require_clean_git: bool = True,
    repo_root: str | Path | None = None,
    expected_teacher_manifest_sha256: str = TEACHER_MANIFEST_SHA256,
    expected_adapter_sha256: str = MEDICAL_ADAPTER_SHA256,
    installed_versions: Mapping[str, str] | None = None,
    gpu_inventory: Sequence[Mapping[str, Any]] | None = None,
    stale_processes: Sequence[str] | None = None,
    gpu_processes: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    run = config.get("run", {})
    if run.get("stage") != "opd_scorer_calibration" or run.get("calibration_only") is not True:
        raise PreflightError("run must be calibration_only OPD scorer calibration")
    if run.get("formal_opd_training") is not False:
        raise PreflightError("formal OPD training is forbidden in calibration")
    if config.get("artifacts", {}).get("allow_formal_checkpoint") is not False:
        raise PreflightError("formal Student checkpoint is forbidden in calibration")
    model = config.get("model", {})
    if model.get("revision") != BASE_REVISION or model.get("tokenizer_revision") != BASE_REVISION:
        raise PreflightError("Base/tokenizer revision mismatch")
    model_root = _path(str(model.get("id", "")), root).resolve()
    if not model_root.is_dir():
        raise PreflightError("verified Base model directory is missing")
    model_manifest = _path(str(model.get("artifact_manifest_path", "")), root)
    if (
        not model_manifest.is_file()
        or _sha256(model_manifest) != model.get("artifact_manifest_sha256")
    ):
        raise PreflightError("Base/tokenizer artifact manifest SHA mismatch")
    model_payload = json.loads(model_manifest.read_text(encoding="utf-8"))
    if model_payload.get("tokenizer_revision") != BASE_REVISION:
        raise PreflightError("Base artifact tokenizer revision mismatch")
    teacher = config.get("teacher", {})
    if teacher.get("adapter_sha256") != expected_adapter_sha256:
        raise PreflightError("Medical adapter SHA mismatch")
    teacher_manifest = _path(str(teacher.get("manifest_path", "")), root)
    if (
        teacher.get("manifest_sha256") != expected_teacher_manifest_sha256
        or not teacher_manifest.is_file()
        or _sha256(teacher_manifest) != expected_teacher_manifest_sha256
    ):
        raise PreflightError("Teacher manifest SHA mismatch")
    teacher_payload = json.loads(teacher_manifest.read_text(encoding="utf-8"))
    if (
        teacher_payload.get("status") != "teacher_frozen_confirmed"
        or teacher_payload.get("teacher_knowledge_ready") is not True
        or teacher_payload.get("OPD_scoring_backend_ready") is not False
        or teacher_payload.get("base_model_revision") != BASE_REVISION
        or teacher_payload.get("tokenizer_revision") != BASE_REVISION
    ):
        raise PreflightError("Teacher is not the frozen confirmed artifact")
    adapter_path = _path(str(teacher.get("adapter_path", "")), root).resolve()
    if str(adapter_path) != str(Path(str(teacher_payload.get("adapter_path", ""))).resolve()):
        raise PreflightError("Teacher adapter path differs from the frozen manifest")
    if (
        teacher_payload.get("adapter_sha256") != teacher.get("adapter_sha256")
        or _ordered_adapter_sha256(adapter_path) != teacher.get("adapter_sha256")
    ):
        raise PreflightError("Teacher ordered adapter SHA mismatch")
    weight_path = adapter_path / "adapter_model.safetensors"
    if (
        teacher_payload.get("adapter_weight_sha256") != teacher.get("adapter_weight_sha256")
        or _sha256(weight_path) != teacher.get("adapter_weight_sha256")
    ):
        raise PreflightError("Teacher adapter weight SHA mismatch")
    if "route_config" in teacher:
        route_config = _path(str(teacher["route_config"]), root)
        if not route_config.is_file() or _sha256(route_config) != teacher.get("route_config_sha256"):
            raise PreflightError("Teacher route config SHA mismatch")
    data = config.get("data", {})
    _assert_no_forbidden_data_reference(data)
    trajectory_path = _path(str(data.get("trajectory_manifest", "")), root)
    live_path = _path(str(data.get("live_prompt_manifest", "")), root)
    trajectory = _validate_manifest(trajectory_path)
    live = _validate_manifest(live_path)
    if int(trajectory["count"]) != 12 or int(live["count"]) != 4:
        raise PreflightError("calibration requires exactly 12 replay and 4 live rows")
    for path, key in (
        (trajectory_path, "trajectory_manifest_file_sha256"),
        (live_path, "live_prompt_manifest_file_sha256"),
    ):
        expected = data.get(key)
        if expected is not None and _sha256(path) != expected:
            raise PreflightError(f"{key} mismatch")
    for path_key, sha_key in (
        ("private_replay_path", "private_replay_sha256"),
        ("private_live_prompt_path", "private_live_prompt_sha256"),
    ):
        if path_key in data or sha_key in data:
            private = _path(str(data.get(path_key, "")), root)
            if not private.is_file() or _sha256(private) != data.get(sha_key):
                raise PreflightError(f"{path_key} identity mismatch")
            private_report = _validate_private_jsonl(private)
            expected_count = trajectory["count"] if path_key == "private_replay_path" else live["count"]
            if private_report["count"] != int(expected_count):
                raise PreflightError(f"{path_key} count differs from its public manifest")
            expected_roles = (
                {"medical_opd_cmb": 6, "p3_7_b0_open_diagnostic": 6}
                if path_key == "private_replay_path"
                else {"medical_opd_o1": 2, "medical_opd_cmb": 2}
            )
            expected_response_rows = 12 if path_key == "private_replay_path" else 0
            if private_report["role_counts"] != expected_roles:
                raise PreflightError(f"{path_key} role composition drift")
            if private_report["response_rows"] != expected_response_rows:
                raise PreflightError(f"{path_key} prompt/response contract drift")
    scoring = config.get("scoring", {})
    if scoring.get("formal_backend") != "transformers_direct_trajectory_logits":
        raise PreflightError("formal scorer must use the Transformers reference backend")
    if scoring.get("repeatability_tolerance") != 1e-4:
        raise PreflightError("repeatability tolerance must remain 1e-4")
    if "maximum_batch_size" in scoring and (
        scoring.get("maximum_batch_size") != 2
        or scoring.get("length_bucket_width") != 128
    ):
        raise PreflightError("Transformers length-bucket batch contract drift")
    if "equivalence_config" in scoring:
        equivalence = _path(str(scoring["equivalence_config"]), root)
        if not equivalence.is_file() or _sha256(equivalence) != scoring.get("equivalence_config_sha256"):
            raise PreflightError("scorer equivalence config SHA mismatch")
    vllm = scoring.get("vllm", {})
    if vllm.get("formal_enabled") is not False or vllm.get("diagnostic_only") is not True:
        raise PreflightError("vLLM prompt-logprobs must remain diagnostic_only")
    algorithm = config.get("algorithm", {})
    if (
        algorithm.get("beta") != 1.0
        or algorithm.get("use_task_rewards") is not False
        or algorithm.get("reference_policy_kl") is not False
        or algorithm.get("old_logprob_source") != "sampling_time_policy"
    ):
        raise PreflightError("calibration algorithm contract drift")
    if algorithm.get("one_step_only") is not True or algorithm.get("save_student_checkpoint") is not False:
        raise PreflightError("calibration must remain one-step-only without a Student checkpoint")
    if "contract" in algorithm:
        contract = _path(str(algorithm["contract"]), root)
        if not contract.is_file() or _sha256(contract) != algorithm.get("contract_sha256"):
            raise PreflightError("PG-OPD contract SHA mismatch")
    if "student_lora" in algorithm and algorithm["student_lora"] != {
        "rank": 16, "alpha": 32, "dropout": 0.0,
        "target_modules": "all-linear", "calibration_lr": 3e-5,
    }:
        raise PreflightError("calibration Student LoRA contract drift")
    output = _path(str(run.get("output_dir", "")), root)
    if output.exists() and any(output.iterdir()):
        raise PreflightError("calibration output directory must be new/empty")
    if require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, text=True, capture_output=True
        ).stdout
        if status:
            raise PreflightError("formal calibration requires a clean committed worktree")
    expected_versions = config.get("versions", {})
    observed_versions = dict(installed_versions or {
        name: package_version(name) for name in ("torch", "transformers", "peft", "vllm", "verl")
    })
    if observed_versions != dict(expected_versions):
        raise PreflightError(
            f"calibration environment version mismatch: expected={expected_versions} "
            f"observed={observed_versions}"
        )
    resources = config.get("resources", {})
    if resources.get("required_gpus") != 2:
        raise PreflightError("calibration topology requires exactly two GPUs")
    free_gib = shutil.disk_usage(output.parent if output.parent.exists() else root).free / 2**30
    if free_gib < float(resources.get("minimum_free_disk_gib", 10)):
        raise PreflightError("insufficient disk for calibration artifacts")
    if execute_gpu:
        if os.environ.get("CA_OPD_ALLOW_OPD_SCORER_CALIBRATION_GPU") != "1":
            raise PreflightError("GPU scorer calibration lacks explicit environment authorization")
        if gpu_inventory is None:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True,
            ).stdout.splitlines()
            parsed_inventory = []
            for line in query:
                if not line.strip():
                    continue
                index, name, uuid, total, used = [item.strip() for item in line.split(",", 4)]
                parsed_inventory.append({
                    "index": int(index), "name": name, "uuid": uuid,
                    "memory_total_mib": int(total), "memory_used_mib": int(used),
                })
            gpu_inventory = parsed_inventory
        if len(gpu_inventory) != 2:
            raise PreflightError("GPU calibration requires exactly two visible GPUs")
        if any(
            "RTX 3090" not in str(item.get("name", ""))
            or int(item.get("memory_total_mib", 0)) < 24576
            for item in gpu_inventory
        ):
            raise PreflightError("GPU calibration requires two 24 GiB RTX 3090s")
        if any(int(item.get("memory_used_mib", -1)) != 0 for item in gpu_inventory):
            raise PreflightError("GPU calibration requires both GPUs to be idle at 0 MiB")
        if gpu_processes is None:
            gpu_processes = [
                line.strip() for line in subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid",
                     "--format=csv,noheader"],
                    check=True, text=True, capture_output=True,
                ).stdout.splitlines() if line.strip()
            ]
        if gpu_processes:
            raise PreflightError(f"unexpected GPU compute process detected: {list(gpu_processes)}")
        if stale_processes is None:
            process_rows = subprocess.run(
                ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
            ).stdout.splitlines()
            markers = ("raylet", "ray::", "vllm", "torchrun", "verl.trainer")
            stale_processes = [row.strip() for row in process_rows if any(marker in row for marker in markers)]
        if stale_processes:
            raise PreflightError(f"stale Ray/vLLM/torchrun process detected: {list(stale_processes)}")
    return {
        "status": "ready_gpu_execution_authorized" if execute_gpu else "ready_waiting_for_gpu_opd_scorer_calibration",
        "loaded_real_model": False,
        "gpu_used": False,
        "trajectory_count": int(trajectory["count"]),
        "live_prompt_count": int(live["count"]),
        "teacher_manifest_sha256": teacher["manifest_sha256"],
        "versions": observed_versions,
        "gpu_idle": True if execute_gpu else None,
        "gpu_inventory": list(gpu_inventory or ()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P4.0 OPD scorer calibration preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--allow-dirty-for-development", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = preflight(
        config,
        execute_gpu=args.execute_gpu,
        require_clean_git=not args.allow_dirty_for_development,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
