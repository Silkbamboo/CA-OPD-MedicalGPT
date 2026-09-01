"""Production veRL launch preparation for prompt-only, routed CA-OPD windows."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.opd.run_config import load_run
from src.opd.ca_runtime import next_ca_window, record_ca_controller_result
from src.opd.export_lora import export_verl_lora_adapter
import yaml
from src.data.chat import format_mcq_question


_SUPERVISION = {
    "answer", "answer_idx", "label", "reasoning", "response", "solution",
    "output", "completion",
}


class OPDProductionError(RuntimeError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
OPD_ROLES = ("general_anchors", "medical_opd_cmb", "medical_opd_o1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_opd_teacher_gate(
    run_config: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Block every real Teacher process until frozen Controller v2 evidence passes."""

    from src.opd.teacher_gate import (
        TeacherGateError,
        assert_teacher_ready_for_opd,
        load_teacher_gate_config,
    )

    values = os.environ if environ is None else environ
    gate_value = run_config.get("run", {}).get("teacher_gate_config")
    if not gate_value:
        raise OPDProductionError("Teacher gate config is missing")
    gate_path = Path(str(gate_value))
    if not gate_path.is_absolute():
        gate_path = REPO_ROOT / gate_path
    required = {
        "CA_OPD_TEACHER_READINESS_PATH": "readiness_path",
        "CA_OPD_TEACHER_READINESS_SHA256": "expected_sha256",
        "CA_OPD_CONTROLLER_V2_ARTIFACT_MANIFEST_PATH": "controller_artifact_manifest_path",
        "CA_OPD_CONTROLLER_V2_ARTIFACT_MANIFEST_SHA256": "controller_artifact_manifest_sha256",
    }
    missing = [name for name in required if not str(values.get(name) or "").strip()]
    if missing:
        raise OPDProductionError(f"Teacher gate evidence is missing: {missing}")
    kwargs = {argument: values[name] for name, argument in required.items()}
    try:
        gate = load_teacher_gate_config(gate_path)
        return assert_teacher_ready_for_opd(gate, **kwargs)
    except (OSError, ValueError, TeacherGateError) as error:
        raise OPDProductionError(f"Teacher gate rejected OPD execution: {error}") from error


def resolve_opd_source_files(manifest_path: str | Path) -> list[Path]:
    """Resolve and verify only the three frozen prompt-only training roles."""

    path = Path(manifest_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("data_protocol_version") != "ca-opd-data-v2"
        or manifest.get("primary_final_frozen") is not True
        or manifest.get("final_authorized") is not False
    ):
        raise OPDProductionError("OPD manifest is not the frozen, unauthorized-final v2 artifact")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(OPD_ROLES):
        raise OPDProductionError("OPD manifest roles differ from the frozen prompt-only contract")
    resolved: list[Path] = []
    for role in OPD_ROLES:
        files = roles[role].get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise OPDProductionError(f"{role} must bind exactly one records artifact")
        item = files[0]
        artifact = Path(str(item.get("path", "")))
        if not artifact.is_absolute():
            artifact = REPO_ROOT / artifact
        if not artifact.is_file() or _sha256(artifact) != str(item.get("sha256", "")):
            raise OPDProductionError(f"{role} artifact SHA mismatch")
        resolved.append(artifact)
    return resolved


def build_execution_plan(method: str, *, total_steps: int, window_steps: int) -> list[dict[str, Any]]:
    """Freeze cumulative checkpoint boundaries without inventing controller scores."""

    if total_steps < 1 or window_steps < 1:
        raise OPDProductionError("total_steps and window_steps must be positive")
    if method in {"medical", "idt_1to1"}:
        return [{
            "segment": 0,
            "teacher_method": method,
            "start_step": 0,
            "end_step": total_steps,
            "resume": False,
            "requires_controller_result": False,
        }]
    if method == "sar":
        midpoint = total_steps // 2
        if midpoint < 1:
            raise OPDProductionError("SAR requires at least two optimizer steps")
        return [
            {
                "segment": 0, "teacher_method": "sar_medical", "start_step": 0,
                "end_step": midpoint, "resume": False, "requires_controller_result": False,
            },
            {
                "segment": 1, "teacher_method": "sar_base", "start_step": midpoint,
                "end_step": total_steps, "resume": True, "requires_controller_result": False,
            },
        ]
    if method == "ca_opd":
        plan = []
        start = 0
        segment = 0
        while start < total_steps:
            end = min(total_steps, start + window_steps)
            plan.append({
                "segment": segment,
                "teacher_method": "ca_opd",
                "start_step": start,
                "end_step": end,
                "resume": start > 0,
                # Window zero starts at the protocol's pre-result 1:1 prior;
                # every later probability must be computed from a real controller result.
                "requires_controller_result": start > 0,
            })
            start, segment = end, segment + 1
        return plan
    raise OPDProductionError(f"unsupported execution method: {method}")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OPDProductionError(f"{path}:{line_number} is not an object")
            yield value


def _route(method: str, sample_id: str, seed: int, p_medical: float | None) -> str:
    if method in {"medical", "sar_medical"}:
        return "medical"
    if method in {"base", "sar_base"}:
        return "base"
    rank = int(hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest(), 16)
    if method == "idt_1to1":
        return "medical" if rank % 2 == 0 else "base"
    if method == "ca_opd":
        if p_medical is None or not 0.0 <= p_medical <= 1.0:
            raise OPDProductionError("CA-OPD routing requires p_medical in [0,1]")
        return "medical" if rank / (2**256 - 1) < p_medical else "base"
    raise OPDProductionError(f"unsupported routing method: {method}")


def write_routed_prompt_file(
    source_files: Iterable[str | Path],
    output_path: str | Path,
    *,
    method: str,
    seed: int,
    p_medical: float | None = None,
) -> dict[str, Any]:
    """Disk-sort prompt records and atomically publish one veRL JSONL window."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_suffix(output.suffix + ".sqlite.tmp")
    temporary = output.with_suffix(output.suffix + ".tmp")
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE rows (sample_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for source in sorted(Path(path) for path in source_files):
            for row in _iter_jsonl(source):
                role = str(row.get("target_role", ""))
                if "final" in role or "controller" in role:
                    raise OPDProductionError(f"final/controller role is forbidden in OPD: {role}")
                if _SUPERVISION & set(row):
                    raise OPDProductionError("OPD input contains a supervision field")
                question = str(row.get("question", "")).strip()
                sample_id = str(row.get("sample_id", "")).strip()
                content_hash = str(row.get("content_hash", "")).strip()
                if not question or not sample_id or len(content_hash) != 64:
                    raise OPDProductionError("OPD record lacks stable prompt identity")
                options = row.get("options")
                if options is not None:
                    if not isinstance(options, list) or not 2 <= len(options) <= 8:
                        raise OPDProductionError("OPD MCQ options are malformed")
                    prompt_content = format_mcq_question(question, [str(item) for item in options])
                else:
                    prompt_content = question
                payload = {
                    "sample_id": sample_id,
                    "target_role": role,
                    "content_hash": content_hash,
                    "prompt": [{"role": "user", "content": prompt_content}],
                    "teacher_route": _route(method, sample_id, seed, p_medical),
                }
                try:
                    connection.execute(
                        "INSERT INTO rows VALUES (?, ?)",
                        (sample_id, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                    )
                except sqlite3.IntegrityError as error:
                    raise OPDProductionError(f"duplicate sample_id in OPD inputs: {sample_id}") from error
        connection.commit()
        digest = hashlib.sha256()
        count = 0
        routes = {"base": 0, "medical": 0}
        with temporary.open("wb") as handle:
            for (encoded,) in connection.execute("SELECT payload FROM rows ORDER BY sample_id"):
                value = json.loads(encoded)
                routes[value["teacher_route"]] += 1
                line = (encoded + "\n").encode("utf-8")
                handle.write(line)
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise OPDProductionError("no OPD prompts were written")
        os.replace(temporary, output)
    finally:
        connection.close()
        database.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "count": count,
        "sha256": digest.hexdigest(),
        "routes": routes,
        "supervision_fields": 0,
        "deterministic": True,
    }


def build_verl_launch(
    config_path: str | Path,
    *,
    routed_file: str,
    calibrated_steps: int | None,
    formal_total_steps: int | None = None,
    run_dir: str | Path | None = None,
    resume_from_path: str | Path | None = None,
    save_at_step: int | None = None,
) -> dict[str, Any]:
    loaded = load_run(config_path)
    calibration = "calibration" in str(loaded.raw["run"]["run_id"])
    if calibrated_steps is None:
        raise OPDProductionError("an explicit calibrated step count is required")
    if calibration:
        if calibrated_steps != 20:
            raise OPDProductionError("OPD calibration is exactly 20 steps")
    elif not 120 <= (formal_total_steps or calibrated_steps) <= 150:
        raise OPDProductionError("formal OPD calibrated step count must be in 120..150")
    if loaded.raw["verl"]["teacher_backend"] != "shared_service":
        raise OPDProductionError("Qwen3-4B deployment requires the shared Teacher service")
    overrides = [
        item
        for item in loaded.verl.to_overrides()
        if not item.startswith("trainer.total_training_steps=")
    ]
    overrides.extend(
        [
            f"trainer.total_training_steps={calibrated_steps}",
            f"data.train_files=[{routed_file}]",
            "data.prompt_key=prompt",
            "data.return_raw_chat=true",
            "data.truncation=error",
            "data.filter_overlong_prompts=true",
            "data.filter_overlong_prompts_workers=1",
            "data.apply_chat_template_kwargs.enable_thinking=false",
            "trainer.n_gpus_per_node=1",
        ]
    )
    if run_dir is not None:
        checkpoint_root = Path(run_dir) / "checkpoints"
        overrides.extend(
            [
                "trainer.project_name=ca-opd-medicalgpt",
                f"trainer.experiment_name={loaded.raw['run']['run_id']}",
                f"trainer.default_local_dir={checkpoint_root}",
                f"trainer.save_freq={save_at_step or loaded.raw['budget']['checkpoint_every_steps']}",
                "trainer.max_actor_ckpt_to_keep=4",
            ]
        )
    if resume_from_path is None:
        overrides.append("trainer.resume_mode=disable")
    else:
        resume = Path(resume_from_path)
        if not resume.name.startswith("global_step_"):
            raise OPDProductionError("resume checkpoint must be a global_step_* directory")
        overrides.extend(["trainer.resume_mode=resume_path", f"trainer.resume_from_path={resume}"])
    return {
        "config_path": str(config_path),
        "routed_file": routed_file,
        "calibrated_steps": calibrated_steps,
        "overrides": overrides,
        "teacher_patch": "teacher_manager_boundary_only",
        "teacher_service_config": loaded.raw["verl"]["teacher_service_config"],
        "full_model_loaded": False,
    }


def execute_verl(launch: dict[str, Any]) -> None:  # pragma: no cover - GPU only
    """Install the narrow patch and hand control to upstream veRL/Hydra."""

    from src.opd.verl_shared_teacher import install_verl_shared_teacher_patch

    os.environ["CA_OPD_TEACHER_CONFIG"] = str(launch["teacher_service_config"])
    install_verl_shared_teacher_patch()
    sys.argv = ["verl.trainer.main_ppo", *launch["overrides"]]
    from verl.trainer.main_ppo import main

    main()


def _start_teacher(config_path: str, python_bin: str) -> subprocess.Popen[str]:  # pragma: no cover - GPU only
    env = dict(os.environ)
    process = subprocess.Popen(
        [python_bin, "-m", "src.teacher.server", "--config", config_path],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    endpoint = "http://127.0.0.1:8011/health"
    for _ in range(120):
        if process.poll() is not None:
            raise OPDProductionError("shared Teacher service exited before becoming healthy")
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.status == 200:
                    return process
        except Exception:
            time.sleep(1)
    process.terminate()
    raise OPDProductionError("shared Teacher service did not become healthy within 120 seconds")


def _run_static_method(
    *, config_path: str, method: str, total_steps: int, run_dir: Path
) -> dict[str, Any]:  # pragma: no cover - GPU only
    loaded = load_run(config_path)
    sources = resolve_opd_source_files(loaded.raw["data"]["manifest"])
    window_steps = int(loaded.raw["budget"]["controller_dev_every_steps"])
    plan = build_execution_plan(method, total_steps=total_steps, window_steps=window_steps)
    if method == "ca_opd":
        raise OPDProductionError(
            "CA-OPD execution requires real B0/B1 controller results and per-window evaluator "
            "feedback; use the prepared CA controller orchestration after GPU calibration"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs = run_dir / "inputs"
    inputs.mkdir()
    previous: Path | None = None
    exported: list[dict[str, Any]] = []
    for segment in plan:
        routed = inputs / f"segment-{segment['segment']:03d}.jsonl"
        route_report = write_routed_prompt_file(
            sources,
            routed,
            method=str(segment["teacher_method"]),
            seed=int(loaded.raw["run"]["seed"]),
        )
        (inputs / f"segment-{segment['segment']:03d}.json").write_text(
            json.dumps(route_report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        launch = build_verl_launch(
            config_path,
            routed_file=str(routed),
            calibrated_steps=int(segment["end_step"]),
            formal_total_steps=total_steps,
            run_dir=run_dir,
            resume_from_path=previous,
            save_at_step=int(segment["end_step"]),
        )
        execute_verl(launch)
        previous = run_dir / "checkpoints" / f"global_step_{segment['end_step']}"
        if not previous.is_dir():
            raise OPDProductionError(f"veRL did not publish expected checkpoint: {previous}")
        exported.append(
            export_verl_lora_adapter(
                previous,
                run_dir / "adapters" / f"global_step_{segment['end_step']}",
            )
        )
    return {
        "run_dir": str(run_dir), "segments": plan, "adapters": exported,
        "status": "completed_pending_gpu_metrics_and_cost_reconciliation",
    }


def _run_ca_window(
    *, config_path: str, state_path: str, total_steps: int, run_dir: Path
) -> dict[str, Any]:  # pragma: no cover - GPU only
    loaded = load_run(config_path)
    sources = resolve_opd_source_files(loaded.raw["data"]["manifest"])
    window = next_ca_window(state_path)
    if int(window["end_step"]) > total_steps:
        raise OPDProductionError("CA state exceeds the calibrated total step budget")
    if int(window["start_step"]) == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "inputs").mkdir()
        previous = None
    else:
        if not run_dir.is_dir():
            raise OPDProductionError("CA resume run directory is missing")
        previous = run_dir / "checkpoints" / f"global_step_{window['start_step']}"
        if not previous.is_dir():
            raise OPDProductionError(f"CA resume checkpoint is missing: {previous}")
    routed = run_dir / "inputs" / f"window-{window['start_step']:04d}-{window['end_step']:04d}.jsonl"
    route_report = write_routed_prompt_file(
        sources,
        routed,
        method="ca_opd",
        seed=int(loaded.raw["run"]["seed"]),
        p_medical=float(window["p_medical"]),
    )
    launch = build_verl_launch(
        config_path,
        routed_file=str(routed),
        calibrated_steps=int(window["end_step"]),
        formal_total_steps=total_steps,
        run_dir=run_dir,
        resume_from_path=previous,
        save_at_step=int(window["end_step"]),
    )
    execute_verl(launch)
    checkpoint = run_dir / "checkpoints" / f"global_step_{window['end_step']}"
    if not checkpoint.is_dir():
        raise OPDProductionError(f"veRL did not publish expected CA checkpoint: {checkpoint}")
    adapter_manifest = export_verl_lora_adapter(
        checkpoint,
        run_dir / "adapters" / f"global_step_{window['end_step']}",
    )
    return {
        "run_dir": str(run_dir),
        "window": window,
        "route_report": route_report,
        "checkpoint": str(checkpoint),
        "controller_adapter": adapter_manifest,
        "status": "awaiting_controller_evaluation",
        "final_used": False,
    }


def build_ca_eval_config(
    template_path: str | Path,
    *,
    adapter_path: str,
    output_root: str,
    completed_step: int,
) -> dict[str, Any]:
    """Bind one exported Student adapter to the frozen controller capability."""

    value = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("capability") != "controller_eval":
        raise OPDProductionError("CA evaluator template must be controller_eval")
    roles = value.get("data", {}).get("roles")
    if roles != ["medical_controller_dev", "general_controller_dev"]:
        raise OPDProductionError("CA evaluator template roles differ from frozen controller")
    if value.get("allow_final_eval") is not False or any("final" in role for role in roles):
        raise OPDProductionError("final capability is forbidden in CA routing")
    value["run_id"] = f"qwen3-4b-ca-controller-step-{completed_step:04d}"
    value["model"]["adapter_path"] = adapter_path
    value["output_root"] = output_root
    return value


def _evaluate_ca_adapter(
    *, run_dir: Path, adapter_path: str, completed_step: int
) -> Path:  # pragma: no cover - GPU only
    controller_root = run_dir / "controller"
    controller_root.mkdir(parents=True, exist_ok=True)
    config = build_ca_eval_config(
        REPO_ROOT / "configs/eval/qwen3_4b/ca_controller.yaml",
        adapter_path=adapter_path,
        output_root=str(controller_root),
        completed_step=completed_step,
    )
    config_path = run_dir / "inputs" / f"controller-step-{completed_step:04d}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "src.eval.runtime", "--config", str(config_path), "--execute"],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        check=False,
    )
    if process.returncode != 0:
        raise OPDProductionError(f"controller evaluator failed at step {completed_step}")
    aggregate = controller_root / config["run_id"] / "aggregate.json"
    if not aggregate.is_file():
        raise OPDProductionError("controller evaluator did not publish aggregate.json")
    return aggregate


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI/GPU only
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=["medical", "sar", "idt_1to1", "ca_opd"])
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--ca-state", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    loaded = load_run(args.config)
    sources = resolve_opd_source_files(loaded.raw["data"]["manifest"])
    plan = build_execution_plan(
        args.method,
        total_steps=args.steps,
        window_steps=int(loaded.raw["budget"]["controller_dev_every_steps"]),
    )
    if not args.execute:
        print(json.dumps({"status": "plan_valid", "source_files": len(sources), "segments": plan}, sort_keys=True))
        return 0
    output_root = Path(str(loaded.raw["run"]["output_root"]))
    run_dir = output_root / str(loaded.raw["run"]["run_id"])
    assert_opd_teacher_gate(loaded.raw)
    teacher = _start_teacher(
        str(loaded.raw["verl"]["teacher_service_config"]),
        sys.executable,
    )
    try:
        if args.method == "ca_opd":
            if not args.ca_state:
                raise OPDProductionError("CA-OPD execution requires --ca-state initialized from B0/B1")
            windows: list[dict[str, Any]] = []
            while True:
                window_result = _run_ca_window(
                    config_path=args.config,
                    state_path=args.ca_state,
                    total_steps=args.steps,
                    run_dir=run_dir,
                )
                aggregate = _evaluate_ca_adapter(
                    run_dir=run_dir,
                    adapter_path=window_result["controller_adapter"]["adapter_path"],
                    completed_step=int(window_result["window"]["end_step"]),
                )
                state = record_ca_controller_result(
                    args.ca_state,
                    aggregate,
                    completed_step=int(window_result["window"]["end_step"]),
                )
                window_result["controller_artifact"] = str(aggregate)
                windows.append(window_result)
                if state["status"] == "completed":
                    break
            result = {
                "run_dir": str(run_dir),
                "windows": windows,
                "status": "completed_pending_gpu_metrics_and_cost_reconciliation",
                "final_used": False,
            }
        else:
            result = _run_static_method(
                config_path=args.config,
                method=args.method,
                total_steps=args.steps,
                run_dir=run_dir,
            )
    finally:
        teacher.terminate()
        try:
            teacher.wait(timeout=20)
        except subprocess.TimeoutExpired:
            teacher.kill()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
