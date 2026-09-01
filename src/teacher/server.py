"""HTTP process boundary for the single GPU1 shared-backbone Teacher engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, MutableMapping

from src.teacher.lora_router_service import (
    TeacherScoreRequest,
    build_vllm_service,
)
from src.teacher.shared_runtime import load_shared_teacher_runtime_config


def prepare_teacher_gpu_environment(config: Any, environ: MutableMapping[str, str]) -> dict[str, Any]:
    """Pin the external shared backbone to physical GPU1 before importing vLLM."""

    environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    return {
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_lora_rank": config.max_lora_rank,
        "max_model_len": config.max_model_len,
    }


def score_payload(service: Any, payload: dict[str, Any]) -> dict[str, Any]:
    sampling = payload.get("sampling_params") or {}
    if sampling.get("max_tokens") != 1 or sampling.get("prompt_logprobs") != 0:
        raise ValueError("Teacher HTTP endpoint accepts prompt-logprob scoring only")
    route = str(payload.get("teacher_route", ""))
    tokens = tuple(int(value) for value in payload.get("prompt_ids") or ())
    request = TeacherScoreRequest(
        request_id=str(payload.get("request_id", "")),
        teacher_id=route,
        token_ids=(tokens,),
        prompt_lengths=(1,),
    )
    response = service.score(request)
    response.validate_against(request)
    scores = list(response.token_logprobs[0])
    return {
        "request_id": response.request_id,
        "teacher_route": response.teacher_id,
        "prompt_ids": list(response.token_ids[0]),
        # veRL expects one entry per sequence token; the first token has no
        # autoregressive target and is represented by a zero placeholder.
        "prompt_logprobs": [0.0, *scores],
        "adapter_applied": response.adapter_applied,
        "shared_engine_instances": 1,
    }


def create_app(service: Any):  # pragma: no cover - exercised on GPU host
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="CA-OPD shared Teacher", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **service.metrics()}

    @app.post("/score")
    def score(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return score_payload(service, payload)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"{type(error).__name__}: {error}") from error

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU only
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_shared_teacher_runtime_config(args.config)
    adapter_manifest = Path(config.medical_adapter_manifest)
    if not adapter_manifest.is_file():
        raise FileNotFoundError(f"Medical adapter manifest is missing: {adapter_manifest}")
    manifest = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    if manifest.get("adapter_sha256") is None:
        raise ValueError("Medical adapter manifest lacks adapter_sha256")
    engine_kwargs = prepare_teacher_gpu_environment(config, os.environ)
    service = build_vllm_service(
        config.model_path,
        config.medical_adapter_path,
        diagnostic_only_ack=os.environ.get("CA_OPD_ALLOW_VLLM_DIAGNOSTIC") == "1",
        **engine_kwargs,
    )
    import uvicorn

    uvicorn.run(create_app(service), host=config.host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
