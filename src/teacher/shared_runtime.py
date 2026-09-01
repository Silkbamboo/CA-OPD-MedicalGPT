"""veRL-facing clients for one external shared-backbone Base/Medical Teacher service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Protocol
from urllib import request as urllib_request

import yaml


MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class SharedTeacherRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SharedTeacherRuntimeConfig:
    model_path: str
    model_revision: str
    tokenizer_revision: str
    medical_adapter_path: str
    medical_adapter_manifest: str
    routes: tuple[str, str]
    host: str
    port: int
    gpu_id: int
    gpu_memory_utilization: float
    max_model_len: int
    max_lora_rank: int
    request_timeout_seconds: int
    health_timeout_seconds: int
    runtime_status: str

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_shared_teacher_runtime_config(path: str | Path) -> SharedTeacherRuntimeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SharedTeacherRuntimeError("teacher service config must be a mapping")
    if payload.get("schema_version") != 1 or payload.get("backend") != "vllm_shared_backbone_lora":
        raise SharedTeacherRuntimeError("unsupported shared Teacher service schema/backend")
    if payload.get("model_revision") != MODEL_REVISION or payload.get("tokenizer_revision") != MODEL_REVISION:
        raise SharedTeacherRuntimeError("Teacher model/tokenizer revision mismatch")
    routes = tuple(payload.get("routes") or ())
    if routes != ("base", "medical"):
        raise SharedTeacherRuntimeError("Teacher routes must be exactly base and medical")
    if payload.get("host") not in {"127.0.0.1", "localhost"}:
        raise SharedTeacherRuntimeError("Teacher scoring endpoint must be loopback-only")
    if int(payload.get("gpu_id", -1)) != 1:
        raise SharedTeacherRuntimeError("shared Teacher must be assigned to GPU1")
    if int(payload.get("max_lora_rank", 0)) != 16:
        raise SharedTeacherRuntimeError("shared Teacher Medical adapter rank must be 16")
    return SharedTeacherRuntimeConfig(
        model_path=str(payload["model_path"]),
        model_revision=str(payload["model_revision"]),
        tokenizer_revision=str(payload["tokenizer_revision"]),
        medical_adapter_path=str(payload["medical_adapter_path"]),
        medical_adapter_manifest=str(payload["medical_adapter_manifest"]),
        routes=("base", "medical"),
        host=str(payload["host"]),
        port=int(payload["port"]),
        gpu_id=1,
        gpu_memory_utilization=float(payload["gpu_memory_utilization"]),
        max_model_len=int(payload["max_model_len"]),
        max_lora_rank=16,
        request_timeout_seconds=int(payload["request_timeout_seconds"]),
        health_timeout_seconds=int(payload["health_timeout_seconds"]),
        runtime_status=str(payload["runtime_status"]),
    )


class TeacherHTTPTransport(Protocol):
    async def post_score(self, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]: ...
    async def health(self, endpoint: str, timeout: int) -> bool: ...


class UrllibTeacherHTTPTransport:
    async def post_score(self, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        def send() -> dict[str, Any]:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            req = urllib_request.Request(
                endpoint + "/score",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=timeout) as response:
                value = json.loads(response.read())
            if not isinstance(value, dict):
                raise SharedTeacherRuntimeError("Teacher server returned a non-object")
            return value

        return await asyncio.to_thread(send)

    async def health(self, endpoint: str, timeout: int) -> bool:
        def check() -> bool:
            try:
                with urllib_request.urlopen(endpoint + "/health", timeout=timeout) as response:
                    return response.status == 200
            except Exception:
                return False

        return await asyncio.to_thread(check)


class SharedTeacherClient:
    def __init__(self, route: str, config: SharedTeacherRuntimeConfig, transport: TeacherHTTPTransport, on_request):
        self.route = route
        self.config = config
        self.transport = transport
        self._on_request = on_request

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        **_: Any,
    ) -> Any:
        if sampling_params.get("prompt_logprobs") != 0 or sampling_params.get("max_tokens") != 1:
            raise SharedTeacherRuntimeError("Teacher client accepts only exact prompt-logprob scoring")
        payload = {
            "request_id": request_id,
            "teacher_route": self.route,
            "prompt_ids": [int(token) for token in prompt_ids],
            "sampling_params": dict(sampling_params),
        }
        result = await self.transport.post_score(
            self.config.endpoint, payload, self.config.request_timeout_seconds
        )
        if result.get("request_id") != request_id or result.get("teacher_route") != self.route:
            raise SharedTeacherRuntimeError("Teacher response identity mismatch")
        if result.get("prompt_ids") != payload["prompt_ids"]:
            raise SharedTeacherRuntimeError("Teacher did not score the exact student trajectory")
        expected_adapter = self.route == "medical"
        if result.get("adapter_applied") is not expected_adapter:
            raise SharedTeacherRuntimeError("Teacher adapter state differs from the selected route")
        scores = result.get("prompt_logprobs")
        if not isinstance(scores, list) or len(scores) != len(prompt_ids):
            raise SharedTeacherRuntimeError("Teacher logprob alignment/length mismatch")
        self._on_request(self.route)
        return SimpleNamespace(
            extra_fields={
                "prompt_ids": list(prompt_ids),
                "prompt_logprobs": scores,
                "teacher_route": self.route,
                "adapter_applied": expected_adapter,
            }
        )


class SharedTeacherRuntimeManager:
    """Drop-in client dictionary boundary used by veRL's AsyncTeacher manager."""

    def __init__(
        self,
        config: SharedTeacherRuntimeConfig,
        *,
        transport: TeacherHTTPTransport | None = None,
        restart_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibTeacherHTTPTransport()
        self.restart_callback = restart_callback
        self._requests = {route: 0 for route in config.routes}
        self._restarts = 0
        self._clients = {
            route: SharedTeacherClient(route, config, self.transport, self._record_request)
            for route in config.routes
        }

    def _record_request(self, route: str) -> None:
        self._requests[route] += 1

    def client_for(self, route: str) -> SharedTeacherClient:
        try:
            return self._clients[route]
        except KeyError as error:
            raise SharedTeacherRuntimeError(f"unknown teacher route: {route}") from error

    def get_client(self) -> dict[str, SharedTeacherClient]:
        return dict(self._clients)

    async def ensure_healthy(self, *, allow_restart: bool = False) -> bool:
        if await self.transport.health(self.config.endpoint, self.config.health_timeout_seconds):
            return True
        if not allow_restart or self.restart_callback is None:
            raise SharedTeacherRuntimeError("shared Teacher service health check failed")
        await self.restart_callback()
        self._restarts += 1
        if not await self.transport.health(self.config.endpoint, self.config.health_timeout_seconds):
            raise SharedTeacherRuntimeError("shared Teacher service remained unhealthy after restart")
        return True

    def metrics(self) -> dict[str, Any]:
        return {
            "shared_engine_instances": 1,
            "endpoint": self.config.endpoint,
            "requests": dict(self._requests),
            "restarts": self._restarts,
        }
