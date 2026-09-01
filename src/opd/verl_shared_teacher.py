"""Legacy diagnostic-only veRL 0.8 patch for one shared vLLM engine.

P3.2 invalidated this prompt-logprob backend for formal LoRA advantages. The
actor, rollout, OPD loss and optimizer remain upstream veRL. Only the
Teacher manager/client boundary is replaced so the two routing keys share one
GPU1 backbone and differ solely by the Medical LoRA request.
"""

from __future__ import annotations

import os
from typing import Any

from src.teacher.shared_runtime import (
    SharedTeacherRuntimeError,
    SharedTeacherRuntimeManager,
    load_shared_teacher_runtime_config,
)


def _runtime_config_path() -> str:
    value = os.environ.get("CA_OPD_TEACHER_CONFIG", "").strip()
    if not value:
        raise SharedTeacherRuntimeError("CA_OPD_TEACHER_CONFIG is required")
    return value


def _validate_alignment_lengths(
    teacher_ids_length: int, teacher_logprobs_length: int, sequence_length: int
) -> None:
    """Fail if either Teacher tensor is not aligned to the Student trajectory."""

    if not teacher_ids_length == teacher_logprobs_length == sequence_length:
        raise SharedTeacherRuntimeError("Teacher token/logprob alignment mismatch")


class ExternalSharedTeacherModelManager:
    """veRL manager replacement that returns two clients for one external endpoint."""

    def __init__(self, config: Any, resource_pool: Any) -> None:
        self.config = config
        self.resource_pool = resource_pool  # GPU1 remains reserved by veRL/Ray.
        self.runtime = SharedTeacherRuntimeManager(
            load_shared_teacher_runtime_config(_runtime_config_path())
        )

    def get_client(self):
        return self.runtime.get_client()


class SharedAsyncTeacherLLMServerManager:
    """Route veRL's exact prompt+response ids to Base/Medical clients."""

    def __init__(self, config: Any, teacher_client: dict[str, Any]) -> None:
        self.teacher_key = str(config.distillation.teacher_key)
        if set(teacher_client) != {"base", "medical"}:
            raise SharedTeacherRuntimeError("shared Teacher client keys must be base and medical")
        self.teacher_client = teacher_client

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ):
        if routing_key not in self.teacher_client:
            raise SharedTeacherRuntimeError(f"unknown teacher route: {routing_key}")
        result = await self.teacher_client[routing_key].generate(
            request_id=f"verl-{routing_key}-{len(sequence_ids)}",
            prompt_ids=sequence_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
            image_data=(multi_modal_data or {}).get("images"),
            video_data=(multi_modal_data or {}).get("videos"),
            audio_data=(multi_modal_data or {}).get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        fields = result.extra_fields
        import torch  # GPU runtime only; kept out of config/CPU patch tests.

        teacher_ids = torch.tensor(fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(fields["prompt_logprobs"])
        _validate_alignment_lengths(
            int(teacher_ids.shape[0]), int(teacher_logprobs.shape[0]), len(sequence_ids)
        )
        return teacher_ids, teacher_logprobs


def install_verl_shared_teacher_patch(
    *, teacher_loop_module: Any | None = None, teacher_manager_module: Any | None = None
) -> dict[str, Any]:
    """Patch exactly two public Teacher-manager symbols before veRL trainer import."""

    if teacher_loop_module is None:
        import verl.experimental.teacher_loop as teacher_loop_module
    if teacher_manager_module is None:
        import verl.experimental.teacher_loop.teacher_manager as teacher_manager_module
    teacher_loop_module.MultiTeacherModelManager = ExternalSharedTeacherModelManager
    teacher_manager_module.AsyncTeacherLLMServerManager = SharedAsyncTeacherLLMServerManager
    return {
        "patch_scope": "teacher_manager_boundary_only",
        "shared_engine_instances": 1,
        "routes": ["base", "medical"],
        "gpu_runtime_exercised": False,
        "formal_enabled": False,
        "diagnostic_only": True,
    }
