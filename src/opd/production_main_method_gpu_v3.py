"""P6 formal IDT/CA GPU session layered on the qualified B2 v2 kernel."""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.opd.production_b2_formal_gpu_v2 import FormalB2SessionV2
from src.opd.production_main_method_v3 import (
    DomainKLSafetyV1,
    MethodRouteStateV1,
    P6FormalMethodError,
)
from src.opd.production_qualification_two_step_gpu_v7 import (
    ProductionTwoStepQualificationV6Error,
    _atomic_json,
)
from src.opd.router import ConstraintAwareRouter, RouterConfig


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FormalMethodSessionV3(FormalB2SessionV2):
    """Qualified three-policy session whose only additions are method fields."""

    def __init__(self, config: Mapping[str, Any], **kwargs: Any) -> None:
        method = config.get("formal_method_v3")
        if not isinstance(method, Mapping) or method.get("method_id") not in {
            "IDT",
            "CA-OPD",
        }:
            raise P6FormalMethodError("formal method package identity differs")
        self.method_config = copy.deepcopy(dict(method))
        self.method_id = str(method["method_id"])
        self.route_state = MethodRouteStateV1(
            method_id=self.method_id,
            seed=int(config["run"]["seed"]),
            window_steps=int(method["window_steps"]),
        )
        if self.method_id == "CA-OPD":
            self.route_state.start_window(p_medical=0.5, start_step=0)
            safety = method.get("domain_kl_safety")
            if not isinstance(safety, Mapping):
                raise P6FormalMethodError("CA domain KL safety is absent")
            self.kl_safety: DomainKLSafetyV1 | None = DomainKLSafetyV1(
                kappa={
                    "medical": float(safety["kappa_medical"]),
                    "base": float(safety["kappa_base"]),
                },
                rho=float(safety["rho"]),
                eps=float(safety["eps"]),
            )
            router = method.get("router")
            if not isinstance(router, Mapping):
                raise P6FormalMethodError("CA Controller router is absent")
            self.ca_router: ConstraintAwareRouter | None = ConstraintAwareRouter(
                RouterConfig.from_mapping(dict(router)),
                evaluator=None,
                seed=int(config["run"]["seed"]),
            )
        else:
            self.kl_safety = None
            self.ca_router = None
        self._current_teacher_routes: tuple[str, ...] = ()
        self._current_method_evidence: dict[str, Any] = {}
        teacher = config.get("teacher")
        if not isinstance(teacher, Mapping):
            raise P6FormalMethodError("formal method Teacher identity is absent")
        adapter = Path(str(teacher["adapter_path"]))
        manifest = Path(str(teacher["manifest_path"]))
        if not (
            adapter.is_dir()
            and _sha_file(adapter / "adapter_model.safetensors")
            == teacher.get("adapter_weight_sha256")
            and manifest.is_file()
            and _sha_file(manifest) == teacher.get("manifest_sha256")
        ):
            raise P6FormalMethodError("formal method Teacher SHA differs before load")
        super().__init__(config, **kwargs)
        (self.output / "method_steps_v3").mkdir(parents=True, exist_ok=True)

    def set_ca_window_probability(self, *, p_medical: float, start_step: int) -> None:
        if self.method_id != "CA-OPD":
            raise P6FormalMethodError("only CA-OPD accepts adaptive windows")
        self.route_state.start_window(
            p_medical=float(p_medical), start_step=int(start_step)
        )

    def update_ca_controller(
        self,
        *,
        medical_accuracy: float,
        general_accuracy: float,
        completed_step: int,
    ) -> dict[str, Any]:
        if self.ca_router is None or not self.ca_router.is_window_boundary(completed_step):
            raise P6FormalMethodError("CA Controller update boundary differs")
        decision = self.ca_router.update(
            medical_accuracy=float(medical_accuracy),
            general_accuracy=float(general_accuracy),
            step=int(completed_step),
        )
        if completed_step < 120:
            self.set_ca_window_probability(
                p_medical=decision.p_medical,
                start_step=int(completed_step),
            )
        return decision.as_dict()

    def _score_step_teacher_rows(
        self,
        rows: list[Mapping[str, Any]],
        *,
        step_index: int,
        old_actor: Any,
        old_mask: Any,
    ) -> tuple[Any, Any]:
        if len(self._current_teacher_routes) != len(rows):
            raise P6FormalMethodError("step Teacher routes are absent")
        self._memory_phase_observer(
            "before", "medical_teacher_load", step=step_index + 1
        )
        teacher_base = self.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            revision=self.base_revision,
            torch_dtype=self.torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda:1")
        self.teacher_model = self.PeftModel.from_pretrained(
            teacher_base,
            self.config["teacher"]["adapter_path"],
            adapter_name="medical",
            is_trainable=False,
        )
        del teacher_base
        self.teacher_model.eval()
        self._memory_phase_observer("after", "medical_teacher_load")
        values = []
        with self.torch.inference_mode():
            for row, route in zip(rows, self._current_teacher_routes, strict=True):
                context = (
                    self.teacher_model.disable_adapter()
                    if route == "base"
                    else nullcontext()
                )
                with context:
                    values.append(
                        self._action_logprobs(
                            self.teacher_model, row, device="cuda:1"
                        ).reshape(-1)
                    )
        teacher, teacher_mask = self._pad(values, device="cuda:0")
        if not self.torch.equal(old_mask, teacher_mask):
            raise P6FormalMethodError("method Teacher/Student token mask differs")

        scales_by_route = {"medical": 1.0, "base": 1.0}
        raw_reverse_kl: dict[str, float] = {}
        if self.kl_safety is not None:
            for route in ("medical", "base"):
                indexes = [
                    index
                    for index, selected in enumerate(self._current_teacher_routes)
                    if selected == route
                ]
                if not indexes:
                    continue
                selected = self.torch.stack(
                    [
                        (old_actor[index] - teacher[index])[teacher_mask[index]]
                        .float()
                        .mean()
                        for index in indexes
                    ]
                ).mean()
                raw_reverse_kl[route] = float(selected.detach().cpu())
                scales_by_route[route] = self.kl_safety.update(
                    route, raw_reverse_kl[route]
                )
            scale = self.torch.ones_like(old_actor, dtype=self.torch.float32)
            for index, route in enumerate(self._current_teacher_routes):
                scale[index, teacher_mask[index]] = scales_by_route[route]
            self._current_advantage_scale = scale.detach()
        else:
            self._current_advantage_scale = None
        route_counts = {
            route: self._current_teacher_routes.count(route)
            for route in ("medical", "base")
        }
        self._current_method_evidence = {
            "schema_version": 3,
            "method_id": self.method_id,
            "step_index": step_index,
            "teacher_routes": list(self._current_teacher_routes),
            "teacher_route_counts": route_counts,
            "teacher_adapter_ordered_sha256": self.config["teacher"]["adapter_sha256"],
            "teacher_adapter_weight_sha256": self.config["teacher"]["adapter_weight_sha256"],
            "teacher_manifest_sha256": self.config["teacher"]["manifest_sha256"],
            "base_revision": self.base_revision,
            "same_token_mask_verified": True,
            "raw_reverse_kl_by_teacher_route": raw_reverse_kl,
            "kl_safety_scale_by_teacher_route": scales_by_route,
            "kl_safety_state": (
                None if self.kl_safety is None else self.kl_safety.state_dict()
            ),
            "ca_router_state": (
                None if self.ca_router is None else self.ca_router.state_dict()
            ),
        }
        return teacher, teacher_mask

    def run_formal_method_step_v3(
        self,
        *,
        step_index: int,
        prompt_rows: list[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        source_roles = tuple(str(row["target_role"]) for row in prompt_rows)
        self._current_teacher_routes = self.route_state.routes_for_step(
            step_index, source_roles
        )
        record = dict(
            super().run_formal_step_v2(
                step_index=step_index,
                prompt_rows=prompt_rows,
                max_new_tokens=max_new_tokens,
            )
        )
        evidence = {
            **copy.deepcopy(self._current_method_evidence),
            "optimizer_step": int(record["optimizer_step"]),
            "source_roles": list(source_roles),
            "source_teacher_counts_cumulative": copy.deepcopy(
                self.route_state.source_teacher_counts
            ),
            "teacher_counts_cumulative": dict(self.route_state.teacher_counts),
            "route_state": self.route_state.state_dict(),
            "final_access_count": 0,
        }
        record["formal_method_v3"] = evidence
        _atomic_json(
            self.output
            / "method_steps_v3"
            / f"step_{int(record['optimizer_step']):03d}.json",
            evidence,
        )
        return record

    def formal_route_state(self) -> dict[str, Any]:
        total = sum(self.route_state.teacher_counts.values())
        return {
            "schema_version": 3,
            "method": self.method_id,
            "teacher_route": self.method_config["teacher_route"],
            "adaptive_routing": self.method_id == "CA-OPD",
            "medical_teacher_fraction": (
                None
                if total == 0
                else self.route_state.teacher_counts["medical"] / total
            ),
            "base_teacher_fraction": (
                None if total == 0 else self.route_state.teacher_counts["base"] / total
            ),
            "route_state": self.route_state.state_dict(),
            "kl_safety_state": (
                None if self.kl_safety is None else self.kl_safety.state_dict()
            ),
            "final_access_count": 0,
        }

    def restore_formal_route_state(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != 3 or value.get("method") != self.method_id:
            raise P6FormalMethodError("formal method checkpoint route schema differs")
        self.route_state.load_state_dict(value["route_state"])
        if self.kl_safety is not None:
            if not isinstance(value.get("kl_safety_state"), Mapping):
                raise P6FormalMethodError("CA checkpoint KL state is absent")
            self.kl_safety.load_state_dict(value["kl_safety_state"])
            if self.ca_router is None or not isinstance(
                value.get("ca_router_state"), Mapping
            ):
                raise P6FormalMethodError("CA checkpoint Controller router is absent")
            self.ca_router.load_state_dict(value["ca_router_state"])
        elif value.get("kl_safety_state") is not None:
            raise P6FormalMethodError("IDT checkpoint contains CA KL state")
        elif value.get("ca_router_state") is not None:
            raise P6FormalMethodError("IDT checkpoint contains CA Router state")


__all__ = ["FormalMethodSessionV3"]
