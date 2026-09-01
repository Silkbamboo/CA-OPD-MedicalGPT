from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from src.opd.trajectory_scorer import (
    MEDICAL_ADAPTER_SHA256,
    SharedBackboneRoutes,
    TrajectoryContractError,
    TrajectoryScoreRequest,
    TransformersTrajectoryLogprobScorer,
    VLLM_TRAJECTORY_BACKEND_POLICY,
)


class PositionModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, *, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.route = "medical"
        self.generated = False
        self.forward_calls = []

    @contextmanager
    def disable_adapter(self):
        previous = self.route
        self.route = "base"
        try:
            yield
        finally:
            self.route = previous

    def set_adapter(self, name):
        assert name == "medical"
        self.route = "medical"

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        self.forward_calls.append(
            {"route": self.route, "ids": input_ids.detach().clone(), "use_cache": use_cache}
        )
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -5.0, dtype=self.dtype)
        for batch_index in range(batch):
            for pos in range(seq):
                expected = int(input_ids[batch_index, min(pos + 1, seq - 1)])
                logits[batch_index, pos, expected] = 5.0 if self.route == "base" else 6.0
        return SimpleNamespace(logits=logits)

    def generate(self, *args, **kwargs):  # pragma: no cover - must never be called
        self.generated = True
        raise AssertionError("Teacher must not generate")


def request(**overrides):
    values = dict(
        request_id="r1",
        route="base",
        prompt_ids=(9, 10, 11),
        response_ids=(32, 33, 1),
        attention_mask=(1, 1, 1, 1, 1, 1),
        eos_token_id=1,
        finish_reason="stop",
        truncated=False,
    )
    values.update(overrides)
    if "attention_mask" not in overrides:
        values["attention_mask"] = (1,) * (len(values["prompt_ids"]) + len(values["response_ids"]))
    return TrajectoryScoreRequest(**values)


def scorer(model=None):
    model = model or PositionModel()
    routes = SharedBackboneRoutes(
        model=model,
        medical_adapter_name="medical",
        medical_adapter_sha256=MEDICAL_ADAPTER_SHA256,
    )
    return TransformersTrajectoryLogprobScorer(
        model=model,
        routes=routes,
        model_id="toy",
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        logprob_chunk_tokens=2,
    )


def test_first_response_uses_prompt_last_logit_position():
    result = scorer().score(request())
    assert result.token_ids == (32, 33, 1)
    assert result.prompt_length == 3
    assert result.response_length == 3
    assert result.action_logit_positions == (2, 3, 4)
    assert all(value > -0.01 for value in result.token_logprobs)


def test_last_response_uses_p_plus_t_minus_two_position():
    result = scorer().score(request(prompt_ids=(9,), response_ids=(34, 35)))
    assert result.action_logit_positions == (0, 1)
    assert len(result.token_logprobs) == 2


def test_prompt_and_padding_are_not_response_mask():
    result = scorer().score(request())
    assert result.response_mask == (1, 1, 1)
    assert len(result.response_mask) == result.response_length


def test_actual_eos_is_scored():
    result = scorer().score(request(response_ids=(32, 1), finish_reason="stop", truncated=False))
    assert result.token_ids[-1] == 1
    assert result.eos_position == 1
    assert result.response_mask[-1] == 1


def test_truncated_trajectory_does_not_append_eos():
    result = scorer().score(
        request(response_ids=(32, 33), finish_reason="length", truncated=True)
    )
    assert result.token_ids == (32, 33)
    assert result.eos_position is None


def test_truncated_trajectory_cannot_claim_or_contain_eos():
    with pytest.raises(TrajectoryContractError, match="truncated"):
        request(response_ids=(32, 1), finish_reason="length", truncated=True)
    with pytest.raises(TrajectoryContractError, match="last response"):
        request(response_ids=(32, 1, 33), finish_reason="stop", truncated=False)


def test_token_length_mismatch_fails_closed():
    with pytest.raises(TrajectoryContractError, match="attention_mask"):
        scorer().score(request(attention_mask=(1, 1)))


def test_nonfinite_logprob_fails_closed():
    class NonFinite(PositionModel):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.logits[:, 2, :] = float("nan")
            return output

    with pytest.raises(TrajectoryContractError, match="finite"):
        scorer(NonFinite()).score(request())


def test_label_and_answer_fields_are_rejected():
    with pytest.raises(TrajectoryContractError, match="supervision"):
        TrajectoryScoreRequest.from_mapping(
            {
                "request_id": "x", "route": "base", "prompt_ids": [1],
                "response_ids": [2], "attention_mask": [1, 1], "label": "A",
            }
        )
    with pytest.raises(TrajectoryContractError, match="supervision"):
        TrajectoryScoreRequest.from_mapping(
            {
                "request_id": "x", "route": "base", "prompt_ids": [1],
                "response_ids": [2], "attention_mask": [1, 1], "metadata": {"answer": "A"},
            }
        )


def test_final_role_is_rejected():
    with pytest.raises(TrajectoryContractError, match="final"):
        TrajectoryScoreRequest.from_mapping(
            {
                "request_id": "x", "route": "base", "prompt_ids": [1],
                "response_ids": [2], "attention_mask": [1, 1], "source_role": "medical_final_test",
            }
        )


def test_base_route_disables_adapter_and_medical_route_activates_sha_bound_adapter():
    model = PositionModel()
    engine = scorer(model)
    base = engine.score(request(route="base"))
    medical = engine.score(request(request_id="m", route="medical"))
    assert model.forward_calls[0]["route"] == "base"
    assert model.forward_calls[1]["route"] == "medical"
    assert base.adapter_sha is None
    assert medical.adapter_sha == MEDICAL_ADAPTER_SHA256
    assert base.token_logprobs != medical.token_logprobs


def test_medical_route_requires_frozen_adapter_sha():
    model = PositionModel()
    with pytest.raises(TrajectoryContractError, match="adapter SHA"):
        SharedBackboneRoutes(
            model=model, medical_adapter_name="medical", medical_adapter_sha256="0" * 64
        )


def test_unknown_route_fails_closed():
    with pytest.raises(TrajectoryContractError, match="route"):
        scorer().score(request(route="other"))


def test_teacher_runs_inference_only_without_generation_or_cache():
    model = PositionModel()
    engine = scorer(model)
    result = engine.score(request())
    assert result.backend == "transformers_direct_trajectory_logits"
    assert result.precision == "model_bfloat16_logsoftmax_float32"
    assert result.finite is True
    assert model.generated is False
    assert model.training is False
    assert model.forward_calls[0]["use_cache"] is False
    assert all(not p.requires_grad for p in model.parameters())


def test_vllm_backend_is_diagnostic_only():
    assert VLLM_TRAJECTORY_BACKEND_POLICY == {
        "backend": "vllm_prompt_logprobs",
        "formal_enabled": False,
        "diagnostic_only": True,
    }


def test_same_route_length_bucket_uses_one_small_batch_forward():
    model = PositionModel()
    engine = scorer(model)
    requests = [
        request(request_id="a", response_ids=(32, 33)),
        request(request_id="b", prompt_ids=(9, 10), response_ids=(34, 35, 1)),
    ]
    results = engine.score_batch(requests, maximum_batch_size=2, length_bucket_width=128)
    assert [result.request_id for result in results] == ["a", "b"]
    assert [result.token_ids for result in results] == [(32, 33), (34, 35, 1)]
    assert len(model.forward_calls) == 1
