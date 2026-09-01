from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.eval.controller_v2 import build_choice_request
from src.eval.direct_logit_scorer import (
    DIRECT_LOGIT_BACKEND,
    DirectLogitScorerError,
    apply_deterministic_runtime,
    direct_logit_model_plan,
    score_direct_request,
    score_last_prompt_position,
    validate_direct_logit_repetitions,
)


class FakeVector:
    def __init__(self, values, *, dtype="float16"):
        self.values = [float(value) for value in values]
        self.dtype = dtype
        self.float_called = False

    def float(self):
        result = FakeVector(self.values, dtype="float32")
        result.float_called = True
        return result

    def __getitem__(self, index):
        return FakeScalar(self.values[index])


class FakeScalar:
    def __init__(self, value):
        self.value = float(value)

    def item(self):
        return self.value


class FakeLogits:
    ndim = 3

    def __init__(self, positions):
        self.positions = positions
        self.shape = (1, len(positions), len(positions[-1]))
        self.last_vector = None

    def __getitem__(self, item):
        assert item == (0, -1, slice(None))
        self.last_vector = FakeVector(self.positions[-1])
        return self.last_vector


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTorch:
    long = "long"

    def __init__(self):
        self.tensor_calls = []
        self.log_softmax_input_dtype = None
        self.deterministic = None
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
        )
        self.cuda = SimpleNamespace(manual_seed_all=lambda seed: setattr(self, "cuda_seed", seed))

    def log_softmax(self, vector, dim):
        assert dim == -1
        self.log_softmax_input_dtype = vector.dtype
        maximum = max(vector.values)
        denominator = maximum + math.log(sum(math.exp(value - maximum) for value in vector.values))
        return FakeVector([value - denominator for value in vector.values], dtype="float32")

    def tensor(self, value, *, dtype):
        self.tensor_calls.append((value, dtype))
        return FakeTensor(value)

    def ones_like(self, value):
        return FakeTensor([[1 for _ in value.values[0]]])

    def inference_mode(self):
        return nullcontext()

    def manual_seed(self, seed):
        self.cpu_seed = seed

    def use_deterministic_algorithms(self, enabled):
        self.deterministic = enabled


class FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.eval_called = False
        self.calls = []
        self.device = "fixture-device"

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(logits=self.logits)


def _request(option_count=4):
    row = {
        "sample_id": "fixture-1",
        "target_role": "medical_controller_dev",
        "domain": "medical",
        "subject": "fixture",
        "question": "fixture question",
        "options": [f"option-{index}" for index in range(option_count)],
    }
    return build_choice_request(row, tokenize=lambda text: [ord(character) for character in text])


def test_single_token_direct_score_uses_last_prompt_position_and_fp32_log_softmax():
    torch = FakeTorch()
    logits = FakeLogits([[50.0, 0.0, 0.0, 0.0], [0.0, 1.0, 4.0, 2.0]])
    prediction = score_last_prompt_position(
        sample_id="fixture-1",
        logits=logits,
        candidate_token_ids={"A": 0, "B": 1, "C": 2, "D": 3},
        torch_module=torch,
    )
    assert prediction.predicted_label == "C"
    assert torch.log_softmax_input_dtype == "float32"
    assert logits.last_vector is not None
    assert set(prediction.candidate_scores) == set("ABCD")
    assert all(math.isfinite(value) for value in prediction.candidate_scores.values())


def test_direct_score_filters_to_legal_abcd_or_abcde_candidates():
    torch = FakeTorch()
    logits = FakeLogits([[float(index) for index in range(40)]])
    four = score_last_prompt_position(
        sample_id="four",
        logits=logits,
        candidate_token_ids={letter: token for letter, token in zip("ABCD", range(32, 36))},
        torch_module=torch,
    )
    five = score_last_prompt_position(
        sample_id="five",
        logits=logits,
        candidate_token_ids={letter: token for letter, token in zip("ABCDE", range(32, 37))},
        torch_module=torch,
    )
    assert four.predicted_label == "D"
    assert five.predicted_label == "E"
    with pytest.raises(DirectLogitScorerError, match="complete 4/5"):
        score_last_prompt_position(
            sample_id="bad",
            logits=logits,
            candidate_token_ids={"A": 32, "C": 34, "D": 35, "E": 36},
            torch_module=torch,
        )


def test_direct_model_forward_is_single_batch_label_free_and_use_cache_false():
    torch = FakeTorch()
    request = _request(4)
    model = FakeModel(FakeLogits([[0.0] * 80 for _ in request.prompt_token_ids]))
    model.logits.positions[-1][ord("B")] = 4.0
    result = score_direct_request(model, request, torch_module=torch)
    assert result["predicted_label"] == "B"
    assert result["choice_backend"] == DIRECT_LOGIT_BACKEND
    assert result["labels_opened_during_execution"] is False
    assert "answer" not in result and "answer_idx" not in result and "label" not in result
    assert model.eval_called is True
    assert len(model.calls) == 1
    assert model.calls[0]["use_cache"] is False
    assert torch.tensor_calls[0][0] == [list(request.prompt_token_ids)]


def test_direct_formal_path_fails_closed_for_multi_token_candidate():
    torch = FakeTorch()
    request = _request(4)
    altered = SimpleNamespace(
        **{**request.__dict__, "candidates": (
            SimpleNamespace(label="A", token_ids=(1, 2)),
            *request.candidates[1:],
        )}
    )
    with pytest.raises(DirectLogitScorerError, match="single-token"):
        score_direct_request(FakeModel(FakeLogits([[0.0] * 128])), altered, torch_module=torch)


def test_base_and_peft_routes_never_merge_lora_and_freeze_loader_contract():
    base = direct_logit_model_plan("B0")
    medical = direct_logit_model_plan("B1")
    for plan in (base, medical):
        assert plan["backend"] == DIRECT_LOGIT_BACKEND
        assert plan["batch_size"] == 1
        assert plan["dtype"] == "bfloat16"
        assert plan["attn_implementation"] == "eager"
        assert plan["use_cache"] is False
        assert plan["torch_compile"] is False
        assert plan["merge_lora"] is False
    assert base["adapter_route"] == "none"
    assert medical["adapter_route"] == "peft_medical_lora"


def test_deterministic_runtime_sets_all_frozen_controls(monkeypatch):
    torch = FakeTorch()
    numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: setattr(numpy, "seed", seed)))
    apply_deterministic_runtime(torch_module=torch, numpy_module=numpy)
    assert torch.cpu_seed == torch.cuda_seed == numpy.seed == 42
    assert torch.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert monkeypatch is not None  # fixture makes environment restoration explicit


def _smoke_row(sample_id, scores, *, prompt_ids=(1, 2), tokens=(32, 33, 34, 35)):
    labels = "ABCD"
    return {
        "sample_id": sample_id,
        "predicted_label": max(labels, key=lambda label: scores[label]),
        "candidate_scores": scores,
        "candidate_tokenization": [
            {"label": label, "token_ids": [token]}
            for label, token in zip(labels, tokens)
        ],
        "prompt_token_ids": list(prompt_ids),
        "prompt_sha256": "a" * 64,
        "labels_opened_during_execution": False,
    }


def _smoke_rows(scores):
    return [
        _smoke_row(f"s{index}", scores, prompt_ids=(index, index + 10))
        for index in range(1, 5)
    ]


def test_three_repeat_micro_smoke_is_strict_and_route_symmetric():
    base = _smoke_rows({"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0})
    medical = _smoke_rows({"A": -0.3, "B": -1.0, "C": -2.0, "D": -3.0})
    result = validate_direct_logit_repetitions({
        "B0": [base, list(base), list(base)],
        "B1": [medical, list(medical), list(medical)],
    })
    assert result["status"] == "PASS"
    assert result["repeat_count"] == 3
    assert result["score_repeat_tolerance"] == 1e-4
    assert result["labels_opened_during_execution"] is False


def test_micro_smoke_fails_for_score_or_candidate_order_drift():
    first = _smoke_rows({"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0})
    drift = list(first)
    drift[0] = _smoke_row("s1", {"A": -1.0, "B": -0.1002, "C": -2.0, "D": -3.0}, prompt_ids=(1, 11))
    stable = {"B0": [first, first, first], "B1": [first, first, first]}
    with pytest.raises(DirectLogitScorerError, match="exceeds"):
        validate_direct_logit_repetitions({**stable, "B1": [first, drift, first]})
    reordered = list(first)
    reordered[0] = _smoke_row("s1", {"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0}, prompt_ids=(1, 11), tokens=(33, 32, 34, 35))
    with pytest.raises(DirectLogitScorerError, match="tokenization"):
        validate_direct_logit_repetitions({**stable, "B1": [first, reordered, first]})


def test_micro_smoke_rejects_label_access_duplicate_ids_and_route_sample_mismatch():
    rows = _smoke_rows({"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0})
    row = rows[0]
    leaked = {**row, "labels_opened_during_execution": True}
    with pytest.raises(DirectLogitScorerError, match="label"):
        validate_direct_logit_repetitions({"B0": [[leaked, *rows[1:]]] * 3, "B1": [rows] * 3})
    with pytest.raises(DirectLogitScorerError, match="duplicate"):
        validate_direct_logit_repetitions({"B0": [[*rows, row]] * 3, "B1": [[*rows, row]] * 3})
    other = _smoke_row("other", {"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0})
    with pytest.raises(DirectLogitScorerError, match="sample sets"):
        validate_direct_logit_repetitions({"B0": [rows] * 3, "B1": [[other, *rows[1:]]] * 3})


def test_micro_smoke_rejects_cross_route_prompt_or_candidate_identity_drift():
    rows = _smoke_rows({"A": -1.0, "B": -0.1, "C": -2.0, "D": -3.0})
    prompt_drift = _smoke_row(
        "s1", {"A": -0.5, "B": -1.0, "C": -2.0, "D": -3.0}, prompt_ids=(8, 9)
    )
    with pytest.raises(DirectLogitScorerError, match="cross-route prompt/candidate"):
        validate_direct_logit_repetitions({"B0": [rows] * 3, "B1": [[prompt_drift, *rows[1:]]] * 3})
