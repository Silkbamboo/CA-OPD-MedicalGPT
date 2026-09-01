from __future__ import annotations

import copy

import pytest

from src.opd import rollout_probability as probability


def _provenance(*, backend: str = "transformers") -> dict:
    generation = (
        probability.formal_transformers_generation_config(max_new_tokens=128)
        if backend == "transformers"
        else probability.formal_vllm_sampling_config(max_tokens=128)
    )
    return {
        "artifact_protocol_version": probability.THREE_POLICY_TRAJECTORY_PROTOCOL_VERSION,
        "trajectory_run_id": "qwen3-4b-p4-3-fresh-full-support-seed42",
        "trajectory_kind": "fresh_full_support",
        "backend": backend,
        "backend_version": "4.56.2" if backend == "transformers" else "0.11.0",
        "model_version": "qwen3-4b-sft-v3-frozen-sha",
        "adapter_version": "base-plus-frozen-lora-sha",
        "generation_config": generation,
        "processor_warper_provenance": {
            "all_support_changing_processors_disabled": True,
            "active_logits_processor_warper_classes": [],
            "selected_token_score_stage": "processed_pre_softmax",
            "identity_source": (
                "effective_generation_config_plus_local_transformers_4.56.2_source"
                if backend == "transformers"
                else "effective_sampling_params_plus_local_vllm_0.11.0_source"
            ),
        },
        "score_source": "generate.scores" if backend == "transformers" else "rollout_log_probs",
        "score_semantics": "normalized_behavior_logprob",
        "behavior_selected_token_logprob_saved": True,
        "raw_selected_token_logprob_saved": True,
        "token_identity_sha256": "a" * 64,
        "eos_and_truncation_saved": True,
        "seed": 42,
        "generator": "explicit_torch_generator" if backend == "transformers" else "vllm_seed",
        "sampler_adapter_version": 0,
        "sampler_adapter_sha256": "b" * 64,
        "final_access": False,
        "controller_access": False,
        "confirmation_access": False,
        "label_access": False,
    }


def test_formal_transformers_config_is_full_support_stochastic() -> None:
    config = probability.formal_transformers_generation_config(max_new_tokens=128)
    audit = probability.validate_full_support_sampling("transformers", config)
    assert audit.full_support_stochastic is True
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_k"] == 0
    assert config["top_p"] == 1.0
    assert config["output_scores"] is True
    assert config["output_logits"] is True


def test_deterministic_greedy_is_not_full_support_stochastic() -> None:
    config = probability.formal_transformers_generation_config(max_new_tokens=128)
    config["do_sample"] = False
    with pytest.raises(probability.RolloutProbabilityError, match="stochastic"):
        probability.validate_full_support_sampling("transformers", config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 20),
        ("top_p", 0.95),
        ("min_p", 0.05),
        ("typical_p", 0.9),
        ("epsilon_cutoff", 0.01),
        ("eta_cutoff", 0.01),
        ("min_new_tokens", 1),
        ("bad_words_ids", [[1]]),
        ("suppress_tokens", [2]),
    ],
)
def test_support_changing_transformers_config_is_rejected(field: str, value: object) -> None:
    config = probability.formal_transformers_generation_config(max_new_tokens=128)
    config[field] = value
    with pytest.raises(probability.RolloutProbabilityError, match="full.support"):
        probability.validate_full_support_sampling("transformers", config)


def test_backend_top_k_disable_semantics_are_explicit() -> None:
    assert probability.backend_disables_top_k("transformers", 0)
    assert not probability.backend_disables_top_k("transformers", -1)
    assert probability.backend_disables_top_k("vllm", 0)
    assert probability.backend_disables_top_k("vllm", -1)
    assert not probability.backend_disables_top_k("vllm", 20)


def test_formal_vllm_adapter_uses_its_own_disable_value() -> None:
    config = probability.formal_vllm_sampling_config(max_tokens=128)
    assert config["top_k"] == 0
    assert probability.validate_full_support_sampling("vllm", config).full_support_stochastic


def test_vllm_min_tokens_support_restriction_is_rejected() -> None:
    config = probability.formal_vllm_sampling_config(max_tokens=128)
    config["min_tokens"] = 1
    with pytest.raises(probability.RolloutProbabilityError, match="full.support"):
        probability.validate_full_support_sampling("vllm", config)


def test_complete_behavior_provenance_passes() -> None:
    probability.validate_rollout_behavior_provenance(
        _provenance(),
        expected_sampler_adapter_sha256="b" * 64,
        expected_trajectory_run_id="qwen3-4b-p4-3-fresh-full-support-seed42",
    )


@pytest.mark.parametrize(
    "field",
    [
        "backend",
        "backend_version",
        "generation_config",
        "processor_warper_provenance",
        "score_source",
        "behavior_selected_token_logprob_saved",
        "token_identity_sha256",
        "seed",
        "generator",
        "sampler_adapter_version",
        "sampler_adapter_sha256",
    ],
)
def test_missing_behavior_provenance_field_fails_closed(field: str) -> None:
    payload = _provenance()
    payload.pop(field)
    with pytest.raises(probability.RolloutProbabilityError, match="missing"):
        probability.validate_rollout_behavior_provenance(payload)


def test_processed_score_cannot_masquerade_as_raw_actor() -> None:
    payload = _provenance()
    payload["score_semantics"] = "raw_actor_logprob"
    with pytest.raises(probability.RolloutProbabilityError, match="processed.*raw"):
        probability.validate_rollout_behavior_provenance(payload)


def test_old_forensic_trajectory_cannot_mix_with_formal_protocol() -> None:
    payload = _provenance()
    payload["trajectory_kind"] = "p4_1_forensic_replay"
    with pytest.raises(probability.RolloutProbabilityError, match="fresh"):
        probability.validate_rollout_behavior_provenance(payload)


def test_wrong_run_id_and_stale_sampler_are_rejected() -> None:
    payload = _provenance()
    with pytest.raises(probability.RolloutProbabilityError, match="run id"):
        probability.validate_rollout_behavior_provenance(
            payload, expected_trajectory_run_id="different-run"
        )
    with pytest.raises(probability.RolloutProbabilityError, match="stale sampler"):
        probability.validate_rollout_behavior_provenance(
            payload, expected_sampler_adapter_sha256="c" * 64
        )


@pytest.mark.parametrize(
    "field", ["final_access", "controller_access", "confirmation_access", "label_access"]
)
def test_forbidden_evaluation_path_access_is_rejected(field: str) -> None:
    payload = copy.deepcopy(_provenance())
    payload[field] = True
    with pytest.raises(probability.RolloutProbabilityError, match="forbidden"):
        probability.validate_rollout_behavior_provenance(payload)


def test_behavior_selected_token_logprob_must_have_finite_support() -> None:
    payload = _provenance()
    payload["behavior_selected_token_logprob_saved"] = False
    with pytest.raises(probability.RolloutProbabilityError, match="support"):
        probability.validate_rollout_behavior_provenance(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_version", "latest"),
        ("model_version", ""),
        ("token_identity_sha256", "short"),
        ("sampler_adapter_sha256", "short"),
        ("sampler_adapter_version", -1),
        ("generator", ""),
    ],
)
def test_behavior_identity_fields_are_validated(field: str, value: object) -> None:
    payload = _provenance()
    payload[field] = value
    with pytest.raises(probability.RolloutProbabilityError):
        probability.validate_rollout_behavior_provenance(payload)


def test_active_processor_warper_identity_is_required() -> None:
    payload = _provenance()
    payload["processor_warper_provenance"].pop("active_logits_processor_warper_classes")
    with pytest.raises(probability.RolloutProbabilityError, match="processor/warper identity"):
        probability.validate_rollout_behavior_provenance(payload)


def test_backend_specific_processor_identity_cannot_be_cross_labeled() -> None:
    payload = _provenance(backend="vllm")
    payload["processor_warper_provenance"]["identity_source"] = (
        "effective_generation_config_plus_local_transformers_4.56.2_source"
    )
    with pytest.raises(probability.RolloutProbabilityError, match="processor/warper identity"):
        probability.validate_rollout_behavior_provenance(payload)


def test_p4_1_effective_top_k_is_forensic_not_formal() -> None:
    config = probability.formal_transformers_generation_config(max_new_tokens=128)
    config["top_k"] = 20
    result = probability.classify_sampling_support("transformers", config)
    assert result == "hard_truncated_support"
