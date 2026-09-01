"""Formal P4.3 rollout probability and provenance contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


THREE_POLICY_TRAJECTORY_PROTOCOL_VERSION = "p4.3-full-support-trajectory-v1"


class RolloutProbabilityError(RuntimeError):
    """Raised when behavior-policy evidence is ambiguous or unsupported."""


@dataclass(frozen=True)
class SamplingSupportAudit:
    backend: str
    full_support_stochastic: bool
    support_classification: str


def formal_transformers_generation_config(*, max_new_tokens: int) -> dict[str, Any]:
    """Return the frozen Transformers 4.56.2 full-support sampling subset."""

    if max_new_tokens <= 0:
        raise RolloutProbabilityError("max_new_tokens must be positive")
    return {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "min_length": 0,
        "bad_words_ids": None,
        "force_words_ids": None,
        "stop_strings": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "renormalize_logits": False,
        "num_beams": 1,
        "min_new_tokens": 0,
        "max_new_tokens": int(max_new_tokens),
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
        "use_cache": True,
    }


def formal_vllm_sampling_config(*, max_tokens: int) -> dict[str, Any]:
    """Return the frozen vLLM 0.11.0 full-support sampling subset."""

    if max_tokens <= 0:
        raise RolloutProbabilityError("max_tokens must be positive")
    return {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "bad_words": None,
        "allowed_token_ids": None,
        "logit_bias": None,
        "min_tokens": 0,
        "max_tokens": int(max_tokens),
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "logprobs": 1,
        "seed": None,
    }


def backend_disables_top_k(backend: str, value: Any) -> bool:
    """Encode pinned backend semantics instead of copying values across adapters."""

    if backend == "transformers":
        return value == 0
    if backend == "vllm":
        return value in (0, -1)
    raise RolloutProbabilityError(f"unsupported rollout backend: {backend}")


def _is_disabled(value: Any, *, zero_allowed: bool = True) -> bool:
    return value is None or value == [] or (zero_allowed and value == 0)


def classify_sampling_support(backend: str, config: Mapping[str, Any]) -> str:
    """Classify token support from the frozen-version sampling configuration."""

    if backend == "transformers":
        if not bool(config.get("do_sample")):
            return "deterministic_not_stochastic"
        hard_truncation = (
            not backend_disables_top_k(backend, config.get("top_k"))
            or float(config.get("top_p", 0.0)) < 1.0
            or config.get("min_p") not in (None, 0, 0.0)
            or float(config.get("typical_p", 0.0)) < 1.0
            or float(config.get("epsilon_cutoff", 0.0)) > 0.0
            or float(config.get("eta_cutoff", 0.0)) > 0.0
            or not _is_disabled(config.get("bad_words_ids"), zero_allowed=False)
            or not _is_disabled(config.get("force_words_ids"), zero_allowed=False)
            or not _is_disabled(config.get("suppress_tokens"), zero_allowed=False)
            or not _is_disabled(config.get("begin_suppress_tokens"), zero_allowed=False)
            or config.get("forced_bos_token_id") is not None
            or config.get("forced_eos_token_id") is not None
            or int(config.get("no_repeat_ngram_size", 0)) != 0
            or int(config.get("min_length", 0) or 0) != 0
            or int(config.get("min_new_tokens", 0) or 0) != 0
        )
        return "hard_truncated_support" if hard_truncation else "full_support_stochastic"
    if backend == "vllm":
        hard_truncation = (
            not backend_disables_top_k(backend, config.get("top_k"))
            or float(config.get("top_p", 0.0)) < 1.0
            or float(config.get("min_p", 0.0)) > 0.0
            or not _is_disabled(config.get("bad_words"), zero_allowed=False)
            or config.get("allowed_token_ids") not in (None, [])
            or config.get("logit_bias") not in (None, {})
            or int(config.get("min_tokens", 0) or 0) != 0
        )
        return "hard_truncated_support" if hard_truncation else "full_support_stochastic"
    raise RolloutProbabilityError(f"unsupported rollout backend: {backend}")


def validate_full_support_sampling(
    backend: str, config: Mapping[str, Any]
) -> SamplingSupportAudit:
    """Fail closed unless sampling is stochastic, full-support, and protocol-exact."""

    support = classify_sampling_support(backend, config)
    if support == "deterministic_not_stochastic":
        raise RolloutProbabilityError("formal rollout must be stochastic")
    if support != "full_support_stochastic":
        raise RolloutProbabilityError("formal rollout must retain full support")

    if backend == "transformers":
        exact = {
            "temperature": 1.0,
            "top_p": 1.0,
            "typical_p": 1.0,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "min_length": 0,
            "min_new_tokens": 0,
            "renormalize_logits": False,
            "num_beams": 1,
            "output_scores": True,
            "output_logits": True,
            "return_dict_in_generate": True,
        }
    else:
        exact = {
            "temperature": 1.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "use_beam_search": False,
            "n": 1,
            "best_of": 1,
            "logprobs": 1,
            "min_tokens": 0,
        }
    mismatches = [key for key, expected in exact.items() if config.get(key) != expected]
    if mismatches:
        raise RolloutProbabilityError(
            "formal full-support config differs from frozen protocol: "
            + ", ".join(sorted(mismatches))
        )
    return SamplingSupportAudit(
        backend=backend,
        full_support_stochastic=True,
        support_classification=support,
    )


_REQUIRED_PROVENANCE_FIELDS = {
    "artifact_protocol_version",
    "trajectory_run_id",
    "trajectory_kind",
    "backend",
    "backend_version",
    "model_version",
    "adapter_version",
    "generation_config",
    "processor_warper_provenance",
    "score_source",
    "score_semantics",
    "behavior_selected_token_logprob_saved",
    "raw_selected_token_logprob_saved",
    "token_identity_sha256",
    "eos_and_truncation_saved",
    "seed",
    "generator",
    "sampler_adapter_version",
    "sampler_adapter_sha256",
    "final_access",
    "controller_access",
    "confirmation_access",
    "label_access",
}


def validate_rollout_behavior_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_sampler_adapter_sha256: str | None = None,
    expected_trajectory_run_id: str | None = None,
) -> None:
    """Validate that selected-token q is complete, finite-support behavior evidence."""

    missing = sorted(_REQUIRED_PROVENANCE_FIELDS - set(provenance))
    if missing:
        raise RolloutProbabilityError("missing behavior provenance fields: " + ", ".join(missing))
    if provenance["artifact_protocol_version"] != THREE_POLICY_TRAJECTORY_PROTOCOL_VERSION:
        raise RolloutProbabilityError("trajectory artifact protocol version mismatch")
    if provenance["trajectory_kind"] != "fresh_full_support":
        raise RolloutProbabilityError("formal correction requires a fresh full-support trajectory")
    for field in (
        "trajectory_run_id",
        "backend",
        "backend_version",
        "model_version",
        "adapter_version",
        "score_source",
        "generator",
    ):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            raise RolloutProbabilityError(f"behavior provenance identity is empty: {field}")
    version_contract = {"transformers": "4.56.2", "vllm": "0.11.0"}
    if provenance["backend"] not in version_contract or provenance["backend_version"] != version_contract[provenance["backend"]]:
        raise RolloutProbabilityError("rollout backend/version is not pinned")
    for field in ("token_identity_sha256", "sampler_adapter_sha256"):
        value = provenance[field]
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower())
        ):
            raise RolloutProbabilityError(f"behavior provenance SHA is invalid: {field}")
    if not isinstance(provenance["seed"], int) or isinstance(provenance["seed"], bool):
        raise RolloutProbabilityError("behavior provenance seed must be an integer")
    if (
        not isinstance(provenance["sampler_adapter_version"], int)
        or isinstance(provenance["sampler_adapter_version"], bool)
        or provenance["sampler_adapter_version"] < 0
    ):
        raise RolloutProbabilityError("sampler adapter version must be a non-negative integer")
    if expected_trajectory_run_id is not None and provenance["trajectory_run_id"] != expected_trajectory_run_id:
        raise RolloutProbabilityError("trajectory run id mismatch")
    if (
        expected_sampler_adapter_sha256 is not None
        and provenance["sampler_adapter_sha256"] != expected_sampler_adapter_sha256
    ):
        raise RolloutProbabilityError("stale sampler adapter SHA")
    for field in ("final_access", "controller_access", "confirmation_access", "label_access"):
        if provenance[field] is not False:
            raise RolloutProbabilityError(f"forbidden evaluation path access: {field}")
    if provenance["score_source"] == "generate.scores" and provenance["score_semantics"] == "raw_actor_logprob":
        raise RolloutProbabilityError("processed generation score cannot be labeled raw actor logprob")
    if provenance["score_semantics"] != "normalized_behavior_logprob":
        raise RolloutProbabilityError("rollout score must be normalized behavior logprob")
    if provenance["behavior_selected_token_logprob_saved"] is not True:
        raise RolloutProbabilityError("selected token lacks finite behavior support evidence")
    if provenance["raw_selected_token_logprob_saved"] is not True:
        raise RolloutProbabilityError("raw selected-token actor diagnostic is required")
    if provenance["eos_and_truncation_saved"] is not True:
        raise RolloutProbabilityError("EOS and truncation provenance is required")
    processor = provenance["processor_warper_provenance"]
    if not isinstance(processor, Mapping) or processor.get(
        "all_support_changing_processors_disabled"
    ) is not True:
        raise RolloutProbabilityError("processor/warper provenance does not prove full support")
    identity_sources = {
        "transformers": "effective_generation_config_plus_local_transformers_4.56.2_source",
        "vllm": "effective_sampling_params_plus_local_vllm_0.11.0_source",
    }
    if (
        not isinstance(processor.get("active_logits_processor_warper_classes"), list)
        or processor["active_logits_processor_warper_classes"]
        or processor.get("identity_source") != identity_sources[provenance["backend"]]
    ):
        raise RolloutProbabilityError("active processor/warper identity is incomplete")
    validate_full_support_sampling(
        str(provenance["backend"]), provenance["generation_config"]
    )
