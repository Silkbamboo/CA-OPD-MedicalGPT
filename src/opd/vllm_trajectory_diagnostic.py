"""One-shot vLLM 0.11 trajectory diagnostic; never a formal OPD backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - authorized GPU only
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--live-trajectories")
    args = parser.parse_args(argv)
    import yaml
    from src.teacher.lora_router_service import (
        TeacherScoreRequest, build_vllm_service,
    )

    config = yaml.safe_load(Path(args.config).read_text())
    if config["scoring"]["vllm"] != {
        "backend": "vllm_prompt_logprobs", "formal_enabled": False, "diagnostic_only": True
    }:
        raise RuntimeError("vLLM diagnostic policy drift")
    rows = [json.loads(line) for line in Path(config["data"]["private_replay_path"]).read_text().splitlines() if line]
    if args.live_trajectories:
        for row in (
            json.loads(line) for line in Path(args.live_trajectories).read_text().splitlines() if line
        ):
            rows.append({
                "fixture_id": f"live:{row['fixture_id']}",
                "prompt_ids": row["prompt_ids"],
                "response_ids": row["response_ids"],
            })
    maximum = max(len(row["prompt_ids"]) + len(row["response_ids"]) for row in rows) + 1
    service = build_vllm_service(
        config["model"]["id"], config["teacher"]["adapter_path"],
        gpu_memory_utilization=0.80, max_lora_rank=16,
        diagnostic_only_ack=True,
        tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
        max_model_len=maximum, seed=42,
    )
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            combined = tuple(row["prompt_ids"] + row["response_ids"])
            prompt_length = len(row["prompt_ids"])
            for route in ("base", "medical"):
                for repeat in range(3):
                    request = TeacherScoreRequest(
                        request_id=f"{row['fixture_id']}-{route}-{repeat}",
                        teacher_id=route,
                        token_ids=(combined,),
                        prompt_lengths=(prompt_length,),
                    )
                    response = service.score(request)
                    all_target_scores = response.token_logprobs[0]
                    action_scores = all_target_scores[prompt_length - 1: prompt_length - 1 + len(row["response_ids"])]
                    value = {
                        "fixture_id": row["fixture_id"], "route": route, "repeat": repeat,
                        "token_ids": row["response_ids"], "token_logprobs": list(action_scores),
                        "response_mask": [1] * len(row["response_ids"]),
                        "diagnostic_only": True, "formal_enabled": False,
                    }
                    handle.write(json.dumps(value, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
