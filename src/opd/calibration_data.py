"""Build frozen, label-free OPD scorer calibration fixtures with bounded memory."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any, Mapping
import argparse

from src.data.chat import format_mcq_question
from src.opd.scorer_calibration import build_public_trajectory_manifest


SUPERVISION_FIELDS = frozenset({
    "answer", "answer_idx", "answer_index", "label", "labels", "solution",
    "reasoning", "response", "completion", "output", "reward", "final",
    "final_answer", "gold", "target", "reference_answer", "ground_truth",
})
_SUPERVISION = SUPERVISION_FIELDS
_REPLAY_LENGTHS = (64, 64, 128, 128, 256, 256, 512, 512, 64, 128, 256, 512)
_HISTORICAL_TARGET_LENGTHS = (64, 128, 256, 512, 128, 512)
_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class CalibrationDataError(RuntimeError):
    pass


def contains_forbidden_supervision(value: Any) -> bool:
    """Reject supervision keys at any nesting depth before prompt rendering."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in SUPERVISION_FIELDS:
                return True
            if contains_forbidden_supervision(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(contains_forbidden_supervision(item) for item in value)
    return False


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise CalibrationDataError(f"calibration output must be new/empty: {path}")


def render_prompt_text(row: Mapping[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise CalibrationDataError("prompt row has an empty question")
    options = row.get("options")
    if options is None:
        return question
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise CalibrationDataError("MCQ prompt options are malformed")
    return format_mcq_question(question, [str(item) for item in options])


def select_prompt_rows(path: str | Path, *, role: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Keep the smallest deterministic hash ranks without buffering the source."""

    if count < 1:
        raise CalibrationDataError("selection count must be positive")
    selected: list[tuple[int, str, dict[str, Any]]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CalibrationDataError(f"{path}:{line_number} is not an object")
            if contains_forbidden_supervision(row):
                raise CalibrationDataError(f"{path}:{line_number} contains supervision")
            observed_role = str(row.get("target_role", ""))
            if "final" in observed_role or "confirmation" in observed_role:
                raise CalibrationDataError(f"forbidden source role: {observed_role}")
            if observed_role != role:
                raise CalibrationDataError(f"unexpected source role: {observed_role}")
            sample_id = str(row.get("sample_id", ""))
            content_hash = str(row.get("content_hash", ""))
            if not sample_id or len(content_hash) != 64:
                raise CalibrationDataError("prompt row lacks stable identity")
            rank = int(hashlib.sha256(f"{seed}\0{sample_id}\0{content_hash}".encode()).hexdigest(), 16)
            item = (-rank, sample_id, row)
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    if len(selected) != count:
        raise CalibrationDataError(f"{role} has only {len(selected)} rows; need {count}")
    return [item[2] for item in sorted(selected, key=lambda item: (-item[0], item[1]))]


def _render_chat(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [int(value) for value in ids]


def _response_ids(tokenizer: Any, *, target_length: int, letter_start: int | None, eos: bool) -> list[int]:
    seed_tokens = [int(value) for value in tokenizer.encode(
        "这是用于同轨迹评分边界校准的固定中文回答。", add_special_tokens=False
    )]
    if not seed_tokens:
        raise CalibrationDataError("tokenizer produced an empty calibration response")
    prefix = [int(letter_start)] if letter_start is not None else []
    reserve = len(prefix) + (1 if eos else 0)
    body_size = target_length - reserve
    if body_size < 1:
        raise CalibrationDataError("calibration response target is too short")
    body = (seed_tokens * ((body_size + len(seed_tokens) - 1) // len(seed_tokens)))[:body_size]
    result = prefix + body
    if eos:
        result.append(int(tokenizer.eos_token_id))
    if len(result) != target_length:
        raise CalibrationDataError("calibration response length construction drift")
    return result


def _select_historical_responses(
    path: str | Path, *, prompt_path: str | Path, count: int
) -> list[dict[str, Any]]:
    """Select label-free historical generations by length, never by correctness."""

    candidates: list[dict[str, Any]] = []
    allowed = {
        "sample_id", "text", "token_ids", "eos_observed", "finish_reason",
        "stop_reason", "prompt_echo", "repetition",
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = sorted(_SUPERVISION & set(row))
            if forbidden or contains_forbidden_supervision(row):
                raise CalibrationDataError(
                    f"historical output {path}:{line_number} contains supervision: {forbidden}"
                )
            unknown = set(row) - allowed
            if unknown:
                raise CalibrationDataError(
                    f"historical output {path}:{line_number} has unknown fields: {sorted(unknown)}"
                )
            token_ids = row.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise CalibrationDataError("historical output lacks response token IDs")
            candidates.append({
                "sample_id": str(row["sample_id"]),
                "token_ids": [int(value) for value in token_ids],
                "eos": bool(row.get("eos_observed")),
                "truncated": str(row.get("finish_reason")) == "length",
            })
    if len(candidates) < count:
        raise CalibrationDataError(f"historical output has only {len(candidates)} rows; need {count}")
    chosen: list[dict[str, Any]] = []
    remaining = list(candidates)
    for target in _HISTORICAL_TARGET_LENGTHS[:count]:
        best = min(
            remaining,
            key=lambda row: (
                abs(len(row["token_ids"]) - target),
                hashlib.sha256(f"42\0{row['sample_id']}".encode()).hexdigest(),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
    needed = {row["sample_id"] for row in chosen}
    prompt_projection: dict[str, dict[str, str]] = {}
    with Path(prompt_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in needed:
                continue
            role = str(row.get("target_role", ""))
            source = str(row.get("source", ""))
            question = str(row.get("question", "")).strip()
            if "final" in role.lower() or role != "audit_holdout" or not question:
                raise CalibrationDataError("historical prompt source is not the frozen audit holdout")
            # Deliberately project only prompt identity/text.  Answer, response,
            # reasoning and every other field are neither copied nor inspected.
            prompt_projection[sample_id] = {
                "question": question, "source": source, "target_role": role,
            }
    if set(prompt_projection) != needed:
        raise CalibrationDataError("historical response/prompt sample IDs are not one-to-one")
    for row in chosen:
        row.update(prompt_projection[row["sample_id"]])
    return chosen


def _atomic_lines(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_stable_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_calibration_sets(
    *,
    o1_path: str | Path,
    cmb_path: str | Path,
    tokenizer: Any,
    private_root: str | Path,
    public_root: str | Path,
    tokenizer_revision: str,
    seed: int,
    historical_output_path: str | Path | None = None,
    historical_prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    if tokenizer_revision != _MODEL_REVISION and tokenizer_revision != "a" * 40:
        raise CalibrationDataError("tokenizer revision differs from the frozen model")
    private = Path(private_root)
    public = Path(public_root)
    _assert_empty(private)
    _assert_empty(public)
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    o1_rows = select_prompt_rows(o1_path, role="medical_opd_o1", count=6, seed=seed)
    cmb_rows = select_prompt_rows(cmb_path, role="medical_opd_cmb", count=6, seed=seed)
    if (historical_output_path is None) != (historical_prompt_path is None):
        raise CalibrationDataError("historical output and prompt artifacts must be supplied together")
    historical = None
    if historical_output_path is not None and historical_prompt_path is not None:
        historical = _select_historical_responses(
            historical_output_path, prompt_path=historical_prompt_path, count=6
        )
    source_rows = [item for pair in zip(cmb_rows, o1_rows, strict=True) for item in pair]

    replay_rows: list[dict[str, Any]] = []
    public_input: list[dict[str, Any]] = []
    for index, (row, target_length) in enumerate(zip(source_rows, _REPLAY_LENGTHS, strict=True)):
        is_cmb = row["target_role"] == "medical_opd_cmb"
        historical_row = historical[index // 2] if historical is not None and not is_cmb else None
        prompt = (
            str(historical_row["question"])
            if historical_row is not None else render_prompt_text(row)
        )
        prompt_ids = _render_chat(tokenizer, prompt)
        if historical_row is not None:
            response_ids = list(historical_row["token_ids"])
            eos = bool(historical_row["eos"])
            truncated = bool(historical_row["truncated"])
            response_style = "historical_b0_open"
        else:
            letter_start = (32 + index // 2) if index in (0, 2) else None
            eos = index % 2 == 0
            response_ids = _response_ids(
                tokenizer, target_length=target_length, letter_start=letter_start, eos=eos
            )
            truncated = not eos
            response_style = "letter_then_text" if letter_start is not None else "open_chinese"
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        fixture_id = f"replay-{index:02d}"
        private_row = {
            "fixture_id": fixture_id,
            "source_role": "p3_7_b0_open_diagnostic" if historical_row is not None else row["target_role"],
            "source_sample_id": historical_row["sample_id"] if historical_row is not None else row["sample_id"],
            "source_content_hash": (
                hashlib.sha256(f"p3.7-b0\0{historical_row['sample_id']}".encode()).hexdigest()
                if historical_row is not None else row["content_hash"]
            ),
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "response": response_text,
            "response_ids": response_ids,
            "response_style": response_style,
            "eos": eos,
            "truncated": truncated,
            "tokenizer_revision": tokenizer_revision,
            "contains_labels": False,
        }
        if historical_row is not None:
            private_row["source_artifact"] = str(historical_output_path)
            private_row["source_prompt_artifact"] = str(historical_prompt_path)
            private_row["historical_output_sample_id"] = historical_row["sample_id"]
        replay_rows.append(private_row)
        public_input.append(private_row)
    private_replay = private / "deterministic_replay.jsonl"
    _atomic_lines(private_replay, replay_rows)
    replay_manifest = build_public_trajectory_manifest(
        public_input, raw_fixture_sha256=_sha_file(private_replay), seed=seed
    )
    _atomic_json(public / "deterministic_replay_manifest.json", replay_manifest)

    live_selected = o1_rows[:2] + cmb_rows[:2]
    live_rows = []
    live_public = []
    for index, row in enumerate(live_selected):
        prompt = render_prompt_text(row)
        prompt_ids = _render_chat(tokenizer, prompt)
        live_rows.append({
            "fixture_id": f"live-{index:02d}",
            "source_role": row["target_role"],
            "source_sample_id": row["sample_id"],
            "source_content_hash": row["content_hash"],
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "contains_labels": False,
        })
        live_public.append({
            "fixture_id": f"live-{index:02d}",
            "source_role": row["target_role"],
            "prompt_sha256": _sha_bytes(prompt.encode()),
            "prompt_length": len(prompt_ids),
            "tokenizer_revision": tokenizer_revision,
        })
    private_live = private / "live_rollout_prompts.jsonl"
    _atomic_lines(private_live, live_rows)
    live_payload = {
        "schema_version": 1,
        "kind": "opd_scorer_calibration_live_prompts",
        "seed": int(seed),
        "count": len(live_public),
        "contains_labels": False,
        "contains_raw_text": False,
        "contains_token_ids": False,
        "raw_fixture_sha256": _sha_file(private_live),
        "fixtures": live_public,
    }
    live_payload["manifest_sha256"] = _sha_bytes(_stable_json(live_payload).encode())
    _atomic_json(public / "live_rollout_prompt_manifest.json", live_payload)
    return {
        "replay_count": len(replay_rows),
        "live_prompt_count": len(live_rows),
        "private_replay_sha256": _sha_file(private_replay),
        "private_live_sha256": _sha_file(private_live),
        "replay_manifest_sha256": _sha_file(public / "deterministic_replay_manifest.json"),
        "live_manifest_sha256": _sha_file(public / "live_rollout_prompt_manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build label-free P4.0 calibration fixtures")
    parser.add_argument("--o1", required=True)
    parser.add_argument("--cmb", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--historical-output")
    parser.add_argument("--historical-prompts")
    args = parser.parse_args(argv)
    # This imports only tokenizer code and requires local files. It never calls
    # AutoModel/PeftModel and is the sole real-tokenizer operation in P4.0.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    report = build_calibration_sets(
        o1_path=args.o1,
        cmb_path=args.cmb,
        tokenizer=tokenizer,
        private_root=args.private_root,
        public_root=args.public_root,
        tokenizer_revision=_MODEL_REVISION,
        seed=42,
        historical_output_path=args.historical_output,
        historical_prompt_path=args.historical_prompts,
    )
    print(_stable_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
