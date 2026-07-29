"""Evaluator wiring: controller-dev and final-test are different *types*.

The isolation requirement ("controller dev 驱动调度和 checkpoint 选择；final test
只在配置与 checkpoint 固定后执行一次") is enforced with the type system plus two
runtime guards:

* :class:`ControllerDevEvaluator` answers ``allows_control_decisions() -> True``;
  the router only accepts such an object.
* :class:`FinalTestEvaluator` answers ``False``, requires an explicit
  ``allow_final_test=True`` plus a ``reason`` at construction, logs the access,
  and refuses a second evaluation unless ``allow_repeat=True`` is passed
  deliberately.

Passing a final-test evaluator to the router therefore fails at construction
time, not after a run has already been contaminated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.data.access import load_split
from src.data.chat import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPLATE, ChatTemplate
from src.data.schema import CONTROLLER_DEV, FINAL_TEST, Sample, may_drive_control
from src.eval.mcq import DecodeSettings, GenerateFn, MCQResult, evaluate_mcq
from src.utils.config import FieldSpec, load_config
from src.utils.io import write_json

EVAL_SCHEMA: Dict[str, object] = {
    "split": FieldSpec((str,), choices=[CONTROLLER_DEV, FINAL_TEST]),
    "data_dir": FieldSpec((str,)),
    "max_samples": FieldSpec((int,), required=False, default=None, bounds=(1, None)),
    "batch_size": FieldSpec((int,), required=False, default=8, bounds=(1, None)),
    "decode": {
        "temperature": FieldSpec((float,), required=False, default=0.0),
        "max_new_tokens": FieldSpec((int,), bounds=(1, None)),
        "system_prompt": FieldSpec((str,), required=False, default=DEFAULT_SYSTEM_PROMPT),
    },
}


class EvaluationPolicyError(RuntimeError):
    """Raised when an evaluator is used outside its permitted role."""


@dataclass
class _BaseEvaluator:
    split: str
    data_dir: str
    decode: DecodeSettings
    max_samples: Optional[int] = None
    batch_size: int = 8
    template: ChatTemplate = DEFAULT_TEMPLATE

    def allows_control_decisions(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def _load(self, **kwargs: Any) -> List[Sample]:
        return load_split(self.data_dir, self.split, max_samples=self.max_samples, **kwargs)

    def describe(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "data_dir": str(self.data_dir),
            "decode": self.decode.as_dict(),
            "max_samples": self.max_samples,
            "may_drive_control": self.allows_control_decisions(),
        }


class ControllerDevEvaluator(_BaseEvaluator):
    """The only evaluator allowed to influence training decisions."""

    def __init__(
        self,
        data_dir: str | Path,
        decode: DecodeSettings | None = None,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        template: ChatTemplate = DEFAULT_TEMPLATE,
    ):
        super().__init__(
            split=CONTROLLER_DEV,
            data_dir=str(data_dir),
            decode=decode or DecodeSettings(),
            max_samples=max_samples,
            batch_size=batch_size,
            template=template,
        )
        if not may_drive_control(self.split):  # pragma: no cover - guards a schema edit
            raise EvaluationPolicyError(f"split {self.split!r} is not a control split")

    def allows_control_decisions(self) -> bool:
        return True

    def evaluate(self, generate_fn: GenerateFn) -> MCQResult:
        samples = self._load()
        return evaluate_mcq(
            samples,
            generate_fn,
            split=self.split,
            decode=self.decode,
            template=self.template,
            batch_size=self.batch_size,
        )

    def ability_pair(self, generate_fn: GenerateFn) -> tuple[float, float]:
        """``(medical_accuracy, general_accuracy)`` for the router."""
        result = self.evaluate(generate_fn)
        medical = result.medical_accuracy
        general = result.general_accuracy
        if medical is None or general is None:
            raise EvaluationPolicyError(
                "controller dev must contain both medical and general samples; "
                f"got domains {sorted(result.counts_by_domain)}"
            )
        return medical, general


class FinalTestEvaluator(_BaseEvaluator):
    """Single-use, post-freeze evaluator. Never a control input."""

    def __init__(
        self,
        data_dir: str | Path,
        reason: str,
        allow_final_test: bool = False,
        decode: DecodeSettings | None = None,
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        template: ChatTemplate = DEFAULT_TEMPLATE,
    ):
        if not allow_final_test:
            raise EvaluationPolicyError(
                "FinalTestEvaluator requires allow_final_test=True: final test may only be "
                "run after the configuration and checkpoint are frozen"
            )
        if not reason or not reason.strip():
            raise EvaluationPolicyError("FinalTestEvaluator requires a non-empty reason for the audit log")
        super().__init__(
            split=FINAL_TEST,
            data_dir=str(data_dir),
            decode=decode or DecodeSettings(),
            max_samples=max_samples,
            batch_size=batch_size,
            template=template,
        )
        self.reason = reason.strip()
        self._evaluations = 0

    def allows_control_decisions(self) -> bool:
        return False

    def evaluate(self, generate_fn: GenerateFn, allow_repeat: bool = False) -> MCQResult:
        if self._evaluations and not allow_repeat:
            raise EvaluationPolicyError(
                "final test has already been evaluated in this process; repeated evaluation "
                "invites test-set hill climbing (pass allow_repeat=True only for an explicitly "
                "documented re-run)"
            )
        samples = self._load(allow_final_test=True, reason=self.reason)
        self._evaluations += 1
        return evaluate_mcq(
            samples,
            generate_fn,
            split=self.split,
            decode=self.decode,
            template=self.template,
            batch_size=self.batch_size,
        )

    @property
    def evaluations(self) -> int:
        return self._evaluations


def build_evaluator(
    config_path: str | Path,
    reason: Optional[str] = None,
    allow_final_test: bool = False,
) -> ControllerDevEvaluator | FinalTestEvaluator:
    """Instantiate the evaluator described by a YAML config."""
    cfg = load_config(config_path, EVAL_SCHEMA)
    decode = DecodeSettings(
        temperature=float(cfg["decode"]["temperature"]),  # type: ignore[index]
        max_new_tokens=int(cfg["decode"]["max_new_tokens"]),  # type: ignore[index]
        system_prompt=cfg["decode"]["system_prompt"],  # type: ignore[index]
    )
    common = dict(
        data_dir=str(cfg["data_dir"]),
        decode=decode,
        max_samples=cfg["max_samples"],
        batch_size=int(cfg["batch_size"]),
    )
    if cfg["split"] == CONTROLLER_DEV:
        return ControllerDevEvaluator(**common)  # type: ignore[arg-type]
    return FinalTestEvaluator(reason=reason or "", allow_final_test=allow_final_test, **common)  # type: ignore[arg-type]


def write_eval_artifacts(
    output_dir: str | Path,
    result: MCQResult,
    tag: str,
    include_samples: bool = True,
) -> Dict[str, str]:
    """Persist aggregate + per-sample evaluation output next to a run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(out / f"eval_{tag}_{result.split}.json", result.as_dict(include_samples=False))
    paths = {"summary": str(summary_path)}
    if include_samples:
        from src.utils.io import write_jsonl

        preds = out / f"eval_{tag}_{result.split}_predictions.jsonl"
        write_jsonl(preds, [s.as_dict() for s in result.samples])
        paths["predictions"] = str(preds)
    return paths
