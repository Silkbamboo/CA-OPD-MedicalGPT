"""Run metrics logging with a frozen metric vocabulary.

docs/REPRODUCIBILITY.md §8 and docs/REPRODUCIBILITY.md §8 fix the metric names so that Phase 1/2/3 runs stay
comparable and plotting scripts never have to guess. Writing an unknown metric
name raises, which is deliberate: renaming a metric must be a conscious change
to :data:`METRIC_NAMES` (and therefore visible in review), not a typo that
silently creates a second series.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .io import append_jsonl, ensure_dir

METRIC_NAMES: frozenset[str] = frozenset(
    {
        # training
        "train/loss",
        "train/lr",
        "train/grad_norm",
        # opd
        "opd/reverse_kl_mean",
        "opd/reverse_kl_std",
        "opd/advantage_mean",
        "opd/advantage_std",
        "opd/advantage_clip_fraction",
        "opd/kl_scale",
        "opd/teacher_id",
        "policy/entropy",
        "ppo/ratio_mean",
        "ppo/clip_fraction",
        # router
        "router/p_medical",
        "router/p_base",
        "router/state",
        "router/medical_gap",
        "router/general_gap",
        "router/medical_ema",
        "router/general_ema",
        # eval (controller dev only; final test is reported separately)
        "eval_dev/medical_accuracy",
        "eval_dev/general_accuracy",
        # system
        "system/rollout_tokens_per_second",
        "system/teacher_prefill_tokens_per_second",
        "system/step_seconds",
        "system/gpu_memory_peak_gb",
    }
)

# Free-form bookkeeping keys allowed alongside metrics.
_RESERVED = frozenset({"step", "window", "wall_time", "domain", "phase", "run_id"})


class UnknownMetricError(KeyError):
    pass


class MetricsLogger:
    """Append-only ``metrics.jsonl`` writer."""

    def __init__(self, run_dir: str | Path, filename: str = "metrics.jsonl", run_id: str | None = None):
        self.run_dir = ensure_dir(run_dir)
        self.path = self.run_dir / filename
        self.run_id = run_id
        self._t0 = time.time()

    def log(self, step: int, metrics: Mapping[str, Any], **context: Any) -> Dict[str, Any]:
        unknown = [k for k in metrics if k not in METRIC_NAMES]
        if unknown:
            raise UnknownMetricError(
                f"unknown metric name(s) {sorted(unknown)}; add them to METRIC_NAMES "
                f"in src/utils/metrics.py if this is an intentional new series"
            )
        bad_context = set(context) - _RESERVED
        if bad_context:
            raise UnknownMetricError(f"unknown context key(s) {sorted(bad_context)}; allowed: {sorted(_RESERVED)}")

        record: Dict[str, Any] = {"step": int(step), "wall_time": round(time.time() - self._t0, 4)}
        if self.run_id:
            record["run_id"] = self.run_id
        record.update(context)
        record.update(dict(metrics))
        append_jsonl(self.path, record)
        return record

    def read_all(self) -> Iterable[Dict[str, Any]]:
        from .io import iter_jsonl

        if not self.path.exists():
            return []
        return iter_jsonl(self.path)
