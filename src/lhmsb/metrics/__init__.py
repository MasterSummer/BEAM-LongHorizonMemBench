"""Scorecard metrics for LongHorizonMemSysBench (spec/02-metrics.md).

v1 dimensions implemented here:
  - Dim 2 — Task performance & goal-directed utilization
    (:mod:`lhmsb.metrics.task_score`): :class:`TaskScore`, :func:`score_task`,
    :func:`utilization_rate`, :func:`improvement_over_time`.
  - Dim 3 — Goal drift & behavioral stability
    (:mod:`lhmsb.metrics.drift`): :func:`drift_index`, :class:`DriftReport`,
    :class:`DriftWeights`.
  - Dim 7 — Token/resource efficiency via the headline **Memory ROI**
    (:mod:`lhmsb.metrics.roi`): :class:`RoiResult`, :func:`compute_roi`,
    :func:`normalized_gain`, :func:`memory_attributable_cost`,
    :func:`scalarize_memory_cost`, :func:`bootstrap_ci`, :func:`pareto_front`.

The per-episode metrics (Dims 2/3) consume a finished
:class:`~lhmsb.types.EpisodeResult` plus its :class:`~lhmsb.types.Episode`; the
cross-cutting ROI (Dim 7) consumes the runner's
:class:`~lhmsb.runner.results.ResultsTable`. Nothing here re-runs the agent,
checkers, or judge.
"""

from __future__ import annotations

from lhmsb.metrics.drift import (
    MAX_JUDGE_FALLBACK_SHARE,
    DriftReport,
    DriftWeights,
    drift_index,
)
from lhmsb.metrics.memory_efficiency import MemoryEfficiencyReport, measure_memory_efficiency
from lhmsb.metrics.memory_roi import MemoryRoiResult, compute_memory_roi
from lhmsb.metrics.roi import (
    RoiResult,
    bootstrap_ci,
    compute_roi,
    memory_attributable_cost,
    normalized_gain,
    pareto_front,
    scalarize_memory_cost,
)
from lhmsb.metrics.task_score import (
    MAX_JUDGE_WEIGHT,
    TaskScore,
    improvement_over_time,
    score_task,
    utilization_rate,
)
from lhmsb.metrics.wide_research import WideSetMetrics, compute_wide_set_metrics

__all__ = [
    "MAX_JUDGE_FALLBACK_SHARE",
    "MAX_JUDGE_WEIGHT",
    "DriftReport",
    "DriftWeights",
    "RoiResult",
    "TaskScore",
    "bootstrap_ci",
    "compute_roi",
    "drift_index",
    "improvement_over_time",
    "memory_attributable_cost",
    "normalized_gain",
    "pareto_front",
    "scalarize_memory_cost",
    "score_task",
    "utilization_rate",
    "WideSetMetrics",
    "compute_wide_set_metrics",
    "MemoryEfficiencyReport",
    "measure_memory_efficiency",
    "MemoryRoiResult",
    "compute_memory_roi",
]
