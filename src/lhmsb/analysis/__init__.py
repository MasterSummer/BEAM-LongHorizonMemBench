"""Statistical analysis layer for LongHorizonMemSysBench (task 23; spec/03-protocol.md §4).

The benchmark is a paired counterfactual, so analysis is paired-by-design:

  * :func:`bootstrap_ci` — seeded, reproducible percentile bootstrap CI
    (:class:`BootstrapCI`), the canonical CI utility for every headline metric.
  * :func:`paired_compare` — paired (episode_id, seed) comparison of two systems on a
    metric (:class:`PairedComparison`): signed mean difference, paired Cohen's d_z, and a
    bootstrap CI of the difference.
  * :func:`aggregate` — per-group means + bootstrap CIs (:class:`AggregatedStats` /
    :class:`GroupStats` / :class:`MetricStats`) with **failed runs INCLUDED** (spec §4.4).
  * :func:`power_note` — observed power + minimum detectable effect (:class:`PowerNote`).
  * :func:`multiple_comparison_note` — Bonferroni note when ranking > 2 systems.
  * :func:`interpret_effect_size` — Cohen's d magnitude label.
"""

from __future__ import annotations

from lhmsb.analysis.stats import (
    COMPLETED_STATUS,
    DEFAULT_AGGREGATE_BY,
    DEFAULT_AGGREGATE_METRICS,
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_TARGET_POWER,
    AggregatedStats,
    BootstrapCI,
    GroupStats,
    MetricStats,
    PairedComparison,
    PowerNote,
    aggregate,
    bootstrap_ci,
    interpret_effect_size,
    multiple_comparison_note,
    paired_compare,
    power_note,
)

__all__ = [
    "AggregatedStats",
    "BootstrapCI",
    "COMPLETED_STATUS",
    "DEFAULT_AGGREGATE_BY",
    "DEFAULT_AGGREGATE_METRICS",
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_N",
    "DEFAULT_TARGET_POWER",
    "GroupStats",
    "MetricStats",
    "PairedComparison",
    "PowerNote",
    "aggregate",
    "bootstrap_ci",
    "interpret_effect_size",
    "multiple_comparison_note",
    "paired_compare",
    "power_note",
]
