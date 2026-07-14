"""Canonical statistical layer for LongHorizonMemSysBench (task 23; spec/03-protocol.md §4).

This module sits over the runner's tidy results table
(:class:`~lhmsb.runner.results.RunRow`) and the headline Memory ROI
(:mod:`lhmsb.metrics.roi`). It turns matched per-episode observations into the
*reported* statistics the scorecard (task 24) consumes — never a bare point estimate.

The benchmark is a **paired counterfactual** (spec §4.1): every ``(episode_id, seed)``
is run under every condition, so the unit of analysis is the episode-condition pair and
the right comparison is **paired** (condition vs. ``no_memory``), not a two-sample test
over unmatched pools. Four guarantees are enforced here:

  * **Bootstrap CIs** (spec §4.2): every mean / difference is reported with a seeded
    percentile bootstrap CI (default 95%, ``n=10,000``) drawn with ``random.Random(seed)``
    — fully reproducible, NEVER the global RNG.
  * **Paired effect size** (spec §4.3): :func:`paired_compare` pairs on
    ``(episode_id, seed)`` within a track, reporting the signed mean difference, the
    paired Cohen's d (``d_z = mean_diff / std_diff``), and a bootstrap CI of the diff.
  * **Failed runs INCLUDED** (spec §4.4): :func:`aggregate` never drops a failed run;
    its recorded ``task_score`` (0 under the failure policy) contributes to the mean and
    it is counted in the group's ``n``. Dropping failures would bias every aggregate up.
  * **Multiple-comparison hygiene**: :func:`multiple_comparison_note` emits a Bonferroni
    note whenever more than two systems are ranked.

NaN / N/A handling: a metric that is genuinely unavailable for a row (``drift_index`` on
an episode with no aligned probes, an optional retrieval precision) is skipped for THAT
metric only — the row is still counted in the group. ``task_score`` is always a real
number (0 on failure), so it is always included.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lhmsb.cost import CostConfig, scalarize
from lhmsb.metrics.roi import scalarize_memory_cost
from lhmsb.runner.results import RunRow

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

#: Default bootstrap resamples (spec §4.2: ≥10,000, configurable).
DEFAULT_BOOTSTRAP_N = 10_000
#: Two-sided significance level → a 95% CI by default (spec §4.2).
DEFAULT_ALPHA = 0.05
#: Conventional power target for the minimum-detectable-effect calculation (spec §4.3).
DEFAULT_TARGET_POWER = 0.80
#: The run-status token that marks a successful run; anything else is a failure
#: (matches :mod:`lhmsb.metrics.roi`). Failed runs are INCLUDED in aggregates (spec §4.4).
COMPLETED_STATUS = "completed"

#: Headline metrics aggregated by default (``cost`` is appended when a CostConfig is given).
DEFAULT_AGGREGATE_METRICS: tuple[str, ...] = (
    "task_score",
    "utilization_rate",
    "drift_index",
    "retrieval_endogenous_precision",
    "retrieval_oracle_precision",
)
#: Default grouping keys; track is always a key so native/controlled never mix (spec §3).
DEFAULT_AGGREGATE_BY: tuple[str, ...] = ("family", "condition", "track")

# Cohen's d magnitude cut points (Cohen 1988): |d| thresholds.
_EFFECT_NEGLIGIBLE = 0.2
_EFFECT_SMALL = 0.5
_EFFECT_MEDIUM = 0.8

# Metric-name → RunRow field classification (so getattr stays typed/validated).
_OPTIONAL_FLOAT_FIELDS = frozenset(
    {
        "utilization_rate",
        "improvement_over_time",
        "retrieval_endogenous_precision",
        "retrieval_endogenous_recall",
        "retrieval_endogenous_context",
        "retrieval_oracle_precision",
        "retrieval_oracle_recall",
        "retrieval_oracle_context",
        "storage_precision",
        "storage_recall",
        "storage_f1",
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_f1",
        "retrieval_false_positive_rate",
        "retrieval_timeliness",
    }
)
_REQUIRED_FLOAT_FIELDS = frozenset({"judge_contribution", "judge_fallback_share"})
_INT_FIELDS = frozenset(
    {
        "attempts",
        "n_probes",
        "stale_fact_violations",
        "constraint_violations",
        "behavioral_flips",
        "stored_memory_count",
        "unique_stored_memory_count",
        "retrieved_memory_count",
        "unique_retrieved_memory_count",
    }
)


# --------------------------------------------------------------------------- #
# Effect-size interpretation
# --------------------------------------------------------------------------- #
def interpret_effect_size(d: float) -> str:
    """Map a (signed) Cohen's d to a magnitude label (Cohen 1988); sign is ignored.

    ``negligible`` (``|d| < 0.2``) / ``small`` (``< 0.5``) / ``medium`` (``< 0.8``) /
    ``large`` (``≥ 0.8``). A ``NaN`` effect maps to ``"undefined"``.
    """
    if math.isnan(d):
        return "undefined"
    magnitude = abs(d)
    if magnitude < _EFFECT_NEGLIGIBLE:
        return "negligible"
    if magnitude < _EFFECT_SMALL:
        return "small"
    if magnitude < _EFFECT_MEDIUM:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Bootstrap CI (seeded, reproducible percentile method) — the canonical API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapCI:
    """A seeded percentile bootstrap confidence interval.

    ``point`` is the statistic on the full sample; ``lo`` / ``hi`` are the
    ``alpha/2`` / ``1 − alpha/2`` percentiles of the resampled statistics. ``n`` is the
    number of observations the interval was computed over (NOT the resample count).
    """

    point: float
    lo: float
    hi: float
    n: int
    statistic: str = "mean"
    alpha: float = DEFAULT_ALPHA

    def excludes_zero(self) -> bool:
        """``True`` iff the whole interval is strictly above or strictly below 0."""
        return self.lo > 0.0 or self.hi < 0.0

    def contains(self, value: float) -> bool:
        """``True`` iff ``value`` lies within the closed interval ``[lo, hi]``."""
        return self.lo <= value <= self.hi

    def width(self) -> float:
        """The interval width ``hi − lo`` (always ``≥ 0``)."""
        return self.hi - self.lo


def _statistic_fn(name: str) -> Callable[[Sequence[float]], float]:
    """Resolve a statistic name to a function over a numeric sequence."""
    if name == "mean":
        return lambda values: statistics.fmean(values)
    if name == "median":
        return lambda values: float(statistics.median(values))
    if name == "sum":
        return lambda values: float(sum(values))
    raise ValueError(f"unknown statistic: {name!r} (expected 'mean', 'median', or 'sum')")


def _percentile_index(quantile: float, n: int) -> int:
    """Index into a length-``n`` sorted list for ``quantile``, clamped to ``[0, n-1]``."""
    return max(0, min(n - 1, int(quantile * n)))


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> BootstrapCI:
    """Seeded percentile bootstrap CI → :class:`BootstrapCI` (``point``, ``lo``, ``hi``).

    ``point`` is ``statistic`` on the full sample; ``lo`` / ``hi`` are the ``alpha/2`` /
    ``1 − alpha/2`` percentiles of ``n`` resampled statistics, each drawn with replacement
    using ``random.Random(seed)`` — fully reproducible (never the global RNG). A single /
    constant sample collapses the interval to the point. Raises on empty input. Supports
    ``"mean"``, ``"median"``, and ``"sum"``.
    """
    data = [float(value) for value in values]
    if not data:
        raise ValueError("bootstrap_ci requires at least one value")
    stat = _statistic_fn(statistic)
    point = stat(data)
    size = len(data)
    if size == 1:
        return BootstrapCI(point, point, point, n=1, statistic=statistic, alpha=alpha)

    rng = random.Random(seed)
    resampled = sorted(stat(rng.choices(data, k=size)) for _ in range(n))
    lo = resampled[_percentile_index(alpha / 2.0, n)]
    hi = resampled[_percentile_index(1.0 - alpha / 2.0, n)]
    return BootstrapCI(point, lo, hi, n=size, statistic=statistic, alpha=alpha)


# --------------------------------------------------------------------------- #
# Metric extraction off a RunRow (string name or custom callable)
# --------------------------------------------------------------------------- #
def _clean(value: float | None) -> float | None:
    """Map ``None`` / ``NaN`` to ``None`` (genuinely N/A); otherwise return the float."""
    if value is None:
        return None
    return None if math.isnan(value) else float(value)


def _name_metric(row: RunRow, name: str, cost_config: CostConfig | None) -> float | None:
    """Extract a named metric from a row, applying the failed-runs / N/A policies.

    ``task_score`` is ALWAYS a real number (0 on failure) so failed runs are never
    dropped from its aggregate (spec §4.4). ``drift_index`` is N/A when ``drift_is_na``.
    ``cost`` / ``memory_cost`` need a :class:`~lhmsb.cost.CostConfig` (else N/A).
    """
    if name == "task_score":
        score = row.task_score
        return 0.0 if math.isnan(score) else float(score)
    if name == "drift_index":
        return None if row.drift_is_na else _clean(row.drift_index)
    if name == "cost":
        if cost_config is None:
            return None
        return scalarize(row.cost, cost_config.weights, cost_config.conversion)
    if name == "memory_cost":
        if cost_config is None:
            return None
        return scalarize_memory_cost(row.cost, cost_config)
    if name in _OPTIONAL_FLOAT_FIELDS or name in _REQUIRED_FLOAT_FIELDS:
        return _clean(getattr(row, name))
    if name in _INT_FIELDS:
        return float(getattr(row, name))
    raise ValueError(f"unknown metric: {name!r}")


def _resolve_metric(
    metric: str | Callable[[RunRow], float | None],
    cost_config: CostConfig | None,
) -> Callable[[RunRow], float | None]:
    """Turn a metric NAME (or a custom accessor) into a ``RunRow -> float | None`` fn."""
    if callable(metric):
        return metric
    return lambda row: _name_metric(row, metric, cost_config)


def _metric_label(metric: str | Callable[[RunRow], float | None]) -> str:
    """A human-readable label for a metric name or callable."""
    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", "custom")


# --------------------------------------------------------------------------- #
# Paired comparison (the counterfactual is paired — NEVER an unpaired test)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairedComparison:
    """A paired comparison of two systems on one metric over matched episodes.

    The difference is ``metric(system_a) − metric(system_b)`` per shared
    ``(episode_id, seed, track)`` cell. ``cohens_dz`` is the paired Cohen's d
    (``mean_diff / std_diff``); its sign follows ``mean_diff``. ``ci`` is a bootstrap CI
    of the per-pair differences — a real effect is one whose CI :meth:`excludes zero
    <BootstrapCI.excludes_zero>`.
    """

    system_a: str
    system_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float
    std_diff: float
    cohens_dz: float
    effect_magnitude: str
    ci: BootstrapCI

    @property
    def significant(self) -> bool:
        """``True`` iff the bootstrap CI of the paired difference excludes 0."""
        return self.ci.excludes_zero()


_PairKey = tuple[str, int, str]


def _keyed_values(
    rows: Sequence[RunRow],
    condition: str,
    accessor: Callable[[RunRow], float | None],
) -> dict[_PairKey, float]:
    """Usable metric values for one condition, keyed by ``(episode_id, seed, track)``.

    Track is part of the key so native and controlled cells never pair across tracks
    (spec §3). Rows whose metric is N/A are omitted (they cannot form a difference).
    """
    values: dict[_PairKey, float] = {}
    for row in rows:
        if row.condition != condition:
            continue
        value = accessor(row)
        if value is None:
            continue
        values[(row.episode_id, row.seed, row.track)] = value
    return values


def paired_compare(
    system_a: str,
    system_b: str,
    metric: str | Callable[[RunRow], float | None],
    rows: Sequence[RunRow],
    *,
    family: str | None = None,
    track: str | None = None,
    cost_config: CostConfig | None = None,
    n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> PairedComparison:
    """Paired comparison of ``system_a`` vs ``system_b`` on ``metric`` (spec §4.1, §4.3).

    Pairs ONLY rows that share an ``(episode_id, seed, track)`` cell (optionally filtered
    to one ``family`` / ``track`` first), computes the per-pair difference
    ``metric(a) − metric(b)``, then the mean difference, the sample std of the differences,
    the paired Cohen's d (``mean_diff / std_diff``, 0 when ``std_diff == 0``), and a seeded
    bootstrap CI of the differences. Raises ``ValueError`` when no cell is shared (a paired
    test is meaningless without matched observations — never silently returns 0).
    """
    accessor = _resolve_metric(metric, cost_config)
    selected = [
        row
        for row in rows
        if (family is None or row.family == family) and (track is None or row.track == track)
    ]
    a_values = _keyed_values(selected, system_a, accessor)
    b_values = _keyed_values(selected, system_b, accessor)
    shared = sorted(set(a_values) & set(b_values))
    if not shared:
        raise ValueError(
            f"no paired (episode_id, seed, track) observations for {system_a!r} vs "
            f"{system_b!r} on metric {_metric_label(metric)!r}"
        )

    paired_a = [a_values[key] for key in shared]
    paired_b = [b_values[key] for key in shared]
    diffs = [a - b for a, b in zip(paired_a, paired_b, strict=True)]
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) >= 2 else 0.0
    cohens_dz = mean_diff / std_diff if std_diff > 0.0 else 0.0
    return PairedComparison(
        system_a=system_a,
        system_b=system_b,
        metric=_metric_label(metric),
        n_pairs=len(shared),
        mean_a=statistics.fmean(paired_a),
        mean_b=statistics.fmean(paired_b),
        mean_diff=mean_diff,
        std_diff=std_diff,
        cohens_dz=cohens_dz,
        effect_magnitude=interpret_effect_size(cohens_dz),
        ci=bootstrap_ci(diffs, "mean", n, alpha, seed),
    )


# --------------------------------------------------------------------------- #
# Aggregation (means + CIs per group; failed runs INCLUDED, not dropped)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricStats:
    """Mean + bootstrap CI for one metric within one group.

    ``n`` is the number of rows with a USABLE value; ``n_na`` is the number whose value
    was N/A for this metric (those rows are still counted in :pyattr:`GroupStats.n_rows`).
    ``mean`` / ``ci`` are ``None`` only when every value in the group was N/A.
    """

    metric: str
    n: int
    n_na: int
    mean: float | None
    ci: BootstrapCI | None
    status: str

    @property
    def is_na(self) -> bool:
        """``True`` when no usable values existed (mean / CI are ``None``)."""
        return self.status == "na"


@dataclass(frozen=True)
class GroupStats:
    """All metric statistics for one group of the aggregation.

    ``n_rows`` counts EVERY row in the group, including failed runs (spec §4.4);
    ``n_failed`` is how many of those had a non-``completed`` status.
    """

    key: dict[str, str]
    n_rows: int
    n_failed: int
    metrics: dict[str, MetricStats]

    def metric(self, name: str) -> MetricStats:
        """The :class:`MetricStats` for ``name`` (raises ``KeyError`` if not aggregated)."""
        return self.metrics[name]


@dataclass(frozen=True)
class AggregatedStats:
    """The full aggregation: one :class:`GroupStats` per group key combination."""

    by: tuple[str, ...]
    metric_names: tuple[str, ...]
    groups: list[GroupStats]

    def group_for(self, **key: str) -> GroupStats | None:
        """The group whose key exactly matches ``key`` (all ``by`` fields), or ``None``."""
        for group in self.groups:
            if group.key == key:
                return group
        return None


def _resolve_aggregate_metrics(
    metrics: Sequence[str] | None, cost_config: CostConfig | None
) -> tuple[str, ...]:
    """Default metric set (``+ "cost"`` when a CostConfig is supplied) or the override."""
    if metrics is not None:
        return tuple(metrics)
    names = list(DEFAULT_AGGREGATE_METRICS)
    if cost_config is not None:
        names.append("cost")
    return tuple(names)


def aggregate(
    rows: Sequence[RunRow],
    by: Sequence[str] = DEFAULT_AGGREGATE_BY,
    metrics: Sequence[str] | None = None,
    *,
    cost_config: CostConfig | None = None,
    n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> AggregatedStats:
    """Per-group mean + bootstrap CI for each metric, with **failed runs INCLUDED**.

    Rows are grouped by the ``by`` keys (RunRow attribute names; track is a default key so
    native / controlled never mix — spec §3). For each group and metric a seeded bootstrap
    CI is computed over the USABLE values; N/A values are skipped for that metric only, but
    the row is still counted in ``n_rows``. A failed run's recorded ``task_score`` (0) is a
    real value, so it contributes to the mean and is never silently dropped (spec §4.4).
    """
    by_keys = tuple(by)
    metric_names = _resolve_aggregate_metrics(metrics, cost_config)
    accessors = {name: _resolve_metric(name, cost_config) for name in metric_names}

    grouped: dict[tuple[str, ...], list[RunRow]] = {}
    for row in rows:
        key_values = tuple(str(getattr(row, field_name)) for field_name in by_keys)
        grouped.setdefault(key_values, []).append(row)

    groups: list[GroupStats] = []
    for group_index, key_values in enumerate(sorted(grouped)):
        group_rows = grouped[key_values]
        key = dict(zip(by_keys, key_values, strict=True))
        n_failed = sum(1 for row in group_rows if row.status != COMPLETED_STATUS)
        metric_stats: dict[str, MetricStats] = {}
        for metric_index, name in enumerate(metric_names):
            accessor = accessors[name]
            collected = [accessor(row) for row in group_rows]
            usable = [value for value in collected if value is not None]
            n_na = len(collected) - len(usable)
            if usable:
                ci = bootstrap_ci(
                    usable, "mean", n, alpha, seed + group_index * len(metric_names) + metric_index
                )
                metric_stats[name] = MetricStats(
                    metric=name, n=len(usable), n_na=n_na, mean=ci.point, ci=ci, status="ok"
                )
            else:
                metric_stats[name] = MetricStats(
                    metric=name, n=0, n_na=n_na, mean=None, ci=None, status="na"
                )
        groups.append(
            GroupStats(key=key, n_rows=len(group_rows), n_failed=n_failed, metrics=metric_stats)
        )
    return AggregatedStats(by=by_keys, metric_names=metric_names, groups=groups)


# --------------------------------------------------------------------------- #
# Power / minimum-detectable-effect note (z-approximation, paired design)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PowerNote:
    """Observed power + minimum detectable effect for a paired comparison at size ``n``.

    ``power`` is the probability of detecting ``effect_size`` (Cohen's d_z) at the current
    ``n`` and two-sided ``alpha``; ``mde`` is the smallest ``|d_z|`` detectable at
    ``target_power``. Both use the normal (z) approximation to the paired-t power.
    """

    effect_size: float
    n: int
    alpha: float
    target_power: float
    power: float
    mde: float
    note: str


def power_note(
    effect_size: float,
    n: int,
    alpha: float = DEFAULT_ALPHA,
    *,
    target_power: float = DEFAULT_TARGET_POWER,
) -> PowerNote:
    """Observed power + minimum detectable effect at ``n`` (spec §4.3; z-approximation).

    For a two-sided paired test at level ``alpha``: the minimum detectable effect (in
    Cohen's d_z units) at ``target_power`` is ``(z_{1−α/2} + z_power) / √n``; the observed
    power for ``effect_size`` is ``Φ(δ − z_{1−α/2}) + Φ(−δ − z_{1−α/2})`` with
    ``δ = |effect_size|·√n``. ``n < 2`` yields an undefined note (``mde = ∞``, ``power = NaN``).
    """
    normal = statistics.NormalDist()
    if n < 2:
        return PowerNote(
            effect_size=float(effect_size),
            n=n,
            alpha=alpha,
            target_power=target_power,
            power=math.nan,
            mde=math.inf,
            note=f"n={n}: power and minimum detectable effect are undefined (need ≥2 pairs).",
        )

    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(target_power)
    root_n = math.sqrt(n)
    mde = (z_alpha + z_power) / root_n
    delta = abs(effect_size) * root_n
    power = min(1.0, max(0.0, normal.cdf(delta - z_alpha) + normal.cdf(-delta - z_alpha)))
    note = (
        f"At n={n} (two-sided α={alpha:g}, z-approx paired test): observed power to detect "
        f"d_z={effect_size:.3f} is {power:.2f}; the minimum effect detectable at "
        f"{target_power:.0%} power is d_z≈{mde:.3f}."
    )
    return PowerNote(
        effect_size=float(effect_size),
        n=n,
        alpha=alpha,
        target_power=target_power,
        power=power,
        mde=mde,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Multiple-comparison note (Bonferroni when ranking > 2 systems)
# --------------------------------------------------------------------------- #
def multiple_comparison_note(n_systems: int, alpha: float = DEFAULT_ALPHA) -> str:
    """Bonferroni note for ranking ``n_systems`` systems (≤ 2 → no correction needed).

    Ranking ``k`` systems entails ``k(k−1)/2`` pairwise comparisons, inflating the
    family-wise error rate; the note reports the Bonferroni-corrected per-comparison
    ``alpha`` and reminds the reader that bootstrap CIs need no parametric correction.
    """
    if n_systems <= 2:
        return (
            f"{n_systems} system(s) compared: no multiple-comparison correction needed "
            f"(a single pairwise comparison is reported at α={alpha:g})."
        )
    pairwise = n_systems * (n_systems - 1) // 2
    corrected = alpha / pairwise
    return (
        f"Ranking {n_systems} systems entails {pairwise} pairwise comparisons, which inflates "
        f"the family-wise error rate. Apply a Bonferroni correction: per-comparison α = "
        f"{alpha:g}/{pairwise} = {corrected:.4g}. Prefer the reported bootstrap CIs (no "
        f"parametric correction needed) and treat a difference as significant only if its CI "
        f"excludes 0 at the corrected level."
    )
