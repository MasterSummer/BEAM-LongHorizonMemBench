"""TDD tests for the statistical analysis layer (lhmsb.analysis.stats, spec/03-protocol.md §4).

The benchmark is a **paired counterfactual**: every (episode_id, seed) is run under every
condition, so comparisons are PAIRED (condition vs. no_memory), never unpaired two-sample
tests. These tests pin the four reviewer hot-buttons:

  * §4.2 Bootstrap CIs — seeded, reproducible, bracketing the point estimate.
  * §4.3 Effect size — paired Cohen's d_z with the correct sign + magnitude; the CI of the
    paired difference excludes 0 only when the effect is real.
  * §4.4 Failed runs INCLUDED — a failed run's task_score (0) contributes to the aggregate
    and is counted in n; it is never silently dropped.
  * Multiple comparisons — a Bonferroni note appears when ranking > 2 systems.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from lhmsb.analysis.stats import (
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
from lhmsb.cost import CostConfig, load_cost_config
from lhmsb.runner import ResultsTable, RunRow
from lhmsb.types import CostVector

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cost_weights.yaml"


@pytest.fixture
def cost_config() -> CostConfig:
    return load_cost_config(_CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _row(
    *,
    condition: str,
    task_score: float,
    episode_id: str = "e1",
    family: str = "research",
    seed: int = 1,
    track: str = "native",
    status: str = "completed",
    utilization_rate: float | None = 0.5,
    drift_index: float = 0.0,
    drift_is_na: bool = False,
    retrieval_endogenous_precision: float | None = 0.5,
    retrieval_oracle_precision: float | None = 0.5,
    cost: CostVector | None = None,
) -> RunRow:
    """A fully-graded RunRow with stat-irrelevant fields at neutral defaults."""
    return RunRow(
        episode_id=episode_id,
        family=family,
        seed=seed,
        condition=condition,
        track=track,
        status=status,
        attempts=1,
        n_probes=2,
        world_event_hash="weh",
        episode_hash="eh",
        task_score=task_score,
        utilization_rate=utilization_rate,
        improvement_over_time=None,
        judge_contribution=0.0,
        drift_index=drift_index,
        drift_is_na=drift_is_na,
        stale_fact_violations=0,
        constraint_violations=0,
        behavioral_flips=0,
        judge_fallback_share=0.0,
        retrieval_endogenous_precision=retrieval_endogenous_precision,
        retrieval_oracle_precision=retrieval_oracle_precision,
        cost=cost if cost is not None else CostVector(),
    )


# Crafted paired data: A beats B by a known, low-variance margin on every episode.
_PAIRED_EPISODES = ("e1", "e2", "e3", "e4", "e5")
_A_SCORES = {"e1": 0.80, "e2": 0.70, "e3": 0.90, "e4": 0.85, "e5": 0.75}
_B_SCORES = {"e1": 0.30, "e2": 0.40, "e3": 0.35, "e4": 0.45, "e5": 0.30}


def _paired_rows() -> list[RunRow]:
    rows: list[RunRow] = []
    for ep in _PAIRED_EPISODES:
        rows.append(_row(condition="chroma", episode_id=ep, task_score=_A_SCORES[ep]))
        rows.append(_row(condition="no_memory", episode_id=ep, task_score=_B_SCORES[ep]))
    return rows


# --------------------------------------------------------------------------- #
# bootstrap_ci — seeded, reproducible, brackets the point estimate
# --------------------------------------------------------------------------- #
def test_bootstrap_ci() -> None:
    """Seeded → reproducible; point is the sample mean; interval brackets it (finite)."""
    values = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7]
    first = bootstrap_ci(values, n=2000, seed=42)
    second = bootstrap_ci(values, n=2000, seed=42)
    assert first == second  # identical (point, lo, hi) for the same seed
    assert isinstance(first, BootstrapCI)
    assert first.point == pytest.approx(statistics.fmean(values))
    assert first.lo <= first.point <= first.hi
    assert math.isfinite(first.lo) and math.isfinite(first.hi)
    assert first.n == len(values)
    assert first.width() >= 0.0


def test_bootstrap_ci_supports_median() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 100.0]  # outlier: median << mean
    ci = bootstrap_ci(values, statistic="median", n=2000, seed=0)
    assert ci.point == pytest.approx(statistics.median(values))
    assert ci.statistic == "median"
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_ci_constant_values_collapse() -> None:
    ci = bootstrap_ci([0.5, 0.5, 0.5], n=500, seed=1)
    assert ci.point == ci.lo == ci.hi == 0.5
    assert ci.contains(0.5)
    assert ci.width() == 0.0
    assert ci.excludes_zero()  # the whole interval sits strictly above 0


def test_bootstrap_ci_single_value_collapses() -> None:
    ci = bootstrap_ci([0.7], n=500, seed=1)
    assert ci.point == ci.lo == ci.hi == 0.7
    assert ci.n == 1


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([], n=10, seed=0)


def test_bootstrap_ci_unknown_statistic_raises() -> None:
    with pytest.raises(ValueError, match="unknown statistic"):
        bootstrap_ci([1.0, 2.0], statistic="variance", n=10, seed=0)


# --------------------------------------------------------------------------- #
# paired_compare — correct sign + magnitude; CI excludes 0 when effect is real
# --------------------------------------------------------------------------- #
def test_paired_episode_aggregation() -> None:
    """A>B by a known margin → positive mean_diff, positive d_z, CI excludes 0."""
    rows = _paired_rows()
    pc = paired_compare("chroma", "no_memory", "task_score", rows, n=3000, seed=7)

    diffs = [_A_SCORES[ep] - _B_SCORES[ep] for ep in _PAIRED_EPISODES]
    expected_mean_diff = statistics.fmean(diffs)
    expected_std = statistics.stdev(diffs)

    assert isinstance(pc, PairedComparison)
    assert pc.n_pairs == 5  # one matched (episode_id, seed, track) per episode
    assert pc.mean_a == pytest.approx(statistics.fmean(list(_A_SCORES.values())))
    assert pc.mean_b == pytest.approx(statistics.fmean(list(_B_SCORES.values())))
    # Correct sign + magnitude on the paired difference.
    assert pc.mean_diff == pytest.approx(expected_mean_diff)
    assert pc.mean_diff > 0.0
    assert pc.std_diff == pytest.approx(expected_std)
    assert pc.cohens_dz == pytest.approx(expected_mean_diff / expected_std)
    assert pc.cohens_dz > 0.0
    assert pc.effect_magnitude == "large"  # d_z ≈ 4.6
    # The bootstrap CI of the difference excludes 0 (a real effect) and is significant.
    assert pc.ci.excludes_zero()
    assert pc.ci.lo > 0.0
    assert pc.significant is True


def test_paired_compare_sign_reverses_when_swapped() -> None:
    """Swapping the operands negates the diff + effect size; CI now excludes 0 below."""
    rows = _paired_rows()
    forward = paired_compare("chroma", "no_memory", "task_score", rows, n=3000, seed=7)
    reverse = paired_compare("no_memory", "chroma", "task_score", rows, n=3000, seed=7)

    assert reverse.mean_diff == pytest.approx(-forward.mean_diff)
    assert reverse.cohens_dz == pytest.approx(-forward.cohens_dz)
    assert reverse.mean_diff < 0.0
    assert reverse.cohens_dz < 0.0
    assert reverse.effect_magnitude == "large"  # magnitude ignores sign
    assert reverse.ci.excludes_zero()
    assert reverse.ci.hi < 0.0


def test_paired_compare_pairs_only_matching_episode_seed() -> None:
    """Only rows sharing (episode_id, seed) are paired; unmatched cells are ignored."""
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.8),
        _row(condition="chroma", episode_id="e2", task_score=0.7),
        _row(condition="chroma", episode_id="e3", task_score=0.9),  # no B partner
        _row(condition="no_memory", episode_id="e1", task_score=0.3),
        _row(condition="no_memory", episode_id="e2", task_score=0.4),
        _row(condition="no_memory", episode_id="e4", task_score=0.5),  # no A partner
    ]
    pc = paired_compare("chroma", "no_memory", "task_score", rows, n=1000, seed=0)
    # Only e1 + e2 are shared → 2 pairs; e3 and e4 are excluded.
    assert pc.n_pairs == 2
    assert pc.mean_diff == pytest.approx(statistics.fmean([0.8 - 0.3, 0.7 - 0.4]))


def test_paired_compare_does_not_cross_tracks() -> None:
    """native and controlled cells with the same (episode_id, seed) never pair together."""
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.80, track="native"),
        _row(condition="chroma", episode_id="e1", task_score=0.60, track="controlled"),
        _row(condition="no_memory", episode_id="e1", task_score=0.30, track="native"),
        _row(condition="no_memory", episode_id="e1", task_score=0.20, track="controlled"),
    ]
    both = paired_compare("chroma", "no_memory", "task_score", rows, n=1000, seed=0)
    assert both.n_pairs == 2  # native pair + controlled pair, kept separate
    assert both.mean_diff == pytest.approx(statistics.fmean([0.80 - 0.30, 0.60 - 0.20]))

    native = paired_compare("chroma", "no_memory", "task_score", rows, track="native", n=1000)
    assert native.n_pairs == 1
    assert native.mean_diff == pytest.approx(0.50)


def test_paired_compare_no_effect_ci_includes_zero() -> None:
    """Zero mean difference with spread → d_z 0, CI straddles 0 (not significant)."""
    a_scores = {"e1": 0.50, "e2": 0.00, "e3": 0.30, "e4": 0.00, "e5": 0.20}
    b_scores = {"e1": 0.00, "e2": 0.50, "e3": 0.00, "e4": 0.30, "e5": 0.20}
    rows: list[RunRow] = []
    for ep in _PAIRED_EPISODES:
        rows.append(_row(condition="a", episode_id=ep, task_score=a_scores[ep]))
        rows.append(_row(condition="b", episode_id=ep, task_score=b_scores[ep]))
    pc = paired_compare("a", "b", "task_score", rows, n=3000, seed=0)
    assert pc.mean_diff == pytest.approx(0.0)
    assert pc.cohens_dz == pytest.approx(0.0)
    assert pc.effect_magnitude == "negligible"
    assert not pc.ci.excludes_zero()
    assert pc.ci.contains(0.0)
    assert pc.significant is False


def test_paired_compare_raises_without_shared_pairs() -> None:
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.8),
        _row(condition="no_memory", episode_id="e2", task_score=0.3),
    ]
    with pytest.raises(ValueError, match="no paired"):
        paired_compare("chroma", "no_memory", "task_score", rows, n=100, seed=0)


def test_paired_compare_accepts_callable_metric() -> None:
    """A custom accessor works in place of a metric name."""
    rows = _paired_rows()
    pc = paired_compare(
        "chroma", "no_memory", lambda row: row.task_score, rows, n=1000, seed=0
    )
    assert pc.n_pairs == 5
    assert pc.mean_diff > 0.0


# --------------------------------------------------------------------------- #
# Effect size — sign / magnitude thresholds (Cohen 1988)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0.0, "negligible"),
        (0.19, "negligible"),
        (0.2, "small"),
        (0.49, "small"),
        (0.5, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
        (1.5, "large"),
    ],
)
def test_interpret_effect_size_thresholds(value: float, label: str) -> None:
    assert interpret_effect_size(value) == label


def test_interpret_effect_size_ignores_sign() -> None:
    assert interpret_effect_size(-0.6) == "medium"
    assert interpret_effect_size(-1.0) == "large"
    assert interpret_effect_size(-0.1) == "negligible"


def test_interpret_effect_size_nan_is_undefined() -> None:
    assert interpret_effect_size(math.nan) == "undefined"


# --------------------------------------------------------------------------- #
# aggregate — failed runs INCLUDED, not silently dropped (spec §4.4)
# --------------------------------------------------------------------------- #
def test_failed_runs_are_included_not_silently_dropped() -> None:
    """A group's 2 failed runs (task_score 0) count in n AND drag the mean down."""
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.60),
        _row(condition="chroma", episode_id="e2", task_score=0.70),
        _row(condition="chroma", episode_id="e3", task_score=0.80),
        _row(condition="chroma", episode_id="e4", task_score=0.0, status="failed"),
        _row(condition="chroma", episode_id="e5", task_score=0.0, status="timeout"),
    ]
    agg = aggregate(rows, by=["condition"], metrics=["task_score"], n=1000, seed=0)
    grp = agg.group_for(condition="chroma")
    assert grp is not None

    # Every row is counted, including the 2 failures (NOT dropped).
    assert grp.n_rows == 5
    assert grp.n_failed == 2
    stat = grp.metrics["task_score"]
    assert stat.n == 5  # all 5 task_scores are usable (failures contribute 0)
    assert stat.n_na == 0
    # The mean includes the two 0s; dropping them would (wrongly) give 0.70.
    assert stat.mean == pytest.approx((0.60 + 0.70 + 0.80) / 5)  # = 0.42
    assert stat.mean != pytest.approx(0.70)
    assert stat.ci is not None and stat.ci.point == pytest.approx(stat.mean)


def test_aggregate_groups_and_brackets_means() -> None:
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.60),
        _row(condition="chroma", episode_id="e2", task_score=0.80),
        _row(condition="no_memory", episode_id="e1", task_score=0.30),
        _row(condition="no_memory", episode_id="e2", task_score=0.50),
    ]
    agg = aggregate(rows, by=["condition"], metrics=["task_score"], n=2000, seed=0)
    assert isinstance(agg, AggregatedStats)
    assert agg.by == ("condition",)
    assert {g.key["condition"] for g in agg.groups} == {"chroma", "no_memory"}

    chroma = agg.group_for(condition="chroma")
    assert chroma is not None
    stat = chroma.metrics["task_score"]
    assert stat.mean == pytest.approx(0.70)
    assert stat.ci is not None
    assert stat.ci.lo <= stat.mean <= stat.ci.hi


def test_aggregate_skips_na_values_but_keeps_rows() -> None:
    """None / NaN metric values are N/A for that metric only; the row is still counted."""
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.6, utilization_rate=0.4),
        _row(condition="chroma", episode_id="e2", task_score=0.7, utilization_rate=None),
        _row(
            condition="chroma",
            episode_id="e3",
            task_score=0.8,
            utilization_rate=None,
            drift_index=math.nan,
            drift_is_na=True,
        ),
    ]
    agg = aggregate(
        rows, by=["condition"], metrics=["task_score", "utilization_rate", "drift_index"], n=500
    )
    grp = agg.group_for(condition="chroma")
    assert grp is not None
    assert grp.n_rows == 3
    # task_score: all present.
    assert grp.metrics["task_score"].n == 3 and grp.metrics["task_score"].n_na == 0
    # utilization_rate: two None skipped, but the rows remain counted in n_rows.
    util = grp.metrics["utilization_rate"]
    assert util.n == 1 and util.n_na == 2
    assert util.mean == pytest.approx(0.4)
    # drift_index: e1,e2 drift 0.0 usable; e3 is flagged N/A and skipped.
    drift = grp.metrics["drift_index"]
    assert drift.n == 2 and drift.n_na == 1
    assert drift.mean == pytest.approx(0.0)


def test_aggregate_all_na_metric_reports_none() -> None:
    rows = [_row(condition="chroma", episode_id="e1", task_score=0.6, utilization_rate=None)]
    agg = aggregate(rows, by=["condition"], metrics=["utilization_rate"], n=100)
    grp = agg.group_for(condition="chroma")
    assert grp is not None
    util = grp.metrics["utilization_rate"]
    assert util.mean is None and util.ci is None and util.is_na is True
    assert util.n == 0 and util.n_na == 1


def test_aggregate_includes_cost_when_config_provided(cost_config: CostConfig) -> None:
    mem_cost = CostVector(mem_internal_in_tokens=1000)
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.6, cost=mem_cost),
        _row(condition="chroma", episode_id="e2", task_score=0.7, cost=mem_cost),
    ]
    agg = aggregate(rows, by=["condition"], cost_config=cost_config, n=500)
    assert "cost" in agg.metric_names  # appended to the default metric set
    grp = agg.group_for(condition="chroma")
    assert grp is not None
    assert grp.metrics["cost"].mean == pytest.approx(1000.0)


def test_aggregate_default_by_keeps_tracks_separate() -> None:
    """The default grouping includes track, so native/controlled never merge (spec §3)."""
    rows = [
        _row(condition="chroma", episode_id="e1", task_score=0.8, track="native"),
        _row(condition="chroma", episode_id="e1", task_score=0.6, track="controlled"),
    ]
    agg = aggregate(rows, metrics=["task_score"], n=200)
    assert agg.by == ("family", "condition", "track")
    native = agg.group_for(family="research", condition="chroma", track="native")
    controlled = agg.group_for(family="research", condition="chroma", track="controlled")
    assert native is not None and controlled is not None
    assert native.metrics["task_score"].mean == pytest.approx(0.8)
    assert controlled.metrics["task_score"].mean == pytest.approx(0.6)


def test_aggregate_from_results_table_rows() -> None:
    """aggregate consumes the runner's ResultsTable.rows directly."""
    table = ResultsTable(
        track="native",
        rows=[
            _row(condition="chroma", episode_id="e1", task_score=0.6),
            _row(condition="chroma", episode_id="e2", task_score=0.8),
        ],
    )
    agg = aggregate(table.rows, by=["condition"], metrics=["task_score"], n=200)
    grp = agg.group_for(condition="chroma")
    assert grp is not None and grp.metrics["task_score"].mean == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# power_note — observed power + minimum detectable effect
# --------------------------------------------------------------------------- #
def test_power_note_basic_shape() -> None:
    pn = power_note(0.5, n=30)
    assert isinstance(pn, PowerNote)
    assert 0.0 < pn.power < 1.0
    assert math.isfinite(pn.mde) and pn.mde > 0.0
    assert pn.target_power == 0.80
    assert pn.note  # human-readable, non-empty


def test_power_note_more_samples_improve_power_and_mde() -> None:
    small = power_note(0.5, n=10)
    large = power_note(0.5, n=100)
    assert large.power > small.power  # more pairs → more power
    assert large.mde < small.mde  # more pairs → smaller detectable effect


def test_power_note_zero_effect_power_equals_alpha() -> None:
    """At a true effect of 0, power equals the type-I rate alpha."""
    pn = power_note(0.0, n=50, alpha=0.05)
    assert pn.power == pytest.approx(0.05, abs=1e-9)


def test_power_note_small_n_is_undefined() -> None:
    pn = power_note(0.5, n=1)
    assert math.isnan(pn.power)
    assert math.isinf(pn.mde)
    assert "undefined" in pn.note


# --------------------------------------------------------------------------- #
# multiple_comparison_note — Bonferroni only when ranking > 2 systems
# --------------------------------------------------------------------------- #
def test_multiple_comparison_note_no_correction_for_two() -> None:
    note = multiple_comparison_note(2)
    assert "Bonferroni" not in note
    assert "no multiple-comparison correction" in note


def test_multiple_comparison_note_bonferroni_for_three() -> None:
    note = multiple_comparison_note(3, alpha=0.05)
    assert "Bonferroni" in note
    assert "3 pairwise comparisons" in note  # 3*(3-1)/2 = 3
    assert f"{0.05 / 3:.4g}" in note  # corrected per-comparison alpha


def test_multiple_comparison_note_scales_pairwise_count() -> None:
    note = multiple_comparison_note(4, alpha=0.05)
    assert "6 pairwise comparisons" in note  # 4*3/2 = 6
    assert f"{0.05 / 6:.4g}" in note


# --------------------------------------------------------------------------- #
# Dataclass shapes — frozen, with the required fields
# --------------------------------------------------------------------------- #
def test_dataclass_shapes() -> None:
    pc = paired_compare("chroma", "no_memory", "task_score", _paired_rows(), n=200, seed=0)
    assert is_dataclass(pc) and isinstance(pc, PairedComparison)
    assert {"mean_diff", "cohens_dz", "ci", "n_pairs", "effect_magnitude"} <= {
        f.name for f in fields(pc)
    }
    assert is_dataclass(pc.ci) and isinstance(pc.ci, BootstrapCI)

    agg = aggregate(_paired_rows(), by=["condition"], metrics=["task_score"], n=200)
    grp = agg.groups[0]
    assert is_dataclass(grp) and isinstance(grp, GroupStats)
    assert is_dataclass(grp.metrics["task_score"]) and isinstance(
        grp.metrics["task_score"], MetricStats
    )
