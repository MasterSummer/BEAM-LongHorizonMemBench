"""TDD tests for the headline Memory ROI metric (lhmsb.metrics.roi, spec/02-metrics.md §1).

Written FIRST (RED) before the implementation. ROI is the counterfactual ratio
``mean(normalized_gain) / mean(memory_attributable_cost)`` per system per family
(and overall), reported WITH a bootstrap CI, a gain floor, the full cost-vector
breakdown, and a Pareto front — never a bare scalar.

The canonical math + edge cases live in ``spec/02-metrics.md`` §1:

  * ``gain        = score(system) − score(no_memory)`` (per episode, seed)
  * ``norm_gain   = clamp(gain / max(ε, max_achievable − score(no_memory)), −1, 1)``
  * ``cost        = scalarize(CostVector)`` with agent-loop tokens ZEROED (only the
    memory system's own cost is the ROI denominator).
  * ``ROI(system) = mean(norm_gain) / mean(cost)``

Edge-case policies (each asserted below):
  * ``no_memory`` → ROI = N/A (``roi is None``), NEVER reported as ``0``.
  * near-zero mean cost → ``undefined_lowcost=True`` + a FINITE placeholder ROI
    (``0.0``), NEVER ``+inf`` / ``NaN``.
  * negative gain → negative ROI (allowed, reported).
  * the cost vector is retained (per-field mean) in every result.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from lhmsb.cost import CostConfig, load_cost_config
from lhmsb.metrics.roi import (
    NO_MEMORY,
    OVERALL_FAMILY,
    PARETO_DOMINATED,
    PARETO_ON_FRONT,
    ROI_NA_BASELINE,
    ROI_OK,
    ROI_UNDEFINED_LOWCOST,
    RoiResult,
    bootstrap_ci,
    compute_roi,
    memory_attributable_cost,
    normalized_gain,
    pareto_front,
    scalarize_memory_cost,
)
from lhmsb.runner import ResultsTable, RunRow
from lhmsb.types import CostVector

# --------------------------------------------------------------------------- #
# Cost configuration: the declared sheet in configs/cost_weights.yaml
# (all token weights 1.0; 1 ms = 0.1 token-equiv; 1 KB = 0.01 token-equiv).
# --------------------------------------------------------------------------- #
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cost_weights.yaml"


@pytest.fixture
def cost_config() -> CostConfig:
    return load_cost_config(_CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _mem_cost(tokens: int, *, agent_in: int = 0, latency_ms: float = 0.0) -> CostVector:
    """A cost vector whose memory-attributable scalar equals ``tokens`` (+ latency).

    ``agent_in`` is agent-loop input tokens — recorded but NEVER part of the memory
    cost (ROI denominator). Setting it large proves agent tokens are excluded.
    """
    return CostVector(
        agent_input_tokens=agent_in,
        mem_internal_in_tokens=tokens,
        retrieval_latency_ms=latency_ms,
    )


def _row(
    *,
    condition: str,
    task_score: float,
    cost: CostVector,
    episode_id: str = "e1",
    family: str = "research",
    seed: int = 1,
    status: str = "completed",
) -> RunRow:
    """A fully-graded RunRow with ROI-irrelevant fields set to neutral defaults."""
    return RunRow(
        episode_id=episode_id,
        family=family,
        seed=seed,
        condition=condition,
        track="native",
        status=status,
        attempts=1,
        n_probes=2,
        world_event_hash="weh",
        episode_hash="eh",
        task_score=task_score,
        utilization_rate=None,
        improvement_over_time=None,
        judge_contribution=0.0,
        drift_index=0.0,
        drift_is_na=False,
        stale_fact_violations=0,
        constraint_violations=0,
        behavioral_flips=0,
        judge_fallback_share=0.0,
        cost=cost,
    )


# The spec §1.4 worked example: no_memory {.30,.45,.20}; system A {.70,.60,.90}.
_EPISODES = ("e1", "e2", "e3")
_NO_MEM_SCORES = {"e1": 0.30, "e2": 0.45, "e3": 0.20}
_SYS_A_SCORES = {"e1": 0.70, "e2": 0.60, "e3": 0.90}
_SYS_A_COSTS = {"e1": 5200, "e2": 4800, "e3": 5100}


def _positive_table() -> ResultsTable:
    """no_memory baseline + a clearly-beneficial system 'chroma' (3 episodes)."""
    rows: list[RunRow] = []
    for ep in _EPISODES:
        rows.append(
            _row(
                condition=NO_MEMORY, episode_id=ep, task_score=_NO_MEM_SCORES[ep], cost=_mem_cost(0)
            )
        )
        rows.append(
            _row(
                condition="chroma",
                episode_id=ep,
                task_score=_SYS_A_SCORES[ep],
                # Huge agent tokens that MUST be excluded from the memory cost.
                cost=_mem_cost(_SYS_A_COSTS[ep], agent_in=100_000),
            )
        )
    return ResultsTable(track="native", rows=rows)


def _by_condition(results: list[RoiResult], *, family: str = "research") -> dict[str, RoiResult]:
    return {r.condition: r for r in results if r.family == family}


# --------------------------------------------------------------------------- #
# normalized_gain — clamp + scale-invariance (prevents task-scale domination)
# --------------------------------------------------------------------------- #
def test_normalized_gain_hand_computed() -> None:
    assert normalized_gain(0.70, 0.30, 1.0) == pytest.approx(0.40 / 0.70)
    assert normalized_gain(0.60, 0.45, 1.0) == pytest.approx(0.15 / 0.55)
    assert normalized_gain(0.90, 0.20, 1.0) == pytest.approx(0.70 / 0.80)


def test_normalized_gain_clamps_to_unit_interval() -> None:
    # Raw gain 5.0 / 1.0 = 5.0 → clamped to +1.0.
    assert normalized_gain(6.0, 1.0, 1.0) == 1.0
    # Raw gain −0.9 / 0.1 = −9 → clamped to −1.0.
    assert normalized_gain(0.0, 0.9, 1.0) == -1.0


def test_normalized_gain_eps_guards_zero_denominator() -> None:
    """A fully-saturated baseline (max_achievable == no_memory) never divides by 0."""
    value = normalized_gain(0.5, 1.0, 1.0)  # denom = max(0.001, 0.0) = 0.001
    assert math.isfinite(value)
    assert value == -1.0  # −0.5 / 0.001 = −500 → clamp −1


def test_normalized_gain_is_scale_invariant() -> None:
    """A 90-point gain on a 0..100 task == a 0.9-point gain on a 0..1 task.

    This is the anti-domination property: a large RAW gain cannot dominate the
    aggregate simply because the task uses a larger numeric scale.
    """
    big = normalized_gain(90.0, 0.0, 100.0)
    small = normalized_gain(0.9, 0.0, 1.0)
    assert big == pytest.approx(small)
    assert big == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# memory_attributable_cost / scalarize_memory_cost — agent tokens excluded
# --------------------------------------------------------------------------- #
def test_memory_attributable_cost_zeros_only_agent_tokens() -> None:
    cv = CostVector(
        agent_input_tokens=1000,
        agent_output_tokens=500,
        mem_internal_in_tokens=300,
        mem_internal_out_tokens=200,
        embedding_tokens=400,
        retrieval_latency_ms=10.0,
        storage_bytes=2048,
    )
    attributable = memory_attributable_cost(cv)
    assert attributable.agent_input_tokens == 0
    assert attributable.agent_output_tokens == 0
    # Every other field is preserved verbatim.
    assert attributable.mem_internal_in_tokens == 300
    assert attributable.mem_internal_out_tokens == 200
    assert attributable.embedding_tokens == 400
    assert attributable.retrieval_latency_ms == 10.0
    assert attributable.storage_bytes == 2048
    # The input vector is not mutated (frozen dataclass).
    assert cv.agent_input_tokens == 1000


def test_scalarize_memory_cost_excludes_agent_tokens(cost_config: CostConfig) -> None:
    cv = CostVector(
        agent_input_tokens=1_000_000,  # must NOT count
        agent_output_tokens=1_000_000,  # must NOT count
        mem_internal_in_tokens=300,
        retrieval_latency_ms=10.0,  # 10 ms × 0.1 = 1.0 token-equiv
    )
    # 300 (tokens × 1.0) + 1.0 (latency) = 301.0 — agent tokens absent.
    assert scalarize_memory_cost(cv, cost_config) == pytest.approx(301.0)


# --------------------------------------------------------------------------- #
# bootstrap_ci — seeded, reproducible, brackets the point estimate
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_is_seeded_reproducible() -> None:
    values = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7]
    first = bootstrap_ci(values, n=2000, seed=42)
    second = bootstrap_ci(values, n=2000, seed=42)
    assert first == second


def test_bootstrap_ci_point_is_mean_and_brackets() -> None:
    values = [0.2, 0.4, 0.6, 0.8]
    point, lo, hi = bootstrap_ci(values, n=3000, seed=0)
    assert point == pytest.approx(0.5)  # mean of the sample
    assert lo <= point <= hi
    assert math.isfinite(lo) and math.isfinite(hi)


def test_bootstrap_ci_constant_values_collapse() -> None:
    point, lo, hi = bootstrap_ci([0.5, 0.5, 0.5], n=500, seed=1)
    assert point == lo == hi == 0.5


# --------------------------------------------------------------------------- #
# pareto_front — domination semantics (anti-gaming context)
# --------------------------------------------------------------------------- #
def test_pareto_front_identifies_dominated_system() -> None:
    # C dominates both A (higher perf, lower cost) and B; A also dominates B.
    systems = [("C", 0.65, 90.0), ("A", 0.60, 100.0), ("B", 0.30, 200.0)]
    assert pareto_front(systems) == {"C"}


def test_pareto_front_keeps_tradeoff_systems() -> None:
    # Cheap-but-weak vs expensive-but-strong: neither dominates → both on front.
    systems = [("cheap", 0.05, 50.0), ("rich", 0.60, 5000.0)]
    assert pareto_front(systems) == {"cheap", "rich"}


def test_pareto_front_dominated_among_three() -> None:
    systems = [("cheap", 0.05, 50.0), ("mid", 0.10, 40.0), ("rich", 0.60, 5000.0)]
    # 'mid' dominates 'cheap' (higher perf, lower cost); 'rich' is the high end.
    assert pareto_front(systems) == {"mid", "rich"}


def test_pareto_front_single_system() -> None:
    assert pareto_front([("solo", 0.5, 100.0)]) == {"solo"}


# --------------------------------------------------------------------------- #
# compute_roi — positive ROI (the headline happy path)
# --------------------------------------------------------------------------- #
def test_compute_roi_positive_case(cost_config: CostConfig) -> None:
    results = compute_roi(_positive_table(), cost_config, bootstrap_n=2000)
    by_cond = _by_condition(results)
    chroma = by_cond["chroma"]

    expected_gain = (0.40 / 0.70 + 0.15 / 0.55 + 0.70 / 0.80) / 3
    expected_cost = (5200 + 4800 + 5100) / 3

    assert chroma.mean_normalized_gain == pytest.approx(expected_gain)
    assert chroma.mean_cost == pytest.approx(expected_cost)  # agent tokens excluded
    assert chroma.roi == pytest.approx(expected_gain / expected_cost)
    assert chroma.roi is not None and chroma.roi > 0.0
    assert chroma.roi_status == ROI_OK
    assert chroma.undefined_lowcost is False
    assert chroma.is_baseline is False
    assert chroma.n_episodes == 3
    # All gains positive → gain floor equals the mean normalized gain.
    assert chroma.gain_floor == pytest.approx(expected_gain)
    assert chroma.below_gain_floor is False
    # CI brackets the point estimate and is finite (never inf/nan).
    assert chroma.ci_low is not None and chroma.ci_high is not None
    assert chroma.ci_low <= chroma.roi <= chroma.ci_high
    assert math.isfinite(chroma.ci_low) and math.isfinite(chroma.ci_high)
    # The single system is on the Pareto front.
    assert chroma.pareto_status == PARETO_ON_FRONT


def test_compute_roi_retains_cost_vector_breakdown(cost_config: CostConfig) -> None:
    results = compute_roi(_positive_table(), cost_config, bootstrap_n=500)
    chroma = _by_condition(results)["chroma"]
    breakdown = chroma.cost_vector_breakdown
    # Full 12-field vector retained, not collapsed.
    cost_field_names = {f.name for f in fields(CostVector)}
    assert set(breakdown) == cost_field_names
    # Agent tokens are zeroed in the memory-attributable breakdown...
    assert breakdown["agent_input_tokens"] == 0.0
    # ...while the memory-internal tokens carry the mean cost.
    assert breakdown["mem_internal_in_tokens"] == pytest.approx((5200 + 4800 + 5100) / 3)


# --------------------------------------------------------------------------- #
# compute_roi — no_memory control → N/A (NOT 0)
# --------------------------------------------------------------------------- #
def test_no_memory_roi_is_not_reported(cost_config: CostConfig) -> None:
    results = compute_roi(_positive_table(), cost_config, bootstrap_n=500)
    no_mem = _by_condition(results)[NO_MEMORY]
    assert no_mem.is_baseline is True
    assert no_mem.roi is None  # N/A — explicitly NOT 0.0
    assert no_mem.roi != 0.0
    assert no_mem.roi_status == ROI_NA_BASELINE
    assert no_mem.ci_low is None and no_mem.ci_high is None
    # No non-baseline result silently reports the control as a real ROI of 0.
    for r in results:
        if r.condition == NO_MEMORY:
            assert r.is_baseline is True


# --------------------------------------------------------------------------- #
# compute_roi — negative gain → negative ROI (allowed, reported)
# --------------------------------------------------------------------------- #
def test_negative_gain_produces_negative_roi(cost_config: CostConfig) -> None:
    rows: list[RunRow] = []
    bad_scores = {"e1": 0.10, "e2": 0.20, "e3": 0.05}
    for ep in _EPISODES:
        rows.append(
            _row(
                condition=NO_MEMORY, episode_id=ep, task_score=_NO_MEM_SCORES[ep], cost=_mem_cost(0)
            )
        )
        rows.append(
            _row(
                condition="bad_mem", episode_id=ep, task_score=bad_scores[ep], cost=_mem_cost(3000)
            )
        )
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=2000)
    bad = _by_condition(results)["bad_mem"]

    expected_gain = (-0.20 / 0.70 + -0.25 / 0.55 + -0.15 / 0.80) / 3
    assert bad.mean_normalized_gain == pytest.approx(expected_gain)
    assert bad.mean_normalized_gain < 0.0
    assert bad.roi is not None and bad.roi < 0.0
    assert bad.roi == pytest.approx(expected_gain / 3000.0)
    assert not math.isinf(bad.roi) and not math.isnan(bad.roi)
    # A net-harmful system is flagged below the gain floor; its gain floor is 0.
    assert bad.gain_floor == 0.0
    assert bad.below_gain_floor is True
    assert bad.roi_status == ROI_OK


# --------------------------------------------------------------------------- #
# compute_roi — near-zero cost → undefined_lowcost (NEVER +inf / NaN)
# --------------------------------------------------------------------------- #
def test_zero_cost_returns_defined_policy(cost_config: CostConfig) -> None:
    """A beneficial system with ZERO memory cost would be +inf — must be flagged instead."""
    rows: list[RunRow] = []
    for ep in _EPISODES:
        rows.append(
            _row(
                condition=NO_MEMORY, episode_id=ep, task_score=_NO_MEM_SCORES[ep], cost=_mem_cost(0)
            )
        )
        rows.append(
            _row(
                condition="free_mem",
                episode_id=ep,
                task_score=_SYS_A_SCORES[ep],
                # Only agent tokens → memory-attributable cost is exactly 0.0.
                cost=_mem_cost(0, agent_in=50_000),
            )
        )
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=500)
    free = _by_condition(results)["free_mem"]

    assert free.undefined_lowcost is True
    assert free.roi_status == ROI_UNDEFINED_LOWCOST
    assert free.mean_cost == pytest.approx(0.0)
    # The ROI placeholder is FINITE — never +inf, never NaN.
    assert free.roi == 0.0
    assert not math.isinf(free.roi)
    assert not math.isnan(free.roi)
    # The gain is still reported (the system DID help) and the cost vector retained.
    assert free.mean_normalized_gain > 0.0
    assert "mem_internal_in_tokens" in free.cost_vector_breakdown


def test_low_cost_threshold_is_configurable(cost_config: CostConfig) -> None:
    """A small (but non-zero) mean cost below the threshold is also undefined_lowcost."""
    rows: list[RunRow] = []
    for ep in _EPISODES:
        rows.append(
            _row(
                condition=NO_MEMORY, episode_id=ep, task_score=_NO_MEM_SCORES[ep], cost=_mem_cost(0)
            )
        )
        rows.append(
            _row(
                condition="cheapish", episode_id=ep, task_score=_SYS_A_SCORES[ep], cost=_mem_cost(5)
            )
        )
    table = ResultsTable(track="native", rows=rows)
    # Threshold 10 > mean cost 5 → undefined_lowcost; never +inf.
    high = _by_condition(compute_roi(table, cost_config, cost_threshold=10.0, bootstrap_n=200))
    assert high["cheapish"].undefined_lowcost is True
    assert high["cheapish"].roi == 0.0
    # Threshold 1 < mean cost 5 → a normal, finite ROI.
    low = _by_condition(compute_roi(table, cost_config, cost_threshold=1.0, bootstrap_n=200))
    assert low["cheapish"].undefined_lowcost is False
    assert low["cheapish"].roi is not None and math.isfinite(low["cheapish"].roi)


# --------------------------------------------------------------------------- #
# compute_roi — anti-gaming: tiny-gain/tiny-cost cannot "win" by scalar alone
# --------------------------------------------------------------------------- #
def test_pareto_and_gain_floor_prevent_tiny_gain_from_winning(cost_config: CostConfig) -> None:
    rows: list[RunRow] = []
    for ep in ("e1", "e2"):
        rows.append(_row(condition=NO_MEMORY, episode_id=ep, task_score=0.0, cost=_mem_cost(0)))
        # tiny gain, tiny (but defined) cost → a HUGE scalar ROI.
        rows.append(_row(condition="tiny", episode_id=ep, task_score=0.05, cost=_mem_cost(50)))
        # large gain, large cost → a SMALLER scalar ROI.
        rows.append(_row(condition="rich", episode_id=ep, task_score=0.60, cost=_mem_cost(5000)))
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=500)
    by_cond = _by_condition(results)
    tiny, rich = by_cond["tiny"], by_cond["rich"]

    assert tiny.roi is not None and rich.roi is not None
    # 'tiny' wins on the bare scalar ROI...
    assert tiny.roi > rich.roi
    # ...but its gain floor exposes that it barely helped (anti-gaming signal #1).
    assert rich.gain_floor > tiny.gain_floor
    assert rich.gain_floor == pytest.approx(0.60)
    assert tiny.gain_floor == pytest.approx(0.05)
    # ...and the Pareto front keeps BOTH (a tradeoff) so 'tiny' is not the sole
    # winner (anti-gaming signal #2): neither dominates the other.
    assert tiny.pareto_status == PARETO_ON_FRONT
    assert rich.pareto_status == PARETO_ON_FRONT


def test_dominated_system_flagged_in_compute_roi(cost_config: CostConfig) -> None:
    """A system dominated on (gain, cost) is marked PARETO_DOMINATED by compute_roi."""
    rows: list[RunRow] = []
    for ep in ("e1", "e2"):
        rows.append(_row(condition=NO_MEMORY, episode_id=ep, task_score=0.0, cost=_mem_cost(0)))
        # 'good' has higher gain AND lower cost than 'weak' → 'weak' is dominated.
        rows.append(_row(condition="good", episode_id=ep, task_score=0.60, cost=_mem_cost(1000)))
        rows.append(_row(condition="weak", episode_id=ep, task_score=0.30, cost=_mem_cost(2000)))
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=500)
    by_cond = _by_condition(results)
    assert by_cond["good"].pareto_status == PARETO_ON_FRONT
    assert by_cond["weak"].pareto_status == PARETO_DOMINATED


# --------------------------------------------------------------------------- #
# compute_roi — overall (cross-family) aggregation
# --------------------------------------------------------------------------- #
def test_compute_roi_overall_pools_all_families(cost_config: CostConfig) -> None:
    rows: list[RunRow] = []
    for family, sys_score in (("research", 0.70), ("software", 0.50)):
        rows.append(
            _row(
                condition=NO_MEMORY,
                family=family,
                episode_id=f"{family}-e1",
                task_score=0.20,
                cost=_mem_cost(0),
            )
        )
        rows.append(
            _row(
                condition="chroma",
                family=family,
                episode_id=f"{family}-e1",
                task_score=sys_score,
                cost=_mem_cost(1000),
            )
        )
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=500)

    # Per-family results exist for both families.
    assert _by_condition(results, family="research")["chroma"].n_episodes == 1
    assert _by_condition(results, family="software")["chroma"].n_episodes == 1

    # An OVERALL result pools both families' paired cells (2 episodes).
    overall = _by_condition(results, family=OVERALL_FAMILY)
    assert "chroma" in overall
    chroma_all = overall["chroma"]
    assert chroma_all.n_episodes == 2
    research_gain = normalized_gain(0.70, 0.20, 1.0)
    software_gain = normalized_gain(0.50, 0.20, 1.0)
    assert chroma_all.mean_normalized_gain == pytest.approx((research_gain + software_gain) / 2)
    # The overall section also carries the no_memory baseline as N/A.
    assert overall[NO_MEMORY].roi is None
    assert overall[NO_MEMORY].is_baseline is True


# --------------------------------------------------------------------------- #
# compute_roi — failed no_memory baseline excludes that pair (spec §1.2)
# --------------------------------------------------------------------------- #
def test_failed_no_memory_baseline_excludes_pair(cost_config: CostConfig) -> None:
    rows = [
        # e1: usable baseline → paired.
        _row(condition=NO_MEMORY, episode_id="e1", task_score=0.20, cost=_mem_cost(0)),
        _row(condition="chroma", episode_id="e1", task_score=0.80, cost=_mem_cost(1000)),
        # e2: the no_memory baseline CRASHED → the pair is excluded from ROI.
        _row(
            condition=NO_MEMORY, episode_id="e2", task_score=0.0, cost=_mem_cost(0), status="failed"
        ),
        _row(condition="chroma", episode_id="e2", task_score=0.90, cost=_mem_cost(1000)),
    ]
    results = compute_roi(ResultsTable(track="native", rows=rows), cost_config, bootstrap_n=500)
    chroma = _by_condition(results)["chroma"]
    # Only the e1 pair contributes; e2 (failed baseline) is dropped from the ratio.
    assert chroma.n_episodes == 1
    assert chroma.mean_normalized_gain == pytest.approx(normalized_gain(0.80, 0.20, 1.0))


# --------------------------------------------------------------------------- #
# RoiResult shape — frozen dataclass with the required fields
# --------------------------------------------------------------------------- #
def test_roi_result_shape(cost_config: CostConfig) -> None:
    chroma = _by_condition(compute_roi(_positive_table(), cost_config, bootstrap_n=200))["chroma"]
    assert is_dataclass(chroma) and isinstance(chroma, RoiResult)
    names = {f.name for f in fields(chroma)}
    assert {
        "condition",
        "family",
        "n_episodes",
        "mean_normalized_gain",
        "mean_cost",
        "roi",
        "ci_low",
        "ci_high",
        "gain_floor",
        "pareto_status",
        "cost_vector_breakdown",
        "undefined_lowcost",
    } <= names
