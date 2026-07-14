"""Memory ROI — the headline metric (spec/02-metrics.md §1, Dim 7 efficiency).

Memory ROI answers *did the memory system improve performance, and at what cost?*
It is a **counterfactual ratio** computed per system per family (and overall) from
the runner's :class:`~lhmsb.runner.results.ResultsTable`, never a bare leaderboard
number — every scalar is paired with a bootstrap CI, a gain floor, the full
cost-vector breakdown, and a Pareto-front status.

For each ``(episode, seed)`` cell under condition (system) ``c`` (spec §1.1)::

    gain(c)         = task_score(c) − task_score(no_memory)
    normalized_gain = clamp(gain / max(ε, max_achievable − task_score(no_memory)), −1, 1)
    cost(c)         = scalarize(memory_attributable_cost(CostVector(c)))
    ROI(c)          = mean(normalized_gain) / mean(cost)

The cost denominator is **memory-attributable only**: the agent-loop tokens
(``agent_input_tokens`` / ``agent_output_tokens``) are zeroed before scalarizing
because they are not a cost of the memory system (they are paid under every
condition, including ``no_memory``). Everything else in the 12-field
:class:`~lhmsb.types.CostVector` — the memory system's internal LLM tokens,
embeddings, storage, latency, reflection — IS the memory cost.

Edge-case policies (spec §1.2 — all enforced here, NEVER violated):

  * **no_memory control** → ROI = N/A (:pyattr:`RoiResult.roi` is ``None``,
    :pyattr:`RoiResult.is_baseline` is ``True``). It is the baseline used to compute
    every other system's gain; its own ROI is undefined by construction and is NEVER
    reported as ``0``.
  * **near-zero cost** (``mean_cost ≤ cost_threshold``) → :pyattr:`undefined_lowcost`
    is ``True`` and ``roi`` is a FINITE placeholder (``0.0``); it is NEVER ``+inf`` or
    ``NaN``. The gain and the cost-vector breakdown are still reported.
  * **negative gain** → negative ROI (allowed and reported — the memory system
    harmed performance). Always paired with the gain floor and Pareto context so a
    negative-ROI system is never misread as beneficial.

This module also exposes a small seeded :func:`bootstrap_ci` (task 23 will
centralize/extend it) and a :func:`pareto_front` so a tiny-gain/tiny-cost system
cannot "win" on the scalar ROI alone.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from lhmsb.cost import CostConfig, scalarize
from lhmsb.types import CostVector

if TYPE_CHECKING:  # avoid a runtime cycle: runner.__init__ → orchestrator → metrics → roi
    from lhmsb.runner.results import ResultsTable, RunRow

__all__ = [
    "DEFAULT_BOOTSTRAP_N",
    "DEFAULT_COST_THRESHOLD",
    "DEFAULT_EPS",
    "NO_MEMORY",
    "OVERALL_FAMILY",
    "PARETO_DOMINATED",
    "PARETO_NA",
    "PARETO_ON_FRONT",
    "ROI_NA_BASELINE",
    "ROI_NA_NO_PAIRS",
    "ROI_OK",
    "ROI_UNDEFINED_LOWCOST",
    "RoiResult",
    "bootstrap_ci",
    "compute_roi",
    "memory_attributable_cost",
    "normalized_gain",
    "pareto_front",
    "scalarize_memory_cost",
]

#: The baseline control condition (ROI = N/A by construction; spec §1.2).
NO_MEMORY = "no_memory"
#: Sentinel ``family`` for the cross-family aggregate produced by :func:`compute_roi`.
OVERALL_FAMILY = "__overall__"

#: ``ε`` in the normalized-gain denominator — prevents division by zero when the
#: baseline already saturates the task (spec §1.1).
DEFAULT_EPS = 0.001
#: ``ε_cost`` — a mean memory cost at or below this (token-equivalents per episode)
#: is flagged ``undefined_lowcost`` rather than producing a ``+inf`` ROI (spec §1.2).
DEFAULT_COST_THRESHOLD = 1.0
#: Bootstrap resamples for the ROI CI (spec §1.1: 95%, seeded, n=10,000).
DEFAULT_BOOTSTRAP_N = 10_000
#: Bootstrap two-sided significance level (95% CI).
DEFAULT_ALPHA = 0.05
#: A system whose mean normalized gain is below this floor is flagged net-harmful.
DEFAULT_GAIN_FLOOR_THRESHOLD = 0.0

# ROI status tags (a defined value for every policy path — never inf / NaN).
ROI_OK = "ok"
ROI_NA_BASELINE = "na_baseline"
ROI_NA_NO_PAIRS = "na_no_pairs"
ROI_UNDEFINED_LOWCOST = "undefined_lowcost"

# Pareto-front membership tags.
PARETO_ON_FRONT = "on_front"
PARETO_DOMINATED = "dominated"
PARETO_NA = "n/a"

# The 12 CostVector field names (canonical order), derived from the dataclass itself
# so this never drifts from lhmsb.types.CostVector.
_COST_FIELD_NAMES: tuple[str, ...] = tuple(asdict(CostVector()).keys())


@dataclass(frozen=True)
class RoiResult:
    """Per-(family, condition) Memory ROI verdict (spec/02-metrics.md §1).

    Attributes:
        condition: The system/condition name (e.g. ``"chroma"``, ``"no_memory"``).
        family: The task family, or :data:`OVERALL_FAMILY` for the cross-family row.
        n_episodes: Number of paired ``(episode, seed)`` cells that contributed
            (cells whose ``no_memory`` baseline crashed are excluded — spec §1.2).
        mean_normalized_gain: Mean of the clamped normalized gains ``∈ [-1, 1]``.
        mean_cost: Mean memory-attributable scalar cost (tokens-equivalent).
        roi: ``mean_normalized_gain / mean_cost``, or ``None`` (N/A) for the
            ``no_memory`` baseline / when there are no paired cells. Finite ``0.0``
            placeholder when :pyattr:`undefined_lowcost`. NEVER ``+inf`` / ``NaN``.
        ci_low: Lower bound of the seeded bootstrap CI on the ROI, or ``None``.
        ci_high: Upper bound of the seeded bootstrap CI on the ROI, or ``None``.
        gain_floor: ``mean(max(0, normalized_gain))`` (spec §1.1) — the non-negative
            gain contribution, distinguishing "not harmful" from "truly beneficial".
        pareto_status: :data:`PARETO_ON_FRONT` / :data:`PARETO_DOMINATED` /
            :data:`PARETO_NA` on the ``(mean_normalized_gain, mean_cost)`` axes.
        roi_status: One of :data:`ROI_OK`, :data:`ROI_NA_BASELINE`,
            :data:`ROI_NA_NO_PAIRS`, :data:`ROI_UNDEFINED_LOWCOST`.
        cost_vector_breakdown: Per-field mean of the memory-attributable
            :class:`~lhmsb.types.CostVector` (all 12 fields retained, never collapsed).
        undefined_lowcost: ``True`` when the mean cost is at/below the threshold.
        is_baseline: ``True`` for the ``no_memory`` control row.
        below_gain_floor: ``True`` when ``mean_normalized_gain`` is below the
            configured gain-floor threshold (a net-harmful system).
    """

    condition: str
    family: str
    n_episodes: int
    mean_normalized_gain: float
    mean_cost: float
    roi: float | None
    ci_low: float | None
    ci_high: float | None
    gain_floor: float
    pareto_status: str
    roi_status: str
    cost_vector_breakdown: dict[str, float]
    undefined_lowcost: bool = False
    is_baseline: bool = False
    below_gain_floor: bool = False


# --------------------------------------------------------------------------- #
# Cost attribution: the ROI denominator is the MEMORY system's cost only
# --------------------------------------------------------------------------- #
def memory_attributable_cost(cost_vector: CostVector) -> CostVector:
    """Return a copy with the agent-loop tokens zeroed (spec §1.3).

    ``agent_input_tokens`` / ``agent_output_tokens`` are the agent loop's cost — paid
    under EVERY condition (including ``no_memory``), so they are not attributable to
    the memory system and must not enter the ROI denominator. Every other field (the
    memory system's internal LLM tokens, embeddings, storage, latency, reflection) is
    retained verbatim. The input is not mutated (``CostVector`` is frozen).
    """
    return replace(cost_vector, agent_input_tokens=0, agent_output_tokens=0)


def scalarize_memory_cost(cost_vector: CostVector, cost_config: CostConfig) -> float:
    """Scalarize ONLY the memory-attributable cost to a tokens-equivalent (spec §1.1).

    Equivalent to :func:`lhmsb.cost.scalarize` applied to
    :func:`memory_attributable_cost` with the declared weights + conversion sheet.
    """
    attributable = memory_attributable_cost(cost_vector)
    return scalarize(attributable, cost_config.weights, cost_config.conversion)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_gain(
    system_score: float,
    no_memory_score: float,
    max_achievable: float,
    eps: float = DEFAULT_EPS,
) -> float:
    """``clamp(gain / max(eps, max_achievable − no_memory_score), −1, 1)`` (spec §1.1).

    Bounding the per-episode gain to ``[-1, 1]`` makes it **scale-invariant**: a large
    raw gain on a high-ceiling task cannot dominate the aggregate merely because the
    task uses a larger numeric scale. ``eps`` guards the denominator when the baseline
    already saturates the task (``max_achievable ≈ no_memory_score``).
    """
    gain = system_score - no_memory_score
    denominator = max(eps, max_achievable - no_memory_score)
    return _clamp(gain / denominator, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Bootstrap CI (small + seeded; task 23 centralizes/extends it)
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _statistic_fn(name: str) -> Callable[[Sequence[float]], float]:
    if name == "mean":
        return _mean
    if name == "median":
        return _median
    if name == "sum":
        return lambda values: float(sum(values))
    raise ValueError(f"unknown statistic: {name!r} (expected 'mean', 'median', or 'sum')")


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Seeded percentile bootstrap CI: returns ``(point, lo, hi)``.

    ``point`` is the statistic on the full sample; ``lo``/``hi`` are the
    ``alpha/2`` / ``1 − alpha/2`` percentiles of ``n`` resampled statistics, drawn
    with replacement using ``random.Random(seed)`` (fully reproducible). A single /
    constant sample collapses the interval to the point. Raises on empty input.
    """
    data = [float(value) for value in values]
    if not data:
        raise ValueError("bootstrap_ci requires at least one value")
    stat = _statistic_fn(statistic)
    point = stat(data)
    if len(data) == 1:
        return point, point, point

    rng = random.Random(seed)
    size = len(data)
    samples = sorted(stat(rng.choices(data, k=size)) for _ in range(n))
    lo_index = _percentile_index(alpha / 2.0, n)
    hi_index = _percentile_index(1.0 - alpha / 2.0, n)
    return point, samples[lo_index], samples[hi_index]


def _percentile_index(quantile: float, n: int) -> int:
    """Index into a length-``n`` sorted list for ``quantile``, clamped to ``[0, n-1]``."""
    return max(0, min(n - 1, int(quantile * n)))


# --------------------------------------------------------------------------- #
# Pareto front (anti-gaming context)
# --------------------------------------------------------------------------- #
def pareto_front(systems: list[tuple[str, float, float]]) -> set[str]:
    """Return the names on the Pareto front of ``(name, performance, cost)`` triples.

    A system is **on the front** iff no OTHER system has both *strictly higher
    performance* AND *lower-or-equal cost* (spec §5.2). Equivalently, a system is
    dominated when some other system beats it on performance without costing more.
    Pure tradeoffs (cheaper-but-weaker vs dearer-but-stronger) both stay on the front,
    so a tiny-gain/tiny-cost system cannot be declared the sole winner.
    """
    front: set[str] = set()
    for name, performance, cost in systems:
        dominated = any(
            other_perf > performance and other_cost <= cost
            for other_name, other_perf, other_cost in systems
            if other_name != name
        )
        if not dominated:
            front.add(name)
    return front


# --------------------------------------------------------------------------- #
# compute_roi: the entry point over a runner ResultsTable
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Cell:
    """One paired ``(system, no_memory)`` cell: its normalized gain + memory cost."""

    norm_gain: float
    scalar_cost: float
    cost_vector: CostVector


def _zero_breakdown() -> dict[str, float]:
    return dict.fromkeys(_COST_FIELD_NAMES, 0.0)


def _mean_breakdown(cost_vectors: Sequence[CostVector]) -> dict[str, float]:
    """Per-field mean of memory-attributable cost vectors (all 12 fields retained).

    Fields are summed via ``CostVector.__add__`` then divided, enumerated explicitly
    (as in :meth:`CostVector.__add__` / :func:`lhmsb.cost.scalarize`) so every value
    stays typed — no ``Any`` leaks from a reflective ``asdict``. The keys are validated
    against the dataclass fields by the test suite.
    """
    if not cost_vectors:
        return _zero_breakdown()
    total = CostVector()
    for cost_vector in cost_vectors:
        total = total + cost_vector
    count = len(cost_vectors)
    return {
        "agent_input_tokens": total.agent_input_tokens / count,
        "agent_output_tokens": total.agent_output_tokens / count,
        "mem_internal_in_tokens": total.mem_internal_in_tokens / count,
        "mem_internal_out_tokens": total.mem_internal_out_tokens / count,
        "embedding_tokens": total.embedding_tokens / count,
        "embedding_calls": total.embedding_calls / count,
        "storage_bytes": total.storage_bytes / count,
        "retrieval_latency_ms": total.retrieval_latency_ms / count,
        "write_latency_ms": total.write_latency_ms / count,
        "update_latency_ms": total.update_latency_ms / count,
        "reflection_tokens": total.reflection_tokens / count,
        "num_retrieval_calls": total.num_retrieval_calls / count,
    }


def _baseline_scores(rows: Sequence[RunRow]) -> dict[tuple[str, str, int], float]:
    """Usable ``no_memory`` task scores keyed by ``(family, episode_id, seed)``.

    Only ``status == "completed"`` baselines are usable; a crashed/timed-out baseline
    yields no key, so the matching system cells are excluded from gain (spec §1.2).
    """
    scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row.condition == NO_MEMORY and row.status == "completed":
            scores[(row.family, row.episode_id, row.seed)] = row.task_score
    return scores


def _baseline_result(
    family: str, condition: str, cells: Sequence[RunRow], cost_config: CostConfig
) -> RoiResult:
    """The ``no_memory`` baseline row: ROI = N/A (never 0), its own cost still reported."""
    cost_vectors = [memory_attributable_cost(row.cost) for row in cells]
    mean_cost = (
        _mean([scalarize(cv, cost_config.weights, cost_config.conversion) for cv in cost_vectors])
        if cost_vectors
        else 0.0
    )
    return RoiResult(
        condition=condition,
        family=family,
        n_episodes=len(cells),
        mean_normalized_gain=0.0,
        mean_cost=mean_cost,
        roi=None,
        ci_low=None,
        ci_high=None,
        gain_floor=0.0,
        pareto_status=PARETO_NA,
        roi_status=ROI_NA_BASELINE,
        cost_vector_breakdown=_mean_breakdown(cost_vectors),
        undefined_lowcost=False,
        is_baseline=True,
        below_gain_floor=False,
    )


def compute_roi(
    table: ResultsTable,
    cost_config: CostConfig,
    *,
    max_achievable: float = 1.0,
    eps: float = DEFAULT_EPS,
    cost_threshold: float = DEFAULT_COST_THRESHOLD,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    gain_floor_threshold: float = DEFAULT_GAIN_FLOOR_THRESHOLD,
) -> list[RoiResult]:
    """Compute per-(family, condition) and overall Memory ROI from a results table.

    The ``no_memory`` control is the counterfactual baseline (its own ROI is N/A); every
    other condition's ROI is ``mean(normalized_gain) / mean(memory_cost)`` over the
    paired ``(episode, seed)`` cells, with a seeded bootstrap CI, a gain floor, the full
    cost-vector breakdown, and a Pareto status. An overall row per condition pools all
    families. No path ever emits ``+inf`` or ``NaN`` (spec §1.2).
    """
    rows = list(table.rows)
    baseline = _baseline_scores(rows)

    results: list[RoiResult] = []
    families = sorted({row.family for row in rows})
    for index, family in enumerate(families):
        family_rows = [row for row in rows if row.family == family]
        results.extend(
            _family_results(
                family,
                family_rows,
                baseline,
                cost_config,
                max_achievable=max_achievable,
                eps=eps,
                cost_threshold=cost_threshold,
                bootstrap_n=bootstrap_n,
                alpha=alpha,
                seed=seed + index,
                gain_floor_threshold=gain_floor_threshold,
            )
        )

    results.extend(
        _family_results(
            OVERALL_FAMILY,
            rows,
            baseline,
            cost_config,
            max_achievable=max_achievable,
            eps=eps,
            cost_threshold=cost_threshold,
            bootstrap_n=bootstrap_n,
            alpha=alpha,
            seed=seed + len(families),
            gain_floor_threshold=gain_floor_threshold,
        )
    )
    return results


def _family_results(
    family: str,
    rows: Sequence[RunRow],
    baseline: dict[tuple[str, str, int], float],
    cost_config: CostConfig,
    *,
    max_achievable: float,
    eps: float,
    cost_threshold: float,
    bootstrap_n: int,
    alpha: float,
    seed: int,
    gain_floor_threshold: float,
) -> list[RoiResult]:
    """All condition rows for one family (or the overall pool), with Pareto assigned."""
    conditions = sorted({row.condition for row in rows}, key=_condition_sort_key)
    built: list[RoiResult] = []
    for offset, condition in enumerate(conditions):
        cells = [row for row in rows if row.condition == condition]
        if condition == NO_MEMORY:
            built.append(_baseline_result(family, condition, cells, cost_config))
            continue
        built.append(
            _system_result(
                family,
                condition,
                cells,
                baseline,
                cost_config,
                max_achievable=max_achievable,
                eps=eps,
                cost_threshold=cost_threshold,
                bootstrap_n=bootstrap_n,
                alpha=alpha,
                seed=seed + offset,
                gain_floor_threshold=gain_floor_threshold,
            )
        )
    return _assign_pareto(built)


def _system_result(
    family: str,
    condition: str,
    cells: Sequence[RunRow],
    baseline: dict[tuple[str, str, int], float],
    cost_config: CostConfig,
    *,
    max_achievable: float,
    eps: float,
    cost_threshold: float,
    bootstrap_n: int,
    alpha: float,
    seed: int,
    gain_floor_threshold: float,
) -> RoiResult:
    """ROI for one non-baseline condition over its paired cells (spec §1.1-1.2)."""
    paired = _paired_cells(cells, baseline, cost_config, max_achievable=max_achievable, eps=eps)
    if not paired:
        return RoiResult(
            condition=condition,
            family=family,
            n_episodes=0,
            mean_normalized_gain=0.0,
            mean_cost=0.0,
            roi=None,
            ci_low=None,
            ci_high=None,
            gain_floor=0.0,
            pareto_status=PARETO_NA,
            roi_status=ROI_NA_NO_PAIRS,
            cost_vector_breakdown=_zero_breakdown(),
            undefined_lowcost=False,
            is_baseline=False,
            below_gain_floor=False,
        )

    norm_gains = [cell.norm_gain for cell in paired]
    mean_gain = _mean(norm_gains)
    mean_cost = _mean([cell.scalar_cost for cell in paired])
    gain_floor = _mean([max(0.0, gain) for gain in norm_gains])
    breakdown = _mean_breakdown([cell.cost_vector for cell in paired])

    if mean_cost <= cost_threshold:
        # Near-zero cost: a finite placeholder ROI (0.0), NEVER +inf / NaN (spec §1.2).
        roi: float | None = 0.0
        ci_low: float | None = None
        ci_high: float | None = None
        roi_status = ROI_UNDEFINED_LOWCOST
        undefined_lowcost = True
    else:
        point, gain_lo, gain_hi = bootstrap_ci(norm_gains, "mean", bootstrap_n, alpha, seed)
        roi = point / mean_cost
        ci_low = gain_lo / mean_cost
        ci_high = gain_hi / mean_cost
        roi_status = ROI_OK
        undefined_lowcost = False

    return RoiResult(
        condition=condition,
        family=family,
        n_episodes=len(paired),
        mean_normalized_gain=mean_gain,
        mean_cost=mean_cost,
        roi=roi,
        ci_low=ci_low,
        ci_high=ci_high,
        gain_floor=gain_floor,
        pareto_status=PARETO_NA,  # assigned by _assign_pareto over the whole group
        roi_status=roi_status,
        cost_vector_breakdown=breakdown,
        undefined_lowcost=undefined_lowcost,
        is_baseline=False,
        below_gain_floor=mean_gain < gain_floor_threshold,
    )


def _paired_cells(
    cells: Sequence[RunRow],
    baseline: dict[tuple[str, str, int], float],
    cost_config: CostConfig,
    *,
    max_achievable: float,
    eps: float,
) -> list[_Cell]:
    """The cells with a usable ``no_memory`` baseline, as normalized gain + memory cost."""
    paired: list[_Cell] = []
    for row in cells:
        key = (row.family, row.episode_id, row.seed)
        if key not in baseline:
            continue  # no usable baseline for this (episode, seed) → excluded (spec §1.2)
        gain = normalized_gain(row.task_score, baseline[key], max_achievable, eps)
        cost_vector = memory_attributable_cost(row.cost)
        scalar_cost = scalarize(cost_vector, cost_config.weights, cost_config.conversion)
        paired.append(_Cell(norm_gain=gain, scalar_cost=scalar_cost, cost_vector=cost_vector))
    return paired


def _assign_pareto(results: list[RoiResult]) -> list[RoiResult]:
    """Stamp each non-baseline result with its Pareto status over ``(gain, cost)``."""
    candidates = [
        (result.condition, result.mean_normalized_gain, result.mean_cost)
        for result in results
        if not result.is_baseline and result.roi_status != ROI_NA_NO_PAIRS
    ]
    if not candidates:
        return results
    front = pareto_front(candidates)
    stamped: list[RoiResult] = []
    for result in results:
        if result.is_baseline or result.roi_status == ROI_NA_NO_PAIRS:
            stamped.append(result)
            continue
        status = PARETO_ON_FRONT if result.condition in front else PARETO_DOMINATED
        stamped.append(replace(result, pareto_status=status))
    return stamped


def _condition_sort_key(condition: str) -> tuple[int, str]:
    """Deterministic order: the ``no_memory`` baseline first, then alphabetical."""
    return (0 if condition == NO_MEMORY else 1, condition)
