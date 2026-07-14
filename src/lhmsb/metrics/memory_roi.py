"""Memory ROI using recorded memory count instead of token/resource cost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from lhmsb.metrics.roi import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_EPS,
    OVERALL_FAMILY,
    PARETO_DOMINATED,
    PARETO_NA,
    PARETO_ON_FRONT,
    bootstrap_ci,
    normalized_gain,
    pareto_front,
)

if TYPE_CHECKING:
    from lhmsb.runner.results import ResultsTable, RunRow

NO_MEMORY_CONDITIONS = frozenset({"no_memory", "no_mem"})
MEMORY_ROI_OK = "ok"
MEMORY_ROI_NA_BASELINE = "na_baseline"
MEMORY_ROI_NA_NO_PAIRS = "na_no_pairs"
MEMORY_ROI_NA_ZERO_MEMORY = "na_zero_memory"


@dataclass(frozen=True)
class MemoryRoiResult:
    """Per-condition gain per recorded memory, with CI and count Pareto context."""

    condition: str
    family: str
    n_episodes: int
    mean_normalized_gain: float
    mean_memory_count: float
    roi: float | None
    ci_low: float | None
    ci_high: float | None
    gain_floor: float
    pareto_status: str
    roi_status: str
    is_baseline: bool = False
    below_gain_floor: bool = False


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _baseline_scores(rows: Sequence[RunRow]) -> dict[tuple[str, str, int], float]:
    scores: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row.condition in NO_MEMORY_CONDITIONS and row.status == "completed":
            scores[(row.family, row.episode_id, row.seed)] = row.task_score
    return scores


def _condition_key(condition: str) -> tuple[int, str]:
    return (0 if condition in NO_MEMORY_CONDITIONS else 1, condition)


def _baseline_result(family: str, condition: str, cells: Sequence[RunRow]) -> MemoryRoiResult:
    return MemoryRoiResult(
        condition=condition,
        family=family,
        n_episodes=len(cells),
        mean_normalized_gain=0.0,
        mean_memory_count=(
            _mean([float(row.stored_memory_count) for row in cells]) if cells else 0.0
        ),
        roi=None,
        ci_low=None,
        ci_high=None,
        gain_floor=0.0,
        pareto_status=PARETO_NA,
        roi_status=MEMORY_ROI_NA_BASELINE,
        is_baseline=True,
    )


def _system_result(
    family: str,
    condition: str,
    cells: Sequence[RunRow],
    baseline: dict[tuple[str, str, int], float],
    *,
    max_achievable: float,
    eps: float,
    bootstrap_n: int,
    alpha: float,
    seed: int,
    gain_floor_threshold: float,
) -> MemoryRoiResult:
    gains: list[float] = []
    counts: list[float] = []
    for row in cells:
        key = (row.family, row.episode_id, row.seed)
        if key not in baseline:
            continue
        gains.append(normalized_gain(row.task_score, baseline[key], max_achievable, eps))
        counts.append(float(row.stored_memory_count))
    if not gains:
        return MemoryRoiResult(
            condition=condition,
            family=family,
            n_episodes=0,
            mean_normalized_gain=0.0,
            mean_memory_count=0.0,
            roi=None,
            ci_low=None,
            ci_high=None,
            gain_floor=0.0,
            pareto_status=PARETO_NA,
            roi_status=MEMORY_ROI_NA_NO_PAIRS,
        )

    mean_gain = _mean(gains)
    mean_count = _mean(counts)
    gain_floor = _mean([max(0.0, gain) for gain in gains])
    if mean_count <= 0.0:
        return MemoryRoiResult(
            condition=condition,
            family=family,
            n_episodes=len(gains),
            mean_normalized_gain=mean_gain,
            mean_memory_count=mean_count,
            roi=None,
            ci_low=None,
            ci_high=None,
            gain_floor=gain_floor,
            pareto_status=PARETO_NA,
            roi_status=MEMORY_ROI_NA_ZERO_MEMORY,
            below_gain_floor=mean_gain < gain_floor_threshold,
        )

    point, gain_lo, gain_hi = bootstrap_ci(gains, "mean", bootstrap_n, alpha, seed)
    return MemoryRoiResult(
        condition=condition,
        family=family,
        n_episodes=len(gains),
        mean_normalized_gain=mean_gain,
        mean_memory_count=mean_count,
        roi=point / mean_count,
        ci_low=gain_lo / mean_count,
        ci_high=gain_hi / mean_count,
        gain_floor=gain_floor,
        pareto_status=PARETO_NA,
        roi_status=MEMORY_ROI_OK,
        below_gain_floor=mean_gain < gain_floor_threshold,
    )


def compute_memory_roi(
    table: ResultsTable,
    *,
    max_achievable: float = 1.0,
    eps: float = DEFAULT_EPS,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    gain_floor_threshold: float = 0.0,
) -> list[MemoryRoiResult]:
    """Compute ``normalized task gain / mean recorded memory count``.

    ``no_mem``/``no_memory`` remains the counterfactual baseline and has ROI=N/A.
    A system with no recorded memories is also N/A, never an infinite score.
    """
    rows = list(table.rows)
    baseline = _baseline_scores(rows)
    families = sorted({row.family for row in rows})
    results: list[MemoryRoiResult] = []
    for index, family in enumerate([*families, OVERALL_FAMILY]):
        family_rows = (
            rows
            if family == OVERALL_FAMILY
            else [row for row in rows if row.family == family]
        )
        built: list[MemoryRoiResult] = []
        conditions = sorted({row.condition for row in family_rows}, key=_condition_key)
        for offset, condition in enumerate(conditions):
            cells = [row for row in family_rows if row.condition == condition]
            if condition in NO_MEMORY_CONDITIONS:
                built.append(_baseline_result(family, condition, cells))
            else:
                built.append(
                    _system_result(
                        family,
                        condition,
                        cells,
                        baseline,
                        max_achievable=max_achievable,
                        eps=eps,
                        bootstrap_n=bootstrap_n,
                        alpha=alpha,
                        seed=seed + index + offset,
                        gain_floor_threshold=gain_floor_threshold,
                    )
                )
        candidates = [
            (item.condition, item.mean_normalized_gain, item.mean_memory_count)
            for item in built
            if not item.is_baseline
            and item.roi_status
            not in {MEMORY_ROI_NA_NO_PAIRS, MEMORY_ROI_NA_ZERO_MEMORY}
        ]
        front = pareto_front(candidates)
        results.extend(
            replace(
                item,
                pareto_status=(
                    PARETO_ON_FRONT if item.condition in front else PARETO_DOMINATED
                ),
            )
            if not item.is_baseline and item.roi_status == MEMORY_ROI_OK
            else item
            for item in built
        )
    return results


__all__ = [
    "MEMORY_ROI_NA_BASELINE",
    "MEMORY_ROI_NA_NO_PAIRS",
    "MEMORY_ROI_NA_ZERO_MEMORY",
    "MEMORY_ROI_OK",
    "MemoryRoiResult",
    "compute_memory_roi",
]
