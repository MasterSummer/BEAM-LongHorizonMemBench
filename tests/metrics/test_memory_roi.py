from __future__ import annotations

import pytest

from lhmsb.metrics.memory_roi import (
    MEMORY_ROI_NA_BASELINE,
    MEMORY_ROI_NA_ZERO_MEMORY,
    MEMORY_ROI_OK,
    compute_memory_roi,
)
from lhmsb.runner.results import ResultsTable, RunRow


def _row(condition: str, score: float, memory_count: int) -> RunRow:
    return RunRow(
        episode_id="e1",
        family="research_wide",
        seed=0,
        condition=condition,
        track="native",
        status="completed",
        attempts=1,
        n_probes=1,
        world_event_hash="world",
        episode_hash="episode",
        task_score=score,
        utilization_rate=None,
        improvement_over_time=None,
        judge_contribution=0.0,
        drift_index=float("nan"),
        drift_is_na=True,
        stale_fact_violations=0,
        constraint_violations=0,
        behavioral_flips=0,
        judge_fallback_share=0.0,
        stored_memory_count=memory_count,
    )


def test_memory_roi_uses_recorded_memory_count() -> None:
    table = ResultsTable(
        track="native",
        rows=[_row("no_mem", 0.2, 0), _row("mem", 0.6, 2)],
    )

    results = compute_memory_roi(table, bootstrap_n=100)
    by_condition = {
        result.condition: result
        for result in results
        if result.family == "research_wide"
    }

    assert by_condition["no_mem"].roi_status == MEMORY_ROI_NA_BASELINE
    assert by_condition["mem"].roi_status == MEMORY_ROI_OK
    assert by_condition["mem"].mean_memory_count == 2.0
    assert by_condition["mem"].roi == pytest.approx(0.25)


def test_memory_roi_with_zero_records_is_na_not_infinite() -> None:
    table = ResultsTable(
        track="native",
        rows=[_row("no_mem", 0.2, 0), _row("wrong_mem", 0.0, 0)],
    )

    result = next(
        item
        for item in compute_memory_roi(table, bootstrap_n=100)
        if item.family == "research_wide" and item.condition == "wrong_mem"
    )

    assert result.roi is None
    assert result.roi_status == MEMORY_ROI_NA_ZERO_MEMORY
