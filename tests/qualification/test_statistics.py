from __future__ import annotations

import pytest

from lhmsb.qualification.metrics import MultisystemMetricInput
from lhmsb.qualification.statistics import (
    compute_episode_cluster_statistics,
    statistics_markdown,
)


def _row(
    episode: str,
    condition: str,
    readout: str,
    score: float,
    *,
    opportunity: int,
) -> MultisystemMetricInput:
    return MultisystemMetricInput(
        policy_profile_id="gpt",
        condition=condition,
        readout=readout,
        result_id=f"{episode}-{condition}-{opportunity}",
        behavior_score=score,
        is_correct=score >= 0.5,
        episode_id=episode,
        drift_eligible_categories=("plan_deviation",),
    )


def test_statistics_cluster_sceu_rows_by_episode_and_pair_cells() -> None:
    rows = []
    for index in range(4):
        episode = f"e{index}"
        for opportunity in range(3):
            rows.append(
                _row(
                    episode,
                    "workspace_only",
                    "none",
                    0.2 + index * 0.05,
                    opportunity=opportunity,
                )
            )
            rows.append(
                _row(
                    episode,
                    "mem0",
                    "common_rerank",
                    0.7 + index * 0.05,
                    opportunity=opportunity,
                )
            )
    first = compute_episode_cluster_statistics(
        rows,
        episode_groups={
            "e0": "scenario-a",
            "e1": "scenario-a",
            "e2": "scenario-b",
            "e3": "scenario-b",
        },
        seed=7,
        bootstrap_resamples=200,
        permutation_resamples=200,
    )
    second = compute_episode_cluster_statistics(
        rows,
        episode_groups={
            "e0": "scenario-a",
            "e1": "scenario-a",
            "e2": "scenario-b",
            "e3": "scenario-b",
        },
        seed=7,
        bootstrap_resamples=200,
        permutation_resamples=200,
    )
    assert first == second
    assert first["n_unique_episodes"] == 4
    behavior_cells = [
        row
        for row in first["cells"]  # type: ignore[index]
        if row["metric"] == "mean_behavior_score"  # type: ignore[index]
    ]
    assert {row["n_episodes"] for row in behavior_cells} == {4}
    comparison = next(
        row
        for row in first["paired_comparisons"]  # type: ignore[index]
        if row["metric"] == "mean_behavior_score"  # type: ignore[index]
    )
    assert comparison["n_pairs"] == 4
    assert comparison["mean_difference"] == pytest.approx(0.5)
    assert comparison["analysis_unit"] == "paired_episode"
    assert "holm_adjusted_p_value" in comparison
    assert first["n_unique_episode_groups"] == 2
    scenario_cell = next(
        row
        for row in first["scenario_cells"]  # type: ignore[index]
        if row["metric"] == "mean_behavior_score"  # type: ignore[index]
        and row["condition"] == "mem0"  # type: ignore[index]
    )
    assert scenario_cell["n_groups"] == 2
    assert scenario_cell["analysis_unit"] == "semantic_scenario_cluster"
    scenario_comparison = next(
        row
        for row in first["scenario_paired_comparisons"]  # type: ignore[index]
        if row["metric"] == "mean_behavior_score"  # type: ignore[index]
    )
    assert scenario_comparison["n_pairs"] == 2
    assert scenario_comparison["mean_difference"] == pytest.approx(0.5)
    assert "Analysis unit: **episode**" in statistics_markdown(first)
    assert "Semantic-scenario sensitivity" in statistics_markdown(first)


def test_statistics_separates_targeted_and_observed_drift() -> None:
    rows = (
        MultisystemMetricInput(
            policy_profile_id="gpt",
            condition="oracle_current_state",
            readout="none",
            result_id="e0-0",
            behavior_score=0.0,
            is_correct=False,
            episode_id="e0",
            drift_flags=("constraint_loss",),
            drift_eligible_categories=("stale_state",),
        ),
        MultisystemMetricInput(
            policy_profile_id="gpt",
            condition="oracle_current_state",
            readout="none",
            result_id="e0-1",
            behavior_score=1.0,
            is_correct=True,
            episode_id="e0",
            drift_eligible_categories=("stale_state",),
        ),
    )
    payload = compute_episode_cluster_statistics(
        rows,
        bootstrap_resamples=10,
        permutation_resamples=10,
    )
    cells = {
        row["metric"]: row
        for row in payload["cells"]  # type: ignore[index]
    }
    assert cells["eligible_drift_rate"]["mean"] == 0.0
    assert cells["observed_drift_rate"]["mean"] == 0.5
