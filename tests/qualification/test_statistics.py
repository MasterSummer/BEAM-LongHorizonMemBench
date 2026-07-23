from __future__ import annotations

import pytest

from lhmsb.qualification.metrics import MultisystemMetricInput
from lhmsb.qualification.statistics import (
    MATCHED_PRIMARY_ESTIMANDS,
    MATCHED_SECONDARY_ESTIMANDS,
    compute_episode_cluster_statistics,
    compute_matched_group_statistics,
    matched_group_statistics_markdown,
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
        opportunity_id=f"opp-{opportunity}",
        checkpoint_session=opportunity,
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
            opportunity_id="opp-0",
            checkpoint_session=0,
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
            opportunity_id="opp-1",
            checkpoint_session=1,
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
    assert cells["eligible_drift_violation_rate"]["mean"] == 0.0
    assert cells["canonical_drift_violation_rate"]["mean"] == 0.5
    assert cells["observed_longitudinal_drift_incidence"]["mean"] == 0.0


def test_statistics_reports_adherence_anchored_longitudinal_drift() -> None:
    rows = tuple(
        MultisystemMetricInput(
            policy_profile_id="gpt",
            condition="mem0",
            readout="native",
            result_id=f"{episode}-{session}",
            behavior_score=0.0 if drift else 1.0,
            is_correct=not drift,
            episode_id=episode,
            opportunity_id=f"opp-{session}",
            checkpoint_session=session,
            drift_flags=(("stale_state",) if drift else ()),
            drift_eligible_categories=("stale_state",),
        )
        for episode, late_drift in (("e0", True), ("e1", False))
        for session, drift in ((1, False), (4, late_drift))
    )

    payload = compute_episode_cluster_statistics(
        rows,
        bootstrap_resamples=20,
        permutation_resamples=20,
    )
    cell = next(
        row
        for row in payload["cells"]  # type: ignore[index]
        if row["metric"] == "observed_longitudinal_drift_incidence"  # type: ignore[index]
    )

    assert cell["n_episodes"] == 2
    assert cell["mean"] == 0.5


def test_statistics_pairs_targeted_drift_at_episode_level() -> None:
    rows = tuple(
        MultisystemMetricInput(
            policy_profile_id="gpt",
            condition=condition,
            readout=readout,
            result_id=f"{episode}-{condition}",
            behavior_score=0.0,
            is_correct=False,
            episode_id=episode,
            drift_flags=("stale_state",) if condition == "workspace_only" else (),
            drift_eligible_categories=("stale_state",),
        )
        for episode in ("e0", "e1", "e2")
        for condition, readout in (
            ("workspace_only", "none"),
            ("mem0", "native"),
        )
    )

    payload = compute_episode_cluster_statistics(
        rows,
        bootstrap_resamples=20,
        permutation_resamples=20,
    )
    comparison = next(
        row
        for row in payload["paired_comparisons"]  # type: ignore[index]
        if row["metric"] == "eligible_drift_rate"  # type: ignore[index]
    )

    assert comparison["analysis_unit"] == "paired_episode"
    assert comparison["n_pairs"] == 3
    assert comparison["mean_difference"] == -1.0


def test_matched_statistics_use_counterfactual_group_as_unit() -> None:
    rows: list[MultisystemMetricInput] = []
    for index in range(4):
        group = f"group-{index}"
        archetype = (
            "current_v2_offline" if index % 2 == 0 else "authorized_cloud"
        )
        for variant, score, drift in (
            ("static", 1.0, False),
            ("evolution", 0.6 - index * 0.1, True),
            ("hierarchical_conflict", 0.4 - index * 0.05, True),
        ):
            rows.append(
                MultisystemMetricInput(
                    policy_profile_id="gpt",
                    condition="mem0",
                    readout="native",
                    result_id=f"{group}-{variant}",
                    behavior_score=score,
                    is_correct=score >= 0.5,
                    episode_id=f"{group}-{variant}",
                    opportunity_id="opp-matched-terminal",
                    counterfactual_group_id=group,
                    counterfactual_variant=variant,
                    counterfactual_terminal_archetype=archetype,
                    is_counterfactual_target=True,
                    drift_flags=(("plan_deviation",) if drift else ()),
                    drift_eligible_categories=("plan_deviation",),
                )
            )

    payload = compute_matched_group_statistics(
        rows,
        seed=9,
        bootstrap_resamples=100,
        permutation_resamples=100,
    )

    assert payload["analysis_unit"] == "counterfactual_group"
    assert payload["primary_analysis_unit"] == "counterfactual_group"
    assert payload["primary_estimands"] == list(MATCHED_PRIMARY_ESTIMANDS)
    assert payload["secondary_estimands"] == list(MATCHED_SECONDARY_ESTIMANDS)
    assert payload["primary_workspace_adjustment"] == (
        "matched_workspace_only_difference_in_differences"
    )
    assert payload["drift_scope"] == "endpoint_violation_only"
    assert payload["multiplicity_scope"] == (
        "within_estimand_across_policy_condition_readout_cells"
    )
    assert payload["n_unique_counterfactual_groups"] == 4
    estimate = next(
        row
        for row in payload["estimates"]  # type: ignore[index]
        if row["metric"] == "state_evolution_penalty_vs_static"  # type: ignore[index]
    )
    assert estimate["n_pairs"] == 4
    assert estimate["mean_difference"] == pytest.approx(0.55)
    assert estimate["analysis_unit"] == "counterfactual_group"
    assert estimate["estimand_role"] == "secondary"
    assert "holm_adjusted_p_value" in estimate
    assert len(payload["terminal_archetype_sensitivity"]) == 12  # type: ignore[arg-type]
    markdown = matched_group_statistics_markdown(payload)
    assert "counterfactual group" in markdown
    assert "Frozen analysis contract" in markdown
    assert "Primary estimands" in markdown
    assert "violation excesses" in markdown


def test_matched_statistics_estimate_penalty_excess_over_workspace() -> None:
    rows = tuple(
        MultisystemMetricInput(
            policy_profile_id="gpt",
            condition=condition,
            readout=readout,
            result_id=f"{group}-{condition}-{variant}",
            behavior_score=score,
            is_correct=score >= 0.5,
            episode_id=f"{group}-{variant}",
            opportunity_id="opp-matched-terminal",
            counterfactual_group_id=group,
            counterfactual_variant=variant,
            counterfactual_terminal_archetype="current_v2_offline",
            is_counterfactual_target=True,
        )
        for group in ("group-0", "group-1", "group-2")
        for condition, readout, scores in (
            ("workspace_only", "none", (0.8, 0.6, 0.5)),
            ("mem0", "native", (0.9, 0.4, 0.2)),
        )
        for variant, score in zip(
            ("static", "evolution", "hierarchical_conflict"),
            scores,
            strict=True,
        )
    )

    payload = compute_matched_group_statistics(
        rows,
        bootstrap_resamples=20,
        permutation_resamples=20,
    )
    estimate = next(
        row
        for row in payload["estimates"]  # type: ignore[index]
        if row["condition"] == "mem0"  # type: ignore[index]
        and row["metric"]  # type: ignore[index]
        == "state_evolution_penalty_excess_over_workspace"
    )

    assert estimate["analysis_unit"] == "counterfactual_group"
    assert estimate["estimand_role"] == "primary"
    assert estimate["n_pairs"] == 3
    assert estimate["mean_difference"] == pytest.approx(0.3)
    assert "difference-in-differences" in payload["notes"][3]  # type: ignore[index]
