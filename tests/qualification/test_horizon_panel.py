from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.qualification.horizon_panel import (
    HORIZON_PRIMARY_ESTIMANDS,
    compute_horizon_panel_contrasts,
    horizon_panel_scorecard,
)
from lhmsb.qualification.metrics import MultisystemMetricInput
from lhmsb.qualification.statistics import (
    compute_horizon_panel_statistics,
    horizon_panel_statistics_markdown,
)


def _observation(
    *,
    condition: str,
    readout: str,
    level: str,
    variant: str,
    score: float,
) -> MultisystemMetricInput:
    return MultisystemMetricInput(
        policy_profile_id="policy",
        condition=condition,
        readout=readout,
        result_id=f"{condition}-{level}-{variant}",
        behavior_score=score,
        is_correct=score >= 0.5,
        episode_id=f"episode-{condition}-{level}-{variant}",
        sceu_id="sceu-00",
        opportunity_id="opp-matched-terminal",
        construct_kind={
            "static": "static_recall",
            "evolution": "state_evolution",
            "hierarchical_conflict": "hierarchical_conflict",
        }[variant],
        counterfactual_group_id=f"group-{level}",
        counterfactual_variant=variant,
        counterfactual_terminal_archetype="current_v2_offline",
        is_counterfactual_target=True,
        horizon_panel_id="panel-01",
        horizon_level=level,
        horizon_axis="joint_effective_transition_and_session_handoff_dose",
    )


def _complete_rows() -> tuple[MultisystemMetricInput, ...]:
    scores = {
        "workspace_only": {
            "short": (0.8, 0.7, 0.7),
            "medium": (0.8, 0.65, 0.65),
            "long": (0.8, 0.6, 0.6),
        },
        "flat_retrieval": {
            "short": (0.9, 0.8, 0.7),
            "medium": (0.9, 0.7, 0.6),
            "long": (0.9, 0.5, 0.4),
        },
    }
    variants = ("static", "evolution", "hierarchical_conflict")
    output = []
    for condition, levels in scores.items():
        readout = "none" if condition == "workspace_only" else "native"
        for level, values in levels.items():
            for variant, score in zip(variants, values, strict=True):
                output.append(
                    _observation(
                        condition=condition,
                        readout=readout,
                        level=level,
                        variant=variant,
                        score=score,
                    )
                )
    return tuple(output)


def test_horizon_panel_contrast_is_workspace_adjusted_triple_difference() -> None:
    rows = compute_horizon_panel_contrasts(_complete_rows())
    treatment = next(row for row in rows if row["condition"] == "flat_retrieval")

    assert treatment["complete"] is True
    assert treatment["analysis_unit"] == "horizon_panel"
    assert treatment["workspace_matched_control_available"] is True
    assert treatment["short_state_evolution_penalty_vs_static"] == pytest.approx(0.1)
    assert treatment["long_state_evolution_penalty_vs_static"] == pytest.approx(0.4)
    assert treatment[
        "state_evolution_penalty_amplification_long_vs_short"
    ] == pytest.approx(0.3)
    assert treatment[
        "short_state_evolution_penalty_excess_over_workspace"
    ] == pytest.approx(0.0)
    assert treatment[
        "long_state_evolution_penalty_excess_over_workspace"
    ] == pytest.approx(0.2)
    assert treatment[
        "state_evolution_horizon_amplification_excess_over_workspace"
    ] == pytest.approx(0.2)
    assert treatment[
        "hierarchical_conflict_horizon_amplification_excess_over_workspace"
    ] == pytest.approx(0.2)
    assert treatment[
        "evolution_memory_value_change_long_vs_short"
    ] == pytest.approx(-0.2)


def test_horizon_scorecard_counts_panels_not_physical_members() -> None:
    scorecard = horizon_panel_scorecard(_complete_rows())
    treatment = next(
        row for row in scorecard if row["condition"] == "flat_retrieval"
    )

    assert treatment["analysis_unit"] == "horizon_panel"
    assert treatment["n_horizon_panels"] == 1
    assert treatment["n_complete_horizon_panels"] == 1
    for estimand in HORIZON_PRIMARY_ESTIMANDS:
        assert treatment[f"mean_{estimand}"] == pytest.approx(0.2)


def test_horizon_panel_requires_all_nine_same_decision_cells() -> None:
    observations = _complete_rows()
    missing = tuple(
        row
        for row in observations
        if row.result_id != "flat_retrieval-long-evolution"
    )

    contrasts = compute_horizon_panel_contrasts(missing)
    treatment = next(
        row for row in contrasts if row["condition"] == "flat_retrieval"
    )

    assert treatment["complete"] is False
    assert treatment["long_evolution_behavior_score"] is None
    assert treatment[
        "state_evolution_horizon_amplification_excess_over_workspace"
    ] is None


def test_non_panel_observations_are_ignored() -> None:
    row = _observation(
        condition="flat_retrieval",
        readout="native",
        level="long",
        variant="static",
        score=1.0,
    )
    row = replace(row, horizon_panel_id="")

    assert compute_horizon_panel_contrasts((row,)) == ()


def test_horizon_statistics_resample_panels_not_nine_members() -> None:
    observations = tuple(
        replace(
            row,
            horizon_panel_id=f"panel-{panel}",
            episode_id=f"panel-{panel}-{row.episode_id}",
            result_id=f"panel-{panel}-{row.result_id}",
        )
        for panel in range(3)
        for row in _complete_rows()
    )

    payload = compute_horizon_panel_statistics(
        observations,
        bootstrap_resamples=20,
        permutation_resamples=20,
    )
    estimate = next(
        row
        for row in payload["estimates"]  # type: ignore[index]
        if row["condition"] == "flat_retrieval"  # type: ignore[index]
        and row["metric"]  # type: ignore[index]
        == "state_evolution_horizon_amplification_excess_over_workspace"
    )

    assert payload["analysis_unit"] == "horizon_panel"
    assert payload["n_unique_horizon_panels"] == 3
    assert estimate["analysis_unit"] == "horizon_panel"
    assert estimate["n_panels"] == 3
    assert estimate["mean_difference"] == pytest.approx(0.2)
    markdown = horizon_panel_statistics_markdown(payload)
    assert "nine dependent" in markdown
    assert "pure handoff effect" in markdown
