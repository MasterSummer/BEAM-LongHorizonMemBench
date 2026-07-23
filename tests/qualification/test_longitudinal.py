from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from lhmsb.qualification.longitudinal import (
    DriftLineageAlignmentError,
    compute_drift_trajectory_report,
    drift_trajectory_markdown,
)
from lhmsb.qualification.metrics import MultisystemMetricInput
from lhmsb.qualification.validate import _validate_drift_trajectories


def _row(
    episode: str,
    session: int,
    *,
    drift: bool,
    control_kind: str = "native",
    lineage: str = "plan_node:pipeline",
) -> MultisystemMetricInput:
    return MultisystemMetricInput(
        policy_profile_id="gpt",
        condition="mem0",
        readout="native",
        result_id=f"{episode}-{session}",
        behavior_score=0.0 if drift else 1.0,
        is_correct=not drift,
        episode_id=episode,
        opportunity_id=f"opp-{session}",
        checkpoint_session=session,
        control_kind=control_kind,
        drift_flags=("stale_state",) if drift else (),
        drift_eligible_categories=("stale_state",),
        drift_lineage_pairs=(("stale_state", lineage),),
        drift_lineage_evidence_mode="declared",
    )


def test_drift_trajectory_uses_episode_events_persistence_and_recovery() -> None:
    rows = (
        _row("episode-1", 1, drift=False),
        _row("episode-1", 2, drift=True),
        _row("episode-1", 5, drift=True),
        _row("episode-1", 6, drift=False, control_kind="fresh_reminder"),
        _row("episode-2", 1, drift=False),
        _row("episode-2", 2, drift=False),
        _row("episode-2", 6, drift=False, control_kind="fresh_reminder"),
    )

    payload = compute_drift_trajectory_report(rows)

    assert payload["analysis_unit"] == "episode"
    trajectories = payload["trajectories"]
    assert isinstance(trajectories, list)
    first = next(row for row in trajectories if row["episode_id"] == "episode-1")
    assert first["violation_event_observed"] is True
    assert first["state_lineage_id"] == "plan_node:pipeline"
    assert first["lineage_backed"] is True
    assert first["first_violation_session"] == 2
    assert first["adherence_anchor_observed"] is True
    assert first["drift_evaluable"] is True
    assert first["first_drift_session"] == 2
    assert first["persistence_numerator"] == 1
    assert first["persistence_denominator"] == 2
    assert first["recovered"] is True
    summary = payload["summary"]
    assert isinstance(summary, list)
    stale = summary[0]
    assert stale["n_episodes"] == 2
    assert stale["violation_incidence"] == 0.5
    assert stale["n_drift_evaluable_episodes"] == 2
    assert stale["observed_drift_incidence"] == 0.5
    assert stale["cumulative_incidence"] == 0.5
    assert stale["persistence_rate"] == 0.5
    assert stale["recovery_rate"] == 1.0
    survival = payload["survival"]
    assert isinstance(survival, list)
    at_two = next(row for row in survival if row["session"] == 2)
    assert at_two["at_risk"] == 2
    assert at_two["events"] == 1
    assert at_two["survival_probability"] == 0.5
    assert "Analysis unit: **episode**" in drift_trajectory_markdown(payload)


def test_drift_trajectory_skips_categories_without_eligible_opportunities() -> None:
    row = _row("episode-1", 2, drift=False)
    payload = compute_drift_trajectory_report((row,))

    trajectories = payload["trajectories"]
    assert isinstance(trajectories, list)
    assert {item["drift_category"] for item in trajectories} == {"stale_state"}


def test_same_session_probes_do_not_count_as_drift_persistence() -> None:
    rows = (
        _row("episode-1", 1, drift=False),
        _row("episode-1", 2, drift=True),
        replace(
            _row("episode-1", 2, drift=True),
            result_id="episode-1-2-second",
            opportunity_id="opp-2-second",
        ),
        _row("episode-1", 5, drift=False),
    )

    payload = compute_drift_trajectory_report(rows)
    trajectory = payload["trajectories"][0]  # type: ignore[index]

    assert trajectory["eligible_opportunity_count"] == 4
    assert trajectory["eligible_checkpoint_count"] == 3
    assert trajectory["persistence_denominator"] == 1
    assert trajectory["persistence_numerator"] == 0


def test_different_state_lineages_cannot_anchor_each_other() -> None:
    rows = (
        _row("episode-1", 1, drift=False, lineage="constraint:C1"),
        _row("episode-1", 2, drift=True, lineage="constraint:C2"),
    )

    payload = compute_drift_trajectory_report(rows)
    trajectories = payload["trajectories"]
    assert isinstance(trajectories, list)
    by_lineage = {row["state_lineage_id"]: row for row in trajectories}
    assert by_lineage["constraint:C1"]["event_observed"] is False
    assert by_lineage["constraint:C2"]["violation_event_observed"] is True
    assert by_lineage["constraint:C2"]["adherence_anchor_observed"] is False
    assert by_lineage["constraint:C2"]["event_observed"] is False


def test_one_sceu_category_cannot_broadcast_to_multiple_lineages() -> None:
    ambiguous = replace(
        _row("episode-1", 2, drift=True, lineage="constraint:C1"),
        drift_lineage_pairs=(
            ("stale_state", "constraint:C1"),
            ("stale_state", "constraint:C2"),
        ),
    )

    with pytest.raises(DriftLineageAlignmentError, match="split ambiguous"):
        compute_drift_trajectory_report((ambiguous,))


def test_category_only_legacy_rows_are_explicitly_labeled() -> None:
    rows = (
        replace(
            _row("episode-1", 1, drift=False),
            drift_lineage_pairs=(),
            drift_lineage_evidence_mode="unavailable",
        ),
        replace(
            _row("episode-1", 2, drift=True),
            drift_lineage_pairs=(),
            drift_lineage_evidence_mode="unavailable",
        ),
    )

    payload = compute_drift_trajectory_report(rows)
    trajectory = payload["trajectories"][0]  # type: ignore[index]
    assert trajectory["state_lineage_id"] == "__category_only__"
    assert trajectory["lineage_evidence_mode"] == "category_only_legacy"
    assert trajectory["lineage_backed"] is False


def test_schema_v4_validator_requires_state_lineage_evidence() -> None:
    payload = compute_drift_trajectory_report(
        (
            _row("episode-1", 1, drift=False),
            _row("episode-1", 2, drift=True),
        )
    )
    errors: list[str] = []
    _validate_drift_trajectories(payload, errors)
    assert errors == []

    tampered = deepcopy(payload)
    del tampered["trajectories"][0]["lineage_backed"]  # type: ignore[index]
    errors = []
    _validate_drift_trajectories(tampered, errors)
    assert any("boolean lineage_backed" in error for error in errors)


def test_schema_v4_validator_checks_episode_summary_unit() -> None:
    payload = compute_drift_trajectory_report(
        (
            _row("episode-1", 1, drift=False),
            _row("episode-1", 2, drift=True),
        )
    )
    payload["summary"][0]["n_episodes"] = 2  # type: ignore[index]
    errors: list[str] = []

    _validate_drift_trajectories(payload, errors)

    assert any("does not use unique episodes" in error for error in errors)


def test_survival_risk_set_respects_delayed_episode_entry() -> None:
    payload = compute_drift_trajectory_report(
        (
            _row("episode-early", 1, drift=False),
            _row("episode-early", 2, drift=True),
            _row("episode-late", 5, drift=False),
            _row("episode-late", 6, drift=False),
        )
    )
    survival = payload["survival"]
    at_two = next(row for row in survival if row["session"] == 2)  # type: ignore[union-attr]

    assert at_two["at_risk"] == 1
    assert at_two["events"] == 1
    assert at_two["survival_probability"] == 0.0


def test_first_observation_violation_is_not_mislabeled_as_drift() -> None:
    payload = compute_drift_trajectory_report(
        (
            _row("episode-1", 2, drift=True),
            _row("episode-1", 5, drift=True),
        )
    )
    trajectory = payload["trajectories"][0]  # type: ignore[index]
    summary = payload["summary"][0]  # type: ignore[index]

    assert trajectory["violation_event_observed"] is True
    assert trajectory["first_violation_session"] == 2
    assert trajectory["adherence_anchor_observed"] is False
    assert trajectory["drift_evaluable"] is False
    assert trajectory["event_observed"] is False
    assert trajectory["first_drift_session"] is None
    assert summary["violation_incidence"] == 1.0
    assert summary["n_drift_evaluable_episodes"] == 0
    assert summary["observed_drift_incidence"] is None
    assert payload["survival"] == []


def test_drift_can_begin_after_recovery_from_an_initial_violation() -> None:
    payload = compute_drift_trajectory_report(
        (
            _row("episode-1", 2, drift=True),
            _row("episode-1", 5, drift=False),
            _row("episode-1", 7, drift=True),
        )
    )
    trajectory = payload["trajectories"][0]  # type: ignore[index]

    assert trajectory["first_violation_session"] == 2
    assert trajectory["first_adherence_session"] == 5
    assert trajectory["drift_evaluable"] is True
    assert trajectory["first_drift_session"] == 7
