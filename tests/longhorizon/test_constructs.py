from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.constructs import horizon_band, profile_sceu


def _profile(opportunity_id: str):  # type: ignore[no-untyped-def]
    spec = SoftwareMem0VerticalFamily.generate(
        42,
        n_sessions=16,
        trajectory_seed=2,
    )
    sceu = next(item for item in spec.plan.sceu_units if item.opportunity_id == opportunity_id)
    return spec, profile_sceu(spec.plan, sceu)


def test_premature_probe_does_not_treat_future_state_as_required_current_state() -> None:
    _spec, profile = _profile("opp-premature-v2")

    assert profile.construct_kind == "state_evolution"
    assert profile.current_required_state_ids == ("C1", "G0", "P1")
    assert {"U1", "P2"} <= set(profile.future_referenced_state_ids)
    assert "U1" not in profile.memory_reliant_state_ids
    assert "P2" not in profile.memory_reliant_state_ids


def test_late_decision_exposes_state_evolution_and_absent_workspace_need() -> None:
    _spec, profile = _profile("opp-stale-v1")

    assert profile.construct_kind == "state_evolution"
    assert profile.relevant_transition_count >= 2
    assert "P1" in profile.invalidated_referenced_state_ids
    assert {"C1", "P2", "U1"} <= set(profile.memory_reliant_state_ids)
    assert profile.workspace_absent_count >= 3
    assert profile.dependency_depth >= 2
    assert profile.horizon_band == "long"


def test_scope_and_authority_decisions_are_hierarchical_conflicts() -> None:
    _spec, profile = _profile("opp-global-local-conflict")

    assert profile.construct_kind == "hierarchical_conflict"
    assert profile.relevant_event_count > 0
    assert profile.handoff_count == profile.checkpoint_session
    assert profile.to_dict()["construct_kind"] == "hierarchical_conflict"


def test_generated_sceu_contracts_cover_all_current_action_relevant_state() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16, trajectory_seed=2)

    profiles = tuple(profile_sceu(spec.plan, sceu) for sceu in spec.plan.sceu_units)

    assert profiles
    assert all(not profile.missing_current_action_relevant_state_ids for profile in profiles)


def test_profiler_detects_an_incomplete_current_action_state_contract() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16, trajectory_seed=2)
    opportunity = next(
        item
        for item in spec.plan.opportunities
        if item.opportunity_id == "opp-global-local-conflict"
    )
    incomplete = replace(
        opportunity,
        focal_state_ids=tuple(
            state_id for state_id in opportunity.focal_state_ids if state_id != "P2"
        ),
    )
    plan = replace(
        spec.plan,
        opportunities=tuple(
            incomplete if item.opportunity_id == incomplete.opportunity_id else item
            for item in spec.plan.opportunities
        ),
    )
    sceu = next(
        item for item in plan.sceu_units if item.opportunity_id == incomplete.opportunity_id
    )

    profile = profile_sceu(plan, sceu)

    assert "P2" in profile.current_action_relevant_state_ids
    assert "P2" in profile.missing_current_action_relevant_state_ids


@pytest.mark.parametrize(
    ("count", "expected"),
    ((0, "short"), (2, "short"), (3, "medium"), (7, "medium"), (8, "long")),
)
def test_horizon_bands_are_absolute_handoff_counts(count: int, expected: str) -> None:
    assert horizon_band(count) == expected


def test_horizon_band_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        horizon_band(-1)
