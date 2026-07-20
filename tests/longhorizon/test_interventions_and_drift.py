from __future__ import annotations

from lhmsb.longhorizon.drift import (
    DriftEvidence,
    MatchedOutcome,
    classify_long_horizon_drift,
    classify_matched_decay,
)
from lhmsb.longhorizon.interventions import (
    ContinuationOutcome,
    classify_causal_use,
)


def _outcome(
    action_id: str,
    score: float,
    *,
    correct: bool,
    violated: tuple[str, ...] = (),
    drift: tuple[str, ...] = (),
) -> ContinuationOutcome:
    return ContinuationOutcome(
        action_id=action_id,
        behavior_score=score,
        is_correct=correct,
        violated_state_ids=violated,
        drift_flags=drift,
    )


SAFE = _outcome("safe_v2_offline", 1.0, correct=True)
STALE = _outcome(
    "stale_v1",
    0.25,
    correct=False,
    violated=("P2", "U1"),
    drift=("stale-state", "goal-drift"),
)
CLOUD = _outcome(
    "cloud_shortcut",
    0.25,
    correct=False,
    violated=("C1",),
    drift=(
        "constraint-influence-lost",
        "local-subgoal-overwrites-global-goal",
    ),
)


def test_stable_repeated_outcomes_without_change_are_not_causal_use() -> None:
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="leave_one_out",
        memory_role="supports_current_state",
        baseline=(SAFE, SAFE),
        intervention=(SAFE, SAFE),
    )
    assert result.label == "visible_not_causally_used"
    assert result.baseline_stable
    assert result.intervention_stable
    assert not result.behaviorally_used


def test_unstable_baseline_prevents_a_causal_claim() -> None:
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="leave_one_out",
        memory_role="supports_current_state",
        baseline=(SAFE, STALE),
        intervention=(STALE, STALE),
    )
    assert result.label == "unstable_baseline"
    assert not result.behaviorally_used


def test_stable_leave_one_out_can_show_beneficial_use() -> None:
    result = classify_causal_use(
        memory_id="m-current",
        intervention_kind="leave_one_out",
        memory_role="supports_current_state",
        baseline=(SAFE, SAFE),
        intervention=(STALE, STALE),
    )
    assert result.label == "beneficial"
    assert result.effect_direction == "beneficial"
    assert result.behaviorally_used
    assert result.action_changed


def test_stable_leave_one_out_can_show_harmful_use() -> None:
    result = classify_causal_use(
        memory_id="m-stale",
        intervention_kind="leave_one_out",
        memory_role="contradicts_current_state",
        baseline=(CLOUD, CLOUD),
        intervention=(SAFE, SAFE),
    )
    assert result.label == "harmful"
    assert result.effect_direction == "harmful"
    assert result.behaviorally_used


def test_stale_conflict_replacement_is_classified_from_persisted_outcomes() -> None:
    result = classify_causal_use(
        memory_id="m-v1",
        intervention_kind="stale_replacement",
        memory_role="contradicts_current_state",
        baseline=(STALE, STALE),
        intervention=(SAFE, SAFE),
    )
    assert result.label == "harmful"
    assert result.intervention_kind == "stale_replacement"
    assert result.checker_changed


def test_unstable_intervention_prevents_a_causal_claim() -> None:
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="leave_one_out",
        memory_role="supports_current_state",
        baseline=(SAFE, SAFE),
        intervention=(SAFE, STALE),
    )
    assert result.label == "intervention_unstable"
    assert not result.behaviorally_used


def test_checker_change_counts_even_when_the_action_id_is_unchanged() -> None:
    degraded = _outcome(
        "safe_v2_offline",
        0.5,
        correct=False,
        violated=("C1",),
        drift=("constraint-influence-lost",),
    )
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="leave_one_out",
        memory_role="supports_current_state",
        baseline=(SAFE, SAFE),
        intervention=(degraded, degraded),
    )
    assert result.label == "beneficial"
    assert not result.action_changed
    assert result.checker_changed


def test_direction_that_disagrees_with_gold_role_remains_ambiguous() -> None:
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="leave_one_out",
        memory_role="contradicts_current_state",
        baseline=(SAFE, SAFE),
        intervention=(STALE, STALE),
    )
    assert result.label == "causal_direction_ambiguous"
    assert not result.behaviorally_used


def test_four_long_horizon_drift_components_are_programmatic() -> None:
    outcome = _outcome(
        "cloud_shortcut",
        0.0,
        correct=False,
        violated=("C1", "P2"),
    )
    result = classify_long_horizon_drift(
        DriftEvidence(
            outcome=outcome,
            used_state_ids=("P1", "L1"),
            active_constraint_ids=("C1",),
            current_plan_state_ids=("P2",),
            stale_state_ids=("P1",),
            selected_local_state_ids=("L1",),
            global_state_ids=("G0", "C1"),
        )
    )
    assert result.constraint_loss
    assert result.plan_deviation
    assert result.stale_state
    assert result.local_over_global
    assert result.flags == (
        "constraint_loss",
        "local_over_global",
        "plan_deviation",
        "stale_state",
    )


def test_valid_state_update_is_not_drift() -> None:
    result = classify_long_horizon_drift(
        DriftEvidence(
            outcome=SAFE,
            used_state_ids=("P2",),
            active_constraint_ids=("C1", "C2"),
            current_plan_state_ids=("P2",),
            stale_state_ids=("P1",),
            global_state_ids=("G0", "C1", "C2"),
        )
    )
    assert not result.has_drift
    assert result.flags == ()


def test_matched_early_late_pair_detects_behavioral_decay() -> None:
    result = classify_matched_decay(
        MatchedOutcome("branch-choice", 4, SAFE),
        MatchedOutcome("branch-choice", 14, STALE),
    )
    assert result.label == "behavioral_decay"
    assert result.score_delta == -0.75
    assert result.action_changed
    assert result.drift_emerged


def test_count_preserving_intervention_kinds_are_supported() -> None:
    baseline = (
        _outcome("safe_v2_offline", 1.0, correct=True),
        _outcome("safe_v2_offline", 1.0, correct=True),
    )
    changed = (
        _outcome("stale_v1", 0.0, correct=False),
        _outcome("stale_v1", 0.0, correct=False),
    )
    result = classify_causal_use(
        memory_id="m1",
        intervention_kind="neutral_replacement",
        memory_role="supports_current_state",
        baseline=baseline,
        intervention=changed,
    )
    assert result.behaviorally_used
    assert result.label == "beneficial"
