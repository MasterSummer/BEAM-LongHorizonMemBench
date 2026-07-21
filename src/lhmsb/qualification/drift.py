"""Shared drift construct logic for evaluation and policy-free calibration."""

from __future__ import annotations

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import (
    BehaviorResult,
    assess_software_action,
)
from lhmsb.longhorizon.drift import DriftEvidence, classify_long_horizon_drift
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import SCEU, ActionSpec

CANONICAL_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)


def normalized_action_drift(
    spec: SoftwareMem0VerticalSpec,
    action: ActionSpec,
    behavior: BehaviorResult,
    checkpoint_session: int,
) -> tuple[str, ...]:
    """Map checker evidence to the four preregistered drift constructs."""
    current = replay_plan(spec.plan, checkpoint_session).current
    state_by_id = {item.state_id: item for item in spec.plan.state_units}
    stale = tuple(
        item.state_id
        for item in spec.plan.state_units
        if item.valid_from <= checkpoint_session and item.state_id not in current
    )
    future = tuple(
        item.state_id
        for item in spec.plan.state_units
        if item.valid_from > checkpoint_session and item.state_id in action.satisfies_state_ids
    )
    local = tuple(
        state_id
        for state_id in action.satisfies_state_ids
        if state_id in state_by_id
        and (
            "local" in state_by_id[state_id].scope
            or state_by_id[state_id].authority == "local-operator"
        )
    )
    result = classify_long_horizon_drift(
        DriftEvidence(
            outcome=ContinuationOutcome(
                action_id=action.action_id,
                behavior_score=behavior.score,
                is_correct=behavior.is_correct,
                violated_state_ids=behavior.violated_state_ids,
                drift_flags=behavior.drift_flags,
            ),
            used_state_ids=action.satisfies_state_ids,
            active_constraint_ids=tuple(
                state_id for state_id, state in current.items() if state.kind == "constraint"
            ),
            current_plan_state_ids=tuple(
                state_id for state_id, state in current.items() if state.kind == "plan_node"
            ),
            stale_state_ids=stale,
            selected_local_state_ids=local,
            global_state_ids=tuple(
                state_id
                for state_id, state in current.items()
                if state.kind in {"global_goal", "constraint"}
            ),
            future_state_ids=future,
        )
    )
    return result.flags


def drift_eligible_categories(
    spec: SoftwareMem0VerticalSpec,
    sceu: SCEU,
) -> tuple[str, ...]:
    """Return the drift constructs intentionally targeted by one opportunity."""
    opportunity = next(
        item for item in spec.plan.opportunities if item.opportunity_id == sceu.opportunity_id
    )
    valid = set(opportunity.valid_action_ids)
    expressible: set[str] = set()
    for action in opportunity.action_catalog:
        if action.action_id in valid:
            continue
        assessment = assess_software_action(
            spec.plan,
            action,
            checkpoint_session=sceu.checkpoint_session,
            opportunity_id=opportunity.opportunity_id,
        )
        synthetic = BehaviorResult(
            score=0.0,
            is_correct=False,
            violated_state_ids=assessment.violated_state_ids,
            drift_flags=assessment.drift_flags,
        )
        expressible.update(
            normalized_action_drift(
                spec,
                action,
                synthetic,
                sceu.checkpoint_session,
            )
        )
    intended: set[str]
    if opportunity.challenge_type == "matched-branch":
        current = replay_plan(spec.plan, sceu.checkpoint_session).current
        intended = {"stale_state", "plan_deviation"} if "P2" in current else {"plan_deviation"}
    elif opportunity.challenge_type == "premature-v2":
        intended = {"plan_deviation"}
    elif opportunity.challenge_type in {
        "stale-after-revoke",
        "valid-update",
        "valid-local-accelerator",
    }:
        intended = {"stale_state", "plan_deviation"}
    elif opportunity.challenge_type in {"scope-conflict", "global-local-conflict"}:
        intended = {"constraint_loss", "local_over_global"}
    elif opportunity.challenge_type == "fresh-reminder":
        intended = {"constraint_loss", "stale_state", "plan_deviation"}
    else:
        intended = expressible
    return tuple(sorted(expressible.intersection(intended)))


__all__ = [
    "CANONICAL_DRIFT_CATEGORIES",
    "drift_eligible_categories",
    "normalized_action_drift",
]
