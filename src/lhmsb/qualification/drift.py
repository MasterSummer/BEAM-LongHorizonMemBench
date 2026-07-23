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
from lhmsb.longhorizon.schema import SCEU, ActionSpec, EpisodePlan, StateUnit

CANONICAL_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)

DRIFT_LINEAGE_EVIDENCE_MODE = "derived_state_graph_v1"


def _plan(source: SoftwareMem0VerticalSpec | EpisodePlan) -> EpisodePlan:
    return source if isinstance(source, EpisodePlan) else source.plan


def _state_lineage_id(state: StateUnit) -> str:
    """Return the stable semantic lineage used by longitudinal drift.

    Persistent goals and constraints keep their evaluator state identity. Plan
    and scoped-decision replacements may receive a new state ID (for example
    ``P1`` -> ``P2``), so their lineage is defined by state kind and scope.
    This derivation is intentionally evaluator-side and is versioned by
    ``DRIFT_LINEAGE_EVIDENCE_MODE``; future dataset releases may replace it
    with an explicitly declared lineage without changing the report contract.
    """

    if state.kind in {"plan_node", "decision"}:
        return f"{state.kind}:{state.scope}"
    return f"{state.kind}:{state.state_id}"


def _lineage_candidate_ids(
    *,
    category: str,
    action: ActionSpec,
    violated_state_ids: tuple[str, ...],
    future_state_ids: tuple[str, ...],
    state_by_id: dict[str, StateUnit],
) -> tuple[str, ...]:
    """Find the state predicates whose continued influence is being tested."""

    implicated = set(action.satisfies_state_ids)
    implicated.update(action.violates_state_ids)
    implicated.update(violated_state_ids)
    implicated.update(future_state_ids)
    if category in {"constraint_loss", "local_over_global"}:
        preferred = {
            state_id
            for state_id in violated_state_ids
            if state_id in state_by_id
            and state_by_id[state_id].kind in {"constraint", "global_goal"}
        }
        if preferred:
            return tuple(sorted(preferred))
        return tuple(
            sorted(
                state_id
                for state_id in implicated
                if state_id in state_by_id
                and state_by_id[state_id].kind in {"constraint", "global_goal"}
            )
        )
    if category in {"plan_deviation", "stale_state"}:
        preferred = {
            state_id
            for state_id in implicated
            if state_id in state_by_id and state_by_id[state_id].kind == "plan_node"
        }
        if preferred:
            return tuple(sorted(preferred))
        return tuple(
            sorted(
                state_id
                for state_id in implicated
                if state_id in state_by_id and state_by_id[state_id].kind == "global_goal"
            )
        )
    return ()


def drift_lineage_pairs(
    spec: SoftwareMem0VerticalSpec | EpisodePlan,
    sceu: SCEU,
) -> tuple[tuple[str, str], ...]:
    """Return ``(drift category, state lineage)`` pairs for one SCEU.

    A category is not itself a longitudinal identity.  The same persistent
    constraint, goal, plan, or scoped-decision lineage must be observable at
    both the adherence and violation checkpoints before drift onset can be
    assigned.  Candidate lineages are derived from the invalid actions that
    make each preregistered category expressible at this exact opportunity.
    """

    plan = _plan(spec)
    opportunity = next(
        item for item in plan.opportunities if item.opportunity_id == sceu.opportunity_id
    )
    eligible = set(drift_eligible_categories(spec, sceu))
    state_by_id = {item.state_id: item for item in plan.state_units}
    lineages: set[tuple[str, str]] = set()
    for action in opportunity.action_catalog:
        if action.action_id in set(opportunity.valid_action_ids):
            continue
        assessment = assess_software_action(
            plan,
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
        categories = set(
            normalized_action_drift(
                spec,
                action,
                synthetic,
                sceu.checkpoint_session,
            )
        ).intersection(eligible)
        for category in categories:
            for state_id in _lineage_candidate_ids(
                category=category,
                action=action,
                violated_state_ids=assessment.violated_state_ids,
                future_state_ids=assessment.future_state_ids,
                state_by_id=state_by_id,
            ):
                lineages.add((category, _state_lineage_id(state_by_id[state_id])))
    return tuple(sorted(lineages))


def normalized_action_drift(
    spec: SoftwareMem0VerticalSpec | EpisodePlan,
    action: ActionSpec,
    behavior: BehaviorResult,
    checkpoint_session: int,
) -> tuple[str, ...]:
    """Map checker evidence to the four preregistered drift constructs."""
    plan = _plan(spec)
    current = replay_plan(plan, checkpoint_session).current
    state_by_id = {item.state_id: item for item in plan.state_units}
    stale = tuple(
        item.state_id
        for item in plan.state_units
        if item.valid_from <= checkpoint_session and item.state_id not in current
    )
    future = tuple(
        item.state_id
        for item in plan.state_units
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
            action_expressed_state_ids=action.satisfies_state_ids,
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
    spec: SoftwareMem0VerticalSpec | EpisodePlan,
    sceu: SCEU,
) -> tuple[str, ...]:
    """Return the drift constructs intentionally targeted by one opportunity."""
    plan = _plan(spec)
    opportunity = next(
        item for item in plan.opportunities if item.opportunity_id == sceu.opportunity_id
    )
    valid = set(opportunity.valid_action_ids)
    expressible: set[str] = set()
    for action in opportunity.action_catalog:
        if action.action_id in valid:
            continue
        assessment = assess_software_action(
            plan,
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
        current = replay_plan(plan, sceu.checkpoint_session).current
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
    elif opportunity.challenge_type == "longitudinal-recovery-reminder":
        intended = set(CANONICAL_DRIFT_CATEGORIES)
    else:
        intended = expressible
    return tuple(sorted(expressible.intersection(intended)))


__all__ = [
    "CANONICAL_DRIFT_CATEGORIES",
    "DRIFT_LINEAGE_EVIDENCE_MODE",
    "drift_eligible_categories",
    "drift_lineage_pairs",
    "normalized_action_drift",
]
