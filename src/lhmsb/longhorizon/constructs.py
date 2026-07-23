"""Evaluator-side profiles for the constructs that make a task long-horizon.

The profiler is deliberately derived from an already-frozen :class:`EpisodePlan`.
It does not add fields to the public surface or change the plan hash.  This lets
existing frozen releases be re-analysed while making session handoffs, delayed
state need, dependency depth, state evolution, and workspace recoverability
explicit report variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import SCEU, ContinuationOpportunity, EpisodePlan, StateUnit

ConstructKind = Literal[
    "static_recall",
    "state_evolution",
    "hierarchical_conflict",
    "fresh_control",
]
HorizonBand = Literal["short", "medium", "long"]

_EVOLUTION_EVENTS = {
    "replace",
    "revoke",
    "expire",
    "reopen",
    "priority_change",
    "scope_change",
    "invalidate",
}
_EVOLUTION_CHALLENGES = {
    "premature-v2",
    "stale-after-revoke",
    "valid-update",
    "matched-branch",
    "matched-evolution-terminal",
}
_CONFLICT_CHALLENGES = {
    "scope-conflict",
    "valid-local-accelerator",
    "authority-scoped-exception",
    "global-local-conflict",
    "matched-conflict-terminal",
}
_FRESH_CONTROL_CHALLENGES = {
    "fresh-current-v1-reminder",
    "fresh-reminder",
    "longitudinal-recovery-reminder",
}


@dataclass(frozen=True)
class LongHorizonConstructProfile:
    """Derived construct variables for one state-conditioned decision."""

    episode_id: str
    sceu_id: str
    opportunity_id: str
    construct_kind: ConstructKind
    horizon_band: HorizonBand
    checkpoint_session: int
    handoff_count: int
    current_required_state_ids: tuple[str, ...]
    current_action_relevant_state_ids: tuple[str, ...]
    missing_current_action_relevant_state_ids: tuple[str, ...]
    memory_reliant_state_ids: tuple[str, ...]
    nonexplicit_state_ids: tuple[str, ...]
    future_referenced_state_ids: tuple[str, ...]
    invalidated_referenced_state_ids: tuple[str, ...]
    oldest_required_state_age: int | None
    newest_required_state_age: int | None
    latest_decision_event_distance: int | None
    dependency_depth: int
    relevant_event_count: int
    relevant_transition_count: int
    workspace_explicit_count: int
    workspace_derivable_count: int
    workspace_absent_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def horizon_band(handoff_count: int) -> HorizonBand:
    """Return the preregistered absolute handoff band.

    Bands use decision boundaries rather than token counts: 0--2 handoffs are
    short, 3--7 are medium, and 8 or more are long.
    """

    if handoff_count < 0:
        raise ValueError("handoff_count must be non-negative")
    if handoff_count <= 2:
        return "short"
    if handoff_count <= 7:
        return "medium"
    return "long"


def profile_sceu(plan: EpisodePlan, sceu: SCEU) -> LongHorizonConstructProfile:
    """Derive a leak-free evaluator profile for ``sceu``.

    Required current state is recomputed from *active focal states*.  This is
    important for premature-update probes: a future state may be referenced by
    the evaluator to detect anticipation, but it must not become a state that a
    memory system was expected to have stored before it existed.
    """

    if sceu.episode_id != plan.episode_id:
        raise ValueError("SCEU episode does not match plan")
    opportunities = {item.opportunity_id: item for item in plan.opportunities}
    try:
        opportunity = opportunities[sceu.opportunity_id]
    except KeyError as exc:
        raise ValueError(f"unknown SCEU opportunity: {sceu.opportunity_id}") from exc
    if opportunity.checkpoint_session != sceu.checkpoint_session:
        raise ValueError("SCEU checkpoint does not match continuation opportunity")

    state_map = {state.state_id: state for state in plan.state_units}
    replay = replay_plan(plan, sceu.checkpoint_session)
    active_focal = tuple(
        state_id for state_id in opportunity.focal_state_ids if state_id in replay.current
    )
    current_required = _dependency_closure(active_focal, state_map)
    valid_action_ids = set(opportunity.valid_action_ids)
    action_relevant = {
        state_id
        for action in opportunity.action_catalog
        for state_id in (
            *action.violates_state_ids,
            *(
                action.satisfies_state_ids
                if action.action_id in valid_action_ids
                else ()
            ),
        )
    }
    current_action_relevant = tuple(
        sorted(
            state_id
            for state_id in action_relevant.intersection(replay.current)
            if _state_applies_to_opportunity(
                state_map[state_id],
                opportunity,
            )
        )
    )
    missing_current_action_relevant = tuple(
        sorted(set(current_action_relevant).difference(current_required))
    )
    declared = set(sceu.dependency_closure) | set(sceu.required_state_ids)
    future = tuple(
        sorted(
            state_id
            for state_id in declared
            if state_id in state_map and state_map[state_id].valid_from > sceu.checkpoint_session
        )
    )
    recoverability = dict(sceu.workspace_recoverability)
    memory_reliant = tuple(
        state_id
        for state_id in current_required
        if recoverability.get(state_id, "absent") == "absent"
    )
    nonexplicit = tuple(
        state_id
        for state_id in current_required
        if recoverability.get(state_id, "absent") != "explicit"
    )
    workspace_values = [recoverability.get(state_id, "absent") for state_id in current_required]

    decision_state_ids = _decision_state_ids(opportunity, sceu)
    invalidated = tuple(sorted(decision_state_ids.intersection(replay.invalidated)))
    relevant_events = tuple(
        event
        for event in plan.events
        if event.session <= sceu.checkpoint_session
        and (
            event.target_state_id in decision_state_ids
            or bool(decision_state_ids.intersection(event.reason_state_ids))
            or bool(decision_state_ids.intersection(event.invalidates))
        )
    )
    transitions = tuple(event for event in relevant_events if event.type in _EVOLUTION_EVENTS)
    ages = tuple(
        sceu.checkpoint_session - state_map[state_id].valid_from for state_id in current_required
    )
    latest_event_session = max(
        (event.session for event in relevant_events),
        default=None,
    )
    depth_by_state = _dependency_depths(state_map)

    return LongHorizonConstructProfile(
        episode_id=plan.episode_id,
        sceu_id=sceu.sceu_id,
        opportunity_id=sceu.opportunity_id,
        construct_kind=_construct_kind(opportunity, bool(transitions)),
        horizon_band=horizon_band(sceu.checkpoint_session),
        checkpoint_session=sceu.checkpoint_session,
        handoff_count=sceu.checkpoint_session,
        current_required_state_ids=current_required,
        current_action_relevant_state_ids=current_action_relevant,
        missing_current_action_relevant_state_ids=(
            missing_current_action_relevant
        ),
        memory_reliant_state_ids=memory_reliant,
        nonexplicit_state_ids=nonexplicit,
        future_referenced_state_ids=future,
        invalidated_referenced_state_ids=invalidated,
        oldest_required_state_age=max(ages, default=None),
        newest_required_state_age=min(ages, default=None),
        latest_decision_event_distance=(
            None if latest_event_session is None else sceu.checkpoint_session - latest_event_session
        ),
        dependency_depth=max(
            (depth_by_state[state_id] for state_id in current_required),
            default=0,
        ),
        relevant_event_count=len(relevant_events),
        relevant_transition_count=len(transitions),
        workspace_explicit_count=workspace_values.count("explicit"),
        workspace_derivable_count=workspace_values.count("derivable"),
        workspace_absent_count=workspace_values.count("absent"),
    )


def _dependency_closure(
    roots: tuple[str, ...],
    state_map: dict[str, StateUnit],
) -> tuple[str, ...]:
    closure: set[str] = set()
    queue = list(roots)
    while queue:
        state_id = queue.pop(0)
        if state_id in closure:
            continue
        state = state_map.get(state_id)
        if state is None:
            raise ValueError(f"unknown state in dependency closure: {state_id}")
        closure.add(state_id)
        queue.extend(state.dependency_ids)
    return tuple(sorted(closure))


def _dependency_depths(state_map: dict[str, StateUnit]) -> dict[str, int]:
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(state_id: str) -> int:
        if state_id in depths:
            return depths[state_id]
        if state_id in visiting:
            raise ValueError(f"dependency cycle at state: {state_id}")
        try:
            state = state_map[state_id]
        except KeyError as exc:
            raise ValueError(f"unknown dependency state: {state_id}") from exc
        visiting.add(state_id)
        value = (
            0 if not state.dependency_ids else 1 + max(depth(item) for item in state.dependency_ids)
        )
        visiting.remove(state_id)
        depths[state_id] = value
        return value

    for state_id in state_map:
        depth(state_id)
    return depths


def _decision_state_ids(
    opportunity: ContinuationOpportunity,
    sceu: SCEU,
) -> set[str]:
    result = (
        set(opportunity.focal_state_ids)
        | set(sceu.required_state_ids)
        | set(sceu.dependency_closure)
    )
    for action in opportunity.action_catalog:
        result.update(action.satisfies_state_ids)
        result.update(action.violates_state_ids)
    return result


def _state_applies_to_opportunity(
    state: StateUnit,
    opportunity: ContinuationOpportunity,
) -> bool:
    """Return whether a scoped state can govern this continuation.

    The Software family currently has one explicit scoped-exception domain.
    Keeping this rule evaluator-side prevents an isolated-profiler exception
    from being counted as required state for governed project execution.
    """

    if state.scope == "isolated-local-profiler":
        return opportunity.continuation_scope == "isolated_profiler"
    return True


def _construct_kind(
    opportunity: ContinuationOpportunity,
    has_transition: bool,
) -> ConstructKind:
    challenge = opportunity.challenge_type
    if challenge in _CONFLICT_CHALLENGES:
        return "hierarchical_conflict"
    if challenge in _FRESH_CONTROL_CHALLENGES:
        return "fresh_control"
    if challenge in _EVOLUTION_CHALLENGES and (has_transition or challenge != "matched-branch"):
        return "state_evolution"
    if has_transition:
        return "state_evolution"
    return "static_recall"


__all__ = [
    "ConstructKind",
    "HorizonBand",
    "LongHorizonConstructProfile",
    "horizon_band",
    "profile_sceu",
]
