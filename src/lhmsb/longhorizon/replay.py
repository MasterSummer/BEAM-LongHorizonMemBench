"""Deterministic replay and hashing for state-first episode plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from lhmsb.longhorizon.schema import EpisodePlan, StateEvent, StateUnit


class StateReplayError(ValueError):
    """Raised when a state event cannot be applied to an episode plan."""


@dataclass(frozen=True)
class ReplayResult:
    """Current and historical state after replaying through one session."""

    session_index: int
    current: dict[str, StateUnit]
    history: dict[str, tuple[int, ...]]
    invalidated: frozenset[str]


def plan_hash(plan: EpisodePlan) -> str:
    """Return a stable SHA-256 hash over the canonical latent plan JSON."""
    payload = json.dumps(plan.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_reasons(event: StateEvent, states: dict[str, StateUnit]) -> None:
    for state_id in (*event.reason_state_ids, *event.invalidates):
        if state_id not in states:
            raise StateReplayError(f"unknown state referenced by {event.event_id}: {state_id}")


def _validate_dependency_graph(states: dict[str, StateUnit]) -> dict[str, int]:
    """Reject unknown dependency IDs and cycles before replay starts."""
    for state in states.values():
        for dependency_id in state.dependency_ids:
            if dependency_id not in states:
                raise StateReplayError(
                    f"dependency closure for {state.state_id} references unknown state: "
                    f"{dependency_id}"
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state_id: str) -> None:
        if state_id in visiting:
            raise StateReplayError(f"dependency closure contains a cycle at {state_id}")
        if state_id in visited:
            return
        visiting.add(state_id)
        for dependency_id in states[state_id].dependency_ids:
            visit(dependency_id)
        visiting.remove(state_id)
        visited.add(state_id)

    for state_id in states:
        visit(state_id)
    depths: dict[str, int] = {}

    def depth(state_id: str) -> int:
        if state_id in depths:
            return depths[state_id]
        dependencies = states[state_id].dependency_ids
        depths[state_id] = 0 if not dependencies else 1 + max(depth(item) for item in dependencies)
        return depths[state_id]

    for state_id in states:
        depth(state_id)
    return depths


def _check_valid_window(event: StateEvent, target: StateUnit) -> None:
    if event.session < target.valid_from or (
        target.valid_to is not None and event.session > target.valid_to
    ):
        raise StateReplayError(
            f"event {event.event_id} is outside valid window for {target.state_id}: "
            f"[{target.valid_from}, {target.valid_to}]"
        )


def _check_dependencies(
    event: StateEvent, target: StateUnit, current: dict[str, StateUnit]
) -> None:
    missing = [
        dependency_id for dependency_id in target.dependency_ids if dependency_id not in current
    ]
    if missing:
        raise StateReplayError(
            f"dependency closure for {target.state_id} is not replayable at session "
            f"{event.session}: missing {missing}"
        )


def _cascade_invalidations(
    invalidated_ids: set[str],
    current: dict[str, StateUnit],
    states: dict[str, StateUnit],
) -> None:
    """Remove active dependents of invalidated states and record their IDs."""
    queue = list(invalidated_ids)
    while queue:
        parent = queue.pop(0)
        for state_id, state in states.items():
            if state_id not in current or state_id in invalidated_ids:
                continue
            if parent in state.dependency_ids:
                current.pop(state_id, None)
                invalidated_ids.add(state_id)
                queue.append(state_id)


def _versioned_state(state: StateUnit, event: StateEvent, version: int) -> StateUnit:
    """Apply event authority/scope and version metadata to a state snapshot."""
    return replace(
        state,
        version=version,
        authority=event.authority or state.authority,
        scope=event.scope or state.scope,
    )


def _apply_event(
    event: StateEvent,
    states: dict[str, StateUnit],
    current: dict[str, StateUnit],
    history: dict[str, list[int]],
    invalidated: set[str],
) -> None:
    """Apply one event, raising instead of silently repairing malformed plans."""
    _check_reasons(event, states)
    target = states.get(event.target_state_id)
    if target is None:
        raise StateReplayError(f"unknown state target: {event.target_state_id}")
    _check_valid_window(event, target)
    active = current.get(event.target_state_id)
    transition = event.type

    if transition == "add":
        if active is not None:
            raise StateReplayError(f"state already active: {event.target_state_id}")
        _check_dependencies(event, target, current)
        version = event.new_version or target.version
        current[event.target_state_id] = _versioned_state(target, event, version)
        history.setdefault(event.target_state_id, []).append(version)
        invalidated.discard(event.target_state_id)
        return

    if transition == "reopen":
        if active is not None:
            raise StateReplayError(f"state already active for reopen: {event.target_state_id}")
        _check_dependencies(event, target, current)
        version = event.new_version or (
            history.get(event.target_state_id, [target.version])[-1] + 1
        )
        current[event.target_state_id] = _versioned_state(target, event, version)
        history.setdefault(event.target_state_id, []).append(version)
        invalidated.discard(event.target_state_id)
        return

    if active is None:
        raise StateReplayError(f"state is not active for {transition}: {event.target_state_id}")
    if event.old_version is not None and active.version != event.old_version:
        raise StateReplayError(
            f"version mismatch for {event.target_state_id}: expected {event.old_version}, "
            f"current {active.version}"
        )

    if transition in {"replace", "priority_change", "scope_change"}:
        _check_dependencies(event, target, current)
        version = event.new_version or active.version + 1
        if version <= active.version:
            raise StateReplayError(f"new version must increase for {event.target_state_id}")
        current[event.target_state_id] = _versioned_state(target, event, version)
        history.setdefault(event.target_state_id, []).append(version)
        invalidated.update(event.invalidates)
        _cascade_invalidations(invalidated, current, states)
        return

    if transition in {"revoke", "expire", "invalidate"}:
        current.pop(event.target_state_id, None)
        invalidated.add(event.target_state_id)
        invalidated.update(event.invalidates)
        _cascade_invalidations(invalidated, current, states)
        return

    raise StateReplayError(f"unsupported state event type: {transition}")


def replay_plan(plan: EpisodePlan, session_index: int) -> ReplayResult:
    """Replay all events through ``session_index`` and return current state."""
    if session_index < 0 or session_index >= plan.n_sessions:
        raise StateReplayError(f"session_index {session_index} outside [0, {plan.n_sessions - 1}]")
    states = {state.state_id: state for state in plan.state_units}
    depths = _validate_dependency_graph(states)
    for event in plan.events:
        if event.target_state_id not in states:
            raise StateReplayError(f"unknown state target: {event.target_state_id}")
    current: dict[str, StateUnit] = {}
    history: dict[str, list[int]] = {}
    invalidated: set[str] = set()
    for event in sorted(
        plan.events,
        key=lambda item: (item.session, depths.get(item.target_state_id, -1), item.event_id),
    ):
        if event.session > session_index:
            break
        _apply_event(event, states, current, history, invalidated)
    return ReplayResult(
        session_index=session_index,
        current=dict(current),
        history={state_id: tuple(versions) for state_id, versions in history.items()},
        invalidated=frozenset(invalidated),
    )
