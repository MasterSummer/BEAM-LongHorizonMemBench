"""Deterministic decomposition of an ``Episode`` into ordered execution steps.

An episode interleaves exogenous ``WorldEvent``s and ``Probe``s along a single
integer ``step`` axis (``spec/03-protocol.md`` §1.3-1.4). The harness executes
one :class:`Step` at a time, in a fixed order, resetting the agent's working
context at every session boundary so that the only cross-session channel is the
memory adapter.

Session assignment:
  * An event's session is ``int(event.payload["session"])`` (default ``0``).
  * A probe inherits the session of the latest event at or before its step
    (default ``0`` when no event precedes it).

Ordering: ``(step, events-before-probes)`` so a fact injected at step ``S`` is
written before a probe answered at step ``S``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lhmsb.types import Episode, Probe, WorldEvent


def _coerce_session(value: object) -> int:
    """Read a session index from an event payload value (non-int -> 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


@dataclass(frozen=True)
class Step:
    """One unit of episode execution: exactly one of ``event`` / ``probe`` is set."""

    step: int
    session_index: int
    event: WorldEvent | None = None
    probe: Probe | None = None


def _event_session(event: WorldEvent) -> int:
    return _coerce_session(event.payload.get("session", 0))


def plan_steps(episode: Episode) -> list[Step]:
    """Return the episode's steps in deterministic execution order."""
    sorted_events = sorted(episode.events, key=lambda e: e.step)

    def session_for_step(step: int) -> int:
        session = 0
        for event in sorted_events:
            if event.step <= step:
                session = _event_session(event)
            else:
                break
        return session

    steps: list[Step] = [
        Step(step=event.step, session_index=_event_session(event), event=event)
        for event in episode.events
    ]
    steps.extend(
        Step(step=probe.step, session_index=session_for_step(probe.step), probe=probe)
        for probe in episode.probes
    )
    steps.sort(key=lambda s: (s.step, 0 if s.event is not None else 1))
    return steps
