"""Render latent state plans into an agent-visible, gold-free surface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import EpisodePlan, SessionSurface, WorkspaceSnapshot


def _state_text(value: object) -> str:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        return "; ".join(f"{key}: {item}" for key, item in sorted(value.items()))
    return str(value)


def _event_observation(plan: EpisodePlan, event_type: str, target: str) -> str:
    state = next(state for state in plan.state_units if state.state_id == target)
    text = _state_text(state.value)
    if event_type == "add":
        return f"Project update: {text}"
    if event_type in {"replace", "priority_change", "scope_change"}:
        return f"Project update: an existing decision changed — {text}"
    if event_type in {"revoke", "expire", "invalidate"}:
        return f"Project update: an earlier decision is no longer valid — {text}"
    if event_type == "reopen":
        return f"Project update: a previously closed item is active again — {text}"
    return "Project update recorded."


def _surface_workspace(workspace: WorkspaceSnapshot) -> WorkspaceSnapshot:
    """Strip evaluator provenance and recoverability labels from agent surface."""
    return replace(
        workspace,
        artifacts=tuple(
            replace(artifact, source_event_ids=(), memory_owned=False)
            for artifact in workspace.artifacts
        ),
        recoverability_by_state=(),
    )


def render_surfaces(plan: EpisodePlan) -> tuple[SessionSurface, ...]:
    """Render observations/tool results/workspace for every isolated session.

    Only states whose events have occurred are rendered.  State IDs, validity
    labels, dependency graph metadata, provenance IDs, and future values remain
    evaluator-side and never enter the returned surfaces.
    """
    state_map = {state.state_id: state for state in plan.state_units}
    if any(event.target_state_id not in state_map for event in plan.events):
        raise ValueError("cannot render an event targeting an unknown state")
    surfaces: list[SessionSurface] = []
    for session in range(plan.n_sessions):
        replay = replay_plan(plan, session)
        observations = [
            "You are continuing an offline software project. "
            "Inspect the workspace and preserve the project goal."
        ]
        for event in sorted(plan.events, key=lambda item: (item.session, item.event_id)):
            if event.session > session:
                break
            if event.session == session:
                observations.append(_event_observation(plan, event.type, event.target_state_id))
        if session and "P2" in replay.current:
            observations.append(
                "The current implementation branch has passed the latest local audit."
            )
        tool_results = [
            f"workspace listing available for session {session}",
            f"local test runner completed with {len(replay.current)} active project facts",
        ]
        # Deliberately avoid even mentioning IDs in tool output.  The local
        # count is a harmless surface statistic, not evaluator metadata.
        workspace = plan.workspaces[session]
        surfaces.append(
            SessionSurface(
                session_index=session,
                observations=tuple(observations),
                tool_results=tuple(tool_results),
                workspace=_surface_workspace(workspace),
            )
        )
    return tuple(surfaces)


def _surface_dict(surface: SessionSurface) -> dict[str, object]:
    return asdict(surface)


def surfaces_hash(surfaces: tuple[SessionSurface, ...]) -> str:
    """Return a stable hash of the exact agent-visible surface."""
    payload = json.dumps(
        [_surface_dict(surface) for surface in surfaces],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["render_surfaces", "surfaces_hash"]
