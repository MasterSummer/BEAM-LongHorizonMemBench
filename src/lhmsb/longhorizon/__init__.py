"""State-first long-horizon benchmark primitives.

This package is additive: the original v1 episode protocol remains available
under :mod:`lhmsb.types` and is not changed by the vertical slice.
"""

from lhmsb.longhorizon.render import render_surfaces, surfaces_hash
from lhmsb.longhorizon.replay import ReplayResult, StateReplayError, plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    SCEU,
    ActionSpec,
    ContinuationOpportunity,
    ControlKind,
    EpisodePlan,
    SessionSurface,
    StateEvent,
    StateEventKind,
    StateKind,
    StateUnit,
    WorkspaceArtifact,
    WorkspaceRecoverability,
    WorkspaceSnapshot,
)

__all__ = [
    "ActionSpec",
    "ControlKind",
    "ContinuationOpportunity",
    "EpisodePlan",
    "ReplayResult",
    "SCEU",
    "SessionSurface",
    "StateEvent",
    "StateEventKind",
    "StateKind",
    "StateReplayError",
    "StateUnit",
    "WorkspaceArtifact",
    "WorkspaceSnapshot",
    "WorkspaceRecoverability",
    "plan_hash",
    "replay_plan",
    "render_surfaces",
    "surfaces_hash",
]
