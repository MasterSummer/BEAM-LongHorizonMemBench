"""Shared simulator core for LongHorizonMemSysBench.

Public API (all from :mod:`lhmsb.sim.core`):
  - :class:`Fact` / :class:`WorldState` — the fixed exogenous world & replay.
  - :class:`ProbeSpec` / :class:`FamilyContent` / :class:`ScaleParams` — episode
    inputs prior to gold derivation.
  - :class:`EpisodeBuilder` — validates a schedule and derives probe gold from
    the revealed-minus-retracted world state, returning an :class:`lhmsb.types.Episode`.
  - :class:`Checker` / :class:`CheckResult` / :class:`DefaultChecker` — programmatic grading.
  - :class:`SurfaceRenderer` / :class:`StubRenderer` / :class:`RenderCache` /
    :func:`render_episode` — deterministic, frozen-cached surface rendering.
  - :func:`validate_render` — render-vs-ground-truth leak/contradiction guard.
  - :class:`ScheduleError` / :class:`RenderCacheError` / :class:`RenderValidationError`.
"""

from __future__ import annotations

from lhmsb.sim.core import (
    Checker,
    CheckResult,
    DefaultChecker,
    Derivation,
    EpisodeBuilder,
    Fact,
    FamilyContent,
    ProbeKind,
    ProbeSpec,
    RenderCache,
    RenderCacheError,
    RenderValidationError,
    ScaleParams,
    ScheduleError,
    StubRenderer,
    SurfaceRenderer,
    WorldState,
    render_episode,
    validate_render,
)

__all__ = [
    "Checker",
    "CheckResult",
    "DefaultChecker",
    "Derivation",
    "EpisodeBuilder",
    "Fact",
    "FamilyContent",
    "ProbeKind",
    "ProbeSpec",
    "RenderCache",
    "RenderCacheError",
    "RenderValidationError",
    "ScaleParams",
    "ScheduleError",
    "StubRenderer",
    "SurfaceRenderer",
    "WorldState",
    "render_episode",
    "validate_render",
]
