"""Scorecard / leaderboard reporting for LongHorizonMemSysBench (task 24).

Public API:
  - :class:`Scorecard` — the full scorecard (native + controlled, never merged).
  - :class:`TrackResults` — ROI + aggregated statistics for one track.
  - :class:`BareNumberError` — raised when a bare scalar winner is requested.
  - :func:`generate_scorecard` — produce Markdown + CSV + JSON + Pareto plots.
  - :func:`render_bare_number_guard` — API guard that raises BareNumberError.
"""

from __future__ import annotations

from lhmsb.report.scorecard import (
    BareNumberError,
    Scorecard,
    TrackResults,
    generate_scorecard,
    render_bare_number_guard,
)

__all__ = [
    "BareNumberError",
    "Scorecard",
    "TrackResults",
    "generate_scorecard",
    "render_bare_number_guard",
]
