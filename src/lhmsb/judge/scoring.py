"""Bounded composite scoring, calibration, and consistency for the judge.

This module enforces the discipline that the judge NEVER dominates a headline
metric: its contribution to any composite is capped (default 0.20) and reported as
a separate field.  It also provides a calibration harness (agreement against a gold
set) and a repeat-stability (consistency) check.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from lhmsb.judge.judge import Judge
from lhmsb.judge.rubric import Rubric
from lhmsb.types import Probe


# --------------------------------------------------------------------------- #
# Bounded composite
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompositeScore:
    """A composite of a deterministic score and a (bounded) judge score.

    The judge's weight is capped at ``max_judge_weight``; ``judge_contribution`` is
    the applied weight, reported SEPARATELY so the judge's influence is always
    explicit and auditable.  The judge can never set the composite by itself.
    """

    deterministic_score: float
    judge_score: float
    requested_judge_weight: float
    applied_judge_weight: float
    composite: float
    judge_contribution: float
    capped: bool


def combine_scores(
    deterministic_score: float,
    judge_score: float,
    *,
    requested_judge_weight: float,
    max_judge_weight: float = 0.20,
) -> CompositeScore:
    """Combine a deterministic score with a judge score under a hard weight cap.

    Args:
        deterministic_score: Programmatic score in [0, 1].
        judge_score: Judge score in [0, 1].
        requested_judge_weight: Desired judge weight in [0, 1] (e.g. a composite that
            tries to set 100% judge passes 1.0).
        max_judge_weight: Hard cap on the judge weight (default 0.20).

    Returns:
        A :class:`CompositeScore` whose ``applied_judge_weight`` never exceeds the cap.

    Raises:
        ValueError: if any score is outside [0, 1] or any weight is outside [0, 1].
    """
    for name, value in (
        ("deterministic_score", deterministic_score),
        ("judge_score", judge_score),
        ("requested_judge_weight", requested_judge_weight),
        ("max_judge_weight", max_judge_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    applied = min(requested_judge_weight, max_judge_weight)
    capped = requested_judge_weight > max_judge_weight
    composite = (1.0 - applied) * deterministic_score + applied * judge_score
    return CompositeScore(
        deterministic_score=deterministic_score,
        judge_score=judge_score,
        requested_judge_weight=requested_judge_weight,
        applied_judge_weight=applied,
        composite=composite,
        judge_contribution=applied,
        capped=capped,
    )


# --------------------------------------------------------------------------- #
# Calibration harness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationExample:
    """One gold calibration triple: a probe, an answer, and the expected score."""

    probe: Probe
    answer: str
    expected_score: float


@dataclass(frozen=True)
class CalibrationItem:
    """Per-example calibration outcome."""

    probe_id: str
    expected: float
    actual: float
    abs_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class CalibrationResult:
    """Aggregate calibration outcome over a gold set."""

    n: int
    agreement: float
    mean_abs_error: float
    tolerance: float
    items: tuple[CalibrationItem, ...]


def calibrate(
    judge: Judge,
    rubric: Rubric,
    examples: Sequence[CalibrationExample],
    *,
    tolerance: float = 0.1,
) -> CalibrationResult:
    """Run the judge over a gold set and report agreement + mean absolute error.

    Agreement is the fraction of examples whose judge score is within ``tolerance``
    of the gold expected score.

    Raises:
        ValueError: if ``examples`` is empty.
    """
    if not examples:
        raise ValueError("calibrate requires a non-empty gold set")

    items: list[CalibrationItem] = []
    errors: list[float] = []
    for example in examples:
        verdict = judge.score(example.probe, example.answer, rubric)
        abs_error = abs(verdict.score - example.expected_score)
        errors.append(abs_error)
        items.append(
            CalibrationItem(
                probe_id=example.probe.probe_id,
                expected=example.expected_score,
                actual=verdict.score,
                abs_error=abs_error,
                within_tolerance=abs_error <= tolerance,
            )
        )

    agreement = sum(1 for item in items if item.within_tolerance) / len(items)
    return CalibrationResult(
        n=len(items),
        agreement=agreement,
        mean_abs_error=statistics.fmean(errors),
        tolerance=tolerance,
        items=tuple(items),
    )


# --------------------------------------------------------------------------- #
# Consistency (repeat-stability)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConsistencyResult:
    """Repeat-stability of the judge on a single input."""

    scores: tuple[float, ...]
    mean: float
    stdev: float
    max_spread: float
    is_stable: bool


def judge_consistency(
    judge: Judge,
    probe: Probe,
    answer: str,
    rubric: Rubric,
    *,
    repeats: int = 5,
    epsilon: float = 1e-9,
) -> ConsistencyResult:
    """Re-score the same input ``repeats`` times and report score spread.

    A deterministic judge yields zero spread (perfectly stable).  ``is_stable`` is
    True when the max-min spread is within ``epsilon``.

    Raises:
        ValueError: if ``repeats`` < 2 (need at least two samples to measure spread).
    """
    if repeats < 2:
        raise ValueError("judge_consistency requires repeats >= 2")

    scores = tuple(judge.score(probe, answer, rubric).score for _ in range(repeats))
    max_spread = max(scores) - min(scores)
    return ConsistencyResult(
        scores=scores,
        mean=statistics.fmean(scores),
        stdev=statistics.pstdev(scores),
        max_spread=max_spread,
        is_stable=max_spread <= epsilon,
    )
