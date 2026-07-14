"""TDD tests for Dim-2 task-performance + goal-directed utilization (lhmsb.metrics).

Written FIRST (RED) before the implementation. The canonical definitions live in
``spec/02-metrics.md`` §4:

  - ``task_score`` = (Σ deterministic_probe_scores + Σ judge_probe_scores) / |probes|,
    normalized to [0, 1]; the judge's weight in the composite is capped (≤ 0.20) so
    the judge can never dominate the deterministic signal.
  - ``utilization_rate`` = |correct cross-session probes| / |cross-session probes|.
    Probes whose required facts are available in the CURRENT session's context are
    excluded — they don't test memory. No cross-session probes → N/A.
  - ``improvement_over_time`` = trend (least-squares slope) of per-session task score.
  - ``judge_contribution`` is reported as a SEPARATE, bounded field.

The headline behavioural contract (task-17 acceptance):
  a fixture where exactly 2 of 4 probes require cross-session recall →
  FakePerfect utilization = 1.0, NoMemory = 0.0, partial = 0.5; and a
  context-available probe answered correctly under NoMemory does NOT inflate
  utilization (the context guard).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Literal

import pytest

from lhmsb.metrics import (
    TaskScore,
    improvement_over_time,
    score_task,
    utilization_rate,
)
from lhmsb.types import Condition, Episode, EpisodeResult, Probe, ProbeResult, WorldEvent

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
EPISODE_ID = "ep-fixture"
SEED = 7


def _probe(
    probe_id: str,
    *,
    cross_session: bool,
    step: int = 0,
    kind: Literal["factual", "synthesis", "behavioral"] = "factual",
    gold: object = "gold",
) -> Probe:
    """A probe with an explicit cross_session flag (the memory-vs-context signal)."""
    return Probe(
        step=step,
        probe_id=probe_id,
        kind=kind,
        query=f"query for {probe_id}",
        gold=gold,
        cross_session=cross_session,
    )


def _episode(probes: list[Probe], events: list[WorldEvent] | None = None) -> Episode:
    return Episode(
        episode_id=EPISODE_ID,
        family="research",
        seed=SEED,
        events=events or [],
        probes=probes,
    )


def _result(name: str, scores: dict[str, float]) -> EpisodeResult:
    """An EpisodeResult under condition ``name``; is_correct == (score >= 1.0)."""
    return EpisodeResult(
        episode_id=EPISODE_ID,
        condition=Condition(name=name),
        seed=SEED,
        probe_results=[
            ProbeResult(probe_id=pid, score=score, is_correct=score >= 1.0)
            for pid, score in scores.items()
        ],
    )


# The headline fixture: exactly 2 of 4 probes require cross-session recall.
#   p1, p3 -> context-available (cross_session=False, do NOT test memory)
#   p2, p4 -> cross-session-dependent (cross_session=True, DO test memory)
FOUR_PROBES = [
    _probe("p1", cross_session=False, step=1, gold="alpha"),
    _probe("p2", cross_session=True, step=2, gold="beta"),
    _probe("p3", cross_session=False, step=3, gold="gamma"),
    _probe("p4", cross_session=True, step=4, gold="delta"),
]

# FakePerfect answers every probe correctly.
FAKE_PERFECT = {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 1.0}
# NoMemory answers context probes (p1, p3) but misses BOTH cross-session probes.
NO_MEMORY = {"p1": 1.0, "p2": 0.0, "p3": 1.0, "p4": 0.0}
# A partial adapter recalls exactly one of the two cross-session facts (p2, not p4).
PARTIAL = {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 0.0}


@pytest.fixture
def episode() -> Episode:
    return _episode(FOUR_PROBES)


# --------------------------------------------------------------------------- #
# Utilization: isolates cross-session recall (the headline contract)
# --------------------------------------------------------------------------- #
def test_utilization_fake_perfect_is_one(episode: Episode) -> None:
    assert utilization_rate(_result("fake_perfect", FAKE_PERFECT), episode) == 1.0


def test_utilization_no_memory_is_zero(episode: Episode) -> None:
    assert utilization_rate(_result("no_memory", NO_MEMORY), episode) == 0.0


def test_utilization_partial_is_half(episode: Episode) -> None:
    assert utilization_rate(_result("partial", PARTIAL), episode) == 0.5


def test_score_task_utilization_matches_standalone(episode: Episode) -> None:
    """score_task(...).utilization_rate equals the standalone utilization_rate()."""
    for name, scores, expected in (
        ("fake_perfect", FAKE_PERFECT, 1.0),
        ("no_memory", NO_MEMORY, 0.0),
        ("partial", PARTIAL, 0.5),
    ):
        result = _result(name, scores)
        assert score_task(result, episode).utilization_rate == expected
        assert utilization_rate(result, episode) == expected


# --------------------------------------------------------------------------- #
# The context guard: context-available probes are NOT memory utilization
# --------------------------------------------------------------------------- #
def test_context_probe_correct_does_not_inflate_utilization(episode: Episode) -> None:
    """NoMemory answers BOTH context probes (p1, p3) correctly, yet utilization is 0.0.

    If context-available probes were (wrongly) counted as memory use, utilization
    would be 0.5 (2 of 4) instead of 0.0 (0 of 2 cross-session). It must be 0.0.
    """
    assert utilization_rate(_result("no_memory", NO_MEMORY), episode) == 0.0


def test_flipping_context_probe_leaves_utilization_unchanged(episode: Episode) -> None:
    """Toggling a context probe's correctness must not change utilization at all."""
    with_context_right = _result("a", {"p1": 1.0, "p2": 0.0, "p3": 1.0, "p4": 0.0})
    with_context_wrong = _result("b", {"p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0})
    assert utilization_rate(with_context_right, episode) == utilization_rate(
        with_context_wrong, episode
    )
    assert utilization_rate(with_context_right, episode) == 0.0


def test_no_cross_session_probes_is_na() -> None:
    """No cross-session probes → utilization_rate = N/A (None), never 0.0."""
    ep = _episode(
        [
            _probe("c1", cross_session=False, step=1),
            _probe("c2", cross_session=False, step=2),
        ]
    )
    res = _result("any", {"c1": 1.0, "c2": 1.0})
    assert utilization_rate(res, ep) is None
    assert score_task(res, ep).utilization_rate is None


# --------------------------------------------------------------------------- #
# Task score: normalized in [0, 1]
# --------------------------------------------------------------------------- #
def test_task_score_normalized_in_unit_interval(episode: Episode) -> None:
    for name, scores in (
        ("fake_perfect", FAKE_PERFECT),
        ("no_memory", NO_MEMORY),
        ("partial", PARTIAL),
    ):
        ts = score_task(_result(name, scores), episode)
        assert 0.0 <= ts.task_score <= 1.0


def test_task_score_hand_computed(episode: Episode) -> None:
    # All deterministic probes (no judge), so task_score == simple mean of scores.
    assert score_task(_result("fp", FAKE_PERFECT), episode).task_score == pytest.approx(1.0)
    assert score_task(_result("nm", NO_MEMORY), episode).task_score == pytest.approx(0.5)
    assert score_task(_result("pt", PARTIAL), episode).task_score == pytest.approx(0.75)


def test_missing_probe_result_scored_as_zero(episode: Episode) -> None:
    """A timed-out / crashed probe (no recorded result) is scored 0, not dropped."""
    # p4 is absent from the results → counts as 0 in the denominator of 4 probes.
    ts = score_task(_result("partial_missing", {"p1": 1.0, "p2": 1.0, "p3": 1.0}), episode)
    assert ts.task_score == pytest.approx(0.75)
    assert len(ts.per_probe) == 4


# --------------------------------------------------------------------------- #
# Judge contribution: SEPARATE + bounded, never dominates
# --------------------------------------------------------------------------- #
def _worked_example_episode() -> Episode:
    """spec/02-metrics.md §4.3 worked example: 6 probes, 1 open-ended (synthesis)."""
    probes = [
        _probe("p1", cross_session=False, step=1, gold="a"),
        _probe("p2", cross_session=True, step=3, gold="b"),
        _probe("p3", cross_session=False, step=3, gold="c"),
        _probe("p4", cross_session=True, step=5, gold="d"),
        _probe("p5", cross_session=True, step=5, gold="e"),
        _probe("p6", cross_session=False, step=5, kind="synthesis", gold="synth"),
    ]
    return _episode(probes)


def test_worked_example_matches_spec() -> None:
    """task_score ≈ 0.6167, utilization ≈ 0.6667 (spec §4.3 hand-computed values)."""
    ep = _worked_example_episode()
    res = _result(
        "worked",
        {"p1": 1.0, "p2": 1.0, "p3": 0.0, "p4": 0.0, "p5": 1.0, "p6": 0.7},
    )
    ts = score_task(res, ep)
    assert ts.task_score == pytest.approx(3.7 / 6.0, abs=1e-4)
    assert ts.utilization_rate == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_judge_contribution_is_a_separate_bounded_field() -> None:
    ep = _worked_example_episode()
    res = _result(
        "worked",
        {"p1": 1.0, "p2": 1.0, "p3": 0.0, "p4": 0.0, "p5": 1.0, "p6": 0.7},
    )
    ts = score_task(res, ep)
    # Separate, explicit field; bounded to the 20% cap; non-zero (one judge probe).
    assert 0.0 < ts.judge_contribution <= 0.20


def test_no_judge_probe_zero_contribution(episode: Episode) -> None:
    """A fully deterministic episode has judge_contribution == 0.0."""
    assert score_task(_result("fp", FAKE_PERFECT), episode).judge_contribution == 0.0


def test_judge_never_dominates_when_over_cap() -> None:
    """When >20% of probes are judge-scored, the judge weight is capped at 0.20.

    Deterministic probes both score 0.0; judge probes both score 1.0. A naive mean
    would be 0.5, but the cap pins the judge to 20% weight → task_score == 0.20.
    """
    ep = _episode(
        [
            _probe("d1", cross_session=False, step=1),
            _probe("d2", cross_session=False, step=2),
            _probe("j1", cross_session=False, step=3, kind="synthesis"),
            _probe("j2", cross_session=False, step=4, kind="synthesis"),
        ]
    )
    res = _result("judge_heavy", {"d1": 0.0, "d2": 0.0, "j1": 1.0, "j2": 1.0})
    ts = score_task(res, ep)
    assert ts.judge_contribution == pytest.approx(0.20)
    assert ts.task_score == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# Improvement over time: per-session trend
# --------------------------------------------------------------------------- #
def _sessioned_episode(session_scores: list[tuple[int, float]]) -> tuple[Episode, EpisodeResult]:
    """Build an episode whose probe i sits in session i with the given score."""
    events: list[WorldEvent] = []
    probes: list[Probe] = []
    scores: dict[str, float] = {}
    for idx, (session, _score) in enumerate(session_scores):
        step = idx * 2
        events.append(
            WorldEvent(step=step, kind="inject", fact_id=f"f{idx}", payload={"session": session})
        )
        pid = f"p{idx}"
        probes.append(_probe(pid, cross_session=False, step=step + 1))
        scores[pid] = session_scores[idx][1]
    return _episode(probes, events), _result("trend", scores)


def test_improvement_rises_with_memory() -> None:
    ep, res = _sessioned_episode([(0, 0.0), (1, 0.5), (2, 1.0)])
    trend = improvement_over_time(res, ep)
    assert trend is not None
    assert trend == pytest.approx(0.5)  # least-squares slope of (0,0),(1,.5),(2,1)


def test_improvement_degrades_is_negative() -> None:
    ep, res = _sessioned_episode([(0, 1.0), (1, 0.5), (2, 0.0)])
    trend = improvement_over_time(res, ep)
    assert trend is not None and trend < 0.0


def test_improvement_single_session_is_na() -> None:
    ep, res = _sessioned_episode([(0, 0.4)])
    assert improvement_over_time(res, ep) is None
    assert score_task(res, ep).improvement_over_time is None


# --------------------------------------------------------------------------- #
# TaskScore shape
# --------------------------------------------------------------------------- #
def test_taskscore_is_frozen_dataclass_with_required_fields(episode: Episode) -> None:
    ts = score_task(_result("fp", FAKE_PERFECT), episode)
    assert is_dataclass(ts) and isinstance(ts, TaskScore)
    names = {f.name for f in fields(ts)}
    assert {
        "task_score",
        "utilization_rate",
        "improvement_over_time",
        "judge_contribution",
        "per_probe",
    } <= names
    assert all(isinstance(pr, ProbeResult) for pr in ts.per_probe)
    # Non-constant attr name: stays clean under ruff B010 and mypy with no type-ignore.
    frozen_attr = "task_score"
    with pytest.raises(FrozenInstanceError):
        setattr(ts, frozen_attr, 0.0)
