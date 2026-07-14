"""Dim-2 task performance & goal-directed utilization metrics (spec/02-metrics.md §4).

Consumes a finished :class:`~lhmsb.types.EpisodeResult` (the per-probe
:class:`~lhmsb.types.ProbeResult`s produced upstream by the family ``Checker``s —
Research / Software / Default — and, for open-ended *synthesis* probes, by the sparse
``Judge``) together with its :class:`~lhmsb.types.Episode` (which carries the
``cross_session`` flags and the world-event schedule), and reports four numbers:

  - ``task_score`` — normalized fraction of probes answered correctly, in [0, 1]. Per
    §4.1 ``task_score = (Σ deterministic + Σ judge) / |probes|``. The judge term is folded
    in through the bounded composite of :func:`lhmsb.judge.combine_scores`, so the judge
    weight is HARD-CAPPED (default 0.20) and can never dominate the deterministic signal.
    When the judge scores ≤ 20% of the probes the composite equals the plain mean (matching
    the §4.3 worked example exactly); only beyond the cap does the bound bind.
  - ``utilization_rate`` — of the probes whose correct answer REQUIRES a fact from an
    earlier session (``cross_session=True``), the fraction answered correctly. Probes
    answerable from the CURRENT session's context are EXCLUDED — they don't test memory.
    ``None`` (N/A) when there are no cross-session probes; never silently 0.0.
  - ``improvement_over_time`` — least-squares slope of per-session task score (does
    performance rise as memory accumulates?). Supplementary; ``None`` with < 2 sessions.
  - ``judge_contribution`` — the applied (capped) judge weight, reported SEPARATELY so
    the judge's influence on ``task_score`` is always explicit and ≤ the cap.

This module never calls the judge itself (an ``EpisodeResult`` carries no answer text);
synthesis-probe scores are judge-provided upstream and merely bounded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from lhmsb.judge import combine_scores
from lhmsb.types import Episode, EpisodeResult, ProbeResult

# The judge's contribution to the Dim-2 composite is hard-capped (spec §4.2).
MAX_JUDGE_WEIGHT = 0.20
# Probe kinds scored by the sparse judge (open-ended; not programmatically decidable).
_JUDGE_KINDS = frozenset({"synthesis"})


@dataclass(frozen=True)
class TaskScore:
    """Per-episode Dim-2 scorecard (spec/02-metrics.md §4).

    Attributes:
        task_score: Normalized task performance in [0, 1] (judge weight bounded).
        utilization_rate: Cross-session recall fraction in [0, 1], or ``None`` (N/A)
            when the episode has no cross-session-dependent probes.
        improvement_over_time: Per-session task-score slope, or ``None`` (< 2 sessions).
        judge_contribution: Applied (capped) judge weight in the composite, in
            ``[0, MAX_JUDGE_WEIGHT]``; reported separately so it never silently merges
            into the deterministic score.
        per_probe: Per-probe results used, aligned to ``episode.probes`` order (a
            missing / timed-out probe is materialized as a 0.0 result, not dropped).
    """

    task_score: float
    utilization_rate: float | None
    improvement_over_time: float | None
    judge_contribution: float
    per_probe: list[ProbeResult]


def _clamp01(value: float) -> float:
    """Clamp to [0, 1] (guards float drift before the [0, 1]-strict composite)."""
    return max(0.0, min(1.0, value))


def _coerce_session(value: object) -> int:
    """Session index from an event-payload value (mirrors harness.sessions; non-int → 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _probe_sessions(episode: Episode) -> dict[str, int]:
    """Map each probe_id to its session: the latest event at/before its step (default 0).

    Mirrors :mod:`lhmsb.harness.sessions` (spec/03-protocol.md §1.3-1.4) without importing
    the harness, so the metrics layer stays free of the agent / langgraph dependency.
    """
    sorted_events = sorted(episode.events, key=lambda e: e.step)

    def session_for_step(step: int) -> int:
        session = 0
        for event in sorted_events:
            if event.step <= step:
                session = _coerce_session(event.payload.get("session", 0))
            else:
                break
        return session

    return {probe.probe_id: session_for_step(probe.step) for probe in episode.probes}


def _aligned_results(episode: Episode, episode_result: EpisodeResult) -> list[ProbeResult]:
    """Results aligned to ``episode.probes`` order; a missing probe → a 0.0 result.

    A timed-out / crashed / unanswered probe is scored 0 (not dropped) per §4.2.
    """
    by_id = {result.probe_id: result for result in episode_result.probe_results}
    aligned: list[ProbeResult] = []
    for probe in episode.probes:
        existing = by_id.get(probe.probe_id)
        if existing is not None:
            aligned.append(existing)
        else:
            aligned.append(
                ProbeResult(
                    probe_id=probe.probe_id,
                    score=0.0,
                    is_correct=False,
                    metadata={"status": "missing"},
                )
            )
    return aligned


def _is_judge_probe(kind: str) -> bool:
    """True for open-ended probe kinds whose scores come from the sparse judge."""
    return kind in _JUDGE_KINDS


def utilization_rate(episode_result: EpisodeResult, episode: Episode) -> float | None:
    """Fraction of cross-session-dependent probes answered correctly (spec §4.1).

    Only probes with ``cross_session=True`` are counted — those whose correct answer
    REQUIRES a fact from an earlier session, so getting them right is genuine memory
    utilization. Probes answerable from the current session's context
    (``cross_session=False``) are excluded; answering them correctly never inflates
    utilization. Returns ``None`` (N/A) when there are no cross-session probes.
    """
    cross_probes = [probe for probe in episode.probes if probe.cross_session]
    if not cross_probes:
        return None
    by_id = {result.probe_id: result for result in episode_result.probe_results}
    correct = 0
    for probe in cross_probes:
        result = by_id.get(probe.probe_id)
        if result is not None and result.is_correct:
            correct += 1
    return correct / len(cross_probes)


def improvement_over_time(episode_result: EpisodeResult, episode: Episode) -> float | None:
    """Least-squares slope of per-session mean task score (spec §4.1, supplementary).

    Groups probe scores by session, takes each session's mean, and fits a line over
    ``(session_index, mean_score)``. A positive slope means performance rises as memory
    accumulates. Returns ``None`` when fewer than two sessions carry probes.
    """
    sessions = _probe_sessions(episode)
    by_id = {result.probe_id: result for result in episode_result.probe_results}
    grouped: dict[int, list[float]] = {}
    for probe in episode.probes:
        result = by_id.get(probe.probe_id)
        score = result.score if result is not None else 0.0
        grouped.setdefault(sessions[probe.probe_id], []).append(score)
    if len(grouped) < 2:
        return None
    xs = sorted(grouped)
    ys = [sum(grouped[session]) / len(grouped[session]) for session in xs]
    return _slope([float(x) for x in xs], ys)


def _slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary least-squares slope; 0.0 when x has no variance (guarded by caller)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def score_task(episode_result: EpisodeResult, episode: Episode) -> TaskScore:
    """Aggregate per-probe results into the Dim-2 :class:`TaskScore` (spec §4).

    Deterministic-probe scores (from the family ``Checker``s) and judge-probe scores
    (open-ended synthesis, scored upstream by the sparse ``Judge``) are combined via
    :func:`lhmsb.judge.combine_scores` with the judge weight set to the judge probes'
    natural share and HARD-CAPPED at :data:`MAX_JUDGE_WEIGHT`. At/under the cap the
    composite equals the plain mean ``(Σ det + Σ judge) / |probes|``; beyond it the judge
    is down-weighted so it can never dominate. ``judge_contribution`` reports the applied
    weight separately.
    """
    aligned = _aligned_results(episode, episode_result)
    kinds = {probe.probe_id: probe.kind for probe in episode.probes}

    det_scores: list[float] = []
    judge_scores: list[float] = []
    for result in aligned:
        score = _clamp01(result.score)
        if _is_judge_probe(kinds.get(result.probe_id, "factual")):
            judge_scores.append(score)
        else:
            det_scores.append(score)

    util = utilization_rate(episode_result, episode)
    improvement = improvement_over_time(episode_result, episode)

    total = len(aligned)
    if total == 0:
        return TaskScore(
            task_score=0.0,
            utilization_rate=util,
            improvement_over_time=improvement,
            judge_contribution=0.0,
            per_probe=aligned,
        )

    det_mean = _clamp01(sum(det_scores) / len(det_scores)) if det_scores else 0.0
    judge_mean = _clamp01(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0
    requested_judge_weight = len(judge_scores) / total

    composite = combine_scores(
        det_mean,
        judge_mean,
        requested_judge_weight=requested_judge_weight,
        max_judge_weight=MAX_JUDGE_WEIGHT,
    )

    return TaskScore(
        task_score=composite.composite,
        utilization_rate=util,
        improvement_over_time=improvement,
        judge_contribution=composite.judge_contribution,
        per_probe=aligned,
    )
