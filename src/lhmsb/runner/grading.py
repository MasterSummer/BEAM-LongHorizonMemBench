"""Grade the harness's raw probe answers into scored :class:`ProbeResult`s.

The harness (task 9) records each probe's raw answer in
``ProbeResult.metadata["answer"]`` (plus ``query`` / ``retrieved_ids`` / ``kind`` /
``cross_session``) but leaves ``score=0.0`` / ``is_correct=False`` — scoring is the
runner's job. This module closes that gap:

  * :func:`build_checker` picks the family checker for an episode
    (:class:`~lhmsb.families.research.ResearchChecker` from the events;
    :class:`~lhmsb.families.software.SoftwareChecker` from the seed-regenerated spec;
    :class:`~lhmsb.sim.core.DefaultChecker` otherwise).
  * :func:`grade_probe` runs the checker on one probe's answer, defers open-ended
    *synthesis* probes to the sparse :class:`~lhmsb.judge.Judge`, and packages a new
    frozen :class:`ProbeResult` whose ``metadata`` carries everything the downstream
    metrics consume: ``drift_category`` (A/B), ``drift_flags``, ``facts_used``,
    ``judge_needed``, ``answer``, ``retrieved_ids``, ``cross_session`` …
  * :func:`grade_episode` regrades a whole :class:`~lhmsb.types.EpisodeResult`,
    preserving its ``cost`` / ``status`` (the orchestrator overlays the merged cost).

``EpisodeResult`` is frozen, so a NEW result with the graded probe list is built; the
raw harness result is never mutated.
"""

from __future__ import annotations

from dataclasses import replace

from lhmsb.judge import Judge, Rubric
from lhmsb.sim.core import Checker, CheckResult, DefaultChecker
from lhmsb.types import Episode, EpisodeResult, Probe, ProbeResult

#: Judge quality score at/above which a synthesis answer counts as correct.
DEFAULT_JUDGE_CORRECT_THRESHOLD = 0.5

# drift_flag prefixes (mirror lhmsb.metrics.drift) used to assign the drift category.
_STALE_PREFIXES: tuple[str, ...] = ("stale_fact", "stale-api", "stale-decision", "stale-value")
_CONSTRAINT_PREFIXES: tuple[str, ...] = ("constraint_violation", "constraint-violation")


def _meta(metadata: dict[str, object] | None) -> dict[str, object]:
    return dict(metadata) if metadata is not None else {}


def _answer_of(raw: ProbeResult) -> str:
    """The raw agent answer the harness recorded (``""`` when absent/None)."""
    value = _meta(raw.metadata).get("answer")
    return value if isinstance(value, str) else ""


def _flag_has_prefix(flag: str, prefixes: tuple[str, ...]) -> bool:
    return any(flag == prefix or flag.startswith(f"{prefix}:") for prefix in prefixes)


def drift_category_for(probe_kind: str, drift_flags: list[str]) -> str | None:
    """Assign the drift-alignment category (spec/02-metrics.md §2).

    Flags win first (a stale-* flag => A, a constraint-* flag => B) so a Software
    behavioral probe that used a deprecated API aligns to A. Otherwise the probe
    kind decides: ``factual`` => A (stale-fact-eligible), ``behavioral`` => B
    (constraint-eligible), ``synthesis`` => ``None`` (judge-scored, not a
    programmatic drift probe).
    """
    if any(_flag_has_prefix(flag, _STALE_PREFIXES) for flag in drift_flags):
        return "A"
    if any(_flag_has_prefix(flag, _CONSTRAINT_PREFIXES) for flag in drift_flags):
        return "B"
    if probe_kind == "factual":
        return "A"
    if probe_kind == "behavioral":
        return "B"
    return None


def _is_judge_probe(probe: Probe, check: CheckResult) -> bool:
    """True when scoring needs the sparse judge (open-ended synthesis)."""
    return probe.kind == "synthesis" or check.metadata.get("judge_needed") is True


def grade_probe(
    checker: Checker,
    probe: Probe,
    raw: ProbeResult,
    *,
    judge: Judge | None = None,
    rubric: Rubric | None = None,
    judge_correct_threshold: float = DEFAULT_JUDGE_CORRECT_THRESHOLD,
) -> ProbeResult:
    """Grade one harness probe result into a scored :class:`ProbeResult`.

    Programmatic probes are scored by the family ``checker``; synthesis probes are
    scored by the sparse ``judge`` (when provided with a ``rubric``), and the
    judge's contribution is bounded downstream by :func:`lhmsb.metrics.score_task`.
    The returned metadata carries the full drift / retrieval contract.
    """
    answer = _answer_of(raw)
    check = checker.check(probe, answer)
    flags = list(check.drift_flags)
    judge_needed = _is_judge_probe(probe, check)

    metadata: dict[str, object] = _meta(raw.metadata)
    for key, value in check.metadata.items():
        metadata[key] = value
    metadata["answer"] = answer
    metadata["kind"] = probe.kind
    metadata["cross_session"] = probe.cross_session
    metadata["facts_used"] = list(check.facts_used)
    metadata["drift_flags"] = flags
    category = drift_category_for(probe.kind, flags)
    if category is not None:
        metadata["drift_category"] = category

    if judge_needed and judge is not None and rubric is not None:
        verdict = judge.score(probe, answer, rubric)
        metadata["judge_needed"] = True
        metadata["judge_score"] = verdict.score
        metadata["judge_rationale"] = verdict.rationale
        return ProbeResult(
            probe_id=probe.probe_id,
            score=verdict.score,
            is_correct=verdict.score >= judge_correct_threshold,
            metadata=metadata,
        )
    if judge_needed:
        # No judge available: keep the probe judge-flagged but unscored (0.0) so the
        # downstream metric treats it as a deferred synthesis probe, never a hard 1.0.
        metadata["judge_needed"] = True
        return ProbeResult(
            probe_id=probe.probe_id, score=0.0, is_correct=False, metadata=metadata
        )
    return ProbeResult(
        probe_id=probe.probe_id,
        score=check.score,
        is_correct=check.is_correct,
        metadata=metadata,
    )


def grade_episode(
    raw_result: EpisodeResult,
    episode: Episode,
    *,
    checker: Checker,
    judge: Judge | None = None,
    rubric: Rubric | None = None,
    judge_correct_threshold: float = DEFAULT_JUDGE_CORRECT_THRESHOLD,
) -> EpisodeResult:
    """Regrade every raw probe result, returning a new graded :class:`EpisodeResult`.

    The episode's ``cost`` and ``status`` are preserved verbatim (the orchestrator
    overlays the merged adapter+harness cost afterwards). Probes the harness never
    answered (e.g. a crash truncated the run) are simply absent here — the task
    metric materializes them as 0.0, so they are penalized, not dropped.
    """
    probes_by_id: dict[str, Probe] = {probe.probe_id: probe for probe in episode.probes}
    graded: list[ProbeResult] = []
    for raw in raw_result.probe_results:
        probe = probes_by_id.get(raw.probe_id)
        if probe is None:
            graded.append(raw)  # answered a probe not in this episode → keep as-is
            continue
        graded.append(
            grade_probe(
                checker,
                probe,
                raw,
                judge=judge,
                rubric=rubric,
                judge_correct_threshold=judge_correct_threshold,
            )
        )
    return replace(raw_result, probe_results=graded)


def build_checker(episode: Episode) -> Checker:
    """Pick the programmatic checker for ``episode`` based on its family.

    ``research`` builds a :class:`ResearchChecker` straight from the episode's world
    events. ``software`` regenerates the :class:`SoftwareSpec` from the episode seed
    (deterministic — it matches the frozen events) and wraps it in a
    :class:`SoftwareChecker`. Any other family falls back to :class:`DefaultChecker`.
    """
    if episode.family == "research":
        from lhmsb.families.research import ResearchChecker

        return ResearchChecker(events=episode.events)
    if episode.family == "research_wide":
        from lhmsb.families.research import WideResearchChecker

        return WideResearchChecker()
    if episode.family == "software":
        from lhmsb.families.software import SoftwareChecker, SoftwareFamily

        spec = SoftwareFamily().build_spec(episode.seed)
        return SoftwareChecker(spec)
    return DefaultChecker()
