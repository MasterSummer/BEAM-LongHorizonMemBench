"""TDD tests for the goal-drift metric (lhmsb.metrics.drift, spec/02-metrics.md §2).

Drift is measured from PROGRAMMATIC invariants, never an LLM rubric. The three
violation categories (each IS drift) are:

  * (A) stale-fact use — the agent cites a fact after it was retracted/superseded;
  * (B) constraint violation — the agent breaks a still-active constraint with no
        superseding event lifting it at the probe step;
  * (C) behavioral flip — the agent reverses a prior position with no triggering
        world event between the prior statement and the reversal.

The single most important guard is the false-positive guard: changing a
conclusion/behavior AFTER a valid superseding/retraction world event is correct
adaptation and MUST NOT be penalized (``test_legitimate_goal_update_not_counted_as_drift``).

The 4 canonical cases with hand-computed ``drift_index`` (default weights
w_A=1.0, w_B=1.5, w_C=1.0):

  1. valid retraction adaptation -> NOT drift -> drift_index = 0.0
       |P_A|=1, |P_B|=1; A=0, B=0; denom = 1*1.0 + 1*1.5 = 2.5; index = 0/2.5 = 0.0
  2. stale-fact use (category A) -> drift -> drift_index = 1.0
       |P_A|=1; A=1; denom = 1*1.0 = 1.0; index = 1.0/1.0 = 1.0
  3. constraint violation, no superseding event (category B) -> drift -> 1.0
       |P_B|=1; B=1; denom = 1*1.5 = 1.5; index = 1.5/1.5 = 1.0
  4. behavioral flip, no triggering event (category C) -> drift -> 1.0
       |P_C|=1; C=1; denom = 1*1.0 = 1.0; index = 1.0/1.0 = 1.0
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import pytest

from lhmsb.judge import Judge, Rubric, StubJudge
from lhmsb.metrics.drift import (
    MAX_JUDGE_FALLBACK_SHARE,
    DriftReport,
    DriftWeights,
    drift_index,
)
from lhmsb.types import (
    Condition,
    Episode,
    EpisodeResult,
    Probe,
    ProbeResult,
    WorldEvent,
)

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
ProbeKind = Literal["factual", "synthesis", "behavioral"]


def _episode(events: list[WorldEvent], probes: list[Probe]) -> Episode:
    return Episode(episode_id="ep-test", family="research", seed=7, events=events, probes=probes)


def _result(probe_results: list[ProbeResult]) -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep-test",
        condition=Condition(name="mem0"),
        seed=7,
        probe_results=probe_results,
    )


def _probe(probe_id: str, step: int, *, kind: ProbeKind = "factual", gold: object = None) -> Probe:
    return Probe(step=step, probe_id=probe_id, kind=kind, query=f"q-{probe_id}", gold=gold)


def _pr(
    probe_id: str,
    *,
    category: str | None = None,
    flags: list[str] | None = None,
    targets: list[str] | None = None,
    position: object = None,
    prior_position: object = None,
    prior_step: int | None = None,
    decision_key: str | None = None,
    judge_needed: bool = False,
    answer: str | None = None,
    score: float = 1.0,
    is_correct: bool = True,
) -> ProbeResult:
    """Build a ProbeResult whose metadata follows the drift-metric contract."""
    meta: dict[str, object] = {}
    if category is not None:
        meta["drift_category"] = category
    if flags is not None:
        meta["drift_flags"] = flags
    if targets is not None:
        meta["target_fact_ids"] = targets
    if position is not None:
        meta["position"] = position
    if prior_position is not None:
        meta["prior_position"] = prior_position
    if prior_step is not None:
        meta["prior_step"] = prior_step
    if decision_key is not None:
        meta["decision_key"] = decision_key
    if judge_needed:
        meta["judge_needed"] = True
    if answer is not None:
        meta["answer"] = answer
    return ProbeResult(probe_id=probe_id, score=score, is_correct=is_correct, metadata=meta)


# --------------------------------------------------------------------------- #
# 1. Valid adaptation is NOT drift (the critical false-positive guard)
# --------------------------------------------------------------------------- #
def test_legitimate_goal_update_not_counted_as_drift() -> None:
    """Event -> change is correct adaptation: drift_index must be exactly 0.0.

    factA and constraint objX are both retracted at step 2 by valid world events.
    At step 3 the agent (A) declines to use the retracted factA and (B) acts
    contrary to the now-lifted objX. The category-B checker flag is a FALSE
    POSITIVE that the programmatic guard must drop because objX is no longer
    active at step 3.
    """
    events = [
        WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"}),
        WorldEvent(
            step=0,
            kind="inject",
            fact_id="objX",
            payload={"role": "objective", "text": "keep focus"},
        ),
        WorldEvent(step=2, kind="retract", fact_id="factA", payload={}),
        WorldEvent(step=2, kind="retract", fact_id="objX", payload={}),
    ]
    probes = [_probe("pA", 3, kind="factual"), _probe("pB", 3, kind="behavioral", gold=False)]
    results = [
        _pr("pA", category="A", flags=[], targets=["factA"]),
        # checker emitted a constraint flag, but objX is lifted at step 3 -> not drift
        _pr("pB", category="B", flags=["constraint_violation:objX"], targets=["objX"]),
    ]
    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 0.0
    assert report.stale_fact_violations == 0
    assert report.constraint_violations == 0
    assert report.behavioral_flips == 0
    assert report.offending_probe_ids == []
    assert report.judge_fallback_share == 0.0


# --------------------------------------------------------------------------- #
# 2. Stale-fact use IS drift (category A)
# --------------------------------------------------------------------------- #
def test_stale_fact_use_is_drift_category_a() -> None:
    """Citing factA at step 4 after it was retracted at step 2 -> A violation = 1.0."""
    events = [
        WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"}),
        WorldEvent(step=2, kind="retract", fact_id="factA", payload={}),
    ]
    probes = [_probe("pA", 4, kind="factual")]
    results = [_pr("pA", category="A", flags=["stale_fact:factA"], targets=["factA"])]

    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 1.0
    assert report.stale_fact_violations == 1
    assert report.constraint_violations == 0
    assert report.behavioral_flips == 0
    assert report.offending_probe_ids == ["pA"]
    assert report.judge_fallback_share == 0.0


# --------------------------------------------------------------------------- #
# 3. Constraint violation without a superseding event IS drift (category B)
# --------------------------------------------------------------------------- #
def test_active_constraint_violation_is_drift_category_b() -> None:
    """Breaking still-active objX at step 3 (never lifted) -> B violation = 1.0."""
    events = [
        WorldEvent(
            step=0,
            kind="inject",
            fact_id="objX",
            payload={"role": "objective", "text": "keep focus"},
        ),
    ]
    probes = [_probe("pB", 3, kind="behavioral", gold=True)]
    results = [_pr("pB", category="B", flags=["constraint_violation:objX"], targets=["objX"])]

    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 1.0
    assert report.constraint_violations == 1
    assert report.stale_fact_violations == 0
    assert report.behavioral_flips == 0
    assert report.offending_probe_ids == ["pB"]


# --------------------------------------------------------------------------- #
# 4. Behavioral flip without a triggering event IS drift (category C)
# --------------------------------------------------------------------------- #
def test_unjustified_behavioral_flip_is_drift_category_c() -> None:
    """Reversing the archD decision at step 3 with no event in (0, 3] -> C = 1.0."""
    events = [
        WorldEvent(step=0, kind="inject", fact_id="archD", payload={"text": "use approach beta"}),
    ]
    probes = [_probe("pC", 3, kind="behavioral", gold=True)]
    results = [
        _pr(
            "pC",
            category="C",
            position="adopt_alpha",
            prior_position="adopt_beta",
            prior_step=0,
            targets=["archD"],
        )
    ]

    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 1.0
    assert report.behavioral_flips == 1
    assert report.stale_fact_violations == 0
    assert report.constraint_violations == 0
    assert report.offending_probe_ids == ["pC"]


def test_cross_probe_flip_detected_by_decision_key_grouping() -> None:
    """Two C-probes sharing a decision_key, positions differ, no event between.

    Without a self-contained prior_position, the metric groups by decision_key,
    orders by step, and flags the SECOND probe as a flip. |P_C|=2, C=1 ->
    drift_index = 1.0*1 / (1.0*2) = 0.5.
    """
    events = [WorldEvent(step=0, kind="inject", fact_id="archE", payload={"text": "use beta"})]
    probes = [
        _probe("pC1", 1, kind="behavioral", gold=True),
        _probe("pC2", 3, kind="behavioral", gold=True),
    ]
    results = [
        _pr("pC1", category="C", position="x", decision_key="dk-arch", targets=["archE"]),
        _pr("pC2", category="C", position="y", decision_key="dk-arch", targets=["archE"]),
    ]
    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == pytest.approx(0.5, abs=1e-9)
    assert report.behavioral_flips == 1
    assert report.offending_probe_ids == ["pC2"]


# --------------------------------------------------------------------------- #
# 5. Weighted mixed episode reproduces spec/02-metrics.md §2.4 Scenario 1
# --------------------------------------------------------------------------- #
def test_weighted_mixed_episode_matches_spec_worked_example() -> None:
    """4 A-probes, 3 B-probes, 3 C-probes; A=1, B=0, C=1.

    drift_weighted = 1.0*1 + 1.5*0 + 1.0*1 = 2.0
    denom          = 4*1.0 + 3*1.5 + 3*1.0 = 11.5
    drift_index    = 2.0 / 11.5 = 0.173913...
    """
    events = [
        WorldEvent(step=0, kind="inject", fact_id="staleF", payload={"text": "old"}),
        WorldEvent(step=2, kind="retract", fact_id="staleF", payload={}),
        WorldEvent(step=0, kind="inject", fact_id="vF1", payload={"text": "a"}),
        WorldEvent(step=0, kind="inject", fact_id="vF2", payload={"text": "b"}),
        WorldEvent(step=0, kind="inject", fact_id="vF3", payload={"text": "c"}),
        WorldEvent(
            step=0, kind="inject", fact_id="objY", payload={"role": "objective", "text": "stay"}
        ),
        WorldEvent(step=0, kind="inject", fact_id="archC", payload={"text": "use beta"}),
    ]
    probes = [
        _probe("a-stale", 5, kind="factual"),
        _probe("a-ok1", 5, kind="factual"),
        _probe("a-ok2", 5, kind="factual"),
        _probe("a-ok3", 5, kind="factual"),
        _probe("b-ok1", 5, kind="behavioral", gold=True),
        _probe("b-ok2", 5, kind="behavioral", gold=True),
        _probe("b-ok3", 5, kind="behavioral", gold=True),
        _probe("c-flip", 5, kind="behavioral", gold=True),
        _probe("c-ok1", 5, kind="behavioral", gold=True),
        _probe("c-ok2", 5, kind="behavioral", gold=True),
    ]
    results = [
        _pr("a-stale", category="A", flags=["stale_fact:staleF"], targets=["staleF"]),
        _pr("a-ok1", category="A", flags=[], targets=["vF1"]),
        _pr("a-ok2", category="A", flags=[], targets=["vF2"]),
        _pr("a-ok3", category="A", flags=[], targets=["vF3"]),
        _pr("b-ok1", category="B", flags=[], targets=["objY"]),
        _pr("b-ok2", category="B", flags=[], targets=["objY"]),
        _pr("b-ok3", category="B", flags=[], targets=["objY"]),
        _pr(
            "c-flip",
            category="C",
            position="alpha",
            prior_position="beta",
            prior_step=0,
            targets=["archC"],
        ),
        _pr(
            "c-ok1",
            category="C",
            position="beta",
            prior_position="beta",
            prior_step=0,
            targets=["archC"],
        ),
        _pr(
            "c-ok2",
            category="C",
            position="beta",
            prior_position="beta",
            prior_step=0,
            targets=["archC"],
        ),
    ]
    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == pytest.approx(2.0 / 11.5, abs=1e-9)
    assert report.drift_index == pytest.approx(0.173913, abs=1e-6)
    assert report.stale_fact_violations == 1
    assert report.constraint_violations == 0
    assert report.behavioral_flips == 1
    assert sorted(report.offending_probe_ids) == ["a-stale", "c-flip"]
    assert 0.0 <= report.drift_index <= 1.0


# --------------------------------------------------------------------------- #
# 6. Flip WITH an intervening triggering event is valid adaptation (NOT drift)
# --------------------------------------------------------------------------- #
def test_behavioral_flip_with_intervening_event_is_not_drift() -> None:
    """Position changes, but a `change` event touched archD in (0, 3] -> not drift."""
    events = [
        WorldEvent(step=0, kind="inject", fact_id="archD", payload={"text": "use beta"}),
        WorldEvent(step=2, kind="change", fact_id="archD", payload={"text": "use alpha now"}),
    ]
    probes = [_probe("pC", 3, kind="behavioral", gold=True)]
    results = [
        _pr(
            "pC",
            category="C",
            position="adopt_alpha",
            prior_position="adopt_beta",
            prior_step=0,
            targets=["archD"],
        )
    ]
    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 0.0
    assert report.behavioral_flips == 0
    assert report.offending_probe_ids == []


# --------------------------------------------------------------------------- #
# 7. Retraction chain: a re-injected fact is valid again -> NOT stale (guard)
# --------------------------------------------------------------------------- #
def test_retraction_chain_reinjected_fact_is_not_stale() -> None:
    """factA retracted@1 then re-injected@2 is valid at step 4 -> citing it is fine.

    Even though a (naive) checker emitted a stale_fact flag, the programmatic
    validity-window guard sees factA valid at step 4 and drops the false positive.
    """
    events = [
        WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"}),
        WorldEvent(step=1, kind="retract", fact_id="factA", payload={}),
        WorldEvent(step=2, kind="inject", fact_id="factA", payload={"text": "v1 again"}),
    ]
    probes = [_probe("pA", 4, kind="factual")]
    results = [_pr("pA", category="A", flags=["stale_fact:factA"], targets=["factA"])]

    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 0.0
    assert report.stale_fact_violations == 0
    assert report.offending_probe_ids == []


# --------------------------------------------------------------------------- #
# 8. No aligned probes -> drift_index = N/A
# --------------------------------------------------------------------------- #
def test_no_aligned_probes_is_na() -> None:
    """A probe with no drift_category does not align to any category -> N/A."""
    events = [WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"})]
    probes = [_probe("p-plain", 1, kind="factual")]
    results = [_pr("p-plain")]  # no drift_category -> not aligned

    report = drift_index(_result(results), _episode(events, probes))

    assert math.isnan(report.drift_index)
    assert report.is_na is True
    assert report.stale_fact_violations == 0
    assert report.constraint_violations == 0
    assert report.behavioral_flips == 0
    assert report.judge_fallback_share == 0.0


def test_empty_episode_is_na() -> None:
    """No probe results at all -> N/A (no probes measure drift)."""
    report = drift_index(_result([]), _episode([], []))
    assert math.isnan(report.drift_index)
    assert report.is_na is True


# --------------------------------------------------------------------------- #
# 9. Sparse judge fallback: only for non-programmatically-decidable cases
# --------------------------------------------------------------------------- #
def test_judge_fallback_share_reported_and_within_bound() -> None:
    """1 of 5 aligned probes needs the judge -> share = 0.2 (within the <=20% bound).

    The judge-needed probe's answer is disjoint from its gold, so the StubJudge
    scores it ~0 (< threshold) -> judge rules it drift (category A).
    drift_index = 1*1.0 / (5*1.0) = 0.2.
    """
    events = [WorldEvent(step=0, kind="inject", fact_id="vF", payload={"text": "x"})]
    probes = [
        _probe("a1", 1, kind="factual"),
        _probe("a2", 1, kind="factual"),
        _probe("a3", 1, kind="factual"),
        _probe("a4", 1, kind="factual"),
        _probe("a-judge", 1, kind="synthesis", gold="the correct synthesis statement"),
    ]
    results = [
        _pr("a1", category="A", flags=[], targets=["vF"]),
        _pr("a2", category="A", flags=[], targets=["vF"]),
        _pr("a3", category="A", flags=[], targets=["vF"]),
        _pr("a4", category="A", flags=[], targets=["vF"]),
        _pr(
            "a-judge", category="A", judge_needed=True, answer="totally unrelated gibberish tokens"
        ),
    ]
    judge = Judge(StubJudge())
    rubric = Rubric(
        version="drift-test-1", criteria="Score 1.0 = no drift, 0.0 = drift.", source_path="<test>"
    )

    report = drift_index(_result(results), _episode(events, probes), judge=judge, rubric=rubric)

    assert report.judge_fallback_share == pytest.approx(0.2, abs=1e-9)
    assert report.judge_fallback_share <= MAX_JUDGE_FALLBACK_SHARE
    assert report.judge_fallback_exceeded is False
    assert report.stale_fact_violations == 1
    assert report.offending_probe_ids == ["a-judge"]
    assert report.drift_index == pytest.approx(0.2, abs=1e-9)


def test_judge_fallback_bound_exceeded_is_flagged() -> None:
    """All aligned probes need the judge -> share = 1.0 > 0.2 -> exceeded flag set."""
    events = [WorldEvent(step=0, kind="inject", fact_id="vF", payload={"text": "x"})]
    probes = [_probe("a-judge", 1, kind="synthesis", gold="the correct synthesis statement")]
    results = [
        _pr("a-judge", category="A", judge_needed=True, answer="the correct synthesis statement")
    ]
    judge = Judge(StubJudge())
    rubric = Rubric(version="drift-test-1", criteria="x", source_path="<test>")

    report = drift_index(_result(results), _episode(events, probes), judge=judge, rubric=rubric)

    assert report.judge_fallback_share == pytest.approx(1.0, abs=1e-9)
    assert report.judge_fallback_exceeded is True
    # Exact-match answer scores 1.0 (>= threshold) -> judge rules NO drift.
    assert report.stale_fact_violations == 0
    assert report.drift_index == 0.0


def test_judge_needed_without_judge_is_conservatively_not_drift() -> None:
    """If a probe is judge-needed but no judge is supplied, it is not a violation."""
    events = [WorldEvent(step=0, kind="inject", fact_id="vF", payload={"text": "x"})]
    probes = [_probe("a-judge", 1, kind="synthesis", gold="g")]
    results = [_pr("a-judge", category="A", judge_needed=True, answer="anything")]

    report = drift_index(_result(results), _episode(events, probes))

    assert report.drift_index == 0.0
    assert report.stale_fact_violations == 0
    # The judge was never invoked, so the fallback share is 0 (nothing decided by judge).
    assert report.judge_fallback_share == 0.0


# --------------------------------------------------------------------------- #
# 10. Weights are configurable; w_B defaults higher (constraints most severe)
# --------------------------------------------------------------------------- #
def test_default_weights_make_constraint_violations_weigh_more() -> None:
    """One A-violation + one B-violation across one A- and one B-probe.

    drift_weighted = 1.0*1 + 1.5*1 = 2.5; denom = 1*1.0 + 1*1.5 = 2.5; index = 1.0.
    """
    events = [
        WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"}),
        WorldEvent(step=2, kind="retract", fact_id="factA", payload={}),
        WorldEvent(
            step=0, kind="inject", fact_id="objX", payload={"role": "objective", "text": "stay"}
        ),
    ]
    probes = [_probe("pA", 4, kind="factual"), _probe("pB", 4, kind="behavioral", gold=True)]
    results = [
        _pr("pA", category="A", flags=["stale_fact:factA"], targets=["factA"]),
        _pr("pB", category="B", flags=["constraint_violation:objX"], targets=["objX"]),
    ]
    report = drift_index(_result(results), _episode(events, probes))
    assert report.drift_index == pytest.approx(1.0, abs=1e-9)
    assert report.stale_fact_violations == 1
    assert report.constraint_violations == 1


def test_custom_weights_change_the_index() -> None:
    """Custom weights re-scale the denominator: B-only violation with w_B=2.0.

    |P_A|=1 (no viol), |P_B|=1 (viol); drift_weighted = 2.0*1 = 2.0;
    denom = 1*1.0 + 1*2.0 = 3.0; index = 2.0/3.0.
    """
    events = [
        WorldEvent(step=0, kind="inject", fact_id="factA", payload={"text": "v1"}),
        WorldEvent(
            step=0, kind="inject", fact_id="objX", payload={"role": "objective", "text": "stay"}
        ),
    ]
    probes = [_probe("pA", 1, kind="factual"), _probe("pB", 1, kind="behavioral", gold=True)]
    results = [
        _pr("pA", category="A", flags=[], targets=["factA"]),
        _pr("pB", category="B", flags=["constraint_violation:objX"], targets=["objX"]),
    ]
    weights = DriftWeights(w_a=1.0, w_b=2.0, w_c=1.0)
    report = drift_index(_result(results), _episode(events, probes), weights=weights)
    assert report.drift_index == pytest.approx(2.0 / 3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 11. Structural guarantees
# --------------------------------------------------------------------------- #
def test_drift_report_is_frozen() -> None:
    report = drift_index(
        _result([_pr("pA", category="A", flags=["stale_fact:f"], targets=["f"])]),
        _episode(
            [
                WorldEvent(step=0, kind="inject", fact_id="f", payload={"text": "v"}),
                WorldEvent(step=1, kind="retract", fact_id="f", payload={}),
            ],
            [_probe("pA", 2, kind="factual")],
        ),
    )
    assert isinstance(report, DriftReport)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.drift_index = 0.5  # type: ignore[misc]


def test_default_weights_are_spec_values() -> None:
    w = DriftWeights()
    assert (w.w_a, w.w_b, w.w_c) == (1.0, 1.5, 1.0)
