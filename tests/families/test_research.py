"""TDD tests for the Research family (lhmsb.families.research).

Covers (spec/04-datasets.md §2.2):
  - ``ResearchChecker`` deterministic grading + drift detection on a tiny
    hand-derived fixture (correct → 1.0; stale-fact citation → <1.0 + drift
    flag; off-topic → low utilization; synthesis → judge-flagged; objective
    violation → constraint-drift flag).
  - ``lint_no_real_entities`` passes on synthetic text, raises on a planted
    real paper title / DOI.
  - ``ResearchFamily.generate`` respects the family scope caps (15-40 facts,
    20-40% retracted, 3-6 sessions, all four probe kinds, cross-session probes,
    a frozen DAG whose retracted parent cascades) and is deterministic.

The canonical 6-step checker fixture (per the Task 10 spec):
  inject  ev-001 @1  {"text": "Project Chimera elevates the Q-index."}
  inject  ev-002 @2  {"text": "Study Gamma-7 reports Delta stable."}
  retract ev-001 @3
  change  ev-002 @4  {"text": "Study Gamma-7 reports Delta rising."}
  probe@2 recall  (value, ev-001)        -> gold "Project Chimera elevates the Q-index."
  probe@4 update  (value, ev-001)        -> gold None  (retracted @3)
  probe@5 synth   (valid_values, both)   -> gold ["Study Gamma-7 reports Delta rising."]
"""

from __future__ import annotations

import pytest

from lhmsb.families.research import (
    RealEntityLeakError,
    ResearchChecker,
    ResearchFamily,
    lint_no_real_entities,
)
from lhmsb.sim import Checker, EpisodeBuilder, FamilyContent, ProbeSpec, ScaleParams, WorldState
from lhmsb.types import Probe, WorldEvent

# Canonical synthetic values used by the hand fixture.
_EV1_VALUE = "Project Chimera elevates the Q-index."
_EV2_VALUE = "Study Gamma-7 reports Delta stable."
_EV2_VALUE_NEW = "Study Gamma-7 reports Delta rising."


def _fixture_content() -> FamilyContent:
    events = [
        WorldEvent(
            step=1,
            kind="inject",
            fact_id="ev-001",
            payload={
                "text": _EV1_VALUE,
                "entity": "Project Chimera",
                "topic": "Hypothesis Alpha",
                "session": 0,
            },
        ),
        WorldEvent(
            step=2,
            kind="inject",
            fact_id="ev-002",
            payload={
                "text": _EV2_VALUE,
                "entity": "Study Gamma-7",
                "topic": "Hypothesis Alpha",
                "session": 0,
            },
        ),
        WorldEvent(step=3, kind="retract", fact_id="ev-001", payload={"session": 1}),
        WorldEvent(
            step=4,
            kind="change",
            fact_id="ev-002",
            payload={
                "text": _EV2_VALUE_NEW,
                "entity": "Study Gamma-7",
                "topic": "Hypothesis Alpha",
                "session": 1,
            },
        ),
    ]
    probes = [
        ProbeSpec(
            step=2,
            probe_id="recall-ev-001",
            kind="factual",
            query="What does Project Chimera report?",
            derivation="value",
            target_fact_ids=["ev-001"],
            cross_session=False,
        ),
        ProbeSpec(
            step=4,
            probe_id="update-ev-001",
            kind="factual",
            query="Per the latest evidence, what is Project Chimera's finding?",
            derivation="value",
            target_fact_ids=["ev-001"],
            cross_session=True,
        ),
        ProbeSpec(
            step=5,
            probe_id="synth-alpha",
            kind="synthesis",
            query="Summarize the current evidence on Hypothesis Alpha.",
            derivation="valid_values",
            target_fact_ids=["ev-001", "ev-002"],
            cross_session=True,
        ),
    ]
    return FamilyContent(family="research", events=events, probe_specs=probes)


def _fixture_probes() -> dict[str, Probe]:
    episode = EpisodeBuilder().build(_fixture_content(), seed=7)
    return {p.probe_id: p for p in episode.probes}


class TestFixtureGold:
    """Hand-derived gold for the 3 fixture probes (revealed-minus-retracted)."""

    def test_recall_gold_is_current_value(self) -> None:
        assert _fixture_probes()["recall-ev-001"].gold == _EV1_VALUE

    def test_update_gold_is_none_after_retraction(self) -> None:
        assert _fixture_probes()["update-ev-001"].gold is None

    def test_synth_gold_lists_only_valid(self) -> None:
        assert _fixture_probes()["synth-alpha"].gold == [_EV2_VALUE_NEW]


class TestResearchCheckerScoring:
    """Deterministic grade + drift + utilization on the fixture."""

    def test_is_a_checker(self) -> None:
        assert isinstance(ResearchChecker.from_content(_fixture_content()), Checker)

    def test_correct_recall_scores_one(self) -> None:
        checker = ResearchChecker.from_content(_fixture_content())
        result = checker.check(_fixture_probes()["recall-ev-001"], _EV1_VALUE)
        assert result.score == 1.0
        assert result.is_correct
        assert "ev-001" in result.facts_used
        assert result.drift_flags == []

    def test_stale_fact_answer_penalized_and_flagged(self) -> None:
        # probe@4 asks for the CURRENT finding; ev-001 was retracted @3.
        # Citing its old value (no retraction marker) is stale-fact drift.
        checker = ResearchChecker.from_content(_fixture_content())
        result = checker.check(_fixture_probes()["update-ev-001"], _EV1_VALUE)
        assert result.score < 1.0
        assert not result.is_correct
        assert any(flag.startswith("stale_fact:ev-001") for flag in result.drift_flags)
        assert "ev-001" in result.facts_used

    def test_correct_update_answer_notes_retraction(self) -> None:
        checker = ResearchChecker.from_content(_fixture_content())
        result = checker.check(
            _fixture_probes()["update-ev-001"],
            "That earlier finding has been retracted and is no longer valid.",
        )
        assert result.score == 1.0
        assert result.is_correct
        assert result.drift_flags == []

    def test_off_topic_answer_low_utilization(self) -> None:
        checker = ResearchChecker.from_content(_fixture_content())
        result = checker.check(
            _fixture_probes()["recall-ev-001"], "The weather is pleasant today."
        )
        assert result.score == 0.0
        assert result.facts_used == []
        assert not result.is_correct

    def test_synthesis_probe_flagged_for_judge(self) -> None:
        checker = ResearchChecker.from_content(_fixture_content())
        result = checker.check(
            _fixture_probes()["synth-alpha"], "Hypothesis Alpha currently rests on Study Gamma-7."
        )
        assert result.metadata.get("judge_needed") is True

    def test_superseded_value_is_stale(self) -> None:
        # ev-002 was changed @4; citing its OLD value at step 5 is superseded use.
        checker = ResearchChecker.from_content(_fixture_content())
        probe = Probe(
            step=5,
            probe_id="recall-ev-002-late",
            kind="factual",
            query="What does Study Gamma-7 report now?",
            gold=_EV2_VALUE_NEW,
        )
        result = checker.check(probe, _EV2_VALUE)  # the stale, pre-change value
        assert any(flag.startswith("stale_fact:ev-002") for flag in result.drift_flags)
        assert not result.is_correct

    def test_current_value_after_change_is_correct(self) -> None:
        checker = ResearchChecker.from_content(_fixture_content())
        probe = Probe(
            step=5,
            probe_id="recall-ev-002-late",
            kind="factual",
            query="What does Study Gamma-7 report now?",
            gold=_EV2_VALUE_NEW,
        )
        result = checker.check(probe, _EV2_VALUE_NEW)
        assert result.score == 1.0
        assert result.is_correct
        assert "ev-002" in result.facts_used
        assert result.drift_flags == []


class TestObjectiveAdherenceDrift:
    """Behavioral probe: violating a still-active objective is category-B drift."""

    def _content(self) -> FamilyContent:
        events = [
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="ev-000",
                payload={
                    "text": "Keep the investigation focused on Hypothesis Alpha.",
                    "role": "objective",
                    "session": 0,
                },
            ),
            WorldEvent(
                step=2,
                kind="inject",
                fact_id="ev-001",
                payload={"text": "Project Chimera elevates the Q-index.", "session": 0},
            ),
        ]
        probes = [
            ProbeSpec(
                step=3,
                probe_id="objective-000",
                kind="behavioral",
                query="Is this line of investigation still consistent with the objective?",
                derivation="valid",
                target_fact_ids=["ev-000"],
                cross_session=True,
            )
        ]
        return FamilyContent(family="research", events=events, probe_specs=probes)

    def _probe(self) -> Probe:
        episode = EpisodeBuilder().build(self._content(), seed=1)
        return next(p for p in episode.probes if p.probe_id == "objective-000")

    def test_objective_gold_is_true_while_active(self) -> None:
        assert self._probe().gold is True

    def test_adherence_is_correct_no_drift(self) -> None:
        checker = ResearchChecker.from_content(self._content())
        result = checker.check(
            self._probe(), "Yes, the work remains consistent with the stated objective."
        )
        assert result.score == 1.0
        assert result.is_correct
        assert result.drift_flags == []

    def test_abandoning_active_objective_is_drift(self) -> None:
        checker = ResearchChecker.from_content(self._content())
        result = checker.check(
            self._probe(), "No, we have abandoned that objective and pivoted elsewhere."
        )
        assert result.score == 0.0
        assert not result.is_correct
        assert any(flag.startswith("constraint_violation:ev-000") for flag in result.drift_flags)


class TestLeakageGuard:
    """lint_no_real_entities: synthetic passes, real identifiers raise."""

    def test_synthetic_text_passes(self) -> None:
        lint_no_real_entities(
            "Project Chimera elevates the Q-index; Study Gamma-7 reports Delta rising (ev-001)."
        )

    def test_real_paper_title_raises(self) -> None:
        with pytest.raises(RealEntityLeakError):
            lint_no_real_entities("As shown in Attention Is All You Need, transformers scale.")

    def test_doi_pattern_raises(self) -> None:
        with pytest.raises(RealEntityLeakError):
            lint_no_real_entities("See https://doi.org/10.1234/abcd.5678 for details.")

    def test_real_author_name_raises(self) -> None:
        with pytest.raises(RealEntityLeakError):
            lint_no_real_entities("Following Vaswani et al., we adopt self-attention.")


class TestResearchGenerator:
    """Generator respects spec/04 §2.2 caps and is deterministic."""

    _SCALE = ScaleParams(min_facts=15, max_facts=40)

    def _content(self, seed: int = 7) -> FamilyContent:
        return ResearchFamily().generate(seed=seed, scale=self._SCALE)

    def test_builds_a_valid_episode(self) -> None:
        content = self._content()
        episode = EpisodeBuilder().build(content, seed=7, scale=self._SCALE)
        assert episode.family == "research"
        assert episode.events

    def test_fact_count_within_caps(self) -> None:
        content = self._content()
        injected = {e.fact_id for e in content.events if e.kind == "inject"}
        assert 15 <= len(injected) <= 40

    def test_retraction_rate_within_caps(self) -> None:
        content = self._content()
        injected = {e.fact_id for e in content.events if e.kind == "inject"}
        max_step = max(e.step for e in content.events)
        valid = set(WorldState(content.events).valid_facts_at(max_step))
        retracted = injected - valid
        fraction = len(retracted) / len(injected)
        assert 0.20 <= fraction <= 0.40

    def test_session_count_within_caps(self) -> None:
        content = self._content()
        sessions = {
            int(e.payload["session"])
            for e in content.events
            if isinstance(e.payload.get("session"), int)
        }
        assert 3 <= len(sessions) <= 6

    def test_all_probe_kinds_present(self) -> None:
        content = self._content()
        kinds = {spec.kind for spec in content.probe_specs}
        assert kinds == {"factual", "synthesis", "behavioral"}
        prefixes = {spec.probe_id.split("-")[0] for spec in content.probe_specs}
        assert {"recall", "update", "synth", "objective"} <= prefixes

    def test_has_cross_session_probes(self) -> None:
        content = self._content()
        assert any(spec.cross_session for spec in content.probe_specs)

    def test_frozen_dag_cascade_retraction(self) -> None:
        # A retracted parent must cascade to its dependents: there exists a
        # retract step at which >= 2 facts become invalid together.
        content = self._content()
        world = WorldState(content.events)
        retract_steps = sorted({e.step for e in content.events if e.kind == "retract"})
        cascaded = False
        for step in retract_steps:
            before = set(world.valid_facts_at(step - 1))
            after = set(world.valid_facts_at(step))
            if len(before - after) >= 2:
                cascaded = True
                break
        assert cascaded

    def test_synthetic_only_no_leakage(self) -> None:
        content = self._content()
        for event in content.events:
            lint_no_real_entities(str(event.payload))

    def test_deterministic_across_calls(self) -> None:
        assert self._content(seed=7) == self._content(seed=7)

    def test_distinct_seeds_differ(self) -> None:
        assert self._content(seed=7) != self._content(seed=8)

    def test_generated_update_probe_detects_stale_fact(self) -> None:
        # End-to-end: feed a retracted fact's own value to its update probe and
        # expect a stale-fact drift flag from the checker built off the content.
        content = self._content()
        episode = EpisodeBuilder().build(content, seed=7, scale=self._SCALE)
        checker = ResearchChecker.from_content(content)
        probes = {p.probe_id: p for p in episode.probes}
        update_specs = [s for s in content.probe_specs if s.probe_id.startswith("update")]
        assert update_specs
        spec = update_specs[0]
        target_id = spec.target_fact_ids[0]
        stale_value = next(
            str(e.payload.get("text", ""))
            for e in content.events
            if e.fact_id == target_id and e.kind == "inject"
        )
        result = checker.check(probes[spec.probe_id], stale_value)
        assert any(flag.startswith(f"stale_fact:{target_id}") for flag in result.drift_flags)
        assert result.score < 1.0
