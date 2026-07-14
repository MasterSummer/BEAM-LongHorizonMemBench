"""TDD tests for the shared simulator core (lhmsb.sim.core).

Covers:
  - WorldState event replay (inject / change / retract; change-on-retracted no-op).
  - EpisodeBuilder gold derivation (value / valid / valid_values) and validation.
  - world_event_hash determinism + sensitivity across builds.
  - DefaultChecker scoring against None / bool / str / list gold.
  - StubRenderer templates, RenderCache write-once semantics, render_episode.
  - validate_render rejecting future-leak and retracted-fact contradictions.

The canonical fixture (per the Task 7 spec):
  inject F@1 {"text": "Fact A is true."}
  inject G@2 {"text": "Fact B is true."}
  retract F@3
  change  G@4 {"text": "Fact B updated."}
  probe@2 derivation="value"  target=[F] -> gold "Fact A is true."
  probe@4 derivation="valid"  target=[F] -> gold False (retracted @3)
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from lhmsb.hashing import world_event_hash
from lhmsb.sim import (
    Checker,
    CheckResult,
    DefaultChecker,
    EpisodeBuilder,
    Fact,
    FamilyContent,
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
from lhmsb.types import Episode, Probe, WorldEvent


def _fixture_events() -> list[WorldEvent]:
    return [
        WorldEvent(step=1, kind="inject", fact_id="F", payload={"text": "Fact A is true."}),
        WorldEvent(step=2, kind="inject", fact_id="G", payload={"text": "Fact B is true."}),
        WorldEvent(step=3, kind="retract", fact_id="F", payload={}),
        WorldEvent(step=4, kind="change", fact_id="G", payload={"text": "Fact B updated."}),
    ]


def _fixture_content() -> FamilyContent:
    probes = [
        ProbeSpec(
            step=2,
            probe_id="p-value-F",
            kind="factual",
            query="What does finding F state?",
            derivation="value",
            target_fact_ids=["F"],
            value_key=None,
            cross_session=False,
        ),
        ProbeSpec(
            step=4,
            probe_id="p-valid-F",
            kind="factual",
            query="Is finding F still valid?",
            derivation="valid",
            target_fact_ids=["F"],
            value_key=None,
            cross_session=False,
        ),
    ]
    return FamilyContent(family="research", events=_fixture_events(), probe_specs=probes)


class TestWorldState:
    """Event replay and point-in-time validity queries."""

    def test_apply_inject_change_retract(self) -> None:
        world = WorldState(_fixture_events())
        world.apply_event(WorldEvent(step=1, kind="inject", fact_id="F", payload={"text": "a"}))
        world.apply_event(WorldEvent(step=2, kind="inject", fact_id="G", payload={"text": "b"}))
        assert set(world.facts) == {"F", "G"}
        world.apply_event(WorldEvent(step=4, kind="change", fact_id="G", payload={"text": "b2"}))
        assert world.facts["G"].payload["text"] == "b2"
        assert world.facts["G"].version == 2
        world.apply_event(WorldEvent(step=3, kind="retract", fact_id="F", payload={}))
        assert set(world.facts) == {"G"}

    def test_change_on_retracted_is_noop(self) -> None:
        world = WorldState([])
        world.apply_event(WorldEvent(step=1, kind="inject", fact_id="F", payload={"text": "a"}))
        world.apply_event(WorldEvent(step=2, kind="retract", fact_id="F", payload={}))
        # change on a retracted/absent fact must be a no-op (not a resurrection).
        world.apply_event(WorldEvent(step=3, kind="change", fact_id="F", payload={"text": "z"}))
        assert "F" not in world.facts

    def test_valid_facts_at_tracks_retraction(self) -> None:
        world = WorldState(_fixture_events())
        assert set(world.valid_facts_at(1)) == {"F"}
        assert set(world.valid_facts_at(2)) == {"F", "G"}
        assert set(world.valid_facts_at(3)) == {"G"}  # F retracted @3
        at4 = world.valid_facts_at(4)
        assert set(at4) == {"G"}
        assert at4["G"].payload["text"] == "Fact B updated."  # change = latest version
        assert at4["G"].version == 2

    def test_fact_is_frozen(self) -> None:
        fact = Fact(fact_id="F", payload={"text": "a"}, version=1)
        with pytest.raises(FrozenInstanceError):
            fact.version = 2  # frozen dataclass forbids reassignment


class TestEpisodeBuilderGold:
    """Probe gold = world state (revealed-minus-retracted) at the probe step."""

    def test_value_gold_when_valid(self) -> None:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        by_id = {p.probe_id: p for p in episode.probes}
        # probe@2 asks for F's value; F valid @2 -> "Fact A is true."
        assert by_id["p-value-F"].gold == "Fact A is true."

    def test_valid_gold_false_after_retraction(self) -> None:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        by_id = {p.probe_id: p for p in episode.probes}
        # probe@4 asks whether F is valid; F retracted @3 -> False
        assert by_id["p-valid-F"].gold is False

    def test_value_gold_none_when_invalid(self) -> None:
        content = replace(
            _fixture_content(),
            probe_specs=[
                ProbeSpec(
                    step=4,
                    probe_id="p-value-F-late",
                    kind="factual",
                    query="What does F state now?",
                    derivation="value",
                    target_fact_ids=["F"],
                )
            ],
        )
        episode = EpisodeBuilder().build(content, seed=1)
        assert episode.probes[0].gold is None  # F retracted @3 -> no current value

    def test_valid_values_gold_lists_only_valid(self) -> None:
        content = replace(
            _fixture_content(),
            probe_specs=[
                ProbeSpec(
                    step=2,
                    probe_id="vv-early",
                    kind="synthesis",
                    query="current findings?",
                    derivation="valid_values",
                    target_fact_ids=["F", "G"],
                ),
                ProbeSpec(
                    step=4,
                    probe_id="vv-late",
                    kind="synthesis",
                    query="current findings?",
                    derivation="valid_values",
                    target_fact_ids=["F", "G"],
                ),
            ],
        )
        episode = EpisodeBuilder().build(content, seed=1)
        by_id = {p.probe_id: p for p in episode.probes}
        assert by_id["vv-early"].gold == ["Fact A is true.", "Fact B is true."]
        assert by_id["vv-late"].gold == ["Fact B updated."]  # F dropped, G is latest

    def test_value_key_override(self) -> None:
        events = [
            WorldEvent(step=1, kind="inject", fact_id="X", payload={"name": "n", "claim": "C"}),
        ]
        content = FamilyContent(
            family="research",
            events=events,
            probe_specs=[
                ProbeSpec(
                    step=1,
                    probe_id="vk",
                    kind="factual",
                    query="q",
                    derivation="value",
                    target_fact_ids=["X"],
                    value_key="name",
                )
            ],
        )
        episode = EpisodeBuilder().build(content, seed=1)
        assert episode.probes[0].gold == "n"  # value_key wins over auto-detect "claim"

    def test_value_autodetect_and_str_fallback(self) -> None:
        events = [
            WorldEvent(step=1, kind="inject", fact_id="A", payload={"claim": "the claim"}),
            WorldEvent(step=1, kind="inject", fact_id="B", payload={"foo": "bar"}),
        ]
        content = FamilyContent(
            family="research",
            events=events,
            probe_specs=[
                ProbeSpec(step=1, probe_id="a", kind="factual", query="q",
                          derivation="value", target_fact_ids=["A"]),
                ProbeSpec(step=1, probe_id="b", kind="factual", query="q",
                          derivation="value", target_fact_ids=["B"]),
            ],
        )
        episode = EpisodeBuilder().build(content, seed=1)
        by_id = {p.probe_id: p for p in episode.probes}
        assert by_id["a"].gold == "the claim"  # auto-detect "claim"
        assert by_id["b"].gold == str({"foo": "bar"})  # str() fallback (no known key)

    def test_episode_render_is_none(self) -> None:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        assert episode.render is None


class TestWorldEventHash:
    """The schedule hash is the counterfactual integrity guarantee."""

    def test_hash_identical_across_rebuilds(self) -> None:
        e1 = EpisodeBuilder().build(_fixture_content(), seed=7)
        e2 = EpisodeBuilder().build(_fixture_content(), seed=7)
        h1 = world_event_hash(e1.events, e1.probes)
        h2 = world_event_hash(e2.events, e2.probes)
        assert h1 == h2
        assert e1.episode_id == e2.episode_id  # deterministic id too

    def test_hash_changes_when_schedule_changes(self) -> None:
        base = EpisodeBuilder().build(_fixture_content(), seed=7)
        mutated_events = _fixture_events()
        mutated_events[1] = WorldEvent(
            step=2, kind="inject", fact_id="G", payload={"text": "DIFFERENT fact B."}
        )
        mutated = EpisodeBuilder().build(
            replace(_fixture_content(), events=mutated_events), seed=7
        )
        h_base = world_event_hash(base.events, base.probes)
        h_mut = world_event_hash(mutated.events, mutated.probes)
        assert h_base != h_mut


class TestScheduleValidation:
    """EpisodeBuilder rejects malformed schedules (strict, unlike WorldState)."""

    def test_double_inject_rejected(self) -> None:
        events = [
            WorldEvent(step=1, kind="inject", fact_id="F", payload={"text": "a"}),
            WorldEvent(step=2, kind="inject", fact_id="F", payload={"text": "b"}),
        ]
        content = FamilyContent(family="research", events=events, probe_specs=[])
        with pytest.raises(ScheduleError):
            EpisodeBuilder().build(content, seed=1)

    def test_change_on_invalid_rejected(self) -> None:
        events = [WorldEvent(step=1, kind="change", fact_id="F", payload={"text": "a"})]
        content = FamilyContent(family="research", events=events, probe_specs=[])
        with pytest.raises(ScheduleError):
            EpisodeBuilder().build(content, seed=1)

    def test_retract_on_invalid_rejected(self) -> None:
        events = [WorldEvent(step=1, kind="retract", fact_id="F", payload={})]
        content = FamilyContent(family="research", events=events, probe_specs=[])
        with pytest.raises(ScheduleError):
            EpisodeBuilder().build(content, seed=1)

    def test_negative_step_rejected(self) -> None:
        events = [WorldEvent(step=-1, kind="inject", fact_id="F", payload={"text": "a"})]
        content = FamilyContent(family="research", events=events, probe_specs=[])
        with pytest.raises(ScheduleError):
            EpisodeBuilder().build(content, seed=1)

    def test_scale_min_facts_violation(self) -> None:
        with pytest.raises(ScheduleError):
            EpisodeBuilder().build(_fixture_content(), seed=1, scale=ScaleParams(min_facts=10))

    def test_reinject_after_retract_allowed(self) -> None:
        events = [
            WorldEvent(step=1, kind="inject", fact_id="F", payload={"text": "a"}),
            WorldEvent(step=2, kind="retract", fact_id="F", payload={}),
            WorldEvent(step=3, kind="inject", fact_id="F", payload={"text": "a2"}),
        ]
        content = FamilyContent(family="research", events=events, probe_specs=[])
        episode = EpisodeBuilder().build(content, seed=1)  # must not raise
        assert episode.family == "research"


class TestDefaultChecker:
    """Programmatic grading of an answer against probe gold."""

    def _probe(self, gold: object) -> Probe:
        return Probe(step=1, probe_id="p", kind="factual", query="q", gold=gold)

    def test_is_a_checker(self) -> None:
        assert isinstance(DefaultChecker(), Checker)

    def test_none_gold_absence_match(self) -> None:
        result = DefaultChecker().check(self._probe(None), "That finding has been retracted.")
        assert result.is_correct
        assert result.score == 1.0

    def test_none_gold_presence_is_wrong(self) -> None:
        result = DefaultChecker().check(self._probe(None), "Fact A is definitely true.")
        assert not result.is_correct
        assert result.score == 0.0

    def test_bool_gold_true(self) -> None:
        assert DefaultChecker().check(self._probe(True), "Yes, it is still valid.").is_correct

    def test_bool_gold_false_with_negation(self) -> None:
        # "no longer valid" must parse False even though it contains "valid".
        assert DefaultChecker().check(self._probe(False), "It is no longer valid.").is_correct

    def test_bool_gold_false_mismatch(self) -> None:
        assert not DefaultChecker().check(self._probe(False), "Yes it is valid").is_correct

    def test_str_gold_exact(self) -> None:
        assert DefaultChecker().check(self._probe("Fact A is true."), "fact a is true.").is_correct

    def test_str_gold_substring(self) -> None:
        probe = self._probe("Fact A is true.")
        assert DefaultChecker().check(probe, "I recall that Fact A is true. Yes.").is_correct

    def test_str_gold_mismatch(self) -> None:
        result = DefaultChecker().check(self._probe("Fact A is true."), "Fact Z is false")
        assert not result.is_correct

    def test_list_gold_fraction(self) -> None:
        probe = self._probe(["alpha", "beta", "gamma"])
        result = DefaultChecker().check(probe, "We confirmed alpha and beta only.")
        assert result.score == pytest.approx(2 / 3)
        assert not result.is_correct

    def test_list_gold_full(self) -> None:
        probe = self._probe(["alpha", "beta"])
        result = DefaultChecker().check(probe, "Both alpha and beta hold.")
        assert result.score == 1.0
        assert result.is_correct

    def test_check_result_fields(self) -> None:
        result = CheckResult(score=1.0, is_correct=True)
        assert result.facts_used == []
        assert result.drift_flags == []
        assert result.metadata == {}


class TestRendering:
    """Deterministic surface rendering + frozen write-once cache."""

    def test_stub_renderer_is_a_surface_renderer(self) -> None:
        assert isinstance(StubRenderer(), SurfaceRenderer)

    def test_stub_renderer_templates(self) -> None:
        renderer = StubRenderer()
        inject_ev = WorldEvent(
            step=1, kind="inject", fact_id="F", payload={"text": "Fact A is true."}
        )
        inject = renderer.render_step(1, [inject_ev], [])
        assert "New finding F:" in inject
        assert "Fact A is true." in inject

        change_ev = WorldEvent(
            step=4, kind="change", fact_id="G", payload={"text": "Fact B updated."}
        )
        change = renderer.render_step(4, [change_ev], [])
        assert "Finding G updated:" in change
        assert "Fact B updated." in change

        retract_ev = WorldEvent(step=3, kind="retract", fact_id="F", payload={})
        retract = renderer.render_step(3, [retract_ev], [])
        assert retract == "Finding F is no longer valid."

        probe = Probe(step=2, probe_id="p", kind="factual", query="What is X?", gold=None)
        probe_text = renderer.render_step(2, [], [probe])
        assert probe_text == "Question: What is X?"

    def test_render_cache_idempotent_write(self) -> None:
        cache = RenderCache()
        cache.put("ep-1", 7, 1, "hello")
        cache.put("ep-1", 7, 1, "hello")  # identical re-write is allowed
        assert cache.get("ep-1", 7, 1) == "hello"

    def test_render_cache_conflicting_write_raises(self) -> None:
        cache = RenderCache()
        cache.put("ep-1", 7, 1, "hello")
        with pytest.raises(RenderCacheError):
            cache.put("ep-1", 7, 1, "DIFFERENT")

    def test_render_episode_populates_render(self) -> None:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        render = render_episode(episode, StubRenderer())
        assert set(render) == {"1", "2", "3", "4"}
        assert episode.render == render  # populated in place
        assert "Fact A is true." in render["1"]
        assert "no longer valid" in render["3"]

    def test_render_episode_with_cache(self) -> None:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        cache = RenderCache()
        render_episode(episode, StubRenderer(), cache=cache)
        assert cache.get(episode.episode_id, 7, 1) == render_episode_text(episode, 1)


def render_episode_text(episode: Episode, step: int) -> str:
    assert episode.render is not None
    value = episode.render[str(step)]
    assert isinstance(value, str)
    return value


class TestValidateRender:
    """Render guard: no future-leak, no retracted-fact contradiction."""

    def _rendered_episode(self) -> Episode:
        episode = EpisodeBuilder().build(_fixture_content(), seed=7)
        render_episode(episode, StubRenderer())
        return episode

    def test_good_render_passes(self) -> None:
        validate_render(self._rendered_episode())  # must not raise

    def test_contradiction_rejected(self) -> None:
        episode = self._rendered_episode()
        assert episode.render is not None
        bad = dict(episode.render)
        # F retracted @3 -> asserting its value @4 with NO retraction marker is a contradiction.
        bad["4"] = "Fact A is true. This continues to hold."
        with pytest.raises(RenderValidationError):
            validate_render(replace(episode, render=bad))

    def test_future_leak_rejected(self) -> None:
        episode = self._rendered_episode()
        assert episode.render is not None
        leak = dict(episode.render)
        # G is first injected @2; leaking its value at step 1 is a future-leak.
        leak["1"] = "Sneak preview: Fact B is true."
        with pytest.raises(RenderValidationError):
            validate_render(replace(episode, render=leak))

    def test_retracted_value_with_marker_is_allowed(self) -> None:
        episode = self._rendered_episode()
        assert episode.render is not None
        ok = dict(episode.render)
        # Same retracted value, but explicitly marked as retracted -> allowed.
        ok["4"] = "Fact A is true. (This was retracted and is no longer valid.)"
        validate_render(replace(episode, render=ok))  # must not raise
