"""TDD tests for lhmsb core types/contracts.

Tests the shared dataclasses that every downstream module depends on.
"""

from dataclasses import FrozenInstanceError

import pytest

from lhmsb.types import (
    Condition,
    CostVector,
    Episode,
    EpisodeResult,
    MemoryEntry,
    Probe,
    ProbeResult,
    RunConfig,
    SearchResult,
    WorldEvent,
)


class TestMemoryEntry:
    """MemoryEntry is the canonical memory item returned by adapters."""

    def test_creation_with_all_fields(self) -> None:
        entry = MemoryEntry(
            memory_id="mem-001",
            content="The sky is blue",
            metadata={"source": "observation"},
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
            score=0.95,
        )
        assert entry.memory_id == "mem-001"
        assert entry.content == "The sky is blue"
        assert entry.metadata == {"source": "observation"}
        assert entry.created_at == "2025-01-01T00:00:00Z"
        assert entry.updated_at == "2025-01-02T00:00:00Z"
        assert entry.score == 0.95

    def test_score_none_allowed(self) -> None:
        entry = MemoryEntry(
            memory_id="mem-002",
            content="Some content",
            metadata=None,
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            score=None,
        )
        assert entry.score is None
        assert entry.metadata is None

    def test_metadata_none_allowed(self) -> None:
        entry = MemoryEntry(
            memory_id="mem-003",
            content="Content",
            metadata=None,
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            score=0.5,
        )
        assert entry.metadata is None

    def test_frozen_prevents_mutation(self) -> None:
        entry = MemoryEntry(
            memory_id="mem-004",
            content="Immutable",
            metadata=None,
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            score=None,
        )
        with pytest.raises(FrozenInstanceError):
            entry.content = "mutated"  # type: ignore[misc]

    def test_timestamps_are_str_not_datetime(self) -> None:
        """Per spec/05-systems.md §1.1, timestamps are ISO-8601 str, NOT datetime."""
        entry = MemoryEntry(
            memory_id="mem-005",
            content="Content",
            metadata=None,
            created_at="2025-06-20T12:00:00Z",
            updated_at="2025-06-20T12:30:00Z",
            score=None,
        )
        assert isinstance(entry.created_at, str)
        assert isinstance(entry.updated_at, str)


class TestSearchResult:
    """SearchResult wraps a list of MemoryEntry with total_count."""

    def test_empty_results(self) -> None:
        sr = SearchResult(results=[], total_count=0)
        assert sr.results == []
        assert sr.total_count == 0

    def test_with_results(self) -> None:
        entries = [
            MemoryEntry(
                memory_id="m1",
                content="A",
                metadata=None,
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                score=0.9,
            ),
            MemoryEntry(
                memory_id="m2",
                content="B",
                metadata=None,
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                score=0.8,
            ),
        ]
        sr = SearchResult(results=entries, total_count=42)
        assert len(sr.results) == 2
        assert sr.total_count == 42

    def test_total_count_can_exceed_results_length(self) -> None:
        """total_count may be > len(results) when backend truncates."""
        sr = SearchResult(
            results=[
                MemoryEntry(
                    memory_id="only",
                    content="x",
                    metadata=None,
                    created_at="2025-01-01T00:00:00Z",
                    updated_at="2025-01-01T00:00:00Z",
                    score=1.0,
                )
            ],
            total_count=100,
        )
        assert len(sr.results) == 1
        assert sr.total_count == 100

    def test_frozen(self) -> None:
        sr = SearchResult(results=[], total_count=0)
        with pytest.raises(FrozenInstanceError):
            sr.total_count = 5  # type: ignore[misc]


class TestWorldEvent:
    """WorldEvent models changes to the exogenous evidence world."""

    def test_inject_event(self) -> None:
        event = WorldEvent(
            step=1, kind="inject", fact_id="ev-001", payload={"text": "Fact A is true."}
        )
        assert event.step == 1
        assert event.kind == "inject"
        assert event.fact_id == "ev-001"
        assert event.payload == {"text": "Fact A is true."}

    def test_change_event(self) -> None:
        event = WorldEvent(
            step=3,
            kind="change",
            fact_id="ev-001",
            payload={"text": "Fact A has been amended."},
        )
        assert event.kind == "change"

    def test_retract_event(self) -> None:
        event = WorldEvent(
            step=5, kind="retract", fact_id="ev-001", payload={"reason": "debunked"}
        )
        assert event.kind == "retract"

    def test_kind_is_literal(self) -> None:
        """kind must be one of 'inject', 'change', 'retract'."""
        event = WorldEvent(step=1, kind="inject", fact_id="f1", payload={})
        assert event.kind in ("inject", "change", "retract")

    def test_frozen(self) -> None:
        event = WorldEvent(step=1, kind="inject", fact_id="f1", payload={})
        with pytest.raises(FrozenInstanceError):
            event.step = 2  # type: ignore[misc]


class TestProbe:
    """Probe is a question/task the agent must answer at a specific step."""

    def test_factual_probe(self) -> None:
        probe = Probe(
            step=2,
            probe_id="p-001",
            kind="factual",
            query="What did Study X conclude?",
            gold="The treatment was effective.",
            cross_session=False,
        )
        assert probe.probe_id == "p-001"
        assert probe.kind == "factual"
        assert probe.gold == "The treatment was effective."
        assert probe.cross_session is False

    def test_synthesis_probe(self) -> None:
        probe = Probe(
            step=4,
            probe_id="p-002",
            kind="synthesis",
            query="Summarize the current understanding of Z.",
            gold={"key_points": ["A", "B", "C"]},
            cross_session=True,
        )
        assert probe.kind == "synthesis"
        assert isinstance(probe.gold, dict)

    def test_behavioral_probe(self) -> None:
        probe = Probe(
            step=6,
            probe_id="p-003",
            kind="behavioral",
            query="Is this consistent with the original objective?",
            gold=True,
            cross_session=True,
        )
        assert probe.kind == "behavioral"
        assert probe.gold is True

    def test_cross_session_flag(self) -> None:
        """cross_session marks probes requiring facts from prior sessions."""
        local = Probe(
            step=1, probe_id="p", kind="factual", query="Q", gold="A", cross_session=False
        )
        cross = Probe(
            step=5, probe_id="p2", kind="factual", query="Q", gold="A", cross_session=True
        )
        assert local.cross_session is False
        assert cross.cross_session is True

    def test_frozen(self) -> None:
        probe = Probe(
            step=1, probe_id="p", kind="factual", query="Q", gold="A", cross_session=False
        )
        with pytest.raises(FrozenInstanceError):
            probe.gold = "mutated"  # type: ignore[misc]


class TestEpisode:
    """Episode is a self-contained task spanning multiple sessions."""

    def test_minimal_episode(self) -> None:
        events = [WorldEvent(step=1, kind="inject", fact_id="f1", payload={})]
        probes = [
            Probe(
                step=1, probe_id="p1", kind="factual", query="Q", gold="A", cross_session=False
            )
        ]
        episode = Episode(
            episode_id="ep-001",
            family="research",
            seed=42,
            events=events,
            probes=probes,
            render=None,
        )
        assert episode.episode_id == "ep-001"
        assert episode.family == "research"
        assert episode.seed == 42
        assert len(episode.events) == 1
        assert len(episode.probes) == 1
        assert episode.render is None

    def test_with_render(self) -> None:
        episode = Episode(
            episode_id="ep-002",
            family="swdev",
            seed=7,
            events=[],
            probes=[],
            render={"step_1": "Rendered text for step 1"},
        )
        assert episode.render == {"step_1": "Rendered text for step 1"}

    def test_frozen(self) -> None:
        episode = Episode(
            episode_id="ep", family="research", seed=1, events=[], probes=[], render=None
        )
        with pytest.raises(FrozenInstanceError):
            episode.seed = 99  # type: ignore[misc]


class TestCostVector:
    """CostVector records the full lifecycle cost of an episode-condition run.

    MUST have exactly 12 fields per spec/02-metrics.md §1.3.
    """

    def test_all_12_fields_exist(self) -> None:
        """Verify the canonical 12 fields from the spec are present."""
        cv = CostVector(
            agent_input_tokens=100,
            agent_output_tokens=200,
            mem_internal_in_tokens=50,
            mem_internal_out_tokens=30,
            embedding_tokens=10,
            embedding_calls=2,
            storage_bytes=4096,
            retrieval_latency_ms=15.5,
            write_latency_ms=8.0,
            update_latency_ms=4.0,
            reflection_tokens=0,
            num_retrieval_calls=3,
        )
        # All 12 fields accessible by name
        assert cv.agent_input_tokens == 100
        assert cv.agent_output_tokens == 200
        assert cv.mem_internal_in_tokens == 50
        assert cv.mem_internal_out_tokens == 30
        assert cv.embedding_tokens == 10
        assert cv.embedding_calls == 2
        assert cv.storage_bytes == 4096
        assert cv.retrieval_latency_ms == 15.5
        assert cv.write_latency_ms == 8.0
        assert cv.update_latency_ms == 4.0
        assert cv.reflection_tokens == 0
        assert cv.num_retrieval_calls == 3

    def test_default_zero_values(self) -> None:
        """All fields should default to 0 (or 0.0 for float fields)."""
        cv = CostVector()
        assert cv.agent_input_tokens == 0
        assert cv.agent_output_tokens == 0
        assert cv.mem_internal_in_tokens == 0
        assert cv.mem_internal_out_tokens == 0
        assert cv.embedding_tokens == 0
        assert cv.embedding_calls == 0
        assert cv.storage_bytes == 0
        assert cv.retrieval_latency_ms == 0.0
        assert cv.write_latency_ms == 0.0
        assert cv.update_latency_ms == 0.0
        assert cv.reflection_tokens == 0
        assert cv.num_retrieval_calls == 0

    def test_addition_fieldwise_sum(self) -> None:
        """CostVector + CostVector → fieldwise sum."""
        a = CostVector(
            agent_input_tokens=100,
            agent_output_tokens=200,
            mem_internal_in_tokens=10,
            mem_internal_out_tokens=20,
            embedding_tokens=5,
            embedding_calls=1,
            storage_bytes=1024,
            retrieval_latency_ms=10.0,
            write_latency_ms=5.0,
            update_latency_ms=2.0,
            reflection_tokens=0,
            num_retrieval_calls=1,
        )
        b = CostVector(
            agent_input_tokens=50,
            agent_output_tokens=100,
            mem_internal_in_tokens=5,
            mem_internal_out_tokens=10,
            embedding_tokens=3,
            embedding_calls=2,
            storage_bytes=512,
            retrieval_latency_ms=7.5,
            write_latency_ms=3.0,
            update_latency_ms=1.0,
            reflection_tokens=0,
            num_retrieval_calls=2,
        )
        total = a + b
        assert total.agent_input_tokens == 150
        assert total.agent_output_tokens == 300
        assert total.mem_internal_in_tokens == 15
        assert total.mem_internal_out_tokens == 30
        assert total.embedding_tokens == 8
        assert total.embedding_calls == 3
        assert total.storage_bytes == 1536
        assert total.retrieval_latency_ms == 17.5
        assert total.write_latency_ms == 8.0
        assert total.update_latency_ms == 3.0
        assert total.reflection_tokens == 0
        assert total.num_retrieval_calls == 3

    def test_addition_returns_new_instance(self) -> None:
        """Addition must return a new CostVector, not mutate."""
        a = CostVector(agent_input_tokens=10)
        b = CostVector(agent_input_tokens=20)
        c = a + b
        assert c is not a
        assert c is not b
        assert c.agent_input_tokens == 30

    def test_total_tokens_helper(self) -> None:
        """total_tokens sums agent+memory+embedding+reflection token fields."""
        cv = CostVector(
            agent_input_tokens=100,
            agent_output_tokens=200,
            mem_internal_in_tokens=50,
            mem_internal_out_tokens=30,
            embedding_tokens=10,
            embedding_calls=2,
            storage_bytes=4096,
            retrieval_latency_ms=15.0,
            write_latency_ms=8.0,
            update_latency_ms=4.0,
            reflection_tokens=5,
            num_retrieval_calls=3,
        )
        # sum of all token fields:
        # 100 + 200 + 50 + 30 + 10 + 5 = 395
        expected = 100 + 200 + 50 + 30 + 10 + 5
        assert cv.total_tokens() == expected

    def test_total_tokens_all_zero(self) -> None:
        cv = CostVector()
        assert cv.total_tokens() == 0

    def test_frozen(self) -> None:
        cv = CostVector(agent_input_tokens=1)
        with pytest.raises(FrozenInstanceError):
            cv.agent_input_tokens = 99  # type: ignore[misc]


class TestCondition:
    """Condition identifies a memory system configuration."""

    def test_name(self) -> None:
        c = Condition(name="no_memory")
        assert c.name == "no_memory"

    def test_chroma(self) -> None:
        c = Condition(name="chroma")
        assert c.name == "chroma"

    def test_frozen(self) -> None:
        c = Condition(name="no_memory")
        with pytest.raises(FrozenInstanceError):
            c.name = "mutated"  # type: ignore[misc]


class TestProbeResult:
    """ProbeResult holds the outcome of a single probe evaluation."""

    def test_creation(self) -> None:
        pr = ProbeResult(probe_id="p-001", score=0.85, is_correct=True, metadata={"hops": 2})
        assert pr.probe_id == "p-001"
        assert pr.score == 0.85
        assert pr.is_correct is True
        assert pr.metadata == {"hops": 2}

    def test_incorrect_result(self) -> None:
        pr = ProbeResult(probe_id="p-002", score=0.0, is_correct=False, metadata=None)
        assert pr.score == 0.0
        assert pr.is_correct is False

    def test_frozen(self) -> None:
        pr = ProbeResult(probe_id="p", score=1.0, is_correct=True, metadata=None)
        with pytest.raises(FrozenInstanceError):
            pr.score = 0.5  # type: ignore[misc]


class TestEpisodeResult:
    """EpisodeResult aggregates results for one episode under one condition."""

    def test_creation(self) -> None:
        cost = CostVector(agent_input_tokens=100, agent_output_tokens=50)
        prs = [ProbeResult(probe_id="p1", score=0.9, is_correct=True, metadata=None)]
        er = EpisodeResult(
            episode_id="ep-001",
            condition=Condition(name="no_memory"),
            seed=42,
            probe_results=prs,
            cost=cost,
            status="completed",
        )
        assert er.episode_id == "ep-001"
        assert er.condition.name == "no_memory"
        assert er.seed == 42
        assert len(er.probe_results) == 1
        assert er.probe_results[0].score == 0.9
        assert er.cost.agent_input_tokens == 100
        assert er.status == "completed"

    def test_failed_status(self) -> None:
        er = EpisodeResult(
            episode_id="ep-002",
            condition=Condition(name="chroma"),
            seed=1,
            probe_results=[],
            cost=CostVector(),
            status="timeout",
        )
        assert er.status == "timeout"

    def test_frozen(self) -> None:
        er = EpisodeResult(
            episode_id="ep",
            condition=Condition(name="no_memory"),
            seed=1,
            probe_results=[],
            cost=CostVector(),
            status="completed",
        )
        with pytest.raises(FrozenInstanceError):
            er.status = "mutated"  # type: ignore[misc]


class TestRunConfig:
    """RunConfig holds the parameters for an experiment run."""

    def test_creation(self) -> None:
        rc = RunConfig(
            agent_model="meta-llama/Llama-3-8B",
            judge_model="lordx64/Qwable-v1",
            seeds=[42, 43, 44],
            n_episodes=20,
            context_budget=32000,
            track="native",
        )
        assert rc.agent_model == "meta-llama/Llama-3-8B"
        assert rc.judge_model == "lordx64/Qwable-v1"
        assert rc.seeds == [42, 43, 44]
        assert rc.n_episodes == 20
        assert rc.context_budget == 32000
        assert rc.track == "native"

    def test_controlled_track(self) -> None:
        rc = RunConfig(
            agent_model="meta-llama/Llama-3-8B",
            judge_model="lordx64/Qwable-v1",
            seeds=[1],
            n_episodes=10,
            context_budget=16000,
            track="controlled",
        )
        assert rc.track == "controlled"

    def test_frozen(self) -> None:
        rc = RunConfig(
            agent_model="m",
            judge_model="j",
            seeds=[1],
            n_episodes=1,
            context_budget=1000,
            track="native",
        )
        with pytest.raises(FrozenInstanceError):
            rc.n_episodes = 99  # type: ignore[misc]
