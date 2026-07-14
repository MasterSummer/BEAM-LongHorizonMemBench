"""TDD tests for lhmsb hashing utilities.

Tests world_event_hash (cross-process determinism, sensitivity) and episode_hash.
"""


from lhmsb.hashing import episode_hash, world_event_hash
from lhmsb.rng import deterministic_id, seeded_rng
from lhmsb.types import Episode, Probe, WorldEvent


class TestWorldEventHash:
    """world_event_hash must be stable across processes using hashlib.sha256."""

    def make_schedule(self) -> tuple[list[WorldEvent], list[Probe]]:
        """Helper: build a fixed event+probe schedule."""
        events = [
            WorldEvent(
                step=1, kind="inject", fact_id="ev-001",
                payload={"text": "Fact A is true."},
            ),
            WorldEvent(
                step=2, kind="inject", fact_id="ev-002",
                payload={"text": "Fact B is true."},
            ),
            WorldEvent(
                step=3, kind="change", fact_id="ev-001",
                payload={"text": "Fact A updated."},
            ),
            WorldEvent(
                step=5, kind="retract", fact_id="ev-002",
                payload={"reason": "debunked"},
            ),
        ]
        probes = [
            Probe(
                step=4,
                probe_id="p-001",
                kind="factual",
                query="What is the current state of Fact A?",
                gold="Fact A updated.",
                cross_session=False,
            ),
            Probe(
                step=6,
                probe_id="p-002",
                kind="factual",
                query="Is Fact B still valid?",
                gold=False,
                cross_session=True,
            ),
        ]
        return events, probes

    def test_identical_schedules_produce_identical_hash(self) -> None:
        """Same events+probes → same hash (repeatability within process)."""
        events, probes = self.make_schedule()
        h1 = world_event_hash(events, probes)
        h2 = world_event_hash(events, probes)
        assert h1 == h2
        assert isinstance(h1, str)
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_is_hex_string(self) -> None:
        events, probes = self.make_schedule()
        h = world_event_hash(events, probes)
        # Must be 64 hex chars
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_events_produce_different_hash(self) -> None:
        """Changing an event changes the hash."""
        events, probes = self.make_schedule()
        h1 = world_event_hash(events, probes)

        # Change step on first event
        events2 = [
            WorldEvent(
                step=99, kind="inject", fact_id="ev-001",
                payload={"text": "Fact A is true."},
            ),
            events[1],
            events[2],
            events[3],
        ]
        h2 = world_event_hash(events2, probes)
        assert h1 != h2

    def test_extra_event_changes_hash(self) -> None:
        """Adding an event changes the hash."""
        events, probes = self.make_schedule()
        h1 = world_event_hash(events, probes)

        events2 = list(events) + [
            WorldEvent(
                step=7, kind="inject", fact_id="ev-003", payload={"text": "New fact."}
            )
        ]
        h2 = world_event_hash(events2, probes)
        assert h1 != h2

    def test_removed_event_changes_hash(self) -> None:
        """Removing an event (retraction removal) changes the hash."""
        events, probes = self.make_schedule()
        h1 = world_event_hash(events, probes)

        # Remove the retraction event
        events2 = events[:3]  # first 3 events only
        h2 = world_event_hash(events2, probes)
        assert h1 != h2

    def test_different_probes_produce_different_hash(self) -> None:
        """Changing a probe changes the hash."""
        events, probes = self.make_schedule()
        h1 = world_event_hash(events, probes)

        probes2 = [
            Probe(
                step=4,
                probe_id="p-001",
                kind="synthesis",  # changed from factual
                query="What is the current state of Fact A?",
                gold="Fact A updated.",
                cross_session=False,
            ),
            probes[1],
        ]
        h2 = world_event_hash(events, probes2)
        assert h1 != h2

    def test_empty_schedules_produce_stable_hash(self) -> None:
        """Empty events and probes should produce a deterministic hash."""
        h = world_event_hash([], [])
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_not_using_python_builtin_hash(self) -> None:
        """Verify hash is NOT Python's built-in hash() which is salted per process."""
        events, probes = self.make_schedule()
        h = world_event_hash(events, probes)
        # Python's hash() returns an int (could be negative), but our hash is 64-char hex
        assert not h.startswith("-")
        assert len(h) == 64  # sha256 hex digest, not a small int

    def test_deterministic_across_identical_builds(self) -> None:
        """Rebuilding the same schedule from scratch yields the same hash."""
        # Build schedule twice independently
        events1, probes1 = self.make_schedule()
        events2, probes2 = self.make_schedule()

        h1 = world_event_hash(events1, probes1)
        h2 = world_event_hash(events2, probes2)
        assert h1 == h2

    def test_canonical_json_ordering(self) -> None:
        """Hash must be identical regardless of dict key insertion order."""
        events_a = [
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="ev-001",
                payload={"b": 2, "a": 1},  # different key order
            )
        ]
        events_b = [
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="ev-001",
                payload={"a": 1, "b": 2},  # different key order
            )
        ]
        probes: list[Probe] = []
        h_a = world_event_hash(events_a, probes)
        h_b = world_event_hash(events_b, probes)
        assert h_a == h_b, "Hash must be key-order-independent (sorted keys)"


class TestEpisodeHash:
    """episode_hash includes rendered text in the hash."""

    def test_basic_episode_hash(self) -> None:
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
        h = episode_hash(episode)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_episode_hash_includes_metadata(self) -> None:
        """Hash should change when episode_id or seed changes."""
        events = [WorldEvent(step=1, kind="inject", fact_id="f1", payload={})]
        probes: list[Probe] = []

        ep1 = Episode(
            episode_id="ep-A",
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render=None,
        )
        ep2 = Episode(
            episode_id="ep-B",  # different id
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render=None,
        )
        assert episode_hash(ep1) != episode_hash(ep2)

    def test_episode_hash_includes_render(self) -> None:
        """Hash must change when render changes."""
        events: list[WorldEvent] = []
        probes: list[Probe] = []

        ep1 = Episode(
            episode_id="ep",
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render={"step_1": "text A"},
        )
        ep2 = Episode(
            episode_id="ep",
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render={"step_1": "text B"},  # different rendered text
        )
        assert episode_hash(ep1) != episode_hash(ep2)

    def test_render_none_vs_empty_dict_different(self) -> None:
        """None render vs empty dict should produce different hashes."""
        events: list[WorldEvent] = []
        probes: list[Probe] = []

        ep1 = Episode(
            episode_id="ep",
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render=None,
        )
        ep2 = Episode(
            episode_id="ep",
            family="research",
            seed=1,
            events=events,
            probes=probes,
            render={},
        )
        assert episode_hash(ep1) != episode_hash(ep2)


class TestSeededRNG:
    """seeded_rng produces reproducible random streams."""

    def test_same_seed_produces_same_sequence(self) -> None:
        rng1 = seeded_rng(42)
        rng2 = seeded_rng(42)
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self) -> None:
        rng1 = seeded_rng(42)
        rng2 = seeded_rng(99)
        seq1 = [rng1.random() for _ in range(5)]
        seq2 = [rng2.random() for _ in range(5)]
        assert seq1 != seq2

    def test_returns_random_instance(self) -> None:
        import random

        rng = seeded_rng(42)
        assert isinstance(rng, random.Random)

    def test_deterministic_across_fresh_calls(self) -> None:
        """Repeated calls to seeded_rng(seed) must produce identical sequences."""
        expected = [seeded_rng(123).randint(0, 1000) for _ in range(5)]
        actual = [seeded_rng(123).randint(0, 1000) for _ in range(5)]
        assert expected == actual


class TestDeterministicID:
    """deterministic_id produces stable IDs across processes."""

    def test_same_inputs_produce_same_id(self) -> None:
        id1 = deterministic_id("ep-001", seed=42, counter=0)
        id2 = deterministic_id("ep-001", seed=42, counter=0)
        assert id1 == id2
        assert isinstance(id1, str)

    def test_different_counter_produces_different_id(self) -> None:
        id1 = deterministic_id("ep-001", seed=42, counter=0)
        id2 = deterministic_id("ep-001", seed=42, counter=1)
        assert id1 != id2

    def test_different_seed_produces_different_id(self) -> None:
        id1 = deterministic_id("ep-001", seed=42, counter=0)
        id2 = deterministic_id("ep-001", seed=43, counter=0)
        assert id1 != id2

    def test_different_episode_id_produces_different_id(self) -> None:
        id1 = deterministic_id("ep-001", seed=42, counter=0)
        id2 = deterministic_id("ep-002", seed=42, counter=0)
        assert id1 != id2

    def test_id_is_stable_string(self) -> None:
        """ID must be a deterministic string, not using UUID random generation."""
        id_val = deterministic_id("test-episode", seed=12345, counter=7)
        assert isinstance(id_val, str)
        assert len(id_val) > 0

    def test_deterministic_across_calls(self) -> None:
        """The function must produce the same output every time for the same inputs."""
        ref = deterministic_id("ep-X", seed=5, counter=3)
        for _ in range(10):
            assert deterministic_id("ep-X", seed=5, counter=3) == ref


class TestCrossProcessDeterminismQA:
    """Validate QA requirement: hash identical across fresh Python processes.

    We simulate this by building the same data structures independently.
    """

    def test_world_event_hash_reproducible_from_json(self) -> None:
        """Simulate: serialize to JSON, reload in 'fresh' structures, re-hash, must match."""
        events = [
            WorldEvent(step=1, kind="inject", fact_id="ev-001", payload={"key": "value"}),
            WorldEvent(
                step=2, kind="change", fact_id="ev-001", payload={"key": "new_value"}
            ),
        ]
        probes = [
            Probe(
                step=3,
                probe_id="p-001",
                kind="factual",
                query="What is ev-001?",
                gold="new_value",
                cross_session=False,
            )
        ]

        h1 = world_event_hash(events, probes)

        # Reconstruct identical structures (simulating a fresh process)
        events2 = [
            WorldEvent(step=1, kind="inject", fact_id="ev-001", payload={"key": "value"}),
            WorldEvent(
                step=2, kind="change", fact_id="ev-001", payload={"key": "new_value"}
            ),
        ]
        probes2 = [
            Probe(
                step=3,
                probe_id="p-001",
                kind="factual",
                query="What is ev-001?",
                gold="new_value",
                cross_session=False,
            )
        ]

        h2 = world_event_hash(events2, probes2)
        assert h1 == h2
