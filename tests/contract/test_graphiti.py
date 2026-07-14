"""Contract + behavior tests for the Graphiti/Zep adapter (task 15).

``graphiti_core`` is NOT installed in CI, so these tests inject an in-memory async
``FakeGraphiti`` via ``sys.modules["graphiti_core"]`` patching (the adapter imports
it lazily inside ``initialize``). The fake mirrors the real API surface the adapter
uses — ``Graphiti(uri, user, password)``, ``build_indices_and_constraints``,
``add_episode``, ``search`` (returning ``EntityEdge``-like objects), and
``remove_episode`` — plus Graphiti's defining feature: temporal auto-invalidation
(a newer fact about the same subject invalidates the older one, which then drops out
of current search results). This is a deterministic stand-in for Graphiti's real
LLM-driven invalidation, exercising the ADAPTER, not Graphiti itself.

The single live test is gated behind ``LHMSB_LIVE_GRAPHITI=1`` + a running graph DB
(``docker/graphiti-compose.yml``).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
from collections.abc import Callable

import pytest

from contract.adapter_contract import run_contract_suite
from lhmsb.adapters import ForgettingCapability, ReflectionCapability, SessionCapability
from lhmsb.adapters.graphiti_adapter import GraphitiAdapter, GraphitiSetupError
from lhmsb.cost import CostMeter

_UID = "graphiti-user"

_STOPWORDS = frozenset(
    {"the", "is", "are", "was", "were", "of", "a", "an", "to", "in", "on", "and", "for", "no"}
)


def _subject(body: str) -> str:
    """A fact's subject: the text before ``::`` if present, else the whole body."""
    head = body.split("::", 1)[0] if "::" in body else body
    return head.strip().lower()


def _tokens(text: str) -> list[str]:
    cleaned = text.lower().replace("::", " ")
    return [tok for tok in cleaned.split() if len(tok) >= 3 and tok not in _STOPWORDS]


def _matches(query: str, fact: str) -> bool:
    fact_lower = fact.lower()
    return any(tok in fact_lower for tok in _tokens(query))


class _FakeEdge:
    def __init__(
        self, uuid: str, fact: str, group_id: str, reference_time: object, seq: int
    ) -> None:
        self.uuid = uuid
        self.fact = fact
        self.group_id = group_id
        self.created_at = reference_time
        self.valid_at = reference_time
        self.invalid_at: object = None
        self.seq = seq


class _FakeEpisode:
    def __init__(self, uuid: str, name: str, group_id: str) -> None:
        self.uuid = uuid
        self.name = name
        self.group_id = group_id


class _FakeAddResult:
    def __init__(self, episode: _FakeEpisode, edges: list[_FakeEdge]) -> None:
        self.episode = episode
        self.edges = edges
        self.nodes: list[object] = []
        self.episodic_edges: list[object] = []
        self.communities: list[object] = []


class FakeGraphiti:
    """In-memory async stand-in for ``graphiti_core.Graphiti``."""

    def __init__(self, uri: str, user: str, password: str, **kwargs: object) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.kwargs = kwargs
        self.built = False
        self._edges: dict[str, _FakeEdge] = {}
        self._seq = 0

    async def build_indices_and_constraints(self) -> None:
        self.built = True

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: object,
        source: object = None,
        group_id: str = "",
        uuid: str | None = None,
        **kwargs: object,
    ) -> _FakeAddResult:
        self._seq += 1
        episode_uuid = uuid or f"fake-ep-{self._seq}"
        if episode_uuid in self._edges:
            # Re-add with an existing uuid is the documented update path: replace
            # the edge in place (no supersession of a different fact).
            edge = self._edges[episode_uuid]
            edge.fact = episode_body
            edge.valid_at = reference_time
            edge.invalid_at = None
            edge.seq = self._seq
        else:
            subject = _subject(episode_body)
            for existing in self._edges.values():
                if (
                    existing.invalid_at is None
                    and existing.group_id == group_id
                    and _subject(existing.fact) == subject
                ):
                    existing.invalid_at = reference_time
            edge = _FakeEdge(episode_uuid, episode_body, group_id, reference_time, self._seq)
            self._edges[episode_uuid] = edge
        return _FakeAddResult(_FakeEpisode(episode_uuid, name, group_id), [edge])

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results: int = 10,
    ) -> list[_FakeEdge]:
        groups = set(group_ids) if group_ids else None
        hits = [
            edge
            for edge in self._edges.values()
            if edge.invalid_at is None
            and (groups is None or edge.group_id in groups)
            and _matches(query, edge.fact)
        ]
        hits.sort(key=lambda edge: edge.seq, reverse=True)
        return hits[:num_results]

    async def remove_episode(self, episode_uuid: str) -> None:
        self._edges.pop(episode_uuid, None)


class _UnreachableGraphiti(FakeGraphiti):
    async def build_indices_and_constraints(self) -> None:
        raise ConnectionRefusedError("could not connect to the graph DB")


class _HangingGraphiti(FakeGraphiti):
    async def build_indices_and_constraints(self) -> None:
        await asyncio.sleep(10)


InjectFn = Callable[[type[FakeGraphiti]], None]


@pytest.fixture
def inject_graphiti(monkeypatch: pytest.MonkeyPatch) -> InjectFn:
    """Install a fake ``graphiti_core`` module exposing the given client class."""

    def _inject(client_cls: type[FakeGraphiti]) -> None:
        module = types.ModuleType("graphiti_core")
        module.__dict__["Graphiti"] = client_cls
        monkeypatch.setitem(sys.modules, "graphiti_core", module)

    return _inject


def _adapter(meter: CostMeter | None = None) -> GraphitiAdapter:
    return GraphitiAdapter(meter if meter is not None else CostMeter())


# --------------------------------------------------------------------------- #
# Generic contract suite (task 5) against the fake
# --------------------------------------------------------------------------- #
def test_graphiti_passes_full_contract(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    run_contract_suite(_adapter)


# --------------------------------------------------------------------------- #
# Temporal auto-invalidation: a retracted/superseded fact drops out of current
# results (Graphiti's defining capability).
# --------------------------------------------------------------------------- #
def test_temporal_invalidation_excludes_retracted_fact(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)

    old_id = adapter.add_memory("deadline :: the project deadline is March 1", user_id=_UID)
    new_id = adapter.add_memory("deadline :: the project deadline is June 15", user_id=_UID)

    result = adapter.search("deadline", user_id=_UID, top_k=10)
    contents = " ".join(entry.content for entry in result.results)
    ids = {entry.memory_id for entry in result.results}

    assert "June 15" in contents, "the current fact must be returned"
    assert "March 1" not in contents, "the superseded fact must be temporally invalidated"
    assert new_id in ids and old_id not in ids
    assert meter.to_cost_vector().mem_internal_in_tokens > 0
    assert not meter.has_unscoped()


# --------------------------------------------------------------------------- #
# Missing / unreachable DB: clear, actionable error — never a silent hang.
# --------------------------------------------------------------------------- #
def test_missing_db_raises_clear_error(inject_graphiti: InjectFn) -> None:
    inject_graphiti(_UnreachableGraphiti)
    adapter = _adapter()
    with pytest.raises(GraphitiSetupError) as exc_info:
        adapter.initialize(user_id=_UID, uri="bolt://localhost:7687")
    message = str(exc_info.value)
    assert "bolt://localhost:7687" in message, "error must name the unreachable URI"
    assert "docker/graphiti-compose.yml" in message, "error must point at the setup recipe"


def test_missing_db_hang_is_bounded_by_timeout(inject_graphiti: InjectFn) -> None:
    inject_graphiti(_HangingGraphiti)
    adapter = _adapter()
    start = time.monotonic()
    with pytest.raises(GraphitiSetupError) as exc_info:
        adapter.initialize(user_id=_UID, db_timeout_s=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"initialize hung for {elapsed:.1f}s; must be bounded by the timeout"
    message = str(exc_info.value)
    assert "0.3s" in message, "error must report the bounding timeout"
    assert "docker/graphiti-compose.yml" in message, "error must point at the setup recipe"


# --------------------------------------------------------------------------- #
# Capabilities + forgetting semantics
# --------------------------------------------------------------------------- #
def test_capabilities_forgetting_only(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    caps = adapter.get_capabilities()

    assert caps.supports_forgetting is True, "Graphiti forgets via temporal invalidation"
    assert caps.supports_reflection is False
    assert caps.supports_sessions is False
    assert isinstance(adapter, ForgettingCapability)
    assert not isinstance(adapter, ReflectionCapability | SessionCapability)


def test_apply_decay_is_documented_noop(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("structural validity is owned by the KG", user_id=_UID)

    adapter.apply_decay(user_id=_UID)

    # Structural forgetting: the fact remains until a superseding episode arrives.
    result = adapter.search("structural validity", user_id=_UID, top_k=10)
    assert memory_id in {entry.memory_id for entry in result.results}


# --------------------------------------------------------------------------- #
# Cost instrumentation: internal extraction + embedding tokens are counted.
# --------------------------------------------------------------------------- #
def test_internal_extraction_tokens_counted(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)

    adapter.add_memory("Acme Corp signed a partnership with Globex in Berlin.", user_id=_UID)
    adapter.search("Acme partnership", user_id=_UID)

    cost = meter.to_cost_vector()
    assert cost.mem_internal_in_tokens > 0, "internal extraction-LLM tokens must be counted"
    assert cost.mem_internal_out_tokens > 0, "extracted-fact output tokens must be counted"
    assert cost.embedding_tokens > 0, "embedding tokens (add + search) must be counted"
    assert cost.num_retrieval_calls >= 1, "search() must increment the retrieval counter"
    assert not meter.has_unscoped(), "all internal cost must land inside memory_scope"


def test_strict_meter_has_no_uncounted_calls(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    meter = CostMeter(strict_instrumentation=True)
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a fact under strict accounting", user_id=_UID)
    adapter.update_memory(memory_id, content="an updated fact under strict accounting")
    adapter.search("strict accounting", user_id=_UID)
    adapter.delete_memory(memory_id)
    assert not meter.has_unscoped()


# --------------------------------------------------------------------------- #
# Native vs controlled track (model pinning forwarded where supported).
# --------------------------------------------------------------------------- #
def test_native_track_passes_no_client(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    assert adapter.track == "native"
    assert "llm_client" not in adapter._client.kwargs


def test_controlled_track_forwards_pinned_client(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)
    sentinel_llm = object()
    sentinel_embedder = object()
    adapter = _adapter()
    adapter.initialize(
        user_id=_UID,
        track="controlled",
        pinned_model="shared/open-weights-model",
        llm_client=sentinel_llm,
        embedder=sentinel_embedder,
    )
    assert adapter.track == "controlled"
    assert adapter._client.kwargs.get("llm_client") is sentinel_llm
    assert adapter._client.kwargs.get("embedder") is sentinel_embedder


# --------------------------------------------------------------------------- #
# Deterministic episode ids (reproducibility).
# --------------------------------------------------------------------------- #
def test_deterministic_episode_ids(inject_graphiti: InjectFn) -> None:
    inject_graphiti(FakeGraphiti)

    def ids() -> list[str]:
        adapter = _adapter()
        adapter.initialize(user_id="repro-user")
        adapter.reset(user_id="repro-user")
        return [adapter.add_memory(f"fact number {i}", user_id="repro-user") for i in range(3)]

    first, second = ids(), ids()
    assert first == second, "episode ids must be deterministic across identical runs"
    assert len(set(first)) == 3, "ids must be unique within a run"


# --------------------------------------------------------------------------- #
# Live test (gated): real Graphiti + a running graph DB.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("LHMSB_LIVE_GRAPHITI") != "1",
    reason="live Graphiti needs LHMSB_LIVE_GRAPHITI=1 + a graph DB (docker/graphiti-compose.yml)",
)
def test_live_graphiti_round_trip_and_temporal_invalidation() -> None:
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(
        user_id="lhmsb-live-user",
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "lhmsbpass"),
        db_timeout_s=30.0,
    )
    try:
        adapter.reset(user_id="lhmsb-live-user")
        adapter.add_memory("Alice is the project lead.", user_id="lhmsb-live-user")
        result = adapter.search("who is the project lead", user_id="lhmsb-live-user", top_k=10)
        assert result.results, "live search returned nothing"
        assert meter.to_cost_vector().mem_internal_in_tokens > 0

        adapter.add_memory(
            "Alice is no longer the lead; Bob is the project lead now.",
            user_id="lhmsb-live-user",
        )
        current = adapter.search("who is the project lead", user_id="lhmsb-live-user", top_k=10)
        contents = " ".join(entry.content for entry in current.results).lower()
        assert "bob" in contents, "the current lead should be retrievable after supersession"
    finally:
        adapter.close()
