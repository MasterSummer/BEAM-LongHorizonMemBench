"""Contract + behavior tests for the control / baseline / fake adapters (task 12).

Covers the four offline adapters from ``spec/05-systems.md`` §2:

  - ``NoMemoryAdapter`` (control, §2.1 #1) — provably stateless. It deliberately
    stores nothing (``search`` always empty), so it is EXEMPT from the storing
    round-trip checks; it must instead pass the applicable contract subset plus a
    dedicated statelessness proof.
  - ``ChromaAdapter`` (baseline, §2.1 #2) — plain-vector store; full contract.
  - ``FakePerfectAdapter`` / ``FakeBadAdapter`` (§2.2, calibration-only) — full
    contract with an EMPTY oracle store, and a metric-sensitivity proof
    (perfect > bad) with a populated one.

All adapters are offline. ``no_memory`` / ``chroma`` / fakes have no internal LLM,
so their native and controlled tracks coincide (a property asserted directly).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence

import pytest

from contract.adapter_contract import (
    CONTRACT_CHECKS,
    AdapterContractTests,
    AdapterFactory,
    run_contract_suite,
)
from lhmsb.adapters import MemorySystemAdapter
from lhmsb.adapters.chroma import ChromaAdapter
from lhmsb.adapters.fakes import FakeBadAdapter, FakePerfectAdapter, GroundTruthFact
from lhmsb.adapters.no_memory import NoMemoryAdapter
from lhmsb.cost import CostMeter

_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
requires_chroma = pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb extra not installed")

_UID = "task12-user"
_SID = "task12-session"


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def _fake_perfect_factory() -> MemorySystemAdapter:
    return FakePerfectAdapter(facts=())


def _fake_bad_factory() -> MemorySystemAdapter:
    return FakeBadAdapter(facts=())


def _chroma_factory() -> MemorySystemAdapter:
    return ChromaAdapter()


#: Storing backends that must pass the FULL generic contract suite. The fakes are
#: given an EMPTY oracle store so they degrade to a transparent passthrough store.
_STORING: list[object] = [
    pytest.param(_fake_perfect_factory, id="fake_perfect"),
    pytest.param(_fake_bad_factory, id="fake_bad"),
    pytest.param(_chroma_factory, id="chroma", marks=requires_chroma),
]

#: Every task-12 adapter (incl. the non-storing control) for track-coincidence.
_ALL_FACTORIES: list[object] = [
    pytest.param(NoMemoryAdapter, id="no_memory"),
    *_STORING,
]


# --------------------------------------------------------------------------- #
# Storing adapters: full generic contract suite (task 5)
# --------------------------------------------------------------------------- #
class TestFakePerfectContract(AdapterContractTests):
    @staticmethod
    def adapter_factory() -> MemorySystemAdapter:
        return FakePerfectAdapter(facts=())


class TestFakeBadContract(AdapterContractTests):
    @staticmethod
    def adapter_factory() -> MemorySystemAdapter:
        return FakeBadAdapter(facts=())


class TestChromaContract(AdapterContractTests):
    pytestmark = requires_chroma

    @staticmethod
    def adapter_factory() -> MemorySystemAdapter:
        return ChromaAdapter()


@pytest.mark.parametrize("factory", _STORING)
def test_storing_adapter_passes_full_contract(factory: AdapterFactory) -> None:
    run_contract_suite(factory)


# --------------------------------------------------------------------------- #
# NoMemory control: applicable contract subset + statelessness proof
# --------------------------------------------------------------------------- #
#: The control cannot satisfy the storing round-trip / update / delete-removes
#: checks (search is always empty). It must pass everything else.
_NO_MEMORY_APPLICABLE = {
    "search_respects_top_k",
    "delete_is_idempotent",
    "reset_clears",
    "capabilities_introspection",
    "unsupported_op_degrades",
}


@pytest.mark.parametrize(
    "check",
    [pytest.param(fn, id=name) for name, fn in CONTRACT_CHECKS if name in _NO_MEMORY_APPLICABLE],
)
def test_no_memory_applicable_contract(check: Callable[[AdapterFactory], None]) -> None:
    check(NoMemoryAdapter)


def test_no_memory_returns_valid_id_but_stores_nothing() -> None:
    adapter = NoMemoryAdapter()
    adapter.initialize(user_id=_UID, session_id="session-1")
    memory_id = adapter.add_memory(
        "a very secret cross-session fact", user_id=_UID, session_id="session-1"
    )
    assert isinstance(memory_id, str) and memory_id, "add must return a valid non-empty id"

    crossed = adapter.search("secret cross-session fact", user_id=_UID, session_id="session-2")
    assert crossed.results == [], "control leaked memory across sessions"
    assert crossed.total_count == 0


def test_no_memory_is_provably_stateless() -> None:
    adapter = NoMemoryAdapter()
    adapter.initialize(user_id=_UID, session_id="session-1")

    before = repr(sorted(vars(adapter).items()))
    ids = [
        adapter.add_memory(f"fact number {i} payload-{i}", user_id=_UID, session_id="session-1")
        for i in range(8)
    ]
    adapter.update_memory(ids[0], content="changed content")
    adapter.delete_memory(ids[1])
    after = repr(sorted(vars(adapter).items()))

    assert before == after, "NoMemoryAdapter retained cross-session instance state"
    assert "payload-" not in repr(vars(adapter)), "an attribute retained added content"
    assert all(isinstance(i, str) and i for i in ids), "ids must be valid non-empty strings"
    assert len(set(ids)) == len(ids), "ids must be unique despite zero retained state"


def test_no_memory_search_always_empty_after_many_adds() -> None:
    adapter = NoMemoryAdapter()
    adapter.initialize(user_id=_UID)
    for i in range(20):
        adapter.add_memory(f"entry {i}", user_id=_UID, session_id="session-1")
    result = adapter.search("entry", user_id=_UID, session_id="session-1", top_k=10)
    assert result.results == []
    assert result.total_count == 0


# --------------------------------------------------------------------------- #
# Metric sensitivity: FakePerfect must clearly beat FakeBad (else metrics broken)
# --------------------------------------------------------------------------- #
_FIXTURE_FACTS: tuple[GroundTruthFact, ...] = (
    GroundTruthFact(
        "deadline_old", "The project deadline was March 1.", ("project", "deadline"), retracted=True
    ),
    GroundTruthFact("deadline_new", "The project deadline is June 15.", ("project", "deadline")),
    GroundTruthFact(
        "lead_old", "The lead engineer is Alice.", ("lead", "engineer"), retracted=True
    ),
    GroundTruthFact("lead_new", "The lead engineer is Bob.", ("lead", "engineer")),
    GroundTruthFact("budget", "The project budget is 50000 dollars.", ("budget",)),
)
_FIXTURE_PROBES: tuple[tuple[str, str], ...] = (
    ("what is the project deadline", "June 15"),
    ("who is the lead engineer", "Bob"),
    ("what is the budget", "50000"),
)


def _stub_task_score(
    adapter: MemorySystemAdapter, probes: Sequence[tuple[str, str]]
) -> float:
    """A deliberately simple task checker: fraction of probes whose gold answer
    appears in the top-k retrieved memory contents."""
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    correct = 0
    for query, gold in probes:
        result = adapter.search(query, user_id=_UID, session_id=_SID, top_k=3)
        retrieved = " ".join(entry.content for entry in result.results).lower()
        if gold.lower() in retrieved:
            correct += 1
    return correct / len(probes)


def test_fake_perfect_beats_fake_bad_metric_sensitivity() -> None:
    perfect = _stub_task_score(FakePerfectAdapter(facts=_FIXTURE_FACTS), _FIXTURE_PROBES)
    bad = _stub_task_score(FakeBadAdapter(facts=_FIXTURE_FACTS), _FIXTURE_PROBES)

    assert perfect > bad, "metrics are not sensitive: perfect did not beat bad"
    assert perfect - bad >= 0.5, f"margin too small: perfect={perfect}, bad={bad}"
    assert perfect >= 0.8, f"oracle should score near-perfect, got {perfect}"
    assert bad <= 0.2, f"adversary should score near-zero, got {bad}"


def test_fake_bad_returns_retracted_not_current() -> None:
    adapter = FakeBadAdapter(facts=_FIXTURE_FACTS)
    adapter.initialize(user_id=_UID, session_id=_SID)
    result = adapter.search("what is the project deadline", user_id=_UID, session_id=_SID, top_k=3)
    contents = " ".join(e.content for e in result.results)
    assert "March 1" in contents, "fake_bad should surface the retracted (stale) fact"
    assert "June 15" not in contents, "fake_bad must not surface the current fact"


def test_fake_perfect_returns_current_not_retracted() -> None:
    adapter = FakePerfectAdapter(facts=_FIXTURE_FACTS)
    adapter.initialize(user_id=_UID, session_id=_SID)
    result = adapter.search("who is the lead engineer", user_id=_UID, session_id=_SID, top_k=3)
    contents = " ".join(e.content for e in result.results)
    assert "Bob" in contents, "fake_perfect should surface the current fact"
    assert "Alice" not in contents, "fake_perfect must not surface the retracted fact"


# --------------------------------------------------------------------------- #
# Chroma: live offline round-trip + cost instrumentation
# --------------------------------------------------------------------------- #
@requires_chroma
def test_chroma_round_trip_ranks_relevant_top_1() -> None:
    adapter = ChromaAdapter()
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    adapter.add_memory("The capital of France is Paris.", user_id=_UID, session_id=_SID)
    adapter.add_memory(
        "Quokkas are small marsupials from Australia.", user_id=_UID, session_id=_SID
    )
    adapter.add_memory("Python is a popular programming language.", user_id=_UID, session_id=_SID)

    result = adapter.search("what is the capital of France", user_id=_UID, session_id=_SID, top_k=3)
    assert result.results, "chroma returned no results"
    assert "Paris" in result.results[0].content, "the relevant memory did not rank top-1"
    assert result.results[0].score is not None, "search results must carry a relevance score"


@requires_chroma
def test_chroma_counts_embedding_cost_under_memory_scope() -> None:
    meter = CostMeter()
    adapter = ChromaAdapter(meter=meter)
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    adapter.add_memory("The capital of France is Paris.", user_id=_UID, session_id=_SID)
    adapter.search("capital of France", user_id=_UID, session_id=_SID)

    cost = meter.to_cost_vector()
    assert cost.embedding_tokens > 0, "embedding tokens must be counted (add + search)"
    assert cost.embedding_calls >= 2, "expected at least one embed for add and one for search"
    assert cost.num_retrieval_calls >= 1, "search() must increment the retrieval counter"
    assert not meter.has_unscoped(), "all embedding cost must be inside memory_scope"


@requires_chroma
def test_chroma_metadata_round_trips() -> None:
    adapter = ChromaAdapter()
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory(
        "Mission notes.",
        user_id=_UID,
        session_id=_SID,
        metadata={"topic": "mission", "priority": 3},
    )
    result = adapter.search("Mission notes", user_id=_UID, session_id=_SID, top_k=5)
    matched = next(e for e in result.results if e.memory_id == memory_id)
    assert matched.metadata is not None
    assert matched.metadata.get("topic") == "mission"
    assert matched.metadata.get("priority") == 3


# --------------------------------------------------------------------------- #
# Tracks coincide (no internal LLM => native == controlled)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", _ALL_FACTORIES)
def test_tracks_coincide_for_internal_llm_free_adapters(factory: AdapterFactory) -> None:
    def run(track: str) -> list[str]:
        adapter = factory()
        adapter.initialize(user_id=_UID, session_id=_SID, track=track)
        adapter.reset(user_id=_UID)
        adapter.add_memory("alpha beta gamma signal", user_id=_UID, session_id=_SID)
        adapter.add_memory("delta epsilon zeta noise", user_id=_UID, session_id=_SID)
        found = adapter.search("alpha beta signal", user_id=_UID, session_id=_SID, top_k=5)
        return [entry.content for entry in found.results]

    assert run("native") == run("controlled")
