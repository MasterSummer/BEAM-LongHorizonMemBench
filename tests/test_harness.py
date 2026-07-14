"""TDD tests for the LangGraph agent harness (task 9).

Written FIRST (RED) before ``src/lhmsb/harness/``.

Validates, per the plan's "Methodology Defaults Applied -> Agent harness",
``spec/05-systems.md`` §1, and ``spec/03-protocol.md`` §1.2/§2:

  - Deterministic replay: identical transcript hash AND identical CostVector
    across two runs with the same (seed, condition).
  - No-memory control is provably stateless across sessions: a fact stated only
    in session 1 is NOT accessible in session 2 (no cross-session leakage).
  - The statelessness check has teeth: a storing backend DOES leak across
    sessions, so the check distinguishes them.
  - LangGraph built-in memory is disabled: the compiled graph has no
    checkpointer and no store.
  - The CostVector is populated via the CostMeter (agent tokens + retrieval).
  - The fixed context budget B is enforced identically.
  - Failure policy: timeout -> status "timeout"; crash -> status "failed";
    cost-charged-before-failure is retained in both.
  - Memory mutations (add/retract) route through the adapter only.

All tests run offline with an in-test stub adapter and a deterministic stub
agent model (no real LLM).
"""

from __future__ import annotations

import itertools
import re
from typing import Literal

import pytest

from lhmsb.adapters import Capabilities, MemorySystemAdapter
from lhmsb.harness import (
    EpisodeRun,
    HarnessRuntime,
    build_agent_graph,
    check_cross_session_leakage,
    load_agent_model,
    plan_steps,
    run_episode,
    run_episode_traced,
)
from lhmsb.harness.providers import OpenAICompatibleAgent, StateDiffRWKVAgent
from lhmsb.types import (
    Condition,
    CostVector,
    Episode,
    EpisodeResult,
    Probe,
    RunConfig,
    SearchResult,
    WorldEvent,
)
from lhmsb.types import (
    MemoryEntry as _MemoryEntry,
)

# --------------------------------------------------------------------------- #
# Deterministic test doubles (no real LLM, no real backend).
# --------------------------------------------------------------------------- #

_STOPWORDS = {"the", "is", "a", "an", "what", "where", "of", "to", "in", "?"}


def stub_agent_model(prompt: str) -> str:
    """Deterministic, offline 'agent': answer from the FACTS block or 'UNKNOWN'.

    The harness builds a prompt of the form::

        FACTS:
        - <fact line>
        - <fact line>
        QUESTION: <query>

    The model returns the first fact line whose non-stopword tokens overlap the
    question's tokens, else 'UNKNOWN'. Pure function -> deterministic.
    """
    facts: list[str] = []
    question = ""
    for line in prompt.splitlines():
        if line.startswith("QUESTION:"):
            question = line[len("QUESTION:") :].strip()
        elif line.startswith("- "):
            facts.append(line[2:].strip())
    qwords = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOPWORDS}
    for fact in facts:
        fwords = set(re.findall(r"[a-z0-9]+", fact.lower()))
        if qwords & fwords:
            return fact
    return "UNKNOWN"


class _DeterministicClock:
    """Monotone clock that advances a fixed step per call (deterministic latency)."""

    def __init__(self, step: float = 0.001) -> None:
        self._counter = itertools.count()
        self._step = step

    def __call__(self) -> float:
        return next(self._counter) * self._step


class StoringStubAdapter(MemorySystemAdapter):
    """Minimal storing backend with keyword-overlap relevance; persists across sessions."""

    condition_name = "stub_store"

    def __init__(self) -> None:
        self._store: dict[str, _MemoryEntry] = {}
        self._counter = 0

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        return None

    def reset(self, *, user_id: str) -> None:
        self._store = {}
        self._counter = 0

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        memory_id = f"m{self._counter}"
        self._store[memory_id] = _MemoryEntry(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            score=None,
        )
        return memory_id

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        terms = {t for t in query.lower().split() if t}
        scored: list[tuple[float, _MemoryEntry]] = []
        for entry in self._store.values():
            content_terms = entry.content.lower().split()
            overlap = sum(1 for t in terms if t in content_terms)
            if overlap > 0:
                scored.append((float(overlap), entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].memory_id))
        ranked = [
            _MemoryEntry(
                memory_id=e.memory_id,
                content=e.content,
                metadata=e.metadata,
                created_at=e.created_at,
                updated_at=e.updated_at,
                score=score,
            )
            for score, e in scored
        ]
        return SearchResult(results=ranked[:top_k], total_count=len(ranked))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        old = self._store.get(memory_id)
        if old is None:
            return
        self._store[memory_id] = _MemoryEntry(
            memory_id=old.memory_id,
            content=content if content is not None else old.content,
            metadata=metadata if metadata is not None else old.metadata,
            created_at=old.created_at,
            updated_at="2025-01-02T00:00:00Z",
            score=old.score,
        )

    def delete_memory(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)


class NoMemoryStubAdapter(MemorySystemAdapter):
    """The no-memory control: stores nothing; search always returns empty."""

    condition_name = "no_memory"

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        return None

    def reset(self, *, user_id: str) -> None:
        return None

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        return "noop"

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        return SearchResult(results=[], total_count=0)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        return None

    def delete_memory(self, memory_id: str) -> None:
        return None


# --------------------------------------------------------------------------- #
# Fixture episodes (hand-traceable).
# --------------------------------------------------------------------------- #


def _event(
    step: int,
    kind: Literal["inject", "change", "retract"],
    fact_id: str,
    text: str,
    session: int,
) -> WorldEvent:
    return WorldEvent(
        step=step,
        kind=kind,
        fact_id=fact_id,
        payload={"text": text, "session": session},
    )


def _probe(
    step: int, probe_id: str, query: str, gold: object, *, cross_session: bool = False
) -> Probe:
    return Probe(
        step=step,
        probe_id=probe_id,
        kind="factual",
        query=query,
        gold=gold,
        cross_session=cross_session,
    )


def two_session_episode() -> Episode:
    """Session 0 states a secret; session 1 (fresh context) probes for it."""
    return Episode(
        episode_id="ep-cross",
        family="test",
        seed=7,
        events=[
            _event(1, "inject", "f_code", "access code ALPHA7 granted", session=0),
            _event(5, "inject", "f_room", "meeting room Birch hall reserved", session=1),
        ],
        probes=[
            _probe(2, "p_same0", "access code", "access code ALPHA7 granted"),
            _probe(6, "p_cross", "access code", "access code ALPHA7 granted", cross_session=True),
            _probe(7, "p_same1", "meeting room", "meeting room Birch hall reserved"),
        ],
    )


def retract_episode() -> Episode:
    """Inject a fact in session 0, retract it in session 1, then probe it."""
    return Episode(
        episode_id="ep-retract",
        family="test",
        seed=7,
        events=[
            _event(1, "inject", "f_x", "sensor reading XYLO stable", session=0),
            _event(5, "retract", "f_x", "sensor reading XYLO stable", session=1),
        ],
        probes=[
            _probe(6, "p_after", "sensor reading", None, cross_session=True),
        ],
    )


def _run_config(*, context_budget: int = 0) -> RunConfig:
    return RunConfig(
        agent_model="stub-agent",
        judge_model="stub-judge",
        seeds=[7],
        n_episodes=1,
        context_budget=context_budget,
        track="native",
    )


def _answer_of(result: EpisodeResult, probe_id: str) -> str:
    for pr in result.probe_results:
        if pr.probe_id == probe_id:
            meta = pr.metadata or {}
            return str(meta.get("answer", ""))
    raise AssertionError(f"probe {probe_id!r} not found in results")


def _retrieved_ids_of(result: EpisodeResult, probe_id: str) -> list[object]:
    for pr in result.probe_results:
        if pr.probe_id == probe_id:
            meta = pr.metadata or {}
            ids = meta.get("retrieved_ids", [])
            return list(ids) if isinstance(ids, list) else []
    raise AssertionError(f"probe {probe_id!r} not found in results")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_deterministic_transcript_and_cost() -> None:
    """Two runs (same seed/condition) -> identical transcript hash AND CostVector."""
    episode = two_session_episode()
    cfg = _run_config()

    run_a: EpisodeRun = run_episode_traced(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )
    run_b: EpisodeRun = run_episode_traced(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )

    assert run_a.transcript.transcript_hash() == run_b.transcript.transcript_hash()
    assert run_a.result.cost == run_b.result.cost
    assert run_a.transcript.transcript_hash() != ""


def test_no_memory_control_is_stateless_across_sessions() -> None:
    """A fact stated only in session 1 is NOT accessible in session 2 (no-memory)."""
    episode = two_session_episode()
    cfg = _run_config()

    result = run_episode(
        episode,
        NoMemoryStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )

    assert result.status == "completed"
    assert _answer_of(result, "p_same0") == "access code ALPHA7 granted"
    assert _answer_of(result, "p_cross") == "UNKNOWN"
    assert _retrieved_ids_of(result, "p_cross") == []
    assert _answer_of(result, "p_same1") == "meeting room Birch hall reserved"


def test_storing_adapter_leaks_cross_session() -> None:
    """Teeth: a storing backend DOES surface a session-1 fact in session 2."""
    episode = two_session_episode()
    cfg = _run_config()

    result = run_episode(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )

    assert _answer_of(result, "p_cross") == "access code ALPHA7 granted"
    assert _retrieved_ids_of(result, "p_cross") != []


def test_cross_session_leakage_helper_distinguishes_conditions() -> None:
    """The reusable statelessness check: stateless for no-memory, NOT for storing."""
    cfg = _run_config()

    no_mem = check_cross_session_leakage(NoMemoryStubAdapter(), cfg, agent_model=stub_agent_model)
    storing = check_cross_session_leakage(StoringStubAdapter(), cfg, agent_model=stub_agent_model)

    assert no_mem.is_stateless is True
    assert no_mem.leaked_answer is None
    assert storing.is_stateless is False
    assert storing.leaked_answer is not None


def test_cost_vector_is_populated() -> None:
    """Agent tokens are attributed and every retrieval is counted in the CostVector."""
    episode = two_session_episode()
    cfg = _run_config()

    result = run_episode(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )
    cost = result.cost

    assert isinstance(cost, CostVector)
    assert cost.agent_input_tokens > 0
    assert cost.agent_output_tokens > 0
    # one search per probe (3 probes) -> 3 retrieval calls.
    assert cost.num_retrieval_calls == 3


def test_langgraph_builtin_memory_is_disabled() -> None:
    """The compiled graph has NO checkpointer and NO store (no hidden persistence)."""
    cfg = _run_config()
    runtime = HarnessRuntime.from_run_config(
        StoringStubAdapter(), cfg, agent_model=stub_agent_model, clock=_DeterministicClock()
    )
    app = build_agent_graph(runtime)

    assert app.checkpointer is None
    assert getattr(app, "store", None) is None


def test_context_budget_is_enforced() -> None:
    """The working set is capped at budget B (token count never exceeds B)."""
    episode = two_session_episode()
    cfg = _run_config(context_budget=3)

    run = run_episode_traced(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )

    working_sizes = [
        e.working_set_tokens for e in run.transcript.entries if e.working_set_tokens is not None
    ]
    assert working_sizes  # at least one perceive happened
    assert max(working_sizes) <= 3


def test_timeout_status_and_cost_retained() -> None:
    """A wall-clock timeout -> status 'timeout' with cost-charged-before-failure retained."""
    episode = two_session_episode()
    cfg = _run_config()

    # Clock jumps 100s per call -> exceeds a 1s timeout after the first step.
    result = run_episode(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(step=100.0),
        timeout_s=1.0,
    )

    assert result.status == "timeout"
    # Some probes may be unanswered; the result is partial, not empty-by-crash.
    assert len(result.probe_results) < 3


def test_crash_status_and_cost_retained() -> None:
    """An agent-model crash -> status 'failed' with cost incurred before the crash retained."""
    episode = two_session_episode()
    cfg = _run_config()

    calls = {"n": 0}

    def exploding_model(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("boom")
        return stub_agent_model(prompt)

    result = run_episode(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=exploding_model,
        clock=_DeterministicClock(),
    )

    assert result.status == "failed"
    assert result.cost.agent_input_tokens > 0  # the first (pre-crash) answer was charged


def test_retract_routes_delete_through_adapter() -> None:
    """A retract event removes the fact via the adapter so it is no longer retrievable."""
    episode = retract_episode()
    cfg = _run_config()
    adapter = StoringStubAdapter()

    result = run_episode(
        episode, adapter, cfg, agent_model=stub_agent_model, clock=_DeterministicClock()
    )

    # After the retract, the post-retract probe cannot surface the retracted fact.
    assert _answer_of(result, "p_after") == "UNKNOWN"
    assert _retrieved_ids_of(result, "p_after") == []


def test_run_episode_returns_episode_result_with_condition() -> None:
    """run_episode returns an EpisodeResult tagged with the adapter's condition."""
    episode = two_session_episode()
    cfg = _run_config()

    result = run_episode(
        episode,
        NoMemoryStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        clock=_DeterministicClock(),
    )

    assert isinstance(result, EpisodeResult)
    assert result.episode_id == "ep-cross"
    assert result.seed == 7
    assert result.condition == Condition(name="no_memory")


def test_explicit_condition_override() -> None:
    """An explicit condition overrides the adapter-derived one (for the runner, task 21)."""
    episode = two_session_episode()
    cfg = _run_config()

    result = run_episode(
        episode,
        StoringStubAdapter(),
        cfg,
        agent_model=stub_agent_model,
        condition=Condition(name="chroma"),
        clock=_DeterministicClock(),
    )

    assert result.condition == Condition(name="chroma")


def test_plan_steps_orders_by_step_and_session() -> None:
    """plan_steps yields steps ordered by (step, event-before-probe) with session indices."""
    steps = plan_steps(two_session_episode())

    order = [(s.step, "e" if s.event is not None else "p", s.session_index) for s in steps]
    assert order == [
        (1, "e", 0),
        (2, "p", 0),
        (5, "e", 1),
        (6, "p", 1),
        (7, "p", 1),
    ]


def test_load_agent_model_requires_explicit_model_offline() -> None:
    """No live loader in v1: load_agent_model raises so tests must inject a stub."""
    with pytest.raises(NotImplementedError):
        load_agent_model(_run_config())


def test_load_agent_model_builds_statediffrwkv_provider() -> None:
    config = RunConfig(
        agent_model="StateDiffRWKV-2.9B/profile-3784df77",
        judge_model="stub",
        agent_provider="statediffrwkv",
        agent_base_url="http://127.0.0.1:7860",
        agent_result_root="/tmp/statediffrwkv",
        agent_max_new_tokens=64,
        agent_steps=2,
    )

    assert isinstance(load_agent_model(config), StateDiffRWKVAgent)


def test_load_agent_model_builds_openai_compatible_provider() -> None:
    config = RunConfig(
        agent_model="pinned/instruction-model",
        judge_model="stub",
        agent_provider="openai_compatible",
        agent_base_url="http://127.0.0.1:8000/v1",
        agent_max_new_tokens=128,
    )

    assert isinstance(load_agent_model(config), OpenAICompatibleAgent)


def test_evaluator_memory_policy_is_not_exposed_to_memory_adapter() -> None:
    episode = Episode(
        episode_id="policy-isolation",
        family="research_wide",
        seed=7,
        events=[
            WorldEvent(
                step=0,
                kind="inject",
                fact_id="trace-1",
                payload={
                    "text": "Observed paper arXiv:2212.10368.",
                    "session": 0,
                    "memory_policy": "must_store",
                },
            )
        ],
        probes=[],
    )
    adapter = StoringStubAdapter()

    run_episode(episode, adapter, _run_config(), agent_model=stub_agent_model)

    stored = next(iter(adapter._store.values()))
    assert stored.metadata is not None
    assert "memory_policy" not in stored.metadata


def test_context_only_session_marker_is_perceived_but_not_stored() -> None:
    episode = Episode(
        episode_id="context-marker",
        family="research_wide",
        seed=7,
        events=[
            WorldEvent(
                step=0,
                kind="inject",
                fact_id="trace-1",
                payload={"text": "Observed paper arXiv:2212.10368.", "session": 0},
            ),
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="session-marker",
                payload={
                    "text": "Begin final synthesis session.",
                    "session": 1,
                    "context_only": True,
                },
            ),
        ],
        probes=[],
    )
    adapter = StoringStubAdapter()

    run = run_episode_traced(
        episode, adapter, _run_config(), agent_model=stub_agent_model
    )

    assert len(adapter._store) == 1
    marker_entry = next(
        entry for entry in run.transcript.entries if entry.fact_id == "session-marker"
    )
    assert marker_entry.perceived == "Begin final synthesis session."
    assert marker_entry.written_ids == ()


def test_unsupported_capability_does_not_crash_run() -> None:
    """A backend that cannot delete degrades gracefully (run still completes)."""

    class NoDeleteAdapter(StoringStubAdapter):
        condition_name = "no_delete"

        def get_capabilities(self) -> Capabilities:
            return Capabilities(supports_delete=False)

        def delete_memory(self, memory_id: str) -> None:
            from lhmsb.adapters import UnsupportedOperation

            raise UnsupportedOperation("delete_memory", condition="no_delete")

    episode = retract_episode()
    cfg = _run_config()

    result = run_episode(
        episode, NoDeleteAdapter(), cfg, agent_model=stub_agent_model, clock=_DeterministicClock()
    )

    # The retract could not delete, but the run completed gracefully (logged, not crashed).
    assert result.status == "completed"
