"""LangGraph agent harness: run an episode under a swappable memory condition.

This is the runnable core of the benchmark (task 9). A minimal, controllable
LangGraph graph (``perceive -> decide -> act -> memory_io``) executes one episode
step at a time. Mandatory mitigations (plan "Methodology Defaults -> Agent
harness", ``spec/05-systems.md`` §1, ``spec/03-protocol.md`` §1.2/§2):

* **Adapter is the only memory path.** Every read/write goes through the injected
  :class:`~lhmsb.adapters.base.MemorySystemAdapter`. LangGraph's built-in
  checkpointer/persistence is NEVER enabled (``.compile()`` is called with no
  ``checkpointer`` / ``store``); :func:`_assert_persistence_disabled` enforces it.
* **No hidden cross-session state.** The working context is the only in-agent
  state and it is cleared at every session boundary, so information from a prior
  session is reachable only if the memory system stored and can retrieve it.
* **Fixed context budget B** (``RunConfig.context_budget``) caps the working set
  identically across conditions.
* **Instrumentation.** Agent-model tokens are counted under ``agent_scope()`` and
  every adapter retrieval/write under ``memory_scope()`` (so a backend's internal
  LLM usage lands in ``mem_internal_*``), producing a per-run ``CostVector``.
* **Determinism.** Fixed seed (``Episode.seed``), a pinned/stub agent model, and
  an injectable clock make the transcript hash and ``CostVector`` reproducible.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lhmsb.adapters import MemorySystemAdapter, UnsupportedOperation
from lhmsb.cost import CostMeter, count_tokens, instrumented_llm
from lhmsb.harness.sessions import Step, plan_steps
from lhmsb.harness.transcript import EpisodeRun, Transcript, TranscriptEntry
from lhmsb.rng import seeded_rng
from lhmsb.types import (
    Condition,
    Episode,
    EpisodeResult,
    ProbeResult,
    RunConfig,
    WorldEvent,
)

logger = logging.getLogger(__name__)

#: An agent model: maps a prompt to a completion. Tests inject a deterministic
#: stub; the experiment runner (task 21) injects the pinned cluster model.
AgentModel = Callable[[str], str]
#: A monotonic clock returning seconds; injectable for deterministic latency.
Clock = Callable[[], float]


class PaperSearch(Protocol):
    """External, non-memory paper search used by research-family episodes."""

    def search(self, query: str, *, top_k: int = 10) -> Sequence[tuple[str, str]]:
        """Return ``(paper_id, display_text)`` candidates in rank order."""
        ...

_DEFAULT_USER = "agent"
_DEFAULT_TOP_K = 10


class HarnessConfigurationError(RuntimeError):
    """Raised when the harness is wired in a way that violates a guardrail."""


class AgentState(TypedDict):
    """Per-step LangGraph state (no persistence; passed in full each invoke)."""

    session_id: str
    working_context: list[str]
    event_text: str | None
    event_kind: str | None
    store_text: str | None
    store_metadata: dict[str, object] | None
    retract_text: str | None
    probe_query: str | None
    retrieved_contents: list[str]
    retrieved_ids: list[str]
    answer: str | None
    written_ids: list[str]
    working_set_tokens: int


_AgentGraph: TypeAlias = CompiledStateGraph[AgentState, None, AgentState, AgentState]


@dataclass
class HarnessRuntime:
    """Episode-constant dependencies shared by every graph node."""

    adapter: MemorySystemAdapter
    meter: CostMeter
    agent_model: AgentModel
    model_name: str | None
    context_budget: int
    user_id: str
    clock: Clock
    seed: int
    rng: random.Random
    paper_search: PaperSearch | None = None
    top_k: int = _DEFAULT_TOP_K

    @classmethod
    def from_run_config(
        cls,
        adapter: MemorySystemAdapter,
        run_config: RunConfig,
        *,
        agent_model: AgentModel,
        clock: Clock = time.monotonic,
        user_id: str = _DEFAULT_USER,
        seed: int | None = None,
        paper_search: PaperSearch | None = None,
    ) -> HarnessRuntime:
        """Build a runtime from a run config (pins the seeded RNG + tokenizer)."""
        effective_seed = (
            seed if seed is not None else (run_config.seeds[0] if run_config.seeds else 0)
        )
        return cls(
            adapter=adapter,
            meter=CostMeter(model=run_config.agent_model),
            agent_model=agent_model,
            model_name=run_config.agent_model,
            context_budget=run_config.context_budget,
            user_id=user_id,
            clock=clock,
            seed=effective_seed,
            rng=seeded_rng(effective_seed),
            paper_search=paper_search,
        )


def load_agent_model(run_config: RunConfig) -> AgentModel:
    """Build the explicitly configured live provider for ``run_config``."""
    from lhmsb.harness.providers import OpenAICompatibleAgent, StateDiffRWKVAgent

    provider = run_config.agent_provider.strip().lower()
    if provider in {"", "unconfigured", "none"}:
        raise NotImplementedError(
            "No live agent-model provider is configured. Set agent_provider and "
            f"agent_base_url for the pinned model {run_config.agent_model!r}, or pass "
            "agent_model=<callable> explicitly for an offline test."
        )
    if not run_config.agent_base_url.strip():
        raise HarnessConfigurationError(
            f"agent_base_url is required for provider {provider!r}"
        )
    if provider == "openai_compatible":
        api_key = ""
        if run_config.agent_api_key_env:
            api_key = os.environ.get(run_config.agent_api_key_env, "")
            if not api_key:
                raise HarnessConfigurationError(
                    f"agent API key environment variable {run_config.agent_api_key_env!r} "
                    "is not set"
                )
        return OpenAICompatibleAgent(
            base_url=run_config.agent_base_url,
            model=run_config.agent_model,
            api_key=api_key,
            max_new_tokens=run_config.agent_max_new_tokens,
            temperature=run_config.agent_temperature,
            timeout_seconds=run_config.agent_timeout_seconds,
        )
    if provider == "statediffrwkv":
        return StateDiffRWKVAgent(
            base_url=run_config.agent_base_url,
            result_root=run_config.agent_result_root or None,
            max_new_tokens=run_config.agent_max_new_tokens,
            steps=run_config.agent_steps,
            seed=run_config.agent_seed,
            precision=run_config.agent_precision,
            memory_limit_gb=run_config.agent_memory_limit_gb,
            timeout_seconds=run_config.agent_timeout_seconds,
            poll_interval_seconds=run_config.agent_poll_interval_seconds,
        )
    raise HarnessConfigurationError(f"unsupported agent_provider: {provider!r}")


def _render_event(event: WorldEvent) -> str:
    text = event.payload.get("text")
    rendered = text if isinstance(text, str) else event.fact_id
    raw_ids = event.payload.get("arxiv_ids")
    if isinstance(raw_ids, list):
        paper_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        if paper_ids:
            rendered = f"{rendered} | " + " | ".join(
                f"arXiv:{paper_id}" for paper_id in paper_ids
            )
    return rendered


def _working_tokens(items: list[str], model: str | None) -> int:
    return count_tokens("\n".join(items), model)


def _enforce_budget(items: list[str], budget: int, model: str | None) -> list[str]:
    """Evict oldest perceived items until the working set fits the token budget B."""
    if budget <= 0:
        return items
    trimmed = list(items)
    while trimmed and _working_tokens(trimmed, model) > budget:
        trimmed.pop(0)
    return trimmed


def _build_prompt(retrieved: list[str], working_context: list[str], query: str) -> str:
    facts: list[str] = []
    seen: set[str] = set()
    for fact in (*retrieved, *working_context):
        if fact not in seen:
            seen.add(fact)
            facts.append(fact)
    body = "\n".join(f"- {fact}" for fact in facts)
    return f"FACTS:\n{body}\nQUESTION: {query}"


def _str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def _to_mapping(state: object) -> dict[str, object]:
    """Normalize a LangGraph output state to ``dict[str, object]`` (no Any leak)."""
    out: dict[str, object] = {}
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(key, str):
                out[key] = value
    return out


def build_agent_graph(runtime: HarnessRuntime) -> _AgentGraph:
    """Compile the perceive->decide->act->memory_io graph WITHOUT persistence."""
    wrapped_model = instrumented_llm(runtime.agent_model, runtime.meter, model=runtime.model_name)

    def _search(query: str, session_id: str) -> list[tuple[str, str]]:
        meter = runtime.meter
        start = runtime.clock()
        with meter.memory_scope():
            result = runtime.adapter.search(
                query, user_id=runtime.user_id, session_id=session_id, top_k=runtime.top_k
            )
        meter.incr_retrieval()
        meter.record_latency("retrieval", (runtime.clock() - start) * 1000.0)
        hits = [(entry.memory_id, entry.content) for entry in result.results]
        if runtime.paper_search is not None:
            hits.extend(
                (f"paper:{paper_id}", content)
                for paper_id, content in runtime.paper_search.search(
                    query, top_k=runtime.top_k
                )
            )
        return hits

    def perceive(state: AgentState) -> dict[str, object]:
        working = list(state["working_context"])
        event_text = state["event_text"]
        if event_text is not None:
            working.append(event_text)
            working = _enforce_budget(working, runtime.context_budget, runtime.model_name)
        return {
            "working_context": working,
            "working_set_tokens": _working_tokens(working, runtime.model_name),
        }

    def decide(state: AgentState) -> dict[str, object]:
        query = state["probe_query"]
        if query is None:
            return {"retrieved_contents": [], "retrieved_ids": []}
        hits = _search(query, state["session_id"])
        return {
            "retrieved_ids": [memory_id for memory_id, _ in hits],
            "retrieved_contents": [content for _, content in hits],
        }

    def act(state: AgentState) -> dict[str, object]:
        query = state["probe_query"]
        if query is None:
            return {"answer": None}
        prompt = _build_prompt(state["retrieved_contents"], state["working_context"], query)
        with runtime.meter.agent_scope():
            answer = wrapped_model(prompt)
        return {"answer": answer}

    def memory_io(state: AgentState) -> dict[str, object]:
        session_id = state["session_id"]
        meter = runtime.meter
        store_text = state["store_text"]
        if store_text is not None:
            start = runtime.clock()
            with meter.memory_scope():
                memory_id = runtime.adapter.add_memory(
                    store_text,
                    user_id=runtime.user_id,
                    session_id=session_id,
                    metadata=state["store_metadata"],
                )
            meter.record_latency("write", (runtime.clock() - start) * 1000.0)
            return {"written_ids": [memory_id]}

        retract_text = state["retract_text"]
        if retract_text is None:
            return {"written_ids": []}

        deleted: list[str] = []
        for memory_id, _ in _search(retract_text, session_id):
            try:
                start = runtime.clock()
                with meter.memory_scope():
                    runtime.adapter.delete_memory(memory_id)
                meter.record_latency("update", (runtime.clock() - start) * 1000.0)
                deleted.append(memory_id)
            except UnsupportedOperation:
                logger.warning("delete unsupported during retract; cannot remove %s", memory_id)
        return {"written_ids": deleted}

    graph = StateGraph(AgentState)
    graph.add_node("perceive", perceive)
    graph.add_node("decide", decide)
    graph.add_node("act", act)
    graph.add_node("memory_io", memory_io)
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "decide")
    graph.add_edge("decide", "act")
    graph.add_edge("act", "memory_io")
    graph.add_edge("memory_io", END)
    return graph.compile()


def _assert_persistence_disabled(app: _AgentGraph) -> None:
    if app.checkpointer is not None:
        raise HarnessConfigurationError(
            "LangGraph checkpointer must be disabled: all cross-session state flows "
            "through the MemorySystemAdapter, never a framework checkpointer."
        )
    if getattr(app, "store", None) is not None:
        raise HarnessConfigurationError(
            "LangGraph store must be disabled: no framework-level persistence is allowed."
        )


def _resolve_condition(adapter: MemorySystemAdapter, condition: Condition | None) -> Condition:
    if condition is not None:
        return condition
    name = getattr(adapter, "condition_name", None)
    if isinstance(name, str) and name:
        return Condition(name=name)
    return Condition(name=type(adapter).__name__)


def _session_id(session_index: int) -> str:
    return f"s{session_index}"


def _initial_state(step: Step, working_context: list[str], session_id: str) -> AgentState:
    event_text: str | None = None
    event_kind: str | None = None
    store_text: str | None = None
    store_metadata: dict[str, object] | None = None
    retract_text: str | None = None
    probe_query: str | None = None

    if step.event is not None:
        event = step.event
        event_kind = event.kind
        rendered = _render_event(event)
        store_metadata = {
            "fact_id": event.fact_id,
            "kind": event.kind,
            "session": step.session_index,
            "step": event.step,
        }
        if event.kind == "retract":
            retract_text = rendered
        else:
            event_text = rendered
            if event.payload.get("context_only") is not True:
                store_text = rendered
    if step.probe is not None:
        probe_query = step.probe.query

    return {
        "session_id": session_id,
        "working_context": list(working_context),
        "event_text": event_text,
        "event_kind": event_kind,
        "store_text": store_text,
        "store_metadata": store_metadata,
        "retract_text": retract_text,
        "probe_query": probe_query,
        "retrieved_contents": [],
        "retrieved_ids": [],
        "answer": None,
        "written_ids": [],
        "working_set_tokens": 0,
    }


def _record_step(step: Step, out: dict[str, object]) -> tuple[TranscriptEntry, ProbeResult | None]:
    working_set_tokens = _int(out.get("working_set_tokens"))
    if step.event is not None:
        event = step.event
        perceived = None if event.kind == "retract" else _render_event(event)
        entry = TranscriptEntry(
            step=step.step,
            session_index=step.session_index,
            kind="event",
            event_kind=event.kind,
            fact_id=event.fact_id,
            perceived=perceived,
            written_ids=tuple(_str_list(out.get("written_ids"))),
            working_set_tokens=working_set_tokens,
        )
        return entry, None

    probe = step.probe
    assert probe is not None  # plan_steps guarantees exactly one of event/probe
    answer = _opt_str(out.get("answer"))
    retrieved_ids = tuple(_str_list(out.get("retrieved_ids")))
    entry = TranscriptEntry(
        step=step.step,
        session_index=step.session_index,
        kind="probe",
        probe_id=probe.probe_id,
        query=probe.query,
        retrieved_ids=retrieved_ids,
        answer=answer,
        working_set_tokens=working_set_tokens,
    )
    probe_result = ProbeResult(
        probe_id=probe.probe_id,
        score=0.0,
        is_correct=False,
        metadata={
            "answer": answer,
            "query": probe.query,
            "retrieved_ids": list(retrieved_ids),
            "kind": probe.kind,
            "cross_session": probe.cross_session,
        },
    )
    return entry, probe_result


def run_episode_traced(
    episode: Episode,
    adapter: MemorySystemAdapter,
    run_config: RunConfig,
    *,
    agent_model: AgentModel | None = None,
    condition: Condition | None = None,
    timeout_s: float | None = None,
    clock: Clock | None = None,
    paper_search: PaperSearch | None = None,
) -> EpisodeRun:
    """Run an episode and return the scored result PLUS its execution transcript."""
    model = agent_model if agent_model is not None else load_agent_model(run_config)
    active_clock: Clock = clock if clock is not None else time.monotonic
    runtime = HarnessRuntime.from_run_config(
        adapter,
        run_config,
        agent_model=model,
        clock=active_clock,
        seed=episode.seed,
        paper_search=paper_search,
    )
    app = build_agent_graph(runtime)
    _assert_persistence_disabled(app)

    adapter.initialize(user_id=runtime.user_id, session_id=_session_id(0))
    adapter.reset(user_id=runtime.user_id)

    entries: list[TranscriptEntry] = []
    probe_results: list[ProbeResult] = []
    status = "completed"
    working_context: list[str] = []
    current_session: int | None = None
    start = active_clock() if timeout_s is not None else 0.0

    for step in plan_steps(episode):
        if step.session_index != current_session:
            current_session = step.session_index
            working_context = []
        if timeout_s is not None and (active_clock() - start) > timeout_s:
            status = "timeout"
            break
        initial = _initial_state(step, working_context, _session_id(step.session_index))
        try:
            out = _to_mapping(app.invoke(initial))
        except Exception as exc:  # crash policy: any failure -> task failure, cost retained
            status = "failed"
            entries.append(
                TranscriptEntry(
                    step=step.step,
                    session_index=step.session_index,
                    kind="error",
                    error=type(exc).__name__,
                )
            )
            logger.warning("episode %s crashed at step %d: %r", episode.episode_id, step.step, exc)
            break
        working_context = _str_list(out.get("working_context"))
        entry, probe_result = _record_step(step, out)
        entries.append(entry)
        if probe_result is not None:
            probe_results.append(probe_result)

    result = EpisodeResult(
        episode_id=episode.episode_id,
        condition=_resolve_condition(adapter, condition),
        seed=episode.seed,
        probe_results=probe_results,
        cost=runtime.meter.to_cost_vector(),
        status=status,
        execution={
            "written_memory_ids": [
                memory_id
                for entry in entries
                if entry.kind == "event"
                for memory_id in (entry.written_ids or ())
            ],
            "retrieved_memory_ids": [
                memory_id
                for entry in entries
                if entry.kind == "probe"
                for memory_id in (entry.retrieved_ids or ())
            ],
            "storage_trace": [
                {
                    "step": entry.step,
                    "session": entry.session_index,
                    "fact_id": entry.fact_id,
                    "written_ids": list(entry.written_ids or ()),
                }
                for entry in entries
                if entry.kind == "event"
            ],
            "retrieval_trace": [
                {
                    "step": entry.step,
                    "session": entry.session_index,
                    "probe_id": entry.probe_id,
                    "retrieved_ids": list(entry.retrieved_ids or ()),
                }
                for entry in entries
                if entry.kind == "probe"
            ],
        },
    )
    return EpisodeRun(result=result, transcript=Transcript(entries=entries))


def run_episode(
    episode: Episode,
    adapter: MemorySystemAdapter,
    run_config: RunConfig,
    *,
    agent_model: AgentModel | None = None,
    condition: Condition | None = None,
    timeout_s: float | None = None,
    clock: Clock | None = None,
    paper_search: PaperSearch | None = None,
) -> EpisodeResult:
    """Run an episode under a memory condition and return the scored result.

    Honors the failure policy: timeout -> ``status="timeout"``; crash ->
    ``status="failed"``; cost-charged-before-failure is retained in both.
    """
    return run_episode_traced(
        episode,
        adapter,
        run_config,
        agent_model=agent_model,
        condition=condition,
        timeout_s=timeout_s,
        clock=clock,
        paper_search=paper_search,
    ).result
