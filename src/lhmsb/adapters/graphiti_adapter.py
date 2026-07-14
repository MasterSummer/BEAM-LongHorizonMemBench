"""Graphiti/Zep temporal-knowledge-graph adapter (``spec/05-systems.md`` §2.1 ``graphiti``).

Wraps Graphiti behind the synchronous :class:`MemorySystemAdapter` contract.
Graphiti is async and backed by a graph DB (Neo4j/FalkorDB); three design choices
let it fit the benchmark:

1. **Lazy import.** ``graphiti_core`` is imported via ``importlib`` inside
   :meth:`GraphitiAdapter.initialize`, so importing this module never requires the
   optional ``graphiti`` extra — only initializing the adapter does (mirrors
   ``chroma``).
2. **One persistent event loop.** Graphiti's async graph driver binds its
   connection pool to the loop that first drives it, so every coroutine runs on a
   single adapter-owned loop via :meth:`GraphitiAdapter._run` (``asyncio.run`` per
   call would create a fresh loop each time and break the live driver). The loop is
   closed by a ``weakref.finalize`` when the adapter is collected.
3. **Forgetting is structural.** Graphiti auto-invalidates contradicted edges with
   temporal ``valid_at``/``invalid_at`` bounds, so :meth:`GraphitiAdapter.apply_decay`
   (``ForgettingCapability``) is a documented no-op acknowledging the KG owns
   time-validity — there is no score decay to apply.

Behavior is scored, not implementation (``spec/05-systems.md`` §1.1): ``add`` ->
``add_episode``, ``search`` -> ``search`` (mapping each ``EntityEdge`` ->
``MemoryEntry``), ``update`` -> re-``add_episode`` with the same episode uuid,
``delete`` -> ``remove_episode``, ``reset`` removes every episode this adapter added.
Graphiti's internal entity-extraction LLM tokens are counted under ``memory_scope``
via a content-derived proxy (``spec/05-systems.md`` §4) so they are never uncounted.
The adapter is internal-LLM configurable, so native (Graphiti defaults) and
controlled (a caller-pinned ``llm_client``/``embedder`` forwarded via ``initialize``)
tracks are both supported; the chosen track is recorded.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import uuid as _uuid
import weakref
from datetime import UTC, datetime
from time import perf_counter
from types import ModuleType
from typing import Any

from lhmsb.adapters.base import ForgettingCapability, MemorySystemAdapter
from lhmsb.cost import CostMeter, count_tokens
from lhmsb.types import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASSWORD = "lhmsbpass"
_DEFAULT_DB_TIMEOUT_S = 10.0
_SOURCE_DESCRIPTION = "lhmsb"
_FORWARDED_CLIENT_KEYS = ("llm_client", "embedder", "cross_encoder")
_ID_NAMESPACE = _uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


class GraphitiSetupError(RuntimeError):
    """Raised when the Graphiti graph DB cannot be initialized (unreachable or
    timed out). Carries an actionable hint instead of letting the call hang."""


def _load_graphiti() -> ModuleType:
    try:
        return importlib.import_module("graphiti_core")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "GraphitiAdapter requires the optional 'graphiti' extra: "
            "pip install 'lhmsb[graphiti]' (and a running Neo4j/FalkorDB — see "
            "docker/graphiti-compose.yml)."
        ) from exc


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    if not loop.is_closed():
        loop.close()


def _setup_hint(uri: str, problem: str) -> str:
    return (
        f"GraphitiAdapter setup failed: {problem}. Graphiti needs a running graph DB "
        f"at {uri!r}. Start one with `docker compose -f docker/graphiti-compose.yml "
        "up -d`, then pass uri/user/password via initialize(**config)."
    )


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _as_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _iso(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return ""


def _extracted_fact_tokens(result: Any, model: str | None) -> int:
    """Token count of the entity edges Graphiti extracted from an episode (0 if none)."""
    edges = getattr(result, "edges", None)
    if isinstance(edges, list) and edges:
        facts = " ".join(str(getattr(edge, "fact", "")) for edge in edges)
        return count_tokens(facts, model)
    return 0


def _edge_to_entry(edge: Any, rank: int) -> MemoryEntry:
    """Map a Graphiti ``EntityEdge`` to a :class:`MemoryEntry` (edge fact -> content,
    edge uuid -> memory_id, rank -> a descending relevance score)."""
    timestamp = _iso(getattr(edge, "valid_at", None) or getattr(edge, "created_at", None))
    return MemoryEntry(
        memory_id=str(getattr(edge, "uuid", "")),
        content=str(getattr(edge, "fact", "")),
        metadata=None,
        created_at=timestamp,
        updated_at=timestamp,
        score=1.0 / (1.0 + rank),
    )


class GraphitiAdapter(MemorySystemAdapter, ForgettingCapability):
    """Temporal-KG memory backed by Graphiti (Zep). Supports structural forgetting."""

    def __init__(self, cost_meter: CostMeter) -> None:
        self.cost_meter = cost_meter
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._finalizer: weakref.finalize[..., GraphitiAdapter] | None = None
        self._group_id = ""
        self._track = "native"
        self._pinned_model: object = None
        self._uri = _DEFAULT_URI
        self._db_timeout_s = _DEFAULT_DB_TIMEOUT_S
        self._content_cache: dict[str, str] = {}
        self._counter = 0

    @property
    def track(self) -> str:
        """Which comparison track this adapter was initialized for (native/controlled)."""
        return self._track

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        graphiti_core = _load_graphiti()
        self._track = _as_str(config.get("track"), "native")
        self._pinned_model = config.get("pinned_model")
        self._db_timeout_s = _as_float(config.get("db_timeout_s"), _DEFAULT_DB_TIMEOUT_S)
        self._uri = _as_str(config.get("uri"), _DEFAULT_URI)
        user = _as_str(config.get("user"), _DEFAULT_USER)
        password = _as_str(config.get("password"), _DEFAULT_PASSWORD)
        client_kwargs = {key: config[key] for key in _FORWARDED_CLIENT_KEYS if config.get(key)}

        self._group_id = user_id
        self._content_cache = {}
        self._counter = 0
        self._reset_loop()
        try:
            self._client = graphiti_core.Graphiti(self._uri, user, password, **client_kwargs)
        except Exception as exc:
            raise GraphitiSetupError(
                _setup_hint(self._uri, f"failed to construct the Graphiti client ({exc})")
            ) from exc
        self._connect()

    def _connect(self) -> None:
        """First DB round-trip, bounded by a timeout so an unreachable DB fails fast."""
        try:
            self._run(
                asyncio.wait_for(
                    self._client.build_indices_and_constraints(), timeout=self._db_timeout_s
                )
            )
        except GraphitiSetupError:
            raise
        except TimeoutError as exc:
            raise GraphitiSetupError(
                _setup_hint(self._uri, f"no response within {self._db_timeout_s:g}s")
            ) from exc
        except Exception as exc:
            raise GraphitiSetupError(
                _setup_hint(self._uri, f"the graph DB is unreachable ({exc})")
            ) from exc

    def reset(self, *, user_id: str) -> None:
        for memory_id in list(self._content_cache):
            self._remove(memory_id)
        self._content_cache.clear()

    def close(self) -> None:
        """Release the adapter's event loop (optional explicit cleanup)."""
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    # ------------------------------------------------------------------ #
    # Memory operations
    # ------------------------------------------------------------------ #

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        memory_id = self._next_id()
        started = perf_counter()
        with self.cost_meter.memory_scope():
            result = self._run(
                self._client.add_episode(
                    name=f"episode-{memory_id}",
                    episode_body=content,
                    source_description=_SOURCE_DESCRIPTION,
                    reference_time=datetime.now(UTC),
                    group_id=user_id,
                    uuid=memory_id,
                )
            )
            self._record_extraction_cost(content, result)
        self.cost_meter.record_latency("write", (perf_counter() - started) * 1000.0)
        self.cost_meter.add_storage_bytes(len(content.encode("utf-8")))
        self._content_cache[memory_id] = content
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
        self.cost_meter.incr_retrieval()
        started = perf_counter()
        with self.cost_meter.memory_scope():
            self.cost_meter.add_embedding(count_tokens(query, self.cost_meter.model), 1)
            edges = self._run(
                self._client.search(query, group_ids=[user_id], num_results=top_k)
            )
        self.cost_meter.record_latency("retrieval", (perf_counter() - started) * 1000.0)
        edge_list = list(edges) if isinstance(edges, list) else []
        entries = [_edge_to_entry(edge, rank) for rank, edge in enumerate(edge_list)]
        return SearchResult(results=entries[:top_k], total_count=len(edge_list))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if content is None and metadata is None:
            raise ValueError("update_memory requires content and/or metadata.")
        new_content = content if content is not None else self._content_cache.get(memory_id, "")
        started = perf_counter()
        with self.cost_meter.memory_scope():
            result = self._run(
                self._client.add_episode(
                    name=f"episode-{memory_id}",
                    episode_body=new_content,
                    source_description=_SOURCE_DESCRIPTION,
                    reference_time=datetime.now(UTC),
                    group_id=self._group_id,
                    uuid=memory_id,
                )
            )
            self._record_extraction_cost(new_content, result)
        self.cost_meter.record_latency("update", (perf_counter() - started) * 1000.0)
        self._content_cache[memory_id] = new_content

    def delete_memory(self, memory_id: str) -> None:
        self._content_cache.pop(memory_id, None)
        self._remove(memory_id)

    def apply_decay(self, *, user_id: str, **params: object) -> None:
        """Forgetting in Graphiti is structural, not score decay: it sets
        ``valid_at``/``invalid_at`` on edges and auto-invalidates contradicted facts
        when a superseding episode is ingested, so there is no decay step to apply —
        this records that the knowledge graph owns time-validity."""
        logger.info(
            "GraphitiAdapter.apply_decay: temporal validity is auto-managed by the "
            "knowledge graph (valid_at/invalid_at); no score decay applied."
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _next_id(self) -> str:
        """A deterministic episode uuid from (group_id, call counter) so repeated
        runs with the same add order produce identical ids (reproducibility)."""
        self._counter += 1
        return str(_uuid.uuid5(_ID_NAMESPACE, f"{self._group_id}:{self._counter}"))

    def _record_extraction_cost(self, content: str, result: Any) -> None:
        """Attribute Graphiti's internal entity-extraction + embedding cost.

        Graphiti runs an LLM to extract entity edges from each episode; with no way
        to read its native token usage we record a content-derived proxy (input ≈
        episode tokens, output ≈ extracted-fact tokens) so the internal LLM cost is
        never silently uncounted (``spec/05-systems.md`` §4)."""
        model = self.cost_meter.model
        in_tokens = count_tokens(content, model)
        out_tokens = _extracted_fact_tokens(result, model) or max(1, in_tokens // 2)
        self.cost_meter.add_memory_internal_tokens(in_tokens, out_tokens)
        self.cost_meter.add_embedding(in_tokens, 1)

    def _remove(self, memory_id: str) -> None:
        """remove_episode by uuid (positional: the live API param is ``episode_uuid``),
        graceful on a missing/unknown id per the delete contract."""
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._run(self._client.remove_episode(memory_id))

    def _reset_loop(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = asyncio.new_event_loop()
        self._finalizer = weakref.finalize(self, _shutdown_loop, self._loop)

    def _run(self, coro: Any) -> Any:
        """Drive a Graphiti coroutine to completion on the adapter's single loop.

        A persistent loop (not ``asyncio.run`` per call) keeps the graph driver's
        loop-bound connection pool valid across calls, and refuses to nest inside an
        already-running loop rather than raising an opaque error."""
        loop = self._loop
        if loop is None:
            raise GraphitiSetupError("GraphitiAdapter is not initialized; call initialize() first.")
        if loop.is_running():
            raise GraphitiSetupError(
                "GraphitiAdapter's sync bridge cannot run inside an active event loop."
            )
        return loop.run_until_complete(coro)
