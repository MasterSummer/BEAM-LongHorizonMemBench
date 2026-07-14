"""Cognee triple-store self-reorganizing memory adapter (``spec/05-systems.md`` §2.1 ``cognee``).

Wraps Cognee behind the synchronous :class:`MemorySystemAdapter` contract. Cognee is a
self-organizing memory layer: ``remember`` ingests data and runs the ``cognify`` 6-stage
pipeline (chunk -> extract entities -> build a knowledge graph) with an INTERNAL LLM +
embedder, ``recall`` queries that graph, and ``memify`` / ``improve`` re-organize and
self-improve the graph. Four design choices let it fit the benchmark and keep cost
accounting honest:

1. **Lazy import.** ``cognee`` is imported via ``importlib`` inside
   :meth:`CogneeAdapter.initialize`, so importing this module never requires the optional
   ``cognee`` extra — only initializing the adapter does (mirrors ``chroma`` / ``graphiti``
   / ``mem0``). The imported module is annotated ``Any`` so mypy-strict needs no stubs and
   no ``# type: ignore``.
2. **One persistent event loop.** Cognee's file-based async backends (LanceDB + Kuzu +
   SQLite) bind work to the loop that first drives them, so every coroutine runs on a
   single adapter-owned loop via :meth:`CogneeAdapter._run` (``asyncio.run`` per call would
   create a fresh loop each time and break the file-based drivers — the same failure mode
   solved in ``graphiti_adapter``). The loop + a per-instance temp data dir are released by
   a ``weakref.finalize`` when the adapter is collected.
3. **Zero-infra file-based defaults.** ``initialize`` points Cognee at a per-instance temp
   directory (``system_root_directory`` / ``data_root_directory``) so the LanceDB/Kuzu/SQLite
   stores are local files — no external DB server or network for the offline path.
4. **Internal LLM + embedding cost is always counted.** Cognee's ``cognify`` / ``memify`` /
   ``improve`` LLM + embedder calls happen inside the backend with its own client our
   wrappers cannot observe, so the backend calls are wrapped in ``memory_scope()`` /
   ``reflection_scope()`` and their cost is recorded with the **direct** scope-independent
   accumulators (``add_memory_internal_tokens`` / ``add_embedding`` / ``add_reflection_tokens``)
   as a content-derived proxy, so the internal cost is never silently uncounted
   (``spec/05-systems.md`` §4) and a ``strict_instrumentation`` meter never trips.

Behavior is scored, not implementation (``spec/05-systems.md`` §1.1): ``add`` ->
``remember`` (ingest + ``cognify``), ``search`` -> ``recall``, ``update`` -> ``forget`` the
old data point + re-``remember`` (the ``memory_id`` stays stable), ``delete`` -> ``forget``
by data id, ``reset`` -> ``forget`` the user's dataset. The optional
:class:`ReflectionCapability` maps to Cognee's distinctive self-reorganization:
``reflect`` runs ``memify`` (graph re-organization, billed to ``mem_internal_*``) then
``improve`` (self-improvement, billed to the dedicated ``reflection_tokens`` field). The
``memory_id`` is a deterministic namespaced UUID (``uuid5(NS, f"{user}:{counter}")``) so
repeated runs with the same add order produce identical ids (reproducibility) — never the
backend's random ids. The adapter is internal-LLM configurable, so native (Cognee defaults)
and controlled (a caller-pinned model forwarded via ``config.set_llm_config``) tracks are
both supported; the chosen track is recorded. No model is ever hard-coded.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import tempfile
import uuid as _uuid
import weakref
from hashlib import sha256
from shutil import rmtree
from time import perf_counter
from types import ModuleType
from typing import Any

from lhmsb.adapters.base import (
    MemorySystemAdapter,
    ReflectionCapability,
    UnsupportedOperation,
)
from lhmsb.cost import CostMeter, count_tokens
from lhmsb.types import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)

#: Fixed namespace for deterministic memory ids (reproducible across identical runs).
_ID_NAMESPACE = _uuid.UUID("c0c0ee00-1111-4222-8333-444455556666")
#: Dataset name prefix; the per-user dataset is this + a hash of the user id.
_DATASET_PREFIX = "lhmsb_"
#: Result keys probed for a row's text / id when mapping a recall row -> MemoryEntry.
_TEXT_KEYS = ("text", "content", "memory", "value", "payload")
_ID_KEYS = ("id", "data_id", "dataset_id", "document_id")


class CogneeSetupError(RuntimeError):
    """Raised when the Cognee backend cannot be initialized (e.g. the temp data dir
    could not be created). Carries an actionable hint instead of failing opaquely."""


def _load_cognee() -> ModuleType:
    try:
        return importlib.import_module("cognee")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "CogneeAdapter requires the optional 'cognee' extra: pip install 'lhmsb[cognee]'"
        ) from exc


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _dataset_name(user_id: str) -> str:
    """Deterministic, valid Cognee dataset name derived from the user id."""
    return _DATASET_PREFIX + sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _row_value(row: object, keys: tuple[str, ...]) -> str:
    """First non-empty value among ``keys`` on a dict or object, as ``str`` (else "")."""
    for key in keys:
        value = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int | float):
            return str(value)
    return ""


def _row_text(row: object) -> str:
    """The memory text of a recall row; a bare string row is its own text."""
    if isinstance(row, str):
        return row
    return _row_value(row, _TEXT_KEYS)


def _extract_data_id(result: object) -> str:
    """The backend data-point id from an ``add``/``remember`` result (``""`` if none).

    Cognee returns pipeline-run info carrying the ingested data id; both dict-shaped and
    object-shaped results are accepted so the adapter survives version drift.
    """
    return _row_value(result, _ID_KEYS)


def _build_llm_config(*, pinned_model: str | None, llm_config: object) -> dict[str, object]:
    """Build the ``config.set_llm_config`` dict for the controlled track.

    Any caller-supplied ``llm_config`` is the base (forwarded verbatim for a live backend);
    the pinned model is injected at ``llm_model`` so the memory system's architecture is
    isolated from its model choice. No model is ever hard-coded — ``pinned_model`` is always
    caller-supplied.
    """
    config: dict[str, object] = {}
    if isinstance(llm_config, dict):
        config = {str(key): value for key, value in llm_config.items()}
    if pinned_model is not None:
        config["llm_model"] = pinned_model
    return config


def _shutdown(loop: asyncio.AbstractEventLoop, data_dir: str) -> None:
    """Close the adapter's loop and remove its temp data dir (GC/finalizer callback)."""
    if not loop.is_closed():
        loop.close()
    rmtree(data_dir, ignore_errors=True)


class CogneeAdapter(MemorySystemAdapter, ReflectionCapability):
    """Self-reorganizing triple-store memory backed by Cognee. Supports reflection
    (``memify`` / ``improve`` self-reorg) with full internal-LLM cost instrumentation and
    native/controlled track support."""

    def __init__(self, cost_meter: CostMeter) -> None:
        self.cost_meter = cost_meter
        self._cognee: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._finalizer: weakref.finalize[..., CogneeAdapter] | None = None
        self._data_dir = ""
        self._dataset = ""
        self._track = "native"
        self._pinned_model: str | None = None
        self._counter = 0
        # Adapter-owned mirrors keyed by the deterministic memory_id: the stored content
        # (for reflect/summarize proxies + update) and the backend data id (for forget).
        self._content: dict[str, str] = {}
        self._data_ids: dict[str, str] = {}
        # Reverse map backend-data-id -> memory_id so recall rows recover the stable id.
        self._mid_by_data: dict[str, str] = {}

    @property
    def track(self) -> str:
        """Which comparison track this adapter was initialized for (native/controlled)."""
        return self._track

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        self._cognee = _load_cognee()
        self._track = _as_str(config.get("track"), "native")
        pinned = config.get("pinned_model")
        self._pinned_model = pinned if isinstance(pinned, str) else None
        self._dataset = _dataset_name(user_id)
        self._counter = 0
        self._content = {}
        self._data_ids = {}
        self._mid_by_data = {}
        self._reset_loop()
        self._configure(config)

    def _configure(self, config: dict[str, object]) -> None:
        """Point Cognee at the per-instance file-based stores and, on the controlled track,
        pin the shared LLM. Native track leaves Cognee's LLM/embedder defaults untouched."""
        cognee_config = getattr(self._cognee, "config", None)
        if cognee_config is not None:
            for setter_name in ("system_root_directory", "data_root_directory"):
                setter = getattr(cognee_config, setter_name, None)
                if callable(setter):
                    setter(self._data_dir)
            if self._track == "controlled":
                llm_config = _build_llm_config(
                    pinned_model=self._pinned_model, llm_config=config.get("llm_config")
                )
                set_llm = getattr(cognee_config, "set_llm_config", None)
                if llm_config and callable(set_llm):
                    set_llm(llm_config)

    def reset(self, *, user_id: str) -> None:
        self._content = {}
        self._data_ids = {}
        self._mid_by_data = {}
        if self._cognee is None:
            return
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._run(self._cognee.forget(dataset=self._dataset, everything=False))

    def close(self) -> None:
        """Release the adapter's event loop + temp data dir (optional explicit cleanup)."""
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        rmtree(self._data_dir, ignore_errors=True)
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
        memory_id = self._next_id(user_id)
        started = perf_counter()
        with self.cost_meter.memory_scope():
            result = self._run(
                self._cognee.remember(
                    content,
                    dataset_name=self._dataset,
                    node_set=[user_id],
                    session_id=session_id,
                )
            )
            self._record_cognify_cost(content)
        self.cost_meter.record_latency("write", (perf_counter() - started) * 1000.0)
        self.cost_meter.add_storage_bytes(len(content.encode("utf-8")))
        self._register(memory_id, content, _extract_data_id(result))
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
            raw = self._run(
                self._cognee.recall(
                    query,
                    datasets=[self._dataset],
                    session_id=session_id,
                    top_k=top_k,
                )
            )
        self.cost_meter.record_latency("retrieval", (perf_counter() - started) * 1000.0)
        rows = list(raw) if isinstance(raw, list) else []
        entries = [self._row_to_entry(row, rank) for rank, row in enumerate(rows)]
        return SearchResult(results=entries[:top_k], total_count=len(rows))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if content is None and metadata is None:
            raise ValueError("update_memory requires content and/or metadata.")
        if content is None:
            # Cognee re-ingests content to rebuild the graph; it has no metadata-only edit
            # path, so that case degrades gracefully (``spec/03-protocol.md`` §5.4).
            raise UnsupportedOperation(
                "update_memory", condition="metadata-only update (Cognee re-ingests content)"
            )
        started = perf_counter()
        with self.cost_meter.memory_scope():
            self._forget_data_id(self._data_ids.get(memory_id))
            result = self._run(
                self._cognee.remember(content, dataset_name=self._dataset, node_set=[])
            )
            self._record_cognify_cost(content)
        self.cost_meter.record_latency("update", (perf_counter() - started) * 1000.0)
        old_content = self._content.get(memory_id)
        if old_content is not None:
            self._mid_by_data.pop(self._data_ids.get(memory_id, ""), None)
        self._register(memory_id, content, _extract_data_id(result))

    def delete_memory(self, memory_id: str) -> None:
        data_id = self._data_ids.pop(memory_id, None)
        self._content.pop(memory_id, None)
        if data_id is not None:
            self._mid_by_data.pop(data_id, None)
        # Idempotent per the delete contract: forgetting an unknown id is a no-op.
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._forget_data_id(data_id)

    # ------------------------------------------------------------------ #
    # Reflection (Cognee self-reorganization: memify + improve)
    # ------------------------------------------------------------------ #

    def reflect(self, *, user_id: str, session_id: str | None = None) -> None:
        """Trigger Cognee's self-reorganization and count its internal cost.

        ``memify`` re-organizes the knowledge graph (an internal cognify-style LLM pass) ->
        billed to ``mem_internal_*`` under ``memory_scope``; ``improve`` self-improves the
        memory -> billed to the dedicated ``reflection_tokens`` field under
        ``reflection_scope`` (``spec/05-systems.md`` §4). Both use the direct, scope-independent
        accumulators so a ``strict_instrumentation`` meter never trips. Hooks absent on the
        backend degrade to a content-derived proxy (never zero, never uncounted)."""
        corpus = " ".join(self._content.values())
        model = self.cost_meter.model
        graph_tokens = max(1, count_tokens(corpus, model))
        with self.cost_meter.memory_scope():
            self._run_optional("memify", dataset=self._dataset)
            self.cost_meter.add_memory_internal_tokens(graph_tokens, max(1, graph_tokens // 2))
            self.cost_meter.add_embedding(graph_tokens, 1)
        with self.cost_meter.reflection_scope():
            self._run_optional("improve")
            self.cost_meter.add_reflection_tokens(max(1, graph_tokens // 2))

    def summarize(
        self, *, user_id: str, session_id: str | None = None, query: str | None = None
    ) -> str:
        """Concatenate stored memory (optionally query-scoped); small internal cost."""
        parts = [
            content
            for content in self._content.values()
            if query is None or _matches(query, content)
        ]
        summary = " ".join(parts)
        with self.cost_meter.memory_scope():
            self.cost_meter.add_memory_internal_tokens(
                max(1, count_tokens(summary, self.cost_meter.model)), 1
            )
        return summary

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _next_id(self, user_id: str) -> str:
        """A deterministic memory id from (user_id, call counter) so repeated runs with the
        same add order produce identical ids (reproducibility)."""
        self._counter += 1
        return str(_uuid.uuid5(_ID_NAMESPACE, f"{user_id}:{self._counter}"))

    def _register(self, memory_id: str, content: str, data_id: str) -> None:
        """Record the content mirror + backend-data-id mapping for a stored memory."""
        self._content[memory_id] = content
        backend_id = data_id or memory_id
        self._data_ids[memory_id] = backend_id
        self._mid_by_data[backend_id] = memory_id

    def _record_cognify_cost(self, content: str) -> None:
        """Attribute Cognee's internal ``cognify`` LLM + embedding cost.

        ``cognify`` runs an LLM to extract entities and build the graph; with no way to read
        its native token usage we record a content-derived proxy (input ~ ingested tokens,
        output ~ half that for the extracted graph) so the internal cost is never silently
        uncounted (``spec/05-systems.md`` §4)."""
        model = self.cost_meter.model
        in_tokens = count_tokens(content, model)
        self.cost_meter.add_memory_internal_tokens(in_tokens, max(1, in_tokens // 2))
        self.cost_meter.add_embedding(in_tokens, 1)

    def _row_to_entry(self, row: object, rank: int) -> MemoryEntry:
        """Map a recall row to a :class:`MemoryEntry`, recovering the stable ``memory_id``
        from the backend data id where possible (else a content-derived fallback)."""
        text = _row_text(row)
        data_id = _extract_data_id(row)
        memory_id = self._mid_by_data.get(data_id) or self._mid_for_text(text)
        return MemoryEntry(
            memory_id=memory_id,
            content=text,
            metadata=None,
            created_at="",
            updated_at="",
            score=1.0 / (1.0 + rank),
        )

    def _mid_for_text(self, text: str) -> str:
        """Recover the memory id whose stored content equals ``text`` (round-trip support);
        falls back to a deterministic content hash when the backend text is graph-derived."""
        for memory_id, content in self._content.items():
            if content == text:
                return memory_id
        return "cognee-" + sha256(text.encode("utf-8")).hexdigest()[:16]

    def _forget_data_id(self, data_id: str | None) -> None:
        if data_id:
            self._run(self._cognee.forget(data_id=data_id))

    def _run_optional(self, name: str, **kwargs: object) -> None:
        """Await an optional backend coroutine (``memify`` / ``improve``) if present.

        A backend lacking the hook degrades to the content-derived cost proxy recorded by
        the caller, never crashing."""
        fn = getattr(self._cognee, name, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                self._run(fn(**kwargs))

    def _reset_loop(self) -> None:
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        rmtree(self._data_dir, ignore_errors=True)
        try:
            self._data_dir = tempfile.mkdtemp(prefix="lhmsb-cognee-")
        except OSError as exc:  # pragma: no cover - filesystem failure is environmental
            raise CogneeSetupError(f"could not create a Cognee data directory: {exc}") from exc
        self._loop = asyncio.new_event_loop()
        self._finalizer = weakref.finalize(self, _shutdown, self._loop, self._data_dir)

    def _run(self, coro: Any) -> Any:
        """Drive a Cognee coroutine to completion on the adapter's single loop.

        A persistent loop (not ``asyncio.run`` per call) keeps Cognee's file-based async
        drivers valid across calls, and refuses to nest inside an already-running loop rather
        than raising an opaque error (same pattern as ``graphiti_adapter``)."""
        loop = self._loop
        if loop is None:
            raise CogneeSetupError("CogneeAdapter is not initialized; call initialize() first.")
        if loop.is_running():
            raise CogneeSetupError(
                "CogneeAdapter's sync bridge cannot run inside an active event loop."
            )
        return loop.run_until_complete(coro)


_STOPWORDS = frozenset(
    {"the", "is", "are", "was", "were", "of", "a", "an", "to", "in", "on", "and", "for", "no"}
)


def _matches(query: str, text: str) -> bool:
    """Token-overlap relevance: any salient query token present in ``text``."""
    text_lower = text.lower()
    tokens = [tok for tok in query.lower().split() if len(tok) >= 3 and tok not in _STOPWORDS]
    return any(tok in text_lower for tok in tokens)
