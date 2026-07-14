"""Mem0 hybrid-memory adapter (``spec/05-systems.md`` §2.1 ``mem0``).

Wraps Mem0 behind the synchronous :class:`MemorySystemAdapter` contract. Mem0 is
a hybrid semantic + entity memory that runs an INTERNAL LLM on ``add`` to extract
salient facts from raw messages, plus an embedder for vector retrieval. Three
design choices let it fit the benchmark and keep cost accounting honest:

1. **Lazy import.** ``mem0`` is imported via ``importlib`` inside
   :meth:`Mem0Adapter.initialize`, so importing this module never requires the
   optional ``mem0`` extra — only initializing the adapter does (mirrors
   ``chroma`` / ``graphiti``). A typed :class:`_Mem0Backend` ``Protocol`` describes
   the backend surface so the code is mypy-strict clean without ``# type: ignore``.
2. **Internal LLM + embedding cost is always counted.** Mem0's extraction LLM and
   embedder calls happen inside the backend, so the ``add``/``search`` backend
   calls are wrapped in ``memory_scope()`` and the internal cost is recorded as a
   content-derived proxy (input ≈ ingested tokens, output ≈ extracted-fact tokens)
   so the internal LLM cost is never silently uncounted (``spec/05-systems.md`` §4).
3. **Native vs controlled tracks.** Native uses Mem0's default config (its own
   internal model); controlled pins the shared agent model via
   ``Memory.from_config({"llm": {"model": pinned_model}, ...})`` forwarded through
   ``initialize`` (``spec/05-systems.md`` §3). No model is hard-coded — the pinned
   model is always caller-supplied. The chosen track is recorded.

Behavior is scored, not implementation (``spec/05-systems.md`` §1.1): ``add`` ->
``Memory.add``, ``search`` -> ``Memory.search`` (mapping each row -> ``MemoryEntry``),
``update`` -> ``Memory.update`` (content replacement; metadata-only updates degrade
via :class:`UnsupportedOperation`), ``delete`` -> ``Memory.delete``, ``reset`` ->
``Memory.delete_all``. Mem0's reflection is implicit on ``add`` (no explicit
``reflect()`` pass), so this adapter exposes no optional capability mixins.
"""

from __future__ import annotations

import contextlib
import importlib
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol

from lhmsb.adapters.base import MemorySystemAdapter, UnsupportedOperation
from lhmsb.cost import CostMeter, count_tokens
from lhmsb.types import MemoryEntry, SearchResult


class _Mem0Backend(Protocol):
    """Structural type for the ``mem0.Memory`` instance — only the methods the
    adapter calls. Lets mypy check call sites without importing the optional dep."""

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, object] | None = None,
    ) -> object: ...

    def search(self, query: str, *, user_id: str, limit: int = 10) -> object: ...

    def update(self, memory_id: str, *, data: str) -> object: ...

    def delete(self, memory_id: str) -> object: ...

    def delete_all(self, *, user_id: str) -> object: ...


def _load_mem0() -> ModuleType:
    try:
        return importlib.import_module("mem0")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Mem0Adapter requires the optional 'mem0' extra: pip install 'lhmsb[mem0]'"
        ) from exc


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _coerce_results(raw: object) -> list[dict[str, object]]:
    """Normalize a Mem0 ``add``/``search`` response to a list of row dicts.

    Mem0 versions return either ``{"results": [...]}`` (current) or a bare list
    (older); both are accepted, non-dict rows are dropped.
    """
    if isinstance(raw, dict):
        inner = raw.get("results")
        if isinstance(inner, list):
            return [row for row in inner if isinstance(row, dict)]
        return []
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def _parse_memory_id(results: list[dict[str, object]]) -> str:
    """First non-empty ``id`` across the returned rows, or ``""`` if none."""
    for row in results:
        raw_id = row.get("id")
        if isinstance(raw_id, str) and raw_id:
            return raw_id
    return ""


def _result_text(row: dict[str, object]) -> str:
    """The memory text of a row under any of Mem0's text keys."""
    for key in ("memory", "text", "data", "content"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _coerce_metadata(raw: object) -> dict[str, object] | None:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    return None


def _row_to_entry(row: dict[str, object], rank: int) -> MemoryEntry:
    """Map one Mem0 search row to a :class:`MemoryEntry`.

    Uses the backend relevance ``score`` when present, else a descending
    rank-derived score so results stay ordered.
    """
    raw_id = row.get("id")
    raw_score = row.get("score")
    score = float(raw_score) if isinstance(raw_score, int | float) else 1.0 / (1.0 + rank)
    return MemoryEntry(
        memory_id=str(raw_id) if raw_id is not None else "",
        content=_result_text(row),
        metadata=_coerce_metadata(row.get("metadata")),
        created_at=_as_str(row.get("created_at"), ""),
        updated_at=_as_str(row.get("updated_at"), ""),
        score=score,
    )


def _build_controlled_config(*, pinned_model: str | None, mem0_config: object) -> dict[str, object]:
    """Build the ``Memory.from_config`` dict for the controlled track.

    Any caller-supplied ``mem0_config`` is the base (forwarded verbatim for a live
    backend); the pinned model is injected at ``config["llm"]["model"]`` so the
    memory system's architecture is isolated from its model choice. No model is
    ever hard-coded — ``pinned_model`` is always caller-supplied.
    """
    config: dict[str, object] = {}
    if isinstance(mem0_config, dict):
        config = {str(key): value for key, value in mem0_config.items()}
    if pinned_model is not None:
        existing_llm = config.get("llm")
        llm: dict[str, object] = dict(existing_llm) if isinstance(existing_llm, dict) else {}
        llm["model"] = pinned_model
        config["llm"] = llm
    return config


class Mem0Adapter(MemorySystemAdapter):
    """Hybrid semantic + entity memory backed by Mem0, with internal-LLM cost
    instrumentation and native/controlled track support."""

    def __init__(self, cost_meter: CostMeter) -> None:
        self.cost_meter = cost_meter
        self._memory: _Mem0Backend | None = None
        self._track = "native"
        self._pinned_model: str | None = None
        self._counter = 0

    @property
    def track(self) -> str:
        """Which comparison track this adapter was initialized for (native/controlled)."""
        return self._track

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        mem0_module = _load_mem0()
        memory_cls: Any = mem0_module.Memory
        self._track = _as_str(config.get("track"), "native")
        pinned = config.get("pinned_model")
        self._pinned_model = pinned if isinstance(pinned, str) else None
        self._counter = 0
        if self._track == "controlled":
            controlled = _build_controlled_config(
                pinned_model=self._pinned_model, mem0_config=config.get("mem0_config")
            )
            self._memory = memory_cls.from_config(controlled)
        else:
            self._memory = memory_cls()

    def reset(self, *, user_id: str) -> None:
        if self._memory is None:
            return
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._memory.delete_all(user_id=user_id)

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
        backend = self._require_backend()
        merged: dict[str, object] = dict(metadata) if metadata else {}
        if session_id is not None:
            merged.setdefault("session_id", session_id)
        started = perf_counter()
        with self.cost_meter.memory_scope():
            res = backend.add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                metadata=merged or None,
            )
        self.cost_meter.record_latency("write", (perf_counter() - started) * 1000.0)
        results = _coerce_results(res)
        self._record_extraction_cost(content, results)
        self.cost_meter.add_storage_bytes(len(content.encode("utf-8")))
        self._counter += 1
        return _parse_memory_id(results) or f"mem0-{user_id}-{self._counter}"

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        backend = self._require_backend()
        self.cost_meter.incr_retrieval()
        started = perf_counter()
        with self.cost_meter.memory_scope():
            self.cost_meter.add_embedding(count_tokens(query, self.cost_meter.model), 1)
            raw = backend.search(query, user_id=user_id, limit=top_k)
        self.cost_meter.record_latency("retrieval", (perf_counter() - started) * 1000.0)
        rows = _coerce_results(raw)
        entries = [_row_to_entry(row, rank) for rank, row in enumerate(rows)]
        return SearchResult(results=entries[:top_k], total_count=len(rows))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if content is None:
            # Mem0's update replaces the stored memory text; it has no
            # metadata-only edit path, so that case degrades gracefully.
            raise UnsupportedOperation("mem0 metadata-only update", condition="content=None")
        backend = self._require_backend()
        started = perf_counter()
        with self.cost_meter.memory_scope():
            backend.update(memory_id, data=content)
            self.cost_meter.add_embedding(count_tokens(content, self.cost_meter.model), 1)
        self.cost_meter.record_latency("update", (perf_counter() - started) * 1000.0)

    def delete_memory(self, memory_id: str) -> None:
        if self._memory is None:
            return
        # Idempotent per the delete contract: deleting an unknown/already-deleted
        # id is a no-op, not an error.
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._memory.delete(memory_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_backend(self) -> _Mem0Backend:
        if self._memory is None:
            raise RuntimeError("Mem0Adapter.initialize(...) must be called before use.")
        return self._memory

    def _record_extraction_cost(self, content: str, results: list[dict[str, object]]) -> None:
        """Attribute Mem0's internal extraction-LLM + embedding cost.

        Mem0 runs an LLM to extract salient facts from the ingested messages; with
        no way to read its native token usage we record a content-derived proxy
        (input ≈ ingested tokens, output ≈ extracted-fact tokens) so the internal
        LLM cost is never silently uncounted (``spec/05-systems.md`` §4).
        """
        model = self.cost_meter.model
        in_tokens = count_tokens(content, model)
        extracted = " ".join(_result_text(row) for row in results)
        out_tokens = count_tokens(extracted, model) if extracted else max(1, in_tokens // 2)
        self.cost_meter.add_memory_internal_tokens(in_tokens, out_tokens)
        self.cost_meter.add_embedding(in_tokens, 1)
