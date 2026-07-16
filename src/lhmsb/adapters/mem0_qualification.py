"""Trace-complete adapter for the pinned Mem0 OSS 2.0.12 lifecycle."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol
from urllib.parse import urlparse

from lhmsb.qualification.schema import Mem0Profile, PolicyProfile


class Mem0QualificationError(RuntimeError):
    """Typed Mem0 lifecycle failure."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class Mem0V2Backend(Protocol):
    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        run_id: str,
        metadata: dict[str, object] | None,
        infer: bool,
    ) -> object: ...

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        top_k: int,
        threshold: float,
        rerank: bool,
    ) -> object: ...

    def get_all(
        self,
        *,
        filters: dict[str, object],
        top_k: int,
    ) -> object: ...

    def history(self, memory_id: str) -> object: ...


@dataclass(frozen=True)
class NativeMemoryEvent:
    operation_id: str
    session_index: int
    native_event: str
    memory_id: str
    memory_text: str
    old_content_hash: str | None
    new_content_hash: str | None
    source: str
    latency_seconds: float


@dataclass(frozen=True)
class InventoryItem:
    memory_id: str
    content: str
    content_hash: str
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str
    history_length: int


@dataclass(frozen=True)
class InventorySnapshot:
    checkpoint_session: int
    n_write: int
    n_live: int
    items: tuple[InventoryItem, ...]
    store_hash: str
    backend_count: int | None


@dataclass(frozen=True)
class SearchCandidate:
    memory_id: str
    content: str
    content_hash: str
    native_rank: int
    score: float | None
    score_details: tuple[tuple[str, float], ...]
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CandidateSearch:
    checkpoint_session: int
    query: str
    query_hash: str
    candidates: tuple[SearchCandidate, ...]
    candidate_shortfall: bool
    latency_seconds: float


@dataclass(frozen=True)
class WriteSessionResult:
    session_index: int
    events: tuple[NativeMemoryEvent, ...]
    inventory: InventorySnapshot
    n_write: int
    latency_seconds: float


class Mem0QualificationAdapter:
    """Synchronous Mem0 adapter that preserves native objects and events."""

    def __init__(
        self,
        backend: Mem0V2Backend,
        *,
        user_id: str,
        run_id: str,
        candidate_k: int = 20,
        inventory_limit: int = 10000,
        collection_count: Callable[[], int] | None = None,
    ) -> None:
        if not user_id or not run_id:
            raise ValueError("user_id and run_id must be non-empty")
        if candidate_k < 1 or inventory_limit < 1:
            raise ValueError("candidate_k and inventory_limit must be positive")
        self.backend = backend
        self.user_id = user_id
        self.run_id = run_id
        self.candidate_k = candidate_k
        self.inventory_limit = inventory_limit
        self.collection_count = collection_count
        self._n_write = 0

    @property
    def filters(self) -> dict[str, object]:
        return {"user_id": self.user_id, "run_id": self.run_id}

    def restore_write_count(self, n_write: int) -> None:
        """Restore the cumulative native mutation count when resuming a task."""
        if n_write < 0:
            raise ValueError("n_write must be non-negative")
        self._n_write = n_write

    @classmethod
    def create_live(
        cls,
        config: dict[str, object],
        *,
        user_id: str,
        run_id: str,
        candidate_k: int = 20,
        inventory_limit: int = 10000,
        collection_count: Callable[[], int] | None = None,
    ) -> Mem0QualificationAdapter:
        """Lazily import Mem0 after disabling product telemetry."""
        _disable_mem0_telemetry()
        module = _load_mem0()
        memory_class = module.Memory
        backend = memory_class.from_config(config)
        return cls(
            backend,
            user_id=user_id,
            run_id=run_id,
            candidate_k=candidate_k,
            inventory_limit=inventory_limit,
            collection_count=collection_count,
        )

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        """Run one native extraction write and capture its resulting inventory."""
        if session_index < 0:
            raise ValueError("session_index must be non-negative")
        before = self.snapshot_inventory(
            checkpoint_session=session_index,
            include_history=False,
        )
        merged = dict(metadata or {})
        merged["session_index"] = session_index
        started = time.perf_counter()
        try:
            raw = self.backend.add(
                messages,
                user_id=self.user_id,
                run_id=self.run_id,
                metadata=merged,
                infer=True,
            )
        except Exception as exc:
            raise Mem0QualificationError("mem0_write_failure", str(exc)) from exc
        latency = max(0.0, time.perf_counter() - started)
        events = list(_native_events(raw, session_index=session_index, latency=latency))
        counted_ids = {
            event.memory_id
            for event in events
            if event.native_event in {"ADD", "UPDATE", "DELETE"}
        }
        self._n_write += sum(
            event.native_event in {"ADD", "UPDATE", "DELETE"} for event in events
        )
        after = self.snapshot_inventory(checkpoint_session=session_index)
        before_ids = {item.memory_id for item in before.items}
        for index, item in enumerate(after.items):
            if item.memory_id not in before_ids and item.memory_id not in counted_ids:
                events.append(
                    NativeMemoryEvent(
                        operation_id=f"session-{session_index:03d}-delta-{index:03d}",
                        session_index=session_index,
                        native_event="OBSERVED_ADD",
                        memory_id=item.memory_id,
                        memory_text=item.content,
                        old_content_hash=None,
                        new_content_hash=item.content_hash,
                        source="inventory_delta",
                        latency_seconds=0.0,
                    )
                )
                self._n_write += 1
        if after.n_write != self._n_write:
            after = InventorySnapshot(
                checkpoint_session=after.checkpoint_session,
                n_write=self._n_write,
                n_live=after.n_live,
                items=after.items,
                store_hash=after.store_hash,
                backend_count=after.backend_count,
            )
        return WriteSessionResult(
            session_index=session_index,
            events=tuple(events),
            inventory=after,
            n_write=self._n_write,
            latency_seconds=latency,
        )

    def snapshot_inventory(
        self,
        *,
        checkpoint_session: int,
        include_history: bool = True,
    ) -> InventorySnapshot:
        try:
            raw = self.backend.get_all(
                filters=self.filters,
                top_k=self.inventory_limit,
            )
        except Exception as exc:
            raise Mem0QualificationError("inventory_failure", str(exc)) from exc
        rows = _result_rows(raw)
        if len(rows) >= self.inventory_limit:
            raise Mem0QualificationError(
                "inventory_failure",
                f"inventory reached or exceeded limit {self.inventory_limit}",
            )
        items: list[InventoryItem] = []
        for row in rows:
            memory_id = _text(row.get("id"))
            content = _memory_text(row)
            history_length = (
                len(self._history(memory_id)) if include_history and memory_id else 0
            )
            items.append(
                InventoryItem(
                    memory_id=memory_id,
                    content=content,
                    content_hash=_text_hash(content),
                    metadata=_metadata_pairs(row.get("metadata")),
                    created_at=_text(row.get("created_at")),
                    updated_at=_text(row.get("updated_at")),
                    history_length=history_length,
                )
            )
        items.sort(key=lambda item: item.memory_id)
        backend_count = self.collection_count() if self.collection_count is not None else None
        if backend_count is not None and backend_count != len(items):
            raise Mem0QualificationError(
                "inventory_failure",
                f"Mem0 inventory count {len(items)} != Qdrant count {backend_count}",
            )
        payload = [
            {
                "memory_id": item.memory_id,
                "content_hash": item.content_hash,
                "metadata": item.metadata,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "history_length": item.history_length,
            }
            for item in items
        ]
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=self._n_write,
            n_live=len(items),
            items=tuple(items),
            store_hash=_canonical_hash(payload),
            backend_count=backend_count,
        )

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch:
        started = time.perf_counter()
        try:
            raw = self.backend.search(
                query,
                filters=self.filters,
                top_k=self.candidate_k,
                threshold=0.0,
                rerank=False,
            )
        except Exception as exc:
            raise Mem0QualificationError("mem0_search_failure", str(exc)) from exc
        rows = _result_rows(raw)
        candidates: list[SearchCandidate] = []
        for rank, row in enumerate(rows, start=1):
            content = _memory_text(row)
            candidates.append(
                SearchCandidate(
                    memory_id=_text(row.get("id")),
                    content=content,
                    content_hash=_text_hash(content),
                    native_rank=rank,
                    score=_optional_float(row.get("score")),
                    score_details=_score_pairs(row.get("score_details")),
                    metadata=_metadata_pairs(row.get("metadata")),
                    created_at=_text(row.get("created_at")),
                    updated_at=_text(row.get("updated_at")),
                )
            )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=_text_hash(query),
            candidates=tuple(candidates),
            candidate_shortfall=len(candidates) < self.candidate_k,
            latency_seconds=max(0.0, time.perf_counter() - started),
        )

    def history_delta(
        self,
        memory_id: str,
        *,
        previous_length: int,
    ) -> tuple[dict[str, object], ...]:
        if previous_length < 0:
            raise ValueError("previous_length must be non-negative")
        return tuple(self._history(memory_id)[previous_length:])

    def _history(self, memory_id: str) -> list[dict[str, object]]:
        try:
            raw = self.backend.history(memory_id)
        except Exception as exc:
            raise Mem0QualificationError("inventory_failure", str(exc)) from exc
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise Mem0QualificationError("inventory_failure", "history must be an array")
        output: list[dict[str, object]] = []
        for row in raw:
            if isinstance(row, Mapping):
                output.append({str(key): value for key, value in row.items()})
        return output


def build_mem0_live_config(
    profile: Mem0Profile,
    *,
    policy: PolicyProfile,
    internal_llm_api_key: str,
    native_openai_api_key: str,
    qdrant_url: str,
    collection_name: str,
    history_db_path: Path,
    embedding_base_url: str,
    embedding_dimension: int,
    response_callback: Callable[[object, dict[str, object], dict[str, object]], None]
    | None = None,
) -> dict[str, object]:
    """Build the explicit Mem0 2.0.12 config for one isolated task."""
    qdrant = urlparse(qdrant_url)
    if not qdrant.hostname or qdrant.port is None:
        raise ValueError("qdrant_url must include a host and port")
    if profile.track == "controlled":
        llm_config: dict[str, object] = {
            "model": policy.model_id,
            "api_key": internal_llm_api_key,
            f"{policy.provider}_base_url": policy.endpoint,
        }
        if response_callback is not None and policy.provider == "openai":
            llm_config["response_callback"] = response_callback
        llm = {"provider": policy.provider, "config": llm_config}
        embedder = {
            "provider": "huggingface",
            "config": {
                "model": profile.embedding_model,
                "huggingface_base_url": f"{embedding_base_url.rstrip('/')}/v1",
                "embedding_dims": embedding_dimension,
            },
        }
        vector_dimension = embedding_dimension
    else:
        llm = {
            "provider": "openai",
            "config": {
                "model": profile.internal_llm_model,
                "api_key": native_openai_api_key,
                "is_reasoning_model": True,
            },
        }
        vector_dimension = 1536
        embedder = {
            "provider": "openai",
            "config": {
                "model": profile.embedding_model,
                "api_key": native_openai_api_key,
                "embedding_dims": vector_dimension,
            },
        }
    return {
        "llm": llm,
        "embedder": embedder,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": vector_dimension,
                "host": qdrant.hostname,
                "port": qdrant.port,
                "path": None,
                "https": qdrant.scheme == "https",
                "on_disk": True,
            },
        },
        "history_db_path": str(history_db_path),
        "version": "v1.1",
    }


def _native_events(
    raw: object,
    *,
    session_index: int,
    latency: float,
) -> tuple[NativeMemoryEvent, ...]:
    events: list[NativeMemoryEvent] = []
    for index, row in enumerate(_result_rows(raw)):
        native_event = _text(row.get("event"), default="NONE").upper()
        memory = _memory_text(row)
        old_memory = _text(row.get("old_memory"))
        events.append(
            NativeMemoryEvent(
                operation_id=f"session-{session_index:03d}-native-{index:03d}",
                session_index=session_index,
                native_event=native_event,
                memory_id=_text(row.get("id")),
                memory_text=memory,
                old_content_hash=_text_hash(old_memory) if old_memory else None,
                new_content_hash=(
                    _text_hash(memory)
                    if memory and native_event in {"ADD", "UPDATE"}
                    else None
                ),
                source="native_response",
                latency_seconds=latency,
            )
        )
    return tuple(events)


def _result_rows(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, Mapping):
        raw = raw.get("results")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [
        {str(key): value for key, value in row.items()}
        for row in raw
        if isinstance(row, Mapping)
    ]


def _memory_text(row: Mapping[str, object]) -> str:
    for key in ("memory", "text", "data", "content", "new_memory"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _metadata_pairs(value: object) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(key), child) for key, child in value.items()))


def _score_pairs(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        return ()
    output: list[tuple[str, float]] = []
    for key, child in value.items():
        score = _optional_float(child)
        if score is not None:
            output.append((str(key), score))
    return tuple(sorted(output))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _text(value: object, *, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _disable_mem0_telemetry() -> None:
    os.environ["MEM0_TELEMETRY"] = "False"
    for module_name in ("mem0.memory.telemetry", "mem0.memory.main"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "MEM0_TELEMETRY"):
            module.__dict__["MEM0_TELEMETRY"] = False


def _load_mem0() -> ModuleType:
    try:
        return importlib.import_module("mem0")
    except ImportError as exc:
        raise ImportError(
            "Mem0QualificationAdapter requires the qualification extra: "
            "pip install 'lhmsb[qualification]'"
        ) from exc


__all__ = [
    "CandidateSearch",
    "InventoryItem",
    "InventorySnapshot",
    "Mem0QualificationAdapter",
    "Mem0QualificationError",
    "Mem0V2Backend",
    "NativeMemoryEvent",
    "SearchCandidate",
    "WriteSessionResult",
    "build_mem0_live_config",
]
