"""Deterministic in-process adapter used by the offline vertical runner."""

from __future__ import annotations

from dataclasses import dataclass

from lhmsb.adapters.base import MemorySystemAdapter
from lhmsb.types import MemoryEntry, SearchResult


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token}


@dataclass(frozen=True)
class StubTraceEvent:
    """One observable fake-native lifecycle operation."""

    operation: str
    session_id: str | None = None
    memory_ids: tuple[str, ...] = ()
    state_ids: tuple[str, ...] = ()
    query: str = ""


class VerticalStubAdapter(MemorySystemAdapter):
    """A tiny lexical adapter with explicit session-boundary instrumentation."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._users: dict[str, set[str]] = {}
        self._counter = 0
        self._working_context: list[str] = []
        self._current_session: str | None = None
        self._trace: list[StubTraceEvent] = []

    @property
    def trace(self) -> tuple[StubTraceEvent, ...]:
        return tuple(self._trace)

    @property
    def working_context(self) -> tuple[str, ...]:
        return tuple(self._working_context)

    @property
    def stored_memory_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        del config
        self._users.setdefault(user_id, set())
        self._current_session = session_id
        self._trace.append(StubTraceEvent("initialize", session_id=session_id))

    def reset(self, *, user_id: str) -> None:
        for memory_id in self._users.get(user_id, set()):
            self._entries.pop(memory_id, None)
        self._users[user_id] = set()
        self._working_context.clear()
        self._trace.append(StubTraceEvent("reset", session_id=self._current_session))

    def begin_session(self, session_id: str) -> None:
        """Clear short-lived context while retaining persistent memories."""
        self._working_context.clear()
        self._current_session = session_id
        self._trace.append(StubTraceEvent("clear_working_context", session_id=session_id))

    def record_model_visible(
        self,
        state_ids: tuple[str, ...],
        *,
        session_id: str | None = None,
        query: str = "",
    ) -> None:
        """Record the post-retrieval context exposed to the deterministic policy."""
        self._trace.append(
            StubTraceEvent(
                "model_visible",
                session_id=session_id or self._current_session,
                state_ids=state_ids,
                query=query,
            )
        )

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        memory_id = f"vertical-{self._counter:04d}"
        timestamp = f"2025-01-01T00:00:{self._counter:02d}Z"
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            metadata=dict(metadata or {}),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._entries[memory_id] = entry
        self._users.setdefault(user_id, set()).add(memory_id)
        self._working_context.append(memory_id)
        state_ids = self._state_ids(entry)
        self._trace.append(
            StubTraceEvent(
                "write",
                session_id=session_id or self._current_session,
                memory_ids=(memory_id,),
                state_ids=state_ids,
            )
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
        del filters
        query_tokens = _tokens(query)
        scored: list[tuple[int, str, MemoryEntry]] = []
        for memory_id in sorted(self._users.get(user_id, set())):
            entry = self._entries.get(memory_id)
            if entry is None:
                continue
            overlap = len(query_tokens & _tokens(entry.content))
            if overlap:
                scored.append((overlap, memory_id, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [
            MemoryEntry(
                memory_id=entry.memory_id,
                content=entry.content,
                metadata=entry.metadata,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                score=float(overlap),
            )
            for overlap, _memory_id, entry in scored[:top_k]
        ]
        self._trace.append(
            StubTraceEvent(
                "search",
                session_id=session_id or self._current_session,
                memory_ids=tuple(entry.memory_id for entry in results),
                state_ids=tuple(
                    sorted({state_id for entry in results for state_id in self._state_ids(entry)})
                ),
                query=query,
            )
        )
        return SearchResult(results=results, total_count=len(scored))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        entry = self._entries.get(memory_id)
        if entry is None:
            return
        self._counter += 1
        self._entries[memory_id] = MemoryEntry(
            memory_id=memory_id,
            content=entry.content if content is None else content,
            metadata=entry.metadata if metadata is None else dict(metadata),
            created_at=entry.created_at,
            updated_at=f"2025-01-01T00:00:{self._counter:02d}Z",
            score=entry.score,
        )
        self._trace.append(
            StubTraceEvent("update", session_id=self._current_session, memory_ids=(memory_id,))
        )

    def delete_memory(self, memory_id: str) -> None:
        self._entries.pop(memory_id, None)
        for ids in self._users.values():
            ids.discard(memory_id)
        self._working_context = [item for item in self._working_context if item != memory_id]
        self._trace.append(
            StubTraceEvent("delete", session_id=self._current_session, memory_ids=(memory_id,))
        )

    @staticmethod
    def _state_ids(entry: MemoryEntry) -> tuple[str, ...]:
        if entry.metadata is None:
            return ()
        value = entry.metadata.get("state_ids")
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        if isinstance(value, str):
            return (value,)
        return ()


__all__ = ["StubTraceEvent", "VerticalStubAdapter"]
