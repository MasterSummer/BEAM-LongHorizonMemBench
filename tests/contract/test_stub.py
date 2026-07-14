"""Run the reusable contract suite against in-memory reference adapters.

- ``StubAdapter``: a minimal conforming storing backend. Subclassed into
  ``TestStubContract`` so every contract property runs as its own test case.
- ``DegradedStubAdapter``: declares update/delete unsupported and raises
  ``UnsupportedOperation`` for them — proves the graceful-degradation path has
  teeth (the conforming-but-limited case).
- ``BrokenAdapter``: deliberately violates the contract (ignores ``top_k`` and
  returns deleted items) — proves the suite catches non-conformance.
"""

from __future__ import annotations

import pytest

from contract.adapter_contract import (
    AdapterContractTests,
    check_unsupported_op_degrades,
    run_contract_suite,
)
from lhmsb.adapters import (
    Capabilities,
    MemorySystemAdapter,
    UnsupportedOperation,
)
from lhmsb.types import MemoryEntry, SearchResult


def _key(user_id: str) -> str:
    return user_id


class StubAdapter(MemorySystemAdapter):
    """Minimal conforming in-memory adapter (token-overlap relevance)."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, MemoryEntry]] = {}
        self._counter = 0

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        self._store.setdefault(_key(user_id), {})

    def reset(self, *, user_id: str) -> None:
        self._store[_key(user_id)] = {}

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        memory_id = f"stub-{self._counter}"
        ts = f"2025-01-01T00:00:{self._counter:02d}Z"
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            created_at=ts,
            updated_at=ts,
            score=None,
        )
        self._store.setdefault(_key(user_id), {})[memory_id] = entry
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
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._store.get(_key(user_id), {}).values():
            content_terms = entry.content.lower().split()
            overlap = sum(1 for t in terms if t in content_terms)
            if overlap > 0:
                scored.append((float(overlap), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ranked = [
            MemoryEntry(
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
        for entries in self._store.values():
            if memory_id in entries:
                old = entries[memory_id]
                entries[memory_id] = MemoryEntry(
                    memory_id=old.memory_id,
                    content=content if content is not None else old.content,
                    metadata=metadata if metadata is not None else old.metadata,
                    created_at=old.created_at,
                    updated_at="2025-01-02T00:00:00Z",
                    score=old.score,
                )
                return

    def delete_memory(self, memory_id: str) -> None:
        for entries in self._store.values():
            entries.pop(memory_id, None)


class DegradedStubAdapter(StubAdapter):
    """Conforming storing backend that does NOT support update/delete."""

    def get_capabilities(self) -> Capabilities:
        return Capabilities(supports_update=False, supports_delete=False)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        raise UnsupportedOperation("update_memory")

    def delete_memory(self, memory_id: str) -> None:
        raise UnsupportedOperation("delete_memory")


class BrokenAdapter(StubAdapter):
    """Violates the contract: ignores ``top_k`` and never removes on delete."""

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        result = super().search(query, user_id=user_id, session_id=session_id, top_k=10_000)
        return SearchResult(results=result.results, total_count=result.total_count)

    def delete_memory(self, memory_id: str) -> None:
        return None


class TestStubContract(AdapterContractTests):
    """The conforming StubAdapter must pass every contract check."""

    @staticmethod
    def adapter_factory() -> MemorySystemAdapter:
        return StubAdapter()


def test_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MemorySystemAdapter()


def test_full_suite_passes_for_stub() -> None:
    run_contract_suite(StubAdapter)


def test_degraded_adapter_raises_unsupported_gracefully() -> None:
    check_unsupported_op_degrades(DegradedStubAdapter)
    adapter = DegradedStubAdapter()
    adapter.initialize(user_id="u")
    with pytest.raises(UnsupportedOperation):
        adapter.update_memory("x", content="y")
    with pytest.raises(UnsupportedOperation):
        adapter.delete_memory("x")


def test_broken_adapter_fails_contract() -> None:
    with pytest.raises(AssertionError):
        run_contract_suite(BrokenAdapter)
