"""Synthetic calibration adapters (``spec/05-systems.md`` §2.2).

These two adapters are CALIBRATION-ONLY and are excluded from the real
leaderboard. They exist to prove the benchmark's metrics are sensitive: the task
score under :class:`FakePerfectAdapter` (oracle, upper bound) must clearly exceed
the score under :class:`FakeBadAdapter` (adversary, lower bound), or the metrics
are broken.

Both consume the episode's ground-truth fact store at construction. With an
EMPTY fact store they degrade to a transparent in-memory store so they satisfy
the generic adapter contract (round-trip / update / delete / reset). With a
populated fact store they switch to oracle/adversarial retrieval derived purely
from ground truth and ignore agent-added memories, giving a clean bound rather
than a blend. Neither uses an internal LLM, so their native and controlled
tracks coincide.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lhmsb.adapters.base import MemorySystemAdapter
from lhmsb.types import MemoryEntry, SearchResult

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


def _ts(counter: int) -> str:
    return (_EPOCH + timedelta(seconds=counter)).isoformat()


def _tokens(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


@dataclass(frozen=True)
class GroundTruthFact:
    """A single ground-truth fact in an episode's fact store.

    ``query_keywords`` decide which queries the fact is relevant to (oracle
    retrieval). ``retracted`` marks a fact a later world event invalidated: the
    oracle hides it, the adversary surfaces it.
    """

    fact_id: str
    content: str
    query_keywords: tuple[str, ...] = ()
    retracted: bool = False

    def relevance(self, query_tokens: set[str]) -> int:
        """Keyword hit count for a query (0 means irrelevant)."""
        if self.query_keywords:
            return sum(1 for keyword in self.query_keywords if keyword.lower() in query_tokens)
        return len(query_tokens & _tokens(self.content))


def _fact_entry(fact: GroundTruthFact, *, score: float) -> MemoryEntry:
    return MemoryEntry(
        memory_id=fact.fact_id,
        content=fact.content,
        metadata={"fact_id": fact.fact_id, "retracted": fact.retracted},
        created_at=_ts(0),
        updated_at=_ts(0),
        score=score,
    )


class _FakeStoreAdapter(MemorySystemAdapter):
    """Transparent in-memory store shared by the calibration fakes.

    Subclasses override :meth:`_oracle_search` to return ground-truth-derived
    results. When no fact store is configured, search falls back to the
    transparent store so the generic contract suite still holds.
    """

    def __init__(self, *, facts: Sequence[GroundTruthFact] = ()) -> None:
        self._facts: tuple[GroundTruthFact, ...] = tuple(facts)
        self._store: dict[str, dict[str, MemoryEntry]] = {}
        self._counter = 0

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        self._store.setdefault(user_id, {})

    def reset(self, *, user_id: str) -> None:
        self._store[user_id] = {}

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        memory_id = f"fake-{self._counter}"
        timestamp = _ts(self._counter)
        self._store.setdefault(user_id, {})[memory_id] = MemoryEntry(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            created_at=timestamp,
            updated_at=timestamp,
            score=None,
        )
        return memory_id

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        for entries in self._store.values():
            existing = entries.get(memory_id)
            if existing is not None:
                self._counter += 1
                entries[memory_id] = MemoryEntry(
                    memory_id=existing.memory_id,
                    content=existing.content if content is None else content,
                    metadata=existing.metadata if metadata is None else metadata,
                    created_at=existing.created_at,
                    updated_at=_ts(self._counter),
                    score=existing.score,
                )
                return

    def delete_memory(self, memory_id: str) -> None:
        for entries in self._store.values():
            entries.pop(memory_id, None)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        if self._facts:
            return self._oracle_search(query, top_k)
        return self._store_search(query, user_id, top_k)

    def _store_search(self, query: str, user_id: str, top_k: int) -> SearchResult:
        query_tokens = _tokens(query)
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._store.get(user_id, {}).values():
            overlap = len(query_tokens & _tokens(entry.content))
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda pair: (pair[0], pair[1].memory_id), reverse=True)
        ranked = [
            MemoryEntry(
                memory_id=entry.memory_id,
                content=entry.content,
                metadata=entry.metadata,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                score=float(overlap),
            )
            for overlap, entry in scored
        ]
        return SearchResult(results=ranked[:top_k], total_count=len(ranked))

    @abstractmethod
    def _oracle_search(self, query: str, top_k: int) -> SearchResult:
        """Return ground-truth-derived results for a query."""
        ...


class FakePerfectAdapter(_FakeStoreAdapter):
    """Oracle memory (calibration-only): returns the relevant CURRENT
    (non-retracted) facts for any query — the upper bound for the metrics."""

    def _oracle_search(self, query: str, top_k: int) -> SearchResult:
        query_tokens = _tokens(query)
        relevant = [
            (fact.relevance(query_tokens), fact)
            for fact in self._facts
            if not fact.retracted and fact.relevance(query_tokens) > 0
        ]
        relevant.sort(key=lambda pair: (pair[0], pair[1].fact_id), reverse=True)
        results = [_fact_entry(fact, score=1.0) for _, fact in relevant[:top_k]]
        return SearchResult(results=results, total_count=len(relevant))


class FakeBadAdapter(_FakeStoreAdapter):
    """Adversarial memory (calibration-only): returns plausible-but-wrong
    (retracted/superseded) facts — the lower bound for the metrics."""

    def _oracle_search(self, query: str, top_k: int) -> SearchResult:
        query_tokens = _tokens(query)
        retracted = [fact for fact in self._facts if fact.retracted]
        relevant = [(fact.relevance(query_tokens), fact) for fact in retracted]
        relevant = [(hits, fact) for hits, fact in relevant if hits > 0]
        relevant.sort(key=lambda pair: (pair[0], pair[1].fact_id), reverse=True)
        if relevant:
            chosen = [fact for _, fact in relevant[:top_k]]
        else:
            chosen = sorted(retracted, key=lambda fact: fact.fact_id)[:top_k]
        results = [_fact_entry(fact, score=0.95) for fact in chosen]
        return SearchResult(results=results, total_count=len(chosen))


class WrongMemoryAdapter(_FakeStoreAdapter):
    """Controlled ``wrong_mem`` condition for the three-system ablation.

    It preserves the adapter lifecycle and write path, but returns distractor
    content instead of the relevant stored fact. This gives the benchmark a
    deterministic lower-bound replay even when an episode has no retracted fact.
    """

    def _oracle_search(self, query: str, top_k: int) -> SearchResult:
        query_tokens = _tokens(query)
        candidates = [
            (fact.relevance(query_tokens), fact)
            for fact in self._facts
            if fact.relevance(query_tokens) > 0
        ]
        candidates.sort(key=lambda pair: (pair[0], pair[1].fact_id), reverse=True)
        chosen = [fact for _, fact in candidates[:top_k]]
        results = [
            MemoryEntry(
                memory_id=f"wrong-{fact.fact_id}",
                content="Distractor memory: the requested fact is unavailable.",
                metadata={"fact_id": f"wrong-{fact.fact_id}", "wrong_memory": True},
                created_at=_ts(0),
                updated_at=_ts(0),
                score=0.95,
            )
            for fact in chosen
        ]
        return SearchResult(results=results, total_count=len(results))
