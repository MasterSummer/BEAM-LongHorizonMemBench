from __future__ import annotations

import pytest

from lhmsb.adapters.flat_retrieval import (
    FlatRetrievalAdapter,
    FlatRetrievalError,
)
from lhmsb.qualification.context import PublicHistoryUnit
from lhmsb.qualification.qdrant import InMemoryQdrantTransport


class FakeEmbedding:
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(texts)
        vectors = {
            "alpha": (1.0, 0.0, 0.0),
            "beta": (1.0, 0.0, 0.0),
            "gamma": (0.0, 1.0, 0.0),
            "query": (1.0, 0.0, 0.0),
        }
        return tuple(vectors.get(text, (0.0, 0.0, 1.0)) for text in texts)


def _units() -> tuple[PublicHistoryUnit, ...]:
    return (
        PublicHistoryUnit.create(
            episode_id="episode-1",
            source_session=0,
            source_kind="observation",
            source_ordinal=0,
            content="alpha",
        ),
        PublicHistoryUnit.create(
            episode_id="episode-1",
            source_session=0,
            source_kind="tool_result",
            source_ordinal=0,
            content="beta",
        ),
        PublicHistoryUnit.create(
            episode_id="episode-1",
            source_session=1,
            source_kind="observation",
            source_ordinal=0,
            content="gamma",
        ),
    )


def _adapter(
    *,
    namespace: str = "episode-1-flat",
    candidate_k: int = 20,
) -> tuple[FlatRetrievalAdapter, FakeEmbedding, InMemoryQdrantTransport]:
    embedding = FakeEmbedding()
    transport = InMemoryQdrantTransport()
    adapter = FlatRetrievalAdapter(
        _units(),
        namespace=namespace,
        embedding_runtime=embedding,
        qdrant=transport,
        candidate_k=candidate_k,
    )
    return adapter, embedding, transport


def test_write_ingests_one_unchanged_object_and_is_idempotent() -> None:
    adapter, embedding, _ = _adapter()

    first = adapter.write_session([], session_index=0)
    second = adapter.write_session([], session_index=0)

    assert [event.native_event for event in first.events] == [
        "OBSERVED_ADD",
        "OBSERVED_ADD",
    ]
    assert second.events == ()
    assert first.n_write == 2
    assert second.n_write == 2
    assert first.inventory.n_live == 2
    assert list(embedding.calls) == [("alpha", "beta")]
    item = first.inventory.items[0]
    assert item.content in {"alpha", "beta"}
    metadata = dict(item.metadata)
    assert metadata["session_index"] == 0
    assert metadata["source_kind"] in {"observation", "tool_result"}
    assert metadata["source_ordinal"] == 0
    assert item.content_hash == metadata["content_sha256"]


def test_search_preserves_hashes_shortfall_and_deterministic_tie_order() -> None:
    adapter, embedding, _ = _adapter(candidate_k=3)
    adapter.write_session([], session_index=0)
    result = adapter.search_candidates("query", checkpoint_session=1)

    assert result.query_hash
    assert result.candidate_shortfall is True
    assert len(result.candidates) == 2
    # Equal scores are ordered by deterministic point ID, not insertion order.
    assert [candidate.memory_id for candidate in result.candidates] == sorted(
        candidate.memory_id for candidate in result.candidates
    )
    assert all(candidate.content_hash for candidate in result.candidates)
    assert result.usage_events == ()
    assert embedding.calls[-1] == ("query",)


def test_namespace_isolation_and_conflicting_existing_point_are_terminal() -> None:
    transport = InMemoryQdrantTransport()
    transport.create_collection(collection_name="c", vector_size=2)
    transport.upsert(
        collection_name="c",
        namespace="one",
        points=[
            {"id": "p", "vector": [1.0, 0.0], "payload": {"content": "a"}},
        ],
    )
    assert transport.count(collection_name="c", namespace="two") == 0
    assert transport.count(collection_name="c", namespace="one") == 1
    with pytest.raises(FlatRetrievalError):
        FlatRetrievalAdapter(
            namespace="one",
            embedding_runtime=FakeEmbedding(),
            qdrant=transport,
            collection_name="c",
            embedding_dimension=2,
        )


def test_embedding_dimension_is_strict() -> None:
    class WrongEmbedding(FakeEmbedding):
        def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return tuple((1.0, 2.0) for _ in texts)

    adapter = FlatRetrievalAdapter(
        _units(),
        namespace="strict",
        embedding_runtime=WrongEmbedding(),
        qdrant=InMemoryQdrantTransport(),
    )
    with pytest.raises(FlatRetrievalError) as caught:
        adapter.write_session([], session_index=0)
    assert caught.value.error_class == "vector_dimension_mismatch"


def test_flat_mutations_are_explicitly_unsupported() -> None:
    adapter, _, _ = _adapter()
    with pytest.raises(FlatRetrievalError) as caught:
        adapter.update_memory("memory")
    assert caught.value.error_class == "unsupported_operation"
