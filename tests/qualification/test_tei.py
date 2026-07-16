from __future__ import annotations

import json

import httpx
import pytest

from lhmsb.qualification.tei import (
    EmbeddingClient,
    RerankCandidate,
    RerankerClient,
    TeiServiceError,
)


def test_embedding_client_requires_exact_dimension_and_hashes_request() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "BAAI/bge-m3",
                "data": [
                    {"index": 0, "embedding": [0.0] * 1024},
                    {"index": 1, "embedding": [1.0] * 1024},
                ],
            },
        )

    result = EmbeddingClient(
        "http://embedding.local",
        model="BAAI/bge-m3",
        revision="embedding-revision",
        expected_dimension=1024,
        transport=httpx.MockTransport(handler),
    ).embed(("first", "second"))
    assert result.dimension == 1024
    assert result.input_count == 2
    assert len(result.vectors) == 2
    assert seen == [{"input": ["first", "second"], "model": "BAAI/bge-m3"}]
    assert len(result.request_hash) == 64
    assert len(result.response_hash) == 64


def test_embedding_dimension_mismatch_is_terminal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.0] * 3}]},
        )
    )
    with pytest.raises(TeiServiceError) as caught:
        EmbeddingClient(
            "http://embedding.local",
            model="BAAI/bge-m3",
            revision="revision",
            expected_dimension=1024,
            transport=transport,
        ).embed(("text",))
    assert caught.value.error_class == "embedding_failure"
    assert "1024" in str(caught.value)


def test_reranker_preserves_native_rank_as_tie_breaker() -> None:
    candidates = (
        RerankCandidate("m1", "first", 1),
        RerankCandidate("m2", "second", 2),
        RerankCandidate("m3", "third", 3),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "current branch"
        assert body["texts"] == ["first", "second", "third"]
        return httpx.Response(
            200,
            json=[
                {"index": 0, "score": 0.5},
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.9},
            ],
        )

    result = RerankerClient(
        "http://reranker.local",
        model="BAAI/bge-reranker-v2-m3",
        revision="reranker-revision",
        transport=httpx.MockTransport(handler),
    ).rerank("current branch", candidates, top_k=2)
    assert result.ordered_memory_ids == ("m2", "m3")
    assert result.scores == (0.9, 0.9)
    assert result.input_count == 3


def test_reranker_rejects_unknown_indices() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[{"index": 4, "score": 1.0}])
    )
    with pytest.raises(TeiServiceError) as caught:
        RerankerClient(
            "http://reranker.local",
            model="reranker",
            revision="revision",
            transport=transport,
        ).rerank("query", (RerankCandidate("m1", "text", 1),))
    assert caught.value.error_class == "reranker_failure"


def test_health_checks_are_typed() -> None:
    healthy = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    client = EmbeddingClient(
        "http://embedding.local",
        model="model",
        revision="revision",
        expected_dimension=1,
        transport=healthy,
    )
    assert client.health().ok

    unhealthy = httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
    failing = RerankerClient(
        "http://reranker.local",
        model="model",
        revision="revision",
        transport=unhealthy,
    )
    assert not failing.health().ok
    assert failing.health().status_code == 503
