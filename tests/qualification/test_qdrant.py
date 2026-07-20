from __future__ import annotations

import json
import math
import uuid

import httpx
import pytest

from lhmsb.qualification.qdrant import (
    InMemoryQdrantTransport,
    QdrantError,
    QdrantHit,
    QdrantHttpTransport,
    QdrantPoint,
    validate_search_order,
    validate_vector,
)


def test_in_memory_qdrant_validates_dimension_and_namespace() -> None:
    transport = InMemoryQdrantTransport()
    transport.create_collection(collection_name="c", vector_size=2)
    transport.upsert(
        collection_name="c",
        namespace="a",
        points=[QdrantPoint("p", (1.0, 0.0), {"content": "a"})],
    )
    assert transport.count(collection_name="c", namespace="a") == 1
    assert transport.count(collection_name="c", namespace="b") == 0
    with pytest.raises(QdrantError) as caught:
        transport.upsert(
            collection_name="c",
            namespace="a",
            points=[QdrantPoint("bad", (1.0,), {})],
        )
    assert caught.value.error_class == "vector_dimension_mismatch"


def test_search_order_rejects_duplicates_nonfinite_and_unsorted_hits() -> None:
    with pytest.raises(QdrantError) as duplicate:
        validate_search_order(
            (
                QdrantHit("p", 1.0, {}),
                QdrantHit("p", 0.9, {}),
            )
        )
    assert duplicate.value.error_class == "duplicate_point_id"
    with pytest.raises(QdrantError) as unsorted:
        validate_search_order(
            (
                QdrantHit("b", 0.9, {}),
                QdrantHit("a", 1.0, {}),
            )
        )
    assert unsorted.value.error_class == "invalid_point_order"
    with pytest.raises(QdrantError):
        validate_search_order((QdrantHit("p", math.inf, {}),))


def test_http_boundary_sends_namespace_filter_and_validates_responses() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/points/count"):
            return httpx.Response(200, json={"result": {"count": 1}})
        if request.url.path.endswith("/points/search"):
            return httpx.Response(
                200,
                json={"result": [{"id": "p", "score": 0.5, "payload": {"content": "a"}}]},
            )
        return httpx.Response(200, json={"result": {"status": "ok"}})

    client = QdrantHttpTransport(
        "http://qdrant",
        collection_name="c",
        vector_size=2,
        transport=httpx.MockTransport(handler),
    )
    client.create_collection()
    client.upsert(
        namespace="ns",
        points=[QdrantPoint("p", (1.0, 0.0), {"content": "a"})],
    )
    assert client.count(namespace="ns") == 1
    assert client.search(namespace="ns", vector=(1.0, 0.0), limit=1)[0].point_id == "p"
    assert requests[1].url.path.endswith("/points")
    assert requests[2].url.path.endswith("/points/count")
    count_body = requests[2].content.decode("utf-8")
    assert "ns" in count_body
    with pytest.raises(QdrantError) as caught:
        client.search(namespace="ns", vector=(1.0,), limit=1)
    assert caught.value.error_class == "vector_dimension_mismatch"
    client.close()


def test_http_boundary_maps_arbitrary_benchmark_ids_to_qdrant_uuid_and_back() -> None:
    requests: list[httpx.Request] = []
    wire_id: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal wire_id
        requests.append(request)
        if request.url.path.endswith("/points"):
            body = json.loads(request.content.decode("utf-8"))
            wire_id = str(body["points"][0]["id"])
            uuid.UUID(wire_id)
            assert body["points"][0]["payload"]["_lhmsb_point_id"] == "a" * 64
            return httpx.Response(200, json={"result": {"status": "ok"}})
        if request.url.path.endswith("/points/search"):
            assert wire_id is not None
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": wire_id,
                            "score": 0.5,
                            "payload": {
                                "_lhmsb_point_id": "a" * 64,
                                "content": "a",
                            },
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"result": {"status": "ok"}})

    client = QdrantHttpTransport(
        "http://qdrant",
        collection_name="c",
        vector_size=2,
        transport=httpx.MockTransport(handler),
    )
    client.create_collection()
    client.upsert(
        namespace="ns",
        points=[QdrantPoint("a" * 64, (1.0, 0.0), {"content": "a"})],
    )
    hit = client.search(namespace="ns", vector=(1.0, 0.0), limit=1)[0]
    assert hit.point_id == "a" * 64
    assert "_lhmsb_point_id" not in hit.payload
    client.close()


def test_http_boundary_accepts_qdrant_ties_and_orders_restored_ids() -> None:
    wire_ids: list[str] = []
    original_ids = ["a" * 64, "b" * 64]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/points"):
            body = json.loads(request.content.decode("utf-8"))
            wire_ids.extend(str(point["id"]) for point in body["points"])
            return httpx.Response(200, json={"result": {"status": "ok"}})
        if request.url.path.endswith("/points/search"):
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": wire_ids[1],
                            "score": 0.5,
                            "payload": {"_lhmsb_point_id": original_ids[1]},
                        },
                        {
                            "id": wire_ids[0],
                            "score": 0.5,
                            "payload": {"_lhmsb_point_id": original_ids[0]},
                        },
                    ]
                },
            )
        return httpx.Response(200, json={"result": {"status": "ok"}})

    client = QdrantHttpTransport(
        "http://qdrant",
        collection_name="c",
        vector_size=2,
        transport=httpx.MockTransport(handler),
    )
    client.create_collection()
    client.upsert(
        namespace="ns",
        points=[
            QdrantPoint(original_ids[0], (1.0, 0.0), {}),
            QdrantPoint(original_ids[1], (0.0, 1.0), {}),
        ],
    )
    hits = client.search(namespace="ns", vector=(1.0, 0.0), limit=2)
    assert [hit.point_id for hit in hits] == original_ids
    client.close()


def test_validate_vector_rejects_nonfinite_values() -> None:
    with pytest.raises(QdrantError) as caught:
        validate_vector((1.0, math.nan), dimension=2)
    assert caught.value.error_class == "invalid_vector"
