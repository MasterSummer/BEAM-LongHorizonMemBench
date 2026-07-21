from __future__ import annotations

import pytest

from lhmsb.qualification.neo4j import (
    FakeNeo4jTransport,
    Neo4jBoltTransport,
    Neo4jError,
    Neo4jGraphSnapshot,
    Neo4jNode,
    validate_empty_namespace,
)


class _TemporalValue:
    def iso_format(self) -> str:
        return "2026-07-17T00:00:00Z"


class _FakeSession:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def run(self, query: str, **params: object) -> list[dict[str, object]]:
        self.driver.calls.append((query, params))
        return self.driver.responses.pop(0)


class _FakeDriver:
    def __init__(self, responses: list[list[dict[str, object]]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def session(self, *, database: str) -> _FakeSession:
        assert database == "neo4j"
        return _FakeSession(self)

    def close(self) -> None:
        self.closed = True


def test_fake_graph_requires_empty_namespace_and_keeps_edges_separate() -> None:
    graph = FakeNeo4jTransport()
    validate_empty_namespace(graph, "episode")
    graph.add_node(
        namespace="episode", node_id="a", labels=("Content",), properties={"memory": "a"}
    )
    graph.add_node(
        namespace="episode", node_id="topic", labels=("Topic",), properties={"kind": "topic"}
    )
    graph.add_edge(
        namespace="episode",
        edge_id="e1",
        source_id="a",
        target_id="topic",
        relationship="ABOUT",
    )
    snapshot = graph.snapshot(namespace="episode")
    assert snapshot.node_count == 2
    assert snapshot.edge_count == 1
    assert snapshot.live_node_count == 2
    with pytest.raises(Neo4jError, match="not empty"):
        validate_empty_namespace(graph, "episode")


def test_archived_nodes_are_retained_in_snapshot_but_not_live_count() -> None:
    graph = FakeNeo4jTransport()
    graph.add_node(namespace="episode", node_id="a", properties={"memory": "a"})
    graph.archive_node(namespace="episode", node_id="a")
    snapshot = graph.snapshot(namespace="episode")
    assert snapshot.node_count == 1
    assert snapshot.live_node_count == 0
    assert snapshot.nodes[0].status == "ARCHIVED"


def test_snapshot_round_trip_is_canonical() -> None:
    snapshot = Neo4jGraphSnapshot(
        namespace="episode",
        nodes=(Neo4jNode("a", properties=(("memory", "text"),)),),
        edges=(),
    )
    assert Neo4jGraphSnapshot.from_dict(snapshot.to_dict()).to_dict() == snapshot.to_dict()


def test_bolt_exclusive_database_snapshots_official_memos_graph_without_namespace_property(
) -> None:
    driver = _FakeDriver(
        [
            [{"node_count": 0}],
            [
                {
                    "nodes": [
                        {
                            "node_id": "memory-a",
                            "labels": ["Memory"],
                            "properties": {
                                "id": "memory-a",
                                "memory": "offline only",
                                "created_at": _TemporalValue(),
                            },
                        },
                        {
                            "node_id": "memory-b",
                            "labels": ["Memory"],
                            "properties": {"id": "memory-b", "memory": "safe v2"},
                        },
                    ],
                    "edges": [
                        {
                            "edge_id": "edge-1",
                            "source_id": "memory-a",
                            "target_id": "memory-b",
                            "relationship": "RELATE_TO",
                            "properties": {},
                        }
                    ],
                }
            ],
        ]
    )
    transport = Neo4jBoltTransport(
        "bolt://127.0.0.1:7687",
        user="neo4j",
        password="secret",
        driver=driver,
        exclusive_database=True,
    )

    transport.validate_empty(namespace="prefix-task")
    snapshot = transport.snapshot(namespace="prefix-task")

    assert [node.node_id for node in snapshot.nodes] == ["memory-a", "memory-b"]
    assert snapshot.nodes[0].property_map["created_at"] == "2026-07-17T00:00:00Z"
    assert snapshot.edges[0].source_id == "memory-a"
    assert all("lhmsb_namespace" not in query for query, _params in driver.calls)
    assert all(not params for _query, params in driver.calls)
    assert len(driver.calls) == 2
    assert "collect(" in driver.calls[1][0]
