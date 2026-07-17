from __future__ import annotations

import pytest

from lhmsb.adapters.memos_qualification import (
    MemOSLLMConfig,
    MemOSQualificationError,
    MemOSTreeQualificationAdapter,
    _build_tree_config,
)
from lhmsb.qualification.memory_runtime import LifecycleCapabilities
from lhmsb.qualification.neo4j import FakeNeo4jTransport
from lhmsb.qualification.schema import MemOSTreeProfile


class FakeManager:
    def __init__(self) -> None:
        self.wait_calls = 0

    def wait_reorganizer(self, timeout: float | None = None) -> bool:
        del timeout
        self.wait_calls += 1
        return True

    def close(self) -> None:
        return None


class FakeTree:
    def __init__(self, graph: FakeNeo4jTransport, namespace: str) -> None:
        self.graph = graph
        self.namespace = namespace
        self.memory_manager = FakeManager()
        self.nodes: list[str] = []
        self.search_rows: list[dict[str, object]] | None = None
        self.archive_existing_on_add = False

    def add(self, payload: object, **kwargs: object) -> list[str]:
        del kwargs
        index = len(self.nodes)
        node_id = f"m{index}"
        text = str(payload)
        self.graph.add_node(
            namespace=self.namespace,
            node_id=node_id,
            labels=("TextualMemoryItem",),
            properties={"memory": text, "session_index": index},
        )
        self.nodes.append(node_id)
        if self.archive_existing_on_add and index > 0:
            self.graph.archive_node(namespace=self.namespace, node_id=self.nodes[0])
        return [node_id]

    def search(self, query: str, *, top_k: int = 20, **kwargs: object) -> list[dict[str, object]]:
        del query, kwargs
        if self.search_rows is not None:
            return self.search_rows
        return [
            {"id": node_id, "score": 1.0 / (index + 1)}
            for index, node_id in enumerate(self.nodes[:top_k])
        ]


def _adapter() -> tuple[MemOSTreeQualificationAdapter, FakeTree, FakeNeo4jTransport]:
    graph = FakeNeo4jTransport()
    backend = FakeTree(graph, "episode")
    adapter = MemOSTreeQualificationAdapter(
        backend,
        graph=graph,
        namespace="episode",
        episode_id="episode",
        candidate_k=2,
    )
    return adapter, backend, graph


def test_tree_adapter_writes_synchronously_and_normalizes_graph_inventory() -> None:
    adapter, backend, _graph = _adapter()
    result = adapter.write_session([{"role": "user", "content": "offline"}], session_index=0)
    assert backend.memory_manager.wait_calls == 1
    assert result.events[0].native_event == "ADD"
    inventory = adapter.snapshot_inventory(checkpoint_session=0)
    assert inventory.n_live == 1
    assert inventory.items[0].graph_metadata is not None
    assert adapter.capabilities == LifecycleCapabilities(
        add=True, update=True, delete=True, merge=True, links=True, history=True, resumable=False
    )
    search = adapter.search_candidates("offline", checkpoint_session=0)
    assert search.candidates[0].candidate_origin == "native"
    adapter.close()


def test_graph_expansion_and_archive_are_traceable() -> None:
    adapter, backend, graph = _adapter()
    adapter.write_session([{"role": "user", "content": "one"}], session_index=0)
    graph.add_node(
        namespace="episode",
        node_id="topic",
        labels=("Topic",),
        properties={"kind": "topic", "memory": "topic"},
    )
    graph.add_edge(
        namespace="episode", edge_id="edge", source_id="m0", target_id="topic", relationship="ABOUT"
    )
    backend.search_rows = [
        {"id": "m0", "score": 1.0},
        {"id": "topic", "score": 0.5, "is_graph_expanded": True},
    ]
    search = adapter.search_candidates("q", checkpoint_session=0)
    assert [candidate.candidate_origin for candidate in search.candidates] == [
        "native",
        "graph_expanded",
    ]
    backend.archive_existing_on_add = True
    write = adapter.write_session([{"role": "user", "content": "archive"}], session_index=1)
    assert any(event.native_event == "ARCHIVE" for event in write.events)
    assert all(
        item.memory_id != "m0" for item in adapter.snapshot_inventory(checkpoint_session=1).items
    )


def test_missing_wait_api_and_nonempty_namespace_are_terminal() -> None:
    graph = FakeNeo4jTransport()
    graph.add_node(namespace="episode", node_id="contaminated")
    with pytest.raises(MemOSQualificationError, match="not empty"):
        MemOSTreeQualificationAdapter(FakeTree(graph, "episode"), graph=graph, namespace="episode")

    class NoWait:
        def add(self, payload: object) -> None:
            del payload

        def search(self, query: str) -> list[object]:
            del query
            return []

    with pytest.raises(MemOSQualificationError, match="wait"):
        MemOSTreeQualificationAdapter(NoWait(), graph=FakeNeo4jTransport())


def test_tree_config_uses_explicit_native_embedding_and_neo4j_identity() -> None:
    profile = MemOSTreeProfile(profile_id="memos_tree_controlled")
    components = tuple(
        MemOSLLMConfig(
            component=name,
            model_id="deepseek-v4-pro",
            endpoint="https://api.deepseek.com",
            api_key="secret",
        )
        for name in ("reader", "extractor", "reorganizer", "dispatcher")
    )

    config = _build_tree_config(
        profile,
        components,
        embedding_base_url="http://127.0.0.1:18080",
        neo4j_uri="bolt://127.0.0.1:17687",
        neo4j_user="neo4j",
        neo4j_password="benchmark-password",
        neo4j_database="neo4j",
        neo4j_user_name="prefix-task-7",
    )

    assert isinstance(config, dict)
    embedder = config["embedder"]
    graph = config["graph_db"]
    assert isinstance(embedder, dict) and isinstance(graph, dict)
    assert embedder["backend"] == "universal_api"
    assert embedder["config"]["provider"] == "openai"
    assert embedder["config"]["base_url"] == "http://127.0.0.1:18080"
    # MemOS 2.0.23's TreeTextMemoryConfig owns only extractor/dispatcher LLMs;
    # reader and reorganizer are runtime components, and internet retrieval is
    # disabled by leaving its optional config unset.
    assert "reader_llm" not in config
    assert "reorganizer_llm" not in config
    assert "internet_retrieval" not in config
    assert graph["config"] == {
        "uri": "bolt://127.0.0.1:17687",
        "user": "neo4j",
        "password": "benchmark-password",
        "db_name": "neo4j",
        "use_multi_db": False,
        "user_name": "prefix-task-7",
        "auto_create": False,
        "embedding_dimension": 1024,
    }
