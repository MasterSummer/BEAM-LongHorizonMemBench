"""Minimal, strict Neo4j graph boundary for the controlled MemOS-Tree track.

The MemOS release exposes a large graph-memory surface, while the benchmark
only needs a small evaluator-owned view of the graph.  This module deliberately
keeps that view narrow: immutable node/edge snapshots, namespace isolation,
empty-volume validation, and a byte-count hook.  The fake implementation is
used by repository tests; :class:`Neo4jBoltTransport` is a lazy boundary for
the pinned official Bolt driver on the server.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast


class Neo4jError(RuntimeError):
    """Terminal failure at the benchmark-owned Neo4j boundary."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Neo4jError("invalid_graph_field", f"{field} must be a non-empty string")
    return value


def _pairs(value: object, field: str) -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        values = tuple((str(key), child) for key, child in value.items())
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        output: list[tuple[str, object]] = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
                raise Neo4jError("invalid_graph_field", f"{field} must contain pairs")
            output.append((_nonempty(item[0], f"{field} key"), item[1]))
        values = tuple(output)
    else:
        raise Neo4jError("invalid_graph_field", f"{field} must be a mapping or pairs")
    keys = [key for key, _ in values]
    if len(keys) != len(set(keys)):
        raise Neo4jError("duplicate_graph_field", f"{field} keys must be unique")
    try:
        json.dumps(dict(values), ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise Neo4jError("invalid_graph_field", f"{field} must be JSON-compatible") from exc
    return values


def _canonical_bolt_value(value: object) -> object:
    """Normalize Neo4j temporal/container values into stable JSON values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_bolt_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_canonical_bolt_value(child) for child in value]
    for method_name in ("iso_format", "isoformat"):
        method = getattr(value, method_name, None)
        if callable(method):
            rendered = method()
            if isinstance(rendered, str) and rendered:
                return rendered
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        native = to_native()
        if native is not value:
            return _canonical_bolt_value(native)
    raise Neo4jError(
        "invalid_graph_field",
        f"unsupported Neo4j property type: {type(value).__name__}",
    )


@dataclass(frozen=True)
class Neo4jNode:
    """One graph node in evaluator-side canonical form.

    ``labels`` and ``properties`` retain the native graph shape.  A node is
    considered retrievable when it does not have an archived/deleted status;
    this rule is applied by :func:`is_live_node` and the fake/bolt transports.
    """

    node_id: str
    labels: tuple[str, ...] = ()
    properties: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.node_id, "node_id")
        object.__setattr__(self, "labels", tuple(self.labels))
        if any(not isinstance(label, str) or not label for label in self.labels):
            raise Neo4jError("invalid_graph_field", "labels must contain non-empty strings")
        if len(self.labels) != len(set(self.labels)):
            raise Neo4jError("duplicate_graph_field", "node labels must be unique")
        object.__setattr__(self, "properties", _pairs(self.properties, "properties"))

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def memory_id(self) -> str:
        return self.node_id

    @property
    def property_map(self) -> dict[str, object]:
        return dict(self.properties)

    @property
    def status(self) -> str:
        value = self.property_map.get("status", self.property_map.get("memory_status", "ACTIVE"))
        return value.upper() if isinstance(value, str) else "ACTIVE"

    @property
    def kind(self) -> str:
        values = self.property_map
        raw = values.get("node_kind", values.get("kind", values.get("type", "content")))
        return raw if isinstance(raw, str) and raw else "content"

    @property
    def content(self) -> str:
        values = self.property_map
        raw = values.get("memory", values.get("content", values.get("text", "")))
        return raw if isinstance(raw, str) else str(raw)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "labels": list(self.labels),
            "properties": [[key, value] for key, value in self.properties],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Neo4jNode:
        labels = data.get("labels", ())
        if not isinstance(labels, Sequence) or isinstance(labels, str | bytes):
            raise Neo4jError("invalid_graph_field", "labels must be an array")
        return cls(
            node_id=_nonempty(data.get("node_id", data.get("id")), "node_id"),
            labels=tuple(_nonempty(value, "label") for value in labels),
            properties=_pairs(data.get("properties", ()), "properties"),
        )


@dataclass(frozen=True)
class Neo4jEdge:
    """One graph edge; edges are reported separately from memory objects."""

    edge_id: str
    source_id: str
    target_id: str
    relationship: str
    properties: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        for value, field in (
            (self.edge_id, "edge_id"),
            (self.source_id, "source_id"),
            (self.target_id, "target_id"),
            (self.relationship, "relationship"),
        ):
            _nonempty(value, field)
        object.__setattr__(self, "properties", _pairs(self.properties, "edge properties"))

    @property
    def id(self) -> str:
        return self.edge_id

    @property
    def type(self) -> str:
        return self.relationship

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "properties": [[key, value] for key, value in self.properties],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Neo4jEdge:
        return cls(
            edge_id=_nonempty(data.get("edge_id", data.get("id")), "edge_id"),
            source_id=_nonempty(data.get("source_id", data.get("source")), "source_id"),
            target_id=_nonempty(data.get("target_id", data.get("target")), "target_id"),
            relationship=_nonempty(data.get("relationship", data.get("type")), "relationship"),
            properties=_pairs(data.get("properties", ()), "edge properties"),
        )


@dataclass(frozen=True)
class Neo4jGraphSnapshot:
    """Immutable graph snapshot used to derive MemOS mutation lineage."""

    namespace: str
    nodes: tuple[Neo4jNode, ...]
    edges: tuple[Neo4jEdge, ...]
    observed_at: float = 0.0

    def __post_init__(self) -> None:
        _nonempty(self.namespace, "namespace")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise Neo4jError("duplicate_node_id", "graph snapshot node IDs must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise Neo4jError("duplicate_edge_id", "graph snapshot edge IDs must be unique")
        if any(
            edge.source_id not in node_ids or edge.target_id not in node_ids for edge in self.edges
        ):
            raise Neo4jError("dangling_edge", "graph snapshot contains an edge to an unknown node")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def live_nodes(self) -> tuple[Neo4jNode, ...]:
        return tuple(node for node in self.nodes if is_live_node(node))

    @property
    def live_node_count(self) -> int:
        return len(self.live_nodes)

    @property
    def content_hash(self) -> str:
        payload = {
            "namespace": self.namespace,
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.node_id)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: item.edge_id)],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def store_hash(self) -> str:
        return self.content_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Neo4jGraphSnapshot:
        nodes = data.get("nodes", ())
        edges = data.get("edges", ())
        if not isinstance(nodes, Sequence) or isinstance(nodes, str | bytes):
            raise Neo4jError("invalid_graph_field", "nodes must be an array")
        if not isinstance(edges, Sequence) or isinstance(edges, str | bytes):
            raise Neo4jError("invalid_graph_field", "edges must be an array")
        observed = data.get("observed_at", 0.0)
        if isinstance(observed, bool) or not isinstance(observed, int | float):
            raise Neo4jError("invalid_graph_field", "observed_at must be numeric")
        return cls(
            namespace=_nonempty(data.get("namespace"), "namespace"),
            nodes=tuple(Neo4jNode.from_dict(cast(Mapping[str, object], value)) for value in nodes),
            edges=tuple(Neo4jEdge.from_dict(cast(Mapping[str, object], value)) for value in edges),
            observed_at=float(observed),
        )


# Short aliases keep downstream adapter code readable and tolerate both names
# used in internal experiment notebooks.
GraphNode = Neo4jNode
GraphEdge = Neo4jEdge
GraphSnapshot = Neo4jGraphSnapshot


def is_live_node(node: Neo4jNode) -> bool:
    """Return whether a graph node can participate in retrieval."""
    return node.status.upper() not in {"ARCHIVED", "DELETED", "REVOKED", "REMOVED"}


class Neo4jTransport(Protocol):
    """Small graph transport consumed by the MemOS qualification adapter."""

    def validate_empty(self, *, namespace: str) -> None: ...

    def snapshot(self, *, namespace: str) -> Neo4jGraphSnapshot: ...

    def storage_bytes(self, *, namespace: str) -> int | None: ...

    def close(self) -> None: ...


class FakeNeo4jTransport:
    """Deterministic in-memory graph with strict namespace isolation."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Neo4jNode]] = {}
        self._edges: dict[str, dict[str, Neo4jEdge]] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise Neo4jError("transport_closed", "Neo4j transport is closed")

    def validate_empty(self, *, namespace: str) -> None:
        self._ensure_open()
        _nonempty(namespace, "namespace")
        if self._nodes.get(namespace) or self._edges.get(namespace):
            raise Neo4jError(
                "namespace_contamination", f"Neo4j namespace {namespace!r} is not empty"
            )

    def ensure_empty(self, namespace: str) -> None:
        self.validate_empty(namespace=namespace)

    def snapshot(self, *, namespace: str) -> Neo4jGraphSnapshot:
        self._ensure_open()
        _nonempty(namespace, "namespace")
        return Neo4jGraphSnapshot(
            namespace=namespace,
            nodes=tuple(
                sorted(self._nodes.get(namespace, {}).values(), key=lambda item: item.node_id)
            ),
            edges=tuple(
                sorted(self._edges.get(namespace, {}).values(), key=lambda item: item.edge_id)
            ),
            observed_at=0.0,
        )

    def add_node(
        self,
        *,
        namespace: str,
        node_id: str,
        labels: Sequence[str] = (),
        properties: Mapping[str, object] | Sequence[Sequence[object]] = (),
    ) -> Neo4jNode:
        self._ensure_open()
        _nonempty(namespace, "namespace")
        node = Neo4jNode(node_id, tuple(labels), _pairs(properties, "properties"))
        bucket = self._nodes.setdefault(namespace, {})
        if node_id in bucket:
            raise Neo4jError("duplicate_node_id", f"node {node_id!r} already exists")
        bucket[node_id] = node
        return node

    def upsert_node(
        self,
        *,
        namespace: str,
        node_id: str,
        labels: Sequence[str] = (),
        properties: Mapping[str, object] | Sequence[Sequence[object]] = (),
    ) -> Neo4jNode:
        self._ensure_open()
        _nonempty(namespace, "namespace")
        node = Neo4jNode(node_id, tuple(labels), _pairs(properties, "properties"))
        self._nodes.setdefault(namespace, {})[node_id] = node
        return node

    def archive_node(self, *, namespace: str, node_id: str, status: str = "ARCHIVED") -> None:
        self._ensure_open()
        bucket = self._nodes.get(namespace, {})
        old = bucket.get(node_id)
        if old is None:
            raise Neo4jError("unknown_node", f"node {node_id!r} is unknown")
        values = old.property_map
        values["status"] = status
        bucket[node_id] = Neo4jNode(old.node_id, old.labels, tuple(values.items()))

    def delete_node(self, *, namespace: str, node_id: str) -> None:
        self._ensure_open()
        self._nodes.get(namespace, {}).pop(node_id, None)
        edges = self._edges.get(namespace, {})
        for edge_id, edge in list(edges.items()):
            if edge.source_id == node_id or edge.target_id == node_id:
                del edges[edge_id]

    def add_edge(
        self,
        *,
        namespace: str,
        edge_id: str,
        source_id: str,
        target_id: str,
        relationship: str,
        properties: Mapping[str, object] | Sequence[Sequence[object]] = (),
    ) -> Neo4jEdge:
        self._ensure_open()
        nodes = self._nodes.get(namespace, {})
        if source_id not in nodes or target_id not in nodes:
            raise Neo4jError("dangling_edge", "edge endpoints must exist in the namespace")
        edge = Neo4jEdge(
            edge_id, source_id, target_id, relationship, _pairs(properties, "properties")
        )
        bucket = self._edges.setdefault(namespace, {})
        if edge_id in bucket:
            raise Neo4jError("duplicate_edge_id", f"edge {edge_id!r} already exists")
        bucket[edge_id] = edge
        return edge

    def remove_edge(self, *, namespace: str, edge_id: str) -> None:
        self._ensure_open()
        self._edges.get(namespace, {}).pop(edge_id, None)

    def clear(self, *, namespace: str) -> None:
        self._ensure_open()
        self._nodes.pop(namespace, None)
        self._edges.pop(namespace, None)

    def storage_bytes(self, *, namespace: str) -> int | None:
        snapshot = self.snapshot(namespace=namespace)
        return len(json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True).encode())

    def close(self) -> None:
        self._closed = True

    # Convenience names used by fake tests and notebooks.
    graph_snapshot = snapshot


class Neo4jBoltTransport:
    """Lazy official Bolt-driver boundary.

    No ``neo4j`` import happens at module import time.  A preparation job may
    execute several episode prefixes against one Community database.  MemOS
    stores its task identity in the native ``user_name`` property, while other
    adapters may use the benchmark-owned ``lhmsb_namespace`` property.  The
    selected property is validated against a fixed allow-list before it is
    interpolated into Cypher.
    """

    def __init__(
        self,
        uri: str,
        *,
        user: str,
        password: str,
        database: str = "neo4j",
        driver: object | None = None,
        exclusive_database: bool = False,
        namespace_property: str = "lhmsb_namespace",
    ) -> None:
        if not uri or not user or not password:
            raise ValueError("Neo4j uri/user/password must be non-empty")
        self.uri = uri
        self.user = user
        self.database = database
        self.exclusive_database = exclusive_database
        if namespace_property not in {"lhmsb_namespace", "user_name"}:
            raise ValueError("unsupported Neo4j namespace property")
        self.namespace_property = namespace_property
        if driver is None:
            try:
                import neo4j  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - server dependency
                raise Neo4jError(
                    "driver_unavailable", "neo4j Python driver is not installed"
                ) from exc
            driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self._driver = driver
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise Neo4jError("transport_closed", "Neo4j transport is closed")

    def _run(self, query: str, **params: object) -> list[dict[str, object]]:
        self._ensure_open()
        try:
            driver = cast(Any, self._driver)
            with driver.session(database=self.database) as session:
                result = session.run(query, **params)
                rows: list[dict[str, object]] = []
                for row in result:
                    if hasattr(row, "data"):
                        rows.append(dict(row.data()))
                    elif isinstance(row, Mapping):
                        rows.append({str(key): value for key, value in row.items()})
                    else:
                        rows.append(dict(row))
                return rows
        except Neo4jError:
            raise
        except Exception as exc:  # pragma: no cover - server dependency
            raise Neo4jError("query_failure", type(exc).__name__) from exc

    def validate_empty(self, *, namespace: str) -> None:
        _nonempty(namespace, "namespace")
        if self.exclusive_database:
            rows = self._run("MATCH (n) RETURN count(n) AS node_count")
        else:
            property_name = self.namespace_property
            rows = self._run(
                f"MATCH (n) WHERE n.{property_name} = $namespace "
                "RETURN count(n) AS node_count",
                namespace=namespace,
            )
        count = rows[0].get("node_count", 0) if rows else 0
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise Neo4jError(
                "namespace_contamination", f"Neo4j namespace {namespace!r} is not empty"
            )

    ensure_empty = validate_empty

    def snapshot(self, *, namespace: str) -> Neo4jGraphSnapshot:
        _nonempty(namespace, "namespace")
        if self.exclusive_database:
            rows = self._run(
                "MATCH (n) "
                "WITH collect({node_id: coalesce(n.id, elementId(n)), "
                "labels: labels(n), properties: properties(n)}) AS nodes "
                "OPTIONAL MATCH (a)-[r]->(b) "
                "RETURN nodes, collect(CASE WHEN r IS NULL THEN null ELSE "
                "{edge_id: coalesce(r.id, elementId(r)), "
                "source_id: coalesce(a.id, elementId(a)), "
                "target_id: coalesce(b.id, elementId(b)), "
                "relationship: type(r), properties: properties(r)} END) AS edges"
            )
        else:
            property_name = self.namespace_property
            rows = self._run(
                f"MATCH (n) WHERE n.{property_name} = $namespace "
                "WITH collect({node_id: coalesce(n.id, elementId(n)), "
                "labels: labels(n), properties: properties(n)}) AS nodes "
                "OPTIONAL MATCH (a)-[r]->(b) "
                f"WHERE a.{property_name} = $namespace "
                f"AND b.{property_name} = $namespace "
                "RETURN nodes, collect(CASE WHEN r IS NULL THEN null ELSE "
                "{edge_id: coalesce(r.id, elementId(r)), "
                "source_id: coalesce(a.id, elementId(a)), "
                "target_id: coalesce(b.id, elementId(b)), "
                "relationship: type(r), properties: properties(r)} END) AS edges",
                namespace=namespace,
            )
        if len(rows) != 1:
            raise Neo4jError(
                "invalid_graph_result",
                "Neo4j atomic snapshot query must return exactly one row",
            )
        raw_nodes = rows[0].get("nodes", ())
        raw_edges = rows[0].get("edges", ())
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, str | bytes):
            raise Neo4jError("invalid_graph_field", "Neo4j nodes must be an array")
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, str | bytes):
            raise Neo4jError("invalid_graph_field", "Neo4j edges must be an array")
        nodes_list: list[Neo4jNode] = []
        for raw_row in raw_nodes:
            if not isinstance(raw_row, Mapping):
                raise Neo4jError("invalid_graph_field", "Neo4j node must be a mapping")
            row = cast(Mapping[str, object], raw_row)
            raw_labels = row.get("labels", ())
            if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, str | bytes):
                raise Neo4jError("invalid_graph_field", "Neo4j labels must be an array")
            nodes_list.append(
                Neo4jNode(
                    _nonempty(row.get("node_id"), "node_id"),
                    tuple(_nonempty(label, "label") for label in raw_labels),
                    _pairs(
                        _canonical_bolt_value(row.get("properties", {})),
                        "properties",
                    ),
                )
            )
        nodes = tuple(sorted(nodes_list, key=lambda item: item.node_id))
        edges = tuple(
            Neo4jEdge(
                _nonempty(row.get("edge_id"), "edge_id"),
                _nonempty(row.get("source_id"), "source_id"),
                _nonempty(row.get("target_id"), "target_id"),
                _nonempty(row.get("relationship"), "relationship"),
                _pairs(
                    _canonical_bolt_value(row.get("properties", {})),
                    "edge properties",
                ),
            )
            for raw_row in raw_edges
            if isinstance(raw_row, Mapping)
            for row in (cast(Mapping[str, object], raw_row),)
        )
        if len(edges) != len(raw_edges):
            raise Neo4jError("invalid_graph_field", "Neo4j edge must be a mapping")
        return Neo4jGraphSnapshot(
            namespace,
            nodes,
            tuple(sorted(edges, key=lambda item: item.edge_id)),
            observed_at=time.time(),
        )

    def storage_bytes(self, *, namespace: str) -> int | None:
        del namespace
        # Neo4j Community does not expose a portable per-label byte count via
        # the Bolt API.  Keep the unavailable reason at the adapter boundary.
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._driver, "close", None)
        if callable(close):
            close()


def validate_empty_namespace(transport: Neo4jTransport, namespace: str) -> None:
    """Validate a fresh preparation namespace before the first native write."""
    transport.validate_empty(namespace=namespace)


Neo4jGraphStore = Neo4jTransport
FakeNeo4jGraph = FakeNeo4jTransport
InMemoryNeo4jTransport = FakeNeo4jTransport
FakeNeo4jStore = FakeNeo4jTransport
Neo4jGraphBoundary = Neo4jTransport
Neo4jInventoryBoundary = Neo4jTransport


__all__ = [
    "FakeNeo4jGraph",
    "FakeNeo4jStore",
    "FakeNeo4jTransport",
    "GraphEdge",
    "GraphNode",
    "GraphSnapshot",
    "Neo4jBoltTransport",
    "Neo4jEdge",
    "Neo4jError",
    "Neo4jGraphSnapshot",
    "Neo4jGraphStore",
    "Neo4jGraphBoundary",
    "Neo4jInventoryBoundary",
    "Neo4jNode",
    "Neo4jTransport",
    "InMemoryNeo4jTransport",
    "is_live_node",
    "validate_empty_namespace",
]
