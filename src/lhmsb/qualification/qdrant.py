"""Small, strict Qdrant boundary used by the controlled flat baseline.

The benchmark only needs a deliberately tiny subset of Qdrant's HTTP API.  The
boundary in this module keeps the rest of the qualification code independent of
``qdrant-client`` (and therefore easy to exercise offline).  Both
:class:`QdrantHttpTransport` and :class:`InMemoryQdrantTransport` implement the
same :class:`QdrantTransport` protocol.  The latter is intended for tests and
repository-only dry runs; it is not a second retrieval implementation.

Benchmark point IDs (normally the SHA-256 ID of a ``PublicHistoryUnit``) are
kept in evaluator-facing transport records.  The HTTP implementation maps them
to deterministic UUIDs at the Qdrant wire boundary because Qdrant accepts only
unsigned integers or UUIDs as point IDs.  Namespace is encoded in the payload
and every read/write/delete operation requires it explicitly.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import httpx


class QdrantError(RuntimeError):
    """Terminal failure at the benchmark-owned Qdrant boundary."""

    def __init__(
        self,
        error_class: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code


@dataclass(frozen=True)
class QdrantPoint:
    """One point accepted by the minimal Qdrant transport."""

    point_id: str
    vector: tuple[float, ...]
    payload: Mapping[str, object]

    @property
    def id(self) -> str:
        """Alias matching Qdrant's JSON field name."""
        return self.point_id


@dataclass(frozen=True)
class QdrantHit:
    """One scored point returned by a vector search."""

    point_id: str
    score: float
    payload: Mapping[str, object]

    @property
    def id(self) -> str:
        return self.point_id


class QdrantTransport(Protocol):
    """Minimal transport contract consumed by ``FlatRetrievalAdapter``.

    Implementations may return the typed records above, mappings shaped like
    Qdrant JSON, or a sequence of either.  The flat adapter normalizes and
    validates those values before constructing the public trace.
    """

    def create_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None: ...

    def count(self, *, collection_name: str, namespace: str) -> int: ...

    def upsert(
        self,
        *,
        collection_name: str,
        namespace: str,
        points: Sequence[QdrantPoint],
    ) -> None: ...

    def scroll(
        self,
        *,
        collection_name: str,
        namespace: str,
        limit: int = 10_000,
    ) -> Sequence[QdrantPoint | Mapping[str, object]]: ...

    def search(
        self,
        *,
        collection_name: str,
        namespace: str,
        vector: Sequence[float],
        limit: int,
    ) -> Sequence[QdrantHit | Mapping[str, object]]: ...

    def delete_namespace(self, *, collection_name: str, namespace: str) -> None: ...

    def health(self) -> bool: ...

    def close(self) -> None: ...


_WIRE_POINT_ID_KEY = "_lhmsb_point_id"
_WIRE_POINT_NAMESPACE = uuid.UUID("7e2b6f3b-99ee-4a62-9b35-4f4e7e8f5f4e")


def validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise QdrantError("invalid_namespace", "namespace must be a non-empty string")
    return namespace


def validate_collection_name(collection_name: str) -> str:
    if not isinstance(collection_name, str) or not collection_name:
        raise QdrantError(
            "invalid_collection", "collection_name must be a non-empty string"
        )
    return collection_name


def validate_point_id(point_id: object) -> str:
    if not isinstance(point_id, str) or not point_id:
        raise QdrantError("invalid_point_id", "point ID must be a non-empty string")
    return point_id


def validate_vector(
    vector: Iterable[object],
    *,
    dimension: int,
    field: str = "vector",
) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise QdrantError("invalid_vector", f"{field} must be a numeric array")
    if dimension < 1:
        raise QdrantError("invalid_vector_dimension", "vector dimension must be positive")
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise QdrantError("invalid_vector", f"{field} contains a non-number")
        number = float(value)
        if not math.isfinite(number):
            raise QdrantError("invalid_vector", f"{field} contains a non-finite number")
        values.append(number)
    if len(values) != dimension:
        raise QdrantError(
            "vector_dimension_mismatch",
            f"{field} dimension {len(values)} != expected {dimension}",
        )
    return tuple(values)


def validate_score(score: object) -> float:
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise QdrantError("invalid_score", "Qdrant score must be numeric")
    value = float(score)
    if not math.isfinite(value):
        raise QdrantError("invalid_score", "Qdrant score must be finite")
    return value


def _payload_copy(payload: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in payload.items()}


def _wire_point_id(namespace: str, point_id: str) -> str:
    """Return a stable Qdrant-compatible ID for a benchmark point."""
    return str(uuid.uuid5(_WIRE_POINT_NAMESPACE, f"{namespace}\x00{point_id}"))


def _to_wire_point(
    point: QdrantPoint,
    *,
    namespace: str,
) -> QdrantPoint:
    payload = _payload_copy(point.payload)
    existing_namespace = payload.get("_namespace", payload.get("namespace"))
    if existing_namespace is not None and existing_namespace != namespace:
        raise QdrantError("namespace_mismatch", "point payload namespace differs")
    existing = payload.get(_WIRE_POINT_ID_KEY)
    if existing is not None and existing != point.point_id:
        raise QdrantError(
            "point_id_mismatch",
            "point payload contains a conflicting benchmark point ID",
        )
    payload[_WIRE_POINT_ID_KEY] = point.point_id
    payload["_namespace"] = namespace
    return QdrantPoint(
        _wire_point_id(namespace, point.point_id),
        point.vector,
        payload,
    )


def _restore_payload(payload: Mapping[str, object]) -> tuple[str | None, dict[str, object]]:
    copied = _payload_copy(payload)
    raw_point_id = copied.pop(_WIRE_POINT_ID_KEY, None)
    if raw_point_id is None:
        return None, copied
    return validate_point_id(raw_point_id), copied


def _restore_point(point: QdrantPoint) -> QdrantPoint:
    original_id, payload = _restore_payload(point.payload)
    return QdrantPoint(original_id or point.point_id, point.vector, payload)


def _restore_hit(hit: QdrantHit) -> QdrantHit:
    original_id, payload = _restore_payload(hit.payload)
    return QdrantHit(original_id or hit.point_id, hit.score, payload)


def _point_from_value(
    value: QdrantPoint | Mapping[str, object],
    *,
    dimension: int,
) -> QdrantPoint:
    if isinstance(value, QdrantPoint):
        point_id = value.point_id
        vector = value.vector
        payload = value.payload
    elif isinstance(value, Mapping):
        raw_id = value.get("id", value.get("point_id"))
        raw_vector = value.get("vector", ())
        raw_payload = value.get("payload", {})
        point_id = validate_point_id(raw_id)
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
            raise QdrantError("invalid_vector", "point vector must be an array")
        vector = tuple(cast(Iterable[float], raw_vector))
        if not isinstance(raw_payload, Mapping):
            raise QdrantError("invalid_payload", "point payload must be an object")
        payload = raw_payload
    else:
        raise QdrantError("invalid_point", "Qdrant point must be an object")
    point_id = validate_point_id(point_id)
    validated_vector = validate_vector(vector, dimension=dimension)
    return QdrantPoint(point_id, validated_vector, _payload_copy(payload))


def _hit_from_value(
    value: QdrantHit | Mapping[str, object],
    *,
    dimension: int | None = None,
) -> QdrantHit:
    if isinstance(value, QdrantHit):
        point_id = value.point_id
        score = value.score
        payload = value.payload
    elif isinstance(value, Mapping):
        point_id = validate_point_id(value.get("id", value.get("point_id")))
        score = validate_score(value.get("score"))
        raw_payload = value.get("payload", {})
        if not isinstance(raw_payload, Mapping):
            raise QdrantError("invalid_payload", "search payload must be an object")
        payload = raw_payload
    else:
        raise QdrantError("invalid_search_hit", "Qdrant search hit must be an object")
    point_id = validate_point_id(point_id)
    score = validate_score(score)
    if dimension is not None and "vector" in payload:
        raw_vector = payload["vector"]
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
            raise QdrantError("invalid_vector", "search vector must be an array")
        validate_vector(raw_vector, dimension=dimension, field="search vector")
    return QdrantHit(point_id, score, _payload_copy(payload))


def validate_search_order(hits: Sequence[QdrantHit]) -> tuple[QdrantHit, ...]:
    """Validate and return canonical score/ID order.

    Qdrant normally returns this order itself.  We still enforce it at the
    boundary so a malformed fake or server response cannot silently change the
    retrieval trace.  Ties are ordered lexicographically by the deterministic
    benchmark point ID.
    """
    for hit in hits:
        validate_point_id(hit.point_id)
        validate_score(hit.score)
    ids = [hit.point_id for hit in hits]
    if len(ids) != len(set(ids)):
        raise QdrantError("duplicate_point_id", "Qdrant search returned duplicate point IDs")
    ordered = tuple(sorted(hits, key=lambda hit: (-hit.score, hit.point_id)))
    if tuple(hits) != ordered:
        raise QdrantError(
            "invalid_point_order",
            "Qdrant search points are not in descending score/ID order",
        )
    return ordered


def _validate_qdrant_score_order(hits: Sequence[QdrantHit]) -> None:
    """Validate the order guaranteed by Qdrant without imposing tie order."""
    ids: list[str] = []
    previous_score: float | None = None
    for hit in hits:
        validate_point_id(hit.point_id)
        score = validate_score(hit.score)
        ids.append(hit.point_id)
        if previous_score is not None and score > previous_score:
            raise QdrantError(
                "invalid_point_order",
                "Qdrant search points are not in descending score order",
            )
        previous_score = score
    if len(ids) != len(set(ids)):
        raise QdrantError("duplicate_point_id", "Qdrant search returned duplicate point IDs")


class InMemoryQdrantTransport:
    """Strict in-memory implementation used by offline tests and dry runs."""

    def __init__(self) -> None:
        self._collections: dict[str, tuple[int, dict[tuple[str, str], QdrantPoint]]] = {}
        self._closed = False

    def create_collection(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        del distance
        self._ensure_open()
        name = validate_collection_name(collection_name)
        if vector_size < 1:
            raise QdrantError("invalid_vector_dimension", "vector_size must be positive")
        existing = self._collections.get(name)
        if existing is not None and existing[0] != vector_size:
            raise QdrantError(
                "vector_dimension_mismatch",
                f"collection dimension {existing[0]} != requested {vector_size}",
            )
        if existing is None:
            self._collections[name] = (vector_size, {})

    def count(self, *, collection_name: str, namespace: str) -> int:
        dimension, points = self._collection(collection_name)
        del dimension
        ns = validate_namespace(namespace)
        return sum(1 for key in points if key[0] == ns)

    def upsert(
        self,
        *,
        collection_name: str,
        namespace: str,
        points: Sequence[QdrantPoint],
    ) -> None:
        dimension, stored = self._collection(collection_name)
        ns = validate_namespace(namespace)
        seen: set[str] = set()
        for point in points:
            normalized = _point_from_value(point, dimension=dimension)
            if normalized.point_id in seen:
                raise QdrantError("duplicate_point_id", "upsert contains duplicate point IDs")
            seen.add(normalized.point_id)
            payload = _payload_copy(normalized.payload)
            payload_namespace = payload.get("_namespace", payload.get("namespace"))
            if payload_namespace is not None and payload_namespace != ns:
                raise QdrantError("namespace_mismatch", "point payload namespace differs")
            payload["_namespace"] = ns
            key = (ns, normalized.point_id)
            candidate = QdrantPoint(normalized.point_id, normalized.vector, payload)
            previous = stored.get(key)
            if previous is not None and previous != candidate:
                raise QdrantError(
                    "conflicting_point",
                    f"point {normalized.point_id!r} already exists with different content",
                )
            stored[key] = candidate

    def scroll(
        self,
        *,
        collection_name: str,
        namespace: str,
        limit: int = 10_000,
    ) -> Sequence[QdrantPoint]:
        if limit < 1:
            raise QdrantError("invalid_limit", "scroll limit must be positive")
        _, stored = self._collection(collection_name)
        ns = validate_namespace(namespace)
        return tuple(stored[key] for key in sorted(stored) if key[0] == ns)[:limit]

    def search(
        self,
        *,
        collection_name: str,
        namespace: str,
        vector: Sequence[float],
        limit: int,
    ) -> Sequence[QdrantHit]:
        dimension, stored = self._collection(collection_name)
        ns = validate_namespace(namespace)
        if limit < 1:
            raise QdrantError("invalid_limit", "search limit must be positive")
        query = validate_vector(vector, dimension=dimension, field="query vector")
        candidates: list[QdrantHit] = []
        for (point_namespace, point_id), point in stored.items():
            if point_namespace != ns:
                continue
            score = _cosine(query, point.vector)
            candidates.append(QdrantHit(point_id, score, _payload_copy(point.payload)))
        ordered = tuple(sorted(candidates, key=lambda hit: (-hit.score, hit.point_id)))
        return validate_search_order(ordered[:limit])

    def delete_namespace(self, *, collection_name: str, namespace: str) -> None:
        _, stored = self._collection(collection_name)
        ns = validate_namespace(namespace)
        for key in tuple(stored):
            if key[0] == ns:
                del stored[key]

    def health(self) -> bool:
        return not self._closed

    def close(self) -> None:
        self._closed = True

    def _collection(
        self,
        collection_name: str,
    ) -> tuple[int, dict[tuple[str, str], QdrantPoint]]:
        self._ensure_open()
        name = validate_collection_name(collection_name)
        try:
            return self._collections[name]
        except KeyError as exc:
            raise QdrantError("collection_not_found", f"unknown collection {name!r}") from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise QdrantError("transport_closed", "Qdrant transport is closed")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class QdrantHttpTransport:
    """HTTP implementation of the minimal Qdrant boundary."""

    def __init__(
        self,
        base_url: str,
        *,
        collection_name: str,
        vector_size: int,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not base_url or not collection_name:
            raise ValueError("base_url and collection_name must be non-empty")
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        self.collection_name = validate_collection_name(collection_name)
        self.vector_size = vector_size
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )
        self._closed = False

    def create_collection(
        self,
        *,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine",
    ) -> None:
        collection = self._collection(collection_name)
        dimension = self.vector_size if vector_size is None else vector_size
        self._validate_dimension(dimension)
        if not distance:
            raise QdrantError("invalid_distance", "distance must be non-empty")
        self._request(
            "PUT",
            f"/collections/{collection}",
            {"vectors": {"size": dimension, "distance": distance}},
            error_class="create_collection_failure",
        )

    def count(self, *, collection_name: str | None = None, namespace: str) -> int:
        collection = self._collection(collection_name)
        body = {"exact": True, "filter": _namespace_filter(namespace)}
        raw = self._request(
            "POST",
            f"/collections/{collection}/points/count",
            body,
            error_class="count_failure",
        )
        result = _result_object(raw, "count response")
        value = result.get("count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QdrantError("invalid_count", "Qdrant count must be a non-negative integer")
        return value

    def upsert(
        self,
        *,
        collection_name: str | None = None,
        namespace: str,
        points: Sequence[QdrantPoint],
    ) -> None:
        collection = self._collection(collection_name)
        ns = validate_namespace(namespace)
        normalized: list[QdrantPoint] = []
        seen: set[str] = set()
        for point in points:
            value = _point_from_value(point, dimension=self.vector_size)
            if value.point_id in seen:
                raise QdrantError("duplicate_point_id", "upsert contains duplicate point IDs")
            seen.add(value.point_id)
            normalized.append(_to_wire_point(value, namespace=ns))
        self._request(
            "PUT",
            f"/collections/{collection}/points",
            {"wait": True, "points": [_point_json(point) for point in normalized]},
            error_class="upsert_failure",
        )

    def scroll(
        self,
        *,
        collection_name: str | None = None,
        namespace: str,
        limit: int = 10_000,
    ) -> Sequence[QdrantPoint]:
        collection = self._collection(collection_name)
        if limit < 1:
            raise QdrantError("invalid_limit", "scroll limit must be positive")
        raw = self._request(
            "POST",
            f"/collections/{collection}/points/scroll",
            {
                "limit": limit,
                "offset": None,
                "with_payload": True,
                "with_vector": True,
                "filter": _namespace_filter(namespace),
            },
            error_class="scroll_failure",
        )
        result = _result_object(raw, "scroll response")
        raw_points = result.get("points")
        if not isinstance(raw_points, list):
            raise QdrantError("invalid_points", "scroll response points must be an array")
        points = tuple(
            _restore_point(_point_from_value(point, dimension=self.vector_size))
            for point in raw_points
        )
        ids = [point.point_id for point in points]
        if len(ids) != len(set(ids)):
            raise QdrantError("duplicate_point_id", "scroll returned duplicate point IDs")
        return points

    def search(
        self,
        *,
        collection_name: str | None = None,
        namespace: str,
        vector: Sequence[float],
        limit: int,
    ) -> Sequence[QdrantHit]:
        collection = self._collection(collection_name)
        if limit < 1:
            raise QdrantError("invalid_limit", "search limit must be positive")
        query = validate_vector(vector, dimension=self.vector_size, field="query vector")
        raw = self._request(
            "POST",
            f"/collections/{collection}/points/search",
            {
                "vector": list(query),
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
                "filter": _namespace_filter(namespace),
            },
            error_class="search_failure",
        )
        result = _result_value(raw, "search response")
        if not isinstance(result, list):
            raise QdrantError("invalid_points", "search response result must be an array")
        wire_hits = tuple(_hit_from_value(hit) for hit in result)
        _validate_qdrant_score_order(wire_hits)
        hits = tuple(_restore_hit(hit) for hit in wire_hits)
        ordered = tuple(sorted(hits, key=lambda hit: (-hit.score, hit.point_id)))
        return validate_search_order(ordered)

    def delete_namespace(self, *, collection_name: str | None = None, namespace: str) -> None:
        collection = self._collection(collection_name)
        self._request(
            "POST",
            f"/collections/{collection}/points/delete",
            {"wait": True, "filter": _namespace_filter(namespace)},
            error_class="delete_failure",
        )

    def health(self) -> bool:
        if self._closed:
            return False
        try:
            response = self._client.get("/healthz")
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def _collection(self, collection_name: str | None) -> str:
        self._ensure_open()
        return validate_collection_name(collection_name or self.collection_name)

    def _validate_dimension(self, dimension: int) -> None:
        if dimension != self.vector_size:
            raise QdrantError(
                "vector_dimension_mismatch",
                f"vector dimension {dimension} != configured {self.vector_size}",
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise QdrantError("transport_closed", "Qdrant transport is closed")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object],
        *,
        error_class: str,
    ) -> dict[str, object]:
        self._ensure_open()
        try:
            response = self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise QdrantError(error_class, str(exc)) from exc
        if response.status_code >= 400:
            raise QdrantError(error_class, response.text, status_code=response.status_code)
        try:
            raw = response.json()
        except ValueError as exc:
            raise QdrantError(error_class, "Qdrant response is not JSON") from exc
        if not isinstance(raw, Mapping):
            raise QdrantError(error_class, "Qdrant response must be an object")
        return {str(key): value for key, value in raw.items()}


def _point_json(point: QdrantPoint) -> dict[str, object]:
    return {
        "id": point.point_id,
        "vector": list(point.vector),
        "payload": _payload_copy(point.payload),
    }


def _namespace_filter(namespace: str) -> dict[str, object]:
    ns = validate_namespace(namespace)
    return {"must": [{"key": "_namespace", "match": {"value": ns}}]}


def _result_value(raw: Mapping[str, object], label: str) -> object:
    if "result" not in raw:
        raise QdrantError("invalid_response", f"{label} lacks result")
    return raw["result"]


def _result_object(raw: Mapping[str, object], label: str) -> Mapping[str, object]:
    result = _result_value(raw, label)
    if not isinstance(result, Mapping):
        raise QdrantError("invalid_response", f"{label} result must be an object")
    return result


# Compatibility aliases make the boundary discoverable without coupling callers
# to one spelling used in the initial implementation plan.
QdrantClient = QdrantHttpTransport
QdrantHTTPTransport = QdrantHttpTransport


__all__ = [
    "InMemoryQdrantTransport",
    "QdrantClient",
    "QdrantError",
    "QdrantHTTPTransport",
    "QdrantHit",
    "QdrantHttpTransport",
    "QdrantPoint",
    "QdrantTransport",
    "validate_collection_name",
    "validate_namespace",
    "validate_point_id",
    "validate_score",
    "validate_search_order",
    "validate_vector",
]
