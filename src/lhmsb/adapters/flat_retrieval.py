"""Deterministic raw-history retrieval baseline for the controlled track.

``FlatRetrievalAdapter`` is intentionally a very small ``MemoryRuntime``.  It
stores one unchanged :class:`~lhmsb.qualification.context.PublicHistoryUnit`
per object, embeds that text once, and puts the resulting vector in the
benchmark-owned Qdrant boundary.  There is no extraction model, writer,
summarizer, merge, update, delete, link, or consolidation path.  The only
normalized mutation emitted by ``write_session`` is ``OBSERVED_ADD``.

The adapter is prefix-oriented rather than a general user-facing memory
adapter.  ``write_session`` receives the same public transcript boundary as
the managed systems, while callers may provide the canonical units directly at
construction (or in ``metadata['public_units']``).  Retrieval returns the
native candidate set only; the benchmark's common reranker remains an external
stage and is never invoked here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

from lhmsb.qualification.context import PublicHistoryKind, PublicHistoryUnit
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.qdrant import (
    InMemoryQdrantTransport,
    QdrantError,
    QdrantHit,
    QdrantPoint,
    QdrantTransport,
    validate_point_id,
    validate_search_order,
    validate_vector,
)


class FlatRetrievalError(RuntimeError):
    """Typed terminal failure in the deterministic flat baseline."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class EmbeddingRuntime(Protocol):
    """Small injectable embedding boundary.

    A runtime may return a raw sequence of vectors or an object (for example
    ``tei.EmbeddingBatch``) exposing ``vectors`` and optionally ``dimension``.
    ``dimension``/``expected_dimension`` on the runtime is preferred when the
    adapter is constructed without an explicit ``embedding_dimension``.
    """

    def embed(self, texts: tuple[str, ...]) -> object: ...


class DeterministicHashEmbedding:
    """Offline deterministic embedding useful for smoke tests.

    This is not the controlled BGE-M3 service.  Server runs inject the pinned
    TEI ``EmbeddingClient``; this implementation merely keeps repository-only
    tests network-free and enforces the same vector-dimension contract.
    """

    def __init__(self, dimension: int = 8) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            raise ValueError("embedding input must be non-empty")
        self.calls.append(tuple(texts))
        output: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = tuple(
                (digest[index % len(digest)] / 127.5) - 1.0
                for index in range(self.dimension)
            )
            output.append(values)
        return tuple(output)


class FlatRetrievalAdapter:
    """Raw public-history vector retrieval implementing ``MemoryRuntime``."""

    capabilities = LifecycleCapabilities(
        add=True,
        update=False,
        delete=False,
        merge=False,
        links=False,
        history=False,
        resumable=True,
    )

    def __init__(
        self,
        public_units: Sequence[PublicHistoryUnit] = (),
        *,
        units: Sequence[PublicHistoryUnit] | None = None,
        episode_id: str | None = None,
        namespace: str,
        embedding_runtime: EmbeddingRuntime | None = None,
        embedding: EmbeddingRuntime | None = None,
        qdrant: QdrantTransport | None = None,
        qdrant_transport: QdrantTransport | None = None,
        collection_name: str = "lhmsb_flat_retrieval",
        candidate_k: int = 20,
        embedding_dimension: int | None = None,
        inventory_limit: int = 100_000,
    ) -> None:
        if units is not None:
            if public_units:
                raise ValueError("provide public_units or units, not both")
            public_units = units
        if embedding_runtime is not None and embedding is not None:
            raise ValueError("provide embedding_runtime or embedding, not both")
        if qdrant is not None and qdrant_transport is not None:
            raise ValueError("provide qdrant or qdrant_transport, not both")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if inventory_limit < 1:
            raise ValueError("inventory_limit must be positive")
        self.namespace = namespace
        self.collection_name = collection_name
        self.candidate_k = candidate_k
        self.inventory_limit = inventory_limit
        self.episode_id = episode_id
        self._embedding = embedding_runtime or embedding or DeterministicHashEmbedding(
            dimension=embedding_dimension or 8
        )
        self._qdrant = qdrant or qdrant_transport or InMemoryQdrantTransport()
        self._embedding_dimension = self._resolve_dimension(
            embedding_dimension,
            self._embedding,
        )
        self._units_by_session: dict[int, tuple[PublicHistoryUnit, ...]] = {}
        self._objects: dict[str, MemoryObject] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._n_write = 0
        self._last_write_session = -1
        self._closed = False
        for unit in public_units:
            self._register_unit(unit)
        try:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vector_size=self._embedding_dimension,
                distance="Cosine",
            )
            existing = self._qdrant.count(
                collection_name=self.collection_name,
                namespace=self.namespace,
            )
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        if existing != 0:
            raise FlatRetrievalError(
                "non_empty_namespace",
                "flat retrieval requires an empty namespace at task start",
            )

    def restore_write_count(self, n_write: int) -> None:
        self._ensure_open()
        if isinstance(n_write, bool) or not isinstance(n_write, int) or n_write < 0:
            raise ValueError("n_write must be a non-negative integer")
        self._n_write = n_write

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        """Ingest one public session, emitting only deterministic ``OBSERVED_ADD``.

        When canonical units were supplied at construction, the session index
        selects those units.  A caller can override that selection with a
        ``metadata['public_units']`` sequence.  As a convenience for the shared
        Mem0 prefix runner, a JSON write transcript containing ``observations``
        and ``tool_results`` is also accepted.
        """
        self._ensure_open()
        if isinstance(session_index, bool) or not isinstance(session_index, int):
            raise ValueError("session_index must be an integer")
        if session_index < 0:
            raise ValueError("session_index must be non-negative")
        if session_index < self._last_write_session:
            raise FlatRetrievalError(
                "session_order",
                f"session {session_index} follows session {self._last_write_session}",
            )
        started = time.perf_counter()
        selected = self._session_units(messages, session_index, metadata)
        missing = [unit for unit in selected if unit.unit_id not in self._objects]
        events: list[MemoryMutationEvent] = []
        if missing:
            texts = tuple(unit.content for unit in missing)
            vectors = self._embed(texts)
            points = [
                QdrantPoint(
                    point_id=unit.unit_id,
                    vector=vector,
                    payload=self._point_payload(unit),
                )
                for unit, vector in zip(missing, vectors, strict=True)
            ]
            try:
                self._qdrant.upsert(
                    collection_name=self.collection_name,
                    namespace=self.namespace,
                    points=points,
                )
            except QdrantError as exc:
                raise FlatRetrievalError(exc.error_class, str(exc)) from exc
            for unit, vector in zip(missing, vectors, strict=True):
                item = self._memory_object(unit)
                self._objects[unit.unit_id] = item
                self._vectors[unit.unit_id] = vector
                self._n_write += 1
                events.append(
                    MemoryMutationEvent(
                        operation_id=self._operation_id(unit),
                        session_index=session_index,
                        native_event="OBSERVED_ADD",
                        memory_id=unit.unit_id,
                        memory_text=unit.content,
                        old_content_hash=None,
                        new_content_hash=unit.content_sha256,
                        source="public_history",
                        latency_seconds=0.0,
                    )
                )
        self._last_write_session = max(self._last_write_session, session_index)
        latency = max(0.0, time.perf_counter() - started)
        inventory = self.snapshot_inventory(checkpoint_session=session_index)
        return WriteSessionResult(
            session_index=session_index,
            events=tuple(events),
            inventory=inventory,
            n_write=self._n_write,
            latency_seconds=latency,
            usage_events=(),
        )

    def ingest_units(
        self,
        units: Sequence[PublicHistoryUnit],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        """Explicit unit-oriented alias useful to preparation code and tests."""
        return self.write_session(
            [],
            session_index=session_index,
            metadata={**(metadata or {}), "public_units": tuple(units)},
        )

    def snapshot_inventory(self, *, checkpoint_session: int) -> InventorySnapshot:
        self._ensure_open()
        if isinstance(checkpoint_session, bool) or not isinstance(checkpoint_session, int):
            raise ValueError("checkpoint_session must be an integer")
        if checkpoint_session < 0:
            raise ValueError("checkpoint_session must be non-negative")
        try:
            count = self._qdrant.count(
                collection_name=self.collection_name,
                namespace=self.namespace,
            )
            raw_points = self._qdrant.scroll(
                collection_name=self.collection_name,
                namespace=self.namespace,
                limit=self.inventory_limit,
            )
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        if count != len(raw_points):
            raise FlatRetrievalError(
                "inventory_count_mismatch",
                f"Qdrant count={count} differs from scroll={len(raw_points)}",
            )
        if count != len(self._objects):
            raise FlatRetrievalError(
                "inventory_count_mismatch",
                f"Qdrant count={count} differs from local inventory={len(self._objects)}",
            )
        items = tuple(self._objects[key] for key in sorted(self._objects))
        payload = [item.to_dict() for item in items]
        store_hash = _canonical_hash(payload)
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=self._n_write,
            n_live=len(items),
            items=items,
            store_hash=store_hash,
            backend_count=count,
        )

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch:
        self._ensure_open()
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        started = time.perf_counter()
        vector = self._embed((query,))[0]
        try:
            raw_hits = self._qdrant.search(
                collection_name=self.collection_name,
                namespace=self.namespace,
                vector=vector,
                limit=self.candidate_k,
            )
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        hits = tuple(self._normalize_hit(raw) for raw in raw_hits)
        try:
            hits = validate_search_order(hits)
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        candidates: list[RetrievalCandidate] = []
        for rank, hit in enumerate(hits[: self.candidate_k], start=1):
            item = self._objects.get(hit.point_id)
            if item is None:
                raise FlatRetrievalError(
                    "candidate_outside_inventory",
                    f"Qdrant returned unknown point ID {hit.point_id!r}",
                )
            content = _payload_content(hit.payload, item)
            if sha256_text(content) != item.content_hash:
                raise FlatRetrievalError(
                    "candidate_inventory_mismatch",
                    f"Qdrant content hash differs for {hit.point_id!r}",
                )
            candidates.append(
                RetrievalCandidate(
                    memory_id=item.memory_id,
                    content=content,
                    content_hash=item.content_hash,
                    native_rank=rank,
                    score=hit.score,
                    score_details=(("semantic", hit.score),),
                    metadata=self._candidate_metadata(item),
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                )
            )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=sha256_text(query),
            candidates=tuple(candidates),
            candidate_shortfall=len(candidates) < self.candidate_k,
            latency_seconds=max(0.0, time.perf_counter() - started),
            usage_events=(),
        )

    def storage_footprints(self) -> tuple[StorageFootprint, ...]:
        return (
            StorageFootprint(
                component="qdrant",
                bytes=None,
                unavailable_reason="Qdrant transport does not expose physical bytes",
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._qdrant, "close", None)
        if callable(close):
            close()
        embedding_close = getattr(self._embedding, "close", None)
        if callable(embedding_close):
            embedding_close()

    def update_memory(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FlatRetrievalError(
            "unsupported_operation", "flat retrieval does not support update_memory"
        )

    def delete_memory(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FlatRetrievalError(
            "unsupported_operation", "flat retrieval does not support delete_memory"
        )

    def delete_namespace(self) -> None:
        """Delete this isolated namespace during explicit task teardown."""
        self._ensure_open()
        try:
            self._qdrant.delete_namespace(
                collection_name=self.collection_name,
                namespace=self.namespace,
            )
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        self._objects.clear()
        self._vectors.clear()
        self._n_write = 0
        self._last_write_session = -1

    def _register_unit(self, unit: PublicHistoryUnit) -> None:
        if not isinstance(unit, PublicHistoryUnit):
            raise TypeError("public_units must contain PublicHistoryUnit records")
        if self.episode_id is not None and unit.episode_id != self.episode_id:
            raise FlatRetrievalError(
                "episode_mismatch",
                f"unit episode {unit.episode_id!r} differs from {self.episode_id!r}",
            )
        self.episode_id = self.episode_id or unit.episode_id
        current = self._units_by_session.setdefault(unit.source_session, ())
        if any(existing.unit_id == unit.unit_id for existing in current):
            return
        if any(
            existing.source_ordinal == unit.source_ordinal
            and existing.source_kind == unit.source_kind
            for existing in current
        ):
            raise FlatRetrievalError(
                "duplicate_public_unit",
                "two public units share session/kind/ordinal",
            )
        self._units_by_session[unit.source_session] = tuple(
            sorted(
                (*current, unit),
                key=lambda item: (
                    item.source_ordinal,
                    0 if item.source_kind == "observation" else 1,
                    item.unit_id,
                ),
            )
        )

    def _session_units(
        self,
        messages: Sequence[Mapping[str, str]],
        session_index: int,
        metadata: Mapping[str, object] | None,
    ) -> tuple[PublicHistoryUnit, ...]:
        if metadata:
            for key in ("public_units", "history_units", "units"):
                if key in metadata:
                    raw = metadata[key]
                    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                        raise FlatRetrievalError(
                            "invalid_public_units",
                            f"metadata[{key}] must be an array",
                        )
                    selected_units = tuple(self._coerce_unit(item) for item in raw)
                    for unit in selected_units:
                        if unit.source_session != session_index:
                            raise FlatRetrievalError(
                                "session_mismatch",
                                f"unit {unit.unit_id!r} is not from session {session_index}",
                            )
                        self._register_unit(unit)
                    return self._sort_units(selected_units)
        selected_by_session: tuple[PublicHistoryUnit, ...] | None = self._units_by_session.get(
            session_index
        )
        if selected_by_session:
            return selected_by_session
        parsed = self._parse_messages(messages, session_index, metadata)
        for unit in parsed:
            self._register_unit(unit)
        return self._sort_units(parsed)

    def _parse_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        session_index: int,
        metadata: Mapping[str, object] | None,
    ) -> tuple[PublicHistoryUnit, ...]:
        episode_id = self.episode_id
        if metadata is not None and isinstance(metadata.get("episode_id"), str):
            episode_id = cast(str, metadata["episode_id"])
        payloads: list[Mapping[str, object]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise FlatRetrievalError("invalid_public_messages", "messages must contain objects")
            content = message.get("content")
            if not isinstance(content, str):
                raise FlatRetrievalError("invalid_public_messages", "message content must be text")
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                # A non-JSON public message remains a single observation.
                if episode_id is None:
                    raise FlatRetrievalError(
                        "missing_episode_id",
                        "episode_id is required when parsing unstructured messages",
                    ) from None
                return (
                    PublicHistoryUnit.create(
                        episode_id=episode_id,
                        source_session=session_index,
                        source_kind="observation",
                        source_ordinal=0,
                        content=content,
                    ),
                )
            if not isinstance(value, Mapping):
                raise FlatRetrievalError("invalid_public_messages", "transcript must be an object")
            payloads.append(value)
        if not payloads:
            return ()
        payload = payloads[0]
        recorded = payload.get("session_index")
        if isinstance(recorded, bool) or recorded != session_index:
            raise FlatRetrievalError("session_mismatch", "transcript session_index differs")
        if episode_id is None:
            raise FlatRetrievalError("missing_episode_id", "episode_id is required")
        units: list[PublicHistoryUnit] = []
        for kind, key in (("observation", "observations"), ("tool_result", "tool_results")):
            values = payload.get(key, ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise FlatRetrievalError("invalid_public_messages", f"{key} must be an array")
            if any(not isinstance(value, str) for value in values):
                raise FlatRetrievalError("invalid_public_messages", f"{key} must contain strings")
            public_kind = cast(PublicHistoryKind, kind)
            units.extend(
                PublicHistoryUnit.create(
                    episode_id=episode_id,
                    source_session=session_index,
                    source_kind=public_kind,
                    source_ordinal=ordinal,
                    content=cast(str, value),
                )
                for ordinal, value in enumerate(values)
            )
        return tuple(units)

    def _coerce_unit(self, value: object) -> PublicHistoryUnit:
        if isinstance(value, PublicHistoryUnit):
            return value
        if not isinstance(value, Mapping):
            raise FlatRetrievalError("invalid_public_units", "unit must be an object")
        kind = value.get("source_kind")
        if kind not in {"observation", "tool_result"}:
            raise FlatRetrievalError("invalid_public_units", "unknown public unit kind")
        try:
            return PublicHistoryUnit(
                unit_id=str(value["unit_id"]),
                episode_id=str(value["episode_id"]),
                source_session=int(value["source_session"]),
                source_kind=cast(PublicHistoryKind, kind),
                source_ordinal=int(value["source_ordinal"]),
                content=str(value["content"]),
                content_sha256=str(value["content_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FlatRetrievalError("invalid_public_units", "malformed public unit") from exc

    @staticmethod
    def _sort_units(units: Sequence[PublicHistoryUnit]) -> tuple[PublicHistoryUnit, ...]:
        return tuple(
            sorted(
                units,
                key=lambda item: (
                    item.source_ordinal,
                    0 if item.source_kind == "observation" else 1,
                    item.unit_id,
                ),
            )
        )

    def _embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        try:
            method = getattr(self._embedding, "embed", None)
            if not callable(method):
                method = getattr(self._embedding, "embed_batch", None)
            if not callable(method):
                raise FlatRetrievalError(
                    "embedding_contract",
                    "embedding runtime must expose embed or embed_batch",
                )
            raw = method(texts)
        except FlatRetrievalError:
            raise
        except Exception as exc:
            raise FlatRetrievalError("embedding_failure", str(exc)) from exc
        raw_vectors = getattr(raw, "vectors", raw)
        if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes)):
            raise FlatRetrievalError(
                "embedding_contract",
                "embedding result vectors must be an array",
            )
        if len(raw_vectors) != len(texts):
            raise FlatRetrievalError(
                "embedding_contract",
                f"embedding result count {len(raw_vectors)} != input count {len(texts)}",
            )
        vectors: list[tuple[float, ...]] = []
        for index, raw_vector in enumerate(raw_vectors):
            if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
                raise FlatRetrievalError(
                    "embedding_contract",
                    f"embedding vector {index} is not an array",
                )
            try:
                vectors.append(
                    validate_vector(
                        cast(Iterable[object], raw_vector),
                        dimension=self._embedding_dimension,
                        field=f"embedding vector {index}",
                    )
                )
            except QdrantError as exc:
                raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        return tuple(vectors)

    def _resolve_dimension(self, explicit: int | None, runtime: object) -> int:
        candidate = explicit
        if candidate is None:
            for name in ("dimension", "expected_dimension", "embedding_dimension"):
                value = getattr(runtime, name, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    candidate = value
                    break
        if candidate is None or candidate < 1:
            raise ValueError(
                "embedding_dimension is required (or runtime must expose dimension)"
            )
        return candidate

    def _memory_object(self, unit: PublicHistoryUnit) -> MemoryObject:
        metadata = self._candidate_metadata(unit)
        return MemoryObject(
            memory_id=unit.unit_id,
            content=unit.content,
            content_hash=unit.content_sha256,
            metadata=metadata,
            created_at=f"flat-session-{unit.source_session:06d}-{unit.source_ordinal:06d}",
            updated_at=f"flat-session-{unit.source_session:06d}-{unit.source_ordinal:06d}",
            history_length=1,
        )

    @staticmethod
    def _candidate_metadata(
        item: MemoryObject | PublicHistoryUnit,
    ) -> tuple[tuple[str, object], ...]:
        if isinstance(item, MemoryObject):
            value = dict(item.metadata)
            return tuple((str(key), child) for key, child in value.items())
        return (
            ("episode_id", item.episode_id),
            ("session_index", item.source_session),
            ("source_session", item.source_session),
            ("kind", item.source_kind),
            ("source_kind", item.source_kind),
            ("ordinal", item.source_ordinal),
            ("source_ordinal", item.source_ordinal),
            ("content_sha256", item.content_sha256),
            (
                "lhmsb.provenance",
                {
                    "episode_id": item.episode_id,
                    "source_session": item.source_session,
                    "source_kind": item.source_kind,
                    "source_ordinal": item.source_ordinal,
                },
            ),
            ("lhmsb.candidate_origin", "native"),
            ("lhmsb.score_semantics", "higher_is_better"),
        )

    def _point_payload(self, unit: PublicHistoryUnit) -> dict[str, object]:
        payload = {
            "_namespace": self.namespace,
            "memory_id": unit.unit_id,
            "content": unit.content,
            "content_sha256": unit.content_sha256,
            "episode_id": unit.episode_id,
            "source_session": unit.source_session,
            "source_kind": unit.source_kind,
            "source_ordinal": unit.source_ordinal,
        }
        return payload

    def _operation_id(self, unit: PublicHistoryUnit) -> str:
        return (
            f"flat-observed-add-{unit.source_session:06d}-"
            f"{unit.source_kind}-{unit.source_ordinal:06d}-{unit.unit_id}"
        )

    def _normalize_hit(self, raw: QdrantHit | Mapping[str, object]) -> QdrantHit:
        if isinstance(raw, QdrantHit):
            return raw
        if not isinstance(raw, Mapping):
            raise FlatRetrievalError("invalid_search_hit", "Qdrant hit must be an object")
        point_id = raw.get("id", raw.get("point_id"))
        score = raw.get("score")
        payload = raw.get("payload", {})
        try:
            point_id = validate_point_id(point_id)
        except QdrantError as exc:
            raise FlatRetrievalError(exc.error_class, str(exc)) from exc
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
        ):
            raise FlatRetrievalError("invalid_score", "Qdrant score must be finite")
        if not isinstance(payload, Mapping):
            raise FlatRetrievalError("invalid_payload", "Qdrant hit payload must be an object")
        return QdrantHit(point_id, float(score), payload)

    @staticmethod
    def _point_payload_content(payload: Mapping[str, object]) -> str | None:
        value = payload.get("content")
        return value if isinstance(value, str) else None

    def _ensure_open(self) -> None:
        if self._closed:
            raise FlatRetrievalError("transport_closed", "flat retrieval adapter is closed")


def _payload_content(payload: Mapping[str, object], item: MemoryObject) -> str:
    content = payload.get("content")
    if content is None:
        return item.content
    if not isinstance(content, str):
        raise FlatRetrievalError("invalid_payload", "Qdrant content must be text")
    return content


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DeterministicHashEmbedding",
    "EmbeddingRuntime",
    "FlatRetrievalAdapter",
    "FlatRetrievalError",
]


assert isinstance(FlatRetrievalAdapter.capabilities, LifecycleCapabilities)
