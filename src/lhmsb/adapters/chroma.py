"""ChromaDB plain-vector baseline adapter (``spec/05-systems.md`` §2.1 ``chroma``).

A faithful vector-store baseline that runs FULLY OFFLINE (the plan's hard
"no network during an episode" guardrail). Two deliberate design choices make
that possible and keep cost accounting honest:

1. **No ChromaDB embedding function.** Chroma's default embedder downloads an
   ONNX model on first use (network). Instead this adapter computes embeddings
   itself with a deterministic, in-process hashing bag-of-words embedder and
   passes ``embeddings=`` / ``query_embeddings=`` explicitly. This is the same
   Chroma query surface, minus the download, and lets embedding tokens be
   counted exactly under ``memory_scope`` (task 6) rather than hidden inside
   Chroma.
2. **chromadb is imported lazily** via ``importlib`` so importing this module
   never requires the optional ``chroma`` extra; only instantiating
   :class:`ChromaAdapter` does.

Behavior is scored, not implementation (``spec/05-systems.md`` §1.1): ``add`` ->
``collection.add``, ``search`` -> ``collection.query``, ``update`` ->
``collection.upsert``, ``delete`` -> ``collection.delete``, ``reset`` drops the
user's collection. The adapter has no internal LLM, so its native and controlled
tracks coincide.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from hashlib import sha256
from math import sqrt
from time import perf_counter
from types import ModuleType
from typing import Any

from lhmsb.adapters.base import MemorySystemAdapter
from lhmsb.cost import CostMeter, count_tokens
from lhmsb.types import MemoryEntry, SearchResult

_DEFAULT_DIM = 256


def _load_chromadb() -> ModuleType:
    try:
        return importlib.import_module("chromadb")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ChromaAdapter requires the optional 'chroma' extra: pip install 'lhmsb[chroma]'"
        ) from exc


def _embed(text: str, dim: int) -> list[float]:
    """Deterministic, offline, L2-normalized hashing bag-of-words embedding.

    Each token is hashed to a dimension with a sign; the vector is L2-normalized
    so Chroma's L2 distance ranks token-overlapping text nearest. No model, no
    network, reproducible across processes (SHA-256, not salted ``hash()``).
    """
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = int(sha256(token.encode("utf-8")).hexdigest(), 16)
        vector[digest % dim] += 1.0 if (digest >> 8) % 2 == 0 else -1.0
    norm = sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        return [value / norm for value in vector]
    return vector


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _nested_first(raw: object, key: str) -> list[object]:
    """Extract ``raw[key][0]`` from a Chroma ``query`` result (lists-per-query)."""
    if isinstance(raw, dict):
        outer = raw.get(key)
        if isinstance(outer, list) and outer and isinstance(outer[0], list):
            return list(outer[0])
    return []


def _flat_list(raw: object, key: str) -> list[object]:
    """Extract ``raw[key]`` from a Chroma ``get`` result (flat lists)."""
    if isinstance(raw, dict):
        value = raw.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _meta_str(raw: object, key: str) -> str:
    if isinstance(raw, dict):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return ""


def _decode_user_metadata(raw: object) -> dict[str, object] | None:
    blob = _meta_str(raw, "_meta")
    if not blob:
        return None
    loaded = json.loads(blob)
    if isinstance(loaded, dict):
        return {str(key): value for key, value in loaded.items()}
    return None


class ChromaAdapter(MemorySystemAdapter):
    """Plain-vector memory baseline backed by an embedded, offline ChromaDB."""

    def __init__(
        self, *, meter: CostMeter | None = None, embedding_dim: int = _DEFAULT_DIM
    ) -> None:
        chromadb = _load_chromadb()
        settings_cls = importlib.import_module("chromadb.config").Settings
        self._client: Any = chromadb.EphemeralClient(
            settings=settings_cls(anonymized_telemetry=False, allow_reset=True)
        )
        self._meter = CostMeter() if meter is None else meter
        self._dim = embedding_dim
        self._collections: dict[str, Any] = {}
        self._counter = 0

    @property
    def cost_meter(self) -> CostMeter:
        """The cost meter accumulating this adapter's embedding/storage/latency."""
        return self._meter

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        self._collection(user_id)

    def reset(self, *, user_id: str) -> None:
        name = _collection_name(user_id)
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=name)
        self._collections[name] = self._client.get_or_create_collection(name=name)

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._counter += 1
        memory_id = f"chroma-{self._counter}"
        timestamp = self._timestamp()
        encoded = _encode_metadata(
            session_id=session_id, metadata=metadata, created_at=timestamp, updated_at=timestamp
        )
        collection = self._collection(user_id)
        started = perf_counter()
        with self._meter.memory_scope():
            self._meter.record_embedding_call(count_tokens(content, self._meter.model), 1)
            collection.add(
                ids=[memory_id],
                embeddings=[_embed(content, self._dim)],
                documents=[content],
                metadatas=[encoded],
            )
        self._meter.record_latency("write", (perf_counter() - started) * 1000.0)
        self._meter.add_storage_bytes(len(content.encode("utf-8")))
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
        collection = self._collection(user_id)
        count = _as_int(collection.count())
        self._meter.incr_retrieval()
        if count == 0:
            return SearchResult(results=[], total_count=0)

        where_raw = filters.get("where")
        where = where_raw if isinstance(where_raw, dict) else None
        n_results = max(1, min(top_k, count))
        started = perf_counter()
        with self._meter.memory_scope():
            self._meter.record_embedding_call(count_tokens(query, self._meter.model), 1)
            raw = collection.query(
                query_embeddings=[_embed(query, self._dim)], n_results=n_results, where=where
            )
        self._meter.record_latency("retrieval", (perf_counter() - started) * 1000.0)

        entries = _to_entries(raw)
        return SearchResult(results=entries[:top_k], total_count=count)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        for collection in self._collections.values():
            existing = collection.get(ids=[memory_id])
            if memory_id not in {str(found) for found in _flat_list(existing, "ids")}:
                continue
            documents = _flat_list(existing, "documents")
            metadatas = _flat_list(existing, "metadatas")
            old_content = str(documents[0]) if documents else ""
            old_meta = metadatas[0] if metadatas else None
            new_content = old_content if content is None else content
            new_meta = _decode_user_metadata(old_meta) if metadata is None else metadata
            encoded = _encode_metadata(
                session_id=_meta_str(old_meta, "_session_id"),
                metadata=new_meta,
                created_at=_meta_str(old_meta, "_created_at") or self._timestamp(),
                updated_at=self._timestamp(),
            )
            started = perf_counter()
            with self._meter.memory_scope():
                self._meter.record_embedding_call(count_tokens(new_content, self._meter.model), 1)
                collection.upsert(
                    ids=[memory_id],
                    embeddings=[_embed(new_content, self._dim)],
                    documents=[new_content],
                    metadatas=[encoded],
                )
            self._meter.record_latency("update", (perf_counter() - started) * 1000.0)
            return

    def delete_memory(self, memory_id: str) -> None:
        for collection in self._collections.values():
            with contextlib.suppress(Exception):
                collection.delete(ids=[memory_id])

    def _collection(self, user_id: str) -> Any:
        name = _collection_name(user_id)
        collection = self._collections.get(name)
        if collection is None:
            collection = self._client.get_or_create_collection(name=name)
            self._collections[name] = collection
        return collection

    def _timestamp(self) -> str:
        return f"2025-01-01T00:00:00.{self._counter:06d}Z"


def _collection_name(user_id: str) -> str:
    return "lhmsb-" + sha256(user_id.encode("utf-8")).hexdigest()[:24]


def _encode_metadata(
    *,
    session_id: str | None,
    metadata: dict[str, object] | None,
    created_at: str,
    updated_at: str,
) -> dict[str, str]:
    """Serialize entry metadata into Chroma's scalar-only metadata schema.

    Chroma metadata values must be scalars, so arbitrary (possibly nested) user
    metadata is JSON-encoded into a single ``_meta`` string alongside the
    timestamps and session id.
    """
    return {
        "_created_at": created_at,
        "_updated_at": updated_at,
        "_session_id": session_id or "",
        "_meta": json.dumps(metadata) if metadata is not None else "",
    }


def _to_entries(raw: object) -> list[MemoryEntry]:
    ids = _nested_first(raw, "ids")
    documents = _nested_first(raw, "documents")
    metadatas = _nested_first(raw, "metadatas")
    distances = _nested_first(raw, "distances")
    entries: list[MemoryEntry] = []
    for index, raw_id in enumerate(ids):
        meta_raw = metadatas[index] if index < len(metadatas) else None
        distance = _as_float(distances[index]) if index < len(distances) else 0.0
        entries.append(
            MemoryEntry(
                memory_id=str(raw_id),
                content=str(documents[index]) if index < len(documents) else "",
                metadata=_decode_user_metadata(meta_raw),
                created_at=_meta_str(meta_raw, "_created_at"),
                updated_at=_meta_str(meta_raw, "_updated_at"),
                score=1.0 / (1.0 + distance),
            )
        )
    return entries
