"""Clients for pinned local Text Embeddings Inference services."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import httpx


class TeiServiceError(RuntimeError):
    """Typed local embedding/reranking service failure."""

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
class ServiceHealth:
    ok: bool
    status_code: int | None
    latency_seconds: float


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    revision: str
    dimension: int
    input_count: int
    request_hash: str
    response_hash: str
    latency_seconds: float


@dataclass(frozen=True)
class RerankCandidate:
    memory_id: str
    text: str
    native_rank: int


@dataclass(frozen=True)
class RerankResult:
    ordered_memory_ids: tuple[str, ...]
    scores: tuple[float, ...]
    model: str
    revision: str
    input_count: int
    request_hash: str
    response_hash: str
    latency_seconds: float


class _TeiClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None,
        timeout_seconds: float,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=timeout_seconds,
        )

    def health(self) -> ServiceHealth:
        started = time.perf_counter()
        try:
            response = self._client.get("/health")
        except httpx.HTTPError:
            return ServiceHealth(False, None, max(0.0, time.perf_counter() - started))
        return ServiceHealth(
            response.status_code < 400,
            response.status_code,
            max(0.0, time.perf_counter() - started),
        )

    def close(self) -> None:
        self._client.close()


class EmbeddingClient(_TeiClient):
    """OpenAI-compatible TEI embedding client with a strict dimension gate."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        revision: str,
        expected_dimension: int,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            base_url,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        if expected_dimension < 1:
            raise ValueError("expected_dimension must be positive")
        self.model = model
        self.revision = revision
        self.expected_dimension = expected_dimension

    def embed(self, texts: tuple[str, ...]) -> EmbeddingBatch:
        if not texts:
            raise ValueError("embedding input must be non-empty")
        body: dict[str, object] = {"input": list(texts), "model": self.model}
        started = time.perf_counter()
        raw = self._post_json("/v1/embeddings", body, "embedding_failure")
        data = raw.get("data")
        if not isinstance(data, list):
            raise TeiServiceError("embedding_failure", "embedding response lacks data")
        vectors_by_index: dict[int, tuple[float, ...]] = {}
        for row in data:
            if not isinstance(row, dict):
                raise TeiServiceError("embedding_failure", "embedding row must be an object")
            index = row.get("index")
            embedding = row.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not isinstance(embedding, list)
            ):
                raise TeiServiceError("embedding_failure", "malformed embedding row")
            vector: list[float] = []
            for value in embedding:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TeiServiceError(
                        "embedding_failure",
                        "embedding vector contains a non-number",
                    )
                vector.append(float(value))
            if len(vector) != self.expected_dimension:
                raise TeiServiceError(
                    "embedding_failure",
                    f"expected dimension {self.expected_dimension}, got {len(vector)}",
                )
            vectors_by_index[index] = tuple(vector)
        expected_indices = set(range(len(texts)))
        if set(vectors_by_index) != expected_indices:
            raise TeiServiceError(
                "embedding_failure",
                "embedding response indices do not match the input batch",
            )
        return EmbeddingBatch(
            vectors=tuple(vectors_by_index[index] for index in range(len(texts))),
            model=self.model,
            revision=self.revision,
            dimension=self.expected_dimension,
            input_count=len(texts),
            request_hash=_canonical_hash(body),
            response_hash=_canonical_hash(raw),
            latency_seconds=max(0.0, time.perf_counter() - started),
        )

    def _post_json(
        self,
        path: str,
        body: dict[str, object],
        error_class: str,
    ) -> dict[str, object]:
        try:
            response = self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise TeiServiceError(error_class, str(exc)) from exc
        if response.status_code >= 400:
            raise TeiServiceError(
                error_class,
                response.text,
                status_code=response.status_code,
            )
        try:
            raw = response.json()
        except ValueError as exc:
            raise TeiServiceError(error_class, "TEI response is not JSON") from exc
        if not isinstance(raw, dict):
            raise TeiServiceError(error_class, "TEI response must be an object")
        return {str(key): value for key, value in raw.items()}


class RerankerClient(_TeiClient):
    """Benchmark-owned common reranker over a frozen native candidate set."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        revision: str,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            base_url,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        self.model = model
        self.revision = revision

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(
                ordered_memory_ids=(),
                scores=(),
                model=self.model,
                revision=self.revision,
                input_count=0,
                request_hash=_canonical_hash(
                    {"query": query, "texts": [], "model": self.model}
                ),
                response_hash=_canonical_hash([]),
                latency_seconds=0.0,
            )
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or null")
        body = {
            "query": query,
            "texts": [candidate.text for candidate in candidates],
            "model": self.model,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/rerank", json=body)
        except httpx.HTTPError as exc:
            raise TeiServiceError("reranker_failure", str(exc)) from exc
        if response.status_code >= 400:
            raise TeiServiceError(
                "reranker_failure",
                response.text,
                status_code=response.status_code,
            )
        try:
            raw: object = response.json()
        except ValueError as exc:
            raise TeiServiceError("reranker_failure", "reranker response is not JSON") from exc
        rows = raw.get("results") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise TeiServiceError("reranker_failure", "reranker response must be an array")
        scored: list[tuple[RerankCandidate, float]] = []
        seen_indices: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise TeiServiceError("reranker_failure", "reranker row must be an object")
            index = row.get("index")
            score = row.get("score", row.get("relevance_score"))
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(candidates)
                or index in seen_indices
            ):
                raise TeiServiceError("reranker_failure", f"invalid reranker index: {index!r}")
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise TeiServiceError("reranker_failure", "reranker score must be numeric")
            seen_indices.add(index)
            scored.append((candidates[index], float(score)))
        scored.sort(key=lambda item: (-item[1], item[0].native_rank))
        selected = scored if top_k is None else scored[:top_k]
        return RerankResult(
            ordered_memory_ids=tuple(item[0].memory_id for item in selected),
            scores=tuple(item[1] for item in selected),
            model=self.model,
            revision=self.revision,
            input_count=len(candidates),
            request_hash=_canonical_hash(body),
            response_hash=_canonical_hash(raw),
            latency_seconds=max(0.0, time.perf_counter() - started),
        )


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
    "EmbeddingBatch",
    "EmbeddingClient",
    "RerankCandidate",
    "RerankResult",
    "RerankerClient",
    "ServiceHealth",
    "TeiServiceError",
]
