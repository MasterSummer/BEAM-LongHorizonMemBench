"""Trace-complete adapter for the pinned Mem0 OSS 2.0.12 lifecycle."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast
from urllib.parse import urlparse

from lhmsb.qualification.schema import Mem0Profile, PolicyProfile


class Mem0QualificationError(RuntimeError):
    """Typed Mem0 lifecycle failure."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class Mem0V2Backend(Protocol):
    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        run_id: str,
        metadata: dict[str, object] | None,
        infer: bool,
    ) -> object: ...

    def search(
        self,
        query: str,
        *,
        filters: dict[str, object],
        top_k: int,
        threshold: float,
        rerank: bool,
    ) -> object: ...

    def get_all(
        self,
        *,
        filters: dict[str, object],
        top_k: int,
    ) -> object: ...

    def history(self, memory_id: str) -> object: ...


class _ProviderCreateBoundary(Protocol):
    create: Callable[..., object]


@dataclass(frozen=True)
class NativeMemoryEvent:
    operation_id: str
    session_index: int
    native_event: str
    memory_id: str
    memory_text: str
    old_content_hash: str | None
    new_content_hash: str | None
    source: str
    latency_seconds: float


@dataclass(frozen=True)
class InventoryItem:
    memory_id: str
    content: str
    content_hash: str
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str
    history_length: int


@dataclass(frozen=True)
class InventorySnapshot:
    checkpoint_session: int
    n_write: int
    n_live: int
    items: tuple[InventoryItem, ...]
    store_hash: str
    backend_count: int | None


@dataclass(frozen=True)
class SearchCandidate:
    memory_id: str
    content: str
    content_hash: str
    native_rank: int
    score: float | None
    score_details: tuple[tuple[str, float], ...]
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CandidateSearch:
    checkpoint_session: int
    query: str
    query_hash: str
    candidates: tuple[SearchCandidate, ...]
    candidate_shortfall: bool
    latency_seconds: float
    usage_events: tuple[ProviderUsageEvent, ...] = ()


@dataclass(frozen=True)
class WriteSessionResult:
    session_index: int
    events: tuple[NativeMemoryEvent, ...]
    inventory: InventorySnapshot
    n_write: int
    latency_seconds: float
    usage_events: tuple[ProviderUsageEvent, ...] = ()


@dataclass(frozen=True)
class ProviderUsageEvent:
    """One raw Mem0-internal provider or embedding call."""

    call_id: str
    component: str
    provider: str
    model_id: str
    endpoint_identity: str
    request_hash: str
    response_hash: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    usage_observed: bool
    input_count: int
    latency_seconds: float
    retry_count: int | None
    error_class: str | None
    started_at_utc: str
    ended_at_utc: str


class Mem0QualificationAdapter:
    """Synchronous Mem0 adapter that preserves native objects and events."""

    def __init__(
        self,
        backend: Mem0V2Backend,
        *,
        user_id: str,
        run_id: str,
        candidate_k: int = 20,
        inventory_limit: int = 10000,
        collection_count: Callable[[], int] | None = None,
        usage_events: list[ProviderUsageEvent] | None = None,
    ) -> None:
        if not user_id or not run_id:
            raise ValueError("user_id and run_id must be non-empty")
        if candidate_k < 1 or inventory_limit < 1:
            raise ValueError("candidate_k and inventory_limit must be positive")
        self.backend = backend
        self.user_id = user_id
        self.run_id = run_id
        self.candidate_k = candidate_k
        self.inventory_limit = inventory_limit
        self.collection_count = collection_count
        self._usage_events = usage_events if usage_events is not None else []
        self._n_write = 0
        self._closed = False

    @property
    def filters(self) -> dict[str, object]:
        return {"user_id": self.user_id, "run_id": self.run_id}

    def restore_write_count(self, n_write: int) -> None:
        """Restore the cumulative native mutation count when resuming a task."""
        if n_write < 0:
            raise ValueError("n_write must be non-negative")
        self._n_write = n_write

    def close(self) -> None:
        """Release Mem0 history, provider, embedding, and vector clients."""
        if self._closed:
            return
        self._closed = True
        targets = [
            self.backend,
            _nested_attribute(self.backend, "llm", "client"),
            _nested_attribute(self.backend, "embedding_model", "client"),
            _nested_attribute(self.backend, "vector_store", "client"),
        ]
        errors: list[str] = []
        seen: set[int] = set()
        for target in targets:
            if target is None or id(target) in seen:
                continue
            seen.add(id(target))
            close = getattr(target, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive SDK cleanup
                errors.append(f"{type(target).__name__}: {exc}")
        if errors:
            raise Mem0QualificationError(
                "resource_cleanup_failure",
                "; ".join(errors),
            )

    @classmethod
    def create_live(
        cls,
        config: dict[str, object],
        *,
        user_id: str,
        run_id: str,
        candidate_k: int = 20,
        inventory_limit: int = 10000,
        collection_count: Callable[[], int] | None = None,
    ) -> Mem0QualificationAdapter:
        """Lazily import Mem0 after disabling product telemetry."""
        _disable_mem0_telemetry()
        module = _load_mem0()
        memory_class = module.Memory
        backend = memory_class.from_config(config)
        usage_events: list[ProviderUsageEvent] = []
        _install_usage_instrumentation(
            backend,
            config,
            usage_events,
            call_prefix=f"{user_id}:{run_id}",
        )
        return cls(
            backend,
            user_id=user_id,
            run_id=run_id,
            candidate_k=candidate_k,
            inventory_limit=inventory_limit,
            collection_count=collection_count,
            usage_events=usage_events,
        )

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        """Run one native extraction write and capture its resulting inventory."""
        if session_index < 0:
            raise ValueError("session_index must be non-negative")
        before = self.snapshot_inventory(
            checkpoint_session=session_index,
            include_history=False,
        )
        merged = dict(metadata or {})
        merged["session_index"] = session_index
        usage_offset = len(self._usage_events)
        started = time.perf_counter()
        try:
            raw = self.backend.add(
                messages,
                user_id=self.user_id,
                run_id=self.run_id,
                metadata=merged,
                infer=True,
            )
        except Exception as exc:
            raise Mem0QualificationError("mem0_write_failure", str(exc)) from exc
        latency = max(0.0, time.perf_counter() - started)
        events = list(_native_events(raw, session_index=session_index, latency=latency))
        counted_ids = {
            event.memory_id
            for event in events
            if event.native_event in {"ADD", "UPDATE", "DELETE"}
        }
        self._n_write += sum(
            event.native_event in {"ADD", "UPDATE", "DELETE"} for event in events
        )
        after = self.snapshot_inventory(checkpoint_session=session_index)
        before_ids = {item.memory_id for item in before.items}
        for index, item in enumerate(after.items):
            if item.memory_id not in before_ids and item.memory_id not in counted_ids:
                events.append(
                    NativeMemoryEvent(
                        operation_id=f"session-{session_index:03d}-delta-{index:03d}",
                        session_index=session_index,
                        native_event="OBSERVED_ADD",
                        memory_id=item.memory_id,
                        memory_text=item.content,
                        old_content_hash=None,
                        new_content_hash=item.content_hash,
                        source="inventory_delta",
                        latency_seconds=0.0,
                    )
                )
                self._n_write += 1
        if after.n_write != self._n_write:
            after = InventorySnapshot(
                checkpoint_session=after.checkpoint_session,
                n_write=self._n_write,
                n_live=after.n_live,
                items=after.items,
                store_hash=after.store_hash,
                backend_count=after.backend_count,
            )
        return WriteSessionResult(
            session_index=session_index,
            events=tuple(events),
            inventory=after,
            n_write=self._n_write,
            latency_seconds=latency,
            usage_events=tuple(self._usage_events[usage_offset:]),
        )

    def snapshot_inventory(
        self,
        *,
        checkpoint_session: int,
        include_history: bool = True,
    ) -> InventorySnapshot:
        try:
            raw = self.backend.get_all(
                filters=self.filters,
                top_k=self.inventory_limit,
            )
        except Exception as exc:
            raise Mem0QualificationError("inventory_failure", str(exc)) from exc
        rows = _result_rows(raw)
        if len(rows) >= self.inventory_limit:
            raise Mem0QualificationError(
                "inventory_failure",
                f"inventory reached or exceeded limit {self.inventory_limit}",
            )
        items: list[InventoryItem] = []
        for row in rows:
            memory_id = _text(row.get("id"))
            content = _memory_text(row)
            history_length = (
                len(self._history(memory_id)) if include_history and memory_id else 0
            )
            items.append(
                InventoryItem(
                    memory_id=memory_id,
                    content=content,
                    content_hash=_text_hash(content),
                    metadata=_metadata_pairs(row.get("metadata")),
                    created_at=_text(row.get("created_at")),
                    updated_at=_text(row.get("updated_at")),
                    history_length=history_length,
                )
            )
        items.sort(key=lambda item: item.memory_id)
        backend_count = self.collection_count() if self.collection_count is not None else None
        if backend_count is not None and backend_count != len(items):
            raise Mem0QualificationError(
                "inventory_failure",
                f"Mem0 inventory count {len(items)} != Qdrant count {backend_count}",
            )
        payload = [
            {
                "memory_id": item.memory_id,
                "content_hash": item.content_hash,
                "metadata": item.metadata,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "history_length": item.history_length,
            }
            for item in items
        ]
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=self._n_write,
            n_live=len(items),
            items=tuple(items),
            store_hash=_canonical_hash(payload),
            backend_count=backend_count,
        )

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch:
        usage_offset = len(self._usage_events)
        started = time.perf_counter()
        try:
            raw = self.backend.search(
                query,
                filters=self.filters,
                top_k=self.candidate_k,
                threshold=0.0,
                rerank=False,
            )
        except Exception as exc:
            raise Mem0QualificationError("mem0_search_failure", str(exc)) from exc
        rows = _result_rows(raw)
        candidates: list[SearchCandidate] = []
        for rank, row in enumerate(rows, start=1):
            content = _memory_text(row)
            candidates.append(
                SearchCandidate(
                    memory_id=_text(row.get("id")),
                    content=content,
                    content_hash=_text_hash(content),
                    native_rank=rank,
                    score=_optional_float(row.get("score")),
                    score_details=_score_pairs(row.get("score_details")),
                    metadata=_metadata_pairs(row.get("metadata")),
                    created_at=_text(row.get("created_at")),
                    updated_at=_text(row.get("updated_at")),
                )
            )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=_text_hash(query),
            candidates=tuple(candidates),
            candidate_shortfall=len(candidates) < self.candidate_k,
            latency_seconds=max(0.0, time.perf_counter() - started),
            usage_events=tuple(self._usage_events[usage_offset:]),
        )

    def history_delta(
        self,
        memory_id: str,
        *,
        previous_length: int,
    ) -> tuple[dict[str, object], ...]:
        if previous_length < 0:
            raise ValueError("previous_length must be non-negative")
        return tuple(self._history(memory_id)[previous_length:])

    def _history(self, memory_id: str) -> list[dict[str, object]]:
        try:
            raw = self.backend.history(memory_id)
        except Exception as exc:
            raise Mem0QualificationError("inventory_failure", str(exc)) from exc
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise Mem0QualificationError("inventory_failure", "history must be an array")
        output: list[dict[str, object]] = []
        for row in raw:
            if isinstance(row, Mapping):
                output.append({str(key): value for key, value in row.items()})
        return output


def build_mem0_live_config(
    profile: Mem0Profile,
    *,
    policy: PolicyProfile,
    internal_llm_api_key: str,
    native_openai_api_key: str,
    qdrant_url: str,
    collection_name: str,
    history_db_path: Path,
    embedding_base_url: str,
    embedding_dimension: int,
    native_openai_base_url: str = "https://api.openai.com/v1",
    response_callback: Callable[[object, dict[str, object], dict[str, object]], None]
    | None = None,
) -> dict[str, object]:
    """Build the explicit Mem0 2.0.12 config for one isolated task."""
    qdrant = urlparse(qdrant_url)
    if not qdrant.hostname or qdrant.port is None:
        raise ValueError("qdrant_url must include a host and port")
    if profile.track == "controlled":
        if profile.embedding_provider != "openai_compatible_tei":
            raise ValueError(
                "controlled Mem0 embedding_provider must be "
                "openai_compatible_tei"
            )
        llm_config: dict[str, object] = {
            "model": policy.model_id,
            "api_key": internal_llm_api_key,
            f"{policy.provider}_base_url": (
                _openai_sdk_base_url(policy.endpoint)
                if policy.provider == "openai"
                else policy.endpoint
            ),
        }
        if response_callback is not None and policy.provider == "openai":
            llm_config["response_callback"] = response_callback
        llm = {"provider": policy.provider, "config": llm_config}
        embedder = {
            "provider": "openai",
            "config": {
                "model": profile.embedding_model,
                "api_key": "local-tei",
                "openai_base_url": f"{embedding_base_url.rstrip('/')}/v1",
                "embedding_dims": embedding_dimension,
            },
        }
        vector_dimension = embedding_dimension
    else:
        llm = {
            "provider": "openai",
            "config": {
                "model": profile.internal_llm_model,
                "api_key": native_openai_api_key,
                "openai_base_url": _openai_sdk_base_url(
                    native_openai_base_url
                ),
                "is_reasoning_model": True,
            },
        }
        vector_dimension = 1536
        embedder = {
            "provider": "openai",
            "config": {
                "model": profile.embedding_model,
                "api_key": native_openai_api_key,
                "openai_base_url": _openai_sdk_base_url(
                    native_openai_base_url
                ),
                "embedding_dims": vector_dimension,
            },
        }
    return {
        "llm": llm,
        "embedder": embedder,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": vector_dimension,
                "host": qdrant.hostname,
                "port": qdrant.port,
                "path": None,
                "https": qdrant.scheme == "https",
                "on_disk": True,
            },
        },
        "history_db_path": str(history_db_path),
        "version": "v1.1",
    }


def _openai_sdk_base_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("OpenAI base URL must be absolute")
    if parsed.path.rstrip("/") == "":
        parsed = parsed._replace(path="/v1")
    return parsed.geturl().rstrip("/")


def _nested_attribute(root: object, *path: str) -> object | None:
    value: object | None = root
    for name in path:
        if value is None:
            return None
        value = getattr(value, name, None)
    return value


def _native_events(
    raw: object,
    *,
    session_index: int,
    latency: float,
) -> tuple[NativeMemoryEvent, ...]:
    events: list[NativeMemoryEvent] = []
    for index, row in enumerate(_result_rows(raw)):
        native_event = _text(row.get("event"), default="NONE").upper()
        memory = _memory_text(row)
        old_memory = _text(row.get("old_memory"))
        events.append(
            NativeMemoryEvent(
                operation_id=f"session-{session_index:03d}-native-{index:03d}",
                session_index=session_index,
                native_event=native_event,
                memory_id=_text(row.get("id")),
                memory_text=memory,
                old_content_hash=_text_hash(old_memory) if old_memory else None,
                new_content_hash=(
                    _text_hash(memory)
                    if memory and native_event in {"ADD", "UPDATE"}
                    else None
                ),
                source="native_response",
                latency_seconds=latency,
            )
        )
    return tuple(events)


def _install_usage_instrumentation(
    backend: object,
    config: Mapping[str, object],
    sink: list[ProviderUsageEvent],
    *,
    call_prefix: str,
) -> None:
    llm = _config_section(config, "llm")
    llm_provider = _text(llm.get("provider"))
    llm_settings = _config_section(llm, "config")
    llm_model = _text(llm_settings.get("model"))
    llm_endpoint = _endpoint_identity(
        llm_settings,
        default=(
            "https://api.anthropic.com"
            if llm_provider == "anthropic"
            else "https://api.deepseek.com"
            if llm_provider == "deepseek"
            else "https://api.openai.com"
        ),
    )
    llm_path = (
        ("llm", "client", "messages")
        if llm_provider == "anthropic"
        else ("llm", "client", "chat", "completions")
    )
    _wrap_provider_create(
        backend,
        llm_path,
        component="memory_internal_llm",
        provider=llm_provider,
        model_id=llm_model,
        endpoint_identity=llm_endpoint,
        sink=sink,
        call_prefix=call_prefix,
    )

    embedder = _config_section(config, "embedder")
    embedding_provider = _text(embedder.get("provider"))
    embedding_settings = _config_section(embedder, "config")
    embedding_model = _text(embedding_settings.get("model"))
    embedding_endpoint = _endpoint_identity(
        embedding_settings,
        default=(
            "http://embedding:80/v1"
            if embedding_provider == "huggingface"
            else "https://api.openai.com"
        ),
    )
    _wrap_provider_create(
        backend,
        ("embedding_model", "client", "embeddings"),
        component="embedding",
        provider=embedding_provider,
        model_id=embedding_model,
        endpoint_identity=embedding_endpoint,
        sink=sink,
        call_prefix=call_prefix,
    )


def _wrap_provider_create(
    root: object,
    path: tuple[str, ...],
    *,
    component: str,
    provider: str,
    model_id: str,
    endpoint_identity: str,
    sink: list[ProviderUsageEvent],
    call_prefix: str,
) -> None:
    target = root
    try:
        for attribute in path:
            target = getattr(target, attribute)
        boundary = cast(_ProviderCreateBoundary, target)
        original_value = boundary.create
    except AttributeError as exc:
        raise Mem0QualificationError(
            "usage_instrumentation_failure",
            f"Mem0 {component} client boundary is unavailable",
        ) from exc
    if not callable(original_value):
        raise Mem0QualificationError(
            "usage_instrumentation_failure",
            f"Mem0 {component} create boundary is not callable",
        )
    original = original_value

    def instrumented(*args: object, **kwargs: object) -> object:
        request_hash = _canonical_hash(
            {
                "component": component,
                "provider": provider,
                "model_id": model_id,
                "args": args,
                "kwargs": kwargs,
            }
        )
        input_count = _provider_input_count(component, kwargs)
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            ended_at = datetime.now(UTC)
            sink.append(
                ProviderUsageEvent(
                    call_id=_usage_call_id(
                        call_prefix,
                        component,
                        len(sink),
                        request_hash,
                    ),
                    component=component,
                    provider=provider,
                    model_id=model_id,
                    endpoint_identity=endpoint_identity,
                    request_hash=request_hash,
                    response_hash=_canonical_hash(
                        {"error_class": type(exc).__name__}
                    ),
                    input_tokens=None,
                    output_tokens=None,
                    cached_tokens=None,
                    reasoning_tokens=None,
                    usage_observed=False,
                    input_count=input_count,
                    latency_seconds=max(0.0, time.perf_counter() - started),
                    retry_count=None,
                    error_class=type(exc).__name__,
                    started_at_utc=started_at.isoformat(),
                    ended_at_utc=ended_at.isoformat(),
                )
            )
            raise
        ended_at = datetime.now(UTC)
        token_usage = _provider_token_usage(response)
        sink.append(
            ProviderUsageEvent(
                call_id=_usage_call_id(
                    call_prefix,
                    component,
                    len(sink),
                    request_hash,
                ),
                component=component,
                provider=provider,
                model_id=model_id,
                endpoint_identity=endpoint_identity,
                request_hash=request_hash,
                response_hash=_provider_response_hash(response),
                input_tokens=token_usage[0],
                output_tokens=token_usage[1],
                cached_tokens=token_usage[2],
                reasoning_tokens=token_usage[3],
                usage_observed=any(value is not None for value in token_usage),
                input_count=input_count,
                latency_seconds=max(0.0, time.perf_counter() - started),
                retry_count=None,
                error_class=None,
                started_at_utc=started_at.isoformat(),
                ended_at_utc=ended_at.isoformat(),
            )
        )
        return response

    boundary.create = instrumented


def _config_section(
    value: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    section = value.get(key)
    if not isinstance(section, Mapping):
        raise Mem0QualificationError(
            "usage_instrumentation_failure",
            f"Mem0 config section {key!r} is missing",
        )
    return {str(child_key): child for child_key, child in section.items()}


def _endpoint_identity(
    settings: Mapping[str, object],
    *,
    default: str,
) -> str:
    for key in (
        "openai_base_url",
        "anthropic_base_url",
        "deepseek_base_url",
        "huggingface_base_url",
    ):
        value = settings.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _provider_input_count(
    component: str,
    kwargs: Mapping[str, object],
) -> int:
    if component != "embedding":
        return 1
    value = kwargs.get("input")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 1 if value is not None else 0


def _provider_token_usage(
    response: object,
) -> tuple[int | None, int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    input_tokens = _first_integer_attribute(
        usage,
        ("prompt_tokens", "input_tokens"),
    )
    output_tokens = _first_integer_attribute(
        usage,
        ("completion_tokens", "output_tokens"),
    )
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached_tokens = _first_integer_attribute(
        prompt_details,
        ("cached_tokens",),
    )
    if cached_tokens is None:
        cached_tokens = _first_integer_attribute(
            usage,
            (
                "prompt_cache_hit_tokens",
                "cache_read_input_tokens",
                "cached_tokens",
            ),
        )
    reasoning_tokens = _first_integer_attribute(
        completion_details,
        ("reasoning_tokens",),
    )
    if reasoning_tokens is None:
        reasoning_tokens = _first_integer_attribute(
            usage,
            ("reasoning_tokens",),
        )
    return (
        input_tokens,
        output_tokens,
        cached_tokens,
        reasoning_tokens,
    )


def _first_integer_attribute(
    value: object,
    names: tuple[str, ...],
) -> int | None:
    for name in names:
        child = getattr(value, name, None)
        if isinstance(child, int) and not isinstance(child, bool):
            return child
    return None


def _provider_response_hash(response: object) -> str:
    usage = getattr(response, "usage", None)
    return _canonical_hash(
        {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "input_tokens": _first_integer_attribute(
                usage,
                ("prompt_tokens", "input_tokens"),
            ),
            "output_tokens": _first_integer_attribute(
                usage,
                ("completion_tokens", "output_tokens"),
            ),
        }
    )


def _usage_call_id(
    prefix: str,
    component: str,
    index: int,
    request_hash: str,
) -> str:
    return (
        f"{prefix}:{component}:{index:05d}:{request_hash[:12]}"
    )


def _result_rows(raw: object) -> list[dict[str, object]]:
    if isinstance(raw, Mapping):
        raw = raw.get("results")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [
        {str(key): value for key, value in row.items()}
        for row in raw
        if isinstance(row, Mapping)
    ]


def _memory_text(row: Mapping[str, object]) -> str:
    for key in ("memory", "text", "data", "content", "new_memory"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _metadata_pairs(value: object) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(key), child) for key, child in value.items()))


def _score_pairs(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        return ()
    output: list[tuple[str, float]] = []
    for key, child in value.items():
        score = _optional_float(child)
        if score is not None:
            output.append((str(key), score))
    return tuple(sorted(output))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _text(value: object, *, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _disable_mem0_telemetry() -> None:
    os.environ["MEM0_TELEMETRY"] = "False"
    for module_name in ("mem0.memory.telemetry", "mem0.memory.main"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "MEM0_TELEMETRY"):
            module.__dict__["MEM0_TELEMETRY"] = False


def _load_mem0() -> ModuleType:
    try:
        return importlib.import_module("mem0")
    except ImportError as exc:
        raise ImportError(
            "Mem0QualificationAdapter requires the qualification extra: "
            "pip install 'lhmsb[qualification]'"
        ) from exc


__all__ = [
    "CandidateSearch",
    "InventoryItem",
    "InventorySnapshot",
    "Mem0QualificationAdapter",
    "Mem0QualificationError",
    "Mem0V2Backend",
    "NativeMemoryEvent",
    "ProviderUsageEvent",
    "SearchCandidate",
    "WriteSessionResult",
    "build_mem0_live_config",
]
