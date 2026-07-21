"""Controlled MemOS-Tree qualification adapter.

Only the pinned official ``TreeTextMemory`` path is accepted here.  The adapter
keeps the benchmark-owned trace independent of MemOS' rapidly changing Python
internals: native objects and graph nodes are normalized through
``MemoryRuntime`` and graph snapshots are diffed to recover lifecycle events.
There is intentionally no ``GeneralTextMemory`` fallback.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import IO, Any, Protocol, cast
from urllib.parse import urlparse

from lhmsb.qualification.context import PublicHistoryUnit
from lhmsb.qualification.deepseek_writer import DeepSeekJSONBridge
from lhmsb.qualification.memory_runtime import (
    CANDIDATE_ORIGIN_METADATA_KEY,
    GRAPH_METADATA_KEY,
    PROVENANCE_METADATA_KEY,
    SCORE_SEMANTICS_METADATA_KEY,
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    ProviderUsageEvent,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.neo4j import (
    Neo4jError,
    Neo4jGraphSnapshot,
    Neo4jNode,
    Neo4jTransport,
    is_live_node,
)
from lhmsb.qualification.schema import MemOSTreeProfile, PolicyProfile
from lhmsb.qualification.source_manifest import (
    SourceManifestError,
    verified_source_commit_for_module,
)

_PINNED_SOURCE_COMMIT = "583b07b998afc4debb6c5078439b0b3896f5b097"
_OFFICIAL_TREE_MODULE = "memos.memories.textual.tree"
_DEEPSEEK_ENDPOINT_DEFAULT = "https://api.deepseek.com"
_EMBEDDING_ECHO_MARKER = "[search_by_embedding]"


class _MemOSStdoutFilter:
    """Drop an upstream debug print that serializes full embedding vectors."""

    def __init__(self, target: IO[str]) -> None:
        self.target = target

    def write(self, value: str) -> int:
        if _EMBEDDING_ECHO_MARKER in value:
            return len(value)
        return self.target.write(value)

    def flush(self) -> None:
        self.target.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self.target, name)


@contextmanager
def _filter_memos_stdout() -> Iterator[None]:
    """Preserve useful upstream output while removing vector parameter dumps."""
    current = sys.stdout
    if isinstance(current, _MemOSStdoutFilter):
        yield
        return
    with redirect_stdout(_MemOSStdoutFilter(current)):
        yield


class MemOSQualificationError(RuntimeError):
    """Typed terminal failure at the controlled MemOS boundary."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class MemOSLLMConfig:
    """One explicit DeepSeek config injected into a MemOS LLM component."""

    component: str
    model_id: str
    endpoint: str
    api_key: str
    temperature: float = 0.0
    # MemOS' reader can emit several structured memory objects in one call.
    # The former 512-token cap truncated otherwise valid JSON in live runs.
    max_output_tokens: int = 4096
    provider: str = "deepseek"
    output_language: str = "English"

    def __post_init__(self) -> None:
        if not self.component or not self.model_id or not self.api_key:
            raise ValueError("MemOS LLM component/model/api_key must be non-empty")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MemOS LLM endpoint must be an absolute HTTP(S) URL")
        if self.provider != "deepseek":
            raise ValueError("controlled MemOS LLM components must use DeepSeek")
        if "api.openai.com" in parsed.netloc.lower():
            raise ValueError("MemOS controlled writer rejects the OpenAI default endpoint")
        if self.temperature < 0 or self.max_output_tokens < 1:
            raise ValueError("invalid MemOS LLM sampling configuration")
        if self.output_language != "English":
            raise ValueError("controlled MemOS output language must be English")

    @property
    def endpoint_identity(self) -> str:
        return self.endpoint.rstrip("/")

    def to_backend_dict(self) -> dict[str, object]:
        return {
            "backend": "deepseek",
            "config": {
                "model_name_or_path": self.model_id,
                "api_key": self.api_key,
                "api_base": self.endpoint_identity,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
                "top_p": 1.0,
            },
        }


class MemOSLLMBridge(Protocol):
    """Structural bridge accepted by fake and official MemOS LLM components."""

    @property
    def calls(self) -> Sequence[ProviderUsageEvent]: ...

    def close(self) -> None: ...


class MemOSTreeBackend(Protocol):
    """Subset of the official TreeTextMemory surface used by the adapter."""

    def add(self, memory: object, *args: object, **kwargs: object) -> object: ...

    def search(self, query: str, *args: object, **kwargs: object) -> object: ...


class MemOSDeepSeekBridge:
    """A strict DeepSeek-only facade for MemOS' generic ``LLM.generate`` API.

    MemOS calls its LLM components with several equivalent method names across
    releases.  The facade exposes those names while routing all requests to the
    existing benchmark DeepSeek JSON bridge.  Calls are retained for provider
    accounting; no environment/provider fallback is attempted.
    """

    def __init__(self, config: MemOSLLMConfig, *, transport: object | None = None) -> None:
        self.config = config
        self._bridge = DeepSeekJSONBridge(
            api_key=config.api_key,
            model_id=config.model_id,
            endpoint=config.endpoint,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            transport=cast(Any, transport),
        )

    @property
    def calls(self) -> list[ProviderUsageEvent]:
        return self._bridge.calls

    def generate(
        self,
        messages: Sequence[Mapping[str, str]] | Mapping[str, str] | str,
        **kwargs: object,
    ) -> str:
        del kwargs
        return self._bridge.generate_text(
            _memos_messages(messages, output_language=self.config.output_language)
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]] | Mapping[str, str] | str,
        **kwargs: object,
    ) -> str:
        return self.generate(messages, **kwargs)

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: object | None = None,
        **kwargs: object,
    ) -> object:
        del kwargs
        result = self._bridge.generate_json(
            _memos_messages(messages, output_language=self.config.output_language),
            response_format=response_format or {"type": "object"},
        )
        return result.payload

    def close(self) -> None:
        self._bridge.close()


def _memos_messages(
    value: Sequence[Mapping[str, str]] | Mapping[str, str] | str,
    *,
    output_language: str | None = None,
) -> tuple[Mapping[str, str], ...]:
    """Normalize the prompt shapes accepted by MemOS' generic LLM API."""
    if isinstance(value, str):
        messages: tuple[Mapping[str, str], ...] = ({"role": "user", "content": value},)
    elif isinstance(value, Mapping):
        messages = (value,)
    else:
        messages = tuple(value)
    if output_language is None:
        return messages
    instruction = {
        "role": "system",
        "content": (
            "Preserve every format, schema, label, and token constraint in the task. "
            f"Render all natural-language memory content in {output_language}."
        ),
    }
    return (instruction, *messages)


class MemOSTreeQualificationAdapter:
    """Trace-complete adapter around one fresh official TreeTextMemory."""

    capabilities = LifecycleCapabilities(
        add=True,
        update=True,
        delete=True,
        merge=True,
        links=True,
        history=True,
        resumable=False,
    )

    def __init__(
        self,
        backend: MemOSTreeBackend | object,
        *,
        graph: Neo4jTransport | None = None,
        graph_store: Neo4jTransport | None = None,
        namespace: str = "default",
        episode_id: str | None = None,
        candidate_k: int = 20,
        inventory_limit: int = 100_000,
        embedding_runtime: object | None = None,
        embedding: object | None = None,
        reader: object | None = None,
        llm_components: Sequence[MemOSLLMConfig] = (),
        llm_bridges: Sequence[MemOSLLMBridge] = (),
        require_fresh_namespace: bool = True,
        reorganize: bool = True,
        internet_retrieval: bool = False,
        reorganizer_timeout_seconds: float = 300.0,
        expected_source_commit: str = _PINNED_SOURCE_COMMIT,
    ) -> None:
        if graph is not None and graph_store is not None:
            raise ValueError("provide graph or graph_store, not both")
        store = graph_store or graph
        if store is None:
            raise MemOSQualificationError(
                "graph_unavailable", "MemOS Tree requires a Neo4j graph boundary"
            )
        if not namespace:
            raise ValueError("namespace must be non-empty")
        if candidate_k < 1 or inventory_limit < 1:
            raise ValueError("candidate_k and inventory_limit must be positive")
        if reorganizer_timeout_seconds <= 0:
            raise ValueError("reorganizer_timeout_seconds must be positive")
        if not reorganize:
            raise MemOSQualificationError(
                "configuration_mismatch", "MemOS Tree requires reorganize=true"
            )
        if internet_retrieval:
            raise MemOSQualificationError(
                "configuration_mismatch",
                "MemOS Tree controlled retrieval disables internet retrieval",
            )
        if expected_source_commit != _PINNED_SOURCE_COMMIT:
            raise MemOSQualificationError(
                "source_pin_mismatch", "MemOS source commit is not pinned"
            )
        self.backend = cast(MemOSTreeBackend, backend)
        self.graph = store
        self.namespace = namespace
        self.episode_id = episode_id
        self.candidate_k = candidate_k
        self.inventory_limit = inventory_limit
        self.embedding = embedding_runtime or embedding
        self.reader = reader
        self.llm_components = tuple(llm_components)
        self.llm_bridges = tuple(llm_bridges)
        self.reorganizer_timeout_seconds = reorganizer_timeout_seconds
        self._n_write = 0
        self._last_write_session = -1
        self._closed = False
        self.diagnostics: list[tuple[str, object]] = []
        self._usage_events: list[ProviderUsageEvent] = []
        self._validate_tree_api()
        for config in self.llm_components:
            if config.provider != "deepseek":
                raise MemOSQualificationError(
                    "provider_mismatch", "all MemOS LLM components must use DeepSeek"
                )
        if require_fresh_namespace:
            try:
                self.graph.validate_empty(namespace=namespace)
            except Neo4jError as exc:
                raise MemOSQualificationError(exc.error_class, str(exc)) from exc

    @classmethod
    def create_live(
        cls,
        profile: MemOSTreeProfile,
        *,
        policy: PolicyProfile,
        api_key: str,
        embedding_runtime: object,
        embedding_base_url: str = "http://127.0.0.1:8080",
        namespace: str,
        episode_id: str,
        neo4j_transport: Neo4jTransport | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = None,
        neo4j_database: str = "neo4j",
        source_commit: str | None = None,
        module: object | None = None,
        reader_module: object | None = None,
        candidate_k: int | None = None,
        http_transport: object | None = None,
    ) -> MemOSTreeQualificationAdapter:
        """Construct only the official pinned TreeTextMemory path."""
        if profile.source_commit != _PINNED_SOURCE_COMMIT:
            raise MemOSQualificationError(
                "source_pin_mismatch", "MemOS profile source commit is not pinned"
            )
        if policy.provider != "deepseek":
            raise MemOSQualificationError(
                "writer_profile_mismatch", "controlled MemOS requires DeepSeek"
            )
        if not api_key:
            raise MemOSQualificationError("missing_secret", "DeepSeek API key is required")
        if not neo4j_uri or not neo4j_password:
            raise MemOSQualificationError(
                "graph_unavailable", "live MemOS requires Neo4j URI/password"
            )
        module = module or _load_tree_module()
        _validate_official_identity(module, expected_commit=source_commit or profile.source_commit)
        tree_class = getattr(module, "TreeTextMemory", None)
        if not callable(tree_class):
            raise MemOSQualificationError(
                "upstream_api_mismatch", "official package lacks TreeTextMemory"
            )
        if getattr(module, "GeneralTextMemory", None) is tree_class:
            raise MemOSQualificationError(
                "package_identity_mismatch", "TreeTextMemory resolved to GeneralTextMemory"
            )
        components = tuple(
            MemOSLLMConfig(
                component=name,
                model_id=policy.model_id,
                endpoint=policy.endpoint,
                api_key=api_key,
                temperature=0.0,
                max_output_tokens=4096,
                output_language=profile.output_language,
            )
            for name in ("reader", "extractor", "reorganizer", "dispatcher")
        )
        tree_config = _build_tree_config(
            profile,
            components,
            embedding_base_url=embedding_base_url,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            neo4j_user_name=namespace,
        )
        graph = neo4j_transport
        if graph is None:
            from lhmsb.qualification.neo4j import Neo4jBoltTransport

            graph = Neo4jBoltTransport(
                neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
                exclusive_database=False,
                namespace_property="user_name",
            )
        # Validate the native MemOS user namespace before constructing the
        # official Tree backend.  Tree construction owns background workers;
        # detecting contamination afterwards can otherwise leave a failed
        # preparation process alive while those workers drain.
        try:
            graph.validate_empty(namespace=namespace)
        except Neo4jError as exc:
            graph.close()
            raise MemOSQualificationError(exc.error_class, str(exc)) from exc
        try:
            backend = tree_class(tree_config)
        except Exception as exc:
            graph.close()
            raise MemOSQualificationError("upstream_init_failure", str(exc)) from exc
        reader = _build_official_reader(
            reader_module,
            components[0],
            embedding_runtime,
            embedding_model=profile.embedding_model,
            embedding_base_url=embedding_base_url,
        )
        if reader is not None:
            _set_if_possible(backend, "reader", reader)
            _set_if_possible(backend, "memory_reader", reader)
        bridges = {
            component.component: MemOSDeepSeekBridge(
                component,
                transport=http_transport,
            )
            for component in components
        }
        _install_memos_bridges(backend, reader, bridges)
        _assert_runtime_configuration(backend, components)
        return cls(
            backend,
            graph=graph,
            namespace=namespace,
            episode_id=episode_id,
            candidate_k=candidate_k or profile.candidate_k,
            embedding_runtime=embedding_runtime,
            reader=reader,
            llm_components=components,
            llm_bridges=tuple(bridges.values()),
            require_fresh_namespace=False,
            reorganize=True,
            internet_retrieval=False,
            expected_source_commit=profile.source_commit,
        )

    @property
    def n_write(self) -> int:
        return self._n_write

    def restore_write_count(self, n_write: int) -> None:
        self._ensure_open()
        if isinstance(n_write, bool) or not isinstance(n_write, int) or n_write < 0:
            raise ValueError("n_write must be a non-negative integer")
        self._n_write = n_write

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        try:
            self._wait_reorganizer()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            errors.append(f"reorganizer barrier: {exc}")
        manager = getattr(self.backend, "memory_manager", None)
        # The official manager owns background threads that can still be using
        # the injected LLMs.  Stop it before closing those bridges.
        for target in (
            manager,
            self.backend,
            *self.llm_bridges,
            self.graph,
        ):
            close = getattr(target, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    errors.append(f"{type(target).__name__}: {exc}")
        self._closed = True
        if errors:
            raise MemOSQualificationError("resource_cleanup_failure", "; ".join(errors))

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        with _filter_memos_stdout():
            return self._write_session(
                messages,
                session_index=session_index,
                metadata=metadata,
            )

    def _write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        self._ensure_open()
        if (
            isinstance(session_index, bool)
            or not isinstance(session_index, int)
            or session_index < 0
        ):
            raise ValueError("session_index must be a non-negative integer")
        if session_index < self._last_write_session:
            raise MemOSQualificationError("session_order", "MemOS writes must be ordered")
        before = self._graph_snapshot()
        payload = self._reader_payload(messages, metadata, session_index)
        started = time.perf_counter()
        try:
            self._invoke_add(payload, session_index=session_index, metadata=metadata)
        except MemOSQualificationError:
            raise
        except Exception as exc:
            raise MemOSQualificationError("memos_write_failure", str(exc)) from exc
        self._wait_reorganizer()
        after = self._graph_snapshot()
        events, edge_events = _graph_diff_events(before, after, session_index=session_index)
        self.diagnostics.append((f"graph_edges:{session_index}", edge_events))
        self._n_write += len(events)
        self._last_write_session = max(self._last_write_session, session_index)
        inventory = self.snapshot_inventory(checkpoint_session=session_index)
        latency = max(0.0, time.perf_counter() - started)
        return WriteSessionResult(
            session_index=session_index,
            events=tuple(events),
            inventory=InventorySnapshot(
                checkpoint_session=inventory.checkpoint_session,
                n_write=self._n_write,
                n_live=inventory.n_live,
                items=inventory.items,
                store_hash=inventory.store_hash,
                backend_count=inventory.backend_count,
            ),
            n_write=self._n_write,
            latency_seconds=latency,
            usage_events=self._new_usage_events(),
        )

    def snapshot_inventory(self, *, checkpoint_session: int) -> InventorySnapshot:
        self._ensure_open()
        if isinstance(checkpoint_session, bool) or checkpoint_session < 0:
            raise ValueError("checkpoint_session must be a non-negative integer")
        snapshot = self._graph_snapshot()
        live_nodes = snapshot.live_nodes
        if len(live_nodes) > self.inventory_limit:
            raise MemOSQualificationError(
                "inventory_failure", "MemOS graph inventory limit exceeded"
            )
        items = tuple(self._memory_object(node, snapshot) for node in live_nodes)
        payload = [item.to_dict() for item in items]
        store_hash = sha256_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=self._n_write,
            n_live=len(items),
            items=items,
            store_hash=store_hash,
            backend_count=len(items),
        )

    def search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch:
        with _filter_memos_stdout():
            return self._search_candidates(
                query,
                checkpoint_session=checkpoint_session,
            )

    def _search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch:
        self._ensure_open()
        if not isinstance(query, str):
            raise ValueError("query must be text")
        started = time.perf_counter()
        try:
            raw = self._invoke_search(query, checkpoint_session=checkpoint_session)
        except MemOSQualificationError:
            raise
        except Exception as exc:
            raise MemOSQualificationError("memos_search_failure", str(exc)) from exc
        rows = _sequence_or_empty(raw)
        snapshot = self._graph_snapshot()
        live = {node.node_id: node for node in snapshot.live_nodes}
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for rank, row in enumerate(rows[: self.candidate_k], start=1):
            candidate = _candidate_row(row, live=live, rank=rank)
            if candidate.memory_id in seen:
                raise MemOSQualificationError(
                    "duplicate_memory_id", "MemOS search returned duplicate IDs"
                )
            seen.add(candidate.memory_id)
            candidates.append(candidate)
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=sha256_text(query),
            candidates=tuple(candidates),
            candidate_shortfall=len(candidates) < self.candidate_k,
            latency_seconds=max(0.0, time.perf_counter() - started),
            usage_events=self._new_usage_events(),
        )

    def storage_footprints(self) -> tuple[StorageFootprint, ...]:
        try:
            size = self.graph.storage_bytes(namespace=self.namespace)
        except Exception:
            size = None
        if size is None:
            return (
                StorageFootprint(
                    component="memos_neo4j",
                    bytes=None,
                    unavailable_reason="Neo4j Bolt does not expose portable per-namespace bytes",
                ),
            )
        return (StorageFootprint(component="memos_neo4j", bytes=size, unavailable_reason=None),)

    def _validate_tree_api(self) -> None:
        for name in ("add", "search"):
            if not callable(getattr(self.backend, name, None)):
                raise MemOSQualificationError(
                    "upstream_api_mismatch", f"TreeTextMemory lacks {name} API"
                )
        manager = getattr(self.backend, "memory_manager", None)
        wait = getattr(manager, "wait_reorganizer", None) if manager is not None else None
        if not callable(wait):
            wait = getattr(self.backend, "wait_reorganizer", None)
        if not callable(wait):
            raise MemOSQualificationError(
                "upstream_api_mismatch", "TreeTextMemory lacks synchronous reorganizer wait API"
            )

    def _graph_snapshot(self) -> Neo4jGraphSnapshot:
        try:
            return self.graph.snapshot(namespace=self.namespace)
        except Neo4jError as exc:
            raise MemOSQualificationError(exc.error_class, str(exc)) from exc

    def _reader_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        metadata: Mapping[str, object] | None,
        session_index: int,
    ) -> object:
        if self.reader is None:
            return list(messages)
        get_memory = getattr(self.reader, "get_memory", None)
        if not callable(get_memory):
            raise MemOSQualificationError(
                "upstream_api_mismatch", "SimpleStructMemReader lacks get_memory"
            )
        scene_messages: list[dict[str, str]] = []
        raw_units = (metadata or {}).get(
            "public_history_units", (metadata or {}).get("public_units", ())
        )
        if isinstance(raw_units, Sequence) and not isinstance(raw_units, str | bytes):
            for item in raw_units:
                if isinstance(item, PublicHistoryUnit):
                    content = item.content
                elif isinstance(item, Mapping) and isinstance(item.get("content"), str):
                    content = cast(str, item["content"])
                else:
                    raise MemOSQualificationError(
                        "invalid_public_units", "public history unit is malformed"
                    )
                scene_messages.append({"role": "user", "content": content})
        if not scene_messages:
            scene_messages = [dict(message) for message in messages]
        scene_data = [scene_messages]
        info = {
            "user_id": self.namespace,
            "episode_id": self.episode_id or "episode",
            "session_id": f"{self.episode_id or 'episode'}:{session_index}",
        }
        try:
            result = get_memory(scene_data, type="chat", info=info)
        except Exception as exc:
            raise MemOSQualificationError("memos_reader_failure", str(exc)) from exc
        if isinstance(result, Sequence) and not isinstance(result, str | bytes):
            flattened: list[object] = []
            for item in result:
                if isinstance(item, Sequence) and not isinstance(item, str | bytes | Mapping):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return result

    def _invoke_add(
        self,
        payload: object,
        *,
        session_index: int,
        metadata: Mapping[str, object] | None,
    ) -> object:
        add = self.backend.add
        kwargs: dict[str, object] = {}
        signature = _signature(add)
        if _accepts(signature, "reorganize"):
            kwargs["reorganize"] = True
        if _accepts(signature, "info"):
            kwargs["info"] = {
                "user_id": self.namespace,
                "episode_id": self.episode_id or "episode",
                "session_id": f"{self.episode_id or 'episode'}:{session_index}",
            }
        if _accepts(signature, "user_name"):
            kwargs["user_name"] = self.namespace
        if _accepts(signature, "metadata"):
            kwargs["metadata"] = dict(metadata or {})
        try:
            return add(payload, **kwargs)
        except TypeError:
            # Some pinned releases expose ``add(m_list)`` only; retry exactly
            # that native shape, never a different memory implementation.
            if kwargs:
                return add(payload)
            raise

    def _wait_reorganizer(self) -> None:
        manager = getattr(self.backend, "memory_manager", None)
        wait = getattr(manager, "wait_reorganizer", None) if manager is not None else None
        if not callable(wait):
            wait = getattr(self.backend, "wait_reorganizer", None)
        if not callable(wait):
            raise MemOSQualificationError(
                "upstream_api_mismatch", "reorganizer wait API disappeared"
            )
        signature = _signature(wait)
        started = time.perf_counter()
        try:
            if _accepts(signature, "timeout"):
                result = wait(timeout=self.reorganizer_timeout_seconds)
            elif _accepts(signature, "timeout_seconds"):
                result = wait(timeout_seconds=self.reorganizer_timeout_seconds)
            else:
                result = wait()
        except TimeoutError as exc:
            raise MemOSQualificationError("reorganizer_timeout", str(exc)) from exc
        except Exception as exc:
            raise MemOSQualificationError("reorganizer_failure", str(exc)) from exc
        # MemOS 2.0.23 checks ``queue.empty()`` before calling ``queue.join()``.
        # A consumer may already have dequeued an item while its LLM work is
        # still active, so empty() can be true with unfinished_tasks > 0.  Wait
        # on that counter explicitly to make graph snapshots and shutdown a
        # real barrier.
        reorganizer = getattr(manager, "reorganizer", None) if manager is not None else None
        queue = getattr(reorganizer, "queue", None)
        unfinished = getattr(queue, "unfinished_tasks", None)
        while isinstance(unfinished, int) and not isinstance(unfinished, bool) and unfinished > 0:
            if time.perf_counter() - started > self.reorganizer_timeout_seconds:
                raise MemOSQualificationError(
                    "reorganizer_timeout", "MemOS reorganizer still has unfinished tasks"
                )
            time.sleep(0.01)
            unfinished = getattr(queue, "unfinished_tasks", None)
        if result is False or time.perf_counter() - started > self.reorganizer_timeout_seconds:
            raise MemOSQualificationError(
                "reorganizer_timeout", "MemOS reorganizer did not become idle"
            )

    def _invoke_search(self, query: str, *, checkpoint_session: int) -> object:
        search = self.backend.search
        signature = _signature(search)
        kwargs: dict[str, object] = {}
        if _accepts(signature, "top_k"):
            kwargs["top_k"] = self.candidate_k
        if _accepts(signature, "mode"):
            kwargs["mode"] = "fast"
        if _accepts(signature, "info"):
            kwargs["info"] = {
                "user_id": self.namespace,
                "query": query,
                "episode_id": self.episode_id or "episode",
                "session_id": f"{self.episode_id or 'episode'}:{checkpoint_session}",
            }
        if _accepts(signature, "user_name"):
            kwargs["user_name"] = self.namespace
        try:
            return search(query, **kwargs)
        except TypeError:
            return search(query)

    def _memory_object(self, node: Neo4jNode, snapshot: Neo4jGraphSnapshot) -> MemoryObject:
        values = node.property_map
        content = node.content
        created = _text(
            values.get("created_at", values.get("created", "1970-01-01T00:00:00+00:00"))
        )
        updated = _text(values.get("updated_at", values.get("updated", created)))
        neighbors = sum(
            edge.source_id == node.node_id or edge.target_id == node.node_id
            for edge in snapshot.edges
        )
        labels = tuple(node.labels)
        graph_meta = {
            "labels": list(labels),
            "node_kind": node.kind,
            "status": node.status,
            "structural": _is_structural(node),
            "topic": _is_topic(node),
            "edge_count": neighbors,
        }
        provenance = values.get("source_unit_ids", values.get("source_ids", ()))
        if isinstance(provenance, str):
            provenance = [provenance]
        if not isinstance(provenance, Sequence) or isinstance(provenance, str | bytes):
            provenance = []
        metadata_items: list[tuple[str, object]] = [
            (PROVENANCE_METADATA_KEY, {"backend": "memos_tree", "unit_ids": list(provenance)}),
            (GRAPH_METADATA_KEY, graph_meta),
            ("node_kind", node.kind),
            ("labels", list(labels)),
        ]
        # An absent upstream session is unknown, not session zero.  Omitting it
        # lets evaluator-side lifecycle lineage supply the mutation session.
        if "session_index" in values:
            metadata_items.append(("session_index", _int(values["session_index"])))
        return MemoryObject(
            memory_id=node.node_id,
            content=content,
            content_hash=sha256_text(content),
            metadata=tuple(metadata_items),
            created_at=created,
            updated_at=updated,
            history_length=_int(values.get("history_length", values.get("version", 1)), minimum=0),
        )

    def _new_usage_events(self) -> tuple[ProviderUsageEvent, ...]:
        events: list[ProviderUsageEvent] = []
        for component in self.llm_bridges:
            calls = getattr(component, "calls", None)
            if isinstance(calls, Sequence) and not isinstance(calls, str | bytes):
                events.extend(item for item in calls if isinstance(item, ProviderUsageEvent))
        backend_calls = getattr(self.backend, "provider_usage_events", None)
        if isinstance(backend_calls, Sequence) and not isinstance(backend_calls, str | bytes):
            events.extend(item for item in backend_calls if isinstance(item, ProviderUsageEvent))
        fresh: list[ProviderUsageEvent] = []
        seen = {event.call_id for event in self._usage_events}
        for event in events:
            if event.call_id not in seen:
                fresh.append(event)
                seen.add(event.call_id)
        self._usage_events.extend(fresh)
        return tuple(fresh)

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemOSQualificationError("adapter_closed", "MemOS adapter is closed")


# Naming aliases used by early server-run notebooks; they intentionally point
# to the same strict TreeTextMemory implementation rather than a second mode.
MemOSQualificationAdapter = MemOSTreeQualificationAdapter
MemOSTreeAdapter = MemOSTreeQualificationAdapter


def _graph_diff_events(
    before: Neo4jGraphSnapshot,
    after: Neo4jGraphSnapshot,
    *,
    session_index: int,
) -> tuple[list[MemoryMutationEvent], tuple[dict[str, object], ...]]:
    old_nodes = {node.node_id: node for node in before.nodes}
    new_nodes = {node.node_id: node for node in after.nodes}
    events: list[MemoryMutationEvent] = []
    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        old = old_nodes.get(node_id)
        new = new_nodes.get(node_id)
        native_event: str | None = None
        base_node = new if new is not None else old
        memory_text = base_node.content if base_node is not None else ""
        old_hash = sha256_text(old.content) if old is not None else None
        new_hash = sha256_text(new.content) if new is not None and is_live_node(new) else None
        if old is None and new is not None:
            native_event = "ADD" if is_live_node(new) else "ARCHIVE"
        elif old is not None and new is None:
            native_event = "DELETE"
        elif old is not None and new is not None:
            old_live = is_live_node(old)
            new_live = is_live_node(new)
            if old_live and not new_live:
                native_event = "ARCHIVE"
            elif not old_live and new_live:
                native_event = "REOPEN"
            elif (
                old.content != new.content
                or old.properties != new.properties
                or old.labels != new.labels
            ):
                native_event = "UPDATE"
        if native_event is not None:
            events.append(
                MemoryMutationEvent(
                    operation_id=f"memos-{session_index:06d}-{len(events):06d}-{node_id}",
                    session_index=session_index,
                    native_event=native_event,
                    memory_id=node_id,
                    memory_text=memory_text,
                    old_content_hash=old_hash,
                    new_content_hash=new_hash,
                    source="neo4j_graph_diff",
                    latency_seconds=0.0,
                )
            )
    old_edges = {edge.edge_id: edge for edge in before.edges}
    new_edges = {edge.edge_id: edge for edge in after.edges}
    edge_events: list[dict[str, object]] = []
    for edge_id in sorted(set(old_edges) | set(new_edges)):
        old_edge = old_edges.get(edge_id)
        new_edge = new_edges.get(edge_id)
        edge = new_edge if new_edge is not None else old_edge
        if edge is None:
            continue
        edge_events.append(
            {
                "edge_id": edge_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relationship": edge.relationship,
                "event": "ADD" if old_edge is None else "REMOVE" if new_edge is None else "UPDATE",
                "lineage": edge.relationship == "MERGED_TO",
            }
        )
        # Relationships are graph diagnostics, not memory-object mutations.
        # Counting every MERGED_TO edge as a write inflated memory count and
        # duplicated the same target object across reorganizer passes.
    return events, tuple(edge_events)


def _candidate_row(
    row: object,
    *,
    live: Mapping[str, Neo4jNode],
    rank: int,
) -> RetrievalCandidate:
    values = _row_map(row)
    memory_id = _text(values.get("memory_id", values.get("id", values.get("node_id", ""))))
    if not memory_id:
        content = _text(values.get("memory", values.get("content", values.get("text", ""))))
        matches = [node_id for node_id, node in live.items() if node.content == content]
        if len(matches) != 1:
            raise MemOSQualificationError(
                "malformed_upstream_response", "MemOS search row has no unique ID"
            )
        memory_id = matches[0]
    node = live.get(memory_id)
    if node is None:
        raise MemOSQualificationError(
            "candidate_outside_inventory", f"unknown graph node {memory_id!r}"
        )
    content = _text(values.get("memory", values.get("content", values.get("text", node.content))))
    raw_score = values.get("score", values.get("similarity"))
    distance = values.get("distance")
    score = _float_or_none(distance if distance is not None else raw_score)
    semantics = "lower_is_better" if distance is not None else "higher_is_better"
    origin_value = values.get(
        "candidate_origin", values.get("origin", values.get("source", "native"))
    )
    expanded = bool(
        values.get("is_graph_expanded", values.get("graph_expanded", values.get("expanded", False)))
    )
    origin = (
        "graph_expanded"
        if expanded or str(origin_value).lower() in {"graph", "expanded", "neighbor"}
        else "native"
    )
    metadata = (
        (CANDIDATE_ORIGIN_METADATA_KEY, origin),
        (SCORE_SEMANTICS_METADATA_KEY, semantics if score is not None else "unscored"),
        (PROVENANCE_METADATA_KEY, {"backend": "memos_tree", "native_id": memory_id}),
        (
            GRAPH_METADATA_KEY,
            {
                "labels": list(node.labels),
                "node_kind": node.kind,
                "structural": _is_structural(node),
                "topic": _is_topic(node),
            },
        ),
    )
    return RetrievalCandidate(
        memory_id=memory_id,
        content=content,
        content_hash=sha256_text(content),
        native_rank=rank,
        score=score,
        score_details=()
        if score is None
        else (("distance" if distance is not None else "similarity", score),),
        metadata=metadata,
        created_at=_text(node.property_map.get("created_at", "1970-01-01T00:00:00+00:00")),
        updated_at=_text(node.property_map.get("updated_at", "1970-01-01T00:00:00+00:00")),
    )


def _row_map(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    if hasattr(row, "to_dict") and callable(row.to_dict):
        raw = row.to_dict()
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items()}
    values: dict[str, object] = {}
    for name in (
        "id",
        "memory_id",
        "node_id",
        "memory",
        "content",
        "text",
        "score",
        "distance",
        "expanded",
        "is_graph_expanded",
    ):
        if hasattr(row, name):
            values[name] = getattr(row, name)
    if not values:
        raise MemOSQualificationError(
            "malformed_upstream_response", "MemOS search row is not object-like"
        )
    return values


def _sequence_or_empty(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        for key in ("results", "memories", "items", "data"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, str | bytes):
                return tuple(nested)
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(value)
    if value is None:
        return ()
    raise MemOSQualificationError(
        "malformed_upstream_response", "MemOS search must return an array"
    )


def _load_tree_module() -> Any:
    try:
        return importlib.import_module(_OFFICIAL_TREE_MODULE)
    except ImportError as exc:  # pragma: no cover - server dependency
        raise MemOSQualificationError(
            "official_dependency_missing", "pinned MemOS TreeTextMemory package is required"
        ) from exc


def _validate_official_identity(module: object, *, expected_commit: str) -> None:
    name = getattr(module, "__name__", "")
    if not isinstance(name, str) or not name.startswith("memos."):
        raise MemOSQualificationError(
            "package_identity_mismatch", "module is not the official MemOS package"
        )
    commit = None
    for field in ("__source_commit__", "SOURCE_COMMIT", "__commit__", "COMMIT_SHA"):
        value = getattr(module, field, None)
        if isinstance(value, str):
            commit = value
            break
    if commit is None:
        try:
            commit = verified_source_commit_for_module(module, "memos")
        except SourceManifestError as exc:
            raise MemOSQualificationError(
                "source_pin_mismatch",
                f"MemOS external source identity is not verified: {exc}",
            ) from exc
    if commit != expected_commit:
        raise MemOSQualificationError(
            "source_pin_mismatch", f"MemOS source commit {commit!r} != expected {expected_commit!r}"
        )
    if not callable(getattr(module, "TreeTextMemory", None)):
        raise MemOSQualificationError(
            "upstream_api_mismatch", "official MemOS module lacks TreeTextMemory"
        )
    general_memory = getattr(module, "GeneralTextMemory", None)
    tree_memory = getattr(module, "TreeTextMemory", None)
    if callable(general_memory) and tree_memory is general_memory:
        raise MemOSQualificationError(
            "package_identity_mismatch", "GeneralTextMemory cannot back the Tree condition"
        )


def validate_memos_source(module: object, *, expected_commit: str = _PINNED_SOURCE_COMMIT) -> None:
    """Public source/API preflight helper used by the server gate."""
    _validate_official_identity(module, expected_commit=expected_commit)


def _build_tree_config(
    profile: MemOSTreeProfile,
    components: Sequence[MemOSLLMConfig],
    *,
    embedding_base_url: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    neo4j_user_name: str,
) -> object:
    llm = {component.component: component.to_backend_dict() for component in components}
    config: dict[str, object] = {
        "extractor_llm": llm["extractor"],
        "dispatcher_llm": llm["dispatcher"],
        "reorganize": True,
        "embedder": {
            "backend": "universal_api",
            "config": {
                # MemOS 2.0.23 accepts the OpenAI-compatible TEI endpoint
                # through its ``openai`` provider; ``openai_compatible`` is
                # not a registered provider in UniversalAPIEmbedder.
                "provider": "openai",
                "model_name_or_path": profile.embedding_model,
                "base_url": embedding_base_url,
                "api_key": "EMPTY",
            },
        },
        "graph_db": {
            "backend": "neo4j",
            "config": {
                "uri": neo4j_uri,
                "user": neo4j_user,
                "password": neo4j_password,
                "db_name": neo4j_database,
                "use_multi_db": False,
                "user_name": neo4j_user_name,
                "auto_create": False,
                "embedding_dimension": 1024,
            },
        },
    }
    try:
        config_module = importlib.import_module("memos.configs.memory")
    except ImportError:
        return config
    config_class = getattr(config_module, "TreeTextMemoryConfig", None)
    if config_class is None:
        return config
    for method_name in ("model_validate", "from_dict", "from_json"):
        method = getattr(config_class, method_name, None)
        if callable(method):
            try:
                return method(config)
            except Exception:
                continue
    try:
        return config_class(**config)
    except Exception as exc:
        raise MemOSQualificationError("upstream_config_failure", str(exc)) from exc


def _build_official_reader(
    module: object | None,
    component: MemOSLLMConfig,
    embedding: object,
    *,
    embedding_model: str,
    embedding_base_url: str,
) -> object | None:
    if module is None:
        try:
            module = importlib.import_module("memos.mem_reader.simple_struct")
        except ImportError:  # pragma: no cover - server dependency
            return None
    reader_class = getattr(module, "SimpleStructMemReader", None)
    if not callable(reader_class):
        raise MemOSQualificationError(
            "upstream_api_mismatch", "official package lacks SimpleStructMemReader"
        )
    config_module = importlib.import_module("memos.configs.mem_reader")
    config_class = getattr(config_module, "SimpleStructMemReaderConfig", None)
    if config_class is None:
        raise MemOSQualificationError(
            "upstream_api_mismatch", "official package lacks reader config"
        )
    config: dict[str, object] = {
        "llm": component.to_backend_dict(),
        "embedder": {
            "backend": "universal_api",
            "config": {
                # The official reader uses the same OpenAI-compatible TEI
                # endpoint and the same provider registry as TreeTextMemory.
                "provider": "openai",
                "model_name_or_path": embedding_model,
                "api_key": "EMPTY",
                "base_url": embedding_base_url,
            },
        },
        "chunker": {
            "backend": "sentence",
            "config": {
                "chunk_size": 2048,
                "chunk_overlap": 128,
                # Chonkie defaults to the remote ``gpt2`` tokenizer.  The
                # controlled server run is offline, so prefer an explicitly
                # configured local tokenizer and otherwise reuse the local
                # embedding model path.
                "tokenizer_or_token_counter": os.environ.get(
                    "LHMSB_MEMOS_TOKENIZER_PATH", embedding_model
                ),
            },
        },
    }
    config_obj: object = config
    for method_name in ("model_validate", "from_dict", "from_json"):
        method = getattr(config_class, method_name, None)
        if callable(method):
            try:
                config_obj = method(config)
                break
            except Exception:
                continue
    if config_obj is config:
        try:
            config_obj = config_class(**config)
        except Exception as exc:
            raise MemOSQualificationError("upstream_config_failure", str(exc)) from exc
    try:
        reader = reader_class(config_obj)
    except Exception as exc:
        raise MemOSQualificationError("upstream_init_failure", str(exc)) from exc
    _set_if_possible(reader, "embedding_runtime", embedding)
    return cast(object, reader)


def _assert_runtime_configuration(backend: object, components: Sequence[MemOSLLMConfig]) -> None:
    expected = {(item.model_id, item.endpoint_identity) for item in components}
    for name in ("extractor_llm", "dispatcher_llm", "reorganizer_llm", "reader_llm", "llm"):
        value = getattr(backend, name, None)
        if value is None:
            continue
        text = repr(value).lower()
        if "openai.com" in text or "anthropic" in text or "ollama" in text:
            raise MemOSQualificationError(
                "provider_mismatch", f"MemOS component {name} uses an unsupported provider"
            )
        if expected and not any(
            model in repr(value) and endpoint in repr(value) for model, endpoint in expected
        ):
            continue


def _install_memos_bridges(
    backend: object,
    reader: object | None,
    bridges: Mapping[str, MemOSLLMBridge],
) -> None:
    """Replace every official MemOS LLM reference with the controlled bridge.

    TreeTextMemory copies its extractor into the asynchronous reorganizer and
    its nested helpers during construction.  Replacing only the two public
    attributes therefore leaves native reasoning-enabled clients active.  The
    reader likewise keeps three aliases for its LLM.  All of these references
    must point to benchmark-owned DeepSeek bridges before the first write.
    """
    required = {"reader", "extractor", "reorganizer", "dispatcher"}
    if set(bridges) != required:
        raise MemOSQualificationError(
            "writer_bridge_mismatch",
            "MemOS requires reader/extractor/reorganizer/dispatcher bridges",
        )

    _replace_required(backend, "extractor_llm", bridges["extractor"])
    _replace_required(backend, "dispatcher_llm", bridges["dispatcher"])

    manager = getattr(backend, "memory_manager", None)
    reorganizer = getattr(manager, "reorganizer", None)
    if reorganizer is None:
        raise MemOSQualificationError(
            "upstream_api_mismatch", "MemOS memory manager lacks reorganizer"
        )
    reorganizer_bridge = bridges["reorganizer"]
    _replace_required(reorganizer, "llm", reorganizer_bridge)
    _replace_required(
        getattr(reorganizer, "relation_detector", None),
        "llm",
        reorganizer_bridge,
    )
    _replace_required(
        getattr(reorganizer, "resolver", None),
        "llm",
        reorganizer_bridge,
    )

    if reader is None:
        raise MemOSQualificationError(
            "upstream_api_mismatch", "controlled MemOS requires the official reader"
        )
    reader_bridge = bridges["reader"]
    for name in ("llm", "general_llm", "preference_extractor_llm"):
        _replace_required(reader, name, reader_bridge)


def _replace_required(target: object | None, name: str, value: object) -> None:
    if target is None or not hasattr(target, name):
        raise MemOSQualificationError(
            "upstream_api_mismatch", f"MemOS component lacks {name} LLM reference"
        )
    try:
        setattr(target, name, value)
    except Exception as exc:
        raise MemOSQualificationError(
            "writer_bridge_install_failure", f"cannot replace MemOS {name} LLM"
        ) from exc
    if getattr(target, name, None) is not value:
        raise MemOSQualificationError(
            "writer_bridge_install_failure", f"MemOS {name} LLM replacement did not stick"
        )


def _set_if_possible(target: object, name: str, value: object) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        return


def _signature(value: object) -> inspect.Signature | None:
    if not callable(value):
        return None
    try:
        return inspect.signature(value)
    except (TypeError, ValueError):
        return None


def _accepts(signature: inspect.Signature | None, name: str) -> bool:
    if signature is None:
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _text(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _int(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return minimum
    return max(minimum, value)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _is_structural(node: Neo4jNode) -> bool:
    values = node.property_map
    raw = values.get("node_kind", values.get("kind", ""))
    return isinstance(raw, str) and raw.lower() in {
        "structural",
        "structure",
        "topic",
        "root",
        "summary",
    }


def _is_topic(node: Neo4jNode) -> bool:
    values = node.property_map
    raw = values.get("node_kind", values.get("kind", ""))
    return isinstance(raw, str) and raw.lower() in {"topic", "category"}


__all__ = [
    "MemOSDeepSeekBridge",
    "MemOSLLMConfig",
    "MemOSQualificationError",
    "MemOSQualificationAdapter",
    "MemOSTreeAdapter",
    "MemOSTreeQualificationAdapter",
    "MemOSTreeBackend",
    "validate_memos_source",
]
