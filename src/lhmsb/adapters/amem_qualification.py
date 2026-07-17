"""Controlled adapter for the pinned official A-MEM implementation.

The adapter is intentionally a narrow ``MemoryRuntime`` boundary.  It calls
``AgenticMemorySystem.add_note`` exactly as upstream does (and never invokes the
unused ``analyze_content`` helper), preserves the native Chroma distance order,
and records link-expanded rows without assigning them made-up relevance scores.
The official package is imported lazily so repository tests do not need the
optional dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from lhmsb.qualification.context import PublicHistoryUnit
from lhmsb.qualification.deepseek_writer import DeepSeekJSONBridge
from lhmsb.qualification.memory_runtime import (
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
from lhmsb.qualification.schema import AMemProfile, PolicyProfile

_ID_NAMESPACE = uuid.UUID("f6c4b6d5-8b8b-4a1d-99b4-5c8a7a0b4e12")
_PINNED_SOURCE_COMMIT = "ceffb860f0712bbae97b184d440df62bc910ca8d"
_OFFICIAL_MODULE = "agentic_memory.memory_system"


class AMemQualificationError(RuntimeError):
    """Typed terminal error at the A-MEM qualification boundary."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class AMemBackend(Protocol):
    """Small structural view of the official ``AgenticMemorySystem`` API."""

    memories: Mapping[str, object]

    def add_note(self, content: str, **kwargs: object) -> object: ...

    def read(self, memory_id: str) -> object | None: ...

    def update(self, memory_id: str, **kwargs: object) -> object: ...

    def delete(self, memory_id: str) -> object: ...

    def search_agentic(self, query: str, k: int = 5) -> object: ...


class AMemQualificationAdapter:
    """Trace-complete adapter around one fresh official A-MEM process."""

    capabilities = LifecycleCapabilities(
        add=True,
        update=True,
        delete=True,
        merge=False,
        links=True,
        history=True,
        resumable=False,
    )

    def __init__(
        self,
        backend: AMemBackend | object,
        *,
        namespace: str = "default",
        episode_id: str | None = None,
        candidate_k: int = 20,
        inventory_limit: int = 100_000,
        embedding_runtime: object | None = None,
        embedding: object | None = None,
        expected_source_commit: str = _PINNED_SOURCE_COMMIT,
        require_deterministic_ids: bool = False,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must be non-empty")
        if candidate_k < 1 or inventory_limit < 1:
            raise ValueError("candidate_k and inventory_limit must be positive")
        if embedding_runtime is not None and embedding is not None:
            raise ValueError("provide embedding_runtime or embedding, not both")
        self.backend = cast(AMemBackend, backend)
        self.namespace = namespace
        self.episode_id = episode_id
        self.candidate_k = candidate_k
        self.inventory_limit = inventory_limit
        self.embedding = embedding_runtime or embedding
        self.expected_source_commit = expected_source_commit
        self.require_deterministic_ids = require_deterministic_ids
        self._n_write = 0
        self._last_write_session = -1
        self._closed = False
        self._notes: dict[str, object] = {}
        self.diagnostics: list[tuple[str, object]] = []
        self._inject_common_embedding(self.embedding)
        self._validate_backend_api()

    @classmethod
    def create_live(
        cls,
        profile: AMemProfile,
        *,
        policy: PolicyProfile,
        api_key: str,
        embedding_runtime: object,
        namespace: str,
        episode_id: str,
        storage_path: str | None = None,
        source_commit: str | None = None,
        module: object | None = None,
        candidate_k: int | None = None,
        require_deterministic_ids: bool = True,
    ) -> AMemQualificationAdapter:
        """Construct the official source without a package/API fallback."""
        if profile.source_commit != _PINNED_SOURCE_COMMIT:
            raise AMemQualificationError(
                "source_pin_mismatch", "A-MEM profile source commit is not pinned"
            )
        if policy.provider != "deepseek":
            raise AMemQualificationError(
                "writer_profile_mismatch", "controlled A-MEM requires the DeepSeek writer"
            )
        module = module or _load_official_module()
        _validate_official_identity(
            module,
            expected_commit=source_commit or profile.source_commit,
        )
        writer = DeepSeekJSONBridge(
            api_key=api_key,
            model_id=policy.model_id,
            endpoint=policy.endpoint,
            timeout_seconds=policy.timeout_seconds,
            max_retries=policy.max_retries,
            max_output_tokens=512,
        )
        memory_class = getattr(module, "AgenticMemorySystem", None)
        if not callable(memory_class):
            raise AMemQualificationError(
                "upstream_api_mismatch", "official A-MEM module lacks AgenticMemorySystem"
            )
        # The upstream Chroma retriever constructs a SentenceTransformer before
        # the benchmark can inject the common TEI embedding runtime.  On an
        # offline server, the canonical model identifier cannot be resolved
        # through Hugging Face even when the model files are present locally.
        # An explicit path therefore acts only as an offline bootstrap hint;
        # the public profile and the subsequently injected embedding runtime
        # remain unchanged.
        embedding_model = os.environ.get(
            "LHMSB_AMEM_EMBEDDING_MODEL_PATH", profile.embedding_model
        )
        kwargs: dict[str, object] = {
            "model_name": embedding_model,
            # A-MEM's official controller only accepts ``openai`` or ``ollama``.
            # ``ollama`` is a constructor-only inert controller here; it is
            # replaced with the DeepSeek bridge before the first add/search call,
            # so no OpenAI key or default OpenAI URL is ever constructed.
            "llm_backend": "ollama",
            "llm_model": policy.model_id,
        }
        if storage_path is not None:
            kwargs["persist_directory"] = storage_path
        try:
            backend = memory_class(**kwargs)
        except TypeError:
            # The pinned source does not expose a persistence keyword.  Do not
            # silently switch to another implementation; retry only the exact
            # official constructor shape.
            kwargs.pop("persist_directory", None)
            try:
                backend = memory_class(**kwargs)
            except Exception as exc:
                raise AMemQualificationError("upstream_init_failure", str(exc)) from exc
        except Exception as exc:
            raise AMemQualificationError("upstream_init_failure", str(exc)) from exc
        _install_writer(backend, writer)
        return cls(
            cast(AMemBackend, backend),
            namespace=namespace,
            episode_id=episode_id,
            candidate_k=candidate_k or profile.candidate_k,
            embedding_runtime=embedding_runtime,
            expected_source_commit=profile.source_commit,
            require_deterministic_ids=require_deterministic_ids,
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
        self._closed = True
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
        retriever = getattr(self.backend, "retriever", None)
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
        writer = getattr(getattr(self.backend, "llm_controller", None), "llm", None)
        close = getattr(writer, "close", None)
        if callable(close):
            close()

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        """Add each public unit through the native ``add_note`` operation."""
        self._ensure_open()
        if isinstance(session_index, bool) or session_index < 0:
            raise ValueError("session_index must be a non-negative integer")
        if session_index < self._last_write_session:
            raise AMemQualificationError("session_order", "A-MEM writes must be ordered")
        started = time.perf_counter()
        contents = _session_contents(messages, metadata, session_index)
        events: list[MemoryMutationEvent] = []
        for ordinal, content in enumerate(contents):
            before = self._snapshot_notes()
            memory_id = self.add_note(
                content,
                session_index=session_index,
                source_ordinal=ordinal,
                metadata=metadata,
            )
            after = self._snapshot_notes()
            events.extend(
                self._diff_events(
                    before,
                    after,
                    session_index=session_index,
                    operation_seed=f"{session_index:06d}-{ordinal:06d}-{memory_id}",
                )
            )
        self._last_write_session = max(self._last_write_session, session_index)
        self._n_write += len(events)
        latency = max(0.0, time.perf_counter() - started)
        inventory = self.snapshot_inventory(checkpoint_session=session_index)
        if inventory.n_write != self._n_write:
            inventory = InventorySnapshot(
                checkpoint_session=inventory.checkpoint_session,
                n_write=self._n_write,
                n_live=inventory.n_live,
                items=inventory.items,
                store_hash=inventory.store_hash,
                backend_count=inventory.backend_count,
            )
        return WriteSessionResult(
            session_index=session_index,
            events=tuple(events),
            inventory=inventory,
            n_write=self._n_write,
            latency_seconds=latency,
            usage_events=self._writer_usage_events(),
        )

    def add_note(
        self,
        content: str,
        *,
        session_index: int = 0,
        source_ordinal: int = 0,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        """Call upstream ``add_note`` exactly once, with deterministic UUID when supported."""
        self._ensure_open()
        if not isinstance(content, str) or not content:
            raise ValueError("A-MEM note content must be non-empty text")
        kwargs: dict[str, object] = {}
        signature = _safe_signature(getattr(self.backend, "add_note", None))
        deterministic_id = self._deterministic_id(content, session_index, source_ordinal)
        if _accepts_keyword(signature, "id"):
            kwargs["id"] = deterministic_id
        elif self.require_deterministic_ids:
            raise AMemQualificationError(
                "deterministic_id_unsupported",
                "pinned A-MEM add_note does not accept benchmark-owned id",
            )
        if _accepts_keyword(signature, "timestamp"):
            kwargs["timestamp"] = f"20250101{session_index:02d}{source_ordinal:02d}"
        if metadata:
            for key in ("keywords", "context", "tags", "category"):
                if key in metadata and _accepts_keyword(signature, key):
                    kwargs[key] = metadata[key]
        try:
            raw = self.backend.add_note(content, **kwargs)
        except Exception as exc:
            raise AMemQualificationError("amem_write_failure", str(exc)) from exc
        memory_id = _row_id(raw)
        if not memory_id:
            memory_id = _row_id(self._snapshot_notes().get(deterministic_id))
        if not memory_id:
            raise AMemQualificationError(
                "malformed_upstream_response", "A-MEM add_note did not return a memory ID"
            )
        if self.require_deterministic_ids and memory_id != deterministic_id:
            raise AMemQualificationError(
                "deterministic_id_mismatch",
                f"A-MEM returned {memory_id!r}, expected {deterministic_id!r}",
            )
        self._notes[memory_id] = self._read_note(memory_id, raw)
        return memory_id

    def read(self, memory_id: str) -> MemoryObject | None:
        self._ensure_open()
        try:
            raw = self.backend.read(memory_id)
        except Exception as exc:
            raise AMemQualificationError("amem_read_failure", str(exc)) from exc
        if raw is None:
            return None
        return self._memory_object(memory_id, raw)

    def update(self, memory_id: str, **kwargs: object) -> bool:
        self._ensure_open()
        if not memory_id:
            raise ValueError("memory_id must be non-empty")
        try:
            result = self.backend.update(memory_id, **kwargs)
        except Exception as exc:
            raise AMemQualificationError("amem_update_failure", str(exc)) from exc
        if isinstance(result, bool) and not result:
            return False
        self._notes[memory_id] = self._read_note(memory_id, result)
        return True

    def delete(self, memory_id: str) -> bool:
        self._ensure_open()
        try:
            result = self.backend.delete(memory_id)
        except Exception as exc:
            raise AMemQualificationError("amem_delete_failure", str(exc)) from exc
        existed = memory_id in self._snapshot_notes() or memory_id in self._notes
        if isinstance(result, bool) and result is False:
            return False
        self._notes.pop(memory_id, None)
        return existed

    def snapshot_inventory(self, *, checkpoint_session: int) -> InventorySnapshot:
        self._ensure_open()
        if checkpoint_session < 0:
            raise ValueError("checkpoint_session must be non-negative")
        notes = self._snapshot_notes()
        if len(notes) > self.inventory_limit:
            raise AMemQualificationError("inventory_failure", "A-MEM inventory limit exceeded")
        items = tuple(
            self._memory_object(memory_id, note)
            for memory_id, note in sorted(notes.items(), key=lambda pair: pair[0])
        )
        self._record_store_diagnostics(notes)
        chroma_count = self._chroma_count()
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=self._n_write,
            n_live=len(items),
            items=items,
            store_hash=_canonical_hash([item.to_dict() for item in items]),
            # A stale/missing Chroma row is reported through ``diagnostics``;
            # do not make the normalized inventory itself impossible to decode
            # by violating its count invariant.
            backend_count=chroma_count if chroma_count == len(items) else None,
        )

    def search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch:
        self._ensure_open()
        if not isinstance(query, str):
            raise ValueError("query must be text")
        started = time.perf_counter()
        try:
            raw = self.backend.search_agentic(query, k=self.candidate_k)
        except Exception as exc:
            raise AMemQualificationError("amem_search_failure", str(exc)) from exc
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise AMemQualificationError(
                "malformed_upstream_response", "A-MEM search_agentic must return an array"
            )
        notes = self._snapshot_notes()
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for rank, row in enumerate(raw[: self.candidate_k], start=1):
            memory_id = _row_id(row)
            if not memory_id:
                raise AMemQualificationError(
                    "malformed_upstream_response", "A-MEM search row has no id"
                )
            if memory_id in seen:
                raise AMemQualificationError(
                    "duplicate_memory_id", f"A-MEM search returned duplicate ID {memory_id!r}"
                )
            seen.add(memory_id)
            note = notes.get(memory_id)
            if note is None:
                raise AMemQualificationError(
                    "candidate_outside_inventory",
                    f"A-MEM search returned unknown memory {memory_id!r}",
                )
            content = _row_text(row) or _note_text(note)
            raw_neighbor = _row_value(row, "is_neighbor", False)
            if not isinstance(raw_neighbor, bool):
                raise AMemQualificationError(
                    "malformed_upstream_response",
                    "A-MEM search is_neighbor must be boolean",
                )
            is_neighbor = raw_neighbor
            score_value = _row_value(row, "score", None)
            score = (
                None
                if is_neighbor or score_value is None
                else _finite_float(score_value, "search score")
            )
            metadata = self._candidate_metadata(note, is_neighbor=is_neighbor)
            candidates.append(
                RetrievalCandidate(
                    memory_id=memory_id,
                    content=content,
                    content_hash=sha256_text(content),
                    native_rank=rank,
                    score=score,
                    score_details=() if score is None else (("distance", score),),
                    metadata=metadata,
                    created_at=_note_timestamp(note),
                    updated_at=_note_updated(note),
                )
            )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=sha256_text(query),
            candidates=tuple(candidates),
            candidate_shortfall=len(candidates) < self.candidate_k,
            latency_seconds=max(0.0, time.perf_counter() - started),
            usage_events=self._writer_usage_events(),
        )

    def storage_footprints(self) -> tuple[StorageFootprint, ...]:
        return (
            StorageFootprint(
                component="amem_chroma",
                bytes=None,
                unavailable_reason="A-MEM Chroma boundary does not expose physical bytes",
            ),
            StorageFootprint(
                component="amem_process_store",
                bytes=None,
                unavailable_reason="A-MEM in-memory note map does not expose physical bytes",
            ),
        )

    def _validate_backend_api(self) -> None:
        for name in ("add_note", "read", "update", "delete", "search_agentic"):
            if not callable(getattr(self.backend, name, None)):
                raise AMemQualificationError(
                    "upstream_api_mismatch", f"A-MEM backend lacks required API {name}"
                )

    def _snapshot_notes(self) -> dict[str, object]:
        result = dict(self._notes)
        raw = getattr(self.backend, "memories", None)
        if isinstance(raw, Mapping):
            result.update({str(key): value for key, value in raw.items()})
        return result

    def _read_note(self, memory_id: str, raw: object) -> object:
        if raw is not None and not isinstance(raw, (str, bool, int, float)):
            return raw
        notes = self._snapshot_notes()
        if memory_id in notes:
            return notes[memory_id]
        if isinstance(raw, str):
            return {"id": memory_id, "content": raw}
        return raw

    def _memory_object(self, memory_id: str, note: object) -> MemoryObject:
        content = _note_text(note)
        metadata = self._candidate_metadata(note, is_neighbor=False)
        return MemoryObject(
            memory_id=memory_id,
            content=content,
            content_hash=sha256_text(content),
            metadata=metadata,
            created_at=_note_timestamp(note),
            updated_at=_note_updated(note),
            history_length=len(_note_history(note)),
        )

    def _candidate_metadata(
        self,
        note: object,
        *,
        is_neighbor: bool,
    ) -> tuple[tuple[str, object], ...]:
        links = _note_links(note)
        pairs: list[tuple[str, object]] = [
            ("keywords", _json_value(_note_value(note, "keywords", []))),
            ("context", _json_value(_note_value(note, "context", "General"))),
            ("tags", _json_value(_note_value(note, "tags", []))),
            ("category", _json_value(_note_value(note, "category", "Uncategorized"))),
            ("timestamp", _note_timestamp(note)),
            ("last_accessed", _note_updated(note)),
            ("retrieval_count", _json_value(_note_value(note, "retrieval_count", 0))),
            ("links", links),
            (
                "lhmsb.graph",
                {"links": links, "is_neighbor": is_neighbor},
            ),
            ("lhmsb.candidate_origin", "native_link" if is_neighbor else "native"),
            ("lhmsb.score_semantics", "unscored" if is_neighbor else "lower_is_better"),
            ("namespace", self.namespace),
        ]
        return tuple(pairs)

    def _deterministic_id(self, content: str, session_index: int, source_ordinal: int) -> str:
        episode = self.episode_id or self.namespace
        return str(
            uuid.uuid5(
                _ID_NAMESPACE,
                f"{episode}:{session_index}:{source_ordinal}:{sha256_text(content)}",
            )
        )

    def _diff_events(
        self,
        before: Mapping[str, object],
        after: Mapping[str, object],
        *,
        session_index: int,
        operation_seed: str,
    ) -> tuple[MemoryMutationEvent, ...]:
        events: list[MemoryMutationEvent] = []
        for memory_id in sorted(set(after) - set(before)):
            note = after[memory_id]
            text = _note_text(note)
            events.append(
                MemoryMutationEvent(
                    operation_id=f"amem-{operation_seed}-add-{memory_id}",
                    session_index=session_index,
                    native_event="ADD_NOTE",
                    memory_id=memory_id,
                    memory_text=text,
                    old_content_hash=None,
                    new_content_hash=sha256_text(text),
                    source="native_response",
                    latency_seconds=0.0,
                )
            )
        for memory_id in sorted(set(before) & set(after)):
            old = before[memory_id]
            new = after[memory_id]
            old_content = _note_text(old)
            new_content = _note_text(new)
            old_links = _note_links(old)
            new_links = _note_links(new)
            if old_content != new_content:
                native_event = "UPDATE"
            elif old_links != new_links:
                native_event = "LINK_UPDATE"
            else:
                continue
            events.append(
                MemoryMutationEvent(
                    operation_id=f"amem-{operation_seed}-{native_event.lower()}-{memory_id}",
                    session_index=session_index,
                    native_event=native_event,
                    memory_id=memory_id,
                    memory_text=new_content,
                    old_content_hash=sha256_text(old_content),
                    new_content_hash=sha256_text(new_content),
                    source="snapshot_diff",
                    latency_seconds=0.0,
                )
            )
        return tuple(events)

    def _record_store_diagnostics(self, notes: Mapping[str, object]) -> None:
        self.diagnostics = []
        rows = _chroma_rows(self.backend)
        if rows is None:
            self.diagnostics.append(("chroma_consistency", "unavailable"))
            self.diagnostics.append(("in_memory_vs_chroma", "unavailable"))
            self.diagnostics.append(("silent_degradation", True))
            return
        note_ids = set(notes)
        row_ids = set(rows)
        missing = sorted(note_ids - row_ids)
        extra = sorted(row_ids - note_ids)
        self.diagnostics.append(("chroma_missing_ids", missing))
        self.diagnostics.append(("chroma_extra_ids", extra))
        self.diagnostics.append(("missing_target_ids", extra))
        dangling = sorted(
            memory_id
            for memory_id, note in notes.items()
            if any(link not in note_ids for link in _note_links(note))
        )
        self.diagnostics.append(("dangling_link_ids", dangling))
        self.diagnostics.append(("link_validity", not dangling))
        self.diagnostics.append(
            ("in_memory_chroma_consistent", not missing and not extra)
        )
        self.diagnostics.append(("in_memory_vs_chroma", not missing and not extra))
        self.diagnostics.append(("silent_degradation", bool(missing or extra or dangling)))

    def _chroma_count(self) -> int | None:
        rows = _chroma_rows(self.backend)
        return None if rows is None else len(rows)

    def _inject_common_embedding(self, runtime: object | None) -> None:
        if runtime is None:
            return
        retriever = getattr(self.backend, "retriever", None)
        if retriever is None:
            self.diagnostics.append(("embedding_injection", "retriever_unavailable"))
            return
        method = getattr(runtime, "embed", None) or getattr(runtime, "embed_batch", None)
        if not callable(method):
            raise AMemQualificationError(
                "embedding_contract", "common BGE runtime must expose embed or embed_batch"
            )
        injected = False
        for target in (retriever, getattr(retriever, "collection", None)):
            if target is None:
                continue
            for name in ("embedding_function", "embedder", "embedding"):
                if hasattr(target, name):
                    setattr(target, name, method)
                    injected = True
        if not injected:
            # Keep a visible, explicit boundary even when the upstream retriever
            # has no public attribute; live preflight treats this as a diagnostic.
            retriever.lhmsb_embedding_runtime = runtime
            self.diagnostics.append(("embedding_injection", "opaque_boundary"))

    def _writer_usage_events(self) -> tuple[ProviderUsageEvent, ...]:
        writer = getattr(getattr(self.backend, "llm_controller", None), "llm", None)
        calls = getattr(writer, "calls", ())
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            return tuple(
                item for item in calls if isinstance(item, ProviderUsageEvent)
            )
        return ()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AMemQualificationError("adapter_closed", "A-MEM adapter is closed")


def _load_official_module() -> Any:
    try:
        return importlib.import_module(_OFFICIAL_MODULE)
    except ImportError as exc:
        raise AMemQualificationError(
            "official_dependency_missing",
            "official A-MEM source is required; install agentic-memory from the pinned commit",
        ) from exc


def _validate_official_identity(module: object, *, expected_commit: str) -> None:
    module_name = getattr(module, "__name__", "")
    if not isinstance(module_name, str) or not module_name.startswith("agentic_memory"):
        raise AMemQualificationError(
            "package_identity_mismatch", "A-MEM module is not the official agentic_memory package"
        )
    commit = _first_text(
        module,
        "__source_commit__",
        "SOURCE_COMMIT",
        "__commit__",
        "COMMIT_SHA",
    )
    if commit != expected_commit:
        raise AMemQualificationError(
            "source_pin_mismatch",
            f"A-MEM source commit {commit!r} != expected {expected_commit!r}",
        )
    if not callable(getattr(module, "AgenticMemorySystem", None)):
        raise AMemQualificationError(
            "upstream_api_mismatch", "official A-MEM module lacks AgenticMemorySystem"
        )


def validate_amem_source(
    module: object,
    *,
    expected_commit: str = _PINNED_SOURCE_COMMIT,
) -> None:
    """Public preflight helper for package identity/source/API verification."""
    _validate_official_identity(module, expected_commit=expected_commit)


def _install_writer(backend: object, writer: DeepSeekJSONBridge) -> None:
    controller = getattr(backend, "llm_controller", None)
    if controller is None:
        raise AMemQualificationError(
            "upstream_api_mismatch", "A-MEM backend lacks llm_controller for controlled writer"
        )
    if not hasattr(controller, "llm"):
        raise AMemQualificationError(
            "upstream_api_mismatch", "A-MEM llm_controller lacks llm boundary"
        )
    controller.llm = writer


def _safe_signature(value: object) -> inspect.Signature | None:
    if not callable(value):
        return None
    try:
        return inspect.signature(value)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(signature: inspect.Signature | None, name: str) -> bool:
    if signature is None:
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _session_contents(
    messages: Sequence[Mapping[str, str]],
    metadata: Mapping[str, object] | None,
    session_index: int,
) -> tuple[str, ...]:
    if metadata:
        raw_units = metadata.get("public_units")
        if isinstance(raw_units, Sequence) and not isinstance(raw_units, (str, bytes)):
            unit_contents: list[str] = []
            for item in raw_units:
                if isinstance(item, PublicHistoryUnit):
                    if item.source_session != session_index:
                        raise AMemQualificationError(
                            "session_mismatch",
                            "public unit session mismatch",
                        )
                    unit_contents.append(item.content)
                elif isinstance(item, Mapping) and isinstance(item.get("content"), str):
                    unit_contents.append(cast(str, item["content"]))
                else:
                    raise AMemQualificationError(
                        "invalid_public_units",
                        "A-MEM public unit is malformed",
                    )
            return tuple(unit_contents)
    contents: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            raise AMemQualificationError(
                "invalid_public_messages",
                "A-MEM message content must be text",
            )
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            contents.append(content)
            continue
        if isinstance(decoded, Mapping) and decoded.get("session_index") == session_index:
            for key in ("observations", "tool_results"):
                values = decoded.get(key, ())
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    contents.extend(value for value in values if isinstance(value, str))
        else:
            contents.append(content)
    return tuple(contents)


def _row_id(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("id", "memory_id", "uuid"):
            child = value.get(key)
            if isinstance(child, str) and child:
                return child
    for key in ("id", "memory_id", "uuid"):
        child = getattr(value, key, None)
        if isinstance(child, str) and child:
            return child
    return ""


def _row_text(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("content", "text", "memory"):
            child = value.get(key)
            if isinstance(child, str):
                return child
    for key in ("content", "text", "memory"):
        child = getattr(value, key, None)
        if isinstance(child, str):
            return child
    return ""


def _note_text(note: object) -> str:
    text = _row_text(note)
    if not text:
        raise AMemQualificationError("malformed_upstream_response", "A-MEM note has no content")
    return text


def _note_value(note: object, key: str, default: object = None) -> object:
    if isinstance(note, Mapping):
        return note.get(key, default)
    return getattr(note, key, default)


def _row_value(row: object, key: str, default: object = None) -> object:
    return _note_value(row, key, default)


def _note_links(note: object) -> tuple[str, ...]:
    raw = _note_value(note, "links", ())
    if isinstance(raw, Mapping):
        values: Sequence[object] = tuple(raw.keys())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = raw
    else:
        values = ()
    links = tuple(str(value) for value in values if isinstance(value, (str, int)))
    return tuple(dict.fromkeys(links))


def _note_history(note: object) -> tuple[object, ...]:
    raw = _note_value(note, "evolution_history", ())
    return tuple(raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()


def _note_timestamp(note: object) -> str:
    value = _note_value(note, "timestamp", "")
    return value if isinstance(value, str) else str(value)


def _note_updated(note: object) -> str:
    value = _note_value(note, "last_accessed", _note_timestamp(note))
    return value if isinstance(value, str) else str(value)


def _json_value(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AMemQualificationError("malformed_upstream_response", f"{field} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise AMemQualificationError("malformed_upstream_response", f"{field} must be finite")
    return result


def _chroma_rows(backend: object) -> dict[str, object] | None:
    retriever = getattr(backend, "retriever", None)
    if retriever is None:
        return None
    collection = getattr(retriever, "collection", None)
    getter = getattr(collection, "get", None)
    if not callable(getter):
        getter = getattr(retriever, "get", None)
    if callable(getter):
        try:
            raw = getter()
        except Exception:
            return None
        if isinstance(raw, Mapping):
            ids = raw.get("ids", ())
            if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
                return {str(memory_id): raw for memory_id in ids}
    rows = getattr(retriever, "rows", None)
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items()}
    return None


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _first_text(value: object, *names: str) -> str | None:
    for name in names:
        child = getattr(value, name, None)
        if isinstance(child, str) and child:
            return child
    return None


__all__ = [
    "AMemBackend",
    "AMemQualificationAdapter",
    "AMemQualificationError",
    "validate_amem_source",
]
