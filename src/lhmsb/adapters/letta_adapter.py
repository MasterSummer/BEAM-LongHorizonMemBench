"""Letta / AI-Memory-SDK self-editing-block adapter (``spec/05-systems.md`` §2.1 ``letta``).

Wraps Letta behind the synchronous :class:`MemorySystemAdapter` contract. Letta is an
agent-memory system: a message stream is ingested with :meth:`add_messages`, the agent
processes it on a server-side *run* and self-edits its persistent memory *blocks*, and a
periodic *sleeptime* pass consolidates those blocks. Three design choices let it fit the
benchmark:

1. **Lazy import.** The client SDK is imported via ``importlib`` inside
   :meth:`LettaAdapter.initialize` (trying ``ai_memory_sdk`` then ``letta_client``), so
   importing this module never requires the optional ``letta`` extra — only initializing
   the adapter does (mirrors ``chroma`` / ``graphiti``).
2. **Synchronous run bridge.** Letta processes an ingested message asynchronously on a
   server-side run; the adapter calls :meth:`wait_for_run` so the block self-edits (and
   their internal-LLM cost) are complete and captured *before* :meth:`add_memory` returns
   (no fire-and-forget that would lose tokens).
3. **Reflection is sleeptime.** Letta's ``ReflectionCapability`` (``spec/05-systems.md``
   §2.3) triggers a sleeptime/consolidation pass; its internal-LLM tokens are counted as
   memory cost under ``memory_scope`` (``spec/05-systems.md`` §4) so they are never
   silently uncounted.

Behavior is scored, not implementation (``spec/05-systems.md`` §1.1): ``add`` ->
``add_messages`` + ``wait_for_run`` (the message becomes a labelled, searchable block),
``search`` -> ``search`` (mapping each result -> :class:`MemoryEntry`), ``update`` -> an
in-place block edit keyed by the block label (so the ``memory_id`` is stable across
content changes), ``delete`` -> ``delete_block`` (graceful on a missing label), ``reset``
-> ``delete_user``. The ``memory_id`` is a deterministic content hash (``letta-`` +
sha256(content)[:16]) so repeated runs with the same add order produce identical ids
(reproducibility). The adapter is internal-LLM configurable, so native (Letta defaults)
and controlled (a caller-pinned ``pinned_model`` forwarded to ``initialize_memory`` where
supported) tracks are both supported; the chosen track is recorded. Metadata-only updates
degrade gracefully via :class:`UnsupportedOperation` — Letta blocks carry no arbitrary
metadata field (``spec/03-protocol.md`` §5.4).
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from hashlib import sha256
from time import perf_counter
from types import ModuleType
from typing import Any

from lhmsb.adapters.base import (
    MemorySystemAdapter,
    ReflectionCapability,
    UnsupportedOperation,
)
from lhmsb.cost import CostMeter
from lhmsb.types import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)

#: Optional client SDKs, tried in order inside ``initialize`` (``ai-memory-sdk`` is the
#: native surface used by the documented API; ``letta-client`` is the REST fallback).
_MODULE_CANDIDATES = ("ai_memory_sdk", "letta_client")
#: Client class names probed on whichever SDK module imports (the SDKs disagree on the
#: exact symbol; the fake test double exposes one of these).
_CLIENT_CANDIDATES = (
    "AIMemory",
    "AIMemoryClient",
    "MemoryClient",
    "AIMemorySDK",
    "Letta",
    "LettaClient",
    "RESTClient",
    "Client",
)
#: Connection/auth config keys forwarded to the client constructor when present.
_CLIENT_KWARG_KEYS = ("base_url", "token", "project")
#: In-place block-edit method names probed for ``update_memory`` (keeps the block label,
#: hence the ``memory_id``, stable); falls back to a re-add only if none exist.
_BLOCK_UPDATE_METHODS = ("update_block", "modify_block", "block_update")
#: Sleeptime/consolidation trigger names probed for ``reflect`` (guarded by ``getattr`` so
#: a backend without sleeptime degrades to a content-derived cost proxy, never crashes).
_SLEEPTIME_METHODS = ("trigger_sleeptime", "run_sleeptime", "sleeptime", "consolidate")

_DEFAULT_BLOCK_LABEL = "main"
_ID_PREFIX = "letta-"

_STOPWORDS = frozenset(
    {"the", "is", "are", "was", "were", "of", "a", "an", "to", "in", "on", "and", "for", "no"}
)


class LettaSetupError(RuntimeError):
    """Raised when no usable Letta client SDK / client class can be located."""


def _load_letta() -> ModuleType:
    """Import the first available Letta client SDK, lazily (``spec/05-systems.md`` §2.1)."""
    for name in _MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(
        "LettaAdapter requires the optional 'letta' extra: pip install 'lhmsb[letta]' "
        f"(provides one of {_MODULE_CANDIDATES}). See https://github.com/letta-ai/ai-memory-sdk."
    )


def _resolve_client_class(module: ModuleType) -> Any:
    """Find the client class exposed by ``module`` (the SDKs name it differently)."""
    for name in _CLIENT_CANDIDATES:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise LettaSetupError(
        f"LettaAdapter could not find a client class on {module.__name__!r}; "
        f"expected one of {_CLIENT_CANDIDATES}."
    )


def _memory_id(content: str) -> str:
    """Deterministic, content-derived block id (reproducible across identical runs)."""
    return _ID_PREFIX + sha256(content.encode("utf-8")).hexdigest()[:16]


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _tokens(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if len(tok) >= 3 and tok not in _STOPWORDS]


def _matches(query: str, text: str) -> bool:
    """Token-overlap relevance: any salient query token present in ``text``."""
    text_lower = text.lower()
    return any(tok in text_lower for tok in _tokens(query))


def _first_attr(raw: Any, names: tuple[str, ...]) -> str:
    """First non-empty value among ``names`` on a dict or object, as ``str`` (else "")."""
    for name in names:
        value = raw.get(name) if isinstance(raw, dict) else getattr(raw, name, None)
        if value:
            return str(value)
    return ""


def _run_text(run: Any) -> str:
    """Best-effort text a run produced (for output-token counting when no usage is given)."""
    for attr in ("output", "summary", "text"):
        value = getattr(run, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _run_completion_tokens(run: Any, fallback_text: str) -> int:
    """Output tokens of a run's internal-LLM work (block self-edits), always >= 1.

    Prefers the run's native ``usage.completion_tokens`` (real Letta exposes it); falls
    back to counting any text the run produced, then to a content-derived proxy so the
    internal-LLM output cost is never recorded as zero (``spec/05-systems.md`` §4)."""
    usage = getattr(run, "usage", None)
    completion = getattr(usage, "completion_tokens", None) if usage is not None else None
    if isinstance(completion, int) and completion > 0:
        return completion
    words = len(_run_text(run).split())
    if words > 0:
        return words
    return max(1, len(fallback_text.split()) // 2)


def _result_to_entry(raw: Any, rank: int) -> MemoryEntry:
    """Map a Letta search result (block/passage) to a :class:`MemoryEntry`.

    The stored block label -> ``memory_id`` (stable across content edits), the block value
    -> ``content``, and the descending rank -> a relevance score."""
    created = _first_attr(raw, ("created_at", "created"))
    updated = _first_attr(raw, ("updated_at", "modified_at", "last_modified")) or created
    return MemoryEntry(
        memory_id=_first_attr(raw, ("memory_id", "label", "id", "block_id")),
        content=_first_attr(raw, ("content", "value", "text")),
        metadata=None,
        created_at=created,
        updated_at=updated,
        score=1.0 / (1.0 + rank),
    )


class LettaAdapter(MemorySystemAdapter, ReflectionCapability):
    """Agent self-editing-block memory backed by Letta. Supports reflection (sleeptime)."""

    def __init__(self, cost_meter: CostMeter) -> None:
        self.cost_meter = cost_meter
        self._client: Any = None
        self._subject_id = ""
        self._track = "native"
        self._pinned_model: object = None
        # Adapter-owned mirror of {memory_id: content} for summarize/reflect proxies and
        # to keep latency/cost bookkeeping independent of the backend's opaque state.
        self._content: dict[str, str] = {}

    @property
    def track(self) -> str:
        """Which comparison track this adapter was initialized for (native/controlled)."""
        return self._track

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        module = _load_letta()
        client_cls = _resolve_client_class(module)
        self._track = _as_str(config.get("track"), "native")
        self._pinned_model = config.get("pinned_model")
        self._subject_id = user_id
        self._content = {}

        client_kwargs = {
            key: config[key] for key in _CLIENT_KWARG_KEYS if config.get(key) is not None
        }
        try:
            self._client = client_cls(**client_kwargs)
        except Exception as exc:
            raise LettaSetupError(f"failed to construct the Letta client: {exc}") from exc

        self._client.initialize_subject(subject_id=user_id)
        memory_kwargs: dict[str, object] = {"label": _DEFAULT_BLOCK_LABEL}
        # Controlled track: pin the shared model where Letta allows it (block-manager LLM).
        if self._track == "controlled" and isinstance(self._pinned_model, str):
            memory_kwargs["model"] = self._pinned_model
        self._client.initialize_memory(**memory_kwargs)

    def reset(self, *, user_id: str) -> None:
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._client.delete_user(user_id=user_id)
        self._content.clear()

    # ------------------------------------------------------------------ #
    # Memory operations
    # ------------------------------------------------------------------ #

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        memory_id = _memory_id(content)
        started = perf_counter()
        with self.cost_meter.memory_scope():
            run = self._client.add_messages(messages=[{"role": "user", "content": content}])
            self._client.wait_for_run(run)
            # Letta's block self-edits run an internal LLM: input ~ ingested words,
            # output ~ the run's completion tokens (never uncounted; §4).
            self.cost_meter.add_memory_internal_tokens(
                len(content.split()), _run_completion_tokens(run, content)
            )
        self.cost_meter.record_latency("write", (perf_counter() - started) * 1000.0)
        self.cost_meter.add_storage_bytes(len(content.encode("utf-8")))
        self._content[memory_id] = content
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
        self.cost_meter.incr_retrieval()
        started = perf_counter()
        with self.cost_meter.memory_scope():
            results = self._client.search(user_id=user_id, query=query)
        self.cost_meter.record_latency("retrieval", (perf_counter() - started) * 1000.0)
        result_list = list(results) if results is not None else []
        entries = [_result_to_entry(raw, rank) for rank, raw in enumerate(result_list)]
        return SearchResult(results=entries[:top_k], total_count=len(result_list))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if content is None:
            # Letta blocks have no arbitrary metadata field: degrade gracefully (§5.4).
            raise UnsupportedOperation(
                "update_memory", condition="metadata-only update (Letta blocks store no metadata)"
            )
        started = perf_counter()
        with self.cost_meter.memory_scope():
            self._update_block(memory_id, content)
            self.cost_meter.add_memory_internal_tokens(
                len(content.split()), max(1, len(content.split()) // 2)
            )
        self.cost_meter.record_latency("update", (perf_counter() - started) * 1000.0)
        self._content[memory_id] = content

    def delete_memory(self, memory_id: str) -> None:
        self._content.pop(memory_id, None)
        with contextlib.suppress(Exception), self.cost_meter.memory_scope():
            self._client.delete_block(label=memory_id)

    # ------------------------------------------------------------------ #
    # Reflection (sleeptime consolidation)
    # ------------------------------------------------------------------ #

    def reflect(self, *, user_id: str, session_id: str | None = None) -> None:
        """Trigger Letta's sleeptime consolidation; count its internal-LLM tokens.

        Per ``spec/05-systems.md`` §4 reflection/consolidation cost is memory-system cost,
        so the tokens land in ``mem_internal_*`` under ``memory_scope``. If the backend
        exposes no sleeptime hook the cost falls back to a content-derived proxy (never
        zero, never uncounted)."""
        corpus = " ".join(self._content.values())
        in_tokens = max(1, len(corpus.split()))
        with self.cost_meter.memory_scope():
            run = self._trigger_sleeptime()
            if run is not None:
                with contextlib.suppress(Exception):
                    self._client.wait_for_run(run)
                out_tokens = _run_completion_tokens(run, corpus)
            else:
                out_tokens = max(1, in_tokens // 2)
            self.cost_meter.add_memory_internal_tokens(in_tokens, out_tokens)

    def summarize(
        self, *, user_id: str, session_id: str | None = None, query: str | None = None
    ) -> str:
        """Concatenate stored memory (optionally filtered to ``query``); small internal cost."""
        parts = [
            content
            for content in self._content.values()
            if query is None or _matches(query, content)
        ]
        summary = " ".join(parts)
        with self.cost_meter.memory_scope():
            self.cost_meter.add_memory_internal_tokens(max(1, len(summary.split())), 1)
        return summary

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _update_block(self, label: str, value: str) -> None:
        """In-place block edit keyed by ``label`` so the ``memory_id`` stays stable.

        Probes the SDK's block-edit method names; only if none exist does it re-add the
        message (a last resort for backends without a labelled block-edit API)."""
        for name in _BLOCK_UPDATE_METHODS:
            edit = getattr(self._client, name, None)
            if callable(edit):
                edit(label=label, value=value)
                return
        run = self._client.add_messages(messages=[{"role": "user", "content": value}])
        self._client.wait_for_run(run)

    def _trigger_sleeptime(self) -> Any:
        """Invoke the first available sleeptime/consolidation hook, or ``None`` if absent."""
        for name in _SLEEPTIME_METHODS:
            trigger = getattr(self._client, name, None)
            if callable(trigger):
                return trigger(subject_id=self._subject_id)
        return None
