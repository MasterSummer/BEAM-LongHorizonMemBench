"""Abstract memory-system adapter interface + capability introspection.

Canonical source: ``spec/05-systems.md`` §1.1 (required method signatures),
§1.2 (``Capabilities`` + graceful degradation), §1.3 (optional capability
mixins). ``spec/03-protocol.md`` §5.4 defines the ``UnsupportedOperation``
graceful-degradation policy (logged, non-fatal — never crash, never silently
ignore).

The adapter is the ONLY path through which the harness reads or writes memory.
Downstream adapters (tasks 12-16) and the harness (task 9) import
``MemorySystemAdapter`` and implement these EXACT signatures. The base class
contains NO backend-specific logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from lhmsb.types import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)


class UnsupportedOperation(Exception):  # noqa: N818  spec-canonical name (spec/05-systems §1.2)
    """Raised when an adapter receives a call its backend does not support.

    Per ``spec/03-protocol.md`` §5.4 this is a well-defined, LOGGED but
    NON-FATAL exception: the harness catches it, records the capability gap, and
    proceeds. It is logged on construction so the gap is always observable even
    if a caller forgets to log it.
    """

    def __init__(self, operation: str, *, condition: str | None = None) -> None:
        self.operation = operation
        self.condition = condition
        where = f" (condition={condition})" if condition else ""
        message = f"unsupported memory operation: {operation!r}{where}"
        super().__init__(message)
        logger.warning(message)


@dataclass(frozen=True)
class Capabilities:
    """Which optional operations a backend supports (``spec/05-systems.md`` §1.2).

    The harness queries ``get_capabilities()`` before invoking optional ops;
    calling an unsupported op raises :class:`UnsupportedOperation`.
    """

    supports_add: bool = True
    supports_search: bool = True
    supports_update: bool = True
    supports_delete: bool = True
    supports_reset: bool = True
    supports_sessions: bool = False
    supports_reflection: bool = False
    supports_forgetting: bool = False


class MemorySystemAdapter(ABC):
    """Abstract base for every memory system under test.

    Concrete adapters implement the six required methods with the EXACT
    signatures from ``spec/05-systems.md`` §1.1. Optional lifecycle operations
    are exposed by inheriting the mixins in this module; ``get_capabilities()``
    derives the optional flags from mixin membership by default.
    """

    @abstractmethod
    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        """Set up the backend for a user. Called once per user before the first
        session. ``session_id=None`` means no session scope yet."""
        ...

    @abstractmethod
    def reset(self, *, user_id: str) -> None:
        """Delete ALL memory for a user. Idempotent; used between episodes."""
        ...

    @abstractmethod
    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Ingest content; return a unique, stable ``memory_id``. Internal
        LLM/embedding calls MUST be wrapped in ``memory_scope()`` (cost
        instrumentation, task 6)."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        """Retrieve relevance-ranked memories for a query. Internal
        LLM/embedding calls MUST be wrapped in ``memory_scope()``."""
        ...

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Update content and/or metadata of an existing entry. ``None`` keeps
        the existing value; at least one of ``content``/``metadata`` must be
        provided."""
        ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        """Remove a memory entry; it must no longer appear in search results.
        Idempotent — deleting a non-existent entry is a no-op, not an error.
        BEHAVIOR is scored, not implementation (removal | tombstone | edge
        invalidation | retrieval-filter all acceptable)."""
        ...

    def get_capabilities(self) -> Capabilities:
        """Advertise supported operations.

        Core ops default to supported; optional flags are derived from mixin
        membership. Concrete adapters override this to mark a core op
        unsupported (and then raise :class:`UnsupportedOperation` when it is
        called)."""
        return Capabilities(
            supports_sessions=isinstance(self, SessionCapability),
            supports_reflection=isinstance(self, ReflectionCapability),
            supports_forgetting=isinstance(self, ForgettingCapability),
        )


class ReflectionCapability(ABC):
    """Optional: backends that support consolidation / self-reorganization."""

    @abstractmethod
    def reflect(self, *, user_id: str, session_id: str | None = None) -> None:
        """Trigger a reflection/consolidation pass. Internal LLM tokens MUST be
        counted under ``memory_scope()``."""
        ...

    @abstractmethod
    def summarize(
        self, *, user_id: str, session_id: str | None = None, query: str | None = None
    ) -> str:
        """Produce a summary of stored memories, optionally scoped to a query."""
        ...


class ForgettingCapability(ABC):
    """Optional: backends with explicit decay / forgetting mechanisms."""

    @abstractmethod
    def apply_decay(self, *, user_id: str, **params: object) -> None:
        """Apply a forgetting/decay step (lower scores, archive, or delete)."""
        ...


class SessionCapability(ABC):
    """Optional: backends with explicit session/thread grouping."""

    @abstractmethod
    def list_sessions(self, *, user_id: str) -> list[str]:
        """Return all session IDs for a user."""
        ...

    @abstractmethod
    def get_session_memories(self, *, user_id: str, session_id: str) -> list[MemoryEntry]:
        """Return all memory entries scoped to a session."""
        ...

    @abstractmethod
    def promote_session(self, *, user_id: str, session_id: str) -> None:
        """Promote session-scoped memories to global/user scope."""
        ...
