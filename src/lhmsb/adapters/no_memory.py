"""No-memory control adapter (``spec/05-systems.md`` §2.1 condition ``no_memory``).

The counterfactual baseline for Memory ROI: it stores NOTHING across sessions,
so ``search`` is always empty and ``add``/``update``/``delete`` are no-ops that
still return valid ids. It is the ``score(no_memory)`` term in
``gain = score(system) - score(no_memory)`` (``spec/02-metrics.md`` §1).

Provable statelessness is the whole point (``spec/03-protocol.md`` no-memory
policy + the plan's Metis guardrail "the no-memory control must be provably
stateless across sessions"). This class therefore holds NO instance attributes
at all: nothing it is given can be retained, so cross-session leakage is
structurally impossible, not merely unimplemented. ``add`` mints ids with
``uuid4`` (no counter, no registry), so even id generation keeps no state.
"""

from __future__ import annotations

import uuid

from lhmsb.adapters.base import MemorySystemAdapter
from lhmsb.types import SearchResult


class NoMemoryAdapter(MemorySystemAdapter):
    """A structurally-stateless memory system: every write is discarded.

    Conforms to :class:`MemorySystemAdapter` so the harness can run it through
    the identical agent loop as the real systems, but it never persists or
    returns any content. It is exempt from the storing round-trip contract
    checks (search is always empty) and is validated by a dedicated
    statelessness proof instead.
    """

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        """No setup: there is no backend to initialize."""

    def reset(self, *, user_id: str) -> None:
        """No-op: there is never any state to clear."""

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Discard ``content`` and return a fresh, valid, unique id.

        The id is generated with ``uuid4`` and is never recorded anywhere, so no
        attribute can retain the added content.
        """
        return uuid.uuid4().hex

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        """Always return an empty result: nothing was ever stored."""
        return SearchResult(results=[], total_count=0)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """No-op: there is nothing to update."""

    def delete_memory(self, memory_id: str) -> None:
        """No-op: idempotent by construction (nothing is ever stored)."""
