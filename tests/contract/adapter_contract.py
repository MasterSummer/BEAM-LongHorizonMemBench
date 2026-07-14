"""Reusable, parametrizable contract suite for ``MemorySystemAdapter``.

This is the single source of truth for *behavioral* conformance of any memory
adapter. Every adapter task (12-16) plugs its own adapter into this suite and
gets the same guarantees verified. The suite scores BEHAVIOR, not implementation
(per spec/05-systems.md §1.1): how a backend stores/deletes is irrelevant, only
that the observable contract holds.

Two consumption styles are provided:

1. **Imperative helper** — call from anywhere (even outside pytest)::

       from contract.adapter_contract import run_contract_suite
       run_contract_suite(lambda: MyAdapter(...))

   Runs every check in order and raises ``AssertionError`` on the first
   violation (so a non-conforming adapter fails loudly with a clear message).

2. **pytest base class** — subclass and override ``adapter_factory`` so each
   contract property surfaces as its OWN test case (a broken adapter fails the
   exact violated check, not a monolithic pass/fail)::

       class TestMyAdapterContract(AdapterContractTests):
           @staticmethod
           def adapter_factory() -> MemorySystemAdapter:
               return MyAdapter(...)

Scope note: these checks verify a STORING + RETRIEVING backend (chroma, mem0,
letta, graphiti, cognee, fake_perfect). The ``no_memory`` control deliberately
stores nothing (search always empty) and is exempt from the round-trip checks;
it gets a dedicated statelessness test in its own adapter task instead.
"""

from __future__ import annotations

from collections.abc import Callable

from lhmsb.adapters import (
    Capabilities,
    ForgettingCapability,
    MemorySystemAdapter,
    ReflectionCapability,
    SessionCapability,
    UnsupportedOperation,
)
from lhmsb.types import SearchResult

#: A zero-argument callable producing a *fresh, uninitialized* adapter instance.
AdapterFactory = Callable[[], MemorySystemAdapter]

_UID = "contract-user"
_SID = "contract-session"


def _initialized(factory: AdapterFactory) -> MemorySystemAdapter:
    """Build a fresh adapter, initialize + reset it to a clean state."""
    adapter = factory()
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    return adapter


def _ids(result: SearchResult) -> set[str]:
    return {entry.memory_id for entry in result.results}


# --------------------------------------------------------------------------- #
# Individual contract checks. Each takes a factory, builds a fresh adapter,    #
# and asserts ONE behavioral property. Plain ``assert`` / raised              #
# ``AssertionError`` so failures are catchable and the message is clear.      #
# --------------------------------------------------------------------------- #


def check_add_search_roundtrip(factory: AdapterFactory) -> None:
    """add(content) then search() must return that memory (round-trip)."""
    adapter = _initialized(factory)
    memory_id = adapter.add_memory(
        "The capital of France is Paris.", user_id=_UID, session_id=_SID
    )
    assert isinstance(memory_id, str) and memory_id, "add_memory must return a non-empty id"

    result = adapter.search("capital of France", user_id=_UID, session_id=_SID, top_k=10)
    assert isinstance(result, SearchResult), "search must return a SearchResult"
    assert memory_id in _ids(result), (
        "round-trip failed: the added memory was not returned by a matching search"
    )
    matched = next(e for e in result.results if e.memory_id == memory_id)
    assert "Paris" in matched.content, "round-trip returned the wrong content"


def check_search_respects_top_k(factory: AdapterFactory) -> None:
    """search(top_k=k) must never return more than k results."""
    adapter = _initialized(factory)
    for i in range(5):
        adapter.add_memory(f"shared keyword entry number {i}", user_id=_UID, session_id=_SID)

    result = adapter.search("shared keyword", user_id=_UID, session_id=_SID, top_k=2)
    assert len(result.results) <= 2, (
        f"search ignored top_k: requested 2, got {len(result.results)} results"
    )
    assert result.total_count >= len(result.results), (
        "total_count must be >= number of returned results"
    )


def check_update_changes_content(factory: AdapterFactory) -> None:
    """update_memory(content=...) must change what search returns."""
    adapter = _initialized(factory)
    memory_id = adapter.add_memory(
        "alpha distinctive marker", user_id=_UID, session_id=_SID
    )
    adapter.update_memory(memory_id, content="beta distinctive marker")

    result = adapter.search("distinctive marker", user_id=_UID, session_id=_SID, top_k=10)
    assert memory_id in _ids(result), "updated memory disappeared from search"
    matched = next(e for e in result.results if e.memory_id == memory_id)
    assert matched.content == "beta distinctive marker", (
        f"update did not change content: got {matched.content!r}"
    )


def check_delete_removes(factory: AdapterFactory) -> None:
    """delete_memory() must remove the entry from subsequent search results."""
    adapter = _initialized(factory)
    memory_id = adapter.add_memory(
        "ephemeral fact about quokkas", user_id=_UID, session_id=_SID
    )
    before = adapter.search("ephemeral quokkas", user_id=_UID, session_id=_SID, top_k=10)
    assert memory_id in _ids(before), "precondition failed: memory not searchable before delete"

    adapter.delete_memory(memory_id)

    after = adapter.search("ephemeral quokkas", user_id=_UID, session_id=_SID, top_k=10)
    assert memory_id not in _ids(after), (
        "delete violated: a deleted memory is still returned by search"
    )


def check_delete_is_idempotent(factory: AdapterFactory) -> None:
    """Deleting a non-existent / already-deleted id is a no-op, not an error."""
    adapter = _initialized(factory)
    memory_id = adapter.add_memory("transient note", user_id=_UID, session_id=_SID)
    adapter.delete_memory(memory_id)
    # Second delete + unknown id must NOT raise.
    adapter.delete_memory(memory_id)
    adapter.delete_memory("does-not-exist-id")


def check_reset_clears(factory: AdapterFactory) -> None:
    """reset(user_id) must clear ALL memory for the user."""
    adapter = _initialized(factory)
    for i in range(3):
        adapter.add_memory(f"resettable entry {i} token", user_id=_UID, session_id=_SID)

    adapter.reset(user_id=_UID)

    result = adapter.search("resettable token", user_id=_UID, session_id=_SID, top_k=10)
    assert result.results == [], "reset did not clear search results"
    assert result.total_count == 0, "reset did not clear total_count"


def check_capabilities_introspection(factory: AdapterFactory) -> None:
    """get_capabilities() returns a Capabilities consistent with mixin membership."""
    adapter = _initialized(factory)
    caps = adapter.get_capabilities()
    assert isinstance(caps, Capabilities), "get_capabilities must return a Capabilities"

    # Optional-capability flags MUST agree with the mixins the adapter inherits.
    assert caps.supports_reflection == isinstance(adapter, ReflectionCapability), (
        "supports_reflection disagrees with ReflectionCapability membership"
    )
    assert caps.supports_forgetting == isinstance(adapter, ForgettingCapability), (
        "supports_forgetting disagrees with ForgettingCapability membership"
    )
    assert caps.supports_sessions == isinstance(adapter, SessionCapability), (
        "supports_sessions disagrees with SessionCapability membership"
    )


def check_unsupported_op_degrades(factory: AdapterFactory) -> None:
    """Unsupported operations degrade gracefully via ``UnsupportedOperation``.

    - A CORE op the adapter declares unsupported must raise ``UnsupportedOperation``
      when called (never crash with a different error, never silently succeed).
    - An OPTIONAL op is EITHER supported (capability True and method present) OR
      gracefully unavailable (capability False; if a method exists it raises
      ``UnsupportedOperation``).
    """
    adapter = _initialized(factory)
    caps = adapter.get_capabilities()

    # --- Core ops: declared-unsupported must raise UnsupportedOperation. ---
    core_probes: list[tuple[bool, str, Callable[[MemorySystemAdapter], object]]] = [
        (caps.supports_add, "add_memory", lambda a: a.add_memory("c", user_id=_UID)),
        (caps.supports_search, "search", lambda a: a.search("q", user_id=_UID)),
        (
            caps.supports_update,
            "update_memory",
            lambda a: a.update_memory("mid", content="x"),
        ),
        (caps.supports_delete, "delete_memory", lambda a: a.delete_memory("mid")),
        (caps.supports_reset, "reset", lambda a: a.reset(user_id=_UID)),
    ]
    for supported, name, invoke in core_probes:
        if not supported:
            _assert_raises_unsupported(adapter, name, invoke)

    # Optional ops invoked via getattr: avoids static references to mixin-only
    # attributes absent from the base ABC, keeping the suite type-clean.
    optional_probes: list[tuple[bool, type, list[tuple[str, dict[str, object]]]]] = [
        (
            caps.supports_reflection,
            ReflectionCapability,
            [("reflect", {"user_id": _UID}), ("summarize", {"user_id": _UID})],
        ),
        (
            caps.supports_forgetting,
            ForgettingCapability,
            [("apply_decay", {"user_id": _UID})],
        ),
        (
            caps.supports_sessions,
            SessionCapability,
            [("list_sessions", {"user_id": _UID})],
        ),
    ]
    for supported, mixin, ops in optional_probes:
        is_mixin = isinstance(adapter, mixin)
        if supported:
            assert is_mixin, f"capability claims support but adapter is not a {mixin.__name__}"
            for name, _ in ops:
                assert callable(getattr(adapter, name, None)), (
                    f"supported capability {mixin.__name__} is missing method {name}()"
                )
        else:
            # Not supported: either no method at all, or it raises UnsupportedOperation.
            for name, kwargs in ops:
                method = getattr(adapter, name, None)
                if callable(method):
                    try:
                        method(**kwargs)
                    except UnsupportedOperation:
                        continue
                    raise AssertionError(
                        f"{name}() is declared unsupported but did not raise UnsupportedOperation"
                    )


def _assert_raises_unsupported(
    adapter: MemorySystemAdapter,
    name: str,
    invoke: Callable[[MemorySystemAdapter], object],
) -> None:
    try:
        invoke(adapter)
    except UnsupportedOperation:
        return
    raise AssertionError(
        f"{name}() is declared unsupported but did not raise UnsupportedOperation"
    )


#: Ordered, named contract checks. Used by ``run_contract_suite`` and for pytest
#: parametrization so a broken adapter fails on the exact violated property.
CONTRACT_CHECKS: list[tuple[str, Callable[[AdapterFactory], None]]] = [
    ("add_search_roundtrip", check_add_search_roundtrip),
    ("search_respects_top_k", check_search_respects_top_k),
    ("update_changes_content", check_update_changes_content),
    ("delete_removes", check_delete_removes),
    ("delete_is_idempotent", check_delete_is_idempotent),
    ("reset_clears", check_reset_clears),
    ("capabilities_introspection", check_capabilities_introspection),
    ("unsupported_op_degrades", check_unsupported_op_degrades),
]


def run_contract_suite(adapter_factory: AdapterFactory) -> None:
    """Run the FULL contract suite against ``adapter_factory``.

    Reusable entry point for any adapter (tasks 12-16). Runs every check in
    :data:`CONTRACT_CHECKS` in order; raises ``AssertionError`` on the first
    violation with a message naming the failing check.
    """
    for name, check in CONTRACT_CHECKS:
        try:
            check(adapter_factory)
        except AssertionError as exc:  # re-raise with the check name for clarity
            raise AssertionError(f"contract check {name!r} failed: {exc}") from exc


class AdapterContractTests:
    """pytest mixin: subclass and override ``adapter_factory``.

    Each contract property is its own ``test_*`` method, so a non-conforming
    adapter fails the *specific* violated check. This class name does not start
    with ``Test`` so pytest does NOT collect the base directly — only concrete
    ``Test*`` subclasses are collected.
    """

    @staticmethod
    def adapter_factory() -> MemorySystemAdapter:  # pragma: no cover - overridden
        raise NotImplementedError("subclasses must override adapter_factory()")

    def test_add_search_roundtrip(self) -> None:
        check_add_search_roundtrip(self.adapter_factory)

    def test_search_respects_top_k(self) -> None:
        check_search_respects_top_k(self.adapter_factory)

    def test_update_changes_content(self) -> None:
        check_update_changes_content(self.adapter_factory)

    def test_delete_removes(self) -> None:
        check_delete_removes(self.adapter_factory)

    def test_delete_is_idempotent(self) -> None:
        check_delete_is_idempotent(self.adapter_factory)

    def test_reset_clears(self) -> None:
        check_reset_clears(self.adapter_factory)

    def test_capabilities_introspection(self) -> None:
        check_capabilities_introspection(self.adapter_factory)

    def test_unsupported_op_degrades(self) -> None:
        check_unsupported_op_degrades(self.adapter_factory)
