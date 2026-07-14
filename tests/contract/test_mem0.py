"""Contract + behavior tests for the Mem0 adapter (task 13).

``mem0`` (``mem0ai``) is NOT installed in CI, so these tests inject an in-memory
``FakeMem0Memory`` via ``sys.modules["mem0"]`` patching (the adapter imports it
lazily inside ``initialize``). The fake mirrors the real API surface the adapter
uses — ``Memory()`` / ``Memory.from_config(...)``, ``add``, ``search`` (token-overlap
ranked), ``update``, ``delete``, ``delete_all`` — and returns Mem0's current
``{"results": [...]}`` response shape. It exercises the ADAPTER, not Mem0 itself.

The single live test is gated behind ``LHMSB_LIVE_MEM0=1`` (real ``mem0ai`` + an
LLM/embedder backend).
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable

import pytest

from contract.adapter_contract import run_contract_suite
from lhmsb.adapters import ReflectionCapability, SessionCapability, UnsupportedOperation
from lhmsb.adapters.mem0_adapter import Mem0Adapter
from lhmsb.cost import CostMeter

_UID = "mem0-user"
_SID = "mem0-session"


class FakeMem0Memory:
    """In-memory async-free stand-in for ``mem0.Memory`` (offline contract tests)."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, object]] = {}
        self._seq = 0

    @classmethod
    def from_config(cls, config: dict[str, object]) -> FakeMem0Memory:
        instance = cls()
        instance.config = config
        return instance

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        content = " ".join(str(message.get("content", "")) for message in messages)
        self._seq += 1
        memory_id = f"fake-mem0-{user_id}-{self._seq}"
        stamp = f"2025-01-01T00:00:{self._seq:02d}Z"
        self._store[memory_id] = {
            "id": memory_id,
            "memory": content,
            "user_id": user_id,
            "metadata": dict(metadata) if metadata else None,
            "created_at": stamp,
            "updated_at": stamp,
        }
        return {"results": [{"id": memory_id, "memory": content, "event": "ADD"}]}

    def search(self, query: str, *, user_id: str, limit: int = 10) -> dict[str, object]:
        terms = set(query.lower().split())

        def overlap(row: dict[str, object]) -> int:
            return len(terms & set(str(row["memory"]).lower().split()))

        rows = [
            row
            for row in self._store.values()
            if row["user_id"] == user_id and overlap(row) > 0
        ]
        rows.sort(key=overlap, reverse=True)
        results = [
            {
                "id": row["id"],
                "memory": row["memory"],
                "metadata": row["metadata"],
                "score": float(1 + overlap(row)),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows[:limit]
        ]
        return {"results": results}

    def update(self, memory_id: str, *, data: str) -> dict[str, object]:
        if memory_id in self._store:
            self._store[memory_id]["memory"] = data
            self._store[memory_id]["updated_at"] = "2025-01-02T00:00:00Z"
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> dict[str, object]:
        self._store.pop(memory_id, None)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, *, user_id: str) -> dict[str, object]:
        for memory_id in [mid for mid, row in self._store.items() if row["user_id"] == user_id]:
            del self._store[memory_id]
        return {"message": "Memories deleted successfully!"}


InjectFn = Callable[..., types.ModuleType]


@pytest.fixture
def inject_mem0(monkeypatch: pytest.MonkeyPatch) -> InjectFn:
    """Install a fake ``mem0`` module exposing the given ``Memory`` class."""

    def _inject(memory_cls: type = FakeMem0Memory) -> types.ModuleType:
        module = types.ModuleType("mem0")
        module.__dict__["Memory"] = memory_cls
        monkeypatch.setitem(sys.modules, "mem0", module)
        return module

    return _inject


def _adapter(meter: CostMeter | None = None) -> Mem0Adapter:
    return Mem0Adapter(meter if meter is not None else CostMeter())


# --------------------------------------------------------------------------- #
# Generic contract suite (task 5) against the fake.
# --------------------------------------------------------------------------- #
def test_mem0_passes_full_contract(inject_mem0: InjectFn) -> None:
    inject_mem0()
    run_contract_suite(_adapter)


# --------------------------------------------------------------------------- #
# Cost instrumentation: internal extraction-LLM + embedding tokens are counted.
# --------------------------------------------------------------------------- #
def test_add_counts_internal_llm_tokens(inject_mem0: InjectFn) -> None:
    inject_mem0()
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)

    adapter.add_memory("Acme Corp signed a partnership with Globex in Berlin.", user_id=_UID)
    adapter.search("Acme partnership", user_id=_UID)

    cost = meter.to_cost_vector()
    assert cost.mem_internal_in_tokens > 0, "internal extraction-LLM input tokens must be counted"
    assert cost.mem_internal_out_tokens > 0, "extracted-fact output tokens must be counted"
    assert cost.embedding_tokens > 0, "embedding tokens (add + search) must be counted"
    assert cost.num_retrieval_calls >= 1, "search() must increment the retrieval counter"
    assert cost.write_latency_ms >= 0.0 and cost.retrieval_latency_ms >= 0.0
    assert not meter.has_unscoped(), "all internal cost must land inside memory_scope"


def test_strict_meter_no_uncounted_calls(inject_mem0: InjectFn) -> None:
    inject_mem0()
    meter = CostMeter(strict_instrumentation=True)
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a fact under strict accounting", user_id=_UID)
    adapter.update_memory(memory_id, content="an updated fact under strict accounting")
    adapter.search("strict accounting", user_id=_UID)
    adapter.delete_memory(memory_id)
    assert not meter.has_unscoped()


# --------------------------------------------------------------------------- #
# Native vs controlled track (model pinning forwarded where supported).
# --------------------------------------------------------------------------- #
def test_native_track_uses_default_memory(inject_mem0: InjectFn) -> None:
    inject_mem0()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    assert adapter.track == "native"


def test_controlled_track_pins_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _ConfigCapturingMem0(FakeMem0Memory):
        @classmethod
        def from_config(cls, config: dict[str, object]) -> _ConfigCapturingMem0:
            captured["config"] = config
            return cls()

    module = types.ModuleType("mem0")
    module.__dict__["Memory"] = _ConfigCapturingMem0
    monkeypatch.setitem(sys.modules, "mem0", module)

    adapter = _adapter()
    adapter.initialize(user_id=_UID, track="controlled", pinned_model="shared/open-weights-model")

    assert adapter.track == "controlled"
    config = captured["config"]
    assert isinstance(config, dict)
    llm = config["llm"]
    assert isinstance(llm, dict)
    assert llm["model"] == "shared/open-weights-model", "controlled track must pin the shared model"

    # The controlled-track adapter is still a working store.
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a controlled-track fact", user_id=_UID)
    assert memory_id
    result = adapter.search("controlled-track fact", user_id=_UID, top_k=5)
    assert memory_id in {entry.memory_id for entry in result.results}


def test_controlled_track_merges_caller_mem0_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _ConfigCapturingMem0(FakeMem0Memory):
        @classmethod
        def from_config(cls, config: dict[str, object]) -> _ConfigCapturingMem0:
            captured["config"] = config
            return cls()

    module = types.ModuleType("mem0")
    module.__dict__["Memory"] = _ConfigCapturingMem0
    monkeypatch.setitem(sys.modules, "mem0", module)

    adapter = _adapter()
    adapter.initialize(
        user_id=_UID,
        track="controlled",
        pinned_model="shared/open-weights-model",
        mem0_config={"embedder": {"provider": "fake"}, "llm": {"temperature": 0.0}},
    )

    config = captured["config"]
    assert isinstance(config, dict)
    embedder = config["embedder"]
    llm = config["llm"]
    assert isinstance(embedder, dict) and embedder["provider"] == "fake"
    assert isinstance(llm, dict)
    assert llm["model"] == "shared/open-weights-model", "pinned model overrides into caller llm cfg"
    assert llm["temperature"] == 0.0, "caller llm settings are preserved"


# --------------------------------------------------------------------------- #
# Lazy import: importing/constructing the adapter must NOT import mem0.
# --------------------------------------------------------------------------- #
def test_mem0_imported_lazily_not_until_initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "mem0", raising=False)

    adapter = Mem0Adapter(CostMeter())
    assert "mem0" not in sys.modules, "mem0 must not be imported merely by constructing the adapter"

    module = types.ModuleType("mem0")
    module.__dict__["Memory"] = FakeMem0Memory
    monkeypatch.setitem(sys.modules, "mem0", module)
    adapter.initialize(user_id=_UID)
    assert "mem0" in sys.modules, "initialize() must resolve the lazily-imported mem0"


# --------------------------------------------------------------------------- #
# Metadata-only update degrades gracefully (Mem0 has no metadata-only edit path).
# --------------------------------------------------------------------------- #
def test_metadata_only_update_raises_unsupported(inject_mem0: InjectFn) -> None:
    inject_mem0()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a fact", user_id=_UID)
    with pytest.raises(UnsupportedOperation):
        adapter.update_memory(memory_id, metadata={"topic": "x"})


# --------------------------------------------------------------------------- #
# Capabilities: core-only (Mem0 reflects implicitly on add — no explicit mixin).
# --------------------------------------------------------------------------- #
def test_capabilities_core_only(inject_mem0: InjectFn) -> None:
    inject_mem0()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    caps = adapter.get_capabilities()

    assert caps.supports_reflection is False
    assert caps.supports_forgetting is False
    assert caps.supports_sessions is False
    assert not isinstance(adapter, ReflectionCapability | SessionCapability)


# --------------------------------------------------------------------------- #
# session_id + metadata round-trip through search.
# --------------------------------------------------------------------------- #
def test_session_id_recorded_in_metadata(inject_mem0: InjectFn) -> None:
    inject_mem0()
    adapter = _adapter()
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory(
        "session scoped note", user_id=_UID, session_id=_SID, metadata={"topic": "mission"}
    )
    result = adapter.search("session scoped note", user_id=_UID, top_k=5)
    matched = next(entry for entry in result.results if entry.memory_id == memory_id)
    assert matched.metadata is not None
    assert matched.metadata.get("session_id") == _SID
    assert matched.metadata.get("topic") == "mission"


# --------------------------------------------------------------------------- #
# Live test (gated): real Mem0 + an LLM/embedder backend.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("LHMSB_LIVE_MEM0") != "1",
    reason="live Mem0 needs LHMSB_LIVE_MEM0=1 + mem0ai installed + an LLM/embedder backend",
)
def test_live_mem0_round_trip_and_internal_tokens() -> None:
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id="lhmsb-live-user", track="native")
    adapter.reset(user_id="lhmsb-live-user")

    adapter.add_memory("Alice is the project lead.", user_id="lhmsb-live-user")
    result = adapter.search("who is the project lead", user_id="lhmsb-live-user", top_k=10)
    assert result.results, "live search returned nothing"
    assert meter.to_cost_vector().mem_internal_in_tokens > 0, "internal LLM tokens must be captured"
