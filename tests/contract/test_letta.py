"""Contract + behavior tests for the Letta self-editing-block adapter (task 14).

The Letta client SDK is NOT installed in CI, so these tests inject an in-memory
``FakeLettaClient`` via ``sys.modules["ai_memory_sdk"]`` patching (the adapter imports it
lazily inside ``initialize``). The fake mirrors the client surface the adapter uses —
``initialize_subject`` / ``initialize_memory`` / ``add_messages`` (returning a *run* with
token usage) / ``wait_for_run`` / ``search`` / a labelled block edit / ``delete_block`` /
``delete_user`` — plus Letta's defining feature: a ``trigger_sleeptime`` consolidation
pass whose internal-LLM tokens are billed to the memory system. It keys blocks by the
adapter's deterministic content id (reusing ``letta_adapter._memory_id``) so the fake and
the adapter agree on the stable ``memory_id`` the contract round-trips on. This exercises
the ADAPTER, not Letta itself.

The single live test is gated behind ``LHMSB_LIVE_LETTA=1`` + a running Letta server.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable

import pytest

from contract.adapter_contract import run_contract_suite
from lhmsb.adapters import ForgettingCapability, ReflectionCapability, SessionCapability
from lhmsb.adapters.base import UnsupportedOperation
from lhmsb.adapters.letta_adapter import LettaAdapter, _matches, _memory_id
from lhmsb.cost import CostMeter

_UID = "letta-user"
_CREATED_AT = "2026-01-01T00:00:00+00:00"


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeRun:
    def __init__(self, run_id: str, usage: _FakeUsage) -> None:
        self.id = run_id
        self.status = "created"
        self.usage = usage


class _FakeBlock:
    def __init__(self, label: str, value: str, seq: int) -> None:
        self.label = label
        self.value = value
        self.created_at = _CREATED_AT
        self.updated_at = _CREATED_AT
        self.seq = seq


class FakeLettaClient:
    """In-memory stand-in for the Letta / AI-Memory-SDK client.

    Blocks are keyed per subject by the adapter's deterministic content id, so a re-add or
    block edit of the same logical memory keeps a stable label (hence a stable
    ``memory_id``), which is exactly what the update/delete contract checks rely on."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.memory_kwargs: dict[str, object] = {}
        self.subject_id = ""
        self.blocks: dict[str, dict[str, _FakeBlock]] = {}
        self._seq = 0

    def initialize_subject(self, *, subject_id: str) -> None:
        self.subject_id = subject_id
        self.blocks.setdefault(subject_id, {})

    def initialize_memory(self, *, label: str, **kwargs: object) -> None:
        self.memory_kwargs = {"label": label, **kwargs}

    def _store(self) -> dict[str, _FakeBlock]:
        return self.blocks.setdefault(self.subject_id, {})

    def _write_block(self, label: str, value: str) -> None:
        self._seq += 1
        store = self._store()
        if label in store:
            store[label].value = value
            store[label].seq = self._seq
        else:
            store[label] = _FakeBlock(label, value, self._seq)

    def add_messages(self, *, messages: list[dict[str, str]]) -> _FakeRun:
        content = messages[-1]["content"]
        self._write_block(_memory_id(content), content)
        words = len(content.split())
        run = _FakeRun(f"run-{self._seq}", _FakeUsage(words, max(1, words)))
        return run

    def wait_for_run(self, run: _FakeRun) -> _FakeRun:
        run.status = "completed"
        return run

    def update_block(self, *, label: str, value: str) -> None:
        self._write_block(label, value)

    def search(self, *, user_id: str, query: str) -> list[_FakeBlock]:
        store = self.blocks.get(user_id, {})
        hits = [block for block in store.values() if _matches(query, block.value)]
        hits.sort(key=lambda block: block.seq, reverse=True)
        return hits

    def delete_block(self, *, label: str) -> None:
        self._store().pop(label, None)

    def delete_user(self, *, user_id: str) -> None:
        self.blocks.pop(user_id, None)

    def trigger_sleeptime(self, *, subject_id: str) -> _FakeRun:
        self._seq += 1
        corpus = " ".join(block.value for block in self.blocks.get(subject_id, {}).values())
        words = len(corpus.split())
        return _FakeRun(f"sleeptime-{self._seq}", _FakeUsage(words, max(1, words // 2)))


InjectFn = Callable[[type[FakeLettaClient]], None]


@pytest.fixture
def inject_letta(monkeypatch: pytest.MonkeyPatch) -> InjectFn:
    """Install a fake ``ai_memory_sdk`` module exposing the given client class."""

    def _inject(client_cls: type[FakeLettaClient]) -> None:
        module = types.ModuleType("ai_memory_sdk")
        module.__dict__["AIMemory"] = client_cls
        monkeypatch.setitem(sys.modules, "ai_memory_sdk", module)

    return _inject


def _adapter(meter: CostMeter | None = None) -> LettaAdapter:
    return LettaAdapter(meter if meter is not None else CostMeter())


# --------------------------------------------------------------------------- #
# Generic contract suite (task 5) against the fake.
# --------------------------------------------------------------------------- #
def test_letta_passes_full_contract(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    run_contract_suite(_adapter)


# --------------------------------------------------------------------------- #
# Capabilities: reflection only (sleeptime); not forgetting, not sessions.
# --------------------------------------------------------------------------- #
def test_capabilities_reflection_only(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    caps = adapter.get_capabilities()

    assert caps.supports_reflection is True, "Letta reflects via sleeptime consolidation"
    assert caps.supports_forgetting is False
    assert caps.supports_sessions is False
    assert isinstance(adapter, ReflectionCapability)
    assert not isinstance(adapter, ForgettingCapability | SessionCapability)


# --------------------------------------------------------------------------- #
# Reflection (sleeptime) bills its internal-LLM tokens to the memory system.
# --------------------------------------------------------------------------- #
def test_reflect_counts_sleeptime_tokens(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    adapter.add_memory("Acme Corp signed a partnership with Globex in Berlin.", user_id=_UID)

    before = meter.to_cost_vector()
    adapter.reflect(user_id=_UID)
    after = meter.to_cost_vector()

    assert after.mem_internal_in_tokens > before.mem_internal_in_tokens, (
        "sleeptime input tokens must be counted as memory cost"
    )
    assert after.mem_internal_out_tokens > before.mem_internal_out_tokens, (
        "sleeptime output (consolidation) tokens must be counted as memory cost"
    )
    assert not meter.has_unscoped(), "reflection cost must land inside memory_scope"


def test_reflect_without_sleeptime_hook_still_counts_a_proxy(
    inject_letta: InjectFn, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject_letta(FakeLettaClient)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    adapter.add_memory("a durable fact awaiting consolidation", user_id=_UID)
    monkeypatch.setattr(adapter._client, "trigger_sleeptime", None)

    before = meter.to_cost_vector()
    adapter.reflect(user_id=_UID)
    after = meter.to_cost_vector()

    assert after.mem_internal_in_tokens > before.mem_internal_in_tokens
    assert after.mem_internal_out_tokens > before.mem_internal_out_tokens
    assert not meter.has_unscoped()


# --------------------------------------------------------------------------- #
# Graceful degradation: a metadata-only update raises UnsupportedOperation.
# --------------------------------------------------------------------------- #
def test_metadata_only_update_degrades(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a fact to re-tag", user_id=_UID)

    with pytest.raises(UnsupportedOperation):
        adapter.update_memory(memory_id, metadata={"tag": "important"})


# --------------------------------------------------------------------------- #
# Cost instrumentation: internal block-edit tokens are counted under memory_scope.
# --------------------------------------------------------------------------- #
def test_internal_block_edit_tokens_counted(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)

    adapter.add_memory("Alice leads the platform team out of the Berlin office.", user_id=_UID)
    adapter.search("who leads the platform team", user_id=_UID)

    cost = meter.to_cost_vector()
    assert cost.mem_internal_in_tokens > 0, "ingested-message tokens must be counted"
    assert cost.mem_internal_out_tokens > 0, "block self-edit output tokens must be counted"
    assert cost.num_retrieval_calls >= 1, "search() must increment the retrieval counter"
    assert cost.write_latency_ms >= 0.0 and cost.retrieval_latency_ms >= 0.0
    assert cost.storage_bytes > 0, "stored content bytes must be recorded"
    assert not meter.has_unscoped(), "all internal cost must land inside memory_scope"


def test_strict_meter_has_no_uncounted_calls(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    meter = CostMeter(strict_instrumentation=True)
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    memory_id = adapter.add_memory("a fact under strict accounting", user_id=_UID)
    adapter.update_memory(memory_id, content="an updated fact under strict accounting")
    adapter.search("strict accounting", user_id=_UID)
    adapter.reflect(user_id=_UID)
    adapter.delete_memory(memory_id)
    assert not meter.has_unscoped()


# --------------------------------------------------------------------------- #
# Summarize: concatenates stored memory (optionally query-scoped) at a small cost.
# --------------------------------------------------------------------------- #
def test_summarize_concatenates_stored_memory(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    adapter.add_memory("The launch is scheduled for March.", user_id=_UID)
    adapter.add_memory("The budget was approved by finance.", user_id=_UID)

    summary = adapter.summarize(user_id=_UID)
    assert "launch" in summary and "budget" in summary, "summary must include all stored memory"

    scoped = adapter.summarize(user_id=_UID, query="launch schedule")
    assert "launch" in scoped and "budget" not in scoped, "query must scope the summary"
    assert meter.to_cost_vector().mem_internal_in_tokens > 0, "summarize bills a small cost"


# --------------------------------------------------------------------------- #
# Native vs controlled track (pinned model forwarded to initialize_memory).
# --------------------------------------------------------------------------- #
def test_native_track_pins_no_model(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    assert adapter.track == "native"
    assert "model" not in adapter._client.memory_kwargs


def test_controlled_track_forwards_pinned_model(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)
    adapter = _adapter()
    adapter.initialize(user_id=_UID, track="controlled", pinned_model="shared/open-weights-model")
    assert adapter.track == "controlled"
    assert adapter._client.memory_kwargs.get("model") == "shared/open-weights-model"


# --------------------------------------------------------------------------- #
# Deterministic content ids (reproducibility).
# --------------------------------------------------------------------------- #
def test_deterministic_memory_ids(inject_letta: InjectFn) -> None:
    inject_letta(FakeLettaClient)

    def ids() -> list[str]:
        adapter = _adapter()
        adapter.initialize(user_id="repro-user")
        adapter.reset(user_id="repro-user")
        return [adapter.add_memory(f"fact number {i}", user_id="repro-user") for i in range(3)]

    first, second = ids(), ids()
    assert first == second, "memory ids must be deterministic across identical runs"
    assert len(set(first)) == 3, "ids must be unique within a run"
    assert all(mid.startswith("letta-") for mid in first)


# --------------------------------------------------------------------------- #
# Live test (gated): real Letta server + SDK.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("LHMSB_LIVE_LETTA") != "1",
    reason="live Letta needs LHMSB_LIVE_LETTA=1 + a running Letta server / SDK",
)
def test_live_letta_round_trip_and_reflection() -> None:
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(
        user_id="lhmsb-live-user",
        base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
        token=os.environ.get("LETTA_TOKEN"),
    )
    adapter.reset(user_id="lhmsb-live-user")

    memory_id = adapter.add_memory("Alice is the project lead.", user_id="lhmsb-live-user")
    result = adapter.search("who is the project lead", user_id="lhmsb-live-user", top_k=10)
    assert result.results, "live search returned nothing"
    assert meter.to_cost_vector().mem_internal_in_tokens > 0

    adapter.update_memory(memory_id, content="Bob is the project lead now.")
    adapter.reflect(user_id="lhmsb-live-user")
    assert meter.to_cost_vector().mem_internal_out_tokens > 0, "sleeptime tokens must be counted"
