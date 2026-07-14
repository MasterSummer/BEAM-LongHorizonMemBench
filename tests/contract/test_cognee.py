"""Contract + behavior tests for the Cognee self-reorganizing adapter (task 16).

``cognee`` is NOT installed in CI, so these tests inject an in-memory async ``FakeCognee``
via ``sys.modules["cognee"]`` patching (the adapter imports it lazily inside ``initialize``).
Cognee's public surface is a set of module-level coroutines + a ``config`` namespace, so the
fake binds one instance's bound methods onto a fresh ``ModuleType("cognee")`` (one fake
instance per test → isolated state). The fake mirrors the surface the adapter uses —
``remember`` (= ``add`` + ``cognify``), ``recall`` (token-overlap ranked), ``search``,
``forget`` (by data id / dataset / everything), and Cognee's distinctive ``memify`` /
``improve`` self-reorganization — returning a data id the adapter maps back to its stable
``memory_id``. It exercises the ADAPTER, not Cognee itself.

The single live test is gated behind ``LHMSB_LIVE_COGNEE=1`` (real ``cognee`` + an
LLM/embedder backend; the LanceDB/Kuzu/SQLite stores stay file-based, so no external DB).
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
from lhmsb.adapters.cognee_adapter import CogneeAdapter
from lhmsb.cost import CostMeter

_UID = "cognee-user"
_SID = "cognee-session"

_STOPWORDS = frozenset(
    {"the", "is", "are", "was", "were", "of", "a", "an", "to", "in", "on", "and", "for", "no"}
)

#: Module-level coroutine names bound from the fake instance onto the fake ``cognee`` module.
_FUNCTION_NAMES = ("add", "cognify", "remember", "recall", "search", "forget", "memify", "improve")


def _salient(text: str) -> set[str]:
    return {tok for tok in text.lower().split() if len(tok) >= 3 and tok not in _STOPWORDS}


class _FakeEntry:
    def __init__(self, data_id: str, text: str, node_set: list[str], seq: int) -> None:
        self.data_id = data_id
        self.text = text
        self.node_set = node_set
        self.seq = seq


class FakeCognee:
    """In-memory async stand-in for the ``cognee`` module's high-level API.

    State is keyed by dataset so the adapter's per-user dataset isolation holds and
    ``forget(dataset=...)`` (the adapter's ``reset``) clears exactly one user. ``remember``
    composes ``add`` + ``cognify`` (the real 6-stage pipeline) so the cognify call count is
    observable; ``recall`` / ``search`` rank by salient-token overlap and honor ``top_k``."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, _FakeEntry]] = {}
        self._seq = 0
        self.cognify_calls = 0
        self.memify_calls = 0
        self.improve_calls = 0
        self.llm_config: dict[str, object] | None = None
        self.system_root = ""
        self.data_root = ""

    # ---- config namespace -------------------------------------------------- #
    def system_root_directory(self, path: str) -> None:
        self.system_root = path

    def data_root_directory(self, path: str) -> None:
        self.data_root = path

    def set_llm_config(self, config_dict: dict[str, object]) -> None:
        self.llm_config = dict(config_dict)

    # ---- low-level API ----------------------------------------------------- #
    async def add(
        self,
        data: object,
        dataset_name: str = "main_dataset",
        node_set: list[str] | None = None,
    ) -> dict[str, object]:
        self._seq += 1
        data_id = f"data-{self._seq}"
        store = self._data.setdefault(dataset_name, {})
        store[data_id] = _FakeEntry(data_id, str(data), node_set or [], self._seq)
        return {"id": data_id, "dataset": dataset_name}

    async def cognify(self, datasets: object = None, **kwargs: object) -> dict[str, object]:
        self.cognify_calls += 1
        return {"status": "completed"}

    async def search(
        self,
        query_text: str,
        query_type: object = None,
        *,
        datasets: list[str] | None = None,
        top_k: int = 10,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        return self._rank(query_text, datasets, top_k)

    # ---- high-level API ---------------------------------------------------- #
    async def remember(
        self,
        data: object,
        dataset_name: str = "main_dataset",
        session_id: str | None = None,
        node_set: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        result = await self.add(data, dataset_name=dataset_name, node_set=node_set)
        await self.cognify(datasets=[dataset_name])
        return result

    async def recall(
        self,
        query_text: str,
        query_type: object = None,
        *,
        datasets: list[str] | None = None,
        session_id: str | None = None,
        top_k: int = 10,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        return self._rank(query_text, datasets, top_k)

    async def forget(
        self,
        *,
        data_id: str | None = None,
        dataset: str | None = None,
        everything: bool = False,
        **kwargs: object,
    ) -> dict[str, object]:
        if everything:
            self._data.clear()
        elif dataset is not None:
            self._data.pop(dataset, None)
        elif data_id is not None:
            for store in self._data.values():
                store.pop(data_id, None)
        return {"status": "ok"}

    async def memify(self, **kwargs: object) -> dict[str, object]:
        self.memify_calls += 1
        return {"status": "completed"}

    async def improve(self, **kwargs: object) -> dict[str, object]:
        self.improve_calls += 1
        return {"status": "completed"}

    # ---- internals --------------------------------------------------------- #
    def _rank(
        self, query: str, datasets: list[str] | None, top_k: int
    ) -> list[dict[str, object]]:
        terms = _salient(query)
        names = datasets if datasets else list(self._data)
        hits: list[_FakeEntry] = []
        for name in names:
            for entry in self._data.get(name, {}).values():
                if terms & _salient(entry.text):
                    hits.append(entry)
        hits.sort(key=lambda e: (len(terms & _salient(e.text)), e.seq), reverse=True)
        return [
            {"text": e.text, "data_id": e.data_id, "score": float(1 + e.seq)} for e in hits[:top_k]
        ]


InjectFn = Callable[..., FakeCognee]


@pytest.fixture
def inject_cognee(monkeypatch: pytest.MonkeyPatch) -> InjectFn:
    """Install a fake ``cognee`` module backed by the given (or a fresh) ``FakeCognee``.

    ``omit`` drops named coroutines from the module so the adapter's ``getattr``-guarded
    optional hooks (``memify`` / ``improve``) can be exercised as absent."""

    def _inject(fake: FakeCognee | None = None, *, omit: tuple[str, ...] = ()) -> FakeCognee:
        backend = fake if fake is not None else FakeCognee()
        module = types.ModuleType("cognee")
        for name in _FUNCTION_NAMES:
            if name not in omit:
                module.__dict__[name] = getattr(backend, name)
        module.__dict__["config"] = types.SimpleNamespace(
            system_root_directory=backend.system_root_directory,
            data_root_directory=backend.data_root_directory,
            set_llm_config=backend.set_llm_config,
        )
        monkeypatch.setitem(sys.modules, "cognee", module)
        return backend

    return _inject


def _adapter(meter: CostMeter | None = None) -> CogneeAdapter:
    return CogneeAdapter(meter if meter is not None else CostMeter())


# --------------------------------------------------------------------------- #
# Generic contract suite (task 5) against the fake — proves the offline,
# file-based-defaults round-trip (add/search/update/delete/reset) with no network.
# --------------------------------------------------------------------------- #
def test_cognee_passes_full_contract(inject_cognee: InjectFn) -> None:
    inject_cognee()
    run_contract_suite(_adapter)


# --------------------------------------------------------------------------- #
# Lazy import: importing/constructing the adapter must NOT import cognee.
# --------------------------------------------------------------------------- #
def test_cognee_imported_lazily_not_until_initialize(
    monkeypatch: pytest.MonkeyPatch, inject_cognee: InjectFn
) -> None:
    monkeypatch.delitem(sys.modules, "cognee", raising=False)
    adapter = CogneeAdapter(CostMeter())
    assert "cognee" not in sys.modules, "cognee must not import merely by constructing the adapter"

    inject_cognee()
    adapter.initialize(user_id=_UID)
    try:
        assert "cognee" in sys.modules, "initialize() must resolve the lazily-imported cognee"
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Capabilities: reflection only (memify/improve self-reorg); not forgetting/sessions.
# --------------------------------------------------------------------------- #
def test_capabilities_reflection_only(inject_cognee: InjectFn) -> None:
    inject_cognee()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    try:
        caps = adapter.get_capabilities()
        assert caps.supports_reflection is True, "Cognee reflects via memify/improve self-reorg"
        assert caps.supports_forgetting is False
        assert caps.supports_sessions is False
        assert isinstance(adapter, ReflectionCapability)
        assert not isinstance(adapter, ForgettingCapability | SessionCapability)
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Cost instrumentation: internal cognify-LLM + embedding tokens are counted.
# --------------------------------------------------------------------------- #
def test_add_search_counts_internal_tokens(inject_cognee: InjectFn) -> None:
    fake = inject_cognee()
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID, session_id=_SID)
    adapter.reset(user_id=_UID)
    try:
        adapter.add_memory("Acme Corp signed a partnership with Globex in Berlin.", user_id=_UID)
        adapter.search("Acme partnership", user_id=_UID)

        cost = meter.to_cost_vector()
        assert cost.mem_internal_in_tokens > 0, "internal cognify-LLM input tokens must be counted"
        assert cost.mem_internal_out_tokens > 0, "graph-extraction output tokens must be counted"
        assert cost.embedding_tokens > 0, "embedding tokens (add + search) must be counted"
        assert cost.num_retrieval_calls >= 1, "search() must increment the retrieval counter"
        assert cost.storage_bytes > 0, "stored content bytes must be recorded"
        assert cost.write_latency_ms >= 0.0 and cost.retrieval_latency_ms >= 0.0
        assert fake.cognify_calls >= 1, "remember must run the cognify pipeline"
        assert not meter.has_unscoped(), "all internal cost must land inside memory_scope"
    finally:
        adapter.close()


def test_strict_meter_has_no_uncounted_calls(inject_cognee: InjectFn) -> None:
    inject_cognee()
    meter = CostMeter(strict_instrumentation=True)
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        memory_id = adapter.add_memory("a fact under strict accounting", user_id=_UID)
        adapter.update_memory(memory_id, content="an updated fact under strict accounting")
        adapter.search("strict accounting", user_id=_UID)
        adapter.reflect(user_id=_UID)
        adapter.delete_memory(memory_id)
        assert not meter.has_unscoped()
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Reflection (memify + improve) bills its self-reorg tokens to the memory system.
# memify -> mem_internal_* (graph re-organization); improve -> reflection_tokens.
# --------------------------------------------------------------------------- #
def test_reflect_counts_self_reorg_tokens(inject_cognee: InjectFn) -> None:
    fake = inject_cognee()
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        adapter.add_memory("Acme Corp signed a partnership with Globex in Berlin.", user_id=_UID)

        before = meter.to_cost_vector()
        adapter.reflect(user_id=_UID)
        after = meter.to_cost_vector()

        assert after.mem_internal_in_tokens > before.mem_internal_in_tokens, (
            "memify (graph re-organization) input tokens must be counted as memory cost"
        )
        assert after.mem_internal_out_tokens > before.mem_internal_out_tokens, (
            "memify output tokens must be counted as memory cost"
        )
        assert after.reflection_tokens > before.reflection_tokens, (
            "improve (self-improvement) tokens must land in the dedicated reflection field"
        )
        assert fake.memify_calls >= 1 and fake.improve_calls >= 1, "reflect runs memify + improve"
        assert not meter.has_unscoped(), "reflection cost must land inside an explicit scope"
    finally:
        adapter.close()


def test_reflect_without_hooks_still_counts_a_proxy(inject_cognee: InjectFn) -> None:
    # A backend missing the self-reorg hooks must degrade to a content-derived proxy.
    inject_cognee(omit=("memify", "improve"))
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        adapter.add_memory("a durable fact awaiting consolidation", user_id=_UID)
        before = meter.to_cost_vector()
        adapter.reflect(user_id=_UID)
        after = meter.to_cost_vector()
        assert after.mem_internal_in_tokens > before.mem_internal_in_tokens
        assert after.reflection_tokens > before.reflection_tokens
        assert not meter.has_unscoped()
    finally:
        adapter.close()


def test_summarize_concatenates_stored_memory(inject_cognee: InjectFn) -> None:
    inject_cognee()
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        adapter.add_memory("The launch is scheduled for March.", user_id=_UID)
        adapter.add_memory("The budget was approved by finance.", user_id=_UID)

        summary = adapter.summarize(user_id=_UID)
        assert "launch" in summary and "budget" in summary, "summary must include all stored memory"

        scoped = adapter.summarize(user_id=_UID, query="launch schedule")
        assert "launch" in scoped and "budget" not in scoped, "query must scope the summary"
        assert meter.to_cost_vector().mem_internal_in_tokens > 0, "summarize bills a small cost"
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Graceful degradation: a metadata-only update raises UnsupportedOperation.
# --------------------------------------------------------------------------- #
def test_metadata_only_update_degrades(inject_cognee: InjectFn) -> None:
    inject_cognee()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        memory_id = adapter.add_memory("a fact to re-tag", user_id=_UID)
        with pytest.raises(UnsupportedOperation):
            adapter.update_memory(memory_id, metadata={"topic": "x"})
    finally:
        adapter.close()


def test_empty_update_raises_value_error(inject_cognee: InjectFn) -> None:
    inject_cognee()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    adapter.reset(user_id=_UID)
    try:
        memory_id = adapter.add_memory("a fact", user_id=_UID)
        with pytest.raises(ValueError, match="content and/or metadata"):
            adapter.update_memory(memory_id)
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Native vs controlled track (pinned model forwarded via config.set_llm_config).
# --------------------------------------------------------------------------- #
def test_native_track_uses_defaults(inject_cognee: InjectFn) -> None:
    fake = inject_cognee()
    adapter = _adapter()
    adapter.initialize(user_id=_UID)
    try:
        assert adapter.track == "native"
        assert fake.llm_config is None, "native track must not pin the LLM (Cognee defaults)"
        assert fake.data_root, "file-based data root must be configured for offline operation"
    finally:
        adapter.close()


def test_controlled_track_pins_model(inject_cognee: InjectFn) -> None:
    fake = inject_cognee()
    adapter = _adapter()
    adapter.initialize(user_id=_UID, track="controlled", pinned_model="shared/open-weights-model")
    try:
        assert adapter.track == "controlled"
        assert fake.llm_config is not None
        assert fake.llm_config.get("llm_model") == "shared/open-weights-model", (
            "controlled track must pin the shared model"
        )
    finally:
        adapter.close()


def test_controlled_track_merges_caller_llm_config(inject_cognee: InjectFn) -> None:
    fake = inject_cognee()
    adapter = _adapter()
    adapter.initialize(
        user_id=_UID,
        track="controlled",
        pinned_model="shared/open-weights-model",
        llm_config={"llm_provider": "ollama", "llm_endpoint": "http://localhost:11434"},
    )
    try:
        assert fake.llm_config is not None
        assert fake.llm_config["llm_provider"] == "ollama", "caller llm settings are preserved"
        assert fake.llm_config["llm_model"] == "shared/open-weights-model", "pinned model overrides"
    finally:
        adapter.close()


# --------------------------------------------------------------------------- #
# Deterministic memory ids (reproducibility) — minted by the adapter, not the backend.
# --------------------------------------------------------------------------- #
def test_deterministic_memory_ids(inject_cognee: InjectFn) -> None:
    inject_cognee()

    def ids() -> list[str]:
        adapter = _adapter()
        adapter.initialize(user_id="repro-user")
        adapter.reset(user_id="repro-user")
        try:
            return [
                adapter.add_memory(f"fact number {i}", user_id="repro-user") for i in range(3)
            ]
        finally:
            adapter.close()

    first, second = ids(), ids()
    assert first == second, "memory ids must be deterministic across identical runs"
    assert len(set(first)) == 3, "ids must be unique within a run"


# --------------------------------------------------------------------------- #
# Live test (gated): real Cognee + an LLM/embedder backend, file-based stores.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("LHMSB_LIVE_COGNEE") != "1",
    reason="live Cognee needs LHMSB_LIVE_COGNEE=1 + cognee installed + an LLM/embedder backend",
)
def test_live_cognee_remember_recall_and_memify() -> None:
    meter = CostMeter()
    adapter = _adapter(meter)
    adapter.initialize(user_id="lhmsb-live-user", track="native")
    try:
        adapter.reset(user_id="lhmsb-live-user")
        adapter.add_memory("Alice is the project lead.", user_id="lhmsb-live-user")
        adapter.add_memory("The launch is scheduled for March.", user_id="lhmsb-live-user")
        adapter.add_memory("The budget was approved by finance.", user_id="lhmsb-live-user")

        result = adapter.search("who is the project lead", user_id="lhmsb-live-user", top_k=10)
        assert result.results, "live recall returned nothing"
        assert meter.to_cost_vector().mem_internal_in_tokens > 0

        before = meter.to_cost_vector()
        adapter.reflect(user_id="lhmsb-live-user")
        after = meter.to_cost_vector()
        assert after.mem_internal_in_tokens > before.mem_internal_in_tokens, (
            "memify self-reorg must increase mem_internal_* tokens"
        )
        assert after.mem_internal_out_tokens > before.mem_internal_out_tokens
    finally:
        adapter.close()
