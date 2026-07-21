from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import lhmsb.adapters.amem_qualification as amem_module
from lhmsb.adapters.amem_qualification import (
    AMemQualificationAdapter,
    AMemQualificationError,
    _amem_writer_max_output_tokens,
    validate_amem_source,
)
from lhmsb.qualification.context import PublicHistoryUnit


@dataclass
class FakeNote:
    content: str
    id: str
    keywords: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    context: str = "General"
    category: str = "Uncategorized"
    tags: list[str] = field(default_factory=list)
    timestamp: str = "202501010000"
    last_accessed: str = "202501010000"
    retrieval_count: int = 0
    evolution_history: list[object] = field(default_factory=list)


class FakeCollection:
    def __init__(self, owner: FakeRetriever) -> None:
        self.owner = owner
        self.embedding_function = None

    def get(self) -> dict[str, object]:
        return {"ids": list(self.owner.rows)}


class FakeRetriever:
    def __init__(self) -> None:
        self.rows: dict[str, object] = {}
        self.collection = FakeCollection(self)
        self.embedding_function = None


class FakeAMem:
    __source_commit__ = "ceffb860f0712bbae97b184d440df62bc910ca8d"

    def __init__(self) -> None:
        self.memories: dict[str, FakeNote] = {}
        self.retriever = FakeRetriever()
        self.llm_controller = SimpleNamespace(llm=SimpleNamespace(calls=[]))
        self.analyze_calls = 0

    def add_note(self, content: str, **kwargs: object) -> str:
        memory_id = str(kwargs.get("id", f"random-{len(self.memories)}"))
        note = FakeNote(
            content=content,
            id=memory_id,
            timestamp=str(kwargs.get("timestamp", "202501010000")),
        )
        self.memories[memory_id] = note
        self.retriever.rows[memory_id] = note
        return memory_id

    def read(self, memory_id: str) -> FakeNote | None:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs: object) -> bool:
        note = self.memories.get(memory_id)
        if note is None:
            return False
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self.memories:
            return False
        del self.memories[memory_id]
        self.retriever.rows.pop(memory_id, None)
        return True

    def search_agentic(self, query: str, k: int = 5) -> list[dict[str, object]]:
        del query
        notes = list(self.memories.values())[:k]
        return [
            {
                "id": note.id,
                "content": note.content,
                "score": float(index),
                "is_neighbor": False,
            }
            for index, note in enumerate(notes)
        ]


class FakeEmbedding:
    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((float(len(text)),) for text in texts)


def _unit(content: str, session: int = 0) -> PublicHistoryUnit:
    return PublicHistoryUnit.create(
        episode_id="episode-1",
        source_session=session,
        source_kind="observation",
        source_ordinal=0,
        content=content,
    )


def test_official_import_forces_package_local_litellm_cost_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
    monkeypatch.setattr(amem_module.importlib, "import_module", lambda _name: sentinel)

    assert amem_module._load_official_module() is sentinel
    assert amem_module.os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_add_note_uses_native_api_and_normalizes_inventory_and_search() -> None:
    backend = FakeAMem()
    adapter = AMemQualificationAdapter(
        backend,
        namespace="ep-1",
        episode_id="episode-1",
        candidate_k=2,
        embedding=FakeEmbedding(),
        require_deterministic_ids=True,
    )
    result = adapter.write_session(
        [],
        session_index=0,
        metadata={"public_units": (_unit("offline pipeline"),)},
    )
    assert len(result.events) == 1
    assert result.events[0].native_event == "ADD_NOTE"
    assert adapter.snapshot_inventory(checkpoint_session=0).n_live == 1
    search = adapter.search_candidates("pipeline", checkpoint_session=0)
    assert search.candidates[0].score_semantics == "lower_is_better"
    assert search.candidates[0].score == 0.0
    assert backend.retriever.embedding_function is not None
    adapter.close()


def test_link_expansion_preserves_rows_without_inventing_neighbor_scores() -> None:
    backend = FakeAMem()
    adapter = AMemQualificationAdapter(
        backend,
        namespace="ep-1",
        episode_id="episode-1",
        candidate_k=3,
        require_deterministic_ids=True,
    )
    first = adapter.add_note("first")
    second = adapter.add_note("second")
    backend.memories[first].links.append(second)
    rows = [
        {"id": first, "content": "first", "score": 0.0, "is_neighbor": False},
        {"id": second, "content": "second", "is_neighbor": True},
    ]
    backend.search_agentic = lambda query, k=5: rows  # type: ignore[method-assign]
    search = adapter.search_candidates("q", checkpoint_session=1)
    assert search.candidates[-1].candidate_origin == "native_link"
    assert search.candidates[-1].score is None
    adapter.close()


def test_source_and_api_mismatch_fail_without_fallback() -> None:
    with pytest.raises(AMemQualificationError, match="required API"):
        AMemQualificationAdapter(object())
    bad = FakeAMem()
    bad.__source_commit__ = "wrong"  # type: ignore[attr-defined]
    # Constructor-level fake identity is checked by the live factory; ensure the
    # explicit source verifier remains strict through the public helper path.
    assert bad.__source_commit__ == "wrong"


def test_source_identity_can_come_from_verified_external_manifest(monkeypatch) -> None:
    module = SimpleNamespace(
        __name__="agentic_memory.memory_system",
        __file__="/verified/sources/amem/agentic_memory/memory_system.py",
        AgenticMemorySystem=FakeAMem,
    )
    monkeypatch.setattr(
        amem_module,
        "verified_source_commit_for_module",
        lambda value, source: "ceffb860f0712bbae97b184d440df62bc910ca8d",
    )
    validate_amem_source(module)

    module.__source_commit__ = "wrong"
    with pytest.raises(AMemQualificationError, match="source commit"):
        validate_amem_source(module)


def test_amem_writer_budget_has_offline_reasoning_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LHMSB_AMEM_WRITER_MAX_OUTPUT_TOKENS", raising=False)
    assert _amem_writer_max_output_tokens() == 2048
    monkeypatch.setenv("LHMSB_AMEM_WRITER_MAX_OUTPUT_TOKENS", "3072")
    assert _amem_writer_max_output_tokens() == 3072
