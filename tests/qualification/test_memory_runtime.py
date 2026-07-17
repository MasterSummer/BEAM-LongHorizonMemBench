from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import cast

import pytest

from lhmsb.adapters import mem0_qualification
from lhmsb.qualification.memory_runtime import (
    CandidateInventoryDiagnostics,
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    MemoryRuntime,
    MemoryTraceValidationError,
    ProviderUsageEvent,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
    sha256_text,
)


def _memory_object(memory_id: str = "memory-1", content: str = "current plan") -> MemoryObject:
    return MemoryObject(
        memory_id=memory_id,
        content=content,
        content_hash=sha256_text(content),
        metadata=(
            ("lhmsb.provenance", {"source_session": 2}),
            ("lhmsb.graph", {"labels": ["TextualMemory"], "depth": 1}),
        ),
        created_at="2026-07-17T00:00:00+00:00",
        updated_at="2026-07-17T00:00:01+00:00",
        history_length=2,
    )


def _inventory(*items: MemoryObject) -> InventorySnapshot:
    return InventorySnapshot(
        checkpoint_session=3,
        n_write=4,
        n_live=len(items),
        items=tuple(items),
        store_hash=sha256_text("store-v4"),
        backend_count=len(items),
    )


def _candidate(
    memory_id: str = "memory-1",
    content: str = "current plan",
    *,
    rank: int = 1,
    metadata: tuple[tuple[str, object], ...] = (),
) -> RetrievalCandidate:
    return RetrievalCandidate(
        memory_id=memory_id,
        content=content,
        content_hash=sha256_text(content),
        native_rank=rank,
        score=0.8,
        score_details=(("semantic", 0.8),),
        metadata=metadata,
        created_at="2026-07-17T00:00:00+00:00",
        updated_at="2026-07-17T00:00:01+00:00",
    )


def _usage_event() -> ProviderUsageEvent:
    return ProviderUsageEvent(
        call_id="writer-call-1",
        component="memory_internal_llm",
        provider="deepseek",
        model_id="deepseek-v4-pro",
        endpoint_identity="https://api.deepseek.example/v1",
        request_hash=sha256_text("request"),
        response_hash=sha256_text("response"),
        input_tokens=12,
        output_tokens=3,
        cached_tokens=0,
        reasoning_tokens=None,
        usage_observed=True,
        input_count=1,
        latency_seconds=0.2,
        retry_count=0,
        error_class=None,
        started_at_utc="2026-07-17T00:00:00+00:00",
        ended_at_utc="2026-07-17T00:00:00.200000+00:00",
    )


def test_generic_trace_round_trip_preserves_native_and_graph_metadata() -> None:
    item = _memory_object()
    inventory = _inventory(item)
    event = MemoryMutationEvent(
        operation_id="operation-1",
        session_index=3,
        native_event="UPDATE",
        memory_id=item.memory_id,
        memory_text=item.content,
        old_content_hash=sha256_text("old plan"),
        new_content_hash=item.content_hash,
        source="native_response",
        latency_seconds=0.1,
    )
    result = WriteSessionResult(
        session_index=3,
        events=(event,),
        inventory=inventory,
        n_write=4,
        latency_seconds=0.3,
        usage_events=(_usage_event(),),
    )

    encoded = json.loads(json.dumps(result.to_dict(), sort_keys=True))

    assert WriteSessionResult.from_dict(encoded) == result
    assert event.normalized_event == "update"
    assert event.native_id == "memory-1"
    assert event.provenance_metadata == (
        ("source", "native_response"),
        ("operation_id", "operation-1"),
        ("session_index", 3),
    )
    assert item.native_id == "memory-1"
    assert item.provenance_metadata == {"source_session": 2}
    assert item.graph_metadata == {"labels": ["TextualMemory"], "depth": 1}
    result_data = result.to_dict()
    assert isinstance(result_data["events"], list)
    assert result_data["events"][0]["native_event"] == "UPDATE"


def test_nested_tuple_metadata_round_trip_uses_one_immutable_json_array_shape() -> None:
    item = MemoryObject(
        memory_id="memory-tuple",
        content="tuple metadata",
        content_hash=sha256_text("tuple metadata"),
        metadata=(("coordinates", ("x", ("y", "z"))),),
        created_at="t0",
        updated_at="t1",
        history_length=1,
    )

    encoded = json.loads(json.dumps(item.to_dict(), sort_keys=True))
    restored = MemoryObject.from_dict(encoded)

    assert restored == item
    coordinates = cast(list[object], dict(item.metadata)["coordinates"])
    assert coordinates == ["x", ["y", "z"]]
    with pytest.raises(TypeError):
        coordinates.append("cannot mutate")


def test_nested_metadata_is_defensively_frozen_without_changing_json_shape() -> None:
    provenance = {"source_sessions": [1, 2]}
    graph = {
        "labels": ["TextualMemory"],
        "path": [{"edge": "CHILD_OF", "source": "topic-1"}],
    }
    item = MemoryObject(
        memory_id="memory-1",
        content="current plan",
        content_hash=sha256_text("current plan"),
        metadata=(
            ("lhmsb.provenance", provenance),
            ("lhmsb.graph", graph),
        ),
        created_at="t0",
        updated_at="t1",
        history_length=1,
    )
    expected = {
        "memory_id": "memory-1",
        "content": "current plan",
        "content_hash": sha256_text("current plan"),
        "metadata": (
            ("lhmsb.provenance", {"source_sessions": [1, 2]}),
            (
                "lhmsb.graph",
                {
                    "labels": ["TextualMemory"],
                    "path": [{"edge": "CHILD_OF", "source": "topic-1"}],
                },
            ),
        ),
        "created_at": "t0",
        "updated_at": "t1",
        "history_length": 1,
    }
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))

    provenance["source_sessions"].append(99)
    graph["labels"].append("mutated")
    graph["path"][0]["edge"] = "MUTATED"

    assert json.dumps(asdict(item), sort_keys=True, separators=(",", ":")) == expected_json
    serialized_graph = dict(asdict(item)["metadata"])["lhmsb.graph"]
    assert isinstance(serialized_graph, dict)
    assert isinstance(serialized_graph["labels"], list | tuple)

    exposed_graph = cast(dict[str, object], item.graph_metadata)
    exposed_labels = cast(list[object], exposed_graph["labels"])
    with pytest.raises(TypeError):
        exposed_graph["new"] = "cannot mutate"
    with pytest.raises((AttributeError, TypeError)):
        exposed_labels.append("cannot mutate")


def test_score_and_metadata_pair_containers_are_defensively_copied() -> None:
    score_details = [("semantic", 0.8)]
    metadata = [("lhmsb.candidate_origin", "native")]
    candidate = RetrievalCandidate(
        memory_id="memory-1",
        content="current plan",
        content_hash=sha256_text("current plan"),
        native_rank=1,
        score=0.8,
        score_details=cast(tuple[tuple[str, float], ...], score_details),
        metadata=cast(tuple[tuple[str, object], ...], metadata),
        created_at="t0",
        updated_at="t1",
    )

    score_details.append(("mutated", 0.1))
    metadata.append(("lhmsb.score_semantics", "lower_is_better"))

    assert candidate.score_details == (("semantic", 0.8),)
    assert candidate.metadata == (("lhmsb.candidate_origin", "native"),)


def test_retrieval_preserves_native_order_score_semantics_and_origin() -> None:
    first = _candidate(
        metadata=(
            ("lhmsb.candidate_origin", "graph_expansion"),
            ("lhmsb.score_semantics", "lower_is_better"),
            ("lhmsb.graph", {"edge": "CHILD_OF", "source_node": "topic-1"}),
        )
    )
    second = _candidate("memory-2", "offline constraint", rank=2)
    search = CandidateSearch(
        checkpoint_session=3,
        query="what is the current plan?",
        query_hash=sha256_text("what is the current plan?"),
        candidates=(first, second),
        candidate_shortfall=True,
        latency_seconds=0.03,
        usage_events=(_usage_event(),),
    )

    restored = CandidateSearch.from_dict(json.loads(json.dumps(search.to_dict(), sort_keys=True)))

    assert restored == search
    assert first.native_id == "memory-1"
    assert first.candidate_origin == "graph_expansion"
    assert first.score_semantics == "lower_is_better"
    assert first.graph_metadata == {
        "edge": "CHILD_OF",
        "source_node": "topic-1",
    }
    assert second.candidate_origin == "native"
    assert second.score_semantics == "higher_is_better"


def test_mem0_aliases_keep_the_exact_schema_v1_dataclass_layout() -> None:
    assert mem0_qualification.NativeMemoryEvent is MemoryMutationEvent
    assert mem0_qualification.InventoryItem is MemoryObject
    assert mem0_qualification.SearchCandidate is RetrievalCandidate
    event = mem0_qualification.NativeMemoryEvent(
        operation_id="operation-1",
        session_index=1,
        native_event="ADD",
        memory_id="memory-1",
        memory_text="fact",
        old_content_hash=None,
        new_content_hash=sha256_text("fact"),
        source="native_response",
        latency_seconds=0.1,
    )

    assert asdict(event) == {
        "operation_id": "operation-1",
        "session_index": 1,
        "native_event": "ADD",
        "memory_id": "memory-1",
        "memory_text": "fact",
        "old_content_hash": None,
        "new_content_hash": sha256_text("fact"),
        "source": "native_response",
        "latency_seconds": 0.1,
    }
    assert event.normalized_event == "add"


@pytest.mark.parametrize(
    ("items", "n_live", "error_class"),
    [
        ((_memory_object(), _memory_object()), 2, "duplicate_memory_id"),
        ((_memory_object(),), 2, "inventory_count_mismatch"),
    ],
)
def test_inventory_rejects_duplicate_ids_and_n_live_mismatch(
    items: tuple[MemoryObject, ...],
    n_live: int,
    error_class: str,
) -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        InventorySnapshot(
            checkpoint_session=3,
            n_write=4,
            n_live=n_live,
            items=items,
            store_hash=sha256_text("store"),
            backend_count=n_live,
        )

    assert caught.value.error_class == error_class


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MemoryObject(
            memory_id="memory-1",
            content="content",
            content_hash="not-a-sha256",
            metadata=(),
            created_at="",
            updated_at="",
            history_length=0,
        ),
        lambda: MemoryObject(
            memory_id="memory-1",
            content="content",
            content_hash=sha256_text("different content"),
            metadata=(),
            created_at="",
            updated_at="",
            history_length=0,
        ),
        lambda: MemoryMutationEvent(
            operation_id="operation-1",
            session_index=0,
            native_event="ADD",
            memory_id="memory-1",
            memory_text="content",
            old_content_hash=None,
            new_content_hash="bad",
            source="native_response",
            latency_seconds=0.0,
        ),
        lambda: CandidateSearch(
            checkpoint_session=0,
            query="query",
            query_hash="bad",
            candidates=(),
            candidate_shortfall=True,
            latency_seconds=0.0,
        ),
    ],
)
def test_records_reject_invalid_or_inconsistent_hashes(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        factory()

    assert caught.value.error_class == "invalid_hash"


@pytest.mark.parametrize(
    "candidate_factory",
    [
        lambda: (_candidate(rank=2),),
        lambda: (_candidate(), _candidate("memory-2", rank=3)),
        lambda: (_candidate(), _candidate(rank=2)),
    ],
)
def test_candidate_search_rejects_invalid_order_and_duplicate_ids(
    candidate_factory: Callable[[], tuple[RetrievalCandidate, ...]],
) -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        CandidateSearch(
            checkpoint_session=3,
            query="query",
            query_hash=sha256_text("query"),
            candidates=candidate_factory(),
            candidate_shortfall=False,
            latency_seconds=0.0,
        )

    assert caught.value.error_class in {"invalid_native_rank", "duplicate_memory_id"}


def test_retrieval_candidate_rejects_nonpositive_native_rank() -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        _candidate(rank=0)

    assert caught.value.error_class == "invalid_native_rank"


def test_candidate_search_rejects_ids_outside_known_inventory() -> None:
    inventory = _inventory(_memory_object())
    search = CandidateSearch(
        checkpoint_session=3,
        query="query",
        query_hash=sha256_text("query"),
        candidates=(_candidate("unknown", rank=1),),
        candidate_shortfall=True,
        latency_seconds=0.0,
    )

    with pytest.raises(MemoryTraceValidationError) as caught:
        search.validate_against_inventory(inventory)

    assert caught.value.error_class == "candidate_outside_inventory"
    assert "unknown" in str(caught.value)


def test_candidate_inventory_diagnostics_record_anomalies_without_throwing() -> None:
    inventory = _inventory(_memory_object(content="inventory content"))
    search = CandidateSearch(
        checkpoint_session=3,
        query="query",
        query_hash=sha256_text("query"),
        candidates=(
            _candidate("memory-1", "divergent content", rank=1),
            _candidate("missing-memory", "missing content", rank=2),
        ),
        candidate_shortfall=True,
        latency_seconds=0.0,
    )

    diagnostics = search.diagnose_against_inventory(inventory)

    assert diagnostics == CandidateInventoryDiagnostics(
        search_checkpoint_session=3,
        inventory_checkpoint_session=3,
        checkpoint_mismatch=False,
        missing_memory_ids=("missing-memory",),
        content_hash_mismatch_ids=("memory-1",),
    )
    assert not diagnostics.is_consistent
    assert CandidateInventoryDiagnostics.from_dict(diagnostics.to_dict()) == diagnostics
    with pytest.raises(MemoryTraceValidationError) as caught:
        search.validate_against_inventory(inventory)
    assert caught.value.error_class == "candidate_outside_inventory"


def test_write_result_rejects_duplicate_operation_ids() -> None:
    inventory = _inventory(_memory_object())
    event = MemoryMutationEvent(
        operation_id="operation-1",
        session_index=2,
        native_event="ADD",
        memory_id="memory-1",
        memory_text="current plan",
        old_content_hash=None,
        new_content_hash=sha256_text("current plan"),
        source="native_response",
        latency_seconds=0.0,
    )

    with pytest.raises(MemoryTraceValidationError) as caught:
        WriteSessionResult(
            session_index=3,
            events=(event, event),
            inventory=inventory,
            n_write=4,
            latency_seconds=0.0,
        )

    assert caught.value.error_class == "duplicate_operation_id"


def test_write_result_rejects_event_session_mismatch() -> None:
    inventory = _inventory(_memory_object())
    event = MemoryMutationEvent(
        operation_id="operation-1",
        session_index=2,
        native_event="ADD",
        memory_id="memory-1",
        memory_text="current plan",
        old_content_hash=None,
        new_content_hash=sha256_text("current plan"),
        source="native_response",
        latency_seconds=0.0,
    )

    with pytest.raises(MemoryTraceValidationError) as caught:
        WriteSessionResult(
            session_index=3,
            events=(event,),
            inventory=inventory,
            n_write=4,
            latency_seconds=0.0,
        )

    assert caught.value.error_class == "session_mismatch"


def test_mutation_event_validates_new_hash_for_empty_memory_text() -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        MemoryMutationEvent(
            operation_id="operation-empty",
            session_index=0,
            native_event="ADD",
            memory_id="memory-empty",
            memory_text="",
            old_content_hash=None,
            new_content_hash=sha256_text("not empty"),
            source="native_response",
            latency_seconds=0.0,
        )

    assert caught.value.error_class == "invalid_hash"
    matching = MemoryMutationEvent(
        operation_id="operation-empty-valid",
        session_index=0,
        native_event="ADD",
        memory_id="memory-empty",
        memory_text="",
        old_content_hash=None,
        new_content_hash=sha256_text(""),
        source="native_response",
        latency_seconds=0.0,
    )
    assert matching.new_content_hash == sha256_text("")


@pytest.mark.parametrize(
    ("byte_count", "reason", "error_class"),
    [
        (None, None, "invalid_storage_footprint"),
        (1, "also unavailable", "invalid_storage_footprint"),
        (-1, None, "invalid_storage_footprint"),
        (None, "", "invalid_storage_footprint"),
    ],
)
def test_storage_footprint_requires_exactly_one_valid_measurement(
    byte_count: int | None,
    reason: str | None,
    error_class: str,
) -> None:
    with pytest.raises(MemoryTraceValidationError) as caught:
        StorageFootprint(
            component="vector_store",
            bytes=byte_count,
            unavailable_reason=reason,
        )

    assert caught.value.error_class == error_class


def test_storage_footprint_round_trip_accepts_zero_bytes_or_reason() -> None:
    measured = StorageFootprint(component="history_db", bytes=0, unavailable_reason=None)
    unavailable = StorageFootprint(
        component="vector_store",
        bytes=None,
        unavailable_reason="backend does not expose physical bytes",
    )

    assert StorageFootprint.from_dict(measured.to_dict()) == measured
    assert StorageFootprint.from_dict(unavailable.to_dict()) == unavailable


def test_memory_runtime_protocol_declares_lifecycle_and_storage_boundaries() -> None:
    class CompleteRuntime:
        capabilities = LifecycleCapabilities(
            add=True,
            update=True,
            delete=True,
            merge=False,
            links=False,
            history=True,
            resumable=True,
        )

        def restore_write_count(self, n_write: int) -> None:
            del n_write

        def write_session(
            self,
            messages: list[dict[str, str]],
            *,
            session_index: int,
            metadata: dict[str, object] | None = None,
        ) -> WriteSessionResult:
            del messages, session_index, metadata
            raise NotImplementedError

        def snapshot_inventory(self, *, checkpoint_session: int) -> InventorySnapshot:
            del checkpoint_session
            raise NotImplementedError

        def search_candidates(
            self,
            query: str,
            *,
            checkpoint_session: int,
        ) -> CandidateSearch:
            del query, checkpoint_session
            raise NotImplementedError

        def storage_footprints(self) -> tuple[StorageFootprint, ...]:
            return (StorageFootprint("store", bytes=0, unavailable_reason=None),)

        def close(self) -> None:
            return None

    assert isinstance(CompleteRuntime(), MemoryRuntime)
