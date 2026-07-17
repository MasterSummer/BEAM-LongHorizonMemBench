from __future__ import annotations

from dataclasses import replace

import pytest

import lhmsb.qualification.prefix as prefix
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    MemoryMutationEvent,
    MemoryObject,
    RetrievalCandidate,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.tei import RerankResult


def _inventory(checkpoint_session: int = 1) -> InventorySnapshot:
    item = MemoryObject(
        memory_id="memory-1",
        content="keep the pipeline offline",
        content_hash=sha256_text("keep the pipeline offline"),
        metadata=(("nested", {"values": ["offline"]}),),
        created_at="",
        updated_at="",
        history_length=1,
    )
    return InventorySnapshot(
        checkpoint_session=checkpoint_session,
        n_write=1,
        n_live=1,
        items=(item,),
        store_hash="1" * 64,
        backend_count=1,
    )


def _candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        memory_id="memory-1",
        content="keep the pipeline offline",
        content_hash=sha256_text("keep the pipeline offline"),
        native_rank=1,
        score=0.8,
        score_details=(),
        metadata=(),
        created_at="",
        updated_at="",
    )


def _search() -> CandidateSearch:
    return CandidateSearch(
        checkpoint_session=1,
        query="what is the pipeline constraint?",
        query_hash=sha256_text("what is the pipeline constraint?"),
        candidates=(_candidate(),),
        candidate_shortfall=True,
        latency_seconds=0.1,
    )


def _rerank_result(*, ordered_memory_ids: tuple[str, ...] = ("memory-1",)) -> RerankResult:
    return RerankResult(
        ordered_memory_ids=ordered_memory_ids,
        scores=tuple(0.9 for _ in ordered_memory_ids),
        model="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        input_count=1,
        request_hash="2" * 64,
        response_hash="3" * 64,
        latency_seconds=0.2,
    )


def _common_trace() -> object:
    return prefix.CommonRerankTrace(
        opportunity_id="op-1",
        query_hash=sha256_text("what is the pipeline constraint?"),
        candidate_memory_ids=("memory-1",),
        visible_memory_ids=("memory-1",),
        result=_rerank_result(),
    )


def _checkpoint(*, diagnostics: tuple[tuple[str, object], ...] = ()) -> object:
    inventory = _inventory()
    write = WriteSessionResult(
        session_index=0,
        events=(),
        inventory=_inventory(0),
        n_write=1,
        latency_seconds=0.1,
    )
    return prefix.MemoryPrefixCheckpoint(
        checkpoint_session=1,
        surface_hash="b" * 64,
        writes=(write,),
        inventory=inventory,
        retrievals=(_search(),),
        common_reranks=(_common_trace(),),
        graph_diagnostics=diagnostics,
        storage_footprints=(),
    )


def _artifact(*, backend: str = "flat_retrieval") -> object:
    managed = backend != "flat_retrieval"
    return prefix.MemoryPrefixArtifact(
        episode_id="software-42",
        backend=backend,
        profile_id="flat_controlled" if not managed else f"{backend}_controlled",
        config_hash="c" * 64,
        run_identity="4" * 64,
        dataset_release="software-v1",
        dataset_manifest_hash="d" * 64,
        surface_hash="b" * 64,
        writer_profile_id="deepseek_v4_pro_writer" if managed else None,
        embedding_profile_id="bge_m3",
        reranker_profile_id="bge_reranker_v2_m3",
        source_commit="repository" if not managed else "5" * 40,
        model_files_hash="6" * 64,
        checkpoints=(_checkpoint(),),
        graph_diagnostics=(),
        storage_footprints=(),
    )


def test_prefix_round_trip_preserves_real_typed_nested_records_and_bytes() -> None:
    artifact = _artifact()
    encoded = prefix.canonical_prefix_json(artifact)
    decoded = prefix.MemoryPrefixArtifact.from_dict(artifact.to_dict())

    assert isinstance(decoded.checkpoints[0].writes[0], WriteSessionResult)
    assert isinstance(decoded.checkpoints[0].retrievals[0], CandidateSearch)
    assert isinstance(decoded.checkpoints[0].common_reranks[0], prefix.CommonRerankTrace)
    assert isinstance(decoded.checkpoints[0].common_reranks[0].result, RerankResult)
    assert decoded.to_dict() == artifact.to_dict()
    assert prefix.canonical_prefix_json(decoded) == encoded
    assert artifact.artifact_hash == prefix.prefix_artifact_hash(artifact)


def test_common_rerank_rejects_ids_outside_candidate_set() -> None:
    with pytest.raises(prefix.PrefixArtifactError, match="candidate"):
        prefix.CommonRerankTrace(
            opportunity_id="op-1",
            query_hash=sha256_text("what is the pipeline constraint?"),
            candidate_memory_ids=("memory-1",),
            visible_memory_ids=("missing",),
            result=_rerank_result(ordered_memory_ids=("missing",)),
        )


def test_checkpoint_rejects_untyped_write_and_retrieval_members() -> None:
    event = MemoryMutationEvent(
        operation_id="op",
        session_index=1,
        native_event="ADD",
        memory_id="memory-1",
        memory_text="keep the pipeline offline",
        old_content_hash=None,
        new_content_hash=sha256_text("keep the pipeline offline"),
        source="mem0",
        latency_seconds=0.1,
    )
    with pytest.raises(prefix.PrefixArtifactError, match="WriteSessionResult"):
        replace(_checkpoint(), writes=(event,))
    with pytest.raises(prefix.PrefixArtifactError, match="CandidateSearch"):
        replace(_checkpoint(), retrievals=(_candidate(),))


def test_checkpoint_uses_only_prior_sessions_and_keeps_rerank_chain_linked() -> None:
    current_write = replace(
        _checkpoint().writes[0],
        session_index=1,
        inventory=_inventory(1),
    )
    with pytest.raises(prefix.PrefixArtifactError, match="prior|before"):
        replace(_checkpoint(), writes=(current_write,))
    disconnected = replace(_common_trace(), query_hash="9" * 64)
    with pytest.raises(prefix.PrefixArtifactError, match="rerank.*retrieval|chain"):
        replace(_checkpoint(), common_reranks=(disconnected,))


def test_common_rerank_defensively_tuples_shallow_frozen_result() -> None:
    ordered = ["memory-1"]
    scores = [0.9]
    result = RerankResult(
        ordered_memory_ids=ordered,  # type: ignore[arg-type]
        scores=scores,  # type: ignore[arg-type]
        model="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        input_count=1,
        request_hash="2" * 64,
        response_hash="3" * 64,
        latency_seconds=0.2,
    )
    trace = prefix.CommonRerankTrace(
        opportunity_id="op-1",
        query_hash=sha256_text("what is the pipeline constraint?"),
        candidate_memory_ids=("memory-1",),
        visible_memory_ids=("memory-1",),
        result=result,
    )
    ordered[0] = "tampered"
    scores[0] = -100
    assert trace.result.ordered_memory_ids == ("memory-1",)
    assert trace.result.scores == (0.9,)


def test_prefix_nested_json_is_deeply_frozen_at_construction() -> None:
    nested = {"nodes": [{"id": "n1"}]}
    checkpoint = _checkpoint(diagnostics=(("graph", nested),))
    before = prefix.canonical_prefix_json(checkpoint)

    nested["nodes"][0]["id"] = "tampered"  # type: ignore[index]
    assert prefix.canonical_prefix_json(checkpoint) == before
    frozen = checkpoint.graph_diagnostics[0][1]
    with pytest.raises(TypeError):
        frozen["nodes"] = []  # type: ignore[index]


def test_prefix_hash_recomputes_object_content_instead_of_trusting_attribute() -> None:
    artifact = _artifact()
    object.__setattr__(artifact, "artifact_hash", "0" * 64)
    with pytest.raises(prefix.PrefixArtifactError, match="artifact_hash"):
        prefix.prefix_artifact_hash(artifact)


def test_prefix_artifact_enforces_backend_writer_and_source_contract() -> None:
    with pytest.raises(prefix.PrefixArtifactError, match="flat.*writer"):
        replace(_artifact(), writer_profile_id="deepseek_v4_pro_writer", artifact_hash="")
    with pytest.raises(prefix.PrefixArtifactError, match="managed.*writer"):
        replace(_artifact(backend="mem0"), writer_profile_id=None, artifact_hash="")
    with pytest.raises(prefix.PrefixArtifactError, match="source_commit"):
        replace(_artifact(backend="mem0"), source_commit="repository", artifact_hash="")


def test_prefix_artifact_requires_complete_checkpoint_inventory() -> None:
    with pytest.raises(prefix.PrefixArtifactError, match="checkpoint"):
        replace(_artifact(), checkpoints=(), artifact_hash="")
    with pytest.raises(prefix.PrefixArtifactError, match="inventory"):
        replace(
            _artifact(),
            checkpoints=(replace(_checkpoint(), inventory=None),),
            artifact_hash="",
        )


def test_serialized_prefix_rejects_unknown_fields_and_nonstring_json_keys() -> None:
    serialized = _artifact().to_dict()
    serialized["surprise"] = True
    with pytest.raises(prefix.PrefixArtifactError, match="unknown"):
        prefix.MemoryPrefixArtifact.from_dict(serialized)
    with pytest.raises(prefix.PrefixArtifactError, match="string"):
        _checkpoint(diagnostics=(("graph", {1: "collision", "1": "other"}),))


@pytest.mark.parametrize("field", ("run_identity", "config_hash", "model_files_hash"))
def test_prefix_artifact_rejects_nonhex_digests(field: str) -> None:
    with pytest.raises(prefix.PrefixArtifactError, match="lowercase SHA-256"):
        replace(_artifact(), **{field: "r" * 64, "artifact_hash": ""})
