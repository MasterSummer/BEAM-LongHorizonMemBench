from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.prepare import PrefixPreparationError, prepare_prefix
from lhmsb.qualification.schema import PreparationTask
from lhmsb.qualification.storage import QualificationStorage, QualificationStorageError
from lhmsb.qualification.tei import RerankCandidate, RerankResult


class FakeRuntime:
    capabilities = LifecycleCapabilities(
        add=True,
        update=False,
        delete=False,
        merge=False,
        links=False,
        history=False,
        resumable=False,
    )

    def __init__(self, *, fail_session: int | None = None, nonempty: bool = False) -> None:
        self.items: dict[str, MemoryObject] = {}
        self.writes: list[int] = []
        self.searches: list[int] = []
        self.events: list[tuple[str, int]] = []
        self.closed = False
        self.fail_session = fail_session
        if nonempty:
            self._add(0)

    def restore_write_count(self, n_write: int) -> None:
        del n_write

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        del messages, metadata
        if self.fail_session == session_index:
            raise RuntimeError("synthetic write failure")
        self.events.append(("write", session_index))
        self.writes.append(session_index)
        item = self._add(session_index)
        inventory = self._inventory(checkpoint_session=session_index, include_current=True)
        event = MemoryMutationEvent(
            operation_id=f"op-{session_index}",
            session_index=session_index,
            native_event="ADD",
            memory_id=item.memory_id,
            memory_text=item.content,
            old_content_hash=None,
            new_content_hash=item.content_hash,
            source="fake",
            latency_seconds=0.0,
        )
        return WriteSessionResult(
            session_index=session_index,
            events=(event,),
            inventory=inventory,
            n_write=inventory.n_write,
            latency_seconds=0.0,
        )

    def snapshot_inventory(self, *, checkpoint_session: int) -> InventorySnapshot:
        if checkpoint_session == 0 and self.items:
            return self._inventory(checkpoint_session=checkpoint_session, include_current=True)
        return self._inventory(checkpoint_session=checkpoint_session, include_current=False)

    def _inventory(
        self,
        *,
        checkpoint_session: int,
        include_current: bool,
    ) -> InventorySnapshot:
        eligible = tuple(
            item
            for item in self.items.values()
            if dict(item.metadata).get("session_index", -1)
            < (checkpoint_session + 1 if include_current else checkpoint_session)
        )
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=len(eligible),
            n_live=len(eligible),
            items=eligible,
            store_hash=sha256_text("|".join(item.content_hash for item in eligible)),
            backend_count=len(eligible),
        )

    def search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch:
        self.events.append(("search", checkpoint_session))
        self.searches.append(checkpoint_session)
        inventory = self.snapshot_inventory(checkpoint_session=checkpoint_session)
        candidates = tuple(
            RetrievalCandidate(
                memory_id=item.memory_id,
                content=item.content,
                content_hash=item.content_hash,
                native_rank=rank,
                score=float(len(inventory.items) - rank),
                score_details=(),
                metadata=item.metadata,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for rank, item in enumerate(inventory.items, start=1)
        )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=sha256_text(query),
            candidates=candidates,
            candidate_shortfall=len(candidates) < 20,
            latency_seconds=0.0,
        )

    def storage_footprints(self) -> tuple[StorageFootprint, ...]:
        return ()

    def close(self) -> None:
        self.closed = True
        self.events.append(("close", -1))

    def _add(self, session: int) -> MemoryObject:
        content = f"public session {session} project update"
        item = MemoryObject(
            memory_id=f"memory-{session}",
            content=content,
            content_hash=sha256_text(content),
            metadata=(("session_index", session),),
            created_at=f"t{session}",
            updated_at=f"t{session}",
            history_length=1,
        )
        self.items[item.memory_id] = item
        return item


class FakeReranker:
    def __init__(self, *, invalid: bool = False) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.invalid = invalid

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        self.calls.append((query, tuple(item.memory_id for item in candidates)))
        ordered = tuple(item.memory_id for item in candidates)
        if top_k is not None:
            ordered = ordered[:top_k]
        if self.invalid:
            ordered = ("missing-memory",)
        return RerankResult(
            ordered_memory_ids=ordered,
            scores=tuple(float(index) for index, _ in enumerate(ordered, start=1)),
            model="fake-reranker",
            revision="test",
            input_count=len(candidates),
            request_hash=sha256_text(query),
            response_hash=sha256_text("|".join(ordered)),
            latency_seconds=0.0,
        )


def _task(spec_episode: str, *, backend: str = "mem0") -> PreparationTask:
    run_identity = sha256_text("run-identity")
    config_hash = sha256_text("config")
    return PreparationTask(
        task_index=0,
        task_id=f"prepare--{backend}",
        episode_id=spec_episode,
        backend=backend,  # type: ignore[arg-type]
        profile_id=f"{backend}_controlled",
        run_identity=run_identity,
        config_hash=config_hash,
        task_payload_hash=canonical_hash({"episode": spec_episode, "backend": backend}),
    )


def test_prepare_replays_public_sessions_and_searches_before_current_write(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    runtime = FakeRuntime()
    reranker = FakeReranker()
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    artifact = prepare_prefix(_task(spec.plan.episode_id), spec, runtime, reranker, storage)

    assert runtime.closed
    assert runtime.writes == [0, 1, 2, 3]
    assert all(
        event[0] != "write"
        or all(previous[0] != "write" for previous in runtime.events[:index])
        for index, event in enumerate(runtime.events)
        if event[0] == "write" and event[1] == 0
    )
    assert [item.checkpoint_session for item in artifact.checkpoints] == [0, 1, 2, 3, 4]
    for checkpoint in artifact.checkpoints:
        assert all(
            dict(item.metadata).get("session_index", -1) < checkpoint.checkpoint_session
            for item in checkpoint.inventory.items
        )
    for index, event in enumerate(runtime.events):
        if event[0] == "search":
            next_write = next(
                (item for item in runtime.events[index + 1 :] if item[0] == "write"),
                None,
            )
            assert next_write is None or event[1] <= next_write[1]
    assert all(
        trace.result.input_count == len(trace.candidate_memory_ids)
        for checkpoint in artifact.checkpoints
        for trace in checkpoint.common_reranks
    )
    assert storage.load_prefix_artifact(_task(spec.plan.episode_id)) == artifact


def test_prepare_is_deterministic_and_second_run_is_read_only(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    first_runtime = FakeRuntime()
    first = prepare_prefix(
        _task(spec.plan.episode_id), spec, first_runtime, FakeReranker(), storage
    )
    second_runtime = FakeRuntime()
    second = prepare_prefix(
        _task(spec.plan.episode_id), spec, second_runtime, FakeReranker(), storage
    )
    assert second == first
    assert second_runtime.closed
    assert second_runtime.writes == []


def test_failed_prepare_closes_runtime_and_publishes_no_valid_artifact(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    runtime = FakeRuntime(fail_session=2)
    with pytest.raises(PrefixPreparationError, match="synthetic write failure"):
        prepare_prefix(task, spec, runtime, FakeReranker(), storage)
    assert runtime.closed
    assert not storage.prefix_artifact_path(task).exists()
    with pytest.raises(QualificationStorageError, match="synthetic write failure"):
        storage.load_prefix_artifact(task)


def test_prepare_rejects_nonempty_start_and_invalid_common_subset(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    with pytest.raises(PrefixPreparationError, match="empty runtime"):
        prepare_prefix(
            _task(spec.plan.episode_id),
            spec,
            FakeRuntime(nonempty=True),
            FakeReranker(),
            storage,
        )

    storage2 = QualificationStorage(
        tmp_path / "run2", run_identity=sha256_text("run-identity")
    )
    with pytest.raises(PrefixPreparationError, match="candidate subset"):
        prepare_prefix(
            _task(spec.plan.episode_id),
            spec,
            FakeRuntime(),
            FakeReranker(invalid=True),
            storage2,
        )


def test_corrupt_nested_checkpoint_is_rejected(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    prepare_prefix(task, spec, FakeRuntime(), FakeReranker(), storage)
    path = storage.prefix_artifact_path(task)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["surface_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QualificationStorageError, match="hash"):
        storage.load_prefix_artifact(task)
