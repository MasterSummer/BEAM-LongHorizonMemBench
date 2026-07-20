from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.evaluate import EvaluationTaskResult
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    ProviderUsageEvent,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.metrics import multisystem_state_checkpoints_from_artifacts
from lhmsb.qualification.prepare import (
    PrefixPreparationError,
    _artifact_identity,
    prepare_prefix,
)
from lhmsb.qualification.report import _flatten_rows, _metric_usages
from lhmsb.qualification.runner import QualificationMatrixResult
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
            usage_events=(
                ProviderUsageEvent(
                    # Native components may restart their local counter.  The
                    # report must not collapse distinct provider calls merely
                    # because this ID is reused.
                    call_id="deepseek-writer-000000",
                    component="memory_writer",
                    provider="fake",
                    model_id="fake-writer",
                    endpoint_identity="local://fake-writer",
                    request_hash=sha256_text(f"request-{session_index}"),
                    response_hash=sha256_text(f"response-{session_index}"),
                    input_tokens=10,
                    output_tokens=2,
                    cached_tokens=None,
                    reasoning_tokens=None,
                    usage_observed=True,
                    input_count=1,
                    latency_seconds=0.01,
                    retry_count=0,
                    error_class=None,
                    started_at_utc="2026-07-20T00:00:00+00:00",
                    ended_at_utc="2026-07-20T00:00:00.010000+00:00",
                ),
            ),
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


class SecretFailRuntime(FakeRuntime):
    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        del messages, session_index, metadata
        raise RuntimeError(
            "Authorization: Bearer sentinel-bearer-123 "
            '{"api_key":"sentinel-json-key"} '
            "DEEPSEEK_API_KEY: sentinel-env-key"
        )


class UnsafeErrorClassRuntime(FakeRuntime):
    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        del messages, session_index, metadata

        class ProviderFailureError(RuntimeError):
            error_class = "sk-sentinelSecret123"

        raise ProviderFailureError("provider failed")


class InterruptRuntime(FakeRuntime):
    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        del messages, session_index, metadata
        raise KeyboardInterrupt


class MalformedReranker:
    def __init__(self, result: object) -> None:
        self.result = result

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> object:
        del query, candidates, top_k
        return self.result


class WrongInputCountReranker(FakeReranker):
    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        result = super().rerank(query, candidates, top_k=top_k)
        return replace(result, input_count=result.input_count + 999)


class CachedArtifactStorage:
    def __init__(self, artifact: object, *, run_identity: str) -> None:
        self.artifact = artifact
        self.run_identity = run_identity
        self.save_calls = 0

    def prepare_task(self, task: object, *, episode_hash: str) -> None:
        del task, episode_hash

    def load_prefix_artifact(self, task: object) -> object:
        del task
        return self.artifact

    def save_prefix_artifact(self, task: object, artifact: object) -> bool:
        del task, artifact
        self.save_calls += 1
        return True


def _task(spec_episode: str, *, backend: str = "mem0") -> PreparationTask:
    run_identity = sha256_text("run-identity")
    config_hash = sha256_text("config")
    profile_id = f"{backend}_controlled"
    task_id = f"prepare--{backend}"
    payload = {
        "stage": "prepare_prefix",
        "task_index": 0,
        "task_id": task_id,
        "episode_id": spec_episode,
        "backend": backend,
        "profile_id": profile_id,
        "run_identity": run_identity,
        "config_hash": config_hash,
    }
    return PreparationTask(
        task_index=0,
        task_id=task_id,
        episode_id=spec_episode,
        backend=backend,  # type: ignore[arg-type]
        profile_id=profile_id,
        run_identity=run_identity,
        config_hash=config_hash,
        task_payload_hash=canonical_hash(payload),
    )


def _model_hash(
    task: PreparationTask,
    spec: object,
    *,
    embedding_profile_id: str | None = None,
    reranker_profile_id: str | None = None,
    model_files_hash: str | None = None,
) -> str:
    identity = _artifact_identity(
        task=task,
        spec=spec,  # type: ignore[arg-type]
        config_hash=None,
        dataset_manifest_hash=None,
        embedding_profile_id=embedding_profile_id,
        reranker_profile_id=reranker_profile_id,
        writer_profile_id=None,
        source_commit=None,
        model_files_hash=model_files_hash,
        dataset_release=None,
    )
    value = identity["model_files_hash"]
    assert isinstance(value, str)
    return value


def test_default_model_bundle_hash_is_shared_across_all_backends() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    hashes = {
        _model_hash(_task(spec.plan.episode_id, backend=backend), spec)
        for backend in ("flat_retrieval", "mem0", "amem", "memos")
    }
    assert len(hashes) == 1


def test_default_model_bundle_hash_tracks_only_common_model_identity() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    default = _model_hash(task, spec)

    assert _model_hash(task, spec, embedding_profile_id="other-embedding") != default
    assert _model_hash(task, spec, reranker_profile_id="other-reranker") != default
    assert _model_hash(
        _task(spec.plan.episode_id, backend="amem"), spec
    ) == _model_hash(task, spec)


def test_explicit_model_bundle_hash_is_preserved_verbatim() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    explicit = "a" * 64
    assert _model_hash(
        _task(spec.plan.episode_id), spec, model_files_hash=explicit
    ) == explicit


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
    checkpoint_hashes = tuple(item.surface_hash for item in artifact.checkpoints)
    assert len(set(checkpoint_hashes)) == len(checkpoint_hashes)
    assert all(value != artifact.surface_hash for value in checkpoint_hashes)
    assert storage.load_prefix_artifact(_task(spec.plan.episode_id)) == artifact
    state_rows = multisystem_state_checkpoints_from_artifacts(
        (
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                condition="flat_retrieval",
            ),
        ),
        {spec.plan.episode_id: spec},
        prefix_artifacts={"flat_retrieval": artifact},
    )
    assert len(state_rows) == 5
    assert state_rows[-1].is_final_checkpoint


def test_schema_v2_report_exports_and_scores_prefix_writer_usage(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id, backend="mem0")
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)
    artifact = prepare_prefix(task, spec, FakeRuntime(), FakeReranker(), storage)
    result = EvaluationTaskResult(
        task_id="evaluate-mem0",
        episode_id=spec.plan.episode_id,
        policy_profile_id="gpt_5_6_sol_zen",
        condition="mem0",
        prefix_artifact_hash=artifact.artifact_hash,
        status="complete",
        condition_results=(),
        result_hash=sha256_text("result"),
    )

    rows = _flatten_rows(
        QualificationMatrixResult(task.run_identity, (result,)),  # type: ignore[arg-type]
        prefix_artifacts={"mem0": artifact},
        specs={spec.plan.episode_id: spec},
    )
    usage_rows = rows["api_usage.jsonl"]

    assert len(usage_rows) == spec.plan.n_sessions
    assert len({row["call_id"] for row in usage_rows}) == spec.plan.n_sessions
    assert {row["provider_call_id"] for row in usage_rows} == {
        "deepseek-writer-000000"
    }
    assert {row["call_kind"] for row in usage_rows} == {"memory_internal_llm"}
    assert {row["provider_component"] for row in usage_rows} == {"memory_writer"}
    usage_metrics = _metric_usages(usage_rows)
    assert len(usage_metrics) == spec.plan.n_sessions
    assert all(item.component == "memory_internal_llm" for item in usage_metrics)


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


def test_cached_prefix_requires_complete_requested_identity_match(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    initial_storage = QualificationStorage(
        tmp_path / "initial", run_identity=task.run_identity
    )
    base = prepare_prefix(
        task,
        spec,
        FakeRuntime(),
        FakeReranker(),
        initial_storage,
        dataset_release="software-release-v1",
        dataset_manifest_hash="1" * 64,
        embedding_profile_id="bge_m3",
        reranker_profile_id="bge_reranker_v2_m3",
        writer_profile_id="deepseek_v4_pro_writer",
        source_commit="2" * 40,
        model_files_hash="3" * 64,
    )
    mismatches: tuple[tuple[str, object], ...] = (
        ("episode_id", "other-episode"),
        ("backend", "amem"),
        ("profile_id", "other-profile"),
        ("config_hash", "4" * 64),
        ("run_identity", "5" * 64),
        ("dataset_release", "software-release-v2"),
        ("dataset_manifest_hash", "6" * 64),
        ("surface_hash", "7" * 64),
        ("writer_profile_id", "other-writer"),
        ("embedding_profile_id", "other-embedding"),
        ("reranker_profile_id", "other-reranker"),
        ("source_commit", "8" * 40),
        ("model_files_hash", "9" * 64),
    )

    for field, value in mismatches:
        cached = replace(base, **{field: value, "artifact_hash": ""})
        storage = CachedArtifactStorage(cached, run_identity=task.run_identity)
        runtime = FakeRuntime()
        with pytest.raises(PrefixPreparationError) as exc_info:
            prepare_prefix(
                task,
                spec,
                runtime,
                FakeReranker(),
                storage,  # type: ignore[arg-type]
                dataset_release="software-release-v1",
                dataset_manifest_hash="1" * 64,
                embedding_profile_id="bge_m3",
                reranker_profile_id="bge_reranker_v2_m3",
                writer_profile_id="deepseek_v4_pro_writer",
                source_commit="2" * 40,
                model_files_hash="3" * 64,
            )
        assert exc_info.value.error_class == "identity_mismatch", field
        assert runtime.closed, field
        assert runtime.writes == [], field
        assert storage.save_calls == 0, field


def test_failed_prepare_closes_runtime_and_publishes_no_valid_artifact(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run-identity"))
    runtime = FakeRuntime(fail_session=2)
    with pytest.raises(PrefixPreparationError, match="RuntimeError during prefix preparation"):
        prepare_prefix(task, spec, runtime, FakeReranker(), storage)
    assert runtime.closed
    assert not storage.prefix_artifact_path(task).exists()
    with pytest.raises(QualificationStorageError, match="RuntimeError during prefix preparation"):
        storage.load_prefix_artifact(task)


def test_failed_prepare_can_be_retried_after_any_failure_marker(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)
    with pytest.raises(PrefixPreparationError):
        prepare_prefix(task, spec, FakeRuntime(fail_session=2), FakeReranker(), storage)

    artifact = prepare_prefix(task, spec, FakeRuntime(), FakeReranker(), storage)
    assert artifact.episode_id == spec.plan.episode_id
    assert not storage.prefix_failure_path(task).exists()


def test_failed_prepare_marker_never_persists_exception_secrets(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)

    with pytest.raises(PrefixPreparationError):
        prepare_prefix(task, spec, SecretFailRuntime(), FakeReranker(), storage)

    marker = storage.prefix_failure_path(task).read_text(encoding="utf-8")
    assert "sentinel-bearer-123" not in marker
    assert "sentinel-json-key" not in marker
    assert "sentinel-env-key" not in marker
    assert "Authorization" not in marker
    assert "DEEPSEEK_API_KEY" not in marker


def test_failed_prepare_does_not_trust_arbitrary_error_class(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)

    with pytest.raises(PrefixPreparationError) as exc_info:
        prepare_prefix(task, spec, UnsafeErrorClassRuntime(), FakeReranker(), storage)

    marker = storage.prefix_failure_path(task).read_text(encoding="utf-8")
    assert "sentinelSecret123" not in marker
    assert exc_info.value.error_class == "prefix_preparation_failure"


def test_prepare_propagates_keyboard_interrupt_without_failure_marker(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    runtime = InterruptRuntime()
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)

    with pytest.raises(KeyboardInterrupt):
        prepare_prefix(task, spec, runtime, FakeReranker(), storage)

    assert runtime.closed
    assert not storage.prefix_failure_path(task).exists()
    assert not storage.prefix_artifact_path(task).exists()


@pytest.mark.parametrize(
    "result",
    (
        {"ordered_memory_ids": [1], "scores": [0.5]},
        {"ordered_memory_ids": ["memory-0"], "scores": ["0.5"]},
        {"ordered_memory_ids": ["memory-0"], "scores": [True]},
        {"ordered_memory_ids": ["memory-0"], "scores": [float("nan")]},
        {"ordered_memory_ids": ["memory-0"], "scores": []},
        {"ordered_memory_ids": [], "scores": []},
        {"ordered_memory_ids": ["memory-0"], "scores": [0.5], "input_count": 999},
        {"ordered_memory_ids": ["memory-0"], "scores": [0.5], "input_count": True},
        {"ordered_memory_ids": ["memory-0"], "scores": [0.5], "input_count": -1},
        ["memory-0"],
    ),
    ids=(
        "non-string-id",
        "string-score",
        "boolean-score",
        "non-finite-score",
        "length-mismatch",
        "candidate-shortfall",
        "wrong-input-count",
        "boolean-input-count",
        "negative-input-count",
        "ids-without-scores",
    ),
)
def test_prepare_rejects_malformed_reranker_results(
    tmp_path: Path,
    result: object,
) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(
        tmp_path / canonical_hash(result),
        run_identity=task.run_identity,
    )

    with pytest.raises(PrefixPreparationError) as exc_info:
        prepare_prefix(
            task,
            spec,
            FakeRuntime(),
            MalformedReranker(result),  # type: ignore[arg-type]
            storage,
        )

    assert exc_info.value.error_class == "reranker_failure"
    assert not storage.prefix_artifact_path(task).exists()


def test_prepare_rejects_reranker_input_count_mismatch(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(spec.plan.episode_id)
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)

    with pytest.raises(PrefixPreparationError) as exc_info:
        prepare_prefix(
            task,
            spec,
            FakeRuntime(),
            WrongInputCountReranker(),
            storage,
        )

    assert exc_info.value.error_class == "reranker_failure"
    assert not storage.prefix_artifact_path(task).exists()


def test_prepare_closes_runtime_on_pre_replay_identity_failure(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task("different-episode")
    runtime = FakeRuntime()
    storage = QualificationStorage(tmp_path / "run", run_identity=task.run_identity)

    with pytest.raises(PrefixPreparationError, match="episode"):
        prepare_prefix(task, spec, runtime, FakeReranker(), storage)

    assert runtime.closed
    assert not storage.prefix_failure_path(task).exists()


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
