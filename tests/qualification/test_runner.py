from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lhmsb.adapters.mem0_qualification import (
    CandidateSearch,
    InventoryItem,
    InventorySnapshot,
    NativeMemoryEvent,
    SearchCandidate,
    WriteSessionResult,
)
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.families.software.vertical_checker import BehaviorResult
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.qualification.config import (
    build_qualification_tasks,
    load_qualification_config,
)
from lhmsb.qualification.providers import (
    PolicyCallError,
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)
from lhmsb.qualification.runner import (
    QualificationMatrixResult,
    TaskComponents,
    TaskIsolation,
    hash_text,
    policy_request_hash,
    qualification_task_result_from_dict,
    run_qualification_matrix,
    run_qualification_task,
)
from lhmsb.qualification.storage import (
    QualificationStorage,
    QualificationStorageError,
)
from lhmsb.qualification.tei import (
    RerankCandidate,
    RerankResult,
    TeiServiceError,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"


class FakePolicy:
    def __init__(self, owner: FakeFactory) -> None:
        self.owner = owner

    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        self.owner.policy_requests.append(request)
        if self.owner.fail_policy:
            raise PolicyCallError("provider_timeout", "simulated timeout")
        selected = next(
            option.option_id
            for option in request.options
            if '"version": "v2"' in option.files[0][1]
            and '"offline": True' in option.files[0][1]
        )
        request_hash = policy_request_hash(request)
        return PolicyResponse(
            request_id=request.request_id,
            provider="fake",
            model_id="fake-policy",
            endpoint_identity="local://fake",
            selected_option_id=selected,
            optional_patch=None,
            concise_rationale="fake rationale",
            provider_request_id=f"provider-{request.request_id}",
            usage=PolicyUsage(input_tokens=10, output_tokens=2),
            request_hash=request_hash,
            response_hash=hash_text(f"response:{request.request_id}"),
            started_at_utc="2026-07-16T00:00:00+00:00",
            ended_at_utc="2026-07-16T00:00:01+00:00",
            latency_seconds=1.0,
            retry_count=0,
            format_repair_used=False,
        )


class FakeMemory:
    def __init__(self, owner: FakeFactory, isolation: TaskIsolation) -> None:
        self.owner = owner
        self.isolation = isolation
        self.items: list[InventoryItem] = []
        self.n_write = 0
        self.write_messages: list[list[dict[str, str]]] = []
        self.search_calls = 0

    def restore_write_count(self, n_write: int) -> None:
        self.n_write = n_write

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        self.owner.memory_write_calls += 1
        self.write_messages.append(messages)
        content = {
            0: (
                "Pipeline execution must remain completely offline; "
                "do not call cloud services."
            ),
            1: "Data leakage was found in the v1 implementation.",
            2: "The v2 branch is the current implementation after the leakage fix.",
            3: "The v2 branch passes the held-out integrity audit.",
        }.get(session_index, f"Session {session_index} completed.")
        memory_id = f"{self.isolation.task_id}-m{session_index}"
        item = InventoryItem(
            memory_id=memory_id,
            content=content,
            content_hash=hash_text(content),
            metadata=(("session_index", session_index),),
            created_at=f"t{session_index}",
            updated_at=f"t{session_index}",
            history_length=1,
        )
        self.items.append(item)
        self.n_write += 1
        inventory = InventorySnapshot(
            checkpoint_session=session_index,
            n_write=self.n_write,
            n_live=len(self.items),
            items=tuple(self.items),
            store_hash=hash_text("|".join(child.content_hash for child in self.items)),
            backend_count=len(self.items),
        )
        event = NativeMemoryEvent(
            operation_id=f"write-{session_index}",
            session_index=session_index,
            native_event="ADD",
            memory_id=memory_id,
            memory_text=content,
            old_content_hash=None,
            new_content_hash=item.content_hash,
            source="native_response",
            latency_seconds=0.01,
        )
        return WriteSessionResult(
            session_index=session_index,
            events=(event,),
            inventory=inventory,
            n_write=self.n_write,
            latency_seconds=0.01,
        )

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch:
        self.owner.memory_search_calls += 1
        self.search_calls += 1
        candidates = tuple(
            SearchCandidate(
                memory_id=item.memory_id,
                content=item.content,
                content_hash=item.content_hash,
                native_rank=rank,
                score=1.0 / rank,
                score_details=(),
                metadata=item.metadata,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for rank, item in enumerate(reversed(self.items), start=1)
        )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=hash_text(query),
            candidates=candidates,
            candidate_shortfall=True,
            latency_seconds=0.02,
        )


class FakeReranker:
    def __init__(self, owner: FakeFactory) -> None:
        self.owner = owner

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        self.owner.rerank_calls.append(
            (query, tuple(candidate.memory_id for candidate in candidates))
        )
        if self.owner.fail_reranker:
            raise TeiServiceError("reranker_failure", "simulated reranker failure")
        ordered = tuple(candidate.memory_id for candidate in reversed(candidates))
        if top_k is not None:
            ordered = ordered[:top_k]
        return RerankResult(
            ordered_memory_ids=ordered,
            scores=tuple(float(index) for index in range(len(ordered), 0, -1)),
            model="fake-reranker",
            revision="fake-revision",
            input_count=len(candidates),
            request_hash=hash_text(f"rerank:{query}:{len(candidates)}"),
            response_hash=hash_text("|".join(ordered)),
            latency_seconds=0.01,
        )


class FakeChecker:
    def __init__(self, spec: SoftwareMem0VerticalSpec) -> None:
        self.spec = spec

    def check_action(
        self,
        action: str,
        *,
        checkpoint_session: int,
        visible_state_ids: tuple[str, ...] | None = None,
    ) -> BehaviorResult:
        current = replay_plan(self.spec.plan, checkpoint_session).current
        expected = "safe_v2_offline" if "P2" in current else "stale_v1"
        correct = action == expected
        violated: tuple[str, ...] = ()
        drift: tuple[str, ...] = ()
        if action == "stale_v1" and "P2" in current:
            violated = ("P2", "U1")
            drift = ("stale-state", "goal-drift")
        elif action == "cloud_shortcut" and "C1" in current:
            violated = ("C1",)
            drift = (
                "constraint-influence-lost",
                "local-subgoal-overwrites-global-goal",
            )
        return BehaviorResult(
            score=1.0 if correct else 0.25,
            is_correct=correct,
            violated_state_ids=violated,
            passed_tests=("fake",) if correct else (),
            failed_tests=() if correct else ("fake",),
            drift_flags=drift,
            metadata=(
                ("visible_count", str(len(visible_state_ids or ()))),
            ),
        )


class FakeFactory:
    def __init__(
        self,
        spec: SoftwareMem0VerticalSpec,
        *,
        fail_reranker: bool = False,
        fail_policy: bool = False,
    ) -> None:
        self.spec = spec
        self.fail_reranker = fail_reranker
        self.fail_policy = fail_policy
        self.policy_requests: list[PolicyRequest] = []
        self.memory_instances: list[FakeMemory] = []
        self.isolations: list[TaskIsolation] = []
        self.rerank_calls: list[tuple[str, tuple[str, ...]]] = []
        self.memory_write_calls = 0
        self.memory_search_calls = 0

    def __call__(
        self,
        task: object,
        isolation: TaskIsolation,
    ) -> TaskComponents:
        self.isolations.append(isolation)
        condition = task.condition  # type: ignore[attr-defined]
        memory = None
        reranker = None
        if condition.startswith("mem0_"):
            memory = FakeMemory(self, isolation)
            self.memory_instances.append(memory)
        if condition == "mem0_controlled":
            reranker = FakeReranker(self)
        return TaskComponents(
            policy=FakePolicy(self),
            checker=FakeChecker(self.spec),
            memory=memory,
            reranker=reranker,
        )


def _fixture() -> tuple[
    SoftwareMem0VerticalSpec,
    tuple[object, ...],
]:
    spec = SoftwareMem0VerticalFamily.generate(
        42,
        n_sessions=4,
        trajectory_seed=2,
    )
    config = load_qualification_config(CONFIG)
    tasks = build_qualification_tasks(
        config,
        episode_ids=(spec.plan.episode_id,),
        run_identity="run-identity",
    )
    return spec, tasks


def test_matrix_executes_twelve_tasks_and_fifteen_scored_results(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    factory = FakeFactory(spec)
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    matrix = run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=factory,
        storage=storage,
        visible_k=5,
    )
    assert isinstance(matrix, QualificationMatrixResult)
    assert len(matrix.task_results) == 12
    conditions = [
        condition
        for task in matrix.task_results
        for condition in task.condition_results
    ]
    assert len(conditions) == 15
    assert all(condition.status == "complete" for condition in conditions)


def test_conditions_share_workspace_but_sessions_have_fresh_context(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    factory = FakeFactory(spec)
    matrix = run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=factory,
        storage=QualificationStorage(tmp_path / "run", run_identity="run-identity"),
    )
    grouped: dict[tuple[str, str], set[str]] = {}
    for task in matrix.task_results:
        for condition in task.condition_results:
            for sceu in condition.sceu_results:
                key = (task.policy_profile_id, sceu.opportunity_id)
                grouped.setdefault(key, set()).add(sceu.workspace_hash)
                assert all(
                    len(evaluation.request.messages) == 1
                    for evaluation in sceu.baseline_evaluations
                )
    assert all(len(hashes) == 1 for hashes in grouped.values())


def test_mem0_prefix_excludes_raw_workspace_and_continuation_answers(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    factory = FakeFactory(spec)
    run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=factory,
        storage=QualificationStorage(tmp_path / "run", run_identity="run-identity"),
    )
    transcripts = [
        message["content"]
        for memory in factory.memory_instances
        for write in memory.write_messages
        for message in write
    ]
    assert transcripts
    assert all('"artifacts"' not in transcript for transcript in transcripts)
    assert all("local run completed" not in transcript for transcript in transcripts)
    assert all("fake rationale" not in transcript for transcript in transcripts)
    assert all("selected_option" not in transcript for transcript in transcripts)


def test_controlled_readouts_share_one_store_and_candidate_set(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    controlled = tuple(task for task in tasks if task.condition == "mem0_controlled")
    factory = FakeFactory(spec)
    matrix = run_qualification_matrix(
        controlled,
        {spec.plan.episode_id: spec},
        component_factory=factory,
        storage=QualificationStorage(tmp_path / "run", run_identity="run-identity"),
    )
    assert len(factory.memory_instances) == 3
    assert all(
        memory.search_calls == len(spec.plan.sceu_units)
        for memory in factory.memory_instances
    )
    for task in matrix.task_results:
        native, common = task.condition_results
        assert native.result_id != common.result_id
        for native_sceu, common_sceu in zip(
            native.sceu_results,
            common.sceu_results,
            strict=True,
        ):
            assert native_sceu.candidate_memory_ids == common_sceu.candidate_memory_ids


def test_isolation_request_hashes_and_response_before_mapping_are_auditable(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    factory = FakeFactory(spec)
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    matrix = run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=factory,
        storage=storage,
    )
    mem_isolations = [
        isolation
        for isolation in factory.isolations
        if isolation.collection_name != "none"
    ]
    assert len({item.user_id for item in mem_isolations}) == 6
    assert len({item.run_id for item in mem_isolations}) == 6
    assert len({item.collection_name for item in mem_isolations}) == 6
    assert len({item.history_db_path for item in mem_isolations}) == 6
    for task in matrix.task_results:
        for condition in task.condition_results:
            for sceu in condition.sceu_results:
                for evaluation in sceu.baseline_evaluations:
                    assert evaluation.policy_request_hash == policy_request_hash(
                        evaluation.request
                    )
                    assert evaluation.model_visible_context_hash == hash_text(
                        "\n\n".join(evaluation.model_visible_blocks)
                    )
    paths = [path for _, path in storage.operation_log]
    for evaluation_path in [
        path for path in paths if "/evaluations/" in f"/{path}"
    ]:
        response_path = evaluation_path.replace("/evaluations/", "/responses/")
        assert response_path in paths
        assert paths.index(response_path) < paths.index(evaluation_path)


def test_completed_cells_resume_without_external_calls(tmp_path: Path) -> None:
    spec, tasks = _fixture()
    task = next(task for task in tasks if task.condition == "mem0_controlled")
    factory = FakeFactory(spec)
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    first = run_qualification_task(
        task,
        spec,
        components=factory(
            task,
            TaskIsolation.for_task(task, storage.task_directory(task)),
        ),
        storage=storage,
    )
    counts = (
        len(factory.policy_requests),
        factory.memory_write_calls,
        factory.memory_search_calls,
        len(factory.rerank_calls),
    )
    second = run_qualification_task(
        task,
        spec,
        components=factory(
            task,
            TaskIsolation.for_task(task, storage.task_directory(task)),
        ),
        storage=storage,
    )
    assert first == second
    assert counts == (
        len(factory.policy_requests),
        factory.memory_write_calls,
        factory.memory_search_calls,
        len(factory.rerank_calls),
    )


def test_task_result_has_a_portable_lossless_json_round_trip(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    task = next(task for task in tasks if task.condition == "mem0_controlled")
    factory = FakeFactory(spec)
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    result = run_qualification_task(
        task,
        spec,
        components=factory(
            task,
            TaskIsolation.for_task(task, storage.task_directory(task)),
        ),
        storage=storage,
    )

    restored = qualification_task_result_from_dict(asdict(result))

    assert restored == result


def test_reranker_failure_does_not_invalidate_native_readout(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    task = next(task for task in tasks if task.condition == "mem0_controlled")
    factory = FakeFactory(spec, fail_reranker=True)
    result = run_qualification_task(
        task,
        spec,
        components=factory(
            task,
            TaskIsolation.for_task(
                task,
                QualificationStorage(
                    tmp_path / "run",
                    run_identity="run-identity",
                ).task_directory(task),
            ),
        ),
        storage=QualificationStorage(
            tmp_path / "run",
            run_identity="run-identity",
        ),
    )
    native, common = result.condition_results
    assert native.status == "complete"
    assert common.status == "failed"
    assert common.error_class == "reranker_failure"
    assert result.status == "partial"


def test_policy_failure_is_typed_and_does_not_create_an_evaluation_cell(
    tmp_path: Path,
) -> None:
    spec, tasks = _fixture()
    task = next(task for task in tasks if task.condition == "workspace_only")
    factory = FakeFactory(spec, fail_policy=True)
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    result = run_qualification_task(
        task,
        spec,
        components=factory(
            task,
            TaskIsolation.for_task(task, storage.task_directory(task)),
        ),
        storage=storage,
    )
    assert result.status == "failed"
    assert result.condition_results[0].error_class == "provider_timeout"
    assert not list(storage.task_directory(task).glob("**/evaluations/*.json"))


def test_task_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    spec, tasks = _fixture()
    task = tasks[0]
    storage = QualificationStorage(tmp_path / "run", run_identity="run-identity")
    storage.prepare_task(task, episode_hash=spec.surface_hash)
    with pytest.raises(QualificationStorageError) as caught:
        storage.prepare_task(
            replace(task, task_payload_hash="different"),
            episode_hash=spec.surface_hash,
        )
    assert caught.value.error_class == "identity_mismatch"
