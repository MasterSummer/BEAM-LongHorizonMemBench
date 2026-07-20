from __future__ import annotations

from pathlib import Path

from lhmsb.adapters.mem0_qualification import (
    CandidateSearch,
    InventoryItem,
    InventorySnapshot,
    NativeMemoryEvent,
    ProviderUsageEvent,
    SearchCandidate,
    WriteSessionResult,
)
from lhmsb.datasets.mem0_stateful_pipeline import (
    freeze_mem0_stateful,
    generate_mem0_stateful_to_staging,
    verify_mem0_stateful,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import BehaviorResult
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.qualification.config import (
    build_qualification_tasks,
    load_qualification_config,
)
from lhmsb.qualification.preflight import load_mem0_specs
from lhmsb.qualification.providers import (
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)
from lhmsb.qualification.report import write_qualification_report
from lhmsb.qualification.runner import (
    QualificationMatrixResult,
    TaskComponents,
    TaskIsolation,
    hash_text,
    policy_request_hash,
    run_qualification_matrix,
)
from lhmsb.qualification.schema import QualificationTask
from lhmsb.qualification.storage import QualificationStorage
from lhmsb.qualification.tei import (
    RerankCandidate,
    RerankResult,
)
from lhmsb.qualification.validate import validate_qualification_artifacts

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"


class CausalFakePolicy:
    """Select an action from exact model-visible state, including LOO changes."""

    def __init__(self, owner: CausalFakeFactory) -> None:
        self.owner = owner

    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        self.owner.policy_requests.append(request)
        visible = request.messages[0].content.casefold()
        if "v2 is the current implementation after the leakage fix" not in visible:
            desired_version = '"version": "v1"'
            desired_profiler = '"profiler_backend": "local"'
        elif "pipeline execution must remain completely offline" not in visible:
            desired_version = '"version": "v2"'
            desired_profiler = '"profiler_backend": "hosted"'
        else:
            desired_version = '"version": "v2"'
            desired_profiler = '"profiler_backend": "local"'
        selected = next(
            option.option_id
            for option in request.options
            if desired_version in option.files[0][1]
            and desired_profiler in option.files[0][1]
        )
        request_hash = policy_request_hash(request)
        return PolicyResponse(
            request_id=request.request_id,
            provider="fake",
            model_id="causal-fake-policy",
            endpoint_identity="local://causal-fake",
            selected_option_id=selected,
            optional_patch=None,
            concise_rationale="deterministic fixture",
            provider_request_id=f"provider-{request.request_id}",
            usage=PolicyUsage(input_tokens=10, output_tokens=2),
            request_hash=request_hash,
            response_hash=hash_text(f"response:{request.request_id}:{selected}"),
            started_at_utc="2026-07-16T00:00:00+00:00",
            ended_at_utc="2026-07-16T00:00:01+00:00",
            latency_seconds=1.0,
            retry_count=0,
            format_repair_used=False,
        )


class CausalFakeMemory:
    def __init__(
        self,
        owner: CausalFakeFactory,
        isolation: TaskIsolation,
    ) -> None:
        self.owner = owner
        self.isolation = isolation
        self.items: list[InventoryItem] = []
        self.n_write = 0

    def restore_write_count(self, n_write: int) -> None:
        self.n_write = n_write

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult:
        del messages, metadata
        self.owner.memory_write_calls += 1
        content = {
            0: (
                "Pipeline execution must remain completely offline; "
                "do not call cloud services."
            ),
            1: "Data leakage was found in the v1 implementation.",
            2: "V2 is the current implementation after the leakage fix.",
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
            store_hash=hash_text(
                "|".join(child.content_hash for child in self.items)
            ),
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
            usage_events=(
                _usage_event(
                    f"{self.isolation.task_id}:llm:{session_index}",
                    "memory_internal_llm",
                    input_tokens=8,
                    output_tokens=2,
                ),
                _usage_event(
                    f"{self.isolation.task_id}:embed-write:{session_index}",
                    "embedding",
                    input_tokens=None,
                    output_tokens=None,
                ),
            ),
        )

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch:
        self.owner.memory_search_calls += 1
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
            usage_events=(
                _usage_event(
                    (
                        f"{self.isolation.task_id}:embed-search:"
                        f"{checkpoint_session}"
                    ),
                    "embedding",
                    input_tokens=None,
                    output_tokens=None,
                ),
            ),
        )


class CausalFakeReranker:
    def __init__(self, owner: CausalFakeFactory) -> None:
        self.owner = owner

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        self.owner.rerank_calls += 1
        ordered = tuple(candidate.memory_id for candidate in reversed(candidates))
        if top_k is not None:
            ordered = ordered[:top_k]
        return RerankResult(
            ordered_memory_ids=ordered,
            scores=tuple(
                float(index)
                for index in range(len(ordered), 0, -1)
            ),
            model="fake-reranker",
            revision="fake-revision",
            input_count=len(candidates),
            request_hash=hash_text(f"rerank:{query}:{len(candidates)}"),
            response_hash=hash_text("|".join(ordered)),
            latency_seconds=0.01,
        )


class CausalFakeChecker:
    def __init__(self, spec: SoftwareMem0VerticalSpec) -> None:
        self.spec = spec

    def check_action(
        self,
        action: str,
        *,
        checkpoint_session: int,
        visible_state_ids: tuple[str, ...] | None = None,
        opportunity_id: str | None = None,
    ) -> BehaviorResult:
        del visible_state_ids
        current = replay_plan(self.spec.plan, checkpoint_session).current
        local_exception = opportunity_id in {
            "opp-local-valid",
            "opp-local-valid-recheck",
        } and "L1" in current
        expected = (
            "cloud_shortcut"
            if local_exception
            else ("safe_v2_offline" if "P2" in current else "stale_v1")
        )
        correct = action == expected
        violated: tuple[str, ...] = ()
        drift: tuple[str, ...] = ()
        if action == "stale_v1" and "P2" in current:
            violated = ("P2", "U1")
            drift = ("stale-state", "goal-drift")
        elif action == "cloud_shortcut" and "C1" in current and not local_exception:
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
            metadata=(),
        )


class CausalFakeFactory:
    def __init__(self, spec: SoftwareMem0VerticalSpec) -> None:
        self.spec = spec
        self.policy_requests: list[PolicyRequest] = []
        self.memory_write_calls = 0
        self.memory_search_calls = 0
        self.rerank_calls = 0
        self.isolations: list[TaskIsolation] = []

    def __call__(
        self,
        task: QualificationTask,
        isolation: TaskIsolation,
    ) -> TaskComponents:
        self.isolations.append(isolation)
        memory = None
        reranker = None
        if task.condition.startswith("mem0_"):
            memory = CausalFakeMemory(self, isolation)
        if task.condition == "mem0_controlled":
            reranker = CausalFakeReranker(self)
        return TaskComponents(
            policy=CausalFakePolicy(self),
            checker=CausalFakeChecker(self.spec),
            memory=memory,
            reranker=reranker,
        )


def _usage_event(
    call_id: str,
    component: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
) -> ProviderUsageEvent:
    return ProviderUsageEvent(
        call_id=call_id,
        component=component,
        provider="fake",
        model_id=f"fake-{component}",
        endpoint_identity="local://fake",
        request_hash=hash_text(f"request:{call_id}"),
        response_hash=hash_text(f"response:{call_id}"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=None,
        reasoning_tokens=None,
        usage_observed=input_tokens is not None or output_tokens is not None,
        input_count=1,
        latency_seconds=0.001,
        retry_count=0,
        error_class=None,
        started_at_utc="2026-07-16T00:00:00+00:00",
        ended_at_utc="2026-07-16T00:00:00.001000+00:00",
    )


def test_four_session_mem0_vertical_slice_is_frozen_resumable_and_auditable(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        staging,
        seeds=(42,),
        n_episodes=1,
        n_sessions=4,
    )
    manifest = freeze_mem0_stateful(staging, frozen)
    verification = verify_mem0_stateful(frozen)
    specs = load_mem0_specs(frozen)

    assert len(generated) == len(specs) == manifest.n_episodes == 1
    assert manifest.n_sessions == specs[0].plan.n_sessions == 4
    assert verification.ok

    spec = specs[0]
    config = load_qualification_config(CONFIG)
    run_identity = "mem0-vertical-slice"
    tasks = build_qualification_tasks(
        config,
        episode_ids=(spec.plan.episode_id,),
        run_identity=run_identity,
    )
    storage = QualificationStorage(tmp_path / "run", run_identity=run_identity)
    first_factory = CausalFakeFactory(spec)
    matrix = run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=first_factory,
        storage=storage,
        visible_k=config.retrieval.visible_k,
    )

    assert isinstance(matrix, QualificationMatrixResult)
    assert len(matrix.task_results) == 12
    scored = [
        result
        for task_result in matrix.task_results
        for result in task_result.condition_results
    ]
    assert len(scored) == 15
    assert all(result.status == "complete" for result in scored)

    controlled = next(
        result
        for result in matrix.task_results
        if result.condition == "mem0_controlled"
    )
    causal_row = next(
        row
        for condition in controlled.condition_results
        for row in condition.sceu_results
        if any(
            intervention.classification.behaviorally_used
            and intervention.classification.label == "beneficial"
            for intervention in row.interventions
        )
    )
    causal_intervention = next(
        intervention
        for intervention in causal_row.interventions
        if intervention.classification.behaviorally_used
        and intervention.classification.label == "beneficial"
    )
    target_memory_id = causal_intervention.target_memory_id
    event_memory_ids = {
        event.memory_id
        for write in controlled.writes
        for event in write.events
    }
    inventory_memory_ids = {
        item.memory_id
        for write in controlled.writes
        for item in write.inventory.items
    }
    trace = next(
        item
        for item in controlled.retrieval_traces
        if item.trace_id == causal_row.retrieval_trace_id
    )
    assert target_memory_id in event_memory_ids
    assert target_memory_id in inventory_memory_ids
    assert target_memory_id in trace.candidate_memory_ids
    assert target_memory_id in causal_row.retrieved_memory_ids
    assert target_memory_id in causal_row.model_visible_memory_ids
    assert causal_intervention.classification.action_changed
    assert causal_intervention.classification.checker_changed
    assert causal_row.behavior.is_correct

    report_directory = tmp_path / "report"
    artifacts = write_qualification_report(
        matrix,
        {spec.plan.episode_id: spec},
        report_directory,
        run_metadata={
            "dataset_manifest_sha256": manifest.files.get(
                "MANIFEST.json",
                "self-excluded",
            ),
            "planned_task_count": len(tasks),
        },
    )
    validation = validate_qualification_artifacts(
        report_directory,
        expected_run_identity=run_identity,
    )
    assert artifacts.manifest_sha256
    assert validation.ok, validation.errors
    api_usage = (
        report_directory / "api_usage.jsonl"
    ).read_text(encoding="utf-8")
    assert '"call_kind":"memory_internal_llm"' in api_usage
    assert '"call_kind":"embedding"' in api_usage
    assert '"call_kind":"reranker"' in api_usage

    external_counts = (
        len(first_factory.policy_requests),
        first_factory.memory_write_calls,
        first_factory.memory_search_calls,
        first_factory.rerank_calls,
    )
    assert all(count > 0 for count in external_counts)
    resumed_factory = CausalFakeFactory(spec)
    resumed = run_qualification_matrix(
        tasks,
        {spec.plan.episode_id: spec},
        component_factory=resumed_factory,
        storage=storage,
        visible_k=config.retrieval.visible_k,
    )
    assert resumed == matrix
    assert len(resumed_factory.policy_requests) == 0
    assert resumed_factory.memory_write_calls == 0
    assert resumed_factory.memory_search_calls == 0
    assert resumed_factory.rerank_calls == 0
