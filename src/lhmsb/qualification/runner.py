"""Resumable Mem0 qualification execution over leak-free Software episodes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from lhmsb.adapters.mem0_qualification import Mem0QualificationError
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import BehaviorResult
from lhmsb.longhorizon.attribution import (
    AttributionMethod,
    FactSignature,
    MemoryAttribution,
    ProvenanceMode,
    attribute_memory,
    build_software_fact_signatures,
    eligible_write_state_ids,
)
from lhmsb.longhorizon.drift import (
    DriftEvidence,
    classify_long_horizon_drift,
)
from lhmsb.longhorizon.interventions import (
    CausalUseLabel,
    CausalUseResult,
    ContinuationOutcome,
    EffectDirection,
    InterventionKind,
    MemoryRole,
    classify_causal_use,
)
from lhmsb.longhorizon.public_surface import (
    PublicActionOption,
    PublicContinuation,
)
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import SCEU, ActionSpec, SessionSurface, StateUnit
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    MemoryMutationEvent,
    MemoryRuntime,
    MemoryTraceValidationError,
    ProviderUsageEvent,
    SearchCandidate,
    WriteSessionResult,
)
from lhmsb.qualification.providers import (
    PolicyCallError,
    PolicyClient,
    PolicyMessage,
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)
from lhmsb.qualification.schema import (
    QualificationTask,
    ScoredCondition,
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

RunStatus = Literal["complete", "partial", "failed"]
ConditionStatus = Literal["complete", "failed"]

_SYSTEM_PROMPT = (
    "Continue the software project using only the supplied current-session "
    "surface, workspace, and optional memory context. Select exactly one "
    "opaque implementation option."
)


class QualificationRunError(RuntimeError):
    """Typed runner failure that must not silently change conditions."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class RerankerRuntime(Protocol):
    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult: ...


class BehaviorChecker(Protocol):
    def check_action(
        self,
        action: str,
        *,
        checkpoint_session: int,
        visible_state_ids: tuple[str, ...] | None = None,
        opportunity_id: str | None = None,
    ) -> BehaviorResult: ...


@dataclass(frozen=True)
class TaskIsolation:
    task_id: str
    user_id: str
    run_id: str
    collection_name: str
    history_db_path: Path

    @classmethod
    def for_task(
        cls,
        task: QualificationTask,
        task_directory: Path,
    ) -> TaskIsolation:
        return cls(
            task_id=task.task_id,
            user_id=f"lhmsb-user--{task.task_id}",
            run_id=f"lhmsb-run--{task.task_id}",
            collection_name=task.store_namespace,
            history_db_path=task_directory / "store" / "history.sqlite",
        )


@dataclass(frozen=True)
class TaskComponents:
    policy: PolicyClient
    checker: BehaviorChecker
    memory: MemoryRuntime | None = None
    reranker: RerankerRuntime | None = None


TaskComponentFactory = Callable[[QualificationTask, TaskIsolation], TaskComponents]


@dataclass(frozen=True)
class MemoryAlignmentSnapshot:
    checkpoint_session: int
    inventory_store_hash: str
    attributions: tuple[MemoryAttribution, ...]

    @property
    def provenance_modes(self) -> tuple[str, ...]:
        """Return the provenance modes represented at this checkpoint."""
        return tuple(sorted({item.provenance_mode for item in self.attributions}))


@dataclass(frozen=True)
class RetrievalTrace:
    trace_id: str
    sceu_id: str
    opportunity_id: str
    checkpoint_session: int
    query: str
    query_hash: str
    candidates: tuple[SearchCandidate, ...]
    candidate_memory_ids: tuple[str, ...]
    native_retrieved_memory_ids: tuple[str, ...]
    common_reranked_memory_ids: tuple[str, ...]
    candidate_shortfall: bool
    search_latency_seconds: float
    rerank_result: RerankResult | None = None
    internal_usage: tuple[ProviderUsageEvent, ...] = ()


@dataclass(frozen=True)
class PolicyEvaluation:
    call_id: str
    call_kind: str
    request: PolicyRequest
    response: PolicyResponse
    selected_action_id: str
    checker_result: BehaviorResult
    outcome: ContinuationOutcome
    normalized_drift_flags: tuple[str, ...]
    workspace_hash: str
    transcript_hash: str
    policy_request_hash: str
    model_visible_memory_ids: tuple[str, ...]
    model_visible_blocks: tuple[str, ...]
    model_visible_context_hash: str


@dataclass(frozen=True)
class InterventionRun:
    intervention_kind: str
    target_memory_id: str
    replacement_memory_id: str | None
    evaluations: tuple[PolicyEvaluation, PolicyEvaluation]
    classification: CausalUseResult
    baseline_memory_count: int = 0
    intervention_memory_count: int = 0
    count_contrast: str | None = None


@dataclass(frozen=True)
class SCEURunResult:
    result_id: str
    sceu_id: str
    opportunity_id: str
    checkpoint_session: int
    matched_group: str
    control_kind: str
    workspace_hash: str
    candidate_memory_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    model_visible_memory_ids: tuple[str, ...]
    selected_option_id: str
    selected_action_id: str
    behavior: ContinuationOutcome
    normalized_drift_flags: tuple[str, ...]
    baseline_stable: bool
    baseline_evaluations: tuple[PolicyEvaluation, ...]
    interventions: tuple[InterventionRun, ...]
    retrieval_trace_id: str | None


@dataclass(frozen=True)
class ConditionRunResult:
    result_id: str
    condition: str
    readout: str
    status: ConditionStatus
    sceu_results: tuple[SCEURunResult, ...]
    error_class: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class QualificationTaskResult:
    task_id: str
    episode_id: str
    policy_profile_id: str
    condition: str
    status: RunStatus
    condition_results: tuple[ConditionRunResult, ...]
    writes: tuple[WriteSessionResult, ...]
    alignments: tuple[MemoryAlignmentSnapshot, ...]
    retrieval_traces: tuple[RetrievalTrace, ...]
    error_class: str | None = None
    error_message: str | None = None
    qdrant_store_bytes: int | None = None
    history_store_bytes: int | None = None


@dataclass(frozen=True)
class QualificationMatrixResult:
    run_identity: str
    task_results: tuple[QualificationTaskResult, ...]


def qualification_task_result_from_dict(
    data: Mapping[str, object],
) -> QualificationTaskResult:
    """Restore one portable task result written with ``dataclasses.asdict``."""
    return QualificationTaskResult(
        task_id=str(data["task_id"]),
        episode_id=str(data["episode_id"]),
        policy_profile_id=str(data["policy_profile_id"]),
        condition=str(data["condition"]),
        status=cast(RunStatus, str(data["status"])),
        condition_results=tuple(
            _condition_run_result_from_dict(item)
            for item in _mapping_sequence(data.get("condition_results"))
        ),
        writes=tuple(
            _write_result_from_dict(item)
            for item in _mapping_sequence(data.get("writes"))
        ),
        alignments=tuple(
            _alignment_from_dict(item)
            for item in _mapping_sequence(data.get("alignments"))
        ),
        retrieval_traces=tuple(
            _retrieval_trace_from_dict(item)
            for item in _mapping_sequence(data.get("retrieval_traces"))
        ),
        error_class=_optional_string(data.get("error_class")),
        error_message=_optional_string(data.get("error_message")),
        qdrant_store_bytes=_optional_integer(
            data.get("qdrant_store_bytes")
        ),
        history_store_bytes=_optional_integer(
            data.get("history_store_bytes")
        ),
    )


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def policy_request_hash(request: PolicyRequest) -> str:
    """Hash exact model-facing semantics while excluding bookkeeping ID."""
    digest: str = canonical_hash(
        {
            "system_prompt": request.system_prompt,
            "messages": [asdict(message) for message in request.messages],
            "options": [option.to_dict() for option in request.options],
            "max_output_tokens": request.max_output_tokens,
        }
    )
    return digest


def run_qualification_matrix(
    tasks: Sequence[QualificationTask],
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    component_factory: TaskComponentFactory,
    storage: QualificationStorage,
    visible_k: int = 5,
    max_output_tokens: int = 512,
) -> QualificationMatrixResult:
    """Execute a deterministic task matrix through one shared core runner."""
    results: list[QualificationTaskResult] = []
    for task in tasks:
        try:
            spec = specs[task.episode_id]
        except KeyError as exc:
            raise QualificationRunError(
                "trace_incomplete",
                f"missing episode spec: {task.episode_id}",
            ) from exc
        isolation = TaskIsolation.for_task(
            task,
            storage.task_directory(task),
        )
        components = component_factory(task, isolation)
        results.append(
            run_qualification_task(
                task,
                spec,
                components=components,
                storage=storage,
                visible_k=visible_k,
                max_output_tokens=max_output_tokens,
            )
        )
    return QualificationMatrixResult(
        run_identity=storage.run_identity,
        task_results=tuple(results),
    )


def run_qualification_task(
    task: QualificationTask,
    spec: SoftwareMem0VerticalSpec,
    *,
    components: TaskComponents,
    storage: QualificationStorage,
    visible_k: int = 5,
    max_output_tokens: int = 512,
) -> QualificationTaskResult:
    """Execute or resume one independently isolated qualification task."""
    if visible_k < 1:
        raise ValueError("visible_k must be positive")
    storage.prepare_task(task, episode_hash=spec.surface_hash)
    if task.condition.startswith("mem0_") and components.memory is None:
        return _terminal_task_failure(
            task,
            "trace_incomplete",
            "Mem0 condition requires a memory runtime",
        )

    condition_rows: dict[str, list[SCEURunResult]] = {
        item.result_id: [] for item in task.scored_conditions
    }
    condition_errors: dict[str, tuple[str, str]] = {}
    writes: list[WriteSessionResult] = []
    alignments: list[MemoryAlignmentSnapshot] = []
    retrieval_traces: list[RetrievalTrace] = []
    signatures = build_software_fact_signatures(spec.plan)
    latest_alignment: dict[str, MemoryAttribution] = {}
    latest_inventory: InventorySnapshot | None = None
    public_by_opportunity = {
        item.opportunity_id: item for item in spec.public_continuations
    }
    sceu_by_session: dict[int, list[SCEU]] = {}
    for sceu in spec.plan.sceu_units:
        sceu_by_session.setdefault(sceu.checkpoint_session, []).append(sceu)

    fatal_error: tuple[str, str] | None = None
    try:
        for session_index in range(spec.plan.n_sessions):
            if components.memory is not None:
                write = _write_or_load(
                    task,
                    spec,
                    components.memory,
                    storage,
                    session_index=session_index,
                    previous_inventory=(
                        writes[-1].inventory if writes else None
                    ),
                )
                writes.append(write)
                latest_inventory = write.inventory
                alignment = _alignment_snapshot(
                    task,
                    spec,
                    write.inventory,
                    signatures,
                    storage,
                    lifecycle_events=write.events,
                    prior_attributions=latest_alignment,
                )
                alignments.append(alignment)
                latest_alignment = {
                    item.memory_id: item for item in alignment.attributions
                }

            for sceu in sceu_by_session.get(session_index, ()):
                public = public_by_opportunity[sceu.opportunity_id]
                search: CandidateSearch | None = None
                rerank: RerankResult | None = None
                trace: RetrievalTrace | None = None
                if components.memory is not None:
                    if latest_inventory is None:
                        raise QualificationRunError(
                            "trace_incomplete",
                            "memory checkpoint lacks an inventory snapshot",
                        )
                    search = _search_or_load(
                        task,
                        components.memory,
                        storage,
                        sceu=sceu,
                        public=public,
                        inventory=latest_inventory,
                    )
                    native_ids = tuple(
                        item.memory_id for item in search.candidates[:visible_k]
                    )
                    common_ids: tuple[str, ...] = ()
                    common_condition = next(
                        (
                            item
                            for item in task.scored_conditions
                            if item.readout == "common_rerank"
                        ),
                        None,
                    )
                    if common_condition is not None and (
                        common_condition.result_id not in condition_errors
                    ):
                        if components.reranker is None:
                            condition_errors[common_condition.result_id] = (
                                "reranker_failure",
                                "controlled common readout requires a reranker",
                            )
                        else:
                            try:
                                rerank = _rerank_or_load(
                                    task,
                                    components.reranker,
                                    storage,
                                    sceu=sceu,
                                    search=search,
                                    visible_k=visible_k,
                                )
                                common_ids = rerank.ordered_memory_ids
                            except QualificationStorageError:
                                raise
                            except Exception as exc:
                                condition_errors[common_condition.result_id] = (
                                    _error_class(exc, default="reranker_failure"),
                                    str(exc),
                                )
                    trace = RetrievalTrace(
                        trace_id=f"{task.task_id}:{sceu.sceu_id}",
                        sceu_id=sceu.sceu_id,
                        opportunity_id=sceu.opportunity_id,
                        checkpoint_session=sceu.checkpoint_session,
                        query=search.query,
                        query_hash=search.query_hash,
                        candidates=search.candidates,
                        candidate_memory_ids=tuple(
                            item.memory_id for item in search.candidates
                        ),
                        native_retrieved_memory_ids=native_ids,
                        common_reranked_memory_ids=common_ids,
                        candidate_shortfall=search.candidate_shortfall,
                        search_latency_seconds=search.latency_seconds,
                        rerank_result=rerank,
                        internal_usage=search.usage_events,
                    )
                    retrieval_traces.append(trace)

                for scored in task.scored_conditions:
                    if scored.result_id in condition_errors:
                        continue
                    try:
                        row = _run_sceu_branch(
                            task,
                            scored,
                            spec,
                            components,
                            storage,
                            sceu=sceu,
                            public=public,
                            surface=spec.plan.sessions[session_index],
                            search=search,
                            retrieval_trace=trace,
                            alignments=latest_alignment,
                            visible_k=visible_k,
                            max_output_tokens=max_output_tokens,
                        )
                    except QualificationStorageError:
                        raise
                    except Exception as exc:
                        condition_errors[scored.result_id] = (
                            _error_class(exc),
                            str(exc),
                        )
                        continue
                    condition_rows[scored.result_id].append(row)
    except QualificationStorageError:
        raise
    except Exception as exc:
        fatal_error = (_error_class(exc), str(exc))
        for scored in task.scored_conditions:
            condition_errors.setdefault(scored.result_id, fatal_error)

    condition_results = tuple(
        _condition_result(
            scored,
            rows=condition_rows[scored.result_id],
            error=condition_errors.get(scored.result_id),
            expected_sceu=len(spec.plan.sceu_units),
        )
        for scored in task.scored_conditions
    )
    completed = sum(item.status == "complete" for item in condition_results)
    if completed == len(condition_results):
        status: RunStatus = "complete"
    elif completed:
        status = "partial"
    else:
        status = "failed"
    first_error = fatal_error or next(
        (
            (item.error_class or "trace_incomplete", item.error_message or "")
            for item in condition_results
            if item.status == "failed"
        ),
        None,
    )
    result = QualificationTaskResult(
        task_id=task.task_id,
        episode_id=task.episode_id,
        policy_profile_id=task.policy_profile_id,
        condition=task.condition,
        status=status,
        condition_results=condition_results,
        writes=tuple(writes),
        alignments=tuple(alignments),
        retrieval_traces=tuple(retrieval_traces),
        error_class=first_error[0] if first_error else None,
        error_message=first_error[1] if first_error else None,
    )
    final_input_hash = canonical_hash(
        {
            "task_payload_hash": task.task_payload_hash,
            "surface_hash": spec.surface_hash,
            "visible_k": visible_k,
            "max_output_tokens": max_output_tokens,
        }
    )
    storage.save_cell(
        task,
        "task_result.json",
        input_hash=final_input_hash,
        payload=asdict(result),
    )
    return result


def _write_or_load(
    task: QualificationTask,
    spec: SoftwareMem0VerticalSpec,
    memory: MemoryRuntime,
    storage: QualificationStorage,
    *,
    session_index: int,
    previous_inventory: InventorySnapshot | None = None,
) -> WriteSessionResult:
    transcript = spec.write_transcript(session_index)
    input_hash = canonical_hash(
        {
            "task_payload_hash": task.task_payload_hash,
            "session_index": session_index,
            "transcript": transcript,
        }
    )
    relative = f"prefix/writes/session_{session_index:03d}.json"
    stored = storage.load_cell(task, relative, input_hash=input_hash)
    if stored is not None:
        result = _write_result_from_dict(_mapping(stored, relative))
        memory.restore_write_count(result.n_write)
    else:
        result = memory.write_session(
            [{"role": "user", "content": transcript}],
            session_index=session_index,
            metadata={
                "write_origin": "system_managed_extraction",
                "episode_id": task.episode_id,
            },
        )
    completed = _complete_write_provenance(
        result,
        previous_inventory=previous_inventory,
    )
    if completed != result:
        result = completed
    storage.save_cell(
        task,
        relative,
        input_hash=input_hash,
        payload=asdict(result),
    )
    return result


def _complete_write_provenance(
    result: WriteSessionResult,
    *,
    previous_inventory: InventorySnapshot | None,
) -> WriteSessionResult:
    """Complete lifecycle provenance whenever inventory makes it observable.

    Native adapters are authoritative when they return mutation events.  When a
    backend only exposes before/after inventory, the deterministic diff is kept
    as an ``INFERRED_*`` event with an explicit source marker.  If a backend
    reports a write but exposes neither a native event nor an inventory delta,
    the empty event list is retained and the report marks that write
    ``incomplete`` rather than inventing a memory object.
    """
    before = {
        item.memory_id: item for item in previous_inventory.items
    } if previous_inventory is not None else {}
    after = {item.memory_id: item for item in result.inventory.items}
    events = list(result.events)
    represented = {event.memory_id for event in events}
    operation_index = len(events)

    def add_event(
        *,
        native_event: str,
        memory_id: str,
        memory_text: str,
        old_hash: str | None,
        new_hash: str | None,
    ) -> None:
        nonlocal operation_index
        events.append(
            MemoryMutationEvent(
                operation_id=(
                    f"session-{result.session_index:03d}-inferred-"
                    f"{operation_index:04d}"
                ),
                session_index=result.session_index,
                native_event=native_event,
                memory_id=memory_id,
                memory_text=memory_text,
                old_content_hash=old_hash,
                new_content_hash=new_hash,
                source="inventory_diff",
                latency_seconds=0.0,
            )
        )
        operation_index += 1

    for memory_id in sorted(set(after) - set(before)):
        if memory_id in represented:
            continue
        item = after[memory_id]
        add_event(
            native_event="INFERRED_ADD",
            memory_id=memory_id,
            memory_text=item.content,
            old_hash=None,
            new_hash=item.content_hash,
        )
    for memory_id in sorted(set(before) - set(after)):
        if memory_id in represented:
            continue
        item = before[memory_id]
        add_event(
            native_event="INFERRED_DELETE",
            memory_id=memory_id,
            memory_text=item.content,
            old_hash=item.content_hash,
            new_hash=None,
        )
    for memory_id in sorted(set(before) & set(after)):
        if memory_id in represented:
            continue
        old = before[memory_id]
        new = after[memory_id]
        if old.content_hash != new.content_hash:
            add_event(
                native_event="INFERRED_UPDATE",
                memory_id=memory_id,
                memory_text=new.content,
                old_hash=old.content_hash,
                new_hash=new.content_hash,
            )

    mutation_count = sum(
        event.normalized_event in {"add", "update", "delete"}
        for event in events
    )
    inferred_n_write = max(
        result.n_write,
        (previous_inventory.n_write if previous_inventory is not None else 0)
        + mutation_count,
    )
    if not events and inferred_n_write == 0:
        return result
    if (
        tuple(events) == result.events
        and inferred_n_write == result.n_write
    ):
        return result
    inventory = result.inventory
    if inventory.n_write != inferred_n_write:
        inventory = InventorySnapshot(
            checkpoint_session=inventory.checkpoint_session,
            n_write=inferred_n_write,
            n_live=inventory.n_live,
            items=inventory.items,
            store_hash=inventory.store_hash,
            backend_count=inventory.backend_count,
        )
    return WriteSessionResult(
        session_index=result.session_index,
        events=tuple(events),
        inventory=inventory,
        n_write=inferred_n_write,
        latency_seconds=result.latency_seconds,
        usage_events=result.usage_events,
    )


def _alignment_snapshot(
    task: QualificationTask,
    spec: SoftwareMem0VerticalSpec,
    inventory: InventorySnapshot,
    signatures: tuple[FactSignature, ...],
    storage: QualificationStorage,
    *,
    lifecycle_events: Sequence[MemoryMutationEvent] = (),
    prior_attributions: Mapping[str, MemoryAttribution] | None = None,
) -> MemoryAlignmentSnapshot:
    attributions: list[MemoryAttribution] = []
    events_by_session: dict[int, tuple[str, ...]] = {}
    for session in range(spec.plan.n_sessions):
        events_by_session[session] = eligible_write_state_ids(
            spec.plan,
            session,
        )
    previous = prior_attributions or {}
    events_by_memory: dict[str, list[MemoryMutationEvent]] = defaultdict(list)
    for event in lifecycle_events:
        events_by_memory[event.memory_id].append(event)
    signature_by_state = {signature.state_id: signature for signature in signatures}
    for item in inventory.items:
        metadata = dict(item.metadata)
        metadata_session = metadata.get("session_index")
        lifecycle_candidates = events_by_memory.get(item.memory_id, [])
        lifecycle = next(
            (
                event
                for event in reversed(lifecycle_candidates)
                if event.new_content_hash == item.content_hash
            ),
            lifecycle_candidates[-1] if lifecycle_candidates else None,
        )
        prior = previous.get(item.memory_id)
        mode: ProvenanceMode
        if lifecycle is not None:
            mode = (
                "inferred"
                if lifecycle.source
                in {
                    "inventory_diff",
                    "inventory_delta",
                    "inventory_snapshot_diff",
                    "write_inventory_diff",
                }
                or lifecycle.native_event.startswith("INFERRED")
                else "native/exact"
            )
            source_session_value: int | None = lifecycle.session_index
            lifecycle_ids: tuple[str, ...] = (lifecycle.operation_id,)
        elif prior is not None:
            mode = prior.provenance_mode
            source_session_value = prior.source_session
            lifecycle_ids = prior.source_event_ids
        else:
            mode = "unavailable"
            source_session_value = (
                metadata_session if isinstance(metadata_session, int) else None
            )
            lifecycle_ids = ()
        source_session = (
            metadata_session
            if isinstance(metadata_session, int)
            else source_session_value
        )
        eligible = (
            events_by_session.get(source_session, ())
            if isinstance(source_session, int)
            else ()
        )
        attribution = attribute_memory(
            item.memory_id,
            item.content,
            signatures,
            unique_write_state_ids=eligible,
            provenance_mode=mode,
            source_event_ids=lifecycle_ids,
            source_session=source_session_value,
        )
        gold_event_ids = tuple(
            sorted(
                {
                    event_id
                    for state_id in attribution.state_ids
                    for event_id in (
                        signature_by_state[state_id].source_event_ids
                        if state_id in signature_by_state
                        else ()
                    )
                }
            )
        )
        # Keep both the lifecycle operation and evaluator state event IDs.  The
        # former answers “which backend mutation?”, the latter answers “which
        # latent fact did it represent?”.
        if gold_event_ids:
            attribution = MemoryAttribution(
                memory_id=attribution.memory_id,
                state_ids=attribution.state_ids,
                method=attribution.method,
                contributes_positive_coverage=attribution.contributes_positive_coverage,
                reason=attribution.reason,
                provenance_mode=attribution.provenance_mode,
                source_event_ids=tuple(
                    sorted(set(attribution.source_event_ids) | set(gold_event_ids))
                ),
                source_session=attribution.source_session,
            )
        attributions.append(attribution)
    snapshot = MemoryAlignmentSnapshot(
        checkpoint_session=inventory.checkpoint_session,
        inventory_store_hash=inventory.store_hash,
        attributions=tuple(attributions),
    )
    input_hash = canonical_hash(
        {
            "store_hash": inventory.store_hash,
            "signature_hash": canonical_hash([asdict(item) for item in signatures]),
        }
    )
    storage.save_cell(
        task,
        f"prefix/alignments/session_{inventory.checkpoint_session:03d}.json",
        input_hash=input_hash,
        payload=asdict(snapshot),
    )
    return snapshot


def _search_or_load(
    task: QualificationTask,
    memory: MemoryRuntime,
    storage: QualificationStorage,
    *,
    sceu: SCEU,
    public: PublicContinuation,
    inventory: InventorySnapshot,
) -> CandidateSearch:
    input_hash = canonical_hash(
        {
            "task_payload_hash": task.task_payload_hash,
            "sceu_id": sceu.sceu_id,
            "query": public.request,
            "store_hash": inventory.store_hash,
        }
    )
    relative = f"retrieval/{sceu.sceu_id}/candidate.json"
    stored = storage.load_cell(task, relative, input_hash=input_hash)
    if stored is not None:
        result = _candidate_search_from_dict(_mapping(stored, relative))
        result.validate_against_inventory(inventory)
        return result
    result = memory.search_candidates(
        public.request,
        checkpoint_session=sceu.checkpoint_session,
    )
    result.validate_against_inventory(inventory)
    storage.save_cell(
        task,
        relative,
        input_hash=input_hash,
        payload=asdict(result),
    )
    return result


def _rerank_or_load(
    task: QualificationTask,
    reranker: RerankerRuntime,
    storage: QualificationStorage,
    *,
    sceu: SCEU,
    search: CandidateSearch,
    visible_k: int,
) -> RerankResult:
    input_hash = canonical_hash(
        {
            "query_hash": search.query_hash,
            "candidates": [
                (item.memory_id, item.content_hash, item.native_rank)
                for item in search.candidates
            ],
            "visible_k": visible_k,
        }
    )
    relative = f"retrieval/{sceu.sceu_id}/common_rerank.json"
    stored = storage.load_cell(task, relative, input_hash=input_hash)
    if stored is not None:
        result = _rerank_result_from_dict(_mapping(stored, relative))
    else:
        result = reranker.rerank(
            search.query,
            tuple(
                RerankCandidate(
                    memory_id=item.memory_id,
                    text=item.content,
                    native_rank=item.native_rank,
                )
                for item in search.candidates
            ),
            top_k=visible_k,
        )
        storage.save_cell(
            task,
            relative,
            input_hash=input_hash,
            payload=asdict(result),
        )
    candidate_ids = {item.memory_id for item in search.candidates}
    if (
        len(result.ordered_memory_ids) != len(set(result.ordered_memory_ids))
        or not set(result.ordered_memory_ids) <= candidate_ids
        or len(result.ordered_memory_ids) > visible_k
    ):
        raise QualificationRunError(
            "trace_incomplete",
            "reranker output is not a valid subset of the frozen candidate set",
        )
    return result


def _run_sceu_branch(
    task: QualificationTask,
    scored: ScoredCondition,
    spec: SoftwareMem0VerticalSpec,
    components: TaskComponents,
    storage: QualificationStorage,
    *,
    sceu: SCEU,
    public: PublicContinuation,
    surface: SessionSurface,
    search: CandidateSearch | None,
    retrieval_trace: RetrievalTrace | None,
    alignments: Mapping[str, MemoryAttribution],
    visible_k: int,
    max_output_tokens: int,
) -> SCEURunResult:
    candidates = {item.memory_id: item for item in search.candidates} if search else {}
    if scored.readout == "native" and retrieval_trace is not None:
        retrieved_ids = retrieval_trace.native_retrieved_memory_ids
    elif scored.readout == "common_rerank" and retrieval_trace is not None:
        retrieved_ids = retrieval_trace.common_reranked_memory_ids
        if not retrieved_ids and search and search.candidates:
            raise QualificationRunError(
                "reranker_failure",
                "common-rerank branch has no completed reranker output",
            )
    else:
        retrieved_ids = ()
    visible_candidates = tuple(
        candidates[memory_id]
        for memory_id in retrieved_ids
        if memory_id in candidates and candidates[memory_id].content
    )[:visible_k]
    visible_ids = tuple(item.memory_id for item in visible_candidates)
    oracle_context = (
        _oracle_context(spec, sceu)
        if task.condition == "oracle_current_state"
        else ""
    )
    workspace_hash = canonical_hash(_workspace_payload(surface))
    visible_state_ids = _visible_state_ids(visible_ids, alignments)
    # Every memory-backed condition gets the same repeated baseline and
    # matched interventions.  The old prefix check only recognized legacy
    # ``mem0_*`` names, silently disabling causal/count probes for the v2
    # canonical ``flat_retrieval``, ``mem0``, ``amem`` and ``memos`` cells.
    baseline_count = 2 if components.memory is not None else 1
    baseline = tuple(
        _policy_evaluation(
            task,
            scored,
            spec,
            components,
            storage,
            sceu=sceu,
            public=public,
            surface=surface,
            call_kind="baseline",
            call_key=f"baseline-{index}",
            visible_candidates=visible_candidates,
            additional_context=oracle_context,
            workspace_hash=workspace_hash,
            visible_state_ids=visible_state_ids,
            max_output_tokens=max_output_tokens,
        )
        for index in range(baseline_count)
    )
    interventions: list[InterventionRun] = []
    if len(baseline) == 2:
        baseline_outcomes = (baseline[0].outcome, baseline[1].outcome)
        current_ids = set(replay_plan(spec.plan, sceu.checkpoint_session).current)
        for index, target in enumerate(visible_candidates):
            remaining = tuple(
                item
                for item in visible_candidates
                if item.memory_id != target.memory_id
            )
            intervention_evaluations = tuple(
                _policy_evaluation(
                    task,
                    scored,
                    spec,
                    components,
                    storage,
                    sceu=sceu,
                    public=public,
                    surface=surface,
                    call_kind="leave_one_out",
                    call_key=f"loo-{_slug(target.memory_id)}-{repeat}",
                    visible_candidates=remaining,
                    additional_context=oracle_context,
                    workspace_hash=workspace_hash,
                    visible_state_ids=_visible_state_ids(
                        tuple(item.memory_id for item in remaining),
                        alignments,
                    ),
                    max_output_tokens=max_output_tokens,
                )
                for repeat in range(2)
            )
            role = _memory_role(
                alignments.get(target.memory_id),
                current_ids=current_ids,
            )
            classification = classify_causal_use(
                memory_id=target.memory_id,
                intervention_kind="leave_one_out",
                memory_role=role,
                baseline=baseline_outcomes,
                intervention=(
                    intervention_evaluations[0].outcome,
                    intervention_evaluations[1].outcome,
                ),
            )
            interventions.append(
                InterventionRun(
                    intervention_kind="leave_one_out",
                    target_memory_id=target.memory_id,
                    replacement_memory_id=None,
                    evaluations=(
                        intervention_evaluations[0],
                        intervention_evaluations[1],
                    ),
                    classification=classification,
                    baseline_memory_count=len(visible_candidates),
                    intervention_memory_count=len(remaining),
                    count_contrast="delete_one",
                )
            )
            if role != "contradicts_current_state":
                continue
            replacement = _replacement_candidate(
                search.candidates if search else (),
                excluded_ids={item.memory_id for item in remaining} | {target.memory_id},
                alignments=alignments,
                current_ids=current_ids,
            )
            if replacement is None:
                continue
            replaced = list(visible_candidates)
            replaced[index] = replacement
            replacement_tuple = tuple(replaced)
            replacement_evaluations = tuple(
                _policy_evaluation(
                    task,
                    scored,
                    spec,
                    components,
                    storage,
                    sceu=sceu,
                    public=public,
                    surface=surface,
                    call_kind="stale_replacement",
                    call_key=(
                        f"replace-{_slug(target.memory_id)}-"
                        f"{_slug(replacement.memory_id)}-{repeat}"
                    ),
                    visible_candidates=replacement_tuple,
                    additional_context=oracle_context,
                    workspace_hash=workspace_hash,
                    visible_state_ids=_visible_state_ids(
                        tuple(item.memory_id for item in replacement_tuple),
                        alignments,
                    ),
                    max_output_tokens=max_output_tokens,
                )
                for repeat in range(2)
            )
            replacement_classification = classify_causal_use(
                memory_id=target.memory_id,
                intervention_kind="stale_replacement",
                memory_role=role,
                baseline=baseline_outcomes,
                intervention=(
                    replacement_evaluations[0].outcome,
                    replacement_evaluations[1].outcome,
                ),
            )
            interventions.append(
                InterventionRun(
                    intervention_kind="stale_replacement",
                    target_memory_id=target.memory_id,
                    replacement_memory_id=replacement.memory_id,
                    evaluations=(
                        replacement_evaluations[0],
                        replacement_evaluations[1],
                    ),
                    classification=replacement_classification,
                    baseline_memory_count=len(visible_candidates),
                    intervention_memory_count=len(replacement_tuple),
                    count_contrast="replace_one",
                )
            )
        if visible_candidates:
            add_candidate = _count_control_candidate(spec, sceu)
            added_candidates = (*visible_candidates, add_candidate)
            add_evaluations = tuple(
                _policy_evaluation(
                    task,
                    scored,
                    spec,
                    components,
                    storage,
                    sceu=sceu,
                    public=public,
                    surface=surface,
                    call_kind="count_add",
                    call_key=(
                        f"count-add-{_slug(add_candidate.memory_id)}-{repeat}"
                    ),
                    visible_candidates=added_candidates,
                    additional_context=oracle_context,
                    workspace_hash=workspace_hash,
                    visible_state_ids=_visible_state_ids(
                        tuple(item.memory_id for item in added_candidates),
                        alignments,
                    ),
                    max_output_tokens=max_output_tokens,
                )
                for repeat in range(2)
            )
            add_classification = classify_causal_use(
                memory_id=add_candidate.memory_id,
                intervention_kind="count_add",
                memory_role="supports_current_state",
                baseline=baseline_outcomes,
                intervention=(
                    add_evaluations[0].outcome,
                    add_evaluations[1].outcome,
                ),
            )
            interventions.append(
                InterventionRun(
                    intervention_kind="count_add",
                    target_memory_id=add_candidate.memory_id,
                    replacement_memory_id=None,
                    evaluations=(add_evaluations[0], add_evaluations[1]),
                    classification=add_classification,
                    baseline_memory_count=len(visible_candidates),
                    intervention_memory_count=len(added_candidates),
                    count_contrast="add_one",
                )
            )
    primary = baseline[0]
    baseline_stable = len(baseline) == 1 or (
        baseline[0].outcome == baseline[1].outcome
    )
    return SCEURunResult(
        result_id=scored.result_id,
        sceu_id=sceu.sceu_id,
        opportunity_id=sceu.opportunity_id,
        checkpoint_session=sceu.checkpoint_session,
        matched_group=sceu.matched_group,
        control_kind=next(
            item.control_kind
            for item in spec.plan.opportunities
            if item.opportunity_id == sceu.opportunity_id
        ),
        workspace_hash=workspace_hash,
        candidate_memory_ids=(
            retrieval_trace.candidate_memory_ids if retrieval_trace else ()
        ),
        retrieved_memory_ids=retrieved_ids,
        model_visible_memory_ids=visible_ids,
        selected_option_id=primary.response.selected_option_id,
        selected_action_id=primary.selected_action_id,
        behavior=primary.outcome,
        normalized_drift_flags=primary.normalized_drift_flags,
        baseline_stable=baseline_stable,
        baseline_evaluations=baseline,
        interventions=tuple(interventions),
        retrieval_trace_id=(
            retrieval_trace.trace_id if retrieval_trace is not None else None
        ),
    )


def _policy_evaluation(
    task: QualificationTask,
    scored: ScoredCondition,
    spec: SoftwareMem0VerticalSpec,
    components: TaskComponents,
    storage: QualificationStorage,
    *,
    sceu: SCEU,
    public: PublicContinuation,
    surface: SessionSurface,
    call_kind: str,
    call_key: str,
    visible_candidates: tuple[SearchCandidate, ...],
    additional_context: str,
    workspace_hash: str,
    visible_state_ids: tuple[str, ...],
    max_output_tokens: int,
) -> PolicyEvaluation:
    blocks = _memory_blocks(visible_candidates)
    visible_ids = tuple(item.memory_id for item in visible_candidates)
    user_content = _continuation_user_content(
        surface,
        public,
        blocks=blocks,
        additional_context=additional_context,
    )
    request = PolicyRequest(
        request_id=(
            f"{task.task_id}:{scored.result_id}:{sceu.sceu_id}:{call_key}"
        ),
        system_prompt=_SYSTEM_PROMPT,
        messages=(PolicyMessage(role="user", content=user_content),),
        options=public.options,
        max_output_tokens=max_output_tokens,
    )
    request_hash = policy_request_hash(request)
    call_id = request.request_id
    base_path = (
        f"calls/{_slug(scored.result_id)}/{_slug(sceu.sceu_id)}"
    )
    response_path = f"{base_path}/responses/{_slug(call_key)}.json"
    stored_response = storage.load_cell(
        task,
        response_path,
        input_hash=request_hash,
    )
    if stored_response is None:
        response = components.policy.submit_action(request)
        response_payload = {
            "request": _policy_request_to_dict(request),
            "response": asdict(response),
            "model_visible_memory_ids": list(visible_ids),
            "model_visible_blocks": list(blocks),
            "workspace_hash": workspace_hash,
            "transcript_hash": hash_text(user_content),
        }
        storage.save_cell(
            task,
            response_path,
            input_hash=request_hash,
            payload=response_payload,
        )
    else:
        response_record = _mapping(stored_response, response_path)
        request = _policy_request_from_dict(
            _mapping(response_record.get("request"), f"{response_path}.request")
        )
        response = _policy_response_from_dict(
            _mapping(response_record.get("response"), f"{response_path}.response")
        )
        visible_ids = _string_tuple(
            response_record.get("model_visible_memory_ids")
        )
        blocks = _string_tuple(response_record.get("model_visible_blocks"))
        user_content = request.messages[0].content
    if response.request_id != request.request_id:
        raise QualificationRunError(
            "trace_incomplete",
            "policy response request ID does not match the persisted request",
        )

    evaluator = spec.evaluator_continuation_map[sceu.opportunity_id]
    try:
        selected_action_id = evaluator.action_for_option(
            response.selected_option_id
        )
    except KeyError as exc:
        raise QualificationRunError(
            "structured_output_failure",
            str(exc),
        ) from exc
    evaluation_path = f"{base_path}/evaluations/{_slug(call_key)}.json"
    evaluation_input_hash = canonical_hash(
        {
            "response_hash": response.response_hash,
            "selected_option_id": response.selected_option_id,
            "mapping": evaluator.option_to_action,
            "checkpoint_session": sceu.checkpoint_session,
            "visible_state_ids": visible_state_ids,
        }
    )
    stored_evaluation = storage.load_cell(
        task,
        evaluation_path,
        input_hash=evaluation_input_hash,
    )
    if stored_evaluation is None:
        try:
            try:
                checker_result = components.checker.check_action(
                    selected_action_id,
                    checkpoint_session=sceu.checkpoint_session,
                    visible_state_ids=visible_state_ids,
                    opportunity_id=sceu.opportunity_id,
                )
            except TypeError as exc:
                # Keep old checker/test doubles executable while enabling
                # opportunity-specific validity for the v0.3 local exception.
                if "opportunity_id" not in str(exc):
                    raise
                checker_result = components.checker.check_action(
                    selected_action_id,
                    checkpoint_session=sceu.checkpoint_session,
                    visible_state_ids=visible_state_ids,
                )
        except Exception as exc:
            raise QualificationRunError("checker_failure", str(exc)) from exc
        action = spec.action_map[selected_action_id]
        normalized = _normalized_drift(
            spec,
            action,
            checker_result,
            checkpoint_session=sceu.checkpoint_session,
            opportunity_id=sceu.opportunity_id,
        )
        outcome = ContinuationOutcome(
            action_id=selected_action_id,
            behavior_score=checker_result.score,
            is_correct=checker_result.is_correct,
            violated_state_ids=checker_result.violated_state_ids,
            drift_flags=normalized,
        )
        storage.save_cell(
            task,
            evaluation_path,
            input_hash=evaluation_input_hash,
            payload={
                "selected_action_id": selected_action_id,
                "checker_result": asdict(checker_result),
                "outcome": asdict(outcome),
                "normalized_drift_flags": list(normalized),
            },
        )
    else:
        evaluation_record = _mapping(
            stored_evaluation,
            evaluation_path,
        )
        selected_action_id = str(
            evaluation_record.get("selected_action_id", "")
        )
        checker_result = _behavior_result_from_dict(
            _mapping(
                evaluation_record.get("checker_result"),
                f"{evaluation_path}.checker_result",
            )
        )
        outcome = _outcome_from_dict(
            _mapping(
                evaluation_record.get("outcome"),
                f"{evaluation_path}.outcome",
            )
        )
        normalized = _string_tuple(
            evaluation_record.get("normalized_drift_flags")
        )
    return PolicyEvaluation(
        call_id=call_id,
        call_kind=call_kind,
        request=request,
        response=response,
        selected_action_id=selected_action_id,
        checker_result=checker_result,
        outcome=outcome,
        normalized_drift_flags=normalized,
        workspace_hash=workspace_hash,
        transcript_hash=hash_text(user_content),
        policy_request_hash=request_hash,
        model_visible_memory_ids=visible_ids,
        model_visible_blocks=blocks,
        model_visible_context_hash=hash_text("\n\n".join(blocks)),
    )


def _normalized_drift(
    spec: SoftwareMem0VerticalSpec,
    action: ActionSpec,
    behavior: BehaviorResult,
    *,
    checkpoint_session: int,
    opportunity_id: str | None = None,
) -> tuple[str, ...]:
    replay = replay_plan(spec.plan, checkpoint_session)
    current = replay.current
    state_map = {state.state_id: state for state in spec.plan.state_units}
    stale_ids = tuple(
        state.state_id
        for state in spec.plan.state_units
        if state.valid_from <= checkpoint_session and state.state_id not in current
    )
    local_ids = tuple(
        state_id
        for state_id in action.satisfies_state_ids
        if state_id in state_map
        and (
            "local" in state_map[state_id].scope
            or state_map[state_id].authority == "local-operator"
        )
    )
    future_ids = tuple(
        state.state_id
        for state in spec.plan.state_units
        if state.valid_from > checkpoint_session
        and state.state_id in action.satisfies_state_ids
    )
    result = classify_long_horizon_drift(
        DriftEvidence(
            outcome=ContinuationOutcome(
                action_id=action.action_id,
                behavior_score=behavior.score,
                is_correct=behavior.is_correct,
                violated_state_ids=behavior.violated_state_ids,
                drift_flags=behavior.drift_flags,
            ),
            action_expressed_state_ids=action.satisfies_state_ids,
            active_constraint_ids=tuple(
                state_id
                for state_id, state in current.items()
                if state.kind == "constraint"
            ),
            current_plan_state_ids=tuple(
                state_id
                for state_id, state in current.items()
                if state.kind == "plan_node"
            ),
            stale_state_ids=stale_ids,
            selected_local_state_ids=local_ids,
            global_state_ids=tuple(
                state_id
                for state_id, state in current.items()
                if state.kind in {"global_goal", "constraint"}
            ),
            future_state_ids=future_ids,
        )
    )
    return tuple(str(flag) for flag in result.flags)


def _condition_result(
    scored: ScoredCondition,
    *,
    rows: list[SCEURunResult],
    error: tuple[str, str] | None,
    expected_sceu: int,
) -> ConditionRunResult:
    if error is None and len(rows) != expected_sceu:
        error = (
            "trace_incomplete",
            f"expected {expected_sceu} SCEU results, got {len(rows)}",
        )
    return ConditionRunResult(
        result_id=scored.result_id,
        condition=scored.condition,
        readout=scored.readout,
        status="complete" if error is None else "failed",
        sceu_results=tuple(rows),
        error_class=error[0] if error else None,
        error_message=error[1] if error else None,
    )


def _terminal_task_failure(
    task: QualificationTask,
    error_class: str,
    message: str,
) -> QualificationTaskResult:
    return QualificationTaskResult(
        task_id=task.task_id,
        episode_id=task.episode_id,
        policy_profile_id=task.policy_profile_id,
        condition=task.condition,
        status="failed",
        condition_results=tuple(
            ConditionRunResult(
                result_id=item.result_id,
                condition=item.condition,
                readout=item.readout,
                status="failed",
                sceu_results=(),
                error_class=error_class,
                error_message=message,
            )
            for item in task.scored_conditions
        ),
        writes=(),
        alignments=(),
        retrieval_traces=(),
        error_class=error_class,
        error_message=message,
    )


def _memory_blocks(
    candidates: tuple[SearchCandidate, ...],
) -> tuple[str, ...]:
    return tuple(
        f"Retrieved memory {index}:\n{candidate.content}"
        for index, candidate in enumerate(candidates, start=1)
    )


def _count_control_candidate(
    spec: SoftwareMem0VerticalSpec,
    sceu: SCEU,
) -> SearchCandidate:
    """Build one deterministic evaluator-side add-one memory object."""
    current = replay_plan(spec.plan, sceu.checkpoint_session).current
    preferred = [
        state_id
        for state_id in sceu.required_state_ids
        if state_id in current
    ] or list(sceu.focal_state_ids)
    lines: list[str] = []
    for state_id in preferred:
        state = current.get(state_id)
        if state is None:
            continue
        if isinstance(state.value, dict):
            text = state.value.get("text")
            lines.append(str(text) if isinstance(text, str) else str(state.value))
        else:
            lines.append(str(state.value))
    content = "Controlled count intervention: " + " ".join(lines)
    memory_id = f"__count_add__{sceu.sceu_id}"
    return SearchCandidate(
        memory_id=memory_id,
        content=content,
        content_hash=hash_text(content),
        native_rank=10_000,
        score=0.0,
        score_details=(("count_control", 0.0),),
        metadata=(
            ("lhmsb.provenance", {"mode": "inferred", "sceu_id": sceu.sceu_id}),
            ("lhmsb.candidate_origin", "count_control"),
        ),
        created_at="evaluator",
        updated_at="evaluator",
    )


def _visible_state_ids(
    memory_ids: tuple[str, ...],
    alignments: Mapping[str, MemoryAttribution],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                state_id
                for memory_id in memory_ids
                for state_id in alignments.get(
                    memory_id,
                    MemoryAttribution(
                        memory_id=memory_id,
                        state_ids=(),
                        method="ambiguous",
                        contributes_positive_coverage=False,
                        reason="missing evaluator attribution",
                    ),
                ).state_ids
            }
        )
    )


def _memory_role(
    attribution: MemoryAttribution | None,
    *,
    current_ids: set[str],
) -> MemoryRole:
    if (
        attribution is None
        or not attribution.contributes_positive_coverage
        or not attribution.state_ids
    ):
        return "unknown"
    current = [state_id in current_ids for state_id in attribution.state_ids]
    if all(current):
        return "supports_current_state"
    if not any(current):
        return "contradicts_current_state"
    return "unknown"


def _replacement_candidate(
    candidates: tuple[SearchCandidate, ...],
    *,
    excluded_ids: set[str],
    alignments: Mapping[str, MemoryAttribution],
    current_ids: set[str],
) -> SearchCandidate | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.memory_id not in excluded_ids
            and _memory_role(
                alignments.get(candidate.memory_id),
                current_ids=current_ids,
            )
            == "supports_current_state"
        ),
        None,
    )


def _oracle_context(
    spec: SoftwareMem0VerticalSpec,
    sceu: SCEU,
) -> str:
    current = replay_plan(spec.plan, sceu.checkpoint_session).current
    required = tuple(
        state_id for state_id in sceu.required_state_ids if state_id in current
    )
    if not required:
        required = tuple(
            state_id for state_id in sceu.focal_state_ids if state_id in current
        )
    lines = [_state_text(current[state_id]) for state_id in required]
    return (
        "Evaluator-provided current project state:\n"
        + "\n".join(f"- {line}" for line in lines)
        if lines
        else ""
    )


def _state_text(state: StateUnit) -> str:
    if isinstance(state.value, dict):
        text = state.value.get("text")
        if isinstance(text, str):
            return text
        return "; ".join(
            f"{key}: {value}" for key, value in sorted(state.value.items())
        )
    return str(state.value)


def _continuation_user_content(
    surface: SessionSurface,
    public: PublicContinuation,
    *,
    blocks: tuple[str, ...],
    additional_context: str,
) -> str:
    sections = [
        "Current session observations:\n"
        + json.dumps(
            list(surface.observations),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "Current session tool results:\n"
        + json.dumps(
            list(surface.tool_results),
            sort_keys=True,
            ensure_ascii=False,
        ),
        "Current workspace:\n"
        + json.dumps(
            _workspace_payload(surface),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        f"Continuation request:\n{public.request}",
    ]
    if additional_context:
        sections.append(additional_context)
    if blocks:
        sections.append("Memory readout:\n" + "\n\n".join(blocks))
    return "\n\n".join(sections)


def _workspace_payload(surface: SessionSurface) -> dict[str, object]:
    return {
        "checkpoint_session": surface.workspace.checkpoint_session,
        "artifacts": [
            {
                "path": item.path,
                "content": item.content,
                "version": item.version,
                "created_session": item.created_session,
                "updated_session": item.updated_session,
            }
            for item in surface.workspace.artifacts
        ],
    }


def _policy_request_to_dict(request: PolicyRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "system_prompt": request.system_prompt,
        "messages": [asdict(item) for item in request.messages],
        "options": [item.to_dict() for item in request.options],
        "max_output_tokens": request.max_output_tokens,
    }


def _policy_request_from_dict(data: Mapping[str, object]) -> PolicyRequest:
    return PolicyRequest(
        request_id=str(data["request_id"]),
        system_prompt=str(data["system_prompt"]),
        messages=tuple(
            PolicyMessage(
                role=cast(
                    Literal["system", "user", "assistant"],
                    str(item["role"]),
                ),
                content=str(item["content"]),
            )
            for item in _mapping_sequence(data.get("messages"))
        ),
        options=tuple(
            PublicActionOption.from_dict(item)
            for item in _mapping_sequence(data.get("options"))
        ),
        max_output_tokens=_integer(data.get("max_output_tokens")),
    )


def _policy_response_from_dict(data: Mapping[str, object]) -> PolicyResponse:
    usage = _mapping(data.get("usage"), "policy usage")
    return PolicyResponse(
        request_id=str(data["request_id"]),
        provider=str(data["provider"]),
        model_id=str(data["model_id"]),
        endpoint_identity=str(data["endpoint_identity"]),
        selected_option_id=str(data["selected_option_id"]),
        optional_patch=_optional_string(data.get("optional_patch")),
        concise_rationale=str(data["concise_rationale"]),
        provider_request_id=_optional_string(data.get("provider_request_id")),
        usage=PolicyUsage(
            input_tokens=_optional_integer(usage.get("input_tokens")),
            output_tokens=_optional_integer(usage.get("output_tokens")),
            cached_tokens=_optional_integer(usage.get("cached_tokens")),
            reasoning_tokens=_optional_integer(usage.get("reasoning_tokens")),
            observed=bool(usage.get("observed", True)),
        ),
        request_hash=str(data["request_hash"]),
        response_hash=str(data["response_hash"]),
        started_at_utc=str(data["started_at_utc"]),
        ended_at_utc=str(data["ended_at_utc"]),
        latency_seconds=_number(data.get("latency_seconds")),
        retry_count=_integer(data.get("retry_count")),
        format_repair_used=bool(data.get("format_repair_used", False)),
    )


def _behavior_result_from_dict(data: Mapping[str, object]) -> BehaviorResult:
    return BehaviorResult(
        score=_number(data.get("score")),
        is_correct=bool(data.get("is_correct", False)),
        violated_state_ids=_string_tuple(data.get("violated_state_ids")),
        passed_tests=_string_tuple(data.get("passed_tests")),
        failed_tests=_string_tuple(data.get("failed_tests")),
        drift_flags=_string_tuple(data.get("drift_flags")),
        metadata=tuple(
            (str(pair[0]), str(pair[1]))
            for pair in _pair_sequence(data.get("metadata"))
        ),
    )


def _outcome_from_dict(data: Mapping[str, object]) -> ContinuationOutcome:
    return ContinuationOutcome(
        action_id=str(data["action_id"]),
        behavior_score=_number(data.get("behavior_score")),
        is_correct=bool(data.get("is_correct", False)),
        violated_state_ids=_string_tuple(data.get("violated_state_ids")),
        drift_flags=_string_tuple(data.get("drift_flags")),
    )


def _condition_run_result_from_dict(
    data: Mapping[str, object],
) -> ConditionRunResult:
    return ConditionRunResult(
        result_id=str(data["result_id"]),
        condition=str(data["condition"]),
        readout=cast(
            Literal["none", "native", "common_rerank"],
            str(data["readout"]),
        ),
        status=cast(ConditionStatus, str(data["status"])),
        sceu_results=tuple(
            _sceu_run_result_from_dict(item)
            for item in _mapping_sequence(data.get("sceu_results"))
        ),
        error_class=_optional_string(data.get("error_class")),
        error_message=_optional_string(data.get("error_message")),
    )


def _sceu_run_result_from_dict(
    data: Mapping[str, object],
) -> SCEURunResult:
    return SCEURunResult(
        result_id=str(data["result_id"]),
        sceu_id=str(data["sceu_id"]),
        opportunity_id=str(data["opportunity_id"]),
        checkpoint_session=_integer(data.get("checkpoint_session")),
        matched_group=str(data.get("matched_group", "")),
        control_kind=str(data.get("control_kind", "")),
        workspace_hash=str(data.get("workspace_hash", "")),
        candidate_memory_ids=_string_tuple(data.get("candidate_memory_ids")),
        retrieved_memory_ids=_string_tuple(data.get("retrieved_memory_ids")),
        model_visible_memory_ids=_string_tuple(
            data.get("model_visible_memory_ids")
        ),
        selected_option_id=str(data.get("selected_option_id", "")),
        selected_action_id=str(data.get("selected_action_id", "")),
        behavior=_outcome_from_dict(
            _mapping(data.get("behavior"), "SCEU behavior")
        ),
        normalized_drift_flags=_string_tuple(
            data.get("normalized_drift_flags")
        ),
        baseline_stable=bool(data.get("baseline_stable", False)),
        baseline_evaluations=tuple(
            _policy_evaluation_from_dict(item)
            for item in _mapping_sequence(
                data.get("baseline_evaluations")
            )
        ),
        interventions=tuple(
            _intervention_run_from_dict(item)
            for item in _mapping_sequence(data.get("interventions"))
        ),
        retrieval_trace_id=_optional_string(
            data.get("retrieval_trace_id")
        ),
    )


def _policy_evaluation_from_dict(
    data: Mapping[str, object],
) -> PolicyEvaluation:
    return PolicyEvaluation(
        call_id=str(data["call_id"]),
        call_kind=str(data["call_kind"]),
        request=_policy_request_from_dict(
            _mapping(data.get("request"), "policy request")
        ),
        response=_policy_response_from_dict(
            _mapping(data.get("response"), "policy response")
        ),
        selected_action_id=str(data["selected_action_id"]),
        checker_result=_behavior_result_from_dict(
            _mapping(data.get("checker_result"), "checker result")
        ),
        outcome=_outcome_from_dict(
            _mapping(data.get("outcome"), "policy outcome")
        ),
        normalized_drift_flags=_string_tuple(
            data.get("normalized_drift_flags")
        ),
        workspace_hash=str(data.get("workspace_hash", "")),
        transcript_hash=str(data.get("transcript_hash", "")),
        policy_request_hash=str(data.get("policy_request_hash", "")),
        model_visible_memory_ids=_string_tuple(
            data.get("model_visible_memory_ids")
        ),
        model_visible_blocks=_string_tuple(
            data.get("model_visible_blocks")
        ),
        model_visible_context_hash=str(
            data.get("model_visible_context_hash", "")
        ),
    )


def _intervention_run_from_dict(
    data: Mapping[str, object],
) -> InterventionRun:
    evaluations = tuple(
        _policy_evaluation_from_dict(item)
        for item in _mapping_sequence(data.get("evaluations"))
    )
    if len(evaluations) != 2:
        raise QualificationStorageError(
            "trace_incomplete",
            "intervention result must contain exactly two evaluations",
        )
    classification_data = _mapping(
        data.get("classification"),
        "intervention classification",
    )
    return InterventionRun(
        intervention_kind=str(data["intervention_kind"]),
        target_memory_id=str(data["target_memory_id"]),
        replacement_memory_id=_optional_string(
            data.get("replacement_memory_id")
        ),
        evaluations=(evaluations[0], evaluations[1]),
        classification=CausalUseResult(
            memory_id=str(classification_data["memory_id"]),
            intervention_kind=cast(
                InterventionKind,
                str(classification_data["intervention_kind"]),
            ),
            memory_role=cast(
                MemoryRole,
                str(classification_data["memory_role"]),
            ),
            label=cast(
                CausalUseLabel,
                str(classification_data["label"]),
            ),
            effect_direction=cast(
                EffectDirection,
                str(classification_data["effect_direction"]),
            ),
            behaviorally_used=bool(
                classification_data.get("behaviorally_used", False)
            ),
            baseline_stable=bool(
                classification_data.get("baseline_stable", False)
            ),
            intervention_stable=bool(
                classification_data.get("intervention_stable", False)
            ),
            action_changed=bool(
                classification_data.get("action_changed", False)
            ),
            checker_changed=bool(
                classification_data.get("checker_changed", False)
            ),
        ),
        baseline_memory_count=_integer(data.get("baseline_memory_count", 0)),
        intervention_memory_count=_integer(data.get("intervention_memory_count", 0)),
        count_contrast=(
            None
            if data.get("count_contrast") is None
            else str(data.get("count_contrast"))
        ),
    )


def _alignment_from_dict(
    data: Mapping[str, object],
) -> MemoryAlignmentSnapshot:
    return MemoryAlignmentSnapshot(
        checkpoint_session=_integer(data.get("checkpoint_session")),
        inventory_store_hash=str(data.get("inventory_store_hash", "")),
        attributions=tuple(
            MemoryAttribution(
                memory_id=str(item["memory_id"]),
                state_ids=_string_tuple(item.get("state_ids")),
                method=cast(
                    AttributionMethod,
                    str(item["method"]),
                ),
                contributes_positive_coverage=bool(
                    item.get("contributes_positive_coverage", False)
                ),
                reason=str(item.get("reason", "")),
                provenance_mode=cast(
                    Literal["native/exact", "inferred", "unavailable"],
                    str(item.get("provenance_mode", "unavailable")),
                ),
                source_event_ids=_string_tuple(item.get("source_event_ids")),
                source_session=(
                    None
                    if item.get("source_session") is None
                    else _integer(item.get("source_session"))
                ),
            )
            for item in _mapping_sequence(data.get("attributions"))
        ),
    )


def _retrieval_trace_from_dict(
    data: Mapping[str, object],
) -> RetrievalTrace:
    candidate_search = _candidate_search_from_dict(
        {
            **data,
            "latency_seconds": data.get("search_latency_seconds"),
        }
    )
    rerank_data = data.get("rerank_result")
    return RetrievalTrace(
        trace_id=str(data["trace_id"]),
        sceu_id=str(data["sceu_id"]),
        opportunity_id=str(data["opportunity_id"]),
        checkpoint_session=_integer(data.get("checkpoint_session")),
        query=str(data.get("query", "")),
        query_hash=str(data.get("query_hash", "")),
        candidates=candidate_search.candidates,
        candidate_memory_ids=_string_tuple(
            data.get("candidate_memory_ids")
        ),
        native_retrieved_memory_ids=_string_tuple(
            data.get("native_retrieved_memory_ids")
        ),
        common_reranked_memory_ids=_string_tuple(
            data.get("common_reranked_memory_ids")
        ),
        candidate_shortfall=candidate_search.candidate_shortfall,
        search_latency_seconds=_number(
            data.get("search_latency_seconds")
        ),
        rerank_result=(
            _rerank_result_from_dict(
                _mapping(rerank_data, "rerank result")
            )
            if rerank_data is not None
            else None
        ),
        internal_usage=tuple(
            _provider_usage_event_from_dict(item)
            for item in _mapping_sequence(data.get("internal_usage", ()))
        ),
    )


def _write_result_from_dict(data: Mapping[str, object]) -> WriteSessionResult:
    try:
        return WriteSessionResult.from_dict(data)
    except MemoryTraceValidationError as exc:
        raise QualificationStorageError(
            "trace_incomplete",
            f"invalid persisted write result: {exc}",
        ) from exc


def _candidate_search_from_dict(
    data: Mapping[str, object],
) -> CandidateSearch:
    try:
        return CandidateSearch.from_dict(data)
    except MemoryTraceValidationError as exc:
        raise QualificationStorageError(
            "trace_incomplete",
            f"invalid persisted candidate search: {exc}",
        ) from exc


def _provider_usage_event_from_dict(
    data: Mapping[str, object],
) -> ProviderUsageEvent:
    try:
        return ProviderUsageEvent.from_dict(data)
    except MemoryTraceValidationError as exc:
        raise QualificationStorageError(
            "trace_incomplete",
            f"invalid persisted provider usage event: {exc}",
        ) from exc


def _rerank_result_from_dict(data: Mapping[str, object]) -> RerankResult:
    return RerankResult(
        ordered_memory_ids=_string_tuple(data.get("ordered_memory_ids")),
        scores=tuple(
            _number(item) for item in _sequence(data.get("scores"))
        ),
        model=str(data.get("model", "")),
        revision=str(data.get("revision", "")),
        input_count=_integer(data.get("input_count")),
        request_hash=str(data.get("request_hash", "")),
        response_hash=str(data.get("response_hash", "")),
        latency_seconds=_number(data.get("latency_seconds")),
    )


def _error_class(
    exc: Exception,
    *,
    default: str = "checker_failure",
) -> str:
    if isinstance(
        exc,
        (
            Mem0QualificationError,
            PolicyCallError,
            QualificationRunError,
            QualificationStorageError,
            TeiServiceError,
        ),
    ):
        return exc.error_class
    return default


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return result or "cell"


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise QualificationStorageError(
            "trace_incomplete",
            f"{label} must be an object",
        )
    return {str(key): child for key, child in value.items()}


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QualificationStorageError(
            "trace_incomplete",
            "persisted value must be an array",
        )
    return tuple(value)


def _mapping_sequence(value: object) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "array item") for item in _sequence(value)
    )


def _pair_sequence(value: object) -> tuple[tuple[object, object], ...]:
    output: list[tuple[object, object]] = []
    for item in _sequence(value):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise QualificationStorageError(
                "trace_incomplete",
                "persisted metadata must contain pairs",
            )
        output.append((item[0], item[1]))
    return tuple(output)


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise QualificationStorageError(
            "trace_incomplete",
            f"expected integer, got {value!r}",
        )
    return int(value)


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise QualificationStorageError(
            "trace_incomplete",
            f"expected number, got {value!r}",
        )
    return float(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ConditionRunResult",
    "InterventionRun",
    "MemoryAlignmentSnapshot",
    "PolicyEvaluation",
    "QualificationMatrixResult",
    "QualificationRunError",
    "QualificationTaskResult",
    "RetrievalTrace",
    "SCEURunResult",
    "TaskComponentFactory",
    "TaskComponents",
    "TaskIsolation",
    "hash_text",
    "policy_request_hash",
    "qualification_task_result_from_dict",
    "run_qualification_matrix",
    "run_qualification_task",
]
