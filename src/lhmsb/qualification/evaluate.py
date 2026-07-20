"""Read-only schema-v2 continuation policy evaluation.

Prefix preparation is deliberately separate from this module.  ``evaluate_task``
consumes an immutable :class:`MemoryPrefixArtifact` and never accepts a memory
runtime, so policy evaluation cannot mutate a prepared backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import BehaviorResult
from lhmsb.longhorizon.attribution import (
    MemoryAttribution,
    attribute_memory,
    build_software_fact_signatures,
    eligible_write_state_ids,
)
from lhmsb.longhorizon.drift import DriftEvidence, classify_long_horizon_drift
from lhmsb.longhorizon.interventions import (
    CausalUseResult,
    ContinuationOutcome,
    MemoryRole,
    classify_causal_use,
)
from lhmsb.longhorizon.public_surface import PublicContinuation
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import SCEU, ActionSpec, SessionSurface, StateUnit
from lhmsb.qualification.config import NO_PREFIX_ARTIFACT, canonical_hash
from lhmsb.qualification.context import build_public_history_units, render_full_context
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    MemoryObject,
    RetrievalCandidate,
)
from lhmsb.qualification.prefix import (
    CommonRerankTrace,
    MemoryPrefixArtifact,
    MemoryPrefixCheckpoint,
    PrefixArtifactError,
    prefix_artifact_hash,
)
from lhmsb.qualification.providers import (
    PolicyClient,
    PolicyMessage,
    PolicyRequest,
    PolicyResponse,
)
from lhmsb.qualification.schema import CausalSamplingProfile, EvaluationTask, ReadoutKind

_SYSTEM_PROMPT = (
    "Continue the software project using only the supplied current-session surface, "
    "workspace, and optional context. Select exactly one opaque implementation option."
)


class BehaviorChecker(Protocol):
    def check_action(
        self,
        action: str,
        *,
        checkpoint_session: int,
        visible_state_ids: tuple[str, ...] | None = None,
    ) -> BehaviorResult: ...


class EvaluationError(RuntimeError):
    """Terminal policy-evaluation validation or provider failure."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


EvaluationValidationError = EvaluationError


@dataclass(frozen=True)
class EvaluationRetrievalTrace:
    checkpoint_session: int
    opportunity_id: str
    candidate_memory_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    visible_memory_ids: tuple[str, ...]
    candidate_shortfall: bool
    query_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_session": self.checkpoint_session,
            "opportunity_id": self.opportunity_id,
            "candidate_memory_ids": list(self.candidate_memory_ids),
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "visible_memory_ids": list(self.visible_memory_ids),
            "candidate_shortfall": self.candidate_shortfall,
            "query_hash": self.query_hash,
        }


@dataclass(frozen=True)
class EvaluationCall:
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
    visible_object_count: int
    visible_object_chars: int

    @property
    def provider(self) -> str:
        return self.response.provider

    @property
    def model_id(self) -> str:
        return self.response.model_id

    @property
    def route_id(self) -> str:
        return self.response.endpoint_identity

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "call_kind": self.call_kind,
            "request": _request_to_dict(self.request),
            "response": asdict(self.response),
            "selected_action_id": self.selected_action_id,
            "checker_result": asdict(self.checker_result),
            "outcome": asdict(self.outcome),
            "normalized_drift_flags": list(self.normalized_drift_flags),
            "workspace_hash": self.workspace_hash,
            "transcript_hash": self.transcript_hash,
            "policy_request_hash": self.policy_request_hash,
            "model_visible_memory_ids": list(self.model_visible_memory_ids),
            "model_visible_blocks": list(self.model_visible_blocks),
            "model_visible_context_hash": self.model_visible_context_hash,
            "visible_object_count": self.visible_object_count,
            "visible_object_chars": self.visible_object_chars,
        }


@dataclass(frozen=True)
class EvaluationIntervention:
    intervention_kind: str
    target_memory_id: str
    replacement_memory_id: str | None
    evaluations: tuple[EvaluationCall, ...]
    classification: CausalUseResult

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention_kind": self.intervention_kind,
            "target_memory_id": self.target_memory_id,
            "replacement_memory_id": self.replacement_memory_id,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "classification": asdict(self.classification),
        }


@dataclass(frozen=True)
class EvaluationSCEUResult:
    result_id: str
    sceu_id: str
    opportunity_id: str
    checkpoint_session: int
    matched_group: str
    control_kind: str
    prefix_artifact_hash: str
    workspace_hash: str
    candidate_memory_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    model_visible_memory_ids: tuple[str, ...]
    model_visible_object_count: int
    model_visible_chars: int
    selected_option_id: str
    selected_action_id: str
    behavior: ContinuationOutcome
    normalized_drift_flags: tuple[str, ...]
    baseline_stable: bool
    baseline_evaluations: tuple[EvaluationCall, ...]
    interventions: tuple[EvaluationIntervention, ...]
    retrieval_trace_id: str | None
    transcript_hash: str
    model_visible_context_hash: str
    candidate_shortfall: bool = False

    @property
    def behavior_score(self) -> float:
        return self.behavior.behavior_score

    @property
    def is_correct(self) -> bool:
        return self.behavior.is_correct

    def to_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "sceu_id": self.sceu_id,
            "opportunity_id": self.opportunity_id,
            "checkpoint_session": self.checkpoint_session,
            "matched_group": self.matched_group,
            "control_kind": self.control_kind,
            "prefix_artifact_hash": self.prefix_artifact_hash,
            "workspace_hash": self.workspace_hash,
            "candidate_memory_ids": list(self.candidate_memory_ids),
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "model_visible_memory_ids": list(self.model_visible_memory_ids),
            "model_visible_object_count": self.model_visible_object_count,
            "model_visible_chars": self.model_visible_chars,
            "selected_option_id": self.selected_option_id,
            "selected_action_id": self.selected_action_id,
            "behavior": asdict(self.behavior),
            "normalized_drift_flags": list(self.normalized_drift_flags),
            "baseline_stable": self.baseline_stable,
            "baseline_evaluations": [item.to_dict() for item in self.baseline_evaluations],
            "interventions": [item.to_dict() for item in self.interventions],
            "retrieval_trace_id": self.retrieval_trace_id,
            "transcript_hash": self.transcript_hash,
            "model_visible_context_hash": self.model_visible_context_hash,
            "candidate_shortfall": self.candidate_shortfall,
        }


@dataclass(frozen=True)
class EvaluationConditionResult:
    result_id: str
    condition: str
    readout: ReadoutKind
    status: str
    sceu_results: tuple[EvaluationSCEUResult, ...]
    error_class: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "condition": self.condition,
            "readout": self.readout,
            "status": self.status,
            "sceu_results": [item.to_dict() for item in self.sceu_results],
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class EvaluationTaskResult:
    task_id: str
    episode_id: str
    policy_profile_id: str
    condition: str
    prefix_artifact_hash: str
    status: str
    condition_results: tuple[EvaluationConditionResult, ...]
    result_hash: str
    error_class: str | None = None
    error_message: str | None = None

    @property
    def sceu_results(self) -> tuple[EvaluationSCEUResult, ...]:
        return tuple(row for condition in self.condition_results for row in condition.sceu_results)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy_profile_id": self.policy_profile_id,
            "condition": self.condition,
            "prefix_artifact_hash": self.prefix_artifact_hash,
            "status": self.status,
            "condition_results": [item.to_dict() for item in self.condition_results],
            "result_hash": self.result_hash,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


# Names used by schema-v1 reports remain discoverable without importing the mutable
# runner.  They are aliases only; no schema-v1 bytes are changed.
PolicyEvaluation = EvaluationCall
InterventionRun = EvaluationIntervention
SCEURunResult = EvaluationSCEUResult
ConditionRunResult = EvaluationConditionResult
EvaluationResult = EvaluationTaskResult


@dataclass(frozen=True)
class _Readout:
    inventory: tuple[MemoryObject, ...]
    candidate_candidates: tuple[RetrievalCandidate, ...]
    candidate_memory_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    visible_memory_ids: tuple[str, ...]
    visible_candidates: tuple[RetrievalCandidate, ...]
    candidate_shortfall: bool
    trace: CommonRerankTrace | None


def evaluate_task(
    task: EvaluationTask,
    spec: SoftwareMem0VerticalSpec,
    prefix_artifact: MemoryPrefixArtifact | Mapping[str, object] | None = None,
    policy: PolicyClient | None = None,
    checker: BehaviorChecker | None = None,
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
    sampling: CausalSamplingProfile | None = None,
    full_context_max_chars: int = 100_000,
    visible_k: int = 5,
    max_output_tokens: int = 512,
) -> EvaluationTaskResult:
    """Evaluate one executable task using only frozen public/prefix inputs."""
    if policy is None:
        raise EvaluationError("policy_missing", "policy client is required")
    if checker is None:
        raise EvaluationError("checker_missing", "behavior checker is required")
    if visible_k < 1 or max_output_tokens < 1:
        raise ValueError("visible_k and max_output_tokens must be positive")
    profile = sampling or CausalSamplingProfile(max_output_tokens=max_output_tokens)
    if profile.max_output_tokens != max_output_tokens:
        raise EvaluationError(
            "sampling_mismatch", "max_output_tokens differs from sampling profile"
        )
    if task.episode_id != spec.plan.episode_id:
        raise EvaluationError("identity_mismatch", "task episode differs from spec")
    artifact = _resolve_artifact(task, prefix_artifact, prefix_artifacts)
    artifact_hash = _verify_task_boundary(task, spec, artifact)
    history = build_public_history_units(spec)
    public_by_id = {item.opportunity_id: item for item in spec.public_continuations}
    actions = spec.action_map
    rows: list[EvaluationConditionResult] = []
    for scored in task.scored_conditions:
        _validate_readout(task.condition, scored.readout)
        sceu_rows: list[EvaluationSCEUResult] = []
        for sceu in spec.plan.sceu_units:
            public = public_by_id.get(sceu.opportunity_id)
            if public is None:
                raise EvaluationError("trace_incomplete", "missing public continuation")
            surface = _surface_at(spec, sceu)
            current = replay_plan(spec.plan, sceu.checkpoint_session).current
            readout = _readout(
                task.condition,
                scored.readout,
                artifact,
                sceu,
                visible_k=visible_k,
            )
            if task.condition == "full_context":
                context = render_full_context(
                    history,
                    checkpoint_session=sceu.checkpoint_session,
                    full_context_max_chars=full_context_max_chars,
                )
            elif task.condition == "oracle_current_state":
                context = _oracle_context(sceu, current)
            else:
                context = ""
            workspace_hash = _workspace_hash(surface)
            attributions = _attributions(spec, readout.inventory, sceu)
            visible_state_ids = _visible_state_ids(readout.visible_memory_ids, attributions)
            repeats = 1 if _is_control(task.condition) else profile.baseline_repeats
            baselines = _calls(
                policy,
                checker,
                task,
                scored.result_id,
                spec,
                public,
                surface,
                sceu,
                readout.visible_candidates,
                context,
                workspace_hash,
                visible_state_ids,
                "baseline",
                repeats,
                max_output_tokens,
                actions,
            )
            interventions = _interventions(
                policy,
                checker,
                task,
                scored.result_id,
                spec,
                public,
                surface,
                sceu,
                readout,
                context,
                workspace_hash,
                attributions,
                baselines,
                profile.intervention_repeats,
                max_output_tokens,
                actions,
                enabled=not _is_control(task.condition),
            )
            primary = baselines[0]
            sceu_rows.append(
                EvaluationSCEUResult(
                    result_id=scored.result_id,
                    sceu_id=sceu.sceu_id,
                    opportunity_id=sceu.opportunity_id,
                    checkpoint_session=sceu.checkpoint_session,
                    matched_group=sceu.matched_group,
                    control_kind=_control_kind(spec, sceu),
                    prefix_artifact_hash=artifact_hash,
                    workspace_hash=workspace_hash,
                    candidate_memory_ids=readout.candidate_memory_ids,
                    retrieved_memory_ids=readout.retrieved_memory_ids,
                    model_visible_memory_ids=readout.visible_memory_ids,
                    model_visible_object_count=len(readout.visible_memory_ids),
                    model_visible_chars=sum(
                        len(item.content) for item in readout.visible_candidates
                    ),
                    selected_option_id=primary.response.selected_option_id,
                    selected_action_id=primary.selected_action_id,
                    behavior=primary.outcome,
                    normalized_drift_flags=primary.normalized_drift_flags,
                    baseline_stable=(
                        len(baselines) == 1
                        or baselines[0].outcome.signature == baselines[1].outcome.signature
                    ),
                    baseline_evaluations=baselines,
                    interventions=interventions,
                    retrieval_trace_id=(
                        None if readout.trace is None else f"{task.task_id}:{sceu.sceu_id}"
                    ),
                    transcript_hash=primary.transcript_hash,
                    model_visible_context_hash=primary.model_visible_context_hash,
                    candidate_shortfall=readout.candidate_shortfall,
                )
            )
        rows.append(
            EvaluationConditionResult(
                result_id=scored.result_id,
                condition=scored.condition,
                readout=scored.readout,
                status="complete",
                sceu_results=tuple(sceu_rows),
            )
        )
    payload = {
        "task_id": task.task_id,
        "episode_id": task.episode_id,
        "policy_profile_id": task.policy_profile_id,
        "condition": task.condition,
        "prefix_artifact_hash": artifact_hash,
        "status": "complete",
        "condition_results": [item.to_dict() for item in rows],
    }
    return EvaluationTaskResult(
        task_id=task.task_id,
        episode_id=task.episode_id,
        policy_profile_id=task.policy_profile_id,
        condition=task.condition,
        prefix_artifact_hash=artifact_hash,
        status="complete",
        condition_results=tuple(rows),
        result_hash=canonical_hash(payload),
    )


def _resolve_artifact(
    task: EvaluationTask,
    direct: MemoryPrefixArtifact | Mapping[str, object] | None,
    prefix_artifacts: Mapping[str, object] | None,
) -> MemoryPrefixArtifact | None:
    if task.prefix_backend is None:
        return None
    candidate: object | None = direct
    if candidate is None and prefix_artifacts is not None:
        candidate = prefix_artifacts.get(
            f"{task.episode_id}--{task.prefix_backend}",
            prefix_artifacts.get(task.prefix_backend),
        )
    if candidate is None:
        raise EvaluationError("prefix_missing", "memory task requires a prefix artifact")
    if isinstance(candidate, MemoryPrefixArtifact):
        return candidate
    if isinstance(candidate, Mapping):
        try:
            return MemoryPrefixArtifact.from_dict(candidate)
        except PrefixArtifactError as exc:
            raise EvaluationError("prefix_invalid", str(exc)) from exc
    raise EvaluationError("prefix_invalid", "prefix artifact must be a typed artifact record")


def _verify_task_boundary(
    task: EvaluationTask,
    spec: SoftwareMem0VerticalSpec,
    artifact: MemoryPrefixArtifact | None,
) -> str:
    if task.prefix_backend is None:
        if task.prefix_artifact_hash != NO_PREFIX_ARTIFACT:
            raise EvaluationError("prefix_mismatch", "control task must use NO_PREFIX_ARTIFACT")
        return NO_PREFIX_ARTIFACT
    if artifact is None:
        raise EvaluationError("prefix_missing", "memory task requires a prefix artifact")
    try:
        actual_hash = prefix_artifact_hash(artifact)
    except PrefixArtifactError as exc:
        raise EvaluationError("prefix_invalid", str(exc)) from exc
    if actual_hash != task.prefix_artifact_hash:
        raise EvaluationError("prefix_mismatch", "prefix artifact hash differs from task")
    if artifact.episode_id != spec.plan.episode_id:
        raise EvaluationError("identity_mismatch", "prefix episode differs from public spec")
    if artifact.backend != task.prefix_backend:
        raise EvaluationError("prefix_mismatch", "prefix backend differs from task")
    if artifact.surface_hash != spec.surface_hash:
        raise EvaluationError("prefix_mismatch", "prefix surface hash differs from public spec")
    expected = tuple(range(spec.plan.n_sessions + 1))
    actual = tuple(item.checkpoint_session for item in artifact.checkpoints)
    if actual != expected:
        raise EvaluationError("prefix_incomplete", "prefix artifact checkpoints are incomplete")
    return actual_hash


def _validate_readout(condition: str, readout: ReadoutKind) -> None:
    expected: tuple[str, ...]
    if condition in {"workspace_only", "full_context", "oracle_current_state"}:
        expected = ("none",)
    elif condition == "flat_retrieval":
        expected = ("common_rerank",)
    elif condition in {"mem0", "amem", "memos"}:
        expected = ("native", "common_rerank")
    else:
        raise EvaluationError("unknown_condition", f"unknown condition {condition!r}")
    if readout not in expected:
        raise EvaluationError(
            "missing_readout", f"unsupported readout {readout!r} for {condition!r}"
        )


def _is_control(condition: str) -> bool:
    return condition in {"workspace_only", "full_context", "oracle_current_state"}


def _surface_at(spec: SoftwareMem0VerticalSpec, sceu: SCEU) -> SessionSurface:
    if sceu.checkpoint_session >= len(spec.plan.sessions):
        raise EvaluationError("trace_incomplete", "current surface is missing")
    surface = spec.plan.sessions[sceu.checkpoint_session]
    if surface.session_index != sceu.checkpoint_session:
        raise EvaluationError("trace_incomplete", "surface session does not match SCEU")
    return surface


def _checkpoint_for(artifact: MemoryPrefixArtifact, sceu: SCEU) -> MemoryPrefixCheckpoint:
    for checkpoint in artifact.checkpoints:
        if checkpoint.checkpoint_session == sceu.checkpoint_session:
            if checkpoint.inventory is None:
                raise EvaluationError("prefix_incomplete", "checkpoint inventory is missing")
            return checkpoint
    raise EvaluationError("prefix_incomplete", "prefix lacks the SCEU checkpoint")


def _readout(
    condition: str,
    readout: ReadoutKind,
    artifact: MemoryPrefixArtifact | None,
    sceu: SCEU,
    *,
    visible_k: int,
) -> _Readout:
    if _is_control(condition):
        return _Readout((), (), (), (), (), (), False, None)
    if artifact is None:
        raise EvaluationError("prefix_missing", "memory readout requires a prefix artifact")
    checkpoint = _checkpoint_for(artifact, sceu)
    if checkpoint.inventory is None:
        raise EvaluationError("prefix_incomplete", "checkpoint inventory is missing")
    inventory = tuple(checkpoint.inventory.items)
    search = _search_for(checkpoint, sceu)
    candidate_ids = tuple(item.memory_id for item in search.candidates)
    inventory_by_id = {item.memory_id: item for item in inventory}
    for candidate in search.candidates:
        stored = inventory_by_id.get(candidate.memory_id)
        if (
            stored is None
            or stored.content_hash != candidate.content_hash
            or stored.content != candidate.content
        ):
            raise EvaluationError("prefix_invalid", "candidate does not match frozen inventory")
    if readout == "native":
        retrieved = candidate_ids[:visible_k]
        trace = None
    else:
        trace = _common_for(checkpoint, sceu, search)
        retrieved = tuple(trace.visible_memory_ids)
        if len(retrieved) > visible_k:
            raise EvaluationError("prefix_invalid", "visible IDs exceed visible_k")
    if not set(retrieved).issubset(candidate_ids):
        raise EvaluationError("prefix_invalid", "retrieved IDs are outside candidates")
    visible = tuple(retrieved[:visible_k])
    candidate_by_id = {item.memory_id: item for item in search.candidates}
    try:
        visible_candidates = tuple(candidate_by_id[item] for item in visible)
    except KeyError as exc:
        raise EvaluationError("prefix_invalid", "visible ID is outside candidate set") from exc
    return _Readout(
        inventory=inventory,
        candidate_candidates=tuple(search.candidates),
        candidate_memory_ids=candidate_ids,
        retrieved_memory_ids=tuple(retrieved),
        visible_memory_ids=visible,
        visible_candidates=visible_candidates,
        candidate_shortfall=search.candidate_shortfall,
        trace=trace,
    )


def _search_for(checkpoint: MemoryPrefixCheckpoint, sceu: SCEU) -> CandidateSearch:
    common = next(
        (item for item in checkpoint.common_reranks if item.opportunity_id == sceu.opportunity_id),
        None,
    )
    if common is not None:
        for search in checkpoint.retrievals:
            if search.query_hash == common.query_hash:
                return search
    # A prefix with one SCEU at this checkpoint has an unambiguous query even when
    # the common readout is absent/corrupt; the common-chain validator will report
    # the latter separately.
    if len(checkpoint.retrievals) == 1:
        return checkpoint.retrievals[0]
    raise EvaluationError("missing_retrieval", "checkpoint lacks retrieval for opportunity")


def _common_for(
    checkpoint: MemoryPrefixCheckpoint,
    sceu: SCEU,
    search: CandidateSearch,
) -> CommonRerankTrace:
    traces = [
        item for item in checkpoint.common_reranks if item.opportunity_id == sceu.opportunity_id
    ]
    if len(traces) != 1:
        raise EvaluationError("missing_readout", "common rerank readout is missing")
    trace = traces[0]
    candidate_ids = tuple(item.memory_id for item in search.candidates)
    if trace.query_hash != search.query_hash or trace.candidate_memory_ids != candidate_ids:
        raise EvaluationError("prefix_invalid", "common rerank is not bound to candidates")
    if not set(trace.visible_memory_ids).issubset(candidate_ids):
        raise EvaluationError("prefix_invalid", "visible IDs are outside candidates")
    return trace


def _calls(
    policy: PolicyClient,
    checker: BehaviorChecker,
    task: EvaluationTask,
    result_id: str,
    spec: SoftwareMem0VerticalSpec,
    public: PublicContinuation,
    surface: SessionSurface,
    sceu: SCEU,
    visible: tuple[RetrievalCandidate, ...],
    additional_context: str,
    workspace_hash: str,
    visible_state_ids: tuple[str, ...],
    call_kind: str,
    repeats: int,
    max_output_tokens: int,
    actions: Mapping[str, ActionSpec],
) -> tuple[EvaluationCall, ...]:
    if repeats < 1:
        raise EvaluationError("sampling_mismatch", "repeat count must be positive")
    blocks = _memory_blocks(visible)
    content = _continuation_content(
        surface,
        public,
        blocks=blocks,
        additional_context=additional_context,
    )
    output: list[EvaluationCall] = []
    for index in range(repeats):
        request = PolicyRequest(
            request_id=f"{task.task_id}:{result_id}:{sceu.sceu_id}:{call_kind}-{index}",
            system_prompt=_SYSTEM_PROMPT,
            messages=(PolicyMessage(role="user", content=content),),
            options=public.options,
            max_output_tokens=max_output_tokens,
        )
        response = _submit(policy, request)
        action_id = _action_for_option(spec, sceu, response.selected_option_id)
        if action_id not in actions:
            raise EvaluationError("structured_output_failure", "unknown evaluator action")
        try:
            behavior = checker.check_action(
                action_id,
                checkpoint_session=sceu.checkpoint_session,
                visible_state_ids=visible_state_ids,
            )
        except Exception as exc:
            raise EvaluationError("checker_failure", str(exc)) from exc
        normalized = _normalized_drift(
            spec,
            actions[action_id],
            behavior,
            sceu.checkpoint_session,
        )
        outcome = ContinuationOutcome(
            action_id=action_id,
            behavior_score=behavior.score,
            is_correct=behavior.is_correct,
            violated_state_ids=behavior.violated_state_ids,
            drift_flags=normalized,
        )
        output.append(
            EvaluationCall(
                call_id=request.request_id,
                call_kind=call_kind,
                request=request,
                response=response,
                selected_action_id=action_id,
                checker_result=behavior,
                outcome=outcome,
                normalized_drift_flags=normalized,
                workspace_hash=workspace_hash,
                transcript_hash=_sha256(content),
                policy_request_hash=_request_hash(request),
                model_visible_memory_ids=tuple(item.memory_id for item in visible),
                model_visible_blocks=blocks,
                model_visible_context_hash=_sha256("\n\n".join(blocks)),
                visible_object_count=len(visible),
                visible_object_chars=sum(len(item.content) for item in visible),
            )
        )
    return tuple(output)


def _interventions(
    policy: PolicyClient,
    checker: BehaviorChecker,
    task: EvaluationTask,
    result_id: str,
    spec: SoftwareMem0VerticalSpec,
    public: PublicContinuation,
    surface: SessionSurface,
    sceu: SCEU,
    readout: _Readout,
    additional_context: str,
    workspace_hash: str,
    attributions: Mapping[str, MemoryAttribution],
    baselines: tuple[EvaluationCall, ...],
    repeats: int,
    max_output_tokens: int,
    actions: Mapping[str, ActionSpec],
    *,
    enabled: bool,
) -> tuple[EvaluationIntervention, ...]:
    if not enabled:
        return ()
    current_ids = set(replay_plan(spec.plan, sceu.checkpoint_session).current)
    result: list[EvaluationIntervention] = []
    for target in readout.visible_candidates:
        remaining = tuple(
            item for item in readout.visible_candidates if item.memory_id != target.memory_id
        )
        loo = _calls(
            policy,
            checker,
            task,
            result_id,
            spec,
            public,
            surface,
            sceu,
            remaining,
            additional_context,
            workspace_hash,
            _visible_state_ids(tuple(item.memory_id for item in remaining), attributions),
            "leave_one_out",
            repeats,
            max_output_tokens,
            actions,
        )
        role = _memory_role(attributions.get(target.memory_id), current_ids=current_ids)
        classification = classify_causal_use(
            memory_id=target.memory_id,
            intervention_kind="leave_one_out",
            memory_role=role,
            baseline=_outcome_pair(baselines),
            intervention=_outcome_pair(loo),
        )
        result.append(
            EvaluationIntervention(
                intervention_kind="leave_one_out",
                target_memory_id=target.memory_id,
                replacement_memory_id=None,
                evaluations=loo,
                classification=classification,
            )
        )
        if role != "contradicts_current_state":
            continue
        replacement = _replacement_candidate(
            readout.candidate_candidates,
            excluded={item.memory_id for item in remaining} | {target.memory_id},
            attributions=attributions,
            current_ids=current_ids,
        )
        if replacement is None:
            continue
        replaced = list(readout.visible_candidates)
        replaced[replaced.index(target)] = replacement
        replacement_calls = _calls(
            policy,
            checker,
            task,
            result_id,
            spec,
            public,
            surface,
            sceu,
            tuple(replaced),
            additional_context,
            workspace_hash,
            _visible_state_ids(tuple(item.memory_id for item in replaced), attributions),
            "stale_replacement",
            repeats,
            max_output_tokens,
            actions,
        )
        replacement_classification = classify_causal_use(
            memory_id=target.memory_id,
            intervention_kind="stale_replacement",
            memory_role=role,
            baseline=_outcome_pair(baselines),
            intervention=_outcome_pair(replacement_calls),
        )
        result.append(
            EvaluationIntervention(
                intervention_kind="stale_replacement",
                target_memory_id=target.memory_id,
                replacement_memory_id=replacement.memory_id,
                evaluations=replacement_calls,
                classification=replacement_classification,
            )
        )
    return tuple(result)


def _submit(policy: PolicyClient, request: PolicyRequest) -> PolicyResponse:
    try:
        response = policy.submit_action(request)
    except Exception as exc:
        raise EvaluationError("policy_failure", str(exc)) from exc
    if response.request_id != request.request_id:
        raise EvaluationError("trace_incomplete", "policy response request ID mismatch")
    if response.selected_option_id not in {item.option_id for item in request.options}:
        raise EvaluationError("structured_output_failure", "policy selected an unknown option")
    return response


def _outcome_pair(
    values: tuple[EvaluationCall, ...],
) -> tuple[ContinuationOutcome, ContinuationOutcome]:
    if not values:
        raise EvaluationError("sampling_mismatch", "empty outcome repeat")
    if len(values) == 1:
        return values[0].outcome, values[0].outcome
    return values[0].outcome, values[1].outcome


def _action_for_option(spec: SoftwareMem0VerticalSpec, sceu: SCEU, option_id: str) -> str:
    try:
        return spec.evaluator_continuation_map[sceu.opportunity_id].action_for_option(option_id)
    except KeyError as exc:
        raise EvaluationError("structured_output_failure", "unknown public option") from exc


def _attributions(
    spec: SoftwareMem0VerticalSpec,
    inventory: Sequence[MemoryObject],
    sceu: SCEU,
) -> dict[str, MemoryAttribution]:
    if not inventory:
        return {}
    signatures = build_software_fact_signatures(spec.plan)
    result: dict[str, MemoryAttribution] = {}
    for item in inventory:
        source_session = dict(item.metadata).get("session_index")
        eligible = (
            eligible_write_state_ids(spec.plan, source_session)
            if isinstance(source_session, int) and source_session < sceu.checkpoint_session
            else ()
        )
        result[item.memory_id] = attribute_memory(
            item.memory_id,
            item.content,
            signatures,
            unique_write_state_ids=eligible,
        )
    return result


def _visible_state_ids(
    memory_ids: Sequence[str],
    attributions: Mapping[str, MemoryAttribution],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                state_id
                for memory_id in memory_ids
                for state_id in attributions.get(memory_id, _empty_attribution(memory_id)).state_ids
            }
        )
    )


def _empty_attribution(memory_id: str) -> MemoryAttribution:
    return MemoryAttribution(
        memory_id=memory_id,
        state_ids=(),
        method="ambiguous",
        contributes_positive_coverage=False,
        reason="missing evaluator attribution",
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
    states = [state_id in current_ids for state_id in attribution.state_ids]
    if all(states):
        return "supports_current_state"
    if not any(states):
        return "contradicts_current_state"
    return "unknown"


def _replacement_candidate(
    candidates: Sequence[RetrievalCandidate],
    *,
    excluded: set[str],
    attributions: Mapping[str, MemoryAttribution],
    current_ids: set[str],
) -> RetrievalCandidate | None:
    for candidate in candidates:
        if (
            candidate.memory_id not in excluded
            and _memory_role(attributions.get(candidate.memory_id), current_ids=current_ids)
            == "supports_current_state"
        ):
            return candidate
    return None


def _normalized_drift(
    spec: SoftwareMem0VerticalSpec,
    action: ActionSpec,
    behavior: BehaviorResult,
    checkpoint_session: int,
) -> tuple[str, ...]:
    current = replay_plan(spec.plan, checkpoint_session).current
    state_by_id = {item.state_id: item for item in spec.plan.state_units}
    stale = tuple(
        item.state_id
        for item in spec.plan.state_units
        if item.valid_from <= checkpoint_session and item.state_id not in current
    )
    local = tuple(
        state_id
        for state_id in action.satisfies_state_ids
        if state_id in state_by_id
        and (
            "local" in state_by_id[state_id].scope
            or state_by_id[state_id].authority == "local-operator"
        )
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
            used_state_ids=action.satisfies_state_ids,
            active_constraint_ids=tuple(
                state_id for state_id, state in current.items() if state.kind == "constraint"
            ),
            current_plan_state_ids=tuple(
                state_id for state_id, state in current.items() if state.kind == "plan_node"
            ),
            stale_state_ids=stale,
            selected_local_state_ids=local,
            global_state_ids=tuple(
                state_id
                for state_id, state in current.items()
                if state.kind in {"global_goal", "constraint"}
            ),
        )
    )
    return result.flags


def _oracle_context(
    sceu: SCEU,
    current: Mapping[str, StateUnit],
) -> str:
    required = tuple(item for item in sceu.required_state_ids if item in current)
    if not required:
        required = tuple(item for item in sceu.focal_state_ids if item in current)
    lines = [_state_text(current[item]) for item in required]
    return (
        "Current project state (evaluator-provided):\n" + "\n".join(f"- {line}" for line in lines)
        if lines
        else ""
    )


def _state_text(state: StateUnit) -> str:
    if isinstance(state.value, Mapping):
        text = state.value.get("text")
        if isinstance(text, str):
            return text
        return "; ".join(f"{key}: {value}" for key, value in sorted(state.value.items()))
    return str(state.value)


def _control_kind(spec: SoftwareMem0VerticalSpec, sceu: SCEU) -> str:
    for opportunity in spec.plan.opportunities:
        if opportunity.opportunity_id == sceu.opportunity_id:
            return opportunity.control_kind
    raise EvaluationError("trace_incomplete", "SCEU opportunity is missing")


def _workspace_hash(surface: SessionSurface) -> str:
    payload = {
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
    return canonical_hash(payload)


def _memory_blocks(candidates: Sequence[RetrievalCandidate]) -> tuple[str, ...]:
    return tuple(
        f"Retrieved memory {index}:\n{candidate.content}"
        for index, candidate in enumerate(candidates, 1)
    )


def _continuation_content(
    surface: SessionSurface,
    public: PublicContinuation,
    *,
    blocks: Sequence[str],
    additional_context: str,
) -> str:
    sections = [
        "Current session observations:\n"
        + json.dumps(list(surface.observations), sort_keys=True, ensure_ascii=False),
        "Current session tool results:\n"
        + json.dumps(list(surface.tool_results), sort_keys=True, ensure_ascii=False),
        "Current workspace:\n"
        + json.dumps(
            {
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
            },
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


def _request_hash(request: PolicyRequest) -> str:
    return canonical_hash(
        {
            "system_prompt": request.system_prompt,
            "messages": [asdict(item) for item in request.messages],
            "options": [item.to_dict() for item in request.options],
            "max_output_tokens": request.max_output_tokens,
        }
    )


def _request_to_dict(request: PolicyRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "system_prompt": request.system_prompt,
        "messages": [asdict(item) for item in request.messages],
        "options": [item.to_dict() for item in request.options],
        "max_output_tokens": request.max_output_tokens,
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "BehaviorChecker",
    "ConditionRunResult",
    "EvaluationCall",
    "EvaluationConditionResult",
    "EvaluationError",
    "EvaluationIntervention",
    "EvaluationResult",
    "EvaluationRetrievalTrace",
    "EvaluationSCEUResult",
    "EvaluationTaskResult",
    "EvaluationValidationError",
    "InterventionRun",
    "PolicyEvaluation",
    "SCEURunResult",
    "evaluate_task",
]
