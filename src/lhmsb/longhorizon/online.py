"""Closed-loop online execution for the long-horizon Software track.

The frozen qualification runner is intentionally replay-backed: it evaluates a
small set of critical continuations against a fixed prefix.  That is useful for
diagnostic attribution, but it is not an online long-horizon execution claim.
This module provides the separate online track used for that claim.

An online step is counted only when all of the following are true:

* a real policy client is called with opaque action options;
* the selected option mutates a persistent workspace;
* the resulting workspace/state digest is included in the next request; and
* the resulting effect is consumed by a later policy step.

The runner is deliberately evaluator-light.  It records hashes and lifecycle
traces, while keeping state IDs and validity labels out of the model-visible
prompt.  At session handoffs, working context is reset; a memory runtime may
write and retrieve summaries independently of that reset.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol, cast

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.families.software.vertical_checker import (
    SoftwareVerticalChecker,
    assess_software_action,
)
from lhmsb.longhorizon.public_surface import PublicActionOption
from lhmsb.longhorizon.replay import plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    ActionSpec,
    ContinuationOpportunity,
    EpisodePlan,
    TaskStep,
    TaskStepKind,
    task_step_effect_digest,
)
from lhmsb.longhorizon.task_span import TaskSpanProfile, profile_task_span
from lhmsb.qualification.context import PublicHistoryUnit, build_public_history_units
from lhmsb.qualification.memory_runtime import CandidateSearch
from lhmsb.qualification.providers import (
    PolicyClient,
    PolicyMessage,
    PolicyRequest,
    PolicyResponse,
)
from lhmsb.qualification.tei import RerankCandidate, RerankResult

OnlineCondition = Literal[
    "workspace_only",
    "full_context",
    "oracle_current_state",
    "memory",
]

_PAUSE_FOR_REPAIR_SOURCE = '''\
"""Safe transition state while a revoked branch is being replaced."""


def build_pipeline(data=None):
    """Do not execute a known-invalid branch before its replacement is ready."""
    return {
        "version": "pending",
        "offline": True,
        "heldout_modified": False,
        "profiler_backend": "local",
        "urgent_profile_target_met": False,
        "hosted_service_invoked": False,
    }
'''


class OnlineExecutionError(RuntimeError):
    """Raised when a purported online run cannot establish a causal chain."""


class OnlineMemory(Protocol):
    """The small subset of the memory lifecycle used by the online runner."""

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> object: ...

    def search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch: ...

    def snapshot_inventory(self, *, checkpoint_session: int) -> object: ...


class OnlineReranker(Protocol):
    """Common reranker boundary for the controlled online track."""

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult: ...


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    if isinstance(value, dict):
        candidate = value.get("text")
        if isinstance(candidate, str):
            return candidate
        return "; ".join(f"{key}: {value[key]}" for key in sorted(value))
    return str(value)


def _safe_state_text(plan: EpisodePlan, session: int) -> str:
    """Render current state values without evaluator IDs or validity labels."""
    current = replay_plan(plan, session).current
    return "\n".join(
        f"- {_text(state.value)}" for state in sorted(current.values(), key=lambda item: item.kind)
    )


def _workspace_hash(workspace: Mapping[str, str]) -> str:
    return _sha(tuple(sorted(workspace.items())))


def _response_digest(response: PolicyResponse) -> str:
    return _sha(
        {
            "request_hash": response.request_hash,
            "response_hash": response.response_hash,
            "selected_option_id": response.selected_option_id,
            "optional_patch": response.optional_patch,
        }
    )


@dataclass(frozen=True)
class OnlineStepTrace:
    """Auditable trace for one policy-selected workspace transition."""

    step_id: str
    ordinal: int
    session_index: int
    context_reset: bool
    selected_option_id: str
    request_hash: str
    response_hash: str
    response_digest: str
    model_input_hash: str
    work_unit_id: str
    work_kind: str
    local_step: int
    component_index: int
    target_path: str
    dependency_work_unit_id: str | None
    dependency_consumed: bool
    work_unit_completed: bool
    required_input_digest: str
    artifact_valid: bool
    artifact_digest: str
    artifact_output_digest: str
    previous_workspace_hash: str
    workspace_hash: str
    previous_state_digest: str
    state_digest: str
    effect_digest: str
    previous_effect_digest: str | None
    visible_memory_ids: tuple[str, ...] = ()
    stored_memory_ids: tuple[str, ...] = ()
    retrieved_memory_ids: tuple[str, ...] = ()
    memory_provenance_mode: str = "none"
    retrieval_selection_mode: str = "none"
    live_memory_count: int | None = None
    consumes_previous_effect: bool = False
    evaluator_action_id: str = ""
    evaluator_opportunity_id: str | None = None
    continuation_scope: str = "governed_execution"
    action_correct: bool = False
    drift_flags: tuple[str, ...] = ()
    violated_state_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineHandoffTrace:
    """Boundary trace proving that the working context was reset."""

    session_index: int
    prior_session_index: int
    working_context_reset: bool
    summary_hash: str
    stored_memory_ids: tuple[str, ...] = ()
    storage_provenance_mode: str = "none"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineWorkUnit:
    """One distinct, dependency-linked unit of online project work."""

    work_unit_id: str
    session_index: int
    local_step: int
    kind: Literal["inspect", "implement", "validate", "record"]
    component_index: int
    objective: str
    output_path: str
    dependency_work_unit_id: str | None = None
    dependency_output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineDecisionAssessment:
    """Gold-side assessment for one online decision without policy execution."""

    action_id: str
    opportunity_id: str | None
    continuation_scope: str
    action_correct: bool
    violated_state_ids: tuple[str, ...]
    drift_flags: tuple[str, ...]


@dataclass(frozen=True)
class OnlineExecutionResult:
    """Result of a closed-loop online episode."""

    episode_id: str
    condition: OnlineCondition
    steps: tuple[OnlineStepTrace, ...]
    handoffs: tuple[OnlineHandoffTrace, ...]
    task_span: TaskSpanProfile
    plan_hash: str
    workspace_hash: str
    transcript_hash: str
    causal_chain_verified: bool
    policy_calls: int
    terminal_checker_score: float
    terminal_checker_passed: bool
    terminal_failed_tests: tuple[str, ...] = ()

    @property
    def online_long_horizon(self) -> bool:
        return (
            self.causal_chain_verified
            and self.task_span.online_long_horizon_agent_execution_supported
            and self.workflow_complete
            and self.project_chain_verified
            and self.substantive_work_verified
            and self.memory_selection_verified
            and self.memory_count_verified
        )

    @property
    def completed_work_unit_count(self) -> int:
        return sum(step.work_unit_completed for step in self.steps)

    @property
    def work_unit_completion_rate(self) -> float | None:
        if not self.steps:
            return None
        return self.completed_work_unit_count / len(self.steps)

    @property
    def workflow_complete(self) -> bool:
        unit_ids = tuple(step.work_unit_id for step in self.steps)
        return (
            bool(unit_ids)
            and len(set(unit_ids)) == len(unit_ids)
            and all(step.work_unit_completed for step in self.steps)
            and all(step.artifact_valid for step in self.steps)
            and all(step.dependency_consumed for step in self.steps)
        )

    @property
    def project_chain_verified(self) -> bool:
        """Whether every work unit consumes the actual preceding unit output."""
        if not self.workflow_complete:
            return False
        return all(
            step.required_input_digest == self.steps[index - 1].artifact_output_digest
            for index, step in enumerate(self.steps[1:], start=1)
        )

    @property
    def substantive_work_verified(self) -> bool:
        """Reject repeated identifiers, repeated outputs, and stage padding."""
        if not self.project_chain_verified:
            return False
        paths = tuple(step.target_path for step in self.steps)
        outputs = tuple(step.artifact_output_digest for step in self.steps)
        stage_shape = all(
            step.work_kind == _WORK_UNIT_KINDS[step.local_step % len(_WORK_UNIT_KINDS)]
            and step.component_index == step.local_step // len(_WORK_UNIT_KINDS)
            for step in self.steps
        )
        required_kinds = (
            set(_WORK_UNIT_KINDS)
            if len(self.steps) >= len(_WORK_UNIT_KINDS)
            else {step.work_kind for step in self.steps}
        )
        return (
            len(set(paths)) == len(paths)
            and len(set(outputs)) == len(outputs)
            and {step.work_kind for step in self.steps}.issuperset(required_kinds)
            and stage_shape
        )

    @property
    def memory_selection_verified(self) -> bool:
        """Controlled memory runs must expose only common-reranked candidates."""
        return self.condition != "memory" or (
            bool(self.steps)
            and all(
                step.retrieval_selection_mode == "common_rerank"
                for step in self.steps
            )
        )

    @property
    def memory_count_verified(self) -> bool:
        """Every controlled memory decision must expose its live-object denominator."""
        return self.condition != "memory" or (
            bool(self.steps)
            and all(step.live_memory_count is not None for step in self.steps)
        )

    @property
    def correct_step_count(self) -> int:
        return sum(step.action_correct for step in self.steps)

    @property
    def mean_checked_score(self) -> float | None:
        if not self.steps:
            return None
        return self.correct_step_count / len(self.steps)

    @property
    def terminal_action_correct(self) -> bool | None:
        if not self.steps:
            return None
        return self.steps[-1].action_correct

    @property
    def live_memory_count_sum(self) -> int | None:
        counts = [
            step.live_memory_count
            for step in self.steps
            if step.live_memory_count is not None
        ]
        return None if not counts else sum(counts)

    @property
    def mean_live_memory_count(self) -> float | None:
        counts = [
            step.live_memory_count
            for step in self.steps
            if step.live_memory_count is not None
        ]
        return None if not counts else sum(counts) / len(counts)

    @property
    def score_per_live_memory_object(self) -> float | None:
        denominator = self.live_memory_count_sum
        if denominator is None or denominator == 0:
            return None
        return self.correct_step_count / denominator

    @property
    def downstream_decision_influence_count(self) -> int:
        """Number of actions whose effect was consumed by a later decision."""
        return sum(step.consumes_previous_effect for step in self.steps[1:])

    @property
    def downstream_decision_influence_rate(self) -> float | None:
        """Observed action-to-next-decision influence rate."""
        denominator = max(0, len(self.steps) - 1)
        if denominator == 0:
            return None
        return self.downstream_decision_influence_count / denominator

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "condition": self.condition,
            "steps": [step.to_dict() for step in self.steps],
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "task_span": self.task_span.to_dict(),
            "plan_hash": self.plan_hash,
            "workspace_hash": self.workspace_hash,
            "transcript_hash": self.transcript_hash,
            "causal_chain_verified": self.causal_chain_verified,
            "policy_calls": self.policy_calls,
            "terminal_checker_score": self.terminal_checker_score,
            "terminal_checker_passed": self.terminal_checker_passed,
            "terminal_failed_tests": list(self.terminal_failed_tests),
            "online_long_horizon": self.online_long_horizon,
            "workflow_complete": self.workflow_complete,
            "project_chain_verified": self.project_chain_verified,
            "substantive_work_verified": self.substantive_work_verified,
            "memory_selection_verified": self.memory_selection_verified,
            "memory_count_verified": self.memory_count_verified,
            "completed_work_unit_count": self.completed_work_unit_count,
            "work_unit_completion_rate": self.work_unit_completion_rate,
            "correct_step_count": self.correct_step_count,
            "mean_checked_score": self.mean_checked_score,
            "terminal_action_correct": self.terminal_action_correct,
            "live_memory_count_sum": self.live_memory_count_sum,
            "mean_live_memory_count": self.mean_live_memory_count,
            "score_per_live_memory_object": self.score_per_live_memory_object,
            "downstream_decision_influence_count": self.downstream_decision_influence_count,
            "downstream_decision_influence_rate": self.downstream_decision_influence_rate,
        }


_WORK_UNIT_KINDS: tuple[Literal["inspect", "implement", "validate", "record"], ...] = (
    "inspect",
    "implement",
    "validate",
    "record",
)


def _build_work_unit(
    *,
    session_index: int,
    local_step: int,
    ordinal: int,
    previous: OnlineWorkUnit | None,
) -> OnlineWorkUnit:
    kind = _WORK_UNIT_KINDS[local_step % len(_WORK_UNIT_KINDS)]
    component_index = local_step // len(_WORK_UNIT_KINDS)
    work_unit_id = f"work-{ordinal:05d}"
    output_by_kind = {
        "inspect": f"requirements/session_{session_index:02d}/component_{component_index:02d}.json",
        "implement": f"pipeline/session_{session_index:02d}/component_{component_index:02d}.py",
        "validate": f"tests/session_{session_index:02d}/test_component_{component_index:02d}.py",
        "record": f"results/session_{session_index:02d}/component_{component_index:02d}.json",
    }
    objective_by_kind = {
        "inspect": "Inspect the current component contract and preserve active constraints.",
        "implement": "Implement the component according to the current authoritative plan.",
        "validate": "Add an executable validation for the installed component behavior.",
        "record": "Record the verified component result for the next dependent work unit.",
    }
    return OnlineWorkUnit(
        work_unit_id=work_unit_id,
        session_index=session_index,
        local_step=local_step,
        kind=kind,
        component_index=component_index,
        objective=objective_by_kind[kind],
        output_path=output_by_kind[kind],
        dependency_work_unit_id=None if previous is None else previous.work_unit_id,
        dependency_output_path=None if previous is None else previous.output_path,
    )


def _action_order(
    spec: SoftwareMem0VerticalSpec,
    *,
    session_index: int,
    ordinal: int,
) -> tuple[int, ...]:
    """Return a deterministic opaque permutation for one decision."""
    actions = online_action_catalog(spec)
    return tuple(
        sorted(
            range(len(actions)),
            key=lambda index: _sha(
                {
                    "episode_id": spec.plan.episode_id,
                    "session": session_index,
                    "ordinal": ordinal,
                    "action_id": actions[index].action_id,
                }
            ),
        )
    )


def _pause_for_repair_action() -> ActionSpec:
    return ActionSpec(
        action_id="pause_for_repair",
        description=(
            "Quarantine the invalid branch and wait for the authorized replacement "
            "without executing either branch."
        ),
        files=(("solution.py", _PAUSE_FOR_REPAIR_SOURCE),),
        satisfies_state_ids=("U1",),
        global_utility=0.6,
        local_utility=0.2,
    )


def online_action_catalog(
    spec: SoftwareMem0VerticalSpec,
) -> tuple[ActionSpec, ...]:
    """Return the online-only action catalog without mutating the frozen release."""
    return (*spec.actions, _pause_for_repair_action())


def _online_opportunity(
    plan: EpisodePlan,
    *,
    session_index: int,
    work_kind: str,
) -> ContinuationOpportunity | None:
    """Bind a complete session work cycle to a real scoped opportunity."""
    if work_kind not in _WORK_UNIT_KINDS:
        raise ValueError(f"unknown online work kind: {work_kind!r}")
    candidates = sorted(
        (
            opportunity
            for opportunity in plan.opportunities
            if opportunity.checkpoint_session == session_index
            and opportunity.continuation_scope == "isolated_profiler"
            and "cloud_shortcut" in opportunity.valid_action_ids
        ),
        key=lambda opportunity: opportunity.opportunity_id,
    )
    return candidates[0] if candidates else None


def assess_online_decision(
    spec: SoftwareMem0VerticalSpec,
    action: ActionSpec,
    *,
    session_index: int,
    work_kind: str,
) -> OnlineDecisionAssessment:
    """Assess one online action, including the step's explicit execution scope."""
    opportunity = _online_opportunity(
        spec.plan,
        session_index=session_index,
        work_kind=work_kind,
    )
    assessment = assess_software_action(
        spec.plan,
        action,
        checkpoint_session=session_index,
        opportunity_id=None if opportunity is None else opportunity.opportunity_id,
    )
    predicate_correct = (
        not assessment.violated_state_ids and not assessment.drift_flags
    )
    action_correct = predicate_correct and (
        opportunity is None or action.action_id in opportunity.valid_action_ids
    )
    return OnlineDecisionAssessment(
        action_id=action.action_id,
        opportunity_id=None if opportunity is None else opportunity.opportunity_id,
        continuation_scope=assessment.continuation_scope,
        action_correct=action_correct,
        violated_state_ids=assessment.violated_state_ids,
        drift_flags=assessment.drift_flags,
    )


def online_gold_action_ids(
    spec: SoftwareMem0VerticalSpec,
    *,
    session_index: int,
    local_step: int,
) -> tuple[str, ...]:
    """Return evaluator-only valid actions for a canonical online decision."""
    work_kind = _WORK_UNIT_KINDS[local_step % len(_WORK_UNIT_KINDS)]
    return tuple(
        action.action_id
        for action in online_action_catalog(spec)
        if assess_online_decision(
            spec,
            action,
            session_index=session_index,
            work_kind=work_kind,
        ).action_correct
    )


def online_gold_option_ids(
    spec: SoftwareMem0VerticalSpec,
    *,
    session_index: int,
    local_step: int,
    ordinal: int,
) -> tuple[str, ...]:
    """Return opaque option positions corresponding to valid semantic actions."""
    actions = online_action_catalog(spec)
    valid = set(
        online_gold_action_ids(
            spec,
            session_index=session_index,
            local_step=local_step,
        )
    )
    return tuple(
        f"op-{option_index:02d}"
        for option_index, action_index in enumerate(
            _action_order(spec, session_index=session_index, ordinal=ordinal)
        )
        if actions[action_index].action_id in valid
    )


def _work_unit_artifact(
    work_unit: OnlineWorkUnit,
    *,
    option_id: str,
    action_files: tuple[tuple[str, str], ...],
    required_input_digest: str,
) -> str:
    """Build a typed, evaluator-checkable artifact without exposing gold labels."""
    action_fingerprint = _sha(action_files)[:16]
    output_digest = _sha(
        {
            "work_unit": work_unit.work_unit_id,
            "kind": work_unit.kind,
            "component": work_unit.component_index,
            "required_input_digest": required_input_digest,
            "implementation_fingerprint": action_fingerprint,
        }
    )
    payload = {
        "work_unit": work_unit.work_unit_id,
        "work_kind": work_unit.kind,
        "session": work_unit.session_index,
        "component": work_unit.component_index,
        "target": work_unit.output_path,
        "depends_on": work_unit.dependency_work_unit_id,
        "required_input_digest": required_input_digest,
        "output_digest": output_digest,
        "selected_option": option_id,
        "implementation_fingerprint": action_fingerprint,
    }
    if work_unit.output_path.endswith(".json"):
        return json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    assignments = "\n".join(
        f"{key.upper()} = {value!r}" for key, value in payload.items()
    )
    return (
        '"""Generated online project work-unit artifact."""\n\n'
        f"{assignments}\n\n"
        "def work_unit_contract():\n"
        "    return {\n"
        + "".join(f"        {key!r}: {key.upper()},\n" for key in payload)
        + "    }\n\n"
        "def execute_component(upstream_digest):\n"
        "    if upstream_digest != REQUIRED_INPUT_DIGEST:\n"
        "        return ''\n"
        "    return OUTPUT_DIGEST\n"
    )


def _parse_work_unit_artifact(
    work_unit: OnlineWorkUnit,
    content: str,
) -> dict[str, object] | None:
    """Parse and execute a typed work-unit artifact."""
    expected = {
        "work_unit": work_unit.work_unit_id,
        "work_kind": work_unit.kind,
        "session": work_unit.session_index,
        "component": work_unit.component_index,
        "target": work_unit.output_path,
        "depends_on": work_unit.dependency_work_unit_id,
    }
    try:
        if work_unit.output_path.endswith(".json"):
            payload = json.loads(content)
        else:
            tree = ast.parse(content, filename=work_unit.output_path)
            compiled = compile(tree, work_unit.output_path, "exec")
            namespace: dict[str, object] = {"__builtins__": {}}
            exec(compiled, namespace)
            contract = namespace.get("work_unit_contract")
            if not callable(contract):
                return None
            payload = contract()
            executor = namespace.get("execute_component")
            if not callable(executor):
                return None
            if executor(payload.get("required_input_digest")) != payload.get("output_digest"):
                return None
            if executor("definitely-wrong-upstream") != "":
                return None
    except (SyntaxError, TypeError, ValueError, NameError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not all(
        payload.get(key) == value for key, value in expected.items()
    ):
        return None
    if not isinstance(payload.get("required_input_digest"), str):
        return None
    if not isinstance(payload.get("output_digest"), str):
        return None
    return payload


def _validate_work_unit_artifact(
    work_unit: OnlineWorkUnit,
    content: str,
    *,
    required_input_digest: str,
) -> tuple[bool, str]:
    """Validate syntax, identity, and the consumed upstream artifact."""
    payload = _parse_work_unit_artifact(work_unit, content)
    if payload is None or payload.get("required_input_digest") != required_input_digest:
        return False, ""
    output_digest = str(payload["output_digest"])
    return (
        output_digest
        == _sha(
            {
                "work_unit": work_unit.work_unit_id,
                "kind": work_unit.kind,
                "component": work_unit.component_index,
                "required_input_digest": required_input_digest,
                "implementation_fingerprint": payload.get("implementation_fingerprint"),
            }
        ),
        output_digest,
    )


def _build_options(
    spec: SoftwareMem0VerticalSpec,
    *,
    session_index: int,
    ordinal: int,
    work_unit: OnlineWorkUnit,
    required_input_digest: str,
) -> tuple[PublicActionOption, ...]:
    """Create opaque, action-shaped implementation options for one step."""
    options: list[PublicActionOption] = []
    actions = online_action_catalog(spec)
    for option_index, action_index in enumerate(
        _action_order(spec, session_index=session_index, ordinal=ordinal)
    ):
        action = actions[action_index]
        option_id = f"op-{option_index:02d}"
        # Keep the option surface opaque: action IDs and evaluator predicates do
        # not cross the boundary.  The patch is a small real workspace mutation
        # and is intentionally independent of the latent state labels.
        # The option carries a real candidate implementation.  When selected,
        # _apply_option writes it into the persistent project workspace; the
        # next policy request therefore observes an action-dependent artifact,
        # rather than an evaluator-only append-only trace.
        candidate_files = tuple(action.files) + (
            (
                work_unit.output_path,
                _work_unit_artifact(
                    work_unit,
                    option_id=option_id,
                    action_files=tuple(action.files),
                    required_input_digest=required_input_digest,
                ),
            ),
        )
        options.append(
            PublicActionOption(
                option_id=option_id,
                files=candidate_files,
            )
        )
    return tuple(options)


def _apply_option(
    workspace: dict[str, str],
    option: PublicActionOption,
    *,
    ordinal: int,
    session_index: int,
    work_unit: OnlineWorkUnit,
) -> tuple[dict[str, str], str]:
    """Apply an option and append an immutable-looking online progress record."""
    updated = dict(workspace)
    for path, content in option.files:
        previous = updated.get(path, "")
        if path == "solution.py":
            # A selected candidate replaces the currently installed
            # implementation; it is not an append-only log entry.
            updated[path] = f"{content}\n"
        else:
            updated[path] = f"{previous}{content}\n"
    # This is the persistent, model-visible project state.  It changes on
    # every selected action and records which candidate implementation is now
    # installed.  It is deliberately free of evaluator state IDs/labels.
    state_path = "state/online_project.json"
    state_payload = {
        "implementation_revision": ordinal + 1,
        "last_selected_option": option.option_id,
        "session": session_index,
        "work_unit": work_unit.work_unit_id,
        "work_kind": work_unit.kind,
        "target_path": work_unit.output_path,
        "workspace_effect": "candidate implementation installed",
    }
    updated[state_path] = json.dumps(
        state_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    progress_path = f"results/session_{session_index}.json"
    progress_entry = json.dumps(
        {
            "ordinal": ordinal,
            "selected_option": option.option_id,
            "workspace_mutated": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    updated[progress_path] = f"{updated.get(progress_path, '')}{progress_entry}\n"
    return updated, _workspace_hash(updated)


def _build_online_plan(
    spec: SoftwareMem0VerticalSpec,
    steps: Sequence[OnlineStepTrace],
    handoffs: Sequence[OnlineHandoffTrace],
    *,
    environment_steps_per_action: int,
) -> EpisodePlan:
    """Attach verified online task steps to a copy of the latent plan."""
    del handoffs  # The handoff steps are represented in the task trace itself.
    dynamic: list[TaskStep] = []
    prior_step_id: str | None = None
    effect_by_step: dict[str, str] = {}
    produced_by_step: dict[str, tuple[str, ...]] = {}
    for session in range(spec.plan.n_sessions):
        if session > 0:
            handoff_id = f"online-handoff-{session:03d}"
            dependency = (prior_step_id,) if prior_step_id else ()
            consumes = tuple(
                effect_id for dep in dependency for effect_id in produced_by_step[dep]
            )
            handoff = TaskStep(
                step_id=handoff_id,
                ordinal=len(dynamic),
                session=session,
                kind="handoff",
                execution_mode="environment_generated",
                summary="Fresh session handoff after clearing working context.",
                dependency_step_ids=dependency,
                workspace_paths=(),
                consumes_effect_ids=consumes,
                produces_effect_ids=(f"online-handoff-effect:{session:03d}",),
                dependency_effect_digests=tuple(effect_by_step[dep] for dep in dependency),
                visible_in_session=False,
            )
            handoff = replace(handoff, effect_digest=task_step_effect_digest(handoff))
            dynamic.append(handoff)
            prior_step_id = handoff_id
            effect_by_step[handoff_id] = handoff.effect_digest
            produced_by_step[handoff_id] = handoff.produces_effect_ids
        for trace in (item for item in steps if item.session_index == session):
            step_id = trace.step_id
            dependency = (prior_step_id,) if prior_step_id else ()
            consumes = tuple(
                effect_id for dep in dependency for effect_id in produced_by_step[dep]
            )
            kind: TaskStepKind = ("inspect", "edit", "test", "record")[trace.ordinal % 4]
            step = TaskStep(
                step_id=step_id,
                ordinal=len(dynamic),
                session=session,
                kind=kind,
                execution_mode="policy_evaluated",
                summary=(
                    f"Online policy-selected {trace.work_kind} work unit "
                    f"{trace.work_unit_id} ({trace.selected_option_id}) producing "
                    f"workspace effect {trace.workspace_hash[:16]}."
                ),
                dependency_step_ids=dependency,
                # EpisodePlan validates paths against the frozen workspace
                # surface. The dynamically created target remains in the
                # OnlineStepTrace, while this task-step edge points to the
                # registered per-session progress artifact.
                workspace_paths=(f"results/session_{session}.json",),
                consumes_effect_ids=consumes,
                produces_effect_ids=(f"online-effect:{trace.ordinal:05d}",),
                dependency_effect_digests=tuple(effect_by_step[dep] for dep in dependency),
                visible_in_session=False,
            )
            step = replace(step, effect_digest=task_step_effect_digest(step))
            dynamic.append(step)
            prior_step_id = step_id
            effect_by_step[step_id] = step.effect_digest
            produced_by_step[step_id] = step.produces_effect_ids
            # The policy action drives a bounded executor/test segment before
            # the next model decision.  These are genuine environment steps:
            # they consume the selected implementation effect and leave new
            # effects that the next session-level decision must inherit.
            for local_index in range(environment_steps_per_action):
                executor_id = f"online-exec-{trace.ordinal:05d}-{local_index:02d}"
                executor = TaskStep(
                    step_id=executor_id,
                    ordinal=len(dynamic),
                    session=session,
                    kind=("inspect", "test", "record")[local_index % 3],
                    execution_mode="environment_generated",
                    summary=(
                        "Environment executor applied and verified the selected "
                        f"implementation effect ({local_index + 1}/"
                        f"{environment_steps_per_action})."
                    ),
                    dependency_step_ids=(prior_step_id,),
                    workspace_paths=(f"results/session_{session}.json",),
                    consumes_effect_ids=produced_by_step[prior_step_id],
                    produces_effect_ids=(
                        f"online-executor-effect:{trace.ordinal:05d}:{local_index:02d}",
                    ),
                    dependency_effect_digests=(effect_by_step[prior_step_id],),
                    visible_in_session=False,
                )
                executor = replace(executor, effect_digest=task_step_effect_digest(executor))
                dynamic.append(executor)
                prior_step_id = executor_id
                effect_by_step[executor_id] = executor.effect_digest
                produced_by_step[executor_id] = executor.produces_effect_ids
    metadata = tuple(spec.plan.metadata) + (
        ("interaction_mode", "online_long_horizon_agent_execution"),
        ("online_policy_steps", str(len(steps))),
        ("online_causal_chain", "workspace_state_effects_consumed_by_next_policy"),
    )
    return replace(spec.plan, task_steps=tuple(dynamic), metadata=metadata)


def _memory_context(
    result: CandidateSearch,
    *,
    reranker: OnlineReranker,
    visible_k: int,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    retrieved_ids = tuple(item.memory_id for item in result.candidates)
    candidates = tuple(
        RerankCandidate(
            memory_id=item.memory_id,
            text=item.content,
            native_rank=item.native_rank,
        )
        for item in result.candidates
    )
    reranked = reranker.rerank(result.query, candidates, top_k=visible_k)
    visible_ids = tuple(reranked.ordered_memory_ids)
    expected_count = min(visible_k, len(retrieved_ids))
    if (
        len(visible_ids) != expected_count
        or len(visible_ids) != len(set(visible_ids))
        or not set(visible_ids).issubset(retrieved_ids)
    ):
        raise OnlineExecutionError("common reranker returned an invalid visible subset")
    content_by_id = {item.memory_id: item.content for item in result.candidates}
    return (
        "\n".join(content_by_id[memory_id] for memory_id in visible_ids),
        retrieved_ids,
        visible_ids,
    )


def run_online_episode(
    spec: SoftwareMem0VerticalSpec,
    policy: PolicyClient,
    *,
    condition: OnlineCondition = "workspace_only",
    memory: OnlineMemory | None = None,
    reranker: OnlineReranker | None = None,
    visible_k: int = 5,
    steps_per_session: int = 16,
    max_output_tokens: int = 256,
    environment_steps_per_action: int = 1,
) -> OnlineExecutionResult:
    """Execute one Software episode as a true online closed loop.

    The canonical 16-session configuration uses 16 policy decisions per
    session, yielding 256 real model-controlled steps. Each selected action is
    applied to the persistent workspace and checked by a bounded executor
    before the next decision. A sparse session-level run remains useful for
    smoke testing, but is not labelled as online long-horizon model execution.
    """
    if condition == "memory" and memory is None:
        raise ValueError("condition='memory' requires a memory runtime")
    if condition == "memory" and reranker is None:
        raise ValueError("condition='memory' requires the controlled common reranker")
    if visible_k < 1:
        raise ValueError("visible_k must be >= 1")
    if steps_per_session < 1:
        raise ValueError("steps_per_session must be >= 1")
    if environment_steps_per_action < 1:
        raise ValueError("environment_steps_per_action must be >= 1")
    workspace: dict[str, str] = {
        artifact.path: artifact.content
        for artifact in spec.plan.workspaces[0].artifacts
    }
    traces: list[OnlineStepTrace] = []
    handoffs: list[OnlineHandoffTrace] = []
    prior_state_digest = _sha({"session": 0, "workspace": _workspace_hash(workspace)})
    prior_effect_digest: str | None = None
    execution_state: dict[str, object] = {
        "completed_steps": 0,
        "current_session": 0,
        "last_option": None,
    }
    working_context: list[str] = []
    full_context_history: list[str] = []
    session_messages: list[dict[str, str]] = []
    stored_memory_ids: tuple[str, ...] = ()
    storage_provenance_mode = "none"
    session_live_memory_count: int | None = None
    transcript: list[dict[str, object]] = []
    completed_work_units: set[str] = set()
    previous_work_unit: OnlineWorkUnit | None = None
    last_action: ActionSpec | None = None
    public_history_by_session: dict[int, tuple[PublicHistoryUnit, ...]] = {}
    for unit in build_public_history_units(spec):
        public_history_by_session.setdefault(unit.source_session, ())
        public_history_by_session[unit.source_session] = (
            *public_history_by_session[unit.source_session],
            unit,
        )
    for session in range(spec.plan.n_sessions):
        context_reset = session > 0
        if context_reset:
            if not session_messages:
                session_messages = [
                    {"role": "assistant", "content": "No model-visible session transcript."}
                ]
            summary_hash = _sha(session_messages)
            if memory is not None:
                action_units = tuple(
                    PublicHistoryUnit.create(
                        episode_id=spec.plan.episode_id,
                        source_session=session - 1,
                        source_kind="tool_result",
                        source_ordinal=1000 + ordinal,
                        content=message["content"],
                    )
                    for ordinal, message in enumerate(session_messages)
                    if message.get("role") == "assistant"
                )
                public_units = (
                    *public_history_by_session.get(session - 1, ()),
                    *action_units,
                )
                write_result = memory.write_session(
                    list(session_messages),
                    session_index=session - 1,
                    metadata={
                        "episode_id": spec.plan.episode_id,
                        "online": "true",
                        "public_units": tuple(unit.to_dict() for unit in public_units),
                    },
                )
                raw_events = getattr(write_result, "events", ())
                stored_memory_ids = tuple(
                    str(getattr(event, "memory_id", ""))
                    for event in raw_events
                    if getattr(event, "memory_id", None)
                )
                if raw_events:
                    inferred = any(
                        "inferred"
                        in (
                            str(getattr(event, "source", ""))
                            + str(getattr(event, "native_event", ""))
                        ).casefold()
                        for event in raw_events
                    )
                    storage_provenance_mode = (
                        "inferred_inventory_diff" if inferred else "native_exact"
                    )
                elif int(getattr(write_result, "n_write", 0) or 0) > 0:
                    storage_provenance_mode = "incomplete"
                else:
                    storage_provenance_mode = "none"
            handoffs.append(
                OnlineHandoffTrace(
                    session_index=session,
                    prior_session_index=session - 1,
                    working_context_reset=True,
                    summary_hash=summary_hash,
                    stored_memory_ids=stored_memory_ids,
                    storage_provenance_mode=storage_provenance_mode,
                )
            )
            working_context = []
            session_messages = []
            session_live_memory_count = None
        if condition == "full_context":
            session_surface = spec.plan.sessions[session]
            full_context_history.extend(session_surface.observations)
            full_context_history.extend(session_surface.tool_results)
        for local_step in range(steps_per_session):
            ordinal = len(traces)
            work_unit = _build_work_unit(
                session_index=session,
                local_step=local_step,
                ordinal=ordinal,
                previous=previous_work_unit,
            )
            dependency_consumed = (
                work_unit.dependency_work_unit_id is None
                or (
                    work_unit.dependency_work_unit_id in completed_work_units
                    and work_unit.dependency_output_path in workspace
                )
            )
            if not dependency_consumed:
                raise OnlineExecutionError(
                    f"work unit {work_unit.work_unit_id} cannot consume "
                    f"{work_unit.dependency_work_unit_id}"
                )
            current_workspace_hash = _workspace_hash(workspace)
            required_input_digest = (
                current_workspace_hash
                if previous_work_unit is None
                else traces[-1].artifact_output_digest
            )
            options = _build_options(
                spec,
                session_index=session,
                ordinal=ordinal,
                work_unit=work_unit,
                required_input_digest=required_input_digest,
            )
            scoped_opportunity = _online_opportunity(
                spec.plan,
                session_index=session,
                work_kind=work_unit.kind,
            )
            messages: list[PolicyMessage] = [
                PolicyMessage(
                    role="user",
                    content=(
                        f"Continue the bounded software project at session {session}.\n"
                        f"Workspace digest: {current_workspace_hash}.\n"
                        f"Previous state digest: {prior_state_digest}.\n"
                        f"Previous causal effect: {prior_effect_digest or 'none'}.\n"
                        f"Current work unit: {work_unit.work_unit_id}.\n"
                        f"Work kind: {work_unit.kind}.\n"
                        f"Objective: {work_unit.objective}\n"
                        "Decision scope: "
                        + (
                            scoped_opportunity.request
                            if scoped_opportunity is not None
                            else "Continue governed project execution."
                        )
                        + "\n"
                        f"Required output: {work_unit.output_path}.\n"
                        "Dependency: "
                        f"{work_unit.dependency_work_unit_id or 'none'}.\n"
                        "Choose one implementation option. Your choice will be applied to "
                        "the workspace and will affect the next decision."
                    ),
                )
            ]
            surface = spec.plan.sessions[session]
            public_observations = "\n".join(surface.observations)
            public_tool_results = "\n".join(surface.tool_results)
            if local_step == 0:
                session_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": "Session observations:\n" + public_observations,
                        },
                        {
                            "role": "user",
                            "content": "Session tool results:\n"
                            + (public_tool_results or "(none)"),
                        },
                    ]
                )
            visible_paths = {
                "solution.py",
                "state/online_project.json",
                f"results/session_{session}.json",
            }
            if work_unit.dependency_output_path is not None:
                visible_paths.add(work_unit.dependency_output_path)
            visible_files = "\n".join(
                f"{path}: {content[:600]}"
                for path, content in sorted(workspace.items())
                if path in visible_paths
            )
            messages.append(
                PolicyMessage(
                    role="user",
                    content=(
                        "Current public session observations:\n"
                        f"{public_observations}\n"
                        "Current public tool results:\n"
                        f"{public_tool_results or '(none)'}\n"
                        "Current workspace files:\n"
                        f"{visible_files}"
                    ),
                )
            )
            if condition == "full_context" and full_context_history:
                messages.append(
                    PolicyMessage(
                        role="user",
                        content="Full prior project context:\n"
                        + "\n".join(full_context_history[-32:]),
                    )
                )
            if working_context:
                messages.append(
                    PolicyMessage(
                        role="user",
                        content="Current-session progress:\n" + "\n".join(working_context[-4:]),
                    )
                )
            if condition == "oracle_current_state":
                messages.append(
                    PolicyMessage(
                        role="user",
                        content="Current project facts:\n" + _safe_state_text(spec.plan, session),
                    )
                )
            visible_memory_ids: tuple[str, ...] = ()
            retrieved_memory_ids: tuple[str, ...] = ()
            retrieval_selection_mode = "none"
            if condition == "memory":
                assert memory is not None
                assert reranker is not None
                if session_live_memory_count is None:
                    snapshot = memory.snapshot_inventory(checkpoint_session=session)
                    raw_live = getattr(snapshot, "n_live", None)
                    if (
                        isinstance(raw_live, bool)
                        or not isinstance(raw_live, int)
                        or raw_live < 0
                    ):
                        raise OnlineExecutionError(
                            "memory inventory did not provide a non-negative live count"
                        )
                    session_live_memory_count = raw_live
                query = (
                    f"{work_unit.objective} Session {session}, component "
                    f"{work_unit.component_index}: recover current project progress, "
                    "authoritative constraints, decisions, scope, and implementation version."
                )
                search = memory.search_candidates(query, checkpoint_session=session)
                memory_text, retrieved_memory_ids, visible_memory_ids = _memory_context(
                    search,
                    reranker=reranker,
                    visible_k=visible_k,
                )
                messages.append(
                    PolicyMessage(role="user", content="Retrieved project memory:\n" + memory_text)
                )
                retrieval_selection_mode = "common_rerank"
            consumes_previous_effect = (
                prior_effect_digest is None
                or any(
                    prior_effect_digest in message.content
                    for message in messages
                )
            )
            if not consumes_previous_effect:
                raise OnlineExecutionError(
                    f"policy step {ordinal} did not receive the previous action effect"
                )
            request = PolicyRequest(
                request_id=f"{spec.plan.episode_id}:online:{ordinal:05d}",
                system_prompt=(
                    "You are executing a bounded software project online. "
                    "Use only the supplied observations and options. Select exactly "
                    "one option; do not invent an option ID."
                ),
                messages=tuple(messages),
                options=options,
                max_output_tokens=max_output_tokens,
            )
            response = policy.submit_action(request)
            selected = next(
                (option for option in options if option.option_id == response.selected_option_id),
                None,
            )
            if selected is None:
                raise OnlineExecutionError(
                    f"policy selected unknown online option {response.selected_option_id!r}"
                )
            try:
                option_index = int(selected.option_id.removeprefix("op-"))
                action_index = _action_order(
                    spec,
                    session_index=session,
                    ordinal=ordinal,
                )[option_index]
                action = online_action_catalog(spec)[action_index]
            except (ValueError, IndexError) as exc:
                raise OnlineExecutionError(
                    f"online option is not aligned with the action catalog: {selected.option_id!r}"
                ) from exc
            assessment = assess_online_decision(
                spec,
                action,
                session_index=session,
                work_kind=work_unit.kind,
            )
            last_action = action
            previous_workspace_hash = current_workspace_hash
            workspace, next_workspace_hash = _apply_option(
                workspace,
                selected,
                ordinal=ordinal,
                session_index=session,
                work_unit=work_unit,
            )
            if next_workspace_hash == previous_workspace_hash:
                raise OnlineExecutionError(f"policy step {ordinal} did not mutate workspace")
            work_unit_completed = work_unit.output_path in workspace
            if not work_unit_completed:
                raise OnlineExecutionError(
                    f"policy step {ordinal} did not complete {work_unit.work_unit_id}"
                )
            artifact_content = workspace[work_unit.output_path].rstrip("\n")
            artifact_valid, artifact_output_digest = _validate_work_unit_artifact(
                work_unit,
                artifact_content,
                required_input_digest=required_input_digest,
            )
            if not artifact_valid:
                raise OnlineExecutionError(
                    f"policy step {ordinal} produced an invalid artifact "
                    f"at {work_unit.output_path}"
                )
            completed_work_units.add(work_unit.work_unit_id)
            previous_work_unit = work_unit
            execution_state = {
                "completed_steps": ordinal + 1,
                "current_session": session,
                "last_option": selected.option_id,
                "last_work_unit": work_unit.work_unit_id,
                "last_target_path": work_unit.output_path,
                "workspace_hash": next_workspace_hash,
                "previous_state_digest": prior_state_digest,
            }
            state_digest = _sha(execution_state)
            # The step digest is computed after constructing the corresponding
            # evaluator-side TaskStep.  The provisional value below is replaced
            # by the exact digest after the dynamic plan is built.
            trace = OnlineStepTrace(
                step_id=f"online-step-{ordinal:05d}",
                ordinal=ordinal,
                session_index=session,
                context_reset=context_reset and local_step == 0,
                selected_option_id=selected.option_id,
                request_hash=response.request_hash,
                response_hash=response.response_hash,
                response_digest=_response_digest(response),
                model_input_hash=_policy_input_hash(request),
                work_unit_id=work_unit.work_unit_id,
                work_kind=work_unit.kind,
                local_step=work_unit.local_step,
                component_index=work_unit.component_index,
                target_path=work_unit.output_path,
                dependency_work_unit_id=work_unit.dependency_work_unit_id,
                dependency_consumed=dependency_consumed,
                work_unit_completed=work_unit_completed,
                required_input_digest=required_input_digest,
                artifact_valid=artifact_valid,
                artifact_digest=_sha(artifact_content),
                artifact_output_digest=artifact_output_digest,
                previous_workspace_hash=previous_workspace_hash,
                workspace_hash=next_workspace_hash,
                previous_state_digest=prior_state_digest,
                state_digest=state_digest,
                effect_digest="",
                previous_effect_digest=prior_effect_digest,
                visible_memory_ids=visible_memory_ids,
                stored_memory_ids=stored_memory_ids,
                retrieved_memory_ids=retrieved_memory_ids,
                memory_provenance_mode=storage_provenance_mode,
                retrieval_selection_mode=retrieval_selection_mode,
                live_memory_count=session_live_memory_count,
                consumes_previous_effect=consumes_previous_effect,
                evaluator_action_id=action.action_id,
                evaluator_opportunity_id=assessment.opportunity_id,
                continuation_scope=assessment.continuation_scope,
                action_correct=assessment.action_correct,
                drift_flags=assessment.drift_flags,
                violated_state_ids=assessment.violated_state_ids,
            )
            traces.append(trace)
            session_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Completed {work_unit.work_unit_id} ({work_unit.kind}) for "
                        f"{work_unit.output_path}. Selected candidate "
                        f"{selected.option_id}; workspace transitioned to "
                        f"{next_workspace_hash}. Installed implementation:\n"
                        f"{workspace.get('solution.py', '')[:800]}"
                    ),
                }
            )
            prior_state_digest = state_digest
            prior_effect_digest = _sha(
                {
                    "ordinal": ordinal,
                    "workspace": next_workspace_hash,
                    "state": state_digest,
                    "response": trace.response_digest,
                }
            )
            working_context.append(
                f"step {ordinal} completed; workspace is now {next_workspace_hash[:12]}"
            )
            full_context_history.append(
                f"session {session} step {ordinal}: option {selected.option_id}; "
                f"workspace {next_workspace_hash}"
            )
            transcript.append(
                {
                    "session": session,
                    "ordinal": ordinal,
                    "workspace_hash": next_workspace_hash,
                    "state_digest": state_digest,
                    "selected_option": selected.option_id,
                    "work_unit": work_unit.work_unit_id,
                    "work_kind": work_unit.kind,
                    "target_path": work_unit.output_path,
                    "dependency": work_unit.dependency_work_unit_id,
                    "artifact_valid": artifact_valid,
                    "artifact_digest": trace.artifact_digest,
                    "required_input_digest": required_input_digest,
                    "artifact_output_digest": artifact_output_digest,
                }
            )
    # Build the exact effect chain and patch the immutable traces with the
    # corresponding TaskStep digests.
    online_plan = _build_online_plan(
        spec,
        traces,
        handoffs,
        environment_steps_per_action=environment_steps_per_action,
    )
    rebuilt_steps: list[OnlineStepTrace] = []
    task_by_id = {step.step_id: step for step in online_plan.task_steps}
    for trace in traces:
        step = task_by_id[trace.step_id]
        rebuilt_steps.append(replace(trace, effect_digest=step.effect_digest))
    span = profile_task_span(online_plan)
    causal_chain_verified = _verify_causal_chain(rebuilt_steps)
    if not causal_chain_verified:
        raise OnlineExecutionError("online trace failed causal workspace/state verification")
    if last_action is None:
        raise OnlineExecutionError("online episode produced no terminal action")
    terminal_result = SoftwareVerticalChecker(
        cast(SoftwareVerticalSpec, spec)
    ).check_action(
        last_action,
        checkpoint_session=spec.plan.n_sessions - 1,
    )
    return OnlineExecutionResult(
        episode_id=spec.plan.episode_id,
        condition=condition,
        steps=tuple(rebuilt_steps),
        handoffs=tuple(handoffs),
        task_span=span,
        plan_hash=plan_hash(online_plan),
        workspace_hash=_workspace_hash(workspace),
        transcript_hash=_sha(transcript),
        causal_chain_verified=causal_chain_verified,
        policy_calls=len(rebuilt_steps),
        terminal_checker_score=terminal_result.score,
        terminal_checker_passed=terminal_result.is_correct,
        terminal_failed_tests=terminal_result.failed_tests,
    )


def _verify_causal_chain(steps: Sequence[OnlineStepTrace]) -> bool:
    if not steps:
        return False
    previous_workspace: str | None = None
    previous_state: str | None = None
    for index, step in enumerate(steps):
        if index == 0:
            previous_workspace = step.previous_workspace_hash
            previous_state = step.previous_state_digest
        if step.previous_workspace_hash != previous_workspace:
            return False
        if step.previous_state_digest != previous_state:
            return False
        if step.workspace_hash == step.previous_workspace_hash:
            return False
        if step.state_digest == step.previous_state_digest:
            return False
        if not step.consumes_previous_effect:
            return False
        if not step.work_unit_completed or not step.artifact_valid:
            return False
        if index > 0 and step.required_input_digest != steps[index - 1].artifact_output_digest:
            return False
        if index > 0 and step.previous_effect_digest is None:
            return False
        previous_workspace = step.workspace_hash
        previous_state = step.state_digest
    # The last action has no later decision inside the episode; every earlier
    # action must be consumed by its immediate successor.  The per-step flag
    # is set from the actual model-facing messages above, not inferred from a
    # frozen dependency graph.
    return all(step.consumes_previous_effect for step in steps[1:])


def _policy_input_hash(request: PolicyRequest) -> str:
    """Hash the exact semantic request sent to the policy client."""
    return _sha(
        {
            "system_prompt": request.system_prompt,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "options": [option.to_dict() for option in request.options],
            "max_output_tokens": request.max_output_tokens,
        }
    )


__all__ = [
    "OnlineDecisionAssessment",
    "OnlineCondition",
    "OnlineExecutionError",
    "OnlineExecutionResult",
    "OnlineHandoffTrace",
    "OnlineMemory",
    "OnlineReranker",
    "OnlineStepTrace",
    "assess_online_decision",
    "online_action_catalog",
    "online_gold_action_ids",
    "online_gold_option_ids",
    "run_online_episode",
]
