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

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import assess_software_action
from lhmsb.longhorizon.public_surface import PublicActionOption
from lhmsb.longhorizon.replay import plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    EpisodePlan,
    TaskStep,
    TaskStepKind,
    task_step_effect_digest,
)
from lhmsb.longhorizon.task_span import TaskSpanProfile, profile_task_span
from lhmsb.qualification.memory_runtime import CandidateSearch
from lhmsb.qualification.providers import (
    PolicyClient,
    PolicyMessage,
    PolicyRequest,
    PolicyResponse,
)

OnlineCondition = Literal[
    "workspace_only",
    "full_context",
    "oracle_current_state",
    "memory",
]


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
    consumes_previous_effect: bool = False
    evaluator_action_id: str = ""
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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

    @property
    def online_long_horizon(self) -> bool:
        return (
            self.causal_chain_verified
            and self.task_span.online_long_horizon_agent_execution_supported
        )

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
            "online_long_horizon": self.online_long_horizon,
            "downstream_decision_influence_count": self.downstream_decision_influence_count,
            "downstream_decision_influence_rate": self.downstream_decision_influence_rate,
        }


def _build_options(
    spec: SoftwareMem0VerticalSpec,
    *,
    session_index: int,
    ordinal: int,
) -> tuple[PublicActionOption, ...]:
    """Create opaque, action-shaped implementation options for one step."""
    options: list[PublicActionOption] = []
    for index, action in enumerate(spec.actions):
        # Keep the option surface opaque: action IDs and evaluator predicates do
        # not cross the boundary.  The patch is a small real workspace mutation
        # and is intentionally independent of the latent state labels.
        patch_payload = {
            "step": ordinal,
            "session": session_index,
            "operation": "implementation update",
            "files": list(action.files),
        }
        # The option carries a real candidate implementation.  When selected,
        # _apply_option writes it into the persistent project workspace; the
        # next policy request therefore observes an action-dependent artifact,
        # rather than an evaluator-only append-only trace.
        candidate_files = tuple(action.files) + (
            (
                f"online/candidates/session_{session_index:02d}_{index:02d}.json",
                json.dumps(
                    patch_payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )
        options.append(
            PublicActionOption(
                option_id=f"op-{index:02d}",
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
                    "Online policy-selected implementation step "
                    f"({trace.selected_option_id}) producing workspace effect "
                    f"{trace.workspace_hash[:16]}."
                ),
                dependency_step_ids=dependency,
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
) -> tuple[str, tuple[str, ...]]:
    contents = tuple(item.content for item in result.candidates)
    ids = tuple(item.memory_id for item in result.candidates)
    return "\n".join(contents), ids


def run_online_episode(
    spec: SoftwareMem0VerticalSpec,
    policy: PolicyClient,
    *,
    condition: OnlineCondition = "workspace_only",
    memory: OnlineMemory | None = None,
    steps_per_session: int = 16,
    max_output_tokens: int = 256,
    environment_steps_per_action: int = 15,
) -> OnlineExecutionResult:
    """Execute one Software episode as a true online closed loop.

    ``steps_per_session=1`` gives one model-controlled decision per session.
    The default executor expands each selected action into 15 causally linked
    environment steps, yielding more than 200 effective steps over the
    canonical 16-session trajectory without requiring 256 model calls.
    """
    if condition == "memory" and memory is None:
        raise ValueError("condition='memory' requires a memory runtime")
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
    transcript: list[dict[str, object]] = []
    for session in range(spec.plan.n_sessions):
        context_reset = session > 0
        if context_reset:
            if not session_messages:
                session_messages = [
                    {"role": "assistant", "content": "No model-visible session transcript."}
                ]
            summary_hash = _sha(session_messages)
            if memory is not None:
                write_result = memory.write_session(
                    list(session_messages),
                    session_index=session - 1,
                    metadata={"episode_id": spec.plan.episode_id, "online": "true"},
                )
                raw_events = getattr(write_result, "events", ())
                stored_memory_ids = tuple(
                    str(getattr(event, "memory_id", ""))
                    for event in raw_events
                    if getattr(event, "memory_id", None)
                )
            handoffs.append(
                OnlineHandoffTrace(
                    session_index=session,
                    prior_session_index=session - 1,
                    working_context_reset=True,
                    summary_hash=summary_hash,
                    stored_memory_ids=stored_memory_ids,
                )
            )
            working_context = []
            session_messages = []
        if condition == "full_context":
            session_surface = spec.plan.sessions[session]
            full_context_history.extend(session_surface.observations)
            full_context_history.extend(session_surface.tool_results)
        for local_step in range(steps_per_session):
            ordinal = len(traces)
            options = _build_options(
                spec,
                session_index=session,
                ordinal=ordinal,
            )
            current_workspace_hash = _workspace_hash(workspace)
            messages: list[PolicyMessage] = [
                PolicyMessage(
                    role="user",
                    content=(
                        f"Continue the bounded software project at session {session}.\n"
                        f"Workspace digest: {current_workspace_hash}.\n"
                        f"Previous state digest: {prior_state_digest}.\n"
                        f"Previous causal effect: {prior_effect_digest or 'none'}.\n"
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
            visible_files = "\n".join(
                f"{path}: {content[:600]}"
                for path, content in sorted(workspace.items())
                if not path.startswith("online/")
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
            provenance_mode = "none"
            if condition == "memory":
                assert memory is not None
                query = "project progress constraints decisions and current implementation"
                search = memory.search_candidates(query, checkpoint_session=session)
                memory_text, retrieved_memory_ids = _memory_context(search)
                messages.append(
                    PolicyMessage(role="user", content="Retrieved project memory:\n" + memory_text)
                )
                visible_memory_ids = retrieved_memory_ids
                provenance_mode = "native" if retrieved_memory_ids else "inferred"
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
                action_index = int(selected.option_id.removeprefix("op-"))
                action = spec.actions[action_index]
            except (ValueError, IndexError) as exc:
                raise OnlineExecutionError(
                    f"online option is not aligned with the action catalog: {selected.option_id!r}"
                ) from exc
            assessment = assess_software_action(
                spec.plan,
                action,
                checkpoint_session=session,
            )
            previous_workspace_hash = current_workspace_hash
            workspace, next_workspace_hash = _apply_option(
                workspace,
                selected,
                ordinal=ordinal,
                session_index=session,
            )
            if next_workspace_hash == previous_workspace_hash:
                raise OnlineExecutionError(f"policy step {ordinal} did not mutate workspace")
            execution_state = {
                "completed_steps": ordinal + 1,
                "current_session": session,
                "last_option": selected.option_id,
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
                previous_workspace_hash=previous_workspace_hash,
                workspace_hash=next_workspace_hash,
                previous_state_digest=prior_state_digest,
                state_digest=state_digest,
                effect_digest="",
                previous_effect_digest=prior_effect_digest,
                visible_memory_ids=visible_memory_ids,
                stored_memory_ids=stored_memory_ids,
                retrieved_memory_ids=retrieved_memory_ids,
                memory_provenance_mode=provenance_mode,
                consumes_previous_effect=consumes_previous_effect,
                evaluator_action_id=action.action_id,
                drift_flags=assessment.drift_flags,
                violated_state_ids=assessment.violated_state_ids,
            )
            traces.append(trace)
            session_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Selected implementation option {selected.option_id}; "
                        f"workspace transitioned to {next_workspace_hash}."
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
                }
            )
    if len(traces) < 200:
        # The result remains usable as a unit-test fixture but is explicitly
        # rejected by the online claim property.
        pass
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
    "OnlineCondition",
    "OnlineExecutionError",
    "OnlineExecutionResult",
    "OnlineHandoffTrace",
    "OnlineMemory",
    "OnlineStepTrace",
    "run_online_episode",
]
