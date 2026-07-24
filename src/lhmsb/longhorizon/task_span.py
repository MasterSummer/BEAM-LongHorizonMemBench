"""Task-step provenance for sparse evaluation of long-horizon trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, cast

from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import (
    EpisodePlan,
    StateUnit,
    TaskStep,
    TaskStepExecutionMode,
    TaskStepKind,
    task_step_effect_digest,
)

MIN_LONG_HORIZON_EFFECTIVE_STEPS = 200
# Online execution is session-level control over a long executable trace.  We
# require one real policy decision per canonical session, while the executor
# may carry out many causally dependent implementation/test steps between
# decisions.  This avoids conflating horizon with the number of model calls.
MIN_ONLINE_LONG_HORIZON_POLICY_STEPS = 16

TrajectoryInteractionMode = Literal[
    "no_policy_evaluation",
    "replay_backed_critical_decision",
    "sparse_closed_loop",
    "online_long_horizon_agent_execution",
]

_WORK_KINDS = ("inspect", "edit", "test", "record")


@dataclass(frozen=True)
class TaskSpanProfile:
    """Auditable task-horizon counts for one episode."""

    episode_id: str
    total_step_count: int
    effective_step_count: int
    visible_prefix_step_count: int
    policy_evaluated_step_count: int
    frozen_replay_step_count: int
    environment_generated_step_count: int
    policy_conditioned_future_step_count: int
    policy_steps_with_downstream_effect_count: int
    policy_dependent_decision_count: int
    policy_dependency_coverage: float | None
    minimum_decision_causal_span: int | None
    maximum_decision_causal_span: int | None
    long_horizon_decision_count: int
    interaction_mode: TrajectoryInteractionMode
    declared_closed_loop_dependency: bool
    online_long_horizon_agent_execution_supported: bool
    session_handoff_count: int
    max_dependency_depth: int
    causally_linked_step_fraction: float | None
    semantic_effect_step_count: int
    semantic_effect_coverage: float | None
    consumed_prefix_effect_fraction: float | None
    anti_padding_verified: bool
    effect_chain_verified: bool
    meets_long_horizon_step_threshold: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_software_task_steps(
    plan: EpisodePlan,
    *,
    steps_per_session: int,
) -> tuple[TaskStep, ...]:
    """Build a deterministic, causally linked Software task trace.

    ``steps_per_session`` is a minimum number of public prefix transitions per
    session. State transitions and handoffs are always represented; additional
    inspect/edit/test/record steps carry realistic progress and distractor
    load. Continuation decisions are evaluator-side branches and are not
    rendered back into the prefix.
    """

    if steps_per_session < 1:
        raise ValueError("steps_per_session must be >= 1")
    if plan.task_steps:
        raise ValueError("plan already contains task steps")
    states = {state.state_id: state for state in plan.state_units}
    events_by_session = {
        session: tuple(
            event
            for event in sorted(
                plan.events,
                key=lambda item: (item.session, item.event_id),
            )
            if event.session == session
        )
        for session in range(plan.n_sessions)
    }
    opportunities_by_session = {
        session: tuple(
            opportunity
            for opportunity in plan.opportunities
            if opportunity.checkpoint_session == session
        )
        for session in range(plan.n_sessions)
    }
    steps: list[TaskStep] = []
    effect_by_step: dict[str, str] = {}
    semantic_effects_by_step: dict[str, tuple[str, ...]] = {}
    previous_prefix_step_id: str | None = None

    def add_step(
        *,
        session: int,
        kind: str,
        execution_mode: str,
        summary: str,
        dependency_step_ids: tuple[str, ...] = (),
        reads_state_ids: tuple[str, ...] = (),
        writes_state_ids: tuple[str, ...] = (),
        workspace_paths: tuple[str, ...] = (),
        visible_in_session: bool = True,
        semantic_effect_id: str,
    ) -> str:
        step_id = f"step-{len(steps):05d}"
        consumes_effect_ids = tuple(
            effect_id
            for dependency_id in dependency_step_ids
            for effect_id in semantic_effects_by_step[dependency_id]
        )
        provisional = TaskStep(
            step_id=step_id,
            ordinal=len(steps),
            session=session,
            kind=cast(TaskStepKind, kind),
            execution_mode=cast(
                TaskStepExecutionMode,
                execution_mode,
            ),
            summary=summary,
            dependency_step_ids=dependency_step_ids,
            reads_state_ids=reads_state_ids,
            writes_state_ids=writes_state_ids,
            workspace_paths=workspace_paths,
            consumes_effect_ids=consumes_effect_ids,
            produces_effect_ids=(semantic_effect_id,),
            dependency_effect_digests=tuple(
                effect_by_step[step_id] for step_id in dependency_step_ids
            ),
            visible_in_session=visible_in_session,
        )
        step = replace(
            provisional,
            effect_digest=task_step_effect_digest(provisional),
        )
        steps.append(step)
        effect_by_step[step_id] = step.effect_digest
        semantic_effects_by_step[step_id] = step.produces_effect_ids
        return step_id

    for session in range(plan.n_sessions):
        session_prefix_start = len(steps)
        if session > 0:
            previous_prefix_step_id = add_step(
                session=session,
                kind="handoff",
                execution_mode="environment_generated",
                summary=(
                    "Resumed the persistent project in a fresh session and "
                    "continued from the preceding verified checkpoint."
                ),
                dependency_step_ids=(previous_prefix_step_id,) if previous_prefix_step_id else (),
                semantic_effect_id=f"handoff:{session:03d}",
            )
        for event in events_by_session[session]:
            state = states[event.target_state_id]
            dependency = (previous_prefix_step_id,) if previous_prefix_step_id else ()
            previous_prefix_step_id = add_step(
                session=session,
                kind="state_transition",
                execution_mode="environment_generated",
                summary=_event_summary(event.type, state),
                dependency_step_ids=dependency,
                reads_state_ids=event.reason_state_ids,
                writes_state_ids=(event.target_state_id,),
                semantic_effect_id=f"state-transition:{event.event_id}",
            )
        snapshot = next(
            (item for item in plan.workspaces if item.checkpoint_session == session),
            None,
        )
        paths = () if snapshot is None else tuple(artifact.path for artifact in snapshot.artifacts)
        current_state_ids = tuple(sorted(replay_plan(plan, session).current))
        visible_count = sum(step.visible_in_session for step in steps[session_prefix_start:])
        work_index = 0
        while visible_count < steps_per_session:
            kind = _WORK_KINDS[work_index % len(_WORK_KINDS)]
            path = paths[work_index % len(paths)] if paths else ""
            state_id = (
                current_state_ids[work_index % len(current_state_ids)] if current_state_ids else ""
            )
            dependency = (previous_prefix_step_id,) if previous_prefix_step_id else ()
            previous_prefix_step_id = add_step(
                session=session,
                kind=kind,
                execution_mode="frozen_replay",
                summary=_work_summary(
                    session=session,
                    index=work_index,
                    kind=kind,
                    path=path,
                ),
                dependency_step_ids=dependency,
                reads_state_ids=(state_id,) if state_id else (),
                workspace_paths=(path,) if path else (),
                semantic_effect_id=(
                    f"work:{session:03d}:{work_index:03d}:{kind}"
                ),
            )
            work_index += 1
            visible_count += 1

        decision_parent = previous_prefix_step_id
        for opportunity in opportunities_by_session[session]:
            add_step(
                session=session,
                kind="continuation_decision",
                execution_mode="policy_evaluated",
                summary="",
                dependency_step_ids=(decision_parent,) if decision_parent else (),
                reads_state_ids=tuple(
                    state_id for state_id in opportunity.focal_state_ids if state_id in states
                ),
                visible_in_session=False,
                semantic_effect_id=(
                    f"continuation:{opportunity.opportunity_id}"
                ),
            )
    return tuple(steps)


def profile_task_span(plan: EpisodePlan) -> TaskSpanProfile:
    """Compute task-span evidence without treating tokens as horizon."""

    steps = plan.task_steps
    effective = tuple(step for step in steps if step.effective)
    depths: dict[str, int] = {}
    ancestors: dict[str, frozenset[str]] = {}
    for step in steps:
        depths[step.step_id] = (
            0
            if not step.dependency_step_ids
            else 1 + max(depths[item] for item in step.dependency_step_ids)
        )
        ancestor_ids: set[str] = set()
        for dependency_id in step.dependency_step_ids:
            ancestor_ids.add(dependency_id)
            ancestor_ids.update(ancestors[dependency_id])
        ancestors[step.step_id] = frozenset(ancestor_ids)
    linked = sum(bool(step.dependency_step_ids) or step.ordinal == 0 for step in effective)
    effective_count = len(effective)
    effective_ids = {step.step_id for step in effective}
    step_by_id = {step.step_id: step for step in steps}
    policy_ids = {
        step.step_id
        for step in effective
        if step.execution_mode == "policy_evaluated"
    }
    policy_count = len(policy_ids)
    decision_ancestor_ids = set().union(
        *(ancestors[step_id] for step_id in policy_ids),
    )
    decision_prefix_ids = decision_ancestor_ids.intersection(
        effective_ids.difference(policy_ids)
    )
    decision_causal_spans = tuple(
        len(
            ancestors[step.step_id].intersection(
                effective_ids.difference(policy_ids)
            )
        )
        for step in effective
        if step.step_id in policy_ids
    )
    policy_ancestors_by_step = {
        step.step_id: ancestors[step.step_id].intersection(policy_ids)
        for step in effective
    }
    policy_conditioned_future_step_count = sum(
        bool(policy_ancestors_by_step[step.step_id]) for step in effective
    )
    policy_steps_with_downstream_effect = set().union(
        *(policy_ancestors_by_step[step.step_id] for step in effective),
    )
    policy_dependent_decision_count = sum(
        step.step_id in policy_ids
        and bool(policy_ancestors_by_step[step.step_id])
        for step in effective
    )
    policy_dependency_coverage = (
        None
        if policy_count <= 1
        else policy_dependent_decision_count / (policy_count - 1)
    )
    declared_closed_loop_dependency = (
        bool(policy_steps_with_downstream_effect)
        and policy_dependent_decision_count > 0
    )
    produced_effect_ids = {
        effect_id
        for step in effective
        for effect_id in step.produces_effect_ids
    }
    semantic_effect_step_count = sum(
        bool(step.produces_effect_ids) for step in effective
    )
    semantic_effect_coverage = (
        None
        if effective_count == 0
        else semantic_effect_step_count / effective_count
    )
    consumed_effect_ids = {
        effect_id
        for step in effective
        for effect_id in step.consumes_effect_ids
    }
    prefix_produced_effect_ids = {
        effect_id
        for step in effective
        if step.step_id in decision_prefix_ids
        for effect_id in step.produces_effect_ids
    }
    consumed_prefix_effect_fraction = (
        None
        if not prefix_produced_effect_ids
        else len(prefix_produced_effect_ids.intersection(consumed_effect_ids))
        / len(prefix_produced_effect_ids)
    )
    unique_semantic_effects = len(produced_effect_ids) == sum(
        len(step.produces_effect_ids) for step in effective
    )
    semantic_edges_align = all(
        step.consumes_effect_ids
        == tuple(
            effect_id
            for dependency_id in step.dependency_step_ids
            for effect_id in step_by_id[dependency_id].produces_effect_ids
        )
        for step in effective
    )
    digest_chain_verified = bool(steps) and all(
        step.effect_digest and step.effect_digest == task_step_effect_digest(step)
        for step in steps
    )
    anti_padding_verified = (
        semantic_effect_coverage == 1.0
        and consumed_prefix_effect_fraction == 1.0
        and unique_semantic_effects
        and semantic_edges_align
    )
    effect_chain_verified = digest_chain_verified and anti_padding_verified
    online_long_horizon_agent_execution_supported = (
        policy_count >= MIN_ONLINE_LONG_HORIZON_POLICY_STEPS
        and effective_count >= MIN_LONG_HORIZON_EFFECTIVE_STEPS
        and policy_dependency_coverage is not None
        and policy_dependency_coverage >= 0.99
        and declared_closed_loop_dependency
        and effect_chain_verified
    )
    interaction_mode: TrajectoryInteractionMode
    if policy_count == 0:
        interaction_mode = "no_policy_evaluation"
    elif online_long_horizon_agent_execution_supported:
        interaction_mode = "online_long_horizon_agent_execution"
    elif declared_closed_loop_dependency:
        interaction_mode = "sparse_closed_loop"
    else:
        interaction_mode = "replay_backed_critical_decision"
    return TaskSpanProfile(
        episode_id=plan.episode_id,
        total_step_count=len(steps),
        effective_step_count=effective_count,
        visible_prefix_step_count=sum(step.visible_in_session for step in effective),
        policy_evaluated_step_count=policy_count,
        frozen_replay_step_count=sum(step.execution_mode == "frozen_replay" for step in effective),
        environment_generated_step_count=sum(
            step.execution_mode == "environment_generated" for step in effective
        ),
        policy_conditioned_future_step_count=(
            policy_conditioned_future_step_count
        ),
        policy_steps_with_downstream_effect_count=len(
            policy_steps_with_downstream_effect
        ),
        policy_dependent_decision_count=policy_dependent_decision_count,
        policy_dependency_coverage=policy_dependency_coverage,
        minimum_decision_causal_span=min(decision_causal_spans, default=None),
        maximum_decision_causal_span=max(decision_causal_spans, default=None),
        long_horizon_decision_count=sum(
            span >= MIN_LONG_HORIZON_EFFECTIVE_STEPS
            for span in decision_causal_spans
        ),
        interaction_mode=interaction_mode,
        declared_closed_loop_dependency=declared_closed_loop_dependency,
        online_long_horizon_agent_execution_supported=(
            online_long_horizon_agent_execution_supported
        ),
        session_handoff_count=sum(step.kind == "handoff" for step in effective),
        max_dependency_depth=max(depths.values(), default=0),
        causally_linked_step_fraction=(None if effective_count == 0 else linked / effective_count),
        semantic_effect_step_count=semantic_effect_step_count,
        semantic_effect_coverage=semantic_effect_coverage,
        consumed_prefix_effect_fraction=consumed_prefix_effect_fraction,
        anti_padding_verified=anti_padding_verified,
        effect_chain_verified=effect_chain_verified,
        meets_long_horizon_step_threshold=(
            bool(decision_causal_spans)
            and max(decision_causal_spans) >= MIN_LONG_HORIZON_EFFECTIVE_STEPS
            and anti_padding_verified
            and effect_chain_verified
        ),
    )


def _state_text(state: StateUnit) -> str:
    if isinstance(state.value, dict):
        text = state.value.get("text")
        if isinstance(text, str):
            return text
        return "; ".join(f"{key}: {value}" for key, value in sorted(state.value.items()))
    return str(state.value)


def _event_summary(event_type: str, state: StateUnit) -> str:
    text = _state_text(state)
    if event_type in {"replace", "revoke", "invalidate"}:
        return f"Session update: an earlier item changed — {text}"
    return f"Session update: {text}"


def _work_summary(*, session: int, index: int, kind: str, path: str) -> str:
    action = {
        "inspect": "inspected the current artifact",
        "edit": "applied a deterministic implementation increment to",
        "test": "ran the dependent local verification for",
        "record": "recorded auditable progress for",
    }[kind]
    target = path or "the bounded project workspace"
    return (
        f"Progress {session:02d}.{index:02d}: {action} {target}; "
        "the step completed locally and its successor depends on this result."
    )


__all__ = [
    "MIN_LONG_HORIZON_EFFECTIVE_STEPS",
    "MIN_ONLINE_LONG_HORIZON_POLICY_STEPS",
    "TaskSpanProfile",
    "TrajectoryInteractionMode",
    "build_software_task_steps",
    "profile_task_span",
]
