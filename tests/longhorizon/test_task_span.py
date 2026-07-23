from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.schema import EpisodePlan, TaskStep, task_step_effect_digest
from lhmsb.longhorizon.task_span import (
    build_software_task_steps,
    profile_task_span,
)


def test_task_span_builds_causal_visible_prefix_and_policy_branches() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    steps = build_software_task_steps(spec.plan, steps_per_session=16)
    plan = replace(spec.plan, task_steps=steps)
    profile = profile_task_span(plan)

    assert profile.effective_step_count >= 16 * 16
    assert profile.visible_prefix_step_count == 16 * 16
    assert profile.policy_evaluated_step_count == len(plan.opportunities)
    assert profile.policy_conditioned_future_step_count == 0
    assert profile.policy_steps_with_downstream_effect_count == 0
    assert profile.policy_dependent_decision_count == 0
    assert profile.policy_dependency_coverage == 0.0
    assert profile.interaction_mode == "replay_backed_critical_decision"
    assert not profile.declared_closed_loop_dependency
    assert not profile.online_long_horizon_agent_execution_supported
    assert profile.session_handoff_count == 15
    assert profile.causally_linked_step_fraction == 1.0
    assert profile.semantic_effect_coverage == 1.0
    assert profile.consumed_prefix_effect_fraction == 1.0
    assert profile.anti_padding_verified
    assert profile.effect_chain_verified
    assert profile.maximum_decision_causal_span is not None
    assert profile.maximum_decision_causal_span >= 200
    assert profile.long_horizon_decision_count > 0
    assert profile.max_dependency_depth >= 255
    assert profile.meets_long_horizon_step_threshold
    assert all(
        step.execution_mode != "policy_evaluated" or not step.visible_in_session
        for step in steps
    )
    assert EpisodePlan.from_dict(plan.to_dict()) == plan


def test_task_span_rejects_zero_steps_and_existing_trace() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    with pytest.raises(ValueError, match="steps_per_session"):
        build_software_task_steps(spec.plan, steps_per_session=0)
    steps = build_software_task_steps(spec.plan, steps_per_session=4)
    with pytest.raises(ValueError, match="already contains"):
        build_software_task_steps(
            replace(spec.plan, task_steps=steps),
            steps_per_session=4,
        )


def test_empty_legacy_plan_does_not_claim_long_horizon_step_coverage() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    profile = profile_task_span(spec.plan)

    assert profile.total_step_count == 0
    assert profile.causally_linked_step_fraction is None
    assert not profile.effect_chain_verified
    assert not profile.meets_long_horizon_step_threshold
    assert profile.interaction_mode == "no_policy_evaluation"
    assert not profile.online_long_horizon_agent_execution_supported


def test_digest_only_trace_cannot_claim_long_horizon_without_semantic_effects() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    generated = build_software_task_steps(spec.plan, steps_per_session=16)
    digest_by_step: dict[str, str] = {}
    legacy_steps: list[TaskStep] = []
    for generated_step in generated:
        step = replace(
            generated_step,
            consumes_effect_ids=(),
            produces_effect_ids=(),
            dependency_effect_digests=tuple(
                digest_by_step[item] for item in generated_step.dependency_step_ids
            ),
            effect_digest="",
        )
        step = replace(step, effect_digest=task_step_effect_digest(step))
        digest_by_step[step.step_id] = step.effect_digest
        legacy_steps.append(step)

    profile = profile_task_span(replace(spec.plan, task_steps=tuple(legacy_steps)))

    assert profile.maximum_decision_causal_span is not None
    assert profile.maximum_decision_causal_span >= 200
    assert profile.semantic_effect_coverage == 0.0
    assert not profile.anti_padding_verified
    assert not profile.effect_chain_verified
    assert not profile.meets_long_horizon_step_threshold


def test_task_span_distinguishes_sparse_declared_closed_loop_from_replay() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    state_id = spec.plan.state_units[0].state_id
    steps: list[TaskStep] = []

    def append_step(step: TaskStep) -> None:
        steps.append(replace(step, effect_digest=task_step_effect_digest(step)))

    append_step(
        TaskStep(
            step_id="decision-0",
            ordinal=0,
            session=0,
            kind="continuation_decision",
            execution_mode="policy_evaluated",
            summary="",
            reads_state_ids=(state_id,),
            visible_in_session=False,
        )
    )
    append_step(
        TaskStep(
            step_id="effect-1",
            ordinal=1,
            session=1,
            kind="edit",
            execution_mode="environment_generated",
            summary="Applied the selected continuation to the project state.",
            dependency_step_ids=("decision-0",),
            dependency_effect_digests=(steps[0].effect_digest,),
            writes_state_ids=(state_id,),
        )
    )
    append_step(
        TaskStep(
            step_id="decision-2",
            ordinal=2,
            session=2,
            kind="continuation_decision",
            execution_mode="policy_evaluated",
            summary="",
            dependency_step_ids=("effect-1",),
            dependency_effect_digests=(steps[1].effect_digest,),
            reads_state_ids=(state_id,),
            visible_in_session=False,
        )
    )
    plan = replace(spec.plan, task_steps=tuple(steps))

    profile = profile_task_span(plan)

    assert profile.policy_conditioned_future_step_count == 2
    assert profile.policy_steps_with_downstream_effect_count == 1
    assert profile.policy_dependent_decision_count == 1
    assert profile.policy_dependency_coverage == 1.0
    assert profile.interaction_mode == "sparse_closed_loop"
    assert profile.declared_closed_loop_dependency
    assert not profile.online_long_horizon_agent_execution_supported
