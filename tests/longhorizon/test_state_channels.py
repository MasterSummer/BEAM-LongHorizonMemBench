from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.longhorizon.state_channel_plan import (
    build_state_channel_experiment_plan,
)
from lhmsb.longhorizon.state_channels import (
    ChallengeAxes,
    DecisionAnchor,
    audit_core_channel_matrix,
    build_core_channel_matrix,
    core_channel_contrasts,
    derive_core_channel_matrices,
)


def _anchor() -> DecisionAnchor:
    return DecisionAnchor(
        episode_id="software-longitudinal-42",
        opportunity_id="opp-stale-v1",
        checkpoint_session=8,
        decision_family="branch_selection",
        authoritative_state_ids=("P2",),
        stale_state_ids=("P1",),
        gold_action_ids=("safe_v2_offline",),
        prompt_template_id="continue-governed-project-v1",
    )


def _axes() -> ChallengeAxes:
    return ChallengeAxes(
        horizon_handoffs=8,
        state_transition_count=2,
        workspace_version_depth=4,
        workspace_distractor_count=3,
        memory_distractor_count=3,
        authority_depth=2,
        dependency_depth=3,
        scope_switch_count=1,
    )


def test_core_channel_matrix_is_an_exact_matched_three_by_three() -> None:
    arms = build_core_channel_matrix(_anchor(), _axes(), block_id="block-001")
    audit = audit_core_channel_matrix(arms)

    assert audit.ok
    assert audit.arm_count == 9
    assert audit.combination_count == 9
    assert audit.anchor_invariant
    assert audit.axes_invariant
    assert {arm.relation for arm in arms} == {
        "aligned_current",
        "aligned_stale",
        "cross_channel_conflict",
        "single_current_channel",
        "single_stale_channel",
        "both_absent",
    }
    assert sum(arm.workspace_condition == "current" for arm in arms) == 3
    assert sum(arm.memory_condition == "current" for arm in arms) == 3


def test_matrix_audit_rejects_question_or_nonchannel_axis_confounds() -> None:
    arms = list(build_core_channel_matrix(_anchor(), _axes(), block_id="block-001"))
    arms[0] = replace(
        arms[0],
        anchor=replace(arms[0].anchor, prompt_template_id="different-question"),
        axes=replace(arms[0].axes, workspace_distractor_count=99),
    )

    audit = audit_core_channel_matrix(tuple(arms))

    assert not audit.ok
    assert "decision-anchor-varies" in audit.errors
    assert "non-channel-axes-vary" in audit.errors


def test_core_contrasts_identify_rescue_harm_conflict_and_synergy() -> None:
    arms = build_core_channel_matrix(_anchor(), _axes(), block_id="block-001")
    contrasts = {item.contrast_id: item for item in core_channel_contrasts(arms)}
    scores = {
        arm.arm_id: {
            ("current", "current"): 0.95,
            ("current", "stale"): 0.80,
            ("current", "absent"): 0.85,
            ("stale", "current"): 0.70,
            ("stale", "stale"): 0.10,
            ("stale", "absent"): 0.15,
            ("absent", "current"): 0.75,
            ("absent", "stale"): 0.20,
            ("absent", "absent"): 0.30,
        }[(arm.workspace_condition, arm.memory_condition)]
        for arm in arms
    }

    assert len(contrasts) == 8
    assert contrasts[
        "workspace-current-rescue-given-memory-absent"
    ].evaluate(scores) == pytest.approx(0.55)
    assert contrasts[
        "memory-current-rescue-given-workspace-absent"
    ].evaluate(scores) == pytest.approx(0.45)
    assert contrasts[
        "workspace-stale-harm-given-memory-absent"
    ].evaluate(scores) == pytest.approx(-0.15)
    assert contrasts[
        "cross-channel-current-source-preference"
    ].evaluate(scores) == pytest.approx(0.10)
    assert contrasts["current-channel-synergy"].evaluate(scores) == pytest.approx(-0.35)


def test_stale_arms_require_real_versioned_state_evolution() -> None:
    with pytest.raises(ValueError, match="at least one state transition"):
        replace(_axes(), state_transition_count=0)
    with pytest.raises(ValueError, match="version depth"):
        replace(_axes(), workspace_version_depth=1)


def test_real_longitudinal_plan_yields_multiple_flexible_matched_blocks() -> None:
    spec = SoftwareLongitudinalTrajectoryFamily.generate(
        42,
        n_sessions=16,
        trajectory_seed=2,
        steps_per_session=16,
    )

    matrices = derive_core_channel_matrices(spec.plan)

    assert len(matrices) >= 10
    assert all(len(matrix) == 9 for matrix in matrices)
    assert all(audit_core_channel_matrix(matrix).ok for matrix in matrices)
    anchors = {matrix[0].anchor for matrix in matrices}
    assert len({anchor.opportunity_id for anchor in anchors}) == len(matrices)
    assert {anchor.decision_family for anchor in anchors} == {
        "branch_selection",
        "constraint_compliance",
        "scope_authority",
        "repair_after_failure",
    }
    assert all(anchor.authoritative_state_ids == ("P2",) for anchor in anchors)
    assert all(anchor.stale_state_ids == ("P1",) for anchor in anchors)

    conflict_arm = next(
        arm
        for matrix in matrices
        for arm in matrix
        if arm.workspace_condition == "current"
        and arm.memory_condition == "stale"
    )
    assert conflict_arm.workspace_state_ids == ("P2",)
    assert conflict_arm.memory_state_ids == ("P1",)


def test_balanced_experiment_plan_has_exact_model_call_budget() -> None:
    specs = tuple(
        SoftwareLongitudinalTrajectoryFamily.generate(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
            steps_per_session=16,
        )
        for seed in (42, 43)
    )

    plan = build_state_channel_experiment_plan(specs)

    assert plan["design_audit"]["ok"]
    assert plan["episode_count"] == 2
    assert plan["block_count"] == 8
    assert plan["arm_count"] == 72
    assert plan["decision_family_counts"] == {
        "branch_selection": 2,
        "constraint_compliance": 2,
        "repair_after_failure": 2,
        "scope_authority": 2,
    }
    # 9 arms × 2 repeats + 4 protocols × (repair pair + retention pair).
    assert plan["planned_policy_calls_per_system"] == 8 * 34
