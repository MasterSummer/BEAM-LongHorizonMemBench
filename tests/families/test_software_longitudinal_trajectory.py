from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from lhmsb.families.software.longitudinal_trajectory import (
    LONGITUDINAL_RECOVERY_OPPORTUNITY_ID,
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.design_audit import (
    build_analysis_contract,
    compute_experiment_design_audit,
)


def _spec() -> SoftwareMem0VerticalSpec:
    return SoftwareLongitudinalTrajectoryFamily.generate(
        42,
        n_sessions=16,
        trajectory_seed=42,
        steps_per_session=16,
    )


def _checks(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row["check_id"]): row
        for row in payload["checks"]  # type: ignore[index]
    }


def test_longitudinal_release_adds_a_final_same_lineage_recovery_decision() -> None:
    base = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    spec = _spec()
    opportunity = spec.plan.opportunities[-1]

    assert len(base.plan.opportunities) == 17
    assert len(spec.plan.opportunities) == 18
    assert len(spec.plan.sceu_units) == 18
    assert opportunity.opportunity_id == LONGITUDINAL_RECOVERY_OPPORTUNITY_ID
    assert opportunity.checkpoint_session == 15
    assert opportunity.control_kind == "fresh_reminder"
    assert opportunity.continuation_scope == "governed_execution"
    assert opportunity.valid_action_ids == ("safe_v2_offline",)
    assert Counter(
        item.valid_action_ids[0] for item in spec.plan.opportunities
    ) == {
        "safe_v2_offline": 7,
        "stale_v1": 5,
        "cloud_shortcut": 6,
    }


def test_longitudinal_sceu_contract_uses_only_current_action_relevant_targets() -> None:
    spec = _spec()

    for sceu in spec.plan.sceu_units:
        profile = profile_sceu(spec.plan, sceu)
        assert sceu.required_state_ids == profile.current_required_state_ids
        assert sceu.dependency_closure == profile.current_required_state_ids
        assert set(sceu.intervention_target_ids).issubset(
            profile.current_action_relevant_state_ids
        )
        assert not profile.future_referenced_state_ids
        assert not profile.missing_current_action_relevant_state_ids


def test_longitudinal_task_span_is_causal_and_not_an_online_rollout_claim() -> None:
    span = profile_task_span(_spec().plan)

    assert span.total_step_count == 274
    assert span.effective_step_count == 274
    assert span.visible_prefix_step_count == 256
    assert span.policy_evaluated_step_count == 18
    assert span.maximum_decision_causal_span == 256
    assert span.long_horizon_decision_count == 4
    assert span.anti_padding_verified
    assert span.effect_chain_verified
    assert span.meets_long_horizon_step_threshold
    assert span.interaction_mode == "replay_backed_critical_decision"
    assert not span.online_long_horizon_agent_execution_supported


def test_longitudinal_design_freezes_c2_lineage_recovery_and_c3_targets() -> None:
    spec = _spec()
    payload = compute_experiment_design_audit({spec.plan.episode_id: spec})
    checks = _checks(payload)

    assert payload["scope"] == "longitudinal_trajectory"
    assert payload["analysis_unit"] == "episode"
    assert payload["run_ready"] is True
    assert payload["analysis_contract"] == build_analysis_contract(
        matched=False,
        longitudinal=True,
    )
    assert payload["analysis_contract"]["claim_id"] == "C1-C3-longitudinal"  # type: ignore[index]
    for check_id in (
        "longitudinal_release_membership",
        "c2_longitudinal_drift_checker_calibration",
        "c2_longitudinal_lineage_design",
        "c2_longitudinal_recovery_design",
        "c3_intervention_target_contract",
        "long_horizon_effective_step_span",
        "task_step_effect_chain_integrity",
        "task_step_anti_padding_integrity",
        "trajectory_interaction_claim_boundary",
    ):
        assert checks[check_id]["status"] == "pass"


def test_removing_final_recovery_fails_c2_before_policy_calls() -> None:
    spec = _spec()
    plan = replace(
        spec.plan,
        opportunities=tuple(
            item
            for item in spec.plan.opportunities
            if item.opportunity_id != LONGITUDINAL_RECOVERY_OPPORTUNITY_ID
        ),
        sceu_units=tuple(
            item
            for item in spec.plan.sceu_units
            if item.opportunity_id != LONGITUDINAL_RECOVERY_OPPORTUNITY_ID
        ),
    )
    changed = replace(spec, plan=plan)
    payload = compute_experiment_design_audit({plan.episode_id: changed})

    assert payload["run_ready"] is False
    assert _checks(payload)["c2_longitudinal_recovery_design"]["status"] == "fail"


def test_removing_memory_reliant_target_fails_c3_before_policy_calls() -> None:
    spec = _spec()
    target = next(
        item
        for item in spec.plan.sceu_units
        if profile_sceu(spec.plan, item).memory_reliant_state_ids
    )
    plan = replace(
        spec.plan,
        sceu_units=tuple(
            replace(item, intervention_target_ids=())
            if item.sceu_id == target.sceu_id
            else item
            for item in spec.plan.sceu_units
        ),
    )
    changed = replace(spec, plan=plan)
    payload = compute_experiment_design_audit({plan.episode_id: changed})

    assert payload["run_ready"] is False
    assert _checks(payload)["c3_intervention_target_contract"]["status"] == "fail"


def test_every_longitudinal_gold_action_passes_programmatic_checker() -> None:
    spec = _spec()
    checker = SoftwareVerticalChecker(spec)

    for opportunity in spec.plan.opportunities:
        result = checker.check_action(
            opportunity.valid_action_ids[0],
            checkpoint_session=opportunity.checkpoint_session,
            opportunity_id=opportunity.opportunity_id,
        )
        assert result.is_correct, (opportunity.opportunity_id, result)


def test_longitudinal_release_requires_distinct_temporal_checkpoints() -> None:
    with pytest.raises(ValueError, match="at least 8 sessions"):
        SoftwareLongitudinalTrajectoryFamily.generate(42, n_sessions=7)
