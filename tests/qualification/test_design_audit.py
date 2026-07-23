from __future__ import annotations

from dataclasses import replace

from lhmsb.families.software.horizon_panel import SoftwareHorizonPanelFamily
from lhmsb.families.software.matched_constructs import (
    SoftwareMatchedConstructFamily,
)
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.qualification.design_audit import (
    build_analysis_contract,
    compute_experiment_design_audit,
    experiment_design_audit_markdown,
)
from lhmsb.qualification.horizon_panel import HORIZON_PRIMARY_ESTIMANDS
from lhmsb.qualification.statistics import MATCHED_PRIMARY_ESTIMANDS


def _matched_specs(*seeds: int) -> dict[str, SoftwareMem0VerticalSpec]:
    specs = tuple(
        spec
        for seed in seeds
        for spec in SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
            steps_per_session=16,
        )
    )
    return {spec.plan.episode_id: spec for spec in specs}


def _horizon_specs(*seeds: int) -> dict[str, SoftwareMem0VerticalSpec]:
    specs = tuple(
        spec
        for seed in seeds
        for spec in SoftwareHorizonPanelFamily.generate_panel(
            seed,
            trajectory_seed=seed,
        )
    )
    return {spec.plan.episode_id: spec for spec in specs}


def test_three_group_matched_release_is_audited_before_policy_calls() -> None:
    payload = compute_experiment_design_audit(_matched_specs(42, 43, 44))
    checks = {row["check_id"]: row for row in payload["checks"]}  # type: ignore[index]

    assert payload["run_ready"] is True
    assert payload["balanced_mechanism_design_ready"] is True
    assert payload["audit_status"] == "ready_for_calibration"
    assert payload["analysis_contract"] == build_analysis_contract(matched=True)
    contract = payload["analysis_contract"]
    assert contract["status"] == "pre_call_frozen"  # type: ignore[index]
    assert contract["primary_estimands"] == list(  # type: ignore[index]
        MATCHED_PRIMARY_ESTIMANDS
    )
    assert contract["analysis_unit"] == "counterfactual_group"  # type: ignore[index]
    assert contract["drift_scope"] == "endpoint_violation_only"  # type: ignore[index]
    assert contract["history_availability_control"] == "full_context"  # type: ignore[index]
    assert contract["terminal_solvability_control"] == (  # type: ignore[index]
        "oracle_current_state"
    )
    assert contract["trajectory_interaction_mode"] == (  # type: ignore[index]
        "replay_backed_critical_decision"
    )
    assert contract["online_long_horizon_agent_execution_claim"] is False  # type: ignore[index]
    assert set(contract["contribution_contracts"]) == {"C1", "C2", "C3"}  # type: ignore[arg-type,index]
    assert contract["contribution_contracts"]["C2"]["claim_scope"] == (  # type: ignore[index]
        "endpoint_violation_only"
    )
    assert contract["contribution_contracts"]["C3"]["primary_intervention"] == (  # type: ignore[index]
        "repeat_stable_neutral_replacement"
    )
    assert payload["trajectory_interaction_mode_counts"] == {
        "replay_backed_critical_decision": 9,
    }
    assert payload["online_long_horizon_agent_execution_supported"] is False
    assert checks["matched_workspace_recoverability_balance"]["status"] == "pass"
    assert checks["matched_drift_checker_calibration"]["status"] == "pass"
    assert checks["long_horizon_effective_step_span"]["status"] == "pass"
    assert checks["current_action_state_contract_complete"]["status"] == "pass"
    assert checks["task_step_anti_padding_integrity"]["status"] == "pass"
    assert checks["trajectory_interaction_claim_boundary"]["status"] == "pass"
    markdown = experiment_design_audit_markdown(payload)
    assert "ready_for_calibration" in markdown
    assert "not a positive" in markdown
    assert "Primary estimands" in markdown
    assert "endpoint violation only" in markdown
    assert "Online multi-hundred-step policy execution claimed: **false**" in markdown


def test_one_group_matched_release_remains_diagnostic_only() -> None:
    payload = compute_experiment_design_audit(_matched_specs(42))
    checks = {row["check_id"]: row for row in payload["checks"]}  # type: ignore[index]

    assert payload["run_ready"] is True
    assert payload["balanced_mechanism_design_ready"] is False
    assert payload["audit_status"] == "diagnostic_only"
    assert checks["matched_gold_action_balance"]["status"] == "not_applicable"


def test_unbalanced_recoverability_fails_before_policy_calls() -> None:
    original = _matched_specs(42, 43, 44)
    modified = {}
    for episode_id, spec in original.items():
        metadata = tuple(
            (
                key,
                "absent" if key == "recoverability_variant" else child,
            )
            for key, child in spec.plan.metadata
        )
        changed = replace(
            spec,
            plan=replace(spec.plan, metadata=metadata),
        )
        modified[episode_id] = changed

    payload = compute_experiment_design_audit(modified)

    assert payload["run_ready"] is False
    assert payload["audit_status"] == "invalid"
    assert "matched_workspace_recoverability_balance" in payload[
        "failed_check_ids"
    ]


def test_incomplete_current_action_state_contract_fails_before_policy_calls() -> None:
    original = _matched_specs(42, 43, 44)
    first_episode_id = sorted(original)[0]
    first = original[first_episode_id]
    target = first.plan.opportunities[0]
    incomplete = replace(target, focal_state_ids=())
    changed = replace(
        first,
        plan=replace(
            first.plan,
            opportunities=(
                incomplete,
                *first.plan.opportunities[1:],
            ),
        ),
    )
    modified = {**original, first_episode_id: changed}

    payload = compute_experiment_design_audit(modified)
    checks = {row["check_id"]: row for row in payload["checks"]}  # type: ignore[index]

    assert payload["run_ready"] is False
    assert checks["current_action_state_contract_complete"]["status"] == "fail"
    assert checks["current_action_state_contract_complete"]["detail"]


def test_standard_release_freezes_nonmatched_contribution_contracts() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)

    payload = compute_experiment_design_audit({spec.plan.episode_id: spec})

    assert payload["scope"] == "standard_trajectory"
    assert payload["analysis_unit"] == "episode"
    assert payload["analysis_contract"] == build_analysis_contract(matched=False)
    assert payload["analysis_contract"]["status"] == "pre_call_frozen"  # type: ignore[index]
    assert payload["analysis_contract"]["analysis_unit"] == "episode"  # type: ignore[index]
    assert payload["analysis_contract"]["contribution_contracts"]["C2"][  # type: ignore[index]
        "claim_scope"
    ] == "state_lineage_longitudinal_if_coverage_gates_pass"
    markdown = experiment_design_audit_markdown(payload)
    assert "Without matched static/evolution/conflict histories" in markdown


def test_horizon_release_freezes_panel_level_diagnostic_contract() -> None:
    payload = compute_experiment_design_audit(_horizon_specs(42))
    checks = {row["check_id"]: row for row in payload["checks"]}  # type: ignore[index]

    assert payload["scope"] == "horizon_dose_diagnostic"
    assert payload["analysis_unit"] == "horizon_panel"
    assert payload["physical_episode_count"] == 9
    assert payload["counterfactual_group_count"] == 3
    assert payload["horizon_panel_count"] == 1
    assert payload["run_ready"] is True
    assert payload["balanced_mechanism_design_ready"] is False
    assert payload["audit_status"] == "diagnostic_only"
    contract = payload["analysis_contract"]
    assert contract == build_analysis_contract(matched=True, horizon=True)
    assert contract["analysis_unit"] == "horizon_panel"  # type: ignore[index]
    assert contract["primary_estimands"] == list(  # type: ignore[index]
        HORIZON_PRIMARY_ESTIMANDS
    )
    assert contract["contribution_contracts"]["C1"]["claim_scope"] == (  # type: ignore[index]
        "same_decision_horizon_amplification_beyond_workspace"
    )
    assert checks["long_horizon_effective_step_span"]["status"] == "pass"
    assert checks["horizon_panel_structural_invariance"]["status"] == "pass"
    assert checks["horizon_joint_dose_monotonic"]["status"] == "pass"
    assert checks["horizon_long_only_step_threshold"]["status"] == "pass"
    assert checks["task_step_anti_padding_integrity"]["status"] == "pass"
    assert checks["trajectory_interaction_claim_boundary"]["status"] == "pass"


def test_three_horizon_panels_are_three_not_twenty_seven_units() -> None:
    payload = compute_experiment_design_audit(_horizon_specs(42, 43, 44))

    assert payload["physical_episode_count"] == 27
    assert payload["counterfactual_group_count"] == 9
    assert payload["horizon_panel_count"] == 3
    assert payload["balanced_mechanism_design_ready"] is True
    assert payload["audit_status"] == "ready_for_calibration"
