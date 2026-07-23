from __future__ import annotations

from lhmsb.qualification.contribution_evidence import (
    build_contribution_evidence,
    contribution_evidence_markdown,
)
from lhmsb.qualification.design_audit import build_analysis_contract
from lhmsb.qualification.horizon_panel import HORIZON_PRIMARY_ESTIMANDS
from lhmsb.qualification.statistics import MATCHED_PRIMARY_ESTIMANDS

_BASE_GATES = (
    "task_completion",
    "oracle_accuracy",
    "oracle_accuracy_by_opportunity",
    "oracle_accuracy_by_scenario",
    "current_action_state_contract_completeness",
    "workspace_oracle_action_separation",
    "effective_long_horizon_step_threshold",
    "task_step_causal_linkage",
    "task_step_effect_chain_integrity",
    "task_step_anti_padding_integrity",
    "matched_construct_structural_invariance",
    "matched_construct_outcome_completeness",
    "matched_workspace_adjustment_available",
    "matched_oracle_terminal_contract_solvability",
    "matched_full_context_terminal_contract_solvability",
    "matched_workspace_recoverability_balance",
    "matched_workspace_oracle_action_separation",
    "matched_gold_action_balance",
    "action_dominance",
    "option_dominance",
    "drift_category_exposure",
    "drift_action_calibration",
    "workspace_oracle_drift_separation",
    "decision_failure_attribution_completeness",
    "decision_storage_evidence_availability",
    "causal_use_evidence_consistency",
    "lifecycle_provenance_complete",
    "semantic_attribution_complete",
    "semantic_attribution_resolvability",
    "stored_object_provenance_complete",
    "flat_causal_probe_coverage",
    "stored_retrieved_visible_behavior_chain",
)


def _gates(*, longitudinal: str = "not_applicable") -> dict[str, object]:
    rows = [
        {"gate_id": gate_id, "status": "pass"}
        for gate_id in _BASE_GATES
    ]
    rows.extend(
        {
            "gate_id": gate_id,
            "status": longitudinal,
        }
        for gate_id in (
            "longitudinal_drift_state_lineage_coverage",
            "longitudinal_drift_repeated_checkpoint_coverage",
            "longitudinal_drift_adherence_anchor_coverage",
            "longitudinal_drift_control_cleanliness",
        )
    )
    return {
        "measurement_ready": longitudinal in {"pass", "not_applicable"},
        "gates": rows,
    }


def _summary() -> dict[str, object]:
    return {
        "construct_mode": "matched_triplets",
        "primary_analysis_unit": "counterfactual_group",
        "n_long_horizon_control_contrasts": 21,
        "n_matched_construct_contrasts": 21,
        "n_counterfactual_groups": 3,
    }


def _longitudinal_summary() -> dict[str, object]:
    return {
        "construct_mode": "longitudinal_trajectories",
        "primary_analysis_unit": "episode",
        "n_long_horizon_control_contrasts": 35,
        "n_matched_construct_contrasts": 0,
        "n_counterfactual_groups": 0,
    }


def _statistics(*, include_workspace_adjustment: bool = True) -> dict[str, object]:
    metrics = ["state_evolution_penalty_vs_static"]
    if include_workspace_adjustment:
        metrics.extend(
            [
                "state_evolution_penalty_excess_over_workspace",
                "hierarchical_conflict_penalty_excess_over_workspace",
            ]
        )
    return {
        "estimates": [
            {
                "metric": metric,
                "condition": "mem0",
            }
            for metric in metrics
        ]
    }


def _design_audit(
    *,
    ready: bool = True,
    horizon: bool = False,
    longitudinal: bool = False,
) -> dict[str, object]:
    c2_checks = (
        (
            "c2_longitudinal_drift_checker_calibration",
            "c2_longitudinal_lineage_design",
            "c2_longitudinal_recovery_design",
        )
        if longitudinal
        else ("matched_drift_checker_calibration",)
    )
    return {
        "audit_status": "ready_for_calibration" if ready else "diagnostic_only",
        "balanced_mechanism_design_ready": ready and not longitudinal,
        "trajectory_interaction_mode_counts": {
            "replay_backed_critical_decision": 1 if longitudinal else 9,
        },
        "online_long_horizon_agent_execution_supported": False,
        "checks": [
            {"check_id": check_id, "status": "pass"}
            for check_id in (*c2_checks, "c3_intervention_target_contract")
        ],
        "analysis_contract": build_analysis_contract(
            matched=not longitudinal,
            horizon=horizon,
            longitudinal=longitudinal,
        ),
    }


def _fault_profile(*, outcome_equivalent_pairs: int = 1) -> dict[str, object]:
    return {
        "n_aligned_decision_pairs": outcome_equivalent_pairs,
        "n_outcome_equivalent_pairs": outcome_equivalent_pairs,
    }


def test_contribution_evidence_separates_endpoint_and_longitudinal_claims() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": [{"drift_category": "stale_state"}]},
        decision_attribution_rows=({"stage": "retrieval_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit=_design_audit(),
    )
    rows = {
        row["contribution_id"]: row
        for row in payload["contributions"]  # type: ignore[index]
    }

    assert rows["C1"]["evidence_status"] == "ready"
    assert rows["C1"]["claim_scope"] == "matched_workspace_adjusted_mechanism"
    assert rows["C1"]["evidence_counts"][  # type: ignore[index]
        "trajectory_interaction_mode_counts"
    ] == {"replay_backed_critical_decision": 9}
    assert "did every prefix step online" not in rows["C1"]["claim_boundary"]
    assert "executed every prefix step online" in rows["C1"]["claim_boundary"]
    assert rows["C2"]["evidence_status"] == "ready"
    assert rows["C2"]["claim_scope"] == "endpoint_violation_only"
    assert rows["C3"]["evidence_status"] == "ready"
    assert rows["C3"]["observed_failure_stages"] == ["retrieval_failure"]
    assert payload["analysis_timing"] == "pre_specified"
    assert payload["confirmatory_timing_eligible"] is True
    assert "does not mean" in contribution_evidence_markdown(payload)


def test_post_hoc_contribution_audit_cannot_claim_confirmatory_timing() -> None:
    payload = build_contribution_evidence(
        summary={**_summary(), "analysis_timing": "post_hoc_scope_audit"},
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=(),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(),
    )

    assert payload["confirmatory_timing_eligible"] is False
    assert "post-hoc" in payload["interpretation"]
    assert all(
        row["claim_timing"] == "post_hoc_scope_audit"
        for row in payload["contributions"]  # type: ignore[index]
    )


def test_horizon_contribution_uses_panel_estimands_and_claim_boundary() -> None:
    summary = {
        **_summary(),
        "construct_mode": "horizon_panels",
        "primary_analysis_unit": "horizon_panel",
        "n_horizon_panel_contrasts": 21,
        "n_horizon_panels": 3,
    }
    horizon_statistics = {
        "estimates": [
            {"metric": metric, "condition": "mem0"}
            for metric in HORIZON_PRIMARY_ESTIMANDS
        ]
    }

    payload = build_contribution_evidence(
        summary=summary,
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        horizon_statistics=horizon_statistics,
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=(),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(horizon=True),
    )
    c1 = payload["contributions"][0]  # type: ignore[index]

    assert c1["evidence_status"] == "ready"
    assert c1["claim_scope"] == (
        "same_decision_horizon_amplification_beyond_workspace"
    )
    assert c1["required_estimands"] == list(HORIZON_PRIMARY_ESTIMANDS)
    assert c1["evidence_counts"]["horizon_panels"] == 3
    assert "not a pure handoff effect" in c1["claim_boundary"]


def test_contribution_evidence_requires_workspace_adjusted_estimands() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(),
        matched_statistics=_statistics(include_workspace_adjustment=False),
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=(),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(),
    )
    c1 = payload["contributions"][0]  # type: ignore[index]

    assert c1["evidence_status"] == "not_ready"  # type: ignore[index]
    assert c1["missing_estimands"] == list(  # type: ignore[index]
        MATCHED_PRIMARY_ESTIMANDS
    )


def test_c1_requires_matched_full_context_and_oracle_controls() -> None:
    gates = _gates()
    for row in gates["gates"]:  # type: ignore[index]
        if row["gate_id"] == "matched_full_context_terminal_contract_solvability":
            row["status"] = "fail"

    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=gates,
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=(),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(),
    )
    c1 = payload["contributions"][0]  # type: ignore[index]

    assert c1["evidence_status"] == "not_ready"
    assert c1["gate_statuses"][  # type: ignore[index]
        "matched_full_context_terminal_contract_solvability"
    ] == "fail"


def test_c1_requires_a_complete_current_action_state_contract() -> None:
    gates = _gates()
    for row in gates["gates"]:  # type: ignore[index]
        if row["gate_id"] == "current_action_state_contract_completeness":
            row["status"] = "fail"

    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=gates,
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=(),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(),
    )
    c1 = payload["contributions"][0]  # type: ignore[index]

    assert c1["evidence_status"] == "not_ready"
    assert c1["gate_statuses"][  # type: ignore[index]
        "current_action_state_contract_completeness"
    ] == "fail"


def test_contribution_evidence_requires_adherence_gates_for_longitudinal_scope() -> None:
    payload = build_contribution_evidence(
        summary=_longitudinal_summary(),
        measurement_gates=_gates(longitudinal="pass"),
        matched_statistics=_statistics(),
        drift_trajectories={
            "trajectories": [
                {
                    "drift_evaluable": True,
                    "lineage_backed": True,
                    "event_observed": True,
                }
            ]
        },
        decision_attribution_rows=({"stage": "utilization_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit=_design_audit(longitudinal=True),
    )
    c2 = payload["contributions"][1]  # type: ignore[index]

    assert c2["evidence_status"] == "ready"  # type: ignore[index]
    assert c2["claim_scope"] == (  # type: ignore[index]
        "state_lineage_longitudinal_drift"
    )
    assert "adherence_anchored_onset" in c2["observed_estimands"]  # type: ignore[operator,index]
    assert c2["evidence_counts"][  # type: ignore[index]
        "lineage_backed_evaluable_trajectories"
    ] == 1


def test_matched_endpoint_contract_cannot_be_upgraded_to_longitudinal_post_run() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(longitudinal="pass"),
        matched_statistics=_statistics(),
        drift_trajectories={
            "trajectories": [
                {
                    "drift_evaluable": True,
                    "lineage_backed": True,
                    "event_observed": True,
                }
            ]
        },
        decision_attribution_rows=({"stage": "utilization_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit=_design_audit(),
    )
    c2 = payload["contributions"][1]  # type: ignore[index]

    assert c2["evidence_status"] == "not_ready"  # type: ignore[index]
    assert c2["missing_design_requirements"] == [  # type: ignore[index]
        "C2_pre_call_longitudinal_scope"
    ]


def test_category_only_legacy_trajectory_cannot_support_c2() -> None:
    payload = build_contribution_evidence(
        summary=_longitudinal_summary(),
        measurement_gates=_gates(longitudinal="pass"),
        matched_statistics=_statistics(),
        drift_trajectories={
            "trajectories": [
                {
                    "drift_evaluable": True,
                    "lineage_backed": False,
                    "event_observed": True,
                }
            ]
        },
        decision_attribution_rows=({"stage": "utilization_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit=_design_audit(longitudinal=True),
    )
    c2 = payload["contributions"][1]  # type: ignore[index]

    assert c2["evidence_status"] == "not_ready"  # type: ignore[index]
    assert c2["missing_estimands"] == [  # type: ignore[index]
        "state_lineage_anchoring",
        "category_only_legacy_trajectories_excluded",
    ]
    assert c2["observed_estimands"] == [  # type: ignore[index]
        "drift_compatible_endpoint_violation"
    ]


def test_contribution_evidence_requires_balanced_matched_design() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": [{"drift_evaluable": False}]},
        decision_attribution_rows=({"stage": "retrieval_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit=_design_audit(ready=False),
    )
    c1 = payload["contributions"][0]  # type: ignore[index]

    assert c1["evidence_status"] == "not_ready"  # type: ignore[index]
    assert c1["missing_design_requirements"] == [  # type: ignore[index]
        "balanced_mechanism_design_ready"
    ]
    markdown = contribution_evidence_markdown(payload)
    assert "item:balanced_mechanism_design_ready" in markdown


def test_c3_requires_an_outcome_equivalent_fault_profile_comparison() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": []},
        decision_attribution_rows=({"stage": "retrieval_failure"},),
        fault_profile_divergence=_fault_profile(outcome_equivalent_pairs=0),
        experiment_design_audit=_design_audit(),
    )
    c3 = payload["contributions"][2]  # type: ignore[index]

    assert c3["evidence_status"] == "not_ready"
    assert c3["missing_estimands"] == [
        "outcome_equivalent_fault_profile_divergence"
    ]


def test_pre_specified_claims_require_contribution_specific_pre_call_contracts() -> None:
    payload = build_contribution_evidence(
        summary=_summary(),
        measurement_gates=_gates(),
        matched_statistics=_statistics(),
        drift_trajectories={"trajectories": [{"drift_evaluable": False}]},
        decision_attribution_rows=({"stage": "retrieval_failure"},),
        fault_profile_divergence=_fault_profile(),
        experiment_design_audit={
            **_design_audit(),
            "analysis_contract": {
                "status": "pre_call_frozen",
                "claim_id": "C1",
            },
        },
    )
    rows = {
        row["contribution_id"]: row
        for row in payload["contributions"]  # type: ignore[index]
    }

    assert payload["confirmatory_timing_eligible"] is False
    assert payload["pre_call_contract_statuses"] == {
        "C1": "missing",
        "C2": "missing",
        "C3": "missing",
    }
    assert rows["C1"]["evidence_status"] == "not_ready"
    assert rows["C2"]["evidence_status"] == "not_ready"
    assert rows["C3"]["evidence_status"] == "not_ready"
    assert rows["C3"]["missing_design_requirements"] == [
        "C3_pre_call_analysis_contract"
    ]
