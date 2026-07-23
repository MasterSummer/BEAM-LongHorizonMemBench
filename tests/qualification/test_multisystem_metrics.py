from __future__ import annotations

import pytest

from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    StateCheckpointMetricInput,
    _storage_evidence_mode,
    compute_failure_attribution_scorecard,
    compute_long_horizon_control_contrasts,
    compute_long_horizon_scorecard,
    compute_matched_construct_contrasts,
    compute_matched_construct_scorecard,
    compute_multisystem_metrics,
    compute_multisystem_metrics_by_cell,
    compute_multisystem_scorecard,
)


def _row(
    condition: str,
    readout: str,
    score: float,
    *,
    live_count: int | None = None,
) -> MultisystemMetricInput:
    return MultisystemMetricInput(
        policy_profile_id="policy-a",
        condition=condition,
        readout=readout,
        result_id=f"{condition}-{readout}",
        behavior_score=score,
        is_correct=score >= 0.5,
        required_state_ids=("C1",),
        stale_state_ids=("V1",),
        candidate_memory_state_ids=(("C1",), ("V1",)),
        retrieved_memory_state_ids=(("C1",), ("V1",)),
        visible_memory_state_ids=(("C1",),),
        candidate_shortfall=False,
        visible_memory_count=1,
        live_memory_count=live_count,
        causal_labels=("beneficial",),
        behaviorally_used_memory_ids=("memory-C1",),
        behavioral_use_probe_count=1,
        intervention_labels=("beneficial",),
        leave_one_out_count=1,
        leave_one_out_action_flips=1,
        drift_flags=("stale_state",) if score < 0.5 else (),
    )


def test_schema_v2_controlled_comparisons_are_separate_and_nullable() -> None:
    rows = (
        _row("workspace_only", "none", 0.2),
        _row("full_context", "none", 0.8),
        _row("oracle_current_state", "none", 1.0),
        _row("flat_retrieval", "common_rerank", 0.4),
        _row("mem0", "native", 0.6),
        _row("mem0", "common_rerank", 0.7),
        _row("amem", "native", 0.5),
        _row("amem", "common_rerank", 0.6),
        _row("memos", "native", 0.3),
        _row("memos", "common_rerank", 0.4),
    )
    metrics = compute_multisystem_metrics(rows)
    assert metrics["mem0_native_gain_beyond_workspace"].value == 0.4
    assert metrics["mem0_common_rerank_gain_beyond_workspace"].value == 0.5
    assert metrics["mem0_native_gain_over_flat"].value == 0.2
    assert metrics["mem0_native_gap_to_full_context"].value == 0.2
    assert metrics["mem0_native_oracle_gap_closed"].value == 0.5
    assert metrics["mem0_common_rerank_minus_native"].value == 0.1
    # Flat has no native readout and thus no fabricated native metric.
    assert metrics["flat_retrieval_native_gain_beyond_workspace"].value is None


def test_schema_v2_state_retrieval_and_memory_count_metrics() -> None:
    rows = (
        _row("workspace_only", "none", 0.2, live_count=0),
        _row("mem0", "native", 0.8, live_count=2),
    )
    metrics = compute_multisystem_metrics(
        rows,
        state_checkpoints=(
            StateCheckpointMetricInput(
                eligible_write_state_ids=("C1",),
                new_memory_state_ids=(("C1",),),
                current_state_ids=("C1",),
                future_needed_state_ids=("C1",),
                retired_state_ids=("V1",),
                live_memory_state_ids=(("C1",), ("V1",)),
                live_content_hashes=("h1", "h2"),
                n_write=1,
                n_live=2,
                mutation_counts=(("add", 1), ("update", 0)),
            ),
        ),
    )
    assert metrics["live_memory_count"].value == 2
    assert metrics["mutation_add_count"].value == 1
    assert metrics["stale_candidate_exposure"].value == 0.5
    assert metrics["stale_visible_exposure"].value == 0.0
    assert metrics["retrieval_to_visible_yield"].value == 0.5
    assert metrics["behavior_correct_rate_live_memory_count_2"].value == 1.0
    assert metrics["behavioral_use_probe_coverage"].value == 1.0
    assert metrics["probed_memory_causal_use_rate"].value == 1.0
    assert metrics["model_visible_behavioral_use_lower_bound"].value == 1.0


def test_cell_groups_and_scorecard_do_not_mix_native_and_common() -> None:
    rows = (
        _row("mem0", "native", 0.5),
        _row("mem0", "common_rerank", 0.7),
    )
    groups = compute_multisystem_metrics_by_cell(rows)
    assert {(item["condition"], item["readout"]) for item in groups} == {
        ("mem0", "native"),
        ("mem0", "common_rerank"),
    }
    scorecard = compute_multisystem_scorecard(rows)
    assert len(scorecard) == 2
    assert {item["readout"] for item in scorecard} == {"native", "common_rerank"}
    assert {item["status"] for item in scorecard} == {"complete"}
    assert {item["baseline_stability_rate"] for item in scorecard} == {1.0}
    assert {item["unstable_intervention_rate"] for item in scorecard} == {0.0}


def test_drift_rates_use_category_eligible_denominators() -> None:
    rows = (
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="mem0",
            readout="native",
            result_id="r",
            behavior_score=0.0,
            is_correct=False,
            drift_flags=("stale_state",),
            drift_eligible_categories=("stale_state",),
        ),
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="mem0",
            readout="native",
            result_id="r",
            behavior_score=1.0,
            is_correct=True,
            drift_eligible_categories=("constraint_loss",),
        ),
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="mem0",
            readout="native",
            result_id="r",
            behavior_score=1.0,
            is_correct=True,
            drift_eligible_categories=(),
        ),
    )
    metrics = compute_multisystem_metrics(rows)
    assert metrics["stale_state_action_rate"].value == 1.0
    assert metrics["stale_state_action_rate"].denominator == 1
    assert metrics["constraint_loss_rate"].value == 0.0
    assert metrics["constraint_loss_rate"].denominator == 1
    assert metrics["aggregate_drift_rate"].denominator == 2
    scorecard = compute_multisystem_scorecard(rows)[0]
    assert scorecard["stale_state_eligible_n"] == 1
    assert scorecard["aggregate_drift_eligible_n"] == 2


def test_off_target_drift_is_reported_without_inflating_targeted_rates() -> None:
    rows = (
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="oracle_current_state",
            readout="none",
            result_id="targeted",
            behavior_score=0.0,
            is_correct=False,
            drift_flags=("stale_state",),
            drift_eligible_categories=("stale_state",),
        ),
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="oracle_current_state",
            readout="none",
            result_id="off-target",
            behavior_score=0.0,
            is_correct=False,
            drift_flags=("constraint_loss",),
            drift_eligible_categories=("stale_state",),
        ),
    )

    metrics = compute_multisystem_metrics(rows)
    assert metrics["aggregate_drift_rate"].value == 0.5
    assert metrics["targeted_aggregate_drift_rate"].value == 0.5
    assert metrics["observed_aggregate_drift_rate"].value == 1.0
    assert metrics["canonical_drift_violation_rate"].value == 1.0
    assert metrics["off_target_drift_rate"].value == 0.5
    assert metrics["observed_constraint_loss_rate"].value == 0.5
    assert metrics["targeted_constraint_loss_rate"].value is None

    scorecard = compute_multisystem_scorecard(rows)[0]
    assert scorecard["targeted_aggregate_drift_rate"] == 0.5
    assert scorecard["observed_aggregate_drift_rate"] == 1.0
    assert scorecard["canonical_drift_violation_rate"] == 1.0
    assert scorecard["off_target_drift_rate"] == 0.5
    assert scorecard["off_target_drift_n"] == 1


def test_same_decision_failure_scorecard_uses_conditional_stage_denominators() -> None:
    rows = (
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="mem0",
            readout="native",
            result_id="storage-loss",
            behavior_score=0.0,
            is_correct=False,
            memory_reliant_state_ids=("C1", "P2"),
            stored_memory_state_ids=("C1",),
            storage_evidence_mode="inferred",
        ),
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="mem0",
            readout="native",
            result_id="causal-success",
            behavior_score=1.0,
            is_correct=True,
            memory_reliant_state_ids=("C1", "P2"),
            stored_memory_state_ids=("C1", "P2"),
            retrieved_memory_state_ids=(("C1",), ("P2",)),
            visible_memory_state_ids=(("C1",), ("P2",)),
            behaviorally_probed_state_ids=("C1",),
            behaviorally_used_state_ids=("C1",),
            storage_evidence_mode="inferred",
        ),
    )

    scorecard = compute_failure_attribution_scorecard(rows)[0]

    assert scorecard["attribution_applicable_n"] == 2
    assert scorecard["memory_required_storage_recall"] == 0.75
    assert scorecard["stored_to_retrieved_yield"] == pytest.approx(2 / 3)
    assert scorecard["retrieved_to_visible_yield"] == 1.0
    assert scorecard["visible_required_probe_coverage"] == 0.5
    assert scorecard["probed_required_causal_use_rate"] == 1.0
    assert scorecard["storage_failure_rate"] == 0.5
    assert scorecard["behavior_success_causal_rate"] == 0.5


def test_unobservable_storage_is_excluded_from_failure_denominator() -> None:
    row = MultisystemMetricInput(
        policy_profile_id="policy-a",
        condition="mem0",
        readout="native",
        result_id="unknown-storage",
        behavior_score=0.0,
        is_correct=False,
        memory_reliant_state_ids=("C1",),
        storage_evidence_mode="unavailable",
    )

    scorecard = compute_failure_attribution_scorecard((row,))[0]

    assert row.decision_attribution().stage == "storage_evidence_unavailable"
    assert scorecard["memory_reliant_n"] == 1
    assert scorecard["attribution_applicable_n"] == 0
    assert scorecard["storage_evidence_unavailable_n"] == 1
    assert scorecard["storage_failure_rate"] is None


def test_utilization_failures_are_subtyped_by_causal_use_evidence() -> None:
    common = {
        "policy_profile_id": "policy-a",
        "condition": "mem0",
        "readout": "native",
        "behavior_score": 0.0,
        "is_correct": False,
        "memory_reliant_state_ids": ("C1",),
        "stored_memory_state_ids": ("C1",),
        "retrieved_memory_state_ids": (("C1",),),
        "visible_memory_state_ids": (("C1",),),
        "behaviorally_probed_state_ids": ("C1",),
        "storage_evidence_mode": "native/exact",
    }
    rows = (
        MultisystemMetricInput(
            **common,
            result_id="no-effect",
        ),
        MultisystemMetricInput(
            **common,
            result_id="causal-wrong",
            behaviorally_used_state_ids=("C1",),
        ),
    )

    scorecard = compute_failure_attribution_scorecard(rows)[0]

    assert scorecard["utilization_failure_n"] == 2
    assert scorecard[
        "visible_without_detected_unique_causal_effect_n"
    ] == 1
    assert scorecard["visible_causally_influential_but_wrong_n"] == 1
    assert scorecard["visible_use_evidence_incomplete_n"] == 0


def test_checkpoint_evidence_can_prove_a_required_state_was_not_stored() -> None:
    assert _storage_evidence_mode(
        ("C1",),
        (),
        (),
        (),
        inventory_observed=True,
        checkpoint_evidence_mode="native/exact",
    ) == "native/exact"
    assert _storage_evidence_mode(
        ("C1",),
        (),
        (),
        ("C1",),
        inventory_observed=True,
        checkpoint_evidence_mode="native/exact",
    ) == "unavailable"


def test_failure_attribution_separates_backend_retrieval_from_model_exposure() -> None:
    reranker_filtered = MultisystemMetricInput(
        policy_profile_id="policy-a",
        condition="mem0",
        readout="common_rerank",
        result_id="filtered",
        behavior_score=0.0,
        is_correct=False,
        memory_reliant_state_ids=("C1",),
        stored_memory_state_ids=("C1",),
        backend_retrieved_memory_state_ids=(("C1",),),
        retrieved_memory_state_ids=(),
        visible_memory_state_ids=(),
        storage_evidence_mode="inferred",
    )
    backend_miss = MultisystemMetricInput(
        policy_profile_id="policy-a",
        condition="mem0",
        readout="common_rerank",
        result_id="backend-miss",
        behavior_score=0.0,
        is_correct=False,
        memory_reliant_state_ids=("C1",),
        stored_memory_state_ids=("C1",),
        candidate_memory_state_ids=(("C1",),),
        backend_retrieved_memory_state_ids=(),
        retrieved_memory_state_ids=(),
        visible_memory_state_ids=(),
        storage_evidence_mode="inferred",
    )

    assert reranker_filtered.decision_attribution().stage == "exposure_failure"
    assert backend_miss.decision_attribution().stage == "retrieval_failure"


def test_long_horizon_scorecard_keeps_construct_and_handoff_band_separate() -> None:
    rows = (
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="workspace_only",
            readout="none",
            result_id="early",
            behavior_score=1.0,
            is_correct=True,
            episode_id="episode-1",
            construct_kind="static_recall",
            horizon_band="short",
            handoff_count=2,
            oldest_required_state_age=2,
            dependency_depth=1,
            relevant_transition_count=0,
            drift_eligible_categories=(),
        ),
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition="workspace_only",
            readout="none",
            result_id="late",
            behavior_score=0.0,
            is_correct=False,
            episode_id="episode-1",
            construct_kind="state_evolution",
            horizon_band="long",
            handoff_count=10,
            oldest_required_state_age=10,
            latest_decision_event_distance=4,
            dependency_depth=3,
            relevant_transition_count=2,
            drift_flags=("stale_state",),
            drift_eligible_categories=("stale_state",),
        ),
    )

    scorecard = compute_long_horizon_scorecard(rows)

    assert len(scorecard) == 2
    late = next(row for row in scorecard if row["horizon_band"] == "long")
    assert late["construct_kind"] == "state_evolution"
    assert late["mean_handoff_count"] == 10.0
    assert late["mean_relevant_transition_count"] == 2.0
    assert late["targeted_drift_rate"] == 1.0


def test_long_horizon_control_contrasts_are_paired_on_the_same_decision() -> None:
    common = {
        "policy_profile_id": "policy-a",
        "episode_id": "episode-1",
        "opportunity_id": "opp-late",
        "construct_kind": "state_evolution",
        "horizon_band": "long",
        "handoff_count": 10,
        "drift_eligible_categories": ("stale_state",),
    }
    rows = (
        MultisystemMetricInput(
            **common,
            condition="workspace_only",
            readout="none",
            result_id="workspace",
            behavior_score=0.0,
            is_correct=False,
            drift_flags=("stale_state",),
        ),
        MultisystemMetricInput(
            **common,
            condition="oracle_current_state",
            readout="none",
            result_id="oracle",
            behavior_score=1.0,
            is_correct=True,
        ),
        MultisystemMetricInput(
            **common,
            condition="mem0",
            readout="native",
            result_id="mem0",
            behavior_score=0.75,
            is_correct=True,
        ),
        # This unmatched decision must not enter a cross-SCEU comparison.
        MultisystemMetricInput(
            **{**common, "opportunity_id": "opp-other"},
            condition="mem0",
            readout="native",
            result_id="unmatched",
            behavior_score=1.0,
            is_correct=True,
        ),
    )

    contrast = compute_long_horizon_control_contrasts(rows)[0]

    assert contrast["n_matched_decisions"] == 1
    assert contrast["mean_behavior_gain_beyond_workspace"] == 0.75
    assert contrast["mean_behavior_gap_to_oracle"] == 0.25
    assert contrast["oracle_gap_closed"] == 0.75
    assert contrast["targeted_drift_risk_difference_vs_workspace"] == -1.0
    assert contrast["targeted_drift_risk_difference_vs_oracle"] == 0.0


def test_matched_construct_scorecard_is_cross_episode_but_same_decision() -> None:
    common = {
        "policy_profile_id": "policy-a",
        "condition": "mem0",
        "readout": "native",
        "opportunity_id": "opp-matched-terminal",
        "counterfactual_group_id": "group-1",
        "counterfactual_terminal_archetype": "current_v2_offline",
        "is_counterfactual_target": True,
        "memory_reliant_state_ids": ("C1",),
        "drift_eligible_categories": ("plan_deviation",),
        "storage_evidence_mode": "native/exact",
    }
    rows = (
        MultisystemMetricInput(
            **common,
            episode_id="episode-static",
            counterfactual_variant="static",
            result_id="static",
            behavior_score=0.9,
            is_correct=True,
            stored_memory_state_ids=("C1",),
            backend_retrieved_memory_state_ids=(("C1",),),
            visible_memory_state_ids=(("C1",),),
            behaviorally_probed_state_ids=("C1",),
            behaviorally_used_state_ids=("C1",),
        ),
        MultisystemMetricInput(
            **common,
            episode_id="episode-evolution",
            counterfactual_variant="evolution",
            result_id="evolution",
            behavior_score=0.4,
            is_correct=False,
            drift_flags=("plan_deviation",),
        ),
        MultisystemMetricInput(
            **common,
            episode_id="episode-conflict",
            counterfactual_variant="hierarchical_conflict",
            result_id="conflict",
            behavior_score=0.2,
            is_correct=False,
            stored_memory_state_ids=("C1",),
            backend_retrieved_memory_state_ids=(("C1",),),
            visible_memory_state_ids=(("C1",),),
            behaviorally_probed_state_ids=("C1",),
            drift_flags=("plan_deviation",),
        ),
    )

    contrast = compute_matched_construct_contrasts(rows)[0]
    assert contrast["complete"] is True
    assert contrast["terminal_archetype"] == "current_v2_offline"
    assert contrast["state_evolution_penalty_vs_static"] == 0.5
    assert contrast["hierarchical_conflict_penalty_vs_static"] == 0.7
    assert contrast["state_evolution_drift_excess_vs_static"] == 1.0
    assert (
        contrast["state_evolution_drift_violation_excess_vs_static"]
        == 1.0
    )
    assert contrast["evolution_attribution_stage_changed"] is True
    assert contrast["hierarchical_conflict_attribution_stage_changed"] is True

    scorecard = compute_matched_construct_scorecard(rows)[0]
    assert scorecard["n_counterfactual_groups"] == 1
    assert scorecard["n_complete_groups"] == 1
    assert scorecard["n_current_v2_offline_groups"] == 1
    assert scorecard["all_terminal_archetypes_covered"] is False
    assert scorecard["mean_state_evolution_penalty_vs_static"] == 0.5
    assert (
        scorecard["mean_state_evolution_drift_violation_excess_vs_static"]
        == 1.0
    )


def test_matched_construct_reports_workspace_adjusted_difference_in_differences() -> None:
    rows = tuple(
        MultisystemMetricInput(
            policy_profile_id="policy-a",
            condition=condition,
            readout=readout,
            opportunity_id="opp-matched-terminal",
            counterfactual_group_id="group-1",
            counterfactual_terminal_archetype="current_v2_offline",
            is_counterfactual_target=True,
            episode_id=f"episode-{variant}",
            counterfactual_variant=variant,
            result_id=f"{condition}-{variant}",
            behavior_score=score,
            is_correct=score >= 0.5,
        )
        for condition, readout, scores in (
            ("workspace_only", "none", (0.8, 0.5, 0.4)),
            ("mem0", "native", (0.9, 0.4, 0.2)),
        )
        for variant, score in zip(
            ("static", "evolution", "hierarchical_conflict"),
            scores,
            strict=True,
        )
    )

    contrasts = compute_matched_construct_contrasts(rows)
    mem0 = next(row for row in contrasts if row["condition"] == "mem0")

    assert mem0["workspace_matched_control_available"] is True
    assert mem0["static_gain_beyond_workspace"] == pytest.approx(0.1)
    assert mem0["evolution_gain_beyond_workspace"] == pytest.approx(-0.1)
    assert mem0["hierarchical_conflict_gain_beyond_workspace"] == pytest.approx(-0.2)
    assert mem0["state_evolution_penalty_excess_over_workspace"] == pytest.approx(0.2)
    assert mem0[
        "hierarchical_conflict_penalty_excess_over_workspace"
    ] == pytest.approx(0.3)

    scorecard = next(
        row
        for row in compute_matched_construct_scorecard(rows)
        if row["condition"] == "mem0"
    )
    assert scorecard[
        "mean_state_evolution_penalty_excess_over_workspace"
    ] == pytest.approx(0.2)
