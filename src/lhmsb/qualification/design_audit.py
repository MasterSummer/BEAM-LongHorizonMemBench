"""Policy-free audit of the experimental design before model calls.

This audit is deliberately separate from post-run measurement readiness.  It
checks whether a frozen release can identify the declared matched mechanism at
all, before native writers or continuation policies consume paid API calls.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

from lhmsb.families.software.horizon_panel import (
    HorizonDose,
    HorizonPanelAudit,
    audit_horizon_panel,
)
from lhmsb.families.software.matched_constructs import (
    audit_matched_construct_triplet,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.drift import (
    CANONICAL_DRIFT_CATEGORIES,
    drift_eligible_categories,
    drift_lineage_pairs,
)
from lhmsb.qualification.horizon_panel import (
    HORIZON_PRIMARY_ESTIMANDS,
    HORIZON_SECONDARY_ESTIMANDS,
)
from lhmsb.qualification.readiness import (
    compute_drift_action_calibration,
    compute_heuristic_baselines,
)
from lhmsb.qualification.statistics import (
    MATCHED_DRIFT_SCOPE,
    MATCHED_MULTIPLICITY_SCOPE,
    MATCHED_PAIRED_TEST,
    MATCHED_PRIMARY_ANALYSIS_UNIT,
    MATCHED_PRIMARY_EFFECT_DIRECTION,
    MATCHED_PRIMARY_ESTIMANDS,
    MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT,
    MATCHED_SECONDARY_ESTIMANDS,
)

EXPERIMENT_DESIGN_AUDIT_SCHEMA_VERSION = 6
EXPERIMENT_DESIGN_CHECK_IDS = (
    "matched_release_membership",
    "longitudinal_release_membership",
    "matched_triplet_structural_invariance",
    "matched_gold_action_balance",
    "matched_action_shortcut_resistance",
    "matched_option_shortcut_resistance",
    "matched_workspace_recoverability_balance",
    "matched_terminal_archetype_balance",
    "matched_drift_checker_calibration",
    "current_requirements_exclude_future_state",
    "current_action_state_contract_complete",
    "workspace_absent_memory_reliant_decisions_present",
    "c2_longitudinal_drift_checker_calibration",
    "c2_longitudinal_lineage_design",
    "c2_longitudinal_recovery_design",
    "c3_intervention_target_contract",
    "long_horizon_effective_step_span",
    "task_step_effect_chain_integrity",
    "task_step_anti_padding_integrity",
    "trajectory_interaction_claim_boundary",
    "horizon_panel_membership",
    "horizon_panel_structural_invariance",
    "horizon_levels_complete",
    "horizon_joint_dose_monotonic",
    "horizon_long_only_step_threshold",
)

_EXPECTED_GOLD_ACTIONS = {
    "cloud_shortcut",
    "safe_v2_offline",
    "stale_v1",
}
_EXPECTED_RECOVERABILITY = {"absent", "derivable", "explicit"}
_EXPECTED_TERMINAL_ARCHETYPES = {
    "authorized_cloud",
    "current_v1_offline",
    "current_v2_offline",
}


def build_contribution_analysis_contracts(
    *,
    matched: bool,
    horizon: bool = False,
    longitudinal: bool = False,
) -> dict[str, dict[str, object]]:
    """Freeze the allowed analysis scope for C1--C3 before model calls."""

    if horizon and not matched:
        raise ValueError("a horizon contribution contract requires matched constructs")
    if longitudinal and matched:
        raise ValueError("a longitudinal contribution contract cannot be matched")
    c1_scope = (
        "same_decision_horizon_amplification_beyond_workspace"
        if horizon
        else (
            "matched_workspace_adjusted_mechanism"
            if matched
            else "paired_value_beyond_workspace"
        )
    )
    c2_scope = (
        "endpoint_violation_only"
        if matched
        else (
            "state_lineage_longitudinal"
            if longitudinal
            else "state_lineage_longitudinal_if_coverage_gates_pass"
        )
    )
    c2_estimands = (
        [
            "category_eligible_drift_compatible_violation_rate",
            "matched_endpoint_violation_excess",
        ]
        if matched
        else [
            "state_lineage_anchoring",
            "adherence_anchored_onset",
            "drift_free_survival",
            "persistence",
            "recovery",
        ]
    )
    return {
        "C1": {
            "status": "pre_call_frozen",
            "claim_scope": c1_scope,
            "analysis_unit": (
                "horizon_panel"
                if horizon
                else ("counterfactual_group" if matched else "episode")
            ),
            "workspace_control": "workspace_only",
            "history_availability_control": "full_context",
            "terminal_solvability_control": "oracle_current_state",
            "claim_boundary": (
                "Positive memory-channel or horizon effects require the frozen "
                "paired estimands, uncertainty, and clean control gates; design "
                "readiness alone is not an empirical result."
            ),
        },
        "C2": {
            "status": "pre_call_frozen",
            "claim_scope": c2_scope,
            "analysis_unit": (
                "counterfactual_group_endpoint"
                if matched
                else "episode_state_lineage"
            ),
            "required_estimands": c2_estimands,
            "required_controls": [
                "oracle_current_state",
                "full_context",
            ],
            "required_design_checks": (
                [
                    "c2_longitudinal_drift_checker_calibration",
                    "c2_longitudinal_lineage_design",
                    "c2_longitudinal_recovery_design",
                ]
                if longitudinal
                else ["matched_drift_checker_calibration"]
                if matched
                else []
            ),
            "claim_boundary": (
                (
                    "A matched single endpoint supports only drift-compatible "
                    "violation excess, not longitudinal onset."
                )
                if matched
                else (
                    "The release can estimate longitudinal onset, persistence, "
                    "and recovery only for same-lineage risk sets that retain an "
                    "observed prior-adherence anchor and clean oracle/full-context "
                    "trajectories. Design coverage alone is not an observed event."
                )
            ),
        },
        "C3": {
            "status": "pre_call_frozen",
            "claim_scope": (
                "earliest_supported_stage_and_unique_causal_effect_lower_bound"
            ),
            "analysis_unit": "state_conditioned_evaluation_unit",
            "required_estimands": [
                "conditional_stage_yields",
                "earliest_supported_failure_stage",
                "unique_causal_effect_lower_bound",
                "outcome_equivalent_fault_profile_divergence",
            ],
            "required_trace_order": [
                "stored",
                "backend_retrieved",
                "model_visible",
                "intervention_evidence",
                "checked_behavior",
            ],
            "primary_intervention": "repeat_stable_neutral_replacement",
            "storage_provenance_tracks": ["native/exact", "inferred"],
            "required_design_checks": ["c3_intervention_target_contract"],
            "claim_boundary": (
                "The intervention detects unique observable causal influence, "
                "not internal reasoning. No detected effect does not exclude "
                "redundant or compensated use, and outcome-equivalent pairs are "
                "dependent diagnostics rather than independent samples."
            ),
        },
    }


def build_analysis_contract(
    *,
    matched: bool,
    horizon: bool = False,
    longitudinal: bool = False,
) -> dict[str, object]:
    """Return the analysis choices frozen before native/model calls.

    ``pre_call_frozen`` is intentionally narrower than a claim of external
    public preregistration.  The contract is content-addressed in the immutable
    run identity, which prevents a completed run from silently promoting a raw
    penalty or endpoint violation to the primary analysis.
    """

    if horizon and not matched:
        raise ValueError("a horizon analysis contract requires matched constructs")
    if longitudinal and matched:
        raise ValueError("a longitudinal analysis contract cannot be matched")
    contribution_contracts = build_contribution_analysis_contracts(
        matched=matched,
        horizon=horizon,
        longitudinal=longitudinal,
    )
    if not matched:
        return {
            "status": "pre_call_frozen",
            "claim_id": (
                "C1-C3-longitudinal" if longitudinal else "C1-C3-standard"
            ),
            "analysis_role": (
                "longitudinal_state_control_drift_and_diagnosis"
                if longitudinal
                else "standard_state_control_and_diagnosis"
            ),
            "analysis_unit": "episode",
            "primary_estimands": [
                "mean_behavior_gain_beyond_workspace",
                "oracle_gap_closed",
            ],
            "workspace_control": "workspace_only",
            "history_availability_control": "full_context",
            "terminal_solvability_control": "oracle_current_state",
            "trajectory_interaction_mode": "replay_backed_critical_decision",
            "online_long_horizon_agent_execution_claim": False,
            "specification_timing": "before_native_writer_and_policy_calls",
            "external_preregistration": False,
            "claim_boundary": (
                "The release estimates paired value beyond workspace, "
                "lineage-backed drift only when its design, coverage, and control "
                "gates pass, and a same-decision unique-effect lower bound. Without "
                "matched static/evolution/conflict histories it does not identify "
                "the C1 mechanism penalty, and without a horizon-dose panel it "
                "does not identify horizon amplification."
            ),
            "contribution_contracts": contribution_contracts,
        }
    if horizon:
        return {
            "status": "pre_call_frozen",
            "claim_id": "C1-H",
            "analysis_role": "supplementary_construct_validity_diagnostic",
            "analysis_unit": "horizon_panel",
            "reference_horizon_level": "short",
            "intermediate_horizon_level": "medium",
            "target_horizon_level": "long",
            "reference_variant": "static",
            "treatment_variants": ["evolution", "hierarchical_conflict"],
            "workspace_control": "workspace_only",
            "history_availability_control": "full_context",
            "terminal_solvability_control": "oracle_current_state",
            "required_post_run_control_gates": [
                "matched_full_context_terminal_contract_solvability",
                "matched_oracle_terminal_contract_solvability",
            ],
            "primary_estimands": list(HORIZON_PRIMARY_ESTIMANDS),
            "secondary_estimands": list(HORIZON_SECONDARY_ESTIMANDS),
            "workspace_adjustment": (
                "difference_in_differences_in_differences_against_workspace_only"
            ),
            "effect_direction": (
                "positive_means_construct_penalty_grows_more_from_short_to_long_"
                "than_the_matched_workspace_only_penalty"
            ),
            "horizon_axis": (
                "joint_effective_transition_and_session_handoff_dose"
            ),
            "trajectory_interaction_mode": (
                "replay_backed_critical_decision"
            ),
            "online_long_horizon_agent_execution_claim": False,
            "horizon_evidence_scope": (
                "predecision_causal_span_and_delayed_state_dependence"
            ),
            "uncertainty_unit": "horizon_panel",
            "paired_test": "panel_level_sign_flip",
            "multiplicity_scope": "two_primary_horizon_amplification_estimands",
            "specification_timing": "before_native_writer_and_policy_calls",
            "external_preregistration": False,
            "contribution_contracts": contribution_contracts,
            "claim_boundary": (
                "This panel tests whether state-evolution and hierarchical-"
                "conflict penalties amplify under a joint transition/handoff "
                "dose while the terminal decision is held fixed. It is "
                "supplementary construct-validity evidence, not a pure handoff "
                "effect, not nine independent samples per panel, and not by "
                "itself a positive or confirmatory result. The tested policy "
                "selects the preregistered terminal continuation after a frozen, "
                "auditable prefix; this contract does not claim an online multi-"
                "hundred-step policy rollout. A policy cell cannot "
                "support memory-channel attribution if its full-context or "
                "oracle matched control fails."
            ),
        }
    return {
        "status": "pre_call_frozen",
        "claim_id": "C1",
        "analysis_unit": MATCHED_PRIMARY_ANALYSIS_UNIT,
        "reference_variant": "static",
        "treatment_variants": ["evolution", "hierarchical_conflict"],
        "workspace_control": "workspace_only",
        "history_availability_control": "full_context",
        "terminal_solvability_control": "oracle_current_state",
        "required_post_run_control_gates": [
            "matched_full_context_terminal_contract_solvability",
            "matched_oracle_terminal_contract_solvability",
        ],
        "primary_estimands": list(MATCHED_PRIMARY_ESTIMANDS),
        "secondary_estimands": list(MATCHED_SECONDARY_ESTIMANDS),
        "workspace_adjustment": MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT,
        "effect_direction": MATCHED_PRIMARY_EFFECT_DIRECTION,
        "drift_scope": MATCHED_DRIFT_SCOPE,
        "trajectory_interaction_mode": "replay_backed_critical_decision",
        "online_long_horizon_agent_execution_claim": False,
        "horizon_evidence_scope": (
            "predecision_causal_span_and_delayed_state_dependence"
        ),
        "uncertainty_unit": MATCHED_PRIMARY_ANALYSIS_UNIT,
        "paired_test": MATCHED_PAIRED_TEST,
        "multiplicity_scope": MATCHED_MULTIPLICITY_SCOPE,
        "specification_timing": "before_native_writer_and_policy_calls",
        "external_preregistration": False,
        "contribution_contracts": contribution_contracts,
        "claim_boundary": (
            "This contract fixes the matched C1 mechanism analysis. It does not "
            "promote the matched C2 endpoint to longitudinal onset, and it does "
            "not establish a positive or confirmatory C1--C3 result. The "
            "tested policy selects the terminal continuation after a frozen, "
            "auditable prefix; this contract does not claim an online multi-"
            "hundred-step policy rollout. A "
            "policy cell cannot support memory-channel attribution if its "
            "full-context or oracle matched control fails."
        ),
    }


def compute_experiment_design_audit(
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    """Return a deterministic, policy-free claim-identification audit."""

    ordered_specs = tuple(specs[key] for key in sorted(specs))
    groups: dict[str, list[SoftwareMem0VerticalSpec]] = defaultdict(list)
    panels: dict[str, list[SoftwareMem0VerticalSpec]] = defaultdict(list)
    ungrouped: list[str] = []
    unpanelled_matched: list[str] = []
    for spec in ordered_specs:
        group_id = spec.plan.metadata_dict.get("counterfactual_group_id", "")
        panel_id = spec.plan.metadata_dict.get("horizon_panel_id", "")
        if group_id:
            groups[group_id].append(spec)
        else:
            ungrouped.append(spec.plan.episode_id)
        if panel_id:
            panels[panel_id].append(spec)
        elif group_id:
            unpanelled_matched.append(spec.plan.episode_id)
    matched = bool(groups)
    horizon = bool(panels)
    longitudinal_members = tuple(
        spec.plan.metadata_dict.get("construct_mode")
        == "longitudinal_trajectory"
        for spec in ordered_specs
    )
    longitudinal = bool(longitudinal_members) and all(longitudinal_members)
    mixed_longitudinal_release = any(longitudinal_members) and not longitudinal
    mixed_release = matched and bool(ungrouped)
    mixed_horizon_release = horizon and bool(unpanelled_matched)
    matched_audits = tuple(
        audit_matched_construct_triplet(tuple(groups[group_id]))
        for group_id in sorted(groups)
    )
    panel_audits = tuple(
        _audit_horizon_specs(tuple(panels[panel_id]))
        for panel_id in sorted(panels)
    )
    balanced_scope = matched and (
        len(panels) >= 3 if horizon else len(groups) >= 3
    )
    legacy_long_horizon_scope = (
        matched
        and not horizon
        and bool(ordered_specs)
        and all(spec.plan.n_sessions >= 16 for spec in ordered_specs)
    )
    long_horizon_scope = legacy_long_horizon_scope or horizon or longitudinal

    heuristic = compute_heuristic_baselines(specs)
    drift_calibration = compute_drift_action_calibration(specs)
    longitudinal_design = _profile_longitudinal_design(ordered_specs)
    intervention_design = _profile_intervention_target_design(ordered_specs)
    best_action_accuracy = _finite_number(
        heuristic.get("best_always_action_accuracy")
    )
    best_option_accuracy = _finite_number(
        heuristic.get("best_always_option_accuracy")
    )

    recoverability_by_group = _group_metadata_values(
        groups,
        "recoverability_variant",
    )
    archetypes_by_group = _group_metadata_values(groups, "terminal_archetype")
    recoverability_counts = _single_value_counts(recoverability_by_group)
    archetype_counts = _single_value_counts(archetypes_by_group)
    gold_action_counts: Counter[str] = Counter()
    for audit in matched_audits:
        gold_action_counts.update(audit.gold_action_ids)

    spans = tuple(profile_task_span(spec.plan) for spec in ordered_specs)
    long_spans = tuple(
        profile_task_span(spec.plan)
        for spec in ordered_specs
        if not horizon or spec.plan.metadata_dict.get("horizon_level") == "long"
    )
    future_overlap: dict[str, list[str]] = {}
    missing_action_state: dict[str, list[str]] = {}
    memory_reliant_by_group: Counter[str] = Counter()
    memory_reliant_by_episode: Counter[str] = Counter()
    for spec in ordered_specs:
        group_id = spec.plan.metadata_dict.get("counterfactual_group_id", "")
        for sceu in spec.plan.sceu_units:
            construct = profile_sceu(spec.plan, sceu)
            overlap = sorted(
                set(construct.current_required_state_ids).intersection(
                    construct.future_referenced_state_ids
                )
            )
            if overlap:
                future_overlap[f"{spec.plan.episode_id}|{sceu.sceu_id}"] = overlap
            if construct.missing_current_action_relevant_state_ids:
                missing_action_state[
                    f"{spec.plan.episode_id}|{sceu.sceu_id}"
                ] = list(construct.missing_current_action_relevant_state_ids)
            if group_id and construct.memory_reliant_state_ids:
                memory_reliant_by_group[group_id] += 1
            if construct.memory_reliant_state_ids:
                memory_reliant_by_episode[spec.plan.episode_id] += 1

    absent_groups = tuple(
        sorted(
            group_id
            for group_id, values in recoverability_by_group.items()
            if values == {"absent"}
        )
    )
    checks: list[dict[str, object]] = []
    _check(
        checks,
        "matched_release_membership",
        not mixed_release,
        applicable=matched,
        description=(
            "Matched releases contain no ungrouped physical episodes."
        ),
        detail={"ungrouped_episode_ids": sorted(ungrouped)},
    )
    _check(
        checks,
        "longitudinal_release_membership",
        not mixed_longitudinal_release and not (longitudinal and matched),
        applicable=any(longitudinal_members),
        description=(
            "A longitudinal release contains only episodes that declare the "
            "v0.13 longitudinal trajectory contract."
        ),
        detail={
            "longitudinal_episode_ids": [
                spec.plan.episode_id
                for spec, declared in zip(
                    ordered_specs,
                    longitudinal_members,
                    strict=True,
                )
                if declared
            ],
            "other_episode_ids": [
                spec.plan.episode_id
                for spec, declared in zip(
                    ordered_specs,
                    longitudinal_members,
                    strict=True,
                )
                if not declared
            ],
        },
    )
    _check(
        checks,
        "matched_triplet_structural_invariance",
        bool(matched_audits) and all(audit.ok for audit in matched_audits),
        applicable=matched,
        description=(
            "Every group contains an invariant static/evolution/conflict triplet."
        ),
        detail={audit.group_id: audit.to_dict() for audit in matched_audits},
    )
    _check(
        checks,
        "matched_gold_action_balance",
        set(gold_action_counts) == _EXPECTED_GOLD_ACTIONS
        and _balanced(gold_action_counts, _EXPECTED_GOLD_ACTIONS),
        applicable=balanced_scope,
        description=(
            "Matched groups cover the three terminal gold actions with balanced "
            "frequency."
        ),
        detail=dict(sorted(gold_action_counts.items())),
    )
    _check(
        checks,
        "matched_action_shortcut_resistance",
        best_action_accuracy is not None and best_action_accuracy <= 0.50,
        applicable=balanced_scope,
        description="No fixed action solves more than half of matched decisions.",
        detail={
            "best_always_action": heuristic.get("best_always_action"),
            "best_always_action_accuracy": best_action_accuracy,
        },
    )
    _check(
        checks,
        "matched_option_shortcut_resistance",
        best_option_accuracy is not None and best_option_accuracy <= 0.40,
        applicable=balanced_scope,
        description=(
            "No fixed opaque option position solves more than 40% of matched "
            "decisions."
        ),
        detail={
            "best_always_option": heuristic.get("best_always_option"),
            "best_always_option_accuracy": best_option_accuracy,
        },
    )
    _check(
        checks,
        "matched_workspace_recoverability_balance",
        set(recoverability_counts) == _EXPECTED_RECOVERABILITY
        and _balanced(recoverability_counts, _EXPECTED_RECOVERABILITY)
        and all(len(values) == 1 for values in recoverability_by_group.values()),
        applicable=balanced_scope,
        description=(
            "Counterfactual groups balance explicit, derivable, and absent "
            "workspace recoverability."
        ),
        detail={
            "counts": dict(sorted(recoverability_counts.items())),
            "by_group": {
                group_id: sorted(values)
                for group_id, values in sorted(recoverability_by_group.items())
            },
        },
    )
    _check(
        checks,
        "matched_terminal_archetype_balance",
        set(archetype_counts) == _EXPECTED_TERMINAL_ARCHETYPES
        and _balanced(archetype_counts, _EXPECTED_TERMINAL_ARCHETYPES)
        and all(len(values) == 1 for values in archetypes_by_group.values()),
        applicable=balanced_scope,
        description=(
            "Terminal archetypes are balanced across groups rather than "
            "confounded with one history construct."
        ),
        detail={
            "counts": dict(sorted(archetype_counts.items())),
            "by_group": {
                group_id: sorted(values)
                for group_id, values in sorted(archetypes_by_group.items())
            },
        },
    )
    _check(
        checks,
        "matched_drift_checker_calibration",
        drift_calibration.get("all_categories_calibrated") is True,
        applicable=balanced_scope,
        description=(
            "Every canonical drift category has policy-free checker-positive and "
            "checker-negative actions across the balanced matched release."
        ),
        detail={
            "all_categories_calibrated": drift_calibration.get(
                "all_categories_calibrated"
            ),
            "category_calibration": drift_calibration.get(
                "categories",
                {},
            ),
            "semantic_scenarios": drift_calibration.get(
                "semantic_scenarios",
                {},
            ),
        },
    )
    _check(
        checks,
        "current_requirements_exclude_future_state",
        not future_overlap,
        applicable=matched or longitudinal,
        description=(
            "No checkpoint requires a state that is only valid in the future."
        ),
        detail=future_overlap,
    )
    _check(
        checks,
        "current_action_state_contract_complete",
        not missing_action_state,
        applicable=matched or longitudinal,
        description=(
            "Every current checker-relevant state is part of the terminal "
            "SCEU required closure, so oracle solvability is not limited by an "
            "incomplete evaluator state contract."
        ),
        detail=missing_action_state,
    )
    _check(
        checks,
        "workspace_absent_memory_reliant_decisions_present",
        (
            bool(ordered_specs)
            and all(
                memory_reliant_by_episode[spec.plan.episode_id] > 0
                for spec in ordered_specs
            )
            if longitudinal
            else bool(absent_groups)
            and all(
                memory_reliant_by_group[group_id] > 0
                for group_id in absent_groups
            )
        ),
        applicable=balanced_scope or longitudinal,
        description=(
            "Every applicable episode or workspace-absent matched group "
            "contains a decision whose current required state is not "
            "recoverable from the workspace."
        ),
        detail={
            "absent_groups": list(absent_groups),
            "memory_reliant_sceu_by_group": {
                group_id: memory_reliant_by_group[group_id]
                for group_id in sorted(groups)
            },
            "memory_reliant_sceu_by_episode": {
                spec.plan.episode_id: memory_reliant_by_episode[
                    spec.plan.episode_id
                ]
                for spec in ordered_specs
            },
        },
    )
    _check(
        checks,
        "c2_longitudinal_drift_checker_calibration",
        drift_calibration.get("all_categories_calibrated") is True
        and drift_calibration.get("all_represented_scenarios_calibrated") is True,
        applicable=longitudinal,
        description=(
            "Every canonical drift category and represented semantic scenario "
            "has policy-free checker-positive and checker-negative actions."
        ),
        detail={
            "all_categories_calibrated": drift_calibration.get(
                "all_categories_calibrated"
            ),
            "all_represented_scenarios_calibrated": drift_calibration.get(
                "all_represented_scenarios_calibrated"
            ),
            "categories": drift_calibration.get("categories", {}),
            "semantic_scenarios": drift_calibration.get(
                "semantic_scenarios",
                {},
            ),
        },
    )
    _check(
        checks,
        "c2_longitudinal_lineage_design",
        longitudinal_design["lineage_design_complete"] is True,
        applicable=longitudinal,
        description=(
            "Every episode and drift category has one unambiguous state lineage "
            "observed at distinct anchor and later challenge checkpoints."
        ),
        detail=longitudinal_design,
    )
    _check(
        checks,
        "c2_longitudinal_recovery_design",
        longitudinal_design["recovery_design_complete"] is True,
        applicable=longitudinal,
        description=(
            "Every episode and drift category has a same-lineage anchor, a later "
            "ordinary challenge, and a still-later reminder/update checkpoint."
        ),
        detail=longitudinal_design,
    )
    _check(
        checks,
        "c3_intervention_target_contract",
        intervention_design["contract_complete"] is True,
        applicable=bool(
            intervention_design["memory_reliant_decision_count"]
        ),
        description=(
            "Every memory-reliant decision declares a current, action-relevant "
            "state target for same-decision neutral-replacement intervention."
        ),
        detail=intervention_design,
    )
    _check(
        checks,
        "long_horizon_effective_step_span",
        bool(long_spans)
        and all(span.meets_long_horizon_step_threshold for span in long_spans),
        applicable=long_horizon_scope,
        description=(
            "Every applicable long-horizon trajectory meets the preregistered "
            "effective causal-step threshold before a scored decision; short "
            "and medium horizon-panel comparison doses are exempt."
        ),
        detail={
            "minimum_terminal_decision_causal_span": min(
                (
                    span.maximum_decision_causal_span
                    for span in long_spans
                    if span.maximum_decision_causal_span is not None
                ),
                default=None,
            )
        },
    )
    _check(
        checks,
        "task_step_effect_chain_integrity",
        bool(spans) and all(span.effect_chain_verified for span in spans),
        applicable=long_horizon_scope,
        description=(
            "Every declared effective step has a verified predecessor/effect digest "
            "chain."
        ),
        detail={
            "verified_episode_count": sum(span.effect_chain_verified for span in spans),
            "episode_count": len(spans),
        },
    )
    _check(
        checks,
        "task_step_anti_padding_integrity",
        bool(spans) and all(span.anti_padding_verified for span in spans),
        applicable=long_horizon_scope,
        description=(
            "Every counted task step produces a unique semantic effect and "
            "every pre-decision effect is consumed downstream; a chain of "
            "unconsumed observations cannot satisfy the horizon threshold."
        ),
        detail={
            "verified_episode_count": sum(
                span.anti_padding_verified for span in spans
            ),
            "episode_count": len(spans),
            "minimum_semantic_effect_coverage": min(
                (
                    span.semantic_effect_coverage
                    for span in spans
                    if span.semantic_effect_coverage is not None
                ),
                default=None,
            ),
            "minimum_consumed_prefix_effect_fraction": min(
                (
                    span.consumed_prefix_effect_fraction
                    for span in spans
                    if span.consumed_prefix_effect_fraction is not None
                ),
                default=None,
            ),
        },
    )
    interaction_mode_counts = Counter(
        span.interaction_mode for span in spans
    )
    _check(
        checks,
        "trajectory_interaction_claim_boundary",
        bool(spans)
        and set(interaction_mode_counts) == {
            "replay_backed_critical_decision"
        }
        and not any(
            span.online_long_horizon_agent_execution_supported
            for span in spans
        ),
        applicable=long_horizon_scope,
        description=(
            "Every current long-horizon release declares replay-backed critical-"
            "decision evaluation and explicitly withholds the stronger claim "
            "that the tested policy executed the full long-horizon trajectory "
            "online."
        ),
        detail={
            "interaction_mode_counts": dict(
                sorted(interaction_mode_counts.items())
            ),
            "policy_conditioned_future_step_count": sum(
                span.policy_conditioned_future_step_count for span in spans
            ),
            "policy_dependent_decision_count": sum(
                span.policy_dependent_decision_count for span in spans
            ),
            "online_long_horizon_agent_execution_supported": all(
                span.online_long_horizon_agent_execution_supported
                for span in spans
            ) if spans else False,
            "allowed_claim_scope": (
                "memory_supported_critical_decisions_after_audited_long_"
                "horizon_prefixes"
            ),
        },
    )
    _check(
        checks,
        "horizon_panel_membership",
        not mixed_horizon_release
        and bool(panel_audits)
        and sum(audit.n_physical_episodes for audit in panel_audits)
        == len(ordered_specs),
        applicable=horizon,
        description=(
            "Every physical member belongs to exactly one complete horizon panel."
        ),
        detail={
            "horizon_panel_ids": sorted(panels),
            "unpanelled_episode_ids": sorted(unpanelled_matched),
        },
    )
    _check(
        checks,
        "horizon_panel_structural_invariance",
        bool(panel_audits)
        and all(_panel_structurally_invariant(audit) for audit in panel_audits),
        applicable=horizon,
        description=(
            "Terminal state, workspace, opaque options, checker, and decision "
            "remain invariant across horizon doses."
        ),
        detail={audit.panel_id: audit.to_dict() for audit in panel_audits},
    )
    _check(
        checks,
        "horizon_levels_complete",
        bool(panel_audits)
        and all(
            audit.levels == ("short", "medium", "long")
            and audit.n_physical_episodes == audit.expected_physical_episodes == 9
            for audit in panel_audits
        ),
        applicable=horizon,
        description=(
            "Every panel has one static/evolution/conflict triplet at short, "
            "medium, and long dose."
        ),
        detail={
            audit.panel_id: {
                "levels": list(audit.levels),
                "n_physical_episodes": audit.n_physical_episodes,
            }
            for audit in panel_audits
        },
    )
    _check(
        checks,
        "horizon_joint_dose_monotonic",
        bool(panel_audits)
        and all(
            all(item.strictly_increasing_joint_dose for item in audit.variant_audits)
            for audit in panel_audits
        ),
        applicable=horizon,
        description=(
            "Effective transitions, session handoffs, and dependency depth all "
            "increase from short to medium to long."
        ),
        detail={
            audit.panel_id: {
                item.variant: {
                    "effective_step_counts": list(item.effective_step_counts),
                    "handoff_counts": list(item.handoff_counts),
                    "dependency_depths": list(item.dependency_depths),
                }
                for item in audit.variant_audits
            }
            for audit in panel_audits
        },
    )
    _check(
        checks,
        "horizon_long_only_step_threshold",
        bool(panel_audits)
        and all(audit.long_level_meets_effective_step_threshold for audit in panel_audits),
        applicable=horizon,
        description=(
            "Only the long dose is required to cross the preregistered "
            "long-horizon effective-step threshold."
        ),
        detail={
            "long_minimum_effective_step_count": min(
                (span.effective_step_count for span in long_spans),
                default=None,
            ),
            "short_and_medium_are_comparison_doses": True,
        },
    )

    failed = tuple(
        str(row["check_id"])
        for row in checks
        if row["status"] == "fail"
    )
    balanced_ready = (
        balanced_scope
        and long_horizon_scope
        and not failed
    )
    return {
        "schema_version": EXPERIMENT_DESIGN_AUDIT_SCHEMA_VERSION,
        "scope": (
            "horizon_dose_diagnostic"
            if horizon
            else (
                "matched_mechanism"
                if matched
                else (
                    "longitudinal_trajectory"
                    if longitudinal
                    else "standard_trajectory"
                )
            )
        ),
        "analysis_unit": (
            "horizon_panel"
            if horizon
            else ("counterfactual_group" if matched else "episode")
        ),
        "physical_episode_count": len(ordered_specs),
        "counterfactual_group_count": len(groups),
        "horizon_panel_count": len(panels),
        "analysis_contract": build_analysis_contract(
            matched=matched,
            horizon=horizon,
            longitudinal=longitudinal,
        ),
        "trajectory_interaction_mode_counts": dict(
            sorted(interaction_mode_counts.items())
        ),
        "online_long_horizon_agent_execution_supported": (
            bool(spans)
            and all(
                span.online_long_horizon_agent_execution_supported
                for span in spans
            )
        ),
        "run_ready": not failed,
        "balanced_mechanism_design_ready": balanced_ready,
        "audit_status": (
            "invalid"
            if failed
            else (
                "ready_for_calibration"
                if balanced_ready
                else "diagnostic_only"
            )
        ),
        "failed_check_ids": list(failed),
        "checks": checks,
        "interpretation": (
            "This policy-free audit establishes design identifiability only. It "
            "does not establish model performance, effect direction, uncertainty, "
            "or confirmatory status."
        ),
    }


def experiment_design_audit_markdown(payload: Mapping[str, object]) -> str:
    """Render the pre-call design audit for human report review."""

    lines = [
        "# Experiment design audit",
        "",
        f"Audit status: **{payload.get('audit_status', 'missing')}**.",
        "",
        (
            "This audit is policy-free and runs before native writer or policy "
            "calls. Passing it establishes design identifiability, not a positive "
            "or statistically significant effect."
        ),
        "",
        "| Check | Status | Purpose |",
        "|---|---|---|",
    ]
    raw_checks = payload.get("checks")
    if isinstance(raw_checks, list):
        for raw in raw_checks:
            if not isinstance(raw, Mapping):
                continue
            lines.append(
                "| `{}` | **{}** | {} |".format(
                    raw.get("check_id", ""),
                    raw.get("status", ""),
                    raw.get("description", ""),
                )
            )
    lines.extend(
        [
            "",
            (
                "Balanced matched-mechanism design ready: **{}**."
            ).format(
                str(
                    payload.get("balanced_mechanism_design_ready") is True
                ).lower()
            ),
            "",
        ]
    )
    contract = payload.get("analysis_contract")
    if isinstance(contract, Mapping):
        lines.extend(
            [
                "## Analysis contract",
                "",
                f"Status: **{contract.get('status', 'missing')}**.",
                "",
            ]
        )
        if contract.get("status") == "pre_call_frozen":
            lines.extend(
                [
                    (
                        "Primary analysis unit: "
                        f"**{contract.get('analysis_unit', '')}**."
                    ),
                    "",
                    "Primary estimands:",
                ]
            )
            primary = contract.get("primary_estimands")
            if isinstance(primary, list):
                lines.extend(f"- `{value}`" for value in primary)
            lines.extend(
                [
                    "",
                    (
                        "Trajectory interaction mode: "
                        f"**{contract.get('trajectory_interaction_mode', '')}**."
                    ),
                    "",
                    (
                        "Online multi-hundred-step policy execution claimed: "
                        "**{}**."
                    ).format(
                        str(
                            contract.get(
                                "online_long_horizon_agent_execution_claim"
                            )
                            is True
                        ).lower()
                    ),
                ]
            )
            if contract.get("analysis_unit") == "horizon_panel":
                lines.extend(
                    [
                        "",
                        (
                            "Short, medium, and long members are repeated conditions; "
                            "the complete horizon panel is the statistical unit."
                        ),
                        "",
                        (
                            "The dose jointly changes effective transitions, "
                            "dependency depth, and handoffs and is not a pure "
                            "handoff manipulation."
                        ),
                        "",
                        str(contract.get("claim_boundary", "")),
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        (
                            "Raw history penalties, correctness penalties, and "
                            "matched endpoint drift violations are secondary analyses."
                        ),
                        "",
                        (
                            "The matched drift scope is endpoint violation only; "
                            "longitudinal onset requires an earlier adherence "
                            "checkpoint."
                        ),
                        "",
                        str(contract.get("claim_boundary", "")),
                        "",
                    ]
                )
        else:
            lines.extend([str(contract.get("reason", "")), ""])
        contribution_contracts = contract.get("contribution_contracts")
        if isinstance(contribution_contracts, Mapping):
            lines.extend(
                [
                    "## Contribution-specific pre-call contracts",
                    "",
                    "| Contribution | Status | Claim scope | Analysis unit |",
                    "|---|---|---|---|",
                ]
            )
            for contribution_id in ("C1", "C2", "C3"):
                raw = contribution_contracts.get(contribution_id)
                if not isinstance(raw, Mapping):
                    continue
                lines.append(
                    "| `{}` | **{}** | `{}` | `{}` |".format(
                        contribution_id,
                        raw.get("status", "missing"),
                        raw.get("claim_scope", ""),
                        raw.get("analysis_unit", ""),
                    )
                )
            lines.append("")
    return "\n".join(lines)


def _profile_longitudinal_design(
    specs: tuple[SoftwareMem0VerticalSpec, ...],
) -> dict[str, object]:
    """Audit whether frozen gold permits onset, persistence, and recovery.

    This profile is deliberately policy-free.  It proves only that every
    category has a same-lineage temporal risk set with a possible adhering
    action, a later ordinary challenge, and a still-later recovery control.
    Whether a tested system actually follows that trajectory is post-run
    evidence.
    """

    recovery_controls = {"fresh_reminder", "valid_update"}
    episode_details: dict[str, object] = {}
    missing_lineage_windows: dict[str, list[str]] = {}
    missing_recovery_windows: dict[str, list[str]] = {}
    ambiguous_sceu_categories: dict[str, list[str]] = {}
    for spec in specs:
        opportunity_by_id = {
            item.opportunity_id: item for item in spec.plan.opportunities
        }
        rows_by_category_lineage: dict[
            tuple[str, str], list[tuple[int, str, str]]
        ] = defaultdict(list)
        ambiguous: list[str] = []
        for sceu in spec.plan.sceu_units:
            pairs = drift_lineage_pairs(spec, sceu)
            by_category: dict[str, set[str]] = defaultdict(set)
            for category, lineage in pairs:
                by_category[category].add(lineage)
            for category in drift_eligible_categories(spec, sceu):
                lineages = by_category.get(category, set())
                if len(lineages) != 1:
                    ambiguous.append(
                        f"{sceu.opportunity_id}|{category}|"
                        + ",".join(sorted(lineages))
                    )
                    continue
                lineage = next(iter(lineages))
                opportunity = opportunity_by_id[sceu.opportunity_id]
                rows_by_category_lineage[(category, lineage)].append(
                    (
                        sceu.checkpoint_session,
                        opportunity.opportunity_id,
                        opportunity.control_kind,
                    )
                )

        category_details: dict[str, object] = {}
        missing_lineage: list[str] = []
        missing_recovery: list[str] = []
        for category in CANONICAL_DRIFT_CATEGORIES:
            candidates: list[dict[str, object]] = []
            for (declared_category, lineage), raw_rows in sorted(
                rows_by_category_lineage.items()
            ):
                if declared_category != category:
                    continue
                rows = tuple(sorted(set(raw_rows)))
                sessions = tuple(sorted({row[0] for row in rows}))
                ordinary_challenges = tuple(
                    sorted(
                        {
                            row[0]
                            for row in rows
                            if row[2] not in recovery_controls
                        }
                    )
                )
                recovery_sessions = tuple(
                    sorted(
                        {
                            row[0]
                            for row in rows
                            if row[2] in recovery_controls
                        }
                    )
                )
                lineage_window = any(
                    earlier < later
                    for earlier in sessions
                    for later in sessions
                )
                recovery_window = next(
                    (
                        (anchor, challenge, recovery)
                        for anchor in sessions
                        for challenge in ordinary_challenges
                        for recovery in recovery_sessions
                        if anchor < challenge < recovery
                    ),
                    None,
                )
                candidates.append(
                    {
                        "state_lineage_id": lineage,
                        "eligible_checkpoint_sessions": list(sessions),
                        "ordinary_challenge_sessions": list(
                            ordinary_challenges
                        ),
                        "recovery_control_sessions": list(recovery_sessions),
                        "lineage_window_complete": lineage_window,
                        "recovery_window": (
                            None
                            if recovery_window is None
                            else list(recovery_window)
                        ),
                        "rows": [list(row) for row in rows],
                    }
                )
            lineage_complete = any(
                item["lineage_window_complete"] is True
                for item in candidates
            )
            recovery_complete = any(
                item["recovery_window"] is not None
                for item in candidates
            )
            if not lineage_complete:
                missing_lineage.append(category)
            if not recovery_complete:
                missing_recovery.append(category)
            category_details[category] = {
                "lineage_window_complete": lineage_complete,
                "recovery_design_complete": recovery_complete,
                "candidate_lineages": candidates,
            }
        if ambiguous:
            ambiguous_sceu_categories[spec.plan.episode_id] = sorted(ambiguous)
        if missing_lineage:
            missing_lineage_windows[spec.plan.episode_id] = missing_lineage
        if missing_recovery:
            missing_recovery_windows[spec.plan.episode_id] = missing_recovery
        episode_details[spec.plan.episode_id] = {
            "categories": category_details,
            "ambiguous_sceu_categories": sorted(ambiguous),
        }
    return {
        "lineage_design_complete": bool(specs)
        and not missing_lineage_windows
        and not ambiguous_sceu_categories,
        "recovery_design_complete": bool(specs)
        and not missing_recovery_windows
        and not ambiguous_sceu_categories,
        "missing_lineage_windows": missing_lineage_windows,
        "missing_recovery_windows": missing_recovery_windows,
        "ambiguous_sceu_categories": ambiguous_sceu_categories,
        "episodes": episode_details,
    }


def _profile_intervention_target_design(
    specs: tuple[SoftwareMem0VerticalSpec, ...],
) -> dict[str, object]:
    """Check target-level identifiability before running C3 interventions."""

    memory_reliant_decision_count = 0
    targeted_memory_reliant_decision_count = 0
    missing_targets: dict[str, dict[str, list[str]]] = {}
    unknown_targets: dict[str, dict[str, list[str]]] = {}
    strict_target_errors: dict[str, dict[str, list[str]]] = {}
    for spec in specs:
        state_ids = {state.state_id for state in spec.plan.state_units}
        strict = (
            spec.plan.metadata_dict.get("construct_mode")
            == "longitudinal_trajectory"
        )
        for sceu in spec.plan.sceu_units:
            profile = profile_sceu(spec.plan, sceu)
            targets = set(sceu.intervention_target_ids)
            unknown = sorted(targets.difference(state_ids))
            if unknown:
                unknown_targets.setdefault(spec.plan.episode_id, {})[
                    sceu.sceu_id
                ] = unknown
            if strict:
                invalid = sorted(
                    targets.difference(
                        set(profile.current_required_state_ids).intersection(
                            profile.current_action_relevant_state_ids
                        )
                    )
                )
                if invalid:
                    strict_target_errors.setdefault(
                        spec.plan.episode_id,
                        {},
                    )[sceu.sceu_id] = invalid
            if not profile.memory_reliant_state_ids:
                continue
            memory_reliant_decision_count += 1
            supported_targets = targets.intersection(
                profile.memory_reliant_state_ids
            ).intersection(profile.current_action_relevant_state_ids)
            if supported_targets:
                targeted_memory_reliant_decision_count += 1
            else:
                missing_targets.setdefault(spec.plan.episode_id, {})[
                    sceu.sceu_id
                ] = list(profile.memory_reliant_state_ids)
    return {
        "contract_complete": memory_reliant_decision_count > 0
        and targeted_memory_reliant_decision_count
        == memory_reliant_decision_count
        and not missing_targets
        and not unknown_targets
        and not strict_target_errors,
        "memory_reliant_decision_count": memory_reliant_decision_count,
        "targeted_memory_reliant_decision_count": (
            targeted_memory_reliant_decision_count
        ),
        "missing_action_relevant_targets": missing_targets,
        "unknown_target_state_ids": unknown_targets,
        "longitudinal_noncurrent_or_nondiscriminative_targets": (
            strict_target_errors
        ),
        "intervention_protocol": {
            "primary": "repeat_stable_neutral_replacement",
            "negative_control": "rank_count_length_matched_neutral_sham",
            "unit": "same_state_conditioned_evaluation_unit",
        },
    }


def _group_metadata_values(
    groups: Mapping[str, list[SoftwareMem0VerticalSpec]],
    field: str,
) -> dict[str, set[str]]:
    return {
        group_id: {
            spec.plan.metadata_dict.get(field, "") for spec in group_specs
        }
        for group_id, group_specs in groups.items()
    }


def _audit_horizon_specs(
    specs: tuple[SoftwareMem0VerticalSpec, ...],
) -> HorizonPanelAudit:
    by_level = {
        spec.plan.metadata_dict.get("horizon_level", ""): spec
        for spec in specs
    }
    if set(by_level) != {"short", "medium", "long"}:
        return audit_horizon_panel(specs)
    doses = tuple(
        HorizonDose(
            level,
            by_level[level].plan.n_sessions,
            int(by_level[level].plan.metadata_dict["horizon_steps_per_session"]),
        )
        for level in ("short", "medium", "long")
    )
    return audit_horizon_panel(specs, doses=doses)


def _panel_structurally_invariant(audit: HorizonPanelAudit) -> bool:
    return (
        audit.unique_episode_ids
        and audit.within_dose_triplets_ok
        and bool(audit.variant_audits)
        and all(
            item.terminal_decision_signature_count == 1
            and item.terminal_state_signature_count == 1
            and item.terminal_workspace_signature_count == 1
            and item.opaque_option_signature_count == 1
            and item.executable_checker_signature_count == 1
            and item.terminal_condition_signature_count == 1
            and item.all_targets_at_final_session
            for item in audit.variant_audits
        )
    )


def _single_value_counts(values_by_group: Mapping[str, set[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for values in values_by_group.values():
        if len(values) == 1:
            counts.update(values)
    return counts


def _balanced(counts: Mapping[str, int], expected: set[str]) -> bool:
    values = [counts.get(key, 0) for key in expected]
    return bool(values) and min(values) > 0 and max(values) - min(values) <= 1


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    converted = float(value)
    return converted if float("-inf") < converted < float("inf") else None


def _check(
    checks: list[dict[str, object]],
    check_id: str,
    passed: bool,
    *,
    applicable: bool,
    description: str,
    detail: object,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": (
                "pass" if applicable and passed else (
                    "fail" if applicable else "not_applicable"
                )
            ),
            "description": description,
            "detail": detail,
        }
    )


__all__ = [
    "EXPERIMENT_DESIGN_AUDIT_SCHEMA_VERSION",
    "EXPERIMENT_DESIGN_CHECK_IDS",
    "build_contribution_analysis_contracts",
    "build_analysis_contract",
    "compute_experiment_design_audit",
    "experiment_design_audit_markdown",
]
