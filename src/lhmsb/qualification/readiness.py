"""Policy-free baselines and preregistered measurement-readiness gates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from lhmsb.families.software.matched_constructs import (
    audit_matched_construct_triplet,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical_checker import (
    BehaviorResult,
    assess_software_action,
)
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.task_span import (
    profile_task_span,
)
from lhmsb.qualification.drift import (
    CANONICAL_DRIFT_CATEGORIES,
    drift_eligible_categories,
    normalized_action_drift,
)
from lhmsb.qualification.longitudinal import compute_drift_trajectory_report
from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    compute_matched_construct_contrasts,
)

BASELINE_STABILITY_MIN = 0.90
SHAM_ACTION_FLIP_MAX = 0.05
ORACLE_ACCURACY_MIN = 0.95
ORACLE_GROUP_ACCURACY_MIN = 0.90
MAX_ALWAYS_ACTION_ACCURACY = 0.50
MAX_ALWAYS_OPTION_ACCURACY = 0.40
MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE = 2
SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN = 0.90
FLAT_CAUSAL_PROBE_COVERAGE_MIN = 0.50
_ONE_SIDED_95_Z = 1.6448536269514722
_MEMORY_CONDITIONS = frozenset({"flat_retrieval", "mem0", "amem", "memos"})
_LONGITUDINAL_CONTROL_CONDITIONS = (
    "oracle_current_state",
    "full_context",
)
_DRIFT_CATEGORIES = CANONICAL_DRIFT_CATEGORIES
_MATCHED_CONTROL_VARIANTS = (
    "static",
    "evolution",
    "hierarchical_conflict",
)


def compute_heuristic_baselines(
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    """Score deterministic action/option heuristics without any model calls."""
    action_correct: Counter[str] = Counter()
    option_correct: Counter[str] = Counter()
    gold_assignments: Counter[str] = Counter()
    scenario_opportunities: Counter[str] = Counter()
    random_expected_correct = 0.0
    opportunities = 0
    per_episode: list[dict[str, object]] = []

    for episode_id, spec in sorted(specs.items()):
        episode_action_correct: Counter[str] = Counter()
        episode_opportunities = 0
        evaluator_by_id = spec.evaluator_continuation_map
        scenario = dict(spec.plan.metadata).get("semantic_scenario", "unknown")
        for opportunity in spec.plan.opportunities:
            valid = set(opportunity.valid_action_ids)
            action_ids = tuple(action.action_id for action in opportunity.action_catalog)
            evaluator = evaluator_by_id[opportunity.opportunity_id]
            option_map = dict(evaluator.option_to_action)
            opportunities += 1
            episode_opportunities += 1
            scenario_opportunities[str(scenario)] += 1
            random_expected_correct += len(valid) / len(action_ids)
            for action_id in action_ids:
                if action_id in valid:
                    action_correct[action_id] += 1
                    episode_action_correct[action_id] += 1
            for option_id, action_id in option_map.items():
                if action_id in valid:
                    option_correct[option_id] += 1
            gold_assignments.update(valid)
        per_episode.append(
            {
                "episode_id": episode_id,
                "semantic_scenario": str(scenario),
                "n_opportunities": episode_opportunities,
                "always_action_accuracy": {
                    action_id: count / episode_opportunities
                    for action_id, count in sorted(episode_action_correct.items())
                },
            }
        )

    action_accuracy = (
        {action_id: count / opportunities for action_id, count in sorted(action_correct.items())}
        if opportunities
        else {}
    )
    option_accuracy = (
        {option_id: count / opportunities for option_id, count in sorted(option_correct.items())}
        if opportunities
        else {}
    )
    best_action, best_accuracy = _best_baseline(action_accuracy)
    best_option, best_option_accuracy = _best_baseline(option_accuracy)
    return {
        "schema_version": 1,
        "scope": "policy_free_frozen_gold",
        "n_episodes": len(specs),
        "n_opportunities": opportunities,
        "gold_valid_assignment_counts": dict(sorted(gold_assignments.items())),
        "semantic_scenario_opportunity_counts": dict(sorted(scenario_opportunities.items())),
        "always_action_accuracy": action_accuracy,
        "always_option_accuracy": option_accuracy,
        "uniform_random_expected_accuracy": (
            None if opportunities == 0 else random_expected_correct / opportunities
        ),
        "best_always_action": best_action,
        "best_always_action_accuracy": best_accuracy,
        "best_always_option": best_option,
        "best_always_option_accuracy": best_option_accuracy,
        "per_episode": per_episode,
        "note": (
            "These baselines use no workspace, history, memory, or policy model. "
            "They diagnose action-label and opaque-option shortcuts."
        ),
    }


def compute_drift_action_calibration(
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    """Prove checker sensitivity and specificity without policy-model calls.

    Every catalog action is classified at every eligible opportunity using the
    same state assessment and normalized drift logic as the scored evaluator.
    A category is calibrated only if it has both positive and negative action
    assignments, at least one invalid positive, and no positive gold-valid
    assignment.  The same invariant must hold within every represented semantic
    scenario so lexical variants cannot silently remove a construct.
    """
    counts = {category: Counter[str]() for category in _DRIFT_CATEGORIES}
    scenario_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {category: Counter() for category in _DRIFT_CATEGORIES}
    )
    examples: dict[str, dict[str, list[dict[str, object]]]] = {
        category: {"positive": [], "negative": [], "valid_false_positive": []}
        for category in _DRIFT_CATEGORIES
    }
    n_opportunities = 0
    n_action_assignments = 0

    for episode_id, spec in sorted(specs.items()):
        scenario = str(dict(spec.plan.metadata).get("semantic_scenario", "unknown"))
        sceu_by_opportunity = {sceu.opportunity_id: sceu for sceu in spec.plan.sceu_units}
        for opportunity in spec.plan.opportunities:
            sceu = sceu_by_opportunity[opportunity.opportunity_id]
            eligible = drift_eligible_categories(spec, sceu)
            if not eligible:
                continue
            n_opportunities += 1
            for category in eligible:
                counts[category]["eligible_opportunities"] += 1
                scenario_counts[scenario][category]["eligible_opportunities"] += 1
            valid = set(opportunity.valid_action_ids)
            for action in opportunity.action_catalog:
                n_action_assignments += 1
                assessment = assess_software_action(
                    spec.plan,
                    action,
                    checkpoint_session=opportunity.checkpoint_session,
                    opportunity_id=opportunity.opportunity_id,
                )
                behavior = BehaviorResult(
                    score=0.0,
                    is_correct=False,
                    violated_state_ids=assessment.violated_state_ids,
                    drift_flags=assessment.drift_flags,
                )
                normalized = set(
                    normalized_action_drift(
                        spec,
                        action,
                        behavior,
                        opportunity.checkpoint_session,
                    )
                )
                is_valid = action.action_id in valid
                for category in eligible:
                    target = counts[category]
                    scenario_target = scenario_counts[scenario][category]
                    target["action_assignments"] += 1
                    scenario_target["action_assignments"] += 1
                    target[
                        "valid_action_assignments" if is_valid else "invalid_action_assignments"
                    ] += 1
                    scenario_target[
                        "valid_action_assignments" if is_valid else "invalid_action_assignments"
                    ] += 1
                    positive = category in normalized
                    label = "positive_assignments" if positive else "negative_assignments"
                    target[label] += 1
                    scenario_target[label] += 1
                    if positive and is_valid:
                        target["valid_positive_assignments"] += 1
                        scenario_target["valid_positive_assignments"] += 1
                    if positive and not is_valid:
                        target["invalid_positive_assignments"] += 1
                        scenario_target["invalid_positive_assignments"] += 1
                    example_kind = (
                        "valid_false_positive"
                        if positive and is_valid
                        else ("positive" if positive else "negative")
                    )
                    if len(examples[category][example_kind]) < 3:
                        examples[category][example_kind].append(
                            {
                                "episode_id": episode_id,
                                "semantic_scenario": scenario,
                                "opportunity_id": opportunity.opportunity_id,
                                "action_id": action.action_id,
                                "gold_valid": is_valid,
                                "normalized_drift_flags": sorted(normalized),
                            }
                        )

    category_payload: dict[str, object] = {}
    all_categories_calibrated = True
    for category in _DRIFT_CATEGORIES:
        detail = _drift_calibration_detail(counts[category])
        detail["examples"] = examples[category]
        category_payload[category] = detail
        all_categories_calibrated = all_categories_calibrated and bool(detail["calibrated"])

    scenario_payload: dict[str, object] = {}
    all_scenarios_calibrated = bool(scenario_counts)
    for scenario, by_category in sorted(scenario_counts.items()):
        categories = {
            category: _drift_calibration_detail(by_category[category])
            for category in _DRIFT_CATEGORIES
        }
        calibrated = all(bool(detail["calibrated"]) for detail in categories.values())
        all_scenarios_calibrated = all_scenarios_calibrated and calibrated
        scenario_payload[scenario] = {
            "calibrated": calibrated,
            "categories": categories,
        }

    return {
        "schema_version": 1,
        "scope": "policy_free_frozen_gold_checker_calibration",
        "n_episodes": len(specs),
        "n_eligible_opportunities": n_opportunities,
        "n_eligible_action_assignments": n_action_assignments,
        "all_categories_calibrated": all_categories_calibrated,
        "all_represented_scenarios_calibrated": all_scenarios_calibrated,
        "categories": category_payload,
        "semantic_scenarios": scenario_payload,
        "note": (
            "Calibration invokes the checker state predicates and normalized drift "
            "classifier directly; it makes no policy, writer, embedding, or reranker calls."
        ),
    }


def compute_measurement_gates(
    matrix: object,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    summary: Mapping[str, object],
    heuristic_baselines: Mapping[str, object],
    drift_calibration: Mapping[str, object] | None = None,
    expected_task_count: int | None = None,
    observations: Sequence[MultisystemMetricInput] = (),
) -> dict[str, object]:
    """Evaluate preregistered scientific gates without changing run results."""
    tasks = tuple(getattr(matrix, "task_results", ()))
    condition_records = tuple(
        (str(getattr(task, "episode_id", "")), condition)
        for task in tasks
        for condition in getattr(task, "condition_results", ())
    )
    condition_results = tuple(condition for _episode_id, condition in condition_records)
    memory_cells: dict[str, list[object]] = defaultdict(list)
    for task in tasks:
        policy_profile_id = str(getattr(task, "policy_profile_id", "unknown"))
        for condition in getattr(task, "condition_results", ()):
            condition_name = str(getattr(condition, "condition", ""))
            if condition_name not in _MEMORY_CONDITIONS:
                continue
            cell_id = "|".join(
                (
                    policy_profile_id,
                    condition_name,
                    str(getattr(condition, "readout", "none")),
                )
            )
            memory_cells[cell_id].extend(getattr(condition, "sceu_results", ()))
    memory_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) in _MEMORY_CONDITIONS
        for row in getattr(condition, "sceu_results", ())
    )
    oracle_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) == "oracle_current_state"
        for row in getattr(condition, "sceu_results", ())
    )
    oracle_records = tuple(
        (episode_id, row)
        for episode_id, condition in condition_records
        if str(getattr(condition, "condition", "")) == "oracle_current_state"
        for row in getattr(condition, "sceu_results", ())
    )
    sham = tuple(
        intervention
        for row in memory_rows
        for intervention in getattr(row, "interventions", ())
        if str(getattr(intervention, "intervention_kind", "")) == "sham_replacement"
    )
    lifecycle = summary.get("storage_provenance")
    semantic = summary.get("semantic_attribution")
    gates: list[dict[str, object]] = []

    evaluated_episode_ids = {
        str(getattr(task, "episode_id", "")) for task in tasks
    }
    construct_profiles = tuple(
        profile_sceu(spec.plan, sceu)
        for episode_id, spec in sorted(specs.items())
        if episode_id in evaluated_episode_ids
        for sceu in spec.plan.sceu_units
    )
    expected_construct_profiles = sum(
        len(spec.plan.sceu_units)
        for episode_id, spec in specs.items()
        if episode_id in evaluated_episode_ids
    )
    _gate_ratio(
        gates,
        "long_horizon_construct_profile_completeness",
        len(construct_profiles),
        expected_construct_profiles,
        minimum=1.0,
        description=(
            "Every evaluated continuation has an explicit handoff, dependency, "
            "transition, and workspace-recoverability profile."
        ),
    )
    future_overlap = {
        f"{profile.episode_id}|{profile.sceu_id}": sorted(
            set(profile.current_required_state_ids).intersection(
                profile.future_referenced_state_ids
            )
        )
        for profile in construct_profiles
        if set(profile.current_required_state_ids).intersection(
            profile.future_referenced_state_ids
        )
    }
    _gate_boolean(
        gates,
        "current_state_future_leakage",
        not future_overlap,
        applicable=bool(construct_profiles),
        description=(
            "No future state is counted as currently required at an early decision."
        ),
        detail={"overlap_by_sceu": future_overlap},
    )
    missing_action_state = {
        f"{profile.episode_id}|{profile.sceu_id}": list(
            profile.missing_current_action_relevant_state_ids
        )
        for profile in construct_profiles
        if profile.missing_current_action_relevant_state_ids
    }
    _gate_boolean(
        gates,
        "current_action_state_contract_completeness",
        not missing_action_state,
        applicable=bool(construct_profiles),
        description=(
            "Every current state atom that can make an offered executable "
            "action valid or invalid is included in the SCEU required-state "
            "closure; oracle controls therefore receive a sufficient state "
            "contract."
        ),
        detail={"missing_by_sceu": missing_action_state},
    )
    construct_counts = Counter(
        profile.construct_kind for profile in construct_profiles
    )
    required_constructs = {
        "static_recall",
        "state_evolution",
        "hierarchical_conflict",
    }
    _gate_boolean(
        gates,
        "long_horizon_construct_coverage",
        required_constructs.issubset(construct_counts),
        applicable=bool(construct_profiles),
        description=(
            "The evaluated release includes static, evolving-state, and "
            "hierarchical-conflict decisions under one task contract."
        ),
        detail=dict(sorted(construct_counts.items())),
    )

    evaluated_specs = tuple(
        spec
        for episode_id, spec in sorted(specs.items())
        if episode_id in evaluated_episode_ids
    )
    evaluated_horizon_panel_ids = {
        spec.plan.metadata_dict.get("horizon_panel_id", "")
        for spec in evaluated_specs
        if spec.plan.metadata_dict.get("horizon_panel_id", "")
    }
    horizon_release = bool(evaluated_horizon_panel_ids) and all(
        spec.plan.metadata_dict.get("horizon_panel_id", "")
        for spec in evaluated_specs
    )
    task_span_profiles = tuple(
        profile_task_span(spec.plan)
        for spec in evaluated_specs
        if spec.plan.task_steps
    )
    threshold_task_span_profiles = tuple(
        profile_task_span(spec.plan)
        for spec in evaluated_specs
        if spec.plan.task_steps
        and (
            not horizon_release
            or spec.plan.metadata_dict.get("horizon_level", "") == "long"
        )
    )
    _gate_ratio(
        gates,
        "task_span_provenance_completeness",
        len(task_span_profiles),
        sum(bool(spec.plan.task_steps) for spec in evaluated_specs),
        minimum=1.0,
        description=(
            "Every episode claiming an effective task span exposes step-level "
            "execution mode, dependency, session, state, and workspace provenance."
        ),
    )
    _gate_ratio(
        gates,
        "effective_long_horizon_step_threshold",
        sum(
            profile.meets_long_horizon_step_threshold
            for profile in threshold_task_span_profiles
        ),
        len(threshold_task_span_profiles),
        minimum=1.0,
        description=(
            "Each long-dose terminal decision has at least 200 effective causal "
            "ancestors whose semantic effects pass the anti-padding audit; "
            "short and medium members are comparison doses, and token count is "
            "not used as the horizon variable."
        ),
    )
    _gate_ratio(
        gates,
        "task_step_causal_linkage",
        sum(
            profile.causally_linked_step_fraction is not None
            and profile.causally_linked_step_fraction >= 0.99
            for profile in task_span_profiles
        ),
        len(task_span_profiles),
        minimum=1.0,
        description=(
            "At least 99% of effective steps in every long-horizon trace are roots "
            "or have an explicit dependency on an earlier effective step."
        ),
    )
    _gate_ratio(
        gates,
        "task_step_effect_chain_integrity",
        sum(profile.effect_chain_verified for profile in task_span_profiles),
        len(task_span_profiles),
        minimum=1.0,
        description=(
            "Every claimed effective step has a verified operation digest and "
            "the exact effect digests of its causal predecessors."
        ),
    )
    _gate_ratio(
        gates,
        "task_step_anti_padding_integrity",
        sum(profile.anti_padding_verified for profile in task_span_profiles),
        len(task_span_profiles),
        minimum=1.0,
        description=(
            "Every counted effective step produces a unique semantic task "
            "effect, and every pre-decision effect is consumed by a later step "
            "or scored continuation."
        ),
    )

    counterfactual_groups: dict[str, list[SoftwareMem0VerticalSpec]] = (
        defaultdict(list)
    )
    for spec in evaluated_specs:
        group_id = spec.plan.metadata_dict.get("counterfactual_group_id", "")
        if group_id:
            counterfactual_groups[group_id].append(spec)
    matched_release = bool(counterfactual_groups)
    matched_audits = tuple(
        audit_matched_construct_triplet(tuple(counterfactual_groups[group_id]))
        for group_id in sorted(counterfactual_groups)
    )
    matched_balance_applicable = (
        bool(counterfactual_groups)
        and (
            len(evaluated_horizon_panel_ids) >= 3
            if horizon_release
            else len(matched_audits) >= 3
        )
    )
    _gate_boolean(
        gates,
        "matched_construct_structural_invariance",
        bool(matched_audits) and all(audit.ok for audit in matched_audits),
        applicable=bool(counterfactual_groups),
        description=(
            "Every counterfactual triplet fixes the terminal decision, opaque "
            "options, and prefix/workspace shape while manipulating only the "
            "long-horizon construct."
        ),
        detail={audit.group_id: audit.to_dict() for audit in matched_audits},
    )
    _gate_boolean(
        gates,
        "matched_gold_action_balance",
        {
            action_id
            for audit in matched_audits
            for action_id in audit.gold_action_ids
        }
        == {"safe_v2_offline", "stale_v1", "cloud_shortcut"},
        applicable=matched_balance_applicable,
        description=(
            "At least three matched groups cover all three terminal gold actions, "
            "so no default-safe action can solve the construct comparison."
        ),
    )
    recoverability_by_group = {
        group_id: {
            spec.plan.metadata_dict.get("recoverability_variant", "")
            for spec in group_specs
        }
        for group_id, group_specs in counterfactual_groups.items()
    }
    _gate_boolean(
        gates,
        "matched_workspace_recoverability_balance",
        {
            next(iter(values))
            for values in recoverability_by_group.values()
            if len(values) == 1
        }
        == {"explicit", "derivable", "absent"}
        and all(len(values) == 1 for values in recoverability_by_group.values()),
        applicable=matched_balance_applicable,
        description=(
            "At least three matched groups cover explicit, derivable, and absent "
            "workspace recoverability, with one consistent variant per triplet."
        ),
        detail={
            group_id: sorted(values)
            for group_id, values in sorted(recoverability_by_group.items())
        },
    )
    matched_contrasts = compute_matched_construct_contrasts(observations)
    expected_matched_decisions = {
        (
            spec.plan.metadata_dict.get("counterfactual_group_id", ""),
            spec.plan.metadata_dict.get(
                "counterfactual_target_opportunity_id",
                "",
            ),
        )
        for spec in evaluated_specs
        if spec.plan.metadata_dict.get("counterfactual_group_id", "")
        and spec.plan.metadata_dict.get(
            "counterfactual_target_opportunity_id",
            "",
        )
    }
    evaluated_policy_profile_ids = {
        str(getattr(task, "policy_profile_id", "unknown"))
        for task in tasks
    }
    _gate_boolean(
        gates,
        "matched_construct_outcome_completeness",
        bool(matched_contrasts)
        and all(row.get("complete") is True for row in matched_contrasts),
        applicable=bool(counterfactual_groups),
        description=(
            "Every reported policy/backend/readout counterfactual cell contains "
            "static, state-evolution, and hierarchical-conflict outcomes."
        ),
        detail={"n_contrasts": len(matched_contrasts)},
    )
    workspace_adjusted_fields = (
        "state_evolution_penalty_excess_over_workspace",
        "hierarchical_conflict_penalty_excess_over_workspace",
    )
    _gate_boolean(
        gates,
        "matched_workspace_adjustment_available",
        bool(matched_contrasts)
        and all(
            row.get("workspace_matched_control_available") is True
            and all(
                isinstance(row.get(field), int | float)
                and not isinstance(row.get(field), bool)
                for field in workspace_adjusted_fields
            )
            for row in matched_contrasts
        ),
        applicable=bool(counterfactual_groups),
        description=(
            "Every matched construct cell has a same-policy workspace-only "
            "triplet and finite difference-in-differences, so a changed "
            "workspace surface is not attributed to the memory channel."
        ),
        detail={
            "required_fields": list(workspace_adjusted_fields),
            "n_contrasts": len(matched_contrasts),
        },
    )
    for condition, gate_id, description in (
        (
            "oracle_current_state",
            "matched_oracle_terminal_contract_solvability",
            (
                "For every evaluated policy, oracle current state solves the "
                "same terminal decision across static, state-evolution, and "
                "hierarchical-conflict histories."
            ),
        ),
        (
            "full_context",
            "matched_full_context_terminal_contract_solvability",
            (
                "For every evaluated policy, complete public history supports "
                "the same terminal decision across static, state-evolution, "
                "and hierarchical-conflict histories; otherwise a memory "
                "failure is confounded with history interpretation."
            ),
        ),
    ):
        control_detail = _matched_control_solvability_detail(
            matched_contrasts,
            condition=condition,
            expected_decisions=expected_matched_decisions,
            expected_policy_profile_ids=evaluated_policy_profile_ids,
        )
        _gate_boolean(
            gates,
            gate_id,
            bool(control_detail["all_cells_pass"]),
            applicable=matched_release,
            description=description,
            detail=control_detail,
        )

    drift_trajectory_payload = compute_drift_trajectory_report(observations)
    raw_drift_trajectories = drift_trajectory_payload.get("trajectories", ())
    drift_trajectories = tuple(
        row
        for row in raw_drift_trajectories
        if isinstance(row, Mapping)
    ) if isinstance(raw_drift_trajectories, Sequence) else ()
    repeated_any_by_category = Counter(
        str(row.get("drift_category", ""))
        for row in drift_trajectories
        if isinstance(
            (checkpoint_count := row.get("eligible_checkpoint_count")),
            int,
        )
        and not isinstance(checkpoint_count, bool)
        and checkpoint_count >= 2
    )
    repeated_by_category = Counter(
        str(row.get("drift_category", ""))
        for row in drift_trajectories
        if bool(row.get("lineage_backed"))
        and isinstance(
            (checkpoint_count := row.get("eligible_checkpoint_count")),
            int,
        )
        and not isinstance(checkpoint_count, bool)
        and checkpoint_count >= 2
    )
    anchored_by_category = Counter(
        str(row.get("drift_category", ""))
        for row in drift_trajectories
        if bool(row.get("lineage_backed"))
        and bool(row.get("drift_evaluable"))
    )
    category_only_by_category = Counter(
        str(row.get("drift_category", ""))
        for row in drift_trajectories
        if not bool(row.get("lineage_backed"))
        and isinstance(
            (checkpoint_count := row.get("eligible_checkpoint_count")),
            int,
        )
        and not isinstance(checkpoint_count, bool)
        and checkpoint_count >= 2
    )
    longitudinal_applicable = bool(repeated_any_by_category)
    _gate_boolean(
        gates,
        "longitudinal_drift_state_lineage_coverage",
        all(
            repeated_by_category[category] > 0
            and category_only_by_category[category] == 0
            for category in _DRIFT_CATEGORIES
        ),
        applicable=longitudinal_applicable,
        description=(
            "Every repeated-checkpoint trajectory used for a longitudinal claim "
            "is anchored to an explicit or evaluator-derived state lineage; "
            "category-only legacy trajectories remain descriptive only."
        ),
        detail={
            category: {
                "repeated_lineage_backed": repeated_by_category[category],
                "repeated_category_only": category_only_by_category[category],
            }
            for category in _DRIFT_CATEGORIES
        },
    )
    _gate_boolean(
        gates,
        "longitudinal_drift_repeated_checkpoint_coverage",
        all(repeated_by_category[category] > 0 for category in _DRIFT_CATEGORIES),
        applicable=longitudinal_applicable,
        description=(
            "Every claimed drift category is observed at two or more distinct "
            "eligible checkpoints within at least one episode/cell."
        ),
        detail={
            category: repeated_by_category[category]
            for category in _DRIFT_CATEGORIES
        },
    )
    _gate_boolean(
        gates,
        "longitudinal_drift_adherence_anchor_coverage",
        all(anchored_by_category[category] > 0 for category in _DRIFT_CATEGORIES),
        applicable=longitudinal_applicable,
        description=(
            "Every claimed drift category has at least one trajectory with prior "
            "adherence and a later eligible checkpoint, so onset is identifiable."
        ),
        detail={
            category: anchored_by_category[category]
            for category in _DRIFT_CATEGORIES
        },
    )
    control_coverage: dict[str, dict[str, int]] = {
        condition: {
            category: sum(
                str(row.get("condition", "")) == condition
                and str(row.get("drift_category", "")) == category
                and bool(row.get("lineage_backed"))
                and bool(row.get("drift_evaluable"))
                for row in drift_trajectories
            )
            for category in _DRIFT_CATEGORIES
        }
        for condition in _LONGITUDINAL_CONTROL_CONDITIONS
    }
    control_drift: dict[str, dict[str, int]] = {
        condition: {
            category: sum(
                str(row.get("condition", "")) == condition
                and str(row.get("drift_category", "")) == category
                and bool(row.get("lineage_backed"))
                and bool(row.get("drift_evaluable"))
                and bool(row.get("event_observed"))
                for row in drift_trajectories
            )
            for category in _DRIFT_CATEGORIES
        }
        for condition in _LONGITUDINAL_CONTROL_CONDITIONS
    }
    _gate_boolean(
        gates,
        "longitudinal_drift_control_cleanliness",
        all(
            control_coverage[condition][category] > 0
            and control_drift[condition][category] == 0
            for condition in _LONGITUDINAL_CONTROL_CONDITIONS
            for category in _DRIFT_CATEGORIES
        ),
        applicable=longitudinal_applicable,
        description=(
            "Oracle-current-state and full-context controls both cover every "
            "state-lineage drift category and show no adherence-to-violation "
            "transition; otherwise the observed drift is not memory-specific."
        ),
        detail={
            "evaluable_trajectory_count": control_coverage,
            "observed_drift_count": control_drift,
        },
    )

    memory_observations = tuple(
        row
        for row in observations
        if str(getattr(row, "condition", "")) in _MEMORY_CONDITIONS
        and str(getattr(row, "readout", "none")) != "none"
    )
    attributable = tuple(
        row
        for row in memory_observations
        if bool(getattr(row, "memory_reliant_state_ids", ()))
    )
    attribution_stages = tuple(
        row.decision_attribution().stage
        for row in attributable
    )
    supported_stages = {
        "storage_evidence_unavailable",
        "storage_failure",
        "retrieval_failure",
        "exposure_failure",
        "utilization_failure",
        "behavior_success_causal",
        "behavior_success_without_detected_unique_causal_effect",
        # Completed schema-v1 reports can still be audited.
        "behavior_success_without_detected_use",
        "behavior_success_unprobed",
    }
    _gate_ratio(
        gates,
        "decision_failure_attribution_completeness",
        sum(stage in supported_stages for stage in attribution_stages),
        len(attributable),
        minimum=1.0,
        description=(
            "Every memory-reliant decision has one explicit earliest failure or "
            "success stage on the stored-to-used chain."
        ),
    )
    _gate_ratio(
        gates,
        "decision_storage_evidence_availability",
        sum(stage != "storage_evidence_unavailable" for stage in attribution_stages),
        len(attributable),
        minimum=1.0,
        description=(
            "Every memory-reliant decision has native/exact or inventory-inferred "
            "storage evidence before a storage failure is attributed."
        ),
    )
    causal_evidence_violations = {
        str(getattr(row, "result_id", "")): sorted(
            set(getattr(row, "behaviorally_used_state_ids", ())).difference(
                getattr(row, "behaviorally_probed_state_ids", ())
            )
        )
        for row in memory_observations
        if set(getattr(row, "behaviorally_used_state_ids", ())).difference(
            getattr(row, "behaviorally_probed_state_ids", ())
        )
    }
    _gate_boolean(
        gates,
        "causal_use_evidence_consistency",
        not causal_evidence_violations,
        applicable=bool(memory_observations),
        description=(
            "A state is labelled behaviorally used only when a registered "
            "counterfactual probe targeted that state."
        ),
        detail={"violations_by_result": causal_evidence_violations},
    )

    completed = sum(
        str(getattr(task, "status", "")) == "complete"
        and all(
            str(getattr(condition, "status", "")) == "complete"
            for condition in getattr(task, "condition_results", ())
        )
        for task in tasks
    )
    planned_tasks = len(tasks) if expected_task_count is None else expected_task_count
    if planned_tasks < len(tasks):
        raise ValueError("expected_task_count cannot be smaller than observed task results")
    _gate_ratio(
        gates,
        "task_completion",
        completed,
        planned_tasks,
        minimum=1.0,
        description="All planned tasks and every nested condition completed.",
        detail={
            "observed_task_results": len(tasks),
            "missing_task_results": planned_tasks - len(tasks),
        },
    )
    _gate_ratio(
        gates,
        "memory_baseline_stability",
        sum(bool(getattr(row, "baseline_stable", False)) for row in memory_rows),
        len(memory_rows),
        minimum=BASELINE_STABILITY_MIN,
        description="Repeated memory-condition baselines agree.",
    )
    stability_by_cell = {
        cell_id: _rate_detail(
            sum(bool(getattr(row, "baseline_stable", False)) for row in rows),
            len(rows),
        )
        for cell_id, rows in sorted(memory_cells.items())
    }
    _gate_boolean(
        gates,
        "memory_baseline_stability_by_cell",
        bool(stability_by_cell)
        and all(
            float(detail["rate"]) >= BASELINE_STABILITY_MIN for detail in stability_by_cell.values()
        ),
        applicable=bool(stability_by_cell),
        description=(
            "Every policy/backend/readout cell meets the repeated-baseline stability threshold."
        ),
        detail=stability_by_cell,
    )
    sham_flips = sum(
        bool(getattr(getattr(item, "classification", None), "action_changed", False))
        for item in sham
    )
    _gate_ratio(
        gates,
        "sham_action_flip_rate",
        sham_flips,
        len(sham),
        maximum=SHAM_ACTION_FLIP_MAX,
        description=(
            "Count-, rank-, and length-matched neutral-A versus neutral-B sham "
            "replacements rarely flip actions."
        ),
    )
    _gate_scalar(
        gates,
        "sham_action_flip_upper_bound",
        _wilson_upper_bound(sham_flips, len(sham)),
        maximum=SHAM_ACTION_FLIP_MAX,
        description=(
            "The one-sided 95% Wilson upper bound for paired-neutral sham action "
            "flips is below the preregistered false-positive ceiling."
        ),
        detail={
            "numerator": sham_flips,
            "denominator": len(sham),
            "point_rate": None if not sham else sham_flips / len(sham),
            "confidence": 0.95,
        },
    )
    _gate_ratio(
        gates,
        "oracle_accuracy",
        sum(bool(getattr(row, "is_correct", False)) for row in oracle_rows),
        len(oracle_rows),
        minimum=ORACLE_ACCURACY_MIN,
        description="Oracle current state confirms task solvability.",
    )
    oracle_by_opportunity = _grouped_accuracy(
        (
            str(getattr(row, "opportunity_id", "unknown")),
            bool(getattr(row, "is_correct", False)),
        )
        for _episode_id, row in oracle_records
    )
    _gate_boolean(
        gates,
        "oracle_accuracy_by_opportunity",
        bool(oracle_by_opportunity)
        and all(
            float(detail["rate"]) >= ORACLE_GROUP_ACCURACY_MIN
            for detail in oracle_by_opportunity.values()
        ),
        applicable=bool(oracle_by_opportunity),
        description="Every continuation opportunity remains solvable with oracle state.",
        detail=oracle_by_opportunity,
    )
    scenario_by_episode = {
        episode_id: str(dict(spec.plan.metadata).get("semantic_scenario", "unknown"))
        for episode_id, spec in specs.items()
    }
    oracle_by_scenario = _grouped_accuracy(
        (
            scenario_by_episode.get(episode_id, "unknown"),
            bool(getattr(row, "is_correct", False)),
        )
        for episode_id, row in oracle_records
    )
    _gate_boolean(
        gates,
        "oracle_accuracy_by_scenario",
        bool(oracle_by_scenario)
        and all(
            float(detail["rate"]) >= ORACLE_GROUP_ACCURACY_MIN
            for detail in oracle_by_scenario.values()
        ),
        applicable=bool(oracle_by_scenario),
        description="Every semantic scenario remains solvable with oracle state.",
        detail=oracle_by_scenario,
    )
    _gate_boolean(
        gates,
        "lifecycle_provenance_complete",
        isinstance(lifecycle, Mapping) and lifecycle.get("status") == "complete",
        applicable=isinstance(lifecycle, Mapping) and bool(summary.get("n_inventory_snapshots", 0)),
        description="Every observed write has native or explicitly inferred lifecycle provenance.",
    )
    _gate_boolean(
        gates,
        "semantic_attribution_complete",
        isinstance(semantic, Mapping) and semantic.get("status") == "complete",
        applicable=isinstance(semantic, Mapping) and bool(semantic.get("n_memory_objects", 0)),
        description="Every final memory object has an explicit semantic-attribution method.",
    )
    semantic_methods = semantic.get("method_counts", {}) if isinstance(semantic, Mapping) else {}
    semantic_total = (
        int(semantic.get("n_memory_objects", 0)) if isinstance(semantic, Mapping) else 0
    )
    ambiguous = (
        int(semantic_methods.get("ambiguous", 0)) if isinstance(semantic_methods, Mapping) else 0
    )
    unavailable = (
        int(semantic_methods.get("unavailable", 0)) if isinstance(semantic_methods, Mapping) else 0
    )
    _gate_ratio(
        gates,
        "semantic_attribution_resolvability",
        max(0, semantic_total - ambiguous - unavailable),
        semantic_total,
        minimum=SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN,
        description=(
            "Semantic attribution resolves each object as a fact match or a supported "
            "no-match without evaluator ambiguity."
        ),
    )
    stored_lifecycle = (
        semantic.get("lifecycle_provenance_counts", {})
        if isinstance(semantic, Mapping)
        else {}
    )
    stored_exact = (
        int(stored_lifecycle.get("native/exact", 0))
        if isinstance(stored_lifecycle, Mapping)
        else 0
    )
    stored_inferred = (
        int(stored_lifecycle.get("inferred", 0))
        if isinstance(stored_lifecycle, Mapping)
        else 0
    )
    stored_unavailable = (
        int(stored_lifecycle.get("unavailable", 0))
        if isinstance(stored_lifecycle, Mapping)
        else semantic_total
    )
    _gate_boolean(
        gates,
        "stored_object_provenance_complete",
        stored_unavailable == 0 and stored_exact + stored_inferred == semantic_total,
        applicable=semantic_total > 0,
        description=(
            "Every object in the final native inventories has exact or explicitly "
            "inferred lifecycle provenance."
        ),
        detail={
            "native_exact": stored_exact,
            "inferred": stored_inferred,
            "unavailable": stored_unavailable,
            "total": semantic_total,
        },
    )
    best_accuracy = heuristic_baselines.get("best_always_action_accuracy")
    _gate_scalar(
        gates,
        "action_dominance",
        (
            best_accuracy
            if not counterfactual_groups or matched_balance_applicable
            else None
        ),
        maximum=MAX_ALWAYS_ACTION_ACCURACY,
        description="No fixed action solves more than the preregistered share.",
    )
    best_option_accuracy = heuristic_baselines.get("best_always_option_accuracy")
    _gate_scalar(
        gates,
        "option_dominance",
        (
            best_option_accuracy
            if not counterfactual_groups or matched_balance_applicable
            else None
        ),
        maximum=MAX_ALWAYS_OPTION_ACCURACY,
        description="No fixed opaque option position solves the benchmark.",
    )

    eligible_counts = Counter(
        category
        for row in (
            oracle_rows
            or tuple(
                row
                for condition in condition_results
                if str(getattr(condition, "condition", "")) == "workspace_only"
                for row in getattr(condition, "sceu_results", ())
            )
        )
        for category in (getattr(row, "drift_eligible_categories", ()) or ())
    )
    _gate_boolean(
        gates,
        "drift_category_exposure",
        all(eligible_counts[category] > 0 for category in _DRIFT_CATEGORIES),
        applicable=bool(oracle_rows or condition_results),
        description="Every canonical drift category has an eligible opportunity.",
        detail={category: eligible_counts[category] for category in _DRIFT_CATEGORIES},
    )
    calibration = (
        drift_calibration
        if drift_calibration is not None
        else compute_drift_action_calibration(specs)
    )
    _gate_boolean(
        gates,
        "drift_action_calibration",
        calibration.get("all_categories_calibrated") is True
        and (
            matched_release
            or calibration.get("all_represented_scenarios_calibrated") is True
        ),
        applicable=bool(specs),
        description=(
            "Every drift construct has checker-positive and checker-negative actions, "
            "including an invalid positive and no gold-valid false positive, within "
            "the declared release unit. Standard trajectories require this within "
            "every semantic scenario; matched triplets require it across the balanced "
            "counterfactual release because terminal archetypes are rotated by group."
        ),
        detail={
            "all_categories_calibrated": calibration.get("all_categories_calibrated"),
            "all_represented_scenarios_calibrated": calibration.get(
                "all_represented_scenarios_calibrated"
            ),
            "calibration_unit": (
                "matched_counterfactual_release"
                if matched_release
                else "semantic_scenario"
            ),
        },
    )

    divergence_by_episode = _control_action_divergence(condition_records)
    _gate_boolean(
        gates,
        "workspace_oracle_action_separation",
        bool(divergence_by_episode)
        and min(divergence_by_episode.values()) >= MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE,
        applicable=bool(divergence_by_episode) and not matched_release,
        description="Workspace-only and oracle require distinct behavior within every episode.",
        detail=dict(sorted(divergence_by_episode.items())),
    )
    matched_divergence = _matched_control_action_divergence(
        divergence_by_episode,
        specs,
    )
    raw_absent_group_divergence = matched_divergence[
        "absent_group_divergence"
    ]
    absent_group_divergence = (
        raw_absent_group_divergence
        if isinstance(raw_absent_group_divergence, Mapping)
        else {}
    )
    _gate_boolean(
        gates,
        "matched_workspace_oracle_action_separation",
        bool(absent_group_divergence)
        and all(
            isinstance(count, int) and count >= 1
            for count in absent_group_divergence.values()
        ),
        applicable=matched_release and bool(divergence_by_episode),
        description=(
            "Every workspace-absent counterfactual group contains at least one "
            "terminal decision on which workspace-only and oracle choose different "
            "actions. The triplet, rather than a one-decision physical member, is "
            "the unit for this gate."
        ),
        detail=matched_divergence,
    )
    drift_separation = _control_drift_separation(condition_records)
    _gate_boolean(
        gates,
        "workspace_oracle_drift_separation",
        all(drift_separation[category] > 0 for category in _DRIFT_CATEGORIES),
        applicable=(
            bool(drift_separation.get("matched_pairs", 0))
            and not matched_release
        ),
        description=(
            "Workspace-only produces every canonical behavioral-drift construct at "
            "least once while the matched oracle continuation does not."
        ),
        detail=dict(drift_separation),
    )
    flat_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) == "flat_retrieval"
        for row in getattr(condition, "sceu_results", ())
    )
    flat_probed_rows = sum(
        any(
            str(getattr(item, "intervention_kind", "")) == "neutral_replacement"
            for item in getattr(row, "interventions", ())
        )
        for row in flat_rows
    )
    _gate_ratio(
        gates,
        "flat_causal_probe_coverage",
        flat_probed_rows,
        len(flat_rows),
        minimum=FLAT_CAUSAL_PROBE_COVERAGE_MIN,
        description=(
            "The controlled flat-retrieval condition exposes enough visible focal "
            "memories for matched causal probes."
        ),
    )
    causal_chains_by_cell = {
        cell_id: sum(bool(getattr(row, "behaviorally_used_memory_ids", ())) for row in rows)
        for cell_id, rows in sorted(memory_cells.items())
    }
    flat_causal_chains = sum(
        bool(getattr(row, "behaviorally_used_memory_ids", ())) for row in flat_rows
    )
    total_causal_chains = sum(causal_chains_by_cell.values())
    _gate_boolean(
        gates,
        "stored_retrieved_visible_behavior_chain",
        total_causal_chains > 0,
        applicable=bool(memory_rows),
        description=(
            "At least one preregistered memory condition establishes a stable "
            "stored-to-retrieved-to-visible-to-behavior chain. Flat retrieval remains "
            "reported separately and is not assumed to be behaviorally sufficient."
        ),
        detail={
            "all_memory_qualifying_sceu": total_causal_chains,
            "flat_qualifying_sceu": flat_causal_chains,
            "qualifying_sceu_by_cell": causal_chains_by_cell,
        },
    )

    ready = all(item["status"] in {"pass", "not_applicable"} for item in gates)
    return {
        "schema_version": 1,
        "measurement_ready": ready,
        "gate_counts": dict(Counter(str(item["status"]) for item in gates)),
        "gates": gates,
        "thresholds": {
            "baseline_stability_min": BASELINE_STABILITY_MIN,
            "sham_action_flip_max": SHAM_ACTION_FLIP_MAX,
            "oracle_accuracy_min": ORACLE_ACCURACY_MIN,
            "oracle_group_accuracy_min": ORACLE_GROUP_ACCURACY_MIN,
            "max_always_action_accuracy": MAX_ALWAYS_ACTION_ACCURACY,
            "max_always_option_accuracy": MAX_ALWAYS_OPTION_ACCURACY,
            "min_control_action_divergences_per_episode": (
                MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE
            ),
            "min_control_action_divergences_per_absent_counterfactual_group": 1,
            "semantic_attribution_resolvability_min": (SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN),
            "flat_causal_probe_coverage_min": FLAT_CAUSAL_PROBE_COVERAGE_MIN,
        },
        "note": (
            "Artifact validation and measurement readiness are separate. A complete "
            "run remains auditable even when a scientific gate fails."
        ),
    }


def heuristic_baselines_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Policy-free heuristic baselines",
        "",
        "These controls use no workspace, history, memory, or model calls.",
        "",
        "| Heuristic | Accuracy |",
        "|---|---:|",
    ]
    for action_id, value in _mapping(payload.get("always_action_accuracy")).items():
        lines.append(f"| always action: `{action_id}` | {_format_rate(value)} |")
    for option_id, value in _mapping(payload.get("always_option_accuracy")).items():
        lines.append(f"| always option: `{option_id}` | {_format_rate(value)} |")
    lines.append(
        "| uniform random expected | "
        f"{_format_rate(payload.get('uniform_random_expected_accuracy'))} |"
    )
    lines.append("")
    return "\n".join(lines)


def drift_action_calibration_markdown(payload: Mapping[str, object]) -> str:
    """Render the policy-free checker calibration as a compact audit table."""
    lines = [
        "# Policy-free drift action calibration",
        "",
        (
            "This audit applies the same latent-state predicates and normalized drift "
            "classifier used for scored continuations to every catalog action."
        ),
        "",
        "| Drift construct | Eligible opportunities | Positive | Negative | "
        "Invalid positive | Gold-valid false positive | Calibrated |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    categories = _mapping(payload.get("categories"))
    for category in _DRIFT_CATEGORIES:
        detail = _mapping(categories.get(category))
        lines.append(
            "| `{category}` | {eligible} | {positive} | {negative} | {invalid} | "
            "{valid_positive} | {calibrated} |".format(
                category=category,
                eligible=detail.get("eligible_opportunities", 0),
                positive=detail.get("positive_assignments", 0),
                negative=detail.get("negative_assignments", 0),
                invalid=detail.get("invalid_positive_assignments", 0),
                valid_positive=detail.get("valid_positive_assignments", 0),
                calibrated=str(detail.get("calibrated", False)).lower(),
            )
        )
    lines.extend(
        (
            "",
            "All categories calibrated: "
            f"**{str(payload.get('all_categories_calibrated', False)).lower()}**.",
            "",
            "All represented semantic scenarios calibrated: "
            "**"
            f"{str(payload.get('all_represented_scenarios_calibrated', False)).lower()}"
            "**.",
            "",
        )
    )
    return "\n".join(lines)


def measurement_gates_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Measurement readiness gates",
        "",
        f"Overall measurement ready: **{str(payload.get('measurement_ready', False)).lower()}**.",
        "",
        "| Gate | Status | Value | Requirement |",
        "|---|---|---:|---|",
    ]
    for item in _sequence_of_mappings(payload.get("gates")):
        lines.append(
            "| `{gate}` | {status} | {value} | {requirement} |".format(
                gate=item.get("gate_id", ""),
                status=item.get("status", ""),
                value=_format_rate(item.get("value")),
                requirement=item.get("requirement", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _control_action_divergence(
    condition_records: Sequence[tuple[str, object]],
) -> dict[str, int]:
    selected: dict[tuple[str, str, str], str] = {}
    for episode_id, condition in condition_records:
        name = str(getattr(condition, "condition", ""))
        if name not in {"workspace_only", "oracle_current_state"}:
            continue
        for row in getattr(condition, "sceu_results", ()):
            selected[(episode_id, name, str(getattr(row, "opportunity_id", "")))] = str(
                getattr(row, "selected_action_id", "")
            )
    by_episode: dict[str, int] = defaultdict(int)
    episodes = {key[0] for key in selected}
    opportunities = {key[2] for key in selected}
    for episode_id in episodes:
        for opportunity_id in opportunities:
            workspace = selected.get((episode_id, "workspace_only", opportunity_id))
            oracle = selected.get((episode_id, "oracle_current_state", opportunity_id))
            if workspace is None or oracle is None:
                continue
            by_episode.setdefault(episode_id, 0)
            if workspace != oracle:
                by_episode[episode_id] += 1
    return dict(by_episode)


def _matched_control_action_divergence(
    divergence_by_episode: Mapping[str, int],
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    group_divergence: Counter[str] = Counter()
    recoverability_by_group: dict[str, set[str]] = defaultdict(set)
    for episode_id, spec in specs.items():
        metadata = spec.plan.metadata_dict
        group_id = metadata.get("counterfactual_group_id", "")
        if not group_id:
            continue
        group_divergence[group_id] += divergence_by_episode.get(episode_id, 0)
        recoverability_by_group[group_id].add(
            metadata.get("recoverability_variant", "")
        )
    absent_groups = tuple(
        sorted(
            group_id
            for group_id, variants in recoverability_by_group.items()
            if variants == {"absent"}
        )
    )
    return {
        "group_divergence": {
            group_id: group_divergence[group_id]
            for group_id in sorted(recoverability_by_group)
        },
        "recoverability_by_group": {
            group_id: sorted(variants)
            for group_id, variants in sorted(recoverability_by_group.items())
        },
        "absent_groups": list(absent_groups),
        "absent_group_divergence": {
            group_id: group_divergence[group_id]
            for group_id in absent_groups
        },
    }


def _control_drift_separation(
    condition_records: Sequence[tuple[str, object]],
) -> dict[str, int]:
    flags: dict[tuple[str, str, str], frozenset[str]] = {}
    for episode_id, condition in condition_records:
        name = str(getattr(condition, "condition", ""))
        if name not in {"workspace_only", "oracle_current_state"}:
            continue
        for row in getattr(condition, "sceu_results", ()):
            flags[(episode_id, name, str(getattr(row, "opportunity_id", "")))] = frozenset(
                str(value)
                for value in (getattr(row, "normalized_drift_flags", ()) or ())
                if str(value) in _DRIFT_CATEGORIES
            )
    counts: Counter[str] = Counter()
    matched = 0
    keys = {(episode_id, opportunity_id) for episode_id, _name, opportunity_id in flags}
    for episode_id, opportunity_id in sorted(keys):
        workspace = flags.get((episode_id, "workspace_only", opportunity_id))
        oracle = flags.get((episode_id, "oracle_current_state", opportunity_id))
        if workspace is None or oracle is None:
            continue
        matched += 1
        for category in _DRIFT_CATEGORIES:
            counts[category] += category in workspace and category not in oracle
    return {
        "matched_pairs": matched,
        **{category: counts[category] for category in _DRIFT_CATEGORIES},
    }


def _grouped_accuracy(
    values: Iterable[tuple[str, bool]],
) -> dict[str, dict[str, int | float]]:
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for group, is_correct in values:
        total[group] += 1
        correct[group] += bool(is_correct)
    return {group: _rate_detail(correct[group], total[group]) for group in sorted(total)}


def _matched_control_solvability_detail(
    contrasts: Sequence[Mapping[str, object]],
    *,
    condition: str,
    expected_decisions: set[tuple[str, str]],
    expected_policy_profile_ids: set[str],
) -> dict[str, object]:
    """Audit a matched full-information control separately for each policy.

    Pooling policies could let one capable policy hide another policy that
    cannot interpret even the oracle state or complete public history.  Such a
    cell cannot support a memory-channel attribution, so each policy must cover
    every frozen group/decision and meet the threshold in all three history
    variants.
    """

    rows = tuple(
        row for row in contrasts if str(row.get("condition", "")) == condition
    )
    rows_by_policy: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_policy[str(row.get("policy_profile_id", "unknown"))].append(row)

    policy_ids = sorted(expected_policy_profile_ids or rows_by_policy)
    cells: dict[str, dict[str, object]] = {}
    for policy_profile_id in policy_ids:
        policy_rows = rows_by_policy.get(policy_profile_id, [])
        observed_decisions = {
            (
                str(row.get("counterfactual_group_id", "")),
                str(row.get("opportunity_id", "")),
            )
            for row in policy_rows
        }
        readouts = sorted({str(row.get("readout", "")) for row in policy_rows})
        variant_rates: dict[str, float | None] = {}
        for variant in _MATCHED_CONTROL_VARIANTS:
            field = (
                "hierarchical_conflict_correct_rate"
                if variant == "hierarchical_conflict"
                else f"{variant}_correct_rate"
            )
            values = tuple(
                float(value)
                for row in policy_rows
                if isinstance((value := row.get(field)), int | float)
                and not isinstance(value, bool)
            )
            variant_rates[variant] = (
                None
                if len(values) != len(policy_rows) or not values
                else sum(values) / len(values)
            )
        complete = (
            bool(policy_rows)
            and all(row.get("complete") is True for row in policy_rows)
            and observed_decisions == expected_decisions
            and len(policy_rows) == len(expected_decisions)
            and readouts == ["none"]
        )
        variant_thresholds_pass = all(
            rate is not None and rate >= ORACLE_GROUP_ACCURACY_MIN
            for rate in variant_rates.values()
        )
        cell_pass = complete and variant_thresholds_pass
        cells[policy_profile_id] = {
            "status": "pass" if cell_pass else "fail",
            "n_expected_decisions": len(expected_decisions),
            "n_observed_decisions": len(observed_decisions),
            "missing_decisions": [
                list(key) for key in sorted(expected_decisions - observed_decisions)
            ],
            "unexpected_decisions": [
                list(key) for key in sorted(observed_decisions - expected_decisions)
            ],
            "readouts": readouts,
            "all_rows_complete": bool(policy_rows)
            and all(row.get("complete") is True for row in policy_rows),
            "variant_correct_rates": variant_rates,
            "minimum_variant_correct_rate": ORACLE_GROUP_ACCURACY_MIN,
            "state_evolution_correctness_penalty_vs_static": (
                _optional_rate_difference(
                    variant_rates["static"],
                    variant_rates["evolution"],
                )
            ),
            "hierarchical_conflict_correctness_penalty_vs_static": (
                _optional_rate_difference(
                    variant_rates["static"],
                    variant_rates["hierarchical_conflict"],
                )
            ),
        }

    return {
        "condition": condition,
        "threshold": ORACLE_GROUP_ACCURACY_MIN,
        "n_expected_policies": len(policy_ids),
        "n_observed_rows": len(rows),
        "expected_decisions": [list(key) for key in sorted(expected_decisions)],
        "missing_policy_profile_ids": sorted(
            set(policy_ids) - set(rows_by_policy)
        ),
        "all_cells_pass": bool(cells)
        and all(cell["status"] == "pass" for cell in cells.values()),
        "cells": cells,
    }


def _optional_rate_difference(
    minuend: float | None,
    subtrahend: float | None,
) -> float | None:
    if minuend is None or subtrahend is None:
        return None
    return minuend - subtrahend


def _rate_detail(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": 0.0 if denominator == 0 else numerator / denominator,
    }


def _drift_calibration_detail(counts: Mapping[str, int]) -> dict[str, object]:
    values = {
        key: int(counts.get(key, 0))
        for key in (
            "eligible_opportunities",
            "action_assignments",
            "positive_assignments",
            "negative_assignments",
            "valid_action_assignments",
            "invalid_action_assignments",
            "invalid_positive_assignments",
            "valid_positive_assignments",
        )
    }
    calibrated = (
        values["eligible_opportunities"] > 0
        and values["positive_assignments"] > 0
        and values["negative_assignments"] > 0
        and values["invalid_positive_assignments"] > 0
        and values["valid_positive_assignments"] == 0
    )
    detail: dict[str, object] = dict(values)
    detail["calibrated"] = calibrated
    return detail


def _wilson_upper_bound(successes: int, total: int) -> float | None:
    """Return the one-sided 95% Wilson upper confidence bound."""
    if total <= 0:
        return None
    proportion = successes / total
    z2 = _ONE_SIDED_95_Z * _ONE_SIDED_95_Z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = _ONE_SIDED_95_Z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return min(1.0, (center + radius) / denominator)


def _gate_ratio(
    gates: list[dict[str, object]],
    gate_id: str,
    numerator: int,
    denominator: int,
    *,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
    detail: Mapping[str, object] | None = None,
) -> None:
    value = None if denominator == 0 else numerator / denominator
    ratio_detail: dict[str, object] = {
        "numerator": numerator,
        "denominator": denominator,
    }
    if detail is not None:
        ratio_detail.update(detail)
    _gate_scalar(
        gates,
        gate_id,
        value,
        minimum=minimum,
        maximum=maximum,
        description=description,
        detail=ratio_detail,
    )


def _gate_scalar(
    gates: list[dict[str, object]],
    gate_id: str,
    value: object,
    *,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
    detail: Mapping[str, object] | None = None,
) -> None:
    numeric = float(value) if isinstance(value, int | float) else None
    passed = numeric is not None
    requirements: list[str] = []
    if minimum is not None:
        requirements.append(f">= {minimum:.3f}")
        passed = passed and numeric >= minimum  # type: ignore[operator]
    if maximum is not None:
        requirements.append(f"<= {maximum:.3f}")
        passed = passed and numeric <= maximum  # type: ignore[operator]
    gates.append(
        {
            "gate_id": gate_id,
            "status": "pass" if passed else ("not_applicable" if numeric is None else "fail"),
            "value": numeric,
            "requirement": " and ".join(requirements),
            "description": description,
            "detail": dict(detail or {}),
        }
    )


def _gate_boolean(
    gates: list[dict[str, object]],
    gate_id: str,
    passed: bool,
    *,
    applicable: bool,
    description: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    gates.append(
        {
            "gate_id": gate_id,
            "status": "pass" if passed else ("fail" if applicable else "not_applicable"),
            "value": passed if applicable else None,
            "requirement": "true",
            "description": description,
            "detail": dict(detail or {}),
        }
    )


def _best_baseline(values: Mapping[str, float]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    key = min(values, key=lambda item: (-values[item], item))
    return key, values[key]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): child for key, child in sorted(value.items())}


def _sequence_of_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _format_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "—"
    return f"{float(value):.4f}"


__all__ = [
    "BASELINE_STABILITY_MIN",
    "FLAT_CAUSAL_PROBE_COVERAGE_MIN",
    "MAX_ALWAYS_ACTION_ACCURACY",
    "MAX_ALWAYS_OPTION_ACCURACY",
    "MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE",
    "ORACLE_ACCURACY_MIN",
    "ORACLE_GROUP_ACCURACY_MIN",
    "SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN",
    "SHAM_ACTION_FLIP_MAX",
    "compute_drift_action_calibration",
    "compute_heuristic_baselines",
    "compute_measurement_gates",
    "drift_action_calibration_markdown",
    "heuristic_baselines_markdown",
    "measurement_gates_markdown",
]
