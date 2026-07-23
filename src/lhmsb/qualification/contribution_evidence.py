"""Machine-readable evidence contracts for BEAM's three contributions.

The artifact produced here does not decide whether a memory backend is better.
It records whether a report contains the controls, estimands, provenance, and
measurement gates needed to make each *kind* of scientific claim.  Effect
direction, uncertainty, and significance remain in the statistical reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from lhmsb.qualification.horizon_panel import HORIZON_PRIMARY_ESTIMANDS
from lhmsb.qualification.statistics import MATCHED_PRIMARY_ESTIMANDS

ContributionEvidenceStatus = Literal[
    "ready",
    "not_ready",
    "not_applicable",
]

CONTRIBUTION_EVIDENCE_SCHEMA_VERSION = 6

_C1_COMMON_GATES = (
    "task_completion",
    "current_action_state_contract_completeness",
    "oracle_accuracy",
    "oracle_accuracy_by_opportunity",
    "oracle_accuracy_by_scenario",
)
_C1_STANDARD_GATES = (
    "workspace_oracle_action_separation",
)
_C1_MATCHED_GATES = (
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
)
_C2_COMMON_GATES = (
    "drift_category_exposure",
    "drift_action_calibration",
)
_C2_STANDARD_GATES = (
    "workspace_oracle_drift_separation",
)
_C2_LONGITUDINAL_GATES = (
    "longitudinal_drift_state_lineage_coverage",
    "longitudinal_drift_repeated_checkpoint_coverage",
    "longitudinal_drift_adherence_anchor_coverage",
    "longitudinal_drift_control_cleanliness",
)
_C3_GATES = (
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


def build_contribution_evidence(
    *,
    summary: Mapping[str, object],
    measurement_gates: Mapping[str, object],
    matched_statistics: Mapping[str, object],
    drift_trajectories: Mapping[str, object],
    decision_attribution_rows: Sequence[Mapping[str, object]],
    fault_profile_divergence: Mapping[str, object],
    experiment_design_audit: Mapping[str, object],
    horizon_statistics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the report-level claim-to-evidence contract.

    A ``ready`` contribution means that its measurement evidence is present
    and its required gates pass.  It does not mean that the measured effect is
    positive, statistically significant, or confirmatory.
    """

    gate_statuses = _gate_statuses(measurement_gates)
    construct_mode = str(summary.get("construct_mode", "standard"))
    analysis_timing = str(
        summary.get("analysis_timing", "pre_specified")
    )
    horizon = construct_mode == "horizon_panels"
    matched = construct_mode in {"matched_triplets", "horizon_panels"}
    pre_call_contract_statuses = _contribution_contract_statuses(
        experiment_design_audit
    )
    contribution_design_requirements = _contribution_design_requirements(
        experiment_design_audit
    )
    contribution_claim_scopes = _contribution_claim_scopes(
        experiment_design_audit
    )
    design_check_statuses = _design_check_statuses(experiment_design_audit)
    pre_call_contract_required = analysis_timing == "pre_specified"
    statistic_metrics = _statistic_metric_names(
        horizon_statistics or {} if horizon else matched_statistics
    )

    c1_required_gates = (
        (*_C1_COMMON_GATES, *_C1_MATCHED_GATES)
        if matched
        else (*_C1_COMMON_GATES, *_C1_STANDARD_GATES)
    )
    c1_required_estimands = (
        HORIZON_PRIMARY_ESTIMANDS
        if horizon
        else (MATCHED_PRIMARY_ESTIMANDS if matched else ())
    )
    c1_missing_estimands = tuple(
        metric
        for metric in c1_required_estimands
        if metric not in statistic_metrics
    )
    c1_missing_design_requirements = tuple(
        item
        for item, missing in (
            (
                "balanced_mechanism_design_ready",
                matched
                and experiment_design_audit.get(
                    "balanced_mechanism_design_ready"
                )
                is not True,
            ),
            (
                "C1_pre_call_analysis_contract",
                pre_call_contract_required
                and pre_call_contract_statuses["C1"]
                != "pre_call_frozen",
            ),
        )
        if missing
    )
    c1_applicable = bool(
        _nonnegative_int(summary.get("n_horizon_panel_contrasts"))
        if horizon
        else (
            _nonnegative_int(summary.get("n_long_horizon_control_contrasts"))
            or _nonnegative_int(summary.get("n_matched_construct_contrasts"))
        )
    )

    longitudinal_statuses = tuple(
        gate_statuses.get(gate_id, "missing")
        for gate_id in _C2_LONGITUDINAL_GATES
    )
    longitudinal_applicable = any(
        gate_id in gate_statuses
        and gate_statuses[gate_id] != "not_applicable"
        for gate_id in _C2_LONGITUDINAL_GATES
    )
    drift_mode = (
        "state_lineage_longitudinal_drift"
        if longitudinal_applicable
        else "endpoint_violation_only"
    )
    c2_required_gates = (
        (
            *_C2_COMMON_GATES,
            *(() if matched else _C2_STANDARD_GATES),
            *_C2_LONGITUDINAL_GATES,
        )
        if longitudinal_applicable
        else (
            *_C2_COMMON_GATES,
            *(() if matched else _C2_STANDARD_GATES),
        )
    )
    raw_drift_trajectory_rows = drift_trajectories.get("trajectories")
    drift_trajectory_rows = tuple(
        row
        for row in (
            raw_drift_trajectory_rows
            if isinstance(raw_drift_trajectory_rows, Sequence)
            and not isinstance(raw_drift_trajectory_rows, (str, bytes))
            else ()
        )
        if isinstance(row, Mapping)
    )
    n_drift_trajectories = len(drift_trajectory_rows)
    n_lineage_backed_trajectories = sum(
        row.get("lineage_backed") is True for row in drift_trajectory_rows
    )
    n_category_only_trajectories = sum(
        row.get("lineage_backed") is False for row in drift_trajectory_rows
    )
    n_category_only_evaluable = sum(
        row.get("lineage_backed") is False
        and row.get("drift_evaluable") is True
        for row in drift_trajectory_rows
    )
    n_lineage_backed_evaluable = sum(
        row.get("lineage_backed") is True
        and row.get("drift_evaluable") is True
        for row in drift_trajectory_rows
    )
    n_lineage_backed_observed = sum(
        row.get("lineage_backed") is True
        and row.get("event_observed") is True
        for row in drift_trajectory_rows
    )
    control_observed_by_condition = {
        condition: sum(
            str(row.get("condition", "")) == condition
            and row.get("lineage_backed") is True
            and row.get("event_observed") is True
            for row in drift_trajectory_rows
        )
        for condition in ("oracle_current_state", "full_context")
    }
    c2_missing_estimands = (
        tuple(
            item
            for item, missing in (
                (
                    "state_lineage_anchoring",
                    n_lineage_backed_evaluable == 0,
                ),
                (
                    "category_only_legacy_trajectories_excluded",
                    n_category_only_evaluable > 0,
                ),
            )
            if missing
        )
        if longitudinal_applicable
        else ()
    )
    c2_missing_design_requirements = tuple(
        item
        for item, missing in (
            (
                "C2_pre_call_analysis_contract",
                pre_call_contract_required
                and pre_call_contract_statuses["C2"] != "pre_call_frozen",
            ),
            (
                "C2_pre_call_longitudinal_scope",
                pre_call_contract_required
                and longitudinal_applicable
                and contribution_claim_scopes["C2"]
                != "state_lineage_longitudinal",
            ),
            *(
                (
                    check_id,
                    design_check_statuses.get(check_id) != "pass",
                )
                for check_id in contribution_design_requirements["C2"]
            ),
        )
        if missing
    )

    n_decisions = len(decision_attribution_rows)
    observed_stages = tuple(
        sorted(
            {
                str(row.get("stage", ""))
                for row in decision_attribution_rows
                if str(row.get("stage", ""))
            }
        )
    )
    n_aligned_fault_pairs = _nonnegative_int(
        fault_profile_divergence.get("n_aligned_decision_pairs")
    )
    n_outcome_equivalent_fault_pairs = _nonnegative_int(
        fault_profile_divergence.get("n_outcome_equivalent_pairs")
    )
    c3_missing_estimands = (
        ("outcome_equivalent_fault_profile_divergence",)
        if n_decisions > 0 and n_outcome_equivalent_fault_pairs == 0
        else ()
    )
    c3_missing_design_requirements = tuple(
        item
        for item, missing in (
            (
                "C3_pre_call_analysis_contract",
                pre_call_contract_required
                and pre_call_contract_statuses["C3"] != "pre_call_frozen",
            ),
            *(
                (
                    check_id,
                    design_check_statuses.get(check_id) != "pass",
                )
                for check_id in contribution_design_requirements["C3"]
            ),
        )
        if missing
    )

    contributions = (
        {
            "contribution_id": "C1",
            "claim_timing": analysis_timing,
            "pre_call_contract_status": pre_call_contract_statuses["C1"],
            "name": (
                "counterfactually_identified_workspace_adjusted_long_horizon_"
                "state_control"
            ),
            "evidence_status": _evidence_status(
                applicable=c1_applicable,
                required_gate_ids=c1_required_gates,
                gate_statuses=gate_statuses,
                missing_required_items=(
                    *c1_missing_estimands,
                    *c1_missing_design_requirements,
                ),
            ),
            "claim_scope": (
                "same_decision_horizon_amplification_beyond_workspace"
                if horizon
                else (
                    "matched_workspace_adjusted_mechanism"
                    if matched
                    else "paired_value_beyond_workspace"
                )
            ),
            "required_gate_ids": list(c1_required_gates),
            "gate_statuses": _selected_gate_statuses(
                c1_required_gates,
                gate_statuses,
            ),
            "required_estimands": list(c1_required_estimands),
            "observed_estimands": sorted(
                set(c1_required_estimands).intersection(statistic_metrics)
            ),
            "missing_estimands": list(c1_missing_estimands),
            "missing_design_requirements": list(
                c1_missing_design_requirements
            ),
            "evidence_counts": {
                "long_horizon_control_contrasts": _nonnegative_int(
                    summary.get("n_long_horizon_control_contrasts")
                ),
                "matched_construct_contrasts": _nonnegative_int(
                    summary.get("n_matched_construct_contrasts")
                ),
                "counterfactual_groups": _nonnegative_int(
                    summary.get("n_counterfactual_groups")
                ),
                "horizon_panels": _nonnegative_int(
                    summary.get("n_horizon_panels")
                ),
                "design_audit_status": str(
                    experiment_design_audit.get("audit_status", "missing")
                ),
                "trajectory_interaction_mode_counts": (
                    experiment_design_audit.get(
                        "trajectory_interaction_mode_counts",
                        {},
                    )
                ),
                "online_long_horizon_agent_execution_supported": (
                    experiment_design_audit.get(
                        "online_long_horizon_agent_execution_supported"
                    )
                    is True
                ),
            },
            "artifacts": [
                "experiment_design_audit.json",
                "long_horizon_control_contrasts.csv",
                "matched_construct_contrasts.jsonl",
                "matched_construct_statistics.json",
                "horizon_panel_contrasts.jsonl",
                "horizon_panel_statistics.json",
                "measurement_gates.json",
            ],
            "claim_boundary": (
                (
                    "The panel jointly changes effective transitions, dependency "
                    "depth, and handoffs, so it identifies horizon amplification "
                    "but not a pure handoff effect. "
                )
                if horizon
                else ""
            )
            + (
                "A ready measurement contract does not establish a positive or "
                "significant memory effect; use the paired estimates and intervals. "
                "Long-horizon qualification is based on semantic-effect-producing "
                "causal ancestors of the scored decision, not episode length or a "
                "digest-only chain. Every current action-relevant state must appear "
                "in the SCEU contract, and task-governance semantics are shared by "
                "all conditions. "
                "The current release evaluates preregistered critical decisions "
                "after audited replay-backed prefixes. It supports claims about "
                "delayed task-state control, not a claim that the tested policy "
                "executed every prefix step online."
            ),
        },
        {
            "contribution_id": "C2",
            "claim_timing": analysis_timing,
            "pre_call_contract_status": pre_call_contract_statuses["C2"],
            "name": "goal_relative_longitudinal_behavioral_drift",
            "evidence_status": _evidence_status(
                applicable=n_drift_trajectories > 0,
                required_gate_ids=c2_required_gates,
                gate_statuses=gate_statuses,
                missing_required_items=(
                    *c2_missing_estimands,
                    *c2_missing_design_requirements,
                ),
            ),
            "claim_scope": drift_mode,
            "required_gate_ids": list(c2_required_gates),
            "gate_statuses": _selected_gate_statuses(
                c2_required_gates,
                gate_statuses,
            ),
            "required_estimands": (
                [
                    "state_lineage_anchoring",
                    "adherence_anchored_onset",
                    "drift_free_survival",
                    "persistence",
                    "recovery",
                ]
                if longitudinal_applicable
                else ["drift_compatible_endpoint_violation"]
            ),
            "observed_estimands": (
                [
                    "state_lineage_anchoring",
                    "adherence_anchored_onset",
                    "drift_free_survival",
                    "persistence",
                    "recovery",
                ]
                if longitudinal_applicable
                and all(status == "pass" for status in longitudinal_statuses)
                and not c2_missing_estimands
                else ["drift_compatible_endpoint_violation"]
            ),
            "missing_estimands": list(c2_missing_estimands),
            "missing_design_requirements": list(
                c2_missing_design_requirements
            ),
            "evidence_counts": {
                "drift_trajectories": n_drift_trajectories,
                "lineage_backed_trajectories": n_lineage_backed_trajectories,
                "category_only_legacy_trajectories": (
                    n_category_only_trajectories
                ),
                "category_only_evaluable_trajectories": (
                    n_category_only_evaluable
                ),
                "lineage_backed_evaluable_trajectories": (
                    n_lineage_backed_evaluable
                ),
                "lineage_backed_observed_drift_trajectories": (
                    n_lineage_backed_observed
                ),
                "control_observed_drift_trajectories": (
                    control_observed_by_condition
                ),
            },
            "artifacts": [
                "drift_trajectories.json",
                "drift_calibration.json",
                "long_horizon_control_contrasts.csv",
                "measurement_gates.json",
            ],
            "claim_boundary": (
                "Endpoint violation excess is not longitudinal drift onset. Onset "
                "requires prior adherence at an earlier distinct eligible checkpoint "
                "for the same state lineage. Category-only legacy trajectories are "
                "descriptive, and drift observed in oracle-current-state or full-context "
                "controls cannot be attributed specifically to the memory system."
            ),
        },
        {
            "contribution_id": "C3",
            "claim_timing": analysis_timing,
            "pre_call_contract_status": pre_call_contract_statuses["C3"],
            "name": "decision_aligned_memory_to_behavior_fault_localization",
            "evidence_status": _evidence_status(
                applicable=n_decisions > 0,
                required_gate_ids=_C3_GATES,
                gate_statuses=gate_statuses,
                missing_required_items=(
                    *c3_missing_estimands,
                    *c3_missing_design_requirements,
                ),
            ),
            "claim_scope": "earliest_supported_stage_and_causal_use_lower_bound",
            "required_gate_ids": list(_C3_GATES),
            "gate_statuses": _selected_gate_statuses(
                _C3_GATES,
                gate_statuses,
            ),
            "required_estimands": [
                "conditional_stage_yields",
                "earliest_supported_failure_stage",
                "causal_use_lower_bound",
                "outcome_equivalent_fault_profile_divergence",
            ],
            "observed_estimands": (
                [
                    "conditional_stage_yields",
                    "earliest_supported_failure_stage",
                    "causal_use_lower_bound",
                    *(
                        ["outcome_equivalent_fault_profile_divergence"]
                        if n_outcome_equivalent_fault_pairs > 0
                        else []
                    ),
                ]
                if n_decisions > 0
                else []
            ),
            "missing_estimands": list(c3_missing_estimands),
            "missing_design_requirements": list(
                c3_missing_design_requirements
            ),
            "evidence_counts": {
                "decision_attribution_rows": n_decisions,
                "observed_failure_stages": len(observed_stages),
                "aligned_fault_profile_pairs": n_aligned_fault_pairs,
                "outcome_equivalent_fault_profile_pairs": (
                    n_outcome_equivalent_fault_pairs
                ),
            },
            "observed_failure_stages": list(observed_stages),
            "artifacts": [
                "decision_attribution.jsonl",
                "failure_attribution_scorecard.csv",
                "memory_events.jsonl",
                "retrieval_trace.jsonl",
                "interventions.jsonl",
                "fault_profile_divergence.json",
                "measurement_gates.json",
            ],
            "claim_boundary": (
                "The stage is the earliest failure supported by observed traces. "
                "Causal use is a repeat-stable intervention lower bound, not direct "
                "access to the model's internal reasoning. No detected unique "
                "causal effect does not exclude redundant or compensated use. "
                "Fault-profile pairs are "
                "dependent descriptive comparisons at the same decision; a zero "
                "divergence estimate is valid and is not converted into evidence "
                "of equivalence."
            ),
        },
    )
    return {
        "schema_version": CONTRIBUTION_EVIDENCE_SCHEMA_VERSION,
        "benchmark_object": (
            "memory_supported_delayed_task_state_control_under_competing_"
            "persistent_channels"
        ),
        "construct_mode": construct_mode,
        "primary_analysis_unit": str(
            summary.get("primary_analysis_unit", "episode")
        ),
        "analysis_phase": str(summary.get("analysis_phase", "development")),
        "analysis_timing": analysis_timing,
        "pre_call_contract_statuses": pre_call_contract_statuses,
        "confirmatory_timing_eligible": (
            analysis_timing == "pre_specified"
            and all(
                status == "pre_call_frozen"
                for status in pre_call_contract_statuses.values()
            )
        ),
        "measurement_ready": measurement_gates.get("measurement_ready") is True,
        "interpretation": (
            "Evidence readiness validates the measurement contract only. Effect "
            "direction and uncertainty must be read from statistical artifacts. "
            + (
                "The C1--C3 analysis scopes were fixed before policy calls."
                if analysis_timing == "pre_specified"
                and all(
                    status == "pre_call_frozen"
                    for status in pre_call_contract_statuses.values()
                )
                else (
                    "The run is labelled pre-specified, but one or more C1--C3 "
                    "pre-call contracts are missing; affected claims are not "
                    "timing-eligible."
                    if analysis_timing == "pre_specified"
                else (
                    "This is a post-hoc analysis and cannot be promoted to "
                    "confirmatory evidence."
                )
                )
            )
        ),
        "contributions": list(contributions),
    }


def contribution_evidence_markdown(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable companion artifact."""

    lines = [
        "# Contribution evidence contract",
        "",
        (
            "This table records whether each contribution's measurement evidence "
            "is present. `ready` does not mean that an effect is positive or "
            "statistically significant."
        ),
        f"Analysis phase: **{payload.get('analysis_phase', 'development')}**.",
        f"Analysis timing: **{payload.get('analysis_timing', 'missing')}**.",
        "",
        (
            "| ID | Contribution | Pre-call contract | Evidence status | "
            "Claim scope | Required gates |"
        ),
        "|---|---|---|---|---|---:|",
    ]
    for row in _mapping_sequence(payload.get("contributions")):
        required = row.get("required_gate_ids")
        lines.append(
            (
                "| {identifier} | `{name}` | `{contract}` | **{status}** | "
                "`{scope}` | {count} |"
            ).format(
                identifier=row.get("contribution_id", ""),
                name=row.get("name", ""),
                contract=row.get("pre_call_contract_status", "missing"),
                status=row.get("evidence_status", ""),
                scope=row.get("claim_scope", ""),
                count=_sequence_length(required),
            )
        )
    lines.extend(["", "## Evidence gaps", ""])
    for row in _mapping_sequence(payload.get("contributions")):
        raw_statuses = row.get("gate_statuses")
        statuses = raw_statuses if isinstance(raw_statuses, Mapping) else {}
        failed_gates = tuple(
            str(gate_id)
            for gate_id, status in statuses.items()
            if status != "pass"
        )
        missing_items = (
            *_string_sequence(row.get("missing_estimands")),
            *_string_sequence(row.get("missing_design_requirements")),
        )
        gaps = (
            *(f"gate:{gate_id}" for gate_id in failed_gates),
            *(f"item:{item}" for item in missing_items),
        )
        lines.append(
            "- **{}:** {}".format(
                row.get("contribution_id", ""),
                ", ".join(f"`{gap}`" for gap in gaps)
                if gaps
                else "no missing measurement-contract evidence",
            )
        )
    lines.extend(["", "## Claim boundaries", ""])
    for row in _mapping_sequence(payload.get("contributions")):
        lines.append(
            f"- **{row.get('contribution_id', '')}:** "
            f"{row.get('claim_boundary', '')}"
        )
    lines.extend(
        [
            "",
            (
                "Overall measurement readiness: **{}**. Statistical conclusions "
                "remain conditional on the declared calibration/confirmatory split."
            ).format(str(payload.get("measurement_ready") is True).lower()),
            "",
        ]
    )
    return "\n".join(lines)


def _gate_statuses(payload: Mapping[str, object]) -> dict[str, str]:
    return {
        str(row.get("gate_id", "")): str(row.get("status", "missing"))
        for row in _mapping_sequence(payload.get("gates"))
        if str(row.get("gate_id", ""))
    }


def _contribution_contract_statuses(
    experiment_design_audit: Mapping[str, object],
) -> dict[str, str]:
    analysis_contract = experiment_design_audit.get("analysis_contract")
    contribution_contracts = (
        analysis_contract.get("contribution_contracts")
        if isinstance(analysis_contract, Mapping)
        else None
    )
    contracts = (
        contribution_contracts
        if isinstance(contribution_contracts, Mapping)
        else {}
    )
    result: dict[str, str] = {}
    for contribution_id in ("C1", "C2", "C3"):
        raw = contracts.get(contribution_id)
        result[contribution_id] = (
            str(raw.get("status", "missing"))
            if isinstance(raw, Mapping)
            else "missing"
        )
    return result


def _contribution_design_requirements(
    experiment_design_audit: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    analysis_contract = experiment_design_audit.get("analysis_contract")
    contribution_contracts = (
        analysis_contract.get("contribution_contracts")
        if isinstance(analysis_contract, Mapping)
        else None
    )
    contracts = (
        contribution_contracts
        if isinstance(contribution_contracts, Mapping)
        else {}
    )
    result: dict[str, tuple[str, ...]] = {}
    for contribution_id in ("C1", "C2", "C3"):
        raw = contracts.get(contribution_id)
        requirements = (
            raw.get("required_design_checks")
            if isinstance(raw, Mapping)
            else None
        )
        result[contribution_id] = tuple(
            str(item)
            for item in (
                requirements
                if isinstance(requirements, Sequence)
                and not isinstance(requirements, str | bytes)
                else ()
            )
            if str(item)
        )
    return result


def _contribution_claim_scopes(
    experiment_design_audit: Mapping[str, object],
) -> dict[str, str]:
    analysis_contract = experiment_design_audit.get("analysis_contract")
    contribution_contracts = (
        analysis_contract.get("contribution_contracts")
        if isinstance(analysis_contract, Mapping)
        else None
    )
    contracts = (
        contribution_contracts
        if isinstance(contribution_contracts, Mapping)
        else {}
    )
    result: dict[str, str] = {}
    for contribution_id in ("C1", "C2", "C3"):
        raw = contracts.get(contribution_id)
        result[contribution_id] = (
            str(raw.get("claim_scope", "missing"))
            if isinstance(raw, Mapping)
            else "missing"
        )
    return result


def _design_check_statuses(
    experiment_design_audit: Mapping[str, object],
) -> dict[str, str]:
    raw_checks = experiment_design_audit.get("checks")
    return {
        str(row.get("check_id", "")): str(row.get("status", "missing"))
        for row in (
            raw_checks
            if isinstance(raw_checks, Sequence)
            and not isinstance(raw_checks, str | bytes)
            else ()
        )
        if isinstance(row, Mapping) and str(row.get("check_id", ""))
    }


def _statistic_metric_names(payload: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        str(row.get("metric", ""))
        for row in _mapping_sequence(payload.get("estimates"))
        if str(row.get("metric", ""))
    )


def _selected_gate_statuses(
    gate_ids: Sequence[str],
    statuses: Mapping[str, str],
) -> dict[str, str]:
    return {gate_id: statuses.get(gate_id, "missing") for gate_id in gate_ids}


def _evidence_status(
    *,
    applicable: bool,
    required_gate_ids: Sequence[str],
    gate_statuses: Mapping[str, str],
    missing_required_items: Sequence[str] = (),
) -> ContributionEvidenceStatus:
    if not applicable:
        return "not_applicable"
    statuses = tuple(
        gate_statuses.get(gate_id, "missing") for gate_id in required_gate_ids
    )
    if missing_required_items or any(status != "pass" for status in statuses):
        return "not_ready"
    return "ready"


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _sequence_length(value: object) -> int:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return 0
    return len(value)


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(str(item) for item in value)


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


__all__ = [
    "CONTRIBUTION_EVIDENCE_SCHEMA_VERSION",
    "ContributionEvidenceStatus",
    "build_contribution_evidence",
    "contribution_evidence_markdown",
]
