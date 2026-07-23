"""Deterministic, hash-addressed qualification report artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.analysis_phase import (
    parse_analysis_phase,
    parse_analysis_timing,
)
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.contribution_evidence import (
    build_contribution_evidence,
    contribution_evidence_markdown,
)
from lhmsb.qualification.design_audit import (
    compute_experiment_design_audit,
    experiment_design_audit_markdown,
)
from lhmsb.qualification.fault_profile import (
    compute_fault_profile_divergence,
    fault_profile_divergence_markdown,
)
from lhmsb.qualification.horizon_panel import (
    HORIZON_PRIMARY_ESTIMANDS,
    HORIZON_SECONDARY_ESTIMANDS,
    compute_horizon_panel_contrasts,
    horizon_panel_scorecard,
)
from lhmsb.qualification.longitudinal import (
    compute_drift_trajectory_report,
    drift_trajectory_markdown,
)
from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    StateCheckpointMetricInput,
    UsageMetricInput,
    _artifact_attributions,
    _is_memory_count_contrast,
    _is_memory_count_load_contrast,
    compute_failure_attribution_scorecard,
    compute_long_horizon_control_contrasts,
    compute_long_horizon_scorecard,
    compute_matched_construct_contrasts,
    compute_matched_construct_scorecard,
    compute_multisystem_metrics,
    compute_multisystem_metrics_by_cell,
    compute_multisystem_scorecard,
    compute_qualification_metrics,
    decision_attribution_rows,
    multisystem_observations_from_results,
    multisystem_state_checkpoints_from_artifacts,
)
from lhmsb.qualification.prefix import CommonRerankTrace, MemoryPrefixArtifact
from lhmsb.qualification.readiness import (
    compute_drift_action_calibration,
    compute_heuristic_baselines,
    compute_measurement_gates,
    drift_action_calibration_markdown,
    heuristic_baselines_markdown,
    measurement_gates_markdown,
)
from lhmsb.qualification.runner import (
    PolicyEvaluation,
    QualificationMatrixResult,
    QualificationTaskResult,
    RetrievalTrace,
    SCEURunResult,
)
from lhmsb.qualification.statistics import (
    compute_episode_cluster_statistics,
    compute_horizon_panel_statistics,
    compute_matched_group_statistics,
    horizon_panel_statistics_markdown,
    matched_group_statistics_markdown,
    statistics_markdown,
)

REPORT_SCHEMA_VERSION = 7
_CANONICAL_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)
REQUIRED_REPORT_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "tasks.jsonl",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "memory_events.jsonl",
    "memory_inventory.jsonl",
    "retrieval_trace.jsonl",
    "interventions.jsonl",
    "api_usage.jsonl",
    "policy_calls.jsonl",
    "prefix_manifests.jsonl",
    "graph_diagnostics.jsonl",
    "metrics.json",
    "metrics_by_cell.json",
    "summary.json",
    "scorecard.csv",
    "scorecard.md",
    "storage_scorecard.csv",
    "storage_scorecard.md",
    "memory_count_scorecard.csv",
    "memory_count_scorecard.md",
    "failure_attribution_scorecard.csv",
    "failure_attribution_scorecard.md",
    "decision_attribution.jsonl",
    "fault_profile_divergence.json",
    "fault_profile_divergence.md",
    "long_horizon_scorecard.csv",
    "long_horizon_scorecard.md",
    "long_horizon_control_contrasts.csv",
    "long_horizon_control_contrasts.md",
    "long_horizon_constructs.jsonl",
    "task_span.jsonl",
    "matched_construct_contrasts.jsonl",
    "matched_construct_scorecard.csv",
    "matched_construct_scorecard.md",
    "matched_construct_statistics.json",
    "matched_construct_statistics.md",
    "horizon_panel_contrasts.jsonl",
    "horizon_panel_scorecard.csv",
    "horizon_panel_scorecard.md",
    "horizon_panel_statistics.json",
    "horizon_panel_statistics.md",
    "drift_trajectories.json",
    "drift_trajectories.md",
    "statistics.json",
    "statistics.md",
    "heuristic_baselines.json",
    "heuristic_baselines.md",
    "drift_calibration.json",
    "drift_calibration.md",
    "measurement_gates.json",
    "measurement_gates.md",
    "experiment_design_audit.json",
    "experiment_design_audit.md",
    "contribution_evidence.json",
    "contribution_evidence.md",
    "limitations.md",
    "validation.json",
)
_JSONL_ARTIFACTS = (
    "tasks.jsonl",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "memory_events.jsonl",
    "memory_inventory.jsonl",
    "retrieval_trace.jsonl",
    "interventions.jsonl",
    "api_usage.jsonl",
    "policy_calls.jsonl",
    "prefix_manifests.jsonl",
    "graph_diagnostics.jsonl",
)
_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "status",
    "n_sceu",
    "mean_behavior_score",
    "behavior_correct_rate",
    "baseline_stability_rate",
    "mean_visible_memory_count",
    "mean_live_memory_count",
    "causal_memory_use_rate",
    "unique_causal_effect_rate",
    "beneficial_intervention_rate",
    "harmful_intervention_rate",
    "unstable_intervention_rate",
    "sham_replacement_action_flip_rate",
    "behavioral_use_probe_coverage",
    "probed_memory_causal_use_rate",
    "constraint_loss_rate",
    "constraint_loss_eligible_n",
    "targeted_constraint_loss_rate",
    "observed_constraint_loss_rate",
    "canonical_constraint_loss_violation_rate",
    "current_plan_deviation_rate",
    "plan_deviation_eligible_n",
    "targeted_plan_deviation_rate",
    "observed_plan_deviation_rate",
    "canonical_plan_deviation_violation_rate",
    "stale_state_action_rate",
    "stale_state_eligible_n",
    "targeted_stale_state_rate",
    "observed_stale_state_rate",
    "canonical_stale_state_violation_rate",
    "local_over_global_rate",
    "local_over_global_eligible_n",
    "targeted_local_over_global_rate",
    "observed_local_over_global_rate",
    "canonical_local_over_global_violation_rate",
    "aggregate_drift_rate",
    "aggregate_drift_eligible_n",
    "targeted_aggregate_drift_rate",
    "observed_aggregate_drift_rate",
    "canonical_drift_violation_rate",
    "off_target_drift_rate",
    "off_target_drift_n",
    "memory_count_contrast_rate",
    "memory_count_behavior_change_rate",
)
_HORIZON_PANEL_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "analysis_unit",
    "n_horizon_panels",
    "n_complete_horizon_panels",
    *tuple(f"mean_{estimand}" for estimand in HORIZON_PRIMARY_ESTIMANDS),
    *tuple(f"mean_{estimand}" for estimand in HORIZON_SECONDARY_ESTIMANDS),
)
_STORAGE_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "provenance_track",
    "source_readout",
    "write_coverage",
    "write_selectivity",
    "current_state_storage_precision",
    "current_state_storage_recall",
    "current_state_storage_f1",
    "stale_state_retention_rate",
    "update_delete_responsiveness",
    "physical_retirement_rate",
    "superseding_state_storage_rate",
    "write_to_continuation_alignment",
    "semantic_attribution_resolvability",
    "storage_provenance_completeness",
    "live_memory_count",
    "native_objects_per_logical_state_unit",
)
_MEMORY_COUNT_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "opportunity_id",
    "count_delta",
    "n_contrasts",
    "action_flip_rate",
    "behavior_change_rate",
    "mean_baseline_visible_memory_count",
    "mean_intervention_visible_memory_count",
)
_FAILURE_ATTRIBUTION_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "storage_evidence_mode",
    "n_sceu",
    "memory_reliant_n",
    "attribution_applicable_n",
    "no_memory_channel_n",
    "not_memory_reliant_n",
    "memory_required_state_count",
    "memory_required_storage_recall",
    "stored_to_retrieved_yield",
    "retrieved_to_visible_yield",
    "visible_required_probe_coverage",
    "probed_required_causal_use_rate",
    "storage_evidence_unavailable_n",
    "storage_evidence_unavailable_rate",
    "storage_failure_n",
    "storage_failure_rate",
    "retrieval_failure_n",
    "retrieval_failure_rate",
    "exposure_failure_n",
    "exposure_failure_rate",
    "utilization_failure_n",
    "utilization_failure_rate",
    "visible_without_detected_unique_causal_effect_n",
    "visible_without_detected_unique_causal_effect_rate",
    # Legacy columns remain readable when merging completed reports.
    "visible_without_detected_use_n",
    "visible_without_detected_use_rate",
    "visible_causally_influential_but_wrong_n",
    "visible_causally_influential_but_wrong_rate",
    "visible_use_evidence_incomplete_n",
    "visible_use_evidence_incomplete_rate",
    "behavior_success_causal_n",
    "behavior_success_causal_rate",
    "behavior_success_without_detected_unique_causal_effect_n",
    "behavior_success_without_detected_unique_causal_effect_rate",
    # Legacy columns remain readable when merging completed reports.
    "behavior_success_without_detected_use_n",
    "behavior_success_without_detected_use_rate",
    "behavior_success_unprobed_n",
    "behavior_success_unprobed_rate",
)
_LONG_HORIZON_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "construct_kind",
    "horizon_band",
    "n_sceu",
    "n_episodes",
    "mean_handoff_count",
    "mean_oldest_required_state_age",
    "mean_latest_decision_event_distance",
    "mean_dependency_depth",
    "mean_relevant_transition_count",
    "mean_effective_task_step_count",
    "mean_max_task_dependency_depth",
    "mean_causally_linked_task_step_fraction",
    "mean_memory_reliant_state_count",
    "mean_behavior_score",
    "behavior_correct_rate",
    "targeted_drift_rate",
    "targeted_drift_violation_rate",
    "drift_eligible_n",
    "observed_drift_rate",
    "canonical_drift_violation_rate",
)
_LONG_HORIZON_CONTROL_CONTRAST_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "construct_kind",
    "horizon_band",
    "n_matched_decisions",
    "n_episodes",
    "mean_behavior_gain_beyond_workspace",
    "mean_behavior_gap_to_oracle",
    "oracle_gap_closed",
    "workspace_behavior_correct_rate",
    "system_behavior_correct_rate",
    "oracle_behavior_correct_rate",
    "drift_matched_decisions",
    "targeted_drift_risk_difference_vs_workspace",
    "targeted_drift_risk_difference_vs_oracle",
)
_MATCHED_CONSTRUCT_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "n_counterfactual_groups",
    "n_complete_groups",
    "n_current_v1_offline_groups",
    "n_current_v2_offline_groups",
    "n_authorized_cloud_groups",
    "all_terminal_archetypes_covered",
    "mean_static_behavior_score",
    "mean_evolution_behavior_score",
    "mean_hierarchical_conflict_behavior_score",
    "mean_state_evolution_penalty_vs_static",
    "mean_hierarchical_conflict_penalty_vs_static",
    "mean_static_gain_beyond_workspace",
    "mean_evolution_gain_beyond_workspace",
    "mean_hierarchical_conflict_gain_beyond_workspace",
    "mean_state_evolution_penalty_excess_over_workspace",
    "mean_hierarchical_conflict_penalty_excess_over_workspace",
    "mean_state_evolution_drift_violation_excess_vs_static",
    "mean_hierarchical_conflict_drift_violation_excess_vs_static",
    "evolution_attribution_stage_change_rate",
    "hierarchical_conflict_attribution_stage_change_rate",
)


@dataclass(frozen=True)
class ReportArtifacts:
    root: Path
    artifact_hashes: tuple[tuple[str, str], ...]
    manifest_sha256: str


@dataclass(frozen=True)
class _ScorecardObservation:
    policy_profile_id: str
    condition: str
    readout: str
    status: str
    row: SCEURunResult


def write_qualification_report(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    output_directory: Path,
    *,
    run_metadata: Mapping[str, object] | None = None,
    prefix_artifacts: Mapping[str, object] | None = None,
) -> ReportArtifacts:
    """Write the complete deterministic report and hash every non-manifest file."""
    analysis_phase = parse_analysis_phase(
        (run_metadata or {}).get("analysis_phase", "development")
    )
    analysis_timing = parse_analysis_timing(
        (run_metadata or {}).get("analysis_timing", "pre_specified")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _flatten_rows(
        matrix,
        prefix_artifacts=prefix_artifacts,
        specs=specs,
    )
    for name in _JSONL_ARTIFACTS:
        _atomic_write(
            output_directory / name,
            _jsonl_bytes(rows[name]),
        )
    construct_rows = _long_horizon_construct_rows(matrix, specs)
    _atomic_write(
        output_directory / "long_horizon_constructs.jsonl",
        _jsonl_bytes(construct_rows),
    )
    task_span_rows = _task_span_rows(matrix, specs)
    _atomic_write(
        output_directory / "task_span.jsonl",
        _jsonl_bytes(task_span_rows),
    )

    evaluation_matrix = any(not hasattr(task, "writes") for task in matrix.task_results)
    if evaluation_matrix:
        observations = multisystem_observations_from_results(
            matrix,
            specs,
            prefix_artifacts=prefix_artifacts,
        )
        state_checkpoints = multisystem_state_checkpoints_from_artifacts(
            matrix,
            specs,
            prefix_artifacts=prefix_artifacts,
        )
        metrics = compute_multisystem_metrics(
            observations,
            state_checkpoints=state_checkpoints,
            usages=_metric_usages(rows["api_usage.jsonl"]),
        )
        metrics_by_cell = compute_multisystem_metrics_by_cell(
            observations,
            state_checkpoints_by_cell=_state_checkpoints_by_cell(
                matrix,
                specs,
                prefix_artifacts or {},
            ),
            usages_by_cell=_metric_usages_by_cell(
                observations,
                rows["api_usage.jsonl"],
            ),
        )
        scorecard_rows = list(compute_multisystem_scorecard(observations))
        storage_scorecard_rows = _storage_scorecard_rows(metrics_by_cell)
    else:
        metrics = compute_qualification_metrics(matrix, specs)
        metrics_by_cell = ()
        scorecard_rows = _scorecard_rows(matrix)
        storage_scorecard_rows = _storage_scorecard_rows(_metrics_by_cell(matrix, specs))
        observations = ()
    episode_observations = (
        observations
        if observations
        else multisystem_observations_from_results(
            matrix,
            specs,
            prefix_artifacts=prefix_artifacts,
        )
    )
    _atomic_write(
        output_directory / "metrics.json",
        _json_bytes(metrics.to_dict()),
    )
    _atomic_write(
        output_directory / "metrics_by_cell.json",
        _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "groups": (
                    list(metrics_by_cell) if evaluation_matrix else _metrics_by_cell(matrix, specs)
                ),
            }
        ),
    )
    expected_task_count = _expected_task_count(run_metadata, len(matrix.task_results))
    failure_attribution_rows = list(
        compute_failure_attribution_scorecard(episode_observations)
    )
    decision_rows = decision_attribution_rows(episode_observations)
    fault_profile_divergence = compute_fault_profile_divergence(decision_rows)
    _atomic_write(
        output_directory / "decision_attribution.jsonl",
        _jsonl_bytes(decision_rows),
    )
    long_horizon_rows = list(compute_long_horizon_scorecard(episode_observations))
    long_horizon_contrasts = list(
        compute_long_horizon_control_contrasts(episode_observations)
    )
    matched_construct_contrasts = list(
        compute_matched_construct_contrasts(episode_observations)
    )
    matched_construct_scorecard = list(
        compute_matched_construct_scorecard(episode_observations)
    )
    horizon_release = any(
        observation.horizon_panel_id for observation in episode_observations
    )
    matched_construct_statistics = (
        {
            "schema_version": 2,
            "status": "suppressed_within_panel_triplets",
            "analysis_unit": "horizon_panel",
            "estimates": [],
            "reason": (
                "The three horizon-level construct triplets within one panel are "
                "dependent repeated conditions. Use horizon_panel_statistics.json; "
                "do not treat them as three counterfactual-group samples."
            ),
        }
        if horizon_release
        else compute_matched_group_statistics(episode_observations)
    )
    horizon_panel_contrasts = list(
        compute_horizon_panel_contrasts(episode_observations)
    )
    horizon_panel_scorecard_rows = list(
        horizon_panel_scorecard(episode_observations)
    )
    horizon_panel_statistics = compute_horizon_panel_statistics(
        episode_observations
    )
    drift_trajectory_payload = compute_drift_trajectory_report(
        episode_observations
    )
    drift_trajectory_rows = drift_trajectory_payload.get("trajectories")
    n_drift_trajectories = (
        len(drift_trajectory_rows)
        if isinstance(drift_trajectory_rows, Sequence)
        and not isinstance(drift_trajectory_rows, (str, bytes))
        else 0
    )
    summary_payload = {
        **_summary(
            matrix,
            rows,
            specs=specs,
            expected_task_count=expected_task_count,
        ),
        "n_long_horizon_construct_profiles": len(construct_rows),
        "n_failure_attribution_cells": len(failure_attribution_rows),
        "n_decision_attribution_rows": len(decision_rows),
        "n_fault_profile_aligned_pairs": _as_int(
            fault_profile_divergence["n_aligned_decision_pairs"]
        ),
        "n_fault_profile_outcome_equivalent_pairs": _as_int(
            fault_profile_divergence["n_outcome_equivalent_pairs"]
        ),
        "n_long_horizon_scorecard_cells": len(long_horizon_rows),
        "n_long_horizon_control_contrasts": len(long_horizon_contrasts),
        "n_task_span_profiles": len(task_span_rows),
        "trajectory_interaction_mode_counts": dict(
            sorted(
                Counter(
                    str(row.get("interaction_mode", "missing"))
                    for row in task_span_rows
                ).items()
            )
        ),
        "n_online_long_horizon_agent_execution_profiles": sum(
            row.get("online_long_horizon_agent_execution_supported") is True
            for row in task_span_rows
        ),
        "n_matched_construct_contrasts": len(
            matched_construct_contrasts
        ),
        "n_matched_construct_scorecard_cells": len(
            matched_construct_scorecard
        ),
        "n_horizon_panel_contrasts": len(horizon_panel_contrasts),
        "n_horizon_panel_scorecard_cells": len(
            horizon_panel_scorecard_rows
        ),
        "n_drift_trajectories": n_drift_trajectories,
        "analysis_phase": analysis_phase,
        "analysis_timing": analysis_timing,
    }
    memory_count_scorecard_rows = _memory_count_scorecard_rows(rows["interventions.jsonl"])
    design_audit_payload = compute_experiment_design_audit(specs)
    design_audit_hash = canonical_hash(design_audit_payload)
    planned_design_audit_hash = (
        None
        if run_metadata is None
        else run_metadata.get("experiment_design_audit_hash")
    )
    if (
        planned_design_audit_hash is not None
        and planned_design_audit_hash != design_audit_hash
    ):
        raise ValueError(
            "report experiment design audit differs from the immutable run plan"
        )
    heuristic_payload = compute_heuristic_baselines(specs)
    drift_calibration_payload = compute_drift_action_calibration(specs)
    measurement_payload = compute_measurement_gates(
        matrix,
        specs,
        summary=summary_payload,
        heuristic_baselines=heuristic_payload,
        drift_calibration=drift_calibration_payload,
        expected_task_count=expected_task_count,
        observations=episode_observations,
    )
    contribution_evidence_payload = build_contribution_evidence(
        summary=summary_payload,
        measurement_gates=measurement_payload,
        matched_statistics=matched_construct_statistics,
        drift_trajectories=drift_trajectory_payload,
        decision_attribution_rows=decision_rows,
        fault_profile_divergence=fault_profile_divergence,
        experiment_design_audit=design_audit_payload,
        horizon_statistics=horizon_panel_statistics,
    )
    _atomic_write(
        output_directory / "summary.json",
        _json_bytes(summary_payload),
    )
    _atomic_write(
        output_directory / "scorecard.csv",
        _scorecard_csv(scorecard_rows).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "scorecard.md",
        _scorecard_markdown(scorecard_rows).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "storage_scorecard.csv",
        _table_csv(storage_scorecard_rows, _STORAGE_SCORECARD_FIELDS).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "storage_scorecard.md",
        _table_markdown(
            storage_scorecard_rows,
            _STORAGE_SCORECARD_FIELDS,
            title="Storage lifecycle scorecard",
            note=(
                "Exact and inferred lifecycle provenance are reported separately; "
                "readout is recorded only as the deduplicated source cell."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "memory_count_scorecard.csv",
        _table_csv(
            memory_count_scorecard_rows,
            _MEMORY_COUNT_SCORECARD_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "memory_count_scorecard.md",
        _table_markdown(
            memory_count_scorecard_rows,
            _MEMORY_COUNT_SCORECARD_FIELDS,
            title="Matched visible-memory-count scorecard",
            note=(
                "Each row compares the same checkpoint and SCEU after adding a "
                "pre-registered number of neutral, model-visible memory objects. "
                "Native live-store size is observational and is reported separately."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "failure_attribution_scorecard.csv",
        _table_csv(
            failure_attribution_rows,
            _FAILURE_ATTRIBUTION_SCORECARD_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "failure_attribution_scorecard.md",
        _table_markdown(
            failure_attribution_rows,
            _FAILURE_ATTRIBUTION_SCORECARD_FIELDS,
            title="Decision-aligned memory failure attribution",
            note=(
                "Each row preserves one policy/backend/readout cell. Stage yields are "
                "conditional: retrieval is scored only for stored required state, "
                "exposure only for retrieved state, and use only for visible state "
                "covered by a stable counterfactual probe."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "fault_profile_divergence.json",
        _json_bytes(fault_profile_divergence),
    )
    _atomic_write(
        output_directory / "fault_profile_divergence.md",
        fault_profile_divergence_markdown(
            fault_profile_divergence
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "long_horizon_scorecard.csv",
        _table_csv(
            long_horizon_rows,
            _LONG_HORIZON_SCORECARD_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "long_horizon_scorecard.md",
        _table_markdown(
            long_horizon_rows,
            _LONG_HORIZON_SCORECARD_FIELDS,
            title="Long-horizon construct scorecard",
            note=(
                "Horizon is represented by session handoffs, required-state age, "
                "dependency depth, and state-transition load rather than token count."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "long_horizon_control_contrasts.csv",
        _table_csv(
            long_horizon_contrasts,
            _LONG_HORIZON_CONTROL_CONTRAST_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "long_horizon_control_contrasts.md",
        _table_markdown(
            long_horizon_contrasts,
            _LONG_HORIZON_CONTROL_CONTRAST_FIELDS,
            title="Same-decision long-horizon control contrasts",
            note=(
                "Every system row is paired with workspace-only and "
                "oracle-current-state at the identical episode and continuation."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "matched_construct_contrasts.jsonl",
        _jsonl_bytes(matched_construct_contrasts),
    )
    _atomic_write(
        output_directory / "matched_construct_scorecard.csv",
        _table_csv(
            matched_construct_scorecard,
            _MATCHED_CONSTRUCT_SCORECARD_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "matched_construct_scorecard.md",
        _table_markdown(
            matched_construct_scorecard,
            _MATCHED_CONSTRUCT_SCORECARD_FIELDS,
            title="Counterfactually matched long-horizon construct scorecard",
            note=(
                "Each group holds the checkpoint, continuation request, action "
                "catalog, gold action, opaque option mapping, and prefix/workspace "
                "shape fixed while changing only static history, state evolution, "
                "or hierarchical conflict. Positive penalties mean worse behavior "
                "than the matched static history. Drift columns are endpoint "
                "violation excesses, not longitudinal drift onset. The "
                "penalty-excess-over-workspace columns are difference-in-differences "
                "that remove the matched workspace-only construct penalty."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "matched_construct_statistics.json",
        _json_bytes(matched_construct_statistics),
    )
    _atomic_write(
        output_directory / "matched_construct_statistics.md",
        matched_group_statistics_markdown(
            matched_construct_statistics
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "horizon_panel_contrasts.jsonl",
        _jsonl_bytes(horizon_panel_contrasts),
    )
    _atomic_write(
        output_directory / "horizon_panel_scorecard.csv",
        _table_csv(
            horizon_panel_scorecard_rows,
            _HORIZON_PANEL_SCORECARD_FIELDS,
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "horizon_panel_scorecard.md",
        _table_markdown(
            horizon_panel_scorecard_rows,
            _HORIZON_PANEL_SCORECARD_FIELDS,
            title="Same-decision horizon-dose scorecard",
            note=(
                "Each complete panel holds the terminal decision fixed while "
                "jointly increasing effective transitions, dependency depth, and "
                "session handoffs. Nine physical members contribute one panel, "
                "and the result is not interpreted as a pure handoff effect."
            ),
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "horizon_panel_statistics.json",
        _json_bytes(horizon_panel_statistics),
    )
    _atomic_write(
        output_directory / "horizon_panel_statistics.md",
        horizon_panel_statistics_markdown(
            horizon_panel_statistics
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "drift_trajectories.json",
        _json_bytes(drift_trajectory_payload),
    )
    _atomic_write(
        output_directory / "drift_trajectories.md",
        drift_trajectory_markdown(drift_trajectory_payload).encode("utf-8"),
    )
    if summary_payload["primary_analysis_unit"] == "episode":
        episode_groups = {
            episode_id: str(
                dict(spec.plan.metadata).get("semantic_scenario", "unknown")
            )
            for episode_id, spec in specs.items()
        }
        statistics_payload = compute_episode_cluster_statistics(
            observations,
            episode_groups=episode_groups,
            episode_group_name="semantic_scenario",
        )
    else:
        statistics_payload = {
            "schema_version": 1,
            "status": "suppressed_dependent_physical_members",
            "analysis_unit": summary_payload["primary_analysis_unit"],
            "n_unique_episodes": len(specs),
            "cells": [],
            "paired_comparisons": [],
            "reason": (
                "Generic episode-clustered inference is suppressed because "
                "counterfactual physical members are dependent repeated conditions. "
                "Use the matched-construct or horizon-panel statistics artifact."
            ),
        }
    _atomic_write(
        output_directory / "statistics.json",
        _json_bytes(statistics_payload),
    )
    _atomic_write(
        output_directory / "statistics.md",
        statistics_markdown(statistics_payload).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "heuristic_baselines.json",
        _json_bytes(heuristic_payload),
    )
    _atomic_write(
        output_directory / "heuristic_baselines.md",
        heuristic_baselines_markdown(heuristic_payload).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "drift_calibration.json",
        _json_bytes(drift_calibration_payload),
    )
    _atomic_write(
        output_directory / "drift_calibration.md",
        drift_action_calibration_markdown(drift_calibration_payload).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "measurement_gates.json",
        _json_bytes(measurement_payload),
    )
    _atomic_write(
        output_directory / "measurement_gates.md",
        measurement_gates_markdown(measurement_payload).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "experiment_design_audit.json",
        _json_bytes(design_audit_payload),
    )
    _atomic_write(
        output_directory / "experiment_design_audit.md",
        experiment_design_audit_markdown(design_audit_payload).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "contribution_evidence.json",
        _json_bytes(contribution_evidence_payload),
    )
    _atomic_write(
        output_directory / "contribution_evidence.md",
        contribution_evidence_markdown(
            contribution_evidence_payload
        ).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "limitations.md",
        _limitations_markdown(
            matrix,
            specs,
            measurement_gates=measurement_payload,
            heuristic_baselines=heuristic_payload,
        ).encode("utf-8"),
    )
    episode_artifacts = _write_episode_reports(
        output_directory,
        episode_observations,
        matrix=matrix,
    )
    _atomic_write(
        output_directory / "validation.json",
        _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "pending_external_validation",
                "run_identity": matrix.run_identity,
            }
        ),
    )

    artifact_names = (
        tuple(name for name in REQUIRED_REPORT_ARTIFACTS if name != "run_manifest.json")
        + episode_artifacts
    )
    artifact_hashes = tuple(
        (name, _sha256_file(output_directory / name)) for name in sorted(artifact_names)
    )
    metadata = dict(run_metadata or {})
    metadata.setdefault("analysis_phase", analysis_phase)
    metadata.setdefault("analysis_timing", analysis_timing)
    metadata.setdefault("experiment_design_audit_hash", design_audit_hash)
    metadata.setdefault(
        "experiment_design_audit_status",
        design_audit_payload["audit_status"],
    )
    metadata.setdefault(
        "balanced_mechanism_design_ready",
        design_audit_payload["balanced_mechanism_design_ready"],
    )
    for reserved in (
        "schema_version",
        "policy_trace_schema_version",
        "run_identity",
        "artifact_hashes",
    ):
        metadata.pop(reserved, None)
    routed_policy_fields = (
        "provider",
        "model_id",
        "route_id",
        "endpoint_identity",
        "request_hash",
        "response_hash",
        "policy_request_hash",
    )
    policy_calls = rows["policy_calls.jsonl"]
    policy_trace_schema_version = (
        2
        if policy_calls
        and all(
            all(isinstance(row.get(field), str) and row[field] for field in routed_policy_fields)
            for row in policy_calls
        )
        else 1
    )
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "policy_trace_schema_version": policy_trace_schema_version,
        **metadata,
        "run_identity": matrix.run_identity,
        "artifact_hashes": dict(artifact_hashes),
    }
    manifest_path = output_directory / "run_manifest.json"
    _atomic_write(manifest_path, _json_bytes(manifest))
    return ReportArtifacts(
        root=output_directory,
        artifact_hashes=artifact_hashes,
        manifest_sha256=_sha256_file(manifest_path),
    )


def _long_horizon_construct_rows(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> list[dict[str, object]]:
    """Materialize one evaluator-side construct profile per evaluated SCEU."""
    evaluated = {
        str(getattr(task, "episode_id", "")) for task in matrix.task_results
    }
    output: list[dict[str, object]] = []
    for episode_id in sorted(evaluated):
        spec = specs.get(episode_id)
        if spec is None:
            continue
        metadata = dict(spec.plan.metadata)
        task_span = profile_task_span(spec.plan)
        for sceu in spec.plan.sceu_units:
            output.append(
                {
                    **profile_sceu(spec.plan, sceu).to_dict(),
                    "semantic_scenario": metadata.get(
                        "semantic_scenario", "unknown"
                    ),
                    "phase_signature": metadata.get("phase_signature", "unknown"),
                    "recoverability_variant": metadata.get(
                        "recoverability_variant", "unknown"
                    ),
                    "counterfactual_group_id": metadata.get(
                        "counterfactual_group_id", ""
                    ),
                    "counterfactual_variant": metadata.get(
                        "counterfactual_variant", ""
                    ),
                    "counterfactual_terminal_archetype": metadata.get(
                        "terminal_archetype", ""
                    ),
                    "is_counterfactual_target": (
                        sceu.opportunity_id
                        == metadata.get(
                            "counterfactual_target_opportunity_id",
                            "",
                        )
                    ),
                    "effective_task_step_count": (
                        task_span.effective_step_count
                    ),
                    "max_task_dependency_depth": (
                        task_span.max_dependency_depth
                    ),
                    "causally_linked_task_step_fraction": (
                        task_span.causally_linked_step_fraction
                    ),
                }
            )
    return output


def _task_span_rows(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> list[dict[str, object]]:
    """Materialize one auditable effective-step profile per evaluated episode."""

    evaluated = {
        str(getattr(task, "episode_id", "")) for task in matrix.task_results
    }
    output: list[dict[str, object]] = []
    for episode_id in sorted(evaluated):
        spec = specs.get(episode_id)
        if spec is None:
            continue
        metadata = spec.plan.metadata_dict
        output.append(
            {
                **profile_task_span(spec.plan).to_dict(),
                "construct_mode": metadata.get("construct_mode", "mixed"),
                "counterfactual_group_id": metadata.get(
                    "counterfactual_group_id",
                    "",
                ),
                "counterfactual_variant": metadata.get(
                    "counterfactual_variant",
                    "",
                ),
                "counterfactual_terminal_archetype": metadata.get(
                    "terminal_archetype",
                    "",
                ),
            }
        )
    return output


def _write_episode_reports(
    output_directory: Path,
    observations: Sequence[MultisystemMetricInput],
    *,
    matrix: QualificationMatrixResult,
) -> tuple[str, ...]:
    """Write descriptive, independently auditable reports for every episode."""
    grouped: dict[str, list[MultisystemMetricInput]] = defaultdict(list)
    for observation in observations:
        if observation.episode_id:
            grouped[observation.episode_id].append(observation)
    root = output_directory / "episodes"
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    index: list[dict[str, object]] = []
    for episode_id in sorted(grouped):
        episode_rows = tuple(grouped[episode_id])
        directory_name = _episode_directory_name(episode_id)
        episode_root = root / directory_name
        episode_root.mkdir(parents=True, exist_ok=True)
        metrics = compute_multisystem_metrics(episode_rows)
        metrics_by_cell = compute_multisystem_metrics_by_cell(episode_rows)
        scorecard = list(compute_multisystem_scorecard(episode_rows))
        failure_attribution = list(
            compute_failure_attribution_scorecard(episode_rows)
        )
        long_horizon_scorecard = list(
            compute_long_horizon_scorecard(episode_rows)
        )
        long_horizon_contrasts = list(
            compute_long_horizon_control_contrasts(episode_rows)
        )
        drift_trajectories = compute_drift_trajectory_report(episode_rows)
        decision_attributions = decision_attribution_rows(episode_rows)
        fault_profile_divergence = compute_fault_profile_divergence(
            decision_attributions
        )
        statuses = tuple(row.status for row in episode_rows)
        summary = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "episode_id": episode_id,
            "analysis_scope": "single_episode_descriptive",
            "n_sceu": len(episode_rows),
            "n_tasks": sum(
                str(getattr(task, "episode_id", "")) == episode_id for task in matrix.task_results
            ),
            "status": _aggregate_status(statuses),
            "conditions": sorted({row.condition for row in episode_rows}),
            "readouts": sorted({row.readout for row in episode_rows}),
            "note": (
                "Inferential statistics are reported only at aggregate episode level; "
                "this file is descriptive and audit-oriented."
            ),
        }
        payloads = {
            "metrics.json": _json_bytes(metrics.to_dict()),
            "metrics_by_cell.json": _json_bytes(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "groups": list(metrics_by_cell),
                }
            ),
            "scorecard.csv": _scorecard_csv(scorecard).encode("utf-8"),
            "scorecard.md": _scorecard_markdown(scorecard).encode("utf-8"),
            "failure_attribution_scorecard.csv": _table_csv(
                failure_attribution,
                _FAILURE_ATTRIBUTION_SCORECARD_FIELDS,
            ).encode("utf-8"),
            "failure_attribution_scorecard.md": _table_markdown(
                failure_attribution,
                _FAILURE_ATTRIBUTION_SCORECARD_FIELDS,
                title="Decision-aligned memory failure attribution",
                note=(
                    "Single-episode descriptive funnel; inference remains at the "
                    "aggregate episode level."
                ),
            ).encode("utf-8"),
            "decision_attribution.jsonl": _jsonl_bytes(
                decision_attributions
            ),
            "fault_profile_divergence.json": _json_bytes(
                fault_profile_divergence
            ),
            "fault_profile_divergence.md": fault_profile_divergence_markdown(
                fault_profile_divergence
            ).encode("utf-8"),
            "long_horizon_scorecard.csv": _table_csv(
                long_horizon_scorecard,
                _LONG_HORIZON_SCORECARD_FIELDS,
            ).encode("utf-8"),
            "long_horizon_scorecard.md": _table_markdown(
                long_horizon_scorecard,
                _LONG_HORIZON_SCORECARD_FIELDS,
                title="Long-horizon construct scorecard",
                note=(
                    "Single-episode descriptive breakdown by construct and horizon."
                ),
            ).encode("utf-8"),
            "long_horizon_control_contrasts.csv": _table_csv(
                long_horizon_contrasts,
                _LONG_HORIZON_CONTROL_CONTRAST_FIELDS,
            ).encode("utf-8"),
            "long_horizon_control_contrasts.md": _table_markdown(
                long_horizon_contrasts,
                _LONG_HORIZON_CONTROL_CONTRAST_FIELDS,
                title="Same-decision long-horizon control contrasts",
                note=(
                    "Single-episode paired comparisons against workspace-only "
                    "and oracle-current-state."
                ),
            ).encode("utf-8"),
            "drift_trajectories.json": _json_bytes(drift_trajectories),
            "drift_trajectories.md": drift_trajectory_markdown(
                drift_trajectories
            ).encode("utf-8"),
            "summary.json": _json_bytes(summary),
        }
        for name, payload in payloads.items():
            path = episode_root / name
            _atomic_write(path, payload)
            artifacts.append(path.relative_to(output_directory).as_posix())
        index.append(
            {
                "episode_id": episode_id,
                "directory": f"episodes/{directory_name}",
                "n_sceu": len(episode_rows),
                "status": summary["status"],
            }
        )
    index_path = root / "index.json"
    _atomic_write(
        index_path,
        _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "episode_count": len(index),
                "episodes": index,
            }
        ),
    )
    artifacts.append(index_path.relative_to(output_directory).as_posix())
    return tuple(sorted(artifacts))


def _limitations_markdown(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    measurement_gates: Mapping[str, object],
    heuristic_baselines: Mapping[str, object],
) -> str:
    """Render deterministic scope limits alongside every experiment report."""
    frozen_scenarios = sorted(
        {
            str(dict(spec.plan.metadata).get("semantic_scenario", "unknown"))
            for spec in specs.values()
        }
    )
    frozen_schedules = sorted(
        {str(dict(spec.plan.metadata).get("phase_signature", "unknown")) for spec in specs.values()}
    )
    evaluated_episode_ids = {str(getattr(task, "episode_id", "")) for task in matrix.task_results}
    evaluated_specs = tuple(
        spec for episode_id, spec in specs.items() if episode_id in evaluated_episode_ids
    )
    evaluated_scenarios = sorted(
        {
            str(dict(spec.plan.metadata).get("semantic_scenario", "unknown"))
            for spec in evaluated_specs
        }
    )
    evaluated_schedules = sorted(
        {
            str(dict(spec.plan.metadata).get("phase_signature", "unknown"))
            for spec in evaluated_specs
        }
    )
    evaluated_session_counts = sorted(
        {spec.plan.n_sessions for spec in evaluated_specs}
    )
    profiles = sorted(
        {str(getattr(task, "policy_profile_id", "unknown")) for task in matrix.task_results}
    )
    conditions = sorted(
        {
            str(getattr(condition, "condition", "unknown"))
            for task in matrix.task_results
            for condition in getattr(task, "condition_results", ())
        }
    )
    task_span_profiles = tuple(
        profile_task_span(spec.plan) for spec in evaluated_specs
    )
    interaction_mode_counts = Counter(
        profile.interaction_mode for profile in task_span_profiles
    )
    raw_gates = measurement_gates.get("gates", ())
    gates = (
        tuple(item for item in raw_gates if isinstance(item, Mapping))
        if isinstance(raw_gates, Sequence) and not isinstance(raw_gates, str | bytes)
        else ()
    )
    unresolved = tuple(item for item in gates if str(item.get("status")) != "pass")
    ready = measurement_gates.get("measurement_ready") is True
    best_action = heuristic_baselines.get("best_always_action")
    best_accuracy = heuristic_baselines.get("best_always_action_accuracy")
    lines = [
        "# Scope and limitations",
        "",
        f"Measurement readiness: **{str(ready).lower()}**.",
        "",
        "## Scope",
        "",
        f"- Evaluated episodes: {len(evaluated_specs)} of {len(specs)} frozen "
        "software-project trajectories.",
        f"- Evaluated semantic scenarios: {len(evaluated_scenarios)} "
        f"({', '.join(evaluated_scenarios) or 'none'}).",
        f"- Frozen-dataset semantic scenarios: {len(frozen_scenarios)} "
        f"({', '.join(frozen_scenarios) or 'none'}).",
        f"- Evaluated phase schedules: {len(evaluated_schedules)} "
        f"({', '.join(evaluated_schedules) or 'none'}).",
        "- Sessions per evaluated trajectory: "
        f"{', '.join(str(value) for value in evaluated_session_counts) or 'none'}.",
        f"- Frozen-dataset phase schedules: {len(frozen_schedules)} "
        f"({', '.join(frozen_schedules) or 'none'}).",
        f"- Policy profiles: {', '.join(profiles) or 'none'}.",
        f"- Conditions: {', '.join(conditions) or 'none'}.",
        "- Trajectory interaction modes: "
        f"{dict(sorted(interaction_mode_counts.items()))!s}.",
        "- Policy-free fixed-action and opaque-option baselines use the full frozen "
        "dataset, not only the evaluated subset.",
        "- Generated trajectory/schedule variants are not independent semantic task templates; "
        "episode-level inference is accompanied by semantic-scenario sensitivity analysis.",
        "",
        "## Interpretation constraints",
        "",
        "- This release evaluates critical continuation decisions sampled from replayable "
        "persistent-task trajectories. Unless a task-span row is explicitly marked "
        "`online_long_horizon_agent_execution`, it does not claim that the tested policy "
        "executed hundreds or thousands of mutually dependent steps online. Current "
        "long-horizon claims are restricted to delayed task-state control after audited "
        "handoffs, state transitions, dependency depth, and workspace changes.",
        "- Lifecycle provenance (native event versus inventory-inferred change) and semantic "
        "state attribution are separate axes. Ambiguous semantic attribution earns no positive "
        "storage coverage.",
        "- Behaviorally used memory is a conservative lower bound from repeat-stable, "
        "state-targeted replacement interventions. Failure to identify a memory as used does "
        "not prove that it had no influence: redundant or compensable use may produce no "
        "unique action-level effect.",
        "- Targeted drift-compatible violation rates use preregistered category-specific "
        "opportunities. Longitudinal drift onset additionally requires prior adherence at "
        "an earlier distinct eligible checkpoint; first-observation errors are violations, "
        "not observed drift. Off-target violations are reported separately.",
        "- Matched static/evolution/conflict endpoint effects use counterfactual group as the "
        "analysis unit. Their drift fields are violation excesses, not longitudinal onset.",
        "- Memory-count effects use matched within-opportunity contrasts and must not be "
        "interpreted as checkpoint-length scaling.",
        "- Controlled/common-readout and native-readout comparisons answer different questions "
        "and should not be pooled into one ranking.",
        f"- The strongest policy-free fixed-action baseline is {best_action!s} at "
        f"{_format_optional_rate(best_accuracy)} accuracy.",
        "",
        "## Unresolved measurement gates",
        "",
    ]
    if not unresolved:
        lines.append("- None. All preregistered gates passed.")
    else:
        for item in unresolved:
            lines.append(
                f"- `{item.get('gate_id', 'unknown')}`: "
                f"{item.get('status', 'unknown')} — "
                f"{item.get('description', 'No description.')}"
            )
    lines.extend(
        (
            "",
            "Artifact validation and scientific measurement readiness remain separate: a "
            "hash-valid run can still fail one or more gates above.",
            "",
        )
    )
    return "\n".join(lines)


def _format_optional_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "unavailable"
    return f"{float(value):.3f}"


def _episode_directory_name(episode_id: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in episode_id
    ).strip("-")
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'episode'}--{digest}"


def _flatten_rows(
    matrix: QualificationMatrixResult,
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
    specs: Mapping[str, SoftwareMem0VerticalSpec] | None = None,
) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in _JSONL_ARTIFACTS}
    seen_calls: set[str] = set()
    for task in matrix.task_results:
        task_context = _task_context(task)
        artifact = _prefix_artifact_for_task(task, prefix_artifacts or {})
        if artifact is not None:
            rows["prefix_manifests.jsonl"].append(
                {
                    **task_context,
                    "prefix_artifact_hash": artifact.artifact_hash,
                    "backend": artifact.backend,
                    "profile_id": artifact.profile_id,
                    "config_hash": artifact.config_hash,
                    "run_identity": artifact.run_identity,
                    "dataset_release": artifact.dataset_release,
                    "surface_hash": artifact.surface_hash,
                    "source_commit": artifact.source_commit,
                }
            )
            # Schema-v2 evaluation results carry their immutable memory prefix
            # outside the task result (rather than through legacy ``writes``).
            # Export one inventory snapshot per checkpoint so the report keeps
            # the stored-state side of the retrieval chain auditable.
            if not getattr(task, "writes", ()):
                spec = (specs or {}).get(str(getattr(task, "episode_id", "")))
                for checkpoint in artifact.checkpoints:
                    for rerank in checkpoint.common_reranks:
                        _append_prefix_reranker_usage(
                            rows["api_usage.jsonl"],
                            seen_calls,
                            task_context,
                            checkpoint_session=checkpoint.checkpoint_session,
                            trace=rerank,
                        )
                    if checkpoint.inventory is not None:
                        attribution_payload: dict[str, object] = {}
                        if spec is not None:
                            attribution_payload = {
                                item.memory_id: {
                                    "state_ids": list(item.state_ids),
                                    "provenance_mode": item.provenance_mode,
                                    "source_event_ids": list(item.source_event_ids),
                                    "source_session": item.source_session,
                                    "method": item.method,
                                    "contributes_positive_coverage": (
                                        item.contributes_positive_coverage
                                    ),
                                    "reason": item.reason,
                                }
                                for item in _artifact_attributions(
                                    spec,
                                    checkpoint.inventory,
                                    checkpoint,
                                    checkpoint.checkpoint_session,
                                    artifact=artifact,
                                ).values()
                            }
                        rows["memory_inventory.jsonl"].append(
                            {
                                **task_context,
                                "evaluator_attribution_by_memory": attribution_payload,
                                **_jsonable(asdict(checkpoint.inventory)),
                            }
                        )
                    for write in checkpoint.writes:
                        for event in write.events:
                            rows["memory_events.jsonl"].append(
                                {
                                    **task_context,
                                    "session_index": write.session_index,
                                    "provenance_mode": (
                                        "inferred"
                                        if event.source
                                        in {
                                            "inventory_diff",
                                            "inventory_delta",
                                            "inventory_snapshot_diff",
                                            "write_inventory_diff",
                                            "snapshot_diff",
                                            "neo4j_graph_diff",
                                        }
                                        or event.native_event.upper().startswith("INFERRED")
                                        else "native/exact"
                                    ),
                                    **_jsonable(asdict(event)),
                                }
                            )
                        for usage in write.usage_events:
                            _append_internal_usage(
                                rows["api_usage.jsonl"],
                                seen_calls,
                                {
                                    **task_context,
                                    "session_index": write.session_index,
                                },
                                usage,
                            )
            for checkpoint in artifact.checkpoints:
                for key, value in checkpoint.graph_diagnostics:
                    rows["graph_diagnostics.jsonl"].append(
                        {
                            **task_context,
                            "checkpoint_session": checkpoint.checkpoint_session,
                            "key": key,
                            "value": _jsonable(value),
                        }
                    )
        rows["tasks.jsonl"].append(
            {
                **task_context,
                "status": task.status,
                "result_ids": [condition.result_id for condition in task.condition_results],
            }
        )
        rows["task_results.jsonl"].append(_jsonable(asdict(task)))
        for write in getattr(task, "writes", ()):
            write_context = {
                **task_context,
                "session_index": write.session_index,
            }
            for event in write.events:
                rows["memory_events.jsonl"].append(
                    {
                        **write_context,
                        **_jsonable(asdict(event)),
                    }
                )
            rows["memory_inventory.jsonl"].append(
                {
                    **write_context,
                    **_jsonable(asdict(write.inventory)),
                }
            )
            for usage in write.usage_events:
                _append_internal_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    write_context,
                    usage,
                )
        for trace in getattr(task, "retrieval_traces", ()):
            rows["retrieval_trace.jsonl"].append(
                {
                    **task_context,
                    **_jsonable(asdict(trace)),
                }
            )
            for usage in trace.internal_usage:
                _append_internal_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    {
                        **task_context,
                        "sceu_id": trace.sceu_id,
                        "opportunity_id": trace.opportunity_id,
                        "checkpoint_session": trace.checkpoint_session,
                    },
                    usage,
                )
            if trace.rerank_result is not None:
                _append_reranker_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    task_context,
                    trace,
                )
        for condition in task.condition_results:
            condition_context = {
                **task_context,
                "result_id": condition.result_id,
                "condition": condition.condition,
                "readout": condition.readout,
                "condition_status": condition.status,
            }
            for row in condition.sceu_results:
                trace_id = _evaluation_trace_id(
                    task.task_id,
                    row.sceu_id,
                    condition.readout,
                    row.retrieval_trace_id,
                )
                row_payload = {
                    **condition_context,
                    **_jsonable(asdict(row)),
                }
                if trace_id is not None:
                    row_payload["retrieval_trace_id"] = trace_id
                rows["sceu_results.jsonl"].append(row_payload)
                for evaluation in row.baseline_evaluations:
                    _append_api_usage(
                        rows["api_usage.jsonl"],
                        seen_calls,
                        condition_context,
                        row,
                        evaluation,
                        intervention_kind="baseline",
                        target_memory_id=None,
                    )
                for intervention in row.interventions:
                    rows["interventions.jsonl"].append(
                        {
                            **condition_context,
                            "sceu_id": row.sceu_id,
                            "opportunity_id": row.opportunity_id,
                            **_jsonable(asdict(intervention)),
                        }
                    )
                    for evaluation in intervention.evaluations:
                        _append_api_usage(
                            rows["api_usage.jsonl"],
                            seen_calls,
                            condition_context,
                            row,
                            evaluation,
                            intervention_kind=intervention.intervention_kind,
                            target_memory_id=intervention.target_memory_id,
                        )
            # The schema-v2 evaluator persists retrievals inside each immutable
            # SCEU result rather than a mutable runner task trace.  Normalize
            # them to the same trace artifact here.
            if not getattr(task, "retrieval_traces", ()) and condition.condition in {
                "flat_retrieval",
                "mem0",
                "amem",
                "memos",
            }:
                for row in condition.sceu_results:
                    rows["retrieval_trace.jsonl"].append(
                        {
                            **condition_context,
                            "trace_id": _evaluation_trace_id(
                                task.task_id,
                                row.sceu_id,
                                condition.readout,
                                row.retrieval_trace_id,
                            ),
                            "sceu_id": row.sceu_id,
                            "opportunity_id": row.opportunity_id,
                            "checkpoint_session": row.checkpoint_session,
                            "query": "",
                            "query_hash": "",
                            "candidate_memory_ids": list(row.candidate_memory_ids),
                            "backend_retrieved_memory_ids": list(
                                getattr(
                                    row,
                                    "backend_retrieved_memory_ids",
                                    row.candidate_memory_ids,
                                )
                            ),
                            "selected_memory_ids": list(
                                getattr(
                                    row,
                                    "selected_memory_ids",
                                    row.retrieved_memory_ids,
                                )
                            ),
                            "model_visible_memory_ids": list(row.model_visible_memory_ids),
                            "behaviorally_used_memory_ids": list(
                                getattr(row, "behaviorally_used_memory_ids", ())
                            ),
                            "native_retrieved_memory_ids": list(
                                row.retrieved_memory_ids if condition.readout == "native" else ()
                            ),
                            "common_reranked_memory_ids": list(
                                row.retrieved_memory_ids
                                if condition.readout == "common_rerank"
                                else ()
                            ),
                            "candidate_shortfall": bool(getattr(row, "candidate_shortfall", False)),
                            "search_latency_seconds": 0.0,
                            "rerank_result": None,
                            "internal_usage": [],
                        }
                    )
    rows["policy_calls.jsonl"] = [
        dict(row)
        for row in rows["api_usage.jsonl"]
        if isinstance(row.get("policy_request_hash"), str) and bool(row["policy_request_hash"])
    ]
    return rows


def _append_api_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    condition_context: Mapping[str, object],
    sceu: SCEURunResult,
    evaluation: PolicyEvaluation,
    *,
    intervention_kind: str,
    target_memory_id: str | None,
) -> None:
    if evaluation.call_id in seen_calls:
        return
    seen_calls.add(evaluation.call_id)
    rows.append(
        {
            **condition_context,
            "sceu_id": sceu.sceu_id,
            "opportunity_id": sceu.opportunity_id,
            "intervention_kind": intervention_kind,
            "target_memory_id": target_memory_id,
            "call_id": evaluation.call_id,
            "call_kind": evaluation.call_kind,
            "provider": evaluation.response.provider,
            "model_id": evaluation.response.model_id,
            "endpoint_identity": evaluation.response.endpoint_identity,
            "provider_request_id": evaluation.response.provider_request_id,
            "request_hash": evaluation.response.request_hash,
            "response_hash": evaluation.response.response_hash,
            "policy_request_hash": evaluation.policy_request_hash,
            "input_tokens": evaluation.response.usage.input_tokens,
            "output_tokens": evaluation.response.usage.output_tokens,
            "cached_tokens": evaluation.response.usage.cached_tokens,
            "reasoning_tokens": evaluation.response.usage.reasoning_tokens,
            "usage_observed": evaluation.response.usage.observed,
            "input_count": 1,
            "latency_seconds": evaluation.response.latency_seconds,
            "retry_count": evaluation.response.retry_count,
            "format_repair_used": evaluation.response.format_repair_used,
            "error_class": None,
            "started_at_utc": evaluation.response.started_at_utc,
            "ended_at_utc": evaluation.response.ended_at_utc,
        }
    )


def _append_internal_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    context: Mapping[str, object],
    usage: object,
) -> None:
    from lhmsb.adapters.mem0_qualification import ProviderUsageEvent

    if not isinstance(usage, ProviderUsageEvent):
        raise TypeError("internal provider usage has the wrong type")
    call_id = _provider_usage_call_id(usage)
    if call_id in seen_calls:
        return
    seen_calls.add(call_id)
    call_kind = _canonical_usage_component(usage.component)
    rows.append(
        {
            **context,
            "call_id": call_id,
            "provider_call_id": usage.call_id,
            "call_kind": call_kind,
            "provider_component": usage.component,
            "provider": usage.provider,
            "model_id": usage.model_id,
            "endpoint_identity": usage.endpoint_identity,
            "provider_request_id": None,
            "request_hash": usage.request_hash,
            "response_hash": usage.response_hash,
            "policy_request_hash": None,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "usage_observed": usage.usage_observed,
            "input_count": usage.input_count,
            "latency_seconds": usage.latency_seconds,
            "retry_count": usage.retry_count,
            "format_repair_used": False,
            "error_class": usage.error_class,
            "started_at_utc": usage.started_at_utc,
            "ended_at_utc": usage.ended_at_utc,
        }
    )


def _provider_usage_call_id(usage: object) -> str:
    """Address one actual provider call despite backend-local ID reuse.

    Some native systems expose cumulative usage lists at later checkpoints,
    while separately constructed writer components restart their local call-ID
    counters.  Provider request/response identity plus timestamps distinguishes
    real repeated calls and collapses only the cumulative copies.
    """
    payload = {
        "provider": getattr(usage, "provider", None),
        "endpoint_identity": getattr(usage, "endpoint_identity", None),
        "request_hash": getattr(usage, "request_hash", None),
        "response_hash": getattr(usage, "response_hash", None),
        "started_at_utc": getattr(usage, "started_at_utc", None),
        "ended_at_utc": getattr(usage, "ended_at_utc", None),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"provider:{digest}"


def _canonical_usage_component(component: object) -> str:
    """Normalize historical writer labels to the report metric namespace."""
    value = str(component)
    if value == "memory_writer":
        return "memory_internal_llm"
    return value


def _metric_usages(
    rows: Sequence[Mapping[str, object]],
) -> tuple[UsageMetricInput, ...]:
    """Convert the deduplicated API ledger into aggregate metric inputs."""
    return tuple(_metric_usage(row) for row in rows)


def _metric_usages_by_cell(
    observations: Sequence[object],
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], tuple[UsageMetricInput, ...]]:
    """Attach policy calls exactly and shared prefix costs to each readout cell."""
    cell_keys = {
        (
            str(getattr(item, "policy_profile_id", "")),
            str(getattr(item, "condition", "")),
            str(getattr(item, "readout", "")),
        )
        for item in observations
    }
    grouped: dict[tuple[str, str, str], list[UsageMetricInput]] = defaultdict(list)
    for row in rows:
        policy_id = str(row.get("policy_profile_id", ""))
        condition = str(row.get("condition", ""))
        readout = row.get("readout")
        usage = _metric_usage(row)
        if isinstance(readout, str) and readout:
            key = (policy_id, condition, readout)
            if key in cell_keys:
                grouped[key].append(usage)
            continue
        # Prefix preparation is shared by all readouts of the same backend.
        # Each cell reports the cost of independently running that cell; the
        # aggregate ledger above still counts every provider call only once.
        for key in sorted(cell_keys):
            if key[:2] == (policy_id, condition):
                grouped[key].append(usage)
    return {key: tuple(value) for key, value in grouped.items()}


def _state_checkpoints_by_cell(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    prefix_artifacts: Mapping[str, object],
) -> dict[tuple[str, str, str], tuple[StateCheckpointMetricInput, ...]]:
    """Project each immutable backend prefix onto its scored readout cells."""
    grouped: dict[tuple[str, str, str], list[StateCheckpointMetricInput]] = {}
    for task in matrix.task_results:
        checkpoints = multisystem_state_checkpoints_from_artifacts(
            (task,),
            specs,
            prefix_artifacts=prefix_artifacts,
        )
        if not checkpoints:
            continue
        policy_id = str(getattr(task, "policy_profile_id", ""))
        task_condition = str(getattr(task, "condition", ""))
        for condition_result in getattr(task, "condition_results", ()):
            condition = str(getattr(condition_result, "condition", task_condition))
            readout = str(getattr(condition_result, "readout", "none"))
            grouped.setdefault((policy_id, condition, readout), []).extend(checkpoints)
    return {key: tuple(value) for key, value in grouped.items()}


def _metric_usage(row: Mapping[str, object]) -> UsageMetricInput:
    is_policy = isinstance(row.get("policy_request_hash"), str) and bool(
        row.get("policy_request_hash")
    )
    component = "policy" if is_policy else _canonical_usage_component(row.get("call_kind", ""))
    return UsageMetricInput(
        input_tokens=_optional_int(row.get("input_tokens")),
        output_tokens=_optional_int(row.get("output_tokens")),
        cached_tokens=_optional_int(row.get("cached_tokens")),
        reasoning_tokens=_optional_int(row.get("reasoning_tokens")),
        latency_seconds=_as_float(row.get("latency_seconds")),
        retry_count=_as_int(row.get("retry_count")),
        terminal_failure=row.get("error_class") is not None,
        component=component,
        input_count=_as_int(row.get("input_count", 1)),
        usage_observed=bool(row.get("usage_observed", False)),
    )


def _append_reranker_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    context: Mapping[str, object],
    trace: RetrievalTrace,
) -> None:
    result = trace.rerank_result
    if result is None:
        return
    call_id = f"reranker:{trace.trace_id}"
    if call_id in seen_calls:
        return
    seen_calls.add(call_id)
    rows.append(
        {
            **context,
            "sceu_id": trace.sceu_id,
            "opportunity_id": trace.opportunity_id,
            "checkpoint_session": trace.checkpoint_session,
            "call_id": call_id,
            "call_kind": "reranker",
            "provider": "local_tei",
            "model_id": result.model,
            "model_revision": result.revision,
            "endpoint_identity": "local://tei-reranker",
            "provider_request_id": None,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "policy_request_hash": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "usage_observed": False,
            "input_count": result.input_count,
            "latency_seconds": result.latency_seconds,
            "retry_count": 0,
            "format_repair_used": False,
            "error_class": None,
            "started_at_utc": None,
            "ended_at_utc": None,
        }
    )


def _append_prefix_reranker_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    context: Mapping[str, object],
    *,
    checkpoint_session: int,
    trace: CommonRerankTrace,
) -> None:
    """Export one benchmark-owned prefix rerank to the API ledger.

    Schema-v2 evaluation tasks reference an immutable prefix artifact instead
    of copying its retrieval traces into every task result.  Consequently the
    generic task-trace exporter above cannot see these calls.  Keep them bound
    to the common-rerank readout so native cells do not inherit a reranker cost
    they did not use.
    """
    result = trace.result
    task_id = str(context.get("task_id", ""))
    call_id = (
        f"reranker-prefix:{task_id}:{checkpoint_session}:"
        f"{trace.opportunity_id}:{result.request_hash}"
    )
    if call_id in seen_calls:
        return
    seen_calls.add(call_id)
    rows.append(
        {
            **context,
            "readout": "common_rerank",
            "opportunity_id": trace.opportunity_id,
            "checkpoint_session": checkpoint_session,
            "call_id": call_id,
            "call_kind": "reranker",
            "provider": "local_tei",
            "model_id": result.model,
            "model_revision": result.revision,
            "endpoint_identity": "local://tei-reranker",
            "provider_request_id": None,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "policy_request_hash": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "usage_observed": False,
            "input_count": result.input_count,
            "latency_seconds": result.latency_seconds,
            "retry_count": 0,
            "format_repair_used": False,
            "error_class": None,
            "started_at_utc": None,
            "ended_at_utc": None,
        }
    )


def _task_context(task: QualificationTaskResult) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "episode_id": task.episode_id,
        "policy_profile_id": task.policy_profile_id,
        "condition": task.condition,
    }


def _evaluation_trace_id(
    task_id: str,
    sceu_id: str,
    readout: str,
    existing: str | None,
) -> str | None:
    """Return a stable report trace ID for one schema-v2 SCEU/readout.

    Common-rerank rows already carry the evaluator trace ID.  Native rows in
    early schema-v2 result files intentionally left that field empty because
    their order is directly inherited from the frozen candidate search.  The
    report still needs a distinct trace record for validation and provenance,
    so synthesize one without changing the original task result bytes.
    """
    if existing:
        return existing
    if readout == "native":
        return f"{task_id}:{sceu_id}:native"
    return None


def _prefix_artifact_for_task(
    task: object,
    artifacts: Mapping[str, object],
) -> MemoryPrefixArtifact | None:
    condition = str(getattr(task, "condition", ""))
    episode_id = str(getattr(task, "episode_id", ""))
    backend = "mem0" if condition in {"mem0_controlled", "mem0_native"} else condition
    for key in (
        f"{episode_id}--{backend}",
        backend,
        f"{episode_id}--{condition}",
        condition,
    ):
        raw = artifacts.get(key)
        if raw is None:
            continue
        if isinstance(raw, MemoryPrefixArtifact):
            return raw
        if isinstance(raw, Mapping):
            try:
                return MemoryPrefixArtifact.from_dict(raw)
            except Exception:
                return None
    return None


def _summary(
    matrix: QualificationMatrixResult,
    rows: Mapping[str, Sequence[dict[str, object]]],
    *,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    expected_task_count: int | None = None,
) -> dict[str, object]:
    statuses: dict[str, int] = defaultdict(int)
    condition_statuses: dict[str, int] = defaultdict(int)
    for task in matrix.task_results:
        statuses[task.status] += 1
        for condition in task.condition_results:
            condition_statuses[condition.status] += 1
    evaluated_episode_ids = sorted(
        {str(getattr(task, "episode_id", "")) for task in matrix.task_results}
    )
    evaluated_specs = tuple(
        specs[episode_id]
        for episode_id in evaluated_episode_ids
        if episode_id in specs
    )
    evaluated_group_ids = tuple(
        sorted(
            {
                spec.plan.metadata_dict.get("counterfactual_group_id", "")
                for spec in evaluated_specs
                if spec.plan.metadata_dict.get("counterfactual_group_id", "")
            }
        )
    )
    frozen_group_ids = tuple(
        sorted(
            {
                spec.plan.metadata_dict.get("counterfactual_group_id", "")
                for spec in specs.values()
                if spec.plan.metadata_dict.get("counterfactual_group_id", "")
            }
        )
    )
    evaluated_panel_ids = tuple(
        sorted(
            {
                spec.plan.metadata_dict.get("horizon_panel_id", "")
                for spec in evaluated_specs
                if spec.plan.metadata_dict.get("horizon_panel_id", "")
            }
        )
    )
    frozen_panel_ids = tuple(
        sorted(
            {
                spec.plan.metadata_dict.get("horizon_panel_id", "")
                for spec in specs.values()
                if spec.plan.metadata_dict.get("horizon_panel_id", "")
            }
        )
    )
    fully_grouped = bool(evaluated_specs) and all(
        spec.plan.metadata_dict.get("counterfactual_group_id", "")
        for spec in evaluated_specs
    )
    fully_panelled = bool(evaluated_specs) and all(
        spec.plan.metadata_dict.get("horizon_panel_id", "")
        for spec in evaluated_specs
    )
    primary_analysis_unit = (
        "horizon_panel"
        if fully_panelled
        else ("counterfactual_group" if fully_grouped else "episode")
    )
    observed_task_count = len(matrix.task_results)
    planned_task_count = (
        observed_task_count if expected_task_count is None else expected_task_count
    )
    if planned_task_count < observed_task_count:
        raise ValueError("expected_task_count cannot be smaller than observed task results")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_identity": matrix.run_identity,
        "n_tasks": observed_task_count,
        "n_planned_tasks": planned_task_count,
        "n_observed_task_results": observed_task_count,
        "n_missing_task_results": planned_task_count - observed_task_count,
        "n_evaluated_episodes": len(evaluated_episode_ids),
        "n_frozen_dataset_episodes": len(specs),
        "construct_mode": (
            "horizon_panels"
            if fully_panelled
            else ("matched_triplets" if fully_grouped else "mixed")
        ),
        "primary_analysis_unit": primary_analysis_unit,
        "n_physical_episodes": len(evaluated_episode_ids),
        "n_frozen_physical_episodes": len(specs),
        "n_counterfactual_groups": len(evaluated_group_ids),
        "n_frozen_counterfactual_groups": len(frozen_group_ids),
        "n_horizon_panels": len(evaluated_panel_ids),
        "n_frozen_horizon_panels": len(frozen_panel_ids),
        "n_statistical_units": (
            len(evaluated_panel_ids)
            if primary_analysis_unit == "horizon_panel"
            else (
                len(evaluated_group_ids)
                if primary_analysis_unit == "counterfactual_group"
                else len(evaluated_episode_ids)
            )
        ),
        "counterfactual_group_ids": list(evaluated_group_ids),
        "horizon_panel_ids": list(evaluated_panel_ids),
        "evaluated_episode_ids": evaluated_episode_ids,
        "task_status_counts": dict(sorted(statuses.items())),
        "condition_status_counts": dict(sorted(condition_statuses.items())),
        "n_sceu_results": len(rows["sceu_results.jsonl"]),
        "n_memory_events": len(rows["memory_events.jsonl"]),
        "n_inventory_snapshots": len(rows["memory_inventory.jsonl"]),
        "n_retrieval_traces": len(rows["retrieval_trace.jsonl"]),
        "n_interventions": len(rows["interventions.jsonl"]),
        "n_api_calls": len(rows["api_usage.jsonl"]),
        "n_policy_calls": len(rows["policy_calls.jsonl"]),
        "n_memory_internal_calls": sum(
            row.get("call_kind") == "memory_internal_llm" for row in rows["api_usage.jsonl"]
        ),
        "n_embedding_calls": sum(
            row.get("call_kind") == "embedding" for row in rows["api_usage.jsonl"]
        ),
        "n_reranker_calls": sum(
            row.get("call_kind") == "reranker" for row in rows["api_usage.jsonl"]
        ),
        "n_prefix_manifests": len(rows["prefix_manifests.jsonl"]),
        "n_graph_diagnostics": len(rows["graph_diagnostics.jsonl"]),
        "storage_provenance": _storage_provenance_diagnostics(rows),
        "semantic_attribution": _semantic_attribution_diagnostics(rows),
        "n_memory_count_contrasts": sum(
            _is_memory_count_load_contrast(row.get("count_contrast"))
            and row.get("intervention_kind") in {"count_add", "count_contrast"}
            for row in rows["interventions.jsonl"]
        ),
    }


def _expected_task_count(
    run_metadata: Mapping[str, object] | None,
    observed_task_count: int,
) -> int:
    if run_metadata is None or "evaluation_task_count" not in run_metadata:
        return observed_task_count
    raw = run_metadata["evaluation_task_count"]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError("evaluation_task_count must be a non-negative integer")
    if raw < observed_task_count:
        raise ValueError("evaluation_task_count cannot be smaller than observed results")
    return raw


def _storage_provenance_diagnostics(
    rows: Mapping[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    counts: dict[str, int] = {"native/exact": 0, "inferred": 0, "unavailable": 0}
    by_source: dict[str, int] = defaultdict(int)
    event_counts_by_checkpoint: Counter[tuple[str, int]] = Counter()
    for row in rows["memory_events.jsonl"]:
        mode = str(row.get("provenance_mode", ""))
        if mode not in counts:
            mode = "inferred" if mode else "unavailable"
        counts[mode] += 1
        source = str(row.get("source", "unknown"))
        by_source[source] += 1
        event_counts_by_checkpoint[
            (
                str(row.get("task_id", "")),
                _as_int(row.get("session_index", -1)) + 1,
            )
        ] += 1

    inventories_by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows["memory_inventory.jsonl"]:
        inventories_by_task[str(row.get("task_id", ""))].append(row)
    incomplete_checkpoints: list[dict[str, object]] = []
    for task_id, inventories in sorted(inventories_by_task.items()):
        previous_n_write = 0
        for row in sorted(
            inventories,
            key=lambda item: _as_int(item.get("checkpoint_session", 0)),
        ):
            checkpoint_session = _as_int(row.get("checkpoint_session", 0))
            current_n_write = _as_int(row.get("n_write", 0))
            write_delta = max(0, current_n_write - previous_n_write)
            previous_n_write = current_n_write
            event_count = event_counts_by_checkpoint[(task_id, checkpoint_session)]
            if write_delta <= event_count:
                continue
            incomplete_checkpoints.append(
                {
                    "task_id": task_id,
                    "checkpoint_session": checkpoint_session,
                    "write_delta": write_delta,
                    "event_count": event_count,
                }
            )
    incomplete_tasks = sorted({str(item["task_id"]) for item in incomplete_checkpoints})
    return {
        "native_exact_event_count": counts["native/exact"],
        "inferred_event_count": counts["inferred"],
        "unavailable_event_count": counts["unavailable"],
        "event_source_counts": dict(sorted(by_source.items())),
        "incomplete_write_tasks": incomplete_tasks,
        "incomplete_write_checkpoints": incomplete_checkpoints,
        "status": (
            "complete" if counts["unavailable"] == 0 and not incomplete_tasks else "incomplete"
        ),
    }


def _semantic_attribution_diagnostics(
    rows: Mapping[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    """Summarize final inventories on a separate semantic-attribution axis."""
    latest_by_task: dict[str, dict[str, object]] = {}
    for row in rows["memory_inventory.jsonl"]:
        task_id = str(row.get("task_id", ""))
        checkpoint = _as_int(row.get("checkpoint_session", -1))
        previous = latest_by_task.get(task_id)
        if previous is None or checkpoint > _as_int(previous.get("checkpoint_session", -1)):
            latest_by_task[task_id] = row

    method_counts: Counter[str] = Counter()
    lifecycle_counts: Counter[str] = Counter()
    cross_counts: Counter[str] = Counter()
    incomplete_objects: list[str] = []
    positive = 0
    total = 0
    for task_id, row in sorted(latest_by_task.items()):
        raw = row.get("evaluator_attribution_by_memory")
        if not isinstance(raw, Mapping):
            continue
        for memory_id, value in sorted(raw.items(), key=lambda item: str(item[0])):
            total += 1
            if not isinstance(value, Mapping):
                method = "unavailable"
                lifecycle = "unavailable"
                contributes = False
                incomplete_objects.append(f"{task_id}:{memory_id}")
            else:
                method = str(value.get("method", "unavailable"))
                lifecycle = str(value.get("provenance_mode", "unavailable"))
                contributes = value.get("contributes_positive_coverage") is True
                if method not in {
                    "exact_signature",
                    "multi_signature",
                    "lexical_signature",
                    "unique_provenance",
                    "no_match",
                    "ambiguous",
                }:
                    method = "unavailable"
                if lifecycle not in {"native/exact", "inferred", "unavailable"}:
                    lifecycle = "unavailable"
                if (
                    "method" not in value
                    or "provenance_mode" not in value
                    or not isinstance(value.get("contributes_positive_coverage"), bool)
                ):
                    incomplete_objects.append(f"{task_id}:{memory_id}")
            method_counts[method] += 1
            lifecycle_counts[lifecycle] += 1
            cross_counts[f"{lifecycle}|{method}"] += 1
            positive += contributes
    return {
        "scope": "latest_inventory_per_task",
        "n_tasks": len(latest_by_task),
        "n_memory_objects": total,
        "method_counts": dict(sorted(method_counts.items())),
        "lifecycle_provenance_counts": dict(sorted(lifecycle_counts.items())),
        "lifecycle_by_semantic_method": dict(sorted(cross_counts.items())),
        "positive_coverage_rate": None if total == 0 else positive / total,
        "incomplete_objects": incomplete_objects,
        "status": "complete" if not incomplete_objects else "incomplete",
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _scorecard_rows(
    matrix: QualificationMatrixResult,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str],
        list[_ScorecardObservation],
    ] = defaultdict(list)
    for task in matrix.task_results:
        for condition in task.condition_results:
            for row in condition.sceu_results:
                key = (
                    task.policy_profile_id,
                    condition.condition,
                    condition.readout,
                )
                grouped[key].append(
                    _ScorecardObservation(
                        policy_profile_id=task.policy_profile_id,
                        condition=condition.condition,
                        readout=condition.readout,
                        status=condition.status,
                        row=row,
                    )
                )
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        observations = grouped[key]
        rows = [item.row for item in observations]
        interventions = [intervention for row in rows for intervention in row.interventions]
        causal = [
            intervention.classification.label
            for intervention in interventions
            if intervention.intervention_kind == "leave_one_out"
        ]
        intervention_labels = [intervention.classification.label for intervention in interventions]
        count_contrasts = [
            intervention
            for intervention in interventions
            if _is_memory_count_contrast(getattr(intervention, "count_contrast", None))
            or getattr(intervention, "intervention_kind", None) == "count_contrast"
        ]
        eligible_by_flag = {
            flag: [
                row
                for row in rows
                if getattr(row, "drift_eligible_categories", None) is None
                or flag in getattr(row, "drift_eligible_categories", ())
            ]
            for flag in _CANONICAL_DRIFT_CATEGORIES
        }
        aggregate_eligible = [
            row
            for row in rows
            if getattr(row, "drift_eligible_categories", None) is None
            or bool(getattr(row, "drift_eligible_categories", ()))
        ]

        def has_targeted_drift(row: SCEURunResult) -> bool:
            eligible = getattr(row, "drift_eligible_categories", None)
            targeted = set(_CANONICAL_DRIFT_CATEGORIES) if eligible is None else set(eligible)
            return bool(targeted.intersection(row.normalized_drift_flags))

        def has_canonical_drift_violation(row: SCEURunResult) -> bool:
            return bool(set(_CANONICAL_DRIFT_CATEGORIES).intersection(row.normalized_drift_flags))

        def has_off_target_drift(row: SCEURunResult) -> bool:
            eligible = getattr(row, "drift_eligible_categories", None)
            if eligible is None:
                return False
            observed = set(_CANONICAL_DRIFT_CATEGORIES).intersection(row.normalized_drift_flags)
            return bool(observed.difference(eligible))

        targeted_rates = {
            flag: _ratio_value(
                sum(flag in row.normalized_drift_flags for row in eligible_by_flag[flag]),
                len(eligible_by_flag[flag]),
            )
            for flag in _CANONICAL_DRIFT_CATEGORIES
        }
        observed_rates = {
            flag: _ratio_value(
                sum(flag in row.normalized_drift_flags for row in rows),
                len(rows),
            )
            for flag in _CANONICAL_DRIFT_CATEGORIES
        }
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "status": _aggregate_status(item.status for item in observations),
                "n_sceu": len(rows),
                "mean_behavior_score": _ratio_value(
                    sum(row.behavior.behavior_score for row in rows),
                    len(rows),
                ),
                "behavior_correct_rate": _ratio_value(
                    sum(row.behavior.is_correct for row in rows),
                    len(rows),
                ),
                "baseline_stability_rate": _ratio_value(
                    sum(row.baseline_stable for row in rows),
                    len(rows),
                ),
                "mean_visible_memory_count": _ratio_value(
                    sum(len(row.model_visible_memory_ids) for row in rows),
                    len(rows),
                ),
                "mean_live_memory_count": _ratio_value(
                    sum(
                        _live_memory_count_or_zero(row)
                        for row in rows
                        if _live_memory_count_from_row(row) is not None
                    ),
                    sum(1 for row in rows if _live_memory_count_from_row(row) is not None),
                ),
                "causal_memory_use_rate": _ratio_value(
                    sum(
                        label
                        in {
                            "beneficial",
                            "harmful",
                            "causal_direction_ambiguous",
                        }
                        for label in causal
                    ),
                    len(causal),
                ),
                "unique_causal_effect_rate": _ratio_value(
                    sum(
                        label
                        in {
                            "beneficial",
                            "harmful",
                            "causal_direction_ambiguous",
                        }
                        for label in causal
                    ),
                    len(causal),
                ),
                "beneficial_intervention_rate": _ratio_value(
                    intervention_labels.count("beneficial"),
                    len(intervention_labels),
                ),
                "harmful_intervention_rate": _ratio_value(
                    intervention_labels.count("harmful"),
                    len(intervention_labels),
                ),
                "unstable_intervention_rate": _ratio_value(
                    sum(
                        label in {"unstable_baseline", "intervention_unstable"}
                        for label in intervention_labels
                    ),
                    len(intervention_labels),
                ),
                "sham_replacement_action_flip_rate": _ratio_value(
                    sum(
                        bool(getattr(item.classification, "action_changed", False))
                        for row in rows
                        for item in row.interventions
                        if getattr(item, "intervention_kind", "") == "sham_replacement"
                    ),
                    sum(
                        1
                        for row in rows
                        for item in row.interventions
                        if getattr(item, "intervention_kind", "") == "sham_replacement"
                    ),
                ),
                "constraint_loss_rate": targeted_rates["constraint_loss"],
                "constraint_loss_eligible_n": len(eligible_by_flag["constraint_loss"]),
                "targeted_constraint_loss_rate": targeted_rates["constraint_loss"],
                "observed_constraint_loss_rate": observed_rates["constraint_loss"],
                "canonical_constraint_loss_violation_rate": observed_rates[
                    "constraint_loss"
                ],
                "current_plan_deviation_rate": targeted_rates["plan_deviation"],
                "plan_deviation_eligible_n": len(eligible_by_flag["plan_deviation"]),
                "targeted_plan_deviation_rate": targeted_rates["plan_deviation"],
                "observed_plan_deviation_rate": observed_rates["plan_deviation"],
                "canonical_plan_deviation_violation_rate": observed_rates[
                    "plan_deviation"
                ],
                "stale_state_action_rate": targeted_rates["stale_state"],
                "stale_state_eligible_n": len(eligible_by_flag["stale_state"]),
                "targeted_stale_state_rate": targeted_rates["stale_state"],
                "observed_stale_state_rate": observed_rates["stale_state"],
                "canonical_stale_state_violation_rate": observed_rates[
                    "stale_state"
                ],
                "local_over_global_rate": targeted_rates["local_over_global"],
                "local_over_global_eligible_n": len(eligible_by_flag["local_over_global"]),
                "targeted_local_over_global_rate": targeted_rates["local_over_global"],
                "observed_local_over_global_rate": observed_rates["local_over_global"],
                "canonical_local_over_global_violation_rate": observed_rates[
                    "local_over_global"
                ],
                "aggregate_drift_rate": _ratio_value(
                    sum(has_targeted_drift(row) for row in aggregate_eligible),
                    len(aggregate_eligible),
                ),
                "aggregate_drift_eligible_n": len(aggregate_eligible),
                "targeted_aggregate_drift_rate": _ratio_value(
                    sum(has_targeted_drift(row) for row in aggregate_eligible),
                    len(aggregate_eligible),
                ),
                "observed_aggregate_drift_rate": _ratio_value(
                    sum(has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
                "canonical_drift_violation_rate": _ratio_value(
                    sum(has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
                "off_target_drift_rate": _ratio_value(
                    sum(has_off_target_drift(row) for row in rows),
                    len(rows),
                ),
                "off_target_drift_n": sum(has_off_target_drift(row) for row in rows),
                "memory_count_contrast_rate": _ratio_value(
                    sum(
                        bool(getattr(item.classification, "action_changed", False))
                        for item in count_contrasts
                    ),
                    len(count_contrasts),
                ),
                "memory_count_behavior_change_rate": _ratio_value(
                    sum(
                        bool(getattr(item.classification, "action_changed", False))
                        or bool(getattr(item.classification, "checker_changed", False))
                        for item in count_contrasts
                    ),
                    len(count_contrasts),
                ),
            }
        )
    return output


def _metrics_by_cell(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                task.policy_profile_id,
                condition.condition,
                condition.readout,
            )
            for task in matrix.task_results
            for condition in task.condition_results
        }
    )
    groups: list[dict[str, object]] = []
    for policy_profile_id, condition_name, readout in keys:
        selected_tasks: list[QualificationTaskResult] = []
        for task in matrix.task_results:
            if task.policy_profile_id != policy_profile_id or task.condition != condition_name:
                continue
            selected_conditions = tuple(
                condition
                for condition in task.condition_results
                if condition.condition == condition_name and condition.readout == readout
            )
            if not selected_conditions:
                continue
            traces = task.retrieval_traces
            if readout != "common_rerank":
                traces = tuple(replace(trace, rerank_result=None) for trace in traces)
            selected_tasks.append(
                replace(
                    task,
                    condition_results=selected_conditions,
                    retrieval_traces=traces,
                )
            )
        selected_matrix = QualificationMatrixResult(
            run_identity=matrix.run_identity,
            task_results=tuple(selected_tasks),
        )
        groups.append(
            {
                "policy_profile_id": policy_profile_id,
                "condition": condition_name,
                "readout": readout,
                "metrics": compute_qualification_metrics(
                    selected_matrix,
                    specs,
                ).to_dict(),
            }
        )
    return groups


def _storage_scorecard_rows(
    groups: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Deduplicate readouts and expose lifecycle metrics by provenance track."""
    preferred: dict[tuple[str, str], Mapping[str, object]] = {}
    memory_conditions = {"flat_retrieval", "mem0", "amem", "memos"}
    for group in groups:
        condition = str(group.get("condition", ""))
        if condition not in memory_conditions:
            continue
        key = (str(group.get("policy_profile_id", "")), condition)
        existing = preferred.get(key)
        if existing is None or (
            str(group.get("readout", "")) == "common_rerank"
            and str(existing.get("readout", "")) != "common_rerank"
        ):
            preferred[key] = group

    metric_names = (
        "write_coverage",
        "write_selectivity",
        "current_state_storage_precision",
        "current_state_storage_recall",
        "current_state_storage_f1",
        "stale_state_retention_rate",
        "update_delete_responsiveness",
        "physical_retirement_rate",
        "superseding_state_storage_rate",
        "write_to_continuation_alignment",
        "storage_provenance_completeness",
        "live_memory_count",
        "native_objects_per_logical_state_unit",
    )
    rows: list[dict[str, object]] = []
    for (policy_profile_id, condition), group in sorted(preferred.items()):
        raw_metrics = group.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        for track, prefix in (
            ("all", ""),
            ("exact", "storage_exact_"),
            ("inferred", "storage_inferred_"),
        ):
            row: dict[str, object] = {
                "policy_profile_id": policy_profile_id,
                "condition": condition,
                "provenance_track": track,
                "source_readout": str(group.get("readout", "")),
            }
            for name in metric_names:
                row[name] = _metric_value(metrics, f"{prefix}{name}")
            ambiguous = _metric_value(
                metrics,
                f"{prefix}semantic_attribution_ambiguous_rate",
            )
            unavailable = _metric_value(
                metrics,
                f"{prefix}semantic_attribution_unavailable_rate",
            )
            row["semantic_attribution_resolvability"] = (
                None
                if ambiguous is None and unavailable is None
                else max(0.0, 1.0 - (ambiguous or 0.0) - (unavailable or 0.0))
            )
            rows.append(row)
    return rows


def _memory_count_scorecard_rows(
    intervention_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate only matched evaluator-controlled count-load contrasts."""

    grouped: dict[
        tuple[str, str, str, str, int],
        list[tuple[int, int, bool, bool]],
    ] = defaultdict(list)
    for row in intervention_rows:
        if str(row.get("intervention_kind", "")) != "count_add":
            continue
        if not _is_memory_count_load_contrast(row.get("count_contrast")):
            continue
        baseline = _as_int(row.get("baseline_memory_count", 0))
        intervention = _as_int(row.get("intervention_memory_count", 0))
        delta = intervention - baseline
        if delta <= 0:
            continue
        raw_classification = row.get("classification")
        classification = raw_classification if isinstance(raw_classification, Mapping) else {}
        key = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("opportunity_id", "")),
            delta,
        )
        grouped[key].append(
            (
                baseline,
                intervention,
                bool(classification.get("action_changed", False)),
                bool(classification.get("action_changed", False))
                or bool(classification.get("checker_changed", False)),
            )
        )

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        policy, condition, readout, opportunity, delta = key
        values = grouped[key]
        n = len(values)
        output.append(
            {
                "policy_profile_id": policy,
                "condition": condition,
                "readout": readout,
                "opportunity_id": opportunity,
                "count_delta": delta,
                "n_contrasts": n,
                "action_flip_rate": _ratio_value(
                    sum(value[2] for value in values),
                    n,
                ),
                "behavior_change_rate": _ratio_value(
                    sum(value[3] for value in values),
                    n,
                ),
                "mean_baseline_visible_memory_count": _ratio_value(
                    sum(value[0] for value in values),
                    n,
                ),
                "mean_intervention_visible_memory_count": _ratio_value(
                    sum(value[1] for value in values),
                    n,
                ),
            }
        )
    return output


def _metric_value(metrics: Mapping[str, object], name: str) -> float | None:
    raw = metrics.get(name)
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _aggregate_status(statuses: Sequence[str] | Any) -> str:
    values = set(statuses)
    if not values:
        return "unknown"
    if values == {"complete"}:
        return "complete"
    if "failed" in values:
        return "failed"
    return "partial"


def _live_memory_count_from_row(row: object) -> int | None:
    value = getattr(row, "live_memory_count", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _live_memory_count_or_zero(row: object) -> int:
    value = _live_memory_count_from_row(row)
    return 0 if value is None else value


def _ratio_value(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _scorecard_csv(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_SCORECARD_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _display_value(row.get(field)) for field in _SCORECARD_FIELDS})
    return stream.getvalue()


def _scorecard_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    header = "| " + " | ".join(_SCORECARD_FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in _SCORECARD_FIELDS) + " |"
    body = [
        "| " + " | ".join(_display_value(row.get(field)) for field in _SCORECARD_FIELDS) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _table_csv(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _display_value(row.get(field)) for field in fields})
    return stream.getvalue()


def _table_markdown(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    *,
    title: str,
    note: str,
) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_display_value(row.get(field)) for field in fields) + " |" for row in rows
    ]
    return "\n".join((f"# {title}", "", note, "", header, divider, *body, ""))


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _jsonl_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    ordered = sorted(
        (_jsonable(row) for row in rows),
        key=lambda row: json.dumps(row, sort_keys=True, default=str),
    )
    if not ordered:
        return b""
    return (
        "\n".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
            for row in ordered
        )
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(child) for child in value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_REPORT_ARTIFACTS",
    "ReportArtifacts",
    "write_qualification_report",
]
