"""Schema, hash, identity, and trace-chain validation for report artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lhmsb.longhorizon.task_span import MIN_LONG_HORIZON_EFFECTIVE_STEPS
from lhmsb.qualification.analysis_phase import (
    AnalysisPhase,
    AnalysisPhaseError,
    AnalysisTiming,
    parse_analysis_phase,
    parse_analysis_timing,
    validate_analysis_phase,
)
from lhmsb.qualification.conditions import condition_definition
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.contribution_evidence import (
    CONTRIBUTION_EVIDENCE_SCHEMA_VERSION,
    build_contribution_evidence,
)
from lhmsb.qualification.design_audit import (
    EXPERIMENT_DESIGN_AUDIT_SCHEMA_VERSION,
    EXPERIMENT_DESIGN_CHECK_IDS,
    build_analysis_contract,
)
from lhmsb.qualification.fault_profile import (
    FAULT_PROFILE_DIVERGENCE_SCHEMA_VERSION,
    FaultProfileAlignmentError,
    compute_fault_profile_divergence,
)
from lhmsb.qualification.horizon_panel import (
    HORIZON_CONSTRUCTS,
    HORIZON_LEVELS,
    HORIZON_PRIMARY_ESTIMANDS,
    HORIZON_SECONDARY_ESTIMANDS,
)
from lhmsb.qualification.report import (
    REPORT_SCHEMA_VERSION,
    REQUIRED_REPORT_ARTIFACTS,
)
from lhmsb.qualification.statistics import (
    HORIZON_ALL_ESTIMANDS,
    HORIZON_MULTIPLICITY_SCOPE,
    HORIZON_PAIRED_TEST,
    HORIZON_PRIMARY_ANALYSIS_UNIT,
    HORIZON_PRIMARY_EFFECT_DIRECTION,
    HORIZON_PRIMARY_WORKSPACE_ADJUSTMENT,
    HORIZON_STATISTICS_SCHEMA_VERSION,
    MATCHED_ALL_ESTIMANDS,
    MATCHED_DRIFT_SCOPE,
    MATCHED_MULTIPLICITY_SCOPE,
    MATCHED_PAIRED_TEST,
    MATCHED_PRIMARY_ANALYSIS_UNIT,
    MATCHED_PRIMARY_EFFECT_DIRECTION,
    MATCHED_PRIMARY_ESTIMANDS,
    MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT,
    MATCHED_SECONDARY_ESTIMANDS,
    MATCHED_STATISTICS_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class ArtifactValidationReport:
    ok: bool
    errors: tuple[str, ...]
    checked_artifacts: int
    run_identity: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "checked_artifacts": self.checked_artifacts,
            "run_identity": self.run_identity,
        }


def validate_qualification_artifacts(
    report_directory: Path,
    *,
    expected_run_identity: str | None = None,
) -> ArtifactValidationReport:
    """Validate every artifact without trusting manifest-declared hashes."""
    errors: list[str] = []
    missing = [
        name for name in REQUIRED_REPORT_ARTIFACTS if not (report_directory / name).is_file()
    ]
    errors.extend(f"missing required artifact: {name}" for name in missing)
    manifest = _read_json(
        report_directory / "run_manifest.json",
        errors,
        required=False,
    )
    if manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append("run manifest has an unsupported report schema version")
    run_identity = _optional_text(manifest.get("run_identity"))
    if expected_run_identity is not None and run_identity != expected_run_identity:
        errors.append(
            f"run identity mismatch: expected {expected_run_identity}, got {run_identity}"
        )
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        errors.append("run manifest artifact_hashes must be an object")
        hashes = {}
    checked = 0
    for raw_name, raw_expected in sorted(hashes.items()):
        name = str(raw_name)
        path = report_directory / name
        if name == "run_manifest.json":
            errors.append("run manifest must not self-declare its own hash")
            continue
        if not path.is_file():
            errors.append(f"manifest-hashed artifact is missing: {name}")
            continue
        checked += 1
        actual = _sha256(path)
        if actual != str(raw_expected):
            errors.append(f"artifact hash mismatch: {name}")

    jsonl: dict[str, list[dict[str, object]]] = {}
    for name in (
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
        "long_horizon_constructs.jsonl",
        "decision_attribution.jsonl",
        "task_span.jsonl",
        "matched_construct_contrasts.jsonl",
        "horizon_panel_contrasts.jsonl",
    ):
        jsonl[name] = _read_jsonl(report_directory / name, errors)
        canonical = sorted(
            jsonl[name],
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        if jsonl[name] != canonical:
            errors.append(f"JSONL rows are not deterministically sorted: {name}")

    construct_keys: set[tuple[str, str]] = set()
    for row in jsonl["long_horizon_constructs.jsonl"]:
        construct_key = (
            str(row.get("episode_id", "")),
            str(row.get("sceu_id", "")),
        )
        if not all(construct_key):
            errors.append("long-horizon construct row requires episode_id and sceu_id")
            continue
        if construct_key in construct_keys:
            errors.append(
                f"duplicate long-horizon construct identity: {construct_key}"
            )
        construct_keys.add(construct_key)
        checkpoint = row.get("checkpoint_session")
        handoffs = row.get("handoff_count")
        if checkpoint != handoffs:
            errors.append(
                "handoff/checkpoint mismatch for long-horizon construct "
                f"{construct_key}"
            )
        current_required = set(
            _string_list(
                row.get("current_required_state_ids"),
                f"{construct_key} current_required_state_ids",
                errors,
            )
        )
        future_referenced = set(
            _string_list(
                row.get("future_referenced_state_ids"),
                f"{construct_key} future_referenced_state_ids",
                errors,
            )
        )
        if current_required.intersection(future_referenced):
            errors.append(
                "future state counted as current requirement for long-horizon construct "
                f"{construct_key}"
            )

    _validate_decision_attributions(
        jsonl["decision_attribution.jsonl"],
        errors,
    )
    fault_profile_divergence = _read_json(
        report_directory / "fault_profile_divergence.json",
        errors,
        required=False,
    )
    _validate_fault_profile_divergence(
        fault_profile_divergence,
        decision_attribution_rows=jsonl["decision_attribution.jsonl"],
        errors=errors,
    )
    _validate_task_spans(jsonl["task_span.jsonl"], errors)
    _validate_matched_construct_contrasts(
        jsonl["matched_construct_contrasts.jsonl"],
        errors,
    )
    _validate_horizon_panel_contrasts(
        jsonl["horizon_panel_contrasts.jsonl"],
        errors,
    )

    task_ids = _unique_ids(
        jsonl["tasks.jsonl"],
        "task_id",
        "task",
        errors,
    )
    result_task_ids = {
        str(row.get("task_id", "")) for row in jsonl["task_results.jsonl"] if row.get("task_id")
    }
    unknown_result_tasks = sorted(result_task_ids - task_ids)
    if unknown_result_tasks:
        errors.append(f"task_results contain unknown task IDs: {unknown_result_tasks}")
    expected_task_count = manifest.get("evaluation_task_count")
    if expected_task_count is not None:
        if (
            isinstance(expected_task_count, bool)
            or not isinstance(expected_task_count, int)
            or expected_task_count < 0
        ):
            errors.append("run manifest evaluation_task_count must be a non-negative integer")
        else:
            if len(task_ids) != expected_task_count:
                errors.append(
                    "tasks.jsonl coverage does not match run manifest "
                    f"evaluation_task_count: {len(task_ids)}/{expected_task_count}"
                )
            if len(result_task_ids) != expected_task_count:
                errors.append(
                    "task_results.jsonl coverage does not match run manifest "
                    f"evaluation_task_count: {len(result_task_ids)}/{expected_task_count}"
                )

    traces = {
        str(row.get("trace_id")): row
        for row in jsonl["retrieval_trace.jsonl"]
        if row.get("trace_id")
    }
    sceu_keys: set[tuple[str, str, str]] = set()
    behaviorally_used_by_sceu: dict[
        tuple[str, str, str],
        set[str],
    ] = {}
    memory_tasks_with_sceu: set[str] = set()
    for row in jsonl["sceu_results.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"SCEU result references unknown task ID: {task_id}")
        result_id = str(row.get("result_id", ""))
        sceu_id = str(row.get("sceu_id", ""))
        key = (task_id, result_id, sceu_id)
        if key in sceu_keys:
            errors.append(f"duplicate SCEU result identity: {key}")
        sceu_keys.add(key)
        condition = str(row.get("condition", ""))
        readout = str(row.get("readout", ""))
        candidates = _string_list(
            row.get("candidate_memory_ids"),
            f"{key} candidate_memory_ids",
            errors,
        )
        retrieved = _string_list(
            row.get("retrieved_memory_ids"),
            f"{key} retrieved_memory_ids",
            errors,
        )
        visible = _string_list(
            row.get("model_visible_memory_ids"),
            f"{key} model_visible_memory_ids",
            errors,
        )
        backend_retrieved = _string_list(
            row.get("backend_retrieved_memory_ids", candidates),
            f"{key} backend_retrieved_memory_ids",
            errors,
        )
        selected = _string_list(
            row.get("selected_memory_ids", retrieved),
            f"{key} selected_memory_ids",
            errors,
        )
        behaviorally_used = _string_list(
            row.get("behaviorally_used_memory_ids", []),
            f"{key} behaviorally_used_memory_ids",
            errors,
        )
        behaviorally_used_by_sceu[key] = set(behaviorally_used)
        if not set(backend_retrieved).issubset(candidates):
            errors.append(f"backend-retrieved memories are not a subset of candidates for {key}")
        if not set(selected).issubset(backend_retrieved):
            errors.append(f"selected memories are not a subset of backend retrieval for {key}")
        if not set(visible).issubset(selected):
            errors.append(f"model-visible memories are not a subset of selection for {key}")
        if not set(behaviorally_used).issubset(visible):
            errors.append(f"behaviorally-used memories are not model-visible for {key}")
        if not set(retrieved).issubset(candidates):
            errors.append(f"retrieved memories are not a subset of candidates for {key}")
        if not set(visible).issubset(retrieved):
            errors.append(f"model-visible memories are not a subset of retrieved for {key}")
        if visible != retrieved[: len(visible)]:
            errors.append(f"model-visible ordering is not a prefix of retrieved for {key}")
        if readout == "native" and retrieved != candidates[: len(retrieved)]:
            errors.append(f"native retrieved ordering is not a candidate prefix for {key}")
        if condition in {"workspace_only", "oracle_current_state"} and any(
            (candidates, retrieved, visible)
        ):
            errors.append(f"non-memory condition exposes memory IDs for {key}")
        trace_id = row.get("retrieval_trace_id")
        try:
            is_memory_condition = condition_definition(condition).prefix_backend is not None
        except ValueError:
            errors.append(f"unknown condition in SCEU result: {condition}")
            is_memory_condition = False
        if is_memory_condition:
            memory_tasks_with_sceu.add(task_id)
            if not isinstance(trace_id, str) or trace_id not in traces:
                errors.append(f"Mem0 SCEU lacks a known retrieval trace for {key}")
            else:
                _validate_trace_match(row, traces[trace_id], key, errors)
        elif trace_id is not None:
            errors.append(f"non-memory SCEU unexpectedly references retrieval trace for {key}")

    inventory_tasks = {str(row.get("task_id", "")) for row in jsonl["memory_inventory.jsonl"]}
    trace_tasks = {str(row.get("task_id", "")) for row in jsonl["retrieval_trace.jsonl"]}
    for task_id in sorted(memory_tasks_with_sceu):
        if task_id not in inventory_tasks:
            errors.append(f"memory task lacks inventory snapshots: {task_id}")
        if task_id not in trace_tasks:
            errors.append(f"Mem0 task lacks retrieval traces: {task_id}")

    for row in jsonl["memory_events.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"memory event references unknown task ID: {task_id}")
    for row in jsonl["memory_inventory.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"inventory references unknown task ID: {task_id}")
    _validate_semantic_attributions(jsonl["memory_inventory.jsonl"], errors)
    for row in jsonl["retrieval_trace.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"retrieval trace references unknown task ID: {task_id}")

    causally_influential_neutral_by_sceu: dict[
        tuple[str, str, str],
        set[str],
    ] = {}
    for row in jsonl["interventions.jsonl"]:
        key = (
            str(row.get("task_id", "")),
            str(row.get("result_id", "")),
            str(row.get("sceu_id", "")),
        )
        if key not in sceu_keys:
            errors.append(f"intervention references unknown SCEU result: {key}")
        evaluations = row.get("evaluations")
        if not isinstance(evaluations, Sequence) or isinstance(
            evaluations,
            (str, bytes),
        ):
            errors.append(f"intervention evaluations must be an array for {key}")
        elif len(evaluations) != 2:
            errors.append(f"intervention must contain exactly two repeated evaluations for {key}")
        classification = row.get("classification")
        if not isinstance(classification, Mapping) or not classification.get("label"):
            errors.append(f"intervention classification is incomplete for {key}")
        elif _validate_intervention_classification(
            classification,
            key=key,
            errors=errors,
        ) and row.get("intervention_kind") == "neutral_replacement":
            target_memory_id = str(row.get("target_memory_id", ""))
            if not target_memory_id:
                errors.append(
                    "causally influential neutral replacement lacks a target "
                    f"memory for {key}"
                )
            else:
                causally_influential_neutral_by_sceu.setdefault(
                    key,
                    set(),
                ).add(target_memory_id)

    for key, claimed_ids in behaviorally_used_by_sceu.items():
        supported_ids = causally_influential_neutral_by_sceu.get(key, set())
        if claimed_ids != supported_ids:
            errors.append(
                "behaviorally-used memory IDs do not match repeat-stable "
                "neutral-replacement causal evidence for "
                f"{key}: claimed={sorted(claimed_ids)}, "
                f"supported={sorted(supported_ids)}"
            )

    _unique_ids(
        jsonl["api_usage.jsonl"],
        "call_id",
        "API call",
        errors,
        allow_empty=True,
    )
    policy_calls = jsonl.get("policy_calls.jsonl", [])
    policy_ids = _unique_ids(
        policy_calls,
        "call_id",
        "policy call",
        errors,
        allow_empty=False,
    )
    usage_by_id = {
        str(row.get("call_id")): row for row in jsonl["api_usage.jsonl"] if row.get("call_id")
    }
    schema_version = manifest.get("schema_version")
    policy_trace_schema_version = manifest.get("policy_trace_schema_version")
    if (
        isinstance(schema_version, int)
        and schema_version >= 3
        and policy_trace_schema_version not in {1, 2}
    ):
        errors.append("run manifest lacks a supported policy_trace_schema_version")
    strict_policy_routes = bool(
        isinstance(manifest.get("policy_routes"), Mapping)
        or policy_trace_schema_version == 2
    )
    for row in policy_calls:
        call_id = str(row.get("call_id", ""))
        required_fields = (
            "provider",
            "model_id",
            "route_id",
            "endpoint_identity",
            "request_hash",
            "response_hash",
            "policy_request_hash",
        )
        if strict_policy_routes:
            for field in required_fields:
                if not isinstance(row.get(field), str) or not row[field]:
                    errors.append(f"policy call {call_id} lacks {field}")
        usage = usage_by_id.get(call_id)
        if usage is None:
            errors.append(f"policy call references unknown API call: {call_id}")
            continue
        if strict_policy_routes:
            for field in required_fields:
                if row.get(field) != usage.get(field):
                    errors.append(f"policy/API usage mismatch for {call_id}: {field}")
    if set(policy_ids) != {
        str(row.get("call_id"))
        for row in jsonl["api_usage.jsonl"]
        if row.get("policy_request_hash")
    }:
        errors.append("policy_calls coverage does not match api_usage policy calls")
    _read_json(report_directory / "metrics.json", errors, required=False)
    metrics_by_cell = _read_json(
        report_directory / "metrics_by_cell.json",
        errors,
        required=False,
    )
    _validate_metrics_by_cell(
        metrics_by_cell,
        jsonl["task_results.jsonl"],
        errors,
    )
    summary = _read_json(
        report_directory / "summary.json", errors, required=False
    )
    _validate_summary_analysis_unit(summary, errors)
    for summary_field, profile_field in (
        ("n_fault_profile_aligned_pairs", "n_aligned_decision_pairs"),
        (
            "n_fault_profile_outcome_equivalent_pairs",
            "n_outcome_equivalent_pairs",
        ),
    ):
        if summary.get(summary_field) != fault_profile_divergence.get(
            profile_field
        ):
            errors.append(
                f"summary {summary_field} differs from fault-profile diagnostic"
            )
    generic_statistics = _read_json(
        report_directory / "statistics.json",
        errors,
        required=False,
    )
    _validate_statistics_routing(summary, generic_statistics, errors)
    raw_analysis_phase = summary.get("analysis_phase")
    analysis_phase: AnalysisPhase | None
    try:
        analysis_phase = parse_analysis_phase(raw_analysis_phase)
    except AnalysisPhaseError:
        analysis_phase = None
        errors.append("summary has an invalid analysis phase")
    if manifest.get("analysis_phase") != raw_analysis_phase:
        errors.append("summary analysis phase differs from report manifest")
    raw_analysis_timing = summary.get("analysis_timing")
    _analysis_timing: AnalysisTiming | None
    try:
        _analysis_timing = parse_analysis_timing(raw_analysis_timing)
    except AnalysisPhaseError:
        _analysis_timing = None
        errors.append("summary has an invalid analysis timing")
    if manifest.get("analysis_timing") != raw_analysis_timing:
        errors.append("summary analysis timing differs from report manifest")
    _read_json(
        report_directory / "heuristic_baselines.json",
        errors,
        required=False,
    )
    drift_calibration = _read_json(
        report_directory / "drift_calibration.json",
        errors,
        required=False,
    )
    if drift_calibration:
        for field in (
            "all_categories_calibrated",
            "all_represented_scenarios_calibrated",
        ):
            if not isinstance(drift_calibration.get(field), bool):
                errors.append(f"drift_calibration.json lacks boolean {field}")
    measurement_gates = _read_json(
        report_directory / "measurement_gates.json",
        errors,
        required=False,
    )
    if measurement_gates and not isinstance(measurement_gates.get("measurement_ready"), bool):
        errors.append("measurement_gates.json lacks boolean measurement_ready")
    experiment_design_audit = _read_json(
        report_directory / "experiment_design_audit.json",
        errors,
        required=False,
    )
    _validate_experiment_design_audit(experiment_design_audit, errors)
    if experiment_design_audit:
        actual_design_audit_hash = canonical_hash(experiment_design_audit)
        if manifest.get("experiment_design_audit_hash") != actual_design_audit_hash:
            errors.append(
                "experiment design audit hash differs from the report run identity"
            )
        if manifest.get("experiment_design_audit_status") != (
            experiment_design_audit.get("audit_status")
        ):
            errors.append(
                "experiment design audit status differs from the report manifest"
            )
        if manifest.get("balanced_mechanism_design_ready") is not (
            experiment_design_audit.get("balanced_mechanism_design_ready")
        ):
            errors.append(
                "balanced mechanism readiness differs from the report manifest"
            )
    if analysis_phase is not None:
        try:
            validate_analysis_phase(
                analysis_phase,
                construct_mode=summary.get("construct_mode"),
                n_statistical_units=summary.get("n_statistical_units"),
                balanced_mechanism_design_ready=experiment_design_audit.get(
                    "balanced_mechanism_design_ready"
                ),
            )
        except AnalysisPhaseError as exc:
            errors.append(f"analysis phase eligibility failed: {exc}")
    drift_trajectories = _read_json(
        report_directory / "drift_trajectories.json",
        errors,
        required=False,
    )
    _validate_drift_trajectories(drift_trajectories, errors)
    matched_statistics = _read_json(
        report_directory / "matched_construct_statistics.json",
        errors,
        required=False,
    )
    _validate_matched_construct_statistics(matched_statistics, errors)
    horizon_statistics = _read_json(
        report_directory / "horizon_panel_statistics.json",
        errors,
        required=False,
    )
    _validate_horizon_panel_statistics(horizon_statistics, errors)
    contribution_evidence = _read_json(
        report_directory / "contribution_evidence.json",
        errors,
        required=False,
    )
    _validate_contribution_evidence(
        contribution_evidence,
        summary=summary,
        measurement_gates=measurement_gates,
        matched_statistics=matched_statistics,
        drift_trajectories=drift_trajectories,
        decision_attribution_rows=jsonl["decision_attribution.jsonl"],
        fault_profile_divergence=fault_profile_divergence,
        experiment_design_audit=experiment_design_audit,
        horizon_statistics=horizon_statistics,
        errors=errors,
    )
    validation_payload = _read_json(report_directory / "validation.json", errors, required=False)
    if validation_payload and validation_payload.get("run_identity") not in {
        None,
        run_identity,
    }:
        errors.append("validation.json run identity does not match report manifest")
    return ArtifactValidationReport(
        ok=not errors,
        errors=tuple(errors),
        checked_artifacts=checked,
        run_identity=run_identity,
    )


def _validate_trace_match(
    sceu: Mapping[str, object],
    trace: Mapping[str, object],
    key: tuple[str, str, str],
    errors: list[str],
) -> None:
    candidates = _string_list(
        sceu.get("candidate_memory_ids"),
        "candidate_memory_ids",
        errors,
    )
    trace_candidates = _string_list(
        trace.get("candidate_memory_ids"),
        "trace candidate_memory_ids",
        errors,
    )
    if candidates != trace_candidates:
        errors.append(f"SCEU candidate IDs do not match retrieval trace for {key}")
    retrieved = _string_list(
        sceu.get("retrieved_memory_ids"),
        "retrieved_memory_ids",
        errors,
    )
    readout = str(sceu.get("readout", ""))
    trace_field = (
        "common_reranked_memory_ids"
        if readout == "common_rerank"
        else "native_retrieved_memory_ids"
    )
    trace_retrieved = _string_list(
        trace.get(trace_field),
        f"trace {trace_field}",
        errors,
    )
    if retrieved != trace_retrieved[: len(retrieved)]:
        errors.append(f"SCEU retrieved IDs do not match retrieval trace for {key}")


def _validate_intervention_classification(
    classification: Mapping[str, object],
    *,
    key: tuple[str, str, str],
    errors: list[str],
) -> bool:
    """Validate the observable unique-effect contract for one intervention.

    The return value means that a repeat-stable unique causal effect was
    detected. It is deliberately independent of whether the effect direction
    agrees with the evaluator-side role of the memory object.
    """

    label = str(classification.get("label", ""))
    effect_labels = {
        "beneficial",
        "harmful",
        "causal_direction_ambiguous",
    }
    no_effect_label = "visible_without_detected_unique_causal_effect"
    unstable_labels = {"unstable_baseline", "intervention_unstable"}
    allowed_labels = {*effect_labels, no_effect_label, *unstable_labels}
    if label not in allowed_labels:
        errors.append(
            f"intervention classification uses an unsupported label for {key}: {label}"
        )
        return False

    boolean_fields = (
        "behaviorally_used",
        "baseline_stable",
        "intervention_stable",
        "action_changed",
        "checker_changed",
    )
    for field in boolean_fields:
        if not isinstance(classification.get(field), bool):
            errors.append(
                f"intervention classification {field} must be boolean for {key}"
            )
    behaviorally_used = classification.get("behaviorally_used") is True
    baseline_stable = classification.get("baseline_stable") is True
    intervention_stable = classification.get("intervention_stable") is True
    observable_change = (
        classification.get("action_changed") is True
        or classification.get("checker_changed") is True
    )
    effect_detected = label in effect_labels

    if effect_detected and not (
        behaviorally_used
        and baseline_stable
        and intervention_stable
        and observable_change
    ):
        errors.append(
            "causal-effect intervention label lacks stable observable-change "
            f"evidence for {key}"
        )
    if label == no_effect_label and (
        behaviorally_used
        or not baseline_stable
        or not intervention_stable
        or observable_change
    ):
        errors.append(
            "no-detected-unique-effect label contradicts its intervention "
            f"evidence for {key}"
        )
    if label in unstable_labels and behaviorally_used:
        errors.append(
            f"unstable intervention cannot claim a causal effect for {key}"
        )
    if behaviorally_used != effect_detected:
        errors.append(
            "legacy behaviorally_used field must equal the repeat-stable unique "
            f"causal-effect indicator for {key}"
        )
    return (
        effect_detected
        and behaviorally_used
        and baseline_stable
        and intervention_stable
        and observable_change
    )


def _validate_task_spans(
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    episodes: set[str] = set()
    integer_fields = (
        "total_step_count",
        "effective_step_count",
        "visible_prefix_step_count",
        "policy_evaluated_step_count",
        "frozen_replay_step_count",
        "environment_generated_step_count",
        "policy_conditioned_future_step_count",
        "policy_steps_with_downstream_effect_count",
        "policy_dependent_decision_count",
        "long_horizon_decision_count",
        "session_handoff_count",
        "max_dependency_depth",
        "semantic_effect_step_count",
    )
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            errors.append("task-span row requires episode_id")
            continue
        if episode_id in episodes:
            errors.append(f"duplicate task-span episode: {episode_id}")
        episodes.add(episode_id)
        values: dict[str, int] = {}
        for field in integer_fields:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(
                    f"task-span {field} must be a non-negative integer for "
                    f"{episode_id}"
                )
                continue
            values[field] = value
        total = values.get("total_step_count")
        effective = values.get("effective_step_count")
        visible = values.get("visible_prefix_step_count")
        if total is not None and effective is not None and effective > total:
            errors.append(f"effective task steps exceed total for {episode_id}")
        if effective is not None and visible is not None and visible > effective:
            errors.append(f"visible task steps exceed effective for {episode_id}")
        semantic_count = values.get("semantic_effect_step_count")
        if (
            effective is not None
            and semantic_count is not None
            and semantic_count > effective
        ):
            errors.append(
                f"semantic task steps exceed effective steps for {episode_id}"
            )
        for field in (
            "minimum_decision_causal_span",
            "maximum_decision_causal_span",
        ):
            if field not in row:
                errors.append(f"task-span {field} is required for {episode_id}")
                continue
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                errors.append(
                    f"task-span {field} must be null or a non-negative integer "
                    f"for {episode_id}"
                )
        minimum_span = row.get("minimum_decision_causal_span")
        maximum_span = row.get("maximum_decision_causal_span")
        if (
            isinstance(minimum_span, int)
            and not isinstance(minimum_span, bool)
            and isinstance(maximum_span, int)
            and not isinstance(maximum_span, bool)
            and minimum_span > maximum_span
        ):
            errors.append(
                f"minimum decision causal span exceeds maximum for {episode_id}"
            )
        for field in (
            "causally_linked_step_fraction",
            "semantic_effect_coverage",
            "consumed_prefix_effect_fraction",
        ):
            if field not in row:
                errors.append(f"task-span {field} is required for {episode_id}")
                continue
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(
                    f"task-span {field} must be null or in [0, 1] for "
                    f"{episode_id}"
                )
        boolean_fields = (
            "anti_padding_verified",
            "effect_chain_verified",
            "meets_long_horizon_step_threshold",
        )
        for field in boolean_fields:
            if not isinstance(row.get(field), bool):
                errors.append(f"task-span {field} must be boolean for {episode_id}")
        anti_padding_verified = row.get("anti_padding_verified")
        effect_verified = row.get("effect_chain_verified")
        meets_threshold = row.get("meets_long_horizon_step_threshold")
        semantic_coverage = row.get("semantic_effect_coverage")
        consumed_prefix_fraction = row.get(
            "consumed_prefix_effect_fraction"
        )
        if anti_padding_verified is True and (
            semantic_coverage != 1.0 or consumed_prefix_fraction != 1.0
        ):
            errors.append(
                "anti-padding verification lacks complete semantic-effect "
                f"coverage for {episode_id}"
            )
        if effect_verified is True and anti_padding_verified is not True:
            errors.append(
                f"effect-chain verification lacks anti-padding proof for {episode_id}"
            )
        if meets_threshold is True and (
            not isinstance(maximum_span, int)
            or isinstance(maximum_span, bool)
            or maximum_span < MIN_LONG_HORIZON_EFFECTIVE_STEPS
            or anti_padding_verified is not True
            or effect_verified is not True
        ):
            errors.append(
                "long-horizon threshold lacks a 200-step terminal causal span "
                f"with anti-padding proof for {episode_id}"
            )
        policy_count = values.get("policy_evaluated_step_count")
        downstream_policy_count = values.get(
            "policy_steps_with_downstream_effect_count"
        )
        dependent_decision_count = values.get(
            "policy_dependent_decision_count"
        )
        long_horizon_decision_count = values.get(
            "long_horizon_decision_count"
        )
        if (
            policy_count is not None
            and downstream_policy_count is not None
            and downstream_policy_count > policy_count
        ):
            errors.append(
                "task-span policy steps with downstream effects exceed policy "
                f"steps for {episode_id}"
            )
        if (
            policy_count is not None
            and dependent_decision_count is not None
            and dependent_decision_count > policy_count
        ):
            errors.append(
                "task-span policy-dependent decisions exceed policy steps for "
                f"{episode_id}"
            )
        if (
            policy_count is not None
            and long_horizon_decision_count is not None
            and long_horizon_decision_count > policy_count
        ):
            errors.append(
                "task-span long-horizon decisions exceed policy decisions for "
                f"{episode_id}"
            )
        coverage = row.get("policy_dependency_coverage")
        if policy_count is not None and policy_count <= 1:
            if coverage is not None:
                errors.append(
                    "policy_dependency_coverage must be null with fewer than two "
                    f"policy steps for {episode_id}"
                )
        elif coverage is None or (
            isinstance(coverage, bool)
            or not isinstance(coverage, int | float)
            or not 0.0 <= float(coverage) <= 1.0
        ):
            errors.append(
                "policy_dependency_coverage must be in [0, 1] with multiple "
                f"policy steps for {episode_id}"
            )
        interaction_mode = row.get("interaction_mode")
        declared_closed_loop = row.get("declared_closed_loop_dependency")
        online_supported = row.get(
            "online_long_horizon_agent_execution_supported"
        )
        valid_modes = {
            "no_policy_evaluation",
            "replay_backed_critical_decision",
            "sparse_closed_loop",
            "online_long_horizon_agent_execution",
        }
        if interaction_mode not in valid_modes:
            errors.append(
                f"task-span interaction_mode is invalid for {episode_id}"
            )
        if not isinstance(declared_closed_loop, bool):
            errors.append(
                "declared_closed_loop_dependency must be boolean for "
                f"{episode_id}"
            )
        if not isinstance(online_supported, bool):
            errors.append(
                "online_long_horizon_agent_execution_supported must be boolean "
                f"for {episode_id}"
            )
        if declared_closed_loop is True and (
            downstream_policy_count in {None, 0}
            or dependent_decision_count in {None, 0}
        ):
            errors.append(
                "declared closed-loop dependency lacks a policy-conditioned "
                f"downstream decision for {episode_id}"
            )
        if online_supported is True and (
            policy_count is None
            or policy_count < 200
            or not isinstance(coverage, int | float)
            or isinstance(coverage, bool)
            or float(coverage) < 0.99
            or effect_verified is not True
        ):
            errors.append(
                "online long-horizon execution support lacks the required "
                f"policy span or dependency coverage for {episode_id}"
            )
        if interaction_mode == "no_policy_evaluation" and policy_count != 0:
            errors.append(
                f"no-policy interaction mode has policy steps for {episode_id}"
            )
        if interaction_mode == "replay_backed_critical_decision" and (
            policy_count == 0 or declared_closed_loop is not False
        ):
            errors.append(
                "replay-backed interaction mode is inconsistent for "
                f"{episode_id}"
            )
        if interaction_mode == "sparse_closed_loop" and (
            declared_closed_loop is not True or online_supported is not False
        ):
            errors.append(
                f"sparse closed-loop mode is inconsistent for {episode_id}"
            )
        if interaction_mode == "online_long_horizon_agent_execution" and (
            declared_closed_loop is not True or online_supported is not True
        ):
            errors.append(
                "online long-horizon interaction mode is inconsistent for "
                f"{episode_id}"
            )


def _validate_matched_construct_contrasts(
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    identities: set[tuple[str, str, str, str]] = set()
    for row in rows:
        identity = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("counterfactual_group_id", "")),
        )
        if not all(identity):
            errors.append("matched-construct contrast lacks a complete identity")
            continue
        if identity in identities:
            errors.append(f"duplicate matched-construct contrast: {identity}")
        identities.add(identity)
        complete = row.get("complete") is True
        terminal_archetype = str(row.get("terminal_archetype", ""))
        if complete and terminal_archetype not in {
            "current_v1_offline",
            "current_v2_offline",
            "authorized_cloud",
        }:
            errors.append(
                "complete matched-construct contrast lacks a valid terminal "
                f"archetype: {identity}"
            )
        counts = tuple(
            row.get(field)
            for field in (
                "n_static",
                "n_evolution",
                "n_hierarchical_conflict",
            )
        )
        if complete and any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in counts
        ):
            errors.append(
                "complete matched-construct contrast must contain every variant: "
                f"{identity}"
            )


def _validate_horizon_panel_contrasts(
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    identities: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        identity = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("horizon_panel_id", "")),
            str(row.get("opportunity_id", "")),
        )
        if not all(identity):
            errors.append("horizon-panel contrast lacks a complete identity")
            continue
        if identity in identities:
            errors.append(f"duplicate horizon-panel contrast: {identity}")
        identities.add(identity)
        if row.get("analysis_unit") != "horizon_panel":
            errors.append(f"horizon-panel contrast has the wrong unit: {identity}")
        if row.get("horizon_axis") != (
            "joint_effective_transition_and_session_handoff_dose"
        ):
            errors.append(f"horizon-panel contrast has an invalid axis: {identity}")
        if row.get("complete") is not True:
            continue
        for level in HORIZON_LEVELS:
            for construct in HORIZON_CONSTRUCTS:
                count = row.get(f"n_{level}_{construct}")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                ):
                    errors.append(
                        "complete horizon-panel contrast lacks a physical member: "
                        f"{identity}|{level}|{construct}"
                    )
        if row.get("workspace_matched_control_available") is True:
            for estimand in HORIZON_PRIMARY_ESTIMANDS:
                value = row.get(estimand)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not float("-inf") < float(value) < float("inf")
                ):
                    errors.append(
                        "complete horizon-panel contrast has invalid primary "
                        f"estimand: {identity}|{estimand}"
                    )


def _validate_matched_construct_statistics(
    payload: Mapping[str, object],
    errors: list[str],
) -> None:
    if not payload:
        return
    if payload.get("schema_version") != MATCHED_STATISTICS_SCHEMA_VERSION:
        errors.append("matched construct statistics has an unsupported schema version")
    if payload.get("status") == "suppressed_within_panel_triplets":
        if payload.get("analysis_unit") != "horizon_panel":
            errors.append(
                "suppressed within-panel statistics must declare horizon_panel"
            )
        if payload.get("estimates") != []:
            errors.append(
                "suppressed within-panel statistics must not contain estimates"
            )
        return
    if payload.get("analysis_unit") != "counterfactual_group":
        errors.append(
            "matched construct statistics must use counterfactual_group as the "
            "analysis unit"
        )
    expected_contract_fields: dict[str, object] = {
        "primary_analysis_unit": MATCHED_PRIMARY_ANALYSIS_UNIT,
        "primary_estimands": list(MATCHED_PRIMARY_ESTIMANDS),
        "secondary_estimands": list(MATCHED_SECONDARY_ESTIMANDS),
        "primary_workspace_adjustment": MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT,
        "primary_effect_direction": MATCHED_PRIMARY_EFFECT_DIRECTION,
        "drift_scope": MATCHED_DRIFT_SCOPE,
        "paired_test": MATCHED_PAIRED_TEST,
        "multiplicity_scope": MATCHED_MULTIPLICITY_SCOPE,
    }
    for field, expected in expected_contract_fields.items():
        if payload.get(field) != expected:
            errors.append(
                "matched construct statistics analysis contract differs for "
                f"{field}"
            )
    raw_rows = payload.get("estimates")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        errors.append("matched construct statistical estimates must be an array")
        return
    identities: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            errors.append(f"matched construct estimate {index} must be an object")
            continue
        identity = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("metric", "")),
        )
        if not all(identity):
            errors.append(
                f"matched construct estimate {index} lacks a complete identity"
            )
            continue
        if identity in identities:
            errors.append(f"duplicate matched construct estimate: {identity}")
        identities.add(identity)
        metric = identity[3]
        if metric not in MATCHED_ALL_ESTIMANDS:
            errors.append(
                f"matched construct estimate uses an undeclared estimand: {identity}"
            )
        expected_role = (
            "primary" if metric in MATCHED_PRIMARY_ESTIMANDS else "secondary"
        )
        if row.get("estimand_role") != expected_role:
            errors.append(
                f"matched construct estimate has the wrong estimand role: {identity}"
            )
        if row.get("analysis_unit") != "counterfactual_group":
            errors.append(
                f"matched construct estimate has the wrong unit: {identity}"
            )
        n_pairs = row.get("n_pairs")
        if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 1:
            errors.append(
                f"matched construct estimate has an invalid group count: {identity}"
            )
        for field in (
            "mean_difference",
            "ci_low",
            "ci_high",
            "permutation_p_value",
            "holm_adjusted_p_value",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not float("-inf") < float(value) < float("inf")
            ):
                errors.append(
                    f"matched construct estimate has invalid {field}: {identity}"
                )
        for field in ("permutation_p_value", "holm_adjusted_p_value"):
            value = row.get(field)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and not 0.0 <= float(value) <= 1.0
            ):
                errors.append(
                    f"matched construct estimate has out-of-range {field}: "
                    f"{identity}"
                )


def _validate_horizon_panel_statistics(
    payload: Mapping[str, object],
    errors: list[str],
) -> None:
    if not payload:
        return
    if payload.get("schema_version") != HORIZON_STATISTICS_SCHEMA_VERSION:
        errors.append("horizon-panel statistics has an unsupported schema version")
    expected_contract_fields: dict[str, object] = {
        "analysis_unit": HORIZON_PRIMARY_ANALYSIS_UNIT,
        "primary_analysis_unit": HORIZON_PRIMARY_ANALYSIS_UNIT,
        "primary_estimands": list(HORIZON_PRIMARY_ESTIMANDS),
        "secondary_estimands": list(HORIZON_SECONDARY_ESTIMANDS),
        "primary_workspace_adjustment": HORIZON_PRIMARY_WORKSPACE_ADJUSTMENT,
        "primary_effect_direction": HORIZON_PRIMARY_EFFECT_DIRECTION,
        "paired_test": HORIZON_PAIRED_TEST,
        "multiplicity_scope": HORIZON_MULTIPLICITY_SCOPE,
        "horizon_axis": (
            "joint_effective_transition_and_session_handoff_dose"
        ),
    }
    for field, expected in expected_contract_fields.items():
        if payload.get(field) != expected:
            errors.append(
                f"horizon-panel statistics analysis contract differs for {field}"
            )
    raw_rows = payload.get("estimates")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        errors.append("horizon-panel statistical estimates must be an array")
        return
    identities: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            errors.append(f"horizon-panel estimate {index} must be an object")
            continue
        identity = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("metric", "")),
        )
        if not all(identity):
            errors.append(f"horizon-panel estimate {index} lacks a complete identity")
            continue
        if identity in identities:
            errors.append(f"duplicate horizon-panel estimate: {identity}")
        identities.add(identity)
        metric = identity[3]
        if metric not in HORIZON_ALL_ESTIMANDS:
            errors.append(
                f"horizon-panel estimate uses an undeclared estimand: {identity}"
            )
        expected_role = (
            "primary" if metric in HORIZON_PRIMARY_ESTIMANDS else "secondary"
        )
        if row.get("estimand_role") != expected_role:
            errors.append(f"horizon-panel estimate has the wrong role: {identity}")
        if row.get("analysis_unit") != "horizon_panel":
            errors.append(f"horizon-panel estimate has the wrong unit: {identity}")
        n_panels = row.get("n_panels")
        if (
            isinstance(n_panels, bool)
            or not isinstance(n_panels, int)
            or n_panels < 1
        ):
            errors.append(f"horizon-panel estimate has an invalid n: {identity}")
        for field in (
            "mean_difference",
            "ci_low",
            "ci_high",
            "permutation_p_value",
            "holm_adjusted_p_value",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not float("-inf") < float(value) < float("inf")
            ):
                errors.append(
                    f"horizon-panel estimate has invalid {field}: {identity}"
                )
        for field in ("permutation_p_value", "holm_adjusted_p_value"):
            value = row.get(field)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and not 0.0 <= float(value) <= 1.0
            ):
                errors.append(
                    f"horizon-panel estimate has out-of-range {field}: {identity}"
                )


def _validate_contribution_evidence(
    payload: Mapping[str, object],
    *,
    summary: Mapping[str, object],
    measurement_gates: Mapping[str, object],
    matched_statistics: Mapping[str, object],
    drift_trajectories: Mapping[str, object],
    decision_attribution_rows: Sequence[Mapping[str, object]],
    fault_profile_divergence: Mapping[str, object],
    experiment_design_audit: Mapping[str, object],
    horizon_statistics: Mapping[str, object],
    errors: list[str],
) -> None:
    if not payload:
        return
    if payload.get("schema_version") != CONTRIBUTION_EVIDENCE_SCHEMA_VERSION:
        errors.append("contribution evidence has an unsupported schema version")
    raw_contributions = payload.get("contributions")
    if not isinstance(raw_contributions, Sequence) or isinstance(
        raw_contributions,
        str | bytes,
    ):
        errors.append("contribution evidence contributions must be an array")
        return
    identifiers = tuple(
        str(row.get("contribution_id", ""))
        for row in raw_contributions
        if isinstance(row, Mapping)
    )
    if identifiers != ("C1", "C2", "C3"):
        errors.append("contribution evidence must contain ordered C1, C2, and C3")
    expected = build_contribution_evidence(
        summary=summary,
        measurement_gates=measurement_gates,
        matched_statistics=matched_statistics,
        drift_trajectories=drift_trajectories,
        decision_attribution_rows=decision_attribution_rows,
        fault_profile_divergence=fault_profile_divergence,
        experiment_design_audit=experiment_design_audit,
        horizon_statistics=horizon_statistics,
    )
    if payload != expected:
        errors.append(
            "contribution evidence does not match report estimands, gates, or counts"
        )


def _validate_fault_profile_divergence(
    payload: Mapping[str, object],
    *,
    decision_attribution_rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    if not payload:
        return
    if payload.get("schema_version") != FAULT_PROFILE_DIVERGENCE_SCHEMA_VERSION:
        errors.append("fault-profile divergence has an unsupported schema version")
    try:
        expected = compute_fault_profile_divergence(decision_attribution_rows)
    except FaultProfileAlignmentError as exc:
        errors.append(f"fault-profile decision alignment failed: {exc}")
        return
    if payload != expected:
        errors.append(
            "fault-profile divergence does not match decision attribution rows"
        )


def _validate_experiment_design_audit(
    payload: Mapping[str, object],
    errors: list[str],
) -> None:
    if not payload:
        return
    if payload.get("schema_version") != EXPERIMENT_DESIGN_AUDIT_SCHEMA_VERSION:
        errors.append("experiment design audit has an unsupported schema version")
    if not isinstance(payload.get("run_ready"), bool):
        errors.append("experiment design audit lacks boolean run_ready")
    if not isinstance(payload.get("balanced_mechanism_design_ready"), bool):
        errors.append(
            "experiment design audit lacks boolean balanced_mechanism_design_ready"
        )
    horizon_scope = payload.get("scope") == "horizon_dose_diagnostic"
    matched_scope = payload.get("scope") == "matched_mechanism"
    longitudinal_scope = payload.get("scope") == "longitudinal_trajectory"
    counterfactual_scope = matched_scope or horizon_scope
    expected_analysis_unit = (
        "horizon_panel"
        if horizon_scope
        else ("counterfactual_group" if matched_scope else "episode")
    )
    if payload.get("analysis_unit") != expected_analysis_unit:
        errors.append("experiment design audit analysis unit differs from its scope")
    expected_contract = build_analysis_contract(
        matched=counterfactual_scope,
        horizon=horizon_scope,
        longitudinal=longitudinal_scope,
    )
    if payload.get("analysis_contract") != expected_contract:
        errors.append(
            "experiment design audit analysis contract differs from the frozen "
            "matched-mechanism contract"
        )
    interaction_counts = payload.get("trajectory_interaction_mode_counts")
    if not isinstance(interaction_counts, Mapping):
        errors.append(
            "experiment design audit lacks trajectory interaction mode counts"
        )
    elif counterfactual_scope:
        invalid_modes = set(interaction_counts).difference(
            {"replay_backed_critical_decision"}
        )
        if invalid_modes or not interaction_counts:
            errors.append(
                "current matched design audit must declare replay-backed "
                "critical-decision interaction"
            )
    online_execution_supported = payload.get(
        "online_long_horizon_agent_execution_supported"
    )
    if not isinstance(online_execution_supported, bool):
        errors.append(
            "experiment design audit lacks boolean online execution support"
        )
    elif counterfactual_scope and online_execution_supported:
        errors.append(
            "current matched analysis contract cannot claim online long-horizon "
            "agent execution"
        )
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, str | bytes):
        errors.append("experiment design audit checks must be an array")
        return
    check_ids: set[str] = set()
    ordered_check_ids: list[str] = []
    failed: list[str] = []
    statuses: list[str] = []
    status_by_id: dict[str, str] = {}
    for index, raw in enumerate(raw_checks):
        if not isinstance(raw, Mapping):
            errors.append(f"experiment design audit check {index} is not an object")
            continue
        check_id = str(raw.get("check_id", ""))
        status = str(raw.get("status", ""))
        if not check_id or check_id in check_ids:
            errors.append("experiment design audit has missing or duplicate check IDs")
        check_ids.add(check_id)
        ordered_check_ids.append(check_id)
        statuses.append(status)
        status_by_id[check_id] = status
        if status not in {"pass", "fail", "not_applicable"}:
            errors.append(f"experiment design audit check has invalid status: {check_id}")
        if status == "fail":
            failed.append(check_id)
    if payload.get("failed_check_ids") != failed:
        errors.append("experiment design audit failed-check index is inconsistent")
    if tuple(ordered_check_ids) != EXPERIMENT_DESIGN_CHECK_IDS:
        errors.append(
            "experiment design audit must contain the complete ordered check set"
        )
    if payload.get("run_ready") is not (not failed):
        errors.append("experiment design audit run_ready is inconsistent")
    raw_group_count = payload.get("counterfactual_group_count")
    group_count = (
        raw_group_count
        if isinstance(raw_group_count, int) and not isinstance(raw_group_count, bool)
        else 0
    )
    raw_panel_count = payload.get("horizon_panel_count")
    panel_count = (
        raw_panel_count
        if isinstance(raw_panel_count, int)
        and not isinstance(raw_panel_count, bool)
        else 0
    )
    statistical_unit_count = panel_count if horizon_scope else group_count
    balanced_expected = (
        counterfactual_scope
        and statistical_unit_count >= 3
        and bool(statuses)
        and not failed
        and status_by_id.get("long_horizon_effective_step_span") == "pass"
        and status_by_id.get("task_step_effect_chain_integrity") == "pass"
    )
    if payload.get("balanced_mechanism_design_ready") is not balanced_expected:
        errors.append(
            "experiment design audit balanced-mechanism readiness is inconsistent"
        )
    expected_status = (
        "invalid"
        if failed
        else (
            "ready_for_calibration" if balanced_expected else "diagnostic_only"
        )
    )
    if payload.get("audit_status") != expected_status:
        errors.append("experiment design audit status is inconsistent")


def _validate_summary_analysis_unit(
    summary: Mapping[str, object],
    errors: list[str],
) -> None:
    if not summary:
        return
    analysis_unit = summary.get("primary_analysis_unit")
    construct_mode = summary.get("construct_mode")
    if analysis_unit not in {"episode", "counterfactual_group", "horizon_panel"}:
        errors.append(
            "summary primary_analysis_unit must be episode, counterfactual_group, "
            "or horizon_panel"
        )
        return
    integer_fields = (
        "n_evaluated_episodes",
        "n_frozen_dataset_episodes",
        "n_physical_episodes",
        "n_frozen_physical_episodes",
        "n_counterfactual_groups",
        "n_frozen_counterfactual_groups",
        "n_horizon_panels",
        "n_frozen_horizon_panels",
        "n_statistical_units",
        "n_fault_profile_aligned_pairs",
        "n_fault_profile_outcome_equivalent_pairs",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"summary {field} must be a non-negative integer")
            continue
        values[field] = value
    if values.get("n_physical_episodes") != values.get("n_evaluated_episodes"):
        errors.append("summary physical/evaluated episode counts disagree")
    if values.get("n_frozen_physical_episodes") != values.get(
        "n_frozen_dataset_episodes"
    ):
        errors.append("summary frozen physical/dataset episode counts disagree")
    group_ids = summary.get("counterfactual_group_ids")
    if not isinstance(group_ids, Sequence) or isinstance(group_ids, (str, bytes)):
        errors.append("summary counterfactual_group_ids must be an array")
        group_count = -1
    else:
        normalized = [str(item) for item in group_ids]
        if any(not item for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            errors.append(
                "summary counterfactual_group_ids must be non-empty and unique"
            )
        group_count = len(normalized)
    panel_ids = summary.get("horizon_panel_ids")
    if not isinstance(panel_ids, Sequence) or isinstance(panel_ids, (str, bytes)):
        errors.append("summary horizon_panel_ids must be an array")
        panel_count = -1
    else:
        normalized_panels = [str(item) for item in panel_ids]
        if any(not item for item in normalized_panels) or len(
            normalized_panels
        ) != len(set(normalized_panels)):
            errors.append(
                "summary horizon_panel_ids must be non-empty and unique"
            )
        panel_count = len(normalized_panels)
    if analysis_unit == "counterfactual_group":
        if construct_mode != "matched_triplets":
            errors.append(
                "counterfactual-group analysis requires matched_triplets construct_mode"
            )
        if group_count != values.get("n_counterfactual_groups"):
            errors.append("summary counterfactual group IDs/count disagree")
        if values.get("n_statistical_units") != values.get(
            "n_counterfactual_groups"
        ):
            errors.append(
                "matched summary must count counterfactual groups as statistical units"
            )
        if values.get("n_horizon_panels") != 0 or panel_count not in {0, -1}:
            errors.append("matched-triplet analysis must not declare horizon panels")
    elif analysis_unit == "horizon_panel":
        if construct_mode != "horizon_panels":
            errors.append(
                "horizon-panel analysis requires horizon_panels construct_mode"
            )
        if panel_count != values.get("n_horizon_panels"):
            errors.append("summary horizon panel IDs/count disagree")
        if group_count != values.get("n_counterfactual_groups"):
            errors.append("horizon summary counterfactual group IDs/count disagree")
        if values.get("n_statistical_units") != values.get("n_horizon_panels"):
            errors.append(
                "horizon summary must count complete panels as statistical units"
            )
        if values.get("n_physical_episodes") != 9 * values.get(
            "n_horizon_panels", -1
        ):
            errors.append("horizon summary must contain nine members per panel")
        if values.get("n_counterfactual_groups") != 3 * values.get(
            "n_horizon_panels", -1
        ):
            errors.append(
                "horizon summary must contain three construct triplets per panel"
            )
    else:
        if (
            values.get("n_counterfactual_groups") != 0
            or group_count not in {0, -1}
        ):
            errors.append("episode analysis must not declare counterfactual groups")
        if values.get("n_horizon_panels") != 0 or panel_count not in {0, -1}:
            errors.append("episode analysis must not declare horizon panels")
        if values.get("n_statistical_units") != values.get("n_physical_episodes"):
            errors.append(
                "episode summary must count physical episodes as statistical units"
            )


def _validate_statistics_routing(
    summary: Mapping[str, object],
    statistics: Mapping[str, object],
    errors: list[str],
) -> None:
    if not summary or not statistics:
        return
    analysis_unit = summary.get("primary_analysis_unit")
    suppressed = (
        statistics.get("status") == "suppressed_dependent_physical_members"
    )
    if analysis_unit == "episode":
        if suppressed or statistics.get("analysis_unit") != "episode":
            errors.append(
                "episode release must route generic inference to episode statistics"
            )
        return
    if analysis_unit in {"counterfactual_group", "horizon_panel"} and (
        not suppressed or statistics.get("analysis_unit") != analysis_unit
    ):
        errors.append(
            "counterfactual release must suppress generic physical-episode "
            "inference and route to its declared analysis unit"
        )


def _validate_decision_attributions(
    rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    allowed_stages = {
        "no_memory_channel",
        "not_memory_reliant",
        "storage_evidence_unavailable",
        "storage_failure",
        "retrieval_failure",
        "exposure_failure",
        "utilization_failure",
        "behavior_success_causal",
        "behavior_success_without_detected_unique_causal_effect",
        # Legacy completed-report label.
        "behavior_success_without_detected_use",
        "behavior_success_unprobed",
    }
    allowed_evidence = {
        "native/exact",
        "inferred",
        "mixed",
        "unavailable",
        "not_applicable",
    }
    identities: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        identity = (
            str(row.get("episode_id", "")),
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("sceu_id", "")),
            str(row.get("result_id", "")),
        )
        if not all(identity):
            errors.append("decision attribution row lacks a complete identity")
            continue
        if identity in identities:
            errors.append(f"duplicate decision attribution identity: {identity}")
        identities.add(identity)
        label = "|".join(identity)
        stage = str(row.get("stage", ""))
        if stage not in allowed_stages:
            errors.append(f"unknown decision attribution stage for {label}: {stage}")
        evidence = str(row.get("storage_evidence_mode", ""))
        if evidence not in allowed_evidence:
            errors.append(
                f"unknown decision storage evidence mode for {label}: {evidence}"
            )
        required = set(
            _string_list(row.get("required_state_ids"), f"{label} required", errors)
        )
        stored = set(
            _string_list(
                row.get("stored_required_state_ids"),
                f"{label} stored",
                errors,
            )
        )
        retrieved = set(
            _string_list(
                row.get("retrieved_stored_state_ids"),
                f"{label} retrieved",
                errors,
            )
        )
        visible = set(
            _string_list(
                row.get("visible_retrieved_state_ids"),
                f"{label} visible",
                errors,
            )
        )
        probed = set(
            _string_list(
                row.get("probed_visible_state_ids"),
                f"{label} probed",
                errors,
            )
        )
        used = set(
            _string_list(
                row.get("causally_used_probed_state_ids"),
                f"{label} causally used",
                errors,
            )
        )
        for child, parent, child_name, parent_name in (
            (stored, required, "stored", "required"),
            (retrieved, stored, "retrieved", "stored"),
            (visible, retrieved, "visible", "retrieved"),
            (probed, visible, "probed", "visible"),
            (used, probed, "causally used", "probed"),
        ):
            if not child.issubset(parent):
                errors.append(
                    f"decision attribution {child_name} states are not a subset "
                    f"of {parent_name} states for {label}"
                )
        behavior_correct = row.get("behavior_correct") is True
        no_unique_effect_success_stages = {
            "behavior_success_without_detected_unique_causal_effect",
            "behavior_success_without_detected_use",
        }
        inconsistent = (
            (
                evidence == "unavailable"
                and stage
                not in {
                    "no_memory_channel",
                    "not_memory_reliant",
                    "storage_evidence_unavailable",
                }
            )
            or
            (
                stage == "storage_evidence_unavailable"
                and evidence != "unavailable"
            )
            or (
                stage == "storage_failure"
                and (evidence == "unavailable" or stored == required)
            )
            or (
                stage == "retrieval_failure"
                and (stored != required or retrieved == stored)
            )
            or (
                stage == "exposure_failure"
                and (stored != required or retrieved != stored or visible == retrieved)
            )
            or (
                stage == "utilization_failure"
                and (
                    stored != required
                    or retrieved != stored
                    or visible != retrieved
                    or behavior_correct
                )
            )
            or (stage.startswith("behavior_success") and not behavior_correct)
            or (stage == "behavior_success_causal" and not used)
            or (
                stage in no_unique_effect_success_stages
                and (not probed or used)
            )
            or (stage == "behavior_success_unprobed" and probed)
        )
        if inconsistent:
            errors.append(
                f"decision attribution stage is inconsistent with its funnel for {label}"
            )
        claim_boundary = row.get("causal_use_claim_boundary")
        if claim_boundary is not None:
            expected_claim_boundary = (
                "repeat_stable_unique_causal_effect_detected"
                if used
                else (
                    "no_unique_causal_effect_detected_redundant_or_compensated_"
                    "use_not_excluded"
                    if visible and probed == visible
                    else "causal_use_not_fully_identified"
                    if visible
                    else "not_applicable"
                )
            )
            if claim_boundary != expected_claim_boundary:
                errors.append(
                    "decision attribution overstates its causal-use evidence for "
                    f"{label}"
                )


def _validate_drift_trajectories(
    payload: Mapping[str, object],
    errors: list[str],
) -> None:
    if not payload:
        return
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        errors.append("drift trajectories have an invalid schema version")
        return
    if payload.get("analysis_unit") != "episode":
        errors.append("drift trajectories must use episode as the analysis unit")
    state_lineage_schema = schema_version >= 4
    if state_lineage_schema and payload.get("trajectory_unit") != (
        "state_lineage_within_episode"
    ):
        errors.append(
            "schema-v4 drift trajectories must use state_lineage_within_episode"
        )
    raw_rows = payload.get("trajectories")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        errors.append("drift trajectory rows must be an array")
        return
    identities: set[tuple[str, ...]] = set()
    rows_by_summary_key: dict[
        tuple[str, str, str, str], list[Mapping[str, object]]
    ] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            errors.append(f"drift trajectory row {index} must be an object")
            continue
        base_identity = (
            str(row.get("episode_id", "")),
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("drift_category", "")),
        )
        lineage = str(row.get("state_lineage_id", ""))
        identity = (*base_identity, lineage) if state_lineage_schema else base_identity
        if not all(identity):
            errors.append(f"drift trajectory row {index} lacks a complete identity")
            continue
        if identity in identities:
            errors.append(f"duplicate drift trajectory identity: {identity}")
        identities.add(identity)
        rows_by_summary_key.setdefault(base_identity[1:], []).append(row)
        if state_lineage_schema:
            evidence_mode = str(row.get("lineage_evidence_mode", ""))
            lineage_backed = row.get("lineage_backed")
            if not evidence_mode:
                errors.append(
                    f"drift trajectory lacks a lineage evidence mode: {identity}"
                )
            if not isinstance(lineage_backed, bool):
                errors.append(
                    f"drift trajectory lacks a boolean lineage_backed: {identity}"
                )
            if lineage == "__category_only__" and (
                lineage_backed is not False
                or evidence_mode != "category_only_legacy"
            ):
                errors.append(
                    "category-only drift trajectory is mislabeled as lineage-backed: "
                    f"{identity}"
                )
            if lineage != "__category_only__" and (
                lineage_backed is not True
                or evidence_mode in {"", "unavailable", "category_only_legacy"}
            ):
                errors.append(
                    "state-lineage drift trajectory lacks positive lineage evidence: "
                    f"{identity}"
                )
        first = row.get("first_drift_session")
        first_violation = row.get("first_violation_session")
        first_adherence = row.get("first_adherence_session")
        entry = row.get("entry_session")
        drift_entry = row.get("drift_entry_session")
        censor = row.get("censor_session")
        observed = row.get("event_observed")
        violation_observed = row.get("violation_event_observed")
        drift_evaluable = row.get("drift_evaluable")
        if not isinstance(entry, int):
            errors.append(f"drift trajectory entry session must be an integer: {identity}")
        if not isinstance(censor, int):
            errors.append(f"drift trajectory censor session must be an integer: {identity}")
        if observed is True and not isinstance(first, int):
            errors.append(f"observed drift trajectory lacks first session: {identity}")
        if observed is False and first is not None:
            errors.append(f"censored drift trajectory has a first session: {identity}")
        if violation_observed is True and not isinstance(first_violation, int):
            errors.append(f"observed drift-compatible violation lacks first session: {identity}")
        if violation_observed is False and first_violation is not None:
            errors.append(f"violation-free trajectory has a first violation session: {identity}")
        if drift_evaluable is True and not isinstance(drift_entry, int):
            errors.append(f"drift-evaluable trajectory lacks adherence entry: {identity}")
        if drift_evaluable is False and drift_entry is not None:
            errors.append(f"non-evaluable drift trajectory has an entry: {identity}")
        if observed is True and drift_evaluable is not True:
            errors.append(f"observed drift lacks an evaluable adherence history: {identity}")
        if (
            isinstance(first_adherence, int)
            and isinstance(drift_entry, int)
            and first_adherence != drift_entry
        ):
            errors.append(f"drift entry does not match first adherence: {identity}")
        if isinstance(first, int) and isinstance(censor, int) and first > censor:
            errors.append(f"drift occurs after censoring for trajectory: {identity}")
        if isinstance(entry, int) and isinstance(censor, int) and entry > censor:
            errors.append(f"drift trajectory enters after censoring: {identity}")
        if isinstance(entry, int) and isinstance(first, int) and entry > first:
            errors.append(f"drift occurs before trajectory entry: {identity}")
        if isinstance(drift_entry, int) and isinstance(first, int) and drift_entry >= first:
            errors.append(f"drift onset lacks an earlier adherence checkpoint: {identity}")
        if (
            isinstance(first_violation, int)
            and isinstance(censor, int)
            and first_violation > censor
        ):
            errors.append(f"violation occurs after censoring for trajectory: {identity}")
        numerator = row.get("persistence_numerator")
        denominator = row.get("persistence_denominator")
        if (
            not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or numerator < 0
            or denominator < numerator
        ):
            errors.append(f"invalid drift persistence counts for trajectory: {identity}")
        violation_numerator = row.get("violation_persistence_numerator")
        violation_denominator = row.get("violation_persistence_denominator")
        if (
            not isinstance(violation_numerator, int)
            or not isinstance(violation_denominator, int)
            or violation_numerator < 0
            or violation_denominator < violation_numerator
        ):
            errors.append(f"invalid violation persistence counts for trajectory: {identity}")

    if not state_lineage_schema:
        return
    raw_summary = payload.get("summary")
    if not isinstance(raw_summary, Sequence) or isinstance(
        raw_summary,
        (str, bytes),
    ):
        errors.append("drift trajectory summary must be an array")
        return
    seen_summary_keys: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(raw_summary):
        if not isinstance(row, Mapping):
            errors.append(f"drift trajectory summary row {index} must be an object")
            continue
        key = (
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            str(row.get("drift_category", "")),
        )
        if not all(key):
            errors.append(f"drift trajectory summary row {index} lacks an identity")
            continue
        if key in seen_summary_keys:
            errors.append(f"duplicate drift trajectory summary identity: {key}")
        seen_summary_keys.add(key)
        source_rows = rows_by_summary_key.get(key, [])
        expected_episodes = len(
            {str(source.get("episode_id", "")) for source in source_rows}
        )
        if row.get("n_episodes") != expected_episodes:
            errors.append(
                "drift trajectory summary does not use unique episodes: "
                f"{key}"
            )
        if row.get("n_state_lineage_trajectories") != len(source_rows):
            errors.append(
                "drift trajectory summary has an inconsistent lineage count: "
                f"{key}"
            )


def _validate_metrics_by_cell(
    payload: Mapping[str, object],
    task_results: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, Sequence) or isinstance(
        raw_groups,
        (str, bytes),
    ):
        errors.append("metrics_by_cell groups must be an array")
        return
    expected: set[tuple[str, str, str]] = set()
    for task in task_results:
        policy_profile_id = str(task.get("policy_profile_id", ""))
        conditions = task.get("condition_results")
        if not isinstance(conditions, Sequence) or isinstance(
            conditions,
            (str, bytes),
        ):
            continue
        for condition in conditions:
            if not isinstance(condition, Mapping):
                continue
            expected.add(
                (
                    policy_profile_id,
                    str(condition.get("condition", "")),
                    str(condition.get("readout", "")),
                )
            )
    actual: set[tuple[str, str, str]] = set()
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping):
            errors.append(f"metrics_by_cell group {index} must be an object")
            continue
        key = (
            str(group.get("policy_profile_id", "")),
            str(group.get("condition", "")),
            str(group.get("readout", "")),
        )
        if not all(key):
            errors.append(f"metrics_by_cell group {index} lacks a complete key")
            continue
        if key in actual:
            errors.append(f"duplicate metrics_by_cell group: {key}")
        actual.add(key)
        if not isinstance(group.get("metrics"), Mapping):
            errors.append(f"metrics_by_cell group {key} lacks metrics")
    if actual != expected:
        errors.append(
            "metrics_by_cell coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _validate_semantic_attributions(
    inventory_rows: Sequence[Mapping[str, object]],
    errors: list[str],
) -> None:
    """Keep lifecycle provenance distinct from evaluator state attribution."""
    allowed_methods = {
        "exact_signature",
        "multi_signature",
        "lexical_signature",
        "unique_provenance",
        "no_match",
        "ambiguous",
    }
    allowed_lifecycle = {"native/exact", "inferred", "unavailable"}
    for row in inventory_rows:
        raw = row.get("evaluator_attribution_by_memory")
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            errors.append("inventory evaluator attribution must be an object")
            continue
        task_id = str(row.get("task_id", ""))
        checkpoint = row.get("checkpoint_session", "")
        for memory_id, value in raw.items():
            label = f"{task_id}:{checkpoint}:{memory_id}"
            if not isinstance(value, Mapping):
                errors.append(f"semantic attribution must be an object for {label}")
                continue
            method = value.get("method")
            lifecycle = value.get("provenance_mode")
            contributes = value.get("contributes_positive_coverage")
            if method not in allowed_methods:
                errors.append(f"unknown semantic attribution method for {label}")
            if lifecycle not in allowed_lifecycle:
                errors.append(f"unknown lifecycle provenance mode for {label}")
            if not isinstance(contributes, bool):
                errors.append(f"semantic attribution coverage flag is missing for {label}")
            if method == "ambiguous" and contributes is True:
                errors.append(
                    f"ambiguous semantic attribution contributes positive coverage for {label}"
                )
            if method == "no_match" and contributes is True:
                errors.append(
                    f"no-match semantic attribution contributes positive coverage for {label}"
                )


def _unique_ids(
    rows: Sequence[Mapping[str, object]],
    field: str,
    label: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> set[str]:
    output: set[str] = set()
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            if not allow_empty:
                errors.append(f"{label} row lacks {field}")
            continue
        if value in output:
            errors.append(f"duplicate {label} {field}: {value}")
        output.add(value)
    return output


def _string_list(
    value: object,
    label: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"{label} must be an array")
        return []
    output = [str(item) for item in value]
    if len(output) != len(set(output)):
        errors.append(f"{label} contains duplicate memory IDs")
    return output


def _read_json(
    path: Path,
    errors: list[str],
    *,
    required: bool,
) -> dict[str, object]:
    if not path.is_file():
        if required:
            errors.append(f"missing JSON artifact: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON artifact {path.name}: {exc}")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"JSON artifact must be an object: {path.name}")
        return {}
    return {str(key): child for key, child in value.items()}


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read JSONL artifact {path.name}: {exc}")
        return []
    output: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL {path.name}:{number}: {exc}")
            continue
        if not isinstance(value, Mapping):
            errors.append(f"JSONL row must be an object: {path.name}:{number}")
            continue
        output.append({str(key): child for key, child in value.items()})
    return output


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ArtifactValidationReport",
    "validate_qualification_artifacts",
]
