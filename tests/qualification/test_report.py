from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from lhmsb.families.software.horizon_panel import SoftwareHorizonPanelFamily
from lhmsb.families.software.matched_constructs import (
    SoftwareMatchedConstructFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.prefix import CommonRerankTrace
from lhmsb.qualification.report import (
    REQUIRED_REPORT_ARTIFACTS,
    _append_prefix_reranker_usage,
    _evaluation_trace_id,
    _memory_count_scorecard_rows,
    _semantic_attribution_diagnostics,
    _storage_provenance_diagnostics,
    _storage_scorecard_rows,
    _summary,
    write_qualification_report,
)
from lhmsb.qualification.runner import (
    ConditionRunResult,
    QualificationMatrixResult,
    QualificationTaskResult,
    SCEURunResult,
)
from lhmsb.qualification.tei import RerankResult
from lhmsb.qualification.validate import (
    _validate_intervention_classification,
    _validate_matched_construct_statistics,
    _validate_semantic_attributions,
    validate_qualification_artifacts,
)


def test_intervention_validation_separates_effect_detection_from_direction() -> None:
    key = ("task", "result", "sceu")
    errors: list[str] = []
    detected = _validate_intervention_classification(
        {
            "label": "causal_direction_ambiguous",
            "behaviorally_used": True,
            "baseline_stable": True,
            "intervention_stable": True,
            "action_changed": True,
            "checker_changed": False,
        },
        key=key,
        errors=errors,
    )

    assert detected is True
    assert errors == []

    inconsistent_errors: list[str] = []
    inconsistent = _validate_intervention_classification(
        {
            "label": "causal_direction_ambiguous",
            "behaviorally_used": False,
            "baseline_stable": True,
            "intervention_stable": True,
            "action_changed": True,
            "checker_changed": False,
        },
        key=key,
        errors=inconsistent_errors,
    )

    assert inconsistent is False
    assert any("stable observable-change" in error for error in inconsistent_errors)
    assert any("unique causal-effect indicator" in error for error in inconsistent_errors)


def test_intervention_validation_does_not_call_no_effect_unused() -> None:
    errors: list[str] = []
    detected = _validate_intervention_classification(
        {
            "label": "visible_without_detected_unique_causal_effect",
            "behaviorally_used": False,
            "baseline_stable": True,
            "intervention_stable": True,
            "action_changed": False,
            "checker_changed": False,
        },
        key=("task", "result", "sceu"),
        errors=errors,
    )

    assert detected is False
    assert errors == []


def _matrix() -> tuple[QualificationMatrixResult, dict[str, object]]:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    sceu = spec.plan.sceu_units[0]
    behavior = ContinuationOutcome(
        action_id="safe_v2_offline",
        behavior_score=1.0,
        is_correct=True,
    )
    row = SCEURunResult(
        result_id="result-workspace",
        sceu_id=sceu.sceu_id,
        opportunity_id=sceu.opportunity_id,
        checkpoint_session=sceu.checkpoint_session,
        matched_group=sceu.matched_group,
        control_kind="workspace",
        workspace_hash="workspace-hash",
        candidate_memory_ids=(),
        retrieved_memory_ids=(),
        model_visible_memory_ids=(),
        selected_option_id="option-03",
        selected_action_id="safe_v2_offline",
        behavior=behavior,
        normalized_drift_flags=(),
        baseline_stable=True,
        baseline_evaluations=(),
        interventions=(),
        retrieval_trace_id=None,
    )
    condition = ConditionRunResult(
        result_id="result-workspace",
        condition="workspace_only",
        readout="none",
        status="complete",
        sceu_results=(row,),
    )
    task = QualificationTaskResult(
        task_id="task-001",
        episode_id=spec.plan.episode_id,
        policy_profile_id="policy-a",
        condition="workspace_only",
        status="complete",
        condition_results=(condition,),
        writes=(),
        alignments=(),
        retrieval_traces=(),
        qdrant_store_bytes=4096,
        history_store_bytes=1024,
    )
    return (
        QualificationMatrixResult(
            run_identity="run-identity",
            task_results=(task,),
        ),
        {spec.plan.episode_id: spec},
    )


def test_report_emits_required_deterministic_hashed_artifacts(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    first = write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        tmp_path / "first",
        run_metadata={"code_commit": "abc123"},
    )
    second = write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        tmp_path / "second",
        run_metadata={"code_commit": "abc123"},
    )
    assert set(REQUIRED_REPORT_ARTIFACTS) <= {
        path.name for path in (tmp_path / "first").iterdir()
    }
    assert first.artifact_hashes == second.artifact_hashes
    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(
        (tmp_path / "first" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_identity"] == "run-identity"
    assert manifest["code_commit"] == "abc123"
    assert manifest["analysis_phase"] == "development"
    assert manifest["analysis_timing"] == "pre_specified"
    assert manifest["policy_trace_schema_version"] == 1
    assert manifest["artifact_hashes"] == dict(first.artifact_hashes)
    limitations = (tmp_path / "first" / "limitations.md").read_text(
        encoding="utf-8"
    )
    assert "Generated trajectory/schedule variants are not independent" in limitations
    assert "critical continuation decisions" in limitations
    assert "does not claim that the tested policy" in limitations
    assert "mutually dependent steps online" in limitations
    assert "Artifact validation and scientific measurement readiness" in limitations
    summary = json.loads(
        (tmp_path / "first" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_evaluated_episodes"] == 1
    assert summary["n_frozen_dataset_episodes"] == 1
    assert summary["construct_mode"] == "mixed"
    assert summary["primary_analysis_unit"] == "episode"
    assert summary["n_physical_episodes"] == 1
    assert summary["n_statistical_units"] == 1
    assert summary["n_counterfactual_groups"] == 0
    assert summary["n_fault_profile_aligned_pairs"] == 0
    assert summary["n_fault_profile_outcome_equivalent_pairs"] == 0
    assert summary["analysis_phase"] == "development"
    assert summary["analysis_timing"] == "pre_specified"
    assert summary["trajectory_interaction_mode_counts"] == {
        "no_policy_evaluation": 1,
    }
    assert summary["n_online_long_horizon_agent_execution_profiles"] == 0
    assert summary["evaluated_episode_ids"] == [next(iter(specs))]
    metrics = json.loads(
        (tmp_path / "first" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["write_coverage"]["value"] is None
    assert metrics["mean_behavior_score"]["value"] == 1.0
    assert metrics["qdrant_store_bytes"]["value"] == 4096
    assert metrics["history_store_bytes"]["value"] == 1024
    metrics_by_cell = json.loads(
        (tmp_path / "first" / "metrics_by_cell.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics_by_cell["schema_version"] == 7
    assert metrics_by_cell["groups"] == [
        {
            "condition": "workspace_only",
            "metrics": metrics,
            "policy_profile_id": "policy-a",
            "readout": "none",
        }
    ]
    scorecard = (tmp_path / "first" / "scorecard.csv").read_text(
        encoding="utf-8"
    )
    assert "policy_profile_id,condition,readout" in scorecard
    assert "policy-a,workspace_only,none" in scorecard
    storage_scorecard = (tmp_path / "first" / "storage_scorecard.csv").read_text(
        encoding="utf-8"
    )
    assert "policy_profile_id,condition,provenance_track" in storage_scorecard
    memory_count_scorecard = (
        tmp_path / "first" / "memory_count_scorecard.csv"
    ).read_text(encoding="utf-8")
    assert "policy_profile_id,condition,readout,opportunity_id,count_delta" in (
        memory_count_scorecard
    )
    failure_attribution = (
        tmp_path / "first" / "failure_attribution_scorecard.csv"
    ).read_text(encoding="utf-8")
    assert "memory_required_storage_recall" in failure_attribution
    long_horizon_scorecard = (
        tmp_path / "first" / "long_horizon_scorecard.csv"
    ).read_text(encoding="utf-8")
    assert "construct_kind,horizon_band" in long_horizon_scorecard
    control_contrasts = (
        tmp_path / "first" / "long_horizon_control_contrasts.csv"
    ).read_text(encoding="utf-8")
    assert "mean_behavior_gain_beyond_workspace" in control_contrasts
    task_span_rows = [
        json.loads(line)
        for line in (tmp_path / "first" / "task_span.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert task_span_rows[0]["interaction_mode"] == "no_policy_evaluation"
    assert task_span_rows[0][
        "online_long_horizon_agent_execution_supported"
    ] is False
    assert (
        tmp_path / "first" / "matched_construct_contrasts.jsonl"
    ).is_file()
    matched_scorecard = (
        tmp_path / "first" / "matched_construct_scorecard.csv"
    ).read_text(encoding="utf-8")
    assert "mean_state_evolution_penalty_vs_static" in matched_scorecard
    matched_statistics = json.loads(
        (tmp_path / "first" / "matched_construct_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert matched_statistics["analysis_unit"] == "counterfactual_group"
    assert matched_statistics["primary_estimands"] == [
        "state_evolution_penalty_excess_over_workspace",
        "hierarchical_conflict_penalty_excess_over_workspace",
    ]
    assert matched_statistics["drift_scope"] == "endpoint_violation_only"
    assert (tmp_path / "first" / "matched_construct_statistics.md").is_file()
    construct_rows = [
        json.loads(line)
        for line in (
            tmp_path / "first" / "long_horizon_constructs.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(construct_rows) == len(next(iter(specs.values())).plan.sceu_units)
    assert all(
        not set(row["current_required_state_ids"]).intersection(
            row["future_referenced_state_ids"]
        )
        for row in construct_rows
    )
    decision_rows = [
        json.loads(line)
        for line in (
            tmp_path / "first" / "decision_attribution.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(decision_rows) == 1
    assert decision_rows[0]["stage"] == "no_memory_channel"
    assert decision_rows[0]["storage_evidence_mode"] == "not_applicable"
    fault_profile = json.loads(
        (tmp_path / "first" / "fault_profile_divergence.json").read_text(
            encoding="utf-8"
        )
    )
    assert fault_profile["n_aligned_decision_pairs"] == 0
    assert (tmp_path / "first" / "fault_profile_divergence.md").is_file()
    drift_trajectories = json.loads(
        (tmp_path / "first" / "drift_trajectories.json").read_text(
            encoding="utf-8"
        )
    )
    assert drift_trajectories["analysis_unit"] == "episode"
    design_audit = json.loads(
        (tmp_path / "first" / "experiment_design_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert design_audit["run_ready"] is True
    assert design_audit["audit_status"] == "diagnostic_only"
    assert design_audit["analysis_contract"]["status"] == "pre_call_frozen"
    assert "policy-free" in (
        tmp_path / "first" / "experiment_design_audit.md"
    ).read_text(encoding="utf-8")
    assert manifest["experiment_design_audit_hash"] == canonical_hash(
        design_audit
    )
    contribution_evidence = json.loads(
        (tmp_path / "first" / "contribution_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert contribution_evidence["benchmark_object"] == (
        "memory_supported_delayed_task_state_control_under_competing_"
        "persistent_channels"
    )
    assert contribution_evidence["analysis_phase"] == "development"
    assert contribution_evidence["analysis_timing"] == "pre_specified"
    assert contribution_evidence["confirmatory_timing_eligible"] is True
    assert [
        row["contribution_id"]
        for row in contribution_evidence["contributions"]
    ] == ["C1", "C2", "C3"]
    assert "does not mean" in (
        tmp_path / "first" / "contribution_evidence.md"
    ).read_text(encoding="utf-8")
    assert validate_qualification_artifacts(tmp_path / "first").ok is True
    episode_index = json.loads(
        (tmp_path / "first" / "episodes" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert episode_index["episode_count"] == 1
    episode_directory = tmp_path / "first" / episode_index["episodes"][0]["directory"]
    assert (episode_directory / "metrics.json").is_file()
    assert (episode_directory / "metrics_by_cell.json").is_file()
    assert (episode_directory / "scorecard.csv").is_file()
    assert (episode_directory / "decision_attribution.jsonl").is_file()
    assert (episode_directory / "fault_profile_divergence.json").is_file()
    assert (episode_directory / "drift_trajectories.json").is_file()
    assert (episode_directory / "long_horizon_control_contrasts.csv").is_file()
    assert (episode_directory / "summary.json").is_file()
    assert "episodes/index.json" in manifest["artifact_hashes"]


def test_report_propagates_analysis_phase_and_timing_to_claim_artifacts(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "phase-report"

    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
        run_metadata={
            "analysis_phase": "diagnostic",
            "analysis_timing": "post_hoc_scope_audit",
        },
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (out / "run_manifest.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (out / "contribution_evidence.json").read_text(encoding="utf-8")
    )
    assert summary["analysis_phase"] == "diagnostic"
    assert summary["analysis_timing"] == "post_hoc_scope_audit"
    assert manifest["analysis_phase"] == "diagnostic"
    assert manifest["analysis_timing"] == "post_hoc_scope_audit"
    assert evidence["analysis_phase"] == "diagnostic"
    assert evidence["analysis_timing"] == "post_hoc_scope_audit"
    assert evidence["confirmatory_timing_eligible"] is False
    assert "cannot be promoted" in evidence["interpretation"]
    assert validate_qualification_artifacts(out).ok is True


def test_report_rejects_unknown_analysis_phase(tmp_path: Path) -> None:
    matrix, specs = _matrix()

    with pytest.raises(ValueError, match="unknown analysis phase"):
        write_qualification_report(
            matrix,
            specs,  # type: ignore[arg-type]
            tmp_path / "bad-phase-report",
            run_metadata={"analysis_phase": "publication"},
        )

    with pytest.raises(ValueError, match="unknown analysis timing"):
        write_qualification_report(
            matrix,
            specs,  # type: ignore[arg-type]
            tmp_path / "bad-timing-report",
            run_metadata={"analysis_timing": "backdated"},
        )


def test_validator_rejects_undersized_calibration_report(tmp_path: Path) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "undersized-calibration"
    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
        run_metadata={"analysis_phase": "calibration"},
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "analysis phase calibration requires at least 5 statistical units"
        in error
        for error in validation.errors
    )


def test_validator_rejects_analysis_phase_tampering(tmp_path: Path) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-analysis-phase"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis_phase"] = "confirmatory"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["summary.json"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "summary analysis phase differs from report manifest" in error
        for error in validation.errors
    )


def test_report_rejects_missing_planned_evaluation_results(tmp_path: Path) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "partial"

    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
        run_metadata={
            "code_commit": "abc123",
            "evaluation_task_count": 2,
        },
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((out / "measurement_gates.json").read_text(encoding="utf-8"))
    completion = {item["gate_id"]: item for item in gates["gates"]}["task_completion"]
    validation = validate_qualification_artifacts(out)

    assert summary["n_planned_tasks"] == 2
    assert summary["n_observed_task_results"] == 1
    assert summary["n_missing_task_results"] == 1
    assert completion["status"] == "fail"
    assert completion["detail"]["denominator"] == 2
    assert validation.ok is False
    assert any("evaluation_task_count" in error for error in validation.errors)


def test_validator_recomputes_contribution_evidence_from_source_artifacts(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-contribution"
    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
    )
    evidence_path = out / "contribution_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["contributions"][0]["evidence_status"] = "ready"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["contribution_evidence.json"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "contribution evidence does not match" in error
        for error in validation.errors
    )


def test_validator_recomputes_fault_profile_from_decision_rows(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-fault-profile"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    profile_path = out / "fault_profile_divergence.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["n_aligned_decision_pairs"] = 99
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["fault_profile_divergence.json"] = (
        hashlib.sha256(profile_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "fault-profile divergence does not match" in error
        for error in validation.errors
    )


def test_validator_rejects_online_rollout_claim_without_policy_span(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-interaction-tier"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    span_path = out / "task_span.jsonl"
    rows = [
        json.loads(line)
        for line in span_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["interaction_mode"] = "online_long_horizon_agent_execution"
    rows[0]["declared_closed_loop_dependency"] = True
    rows[0]["online_long_horizon_agent_execution_supported"] = True
    span_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["task_span.jsonl"] = hashlib.sha256(
        span_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "online long-horizon execution support lacks" in error
        for error in validation.errors
    )


def test_validator_rejects_long_horizon_claim_without_anti_padding_proof(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-anti-padding"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    span_path = out / "task_span.jsonl"
    rows = [
        json.loads(line)
        for line in span_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["maximum_decision_causal_span"] = 250
    rows[0]["semantic_effect_coverage"] = 0.5
    rows[0]["consumed_prefix_effect_fraction"] = 0.5
    rows[0]["anti_padding_verified"] = False
    rows[0]["effect_chain_verified"] = True
    rows[0]["meets_long_horizon_step_threshold"] = True
    span_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["task_span.jsonl"] = hashlib.sha256(
        span_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "effect-chain verification lacks anti-padding proof" in error
        for error in validation.errors
    )
    assert any(
        "long-horizon threshold lacks a 200-step terminal causal span" in error
        for error in validation.errors
    )


def test_validator_rejects_internally_inconsistent_design_audit(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "tampered-design-audit"
    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
    )
    audit_path = out / "experiment_design_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["run_ready"] = False
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["experiment_design_audit.json"] = hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any("design audit run_ready is inconsistent" in error for error in validation.errors)


def test_validator_rejects_truncated_design_audit_check_set(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "truncated-design-audit"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    audit_path = out / "experiment_design_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checks"].pop()
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["experiment_design_audit.json"] = hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any("complete ordered check set" in error for error in validation.errors)


def test_report_rejects_design_audit_that_differs_from_run_plan(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()

    with pytest.raises(ValueError, match="differs from the immutable run plan"):
        write_qualification_report(
            matrix,
            specs,  # type: ignore[arg-type]
            tmp_path / "wrong-planned-audit",
            run_metadata={"experiment_design_audit_hash": "0" * 64},
        )


def test_validator_binds_design_audit_content_to_report_manifest(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "changed-audit-detail"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    audit_path = out / "experiment_design_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["interpretation"] += " Changed after evaluation."
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["experiment_design_audit.json"] = hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any("audit hash differs" in error for error in validation.errors)


def test_validator_rejects_relabelled_pre_call_analysis_contract(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "changed-analysis-contract"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    audit_path = out / "experiment_design_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["analysis_contract"] = {
        "status": "pre_call_frozen",
        "claim_id": "C1",
        "primary_estimands": ["state_evolution_penalty_vs_static"],
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["experiment_design_audit.json"] = (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    manifest["experiment_design_audit_hash"] = canonical_hash(audit)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "analysis contract differs from the frozen" in error
        for error in validation.errors
    )


def test_validator_rejects_post_hoc_primary_estimand_relabeling(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "changed-primary-estimand"
    write_qualification_report(matrix, specs, out)  # type: ignore[arg-type]
    statistics_path = out / "matched_construct_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["primary_estimands"] = ["state_evolution_penalty_vs_static"]
    statistics_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["matched_construct_statistics.json"] = (
        hashlib.sha256(statistics_path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_qualification_artifacts(out)

    assert validation.ok is False
    assert any(
        "analysis contract differs for primary_estimands" in error
        for error in validation.errors
    )


def test_matched_statistics_validator_rejects_episode_pseudoreplication() -> None:
    errors: list[str] = []
    _validate_matched_construct_statistics(
        {
            "analysis_unit": "episode",
            "estimates": [
                {
                    "policy_profile_id": "gpt",
                    "condition": "mem0",
                    "readout": "native",
                    "metric": "state_evolution_penalty_vs_static",
                    "analysis_unit": "episode",
                    "n_pairs": 3,
                    "mean_difference": 0.2,
                    "ci_low": 0.1,
                    "ci_high": 0.3,
                    "permutation_p_value": 0.2,
                    "holm_adjusted_p_value": 0.2,
                }
            ],
        },
        errors,
    )

    assert any("counterfactual_group" in error for error in errors)
    assert any("wrong unit" in error for error in errors)


def test_report_separates_evaluated_subset_from_frozen_dataset_scope(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    extra = SoftwareMem0VerticalFamily.generate(43, n_sessions=4)
    frozen_specs = {**specs, extra.plan.episode_id: extra}

    write_qualification_report(
        matrix,
        frozen_specs,  # type: ignore[arg-type]
        tmp_path / "report",
    )

    limitations = (tmp_path / "report" / "limitations.md").read_text(
        encoding="utf-8"
    )
    assert "Evaluated episodes: 1 of 2 frozen" in limitations
    assert "Policy-free fixed-action and opaque-option baselines use the full frozen" in (
        limitations
    )
    summary = json.loads(
        (tmp_path / "report" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_evaluated_episodes"] == 1
    assert summary["n_frozen_dataset_episodes"] == 2


def test_summary_count_load_does_not_mix_in_leave_one_out_deletions() -> None:
    matrix, specs = _matrix()
    rows: defaultdict[str, tuple[dict[str, object], ...]] = defaultdict(tuple)
    rows["interventions.jsonl"] = (
        {
            "intervention_kind": "leave_one_out",
            "count_contrast": "delete_one",
        },
        {
            "intervention_kind": "count_add",
            "count_contrast": "add_5",
        },
    )

    summary = _summary(matrix, rows, specs=specs)  # type: ignore[arg-type]

    assert summary["n_memory_count_contrasts"] == 1


def test_summary_counts_matched_groups_instead_of_physical_members() -> None:
    base_matrix, _ = _matrix()
    base_task = base_matrix.task_results[0]
    triplet = SoftwareMatchedConstructFamily.generate_triplet(
        101,
        n_sessions=4,
        trajectory_seed=101,
        steps_per_session=2,
    )
    tasks = tuple(
        replace(
            base_task,
            task_id=f"task-{index}",
            episode_id=spec.plan.episode_id,
        )
        for index, spec in enumerate(triplet)
    )
    matrix = replace(base_matrix, task_results=tasks)
    specs = {spec.plan.episode_id: spec for spec in triplet}
    rows: defaultdict[str, tuple[dict[str, object], ...]] = defaultdict(tuple)

    summary = _summary(matrix, rows, specs=specs)

    assert summary["construct_mode"] == "matched_triplets"
    assert summary["primary_analysis_unit"] == "counterfactual_group"
    assert summary["n_physical_episodes"] == 3
    assert summary["n_counterfactual_groups"] == 1
    assert summary["n_statistical_units"] == 1
    assert summary["counterfactual_group_ids"] == ["software-cf-101-101"]


def test_summary_counts_a_horizon_panel_not_its_nine_members() -> None:
    base_matrix, _ = _matrix()
    base_task = base_matrix.task_results[0]
    panel = SoftwareHorizonPanelFamily.generate_panel(42, trajectory_seed=42)
    tasks = tuple(
        replace(
            base_task,
            task_id=f"task-{index}",
            episode_id=spec.plan.episode_id,
        )
        for index, spec in enumerate(panel)
    )
    matrix = replace(base_matrix, task_results=tasks)
    specs = {spec.plan.episode_id: spec for spec in panel}
    rows: defaultdict[str, tuple[dict[str, object], ...]] = defaultdict(tuple)

    summary = _summary(matrix, rows, specs=specs)

    assert summary["construct_mode"] == "horizon_panels"
    assert summary["primary_analysis_unit"] == "horizon_panel"
    assert summary["n_physical_episodes"] == 9
    assert summary["n_counterfactual_groups"] == 3
    assert summary["n_horizon_panels"] == 1
    assert summary["n_statistical_units"] == 1
    assert summary["horizon_panel_ids"] == [
        "software-horizon-panel-42-42"
    ]


def test_horizon_panel_report_round_trips_through_artifact_validation(
    tmp_path: Path,
) -> None:
    panel = SoftwareHorizonPanelFamily.generate_panel(42, trajectory_seed=42)
    tasks = []
    for index, spec in enumerate(panel):
        sceu = spec.plan.sceu_units[0]
        opportunity = spec.plan.opportunities[0]
        action_id = opportunity.valid_action_ids[0]
        behavior = ContinuationOutcome(
            action_id=action_id,
            behavior_score=1.0,
            is_correct=True,
        )
        row = SCEURunResult(
            result_id=f"result-{index}",
            sceu_id=sceu.sceu_id,
            opportunity_id=sceu.opportunity_id,
            checkpoint_session=sceu.checkpoint_session,
            matched_group=sceu.matched_group,
            control_kind="workspace",
            workspace_hash=f"workspace-{index}",
            candidate_memory_ids=(),
            retrieved_memory_ids=(),
            model_visible_memory_ids=(),
            selected_option_id="option-01",
            selected_action_id=action_id,
            behavior=behavior,
            normalized_drift_flags=(),
            baseline_stable=True,
            baseline_evaluations=(),
            interventions=(),
            retrieval_trace_id=None,
        )
        condition = ConditionRunResult(
            result_id=f"condition-{index}",
            condition="workspace_only",
            readout="none",
            status="complete",
            sceu_results=(row,),
        )
        tasks.append(
            QualificationTaskResult(
                task_id=f"task-{index}",
                episode_id=spec.plan.episode_id,
                policy_profile_id="policy-a",
                condition="workspace_only",
                status="complete",
                condition_results=(condition,),
                writes=(),
                alignments=(),
                retrieval_traces=(),
            )
        )
    matrix = QualificationMatrixResult(
        run_identity="horizon-run",
        task_results=tuple(tasks),
    )
    specs = {spec.plan.episode_id: spec for spec in panel}
    out = tmp_path / "horizon-report"

    write_qualification_report(matrix, specs, out)
    validation = validate_qualification_artifacts(
        out,
        expected_run_identity="horizon-run",
    )

    assert validation.ok, validation.errors
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    statistics = json.loads(
        (out / "horizon_panel_statistics.json").read_text(encoding="utf-8")
    )
    generic_statistics = json.loads(
        (out / "statistics.json").read_text(encoding="utf-8")
    )
    matched_statistics = json.loads(
        (out / "matched_construct_statistics.json").read_text(encoding="utf-8")
    )
    assert summary["primary_analysis_unit"] == "horizon_panel"
    assert summary["n_statistical_units"] == 1
    assert statistics["n_unique_horizon_panels"] == 1
    assert generic_statistics["status"] == (
        "suppressed_dependent_physical_members"
    )
    assert generic_statistics["analysis_unit"] == "horizon_panel"
    assert matched_statistics["status"] == "suppressed_within_panel_triplets"
    assert matched_statistics["estimates"] == []
    assert all(
        row["n_panels"] == 1 for row in statistics["estimates"]
    )


def test_storage_scorecard_separates_exact_and_inferred_provenance() -> None:
    def metric(value: float) -> dict[str, float]:
        return {"numerator": value, "denominator": 1.0, "value": value}

    rows = _storage_scorecard_rows(
        (
            {
                "policy_profile_id": "gpt",
                "condition": "memos",
                "readout": "native",
                "metrics": {"write_coverage": metric(0.1)},
            },
            {
                "policy_profile_id": "gpt",
                "condition": "memos",
                "readout": "common_rerank",
                "metrics": {
                    "write_coverage": metric(0.5),
                    "storage_exact_write_coverage": metric(0.75),
                    "storage_exact_semantic_attribution_ambiguous_rate": metric(0.1),
                    "storage_exact_semantic_attribution_unavailable_rate": metric(0.0),
                    "storage_inferred_write_coverage": metric(0.25),
                    "storage_inferred_semantic_attribution_ambiguous_rate": metric(0.2),
                    "storage_inferred_semantic_attribution_unavailable_rate": metric(0.1),
                },
            },
        )
    )

    assert [row["provenance_track"] for row in rows] == [
        "all",
        "exact",
        "inferred",
    ]
    assert {row["source_readout"] for row in rows} == {"common_rerank"}
    assert rows[1]["write_coverage"] == 0.75
    assert rows[1]["semantic_attribution_resolvability"] == 0.9
    assert rows[2]["semantic_attribution_resolvability"] == pytest.approx(0.7)


def test_memory_count_scorecard_keeps_matched_levels_separate() -> None:
    rows = _memory_count_scorecard_rows(
        (
            {
                "policy_profile_id": "gpt",
                "condition": "flat_retrieval",
                "readout": "common_rerank",
                "opportunity_id": "opp-stale-v1",
                "intervention_kind": "count_add",
                "count_contrast": "add_1",
                "baseline_memory_count": 5,
                "intervention_memory_count": 6,
                "classification": {
                    "action_changed": False,
                    "checker_changed": False,
                },
            },
            {
                "policy_profile_id": "gpt",
                "condition": "flat_retrieval",
                "readout": "common_rerank",
                "opportunity_id": "opp-stale-v1",
                "intervention_kind": "count_add",
                "count_contrast": "add_20",
                "baseline_memory_count": 5,
                "intervention_memory_count": 25,
                "classification": {
                    "action_changed": True,
                    "checker_changed": True,
                },
            },
            {
                "intervention_kind": "leave_one_out",
                "count_contrast": "delete_one",
                "baseline_memory_count": 5,
                "intervention_memory_count": 4,
            },
        )
    )

    assert [row["count_delta"] for row in rows] == [1, 20]
    assert rows[0]["action_flip_rate"] == 0.0
    assert rows[1]["behavior_change_rate"] == 1.0


def test_report_jsonl_files_are_valid_and_deterministically_sorted(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "report"
    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
    )
    for name in (
        "tasks.jsonl",
        "task_results.jsonl",
        "sceu_results.jsonl",
        "memory_events.jsonl",
        "memory_inventory.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
        "api_usage.jsonl",
    ):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        assert parsed == sorted(
            parsed,
            key=lambda item: json.dumps(item, sort_keys=True),
        )


def test_native_evaluation_trace_id_is_distinct_when_row_has_no_trace() -> None:
    assert (
        _evaluation_trace_id("task-001", "sceu-00", "common_rerank", "trace-1")
        == "trace-1"
    )
    assert (
        _evaluation_trace_id("task-001", "sceu-00", "native", None)
        == "task-001:sceu-00:native"
    )


def test_storage_provenance_uses_checkpoint_write_deltas() -> None:
    rows = {
        "memory_events.jsonl": [
            {
                "task_id": "task-memory",
                "session_index": 0,
                "provenance_mode": "native/exact",
                "source": "native_response",
            }
        ],
        "memory_inventory.jsonl": [
            {"task_id": "task-memory", "checkpoint_session": 0, "n_write": 0},
            {"task_id": "task-memory", "checkpoint_session": 1, "n_write": 1},
            # A no-op session retains the cumulative write count and needs no
            # new lifecycle event.
            {"task_id": "task-memory", "checkpoint_session": 2, "n_write": 1},
        ],
    }

    complete = _storage_provenance_diagnostics(rows)
    assert complete["status"] == "complete"
    assert complete["incomplete_write_checkpoints"] == []

    rows["memory_inventory.jsonl"].append(
        {"task_id": "task-memory", "checkpoint_session": 3, "n_write": 2}
    )
    incomplete = _storage_provenance_diagnostics(rows)
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_write_tasks"] == ["task-memory"]
    assert incomplete["incomplete_write_checkpoints"] == [
        {
            "task_id": "task-memory",
            "checkpoint_session": 3,
            "write_delta": 1,
            "event_count": 0,
        }
    ]


def test_semantic_attribution_is_reported_independently_from_event_provenance() -> None:
    rows = {
        "memory_inventory.jsonl": [
            {
                "task_id": "task-memory",
                "checkpoint_session": 1,
                "evaluator_attribution_by_memory": {
                    "old": {
                        "method": "ambiguous",
                        "provenance_mode": "native/exact",
                        "contributes_positive_coverage": False,
                    }
                },
            },
            {
                "task_id": "task-memory",
                "checkpoint_session": 2,
                "evaluator_attribution_by_memory": {
                    "exact": {
                        "method": "exact_signature",
                        "provenance_mode": "native/exact",
                        "contributes_positive_coverage": True,
                    },
                    "ambiguous": {
                        "method": "ambiguous",
                        "provenance_mode": "native/exact",
                        "contributes_positive_coverage": False,
                    },
                },
            },
        ]
    }

    diagnostics = _semantic_attribution_diagnostics(rows)

    assert diagnostics["scope"] == "latest_inventory_per_task"
    assert diagnostics["n_memory_objects"] == 2
    assert diagnostics["method_counts"] == {
        "ambiguous": 1,
        "exact_signature": 1,
    }
    assert diagnostics["lifecycle_provenance_counts"] == {"native/exact": 2}
    assert diagnostics["positive_coverage_rate"] == 0.5
    assert diagnostics["status"] == "complete"

    rows["memory_inventory.jsonl"][-1]["evaluator_attribution_by_memory"][
        "exact"
    ].pop("contributes_positive_coverage")
    incomplete = _semantic_attribution_diagnostics(rows)
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_objects"] == ["task-memory:exact"]


def test_ambiguous_semantic_attribution_cannot_score_positive_coverage() -> None:
    errors: list[str] = []
    _validate_semantic_attributions(
        (
            {
                "task_id": "task-memory",
                "checkpoint_session": 2,
                "evaluator_attribution_by_memory": {
                    "memory-1": {
                        "method": "ambiguous",
                        "provenance_mode": "native/exact",
                        "contributes_positive_coverage": True,
                    }
                },
            },
        ),
        errors,
    )

    assert errors == [
        "ambiguous semantic attribution contributes positive coverage for "
        "task-memory:2:memory-1"
    ]


def test_prefix_reranker_usage_is_exported_for_common_readout_only() -> None:
    trace = CommonRerankTrace(
        opportunity_id="opp-early",
        query_hash="1" * 64,
        candidate_memory_ids=("memory-1", "memory-2"),
        visible_memory_ids=("memory-2",),
        result=RerankResult(
            ordered_memory_ids=("memory-2",),
            scores=(0.9,),
            model="BAAI/bge-reranker-v2-m3",
            revision="revision-1",
            input_count=2,
            request_hash="2" * 64,
            response_hash="3" * 64,
            latency_seconds=0.125,
        ),
    )
    rows: list[dict[str, object]] = []
    seen_calls: set[str] = set()

    _append_prefix_reranker_usage(
        rows,
        seen_calls,
        {"task_id": "task-memory", "condition": "mem0"},
        checkpoint_session=3,
        trace=trace,
    )
    _append_prefix_reranker_usage(
        rows,
        seen_calls,
        {"task_id": "task-memory", "condition": "mem0"},
        checkpoint_session=3,
        trace=trace,
    )

    assert len(rows) == 1
    assert rows[0]["readout"] == "common_rerank"
    assert rows[0]["call_kind"] == "reranker"
    assert rows[0]["input_count"] == 2
    assert rows[0]["latency_seconds"] == 0.125
