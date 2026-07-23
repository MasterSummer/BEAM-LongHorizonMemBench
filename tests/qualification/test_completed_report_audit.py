from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lhmsb.qualification.cli import main
from lhmsb.qualification.completed_report_audit import (
    CompletedReportAuditError,
    audit_completed_report,
    write_completed_report_audit,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _text(path: Path, value: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _seal_report(
    root: Path,
    *,
    timing: str | None = None,
    phase: str | None = None,
    code_dirty: bool = False,
    dataset_manifest_sha256: str | None = None,
) -> None:
    _json(
        root / "validation.json",
        {"schema_version": 2, "ok": True, "errors": []},
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "run_identity": "run-123",
        "dataset_release": "test-release",
        "code_commit": "a" * 40,
        "code_dirty": code_dirty,
    }
    if timing is not None:
        manifest["analysis_timing"] = timing
    if phase is not None:
        manifest["analysis_phase"] = phase
    if dataset_manifest_sha256 is not None:
        manifest["dataset_manifest_sha256"] = dataset_manifest_sha256
    manifest["artifact_hashes"] = {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    _json(root / "run_manifest.json", manifest)


def _frozen_dataset(tmp_path: Path, *, episode_id: str = "episode-1") -> Path:
    root = tmp_path / "frozen-dataset"
    artifacts = {
        "evaluator/episodes.jsonl": json.dumps({"episode_id": episode_id}) + "\n",
        "evaluator/sceu.jsonl": json.dumps(
            {
                "episode_id": episode_id,
                "sceu_id": "sceu-1",
                "required_state_ids": ["G0"],
            }
        )
        + "\n",
        "evaluator/state_units.jsonl": "{}\n",
        "evaluator/state_events.jsonl": "{}\n",
        "evaluator/dependencies.json": "{}\n",
        "evaluator/continuation_mappings.jsonl": "{}\n",
    }
    for relative, value in artifacts.items():
        _text(root / relative, value)
    _json(
        root / "MANIFEST.json",
        {
            "schema_version": 1,
            "files": {
                relative: _digest(root / relative) for relative in sorted(artifacts)
            },
        },
    )
    return root


def _legacy_report(
    tmp_path: Path,
    *,
    dataset_manifest_sha256: str | None = None,
) -> Path:
    root = tmp_path / "legacy-report"
    _json(
        root / "summary.json",
        {
            "schema_version": 2,
            "run_identity": "run-123",
            "n_tasks": 35,
            "evaluated_episode_ids": ["episode-1"],
            "storage_provenance": {
                "status": "complete",
                "incomplete_write_checkpoints": [],
                "incomplete_write_tasks": [],
            },
        },
    )
    _json(
        root / "measurement_gates.json",
        {
            "schema_version": 1,
            "measurement_ready": False,
            "gates": [
                {"gate_id": "oracle_accuracy_by_scenario", "status": "fail"},
                {"gate_id": "lifecycle_provenance_complete", "status": "pass"},
            ],
        },
    )
    _json(root / "drift_calibration.json", {"schema_version": 1})
    _text(root / "scorecard.csv", "condition,behavior_correct_rate\n")
    for name in (
        "task_results.jsonl",
        "sceu_results.jsonl",
        "memory_events.jsonl",
        "memory_inventory.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
    ):
        _text(root / name, "{}\n")
    _seal_report(root, dataset_manifest_sha256=dataset_manifest_sha256)
    return root


def _current_report(
    tmp_path: Path,
    *,
    summary_timing: str = "pre_specified",
    manifest_timing: str = "pre_specified",
) -> Path:
    root = tmp_path / "current-report"
    _json(
        root / "summary.json",
        {
            "schema_version": 7,
            "run_identity": "run-123",
            "construct_mode": "matched_triplets",
            "analysis_phase": "confirmatory",
            "analysis_timing": summary_timing,
            "n_matched_construct_contrasts": 3,
            "n_horizon_panel_contrasts": 0,
            "n_long_horizon_control_contrasts": 3,
            "n_drift_trajectories": 3,
            "n_decision_attribution_rows": 3,
        },
    )
    _json(
        root / "measurement_gates.json",
        {
            "schema_version": 4,
            "measurement_ready": True,
            "gates": [{"gate_id": "task_completion", "status": "pass"}],
        },
    )
    _json(
        root / "contribution_evidence.json",
        {
            "schema_version": 3,
            "analysis_phase": "confirmatory",
            "analysis_timing": summary_timing,
            "contributions": [
                {
                    "contribution_id": contribution_id,
                    "evidence_status": "ready",
                    "claim_scope": "test-scope",
                    "claim_timing": summary_timing,
                }
                for contribution_id in ("C1", "C2", "C3")
            ],
        },
    )
    json_artifacts = (
        "experiment_design_audit.json",
        "matched_construct_statistics.json",
        "horizon_panel_statistics.json",
        "drift_calibration.json",
        "drift_trajectories.json",
        "fault_profile_divergence.json",
    )
    for name in json_artifacts:
        _json(root / name, {"schema_version": 1})
    jsonl_artifacts = (
        "task_results.jsonl",
        "sceu_results.jsonl",
        "memory_events.jsonl",
        "memory_inventory.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
        "long_horizon_constructs.jsonl",
        "task_span.jsonl",
        "matched_construct_contrasts.jsonl",
        "horizon_panel_contrasts.jsonl",
        "decision_attribution.jsonl",
    )
    for name in jsonl_artifacts:
        _text(root / name, "{}\n")
    for name in (
        "scorecard.csv",
        "long_horizon_control_contrasts.csv",
        "failure_attribution_scorecard.csv",
    ):
        _text(root / name, "field\n")
    _seal_report(
        root,
        timing=manifest_timing,
        phase="confirmatory",
    )
    return root


def test_legacy_report_is_not_backdated_to_current_contribution_contract(
    tmp_path: Path,
) -> None:
    report = _legacy_report(tmp_path)

    audit = audit_completed_report(report)

    assert audit["audit_analysis_timing"] == "post_hoc_scope_audit"
    assert audit["source_identity"]["source_analysis_timing"] == "undeclared_legacy"
    assert audit["source_identity"]["source_analysis_phase"] == "undeclared_legacy"
    assert audit["source_integrity"]["ok"] is True
    assert audit["measurement_contract"]["measurement_ready"] is False
    assert audit["raw_reanalysis"]["raw_trace_bundle_complete"] is True
    assert audit["raw_reanalysis"]["zero_API_reaggregation_candidate"] is False
    assert audit["raw_reanalysis"]["frozen_evaluator_dataset"]["status"] == "not_provided"
    contributions = {row["contribution_id"]: row for row in audit["contributions"]}
    assert contributions["C1"]["strongest_observed_artifact_level"] == (
        "endpoint_behavior_artifacts"
    )
    assert contributions["C2"]["strongest_observed_artifact_level"] == (
        "endpoint_violation_artifacts"
    )
    assert contributions["C3"]["strongest_observed_artifact_level"] == (
        "trace_and_intervention_artifacts_without_current_contract"
    )
    assert audit["claim_permissions"]["source_measurement_contract_confirmatory_eligible"] is False
    assert "current_contribution_evidence_contract_missing" in audit["gaps"]
    assert "measurement_gate:oracle_accuracy_by_scenario" in audit["gaps"]


def test_current_prespecified_report_can_retain_source_contract_eligibility(
    tmp_path: Path,
) -> None:
    report = _current_report(tmp_path)

    audit = audit_completed_report(
        report,
        audit_analysis_timing="post_hoc_exploratory",
    )

    assert audit["source_identity"]["source_analysis_timing"] == "pre_specified"
    assert audit["source_identity"]["source_analysis_phase"] == "confirmatory"
    assert audit["audit_analysis_timing"] == "post_hoc_exploratory"
    assert audit["measurement_contract"]["all_contributions_evidence_ready"] is True
    assert all(row["current_contract_artifacts_complete"] for row in audit["contributions"])
    assert audit["claim_permissions"]["source_measurement_contract_confirmatory_eligible"] is True
    assert audit["claim_permissions"]["effect_claim_established_by_this_audit"] is False


def test_inconsistent_source_timing_blocks_confirmatory_eligibility(
    tmp_path: Path,
) -> None:
    report = _current_report(
        tmp_path,
        summary_timing="post_hoc_scope_audit",
        manifest_timing="pre_specified",
    )

    audit = audit_completed_report(report)

    assert audit["source_identity"]["source_analysis_timing"] == "inconsistent"
    assert audit["claim_permissions"]["source_measurement_contract_confirmatory_eligible"] is False


def test_writer_is_separate_hashed_and_does_not_mutate_source(
    tmp_path: Path,
) -> None:
    report = _legacy_report(tmp_path)
    before = _source_bytes(report)
    output = tmp_path / "report-posthoc-audit"

    written = write_completed_report_audit(report, output)

    assert written == output.resolve()
    assert before == _source_bytes(report)
    assert (output / "completed_report_audit.json").is_file()
    assert (output / "completed_report_audit.md").is_file()
    manifest = json.loads((output / "audit_manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        assert _digest(output / name) == expected


def test_writer_rejects_source_tree_output_and_prespecified_audit_label(
    tmp_path: Path,
) -> None:
    report = _legacy_report(tmp_path)

    with pytest.raises(CompletedReportAuditError, match="outside the source"):
        write_completed_report_audit(report, report / "audit")
    with pytest.raises(CompletedReportAuditError, match="cannot be labelled"):
        audit_completed_report(report, audit_analysis_timing="pre_specified")


def test_cli_dispatches_completed_report_audit_without_api_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = _frozen_dataset(tmp_path)
    report = _legacy_report(
        tmp_path,
        dataset_manifest_sha256=_digest(dataset / "MANIFEST.json"),
    )
    output = tmp_path / "audit"

    code = main(
        [
            "audit-completed-report",
            "--report",
            str(report),
            "--dataset",
            str(dataset),
            "--out",
            str(output),
        ]
    )

    assert code == 0
    assert "completed-report contribution audit" in capsys.readouterr().out
    payload = json.loads((output / "completed_report_audit.json").read_text())
    assert payload["raw_reanalysis"]["zero_API_reaggregation_candidate"] is True


def test_zero_api_candidate_requires_exact_frozen_dataset(tmp_path: Path) -> None:
    dataset = _frozen_dataset(tmp_path)
    report = _legacy_report(
        tmp_path,
        dataset_manifest_sha256=_digest(dataset / "MANIFEST.json"),
    )

    without_dataset = audit_completed_report(report)
    with_dataset = audit_completed_report(report, frozen_dataset=dataset)

    assert without_dataset["raw_reanalysis"]["zero_API_reaggregation_candidate"] is False
    assert with_dataset["raw_reanalysis"]["zero_API_reaggregation_candidate"] is True
    support = with_dataset["raw_reanalysis"]["frozen_evaluator_dataset"]
    assert support["status"] == "verified"
    assert support["manifest_hash_matches_source"] is True
    assert support["evaluator_sceu_rows_for_evaluated_episodes"] == 1


def test_mismatched_frozen_dataset_cannot_support_reaggregation(tmp_path: Path) -> None:
    dataset = _frozen_dataset(tmp_path)
    report = _legacy_report(tmp_path, dataset_manifest_sha256="0" * 64)

    audit = audit_completed_report(report, frozen_dataset=dataset)

    support = audit["raw_reanalysis"]["frozen_evaluator_dataset"]
    assert support["status"] == "dataset_manifest_hash_mismatch"
    assert audit["raw_reanalysis"]["zero_API_reaggregation_candidate"] is False
