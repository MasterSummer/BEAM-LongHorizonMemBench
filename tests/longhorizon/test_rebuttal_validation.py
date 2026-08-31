from __future__ import annotations

import csv
import json
from pathlib import Path

from lhmsb.longhorizon.rebuttal_validation import (
    validate_calibration_evidence,
    validate_terminal_audit,
)


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _calibration_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "calibration"
    root.mkdir()
    _json(
        root / "summary.json",
        {
            "analysis_phase": "calibration",
            "analysis_timing": "pre_specified",
            "primary_analysis_unit": "episode",
            "n_evaluated_episodes": 5,
            "n_missing_task_results": 0,
            "n_observed_task_results": 35,
            "n_planned_tasks": 35,
            "task_status_counts": {"complete": 35},
            "condition_status_counts": {"complete": 50},
            "n_sceu_results": 900,
            "n_decision_attribution_rows": 900,
            "n_failure_attribution_cells": 20,
            "n_drift_trajectories": 200,
            "storage_provenance": {
                "status": "complete",
                "unavailable_event_count": 0,
                "native_exact_event_count": 12,
                "inferred_event_count": 4,
            },
        },
    )
    _json(
        root / "drift_calibration.json",
        {
            "all_categories_calibrated": True,
            "all_represented_scenarios_calibrated": True,
        },
    )
    trajectories = [
        {
            "episode_id": f"episode-{index % 5}",
            "condition": "mem0",
            "readout": "common_rerank",
            "drift_category": "stale_state",
            "event_observed": index == 0,
            "adherence_anchor_observed": index == 0,
            "lineage_backed": True,
            "first_adherence_session": 1,
            "first_drift_session": 3,
            "recovered": index == 0,
            "recovery_session": 4,
        }
        for index in range(200)
    ]
    _json(root / "drift_trajectories.json", {"trajectories": trajectories})
    _json(root / "statistics.json", {"analysis_unit": "episode", "n_unique_episodes": 5})
    _csv(
        root / "scorecard.csv",
        ["status", "n_sceu"],
        [{"status": "complete", "n_sceu": "90"} for _ in range(10)],
    )
    systems = ["flat_retrieval", "mem0", "amem", "memos"]
    relevant = [
        {
            "condition": system,
            "readout": "common_rerank",
            "storage_evidence_mode": "inferred" if system == "memos" else "native/exact",
            "n_sceu": "54",
        }
        for system in systems
    ]
    filler = [
        {
            "condition": systems[index % 4],
            "readout": "native",
            "storage_evidence_mode": "not_applicable",
            "n_sceu": "36",
        }
        for index in range(16)
    ]
    fields = ["condition", "readout", "storage_evidence_mode", "n_sceu"]
    _csv(root / "failure_attribution_scorecard.csv", fields, relevant + filler)
    _csv(root / "memory_count_scorecard.csv", ["condition"], [{"condition": "mem0"}])
    _csv(
        root / "storage_scorecard.csv",
        ["provenance_track"],
        [{"provenance_track": "exact"}, {"provenance_track": "inferred"}],
    )
    return root


def test_valid_calibration_evidence_passes_all_gates(tmp_path: Path) -> None:
    report = validate_calibration_evidence(_calibration_fixture(tmp_path))

    assert report["ok"]
    assert report["failed_checks"] == []
    assert report["passed_check_count"] == report["check_count"]


def test_drift_without_prior_adherence_fails_lineage_gate(tmp_path: Path) -> None:
    root = _calibration_fixture(tmp_path)
    payload = json.loads((root / "drift_trajectories.json").read_text())
    payload["trajectories"][0]["adherence_anchor_observed"] = False
    _json(root / "drift_trajectories.json", payload)

    report = validate_calibration_evidence(root)

    assert not report["ok"]
    assert "drift_requires_prior_same_lineage_adherence" in report["failed_checks"]


def test_missing_artifact_fails_fast(tmp_path: Path) -> None:
    report = validate_calibration_evidence(tmp_path)

    assert not report["ok"]
    assert report["checks"][0]["name"] == "required_artifacts_present"  # type: ignore[index]


def test_terminal_audit_gate_checks_integrity_not_scientific_outcome(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    _json(
        path,
        {
            "artifact_count": 1,
            "replayed_count": 1,
            "infrastructure_invalid_count": 0,
            "status_counts": {"replayed": 1},
            "verdict_changed_count": 1,
            "records": [
                {
                    "result_id": "result-1",
                    "replayed_terminal": {
                        "workspace_contract_valid": True,
                        "is_correct": False,
                    },
                    "trace_reconstruction": {"trace_integrity_valid": True},
                }
            ],
        },
    )

    report = validate_terminal_audit(path, expected_result_count=1)

    assert report["ok"]
    assert report["outcomes_are_not_integrity_gates"]


def test_terminal_audit_gate_rejects_invalid_workspace_contract(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    _json(
        path,
        {
            "artifact_count": 1,
            "replayed_count": 1,
            "infrastructure_invalid_count": 0,
            "status_counts": {"replayed": 1},
            "records": [
                {
                    "result_id": "result-1",
                    "replayed_terminal": {"workspace_contract_valid": False},
                }
            ],
        },
    )

    report = validate_terminal_audit(path, expected_result_count=1)

    assert not report["ok"]
    assert "all_workspace_contracts_valid" in report["failed_checks"]
