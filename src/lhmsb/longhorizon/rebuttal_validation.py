"""Zero-API validation gates for saved AAAI rebuttal calibration evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CALIBRATION_VALIDATION_SCHEMA_VERSION = "memhorizon-calibration-validation-v1"
REQUIRED_FILES = (
    "summary.json",
    "drift_calibration.json",
    "drift_trajectories.json",
    "statistics.json",
    "scorecard.csv",
    "failure_attribution_scorecard.csv",
    "memory_count_scorecard.csv",
    "storage_scorecard.csv",
)
MEMORY_SYSTEMS = {"flat_retrieval", "mem0", "amem", "memos"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check(checks: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def validate_calibration_evidence(calibration_dir: Path) -> dict[str, object]:
    """Apply contribution-facing completeness and lineage-integrity gates."""
    checks: list[dict[str, object]] = []
    missing_files = [name for name in REQUIRED_FILES if not (calibration_dir / name).is_file()]
    _check(checks, "required_artifacts_present", not missing_files, missing_files)
    if missing_files:
        return {
            "schema_version": CALIBRATION_VALIDATION_SCHEMA_VERSION,
            "calibration_dir": str(calibration_dir.resolve()),
            "ok": False,
            "checks": checks,
        }

    summary = _read_json(calibration_dir / "summary.json")
    _check(
        checks,
        "analysis_phase_is_calibration",
        summary.get("analysis_phase") == "calibration",
        summary.get("analysis_phase"),
    )
    _check(
        checks,
        "analysis_is_prespecified",
        summary.get("analysis_timing") == "pre_specified",
        summary.get("analysis_timing"),
    )
    _check(
        checks,
        "episode_is_analysis_unit",
        summary.get("primary_analysis_unit") == "episode",
        summary.get("primary_analysis_unit"),
    )
    _check(
        checks,
        "five_calibration_episodes",
        summary.get("n_evaluated_episodes") == 5,
        summary.get("n_evaluated_episodes"),
    )
    _check(
        checks,
        "all_tasks_observed",
        summary.get("n_missing_task_results") == 0
        and summary.get("n_observed_task_results") == summary.get("n_planned_tasks") == 35,
        {
            key: summary.get(key)
            for key in ("n_missing_task_results", "n_observed_task_results", "n_planned_tasks")
        },
    )
    _check(
        checks,
        "all_tasks_complete",
        summary.get("task_status_counts") == {"complete": 35},
        summary.get("task_status_counts"),
    )
    _check(
        checks,
        "all_conditions_complete",
        summary.get("condition_status_counts") == {"complete": 50},
        summary.get("condition_status_counts"),
    )
    _check(
        checks,
        "decision_attribution_complete",
        summary.get("n_sceu_results") == summary.get("n_decision_attribution_rows") == 900,
        {key: summary.get(key) for key in ("n_sceu_results", "n_decision_attribution_rows")},
    )
    _check(
        checks,
        "failure_cells_complete",
        summary.get("n_failure_attribution_cells") == 20,
        summary.get("n_failure_attribution_cells"),
    )
    storage = summary.get("storage_provenance")
    storage_ok = (
        isinstance(storage, Mapping)
        and storage.get("status") == "complete"
        and storage.get("unavailable_event_count") == 0
        and int(storage.get("native_exact_event_count", 0)) > 0
        and int(storage.get("inferred_event_count", 0)) > 0
    )
    _check(checks, "storage_provenance_complete_and_labeled", storage_ok, storage)

    drift_calibration = _read_json(calibration_dir / "drift_calibration.json")
    _check(
        checks,
        "drift_categories_calibrated",
        drift_calibration.get("all_categories_calibrated") is True,
        drift_calibration.get("all_categories_calibrated"),
    )
    _check(
        checks,
        "represented_scenarios_calibrated",
        drift_calibration.get("all_represented_scenarios_calibrated") is True,
        drift_calibration.get("all_represented_scenarios_calibrated"),
    )

    drift_payload = _read_json(calibration_dir / "drift_trajectories.json")
    raw_trajectories = drift_payload.get("trajectories")
    trajectories = raw_trajectories if isinstance(raw_trajectories, list) else []
    events = [
        row
        for row in trajectories
        if isinstance(row, Mapping) and row.get("event_observed") is True
    ]
    invalid_events = [
        {
            "episode_id": row.get("episode_id"),
            "condition": row.get("condition"),
            "readout": row.get("readout"),
            "category": row.get("drift_category"),
        }
        for row in events
        if not (
            row.get("adherence_anchor_observed") is True
            and row.get("lineage_backed") is True
            and isinstance(row.get("first_adherence_session"), int)
            and isinstance(row.get("first_drift_session"), int)
            and row["first_adherence_session"] < row["first_drift_session"]
        )
    ]
    _check(
        checks,
        "trajectory_count_matches_summary",
        len(trajectories) == summary.get("n_drift_trajectories") == 200,
        len(trajectories),
    )
    _check(
        checks,
        "drift_requires_prior_same_lineage_adherence",
        bool(events) and not invalid_events,
        {"event_count": len(events), "invalid": invalid_events},
    )
    invalid_recovery = [
        row.get("episode_id")
        for row in trajectories
        if isinstance(row, Mapping)
        and row.get("recovered") is True
        and not (
            isinstance(row.get("recovery_session"), int)
            and isinstance(row.get("first_drift_session"), int)
            and row["recovery_session"] > row["first_drift_session"]
        )
    ]
    _check(checks, "recovery_occurs_after_drift", not invalid_recovery, invalid_recovery)

    scorecard = _read_csv(calibration_dir / "scorecard.csv")
    _check(
        checks,
        "scorecard_cells_complete",
        len(scorecard) == 10
        and all(row.get("status") == "complete" and row.get("n_sceu") == "90" for row in scorecard),
        {
            "row_count": len(scorecard),
            "incomplete": sum(row.get("status") != "complete" for row in scorecard),
        },
    )
    failure = _read_csv(calibration_dir / "failure_attribution_scorecard.csv")
    relevant = [
        row
        for row in failure
        if row.get("readout") == "common_rerank"
        and row.get("condition") in MEMORY_SYSTEMS
        and row.get("storage_evidence_mode") in {"native/exact", "inferred"}
    ]
    _check(
        checks,
        "failure_attribution_common_readout_complete",
        len(failure) == 20
        and {row.get("condition") for row in relevant} == MEMORY_SYSTEMS
        and all(row.get("n_sceu") == "54" for row in relevant),
        {
            "row_count": len(failure),
            "relevant_conditions": sorted(str(row.get("condition")) for row in relevant),
        },
    )
    _check(
        checks,
        "memory_count_contrasts_present",
        len(_read_csv(calibration_dir / "memory_count_scorecard.csv")) > 0,
        None,
    )
    storage_rows = _read_csv(calibration_dir / "storage_scorecard.csv")
    _check(
        checks,
        "storage_tracks_labeled",
        {row.get("provenance_track") for row in storage_rows} >= {"exact", "inferred"},
        sorted({str(row.get("provenance_track")) for row in storage_rows}),
    )

    statistics = _read_json(calibration_dir / "statistics.json")
    _check(
        checks,
        "statistics_use_episode_unit",
        statistics.get("analysis_unit") == "episode" and statistics.get("n_unique_episodes") == 5,
        {
            "analysis_unit": statistics.get("analysis_unit"),
            "n_unique_episodes": statistics.get("n_unique_episodes"),
        },
    )
    failed = [str(check["name"]) for check in checks if not check["passed"]]
    return {
        "schema_version": CALIBRATION_VALIDATION_SCHEMA_VERSION,
        "calibration_dir": str(calibration_dir.resolve()),
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_checks": failed,
        "ok": not failed,
        "checks": checks,
    }


def validate_terminal_audit(
    audit_path: Path,
    *,
    expected_result_count: int,
) -> dict[str, object]:
    """Gate replay integrity while leaving scientific outcomes unconstrained."""
    if expected_result_count < 1:
        raise ValueError("expected result count must be positive")
    payload = _read_json(audit_path)
    raw_records = payload.get("records")
    records = raw_records if isinstance(raw_records, list) else []
    checks: list[dict[str, object]] = []
    _check(
        checks,
        "all_expected_artifacts_audited",
        payload.get("artifact_count") == expected_result_count
        and len(records) == expected_result_count,
        {"artifact_count": payload.get("artifact_count"), "record_count": len(records)},
    )
    _check(
        checks,
        "all_artifacts_replayed",
        payload.get("replayed_count") == expected_result_count
        and payload.get("status_counts") == {"replayed": expected_result_count},
        {
            "replayed_count": payload.get("replayed_count"),
            "status_counts": payload.get("status_counts"),
        },
    )
    _check(
        checks,
        "no_infrastructure_invalid_artifacts",
        payload.get("infrastructure_invalid_count") == 0,
        payload.get("infrastructure_invalid_count"),
    )
    invalid_contracts = [
        record.get("result_id")
        for record in records
        if isinstance(record, Mapping)
        and (
            not isinstance(record.get("replayed_terminal"), Mapping)
            or record["replayed_terminal"].get("workspace_contract_valid") is not True
        )
    ]
    _check(
        checks,
        "all_workspace_contracts_valid",
        not invalid_contracts,
        invalid_contracts,
    )
    invalid_traces = [
        record.get("result_id")
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("trace_reconstruction"), Mapping)
        and record["trace_reconstruction"].get("trace_integrity_valid") is not True
    ]
    _check(checks, "all_reconstructed_traces_valid", not invalid_traces, invalid_traces)
    failed = [str(check["name"]) for check in checks if not check["passed"]]
    return {
        "schema_version": "memhorizon-terminal-audit-validation-v1",
        "audit_path": str(audit_path.resolve()),
        "expected_result_count": expected_result_count,
        "outcomes_are_not_integrity_gates": True,
        "failed_checks": failed,
        "ok": not failed,
        "checks": checks,
    }


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration-dir", type=Path)
    source.add_argument("--terminal-audit", type=Path)
    parser.add_argument("--expected-result-count", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.calibration_dir is not None:
        report = validate_calibration_evidence(args.calibration_dir)
    else:
        if args.expected_result_count is None:
            raise SystemExit("--terminal-audit requires --expected-result-count")
        report = validate_terminal_audit(
            args.terminal_audit,
            expected_result_count=args.expected_result_count,
        )
    if args.out is not None:
        _atomic_write(args.out, report)
    print(
        json.dumps({key: value for key, value in report.items() if key != "checks"}, sort_keys=True)
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_VALIDATION_SCHEMA_VERSION",
    "REQUIRED_FILES",
    "main",
    "validate_calibration_evidence",
    "validate_terminal_audit",
]
