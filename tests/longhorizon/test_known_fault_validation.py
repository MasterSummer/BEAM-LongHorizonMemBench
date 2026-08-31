from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lhmsb.longhorizon.known_fault_validation import (
    RawLifecycleEvent,
    attribute_raw_trace,
    build_known_fault_trace,
    run_known_fault_suite,
    write_known_fault_report,
)


def test_canonical_known_fault_suite_is_exact_and_complete() -> None:
    report = run_known_fault_suite()

    assert report["ok"]
    assert report["case_count"] == 100
    assert report["passed_case_count"] == 100
    assert report["exact_case_count"] == 75
    assert report["exact_passed_case_count"] == 75
    assert report["inferred_case_count"] == 25
    assert report["inferred_passed_case_count"] == 25
    assert report["no_fault_false_positive_count"] == 0
    assert report["failed_case_ids"] == []
    confusion = report["confusion"]
    for label in (
        "no_fault",
        "storage_missing",
        "retrieval_missing",
        "exposure_missing",
        "visible_but_wrong",
    ):
        assert confusion[label][label] == 20
        assert sum(confusion[label].values()) == 20


@pytest.mark.parametrize(
    ("fault", "stage"),
    (
        ("no_fault", "behavior_success_causal"),
        ("storage_missing", "storage_failure"),
        ("retrieval_missing", "retrieval_failure"),
        ("exposure_missing", "exposure_failure"),
        ("visible_but_wrong", "utilization_failure"),
    ),
)
def test_raw_backend_events_reach_expected_production_stage(
    fault: str,
    stage: str,
) -> None:
    trace = build_known_fault_trace(
        "software-longitudinal-301",
        "mem0",
        fault,  # type: ignore[arg-type]
    )

    assert attribute_raw_trace(trace).stage == stage


def test_raw_trace_rejects_downstream_event_without_upstream_evidence() -> None:
    trace = build_known_fault_trace(
        "software-longitudinal-301",
        "mem0",
        "retrieval_missing",
    )
    invalid = replace(
        trace,
        events=(
            RawLifecycleEvent(
                sequence=0,
                event_kind="common_rerank_visible",
                state_id="P2",
                object_id="object-0",
            ),
        ),
    )

    with pytest.raises(ValueError, match="upstream retrieval"):
        attribute_raw_trace(invalid)


def test_known_fault_report_writes_summary_ledger_and_confusion(tmp_path: Path) -> None:
    report = run_known_fault_suite(("episode-1",))
    write_known_fault_report(report, tmp_path)

    summary = json.loads((tmp_path / "known_fault_summary.json").read_text())
    cases = (tmp_path / "known_fault_cases.jsonl").read_text().splitlines()
    confusion = list(csv.DictReader((tmp_path / "known_fault_confusion.csv").open()))
    assert summary["case_count"] == 20
    assert "cases" not in summary
    assert len(cases) == 20
    assert len(confusion) == 5
