from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.qualification.cli import main
from lhmsb.qualification.completed_report_reanalysis import (
    CompletedReportReanalysisError,
    reanalyze_completed_report,
    write_completed_report_reanalysis,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    spec = SoftwareMem0VerticalFamily.generate(0, n_sessions=16, trajectory_seed=2)
    sceu = next(
        item
        for item in spec.plan.sceu_units
        if profile_sceu(spec.plan, item).memory_reliant_state_ids
    )
    required_state = profile_sceu(spec.plan, sceu).memory_reliant_state_ids[0]
    episode_id = spec.plan.episode_id
    plan_payload = spec.plan.to_dict()

    dataset = tmp_path / "dataset"
    dataset_artifacts = {
        "evaluator/episodes.jsonl": [
            {
                "episode_id": episode_id,
                "plan": plan_payload,
            }
        ],
        "evaluator/sceu.jsonl": list(plan_payload["sceu_units"]),
        "evaluator/state_units.jsonl": list(plan_payload["state_units"]),
        "evaluator/state_events.jsonl": list(plan_payload["events"]),
        "evaluator/continuation_mappings.jsonl": [{}],
    }
    for relative, rows in dataset_artifacts.items():
        _jsonl(dataset / relative, rows)
    _json(dataset / "evaluator/dependencies.json", {})
    declared_files = {
        path.relative_to(dataset).as_posix(): _digest(path)
        for path in sorted(item for item in dataset.rglob("*") if item.is_file())
    }
    _json(dataset / "MANIFEST.json", {"schema_version": 1, "files": declared_files})

    report = tmp_path / "report"
    conditions = {
        "mem0": "storage_failure",
        "flat_retrieval": "retrieval_failure",
        "amem": "exposure_failure",
        "memos": "utilization_failure",
    }
    result_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for condition, stage in conditions.items():
        memory_id = f"memory-{condition}"
        has_stored = stage != "storage_failure"
        is_retrieved = stage not in {"storage_failure", "retrieval_failure"}
        is_visible = stage == "utilization_failure"
        attribution = (
            {
                memory_id: {
                    "contributes_positive_coverage": True,
                    "method": "exact_signature",
                    "provenance_mode": "native/exact",
                    "state_ids": [required_state],
                }
            }
            if has_stored
            else {}
        )
        inventory_rows.append(
            {
                "episode_id": episode_id,
                "condition": condition,
                "checkpoint_session": sceu.checkpoint_session,
                "items": ([{"memory_id": memory_id}] if has_stored else []),
                "evaluator_attribution_by_memory": attribution,
                "n_live": int(has_stored),
                "n_write": 1,
            }
        )
        event_rows.append(
            {
                "episode_id": episode_id,
                "condition": condition,
                "session_index": 0,
                "provenance_mode": "native/exact",
            }
        )
        result_rows.append(
            {
                "episode_id": episode_id,
                "sceu_id": sceu.sceu_id,
                "opportunity_id": sceu.opportunity_id,
                "checkpoint_session": sceu.checkpoint_session,
                "policy_profile_id": "policy",
                "condition": condition,
                "readout": "common_rerank",
                "result_id": f"result-{condition}",
                "selected_action_id": "same-wrong-action",
                "current_state_signature": "same-state",
                "condition_status": "complete",
                "baseline_stable": True,
                "backend_retrieved_memory_ids": ([memory_id] if is_retrieved else []),
                "model_visible_memory_ids": ([memory_id] if is_visible else []),
                "behaviorally_used_memory_ids": [],
                "interventions": (
                    [
                        {
                            "intervention_kind": "neutral_replacement",
                            "target_memory_id": memory_id,
                        }
                    ]
                    if is_visible
                    else []
                ),
                "behavior": {
                    "action_id": "same-wrong-action",
                    "behavior_score": 0.0,
                    "is_correct": False,
                },
                "normalized_drift_flags": [],
            }
        )
    _json(
        report / "summary.json",
        {
            "schema_version": 2,
            "run_identity": "run-legacy",
            "evaluated_episode_ids": [episode_id],
            "storage_provenance": {
                "status": "complete",
                "incomplete_write_checkpoints": [],
                "incomplete_write_tasks": [],
            },
        },
    )
    _json(report / "validation.json", {"ok": True})
    _json(report / "measurement_gates.json", {"measurement_ready": False, "gates": []})
    _jsonl(report / "task_results.jsonl", [{}])
    _jsonl(report / "sceu_results.jsonl", result_rows)
    _jsonl(report / "memory_events.jsonl", event_rows)
    _jsonl(report / "memory_inventory.jsonl", inventory_rows)
    _jsonl(report / "retrieval_trace.jsonl", [{}])
    _jsonl(report / "interventions.jsonl", [{}])
    artifact_hashes = {
        path.relative_to(report).as_posix(): _digest(path)
        for path in sorted(item for item in report.rglob("*") if item.is_file())
    }
    _json(
        report / "run_manifest.json",
        {
            "schema_version": 2,
            "run_identity": "run-legacy",
            "dataset_release": "test-release",
            "dataset_manifest_sha256": _digest(dataset / "MANIFEST.json"),
            "artifact_hashes": artifact_hashes,
        },
    )
    return report, dataset


def test_reanalysis_localizes_four_failure_stages_at_same_decision(tmp_path: Path) -> None:
    report, dataset = _fixture(tmp_path)

    payload = reanalyze_completed_report(report, dataset)

    assert payload["analysis_timing"] == "post_hoc_exploratory"
    assert payload["schema_version"] == 3
    assert payload["claim_boundary"]["new_model_or_memory_calls"] == 0
    assert (
        payload["claim_boundary"]["C2_post_hoc_longitudinal_description_available"]
        is False
    )
    assert (
        payload["claim_boundary"][
            "C2_post_hoc_control_clean_category_description_available"
        ]
        is False
    )
    assert payload["failure_stage_counts"] == {
        "exposure_failure": 1,
        "retrieval_failure": 1,
        "storage_failure": 1,
        "utilization_failure": 1,
    }
    rows = {row["condition"]: row for row in payload["decision_attribution_rows"]}
    assert rows["mem0"]["stage"] == "storage_failure"
    assert rows["flat_retrieval"]["stage"] == "retrieval_failure"
    assert rows["amem"]["stage"] == "exposure_failure"
    assert rows["memos"]["decision_layer_diagnosis"] == (
        "visible_without_detected_unique_causal_effect"
    )
    divergence = payload["fault_profile_divergence"]
    assert divergence["n_same_incorrect_action_pairs"] == 6
    assert divergence["n_same_incorrect_action_fault_profile_divergences"] == 6


def test_writer_is_hashed_and_keeps_source_immutable(tmp_path: Path) -> None:
    report, dataset = _fixture(tmp_path)
    before = _source_bytes(report)
    output = tmp_path / "posthoc-reanalysis"

    written = write_completed_report_reanalysis(report, dataset, output)

    assert written == output.resolve()
    assert before == _source_bytes(report)
    manifest = json.loads((output / "reanalysis_manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        assert _digest(output / name) == expected
    assert (output / "decision_attribution.jsonl").is_file()
    assert (output / "failure_attribution_scorecard.csv").is_file()
    assert (output / "drift_trajectories.json").is_file()
    assert (output / "drift_trajectories.md").is_file()

    second = tmp_path / "posthoc-reanalysis-repeat"
    write_completed_report_reanalysis(report, dataset, second)
    second_manifest = json.loads((second / "reanalysis_manifest.json").read_text())
    assert second_manifest["artifact_hashes"] == manifest["artifact_hashes"]


def test_reanalysis_rejects_mismatched_dataset_and_source_tree_output(
    tmp_path: Path,
) -> None:
    report, dataset = _fixture(tmp_path)
    (dataset / "MANIFEST.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CompletedReportReanalysisError, match="do not support"):
        reanalyze_completed_report(report, dataset)
    with pytest.raises(CompletedReportReanalysisError, match="outside the source"):
        write_completed_report_reanalysis(report, dataset, report / "posthoc")


def test_cli_dispatches_zero_api_reanalysis(tmp_path: Path) -> None:
    report, dataset = _fixture(tmp_path)
    output = tmp_path / "cli-output"

    code = main(
        [
            "reanalyze-completed-report",
            "--report",
            str(report),
            "--dataset",
            str(dataset),
            "--out",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "reanalysis_summary.json").is_file()
