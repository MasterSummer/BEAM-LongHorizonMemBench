from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.qualification.cli import main
from lhmsb.qualification.report import write_qualification_report
from lhmsb.qualification.runner import (
    ConditionRunResult,
    QualificationMatrixResult,
    QualificationTaskResult,
    SCEURunResult,
)
from lhmsb.qualification.validate import validate_qualification_artifacts

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"
DATASET = ROOT / "runs" / "vertical" / "software_mem0_v2"


def test_module_help_lists_all_qualification_commands() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "lhmsb.qualification", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for command in (
        "plan",
        "run-task",
        "run-matrix",
        "aggregate",
        "validate",
        "preflight",
        "smoke",
    ):
        assert command in completed.stdout


def test_plan_writes_redacted_identity_bound_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-written")
    report_path = tmp_path / "plan.json"
    run_dir = tmp_path / "run"

    status = main(
        [
            "plan",
            "--dataset",
            str(DATASET),
            "--config",
            str(CONFIG),
            "--out",
            str(run_dir),
            "--allow-dirty",
            "--json",
            str(report_path),
        ]
    )

    assert status == 0
    manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    assert "must-not-be-written" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["task_count"] == 12
    assert manifest["required_secret_env"] == [
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ]
    assert len((run_dir / "tasks.jsonl").read_text().splitlines()) == 12
    assert json.loads(report_path.read_text())["run_identity"] == manifest["run_identity"]


def test_run_task_dry_run_needs_no_credentials_but_live_execution_is_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(DATASET),
                "--config",
                str(CONFIG),
                "--out",
                str(run_dir),
                "--allow-dirty",
            ]
        )
        == 0
    )
    monkeypatch.delenv("LHMSB_LIVE_QUALIFICATION", raising=False)

    assert (
        main(
            [
                "run-task",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run-task",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
            ]
        )
        == 2
    )
    assert "LHMSB_LIVE_QUALIFICATION" in capsys.readouterr().err


def test_run_contract_identity_mismatch_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(DATASET),
                "--config",
                str(CONFIG),
                "--out",
                str(run_dir),
                "--allow-dirty",
            ]
        )
        == 0
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_identity"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = main(
        [
            "run-task",
            "--run-dir",
            str(run_dir),
            "--task-index",
            "0",
            "--dry-run",
        ]
    )

    assert status == 2
    assert "identity" in capsys.readouterr().err.casefold()


def _tiny_report(tmp_path: Path) -> Path:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    sceu = spec.plan.sceu_units[0]
    row = SCEURunResult(
        result_id="workspace-result",
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
        behavior=ContinuationOutcome(
            action_id="safe_v2_offline",
            behavior_score=1.0,
            is_correct=True,
        ),
        normalized_drift_flags=(),
        baseline_stable=True,
        baseline_evaluations=(),
        interventions=(),
        retrieval_trace_id=None,
    )
    condition = ConditionRunResult(
        result_id="workspace-result",
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
    )
    out = tmp_path / "report"
    write_qualification_report(
        QualificationMatrixResult("run-identity", (task,)),
        {spec.plan.episode_id: spec},
        out,
    )
    return out


def test_validate_command_checks_hashes_and_trace_ordering(
    tmp_path: Path,
) -> None:
    report = _tiny_report(tmp_path)
    valid = validate_qualification_artifacts(
        report,
        expected_run_identity="run-identity",
    )
    assert valid.ok is True
    assert main(["validate", "--report", str(report)]) == 0

    sceu_path = report / "sceu_results.jsonl"
    row = json.loads(sceu_path.read_text())
    row["candidate_memory_ids"] = ["memory-a"]
    row["retrieved_memory_ids"] = ["memory-b"]
    sceu_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    invalid = validate_qualification_artifacts(report)
    assert invalid.ok is False
    assert any("retrieved" in error for error in invalid.errors)
    assert main(["validate", "--report", str(report)]) == 1
