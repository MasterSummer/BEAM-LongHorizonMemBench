from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.experiments import vertical_runner
from lhmsb.experiments.vertical import main
from lhmsb.experiments.vertical_config import GitSnapshot


def test_cli_plan_run_task_and_partial_aggregate(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"

    assert (
        main(
            [
                "plan",
                "--dataset",
                str(frozen_vertical),
                "--config",
                str(offline_config),
                "--out",
                str(run_dir),
                "--allow-dirty",
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
        == 0
    )
    assert main(["aggregate", "--run-dir", str(run_dir)]) == 1

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_tasks"] == 1
    assert summary["missing_tasks"] == 5


def test_cli_run_completes_and_resumes_default_matrix(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    arguments = [
        "run",
        "--dataset",
        str(frozen_vertical),
        "--config",
        str(offline_config),
        "--out",
        str(run_dir),
        "--allow-dirty",
    ]

    assert main(arguments) == 0
    result_paths = sorted((run_dir / "tasks").glob("*/result.json"))
    mtimes = {path: path.stat().st_mtime_ns for path in result_paths}
    aggregates = {
        name: (run_dir / name).read_bytes()
        for name in ("task_results.jsonl", "sceu_results.jsonl", "summary.json")
    }

    assert len(result_paths) == 6
    assert json.loads((run_dir / "summary.json").read_text())["complete"] is True
    assert main(arguments) == 0
    assert {path: path.stat().st_mtime_ns for path in result_paths} == mtimes
    assert aggregates == {name: (run_dir / name).read_bytes() for name in aggregates}


def test_cli_reports_contract_errors_without_traceback(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "pilot"
    monkeypatch.setattr(
        vertical_runner,
        "current_git_snapshot",
        lambda root=None: GitSnapshot(commit="a" * 40, dirty=True, ref="feature"),
    )
    status = main(
        [
            "plan",
            "--dataset",
            str(frozen_vertical),
            "--config",
            str(offline_config),
            "--out",
            str(run_dir),
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "dirty" in captured.err
    assert "Traceback" not in captured.err


def test_cli_force_replaces_successful_atomic_task(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(frozen_vertical),
                "--config",
                str(offline_config),
                "--out",
                str(run_dir),
                "--allow-dirty",
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
        == 0
    )
    result = next((run_dir / "tasks").glob("*/result.json"))
    first_mtime = result.stat().st_mtime_ns

    assert (
        main(
            [
                "run-task",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
                "--force",
            ]
        )
        == 0
    )
    assert result.stat().st_mtime_ns >= first_mtime
