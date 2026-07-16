from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lhmsb.experiments import vertical_runner
from lhmsb.experiments.vertical import main
from lhmsb.experiments.vertical_config import GitSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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


def test_module_entrypoint_runs_resumes_and_preserves_frozen_dataset(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    command = [
        sys.executable,
        "-m",
        "lhmsb.experiments.vertical",
        "run",
        "--dataset",
        str(frozen_vertical),
        "--config",
        str(offline_config),
        "--out",
        str(run_dir),
        "--allow-dirty",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    frozen_hashes = _tree_hashes(frozen_vertical)

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    aggregates = {
        name: (run_dir / name).read_bytes()
        for name in ("task_results.jsonl", "sceu_results.jsonl", "summary.json")
    }

    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert aggregates == {name: (run_dir / name).read_bytes() for name in aggregates}
    assert _tree_hashes(frozen_vertical) == frozen_hashes
