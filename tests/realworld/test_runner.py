from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lhmsb.realworld.runner import CommandExecutor, run_experiment
from lhmsb.realworld.schema import ConditionSpec, load_experiment


def _repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "value.txt").write_text("old\n", encoding="utf-8")
    (path / "test_hidden.py").write_text(
        "from pathlib import Path\n"
        "def test_value():\n"
        "    assert Path('value.txt').read_text() == 'new\\n'\n",
        encoding="utf-8",
    )
    (path / "test_regression.py").write_text(
        "def test_regression():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)
    return path


def _experiment_file(path: Path, repository: Path) -> Path:
    config = path / "experiment.yaml"
    config.write_text(
        """experiment_id: smoke
repeats: 1
timeout_seconds: 10
conditions:
  - condition_id: memory-off
    memory_mode: none
    workspace_files_visible: true
  - condition_id: memory-on
    memory_mode: explicit
    workspace_files_visible: false
tasks:
  - task_id: edit-value
    repository: REPOSITORY
    revision: HEAD
    sessions:
      - session_id: implement
        prompt: Change value.txt to new.
      - session_id: verify
        prompt: Check the change and finish.
    hidden_test_commands:
      - python -m pytest -q test_hidden.py
    regression_test_commands:
      - python -m pytest -q test_regression.py
""".replace("REPOSITORY", str(repository)),
        encoding="utf-8",
    )
    return config


class EditingExecutor:
    def __init__(self) -> None:
        self.workspaces: dict[str, Path] = {}

    def run_session(self, *, workspace, prompt, session_id, condition, transcript_path):
        if session_id == "implement":
            assert (workspace / "value.txt").read_text(encoding="utf-8") == "old\n"
            (workspace / "value.txt").write_text("new\n", encoding="utf-8")
        else:
            assert (workspace / "value.txt").read_text(encoding="utf-8") == "new\n"
        self.workspaces[condition.condition_id] = workspace
        transcript_path.write_text(json.dumps({"prompt": prompt}), encoding="utf-8")
        return {"success": True}


def test_runs_paired_conditions_from_same_revision_and_scores_real_tests(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    spec = load_experiment(_experiment_file(tmp_path, repository))
    executor = EditingExecutor()

    results = run_experiment(spec, output=tmp_path / "out", executor=executor)

    assert len(results) == 2
    assert all(result["task_success"] is True for result in results)
    assert all(result["regression_pass"] is True for result in results)
    assert executor.workspaces["memory-off"] != executor.workspaces["memory-on"]
    assert all((path / "value.txt").read_text() == "new\n" for path in executor.workspaces.values())
    assert len((tmp_path / "out" / "results.jsonl").read_text().splitlines()) == 2
    for result in results:
        unit = tmp_path / "out" / str(result["unit_id"])
        assert (unit / "changes.patch").is_file()
        assert (unit / "implement.json").is_file()
        assert (unit / "hidden-test-1.stdout.txt").is_file()


def test_nonzero_hidden_test_is_failure_and_existing_run_is_not_overwritten(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    config = _experiment_file(tmp_path, repository)
    config.write_text(
        config.read_text().replace(
            "python -m pytest -q test_hidden.py", "python -c 'raise SystemExit(3)'"
        )
    )
    spec = load_experiment(config)
    output = tmp_path / "out"
    results = run_experiment(spec, output=output, executor=EditingExecutor())
    assert all(result["task_success"] is False for result in results)
    with pytest.raises(FileExistsError):
        run_experiment(spec, output=output, executor=EditingExecutor())


def test_command_executor_passes_prompt_and_condition_without_shell(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcript = tmp_path / "turn.json"
    executor = CommandExecutor(
        'python -c "import os, pathlib; '
        "pathlib.Path('seen.txt').write_text("
        "pathlib.Path(os.environ['LHMSB_PROMPT_FILE']).read_text())\""
    )
    result = executor.run_session(
        workspace=workspace,
        prompt="Implement the request",
        session_id="s1",
        condition=ConditionSpec("c1", "none", True),
        transcript_path=transcript,
    )
    assert result["success"] is True
    assert (workspace / "seen.txt").read_text() == "Implement the request"
    assert json.loads(transcript.read_text())["returncode"] == 0
