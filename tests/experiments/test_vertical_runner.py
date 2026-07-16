from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lhmsb.datasets.stateful_pipeline import StatefulDatasetError
from lhmsb.experiments import vertical_runner
from lhmsb.experiments.vertical_config import GitSnapshot, VerticalExperimentError
from lhmsb.experiments.vertical_runner import (
    current_git_snapshot,
    plan_vertical_run,
    read_vertical_tasks,
    run_vertical_task,
)


def test_public_experiment_api_exports_runner_primitives() -> None:
    import lhmsb.experiments as public_api

    assert public_api.VerticalRunManifest
    assert public_api.current_git_snapshot is current_git_snapshot
    assert public_api.plan_vertical_run is plan_vertical_run
    assert public_api.read_vertical_tasks is read_vertical_tasks
    assert public_api.run_vertical_task is run_vertical_task


@pytest.fixture
def planned_run(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> Path:
    run_dir = tmp_path / "run"
    plan_vertical_run(
        frozen_vertical,
        offline_config,
        run_dir,
        allow_dirty=True,
    )
    return run_dir


def _run_git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_current_git_snapshot_detects_commit_ref_and_dirty_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-b", "experiment")
    tracked = repo / "tracked.txt"
    tracked.write_text("sealed\n", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "LHMSB",
        "GIT_AUTHOR_EMAIL": "lhmsb@example.invalid",
        "GIT_COMMITTER_NAME": "LHMSB",
        "GIT_COMMITTER_EMAIL": "lhmsb@example.invalid",
    }
    _run_git(repo, "commit", "-m", "initial", env=env)

    clean = current_git_snapshot(repo)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = current_git_snapshot(repo)

    assert len(clean.commit) == 40
    assert clean.ref == "experiment"
    assert not clean.dirty
    assert dirty.commit == clean.commit
    assert dirty.dirty


def test_plan_is_idempotent_and_binds_dataset_code_and_config(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    first = plan_vertical_run(
        frozen_vertical,
        offline_config,
        run_dir,
        allow_dirty=True,
    )
    manifest_bytes = (run_dir / "run_manifest.json").read_bytes()
    tasks_bytes = (run_dir / "tasks.jsonl").read_bytes()
    config_bytes = (run_dir / "run_config.yaml").read_bytes()
    second = plan_vertical_run(
        frozen_vertical,
        offline_config,
        run_dir,
        allow_dirty=True,
    )

    assert first == second
    assert first.task_count == 6
    assert first.dataset_manifest_sha256
    assert first.config_hash
    assert first.code_commit
    assert first.code_dirty
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_bytes
    assert (run_dir / "tasks.jsonl").read_bytes() == tasks_bytes
    assert (run_dir / "run_config.yaml").read_bytes() == config_bytes


def test_plan_rejects_changed_config_identity(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)
    offline_config.write_text(
        offline_config.read_text(encoding="utf-8").replace("U1]", "U1, G0]"),
        encoding="utf-8",
    )

    with pytest.raises(VerticalExperimentError, match="identity"):
        plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)


def test_force_replans_recognized_run_directory(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    first = plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)
    run_vertical_task(run_dir, 0)
    offline_config.write_text(
        offline_config.read_text(encoding="utf-8").replace(", U1", ""),
        encoding="utf-8",
    )

    second = plan_vertical_run(
        frozen_vertical,
        offline_config,
        run_dir,
        allow_dirty=True,
        force=True,
    )

    assert second.run_identity != first.run_identity
    assert second.task_count == 5
    assert not (run_dir / "tasks").exists()


def test_plan_rejects_unrelated_nonempty_directory_even_with_force(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    user_file = run_dir / "user-data.txt"
    user_file.write_text("preserve me", encoding="utf-8")

    with pytest.raises(VerticalExperimentError, match="unrelated"):
        plan_vertical_run(
            frozen_vertical,
            offline_config,
            run_dir,
            allow_dirty=True,
            force=True,
        )

    assert user_file.read_text(encoding="utf-8") == "preserve me"


def test_plan_requires_explicit_allow_dirty(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vertical_runner,
        "current_git_snapshot",
        lambda root=None: GitSnapshot(commit="a" * 40, dirty=True, ref="feature"),
    )

    with pytest.raises(VerticalExperimentError, match="dirty"):
        plan_vertical_run(frozen_vertical, offline_config, tmp_path / "run")


def test_run_task_is_independent_and_skips_success(planned_run: Path) -> None:
    result_path = run_vertical_task(planned_run, 3)
    first_bytes = result_path.read_bytes()
    first_mtime = result_path.stat().st_mtime_ns
    payload = json.loads(first_bytes)

    assert payload["task"]["intervention_state_id"] == "P2"
    assert payload["result"]["behavior_score"] >= 0
    assert payload["result"]["sceu_results"]
    assert run_vertical_task(planned_run, 3) == result_path
    assert result_path.read_bytes() == first_bytes
    assert result_path.stat().st_mtime_ns == first_mtime


def test_run_task_rejects_invalid_index(planned_run: Path) -> None:
    with pytest.raises(VerticalExperimentError, match="task index"):
        run_vertical_task(planned_run, 99)


def test_run_task_requires_force_for_stale_result(planned_run: Path) -> None:
    result_path = run_vertical_task(planned_run, 0)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["run_identity"] = "stale"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VerticalExperimentError, match="stale"):
        run_vertical_task(planned_run, 0)

    repaired = run_vertical_task(planned_run, 0, force=True)
    assert json.loads(repaired.read_text(encoding="utf-8"))["run_identity"] != "stale"


def test_failed_task_can_retry(
    planned_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = vertical_runner.run_vertical_episode

    def fail_once(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(vertical_runner, "run_vertical_episode", fail_once)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_vertical_task(planned_run, 0)
    task = read_vertical_tasks(planned_run)[0]
    failure_path = planned_run / "tasks" / task.task_id / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error_type"] == "RuntimeError"
    assert "injected failure" in failure["message"]

    monkeypatch.setattr(vertical_runner, "run_vertical_episode", original)
    result_path = run_vertical_task(planned_run, 0)

    assert result_path.is_file()
    assert not failure_path.exists()


def test_run_task_rejects_mutated_frozen_dataset(planned_run: Path) -> None:
    manifest = json.loads((planned_run / "run_manifest.json").read_text(encoding="utf-8"))
    dataset = Path(manifest["dataset_path"])
    episodes = dataset / "episodes.jsonl"
    episodes.write_text(episodes.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(StatefulDatasetError, match="checksum"):
        run_vertical_task(planned_run, 0)


def test_run_task_rejects_code_snapshot_mismatch(
    planned_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads((planned_run / "run_manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        vertical_runner,
        "current_git_snapshot",
        lambda root=None: GitSnapshot(
            commit="f" * 40,
            dirty=bool(manifest["code_dirty"]),
            ref="other",
        ),
    )

    with pytest.raises(VerticalExperimentError, match="code snapshot"):
        run_vertical_task(planned_run, 0)
