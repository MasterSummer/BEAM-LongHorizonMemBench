from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.datasets.mem0_stateful_pipeline import (
    freeze_mem0_stateful,
    generate_mem0_stateful_to_staging,
)
from lhmsb.qualification.cli import main


def _dataset(tmp_path: Path, *, sessions: int = 4) -> Path:
    stage = tmp_path / "stage"
    frozen = tmp_path / "dataset"
    generate_mem0_stateful_to_staging(
        stage,
        seeds=(42,),
        n_episodes=1,
        n_sessions=sessions,
    )
    freeze_mem0_stateful(stage, frozen)
    return frozen


def test_plan_systems_writes_stable_two_stage_contract(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run = tmp_path / "run"
    code = main(
        [
            "plan-systems",
            "--dataset",
            str(dataset),
            "--config",
            "configs/experiments/systems_controlled_zen.yaml",
            "--out",
            str(run),
            "--allow-dirty",
            "--n-sessions",
            "4",
        ]
    )
    assert code == 0
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["preparation_task_count"] == 4
    assert manifest["evaluation_template_count"] == 21
    assert len((run / "prepare_tasks.jsonl").read_text().splitlines()) == 4
    assert len((run / "evaluation_task_templates.jsonl").read_text().splitlines()) == 21
    assert not (run / "tasks.jsonl").exists()


def test_smoke_systems_dry_run_is_network_free(capsys) -> None:
    code = main(["smoke-systems", "--dry-run"])
    assert code == 0
    output = capsys.readouterr().out
    assert "plan-systems" in output
    assert "prepare-task" in output
    assert "run-evaluation-matrix" in output


def test_prepare_task_dry_run_does_not_require_backend(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run = tmp_path / "run"
    assert (
        main(
            [
                "plan-systems",
                "--dataset",
                str(dataset),
                "--config",
                "configs/experiments/systems_controlled_zen.yaml",
                "--out",
                str(run),
                "--allow-dirty",
                "--n-sessions",
                "4",
            ]
        )
        == 0
    )
    assert main(["prepare-task", "--run-dir", str(run), "--task-index", "0", "--dry-run"]) == 0


def test_plan_binds_runtime_source_and_model_bundle_manifests(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run = tmp_path / "run"
    runtime = "a" * 64
    models = "b" * 64
    sources = "c" * 64
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    payload = plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        run,
        allow_dirty=True,
        n_sessions=4,
        environment={
            "LHMSB_RUNTIME_MANIFEST_HASH": runtime,
            "LHMSB_MODEL_BUNDLE_HASH": models,
            "LHMSB_SOURCE_TREE_MANIFEST_HASH": sources,
        },
    )
    assert payload["runtime_manifest_hash"] == runtime
    assert payload["source_tree_manifest_hash"] == sources
    assert payload["model_bundle_hash"] == models
    assert payload["model_files_hash"] == models
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["run_identity"] == payload["run_identity"]


def test_episode_limit_binds_a_distinct_one_episode_smoke_identity(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    dataset = tmp_path / "dataset"
    generate_mem0_stateful_to_staging(
        stage,
        seeds=(42, 43),
        n_sessions=4,
    )
    freeze_mem0_stateful(stage, dataset)
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    smoke = plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        tmp_path / "smoke",
        allow_dirty=True,
        n_sessions=4,
        episode_limit=1,
    )
    full = plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        tmp_path / "full",
        allow_dirty=True,
        n_sessions=4,
    )

    assert smoke["episode_ids"] == ["software-mem0-42"]
    assert smoke["episode_limit"] == 1
    assert smoke["preparation_task_count"] == 4
    assert full["preparation_task_count"] == 8
    assert smoke["run_identity"] != full["run_identity"]


def test_formal_worker_requires_the_planned_clean_commit(monkeypatch) -> None:
    from lhmsb.qualification import multisystem_cli

    manifest = {"code_commit": "planned", "code_dirty": False}
    monkeypatch.setattr(
        multisystem_cli,
        "_git_identity",
        lambda: ("planned", False, "main"),
    )
    assert multisystem_cli._assert_planned_code_identity(manifest) == (
        "planned",
        False,
        "main",
    )

    monkeypatch.setattr(
        multisystem_cli,
        "_git_identity",
        lambda: ("other", False, "main"),
    )
    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="differs from the immutable run plan",
    ):
        multisystem_cli._assert_planned_code_identity(manifest)

    monkeypatch.setattr(
        multisystem_cli,
        "_git_identity",
        lambda: ("planned", True, "main"),
    )
    with pytest.raises(multisystem_cli.MultisystemCliError, match="checkout is dirty"):
        multisystem_cli._assert_planned_code_identity(manifest)


def test_runtime_source_must_match_selected_repository(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from lhmsb.qualification import multisystem_cli

    actual = tmp_path / "actual"
    selected = tmp_path / "selected"
    monkeypatch.setattr(multisystem_cli, "_runtime_source_root", lambda: actual)
    monkeypatch.setenv("LHMSB_REPO_ROOT", str(selected))

    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="runtime lhmsb source differs",
    ):
        multisystem_cli._assert_runtime_source_root()

    monkeypatch.setenv("LHMSB_REPO_ROOT", str(actual))
    assert multisystem_cli._assert_runtime_source_root() == actual


def test_manifest_hash_rejects_non_digest_environment_values(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    from lhmsb.qualification.multisystem_cli import MultisystemCliError, plan_systems_run

    try:
        plan_systems_run(
            dataset,
            Path("configs/experiments/systems_controlled_zen.yaml"),
            tmp_path / "run",
            allow_dirty=True,
            n_sessions=4,
            environment={"LHMSB_RUNTIME_MANIFEST_HASH": "not-a-digest"},
        )
    except MultisystemCliError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("invalid runtime manifest digest was accepted")


def test_preparation_worker_rejects_changed_source_tree_manifest() -> None:
    from lhmsb.qualification import multisystem_cli

    manifest = {
        "runtime_manifest_hash": "a" * 64,
        "source_tree_manifest_hash": "b" * 64,
        "model_bundle_hash": "c" * 64,
    }
    environment = {
        "LHMSB_RUNTIME_MANIFEST_HASH": "a" * 64,
        "LHMSB_SOURCE_TREE_MANIFEST_HASH": "b" * 64,
        "LHMSB_MODEL_BUNDLE_HASH": "c" * 64,
    }
    assert multisystem_cli._assert_planned_preparation_manifests(
        manifest,
        environment,
    ) == ("a" * 64, "b" * 64, "c" * 64)

    environment["LHMSB_SOURCE_TREE_MANIFEST_HASH"] = "d" * 64
    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="source-tree manifest identity differs",
    ):
        multisystem_cli._assert_planned_preparation_manifests(
            manifest,
            environment,
        )
