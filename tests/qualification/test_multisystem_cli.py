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


def _matched_dataset(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (101, 102),
    sessions: int = 4,
) -> Path:
    stage = tmp_path / "matched-stage"
    frozen = tmp_path / "matched-dataset"
    generate_mem0_stateful_to_staging(
        stage,
        seeds=seeds,
        n_episodes=1,
        n_sessions=sessions,
        construct_mode="matched_triplets",
        steps_per_session=2,
    )
    freeze_mem0_stateful(stage, frozen)
    return frozen


def _horizon_dataset(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (42,),
) -> Path:
    stage = tmp_path / "horizon-stage"
    frozen = tmp_path / "horizon-dataset"
    generate_mem0_stateful_to_staging(
        stage,
        seeds=seeds,
        n_episodes=1,
        n_sessions=16,
        construct_mode="horizon_panels",
        steps_per_session=16,
        horizon_sessions=(4, 8, 16),
    )
    freeze_mem0_stateful(stage, frozen)
    return frozen


def _longitudinal_dataset(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (42,),
) -> Path:
    stage = tmp_path / "longitudinal-stage"
    frozen = tmp_path / "longitudinal-dataset"
    generate_mem0_stateful_to_staging(
        stage,
        seeds=seeds,
        n_episodes=1,
        n_sessions=16,
        construct_mode="longitudinal_trajectories",
        steps_per_session=16,
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
    assert manifest["analysis_phase"] == "development"
    assert manifest["analysis_timing"] == "pre_specified"
    assert manifest["preparation_task_count"] == 4
    assert manifest["evaluation_template_count"] == 21
    assert manifest["experiment_design_audit_status"] == "diagnostic_only"
    design_audit = json.loads(
        (run / "experiment_design_audit.json").read_text(encoding="utf-8")
    )
    assert design_audit["run_ready"] is True
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
    python_locks = "d" * 64
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
            "LHMSB_PYTHON_LOCK_MANIFEST_HASH": python_locks,
        },
    )
    assert payload["runtime_manifest_hash"] == runtime
    assert payload["source_tree_manifest_hash"] == sources
    assert payload["model_bundle_hash"] == models
    assert payload["python_lock_manifest_hash"] == python_locks
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


def test_matched_plan_records_counterfactual_group_as_analysis_unit(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    dataset = _matched_dataset(tmp_path)
    payload = plan_systems_run(
        dataset,
        Path(
            "configs/experiments/systems_controlled_gpt_only_matched_v011.yaml"
        ),
        tmp_path / "matched-run",
        allow_dirty=True,
        n_sessions=4,
        episode_limit=3,
    )

    assert payload["construct_mode"] == "matched_triplets"
    assert payload["primary_analysis_unit"] == "counterfactual_group"
    assert payload["physical_episode_count"] == 3
    assert payload["n_statistical_units"] == 1
    assert payload["dataset_physical_episode_count"] == 6
    assert payload["dataset_statistical_unit_count"] == 2
    assert payload["counterfactual_group_ids"] == ["software-cf-101-101"]
    assert payload["preparation_task_count"] == 12
    assert payload["evaluation_template_count"] == 21
    assert payload["experiment_design_audit_status"] == "diagnostic_only"
    assert payload["balanced_mechanism_design_ready"] is False
    assert payload["analysis_phase"] == "development"


def test_analysis_phase_thresholds_use_statistical_not_physical_units() -> None:
    from lhmsb.qualification.multisystem_cli import (
        MultisystemCliError,
        _validate_analysis_phase,
    )

    ready_audit = {"balanced_mechanism_design_ready": True}
    unbalanced_audit = {"balanced_mechanism_design_ready": False}
    matched_three = {
        "construct_mode": "matched_triplets",
        "n_statistical_units": 3,
        "physical_episode_count": 9,
    }
    matched_thirty = {
        "construct_mode": "matched_triplets",
        "n_statistical_units": 30,
        "physical_episode_count": 90,
    }
    standard_five = {
        "construct_mode": "standard",
        "n_statistical_units": 5,
        "physical_episode_count": 5,
    }
    standard_fifty = {
        "construct_mode": "standard",
        "n_statistical_units": 50,
        "physical_episode_count": 50,
    }

    _validate_analysis_phase(
        "calibration",
        dataset_design=matched_three,
        design_audit=ready_audit,
    )
    _validate_analysis_phase(
        "confirmatory",
        dataset_design=matched_thirty,
        design_audit=ready_audit,
    )
    _validate_analysis_phase(
        "calibration",
        dataset_design=standard_five,
        design_audit=ready_audit,
    )
    _validate_analysis_phase(
        "confirmatory",
        dataset_design=standard_fifty,
        design_audit=ready_audit,
    )

    with pytest.raises(MultisystemCliError, match="requires at least 30"):
        _validate_analysis_phase(
            "confirmatory",
            dataset_design=matched_three,
            design_audit=ready_audit,
        )
    with pytest.raises(MultisystemCliError, match="balanced counterfactual"):
        _validate_analysis_phase(
            "calibration",
            dataset_design=matched_three,
            design_audit=unbalanced_audit,
        )
    with pytest.raises(MultisystemCliError, match="requires at least 50"):
        _validate_analysis_phase(
            "confirmatory",
            dataset_design={**standard_fifty, "n_statistical_units": 49},
            design_audit=ready_audit,
        )


def test_analysis_phase_is_bound_into_run_identity(tmp_path: Path) -> None:
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    dataset = _dataset(tmp_path)
    development = plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        tmp_path / "development-run",
        allow_dirty=True,
        n_sessions=4,
        analysis_phase="development",
    )
    diagnostic = plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        tmp_path / "diagnostic-run",
        allow_dirty=True,
        n_sessions=4,
        analysis_phase="diagnostic",
    )

    assert development["analysis_phase"] == "development"
    assert diagnostic["analysis_phase"] == "diagnostic"
    assert development["run_identity"] != diagnostic["run_identity"]


def test_matched_plan_rejects_an_episode_limit_that_splits_a_triplet(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification.multisystem_cli import (
        MultisystemCliError,
        plan_systems_run,
    )

    dataset = _matched_dataset(tmp_path, seeds=(101,))
    with pytest.raises(MultisystemCliError, match="splits a matched"):
        plan_systems_run(
            dataset,
            Path(
                "configs/experiments/systems_controlled_gpt_only_matched_v011.yaml"
            ),
            tmp_path / "bad-matched-run",
            allow_dirty=True,
            n_sessions=4,
            episode_limit=1,
        )


def test_horizon_plan_counts_one_complete_panel_as_one_analysis_unit(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    payload = plan_systems_run(
        _horizon_dataset(tmp_path),
        Path(
            "configs/experiments/systems_controlled_gpt_only_horizon_v012.yaml"
        ),
        tmp_path / "horizon-run",
        allow_dirty=True,
        n_sessions=16,
        episode_limit=9,
    )

    assert payload["construct_mode"] == "horizon_panels"
    assert payload["primary_analysis_unit"] == "horizon_panel"
    assert payload["physical_episode_count"] == 9
    assert payload["n_statistical_units"] == 1
    assert payload["dataset_physical_episode_count"] == 9
    assert payload["dataset_statistical_unit_count"] == 1
    assert payload["horizon_panel_ids"] == ["software-horizon-panel-42-42"]
    assert payload["preparation_task_count"] == 36
    assert payload["evaluation_template_count"] == 63
    assert payload["experiment_design_audit_status"] == "diagnostic_only"
    assert payload["balanced_mechanism_design_ready"] is False
    assert payload["n_sessions"] == 16


def test_longitudinal_plan_counts_episode_not_repeated_decisions_as_unit(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification.multisystem_cli import plan_systems_run

    run = tmp_path / "longitudinal-run"
    payload = plan_systems_run(
        _longitudinal_dataset(tmp_path),
        Path(
            "configs/experiments/"
            "systems_controlled_gpt_only_longitudinal_v013.yaml"
        ),
        run,
        allow_dirty=True,
        n_sessions=16,
        episode_limit=1,
    )

    assert payload["construct_mode"] == "longitudinal_trajectories"
    assert payload["primary_analysis_unit"] == "episode"
    assert payload["physical_episode_count"] == 1
    assert payload["n_statistical_units"] == 1
    assert payload["dataset_physical_episode_count"] == 1
    assert payload["dataset_statistical_unit_count"] == 1
    assert payload["counterfactual_group_ids"] == []
    assert payload["horizon_panel_ids"] == []
    assert payload["preparation_task_count"] == 4
    assert payload["evaluation_template_count"] == 7
    design = json.loads(
        (run / "experiment_design_audit.json").read_text(encoding="utf-8")
    )
    assert design["scope"] == "longitudinal_trajectory"
    assert design["run_ready"] is True
    statuses = {
        row["check_id"]: row["status"] for row in design["checks"]
    }
    assert statuses["c2_longitudinal_lineage_design"] == "pass"
    assert statuses["c2_longitudinal_recovery_design"] == "pass"
    assert statuses["c3_intervention_target_contract"] == "pass"
    assert statuses["long_horizon_effective_step_span"] == "pass"
    assert statuses["task_step_anti_padding_integrity"] == "pass"


@pytest.mark.parametrize("episode_limit", (1, 3, 6, 8))
def test_horizon_plan_rejects_an_episode_limit_that_splits_a_panel(
    tmp_path: Path,
    episode_limit: int,
) -> None:
    from lhmsb.qualification.multisystem_cli import (
        MultisystemCliError,
        plan_systems_run,
    )

    with pytest.raises(MultisystemCliError, match="splits a horizon panel"):
        plan_systems_run(
            _horizon_dataset(tmp_path),
            Path(
                "configs/experiments/systems_controlled_gpt_only_horizon_v012.yaml"
            ),
            tmp_path / f"bad-horizon-{episode_limit}",
            allow_dirty=True,
            n_sessions=16,
            episode_limit=episode_limit,
        )


def test_plan_rejects_dataset_release_mismatch_before_api_work(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification.multisystem_cli import (
        MultisystemCliError,
        plan_systems_run,
    )

    with pytest.raises(MultisystemCliError, match="dataset release"):
        plan_systems_run(
            _dataset(tmp_path),
            Path(
                "configs/experiments/systems_controlled_gpt_only_matched_v011.yaml"
            ),
            tmp_path / "wrong-release-run",
            allow_dirty=True,
            n_sessions=4,
        )


def test_workers_reject_dataset_manifest_changes_after_planning(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification import multisystem_cli

    dataset = _dataset(tmp_path)
    run = tmp_path / "immutable-run"
    multisystem_cli.plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        run,
        allow_dirty=True,
        n_sessions=4,
    )
    manifest_path = dataset / "MANIFEST.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="immutable run plan",
    ):
        multisystem_cli._load_contract(run)


def test_workers_recompute_the_policy_free_design_audit(
    tmp_path: Path,
) -> None:
    from lhmsb.qualification import multisystem_cli

    dataset = _dataset(tmp_path)
    run = tmp_path / "design-audit-run"
    multisystem_cli.plan_systems_run(
        dataset,
        Path("configs/experiments/systems_controlled_zen.yaml"),
        run,
        allow_dirty=True,
        n_sessions=4,
    )
    audit_path = run / "experiment_design_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["audit_status"] = "invalid"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="audit differs from the selected frozen dataset",
    ):
        multisystem_cli._load_contract(run)


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
        "python_lock_manifest_hash": "d" * 64,
    }
    environment = {
        "LHMSB_RUNTIME_MANIFEST_HASH": "a" * 64,
        "LHMSB_SOURCE_TREE_MANIFEST_HASH": "b" * 64,
        "LHMSB_MODEL_BUNDLE_HASH": "c" * 64,
        "LHMSB_PYTHON_LOCK_MANIFEST_HASH": "d" * 64,
    }
    assert multisystem_cli._assert_planned_preparation_manifests(
        manifest,
        environment,
    ) == ("a" * 64, "b" * 64, "c" * 64, "d" * 64)

    environment["LHMSB_SOURCE_TREE_MANIFEST_HASH"] = "d" * 64
    with pytest.raises(
        multisystem_cli.MultisystemCliError,
        match="source-tree manifest identity differs",
    ):
        multisystem_cli._assert_planned_preparation_manifests(
            manifest,
            environment,
        )
