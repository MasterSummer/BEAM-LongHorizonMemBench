from __future__ import annotations

from pathlib import Path

import pytest

from lhmsb.datasets.stateful_loader import load_software_vertical_specs
from lhmsb.experiments.vertical_config import (
    VerticalExperimentError,
    VerticalTask,
    build_vertical_tasks,
    load_vertical_offline_config,
)


def test_default_matrix_has_six_ordered_tasks(
    frozen_vertical: Path,
    offline_config: Path,
) -> None:
    config = load_vertical_offline_config(offline_config)
    specs = load_software_vertical_specs(frozen_vertical)

    tasks = build_vertical_tasks(specs, config, run_identity="r" * 64)

    assert [(task.condition, task.intervention_state_id) for task in tasks] == [
        ("workspace_only", None),
        ("oracle_current_state", None),
        ("fake_native", None),
        ("fake_native", "P2"),
        ("fake_native", "C1"),
        ("fake_native", "U1"),
    ]
    assert [task.task_index for task in tasks] == list(range(6))
    assert len({task.task_id for task in tasks}) == 6
    assert all(task.run_identity == "r" * 64 for task in tasks)
    assert all(len(task.task_payload_hash) == 64 for task in tasks)


def test_config_hash_uses_normalized_content(tmp_path: Path) -> None:
    compact = tmp_path / "compact.yaml"
    expanded = tmp_path / "expanded.yaml"
    compact.write_text(
        "schema_version: 1\n"
        "experiment_id: pilot\n"
        "conditions: {workspace_only: [null], fake_native: [null, P2]}\n",
        encoding="utf-8",
    )
    expanded.write_text(
        "schema_version: 1\n"
        "experiment_id: pilot\n"
        "conditions:\n"
        "  workspace_only:\n"
        "    - null\n"
        "  fake_native:\n"
        "    - null\n"
        "    - P2\n",
        encoding="utf-8",
    )

    first = load_vertical_offline_config(compact)
    second = load_vertical_offline_config(expanded)

    assert first == second
    assert first.config_hash == second.config_hash


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            "schema_version: 2\nexperiment_id: x\nconditions:\n  fake_native: [null]\n",
            "schema version",
        ),
        (
            "schema_version: 1\nexperiment_id: ''\nconditions:\n  fake_native: [null]\n",
            "experiment_id",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n  unknown: [null]\n",
            "condition",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n  fake_native: []\n",
            "empty",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n  fake_native: [P2, P2]\n",
            "duplicate intervention",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n  workspace_only: [P2]\n",
            "fake_native",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n  fake_native: [3]\n",
            "string or null",
        ),
        (
            "schema_version: 1\nexperiment_id: x\nconditions:\n"
            "  fake_native: [null]\n  fake_native: [P2]\n",
            "duplicate key",
        ),
    ],
)
def test_config_rejects_invalid_matrix(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(VerticalExperimentError, match=message):
        load_vertical_offline_config(path)


def test_task_round_trip_preserves_payload_identity(
    frozen_vertical: Path,
    offline_config: Path,
) -> None:
    config = load_vertical_offline_config(offline_config)
    task = build_vertical_tasks(
        load_software_vertical_specs(frozen_vertical),
        config,
        run_identity="a" * 64,
    )[3]

    restored = VerticalTask.from_dict(task.to_dict())

    assert restored == task
    assert restored.task_id.endswith("-fake-native-P2")


def test_task_from_dict_rejects_unknown_condition() -> None:
    with pytest.raises(VerticalExperimentError, match="condition"):
        VerticalTask.from_dict(
            {
                "task_index": 0,
                "task_id": "task",
                "episode_id": "episode",
                "condition": "unknown",
                "intervention_state_id": None,
                "run_identity": "r",
                "task_payload_hash": "h",
            }
        )
