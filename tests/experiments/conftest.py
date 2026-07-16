from __future__ import annotations

from pathlib import Path

import pytest

from lhmsb.datasets.stateful_pipeline import freeze_stateful, generate_stateful_to_staging


@pytest.fixture
def frozen_vertical(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    frozen = tmp_path / "software_v1"
    generate_stateful_to_staging(
        stage,
        family="software",
        seeds=(42,),
        n_episodes=1,
        n_sessions=4,
    )
    freeze_stateful(stage, frozen)
    return frozen


@pytest.fixture
def offline_config(tmp_path: Path) -> Path:
    path = tmp_path / "vertical.yaml"
    path.write_text(
        "schema_version: 1\n"
        "experiment_id: software-vertical-offline-pilot\n"
        "conditions:\n"
        "  workspace_only: [null]\n"
        "  oracle_current_state: [null]\n"
        "  fake_native: [null, P2, C1, U1]\n",
        encoding="utf-8",
    )
    return path
