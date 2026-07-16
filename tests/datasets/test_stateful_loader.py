from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.datasets.stateful_loader import load_software_vertical_specs
from lhmsb.datasets.stateful_pipeline import (
    StatefulDatasetError,
    freeze_stateful,
    generate_stateful_to_staging,
)
from lhmsb.families.software.vertical import SoftwareVerticalFamily
from lhmsb.longhorizon.replay import plan_hash


def _freeze(tmp_path: Path) -> Path:
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


def _episode_record(frozen: Path) -> dict[str, object]:
    line = (frozen / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    value = json.loads(line)
    assert isinstance(value, dict)
    return value


def test_load_frozen_spec_round_trips_record(tmp_path: Path) -> None:
    frozen = _freeze(tmp_path)
    record = _episode_record(frozen)

    specs = load_software_vertical_specs(frozen)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.plan.episode_id == record["episode_id"]
    assert spec.plan.semantic_seed == record["semantic_seed"]
    assert spec.plan.trajectory_seed == record["trajectory_seed"]
    assert spec.plan.n_sessions == record["n_sessions"]
    assert plan_hash(spec.plan) == record["plan_hash"]
    assert spec.surface_hash == record["surface_hash"]
    assert {action.action_id for action in spec.actions} == {
        "safe_v2_offline",
        "stale_v1",
        "cloud_shortcut",
    }


def test_load_frozen_spec_does_not_regenerate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _freeze(tmp_path)

    def fail_generate(*args: object, **kwargs: object) -> object:
        raise AssertionError("frozen loading must not call the generator")

    monkeypatch.setattr(SoftwareVerticalFamily, "generate", fail_generate)

    specs = load_software_vertical_specs(frozen)

    assert specs[0].plan.episode_id == "software-42"


def test_load_rejects_tampered_frozen_file(tmp_path: Path) -> None:
    frozen = _freeze(tmp_path)
    episodes = frozen / "episodes.jsonl"
    episodes.write_text(episodes.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(StatefulDatasetError, match="checksum"):
        load_software_vertical_specs(frozen)


def test_load_rejects_record_hash_drift_without_checksum_verification(tmp_path: Path) -> None:
    frozen = _freeze(tmp_path)
    record = _episode_record(frozen)
    record["plan_hash"] = "0" * 64
    (frozen / "episodes.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StatefulDatasetError, match="plan hash"):
        load_software_vertical_specs(frozen, verify=False)


def test_load_rejects_manifest_episode_hash_drift(tmp_path: Path) -> None:
    frozen = _freeze(tmp_path)
    manifest_path = frozen / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][0]["surface_hash"] = "f" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StatefulDatasetError, match="manifest surface hash"):
        load_software_vertical_specs(frozen)


def test_load_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    frozen = _freeze(tmp_path)
    manifest_path = frozen / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StatefulDatasetError, match="schema version"):
        load_software_vertical_specs(frozen)
