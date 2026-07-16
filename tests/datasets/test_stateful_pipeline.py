from __future__ import annotations

import json
from pathlib import Path

from lhmsb.datasets.stateful_pipeline import (
    freeze_stateful,
    generate_stateful_to_staging,
    regen_check_stateful,
    verify_stateful,
)


def test_generate_freeze_verify_and_regen_stateful_dataset(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_stateful_to_staging(
        stage,
        family="software",
        seeds=(42,),
        n_episodes=1,
        n_sessions=4,
    )
    assert len(generated) == 1
    assert (stage / "episodes.jsonl").is_file()
    assert (stage / "surfaces" / generated[0].plan.episode_id / "sessions").is_dir()

    manifest = freeze_stateful(stage, frozen)
    assert manifest.schema_version
    assert manifest.episodes[0]["n_sessions"] == 4
    assert (frozen / "MANIFEST.json").is_file()
    assert (frozen / "evaluator" / "state_events.jsonl").is_file()
    assert (frozen / "hashes" / "files.json").is_file()

    report = verify_stateful(frozen)
    assert report.ok
    regen = regen_check_stateful(frozen)
    assert regen.ok
    assert regen.checked == 1


def test_verify_detects_tampering_and_regen_detects_seed_drift(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_stateful_to_staging(
        stage,
        family="software",
        seeds=(42,),
        n_episodes=1,
        n_sessions=4,
    )
    freeze_stateful(stage, frozen)
    target = frozen / "episodes.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = verify_stateful(frozen)
    assert not report.ok
    assert "episodes.jsonl" in {item[0] for item in report.mismatches}

    # Restore file integrity, then alter the stored trajectory seed.  The
    # manifest's file hash should now fail first; regen still reports the drift
    # when verification is repaired from the modified record.
    target.write_text(target.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    payload = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    payload["trajectory_seed"] = 99
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert not regen_check_stateful(frozen).ok


def test_frozen_manifest_records_surface_and_workspace_hashes(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_stateful_to_staging(
        stage,
        family="software",
        seeds=(42,),
        n_episodes=1,
        n_sessions=16,
    )
    freeze_stateful(stage, frozen)
    manifest = json.loads((frozen / "MANIFEST.json").read_text(encoding="utf-8"))
    episode = manifest["episodes"][0]
    assert episode["plan_hash"]
    assert episode["surface_hash"]
    assert episode["workspace_hash"]
    assert manifest["n_sessions"] == 16
    assert manifest["semantic_seeds"] == [42]
