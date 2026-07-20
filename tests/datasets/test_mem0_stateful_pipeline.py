from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from lhmsb.datasets.cli import main
from lhmsb.datasets.mem0_stateful_pipeline import (
    MEM0_STATEFUL_GENERATOR_VERSION,
    MEM0_STATEFUL_GENERATOR_VERSION_V3,
    MEM0_STATEFUL_GENERATOR_VERSION_V4,
    MEM0_STATEFUL_RELEASE_ID_V3,
    MEM0_STATEFUL_RELEASE_ID_V4,
    build_mem0_release_archive,
    freeze_mem0_stateful,
    generate_mem0_stateful_to_staging,
    regen_check_mem0_stateful,
    verify_mem0_stateful,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_json_files(root: Path) -> list[Path]:
    return sorted((root / "public").rglob("*.json"))


def test_generate_separates_public_and_evaluator_trees(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42],
        n_episodes=1,
        n_sessions=4,
    )
    assert len(generated) == 1
    assert (stage / "public" / "software-mem0-42" / "sessions").is_dir()
    assert (stage / "public" / "software-mem0-42" / "continuation").is_dir()
    assert (stage / "evaluator" / "episodes.jsonl").is_file()
    assert (stage / "evaluator" / "state_units.jsonl").is_file()
    assert (stage / "evaluator" / "fact_signatures.jsonl").is_file()
    assert (stage / "evaluator" / "continuation_mappings.jsonl").is_file()
    signatures = [
        json.loads(line)
        for line in (stage / "evaluator" / "fact_signatures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {item["state_id"] for item in signatures} == {
        "G0",
        "C1",
        "C2",
        "P1",
        "U1",
        "P2",
        "D1",
        "L1",
        "V2",
    }
    c1 = next(item for item in signatures if item["state_id"] == "C1")
    assert c1["source_sessions"] == [0]
    assert c1["source_event_ids"] == ["e-01-offline"]
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_json_files(stage))
    for forbidden in (
        "source_event_ids",
        "recoverability_by_state",
        "valid_action_ids",
        "option_to_action",
        "safe_v2_offline",
        "stale_v1",
        "cloud_shortcut",
    ):
        assert forbidden not in public_text


def test_freeze_verify_and_regen_are_reproducible(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    manifest = freeze_mem0_stateful(stage, frozen)
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION
    assert manifest.release_id == "software-vertical-mem0-v0.2.0"
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok
    assert manifest.files == json.loads(
        (frozen / "hashes" / "files.json").read_text(encoding="utf-8")
    )


def test_full_horizon_smoke_uses_v03_release_contract(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=16)
    manifest = freeze_mem0_stateful(stage, frozen)

    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V3
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V3


def test_fifty_episode_release_passes_all_audits_and_uses_v04(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=range(50),
        n_sessions=16,
    )
    manifest = freeze_mem0_stateful(stage, frozen)

    assert len(generated) == 50
    assert len({item.plan_hash for item in generated}) == 50
    assert len({item.surface_hash for item in generated}) == 50
    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V4
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V4
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok


def test_verify_detects_public_tampering(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    freeze_mem0_stateful(stage, frozen)
    target = _public_json_files(frozen)[0]
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = verify_mem0_stateful(frozen)
    assert not report.ok
    assert report.mismatches[0][0] == target.relative_to(frozen).as_posix()


def test_release_archive_is_byte_deterministic(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "software_mem0_v2"
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    freeze_mem0_stateful(stage, frozen)
    build_mem0_release_archive(frozen, first)
    build_mem0_release_archive(frozen, second)
    assert _sha256(first) == _sha256(second)
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert names[0] == "software_mem0_v2"
        assert all(
            name == "software_mem0_v2" or name.startswith("software_mem0_v2/")
            for name in names
        )
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_cli_generate_freeze_verify_and_regen(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    assert (
        main(
            [
                "generate-mem0-stateful",
                "--seeds",
                "42",
                "--n-sessions",
                "4",
                "--out",
                str(stage),
            ]
        )
        == 0
    )
    assert main(["freeze-mem0-stateful", "--src", str(stage), "--out", str(frozen)]) == 0
    assert main(["verify-mem0-stateful", "--frozen", str(frozen)]) == 0
    assert main(["regen-check-mem0-stateful", "--frozen", str(frozen)]) == 0
