from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_RELEASE = ROOT / "datasets" / "releases" / "software-vertical-v0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_mem0_extra_is_exactly_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["optional-dependencies"]["mem0"] == ["mem0ai==2.0.12"]


def test_legacy_release_archive_matches_declared_sha() -> None:
    release = json.loads((LEGACY_RELEASE / "RELEASE.json").read_text(encoding="utf-8"))
    archive = LEGACY_RELEASE / release["dataset_archive"]
    assert _sha256(archive) == release["dataset_archive_sha256"]
    assert (
        LEGACY_RELEASE / f"{release['dataset_archive']}.sha256"
    ).read_text(encoding="utf-8").strip() == (
        f"{release['dataset_archive_sha256']}  {release['dataset_archive']}"
    )


def test_mem0_lock_records_package_source_and_wheel_hash() -> None:
    lock = (ROOT / "constraints" / "mem0.lock.txt").read_text(encoding="utf-8")
    assert "mem0ai==2.0.12" in lock
    assert "42cf18c4e6adb448e981aa1c7b55c1602b0cb670" in lock
    assert "6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2" in lock
