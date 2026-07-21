from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lhmsb.qualification.source_manifest import (
    SourceManifestError,
    SourceSpec,
    create_source_manifest,
    verified_source_commit_for_module,
    verify_source_manifest,
)


def _repository(root: Path, name: str) -> tuple[Path, str, str]:
    path = root / "sources" / name
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test Runner"],
        check=True,
    )
    (path / "package.py").write_text(f"NAME = {name!r}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "package.py"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True
    )
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    url = f"https://example.com/{name}"
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", url + ".git"],
        check=True,
    )
    return path, commit, url


def _specs(data_root: Path) -> tuple[dict[str, SourceSpec], dict[str, Path]]:
    specs: dict[str, SourceSpec] = {}
    roots: dict[str, Path] = {}
    for name in ("amem", "memos"):
        root, commit, url = _repository(data_root, name)
        roots[name] = root
        specs[name] = SourceSpec(
            relative_root=f"sources/{name}",
            source_url=url,
            source_commit=commit,
        )
    return specs, roots


def test_source_manifest_binds_git_identity_and_complete_file_tree(
    tmp_path: Path,
) -> None:
    specs, roots = _specs(tmp_path)
    manifest = tmp_path / "manifests" / "system-sources.json"

    created = create_source_manifest(tmp_path, manifest, specs=specs)
    verified = verify_source_manifest(tmp_path, manifest, specs=specs)

    assert verified == created
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["sources"]) == {"amem", "memos"}
    assert payload["sources"]["amem"]["file_count"] == 1
    assert len(payload["sources"]["memos"]["source_tree_sha256"]) == 64

    (roots["amem"] / "untracked_runtime.py").write_text("VALUE = 1\n")
    with pytest.raises(SourceManifestError, match="untracked source files"):
        verify_source_manifest(tmp_path, manifest, specs=specs)


def test_source_manifest_cannot_bless_untracked_runtime_code(tmp_path: Path) -> None:
    specs, roots = _specs(tmp_path)
    (roots["amem"] / "untracked_runtime.py").write_text("VALUE = 1\n")

    with pytest.raises(SourceManifestError, match="untracked source files"):
        create_source_manifest(tmp_path, specs=specs)


def test_source_manifest_ignores_cache_sidecars_but_rejects_tracked_edits(
    tmp_path: Path,
) -> None:
    specs, roots = _specs(tmp_path)
    manifest = tmp_path / "manifests" / "system-sources.json"
    create_source_manifest(tmp_path, manifest, specs=specs)

    cache = roots["amem"] / "__pycache__"
    cache.mkdir()
    (cache / "package.cpython-311.pyc").write_bytes(b"cache")
    verify_source_manifest(tmp_path, manifest, specs=specs)

    (roots["amem"] / "package.py").write_text("NAME = 'changed'\n", encoding="utf-8")
    with pytest.raises(SourceManifestError, match="tracked source files are dirty"):
        verify_source_manifest(tmp_path, manifest, specs=specs)


def test_source_manifest_rejects_exported_tree_without_git_identity(
    tmp_path: Path,
) -> None:
    specs, roots = _specs(tmp_path)
    manifest = tmp_path / "manifests" / "system-sources.json"
    create_source_manifest(tmp_path, manifest, specs=specs)
    shutil.rmtree(roots["memos"] / ".git")

    with pytest.raises(SourceManifestError, match="cannot determine Git identity"):
        verify_source_manifest(tmp_path, manifest, specs=specs)


def test_verified_module_must_be_imported_from_the_manifest_tree(tmp_path: Path) -> None:
    specs, roots = _specs(tmp_path)
    manifest = tmp_path / "manifests" / "system-sources.json"
    create_source_manifest(tmp_path, manifest, specs=specs)

    module = SimpleNamespace(__file__=str(roots["amem"] / "package.py"))
    assert (
        verified_source_commit_for_module(
            module,
            "amem",
            data_root=tmp_path,
            manifest_path=manifest,
            specs=specs,
        )
        == specs["amem"].source_commit
    )

    outsider = tmp_path / "outside.py"
    outsider.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(SourceManifestError, match="was not imported"):
        verified_source_commit_for_module(
            SimpleNamespace(__file__=str(outsider)),
            "amem",
            data_root=tmp_path,
            manifest_path=manifest,
            specs=specs,
        )
