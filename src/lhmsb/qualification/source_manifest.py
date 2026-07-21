"""Create and verify portable source-tree manifests for native backends.

The experiment configuration pins upstream commits, but a commit string alone
does not prove that the source imported on a worker came from that checkout.
This module binds the A-MEM and MemOS Git identities to a complete, portable
snapshot of every runtime-relevant file outside Git and cache directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class SourceManifestError(RuntimeError):
    """The native source checkout or its manifest is not reproducible."""


@dataclass(frozen=True)
class SourceSpec:
    """One upstream source identity required by the native workflow."""

    relative_root: str
    source_url: str
    source_commit: str


EXPECTED_SOURCES: Mapping[str, SourceSpec] = {
    "amem": SourceSpec(
        relative_root="sources/amem",
        source_url="https://github.com/agiresearch/A-mem",
        source_commit="ceffb860f0712bbae97b184d440df62bc910ca8d",
    ),
    "memos": SourceSpec(
        relative_root="sources/memos",
        source_url="https://github.com/MemTensor/MemOS",
        source_commit="583b07b998afc4debb6c5078439b0b3896f5b097",
    ),
}

_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "__pycache__",
    }
)
_IGNORED_FILE_NAMES = frozenset({".DS_Store"})
_IGNORED_SUFFIXES = (".pyc", ".pyo")


def create_source_manifest(
    data_root: Path,
    output_path: Path | None = None,
    *,
    specs: Mapping[str, SourceSpec] = EXPECTED_SOURCES,
) -> dict[str, object]:
    """Snapshot exact Git checkouts and atomically write their manifest."""
    root = data_root.expanduser().resolve()
    output = output_path or root / "manifests" / "system-sources.json"
    sources: dict[str, object] = {}
    for name, spec in sorted(specs.items()):
        source_root = _resolve_source_root(root, spec.relative_root)
        identity = _git_identity(source_root)
        _verify_expected_identity(name, identity, spec)
        files = _snapshot_files(source_root)
        sources[name] = {
            "root": spec.relative_root,
            "source_url": spec.source_url,
            "source_commit": spec.source_commit,
            "git_tree": identity["git_tree"],
            "file_count": len(files),
            "source_tree_sha256": _canonical_hash(files),
            "files": files,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "sources": sources,
    }
    _atomic_json(output, payload)
    return payload


def verify_source_manifest(
    data_root: Path,
    manifest_path: Path | None = None,
    *,
    specs: Mapping[str, SourceSpec] = EXPECTED_SOURCES,
) -> dict[str, object]:
    """Verify Git identity plus the complete non-cache source file set."""
    root = data_root.expanduser().resolve()
    path = manifest_path or root / "manifests" / "system-sources.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError(f"cannot read source manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SourceManifestError("invalid source manifest schema")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(specs):
        raise SourceManifestError("source manifest backend set does not match the lock")

    for name, spec in sorted(specs.items()):
        record = sources.get(name)
        if not isinstance(record, dict):
            raise SourceManifestError(f"invalid source manifest record: {name}")
        expected_fields = {
            "root": spec.relative_root,
            "source_url": spec.source_url,
            "source_commit": spec.source_commit,
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise SourceManifestError(
                    f"source manifest {name}.{field} does not match the lock"
                )
        source_root = _resolve_source_root(root, spec.relative_root)
        identity = _git_identity(source_root)
        _verify_expected_identity(name, identity, spec)
        if record.get("git_tree") != identity["git_tree"]:
            raise SourceManifestError(f"Git tree changed for {name}")
        expected_files = record.get("files")
        if not isinstance(expected_files, list):
            raise SourceManifestError(f"source file inventory is missing for {name}")
        actual_files = _snapshot_files(source_root)
        if actual_files != expected_files:
            raise SourceManifestError(f"source file inventory changed for {name}")
        if record.get("file_count") != len(actual_files):
            raise SourceManifestError(f"source file count changed for {name}")
        if record.get("source_tree_sha256") != _canonical_hash(actual_files):
            raise SourceManifestError(f"source tree hash changed for {name}")
    return payload


def verified_source_commit_for_module(
    module: object,
    source_name: str,
    *,
    data_root: Path | None = None,
    manifest_path: Path | None = None,
    specs: Mapping[str, SourceSpec] = EXPECTED_SOURCES,
) -> str:
    """Return a verified commit only when ``module`` is loaded from that tree.

    Upstream A-MEM and MemOS do not publish commit metadata in their Python
    modules.  Mutating their source to inject a marker makes the checkout dirty,
    so live adapters use this externally verified identity instead.
    """
    if source_name not in specs:
        raise SourceManifestError(f"unknown source backend: {source_name}")
    root = data_root
    if root is None:
        configured_root = os.environ.get("LHMSB_DATA_ROOT")
        if not configured_root:
            raise SourceManifestError("LHMSB_DATA_ROOT is required for source identity")
        root = Path(configured_root)
    resolved_root = root.expanduser().resolve()
    path = manifest_path
    if path is None:
        configured_manifest = os.environ.get("LHMSB_SOURCE_TREE_MANIFEST_PATH")
        path = (
            Path(configured_manifest)
            if configured_manifest
            else resolved_root / "manifests" / "system-sources.json"
        )
    payload = verify_source_manifest(resolved_root, path, specs=specs)
    record = payload["sources"]
    if not isinstance(record, dict) or not isinstance(record.get(source_name), dict):
        raise SourceManifestError(f"source manifest record is missing: {source_name}")
    spec = specs[source_name]
    source_root = _resolve_source_root(resolved_root, spec.relative_root)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise SourceManifestError(f"module path is unavailable for {source_name}")
    resolved_module = Path(module_file).expanduser().resolve()
    try:
        resolved_module.relative_to(source_root)
    except ValueError as exc:
        raise SourceManifestError(
            f"{source_name} module was not imported from the verified source tree: "
            f"{resolved_module}"
        ) from exc
    return spec.source_commit


def _resolve_source_root(data_root: Path, relative_root: str) -> Path:
    relative = Path(relative_root)
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceManifestError(f"source root must be data-root relative: {relative_root}")
    root = (data_root / relative).resolve()
    if not root.is_dir():
        raise SourceManifestError(f"source checkout is missing: {root}")
    try:
        root.relative_to(data_root)
    except ValueError as exc:
        raise SourceManifestError(f"source checkout escapes data root: {root}") from exc
    return root


def _git_identity(root: Path) -> dict[str, str]:
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise SourceManifestError(
            f"source checkout is not its own Git worktree: {root} (resolved {top})"
        )
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise SourceManifestError(f"tracked source files are dirty: {root}")
    untracked = tuple(
        path
        for path in _git_untracked_paths(root)
        if not _is_ignored_relative_path(path)
    )
    if untracked:
        sample = ", ".join(untracked[:3])
        raise SourceManifestError(
            f"untracked source files are present in {root}: {sample}"
        )
    return {
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "source_url": _git(root, "remote", "get-url", "origin"),
    }


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise SourceManifestError(
            f"cannot determine Git identity for {root}: {str(detail).strip()}"
        ) from exc
    return result.stdout.strip()


def _git_untracked_paths(root: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        if isinstance(detail, bytes):
            detail = os.fsdecode(detail)
        raise SourceManifestError(
            f"cannot enumerate untracked files for {root}: {str(detail).strip()}"
        ) from exc
    return tuple(
        sorted(
            os.fsdecode(raw)
            for raw in result.stdout.split(b"\0")
            if raw
        )
    )


def _verify_expected_identity(
    name: str,
    identity: Mapping[str, str],
    spec: SourceSpec,
) -> None:
    if identity.get("source_commit") != spec.source_commit:
        raise SourceManifestError(f"source commit does not match the lock for {name}")
    if _normalize_git_url(identity.get("source_url", "")) != _normalize_git_url(
        spec.source_url
    ):
        raise SourceManifestError(f"source origin does not match the lock for {name}")


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.casefold()


def _snapshot_files(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_ignored_directory_name(name)
        )
        base = Path(directory)
        for name in sorted(file_names):
            if name in _IGNORED_FILE_NAMES or name.endswith(_IGNORED_SUFFIXES):
                continue
            path = base / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            executable = bool(metadata.st_mode & stat.S_IXUSR)
            if path.is_symlink():
                target = os.readlink(path)
                content = target.encode("utf-8", errors="surrogateescape")
                records.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": target,
                        "size": len(content),
                        "executable": executable,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
                continue
            if not path.is_file():
                raise SourceManifestError(f"unsupported source file type: {path}")
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": metadata.st_size,
                    "executable": executable,
                    "sha256": _sha256_file(path),
                }
            )
    records.sort(key=lambda item: str(item["path"]))
    return records


def _is_ignored_directory_name(name: str) -> bool:
    return name in _IGNORED_DIRECTORY_NAMES or name.endswith(".egg-info")


def _is_ignored_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        any(_is_ignored_directory_name(part) for part in path.parts[:-1])
        or path.name in _IGNORED_FILE_NAMES
        or path.name.endswith(_IGNORED_SUFFIXES)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument("--manifest", type=Path)
    module = subparsers.add_parser("verify-module")
    module.add_argument("--data-root", type=Path, required=True)
    module.add_argument("--manifest", type=Path)
    module.add_argument("--source", choices=tuple(sorted(EXPECTED_SOURCES)), required=True)
    module.add_argument("--module", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            create_source_manifest(args.data_root, args.manifest)
            verb = "created"
        elif args.command == "verify":
            verify_source_manifest(args.data_root, args.manifest)
            verb = "verified"
        else:
            module = importlib.import_module(args.module)
            verified_source_commit_for_module(
                module,
                args.source,
                data_root=args.data_root,
                manifest_path=args.manifest,
            )
            verb = f"verified for module {args.module}"
    except SourceManifestError as exc:
        raise SystemExit(str(exc)) from exc
    path = args.manifest or args.data_root / "manifests" / "system-sources.json"
    print(f"source manifest {verb}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
