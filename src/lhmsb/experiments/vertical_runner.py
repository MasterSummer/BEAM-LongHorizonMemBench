"""Immutable planning and atomic execution for the vertical offline pilot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lhmsb.datasets.stateful_loader import load_software_vertical_specs
from lhmsb.datasets.stateful_pipeline import StatefulDatasetError, StatefulManifest
from lhmsb.experiments.vertical_config import (
    GitSnapshot,
    VerticalExperimentError,
    VerticalTask,
    build_vertical_tasks,
    canonical_hash,
    load_vertical_offline_config,
)
from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.longhorizon.runner import VerticalRunResult, run_vertical_episode

VERTICAL_RUN_SCHEMA_VERSION = 1
_MANAGED_FILES = {
    "run_config.yaml",
    "run_manifest.json",
    "tasks.jsonl",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "summary.json",
}


@dataclass(frozen=True)
class VerticalRunManifest:
    """Reproducibility contract written before any atomic task executes."""

    schema_version: int
    run_identity: str
    experiment_id: str
    code_commit: str
    code_dirty: bool
    code_ref: str
    dataset_path: str
    dataset_manifest_sha256: str
    dataset_schema_version: int
    dataset_generator_version: str
    dataset_family: str
    dataset_episodes: tuple[dict[str, object], ...]
    config_path: str
    config_hash: str
    config: dict[str, object]
    task_count: int
    created_at_utc: str
    environment: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_identity": self.run_identity,
            "experiment_id": self.experiment_id,
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
            "code_ref": self.code_ref,
            "dataset_path": self.dataset_path,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_generator_version": self.dataset_generator_version,
            "dataset_family": self.dataset_family,
            "dataset_episodes": [dict(item) for item in self.dataset_episodes],
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "config": dict(self.config),
            "task_count": self.task_count,
            "created_at_utc": self.created_at_utc,
            "environment": dict(self.environment),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> VerticalRunManifest:
        try:
            return cls(
                schema_version=_integer(data["schema_version"], "schema_version"),
                run_identity=_string(data["run_identity"], "run_identity"),
                experiment_id=_string(data["experiment_id"], "experiment_id"),
                code_commit=_string(data["code_commit"], "code_commit"),
                code_dirty=_boolean(data["code_dirty"], "code_dirty"),
                code_ref=_string(data["code_ref"], "code_ref"),
                dataset_path=_string(data["dataset_path"], "dataset_path"),
                dataset_manifest_sha256=_string(
                    data["dataset_manifest_sha256"], "dataset_manifest_sha256"
                ),
                dataset_schema_version=_integer(
                    data["dataset_schema_version"], "dataset_schema_version"
                ),
                dataset_generator_version=_string(
                    data["dataset_generator_version"], "dataset_generator_version"
                ),
                dataset_family=_string(data["dataset_family"], "dataset_family"),
                dataset_episodes=_dict_sequence(
                    data["dataset_episodes"], "dataset_episodes"
                ),
                config_path=_string(data["config_path"], "config_path"),
                config_hash=_string(data["config_hash"], "config_hash"),
                config=_string_dict_keys(data["config"], "config"),
                task_count=_integer(data["task_count"], "task_count"),
                created_at_utc=_string(data["created_at_utc"], "created_at_utc"),
                environment=_string_mapping(data["environment"], "environment"),
            )
        except KeyError as exc:
            raise VerticalExperimentError(f"run manifest missing field: {exc.args[0]}") from exc


def current_git_snapshot(root: Path | None = None) -> GitSnapshot:
    """Return the current checkout SHA, dirty flag, and branch/ref label."""
    cwd = root or Path(__file__).resolve().parents[3]
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain", "--untracked-files=normal"], cwd)
    try:
        ref = _git(["symbolic-ref", "--short", "-q", "HEAD"], cwd)
    except VerticalExperimentError:
        ref = "detached"
    return GitSnapshot(commit=commit, dirty=bool(status), ref=ref)


def plan_vertical_run(
    dataset: Path,
    config_path: Path,
    run_dir: Path,
    *,
    allow_dirty: bool = False,
    force: bool = False,
) -> VerticalRunManifest:
    """Create or idempotently reopen an immutable vertical run contract."""
    config = load_vertical_offline_config(config_path)
    specs = load_software_vertical_specs(dataset)
    dataset_root = dataset.resolve()
    dataset_manifest_path = dataset_root / "MANIFEST.json"
    dataset_manifest = _stateful_manifest(dataset_manifest_path)
    dataset_manifest_sha = _sha256_file(dataset_manifest_path)
    snapshot = current_git_snapshot()
    if snapshot.dirty and not allow_dirty:
        raise VerticalExperimentError(
            "Git worktree is dirty; commit changes or pass allow_dirty=True"
        )
    identity_payload = {
        "schema_version": VERTICAL_RUN_SCHEMA_VERSION,
        "code_commit": snapshot.commit,
        "code_dirty": snapshot.dirty,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "dataset_schema_version": dataset_manifest.schema_version,
        "dataset_generator_version": dataset_manifest.generator_version,
        "config_hash": config.config_hash,
    }
    run_identity = canonical_hash(identity_payload)
    tasks = build_vertical_tasks(specs, config, run_identity=run_identity)
    existing = _prepare_run_directory(run_dir, run_identity=run_identity, force=force)
    if existing is not None:
        _validate_existing_plan(run_dir, existing)
        return existing
    manifest = VerticalRunManifest(
        schema_version=VERTICAL_RUN_SCHEMA_VERSION,
        run_identity=run_identity,
        experiment_id=config.experiment_id,
        code_commit=snapshot.commit,
        code_dirty=snapshot.dirty,
        code_ref=snapshot.ref,
        dataset_path=str(dataset_root),
        dataset_manifest_sha256=dataset_manifest_sha,
        dataset_schema_version=dataset_manifest.schema_version,
        dataset_generator_version=dataset_manifest.generator_version,
        dataset_family=dataset_manifest.family,
        dataset_episodes=dataset_manifest.episodes,
        config_path=str(config_path.resolve()),
        config_hash=config.config_hash,
        config=config.to_dict(),
        task_count=len(tasks),
        created_at_utc=datetime.now(UTC).isoformat(),
        environment=_environment_snapshot(),
    )
    _atomic_bytes(run_dir / "run_config.yaml", config_path.read_bytes())
    _atomic_json(run_dir / "run_manifest.json", manifest.to_dict())
    _atomic_jsonl(run_dir / "tasks.jsonl", [task.to_dict() for task in tasks])
    return manifest


def read_vertical_tasks(run_dir: Path) -> tuple[VerticalTask, ...]:
    """Read and validate the ordered atomic task table."""
    path = run_dir / "tasks.jsonl"
    records = _read_jsonl(path)
    tasks = tuple(VerticalTask.from_dict(record) for record in records)
    for expected, task in enumerate(tasks):
        if task.task_index != expected:
            raise VerticalExperimentError(
                f"task index sequence is invalid at row {expected}: {task.task_index}"
            )
    return tasks


def run_vertical_task(
    run_dir: Path,
    task_index: int,
    *,
    force: bool = False,
) -> Path:
    """Execute one task independently and atomically persist its result."""
    manifest, tasks = _load_run_contract(run_dir)
    _verify_current_code(manifest)
    if task_index < 0 or task_index >= len(tasks):
        raise VerticalExperimentError(
            f"task index {task_index} is outside [0, {len(tasks) - 1}]"
        )
    task = tasks[task_index]
    task_dir = run_dir / "tasks" / task.task_id
    result_path = task_dir / "result.json"
    if result_path.exists() and not force:
        _validate_result_file(result_path, manifest, task)
        return result_path
    specs = _verified_run_specs(manifest)
    spec = _select_spec(specs, task.episode_id)
    try:
        result = run_vertical_episode(
            spec,
            task.condition,
            intervention_state_id=task.intervention_state_id,
        )
        _atomic_json(result_path, _result_envelope(manifest, task, result))
        (task_dir / "failure.json").unlink(missing_ok=True)
        return result_path
    except Exception as exc:
        _atomic_json(task_dir / "failure.json", _failure_envelope(manifest, task, exc))
        raise


def _prepare_run_directory(
    run_dir: Path,
    *,
    run_identity: str,
    force: bool,
) -> VerticalRunManifest | None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return None
    if not run_dir.is_dir():
        raise VerticalExperimentError(f"run output is not a directory: {run_dir}")
    contents = tuple(run_dir.iterdir())
    if not contents:
        return None
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise VerticalExperimentError(
            f"refusing to modify unrelated non-empty directory: {run_dir}"
        )
    existing = _run_manifest(manifest_path)
    if existing.run_identity == run_identity and not force:
        return existing
    if not force:
        raise VerticalExperimentError(
            "run identity differs from the existing directory; pass force=True to replace it"
        )
    _clear_managed_run_files(run_dir)
    return None


def _clear_managed_run_files(run_dir: Path) -> None:
    for name in _MANAGED_FILES:
        path = run_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    tasks = run_dir / "tasks"
    if tasks.exists():
        shutil.rmtree(tasks)


def _validate_existing_plan(run_dir: Path, manifest: VerticalRunManifest) -> None:
    tasks = read_vertical_tasks(run_dir)
    if len(tasks) != manifest.task_count:
        raise VerticalExperimentError(
            f"existing task count mismatch: {len(tasks)} != {manifest.task_count}"
        )
    if any(task.run_identity != manifest.run_identity for task in tasks):
        raise VerticalExperimentError("existing tasks do not match the run identity")
    config_path = run_dir / "run_config.yaml"
    if not config_path.is_file():
        raise VerticalExperimentError("existing run is missing run_config.yaml")


def _load_run_contract(
    run_dir: Path,
) -> tuple[VerticalRunManifest, tuple[VerticalTask, ...]]:
    manifest = _run_manifest(run_dir / "run_manifest.json")
    if manifest.schema_version != VERTICAL_RUN_SCHEMA_VERSION:
        raise VerticalExperimentError(
            f"unsupported run schema version: {manifest.schema_version}"
        )
    tasks = read_vertical_tasks(run_dir)
    if len(tasks) != manifest.task_count:
        raise VerticalExperimentError(
            f"task count mismatch: {len(tasks)} != {manifest.task_count}"
        )
    for task in tasks:
        if task.run_identity != manifest.run_identity:
            raise VerticalExperimentError(f"task {task.task_id} has a stale run identity")
    return manifest, tasks


def _verify_current_code(manifest: VerticalRunManifest) -> None:
    snapshot = current_git_snapshot()
    if snapshot.commit != manifest.code_commit or snapshot.dirty != manifest.code_dirty:
        raise VerticalExperimentError(
            "current code snapshot does not match the planned run: "
            f"planned={manifest.code_commit}/dirty={manifest.code_dirty}, "
            f"current={snapshot.commit}/dirty={snapshot.dirty}"
        )


def _verified_run_specs(
    manifest: VerticalRunManifest,
) -> tuple[SoftwareVerticalSpec, ...]:
    dataset = Path(manifest.dataset_path)
    manifest_path = dataset / "MANIFEST.json"
    actual_sha = _sha256_file(manifest_path)
    if actual_sha != manifest.dataset_manifest_sha256:
        raise StatefulDatasetError(
            "dataset manifest checksum changed after planning: "
            f"expected {manifest.dataset_manifest_sha256}, got {actual_sha}"
        )
    specs: tuple[SoftwareVerticalSpec, ...] = load_software_vertical_specs(dataset)
    return specs


def _select_spec(
    specs: Sequence[SoftwareVerticalSpec],
    episode_id: str,
) -> SoftwareVerticalSpec:
    matches = [spec for spec in specs if spec.plan.episode_id == episode_id]
    if len(matches) != 1:
        raise VerticalExperimentError(
            f"expected one frozen spec for {episode_id}, found {len(matches)}"
        )
    return matches[0]


def _result_envelope(
    manifest: VerticalRunManifest,
    task: VerticalTask,
    result: VerticalRunResult,
) -> dict[str, object]:
    return {
        "schema_version": VERTICAL_RUN_SCHEMA_VERSION,
        "run_identity": manifest.run_identity,
        "task_payload_hash": task.task_payload_hash,
        "task": task.to_dict(),
        "result": {
            "episode_id": result.episode_id,
            "condition": result.condition,
            "behavior_score": result.behavior_score,
            "selected_actions": list(result.selected_actions),
            "sceu_results": [asdict(item) for item in result.sceu_results],
            "workspace_snapshot_hash": result.workspace_snapshot_hash,
            "prefix_hash": result.prefix_hash,
            "transcript_hash": result.transcript_hash,
            "native_trace": [asdict(item) for item in result.native_trace],
        },
    }


def _failure_envelope(
    manifest: VerticalRunManifest,
    task: VerticalTask,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema_version": VERTICAL_RUN_SCHEMA_VERSION,
        "run_identity": manifest.run_identity,
        "task_payload_hash": task.task_payload_hash,
        "task": task.to_dict(),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _validate_result_file(
    path: Path,
    manifest: VerticalRunManifest,
    task: VerticalTask,
) -> None:
    try:
        payload = _read_json(path)
    except VerticalExperimentError as exc:
        raise VerticalExperimentError(f"stale or corrupt result file {path}: {exc}") from exc
    if (
        payload.get("run_identity") != manifest.run_identity
        or payload.get("task_payload_hash") != task.task_payload_hash
        or payload.get("task") != task.to_dict()
        or not isinstance(payload.get("result"), Mapping)
    ):
        raise VerticalExperimentError(
            f"stale or corrupt result file for task {task.task_id}; pass force=True"
        )


def _stateful_manifest(path: Path) -> StatefulManifest:
    data = _read_json(path)
    try:
        return StatefulManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise StatefulDatasetError(f"malformed stateful manifest: {exc}") from exc


def _run_manifest(path: Path) -> VerticalRunManifest:
    data = _read_json(path)
    return VerticalRunManifest.from_dict(data)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerticalExperimentError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerticalExperimentError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VerticalExperimentError(f"cannot read JSONL file {path}: {exc}") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerticalExperimentError(
                f"invalid JSONL record {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise VerticalExperimentError(f"expected JSON object at {path}:{line_number}")
        records.append({str(key): item for key, item in value.items()})
    return tuple(records)


def _atomic_json(path: Path, value: object) -> None:
    data = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, data)


def _atomic_jsonl(path: Path, values: Sequence[object]) -> None:
    data = b"".join(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for value in values
    )
    _atomic_bytes(path, data)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StatefulDatasetError(f"cannot hash dataset file {path}: {exc}") from exc
    return digest.hexdigest()


def _environment_snapshot() -> dict[str, str]:
    try:
        package_version = importlib.metadata.version("lhmsb")
    except importlib.metadata.PackageNotFoundError:
        package_version = "uninstalled"
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "lhmsb_version": package_version,
    }


def _git(arguments: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VerticalExperimentError(
            f"cannot inspect Git checkout with {' '.join(arguments)}"
        ) from exc
    return result.stdout.strip()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerticalExperimentError(f"{label} must be an integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerticalExperimentError(f"{label} must be a non-empty string")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise VerticalExperimentError(f"{label} must be a boolean")
    return value


def _dict_sequence(value: object, label: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise VerticalExperimentError(f"{label} must be an array")
    return tuple(_string_dict_keys(item, f"{label} item") for item in value)


def _string_dict_keys(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise VerticalExperimentError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_mapping(value: object, label: str) -> dict[str, str]:
    mapping = _string_dict_keys(value, label)
    if any(not isinstance(item, str) for item in mapping.values()):
        raise VerticalExperimentError(f"{label} values must be strings")
    return {key: str(item) for key, item in mapping.items()}


__all__ = [
    "VERTICAL_RUN_SCHEMA_VERSION",
    "VerticalRunManifest",
    "current_git_snapshot",
    "plan_vertical_run",
    "read_vertical_tasks",
    "run_vertical_task",
]
