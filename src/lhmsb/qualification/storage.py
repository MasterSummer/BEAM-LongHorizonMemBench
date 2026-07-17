"""Atomic task-local storage with identity and payload hash validation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.prefix import (
    MemoryPrefixArtifact,
    PrefixArtifactError,
)
from lhmsb.qualification.schema import PreparationTask, QualificationTask

QUALIFICATION_STORAGE_SCHEMA_VERSION = 1


class QualificationStorageError(RuntimeError):
    """Typed persistence or resume failure."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class QualificationStorage:
    """Persist independently verifiable cells under one immutable run root."""

    def __init__(self, root: Path, *, run_identity: str) -> None:
        if not run_identity:
            raise ValueError("run_identity must be non-empty")
        self.root = root
        self.run_identity = run_identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.operation_log: list[tuple[str, str]] = []

    def task_directory(self, task: QualificationTask | PreparationTask) -> Path:
        return self.root / "tasks" / str(task.task_id)

    def prefix_directory(self, task: PreparationTask) -> Path:
        """Return the isolated directory for one immutable prefix task."""
        return self.task_directory(task) / "prefix"

    def prefix_artifact_path(self, task: PreparationTask) -> Path:
        """Return the canonical path of the complete prefix artifact."""
        return self.prefix_directory(task) / "artifact.json"

    # ``artifact_path`` is kept as a small compatibility alias for workers that
    # use the terminology from the dataset manifest.
    def artifact_path(self, task: PreparationTask) -> Path:
        return self.prefix_artifact_path(task)

    def prefix_failure_path(self, task: PreparationTask) -> Path:
        return self.prefix_directory(task) / "FAILED.json"

    def prepare_task(
        self,
        task: QualificationTask | PreparationTask,
        *,
        episode_hash: str,
    ) -> Path:
        """Create or validate the immutable task identity record."""
        if task.run_identity != self.run_identity:
            raise QualificationStorageError(
                "identity_mismatch",
                f"task run identity {task.run_identity!r} != {self.run_identity!r}",
            )
        directory = self.task_directory(task)
        directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "storage_schema_version": QUALIFICATION_STORAGE_SCHEMA_VERSION,
            "run_identity": self.run_identity,
            "episode_hash": episode_hash,
            "task": asdict(task),
        }
        path = directory / "task_identity.json"
        if path.exists():
            existing = self._read_json(path)
            if canonical_hash(existing) != canonical_hash(identity):
                raise QualificationStorageError(
                    "identity_mismatch",
                    f"task identity does not match existing record: {task.task_id}",
                )
            return directory
        self._atomic_write(path, identity)
        return directory

    def load_cell(
        self,
        task: QualificationTask | PreparationTask,
        relative_path: str,
        *,
        input_hash: str,
    ) -> object | None:
        """Load one valid cell, returning ``None`` when it has not run yet."""
        path = self.task_directory(task) / relative_path
        if not path.exists():
            return None
        envelope = self._read_json(path)
        if envelope.get("schema_version") != QUALIFICATION_STORAGE_SCHEMA_VERSION:
            raise QualificationStorageError(
                "trace_incomplete",
                f"unsupported cell schema: {path}",
            )
        if envelope.get("input_hash") != input_hash:
            raise QualificationStorageError(
                "identity_mismatch",
                f"cell input hash changed: {path}",
            )
        payload = envelope.get("payload")
        if envelope.get("payload_hash") != canonical_hash(payload):
            raise QualificationStorageError(
                "trace_incomplete",
                f"cell payload hash mismatch: {path}",
            )
        return payload

    def save_cell(
        self,
        task: QualificationTask | PreparationTask,
        relative_path: str,
        *,
        input_hash: str,
        payload: object,
    ) -> bool:
        """Atomically write a cell; return ``False`` when already identical."""
        existing = self.load_cell(
            task,
            relative_path,
            input_hash=input_hash,
        )
        if existing is not None:
            if canonical_hash(existing) != canonical_hash(payload):
                raise QualificationStorageError(
                    "trace_incomplete",
                    f"completed cell changed output: {relative_path}",
                )
            return False
        envelope = {
            "schema_version": QUALIFICATION_STORAGE_SCHEMA_VERSION,
            "input_hash": input_hash,
            "payload_hash": canonical_hash(payload),
            "payload": payload,
        }
        path = self.task_directory(task) / relative_path
        self._atomic_write(path, envelope)
        return True

    def save_prefix_artifact(
        self,
        task: PreparationTask,
        artifact: MemoryPrefixArtifact,
    ) -> bool:
        """Atomically publish a complete, hash-verified prefix artifact.

        The artifact is written as its own canonical JSON document rather than a
        resumable cell envelope.  This makes the content-addressed hash visible
        to the Stage-B planner and ensures a partially written prefix can never
        be mistaken for an executable artifact.  A second identical publication
        is idempotent; a different artifact for the same task is terminal.
        """
        self._validate_prefix_identity(task, artifact)
        # Recalculate all nested hashes before touching disk.  ``from_dict`` is
        # intentionally used even for an object instance so future schema
        # changes cannot bypass the decoder/validator boundary.
        try:
            verified = MemoryPrefixArtifact.from_dict(artifact.to_dict())
        except PrefixArtifactError as exc:
            raise QualificationStorageError("trace_incomplete", str(exc)) from exc
        path = self.prefix_artifact_path(task)
        failure = self.prefix_failure_path(task)
        if path.exists():
            existing = self.load_prefix_artifact(task)
            if existing is None:
                raise QualificationStorageError(
                    "trace_incomplete",
                    f"prefix artifact path exists but could not be loaded: {path}",
                )
            if existing.artifact_hash != verified.artifact_hash:
                raise QualificationStorageError(
                    "identity_mismatch",
                    f"immutable prefix artifact changed for task {task.task_id}",
                )
            return False
        self._atomic_write(path, verified.to_dict())
        # A successful rerun replaces an explicit failure marker only after the
        # complete artifact has been atomically installed.
        with suppress(FileNotFoundError):
            failure.unlink()
        return True

    def load_prefix_artifact(
        self,
        task: PreparationTask,
        *,
        expected: Mapping[str, object] | None = None,
    ) -> MemoryPrefixArtifact | None:
        """Load and verify a complete prefix artifact, if one exists."""
        path = self.prefix_artifact_path(task)
        failure = self.prefix_failure_path(task)
        if failure.exists() and not path.exists():
            record = self._read_json(failure)
            error_class = record.get("error_class", "preparation_failed")
            message = record.get("error_message", "prefix preparation failed")
            raise QualificationStorageError(
                str(error_class),
                str(message),
            )
        if not path.exists():
            return None
        try:
            artifact = MemoryPrefixArtifact.from_dict(self._read_json(path))
        except PrefixArtifactError as exc:
            raise QualificationStorageError("trace_incomplete", str(exc)) from exc
        self._validate_prefix_identity(task, artifact)
        if expected is not None:
            for key, value in expected.items():
                actual = getattr(artifact, str(key), None)
                if actual != value:
                    raise QualificationStorageError(
                        "identity_mismatch",
                        f"prefix artifact {key} does not match expected identity",
                    )
        if failure.exists():
            raise QualificationStorageError(
                "trace_incomplete",
                "prefix artifact has a stale failure marker",
            )
        return artifact

    def verify_prefix_artifact(
        self,
        task: PreparationTask,
        *,
        expected: Mapping[str, object] | None = None,
    ) -> MemoryPrefixArtifact:
        """Strict variant of :meth:`load_prefix_artifact`."""
        artifact = self.load_prefix_artifact(task, expected=expected)
        if artifact is None:
            raise QualificationStorageError(
                "trace_incomplete",
                f"missing prefix artifact for task {task.task_id}",
            )
        return artifact

    def mark_prefix_failed(
        self,
        task: PreparationTask,
        *,
        error_class: str,
        error_message: str,
    ) -> None:
        """Persist an explicit failed preparation without publishing an artifact."""
        path = self.prefix_artifact_path(task)
        if path.exists():
            raise QualificationStorageError(
                "identity_mismatch",
                "cannot mark a task failed after publishing an immutable artifact",
            )
        if not error_class or not error_message:
            raise ValueError("error_class and error_message must be non-empty")
        self._atomic_write(
            self.prefix_failure_path(task),
            {
                "schema_version": QUALIFICATION_STORAGE_SCHEMA_VERSION,
                "task_id": task.task_id,
                "run_identity": self.run_identity,
                "status": "failed",
                "error_class": error_class,
                "error_message": error_message,
            },
        )

    def clear_prefix_failure(self, task: PreparationTask) -> None:
        """Clear a failed-preparation marker before an explicit full rerun."""
        try:
            self.prefix_failure_path(task).unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _validate_prefix_identity(
        task: PreparationTask,
        artifact: MemoryPrefixArtifact,
    ) -> None:
        if artifact.episode_id != task.episode_id:
            raise QualificationStorageError(
                "identity_mismatch",
                "prefix artifact episode does not match preparation task",
            )
        if artifact.backend != task.backend:
            raise QualificationStorageError(
                "identity_mismatch",
                "prefix artifact backend does not match preparation task",
            )
        if artifact.profile_id != task.profile_id:
            raise QualificationStorageError(
                "identity_mismatch",
                "prefix artifact profile does not match preparation task",
            )
        if not task.run_identity:
            raise QualificationStorageError(
                "identity_mismatch",
                "preparation task run identity is empty",
            )

    def _atomic_write(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        )
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
        self.operation_log.append(
            ("write", path.relative_to(self.root).as_posix())
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QualificationStorageError(
                "trace_incomplete",
                f"cannot read persisted cell {path}: {exc}",
            ) from exc
        if not isinstance(value, dict):
            raise QualificationStorageError(
                "trace_incomplete",
                f"persisted cell must be an object: {path}",
            )
        return {str(key): child for key, child in value.items()}


__all__ = [
    "QUALIFICATION_STORAGE_SCHEMA_VERSION",
    "QualificationStorage",
    "QualificationStorageError",
]
