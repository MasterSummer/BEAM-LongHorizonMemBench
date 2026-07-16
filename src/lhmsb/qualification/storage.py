"""Atomic task-local storage with identity and payload hash validation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.schema import QualificationTask

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

    def task_directory(self, task: QualificationTask) -> Path:
        return self.root / "tasks" / str(task.task_id)

    def prepare_task(
        self,
        task: QualificationTask,
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
        task: QualificationTask,
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
        task: QualificationTask,
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
