"""Validated task and condition schemas for real-repository experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must contain only letters, digits, dot, underscore, or dash")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    prompt: str


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    memory_mode: str
    workspace_files_visible: bool
    settings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    repository: Path
    revision: str
    sessions: tuple[SessionSpec, ...]
    setup_commands: tuple[str, ...]
    hidden_test_commands: tuple[str, ...]
    regression_test_commands: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    repeats: int
    timeout_seconds: int
    tasks: tuple[TaskSpec, ...]
    conditions: tuple[ConditionSpec, ...]
    source_path: Path


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Load and validate a YAML experiment definition."""
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML root must be a mapping")
    experiment_id = _identifier(raw.get("experiment_id"), "experiment_id")
    repeats = raw.get("repeats", 1)
    timeout = raw.get("timeout_seconds", 1800)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be an integer >= 1")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise ValueError("timeout_seconds must be an integer >= 1")

    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must be a non-empty list")
    conditions: list[ConditionSpec] = []
    for item in raw_conditions:
        if not isinstance(item, dict):
            raise ValueError("each condition must be a mapping")
        condition_id = _identifier(item.get("condition_id"), "condition_id")
        mode = item.get("memory_mode", "none")
        visible = item.get("workspace_files_visible", True)
        settings = item.get("settings", {})
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError(f"condition {condition_id}: memory_mode must be non-empty")
        if not isinstance(visible, bool):
            raise ValueError(f"condition {condition_id}: workspace_files_visible must be boolean")
        if not isinstance(settings, dict) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float, bool))
            for key, value in settings.items()
        ):
            raise ValueError(f"condition {condition_id}: settings must be a scalar mapping")
        conditions.append(
            ConditionSpec(
                condition_id=condition_id,
                memory_mode=mode,
                workspace_files_visible=visible,
                settings=tuple(sorted((key, str(value)) for key, value in settings.items())),
            )
        )
    if len({item.condition_id for item in conditions}) != len(conditions):
        raise ValueError("condition_id values must be unique")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a non-empty list")
    tasks: list[TaskSpec] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise ValueError("each task must be a mapping")
        task_id = _identifier(item.get("task_id"), "task_id")
        repo_value = item.get("repository")
        if not isinstance(repo_value, str) or not repo_value.strip():
            raise ValueError(f"task {task_id}: repository must be a path")
        repo = Path(repo_value).expanduser()
        if not repo.is_absolute():
            repo = (source.parent / repo).resolve()
        revision = item.get("revision", "HEAD")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"task {task_id}: revision must be non-empty")
        raw_sessions = item.get("sessions")
        if not isinstance(raw_sessions, list) or not raw_sessions:
            raise ValueError(f"task {task_id}: sessions must be a non-empty list")
        sessions: list[SessionSpec] = []
        for index, session in enumerate(raw_sessions):
            if not isinstance(session, dict) or not isinstance(session.get("prompt"), str):
                raise ValueError(f"task {task_id}: each session needs a prompt")
            session_id = _identifier(
                session.get("session_id", f"session-{index + 1}"), "session_id"
            )
            sessions.append(SessionSpec(session_id, session["prompt"]))
        hidden_test_commands = _string_tuple(
            item.get("hidden_test_commands", []), "hidden_test_commands"
        )
        if not hidden_test_commands:
            raise ValueError(f"task {task_id}: at least one hidden_test_command is required")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                repository=repo,
                revision=revision,
                sessions=tuple(sessions),
                setup_commands=_string_tuple(item.get("setup_commands", []), "setup_commands"),
                hidden_test_commands=hidden_test_commands,
                regression_test_commands=_string_tuple(
                    item.get("regression_test_commands", []), "regression_test_commands"
                ),
            )
        )
    if len({item.task_id for item in tasks}) != len(tasks):
        raise ValueError("task_id values must be unique")
    for task in tasks:
        if len({session.session_id for session in task.sessions}) != len(task.sessions):
            raise ValueError(f"task {task.task_id}: session_id values must be unique")
    return ExperimentSpec(
        experiment_id=experiment_id,
        repeats=repeats,
        timeout_seconds=timeout,
        tasks=tuple(tasks),
        conditions=tuple(conditions),
        source_path=source,
    )
