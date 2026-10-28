"""Run paired multi-session experiments against clean Git repository snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from lhmsb.realworld.schema import ConditionSpec, ExperimentSpec, TaskSpec


class AgentExecutor(Protocol):
    """One agent turn. Implementations may call SWE-agent or another harness."""

    def run_session(
        self,
        *,
        workspace: Path,
        prompt: str,
        session_id: str,
        condition: ConditionSpec,
        transcript_path: Path,
    ) -> dict[str, object]:
        """Run a session in the persistent task workspace and return auditable metadata."""


class CommandExecutor:
    """Invoke an external agent once per session with an auditable interface."""

    def __init__(self, command: str, *, timeout_seconds: int = 3600) -> None:
        self._argv = tuple(shlex.split(command))
        self._timeout = timeout_seconds
        if not self._argv:
            raise ValueError("agent command must not be empty")
        if timeout_seconds < 1:
            raise ValueError("agent timeout_seconds must be >= 1")

    def run_session(
        self,
        *,
        workspace: Path,
        prompt: str,
        session_id: str,
        condition: ConditionSpec,
        transcript_path: Path,
    ) -> dict[str, object]:
        """Pass prompt/condition through env and save the complete process transcript."""
        prompt_path = transcript_path.with_suffix(".prompt.txt")
        prompt_path.write_text(prompt, encoding="utf-8")
        argv = [
            part.replace("{workspace}", str(workspace)).replace("{session_id}", session_id)
            for part in self._argv
        ]
        env = {
            **os.environ,
            "LHMSB_WORKSPACE": str(workspace),
            "LHMSB_PROMPT_FILE": str(prompt_path),
            "LHMSB_SESSION_ID": session_id,
            "LHMSB_CONDITION_JSON": json.dumps(
                asdict(condition), ensure_ascii=False, sort_keys=True
            ),
            "LHMSB_TRANSCRIPT_PATH": str(transcript_path),
        }
        started = time.time()
        try:
            result = subprocess.run(
                argv,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            stdout, stderr, returncode, timed_out = (
                result.stdout,
                result.stderr,
                result.returncode,
                False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            returncode, timed_out = 124, True
        transcript_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "argv": argv,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "stdout": stdout,
                    "stderr": stderr,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "success": returncode == 0 and not timed_out,
            "returncode": returncode,
            "timed_out": timed_out,
        }


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    timed_out: bool
    stdout_path: str
    stderr_path: str


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "-" for char in value)


def _materialize_snapshot(task: TaskSpec, workspace: Path) -> str:
    if not task.repository.is_dir():
        raise FileNotFoundError(f"repository does not exist: {task.repository}")
    revision = _git(task.repository, "rev-parse", "--verify", f"{task.revision}^{{commit}}")
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=task.repository,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as bundle:
        for member in bundle.getmembers():
            target = (workspace / member.name).resolve()
            if target != workspace.resolve() and workspace.resolve() not in target.parents:
                raise ValueError(f"unsafe path in git archive: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                target.write_bytes(source.read())
                target.chmod(member.mode & 0o777)
            elif member.issym():
                link_target = (target.parent / member.linkname).resolve()
                if (
                    target.parent.resolve() not in link_target.parents
                    and link_target != workspace.resolve()
                ):
                    raise ValueError(f"unsafe symlink in git archive: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
            else:
                raise ValueError(f"unsupported archive member type: {member.name!r}")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "experiment@localhost"], cwd=workspace, check=True
    )
    subprocess.run(["git", "config", "user.name", "Experiment"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "experiment baseline"], cwd=workspace, check=True)
    return revision


def _run_commands(
    commands: tuple[str, ...], *, workspace: Path, output: Path, timeout_seconds: int, phase: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, command in enumerate(commands):
        stdout_path = output / f"{phase}-{index + 1}.stdout.txt"
        stderr_path = output / f"{phase}-{index + 1}.stderr.txt"
        try:
            result = subprocess.run(
                command,
                cwd=workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "LHMSB_EXPERIMENT_WORKSPACE": str(workspace)},
            )
            timed_out = False
            returncode = result.returncode
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        records.append(
            asdict(
                CommandResult(
                    command=command,
                    returncode=returncode,
                    timed_out=timed_out,
                    stdout_path=stdout_path.name,
                    stderr_path=stderr_path.name,
                )
            )
        )
    return records


def _snapshot_diff(workspace: Path, output: Path) -> dict[str, object]:
    subprocess.run(["git", "add", "-N", "-A"], cwd=workspace, check=False, capture_output=True)
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    patch_path = output / "changes.patch"
    patch_path.write_bytes(patch)
    status = subprocess.run(
        ["git", "status", "--short"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "patch_path": patch_path.name,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "changed_paths": status,
    }


def _run_unit(
    *,
    spec: ExperimentSpec,
    task: TaskSpec,
    condition: ConditionSpec,
    repeat: int,
    output: Path,
    executor: AgentExecutor,
) -> dict[str, object]:
    unit_id = "__".join(
        _safe_name(value)
        for value in (spec.experiment_id, task.task_id, condition.condition_id, f"r{repeat:02d}")
    )
    unit_dir = output / unit_id
    if unit_dir.exists():
        raise FileExistsError(f"run directory already exists; refusing overwrite: {unit_dir}")
    unit_dir.mkdir(parents=True)
    workspace = unit_dir / "workspace"
    workspace.mkdir()
    revision = _materialize_snapshot(task, workspace)
    started_at = time.time()
    setup_records = _run_commands(
        task.setup_commands,
        workspace=workspace,
        output=unit_dir,
        timeout_seconds=spec.timeout_seconds,
        phase="setup",
    )
    sessions: list[dict[str, object]] = []
    if all(item["returncode"] == 0 and not item["timed_out"] for item in setup_records):
        for session in task.sessions:
            session_output = unit_dir / f"{_safe_name(session.session_id)}.json"
            try:
                record = executor.run_session(
                    workspace=workspace,
                    prompt=session.prompt,
                    session_id=session.session_id,
                    condition=condition,
                    transcript_path=session_output,
                )
            except Exception as exc:
                record = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                session_output.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            sessions.append(
                {"session_id": session.session_id, "record_path": session_output.name, **record}
            )
            if record.get("success") is False:
                break
    hidden = _run_commands(
        task.hidden_test_commands,
        workspace=workspace,
        output=unit_dir,
        timeout_seconds=spec.timeout_seconds,
        phase="hidden-test",
    )
    regression = _run_commands(
        task.regression_test_commands,
        workspace=workspace,
        output=unit_dir,
        timeout_seconds=spec.timeout_seconds,
        phase="regression-test",
    )
    changes = _snapshot_diff(workspace, unit_dir)
    setup_pass = all(item["returncode"] == 0 and not item["timed_out"] for item in setup_records)
    agent_pass = len(sessions) == len(task.sessions) and all(
        item.get("success") is True for item in sessions
    )
    hidden_pass = bool(hidden) and all(
        item["returncode"] == 0 and not item["timed_out"] for item in hidden
    )
    regression_pass = bool(regression) and all(
        item["returncode"] == 0 and not item["timed_out"] for item in regression
    )
    payload: dict[str, object] = {
        "schema_version": "lhmsb-realworld-run-v1",
        "unit_id": unit_id,
        "experiment_id": spec.experiment_id,
        "task_id": task.task_id,
        "condition": asdict(condition),
        "repeat": repeat,
        "repository": str(task.repository),
        "revision": revision,
        "setup": setup_records,
        "sessions": sessions,
        "hidden_tests": hidden,
        "regression_tests": regression,
        "setup_pass": setup_pass,
        "agent_pass": agent_pass,
        "hidden_test_pass": hidden_pass,
        "task_success": setup_pass and agent_pass and hidden_pass,
        "regression_pass": regression_pass,
        "changes": changes,
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
    }
    (unit_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["result_sha256"] = _sha256(unit_dir / "result.json")
    return payload


def run_experiment(
    spec: ExperimentSpec,
    *,
    output: str | Path,
    executor: AgentExecutor,
    repeats: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Run every task-condition-repeat cell, refusing to overwrite prior output."""
    run_count = spec.repeats if repeats is None else repeats
    if run_count < 1:
        raise ValueError("repeats must be >= 1")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for task in spec.tasks:
        for repeat in range(1, run_count + 1):
            # Rotate condition order across repeats to reduce order and service-load bias.
            offset = (repeat - 1) % len(spec.conditions)
            ordered = spec.conditions[offset:] + spec.conditions[:offset]
            for condition in ordered:
                records.append(
                    _run_unit(
                        spec=spec,
                        task=task,
                        condition=condition,
                        repeat=repeat,
                        output=root,
                        executor=executor,
                    )
                )
    ledger = root / "results.jsonl"
    with ledger.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return tuple(records)
