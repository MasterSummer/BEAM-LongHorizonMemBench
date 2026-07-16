"""Server-facing CLI for planning, executing, aggregating, and validating Mem0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from lhmsb.adapters.mem0_qualification import (
    Mem0QualificationAdapter,
    build_mem0_live_config,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker
from lhmsb.qualification.config import (
    QualificationConfig,
    QualificationConfigError,
    build_qualification_tasks,
    canonical_hash,
    load_qualification_config,
)
from lhmsb.qualification.preflight import (
    PreflightContext,
    PreflightError,
    current_repository_snapshot,
    load_mem0_specs,
    require_live_gate,
    run_preflight,
)
from lhmsb.qualification.providers import HttpPolicyClient
from lhmsb.qualification.report import write_qualification_report
from lhmsb.qualification.runner import (
    QualificationMatrixResult,
    QualificationRunError,
    QualificationTaskResult,
    TaskComponents,
    TaskIsolation,
    qualification_task_result_from_dict,
    run_qualification_task,
)
from lhmsb.qualification.schema import (
    PolicyProfile,
    QualificationCondition,
    QualificationTask,
    ReadoutKind,
    ScoredCondition,
)
from lhmsb.qualification.storage import (
    QualificationStorage,
    QualificationStorageError,
)
from lhmsb.qualification.tei import RerankerClient
from lhmsb.qualification.validate import validate_qualification_artifacts

QUALIFICATION_RUN_SCHEMA_VERSION = 1
_PROG = "python -m lhmsb.qualification"


class QualificationCliError(RuntimeError):
    """Invalid plan or CLI state that must not be silently repaired."""


@dataclass(frozen=True)
class QualificationRunManifest:
    schema_version: int
    run_identity: str
    experiment_id: str
    code_commit: str
    code_dirty: bool
    code_ref: str
    dataset_path: str
    dataset_manifest_sha256: str
    config_path: str
    config_hash: str
    dependency_lock_sha256: str
    task_count: int
    episode_ids: tuple[str, ...]
    n_sessions: int
    required_secret_env: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "episode_ids": list(self.episode_ids),
            "required_secret_env": list(self.required_secret_env),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
    ) -> QualificationRunManifest:
        return cls(
            schema_version=_integer(data.get("schema_version"), "schema_version"),
            run_identity=_text(data.get("run_identity"), "run_identity"),
            experiment_id=_text(data.get("experiment_id"), "experiment_id"),
            code_commit=_text(data.get("code_commit"), "code_commit"),
            code_dirty=_boolean(data.get("code_dirty"), "code_dirty"),
            code_ref=_text(data.get("code_ref"), "code_ref"),
            dataset_path=_text(data.get("dataset_path"), "dataset_path"),
            dataset_manifest_sha256=_text(
                data.get("dataset_manifest_sha256"),
                "dataset_manifest_sha256",
            ),
            config_path=_text(data.get("config_path"), "config_path"),
            config_hash=_text(data.get("config_hash"), "config_hash"),
            dependency_lock_sha256=_text(
                data.get("dependency_lock_sha256"),
                "dependency_lock_sha256",
            ),
            task_count=_integer(data.get("task_count"), "task_count"),
            episode_ids=_string_tuple(
                data.get("episode_ids"),
                "episode_ids",
            ),
            n_sessions=_integer(data.get("n_sessions"), "n_sessions"),
            required_secret_env=_string_tuple(
                data.get("required_secret_env"),
                "required_secret_env",
            ),
        )


@dataclass(frozen=True)
class AggregateStatus:
    run_identity: str
    planned_tasks: int
    completed_results: int
    missing_results: int
    non_complete_results: int
    report_directory: Path
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "report_directory": self.report_directory.as_posix(),
        }


def plan_qualification_run(
    dataset: Path,
    config_path: Path,
    run_directory: Path,
    *,
    allow_dirty: bool = False,
    force: bool = False,
) -> QualificationRunManifest:
    """Write an immutable, redacted run identity and its atomic task table."""
    repository_root = Path(__file__).resolve().parents[3]
    snapshot = current_repository_snapshot(repository_root)
    if snapshot.dirty and not allow_dirty:
        raise QualificationCliError(
            "Git worktree is dirty; commit changes or pass --allow-dirty"
        )
    config = load_qualification_config(config_path)
    specs = load_mem0_specs(dataset)
    if not specs:
        raise QualificationCliError("frozen dataset contains no episodes")
    session_counts = {spec.plan.n_sessions for spec in specs}
    if len(session_counts) != 1:
        raise QualificationCliError(
            f"episodes use inconsistent session counts: {sorted(session_counts)}"
        )
    dataset_manifest = dataset / "MANIFEST.json"
    uv_lock = repository_root / "uv.lock"
    dataset_manifest_sha256 = _sha256(dataset_manifest)
    dependency_lock_sha256 = _sha256(uv_lock)
    identity_payload = {
        "schema_version": QUALIFICATION_RUN_SCHEMA_VERSION,
        "code_commit": snapshot.commit,
        "code_dirty": snapshot.dirty,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "config_hash": config.config_hash,
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    run_identity = canonical_hash(identity_payload)
    tasks = build_qualification_tasks(
        config,
        episode_ids=tuple(spec.plan.episode_id for spec in specs),
        run_identity=run_identity,
    )
    manifest = QualificationRunManifest(
        schema_version=QUALIFICATION_RUN_SCHEMA_VERSION,
        run_identity=run_identity,
        experiment_id=config.experiment_id,
        code_commit=snapshot.commit,
        code_dirty=snapshot.dirty,
        code_ref=snapshot.ref,
        dataset_path=str(dataset.resolve()),
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_path=str(config_path.resolve()),
        config_hash=config.config_hash,
        dependency_lock_sha256=dependency_lock_sha256,
        task_count=len(tasks),
        episode_ids=tuple(spec.plan.episode_id for spec in specs),
        n_sessions=next(iter(session_counts)),
        required_secret_env=config.required_secret_env,
    )
    manifest_path = run_directory / "run_manifest.json"
    if manifest_path.is_file() and not force:
        existing = QualificationRunManifest.from_dict(_read_json(manifest_path))
        if existing != manifest:
            raise QualificationCliError(
                "existing run directory has a different run identity"
            )
        _validate_task_table(
            _read_tasks(run_directory / "tasks.jsonl"),
            tasks,
        )
        return existing
    if run_directory.exists():
        if not force and any(run_directory.iterdir()):
            raise QualificationCliError(
                "run directory is non-empty; pass --force to replace it"
            )
        if force:
            shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(run_directory / "run_config.yaml", config_path.read_bytes())
    _atomic_json(manifest_path, manifest.to_dict())
    _atomic_jsonl(
        run_directory / "tasks.jsonl",
        [_task_to_dict(task) for task in tasks],
    )
    return manifest


def execute_qualification_task(
    run_directory: Path,
    task_index: int,
    *,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> QualificationTaskResult | None:
    """Execute one identity-bound task, or validate it without live calls."""
    manifest, config, specs, tasks = _load_run_contract(run_directory)
    if task_index < 0 or task_index >= len(tasks):
        raise QualificationCliError(
            f"task index {task_index} is outside [0, {len(tasks) - 1}]"
        )
    task = tasks[task_index]
    if dry_run:
        return None
    env = dict(os.environ if environment is None else environment)
    require_live_gate(env)
    result_path = _task_result_path(run_directory, task)
    if result_path.is_file() and not force:
        return _read_task_result(result_path, manifest, task)
    storage = QualificationStorage(
        run_directory / "cells",
        run_identity=manifest.run_identity,
    )
    spec = specs[task.episode_id]
    isolation = TaskIsolation.for_task(
        task,
        storage.task_directory(task),
    )
    components = _live_components(
        task,
        isolation,
        spec=spec,
        config=config,
        environment=env,
    )
    result = run_qualification_task(
        task,
        spec,
        components=components,
        storage=storage,
        visible_k=config.retrieval.visible_k,
    )
    _write_task_result(result_path, manifest, task, result)
    return result


def run_qualification_matrix_cli(
    run_directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    keep_going: bool = False,
) -> AggregateStatus | None:
    """Execute planned tasks in order, stopping at the first non-complete result."""
    manifest, _, _, tasks = _load_run_contract(run_directory)
    if dry_run:
        for index in range(len(tasks)):
            execute_qualification_task(
                run_directory,
                index,
                environment=environment,
                dry_run=True,
            )
        return None
    require_live_gate(dict(os.environ if environment is None else environment))
    for index in range(len(tasks)):
        result = execute_qualification_task(
            run_directory,
            index,
            environment=environment,
            force=force,
        )
        if result is not None and result.status != "complete" and not keep_going:
            break
    status = aggregate_qualification_run(run_directory)
    if status.run_identity != manifest.run_identity:
        raise QualificationCliError("aggregate run identity changed")
    return status


def aggregate_qualification_run(
    run_directory: Path,
    *,
    output_directory: Path | None = None,
) -> AggregateStatus:
    """Aggregate every available portable task result into a validated report."""
    manifest, _, specs, tasks = _load_run_contract(run_directory)
    results: list[QualificationTaskResult] = []
    missing = 0
    for task in tasks:
        path = _task_result_path(run_directory, task)
        if path.is_file():
            results.append(_read_task_result(path, manifest, task))
        else:
            missing += 1
    report_directory = output_directory or run_directory / "report"
    matrix = QualificationMatrixResult(
        run_identity=manifest.run_identity,
        task_results=tuple(results),
    )
    write_qualification_report(
        matrix,
        specs,
        report_directory,
        run_metadata={
            "code_commit": manifest.code_commit,
            "code_dirty": manifest.code_dirty,
            "code_ref": manifest.code_ref,
            "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
            "config_hash": manifest.config_hash,
            "dependency_lock_sha256": manifest.dependency_lock_sha256,
            "planned_task_count": manifest.task_count,
            "missing_task_count": missing,
            "required_secret_env": list(manifest.required_secret_env),
        },
    )
    validation = validate_qualification_artifacts(
        report_directory,
        expected_run_identity=manifest.run_identity,
    )
    if not validation.ok:
        raise QualificationCliError(
            "generated report failed validation: "
            + "; ".join(validation.errors)
        )
    non_complete = sum(result.status != "complete" for result in results)
    return AggregateStatus(
        run_identity=manifest.run_identity,
        planned_tasks=len(tasks),
        completed_results=len(results),
        missing_results=missing,
        non_complete_results=non_complete,
        report_directory=report_directory,
        complete=missing == 0 and non_complete == 0,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Plan and execute the Mem0 long-horizon qualification.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="write an immutable run contract")
    _add_plan_args(plan)

    run_task = commands.add_parser(
        "run-task",
        help="execute one zero-based task index",
    )
    run_task.add_argument("--run-dir", required=True, type=Path)
    run_task.add_argument("--task-index", required=True, type=int)
    run_task.add_argument("--dry-run", action="store_true")
    run_task.add_argument("--force", action="store_true")
    run_task.add_argument("--json", type=Path)

    matrix = commands.add_parser(
        "run-matrix",
        help="execute the planned task matrix",
    )
    matrix.add_argument("--run-dir", required=True, type=Path)
    matrix.add_argument("--dry-run", action="store_true")
    matrix.add_argument("--force", action="store_true")
    matrix.add_argument("--keep-going", action="store_true")
    matrix.add_argument("--json", type=Path)

    aggregate = commands.add_parser(
        "aggregate",
        help="write metrics, traces, and scorecards",
    )
    aggregate.add_argument("--run-dir", required=True, type=Path)
    aggregate.add_argument("--out", type=Path)
    aggregate.add_argument("--json", type=Path)

    validate = commands.add_parser(
        "validate",
        help="validate report schemas, hashes, and trace ordering",
    )
    validate.add_argument("--report", required=True, type=Path)
    validate.add_argument("--run-identity")
    validate.add_argument("--json", type=Path)

    preflight = commands.add_parser(
        "preflight",
        help="run ordered repository and optional live service gates",
    )
    preflight.add_argument(
        "--dataset",
        default=Path("runs/vertical/software_mem0_v2"),
        type=Path,
    )
    preflight.add_argument(
        "--config",
        default=Path("configs/experiments/mem0_qualification.yaml"),
        type=Path,
    )
    preflight.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LHMSB_DATA_ROOT", "/data/lhmsb")),
    )
    preflight.add_argument("--repository-only", action="store_true")
    preflight.add_argument("--allow-dirty", action="store_true")
    preflight.add_argument("--json", type=Path)

    smoke = commands.add_parser(
        "smoke",
        help="plan and run a frozen four-session qualification dataset",
    )
    _add_plan_args(smoke)
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--keep-going", action="store_true")
    return parser


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", type=Path)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one qualification command and return a stable exit status."""
    args = _build_parser().parse_args(argv)
    command = str(args.command)
    try:
        if command == "plan":
            manifest = plan_qualification_run(
                args.dataset,
                args.config,
                args.out,
                allow_dirty=args.allow_dirty,
                force=args.force,
            )
            _maybe_write_json(args.json, manifest.to_dict())
            print(
                f"planned {manifest.task_count} task(s) "
                f"for run {manifest.run_identity} -> {args.out}"
            )
            return 0
        if command == "run-task":
            result = execute_qualification_task(
                args.run_dir,
                args.task_index,
                dry_run=args.dry_run,
                force=args.force,
            )
            payload = (
                {
                    "dry_run": True,
                    "task_index": args.task_index,
                }
                if result is None
                else asdict(result)
            )
            _maybe_write_json(args.json, payload)
            if result is None:
                print(f"task {args.task_index} dry-run passed")
                return 0
            print(f"task {args.task_index} status={result.status}")
            return 0 if result.status == "complete" else 1
        if command == "run-matrix":
            status = run_qualification_matrix_cli(
                args.run_dir,
                dry_run=args.dry_run,
                force=args.force,
                keep_going=args.keep_going,
            )
            payload = (
                {"dry_run": True}
                if status is None
                else status.to_dict()
            )
            _maybe_write_json(args.json, payload)
            if status is None:
                print("matrix dry-run passed")
                return 0
            print(
                f"aggregated {status.completed_results}/"
                f"{status.planned_tasks} task result(s)"
            )
            return 0 if status.complete else 1
        if command == "aggregate":
            status = aggregate_qualification_run(
                args.run_dir,
                output_directory=args.out,
            )
            _maybe_write_json(args.json, status.to_dict())
            print(f"report -> {status.report_directory}")
            return 0 if status.complete else 1
        if command == "validate":
            validation_report = validate_qualification_artifacts(
                args.report,
                expected_run_identity=args.run_identity,
            )
            _maybe_write_json(args.json, validation_report.to_dict())
            if validation_report.ok:
                print(
                    f"validated "
                    f"{validation_report.checked_artifacts} artifact(s)"
                )
                return 0
            print(
                "validation FAILED: "
                + "; ".join(validation_report.errors),
                file=sys.stderr,
            )
            return 1
        if command == "preflight":
            repository_root = Path(__file__).resolve().parents[3]
            preflight_report = run_preflight(
                PreflightContext(
                    repository_root=repository_root,
                    dataset_root=args.dataset.resolve(),
                    config_path=args.config.resolve(),
                    data_root=args.data_root.resolve(),
                    allow_dirty=args.allow_dirty,
                    repository_only=args.repository_only,
                    environment=dict(os.environ),
                ),
                output_json=args.json,
            )
            if preflight_report.ok:
                print(
                    "preflight passed "
                    f"({len(preflight_report.checks)} gate records)"
                )
                return 0
            print(
                f"preflight FAILED at {preflight_report.stopped_at}",
                file=sys.stderr,
            )
            return 1
        if command == "smoke":
            manifest = plan_qualification_run(
                args.dataset,
                args.config,
                args.out,
                allow_dirty=args.allow_dirty,
                force=args.force,
            )
            if manifest.n_sessions > 4:
                raise QualificationCliError(
                    "smoke requires a frozen dataset with at most four sessions"
                )
            status = run_qualification_matrix_cli(
                args.out,
                dry_run=args.dry_run,
                keep_going=args.keep_going,
            )
            payload = (
                {"dry_run": True, "run_identity": manifest.run_identity}
                if status is None
                else status.to_dict()
            )
            _maybe_write_json(args.json, payload)
            return 0 if status is None or status.complete else 1
        raise QualificationCliError(f"unknown command: {command}")
    except (
        QualificationCliError,
        QualificationConfigError,
        QualificationRunError,
        QualificationStorageError,
        PreflightError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"{command} FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


def _load_run_contract(
    run_directory: Path,
) -> tuple[
    QualificationRunManifest,
    QualificationConfig,
    dict[str, SoftwareMem0VerticalSpec],
    tuple[QualificationTask, ...],
]:
    manifest = QualificationRunManifest.from_dict(
        _read_json(run_directory / "run_manifest.json")
    )
    if manifest.schema_version != QUALIFICATION_RUN_SCHEMA_VERSION:
        raise QualificationCliError(
            f"unsupported run schema: {manifest.schema_version}"
        )
    config = load_qualification_config(Path(manifest.config_path))
    specs_tuple = load_mem0_specs(Path(manifest.dataset_path))
    specs = {spec.plan.episode_id: spec for spec in specs_tuple}
    tasks = _read_tasks(run_directory / "tasks.jsonl")
    expected = build_qualification_tasks(
        config,
        episode_ids=manifest.episode_ids,
        run_identity=manifest.run_identity,
    )
    _validate_task_table(tasks, expected)
    if manifest.task_count != len(tasks):
        raise QualificationCliError(
            "run identity task count does not match tasks.jsonl"
        )
    snapshot = current_repository_snapshot(
        Path(__file__).resolve().parents[3]
    )
    if snapshot.commit != manifest.code_commit:
        raise QualificationCliError(
            "current code commit does not match the planned run identity"
        )
    if config.config_hash != manifest.config_hash:
        raise QualificationCliError(
            "configuration hash does not match the planned run identity"
        )
    if _sha256(Path(manifest.dataset_path) / "MANIFEST.json") != (
        manifest.dataset_manifest_sha256
    ):
        raise QualificationCliError(
            "dataset manifest hash does not match the planned run identity"
        )
    return manifest, config, specs, tasks


def _live_components(
    task: QualificationTask,
    isolation: TaskIsolation,
    *,
    spec: SoftwareMem0VerticalSpec,
    config: QualificationConfig,
    environment: Mapping[str, str],
) -> TaskComponents:
    policy_profile = next(
        profile
        for profile in config.policy_profiles
        if profile.profile_id == task.policy_profile_id
    )
    effective_policy = _effective_policy(policy_profile, environment)
    try:
        policy_key = environment[policy_profile.api_key_env]
    except KeyError as exc:
        raise PreflightError(
            "provider_auth_failure",
            f"missing {policy_profile.api_key_env}",
        ) from exc
    policy = HttpPolicyClient(effective_policy, api_key=policy_key)
    legacy_spec = SoftwareVerticalSpec(
        plan=spec.plan,
        package_files=spec.package_files,
        hidden_tests=spec.hidden_tests,
        actions=spec.actions,
        surface_hash=spec.surface_hash,
    )
    checker = SoftwareVerticalChecker(legacy_spec)
    memory = None
    reranker = None
    if task.condition.startswith("mem0_"):
        profile = (
            config.controlled_mem0
            if task.condition == "mem0_controlled"
            else config.native_mem0
        )
        isolation.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        live_config = build_mem0_live_config(
            profile,
            policy=effective_policy,
            internal_llm_api_key=policy_key,
            native_openai_api_key=environment.get("OPENAI_API_KEY", ""),
            qdrant_url=environment.get(
                "LHMSB_QDRANT_URL",
                "http://qdrant:6333",
            ),
            collection_name=isolation.collection_name,
            history_db_path=isolation.history_db_path,
            embedding_base_url=environment.get(
                "LHMSB_EMBEDDING_URL",
                "http://embedding:80",
            ),
            embedding_dimension=config.retrieval.embedding_dimension,
        )
        memory = Mem0QualificationAdapter.create_live(
            live_config,
            user_id=isolation.user_id,
            run_id=isolation.run_id,
            candidate_k=config.retrieval.candidate_k,
        )
        if task.condition == "mem0_controlled":
            reranker = RerankerClient(
                environment.get(
                    "LHMSB_RERANKER_URL",
                    "http://reranker:80",
                ),
                model=config.retrieval.reranker_model,
                revision=config.retrieval.reranker_revision,
            )
    return TaskComponents(
        policy=policy,
        checker=checker,
        memory=memory,
        reranker=reranker,
    )


def _effective_policy(
    profile: PolicyProfile,
    environment: Mapping[str, str],
) -> PolicyProfile:
    override = (
        environment.get(profile.endpoint_override_env)
        if profile.endpoint_override_env
        else None
    )
    return replace(profile, endpoint=override or profile.endpoint)


def _write_task_result(
    path: Path,
    manifest: QualificationRunManifest,
    task: QualificationTask,
    result: QualificationTaskResult,
) -> None:
    result_payload = asdict(result)
    envelope = {
        "schema_version": QUALIFICATION_RUN_SCHEMA_VERSION,
        "run_identity": manifest.run_identity,
        "task_id": task.task_id,
        "task_payload_hash": task.task_payload_hash,
        "result_hash": canonical_hash(result_payload),
        "result": result_payload,
    }
    _atomic_json(path, envelope)


def _read_task_result(
    path: Path,
    manifest: QualificationRunManifest,
    task: QualificationTask,
) -> QualificationTaskResult:
    envelope = _read_json(path)
    if envelope.get("run_identity") != manifest.run_identity:
        raise QualificationCliError(
            f"task result run identity mismatch: {task.task_id}"
        )
    if envelope.get("task_payload_hash") != task.task_payload_hash:
        raise QualificationCliError(
            f"task result payload identity mismatch: {task.task_id}"
        )
    payload = _mapping(envelope.get("result"), "task result")
    if envelope.get("result_hash") != canonical_hash(payload):
        raise QualificationCliError(
            f"task result hash mismatch: {task.task_id}"
        )
    result = qualification_task_result_from_dict(payload)
    if result.task_id != task.task_id:
        raise QualificationCliError(
            f"task result ID mismatch: {task.task_id}"
        )
    return result


def _task_result_path(
    run_directory: Path,
    task: QualificationTask,
) -> Path:
    return run_directory / "results" / f"{task.task_id}.json"


def _task_to_dict(task: QualificationTask) -> dict[str, object]:
    return {
        **asdict(task),
        "scored_conditions": [
            asdict(item) for item in task.scored_conditions
        ],
    }


def _task_from_dict(data: Mapping[str, object]) -> QualificationTask:
    return QualificationTask(
        task_index=_integer(data.get("task_index"), "task_index"),
        task_id=_text(data.get("task_id"), "task_id"),
        episode_id=_text(data.get("episode_id"), "episode_id"),
        policy_profile_id=_text(
            data.get("policy_profile_id"),
            "policy_profile_id",
        ),
        condition=cast(
            QualificationCondition,
            _text(data.get("condition"), "condition"),
        ),
        store_namespace=_text(
            data.get("store_namespace"),
            "store_namespace",
        ),
        run_identity=_text(data.get("run_identity"), "run_identity"),
        task_payload_hash=_text(
            data.get("task_payload_hash"),
            "task_payload_hash",
        ),
        scored_conditions=tuple(
            ScoredCondition(
                result_id=_text(item.get("result_id"), "result_id"),
                condition=_text(item.get("condition"), "condition"),
                readout=cast(
                    ReadoutKind,
                    _text(item.get("readout"), "readout"),
                ),
            )
            for item in _mapping_sequence(
                data.get("scored_conditions"),
                "scored_conditions",
            )
        ),
    )


def _read_tasks(path: Path) -> tuple[QualificationTask, ...]:
    tasks = tuple(_task_from_dict(row) for row in _read_jsonl(path))
    for index, task in enumerate(tasks):
        if task.task_index != index:
            raise QualificationCliError(
                f"task index sequence mismatch at row {index}"
            )
    return tasks


def _validate_task_table(
    actual: tuple[QualificationTask, ...],
    expected: tuple[QualificationTask, ...],
) -> None:
    if actual != expected:
        raise QualificationCliError(
            "task table identity does not match the planned run identity"
        )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationCliError(f"cannot read JSON {path}: {exc}") from exc
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QualificationCliError(f"cannot read JSONL {path}: {exc}") from exc
    output: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            output.append(_mapping(json.loads(line), f"{path}:{number}"))
        except json.JSONDecodeError as exc:
            raise QualificationCliError(
                f"invalid JSONL {path}:{number}: {exc}"
            ) from exc
    return tuple(output)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    payload = "".join(
        json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
        for row in rows
    )
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _maybe_write_json(path: Path | None, value: object) -> None:
    if path is not None:
        _atomic_json(path, value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise QualificationCliError(f"{label} must be an object")
    return {str(key): child for key, child in value.items()}


def _mapping_sequence(
    value: object,
    label: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QualificationCliError(f"{label} must be an array")
    return tuple(_mapping(item, label) for item in value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QualificationCliError(f"{label} must be an array")
    return tuple(str(item) for item in value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationCliError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise QualificationCliError(f"{label} must be an integer")
    return int(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise QualificationCliError(f"{label} must be a boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise QualificationCliError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AggregateStatus",
    "QUALIFICATION_RUN_SCHEMA_VERSION",
    "QualificationCliError",
    "QualificationRunManifest",
    "aggregate_qualification_run",
    "execute_qualification_task",
    "main",
    "plan_qualification_run",
    "run_qualification_matrix_cli",
]
