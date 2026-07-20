"""Schema-v2 multisystem orchestration commands.

The legacy Mem0 CLI remains in :mod:`lhmsb.qualification.cli`.  This module is
the server-facing two-stage driver for the controlled matrix: it writes a
stable preparation/template contract, binds verified prefix artifacts, and
then executes read-only policy cells.  Every command is independently
restartable and all persisted records are canonical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from lhmsb.datasets.mem0_stateful_pipeline import Mem0StatefulDatasetError
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.interventions import (
    CausalUseLabel,
    EffectDirection,
    InterventionKind,
    MemoryRole,
)
from lhmsb.qualification.config import (
    build_evaluation_task_templates,
    build_preparation_tasks,
    canonical_hash,
    finalize_evaluation_plan,
    load_qualification_config,
)
from lhmsb.qualification.evaluate import EvaluationTaskResult
from lhmsb.qualification.prefix import MemoryPrefixArtifact
from lhmsb.qualification.preflight import load_mem0_specs
from lhmsb.qualification.providers import PolicyRole
from lhmsb.qualification.runner import QualificationMatrixResult
from lhmsb.qualification.schema import (
    EvaluationTask,
    EvaluationTaskTemplate,
    PreparationTask,
    QualificationCondition,
    ReadoutKind,
    SystemBackend,
    SystemsQualificationConfig,
)
from lhmsb.qualification.storage import QualificationStorage, QualificationStorageError


class MultisystemCliError(RuntimeError):
    """Invalid schema-v2 run state; never silently repair a run."""


SCHEMA_VERSION = 2
_SYSTEM_COMMANDS = frozenset(
    {
        "plan-systems",
        "prepare-task",
        "finalize-evaluation-plan",
        "evaluate-task",
        "run-evaluation-matrix",
        "aggregate-systems",
        "validate-systems",
        "preflight-systems",
        "smoke-systems",
    }
)


def is_multisystem_command(command: str) -> bool:
    return command in _SYSTEM_COMMANDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_bytes(value) + b"\n")


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    body = b"".join(_canonical_bytes(value) + b"\n" for value in values)
    _atomic_write(path, body)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultisystemCliError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MultisystemCliError(f"JSON document must be an object: {path}")
    return {str(key): child for key, child in value.items()}


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MultisystemCliError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MultisystemCliError(f"invalid JSONL {path}:{index}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise MultisystemCliError(f"JSONL row must be an object: {path}:{index}")
        rows.append({str(key): child for key, child in raw.items()})
    return tuple(rows)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MultisystemCliError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultisystemCliError(f"{label} must be an integer")
    return value


def _git_identity() -> tuple[str, bool, str]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        ref = subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MultisystemCliError(f"cannot determine git identity: {exc}") from exc
    return commit, dirty, ref


_HASH_HEX = frozenset("0123456789abcdef")


def _manifest_hash(
    environment: Mapping[str, str],
    *,
    path_keys: Sequence[str],
    hash_keys: Sequence[str],
    unavailable_label: str,
) -> str:
    """Resolve a non-secret manifest identity from an environment mapping.

    Server bootstrap writes immutable manifests under the configured data root,
    while local tests commonly provide an explicit digest.  Accept both forms
    but never include a manifest path's contents other than its SHA-256 digest
    in a run identity.  A deterministic unavailable value keeps dry-run and
    repository-only plans reproducible without pretending that a runtime was
    verified.
    """
    for key in hash_keys:
        value = environment.get(key)
        if value is None or not value:
            continue
        if len(value) != 64 or any(char not in _HASH_HEX for char in value):
            raise MultisystemCliError(f"{key} must be a lowercase SHA-256 digest")
        return value
    for key in path_keys:
        value = environment.get(key)
        if value is None or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            raise MultisystemCliError(f"manifest path from {key} is not a file")
        return _sha256(path)
    return hashlib.sha256(unavailable_label.encode("utf-8")).hexdigest()


def _runtime_manifest_hash(environment: Mapping[str, str]) -> str:
    return _manifest_hash(
        environment,
        path_keys=(
            "LHMSB_RUNTIME_MANIFEST",
            "LHMSB_RUNTIME_MANIFEST_PATH",
            "LHMSB_SERVICE_MANIFEST",
        ),
        hash_keys=("LHMSB_RUNTIME_MANIFEST_HASH", "LHMSB_SERVICE_MANIFEST_HASH"),
        unavailable_label="runtime-manifest-unavailable",
    )


def _config(path: Path) -> SystemsQualificationConfig:
    loaded = load_qualification_config(path.resolve())
    if not isinstance(loaded, SystemsQualificationConfig):
        raise MultisystemCliError("schema-v2 command requires a schema-v2 config")
    return loaded


def _specs(dataset: Path) -> tuple[SoftwareMem0VerticalSpec, ...]:
    try:
        return load_mem0_specs(dataset.resolve())
    except (Mem0StatefulDatasetError, OSError, KeyError, TypeError, ValueError) as exc:
        raise MultisystemCliError(f"cannot load frozen dataset: {exc}") from exc


def _prep_to_dict(task: PreparationTask) -> dict[str, object]:
    return task.to_dict()


def _prep_from_dict(raw: Mapping[str, object]) -> PreparationTask:
    return PreparationTask(
        task_index=_integer(raw.get("task_index"), "task_index"),
        task_id=_text(raw.get("task_id"), "task_id"),
        episode_id=_text(raw.get("episode_id"), "episode_id"),
        backend=cast(SystemBackend, _text(raw.get("backend"), "backend")),
        profile_id=_text(raw.get("profile_id"), "profile_id"),
        run_identity=_text(raw.get("run_identity"), "run_identity"),
        config_hash=_text(raw.get("config_hash"), "config_hash"),
        task_payload_hash=_text(raw.get("task_payload_hash"), "task_payload_hash"),
    )


def _template_to_dict(task: EvaluationTaskTemplate) -> dict[str, object]:
    return task.to_dict()


def _template_from_dict(raw: Mapping[str, object]) -> EvaluationTaskTemplate:
    from lhmsb.qualification.schema import ScoredCondition

    scored_raw = raw.get("scored_conditions")
    if not isinstance(scored_raw, Sequence) or isinstance(scored_raw, (str, bytes)):
        raise MultisystemCliError("scored_conditions must be an array")
    scored = tuple(
        ScoredCondition(
            result_id=_text(item.get("result_id"), "result_id"),
            condition=_text(item.get("condition"), "condition"),
            readout=cast(ReadoutKind, _text(item.get("readout"), "readout")),
        )
        for item in scored_raw
        if isinstance(item, Mapping)
    )
    prefix_backend = raw.get("prefix_backend")
    return EvaluationTaskTemplate(
        task_index=_integer(raw.get("task_index"), "task_index"),
        task_id=_text(raw.get("task_id"), "task_id"),
        episode_id=_text(raw.get("episode_id"), "episode_id"),
        policy_profile_id=_text(raw.get("policy_profile_id"), "policy_profile_id"),
        condition=cast(
            QualificationCondition,
            _text(raw.get("condition"), "condition"),
        ),
        run_identity=_text(raw.get("run_identity"), "run_identity"),
        config_hash=_text(raw.get("config_hash"), "config_hash"),
        task_payload_hash=_text(raw.get("task_payload_hash"), "task_payload_hash"),
        scored_conditions=scored,
        prefix_backend=(
            cast(SystemBackend, prefix_backend)
            if prefix_backend is not None
            else None
        ),
    )


def _task_to_dict(task: EvaluationTask) -> dict[str, object]:
    return task.to_dict()


def _task_from_dict(raw: Mapping[str, object]) -> EvaluationTask:
    from lhmsb.qualification.schema import ScoredCondition

    scored_raw = raw.get("scored_conditions")
    if not isinstance(scored_raw, Sequence) or isinstance(scored_raw, (str, bytes)):
        raise MultisystemCliError("scored_conditions must be an array")
    scored = tuple(
        ScoredCondition(
            result_id=_text(item.get("result_id"), "result_id"),
            condition=_text(item.get("condition"), "condition"),
            readout=cast(ReadoutKind, _text(item.get("readout"), "readout")),
        )
        for item in scored_raw
        if isinstance(item, Mapping)
    )
    prefix_backend = raw.get("prefix_backend")
    return EvaluationTask(
        task_index=_integer(raw.get("task_index"), "task_index"),
        task_id=_text(raw.get("task_id"), "task_id"),
        episode_id=_text(raw.get("episode_id"), "episode_id"),
        policy_profile_id=_text(raw.get("policy_profile_id"), "policy_profile_id"),
        condition=cast(
            QualificationCondition,
            _text(raw.get("condition"), "condition"),
        ),
        prefix_artifact_hash=_text(raw.get("prefix_artifact_hash"), "prefix_artifact_hash"),
        run_identity=_text(raw.get("run_identity"), "run_identity"),
        config_hash=_text(raw.get("config_hash"), "config_hash"),
        task_payload_hash=_text(raw.get("task_payload_hash"), "task_payload_hash"),
        scored_conditions=scored,
        prefix_backend=(
            cast(SystemBackend, prefix_backend)
            if prefix_backend is not None
            else None
        ),
    )


def plan_systems_run(
    dataset: Path,
    config_path: Path,
    run_directory: Path,
    *,
    allow_dirty: bool = False,
    force: bool = False,
    n_sessions: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Write the immutable Stage-A preparation/template contract."""
    config = _config(config_path)
    specs = _specs(dataset)
    if not specs:
        raise MultisystemCliError("frozen dataset contains no episodes")
    if n_sessions is not None and any(spec.plan.n_sessions != n_sessions for spec in specs):
        raise MultisystemCliError("dataset session count does not match --n-sessions")
    commit, dirty, ref = _git_identity()
    if dirty and not allow_dirty:
        raise MultisystemCliError("Git worktree is dirty; pass --allow-dirty")
    env = dict(os.environ if environment is None else environment)
    runtime_manifest_hash = _runtime_manifest_hash(env)
    model_bundle_hash = _model_files_hash(env)
    manifest_path = run_directory / "run_manifest.json"
    dataset_manifest = dataset.resolve() / "MANIFEST.json"
    if not dataset_manifest.is_file():
        raise MultisystemCliError(f"missing dataset manifest: {dataset_manifest}")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": commit,
        "code_dirty": dirty,
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "config_hash": config.config_hash,
        "source_lock_hash": config.source_lock_hash,
        "runtime_manifest_hash": runtime_manifest_hash,
        "model_bundle_hash": model_bundle_hash,
        # Prefix artifacts use the historical ``model_files_hash`` name.
        "model_files_hash": model_bundle_hash,
        "policy_profiles": [asdict(item) for item in config.policy_profiles],
        "writer_profile": asdict(config.writer_profile),
        "retrieval": asdict(config.retrieval),
    }
    run_identity = canonical_hash(identity)
    preparations = build_preparation_tasks(
        config,
        episode_ids=tuple(spec.plan.episode_id for spec in specs),
        run_identity=run_identity,
    )
    templates = build_evaluation_task_templates(
        config,
        episode_ids=tuple(spec.plan.episode_id for spec in specs),
        run_identity=run_identity,
    )
    payload = {
        **identity,
        "run_identity": run_identity,
        "experiment_id": config.experiment_id,
        "dataset_path": str(dataset.resolve()),
        "config_path": str(config_path.resolve()),
        "dataset_release": config.dataset_release,
        "code_ref": ref,
        "task_count": len(preparations),
        "preparation_task_count": len(preparations),
        "evaluation_template_count": len(templates),
        "scored_cell_count": sum(len(item.scored_conditions) for item in templates),
        "episode_ids": [spec.plan.episode_id for spec in specs],
        "n_sessions": specs[0].plan.n_sessions,
        "required_secret_env": list(config.required_secret_env),
        "finalized": False,
    }
    if manifest_path.is_file() and not force:
        existing = _read_json(manifest_path)
        if existing.get("run_identity") != run_identity:
            raise MultisystemCliError("existing run has a different identity; pass --force")
        return existing
    if run_directory.exists() and force:
        shutil.rmtree(run_directory)
    if run_directory.exists() and any(run_directory.iterdir()):
        raise MultisystemCliError("run directory is non-empty; pass --force")
    run_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(run_directory / "run_config.yaml", config_path.read_bytes())
    _atomic_json(manifest_path, payload)
    _atomic_jsonl(
        run_directory / "prepare_tasks.jsonl",
        [_prep_to_dict(item) for item in preparations],
    )
    _atomic_jsonl(
        run_directory / "evaluation_task_templates.jsonl",
        [_template_to_dict(item) for item in templates],
    )
    return payload


def _load_contract(
    run_directory: Path,
) -> tuple[
    dict[str, object],
    SystemsQualificationConfig,
    tuple[SoftwareMem0VerticalSpec, ...],
    tuple[PreparationTask, ...],
    tuple[EvaluationTaskTemplate, ...],
]:
    manifest = _read_json(run_directory / "run_manifest.json")
    if _integer(manifest.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise MultisystemCliError("unsupported schema-v2 run manifest")
    config = _config(Path(_text(manifest.get("config_path"), "config_path")))
    specs = _specs(Path(_text(manifest.get("dataset_path"), "dataset_path")))
    preparations = tuple(
        _prep_from_dict(item)
        for item in _read_jsonl(run_directory / "prepare_tasks.jsonl")
    )
    templates = tuple(
        _template_from_dict(item)
        for item in _read_jsonl(run_directory / "evaluation_task_templates.jsonl")
    )
    if preparations and manifest.get("run_identity") != preparations[0].run_identity:
        raise MultisystemCliError("preparation task run identity mismatch")
    return manifest, config, specs, preparations, templates


def _load_tasks(run_directory: Path) -> tuple[EvaluationTask, ...]:
    path = run_directory / "tasks.jsonl"
    if not path.is_file():
        raise MultisystemCliError("evaluation plan is not finalized")
    tasks = tuple(_task_from_dict(item) for item in _read_jsonl(path))
    if tuple(item.task_index for item in tasks) != tuple(range(len(tasks))):
        raise MultisystemCliError("evaluation task indices are not stable")
    return tasks


def finalize_systems_run(run_directory: Path) -> dict[str, object]:
    manifest, config, specs, preparations, templates = _load_contract(run_directory)
    storage = QualificationStorage(
        run_directory / "cells",
        run_identity=_text(manifest.get("run_identity"), "run_identity"),
    )
    artifact_map: dict[str, MemoryPrefixArtifact] = {}
    for prep in preparations:
        try:
            artifact = storage.verify_prefix_artifact(prep)
        except QualificationStorageError as exc:
            raise MultisystemCliError(f"prefix {prep.task_id} is not ready: {exc}") from exc
        artifact_map[f"{prep.episode_id}--{prep.backend}"] = artifact
    tasks = finalize_evaluation_plan(
        config,
        templates,
        artifact_map,
        run_identity=_text(manifest.get("run_identity"), "run_identity"),
    )
    _atomic_jsonl(run_directory / "tasks.jsonl", [_task_to_dict(item) for item in tasks])
    updated = dict(manifest)
    updated.update(
        {
            "finalized": True,
            "evaluation_task_count": len(tasks),
            "scored_cell_count": sum(len(item.scored_conditions) for item in tasks),
        }
    )
    _atomic_json(run_directory / "run_manifest.json", updated)
    return updated


def _model_files_hash(environment: Mapping[str, str]) -> str:
    return _manifest_hash(
        environment,
        path_keys=(
            "LHMSB_MODEL_BUNDLE_MANIFEST",
            "LHMSB_MODEL_BUNDLE_MANIFEST_PATH",
            "LHMSB_MODEL_FILE_MANIFEST",
        ),
        hash_keys=(
            "LHMSB_MODEL_BUNDLE_HASH",
            "LHMSB_MODEL_FILES_HASH",
            "LHMSB_MODEL_FILE_MANIFEST_HASH",
        ),
        unavailable_label="model-files-unavailable",
    )


def _live_data_root(
    run_directory: Path,
    config: SystemsQualificationConfig,
    environment: Mapping[str, str],
) -> Path:
    """Resolve the native data root without a container-specific fallback."""
    configured = environment.get(config.data_root_env) or environment.get(
        "LHMSB_DATA_ROOT"
    )
    if configured:
        return Path(configured).expanduser()
    # A local fallback is useful for unit tests and repository-only smoke runs;
    # production Slurm jobs must export the shared persistent root explicitly.
    return run_directory.parent / ".lhmsb-data"


def _prepare_live_task(
    run_directory: Path,
    task_index: int,
    *,
    environment: Mapping[str, str] | None = None,
    force: bool = False,
) -> MemoryPrefixArtifact:
    from lhmsb.qualification.factory import build_preparation_components
    from lhmsb.qualification.prepare import prepare_prefix

    manifest, config, specs, preparations, _ = _load_contract(run_directory)
    if task_index < 0 or task_index >= len(preparations):
        raise MultisystemCliError("preparation task index is out of range")
    task = preparations[task_index]
    spec = next((item for item in specs if item.plan.episode_id == task.episode_id), None)
    if spec is None:
        raise MultisystemCliError(f"missing episode for preparation task {task.task_id}")
    storage = QualificationStorage(
        run_directory / "cells",
        run_identity=_text(manifest.get("run_identity"), "run_identity"),
    )
    env = dict(os.environ if environment is None else environment)
    runtime_manifest_hash = _runtime_manifest_hash(env)
    model_bundle_hash = _model_files_hash(env)
    expected_runtime_hash = manifest.get("runtime_manifest_hash")
    if expected_runtime_hash is not None and expected_runtime_hash != runtime_manifest_hash:
        raise MultisystemCliError(
            "runtime manifest identity differs from the immutable run plan"
        )
    expected_model_hash = manifest.get("model_bundle_hash", manifest.get("model_files_hash"))
    if expected_model_hash is not None and expected_model_hash != model_bundle_hash:
        raise MultisystemCliError(
            "model bundle identity differs from the immutable run plan"
        )
    profile = config.system_profiles[task.backend]
    source_commit = getattr(profile, "source_commit", "repository")
    expected_identity = {
        "run_identity": task.run_identity,
        "config_hash": config.config_hash,
        "dataset_release": config.dataset_release,
        "dataset_manifest_hash": _text(
            manifest.get("dataset_manifest_sha256"),
            "dataset_manifest_sha256",
        ),
        "embedding_profile_id": config.retrieval.embedding_profile_id,
        "reranker_profile_id": config.retrieval.reranker_profile_id,
        "writer_profile_id": (
            None if task.backend == "flat_retrieval" else config.writer_profile.profile_id
        ),
        "source_commit": source_commit,
        "model_files_hash": model_bundle_hash,
    }
    if not force:
        existing = storage.load_prefix_artifact(task, expected=expected_identity)
        if existing is not None:
            return existing
    components = build_preparation_components(
        task,
        spec,
        config,
        data_root=_live_data_root(run_directory, config, env),
        environment=env,
    )
    try:
        return prepare_prefix(
            task,
            spec,
            components.runtime,
            components.reranker,
            storage,
            config_hash=config.config_hash,
            dataset_manifest_hash=_text(
                manifest.get("dataset_manifest_sha256"),
                "dataset_manifest_sha256",
            ),
            embedding_profile_id=config.retrieval.embedding_profile_id,
            reranker_profile_id=config.retrieval.reranker_profile_id,
            writer_profile_id=(
                None if task.backend == "flat_retrieval" else config.writer_profile.profile_id
            ),
            source_commit=source_commit,
            model_files_hash=model_bundle_hash,
            dataset_release=config.dataset_release,
            visible_k=config.retrieval.visible_k,
        )
    finally:
        close = getattr(components.reranker, "close", None)
        if callable(close):
            close()


def _result_path(run_directory: Path, task: EvaluationTask) -> Path:
    return run_directory / "results" / f"{task.task_id}.json"


def _evaluate_live_task(
    run_directory: Path,
    task_index: int,
    *,
    environment: Mapping[str, str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    from lhmsb.qualification.evaluate import BehaviorChecker, evaluate_task
    from lhmsb.qualification.factory import build_checker, build_policy_client
    from lhmsb.qualification.providers import PolicyClient

    manifest, config, specs, preparations, _ = _load_contract(run_directory)
    tasks = _load_tasks(run_directory)
    if task_index < 0 or task_index >= len(tasks):
        raise MultisystemCliError("evaluation task index is out of range")
    task = tasks[task_index]
    path = _result_path(run_directory, task)
    if path.is_file() and not force:
        return _read_json(path)
    spec = next((item for item in specs if item.plan.episode_id == task.episode_id), None)
    if spec is None:
        raise MultisystemCliError(f"missing episode for evaluation task {task.task_id}")
    storage = QualificationStorage(
        run_directory / "cells",
        run_identity=_text(manifest.get("run_identity"), "run_identity"),
    )
    artifact: MemoryPrefixArtifact | None = None
    if task.prefix_backend is not None:
        prep = next(
            item
            for item in preparations
            if item.episode_id == task.episode_id and item.backend == task.prefix_backend
        )
        artifact = storage.verify_prefix_artifact(prep)
    policy_profile = next(
        item for item in config.policy_profiles if item.profile_id == task.policy_profile_id
    )
    policy = cast(PolicyClient, build_policy_client(policy_profile, environment=environment))
    try:
        result = evaluate_task(
            task,
            spec,
            prefix_artifact=artifact,
            policy=policy,
            checker=cast(BehaviorChecker, build_checker(spec)),
            sampling=config.sampling,
            full_context_max_chars=config.full_context_max_chars,
            visible_k=config.retrieval.visible_k,
            max_output_tokens=config.sampling.max_output_tokens,
        )
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_identity": manifest["run_identity"],
        "task_id": task.task_id,
        "task_payload_hash": task.task_payload_hash,
        "result_hash": result.result_hash,
        "result": result.to_dict(),
    }
    _atomic_json(path, payload)
    return payload


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MultisystemCliError("result sequence field must be an array")
    return tuple(str(item) for item in value)


def _mapping_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MultisystemCliError(f"{label} must be an object")
    return value


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value))


def _evaluation_result_from_dict(raw: Mapping[str, object]) -> EvaluationTaskResult:
    """Restore one schema-v2 result for report/metric aggregation.

    Evaluation workers persist canonical JSON envelopes so they can run in
    separate Slurm jobs.  Aggregation deliberately reconstructs the immutable
    evaluator DTOs here instead of recomputing behavior from a score-only
    projection; this preserves retrieval, intervention, usage, and drift
    metrics in the report layer.
    """
    from lhmsb.families.software.vertical_checker import BehaviorResult
    from lhmsb.longhorizon.interventions import CausalUseResult, ContinuationOutcome
    from lhmsb.longhorizon.public_surface import PublicActionOption
    from lhmsb.qualification.evaluate import (
        EvaluationCall,
        EvaluationConditionResult,
        EvaluationIntervention,
        EvaluationSCEUResult,
        EvaluationTaskResult,
    )
    from lhmsb.qualification.providers import (
        PolicyMessage,
        PolicyRequest,
        PolicyResponse,
        PolicyUsage,
    )

    def outcome(value: object) -> ContinuationOutcome:
        data = _mapping_value(value, "continuation outcome")
        return ContinuationOutcome(
            action_id=_text(data.get("action_id"), "outcome.action_id"),
            behavior_score=_as_float(data.get("behavior_score")),
            is_correct=bool(data.get("is_correct", False)),
            violated_state_ids=_string_tuple(data.get("violated_state_ids")),
            drift_flags=_string_tuple(data.get("drift_flags")),
        )

    def checker(value: object) -> BehaviorResult:
        data = _mapping_value(value, "checker result")
        raw_metadata = data.get("metadata", ())
        metadata: list[tuple[str, str]] = []
        if isinstance(raw_metadata, Sequence) and not isinstance(raw_metadata, (str, bytes)):
            for pair in raw_metadata:
                if (
                    isinstance(pair, Sequence)
                    and not isinstance(pair, (str, bytes))
                    and len(pair) == 2
                ):
                    metadata.append((str(pair[0]), str(pair[1])))
        return BehaviorResult(
            score=_as_float(data.get("score")),
            is_correct=bool(data.get("is_correct", False)),
            violated_state_ids=_string_tuple(data.get("violated_state_ids")),
            passed_tests=_string_tuple(data.get("passed_tests")),
            failed_tests=_string_tuple(data.get("failed_tests")),
            drift_flags=_string_tuple(data.get("drift_flags")),
            metadata=tuple(metadata),
        )

    def policy_request(value: object) -> PolicyRequest:
        data = _mapping_value(value, "policy request")
        raw_messages = data.get("messages", ())
        messages: list[PolicyMessage] = []
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
            for item in raw_messages:
                message = _mapping_value(item, "policy message")
                messages.append(
                    PolicyMessage(
                        role=cast(PolicyRole, str(message.get("role", "user"))),
                        content=str(message.get("content", "")),
                    )
                )
        raw_options = data.get("options", ())
        options: list[PublicActionOption] = []
        if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
            for item in raw_options:
                options.append(PublicActionOption.from_dict(_mapping_value(item, "action option")))
        return PolicyRequest(
            request_id=_text(data.get("request_id"), "request.request_id"),
            system_prompt=str(data.get("system_prompt", "")),
            messages=tuple(messages),
            options=tuple(options),
            max_output_tokens=_as_int(data.get("max_output_tokens")),
        )

    def policy_response(value: object) -> PolicyResponse:
        data = _mapping_value(value, "policy response")
        usage = _mapping_value(data.get("usage", {}), "policy usage")
        return PolicyResponse(
            request_id=_text(data.get("request_id"), "response.request_id"),
            provider=str(data.get("provider", "")),
            model_id=str(data.get("model_id", "")),
            endpoint_identity=str(data.get("endpoint_identity", "")),
            selected_option_id=str(data.get("selected_option_id", "")),
            optional_patch=(
                None if data.get("optional_patch") is None else str(data.get("optional_patch"))
            ),
            concise_rationale=str(data.get("concise_rationale", "")),
            provider_request_id=(
                None
                if data.get("provider_request_id") is None
                else str(data.get("provider_request_id"))
            ),
            usage=PolicyUsage(
                input_tokens=(
                    None
                    if usage.get("input_tokens") is None
                    else _as_int(usage["input_tokens"])
                ),
                output_tokens=(
                    None
                    if usage.get("output_tokens") is None
                    else _as_int(usage["output_tokens"])
                ),
                cached_tokens=(
                    None
                    if usage.get("cached_tokens") is None
                    else _as_int(usage["cached_tokens"])
                ),
                reasoning_tokens=(
                    None
                    if usage.get("reasoning_tokens") is None
                    else _as_int(usage["reasoning_tokens"])
                ),
                observed=bool(usage.get("observed", True)),
            ),
            request_hash=str(data.get("request_hash", "")),
            response_hash=str(data.get("response_hash", "")),
            started_at_utc=str(data.get("started_at_utc", "")),
            ended_at_utc=str(data.get("ended_at_utc", "")),
            latency_seconds=_as_float(data.get("latency_seconds")),
            retry_count=_as_int(data.get("retry_count")),
            format_repair_used=bool(data.get("format_repair_used", False)),
        )

    def evaluation_call(value: object) -> EvaluationCall:
        data = _mapping_value(value, "evaluation call")
        return EvaluationCall(
            call_id=_text(data.get("call_id"), "call.call_id"),
            call_kind=str(data.get("call_kind", "baseline")),
            request=policy_request(data.get("request")),
            response=policy_response(data.get("response")),
            selected_action_id=str(data.get("selected_action_id", "")),
            checker_result=checker(data.get("checker_result")),
            outcome=outcome(data.get("outcome")),
            normalized_drift_flags=_string_tuple(data.get("normalized_drift_flags")),
            workspace_hash=str(data.get("workspace_hash", "")),
            transcript_hash=str(data.get("transcript_hash", "")),
            policy_request_hash=str(data.get("policy_request_hash", "")),
            model_visible_memory_ids=_string_tuple(data.get("model_visible_memory_ids")),
            model_visible_blocks=_string_tuple(data.get("model_visible_blocks")),
            model_visible_context_hash=str(data.get("model_visible_context_hash", "")),
            visible_object_count=_as_int(data.get("visible_object_count")),
            visible_object_chars=_as_int(data.get("visible_object_chars")),
        )

    def classification(value: object) -> CausalUseResult:
        data = _mapping_value(value, "causal classification")
        return CausalUseResult(
            memory_id=str(data.get("memory_id", "")),
            intervention_kind=cast(
                InterventionKind,
                str(data.get("intervention_kind", "leave_one_out")),
            ),
            memory_role=cast(MemoryRole, str(data.get("memory_role", "unknown"))),
            label=cast(
                CausalUseLabel,
                str(data.get("label", "causal_direction_ambiguous")),
            ),
            effect_direction=cast(
                EffectDirection,
                str(data.get("effect_direction", "ambiguous")),
            ),
            behaviorally_used=bool(data.get("behaviorally_used", False)),
            baseline_stable=bool(data.get("baseline_stable", False)),
            intervention_stable=bool(data.get("intervention_stable", False)),
            action_changed=bool(data.get("action_changed", False)),
            checker_changed=bool(data.get("checker_changed", False)),
        )

    def intervention(value: object) -> EvaluationIntervention:
        data = _mapping_value(value, "evaluation intervention")
        raw_evaluations = data.get("evaluations", ())
        evaluations = (
            tuple(
                evaluation_call(item)
                for item in raw_evaluations
                if isinstance(item, Mapping)
            )
            if isinstance(raw_evaluations, Sequence)
            and not isinstance(raw_evaluations, (str, bytes))
            else ()
        )
        return EvaluationIntervention(
            intervention_kind=cast(
                InterventionKind,
                str(data.get("intervention_kind", "leave_one_out")),
            ),
            target_memory_id=str(data.get("target_memory_id", "")),
            replacement_memory_id=(
                None
                if data.get("replacement_memory_id") is None
                else str(data.get("replacement_memory_id"))
            ),
            evaluations=evaluations,
            classification=classification(data.get("classification")),
            baseline_memory_count=_as_int(data.get("baseline_memory_count")),
            intervention_memory_count=_as_int(data.get("intervention_memory_count")),
            count_contrast=(
                None
                if data.get("count_contrast") is None
                else str(data.get("count_contrast"))
            ),
            provenance_mode=str(data.get("provenance_mode", "unavailable")),
        )

    def sceu(value: object) -> EvaluationSCEUResult:
        data = _mapping_value(value, "SCEU result")
        raw_baselines = data.get("baseline_evaluations", ())
        raw_interventions = data.get("interventions", ())
        baselines = (
            tuple(
                evaluation_call(item)
                for item in raw_baselines
                if isinstance(item, Mapping)
            )
            if isinstance(raw_baselines, Sequence)
            and not isinstance(raw_baselines, (str, bytes))
            else ()
        )
        interventions = (
            tuple(
                intervention(item)
                for item in raw_interventions
                if isinstance(item, Mapping)
            )
            if isinstance(raw_interventions, Sequence)
            and not isinstance(raw_interventions, (str, bytes))
            else ()
        )
        return EvaluationSCEUResult(
            result_id=str(data.get("result_id", "")),
            sceu_id=str(data.get("sceu_id", "")),
            opportunity_id=str(data.get("opportunity_id", "")),
            checkpoint_session=_as_int(data.get("checkpoint_session")),
            matched_group=str(data.get("matched_group", "")),
            control_kind=str(data.get("control_kind", "")),
            prefix_artifact_hash=str(data.get("prefix_artifact_hash", "")),
            workspace_hash=str(data.get("workspace_hash", "")),
            candidate_memory_ids=_string_tuple(data.get("candidate_memory_ids")),
            retrieved_memory_ids=_string_tuple(data.get("retrieved_memory_ids")),
            model_visible_memory_ids=_string_tuple(data.get("model_visible_memory_ids")),
            model_visible_object_count=_as_int(data.get("model_visible_object_count")),
            model_visible_chars=_as_int(data.get("model_visible_chars")),
            selected_option_id=str(data.get("selected_option_id", "")),
            selected_action_id=str(data.get("selected_action_id", "")),
            behavior=outcome(data.get("behavior")),
            normalized_drift_flags=_string_tuple(data.get("normalized_drift_flags")),
            baseline_stable=bool(data.get("baseline_stable", False)),
            baseline_evaluations=baselines,
            interventions=interventions,
            retrieval_trace_id=(
                None
                if data.get("retrieval_trace_id") is None
                else str(data.get("retrieval_trace_id"))
            ),
            transcript_hash=str(data.get("transcript_hash", "")),
            model_visible_context_hash=str(data.get("model_visible_context_hash", "")),
            candidate_shortfall=bool(data.get("candidate_shortfall", False)),
            backend_retrieved_memory_ids=_string_tuple(
                data.get(
                    "backend_retrieved_memory_ids",
                    data.get("candidate_memory_ids"),
                )
            ),
            selected_memory_ids=_string_tuple(
                data.get(
                    "selected_memory_ids",
                    data.get("retrieved_memory_ids"),
                )
            ),
            behaviorally_used_memory_ids=_string_tuple(
                data.get("behaviorally_used_memory_ids")
            ),
            drift_eligible_categories=(
                None
                if data.get("drift_eligible_categories") is None
                else _string_tuple(data.get("drift_eligible_categories"))
            ),
            current_state_signature=str(
                data.get("current_state_signature", "")
            ),
        )

    raw_conditions = raw.get("condition_results", ())
    condition_rows: list[EvaluationConditionResult] = []
    if isinstance(raw_conditions, Sequence) and not isinstance(
        raw_conditions, (str, bytes)
    ):
        for raw_condition in raw_conditions:
            if not isinstance(raw_condition, Mapping):
                continue
            raw_sceu = raw_condition.get("sceu_results", ())
            sceu_rows = (
                tuple(sceu(item) for item in raw_sceu if isinstance(item, Mapping))
                if isinstance(raw_sceu, Sequence)
                and not isinstance(raw_sceu, (str, bytes))
                else ()
            )
            condition_rows.append(
                EvaluationConditionResult(
                    result_id=str(raw_condition.get("result_id", "")),
                    condition=str(raw_condition.get("condition", "")),
                    readout=cast(
                        ReadoutKind,
                        str(raw_condition.get("readout", "none")),
                    ),
                    status=str(raw_condition.get("status", "complete")),
                    sceu_results=sceu_rows,
                    error_class=(
                        None
                        if raw_condition.get("error_class") is None
                        else str(raw_condition.get("error_class"))
                    ),
                    error_message=(
                        None
                        if raw_condition.get("error_message") is None
                        else str(raw_condition.get("error_message"))
                    ),
                )
            )
    conditions = tuple(condition_rows)
    return EvaluationTaskResult(
        task_id=_text(raw.get("task_id"), "result.task_id"),
        episode_id=_text(raw.get("episode_id"), "result.episode_id"),
        policy_profile_id=_text(raw.get("policy_profile_id"), "result.policy_profile_id"),
        condition=str(raw.get("condition", "")),
        prefix_artifact_hash=str(raw.get("prefix_artifact_hash", "")),
        status=str(raw.get("status", "complete")),
        condition_results=conditions,
        result_hash=_text(raw.get("result_hash"), "result.result_hash"),
        error_class=(None if raw.get("error_class") is None else str(raw.get("error_class"))),
        error_message=(None if raw.get("error_message") is None else str(raw.get("error_message"))),
    )


def _load_evaluation_matrix(
    run_directory: Path,
) -> tuple[QualificationMatrixResult, dict[str, SoftwareMem0VerticalSpec], dict[str, object]]:
    manifest, _config_value, specs_tuple, preparations, _templates = _load_contract(run_directory)
    expected_identity = _text(manifest.get("run_identity"), "run_identity")
    task_map = {task.task_id: task for task in _load_tasks(run_directory)}
    results: list[EvaluationTaskResult] = []
    results_directory = run_directory / "results"
    for path in sorted(results_directory.glob("*.json")):
        envelope = _read_json(path)
        if envelope.get("run_identity") != expected_identity:
            raise MultisystemCliError(f"result run identity mismatch: {path.name}")
        result_raw = _mapping_value(envelope.get("result"), f"result envelope {path.name}")
        task_id = _text(result_raw.get("task_id"), f"result task_id {path.name}")
        if task_id not in task_map:
            raise MultisystemCliError(f"result references unknown evaluation task: {task_id}")
        payload_hash = result_raw.get("task_payload_hash")
        expected_payload_hash = task_map[task_id].task_payload_hash
        if payload_hash is not None and payload_hash != expected_payload_hash:
            raise MultisystemCliError(f"result task payload hash mismatch: {task_id}")
        results.append(_evaluation_result_from_dict(result_raw))
    matrix = QualificationMatrixResult(
        run_identity=expected_identity,
        task_results=cast(Any, tuple(results)),
    )
    storage = QualificationStorage(run_directory / "cells", run_identity=expected_identity)
    artifacts: dict[str, object] = {}
    for task in preparations:
        artifact = storage.load_prefix_artifact(task)
        if artifact is not None:
            artifacts[f"{task.episode_id}--{task.backend}"] = artifact
            artifacts.setdefault(task.backend, artifact)
    return matrix, {spec.plan.episode_id: spec for spec in specs_tuple}, artifacts


def _run_evaluation_matrix(
    run_directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    force: bool = False,
    keep_going: bool = False,
) -> tuple[int, int]:
    tasks = _load_tasks(run_directory)
    completed = 0
    failed = 0
    for index in range(len(tasks)):
        try:
            _evaluate_live_task(
                run_directory,
                index,
                environment=environment,
                force=force,
            )
            completed += 1
        except Exception:
            failed += 1
            if not keep_going:
                break
    return completed, failed


def _aggregate_systems(run_directory: Path, output: Path | None) -> Path:
    from lhmsb.qualification.report import write_qualification_report

    report = output or run_directory / "report"
    matrix, specs, artifacts = _load_evaluation_matrix(run_directory)
    manifest = _read_json(run_directory / "run_manifest.json")
    aggregation_commit, aggregation_dirty, aggregation_ref = _git_identity()
    report_metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_identity", "artifact_hashes"}
    }
    report_metadata.update(
        {
            "evaluation_code_commit": manifest.get("code_commit"),
            "evaluation_code_dirty": manifest.get("code_dirty"),
            "aggregation_code_commit": aggregation_commit,
            "aggregation_code_dirty": aggregation_dirty,
            "aggregation_code_ref": aggregation_ref,
        }
    )
    write_qualification_report(
        matrix,
        specs,
        report,
        run_metadata=report_metadata,
        prefix_artifacts=artifacts,
    )
    # ``write_qualification_report`` emits a pending validation record before
    # aggregation.  Run the canonical validator immediately and publish its
    # result while keeping the manifest's per-file hash consistent.
    from lhmsb.qualification.validate import validate_qualification_artifacts

    validation = validate_qualification_artifacts(
        report,
        expected_run_identity=_text(manifest.get("run_identity"), "run_identity"),
    )
    _atomic_json(
        report / "validation.json",
        {
            "schema_version": 2,
            **validation.to_dict(),
        },
    )
    report_manifest_path = report / "run_manifest.json"
    report_manifest = _read_json(report_manifest_path)
    hashes = report_manifest.get("artifact_hashes")
    if isinstance(hashes, Mapping):
        updated_hashes = dict(hashes)
        updated_hashes["validation.json"] = _sha256(report / "validation.json")
        report_manifest["artifact_hashes"] = updated_hashes
        _atomic_json(report_manifest_path, report_manifest)
    return report


def _validate_systems(report: Path, expected_run_identity: str | None) -> dict[str, object]:
    from lhmsb.qualification.validate import validate_qualification_artifacts

    full_report = all(
        (report / name).is_file()
        for name in (
            "run_manifest.json",
            "tasks.jsonl",
            "task_results.jsonl",
            "sceu_results.jsonl",
            "metrics.json",
            "metrics_by_cell.json",
            "summary.json",
            "scorecard.csv",
            "validation.json",
        )
    )
    if full_report:
        validation = validate_qualification_artifacts(
            report,
            expected_run_identity=expected_run_identity,
        )
        result = validation.to_dict()
        _atomic_json(report / "validation.json", {"schema_version": 2, **result})
        return result
    required = ("metrics.json", "metrics_by_cell.json", "summary.json", "scorecard.csv")
    missing = [name for name in required if not (report / name).is_file()]
    summary = _read_json(report / "summary.json") if not missing else {}
    ok = not missing and bool(summary.get("complete"))
    result = {
        "ok": ok,
        "missing": missing,
        "expected_run_identity": expected_run_identity,
    }
    _atomic_json(report / "validation.json", result)
    return result


def _dry_run_lines() -> tuple[str, ...]:
    return (
        "plan-systems --n-sessions 4",
        "prepare-task 0 (flat_retrieval)",
        "prepare-task 1 (mem0)",
        "prepare-task 2 (amem)",
        "prepare-task 3 (memos)",
        "finalize-evaluation-plan",
        "run-evaluation-matrix",
        "aggregate-systems",
        "validate-systems",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lhmsb.qualification")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan-systems")
    plan.add_argument("--dataset", required=True, type=Path)
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--allow-dirty", action="store_true")
    plan.add_argument("--force", action="store_true")
    plan.add_argument("--n-sessions", type=int)
    plan.add_argument("--json", type=Path)

    prep = sub.add_parser("prepare-task")
    prep.add_argument("--run-dir", required=True, type=Path)
    prep.add_argument("--task-index", required=True, type=int)
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--force", action="store_true")

    final = sub.add_parser("finalize-evaluation-plan")
    final.add_argument("--run-dir", required=True, type=Path)

    evaluate = sub.add_parser("evaluate-task")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--task-index", required=True, type=int)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--force", action="store_true")

    matrix = sub.add_parser("run-evaluation-matrix")
    matrix.add_argument("--run-dir", required=True, type=Path)
    matrix.add_argument("--dry-run", action="store_true")
    matrix.add_argument("--keep-going", action="store_true")
    matrix.add_argument("--force", action="store_true")

    aggregate = sub.add_parser("aggregate-systems")
    aggregate.add_argument("--run-dir", required=True, type=Path)
    aggregate.add_argument("--out", type=Path)
    aggregate.add_argument("--json", type=Path)

    validate = sub.add_parser("validate-systems")
    validate.add_argument("--report", required=True, type=Path)
    validate.add_argument("--run-identity")
    validate.add_argument("--json", type=Path)

    preflight = sub.add_parser("preflight-systems")
    preflight.add_argument("--dataset", required=True, type=Path)
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--data-root", type=Path, default=Path("/data/lhmsb"))
    preflight.add_argument("--repository-only", action="store_true")
    preflight.add_argument("--allow-dirty", action="store_true")
    preflight.add_argument("--json", type=Path)

    smoke = sub.add_parser("smoke-systems")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke-systems" and args.dry_run:
            for line in _dry_run_lines():
                print(f"DRY-RUN {line}")
            return 0
        if args.command == "plan-systems":
            payload = plan_systems_run(
                args.dataset,
                args.config,
                args.out,
                allow_dirty=args.allow_dirty,
                force=args.force,
                n_sessions=args.n_sessions,
            )
            if args.json:
                _atomic_json(args.json, payload)
            print(f"planned systems run {payload['run_identity']} -> {args.out}")
            return 0
        if args.command == "prepare-task":
            preparations = _load_contract(args.run_dir)[3]
            if args.task_index < 0 or args.task_index >= len(preparations):
                raise MultisystemCliError("preparation task index is out of range")
            if args.dry_run:
                print(f"preparation task {args.task_index} dry-run passed")
                return 0
            artifact = _prepare_live_task(
                args.run_dir,
                args.task_index,
                force=args.force,
            )
            print(f"prepared {preparations[args.task_index].task_id} ({artifact.artifact_hash})")
            return 0
        if args.command == "finalize-evaluation-plan":
            payload = finalize_systems_run(args.run_dir)
            print(f"finalized {payload['evaluation_task_count']} evaluation task(s)")
            return 0
        if args.command == "evaluate-task":
            if args.dry_run:
                tasks = _load_tasks(args.run_dir)
                if args.task_index < 0 or args.task_index >= len(tasks):
                    raise MultisystemCliError("evaluation task index is out of range")
                print(f"evaluation task {args.task_index} dry-run passed")
                return 0
            payload = _evaluate_live_task(
                args.run_dir,
                args.task_index,
                force=args.force,
            )
            print(f"evaluated {payload['task_id']}")
            return 0
        if args.command == "run-evaluation-matrix":
            tasks = _load_tasks(args.run_dir)
            if args.dry_run:
                print(f"evaluation matrix dry-run passed ({len(tasks)} task(s))")
                return 0
            completed, failed = _run_evaluation_matrix(
                args.run_dir,
                force=args.force,
                keep_going=args.keep_going,
            )
            print(f"evaluated {completed}/{len(tasks)} task(s); failed={failed}")
            return 0 if failed == 0 else 1
        if args.command == "aggregate-systems":
            report = _aggregate_systems(args.run_dir, args.out)
            if args.json:
                _atomic_json(args.json, _read_json(report / "summary.json"))
            print(f"systems report -> {report}")
            return 0
        if args.command == "validate-systems":
            validation = _validate_systems(args.report, args.run_identity)
            if args.json:
                _atomic_json(args.json, validation)
            print("systems report validated" if validation["ok"] else "systems report FAILED")
            return 0 if validation["ok"] else 1
        if args.command == "preflight-systems":
            config = _config(args.config)
            specs = _specs(args.dataset)
            preflight_report = {
                "ok": bool(specs),
                "config_hash": config.config_hash,
                "n_episodes": len(specs),
                "repository_only": args.repository_only,
            }
            if args.json:
                _atomic_json(args.json, preflight_report)
            print(
                "systems preflight passed"
                if preflight_report["ok"]
                else "systems preflight FAILED"
            )
            return 0 if preflight_report["ok"] else 1
        raise MultisystemCliError(f"unknown command: {args.command}")
    except (MultisystemCliError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"{args.command} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["MultisystemCliError", "is_multisystem_command", "main", "plan_systems_run"]
