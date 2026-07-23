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
from lhmsb.families.software.horizon_panel import (
    HorizonDose,
    HorizonPanelAudit,
    audit_horizon_panel,
)
from lhmsb.families.software.matched_constructs import (
    MATCHED_CONSTRUCT_VARIANTS,
    audit_matched_construct_triplet,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.interventions import (
    CausalUseLabel,
    EffectDirection,
    InterventionKind,
    MemoryRole,
)
from lhmsb.qualification.analysis_phase import (
    ANALYSIS_PHASES,
    AnalysisPhase,
    AnalysisPhaseError,
    parse_analysis_timing,
    validate_analysis_phase,
)
from lhmsb.qualification.completed_report_audit import (
    CompletedReportAuditError,
    write_completed_report_audit,
)
from lhmsb.qualification.completed_report_reanalysis import (
    CompletedReportReanalysisError,
    write_completed_report_reanalysis,
)
from lhmsb.qualification.config import (
    build_evaluation_task_templates,
    build_preparation_tasks,
    canonical_hash,
    finalize_evaluation_plan,
    load_qualification_config,
)
from lhmsb.qualification.design_audit import compute_experiment_design_audit
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
        "audit-completed-report",
        "reanalyze-completed-report",
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


def _json_normalize(value: object) -> object:
    """Normalize in-memory tuples to the canonical JSON representation."""
    return json.loads(_canonical_bytes(value).decode("utf-8"))


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


def _runtime_source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _assert_runtime_source_root() -> Path:
    """Ensure the imported package comes from the operator-selected checkout."""
    root = _runtime_source_root()
    configured = os.environ.get("LHMSB_REPO_ROOT")
    if configured and root != Path(configured).expanduser().resolve():
        raise MultisystemCliError(
            "runtime lhmsb source differs from LHMSB_REPO_ROOT: "
            f"imported {root}, configured {Path(configured).expanduser().resolve()}"
        )
    return root


def _git_identity() -> tuple[str, bool, str]:
    root = _assert_runtime_source_root()
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


def _assert_planned_code_identity(manifest: Mapping[str, object]) -> tuple[str, bool, str]:
    """Reject formal workers that no longer execute the planned clean commit."""
    current_commit, current_dirty, current_ref = _git_identity()
    planned_dirty = bool(manifest.get("code_dirty", False))
    if planned_dirty:
        # Non-formal runs created with --allow-dirty remain useful for unit and
        # operator diagnostics, but are never accepted by the paper protocol.
        return current_commit, current_dirty, current_ref
    planned_commit = _text(manifest.get("code_commit"), "code_commit")
    if current_dirty:
        raise MultisystemCliError(
            "formal worker checkout is dirty; restore the planned clean commit"
        )
    if current_commit != planned_commit:
        raise MultisystemCliError("formal worker code commit differs from the immutable run plan")
    return current_commit, current_dirty, current_ref


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


def _source_tree_manifest_hash(environment: Mapping[str, str]) -> str:
    return _manifest_hash(
        environment,
        path_keys=("LHMSB_SOURCE_TREE_MANIFEST_PATH",),
        hash_keys=("LHMSB_SOURCE_TREE_MANIFEST_HASH",),
        unavailable_label="source-tree-manifest-unavailable",
    )


def _python_lock_manifest_hash(environment: Mapping[str, str]) -> str:
    return _manifest_hash(
        environment,
        path_keys=("LHMSB_PYTHON_LOCK_MANIFEST_PATH",),
        hash_keys=("LHMSB_PYTHON_LOCK_MANIFEST_HASH",),
        unavailable_label="python-lock-manifest-unavailable",
    )


def _assert_planned_preparation_manifests(
    manifest: Mapping[str, object],
    environment: Mapping[str, str],
) -> tuple[str, str, str, str]:
    """Require every preparation worker to use the four planned manifests."""
    runtime_hash = _runtime_manifest_hash(environment)
    source_tree_hash = _source_tree_manifest_hash(environment)
    model_bundle_hash = _model_files_hash(environment)
    python_lock_hash = _python_lock_manifest_hash(environment)
    comparisons = (
        (
            "runtime manifest",
            manifest.get("runtime_manifest_hash"),
            runtime_hash,
        ),
        (
            "source-tree manifest",
            manifest.get("source_tree_manifest_hash"),
            source_tree_hash,
        ),
        (
            "model bundle",
            manifest.get("model_bundle_hash", manifest.get("model_files_hash")),
            model_bundle_hash,
        ),
        (
            "Python lock manifest",
            manifest.get("python_lock_manifest_hash"),
            python_lock_hash,
        ),
    )
    for label, expected, actual in comparisons:
        if expected is not None and expected != actual:
            raise MultisystemCliError(f"{label} identity differs from the immutable run plan")
    return runtime_hash, source_tree_hash, model_bundle_hash, python_lock_hash


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


def _dataset_selection(
    manifest: Mapping[str, object],
    all_specs: Sequence[SoftwareMem0VerticalSpec],
    *,
    configured_release: str,
    episode_limit: int | None,
) -> tuple[tuple[SoftwareMem0VerticalSpec, ...], dict[str, object]]:
    """Validate dataset identity and select complete statistical units.

    Mixed and longitudinal trajectory releases use physical episodes as their
    analysis units. Matched releases use counterfactual groups. Horizon
    releases use complete short/medium/long panels, each containing three
    matched triplets. A physical-episode prefix may therefore be selected only
    when it contains complete statistical units.
    """
    release_id = _text(manifest.get("release_id"), "dataset release_id")
    if release_id != configured_release:
        raise MultisystemCliError(
            "dataset release does not match experiment config: "
            f"manifest={release_id!r}, config={configured_release!r}"
        )
    manifest_rows = manifest.get("episodes")
    if not isinstance(manifest_rows, Sequence) or isinstance(manifest_rows, (str, bytes)):
        raise MultisystemCliError("dataset manifest episodes must be an array")
    manifest_episode_ids: list[str] = []
    for index, row in enumerate(manifest_rows):
        if not isinstance(row, Mapping):
            raise MultisystemCliError(f"dataset manifest episode {index} must be an object")
        manifest_episode_ids.append(
            _text(row.get("episode_id"), f"dataset manifest episode {index} ID")
        )
    spec_episode_ids = tuple(spec.plan.episode_id for spec in all_specs)
    if tuple(manifest_episode_ids) != spec_episode_ids:
        raise MultisystemCliError("dataset manifest episode order differs from evaluator episodes")
    declared_episode_count = _integer(manifest.get("n_episodes"), "dataset n_episodes")
    if declared_episode_count != len(all_specs):
        raise MultisystemCliError("dataset manifest episode count differs from evaluator episodes")

    grouped: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    ungrouped: list[str] = []
    for spec in all_specs:
        metadata = spec.plan.metadata_dict
        group_id = metadata.get("counterfactual_group_id", "")
        variant = metadata.get("counterfactual_variant", "")
        if bool(group_id) != bool(variant):
            raise MultisystemCliError(
                "matched episode must declare both counterfactual group and variant: "
                f"{spec.plan.episode_id}"
            )
        if group_id:
            if variant not in MATCHED_CONSTRUCT_VARIANTS:
                raise MultisystemCliError(
                    f"unknown counterfactual variant for {spec.plan.episode_id}: {variant!r}"
                )
            grouped.setdefault(group_id, []).append(spec)
        else:
            ungrouped.append(spec.plan.episode_id)
    if grouped and ungrouped:
        raise MultisystemCliError(
            "dataset mixes matched and ungrouped episodes; analysis unit is ambiguous"
        )

    construct_mode = str(manifest.get("construct_mode", "mixed"))
    is_matched = bool(grouped)
    if is_matched != (construct_mode in {"matched_triplets", "horizon_panels"}):
        raise MultisystemCliError(
            "dataset construct_mode disagrees with evaluator counterfactual metadata"
        )
    if construct_mode not in {
        "mixed",
        "matched_triplets",
        "horizon_panels",
        "longitudinal_trajectories",
    }:
        raise MultisystemCliError(f"unknown dataset construct_mode: {construct_mode!r}")
    is_longitudinal = construct_mode == "longitudinal_trajectories"
    longitudinal_members = tuple(
        spec.plan.metadata_dict.get("construct_mode") == "longitudinal_trajectory"
        for spec in all_specs
    )
    if is_longitudinal != (bool(longitudinal_members) and all(longitudinal_members)):
        raise MultisystemCliError(
            "dataset construct_mode disagrees with evaluator longitudinal metadata"
        )
    if not is_longitudinal and any(longitudinal_members):
        raise MultisystemCliError(
            "non-longitudinal dataset contains a longitudinal trajectory episode"
        )
    if is_matched:
        for group_id, group_specs in sorted(grouped.items()):
            matched_audit = audit_matched_construct_triplet(tuple(group_specs))
            if not matched_audit.ok:
                raise MultisystemCliError(
                    f"invalid counterfactual group {group_id}: " + "; ".join(matched_audit.errors)
                )
        declared_groups = _integer(
            manifest.get("n_counterfactual_groups"),
            "dataset n_counterfactual_groups",
        )
        if declared_groups != len(grouped):
            raise MultisystemCliError(
                "dataset counterfactual-group count differs from evaluator episodes"
            )

    horizon_panels: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    nonpanel_matched: list[str] = []
    for spec in all_specs:
        panel_id = spec.plan.metadata_dict.get("horizon_panel_id", "")
        if panel_id:
            horizon_panels.setdefault(panel_id, []).append(spec)
        elif is_matched:
            nonpanel_matched.append(spec.plan.episode_id)
    is_horizon = construct_mode == "horizon_panels"
    if is_horizon and nonpanel_matched:
        raise MultisystemCliError(
            "horizon dataset contains matched episodes without a horizon panel ID"
        )
    if bool(horizon_panels) != is_horizon:
        raise MultisystemCliError(
            "dataset construct_mode disagrees with evaluator horizon metadata"
        )
    if is_horizon:
        for panel_id, panel_specs in sorted(horizon_panels.items()):
            horizon_audit = _audit_horizon_specs(tuple(panel_specs))
            if not horizon_audit.ok:
                raise MultisystemCliError(
                    f"invalid horizon panel {panel_id}: "
                    + "; ".join(
                        [
                            *horizon_audit.errors,
                            *(
                                error
                                for item in horizon_audit.variant_audits
                                for error in item.errors
                            ),
                        ]
                    )
                )
        declared_panels = _integer(
            manifest.get("n_horizon_panels"),
            "dataset n_horizon_panels",
        )
        if declared_panels != len(horizon_panels):
            raise MultisystemCliError("dataset horizon-panel count differs from evaluator episodes")

    selected = tuple(all_specs if episode_limit is None else all_specs[:episode_limit])
    selected_groups: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    if is_matched:
        for spec in selected:
            group_id = spec.plan.metadata_dict["counterfactual_group_id"]
            selected_groups.setdefault(group_id, []).append(spec)
    if is_matched and not is_horizon:
        incomplete = {
            group_id: sorted(
                spec.plan.metadata_dict["counterfactual_variant"] for spec in group_specs
            )
            for group_id, group_specs in selected_groups.items()
            if {spec.plan.metadata_dict["counterfactual_variant"] for spec in group_specs}
            != set(MATCHED_CONSTRUCT_VARIANTS)
        }
        if incomplete:
            raise MultisystemCliError(
                "--episode-limit splits a matched counterfactual triplet; "
                "select a complete three-member group prefix: "
                f"{incomplete}"
            )
    selected_panels: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    if is_horizon:
        for spec in selected:
            panel_id = spec.plan.metadata_dict["horizon_panel_id"]
            selected_panels.setdefault(panel_id, []).append(spec)
        incomplete_panels = {
            panel_id: len(panel_specs)
            for panel_id, panel_specs in selected_panels.items()
            if len(panel_specs) != 9 or not _audit_horizon_specs(tuple(panel_specs)).ok
        }
        if incomplete_panels:
            raise MultisystemCliError(
                "--episode-limit splits a horizon panel; select a complete "
                f"nine-member panel prefix: {incomplete_panels}"
            )
    selected_group_ids = tuple(sorted(selected_groups))
    selected_panel_ids = tuple(sorted(selected_panels))
    primary_analysis_unit = (
        "horizon_panel" if is_horizon else ("counterfactual_group" if is_matched else "episode")
    )
    n_statistical_units = (
        len(selected_panel_ids)
        if is_horizon
        else (len(selected_group_ids) if is_matched else len(selected))
    )
    dataset_statistical_units = (
        len(horizon_panels) if is_horizon else (len(grouped) if is_matched else len(all_specs))
    )
    return selected, {
        "construct_mode": construct_mode,
        "primary_analysis_unit": primary_analysis_unit,
        "physical_episode_count": len(selected),
        "n_statistical_units": n_statistical_units,
        "counterfactual_group_ids": list(selected_group_ids),
        "horizon_panel_ids": list(selected_panel_ids),
        "dataset_physical_episode_count": len(all_specs),
        "dataset_statistical_unit_count": dataset_statistical_units,
    }


def _audit_horizon_specs(
    specs: tuple[SoftwareMem0VerticalSpec, ...],
) -> HorizonPanelAudit:
    by_level = {spec.plan.metadata_dict.get("horizon_level", ""): spec for spec in specs}
    if set(by_level) != {"short", "medium", "long"}:
        return audit_horizon_panel(specs)
    doses = tuple(
        HorizonDose(
            level,
            by_level[level].plan.n_sessions,
            int(by_level[level].plan.metadata_dict["horizon_steps_per_session"]),
        )
        for level in ("short", "medium", "long")
    )
    return audit_horizon_panel(specs, doses=doses)


def _validate_analysis_phase(
    phase: AnalysisPhase,
    *,
    dataset_design: Mapping[str, object],
    design_audit: Mapping[str, object],
) -> None:
    """Prevent diagnostic/calibration samples from being labelled confirmatory."""

    try:
        validate_analysis_phase(
            phase,
            construct_mode=dataset_design.get("construct_mode"),
            n_statistical_units=dataset_design.get("n_statistical_units"),
            balanced_mechanism_design_ready=design_audit.get("balanced_mechanism_design_ready"),
        )
    except AnalysisPhaseError as exc:
        raise MultisystemCliError(str(exc)) from exc


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
            cast(SystemBackend, prefix_backend) if prefix_backend is not None else None
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
            cast(SystemBackend, prefix_backend) if prefix_backend is not None else None
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
    episode_limit: int | None = None,
    analysis_phase: AnalysisPhase = "development",
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Write the immutable Stage-A preparation/template contract."""
    config = _config(config_path)
    all_specs = _specs(dataset)
    if not all_specs:
        raise MultisystemCliError("frozen dataset contains no episodes")
    dataset_manifest = dataset.resolve() / "MANIFEST.json"
    if not dataset_manifest.is_file():
        raise MultisystemCliError(f"missing dataset manifest: {dataset_manifest}")
    dataset_metadata = _read_json(dataset_manifest)
    manifest_construct_mode = str(dataset_metadata.get("construct_mode", "mixed"))
    if n_sessions is not None:
        session_count_matches = (
            max((spec.plan.n_sessions for spec in all_specs), default=0) == n_sessions
            if manifest_construct_mode == "horizon_panels"
            else all(spec.plan.n_sessions == n_sessions for spec in all_specs)
        )
        if not session_count_matches:
            raise MultisystemCliError("dataset session count does not match --n-sessions")
    if episode_limit is not None and episode_limit < 1:
        raise MultisystemCliError("--episode-limit must be positive")
    if episode_limit is not None and episode_limit > len(all_specs):
        raise MultisystemCliError("--episode-limit exceeds the frozen physical episode count")
    specs, dataset_design = _dataset_selection(
        dataset_metadata,
        all_specs,
        configured_release=config.dataset_release,
        episode_limit=episode_limit,
    )
    design_audit = compute_experiment_design_audit({spec.plan.episode_id: spec for spec in specs})
    if design_audit.get("run_ready") is not True:
        raw_failures = design_audit.get("failed_check_ids")
        failures = (
            raw_failures
            if isinstance(raw_failures, Sequence) and not isinstance(raw_failures, str | bytes)
            else ()
        )
        raise MultisystemCliError(
            "policy-free experiment design audit failed: "
            + ", ".join(str(item) for item in failures)
        )
    design_audit_hash = canonical_hash(design_audit)
    _validate_analysis_phase(
        analysis_phase,
        dataset_design=dataset_design,
        design_audit=design_audit,
    )
    selected_episode_ids = tuple(spec.plan.episode_id for spec in specs)
    commit, dirty, ref = _git_identity()
    if dirty and not allow_dirty:
        raise MultisystemCliError("Git worktree is dirty; pass --allow-dirty")
    env = dict(os.environ if environment is None else environment)
    runtime_manifest_hash = _runtime_manifest_hash(env)
    source_tree_manifest_hash = _source_tree_manifest_hash(env)
    model_bundle_hash = _model_files_hash(env)
    python_lock_manifest_hash = _python_lock_manifest_hash(env)
    manifest_path = run_directory / "run_manifest.json"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": commit,
        "code_dirty": dirty,
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "config_hash": config.config_hash,
        "source_lock_hash": config.source_lock_hash,
        "runtime_manifest_hash": runtime_manifest_hash,
        "source_tree_manifest_hash": source_tree_manifest_hash,
        "model_bundle_hash": model_bundle_hash,
        "python_lock_manifest_hash": python_lock_manifest_hash,
        # Prefix artifacts use the historical ``model_files_hash`` name.
        "model_files_hash": model_bundle_hash,
        # A smoke subset and the full run share the same frozen dataset hash,
        # so the selected episode identities must also bind the run identity.
        "episode_ids": list(selected_episode_ids),
        "construct_mode": dataset_design["construct_mode"],
        "primary_analysis_unit": dataset_design["primary_analysis_unit"],
        "counterfactual_group_ids": dataset_design["counterfactual_group_ids"],
        "horizon_panel_ids": dataset_design["horizon_panel_ids"],
        "analysis_phase": analysis_phase,
        "analysis_timing": "pre_specified",
        "experiment_design_audit_hash": design_audit_hash,
        "policy_profiles": [asdict(item) for item in config.policy_profiles],
        "writer_profile": asdict(config.writer_profile),
        "retrieval": asdict(config.retrieval),
    }
    run_identity = canonical_hash(identity)
    preparations = build_preparation_tasks(
        config,
        episode_ids=selected_episode_ids,
        run_identity=run_identity,
    )
    templates = build_evaluation_task_templates(
        config,
        episode_ids=selected_episode_ids,
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
        "code_source_root": str(_runtime_source_root()),
        "task_count": len(preparations),
        "preparation_task_count": len(preparations),
        "evaluation_template_count": len(templates),
        "scored_cell_count": sum(len(item.scored_conditions) for item in templates),
        "episode_ids": list(selected_episode_ids),
        "episode_limit": episode_limit,
        **dataset_design,
        "experiment_design_audit_status": design_audit["audit_status"],
        "balanced_mechanism_design_ready": design_audit["balanced_mechanism_design_ready"],
        "n_sessions": max(spec.plan.n_sessions for spec in specs),
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
    _atomic_json(run_directory / "experiment_design_audit.json", design_audit)
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
    if config.config_hash != _text(manifest.get("config_hash"), "config_hash"):
        raise MultisystemCliError("experiment config differs from the immutable run plan")
    dataset_path = Path(_text(manifest.get("dataset_path"), "dataset_path"))
    dataset_manifest_path = dataset_path / "MANIFEST.json"
    if not dataset_manifest_path.is_file():
        raise MultisystemCliError(f"missing dataset manifest: {dataset_manifest_path}")
    expected_dataset_hash = _text(
        manifest.get("dataset_manifest_sha256"),
        "dataset_manifest_sha256",
    )
    if _sha256(dataset_manifest_path) != expected_dataset_hash:
        raise MultisystemCliError("dataset manifest differs from the immutable run plan")
    specs = _specs(dataset_path)
    raw_episode_limit = manifest.get("episode_limit")
    episode_limit = (
        None if raw_episode_limit is None else _integer(raw_episode_limit, "episode_limit")
    )
    selected_specs, dataset_design = _dataset_selection(
        _read_json(dataset_manifest_path),
        specs,
        configured_release=config.dataset_release,
        episode_limit=episode_limit,
    )
    design_audit_path = run_directory / "experiment_design_audit.json"
    if not design_audit_path.is_file():
        raise MultisystemCliError("missing immutable experiment design audit")
    persisted_design_audit = _read_json(design_audit_path)
    recomputed_design_audit = cast(
        dict[str, object],
        _json_normalize(
            compute_experiment_design_audit(
                {spec.plan.episode_id: spec for spec in selected_specs}
            )
        ),
    )
    if persisted_design_audit != recomputed_design_audit:
        raise MultisystemCliError(
            "experiment design audit differs from the selected frozen dataset"
        )
    expected_design_audit_hash = _text(
        manifest.get("experiment_design_audit_hash"),
        "experiment_design_audit_hash",
    )
    if canonical_hash(persisted_design_audit) != expected_design_audit_hash:
        raise MultisystemCliError(
            "experiment design audit hash differs from the immutable run plan"
        )
    if persisted_design_audit.get("run_ready") is not True:
        raise MultisystemCliError("experiment design audit is not run-ready")
    raw_analysis_phase = _text(
        manifest.get("analysis_phase"),
        "analysis_phase",
    )
    if raw_analysis_phase not in ANALYSIS_PHASES:
        raise MultisystemCliError(
            f"unknown analysis phase in immutable run plan: {raw_analysis_phase}"
        )
    _validate_analysis_phase(
        raw_analysis_phase,
        dataset_design=dataset_design,
        design_audit=persisted_design_audit,
    )
    try:
        analysis_timing = parse_analysis_timing(manifest.get("analysis_timing", "pre_specified"))
    except AnalysisPhaseError as exc:
        raise MultisystemCliError(str(exc)) from exc
    if analysis_timing != "pre_specified":
        raise MultisystemCliError("immutable live-run contract must be fixed before policy calls")
    planned_episode_ids = manifest.get("episode_ids")
    if not isinstance(planned_episode_ids, Sequence) or isinstance(
        planned_episode_ids, (str, bytes)
    ):
        raise MultisystemCliError("run manifest episode_ids must be an array")
    if tuple(str(item) for item in planned_episode_ids) != tuple(
        spec.plan.episode_id for spec in selected_specs
    ):
        raise MultisystemCliError("selected dataset episodes differ from the immutable run plan")
    for field in (
        "construct_mode",
        "primary_analysis_unit",
        "physical_episode_count",
        "n_statistical_units",
        "counterfactual_group_ids",
        "horizon_panel_ids",
        "experiment_design_audit_status",
        "balanced_mechanism_design_ready",
    ):
        expected = (
            persisted_design_audit[
                "audit_status"
                if field == "experiment_design_audit_status"
                else "balanced_mechanism_design_ready"
            ]
            if field
            in {
                "experiment_design_audit_status",
                "balanced_mechanism_design_ready",
            }
            else dataset_design[field]
        )
        if field in manifest and manifest[field] != expected:
            raise MultisystemCliError(f"dataset {field} differs from the immutable run plan")
    preparations = tuple(
        _prep_from_dict(item) for item in _read_jsonl(run_directory / "prepare_tasks.jsonl")
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
    _assert_planned_code_identity(manifest)
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
    configured = environment.get(config.data_root_env) or environment.get("LHMSB_DATA_ROOT")
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
    _assert_planned_code_identity(manifest)
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
    _runtime_manifest, _source_tree_manifest, model_bundle_hash, _python_locks = (
        _assert_planned_preparation_manifests(manifest, env)
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
    evaluation_commit, evaluation_dirty, evaluation_ref = _assert_planned_code_identity(manifest)
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
        "evaluation_code_commit": evaluation_commit,
        "evaluation_code_dirty": evaluation_dirty,
        "evaluation_code_ref": evaluation_ref,
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


def _string_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MultisystemCliError("result pair field must be an array")
    output: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise MultisystemCliError("result pair entries must contain two strings")
        output.append((str(item[0]), str(item[1])))
    return tuple(output)


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
                    None if usage.get("input_tokens") is None else _as_int(usage["input_tokens"])
                ),
                output_tokens=(
                    None if usage.get("output_tokens") is None else _as_int(usage["output_tokens"])
                ),
                cached_tokens=(
                    None if usage.get("cached_tokens") is None else _as_int(usage["cached_tokens"])
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
            tuple(evaluation_call(item) for item in raw_evaluations if isinstance(item, Mapping))
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
                None if data.get("count_contrast") is None else str(data.get("count_contrast"))
            ),
            provenance_mode=str(data.get("provenance_mode", "unavailable")),
        )

    def sceu(value: object) -> EvaluationSCEUResult:
        data = _mapping_value(value, "SCEU result")
        raw_baselines = data.get("baseline_evaluations", ())
        raw_interventions = data.get("interventions", ())
        baselines = (
            tuple(evaluation_call(item) for item in raw_baselines if isinstance(item, Mapping))
            if isinstance(raw_baselines, Sequence) and not isinstance(raw_baselines, (str, bytes))
            else ()
        )
        interventions = (
            tuple(intervention(item) for item in raw_interventions if isinstance(item, Mapping))
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
            behaviorally_used_memory_ids=_string_tuple(data.get("behaviorally_used_memory_ids")),
            drift_eligible_categories=(
                None
                if data.get("drift_eligible_categories") is None
                else _string_tuple(data.get("drift_eligible_categories"))
            ),
            drift_lineage_pairs=_string_pairs(data.get("drift_lineage_pairs")),
            drift_lineage_evidence_mode=str(
                data.get("drift_lineage_evidence_mode", "unavailable")
            ),
            current_state_signature=str(data.get("current_state_signature", "")),
        )

    raw_conditions = raw.get("condition_results", ())
    condition_rows: list[EvaluationConditionResult] = []
    if isinstance(raw_conditions, Sequence) and not isinstance(raw_conditions, (str, bytes)):
        for raw_condition in raw_conditions:
            if not isinstance(raw_condition, Mapping):
                continue
            raw_sceu = raw_condition.get("sceu_results", ())
            sceu_rows = (
                tuple(sceu(item) for item in raw_sceu if isinstance(item, Mapping))
                if isinstance(raw_sceu, Sequence) and not isinstance(raw_sceu, (str, bytes))
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
        worker_commit = envelope.get("evaluation_code_commit")
        if worker_commit is not None and worker_commit != manifest.get("code_commit"):
            raise MultisystemCliError(f"result code commit mismatch: {path.name}")
        if envelope.get("evaluation_code_dirty") is True:
            raise MultisystemCliError(f"result came from a dirty worker: {path.name}")
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
        "plan-systems --episode-limit 1",
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
    plan.add_argument("--episode-limit", type=int)
    plan.add_argument(
        "--analysis-phase",
        choices=ANALYSIS_PHASES,
        default=os.environ.get("LHMSB_ANALYSIS_PHASE", "development"),
    )
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

    audit = sub.add_parser("audit-completed-report")
    audit.add_argument("--report", required=True, type=Path)
    audit.add_argument("--out", required=True, type=Path)
    audit.add_argument(
        "--dataset",
        type=Path,
        help="Exact frozen dataset declared by the source run; required for zero-API attribution",
    )
    audit.add_argument(
        "--analysis-timing",
        choices=("post_hoc_scope_audit", "post_hoc_exploratory"),
        default="post_hoc_scope_audit",
    )
    audit.add_argument("--force", action="store_true")

    reanalysis = sub.add_parser("reanalyze-completed-report")
    reanalysis.add_argument("--report", required=True, type=Path)
    reanalysis.add_argument("--dataset", required=True, type=Path)
    reanalysis.add_argument("--out", required=True, type=Path)
    reanalysis.add_argument("--force", action="store_true")
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
                episode_limit=args.episode_limit,
                analysis_phase=cast(AnalysisPhase, args.analysis_phase),
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
                "systems preflight passed" if preflight_report["ok"] else "systems preflight FAILED"
            )
            return 0 if preflight_report["ok"] else 1
        if args.command == "audit-completed-report":
            output = write_completed_report_audit(
                args.report,
                args.out,
                frozen_dataset=args.dataset,
                audit_analysis_timing=parse_analysis_timing(args.analysis_timing),
                force=args.force,
            )
            print(f"completed-report contribution audit -> {output}")
            return 0
        if args.command == "reanalyze-completed-report":
            output = write_completed_report_reanalysis(
                args.report,
                args.dataset,
                args.out,
                force=args.force,
            )
            print(f"completed-report decision reanalysis -> {output}")
            return 0
        raise MultisystemCliError(f"unknown command: {args.command}")
    except (
        CompletedReportAuditError,
        CompletedReportReanalysisError,
        MultisystemCliError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{args.command} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["MultisystemCliError", "is_multisystem_command", "main", "plan_systems_run"]
