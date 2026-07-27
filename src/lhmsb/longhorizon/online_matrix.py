"""Plan and validate the canonical native online long-horizon matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from lhmsb.longhorizon.online import (
    OnlineCondition,
    online_gold_action_ids,
    online_gold_option_ids,
)
from lhmsb.longhorizon.online_cli import online_result_id
from lhmsb.longhorizon.online_report import write_online_report
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.preflight import load_mem0_specs
from lhmsb.qualification.schema import SystemBackend, SystemsQualificationConfig

ONLINE_MATRIX_SCHEMA_VERSION = "memhorizon-online-matrix-v1"
ONLINE_SYSTEM_CONDITIONS = (
    "workspace_only",
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
)
ONLINE_COMMON_MODEL_CONDITIONS = (
    "workspace_only",
    "flat_retrieval",
    "mem0",
    "memos",
)
MAX_ONLINE_GOLD_ACTION_SHARE = 0.60
MAX_ONLINE_GOLD_OPTION_SHARE = 0.40
CANONICAL_STEPS_PER_SESSION = 16


@dataclass(frozen=True)
class OnlineEpisodeTask:
    """One Slurm task that runs the frozen condition subset for one episode."""

    task_index: int
    episode_id: str
    recoverability_variant: str
    semantic_scenario: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_snapshot() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": "unknown", "code_dirty": None}
    return {"code_commit": commit, "code_dirty": dirty}


def _metadata(spec: Any) -> dict[str, str]:
    return {str(key): str(value) for key, value in spec.plan.metadata}


def select_balanced_episode_ids(specs: tuple[Any, ...], n_episodes: int) -> tuple[str, ...]:
    """Select a deterministic round-robin sample over scenario × recoverability."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")
    if n_episodes > len(specs):
        raise ValueError(
            f"requested {n_episodes} online episodes from a {len(specs)}-episode release"
        )
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for spec in specs:
        metadata = _metadata(spec)
        key = (
            metadata.get("recoverability_variant", "unknown"),
            metadata.get("semantic_scenario", "unknown"),
        )
        groups[key].append(spec)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                item.plan.semantic_seed,
                item.plan.trajectory_seed,
                item.plan.episode_id,
            )
        )
    selected: list[str] = []
    variants = sorted({key[0] for key in groups})
    scenarios = sorted({key[1] for key in groups})
    # Diagonal ordering makes a truncated round cover each scenario before
    # repeating it while rotating recoverability variants. For the canonical
    # 5 × 3 strata and n=20, this yields scenario counts 4/4/4/4/4 and
    # recoverability counts 7/7/6 instead of a lexicographic 10/5/5 bias.
    keys = [
        (variants[(scenario_index + layer) % len(variants)], scenario)
        for layer in range(len(variants))
        for scenario_index, scenario in enumerate(scenarios)
        if (variants[(scenario_index + layer) % len(variants)], scenario) in groups
    ]
    keys.extend(key for key in sorted(groups) if key not in keys)
    while len(selected) < n_episodes:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < n_episodes:
                selected.append(groups[key].pop(0).plan.episode_id)
                advanced = True
        if not advanced:
            break
    if len(selected) != n_episodes:
        raise ValueError(
            f"balanced selector produced {len(selected)} of {n_episodes} requested episodes"
        )
    return tuple(selected)


def normalize_online_conditions(
    conditions: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Validate a frozen subset of the canonical online conditions."""
    normalized = tuple(str(item).strip() for item in conditions if str(item).strip())
    if not normalized:
        raise ValueError("online condition subset must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("online condition subset must not contain duplicates")
    unknown = sorted(set(normalized) - set(ONLINE_SYSTEM_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown online conditions: {unknown}")
    if "workspace_only" not in normalized:
        raise ValueError("online condition subset must include workspace_only")
    return normalized


def audit_online_decision_design(
    specs: tuple[Any, ...],
    selected_ids: tuple[str, ...],
    *,
    steps_per_session: int = CANONICAL_STEPS_PER_SESSION,
) -> dict[str, object]:
    """Prove every online decision is solvable and resist fixed-answer shortcuts."""
    if steps_per_session < 1:
        raise ValueError("steps_per_session must be >= 1")
    by_id = {spec.plan.episode_id: spec for spec in specs}
    action_counts: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    unsolvable: list[str] = []
    ambiguous: list[str] = []
    total_decisions = 0
    for episode_id in selected_ids:
        try:
            spec = by_id[episode_id]
        except KeyError as exc:
            raise ValueError(f"design audit received unknown episode: {episode_id}") from exc
        for session_index in range(spec.plan.n_sessions):
            for local_step in range(steps_per_session):
                ordinal = session_index * steps_per_session + local_step
                decision_id = f"{episode_id}:s{session_index:02d}:d{local_step:02d}"
                gold_actions = online_gold_action_ids(
                    spec,
                    session_index=session_index,
                    local_step=local_step,
                )
                gold_options = online_gold_option_ids(
                    spec,
                    session_index=session_index,
                    local_step=local_step,
                    ordinal=ordinal,
                )
                total_decisions += 1
                if not gold_actions:
                    unsolvable.append(decision_id)
                elif len(gold_actions) > 1:
                    ambiguous.append(decision_id)
                action_counts.update(gold_actions)
                option_counts.update(gold_options)
    action_denominator = sum(action_counts.values())
    option_denominator = sum(option_counts.values())
    max_action_share = (
        0.0
        if action_denominator == 0
        else max(action_counts.values(), default=0) / action_denominator
    )
    max_option_share = (
        0.0
        if option_denominator == 0
        else max(option_counts.values(), default=0) / option_denominator
    )
    ok = (
        total_decisions > 0
        and not unsolvable
        and not ambiguous
        and action_denominator == total_decisions
        and option_denominator == total_decisions
        and max_action_share <= MAX_ONLINE_GOLD_ACTION_SHARE
        and max_option_share <= MAX_ONLINE_GOLD_OPTION_SHARE
    )
    return {
        "ok": ok,
        "total_decisions": total_decisions,
        "unique_gold_action_per_decision": not unsolvable and not ambiguous,
        "unsolvable_decision_count": len(unsolvable),
        "ambiguous_decision_count": len(ambiguous),
        "unsolvable_decision_ids": unsolvable,
        "ambiguous_decision_ids": ambiguous,
        "gold_action_counts": dict(sorted(action_counts.items())),
        "gold_action_shares": {
            action_id: count / action_denominator
            for action_id, count in sorted(action_counts.items())
        }
        if action_denominator
        else {},
        "max_gold_action_share": max_action_share,
        "max_gold_action_share_threshold": MAX_ONLINE_GOLD_ACTION_SHARE,
        "gold_option_counts": dict(sorted(option_counts.items())),
        "gold_option_shares": {
            option_id: count / option_denominator
            for option_id, count in sorted(option_counts.items())
        }
        if option_denominator
        else {},
        "max_gold_option_share": max_option_share,
        "max_gold_option_share_threshold": MAX_ONLINE_GOLD_OPTION_SHARE,
    }


def plan_online_matrix(
    *,
    dataset: Path,
    config_path: Path,
    out: Path,
    n_episodes: int = 20,
    policy_profile: str | None = None,
    system_conditions: tuple[str, ...] = ONLINE_SYSTEM_CONDITIONS,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Freeze the episode selection and expected condition result identities."""
    config = load_qualification_config(config_path)
    if not isinstance(config, SystemsQualificationConfig):
        raise ValueError("online matrix requires a schema-v2 systems config")
    profile_id = policy_profile or config.policy_profiles[0].profile_id
    if profile_id not in {profile.profile_id for profile in config.policy_profiles}:
        raise ValueError(f"unknown policy profile: {profile_id}")
    frozen_conditions = normalize_online_conditions(system_conditions)
    missing_backends = sorted(set(frozen_conditions[1:]) - set(config.system_profiles))
    if missing_backends:
        raise ValueError(f"online config is missing backends: {missing_backends}")
    dataset_manifest = json.loads(
        (dataset / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if dataset_manifest.get("release_id") != config.dataset_release:
        raise ValueError(
            "dataset/config release mismatch: "
            f"{dataset_manifest.get('release_id')!r} != {config.dataset_release!r}"
        )
    if dataset_manifest.get("construct_mode") != "longitudinal_trajectories":
        raise ValueError("online matrix requires a longitudinal_trajectories release")
    specs = load_mem0_specs(dataset)
    if len(specs) < 50:
        raise ValueError("canonical online selection requires a 50-episode frozen population")
    if any(spec.plan.n_sessions != 16 for spec in specs):
        raise ValueError("every canonical online episode must contain 16 sessions")
    by_id = {spec.plan.episode_id: spec for spec in specs}
    selected_ids = select_balanced_episode_ids(specs, n_episodes)
    design_audit = audit_online_decision_design(specs, selected_ids)
    if not design_audit["ok"]:
        raise ValueError(
            "online decision design failed solvability/balance gates: "
            f"{json.dumps(design_audit, sort_keys=True)}"
        )
    tasks: list[OnlineEpisodeTask] = []
    expected_result_ids: list[str] = []
    for task_index, episode_id in enumerate(selected_ids):
        metadata = _metadata(by_id[episode_id])
        tasks.append(
            OnlineEpisodeTask(
                task_index=task_index,
                episode_id=episode_id,
                recoverability_variant=metadata.get("recoverability_variant", "unknown"),
                semantic_scenario=metadata.get("semantic_scenario", "unknown"),
            )
        )
        for system_condition in frozen_conditions:
            condition = "workspace_only" if system_condition == "workspace_only" else "memory"
            backend = None if system_condition == "workspace_only" else system_condition
            expected_result_ids.append(
                online_result_id(
                    episode_id=episode_id,
                    condition=cast(OnlineCondition, condition),
                    backend=(
                        None if backend is None else cast(SystemBackend, backend)
                    ),
                    policy_profile=profile_id,
                )
            )
    repository = _repository_snapshot()
    if repository.get("code_dirty") is not False and not allow_dirty:
        raise ValueError("refusing to plan a final online matrix from a dirty checkout")
    manifest = {
        "schema_version": ONLINE_MATRIX_SCHEMA_VERSION,
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": _sha_file(dataset / "MANIFEST.json"),
        "config": str(config_path.resolve()),
        "config_hash": config.config_hash,
        "dataset_release": config.dataset_release,
        "policy_profile": profile_id,
        "required_secret_env": list(config.required_secret_env),
        "n_episodes": len(tasks),
        "selection_method": "deterministic_diagonal_round_robin_v1",
        "recoverability_counts": dict(
            sorted(Counter(task.recoverability_variant for task in tasks).items())
        ),
        "semantic_scenario_counts": dict(
            sorted(Counter(task.semantic_scenario for task in tasks).items())
        ),
        "stratum_counts": dict(
            sorted(
                Counter(
                    f"{task.recoverability_variant}::{task.semantic_scenario}"
                    for task in tasks
                ).items()
            )
        ),
        "conditions": list(frozen_conditions),
        "expected_result_count": len(expected_result_ids),
        "expected_policy_calls": len(expected_result_ids) * 16 * 16,
        "decision_design_audit": design_audit,
        "expected_result_ids": expected_result_ids,
        "repository": repository,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text(
        "".join(
            json.dumps(task.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    (out / "MATRIX.json").write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_online_task(run_dir: Path, task_index: int) -> dict[str, object]:
    """Load exactly one contiguous task record for a Slurm array worker."""
    records = [
        json.loads(line)
        for line in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if task_index < 0 or task_index >= len(records):
        raise IndexError(f"online task index out of range: {task_index}")
    record = records[task_index]
    if record.get("task_index") != task_index:
        raise ValueError(
            f"non-contiguous online task index: expected {task_index}, "
            f"got {record.get('task_index')!r}"
        )
    return {str(key): value for key, value in record.items()}


def aggregate_online_matrix(run_dir: Path) -> dict[str, object]:
    """Aggregate all task directories and enforce only measurement-integrity gates."""
    manifest = json.loads((run_dir / "MATRIX.json").read_text(encoding="utf-8"))
    report = write_online_report(run_dir / "results")
    raw_results = report.get("results", [])
    result_records = raw_results if isinstance(raw_results, list) else []
    raw_repositories = report.get("repositories", [])
    repository_records = raw_repositories if isinstance(raw_repositories, list) else []
    observed_ids = {
        str(item.get("result_id"))
        for item in result_records
        if isinstance(item, dict)
    }
    expected_ids = {str(item) for item in manifest["expected_result_ids"]}
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    integrity_fields = (
        "online_long_horizon",
        "causal_chain_verified",
        "workflow_complete",
        "project_chain_verified",
        "substantive_work_verified",
        "memory_selection_verified",
        "memory_count_verified",
        "efficiency_measurement_verified",
    )
    failed_integrity = [
        {
            "result_id": item.get("result_id"),
            "failed": [field for field in integrity_fields if not item.get(field)],
        }
        for item in result_records
        if isinstance(item, dict) and any(not item.get(field) for field in integrity_fields)
    ]
    dirty_results = [
        record
        for record in repository_records
        if isinstance(record, dict) and record.get("code_dirty") is not False
    ]
    expected_commit = str(manifest.get("repository", {}).get("code_commit", ""))
    wrong_commit_results = [
        record
        for record in repository_records
        if isinstance(record, dict) and str(record.get("code_commit", "")) != expected_commit
    ]
    validation = {
        "ok": (
            not missing
            and not unexpected
            and not failed_integrity
            and not dirty_results
            and not wrong_commit_results
        ),
        "expected_result_count": len(expected_ids),
        "observed_result_count": len(observed_ids),
        "missing_result_ids": missing,
        "unexpected_result_ids": unexpected,
        "failed_integrity": failed_integrity,
        "dirty_result_count": len(dirty_results),
        "wrong_commit_result_count": len(wrong_commit_results),
        # Low scores, drift, and terminal task failures are scientific outcomes,
        # not reasons to discard a completed trajectory.
        "outcome_metrics_are_not_integrity_gates": True,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(validation, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--n-episodes", type=int, default=20)
    plan.add_argument("--policy-profile", default=None)
    plan.add_argument(
        "--system-condition",
        action="append",
        dest="system_conditions",
        choices=ONLINE_SYSTEM_CONDITIONS,
    )
    plan.add_argument("--allow-dirty", action="store_true")
    task = subparsers.add_parser("task")
    task.add_argument("--run-dir", type=Path, required=True)
    task.add_argument("--task-index", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        output = plan_online_matrix(
            dataset=args.dataset,
            config_path=args.config,
            out=args.out,
            n_episodes=args.n_episodes,
            policy_profile=args.policy_profile,
            system_conditions=(
                tuple(args.system_conditions)
                if args.system_conditions
                else ONLINE_SYSTEM_CONDITIONS
            ),
            allow_dirty=args.allow_dirty,
        )
    elif args.command == "task":
        output = load_online_task(args.run_dir, args.task_index)
    else:
        output = aggregate_online_matrix(args.run_dir)
    print(json.dumps(output, sort_keys=True, ensure_ascii=False))
    return 0 if output.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_STEPS_PER_SESSION",
    "MAX_ONLINE_GOLD_ACTION_SHARE",
    "MAX_ONLINE_GOLD_OPTION_SHARE",
    "ONLINE_MATRIX_SCHEMA_VERSION",
    "ONLINE_COMMON_MODEL_CONDITIONS",
    "ONLINE_SYSTEM_CONDITIONS",
    "OnlineEpisodeTask",
    "aggregate_online_matrix",
    "audit_online_decision_design",
    "load_online_task",
    "normalize_online_conditions",
    "plan_online_matrix",
    "select_balanced_episode_ids",
]
