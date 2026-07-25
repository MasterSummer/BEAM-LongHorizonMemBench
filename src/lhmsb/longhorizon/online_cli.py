"""Command-line entry point for the online long-horizon execution track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import cast

from lhmsb.longhorizon.online import OnlineCondition, run_online_episode
from lhmsb.longhorizon.online_report import write_online_report
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.factory import build_policy_client, build_preparation_components
from lhmsb.qualification.preflight import load_mem0_specs
from lhmsb.qualification.schema import (
    PreparationTask,
    SystemBackend,
    SystemsQualificationConfig,
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def online_result_id(
    *,
    episode_id: str,
    condition: OnlineCondition,
    backend: SystemBackend | None,
    policy_profile: str,
) -> str:
    """Return a stable, collision-free identity for one online trajectory."""
    backend_label = backend if condition == "memory" else "control"
    raw = f"{episode_id}__{condition}__{backend_label}__{policy_profile}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    return safe or _canonical_hash(raw)[:24]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _preparation_task(
    config: SystemsQualificationConfig,
    spec_episode_id: str,
    backend: SystemBackend,
) -> PreparationTask:
    profile = config.system_profiles[backend]
    run_identity = _canonical_hash(
        {
            "track": "online",
            "experiment_id": config.experiment_id,
            "episode_id": spec_episode_id,
            "backend": backend,
            "config_hash": config.config_hash,
        }
    )
    task_id = f"online-{backend}-{spec_episode_id}"
    payload = {
        "stage": "prepare_prefix",
        "task_index": 0,
        "task_id": task_id,
        "episode_id": spec_episode_id,
        "backend": backend,
        "profile_id": profile.profile_id,
        "run_identity": run_identity,
        "config_hash": config.config_hash,
    }
    return PreparationTask(
        task_index=0,
        task_id=task_id,
        episode_id=spec_episode_id,
        backend=backend,
        profile_id=profile.profile_id,
        run_identity=run_identity,
        config_hash=config.config_hash,
        task_payload_hash=_canonical_hash(payload),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--policy-profile", default=None)
    parser.add_argument(
        "--condition",
        choices=("workspace_only", "full_context", "oracle_current_state", "memory"),
        default="workspace_only",
    )
    parser.add_argument("--backend", choices=("flat_retrieval", "mem0", "amem", "memos"))
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="attempt-local backend state root; defaults to data-root",
    )
    parser.add_argument("--steps-per-session", type=int, default=16)
    parser.add_argument("--environment-steps-per-action", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment = dict(os.environ)
    if args.env_file is not None:
        environment.update(_load_env_file(args.env_file))
    loaded = load_qualification_config(args.config)
    if not isinstance(loaded, SystemsQualificationConfig):
        raise SystemExit("online track requires a schema-v2 systems configuration")
    config = loaded
    specs = load_mem0_specs(args.dataset)
    if args.episode_id is None:
        spec = specs[0]
    else:
        try:
            spec = next(item for item in specs if item.plan.episode_id == args.episode_id)
        except StopIteration as exc:
            raise SystemExit(f"unknown episode ID: {args.episode_id}") from exc
    profile_id = args.policy_profile or config.policy_profiles[0].profile_id
    try:
        policy_profile = next(
            item for item in config.policy_profiles if item.profile_id == profile_id
        )
    except StopIteration as exc:
        raise SystemExit(f"unknown policy profile: {profile_id}") from exc
    policy = build_policy_client(policy_profile, environment=environment)
    runtime = None
    reranker = None
    try:
        condition = cast(OnlineCondition, args.condition)
        if condition == "memory":
            if args.backend is None:
                raise SystemExit("--backend is required for --condition memory")
            backend = cast(SystemBackend, args.backend)
            data_root = args.data_root or Path(
                environment.get(config.data_root_env, str(args.dataset.parent))
            )
            runtime_root = args.runtime_root or data_root
            components = build_preparation_components(
                _preparation_task(config, spec.plan.episode_id, backend),
                spec,
                config,
                data_root=runtime_root,
                environment=environment,
            )
            runtime = components.runtime
            reranker = components.reranker
        result = run_online_episode(
            spec,
            policy,  # type: ignore[arg-type]
            condition=condition,
            memory=runtime,
            reranker=reranker,
            visible_k=config.retrieval.visible_k,
            steps_per_session=args.steps_per_session,
            environment_steps_per_action=args.environment_steps_per_action,
            max_output_tokens=config.sampling.max_output_tokens,
        )
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        close_reranker = getattr(reranker, "close", None)
        if callable(close_reranker):
            close_reranker()
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
    args.out.mkdir(parents=True, exist_ok=True)
    repository = _repository_snapshot()
    backend_identity = (
        cast(SystemBackend, args.backend)
        if condition == "memory" and args.backend is not None
        else None
    )
    result_id = online_result_id(
        episode_id=spec.plan.episode_id,
        condition=condition,
        backend=backend_identity,
        policy_profile=profile_id,
    )
    payload = {
        "result_id": result_id,
        "track": "online_long_horizon_agent_execution",
        "config_hash": config.config_hash,
        "dataset_release": config.dataset_release,
        "policy_profile": profile_id,
        "condition": condition,
        "backend": backend_identity,
        "repository": repository,
        "result": result.to_dict(),
    }
    (args.out / f"{result_id}.json").write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "MANIFEST.json").write_text(
        json.dumps(
            {
                "track": "online_long_horizon_agent_execution",
                "config_hash": config.config_hash,
                "dataset_release": config.dataset_release,
                "policy_profile": profile_id,
                "repository": repository,
                "episode_id": spec.plan.episode_id,
                "result_id": result_id,
                "condition": condition,
                "backend": backend_identity,
                "policy_calls": result.policy_calls,
                "online_long_horizon": result.online_long_horizon,
                "causal_chain_verified": result.causal_chain_verified,
                "workflow_complete": result.workflow_complete,
                "project_chain_verified": result.project_chain_verified,
                "substantive_work_verified": result.substantive_work_verified,
                "memory_selection_verified": result.memory_selection_verified,
                "memory_count_verified": result.memory_count_verified,
                "efficiency_measurement_verified": (
                    result.efficiency_measurement_verified
                ),
                "work_unit_completion_rate": result.work_unit_completion_rate,
                "artifact_validation_rate": (
                    None
                    if not result.steps
                    else sum(step.artifact_valid for step in result.steps)
                    / len(result.steps)
                ),
                "mean_checked_score": result.mean_checked_score,
                "terminal_action_correct": result.terminal_action_correct,
                "terminal_checker_score": result.terminal_checker_score,
                "terminal_checker_passed": result.terminal_checker_passed,
                "live_memory_count_sum": result.live_memory_count_sum,
                "live_memory_object_exposure_sum": (
                    result.live_memory_object_exposure_sum
                ),
                "mean_live_memory_count": result.mean_live_memory_count,
                "score_per_live_memory_object": result.score_per_live_memory_object,
                "score_per_live_memory_object_exposure": (
                    result.score_per_live_memory_object_exposure
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_online_report(args.out)
    print(
        json.dumps(
            {
                "episode_id": spec.plan.episode_id,
                "result_id": result_id,
                "online": result.online_long_horizon,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
