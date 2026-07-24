"""Command-line entry point for the online long-horizon execution track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    parser.add_argument("--steps-per-session", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    return parser


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
    try:
        condition = cast(OnlineCondition, args.condition)
        if condition == "memory":
            if args.backend is None:
                raise SystemExit("--backend is required for --condition memory")
            backend = cast(SystemBackend, args.backend)
            data_root = args.data_root or Path(
                environment.get(config.data_root_env, str(args.dataset.parent))
            )
            runtime = build_preparation_components(
                _preparation_task(config, spec.plan.episode_id, backend),
                spec,
                config,
                data_root=data_root,
                environment=environment,
            ).runtime
        result = run_online_episode(
            spec,
            policy,  # type: ignore[arg-type]
            condition=condition,
            memory=runtime,
            steps_per_session=args.steps_per_session,
            max_output_tokens=config.sampling.max_output_tokens,
        )
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": "online_long_horizon_agent_execution",
        "config_hash": config.config_hash,
        "dataset_release": config.dataset_release,
        "policy_profile": profile_id,
        "result": result.to_dict(),
    }
    (args.out / f"{spec.plan.episode_id}.json").write_text(
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
                "episode_id": spec.plan.episode_id,
                "policy_calls": result.policy_calls,
                "online_long_horizon": result.online_long_horizon,
                "causal_chain_verified": result.causal_chain_verified,
            },
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_online_report(args.out)
    print(json.dumps({"episode_id": spec.plan.episode_id, "online": result.online_long_horizon}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
