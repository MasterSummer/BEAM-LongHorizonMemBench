"""CLI for real-repository multi-session experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lhmsb.realworld.runner import CommandExecutor, run_experiment
from lhmsb.realworld.schema import load_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    parser.add_argument("--agent-command", required=True, help="agent CLI command, passed as argv")
    parser.add_argument("--agent-timeout", type=int, default=3600)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args(argv)
    spec = load_experiment(args.experiment)
    records = run_experiment(
        spec,
        output=args.out,
        executor=CommandExecutor(args.agent_command, timeout_seconds=args.agent_timeout),
        repeats=args.repeats,
    )
    successes = sum(record["task_success"] is True for record in records)
    print(
        json.dumps(
            {"units": len(records), "task_successes": successes, "output": str(args.out.resolve())}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
