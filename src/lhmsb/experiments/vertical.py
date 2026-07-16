"""Command-line interface for the frozen Software vertical offline pilot."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lhmsb.datasets.stateful_pipeline import StatefulDatasetError
from lhmsb.experiments.vertical_config import VerticalExperimentError
from lhmsb.experiments.vertical_runner import (
    aggregate_vertical_run,
    plan_vertical_run,
    run_vertical_matrix,
    run_vertical_task,
)

_PROG = "python -m lhmsb.experiments.vertical"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Run the frozen Software vertical offline pilot.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="verify inputs and write an atomic task table")
    _add_plan_arguments(plan)

    run = commands.add_parser("run", help="plan, run all tasks sequentially, and aggregate")
    _add_plan_arguments(run)

    run_task = commands.add_parser("run-task", help="execute one zero-based atomic task index")
    run_task.add_argument("--run-dir", required=True, type=Path)
    run_task.add_argument("--task-index", required=True, type=int)
    run_task.add_argument("--force", action="store_true")

    aggregate = commands.add_parser("aggregate", help="aggregate all completed task outputs")
    aggregate.add_argument("--run-dir", required=True, type=Path)
    return parser


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one vertical experiment CLI command and return its exit status."""
    args = _build_parser().parse_args(argv)
    command = str(args.command)
    try:
        if command == "plan":
            manifest = plan_vertical_run(
                args.dataset,
                args.config,
                args.out,
                allow_dirty=args.allow_dirty,
                force=args.force,
            )
            print(
                f"planned {manifest.task_count} task(s) "
                f"for run {manifest.run_identity} -> {args.out}"
            )
            return 0
        if command == "run":
            aggregate = run_vertical_matrix(
                args.dataset,
                args.config,
                args.out,
                allow_dirty=args.allow_dirty,
                force=args.force,
            )
            print(
                f"completed {aggregate.completed_tasks}/{aggregate.planned_tasks} "
                f"task(s) -> {args.out}"
            )
            return 0 if aggregate.complete else 1
        if command == "run-task":
            result = run_vertical_task(
                args.run_dir,
                args.task_index,
                force=args.force,
            )
            print(f"task {args.task_index} result -> {result}")
            return 0
        if command == "aggregate":
            aggregate = aggregate_vertical_run(args.run_dir)
            print(
                f"aggregated {aggregate.completed_tasks}/{aggregate.planned_tasks} "
                f"task(s) -> {args.run_dir}"
            )
            return 0 if aggregate.complete else 1
        raise VerticalExperimentError(f"unknown vertical command: {command}")
    except (
        StatefulDatasetError,
        VerticalExperimentError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{command} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
