"""Zero-API terminal re-evaluation for saved online trajectories."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.online import (
    check_online_workspace,
    reconstruct_online_workspace,
)
from lhmsb.qualification.preflight import load_mem0_specs

_IGNORED_NAMES = {
    "MANIFEST.json",
    "report.json",
    "terminal_audit.json",
    "validation.json",
}


def _workspace_mapping(value: object) -> dict[str, str] | None:
    if isinstance(value, Mapping):
        if all(isinstance(path, str) and isinstance(text, str) for path, text in value.items()):
            return dict(value)
        return None
    if not isinstance(value, list):
        return None
    workspace: dict[str, str] = {}
    for item in value:
        if isinstance(item, Mapping):
            path = item.get("path")
            text = item.get("content")
        elif isinstance(item, list | tuple) and len(item) == 2:
            path, text = item
        else:
            return None
        if not isinstance(path, str) or not isinstance(text, str) or path in workspace:
            return None
        workspace[path] = text
    return workspace


def _steps_per_session(
    result: Mapping[str, Any],
    spec: SoftwareMem0VerticalSpec,
) -> int | None:
    explicit = result.get("steps_per_session")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    policy_calls = result.get("policy_calls")
    if (
        isinstance(policy_calls, int)
        and policy_calls > 0
        and policy_calls % spec.plan.n_sessions == 0
    ):
        return policy_calls // spec.plan.n_sessions
    return None


def infer_longitudinal_specs_from_artifacts(
    run_dir: Path,
) -> tuple[tuple[SoftwareMem0VerticalSpec, ...], dict[str, object]]:
    """Resolve legacy v0.14 specs from byte-verified 256-step traces.

    The frozen release used three workspace-recoverability variants selected
    by ``trajectory_seed % 3``.  A candidate is accepted only when rebuilding
    its complete trace reproduces every saved workspace and artifact hash.
    """
    source_by_episode: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in _IGNORED_NAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            continue
        result = payload["result"]
        episode_id = result.get("episode_id")
        if isinstance(episode_id, str):
            source_by_episode.setdefault(episode_id, (path, result))

    specs: list[SoftwareMem0VerticalSpec] = []
    resolutions: list[dict[str, object]] = []
    pattern = re.compile(r"^software-longitudinal-(\d+)$")
    for episode_id, (path, result) in sorted(source_by_episode.items()):
        match = pattern.fullmatch(episode_id)
        steps = result.get("steps")
        handoffs = result.get("handoffs")
        n_sessions = len(handoffs) + 1 if isinstance(handoffs, list) else 0
        if (
            match is None
            or not isinstance(steps, list)
            or not all(isinstance(step, Mapping) for step in steps)
            or n_sessions < 1
        ):
            resolutions.append(
                {
                    "episode_id": episode_id,
                    "status": "unsupported_trace_identity",
                    "source": str(path),
                }
            )
            continue
        policy_calls = result.get("policy_calls")
        if (
            not isinstance(policy_calls, int)
            or policy_calls < 1
            or policy_calls % n_sessions != 0
        ):
            resolutions.append(
                {
                    "episode_id": episode_id,
                    "status": "invalid_step_count",
                    "source": str(path),
                }
            )
            continue
        steps_per_session = policy_calls // n_sessions
        semantic_seed = int(match.group(1))
        matches: list[tuple[int, SoftwareMem0VerticalSpec]] = []
        reconstruction_errors: dict[str, list[str]] = {}
        for trajectory_variant in range(3):
            candidate = SoftwareLongitudinalTrajectoryFamily.generate(
                semantic_seed,
                n_sessions=n_sessions,
                trajectory_seed=trajectory_variant,
                steps_per_session=steps_per_session,
            )
            reconstructed = reconstruct_online_workspace(
                candidate,
                steps,
                steps_per_session=steps_per_session,
            )
            if reconstructed.trace_integrity_valid:
                matches.append((trajectory_variant, candidate))
            else:
                reconstruction_errors[str(trajectory_variant)] = list(
                    reconstructed.errors[:5]
                )
        if len(matches) != 1:
            resolutions.append(
                {
                    "episode_id": episode_id,
                    "status": "ambiguous_or_unmatched",
                    "candidate_count": len(matches),
                    "source": str(path),
                    "sample_errors": reconstruction_errors,
                }
            )
            continue
        trajectory_variant, spec = matches[0]
        specs.append(spec)
        resolutions.append(
            {
                "episode_id": episode_id,
                "status": "resolved_by_full_trace_hash_chain",
                "trajectory_seed_mod_3": trajectory_variant,
                "n_sessions": n_sessions,
                "steps_per_session": steps_per_session,
                "source": str(path),
            }
        )
    return tuple(specs), {
        "method": "full_trace_hash_chain_over_three_recoverability_variants",
        "episode_count": len(source_by_episode),
        "resolved_count": len(specs),
        "records": resolutions,
    }


def audit_online_artifacts(
    run_dir: Path,
    specs: Iterable[SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    """Re-evaluate saved final workspaces without policy or memory API calls."""
    spec_by_id = {spec.plan.episode_id: spec for spec in specs}
    records: list[dict[str, object]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in _IGNORED_NAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append(
                {
                    "path": str(path),
                    "status": "invalid_json",
                    "error": str(exc),
                }
            )
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            continue
        result = payload["result"]
        episode_id = result.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        record: dict[str, object] = {
            "path": str(path),
            "result_id": payload.get("result_id"),
            "episode_id": episode_id,
            "condition": payload.get("condition", result.get("condition")),
            "backend": payload.get("backend"),
            "policy_profile": payload.get("policy_profile"),
            "repository": payload.get("repository"),
            "stored_correct_step_count": result.get("correct_step_count"),
            "stored_mean_checked_score": result.get("mean_checked_score"),
            "stored_terminal_checker_passed": result.get(
                "terminal_checker_passed"
            ),
            "stored_terminal_checker_score": result.get("terminal_checker_score"),
        }
        spec = spec_by_id.get(episode_id)
        if spec is None:
            record["status"] = "unknown_episode"
            records.append(record)
            continue
        steps_per_session = _steps_per_session(result, spec)
        if steps_per_session is None:
            record["status"] = "unreplayable_step_count"
            records.append(record)
            continue
        workspace = _workspace_mapping(result.get("final_workspace_files"))
        if workspace is None:
            raw_steps = result.get("steps")
            if not isinstance(raw_steps, list) or not all(
                isinstance(step, Mapping) for step in raw_steps
            ):
                record["status"] = "legacy_unreplayable_missing_trace"
                records.append(record)
                continue
            reconstruction = reconstruct_online_workspace(
                spec,
                raw_steps,
                steps_per_session=steps_per_session,
            )
            record["trace_reconstruction"] = reconstruction.to_dict()
            if not reconstruction.trace_integrity_valid:
                record["status"] = "trace_reconstruction_failed"
                records.append(record)
                continue
            workspace = reconstruction.workspace
            record["workspace_source"] = "reconstructed_trace"
        else:
            record["workspace_source"] = "saved_final_workspace"
        checked = check_online_workspace(
            spec,
            workspace,
            steps_per_session=steps_per_session,
        )
        stored_passed = result.get("terminal_checker_passed")
        stored_score = result.get("terminal_checker_score")
        record.update(
            {
                "status": "replayed",
                "steps_per_session": steps_per_session,
                "replayed_terminal": checked.to_dict(),
                "verdict_changed": (
                    isinstance(stored_passed, bool)
                    and stored_passed != checked.is_correct
                ),
                "score_changed": (
                    isinstance(stored_score, int | float)
                    and not isinstance(stored_score, bool)
                    and abs(float(stored_score) - checked.score) > 1e-9
                ),
            }
        )
        records.append(record)

    replayed = [record for record in records if record.get("status") == "replayed"]
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    replayed_terminals = [
        terminal
        for record in replayed
        if isinstance((terminal := record.get("replayed_terminal")), Mapping)
    ]
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "artifact_count": len(records),
        "replayed_count": len(replayed),
        "replayed_pass_count": sum(
            bool(terminal.get("is_correct")) for terminal in replayed_terminals
        ),
        "infrastructure_invalid_count": sum(
            not bool(terminal.get("infrastructure_valid"))
            for terminal in replayed_terminals
        ),
        "verdict_changed_count": sum(
            bool(record.get("verdict_changed")) for record in replayed
        ),
        "score_changed_count": sum(
            bool(record.get("score_changed")) for record in replayed
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--infer-longitudinal-traces", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset is not None:
        specs = load_mem0_specs(args.dataset)
        resolution: dict[str, object] = {
            "method": "frozen_dataset",
            "dataset": str(args.dataset),
            "resolved_count": len(specs),
        }
    else:
        specs, resolution = infer_longitudinal_specs_from_artifacts(args.run_dir)
    report = audit_online_artifacts(
        args.run_dir,
        specs,
    )
    report["spec_resolution"] = resolution
    output = args.out or args.run_dir / "terminal_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "audit_online_artifacts",
    "infer_longitudinal_specs_from_artifacts",
    "main",
]
