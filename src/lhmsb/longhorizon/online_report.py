"""Small, deterministic report writer for online execution artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_results(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in {"MANIFEST.json", "report.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            records.append(payload)
    return records


def build_online_report(run_dir: Path) -> dict[str, object]:
    records = _read_results(run_dir)
    results = [record["result"] for record in records]
    repositories = [record.get("repository", {}) for record in records]
    total_steps = sum(int(result.get("policy_calls", 0)) for result in results)
    online_count = sum(bool(result.get("online_long_horizon", False)) for result in results)
    causal_count = sum(bool(result.get("causal_chain_verified", False)) for result in results)
    transitions = [
        step
        for result in results
        for step in result.get("steps", [])
        if isinstance(step, dict)
    ]
    mutated = sum(
        step.get("workspace_hash") != step.get("previous_workspace_hash")
        for step in transitions
    )
    drift_steps = [
        step
        for step in transitions
        if step.get("drift_flags")
    ]
    drift_categories = Counter(
        flag
        for step in transitions
        for flag in step.get("drift_flags", [])
    )
    influence_denominator = sum(
        max(0, int(result.get("policy_calls", 0)) - 1)
        for result in results
    )
    influence_count = sum(
        int(result.get("downstream_decision_influence_count", 0))
        for result in results
    )
    return {
        "track": "online_long_horizon_agent_execution",
        "run_dir": str(run_dir),
        "repositories": repositories,
        "n_episodes": len(results),
        "online_episode_count": online_count,
        "causal_chain_verified_count": causal_count,
        "total_policy_calls": total_steps,
        "workspace_mutation_rate": None if not transitions else mutated / len(transitions),
        "downstream_decision_influence_rate": (
            None if influence_denominator == 0 else influence_count / influence_denominator
        ),
        "drift_step_rate": None if not transitions else len(drift_steps) / len(transitions),
        "drift_category_counts": dict(sorted(drift_categories.items())),
        "conditions": sorted(
            {str(result.get("condition", "unknown")) for result in results}
        ),
        "interaction_modes": sorted(
            {
                str(result.get("task_span", {}).get("interaction_mode", "unknown"))
                for result in results
            }
        ),
        "diagnostic_probes": "sampled_at_preregistered_checkpoints_only",
        "results": [
            {
                "episode_id": result.get("episode_id"),
                "condition": result.get("condition"),
                "policy_calls": result.get("policy_calls"),
                "online_long_horizon": result.get("online_long_horizon"),
                "causal_chain_verified": result.get("causal_chain_verified"),
                "workspace_hash": result.get("workspace_hash"),
                "transcript_hash": result.get("transcript_hash"),
            }
            for result in results
        ],
    }


def write_online_report(run_dir: Path) -> dict[str, object]:
    report = build_online_report(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Online long-horizon execution report",
        "",
        f"- Episodes: {report['n_episodes']}",
        f"- Online episodes (>=200 causal policy calls): {report['online_episode_count']}",
        f"- Verified causal chains: {report['causal_chain_verified_count']}",
        f"- Policy calls: {report['total_policy_calls']}",
        f"- Workspace mutation rate: {report['workspace_mutation_rate']}",
        f"- Downstream decision influence rate: {report['downstream_decision_influence_rate']}",
        f"- Drift step rate: {report['drift_step_rate']}",
        f"- Drift categories: {report['drift_category_counts']}",
        "",
        "This report covers the online closed-loop track. Attribution probes are "
        "sampled at preregistered checkpoints and are not a prerequisite for the "
        "online execution claim.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args(argv)
    report = write_online_report(args.runs)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
