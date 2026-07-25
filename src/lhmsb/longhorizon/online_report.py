"""Small, deterministic report writer for online execution artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bootstrap_mean_ci95(
    values: list[float],
    *,
    n_resamples: int = 4000,
    seed: int = 20270725,
) -> tuple[float | None, list[float] | None]:
    """Return a deterministic episode-level bootstrap mean and 95% interval."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, [mean, mean]
    rng = random.Random(seed)
    n = len(values)
    bootstrap = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    lower = bootstrap[int(0.025 * (n_resamples - 1))]
    upper = bootstrap[int(0.975 * (n_resamples - 1))]
    return mean, [lower, upper]


def _score_count_pareto_frontier(
    by_system: dict[str, dict[str, object]],
) -> list[str]:
    """Return systems not dominated by a higher-score, lower-count system."""
    candidates: list[tuple[str, float, float]] = []
    for label, values in by_system.items():
        score = _numeric(values.get("mean_checked_score"))
        count = _numeric(values.get("mean_live_memory_count"))
        if score is not None and count is not None:
            candidates.append((label, score, count))
    frontier: list[str] = []
    for label, score, count in candidates:
        dominated = any(
            other_score >= score
            and other_count <= count
            and (other_score > score or other_count < count)
            for other_label, other_score, other_count in candidates
            if other_label != label
        )
        if not dominated:
            frontier.append(label)
    return sorted(frontier)


def _write_score_count_pareto_svg(
    path: Path,
    by_system: dict[str, dict[str, object]],
    frontier: list[str],
) -> None:
    """Write a dependency-free score--count plot for the experiment report."""
    points: list[tuple[str, float, float]] = []
    for label, values in sorted(by_system.items()):
        score = _numeric(values.get("mean_checked_score"))
        count = _numeric(values.get("mean_live_memory_count"))
        if score is not None and count is not None:
            points.append((label, score, count))
    width, height = 760, 480
    left, right, top, bottom = 90, 30, 40, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(max((count for _, _, count in points), default=1.0), 1.0)

    def x_position(count: float) -> float:
        return left + (count / max_count) * plot_width

    def y_position(score: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, score))) * plot_height

    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="#222"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#222"/>'
        ),
        (
            f'<text x="{left + plot_width / 2}" y="{height - 25}" '
            'text-anchor="middle" font-family="sans-serif" font-size="14">'
            'Mean live native-memory-object count per decision</text>'
        ),
        (
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14" '
            f'transform="rotate(-90 22 {top + plot_height / 2})">'
            'Mean checked decision score</text>'
        ),
        (
            f'<text x="{left}" y="24" font-family="sans-serif" font-size="16" '
            'font-weight="bold">Score--count Pareto view</text>'
        ),
    ]
    for tick in range(6):
        score = tick / 5
        y = y_position(score)
        elements.extend(
            [
                (
                    f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" '
                    f'y2="{y:.2f}" stroke="#222"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="11">{score:.1f}</text>'
                ),
            ]
        )
    for tick in range(6):
        count = max_count * tick / 5
        x = x_position(count)
        elements.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{top + plot_height}" '
                    f'x2="{x:.2f}" y2="{top + plot_height + 5}" stroke="#222"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{top + plot_height + 22}" '
                    f'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11">{count:.1f}</text>'
                ),
            ]
        )
    for label, score, count in points:
        x = x_position(count)
        y = y_position(score)
        color = "#c62828" if label in frontier else "#607d8b"
        elements.extend(
            [
                (
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" '
                    'stroke="white" stroke-width="1.5"/>'
                ),
                (
                    f'<text x="{x + 9:.2f}" y="{y - 8:.2f}" '
                    f'font-family="sans-serif" font-size="12" fill="#222">'
                    f'{html.escape(label)}</text>'
                ),
            ]
        )
    if not points:
        elements.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14" fill="#555">'
            'No memory condition has a defined live-object denominator.</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _read_results(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in {"MANIFEST.json", "report.json", "validation.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            result = payload["result"]
            result_id = str(
                payload.get("result_id")
                or "::".join(
                    (
                        str(result.get("episode_id", "")),
                        str(result.get("condition", "")),
                        str(payload.get("policy_profile", "")),
                    )
                )
            )
            if result_id in seen:
                raise ValueError(
                    f"duplicate online result identity {result_id!r}: "
                    f"{seen[result_id]} and {path}"
                )
            seen[result_id] = path
            records.append(payload)
    return records


def build_online_report(run_dir: Path) -> dict[str, object]:
    records = _read_results(run_dir)
    results = [record["result"] for record in records]
    repositories = [record.get("repository", {}) for record in records]
    total_steps = sum(int(result.get("policy_calls", 0)) for result in results)
    online_count = sum(bool(result.get("online_long_horizon", False)) for result in results)
    causal_count = sum(bool(result.get("causal_chain_verified", False)) for result in results)
    workflow_complete_count = sum(
        bool(result.get("workflow_complete", False)) for result in results
    )
    project_chain_verified_count = sum(
        bool(result.get("project_chain_verified", False)) for result in results
    )
    substantive_work_verified_count = sum(
        bool(result.get("substantive_work_verified", False)) for result in results
    )
    terminal_checker_pass_count = sum(
        bool(result.get("terminal_checker_passed", False)) for result in results
    )
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
    storage_provenance_modes = Counter(
        str(step.get("memory_provenance_mode", "none")) for step in transitions
    )
    influence_denominator = sum(
        max(0, int(result.get("policy_calls", 0)) - 1)
        for result in results
    )
    influence_count = sum(
        int(result.get("downstream_decision_influence_count", 0))
        for result in results
    )
    valid_artifacts = sum(bool(step.get("artifact_valid", False)) for step in transitions)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, result in zip(records, results, strict=True):
        label = (
            str(record.get("backend"))
            if result.get("condition") == "memory"
            else str(result.get("condition", "unknown"))
        )
        grouped[label].append((record, result))
    by_system: dict[str, dict[str, object]] = {}
    for label, pairs in sorted(grouped.items()):
        system_results = [result for _, result in pairs]
        system_steps = [
            step
            for result in system_results
            for step in result.get("steps", [])
            if isinstance(step, dict)
        ]
        system_drift = [step for step in system_steps if step.get("drift_flags")]
        categories = Counter(
            flag for step in system_steps for flag in step.get("drift_flags", [])
        )
        provenance_modes = Counter(
            str(step.get("memory_provenance_mode", "none")) for step in system_steps
        )
        live_counts = [
            int(step["live_memory_count"])
            for step in system_steps
            if isinstance(step.get("live_memory_count"), int)
            and not isinstance(step.get("live_memory_count"), bool)
        ]
        checked_score_sum = sum(bool(step.get("action_correct")) for step in system_steps)
        live_count_sum = sum(live_counts)
        episode_efficiencies = [
            value
            for result in system_results
            if (
                value := _numeric(
                    result.get("score_per_live_memory_object_exposure")
                    if "score_per_live_memory_object_exposure" in result
                    else result.get("score_per_live_memory_object")
                )
            )
            is not None
        ]
        episode_efficiency_mean, episode_efficiency_ci95 = _bootstrap_mean_ci95(
            episode_efficiencies
        )
        by_system[label] = {
            "n_episodes": len(system_results),
            "policy_calls": sum(
                int(result.get("policy_calls", 0)) for result in system_results
            ),
            "online_episode_rate": (
                sum(bool(result.get("online_long_horizon")) for result in system_results)
                / len(system_results)
            ),
            "mean_checked_score": (
                sum(
                    float(result.get("mean_checked_score") or 0.0)
                    for result in system_results
                )
                / len(system_results)
            ),
            "terminal_checker_pass_rate": (
                sum(bool(result.get("terminal_checker_passed")) for result in system_results)
                / len(system_results)
            ),
            "drift_step_rate": (
                None if not system_steps else len(system_drift) / len(system_steps)
            ),
            "drift_category_counts": dict(sorted(categories.items())),
            "storage_provenance_mode_counts": dict(sorted(provenance_modes.items())),
            "checked_score_sum": checked_score_sum,
            "live_memory_count_sum": (None if not live_counts else live_count_sum),
            "live_memory_object_exposure_sum": (
                None if not live_counts else live_count_sum
            ),
            "memory_counted_decision_count": len(live_counts),
            "mean_live_memory_count": (
                None if not live_counts else live_count_sum / len(live_counts)
            ),
            # The pooled ratio is a descriptive total.  The primary
            # uncertainty-bearing estimate is the mean of episode-level
            # ratios below, which keeps episodes as the independent units.
            "score_per_live_memory_object": (
                None
                if not live_counts or live_count_sum == 0
                else checked_score_sum / live_count_sum
            ),
            "pooled_score_per_live_memory_object_exposure": (
                None
                if not live_counts or live_count_sum == 0
                else checked_score_sum / live_count_sum
            ),
            "mean_episode_score_per_live_memory_object_exposure": (
                episode_efficiency_mean
            ),
            "episode_score_per_live_memory_object_exposure_ci95": (
                episode_efficiency_ci95
            ),
            "efficiency_episode_count": len(episode_efficiencies),
            "artifact_validation_rate": (
                None
                if not system_steps
                else sum(bool(step.get("artifact_valid")) for step in system_steps)
                / len(system_steps)
            ),
            "common_rerank_step_rate": (
                None
                if not system_steps
                else sum(
                    step.get("retrieval_selection_mode") == "common_rerank"
                    for step in system_steps
                )
                / len(system_steps)
            ),
        }
    pareto_frontier = _score_count_pareto_frontier(by_system)
    for label, values in by_system.items():
        values["score_count_pareto_frontier"] = label in pareto_frontier

    episode_system: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record, result in zip(records, results, strict=True):
        label = (
            str(record.get("backend"))
            if result.get("condition") == "memory"
            else str(result.get("condition", "unknown"))
        )
        episode_system[str(result.get("episode_id", ""))][label] = result
    paired_vs_workspace: dict[str, dict[str, object]] = {}
    for label in sorted(set(grouped) - {"workspace_only"}):
        score_deltas: list[float] = []
        terminal_deltas: list[int] = []
        drift_deltas: list[float] = []
        for systems in episode_system.values():
            baseline = systems.get("workspace_only")
            treatment = systems.get(label)
            if baseline is None or treatment is None:
                continue
            score_deltas.append(
                float(treatment.get("mean_checked_score") or 0.0)
                - float(baseline.get("mean_checked_score") or 0.0)
            )
            terminal_deltas.append(
                int(bool(treatment.get("terminal_checker_passed")))
                - int(bool(baseline.get("terminal_checker_passed")))
            )
            treatment_steps = [
                step for step in treatment.get("steps", []) if isinstance(step, dict)
            ]
            baseline_steps = [
                step for step in baseline.get("steps", []) if isinstance(step, dict)
            ]
            if treatment_steps and baseline_steps:
                drift_deltas.append(
                    sum(bool(step.get("drift_flags")) for step in treatment_steps)
                    / len(treatment_steps)
                    - sum(bool(step.get("drift_flags")) for step in baseline_steps)
                    / len(baseline_steps)
                )
        paired_vs_workspace[label] = {
            "n_paired_episodes": len(score_deltas),
            "mean_checked_score_gain": (
                None if not score_deltas else sum(score_deltas) / len(score_deltas)
            ),
            "terminal_pass_rate_gain": (
                None if not terminal_deltas else sum(terminal_deltas) / len(terminal_deltas)
            ),
            "drift_step_rate_difference": (
                None if not drift_deltas else sum(drift_deltas) / len(drift_deltas)
            ),
        }
    return {
        "track": "online_long_horizon_agent_execution",
        "run_dir": str(run_dir),
        "repositories": repositories,
        "n_episodes": len(results),
        "online_episode_count": online_count,
        "causal_chain_verified_count": causal_count,
        "workflow_complete_count": workflow_complete_count,
        "project_chain_verified_count": project_chain_verified_count,
        "substantive_work_verified_count": substantive_work_verified_count,
        "terminal_checker_pass_count": terminal_checker_pass_count,
        "total_policy_calls": total_steps,
        "workspace_mutation_rate": None if not transitions else mutated / len(transitions),
        "artifact_validation_rate": (
            None if not transitions else valid_artifacts / len(transitions)
        ),
        "downstream_decision_influence_rate": (
            None if influence_denominator == 0 else influence_count / influence_denominator
        ),
        "drift_step_rate": None if not transitions else len(drift_steps) / len(transitions),
        "drift_category_counts": dict(sorted(drift_categories.items())),
        "storage_provenance_mode_counts": dict(sorted(storage_provenance_modes.items())),
        "mean_checked_score": (
            None
            if not results
            else sum(float(result.get("mean_checked_score") or 0.0) for result in results)
            / len(results)
        ),
        "conditions": sorted(
            {str(result.get("condition", "unknown")) for result in results}
        ),
        "system_conditions": sorted(
            {
                (
                    str(record.get("backend"))
                    if result.get("condition") == "memory"
                    else str(result.get("condition", "unknown"))
                )
                for record, result in zip(records, results, strict=True)
            }
        ),
        "interaction_modes": sorted(
            {
                str(result.get("task_span", {}).get("interaction_mode", "unknown"))
                for result in results
            }
        ),
        "diagnostic_probes": "sampled_at_preregistered_checkpoints_only",
        "by_system": by_system,
        "score_count_pareto_frontier": pareto_frontier,
        "paired_vs_workspace": paired_vs_workspace,
        "results": [
            {
                "episode_id": result.get("episode_id"),
                "result_id": record.get("result_id"),
                "condition": result.get("condition"),
                "backend": record.get("backend"),
                "system_condition": (
                    record.get("backend")
                    if result.get("condition") == "memory"
                    else result.get("condition")
                ),
                "policy_calls": result.get("policy_calls"),
                "online_long_horizon": result.get("online_long_horizon"),
                "causal_chain_verified": result.get("causal_chain_verified"),
                "workflow_complete": result.get("workflow_complete"),
                "project_chain_verified": result.get("project_chain_verified"),
                "substantive_work_verified": result.get("substantive_work_verified"),
                "memory_selection_verified": result.get("memory_selection_verified"),
                "memory_count_verified": result.get("memory_count_verified"),
                "efficiency_measurement_verified": result.get(
                    "efficiency_measurement_verified"
                ),
                "work_unit_completion_rate": result.get("work_unit_completion_rate"),
                "artifact_validation_rate": (
                    None
                    if not result.get("steps")
                    else sum(
                        bool(step.get("artifact_valid", False))
                        for step in result["steps"]
                    )
                    / len(result["steps"])
                ),
                "mean_checked_score": result.get("mean_checked_score"),
                "terminal_action_correct": result.get("terminal_action_correct"),
                "terminal_checker_score": result.get("terminal_checker_score"),
                "terminal_checker_passed": result.get("terminal_checker_passed"),
                "live_memory_count_sum": result.get("live_memory_count_sum"),
                "live_memory_object_exposure_sum": result.get(
                    "live_memory_object_exposure_sum",
                    result.get("live_memory_count_sum"),
                ),
                "mean_live_memory_count": result.get("mean_live_memory_count"),
                "score_per_live_memory_object": result.get(
                    "score_per_live_memory_object"
                ),
                "score_per_live_memory_object_exposure": result.get(
                    "score_per_live_memory_object_exposure",
                    result.get("score_per_live_memory_object"),
                ),
                "workspace_hash": result.get("workspace_hash"),
                "transcript_hash": result.get("transcript_hash"),
            }
            for record, result in zip(records, results, strict=True)
        ],
    }


def write_online_report(run_dir: Path) -> dict[str, object]:
    report = build_online_report(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raw_by_system = report.get("by_system")
    report_by_system = raw_by_system if isinstance(raw_by_system, dict) else {}
    per_system_lines = [
        (
            f"- {label}: episodes={values.get('n_episodes')}, "
            f"score={values.get('mean_checked_score')}, "
            f"mean_live_count={values.get('mean_live_memory_count')}, "
            "score/live-object-exposure="
            f"{values.get('mean_episode_score_per_live_memory_object_exposure')} "
            f"(95% CI {values.get('episode_score_per_live_memory_object_exposure_ci95')}), "
            f"pareto={values.get('score_count_pareto_frontier')}, "
            f"terminal_pass={values.get('terminal_checker_pass_rate')}, "
            f"drift={values.get('drift_step_rate')}"
        )
        for label, values in report_by_system.items()
        if isinstance(values, dict)
    ]
    raw_paired = report.get("paired_vs_workspace")
    report_paired = raw_paired if isinstance(raw_paired, dict) else {}
    paired_lines = [
        (
            f"- {label}: n={values.get('n_paired_episodes')}, "
            f"score_gain={values.get('mean_checked_score_gain')}, "
            f"terminal_gain={values.get('terminal_pass_rate_gain')}, "
            f"drift_difference={values.get('drift_step_rate_difference')}"
        )
        for label, values in report_paired.items()
        if isinstance(values, dict)
    ]
    lines = [
        "# Online long-horizon execution report",
        "",
        f"- Episodes: {report['n_episodes']}",
        "- Online episodes (>=200 causally linked policy decisions): "
        f"{report['online_episode_count']}",
        f"- Verified causal chains: {report['causal_chain_verified_count']}",
        f"- Complete dependency-linked workflows: {report['workflow_complete_count']}",
        f"- Content-linked project chains: {report['project_chain_verified_count']}",
        f"- Anti-padding work traces: {report['substantive_work_verified_count']}",
        f"- Policy calls: {report['total_policy_calls']}",
        f"- Mean checked score: {report['mean_checked_score']}",
        f"- Workspace mutation rate: {report['workspace_mutation_rate']}",
        f"- Work-unit artifact validation rate: {report['artifact_validation_rate']}",
        "- Terminal executable checker pass count: "
        f"{report['terminal_checker_pass_count']}",
        f"- Downstream decision influence rate: {report['downstream_decision_influence_rate']}",
        f"- Drift step rate: {report['drift_step_rate']}",
        f"- Drift categories: {report['drift_category_counts']}",
        f"- Storage provenance modes: {report['storage_provenance_mode_counts']}",
        "- Efficiency denominator: cumulative live native-memory-object exposure "
        "sampled immediately before every scored retrieval.",
        f"- Score--count Pareto frontier: {report['score_count_pareto_frontier']}",
        "",
        "## Per-system outcomes",
        "",
        *per_system_lines,
        "",
        "## Paired gain beyond workspace",
        "",
        *paired_lines,
        "",
        "This report covers the online closed-loop track. Attribution probes are "
        "sampled at preregistered checkpoints and are not a prerequisite for the "
        "online execution claim.",
        "",
        "Efficiency artifacts: `efficiency.csv` and `score_count_pareto.svg`.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (run_dir / "efficiency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "system",
                "n_episodes",
                "mean_checked_score",
                "checked_score_sum",
                "mean_live_memory_count",
                "live_memory_object_exposure_sum",
                "mean_episode_score_per_live_memory_object_exposure",
                "episode_efficiency_ci95_lower",
                "episode_efficiency_ci95_upper",
                "pooled_score_per_live_memory_object_exposure",
                "score_count_pareto_frontier",
            ),
        )
        writer.writeheader()
        for label, values in sorted(report_by_system.items()):
            if not isinstance(values, dict):
                continue
            interval = values.get(
                "episode_score_per_live_memory_object_exposure_ci95"
            )
            lower = (
                interval[0]
                if isinstance(interval, list) and len(interval) == 2
                else None
            )
            upper = (
                interval[1]
                if isinstance(interval, list) and len(interval) == 2
                else None
            )
            writer.writerow(
                {
                    "system": label,
                    "n_episodes": values.get("n_episodes"),
                    "mean_checked_score": values.get("mean_checked_score"),
                    "checked_score_sum": values.get("checked_score_sum"),
                    "mean_live_memory_count": values.get("mean_live_memory_count"),
                    "live_memory_object_exposure_sum": values.get(
                        "live_memory_object_exposure_sum"
                    ),
                    "mean_episode_score_per_live_memory_object_exposure": values.get(
                        "mean_episode_score_per_live_memory_object_exposure"
                    ),
                    "episode_efficiency_ci95_lower": lower,
                    "episode_efficiency_ci95_upper": upper,
                    "pooled_score_per_live_memory_object_exposure": values.get(
                        "pooled_score_per_live_memory_object_exposure"
                    ),
                    "score_count_pareto_frontier": values.get(
                        "score_count_pareto_frontier"
                    ),
                }
            )
    raw_frontier = report.get("score_count_pareto_frontier")
    report_frontier = raw_frontier if isinstance(raw_frontier, list) else []
    _write_score_count_pareto_svg(
        run_dir / "score_count_pareto.svg",
        {
            str(label): values
            for label, values in report_by_system.items()
            if isinstance(values, dict)
        },
        [str(label) for label in report_frontier],
    )
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
