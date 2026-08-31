"""Summarize MemHorizon zero-API terminal audits across saved run batches."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _group_summary(records: list[Mapping[str, object]]) -> dict[str, object]:
    old_scores: list[float] = []
    new_scores: list[float] = []
    terminal_behavior_scores: list[float] = []
    accuracies: list[float] = []
    wrong_counts: list[int] = []
    old_pass = new_pass = score_changed = new_higher = new_lower = 0
    infrastructure_invalid = workspace_contract_invalid = 0
    terminal_behavior_pass = 0
    for record in records:
        terminal = record.get("replayed_terminal")
        if not isinstance(terminal, Mapping):
            continue
        old_score = _number(record.get("stored_terminal_checker_score"))
        new_score = _number(terminal.get("score"))
        terminal_behavior_score = _number(terminal.get("terminal_behavior_score"))
        correct_count = _number(terminal.get("correct_decision_count"))
        decision_count = _number(terminal.get("decision_count"))
        wrong_effects = terminal.get("wrong_effects")
        if old_score is not None and new_score is not None:
            old_scores.append(old_score)
            new_scores.append(new_score)
            changed = abs(old_score - new_score) > 1e-9
            score_changed += changed
            new_higher += changed and new_score > old_score
            new_lower += changed and new_score < old_score
        if correct_count is not None and decision_count:
            accuracies.append(correct_count / decision_count)
        if terminal_behavior_score is not None:
            terminal_behavior_scores.append(terminal_behavior_score)
        if isinstance(wrong_effects, list):
            wrong_counts.append(len(wrong_effects))
        old_pass += bool(record.get("stored_terminal_checker_passed"))
        new_pass += bool(terminal.get("is_correct"))
        terminal_behavior_pass += bool(terminal.get("terminal_behavior_passed"))
        infrastructure_invalid += not bool(terminal.get("infrastructure_valid"))
        workspace_contract_invalid += not bool(
            terminal.get("workspace_contract_valid")
        )
    return {
        "n": len(records),
        "old_terminal_pass_count": old_pass,
        "replayed_terminal_pass_count": new_pass,
        "terminal_behavior_pass_count": terminal_behavior_pass,
        "old_mean_terminal_score": (
            None if not old_scores else round(statistics.mean(old_scores), 4)
        ),
        "replayed_mean_terminal_score": (
            None if not new_scores else round(statistics.mean(new_scores), 4)
        ),
        "mean_decision_accuracy": (
            None if not accuracies else round(statistics.mean(accuracies), 4)
        ),
        "mean_terminal_behavior_score": (
            None
            if not terminal_behavior_scores
            else round(statistics.mean(terminal_behavior_scores), 4)
        ),
        "mean_wrong_effect_count": (
            None if not wrong_counts else round(statistics.mean(wrong_counts), 2)
        ),
        "score_changed_count": score_changed,
        "replayed_score_higher_count": new_higher,
        "replayed_score_lower_count": new_lower,
        "infrastructure_invalid_count": infrastructure_invalid,
        "workspace_contract_invalid_count": workspace_contract_invalid,
    }


def summarize_terminal_audits(paths: Iterable[Path]) -> dict[str, object]:
    """Combine audit reports; all results remain descriptive at this stage."""
    records: list[Mapping[str, object]] = []
    report_summaries: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"not a terminal audit report: {path}")
        report_records = [
            record
            for record in payload["records"]
            if isinstance(record, Mapping)
        ]
        records.extend(report_records)
        report_summaries.append(
            {
                "path": str(path),
                "artifact_count": payload.get("artifact_count"),
                "replayed_count": payload.get("replayed_count"),
                "verdict_changed_count": payload.get("verdict_changed_count"),
                "score_changed_count": payload.get("score_changed_count"),
                "resolved_episode_count": (
                    payload.get("spec_resolution", {}).get("resolved_count")
                    if isinstance(payload.get("spec_resolution"), Mapping)
                    else None
                ),
            }
        )
    groups: dict[str, dict[str, list[Mapping[str, object]]]] = {
        "by_policy_profile": defaultdict(list),
        "by_backend": defaultdict(list),
    }
    for record in records:
        profile = str(record.get("policy_profile") or "unknown")
        backend = str(record.get("backend") or "workspace_only_control")
        groups["by_policy_profile"][profile].append(record)
        groups["by_backend"][backend].append(record)
    trace_reconstructions = [
        record.get("trace_reconstruction")
        for record in records
        if isinstance(record.get("trace_reconstruction"), Mapping)
    ]
    return {
        "schema_version": "memhorizon-terminal-audit-summary-v1",
        "benchmark_name": "MemHorizon",
        "analysis_status": "descriptive_measurement_audit_not_backend_inference",
        "overall": _group_summary(records),
        "all_trace_hash_chains_valid": all(
            bool(item.get("trace_integrity_valid"))
            for item in trace_reconstructions
            if isinstance(item, Mapping)
        ),
        "legacy_serialization_count": sum(
            item.get("serialization_mode") == "legacy_adjacent_json"
            for item in trace_reconstructions
            if isinstance(item, Mapping)
        ),
        "by_policy_profile": {
            label: _group_summary(group)
            for label, group in sorted(groups["by_policy_profile"].items())
        },
        "by_backend": {
            label: _group_summary(group)
            for label, group in sorted(groups["by_backend"].items())
        },
        "source_reports": report_summaries,
        "interpretation": {
            "verdict_changes": (
                "Stored last-action pass verdicts are not end-to-end passes when earlier "
                "persistent effects are wrong; final behavior and cumulative control are "
                "reported as separate estimands."
            ),
            "score_changes": (
                "The score vector exposes structural validity, 256-decision accuracy, "
                "and final hidden-test behavior; the scalar cumulative score is only a "
                "bounded summary of those components."
            ),
            "claim_boundary": (
                "Backend groups are incomplete and unbalanced across saved batches; "
                "no ranking or significance claim is authorized by this audit."
            ),
        },
    }


def _markdown(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]
    if not isinstance(overall, Mapping):
        raise ValueError("summary is missing overall metrics")
    lines = [
        "# MemHorizon terminal measurement audit",
        "",
        "This is a zero-API re-analysis of saved trajectories, not a backend ranking.",
        "",
        (
            "| Scope | N | Old pass | Cumulative pass | Old mean | "
            "Cumulative mean | Decision acc. | Final behavior | Mean wrong effects |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(label: str, values: Mapping[str, object]) -> str:
        return "| " + " | ".join(
            (
                label,
                str(values.get("n")),
                str(values.get("old_terminal_pass_count")),
                str(values.get("replayed_terminal_pass_count")),
                str(values.get("old_mean_terminal_score")),
                str(values.get("replayed_mean_terminal_score")),
                str(values.get("mean_decision_accuracy")),
                str(values.get("mean_terminal_behavior_score")),
                str(values.get("mean_wrong_effect_count")),
            )
        ) + " |"

    lines.append(row("Overall", overall))
    backends = summary.get("by_backend")
    if isinstance(backends, Mapping):
        for label, values in backends.items():
            if isinstance(values, Mapping):
                lines.append(row(str(label), values))
    lines.extend(
        (
            "",
            f"- All trace hash chains valid: {summary.get('all_trace_hash_chains_valid')}",
            f"- Legacy trajectories reconstructed: {summary.get('legacy_serialization_count')}",
            (
                "- Claim boundary: descriptive audit only; saved backend cells "
                "are incomplete and unbalanced."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = summarize_terminal_audits(args.audit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "summarize_terminal_audits"]
