"""Scorecard / leaderboard reporting module (task 24; spec/03-protocol.md).

Turns analysis output (:func:`~lhmsb.metrics.memory_roi.compute_memory_roi` +
:func:`~lhmsb.analysis.stats.aggregate`) into publishable artifacts:

  * ``scorecard.md`` — Markdown with separate Native / Controlled track tables,
    per-system task-score, utilization, drift, storage/retrieval efficiency, and
    count-based Memory ROI with bootstrap CI.
  * ``scorecard.csv`` — same data in tabular form with a ``track`` column.
  * ``scorecard.json`` — structured output with ``native`` / ``controlled`` keys.
  * ``pareto_{family}.png`` + ``pareto_overall.png`` — Pareto-front plots with
    headless matplotlib (Agg backend).

Native and controlled tracks are NEVER merged into one table or one JSON array.

The :func:`render_bare_number_guard` raises :class:`BareNumberError` if any
downstream code requests a single scalar "winner" without CI + Pareto context.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

from lhmsb.analysis.stats import DEFAULT_AGGREGATE_BY, AggregatedStats, aggregate
from lhmsb.cost import CostConfig
from lhmsb.metrics.memory_roi import MemoryRoiResult, compute_memory_roi
from lhmsb.metrics.roi import (
    OVERALL_FAMILY,
    PARETO_DOMINATED,
    PARETO_ON_FRONT,
)
from lhmsb.runner.results import ResultsTable

__all__ = [
    "BareNumberError",
    "Scorecard",
    "TrackResults",
    "generate_scorecard",
    "render_bare_number_guard",
]

# Headline metrics. Token/resource cost is intentionally excluded.
_HEADLINE_METRICS: tuple[str, ...] = (
    "task_score",
    "utilization_rate",
    "drift_index",
    "retrieval_endogenous_precision",
    "retrieval_oracle_precision",
    "stored_memory_count",
    "retrieved_memory_count",
    "storage_precision",
    "storage_recall",
    "storage_f1",
    "retrieval_precision",
    "retrieval_recall",
    "retrieval_f1",
    "retrieval_false_positive_rate",
    "retrieval_timeliness",
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class BareNumberError(Exception):
    """Raised when code requests a single scalar winner without CI/Pareto context.

    A leaderboard number without a bootstrap CI and Pareto-front status is
    meaningless (and potentially misleading) for a counterfactual benchmark.
    """


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackResults:
    """ROI + aggregated statistics for one track (native or controlled)."""

    track: str
    roi_results: list[MemoryRoiResult]
    aggregated: AggregatedStats


@dataclass(frozen=True)
class Scorecard:
    """The full scorecard: native + controlled results, never merged."""

    native_results: TrackResults
    controlled_results: TrackResults
    families: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def generate_scorecard(
    native_table: ResultsTable,
    controlled_table: ResultsTable,
    cost_config: CostConfig,
    out_dir: Path | str,
    *,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> Scorecard:
    """Generate the full scorecard (Markdown + CSV + JSON + Pareto plots).

    Runs :func:`~lhmsb.metrics.memory_roi.compute_memory_roi` and
    :func:`~lhmsb.analysis.stats.aggregate` per track, then writes all artifacts
    to ``out_dir``. Native and controlled tracks are processed and rendered
    independently — they are NEVER merged.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ``cost_config`` remains in the signature for backwards-compatible callers;
    # the scorecard no longer consumes it as a metric.
    _ = cost_config
    native_roi = compute_memory_roi(native_table, bootstrap_n=bootstrap_n, seed=seed)
    controlled_roi = compute_memory_roi(
        controlled_table, bootstrap_n=bootstrap_n, seed=seed + 1000
    )

    # Aggregate stats per track
    native_agg = aggregate(
        native_table.rows,
        by=DEFAULT_AGGREGATE_BY,
        metrics=_HEADLINE_METRICS,
        cost_config=None,
        n=bootstrap_n,
        seed=seed,
    )
    controlled_agg = aggregate(
        controlled_table.rows,
        by=DEFAULT_AGGREGATE_BY,
        metrics=_HEADLINE_METRICS,
        cost_config=None,
        n=bootstrap_n,
        seed=seed + 1000,
    )

    native_track = TrackResults(
        track="native", roi_results=native_roi, aggregated=native_agg
    )
    controlled_track = TrackResults(
        track="controlled", roi_results=controlled_roi, aggregated=controlled_agg
    )

    families = _extract_families(native_roi)

    scorecard = Scorecard(
        native_results=native_track,
        controlled_results=controlled_track,
        families=families,
    )

    # Write artifacts
    _write_markdown(scorecard, out / "scorecard.md")
    _write_csv(scorecard, out / "scorecard.csv")
    _write_json(scorecard, out / "scorecard.json")
    _write_pareto_plots(scorecard, out)

    return scorecard


# --------------------------------------------------------------------------- #
# Bare-number guard
# --------------------------------------------------------------------------- #
def render_bare_number_guard(scorecard: Scorecard) -> None:
    """Raise :class:`BareNumberError` — a single scalar winner is forbidden.

    Every leaderboard number must be accompanied by a bootstrap CI and
    Pareto-front context. This guard exists so downstream code cannot
    accidentally reduce the scorecard to a bare ranking.
    """
    raise BareNumberError(
        "A single scalar 'winner' without CI + Pareto context is forbidden. "
        "Use the full scorecard (ROI [ci_low, ci_high], pareto_status, "
        "memory-count and efficiency metrics) for every system."
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_families(roi_results: list[MemoryRoiResult]) -> tuple[str, ...]:
    """Sorted unique families (excluding the overall sentinel)."""
    families = sorted(
        {r.family for r in roi_results if r.family != OVERALL_FAMILY}
    )
    return tuple(families)


def _format_roi(result: MemoryRoiResult) -> str:
    """Format ROI as ``value [lo, hi]`` or ``N/A``."""
    if result.is_baseline or result.roi is None:
        return "N/A"
    roi_str = f"{result.roi:.4f}"
    if result.ci_low is not None and result.ci_high is not None:
        roi_str += f" [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    if result.below_gain_floor:
        roi_str += " (below_gain_floor)"
    return roi_str


def _format_float(value: float | None) -> str:
    """Format a float or N/A."""
    if value is None:
        return "N/A"
    if math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def _roi_by_family(
    roi_results: list[MemoryRoiResult],
) -> dict[str, list[MemoryRoiResult]]:
    """Group ROI results by family."""
    grouped: dict[str, list[MemoryRoiResult]] = {}
    for result in roi_results:
        grouped.setdefault(result.family, []).append(result)
    return grouped


def _build_row_dict(
    roi: MemoryRoiResult,
    agg: AggregatedStats,
    track: str,
) -> dict[str, object]:
    """Build a flat dict combining ROI + aggregate stats for one condition/family."""
    # Find matching group stats
    group = agg.group_for(
        family=roi.family, condition=roi.condition, track=track
    )

    row: dict[str, object] = {
        "track": track,
        "family": roi.family,
        "condition": roi.condition,
        "n_episodes": roi.n_episodes,
        "task_score": _group_metric(group, "task_score"),
        "utilization_rate": _group_metric(group, "utilization_rate"),
        "drift_index": _group_metric(group, "drift_index"),
        "retrieval_endogenous_precision": _group_metric(
            group, "retrieval_endogenous_precision"
        ),
        "retrieval_oracle_precision": _group_metric(group, "retrieval_oracle_precision"),
        "stored_memory_count": _group_metric(group, "stored_memory_count"),
        "retrieved_memory_count": _group_metric(group, "retrieved_memory_count"),
        "storage_precision": _group_metric(group, "storage_precision"),
        "storage_recall": _group_metric(group, "storage_recall"),
        "storage_f1": _group_metric(group, "storage_f1"),
        "retrieval_precision": _group_metric(group, "retrieval_precision"),
        "retrieval_recall": _group_metric(group, "retrieval_recall"),
        "retrieval_f1": _group_metric(group, "retrieval_f1"),
        "retrieval_false_positive_rate": _group_metric(
            group, "retrieval_false_positive_rate"
        ),
        "retrieval_timeliness": _group_metric(group, "retrieval_timeliness"),
        "roi": _format_roi(roi),
        "roi_value": roi.roi,
        "ci_low": roi.ci_low,
        "ci_high": roi.ci_high,
        "pareto_status": roi.pareto_status,
        "roi_status": roi.roi_status,
        "is_baseline": roi.is_baseline,
        "below_gain_floor": roi.below_gain_floor,
        "mean_normalized_gain": roi.mean_normalized_gain,
        "mean_memory_count": roi.mean_memory_count,
        "gain_floor": roi.gain_floor,
        "n_rows": group.n_rows if group is not None else 0,
        "n_failed": group.n_failed if group is not None else 0,
    }
    return row


def _group_metric(
    group: object, metric: str
) -> str:
    """Extract a formatted metric from a GroupStats, or N/A."""
    from lhmsb.analysis.stats import GroupStats

    if not isinstance(group, GroupStats):
        return "N/A"
    stats = group.metrics.get(metric)
    if stats is None or stats.mean is None:
        return "N/A"
    return _format_float(stats.mean)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _write_markdown(scorecard: Scorecard, path: Path) -> None:
    """Write the scorecard as Markdown with separate track sections."""
    lines: list[str] = []
    lines.append("# LongHorizonMemSysBench Scorecard")
    lines.append("")

    for track_label, track_results in (
        ("Native Track", scorecard.native_results),
        ("Controlled Track", scorecard.controlled_results),
    ):
        lines.append(f"## {track_label}")
        lines.append("")

        roi_by_fam = _roi_by_family(track_results.roi_results)

        for family in sorted(roi_by_fam):
            lines.append(f"### {family}")
            lines.append("")
            family_results = roi_by_fam[family]
            _render_markdown_table(
                lines, family_results, track_results.aggregated, track_results.track
            )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_markdown_table(
    lines: list[str],
    roi_results: list[MemoryRoiResult],
    agg: AggregatedStats,
    track: str,
) -> None:
    """Render one Markdown table for a family's ROI results."""
    # Header
    header_cols = [
        "condition",
        "task_score",
        "utilization",
        "drift_index",
        "retrieval_endo_p",
        "retrieval_oracle_p",
        "stored_memories",
        "retrieved_memories",
        "storage_p",
        "storage_r",
        "storage_f1",
        "retrieval_p",
        "retrieval_r",
        "retrieval_f1",
        "retrieval_fp",
        "retrieval_timeliness",
        "roi [ci]",
        "pareto",
        "n_failed",
    ]

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cols) + " |")

    for roi in roi_results:
        row = _build_row_dict(roi, agg, track)
        cells = [
            str(row["condition"]),
            str(row["task_score"]),
            str(row["utilization_rate"]),
            str(row["drift_index"]),
            str(row["retrieval_endogenous_precision"]),
            str(row["retrieval_oracle_precision"]),
            str(row["stored_memory_count"]),
            str(row["retrieved_memory_count"]),
            str(row["storage_precision"]),
            str(row["storage_recall"]),
            str(row["storage_f1"]),
            str(row["retrieval_precision"]),
            str(row["retrieval_recall"]),
            str(row["retrieval_f1"]),
            str(row["retrieval_false_positive_rate"]),
            str(row["retrieval_timeliness"]),
            str(row["roi"]),
            str(row["pareto_status"]),
            str(row["n_failed"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")


# --------------------------------------------------------------------------- #
# CSV rendering
# --------------------------------------------------------------------------- #
def _write_csv(scorecard: Scorecard, path: Path) -> None:
    """Write the scorecard as CSV with a track column."""
    all_rows: list[dict[str, object]] = []
    for track_results in (scorecard.native_results, scorecard.controlled_results):
        for roi in track_results.roi_results:
            all_rows.append(
                _build_row_dict(roi, track_results.aggregated, track_results.track)
            )

    if not all_rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(all_rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)
    path.write_text(buf.getvalue(), encoding="utf-8")


# --------------------------------------------------------------------------- #
# JSON rendering
# --------------------------------------------------------------------------- #
def _write_json(scorecard: Scorecard, path: Path) -> None:
    """Write the scorecard as JSON with separate native/controlled keys."""
    data: dict[str, list[dict[str, object]]] = {"native": [], "controlled": []}
    for track_results in (scorecard.native_results, scorecard.controlled_results):
        track_key = track_results.track
        for roi in track_results.roi_results:
            row = _build_row_dict(roi, track_results.aggregated, track_key)
            # Convert non-JSON-serializable values
            clean_row: dict[str, object] = {}
            for key, value in row.items():
                if isinstance(value, float) and math.isnan(value):
                    clean_row[key] = None
                else:
                    clean_row[key] = value
            data[track_key].append(clean_row)

    path.write_text(
        json.dumps(data, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Pareto plots (matplotlib, headless Agg backend)
# --------------------------------------------------------------------------- #
def _write_pareto_plots(scorecard: Scorecard, out_dir: Path) -> None:
    """Render Pareto-front plots per family + overall (headless matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for track_results in (scorecard.native_results, scorecard.controlled_results):
        roi_by_fam = _roi_by_family(track_results.roi_results)

        # Per-family plots
        for family, results in roi_by_fam.items():
            if family == OVERALL_FAMILY:
                continue
            _render_pareto_figure(
                results,
                out_dir / f"pareto_{family}.png",
                title=f"Pareto Front: {family} ({track_results.track})",
                plt=plt,
            )

        # Overall plot
        overall_results = roi_by_fam.get(OVERALL_FAMILY, [])
        if overall_results:
            _render_pareto_figure(
                overall_results,
                out_dir / "pareto_overall.png",
                title=f"Pareto Front: overall ({track_results.track})",
                plt=plt,
            )

    plt.close("all")


def _render_pareto_figure(
    results: list[MemoryRoiResult],
    path: Path,
    title: str,
    plt: object,
) -> None:
    """Render one Pareto scatter plot and save to disk."""
    from typing import Any

    plt_any: Any = plt
    fig, ax = plt_any.subplots(figsize=(8, 5))

    front_x: list[float] = []
    front_y: list[float] = []
    front_labels: list[str] = []
    dom_x: list[float] = []
    dom_y: list[float] = []
    dom_labels: list[str] = []

    for result in results:
        if result.is_baseline:
            continue
        x = result.mean_memory_count
        y = result.mean_normalized_gain
        if result.pareto_status == PARETO_ON_FRONT:
            front_x.append(x)
            front_y.append(y)
            front_labels.append(result.condition)
        elif result.pareto_status == PARETO_DOMINATED:
            dom_x.append(x)
            dom_y.append(y)
            dom_labels.append(result.condition)

    if dom_x:
        ax.scatter(dom_x, dom_y, c="gray", marker="x", s=60, label="Dominated")
        for label, x, y in zip(dom_labels, dom_x, dom_y, strict=True):
            ax.annotate(label, (x, y), fontsize=7, ha="left")

    if front_x:
        ax.scatter(
            front_x, front_y, c="blue", marker="o", s=80, label="Pareto front"
        )
        for label, x, y in zip(front_labels, front_x, front_y, strict=True):
            ax.annotate(label, (x, y), fontsize=8, fontweight="bold", ha="left")

    ax.set_xlabel("Mean Recorded Memories")
    ax.set_ylabel("Mean Normalized Gain")
    ax.set_title(title)
    if front_x or dom_x:
        ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt_any.close(fig)
