"""Episode-clustered inference for schema-v2 qualification reports.

SCEUs are repeated measurements inside an episode.  This module first reduces
each metric to one value per episode/cell, then resamples or pairs episodes.  It
therefore never inflates the inferential sample size by treating probes from the
same generated trajectory as independent observations.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from lhmsb.qualification.metrics import MultisystemMetricInput

BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 10_000
ALPHA = 0.05

_Cell = tuple[str, str, str]


@dataclass(frozen=True)
class _Interval:
    point: float
    low: float
    high: float


def compute_episode_cluster_statistics(
    observations: Sequence[MultisystemMetricInput],
    *,
    episode_groups: Mapping[str, str] | None = None,
    episode_group_name: str = "semantic_scenario",
    seed: int = 20260720,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    permutation_resamples: int = PERMUTATION_RESAMPLES,
) -> dict[str, object]:
    """Return deterministic episode-level estimates and paired comparisons."""
    if bootstrap_resamples < 1 or permutation_resamples < 1:
        raise ValueError("resample counts must be positive")
    metrics: dict[str, Callable[[Sequence[MultisystemMetricInput]], float | None]] = {
        "mean_behavior_score": _mean_behavior_score,
        "behavior_correct_rate": _behavior_correct_rate,
        "eligible_drift_rate": _eligible_drift_rate,
        "observed_drift_rate": _observed_drift_rate,
        "causal_memory_use_rate": _causal_memory_use_rate,
    }
    by_episode_cell: dict[
        tuple[str, _Cell], list[MultisystemMetricInput]
    ] = defaultdict(list)
    for row in observations:
        if not row.episode_id:
            continue
        cell = (row.policy_profile_id, row.condition, row.readout)
        by_episode_cell[(row.episode_id, cell)].append(row)

    episode_values: dict[
        tuple[_Cell, str], dict[str, float]
    ] = defaultdict(dict)
    for (episode_id, cell), rows in sorted(by_episode_cell.items()):
        for metric_name, accessor in metrics.items():
            value = accessor(rows)
            if value is not None:
                episode_values[(cell, metric_name)][episode_id] = value

    cells: list[dict[str, object]] = []
    for index, ((cell, metric_name), values_by_episode) in enumerate(
        sorted(episode_values.items())
    ):
        values = [values_by_episode[key] for key in sorted(values_by_episode)]
        interval = _bootstrap_interval(
            values,
            seed=seed + index,
            n_resamples=bootstrap_resamples,
        )
        cells.append(
            {
                "policy_profile_id": cell[0],
                "condition": cell[1],
                "readout": cell[2],
                "metric": metric_name,
                "analysis_unit": "episode",
                "n_episodes": len(values),
                "mean": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "alpha": ALPHA,
            }
        )

    raw_comparisons: list[dict[str, object]] = []
    policies = sorted({cell[0] for cell, _ in episode_values})
    for policy in policies:
        policy_cells = sorted(
            {
                cell
                for cell, _metric in episode_values
                if cell[0] == policy
            }
        )
        pairs = _comparison_pairs(policy_cells)
        for metric_name in ("mean_behavior_score", "behavior_correct_rate"):
            for pair_index, (left, right, contrast) in enumerate(pairs):
                left_values = episode_values.get((left, metric_name), {})
                right_values = episode_values.get((right, metric_name), {})
                shared = sorted(set(left_values).intersection(right_values))
                if not shared:
                    continue
                differences = [
                    left_values[episode] - right_values[episode]
                    for episode in shared
                ]
                interval = _bootstrap_interval(
                    differences,
                    seed=seed + 10_000 + len(raw_comparisons),
                    n_resamples=bootstrap_resamples,
                )
                std = statistics.stdev(differences) if len(differences) > 1 else 0.0
                effect = interval.point / std if std else 0.0
                p_value = _paired_sign_flip_p_value(
                    differences,
                    seed=seed + 20_000 + pair_index + len(raw_comparisons),
                    n_resamples=permutation_resamples,
                )
                raw_comparisons.append(
                    {
                        "policy_profile_id": policy,
                        "metric": metric_name,
                        "contrast": contrast,
                        "left_condition": left[1],
                        "left_readout": left[2],
                        "right_condition": right[1],
                        "right_readout": right[2],
                        "analysis_unit": "paired_episode",
                        "n_pairs": len(shared),
                        "mean_difference": interval.point,
                        "ci_low": interval.low,
                        "ci_high": interval.high,
                        "paired_cohens_dz": effect,
                        "permutation_p_value": p_value,
                        "minimum_detectable_dz_80pct": _minimum_detectable_effect(
                            len(shared)
                        ),
                    }
                )

    adjusted = _holm_adjust(raw_comparisons)
    group_payload = _group_cluster_sensitivity(
        episode_values,
        episode_groups or {},
        group_name=episode_group_name,
        seed=seed + 30_000,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
    )
    return {
        "schema_version": 1,
        "analysis_unit": "episode",
        "within_episode_unit": "SCEU",
        "alpha": ALPHA,
        "bootstrap_resamples": bootstrap_resamples,
        "permutation_resamples": permutation_resamples,
        "seed": seed,
        "n_unique_episodes": len(
            {episode_id for episode_id, _cell in by_episode_cell}
        ),
        "cells": cells,
        "paired_comparisons": adjusted,
        **group_payload,
        "notes": [
            "Confidence intervals resample episodes, not SCEUs.",
            "Paired tests retain only episodes observed in both compared cells.",
            "Permutation p-values use paired sign flips and Holm correction within each metric.",
            (
                "Semantic-scenario sensitivity first averages schedules within each "
                "scenario and never treats 50 generated trajectories as 50 distinct "
                "semantic templates."
            ),
            (
                "Exact and inferred storage-provenance tracks remain descriptive "
                "unless enough episode-level exact observations exist."
            ),
        ],
    }


def statistics_markdown(payload: dict[str, object]) -> str:
    """Render a concise human-readable companion to ``statistics.json``."""
    lines = [
        "# Episode-clustered statistical report",
        "",
        (
            f"Analysis unit: **episode**; unique episodes: "
            f"**{payload.get('n_unique_episodes', 0)}**. SCEUs are repeated "
            "measurements and are not treated as independent samples."
        ),
        "",
        "## Cell estimates",
        "",
        "| Policy | Condition | Readout | Metric | n | Mean | 95% CI |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for raw in _record_sequence(payload.get("cells")):
        lines.append(
            "| {policy} | {condition} | {readout} | {metric} | {n} | {mean:.4f} | "
            "[{low:.4f}, {high:.4f}] |".format(
                policy=raw.get("policy_profile_id", ""),
                condition=raw.get("condition", ""),
                readout=raw.get("readout", ""),
                metric=raw.get("metric", ""),
                n=int(_number(raw.get("n_episodes", 0))),
                mean=_number(raw.get("mean", 0.0)),
                low=_number(raw.get("ci_low", 0.0)),
                high=_number(raw.get("ci_high", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Paired episode contrasts",
            "",
            "| Metric | Contrast | n | Difference | 95% CI | d_z | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for raw in _record_sequence(payload.get("paired_comparisons")):
        lines.append(
            "| {metric} | {contrast} | {n} | {diff:.4f} | [{low:.4f}, "
            "{high:.4f}] | {effect:.3f} | {p:.4g} |".format(
                metric=raw.get("metric", ""),
                contrast=raw.get("contrast", ""),
                n=int(_number(raw.get("n_pairs", 0))),
                diff=_number(raw.get("mean_difference", 0.0)),
                low=_number(raw.get("ci_low", 0.0)),
                high=_number(raw.get("ci_high", 0.0)),
                effect=_number(raw.get("paired_cohens_dz", 0.0)),
                p=_number(raw.get("holm_adjusted_p_value", 1.0)),
            )
        )
    scenario_cells = _record_sequence(payload.get("scenario_cells"))
    if scenario_cells:
        lines.extend(
            [
                "",
                "## Semantic-scenario sensitivity",
                "",
                (
                    f"Grouping variable: **{payload.get('episode_group_name', '')}**; "
                    f"unique groups: **{payload.get('n_unique_episode_groups', 0)}**. "
                    "Schedules are averaged within scenario before resampling."
                ),
                "",
                "| Policy | Condition | Readout | Metric | groups | Mean | 95% CI | LOO range |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for raw in scenario_cells:
            lines.append(
                "| {policy} | {condition} | {readout} | {metric} | {n} | "
                "{mean:.4f} | [{low:.4f}, {high:.4f}] | [{loo_low:.4f}, "
                "{loo_high:.4f}] |".format(
                    policy=raw.get("policy_profile_id", ""),
                    condition=raw.get("condition", ""),
                    readout=raw.get("readout", ""),
                    metric=raw.get("metric", ""),
                    n=int(_number(raw.get("n_groups", 0))),
                    mean=_number(raw.get("mean", 0.0)),
                    low=_number(raw.get("ci_low", 0.0)),
                    high=_number(raw.get("ci_high", 0.0)),
                    loo_low=_number(raw.get("leave_one_group_out_low", 0.0)),
                    loo_high=_number(raw.get("leave_one_group_out_high", 0.0)),
                )
            )
    lines.extend(
        [
            "",
            "Interpretation should emphasize effect sizes and confidence intervals; "
            "corrected p-values are secondary evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _group_cluster_sensitivity(
    episode_values: Mapping[tuple[_Cell, str], Mapping[str, float]],
    episode_groups: Mapping[str, str],
    *,
    group_name: str,
    seed: int,
    bootstrap_resamples: int,
    permutation_resamples: int,
) -> dict[str, object]:
    valid_groups = {
        episode_id: str(group)
        for episode_id, group in episode_groups.items()
        if str(group)
    }
    if not valid_groups:
        return {
            "episode_group_name": group_name,
            "n_unique_episode_groups": 0,
            "scenario_cells": [],
            "scenario_paired_comparisons": [],
        }

    group_values: dict[
        tuple[_Cell, str], dict[str, float]
    ] = defaultdict(dict)
    for key, values_by_episode in episode_values.items():
        collected: dict[str, list[float]] = defaultdict(list)
        for episode_id, value in values_by_episode.items():
            group = valid_groups.get(episode_id)
            if group is not None:
                collected[group].append(value)
        group_values[key] = {
            group: statistics.fmean(values)
            for group, values in sorted(collected.items())
        }

    cells: list[dict[str, object]] = []
    for index, ((cell, metric_name), values_by_group) in enumerate(
        sorted(group_values.items())
    ):
        values = [values_by_group[key] for key in sorted(values_by_group)]
        if not values:
            continue
        interval = _bootstrap_interval(
            values,
            seed=seed + index,
            n_resamples=bootstrap_resamples,
        )
        loo = _leave_one_out_means(values)
        cells.append(
            {
                "policy_profile_id": cell[0],
                "condition": cell[1],
                "readout": cell[2],
                "metric": metric_name,
                "analysis_unit": f"{group_name}_cluster",
                "n_groups": len(values),
                "mean": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "leave_one_group_out_low": min(loo),
                "leave_one_group_out_high": max(loo),
                "alpha": ALPHA,
            }
        )

    raw_comparisons: list[dict[str, object]] = []
    policies = sorted({cell[0] for cell, _metric in group_values})
    for policy in policies:
        policy_cells = sorted(
            {cell for cell, _metric in group_values if cell[0] == policy}
        )
        for metric_name in ("mean_behavior_score", "behavior_correct_rate"):
            for pair_index, (left, right, contrast) in enumerate(
                _comparison_pairs(policy_cells)
            ):
                left_values = group_values.get((left, metric_name), {})
                right_values = group_values.get((right, metric_name), {})
                shared = sorted(set(left_values).intersection(right_values))
                if not shared:
                    continue
                differences = [
                    left_values[group] - right_values[group] for group in shared
                ]
                interval = _bootstrap_interval(
                    differences,
                    seed=seed + 10_000 + len(raw_comparisons),
                    n_resamples=bootstrap_resamples,
                )
                std = statistics.stdev(differences) if len(differences) > 1 else 0.0
                raw_comparisons.append(
                    {
                        "policy_profile_id": policy,
                        "metric": metric_name,
                        "contrast": contrast,
                        "left_condition": left[1],
                        "left_readout": left[2],
                        "right_condition": right[1],
                        "right_readout": right[2],
                        "analysis_unit": f"paired_{group_name}",
                        "n_pairs": len(shared),
                        "mean_difference": interval.point,
                        "ci_low": interval.low,
                        "ci_high": interval.high,
                        "paired_cohens_dz": interval.point / std if std else 0.0,
                        "permutation_p_value": _paired_sign_flip_p_value(
                            differences,
                            seed=seed + 20_000 + pair_index + len(raw_comparisons),
                            n_resamples=permutation_resamples,
                        ),
                        "minimum_detectable_dz_80pct": _minimum_detectable_effect(
                            len(shared)
                        ),
                    }
                )
    return {
        "episode_group_name": group_name,
        "n_unique_episode_groups": len(set(valid_groups.values())),
        "scenario_cells": cells,
        "scenario_paired_comparisons": _holm_adjust(raw_comparisons),
    }


def _leave_one_out_means(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) <= 1:
        return (statistics.fmean(values),)
    return tuple(
        statistics.fmean((*values[:index], *values[index + 1 :]))
        for index in range(len(values))
    )


def _comparison_pairs(cells: Sequence[_Cell]) -> tuple[tuple[_Cell, _Cell, str], ...]:
    by_condition = {(cell[1], cell[2]): cell for cell in cells}
    workspace = by_condition.get(("workspace_only", "none"))
    flat = by_condition.get(("flat_retrieval", "common_rerank"))
    output: list[tuple[_Cell, _Cell, str]] = []
    memory_cells = [
        cell for cell in cells if cell[1] in {"flat_retrieval", "mem0", "amem", "memos"}
    ]
    if workspace is not None:
        output.extend(
            (cell, workspace, "gain_beyond_workspace")
            for cell in memory_cells
        )
    if flat is not None:
        output.extend(
            (cell, flat, "gain_over_flat_retrieval")
            for cell in memory_cells
            if cell != flat
        )
    for condition in ("mem0", "amem", "memos"):
        native = by_condition.get((condition, "native"))
        common = by_condition.get((condition, "common_rerank"))
        if native is not None and common is not None:
            output.append((common, native, "common_rerank_minus_native"))
    return tuple(output)


def _mean_behavior_score(rows: Sequence[MultisystemMetricInput]) -> float:
    return statistics.fmean(row.behavior_score for row in rows)


def _behavior_correct_rate(rows: Sequence[MultisystemMetricInput]) -> float:
    return statistics.fmean(float(row.is_correct) for row in rows)


def _eligible_drift_rate(
    rows: Sequence[MultisystemMetricInput],
) -> float | None:
    eligible = [
        row
        for row in rows
        if row.drift_eligible_categories is None
        or bool(row.drift_eligible_categories)
    ]
    if not eligible:
        return None
    return statistics.fmean(
        float(
            bool(
                set(row.drift_flags).intersection(
                    {
                        "constraint_loss",
                        "plan_deviation",
                        "stale_state",
                        "local_over_global",
                    }
                    if row.drift_eligible_categories is None
                    else set(row.drift_eligible_categories)
                )
            )
        )
        for row in eligible
    )


def _observed_drift_rate(
    rows: Sequence[MultisystemMetricInput],
) -> float | None:
    if not rows:
        return None
    canonical = {
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    }
    return statistics.fmean(
        float(bool(canonical.intersection(row.drift_flags))) for row in rows
    )


def _causal_memory_use_rate(
    rows: Sequence[MultisystemMetricInput],
) -> float | None:
    labels = [label for row in rows for label in row.causal_labels]
    if not labels:
        return None
    return statistics.fmean(
        float(label in {"beneficial", "harmful"}) for label in labels
    )


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    n_resamples: int,
) -> _Interval:
    if not values:
        raise ValueError("bootstrap requires at least one observation")
    point = statistics.fmean(values)
    if len(values) == 1 or len(set(values)) == 1:
        return _Interval(point, point, point)
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        statistics.fmean(rng.choices(values, k=size))
        for _ in range(n_resamples)
    )
    low_index = max(0, int((ALPHA / 2) * n_resamples) - 1)
    high_index = min(
        n_resamples - 1,
        int((1 - ALPHA / 2) * n_resamples),
    )
    return _Interval(point, estimates[low_index], estimates[high_index])


def _paired_sign_flip_p_value(
    differences: Sequence[float],
    *,
    seed: int,
    n_resamples: int,
) -> float:
    if not differences or all(value == 0 for value in differences):
        return 1.0
    observed = abs(statistics.fmean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        permuted = statistics.fmean(
            value if rng.random() < 0.5 else -value
            for value in differences
        )
        extreme += abs(permuted) >= observed - 1e-15
    return (extreme + 1) / (n_resamples + 1)


def _holm_adjust(
    comparisons: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    output = [dict(item) for item in comparisons]
    by_metric: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        by_metric[str(row["metric"])].append(index)
    for indices in by_metric.values():
        ordered = sorted(
            indices,
            key=lambda index: _number(output[index]["permutation_p_value"]),
        )
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            raw = _number(output[index]["permutation_p_value"])
            adjusted = min(1.0, raw * (total - rank))
            running = max(running, adjusted)
            output[index]["holm_adjusted_p_value"] = running
            output[index]["significant_after_holm"] = running < ALPHA
    return output


def _record_sequence(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("statistical value must be numeric")
    return float(value)


def _minimum_detectable_effect(n_pairs: int) -> float | None:
    if n_pairs < 2:
        return None
    # Normal approximation for a two-sided paired test at alpha=.05, power=.80.
    return (1.959963984540054 + 0.8416212335729143) / math.sqrt(n_pairs)


__all__ = [
    "ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "PERMUTATION_RESAMPLES",
    "compute_episode_cluster_statistics",
    "statistics_markdown",
]
