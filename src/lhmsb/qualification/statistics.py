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

from lhmsb.qualification.horizon_panel import (
    HORIZON_PRIMARY_ESTIMANDS,
    HORIZON_SECONDARY_ESTIMANDS,
    compute_horizon_panel_contrasts,
)
from lhmsb.qualification.longitudinal import (
    episode_observed_drift_incidence,
)
from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    compute_matched_construct_contrasts,
)

BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 10_000
ALPHA = 0.05

MATCHED_STATISTICS_SCHEMA_VERSION = 2
MATCHED_PRIMARY_ESTIMANDS = (
    "state_evolution_penalty_excess_over_workspace",
    "hierarchical_conflict_penalty_excess_over_workspace",
)
MATCHED_SECONDARY_ESTIMANDS = (
    "state_evolution_penalty_vs_static",
    "hierarchical_conflict_penalty_vs_static",
    "state_evolution_correctness_penalty_vs_static",
    "hierarchical_conflict_correctness_penalty_vs_static",
    "state_evolution_drift_violation_excess_vs_static",
    "hierarchical_conflict_drift_violation_excess_vs_static",
)
MATCHED_ALL_ESTIMANDS = (
    *MATCHED_PRIMARY_ESTIMANDS,
    *MATCHED_SECONDARY_ESTIMANDS,
)

MATCHED_PRIMARY_ANALYSIS_UNIT = "counterfactual_group"
MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT = (
    "matched_workspace_only_difference_in_differences"
)
MATCHED_PRIMARY_EFFECT_DIRECTION = (
    "positive_means_additional_degradation_beyond_workspace"
)
MATCHED_DRIFT_SCOPE = "endpoint_violation_only"
MATCHED_PAIRED_TEST = "sign_flip"
MATCHED_MULTIPLICITY_SCOPE = (
    "within_estimand_across_policy_condition_readout_cells"
)

HORIZON_STATISTICS_SCHEMA_VERSION = 1
HORIZON_ALL_ESTIMANDS = (
    *HORIZON_PRIMARY_ESTIMANDS,
    *HORIZON_SECONDARY_ESTIMANDS,
)
HORIZON_PRIMARY_ANALYSIS_UNIT = "horizon_panel"
HORIZON_PRIMARY_WORKSPACE_ADJUSTMENT = (
    "difference_in_differences_in_differences_against_workspace_only"
)
HORIZON_PRIMARY_EFFECT_DIRECTION = (
    "positive_means_construct_penalty_grows_more_from_short_to_long_than_"
    "the_matched_workspace_only_penalty"
)
HORIZON_PAIRED_TEST = "panel_level_sign_flip"
HORIZON_MULTIPLICITY_SCOPE = "two_primary_horizon_amplification_estimands"

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
        "eligible_drift_violation_rate": _eligible_drift_rate,
        "canonical_drift_violation_rate": _canonical_drift_violation_rate,
        "observed_longitudinal_drift_incidence": (
            episode_observed_drift_incidence
        ),
        # Schema-v3 compatibility aliases. New analyses should use the
        # explicitly named violation and longitudinal metrics above.
        "eligible_drift_rate": _eligible_drift_rate,
        "observed_drift_rate": _canonical_drift_violation_rate,
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
        for metric_name in (
            "mean_behavior_score",
            "behavior_correct_rate",
            "eligible_drift_violation_rate",
            "canonical_drift_violation_rate",
            "observed_longitudinal_drift_incidence",
            "eligible_drift_rate",
        ):
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
            (
                "Drift-compatible violations and adherence-anchored longitudinal "
                "drift are distinct; first-observation errors never enter the "
                "longitudinal drift incidence."
            ),
            (
                "eligible_drift_rate and observed_drift_rate are compatibility "
                "aliases for violation rates and are not used as longitudinal "
                "drift estimands."
            ),
        ],
    }


def statistics_markdown(payload: dict[str, object]) -> str:
    """Render a concise human-readable companion to ``statistics.json``."""
    if payload.get("status") == "suppressed_dependent_physical_members":
        return "\n".join(
            (
                "# Generic episode statistics suppressed",
                "",
                str(payload.get("reason", "")),
                "",
                (
                    "Declared primary analysis unit: "
                    f"**{payload.get('analysis_unit', 'missing')}**."
                ),
                "",
            )
        )
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


def compute_matched_group_statistics(
    observations: Sequence[MultisystemMetricInput],
    *,
    seed: int = 20260723,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    permutation_resamples: int = PERMUTATION_RESAMPLES,
) -> dict[str, object]:
    """Estimate matched history effects using counterfactual groups as units.

    Static, evolution, and conflict are three members of one counterfactual
    group, not three independent episodes.  The contrast builder first reduces
    them to one within-group effect; only those effects are bootstrapped and
    sign-flipped here.
    """

    if bootstrap_resamples < 1 or permutation_resamples < 1:
        raise ValueError("resample counts must be positive")
    contrasts = tuple(
        row
        for row in compute_matched_construct_contrasts(observations)
        if row.get("complete") is True
    )
    by_cell: dict[tuple[str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in contrasts:
        by_cell[
            (
                str(row["policy_profile_id"]),
                str(row["condition"]),
                str(row["readout"]),
            )
        ].append(row)

    raw_estimates: list[dict[str, object]] = []
    for cell_index, (cell, rows) in enumerate(sorted(by_cell.items())):
        for estimand_index, estimand in enumerate(MATCHED_ALL_ESTIMANDS):
            values = [
                float(value)
                for row in rows
                if (value := row.get(estimand)) is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            if not values:
                continue
            interval = _bootstrap_interval(
                values,
                seed=seed + cell_index * 100 + estimand_index,
                n_resamples=bootstrap_resamples,
            )
            standard_deviation = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            raw_estimates.append(
                {
                    "policy_profile_id": cell[0],
                    "condition": cell[1],
                    "readout": cell[2],
                    "metric": estimand,
                    "estimand_role": (
                        "primary"
                        if estimand in MATCHED_PRIMARY_ESTIMANDS
                        else "secondary"
                    ),
                    "contrast": "matched_history_vs_static",
                    "analysis_unit": "counterfactual_group",
                    "n_pairs": len(values),
                    "mean_difference": interval.point,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "paired_cohens_dz": (
                        interval.point / standard_deviation
                        if standard_deviation
                        else 0.0
                    ),
                    "permutation_p_value": _paired_sign_flip_p_value(
                        values,
                        seed=(
                            seed
                            + 10_000
                            + cell_index * 100
                            + estimand_index
                        ),
                        n_resamples=permutation_resamples,
                    ),
                    "minimum_detectable_dz_80pct": (
                        _minimum_detectable_effect(len(values))
                    ),
                    "effect_direction": (
                        "positive means worse than the matched static history"
                    ),
                }
            )

    archetype_rows: list[dict[str, object]] = []
    archetype_groups: dict[
        tuple[str, str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in contrasts:
        archetype_groups[
            (
                str(row["policy_profile_id"]),
                str(row["condition"]),
                str(row["readout"]),
                str(row.get("terminal_archetype", "")),
            )
        ].append(row)
    for key, rows in sorted(archetype_groups.items()):
        for estimand in MATCHED_ALL_ESTIMANDS:
            values = [
                float(value)
                for row in rows
                if (value := row.get(estimand)) is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            if not values:
                continue
            archetype_rows.append(
                {
                    "policy_profile_id": key[0],
                    "condition": key[1],
                    "readout": key[2],
                    "terminal_archetype": key[3],
                    "metric": estimand,
                    "estimand_role": (
                        "primary"
                        if estimand in MATCHED_PRIMARY_ESTIMANDS
                        else "secondary"
                    ),
                    "analysis_unit": "counterfactual_group",
                    "n_groups": len(values),
                    "mean_difference": statistics.fmean(values),
                }
            )

    adjusted = _holm_adjust(raw_estimates)
    return {
        "schema_version": MATCHED_STATISTICS_SCHEMA_VERSION,
        "analysis_unit": "counterfactual_group",
        "primary_analysis_unit": MATCHED_PRIMARY_ANALYSIS_UNIT,
        "physical_member_role": "within_group_repeated_condition",
        "primary_estimands": list(MATCHED_PRIMARY_ESTIMANDS),
        "secondary_estimands": list(MATCHED_SECONDARY_ESTIMANDS),
        "primary_workspace_adjustment": MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT,
        "primary_effect_direction": MATCHED_PRIMARY_EFFECT_DIRECTION,
        "drift_scope": MATCHED_DRIFT_SCOPE,
        "paired_test": MATCHED_PAIRED_TEST,
        "multiplicity_scope": MATCHED_MULTIPLICITY_SCOPE,
        "alpha": ALPHA,
        "bootstrap_resamples": bootstrap_resamples,
        "permutation_resamples": permutation_resamples,
        "seed": seed,
        "n_complete_group_cells": len(contrasts),
        "n_unique_counterfactual_groups": len(
            {str(row["counterfactual_group_id"]) for row in contrasts}
        ),
        "estimates": adjusted,
        "terminal_archetype_sensitivity": archetype_rows,
        "notes": [
            "Each static/evolution/conflict triplet contributes one paired effect.",
            "Physical triplet members are never treated as independent samples.",
            "Positive effects denote degradation relative to matched static history.",
            (
                "Penalty-excess-over-workspace estimands are difference-in-differences: "
                "they subtract the matched workspace-only construct penalty so that "
                "workspace-surface difficulty is not attributed to the memory channel."
            ),
            (
                "Drift fields in this matched endpoint report are violation "
                "excesses, not longitudinal onset."
            ),
            "Confidence intervals resample counterfactual groups; p-values use paired sign flips.",
        ],
    }


def matched_group_statistics_markdown(payload: Mapping[str, object]) -> str:
    """Render matched state-control mechanism estimates."""

    if payload.get("status") == "suppressed_within_panel_triplets":
        return "\n".join(
            (
                "# Within-dose matched statistics suppressed",
                "",
                str(payload.get("reason", "")),
                "",
                "Declared analysis unit: **horizon panel**.",
                "",
            )
        )

    lines = [
        "# Counterfactually matched state-control statistics",
        "",
        (
            "Analysis unit: **counterfactual group**. Static, evolution, and "
            "hierarchical-conflict members are repeated conditions within a group."
        ),
        "",
        "Positive effects mean worse performance than the matched static history.",
        (
            "For `*_penalty_excess_over_workspace`, positive values mean the "
            "condition loses more performance than workspace-only under the same "
            "history manipulation."
        ),
        "",
        "## Frozen analysis contract",
        "",
        "Primary estimands:",
        *(
            f"- `{estimand}`"
            for estimand in _string_sequence(payload.get("primary_estimands"))
        ),
        "",
        (
            "Primary workspace adjustment: "
            f"`{payload.get('primary_workspace_adjustment', '')}`."
        ),
        (
            "Matched endpoint drift scope: "
            f"`{payload.get('drift_scope', '')}`; it is not longitudinal onset."
        ),
        (
            "Multiplicity scope: "
            f"`{payload.get('multiplicity_scope', '')}`."
        ),
        "",
        "Secondary estimands are descriptive mechanism, correctness, and endpoint-"
        "violation contrasts; they cannot replace the workspace-adjusted primary "
        "analysis.",
        "",
        "## Estimates",
        "",
        "| Role | Policy | Condition | Readout | Estimand | Groups | Mean | 95% CI | "
        "d_z | Holm p |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _record_sequence(payload.get("estimates")):
        lines.append(
            "| {role} | {policy} | {condition} | {readout} | `{metric}` | {n} | "
            "{mean:.4f} | [{low:.4f}, {high:.4f}] | {effect:.3f} | {p:.4g} |".format(
                role=row.get("estimand_role", ""),
                policy=row.get("policy_profile_id", ""),
                condition=row.get("condition", ""),
                readout=row.get("readout", ""),
                metric=row.get("metric", ""),
                n=int(_number(row.get("n_pairs", 0))),
                mean=_number(row.get("mean_difference", 0.0)),
                low=_number(row.get("ci_low", 0.0)),
                high=_number(row.get("ci_high", 0.0)),
                effect=_number(row.get("paired_cohens_dz", 0.0)),
                p=_number(row.get("holm_adjusted_p_value", 1.0)),
            )
        )
    lines.extend(
        [
            "",
            "Matched drift estimands are drift-compatible violation excesses. "
            "Longitudinal onset and survival are reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


def compute_horizon_panel_statistics(
    observations: Sequence[MultisystemMetricInput],
    *,
    seed: int = 20260723,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    permutation_resamples: int = PERMUTATION_RESAMPLES,
) -> dict[str, object]:
    """Estimate horizon amplification with complete panels as the only units.

    The nine physical members and any repeated target readouts within one panel
    are reduced to one value before resampling. Transition count, dependency
    depth, and session handoffs change jointly, so these estimates are not
    interpreted as a pure handoff effect.
    """

    if bootstrap_resamples < 1 or permutation_resamples < 1:
        raise ValueError("resample counts must be positive")
    contrasts = tuple(
        row
        for row in compute_horizon_panel_contrasts(observations)
        if row.get("complete") is True
    )
    panel_values: dict[
        tuple[_Cell, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in contrasts:
        cell = (
            str(row["policy_profile_id"]),
            str(row["condition"]),
            str(row["readout"]),
        )
        panel_id = str(row["horizon_panel_id"])
        for estimand in HORIZON_ALL_ESTIMANDS:
            value = row.get(estimand)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                panel_values[(cell, estimand)][panel_id].append(float(value))

    raw_estimates: list[dict[str, object]] = []
    for cell_index, ((cell, estimand), values_by_panel) in enumerate(
        sorted(panel_values.items())
    ):
        values = [
            statistics.fmean(values_by_panel[panel_id])
            for panel_id in sorted(values_by_panel)
        ]
        if not values:
            continue
        interval = _bootstrap_interval(
            values,
            seed=seed + cell_index,
            n_resamples=bootstrap_resamples,
        )
        standard_deviation = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
        raw_estimates.append(
            {
                "policy_profile_id": cell[0],
                "condition": cell[1],
                "readout": cell[2],
                "metric": estimand,
                "estimand_role": (
                    "primary"
                    if estimand in HORIZON_PRIMARY_ESTIMANDS
                    else "secondary"
                ),
                "contrast": "long_vs_short_joint_horizon_dose",
                "analysis_unit": HORIZON_PRIMARY_ANALYSIS_UNIT,
                "n_panels": len(values),
                "mean_difference": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "paired_cohens_dz": (
                    interval.point / standard_deviation
                    if standard_deviation
                    else 0.0
                ),
                "permutation_p_value": _paired_sign_flip_p_value(
                    values,
                    seed=seed + 10_000 + cell_index,
                    n_resamples=permutation_resamples,
                ),
                "minimum_detectable_dz_80pct": _minimum_detectable_effect(
                    len(values)
                ),
                "effect_direction": HORIZON_PRIMARY_EFFECT_DIRECTION,
            }
        )

    adjusted = _holm_adjust(raw_estimates)
    return {
        "schema_version": HORIZON_STATISTICS_SCHEMA_VERSION,
        "analysis_role": "supplementary_construct_validity_diagnostic",
        "analysis_unit": HORIZON_PRIMARY_ANALYSIS_UNIT,
        "primary_analysis_unit": HORIZON_PRIMARY_ANALYSIS_UNIT,
        "physical_member_role": "within_panel_repeated_condition",
        "horizon_axis": (
            "joint_effective_transition_and_session_handoff_dose"
        ),
        "reference_horizon_level": "short",
        "intermediate_horizon_level": "medium",
        "target_horizon_level": "long",
        "primary_estimands": list(HORIZON_PRIMARY_ESTIMANDS),
        "secondary_estimands": list(HORIZON_SECONDARY_ESTIMANDS),
        "primary_workspace_adjustment": HORIZON_PRIMARY_WORKSPACE_ADJUSTMENT,
        "primary_effect_direction": HORIZON_PRIMARY_EFFECT_DIRECTION,
        "paired_test": HORIZON_PAIRED_TEST,
        "multiplicity_scope": HORIZON_MULTIPLICITY_SCOPE,
        "alpha": ALPHA,
        "bootstrap_resamples": bootstrap_resamples,
        "permutation_resamples": permutation_resamples,
        "seed": seed,
        "n_complete_panel_cells": len(contrasts),
        "n_unique_horizon_panels": len(
            {str(row["horizon_panel_id"]) for row in contrasts}
        ),
        "estimates": adjusted,
        "notes": [
            "Each complete 3-construct by 3-dose grid contributes one panel effect.",
            "Nine physical members are never treated as independent samples.",
            (
                "Primary estimands are workspace-adjusted triple differences in "
                "construct-penalty amplification from short to long."
            ),
            (
                "The dose jointly changes effective transitions, dependency depth, "
                "and session handoffs; it is not a pure handoff effect."
            ),
            (
                "Confidence intervals resample panels and p-values use panel-level "
                "sign flips."
            ),
        ],
    }


def horizon_panel_statistics_markdown(payload: Mapping[str, object]) -> str:
    """Render the preregistered horizon-panel diagnostic."""

    lines = [
        "# Same-decision horizon-dose statistics",
        "",
        (
            "Analysis unit: **horizon panel**. Each panel contains nine dependent "
            "physical members; those members are never counted as independent n."
        ),
        "",
        (
            "The short-to-long dose jointly increases effective transitions, "
            "dependency depth, and session handoffs. Results therefore diagnose "
            "horizon sensitivity but do not identify a pure handoff effect."
        ),
        "",
        "Primary estimands:",
        *(
            f"- `{estimand}`"
            for estimand in _string_sequence(payload.get("primary_estimands"))
        ),
        "",
        (
            "| Role | Policy | Condition | Readout | Estimand | Panels | Mean | "
            "95% CI | d_z | Holm p |"
        ),
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _record_sequence(payload.get("estimates")):
        lines.append(
            "| {role} | {policy} | {condition} | {readout} | `{metric}` | {n} | "
            "{mean:.4f} | [{low:.4f}, {high:.4f}] | {effect:.3f} | {p:.4g} |".format(
                role=row.get("estimand_role", ""),
                policy=row.get("policy_profile_id", ""),
                condition=row.get("condition", ""),
                readout=row.get("readout", ""),
                metric=row.get("metric", ""),
                n=int(_number(row.get("n_panels", 0))),
                mean=_number(row.get("mean_difference", 0.0)),
                low=_number(row.get("ci_low", 0.0)),
                high=_number(row.get("ci_high", 0.0)),
                effect=_number(row.get("paired_cohens_dz", 0.0)),
                p=_number(row.get("holm_adjusted_p_value", 1.0)),
            )
        )
    lines.extend(
        [
            "",
            "This is supplementary construct-validity evidence and cannot by "
            "itself establish a confirmatory long-horizon effect.",
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
        for metric_name in (
            "mean_behavior_score",
            "behavior_correct_rate",
            "eligible_drift_rate",
        ):
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
    full_context = by_condition.get(("full_context", "none"))
    oracle = by_condition.get(("oracle_current_state", "none"))
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
    if full_context is not None:
        output.extend(
            (cell, full_context, "difference_to_full_context")
            for cell in memory_cells
        )
    if oracle is not None:
        output.extend(
            (cell, oracle, "difference_to_oracle_current_state")
            for cell in memory_cells
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


def _canonical_drift_violation_rate(
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
        float(
            label
            in {
                "beneficial",
                "harmful",
                "causal_direction_ambiguous",
            }
        )
        for label in labels
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


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


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
    "HORIZON_ALL_ESTIMANDS",
    "HORIZON_MULTIPLICITY_SCOPE",
    "HORIZON_PAIRED_TEST",
    "HORIZON_PRIMARY_ANALYSIS_UNIT",
    "HORIZON_PRIMARY_EFFECT_DIRECTION",
    "HORIZON_PRIMARY_WORKSPACE_ADJUSTMENT",
    "HORIZON_STATISTICS_SCHEMA_VERSION",
    "MATCHED_ALL_ESTIMANDS",
    "MATCHED_DRIFT_SCOPE",
    "MATCHED_MULTIPLICITY_SCOPE",
    "MATCHED_PAIRED_TEST",
    "MATCHED_PRIMARY_ANALYSIS_UNIT",
    "MATCHED_PRIMARY_EFFECT_DIRECTION",
    "MATCHED_PRIMARY_ESTIMANDS",
    "MATCHED_PRIMARY_WORKSPACE_ADJUSTMENT",
    "MATCHED_SECONDARY_ESTIMANDS",
    "MATCHED_STATISTICS_SCHEMA_VERSION",
    "PERMUTATION_RESAMPLES",
    "compute_episode_cluster_statistics",
    "compute_horizon_panel_statistics",
    "compute_matched_group_statistics",
    "horizon_panel_statistics_markdown",
    "matched_group_statistics_markdown",
    "statistics_markdown",
]
