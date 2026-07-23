"""Same-decision horizon-dose estimands for supplementary analysis."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from lhmsb.qualification.metrics import MultisystemMetricInput

HORIZON_PANEL_SCHEMA_VERSION = 1
HORIZON_LEVELS = ("short", "medium", "long")
HORIZON_CONSTRUCTS = ("static", "evolution", "hierarchical_conflict")
HORIZON_PRIMARY_ESTIMANDS = (
    "state_evolution_horizon_amplification_excess_over_workspace",
    "hierarchical_conflict_horizon_amplification_excess_over_workspace",
)
HORIZON_SECONDARY_ESTIMANDS = (
    "state_evolution_penalty_amplification_long_vs_short",
    "hierarchical_conflict_penalty_amplification_long_vs_short",
    "static_memory_value_change_long_vs_short",
    "evolution_memory_value_change_long_vs_short",
    "hierarchical_conflict_memory_value_change_long_vs_short",
)


def compute_horizon_panel_contrasts(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Compute one joint horizon-dose contrast per panel and experiment cell.

    Positive ``*_horizon_amplification_excess_over_workspace`` means that the
    condition's construct penalty grows more from short to long than the
    matched workspace-only penalty does. The function does not interpret the
    joint dose as a pure handoff effect.
    """

    grouped: dict[
        tuple[str, str, str, str, str],
        dict[tuple[str, str], list[MultisystemMetricInput]],
    ] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if (
            not observation.is_counterfactual_target
            or not observation.horizon_panel_id
            or observation.horizon_level not in HORIZON_LEVELS
            or observation.counterfactual_variant not in HORIZON_CONSTRUCTS
        ):
            continue
        grouped[
            (
                observation.policy_profile_id,
                observation.condition,
                observation.readout,
                observation.horizon_panel_id,
                observation.opportunity_id,
            )
        ][
            (observation.horizon_level, observation.counterfactual_variant)
        ].append(observation)

    base_rows: list[dict[str, object]] = []
    for cell_key in sorted(grouped):
        cells = grouped[cell_key]
        complete = all(
            cells.get((level, construct))
            for level in HORIZON_LEVELS
            for construct in HORIZON_CONSTRUCTS
        )
        axes = {
            observation.horizon_axis
            for cell_rows in cells.values()
            for observation in cell_rows
            if observation.horizon_axis
        }
        archetypes = {
            observation.counterfactual_terminal_archetype
            for cell_rows in cells.values()
            for observation in cell_rows
            if observation.counterfactual_terminal_archetype
        }
        scores = {
            (level, construct): _mean_behavior(
                cells.get((level, construct), ())
            )
            for level in HORIZON_LEVELS
            for construct in HORIZON_CONSTRUCTS
        }
        correct = {
            (level, construct): _mean_correct(
                cells.get((level, construct), ())
            )
            for level in HORIZON_LEVELS
            for construct in HORIZON_CONSTRUCTS
        }
        contrast: dict[str, object] = {
            "schema_version": HORIZON_PANEL_SCHEMA_VERSION,
            "policy_profile_id": cell_key[0],
            "condition": cell_key[1],
            "readout": cell_key[2],
            "horizon_panel_id": cell_key[3],
            "opportunity_id": cell_key[4],
            "analysis_unit": "horizon_panel",
            "horizon_axis": next(iter(axes)) if len(axes) == 1 else "inconsistent",
            "terminal_archetype": (
                next(iter(archetypes))
                if len(archetypes) == 1
                else "inconsistent"
            ),
            "complete": complete and len(axes) == 1 and len(archetypes) == 1,
        }
        for level in HORIZON_LEVELS:
            for construct in HORIZON_CONSTRUCTS:
                contrast[f"{level}_{construct}_behavior_score"] = scores[
                    (level, construct)
                ]
                contrast[f"{level}_{construct}_correct_rate"] = correct[
                    (level, construct)
                ]
                contrast[f"n_{level}_{construct}"] = len(
                    cells.get((level, construct), ())
                )
            contrast[f"{level}_state_evolution_penalty_vs_static"] = (
                _optional_difference(
                    scores[(level, "static")],
                    scores[(level, "evolution")],
                )
            )
            contrast[f"{level}_hierarchical_conflict_penalty_vs_static"] = (
                _optional_difference(
                    scores[(level, "static")],
                    scores[(level, "hierarchical_conflict")],
                )
            )
        contrast["state_evolution_penalty_amplification_long_vs_short"] = (
            _optional_difference(
                _number(contrast, "long_state_evolution_penalty_vs_static"),
                _number(contrast, "short_state_evolution_penalty_vs_static"),
            )
        )
        contrast[
            "hierarchical_conflict_penalty_amplification_long_vs_short"
        ] = _optional_difference(
            _number(
                contrast,
                "long_hierarchical_conflict_penalty_vs_static",
            ),
            _number(
                contrast,
                "short_hierarchical_conflict_penalty_vs_static",
            ),
        )
        base_rows.append(contrast)

    workspace_rows: dict[tuple[str, str, str], Mapping[str, object]] = {}
    ambiguous_workspace_keys: set[tuple[str, str, str]] = set()
    for base in base_rows:
        if base["condition"] != "workspace_only" or base["complete"] is not True:
            continue
        workspace_key = (
            str(base["policy_profile_id"]),
            str(base["horizon_panel_id"]),
            str(base["opportunity_id"]),
        )
        if workspace_key in workspace_rows:
            ambiguous_workspace_keys.add(workspace_key)
        else:
            workspace_rows[workspace_key] = base

    output: list[dict[str, object]] = []
    for base in base_rows:
        workspace_key = (
            str(base["policy_profile_id"]),
            str(base["horizon_panel_id"]),
            str(base["opportunity_id"]),
        )
        workspace = (
            None
            if workspace_key in ambiguous_workspace_keys
            else workspace_rows.get(workspace_key)
        )
        adjusted: dict[str, object] = {
            **base,
            "workspace_matched_control_available": workspace is not None,
        }
        for level in HORIZON_LEVELS:
            for construct in HORIZON_CONSTRUCTS:
                metric = f"{level}_{construct}_behavior_score"
                adjusted[f"{level}_{construct}_gain_beyond_workspace"] = (
                    _optional_difference(
                        _number(base, metric),
                        _number(workspace, metric),
                    )
                )
        for construct in HORIZON_CONSTRUCTS:
            adjusted[f"{construct}_memory_value_change_long_vs_short"] = (
                _optional_difference(
                    _number(
                        adjusted,
                        f"long_{construct}_gain_beyond_workspace",
                    ),
                    _number(
                        adjusted,
                        f"short_{construct}_gain_beyond_workspace",
                    ),
                )
            )
        for construct, label in (
            ("state_evolution", "state_evolution"),
            ("hierarchical_conflict", "hierarchical_conflict"),
        ):
            for level in HORIZON_LEVELS:
                penalty = f"{level}_{construct}_penalty_vs_static"
                adjusted[f"{level}_{construct}_penalty_excess_over_workspace"] = (
                    _optional_difference(
                        _number(base, penalty),
                        _number(workspace, penalty),
                    )
                )
            adjusted[
                f"{label}_horizon_amplification_excess_over_workspace"
            ] = _optional_difference(
                _number(
                    adjusted,
                    f"long_{construct}_penalty_excess_over_workspace",
                ),
                _number(
                    adjusted,
                    f"short_{construct}_penalty_excess_over_workspace",
                ),
            )
        output.append(adjusted)
    return tuple(output)


def horizon_panel_scorecard(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Aggregate complete panel effects without treating grid members as n."""

    contrasts = compute_horizon_panel_contrasts(observations)
    grouped: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in contrasts:
        key = (
            str(row["policy_profile_id"]),
            str(row["condition"]),
            str(row["readout"]),
        )
        totals[key] += 1
        if row["complete"] is True:
            grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(totals):
        rows = grouped.get(key, [])
        output.append(
            {
                "schema_version": HORIZON_PANEL_SCHEMA_VERSION,
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "analysis_unit": "horizon_panel",
                "n_horizon_panels": totals[key],
                "n_complete_horizon_panels": len(rows),
                **{
                    f"mean_{estimand}": _mean_mapping(rows, estimand)
                    for estimand in (
                        *HORIZON_PRIMARY_ESTIMANDS,
                        *HORIZON_SECONDARY_ESTIMANDS,
                    )
                },
            }
        )
    return tuple(output)


def _mean_behavior(rows: Sequence[MultisystemMetricInput]) -> float | None:
    return (
        None
        if not rows
        else statistics.fmean(row.behavior_score for row in rows)
    )


def _mean_correct(rows: Sequence[MultisystemMetricInput]) -> float | None:
    return (
        None
        if not rows
        else statistics.fmean(float(row.is_correct) for row in rows)
    )


def _mean_mapping(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> float | None:
    values = [value for row in rows if (value := _number(row, key)) is not None]
    return None if not values else statistics.fmean(values)


def _number(
    row: Mapping[str, object] | None,
    key: str,
) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return left - right


__all__ = [
    "HORIZON_CONSTRUCTS",
    "HORIZON_LEVELS",
    "HORIZON_PANEL_SCHEMA_VERSION",
    "HORIZON_PRIMARY_ESTIMANDS",
    "HORIZON_SECONDARY_ESTIMANDS",
    "compute_horizon_panel_contrasts",
    "horizon_panel_scorecard",
]
