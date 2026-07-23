"""Same-decision evidence that end-task outcomes can hide different failures.

The per-decision attribution funnel already localizes the earliest supported
memory failure.  This module adds the complementary *identification* result:
two memory systems can take the same action at the same continuation while the
observed memory-to-action chain breaks at different stages.  Comparisons are
therefore paired only within one policy, readout, episode, SCEU, opportunity,
and checkpoint.  They are descriptive repeated conditions, not independent
statistical samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations

FAULT_PROFILE_DIVERGENCE_SCHEMA_VERSION = 1

_COMPARABLE_STAGES = frozenset(
    {
        "storage_failure",
        "retrieval_failure",
        "exposure_failure",
        "utilization_failure",
        "behavior_success_causal",
        "behavior_success_without_detected_unique_causal_effect",
        # Legacy completed-report label.
        "behavior_success_without_detected_use",
        "behavior_success_unprobed",
    }
)
_UTILIZATION_DIAGNOSES = frozenset(
    {
        "visible_use_evidence_incomplete",
        "visible_without_detected_unique_causal_effect",
        # Legacy completed-report label.
        "visible_without_detected_use",
        "visible_causally_influential_but_wrong",
    }
)


class FaultProfileAlignmentError(ValueError):
    """Raised when purported same-decision rows do not share evaluator truth."""


def compute_fault_profile_divergence(
    decision_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare attributable memory conditions at exactly the same decision.

    The strict outcome-equivalence estimand requires the same selected action
    and correctness label.  A profile still counts as different when the broad
    stage is the same utilization failure but intervention evidence separates
    no detected unique causal effect from causally influential misuse.
    """

    grouped: dict[
        tuple[str, str, str, str, str, int],
        dict[str, Mapping[str, object]],
    ] = defaultdict(dict)
    for row in decision_rows:
        stage = str(row.get("stage", ""))
        condition = str(row.get("condition", ""))
        readout = str(row.get("readout", ""))
        if stage not in _COMPARABLE_STAGES or not condition or readout == "none":
            continue
        key = _decision_key(row)
        if not all(str(value) for value in key[:5]):
            continue
        if condition in grouped[key]:
            raise FaultProfileAlignmentError(
                "duplicate memory condition at aligned decision: "
                f"{key}|{condition}"
            )
        grouped[key][condition] = row

    comparisons: list[dict[str, object]] = []
    for key in sorted(grouped):
        by_condition = grouped[key]
        for condition_a, condition_b in combinations(sorted(by_condition), 2):
            left = by_condition[condition_a]
            right = by_condition[condition_b]
            _require_evaluator_alignment(key, left, right)
            action_a = str(left.get("selected_action_id", ""))
            action_b = str(right.get("selected_action_id", ""))
            correct_a = left.get("behavior_correct") is True
            correct_b = right.get("behavior_correct") is True
            stage_a = str(left.get("stage", ""))
            stage_b = str(right.get("stage", ""))
            label_a = _diagnostic_label(left)
            label_b = _diagnostic_label(right)
            same_action = bool(action_a) and action_a == action_b
            same_correctness = correct_a == correct_b
            outcomes_equivalent = same_action and same_correctness
            same_incorrect_action = same_action and not correct_a and not correct_b
            comparisons.append(
                {
                    "policy_profile_id": key[0],
                    "readout": key[1],
                    "episode_id": key[2],
                    "sceu_id": key[3],
                    "opportunity_id": key[4],
                    "checkpoint_session": key[5],
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "result_id_a": str(left.get("result_id", "")),
                    "result_id_b": str(right.get("result_id", "")),
                    "selected_action_a": action_a,
                    "selected_action_b": action_b,
                    "behavior_correct_a": correct_a,
                    "behavior_correct_b": correct_b,
                    "stage_a": stage_a,
                    "stage_b": stage_b,
                    "diagnostic_label_a": label_a,
                    "diagnostic_label_b": label_b,
                    "same_correctness": same_correctness,
                    "same_selected_action": same_action,
                    "outcome_equivalent": outcomes_equivalent,
                    "same_incorrect_action": same_incorrect_action,
                    "earliest_stage_diverged": stage_a != stage_b,
                    "fault_profile_diverged": label_a != label_b,
                }
            )

    comparisons.sort(key=_comparison_sort_key)
    outcome_equivalent_rows = tuple(
        row for row in comparisons if row["outcome_equivalent"] is True
    )
    same_incorrect = tuple(
        row for row in comparisons if row["same_incorrect_action"] is True
    )
    return {
        "schema_version": FAULT_PROFILE_DIVERGENCE_SCHEMA_VERSION,
        "analysis_role": "descriptive_contribution_diagnostic",
        "analysis_unit": "aligned_decision_condition_pair",
        "comparison_scope": (
            "same_policy_readout_episode_sceu_opportunity_checkpoint"
        ),
        "n_aligned_decision_pairs": len(comparisons),
        "n_condition_pairs": len(
            {
                (str(row["condition_a"]), str(row["condition_b"]))
                for row in comparisons
            }
        ),
        "n_same_correctness_pairs": sum(
            row["same_correctness"] is True for row in comparisons
        ),
        "n_outcome_equivalent_pairs": len(outcome_equivalent_rows),
        "n_outcome_equivalent_fault_profile_divergences": sum(
            row["fault_profile_diverged"] is True
            for row in outcome_equivalent_rows
        ),
        "outcome_equivalent_fault_profile_divergence_rate": _ratio(
            sum(
                row["fault_profile_diverged"] is True
                for row in outcome_equivalent_rows
            ),
            len(outcome_equivalent_rows),
        ),
        "n_same_incorrect_action_pairs": len(same_incorrect),
        "n_same_incorrect_action_fault_profile_divergences": sum(
            row["fault_profile_diverged"] is True for row in same_incorrect
        ),
        "same_incorrect_action_fault_profile_divergence_rate": _ratio(
            sum(row["fault_profile_diverged"] is True for row in same_incorrect),
            len(same_incorrect),
        ),
        "n_all_fault_profile_divergences": sum(
            row["fault_profile_diverged"] is True for row in comparisons
        ),
        "all_pair_fault_profile_divergence_rate": _ratio(
            sum(row["fault_profile_diverged"] is True for row in comparisons),
            len(comparisons),
        ),
        "scorecard": list(_scorecard(comparisons)),
        "comparisons": comparisons,
        "interpretation": (
            "A positive outcome-equivalent divergence shows that identical "
            "checked behavior can conceal different observed memory failure "
            "profiles. A zero estimate is valid. Pair rows are dependent "
            "within decisions and are descriptive rather than inferential units."
        ),
    }


def fault_profile_divergence_markdown(payload: Mapping[str, object]) -> str:
    """Render the diagnostic in a compact report-facing form."""

    lines = [
        "# Outcome-equivalent fault-profile divergence",
        "",
        (
            "This diagnostic compares memory conditions only at the identical "
            "policy, readout, episode, SCEU, opportunity, and checkpoint. It asks "
            "whether the same selected action can conceal a different earliest "
            "supported memory failure profile."
        ),
        "",
        f"Aligned decision pairs: **{payload.get('n_aligned_decision_pairs', 0)}**.",
        (
            "Outcome-equivalent pairs: "
            f"**{payload.get('n_outcome_equivalent_pairs', 0)}**."
        ),
        (
            "Outcome-equivalent fault-profile divergence: "
            f"**{_format_rate(payload.get('outcome_equivalent_fault_profile_divergence_rate'))}**."
        ),
        "",
        "| Policy | Readout | Condition pair | Aligned | Same action | Profile divergence |",
        "|---|---|---|---:|---:|---:|",
    ]
    raw_scorecard = payload.get("scorecard")
    if isinstance(raw_scorecard, Sequence) and not isinstance(
        raw_scorecard, str | bytes
    ):
        for raw in raw_scorecard:
            if not isinstance(raw, Mapping):
                continue
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} |".format(
                    raw.get("policy_profile_id", ""),
                    raw.get("readout", ""),
                    raw.get("condition_pair", ""),
                    raw.get("n_aligned_decision_pairs", 0),
                    raw.get("n_outcome_equivalent_pairs", 0),
                    _format_rate(
                        raw.get(
                            "outcome_equivalent_fault_profile_divergence_rate"
                        )
                    ),
                )
            )
    lines.extend(
        [
            "",
            str(payload.get("interpretation", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _decision_key(
    row: Mapping[str, object],
) -> tuple[str, str, str, str, str, int]:
    checkpoint = row.get("checkpoint_session")
    return (
        str(row.get("policy_profile_id", "")),
        str(row.get("readout", "")),
        str(row.get("episode_id", "")),
        str(row.get("sceu_id", "")),
        str(row.get("opportunity_id", "")),
        checkpoint
        if isinstance(checkpoint, int) and not isinstance(checkpoint, bool)
        else -1,
    )


def _require_evaluator_alignment(
    key: tuple[str, str, str, str, str, int],
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> None:
    left_required = _string_tuple(left.get("required_state_ids"))
    right_required = _string_tuple(right.get("required_state_ids"))
    if not left_required or left_required != right_required:
        raise FaultProfileAlignmentError(
            "same-decision fault-profile rows have different required state: "
            f"{key}"
        )
    left_signature = str(left.get("current_state_signature", ""))
    right_signature = str(right.get("current_state_signature", ""))
    if left_signature and right_signature and left_signature != right_signature:
        raise FaultProfileAlignmentError(
            "same-decision fault-profile rows have different current state: "
            f"{key}"
        )


def _diagnostic_label(row: Mapping[str, object]) -> str:
    stage = str(row.get("stage", ""))
    if stage != "utilization_failure":
        return stage
    diagnosis = str(row.get("decision_layer_diagnosis", ""))
    if diagnosis not in _UTILIZATION_DIAGNOSES:
        diagnosis = "visible_use_evidence_incomplete"
    return f"{stage}:{diagnosis}"


def _scorecard(
    comparisons: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in comparisons:
        grouped[
            (
                str(row.get("policy_profile_id", "")),
                str(row.get("readout", "")),
                "|".join(
                    sorted(
                        (
                            str(row.get("condition_a", "")),
                            str(row.get("condition_b", "")),
                        )
                    )
                ),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        outcome_equivalent = tuple(
            row for row in rows if row.get("outcome_equivalent") is True
        )
        same_incorrect = tuple(
            row for row in rows if row.get("same_incorrect_action") is True
        )
        output.append(
            {
                "policy_profile_id": key[0],
                "readout": key[1],
                "condition_pair": key[2],
                "n_aligned_decision_pairs": len(rows),
                "n_outcome_equivalent_pairs": len(outcome_equivalent),
                "n_outcome_equivalent_fault_profile_divergences": sum(
                    row.get("fault_profile_diverged") is True
                    for row in outcome_equivalent
                ),
                "outcome_equivalent_fault_profile_divergence_rate": _ratio(
                    sum(
                        row.get("fault_profile_diverged") is True
                        for row in outcome_equivalent
                    ),
                    len(outcome_equivalent),
                ),
                "n_same_incorrect_action_pairs": len(same_incorrect),
                "same_incorrect_action_fault_profile_divergence_rate": _ratio(
                    sum(
                        row.get("fault_profile_diverged") is True
                        for row in same_incorrect
                    ),
                    len(same_incorrect),
                ),
            }
        )
    return tuple(output)


def _comparison_sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(row.get(field, ""))
        for field in (
            "policy_profile_id",
            "readout",
            "episode_id",
            "sceu_id",
            "opportunity_id",
            "checkpoint_session",
            "condition_a",
            "condition_b",
        )
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(sorted(str(item) for item in value if str(item)))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _format_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "—"
    return f"{float(value):.4f}"


__all__ = [
    "FAULT_PROFILE_DIVERGENCE_SCHEMA_VERSION",
    "FaultProfileAlignmentError",
    "compute_fault_profile_divergence",
    "fault_profile_divergence_markdown",
]
