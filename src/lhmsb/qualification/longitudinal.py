"""Episode-level longitudinal summaries for long-horizon behavioral drift."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from lhmsb.qualification.metrics import MultisystemMetricInput

_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)
_RECOVERY_CONTROLS = {"fresh_reminder", "valid_update"}
_CATEGORY_ONLY_LINEAGE = "__category_only__"


class DriftLineageAlignmentError(ValueError):
    """A scored SCEU does not identify one focal lineage per drift category."""


def compute_drift_trajectory_report(
    observations: Sequence[MultisystemMetricInput],
) -> dict[str, object]:
    """Compute episode-unit violations and longitudinal behavioral drift.

    A drift-compatible action error is a *violation*.  It becomes an observed
    longitudinal drift event only when an earlier distinct eligible checkpoint
    established adherence to the same state lineage and category.  This
    prevents adherence to one constraint or plan from anchoring a later failure
    of another. SCEUs and state lineages order the within-episode trajectory but
    never become independent statistical units.
    """

    by_episode_cell: dict[
        tuple[str, str, str, str], list[MultisystemMetricInput]
    ] = defaultdict(list)
    for row in observations:
        if not row.episode_id:
            continue
        by_episode_cell[
            (row.episode_id, row.policy_profile_id, row.condition, row.readout)
        ].append(row)

    trajectories: list[dict[str, object]] = []
    for (episode_id, policy, condition, readout), rows in sorted(
        by_episode_cell.items()
    ):
        for category in _DRIFT_CATEGORIES:
            category_rows = tuple(row for row in rows if _eligible(row, category))
            ambiguous = {
                row.result_id: _lineages_for(row, category)
                for row in category_rows
                if len(_lineages_for(row, category)) > 1
            }
            if ambiguous:
                raise DriftLineageAlignmentError(
                    "one SCEU/category must identify exactly one focal state "
                    f"lineage; split ambiguous opportunities instead: {ambiguous}"
                )
            lineages = sorted(
                {
                    lineage
                    for row in category_rows
                    for lineage in _lineages_for(row, category)
                }
            )
            for lineage in lineages:
                eligible = tuple(
                    sorted(
                        (
                            row
                            for row in category_rows
                            if lineage in _lineages_for(row, category)
                        ),
                        key=lambda row: (
                            row.checkpoint_session,
                            row.opportunity_id,
                            row.result_id,
                        ),
                    )
                )
                if eligible:
                    trajectories.append(
                        _lineage_trajectory(
                            episode_id=episode_id,
                            policy=policy,
                            condition=condition,
                            readout=readout,
                            category=category,
                            lineage=lineage,
                            eligible=eligible,
                        )
                    )

    summary = _trajectory_summary(trajectories)
    survival = _survival_rows(trajectories)
    return {
        "schema_version": 4,
        "analysis_unit": "episode",
        "within_episode_unit": "SCEU",
        "trajectory_unit": "state_lineage_within_episode",
        "persistence_unit": "distinct_checkpoint",
        "violation_definition": (
            "A category-eligible action has the corresponding canonical drift flag."
        ),
        "observed_drift_definition": (
            "A violation occurs after adherence was observed at an earlier distinct "
            "eligible checkpoint in the same episode/cell/category/state-lineage."
        ),
        "trajectories": trajectories,
        "summary": summary,
        "survival": survival,
        "notes": [
            "Violation incidence and observed drift are reported separately.",
            "A first-observation error is a violation, not evidence of longitudinal drift.",
            "Adherence to one state lineage cannot anchor failure of another lineage.",
            "Category-only legacy rows remain descriptive and are explicitly labeled.",
            "Observed-drift persistence uses adjacent distinct checkpoints after onset.",
            "Recovery requires a later eligible fresh-reminder or valid-update control.",
            "Survival risk sets begin at an observed adherence checkpoint.",
        ],
    }


def _lineage_trajectory(
    *,
    episode_id: str,
    policy: str,
    condition: str,
    readout: str,
    category: str,
    lineage: str,
    eligible: Sequence[MultisystemMetricInput],
) -> dict[str, object]:
    by_checkpoint: dict[int, list[MultisystemMetricInput]] = defaultdict(list)
    for row in eligible:
        by_checkpoint[row.checkpoint_session].append(row)
    checkpoint_drift = tuple(
        (
            session,
            any(category in row.drift_flags for row in checkpoint_rows),
        )
        for session, checkpoint_rows in sorted(by_checkpoint.items())
    )
    event_sessions = tuple(
        session for session, has_violation in checkpoint_drift if has_violation
    )
    first_violation_session = event_sessions[0] if event_sessions else None
    adherence_sessions = tuple(
        session for session, has_violation in checkpoint_drift if not has_violation
    )
    first_adherence_session = adherence_sessions[0] if adherence_sessions else None
    drift_evaluable = first_adherence_session is not None and any(
        session > first_adherence_session for session, _has_violation in checkpoint_drift
    )
    first_drift_session = next(
        (
            session
            for session, has_violation in checkpoint_drift
            if has_violation
            and first_adherence_session is not None
            and session > first_adherence_session
        ),
        None,
    )

    violation_persistence_denominator = 0
    violation_persistence_numerator = 0
    for index, (_session, has_violation) in enumerate(checkpoint_drift[:-1]):
        if not has_violation:
            continue
        violation_persistence_denominator += 1
        violation_persistence_numerator += checkpoint_drift[index + 1][1]

    drift_persistence_denominator = 0
    drift_persistence_numerator = 0
    if first_drift_session is not None:
        onset_index = next(
            index
            for index, (session, _has_violation) in enumerate(checkpoint_drift)
            if session == first_drift_session
        )
        for index in range(onset_index, len(checkpoint_drift) - 1):
            if not checkpoint_drift[index][1]:
                continue
            drift_persistence_denominator += 1
            drift_persistence_numerator += checkpoint_drift[index + 1][1]

    recovery_evaluable = False
    recovered = False
    recovery_session: int | None = None
    if first_drift_session is not None:
        recovery_sessions = sorted(
            {
                later.checkpoint_session
                for later in eligible
                if later.checkpoint_session > first_drift_session
                and later.control_kind in _RECOVERY_CONTROLS
            }
        )
        if recovery_sessions:
            recovery_evaluable = True
            recovery_session = recovery_sessions[0]
            recovered = not any(
                category in later.drift_flags
                for later in eligible
                if later.checkpoint_session == recovery_session
                and later.control_kind in _RECOVERY_CONTROLS
            )

    evidence_modes = sorted(
        {
            row.drift_lineage_evidence_mode
            for row in eligible
            if row.drift_lineage_evidence_mode
            and row.drift_lineage_evidence_mode != "unavailable"
        }
    )
    evidence_mode = (
        "category_only_legacy"
        if lineage == _CATEGORY_ONLY_LINEAGE
        else evidence_modes[0]
        if len(evidence_modes) == 1
        else "mixed"
        if evidence_modes
        else "unavailable"
    )
    return {
        "episode_id": episode_id,
        "policy_profile_id": policy,
        "condition": condition,
        "readout": readout,
        "drift_category": category,
        "state_lineage_id": lineage,
        "lineage_evidence_mode": evidence_mode,
        "lineage_backed": lineage != _CATEGORY_ONLY_LINEAGE,
        "eligible_opportunity_count": len(eligible),
        "eligible_checkpoint_count": len(checkpoint_drift),
        "violation_opportunity_count": sum(
            category in row.drift_flags for row in eligible
        ),
        "violation_checkpoint_count": len(event_sessions),
        "violation_event_observed": first_violation_session is not None,
        "first_violation_session": first_violation_session,
        "adherence_anchor_observed": first_adherence_session is not None,
        "first_adherence_session": first_adherence_session,
        "drift_evaluable": drift_evaluable,
        "event_observed": first_drift_session is not None,
        "entry_session": checkpoint_drift[0][0],
        "drift_entry_session": first_adherence_session if drift_evaluable else None,
        "first_drift_session": first_drift_session,
        "censor_session": eligible[-1].checkpoint_session,
        "violation_persistence_numerator": violation_persistence_numerator,
        "violation_persistence_denominator": violation_persistence_denominator,
        "persistence_numerator": drift_persistence_numerator,
        "persistence_denominator": drift_persistence_denominator,
        "recovery_evaluable": recovery_evaluable,
        "recovered": recovered,
        "recovery_session": recovery_session,
    }


def drift_trajectory_markdown(payload: Mapping[str, object]) -> str:
    """Render the aggregate episode-level drift trajectory table."""
    lines = [
        "# Long-horizon behavioral-drift trajectories",
        "",
        (
            "Analysis unit: **episode**. SCEUs order decisions within a trajectory "
            "and are not treated as independent samples."
        ),
        "",
        "A drift-compatible error is reported as a violation. Observed drift "
        "requires an earlier adherence checkpoint for the same state lineage; "
        "category-only legacy trajectories are not equivalent evidence.",
        "",
        "| Policy | Condition | Readout | Drift category | Episodes | Violation "
        "incidence | Drift-evaluable | Observed drift incidence | Mean onset | "
        "Persistence | Recovery |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _mapping_sequence(payload.get("summary")):
        lines.append(
            "| {policy} | {condition} | {readout} | {category} | {n} | "
            "{violation} | {evaluable} | {incidence} | {first} | {persistence} | "
            "{recovery} |".format(
                policy=row.get("policy_profile_id", ""),
                condition=row.get("condition", ""),
                readout=row.get("readout", ""),
                category=row.get("drift_category", ""),
                n=row.get("n_episodes", 0),
                violation=_format_rate(row.get("violation_incidence")),
                evaluable=row.get("n_drift_evaluable_episodes", 0),
                incidence=_format_rate(row.get("observed_drift_incidence")),
                first=_format_rate(row.get("mean_first_drift_session")),
                persistence=_format_rate(row.get("persistence_rate")),
                recovery=_format_rate(row.get("recovery_rate")),
            )
        )
    lines.extend(
        [
            "",
            "Kaplan–Meier-style survival points are available in the companion JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def episode_observed_drift_incidence(
    observations: Sequence[MultisystemMetricInput],
) -> float | None:
    """Return the anchored drift incidence for one episode/cell.

    The caller is responsible for passing one episode and one policy/condition/
    readout cell.  Categories without a prior adherence anchor and a later
    checkpoint are excluded rather than scored as zero.
    """

    identities = {
        (row.episode_id, row.policy_profile_id, row.condition, row.readout)
        for row in observations
    }
    if len(identities) > 1:
        raise ValueError(
            "episode_observed_drift_incidence requires one episode/cell"
        )
    payload = compute_drift_trajectory_report(observations)
    trajectories = _mapping_sequence(payload.get("trajectories"))
    evaluable_by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in trajectories:
        if bool(row.get("drift_evaluable")):
            evaluable_by_category[str(row.get("drift_category", ""))].append(row)
    if not evaluable_by_category:
        return None
    return sum(
        any(bool(row.get("event_observed")) for row in rows)
        for rows in evaluable_by_category.values()
    ) / len(evaluable_by_category)


def _trajectory_summary(
    trajectories: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in trajectories:
        grouped[
            (
                str(row["policy_profile_id"]),
                str(row["condition"]),
                str(row["readout"]),
                str(row["drift_category"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        by_episode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            by_episode[str(row["episode_id"])].append(row)
        violation_sessions = [
            min(
                _required_int(row, "first_violation_session")
                for row in episode_rows
                if row.get("first_violation_session") is not None
            )
            for episode_rows in by_episode.values()
            if any(row.get("first_violation_session") is not None for row in episode_rows)
        ]
        evaluable_by_episode = {
            episode_id: tuple(
                row for row in episode_rows if bool(row["drift_evaluable"])
            )
            for episode_id, episode_rows in by_episode.items()
            if any(bool(row["drift_evaluable"]) for row in episode_rows)
        }
        first_sessions = [
            min(
                _required_int(row, "first_drift_session")
                for row in episode_rows
                if row.get("first_drift_session") is not None
            )
            for episode_rows in evaluable_by_episode.values()
            if any(row.get("first_drift_session") is not None for row in episode_rows)
        ]
        violation_persistence_denominator = sum(
            _required_int(row, "violation_persistence_denominator")
            for row in rows
        )
        persistence_denominator = sum(
            _required_int(row, "persistence_denominator") for row in rows
        )
        recovery_by_episode = {
            episode_id: tuple(
                row for row in episode_rows if bool(row["recovery_evaluable"])
            )
            for episode_id, episode_rows in by_episode.items()
            if any(bool(row["recovery_evaluable"]) for row in episode_rows)
        }
        episodes_with_violation = sum(
            any(bool(row["violation_event_observed"]) for row in episode_rows)
            for episode_rows in by_episode.values()
        )
        episodes_with_drift = sum(
            any(bool(row["event_observed"]) for row in episode_rows)
            for episode_rows in evaluable_by_episode.values()
        )
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "drift_category": key[3],
                "n_episodes": len(by_episode),
                "n_state_lineage_trajectories": len(rows),
                "n_lineage_backed_trajectories": sum(
                    bool(row.get("lineage_backed")) for row in rows
                ),
                "episodes_with_violation": episodes_with_violation,
                "violation_incidence": _ratio(
                    episodes_with_violation,
                    len(by_episode),
                ),
                "mean_first_violation_session": (
                    None
                    if not violation_sessions
                    else statistics.fmean(violation_sessions)
                ),
                "n_drift_evaluable_episodes": len(evaluable_by_episode),
                "episodes_with_drift": episodes_with_drift,
                "observed_drift_incidence": _ratio(
                    episodes_with_drift,
                    len(evaluable_by_episode),
                ),
                # Backward-readable alias with the corrected anchored meaning.
                "cumulative_incidence": _ratio(
                    episodes_with_drift,
                    len(evaluable_by_episode),
                ),
                "mean_first_drift_session": (
                    None
                    if not first_sessions
                    else statistics.fmean(first_sessions)
                ),
                "persistence_rate": _ratio(
                    sum(
                        _required_int(row, "persistence_numerator")
                        for row in rows
                    ),
                    persistence_denominator,
                ),
                "persistence_pair_count": persistence_denominator,
                "violation_persistence_rate": _ratio(
                    sum(
                        _required_int(row, "violation_persistence_numerator")
                        for row in rows
                    ),
                    violation_persistence_denominator,
                ),
                "violation_persistence_pair_count": (
                    violation_persistence_denominator
                ),
                "recovery_rate": _ratio(
                    sum(
                        all(bool(row["recovered"]) for row in episode_rows)
                        for episode_rows in recovery_by_episode.values()
                    ),
                    len(recovery_by_episode),
                ),
                "recovery_evaluable_n": len(recovery_by_episode),
            }
        )
    return output


def _survival_rows(
    trajectories: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in trajectories:
        if not bool(row.get("drift_evaluable")):
            continue
        grouped[
            (
                str(row["policy_profile_id"]),
                str(row["condition"]),
                str(row["readout"]),
                str(row["drift_category"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        by_episode: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in grouped[key]:
            by_episode[str(row["episode_id"])].append(row)
        rows: list[dict[str, object]] = []
        for episode_id, episode_lineages in sorted(by_episode.items()):
            event_sessions = [
                _required_int(row, "first_drift_session")
                for row in episode_lineages
                if bool(row.get("event_observed"))
            ]
            rows.append(
                {
                    "episode_id": episode_id,
                    "drift_entry_session": min(
                        _required_int(row, "drift_entry_session")
                        for row in episode_lineages
                    ),
                    "event_observed": bool(event_sessions),
                    "first_drift_session": (
                        min(event_sessions) if event_sessions else None
                    ),
                    "censor_session": max(
                        _required_int(row, "censor_session")
                        for row in episode_lineages
                    ),
                }
            )
        times = sorted(
            {
                _required_int(
                    row,
                    (
                        "first_drift_session"
                        if row.get("first_drift_session") is not None
                        else "censor_session"
                    ),
                )
                for row in rows
            }
        )
        survival = 1.0
        for session in times:
            at_risk = sum(
                _required_int(row, "drift_entry_session") <= session
                <= _observed_or_censor_session(row)
                for row in rows
            )
            events = sum(
                bool(row["event_observed"])
                and _required_int(row, "first_drift_session") == session
                for row in rows
            )
            censored = sum(
                not bool(row["event_observed"])
                and _required_int(row, "censor_session") == session
                for row in rows
            )
            if at_risk:
                survival *= 1.0 - events / at_risk
            output.append(
                {
                    "policy_profile_id": key[0],
                    "condition": key[1],
                    "readout": key[2],
                    "drift_category": key[3],
                    "session": session,
                    "at_risk": at_risk,
                    "events": events,
                    "censored": censored,
                    "survival_probability": survival,
                    "cumulative_incidence": 1.0 - survival,
                }
            )
    return output


def _observed_or_censor_session(row: Mapping[str, object]) -> int:
    return _required_int(
        row,
        (
            "first_drift_session"
            if bool(row.get("event_observed"))
            else "censor_session"
        ),
    )


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise TypeError(f"trajectory field {key!r} must be an integer")
    return value


def _eligible(row: MultisystemMetricInput, category: str) -> bool:
    eligible = row.drift_eligible_categories
    return eligible is None or category in eligible


def _lineages_for(
    row: MultisystemMetricInput,
    category: str,
) -> tuple[str, ...]:
    lineages = tuple(
        lineage
        for declared_category, lineage in row.drift_lineage_pairs
        if declared_category == category and lineage
    )
    return tuple(sorted(set(lineages))) or (_CATEGORY_ONLY_LINEAGE,)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _format_rate(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{float(value):.4f}"


__all__ = [
    "DriftLineageAlignmentError",
    "compute_drift_trajectory_report",
    "drift_trajectory_markdown",
    "episode_observed_drift_incidence",
]
