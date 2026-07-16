"""Programmatic long-horizon drift components and matched decay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lhmsb.longhorizon.interventions import (
    ContinuationOutcome,
    compare_outcome_quality,
)

MatchedDecayLabel = Literal[
    "behavioral_decay",
    "behavioral_improvement",
    "behavior_changed",
    "no_decay",
]


@dataclass(frozen=True)
class DriftEvidence:
    """Evaluator evidence for the four long-horizon behavioral drift modes."""

    outcome: ContinuationOutcome
    used_state_ids: tuple[str, ...] = ()
    active_constraint_ids: tuple[str, ...] = ()
    current_plan_state_ids: tuple[str, ...] = ()
    stale_state_ids: tuple[str, ...] = ()
    selected_local_state_ids: tuple[str, ...] = ()
    global_state_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LongHorizonDriftResult:
    """Four separately reportable drift indicators."""

    constraint_loss: bool
    plan_deviation: bool
    stale_state: bool
    local_over_global: bool

    @property
    def has_drift(self) -> bool:
        return any(
            (
                self.constraint_loss,
                self.plan_deviation,
                self.stale_state,
                self.local_over_global,
            )
        )

    @property
    def flags(self) -> tuple[str, ...]:
        values = {
            "constraint_loss": self.constraint_loss,
            "plan_deviation": self.plan_deviation,
            "stale_state": self.stale_state,
            "local_over_global": self.local_over_global,
        }
        return tuple(sorted(name for name, present in values.items() if present))


@dataclass(frozen=True)
class MatchedOutcome:
    """One early or late outcome belonging to a matched opportunity group."""

    matched_group: str
    checkpoint_session: int
    outcome: ContinuationOutcome

    def __post_init__(self) -> None:
        if not self.matched_group:
            raise ValueError("matched_group must be non-empty")
        if self.checkpoint_session < 0:
            raise ValueError("checkpoint_session must be non-negative")


@dataclass(frozen=True)
class MatchedDecayResult:
    """Direction of behavior change between matched early and late probes."""

    matched_group: str
    early_session: int
    late_session: int
    label: MatchedDecayLabel
    score_delta: float
    action_changed: bool
    drift_emerged: bool


def classify_long_horizon_drift(
    evidence: DriftEvidence,
) -> LongHorizonDriftResult:
    """Resolve drift from state predicates plus checker-emitted evidence."""
    violated = set(evidence.outcome.violated_state_ids)
    used = set(evidence.used_state_ids)
    flags = set(evidence.outcome.drift_flags)
    constraint_loss = bool(
        violated.intersection(evidence.active_constraint_ids)
    ) or _has_flag(
        flags,
        (
            "constraint-influence-lost",
            "constraint-loss",
            "constraint_loss",
            "constraint-violation",
            "constraint_violation",
        ),
    )
    stale_state = bool(used.intersection(evidence.stale_state_ids)) or _has_flag(
        flags,
        ("stale-state", "stale_state", "stale-fact", "stale_fact"),
    )
    plan_deviation = bool(
        violated.intersection(evidence.current_plan_state_ids)
    ) or _has_flag(
        flags,
        (
            "goal-drift",
            "plan-deviation",
            "plan_deviation",
            "current-plan-overwritten",
        ),
    )
    local_over_global = bool(
        evidence.selected_local_state_ids
        and violated.intersection(evidence.global_state_ids)
    ) or _has_flag(
        flags,
        (
            "local-subgoal-overwrites-global-goal",
            "local-over-global",
            "local_over_global",
            "authority-conflict",
            "scope-overreach",
        ),
    )
    return LongHorizonDriftResult(
        constraint_loss=constraint_loss,
        plan_deviation=plan_deviation,
        stale_state=stale_state,
        local_over_global=local_over_global,
    )


def classify_matched_decay(
    early: MatchedOutcome,
    late: MatchedOutcome,
) -> MatchedDecayResult:
    """Compare matched early/late outcomes without conflating valid updates."""
    if early.matched_group != late.matched_group:
        raise ValueError("matched outcomes must use the same matched_group")
    if early.checkpoint_session >= late.checkpoint_session:
        raise ValueError("early checkpoint must precede the late checkpoint")
    comparison = compare_outcome_quality(late.outcome, early.outcome)
    action_changed = early.outcome.action_id != late.outcome.action_id
    drift_emerged = (
        not early.outcome.drift_flags and bool(late.outcome.drift_flags)
    )
    if comparison < 0:
        label: MatchedDecayLabel = "behavioral_decay"
    elif comparison > 0:
        label = "behavioral_improvement"
    elif action_changed or (
        early.outcome.checker_signature != late.outcome.checker_signature
    ):
        label = "behavior_changed"
    else:
        label = "no_decay"
    return MatchedDecayResult(
        matched_group=early.matched_group,
        early_session=early.checkpoint_session,
        late_session=late.checkpoint_session,
        label=label,
        score_delta=round(
            late.outcome.behavior_score - early.outcome.behavior_score,
            6,
        ),
        action_changed=action_changed,
        drift_emerged=drift_emerged,
    )


def _has_flag(flags: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(
        flag == prefix or flag.startswith(f"{prefix}:")
        for flag in flags
        for prefix in prefixes
    )


__all__ = [
    "DriftEvidence",
    "LongHorizonDriftResult",
    "MatchedDecayLabel",
    "MatchedDecayResult",
    "MatchedOutcome",
    "classify_long_horizon_drift",
    "classify_matched_decay",
]
