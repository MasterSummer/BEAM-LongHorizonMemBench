"""Programmatic long-horizon drift components and matched decay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
DriftMode = Literal[
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
]


@dataclass(frozen=True)
class DriftEvidence:
    """Evaluator evidence for the four long-horizon behavioral drift modes."""

    outcome: ContinuationOutcome
    action_expressed_state_ids: tuple[str, ...] = ()
    # Backward-compatible input for pre-C3 callers. This describes state
    # predicates expressed by an action, not memory-object causal use.
    used_state_ids: tuple[str, ...] = ()
    active_constraint_ids: tuple[str, ...] = ()
    current_plan_state_ids: tuple[str, ...] = ()
    stale_state_ids: tuple[str, ...] = ()
    selected_local_state_ids: tuple[str, ...] = ()
    global_state_ids: tuple[str, ...] = ()
    future_state_ids: tuple[str, ...] = ()


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
class DriftModeEvidence:
    """Auditable predicate and checker evidence for one drift phenotype."""

    mode: DriftMode
    predicate_state_ids: tuple[str, ...]
    checker_flags: tuple[str, ...]

    @property
    def evidence_sources(self) -> tuple[str, ...]:
        sources: list[str] = []
        if self.predicate_state_ids:
            sources.append("direct_state_predicate")
        if self.checker_flags:
            sources.append("checker_flag")
        return tuple(sources)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "evidence_sources": self.evidence_sources}


@dataclass(frozen=True)
class TraceableDriftDiagnosis:
    """Drift result with a machine-checkable explanation for every flag."""

    result: LongHorizonDriftResult
    mode_evidence: tuple[DriftModeEvidence, ...]

    @property
    def traceable(self) -> bool:
        return all(item.evidence_sources for item in self.mode_evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "flags": list(self.result.flags),
            "has_drift": self.result.has_drift,
            "traceable": self.traceable,
            "mode_evidence": [item.to_dict() for item in self.mode_evidence],
        }


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
    return diagnose_long_horizon_drift(evidence).result


def diagnose_long_horizon_drift(
    evidence: DriftEvidence,
) -> TraceableDriftDiagnosis:
    """Resolve drift and retain the exact predicate/flag evidence behind it."""
    violated = set(evidence.outcome.violated_state_ids)
    action_expressed = set(evidence.action_expressed_state_ids).union(
        evidence.used_state_ids
    )
    flags = set(evidence.outcome.drift_flags)
    constraint_states = violated.intersection(evidence.active_constraint_ids)
    constraint_flags = _matching_flags(
        flags,
        (
            "constraint-influence-lost",
            "constraint-loss",
            "constraint_loss",
            "constraint-violation",
            "constraint_violation",
        ),
    )
    stale_states = action_expressed.intersection(evidence.stale_state_ids)
    stale_flags = _matching_flags(
        flags,
        ("stale-state", "stale_state", "stale-fact", "stale_fact"),
    )
    plan_states = violated.intersection(evidence.current_plan_state_ids).union(
        action_expressed.intersection(evidence.future_state_ids)
    )
    plan_flags = _matching_flags(
        flags,
        (
            "goal-drift",
            "plan-deviation",
            "plan_deviation",
            "current-plan-overwritten",
            "future-state-adoption",
            "future_state_adoption",
        ),
    )
    authority_states = (
        violated.intersection(evidence.global_state_ids)
        if evidence.selected_local_state_ids
        else set()
    )
    authority_flags = _matching_flags(
        flags,
        (
            "local-subgoal-overwrites-global-goal",
            "local-over-global",
            "local_over_global",
            "authority-conflict",
            "scope-overreach",
        ),
    )
    result = LongHorizonDriftResult(
        constraint_loss=bool(constraint_states or constraint_flags),
        plan_deviation=bool(plan_states or plan_flags),
        stale_state=bool(stale_states or stale_flags),
        local_over_global=bool(authority_states or authority_flags),
    )
    evidence_by_mode: dict[DriftMode, tuple[set[str], tuple[str, ...]]] = {
        "constraint_loss": (constraint_states, constraint_flags),
        "plan_deviation": (plan_states, plan_flags),
        "stale_state": (stale_states, stale_flags),
        "local_over_global": (authority_states, authority_flags),
    }
    return TraceableDriftDiagnosis(
        result=result,
        mode_evidence=tuple(
            DriftModeEvidence(
                mode=mode,
                predicate_state_ids=tuple(sorted(state_ids)),
                checker_flags=checker_flags,
            )
            for mode, (state_ids, checker_flags) in evidence_by_mode.items()
            if getattr(result, mode)
        ),
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


def _matching_flags(
    flags: set[str],
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            flag
            for flag in flags
            if any(
                flag == prefix or flag.startswith(f"{prefix}:")
                for prefix in prefixes
            )
        )
    )


__all__ = [
    "DriftEvidence",
    "DriftMode",
    "DriftModeEvidence",
    "LongHorizonDriftResult",
    "MatchedDecayLabel",
    "MatchedDecayResult",
    "MatchedOutcome",
    "TraceableDriftDiagnosis",
    "classify_long_horizon_drift",
    "classify_matched_decay",
    "diagnose_long_horizon_drift",
]
