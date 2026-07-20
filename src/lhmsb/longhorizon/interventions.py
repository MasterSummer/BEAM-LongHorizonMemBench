"""Pure causal-use classification over persisted continuation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InterventionKind = Literal[
    "leave_one_out",
    "neutral_replacement",
    "sham_replacement",
    "stale_replacement",
    "count_add",
]
MemoryRole = Literal[
    "supports_current_state",
    "contradicts_current_state",
    "unknown",
]
EffectDirection = Literal["beneficial", "harmful", "neutral", "ambiguous"]
CausalUseLabel = Literal[
    "beneficial",
    "harmful",
    "visible_not_causally_used",
    "causal_direction_ambiguous",
    "unstable_baseline",
    "intervention_unstable",
]


@dataclass(frozen=True)
class ContinuationOutcome:
    """Persisted action and checker result for one continuation call."""

    action_id: str
    behavior_score: float
    is_correct: bool
    violated_state_ids: tuple[str, ...] = ()
    drift_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not 0.0 <= self.behavior_score <= 1.0:
            raise ValueError("behavior_score must be in [0, 1]")

    @property
    def checker_signature(self) -> tuple[object, ...]:
        """Canonical checker-only identity, excluding the selected action."""
        return (
            self.behavior_score,
            self.is_correct,
            tuple(sorted(self.violated_state_ids)),
            tuple(sorted(self.drift_flags)),
        )

    @property
    def signature(self) -> tuple[object, ...]:
        """Canonical action plus checker identity used for repeat stability."""
        return (self.action_id, *self.checker_signature)


@dataclass(frozen=True)
class CausalUseResult:
    """Stable counterfactual classification for one visible memory object."""

    memory_id: str
    intervention_kind: InterventionKind
    memory_role: MemoryRole
    label: CausalUseLabel
    effect_direction: EffectDirection
    behaviorally_used: bool
    baseline_stable: bool
    intervention_stable: bool
    action_changed: bool
    checker_changed: bool


def classify_causal_use(
    *,
    memory_id: str,
    intervention_kind: InterventionKind,
    memory_role: MemoryRole,
    baseline: tuple[ContinuationOutcome, ContinuationOutcome],
    intervention: tuple[ContinuationOutcome, ContinuationOutcome],
) -> CausalUseResult:
    """Classify use only when both repeated pairs are internally stable."""
    if not memory_id:
        raise ValueError("memory_id must be non-empty")
    if intervention_kind not in {
        "leave_one_out",
        "neutral_replacement",
        "sham_replacement",
        "stale_replacement",
        "count_add",
    }:
        raise ValueError(f"unknown intervention kind: {intervention_kind!r}")
    if memory_role not in {
        "supports_current_state",
        "contradicts_current_state",
        "unknown",
    }:
        raise ValueError(f"unknown memory role: {memory_role!r}")

    baseline_stable = _pair_is_stable(baseline)
    intervention_stable = _pair_is_stable(intervention)
    if not baseline_stable:
        return _unstable_result(
            memory_id,
            intervention_kind,
            memory_role,
            "unstable_baseline",
            baseline_stable=False,
            intervention_stable=intervention_stable,
        )
    if not intervention_stable:
        return _unstable_result(
            memory_id,
            intervention_kind,
            memory_role,
            "intervention_unstable",
            baseline_stable=True,
            intervention_stable=False,
        )

    baseline_outcome = baseline[0]
    intervention_outcome = intervention[0]
    action_changed = baseline_outcome.action_id != intervention_outcome.action_id
    checker_changed = (
        baseline_outcome.checker_signature
        != intervention_outcome.checker_signature
    )
    if not action_changed and not checker_changed:
        return CausalUseResult(
            memory_id=memory_id,
            intervention_kind=intervention_kind,
            memory_role=memory_role,
            label="visible_not_causally_used",
            effect_direction="neutral",
            behaviorally_used=False,
            baseline_stable=True,
            intervention_stable=True,
            action_changed=False,
            checker_changed=False,
        )

    comparison = compare_outcome_quality(baseline_outcome, intervention_outcome)
    if comparison > 0:
        direction: EffectDirection = "beneficial"
    elif comparison < 0:
        direction = "harmful"
    else:
        direction = "ambiguous"
    expected = {
        "supports_current_state": "beneficial",
        "contradicts_current_state": "harmful",
        "unknown": None,
    }[memory_role]
    direction_agrees = expected is not None and direction == expected
    label: CausalUseLabel
    if direction_agrees and direction == "beneficial":
        label = "beneficial"
    elif direction_agrees and direction == "harmful":
        label = "harmful"
    else:
        label = "causal_direction_ambiguous"
    return CausalUseResult(
        memory_id=memory_id,
        intervention_kind=intervention_kind,
        memory_role=memory_role,
        label=label,
        effect_direction=direction,
        behaviorally_used=label in {"beneficial", "harmful"},
        baseline_stable=True,
        intervention_stable=True,
        action_changed=action_changed,
        checker_changed=checker_changed,
    )


def compare_outcome_quality(
    first: ContinuationOutcome,
    second: ContinuationOutcome,
) -> int:
    """Return 1 when first is better, -1 when second is better, else 0."""
    first_key = _quality_key(first)
    second_key = _quality_key(second)
    if first_key > second_key:
        return 1
    if first_key < second_key:
        return -1
    return 0


def _quality_key(outcome: ContinuationOutcome) -> tuple[object, ...]:
    return (
        int(outcome.is_correct),
        outcome.behavior_score,
        -len(set(outcome.violated_state_ids)),
        -len(set(outcome.drift_flags)),
    )


def _pair_is_stable(
    outcomes: tuple[ContinuationOutcome, ContinuationOutcome],
) -> bool:
    if len(outcomes) != 2:
        raise ValueError("a repeated outcome pair must contain exactly two calls")
    return outcomes[0].signature == outcomes[1].signature


def _unstable_result(
    memory_id: str,
    intervention_kind: InterventionKind,
    memory_role: MemoryRole,
    label: Literal["unstable_baseline", "intervention_unstable"],
    *,
    baseline_stable: bool,
    intervention_stable: bool,
) -> CausalUseResult:
    return CausalUseResult(
        memory_id=memory_id,
        intervention_kind=intervention_kind,
        memory_role=memory_role,
        label=label,
        effect_direction="ambiguous",
        behaviorally_used=False,
        baseline_stable=baseline_stable,
        intervention_stable=intervention_stable,
        action_changed=False,
        checker_changed=False,
    )


__all__ = [
    "CausalUseLabel",
    "CausalUseResult",
    "ContinuationOutcome",
    "EffectDirection",
    "InterventionKind",
    "MemoryRole",
    "classify_causal_use",
    "compare_outcome_quality",
]
