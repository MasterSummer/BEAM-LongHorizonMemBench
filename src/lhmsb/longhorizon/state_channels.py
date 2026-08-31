"""Controlled workspace × memory challenge axes for MemHorizon."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import EpisodePlan

ChannelCondition = Literal["current", "stale", "absent"]
DecisionFamily = Literal[
    "branch_selection",
    "constraint_compliance",
    "scope_authority",
    "repair_after_failure",
]
CrossChannelRelation = Literal[
    "aligned_current",
    "aligned_stale",
    "cross_channel_conflict",
    "single_current_channel",
    "single_stale_channel",
    "both_absent",
]

CORE_CHANNEL_CONDITIONS: tuple[ChannelCondition, ...] = (
    "current",
    "stale",
    "absent",
)


@dataclass(frozen=True)
class DecisionAnchor:
    """The latent decision held invariant across a matched channel matrix."""

    episode_id: str
    opportunity_id: str
    checkpoint_session: int
    decision_family: DecisionFamily
    authoritative_state_ids: tuple[str, ...]
    stale_state_ids: tuple[str, ...]
    gold_action_ids: tuple[str, ...]
    prompt_template_id: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "opportunity_id", "prompt_template_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.checkpoint_session < 0:
            raise ValueError("checkpoint_session must be non-negative")
        if not self.authoritative_state_ids:
            raise ValueError("authoritative_state_ids must be non-empty")
        if not self.stale_state_ids:
            raise ValueError("stale_state_ids must be non-empty")
        if set(self.authoritative_state_ids).intersection(self.stale_state_ids):
            raise ValueError("current and stale state IDs must be disjoint")
        if not self.gold_action_ids or len(set(self.gold_action_ids)) != len(
            self.gold_action_ids
        ):
            raise ValueError("gold_action_ids must be non-empty and unique")


@dataclass(frozen=True)
class ChallengeAxes:
    """Difficulty controls fixed within one matched 3×3 channel block."""

    horizon_handoffs: int
    state_transition_count: int
    workspace_version_depth: int
    workspace_distractor_count: int
    memory_distractor_count: int
    authority_depth: int
    dependency_depth: int
    scope_switch_count: int = 0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("challenge axes must be non-negative integers")
        if self.state_transition_count < 1:
            raise ValueError("a stale condition requires at least one state transition")
        if self.workspace_version_depth < 2:
            raise ValueError("workspace version depth must expose current/stale separation")
        if self.authority_depth < 1 or self.dependency_depth < 1:
            raise ValueError("authority and dependency depth must be >= 1")


def cross_channel_relation(
    workspace: ChannelCondition,
    memory: ChannelCondition,
) -> CrossChannelRelation:
    """Return the evidence relationship represented by one factorial arm."""
    if workspace == memory == "current":
        return "aligned_current"
    if workspace == memory == "stale":
        return "aligned_stale"
    if workspace == memory == "absent":
        return "both_absent"
    if {workspace, memory} == {"current", "stale"}:
        return "cross_channel_conflict"
    if "current" in {workspace, memory}:
        return "single_current_channel"
    return "single_stale_channel"


@dataclass(frozen=True)
class ChannelArm:
    """One matched workspace × memory evidence intervention."""

    block_id: str
    anchor: DecisionAnchor
    axes: ChallengeAxes
    workspace_condition: ChannelCondition
    memory_condition: ChannelCondition

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if self.workspace_condition not in CORE_CHANNEL_CONDITIONS:
            raise ValueError("unknown workspace condition")
        if self.memory_condition not in CORE_CHANNEL_CONDITIONS:
            raise ValueError("unknown memory condition")

    @property
    def arm_id(self) -> str:
        return (
            f"{self.block_id}__W-{self.workspace_condition}"
            f"__M-{self.memory_condition}"
        )

    @property
    def relation(self) -> CrossChannelRelation:
        return cross_channel_relation(
            self.workspace_condition,
            self.memory_condition,
        )

    @property
    def workspace_state_ids(self) -> tuple[str, ...]:
        """Evaluator-side evidence assigned to the workspace channel."""
        if self.workspace_condition == "current":
            return self.anchor.authoritative_state_ids
        if self.workspace_condition == "stale":
            return self.anchor.stale_state_ids
        return ()

    @property
    def memory_state_ids(self) -> tuple[str, ...]:
        """Evaluator-side evidence assigned to the memory channel."""
        if self.memory_condition == "current":
            return self.anchor.authoritative_state_ids
        if self.memory_condition == "stale":
            return self.anchor.stale_state_ids
        return ()

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "block_id": self.block_id,
            "anchor": asdict(self.anchor),
            "axes": asdict(self.axes),
            "workspace_condition": self.workspace_condition,
            "memory_condition": self.memory_condition,
            "cross_channel_relation": self.relation,
            "workspace_state_ids": list(self.workspace_state_ids),
            "memory_state_ids": list(self.memory_state_ids),
        }


@dataclass(frozen=True)
class ChannelMatrixAudit:
    """Generation-time identifiability checks for one core factorial block."""

    ok: bool
    arm_count: int
    combination_count: int
    anchor_invariant: bool
    axes_invariant: bool
    exact_factorial_product: bool
    unique_arm_ids: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_core_channel_matrix(
    anchor: DecisionAnchor,
    axes: ChallengeAxes,
    *,
    block_id: str,
) -> tuple[ChannelArm, ...]:
    """Build the canonical 3×3 current/stale/absent evidence matrix."""
    arms = tuple(
        ChannelArm(
            block_id=block_id,
            anchor=anchor,
            axes=axes,
            workspace_condition=workspace,
            memory_condition=memory,
        )
        for workspace, memory in product(
            CORE_CHANNEL_CONDITIONS,
            CORE_CHANNEL_CONDITIONS,
        )
    )
    audit = audit_core_channel_matrix(arms)
    if not audit.ok:
        raise ValueError(f"invalid core channel matrix: {audit.errors}")
    return arms


def audit_core_channel_matrix(
    arms: tuple[ChannelArm, ...],
) -> ChannelMatrixAudit:
    """Reject confounded or incomplete channel matrices before model calls."""
    errors: list[str] = []
    anchor_invariant = len({arm.anchor for arm in arms}) == 1
    axes_invariant = len({arm.axes for arm in arms}) == 1
    expected = set(product(CORE_CHANNEL_CONDITIONS, CORE_CHANNEL_CONDITIONS))
    observed = {
        (arm.workspace_condition, arm.memory_condition)
        for arm in arms
    }
    exact_product = observed == expected and len(arms) == len(expected)
    unique_ids = len({arm.arm_id for arm in arms}) == len(arms)
    if not anchor_invariant:
        errors.append("decision-anchor-varies")
    if not axes_invariant:
        errors.append("non-channel-axes-vary")
    if not exact_product:
        errors.append("not-exact-3x3-product")
    if not unique_ids:
        errors.append("duplicate-arm-id")
    return ChannelMatrixAudit(
        ok=not errors,
        arm_count=len(arms),
        combination_count=len(observed),
        anchor_invariant=anchor_invariant,
        axes_invariant=axes_invariant,
        exact_factorial_product=exact_product,
        unique_arm_ids=unique_ids,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class ChannelContrast:
    """A preregisterable linear contrast over matched arm-level outcomes."""

    contrast_id: str
    interpretation: str
    terms: tuple[tuple[str, float], ...]

    def evaluate(self, scores: Mapping[str, float]) -> float:
        missing = sorted(arm_id for arm_id, _ in self.terms if arm_id not in scores)
        if missing:
            raise ValueError(f"missing arm scores for contrast: {missing}")
        return sum(scores[arm_id] * coefficient for arm_id, coefficient in self.terms)


def core_channel_contrasts(
    arms: tuple[ChannelArm, ...],
) -> tuple[ChannelContrast, ...]:
    """Return the primary rescue, stale-harm, conflict, and synergy estimands."""
    audit = audit_core_channel_matrix(arms)
    if not audit.ok:
        raise ValueError(f"contrasts require an exact core matrix: {audit.errors}")
    by_condition = {
        (arm.workspace_condition, arm.memory_condition): arm.arm_id
        for arm in arms
    }

    def arm(workspace: ChannelCondition, memory: ChannelCondition) -> str:
        return by_condition[(workspace, memory)]

    return (
        ChannelContrast(
            "workspace-current-rescue-given-memory-absent",
            "Benefit of current workspace evidence when memory is unavailable.",
            ((arm("current", "absent"), 1.0), (arm("absent", "absent"), -1.0)),
        ),
        ChannelContrast(
            "memory-current-rescue-given-workspace-absent",
            "Benefit of current memory when workspace evidence is unavailable.",
            ((arm("absent", "current"), 1.0), (arm("absent", "absent"), -1.0)),
        ),
        ChannelContrast(
            "workspace-stale-harm-given-memory-absent",
            "Harm from stale workspace evidence relative to no workspace evidence.",
            ((arm("stale", "absent"), 1.0), (arm("absent", "absent"), -1.0)),
        ),
        ChannelContrast(
            "memory-stale-harm-given-workspace-absent",
            "Harm from stale memory relative to no memory evidence.",
            ((arm("absent", "stale"), 1.0), (arm("absent", "absent"), -1.0)),
        ),
        ChannelContrast(
            "cross-channel-current-source-preference",
            "Preference for current workspace versus current memory under conflict.",
            ((arm("current", "stale"), 1.0), (arm("stale", "current"), -1.0)),
        ),
        ChannelContrast(
            "current-channel-synergy",
            "Interaction beyond the sum of workspace-only and memory-only rescue.",
            (
                (arm("current", "current"), 1.0),
                (arm("current", "absent"), -1.0),
                (arm("absent", "current"), -1.0),
                (arm("absent", "absent"), 1.0),
            ),
        ),
        ChannelContrast(
            "workspace-rescue-against-stale-memory",
            "Whether a current workspace corrects a stale memory channel.",
            ((arm("current", "stale"), 1.0), (arm("absent", "stale"), -1.0)),
        ),
        ChannelContrast(
            "memory-rescue-against-stale-workspace",
            "Whether current memory corrects a stale workspace channel.",
            ((arm("stale", "current"), 1.0), (arm("stale", "absent"), -1.0)),
        ),
    )


def derive_core_channel_matrices(
    plan: EpisodePlan,
    *,
    minimum_handoffs: int = 3,
    memory_distractor_count: int = 3,
) -> tuple[tuple[ChannelArm, ...], ...]:
    """Derive flexible matched channel blocks from real versioned decisions.

    Eligible opportunities must reference an invalidated state and a current
    successor that explicitly invalidated it.  This grounds ``stale`` in the
    plan's event graph rather than in hand-written distractor text.  Different
    opportunity requests and decision families become separate matched blocks;
    only W/M channel state changes inside each block.
    """
    if minimum_handoffs < 0 or memory_distractor_count < 0:
        raise ValueError("minimum_handoffs and distractor count must be non-negative")
    opportunity_by_id = {
        opportunity.opportunity_id: opportunity
        for opportunity in plan.opportunities
    }
    state_by_id = {state.state_id: state for state in plan.state_units}
    workspace_by_session = {
        snapshot.checkpoint_session: snapshot
        for snapshot in plan.workspaces
    }
    matrices: list[tuple[ChannelArm, ...]] = []
    for sceu in plan.sceu_units:
        profile = profile_sceu(plan, sceu)
        if (
            profile.handoff_count < minimum_handoffs
            or not profile.invalidated_referenced_state_ids
            or profile.relevant_transition_count < 1
        ):
            continue
        opportunity = opportunity_by_id[sceu.opportunity_id]
        replay = replay_plan(plan, opportunity.checkpoint_session)
        stale_ids = set(profile.invalidated_referenced_state_ids)
        successor_ids = {
            event.target_state_id
            for event in plan.events
            if event.session <= opportunity.checkpoint_session
            and stale_ids.intersection(event.invalidates)
            and event.target_state_id in replay.current
        }
        if not successor_ids:
            stale_kinds = {
                state_by_id[state_id].kind
                for state_id in stale_ids
                if state_id in state_by_id
            }
            stale_end = max(
                (
                    state_by_id[state_id].valid_to or 0
                    for state_id in stale_ids
                    if state_id in state_by_id
                ),
                default=0,
            )
            successor_ids = {
                state_id
                for state_id, state in replay.current.items()
                if state.kind in stale_kinds and state.valid_from >= stale_end
            }
        if not successor_ids:
            continue
        authoritative_ids = tuple(sorted(successor_ids))
        stale_state_ids = tuple(sorted(stale_ids))
        authorities = {
            state_by_id[state_id].authority
            for state_id in profile.current_action_relevant_state_ids
            if state_id in state_by_id
        }
        scope_switch_count = sum(
            event.type == "scope_change"
            for event in plan.events
            if event.session <= opportunity.checkpoint_session
        )
        snapshot = workspace_by_session[opportunity.checkpoint_session]
        axes = ChallengeAxes(
            horizon_handoffs=profile.handoff_count,
            state_transition_count=profile.relevant_transition_count,
            workspace_version_depth=profile.relevant_transition_count + 1,
            workspace_distractor_count=max(0, len(snapshot.artifacts) - 2),
            memory_distractor_count=memory_distractor_count,
            authority_depth=max(1, len(authorities)),
            dependency_depth=max(1, profile.dependency_depth),
            scope_switch_count=scope_switch_count,
        )
        anchor = DecisionAnchor(
            episode_id=plan.episode_id,
            opportunity_id=opportunity.opportunity_id,
            checkpoint_session=opportunity.checkpoint_session,
            decision_family=_decision_family(opportunity.challenge_type),
            authoritative_state_ids=authoritative_ids,
            stale_state_ids=stale_state_ids,
            gold_action_ids=opportunity.valid_action_ids,
            prompt_template_id=opportunity.challenge_type,
        )
        matrices.append(
            build_core_channel_matrix(
                anchor,
                axes,
                block_id=f"channel-{sceu.sceu_id}",
            )
        )
    return tuple(matrices)


def _decision_family(challenge_type: str) -> DecisionFamily:
    if "recovery" in challenge_type:
        return "repair_after_failure"
    if any(token in challenge_type for token in ("scope", "local", "authority")):
        return "scope_authority"
    if any(token in challenge_type for token in ("update", "reminder", "conflict")):
        return "constraint_compliance"
    return "branch_selection"


__all__ = [
    "CORE_CHANNEL_CONDITIONS",
    "ChallengeAxes",
    "ChannelArm",
    "ChannelCondition",
    "ChannelContrast",
    "ChannelMatrixAudit",
    "CrossChannelRelation",
    "DecisionAnchor",
    "DecisionFamily",
    "audit_core_channel_matrix",
    "build_core_channel_matrix",
    "core_channel_contrasts",
    "cross_channel_relation",
    "derive_core_channel_matrices",
]
