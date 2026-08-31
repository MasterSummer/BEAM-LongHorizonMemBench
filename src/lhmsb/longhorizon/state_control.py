"""Known-fault state-control epochs with causal repair verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from lhmsb.longhorizon.drift import TraceableDriftDiagnosis
from lhmsb.longhorizon.interventions import (
    ContinuationOutcome,
    compare_outcome_quality,
)
from lhmsb.longhorizon.state_channels import (
    ChannelArm,
    ChannelCondition,
    audit_core_channel_matrix,
)

FaultChannel = Literal["workspace", "memory"]
KnownFaultKind = Literal[
    "stale_replacement",
    "current_evidence_deletion",
    "version_collision",
    "scope_authority_swap",
]
EpochLabel = Literal[
    "unstable_baseline",
    "unstable_fault",
    "unstable_repair",
    "baseline_not_controlled",
    "fault_not_effective",
    "fault_effect_identified_repair_failed",
    "recovered",
    "recovered_and_retained",
    "recurrence_after_repair",
]


@dataclass(frozen=True)
class ChannelSnapshot:
    """Evaluator-side state of both evidence channels at one control attempt."""

    workspace_condition: ChannelCondition
    memory_condition: ChannelCondition
    workspace_state_ids: tuple[str, ...]
    memory_state_ids: tuple[str, ...]
    workspace_digest: str
    memory_digest: str

    def __post_init__(self) -> None:
        if not self.workspace_digest or not self.memory_digest:
            raise ValueError("channel digests must be non-empty")
        for channel, condition, state_ids in (
            ("workspace", self.workspace_condition, self.workspace_state_ids),
            ("memory", self.memory_condition, self.memory_state_ids),
        ):
            if condition == "absent" and state_ids:
                raise ValueError(f"absent {channel} channel cannot expose state IDs")
            if condition != "absent" and not state_ids:
                raise ValueError(f"non-absent {channel} channel requires state IDs")

    def condition(self, channel: FaultChannel) -> ChannelCondition:
        return (
            self.workspace_condition
            if channel == "workspace"
            else self.memory_condition
        )

    def digest(self, channel: FaultChannel) -> str:
        return self.workspace_digest if channel == "workspace" else self.memory_digest

    def other_digest(self, channel: FaultChannel) -> str:
        return self.memory_digest if channel == "workspace" else self.workspace_digest


@dataclass(frozen=True)
class KnownFaultInjection:
    """One isolated, reversible evidence-channel fault."""

    fault_id: str
    target_channel: FaultChannel
    kind: KnownFaultKind
    target_state_ids: tuple[str, ...]
    before_condition: ChannelCondition
    after_condition: ChannelCondition
    injection_digest: str

    def __post_init__(self) -> None:
        if not self.fault_id or not self.injection_digest:
            raise ValueError("fault ID and injection digest must be non-empty")
        if not self.target_state_ids:
            raise ValueError("known fault must target at least one state")
        if self.before_condition != "current":
            raise ValueError("known-fault epochs must start from current evidence")
        if self.after_condition not in {"stale", "absent"}:
            raise ValueError("known fault must make evidence stale or absent")


@dataclass(frozen=True)
class ControlAttempt:
    """One policy decision and checker outcome inside a state-control epoch."""

    attempt_id: str
    channels: ChannelSnapshot
    outcome: ContinuationOutcome
    declared_fault_channel: FaultChannel | None = None
    drift_diagnosis: TraceableDriftDiagnosis | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("attempt_id must be non-empty")

    @property
    def signature(self) -> tuple[object, ...]:
        diagnosis_flags = (
            ()
            if self.drift_diagnosis is None
            else self.drift_diagnosis.result.flags
        )
        return (*self.outcome.signature, diagnosis_flags)


@dataclass(frozen=True)
class StateControlEpoch:
    """Repeated baseline → isolated fault → repair → retention protocol."""

    epoch_id: str
    injection: KnownFaultInjection
    baseline: tuple[ControlAttempt, ControlAttempt]
    faulted: tuple[ControlAttempt, ControlAttempt]
    repaired: tuple[ControlAttempt, ControlAttempt]
    repair_rounds_to_candidate: int
    retention: tuple[ControlAttempt, ControlAttempt] | tuple[()] = ()

    def __post_init__(self) -> None:
        if not self.epoch_id:
            raise ValueError("epoch_id must be non-empty")
        if self.repair_rounds_to_candidate < 1:
            raise ValueError("repair_rounds_to_candidate must be >= 1")
        if len(self.retention) not in {0, 2}:
            raise ValueError("retention must be absent or a repeated pair")
        baseline_channels = _same_channels(self.baseline, "baseline")
        faulted_channels = _same_channels(self.faulted, "faulted")
        repaired_channels = _same_channels(self.repaired, "repaired")
        target = self.injection.target_channel
        if baseline_channels.condition(target) != self.injection.before_condition:
            raise ValueError("baseline target channel does not match fault contract")
        if faulted_channels.condition(target) != self.injection.after_condition:
            raise ValueError("faulted target channel does not match fault contract")
        if repaired_channels.condition(target) != self.injection.before_condition:
            raise ValueError("repair did not restore current channel semantics")
        if baseline_channels.other_digest(target) != faulted_channels.other_digest(target):
            raise ValueError("known-fault injection changed the non-target channel")
        if faulted_channels.other_digest(target) != repaired_channels.other_digest(target):
            raise ValueError("repair changed the non-target channel")
        if baseline_channels.digest(target) == faulted_channels.digest(target):
            raise ValueError("fault injection did not mutate the target channel")
        if faulted_channels.digest(target) == repaired_channels.digest(target):
            raise ValueError("repair did not mutate the faulted target channel")
        if self.retention:
            retention_channels = _same_channels(self.retention, "retention")
            if retention_channels != repaired_channels:
                raise ValueError("retention probe must hold repaired channels fixed")


@dataclass(frozen=True)
class StateControlEpochResult:
    """Causal fault, diagnosis, repair, and recurrence result for one epoch."""

    epoch_id: str
    label: EpochLabel
    fault_channel: FaultChannel
    fault_kind: KnownFaultKind
    root_cause: str
    baseline_stable: bool
    fault_stable: bool
    repair_stable: bool
    retention_stable: bool | None
    baseline_controlled: bool
    fault_effective: bool
    causal_fault_identified: bool
    diagnosis_correct: bool | None
    repair_success: bool
    retained: bool | None
    repair_rounds_to_candidate: int
    baseline_score: float
    faulted_score: float
    repaired_score: float
    fault_effect: float
    repair_gain: float
    failure_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class KnownFaultProtocol:
    """Pre-model-call schedule for one isolated fault and repair epoch."""

    protocol_id: str
    block_id: str
    target_channel: FaultChannel
    fault_kind: KnownFaultKind
    baseline_arm_id: str
    fault_arm_id: str
    repair_arm_id: str
    repeated_calls_per_phase: int = 2
    retention_handoffs: int = 2

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.block_id:
            raise ValueError("protocol and block IDs must be non-empty")
        if self.repeated_calls_per_phase != 2:
            raise ValueError("causal stability contract requires two calls per phase")
        if self.retention_handoffs < 0:
            raise ValueError("retention_handoffs must be non-negative")

    @property
    def planned_policy_calls(self) -> int:
        phases = 4 if self.retention_handoffs > 0 else 3
        return phases * self.repeated_calls_per_phase

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "planned_policy_calls": self.planned_policy_calls}


def plan_core_known_fault_protocols(
    arms: tuple[ChannelArm, ...],
    *,
    retention_handoffs: int = 2,
) -> tuple[KnownFaultProtocol, ...]:
    """Plan stale/deletion faults in each channel from one matched 3×3 block."""
    audit = audit_core_channel_matrix(arms)
    if not audit.ok:
        raise ValueError(f"known-fault protocols require a core matrix: {audit.errors}")
    by_condition = {
        (arm.workspace_condition, arm.memory_condition): arm
        for arm in arms
    }
    baseline = by_condition[("current", "current")]
    fault_definitions: tuple[
        tuple[FaultChannel, KnownFaultKind, tuple[ChannelCondition, ChannelCondition]],
        ...,
    ] = (
        ("workspace", "stale_replacement", ("stale", "current")),
        ("workspace", "current_evidence_deletion", ("absent", "current")),
        ("memory", "stale_replacement", ("current", "stale")),
        ("memory", "current_evidence_deletion", ("current", "absent")),
    )
    return tuple(
        KnownFaultProtocol(
            protocol_id=(
                f"{baseline.block_id}__fault-{target_channel}-{fault_kind}"
            ),
            block_id=baseline.block_id,
            target_channel=target_channel,
            fault_kind=fault_kind,
            baseline_arm_id=baseline.arm_id,
            fault_arm_id=by_condition[fault_conditions].arm_id,
            repair_arm_id=baseline.arm_id,
            retention_handoffs=retention_handoffs,
        )
        for target_channel, fault_kind, fault_conditions in fault_definitions
    )


def analyze_state_control_epoch(epoch: StateControlEpoch) -> StateControlEpochResult:
    """Analyze one isolated known fault without inferring model internals."""
    baseline_stable = _attempt_pair_stable(epoch.baseline)
    fault_stable = _attempt_pair_stable(epoch.faulted)
    repair_stable = _attempt_pair_stable(epoch.repaired)
    retention_stable = (
        None if not epoch.retention else _attempt_pair_stable(epoch.retention)
    )
    baseline_outcome = epoch.baseline[0].outcome
    faulted_outcome = epoch.faulted[0].outcome
    repaired_outcome = epoch.repaired[0].outcome
    baseline_controlled = baseline_stable and baseline_outcome.is_correct
    fault_effective = (
        baseline_stable
        and fault_stable
        and compare_outcome_quality(baseline_outcome, faulted_outcome) > 0
    )
    causal_fault_identified = baseline_controlled and fault_effective
    repair_success = (
        causal_fault_identified
        and repair_stable
        and repaired_outcome.is_correct
        and compare_outcome_quality(repaired_outcome, faulted_outcome) > 0
    )
    declared = [
        attempt.declared_fault_channel
        for attempt in epoch.faulted
        if attempt.declared_fault_channel is not None
    ]
    diagnosis_correct = (
        None
        if not declared
        else len(declared) == 2
        and all(channel == epoch.injection.target_channel for channel in declared)
    )
    retained = (
        None
        if not epoch.retention or retention_stable is not True
        else epoch.retention[0].outcome.is_correct
    )

    if not baseline_stable:
        label: EpochLabel = "unstable_baseline"
    elif not baseline_controlled:
        label = "baseline_not_controlled"
    elif not fault_stable:
        label = "unstable_fault"
    elif not fault_effective:
        label = "fault_not_effective"
    elif not repair_stable:
        label = "unstable_repair"
    elif not repair_success:
        label = "fault_effect_identified_repair_failed"
    elif retained is False:
        label = "recurrence_after_repair"
    elif retained is True:
        label = "recovered_and_retained"
    else:
        label = "recovered"

    failure_modes = tuple(
        sorted(
            {
                flag
                for attempt in epoch.faulted
                if attempt.drift_diagnosis is not None
                for flag in attempt.drift_diagnosis.result.flags
            }
        )
    )
    return StateControlEpochResult(
        epoch_id=epoch.epoch_id,
        label=label,
        fault_channel=epoch.injection.target_channel,
        fault_kind=epoch.injection.kind,
        root_cause=(
            f"{epoch.injection.target_channel}:{epoch.injection.kind}"
            if causal_fault_identified
            else "not_identified"
        ),
        baseline_stable=baseline_stable,
        fault_stable=fault_stable,
        repair_stable=repair_stable,
        retention_stable=retention_stable,
        baseline_controlled=baseline_controlled,
        fault_effective=fault_effective,
        causal_fault_identified=causal_fault_identified,
        diagnosis_correct=diagnosis_correct,
        repair_success=repair_success,
        retained=retained,
        repair_rounds_to_candidate=epoch.repair_rounds_to_candidate,
        baseline_score=baseline_outcome.behavior_score,
        faulted_score=faulted_outcome.behavior_score,
        repaired_score=repaired_outcome.behavior_score,
        fault_effect=round(
            baseline_outcome.behavior_score - faulted_outcome.behavior_score,
            6,
        ),
        repair_gain=round(
            repaired_outcome.behavior_score - faulted_outcome.behavior_score,
            6,
        ),
        failure_modes=failure_modes,
    )


def _same_channels(
    attempts: tuple[ControlAttempt, ControlAttempt],
    phase: str,
) -> ChannelSnapshot:
    if attempts[0].channels != attempts[1].channels:
        raise ValueError(f"{phase} repeated pair changed channel state")
    return attempts[0].channels


def _attempt_pair_stable(
    attempts: tuple[ControlAttempt, ControlAttempt],
) -> bool:
    return attempts[0].signature == attempts[1].signature


__all__ = [
    "ChannelSnapshot",
    "ControlAttempt",
    "EpochLabel",
    "FaultChannel",
    "KnownFaultInjection",
    "KnownFaultKind",
    "KnownFaultProtocol",
    "StateControlEpoch",
    "StateControlEpochResult",
    "analyze_state_control_epoch",
    "plan_core_known_fault_protocols",
]
