from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.longhorizon.drift import (
    DriftEvidence,
    TraceableDriftDiagnosis,
    diagnose_long_horizon_drift,
)
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.longhorizon.state_channels import (
    ChallengeAxes,
    ChannelCondition,
    DecisionAnchor,
    build_core_channel_matrix,
)
from lhmsb.longhorizon.state_control import (
    ChannelSnapshot,
    ControlAttempt,
    FaultChannel,
    KnownFaultInjection,
    StateControlEpoch,
    analyze_state_control_epoch,
    plan_core_known_fault_protocols,
)


def _outcome(
    action: str,
    score: float,
    correct: bool,
    *,
    violated: tuple[str, ...] = (),
    flags: tuple[str, ...] = (),
) -> ContinuationOutcome:
    return ContinuationOutcome(
        action_id=action,
        behavior_score=score,
        is_correct=correct,
        violated_state_ids=violated,
        drift_flags=flags,
    )


def _channels(
    workspace: ChannelCondition,
    memory: ChannelCondition,
    *,
    workspace_digest: str,
    memory_digest: str = "memory-current-hash",
) -> ChannelSnapshot:
    def state_ids(condition: ChannelCondition) -> tuple[str, ...]:
        if condition == "current":
            return ("P2",)
        if condition == "stale":
            return ("P1",)
        return ()

    return ChannelSnapshot(
        workspace_condition=workspace,
        memory_condition=memory,
        workspace_state_ids=state_ids(workspace),
        memory_state_ids=state_ids(memory),
        workspace_digest=workspace_digest,
        memory_digest=memory_digest,
    )


def _attempt(
    attempt_id: str,
    channels: ChannelSnapshot,
    outcome: ContinuationOutcome,
    *,
    declared: FaultChannel | None = None,
    diagnosis: TraceableDriftDiagnosis | None = None,
) -> ControlAttempt:
    return ControlAttempt(
        attempt_id=attempt_id,
        channels=channels,
        outcome=outcome,
        declared_fault_channel=declared,
        drift_diagnosis=diagnosis,
    )


def _pair(
    prefix: str,
    channels: ChannelSnapshot,
    outcome: ContinuationOutcome,
    **kwargs: object,
) -> tuple[ControlAttempt, ControlAttempt]:
    return (
        _attempt(f"{prefix}-r1", channels, outcome, **kwargs),
        _attempt(f"{prefix}-r2", channels, outcome, **kwargs),
    )


def _recovered_epoch(*, with_retention: bool = True) -> StateControlEpoch:
    baseline_channels = _channels(
        "current",
        "current",
        workspace_digest="workspace-current-v1",
    )
    faulted_channels = _channels(
        "stale",
        "current",
        workspace_digest="workspace-stale-fault",
    )
    repaired_channels = _channels(
        "current",
        "current",
        workspace_digest="workspace-current-repair",
    )
    correct = _outcome("safe_v2_offline", 1.0, True)
    stale = _outcome(
        "stale_v1",
        0.0,
        False,
        violated=("P2",),
        flags=("stale-state:P1", "plan-deviation"),
    )
    diagnosis = diagnose_long_horizon_drift(
        DriftEvidence(
            outcome=stale,
            action_expressed_state_ids=("P1",),
            current_plan_state_ids=("P2",),
            stale_state_ids=("P1",),
        )
    )
    return StateControlEpoch(
        epoch_id="epoch-001",
        injection=KnownFaultInjection(
            fault_id="fault-001",
            target_channel="workspace",
            kind="stale_replacement",
            target_state_ids=("P2",),
            before_condition="current",
            after_condition="stale",
            injection_digest="injection-hash",
        ),
        baseline=_pair("baseline", baseline_channels, correct),
        faulted=_pair(
            "faulted",
            faulted_channels,
            stale,
            declared="workspace",
            diagnosis=diagnosis,
        ),
        repaired=_pair("repaired", repaired_channels, correct),
        repair_rounds_to_candidate=1,
        retention=(
            _pair("retention", repaired_channels, correct)
            if with_retention
            else ()
        ),
    )


def test_traceable_drift_diagnosis_exposes_exact_predicates_and_flags() -> None:
    outcome = _outcome(
        "cloud_shortcut",
        0.25,
        False,
        violated=("C1", "P2"),
        flags=("constraint-violation:C1", "scope-overreach"),
    )
    diagnosis = diagnose_long_horizon_drift(
        DriftEvidence(
            outcome=outcome,
            active_constraint_ids=("C1",),
            current_plan_state_ids=("P2",),
            selected_local_state_ids=("L1",),
            global_state_ids=("C1",),
        )
    )

    assert diagnosis.traceable
    assert diagnosis.result.flags == (
        "constraint_loss",
        "local_over_global",
        "plan_deviation",
    )
    evidence = {item.mode: item for item in diagnosis.mode_evidence}
    assert evidence["constraint_loss"].predicate_state_ids == ("C1",)
    assert evidence["constraint_loss"].checker_flags == (
        "constraint-violation:C1",
    )
    assert evidence["local_over_global"].evidence_sources == (
        "direct_state_predicate",
        "checker_flag",
    )


def test_known_workspace_fault_is_causally_identified_repaired_and_retained() -> None:
    result = analyze_state_control_epoch(_recovered_epoch())

    assert result.label == "recovered_and_retained"
    assert result.causal_fault_identified
    assert result.root_cause == "workspace:stale_replacement"
    assert result.diagnosis_correct is True
    assert result.repair_success
    assert result.retained is True
    assert result.fault_effect == 1.0
    assert result.repair_gain == 1.0
    assert result.failure_modes == ("plan_deviation", "stale_state")


def test_fault_effect_without_successful_repair_is_not_called_recovery() -> None:
    epoch = _recovered_epoch(with_retention=False)
    failed_repair = _outcome("stale_v1", 0.0, False, flags=("stale-state:P1",))
    broken = replace(
        epoch,
        repaired=_pair("repair-failed", epoch.repaired[0].channels, failed_repair),
    )

    result = analyze_state_control_epoch(broken)

    assert result.label == "fault_effect_identified_repair_failed"
    assert result.causal_fault_identified
    assert not result.repair_success


def test_ineffective_fault_is_reported_separately_from_agent_robustness() -> None:
    epoch = _recovered_epoch(with_retention=False)
    unchanged = _outcome("safe_v2_offline", 1.0, True)
    ineffective = replace(
        epoch,
        faulted=_pair(
            "fault-no-effect",
            epoch.faulted[0].channels,
            unchanged,
            declared="workspace",
        ),
    )

    result = analyze_state_control_epoch(ineffective)

    assert result.label == "fault_not_effective"
    assert not result.causal_fault_identified
    assert result.root_cause == "not_identified"


def test_epoch_rejects_faults_that_change_both_channels() -> None:
    epoch = _recovered_epoch()
    confounded_channels = replace(
        epoch.faulted[0].channels,
        memory_digest="also-mutated-memory",
    )

    with pytest.raises(ValueError, match="non-target channel"):
        replace(
            epoch,
            faulted=_pair(
                "confounded",
                confounded_channels,
                epoch.faulted[0].outcome,
            ),
        )


def test_core_matrix_plans_four_isolated_known_fault_repair_protocols() -> None:
    arms = build_core_channel_matrix(
        DecisionAnchor(
            episode_id="episode-1",
            opportunity_id="decision-1",
            checkpoint_session=8,
            decision_family="branch_selection",
            authoritative_state_ids=("P2",),
            stale_state_ids=("P1",),
            gold_action_ids=("safe_v2_offline",),
            prompt_template_id="branch-decision-v1",
        ),
        ChallengeAxes(
            horizon_handoffs=8,
            state_transition_count=2,
            workspace_version_depth=3,
            workspace_distractor_count=4,
            memory_distractor_count=4,
            authority_depth=2,
            dependency_depth=3,
        ),
        block_id="block-1",
    )

    protocols = plan_core_known_fault_protocols(arms, retention_handoffs=2)

    assert len(protocols) == 4
    assert {item.target_channel for item in protocols} == {"workspace", "memory"}
    assert {item.fault_kind for item in protocols} == {
        "stale_replacement",
        "current_evidence_deletion",
    }
    assert all(item.baseline_arm_id == item.repair_arm_id for item in protocols)
    assert all(item.fault_arm_id != item.baseline_arm_id for item in protocols)
    assert all(item.planned_policy_calls == 8 for item in protocols)
