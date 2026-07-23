from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.longhorizon.replay import StateReplayError, plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    ContinuationOpportunity,
    EpisodePlan,
    StateEvent,
    StateUnit,
    TaskStep,
    WorkspaceArtifact,
    WorkspaceSnapshot,
    task_step_effect_digest,
)
from lhmsb.longhorizon.task_span import build_software_task_steps


def _plan() -> EpisodePlan:
    states = (
        StateUnit(
            state_id="G0",
            kind="global_goal",
            value={"text": "ship an auditable pipeline"},
            authority="owner",
            scope="project",
            valid_from=0,
            workspace_recoverability="explicit",
        ),
        StateUnit(
            state_id="C1",
            kind="constraint",
            value={"text": "offline only"},
            authority="owner",
            scope="all",
            valid_from=0,
            dependency_ids=("G0",),
            workspace_recoverability="absent",
        ),
        StateUnit(
            state_id="P1",
            kind="plan_node",
            value={"version": "v1"},
            authority="engineer",
            scope="pipeline",
            valid_from=0,
            dependency_ids=("G0",),
            workspace_recoverability="derivable",
        ),
    )
    events = (
        StateEvent("add-g0", 0, "add", "G0", new_version=1),
        StateEvent("add-c1", 0, "add", "C1", new_version=1),
        StateEvent("add-p1", 0, "add", "P1", new_version=1),
        StateEvent(
            "replace-p1",
            2,
            "replace",
            "P1",
            old_version=1,
            new_version=2,
            authority="owner",
            invalidates=("P1",),
        ),
        StateEvent("revoke-c1", 3, "revoke", "C1", old_version=1),
        StateEvent("reopen-c1", 4, "reopen", "C1", new_version=2),
    )
    workspace = (
        WorkspaceSnapshot(
            checkpoint_session=0,
            artifacts=(
                WorkspaceArtifact(
                    path="README.md",
                    content="pipeline v1",
                    version=1,
                    source_event_ids=("add-p1",),
                    created_session=0,
                    updated_session=0,
                ),
            ),
            recoverability_by_state=(
                ("G0", "explicit"),
                ("C1", "absent"),
                ("P1", "derivable"),
            ),
        ),
    )
    return EpisodePlan(
        episode_id="test-episode",
        template_id="test-template",
        semantic_seed=1,
        trajectory_seed=2,
        n_sessions=5,
        initial_goal="G0",
        state_units=states,
        events=events,
        workspaces=workspace,
    )


def test_episode_plan_round_trips_to_canonical_json() -> None:
    plan = _plan()
    assert "task_steps" not in plan.to_dict()
    restored = EpisodePlan.from_dict(plan.to_dict())
    assert restored == plan
    assert plan_hash(restored) == plan_hash(plan)


def test_episode_plan_round_trips_valid_causal_task_steps() -> None:
    plan = _plan()
    steps = (
        TaskStep(
            step_id="step-000",
            ordinal=0,
            session=0,
            kind="inspect",
            execution_mode="frozen_replay",
            summary="Inspected the current project handoff.",
            reads_state_ids=("G0",),
            workspace_paths=("README.md",),
        ),
        TaskStep(
            step_id="step-001",
            ordinal=1,
            session=1,
            kind="handoff",
            execution_mode="environment_generated",
            summary="Resumed the dependent work in a fresh session.",
            dependency_step_ids=("step-000",),
        ),
        TaskStep(
            step_id="step-002",
            ordinal=2,
            session=2,
            kind="continuation_decision",
            execution_mode="policy_evaluated",
            summary="",
            dependency_step_ids=("step-001",),
            reads_state_ids=("G0", "P1"),
            visible_in_session=False,
        ),
    )
    enriched = EpisodePlan.from_dict(
        {**plan.to_dict(), "task_steps": [step.__dict__ for step in steps]}
    )

    assert enriched.task_steps == steps
    assert EpisodePlan.from_dict(enriched.to_dict()) == enriched
    assert "task_steps" in enriched.to_dict()


def test_episode_plan_rejects_noncausal_or_forward_task_steps() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="causal"):
        EpisodePlan.from_dict(
            {
                **plan.to_dict(),
                "task_steps": [
                    {
                        "step_id": "padding",
                        "ordinal": 0,
                        "session": 0,
                        "kind": "record",
                        "execution_mode": "frozen_replay",
                        "summary": "A filler line.",
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="earlier step"):
        EpisodePlan.from_dict(
            {
                **plan.to_dict(),
                "task_steps": [
                    {
                        "step_id": "step-000",
                        "ordinal": 0,
                        "session": 0,
                        "kind": "handoff",
                        "execution_mode": "environment_generated",
                        "summary": "Start.",
                        "dependency_step_ids": ["step-001"],
                    }
                ],
            }
        )


def test_episode_plan_rejects_tampered_task_effect_chain() -> None:
    plan = _plan()
    spec_steps = build_software_task_steps(plan, steps_per_session=2)
    tampered = replace(
        spec_steps[-1],
        dependency_effect_digests=("0" * 64,),
    )

    with pytest.raises(ValueError, match="dependency effect digest mismatch"):
        replace(plan, task_steps=(*spec_steps[:-1], tampered))


def test_episode_plan_rejects_duplicate_semantic_effects() -> None:
    plan = _plan()
    steps = build_software_task_steps(plan, steps_per_session=2)
    duplicate = replace(
        steps[1],
        produces_effect_ids=steps[0].produces_effect_ids,
        effect_digest="",
    )
    duplicate = replace(duplicate, effect_digest=task_step_effect_digest(duplicate))

    with pytest.raises(ValueError, match="duplicate semantic effect"):
        replace(plan, task_steps=(steps[0], duplicate, *steps[2:]))


def test_episode_plan_rejects_semantic_effect_dependency_mismatch() -> None:
    plan = _plan()
    steps = build_software_task_steps(plan, steps_per_session=2)
    mismatched = replace(
        steps[1],
        consumes_effect_ids=(),
        effect_digest="",
    )
    mismatched = replace(mismatched, effect_digest=task_step_effect_digest(mismatched))

    with pytest.raises(ValueError, match="do not align with causal dependencies"):
        replace(plan, task_steps=(steps[0], mismatched, *steps[2:]))


def test_unknown_continuation_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown continuation scope"):
        ContinuationOpportunity(
            opportunity_id="opp",
            checkpoint_session=0,
            focal_state_ids=(),
            challenge_type="test",
            request="continue",
            action_catalog=(),
            valid_action_ids=(),
            matched_group="test",
            continuation_scope="unknown",  # type: ignore[arg-type]
        )


def test_replay_tracks_current_state_and_history() -> None:
    plan = _plan()

    at_one = replay_plan(plan, 1)
    assert set(at_one.current) == {"G0", "C1", "P1"}
    assert at_one.current["P1"].version == 1

    at_two = replay_plan(plan, 2)
    assert set(at_two.current) == {"G0", "C1", "P1"}
    assert at_two.current["P1"].version == 2
    assert at_two.history["P1"] == (1, 2)
    assert "P1" in at_two.invalidated

    at_three = replay_plan(plan, 3)
    assert "C1" not in at_three.current
    at_four = replay_plan(plan, 4)
    assert at_four.current["C1"].version == 2


def test_replay_rejects_unknown_target_and_bad_version() -> None:
    plan = _plan()
    unknown_payload = plan.to_dict()
    unknown_payload["events"] = [
        *unknown_payload["events"],  # type: ignore[index]
        {
            "event_id": "bad",
            "session": 1,
            "type": "revoke",
            "target_state_id": "MISSING",
            "old_version": 1,
            "new_version": None,
            "authority": "owner",
            "scope": "project",
            "reason_state_ids": [],
            "invalidates": [],
        },
    ]
    unknown = EpisodePlan.from_dict(unknown_payload)
    with pytest.raises(StateReplayError, match="unknown state"):
        replay_plan(unknown, 1)

    bad_payload = plan.to_dict()
    bad_payload["events"] = [
        *bad_payload["events"][:3],  # type: ignore[index]
        {
            "event_id": "bad-version",
            "session": 2,
            "type": "replace",
            "target_state_id": "P1",
            "old_version": 99,
            "new_version": 2,
            "authority": "owner",
            "scope": "project",
            "reason_state_ids": [],
            "invalidates": [],
        },
    ]
    bad_version = EpisodePlan.from_dict(bad_payload)
    with pytest.raises(StateReplayError, match="version"):
        replay_plan(bad_version, 2)


def test_replay_rejects_unreplayable_dependency_closure_and_window() -> None:
    plan = _plan()
    dependent = StateUnit(
        state_id="D1",
        kind="decision",
        value="depends on missing state",
        authority="owner",
        scope="project",
        valid_from=0,
        dependency_ids=("MISSING",),
    )
    payload = plan.to_dict()
    payload["state_units"] = [*payload["state_units"], dependent.__dict__]  # type: ignore[index]
    payload["events"] = [
        *payload["events"],  # type: ignore[index]
        {
            "event_id": "add-d1",
            "session": 1,
            "type": "add",
            "target_state_id": "D1",
            "old_version": None,
            "new_version": 1,
            "authority": "owner",
            "scope": "project",
            "reason_state_ids": [],
            "invalidates": [],
        },
    ]
    with pytest.raises(StateReplayError, match="dependency"):
        replay_plan(EpisodePlan.from_dict(payload), 1)

    invalid_window = StateUnit(
        state_id="W1",
        kind="fact",
        value="future",
        authority="owner",
        scope="project",
        valid_from=2,
    )
    window_payload = plan.to_dict()
    window_payload["state_units"] = [*window_payload["state_units"], invalid_window.__dict__]  # type: ignore[index]
    window_payload["events"] = [
        *window_payload["events"],  # type: ignore[index]
        {
            "event_id": "add-w1",
            "session": 1,
            "type": "add",
            "target_state_id": "W1",
            "new_version": 1,
        },
    ]
    with pytest.raises(StateReplayError, match="valid window"):
        replay_plan(EpisodePlan.from_dict(window_payload), 1)
