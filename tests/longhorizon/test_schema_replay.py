from __future__ import annotations

import pytest

from lhmsb.longhorizon.replay import StateReplayError, plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    EpisodePlan,
    StateEvent,
    StateUnit,
    WorkspaceArtifact,
    WorkspaceSnapshot,
)


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
    restored = EpisodePlan.from_dict(plan.to_dict())
    assert restored == plan
    assert plan_hash(restored) == plan_hash(plan)


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
