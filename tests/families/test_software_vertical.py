from __future__ import annotations

from lhmsb.families.software.vertical import SoftwareVerticalFamily, SoftwareVerticalSpec
from lhmsb.longhorizon.replay import plan_hash, replay_plan


def test_software_vertical_has_fixed_state_first_semantics() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=7)

    assert isinstance(spec, SoftwareVerticalSpec)
    assert spec.plan.n_sessions == 4
    assert {state.state_id for state in spec.plan.state_units} >= {
        "G0",
        "C1",
        "C2",
        "P1",
        "U1",
        "P2",
        "D1",
        "L1",
    }
    assert {action.action_id for action in spec.actions} == {
        "safe_v2_offline",
        "stale_v1",
        "cloud_shortcut",
    }
    assert len(spec.plan.events) >= 10
    assert len(spec.plan.workspaces) == 4
    assert len(spec.plan.opportunities) >= 4
    assert len(spec.plan.sceu_units) == len(spec.plan.opportunities)

    at_start = replay_plan(spec.plan, 0)
    assert {"G0", "C1", "C2", "P1"} <= set(at_start.current)
    late = replay_plan(spec.plan, 3)
    assert "P2" in late.current
    assert "P1" not in late.current
    assert "C1" in late.current
    assert "D1" in late.current
    assert "L1" in late.current


def test_local_proposal_and_authorization_are_distinct_state_events() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=2)
    proposal = replay_plan(spec.plan, 8)
    authorized = replay_plan(spec.plan, 9)

    assert "D1" in proposal.current
    assert "L1" not in proposal.current
    assert proposal.current["D1"].authority == "local-operator"
    assert authorized.current["L1"].authority == "project-owner"
    assert authorized.current["L1"].workspace_recoverability == "absent"


def test_vertical_generation_is_reproducible_and_horizon_parameterized() -> None:
    first = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=0)
    second = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=0)
    assert plan_hash(first.plan) == plan_hash(second.plan)
    assert first.surface_hash == second.surface_hash
    assert len(first.plan.workspaces) == 16
    assert len(first.plan.sessions) == 16
    assert len(first.plan.events) >= 10


def test_recoverability_variants_preserve_latent_plan_but_change_workspace() -> None:
    explicit = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    derivable = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=1)
    absent = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=2)

    assert plan_hash(explicit.plan) != plan_hash(derivable.plan)
    assert {state.state_id for state in explicit.plan.state_units} == {
        state.state_id for state in derivable.plan.state_units
    }
    assert explicit.plan.metadata_dict["recoverability_variant"] == "explicit"
    assert derivable.plan.metadata_dict["recoverability_variant"] == "derivable"
    assert absent.plan.metadata_dict["recoverability_variant"] == "absent"
    assert (
        explicit.plan.workspaces[-1].recoverability["P2"]
        != absent.plan.workspaces[-1].recoverability["P2"]
    )


def test_workspace_contains_audit_trail_without_future_branch_at_start() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    start = spec.plan.workspaces[0]
    paths = {artifact.path for artifact in start.artifacts}
    assert "pipeline/v1/core.py" in paths
    assert "pipeline/v2/core.py" not in paths
    assert "results/session_0.json" in paths
    assert any("data leakage" not in artifact.content.lower() for artifact in start.artifacts)
