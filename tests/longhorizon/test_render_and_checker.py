from __future__ import annotations

from lhmsb.families.software.vertical import SoftwareVerticalFamily
from lhmsb.families.software.vertical_checker import BehaviorResult, SoftwareVerticalChecker
from lhmsb.longhorizon.render import render_surfaces, surfaces_hash


def test_renderer_hides_evaluator_metadata_and_future_values() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=0)
    surfaces = render_surfaces(spec.plan)
    early = surfaces[0]
    rendered_text = "\n".join(early.observations + early.tool_results)
    rendered_text += "\n" + "\n".join(artifact.content for artifact in early.workspace.artifacts)
    assert "G0" not in rendered_text
    assert "e-10-leakage" not in rendered_text
    assert "data leakage" not in rendered_text.lower()
    assert "pipeline/v2/core.py" not in {artifact.path for artifact in early.workspace.artifacts}
    assert not early.workspace.recoverability_by_state
    assert all(not artifact.source_event_ids for artifact in early.workspace.artifacts)
    assert surfaces_hash(surfaces) == spec.surface_hash


def test_workspace_recoverability_is_not_latent_state() -> None:
    explicit = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    absent = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=2)
    assert explicit.plan.state_units == absent.plan.state_units
    assert explicit.plan.events == absent.plan.events
    assert explicit.plan.sessions[-1].workspace.recoverability_by_state == ()
    assert absent.plan.sessions[-1].workspace.recoverability_by_state == ()
    assert explicit.plan.workspaces[-1].recoverability["P2"] == "explicit"
    assert absent.plan.workspaces[-1].recoverability["P2"] == "absent"


def test_vertical_checker_distinguishes_safe_stale_and_cloud_actions() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    checker = SoftwareVerticalChecker(spec)

    safe = checker.check_action("safe_v2_offline", checkpoint_session=3)
    stale = checker.check_action("stale_v1", checkpoint_session=3)
    cloud = checker.check_action("cloud_shortcut", checkpoint_session=3)

    assert isinstance(safe, BehaviorResult)
    assert safe.is_correct
    assert safe.score == 1.0
    assert not safe.violated_state_ids
    assert stale.is_correct is False
    assert "stale-state" in stale.drift_flags
    assert "P2" in stale.violated_state_ids
    assert cloud.is_correct is False
    assert "constraint-violation:C1" in cloud.drift_flags
    assert "C1" in cloud.violated_state_ids


def test_checker_accepts_v1_only_before_replacement() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    checker = SoftwareVerticalChecker(spec)
    early = checker.check_action("stale_v1", checkpoint_session=0)
    assert early.is_correct
    assert not early.drift_flags

    premature = checker.check_action("safe_v2_offline", checkpoint_session=0)
    assert not premature.is_correct
    assert "future-state-adoption" in premature.drift_flags
    assert "plan_deviation" in premature.drift_flags


def test_checker_marks_local_convenience_over_global_constraint() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=0)
    checker = SoftwareVerticalChecker(spec)
    local_session = next(
        item.checkpoint_session
        for item in spec.plan.opportunities
        if item.opportunity_id == "opp-local-only"
    )

    cloud = checker.check_action(
        "cloud_shortcut",
        checkpoint_session=local_session,
        opportunity_id="opp-local-only",
    )
    safe = checker.check_action(
        "safe_v2_offline",
        checkpoint_session=local_session,
        opportunity_id="opp-local-only",
    )

    assert {"constraint_loss", "local_over_global"}.issubset(cloud.drift_flags)
    assert safe.is_correct
    assert not safe.drift_flags
