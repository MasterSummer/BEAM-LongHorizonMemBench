from __future__ import annotations

from dataclasses import replace

import pytest

from lhmsb.families.software.horizon_panel import (
    DEFAULT_HORIZON_DOSES,
    HorizonDose,
    SoftwareHorizonPanelFamily,
    audit_horizon_panel,
)
from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker
from lhmsb.longhorizon.task_span import profile_task_span


def test_horizon_panel_holds_terminal_decision_fixed_across_joint_dose() -> None:
    specs = SoftwareHorizonPanelFamily.generate_panel(
        42,
        trajectory_seed=42,
    )
    audit = audit_horizon_panel(specs)

    assert audit.ok, audit.to_dict()
    assert len(specs) == 9
    assert audit.levels == ("short", "medium", "long")
    assert audit.n_sessions == (("short", 4), ("medium", 8), ("long", 16))
    assert audit.unique_episode_ids
    assert audit.within_dose_triplets_ok
    assert audit.long_level_meets_effective_step_threshold
    assert {item.variant for item in audit.variant_audits} == {
        "static",
        "evolution",
        "hierarchical_conflict",
    }
    for item in audit.variant_audits:
        assert item.ok, item.errors
        assert item.terminal_decision_signature_count == 1
        assert item.terminal_state_signature_count == 1
        assert item.terminal_workspace_signature_count == 1
        assert item.opaque_option_signature_count == 1
        assert item.executable_checker_signature_count == 1
        assert item.terminal_condition_signature_count == 1
        assert item.effective_step_counts == (
            ("short", 65),
            ("medium", 129),
            ("long", 257),
        )
        assert item.handoff_counts == (
            ("short", 3),
            ("medium", 7),
            ("long", 15),
        )
        assert item.strictly_increasing_joint_dose


def test_horizon_panel_preserves_within_dose_grouping_and_cross_dose_mapping() -> None:
    specs = SoftwareHorizonPanelFamily.generate_panel(
        44,
        trajectory_seed=9,
    )
    panel_ids = {
        spec.plan.metadata_dict["horizon_panel_id"] for spec in specs
    }
    groups_by_level = {
        level: {
            spec.plan.metadata_dict["counterfactual_group_id"]
            for spec in specs
            if spec.plan.metadata_dict["horizon_level"] == level
        }
        for level in ("short", "medium", "long")
    }

    assert len(panel_ids) == 1
    assert all(len(groups) == 1 for groups in groups_by_level.values())
    assert len({next(iter(groups)) for groups in groups_by_level.values()}) == 3
    assert {
        spec.plan.opportunities[0].valid_action_ids for spec in specs
    } == {("cloud_shortcut",)}
    assert len(
        {
            spec.evaluator_continuations[0].option_to_action
            for spec in specs
        }
    ) == 1


def test_horizon_panel_gold_action_is_executable_at_every_dose() -> None:
    specs = SoftwareHorizonPanelFamily.generate_panel(43, trajectory_seed=43)

    for spec in specs:
        opportunity = spec.plan.opportunities[0]
        result = SoftwareVerticalChecker(spec).check_action(
            opportunity.valid_action_ids[0],
            checkpoint_session=opportunity.checkpoint_session,
            opportunity_id=opportunity.opportunity_id,
        )
        assert result.is_correct, (
            spec.plan.metadata_dict["horizon_level"],
            spec.plan.metadata_dict["counterfactual_variant"],
            result,
        )


def test_only_long_default_dose_claims_effective_long_horizon_span() -> None:
    specs = SoftwareHorizonPanelFamily.generate_panel(42)
    by_level = {
        level: [
            profile_task_span(spec.plan)
            for spec in specs
            if spec.plan.metadata_dict["horizon_level"] == level
        ]
        for level in ("short", "medium", "long")
    }

    assert not any(
        profile.meets_long_horizon_step_threshold
        for profile in by_level["short"]
    )
    assert not any(
        profile.meets_long_horizon_step_threshold
        for profile in by_level["medium"]
    )
    assert all(
        profile.meets_long_horizon_step_threshold
        for profile in by_level["long"]
    )


def test_horizon_audit_rejects_terminal_surface_change() -> None:
    specs = list(SoftwareHorizonPanelFamily.generate_panel(42))
    medium_evolution_index = next(
        index
        for index, spec in enumerate(specs)
        if spec.plan.metadata_dict["horizon_level"] == "medium"
        and spec.plan.metadata_dict["counterfactual_variant"] == "evolution"
    )
    target = specs[medium_evolution_index]
    public = target.public_continuations[0]
    specs[medium_evolution_index] = replace(
        target,
        public_continuations=(
            replace(public, request=public.request + " Changed after freezing."),
        ),
    )

    audit = audit_horizon_panel(tuple(specs))

    assert not audit.ok
    evolution = next(
        item for item in audit.variant_audits if item.variant == "evolution"
    )
    assert evolution.opaque_option_signature_count == 2
    assert any("opaque option mapping" in error for error in evolution.errors)


def test_horizon_doses_must_be_unique_and_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SoftwareHorizonPanelFamily.generate_panel(
            42,
            doses=(HorizonDose("short", 8), HorizonDose("long", 4)),
        )
    with pytest.raises(ValueError, match="unique"):
        audit_horizon_panel(
            (),
            doses=(
                DEFAULT_HORIZON_DOSES[0],
                HorizonDose("short", 8),
            ),
        )
