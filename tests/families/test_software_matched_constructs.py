from __future__ import annotations

from lhmsb.families.software.matched_constructs import (
    MATCHED_TARGET_OPPORTUNITY_ID,
    SoftwareMatchedConstructFamily,
    audit_matched_construct_triplet,
    decision_signature,
    prefix_shape_signature,
    terminal_condition_signature,
    workspace_shape_signature,
)
from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker
from lhmsb.longhorizon.attribution import eligible_write_state_ids
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.longhorizon.task_span import profile_task_span


def test_matched_triplet_holds_decision_and_prefix_shape_constant() -> None:
    specs = SoftwareMatchedConstructFamily.generate_triplet(
        42,
        n_sessions=16,
        trajectory_seed=42,
        steps_per_session=16,
    )
    audit = audit_matched_construct_triplet(specs)

    assert audit.ok, audit.errors
    assert audit.decision_signature_count == 1
    assert audit.prefix_shape_signature_count == 1
    assert audit.workspace_shape_signature_count == 1
    assert audit.option_surface_signature_count == 1
    assert audit.terminal_condition_signature_count == 1
    assert audit.minimum_target_handoff_count == 15
    assert audit.all_targets_at_final_session
    assert audit.all_meet_long_horizon_step_threshold
    assert len({plan_hash(spec.plan) for spec in specs}) == 3
    assert len({spec.surface_hash for spec in specs}) == 3
    assert len(
        {
            decision_signature(spec.plan.opportunities[0])
            for spec in specs
        }
    ) == 1
    assert len(
        {
            terminal_condition_signature(
                spec.plan,
                MATCHED_TARGET_OPPORTUNITY_ID,
            )
            for spec in specs
        }
    ) == 1
    assert len(
        {
            prefix_shape_signature(spec.plan, MATCHED_TARGET_OPPORTUNITY_ID)
            for spec in specs
        }
    ) == 1
    assert len(
        {
            workspace_shape_signature(
                spec.plan,
                MATCHED_TARGET_OPPORTUNITY_ID,
            )
            for spec in specs
        }
    ) == 1


def test_matched_triplet_manipulates_construct_not_terminal_gold() -> None:
    specs = SoftwareMatchedConstructFamily.generate_triplet(
        47,
        n_sessions=16,
        trajectory_seed=47,
        steps_per_session=16,
    )
    by_variant = {
        spec.plan.metadata_dict["counterfactual_variant"]: spec
        for spec in specs
    }
    expected = {
        "static": "static_recall",
        "evolution": "state_evolution",
        "hierarchical_conflict": "hierarchical_conflict",
    }

    for variant, spec in by_variant.items():
        opportunity = spec.plan.opportunities[0]
        sceu = spec.plan.sceu_units[0]
        assert opportunity.opportunity_id == MATCHED_TARGET_OPPORTUNITY_ID
        assert opportunity.valid_action_ids == ("cloud_shortcut",)
        assert opportunity.checkpoint_session == 15
        assert profile_sceu(spec.plan, sceu).construct_kind == expected[variant]
        span = profile_task_span(spec.plan)
        assert span.visible_prefix_step_count == 256
        assert span.policy_evaluated_step_count == 1
        assert span.policy_conditioned_future_step_count == 0
        assert span.policy_dependency_coverage is None
        assert span.interaction_mode == "replay_backed_critical_decision"
        assert not span.declared_closed_loop_dependency
        assert not span.online_long_horizon_agent_execution_supported
        assert all(len(session.observations) == 17 for session in spec.plan.sessions)

    option_mappings = {
        specs[0].evaluator_continuations[0].option_to_action,
        specs[1].evaluator_continuations[0].option_to_action,
        specs[2].evaluator_continuations[0].option_to_action,
    }
    assert len(option_mappings) == 1


def test_terminal_archetypes_balance_gold_and_option_position() -> None:
    specs_by_seed = {
        seed: SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
            steps_per_session=16,
        )
        for seed in (42, 43, 44)
    }

    assert {
        specs[0].plan.opportunities[0].valid_action_ids[0]
        for specs in specs_by_seed.values()
    } == {"safe_v2_offline", "stale_v1", "cloud_shortcut"}
    gold_options = set()
    for specs in specs_by_seed.values():
        mappings = {
            spec.evaluator_continuations[0].option_to_action
            for spec in specs
        }
        assert len(mappings) == 1
        gold_action = specs[0].plan.opportunities[0].valid_action_ids[0]
        gold_options.add(
            next(
                option_id
                for option_id, action_id in next(iter(mappings))
                if action_id == gold_action
            )
        )
    assert gold_options == {"option-01", "option-02", "option-03"}


def test_every_terminal_gold_action_passes_programmatic_checker() -> None:
    for seed in (42, 43, 44):
        for spec in SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
        ):
            opportunity = spec.plan.opportunities[0]
            result = SoftwareVerticalChecker(spec).check_action(
                opportunity.valid_action_ids[0],
                checkpoint_session=opportunity.checkpoint_session,
                opportunity_id=opportunity.opportunity_id,
            )
            assert result.is_correct, (
                seed,
                spec.plan.metadata_dict["counterfactual_variant"],
                result,
            )


def test_neutral_matching_records_never_count_as_storage_targets() -> None:
    specs = SoftwareMatchedConstructFamily.generate_triplet(
        43,
        n_sessions=16,
        trajectory_seed=43,
    )

    assert any(
        state.state_id.startswith("N")
        for spec in specs
        for state in spec.plan.state_units
    )
    for spec in specs:
        for session in range(spec.plan.n_sessions):
            assert all(
                not state_id.startswith("N")
                for state_id in eligible_write_state_ids(spec.plan, session)
            )


def test_short_triplet_is_structurally_matched_without_claiming_full_span() -> None:
    specs = SoftwareMatchedConstructFamily.generate_triplet(
        42,
        n_sessions=4,
        steps_per_session=4,
    )
    audit = audit_matched_construct_triplet(specs)

    assert audit.ok
    assert not audit.all_meet_long_horizon_step_threshold
    assert audit.minimum_effective_step_count >= 16
    assert audit.minimum_target_handoff_count == 3
    assert audit.all_targets_at_final_session
