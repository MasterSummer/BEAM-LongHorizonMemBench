from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.public_surface import canonical_public_json


def _state_text(spec: object, state_id: str) -> str:
    plan = spec.plan  # type: ignore[attr-defined]
    state = next(item for item in plan.state_units if item.state_id == state_id)
    assert isinstance(state.value, dict)
    return str(state.value["text"])


def test_template_separates_goal_from_offline_constraint() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16, trajectory_seed=2)
    assert spec.plan.template_id == "software-project-mem0-v2"
    assert spec.plan.episode_id == "software-mem0-42"
    assert "offline" not in _state_text(spec, "G0").lower()
    assert "cloud" not in _state_text(spec, "G0").lower()
    assert "offline" in _state_text(spec, "C1").lower()
    assert "cloud services" in _state_text(spec, "C1").lower()


def test_valid_local_accelerator_targets_scoped_authorization_for_intervention() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16, trajectory_seed=2)

    targets = {
        unit.opportunity_id: unit.intervention_target_ids
        for unit in spec.plan.sceu_units
        if unit.opportunity_id
        in {
            "opp-local-valid",
            "opp-local-valid-recheck",
            "opp-local-authority-confirmed",
        }
    }
    assert targets == {
        "opp-local-valid": ("L1",),
        "opp-local-valid-recheck": ("L1",),
        "opp-local-authority-confirmed": ("L1",),
    }


def test_gold_actions_are_balanced_against_a_single_default_policy() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    counts = Counter(
        action_id
        for opportunity in spec.plan.opportunities
        for action_id in opportunity.valid_action_ids
    )

    assert counts == {
        "safe_v2_offline": 6,
        "stale_v1": 3,
        "cloud_shortcut": 3,
    }
    assert max(counts.values()) / sum(counts.values()) <= 0.50
    assert min(counts.values()) >= 3


def test_recoverability_variants_share_latent_state_but_change_workspace() -> None:
    explicit = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=0)
    derivable = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=1)
    absent = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    assert explicit.plan.state_units == derivable.plan.state_units == absent.plan.state_units
    assert explicit.plan.events == derivable.plan.events == absent.plan.events
    checkpoint = explicit.plan.n_sessions - 1
    assert explicit.plan.workspaces[checkpoint].recoverability["C1"] == "explicit"
    assert derivable.plan.workspaces[checkpoint].recoverability["C1"] == "derivable"
    assert absent.plan.workspaces[checkpoint].recoverability["C1"] == "absent"
    assert explicit.plan.workspaces[checkpoint] != derivable.plan.workspaces[checkpoint]
    assert derivable.plan.workspaces[checkpoint] != absent.plan.workspaces[checkpoint]


def test_absent_workspace_does_not_semantically_repeat_c1() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    late = spec.plan.workspaces[-1]
    text = "\n".join(artifact.content for artifact in late.artifacts).lower()
    assert "fully offline" not in text
    assert "do not call cloud" not in text
    assert "network_access" not in text
    assert "offline" not in text


def test_neutral_boilerplate_does_not_repeat_old_state() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    late = spec.plan.sessions[-1]
    assert late.observations[0] == (
        "Continue the software project from the current workspace and session updates."
    )
    assert "offline" not in late.observations[0].lower()
    assert "goal" not in late.observations[0].lower()


def test_write_transcript_excludes_unread_raw_workspace() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    last = spec.plan.sessions[-1]
    raw_only_marker = f"session {last.session_index}: local run completed"
    workspace_text = "\n".join(item.content for item in last.workspace.artifacts)
    transcript = spec.write_transcript(last.session_index)
    assert raw_only_marker in workspace_text
    assert raw_only_marker not in transcript
    assert "recoverability_by_state" not in transcript
    assert "source_event_ids" not in transcript


def test_mem0_spec_exposes_checker_compatible_file_maps() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    assert spec.package_file_map == dict(spec.package_files)
    assert spec.hidden_test_map == dict(spec.hidden_tests)


def test_public_continuations_are_opaque_and_evaluator_mapping_is_private() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    assert len(spec.public_continuations) == len(spec.plan.opportunities)
    assert len(spec.evaluator_continuations) == len(spec.plan.opportunities)
    public = canonical_public_json(spec.public_continuations)
    for forbidden in (
        "G0",
        "C1",
        "P2",
        "safe_v2_offline",
        "stale_v1",
        "cloud_shortcut",
        "valid_action_ids",
        "violates_state_ids",
        "global_utility",
    ):
        assert forbidden not in public
    private_actions = {
        action_id
        for continuation in spec.evaluator_continuations
        for _, action_id in continuation.option_to_action
    }
    assert private_actions == {"safe_v2_offline", "stale_v1", "cloud_shortcut"}


def test_four_and_sixteen_session_generators_share_schema() -> None:
    short = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    long = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    assert tuple(state.state_id for state in short.plan.state_units) == tuple(
        state.state_id for state in long.plan.state_units
    )
    assert tuple(event.type for event in short.plan.events) == tuple(
        event.type for event in long.plan.events
    )
    assert tuple(item.challenge_type for item in short.plan.opportunities) == tuple(
        item.challenge_type for item in long.plan.opportunities
    )
    assert set(asdict(short.plan.sessions[0])) == set(asdict(long.plan.sessions[0]))


def test_generation_is_fully_deterministic() -> None:
    first = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    second = SoftwareMem0VerticalFamily.generate(42, trajectory_seed=2)
    assert first == second


def test_formal_episode_seeds_cover_semantic_and_schedule_variants() -> None:
    specs = tuple(
        SoftwareMem0VerticalFamily.generate(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
        )
        for seed in range(50)
    )
    scenarios = {
        dict(spec.plan.metadata)["semantic_scenario"]
        for spec in specs
    }
    schedules = {
        dict(spec.plan.metadata)["phase_signature"]
        for spec in specs
    }
    goals = {
        _state_text(spec, "G0")
        for spec in specs
    }
    assert len(scenarios) == 5
    assert len(goals) == 5
    assert len(schedules) == 10
    assert len(
        {
            (
                dict(spec.plan.metadata)["semantic_scenario"],
                dict(spec.plan.metadata)["phase_signature"],
            )
            for spec in specs
        }
    ) == 50
