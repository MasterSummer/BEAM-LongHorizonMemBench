from __future__ import annotations

from lhmsb.adapters.vertical_stub import VerticalStubAdapter
from lhmsb.families.software.vertical import SoftwareVerticalFamily
from lhmsb.longhorizon.runner import run_vertical_episode


def test_fake_native_has_real_session_write_retrieval_and_context_reset() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    result = run_vertical_episode(spec, "fake_native")

    assert result.condition == "fake_native"
    assert result.episode_id == spec.plan.episode_id
    assert result.sceu_results
    assert result.stored_state_ids
    assert result.retrieved_state_ids
    assert result.model_visible_state_ids
    assert result.transcript_hash
    assert any(record.retrieved_state_ids for record in result.sceu_results)
    assert any(record.used_state_ids for record in result.sceu_results)
    assert any(event.operation == "write" for event in result.native_trace)
    assert any(event.operation == "search" for event in result.native_trace)
    assert any(event.operation == "clear_working_context" for event in result.native_trace)


def test_conditions_share_prefix_and_separate_workspace_from_memory_effects() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=2)
    workspace = run_vertical_episode(spec, "workspace_only")
    oracle = run_vertical_episode(spec, "oracle_current_state")
    native = run_vertical_episode(spec, "fake_native")

    assert (
        workspace.workspace_snapshot_hash
        == oracle.workspace_snapshot_hash
        == native.workspace_snapshot_hash
    )
    assert workspace.prefix_hash == oracle.prefix_hash == native.prefix_hash
    assert oracle.behavior_score >= workspace.behavior_score
    assert oracle.behavior_score >= native.behavior_score
    assert workspace.condition == "workspace_only"
    assert oracle.condition == "oracle_current_state"


def test_oracle_uses_current_state_and_workspace_only_fails_on_absent_branch() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=2)
    workspace = run_vertical_episode(spec, "workspace_only")
    oracle = run_vertical_episode(spec, "oracle_current_state")

    late_workspace = next(
        item for item in workspace.sceu_results if item.opportunity_id == "opp-late"
    )
    late_oracle = next(item for item in oracle.sceu_results if item.opportunity_id == "opp-late")
    assert late_oracle.selected_action == "safe_v2_offline"
    assert late_oracle.behavior.is_correct
    assert late_workspace.selected_action == "stale_v1"
    assert "stale-state" in late_workspace.behavior.drift_flags


def test_leave_one_out_removes_key_native_memory_and_changes_action() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    normal = run_vertical_episode(spec, "fake_native")
    intervention = run_vertical_episode(spec, "fake_native", intervention_state_id="P2")

    normal_late = next(item for item in normal.sceu_results if item.opportunity_id == "opp-late")
    intervention_late = next(
        item for item in intervention.sceu_results if item.opportunity_id == "opp-late"
    )
    assert normal_late.selected_action == "safe_v2_offline"
    assert intervention_late.selected_action == "stale_v1"
    assert intervention_late.intervened_state_id == "P2"


def test_stub_adapter_ids_and_lexical_search_are_deterministic() -> None:
    adapter = VerticalStubAdapter()
    adapter.initialize(user_id="u", session_id="s0")
    first = adapter.add_memory(
        "current v2 branch is offline",
        user_id="u",
        session_id="s0",
        metadata={"state_ids": ("P2",)},
    )
    second = adapter.add_memory(
        "the heldout test set is frozen",
        user_id="u",
        session_id="s0",
        metadata={"state_ids": ("C2",)},
    )
    result = adapter.search("current v2 branch", user_id="u", session_id="s1")
    assert first == "vertical-0001"
    assert second == "vertical-0002"
    assert [entry.memory_id for entry in result.results] == [first]
    adapter.begin_session("s1")
    assert adapter.working_context == ()
