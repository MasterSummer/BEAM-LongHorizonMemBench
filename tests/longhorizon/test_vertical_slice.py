from __future__ import annotations

from pathlib import Path

from lhmsb.datasets.stateful_pipeline import (
    freeze_stateful,
    generate_stateful_to_staging,
    regen_check_stateful,
    verify_stateful,
)
from lhmsb.families.software.vertical import SoftwareVerticalFamily
from lhmsb.longhorizon.runner import run_vertical_episode


def test_four_session_ci_fixture_and_sixteen_session_exemplar_share_schema(
    tmp_path: Path,
) -> None:
    ci = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    exemplar = SoftwareVerticalFamily.generate(seed=42, n_sessions=16, trajectory_seed=0)
    assert ci.plan.n_sessions == 4
    assert exemplar.plan.n_sessions == 16
    assert {state.state_id for state in ci.plan.state_units} == {
        state.state_id for state in exemplar.plan.state_units
    }
    assert set(ci.plan.to_dict()) == set(exemplar.plan.to_dict())

    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_stateful_to_staging(
        stage,
        family="software",
        seeds=(42,),
        n_episodes=1,
        n_sessions=16,
    )
    freeze_stateful(stage, frozen)
    assert verify_stateful(frozen).ok
    assert regen_check_stateful(frozen).ok


def test_vertical_slice_reconstructs_native_chain_and_distinguishes_conditions() -> None:
    spec = SoftwareVerticalFamily.generate(seed=42, n_sessions=4, trajectory_seed=0)
    runs = {
        condition: run_vertical_episode(spec, condition)
        for condition in ("workspace_only", "oracle_current_state", "fake_native")
    }
    assert runs["oracle_current_state"].behavior_score >= runs["workspace_only"].behavior_score
    native = runs["fake_native"]
    late = next(item for item in native.sceu_results if item.opportunity_id == "opp-late")
    assert late.stored_state_ids
    assert late.retrieved_state_ids
    assert late.model_visible_state_ids
    assert late.used_state_ids
    assert late.behavior.metadata_dict["action_id"] == late.selected_action
