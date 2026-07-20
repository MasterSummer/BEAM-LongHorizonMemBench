from __future__ import annotations

from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    StateCheckpointMetricInput,
    compute_multisystem_metrics,
    compute_multisystem_metrics_by_cell,
    compute_multisystem_scorecard,
)


def _row(
    condition: str,
    readout: str,
    score: float,
    *,
    live_count: int | None = None,
) -> MultisystemMetricInput:
    return MultisystemMetricInput(
        policy_profile_id="policy-a",
        condition=condition,
        readout=readout,
        result_id=f"{condition}-{readout}",
        behavior_score=score,
        is_correct=score >= 0.5,
        required_state_ids=("C1",),
        stale_state_ids=("V1",),
        candidate_memory_state_ids=(("C1",), ("V1",)),
        retrieved_memory_state_ids=(("C1",), ("V1",)),
        visible_memory_state_ids=(("C1",),),
        candidate_shortfall=False,
        visible_memory_count=1,
        live_memory_count=live_count,
        causal_labels=("beneficial",),
        intervention_labels=("beneficial",),
        leave_one_out_count=1,
        leave_one_out_action_flips=1,
        drift_flags=("stale_state",) if score < 0.5 else (),
    )


def test_schema_v2_controlled_comparisons_are_separate_and_nullable() -> None:
    rows = (
        _row("workspace_only", "none", 0.2),
        _row("full_context", "none", 0.8),
        _row("oracle_current_state", "none", 1.0),
        _row("flat_retrieval", "common_rerank", 0.4),
        _row("mem0", "native", 0.6),
        _row("mem0", "common_rerank", 0.7),
        _row("amem", "native", 0.5),
        _row("amem", "common_rerank", 0.6),
        _row("memos", "native", 0.3),
        _row("memos", "common_rerank", 0.4),
    )
    metrics = compute_multisystem_metrics(rows)
    assert metrics["mem0_native_gain_beyond_workspace"].value == 0.4
    assert metrics["mem0_common_rerank_gain_beyond_workspace"].value == 0.5
    assert metrics["mem0_native_gain_over_flat"].value == 0.2
    assert metrics["mem0_native_gap_to_full_context"].value == 0.2
    assert metrics["mem0_native_oracle_gap_closed"].value == 0.5
    assert metrics["mem0_common_rerank_minus_native"].value == 0.1
    # Flat has no native readout and thus no fabricated native metric.
    assert metrics["flat_retrieval_native_gain_beyond_workspace"].value is None


def test_schema_v2_state_retrieval_and_memory_count_metrics() -> None:
    rows = (
        _row("workspace_only", "none", 0.2, live_count=0),
        _row("mem0", "native", 0.8, live_count=2),
    )
    metrics = compute_multisystem_metrics(
        rows,
        state_checkpoints=(
            StateCheckpointMetricInput(
                eligible_write_state_ids=("C1",),
                new_memory_state_ids=(("C1",),),
                current_state_ids=("C1",),
                future_needed_state_ids=("C1",),
                retired_state_ids=("V1",),
                live_memory_state_ids=(("C1",), ("V1",)),
                live_content_hashes=("h1", "h2"),
                n_write=1,
                n_live=2,
                mutation_counts=(("add", 1), ("update", 0)),
            ),
        ),
    )
    assert metrics["live_memory_count"].value == 2
    assert metrics["mutation_add_count"].value == 1
    assert metrics["stale_candidate_exposure"].value == 0.5
    assert metrics["stale_visible_exposure"].value == 0.0
    assert metrics["retrieval_to_visible_yield"].value == 0.5
    assert metrics["behavior_correct_rate_live_memory_count_2"].value == 1.0


def test_cell_groups_and_scorecard_do_not_mix_native_and_common() -> None:
    rows = (
        _row("mem0", "native", 0.5),
        _row("mem0", "common_rerank", 0.7),
    )
    groups = compute_multisystem_metrics_by_cell(rows)
    assert {(item["condition"], item["readout"]) for item in groups} == {
        ("mem0", "native"),
        ("mem0", "common_rerank"),
    }
    scorecard = compute_multisystem_scorecard(rows)
    assert len(scorecard) == 2
    assert {item["readout"] for item in scorecard} == {"native", "common_rerank"}
    assert {item["status"] for item in scorecard} == {"complete"}
    assert {item["baseline_stability_rate"] for item in scorecard} == {1.0}
    assert {item["unstable_intervention_rate"] for item in scorecard} == {0.0}
