from __future__ import annotations

from lhmsb.qualification.metrics import (
    BehaviorMetricInput,
    RetrievalMetricInput,
    StateCheckpointMetricInput,
    UsageMetricInput,
    compute_metric_collection,
    safe_ratio,
)


def test_safe_ratio_keeps_undefined_denominators_nullable() -> None:
    defined = safe_ratio(3, 4)
    assert defined.numerator == 3
    assert defined.denominator == 4
    assert defined.value == 0.75
    undefined = safe_ratio(0, 0)
    assert undefined.numerator == 0
    assert undefined.denominator == 0
    assert undefined.value is None


def test_write_state_maintenance_formulas_are_hand_computed() -> None:
    metrics = compute_metric_collection(
        state_checkpoints=(
            StateCheckpointMetricInput(
                eligible_write_state_ids=("A", "B"),
                new_memory_state_ids=(("A",), (), ("X",)),
                current_state_ids=("A", "B"),
                future_needed_state_ids=("A", "B"),
                retired_state_ids=("X", "Y"),
                live_memory_state_ids=(("A",), ("A",), ("X",), ()),
                live_content_hashes=("ha", "ha2", "hx", "unknown"),
                n_write=3,
                n_live=4,
            ),
        ),
    )
    assert metrics["write_coverage"].value == 0.5
    assert metrics["write_selectivity"].value == 1 / 3
    assert metrics["current_state_storage_precision"].value == 0.5
    assert metrics["current_state_storage_recall"].value == 0.5
    assert metrics["current_state_storage_f1"].value == 0.5
    assert metrics["stale_state_retention_rate"].value == 0.25
    assert metrics["duplicate_live_memory_rate"].value == 1 / 3
    assert metrics["update_delete_responsiveness"].value == 0.5
    assert metrics["write_to_continuation_alignment"].value == 0.5
    assert metrics["memory_write_count"].value == 3
    assert metrics["live_memory_count"].value == 4


def test_retrieval_visibility_and_stale_formulas_are_hand_computed() -> None:
    metrics = compute_metric_collection(
        retrievals=(
            RetrievalMetricInput(
                required_state_ids=("A", "B"),
                stale_state_ids=("X",),
                candidate_memory_state_ids=(("A",), ("X",), ()),
                retrieved_memory_state_ids=(("A",), ("X",)),
                visible_memory_state_ids=(("A",),),
                candidate_shortfall=True,
                retrieval_latency_seconds=2.0,
                rerank_latency_seconds=1.0,
            ),
        ),
    )
    assert metrics["candidate_recall"].value == 0.5
    assert metrics["retrieval_precision"].value == 0.5
    assert metrics["retrieval_recall"].value == 0.5
    assert metrics["retrieval_f1"].value == 0.5
    assert metrics["retrieval_false_positive_rate"].value == 0.5
    assert metrics["retrieval_timeliness"].value == 0.5
    assert metrics["candidate_shortfall_rate"].value == 1.0
    assert metrics["visible_sufficiency"].value == 0.5
    assert metrics["visible_contamination"].value == 0.0
    assert metrics["stale_retrieval_rate"].value == 0.5
    assert metrics["retrieved_but_not_visible_rate"].value == 0.5
    assert metrics["mean_retrieval_latency_seconds"].value == 2.0
    assert metrics["mean_rerank_latency_seconds"].value == 1.0


def test_causal_use_drift_behavior_and_reliability_formulas() -> None:
    metrics = compute_metric_collection(
        behaviors=(
            BehaviorMetricInput(
                policy_profile_id="p1",
                condition="mem0_controlled",
                readout="native",
                result_id="r1",
                behavior_score=0.25,
                is_correct=False,
                visible_memory_count=2,
                causal_labels=("beneficial", "visible_not_causally_used"),
                intervention_labels=("beneficial", "visible_not_causally_used"),
                leave_one_out_count=2,
                leave_one_out_action_flips=1,
                drift_flags=("constraint_loss", "plan_deviation"),
                matched_group="matched",
                checkpoint_session=10,
            ),
        ),
        usages=(
            UsageMetricInput(
                input_tokens=10,
                output_tokens=2,
                cached_tokens=3,
                reasoning_tokens=1,
                latency_seconds=4.0,
                retry_count=1,
                terminal_failure=False,
            ),
            UsageMetricInput(
                input_tokens=20,
                output_tokens=4,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=2.0,
                retry_count=0,
                terminal_failure=True,
            ),
        ),
    )
    assert metrics["causal_memory_use_rate"].value == 0.5
    assert metrics["visible_but_not_causally_used_rate"].value == 0.5
    assert metrics["beneficial_intervention_rate"].value == 0.5
    assert metrics["harmful_intervention_rate"].value == 0.0
    assert metrics["ambiguous_intervention_rate"].value == 0.0
    assert metrics["leave_one_memory_out_action_flip_rate"].value == 0.5
    assert metrics["constraint_loss_rate"].value == 1.0
    assert metrics["current_plan_deviation_rate"].value == 1.0
    assert metrics["stale_state_action_rate"].value == 0.0
    assert metrics["local_over_global_rate"].value == 0.0
    assert metrics["aggregate_drift_rate"].value == 1.0
    assert metrics["mean_behavior_score"].value == 0.25
    assert metrics["behavior_correct_rate"].value == 0.0
    assert metrics["policy_input_tokens"].value == 30
    assert metrics["policy_output_tokens"].value == 6
    assert metrics["policy_cached_tokens"].value == 3
    assert metrics["policy_reasoning_tokens"].value == 1
    assert metrics["mean_policy_latency_seconds"].value == 3.0
    assert metrics["policy_retry_rate"].value == 0.5
    assert metrics["terminal_failure_rate"].value == 0.5


def test_cost_metrics_separate_policy_internal_embedding_and_reranking() -> None:
    metrics = compute_metric_collection(
        usages=(
            UsageMetricInput(
                input_tokens=10,
                output_tokens=2,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=4.0,
                retry_count=0,
                terminal_failure=False,
                component="policy",
                input_count=1,
                usage_observed=True,
            ),
            UsageMetricInput(
                input_tokens=6,
                output_tokens=1,
                cached_tokens=2,
                reasoning_tokens=3,
                latency_seconds=2.0,
                retry_count=0,
                terminal_failure=False,
                component="memory_internal_llm",
                input_count=1,
                usage_observed=True,
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.5,
                retry_count=0,
                terminal_failure=False,
                component="embedding",
                input_count=4,
                usage_observed=False,
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.25,
                retry_count=0,
                terminal_failure=False,
                component="reranker",
                input_count=20,
                usage_observed=False,
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.0,
                retry_count=0,
                terminal_failure=False,
                component="qdrant_store",
                input_count=4096,
                usage_observed=True,
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.0,
                retry_count=0,
                terminal_failure=False,
                component="history_store",
                input_count=1024,
                usage_observed=True,
            ),
        )
    )

    assert metrics["policy_input_tokens"].value == 10
    assert metrics["memory_internal_input_tokens"].value == 6
    assert metrics["memory_internal_cached_tokens"].value == 2
    assert metrics["memory_internal_reasoning_tokens"].value == 3
    assert metrics["memory_internal_usage_observed_rate"].value == 1.0
    assert metrics["embedding_call_count"].value == 1
    assert metrics["embedding_input_count"].value == 4
    assert metrics["reranker_call_count"].value == 1
    assert metrics["reranker_candidate_pairs"].value == 20
    assert metrics["qdrant_store_bytes"].value == 4096
    assert metrics["history_store_bytes"].value == 1024


def test_store_footprints_do_not_dilute_terminal_failure_rate() -> None:
    metrics = compute_metric_collection(
        usages=(
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=1.0,
                retry_count=0,
                terminal_failure=True,
                component="policy",
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.0,
                retry_count=0,
                terminal_failure=False,
                component="qdrant_store",
                input_count=4096,
            ),
            UsageMetricInput(
                input_tokens=None,
                output_tokens=None,
                cached_tokens=None,
                reasoning_tokens=None,
                latency_seconds=0.0,
                retry_count=0,
                terminal_failure=False,
                component="history_store",
                input_count=1024,
            ),
        )
    )

    assert metrics["terminal_failure_rate"].value == 1.0


def test_workspace_gain_oracle_gap_and_common_rerank_delta() -> None:
    behaviors = (
        BehaviorMetricInput(
            policy_profile_id="p1",
            condition="workspace_only",
            readout="none",
            result_id="ws",
            behavior_score=0.2,
            is_correct=False,
        ),
        BehaviorMetricInput(
            policy_profile_id="p1",
            condition="oracle_current_state",
            readout="none",
            result_id="oracle",
            behavior_score=1.0,
            is_correct=True,
        ),
        BehaviorMetricInput(
            policy_profile_id="p1",
            condition="mem0_controlled",
            readout="native",
            result_id="native",
            behavior_score=0.6,
            is_correct=True,
        ),
        BehaviorMetricInput(
            policy_profile_id="p1",
            condition="mem0_controlled",
            readout="common_rerank",
            result_id="common",
            behavior_score=0.8,
            is_correct=True,
        ),
    )
    metrics = compute_metric_collection(behaviors=behaviors)
    assert metrics["mem0_gain_beyond_workspace"].value == 0.5
    assert metrics["oracle_gap_closed"].value == 0.625
    assert metrics["common_rerank_behavior_delta"].value == 0.2
