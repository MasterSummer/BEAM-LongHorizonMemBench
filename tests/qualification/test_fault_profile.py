from __future__ import annotations

import pytest

from lhmsb.qualification.fault_profile import (
    FaultProfileAlignmentError,
    compute_fault_profile_divergence,
    fault_profile_divergence_markdown,
)


def _row(
    condition: str,
    *,
    stage: str,
    action: str = "safe_v2_offline",
    correct: bool = True,
    diagnosis: str = "not_applicable",
    required: tuple[str, ...] = ("C1", "P2"),
) -> dict[str, object]:
    return {
        "episode_id": "episode-1",
        "sceu_id": "sceu-1",
        "opportunity_id": "opp-1",
        "result_id": f"result-{condition}",
        "policy_profile_id": "gpt-test",
        "condition": condition,
        "readout": "common_rerank",
        "checkpoint_session": 15,
        "current_state_signature": "state-v2",
        "selected_action_id": action,
        "behavior_correct": correct,
        "stage": stage,
        "decision_layer_diagnosis": diagnosis,
        "required_state_ids": list(required),
    }


def test_same_action_can_hide_different_success_profiles() -> None:
    payload = compute_fault_profile_divergence(
        (
            _row("mem0", stage="behavior_success_causal"),
            _row("amem", stage="behavior_success_without_detected_use"),
        )
    )

    assert payload["n_aligned_decision_pairs"] == 1
    assert payload["n_outcome_equivalent_pairs"] == 1
    assert payload["n_outcome_equivalent_fault_profile_divergences"] == 1
    assert payload["outcome_equivalent_fault_profile_divergence_rate"] == 1.0
    comparison = payload["comparisons"][0]  # type: ignore[index]
    assert comparison["same_selected_action"] is True
    assert comparison["fault_profile_diverged"] is True
    assert "identical" in str(payload["interpretation"])
    assert "Outcome-equivalent" in fault_profile_divergence_markdown(payload)


def test_same_wrong_action_separates_retrieval_from_utilization() -> None:
    payload = compute_fault_profile_divergence(
        (
            _row(
                "mem0",
                stage="retrieval_failure",
                action="stale_v1",
                correct=False,
            ),
            _row(
                "memos",
                stage="utilization_failure",
                action="stale_v1",
                correct=False,
                diagnosis="visible_causally_influential_but_wrong",
            ),
        )
    )

    assert payload["n_same_incorrect_action_pairs"] == 1
    assert payload[
        "same_incorrect_action_fault_profile_divergence_rate"
    ] == 1.0
    comparison = payload["comparisons"][0]  # type: ignore[index]
    assert comparison["diagnostic_label_b"] == (
        "utilization_failure:visible_causally_influential_but_wrong"
    )


def test_utilization_subtypes_are_distinct_profiles() -> None:
    payload = compute_fault_profile_divergence(
        (
            _row(
                "mem0",
                stage="utilization_failure",
                action="stale_v1",
                correct=False,
                diagnosis="visible_without_detected_use",
            ),
            _row(
                "amem",
                stage="utilization_failure",
                action="stale_v1",
                correct=False,
                diagnosis="visible_causally_influential_but_wrong",
            ),
        )
    )

    comparison = payload["comparisons"][0]  # type: ignore[index]
    assert comparison["earliest_stage_diverged"] is False
    assert comparison["fault_profile_diverged"] is True


def test_unavailable_storage_evidence_is_not_compared() -> None:
    payload = compute_fault_profile_divergence(
        (
            _row("mem0", stage="storage_evidence_unavailable"),
            _row("amem", stage="behavior_success_causal"),
        )
    )

    assert payload["n_aligned_decision_pairs"] == 0
    assert payload["outcome_equivalent_fault_profile_divergence_rate"] is None


def test_pairing_rejects_different_required_state() -> None:
    with pytest.raises(FaultProfileAlignmentError, match="different required state"):
        compute_fault_profile_divergence(
            (
                _row("mem0", stage="behavior_success_causal"),
                _row(
                    "amem",
                    stage="behavior_success_causal",
                    required=("C1",),
                ),
            )
        )
