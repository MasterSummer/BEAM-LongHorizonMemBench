from __future__ import annotations

import pytest

from lhmsb.longhorizon.failure_attribution import attribute_decision_memory


@pytest.mark.parametrize(
    ("stored", "retrieved", "visible", "correct", "expected"),
    (
        ((), (), (), False, "storage_failure"),
        (("C1",), (), (), False, "retrieval_failure"),
        (("C1",), ("C1",), (), False, "exposure_failure"),
        (("C1",), ("C1",), ("C1",), False, "utilization_failure"),
    ),
)
def test_first_missing_stage_is_attributed_conditionally(
    stored: tuple[str, ...],
    retrieved: tuple[str, ...],
    visible: tuple[str, ...],
    correct: bool,
    expected: str,
) -> None:
    result = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=stored,
        retrieved_state_ids=retrieved,
        visible_state_ids=visible,
        probed_state_ids=visible,
        causally_used_state_ids=(),
        behavior_correct=correct,
        has_memory_channel=True,
        storage_evidence_mode="inferred",
    )

    assert result.stage == expected
    if expected == "utilization_failure":
        assert result.use_evidence_status == "no_unique_causal_effect_detected"
        assert result.decision_layer_diagnosis == (
            "visible_without_detected_unique_causal_effect"
        )


def test_causal_success_requires_a_probed_visible_state() -> None:
    causal = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=("C1", "noise"),
        retrieved_state_ids=("C1", "noise"),
        visible_state_ids=("C1",),
        probed_state_ids=("C1",),
        causally_used_state_ids=("C1",),
        behavior_correct=True,
        has_memory_channel=True,
        storage_evidence_mode="inferred",
    )
    unprobed = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=("C1",),
        retrieved_state_ids=("C1",),
        visible_state_ids=("C1",),
        probed_state_ids=(),
        causally_used_state_ids=("C1",),
        behavior_correct=True,
        has_memory_channel=True,
        storage_evidence_mode="inferred",
    )

    assert causal.stage == "behavior_success_causal"
    assert causal.causally_used_probed_state_ids == ("C1",)
    assert unprobed.stage == "behavior_success_unprobed"
    assert unprobed.causally_used_probed_state_ids == ()


def test_no_unique_effect_is_not_mislabeled_as_proof_of_nonuse() -> None:
    result = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=("C1",),
        retrieved_state_ids=("C1",),
        visible_state_ids=("C1",),
        probed_state_ids=("C1",),
        causally_used_state_ids=(),
        behavior_correct=True,
        has_memory_channel=True,
        storage_evidence_mode="inferred",
    )

    assert result.stage == (
        "behavior_success_without_detected_unique_causal_effect"
    )
    assert result.probed_visible_count == 1
    assert result.causally_used_probed_count == 0
    assert result.causal_use_claim_boundary == (
        "no_unique_causal_effect_detected_redundant_or_compensated_"
        "use_not_excluded"
    )


def test_visible_wrong_behavior_separates_nonuse_from_causal_misuse() -> None:
    common = {
        "memory_reliant_state_ids": ("C1",),
        "stored_state_ids": ("C1",),
        "retrieved_state_ids": ("C1",),
        "visible_state_ids": ("C1",),
        "probed_state_ids": ("C1",),
        "behavior_correct": False,
        "has_memory_channel": True,
        "storage_evidence_mode": "native/exact",
    }
    no_effect = attribute_decision_memory(
        **common,
        causally_used_state_ids=(),
    )
    causal_wrong = attribute_decision_memory(
        **common,
        causally_used_state_ids=("C1",),
    )
    unprobed = attribute_decision_memory(
        **{
            **common,
            "probed_state_ids": (),
        },
        causally_used_state_ids=(),
    )

    assert no_effect.decision_layer_diagnosis == (
        "visible_without_detected_unique_causal_effect"
    )
    assert (
        causal_wrong.decision_layer_diagnosis
        == "visible_causally_influential_but_wrong"
    )
    assert unprobed.decision_layer_diagnosis == "visible_use_evidence_incomplete"
    assert causal_wrong.to_dict()["use_evidence_status"] == "causal_effect_detected"
    assert causal_wrong.to_dict()["causal_use_claim_boundary"] == (
        "repeat_stable_unique_causal_effect_detected"
    )


def test_controls_and_workspace_sufficient_decisions_are_explicitly_not_applicable() -> None:
    no_channel = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=(),
        retrieved_state_ids=(),
        visible_state_ids=(),
        probed_state_ids=(),
        causally_used_state_ids=(),
        behavior_correct=False,
        has_memory_channel=False,
    )
    workspace_sufficient = attribute_decision_memory(
        memory_reliant_state_ids=(),
        stored_state_ids=(),
        retrieved_state_ids=(),
        visible_state_ids=(),
        probed_state_ids=(),
        causally_used_state_ids=(),
        behavior_correct=True,
        has_memory_channel=True,
    )

    assert no_channel.stage == "no_memory_channel"
    assert workspace_sufficient.stage == "not_memory_reliant"


def test_missing_storage_evidence_is_not_mislabelled_as_storage_failure() -> None:
    result = attribute_decision_memory(
        memory_reliant_state_ids=("C1",),
        stored_state_ids=(),
        retrieved_state_ids=(),
        visible_state_ids=(),
        probed_state_ids=(),
        causally_used_state_ids=(),
        behavior_correct=False,
        has_memory_channel=True,
        storage_evidence_mode="unavailable",
    )

    assert result.stage == "storage_evidence_unavailable"
