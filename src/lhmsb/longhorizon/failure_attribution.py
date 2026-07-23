"""Decision-aligned attribution for the memory-to-action chain.

The first missing stage is assigned on one fixed continuation decision.  Stage
yields are conditional: a state can enter the retrieval denominator only after
it was stored, and can enter the exposure denominator only after retrieval.
Behavioral use remains a conservative counterfactual label supplied by the
intervention pipeline; visibility alone never counts as use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DecisionFailureStage = Literal[
    "no_memory_channel",
    "not_memory_reliant",
    "storage_evidence_unavailable",
    "storage_failure",
    "retrieval_failure",
    "exposure_failure",
    "utilization_failure",
    "behavior_success_causal",
    "behavior_success_without_detected_unique_causal_effect",
    # Legacy schema value accepted when reading completed reports.
    "behavior_success_without_detected_use",
    "behavior_success_unprobed",
]
UseEvidenceStatus = Literal[
    "not_applicable",
    "unprobed",
    "partially_probed",
    "no_unique_causal_effect_detected",
    # Legacy schema value accepted when reading completed reports.
    "no_causal_effect_detected",
    "causal_effect_detected",
]
DecisionLayerDiagnosis = Literal[
    "not_applicable",
    "visible_use_evidence_incomplete",
    "visible_without_detected_unique_causal_effect",
    # Legacy schema value accepted when reading completed reports.
    "visible_without_detected_use",
    "visible_causally_influential_but_wrong",
]
StorageEvidenceMode = Literal[
    "native/exact",
    "inferred",
    "mixed",
    "unavailable",
    "not_applicable",
]


@dataclass(frozen=True)
class DecisionMemoryAttribution:
    """One SCEU's state flow and earliest attributable failure stage."""

    stage: DecisionFailureStage
    required_state_ids: tuple[str, ...]
    stored_required_state_ids: tuple[str, ...]
    retrieved_stored_state_ids: tuple[str, ...]
    visible_retrieved_state_ids: tuple[str, ...]
    probed_visible_state_ids: tuple[str, ...]
    causally_used_probed_state_ids: tuple[str, ...]
    behavior_correct: bool
    storage_evidence_mode: StorageEvidenceMode

    @property
    def required_count(self) -> int:
        return len(self.required_state_ids)

    @property
    def stored_required_count(self) -> int:
        return len(self.stored_required_state_ids)

    @property
    def retrieved_stored_count(self) -> int:
        return len(self.retrieved_stored_state_ids)

    @property
    def visible_retrieved_count(self) -> int:
        return len(self.visible_retrieved_state_ids)

    @property
    def probed_visible_count(self) -> int:
        return len(self.probed_visible_state_ids)

    @property
    def causally_used_probed_count(self) -> int:
        return len(self.causally_used_probed_state_ids)

    @property
    def use_evidence_status(self) -> UseEvidenceStatus:
        """Classify intervention evidence without treating visibility as use."""

        if not self.visible_retrieved_state_ids:
            return "not_applicable"
        if not self.probed_visible_state_ids:
            return "unprobed"
        if self.causally_used_probed_state_ids:
            return "causal_effect_detected"
        if set(self.probed_visible_state_ids) != set(
            self.visible_retrieved_state_ids
        ):
            return "partially_probed"
        return "no_unique_causal_effect_detected"

    @property
    def decision_layer_diagnosis(self) -> DecisionLayerDiagnosis:
        """Subtype a visible-state behavioral failure by causal-use evidence."""

        if self.stage != "utilization_failure":
            return "not_applicable"
        if self.use_evidence_status == "causal_effect_detected":
            return "visible_causally_influential_but_wrong"
        if self.use_evidence_status == "no_unique_causal_effect_detected":
            return "visible_without_detected_unique_causal_effect"
        return "visible_use_evidence_incomplete"

    @property
    def causal_use_claim_boundary(self) -> str:
        """State exactly what the registered intervention can identify."""

        if self.use_evidence_status == "causal_effect_detected":
            return "repeat_stable_unique_causal_effect_detected"
        if self.use_evidence_status == "no_unique_causal_effect_detected":
            return (
                "no_unique_causal_effect_detected_redundant_or_compensated_"
                "use_not_excluded"
            )
        if self.use_evidence_status in {"unprobed", "partially_probed"}:
            return "causal_use_not_fully_identified"
        return "not_applicable"

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "use_evidence_status": self.use_evidence_status,
            "decision_layer_diagnosis": self.decision_layer_diagnosis,
            "causal_use_claim_boundary": self.causal_use_claim_boundary,
        }


def attribute_decision_memory(
    *,
    memory_reliant_state_ids: tuple[str, ...],
    stored_state_ids: tuple[str, ...],
    retrieved_state_ids: tuple[str, ...],
    visible_state_ids: tuple[str, ...],
    probed_state_ids: tuple[str, ...],
    causally_used_state_ids: tuple[str, ...],
    behavior_correct: bool,
    has_memory_channel: bool,
    storage_evidence_mode: StorageEvidenceMode = "unavailable",
) -> DecisionMemoryAttribution:
    """Return the earliest supported failure stage for one fixed decision."""

    required = set(memory_reliant_state_ids)
    stored = required.intersection(stored_state_ids)
    retrieved = stored.intersection(retrieved_state_ids)
    visible = retrieved.intersection(visible_state_ids)
    probed = visible.intersection(probed_state_ids)
    used = probed.intersection(causally_used_state_ids)

    if not has_memory_channel:
        stage: DecisionFailureStage = "no_memory_channel"
        storage_evidence_mode = "not_applicable"
    elif not required:
        stage = "not_memory_reliant"
        storage_evidence_mode = "not_applicable"
    elif storage_evidence_mode == "unavailable":
        # Absence of evidence is not evidence that the backend failed to write.
        # Keep this decision outside storage-failure denominators until either a
        # native lifecycle trace or an inventory-derived trace is available.
        stage = "storage_evidence_unavailable"
    elif stored != required:
        stage = "storage_failure"
    elif retrieved != stored:
        stage = "retrieval_failure"
    elif visible != retrieved:
        stage = "exposure_failure"
    elif not behavior_correct:
        stage = "utilization_failure"
    elif used:
        stage = "behavior_success_causal"
    elif probed:
        stage = "behavior_success_without_detected_unique_causal_effect"
    else:
        stage = "behavior_success_unprobed"

    return DecisionMemoryAttribution(
        stage=stage,
        required_state_ids=tuple(sorted(required)),
        stored_required_state_ids=tuple(sorted(stored)),
        retrieved_stored_state_ids=tuple(sorted(retrieved)),
        visible_retrieved_state_ids=tuple(sorted(visible)),
        probed_visible_state_ids=tuple(sorted(probed)),
        causally_used_probed_state_ids=tuple(sorted(used)),
        behavior_correct=behavior_correct,
        storage_evidence_mode=storage_evidence_mode,
    )


__all__ = [
    "DecisionFailureStage",
    "DecisionLayerDiagnosis",
    "DecisionMemoryAttribution",
    "StorageEvidenceMode",
    "UseEvidenceStatus",
    "attribute_decision_memory",
]
