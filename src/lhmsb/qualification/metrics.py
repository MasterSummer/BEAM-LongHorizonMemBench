"""Denominator-safe lifecycle, retrieval, causal-use, and drift metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.attribution import (
    MemoryAttribution,
    ProvenanceMode,
    attribute_memory,
    build_software_fact_signatures,
    eligible_write_state_ids,
    is_benchmark_state_id,
)
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.failure_attribution import (
    DecisionMemoryAttribution,
    StorageEvidenceMode,
    attribute_decision_memory,
)
from lhmsb.longhorizon.replay import ReplayResult, replay_plan
from lhmsb.longhorizon.schema import ContinuationOpportunity
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.drift import (
    DRIFT_LINEAGE_EVIDENCE_MODE,
    drift_lineage_pairs,
)
from lhmsb.qualification.memory_runtime import InventorySnapshot, WriteSessionResult
from lhmsb.qualification.prefix import MemoryPrefixArtifact, MemoryPrefixCheckpoint
from lhmsb.qualification.runner import (
    PolicyEvaluation,
    QualificationMatrixResult,
)

_CANONICAL_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)
_MEMORY_CONDITIONS = {
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
    "mem0_controlled",
    "mem0_native",
}


def _is_memory_count_contrast(value: object) -> bool:
    """Return whether a serialized contrast changes memory-object count.

    ``add_one`` is retained for schema-v1 result compatibility.  Schema-v2
    count-load probes use explicit ``add_<N>`` labels so reports can recover
    the pre-registered object-count level without consulting call IDs.
    """

    label = str(value)
    if label == "delete_one":
        return True
    if label == "add_one":
        return True
    if not label.startswith("add_"):
        return False
    suffix = label.removeprefix("add_")
    return suffix.isdigit() and int(suffix) > 0


def _is_memory_count_load_contrast(value: object) -> bool:
    """Return whether a contrast adds evaluator-controlled neutral objects.

    Targeted ``delete_one`` probes change the visible object count, but they
    also remove a specific potentially causal memory. They therefore belong
    to leave-one-out causal-use metrics, not to the RQ5 count-load estimate.
    """

    label = str(value)
    if label == "add_one":
        return True
    if not label.startswith("add_"):
        return False
    suffix = label.removeprefix("add_")
    return suffix.isdigit() and int(suffix) > 0


@dataclass(frozen=True)
class MetricValue:
    numerator: float
    denominator: float
    value: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class MetricCollection:
    metrics: tuple[tuple[str, MetricValue], ...]

    def __getitem__(self, name: str) -> MetricValue:
        try:
            return dict(self.metrics)[name]
        except KeyError as exc:
            raise KeyError(f"unknown qualification metric: {name}") from exc

    def to_dict(self) -> dict[str, dict[str, float | None]]:
        return {
            name: metric.to_dict()
            for name, metric in self.metrics
        }


@dataclass(frozen=True)
class StateCheckpointMetricInput:
    eligible_write_state_ids: tuple[str, ...]
    new_memory_state_ids: tuple[tuple[str, ...], ...]
    current_state_ids: tuple[str, ...]
    future_needed_state_ids: tuple[str, ...]
    retired_state_ids: tuple[str, ...]
    live_memory_state_ids: tuple[tuple[str, ...], ...]
    live_content_hashes: tuple[str, ...]
    n_write: int
    n_live: int
    write_latency_seconds: float = 0.0
    is_final_checkpoint: bool = True
    # Native lifecycle event counts are optional so schema-v1 callers keep their
    # exact constructor and serialization semantics.  Schema-v2 adapters fill
    # these from the immutable mutation trace.
    mutation_counts: tuple[tuple[str, int], ...] = ()
    # Per-object lifecycle provenance.  These arrays align with
    # ``new_memory_state_ids`` and ``live_memory_state_ids`` respectively.
    # Keeping them on the immutable checkpoint record lets aggregation report
    # exact and inferred tracks without guessing from a final inventory.
    new_memory_provenance: tuple[str, ...] = ()
    live_memory_provenance: tuple[str, ...] = ()
    # Lifecycle provenance and semantic attribution are orthogonal.  A native
    # add/update event can still contain text that maps ambiguously to latent
    # state, so retain both object-aligned dimensions.
    new_memory_attribution_methods: tuple[str, ...] = ()
    live_memory_attribution_methods: tuple[str, ...] = ()
    provenance_complete: bool = True
    # One tuple per sorted retired state. Each child tuple lists current states
    # that supersede the retired state in the same kind and scope. Retaining an
    # audit copy is responsive when the active replacement is also represented.
    retired_replacement_state_ids: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RetrievalMetricInput:
    required_state_ids: tuple[str, ...]
    stale_state_ids: tuple[str, ...]
    candidate_memory_state_ids: tuple[tuple[str, ...], ...]
    retrieved_memory_state_ids: tuple[tuple[str, ...], ...]
    visible_memory_state_ids: tuple[tuple[str, ...], ...]
    candidate_shortfall: bool
    retrieval_latency_seconds: float
    rerank_latency_seconds: float | None = None
    backend_retrieved_memory_state_ids: tuple[tuple[str, ...], ...] | None = None
    selected_memory_state_ids: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class BehaviorMetricInput:
    policy_profile_id: str
    condition: str
    readout: str
    result_id: str
    behavior_score: float
    is_correct: bool
    visible_memory_count: int = 0
    causal_labels: tuple[str, ...] = ()
    intervention_labels: tuple[str, ...] = ()
    leave_one_out_count: int = 0
    leave_one_out_action_flips: int = 0
    drift_flags: tuple[str, ...] = ()
    matched_group: str = ""
    checkpoint_session: int = 0
    is_conflict_opportunity: bool = False
    # ``visible_memory_count`` is an exposure variable.  The primary scaling
    # variable is the total live native-object count at the checkpoint.
    live_memory_count: int | None = None
    count_contrast_count: int = 0
    count_contrast_action_flips: int = 0
    count_contrast_behavior_changes: int = 0
    sham_replacement_count: int = 0
    sham_replacement_action_flips: int = 0
    behaviorally_used_memory_count: int = 0
    behavioral_use_probe_count: int = 0
    drift_eligible_categories: tuple[str, ...] | None = None
    current_state_signature: str = ""


@dataclass(frozen=True)
class MultisystemMetricInput:
    """One backend-neutral scored SCEU observation.

    This DTO deliberately contains only evaluator-side attribution and the
    model-visible trace.  It can be built from either the schema-v2 evaluator
    result or the legacy runner result, allowing aggregation to stay independent
    of a particular memory backend.
    """

    policy_profile_id: str
    condition: str
    readout: str
    result_id: str
    behavior_score: float
    is_correct: bool
    required_state_ids: tuple[str, ...] = ()
    stale_state_ids: tuple[str, ...] = ()
    candidate_memory_state_ids: tuple[tuple[str, ...], ...] = ()
    retrieved_memory_state_ids: tuple[tuple[str, ...], ...] = ()
    visible_memory_state_ids: tuple[tuple[str, ...], ...] = ()
    candidate_shortfall: bool = False
    visible_memory_count: int = 0
    live_memory_count: int | None = None
    causal_labels: tuple[str, ...] = ()
    intervention_labels: tuple[str, ...] = ()
    leave_one_out_count: int = 0
    leave_one_out_action_flips: int = 0
    drift_flags: tuple[str, ...] = ()
    matched_group: str = ""
    checkpoint_session: int = 0
    is_conflict_opportunity: bool = False
    count_contrast_count: int = 0
    count_contrast_action_flips: int = 0
    count_contrast_behavior_changes: int = 0
    retrieval_latency_seconds: float = 0.0
    rerank_latency_seconds: float | None = None
    status: str = "complete"
    baseline_stable: bool = True
    backend_retrieved_memory_state_ids: tuple[tuple[str, ...], ...] | None = None
    selected_memory_state_ids: tuple[tuple[str, ...], ...] | None = None
    behaviorally_used_memory_ids: tuple[str, ...] = ()
    behavioral_use_probe_count: int = 0
    sham_replacement_count: int = 0
    sham_replacement_action_flips: int = 0
    drift_eligible_categories: tuple[str, ...] | None = None
    drift_lineage_pairs: tuple[tuple[str, str], ...] = ()
    drift_lineage_evidence_mode: str = "unavailable"
    current_state_signature: str = ""
    episode_id: str = ""
    sceu_id: str = ""
    opportunity_id: str = ""
    selected_action_id: str = ""
    control_kind: str = ""
    construct_kind: str = ""
    horizon_band: str = ""
    handoff_count: int = 0
    oldest_required_state_age: int | None = None
    latest_decision_event_distance: int | None = None
    dependency_depth: int = 0
    relevant_transition_count: int = 0
    memory_reliant_state_ids: tuple[str, ...] = ()
    nonexplicit_state_ids: tuple[str, ...] = ()
    stored_memory_state_ids: tuple[str, ...] = ()
    stored_exact_state_ids: tuple[str, ...] = ()
    stored_inferred_state_ids: tuple[str, ...] = ()
    stored_unavailable_state_ids: tuple[str, ...] = ()
    storage_evidence_mode: StorageEvidenceMode = "unavailable"
    behaviorally_probed_state_ids: tuple[str, ...] = ()
    behaviorally_used_state_ids: tuple[str, ...] = ()
    counterfactual_group_id: str = ""
    counterfactual_variant: str = ""
    counterfactual_terminal_archetype: str = ""
    is_counterfactual_target: bool = False
    horizon_panel_id: str = ""
    horizon_level: str = ""
    horizon_axis: str = ""
    effective_task_step_count: int = 0
    max_task_dependency_depth: int = 0
    causally_linked_task_step_fraction: float | None = None

    def decision_attribution(self) -> DecisionMemoryAttribution:
        """Return the first supported memory-to-action stage for this SCEU."""
        # Retrieval is the backend-returned set.  Any later native truncation,
        # common reranking, or prompt-budget filtering belongs to the exposure
        # stage because the model never received the filtered object.
        backend_retrieved = (
            self.retrieved_memory_state_ids
            if self.backend_retrieved_memory_state_ids is None
            else self.backend_retrieved_memory_state_ids
        )
        return attribute_decision_memory(
            memory_reliant_state_ids=self.memory_reliant_state_ids,
            stored_state_ids=self.stored_memory_state_ids,
            retrieved_state_ids=_flatten_state_ids(backend_retrieved),
            visible_state_ids=_flatten_state_ids(self.visible_memory_state_ids),
            probed_state_ids=self.behaviorally_probed_state_ids,
            causally_used_state_ids=self.behaviorally_used_state_ids,
            behavior_correct=self.is_correct,
            has_memory_channel=(
                self.condition in _MEMORY_CONDITIONS and self.readout != "none"
            ),
            storage_evidence_mode=self.storage_evidence_mode,
        )

    def behavior_input(self) -> BehaviorMetricInput:
        return BehaviorMetricInput(
            policy_profile_id=self.policy_profile_id,
            condition=self.condition,
            readout=self.readout,
            result_id=self.result_id,
            behavior_score=self.behavior_score,
            is_correct=self.is_correct,
            visible_memory_count=self.visible_memory_count,
            live_memory_count=self.live_memory_count,
            causal_labels=self.causal_labels,
            intervention_labels=self.intervention_labels,
            leave_one_out_count=self.leave_one_out_count,
            leave_one_out_action_flips=self.leave_one_out_action_flips,
            drift_flags=self.drift_flags,
            matched_group=self.matched_group,
            checkpoint_session=self.checkpoint_session,
            is_conflict_opportunity=self.is_conflict_opportunity,
            count_contrast_count=self.count_contrast_count,
            count_contrast_action_flips=self.count_contrast_action_flips,
            count_contrast_behavior_changes=self.count_contrast_behavior_changes,
            sham_replacement_count=self.sham_replacement_count,
            sham_replacement_action_flips=self.sham_replacement_action_flips,
            behaviorally_used_memory_count=len(
                self.behaviorally_used_memory_ids
            ),
            behavioral_use_probe_count=self.behavioral_use_probe_count,
            drift_eligible_categories=self.drift_eligible_categories,
            current_state_signature=self.current_state_signature,
        )

    def retrieval_input(self) -> RetrievalMetricInput:
        return RetrievalMetricInput(
            required_state_ids=self.required_state_ids,
            stale_state_ids=self.stale_state_ids,
            candidate_memory_state_ids=self.candidate_memory_state_ids,
            retrieved_memory_state_ids=self.retrieved_memory_state_ids,
            visible_memory_state_ids=self.visible_memory_state_ids,
            candidate_shortfall=self.candidate_shortfall,
            retrieval_latency_seconds=self.retrieval_latency_seconds,
            rerank_latency_seconds=self.rerank_latency_seconds,
            backend_retrieved_memory_state_ids=(
                self.backend_retrieved_memory_state_ids
            ),
            selected_memory_state_ids=self.selected_memory_state_ids,
        )


@dataclass(frozen=True)
class UsageMetricInput:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    latency_seconds: float
    retry_count: int
    terminal_failure: bool
    component: str = "policy"
    input_count: int = 1
    usage_observed: bool = True


def safe_ratio(numerator: float, denominator: float) -> MetricValue:
    """Preserve undefined ratios as null instead of coercing them to zero."""
    return MetricValue(
        numerator=float(numerator),
        denominator=float(denominator),
        value=None if denominator == 0 else float(numerator) / float(denominator),
    )


def compute_metric_collection(
    *,
    state_checkpoints: Sequence[StateCheckpointMetricInput] = (),
    retrievals: Sequence[RetrievalMetricInput] = (),
    behaviors: Sequence[BehaviorMetricInput] = (),
    usages: Sequence[UsageMetricInput] = (),
) -> MetricCollection:
    """Compute the complete qualification metric namespace from flat records."""
    values: dict[str, MetricValue] = {}
    _state_metrics(values, state_checkpoints)
    # Provenance is a first-class denominator.  A backend with only inferred
    # inventory diffs must not be silently reported as exact storage quality.
    for mode, prefixes in (
        ("native/exact", ("storage_native_event_", "storage_exact_")),
        (
            "inferred",
            ("storage_inventory_inferred_", "storage_inferred_"),
        ),
    ):
        projected = _filter_state_observations(state_checkpoints, mode)
        if projected:
            for prefix in prefixes:
                _state_metrics(values, projected, prefix=prefix)
    _retrieval_metrics(values, retrievals)
    _behavior_metrics(values, behaviors)
    _usage_metrics(values, usages)
    return MetricCollection(metrics=tuple(sorted(values.items())))


def compute_multisystem_metrics(
    observations: Sequence[MultisystemMetricInput],
    *,
    state_checkpoints: Sequence[StateCheckpointMetricInput] = (),
    usages: Sequence[UsageMetricInput] = (),
) -> MetricCollection:
    """Compute metrics for the schema-v2 seven-condition matrix.

    ``compute_metric_collection`` remains the low-level API used by schema-v1.
    This adapter makes the common evaluator output convenient to aggregate and
    ensures that retrieval metrics are only included for conditions that expose
    a memory readout.  Empty controls therefore retain denominator-safe nulls.
    """
    behaviors = tuple(item.behavior_input() for item in observations)
    retrievals = tuple(
        item.retrieval_input()
        for item in observations
        if item.condition in {"flat_retrieval", "mem0", "amem", "memos",
                              "mem0_controlled", "mem0_native"}
        and item.readout != "none"
    )
    return compute_metric_collection(
        state_checkpoints=state_checkpoints,
        retrievals=retrievals,
        behaviors=behaviors,
        usages=usages,
    )


def compute_multisystem_metrics_by_cell(
    observations: Sequence[MultisystemMetricInput],
    *,
    state_checkpoints_by_cell: Mapping[tuple[str, str, str],
                                       Sequence[StateCheckpointMetricInput]] | None = None,
    usages_by_cell: Mapping[tuple[str, str, str], Sequence[UsageMetricInput]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return deterministic, non-mixed metric groups for policy/readout cells.

    Native and ``common_rerank`` rows are intentionally separate groups.  The
    function does not invent missing cells, which is important for controlled
    tracks that deliberately omit native readouts.
    """
    grouped: dict[tuple[str, str, str], list[MultisystemMetricInput]] = defaultdict(list)
    for item in observations:
        grouped[(item.policy_profile_id, item.condition, item.readout)].append(item)
    output: list[dict[str, object]] = []
    state_map = state_checkpoints_by_cell or {}
    usage_map = usages_by_cell or {}
    for key in sorted(grouped):
        metrics = compute_multisystem_metrics(
            tuple(grouped[key]),
            state_checkpoints=state_map.get(key, ()),
            usages=usage_map.get(key, ()),
        )
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "metrics": metrics.to_dict(),
            }
        )
    return tuple(output)


def compute_multisystem_scorecard(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Build a compact scorecard without averaging readouts together."""
    grouped: dict[tuple[str, str, str], list[MultisystemMetricInput]] = defaultdict(list)
    for item in observations:
        grouped[(item.policy_profile_id, item.condition, item.readout)].append(item)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        labels = [label for row in rows for label in row.intervention_labels]
        causal = [label for row in rows for label in row.causal_labels]
        eligible_by_flag = {
            flag: [row for row in rows if _drift_is_eligible(row, flag)]
            for flag in (
                "constraint_loss",
                "plan_deviation",
                "stale_state",
                "local_over_global",
            )
        }
        aggregate_eligible = [
            row for row in rows if _any_drift_is_eligible(row)
        ]
        observed_by_flag = {
            flag: _ratio_value(
                sum(flag in row.drift_flags for row in rows),
                len(rows),
            )
            for flag in _CANONICAL_DRIFT_CATEGORIES
        }
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "status": _aggregate_observation_status(
                    tuple(row.status for row in rows)
                ),
                "n_sceu": len(rows),
                "mean_behavior_score": _ratio_value(
                    sum(row.behavior_score for row in rows), len(rows)
                ),
                "behavior_correct_rate": _ratio_value(
                    sum(row.is_correct for row in rows), len(rows)
                ),
                "baseline_stability_rate": _ratio_value(
                    sum(row.baseline_stable for row in rows), len(rows)
                ),
                "mean_visible_memory_count": _ratio_value(
                    sum(row.visible_memory_count for row in rows), len(rows)
                ),
                "mean_live_memory_count": _ratio_value(
                    sum(row.live_memory_count for row in rows if row.live_memory_count is not None),
                    sum(row.live_memory_count is not None for row in rows),
                ),
                "causal_memory_use_rate": _ratio_value(
                    sum(
                        label
                        in {
                            "beneficial",
                            "harmful",
                            "causal_direction_ambiguous",
                        }
                        for label in causal
                    ),
                    len(causal),
                ),
                "unique_causal_effect_rate": _ratio_value(
                    sum(
                        label
                        in {
                            "beneficial",
                            "harmful",
                            "causal_direction_ambiguous",
                        }
                        for label in causal
                    ),
                    len(causal),
                ),
                "beneficial_intervention_rate": _ratio_value(
                    labels.count("beneficial"), len(labels)
                ),
                "harmful_intervention_rate": _ratio_value(
                    labels.count("harmful"), len(labels)
                ),
                "unstable_intervention_rate": _ratio_value(
                    sum(
                        label in {"unstable_baseline", "intervention_unstable"}
                        for label in labels
                    ),
                    len(labels),
                ),
                "sham_replacement_action_flip_rate": _ratio_value(
                    sum(row.sham_replacement_action_flips for row in rows),
                    sum(row.sham_replacement_count for row in rows),
                ),
                "behavioral_use_probe_coverage": _ratio_value(
                    sum(row.behavioral_use_probe_count for row in rows),
                    sum(row.visible_memory_count for row in rows),
                ),
                "probed_memory_causal_use_rate": _ratio_value(
                    sum(len(row.behaviorally_used_memory_ids) for row in rows),
                    sum(row.behavioral_use_probe_count for row in rows),
                ),
                "constraint_loss_rate": _eligible_flag_rate(
                    eligible_by_flag["constraint_loss"],
                    "constraint_loss",
                ),
                "constraint_loss_eligible_n": len(
                    eligible_by_flag["constraint_loss"]
                ),
                "targeted_constraint_loss_rate": _eligible_flag_rate(
                    eligible_by_flag["constraint_loss"],
                    "constraint_loss",
                ),
                "observed_constraint_loss_rate": observed_by_flag["constraint_loss"],
                "canonical_constraint_loss_violation_rate": observed_by_flag[
                    "constraint_loss"
                ],
                "current_plan_deviation_rate": _eligible_flag_rate(
                    eligible_by_flag["plan_deviation"],
                    "plan_deviation",
                ),
                "plan_deviation_eligible_n": len(
                    eligible_by_flag["plan_deviation"]
                ),
                "targeted_plan_deviation_rate": _eligible_flag_rate(
                    eligible_by_flag["plan_deviation"],
                    "plan_deviation",
                ),
                "observed_plan_deviation_rate": observed_by_flag["plan_deviation"],
                "canonical_plan_deviation_violation_rate": observed_by_flag[
                    "plan_deviation"
                ],
                "stale_state_action_rate": _eligible_flag_rate(
                    eligible_by_flag["stale_state"],
                    "stale_state",
                ),
                "stale_state_eligible_n": len(
                    eligible_by_flag["stale_state"]
                ),
                "targeted_stale_state_rate": _eligible_flag_rate(
                    eligible_by_flag["stale_state"],
                    "stale_state",
                ),
                "observed_stale_state_rate": observed_by_flag["stale_state"],
                "canonical_stale_state_violation_rate": observed_by_flag[
                    "stale_state"
                ],
                "local_over_global_rate": _eligible_flag_rate(
                    eligible_by_flag["local_over_global"],
                    "local_over_global",
                ),
                "local_over_global_eligible_n": len(
                    eligible_by_flag["local_over_global"]
                ),
                "targeted_local_over_global_rate": _eligible_flag_rate(
                    eligible_by_flag["local_over_global"],
                    "local_over_global",
                ),
                "observed_local_over_global_rate": observed_by_flag[
                    "local_over_global"
                ],
                "canonical_local_over_global_violation_rate": observed_by_flag[
                    "local_over_global"
                ],
                "aggregate_drift_rate": _ratio_value(
                    sum(_has_targeted_drift(row) for row in aggregate_eligible),
                    len(aggregate_eligible),
                ),
                "aggregate_drift_eligible_n": len(aggregate_eligible),
                "targeted_aggregate_drift_rate": _ratio_value(
                    sum(_has_targeted_drift(row) for row in aggregate_eligible),
                    len(aggregate_eligible),
                ),
                "observed_aggregate_drift_rate": _ratio_value(
                    sum(_has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
                "canonical_drift_violation_rate": _ratio_value(
                    sum(_has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
                "off_target_drift_rate": _ratio_value(
                    sum(_has_off_target_drift(row) for row in rows),
                    len(rows),
                ),
                "off_target_drift_n": sum(
                    _has_off_target_drift(row) for row in rows
                ),
                "memory_count_contrast_rate": _ratio_value(
                    sum(row.count_contrast_action_flips for row in rows),
                    sum(row.count_contrast_count for row in rows),
                ),
                "memory_count_behavior_change_rate": _ratio_value(
                    sum(row.count_contrast_behavior_changes for row in rows),
                    sum(row.count_contrast_count for row in rows),
                ),
            }
        )
    return tuple(output)


def compute_failure_attribution_scorecard(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Aggregate the same-decision memory funnel without mixing readouts."""
    grouped: dict[
        tuple[str, str, str, str], list[MultisystemMetricInput]
    ] = defaultdict(list)
    for item in observations:
        grouped[
            (
                item.policy_profile_id,
                item.condition,
                item.readout,
                item.storage_evidence_mode,
            )
        ].append(item)

    stage_names = (
        "storage_evidence_unavailable",
        "storage_failure",
        "retrieval_failure",
        "exposure_failure",
        "utilization_failure",
        "behavior_success_causal",
        "behavior_success_without_detected_unique_causal_effect",
        # Retained so a scorecard can ingest a completed legacy report.
        "behavior_success_without_detected_use",
        "behavior_success_unprobed",
    )
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        attributions = tuple(
            row.decision_attribution() for row in grouped[key]
        )
        counts: Counter[str] = Counter(item.stage for item in attributions)
        decision_layer_diagnoses: Counter[str] = Counter(
            item.decision_layer_diagnosis for item in attributions
        )
        memory_reliant = tuple(
            item
            for item in attributions
            if item.stage not in {"no_memory_channel", "not_memory_reliant"}
        )
        applicable = tuple(
            item
            for item in memory_reliant
            if item.stage != "storage_evidence_unavailable"
        )
        row: dict[str, object] = {
            "policy_profile_id": key[0],
            "condition": key[1],
            "readout": key[2],
            "storage_evidence_mode": key[3],
            "n_sceu": len(attributions),
            "memory_reliant_n": len(memory_reliant),
            "attribution_applicable_n": len(applicable),
            "no_memory_channel_n": counts["no_memory_channel"],
            "not_memory_reliant_n": counts["not_memory_reliant"],
            "memory_required_state_count": sum(
                item.required_count for item in applicable
            ),
            "memory_required_storage_recall": _ratio_value(
                sum(item.stored_required_count for item in applicable),
                sum(item.required_count for item in applicable),
            ),
            "stored_to_retrieved_yield": _ratio_value(
                sum(item.retrieved_stored_count for item in applicable),
                sum(item.stored_required_count for item in applicable),
            ),
            "retrieved_to_visible_yield": _ratio_value(
                sum(item.visible_retrieved_count for item in applicable),
                sum(item.retrieved_stored_count for item in applicable),
            ),
            "visible_required_probe_coverage": _ratio_value(
                sum(item.probed_visible_count for item in applicable),
                sum(item.visible_retrieved_count for item in applicable),
            ),
            "probed_required_causal_use_rate": _ratio_value(
                sum(item.causally_used_probed_count for item in applicable),
                sum(item.probed_visible_count for item in applicable),
            ),
        }
        for stage in stage_names:
            row[f"{stage}_n"] = counts[stage]
            row[f"{stage}_rate"] = _ratio_value(
                counts[stage],
                len(applicable),
            )
        for diagnosis in (
            "visible_without_detected_unique_causal_effect",
            # Retained so a scorecard can ingest a completed legacy report.
            "visible_without_detected_use",
            "visible_causally_influential_but_wrong",
            "visible_use_evidence_incomplete",
        ):
            row[f"{diagnosis}_n"] = decision_layer_diagnoses[diagnosis]
            row[f"{diagnosis}_rate"] = _ratio_value(
                decision_layer_diagnoses[diagnosis],
                counts["utilization_failure"],
            )
        output.append(row)
    return tuple(output)


def decision_attribution_rows(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Materialize the auditable first-failure record for every decision."""
    output: list[dict[str, object]] = []
    for row in observations:
        attribution = row.decision_attribution()
        output.append(
            {
                "episode_id": row.episode_id,
                "sceu_id": row.sceu_id,
                "opportunity_id": row.opportunity_id,
                "result_id": row.result_id,
                "policy_profile_id": row.policy_profile_id,
                "condition": row.condition,
                "readout": row.readout,
                "checkpoint_session": row.checkpoint_session,
                "current_state_signature": row.current_state_signature,
                "handoff_count": row.handoff_count,
                "construct_kind": row.construct_kind,
                "horizon_band": row.horizon_band,
                "counterfactual_group_id": row.counterfactual_group_id,
                "counterfactual_variant": row.counterfactual_variant,
                "counterfactual_terminal_archetype": (
                    row.counterfactual_terminal_archetype
                ),
                "is_counterfactual_target": row.is_counterfactual_target,
                "effective_task_step_count": row.effective_task_step_count,
                "max_task_dependency_depth": (
                    row.max_task_dependency_depth
                ),
                "selected_action_id": row.selected_action_id,
                "behavior_score": row.behavior_score,
                "behavior_correct": row.is_correct,
                "drift_flags": list(row.drift_flags),
                **attribution.to_dict(),
                "stored_exact_state_ids": list(row.stored_exact_state_ids),
                "stored_inferred_state_ids": list(
                    row.stored_inferred_state_ids
                ),
                "stored_unavailable_state_ids": list(
                    row.stored_unavailable_state_ids
                ),
            }
        )
    return tuple(output)


def compute_long_horizon_scorecard(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Report behavior and drift by explicit long-horizon construct strata."""
    grouped: dict[
        tuple[str, str, str, str, str], list[MultisystemMetricInput]
    ] = defaultdict(list)
    for item in observations:
        if not item.construct_kind or not item.horizon_band:
            continue
        grouped[
            (
                item.policy_profile_id,
                item.condition,
                item.readout,
                item.construct_kind,
                item.horizon_band,
            )
        ].append(item)

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        eligible = [row for row in rows if _any_drift_is_eligible(row)]
        ages = [
            row.oldest_required_state_age
            for row in rows
            if row.oldest_required_state_age is not None
        ]
        event_distances = [
            row.latest_decision_event_distance
            for row in rows
            if row.latest_decision_event_distance is not None
        ]
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "construct_kind": key[3],
                "horizon_band": key[4],
                "n_sceu": len(rows),
                "n_episodes": len({row.episode_id for row in rows if row.episode_id}),
                "mean_handoff_count": _ratio_value(
                    sum(row.handoff_count for row in rows), len(rows)
                ),
                "mean_oldest_required_state_age": _ratio_value(
                    sum(ages), len(ages)
                ),
                "mean_latest_decision_event_distance": _ratio_value(
                    sum(event_distances), len(event_distances)
                ),
                "mean_dependency_depth": _ratio_value(
                    sum(row.dependency_depth for row in rows), len(rows)
                ),
                "mean_relevant_transition_count": _ratio_value(
                    sum(row.relevant_transition_count for row in rows), len(rows)
                ),
                "mean_effective_task_step_count": _ratio_value(
                    sum(row.effective_task_step_count for row in rows),
                    len(rows),
                ),
                "mean_max_task_dependency_depth": _ratio_value(
                    sum(row.max_task_dependency_depth for row in rows),
                    len(rows),
                ),
                "mean_causally_linked_task_step_fraction": _mean_optional(
                    row.causally_linked_task_step_fraction for row in rows
                ),
                "mean_memory_reliant_state_count": _ratio_value(
                    sum(len(row.memory_reliant_state_ids) for row in rows),
                    len(rows),
                ),
                "mean_behavior_score": _ratio_value(
                    sum(row.behavior_score for row in rows), len(rows)
                ),
                "behavior_correct_rate": _ratio_value(
                    sum(row.is_correct for row in rows), len(rows)
                ),
                "targeted_drift_rate": _ratio_value(
                    sum(_has_targeted_drift(row) for row in eligible),
                    len(eligible),
                ),
                "targeted_drift_violation_rate": _ratio_value(
                    sum(_has_targeted_drift(row) for row in eligible),
                    len(eligible),
                ),
                "drift_eligible_n": len(eligible),
                "observed_drift_rate": _ratio_value(
                    sum(_has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
                "canonical_drift_violation_rate": _ratio_value(
                    sum(_has_canonical_drift_violation(row) for row in rows),
                    len(rows),
                ),
            }
        )
    return tuple(output)


def compute_long_horizon_control_contrasts(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Pair each system decision with workspace and oracle on the same SCEU."""
    control_rows: dict[
        tuple[str, str, str, str], MultisystemMetricInput
    ] = {}
    for row in observations:
        if row.condition not in {"workspace_only", "oracle_current_state"}:
            continue
        control_rows[
            (
                row.policy_profile_id,
                row.episode_id,
                row.opportunity_id,
                row.condition,
            )
        ] = row

    grouped: dict[
        tuple[str, str, str, str, str],
        list[
            tuple[
                MultisystemMetricInput,
                MultisystemMetricInput,
                MultisystemMetricInput,
            ]
        ],
    ] = defaultdict(list)
    for treatment in observations:
        if treatment.condition in {"workspace_only", "oracle_current_state"}:
            continue
        prefix = (
            treatment.policy_profile_id,
            treatment.episode_id,
            treatment.opportunity_id,
        )
        workspace = control_rows.get((*prefix, "workspace_only"))
        oracle = control_rows.get((*prefix, "oracle_current_state"))
        if workspace is None or oracle is None:
            continue
        grouped[
            (
                treatment.policy_profile_id,
                treatment.condition,
                treatment.readout,
                treatment.construct_kind,
                treatment.horizon_band,
            )
        ].append((treatment, workspace, oracle))

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        triples = grouped[key]
        treatment_gain = sum(
            treatment.behavior_score - workspace.behavior_score
            for treatment, workspace, _oracle in triples
        )
        oracle_advantage = sum(
            oracle.behavior_score - workspace.behavior_score
            for _treatment, workspace, oracle in triples
        )
        drift_triples = tuple(
            triple
            for triple in triples
            if all(_any_drift_is_eligible(row) for row in triple)
        )
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "construct_kind": key[3],
                "horizon_band": key[4],
                "n_matched_decisions": len(triples),
                "n_episodes": len(
                    {treatment.episode_id for treatment, _workspace, _oracle in triples}
                ),
                "mean_behavior_gain_beyond_workspace": _ratio_value(
                    treatment_gain,
                    len(triples),
                ),
                "mean_behavior_gap_to_oracle": _ratio_value(
                    sum(
                        oracle.behavior_score - treatment.behavior_score
                        for treatment, _workspace, oracle in triples
                    ),
                    len(triples),
                ),
                "oracle_gap_closed": _ratio_value(
                    treatment_gain,
                    oracle_advantage,
                ),
                "workspace_behavior_correct_rate": _ratio_value(
                    sum(workspace.is_correct for _treatment, workspace, _oracle in triples),
                    len(triples),
                ),
                "system_behavior_correct_rate": _ratio_value(
                    sum(treatment.is_correct for treatment, _workspace, _oracle in triples),
                    len(triples),
                ),
                "oracle_behavior_correct_rate": _ratio_value(
                    sum(oracle.is_correct for _treatment, _workspace, oracle in triples),
                    len(triples),
                ),
                "drift_matched_decisions": len(drift_triples),
                "targeted_drift_risk_difference_vs_workspace": _ratio_value(
                    sum(
                        _has_targeted_drift(treatment)
                        - _has_targeted_drift(workspace)
                        for treatment, workspace, _oracle in drift_triples
                    ),
                    len(drift_triples),
                ),
                "targeted_drift_risk_difference_vs_oracle": _ratio_value(
                    sum(
                        _has_targeted_drift(treatment)
                        - _has_targeted_drift(oracle)
                        for treatment, _workspace, oracle in drift_triples
                    ),
                    len(drift_triples),
                ),
            }
        )
    return tuple(output)


_MATCHED_CONSTRUCT_VARIANTS = (
    "static",
    "evolution",
    "hierarchical_conflict",
)


def compute_matched_construct_contrasts(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Compare static/evolution/conflict histories at one matched decision.

    These rows are deliberately cross-episode: episode identity changes with
    the manipulated history, while the counterfactual group fixes the terminal
    request, action catalog, gold action, checkpoint, and opaque option map.
    """

    grouped: dict[
        tuple[str, str, str, str, str],
        dict[str, list[MultisystemMetricInput]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in observations:
        if (
            not row.is_counterfactual_target
            or not row.counterfactual_group_id
            or row.counterfactual_variant
            not in _MATCHED_CONSTRUCT_VARIANTS
        ):
            continue
        grouped[
            (
                row.policy_profile_id,
                row.condition,
                row.readout,
                row.counterfactual_group_id,
                row.opportunity_id,
            )
        ][row.counterfactual_variant].append(row)

    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        variants = grouped[key]
        terminal_archetypes = {
            row.counterfactual_terminal_archetype
            for rows in variants.values()
            for row in rows
        }
        complete = (
            all(variants.get(name) for name in _MATCHED_CONSTRUCT_VARIANTS)
            and len(terminal_archetypes) == 1
        )
        static = variants.get("static", [])
        evolution = variants.get("evolution", [])
        conflict = variants.get("hierarchical_conflict", [])
        static_score = _mean_behavior(static)
        evolution_score = _mean_behavior(evolution)
        conflict_score = _mean_behavior(conflict)
        static_drift = _mean_targeted_drift(static)
        evolution_drift = _mean_targeted_drift(evolution)
        conflict_drift = _mean_targeted_drift(conflict)
        static_stage = _attribution_stage_summary(static)
        evolution_stage = _attribution_stage_summary(evolution)
        conflict_stage = _attribution_stage_summary(conflict)
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "counterfactual_group_id": key[3],
                "opportunity_id": key[4],
                "terminal_archetype": (
                    next(iter(terminal_archetypes))
                    if len(terminal_archetypes) == 1
                    else "inconsistent"
                ),
                "complete": complete,
                "n_static": len(static),
                "n_evolution": len(evolution),
                "n_hierarchical_conflict": len(conflict),
                "static_behavior_score": static_score,
                "evolution_behavior_score": evolution_score,
                "hierarchical_conflict_behavior_score": conflict_score,
                "state_evolution_penalty_vs_static": _optional_difference(
                    static_score,
                    evolution_score,
                ),
                "hierarchical_conflict_penalty_vs_static": (
                    _optional_difference(static_score, conflict_score)
                ),
                "static_correct_rate": _mean_correct(static),
                "evolution_correct_rate": _mean_correct(evolution),
                "hierarchical_conflict_correct_rate": _mean_correct(conflict),
                "state_evolution_correctness_penalty_vs_static": (
                    _optional_difference(
                        _mean_correct(static),
                        _mean_correct(evolution),
                    )
                ),
                "hierarchical_conflict_correctness_penalty_vs_static": (
                    _optional_difference(
                        _mean_correct(static),
                        _mean_correct(conflict),
                    )
                ),
                "static_targeted_drift_rate": static_drift,
                "evolution_targeted_drift_rate": evolution_drift,
                "hierarchical_conflict_targeted_drift_rate": conflict_drift,
                "static_targeted_drift_violation_rate": static_drift,
                "evolution_targeted_drift_violation_rate": evolution_drift,
                "hierarchical_conflict_targeted_drift_violation_rate": (
                    conflict_drift
                ),
                "state_evolution_drift_excess_vs_static": (
                    _optional_difference(evolution_drift, static_drift)
                ),
                "hierarchical_conflict_drift_excess_vs_static": (
                    _optional_difference(conflict_drift, static_drift)
                ),
                "state_evolution_drift_violation_excess_vs_static": (
                    _optional_difference(evolution_drift, static_drift)
                ),
                "hierarchical_conflict_drift_violation_excess_vs_static": (
                    _optional_difference(conflict_drift, static_drift)
                ),
                "static_attribution_stages": static_stage,
                "evolution_attribution_stages": evolution_stage,
                "hierarchical_conflict_attribution_stages": conflict_stage,
                "evolution_attribution_stage_changed": (
                    complete and evolution_stage != static_stage
                ),
                "hierarchical_conflict_attribution_stage_changed": (
                    complete and conflict_stage != static_stage
                ),
            }
        )
    workspace_by_decision = {
        (
            str(contrast_row["policy_profile_id"]),
            str(contrast_row["counterfactual_group_id"]),
            str(contrast_row["opportunity_id"]),
        ): contrast_row
        for contrast_row in output
        if contrast_row["condition"] == "workspace_only"
        and contrast_row["complete"] is True
    }
    adjusted: list[dict[str, object]] = []
    for contrast_row in output:
        workspace = workspace_by_decision.get(
            (
                str(contrast_row["policy_profile_id"]),
                str(contrast_row["counterfactual_group_id"]),
                str(contrast_row["opportunity_id"]),
            )
        )
        static_gain = _mapping_optional_difference(
            contrast_row,
            workspace,
            "static_behavior_score",
        )
        evolution_gain = _mapping_optional_difference(
            contrast_row,
            workspace,
            "evolution_behavior_score",
        )
        conflict_gain = _mapping_optional_difference(
            contrast_row,
            workspace,
            "hierarchical_conflict_behavior_score",
        )
        workspace_evolution_penalty = _mapping_optional_number(
            workspace,
            "state_evolution_penalty_vs_static",
        )
        workspace_conflict_penalty = _mapping_optional_number(
            workspace,
            "hierarchical_conflict_penalty_vs_static",
        )
        adjusted.append(
            {
                **contrast_row,
                "workspace_matched_control_available": workspace is not None,
                "static_gain_beyond_workspace": static_gain,
                "evolution_gain_beyond_workspace": evolution_gain,
                "hierarchical_conflict_gain_beyond_workspace": conflict_gain,
                # Difference-in-differences. Positive values mean that this
                # condition loses more performance under the construct than
                # workspace-only does, rather than merely inheriting a harder
                # workspace surface.
                "state_evolution_penalty_excess_over_workspace": (
                    _optional_difference(
                        _mapping_optional_number(
                            contrast_row,
                            "state_evolution_penalty_vs_static",
                        ),
                        workspace_evolution_penalty,
                    )
                ),
                "hierarchical_conflict_penalty_excess_over_workspace": (
                    _optional_difference(
                        _mapping_optional_number(
                            contrast_row,
                            "hierarchical_conflict_penalty_vs_static",
                        ),
                        workspace_conflict_penalty,
                    )
                ),
            }
        )
    return tuple(adjusted)


def compute_matched_construct_scorecard(
    observations: Sequence[MultisystemMetricInput],
) -> tuple[dict[str, object], ...]:
    """Aggregate complete matched construct groups without pooling controls."""

    contrasts = compute_matched_construct_contrasts(observations)
    grouped: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    totals: Counter[tuple[str, str, str]] = Counter()
    for row in contrasts:
        key = (
            str(row["policy_profile_id"]),
            str(row["condition"]),
            str(row["readout"]),
        )
        totals[key] += 1
        if row["complete"] is True:
            grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(totals):
        rows = grouped.get(key, [])
        archetype_counts = Counter(
            str(row.get("terminal_archetype", "")) for row in rows
        )
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "n_counterfactual_groups": totals[key],
                "n_complete_groups": len(rows),
                "n_current_v1_offline_groups": archetype_counts[
                    "current_v1_offline"
                ],
                "n_current_v2_offline_groups": archetype_counts[
                    "current_v2_offline"
                ],
                "n_authorized_cloud_groups": archetype_counts[
                    "authorized_cloud"
                ],
                "all_terminal_archetypes_covered": all(
                    archetype_counts[name] > 0
                    for name in (
                        "current_v1_offline",
                        "current_v2_offline",
                        "authorized_cloud",
                    )
                ),
                "mean_static_behavior_score": _mean_mapping_value(
                    rows,
                    "static_behavior_score",
                ),
                "mean_evolution_behavior_score": _mean_mapping_value(
                    rows,
                    "evolution_behavior_score",
                ),
                "mean_hierarchical_conflict_behavior_score": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_behavior_score",
                    )
                ),
                "mean_state_evolution_penalty_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "state_evolution_penalty_vs_static",
                    )
                ),
                "mean_hierarchical_conflict_penalty_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_penalty_vs_static",
                    )
                ),
                "mean_static_gain_beyond_workspace": _mean_mapping_value(
                    rows,
                    "static_gain_beyond_workspace",
                ),
                "mean_evolution_gain_beyond_workspace": (
                    _mean_mapping_value(
                        rows,
                        "evolution_gain_beyond_workspace",
                    )
                ),
                "mean_hierarchical_conflict_gain_beyond_workspace": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_gain_beyond_workspace",
                    )
                ),
                "mean_state_evolution_penalty_excess_over_workspace": (
                    _mean_mapping_value(
                        rows,
                        "state_evolution_penalty_excess_over_workspace",
                    )
                ),
                "mean_hierarchical_conflict_penalty_excess_over_workspace": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_penalty_excess_over_workspace",
                    )
                ),
                "mean_state_evolution_drift_excess_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "state_evolution_drift_excess_vs_static",
                    )
                ),
                "mean_hierarchical_conflict_drift_excess_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_drift_excess_vs_static",
                    )
                ),
                "mean_state_evolution_drift_violation_excess_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "state_evolution_drift_violation_excess_vs_static",
                    )
                ),
                "mean_hierarchical_conflict_drift_violation_excess_vs_static": (
                    _mean_mapping_value(
                        rows,
                        "hierarchical_conflict_drift_violation_excess_vs_static",
                    )
                ),
                "evolution_attribution_stage_change_rate": _ratio_value(
                    sum(
                        row["evolution_attribution_stage_changed"] is True
                        for row in rows
                    ),
                    len(rows),
                ),
                "hierarchical_conflict_attribution_stage_change_rate": (
                    _ratio_value(
                        sum(
                            row[
                                "hierarchical_conflict_attribution_stage_changed"
                            ]
                            is True
                            for row in rows
                        ),
                        len(rows),
                    )
                ),
            }
        )
    return tuple(output)


def _mean_behavior(rows: Sequence[MultisystemMetricInput]) -> float | None:
    return _ratio_value(sum(row.behavior_score for row in rows), len(rows))


def _mean_correct(rows: Sequence[MultisystemMetricInput]) -> float | None:
    return _ratio_value(sum(row.is_correct for row in rows), len(rows))


def _mean_targeted_drift(
    rows: Sequence[MultisystemMetricInput],
) -> float | None:
    eligible = tuple(row for row in rows if _any_drift_is_eligible(row))
    return _ratio_value(
        sum(_has_targeted_drift(row) for row in eligible),
        len(eligible),
    )


def _attribution_stage_summary(
    rows: Sequence[MultisystemMetricInput],
) -> str:
    counts = Counter(row.decision_attribution().stage for row in rows)
    return "|".join(f"{name}:{counts[name]}" for name in sorted(counts))


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return _stable_difference(left, right)


def _mapping_optional_number(
    row: Mapping[str, object] | None,
    field: str,
) -> float | None:
    if row is None:
        return None
    value = row.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _mapping_optional_difference(
    row: Mapping[str, object],
    reference: Mapping[str, object] | None,
    field: str,
) -> float | None:
    return _optional_difference(
        _mapping_optional_number(row, field),
        _mapping_optional_number(reference, field),
    )


def _mean_mapping_value(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> float | None:
    values = tuple(
        float(value)
        for row in rows
        for value in (row.get(field),)
        if isinstance(value, int | float) and not isinstance(value, bool)
    )
    return _ratio_value(sum(values), len(values))


def _ratio_value(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    observed = tuple(value for value in values if value is not None)
    return _ratio_value(sum(observed), len(observed))


def _flatten_state_ids(
    groups: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    """Return a stable unique state set from object-aligned attribution groups."""
    return tuple(sorted({state_id for group in groups for state_id in group}))


def _flag_rate(flags: Sequence[str], name: str, denominator: int) -> float | None:
    return _ratio_value(flags.count(name), denominator)


def _eligible_flag_rate(
    rows: Sequence[MultisystemMetricInput],
    flag: str,
) -> float | None:
    return _ratio_value(
        sum(flag in row.drift_flags for row in rows),
        len(rows),
    )


def _drift_is_eligible(
    row: BehaviorMetricInput | MultisystemMetricInput,
    flag: str,
) -> bool:
    eligible = row.drift_eligible_categories
    return eligible is None or flag in eligible


def _any_drift_is_eligible(
    row: BehaviorMetricInput | MultisystemMetricInput,
) -> bool:
    eligible = row.drift_eligible_categories
    return eligible is None or bool(eligible)


def _has_targeted_drift(
    row: BehaviorMetricInput | MultisystemMetricInput,
) -> bool:
    """Whether an observed flag matches this probe's preregistered construct."""
    eligible = row.drift_eligible_categories
    targeted = (
        set(_CANONICAL_DRIFT_CATEGORIES)
        if eligible is None
        else set(eligible)
    )
    return bool(targeted.intersection(row.drift_flags))


def _has_canonical_drift_violation(
    row: BehaviorMetricInput | MultisystemMetricInput,
) -> bool:
    """Whether any canonical violation occurred, independent of targeting."""
    return bool(set(_CANONICAL_DRIFT_CATEGORIES).intersection(row.drift_flags))


def _has_off_target_drift(
    row: BehaviorMetricInput | MultisystemMetricInput,
) -> bool:
    """Whether drift occurred outside the probe's preregistered categories."""
    eligible = row.drift_eligible_categories
    if eligible is None:
        return False
    observed = set(_CANONICAL_DRIFT_CATEGORIES).intersection(row.drift_flags)
    return bool(observed.difference(eligible))


def _aggregate_observation_status(statuses: Sequence[str]) -> str:
    values = tuple(statuses)
    if values and all(value == "complete" for value in values):
        return "complete"
    if any(value == "failed" for value in values):
        return "failed"
    return "partial"


def _retired_replacements(
    spec: SoftwareMem0VerticalSpec,
    replay: ReplayResult,
) -> tuple[tuple[str, ...], ...]:
    """Map each retired state to active same-kind, same-scope successors."""
    states = {state.state_id: state for state in spec.plan.state_units}
    active = tuple(replay.current.values())
    return tuple(
        tuple(
            sorted(
                candidate.state_id
                for candidate in active
                if retired is not None
                and candidate.kind == retired.kind
                and candidate.scope == retired.scope
            )
        )
        for retired_id in sorted(replay.invalidated)
        for retired in (states.get(retired_id),)
    )


def compute_qualification_metrics(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> MetricCollection:
    """Flatten runner results into formula inputs and aggregate them."""
    states: list[StateCheckpointMetricInput] = []
    retrievals: list[RetrievalMetricInput] = []
    behaviors: list[BehaviorMetricInput] = []
    usages: list[UsageMetricInput] = []
    seen_calls: set[str] = set()

    for task in matrix.task_results:
        for component, footprint in (
            ("qdrant_store", task.qdrant_store_bytes),
            ("history_store", task.history_store_bytes),
        ):
            if footprint is None:
                continue
            usages.append(
                UsageMetricInput(
                    input_tokens=None,
                    output_tokens=None,
                    cached_tokens=None,
                    reasoning_tokens=None,
                    latency_seconds=0.0,
                    retry_count=0,
                    terminal_failure=False,
                    component=component,
                    input_count=footprint,
                    usage_observed=True,
                )
            )
        spec = specs[task.episode_id]
        alignment_by_session = {
            item.checkpoint_session: item for item in task.alignments
        }
        write_by_session = {item.session_index: item for item in task.writes}
        previous_n_write = 0
        for index, write in enumerate(task.writes):
            write_delta = max(0, write.n_write - previous_n_write)
            previous_n_write = write.n_write
            for event in write.usage_events:
                if event.call_id in seen_calls:
                    continue
                seen_calls.add(event.call_id)
                usages.append(_usage_from_provider_event(event))
            alignment = alignment_by_session.get(write.session_index)
            if alignment is None:
                continue
            attribution_map = {
                item.memory_id: (
                    item.state_ids if item.contributes_positive_coverage else ()
                )
                for item in alignment.attributions
            }
            attribution_method_map = {
                item.memory_id: item.method for item in alignment.attributions
            }
            replay = replay_plan(spec.plan, write.session_index)
            eligible = eligible_write_state_ids(
                spec.plan,
                write.session_index,
            )
            new_memory_ids = {
                event.memory_id
                for event in write.events
                if getattr(event, "normalized_event", "") in {"add", "update"}
            }
            event_by_memory = {
                event.memory_id: event
                for event in write.events
                if event.memory_id in new_memory_ids
            }
            new_provenance = tuple(
                _event_provenance(event_by_memory[memory_id])
                for memory_id in sorted(new_memory_ids)
                if memory_id in event_by_memory
            )
            future_needed = tuple(
                state_id
                for state_id, state in replay.current.items()
                if is_benchmark_state_id(state_id)
                if any(
                    session >= write.session_index
                    for session in state.future_need_sessions
                )
            )
            states.append(
                StateCheckpointMetricInput(
                    eligible_write_state_ids=eligible,
                    new_memory_state_ids=tuple(
                        attribution_map.get(memory_id, ())
                        for memory_id in sorted(new_memory_ids)
                    ),
                    current_state_ids=tuple(
                        sorted(
                            state_id
                            for state_id in replay.current
                            if is_benchmark_state_id(state_id)
                        )
                    ),
                    future_needed_state_ids=tuple(sorted(future_needed)),
                    retired_state_ids=tuple(
                        sorted(
                            state_id
                            for state_id in replay.invalidated
                            if is_benchmark_state_id(state_id)
                        )
                    ),
                    live_memory_state_ids=tuple(
                        attribution_map.get(item.memory_id, ())
                        for item in write.inventory.items
                    ),
                    live_content_hashes=tuple(
                        item.content_hash for item in write.inventory.items
                    ),
                    n_write=write.n_write,
                    n_live=write.inventory.n_live,
                    write_latency_seconds=write.latency_seconds,
                    is_final_checkpoint=index == len(task.writes) - 1,
                    mutation_counts=tuple(
                        sorted(
                            Counter(event.native_event.lower() for event in write.events).items()
                        )
                    ),
                    new_memory_provenance=new_provenance,
                    live_memory_provenance=tuple(
                        _attribution_mode(alignment.attributions, item.memory_id)
                        for item in write.inventory.items
                    ),
                    new_memory_attribution_methods=tuple(
                        attribution_method_map.get(memory_id, "ambiguous")
                        for memory_id in sorted(new_memory_ids)
                    ),
                    live_memory_attribution_methods=tuple(
                        attribution_method_map.get(item.memory_id, "ambiguous")
                        for item in write.inventory.items
                    ),
                    provenance_complete=not (write_delta > 0 and not write.events),
                    retired_replacement_state_ids=_retired_replacements(spec, replay),
                )
            )

        trace_by_sceu = {item.sceu_id: item for item in task.retrieval_traces}
        opportunity_by_id = {
            item.opportunity_id: item for item in spec.plan.opportunities
        }
        sceu_gold = {item.sceu_id: item for item in spec.plan.sceu_units}
        for condition in task.condition_results:
            if condition.status == "failed":
                usages.append(
                    UsageMetricInput(
                        input_tokens=None,
                        output_tokens=None,
                        cached_tokens=None,
                        reasoning_tokens=None,
                        latency_seconds=0.0,
                        retry_count=0,
                        terminal_failure=True,
                    )
                )
            for row in condition.sceu_results:
                baseline = row.baseline_evaluations
                all_evaluations = list(baseline)
                for intervention in row.interventions:
                    all_evaluations.extend(intervention.evaluations)
                for evaluation in all_evaluations:
                    if evaluation.call_id in seen_calls:
                        continue
                    seen_calls.add(evaluation.call_id)
                    usages.append(_usage_from_evaluation(evaluation))

                loo = tuple(
                    item
                    for item in row.interventions
                    if item.intervention_kind == "leave_one_out"
                )
                baseline_action = (
                    baseline[0].selected_action_id if baseline else row.selected_action_id
                )
                action_flips = sum(
                    item.evaluations[0].selected_action_id != baseline_action
                    for item in loo
                )
                count_contrasts = tuple(
                    item
                    for item in row.interventions
                    if _is_memory_count_contrast(
                        getattr(item, "count_contrast", None)
                    )
                )
                opportunity = opportunity_by_id[row.opportunity_id]
                checkpoint_replay = replay_plan(
                    spec.plan,
                    row.checkpoint_session,
                )
                behaviors.append(
                    BehaviorMetricInput(
                        policy_profile_id=task.policy_profile_id,
                        condition=condition.condition,
                        readout=condition.readout,
                        result_id=condition.result_id,
                        behavior_score=row.behavior.behavior_score,
                        is_correct=row.behavior.is_correct,
                        visible_memory_count=len(row.model_visible_memory_ids),
                        causal_labels=tuple(
                            item.classification.label for item in loo
                        ),
                        intervention_labels=tuple(
                            item.classification.label
                            for item in row.interventions
                        ),
                        leave_one_out_count=len(loo),
                        leave_one_out_action_flips=action_flips,
                        count_contrast_count=len(count_contrasts),
                        count_contrast_action_flips=sum(
                            bool(getattr(item.classification, "action_changed", False))
                            for item in count_contrasts
                        ),
                        count_contrast_behavior_changes=sum(
                            bool(getattr(item.classification, "action_changed", False))
                            or bool(getattr(item.classification, "checker_changed", False))
                            for item in count_contrasts
                        ),
                        live_memory_count=_live_count_at_checkpoint(
                            task.writes,
                            row.checkpoint_session,
                        ),
                        drift_flags=row.normalized_drift_flags,
                        matched_group=row.matched_group,
                        checkpoint_session=row.checkpoint_session,
                        is_conflict_opportunity=_is_state_conflict_opportunity(
                            opportunity,
                            checkpoint_replay.invalidated,
                            pre_update_state_ids=tuple(
                                state.state_id
                                for state in spec.plan.state_units
                                if state.valid_from > row.checkpoint_session
                            ),
                        ),
                    )
                )

                trace = trace_by_sceu.get(row.sceu_id)
                if trace is None:
                    continue
                alignment = alignment_by_session.get(row.checkpoint_session)
                attribution_map = {
                    item.memory_id: (
                        item.state_ids
                        if item.contributes_positive_coverage
                        else ()
                    )
                    for item in alignment.attributions
                } if alignment is not None else {}
                gold = sceu_gold[row.sceu_id]
                retrievals.append(
                    RetrievalMetricInput(
                        required_state_ids=gold.required_state_ids,
                        stale_state_ids=tuple(
                            sorted(checkpoint_replay.invalidated)
                        ),
                        candidate_memory_state_ids=tuple(
                            attribution_map.get(item.memory_id, ())
                            for item in trace.candidates
                        ),
                        retrieved_memory_state_ids=tuple(
                            attribution_map.get(memory_id, ())
                            for memory_id in row.retrieved_memory_ids
                        ),
                        visible_memory_state_ids=tuple(
                            attribution_map.get(memory_id, ())
                            for memory_id in row.model_visible_memory_ids
                        ),
                        candidate_shortfall=trace.candidate_shortfall,
                        retrieval_latency_seconds=trace.search_latency_seconds,
                        rerank_latency_seconds=(
                            trace.rerank_result.latency_seconds
                            if condition.readout == "common_rerank"
                            and trace.rerank_result is not None
                            else None
                        ),
                    )
                )

        for trace in task.retrieval_traces:
            for event in trace.internal_usage:
                if event.call_id in seen_calls:
                    continue
                seen_calls.add(event.call_id)
                usages.append(_usage_from_provider_event(event))
            if trace.rerank_result is not None:
                usages.append(
                    UsageMetricInput(
                        input_tokens=None,
                        output_tokens=None,
                        cached_tokens=None,
                        reasoning_tokens=None,
                        latency_seconds=trace.rerank_result.latency_seconds,
                        retry_count=0,
                        terminal_failure=False,
                        component="reranker",
                        input_count=trace.rerank_result.input_count,
                        usage_observed=False,
                    )
                )
        for write in write_by_session.values():
            if write.latency_seconds < 0:
                raise ValueError("write latency must be non-negative")
    return compute_metric_collection(
        state_checkpoints=tuple(states),
        retrievals=tuple(retrievals),
        behaviors=tuple(behaviors),
        usages=tuple(usages),
    )


def _live_count_at_checkpoint(
    writes: Sequence[WriteSessionResult],
    checkpoint_session: int,
) -> int | None:
    """Read the last immutable inventory before a continuation checkpoint."""
    candidates = [
        item
        for item in writes
        if item.session_index < checkpoint_session
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item.session_index)
    return latest.inventory.n_live


def multisystem_observations_from_results(
    results: Sequence[object] | object,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
) -> tuple[MultisystemMetricInput, ...]:
    """Normalize schema-v2 evaluator DTOs into metric inputs.

    The evaluator intentionally stores only a prefix hash in each result.  This
    helper joins the result to the separately persisted immutable artifact so
    state coverage and memory-count metrics can be computed without exposing
    evaluator labels to the policy.  Missing artifacts are tolerated for control
    rows and produce denominator-safe null retrieval metrics for memory rows.
    """
    task_results = getattr(results, "task_results", None)
    rows = tuple(task_results) if task_results is not None else tuple(results)  # type: ignore[arg-type]
    output: list[MultisystemMetricInput] = []
    artifact_map = prefix_artifacts or {}
    for task in rows:
        episode_id = str(getattr(task, "episode_id", ""))
        spec = specs.get(episode_id)
        if spec is None:
            continue
        plan_metadata = spec.plan.metadata_dict
        task_span = profile_task_span(spec.plan)
        counterfactual_target = plan_metadata.get(
            "counterfactual_target_opportunity_id",
            "",
        )
        policy_id = str(getattr(task, "policy_profile_id", ""))
        task_condition = str(getattr(task, "condition", ""))
        artifact = _artifact_for_task(task, artifact_map)
        opportunity_by_id = {item.opportunity_id: item for item in spec.plan.opportunities}
        sceu_by_id = {item.sceu_id: item for item in spec.plan.sceu_units}
        for condition_result in getattr(task, "condition_results", ()):
            condition = str(getattr(condition_result, "condition", task_condition))
            readout = str(getattr(condition_result, "readout", "none"))
            for row in getattr(condition_result, "sceu_results", ()):
                sceu = sceu_by_id.get(str(getattr(row, "sceu_id", "")))
                if sceu is None:
                    continue
                checkpoint = _artifact_checkpoint(artifact, sceu.checkpoint_session)
                inventory = checkpoint.inventory if checkpoint is not None else None
                attribution = _artifact_attributions(
                    spec,
                    inventory,
                    checkpoint,
                    sceu.checkpoint_session,
                    artifact=artifact,
                )
                candidate_ids = tuple(getattr(row, "candidate_memory_ids", ()))
                retrieved_ids = tuple(getattr(row, "retrieved_memory_ids", ()))
                visible_ids = tuple(getattr(row, "model_visible_memory_ids", ()))
                backend_retrieved_ids = (
                    tuple(row.backend_retrieved_memory_ids)
                    if hasattr(row, "backend_retrieved_memory_ids")
                    else candidate_ids
                )
                selected_ids = (
                    tuple(row.selected_memory_ids)
                    if hasattr(row, "selected_memory_ids")
                    else retrieved_ids
                )
                behaviorally_used_ids = tuple(
                    getattr(row, "behaviorally_used_memory_ids", ())
                )
                intervention_rows = tuple(getattr(row, "interventions", ()))
                loo_rows = tuple(
                    item
                    for item in intervention_rows
                    if str(getattr(item, "intervention_kind", "")) == "leave_one_out"
                )
                neutral_rows = tuple(
                    item
                    for item in intervention_rows
                    if str(getattr(item, "intervention_kind", ""))
                    == "neutral_replacement"
                )
                sham_rows = tuple(
                    item
                    for item in intervention_rows
                    if str(getattr(item, "intervention_kind", ""))
                    == "sham_replacement"
                )
                primary_causal_rows = neutral_rows or loo_rows
                behaviorally_probed_ids = tuple(
                    str(getattr(item, "target_memory_id", ""))
                    for item in primary_causal_rows
                    if str(getattr(item, "target_memory_id", ""))
                )
                causal_labels = tuple(
                    str(getattr(getattr(item, "classification", None), "label", "indeterminate"))
                    for item in primary_causal_rows
                )
                intervention_labels = tuple(
                    str(getattr(getattr(item, "classification", None), "label", "indeterminate"))
                    for item in intervention_rows
                )
                count_contrasts = tuple(
                    item
                    for item in intervention_rows
                    if _is_memory_count_load_contrast(
                        getattr(item, "count_contrast", "")
                    )
                    and str(getattr(item, "intervention_kind", ""))
                    in {"count_add", "count_contrast"}
                )
                selected = str(getattr(row, "selected_action_id", ""))
                flips = sum(
                    bool(getattr(item, "evaluations", ()))
                    and str(getattr(item.evaluations[0], "selected_action_id", "")) != selected
                    for item in loo_rows
                )
                behavior = getattr(row, "behavior", None)
                behavior_score = float(
                    getattr(behavior, "behavior_score", getattr(row, "behavior_score", 0.0))
                )
                is_correct = bool(
                    getattr(behavior, "is_correct", getattr(row, "is_correct", False))
                )
                opportunity = opportunity_by_id.get(str(getattr(row, "opportunity_id", "")))
                current = replay_plan(spec.plan, sceu.checkpoint_session)
                construct = profile_sceu(spec.plan, sceu)
                stored_state_ids = _flatten_state_ids(
                    tuple(
                        _attributed_state_ids(attribution, item.memory_id)
                        for item in (() if inventory is None else inventory.items)
                    )
                )
                stored_exact_state_ids = _stored_states_for_provenance(
                    attribution,
                    inventory,
                    "native/exact",
                )
                stored_inferred_state_ids = _stored_states_for_provenance(
                    attribution,
                    inventory,
                    "inferred",
                )
                stored_unavailable_state_ids = _stored_states_for_provenance(
                    attribution,
                    inventory,
                    "unavailable",
                )
                stored_unavailable_state_ids = tuple(
                    sorted(
                        set(stored_unavailable_state_ids)
                        | {
                            state_id
                            for item in attribution.values()
                            if not item.contributes_positive_coverage
                            and item.method == "ambiguous"
                            for state_id in item.state_ids
                        }
                    )
                )
                storage_evidence_mode = _storage_evidence_mode(
                    construct.memory_reliant_state_ids,
                    stored_exact_state_ids,
                    stored_inferred_state_ids,
                    stored_unavailable_state_ids,
                    inventory_observed=inventory is not None,
                    checkpoint_evidence_mode=_artifact_storage_evidence_mode(
                        artifact,
                        sceu.checkpoint_session,
                    ),
                )
                probed_state_ids = _flatten_state_ids(
                    tuple(
                        _attributed_state_ids(attribution, memory_id)
                        for memory_id in behaviorally_probed_ids
                    )
                )
                used_state_ids = _flatten_state_ids(
                    tuple(
                        _attributed_state_ids(attribution, memory_id)
                        for memory_id in behaviorally_used_ids
                    )
                )
                output.append(
                    MultisystemMetricInput(
                        policy_profile_id=policy_id,
                        condition=condition,
                        readout=readout,
                        result_id=str(getattr(row, "result_id", "")),
                        behavior_score=behavior_score,
                        is_correct=is_correct,
                        required_state_ids=construct.current_required_state_ids,
                        stale_state_ids=tuple(sorted(current.invalidated)),
                        candidate_memory_state_ids=tuple(
                            _attributed_state_ids(attribution, memory_id)
                            for memory_id in candidate_ids
                        ),
                        retrieved_memory_state_ids=tuple(
                            _attributed_state_ids(attribution, memory_id)
                            for memory_id in retrieved_ids
                        ),
                        visible_memory_state_ids=tuple(
                            _attributed_state_ids(attribution, memory_id)
                            for memory_id in visible_ids
                        ),
                        backend_retrieved_memory_state_ids=tuple(
                            _attributed_state_ids(attribution, memory_id)
                            for memory_id in backend_retrieved_ids
                        ),
                        selected_memory_state_ids=tuple(
                            _attributed_state_ids(attribution, memory_id)
                            for memory_id in selected_ids
                        ),
                        candidate_shortfall=bool(getattr(row, "candidate_shortfall", False)),
                        visible_memory_count=len(visible_ids),
                        live_memory_count=(None if inventory is None else inventory.n_live),
                        causal_labels=causal_labels,
                        behaviorally_used_memory_ids=behaviorally_used_ids,
                        behavioral_use_probe_count=len(primary_causal_rows),
                        intervention_labels=intervention_labels,
                        leave_one_out_count=len(loo_rows),
                        leave_one_out_action_flips=flips,
                        sham_replacement_count=len(sham_rows),
                        sham_replacement_action_flips=sum(
                            bool(
                                getattr(
                                    getattr(item, "classification", None),
                                    "action_changed",
                                    False,
                                )
                            )
                            for item in sham_rows
                        ),
                        count_contrast_count=len(count_contrasts),
                        count_contrast_action_flips=sum(
                            bool(
                                getattr(
                                    getattr(item, "classification", None),
                                    "action_changed",
                                    False,
                                )
                            )
                            for item in count_contrasts
                        ),
                        count_contrast_behavior_changes=sum(
                            bool(
                                getattr(
                                    getattr(item, "classification", None),
                                    "action_changed",
                                    False,
                                )
                            )
                            or bool(
                                getattr(
                                    getattr(item, "classification", None),
                                    "checker_changed",
                                    False,
                                )
                            )
                            for item in count_contrasts
                        ),
                        drift_flags=tuple(getattr(row, "normalized_drift_flags", ())),
                        matched_group=str(getattr(row, "matched_group", "")),
                        checkpoint_session=sceu.checkpoint_session,
                        is_conflict_opportunity=(
                            False
                            if opportunity is None
                            else _is_state_conflict_opportunity(
                                opportunity,
                                current.invalidated,
                            )
                        ),
                        retrieval_latency_seconds=_checkpoint_retrieval_latency(
                            checkpoint,
                            sceu.opportunity_id,
                        ),
                        rerank_latency_seconds=(
                            _checkpoint_rerank_latency(
                                checkpoint,
                                sceu.opportunity_id,
                            )
                            if readout == "common_rerank"
                            else None
                        ),
                        status=str(getattr(condition_result, "status", "complete")),
                        baseline_stable=bool(
                            getattr(row, "baseline_stable", True)
                        ),
                        drift_eligible_categories=getattr(
                            row,
                            "drift_eligible_categories",
                            None,
                        ),
                        drift_lineage_pairs=(
                            tuple(getattr(row, "drift_lineage_pairs", ()))
                            or drift_lineage_pairs(spec, sceu)
                        ),
                        drift_lineage_evidence_mode=str(
                            getattr(
                                row,
                                "drift_lineage_evidence_mode",
                                DRIFT_LINEAGE_EVIDENCE_MODE,
                            )
                        ),
                        current_state_signature=str(
                            getattr(row, "current_state_signature", "")
                        ),
                        episode_id=episode_id,
                        sceu_id=sceu.sceu_id,
                        opportunity_id=sceu.opportunity_id,
                        selected_action_id=selected,
                        control_kind=(
                            str(getattr(row, "control_kind", ""))
                            if opportunity is None
                            else opportunity.control_kind
                        ),
                        construct_kind=construct.construct_kind,
                        horizon_band=construct.horizon_band,
                        handoff_count=construct.handoff_count,
                        oldest_required_state_age=(
                            construct.oldest_required_state_age
                        ),
                        latest_decision_event_distance=(
                            construct.latest_decision_event_distance
                        ),
                        dependency_depth=construct.dependency_depth,
                        relevant_transition_count=(
                            construct.relevant_transition_count
                        ),
                        memory_reliant_state_ids=(
                            construct.memory_reliant_state_ids
                        ),
                        nonexplicit_state_ids=construct.nonexplicit_state_ids,
                        stored_memory_state_ids=stored_state_ids,
                        stored_exact_state_ids=stored_exact_state_ids,
                        stored_inferred_state_ids=stored_inferred_state_ids,
                        stored_unavailable_state_ids=(
                            stored_unavailable_state_ids
                        ),
                        storage_evidence_mode=storage_evidence_mode,
                        behaviorally_probed_state_ids=probed_state_ids,
                        behaviorally_used_state_ids=used_state_ids,
                        counterfactual_group_id=plan_metadata.get(
                            "counterfactual_group_id",
                            "",
                        ),
                        counterfactual_variant=plan_metadata.get(
                            "counterfactual_variant",
                            "",
                        ),
                        counterfactual_terminal_archetype=plan_metadata.get(
                            "terminal_archetype",
                            "",
                        ),
                        is_counterfactual_target=(
                            bool(counterfactual_target)
                            and sceu.opportunity_id == counterfactual_target
                        ),
                        horizon_panel_id=plan_metadata.get(
                            "horizon_panel_id",
                            "",
                        ),
                        horizon_level=plan_metadata.get(
                            "horizon_level",
                            "",
                        ),
                        horizon_axis=plan_metadata.get(
                            "horizon_axis",
                            "",
                        ),
                        effective_task_step_count=(
                            task_span.effective_step_count
                        ),
                        max_task_dependency_depth=(
                            task_span.max_dependency_depth
                        ),
                        causally_linked_task_step_fraction=(
                            task_span.causally_linked_step_fraction
                        ),
                    )
                )
    return tuple(output)


def compute_schema_v2_metrics(
    results: Sequence[object] | object,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
    state_checkpoints: Sequence[StateCheckpointMetricInput] = (),
    usages: Sequence[UsageMetricInput] = (),
) -> MetricCollection:
    """Convenience facade used by server aggregation workers."""
    observations = multisystem_observations_from_results(
        results,
        specs,
        prefix_artifacts=prefix_artifacts,
    )
    return compute_multisystem_metrics(
        observations,
        state_checkpoints=state_checkpoints,
        usages=usages,
    )


def multisystem_state_checkpoints_from_artifacts(
    results: Sequence[object] | object,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
) -> tuple[StateCheckpointMetricInput, ...]:
    """Extract storage-side checkpoint rows from frozen native prefixes.

    Native evaluation results deliberately do not duplicate mutable backend
    state.  This join reconstructs the immutable write/inventory chain so the
    report can score exact versus inferred storage provenance at the same
    checkpoints used by continuation outcomes.
    """
    rows = getattr(results, "task_results", None)
    tasks = tuple(rows) if rows is not None else tuple(results)  # type: ignore[arg-type]
    artifacts = prefix_artifacts or {}
    output: list[StateCheckpointMetricInput] = []
    seen: set[tuple[str, str, int]] = set()
    for task in tasks:
        spec = specs.get(str(getattr(task, "episode_id", "")))
        if spec is None:
            continue
        artifact = _artifact_for_task(task, artifacts)
        if artifact is None:
            continue
        previous_n_write = 0
        for checkpoint in artifact.checkpoints:
            inventory = checkpoint.inventory
            if inventory is None:
                continue
            write_delta = max(0, inventory.n_write - previous_n_write)
            previous_n_write = inventory.n_write
            key = (
                spec.plan.episode_id,
                str(getattr(artifact, "backend", "")),
                checkpoint.checkpoint_session,
            )
            if key in seen:
                continue
            seen.add(key)
            writes = tuple(checkpoint.writes)
            events = tuple(event for write in writes for event in write.events)
            event_by_memory: dict[str, object] = {}
            for event in events:
                event_by_memory[event.memory_id] = event
            attribution = _artifact_attributions(
                spec,
                inventory,
                checkpoint,
                checkpoint.checkpoint_session,
                artifact=artifact,
            )
            attributed_state_ids = {
                memory_id: (
                    item.state_ids if item.contributes_positive_coverage else ()
                )
                for memory_id, item in attribution.items()
            }
            # Prefix checkpoint ``n_sessions`` is the post-final-write
            # snapshot.  Latent replay indexes sessions themselves and thus
            # ends at ``n_sessions - 1``.
            replay_session = (
                checkpoint.checkpoint_session - 1
                if checkpoint.checkpoint_session == spec.plan.n_sessions
                else checkpoint.checkpoint_session
            )
            replay = replay_plan(spec.plan, replay_session)
            new_ids = tuple(
                sorted(
                    {
                        event.memory_id
                        for event in events
                        if getattr(event, "normalized_event", "") in {"add", "update"}
                    }
                )
            )
            eligible = eligible_write_state_ids(
                spec.plan,
                max(0, checkpoint.checkpoint_session - 1),
            )
            future = tuple(
                state_id
                for state_id, state in replay.current.items()
                if is_benchmark_state_id(state_id)
                if any(
                    session >= checkpoint.checkpoint_session
                    for session in state.future_need_sessions
                )
            )
            output.append(
                StateCheckpointMetricInput(
                    eligible_write_state_ids=eligible if new_ids else (),
                    new_memory_state_ids=tuple(
                        attributed_state_ids.get(memory_id, ())
                        for memory_id in new_ids
                    ),
                    current_state_ids=tuple(
                        sorted(
                            state_id
                            for state_id in replay.current
                            if is_benchmark_state_id(state_id)
                        )
                    ),
                    future_needed_state_ids=tuple(sorted(future)),
                    retired_state_ids=tuple(
                        sorted(
                            state_id
                            for state_id in replay.invalidated
                            if is_benchmark_state_id(state_id)
                        )
                    ),
                    live_memory_state_ids=tuple(
                        attributed_state_ids.get(item.memory_id, ())
                        for item in inventory.items
                    ),
                    live_content_hashes=tuple(item.content_hash for item in inventory.items),
                    n_write=sum(
                        getattr(write, "n_write", 0) for write in writes
                    ),
                    n_live=inventory.n_live,
                    write_latency_seconds=sum(
                        float(getattr(write, "latency_seconds", 0.0)) for write in writes
                    ),
                    is_final_checkpoint=(
                        checkpoint.checkpoint_session == spec.plan.n_sessions
                    ),
                    mutation_counts=tuple(
                        sorted(Counter(event.native_event.lower() for event in events).items())
                    ),
                    new_memory_provenance=tuple(
                        _event_provenance(event_by_memory[memory_id])
                        for memory_id in new_ids
                        if memory_id in event_by_memory
                    ),
                    live_memory_provenance=tuple(
                        attribution.get(item.memory_id, MemoryAttribution(
                            memory_id=item.memory_id,
                            state_ids=(),
                            method="ambiguous",
                            contributes_positive_coverage=False,
                            reason="missing evaluator attribution",
                        )).provenance_mode
                        for item in inventory.items
                    ),
                    new_memory_attribution_methods=tuple(
                        attribution.get(memory_id, MemoryAttribution(
                            memory_id=memory_id,
                            state_ids=(),
                            method="ambiguous",
                            contributes_positive_coverage=False,
                            reason="missing evaluator attribution",
                        )).method
                        for memory_id in new_ids
                    ),
                    live_memory_attribution_methods=tuple(
                        attribution.get(item.memory_id, MemoryAttribution(
                            memory_id=item.memory_id,
                            state_ids=(),
                            method="ambiguous",
                            contributes_positive_coverage=False,
                            reason="missing evaluator attribution",
                        )).method
                        for item in inventory.items
                    ),
                    provenance_complete=not (write_delta > 0 and not events),
                    retired_replacement_state_ids=_retired_replacements(spec, replay),
                )
            )
    return tuple(output)


def _artifact_for_task(
    task: object,
    artifacts: Mapping[str, object],
) -> MemoryPrefixArtifact | None:
    condition = str(getattr(task, "condition", ""))
    episode_id = str(getattr(task, "episode_id", ""))
    candidates = (
        f"{episode_id}--{condition}",
        condition,
        f"{episode_id}--{_backend_alias(condition)}",
        _backend_alias(condition),
    )
    for key in candidates:
        candidate = artifacts.get(key)
        if candidate is None:
            continue
        if isinstance(candidate, MemoryPrefixArtifact):
            return candidate
        if isinstance(candidate, Mapping):
            try:
                return MemoryPrefixArtifact.from_dict(candidate)
            except Exception:
                return None
    return None


def _backend_alias(condition: str) -> str:
    if condition in {"mem0_controlled", "mem0_native"}:
        return "mem0"
    return condition


def _artifact_checkpoint(
    artifact: MemoryPrefixArtifact | None,
    checkpoint_session: int,
) -> MemoryPrefixCheckpoint | None:
    if artifact is None:
        return None
    return next(
        (item for item in artifact.checkpoints if item.checkpoint_session == checkpoint_session),
        None,
    )


def _artifact_attributions(
    spec: SoftwareMem0VerticalSpec,
    inventory: InventorySnapshot | None,
    checkpoint: MemoryPrefixCheckpoint | None,
    checkpoint_session: int,
    *,
    artifact: MemoryPrefixArtifact | None = None,
) -> dict[str, MemoryAttribution]:
    if inventory is None:
        return {}
    signatures = build_software_fact_signatures(spec.plan)
    signature_by_state = {signature.state_id: signature for signature in signatures}
    events_by_memory: dict[str, list[object]] = defaultdict(list)
    relevant_checkpoints = (
        tuple(
            item
            for item in artifact.checkpoints
            if item.checkpoint_session <= checkpoint_session
        )
        if artifact is not None
        else (() if checkpoint is None else (checkpoint,))
    )
    for relevant in relevant_checkpoints:
        for write in relevant.writes:
            for event in write.events:
                events_by_memory[event.memory_id].append(event)
    output: dict[str, MemoryAttribution] = {}
    for item in inventory.items:
        metadata = dict(item.metadata)
        metadata_session = metadata.get("session_index")
        events = events_by_memory.get(item.memory_id, [])
        lifecycle = next(
            (
                event
                for event in reversed(events)
                if getattr(event, "new_content_hash", None) == item.content_hash
            ),
            events[-1] if events else None,
        )
        source_session = (
            metadata_session
            if isinstance(metadata_session, int)
            else getattr(lifecycle, "session_index", None)
        )
        eligible = (
            eligible_write_state_ids(spec.plan, source_session)
            if isinstance(source_session, int) and source_session < checkpoint_session
            else ()
        )
        provenance_mode: ProvenanceMode = cast(
            ProvenanceMode,
            _event_provenance(lifecycle) if lifecycle is not None else "unavailable",
        )
        attribution = attribute_memory(
            item.memory_id,
            item.content,
            signatures,
            unique_write_state_ids=eligible,
            provenance_mode=provenance_mode,
            source_event_ids=tuple(
                str(getattr(event, "operation_id", ""))
                for event in events
                if getattr(event, "operation_id", "")
            ),
            source_session=(
                int(source_session) if isinstance(source_session, int) else None
            ),
        )
        gold_event_ids = tuple(
            sorted(
                {
                    event_id
                    for state_id in attribution.state_ids
                    for event_id in (
                        signature_by_state[state_id].source_event_ids
                        if state_id in signature_by_state
                        else ()
                    )
                }
            )
        )
        if gold_event_ids:
            attribution = MemoryAttribution(
                memory_id=attribution.memory_id,
                state_ids=attribution.state_ids,
                method=attribution.method,
                contributes_positive_coverage=attribution.contributes_positive_coverage,
                reason=attribution.reason,
                provenance_mode=attribution.provenance_mode,
                source_event_ids=tuple(
                    sorted(set(attribution.source_event_ids) | set(gold_event_ids))
                ),
                source_session=attribution.source_session,
            )
        output[item.memory_id] = attribution
    return output


def _attributed_state_ids(
    attribution: Mapping[str, MemoryAttribution],
    memory_id: str,
) -> tuple[str, ...]:
    item = attribution.get(memory_id)
    return () if item is None else tuple(item.state_ids)


def _stored_states_for_provenance(
    attribution: Mapping[str, MemoryAttribution],
    inventory: InventorySnapshot | None,
    provenance_mode: ProvenanceMode,
) -> tuple[str, ...]:
    if inventory is None:
        return ()
    return _flatten_state_ids(
        tuple(
            item.state_ids
            for memory in inventory.items
            for item in (attribution.get(memory.memory_id),)
            if item is not None
            and item.contributes_positive_coverage
            and item.provenance_mode == provenance_mode
        )
    )


def _storage_evidence_mode(
    required_state_ids: Sequence[str],
    exact_state_ids: Sequence[str],
    inferred_state_ids: Sequence[str],
    unavailable_state_ids: Sequence[str],
    *,
    inventory_observed: bool,
    checkpoint_evidence_mode: StorageEvidenceMode = "unavailable",
) -> StorageEvidenceMode:
    required = set(required_state_ids)
    if not required:
        return "not_applicable"
    if not inventory_observed:
        return "unavailable"
    if required.intersection(unavailable_state_ids):
        return "unavailable"
    modes: set[ProvenanceMode] = set()
    if required.intersection(exact_state_ids):
        modes.add("native/exact")
    if required.intersection(inferred_state_ids):
        modes.add("inferred")
    if checkpoint_evidence_mode in {"native/exact", "inferred"}:
        modes.add(checkpoint_evidence_mode)
    elif checkpoint_evidence_mode == "mixed":
        modes.update({"native/exact", "inferred"})
    if not modes:
        return "unavailable"
    if modes == {"native/exact", "inferred"}:
        return "mixed"
    if modes == {"native/exact"}:
        return "native/exact"
    return "inferred"


def _artifact_storage_evidence_mode(
    artifact: MemoryPrefixArtifact | None,
    checkpoint_session: int,
) -> StorageEvidenceMode:
    """Return lifecycle evidence available for proving current-store absence.

    Object-level provenance alone is insufficient when a required state has no
    matching object. The complete checkpoint history is then the evidence:
    native mutation events are exact, normalized inventory diffs are inferred,
    and a positive write delta without either is unavailable. An observed but
    unchanged inventory is conservatively treated as inferred.
    """

    if artifact is None:
        return "unavailable"
    relevant = tuple(
        item
        for item in artifact.checkpoints
        if item.checkpoint_session <= checkpoint_session
    )
    if not relevant or any(item.inventory is None for item in relevant):
        return "unavailable"
    modes: set[str] = set()
    previous_n_write = 0
    for item in relevant:
        inventory = item.inventory
        if inventory is None:  # guarded above; retained for type narrowing
            return "unavailable"
        write_delta = max(0, inventory.n_write - previous_n_write)
        previous_n_write = inventory.n_write
        events = tuple(event for write in item.writes for event in write.events)
        if write_delta > 0 and not events:
            return "unavailable"
        modes.update(_event_provenance(event) for event in events)
    if modes == {"native/exact", "inferred"}:
        return "mixed"
    if modes == {"native/exact"}:
        return "native/exact"
    if modes == {"inferred"}:
        return "inferred"
    return "inferred"


def _checkpoint_retrieval_latency(checkpoint: object | None, opportunity_id: str) -> float:
    if checkpoint is None:
        return 0.0
    for search in getattr(checkpoint, "retrievals", ()):
        # CandidateSearch currently binds the query, not opportunity ID.  A
        # checkpoint with one SCEU has an unambiguous search; otherwise leave
        # latency as an explicitly observed-but-zero placeholder.
        _ = opportunity_id
        return float(getattr(search, "latency_seconds", 0.0))
    return 0.0


def _checkpoint_rerank_latency(checkpoint: object | None, opportunity_id: str) -> float | None:
    if checkpoint is None:
        return None
    for rerank in getattr(checkpoint, "common_reranks", ()):
        if getattr(rerank, "opportunity_id", None) != opportunity_id:
            continue
        result = getattr(rerank, "result", None)
        return float(getattr(result, "latency_seconds", 0.0))
    return None


def _state_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[StateCheckpointMetricInput],
    *,
    prefix: str = "",
) -> None:
    def metric(name: str) -> str:
        return f"{prefix}{name}"

    write_covered = 0
    write_eligible = 0
    selective_objects = 0
    written_objects = 0
    current_objects = 0
    live_objects = 0
    represented_current = 0
    current_states = 0
    stale_objects = 0
    duplicate_objects = 0
    aligned_live_objects = 0
    responsive_retired = 0
    physically_retired = 0
    retired_states = 0
    represented_successors = 0
    retired_states_with_successors = 0
    current_stale_coexistence = 0
    mutation_totals: Counter[str] = Counter()
    aligned_future = 0
    future_states = 0
    write_latency = 0.0
    final_write_count = 0
    final_live_count = 0
    final_logical_state_units = 0
    final_attributed_objects = 0
    final_attribution_methods: Counter[str] = Counter()
    write_attribution_methods: Counter[str] = Counter()
    final_checkpoints = 0
    for item in observations:
        eligible = set(item.eligible_write_state_ids)
        new_objects = [set(states) for states in item.new_memory_state_ids]
        write_attribution_methods.update(
            _aligned_attribution_methods(
                item.new_memory_attribution_methods,
                len(new_objects),
            )
        )
        new_represented = set().union(*new_objects) if new_objects else set()
        write_covered += len(eligible & new_represented)
        write_eligible += len(eligible)
        selective_objects += sum(bool(states & eligible) for states in new_objects)
        written_objects += len(new_objects)

        current = set(item.current_state_ids)
        retired = set(item.retired_state_ids)
        live = [set(states) for states in item.live_memory_state_ids]
        represented = set().union(*live) if live else set()
        current_objects += sum(bool(states & current) for states in live)
        live_objects += item.n_live
        represented_current += len(represented & current)
        current_states += len(current)
        stale_objects += sum(bool(states & retired) for states in live)
        current_stale_coexistence += sum(
            bool(states & current) and bool(states & retired)
            for states in live
        )
        mutation_totals.update(dict(item.mutation_counts))
        counts = Counter(
            state_id
            for states in live
            for state_id in states
        )
        duplicate_objects += sum(max(0, count - 1) for count in counts.values())
        aligned_live_objects += sum(bool(states) for states in live)
        replacements = {
            retired_id: set(successors)
            for retired_id, successors in zip(
                sorted(retired),
                item.retired_replacement_state_ids,
                strict=False,
            )
        }
        for retired_id in sorted(retired):
            absent = retired_id not in represented
            successors = replacements.get(retired_id, set())
            physically_retired += absent
            responsive_retired += absent or bool(successors & represented)
            if successors:
                retired_states_with_successors += 1
                represented_successors += bool(successors & represented)
        retired_states += len(retired)
        future = set(item.future_needed_state_ids)
        aligned_future += len(future & represented)
        future_states += len(future)
        write_latency += item.write_latency_seconds
        if item.is_final_checkpoint:
            final_checkpoints += 1
            final_write_count += item.n_write
            final_live_count += item.n_live
            final_logical_state_units += len(represented)
            final_attributed_objects += sum(bool(states) for states in live)
            final_attribution_methods.update(
                _aligned_attribution_methods(
                    item.live_memory_attribution_methods,
                    len(live),
                )
            )

    values[metric("write_coverage")] = safe_ratio(write_covered, write_eligible)
    values[metric("write_selectivity")] = safe_ratio(selective_objects, written_objects)
    precision = safe_ratio(current_objects, live_objects)
    recall = safe_ratio(represented_current, current_states)
    values[metric("current_state_storage_precision")] = precision
    values[metric("current_state_storage_recall")] = recall
    values[metric("current_state_storage_f1")] = _f1(precision, recall)
    values[metric("stale_state_retention_rate")] = safe_ratio(stale_objects, live_objects)
    values[metric("current_stale_coexistence_rate")] = safe_ratio(
        current_stale_coexistence,
        live_objects,
    )
    values[metric("duplicate_live_memory_rate")] = safe_ratio(
        duplicate_objects,
        aligned_live_objects,
    )
    values[metric("update_delete_responsiveness")] = safe_ratio(
        responsive_retired,
        retired_states,
    )
    values[metric("physical_retirement_rate")] = safe_ratio(
        physically_retired,
        retired_states,
    )
    values[metric("superseding_state_storage_rate")] = safe_ratio(
        represented_successors,
        retired_states_with_successors,
    )
    values[metric("write_to_continuation_alignment")] = safe_ratio(
        aligned_future,
        future_states,
    )
    values[metric("memory_write_count")] = safe_ratio(
        final_write_count,
        final_checkpoints,
    )
    values[metric("memory_write_count_total")] = (
        safe_ratio(final_write_count, 1)
        if final_checkpoints
        else safe_ratio(0, 0)
    )
    values[metric("live_memory_count")] = safe_ratio(
        final_live_count,
        final_checkpoints,
    )
    values[metric("live_native_object_count")] = safe_ratio(
        final_live_count,
        final_checkpoints,
    )
    values[metric("live_logical_state_unit_count")] = safe_ratio(
        final_logical_state_units,
        final_checkpoints,
    )
    values[metric("native_objects_per_logical_state_unit")] = safe_ratio(
        final_live_count,
        final_logical_state_units,
    )
    values[metric("attributed_native_object_rate")] = safe_ratio(
        final_attributed_objects,
        final_live_count,
    )
    for method in (
        "exact_signature",
        "multi_signature",
        "lexical_signature",
        "unique_provenance",
        "no_match",
        "ambiguous",
        "unavailable",
    ):
        values[metric(f"semantic_attribution_{method}_rate")] = safe_ratio(
            final_attribution_methods[method],
            final_live_count,
        )
        values[metric(f"write_semantic_attribution_{method}_rate")] = safe_ratio(
            write_attribution_methods[method],
            sum(write_attribution_methods.values()),
        )
    values[metric("live_memory_count_total")] = (
        safe_ratio(final_live_count, 1)
        if final_checkpoints
        else safe_ratio(0, 0)
    )
    values[metric("mean_write_latency_seconds")] = safe_ratio(
        write_latency,
        len(observations),
    )
    values[metric("storage_provenance_completeness")] = safe_ratio(
        sum(item.provenance_complete for item in observations),
        len(observations),
    )
    for event_name in ("add", "update", "delete", "observed_add", "observed_delta", "merge"):
        values[metric(f"mutation_{event_name}_count")] = safe_ratio(
            mutation_totals[event_name], 1 if observations else 0
        )


def _filter_state_observations(
    observations: Sequence[StateCheckpointMetricInput],
    mode: str,
) -> tuple[StateCheckpointMetricInput, ...]:
    """Project immutable checkpoint records onto one provenance track."""
    projected: list[StateCheckpointMetricInput] = []
    for item in observations:
        new_pairs = tuple(
            pair
            for pair, provenance in zip(
                item.new_memory_state_ids,
                item.new_memory_provenance,
                strict=False,
            )
            if provenance == mode
        )
        live_pairs = tuple(
            pair
            for pair, provenance in zip(
                item.live_memory_state_ids,
                item.live_memory_provenance,
                strict=False,
            )
            if provenance == mode
        )
        if not new_pairs and not live_pairs:
            continue
        new_methods = tuple(
            method
            for method, provenance in zip(
                _aligned_attribution_methods(
                    item.new_memory_attribution_methods,
                    len(item.new_memory_state_ids),
                ),
                item.new_memory_provenance,
                strict=False,
            )
            if provenance == mode
        )
        live_methods = tuple(
            method
            for method, provenance in zip(
                _aligned_attribution_methods(
                    item.live_memory_attribution_methods,
                    len(item.live_memory_state_ids),
                ),
                item.live_memory_provenance,
                strict=False,
            )
            if provenance == mode
        )
        projected.append(
            StateCheckpointMetricInput(
                eligible_write_state_ids=(
                    item.eligible_write_state_ids if new_pairs else ()
                ),
                new_memory_state_ids=new_pairs,
                current_state_ids=item.current_state_ids,
                future_needed_state_ids=item.future_needed_state_ids,
                retired_state_ids=item.retired_state_ids,
                live_memory_state_ids=live_pairs,
                live_content_hashes=tuple(
                    content_hash
                    for content_hash, provenance in zip(
                        item.live_content_hashes,
                        item.live_memory_provenance,
                        strict=False,
                    )
                    if provenance == mode
                ),
                n_write=len(new_pairs),
                n_live=len(live_pairs),
                write_latency_seconds=item.write_latency_seconds,
                is_final_checkpoint=item.is_final_checkpoint,
                mutation_counts=item.mutation_counts,
                new_memory_provenance=(mode,) * len(new_pairs),
                live_memory_provenance=(mode,) * len(live_pairs),
                new_memory_attribution_methods=new_methods,
                live_memory_attribution_methods=live_methods,
                provenance_complete=item.provenance_complete,
                retired_replacement_state_ids=item.retired_replacement_state_ids,
            )
        )
    return tuple(projected)


def _aligned_attribution_methods(
    methods: Sequence[str],
    count: int,
) -> tuple[str, ...]:
    """Return one explicit semantic-attribution label per memory object."""
    allowed = {
        "exact_signature",
        "multi_signature",
        "lexical_signature",
        "unique_provenance",
        "no_match",
        "ambiguous",
    }
    output = tuple(
        method if method in allowed else "unavailable"
        for method in methods[:count]
    )
    if len(output) < count:
        output += ("unavailable",) * (count - len(output))
    return output


def _event_provenance(event: object) -> str:
    source = str(getattr(event, "source", ""))
    native_event = str(getattr(event, "native_event", ""))
    if source in {
        "inventory_diff",
        "inventory_delta",
        "inventory_snapshot_diff",
        "write_inventory_diff",
        "snapshot_diff",
        "neo4j_graph_diff",
    }:
        return "inferred"
    if native_event.upper().startswith("INFERRED"):
        return "inferred"
    return "native/exact"


def _attribution_mode(attributions: Sequence[object], memory_id: str) -> str:
    for item in attributions:
        if getattr(item, "memory_id", None) == memory_id:
            return str(getattr(item, "provenance_mode", "unavailable"))
    return "unavailable"


def _retrieval_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[RetrievalMetricInput],
) -> None:
    candidate_states_found = 0
    required_states = 0
    relevant_retrieved_objects = 0
    retrieved_objects = 0
    retrieved_states_found = 0
    visible_states_found = 0
    visible_objects = 0
    contaminated_visible = 0
    stale_candidates = 0
    candidate_objects = 0
    stale_visible = 0
    stale_retrieved = 0
    retrieved_not_visible = 0
    shortfalls = 0
    retrieval_latency = 0.0
    rerank_latency = 0.0
    rerank_count = 0
    relevant_backend_objects = 0
    backend_objects = 0
    backend_states_found = 0
    relevant_selected_objects = 0
    selected_objects = 0
    selected_states_found = 0
    for item in observations:
        required = set(item.required_state_ids)
        stale = set(item.stale_state_ids)
        candidates = [set(states) for states in item.candidate_memory_state_ids]
        retrieved = [set(states) for states in item.retrieved_memory_state_ids]
        visible = [set(states) for states in item.visible_memory_state_ids]
        backend = [
            set(states)
            for states in (
                item.candidate_memory_state_ids
                if item.backend_retrieved_memory_state_ids is None
                else item.backend_retrieved_memory_state_ids
            )
        ]
        selected = [
            set(states)
            for states in (
                item.retrieved_memory_state_ids
                if item.selected_memory_state_ids is None
                else item.selected_memory_state_ids
            )
        ]
        candidate_union = set().union(*candidates) if candidates else set()
        retrieved_union = set().union(*retrieved) if retrieved else set()
        visible_union = set().union(*visible) if visible else set()
        backend_union = set().union(*backend) if backend else set()
        selected_union = set().union(*selected) if selected else set()
        candidate_states_found += len(required & candidate_union)
        required_states += len(required)
        relevant_retrieved_objects += sum(bool(states & required) for states in retrieved)
        retrieved_objects += len(retrieved)
        retrieved_states_found += len(required & retrieved_union)
        visible_states_found += len(required & visible_union)
        visible_objects += len(visible)
        candidate_objects += len(candidates)
        stale_candidates += sum(bool(states & stale) for states in candidates)
        stale_visible += sum(bool(states & stale) for states in visible)
        contaminated_visible += sum(not bool(states & required) for states in visible)
        stale_retrieved += sum(bool(states & stale) for states in retrieved)
        retrieved_not_visible += max(0, len(retrieved) - len(visible))
        shortfalls += item.candidate_shortfall
        retrieval_latency += item.retrieval_latency_seconds
        if item.rerank_latency_seconds is not None:
            rerank_count += 1
            rerank_latency += item.rerank_latency_seconds
        relevant_backend_objects += sum(bool(states & required) for states in backend)
        backend_objects += len(backend)
        backend_states_found += len(required & backend_union)
        relevant_selected_objects += sum(bool(states & required) for states in selected)
        selected_objects += len(selected)
        selected_states_found += len(required & selected_union)

    values["candidate_recall"] = safe_ratio(
        candidate_states_found,
        required_states,
    )
    precision = safe_ratio(relevant_retrieved_objects, retrieved_objects)
    recall = safe_ratio(retrieved_states_found, required_states)
    values["retrieval_precision"] = precision
    values["retrieval_recall"] = recall
    values["retrieval_f1"] = _f1(precision, recall)
    values["retrieval_false_positive_rate"] = safe_ratio(
        retrieved_objects - relevant_retrieved_objects,
        retrieved_objects,
    )
    values["retrieval_timeliness"] = safe_ratio(
        retrieved_states_found,
        required_states,
    )
    backend_precision = safe_ratio(relevant_backend_objects, backend_objects)
    backend_recall = safe_ratio(backend_states_found, required_states)
    values["backend_retrieval_precision"] = backend_precision
    values["backend_retrieval_recall"] = backend_recall
    values["backend_retrieval_f1"] = _f1(backend_precision, backend_recall)
    selection_precision = safe_ratio(relevant_selected_objects, selected_objects)
    selection_recall = safe_ratio(selected_states_found, required_states)
    values["selection_precision"] = selection_precision
    values["selection_recall"] = selection_recall
    values["selection_f1"] = _f1(selection_precision, selection_recall)
    values["backend_to_selected_yield"] = safe_ratio(
        selected_objects,
        backend_objects,
    )
    values["selected_to_visible_yield"] = safe_ratio(
        visible_objects,
        selected_objects,
    )
    values["candidate_shortfall_rate"] = safe_ratio(
        shortfalls,
        len(observations),
    )
    values["visible_sufficiency"] = safe_ratio(
        visible_states_found,
        required_states,
    )
    values["visible_contamination"] = safe_ratio(
        contaminated_visible,
        visible_objects,
    )
    values["stale_candidate_exposure"] = safe_ratio(
        stale_candidates,
        candidate_objects,
    )
    values["stale_visible_exposure"] = safe_ratio(
        stale_visible,
        visible_objects,
    )
    values["stale_retrieval_rate"] = safe_ratio(
        stale_retrieved,
        retrieved_objects,
    )
    values["retrieved_but_not_visible_rate"] = safe_ratio(
        retrieved_not_visible,
        retrieved_objects,
    )
    values["retrieval_to_visible_yield"] = safe_ratio(
        retrieved_objects - retrieved_not_visible,
        retrieved_objects,
    )
    values["mean_retrieval_latency_seconds"] = safe_ratio(
        retrieval_latency,
        len(observations),
    )
    values["mean_rerank_latency_seconds"] = safe_ratio(
        rerank_latency,
        rerank_count,
    )


def _behavior_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[BehaviorMetricInput],
) -> None:
    causal_labels = [
        label for item in observations for label in item.causal_labels
    ]
    intervention_labels = [
        label for item in observations for label in item.intervention_labels
    ]
    causal_effect_labels = {
        "beneficial",
        "harmful",
        "causal_direction_ambiguous",
    }
    causal_used = sum(label in causal_effect_labels for label in causal_labels)
    values["causal_memory_use_rate"] = safe_ratio(
        causal_used,
        len(causal_labels),
    )
    values["unique_causal_effect_rate"] = safe_ratio(
        causal_used,
        len(causal_labels),
    )
    no_detected_unique_effect = sum(
        label
        in {
            "visible_without_detected_unique_causal_effect",
            # Completed schema-v1 reports used an over-strong name.
            "visible_not_causally_used",
        }
        for label in causal_labels
    )
    values["visible_but_not_causally_used_rate"] = safe_ratio(
        no_detected_unique_effect,
        len(causal_labels),
    )
    values["visible_without_detected_unique_causal_effect_rate"] = safe_ratio(
        no_detected_unique_effect,
        len(causal_labels),
    )
    values["visible_to_causal_use_yield"] = safe_ratio(
        causal_used,
        sum(item.visible_memory_count for item in observations),
    )
    probe_count = sum(item.behavioral_use_probe_count for item in observations)
    visible_count = sum(item.visible_memory_count for item in observations)
    values["behavioral_use_probe_coverage"] = safe_ratio(
        probe_count,
        visible_count,
    )
    values["probed_memory_causal_use_rate"] = safe_ratio(
        sum(item.behaviorally_used_memory_count for item in observations),
        probe_count,
    )
    # The benchmark probes one preregistered focal object per SCEU rather than
    # claiming exhaustive causal attribution over every visible object.  Keep
    # the historical yield key for compatibility and expose its interpretation
    # explicitly as a conservative lower bound.
    values["model_visible_to_behaviorally_used_yield"] = safe_ratio(
        sum(item.behaviorally_used_memory_count for item in observations),
        visible_count,
    )
    values["model_visible_behavioral_use_lower_bound"] = safe_ratio(
        sum(item.behaviorally_used_memory_count for item in observations),
        visible_count,
    )
    values["beneficial_intervention_rate"] = safe_ratio(
        intervention_labels.count("beneficial"),
        len(intervention_labels),
    )
    values["harmful_intervention_rate"] = safe_ratio(
        intervention_labels.count("harmful"),
        len(intervention_labels),
    )
    values["ambiguous_intervention_rate"] = safe_ratio(
        intervention_labels.count("causal_direction_ambiguous"),
        len(intervention_labels),
    )
    values["unstable_intervention_rate"] = safe_ratio(
        sum(
            label in {"unstable_baseline", "intervention_unstable"}
            for label in intervention_labels
        ),
        len(intervention_labels),
    )
    loo_count = sum(item.leave_one_out_count for item in observations)
    loo_flips = sum(item.leave_one_out_action_flips for item in observations)
    values["leave_one_memory_out_action_flip_rate"] = safe_ratio(
        loo_flips,
        loo_count,
    )
    sham_count = sum(item.sham_replacement_count for item in observations)
    values["sham_replacement_action_flip_rate"] = safe_ratio(
        sum(item.sham_replacement_action_flips for item in observations),
        sham_count,
    )
    count_contrasts = sum(item.count_contrast_count for item in observations)
    values["memory_count_contrast_rate"] = safe_ratio(
        sum(item.count_contrast_action_flips for item in observations),
        count_contrasts,
    )
    values["memory_count_behavior_change_rate"] = safe_ratio(
        sum(item.count_contrast_behavior_changes for item in observations),
        count_contrasts,
    )
    for flag, metric_name in (
        ("constraint_loss", "constraint_loss_rate"),
        ("plan_deviation", "current_plan_deviation_rate"),
        ("stale_state", "stale_state_action_rate"),
        ("local_over_global", "local_over_global_rate"),
    ):
        eligible = [
            item for item in observations if _drift_is_eligible(item, flag)
        ]
        values[metric_name] = safe_ratio(
            sum(flag in item.drift_flags for item in eligible),
            len(eligible),
        )
        targeted_name = {
            "constraint_loss": "targeted_constraint_loss_rate",
            "plan_deviation": "targeted_plan_deviation_rate",
            "stale_state": "targeted_stale_state_rate",
            "local_over_global": "targeted_local_over_global_rate",
        }[flag]
        observed_name = {
            "constraint_loss": "observed_constraint_loss_rate",
            "plan_deviation": "observed_plan_deviation_rate",
            "stale_state": "observed_stale_state_rate",
            "local_over_global": "observed_local_over_global_rate",
        }[flag]
        values[targeted_name] = values[metric_name]
        values[observed_name] = safe_ratio(
            sum(flag in item.drift_flags for item in observations),
            len(observations),
        )
        canonical_violation_name = {
            "constraint_loss": "canonical_constraint_loss_violation_rate",
            "plan_deviation": "canonical_plan_deviation_violation_rate",
            "stale_state": "canonical_stale_state_violation_rate",
            "local_over_global": "canonical_local_over_global_violation_rate",
        }[flag]
        values[canonical_violation_name] = values[observed_name]
        values[f"{metric_name}_eligible_n"] = safe_ratio(len(eligible), 1)
    aggregate_eligible = [
        item for item in observations if _any_drift_is_eligible(item)
    ]
    values["aggregate_drift_rate"] = safe_ratio(
        sum(_has_targeted_drift(item) for item in aggregate_eligible),
        len(aggregate_eligible),
    )
    values["targeted_aggregate_drift_rate"] = values["aggregate_drift_rate"]
    values["observed_aggregate_drift_rate"] = safe_ratio(
        sum(_has_canonical_drift_violation(item) for item in observations),
        len(observations),
    )
    values["canonical_drift_violation_rate"] = values[
        "observed_aggregate_drift_rate"
    ]
    values["off_target_drift_rate"] = safe_ratio(
        sum(_has_off_target_drift(item) for item in observations),
        len(observations),
    )
    values["off_target_drift_n"] = safe_ratio(
        sum(_has_off_target_drift(item) for item in observations),
        1,
    )
    values["aggregate_drift_rate_eligible_n"] = safe_ratio(
        len(aggregate_eligible),
        1,
    )
    values["mean_behavior_score"] = safe_ratio(
        sum(item.behavior_score for item in observations),
        len(observations),
    )
    values["behavior_correct_rate"] = safe_ratio(
        sum(item.is_correct for item in observations),
        len(observations),
    )
    conflict = [item for item in observations if item.is_conflict_opportunity]
    values["state_conflict_resolution_accuracy"] = safe_ratio(
        sum(item.is_correct for item in conflict),
        len(conflict),
    )
    values["matched_early_late_behavioral_decay"] = _matched_decay(observations)
    values["long_horizon_invariant_behavioral_drift_rate"] = (
        _matched_decay(observations, invariant_only=True)
    )
    values["state_evolution_late_resolution_accuracy"] = (
        _state_evolution_resolution(observations)
    )
    by_live_count: dict[int, list[BehaviorMetricInput]] = defaultdict(list)
    for item in observations:
        if item.live_memory_count is not None:
            by_live_count[item.live_memory_count].append(item)
    for live_count, rows in sorted(by_live_count.items()):
        suffix = f"_live_memory_count_{live_count}"
        values[f"mean_behavior_score{suffix}"] = safe_ratio(
            sum(item.behavior_score for item in rows), len(rows)
        )
        values[f"behavior_correct_rate{suffix}"] = safe_ratio(
            sum(item.is_correct for item in rows), len(rows)
        )
        values[f"aggregate_drift_rate{suffix}"] = safe_ratio(
            sum(
                _has_targeted_drift(item)
                for item in rows
                if _any_drift_is_eligible(item)
            ),
            sum(_any_drift_is_eligible(item) for item in rows),
        )
    _baseline_comparison_metrics(values, observations)


def _is_state_conflict_opportunity(
    opportunity: ContinuationOpportunity,
    invalidated_state_ids: Collection[str],
    *,
    pre_update_state_ids: Collection[str] = (),
) -> bool:
    if opportunity.challenge_type in {"scope-conflict", "valid-update"}:
        return True
    if opportunity.challenge_type != "matched-branch":
        return False
    invalidated = set(invalidated_state_ids)
    pre_update = set(pre_update_state_ids)
    valid_actions = set(opportunity.valid_action_ids)
    return any(
        action.action_id not in valid_actions
        and bool(
            invalidated.intersection(action.satisfies_state_ids)
            or pre_update.intersection(action.satisfies_state_ids)
        )
        for action in opportunity.action_catalog
    )


def _baseline_comparison_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[BehaviorMetricInput],
) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for item in observations:
        grouped[(item.policy_profile_id, item.condition, item.readout)].append(
            item.behavior_score
        )
    means = {
        key: sum(scores) / len(scores)
        for key, scores in grouped.items()
        if scores
    }
    policies = sorted({item.policy_profile_id for item in observations})
    # Keep the schema-v1 keys exactly as before.  For schema-v2 every backend
    # and readout receives its own key; native and common-rerank are never
    # silently averaged into one cell.
    gains: list[float] = []
    closures: list[float] = []
    rerank_deltas: list[float] = []
    comparisons = (
        (
            "mem0_controlled_native",
            "mem0_controlled",
            "native",
        ),
        (
            "mem0_controlled_common_rerank",
            "mem0_controlled",
            "common_rerank",
        ),
        (
            "mem0_native",
            "mem0_native",
            "native",
        ),
    )
    comparison_gains: dict[str, list[float]] = {prefix: [] for prefix, _, _ in comparisons}
    comparison_closures: dict[str, list[float]] = {prefix: [] for prefix, _, _ in comparisons}
    schema2_conditions = ("flat_retrieval", "mem0", "amem", "memos")
    schema2_readouts = {
        "flat_retrieval": ("common_rerank",),
        "mem0": ("native", "common_rerank"),
        "amem": ("native", "common_rerank"),
        "memos": ("native", "common_rerank"),
    }
    schema2_buckets: dict[str, list[float]] = defaultdict(list)
    has_schema2 = any(item.condition in schema2_conditions for item in observations)

    def append_closure(
        numerator: float,
        workspace: float | None,
        oracle: float | None,
        bucket: list[float],
    ) -> None:
        if workspace is None or oracle is None or oracle == workspace:
            return
        bucket.append(
            float(
                Decimal(str(numerator))
                / (Decimal(str(oracle)) - Decimal(str(workspace)))
            )
        )

    for policy in policies:
        workspace = means.get((policy, "workspace_only", "none"))
        oracle = means.get((policy, "oracle_current_state", "none"))
        mem_values = [
            score
            for (owner, condition, _), score in means.items()
            if owner == policy and condition.startswith("mem0_")
        ]
        if workspace is not None and mem_values:
            mem_mean = sum(mem_values) / len(mem_values)
            gain = _stable_difference(mem_mean, workspace)
            gains.append(gain)
            if oracle is not None and oracle != workspace:
                closures.append(
                    float(
                        Decimal(str(gain))
                        / (
                            Decimal(str(oracle))
                            - Decimal(str(workspace))
                        )
                    )
                )
        for prefix, condition, readout in comparisons:
            score = means.get((policy, condition, readout))
            if workspace is None or score is None:
                continue
            gain = _stable_difference(score, workspace)
            comparison_gains[prefix].append(gain)
            append_closure(gain, workspace, oracle, comparison_closures[prefix])
        native = means.get((policy, "mem0_controlled", "native"))
        common = means.get((policy, "mem0_controlled", "common_rerank"))
        if native is not None and common is not None:
            rerank_deltas.append(_stable_difference(common, native))

        flat = means.get((policy, "flat_retrieval", "common_rerank"))
        full = means.get((policy, "full_context", "none"))
        if not has_schema2:
            continue
        for condition in schema2_conditions:
            # Emit an explicit null for an inapplicable Flat native readout so
            # downstream tables can distinguish "not applicable" from missing
            # computation without inventing a score.
            readouts = schema2_readouts[condition]
            if condition == "flat_retrieval":
                readouts = ("native", "common_rerank")
            for readout in readouts:
                score = means.get((policy, condition, readout))
                prefix = f"{condition}_{readout}"
                gain_values = []
                gap_values = []
                flat_values = []
                if workspace is not None and score is not None:
                    gain_values.append(_stable_difference(score, workspace))
                if flat is not None and score is not None and condition != "flat_retrieval":
                    flat_values.append(_stable_difference(score, flat))
                if full is not None and score is not None:
                    gap_values.append(_stable_difference(full, score))
                schema2_buckets[f"{prefix}_gain_beyond_workspace"].extend(gain_values)
                schema2_buckets[f"{prefix}_gain_over_flat"].extend(flat_values)
                schema2_buckets[f"{prefix}_gap_to_full_context"].extend(gap_values)
                closure: list[float] = []
                if workspace is not None and score is not None:
                    append_closure(
                        _stable_difference(score, workspace),
                        workspace,
                        oracle,
                        closure,
                    )
                schema2_buckets[f"{prefix}_oracle_gap_closed"].extend(closure)
            native = means.get((policy, condition, "native"))
            common = means.get((policy, condition, "common_rerank"))
            if native is not None and common is not None:
                schema2_buckets[f"{condition}_common_rerank_minus_native"].append(
                    _stable_difference(common, native)
                )
    values["mem0_gain_beyond_workspace"] = safe_ratio(
        sum(gains),
        len(gains),
    )
    values["oracle_gap_closed"] = safe_ratio(
        sum(closures),
        len(closures),
    )
    for prefix, _, _ in comparisons:
        gains_for_comparison = comparison_gains[prefix]
        closures_for_comparison = comparison_closures[prefix]
        values[f"{prefix}_gain_beyond_workspace"] = safe_ratio(
            sum(gains_for_comparison),
            len(gains_for_comparison),
        )
        values[f"{prefix}_oracle_gap_closed"] = safe_ratio(
            sum(closures_for_comparison),
            len(closures_for_comparison),
        )
    values["common_rerank_behavior_delta"] = safe_ratio(
        sum(rerank_deltas),
        len(rerank_deltas),
    )
    for name, bucket in sorted(schema2_buckets.items()):
        values[name] = safe_ratio(sum(bucket), len(bucket))


def _usage_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[UsageMetricInput],
) -> None:
    policy = [
        item for item in observations if item.component == "policy"
    ]
    for field, metric_name in (
        ("input_tokens", "policy_input_tokens"),
        ("output_tokens", "policy_output_tokens"),
        ("cached_tokens", "policy_cached_tokens"),
        ("reasoning_tokens", "policy_reasoning_tokens"),
    ):
        numbers = [
            getattr(item, field)
            for item in policy
            if getattr(item, field) is not None
        ]
        values[metric_name] = (
            safe_ratio(sum(numbers), 1) if numbers else safe_ratio(0, 0)
        )
    values["mean_policy_latency_seconds"] = safe_ratio(
        sum(item.latency_seconds for item in policy),
        len(policy),
    )
    values["policy_retry_rate"] = safe_ratio(
        sum(item.retry_count > 0 for item in policy),
        len(policy),
    )
    reliability = [
        item
        for item in observations
        if item.component not in {"qdrant_store", "history_store"}
    ]
    values["terminal_failure_rate"] = safe_ratio(
        sum(item.terminal_failure for item in reliability),
        len(reliability),
    )
    internal = [
        item
        for item in observations
        if item.component == "memory_internal_llm"
    ]
    for field, metric_name in (
        ("input_tokens", "memory_internal_input_tokens"),
        ("output_tokens", "memory_internal_output_tokens"),
        ("cached_tokens", "memory_internal_cached_tokens"),
        ("reasoning_tokens", "memory_internal_reasoning_tokens"),
    ):
        values[metric_name] = _observed_token_total(internal, field)
    values["memory_internal_call_count"] = safe_ratio(len(internal), 1)
    values["memory_internal_usage_observed_rate"] = safe_ratio(
        sum(item.usage_observed for item in internal),
        len(internal),
    )
    values["mean_memory_internal_latency_seconds"] = safe_ratio(
        sum(item.latency_seconds for item in internal),
        len(internal),
    )

    embedding = [
        item for item in observations if item.component == "embedding"
    ]
    values["embedding_call_count"] = safe_ratio(len(embedding), 1)
    values["embedding_input_count"] = safe_ratio(
        sum(item.input_count for item in embedding),
        1,
    )
    values["embedding_input_tokens"] = _observed_token_total(
        embedding,
        "input_tokens",
    )
    values["embedding_usage_observed_rate"] = safe_ratio(
        sum(item.usage_observed for item in embedding),
        len(embedding),
    )
    values["mean_embedding_latency_seconds"] = safe_ratio(
        sum(item.latency_seconds for item in embedding),
        len(embedding),
    )

    reranker = [
        item for item in observations if item.component == "reranker"
    ]
    values["reranker_call_count"] = safe_ratio(len(reranker), 1)
    values["reranker_candidate_pairs"] = safe_ratio(
        sum(item.input_count for item in reranker),
        1,
    )
    values["mean_reranker_service_latency_seconds"] = safe_ratio(
        sum(item.latency_seconds for item in reranker),
        len(reranker),
    )
    for component, metric_name in (
        ("qdrant_store", "qdrant_store_bytes"),
        ("history_store", "history_store_bytes"),
    ):
        footprints = [
            item
            for item in observations
            if item.component == component
        ]
        values[metric_name] = (
            safe_ratio(
                sum(item.input_count for item in footprints),
                1,
            )
            if footprints
            else safe_ratio(0, 0)
        )


def _usage_from_evaluation(evaluation: PolicyEvaluation) -> UsageMetricInput:
    usage = evaluation.response.usage
    return UsageMetricInput(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        latency_seconds=evaluation.response.latency_seconds,
        retry_count=evaluation.response.retry_count,
        terminal_failure=False,
        component="policy",
        input_count=1,
        usage_observed=usage.observed,
    )


def _usage_from_provider_event(
    event: object,
) -> UsageMetricInput:
    from lhmsb.adapters.mem0_qualification import ProviderUsageEvent

    if not isinstance(event, ProviderUsageEvent):
        raise TypeError("provider usage event has the wrong type")
    return UsageMetricInput(
        input_tokens=event.input_tokens,
        output_tokens=event.output_tokens,
        cached_tokens=event.cached_tokens,
        reasoning_tokens=event.reasoning_tokens,
        latency_seconds=event.latency_seconds,
        retry_count=event.retry_count or 0,
        terminal_failure=event.error_class is not None,
        component=event.component,
        input_count=event.input_count,
        usage_observed=event.usage_observed,
    )


def _observed_token_total(
    observations: Sequence[UsageMetricInput],
    field: str,
) -> MetricValue:
    numbers = [
        getattr(item, field)
        for item in observations
        if getattr(item, field) is not None
    ]
    return safe_ratio(sum(numbers), 1) if numbers else safe_ratio(0, 0)


def _matched_decay(
    observations: Sequence[BehaviorMetricInput],
    *,
    invariant_only: bool = False,
) -> MetricValue:
    grouped: dict[tuple[str, str, str], list[BehaviorMetricInput]] = defaultdict(list)
    for item in observations:
        if item.matched_group:
            grouped[
                (item.policy_profile_id, item.result_id, item.matched_group)
            ].append(item)
    decays = 0
    pairs = 0
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: item.checkpoint_session)
        if len(ordered) < 2:
            continue
        if invariant_only and (
            not ordered[0].current_state_signature
            or ordered[0].current_state_signature
            != ordered[-1].current_state_signature
        ):
            continue
        pairs += 1
        decays += (
            ordered[-1].behavior_score < ordered[0].behavior_score
            or (
                not ordered[0].drift_flags
                and bool(ordered[-1].drift_flags)
            )
        )
    return safe_ratio(decays, pairs)


def _state_evolution_resolution(
    observations: Sequence[BehaviorMetricInput],
) -> MetricValue:
    grouped: dict[tuple[str, str, str], list[BehaviorMetricInput]] = defaultdict(list)
    for item in observations:
        if item.matched_group:
            grouped[(item.policy_profile_id, item.result_id, item.matched_group)].append(item)
    resolved = 0
    pairs = 0
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: item.checkpoint_session)
        if len(ordered) < 2:
            continue
        if (
            not ordered[0].current_state_signature
            or not ordered[-1].current_state_signature
            or ordered[0].current_state_signature
            == ordered[-1].current_state_signature
        ):
            continue
        pairs += 1
        resolved += ordered[-1].is_correct
    return safe_ratio(resolved, pairs)


def _stable_difference(left: float, right: float) -> float:
    return float(Decimal(str(left)) - Decimal(str(right)))


def _f1(precision: MetricValue, recall: MetricValue) -> MetricValue:
    if precision.value is None or recall.value is None:
        return safe_ratio(0, 0)
    return safe_ratio(
        2 * precision.value * recall.value,
        precision.value + recall.value,
    )


__all__ = [
    "BehaviorMetricInput",
    "MetricCollection",
    "MetricValue",
    "MultisystemMetricInput",
    "RetrievalMetricInput",
    "StateCheckpointMetricInput",
    "UsageMetricInput",
    "compute_metric_collection",
    "compute_failure_attribution_scorecard",
    "compute_long_horizon_scorecard",
    "compute_long_horizon_control_contrasts",
    "compute_matched_construct_contrasts",
    "compute_matched_construct_scorecard",
    "compute_multisystem_metrics",
    "compute_multisystem_metrics_by_cell",
    "compute_multisystem_scorecard",
    "compute_schema_v2_metrics",
    "compute_qualification_metrics",
    "decision_attribution_rows",
    "multisystem_observations_from_results",
    "multisystem_state_checkpoints_from_artifacts",
    "safe_ratio",
]
