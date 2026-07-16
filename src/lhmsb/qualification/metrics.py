"""Denominator-safe lifecycle, retrieval, causal-use, and drift metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.qualification.runner import (
    PolicyEvaluation,
    QualificationMatrixResult,
)


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
    _retrieval_metrics(values, retrievals)
    _behavior_metrics(values, behaviors)
    _usage_metrics(values, usages)
    return MetricCollection(metrics=tuple(sorted(values.items())))


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
        for index, write in enumerate(task.writes):
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
            replay = replay_plan(spec.plan, write.session_index)
            eligible = tuple(
                event.target_state_id
                for event in spec.plan.events
                if event.session == write.session_index and event.type == "add"
            )
            new_memory_ids = {
                event.memory_id
                for event in write.events
                if event.native_event in {"ADD", "UPDATE", "OBSERVED_ADD"}
            }
            future_needed = tuple(
                state_id
                for state_id, state in replay.current.items()
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
                    current_state_ids=tuple(sorted(replay.current)),
                    future_needed_state_ids=tuple(sorted(future_needed)),
                    retired_state_ids=tuple(sorted(replay.invalidated)),
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
                opportunity = opportunity_by_id[row.opportunity_id]
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
                        drift_flags=row.normalized_drift_flags,
                        matched_group=row.matched_group,
                        checkpoint_session=row.checkpoint_session,
                        is_conflict_opportunity=opportunity.challenge_type
                        in {"scope-conflict", "valid-update", "matched-branch"},
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
                replay = replay_plan(spec.plan, row.checkpoint_session)
                retrievals.append(
                    RetrievalMetricInput(
                        required_state_ids=gold.required_state_ids,
                        stale_state_ids=tuple(sorted(replay.invalidated)),
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


def _state_metrics(
    values: dict[str, MetricValue],
    observations: Sequence[StateCheckpointMetricInput],
) -> None:
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
    retired_states = 0
    aligned_future = 0
    future_states = 0
    write_latency = 0.0
    final_write_count = 0
    final_live_count = 0
    final_checkpoints = 0
    for item in observations:
        eligible = set(item.eligible_write_state_ids)
        new_objects = [set(states) for states in item.new_memory_state_ids]
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
        counts = Counter(
            state_id
            for states in live
            for state_id in states
        )
        duplicate_objects += sum(max(0, count - 1) for count in counts.values())
        aligned_live_objects += sum(bool(states) for states in live)
        responsive_retired += len(retired - represented)
        retired_states += len(retired)
        future = set(item.future_needed_state_ids)
        aligned_future += len(future & represented)
        future_states += len(future)
        write_latency += item.write_latency_seconds
        if item.is_final_checkpoint:
            final_checkpoints += 1
            final_write_count += item.n_write
            final_live_count += item.n_live

    values["write_coverage"] = safe_ratio(write_covered, write_eligible)
    values["write_selectivity"] = safe_ratio(selective_objects, written_objects)
    precision = safe_ratio(current_objects, live_objects)
    recall = safe_ratio(represented_current, current_states)
    values["current_state_storage_precision"] = precision
    values["current_state_storage_recall"] = recall
    values["current_state_storage_f1"] = _f1(precision, recall)
    values["stale_state_retention_rate"] = safe_ratio(stale_objects, live_objects)
    values["duplicate_live_memory_rate"] = safe_ratio(
        duplicate_objects,
        aligned_live_objects,
    )
    values["update_delete_responsiveness"] = safe_ratio(
        responsive_retired,
        retired_states,
    )
    values["write_to_continuation_alignment"] = safe_ratio(
        aligned_future,
        future_states,
    )
    values["memory_write_count"] = (
        safe_ratio(final_write_count, 1)
        if final_checkpoints
        else safe_ratio(0, 0)
    )
    values["live_memory_count"] = (
        safe_ratio(final_live_count, 1)
        if final_checkpoints
        else safe_ratio(0, 0)
    )
    values["mean_write_latency_seconds"] = safe_ratio(
        write_latency,
        len(observations),
    )


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
    stale_retrieved = 0
    retrieved_not_visible = 0
    shortfalls = 0
    retrieval_latency = 0.0
    rerank_latency = 0.0
    rerank_count = 0
    for item in observations:
        required = set(item.required_state_ids)
        stale = set(item.stale_state_ids)
        candidates = [set(states) for states in item.candidate_memory_state_ids]
        retrieved = [set(states) for states in item.retrieved_memory_state_ids]
        visible = [set(states) for states in item.visible_memory_state_ids]
        candidate_union = set().union(*candidates) if candidates else set()
        retrieved_union = set().union(*retrieved) if retrieved else set()
        visible_union = set().union(*visible) if visible else set()
        candidate_states_found += len(required & candidate_union)
        required_states += len(required)
        relevant_retrieved_objects += sum(bool(states & required) for states in retrieved)
        retrieved_objects += len(retrieved)
        retrieved_states_found += len(required & retrieved_union)
        visible_states_found += len(required & visible_union)
        visible_objects += len(visible)
        contaminated_visible += sum(not bool(states & required) for states in visible)
        stale_retrieved += sum(bool(states & stale) for states in retrieved)
        retrieved_not_visible += max(0, len(retrieved) - len(visible))
        shortfalls += item.candidate_shortfall
        retrieval_latency += item.retrieval_latency_seconds
        if item.rerank_latency_seconds is not None:
            rerank_count += 1
            rerank_latency += item.rerank_latency_seconds

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
    values["stale_retrieval_rate"] = safe_ratio(
        stale_retrieved,
        retrieved_objects,
    )
    values["retrieved_but_not_visible_rate"] = safe_ratio(
        retrieved_not_visible,
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
    causal_used = sum(label in {"beneficial", "harmful"} for label in causal_labels)
    values["causal_memory_use_rate"] = safe_ratio(
        causal_used,
        len(causal_labels),
    )
    values["visible_but_not_causally_used_rate"] = safe_ratio(
        causal_labels.count("visible_not_causally_used"),
        len(causal_labels),
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
    flag_names = (
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    )
    for flag, metric_name in (
        ("constraint_loss", "constraint_loss_rate"),
        ("plan_deviation", "current_plan_deviation_rate"),
        ("stale_state", "stale_state_action_rate"),
        ("local_over_global", "local_over_global_rate"),
    ):
        values[metric_name] = safe_ratio(
            sum(flag in item.drift_flags for item in observations),
            len(observations),
        )
    values["aggregate_drift_rate"] = safe_ratio(
        sum(any(flag in item.drift_flags for flag in flag_names) for item in observations),
        len(observations),
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
    _baseline_comparison_metrics(values, observations)


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
    gains: list[float] = []
    closures: list[float] = []
    rerank_deltas: list[float] = []
    policies = sorted({item.policy_profile_id for item in observations})
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
        native = means.get((policy, "mem0_controlled", "native"))
        common = means.get((policy, "mem0_controlled", "common_rerank"))
        if native is not None and common is not None:
            rerank_deltas.append(_stable_difference(common, native))
    values["mem0_gain_beyond_workspace"] = safe_ratio(
        sum(gains),
        len(gains),
    )
    values["oracle_gap_closed"] = safe_ratio(
        sum(closures),
        len(closures),
    )
    values["common_rerank_behavior_delta"] = safe_ratio(
        sum(rerank_deltas),
        len(rerank_deltas),
    )


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
        pairs += 1
        decays += ordered[-1].behavior_score < ordered[0].behavior_score
    return safe_ratio(decays, pairs)


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
    "RetrievalMetricInput",
    "StateCheckpointMetricInput",
    "UsageMetricInput",
    "compute_metric_collection",
    "compute_qualification_metrics",
    "safe_ratio",
]
