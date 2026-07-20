"""Deterministic, hash-addressed qualification report artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.qualification.metrics import (
    compute_multisystem_metrics,
    compute_multisystem_metrics_by_cell,
    compute_multisystem_scorecard,
    compute_qualification_metrics,
    multisystem_observations_from_results,
)
from lhmsb.qualification.prefix import MemoryPrefixArtifact
from lhmsb.qualification.runner import (
    PolicyEvaluation,
    QualificationMatrixResult,
    QualificationTaskResult,
    RetrievalTrace,
    SCEURunResult,
)

REPORT_SCHEMA_VERSION = 2
REQUIRED_REPORT_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "tasks.jsonl",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "memory_events.jsonl",
    "memory_inventory.jsonl",
    "retrieval_trace.jsonl",
    "interventions.jsonl",
    "api_usage.jsonl",
    "policy_calls.jsonl",
    "prefix_manifests.jsonl",
    "graph_diagnostics.jsonl",
    "metrics.json",
    "metrics_by_cell.json",
    "summary.json",
    "scorecard.csv",
    "scorecard.md",
    "validation.json",
)
_JSONL_ARTIFACTS = (
    "tasks.jsonl",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "memory_events.jsonl",
    "memory_inventory.jsonl",
    "retrieval_trace.jsonl",
    "interventions.jsonl",
    "api_usage.jsonl",
    "policy_calls.jsonl",
    "prefix_manifests.jsonl",
    "graph_diagnostics.jsonl",
)
_SCORECARD_FIELDS = (
    "policy_profile_id",
    "condition",
    "readout",
    "status",
    "n_sceu",
    "mean_behavior_score",
    "behavior_correct_rate",
    "baseline_stability_rate",
    "mean_visible_memory_count",
    "mean_live_memory_count",
    "causal_memory_use_rate",
    "beneficial_intervention_rate",
    "harmful_intervention_rate",
    "unstable_intervention_rate",
    "constraint_loss_rate",
    "current_plan_deviation_rate",
    "stale_state_action_rate",
    "local_over_global_rate",
    "aggregate_drift_rate",
)


@dataclass(frozen=True)
class ReportArtifacts:
    root: Path
    artifact_hashes: tuple[tuple[str, str], ...]
    manifest_sha256: str


@dataclass(frozen=True)
class _ScorecardObservation:
    policy_profile_id: str
    condition: str
    readout: str
    status: str
    row: SCEURunResult


def write_qualification_report(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    output_directory: Path,
    *,
    run_metadata: Mapping[str, object] | None = None,
    prefix_artifacts: Mapping[str, object] | None = None,
) -> ReportArtifacts:
    """Write the complete deterministic report and hash every non-manifest file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _flatten_rows(matrix, prefix_artifacts=prefix_artifacts)
    for name in _JSONL_ARTIFACTS:
        _atomic_write(
            output_directory / name,
            _jsonl_bytes(rows[name]),
        )

    evaluation_matrix = any(
        not hasattr(task, "writes") for task in matrix.task_results
    )
    if evaluation_matrix:
        observations = multisystem_observations_from_results(
            matrix,
            specs,
            prefix_artifacts=prefix_artifacts,
        )
        metrics = compute_multisystem_metrics(observations)
        metrics_by_cell = compute_multisystem_metrics_by_cell(observations)
        scorecard_rows = list(compute_multisystem_scorecard(observations))
    else:
        metrics = compute_qualification_metrics(matrix, specs)
        metrics_by_cell = ()
        scorecard_rows = _scorecard_rows(matrix)
    _atomic_write(
        output_directory / "metrics.json",
        _json_bytes(metrics.to_dict()),
    )
    _atomic_write(
        output_directory / "metrics_by_cell.json",
        _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "groups": (
                    list(metrics_by_cell)
                    if evaluation_matrix
                    else _metrics_by_cell(matrix, specs)
                ),
            }
        ),
    )
    _atomic_write(
        output_directory / "summary.json",
        _json_bytes(_summary(matrix, rows)),
    )
    _atomic_write(
        output_directory / "scorecard.csv",
        _scorecard_csv(scorecard_rows).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "scorecard.md",
        _scorecard_markdown(scorecard_rows).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "validation.json",
        _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "pending_external_validation",
                "run_identity": matrix.run_identity,
            }
        ),
    )

    artifact_hashes = tuple(
        (name, _sha256_file(output_directory / name))
        for name in REQUIRED_REPORT_ARTIFACTS
        if name != "run_manifest.json"
    )
    metadata = dict(run_metadata or {})
    for reserved in (
        "schema_version",
        "run_identity",
        "artifact_hashes",
    ):
        metadata.pop(reserved, None)
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        **metadata,
        "run_identity": matrix.run_identity,
        "artifact_hashes": dict(artifact_hashes),
    }
    manifest_path = output_directory / "run_manifest.json"
    _atomic_write(manifest_path, _json_bytes(manifest))
    return ReportArtifacts(
        root=output_directory,
        artifact_hashes=artifact_hashes,
        manifest_sha256=_sha256_file(manifest_path),
    )


def _flatten_rows(
    matrix: QualificationMatrixResult,
    *,
    prefix_artifacts: Mapping[str, object] | None = None,
) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in _JSONL_ARTIFACTS
    }
    seen_calls: set[str] = set()
    for task in matrix.task_results:
        task_context = _task_context(task)
        artifact = _prefix_artifact_for_task(task, prefix_artifacts or {})
        if artifact is not None:
            rows["prefix_manifests.jsonl"].append(
                {
                    **task_context,
                    "prefix_artifact_hash": artifact.artifact_hash,
                    "backend": artifact.backend,
                    "profile_id": artifact.profile_id,
                    "config_hash": artifact.config_hash,
                    "run_identity": artifact.run_identity,
                    "dataset_release": artifact.dataset_release,
                    "surface_hash": artifact.surface_hash,
                    "source_commit": artifact.source_commit,
                    }
                )
            # Schema-v2 evaluation results carry their immutable memory prefix
            # outside the task result (rather than through legacy ``writes``).
            # Export one inventory snapshot per checkpoint so the report keeps
            # the stored-state side of the retrieval chain auditable.
            if not getattr(task, "writes", ()):
                for checkpoint in artifact.checkpoints:
                    if checkpoint.inventory is not None:
                        rows["memory_inventory.jsonl"].append(
                            {
                                **task_context,
                                **_jsonable(asdict(checkpoint.inventory)),
                            }
                        )
            for checkpoint in artifact.checkpoints:
                for key, value in checkpoint.graph_diagnostics:
                    rows["graph_diagnostics.jsonl"].append(
                        {
                            **task_context,
                            "checkpoint_session": checkpoint.checkpoint_session,
                            "key": key,
                            "value": _jsonable(value),
                        }
                    )
        rows["tasks.jsonl"].append(
            {
                **task_context,
                "status": task.status,
                "result_ids": [
                    condition.result_id
                    for condition in task.condition_results
                ],
            }
        )
        rows["task_results.jsonl"].append(
            _jsonable(asdict(task))
        )
        for write in getattr(task, "writes", ()):
            write_context = {
                **task_context,
                "session_index": write.session_index,
            }
            for event in write.events:
                rows["memory_events.jsonl"].append(
                    {
                        **write_context,
                        **_jsonable(asdict(event)),
                    }
                )
            rows["memory_inventory.jsonl"].append(
                {
                    **write_context,
                    **_jsonable(asdict(write.inventory)),
                }
            )
            for usage in write.usage_events:
                _append_internal_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    write_context,
                    usage,
                )
        for trace in getattr(task, "retrieval_traces", ()):
            rows["retrieval_trace.jsonl"].append(
                {
                    **task_context,
                    **_jsonable(asdict(trace)),
                }
            )
            for usage in trace.internal_usage:
                _append_internal_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    {
                        **task_context,
                        "sceu_id": trace.sceu_id,
                        "opportunity_id": trace.opportunity_id,
                        "checkpoint_session": trace.checkpoint_session,
                    },
                    usage,
                )
            if trace.rerank_result is not None:
                _append_reranker_usage(
                    rows["api_usage.jsonl"],
                    seen_calls,
                    task_context,
                    trace,
                )
        for condition in task.condition_results:
            condition_context = {
                **task_context,
                "result_id": condition.result_id,
                "condition": condition.condition,
                "readout": condition.readout,
                "condition_status": condition.status,
            }
            for row in condition.sceu_results:
                trace_id = _evaluation_trace_id(
                    task.task_id,
                    row.sceu_id,
                    condition.readout,
                    row.retrieval_trace_id,
                )
                row_payload = {
                    **condition_context,
                    **_jsonable(asdict(row)),
                }
                if trace_id is not None:
                    row_payload["retrieval_trace_id"] = trace_id
                rows["sceu_results.jsonl"].append(row_payload)
                for evaluation in row.baseline_evaluations:
                    _append_api_usage(
                        rows["api_usage.jsonl"],
                        seen_calls,
                        condition_context,
                        row,
                        evaluation,
                        intervention_kind="baseline",
                        target_memory_id=None,
                    )
                for intervention in row.interventions:
                    rows["interventions.jsonl"].append(
                        {
                            **condition_context,
                            "sceu_id": row.sceu_id,
                            "opportunity_id": row.opportunity_id,
                            **_jsonable(asdict(intervention)),
                        }
                    )
                    for evaluation in intervention.evaluations:
                        _append_api_usage(
                            rows["api_usage.jsonl"],
                            seen_calls,
                            condition_context,
                            row,
                            evaluation,
                            intervention_kind=intervention.intervention_kind,
                            target_memory_id=intervention.target_memory_id,
                        )
            # The schema-v2 evaluator persists retrievals inside each immutable
            # SCEU result rather than a mutable runner task trace.  Normalize
            # them to the same trace artifact here.
            if not getattr(task, "retrieval_traces", ()) and condition.condition in {
                "flat_retrieval",
                "mem0",
                "amem",
                "memos",
            }:
                for row in condition.sceu_results:
                    rows["retrieval_trace.jsonl"].append(
                        {
                            **condition_context,
                            "trace_id": _evaluation_trace_id(
                                task.task_id,
                                row.sceu_id,
                                condition.readout,
                                row.retrieval_trace_id,
                            ),
                            "sceu_id": row.sceu_id,
                            "opportunity_id": row.opportunity_id,
                            "checkpoint_session": row.checkpoint_session,
                            "query": "",
                            "query_hash": "",
                            "candidate_memory_ids": list(row.candidate_memory_ids),
                            "native_retrieved_memory_ids": list(
                                row.retrieved_memory_ids
                                if condition.readout == "native"
                                else ()
                            ),
                            "common_reranked_memory_ids": list(
                                row.retrieved_memory_ids
                                if condition.readout == "common_rerank"
                                else ()
                            ),
                            "candidate_shortfall": bool(
                                getattr(row, "candidate_shortfall", False)
                            ),
                            "search_latency_seconds": 0.0,
                            "rerank_result": None,
                            "internal_usage": [],
                        }
                    )
    rows["policy_calls.jsonl"] = [
        dict(row)
        for row in rows["api_usage.jsonl"]
        if isinstance(row.get("policy_request_hash"), str)
        and bool(row["policy_request_hash"])
    ]
    return rows


def _append_api_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    condition_context: Mapping[str, object],
    sceu: SCEURunResult,
    evaluation: PolicyEvaluation,
    *,
    intervention_kind: str,
    target_memory_id: str | None,
) -> None:
    if evaluation.call_id in seen_calls:
        return
    seen_calls.add(evaluation.call_id)
    rows.append(
        {
            **condition_context,
            "sceu_id": sceu.sceu_id,
            "opportunity_id": sceu.opportunity_id,
            "intervention_kind": intervention_kind,
            "target_memory_id": target_memory_id,
            "call_id": evaluation.call_id,
            "call_kind": evaluation.call_kind,
            "provider": evaluation.response.provider,
            "model_id": evaluation.response.model_id,
            "endpoint_identity": evaluation.response.endpoint_identity,
            "provider_request_id": evaluation.response.provider_request_id,
            "request_hash": evaluation.response.request_hash,
            "response_hash": evaluation.response.response_hash,
            "policy_request_hash": evaluation.policy_request_hash,
            "input_tokens": evaluation.response.usage.input_tokens,
            "output_tokens": evaluation.response.usage.output_tokens,
            "cached_tokens": evaluation.response.usage.cached_tokens,
            "reasoning_tokens": evaluation.response.usage.reasoning_tokens,
            "usage_observed": evaluation.response.usage.observed,
            "input_count": 1,
            "latency_seconds": evaluation.response.latency_seconds,
            "retry_count": evaluation.response.retry_count,
            "format_repair_used": evaluation.response.format_repair_used,
            "error_class": None,
            "started_at_utc": evaluation.response.started_at_utc,
            "ended_at_utc": evaluation.response.ended_at_utc,
        }
    )


def _append_internal_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    context: Mapping[str, object],
    usage: object,
) -> None:
    from lhmsb.adapters.mem0_qualification import ProviderUsageEvent

    if not isinstance(usage, ProviderUsageEvent):
        raise TypeError("internal provider usage has the wrong type")
    if usage.call_id in seen_calls:
        return
    seen_calls.add(usage.call_id)
    rows.append(
        {
            **context,
            "call_id": usage.call_id,
            "call_kind": usage.component,
            "provider": usage.provider,
            "model_id": usage.model_id,
            "endpoint_identity": usage.endpoint_identity,
            "provider_request_id": None,
            "request_hash": usage.request_hash,
            "response_hash": usage.response_hash,
            "policy_request_hash": None,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "usage_observed": usage.usage_observed,
            "input_count": usage.input_count,
            "latency_seconds": usage.latency_seconds,
            "retry_count": usage.retry_count,
            "format_repair_used": False,
            "error_class": usage.error_class,
            "started_at_utc": usage.started_at_utc,
            "ended_at_utc": usage.ended_at_utc,
        }
    )


def _append_reranker_usage(
    rows: list[dict[str, object]],
    seen_calls: set[str],
    context: Mapping[str, object],
    trace: RetrievalTrace,
) -> None:
    result = trace.rerank_result
    if result is None:
        return
    call_id = f"reranker:{trace.trace_id}"
    if call_id in seen_calls:
        return
    seen_calls.add(call_id)
    rows.append(
        {
            **context,
            "sceu_id": trace.sceu_id,
            "opportunity_id": trace.opportunity_id,
            "checkpoint_session": trace.checkpoint_session,
            "call_id": call_id,
            "call_kind": "reranker",
            "provider": "local_tei",
            "model_id": result.model,
            "model_revision": result.revision,
            "endpoint_identity": "local://tei-reranker",
            "provider_request_id": None,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "policy_request_hash": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "usage_observed": False,
            "input_count": result.input_count,
            "latency_seconds": result.latency_seconds,
            "retry_count": 0,
            "format_repair_used": False,
            "error_class": None,
            "started_at_utc": None,
            "ended_at_utc": None,
        }
    )


def _task_context(task: QualificationTaskResult) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "episode_id": task.episode_id,
        "policy_profile_id": task.policy_profile_id,
        "condition": task.condition,
    }


def _evaluation_trace_id(
    task_id: str,
    sceu_id: str,
    readout: str,
    existing: str | None,
) -> str | None:
    """Return a stable report trace ID for one schema-v2 SCEU/readout.

    Common-rerank rows already carry the evaluator trace ID.  Native rows in
    early schema-v2 result files intentionally left that field empty because
    their order is directly inherited from the frozen candidate search.  The
    report still needs a distinct trace record for validation and provenance,
    so synthesize one without changing the original task result bytes.
    """
    if existing:
        return existing
    if readout == "native":
        return f"{task_id}:{sceu_id}:native"
    return None


def _prefix_artifact_for_task(
    task: object,
    artifacts: Mapping[str, object],
) -> MemoryPrefixArtifact | None:
    condition = str(getattr(task, "condition", ""))
    episode_id = str(getattr(task, "episode_id", ""))
    backend = "mem0" if condition in {"mem0_controlled", "mem0_native"} else condition
    for key in (f"{episode_id}--{backend}", backend, f"{episode_id}--{condition}", condition):
        raw = artifacts.get(key)
        if raw is None:
            continue
        if isinstance(raw, MemoryPrefixArtifact):
            return raw
        if isinstance(raw, Mapping):
            try:
                return MemoryPrefixArtifact.from_dict(raw)
            except Exception:
                return None
    return None


def _summary(
    matrix: QualificationMatrixResult,
    rows: Mapping[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    statuses: dict[str, int] = defaultdict(int)
    condition_statuses: dict[str, int] = defaultdict(int)
    for task in matrix.task_results:
        statuses[task.status] += 1
        for condition in task.condition_results:
            condition_statuses[condition.status] += 1
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_identity": matrix.run_identity,
        "n_tasks": len(matrix.task_results),
        "task_status_counts": dict(sorted(statuses.items())),
        "condition_status_counts": dict(sorted(condition_statuses.items())),
        "n_sceu_results": len(rows["sceu_results.jsonl"]),
        "n_memory_events": len(rows["memory_events.jsonl"]),
        "n_inventory_snapshots": len(rows["memory_inventory.jsonl"]),
        "n_retrieval_traces": len(rows["retrieval_trace.jsonl"]),
        "n_interventions": len(rows["interventions.jsonl"]),
        "n_api_calls": len(rows["api_usage.jsonl"]),
        "n_policy_calls": len(rows["policy_calls.jsonl"]),
        "n_memory_internal_calls": sum(
            row.get("call_kind") == "memory_internal_llm"
            for row in rows["api_usage.jsonl"]
        ),
        "n_embedding_calls": sum(
            row.get("call_kind") == "embedding"
            for row in rows["api_usage.jsonl"]
        ),
        "n_reranker_calls": sum(
            row.get("call_kind") == "reranker"
            for row in rows["api_usage.jsonl"]
        ),
        "n_prefix_manifests": len(rows["prefix_manifests.jsonl"]),
        "n_graph_diagnostics": len(rows["graph_diagnostics.jsonl"]),
    }


def _scorecard_rows(
    matrix: QualificationMatrixResult,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str],
        list[_ScorecardObservation],
    ] = defaultdict(list)
    for task in matrix.task_results:
        for condition in task.condition_results:
            for row in condition.sceu_results:
                key = (
                    task.policy_profile_id,
                    condition.condition,
                    condition.readout,
                )
                grouped[key].append(
                    _ScorecardObservation(
                        policy_profile_id=task.policy_profile_id,
                        condition=condition.condition,
                        readout=condition.readout,
                        status=condition.status,
                        row=row,
                    )
                )
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        observations = grouped[key]
        rows = [item.row for item in observations]
        interventions = [
            intervention
            for row in rows
            for intervention in row.interventions
        ]
        causal = [
            intervention.classification.label
            for intervention in interventions
            if intervention.intervention_kind == "leave_one_out"
        ]
        intervention_labels = [
            intervention.classification.label
            for intervention in interventions
        ]
        drift_flags = [
            flag
            for row in rows
            for flag in row.normalized_drift_flags
        ]
        output.append(
            {
                "policy_profile_id": key[0],
                "condition": key[1],
                "readout": key[2],
                "status": _aggregate_status(
                    item.status for item in observations
                ),
                "n_sceu": len(rows),
                "mean_behavior_score": _ratio_value(
                    sum(row.behavior.behavior_score for row in rows),
                    len(rows),
                ),
                "behavior_correct_rate": _ratio_value(
                    sum(row.behavior.is_correct for row in rows),
                    len(rows),
                ),
                "baseline_stability_rate": _ratio_value(
                    sum(row.baseline_stable for row in rows),
                    len(rows),
                ),
                "mean_visible_memory_count": _ratio_value(
                    sum(len(row.model_visible_memory_ids) for row in rows),
                    len(rows),
                ),
                "mean_live_memory_count": _ratio_value(
                    sum(
                        _live_memory_count_or_zero(row)
                        for row in rows
                        if _live_memory_count_from_row(row) is not None
                    ),
                    sum(
                        1
                        for row in rows
                        if _live_memory_count_from_row(row) is not None
                    ),
                ),
                "causal_memory_use_rate": _ratio_value(
                    sum(label in {"beneficial", "harmful"} for label in causal),
                    len(causal),
                ),
                "beneficial_intervention_rate": _ratio_value(
                    intervention_labels.count("beneficial"),
                    len(intervention_labels),
                ),
                "harmful_intervention_rate": _ratio_value(
                    intervention_labels.count("harmful"),
                    len(intervention_labels),
                ),
                "unstable_intervention_rate": _ratio_value(
                    sum(
                        label
                        in {"unstable_baseline", "intervention_unstable"}
                        for label in intervention_labels
                    ),
                    len(intervention_labels),
                ),
                "constraint_loss_rate": _flag_rate(
                    drift_flags,
                    "constraint_loss",
                    len(rows),
                ),
                "current_plan_deviation_rate": _flag_rate(
                    drift_flags,
                    "plan_deviation",
                    len(rows),
                ),
                "stale_state_action_rate": _flag_rate(
                    drift_flags,
                    "stale_state",
                    len(rows),
                ),
                "local_over_global_rate": _flag_rate(
                    drift_flags,
                    "local_over_global",
                    len(rows),
                ),
                "aggregate_drift_rate": _ratio_value(
                    sum(bool(row.normalized_drift_flags) for row in rows),
                    len(rows),
                ),
            }
        )
    return output


def _metrics_by_cell(
    matrix: QualificationMatrixResult,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                task.policy_profile_id,
                condition.condition,
                condition.readout,
            )
            for task in matrix.task_results
            for condition in task.condition_results
        }
    )
    groups: list[dict[str, object]] = []
    for policy_profile_id, condition_name, readout in keys:
        selected_tasks: list[QualificationTaskResult] = []
        for task in matrix.task_results:
            if (
                task.policy_profile_id != policy_profile_id
                or task.condition != condition_name
            ):
                continue
            selected_conditions = tuple(
                condition
                for condition in task.condition_results
                if condition.condition == condition_name
                and condition.readout == readout
            )
            if not selected_conditions:
                continue
            traces = task.retrieval_traces
            if readout != "common_rerank":
                traces = tuple(
                    replace(trace, rerank_result=None)
                    for trace in traces
                )
            selected_tasks.append(
                replace(
                    task,
                    condition_results=selected_conditions,
                    retrieval_traces=traces,
                )
            )
        selected_matrix = QualificationMatrixResult(
            run_identity=matrix.run_identity,
            task_results=tuple(selected_tasks),
        )
        groups.append(
            {
                "policy_profile_id": policy_profile_id,
                "condition": condition_name,
                "readout": readout,
                "metrics": compute_qualification_metrics(
                    selected_matrix,
                    specs,
                ).to_dict(),
            }
        )
    return groups


def _aggregate_status(statuses: Sequence[str] | Any) -> str:
    values = set(statuses)
    if not values:
        return "unknown"
    if values == {"complete"}:
        return "complete"
    if "failed" in values:
        return "failed"
    return "partial"


def _flag_rate(flags: Sequence[str], name: str, denominator: int) -> float | None:
    return _ratio_value(flags.count(name), denominator)


def _live_memory_count_from_row(row: object) -> int | None:
    value = getattr(row, "live_memory_count", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _live_memory_count_or_zero(row: object) -> int:
    value = _live_memory_count_from_row(row)
    return 0 if value is None else value


def _ratio_value(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _scorecard_csv(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_SCORECARD_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: _display_value(row.get(field))
                for field in _SCORECARD_FIELDS
            }
        )
    return stream.getvalue()


def _scorecard_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    header = "| " + " | ".join(_SCORECARD_FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in _SCORECARD_FIELDS) + " |"
    body = [
        "| "
        + " | ".join(_display_value(row.get(field)) for field in _SCORECARD_FIELDS)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _jsonl_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    ordered = sorted(
        (_jsonable(row) for row in rows),
        key=lambda row: json.dumps(row, sort_keys=True, default=str),
    )
    if not ordered:
        return b""
    return (
        "\n".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
            for row in ordered
        )
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(child) for child in value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_REPORT_ARTIFACTS",
    "ReportArtifacts",
    "write_qualification_report",
]
