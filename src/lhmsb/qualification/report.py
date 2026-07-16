"""Deterministic, hash-addressed qualification report artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.qualification.metrics import compute_qualification_metrics
from lhmsb.qualification.runner import (
    PolicyEvaluation,
    QualificationMatrixResult,
    QualificationTaskResult,
    SCEURunResult,
)

REPORT_SCHEMA_VERSION = 1
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
    "metrics.json",
    "summary.json",
    "scorecard.csv",
    "scorecard.md",
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
) -> ReportArtifacts:
    """Write the complete deterministic report and hash every non-manifest file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _flatten_rows(matrix)
    for name in _JSONL_ARTIFACTS:
        _atomic_write(
            output_directory / name,
            _jsonl_bytes(rows[name]),
        )

    metrics = compute_qualification_metrics(matrix, specs)
    _atomic_write(
        output_directory / "metrics.json",
        _json_bytes(metrics.to_dict()),
    )
    _atomic_write(
        output_directory / "summary.json",
        _json_bytes(_summary(matrix, rows)),
    )
    scorecard_rows = _scorecard_rows(matrix)
    _atomic_write(
        output_directory / "scorecard.csv",
        _scorecard_csv(scorecard_rows).encode("utf-8"),
    )
    _atomic_write(
        output_directory / "scorecard.md",
        _scorecard_markdown(scorecard_rows).encode("utf-8"),
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
) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in _JSONL_ARTIFACTS
    }
    seen_calls: set[str] = set()
    for task in matrix.task_results:
        task_context = _task_context(task)
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
        for write in task.writes:
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
        for trace in task.retrieval_traces:
            rows["retrieval_trace.jsonl"].append(
                {
                    **task_context,
                    **_jsonable(asdict(trace)),
                }
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
                rows["sceu_results.jsonl"].append(
                    {
                        **condition_context,
                        **_jsonable(asdict(row)),
                    }
                )
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
            "latency_seconds": evaluation.response.latency_seconds,
            "retry_count": evaluation.response.retry_count,
            "format_repair_used": evaluation.response.format_repair_used,
            "started_at_utc": evaluation.response.started_at_utc,
            "ended_at_utc": evaluation.response.ended_at_utc,
        }
    )


def _task_context(task: QualificationTaskResult) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "episode_id": task.episode_id,
        "policy_profile_id": task.policy_profile_id,
        "condition": task.condition,
    }


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
        "n_policy_calls": len(rows["api_usage.jsonl"]),
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
