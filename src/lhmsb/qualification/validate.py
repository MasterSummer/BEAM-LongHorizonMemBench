"""Schema, hash, identity, and trace-chain validation for report artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lhmsb.qualification.report import REQUIRED_REPORT_ARTIFACTS


@dataclass(frozen=True)
class ArtifactValidationReport:
    ok: bool
    errors: tuple[str, ...]
    checked_artifacts: int
    run_identity: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "checked_artifacts": self.checked_artifacts,
            "run_identity": self.run_identity,
        }


def validate_qualification_artifacts(
    report_directory: Path,
    *,
    expected_run_identity: str | None = None,
) -> ArtifactValidationReport:
    """Validate every artifact without trusting manifest-declared hashes."""
    errors: list[str] = []
    missing = [
        name
        for name in REQUIRED_REPORT_ARTIFACTS
        if not (report_directory / name).is_file()
    ]
    errors.extend(f"missing required artifact: {name}" for name in missing)
    manifest = _read_json(
        report_directory / "run_manifest.json",
        errors,
        required=False,
    )
    run_identity = _optional_text(manifest.get("run_identity"))
    if expected_run_identity is not None and run_identity != expected_run_identity:
        errors.append(
            "run identity mismatch: "
            f"expected {expected_run_identity}, got {run_identity}"
        )
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, Mapping):
        errors.append("run manifest artifact_hashes must be an object")
        hashes = {}
    checked = 0
    for raw_name, raw_expected in sorted(hashes.items()):
        name = str(raw_name)
        path = report_directory / name
        if name == "run_manifest.json":
            errors.append("run manifest must not self-declare its own hash")
            continue
        if not path.is_file():
            errors.append(f"manifest-hashed artifact is missing: {name}")
            continue
        checked += 1
        actual = _sha256(path)
        if actual != str(raw_expected):
            errors.append(f"artifact hash mismatch: {name}")

    jsonl: dict[str, list[dict[str, object]]] = {}
    for name in (
        "tasks.jsonl",
        "task_results.jsonl",
        "sceu_results.jsonl",
        "memory_events.jsonl",
        "memory_inventory.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
        "api_usage.jsonl",
    ):
        jsonl[name] = _read_jsonl(report_directory / name, errors)
        canonical = sorted(
            jsonl[name],
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        if jsonl[name] != canonical:
            errors.append(f"JSONL rows are not deterministically sorted: {name}")

    task_ids = _unique_ids(
        jsonl["tasks.jsonl"],
        "task_id",
        "task",
        errors,
    )
    result_task_ids = {
        str(row.get("task_id", ""))
        for row in jsonl["task_results.jsonl"]
        if row.get("task_id")
    }
    unknown_result_tasks = sorted(result_task_ids - task_ids)
    if unknown_result_tasks:
        errors.append(
            f"task_results contain unknown task IDs: {unknown_result_tasks}"
        )

    traces = {
        str(row.get("trace_id")): row
        for row in jsonl["retrieval_trace.jsonl"]
        if row.get("trace_id")
    }
    sceu_keys: set[tuple[str, str, str]] = set()
    mem0_tasks_with_sceu: set[str] = set()
    for row in jsonl["sceu_results.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"SCEU result references unknown task ID: {task_id}")
        result_id = str(row.get("result_id", ""))
        sceu_id = str(row.get("sceu_id", ""))
        key = (task_id, result_id, sceu_id)
        if key in sceu_keys:
            errors.append(f"duplicate SCEU result identity: {key}")
        sceu_keys.add(key)
        condition = str(row.get("condition", ""))
        readout = str(row.get("readout", ""))
        candidates = _string_list(
            row.get("candidate_memory_ids"),
            f"{key} candidate_memory_ids",
            errors,
        )
        retrieved = _string_list(
            row.get("retrieved_memory_ids"),
            f"{key} retrieved_memory_ids",
            errors,
        )
        visible = _string_list(
            row.get("model_visible_memory_ids"),
            f"{key} model_visible_memory_ids",
            errors,
        )
        if not set(retrieved).issubset(candidates):
            errors.append(
                f"retrieved memories are not a subset of candidates for {key}"
            )
        if not set(visible).issubset(retrieved):
            errors.append(
                f"model-visible memories are not a subset of retrieved for {key}"
            )
        if visible != retrieved[: len(visible)]:
            errors.append(
                f"model-visible ordering is not a prefix of retrieved for {key}"
            )
        if readout == "native" and retrieved != candidates[: len(retrieved)]:
            errors.append(
                f"native retrieved ordering is not a candidate prefix for {key}"
            )
        if condition in {"workspace_only", "oracle_current_state"} and any(
            (candidates, retrieved, visible)
        ):
            errors.append(
                f"non-memory condition exposes memory IDs for {key}"
            )
        trace_id = row.get("retrieval_trace_id")
        if condition.startswith("mem0_"):
            mem0_tasks_with_sceu.add(task_id)
            if not isinstance(trace_id, str) or trace_id not in traces:
                errors.append(
                    f"Mem0 SCEU lacks a known retrieval trace for {key}"
                )
            else:
                _validate_trace_match(row, traces[trace_id], key, errors)
        elif trace_id is not None:
            errors.append(
                f"non-memory SCEU unexpectedly references retrieval trace for {key}"
            )

    inventory_tasks = {
        str(row.get("task_id", ""))
        for row in jsonl["memory_inventory.jsonl"]
    }
    trace_tasks = {
        str(row.get("task_id", ""))
        for row in jsonl["retrieval_trace.jsonl"]
    }
    for task_id in sorted(mem0_tasks_with_sceu):
        if task_id not in inventory_tasks:
            errors.append(f"Mem0 task lacks inventory snapshots: {task_id}")
        if task_id not in trace_tasks:
            errors.append(f"Mem0 task lacks retrieval traces: {task_id}")

    for row in jsonl["memory_events.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"memory event references unknown task ID: {task_id}")
    for row in jsonl["memory_inventory.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"inventory references unknown task ID: {task_id}")
    for row in jsonl["retrieval_trace.jsonl"]:
        task_id = str(row.get("task_id", ""))
        if task_id not in task_ids:
            errors.append(f"retrieval trace references unknown task ID: {task_id}")

    for row in jsonl["interventions.jsonl"]:
        key = (
            str(row.get("task_id", "")),
            str(row.get("result_id", "")),
            str(row.get("sceu_id", "")),
        )
        if key not in sceu_keys:
            errors.append(f"intervention references unknown SCEU result: {key}")
        evaluations = row.get("evaluations")
        if not isinstance(evaluations, Sequence) or isinstance(
            evaluations,
            (str, bytes),
        ):
            errors.append(f"intervention evaluations must be an array for {key}")
        elif len(evaluations) != 2:
            errors.append(
                f"intervention must contain exactly two repeated evaluations for {key}"
            )
        classification = row.get("classification")
        if not isinstance(classification, Mapping) or not classification.get(
            "label"
        ):
            errors.append(f"intervention classification is incomplete for {key}")

    _unique_ids(
        jsonl["api_usage.jsonl"],
        "call_id",
        "API call",
        errors,
        allow_empty=True,
    )
    _read_json(report_directory / "metrics.json", errors, required=False)
    _read_json(report_directory / "summary.json", errors, required=False)
    return ArtifactValidationReport(
        ok=not errors,
        errors=tuple(errors),
        checked_artifacts=checked,
        run_identity=run_identity,
    )


def _validate_trace_match(
    sceu: Mapping[str, object],
    trace: Mapping[str, object],
    key: tuple[str, str, str],
    errors: list[str],
) -> None:
    candidates = _string_list(
        sceu.get("candidate_memory_ids"),
        "candidate_memory_ids",
        errors,
    )
    trace_candidates = _string_list(
        trace.get("candidate_memory_ids"),
        "trace candidate_memory_ids",
        errors,
    )
    if candidates != trace_candidates:
        errors.append(f"SCEU candidate IDs do not match retrieval trace for {key}")
    retrieved = _string_list(
        sceu.get("retrieved_memory_ids"),
        "retrieved_memory_ids",
        errors,
    )
    readout = str(sceu.get("readout", ""))
    trace_field = (
        "common_reranked_memory_ids"
        if readout == "common_rerank"
        else "native_retrieved_memory_ids"
    )
    trace_retrieved = _string_list(
        trace.get(trace_field),
        f"trace {trace_field}",
        errors,
    )
    if retrieved != trace_retrieved[: len(retrieved)]:
        errors.append(f"SCEU retrieved IDs do not match retrieval trace for {key}")


def _unique_ids(
    rows: Sequence[Mapping[str, object]],
    field: str,
    label: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> set[str]:
    output: set[str] = set()
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            if not allow_empty:
                errors.append(f"{label} row lacks {field}")
            continue
        if value in output:
            errors.append(f"duplicate {label} {field}: {value}")
        output.add(value)
    return output


def _string_list(
    value: object,
    label: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"{label} must be an array")
        return []
    output = [str(item) for item in value]
    if len(output) != len(set(output)):
        errors.append(f"{label} contains duplicate memory IDs")
    return output


def _read_json(
    path: Path,
    errors: list[str],
    *,
    required: bool,
) -> dict[str, object]:
    if not path.is_file():
        if required:
            errors.append(f"missing JSON artifact: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON artifact {path.name}: {exc}")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"JSON artifact must be an object: {path.name}")
        return {}
    return {str(key): child for key, child in value.items()}


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read JSONL artifact {path.name}: {exc}")
        return []
    output: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL {path.name}:{number}: {exc}")
            continue
        if not isinstance(value, Mapping):
            errors.append(f"JSONL row must be an object: {path.name}:{number}")
            continue
        output.append({str(key): child for key, child in value.items()})
    return output


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ArtifactValidationReport",
    "validate_qualification_artifacts",
]
