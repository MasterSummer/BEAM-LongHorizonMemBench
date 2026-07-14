"""Tidy results record + persistence for the counterfactual runner (task 21).

A :class:`RunRow` is one fully-graded ``(episode_id, condition, seed)`` cell of the
counterfactual matrix: task / drift / retrieval metrics + the full
:class:`~lhmsb.types.CostVector` + the reproducibility hashes + the run status.
Failed / timed-out runs are kept (never dropped); their partial cost is retained.

A :class:`ResultsTable` is the rows for ONE track (``native`` or ``controlled``).
Tracks are NEVER mixed in a single output file (``spec/03-protocol.md`` §3): the
track is part of every persisted filename.

Persistence (``spec/04-datasets.md`` style — checksummable, canonical bytes):
  * ``<basename>.<track>.jsonl`` — always written (stdlib only).
  * ``<basename>.<track>.parquet`` — written when ``pyarrow`` / ``pandas`` is
    importable (lazy import); otherwise skipped gracefully (jsonl is the source
    of truth either way).

``drift_index`` may be ``float('nan')`` (N/A — no aligned drift probes). NaN is
converted to ``None`` on serialization so the jsonl is valid JSON and the parquet
column is a normal nullable float (the plan's "handle NaN when serializing" note).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lhmsb.types import CostVector

# The 12 CostVector field names, flattened verbatim into every row (the full
# vector is always retained — it is collapsed to a scalar only by task 22).
COST_FIELDS: tuple[str, ...] = (
    "agent_input_tokens",
    "agent_output_tokens",
    "mem_internal_in_tokens",
    "mem_internal_out_tokens",
    "embedding_tokens",
    "embedding_calls",
    "storage_bytes",
    "retrieval_latency_ms",
    "write_latency_ms",
    "update_latency_ms",
    "reflection_tokens",
    "num_retrieval_calls",
)

_EMPTY_FLAGS: tuple[str, ...] = ()


def _nan_to_none(value: float | None) -> float | None:
    """Map ``float('nan')`` (drift N/A) to ``None`` for valid JSON / nullable parquet."""
    if value is None:
        return None
    return None if math.isnan(value) else value


@dataclass(frozen=True)
class RunRow:
    """One graded counterfactual cell, keyed by (episode_id, condition, seed).

    Every numeric metric that can be N/A is ``float | None`` (never NaN in the
    serialized output); ``drift_index`` is the one field that may carry NaN
    internally and is normalized on serialization.
    """

    # ---- keys ----
    episode_id: str
    family: str
    seed: int
    condition: str
    track: str
    # ---- run outcome ----
    status: str
    attempts: int
    n_probes: int
    world_event_hash: str
    episode_hash: str
    # ---- Dim-2 task performance / utilization ----
    task_score: float
    utilization_rate: float | None
    improvement_over_time: float | None
    judge_contribution: float
    # ---- Dim-3 goal drift ----
    drift_index: float
    drift_is_na: bool
    stale_fact_violations: int
    constraint_violations: int
    behavioral_flips: int
    judge_fallback_share: float
    drift_flags: tuple[str, ...] = _EMPTY_FLAGS
    offending_probe_ids: tuple[str, ...] = _EMPTY_FLAGS
    # ---- Dim-4 retrieval quality (endogenous + oracle, never blended) ----
    retrieval_endogenous_precision: float | None = None
    retrieval_endogenous_recall: float | None = None
    retrieval_endogenous_context: float | None = None
    retrieval_oracle_precision: float | None = None
    retrieval_oracle_recall: float | None = None
    retrieval_oracle_context: float | None = None
    # ---- Explicit memory-count / efficiency metrics ----
    stored_memory_count: int = 0
    unique_stored_memory_count: int = 0
    retrieved_memory_count: int = 0
    unique_retrieved_memory_count: int = 0
    storage_precision: float | None = None
    storage_recall: float | None = None
    storage_f1: float | None = None
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    retrieval_f1: float | None = None
    retrieval_false_positive_rate: float | None = None
    retrieval_timeliness: float | None = None
    # ---- Dim-7 full-lifecycle cost vector (12 fields) ----
    cost: CostVector = field(default_factory=CostVector)

    def to_record(self) -> dict[str, object]:
        """Flat, serialization-friendly mapping (one row of the tidy table).

        ``drift_index`` NaN -> ``None``; list fields are kept as lists for jsonl
        and flattened to ``;``-joined strings for parquet by :func:`_to_parquet_cell`.
        The 12 cost fields are flattened inline (no nested object).
        """
        record: dict[str, object] = {
            "episode_id": self.episode_id,
            "family": self.family,
            "seed": self.seed,
            "condition": self.condition,
            "track": self.track,
            "status": self.status,
            "attempts": self.attempts,
            "n_probes": self.n_probes,
            "world_event_hash": self.world_event_hash,
            "episode_hash": self.episode_hash,
            "task_score": self.task_score,
            "utilization_rate": self.utilization_rate,
            "improvement_over_time": self.improvement_over_time,
            "judge_contribution": self.judge_contribution,
            "drift_index": _nan_to_none(self.drift_index),
            "drift_is_na": self.drift_is_na,
            "stale_fact_violations": self.stale_fact_violations,
            "constraint_violations": self.constraint_violations,
            "behavioral_flips": self.behavioral_flips,
            "judge_fallback_share": self.judge_fallback_share,
            "drift_flags": list(self.drift_flags),
            "offending_probe_ids": list(self.offending_probe_ids),
            "retrieval_endogenous_precision": self.retrieval_endogenous_precision,
            "retrieval_endogenous_recall": self.retrieval_endogenous_recall,
            "retrieval_endogenous_context": self.retrieval_endogenous_context,
            "retrieval_oracle_precision": self.retrieval_oracle_precision,
            "retrieval_oracle_recall": self.retrieval_oracle_recall,
            "retrieval_oracle_context": self.retrieval_oracle_context,
            "stored_memory_count": self.stored_memory_count,
            "unique_stored_memory_count": self.unique_stored_memory_count,
            "retrieved_memory_count": self.retrieved_memory_count,
            "unique_retrieved_memory_count": self.unique_retrieved_memory_count,
            "storage_precision": self.storage_precision,
            "storage_recall": self.storage_recall,
            "storage_f1": self.storage_f1,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_f1": self.retrieval_f1,
            "retrieval_false_positive_rate": self.retrieval_false_positive_rate,
            "retrieval_timeliness": self.retrieval_timeliness,
        }
        for field_name, value in asdict(self.cost).items():
            record[field_name] = value
        return record


def _to_parquet_cell(value: object) -> object:
    """Flatten list-valued cells to ``;``-joined strings for a columnar store."""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


@dataclass(frozen=True)
class ResultsTable:
    """The graded rows for ONE track; tracks are never mixed in one table/file."""

    track: str
    rows: list[RunRow] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def to_records(self) -> list[dict[str, object]]:
        """Every row as a flat record (jsonl/parquet-ready, NaN already handled)."""
        return [row.to_record() for row in self.rows]

    def column(self, name: str) -> list[object]:
        """Pluck one column across rows (handy for assertions / quick analysis)."""
        return [record[name] for record in self.to_records()]

    def keys(self) -> list[tuple[str, str, int]]:
        """The (episode_id, condition, seed) key of every row, in row order."""
        return [(row.episode_id, row.condition, row.seed) for row in self.rows]


def _canonical_jsonl(records: Sequence[Mapping[str, object]]) -> str:
    """One canonical JSON object per line (stable bytes -> stable checksums)."""
    lines = [json.dumps(dict(record), sort_keys=True, ensure_ascii=True) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def write_jsonl(table: ResultsTable, path: Path) -> Path:
    """Write the table as canonical jsonl (always available; stdlib only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_jsonl(table.to_records()), encoding="utf-8")
    return path


def write_parquet(table: ResultsTable, path: Path) -> Path | None:
    """Write the table as parquet via a lazily-imported ``pandas`` + ``pyarrow``.

    Returns the written path, or ``None`` when neither backend is importable
    (the run is still fully persisted as jsonl — parquet is a convenience view).
    """
    try:
        pandas = importlib.import_module("pandas")
    except ImportError:
        return None
    flat = [
        {key: _to_parquet_cell(value) for key, value in record.items()}
        for record in table.to_records()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame(flat)
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError):  # pragma: no cover - missing pyarrow engine
        return None
    return path


def write_results(
    table: ResultsTable, out_dir: Path, *, basename: str = "results"
) -> dict[str, Path]:
    """Persist the table to ``out_dir`` as ``<basename>.<track>.{jsonl,parquet}``.

    The track is baked into every filename so native and controlled results can
    never land in the same file. Returns the paths actually written (``parquet``
    omitted when the optional backend is unavailable).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    jsonl_path = out_dir / f"{basename}.{table.track}.jsonl"
    written["jsonl"] = write_jsonl(table, jsonl_path)
    parquet_path = write_parquet(table, out_dir / f"{basename}.{table.track}.parquet")
    if parquet_path is not None:
        written["parquet"] = parquet_path
    return written
