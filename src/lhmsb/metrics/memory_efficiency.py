"""Count-based storage and retrieval efficiency metrics.

These metrics are deliberately separate from the old token-cost instrumentation.
They answer the benchmark question directly: did the system keep the facts that
were marked as worth keeping, and did it return useful memories without flooding
the agent with unrelated ones?
"""

from __future__ import annotations

from dataclasses import dataclass

from lhmsb.sim.core import WorldState
from lhmsb.types import Episode, EpisodeResult

_STORE_POLICIES = frozenset({"must_store", "should_store", "store"})
_DROP_POLICIES = frozenset({"must_not_store", "should_not_store", "drop"})


@dataclass(frozen=True)
class MemoryEfficiencyReport:
    """One episode's count and efficiency measurements."""

    stored_memory_count: int
    unique_stored_memory_count: int
    retrieved_memory_count: int
    unique_retrieved_memory_count: int
    storage_precision: float | None
    storage_recall: float | None
    storage_f1: float | None
    retrieval_precision: float | None
    retrieval_recall: float | None
    retrieval_f1: float | None
    retrieval_false_positive_rate: float | None
    retrieval_timeliness: float | None


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _as_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _wide_id(value: object) -> str:
    text = str(value).strip()
    text = text.removeprefix("paper:").removeprefix("arXiv:")
    if "/abs/" in text:
        text = text.rsplit("/abs/", 1)[1]
    return text.split("v", 1)[0]


def _storage_labels(episode: Episode) -> tuple[set[str], set[str]]:
    required: set[str] = set()
    forbidden: set[str] = set()
    for event in episode.events:
        policy = event.payload.get("memory_policy")
        if not isinstance(policy, str):
            continue
        if policy in _STORE_POLICIES:
            required.add(event.fact_id)
        elif policy in _DROP_POLICIES:
            forbidden.add(event.fact_id)
    return required, forbidden


def _retrieved_fact_ids(
    episode: Episode, execution: dict[str, object]
) -> dict[str, set[str]]:
    """Map backend ids in the trace to stable fact/paper ids when possible."""
    event_by_fact = {event.fact_id: event for event in episode.events}

    def event_ids(fact_id: str) -> set[str]:
        event = event_by_fact.get(fact_id)
        if event is None:
            return {fact_id}
        paper_ids: set[str] = set()
        for key in ("arxiv_ids", "arxiv_id"):
            paper_ids.update(_wide_id(item) for item in _ids(event.payload.get(key)))
        return paper_ids or {fact_id}

    by_memory_id: dict[str, set[str]] = {
        fact_id: event_ids(fact_id) for fact_id in event_by_fact
    }
    for entry in _as_list(execution.get("storage_trace")):
        fact_id = entry.get("fact_id")
        if not isinstance(fact_id, str):
            continue
        for memory_id in _ids(entry.get("written_ids")):
            by_memory_id[memory_id] = event_ids(fact_id)

    return by_memory_id


def _probe_gold(episode: Episode, probe_step: int, probe_kind: str, gold: object) -> set[str]:
    if probe_kind == "wide_set":
        return {_wide_id(item) for item in (gold if isinstance(gold, list) else [])}
    return set(WorldState(episode.events).valid_facts_at(probe_step))


def measure_memory_efficiency(
    episode: Episode, result: EpisodeResult
) -> MemoryEfficiencyReport:
    """Measure storage/retrieval efficiency from an execution trace.

    Storage precision/recall are only defined when the episode explicitly labels
    events with ``memory_policy``. Retrieval precision/recall are defined when a
    probe has a stable fact set (Wide Research uses its gold paper set; synthetic
    families use the current valid fact ids).
    """
    execution = result.execution
    no_memory = result.condition.name in {"no_memory", "no_mem"}
    written = [] if no_memory else _ids(execution.get("written_memory_ids"))
    retrieved = _ids(execution.get("retrieved_memory_ids"))
    stored_by_fact = {
        str(entry.get("fact_id")): (
            False if no_memory else bool(_ids(entry.get("written_ids")))
        )
        for entry in _as_list(execution.get("storage_trace"))
        if isinstance(entry.get("fact_id"), str)
    }
    required, forbidden = _storage_labels(episode)
    stored_fact_ids = {fact_id for fact_id, did_store in stored_by_fact.items() if did_store}
    storage_precision: float | None = None
    storage_recall: float | None = None
    storage_f1: float | None = None
    if required or forbidden:
        true_stores = stored_fact_ids & required
        false_stores = stored_fact_ids & forbidden
        precision_denominator = len(true_stores) + len(false_stores)
        storage_precision = (
            len(true_stores) / precision_denominator if precision_denominator else 1.0
        )
        storage_recall = len(true_stores) / len(required) if required else 1.0
        storage_f1 = _f1(storage_precision, storage_recall)

    id_map = _retrieved_fact_ids(episode, execution)
    retrieval_trace = _as_list(execution.get("retrieval_trace"))
    precisions: list[float] = []
    recalls: list[float] = []
    timely_use: list[float] = []
    for trace in retrieval_trace:
        probe_id = trace.get("probe_id")
        probe = next((item for item in episode.probes if item.probe_id == probe_id), None)
        if probe is None:
            continue
        returned: set[str] = set()
        for memory_id in _ids(trace.get("retrieved_ids")):
            if memory_id.startswith("paper:"):
                returned.add(_wide_id(memory_id))
                continue
            mapped = id_map.get(memory_id)
            if mapped:
                returned.update(mapped)
            else:
                returned.add(memory_id)
        gold = _probe_gold(episode, probe.step, probe.kind, probe.gold)
        if not gold:
            continue
        hits = returned & gold
        precisions.append(len(hits) / len(returned) if returned else 0.0)
        recalls.append(len(hits) / len(gold))
        probe_result = next(
            (item for item in result.probe_results if item.probe_id == probe.probe_id),
            None,
        )
        metadata = probe_result.metadata if probe_result is not None else None
        raw_used = (metadata or {}).get("predicted_arxiv_ids")
        if not isinstance(raw_used, list):
            raw_used = (metadata or {}).get("facts_used")
        used = {_wide_id(item) for item in raw_used} if isinstance(raw_used, list) else set()
        timely_use.append(len(hits & used) / len(hits) if hits else 0.0)

    retrieval_precision: float | None = sum(precisions) / len(precisions) if precisions else None
    retrieval_recall: float | None = sum(recalls) / len(recalls) if recalls else None
    retrieval_f1 = (
        _f1(retrieval_precision, retrieval_recall)
        if retrieval_precision is not None and retrieval_recall is not None
        else None
    )
    retrieval_false_positive_rate = (
        1.0 - retrieval_precision if retrieval_precision is not None else None
    )
    retrieval_timeliness = sum(timely_use) / len(timely_use) if timely_use else None
    return MemoryEfficiencyReport(
        stored_memory_count=len(written),
        unique_stored_memory_count=len(set(written)),
        retrieved_memory_count=len(retrieved),
        unique_retrieved_memory_count=len(set(retrieved)),
        storage_precision=storage_precision,
        storage_recall=storage_recall,
        storage_f1=storage_f1,
        retrieval_precision=retrieval_precision,
        retrieval_recall=retrieval_recall,
        retrieval_f1=retrieval_f1,
        retrieval_false_positive_rate=retrieval_false_positive_rate,
        retrieval_timeliness=retrieval_timeliness,
    )


__all__ = ["MemoryEfficiencyReport", "measure_memory_efficiency"]
