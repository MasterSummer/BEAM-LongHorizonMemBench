"""Retrieval-quality metric (Dim 4) — endogenous + oracle, never blended.

Canonical source: ``spec/02-metrics.md`` §3. Retrieval quality measures how well
the memory system returns relevant items, DECONFOUNDED from the agent's query
quality, by scoring two modes that are ALWAYS reported separately:

* **Endogenous retrieval** — score the memory ``search()`` results that the
  agent's OWN queries triggered during the episode, against the known relevant
  ``fact_id`` set at each probe step. A weak agent issuing poor queries scores
  low here even with a perfect memory — that is an *agent* problem, not a memory
  problem, which is exactly why this mode is kept distinct from the oracle.
* **Oracle retrieval** — a fixed, agent-independent set of benchmark queries with
  KNOWN relevant ids, fired DIRECTLY at ``adapter.search()``. This isolates the
  memory system's retrieval quality.

Formulas (``spec/02-metrics.md`` §3.2, per query ``q`` with gold ``R_q`` and
deduplicated top-k results ``S_{q,k}``):

  precision@k(q)     = |S_{q,k} ∩ R_q| / k          (denominator is the REQUESTED k)
  recall@k(q)        = |S_{q,k} ∩ R_q| / |R_q|       (empty gold → N/A)
  context_relevance  = |{i ∈ S_{q,k} : i.fact current}| / k

Returned items are deduplicated by ``memory_id`` before intersection; the gold /
validity match key is the item's fact identity (``metadata["fact_id"]`` when
present, else ``memory_id`` — they coincide for ground-truth-derived stores).

Retrieval LATENCY is NOT computed here: it lives in the per-episode
``CostVector`` (``retrieval_latency_ms``, task 6) and must not be double-counted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from lhmsb.adapters import MemorySystemAdapter
from lhmsb.sim.core import WorldState
from lhmsb.types import Episode, EpisodeResult, MemoryEntry, SearchResult

#: Default top-k for retrieval scoring (``spec/02-metrics.md`` §3.2). Configurable per call.
DEFAULT_K = 10

#: What a caller may hand the scorers as "returned items": a raw ``SearchResult``,
#: a sequence of ``MemoryEntry`` (full entries, enables context_relevance), or a
#: sequence of ``memory_id`` strings (e.g. the harness ``retrieved_ids`` capture).
ReturnedItems: TypeAlias = SearchResult | Sequence[MemoryEntry] | Sequence[str]

#: Module-level empty sets — used as immutable defaults (ruff B008 forbids a
#: ``frozenset()`` call in an argument/field default).
_NO_IDS: frozenset[str] = frozenset()


def _require_positive_k(k: int) -> None:
    """k is the REQUESTED top-k and the precision/context denominator; must be > 0."""
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _memory_id(item: MemoryEntry | str) -> str:
    """The dedup key (``spec/02-metrics.md`` §3.3: dedup by ``memory_id``)."""
    return item if isinstance(item, str) else item.memory_id


def _relevance_id(item: MemoryEntry | str) -> str:
    """The fact identity used to match gold / validity.

    Prefers ``metadata["fact_id"]`` (real adapters mint a backend ``memory_id``
    distinct from the fact it stores) and falls back to ``memory_id`` (the
    ground-truth-derived stores and calibration fakes set ``memory_id == fact_id``).
    """
    if isinstance(item, str):
        return item
    meta = item.metadata
    if meta is not None:
        fact_id = meta.get("fact_id")
        if isinstance(fact_id, str):
            return fact_id
    return item.memory_id


def _dedup_top_k(returned: ReturnedItems, k: int) -> list[MemoryEntry | str]:
    """Deduplicate by ``memory_id`` (first occurrence), then take the top-k.

    Order matters: dedup collapses literal duplicates first so each ``memory_id``
    is counted once, THEN the requested top-k window is applied.
    """
    source: Sequence[MemoryEntry | str]
    source = returned.results if isinstance(returned, SearchResult) else returned
    seen: set[str] = set()
    deduped: list[MemoryEntry | str] = []
    for item in source:
        memory_id = _memory_id(item)
        if memory_id not in seen:
            seen.add(memory_id)
            deduped.append(item)
    return deduped[:k]


def precision_at_k(returned: ReturnedItems, gold: Iterable[str], k: int = DEFAULT_K) -> float:
    """|S_{q,k} ∩ R_q| / k. Denominator is the REQUESTED k (penalizes too-few results).

    Empty gold → 0.0 (a non-retrieval probe is never "perfectly precise"); empty
    result → 0.0.
    """
    _require_positive_k(k)
    gold_set = set(gold)
    retrieved = {_relevance_id(item) for item in _dedup_top_k(returned, k)}
    return len(retrieved & gold_set) / k


def recall_at_k(returned: ReturnedItems, gold: Iterable[str], k: int = DEFAULT_K) -> float | None:
    """|S_{q,k} ∩ R_q| / |R_q|. Empty gold → ``None`` (N/A, a non-retrieval probe)."""
    _require_positive_k(k)
    gold_set = set(gold)
    if not gold_set:
        return None
    retrieved = {_relevance_id(item) for item in _dedup_top_k(returned, k)}
    return len(retrieved & gold_set) / len(gold_set)


def context_relevance(
    returned: ReturnedItems, valid_facts_at_step: Iterable[str], k: int = DEFAULT_K
) -> float:
    """|{i ∈ S_{q,k} : i.fact is current}| / k — fraction of returned items that are
    valid (not retracted/superseded) at the probe step."""
    _require_positive_k(k)
    valid = set(valid_facts_at_step)
    current = sum(1 for item in _dedup_top_k(returned, k) if _relevance_id(item) in valid)
    return current / k


@dataclass(frozen=True)
class RetrievalMetrics:
    """Scored retrieval for ONE mode (endogenous or oracle).

    Each field is ``None`` when N/A: ``precision_at_k`` / ``context_relevance``
    are ``None`` only when there were no queries; ``recall_at_k`` is ``None`` when
    no scored query carried a non-empty gold set. ``n_queries`` is the number of
    queries that contributed.
    """

    precision_at_k: float | None
    recall_at_k: float | None
    context_relevance: float | None
    n_queries: int

    @classmethod
    def na(cls) -> RetrievalMetrics:
        """The N/A sentinel for a mode with no queries (e.g. no oracle probes)."""
        return cls(precision_at_k=None, recall_at_k=None, context_relevance=None, n_queries=0)


@dataclass(frozen=True)
class RetrievalReport:
    """Both retrieval modes, kept DISTINCT — never collapsed into one number."""

    endogenous: RetrievalMetrics
    oracle: RetrievalMetrics


@dataclass(frozen=True)
class EndogenousQuery:
    """One agent-issued ``search()`` during the episode + the gold at that step.

    ``returned`` is what the memory system gave back for the agent's OWN query;
    ``gold_ids`` is the known relevant fact set at the probe step; ``valid_fact_ids``
    is the non-retracted fact set at the step (``None`` → context_relevance N/A).
    """

    returned: ReturnedItems
    gold_ids: frozenset[str] = _NO_IDS
    valid_fact_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class OracleProbe:
    """A fixed, agent-independent benchmark query with a KNOWN gold relevant set."""

    query: str
    gold_ids: frozenset[str] = _NO_IDS
    valid_fact_ids: frozenset[str] | None = None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def score_endogenous(
    queries: Sequence[EndogenousQuery], k: int = DEFAULT_K
) -> RetrievalMetrics:
    """Aggregate per-query precision/recall/context_relevance over the agent's own queries.

    Precision averages over ALL queries (empty gold contributes 0). Recall averages
    only over queries with a non-empty gold (the rest are N/A). context_relevance
    averages only over queries that supplied a validity set. No queries → N/A mode.
    """
    _require_positive_k(k)
    if not queries:
        return RetrievalMetrics.na()
    precisions: list[float] = []
    recalls: list[float] = []
    contexts: list[float] = []
    for query in queries:
        precisions.append(precision_at_k(query.returned, query.gold_ids, k))
        recall = recall_at_k(query.returned, query.gold_ids, k)
        if recall is not None:
            recalls.append(recall)
        if query.valid_fact_ids is not None:
            contexts.append(context_relevance(query.returned, query.valid_fact_ids, k))
    return RetrievalMetrics(
        precision_at_k=_mean(precisions),
        recall_at_k=_mean(recalls) if recalls else None,
        context_relevance=_mean(contexts) if contexts else None,
        n_queries=len(queries),
    )


def score_oracle(
    adapter: MemorySystemAdapter,
    probes: Sequence[OracleProbe],
    *,
    user_id: str = "oracle",
    session_id: str | None = None,
    k: int = DEFAULT_K,
) -> RetrievalMetrics:
    """Fire each oracle probe DIRECTLY at ``adapter.search()`` and score the results.

    Independent of the agent — isolates the memory system's own retrieval quality.
    """
    _require_positive_k(k)
    if not probes:
        return RetrievalMetrics.na()
    scored: list[EndogenousQuery] = [
        EndogenousQuery(
            returned=adapter.search(
                probe.query, user_id=user_id, session_id=session_id, top_k=k
            ),
            gold_ids=probe.gold_ids,
            valid_fact_ids=probe.valid_fact_ids,
        )
        for probe in probes
    ]
    return score_endogenous(scored, k=k)


def retrieval_report(
    *,
    endogenous: Sequence[EndogenousQuery] = (),
    oracle_adapter: MemorySystemAdapter | None = None,
    oracle_probes: Sequence[OracleProbe] = (),
    episode_result: EpisodeResult | None = None,
    k: int = DEFAULT_K,
    oracle_user_id: str = "oracle",
    oracle_session_id: str | None = None,
) -> RetrievalReport:
    """Assemble the endogenous + oracle report, keeping the two modes distinct.

    When ``episode_result`` is supplied and its ``CostVector`` recorded zero
    retrieval calls (``num_retrieval_calls == 0``), BOTH modes are N/A
    (``spec/02-metrics.md`` §3.3 — the agent never used memory search). The
    retrieval-call count is read from the cost vector; latency is never recomputed.
    """
    _require_positive_k(k)
    if episode_result is not None and episode_result.cost.num_retrieval_calls == 0:
        return RetrievalReport(endogenous=RetrievalMetrics.na(), oracle=RetrievalMetrics.na())
    endogenous_metrics = score_endogenous(endogenous, k=k)
    if oracle_adapter is None or not oracle_probes:
        oracle_metrics = RetrievalMetrics.na()
    else:
        oracle_metrics = score_oracle(
            oracle_adapter,
            oracle_probes,
            user_id=oracle_user_id,
            session_id=oracle_session_id,
            k=k,
        )
    return RetrievalReport(endogenous=endogenous_metrics, oracle=oracle_metrics)


def valid_fact_ids_at(world: WorldState, step: int) -> frozenset[str]:
    """Non-retracted fact ids valid at ``step`` (replays the fixed world schedule)."""
    return frozenset(world.valid_facts_at(step).keys())


def oracle_valid_fact_ids(episode: Episode, step: int) -> frozenset[str]:
    """``valid_fact_ids_at`` for an episode's exogenous event schedule at ``step``."""
    return valid_fact_ids_at(WorldState(episode.events), step)
