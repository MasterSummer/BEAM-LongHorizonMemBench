"""TDD tests for the retrieval-quality metric (task 19, Dim 4).

Written FIRST (RED) before ``src/lhmsb/metrics/retrieval.py``.

Validates ``spec/02-metrics.md`` §3 exactly:
  - precision@k / recall@k / context_relevance formulas (worked example §3.4).
  - The endogenous-vs-oracle split is reported SEPARATELY and never blended:
    a strong memory (FakePerfect) + a weak agent (poor queries) shows HIGH
    oracle p@k but LOW endogenous p@k — the confound is disentangled.
  - Edge cases (§3.3): empty gold -> recall N/A; empty result -> p@k=0;
    fewer-than-k results -> denominator stays k; duplicates deduped by
    memory_id; no oracle probes -> oracle N/A; no retrieval calls -> both N/A.
  - Latency is NEVER recomputed here (it lives in the CostVector).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lhmsb.adapters.fakes import FakePerfectAdapter, GroundTruthFact
from lhmsb.metrics.retrieval import (
    DEFAULT_K,
    EndogenousQuery,
    OracleProbe,
    RetrievalMetrics,
    RetrievalReport,
    context_relevance,
    oracle_valid_fact_ids,
    precision_at_k,
    recall_at_k,
    retrieval_report,
    score_endogenous,
    score_oracle,
    valid_fact_ids_at,
)
from lhmsb.sim.core import WorldState
from lhmsb.types import (
    Condition,
    CostVector,
    Episode,
    EpisodeResult,
    MemoryEntry,
    SearchResult,
    WorldEvent,
)


def _entry(memory_id: str, *, fact_id: str | None = None) -> MemoryEntry:
    """A MemoryEntry whose metadata carries a fact_id (real-adapter shape)."""
    meta: dict[str, object] = {"fact_id": fact_id if fact_id is not None else memory_id}
    return MemoryEntry(memory_id=memory_id, content=memory_id, metadata=meta)


# ---------------------------------------------------------------------------
# Exact IR math on a known set (failure-detecting — QA scenario 2)
# ---------------------------------------------------------------------------


class TestExactIRMath:
    def test_known_set_p_at_3_and_recall_at_3(self) -> None:
        # QA: Relevant={A,B,C}; returned top-3={A,X,B}; p@3=2/3, recall@3=2/3.
        returned = ["A", "X", "B"]
        gold = {"A", "B", "C"}
        assert precision_at_k(returned, gold, 3) == pytest.approx(2 / 3)
        assert recall_at_k(returned, gold, 3) == pytest.approx(2 / 3)

    def test_spec_worked_example_p10_recall10(self) -> None:
        # spec §3.4: R={m1,m2,m3,m4}; S=top-10; p@10=0.2, recall@10=0.5.
        returned = ["m1", "m5", "m2", "m6", "m7", "m8", "m9", "m10", "m11", "m12"]
        gold = {"m1", "m2", "m3", "m4"}
        assert precision_at_k(returned, gold, 10) == pytest.approx(0.20)
        assert recall_at_k(returned, gold, 10) == pytest.approx(0.50)

    def test_spec_worked_example_context_relevance(self) -> None:
        # spec §3.4: if m8 was retracted, context_relevance = 9/10 = 0.90.
        returned = ["m1", "m5", "m2", "m6", "m7", "m8", "m9", "m10", "m11", "m12"]
        valid = {"m1", "m5", "m2", "m6", "m7", "m9", "m10", "m11", "m12"}  # m8 retracted
        assert context_relevance(returned, valid, 10) == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Edge cases (spec §3.3)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_gold_recall_is_na_precision_zero(self) -> None:
        assert recall_at_k(["A"], set(), 5) is None  # N/A
        assert precision_at_k(["A"], set(), 5) == 0.0  # precision 0 always

    def test_empty_result_p_and_recall_zero(self) -> None:
        assert precision_at_k([], {"A", "B"}, 10) == 0.0
        assert recall_at_k([], {"A", "B"}, 10) == 0.0

    def test_fewer_than_k_results_denominator_is_requested_k(self) -> None:
        # Two results, both relevant, k=10 -> p@10 = 2/10 (NOT 2/2=1.0).
        assert precision_at_k(["A", "B"], {"A", "B"}, 10) == pytest.approx(0.20)
        # recall still uses |gold|: both gold hit -> 2/2 = 1.0
        assert recall_at_k(["A", "B"], {"A", "B"}, 10) == pytest.approx(1.0)

    def test_duplicates_deduped_by_memory_id(self) -> None:
        # Duplicate memory_id counted once before intersection.
        returned = ["A", "A", "A", "B"]
        assert precision_at_k(returned, {"A", "B"}, 10) == pytest.approx(0.20)  # 2/10
        assert recall_at_k(returned, {"A", "B"}, 10) == pytest.approx(1.0)

    def test_duplicates_deduped_for_memory_entries(self) -> None:
        returned = [_entry("A"), _entry("A"), _entry("B")]
        assert precision_at_k(returned, {"A", "B"}, 10) == pytest.approx(0.20)

    def test_k_must_be_positive(self) -> None:
        for fn in (precision_at_k, recall_at_k):
            with pytest.raises(ValueError, match="k"):
                fn(["A"], {"A"}, 0)
        with pytest.raises(ValueError, match="k"):
            context_relevance(["A"], {"A"}, -1)

    def test_default_k_is_ten(self) -> None:
        assert DEFAULT_K == 10


# ---------------------------------------------------------------------------
# MemoryEntry / SearchResult inputs + fact-id relevance + dedup
# ---------------------------------------------------------------------------


class TestInputShapes:
    def test_searchresult_input(self) -> None:
        sr = SearchResult(results=[_entry("A"), _entry("X"), _entry("B")], total_count=3)
        assert precision_at_k(sr, {"A", "B", "C"}, 3) == pytest.approx(2 / 3)
        assert recall_at_k(sr, {"A", "B", "C"}, 3) == pytest.approx(2 / 3)

    def test_relevance_via_metadata_fact_id_not_memory_id(self) -> None:
        # Real adapter: memory_id != fact_id; gold is in fact space.
        sr = SearchResult(
            results=[_entry("mem-1", fact_id="ev-1"), _entry("mem-2", fact_id="ev-9")],
            total_count=2,
        )
        # gold ev-1 only -> 1 hit of 2 returned items; p@2 = 1/2.
        assert precision_at_k(sr, {"ev-1"}, 2) == pytest.approx(0.5)

    def test_context_relevance_uses_fact_validity(self) -> None:
        results = [_entry("mem-1", fact_id="ev-1"), _entry("mem-2", fact_id="ev-2")]
        # ev-2 retracted -> only ev-1 current -> 1/2 over k=2.
        assert context_relevance(results, {"ev-1"}, 2) == pytest.approx(0.5)

    def test_relevance_falls_back_to_memory_id_without_metadata(self) -> None:
        # Ground-truth-derived store: memory_id == fact_id, no fact_id metadata.
        results = [MemoryEntry(memory_id="A", content="a"), MemoryEntry(memory_id="B", content="b")]
        assert precision_at_k(results, {"A"}, 2) == pytest.approx(0.5)
        assert context_relevance(results, {"A"}, 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RetrievalMetrics / RetrievalReport dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_retrieval_metrics_is_frozen_with_fields(self) -> None:
        m = RetrievalMetrics(
            precision_at_k=0.2, recall_at_k=0.5, context_relevance=0.9, n_queries=3
        )
        assert m.precision_at_k == 0.2
        assert m.recall_at_k == 0.5
        assert m.context_relevance == 0.9
        assert m.n_queries == 3
        with pytest.raises(FrozenInstanceError):  # frozen dataclass rejects mutation
            m.precision_at_k = 1.0

    def test_na_metrics(self) -> None:
        na = RetrievalMetrics.na()
        assert na.precision_at_k is None
        assert na.recall_at_k is None
        assert na.context_relevance is None
        assert na.n_queries == 0

    def test_report_keeps_modes_distinct(self) -> None:
        endo = RetrievalMetrics(0.1, 0.1, 0.5, 4)
        orac = RetrievalMetrics(0.98, 0.95, 1.0, 4)
        report = RetrievalReport(endogenous=endo, oracle=orac)
        assert report.endogenous is endo
        assert report.oracle is orac
        # never blended into a single number
        assert not hasattr(report, "blended")


# ---------------------------------------------------------------------------
# score_endogenous aggregation
# ---------------------------------------------------------------------------


class TestScoreEndogenous:
    def test_means_over_queries(self) -> None:
        gold = frozenset({"A", "B"})
        q1 = EndogenousQuery(returned=["A", "B"], gold_ids=gold, valid_fact_ids=gold)
        q2 = EndogenousQuery(returned=["X"], gold_ids=gold, valid_fact_ids=gold)
        m = score_endogenous([q1, q2], k=2)
        # q1 p@2 = 2/2 = 1.0 ; q2 p@2 = 0/2 = 0.0 -> mean 0.5
        assert m.precision_at_k == pytest.approx(0.5)
        # q1 recall = 2/2 = 1.0 ; q2 recall = 0/2 = 0.0 -> mean 0.5
        assert m.recall_at_k == pytest.approx(0.5)
        assert m.n_queries == 2

    def test_recall_skips_na_queries(self) -> None:
        # One query has empty gold (recall N/A) -> excluded from recall mean.
        q1 = EndogenousQuery(returned=["A"], gold_ids=frozenset({"A"}))
        q2 = EndogenousQuery(returned=["B"], gold_ids=frozenset())  # empty gold -> recall N/A
        m = score_endogenous([q1, q2], k=1)
        assert m.recall_at_k == pytest.approx(1.0)  # only q1 counts
        # precision still averages BOTH (empty gold -> precision 0)
        assert m.precision_at_k == pytest.approx((1.0 + 0.0) / 2)
        assert m.n_queries == 2

    def test_all_empty_gold_recall_is_none(self) -> None:
        q = EndogenousQuery(returned=["A"], gold_ids=frozenset())
        m = score_endogenous([q], k=5)
        assert m.recall_at_k is None

    def test_no_queries_is_na(self) -> None:
        m = score_endogenous([], k=10)
        assert m.n_queries == 0
        assert m.precision_at_k is None
        assert m.recall_at_k is None

    def test_context_relevance_none_when_not_tracked(self) -> None:
        # valid_fact_ids omitted (None) -> context_relevance N/A, but p@k defined.
        q = EndogenousQuery(returned=["A"], gold_ids=frozenset({"A"}))
        m = score_endogenous([q], k=1)
        assert m.context_relevance is None
        assert m.precision_at_k == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_oracle (fires adapter.search directly) + WorldState/Episode helpers
# ---------------------------------------------------------------------------


class TestScoreOracle:
    def test_oracle_fires_adapter_search_against_gold(self) -> None:
        facts = [
            GroundTruthFact("A", "alpha one", query_keywords=("alpha",)),
            GroundTruthFact("B", "alpha two", query_keywords=("alpha",)),
            GroundTruthFact("C", "alpha three", query_keywords=("alpha",)),
            GroundTruthFact("D", "delta four", query_keywords=("delta",)),
        ]
        adapter = FakePerfectAdapter(facts=facts)
        adapter.initialize(user_id="oracle")
        probe = OracleProbe(
            query="alpha",
            gold_ids=frozenset({"A", "B", "C"}),
            valid_fact_ids=frozenset({"A", "B", "C", "D"}),
        )
        m = score_oracle(adapter, [probe], k=3)
        assert m.precision_at_k == pytest.approx(1.0)  # perfect memory, top-3 all relevant
        assert m.recall_at_k == pytest.approx(1.0)
        assert m.context_relevance == pytest.approx(1.0)
        assert m.n_queries == 1

    def test_no_oracle_probes_is_na(self) -> None:
        facts = [GroundTruthFact("A", "alpha", query_keywords=("alpha",))]
        adapter = FakePerfectAdapter(facts=facts)
        adapter.initialize(user_id="oracle")
        m = score_oracle(adapter, [], k=10)
        assert m.n_queries == 0
        assert m.precision_at_k is None


class TestWorldHelpers:
    def test_valid_fact_ids_at_replays_world(self) -> None:
        events = [
            WorldEvent(0, "inject", "A", {"text": "a"}),
            WorldEvent(0, "inject", "B", {"text": "b"}),
            WorldEvent(1, "retract", "B", {}),
        ]
        world = WorldState(events)
        assert valid_fact_ids_at(world, 0) == frozenset({"A", "B"})
        assert valid_fact_ids_at(world, 1) == frozenset({"A"})

    def test_oracle_valid_fact_ids_from_episode(self) -> None:
        events = [
            WorldEvent(0, "inject", "A", {"text": "a"}),
            WorldEvent(1, "inject", "B", {"text": "b"}),
            WorldEvent(2, "retract", "A", {}),
        ]
        episode = Episode(episode_id="ep", family="research", seed=1, events=events, probes=[])
        assert oracle_valid_fact_ids(episode, 1) == frozenset({"A", "B"})
        assert oracle_valid_fact_ids(episode, 2) == frozenset({"B"})


# ---------------------------------------------------------------------------
# THE confound split (happy path — QA scenario 1)
# ---------------------------------------------------------------------------


class TestEndogenousOracleSplit:
    """Strong memory (FakePerfect) + weak agent (poor query): the split shows
    HIGH oracle p@k but LOW endogenous p@k — disentangling the confound."""

    def _adapter(self) -> FakePerfectAdapter:
        facts = [
            GroundTruthFact("A", "alpha one", query_keywords=("alpha",)),
            GroundTruthFact("B", "alpha two", query_keywords=("alpha",)),
            GroundTruthFact("C", "alpha three", query_keywords=("alpha",)),
            GroundTruthFact("D", "delta four", query_keywords=("delta",)),
        ]
        adapter = FakePerfectAdapter(facts=facts)
        adapter.initialize(user_id="oracle")
        return adapter

    def test_oracle_high_endogenous_low_same_memory(self) -> None:
        adapter = self._adapter()
        gold = frozenset({"A", "B", "C"})
        valid = frozenset({"A", "B", "C", "D"})

        # ORACLE: a well-formed benchmark query independent of the agent.
        oracle_probe = OracleProbe(query="alpha", gold_ids=gold, valid_fact_ids=valid)

        # ENDOGENOUS: the WEAK agent issued a poor query ("delta") for the same
        # information need; the perfect memory faithfully returns D (wrong fact).
        weak_hits = adapter.search("delta", user_id="oracle", top_k=3)
        endo_query = EndogenousQuery(returned=weak_hits, gold_ids=gold, valid_fact_ids=valid)

        report = retrieval_report(
            endogenous=[endo_query],
            oracle_adapter=adapter,
            oracle_probes=[oracle_probe],
            k=3,
        )

        assert report.oracle.precision_at_k is not None
        assert report.endogenous.precision_at_k is not None
        # memory system retrieves perfectly when queried well
        assert report.oracle.precision_at_k == pytest.approx(1.0)
        # but the agent's poor query yields no relevant hits
        assert report.endogenous.precision_at_k == pytest.approx(0.0)
        # the split is decisive: oracle >> endogenous (NOT blended)
        assert report.oracle.precision_at_k > report.endogenous.precision_at_k

    def test_report_endogenous_only_when_no_oracle(self) -> None:
        # spec §3.3: no oracle probes defined -> oracle N/A, endogenous still scored.
        report = retrieval_report(
            endogenous=[EndogenousQuery(returned=["A"], gold_ids=frozenset({"A"}))],
            k=3,
        )
        assert report.oracle.precision_at_k is None
        assert report.oracle.n_queries == 0
        assert report.endogenous.precision_at_k == pytest.approx(1 / 3)
        assert report.endogenous.n_queries == 1

    def test_no_retrieval_calls_both_modes_na(self) -> None:
        # spec §3.3: no retrieval calls in the episode -> both modes N/A.
        result = EpisodeResult(
            episode_id="ep",
            condition=Condition("no_memory"),
            seed=0,
            probe_results=[],
            cost=CostVector(num_retrieval_calls=0),
        )
        adapter = self._adapter()
        report = retrieval_report(
            endogenous=[EndogenousQuery(returned=["A"], gold_ids=frozenset({"A"}))],
            oracle_adapter=adapter,
            oracle_probes=[OracleProbe(query="alpha", gold_ids=frozenset({"A"}))],
            episode_result=result,
            k=3,
        )
        assert report.endogenous.n_queries == 0
        assert report.endogenous.precision_at_k is None
        assert report.oracle.precision_at_k is None
