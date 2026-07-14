"""TDD tests for the counterfactual experiment runner (task 21).

All offline: a deterministic echo agent-model stub, the built-in calibration
fakes / no-memory control, and a tiny custom crashing stub adapter. No real LLM,
no external backend.

Covered:
  * a 2-episode x 3-condition x 2-seed mini-matrix -> a 12-row results table with
    the right (episode_id, condition, seed) keys and graded metrics;
  * the counterfactual invariant: identical ``world_event_hash`` per
    (episode_id, seed) across conditions, and a deliberately diverged dataset
    raises :class:`CounterfactualError` BEFORE any scoring;
  * an injected backend crash -> ``status="failed"`` with the cost-charged-before-
    failure RETAINED and the failed run INCLUDED in the table (never dropped);
  * tidy persistence (jsonl + parquet, track in the filename) and NaN handling;
  * round-tripping a frozen ``episodes.jsonl`` back into ``Episode`` objects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pytest

from lhmsb.adapters import MemorySystemAdapter
from lhmsb.cost import CostMeter
from lhmsb.hashing import episode_hash, world_event_hash
from lhmsb.judge import Judge, Rubric, StubJudge
from lhmsb.runner import (
    CounterfactualError,
    ResultsTable,
    RunRow,
    build_adapter,
    load_frozen_dataset,
    merge_costs,
    run_matrix,
    write_results,
)
from lhmsb.sim.core import EpisodeBuilder, FamilyContent, ProbeSpec
from lhmsb.types import (
    Condition,
    CostVector,
    Episode,
    EpisodeResult,
    ProbeResult,
    RunConfig,
    SearchResult,
    WorldEvent,
)

ProbeKind = Literal["factual", "synthesis", "behavioral"]

CONDITIONS = ("no_memory", "fake_perfect", "fake_bad")


# --------------------------------------------------------------------------- #
# Deterministic, offline test doubles
# --------------------------------------------------------------------------- #
def echo_agent_model(prompt: str) -> str:
    """Echo the prompt's FACTS block as the answer (deterministic, offline).

    The harness builds ``FACTS:\\n- <fact>\\n...\\nQUESTION: <q>``; echoing the
    facts lets the family checker grade recall: a condition that retrieved the
    current fact answers correctly, one that retrieved nothing answers 'UNKNOWN'.
    """
    facts = [line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")]
    return " ; ".join(facts) if facts else "UNKNOWN"


def _research_content(
    fact_alpha: str,
    fact_beta: str,
    fact_gamma: str,
    *,
    entity_a: str,
) -> FamilyContent:
    """A tiny research episode: 3 injects across 2 sessions, 1 retraction, 2 probes.

    ``fact_gamma`` is retracted before the probes (giving the adversary fake a
    plausible-but-wrong stale fact to surface). The factual + synthesis probes are
    cross-session (the relevant facts came from an earlier session), so the
    no-memory control cannot answer them from the cleared working context.
    """
    return FamilyContent(
        family="research",
        events=[
            WorldEvent(0, "inject", "ev-1", {"session": 0, "text": fact_alpha, "entity": entity_a}),
            WorldEvent(0, "inject", "ev-3", {"session": 0, "text": fact_gamma, "entity": "Gamma"}),
            WorldEvent(2, "inject", "ev-2", {"session": 1, "text": fact_beta, "entity": "Beta"}),
            WorldEvent(
                2, "retract", "ev-3", {"session": 1, "text": "Finding ev-3 is no longer valid"}
            ),
        ],
        probe_specs=[
            ProbeSpec(
                step=3,
                probe_id="p-fact",
                kind="factual",
                query=f"What does {entity_a} affect?",
                derivation="value",
                target_fact_ids=["ev-1"],
                value_key="text",
                cross_session=True,
            ),
            ProbeSpec(
                step=3,
                probe_id="p-syn",
                kind="synthesis",
                query=f"Summarize the current {entity_a} and Beta findings.",
                derivation="valid_values",
                target_fact_ids=["ev-1", "ev-2"],
                value_key="text",
                cross_session=True,
            ),
        ],
    )


# Two distinct episode contents (different events -> different world_event_hash).
_CONTENT_A = _research_content(
    "Alpha elevates the Q-index",
    "Beta suppresses Theta levels",
    "Gamma stabilizes Kappa variance",
    entity_a="Alpha",
)
_CONTENT_B = _research_content(
    "Delta amplifies the Sigma response",
    "Beta predicts baseline drift",
    "Zeta attenuates Kappa variance",
    entity_a="Delta",
)


def _build(content: FamilyContent, seed: int) -> Episode:
    return EpisodeBuilder().build(content, seed=seed)


def _dataset_2x2() -> list[Episode]:
    """2 contents x 2 seeds = 4 episodes (the 'episode' x 'seed' axes)."""
    return [
        _build(_CONTENT_A, 1),
        _build(_CONTENT_A, 2),
        _build(_CONTENT_B, 1),
        _build(_CONTENT_B, 2),
    ]


def _run_config(track: str = "native") -> RunConfig:
    return RunConfig(
        agent_model="stub-agent/echo",
        judge_model="stub-judge/deterministic",
        seeds=[1, 2],
        n_episodes=2,
        context_budget=0,
        track=track,
    )


def _judge() -> Judge:
    return Judge(StubJudge())


def _rubric() -> Rubric:
    return Rubric(
        version="test-1.0", criteria="Score the answer vs the gold.", source_path="<test>"
    )


def _zero_clock() -> float:
    """Constant clock -> zero, deterministic latency (no wall-clock noise)."""
    return 0.0


class _CrashingAdapter(MemorySystemAdapter):
    """A storing stub that raises on the Nth ``add_memory`` (mid-episode crash)."""

    def __init__(self, *, crash_on_add: int = 2) -> None:
        self._adds = 0
        self._crash_on = crash_on_add
        self._store: dict[str, str] = {}

    def initialize(self, *, user_id: str, session_id: str | None = None, **config: object) -> None:
        return None

    def reset(self, *, user_id: str) -> None:
        self._adds = 0
        self._store.clear()

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        self._adds += 1
        if self._adds >= self._crash_on:
            raise RuntimeError("injected backend crash")
        memory_id = f"crash-{self._adds}"
        self._store[memory_id] = content
        return memory_id

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 10,
        **filters: object,
    ) -> SearchResult:
        return SearchResult(results=[], total_count=0)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        return None

    def delete_memory(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)


# --------------------------------------------------------------------------- #
# 1. Matrix shape + keys
# --------------------------------------------------------------------------- #
def test_matrix_has_one_row_per_cell() -> None:
    """4 episodes x 3 conditions = 12 rows, with unique (episode_id, condition, seed)."""
    table = run_matrix(
        _dataset_2x2(),
        _run_config(),
        agent_model=echo_agent_model,
        conditions=CONDITIONS,
        judge=_judge(),
        rubric=_rubric(),
        clock=_zero_clock,
    )
    assert isinstance(table, ResultsTable)
    assert len(table) == 12
    keys = table.keys()
    assert len(set(keys)) == 12  # every cell is unique
    assert {condition for _, condition, _ in keys} == set(CONDITIONS)
    assert {row.track for row in table.rows} == {"native"}
    assert {row.family for row in table.rows} == {"research"}


def test_matrix_grades_real_scores_and_includes_all_conditions() -> None:
    """Grading produces real metrics; the oracle fake beats the no-memory control."""
    table = run_matrix(
        _dataset_2x2(),
        _run_config(),
        agent_model=echo_agent_model,
        conditions=CONDITIONS,
        judge=_judge(),
        rubric=_rubric(),
        clock=_zero_clock,
    )
    by_condition: dict[str, list[RunRow]] = {c: [] for c in CONDITIONS}
    for row in table.rows:
        by_condition[row.condition].append(row)
        assert row.status == "completed"
        assert 0.0 <= row.task_score <= 1.0

    # Oracle utilization is > 0 (not always 1.0): the fuzzy judge-scored synthesis
    # probe is also cross-session, so it can miss the is_correct threshold.
    perfect_util = [r.utilization_rate for r in by_condition["fake_perfect"]]
    nomem_util = [r.utilization_rate for r in by_condition["no_memory"]]
    assert all(u is not None and u > 0.0 for u in perfect_util)
    assert all(u == 0.0 for u in nomem_util)
    # The oracle's mean task score strictly exceeds the no-memory control's.
    perfect_mean = sum(r.task_score for r in by_condition["fake_perfect"]) / 4
    nomem_mean = sum(r.task_score for r in by_condition["no_memory"]) / 4
    assert perfect_mean > nomem_mean

    # Drift sensitivity: the adversary cites the retracted fact (drift), the oracle
    # and the no-memory control do not.
    assert all(r.drift_index == 1.0 for r in by_condition["fake_bad"])
    assert all(r.drift_index == 0.0 for r in by_condition["fake_perfect"])
    assert all(r.drift_index == 0.0 for r in by_condition["no_memory"])


def test_per_cell_metrics_are_populated() -> None:
    """Each row carries task / drift / retrieval metrics + the full cost vector."""
    table = run_matrix(
        _dataset_2x2(),
        _run_config(),
        agent_model=echo_agent_model,
        conditions=CONDITIONS,
        judge=_judge(),
        rubric=_rubric(),
        clock=_zero_clock,
    )
    for row in table.rows:
        # A factual probe is drift-category A, so drift_index is a real number in [0,1].
        assert not row.drift_is_na
        assert 0.0 <= row.drift_index <= 1.0
        assert row.n_probes == 2
        # Retrieval is scored (>=1 retrieval call happened), never None here.
        assert row.retrieval_endogenous_precision is not None
        assert row.retrieval_oracle_precision is not None
        # The agent loop charged tokens -> cost vector is populated. Searches: one
        # per probe (2) plus one for the retract event's lookup-then-delete = 3.
        assert row.cost.agent_input_tokens > 0
        assert row.cost.num_retrieval_calls == 3


# --------------------------------------------------------------------------- #
# 2. Counterfactual invariant
# --------------------------------------------------------------------------- #
def test_world_event_hash_identical_per_key_across_conditions() -> None:
    """All conditions for one (episode_id, seed) share a single world_event_hash."""
    table = run_matrix(
        _dataset_2x2(),
        _run_config(),
        agent_model=echo_agent_model,
        conditions=CONDITIONS,
        clock=_zero_clock,
    )
    by_key: dict[tuple[str, int], set[str]] = {}
    for row in table.rows:
        by_key.setdefault((row.episode_id, row.seed), set()).add(row.world_event_hash)
    assert len(by_key) == 4  # 4 distinct (episode_id, seed) keys
    for hashes in by_key.values():
        assert len(hashes) == 1  # identical across all 3 conditions


def test_diverged_world_raises_counterfactual_error_before_scoring() -> None:
    """A dataset where one (episode_id, seed) has two world hashes fails fast."""
    base = _build(_CONTENT_A, 1)
    tampered_events = [
        *base.events,
        WorldEvent(5, "inject", "ev-9", {"session": 2, "text": "leak"}),
    ]
    # Same episode_id + seed, DIFFERENT events -> different world_event_hash.
    diverged = Episode(
        episode_id=base.episode_id,
        family=base.family,
        seed=base.seed,
        events=tampered_events,
        probes=base.probes,
    )
    assert world_event_hash(base.events, base.probes) != world_event_hash(
        diverged.events, diverged.probes
    )
    with pytest.raises(CounterfactualError):
        run_matrix(
            [base, diverged],
            _run_config(),
            agent_model=echo_agent_model,
            conditions=("no_memory",),
            clock=_zero_clock,
        )


# --------------------------------------------------------------------------- #
# 3. Failure isolation: injected backend crash
# --------------------------------------------------------------------------- #
def _crash_content() -> FamilyContent:
    """Inject@0 -> probe@1 (agent answers, charges cost) -> inject@2 (crashes)."""
    return FamilyContent(
        family="research",
        events=[
            WorldEvent(0, "inject", "ev-1", {"session": 0, "text": "Alpha elevates the Q-index"}),
            WorldEvent(2, "inject", "ev-2", {"session": 0, "text": "Beta suppresses Theta levels"}),
        ],
        probe_specs=[
            ProbeSpec(
                step=1,
                probe_id="p-fact",
                kind="factual",
                query="What does Alpha affect?",
                derivation="value",
                target_fact_ids=["ev-1"],
                value_key="text",
            ),
        ],
    )


def test_injected_crash_is_isolated_costed_and_included() -> None:
    """A crashing backend -> status=failed, partial cost retained, run kept; peers OK."""
    episode = _build(_crash_content(), 7)

    def crash_factory(
        condition_name: str,
        run_config: RunConfig,
        cost_meter: CostMeter,
        ep: Episode,
    ) -> MemorySystemAdapter:
        if condition_name == "crash":
            return _CrashingAdapter(crash_on_add=2)
        return build_adapter(condition_name, run_config, cost_meter, episode=ep)

    table = run_matrix(
        [episode],
        _run_config(),
        agent_model=echo_agent_model,
        conditions=("no_memory", "crash"),
        adapter_factory=crash_factory,
        clock=_zero_clock,
        max_attempts=2,
    )

    assert len(table) == 2  # the crashed run is NOT dropped
    rows = {row.condition: row for row in table.rows}
    crash_row = rows["crash"]
    assert crash_row.status == "failed"
    # Cost charged before the crash (the agent answered probe@1) is retained.
    assert crash_row.cost.agent_input_tokens > 0
    assert crash_row.attempts == 2  # bounded retries were exercised
    # The other condition still completes -> one backend's crash did not abort the matrix.
    assert rows["no_memory"].status == "completed"


# --------------------------------------------------------------------------- #
# 4. Cost merge (no double-counting of harness-bracketed latency/retrieval)
# --------------------------------------------------------------------------- #
def test_merge_costs_overlays_adapter_fields_without_double_counting() -> None:
    """Harness owns agent/latency/retrieval; adapter owns memory-internal fields."""
    harness = CostVector(
        agent_input_tokens=10,
        agent_output_tokens=4,
        retrieval_latency_ms=5.0,
        num_retrieval_calls=3,
    )
    adapter = CostVector(
        mem_internal_in_tokens=7,
        mem_internal_out_tokens=2,
        embedding_tokens=9,
        embedding_calls=1,
        storage_bytes=128,
        retrieval_latency_ms=99.0,  # the adapter ALSO timed it; must NOT be added
        num_retrieval_calls=3,  # likewise must NOT be added
    )
    merged = merge_costs(harness, adapter)
    assert merged.agent_input_tokens == 10
    assert merged.mem_internal_in_tokens == 7
    assert merged.embedding_tokens == 9
    assert merged.storage_bytes == 128
    assert merged.retrieval_latency_ms == 5.0  # harness only, not 5+99
    assert merged.num_retrieval_calls == 3  # harness only, not 3+3


# --------------------------------------------------------------------------- #
# 5. Persistence (jsonl + parquet, track in filename, NaN handling)
# --------------------------------------------------------------------------- #
def test_results_persisted_as_jsonl_and_parquet_keyed_by_track(tmp_path: Path) -> None:
    table = run_matrix(
        _dataset_2x2(),
        _run_config(track="controlled"),
        agent_model=echo_agent_model,
        conditions=CONDITIONS,
        judge=_judge(),
        rubric=_rubric(),
        clock=_zero_clock,
    )
    written = write_results(table, tmp_path, basename="results")
    jsonl_path = written["jsonl"]
    assert jsonl_path.name == "results.controlled.jsonl"  # track baked into the name
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 12
    first = json.loads(lines[0])
    # Every required field is present and flat.
    for key in ("episode_id", "family", "seed", "condition", "track", "status"):
        assert key in first
    for cost_field in ("agent_input_tokens", "embedding_tokens", "num_retrieval_calls"):
        assert cost_field in first
    assert first["track"] == "controlled"
    # parquet written when the optional backend is available (pyarrow IS installed here).
    assert "parquet" in written
    assert written["parquet"].name == "results.controlled.parquet"
    assert written["parquet"].is_file()


def test_run_row_serializes_nan_drift_as_none() -> None:
    """drift_index = NaN (N/A) -> None on serialization (valid JSON / nullable parquet)."""
    row = RunRow(
        episode_id="ep",
        family="research",
        seed=1,
        condition="no_memory",
        track="native",
        status="completed",
        attempts=1,
        n_probes=0,
        world_event_hash="abc",
        episode_hash="def",
        task_score=0.0,
        utilization_rate=None,
        improvement_over_time=None,
        judge_contribution=0.0,
        drift_index=math.nan,
        drift_is_na=True,
        stale_fact_violations=0,
        constraint_violations=0,
        behavioral_flips=0,
        judge_fallback_share=0.0,
    )
    record = row.to_record()
    assert record["drift_index"] is None
    assert record["utilization_rate"] is None
    # round-trips through json (no NaN literal, which is invalid JSON).
    reparsed = json.loads(json.dumps(record))
    assert reparsed["drift_index"] is None


# --------------------------------------------------------------------------- #
# 6. Frozen-dataset loader round-trip
# --------------------------------------------------------------------------- #
def _write_frozen_episodes(frozen_dir: Path, episodes: list[Episode]) -> None:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for episode in episodes:
        record = {
            "episode_id": episode.episode_id,
            "family": episode.family,
            "seed": episode.seed,
            "events": [
                {"step": e.step, "kind": e.kind, "fact_id": e.fact_id, "payload": e.payload}
                for e in episode.events
            ],
            "probes": [
                {
                    "step": p.step,
                    "probe_id": p.probe_id,
                    "kind": p.kind,
                    "query": p.query,
                    "gold": p.gold,
                    "cross_session": p.cross_session,
                }
                for p in episode.probes
            ],
            "render": episode.render or {},
            "world_event_hash": world_event_hash(episode.events, episode.probes),
            "episode_hash": episode_hash(episode),
        }
        lines.append(json.dumps(record, sort_keys=True))
    (frozen_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_frozen_dataset_reconstructs_episodes(tmp_path: Path) -> None:
    """A frozen episodes.jsonl loads back into Episodes with matching world hashes."""
    original = [_build(_CONTENT_A, 1), _build(_CONTENT_B, 2)]
    _write_frozen_episodes(tmp_path / "research", original)
    loaded = load_frozen_dataset(tmp_path / "research")
    assert len(loaded) == 2
    by_id = {ep.episode_id: ep for ep in loaded}
    for source in original:
        restored = by_id[source.episode_id]
        assert restored.seed == source.seed
        assert restored.family == source.family
        assert world_event_hash(restored.events, restored.probes) == world_event_hash(
            source.events, source.probes
        )
    # The reconstructed episodes drive a matrix exactly like freshly-built ones.
    table = run_matrix(
        loaded,
        _run_config(),
        agent_model=echo_agent_model,
        conditions=("no_memory", "fake_perfect"),
        judge=_judge(),
        rubric=_rubric(),
        clock=_zero_clock,
    )
    assert len(table) == 4


# --------------------------------------------------------------------------- #
# 7. Graded probe metadata (drift + retrieval contract) sanity
# --------------------------------------------------------------------------- #
def test_graded_probe_metadata_carries_drift_and_answer() -> None:
    """grade_episode populates drift_category/flags/answer the metrics consume."""
    from lhmsb.runner.grading import build_checker, grade_episode

    episode = _build(_CONTENT_A, 1)
    raw = EpisodeResult(
        episode_id=episode.episode_id,
        condition=Condition(name="fake_perfect"),
        seed=1,
        probe_results=[
            ProbeResult(
                probe_id="p-fact",
                score=0.0,
                is_correct=False,
                metadata={
                    "answer": "Alpha elevates the Q-index",
                    "retrieved_ids": ["ev-1"],
                    "kind": "factual",
                    "cross_session": True,
                },
            )
        ],
    )
    graded = grade_episode(
        raw, episode, checker=build_checker(episode), judge=_judge(), rubric=_rubric()
    )
    probe_result = graded.probe_results[0]
    assert probe_result.is_correct is True
    assert probe_result.score == 1.0
    meta = probe_result.metadata or {}
    assert meta["drift_category"] == "A"
    assert meta["answer"] == "Alpha elevates the Q-index"
    assert isinstance(meta["drift_flags"], list)
